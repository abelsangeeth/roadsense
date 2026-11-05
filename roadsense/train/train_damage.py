"""Fine-tune the dashcam road-damage detector on RDD2022 (our block split).

    python -m roadsense.data.rdd                       # once: cache + split
    python -m roadsense.train.train_damage --out runs/damage
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import tv_tensors
from torchvision.io import ImageReadMode, decode_jpeg, read_file
from torchvision.transforms import v2

from ..data.rdd import COUNTRIES, DAMAGE_CLASSES, load_rdd
from ..eval.detection_metrics import detection_report, match
from ..models.detector import build_damage_detector, set_eval_size
from .common import Logger, cosine_warmup, device, save_json, seed_everything

# Mirroring keeps a longitudinal crack longitudinal and a transverse one transverse.
AUG = v2.Compose([
    v2.RandomHorizontalFlip(0.5),
    v2.RandomPhotometricDistort(p=0.8),
    v2.RandomApply([v2.GaussianBlur(5, (0.3, 1.5))], p=0.15),
])


class RDD(Dataset):
    def __init__(self, recs, train):
        self.recs, self.train = recs, train

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, i):
        r = self.recs[i]
        im = decode_jpeg(read_file(str(r["path"])), mode=ImageReadMode.RGB)
        boxes = tv_tensors.BoundingBoxes(torch.from_numpy(r["boxes"]), format="XYXY", canvas_size=im.shape[1:])
        if self.train:
            im, boxes = AUG(tv_tensors.Image(im), boxes)
        return im.as_subclass(torch.Tensor), {"boxes": boxes.as_subclass(torch.Tensor),
                                              "labels": torch.from_numpy(r["labels"]) + 1}  # 0 = background


def collate(b):
    return tuple(zip(*b))


def loader(recs, train, bs, workers=4):
    return DataLoader(RDD(recs, train), bs, shuffle=train, num_workers=workers, collate_fn=collate,
                      persistent_workers=workers > 0, drop_last=train)


@torch.no_grad()
def predict(model, recs, size=640, bs=8):
    model.eval()
    set_eval_size(model, size)
    dev = next(model.parameters()).device
    preds = []
    for ims, _ in loader(recs, False, bs):
        with torch.autocast("cuda", enabled=dev.type == "cuda"):
            out = model([im.to(dev).float() / 255 for im in ims])
        preds += [(o["boxes"].float().cpu().numpy(), o["scores"].float().cpu().numpy(), o["labels"].cpu().numpy() - 1)
                  for o in out]
    return preds


def best_f1(preds, gts):
    """CRDDC metric: F1 at IoU 0.5 with class-aware matching, at the best global score threshold."""
    s, tp, n = match(preds, gts, 0.5, classwise=True)
    order = np.argsort(-s)
    s, tp = s[order], tp[order]
    ctp = np.cumsum(tp)
    k = np.arange(1, len(tp) + 1)
    f1 = 2 * ctp / (k + n)
    i = int(np.argmax(f1)) if len(f1) else 0
    return {"F1": float(f1[i]) if len(f1) else 0.0, "threshold": float(s[i]) if len(s) else 1.0,
            "precision": float(ctp[i] / k[i]) if len(f1) else 0.0, "recall": float(ctp[i] / max(n, 1)) if len(f1) else 0.0}


def report(preds, recs, threshold=None):
    gts = [(r["boxes"], r["labels"]) for r in recs]
    out = {"mAP50": None, "per_class": {}}
    aps = []
    for c, name in enumerate(DAMAGE_CLASSES):
        p = [(b[l == c], s[l == c], l[l == c]) for b, s, l in preds]
        g = [(b[l == c], l[l == c]) for b, l in gts]
        rep = detection_report(p, g)
        out["per_class"][name] = {"AP50": rep["AP50"], "AP50:95": rep["AP50:95"], "n_gt": rep["n_gt"]}
        aps.append(rep["AP50"])
    out["mAP50"] = float(np.nanmean(aps))
    out["mAP50:95"] = float(np.nanmean([v["AP50:95"] for v in out["per_class"].values()]))
    out["best_f1"] = best_f1(preds, gts)
    if threshold is not None:  # fixed operating point chosen on val
        s, tp, n = match(preds, gts, 0.5, classwise=True)
        keep = s >= threshold
        p, r = tp[keep].sum() / max(keep.sum(), 1), tp[keep].sum() / max(n, 1)
        out["at_val_threshold"] = {"threshold": threshold, "precision": float(p), "recall": float(r),
                                   "F1": float(2 * p * r / max(p + r, 1e-9))}
    out["n_images"] = len(recs)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="use only N training images (smoke test)")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args(argv)
    seed_everything(a.seed)
    dev = device()
    train, val = load_rdd("train"), load_rdd("val")
    if a.limit:
        train = [train[i] for i in np.random.default_rng(0).permutation(len(train))[:a.limit]]
        val = val[: max(50, a.limit // 8)]
    print("train", len(train), "val", len(val), flush=True)

    model = build_damage_detector().to(dev)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=a.lr, momentum=0.9, weight_decay=1e-4)
    steps_per_ep = len(train) // a.bs
    sched = cosine_warmup(opt, a.epochs * steps_per_ep, 500)
    scaler = torch.amp.GradScaler(enabled=dev.type == "cuda")
    log = Logger(a.out)
    best, start = -1.0, 0
    if a.resume and (a.out / "last.pt").exists():
        ck = torch.load(a.out / "last.pt", map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"]); best, start, log.rows = ck["best"], ck["epoch"], ck["rows"]
        print("resumed at epoch", start, flush=True)

    dl = loader(train, True, a.bs)
    for ep in range(start, a.epochs):
        model.train()
        tot, nb = 0.0, 0
        for ims, tgts in dl:
            ims = [im.to(dev, non_blocking=True).float() / 255 for im in ims]
            tgts = [{k: v.to(dev) for k, v in t.items()} for t in tgts]
            with torch.autocast("cuda", enabled=dev.type == "cuda"):
                loss = sum(model(ims, tgts).values())
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 10.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item()
            nb += 1
            if nb % 500 == 0:
                print(f"  ep {ep + 1} step {nb}/{steps_per_ep} loss {tot / nb:.4f}", flush=True)
        rep = report(predict(model, val), val)
        row = {"epoch": ep + 1, "loss": tot / nb, "lr": sched.get_last_lr()[0],
               "val_mAP50": rep["mAP50"], "val_F1": rep["best_f1"]["F1"]}
        if rep["mAP50"] >= best:
            best = rep["mAP50"]
            torch.save({"model": model.state_dict(), "config": vars(a) | {"out": str(a.out)}, "classes": DAMAGE_CLASSES},
                       a.out / "best.pt")
        log.log(**row)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "best": best, "epoch": ep + 1, "rows": log.rows}, a.out / "last.pt")

    model.load_state_dict(torch.load(a.out / "best.pt", map_location=dev, weights_only=False)["model"])
    rep_val = report(predict(model, val), val)
    thr = rep_val["best_f1"]["threshold"]
    test = load_rdd("test")
    preds = predict(model, test)
    rep_test = report(preds, test, thr)
    by_country = {}
    for c in COUNTRIES:
        idx = [i for i, r in enumerate(test) if r["country"] == c]
        if idx:
            rc = report([preds[i] for i in idx], [test[i] for i in idx], thr)
            by_country[c] = {"mAP50": rc["mAP50"], "F1_at_val_threshold": rc["at_val_threshold"]["F1"], "n_images": len(idx)}
    out = {"val": rep_val, "test": rep_test, "test_by_country": by_country, "threshold": thr,
           "train_minutes": round(log.rows[-1]["time_s"] / 60, 2), "config": vars(a) | {"out": str(a.out)}}
    save_json(out, a.out / "metrics.json")
    print("RESULT", {"test_mAP50": rep_test["mAP50"], "test_F1": rep_test["at_val_threshold"]["F1"], "thr": thr}, flush=True)


if __name__ == "__main__":
    main()
