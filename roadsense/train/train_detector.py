"""Fine-tune the class-agnostic sign detector on GTSDB.

    python -m roadsense.train.train_detector --out runs/detector
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torchvision.transforms import v2
from torchvision import tv_tensors
from torchvision.transforms.functional import pil_to_tensor

from ..data.gtsdb import load_gtsdb, load_image
from ..eval.detection_metrics import detection_report
from ..models.detector import build_detector, set_eval_size
from .common import Logger, cosine_warmup, device, save_json, seed_everything

AUG = v2.Compose([
    v2.RandomHorizontalFlip(0.5),  # class-agnostic boxes, so mirroring is label-safe here
    v2.RandomPhotometricDistort(p=0.8),
    v2.RandomApply([v2.GaussianBlur(5, (0.3, 1.5))], p=0.15),
])


def load_split(split):
    recs = load_gtsdb(split)
    return [(pil_to_tensor(load_image(r["path"])), r) for r in recs]


@torch.no_grad()
def run_detector(model, images, bs=4, size=800):
    model.eval()
    set_eval_size(model, size)
    dev = next(model.parameters()).device
    preds = []
    for i in range(0, len(images), bs):
        batch = [im.to(dev).float() / 255 for im in images[i:i + bs]]
        with torch.autocast("cuda", enabled=dev.type == "cuda"):
            out = model(batch)
        preds += [(o["boxes"].float().cpu().numpy(), o["scores"].float().cpu().numpy(), np.zeros(len(o["boxes"]), int)) for o in out]
    return preds


def evaluate(model, data, size=800):
    preds = run_detector(model, [im for im, _ in data], size=size)
    gts = [(r["boxes"], np.zeros(len(r["boxes"]), int)) for _, r in data]
    return detection_report(preds, gts), preds


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=2)  # 4 overflows 6 GB VRAM at 896px and spills to host memory
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    seed_everything(a.seed)
    dev = device()
    train, val = load_split("train"), load_split("val")
    print("train", len(train), "val", len(val))

    model = build_detector().to(dev)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=a.lr, momentum=0.9, weight_decay=1e-4)
    steps = a.epochs * ((len(train) + a.bs - 1) // a.bs)
    sched = cosine_warmup(opt, steps, 150)
    scaler = torch.amp.GradScaler(enabled=dev.type == "cuda")
    log = Logger(a.out)
    best = -1.0
    rng = np.random.default_rng(a.seed)

    for ep in range(a.epochs):
        model.train()
        tot, nb = 0.0, 0
        order = rng.permutation(len(train))
        for i in range(0, len(order), a.bs):
            ims, tgts = [], []
            for j in order[i:i + a.bs]:
                im, r = train[j]
                boxes = tv_tensors.BoundingBoxes(torch.from_numpy(r["boxes"]), format="XYXY", canvas_size=im.shape[1:])
                im2, boxes = AUG(tv_tensors.Image(im), boxes)
                ims.append(im2.to(dev).float() / 255)
                tgts.append({"boxes": boxes.as_subclass(torch.Tensor).to(dev),
                             "labels": torch.ones(len(boxes), dtype=torch.int64, device=dev)})
            with torch.autocast("cuda", enabled=dev.type == "cuda"):
                losses = model(ims, tgts)
                loss = sum(losses.values())
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 10.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item()
            nb += 1
        row = {"epoch": ep + 1, "loss": tot / nb, "lr": sched.get_last_lr()[0]}
        if (ep + 1) % 3 == 0 or ep + 1 == a.epochs:
            rep, _ = evaluate(model, val)
            row |= {"val_AP50": rep["AP50"], "val_AP50:95": rep["AP50:95"], "val_recall": rep["recall@0.5score"]}
            if rep["AP50:95"] >= best:
                best = rep["AP50:95"]
                torch.save({"model": model.state_dict(), "config": vars(a) | {"out": str(a.out)}}, a.out / "best.pt")
        log.log(**row)

    model.load_state_dict(torch.load(a.out / "best.pt", map_location=dev)["model"])
    test = load_split("test")
    rep = {"val": evaluate(model, val)[0], "test": evaluate(model, test)[0],
           "train_minutes": round(log.rows[-1]["time_s"] / 60, 2), "config": vars(a) | {"out": str(a.out)}}
    save_json(rep, a.out / "metrics.json")
    print("RESULT", rep)


if __name__ == "__main__":
    main()
