"""Train a road-surface condition model on RSCD.

    python -m roadsense.train.train_surface --arch effnet_b0 --out runs/surface/effnet
    python -m roadsense.train.train_surface --arch cnn --out runs/surface/cnn
    python -m roadsense.train.train_surface --arch effnet_b0 --head flat --out runs/surface/effnet_flat
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

from ..data.augment import augment_surface, normalize, to_float
from ..labels import FRICTION, IGNORE, MATERIAL, RSCD_CLASSES, UNEVENNESS, decompose
from ..models.surfacenet import build_surface_model, joint_probs
from ..data.prepare import day_cache_name
from ..paths import CACHE
from .common import Logger, batches, cosine_warmup, device, make_ema, param_groups, save_json, seed_everything

HEADS = {"friction": FRICTION, "material": MATERIAL, "unevenness": UNEVENNESS}
FACTORS = np.array([decompose(c) for c in RSCD_CLASSES])  # (27, 3)
INPUT_HW = (128, 192)


def load_split(split, protocol="day", hw=INPUT_HW):
    """protocol 'day': capture-day-disjoint re-split (honest); 'official': RSCD's own frame-level split."""
    d = np.load(CACHE / (day_cache_name(split, hw) if protocol == "day" else f"rscd_{split}.npz"))
    return torch.from_numpy(d["x"]), torch.from_numpy(d["y"]), d["files"]


def head_targets(y: torch.Tensor) -> dict[str, torch.Tensor]:
    f = torch.from_numpy(FACTORS).to(y.device)[y]
    return {"friction": f[:, 0], "material": f[:, 1], "unevenness": f[:, 2], "joint": y}


def class_weights(t: torch.Tensor, k: int) -> torch.Tensor:
    t = t[t != IGNORE]
    cnt = torch.bincount(t, minlength=k).float().clamp(min=1)
    w = cnt.pow(-0.5)
    return w / (w * cnt).sum() * cnt.sum()  # mean weight over samples = 1


@torch.no_grad()
def predict(model, x_u8, bs=256):
    model.eval()
    dev = next(model.parameters()).device
    outs = {}
    for idx in batches(len(x_u8), bs, False):
        x = normalize(to_float(x_u8[idx].to(dev, non_blocking=True))).contiguous(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            o = model(x)
        o = {k: v.float() for k, v in o.items()}
        o["joint_p"] = joint_probs(o)
        for k, v in o.items():
            outs.setdefault(k, []).append(v)
    return {k: torch.cat(v) for k, v in outs.items()}


def score(out, y) -> dict:
    y = y.to(out["joint_p"].device)
    t = head_targets(y)
    pj = out["joint_p"].argmax(1)
    m = {"joint_acc": (pj == y).float().mean().item(),
         "joint_macro_f1": f1_score(y.cpu(), pj.cpu(), average="macro")}
    # Per-attribute scores are derived from the 27-way prediction so flat and multi-head
    # models are compared on identical terms.
    pf = torch.from_numpy(FACTORS).to(y.device)[pj]
    for i, h in enumerate(HEADS):
        keep = t[h] != IGNORE
        m[f"{h}_acc"] = (pf[keep, i] == t[h][keep]).float().mean().item()
        m[f"{h}_macro_f1"] = f1_score(t[h][keep].cpu(), pf[keep, i].cpu(), average="macro")
    return m


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--arch", choices=["cnn", "effnet_b0", "convnext_tiny"], default="effnet_b0")
    ap.add_argument("--head", choices=["multi", "flat"], default="multi")
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--aug", choices=["full", "basic", "none"], default="full")
    ap.add_argument("--hw", default="128x192", help="input resolution HxW (needs the matching cache)")
    ap.add_argument("--protocol", choices=["day", "official"], default="day")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    pretrained_lr = {"effnet_b0": 1e-3, "convnext_tiny": 2e-4}
    lr = a.lr or (pretrained_lr.get(a.arch, 3e-3) if not a.no_pretrained else 3e-3)
    hw = tuple(int(v) for v in a.hw.split("x"))

    seed_everything(a.seed)
    dev = device()
    xtr, ytr, _ = load_split("train", a.protocol, hw)
    xva, yva, _ = load_split("val", a.protocol, hw)
    xtr = xtr.pin_memory() if dev.type == "cuda" else xtr
    print("train", len(ytr), "val", len(yva))

    model = build_surface_model(a.arch, a.head, pretrained=not a.no_pretrained).to(dev).to(memory_format=torch.channels_last)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params/1e6:.2f}M")
    ema = make_ema(model, 0.998)
    opt = torch.optim.AdamW(param_groups(model, a.wd), lr=lr)
    steps = a.epochs * ((len(ytr) + a.bs - 1) // a.bs)
    sched = cosine_warmup(opt, steps, steps // 20)

    tt = head_targets(ytr)
    sizes = {"friction": len(FRICTION), "material": len(MATERIAL), "unevenness": len(UNEVENNESS), "joint": 27}
    weights = {h: class_weights(tt[h], k).to(dev) for h, k in sizes.items()}
    gen = torch.Generator().manual_seed(a.seed)
    log = Logger(a.out)
    best = -1.0

    for ep in range(a.epochs):
        model.train()
        tot, n = 0.0, 0
        for idx in batches(len(ytr), a.bs, True, gen):
            x = to_float(xtr[idx].to(dev, non_blocking=True))
            y = ytr[idx].to(dev)
            x = normalize(augment_surface(x, hw, a.aug)).contiguous(memory_format=torch.channels_last)
            tg = head_targets(y)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                out = model(x)
            loss = sum(F.cross_entropy(v.float(), tg[h], weight=weights[h], ignore_index=IGNORE, label_smoothing=0.05)
                       for h, v in out.items())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            ema.update_parameters(model)
            tot += loss.item() * len(y)
            n += len(y)
        m = score(predict(ema.module, xva), yva)
        log.log(epoch=ep + 1, loss=tot / n, **{f"val_{k}": v for k, v in m.items()})
        if m["joint_macro_f1"] >= best:
            best = m["joint_macro_f1"]
            torch.save({"model": ema.module.state_dict(), "config": vars(a) | {"out": str(a.out), "lr": lr, "hw": list(hw)}}, a.out / "best.pt")

    ck = torch.load(a.out / "best.pt", map_location=dev)
    model.load_state_dict(ck["model"])
    xte, yte, _ = load_split("test", a.protocol, hw)
    test = score(predict(model, xte), yte)
    metrics = {"val_best_joint_macro_f1": best, **{f"test_{k}": v for k, v in test.items()},
               "n_test": len(yte), "params": n_params,
               "train_minutes": round(log.rows[-1]["time_s"] / 60, 2), "config": ck["config"]}
    save_json(metrics, a.out / "metrics.json")
    print("TEST", {k: v for k, v in metrics.items() if k != "config"})


if __name__ == "__main__":
    main()
