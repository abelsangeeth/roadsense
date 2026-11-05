"""Train RoadSense-SignNet on GTSRB.

    python -m roadsense.train.train_signs --out runs/signs/full
    python -m roadsense.train.train_signs --no-stn --out runs/signs/ablate_nostn

Validation holds out whole *tracks* (each GTSRB track is ~30 frames of one physical
sign); a random image split would put near-duplicate frames on both sides and report
an optimistic validation accuracy. The official test set is evaluated exactly once, on
the checkpoint chosen by validation accuracy.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

from ..data.augment import augment_signs, normalize, to_float
from ..models.signnet import SignNet
from ..paths import CACHE
from .common import Logger, batches, cosine_warmup, device, make_ema, param_groups, save_json, seed_everything


def track_split(tracks: np.ndarray, labels: np.ndarray, frac: float, seed: int):
    """Hold out `frac` of the tracks of every class (stratified by class, grouped by track)."""
    rng = np.random.default_rng(seed)
    val = np.zeros(len(tracks), bool)
    for c in np.unique(labels):
        ts = np.unique(tracks[labels == c])
        k = max(1, round(len(ts) * frac))
        held = set(rng.choice(ts, k, replace=False))
        val |= np.isin(tracks, list(held)) & (labels == c)
    return ~val, val


def load_gtsrb(dev, val_frac=0.1, seed=0):
    d = np.load(CACHE / "gtsrb48.npz")
    tr, va = track_split(d["track_train"], d["y_train"], val_frac, seed=1234)  # fixed split across seeds
    T = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(dev)
    return {
        "train": (T(d["x_train"][tr]), T(d["y_train"][tr])),
        "val": (T(d["x_train"][va]), T(d["y_train"][va])),
        "test": (T(d["x_test"]), T(d["y_test"])),
    }


@torch.no_grad()
def predict(model, x_u8, bs=1024, tta=False):
    model.eval()
    out = []
    for idx in batches(len(x_u8), bs, False):
        x = normalize(to_float(x_u8[idx])).contiguous(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=x.is_cuda):
            logits = model(x).float()
        out.append(logits)
    return torch.cat(out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--aug", choices=["full", "basic", "none"], default="full")
    ap.add_argument("--no-stn", action="store_true")
    ap.add_argument("--no-se", action="store_true")
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    seed_everything(a.seed)
    dev = device()
    data = load_gtsrb(dev)
    xtr, ytr = data["train"]
    print({k: len(v[1]) for k, v in data.items()})

    model = SignNet(width=a.width, use_stn=not a.no_stn, use_se=not a.no_se).to(dev).to(memory_format=torch.channels_last)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params/1e6:.2f}M")
    ema = make_ema(model, 0.998)
    opt = torch.optim.AdamW(param_groups(model, a.wd), lr=a.lr, betas=(0.9, 0.99))
    steps = a.epochs * ((len(ytr) + a.bs - 1) // a.bs)
    sched = cosine_warmup(opt, steps, warmup_steps=steps // 20)
    gen = torch.Generator().manual_seed(a.seed)
    log = Logger(a.out)
    best = -1.0

    for ep in range(a.epochs):
        model.train()
        tot, correct, n = 0.0, 0, 0
        for idx in batches(len(ytr), a.bs, True, gen):
            idx = idx.to(dev)
            x, y = augment_signs(to_float(xtr[idx]), ytr[idx], a.aug)
            x = normalize(x).contiguous(memory_format=torch.channels_last)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                logits = model(x)
                loss = F.cross_entropy(logits.float(), y, label_smoothing=a.label_smoothing)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            ema.update_parameters(model)
            tot += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            n += len(y)
        xv, yv = data["val"]
        val_acc = (predict(ema.module, xv).argmax(1) == yv).float().mean().item()
        log.log(epoch=ep + 1, loss=tot / n, train_acc=correct / n, val_acc=val_acc, lr=sched.get_last_lr()[0])
        if val_acc >= best:
            best = val_acc
            torch.save({"model": ema.module.state_dict(), "config": vars(a) | {"out": str(a.out)}}, a.out / "best.pt")

    ck = torch.load(a.out / "best.pt", map_location=dev)
    model.load_state_dict(ck["model"])
    xt, yt = data["test"]
    pred = predict(model, xt).argmax(1)
    metrics = {
        "val_acc": best,
        "test_acc": (pred == yt).float().mean().item(),
        "test_macro_f1": f1_score(yt.cpu(), pred.cpu(), average="macro"),
        "test_errors": int((pred != yt).sum()),
        "n_test": len(yt),
        "params": n_params,
        "train_minutes": round(log.rows[-1]["time_s"] / 60, 2),
        "config": vars(a) | {"out": str(a.out)},
    }
    save_json(metrics, a.out / "metrics.json")
    print("TEST", {k: v for k, v in metrics.items() if k != "config"})


if __name__ == "__main__":
    main()
