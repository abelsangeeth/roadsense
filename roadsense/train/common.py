from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def device():
    return torch.device("cuda" if torch.cuda.device_count() > 0 else "cpu")


def make_ema(model, decay=0.999):
    """EMA of weights *and* BN buffers, with warm-up decay min(decay, (1+n)/(10+n)).

    Without the warm-up a short run exports weights that are still mostly the random
    initialisation (0.998**275 ~ 0.58 of the init survives a 2-epoch smoke run)."""
    @torch.no_grad()
    def avg_fn(avg_list, cur_list, n):
        d = min(decay, (1 + int(n)) / (10 + int(n)))
        fa = [a for a in avg_list if a.is_floating_point()]
        fc = [c for c in cur_list if c.is_floating_point()]
        if fa:  # AveragedModel calls this once per (device, dtype) group
            torch._foreach_lerp_(fa, fc, 1 - d)
        for a, c in zip(avg_list, cur_list):  # BN num_batches_tracked
            if not a.is_floating_point():
                a.copy_(c)
    return AveragedModel(model, multi_avg_fn=avg_fn, use_buffers=True)


def cosine_warmup(opt, total_steps, warmup_steps):
    def f(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        p = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * p)) * 0.99 + 0.01
    return torch.optim.lr_scheduler.LambdaLR(opt, f)


def param_groups(model, wd):
    """No weight decay on norms / biases."""
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim <= 1 else decay).append(p)
    return [{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}]


def batches(n, bs, shuffle, gen=None):
    idx = torch.randperm(n, generator=gen) if shuffle else torch.arange(n)
    for i in range(0, n, bs):
        yield idx[i:i + bs]


class Logger:
    def __init__(self, out: Path):
        self.out = out
        out.mkdir(parents=True, exist_ok=True)
        self.rows = []
        self.t0 = time.time()

    def log(self, **row):
        row["time_s"] = round(time.time() - self.t0, 1)
        self.rows.append(row)
        print(" ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()), flush=True)
        (self.out / "history.json").write_text(json.dumps(self.rows, indent=1))


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))
