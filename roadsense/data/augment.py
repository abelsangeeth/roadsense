"""Batched GPU augmentation on uint8 NHWC caches.

Doing this on the GPU keeps a laptop 3060 saturated without DataLoader worker processes
(slow to spawn on Windows). Every op is per-sample random within the batch.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from ..labels import FLIP_SELF, FLIP_SWAP

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def to_float(x_uint8_nhwc: torch.Tensor) -> torch.Tensor:
    return x_uint8_nhwc.permute(0, 3, 1, 2).float().div_(255)


def normalize(x: torch.Tensor) -> torch.Tensor:
    return (x - MEAN.to(x.device)) / STD.to(x.device)


def random_affine(x, rot=0.0, scale=(1.0, 1.0), trans=0.0, shear=0.0, out_hw=None):
    n = x.shape[0]
    dev = x.device
    a = (torch.rand(n, device=dev) * 2 - 1) * math.radians(rot)
    s = torch.empty(n, device=dev).uniform_(*scale)
    sh = (torch.rand(n, device=dev) * 2 - 1) * math.radians(shear)
    tx, ty = [(torch.rand(n, device=dev) * 2 - 1) * trans * 2 for _ in range(2)]
    cos, sin = torch.cos(a), torch.sin(a)
    # theta maps output -> input coords, so divide by scale to zoom in for s > 1
    theta = torch.stack([
        torch.stack([cos / s, (-sin + torch.tan(sh) * cos) / s, tx], -1),
        torch.stack([sin / s, (cos + torch.tan(sh) * sin) / s, ty], -1),
    ], 1)
    size = (n, x.shape[1], *(out_hw or x.shape[2:]))
    grid = F.affine_grid(theta, size, align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)


def color_jitter(x, b=0.3, c=0.3, s=0.3, gamma=0.0):
    n, dev = x.shape[0], x.device
    def u(m):
        return 1 + (torch.rand(n, 1, 1, 1, device=dev) * 2 - 1) * m
    x = x * u(b)
    mean = x.mean((1, 2, 3), keepdim=True)
    x = (x - mean) * u(c) + mean
    gray = (x * torch.tensor([0.299, 0.587, 0.114], device=dev).view(1, 3, 1, 1)).sum(1, keepdim=True)
    x = (x - gray) * u(s) + gray
    if gamma:
        x = x.clamp(1e-4, 1) ** u(gamma)
    return x.clamp(0, 1)


def random_blur(x, p=0.2, max_sigma=1.2):
    n = x.shape[0]
    mask = torch.rand(n, device=x.device) < p
    if not mask.any():
        return x
    sigma = float(torch.empty(1).uniform_(0.4, max_sigma))
    k = int(2 * math.ceil(2 * sigma) + 1)
    t = torch.arange(k, device=x.device, dtype=x.dtype) - k // 2
    g = torch.exp(-t ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    xb = x[mask]
    c = xb.shape[1]
    xb = F.conv2d(F.pad(xb, (k // 2,) * 4, mode="reflect"), g.view(1, 1, 1, k).expand(c, 1, 1, k), groups=c)
    xb = F.conv2d(xb, g.view(1, 1, k, 1).expand(c, 1, k, 1), groups=c)
    x = x.clone()
    x[mask] = xb
    return x


def random_noise(x, p=0.2, max_std=0.06):
    n = x.shape[0]
    std = torch.rand(n, 1, 1, 1, device=x.device) * max_std * (torch.rand(n, 1, 1, 1, device=x.device) < p)
    return (x + torch.randn_like(x) * std).clamp(0, 1)


def random_erase(x, p=0.25, area=(0.02, 0.12)):
    n, _, h, w = x.shape
    dev = x.device
    on = torch.rand(n, device=dev) < p
    ar = torch.empty(n, device=dev).uniform_(*area)
    asp = torch.exp(torch.empty(n, device=dev).uniform_(math.log(0.5), math.log(2)))
    eh = (torch.sqrt(ar * asp) * h).clamp(1, h)
    ew = (torch.sqrt(ar / asp) * w).clamp(1, w)
    y0 = torch.rand(n, device=dev) * (h - eh)
    x0 = torch.rand(n, device=dev) * (w - ew)
    yy = torch.arange(h, device=dev).view(1, h, 1)
    xx = torch.arange(w, device=dev).view(1, 1, w)
    m = (yy >= y0.view(-1, 1, 1)) & (yy < (y0 + eh).view(-1, 1, 1)) & (xx >= x0.view(-1, 1, 1)) & (xx < (x0 + ew).view(-1, 1, 1))
    m = (m & on.view(-1, 1, 1)).unsqueeze(1)
    return torch.where(m, torch.rand(n, 3, 1, 1, device=dev), x)


_SWAP = None


def sign_flip(x, y, p=0.5):
    """Horizontal flip that respects sign semantics: symmetric signs keep their label,
    mirror pairs (turn left <-> turn right, ...) swap it, everything else (speed limits,
    text, asymmetric pictograms) is never flipped."""
    global _SWAP
    if _SWAP is None or _SWAP.device != y.device:
        _SWAP = torch.full((43,), -1, dtype=torch.long)
        for c in FLIP_SELF:
            _SWAP[c] = c
        for a, b in FLIP_SWAP.items():
            _SWAP[a] = b
        _SWAP = _SWAP.to(y.device)
    target = _SWAP[y]
    do = (target >= 0) & (torch.rand(y.shape[0], device=y.device) < p)
    x = torch.where(do.view(-1, 1, 1, 1), x.flip(3), x)
    return x, torch.where(do, target, y)


def augment_signs(x, y, level="full"):
    """x: float NCHW in [0,1]."""
    if level == "none":
        return x, y
    if level == "basic":
        return random_affine(x, rot=10, scale=(0.9, 1.1), trans=0.05), y
    x, y = sign_flip(x, y)
    x = random_affine(x, rot=15, scale=(0.85, 1.15), trans=0.08, shear=8)
    x = color_jitter(x, 0.4, 0.4, 0.4, gamma=0.3)
    x = random_blur(x, 0.2)
    x = random_noise(x, 0.2)
    x = random_erase(x, 0.25)
    return x, y


def augment_surface(x, out_hw, level="full"):
    """Random crop/scale + mild photometric jitter. Kept mild on purpose: wet vs. dry and
    ice vs. snow are partly brightness/specularity cues, so heavy colour jitter would
    destroy the very signal the friction head needs."""
    if level == "none":
        return F.interpolate(x, size=out_hw, mode="bilinear", align_corners=False)
    flip = torch.rand(x.shape[0], device=x.device) < 0.5
    x = torch.where(flip.view(-1, 1, 1, 1), x.flip(3), x)
    x = random_affine(x, rot=5, scale=(1.0, 1.3), trans=0.08, out_hw=out_hw)
    if level == "full":
        x = color_jitter(x, 0.15, 0.15, 0.15)
        x = random_blur(x, 0.15, 1.0)
        x = random_noise(x, 0.15, 0.03)
    return x
