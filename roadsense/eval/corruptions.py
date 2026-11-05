"""Road-specific image corruptions at 3 severities, for robustness benchmarking.

Deterministic given the seed so every model sees exactly the same corrupted test set.
All functions take / return float NCHW tensors in [0, 1].
"""
from __future__ import annotations

import io
import math

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def _gen(x, seed):
    return torch.Generator(device=x.device).manual_seed(seed)


def gaussian_noise(x, s, seed=0):
    std = [0.04, 0.08, 0.14][s]
    return (x + torch.randn(x.shape, generator=_gen(x, seed), device=x.device) * std).clamp(0, 1)


def motion_blur(x, s, seed=0):
    k = [5, 9, 13][s]
    kern = torch.zeros(k, k, device=x.device)
    kern[k // 2, :] = 1 / k  # horizontal: camera / vehicle motion
    c = x.shape[1]
    return F.conv2d(F.pad(x, (k // 2,) * 4, mode="replicate"), kern.expand(c, 1, k, k), groups=c)


def defocus(x, s, seed=0):
    r = [1.0, 1.8, 2.6][s] * x.shape[-1] / 48
    k = int(2 * math.ceil(r) + 1)
    t = torch.arange(k, device=x.device) - k // 2
    disk = ((t.view(-1, 1) ** 2 + t.view(1, -1) ** 2) <= r * r).float()
    disk /= disk.sum()
    c = x.shape[1]
    return F.conv2d(F.pad(x, (k // 2,) * 4, mode="replicate"), disk.expand(c, 1, k, k), groups=c)


def fog(x, s, seed=0):
    a = [0.35, 0.55, 0.75][s]
    n, _, h, w = x.shape
    g = _gen(x, seed)
    haze = F.interpolate(torch.rand(n, 1, 4, 4, generator=g, device=x.device), (h, w), mode="bicubic", align_corners=False)
    density = (a * (0.7 + 0.6 * haze)).clamp(0, 0.95)
    return x * (1 - density) + 0.8 * density


def rain(x, s, seed=0):
    n, _, h, w = x.shape
    g = _gen(x, seed)
    p = [0.004, 0.008, 0.015][s]
    drops = (torch.rand(n, 1, h, w, generator=g, device=x.device) < p).float()
    L = max(3, int(h * [0.12, 0.18, 0.25][s]))
    kern = torch.zeros(L, L, device=x.device)
    for i in range(L):  # slanted streak
        kern[i, min(L - 1, int(i * 0.3))] = 1
    streak = F.conv2d(drops, kern.view(1, 1, L, L), padding=L // 2)[..., :h, :w].clamp(0, 1)
    out = x * (1 - 0.15 * [1, 1.5, 2][s]) + 0.7 * streak  # darker overcast + bright streaks
    return out.clamp(0, 1)


def low_light(x, s, seed=0):
    gain = [0.45, 0.28, 0.16][s]
    y = x ** 1.3 * gain
    y = y + torch.randn(x.shape, generator=_gen(x, seed), device=x.device) * [0.01, 0.02, 0.03][s]  # sensor noise
    return y.clamp(0, 1)


def glare(x, s, seed=0):
    n, _, h, w = x.shape
    g = _gen(x, seed)
    cy, cx = torch.rand(n, 1, 1, generator=g, device=x.device) * h, torch.rand(n, 1, 1, generator=g, device=x.device) * w
    yy, xx = torch.meshgrid(torch.arange(h, device=x.device), torch.arange(w, device=x.device), indexing="ij")
    d2 = ((yy - cy) ** 2 + (xx - cx) ** 2) / (0.35 * max(h, w)) ** 2
    flare = torch.exp(-d2).unsqueeze(1) * [0.5, 0.8, 1.1][s]
    return (x + flare).clamp(0, 1)


def occlusion(x, s, seed=0):
    """A random grey/dark patch (sticker, foliage, dirt) covering a fraction of the image."""
    n, _, h, w = x.shape
    frac = [0.10, 0.20, 0.30][s]
    rng = np.random.default_rng(seed)
    out = x.clone()
    for i in range(n):
        ph, pw = int(h * math.sqrt(frac)), int(w * math.sqrt(frac))
        y0, x0 = rng.integers(0, h - ph + 1), rng.integers(0, w - pw + 1)
        out[i, :, y0:y0 + ph, x0:x0 + pw] = float(rng.uniform(0.05, 0.5))
    return out


def jpeg(x, s, seed=0):
    q = [25, 12, 6][s]
    out = []
    for im in x.mul(255).round().byte().permute(0, 2, 3, 1).cpu().numpy():
        b = io.BytesIO()
        Image.fromarray(im).save(b, "JPEG", quality=q)
        out.append(np.asarray(Image.open(b)))
    return torch.from_numpy(np.stack(out)).permute(0, 3, 1, 2).float().div(255).to(x.device)


CORRUPTIONS = {
    "gaussian_noise": gaussian_noise, "motion_blur": motion_blur, "defocus": defocus,
    "fog": fog, "rain": rain, "low_light": low_light, "glare": glare,
    "occlusion": occlusion, "jpeg": jpeg,
}
