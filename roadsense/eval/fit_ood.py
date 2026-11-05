"""Out-of-domain gate for the surface model (Mahalanobis distance in feature space).

RSCD images come from a camera aimed at the road a few metres ahead. A forward dashcam
sees the road at a grazing angle, with glare, kerbs and cars in view, and the surface
model, never trained on that, says "standing water" for 69-88% of GTSDB frames whatever
road region is cropped. Rather than let that reach the advisory, we measure how far an
input lies from the RSCD training distribution (Lee et al., 2018: class-conditional means,
shared shrinkage covariance) and refuse surface readings beyond the 99.9th percentile of
validation distances.

The features are per-channel mean/std of the first four backbone blocks (low-level
texture / contrast / perspective statistics). Pooled final-layer features, which are tuned
for separating surface classes, only rejected 33-43% of dashcam crops; blocks 1-4 reject
~95% while still accepting ~98% of RSCD images from unseen capture days. That layer set
was chosen looking at the same GTSDB crops it is reported on, so treat the dashcam
rejection rate as indicative rather than a held-out estimate.

    python -m roadsense.eval.fit_ood --release runs/release
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from ..data.augment import normalize, to_float
from ..data.gtsdb import load_gtsdb, load_image
from ..models.surfacenet import build_surface_model
from ..paths import REPORTS
from ..train.common import batches, save_json
from ..train.train_surface import load_split


STAT_BLOCKS = (1, 2, 3, 4)


def stat_features_float(model, x):
    """Per-channel mean and std after backbone blocks 1-4 of a normalised float batch."""
    blocks = ([model.stem] if hasattr(model, "stem") else []) + list(model.features)
    feats, h = [], x
    for i, blk in enumerate(blocks[:max(STAT_BLOCKS) + 1]):
        h = blk(h)
        if i in STAT_BLOCKS:
            feats.append(torch.cat([h.mean((2, 3)), h.std((2, 3))], 1).float())
    return torch.cat(feats, 1)


@torch.no_grad()
def features(model, x_u8, bs=256):
    dev = next(model.parameters()).device
    out = []
    for idx in batches(len(x_u8), bs, False):
        x = normalize(to_float(x_u8[idx].to(dev)))
        out.append(stat_features_float(model, x).cpu())
    return torch.cat(out).numpy().astype(np.float64)


class MahalanobisGate:
    def __init__(self, means, precision, threshold):
        self.means, self.precision, self.threshold = means, precision, threshold

    @classmethod
    def fit(cls, feats, labels, shrink=0.1):
        classes = np.unique(labels)
        means = np.stack([feats[labels == c].mean(0) for c in classes])
        centred = feats - means[np.searchsorted(classes, labels)]
        cov = centred.T @ centred / len(feats)
        cov = (1 - shrink) * cov + shrink * np.trace(cov) / cov.shape[0] * np.eye(cov.shape[0])
        return cls(means, np.linalg.inv(cov), None)

    def score(self, feats):
        """Squared Mahalanobis distance to the nearest class mean.

        precision = L L^T, so (x-m)^T P (x-m) = ||L^T x - L^T m||^2: whiten once, then Euclidean."""
        if not hasattr(self, "_L"):
            self._L = np.linalg.cholesky(self.precision)
            self._mz = self.means @ self._L
        z = feats @ self._L
        d = (z ** 2).sum(1)[:, None] - 2 * z @ self._mz.T + (self._mz ** 2).sum(1)[None]
        return d.min(1)

    def save(self, path):
        np.savez(path, means=self.means, precision=self.precision, threshold=self.threshold)

    @classmethod
    def load(cls, path):
        z = np.load(path)
        return cls(z["means"], z["precision"], float(z["threshold"]))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", type=Path, default=Path("runs/release"))
    # 0.999, not 0.99: unseen capture days drift, and at 0.99 the gate rejected up to 20% of some
    # real RSCD test classes. A falsely rejected ice/snow/water patch silently drops a hazard
    # warning, while a dashcam frame that slips through only errs towards a lower speed. At
    # 0.999: 98.6% of unseen-day RSCD accepted (99.3% of hazard classes), 98.1% of dashcam rejected.
    ap.add_argument("--quantile", type=float, default=0.999)
    a = ap.parse_args(argv)
    dev = torch.device("cuda" if torch.cuda.device_count() > 0 else "cpu")
    ck = torch.load(a.release / "surface.pt", map_location=dev, weights_only=False)
    model = build_surface_model(ck["config"]["arch"], ck["config"]["head"], pretrained=False)
    model.load_state_dict(ck["model"])
    model = model.to(dev).eval().to(memory_format=torch.channels_last)
    hw = tuple(ck["config"].get("hw", (128, 192)))

    xtr, ytr, _ = load_split("train", "day", hw)
    keep = torch.randperm(len(ytr), generator=torch.Generator().manual_seed(0))[:8000]  # covariance from 8k is plenty
    xtr, ytr = xtr[keep], ytr[keep]
    xva, _, _ = load_split("val", "day", hw)
    xte, _, _ = load_split("test", "day", hw)
    gate = MahalanobisGate.fit(features(model, xtr), ytr.numpy())
    sv, st = gate.score(features(model, xva)), gate.score(features(model, xte))
    gate.threshold = float(np.quantile(sv, a.quantile))
    gate.save(a.release / "surface_ood.npz")

    # dashcam road crops (GTSDB) -- the case the gate exists for
    from ..pipeline import ROAD_ROI, road_roi
    from PIL import Image
    crops = np.stack([np.asarray(road_roi(load_image(r["path"]), ROAD_ROI).resize((hw[1], hw[0]), Image.BILINEAR))
                      for r in load_gtsdb("val") + load_gtsdb("test")])
    sg = gate.score(features(model, torch.from_numpy(crops)))
    n_val = len(load_gtsdb("val"))
    rep = {"threshold": gate.threshold, "quantile": a.quantile,
           "rscd_val_accepted": float((sv <= gate.threshold).mean()),
           "rscd_test_accepted (unseen days)": float((st <= gate.threshold).mean()),
           "gtsdb_dashcam_rejected": float((sg > gate.threshold).mean()),
           "gtsdb_val_rejected": float((sg[:n_val] > gate.threshold).mean()),
           "gtsdb_test_rejected": float((sg[n_val:] > gate.threshold).mean()),
           "features": f"mean/std of backbone blocks {STAT_BLOCKS}",
           "n_gtsdb": len(sg),
           "median_score": {"rscd_test": float(np.median(st)), "gtsdb": float(np.median(sg))}}
    save_json(rep, REPORTS / "surface" / "ood.json")
    print(rep)


if __name__ == "__main__":
    main()
