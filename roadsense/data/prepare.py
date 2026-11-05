"""One-off preprocessing into compact uint8 caches so training runs entirely from RAM/GPU.

    python -m roadsense.data.prepare gtsrb
    python -m roadsense.data.prepare rscd
"""
from __future__ import annotations

import io
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from ..labels import RSCD_CLASSES
from ..paths import CACHE, DATA

SIGN_SIZE = 48
SIGN_MARGIN = 0.15          # context kept around the sign box; the detector crop uses the same value
SURFACE_HW = (128, 192)     # RSCD frames are 240x360 (2:3), keep the aspect ratio


def crop_sign(img: Image.Image, box, margin=SIGN_MARGIN, size=SIGN_SIZE) -> np.ndarray:
    """Square crop around box=(x1,y1,x2,y2) expanded by `margin` per side, resized to size x size.

    Shared by GTSRB preprocessing and the scene pipeline so classifier inputs are
    framed identically in training and at inference.
    """
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half = max(x2 - x1, y2 - y1) * (0.5 + margin)
    crop = img.crop((round(cx - half), round(cy - half), round(cx + half), round(cy + half)))  # pads with black off-image
    return np.asarray(crop.convert("RGB").resize((size, size), Image.BICUBIC))


def prepare_gtsrb():
    import pyarrow.parquet as pq

    import csv

    out = {}
    for split in ("train", "test"):
        path = next((DATA / "gtsrb").glob(f"{split}-*.parquet"))
        t = pq.read_table(path).to_pylist()
        # The parquet image structs carry no file name; the official CSV does, and its rows
        # are in the same order (all 7 metadata fields agree row-for-row, checked on import).
        meta = list(csv.DictReader(open(DATA / "gtsrb" / f"{split.title()}.csv")))
        assert len(meta) == len(t) and all(int(m["ClassId"]) == r["ClassId"] and int(m["Roi.X1"]) == r["Roi.X1"]
                                           for m, r in zip(meta, t))
        xs, ys, tracks = [], [], []
        for r, m in zip(t, meta):
            img = Image.open(io.BytesIO(r["Path"]["bytes"]))
            # GTSRB ROI is inclusive pixel coords of the sign
            box = (r["Roi.X1"], r["Roi.Y1"], r["Roi.X2"] + 1, r["Roi.Y2"] + 1)
            xs.append(crop_sign(img, box))
            ys.append(r["ClassId"])
            name = Path(m["Path"]).stem                  # train: CCCCC_TTTTT_FFFFF ; test: NNNNN
            tracks.append("_".join(name.split("_")[:2]) if split == "train" else "")
        out[f"x_{split}"] = np.stack(xs)
        out[f"y_{split}"] = np.array(ys, dtype=np.int64)
        if split == "train":
            out["track_train"] = np.array(tracks)
        print(split, out[f"x_{split}"].shape, "classes", len(set(ys)))
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE / "gtsrb48.npz", **out)
    print("tracks:", len(set(out["track_train"])))


def _load_surface(p: Path, hw=SURFACE_HW) -> np.ndarray:
    h, w = hw
    return np.asarray(Image.open(p).convert("RGB").resize((w, h), Image.BILINEAR))


def prepare_rscd():
    CACHE.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        files, labels = [], []
        for ci, c in enumerate(RSCD_CLASSES):
            fs = sorted((DATA / "rscd" / split / c).glob("*.jpg"))
            files += fs
            labels += [ci] * len(fs)
        missing = [c for c in RSCD_CLASSES if not (DATA / "rscd" / split / c).exists()]
        if missing:
            print(f"WARNING {split}: no folder for {missing}")
        with ThreadPoolExecutor(16) as ex:
            x = np.stack(list(ex.map(_load_surface, files)))
        np.savez(CACHE / f"rscd_{split}.npz", x=x, y=np.array(labels, dtype=np.int64),
                 files=np.array([f.name for f in files]))
        print(split, x.shape, np.bincount(labels, minlength=27))


def capture_day(fname: str) -> str:
    return fname.split("-")[0][:8]  # filenames start with a YYYYMMDDhhmmss... capture timestamp


def day_split(days_per_class: dict, counts: dict, seed=0, tries=4000, frac=(0.7, 0.1, 0.2)):
    """Assign whole capture days to train/val/test.

    RSCD's official split is frame-level random: 84% of its test images share the exact
    capture second with a training image (98% the minute), so the official test set mostly
    measures recall of near-duplicate frames. Splitting by day makes every test image come
    from a drive the model never saw. Searches random assignments for one where every class
    has images in all three splits and each split is close to its target share per class.
    """
    rng = np.random.default_rng(seed)
    days = sorted({d for ds in days_per_class.values() for d in ds})
    best, best_cost = None, np.inf
    for _ in range(tries):
        assign = {d: int(rng.choice(3, p=frac)) for d in days}
        cost = 0.0
        for c, per_day in counts.items():
            tot = sum(per_day.values())
            share = np.zeros(3)
            for d, n in per_day.items():
                share[assign[d]] += n
            share /= tot
            if share[1] == 0 or share[2] == 0 or share[0] < 0.4:
                cost = np.inf
                break
            cost += ((share - np.array(frac)) ** 2).sum()
        if cost < best_cost:
            best, best_cost = assign, cost
    return best, best_cost


def day_cache_name(split: str, hw=SURFACE_HW) -> str:
    return f"rscd_day_{split}.npz" if tuple(hw) == SURFACE_HW else f"rscd_day{hw[0]}_{split}.npz"


def prepare_rscd_daysplit(hw=SURFACE_HW):
    """Pool every downloaded RSCD image and re-split it by capture day -> rscd_day_{split}.npz.

    The day assignment is computed once and saved; later calls (e.g. a higher-resolution
    cache) reuse it, so every resolution sees exactly the same split."""
    import json
    from functools import partial
    from collections import defaultdict

    files, labels = [], []
    for split in ("train", "val", "test"):
        for ci, c in enumerate(RSCD_CLASSES):
            fs = sorted((DATA / "rscd" / split / c).glob("*.jpg"))
            files += fs
            labels += [ci] * len(fs)
    counts = defaultdict(lambda: defaultdict(int))
    for f, y in zip(files, labels):
        counts[y][capture_day(f.name)] += 1
    saved = CACHE / "rscd_day_split.json"
    if saved.exists():
        d = json.loads(saved.read_text())
        assign, cost = d["assign"], d["cost"]
    else:
        assign, cost = day_split({c: set(v) for c, v in counts.items()}, counts)
        saved.write_text(json.dumps({"assign": assign, "cost": cost}, indent=1))
    print(f"day split cost {cost:.4f}; days per split:", np.bincount(list(assign.values()), minlength=3), flush=True)
    with ThreadPoolExecutor(16) as ex:
        x = np.stack(list(ex.map(partial(_load_surface, hw=hw), files)))
    y = np.array(labels, dtype=np.int64)
    s = np.array([assign[capture_day(f.name)] for f in files])
    names = np.array([f.name for f in files])
    for k, split in enumerate(("train", "val", "test")):
        m = s == k
        np.savez(CACHE / day_cache_name(split, hw), x=x[m], y=y[m], files=names[m])
        print(split, int(m.sum()), np.bincount(y[m], minlength=27), flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "rscd_day" and len(sys.argv) > 2:  # e.g. rscd_day 192x288
        prepare_rscd_daysplit(tuple(int(v) for v in sys.argv[2].split("x")))
    else:
        {"gtsrb": prepare_gtsrb, "rscd": prepare_rscd, "rscd_day": prepare_rscd_daysplit}[sys.argv[1]]()
