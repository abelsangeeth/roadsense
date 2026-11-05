"""RDD2022 road-damage scenes (CRDDC'2022, CC BY 4.0) for the dashcam damage detector.

Images come from smartphones on the dashboard / dashcams looking forward along the road
(Japan, India, Czech, Norway, United States, China motorbike). China_Drone is a top-down
view and is left out. Only the four classes shared by every country are kept:
D00 longitudinal crack, D10 transverse crack, D20 alligator crack, D40 pothole.

Only the train folders are labelled, so we make our own 80/10/10 split per country. The
release shuffles file numbers (neighbouring numbers are unrelated scenes), so there is no
capture order to split by; instead `check_duplicates` hashes every image and counts
near-identical pairs that straddle splits (the leak that RSCD's official split had).

    python -m roadsense.data.rdd --src E:/roadsense-env/data/rdd2022/RDD2022
"""
from __future__ import annotations

import argparse
import json
import re
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from ..paths import CACHE, DATA

DAMAGE_CLASSES = ["longitudinal crack", "transverse crack", "alligator crack", "pothole"]
CODES = {"D00": 0, "D10": 1, "D20": 2, "D40": 3}
COUNTRIES = ["Japan", "India", "Czech", "Norway", "United_States", "China_MotorBike"]
MAX_SIDE = 1024   # Norway frames are 3650x2044; everything else is <= 720
FRACTIONS = {"train": 0.8, "val": 0.1, "test": 0.1}
OUT = CACHE / "rdd"


def parse_xml(path: Path) -> tuple[np.ndarray, np.ndarray]:
    root = ET.parse(path).getroot()
    boxes, labels = [], []
    for o in root.iter("object"):
        name = (o.findtext("name") or "").strip()
        if name not in CODES:
            continue
        b = o.find("bndbox")
        boxes.append([float(b.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax")])
        labels.append(CODES[name])
    return np.array(boxes, np.float32).reshape(-1, 4), np.array(labels, np.int64)


def _num(p: Path) -> int:
    m = re.search(r"(\d+)$", p.stem)
    return int(m.group(1)) if m else 0


def _cache_one(args):
    country, img_path, xml_path = args
    boxes, labels = parse_xml(xml_path) if xml_path.exists() else (np.zeros((0, 4), np.float32), np.zeros(0, np.int64))
    im = Image.open(img_path).convert("RGB")
    w, h = im.size
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)
    ok = (boxes[:, 2] - boxes[:, 0] >= 2) & (boxes[:, 3] - boxes[:, 1] >= 2)
    boxes, labels = boxes[ok], labels[ok]
    s = min(1.0, MAX_SIDE / max(w, h))
    if s < 1:
        im = im.resize((round(w * s), round(h * s)), Image.BICUBIC)
        boxes = boxes * s
    dst = OUT / country / (img_path.stem + ".jpg")
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        im.save(dst, quality=93)
    return {"file": f"{country}/{dst.name}", "country": country, "num": _num(img_path),
            "boxes": boxes.round(1).tolist(), "labels": labels.tolist()}


def prepare(src: Path, seed=0):
    jobs = []
    for c in COUNTRIES:
        d = src / c / "train"
        for p in sorted((d / "images").glob("*.jpg")):
            jobs.append((c, p, d / "annotations" / "xmls" / (p.stem + ".xml")))
    print("images", len(jobs))
    with ThreadPoolExecutor(8) as ex:
        recs = list(ex.map(_cache_one, jobs, chunksize=32))
    rng = np.random.default_rng(seed)
    for c in COUNTRIES:
        rs = sorted([r for r in recs if r["country"] == c], key=lambda r: r["num"])
        n_tr, n_va = round(len(rs) * FRACTIONS["train"]), round(len(rs) * FRACTIONS["val"])
        for rank, i in enumerate(rng.permutation(len(rs))):
            rs[i]["split"] = "train" if rank < n_tr else "val" if rank < n_tr + n_va else "test"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "index.json").write_text(json.dumps(recs))
    for sp in FRACTIONS:
        sel = [r for r in recs if r["split"] == sp]
        counts = np.bincount([l for r in sel for l in r["labels"]], minlength=4)
        print(sp, len(sel), "images", sum(1 for r in sel if r["labels"]), "with damage", dict(zip(DAMAGE_CLASSES, counts.tolist())))


def dhash(path, n=16) -> int:
    g = np.asarray(Image.open(path).convert("L").resize((n + 1, n), Image.BILINEAR), np.int16)
    return int("".join("1" if v else "0" for v in (g[:, 1:] > g[:, :-1]).ravel()), 2)


def check_duplicates(max_bits=10) -> dict:
    """Near-identical image pairs (256-bit difference hash within max_bits) across splits."""
    recs = json.loads((OUT / "index.json").read_text())
    with ThreadPoolExecutor(8) as ex:
        hs = list(ex.map(lambda r: dhash(OUT / r["file"]), recs, chunksize=64))
    bits = np.array([[(h >> k) & 1 for k in range(256)] for h in hs], np.uint8)
    split = np.array([r["split"] for r in recs])
    x = near_pairs(bits, max_bits)
    cross = [(i, j) for i, j in x if split[i] != split[j]]
    out = {"pairs_within_%d_bits" % max_bits: len(x), "cross_split_pairs": len(cross),
           "test_images_with_train_twin": len({j if split[j] == "test" else i for i, j in cross
                                               if "test" in (split[i], split[j]) and "train" in (split[i], split[j])})}
    return out | {"examples": [(recs[i]["file"], recs[j]["file"]) for i, j in cross[:20]]}


def near_pairs(bits: np.ndarray, max_bits: int):
    """All pairs (i<j) with Hamming distance <= max_bits, in chunks to bound memory."""
    b = bits.astype(np.float32) * 2 - 1  # +-1, so dot = 256 - 2 * hamming
    pairs = []
    for i in range(0, len(b), 1024):
        d = (256 - b[i:i + 1024] @ b.T) / 2
        ii, jj = np.nonzero(d <= max_bits)
        pairs += [(int(a + i), int(c)) for a, c in zip(ii, jj) if a + i < c]
    return pairs


def load_rdd(split: str, countries=None) -> list[dict]:
    recs = json.loads((OUT / "index.json").read_text())
    out = []
    for r in recs:
        if r["split"] != split or (countries and r["country"] not in countries):
            continue
        r["path"] = OUT / r["file"]
        r["boxes"] = np.array(r["boxes"], np.float32).reshape(-1, 4)
        r["labels"] = np.array(r["labels"], np.int64)
        out.append(r)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=DATA / "rdd2022" / "RDD2022")
    ap.add_argument("--check", action="store_true", help="only run the cross-split duplicate check")
    a = ap.parse_args()
    if not a.check:
        prepare(a.src)
    rep = check_duplicates()
    (OUT / "duplicates.json").write_text(json.dumps(rep, indent=1))
    print({k: v for k, v in rep.items() if k != "examples"})
