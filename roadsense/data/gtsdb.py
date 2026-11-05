"""GTSDB scenes (Roboflow COCO export) with boxes in x1y1x2y2 and labels in GTSRB ids."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from ..labels import GTSDB_TO_GTSRB
from ..paths import DATA

SPLITS = {"train": "train", "val": "valid", "test": "test"}


def load_gtsdb(split: str, root: Path | None = None) -> list[dict]:
    d = (root or DATA / "gtsdb") / SPLITS[split]
    coco = json.loads((d / "_annotations.coco.json").read_text())
    cat = {c["id"]: GTSDB_TO_GTSRB[c["name"]] for c in coco["categories"] if c["name"] in GTSDB_TO_GTSRB}
    by_img = {im["id"]: {"path": d / im["file_name"], "boxes": [], "labels": []} for im in coco["images"]}
    for a in coco["annotations"]:
        x, y, w, h = a["bbox"]
        by_img[a["image_id"]]["boxes"].append([x, y, x + w, y + h])
        by_img[a["image_id"]]["labels"].append(cat[a["category_id"]])
    out = []
    for r in by_img.values():
        r["boxes"] = np.array(r["boxes"], dtype=np.float32).reshape(-1, 4)
        r["labels"] = np.array(r["labels"], dtype=np.int64)
        out.append(r)
    return sorted(out, key=lambda r: r["path"].name)


def load_image(path) -> Image.Image:
    return Image.open(path).convert("RGB")
