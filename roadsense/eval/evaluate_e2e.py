"""End-to-end scene evaluation on GTSDB: detector -> SignNet, against 43-class ground truth.

    python -m roadsense.eval.evaluate_e2e --release runs/release

Reports class-agnostic detection AP, class-aware AP (score = det_score x sign_conf), the
end-to-end recognition rate (GT signs both found at IoU>=0.5 and named correctly) and
throughput. The GTSDB scenes were never used to train SignNet.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from ..data.gtsdb import load_gtsdb, load_image
from ..labels import GTSRB_CLASSES
from ..paths import REPORTS
from ..pipeline import RoadSense
from ..train.common import save_json
from .detection_metrics import box_iou, detection_report

OUT = REPORTS / "e2e"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", type=Path, default=Path("runs/release"))
    ap.add_argument("--split", default="test")
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    rs = RoadSense.load(a.release)
    rs.det_thr = 0.05  # keep low-score boxes for AP; the operating threshold is applied below
    recs = load_gtsdb(a.split)

    preds_agn, preds_cls, gts_agn, gts_cls, rows = [], [], [], [], []
    times = []
    for r in recs:
        img = load_image(r["path"])
        if rs.dev.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        boxes, ds = rs.detect(img)
        probs, _ = rs.classify_signs(img, boxes)
        if rs.dev.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        cls = probs.argmax(1) if len(probs) else np.zeros(0, int)
        conf = probs.max(1) if len(probs) else np.zeros(0)
        preds_agn.append((boxes, ds, np.zeros(len(boxes), int)))
        preds_cls.append((boxes, ds * conf, cls))
        gts_agn.append((r["boxes"], np.zeros(len(r["boxes"]), int)))
        gts_cls.append((r["boxes"], r["labels"]))
        rows.append((r, img, boxes, ds, cls, conf))

    thr = rs.calib.get("detector_threshold", 0.5)
    found = correct = n_gt = fp = 0
    for r, _, boxes, ds, cls, conf in rows:
        keep = ds >= thr
        b, c = boxes[keep], cls[keep]
        n_gt += len(r["boxes"])
        iou = box_iou(r["boxes"], b)
        matched = set()
        for g in range(len(r["boxes"])):
            if iou.shape[1] and iou[g].max() >= 0.5:
                j = int(iou[g].argmax())
                matched.add(j)
                found += 1
                correct += int(c[j] == r["labels"][g])
        fp += int(keep.sum()) - len(matched)

    rep = {
        "split": a.split, "n_images": len(recs), "n_gt": n_gt,
        "detection_class_agnostic": detection_report(preds_agn, gts_agn),
        "detection_class_aware_43": detection_report(preds_cls, gts_cls, classwise=True),
        "operating_point": {"detector_threshold": thr, "recall": found / n_gt,
                            "recognition_rate": correct / n_gt, "classification_acc_of_found": correct / max(found, 1),
                            "false_positives_per_image": fp / len(recs)},
        "latency_ms_median": float(np.median(times[2:]) * 1000), "fps": float(1 / np.median(times[2:])),
        "device": str(rs.dev),
    }
    save_json(rep, OUT / f"report_{a.split}.json")
    print(rep)

    # a few annotated scenes for the report
    for r, img, boxes, ds, cls, conf in rows[:6]:
        im = img.copy()
        d = ImageDraw.Draw(im)
        for g, l in zip(r["boxes"], r["labels"]):
            d.rectangle(list(g), outline=(255, 255, 255), width=1)
        for b, s, c, p in zip(boxes, ds, cls, conf):
            if s < thr:
                continue
            ok = any(box_iou(b[None], g[None])[0, 0] >= 0.5 and c == l for g, l in zip(r["boxes"], r["labels"]))
            col = (27, 175, 122) if ok else (227, 73, 72)
            d.rectangle(list(b), outline=col, width=3)
            d.text((b[0], max(0, b[1] - 12)), f"{GTSRB_CLASSES[c][:22]} {p:.2f}", fill=col)
        im.save(OUT / f"scene_{r['path'].stem[:5]}.jpg", quality=88)


if __name__ == "__main__":
    main()
