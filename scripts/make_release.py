"""Assemble runs/release: chosen checkpoints + calibration + dashboard samples.

Run after training and after `evaluate_signs` / `evaluate_surface` (which write the
calibration files). Every threshold here is chosen on validation data, never test.

    python scripts/make_release.py
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from roadsense.data.gtsdb import load_gtsdb, load_image  # noqa: E402
from roadsense.eval.detection_metrics import match  # noqa: E402
from roadsense.labels import RSCD_CLASSES  # noqa: E402
from roadsense.paths import CACHE, REPORTS, ROOT, RUNS  # noqa: E402
from roadsense.pipeline import RoadSense  # noqa: E402

REL = RUNS / "release"
SAMPLES = ROOT / "frontend" / "samples"


def best_sign_run():
    runs = [p for p in (RUNS / "signs").glob("full_s*") if (p / "metrics.json").exists()]
    return max(runs, key=lambda p: json.loads((p / "metrics.json").read_text())["val_acc"])


def main():
    REL.mkdir(parents=True, exist_ok=True)
    sign_run = best_sign_run()
    shutil.copy(sign_run / "best.pt", REL / "signnet.pt")
    calib = json.loads((REPORTS / "signs" / "calibration.json").read_text())
    surf_cal = json.loads((REPORTS / "surface" / "calibration.json").read_text())
    shutil.copy(RUNS / "surface" / surf_cal["surface_run"] / "best.pt", REL / "surface.pt")
    shutil.copy(RUNS / "detector" / "best.pt", REL / "detector.pt")
    calib |= surf_cal | {"sign_run": sign_run.name}
    (REL / "calibration.json").write_text(json.dumps(calib, indent=2))

    # detector operating threshold: max F1 (IoU 0.5) on the validation scenes
    rs = RoadSense.load(REL)
    rs.det_thr = 0.05
    preds, gts = [], []
    for r in load_gtsdb("val"):
        b, s = rs.detect(load_image(r["path"]))
        preds.append((b, s, np.zeros(len(b), int)))
        gts.append((r["boxes"], np.zeros(len(r["boxes"]), int)))
    s, tp, n = match(preds, gts, 0.5)
    best = (0, 0.5)
    for t in np.arange(0.05, 0.96, 0.01):
        k = s >= t
        p, rc = tp[k].mean() if k.any() else 0, tp[k].sum() / n
        f1 = 2 * p * rc / (p + rc) if p + rc else 0
        if f1 > best[0]:
            best = (f1, float(t))
    calib["detector_threshold"] = round(best[1], 2)
    calib["detector_val_f1"] = round(best[0], 4)
    (REL / "calibration.json").write_text(json.dumps(calib, indent=2))
    print("calibration", calib)

    # ---------------------------------------------------------- dashboard samples (all held-out test data)
    if SAMPLES.exists():
        shutil.rmtree(SAMPLES)
    SAMPLES.mkdir(parents=True)
    test_scenes = load_gtsdb("test")
    for r in test_scenes[::9][:6]:
        load_image(r["path"]).save(SAMPLES / f"scene__{r['path'].stem[:5]}.jpg", quality=90)
    g = np.load(CACHE / "gtsrb48.npz")
    rng = np.random.default_rng(3)
    for i in rng.choice(len(g["y_test"]), 4, replace=False):
        Image.fromarray(g["x_test"][i]).resize((144, 144), Image.BICUBIC).save(SAMPLES / f"sign__gtsrb{i:05d}.jpg", quality=95)
    from roadsense.paths import DATA
    for c in ("dry_asphalt_smooth", "wet_asphalt_slight", "water_concrete_smooth", "dry_asphalt_severe", "fresh_snow", "ice"):
        f = sorted((DATA / "rscd" / "test" / c).glob("*.jpg"))
        if f:
            shutil.copy(f[len(f) // 2], SAMPLES / f"surface__{c}.jpg")
    print("samples:", len(list(SAMPLES.glob("*.jpg"))))


if __name__ == "__main__":
    main()
