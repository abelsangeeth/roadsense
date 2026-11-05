"""Build a slim, shareable zip: code + dashboard + reports + the trained release models.

Training checkpoints, logs and datasets are left out; the release runs on CPU with nothing
downloaded.

    python scripts/package.py --out dist/RoadSense.zip
"""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INCLUDE = ["roadsense", "serve", "frontend", "reports", "docs", "scripts", "tests", "runs/release",
           "README.md", "requirements.txt", "RUN_ME.txt", "run_windows.bat"]
REQUIRED = ["runs/release/signnet.pt", "runs/release/detector.pt", "runs/release/surface.pt",
            "runs/release/surface_ood.npz", "runs/release/damage.pt", "runs/release/calibration.json"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "dist" / "RoadSense.zip")
    a = ap.parse_args()
    missing = [p for p in REQUIRED if not (ROOT / p).exists()]
    if missing:
        raise SystemExit(f"release incomplete, missing: {missing}")
    files = []
    for item in INCLUDE:
        p = ROOT / item
        files += [p] if p.is_file() else [f for f in sorted(p.rglob("*")) if f.is_file()]
    files = [f for f in files if "__pycache__" not in f.parts and f.suffix not in (".pyc", ".log")]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(a.out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f, Path("RoadSense") / f.relative_to(ROOT))
    print(f"{a.out}  {len(files)} files  {a.out.stat().st_size / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
