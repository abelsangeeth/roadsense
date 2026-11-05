# RoadSense

**An intelligent traffic-sign and road-condition detection framework using CNNs.**

RoadSense looks at a forward-facing camera frame and answers three questions a driver
(or a driver-assistance system) needs answered together:

1. **Which signs are in view?** A Faster R-CNN detector finds every sign in the full
   frame, and **SignNet**, a custom CNN with a spatial transformer and squeeze-and-excitation
   attention, recognises each one among the 43 German traffic-sign classes.
2. **What is the road surface like?** A multi-task CNN reads the lane ahead and predicts
   its **friction state** (dry / wet / standing water / fresh snow / melted snow / ice),
   its **material** (asphalt / concrete / gravel / mud) and its **unevenness** (smooth /
   slight / severe damage).
3. **What should I do about it?** A physics-based advisory layer fuses the two. It keeps
   the dry-road stopping distance of the posted limit on the actual surface
   (`v = v_limit · sqrt(μ / μ_dry)`) and raises hazard warnings.

Everything ships with calibrated confidences and an abstain option ("uncertain sign")
for low-confidence predictions, Grad-CAM explanations, a robustness benchmark, and a web
dashboard that runs on images, video files or a live camera.

<!-- RESULTS -->
## Results

| | metric | result |
|---|---|---|
| Traffic-sign recognition | GTSRB official test, 12,630 signs (3 seeds) | **99.84% ± 0.02** |
| Sign detection | GTSDB test scenes, AP@0.5 / AP@[.5:.95] | **95.7%** / 83.5% |
| Scene pipeline | GT signs found *and* named correctly | **93.9%** (0.04 false positives / frame, 8 fps) |
| Road friction state | RSCD, capture days never seen in training | **87.0%** |
| Road condition, all 27 classes | same | 66.6% (official leaky split: 82.2%) |
| Out-of-domain gate | accepts RSCD unseen-day images / rejects dashcam road crops | 98.6% / 98.1% |
| Dashcam road damage | RDD2022 held-out frames, 6 countries: F1 / mAP@0.5 (cracks + potholes) | **0.588** / 0.579 |

All accuracy numbers are computed in fp32, the precision the dashboard serves at. Full details are in [docs/REPORT.md](docs/REPORT.md).

## Architecture

```
            ┌──────────────────────── full frame (e.g. 1360×800) ─────────────────────────┐
            │                                                                             │
   Faster R-CNN R50-FPN v2 (COCO-pretrained,                       lower-centre road ROI
   anchors shifted to 16–256 px), class-agnostic                   (the lane ahead)
            │ boxes                                                        │ 128×192
   square crop, box + 15% margin, 48×48                   EfficientNet-B0 backbone (ImageNet)
            │                                                    ┌─────────┼──────────┐
   SignNet: STN → SE-ResNet (3 stages) → 43-way             friction(6) material(4) unevenness(3)
            │ temperature-scaled softmax                    masked multi-task heads
            │ abstain below a val-chosen threshold                       │
            └───────────────────────┬────────────────────────────────────┘
                              advisory layer
              posted limit · μ = E[μ | friction probs] · braking ratio
                 recommended speed · hazard level · messages
```

**Why two stages for signs?** GTSDB has about 600 annotated signs over 43 classes (a dozen
per class), while GTSRB has about 39,000 sign crops. The detector learns only *where* a
sign is, and SignNet, trained on GTSRB, decides *which* sign it is. Crops are framed
identically in training and at inference (`roadsense.data.prepare.crop_sign`).

**Why factorised surface heads?** Snow and ice hide the material, and gravel or mud has no
damage grade. With three heads and masked losses, every image teaches every attribute it
actually shows, and the advisory layer needs friction separately anyway. A flat 27-way
softmax is included as an ablation.

## Evaluation protocol: what makes the numbers trustworthy

* **GTSRB validation is split by track.** Each track is about 30 frames of one physical
  sign, so a random split would leak near-duplicates. The official 12,630-image test set
  is scored once, on the checkpoint chosen by validation accuracy.
* **RSCD is re-split by capture day.** RSCD's official split is frame-level random: 84%
  of its test images share the *exact capture second* with a training image (98% share
  the minute). We pool the images and assign whole capture days to train/val/test
  (60/9/15 days), so every test image comes from a drive the model never saw. The same
  recipe trained on the official split is reported alongside it to show the size of the
  leakage.
* **GTSDB scenes never train SignNet.** End-to-end recognition numbers are therefore a
  cross-dataset test of the classifier.
* **All thresholds come from validation data:** the temperature, the abstain threshold
  (val error ≤ 0.1%) and the detector threshold (val F1-optimal).

## Quick start

```bash
# environment (CUDA 12.8 wheels; CPU works too, just slower)
python -m venv venv && venv/Scripts/activate        # source venv/bin/activate on Linux/macOS
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

# trained weights (~470 MB) are attached to the GitHub release, not stored in git
python scripts/fetch_models.py

# run the dashboard with the trained release
python -m uvicorn serve.app:app --host 127.0.0.1 --port 8765
# open http://127.0.0.1:8765
```

```python
from PIL import Image
from roadsense.pipeline import RoadSense

rs = RoadSense.load("runs/release")
r = rs.analyze(Image.open("frame.jpg"))                     # dashcam frame
r = rs.analyze(Image.open("patch.jpg"), mode="surface")     # close-up road photo
r = rs.analyze(Image.open("sign.png"), mode="sign")         # single cropped sign
print(r["signs"], r["surface"]["friction"], r["advisory"])
```

## Reproducing everything

Data lives outside the repo (`ROADSENSE_DATA`, default `E:/roadsense-env/data`).

```bash
# 1. data: GTSRB + GTSDB (Hugging Face mirrors), RSCD stratified subsample (~35k images, rate-limited ~1 h)
python scripts/download_rscd.py --out $ROADSENSE_DATA/rscd
python -m roadsense.data.prepare gtsrb
python -m roadsense.data.prepare rscd        # official split cache
python -m roadsense.data.prepare rscd_day    # capture-day-disjoint split cache

# 2. train everything (sequential, about 3.5 h on a laptop RTX 3060; resumable)
bash scripts/run_all.sh

# 3. evaluate, then assemble the release
python -m roadsense.eval.evaluate_signs
python -m roadsense.eval.evaluate_surface
python scripts/make_release.py
python -m roadsense.eval.evaluate_e2e
python scripts/build_model_card.py

# tests (CPU, a few seconds)
pytest -q
```

GTSRB / GTSDB download URLs are in `docs/DATA.md`.

## Repository layout

| path | what |
|---|---|
| `roadsense/labels.py` | label spaces, GTSDB→GTSRB map, RSCD factorisation, flip-safe sign sets |
| `roadsense/data/` | preprocessing caches, batched GPU augmentation, GTSDB loader |
| `roadsense/models/` | SignNet (STN + SE-ResNet), surface CNN / EfficientNet multi-head, detector |
| `roadsense/train/` | training scripts (EMA weights, cosine schedule, bf16) |
| `roadsense/eval/` | metrics, calibration, 9 road-scene corruptions, Grad-CAM, evaluation reports |
| `roadsense/advisory.py` | physics-based fusion of signs and surface into a driving advisory |
| `roadsense/pipeline.py` | `RoadSense` inference API |
| `serve/`, `frontend/` | FastAPI backend and single-page dashboard |
| `reports/` | generated metrics JSON and figures |
| `docs/REPORT.md` | full technical report |

## Limitations

* **Domain.** Signs are German (Vienna Convention) signs, and RSCD comes from Chinese
  roads captured with a vehicle-mounted camera looking at the road surface ahead. On
  other countries' signs, or on a wide-angle dashcam whose road ROI is far away and
  small, accuracy will be lower than reported.
* **The friction coefficients are an engineering prior**, typical tyre-road values per
  surface state, not a measurement. The advisory is decision support, not a control
  signal.
* The GTSDB test split is small (54 scenes / 82 signs). Validation numbers are reported
  next to it for stability.
