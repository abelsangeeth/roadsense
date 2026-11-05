# Data sources

| dataset | use | source |
|---|---|---|
| GTSRB (German Traffic Sign Recognition Benchmark), 39,209 train / 12,630 test | SignNet training and test | Hugging Face mirror `bazyl/GTSRB` (`data/*.parquet` + `Train.csv`/`Test.csv`; the CSV supplies the file names, and so the track ids, which the parquet lacks) |
| GTSDB (German Traffic Sign Detection Benchmark), 383 / 108 / 54 scenes, 1360x800 | detector training, end-to-end test | Hugging Face `keremberke/german-traffic-sign-detection` (Roboflow COCO export, CC BY 4.0) |
| RSCD (Road Surface Classification Dataset), 27 classes | surface model | Hugging Face `rezzzq/RSCD-1million` (partial mirror: train folders capped at 10k per class, 2,215 val, 5,393 test). `scripts/download_rscd.py` takes a seeded sample of <=1,200 train images per class plus all val/test images, 34,585 images in total |

The original GTSRB/GTSDB host (sid.erda.dk) was unreachable from the build machine, hence the mirrors.

The RSCD day-disjoint split (`python -m roadsense.data.prepare rscd_day`) pools all 34,585
downloaded images and assigns each capture day (from the filename timestamp) to one split.
The assignment is saved to `rscd_day_split.json` in the cache directory.

## RDD2022 (dashcam road damage)

Arya et al., *RDD2022: A multi-national image dataset for automatic Road Damage Detection*
(CRDDC'2022), figshare article 21431547, licensed CC BY 4.0. Download
`RDD2022_released_through_CRDDC2022.zip` (13 GB, a zip of per-country zips; Git Bash `unzip`
cannot read it, use Python `zipfile`), extract each country's `train/` folder to
`$ROADSENSE_DATA/rdd2022/RDD2022/<Country>/train`, then run `python -m roadsense.data.rdd`.
The two `frontend/samples/scene__rdd_*.jpg` images come from this dataset.
