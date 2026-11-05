#!/usr/bin/env bash
# Full training queue, sequential on one GPU. Each job skips itself if metrics.json exists,
# so the script is safe to re-run after an interruption.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-/e/roadsense-env/venv/Scripts/python}

job() {  # job <out_dir> <module> <args...>
  local out=$1; shift
  if [ -f "$out/metrics.json" ]; then echo "skip $out"; return; fi
  mkdir -p "$(dirname "$out")"
  echo "=== $(date +%T) $out"
  "$PY" -u -m "$@" --out "$out" > "$out.log" 2>&1
  echo "EXIT=$? $out" | tee -a "$out.log"
}

job runs/signs/full_s0        roadsense.train.train_signs
job runs/detector             roadsense.train.train_detector

if [ ! -f /e/roadsense-env/cache/rscd_day_test.npz ]; then
  while ! tail -1 /e/roadsense-env/rscd_download.log | grep -q "^EXIT="; do sleep 30; done
  "$PY" -m roadsense.data.prepare rscd
  "$PY" -m roadsense.data.prepare rscd_day
fi
# day-disjoint protocol (honest; used for the release model and all ablations)
job runs/surface/effnet_multi roadsense.train.train_surface --arch effnet_b0 --head multi
job runs/surface/cnn_multi    roadsense.train.train_surface --arch cnn --head multi --epochs 40
job runs/surface/effnet_flat  roadsense.train.train_surface --arch effnet_b0 --head flat
# RSCD's official frame-level split, same recipe: comparability with the literature + leakage gap
job runs/surface/effnet_multi_official roadsense.train.train_surface --arch effnet_b0 --head multi --protocol official

job runs/signs/ablate_nostn   roadsense.train.train_signs --no-stn
job runs/signs/ablate_nose    roadsense.train.train_signs --no-se
job runs/signs/ablate_augbasic roadsense.train.train_signs --aug basic
job runs/signs/ablate_augnone roadsense.train.train_signs --aug none
job runs/signs/full_s1        roadsense.train.train_signs --seed 1
job runs/signs/full_s2        roadsense.train.train_signs --seed 2
echo "QUEUE DONE $(date +%T)"
