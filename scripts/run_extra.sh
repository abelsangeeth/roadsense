#!/usr/bin/env bash
# Higher-resolution surface models (192x288 = 0.8x native RSCD). Needs:
#   python -m roadsense.data.prepare rscd_day 192x288
set -u
cd "$(dirname "$0")/.."
PY=${PY:-/e/roadsense-env/venv/Scripts/python}

job() {
  local out=$1; shift
  if [ -f "$out/metrics.json" ]; then echo "skip $out"; return; fi
  mkdir -p "$(dirname "$out")"
  echo "=== $(date +%T) $out"
  "$PY" -u -m "$@" --out "$out" > "$out.log" 2>&1
  echo "EXIT=$? $out" | tee -a "$out.log"
}

# strongest variant first: ConvNeXt-Tiny at 192x288. bs 64 peaks at 4.6 GB, so nothing else may
# hold VRAM (stop the dashboard); bs 128 pages to host memory and stalls at "100% util".
job runs/surface/convnext_multi_192 roadsense.train.train_surface --arch convnext_tiny --head multi --hw 192x288 --bs 64
# resolution ablation: same backbone as the 128x192 baseline
job runs/surface/effnet_multi_192   roadsense.train.train_surface --arch effnet_b0 --head multi --hw 192x288 --bs 64
echo "EXTRA DONE $(date +%T)"
