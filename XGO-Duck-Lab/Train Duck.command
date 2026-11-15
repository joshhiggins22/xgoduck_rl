#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ ! -x .venv/bin/python ]]; then ./setup.sh; fi
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
run_dir="runs/stock-$(date +%Y%m%d-%H%M%S)-$$"
.venv/bin/python -u lab.py train --workers 2 --steps 49152 --batch 128 --lr .00006 --log-std -1.8 --freeze 1024 --seed 127 --foot v2-stock --surface mixed --out "$run_dir"
.venv/bin/python lab.py eval --policy "$run_dir/policy.onnx" --foot v2-stock --surface carpet --start sit --out "$run_dir/carpet-rise.json"
echo "Saved policy and evaluation in $run_dir"
