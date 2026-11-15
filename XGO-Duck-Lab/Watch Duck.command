#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ ! -x .venv/bin/mjpython ]]; then ./setup.sh; fi
exec .venv/bin/mjpython lab.py view --policy runs/stock-49k/policy.onnx --foot v2-stock --surface carpet --start sit "$@"
