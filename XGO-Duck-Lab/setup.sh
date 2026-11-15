#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON_BIN="${XGO_PYTHON:-python3.12}"
"$PYTHON_BIN" -c 'import sys; assert sys.version_info[:2] == (3,12), "Use CPython 3.12"'
"$PYTHON_BIN" -m venv .venv
if [[ "$(uname -s)" == Darwin && "$(uname -m)" == x86_64 ]]; then
  .venv/bin/python -m pip install -r requirements-intel-mac.lock
else
  .venv/bin/python -m pip install -r requirements.txt
fi
.venv/bin/python -m pip check
.venv/bin/python verify_lab.py
