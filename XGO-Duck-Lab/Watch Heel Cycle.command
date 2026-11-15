#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ ! -x .venv/bin/mjpython ]]; then ./setup.sh; fi
exec .venv/bin/mjpython view_cycle.py "$@"
