#!/usr/bin/env bash
set -euo pipefail

python_bin="${PYTHON_BIN:-python}"
exec "${python_bin}" main.py fit \
  -c configs/release/deltaworld_libero_all.yaml \
  "$@"
