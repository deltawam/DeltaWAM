#!/usr/bin/env bash
set -euo pipefail

python_bin="${PYTHON_BIN:-python}"
exec "${python_bin}" main.py fit \
  -c configs/release/actiondit_libero90_pretrain.yaml \
  "$@"
