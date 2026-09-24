#!/usr/bin/env bash
set -euo pipefail

suite="${1:?usage: scripts/finetune_suite.sh spatial|object|goal|libero_10 [extra Lightning args]}"
shift
python_bin="${PYTHON_BIN:-python}"

case "${suite}" in
  spatial|object|goal)
    config="configs/release/actiondit_libero_${suite}.yaml"
    exec "${python_bin}" main.py fit \
      -c "${config}" \
      --ckpt_path checkpoints/actiondit/actiondit_libero90_pretrained.ckpt \
      "$@"
    ;;
  libero_10)
    exec "${python_bin}" main.py fit \
      -c configs/release/actiondit_libero_10.yaml \
      "$@"
    ;;
  *)
    echo "unsupported suite: ${suite}" >&2
    exit 2
    ;;
esac
