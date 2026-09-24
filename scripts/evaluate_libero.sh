#!/usr/bin/env bash
set -euo pipefail

suite="${1:?usage: scripts/evaluate_libero.sh spatial|object|goal|libero_10 [extra evaluator args]}"
shift
python_bin="${PYTHON_BIN:-python}"

case "${suite}" in
  spatial|object|goal)
    benchmark_suite="libero_${suite}"
    config="configs/release/actiondit_libero_${suite}.yaml"
    checkpoint="checkpoints/actiondit/actiondit_libero_${suite}.ckpt"
    ;;
  libero_10)
    benchmark_suite="libero_10"
    config="configs/release/actiondit_libero_10.yaml"
    checkpoint="checkpoints/actiondit/actiondit_libero_10.ckpt"
    ;;
  *)
    echo "unsupported suite: ${suite}" >&2
    exit 2
    ;;
esac

exec "${python_bin}" scripts/eval_feature_action_libero.py \
  --config "${config}" \
  --checkpoint "${checkpoint}" \
  --suite "${benchmark_suite}" \
  --num-trials 50 \
  --replan-steps 8 \
  --temporal-ensemble \
  --temporal-ensemble-decay 0.1 \
  --num-inference-steps 10 \
  "$@"
