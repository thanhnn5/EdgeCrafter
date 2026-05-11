#!/usr/bin/env bash
# Training launcher for ECDet.
#
# AMP and ModelEMA are enabled via YAML config (use_amp: True, use_ema: True,
# decay 0.9999) so no extra CLI flags are needed for them.
#
# Logging defaults to Weights & Biases. The first invocation will prompt for
# `wandb login` if no auth token is present. To use TensorBoard instead, set
# WANDB_DISABLED=true (or WANDB_MODE=disabled).
#
# Usage:
#   bash tools/train.sh                                   # default config, auto GPUs, wandb
#   CONFIG=configs/ecdet/ecdet_s.yml bash tools/train.sh  # override config
#   NUM_GPUS=2 bash tools/train.sh                        # force GPU count
#   RESUME=outputs/ecdet_cnxt_t/last.pth bash tools/train.sh
#   TUNING=outputs/ecdet_cnxt_t/best.pth bash tools/train.sh
#   SEED=42 bash tools/train.sh
#   EXTRA_ARGS="--test-only" bash tools/train.sh
#   WANDB_PROJECT=my-proj WANDB_NAME=run42 bash tools/train.sh
#   WANDB_DISABLED=true bash tools/train.sh               # opt back to TensorBoard
#
# Any additional positional arguments are passed through to train.py.

set -euo pipefail

# Resolve repo paths (script lives in ecdetseg/tools/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ECDETSEG_DIR="$(dirname "$SCRIPT_DIR")"

CONFIG="${CONFIG:-configs/ecdet/ecdet_cnxt_t.yml}"
SEED="${SEED:-0}"
PORT="${PORT:-29500}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

# Wandb defaults. Override by exporting before invoking, or set WANDB_DISABLED=true
# to fall back to TensorBoard.
export WANDB_PROJECT="${WANDB_PROJECT:-edgecrafter}"

# Auto-detect visible GPUs unless NUM_GPUS is set explicitly.
if [[ -z "${NUM_GPUS:-}" ]]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
            NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)
        else
            NUM_GPUS=$(nvidia-smi --list-gpus | wc -l | tr -d ' ')
        fi
    else
        NUM_GPUS=1
    fi
fi

# Optional resume / tuning checkpoints.
CKPT_ARGS=()
if [[ -n "${RESUME:-}" ]]; then
    CKPT_ARGS+=("-r" "$RESUME")
fi
if [[ -n "${TUNING:-}" ]]; then
    CKPT_ARGS+=("-t" "$TUNING")
fi

cd "$ECDETSEG_DIR"

echo "===================================================================="
echo "  ECDet training"
echo "  config:    $CONFIG"
echo "  gpus:      $NUM_GPUS"
echo "  seed:      $SEED"
[[ -n "${RESUME:-}" ]] && echo "  resume:    $RESUME"
[[ -n "${TUNING:-}" ]] && echo "  tuning:    $TUNING"
[[ -n "$EXTRA_ARGS" ]] && echo "  extra:     $EXTRA_ARGS"
echo "  AMP/EMA:   enabled via YAML (use_amp, use_ema)"
if [[ "${WANDB_DISABLED:-}" == "true" || "${WANDB_MODE:-}" == "disabled" ]]; then
    echo "  logging:   TensorBoard (wandb disabled via env)"
else
    echo "  logging:   wandb (project=$WANDB_PROJECT)"
fi
echo "===================================================================="

if [[ "$NUM_GPUS" -le 1 ]]; then
    # Single-process training (single GPU or CPU).
    exec python train.py \
        -c "$CONFIG" \
        --seed "$SEED" \
        "${CKPT_ARGS[@]}" \
        $EXTRA_ARGS \
        "$@"
else
    # Multi-GPU via torchrun. Each rank gets one local GPU.
    exec torchrun \
        --nproc_per_node="$NUM_GPUS" \
        --master_port="$PORT" \
        train.py \
        -c "$CONFIG" \
        --seed "$SEED" \
        "${CKPT_ARGS[@]}" \
        $EXTRA_ARGS \
        "$@"
fi
