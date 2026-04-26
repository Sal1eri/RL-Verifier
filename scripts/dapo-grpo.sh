#!/usr/bin/env bash

set -euo pipefail

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
uv run accelerate launch train_grpo.py \
    --dataset-path "${DATASET_PATH:-./mydata/dapo_math.jsonl}" \
    --reward-func "${REWARD_FUNC:-accuracy}" \
    --output-dir "${OUTPUT_DIR:-./outputs/dapo_grpo_rule_4b}" \
    --model-name "${MODEL_NAME:-Qwen/Qwen3-4B-Base}" \
    --per-device-train-batch-size "${PER_DEVICE_TRAIN_BATCH_SIZE:-1}" \
    --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS:-16}" \
    --num-generations "${NUM_GENERATIONS:-8}" \
    --max-completion-length "${MAX_COMPLETION_LENGTH:-4096}" \
    --learning-rate "${LEARNING_RATE:-5e-7}" \
    --beta "${BETA:-0.001}" \
    --temperature "${TEMPERATURE:-0.7}" \
    --precision "${PRECISION:-bf16}" \
    --max-steps "${MAX_STEPS:-1000}" \
    --save-steps "${SAVE_STEPS:-100}" \
    --logging-steps "${LOGGING_STEPS:-1}" \
    --report-to "${REPORT_TO:-wandb}" \
    --run-name "${RUN_NAME:-qwen3-4b-rule-2gpu}" \
    --target-modules "${TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj}" \
    "$@"
