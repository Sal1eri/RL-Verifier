#!/usr/bin/env bash

set -euo pipefail

WANDB_PROJECT="${WANDB_PROJECT:-rl-verifier}" \
WANDB_GROUP="${WANDB_GROUP:-rule-baseline}" \
WANDB_NAME="${WANDB_NAME:-qwen3-4b-rule-v100x4-max-4096}" \
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" \
uv run accelerate launch --num_processes 4 train_grpo.py \
    --dataset-path ./mydata/deep_math.jsonl \
    --reward-func accuracy \
    --output-dir ./outputs/qwen3_4b_rule_v100x4_max_4096 \
    --model-name Qwen/Qwen3-4B-Base \
    --per-device-train-batch-size 1 \
    --gradient-accumulation-steps 32 \
    --num-generations 4 \
    --max-completion-length 768 \
    --learning-rate 5e-7 \
    --beta 0.001 \
    --temperature 0.7 \
    --precision fp16 \
    --max-steps 50 \
    --save-steps 20 \
    --logging-steps 1 \
    --report-to wandb \
    --run-name qwen3-4b-rule-v100x4-deepmath-max-4096 \
    --target-modules q_proj,k_proj,v_proj,o_proj \
    "$@"
