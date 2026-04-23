# RL-Verifier

An experimental repository for math reasoning with two kinds of reward signals:

- Rule-based reward: extract the final answer from `\boxed{}` and compare it directly with the ground-truth answer.
- Verifier-based reward: call or load `TIGER-Lab/general-verifier` to judge whether the student answer is equivalent to the reference answer.

The current codebase focuses on AIME-style evaluation, Qwen base models, GRPO/LoRA training, and verifier-based scoring.

## What This Repository Includes

- Baseline evaluation on local AIME datasets
- Answer equivalence checking with `general-verifier`
- LoRA training with `trl`'s `GRPOTrainer`
- Evaluation scripts for trained LoRA checkpoints
- A local vLLM service exposed through an OpenAI-compatible API

## Repository Structure

```text
.
├── main.py                         # Placeholder entry point; currently only prints hello
├── train.py                        # GRPO + accuracy reward training entry (DAPO-Math-17k subset)
├── train_grpo.py                   # GRPO + verifier reward training entry (DeepMath-103K)
├── experiments/
│   ├── common.py                   # AIME loading, answer extraction, shared prompts
│   ├── baseline.py                 # AIME 2024 baseline evaluation
│   ├── aime25_baseline.py          # AIME 2025 baseline evaluation
│   ├── verifier.py                 # Local general-verifier loading and vLLM wrapper
│   ├── reward_func.py              # Verifier reward used during training
│   ├── eval_grpo_lora.py           # LoRA checkpoint evaluation
│   ├── eval_grpo_lora_ver.py       # Verifier-reward LoRA checkpoint evaluation
│   └── sampler.py                  # OpenAI-style chat sampler
├── scripts/
│   ├── vllm.sh                     # Starts the local vLLM service
│   ├── dapo-grpo.sh                # Example accelerate launch script
│   ├── rule-based-grpo.sh          # Currently empty
│   └── verifier-based-grpo.sh      # Currently empty
├── aime_2024/                      # Local AIME 2024 data
├── aime_2025/                      # Local AIME 2025 data
├── outputs/                        # Evaluation outputs
├── dapo_math.jsonl                 # Additional dataset file
├── pyproject.toml                  # Dependency definition
└── uv.lock                         # uv lockfile
```

## Requirements

- Python >= 3.10
- CUDA environment
- Linux is recommended
- Multiple GPUs are recommended for training and serving
- `uv` for dependency management

The repository depends on several heavy packages, including:

- `torch==2.6`
- `transformers==4.57.3`
- `trl>=0.29.0`
- `vllm==0.8.5.post1`
- `flash-attn`
- `flashinfer-python`
- `peft`
- `openai`

Install dependencies with:

```bash
uv sync
```

## Datasets

### Local datasets

- `aime_2024/train.jsonl`
- `aime_2024/test.jsonl`
- `aime_2024/chat.jsonl`
- `aime_2025/train.jsonl`
- `aime_2025/test.jsonl`

`experiments/common.py` provides two helpers:

- `load_aime(data_dir, split)` for `train` / `test`
- `load_aime_chat_format(data_dir, split="chat")` for chat-style samples

### Remote datasets

The training scripts also use Hugging Face datasets directly:

- `BytedTsinghua-SIA/DAPO-Math-17k`
- `trl-lib/DeepMath-103K`

You need Hugging Face access the first time you run those scripts.

## Core Idea

### 1. Rule-based scoring

`experiments/common.py` defines answer extraction logic that first tries to find the last `\boxed{...}` in the model response.  
The baseline scripts compare the extracted answer against the dataset `answer` field to produce `rule-based-score`.

### 2. Verifier-based scoring

Both `experiments/verifier.py` and `experiments/reward_func.py` construct a verifier prompt with:

- the original question
- the ground-truth answer
- the student answer
- an instruction to check equivalence without re-solving the problem
- a final output in the form `Final Decision: Yes` or `Final Decision: No`

There are currently two ways to use the verifier:

- Load `TIGER-Lab/general-verifier` locally
- Call a verifier model through the OpenAI-compatible vLLM endpoint

## Quick Start

### 1. Install dependencies

```bash
uv sync
```

### 2. Start the vLLM service

Use the provided script:

```bash
bash scripts/vllm.sh
```

At the moment, the script is equivalent to:

```bash
CUDA_VISIBLE_DEVICES=0,1 uv run vllm serve Qwen/Qwen3-4B-Base \
  --tensor-parallel-size 2 \
  --port 8000 \
  --gpu-memory-utilization 0.8 \
  --max-model-len 8192 \
  --dtype auto \
  --task generate \
  --reasoning-parser deepseek_r1
```

Default API endpoint:

```text
http://localhost:8000/v1
```

### 3. Run baseline evaluation

AIME 2024:

```bash
uv run python experiments/baseline.py
```

AIME 2025:

```bash
uv run python experiments/aime25_baseline.py
```

Outputs are written under:

- `outputs/baseline/`
- `outputs/aime25_baseline/`

### 4. Run training

Rule-based reward training entry:

```bash
uv run accelerate launch train.py
```

Verifier-based reward training entry:

```bash
uv run accelerate launch train_grpo.py
```

You can also use the example shell script:

```bash
bash scripts/dapo-grpo.sh
```

Notes:

- `scripts/rule-based-grpo.sh` is currently empty
- `scripts/verifier-based-grpo.sh` is currently empty
- Check `CUDA_VISIBLE_DEVICES` before multi-GPU runs
- `train.py` and `train_grpo.py` currently hardcode the model name, dataset, and some hyperparameters

### 5. Evaluate a LoRA checkpoint

Standard LoRA evaluation:

```bash
uv run python experiments/eval_grpo_lora.py
```

Verifier-reward LoRA evaluation:

```bash
uv run python experiments/eval_grpo_lora_ver.py
```

These scripts currently:

- load `Qwen/Qwen3-4B-Base`
- attach a specified LoRA checkpoint
- generate answers on AIME samples
- score the outputs again with the verifier
- save results under `outputs/eval_grpo_lora/` or `outputs/eval_grpo_lora_ver/`

Before running them, you will usually need to update the variables at the top of each script:

- `model_name`
- `lora_path`
- `split`
- `dataset_name`

## Main Scripts

### `train.py`

Current default behavior:

- Dataset: `BytedTsinghua-SIA/DAPO-Math-17k`
- Training slice: `train[:2000]`
- Reward: `trl.rewards.accuracy_reward`
- Model: `Qwen/Qwen3-4B-Base`
- Output directory: `./DAPO_GRPO_LORA`

### `train_grpo.py`

Current default behavior:

- Dataset: `trl-lib/DeepMath-103K`
- Reward: `experiments.reward_func.verifier_reward`
- Model: `Qwen/Qwen3-4B-Base`
- Output directory: `./test`
- `max_steps=1000`

### `experiments/baseline.py`

Flow:

1. Load local `aime_2024`
2. Generate model responses through the OpenAI-compatible endpoint
3. Extract the answer from `\boxed{}`
4. Run `GeneralVerifier` for an additional correctness judgment
5. Save both rule-based and verifier-based scores

### `experiments/aime25_baseline.py`

Same overall flow as above, but using `aime_2025`.

### `experiments/reward_func.py`

This is the reward function used by `GRPOTrainer`.  
It extracts answers from completions, calls `TIGER-Lab/general-verifier` through the OpenAI-compatible endpoint, and appends logs to `verifier_log.txt`.

### `experiments/verifier.py`

Provides two verifier paths:

- `GeneralVerifier`: loads the Hugging Face model locally and runs `generate`
- `verify_reward_from_vllm(...)`: calls the verifier through `http://localhost:8000/v1`

## Outputs

Most outputs are stored under `outputs/`:

- `outputs/baseline/`
- `outputs/aime25_baseline/`
- `outputs/eval_grpo_lora/`
- `outputs/eval_grpo_lora_ver/`

Training logs or verifier debug logs may also be written to:

- `verifier_log.txt`

## Current Limitations

This is still an experiment repository, and several parts are intentionally rough:

- `main.py` is not a real project entry point yet
- several scripts hardcode model names, LoRA paths, and dataset paths
- `scripts/rule-based-grpo.sh` and `scripts/verifier-based-grpo.sh` are empty
- baseline scripts depend on a working local or remote verifier path
- training defaults assume substantial GPU memory
- there is no proper `tests/` package yet; validation is mostly script-based

## Suggested Workflow

If you want the shortest path to reproducing the current setup:

1. Run `uv sync`
2. Start `bash scripts/vllm.sh`
3. Run `uv run python experiments/baseline.py`
4. Adjust `train.py` or `train_grpo.py` for your hardware and target experiment
5. Train
6. Run `experiments/eval_grpo_lora.py` or `experiments/eval_grpo_lora_ver.py`

## Possible Improvements

- Add CLI arguments instead of editing variables inside the scripts
- Document `accelerate` and `deepspeed` configs more clearly
- Unify experiment naming for rule-based and verifier-based runs
- Add minimal tests
- Align the README further with the final intended training setup
