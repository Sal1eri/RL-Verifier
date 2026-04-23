# Repository Guidelines

## Project Structure & Module Organization
Core training entry points live at the repository root: `train.py`, `train_grpo.py`, and `main.py`. Reusable experiment logic is under `experiments/`, including reward functions, verifier utilities, dataset helpers, and evaluation scripts such as `experiments/baseline.py` and `experiments/eval_grpo_lora_ver.py`. Dataset snapshots are stored in `aime_2024/`, `aime_2025/`, and `dapo_math.jsonl`. Shell launchers for common workflows live in `scripts/`. Generated artifacts and evaluation outputs should stay in `outputs/` rather than being mixed into source directories.

## Build, Test, and Development Commands
Use `uv` for environment management and dependency resolution:

```bash
uv sync
uv run python main.py
uv run python experiments/baseline.py
bash scripts/vllm.sh
bash scripts/verifier-based-grpo.sh
```

`uv sync` installs locked dependencies from `pyproject.toml` and `uv.lock`. `uv run python experiments/baseline.py` runs baseline evaluation against the local AIME data. `scripts/vllm.sh` starts the local OpenAI-compatible vLLM server used by verifier-based evaluation. Training scripts use `accelerate launch`; review GPU IDs and model paths before running.

## Coding Style & Naming Conventions
Follow existing Python style: 4-space indentation, `snake_case` for functions and variables, and descriptive module names in lowercase. Keep experiment-specific logic inside `experiments/` instead of expanding root-level scripts. Prefer small, explicit helper functions over inline prompt-building or parsing logic repeated across files. There is no enforced formatter in the repo yet, so keep imports tidy and match the surrounding file’s style.

## Testing Guidelines
There is no dedicated `tests/` package yet. For now, validate changes with the smallest runnable target that covers them, such as `uv run python experiments/baseline.py` or a focused script under `experiments/`. When adding tests, use `test_*.py` naming under a new `tests/` directory and keep fixtures small because model-backed checks are expensive. Document any required local services, especially the vLLM verifier endpoint on `http://localhost:8000/v1`.

## Commit & Pull Request Guidelines
Recent commits use short, imperative messages such as `update verifier baseline` and `update dependency`. Keep commit subjects concise, lowercase, and focused on one change. Pull requests should include: the training or evaluation path touched, required environment changes, sample commands used for verification, and any output locations written under `outputs/`. If a script depends on missing local config, say so explicitly in the PR description.
