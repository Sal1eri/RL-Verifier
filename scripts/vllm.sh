CUDA_VISIBLE_DEVICES=0,1 uv run vllm serve Qwen/Qwen3-4B-Base \
  --tensor-parallel-size 2 \
  --port 8000 \
  --gpu-memory-utilization 0.8 \
  --max-model-len 8192 \
  --dtype auto \
  --task generate \
  --reasoning-parser deepseek_r1

