 CUDA_VISIBLE_DEVICES=5,6 accelerate launch train.py \
    --sample_size 512 \
    --target_modules "q_proj,k_proj,v_proj,o_proj" \
    --max_completion_length 512 \
    --num_generations 2 \
    --num_machines 1 \
    --machine_rank 0 \
    --use_deepspeed \
    --deepspeed_config configs/dapo-grpo-deepspeed.json