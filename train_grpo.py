# train_grpo.py
from peft import LoraConfig
from trl import GRPOTrainer, GRPOConfig
from trl.rewards import accuracy_reward
from experiments.reward_func import verifier_reward
from experiments.common import load_aime_chat_format
dataset = load_aime_chat_format()

# model_name = "Qwen/Qwen2-0.5B-Instruct"

model_name = "Qwen/Qwen3-4B-Base"

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)

training_args = GRPOConfig(
    output_dir="./lora_grpo_out_ver",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_generations=8,
    max_completion_length=2048,
    fp16=True,
    gradient_checkpointing=True,
    temperature=0.7,
    learning_rate=5e-7,
    beta=0.001
)

trainer = GRPOTrainer(
    model=model_name,
    args=training_args,
    # reward_funcs=accuracy_reward,
    reward_funcs = verifier_reward,
    train_dataset=dataset,
    peft_config=peft_config,
)

trainer.train()