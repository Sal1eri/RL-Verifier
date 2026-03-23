# train_grpo.py
from datasets import load_dataset
from trl import GRPOTrainer
from trl.rewards import accuracy_reward

dataset = load_dataset("trl-lib/DeepMath-103K", split="train[:10]")


model_name = "Qwen/Qwen2-0.5B-Instruct"
model_name = "Qwen/Qwen3-4B-Base"
trainer = GRPOTrainer(
    model=model_name,
    reward_funcs=accuracy_reward,
    train_dataset=dataset,
)
trainer.train()