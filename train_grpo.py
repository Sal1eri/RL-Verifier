# train_grpo.py
from trl import GRPOTrainer
from trl.rewards import accuracy_reward
from experiments.common import load_aime_chat_format
dataset = load_aime_chat_format()
print(dataset[0])

model_name = "Qwen/Qwen2-0.5B-Instruct"

# model_name = "Qwen/Qwen3-4B-Base"
trainer = GRPOTrainer(
    model=model_name,
    reward_funcs=accuracy_reward,
    train_dataset=dataset,
)
trainer.train()