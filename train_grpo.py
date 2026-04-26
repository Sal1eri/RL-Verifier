import argparse

from datasets import load_dataset
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer
from trl.rewards import accuracy_reward

from experiments.reward_func import verifier_reward


REWARD_FUNC_MAP = {
    "accuracy": accuracy_reward,
    "verifier": verifier_reward,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train GRPO with configurable dataset and reward function.")

    parser.add_argument("--model-name", default="Qwen/Qwen3-4B-Base")
    parser.add_argument("--output-dir", default="./outputs/dapo_grpo")
    parser.add_argument("--reward-func", choices=sorted(REWARD_FUNC_MAP.keys()), default="verifier")

    parser.add_argument("--dataset-path", default="./mydata/dapo_math.jsonl")
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--solution-field", default="solution")
    parser.add_argument("--max-samples", type=int, default=None)

    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--num-generations", type=int, default=8)
    parser.add_argument("--max-completion-length", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--learning-rate", type=float, default=5e-7)
    parser.add_argument("--beta", type=float, default=0.001)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report-to", choices=["none", "wandb"], default="none")
    parser.add_argument("--run-name", default=None)

    parser.add_argument("--precision", choices=["fp16", "bf16", "none"], default="fp16")
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--target-modules", default="q_proj,k_proj,v_proj,o_proj")

    return parser.parse_args()


def resolve_dataset(args):
    if args.dataset_name:
        dataset = load_dataset(args.dataset_name, split=args.dataset_split)
    else:
        dataset = load_dataset("json", data_files=args.dataset_path, split=args.dataset_split)

    if args.max_samples is not None:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    solution_field = args.solution_field
    if solution_field not in dataset.column_names:
        if "reward_model" in dataset.column_names:
            dataset = dataset.map(
                lambda example: {**example, solution_field: example["reward_model"]["ground_truth"]}
            )
        else:
            raise ValueError(
                f"Dataset is missing solution field '{solution_field}', and no reward_model.ground_truth fallback is available."
            )

    required_columns = {"prompt", solution_field}
    missing_columns = required_columns - set(dataset.column_names)
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing_columns)}")

    if solution_field != "solution":
        dataset = dataset.rename_column(solution_field, "solution")

    return dataset


def build_peft_config(args):
    target_modules = [module.strip() for module in args.target_modules.split(",") if module.strip()]
    if not target_modules:
        raise ValueError("target_modules cannot be empty")

    return LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )


def build_training_args(args):
    use_fp16 = args.precision == "fp16"
    use_bf16 = args.precision == "bf16"
    report_to = [] if args.report_to == "none" else [args.report_to]

    return GRPOConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        fp16=use_fp16,
        bf16=use_bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        temperature=args.temperature,
        learning_rate=args.learning_rate,
        beta=args.beta,
        max_steps=args.max_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        save_total_limit=args.save_total_limit,
        seed=args.seed,
        report_to=report_to,
        run_name=args.run_name,
    )


def main():
    args = parse_args()

    dataset = resolve_dataset(args)
    peft_config = build_peft_config(args)
    training_args = build_training_args(args)
    reward_func = REWARD_FUNC_MAP[args.reward_func]

    print(
        "Training config:",
        {
            "model_name": args.model_name,
            "output_dir": args.output_dir,
            "reward_func": args.reward_func,
            "dataset_name": args.dataset_name,
            "dataset_path": args.dataset_path,
            "dataset_split": args.dataset_split,
            "num_samples": len(dataset),
            "target_modules": peft_config.target_modules,
            "report_to": args.report_to,
            "run_name": args.run_name,
        },
    )

    trainer = GRPOTrainer(
        model=args.model_name,
        args=training_args,
        reward_funcs=reward_func,
        train_dataset=dataset,
        peft_config=peft_config,
    )

    trainer.train()


if __name__ == "__main__":
    main()
