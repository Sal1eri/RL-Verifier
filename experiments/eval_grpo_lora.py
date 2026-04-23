from common import QUERY_TEMPLATE
from common import load_aime
from common import ANSWER_PATTERN_BOXED
from verifier import GeneralVerifier,verify_reward_from_vllm

import re
import datetime
from tqdm import tqdm
import json
import os
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


model_name = "Qwen/Qwen3-4B-Base"
lora_path = "./lora_grpo_out/checkpoint-36"   # 改成 None 表示不用 LoRA；或者填你的 LoRA 路径
OPENAI_SYSTEM_MESSAGE_API = "You are a helpful assistant."

print("QUERY_TEMPLATE:", QUERY_TEMPLATE)


def load_model_and_tokenizer(model_name, lora_path=None):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    if lora_path is not None:
        print(f"Loading LoRA from: {lora_path}")
        model = PeftModel.from_pretrained(base_model, lora_path)
    else:
        print("Using base model only")
        model = base_model

    model.eval()
    return tokenizer, model


tokenizer, model = load_model_and_tokenizer(model_name, lora_path=lora_path)


def generate_response(messages, max_tokens=4096, temperature=0.7):
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(text, return_tensors="pt")

    model_device = next(model.parameters()).device
    inputs = {k: v.to(model_device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=(temperature > 0),
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    resp = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return resp.strip()


def eval_aime_grpo_lora(ds, split="train", model_name=model_name, verifier=None, lora_path=None):
    results = []
    acc = 0.0
    ver_acc = 0.0

    model_tag = model_name if lora_path is None else f"{model_name}+LoRA:{lora_path}"

    for data in tqdm(ds):
        question = data["problem"]
        query = QUERY_TEMPLATE.replace("<question>", question)

        messages = [
            {"role": "system", "content": OPENAI_SYSTEM_MESSAGE_API},
            {"role": "user", "content": query},
        ]

        resp = generate_response(
            messages=messages,
            max_tokens=4096,
            temperature=0.7,
        )

        match = re.search(ANSWER_PATTERN_BOXED, resp)
        extracted_answer = match.group(1) if match else None

        verifier_based_score,verifier_judgement = verify_reward_from_vllm(
            question=question,
            ground_truth=data["answer"],
            student_answer=extracted_answer,
        )

        results.append({
            "query": query,
            "response": resp,
            "extracted_answer": extracted_answer,
            "ground_truth": data["answer"],
            "rule-based-score": 1.0 if extracted_answer == data["answer"] else 0.0,
            "verifier-based-score": verifier_based_score,
            "verifier-judgement": verifier_judgement,
        })

        acc += 1.0 if extracted_answer == data["answer"] else 0.0
        ver_acc += 1.0 if verifier_based_score else 0.0

    results.append({
        "system_prompt": OPENAI_SYSTEM_MESSAGE_API,
        "query_template": QUERY_TEMPLATE,
        "model_name": model_tag,
        "timestamp": datetime.datetime.now().isoformat(),
        "num_samples": len(ds),
        "correct_samples": acc,
        "verifier_correct_samples": ver_acc,
        "mode": split,
        "acc": acc / len(ds),
        "verifier_acc": ver_acc / len(ds),
    })

    return results


if __name__ == "__main__":
    split = "test"
    dataset_name='aime_2025'
    ds = load_aime(split=split,data_dir='./'+dataset_name)
    output_root = "outputs"
    results = eval_aime_grpo_lora(
        ds,
        split=split,
        model_name=model_name,
        lora_path=lora_path,
    )
    file_name = os.path.basename(__file__).replace(".py", "")
    sub_path = os.path.join(output_root, file_name)
    os.makedirs(sub_path, exist_ok=True)

    model_file_tag = model_name.replace("/", "_")
    if lora_path is not None:
        lora_tag = os.path.basename(os.path.normpath(lora_path))
        model_file_tag = f"{model_file_tag}_lora_{lora_tag}"

    save_path = os.path.join(sub_path, f"{model_file_tag}_{split}_{dataset_name}.json")

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"Saved results to: {save_path}")