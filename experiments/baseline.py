from openai import OpenAI
from common import QUERY_TEMPLATE
from common import load_aime
from common import ANSWER_PATTERN_BOXED
from verifier import GeneralVerifier
import re
import datetime
from tqdm import tqdm
import json
import os

model_name = "Qwen/Qwen3-4B-Base"
OPENAI_SYSTEM_MESSAGE_API = "You are a helpful assistant."

print("Dataset loaded.")
print("QUERY_TEMPLATE:", QUERY_TEMPLATE)

client = OpenAI(
    base_url="http://0.0.0.0:8000/v1",
    api_key="test",
    
)


def eval_aimebaseline(ds,split='train', model_name=model_name,verifier=None):
    results = []
    acc = 0.0
    ver_acc = 0.0
    for data in tqdm(ds):
        question = data["problem"]
        query = QUERY_TEMPLATE.replace("<question>", question)  
        messages = [
            {"role": "system", "content": OPENAI_SYSTEM_MESSAGE_API},
            {"role": "user", "content": query},
        ]
        chat_response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=4096,
            temperature=0.7,
        )
        resp = chat_response.choices[0].message.content
        match = re.search(ANSWER_PATTERN_BOXED, resp)
        extracted_answer = match.group(1) if match else None
        
        verifier_judge = verifier.verify(
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
            "verifier-based-score": verifier_judge["decision"],
            "verifier-judgement":verifier_judge['raw_output'],
        })
        acc += 1.0 if extracted_answer == data["answer"] else 0.0
        ver_acc += 1.0 if verifier_judge["decision"] else 0.0

    results.append({
        "system_prompt": OPENAI_SYSTEM_MESSAGE_API,
        "query_template": QUERY_TEMPLATE,
        "model_name": model_name,
        "timestamp": datetime.datetime.now().isoformat(),
        "num_samples": len(ds),
        "correct_samples": acc,
        "verifier_correct_samples": ver_acc,
        "mode": split,
        "acc": acc / len(ds),
        "verifier_acc": ver_acc / len(ds)

    })
    return results

if __name__ == "__main__":
    
    verifier = GeneralVerifier()
    split = "test"
    ds = load_aime(split=split)
    output_root = "outputs"
    results = eval_aimebaseline(ds, split=split, model_name=model_name, verifier=verifier)
    file_name = os.path.basename(__file__).replace('.py', '')
    sub_path = os.path.join(output_root, file_name)
    os.makedirs(sub_path, exist_ok=True)
    with open(os.path.join(sub_path, f"{model_name.replace('/', '_')}_{split}.json"), "w") as f:
        json.dump(results, f, indent=4)
