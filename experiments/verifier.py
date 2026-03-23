from typing import List, Dict, Optional, Union

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class GeneralVerifier:
    VERIFIER_PASS_TAG = "Final Decision: Yes"

    def __init__(
        self,
        model_path: str = "TIGER-Lab/general-verifier",
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.float16,
    ):
        self.model_path = model_path
        self.device = device
        self.torch_dtype = torch_dtype

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path,
                                                       flash_attn_implementation="flash_attention_2")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=self.torch_dtype,
        ).to(self.device)
        self.model.eval()

    @staticmethod
    def build_prompt(question: str, ground_truth: str, student_answer: str) -> str:
        return (
            f"User: ### Question: {question}\n\n"
            f"### Ground Truth Answer: {ground_truth}\n\n"
            f"### Student Answer: {student_answer}\n\n"
            "For the above question, please verify if the student's answer is equivalent to the ground truth answer.\n"
            "Do not solve the question by yourself; just check if the student's answer is equivalent to the ground truth answer.\n"
            "If the student's answer is correct, output \"Final Decision: Yes\". If the student's answer is incorrect, output \"Final Decision: No\". Assistant:"
        )

    def parse_decision(self, text: str) -> bool:
        return self.VERIFIER_PASS_TAG in text

    @torch.no_grad()
    def verify(
        self,
        question: str,
        ground_truth: str,
        student_answer: str,
        max_new_tokens: int = 1024,
    ) -> Dict[str, Union[str, bool]]:
        prompt = self.build_prompt(question, ground_truth, student_answer)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        decoded = self.tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
        decision = self.parse_decision(decoded)

        return {
            "raw_output": decoded,
            "decision": decision,
        }

    @torch.no_grad()
    def verify_batch(
        self,
        items: List[Dict[str, str]],
        max_new_tokens: int = 1024,
        batch_size: int = 8,
    ) -> List[Dict[str, Union[str, bool]]]:
        results = []

        for start in range(0, len(items), batch_size):
            batch_items = items[start:start + batch_size]
            prompts = [
                self.build_prompt(
                    x["question"],
                    x["ground_truth"],
                    x["student_answer"],
                )
                for x in batch_items
            ]

            inputs = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(self.device)

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

            decoded_batch = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

            for item, decoded in zip(batch_items, decoded_batch):
                results.append({
                    "question": item["question"],
                    "ground_truth": item["ground_truth"],
                    "student_answer": item["student_answer"],
                    "raw_output": decoded,
                    "decision": self.parse_decision(decoded),
                })

        return results