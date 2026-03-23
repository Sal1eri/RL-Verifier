import re
from openai import OpenAI
import datetime

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key='empty'
)

ANSWER_PATTERN_BOXED = r"(?i)\\boxed\s*{([^\n]+)}"
VERIFIER_PASS_TAG = "Final Decision: Yes"

def verifier_reward(completions: list[list[dict[str, str]]], solution: list[str],prompts: list[list[dict[str, str]]], **kwargs) -> list[float | None]:
    r"""
    Reward function that checks if the completion matches the ground truth.
        - If both gold and prediction are parseable → use math verification.
        - If gold is not parseable → return `None` to skip the example.

    Args:
        completions (`list[list[dict[str, str]]]`):
            List of completions to be evaluated. Each completion must be a list of one message, i.e. a dictionary
            containing the key `"content"` with the value being the text of the completion.
        solution: (`list[str]`):
            List of the raw-text solutions to the questions/problems/prompts.
        **kwargs:
            Additional keyword arguments. This function does not use them, but they are required in the function
            signature to ensure compatibility with trainers like [`GRPOTrainer`].
    Example:
    ```python
    >>> from trl.rewards import accuracy_reward

    >>> solutions = [r"\frac{1}{3}", r"\frac{1}{3}"]
    >>> completions = [
    ...     [{"role": "assistant", "content": r"My answer is \boxed{\frac{1}{3}}"}],
    ...     [{"role": "assistant", "content": r"My answer is \boxed{\frac{1}{2}}"}],
    ... ]
    >>> accuracy_reward(completions, solutions)
    [1.0, 0.0]
    ```
    """
    LOG_FILE = "verifier_log.txt"
    # print('Prompts:', prompts)
    # print('Prompts[0]:', prompts[0])
    contents = [completion[0]["content"] for completion in completions]
    rewards = []

    for content, sol, p in zip(contents, solution, prompts):

        # print('p:',p)
        # print('p[0]:',p[0])
        match = re.search(ANSWER_PATTERN_BOXED, content)
        extracted_answer = match.group(1) if match else None
        prompt = (
        f"User: ### Question: {p[0]['content']}\n\n"
        f"### Ground Truth Answer: {sol}\n\n"
        f"### Student Answer: {extracted_answer}\n\n"
        "For the above question, please verify if the student's answer is equivalent to the ground truth answer.\n"
        "Do not solve the question by yourself; just check if the student's answer is equivalent to the ground truth answer.\n"
        "If the student's answer is correct, output \"Final Decision: Yes\". If the student's answer is incorrect, output \"Final Decision: No\". Assistant:"
        )
        # print('Verifier prompt:', prompt)

        
        resp = client.completions.create(
            model="TIGER-Lab/general-verifier",
            prompt=prompt,
            temperature=0.0,
            max_tokens=1024,
        )

        text = resp.choices[0].text
        print(text)
                # === 写日志（追加模式）===
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write("\n" + "="*80 + "\n")
            f.write(f"Time: {datetime.datetime.now()}\n\n")
            f.write("PROMPT:\n")
            f.write(prompt + "\n\n")
            f.write("RESPONSE:\n")
            f.write(text + "\n")
        reward = 1.0 if VERIFIER_PASS_TAG in text else 0.0
        rewards.append(reward)

    return rewards

def demo():
    # ====== 构造一条和你数据结构一致的样本 ======
    sample = {
        "solution": "902",
        "prompt": [
            {
                "content": "There is a collection of $25$ indistinguishable white chips and $25$ indistinguishable black chips. Find the number of ways to place some of these chips in the $25$ unit cells of a $5\\times5$ grid such that:\n\neach cell contains at most one chip\nall chips in the same row and all chips in the same column have the same colour\nany additional chip placed on the grid would violate one or more of the previous two conditions.",
                "role": "user"
            }
        ]
    }

    # ====== 模拟 trainer 传进来的 batch ======
    solutions = [sample["solution"], sample["solution"]]
    prompts = [sample["prompt"], sample["prompt"]]

    # completion 也要是 trainer 那种格式：list[list[dict]]
    completions = [
        [
            {
                "role": "assistant",
                "content": "The answer is \\boxed{902}."
            }
        ]
        ,
        [
            {
                "role": "assistant",
                "content": "The answer is \\boxed{900}."
            }
        ]
    ]

    # ====== 调用测试 ======
    rewards = verifier_reward(
        completions=completions,
        solution=solutions,
        prompt=prompts
    )

    print(rewards)

if __name__ == "__main__":
    demo()