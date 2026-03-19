import os
import re


ANSWER_PATTERN_MULTICHOICE = r"(?i)Answer[ \t]*:[ \t]*\$?([A-D])\$?"
ANSWER_PATTERN = r"(?i)Answer\s*:\s*([^\n]+)"
ANSWER_PATTERN_BOXED = r"(?i)\\boxed\s*{([^\n]+)}"


QUERY_TEMPLATE = """
<question>

Please reason step by step, and put your final answer within \\boxed{}.
""".strip()

def normalize_response(response: str) -> str:
    """
    Normalize the response by removing markdown and LaTeX formatting that may prevent a match.
    """

    return (
        response.replace("**", "")
        .replace("$\\boxed{", "")
        .replace("}$", "")
        .replace("\\$", "")
        .replace("$\\text{", "")
        .replace("$", "")
        .replace("\\mathrm{", "")
        .replace("\\{", "")
        .replace("\\text", "")
        .replace("\\(", "")
        .replace("\\mathbf{", "")
        .replace("{", "")
        .replace("\\boxed", "")
    )

def normalize_extracted_answer(extracted_answer: str) -> str:
    return (
        # In arabic these are the letters used for A-D in multiple choice questions
        extracted_answer.replace("أ", " A")
        .replace("ب", " B")
        .replace("ج", " C")
        .replace("د", " D")
        # In Bengali these are the letters used for A-D in multiple choice questions
        .replace("অ", " A")
        .replace("ব", " B")
        .replace("ড", " C")
        .replace("ঢ", " D")
        # In Japanese these are the letters sometimes used for A-D in multiple choice questions
        .replace("Ａ", " A")
        .replace("Ｂ", " B")
        .replace("Ｃ", " C")
        .replace("Ｄ", " D")
        .strip()
    )

def extract_last_boxed(text: str) -> str:
    """
    Extract the last occurrence of a boxed answer from the input text.
    
    Returns:
        The content inside the last \boxed{...} or None if not found.
    """
    pattern = r"\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}"
    matches = list(re.finditer(pattern, text))
    if matches:
        return matches[-1].group(1)
    return None


def extract_last_final_answer(text: str) -> str:
    """
    Try to extract the final answer from the text using several candidate patterns.
    
    Returns:
        The extracted answer as a string, or None if none of the patterns match.
    """
    candidate_patterns = [
        r"Final Answer:\s*((?:[^<]|<[^<])*?)\n",
        r"Final Answer is:\s*((?:[^<]|<[^<])*?)\n",
        r"The answer is:\s*((?:[^<]|<[^<])*?)\n",
        r"Answer:\s*((?:[^<]|<[^<])*?)\n",
        r"Solution:\s*((?:[^<]|<[^<])*?)\n",
        r"The solution is:\s*((?:[^<]|<[^<])*?)\n",
    ]
    
    last_match = None
    last_position = -1
    for pattern in candidate_patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            if match.start() > last_position:
                last_position = match.start()
                last_match = match.group(1).strip()

    stop_words = ["</s>", "<|im_end|>", "<|endoftext|>"]
    for stop_word in stop_words:
        if last_match and last_match.endswith(stop_word):
            last_match = last_match[:-len(stop_word)].strip()
    
    return last_match


def extract_solution(solution_str: str) -> str:
    boxed_answer = extract_last_boxed(solution_str)
    if boxed_answer:
        return boxed_answer
    return extract_last_final_answer(solution_str)
# 1. 模拟模型输出
raw_output = "Therefore, the final result is $\\boxed{\\mathbf{42}}$."

# 2. 提取步骤 (使用正则)
match = re.search(ANSWER_PATTERN_BOXED, raw_output)
extracted = match.group(1)  # 提取到的是 "\mathbf{42}"

# 3. 清洗步骤 (使用 normalize_response)
final_answer = normalize_response(extracted)

print(f"原始提取: {extracted}")
print(f"清洗后: {final_answer}")