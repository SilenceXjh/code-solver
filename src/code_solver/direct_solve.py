"""
直接生成问题的解
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path
import re

from code_solver.execution.executor import Executor

# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s %(levelname)s %(message)s",
#     datefmt="%H:%M:%S",
# )
log = logging.getLogger(__name__)


_SYSTEM_PROMPT = """\
You are an expert competitive programmer. Your task is to implement a Python solution \
for a programming problem.
"""

FORMATTING_MESSAGE_WITH_STARTER_CODE = "You will use the following starter code to write the solution to the problem and enclose your code within delimiters."

FORMATTING_WITHOUT_STARTER_CODE = "Read the inputs from stdin, solve the problem, and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows. Ensure that when the python program runs, it reads the inputs, runs the algorithm and writes output to STDOUT."

def lcb_official_prompt(question_content, starter_code):
    prompt = f"### Question:\n{question_content}\n\n"

    if starter_code:
        prompt += (
            f"### Format: {FORMATTING_MESSAGE_WITH_STARTER_CODE}\n"
        )
        prompt += f"```python\n{starter_code}\n```\n\n"
    else:
        prompt += f"### Format: {FORMATTING_WITHOUT_STARTER_CODE}\n"
        prompt += "```python\n# YOUR CODE HERE\n```\n\n"
    
    prompt += f"Only output the solution code without any explanation."
    return prompt

def extract_code(raw: str) -> str:
    """
    从 LLM 输出中提取纯代码。
    处理三种常见格式：
        1. ```python\\n...\\n```
        2. ```\\n...\\n```
        3. 直接输出代码（无 markdown）
    """
    # 尝试提取 markdown 代码块
    pattern = r"```(?:python)?\n(.*?)```"
    match = re.search(pattern, raw, re.DOTALL)
    if match:
        return match.group(1).strip()

    # 去除可能的首尾 ``` 标记
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:python)?\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)

    return raw.strip()


def main():
    parser = argparse.ArgumentParser(description="CodeTree+ evaluation on LiveCodeBench")

    # LLM 配置
    parser.add_argument("--model",     default="gpt-4o-mini")
    parser.add_argument("--api-key",   default=None)
    parser.add_argument("--api-base",  default=None, help="vLLM endpoint URL")
    parser.add_argument("--mock",      action="store_true", help="Use mock LLM for local testing")

    # 数据集配置
    parser.add_argument("--lcb-version",   default="release_v6")
    parser.add_argument("--difficulty",    default=None, choices=["easy","medium","hard"])
    parser.add_argument("--max-problems",  type=int, default=None)
    parser.add_argument("--cache-dir",     default="./lcb_cache")

    # 执行配置
    parser.add_argument("--timeout",          type=int,   default=10)
    parser.add_argument("--num-adv-tests",    type=int,   default=5)
    parser.add_argument("--abort-threshold",  type=float, default=3.0)

    # 消融开关
    parser.add_argument("--no-difficulty",      action="store_true", help="Ablation: disable DifficultyAssessor")
    parser.add_argument("--no-diversity",       action="store_true", help="Ablation: disable diversity constraint")
    parser.add_argument("--no-fault-localizer", action="store_true", help="Ablation: disable FaultLocalizer")
    parser.add_argument("--no-llm-verifier",    action="store_true", help="Ablation: disable LLM verifier (public pass = direct accept)")
    parser.add_argument("--no-adversarial",     action="store_true", help="Ablation: disable AdversarialTester")

    # 输出配置
    parser.add_argument("--output",   default="./results")
    parser.add_argument("--run-name", default="codetree_plus")

    args = parser.parse_args()

    # ── 加载数据集 ────────────────────────────────────────────────────────────
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from code_solver.data.lcb_loader import LCBLoader
    from code_solver.evaluation.evaluator import Evaluator

    loader = LCBLoader(
        release_version=args.lcb_version,
        cache_dir=args.cache_dir,
        difficulty=args.difficulty,
        max_problems=args.max_problems,
    )
    problems = loader.load()
    if not problems:
        log.error("No problems loaded. Check --lcb-version and cache.")
        return

    from code_solver.llm.openai_client import OpenAIClient
    llm = OpenAIClient(
        model=args.model,
        api_key=args.api_key or os.environ.get("OPENAI_API_KEY"),
        api_base=args.api_base,
    )
    log.info(f"Using model: {args.model}")

    executor = Executor()

    # ── 主循环 ────────────────────────────────────────────────────────────────
    total = 0
    right = 0
    for i, problem in enumerate(problems):
        total += 1
        log.info(f"\n{'='*60}")
        log.info(f"Problem {i+1}/{len(problems)}: [{problem.problem_id}] {problem.title} ({problem.difficulty})")

        user_prompt = lcb_official_prompt(problem.description, problem.starter_code)

        response = llm.chat_simple(system=_SYSTEM_PROMPT, user=user_prompt, temperature=0.2)
        code = extract_code(response)

        suite_result = executor.run_suite(code, problem.private_tests)
        if suite_result.all_passed:
            right += 1

        res = {"problem_id": problem.problem_id, "passed": suite_result.all_passed, "pass_rate": suite_result.pass_rate}
        print(res)

    print("right/total:", f"{right}/{total}")



if __name__ == "__main__":
    main()