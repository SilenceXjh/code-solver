"""
Solver Agent

职责：接受题目 + 策略，生成可执行的 Python 代码。
LCB 格式：从 stdin 读输入，结果写到 stdout。
"""

import re

from code_solver.data.lcb_loader import Problem
from code_solver.llm.base import LLMClient

_SOLVER_SYSTEM = """\
You are an expert competitive programmer. Your task is to implement a Python solution \
for a programming problem following a given algorithm strategy.
"""

_SOLVER_USER = """\
### Problem
{problem}

### Strategy to implement
{strategy}

### Format
{format_str}

Write the complete Python solution now:
```python
"""

class SolverAgent:
    """代码生成 Agent"""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def construct_generate_prompt(self, problem: Problem, strategy: str) -> str:
        from code_solver.data.lcb_loader import FORMATTING_MESSAGE_WITH_STARTER_CODE, FORMATTING_WITHOUT_STARTER_CODE

        prompt = f"### Question:\n{problem.description}\n\n"

        prompt += f"### Strategy:\n{strategy}\n\n"
        
        if problem.starter_code:
            prompt += (
                f"### Format: {FORMATTING_MESSAGE_WITH_STARTER_CODE}\n"
            )
            prompt += f"```python\n{problem.starter_code}\n```\n\n"
        else:
            prompt += f"### Format: {FORMATTING_WITHOUT_STARTER_CODE}\n"
            prompt += "```python\n# YOUR CODE HERE\n```\n\n"
        
        return prompt

    def generate(self, problem: Problem, strategy: str) -> str:
        """
        根据题目和策略生成代码。

        Args:
            problem  : 题目
            strategy : Thinker 给出的算法策略

        Returns:
            清理后的 Python 代码字符串
        """

        user_prompt = self.construct_generate_prompt(problem, strategy)
        # print("generate prompt:", user_prompt)
        raw = self.llm.chat_simple(
            system=_SOLVER_SYSTEM,
            user=user_prompt,
            temperature=0.8,    # 代码生成需要多样性以覆盖不同实现
        )
        return self._extract_code(raw)

    def _extract_code(self, raw: str) -> str:
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