"""
Debugger Agent

输入：原始代码 + 执行反馈 + FaultReport（来自 FaultLocalizer）+ Reflection（来自 Thinker）
输出：修复后的代码

相比 CodeTree 原版的改进：
  Debugger 收到的不只是笼统的"执行失败"，而是：
  - FaultLocalizer 定位的具体错误位置和差异描述
  - Thinker 基于错误类型生成的修复方向（reflection）
  这使得 Debugger 可以做有针对性的局部修复，而非整体重写。
"""

import re
from llm.base import LLMClient
from tree.node import FaultReport

_DEBUGGER_SYSTEM = """\
You are an expert Python debugger and competitive programmer.
You will be given a buggy solution and detailed diagnostic information.
Your task is to fix the bug and return a corrected, complete Python solution.

Rules:
- Output ONLY the fixed Python code, no explanation
- The solution must read input from stdin and print to stdout
- Preserve the overall algorithm strategy unless it is fundamentally flawed
- Make the minimal change necessary to fix the identified bug
"""

_DEBUGGER_USER = """\
### Problem
{problem}

### Algorithm Strategy
{strategy}

### Buggy Code
```python
{code}
```

### Execution Feedback
{exec_feedback}

### Fault Analysis
{fault_info}

### Reflection (what to fix)
{reflection}

Output the fixed Python code:
```python
"""


class DebuggerAgent:
    """代码修复 Agent"""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def fix(
        self,
        problem: str,
        strategy: str,
        code: str,
        exec_feedback: str,
        fault_report: FaultReport | None = None,
        reflection: str = "",
    ) -> str:
        """
        修复代码。

        Args:
            problem      : 题目描述
            strategy     : 该节点的算法策略
            code         : 待修复的代码
            exec_feedback: 执行器格式化的错误信息
            fault_report : FaultLocalizer 的定位报告（可选）
            reflection   : Thinker 生成的修复方向（可选）

        Returns:
            修复后的 Python 代码
        """
        fault_info = (
            fault_report.format_for_prompt()
            if fault_report and fault_report.found
            else "No additional fault localization available."
        )
        reflection_text = reflection or "Focus on fixing the identified bug."

        user_prompt = _DEBUGGER_USER.format(
            problem=problem,
            strategy=strategy,
            code=code,
            exec_feedback=exec_feedback,
            fault_info=fault_info,
            reflection=reflection_text,
        )
        raw = self.llm.chat_simple(
            system=_DEBUGGER_SYSTEM,
            user=user_prompt,
            temperature=0.4,    # 修复时需要一定保守性，但也要有创造力
        )
        return self._extract_code(raw)

    def _extract_code(self, raw: str) -> str:
        raw = raw.strip()
        match = re.search(r"```(?:python)?\n(.*?)```", raw, re.DOTALL)
        if match:
            return match.group(1).strip()
        raw = re.sub(r"^```(?:python)?\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
        return raw.strip()