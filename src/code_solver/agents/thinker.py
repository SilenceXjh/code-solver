"""
Thinker Agent（方向 D：多样性约束策略生成）

原始 CodeTree 的 Thinker：
  自回归依次生成策略 S_i | S_{1:i-1}
  问题：后生成的策略受前面影响，容易同质化（DP 的变体再变体）

本实现的改进：
  1. 每次生成策略时，把「已探索范式列表」注入 prompt，
     明确要求新策略使用不同的算法视角
  2. 单独用一次 LLM call 提取策略的算法范式标签（供 SearchTree 去重）
  3. 生成 Reflection：对失败节点生成修复方向，供 Debugger 参考
"""

import json
import re
from dataclasses import dataclass

from code_solver.llm.base import LLMClient
from code_solver.tree.node import FaultReport

# ── 已知的算法范式标签 ────────────────────────────────────────────────────────

PARADIGM_LABELS = [
    "dynamic_programming", "greedy", "binary_search", "bfs_dfs",
    "two_pointers", "sliding_window", "divide_and_conquer", "backtracking",
    "math", "simulation", "hash_map", "sorting", "prefix_sum",
    "monotonic_stack", "union_find", "heap", "trie", "segment_tree",
    "bit_manipulation", "number_theory",
]

# ── Prompt 模板 ────────────────────────────────────────────────────────────────

_STRATEGY_SYSTEM = """\
You are an expert competitive programmer. Your task is to propose \
a solution strategy for the given problem before writing code.

A solution strategy is a concise but concrete plan (3–8 sentences) that:
- Identifies the key idea needed to solve the problem
- Explains how to transform the problem into a solvable form
- Describes the main steps or structure of the solution
- Mentions key data structures or techniques if needed

Focus on how to arrive at the solution, not just describing it.
Do NOT write code or pseudocode.
"""

_STRATEGY_USER = """\
### Problem
{problem}

### Task
Propose ONE solution strategy to solve this problem.

{diversity_hint}

Your strategy must:
- Be clearly different from the already-explored approaches listed above
- Avoid generic templates without concrete details
- Specify the core algorithm paradigm (e.g., DP with memoization, greedy by X, BFS on Y)

Respond with ONLY the strategy text, no headers or bullet points.
"""

_STRATEGY_USER_NO_HISTORY = """\
### Problem
{problem}

### Task
Propose ONE solution strategy to solve this problem.

### Requirements:
- The approach must be specific to this problem
- Avoid generic templates without concrete details
- Specify the core algorithm paradigm (e.g., DP with memoization, greedy by X, BFS on Y)

Respond with ONLY the strategy text, no headers or bullet points.
"""

_ADDITIONAL_STRATEGY_REQUIREMENT ="""
{diversity_hint}

Your strategy must:
- Be clearly different from the already-explored approaches listed above
- Specify the core algorithm paradigm (e.g., DP with memoization, greedy by X, BFS on Y)
- Mention the key data structures if relevant
"""

_DIVERSITY_HINT_TEMPLATE = """\
### Already-Explored Approaches (DO NOT repeat these paradigms)
{explored}

Choose a fundamentally different algorithmic angle from the above.
"""

_PARADIGM_SYSTEM = """\
You are an algorithm classifier. Given a strategy description, output the single \
best-matching paradigm label from the provided list.
Respond with ONLY the label string, nothing else.
"""

_PARADIGM_USER = """\
Strategy: {strategy}

Available labels: {labels}

Output the single best label:
"""

_REFLECTION_SYSTEM = """\
You are an expert code reviewer. Given a failed code solution and its execution \
feedback, generate a concise reflection that pinpoints the root cause and \
specifies exactly what to fix.
"""

_REFLECTION_USER = """\
### Problem
{problem}

### Strategy
{strategy}

### Failed Code
```python
{code}
```

### Execution Feedback
{exec_feedback}

### Task
Write a focused reflection (3-6 sentences) that:
1. Identifies the root cause of the failure
2. Specifies exactly what needs to be changed
3. Does NOT rewrite the code — only describe the fix

Reflection:
"""

# ── Thinker Agent ──────────────────────────────────────────────────────────────

@dataclass
class StrategyResult:
    strategy: str
    algo_paradigm: str


class ThinkerAgent:
    """
    策略生成 Agent

    职责：
      1. generate_strategy()    → 生成一个新策略（含多样性约束）
      2. generate_reflection()  → 对失败节点生成修复方向
    """

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def generate_strategy(
        self,
        problem: str,
        explored_paradigms: list[str],
    ) -> StrategyResult:
        """
        生成一个与已探索范式不重复的新策略。

        Args:
            problem      : 格式化后的题目描述
            explored_paradigms: 已探索的范式标签列表（来自 SearchTree）

        Returns:
            StrategyResult，包含策略文本和范式标签
        """
        if len(explored_paradigms) == 0:
            user_prompt = _STRATEGY_USER_NO_HISTORY.format(
                problem=problem,
            )
        else:
            diversity_hint = self._build_diversity_hint(explored_paradigms)
            user_prompt = _STRATEGY_USER.format(
                problem=problem,
                diversity_hint=diversity_hint,
            )
        strategy = self.llm.chat_simple(
            system=_STRATEGY_SYSTEM,
            user=user_prompt,
            temperature=0.2,    # 策略生成需要一定多样性
        ).strip()

        paradigm = self._classify_paradigm(strategy)
        return StrategyResult(strategy=strategy, algo_paradigm=paradigm)

    def generate_reflection(
        self,
        problem: str,
        strategy: str,
        code: str,
        exec_feedback: str,
        fault_report: FaultReport | None = None,
    ) -> str:
        """
        对失败的代码生成修复方向（reflection），供 Debugger 使用。

        Args:
            problem      : 题目描述
            strategy     : 该节点的算法策略
            code         : 失败的代码
            exec_feedback: 执行器格式化的错误信息
            fault_report : FaultLocalizer 的精准定位报告（可选）

        Returns:
            reflection 文本
        """
        fault_info = ""
        if fault_report and fault_report.found:
            fault_info = fault_report.format_for_prompt()

        user_prompt = _REFLECTION_USER.format(
            problem=problem,
            strategy=strategy,
            code=code,
            exec_feedback=exec_feedback
        )
        # print("reflection input message:", user_prompt)
        return self.llm.chat_simple(
            system=_REFLECTION_SYSTEM,
            user=user_prompt,
            temperature=0.3,
        ).strip()

    # ── 内部方法 ────────────────────────────────────────────────────────────────

    def _build_diversity_hint(self, explored_paradigms: list[str]) -> str:
        """构建多样性约束提示"""
        if not explored_paradigms:
            return ""
        explored_str = "\n".join(f"  - {p}" for p in explored_paradigms)
        return _DIVERSITY_HINT_TEMPLATE.format(explored=explored_str)

    def _classify_paradigm(self, strategy: str) -> str:
        """
        用 LLM 将策略文本分类为标准范式标签。
        如果分类失败或返回非法标签，回退到 'simulation'。
        """
        labels_str = ", ".join(PARADIGM_LABELS)
        raw = self.llm.chat_simple(
            system=_PARADIGM_SYSTEM,
            user=_PARADIGM_USER.format(strategy=strategy, labels=labels_str),
            temperature=0.0,
        ).strip().lower()

        # 清理可能的标点
        raw = re.sub(r"[^a-z_]", "", raw)
        if raw in PARADIGM_LABELS:
            return raw

        # 模糊匹配：包含已知标签作为子串
        for label in PARADIGM_LABELS:
            if label in raw or raw in label:
                return label

        return "simulation"     # 保守回退