"""
DifficultyAssessor（方向 C：难度感知自适应预算分配）

在 Thinker 生成策略之前，对题目做一次评估：
  - 难度：easy / medium / hard
  - 算法范式：推测需要哪类算法
  - 预算：根据难度自适应分配 (width, depth)

预算映射（来自 CodeTree 论文 Table 3 消融实验最优配置）：
  Easy   → width=2, depth=3  （少策略，快速收敛）
  Medium → width=3, depth=3  （平衡）
  Hard   → width=5, depth=2  （多策略探索为主，BFS >> DFS）
"""

import json
import re
from dataclasses import dataclass

from llm.base import LLMClient

PARADIGM_LABELS = [
    "dynamic_programming", "greedy", "binary_search", "bfs_dfs",
    "two_pointers", "sliding_window", "divide_and_conquer", "backtracking",
    "math", "simulation", "hash_map", "sorting", "prefix_sum",
    "monotonic_stack", "union_find", "heap", "trie", "segment_tree",
]

# 预算基础配置
_BUDGET = {
    "easy":   {"width": 2, "depth": 3},
    "medium": {"width": 3, "depth": 3},
    "hard":   {"width": 5, "depth": 2},
}

_SYSTEM = """\
You are an algorithm difficulty assessor for competitive programming problems.
Analyze the given problem and output a JSON assessment.
Respond with ONLY a valid JSON object, no markdown.
"""

_USER = """\
Analyze this competitive programming problem:

{problem}

Output a JSON object with:
- "difficulty": "easy", "medium", or "hard"
- "algo_paradigms": list of likely paradigms (from: {labels})
- "reasoning": one sentence explanation

JSON:
"""


@dataclass
class Assessment:
    difficulty: str             # easy / medium / hard
    algo_paradigms: list[str]
    reasoning: str
    width: int                  # recommended Thinker strategy count
    depth: int                  # recommended Debugger max rounds


class DifficultyAssessor:
    """
    难度评估器

    用法：
        assessor = DifficultyAssessor(llm)
        assessment = assessor.assess(problem.format_for_prompt())
    """

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def assess(self, problem: str) -> Assessment:
        raw = self.llm.chat_simple(
            system=_SYSTEM,
            user=_USER.format(problem=problem, labels=", ".join(PARADIGM_LABELS)),
            temperature=0.0,
            json_mode=True,
        )
        return self._parse(raw)

    def _parse(self, raw: str) -> Assessment:
        raw = re.sub(r"^```(?:json)?\n?", "", raw.strip())
        raw = re.sub(r"\n?```$", "", raw).strip()

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return self._fallback()

        difficulty = str(data.get("difficulty", "medium")).lower()
        if difficulty not in _BUDGET:
            difficulty = "medium"

        paradigms = [
            p for p in data.get("algo_paradigms", [])
            if p in PARADIGM_LABELS
        ] or ["simulation"]

        reasoning = str(data.get("reasoning", ""))
        budget = _BUDGET[difficulty].copy()

        # 微调：backtracking / divide_and_conquer 有多种写法 → +1 width
        if any(p in paradigms for p in ("backtracking", "divide_and_conquer")):
            budget["width"] = min(budget["width"] + 1, 5)
        # dp 容易陷入局部最优 → +1 depth（上限 4）
        if "dynamic_programming" in paradigms:
            budget["depth"] = min(budget["depth"] + 1, 4)

        return Assessment(
            difficulty=difficulty,
            algo_paradigms=paradigms,
            reasoning=reasoning,
            width=budget["width"],
            depth=budget["depth"],
        )

    def _fallback(self) -> Assessment:
        b = _BUDGET["medium"]
        return Assessment(
            difficulty="medium",
            algo_paradigms=["simulation"],
            reasoning="Fallback: parse error, using medium defaults.",
            width=b["width"],
            depth=b["depth"],
        )