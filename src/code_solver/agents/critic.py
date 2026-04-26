"""
Critic Agent（方向 A 核心集成点）

CodeTree 原版 Critic 的两个职责：
  1. Score：对代码质量打分（0-10）
  2. Solution Verification：判断是否能通过 hidden tests（纯 LLM True/False）

本实现的改进：
  Solution Verification 不再依赖 LLM 猜测，改为：
    1. 调用 AdversarialTester 生成边界测试并实际执行
    2. 用执行结果（而非 LLM 判断）决定 Accept / Refine / Abort
  
决策逻辑：
  - public tests 全过 + 对抗测试全过 → ACCEPT
  - public tests 全过 + 对抗测试有失败 → REFINE（带反例信息）
  - public tests 未全过 + score < threshold → ABORT（策略无效）
  - public tests 未全过 + score >= threshold → REFINE（继续调试）
"""

from dataclasses import dataclass
from enum import Enum

from execution.executor import SuiteResult
from llm.base import LLMClient
from modules.adversarial_tester import AdversarialTester, AdversarialResult
from execution.executor import TestCase
from tree.node import NodeStatus

_SCORE_SYSTEM = """\
You are an expert code reviewer for competitive programming.
Score the given solution from 0 to 10 based on:
- Correctness: does it handle edge cases?
- Algorithm quality: is the approach sound?
- Implementation quality: is the code clean and correct?

Respond with ONLY a JSON object: {"score": <int 0-10>, "reason": "<one sentence>"}
"""

_SCORE_USER = """\
### Problem
{problem}

### Strategy
{strategy}

### Code
```python
{code}
```

### Execution Result
{exec_feedback}

Score this solution (0-10):
"""


class CriticDecision(Enum):
    ACCEPT = "accept"
    REFINE = "refine"
    ABORT  = "abort"


@dataclass
class CriticResult:
    decision: CriticDecision
    score: float
    reason: str
    adversarial: AdversarialResult | None = None

    def format_refine_feedback(self) -> str:
        """格式化给 Debugger 看的反馈"""
        parts = [f"Score: {self.score:.1f}/10. {self.reason}"]
        if self.adversarial and not self.adversarial.all_passed:
            from modules.adversarial_tester import AdversarialTester
            parts.append(AdversarialTester(None, None).format_failure_for_prompt(self.adversarial))
        return "\n".join(parts)


class CriticAgent:
    """
    评分 + 验证 Agent

    abort_threshold: score 低于此值时直接 Abort 策略（不再调试）
    """

    def __init__(
        self,
        llm: LLMClient,
        adversarial_tester: AdversarialTester,
        abort_threshold: float = 3.0,
    ):
        self.llm = llm
        self.adversarial_tester = adversarial_tester
        self.abort_threshold = abort_threshold

    def evaluate(
        self,
        problem: str,
        strategy: str,
        code: str,
        suite_result: SuiteResult,
        public_tests: list[TestCase],
    ) -> CriticResult:
        """
        对一个节点进行评分并做出决策。

        Args:
            problem      : 题目描述
            strategy     : 该节点的算法策略
            code         : 节点的代码
            suite_result : public tests 的执行结果
            public_tests : public 测试用例（AdversarialTester 用）

        Returns:
            CriticResult，包含决策、分数、原因
        """
        # Step 1：打分
        score, reason = self._score(problem, strategy, code, suite_result)

        # Step 2：根据 public test 结果分路
        if not suite_result.all_passed:
            # public tests 没过
            if score < self.abort_threshold:
                return CriticResult(
                    decision=CriticDecision.ABORT,
                    score=score,
                    reason=f"{reason} Score too low to continue debugging.",
                )
            return CriticResult(
                decision=CriticDecision.REFINE,
                score=score,
                reason=reason,
            )

        # Step 3：public tests 全过 → 对抗测试验证（方向A核心）
        adv_result = self.adversarial_tester.test(problem, code, public_tests)

        if adv_result.all_passed:
            return CriticResult(
                decision=CriticDecision.ACCEPT,
                score=score,
                reason=f"{reason} Passed all adversarial tests.",
                adversarial=adv_result,
            )
        else:
            # 对抗测试暴露了 bug，带反例信息继续 Refine
            return CriticResult(
                decision=CriticDecision.REFINE,
                score=score,
                reason=f"{reason} Failed adversarial test — hidden bug detected.",
                adversarial=adv_result,
            )

    def verify_by_llm(
        self,
        problem: str,
        strategy: str,
        code: str,
    ) -> tuple[bool, str]:
        """
        消融模式用：让 LLM 判断代码是否能通过 hidden tests。

        与 _score 的区别：
          _score 是对代码质量的综合打分（用于 Refine/Abort 决策）
          verify_by_llm 是专门针对"是否有边界/逻辑漏洞"的 Yes/No 判断
          （用于替代对抗测试做 Solution Verification）

        Returns:
            (looks_correct: bool, reason: str)
        """
        import json, re
        system = """\
You are an expert competitive programmer reviewing a solution that passed all visible test cases.
Your task: identify if the solution is likely to fail on hidden test cases due to edge cases or logical bugs.
Respond ONLY with JSON: {"looks_correct": true/false, "reason": "<one sentence>"}
"""
        user = f"""\
### Problem
{problem}

### Strategy
{strategy}

### Code (passed all visible tests)
```python
{code}
```

Does this solution correctly handle all edge cases and constraints?
Carefully check: boundary values, overflow, empty inputs, off-by-one errors, and special cases.
"""
        raw = self.llm.chat_simple(system=system, user=user, temperature=0.0, json_mode=True)
        try:
            raw = re.sub(r"^```(?:json)?\n?", "", raw.strip())
            raw = re.sub(r"\n?```$", "", raw).strip()
            data = json.loads(raw)
            looks_correct = bool(data.get("looks_correct", True))
            reason = str(data.get("reason", ""))
            if not looks_correct:
                print("critic think not right even pass public tests:", raw)
            return looks_correct, reason
        except Exception:
            return True, "Could not parse LLM verification response."

    def _score(
        self,
        problem: str,
        strategy: str,
        code: str,
        suite_result: SuiteResult,
    ) -> tuple[float, str]:
        """调用 LLM 打分，返回 (score, reason)"""
        import json, re
        exec_feedback = suite_result.format_for_prompt()
        user_prompt = _SCORE_USER.format(
            problem=problem,
            strategy=strategy,
            code=code,
            exec_feedback=exec_feedback,
        )
        raw = self.llm.chat_simple(
            system=_SCORE_SYSTEM,
            user=user_prompt,
            temperature=0.0,
            json_mode=True,
        )
        # 解析 JSON
        try:
            raw = re.sub(r"^```(?:json)?\n?", "", raw.strip())
            raw = re.sub(r"\n?```$", "", raw).strip()
            data = json.loads(raw)
            score  = float(data.get("score", 5))
            reason = str(data.get("reason", ""))
            score  = max(0.0, min(10.0, score))
            return score, reason
        except Exception:
            # 解析失败：根据执行结果给一个保守分数
            base = 6.0 if suite_result.all_passed else 3.0
            return base, "Could not parse score — using heuristic."