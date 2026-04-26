"""
AdversarialTester（方向 A：对抗性测试生成）

解决的核心问题：
  LCB 的 public tests 只有 2-3 个示例，通过这些不代表正确。
  研究表明 medium 题有 20%、hard 题有 40% 的解通过 public 但过不了 private。

本模块在 Critic 的 Solution Verification 阶段介入：
  当代码通过所有 public tests 后，不是直接让 LLM 猜 True/False，
  而是：
  1. 让 LLM 生成针对边界情况的额外测试用例
  2. 实际执行这些测试，用执行结果（而非 LLM 猜测）做验证信号

测试用例生成策略（按题目类型）：
  - 边界值：空输入、单元素、最大值、最小值
  - 对称性：全相同元素、排好序的输入、逆序输入
  - 极端情况：题目描述中隐含的约束边界
"""

import json
import re
from dataclasses import dataclass, field

from execution.executor import Executor, SuiteResult, TestCase
from llm.base import LLMClient

_GEN_SYSTEM = """\
You are an expert at finding bugs in competitive programming solutions through \
adversarial test case design.

Given a problem, a solution, and the public test cases, generate additional test \
cases that are likely to expose bugs the public tests miss.

Focus on:
1. Boundary values (empty, single element, max/min constraints)
2. Edge cases implied by the problem description
3. Cases where the solution's algorithm might fail (e.g., off-by-one, overflow)
4. Inputs that stress-test the algorithm's correctness (not just speed)

Respond ONLY with a valid JSON array of test cases.
"""

_GEN_USER = """\
### Problem
{problem}

### Solution to test
```python
{code}
```

### Public test cases (already passing)
{public_tests}

### Task
Generate {n} additional test cases that might expose bugs this solution has.
Each test case should have an "input" and the correct "output".

IMPORTANT: The output must be the CORRECT expected output, not what the buggy solution produces.
Compute the correct output yourself based on the problem statement.

Respond with ONLY a JSON array (no markdown):
[
  {{"input": "...", "output": "..."}},
  ...
]
"""


@dataclass
class AdversarialResult:
    """对抗测试的完整结果"""
    generated_tests: list[TestCase]     # 生成的测试用例
    suite_result: SuiteResult           # 执行结果
    all_passed: bool                    # 是否全部通过
    first_failure_input: str = ""       # 第一个失败用例的输入（供 Debugger 参考）
    first_failure_expected: str = ""
    first_failure_actual: str = ""


class AdversarialTester:
    """
    对抗性测试生成器

    用法：
        tester = AdversarialTester(llm, executor)
        result = tester.test(problem, code, public_tests)
    """

    def __init__(
        self,
        llm: LLMClient,
        executor: Executor,
        num_tests: int = 5,
    ):
        self.llm = llm
        self.executor = executor
        self.num_tests = num_tests

    def test(
        self,
        problem: str,
        code: str,
        public_tests: list[TestCase],
    ) -> AdversarialResult:
        # 从 public_tests 推断题目类型
        testtype = public_tests[0].testtype if public_tests else "stdin"
        generated = self._generate_tests(problem, code, public_tests, testtype)

        if not generated:
            return AdversarialResult(
                generated_tests=[],
                suite_result=SuiteResult(results=[], passed=0, total=0),
                all_passed=True,
            )

        suite = self.executor.run_suite(code, generated, stop_on_first_failure=False)

        first_fail_input = first_fail_exp = first_fail_act = ""
        for result, tc in zip(suite.results, generated):
            if not result.passed:
                first_fail_input    = tc.input
                first_fail_exp      = result.expected
                first_fail_act      = result.actual
                break

        return AdversarialResult(
            generated_tests=generated,
            suite_result=suite,
            all_passed=suite.all_passed,
            first_failure_input=first_fail_input,
            first_failure_expected=first_fail_exp,
            first_failure_actual=first_fail_act,
        )

    def format_failure_for_prompt(self, result: AdversarialResult) -> str:
        """将对抗测试的失败信息格式化为 Debugger prompt"""
        if result.all_passed:
            return f"✅ Passed all {result.suite_result.total} adversarial tests."
        passed = result.suite_result.passed
        total  = result.suite_result.total
        return (
            f"❌ Failed adversarial tests ({passed}/{total} passed).\n"
            f"  Counterexample found:\n"
            f"  Input   : {result.first_failure_input!r}\n"
            f"  Expected: {result.first_failure_expected!r}\n"
            f"  Actual  : {result.first_failure_actual!r}\n"
            f"  This reveals a bug not caught by the public test cases."
        )

    # ── 内部方法 ─────────────────────────────────────────────────────────────

    def _generate_tests(
        self,
        problem: str,
        code: str,
        public_tests: list[TestCase],
        testtype: str = "stdin",
    ) -> list[TestCase]:
        """调用 LLM 生成测试用例，解析 JSON 输出"""
        public_str = "\n".join(
            f"  Input: {tc.input!r}  →  Output: {tc.output!r}"
            for tc in public_tests
        ) or "  (none)"

        user_prompt = _GEN_USER.format(
            problem=problem,
            code=code,
            public_tests=public_str,
            n=self.num_tests,
        )
        raw = self.llm.chat_simple(
            system=_GEN_SYSTEM,
            user=user_prompt,
            temperature=0.7,
        )
        return self._parse_tests(raw, testtype)

    def _parse_tests(self, raw: str, testtype: str = "stdin") -> list[TestCase]:
        """解析 LLM 返回的 JSON 测试用例列表"""
        raw = raw.strip()
        raw = re.sub(r"^```(?:json)?\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw).strip()

        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if not match:
            return []

        try:
            data = json.loads(match.group())
        except json.JSONDecodeError:
            return []

        tests = []
        for item in data:
            if isinstance(item, dict) and "input" in item and "output" in item:
                tests.append(TestCase(
                    input=str(item["input"]),
                    output=str(item["output"]),
                    testtype=testtype,      # 继承原题 testtype
                    is_public=False,
                ))
        return tests