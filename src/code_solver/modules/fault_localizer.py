"""
FaultLocalizer（方向 B 简化版）

完整版思路（执行轨迹对比）因 Python -c 模式下 sys.settrace
的帧过滤行为复杂，留作后续改进。

当前实现：基于执行反馈的结构化错误分析，仍然比 CodeTree 原版
（只把原始 stderr 丢给 Debugger）提供更多有用信息：

1. 错误类型分类（syntax / runtime / wrong_answer / timeout）
2. 对 wrong_answer：分析期望 vs 实际输出的差异模式
3. 对 runtime_error：提取关键报错行和变量信息
4. 生成结构化的 FaultReport，供 Debugger 使用

未来改进方向：
  - 用独立脚本文件（非 -c 模式）执行，绕开 settrace 限制，
    实现真正的变量级分叉点定位
"""

import re
from execution.executor import ExecutionResult, SuiteResult, TestCase
from tree.node import FaultReport


class FaultLocalizer:
    """
    基于错误信息分析的故障定位器

    用法：
        localizer = FaultLocalizer()
        report = localizer.localize(code, suite_result, test_cases)
    """

    def localize(
        self,
        code: str,
        suite_result: SuiteResult,
        test_cases: list[TestCase],
    ) -> FaultReport:
        failure = suite_result.first_failure()
        if failure is None:
            return FaultReport(found=False, summary="All public tests passed.")

        if failure.error_type == "time_limit_exceeded":
            return self._handle_timeout(code)
        elif failure.error_type == "syntax_error":
            return self._handle_syntax_error(failure)
        elif failure.error_type in ("runtime_error", "name_error", "type_error",
                                    "index_error", "key_error", "attribute_error",
                                    "recursion_error"):
            return self._handle_runtime_error(failure, code)
        else:
            return self._handle_wrong_answer(failure, suite_result, test_cases)

    # ── 各错误类型处理 ──────────────────────────────────────────────────────

    def _handle_timeout(self, code: str) -> FaultReport:
        hints = []
        if re.search(r'for .+ in .+:\n.*for .+ in', code):
            hints.append("nested loops detected — may be O(n²)")
        if re.search(r'while True', code):
            hints.append("'while True' loop — check exit condition")
        if re.search(r'def \w+\(.*\).*:\n.*return \w+\(', code):
            hints.append("possible unbounded recursion")
        hint_str = "; ".join(hints) if hints else "consider a more efficient algorithm"
        return FaultReport(
            found=True,
            summary=f"Time Limit Exceeded. Likely cause: {hint_str}.",
        )

    def _handle_syntax_error(self, failure: ExecutionResult) -> FaultReport:
        line_no = self._extract_error_line(failure.stderr)
        err_msg = self._extract_last_error_line(failure.stderr)
        return FaultReport(
            found=True,
            divergence_line=line_no,
            summary=f"Syntax Error at line {line_no}: {err_msg}.",
        )

    def _handle_runtime_error(
        self, failure: ExecutionResult, code: str
    ) -> FaultReport:
        line_no = self._extract_error_line(failure.stderr)
        err_msg = self._extract_last_error_line(failure.stderr)
        code_context = self._get_code_context(code, line_no)
        fix_hint = self._get_fix_hint(failure.error_type, err_msg, failure.stderr)

        summary = f"{failure.error_type} at line {line_no}: {err_msg}."
        if code_context:
            summary += f"\nCode context:\n{code_context}"
        if fix_hint:
            summary += f"\nFix hint: {fix_hint}"
        return FaultReport(found=True, divergence_line=line_no, summary=summary)

    def _handle_wrong_answer(
        self,
        failure: ExecutionResult,
        suite_result: SuiteResult,
        test_cases: list[TestCase],
    ) -> FaultReport:
        expected = failure.expected or ""
        actual   = failure.actual   or ""

        failing_input = ""
        for result, tc in zip(suite_result.results, test_cases):
            if not result.passed:
                failing_input = tc.input
                break

        diff_pattern = self._analyze_output_diff(expected, actual)
        summary = (
            f"Wrong Answer ({suite_result.passed}/{suite_result.total} public tests passed).\n"
            f"  Input   : {failing_input!r}\n"
            f"  Expected: {expected!r}\n"
            f"  Actual  : {actual!r}\n"
            f"  Pattern : {diff_pattern}"
        )
        return FaultReport(
            found=True,
            divergence_var="output",
            expected_val=expected,
            actual_val=actual,
            context_vars={"input": failing_input},
            summary=summary,
        )

    # ── 辅助方法 ─────────────────────────────────────────────────────────────

    def _extract_error_line(self, stderr: str) -> int:
        matches = re.findall(r'line (\d+)', stderr)
        return int(matches[-1]) if matches else -1

    def _extract_last_error_line(self, stderr: str) -> str:
        lines = [l.strip() for l in stderr.strip().splitlines() if l.strip()]
        return lines[-1] if lines else "Unknown error"

    def _get_code_context(self, code: str, line_no: int, radius: int = 2) -> str:
        if line_no <= 0:
            return ""
        lines = code.splitlines()
        if line_no > len(lines):
            return ""
        start = max(0, line_no - 1 - radius)
        end   = min(len(lines), line_no + radius)
        return "\n".join(
            f"{'>>>' if i + 1 == line_no else '   '} {i+1}: {lines[i]}"
            for i in range(start, end)
        )

    def _get_fix_hint(self, error_type: str, err_msg: str, stderr: str) -> str:
        if error_type == "index_error":
            return "Check array bounds — index may exceed list length."
        if error_type == "key_error":
            m = re.search(r"KeyError: (.+)$", stderr, re.MULTILINE)
            key = m.group(1) if m else "unknown key"
            return f"Key {key} not in dict — use .get() or check existence first."
        if error_type == "recursion_error":
            return "Recursion depth exceeded — add memoization or convert to iterative."
        if error_type == "type_error":
            return f"Type mismatch: {err_msg} — check input parsing and type conversions."
        if error_type == "name_error":
            m = re.search(r"name '(.+)' is not defined", err_msg)
            if m:
                return f"Variable '{m.group(1)}' used before definition."
        return ""

    def _analyze_output_diff(self, expected: str, actual: str) -> str:
        if not actual:
            return "No output produced — missing print() statement?"
        try:
            exp_n = float(expected.strip())
            act_n = float(actual.strip())
            diff = act_n - exp_n
            if diff in (1, -1):
                return f"Off-by-one error (diff={diff:+.0f}) — check loop bounds."
            if act_n == -exp_n:
                return "Sign error — result has wrong sign."
            return f"Numeric difference: expected {exp_n}, got {act_n} (diff={diff:+g})."
        except (ValueError, ZeroDivisionError):
            pass
        exp_lines = expected.strip().splitlines()
        act_lines = actual.strip().splitlines()
        if len(exp_lines) != len(act_lines):
            return f"Line count mismatch: expected {len(exp_lines)}, got {len(act_lines)} lines."
        if sorted(exp_lines) == sorted(act_lines):
            return "Correct values but wrong order."
        return "Output content is incorrect — review the core logic."