"""
LCB 沙箱代码执行器

LCB 题目有两种格式：
  1. stdin  型：从 stdin 读输入，结果打印到 stdout（AtCoder/Codeforces 风格）
  2. functional 型：实现指定函数签名，返回值与期望比较（LeetCode 风格）

判断依据：TestCase.testtype 字段（"stdin" 或 "functional"）

执行方式：
  两种格式均在独立子进程中运行（subprocess），避免状态污染。
  functional 型通过动态生成 wrapper 代码将函数调用转为 stdout 输出来统一接口。
"""

import json
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass
from typing import Optional


# ── 测试用例 ──────────────────────────────────────────────────────────────────

@dataclass
class TestCase:
    """LCB 单个测试用例"""
    input: str                  # stdin 型：原始输入字符串；functional 型：JSON 序列化的参数
    output: str                 # 期望输出（字符串或 JSON）
    testtype: str = "stdin"     # "stdin" | "functional"
    is_public: bool = True


# ── 执行结果 ──────────────────────────────────────────────────────────────────

@dataclass
class ExecutionResult:
    passed: bool
    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool
    elapsed: float
    expected: str = ""
    actual: str = ""

    @property
    def error_type(self) -> str:
        if self.timed_out:
            return "time_limit_exceeded"
        if self.exit_code != 0:
            err = self.stderr
            for kw, et in [
                ("SyntaxError",     "syntax_error"),
                ("IndentationError","syntax_error"),
                ("NameError",       "name_error"),
                ("AttributeError",  "attribute_error"),
                ("TypeError",       "type_error"),
                ("IndexError",      "index_error"),
                ("KeyError",        "key_error"),
                ("RecursionError",  "recursion_error"),
                ("MemoryError",     "memory_error"),
            ]:
                if kw in err:
                    return et
            return "runtime_error"
        if not self.passed:
            return "wrong_answer"
        return "none"

    def format_for_prompt(self) -> str:
        if self.timed_out:
            return "❌ Time Limit Exceeded."
        if self.exit_code != 0:
            err_lines = self.stderr.strip().splitlines()
            snippet = "\n".join(err_lines[-20:])
            return f"❌ Runtime Error ({self.error_type}):\n```\n{snippet}\n```"
        if not self.passed:
            return (
                f"❌ Wrong Answer:\n"
                f"  Expected : {self.expected!r}\n"
                f"  Got      : {self.actual!r}"
            )
        return "✅ All visible test cases passed."


@dataclass
class SuiteResult:
    results: list[ExecutionResult]
    passed: int
    total: int

    @property
    def all_passed(self) -> bool:
        return self.passed == self.total

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total > 0 else 0.0

    def first_failure(self) -> Optional[ExecutionResult]:
        for r in self.results:
            if not r.passed:
                return r
        return None

    def format_for_prompt(self) -> str:
        if self.all_passed:
            return f"✅ Passed all {self.total} visible test cases."
        f = self.first_failure()
        s = f"❌ Passed {self.passed}/{self.total} visible test cases."
        if f:
            s += "\n\nFirst failure:\n" + f.format_for_prompt()
        return s


# ── 执行器 ────────────────────────────────────────────────────────────────────

class Executor:
    """
    统一的 LCB 代码执行器

    自动根据 TestCase.testtype 选择执行方式：
      - "stdin"      → subprocess + stdin 重定向
      - "functional" → 生成 wrapper 代码，在 subprocess 中调用函数并打印结果
    """

    def __init__(self, timeout: int = 10):
        self.timeout = timeout

    def run(self, code: str, test_case: TestCase) -> ExecutionResult:
        if test_case.testtype == "functional":
            return self._run_functional(code, test_case)
        else:
            return self._run_stdin(code, test_case)

    def run_suite(
        self,
        code: str,
        test_cases: list[TestCase],
        stop_on_first_failure: bool = False,
    ) -> SuiteResult:
        results: list[ExecutionResult] = []
        passed = 0
        for tc in test_cases:
            r = self.run(code, tc)
            results.append(r)
            if r.passed:
                passed += 1
            elif stop_on_first_failure:
                for rem in test_cases[len(results):]:
                    results.append(ExecutionResult(
                        passed=False, stdout="", stderr="[SKIPPED]",
                        exit_code=0, timed_out=False, elapsed=0.0,
                        expected=rem.output, actual="[SKIPPED]",
                    ))
                break
        return SuiteResult(results=results, passed=passed, total=len(test_cases))

    # ── stdin 执行 ────────────────────────────────────────────────────────────

    def _run_stdin(self, code: str, tc: TestCase) -> ExecutionResult:
        t0 = time.monotonic()
        try:
            proc = subprocess.run(
                [sys.executable, "-c", code],
                input=tc.input,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            elapsed = time.monotonic() - t0
            actual = proc.stdout.strip()
            expected = tc.output.strip()
            passed = _compare(actual, expected)
            return ExecutionResult(
                passed=passed, stdout=proc.stdout, stderr=proc.stderr,
                exit_code=proc.returncode, timed_out=False, elapsed=elapsed,
                expected=expected, actual=actual,
            )
        except subprocess.TimeoutExpired:
            return ExecutionResult(
                passed=False, stdout="", stderr="",
                exit_code=-1, timed_out=True, elapsed=self.timeout,
                expected=tc.output.strip(), actual="[TIMEOUT]",
            )
        except Exception as e:
            return ExecutionResult(
                passed=False, stdout="", stderr=str(e),
                exit_code=-1, timed_out=False, elapsed=time.monotonic() - t0,
                expected=tc.output.strip(), actual="[ERROR]",
            )

    # ── functional 执行 ───────────────────────────────────────────────────────

    def _run_functional(self, code: str, tc: TestCase) -> ExecutionResult:
        """
        functional 型执行流程：
          1. 解析 tc.input（JSON）得到函数参数
          2. 提取函数名（从 code 或 starter_code）
          3. 生成 wrapper：exec 用户代码 → 调用函数 → print(json.dumps(result))
          4. 在子进程执行 wrapper，比较 JSON 化输出与期望
        """
        wrapper = self._build_functional_wrapper(code, tc.input)
        t0 = time.monotonic()
        try:
            proc = subprocess.run(
                [sys.executable, "-c", wrapper],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            elapsed = time.monotonic() - t0
            actual_raw = proc.stdout.strip()
            expected_raw = tc.output.strip()

            if proc.returncode != 0:
                return ExecutionResult(
                    passed=False, stdout=proc.stdout, stderr=proc.stderr,
                    exit_code=proc.returncode, timed_out=False, elapsed=elapsed,
                    expected=expected_raw, actual=actual_raw,
                )

            passed = _compare_functional(actual_raw, expected_raw)
            return ExecutionResult(
                passed=passed, stdout=proc.stdout, stderr=proc.stderr,
                exit_code=proc.returncode, timed_out=False, elapsed=elapsed,
                expected=expected_raw, actual=actual_raw,
            )
        except subprocess.TimeoutExpired:
            return ExecutionResult(
                passed=False, stdout="", stderr="",
                exit_code=-1, timed_out=True, elapsed=self.timeout,
                expected=tc.output.strip(), actual="[TIMEOUT]",
            )
        except Exception as e:
            return ExecutionResult(
                passed=False, stdout="", stderr=str(e),
                exit_code=-1, timed_out=False, elapsed=time.monotonic() - t0,
                expected=tc.output.strip(), actual="[ERROR]",
            )

    def _build_functional_wrapper(self, code: str, input_json: str) -> str:
        """
        构建 functional 型执行的 wrapper 代码。
        用 base64 编码传递用户代码和输入，避免引号/换行冲突。
        """
        import base64
        code_b64  = base64.b64encode(code.encode()).decode()
        input_b64 = base64.b64encode(input_json.encode()).decode()

        wrapper = f'''\
import json, sys, base64, types

# 解码用户代码和输入
_user_code  = base64.b64decode("{code_b64}").decode()
_input_json = base64.b64decode("{input_b64}").decode()

# 执行用户代码
_user_ns = {{}}
exec(_user_code, _user_ns)

# 找到可调用函数
_fn = None
# 优先找 Solution 类
if "Solution" in _user_ns and isinstance(_user_ns["Solution"], type):
    _sol = _user_ns["Solution"]()
    _methods = [
        k for k in dir(_sol)
        if not k.startswith("_") and callable(getattr(_sol, k))
    ]
    if _methods:
        _fn = getattr(_sol, _methods[0])

# 退而求其次找独立函数
if _fn is None:
    for _k, _v in _user_ns.items():
        if isinstance(_v, types.FunctionType) and not _k.startswith("_"):
            _fn = _v
            break

if _fn is None:
    raise RuntimeError("No callable found in user code")

# 解析输入并调用
_args = json.loads(_input_json)
if isinstance(_args, list):
    _result = _fn(*_args)
elif isinstance(_args, dict):
    _result = _fn(**_args)
else:
    _result = _fn(_args)

print(json.dumps(_result, default=str))
'''
        return wrapper


# ── 输出比较 ──────────────────────────────────────────────────────────────────

def _compare(actual: str, expected: str) -> bool:
    """stdin 型：字符串比较，支持浮点和多行归一化"""
    a, e = actual.strip(), expected.strip()
    if a == e:
        return True
    a_lines = [l.strip() for l in a.splitlines() if l.strip()]
    e_lines = [l.strip() for l in e.splitlines() if l.strip()]
    if a_lines == e_lines:
        return True
    if len(a_lines) == 1 and len(e_lines) == 1:
        try:
            return abs(float(a_lines[0]) - float(e_lines[0])) < 1e-6
        except ValueError:
            pass
    return False


def _compare_functional(actual_json: str, expected_json: str) -> bool:
    """
    functional 型：先尝试 JSON 反序列化后比较，失败则字符串比较。
    处理常见情况：
      - None vs "null"
      - 列表顺序（如果题目不要求顺序则排序比较）
      - 浮点近似
    """
    actual_json = actual_json.strip()
    expected_json = expected_json.strip()
    if actual_json == expected_json:
        return True
    try:
        a = json.loads(actual_json)
        e = json.loads(expected_json)
        if a == e:
            return True
        # 浮点近似
        if isinstance(a, float) and isinstance(e, (int, float)):
            return abs(a - e) < 1e-6
        if isinstance(e, float) and isinstance(a, (int, float)):
            return abs(a - e) < 1e-6
        # 列表：尝试排序后比较（适用于无序结果）
        if isinstance(a, list) and isinstance(e, list):
            try:
                return sorted(str(x) for x in a) == sorted(str(x) for x in e)
            except Exception:
                pass
        return False
    except (json.JSONDecodeError, TypeError):
        return _compare(actual_json, expected_json)