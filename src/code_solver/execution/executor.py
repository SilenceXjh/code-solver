"""
LCB 沙箱代码执行器

基于 LiveCodeBench 官方评测代码重写，使用 exec + signal 超时机制。
LCB 题目有两种格式：
  1. stdin  型：从 stdin 读输入，结果打印到 stdout（AtCoder/Codeforces 风格）
  2. functional 型：实现指定函数签名，返回值与期望比较（LeetCode 风格）
"""

import ast
import faulthandler
import json
import signal
import sys
import time
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from io import StringIO
from types import ModuleType
from typing import Optional
import multiprocessing


IMPORT_STRING = """from string import *
from re import *
from datetime import *
from collections import *
from heapq import *
from bisect import *
from copy import *
from math import *
from random import *
from statistics import *
from itertools import *
from functools import *
from operator import *
from io import *
from sys import *
from json import *
from builtins import *
from typing import *
import string
import re
import datetime
import collections
import heapq
import bisect
import copy
import math
import random
import statistics
import itertools
import functools
import operator
import io
import sys
import json
sys.setrecursionlimit(50000)
"""


class CODE_TYPE(Enum):
    call_based = 0
    standard_input = 1


class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException


class Capturing(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        self._stringio.close = lambda x: 1
        return self

    def __exit__(self, *args):
        self.append(self._stringio.getvalue())
        del self._stringio
        sys.stdout = self._stdout


class MockStdinWithBuffer:
    def __init__(self, inputs: str):
        self.inputs = inputs
        self._stringio = StringIO(inputs)
        self.buffer = MockBuffer(inputs)

    def read(self, *args):
        return self.inputs

    def readline(self, *args):
        return self._stringio.readline(*args)

    def readlines(self, *args):
        return self.inputs.split("\n")

    def __getattr__(self, name):
        return getattr(self._stringio, name)


class MockBuffer:
    def __init__(self, inputs: str):
        self.inputs = inputs.encode("utf-8")

    def read(self, *args):
        return self.inputs

    def readline(self, *args):
        return self.inputs.split(b"\n")[0] + b"\n"


def clean_if_name(code: str) -> str:
    try:
        astree = ast.parse(code)
        last_block = astree.body[-1]
        if isinstance(last_block, ast.If):
            condition = last_block.test
            if ast.unparse(condition).strip() == "__name__ == '__main__'":
                code = (
                    ast.unparse(astree.body[:-1]) + "\n" + ast.unparse(last_block.body)
                )
    except:
        pass
    return code


def make_function(code: str) -> str:
    try:
        import_stmts = []
        all_other_stmts = []
        astree = ast.parse(code)
        for stmt in astree.body:
            if isinstance(stmt, (ast.Import, ast.ImportFrom)):
                import_stmts.append(stmt)
            else:
                all_other_stmts.append(stmt)

        function_ast = ast.FunctionDef(
            name="wrapped_function",
            args=ast.arguments(
                posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]
            ),
            body=all_other_stmts,
            decorator_list=[],
            lineno=-1,
        )
        main_code = (
            IMPORT_STRING
            + "\n"
            + ast.unparse(import_stmts)
            + "\n"
            + ast.unparse(function_ast)
        )
        return main_code
    except Exception as e:
        return code


def call_method(method, inputs):
    if isinstance(inputs, list):
        inputs = "\n".join(inputs)

    mock_stdin = MockStdinWithBuffer(inputs)

    def _inner_call_method(_method):
        try:
            return _method()
        except SystemExit as e:
            pass
        finally:
            pass

    old_stdin = sys.stdin
    try:
        sys.stdin = mock_stdin
        return _inner_call_method(method)
    finally:
        sys.stdin = old_stdin


class _patch:
    def __init__(self, target, value):
        self.target = target
        self.value = value
        self.old = None

    def __enter__(self):
        parts = self.target.split(".")
        obj = sys
        for part in parts[:-1]:
            obj = getattr(obj, part)
        self.old = getattr(obj, parts[-1])
        setattr(obj, parts[-1], self.value)
        return self

    def __exit__(self, *args):
        parts = self.target.split(".")
        obj = sys
        for part in parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, parts[-1], self.old)


def get_function(compiled_sol, fn_name: str):
    try:
        assert hasattr(compiled_sol, fn_name)
        return getattr(compiled_sol, fn_name)
    except Exception:
        return None


def compile_code(code: str, timeout: int):
    signal.alarm(timeout)
    try:
        tmp_sol = ModuleType("tmp_sol", "")
        exec(code, tmp_sol.__dict__)
        if "class Solution" in code:
            compiled_sol = tmp_sol.Solution()
        else:
            compiled_sol = tmp_sol
        assert compiled_sol is not None
    finally:
        signal.alarm(0)
    return compiled_sol


def convert_line_to_decimals(line: str):
    try:
        decimal_line = [Decimal(elem) for elem in line.split()]
    except:
        return False, []
    return True, decimal_line


def get_stripped_lines(val: str):
    val = val.strip()
    return [val_line.strip() for val_line in val.split("\n")]


def reliability_guard():
    faulthandler.disable()

    import builtins

    builtins.quit = None

    import os

    os.environ["OMP_NUM_THREADS"] = "1"

    os.kill = None
    os.system = None
    os.putenv = None
    os.remove = None
    os.removedirs = None
    os.rmdir = None
    os.fchdir = None
    os.fork = None
    os.fexecve = None
    os.spawnl = None
    os.spawnle = None
    os.spawnv = None
    os.spawnve = None
    os.execl = None
    os.execle = None
    os.execlp = None
    os.execlpe = None
    os.execv = None
    os.execve = None
    os.execvp = None
    os.execvpe = None

    import shutil

    shutil.rmtree = None
    shutil.move = None
    shutil.copy = None
    shutil.copy2 = None
    shutil.copyfile = None

    import subprocess

    subprocess.Popen = None
    subprocess.call = None
    subprocess.run = None
    subprocess.getoutput = None
    subprocess.getstatusoutput = None

    import importlib

    importlib.invalidate_caches()


def truncatefn(s, length=300):
    if isinstance(s, str):
        pass
    else:
        s = str(s)
    if len(s) <= length:
        return s
    return s[: length // 2] + "...(truncated) ..." + s[-length // 2:]


@dataclass
class TestCase:
    input: str
    output: str
    testtype: str = "stdin"
    is_public: bool = True
    metadata: dict = field(default_factory=dict)


@dataclass
class ExecutionResult:
    passed: bool
    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool
    elapsed: float
    test_input: str = ""
    expected: str = ""
    actual: str = ""
    error_code: int = 0
    error_message: str = ""

    @property
    def error_type(self) -> str:
        if self.timed_out:
            return "time_limit_exceeded"
        if self.exit_code != 0:
            for kw, et in [
                ("SyntaxError", "syntax_error"),
                ("IndentationError", "syntax_error"),
                ("NameError", "name_error"),
                ("AttributeError", "attribute_error"),
                ("TypeError", "type_error"),
                ("IndexError", "index_error"),
                ("KeyError", "key_error"),
                ("RecursionError", "recursion_error"),
                ("MemoryError", "memory_error"),
            ]:
                if kw in self.stderr:
                    return et
            return "runtime_error"
        if not self.passed:
            return "wrong_answer"
        return "none"

    def format_for_prompt(self) -> str:
        if self.timed_out:
            return (
                "❌ Time Limit Exceeded.\n"
                f"  Input    : {self.test_input!r}"
            )
        if self.exit_code != 0:
            err_lines = self.stderr.strip().splitlines()
            snippet = "\n".join(err_lines[-20:])
            return (
                f"❌ Runtime Error ({self.error_type}):\n"
                f"  Input    : {self.test_input!r}\n"
                f"```\n{snippet}\n```"
            )
        if not self.passed:
            return (
                f"❌ Wrong Answer:\n"
                f"  Input    : {self.test_input!r}\n"
                f"  Expected : {self.expected!r}\n"
                f"  Got      : {self.actual!r}"
            )
        return "✅ All visible test cases passed."


@dataclass
class SuiteResult:
    results: list[ExecutionResult]
    passed: int
    total: int
    execution_time: float = 0.0
    errors: list = field(default_factory=list)

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


class Executor:
    def __init__(self, timeout: int = 6):
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
        manager = multiprocessing.Manager()
        result_list = manager.list()
        p = multiprocessing.Process(
            target=self._temp_run,
            args=(code, test_cases, result_list),
        )
        p.start()
        p.join()
        if p.is_alive():
            p.kill()
        if not result_list:
            return SuiteResult(
                results=[ExecutionResult(
                    passed=False, stdout="", stderr="[SKIPPED]",
                    exit_code=0, timed_out=False, elapsed=0.0,
                    test_input="[SKIPPED]",
                    expected="", actual="[SKIPPED]",
                    error_code=-1, error_message="Skipped",
                )] , passed=0, total=len(test_cases), execution_time=0.0, errors=["Skipped"],
            )
        return result_list[0]


    def _temp_run(self, code, test_cases, result_list):
        res = self._run_suite(code, test_cases)
        result_list.append(res)

    def _run_suite(
        self,
        code: str,
        test_cases: list[TestCase],
        stop_on_first_failure: bool = False,
    ) -> SuiteResult:
        results: list[ExecutionResult] = []
        passed = 0
        total_execution_time = 0.0
        all_errors = []

        reliability_guard()

        for idx, tc in enumerate(test_cases):
            r = self.run(code, tc)
            results.append(r)
            total_execution_time += r.elapsed

            if r.error_code != 0:
                all_errors.append({
                    "index": idx,
                    "inputs": truncatefn(r.test_input) if r.test_input else truncatefn(tc.input),
                    "expected": truncatefn(tc.output),
                    "output": truncatefn(r.actual) if r.actual else None,
                    "error": truncatefn(r.stderr) if r.stderr else None,
                    "error_code": r.error_code,
                    "error_message": r.error_message or r.error_type,
                })

            if r.passed:
                passed += 1
            elif stop_on_first_failure:
                for rem in test_cases[len(results):]:
                    results.append(ExecutionResult(
                        passed=False, stdout="", stderr="[SKIPPED]",
                        exit_code=0, timed_out=False, elapsed=0.0,
                        test_input=rem.input,
                        expected=rem.output, actual="[SKIPPED]",
                        error_code=-1, error_message="Skipped",
                    ))
                break

        return SuiteResult(
            results=results,
            passed=passed,
            total=len(test_cases),
            execution_time=total_execution_time,
            errors=all_errors,
        )

    def _run_stdin(self, code: str, tc: TestCase) -> ExecutionResult:
        code = clean_if_name(code)
        code = make_function(code)

        compiled_sol = None
        method = None

        try:
            compiled_sol = compile_code(code, self.timeout)
        except TimeoutException:
            return ExecutionResult(
                passed=False, stdout="", stderr="timeout",
                exit_code=-1, timed_out=True, elapsed=float(self.timeout),
                test_input=tc.input,
                expected=tc.output, actual="[TIMEOUT]",
                error_code=-3, error_message="Time Limit Exceeded",
            )
        except Exception as e:
            return ExecutionResult(
                passed=False, stdout="", stderr=str(e),
                exit_code=-1, timed_out=False, elapsed=0.0,
                test_input=tc.input,
                expected=tc.output, actual="[COMPILE ERROR]",
                error_code=-4, error_message=f"Compile Error: {e}",
            )

        if compiled_sol is None:
            return ExecutionResult(
                passed=False, stdout="", stderr="Failed to compile",
                exit_code=-1, timed_out=False, elapsed=0.0,
                test_input=tc.input,
                expected=tc.output, actual="[COMPILE ERROR]",
                error_code=-4, error_message="Failed to compile code",
            )

        method = get_function(compiled_sol, "wrapped_function")
        if method is None:
            return ExecutionResult(
                passed=False, stdout="", stderr="wrapped_function not found",
                exit_code=-1, timed_out=False, elapsed=0.0,
                test_input=tc.input,
                expected=tc.output, actual="[FUNCTION NOT FOUND]",
                error_code=-4, error_message="Function wrapped_function not found",
            )

        faulthandler.enable()
        signal.signal(signal.SIGALRM, timeout_handler)
        captured_output = []
        total_execution_time = 0.0

        try:
            signal.alarm(self.timeout)
            with Capturing() as captured:
                try:
                    start = time.time()
                    call_method(method, tc.input)
                    total_execution_time = time.time() - start
                    signal.alarm(0)
                except TimeoutException:
                    signal.alarm(0)
                    return ExecutionResult(
                        passed=False, stdout="", stderr="timeout",
                        exit_code=-1, timed_out=True, elapsed=float(self.timeout),
                        test_input=tc.input,
                        expected=tc.output, actual="[TIMEOUT]",
                        error_code=-3, error_message="Time Limit Exceeded",
                    )
                except Exception as e:
                    signal.alarm(0)
                    error_msg = repr(e)
                    if "timeoutexception" in error_msg.lower():
                        return ExecutionResult(
                            passed=False, stdout="", stderr=error_msg,
                            exit_code=-1, timed_out=True, elapsed=float(self.timeout),
                            test_input=tc.input,
                            expected=tc.output, actual="[TIMEOUT]",
                            error_code=-3, error_message="Time Limit Exceeded",
                        )
                    return ExecutionResult(
                        passed=False, stdout="", stderr=error_msg,
                        exit_code=-1, timed_out=False, elapsed=0.0,
                        test_input=tc.input,
                        expected=tc.output, actual="[RUNTIME ERROR]",
                        error_code=-4, error_message=f"Runtime Error: {e}",
                    )
                finally:
                    signal.alarm(0)
                    faulthandler.disable()

            if captured:
                prediction = captured[0]
            else:
                prediction = ""

        except Exception as e:
            return ExecutionResult(
                passed=False, stdout="", stderr=str(e),
                exit_code=-1, timed_out=False, elapsed=0.0,
                test_input=tc.input,
                expected=tc.output, actual="[ERROR]",
                error_code=-4, error_message=f"Error: {e}",
            )

        stripped_prediction_lines = get_stripped_lines(prediction)
        stripped_gt_out_lines = get_stripped_lines(tc.output)

        if len(stripped_prediction_lines) != len(stripped_gt_out_lines):
            return ExecutionResult(
                passed=False,
                stdout=prediction,
                stderr="",
                exit_code=0,
                timed_out=False,
                elapsed=total_execution_time,
                test_input=tc.input,
                expected=tc.output,
                actual=prediction,
                error_code=-2,
                error_message=f"Wrong answer: mismatched output length (expected {len(stripped_gt_out_lines)} lines, got {len(stripped_prediction_lines)})",
            )

        for output_line_idx, (pred_line, gt_line) in enumerate(
            zip(stripped_prediction_lines, stripped_gt_out_lines)
        ):
            if pred_line == gt_line:
                continue

            success, decimal_pred = convert_line_to_decimals(pred_line)
            if not success:
                return ExecutionResult(
                    passed=False,
                    stdout=prediction,
                    stderr="",
                    exit_code=0,
                    timed_out=False,
                    elapsed=total_execution_time,
                    test_input=tc.input,
                    expected=tc.output,
                    actual=prediction,
                    error_code=-2,
                    error_message=f"Wrong answer at line {output_line_idx}: {truncatefn(pred_line)} != {truncatefn(gt_line)}",
                )

            success, decimal_gt = convert_line_to_decimals(gt_line)
            if not success:
                return ExecutionResult(
                    passed=False,
                    stdout=prediction,
                    stderr="",
                    exit_code=0,
                    timed_out=False,
                    elapsed=total_execution_time,
                    test_input=tc.input,
                    expected=tc.output,
                    actual=prediction,
                    error_code=-2,
                    error_message=f"Wrong answer at line {output_line_idx}: {truncatefn(pred_line)} != {truncatefn(gt_line)}",
                )

            if decimal_pred != decimal_gt:
                return ExecutionResult(
                    passed=False,
                    stdout=prediction,
                    stderr="",
                    exit_code=0,
                    timed_out=False,
                    elapsed=total_execution_time,
                    test_input=tc.input,
                    expected=tc.output,
                    actual=prediction,
                    error_code=-2,
                    error_message=f"Wrong answer at line {output_line_idx}: {truncatefn(pred_line)} != {truncatefn(gt_line)}",
                )

        return ExecutionResult(
            passed=True,
            stdout=prediction,
            stderr="",
            exit_code=0,
            timed_out=False,
            elapsed=total_execution_time,
            test_input=tc.input,
            expected=tc.output,
            actual=prediction,
            error_code=0,
            error_message="",
        )

    def _run_functional(self, code: str, tc: TestCase) -> ExecutionResult:
        code = IMPORT_STRING + "\n\n" + code

        try:
            compiled_sol = compile_code(code, self.timeout)
        except TimeoutException:
            return ExecutionResult(
                passed=False, stdout="", stderr="timeout",
                exit_code=-1, timed_out=True, elapsed=float(self.timeout),
                test_input=tc.input,
                expected=tc.output, actual="[TIMEOUT]",
                error_code=-3, error_message="Time Limit Exceeded",
            )
        except Exception as e:
            return ExecutionResult(
                passed=False, stdout="", stderr=str(e),
                exit_code=-1, timed_out=False, elapsed=0.0,
                test_input=tc.input,
                expected=tc.output, actual="[COMPILE ERROR]",
                error_code=-4, error_message=f"Compile Error: {e}",
            )

        if compiled_sol is None:
            return ExecutionResult(
                passed=False, stdout="", stderr="Failed to compile",
                exit_code=-1, timed_out=False, elapsed=0.0,
                test_input=tc.input,
                expected=tc.output, actual="[COMPILE ERROR]",
                error_code=-4, error_message="Failed to compile code",
            )

        fn_name = None

        if isinstance(tc.metadata, dict):
            fn_name = tc.metadata.get("func_name")

        assert fn_name, "function name not found in metadata"

        # if fn_name is None:
        #     return ExecutionResult(
        #         passed=False, stdout="", stderr="No function found",
        #         exit_code=-1, timed_out=False, elapsed=0.0,
        #         expected=tc.output, actual="[FUNCTION NOT FOUND]",
        #         error_code=-4, error_message="No callable function found in code",
        #     )

        method = get_function(compiled_sol, fn_name)
        if method is None:
            return ExecutionResult(
                passed=False, stdout="", stderr=f"function {fn_name} not found",
                exit_code=-1, timed_out=False, elapsed=0.0,
                test_input=tc.input,
                expected=tc.output, actual="[FUNCTION NOT FOUND]",
                error_code=-4, error_message=f"Function {fn_name} not found",
            )

        gt_inp = [json.loads(line) for line in tc.input.split("\n")]
        gt_out = json.loads(tc.output)

        faulthandler.enable()
        signal.signal(signal.SIGALRM, timeout_handler)
        total_execution_time = 0.0

        try:
            # print("gt_inp:", gt_inp, type(gt_inp))
            # print("gt_out:", gt_out, type(gt_out))

            signal.alarm(self.timeout)
            start = time.time()
            # prediction = method(*gt_inp) if isinstance(gt_inp, list) else method(gt_inp)
            prediction = method(*gt_inp)
            total_execution_time += time.time() - start
            signal.alarm(0)
        except TimeoutException:
            signal.alarm(0)
            return ExecutionResult(
                passed=False, stdout="", stderr="timeout",
                exit_code=-1, timed_out=True, elapsed=float(self.timeout),
                test_input=tc.input,
                expected=tc.output, actual="[TIMEOUT]",
                error_code=-3, error_message="Time Limit Exceeded",
            )
        except SystemExit:
            signal.alarm(0)
            total_execution_time += time.time() - start
            return ExecutionResult(
                passed=False, stdout="", stderr="",
                exit_code=-1, timed_out=False, elapsed=total_execution_time,
                test_input=tc.input,
                expected=tc.output, actual="[RUNTIME ERROR]",
                error_code=-4, error_message=f"Runtime Error: {e}",
            )
        except Exception as e:
            signal.alarm(0)
            error_msg = repr(e)
            if "timeoutexception" in error_msg.lower():
                return ExecutionResult(
                    passed=False, stdout="", stderr=error_msg,
                    exit_code=-1, timed_out=True, elapsed=float(self.timeout),
                    test_input=tc.input,
                    expected=tc.output, actual="[TIMEOUT]",
                    error_code=-3, error_message="Time Limit Exceeded",
                )
            return ExecutionResult(
                passed=False, stdout="", stderr=error_msg,
                exit_code=-1, timed_out=False, elapsed=total_execution_time,
                test_input=tc.input,
                expected=tc.output, actual="[RUNTIME ERROR]",
                error_code=-4, error_message=f"Runtime Error: {e}",
            )
        finally:
            faulthandler.disable()

        if isinstance(prediction, tuple):
            prediction = list(prediction)

        tmp_result = prediction == gt_out

        if not tmp_result:
            return ExecutionResult(
                passed=False,
                stdout="",
                stderr="",
                exit_code=0,
                timed_out=False,
                elapsed=total_execution_time,
                test_input=tc.input,
                expected=json.dumps(gt_out, default=str),
                actual=json.dumps(prediction, default=str),
                error_code=-2,
                error_message="Wrong Answer",
            )

        return ExecutionResult(
            passed=True,
            stdout="",
            stderr="",
            exit_code=0,
            timed_out=False,
            elapsed=total_execution_time,
            test_input=tc.input,
            expected=tc.output,
            actual=json.dumps(prediction, default=str),
            error_code=0,
            error_message="",
        )


def run_test(sample, test=None, timeout=6):
    """
    官方 LiveCodeBench run_test 接口
    sample: 包含 input_output 的字典
    test: 要测试的代码字符串
    timeout: 超时时间（秒）
    """
    signal.signal(signal.SIGALRM, timeout_handler)
    reliability_guard()

    try:
        in_outs = json.loads(sample["input_output"])
    except (ValueError, KeyError):
        in_outs = None

    if in_outs:
        if in_outs.get("fn_name") is None:
            which_type = CODE_TYPE.standard_input
            method_name = None
        else:
            which_type = CODE_TYPE.call_based
            method_name = in_outs["fn_name"]
    else:
        which_type = CODE_TYPE.standard_input
        method_name = None

    if test is None:
        return in_outs, {"error": "no test code provided"}

    executor = Executor(timeout=timeout)

    if which_type == CODE_TYPE.call_based:
        test_cases = [
            TestCase(
                input=json.dumps(inp),
                output=json.dumps(out),
                testtype="functional",
            )
            for inp, out in zip(in_outs["inputs"], in_outs["outputs"])
        ]
        results, metadata = [], {"errors": []}
        for i, tc in enumerate(test_cases):
            r = executor.run(test, tc)
            results.append(r.passed if r.passed else r.error_code)
            if r.error_code != 0:
                metadata["errors"].append({
                    "index": i,
                    "inputs": truncatefn(r.test_input) if r.test_input else truncatefn(tc.input),
                    "expected": truncatefn(tc.output),
                    "output": truncatefn(r.actual),
                    "error_code": r.error_code,
                    "error_message": r.error_message or r.error_type,
                })
        metadata["execution_time"] = sum(r.elapsed for r in executor.run_suite(test, test_cases).results)
        return results, metadata

    elif which_type == CODE_TYPE.standard_input:
        test_cases = [
            TestCase(input=inp, output=out, testtype="stdin")
            for inp, out in zip(in_outs["inputs"], in_outs["outputs"])
        ]
        results, metadata = [], {"errors": []}
        for i, tc in enumerate(test_cases):
            r = executor.run(test, tc)
            results.append(r.passed if r.passed else r.error_code)
            if r.error_code != 0:
                metadata["errors"].append({
                    "index": i,
                    "inputs": truncatefn(r.test_input) if r.test_input else truncatefn(tc.input),
                    "expected": truncatefn(tc.output),
                    "output": truncatefn(r.actual) if r.actual else None,
                    "error": truncatefn(r.stderr) if r.stderr else None,
                    "error_code": r.error_code,
                    "error_message": r.error_message or r.error_type,
                })
        metadata["execution_time"] = sum(r.elapsed for r in executor.run_suite(test, test_cases).results)
        return results, metadata
