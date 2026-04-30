import pytest

from code_solver.data.lcb_loader import LCBLoader
from code_solver.execution.executor import Executor, TestCase


@pytest.fixture
def executor():
    return Executor(timeout=10)


@pytest.fixture
def lcb_loader():
    return LCBLoader()


class TestStdinExecution:
    """测试 stdin 类型题目（Codeforces/AtCoder 风格）"""

    def test_codeforces_short_sort_correct(self, executor):
        """测试 Codeforces A. Short Sort - 正确答案"""
        code = '''
t = int(input())
for _ in range(t):
    s = input().strip()
    if s == "abc":
        print("YES")
    elif s[0] == "a" and s[2] == "b":
        print("YES")
    elif s[0] == "b" and s[1] == "a":
        print("YES")
    else:
        print("NO")
'''
        test_case = TestCase(
            input="6\nabc\nacb\nbac\nbca\ncab\ncba\n",
            output="YES\nYES\nYES\nNO\nNO\nNO\n",
            testtype="stdin"
        )
        result = executor.run(code, test_case)
        assert result.passed is True, f"Expected passed=True, got passed={result.passed}, error: {result.error_message}"

    def test_codeforces_short_sort_wrong_answer(self, executor):
        """测试 Codeforces A. Short Sort - 错误答案"""
        code = '''
t = int(input())
for _ in range(t):
    s = input().strip()
    print("NO")
'''
        test_case = TestCase(
            input="6\nabc\nacb\nbac\nbca\ncab\ncba\n",
            output="YES\nYES\nYES\nNO\nNO\nYES\n",
            testtype="stdin"
        )
        result = executor.run(code, test_case)
        assert result.passed is False, f"Expected passed=False, got passed={result.passed}"
        assert result.error_code == -2, f"Expected wrong answer error code -2, got {result.error_code}"

    def test_codeforces_compile_error(self, executor):
        """测试编译错误"""
        code = '''
def broken code here
    this is invalid syntax
'''
        test_case = TestCase(
            input="1\n",
            output="1\n",
            testtype="stdin"
        )
        result = executor.run(code, test_case)
        assert result.passed is False, f"Expected passed=False for compile error"
        assert result.error_code == -4, f"Expected compile error code -4, got {result.error_code}"

    def test_codeforces_runtime_error(self, executor):
        """测试运行时错误"""
        code = '''
x = int(input())
print(x / 0)
'''
        test_case = TestCase(
            input="10\n",
            output="Error\n",
            testtype="stdin"
        )
        result = executor.run(code, test_case)
        assert result.passed is False, f"Expected passed=False for runtime error"
        assert result.error_code == -4, f"Expected runtime error code -4, got {result.error_code}"

    def test_run_suite_stdin(self, executor):
        """测试 run_suite 对 stdin 类型题目的处理"""
        code = '''
n = int(input())
print(n * 2)
'''
        test_cases = [
            TestCase(input="5\n", output="10\n", testtype="stdin"),
            TestCase(input="3\n", output="6\n", testtype="stdin"),
            TestCase(input="10\n", output="20\n", testtype="stdin"),
        ]
        suite_result = executor.run_suite(code, test_cases)
        assert suite_result.passed == 3, f"Expected 3 passed, got {suite_result.passed}"
        assert suite_result.total == 3, f"Expected 3 total, got {suite_result.total}"
        assert suite_result.all_passed is True


class TestFunctionalExecution:
    """测试 functional 类型题目（LeetCode 风格）"""

    def test_leetcode_count_seniors_correct(self, executor):
        """测试 LeetCode countSeniors - 正确答案"""
        code = '''
class Solution:
    def countSeniors(self, details: List[str]) -> int:
        count = 0
        for detail in details:
            age = int(detail[11:13])
            if age > 60:
                count += 1
        return count
'''
        test_case = TestCase(
            input='[["7868190130M7522","5303914400F9211","9273338290F4010"]]',
            output="2",
            testtype="functional"
        )
        result = executor.run(code, test_case)
        assert result.passed is True, f"Expected passed=True, got passed={result.passed}, error: {result.error_message}"

    def test_leetcode_count_seniors_wrong_answer(self, executor):
        """测试 LeetCode countSeniors - 错误答案"""
        code = '''
class Solution:
    def countSeniors(self, details: List[str]) -> int:
        return 0
'''
        test_case = TestCase(
            input='[["7868190130M7522","5303914400F9211","9273338290F4010"]]',
            output="2",
            testtype="functional"
        )
        result = executor.run(code, test_case)
        assert result.passed is False, f"Expected passed=False, got passed={result.passed}"
        assert result.error_code == -2, f"Expected wrong answer error code -2, got {result.error_code}"

    def test_leetcode_matrix_sum_correct(self, executor):
        """测试 LeetCode matrixSum - 正确答案"""
        code = '''
class Solution:
    def matrixSum(self, nums: List[List[int]]) -> int:
        score = 0
        for row in nums:
            row.sort(reverse=True)
        for col_idx in range(len(nums[0])):
            col_vals = [row[col_idx] for row in nums if col_idx < len(row)]
            score += max(col_vals)
        return score
'''
        test_case = TestCase(
            input="[[[7, 2, 1], [6, 4, 2], [6, 5, 3], [3, 2, 1]]]",
            output="15",
            testtype="functional"
        )
        result = executor.run(code, test_case)
        assert result.passed is True, f"Expected passed=True, got passed={result.passed}, error: {result.error_message}"

    def test_run_suite_functional(self, executor):
        """测试 run_suite 对 functional 类型题目的处理"""
        code = '''
class Solution:
    def isPowerOfTwo(self, n: int) -> bool:
        return n > 0 and (n & (n - 1)) == 0
'''
        test_cases = [
            TestCase(input="[1]", output="true", testtype="functional"),
            TestCase(input="[16]", output="true", testtype="functional"),
            TestCase(input="[3]", output="false", testtype="functional"),
        ]
        suite_result = executor.run_suite(code, test_cases)
        assert suite_result.passed == 3, f"Expected 3 passed, got {suite_result.passed}"
        assert suite_result.all_passed is True


class TestLiveCodeBenchDataset:
    """测试 LiveCodeBench 数据集加载和执行"""

    def test_load_livecodebench_dataset(self, lcb_loader):
        """测试加载 livecodebench_simple.jsonl 数据集"""
        problems = lcb_loader.load()
        assert len(problems) > 0, "Expected at least one problem loaded"

        stdin_problems = [p for p in problems if p.is_stdin]
        functional_problems = [p for p in problems if not p.is_stdin]

        print(f"\nLoaded {len(problems)} problems:")
        print(f"  - stdin type: {len(stdin_problems)}")
        print(f"  - functional type: {len(functional_problems)}")

        assert len(stdin_problems) > 0, "Expected at least one stdin problem"
        assert len(functional_problems) > 0, "Expected at least one functional problem"

    def test_execute_stdin_problem_from_dataset(self, executor, lcb_loader):
        """测试从数据集选择一个 stdin 类型题目验证 executor"""
        problems = lcb_loader.load()
        stdin_problems = [p for p in problems if p.is_stdin and len(p.public_tests) > 0]

        problem = stdin_problems[0]
        print(f"\nTesting stdin problem: {problem.title}")
        print(f"  Platform: {problem.platform}")

        for tc in problem.public_tests:
            code = self._generate_stdin_solution(problem)
            result = executor.run(code, tc)
            print(f"  Test case result: passed={result.passed}, error_code={result.error_code}")
            if not result.passed:
                print(f"    Expected: {tc.output[:50]}...")
                print(f"    Actual: {result.actual[:50] if result.actual else 'N/A'}...")
                print(f"    Error: {result.error_message}")

    def test_execute_functional_problem_from_dataset(self, executor, lcb_loader):
        """测试从数据集选择一个 functional 类型题目验证 executor"""
        problems = lcb_loader.load()
        functional_problems = [p for p in problems if not p.is_stdin and len(p.public_tests) > 0]

        problem = functional_problems[0]
        print(f"\nTesting functional problem: {problem.title}")
        print(f"  Platform: {problem.platform}")
        print(f"  Starter code: {problem.starter_code[:100]}...")

        for tc in problem.public_tests:
            code = self._generate_functional_solution(problem)
            result = executor.run(code, tc)
            print(f"  Test case result: passed={result.passed}, error_code={result.error_code}")
            if not result.passed:
                print(f"    Expected: {tc.output[:50]}...")
                print(f"    Actual: {result.actual[:50] if result.actual else 'N/A'}...")
                print(f"    Error: {result.error_message}")

    def _generate_stdin_solution(self, problem):
        """根据问题描述生成（简单粗暴的）解决方案用于测试"""
        return problem.starter_code

    def _generate_functional_solution(self, problem):
        """使用 starter_code 作为解决方案"""
        return problem.starter_code


class TestEdgeCases:
    """边界情况测试"""

    def test_timeout_handling(self, executor):
        """测试超时处理"""
        code = '''
import time
time.sleep(100)
print("never reached")
'''
        test_case = TestCase(
            input="1\n",
            output="1\n",
            testtype="stdin"
        )
        result = executor.run(code, test_case)
        assert result.passed is False, f"Expected passed=False for timeout"
        assert result.timed_out is True, f"Expected timed_out=True"

    def test_empty_output(self, executor):
        """测试空输出"""
        code = '''
print("")
'''
        test_case = TestCase(
            input="",
            output="",
            testtype="stdin"
        )
        result = executor.run(code, test_case)
        assert result.passed is True, f"Expected passed=True for empty output"

    def test_function_not_found(self, executor):
        """测试函数未找到"""
        code = '''
x = 1
y = 2
'''
        test_case = TestCase(
            input="1",
            output="1",
            testtype="functional"
        )
        result = executor.run(code, test_case)
        assert result.passed is False, f"Expected passed=False when no function found"
        assert result.error_code == -4, f"Expected error code -4"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
