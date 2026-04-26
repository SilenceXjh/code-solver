"""
评测模块

功能：
  1. 对 SearchResult 用 private tests 做最终评测（pass@1）
  2. 汇总多道题的结果，输出统计报告
  3. 保存结果到 JSONL 文件，支持断点续跑
"""

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from data.lcb_loader import Problem
from execution.executor import Executor
from tree.search import SearchResult


@dataclass
class ProblemResult:
    """单道题的最终评测结果"""
    problem_id: str
    title: str
    difficulty: str
    accepted_by_critic: bool     # Critic 是否 Accept
    passed_private: bool         # 实际通过 private tests（最终评测）
    private_pass_rate: float     # private tests 通过率
    best_code: str
    search_stats: dict
    elapsed_seconds: float


@dataclass
class EvalReport:
    """整体评测报告"""
    total: int
    pass_at_1: float             # 通过 private tests 的比例
    critic_accept_rate: float    # Critic Accept 的比例（与 pass@1 的差距反映 Verifier 质量）
    by_difficulty: dict          # 各难度的 pass@1
    results: list[ProblemResult]

    def print_summary(self):
        print("=" * 60)
        print("CodeTree+ Evaluation Report")
        print("=" * 60)
        print(f"Total problems : {self.total}")
        print(f"Pass@1         : {self.pass_at_1:.1%}  ({int(self.pass_at_1 * self.total)}/{self.total})")
        print(f"Critic Accept  : {self.critic_accept_rate:.1%}")
        print()
        print("By difficulty:")
        for diff, stats in sorted(self.by_difficulty.items()):
            p = stats['passed']
            t = stats['total']
            rate = p / t if t > 0 else 0
            print(f"  {diff:8s}: {rate:.1%}  ({p}/{t})")
        print("=" * 60)


class Evaluator:
    """
    评测器

    用法：
        evaluator = Evaluator(executor, output_dir="./results")
        report = evaluator.evaluate(problems, search_results)
    """

    def __init__(
        self,
        executor: Executor,
        output_dir: str = "./results",
        run_name: str = "codetree_plus",
    ):
        self.executor   = executor
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run_name   = run_name
        self.output_file = self.output_dir / f"{run_name}.jsonl"

    def evaluate_one(
        self,
        problem: Problem,
        search_result: SearchResult,
        elapsed: float = 0.0,
    ) -> ProblemResult:
        """对单道题用 private tests 做最终评测"""
        passed_private = False
        private_pass_rate = 0.0

        if search_result.best_code and problem.private_tests:
            suite = self.executor.run_suite(
                search_result.best_code,
                problem.private_tests,
            )
            passed_private   = suite.all_passed
            private_pass_rate = suite.pass_rate
        elif search_result.best_code and not problem.private_tests:
            # 没有 private tests（本地调试时），用 Critic Accept 代替
            passed_private    = search_result.accepted
            private_pass_rate = 1.0 if search_result.accepted else 0.0

        result = ProblemResult(
            problem_id=problem.problem_id,
            title=problem.title,
            difficulty=problem.difficulty,
            accepted_by_critic=search_result.accepted,
            passed_private=passed_private,
            private_pass_rate=private_pass_rate,
            best_code=search_result.best_code,
            search_stats=search_result.tree.stats(),
            elapsed_seconds=round(elapsed, 2),
        )
        # 追加写入 JSONL
        self._append_result(result)
        return result

    def summarize(self, results: list[ProblemResult]) -> EvalReport:
        """汇总多道题的结果"""
        total   = len(results)
        passed  = sum(1 for r in results if r.passed_private)
        accepted = sum(1 for r in results if r.accepted_by_critic)

        by_difficulty: dict[str, dict] = {}
        for r in results:
            d = r.difficulty
            if d not in by_difficulty:
                by_difficulty[d] = {"passed": 0, "total": 0}
            by_difficulty[d]["total"] += 1
            if r.passed_private:
                by_difficulty[d]["passed"] += 1

        report = EvalReport(
            total=total,
            pass_at_1=passed / total if total > 0 else 0.0,
            critic_accept_rate=accepted / total if total > 0 else 0.0,
            by_difficulty=by_difficulty,
            results=results,
        )

        # 保存汇总报告
        summary_file = self.output_dir / f"{self.run_name}_summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump({
                "total": report.total,
                "pass_at_1": report.pass_at_1,
                "critic_accept_rate": report.critic_accept_rate,
                "by_difficulty": report.by_difficulty,
            }, f, indent=2)

        return report

    def already_done(self, problem_id: str) -> Optional[ProblemResult]:
        """检查是否已有该题的结果（断点续跑用）"""
        if not self.output_file.exists():
            return None
        with open(self.output_file, encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if data.get("problem_id") == problem_id:
                        return ProblemResult(**data)
                except Exception:
                    continue
        return None

    def _append_result(self, result: ProblemResult):
        with open(self.output_file, "a", encoding="utf-8") as f:
            # 不保存完整代码到 JSONL（太大），单独保存
            data = {
                "problem_id":        result.problem_id,
                "title":             result.title,
                "difficulty":        result.difficulty,
                "accepted_by_critic":result.accepted_by_critic,
                "passed_private":    result.passed_private,
                "private_pass_rate": result.private_pass_rate,
                "search_stats":      result.search_stats,
                "elapsed_seconds":   result.elapsed_seconds,
            }
            f.write(json.dumps(data, ensure_ascii=False) + "\n")