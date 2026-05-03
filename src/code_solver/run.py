"""
CodeTree+ 主入口

用法示例：
  # 完整评测
  python run.py --model gpt-4o-mini --output ./results

  # 只评测 hard 题
  python run.py --model gpt-4o-mini --difficulty hard

  # 消融实验（关闭对抗测试）
  python run.py --model gpt-4o-mini --no-adversarial

  # 本地调试（只跑前5道题，使用 mock LLM）
  python run.py --mock --max-problems 5
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def build_search_engine(args):
    """根据命令行参数构造 CodeTreeSearch 实例"""
    import sys
    sys.path.insert(0, str(Path(__file__).parent))

    from code_solver.execution.executor import Executor
    from code_solver.agents.thinker import ThinkerAgent
    from code_solver.agents.solver import SolverAgent
    from code_solver.agents.debugger import DebuggerAgent
    from code_solver.agents.critic import CriticAgent
    from code_solver.modules.adversarial_tester import AdversarialTester
    from code_solver.modules.difficulty_assessor import DifficultyAssessor
    from code_solver.modules.fault_localizer import FaultLocalizer
    from code_solver.tree.search import CodeTreeSearch

    if args.mock:
        from llm.mock_client import FixedMockClient
        llm = FixedMockClient(
            '{"difficulty":"medium","algo_paradigms":["simulation"],"reasoning":"mock"}'
        )
        log.info("Using MockLLM (--mock mode)")
    else:
        from llm.openai_client import OpenAIClient
        llm = OpenAIClient(
            model=args.model,
            api_key=args.api_key or os.environ.get("DEEPSEEK_API_KEY"),
            api_base=args.api_base,
        )
        log.info(f"Using model: {args.model}")

    ex = Executor(timeout=args.timeout)

    search = CodeTreeSearch(
        thinker=ThinkerAgent(llm),
        solver=SolverAgent(llm),
        debugger=DebuggerAgent(llm),
        critic=CriticAgent(
            llm,
            AdversarialTester(llm, ex, num_tests=args.num_adv_tests),
            abort_threshold=args.abort_threshold,
        ),
        assessor=DifficultyAssessor(llm),
        localizer=FaultLocalizer(),
        executor=ex,
        use_difficulty_assessor=not args.no_difficulty,
        use_diversity_thinker=not args.no_diversity,
        use_fault_localizer=not args.no_fault_localizer,
        use_llm_verifier=not args.no_llm_verifier,
        use_adversarial_tester=not args.no_adversarial,
    )
    return search, ex, llm


def main():
    parser = argparse.ArgumentParser(description="CodeTree+ evaluation on LiveCodeBench")

    # LLM 配置
    parser.add_argument("--model",     default="deepseek-v4-flash")
    parser.add_argument("--api-key",   default=None)
    parser.add_argument("--api-base",  default=None, help="vLLM endpoint URL")
    parser.add_argument("--mock",      action="store_true", help="Use mock LLM for local testing")

    # 数据集配置
    parser.add_argument("--lcb-version",   default="release_v6")
    parser.add_argument("--difficulty",    default=None, choices=["easy","medium","hard"])
    parser.add_argument("--max-problems",  type=int, default=None)
    parser.add_argument("--cache-dir",     default="./lcb_cache")

    # 执行配置
    parser.add_argument("--timeout",          type=int,   default=10)
    parser.add_argument("--num-adv-tests",    type=int,   default=5)
    parser.add_argument("--abort-threshold",  type=float, default=3.0)

    # 消融开关
    parser.add_argument("--no-difficulty",      action="store_true", help="Ablation: disable DifficultyAssessor")
    parser.add_argument("--no-diversity",       action="store_true", help="Ablation: disable diversity constraint")
    parser.add_argument("--no-fault-localizer", action="store_true", help="Ablation: disable FaultLocalizer")
    parser.add_argument("--no-llm-verifier",    action="store_true", help="Ablation: disable LLM verifier (public pass = direct accept)")
    parser.add_argument("--no-adversarial",     action="store_true", help="Ablation: disable AdversarialTester")

    # 输出配置
    parser.add_argument("--output",   default="./results")
    parser.add_argument("--run-name", default="codetree_plus_ds")

    args = parser.parse_args()

    # ── 加载数据集 ────────────────────────────────────────────────────────────
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from code_solver.data.lcb_loader import LCBLoader
    from code_solver.evaluation.evaluator import Evaluator

    loader = LCBLoader(
        release_version=args.lcb_version,
        cache_dir=args.cache_dir,
        difficulty=args.difficulty,
        max_problems=args.max_problems,
    )
    problems = loader.load()
    problems = problems[100:]

    if not problems:
        log.error("No problems loaded. Check --lcb-version and cache.")
        return

    # ── 构造搜索引擎 ──────────────────────────────────────────────────────────
    search, executor, llm = build_search_engine(args)
    evaluator = Evaluator(executor, output_dir=args.output, run_name=args.run_name)

    # ── 主循环 ────────────────────────────────────────────────────────────────
    all_results = []
    for i, problem in enumerate(problems):
        log.info(f"\n{'='*60}")
        log.info(f"Problem {i+1}/{len(problems)}: [{problem.problem_id}] {problem.title} ({problem.difficulty})")

        # 断点续跑：跳过已完成的题目
        # cached = evaluator.already_done(problem.problem_id)
        # if cached:
        #     log.info(f"  → Already done (passed_private={cached.passed_private}), skipping.")
        #     all_results.append(cached)
        #     continue

        t0 = time.monotonic()
        try:
            if hasattr(llm, "reset_usage"):
                llm.reset_usage()
            search_result = search.solve(problem)
            usage = llm.get_usage() if hasattr(llm, "get_usage") else {}
            if usage:
                search_result.tree.llm_usage = usage
            elapsed = time.monotonic() - t0
            prob_result = evaluator.evaluate_one(problem, search_result, elapsed)
            log.info(
                f"  → passed_private={prob_result.passed_private}, "
                f"private_rate={prob_result.private_pass_rate:.0%}, "
                f"time={elapsed:.1f}s"
            )
            if usage:
                log.info(
                    f"  → llm_calls={usage.get('calls', 0)}, "
                    f"in_tokens={usage.get('input_tokens', 0)}, "
                    f"in_cache_hit={usage.get('input_tokens_cache_hit', 0)}, "
                    f"in_cache_miss={usage.get('input_tokens_cache_miss', 0)}, "
                    f"out_tokens={usage.get('output_tokens', 0)}, "
                    f"cost_usd={usage.get('cost_usd', 0.0):.6f}, "
                    f"unpriced_calls={usage.get('unpriced_calls', 0)}"
                )
        except Exception as e:
            log.error(f"  → ERROR: {e}", exc_info=True)
            elapsed = time.monotonic() - t0
            from tree.search import SearchResult
            from tree.node import SearchTree
            empty = SearchResult(
                problem_id=problem.problem_id,
                best_code="",
                accepted=False,
                tree=SearchTree(problem_id=problem.problem_id),
            )
            usage = llm.get_usage() if hasattr(llm, "get_usage") else {}
            if usage:
                empty.tree.llm_usage = usage
            prob_result = evaluator.evaluate_one(problem, empty, elapsed)

        all_results.append(prob_result)

    # ── 打印汇总 ──────────────────────────────────────────────────────────────
    report = evaluator.summarize(all_results)
    report.print_summary()

    log.info(f"Results saved to: {args.output}/{args.run_name}.jsonl")


if __name__ == "__main__":
    main()
