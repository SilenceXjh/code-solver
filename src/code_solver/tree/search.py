"""
CodeTree+ 主搜索循环

将所有模块串联成完整的搜索流程：

  1. DifficultyAssessor  → 自适应确定 (width, depth)          [方向C]
  2. ThinkerAgent        → 生成多样化策略                      [方向D]
  3. SolverAgent         → 按策略生成代码
  4. Executor            → 执行 public tests
  5. FaultLocalizer      → 结构化错误定位                      [方向B]
  6. ThinkerAgent        → 生成 reflection
  7. CriticAgent         → 打分 + 对抗测试验证                 [方向A]
     ├── ACCEPT          → 搜索结束
     ├── REFINE          → 调用 Debugger 修复，深度递归
     └── ABORT           → 放弃该策略，继续探索下一策略
"""

import logging
from dataclasses import dataclass

from code_solver.agents.critic import CriticAgent, CriticDecision
from code_solver.agents.debugger import DebuggerAgent
from code_solver.agents.solver import SolverAgent
from code_solver.agents.thinker import ThinkerAgent
from code_solver.data.lcb_loader import Problem
from code_solver.execution.executor import Executor
from code_solver.modules.adversarial_tester import AdversarialTester
from code_solver.modules.difficulty_assessor import DifficultyAssessor
from code_solver.modules.fault_localizer import FaultLocalizer
from code_solver.tree.node import FaultReport, Node, NodeStatus, SearchTree

log = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """单道题的搜索结果"""
    problem_id: str
    best_code: str          # 最终提交的代码（空字符串表示未找到）
    accepted: bool          # 是否被 Critic Accept
    tree: SearchTree        # 完整搜索树（供分析和消融实验使用）

    @property
    def found_solution(self) -> bool:
        return bool(self.best_code)


class CodeTreeSearch:
    """
    CodeTree+ 搜索引擎

    用法：
        search = CodeTreeSearch(thinker, solver, debugger, critic,
                                assessor, localizer, executor)
        result = search.solve(problem)
    """

    def __init__(
        self,
        thinker: ThinkerAgent,
        solver: SolverAgent,
        debugger: DebuggerAgent,
        critic: CriticAgent,
        assessor: DifficultyAssessor,
        localizer: FaultLocalizer,
        executor: Executor,
        # 消融实验开关
        use_difficulty_assessor: bool = True,   # 方向C：难度感知预算
        use_diversity_thinker:   bool = True,   # 方向D：多样性策略约束
        use_fault_localizer:     bool = True,   # 方向B：错误定位
        use_llm_verifier:        bool = True,   # public 全过后 LLM 验证
        use_adversarial_tester:  bool = True,   # 方向A：对抗测试执行验证
    ):
        self.thinker  = thinker
        self.solver   = solver
        self.debugger = debugger
        self.critic   = critic
        self.assessor = assessor
        self.localizer = localizer
        self.executor  = executor

        self.use_difficulty_assessor = use_difficulty_assessor
        self.use_diversity_thinker   = use_diversity_thinker
        self.use_fault_localizer     = use_fault_localizer
        self.use_llm_verifier        = use_llm_verifier
        self.use_adversarial_tester  = use_adversarial_tester

    def solve(self, problem: Problem) -> SearchResult:
        """
        对单道题执行完整搜索，返回最优解。
        """
        problem_str = problem.format_for_prompt()
        log.info(f"[{problem.problem_id}] Starting search: {problem.title}")

        # ── Step 1：难度评估 → 自适应预算 ────────────────────────────────────
        if self.use_difficulty_assessor:
            assessment = self.assessor.assess(problem.description)
            width = assessment.width
            depth = assessment.depth
            log.info(
                f"[{problem.problem_id}] Assessed: {assessment.difficulty}, "
                f"paradigms={assessment.algo_paradigms[:2]}, "
                f"width={width}, depth={depth}"
            )
        else:
            # 消融：固定预算（CodeTree 原始默认）
            width, depth = 3, 3

        tree = SearchTree(
            problem_id=problem.problem_id,
            budget_total=width * (depth + 1),  # 粗略估计总预算
        )

        # ── Step 2：BFS 策略探索（宽度 = width）─────────────────────────────
        for strategy_idx in range(width):
            if tree.is_budget_exhausted():
                log.info(f"[{problem.problem_id}] Budget exhausted after {strategy_idx} strategies.")
                break

            # 生成新策略（多样性约束）
            explored = tree.explored_paradigms if self.use_diversity_thinker else []
            strategy_result = self.thinker.generate_strategy(problem.description, explored)

            log.info(
                f"[{problem.problem_id}] Strategy {strategy_idx+1}/{width}: "
                f"{strategy_result.algo_paradigm} — {strategy_result.strategy[:100]}..."
            )

            # 生成初始代码
            code = self.solver.generate(problem, strategy_result.strategy)

            # 创建根节点（深度=1）
            node = Node(
                strategy=strategy_result.strategy,
                algo_paradigm=strategy_result.algo_paradigm,
                code=code,
                depth=1,
            )
            tree.add_node(node)

            # ── Step 3：DFS 调试（深度 = depth）─────────────────────────────
            accepted = self._refine_loop(
                node=node,
                problem_str=problem.description,
                public_tests=problem.public_tests,
                tree=tree,
                max_depth=depth,
            )

            if accepted:
                tree.accepted_node = node  # 可能被子节点覆盖，search_tree.best_node() 处理
                log.info(f"[{problem.problem_id}] ✅ Solution accepted!")
                break

        # ── Step 4：收集最终答案 ──────────────────────────────────────────────
        best = tree.best_node()
        log.info(
            f"[{problem.problem_id}] Search done. "
            f"Stats: {tree.stats()}"
        )

        return SearchResult(
            problem_id=problem.problem_id,
            best_code=best.code if best else "",
            accepted=tree.accepted_node is not None,
            tree=tree,
        )

    def _refine_loop(
        self,
        node: Node,
        problem_str: str,
        public_tests: list,
        tree: SearchTree,
        max_depth: int,
        current_depth: int = 0,
    ) -> bool:
        """
        对单个节点执行 Evaluate → [Refine →] 循环，最多 max_depth 轮。

        Returns:
            True 表示找到了被 Accept 的解
        """
        if current_depth > max_depth or tree.is_budget_exhausted():
            node.status = NodeStatus.ABORTED
            return False

        # ── 执行 public tests ──────────────────────────────────────────────
        suite_result = self.executor.run_suite(node.code, public_tests)
        node.suite_result = suite_result
        log.info(
            f"  {'  ' * current_depth}Depth {current_depth}: "
            f"public {suite_result.passed}/{suite_result.total} passed"
        )

        # ── FaultLocalizer ─────────────────────────────────────────────────
        if self.use_fault_localizer and not suite_result.all_passed:
            node.fault_report = self.localizer.localize(
                node.code, suite_result, public_tests
            )
        else:
            node.fault_report = FaultReport(found=False)

        # ── Critic 评分 + 决策 ─────────────────────────────────────────────
        critic_result = self._make_critic_decision(
            problem_str, node, suite_result, public_tests
        )

        node.critic_score = critic_result.score
        node.status = NodeStatus.PENDING

        log.info(
            f"  {'  ' * current_depth}Critic: {critic_result.decision.value}, "
            f"score={critic_result.score:.1f}"
        )

        # ── 根据决策分路 ────────────────────────────────────────────────────
        if critic_result.decision == CriticDecision.ACCEPT:
            node.status = NodeStatus.ACCEPTED
            tree.accepted_node = node
            return True

        if critic_result.decision == CriticDecision.ABORT:
            node.status = NodeStatus.ABORTED
            return False

        # REFINE：生成 reflection → 调用 Debugger → 递归
        node.status = NodeStatus.REFINING

        exec_feedback = suite_result.format_for_prompt()
        # 如果对抗测试有反例，把反例也加入反馈
        if critic_result.adversarial and not critic_result.adversarial.all_passed:
            exec_feedback += "\n\n" + self.critic.adversarial_tester.format_failure_for_prompt(
                critic_result.adversarial
            ) if hasattr(self.critic, 'adversarial_tester') else ""

        reflection = self.thinker.generate_reflection(
            problem=problem_str,
            strategy=node.strategy,
            code=node.code,
            exec_feedback=exec_feedback,
            fault_report=node.fault_report,
        )
        node.reflection = reflection

        fixed_code = self.debugger.fix(
            problem=problem_str,
            strategy=node.strategy,
            code=node.code,
            exec_feedback=exec_feedback,
            fault_report=node.fault_report,
            reflection=reflection,
        )

        # 创建子节点（refined 版本）
        child = Node(
            strategy=node.strategy,
            algo_paradigm=node.algo_paradigm,
            code=fixed_code,
        )
        node.add_child(child)
        tree.add_node(child)

        return self._refine_loop(
            node=child,
            problem_str=problem_str,
            public_tests=public_tests,
            tree=tree,
            max_depth=max_depth,
            current_depth=current_depth + 1,
        )

    def _make_critic_decision(self, problem_str, node, suite_result, public_tests):
        """
        根据消融开关组合路由到对应的验证策略。

        use_adversarial | use_llm_verifier | 行为
        ----------------|-----------------|------
        False           | False           | public 全过直接 Accept（最简 Baseline）
        False           | True            | public 全过 + LLM 验证
        True            | False/True      | 对抗测试（+ 仲裁）
        """
        from code_solver.agents.critic import CriticResult

        # public tests 未全过：所有模式下统一用打分决定 Refine/Abort
        # 改成：所有模式下都用 Refine
        if not suite_result.all_passed:
            # score, reason = self.critic._score(
            #     problem_str, node.strategy, node.code, suite_result
            # )
            # if score < self.critic.abort_threshold:
            #     return CriticResult(CriticDecision.ABORT, score, reason)
            score = 5.0
            return CriticResult(CriticDecision.REFINE, score, "")

        # public tests 全过 → 根据开关选择验证策略
        if self.use_adversarial_tester:
            # 完整版：对抗测试执行验证（+ 失败时 LLM 仲裁）
            return self.critic.evaluate(
                problem_str, node.strategy, node.code, suite_result, public_tests
            )
        elif self.use_llm_verifier:
            # 消融A：无对抗测试，但有 LLM 验证
            looks_correct, reason = self.critic.verify_by_llm(
                problem_str, node.strategy, node.code
            )
            score = 8.0 if looks_correct else 5.0
            if looks_correct:
                return CriticResult(CriticDecision.ACCEPT, score, reason + " [LLM verified]")
            return CriticResult(CriticDecision.REFINE, score, reason + " [LLM suspects issues]")
        else:
            # 消融B：最简 Baseline，public 全过直接 Accept
            return CriticResult(
                decision=CriticDecision.ACCEPT,
                score=7.0,
                reason="All public tests passed (direct accept, no verification).",
            )