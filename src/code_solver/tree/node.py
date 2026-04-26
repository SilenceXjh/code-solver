"""
树节点数据结构

CodeTree+ 的搜索树以 Node 为基本单位。
每个 Node 对应一个「策略 + 代码实现」的组合，记录了：
  - 该节点的代码、策略描述
  - 执行反馈（来自 public tests）
  - Critic 评分
  - Fault 定位报告（如果执行失败）
  - 子节点列表（refinements）
  - 当前状态（pending / accepted / aborted）

树结构：
  root (虚节点，代表问题)
    ├── Node(strategy_1, code_1)
    │     ├── Node(strategy_1, refined_code_1a)
    │     └── Node(strategy_1, refined_code_1b)
    └── Node(strategy_2, code_2)
          └── Node(strategy_2, refined_code_2a)
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from execution.executor import SuiteResult


class NodeStatus(Enum):
    PENDING  = "pending"    # 初始状态，等待评估
    REFINING = "refining"   # 正在被 Debugger 改进
    ACCEPTED = "accepted"   # Critic 认为可以提交
    ABORTED  = "aborted"    # Critic 认为无继续价值


@dataclass
class FaultReport:
    """FaultLocalizer 生成的定位报告，喂给 Debugger"""
    found: bool = False                         # 是否成功定位到分叉点
    divergence_line: int = -1                   # 出现分叉的行号
    divergence_var: str = ""                    # 分叉的变量名
    expected_val: object = None                 # 期望值（成功执行时该变量的值）
    actual_val: object = None                   # 实际值（失败执行时该变量的值）
    context_vars: dict = field(default_factory=dict)  # 分叉点附近的其他变量
    summary: str = ""                           # 给 Debugger 看的自然语言摘要

    def format_for_prompt(self) -> str:
        if not self.found:
            return ""
        return (
            f"🔍 Fault Localization:\n"
            f"  Line {self.divergence_line}: variable `{self.divergence_var}` diverged.\n"
            f"  Expected: {self.expected_val!r}\n"
            f"  Actual  : {self.actual_val!r}\n"
            f"  Context : {self.context_vars}\n"
            f"  Summary : {self.summary}"
        )


@dataclass
class Node:
    """
    搜索树中的一个节点

    字段说明：
      strategy    : Thinker 给出的高层解题策略（自然语言）
      algo_paradigm: 该策略对应的算法范式标签（用于多样性去重）
      code        : Solver/Debugger 生成的代码
      depth       : 节点在树中的深度（root=0, 第一层策略=1, 第一次 refine=2, ...）
      parent      : 父节点引用（root 节点为 None）
      children    : 子节点列表（Debugger 的 refinements）
      status      : 节点当前状态
      suite_result: 对 public tests 执行的结果
      critic_score: Critic 给出的分数（0~10），-1 表示未评分
      fault_report: FaultLocalizer 的定位报告
      reflection  : Thinker 对本节点的反思（用于指导 Debugger）
      created_at  : 节点创建时间戳
    """
    strategy: str
    algo_paradigm: str
    code: str
    depth: int = 0
    parent: Optional["Node"] = field(default=None, repr=False)
    children: list["Node"] = field(default_factory=list, repr=False)
    status: NodeStatus = NodeStatus.PENDING
    suite_result: Optional[SuiteResult] = None
    critic_score: float = -1.0
    fault_report: Optional[FaultReport] = None
    reflection: str = ""
    created_at: float = field(default_factory=time.monotonic)

    # 对抗测试验证结果（Critic 的 Solution Verification 阶段产生）
    adversarial_passed: Optional[bool] = None   # None=未验证, True/False=验证结果

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    @property
    def passed_public_tests(self) -> bool:
        """是否通过了所有 public tests"""
        if self.suite_result is None:
            return False
        return self.suite_result.all_passed

    @property
    def public_pass_rate(self) -> float:
        if self.suite_result is None:
            return 0.0
        return self.suite_result.pass_rate

    def add_child(self, child: "Node") -> None:
        child.parent = self
        child.depth = self.depth + 1
        self.children.append(child)

    def best_child(self) -> Optional["Node"]:
        """按 critic_score 返回最优子节点"""
        if not self.children:
            return None
        return max(self.children, key=lambda n: n.critic_score)

    def summary(self) -> str:
        """单行摘要，用于日志"""
        status_icon = {
            NodeStatus.PENDING:  "⏳",
            NodeStatus.REFINING: "🔧",
            NodeStatus.ACCEPTED: "✅",
            NodeStatus.ABORTED:  "❌",
        }[self.status]
        pr = f"{self.suite_result.passed}/{self.suite_result.total}" if self.suite_result else "?/?"
        score = f"{self.critic_score:.1f}" if self.critic_score >= 0 else "N/A"
        return (
            f"{status_icon} depth={self.depth} "
            f"paradigm={self.algo_paradigm} "
            f"public={pr} score={score}"
        )


@dataclass
class SearchTree:
    """
    整棵搜索树的容器，管理所有节点和全局搜索状态。
    """
    problem_id: str
    nodes: list[Node] = field(default_factory=list)         # 所有节点（按创建顺序）
    accepted_node: Optional[Node] = None                     # 最终接受的节点
    budget_used: int = 0                                     # 已消耗的生成次数
    budget_total: int = 20                                   # 总预算

    # 已探索的算法范式（用于 Thinker 多样性约束）
    explored_paradigms: list[str] = field(default_factory=list)

    def add_node(self, node: Node) -> None:
        self.nodes.append(node)
        self.budget_used += 1
        if node.algo_paradigm and node.algo_paradigm not in self.explored_paradigms:
            self.explored_paradigms.append(node.algo_paradigm)

    def is_budget_exhausted(self) -> bool:
        return self.budget_used >= self.budget_total

    def best_node(self) -> Optional[Node]:
        """
        预算用尽或搜索结束时，选出最优节点提交。
        优先级：passed_all > critic_score > public_pass_rate
        """
        if self.accepted_node:
            return self.accepted_node
        candidates = [n for n in self.nodes if n.code and n.status != NodeStatus.ABORTED]
        if not candidates:
            return None
        # 先看有没有通过全部 public tests 的
        passed_all = [n for n in candidates if n.passed_public_tests]
        pool = passed_all if passed_all else candidates
        return max(pool, key=lambda n: (n.critic_score, n.public_pass_rate))

    def stats(self) -> dict:
        return {
            "total_nodes": len(self.nodes),
            "budget_used": self.budget_used,
            "accepted": self.accepted_node is not None,
            "explored_paradigms": self.explored_paradigms,
            "pass_distribution": {
                "accepted": sum(1 for n in self.nodes if n.status == NodeStatus.ACCEPTED),
                "aborted":  sum(1 for n in self.nodes if n.status == NodeStatus.ABORTED),
                "pending":  sum(1 for n in self.nodes if n.status == NodeStatus.PENDING),
            },
        }