"""
LiveCodeBench 数据加载器

LCB 题目格式（来自官方仓库 lcb_runner/benchmarks/code_generation.py）：

数据集字段：
  question_id       : 题目唯一 ID
  question_title    : 标题
  question_content  : 题目描述（HTML/Markdown）
  difficulty        : "easy" / "medium" / "hard"
  platform          : "leetcode" / "atcoder" / "codeforces"
  contest_date      : 发布日期
  starter_code      : 函数式题目的函数签名（stdin 型为空字符串）
  public_test_cases : JSON 字符串，格式：
                      [{"input": "...", "output": "...", "testtype": "stdin"/"functional"}, ...]
  private_test_cases: base64(zlib(pickle(json_str))) 压缩格式

题目类型判断：
  is_stdin = any(tc["testtype"] == "stdin" for tc in public_test_cases)
  → stdin 型：读 stdin，写 stdout（AtCoder/Codeforces 风格）
  → functional 型：实现 Solution 类或独立函数（LeetCode 风格）
"""

import base64
import json
import pickle
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from code_solver.execution.executor import TestCase


# ── Problem 数据类 ────────────────────────────────────────────────────────────

FORMATTING_MESSAGE_WITH_STARTER_CODE = "You will use the following starter code to write the solution to the problem and enclose your code within delimiters."

FORMATTING_WITHOUT_STARTER_CODE = "Read the inputs from stdin, solve the problem, and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows. Ensure that when the python program runs, it reads the inputs, runs the algorithm and writes output to STDOUT."

@dataclass
class Problem:
    problem_id: str
    title: str
    description: str
    difficulty: str         # "easy" | "medium" | "hard"
    platform: str           # "leetcode" | "atcoder" | "codeforces"
    release_date: str
    starter_code: str       # functional 型的函数签名；stdin 型为 ""
    is_stdin: bool          # True → stdin 型，False → functional 型

    public_tests: list[TestCase] = field(default_factory=list)
    private_tests: list[TestCase] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)  # 原始数据的元信息，如 func_name

    # def format_for_prompt(self) -> str:
    #     """格式化为 LLM prompt 用的字符串"""
    #     lines = [
    #         f"## {self.title}",
    #         f"Difficulty: {self.difficulty.capitalize()} | Platform: {self.platform.capitalize()}",
    #         f"Format: {'Standard Input/Output' if self.is_stdin else 'Function Implementation'}",
    #         "",
    #         self.description.strip(),
    #     ]
    #     if self.starter_code:
    #         lines.append(f"\n### Starter Code (implement this):\n```python\n{self.starter_code}\n```")
    #     if self.public_tests:
    #         lines.append("\n### Examples:")
    #         for i, tc in enumerate(self.public_tests, 1):
    #             lines.append(f"\n**Example {i}:**")
    #             if tc.testtype == "stdin":
    #                 lines.append(f"Input:\n```\n{tc.input}\n```")
    #                 lines.append(f"Output:\n```\n{tc.output}\n```")
    #             else:
    #                 lines.append(f"Input (args): `{tc.input}`")
    #                 lines.append(f"Output: `{tc.output}`")
    #     return "\n".join(lines)
    def format_for_prompt(self) -> str:
        prompt = f"### Question:\n{self.description}\n\n"
        
        if self.starter_code:
            prompt += (
                f"### Format: {FORMATTING_MESSAGE_WITH_STARTER_CODE}\n"
            )
            prompt += f"```python\n{self.starter_code}\n```\n\n"
        else:
            prompt += f"### Format: {FORMATTING_WITHOUT_STARTER_CODE}\n"
            prompt += "```python\n# YOUR CODE HERE\n```\n\n"
        
        return prompt



# ── 数据解析工具函数 ──────────────────────────────────────────────────────────

def decode_private_tests(encoded: str) -> list[dict]:
    """
    解码 private_test_cases：base64 → zlib 解压 → pickle 反序列化 → JSON 解析
    这是 LCB 官方的压缩格式（见 translate_private_test_cases）
    """
    try:
        decoded = base64.b64decode(encoded)
        decompressed = zlib.decompress(decoded)
        original = pickle.loads(decompressed)
        return json.loads(original)
    except Exception:
        return []


def parse_test_cases(raw: str | list, is_public: bool, metadata: dict) -> list[TestCase]:
    """
    解析 public_test_cases 或 private_test_cases。

    LCB 的测试用例格式（JSON 字符串或列表）：
      [
        {
          "input": "...",
          "output": "...",
          "testtype": "stdin" | "functional"
        },
        ...
      ]
    """
    if not raw:
        return []
    # 字符串 → 解析 JSON
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return []
    if not isinstance(raw, list):
        return []

    tests = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        testtype = item.get("testtype", "stdin")
        inp = str(item.get("input", ""))
        out = str(item.get("output", ""))
        # LCB 的 functional 型 input 有时是字典或列表，需要序列化为 JSON
        if testtype == "functional" and not isinstance(inp, str):
            inp = json.dumps(inp)
        if testtype == "functional" and not isinstance(out, str):
            out = json.dumps(out)
        # 清理 stdin 输出末尾的 "-"（LCB 官方处理）
        if testtype == "stdin" and out.endswith("-"):
            out = out[:out.rfind("-")].rstrip()
        tests.append(TestCase(
            input=inp,
            output=out,
            testtype=testtype,
            is_public=is_public,
            metadata=metadata,
        ))
    return tests


def detect_is_stdin(public_test_cases_raw: str | list) -> bool:
    """判断题目是否为 stdin 型（官方逻辑：任一测试用例 testtype == "stdin"）"""
    if isinstance(public_test_cases_raw, str):
        try:
            public_test_cases_raw = json.loads(public_test_cases_raw)
        except json.JSONDecodeError:
            return True  # 默认 stdin
    if isinstance(public_test_cases_raw, list):
        return any(
            tc.get("testtype") == "stdin"
            for tc in public_test_cases_raw
            if isinstance(tc, dict)
        )
    return True


# ── LCBLoader ─────────────────────────────────────────────────────────────────

class LCBLoader:
    """
    LiveCodeBench 数据加载器

    优先使用本地缓存，否则从 HuggingFace 下载。
    """

    HF_DATASET = "livecodebench/code_generation_lite"

    def __init__(
        self,
        release_version: str = "release_v6",
        cache_dir: str = "./lcb_cache",
        difficulty: Optional[str] = None,
        platform: Optional[str] = None,
        max_problems: Optional[int] = None,
    ):
        self.release_version = release_version
        self.cache_path = Path(cache_dir) / f"{release_version}.json"
        self.difficulty = difficulty
        self.platform = platform
        self.max_problems = max_problems

    def load(self) -> list[Problem]:
        if self.cache_path.exists():
            print(f"[LCBLoader] Loading from cache: {self.cache_path}")
            problems = self._load_cache()
        else:
            print(f"[LCBLoader] Downloading {self.HF_DATASET} ({self.release_version})...")
            problems = self._load_hub()
            self._save_cache(problems)

        if self.difficulty:
            problems = [p for p in problems if p.difficulty == self.difficulty]
        if self.platform:
            problems = [p for p in problems if p.platform == self.platform]
        if self.max_problems:
            problems = problems[:self.max_problems]

        # 统计题型分布
        stdin_n = sum(1 for p in problems if p.is_stdin)
        func_n  = len(problems) - stdin_n
        print(
            f"[LCBLoader] Loaded {len(problems)} problems "
            f"(stdin={stdin_n}, functional={func_n}"
            + (f", difficulty={self.difficulty}" if self.difficulty else "")
            + ")"
        )
        return problems

    # ── HuggingFace 加载 ──────────────────────────────────────────────────────

    def _load_hub(self) -> list[Problem]:
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("pip install datasets")
        # ds = load_dataset(
        #     self.HF_DATASET,
        #     version_tag=self.release_version,
        #     split="test",
        # )
        ds = load_dataset("json", data_files="/data0/xjh/code-solver/src/code_solver/data/livecodebench_all.jsonl")
        problems = []
        for item in ds["train"]:
            p = self._parse_item(item)
            if p:
                problems.append(p)
        return problems

    def _parse_item(self, item: dict) -> Optional[Problem]:
        try:
            raw_metadata = item.get("metadata", "{}")
            if isinstance(raw_metadata, str):
                try:
                    metadata = json.loads(raw_metadata)
                except json.JSONDecodeError:
                    metadata = {}
            else:
                metadata = raw_metadata if isinstance(raw_metadata, dict) else {}

            pub_raw  = item.get("public_test_cases", "[]")
            priv_raw = item.get("private_test_cases", "")

            is_stdin = detect_is_stdin(pub_raw)
            pub_tests  = parse_test_cases(pub_raw, is_public=True, metadata=metadata)

            # private tests：先尝试 base64 解码，失败则直接 JSON 解析
            if isinstance(priv_raw, str) and priv_raw:
                priv_decoded = decode_private_tests(priv_raw)
                if priv_decoded:
                    priv_tests = parse_test_cases(priv_decoded, is_public=False, metadata=metadata)
                else:
                    priv_tests = parse_test_cases(priv_raw, is_public=False, metadata=metadata)
            else:
                priv_tests = parse_test_cases(priv_raw, is_public=False, metadata=metadata)

            return Problem(
                problem_id=str(item.get("question_id", "?")),
                title=str(item.get("question_title", "Untitled")),
                description=str(item.get("question_content", "")),
                difficulty=str(item.get("difficulty", "medium")).lower(),
                platform=str(item.get("platform", "unknown")).lower(),
                release_date=str(item.get("contest_date") or item.get("release_date", "")),
                starter_code=str(item.get("starter_code", "")),
                is_stdin=is_stdin,
                public_tests=pub_tests,
                private_tests=priv_tests,
                metadata=metadata,
            )
        except Exception as e:
            print(f"[LCBLoader] Parse error: {e}")
            return None

    # ── 本地缓存 ──────────────────────────────────────────────────────────────

    def _save_cache(self, problems: list[Problem]):
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        data = []
        for p in problems:
            data.append({
                "problem_id":  p.problem_id,
                "title":       p.title,
                "description": p.description,
                "difficulty":  p.difficulty,
                "platform":    p.platform,
                "release_date":p.release_date,
                "starter_code":p.starter_code,
                "is_stdin":    p.is_stdin,
                "public_tests": [
                    {"input": t.input, "output": t.output, "testtype": t.testtype, "metadata": t.metadata}
                    for t in p.public_tests
                ],
                "private_tests": [
                    {"input": t.input, "output": t.output, "testtype": t.testtype, "metadata": t.metadata}
                    for t in p.private_tests
                ],
                "metadata": p.metadata,
            })
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[LCBLoader] Cached {len(problems)} problems to {self.cache_path}")

    def _load_cache(self) -> list[Problem]:
        with open(self.cache_path, encoding="utf-8") as f:
            data = json.load(f)
        problems = []
        for d in data:
            pub = [
                TestCase(t["input"], t["output"], t.get("testtype","stdin"), True, t.get("metadata", {}))
                for t in d.get("public_tests", [])
            ]
            priv = [
                TestCase(t["input"], t["output"], t.get("testtype","stdin"), False, t.get("metadata", {}))
                for t in d.get("private_tests", [])
            ]
            problems.append(Problem(
                problem_id=d["problem_id"],
                title=d["title"],
                description=d["description"],
                difficulty=d["difficulty"],
                platform=d["platform"],
                release_date=d["release_date"],
                starter_code=d.get("starter_code", ""),
                is_stdin=d.get("is_stdin", True),
                public_tests=pub,
                private_tests=priv,
                metadata=d.get("metadata", {}),
            ))
        return problems


# ── Mock 数据（测试用）────────────────────────────────────────────────────────

def make_mock_stdin_problem() -> Problem:
    """AtCoder/Codeforces 风格：stdin/stdout"""
    return Problem(
        problem_id="mock_stdin_001",
        title="A + B Problem",
        description="Read two integers A and B. Print A + B.",
        difficulty="easy",
        platform="atcoder",
        release_date="2024-01-01",
        starter_code="",
        is_stdin=True,
        public_tests=[
            TestCase("3 5",   "8",  "stdin", True),
            TestCase("10 20", "30", "stdin", True),
        ],
        private_tests=[
            TestCase("0 0",         "0",   "stdin", False),
            TestCase("100 200",     "300", "stdin", False),
            TestCase("-5 5",        "0",   "stdin", False),
            TestCase("1000000000 1","1000000001", "stdin", False),
        ],
    )


def make_mock_functional_problem() -> Problem:
    """LeetCode 风格：函数式"""
    return Problem(
        problem_id="mock_func_001",
        title="Two Sum",
        description=(
            "Given an array of integers `nums` and an integer `target`, "
            "return indices of the two numbers such that they add up to target.\n"
            "You may assume that each input would have exactly one solution."
        ),
        difficulty="easy",
        platform="leetcode",
        release_date="2024-01-01",
        starter_code=(
            "class Solution:\n"
            "    def twoSum(self, nums: List[int], target: int) -> List[int]:\n"
            "        pass"
        ),
        is_stdin=False,
        public_tests=[
            TestCase('[[2,7,11,15], 9]',  '[0, 1]',  "functional", True),
            TestCase('[[3,2,4], 6]',      '[1, 2]',  "functional", True),
        ],
        private_tests=[
            TestCase('[[3,3], 6]',        '[0, 1]',  "functional", False),
            TestCase('[[1,5,3,4], 8]',    '[1, 2]',  "functional", False),
            TestCase('[[0,4,3,0], 0]',    '[0, 3]',  "functional", False),
        ],
    )