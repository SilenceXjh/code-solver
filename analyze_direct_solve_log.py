import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Any
from typing import Optional
from typing import Tuple
 
 
def _repo_root() -> Path:
    return Path(__file__).resolve().parent
 
 
def _add_src_to_syspath():
    src = _repo_root() / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
 
 
def _iter_log_records(log_path: Path):
    with log_path.open("r", encoding="utf-8", errors="replace") as f:
        for i, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if i <= 3:
                continue
            if not line:
                continue
            if not line.startswith("{"):
                continue
            try:
                rec = ast.literal_eval(line)
            except Exception:
                continue
            if not isinstance(rec, dict):
                continue
            pid = rec.get("problem_id")
            passed = rec.get("passed")
            if not isinstance(pid, str):
                continue
            if not isinstance(passed, bool):
                continue
            yield {"problem_id": pid, "passed": passed}
 
 
def _load_is_stdin_from_cache(cache_path: Path) -> Optional[dict[str, bool]]:
    if not cache_path.exists():
        return None
    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(data, list):
        return None
    m: dict[str, bool] = {}
    for item in data:
        if not isinstance(item, dict):
            continue
        pid = item.get("problem_id")
        is_stdin = item.get("is_stdin")
        if isinstance(pid, str) and isinstance(is_stdin, bool):
            m[pid] = is_stdin
    return m
 
 
def _load_is_stdin_from_dataset(dataset_path: Path) -> Optional[dict[str, bool]]:
    if not dataset_path.exists():
        return None
    try:
        data = json.loads(dataset_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(data, list):
        return None
 
    _add_src_to_syspath()
    from code_solver.data.lcb_loader import detect_is_stdin
 
    m: dict[str, bool] = {}
    for item in data:
        if not isinstance(item, dict):
            continue
        pid = item.get("question_id")
        pub = item.get("public_test_cases", "[]")
        if isinstance(pid, str):
            try:
                m[pid] = bool(detect_is_stdin(pub))
            except Exception:
                continue
    return m
 
 
def _heuristic_is_stdin(problem_id: str) -> bool:
    if "_" in problem_id:
        return True
    if problem_id.isdigit():
        return False
    return True
 
 
def _accumulate(
    log_path: Path,
    is_stdin_map: dict[str, bool],
    allow_heuristic: bool,
) -> Tuple[dict[str, int], dict[str, int], int, int]:
    totals = {"stdin": 0, "functional": 0}
    passed = {"stdin": 0, "functional": 0}
    unknown_total = 0
    unknown_passed = 0
 
    for rec in _iter_log_records(log_path):
        pid = rec["problem_id"]
        ok = rec["passed"]
        is_stdin = is_stdin_map.get(pid)
        if is_stdin is None:
            if allow_heuristic:
                is_stdin = _heuristic_is_stdin(pid)
            else:
                unknown_total += 1
                if ok:
                    unknown_passed += 1
                continue
        key = "stdin" if is_stdin else "functional"
        totals[key] += 1
        if ok:
            passed[key] += 1
 
    return totals, passed, unknown_total, unknown_passed
 
 
def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--log", default="direct_solve.log")
    p.add_argument("--cache", default="lcb_cache/release_v6.json")
    p.add_argument("--dataset", default="src/code_solver/data/livecodebench_simple.json")
    p.add_argument("--no-heuristic", action="store_true")
    args = p.parse_args(argv)
 
    repo = _repo_root()
    log_path = (repo / args.log).resolve()
    cache_path = (repo / args.cache).resolve()
    dataset_path = (repo / args.dataset).resolve()
 
    if not log_path.exists():
        print(f"log not found: {log_path}", file=sys.stderr)
        return 2
 
    is_stdin_map: dict[str, bool] = {}
    src = "none"
    cache_map = _load_is_stdin_from_cache(cache_path)
    if cache_map is not None and cache_map:
        is_stdin_map = cache_map
        src = str(cache_path)
    else:
        ds_map = _load_is_stdin_from_dataset(dataset_path)
        if ds_map is not None and ds_map:
            is_stdin_map = ds_map
            src = str(dataset_path)
 
    totals, passed, unknown_total, unknown_passed = _accumulate(
        log_path=log_path,
        is_stdin_map=is_stdin_map,
        allow_heuristic=not args.no_heuristic,
    )
 
    print(f"type_source: {src}")
    print(f"stdin_total: {totals['stdin']}")
    print(f"stdin_passed: {passed['stdin']}")
    print(f"functional_total: {totals['functional']}")
    print(f"functional_passed: {passed['functional']}")
    if unknown_total:
        print(f"unknown_total: {unknown_total}")
        print(f"unknown_passed: {unknown_passed}")
    print(f"overall_total: {totals['stdin'] + totals['functional'] + unknown_total}")
    print(f"overall_passed: {passed['stdin'] + passed['functional'] + unknown_passed}")
    return 0
 
 
if __name__ == "__main__":
    raise SystemExit(main())
