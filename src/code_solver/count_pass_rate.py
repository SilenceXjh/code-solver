from code_solver.data.lcb_loader import LCBLoader
from code_solver.execution.executor import Executor
import os
from tqdm import tqdm
import json

code_dir = "/data0/xjh/code-solver/direct_solve_codes"

output_file = "/data0/xjh/code-solver/eval_direct_results.jsonl"

loader = LCBLoader()
problems = loader.load()
problems = sorted(problems, key=lambda x: x.problem_id)
# problems = problems[:100]

executor = Executor()

total = 0
right = 0

eval_results = []

for problem in tqdm(problems):
    print(f"test {problem.problem_id}")
    total += 1
    with open(os.path.join(code_dir, f"{problem.problem_id}.py"), "r") as f:
        code = f.read()
    suite_result = executor.run_suite(code, problem.private_tests)
    # for res in suite_result.results:
    #     print(res)
    if suite_result.all_passed:
        right += 1
        eval_results.append({
            "question_id": problem.problem_id,
            "is_correct": True,
        })
    else:
        eval_results.append({
            "question_id": problem.problem_id,
            "is_correct": False,
        })

with open(output_file, "w") as f:
    for result in eval_results:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")

print(f"pass rate: {right / total: .4f} {right}/{total}")