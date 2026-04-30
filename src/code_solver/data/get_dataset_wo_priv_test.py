import json

with open("src/code_solver/data/livecodebench_all.jsonl", "r") as f:
    lines = f.readlines()

problems = []
for line in lines[:100]:
    problem = json.loads(line)
    problem.pop("private_test_cases")
    problems.append(problem)

with open("src/code_solver/data/livecodebench_simple.json", "w") as f:
    json.dump(problems, f, ensure_ascii=False, indent=2)