import json

log_file = "results/codetree_plus_qwen7b.jsonl"

data = []
with open(log_file, "r") as f:
    for line in f:
        sample = json.loads(line)
        data.append(sample)

ac_count = 0
diversity_strategy = 0
for sample in data:
    if sample["passed_private"]:
        ac_count += 1
        if len(sample["search_stats"]["explored_paradigms"]) > 1:
            diversity_strategy += 1

print(ac_count, diversity_strategy)
