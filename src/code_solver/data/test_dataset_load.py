from datasets import load_dataset

ds = load_dataset("json", data_files="/data0/xjh/code-solver/src/code_solver/data/livecodebench_all.jsonl")

print(ds["train"][0])
print(type(ds["train"][0]))

