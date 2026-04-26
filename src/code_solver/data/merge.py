input_files = ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl", "test6.jsonl"]
output_file = "livecodebench_all.jsonl"

with open(output_file, "w", encoding="utf-8") as fout:
    for file in input_files:
        with open(file, "r", encoding="utf-8") as fin:
            for line in fin:
                fout.write(line)