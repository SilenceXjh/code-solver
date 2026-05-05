for start in $(seq 0 100 800); do
  end=$((start + 100))
  nohup python src/code_solver/run.py --model qwen7b --api-base http://localhost:8000/v1 --no-difficulty --no-fault-localizer --no-adversarial --width 1 --start-index "$start" --end-index "$end" --run-name "run_qwen7b_w1_${start}_${end}" > "run_qwen7b_w1_${start}_${end}.log" 2>&1 &
done

nohup python src/code_solver/run.py --model qwen7b --api-base http://localhost:8000/v1 --no-difficulty --no-fault-localizer --no-adversarial --width 1 --start-index 900 --end-index 1055 --run-name run_qwen7b_w1_900_1055 > run_qwen7b_w1_900_1055.log 2>&1 &
