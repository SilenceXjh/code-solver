for start in $(seq 0 100 800); do
  end=$((start + 100))
  nohup python src/code_solver/run.py --model deepseek-v4-flash --api-base https://api.deepseek.com --no-difficulty --no-fault-localizer --no-adversarial --width 1 --start-index "$start" --end-index "$end" --run-name "run_ds_w1_${start}_${end}" > "run_ds_w1_${start}_${end}.log" 2>&1 &
done

nohup python src/code_solver/run.py --model deepseek-v4-flash --api-base https://api.deepseek.com --no-difficulty --no-fault-localizer --no-adversarial --width 1 --start-index 900 --end-index 1055 --run-name run_ds_w1_900_1055 > run_ds_w1_900_1055.log 2>&1 &
