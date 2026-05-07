for start in $(seq 0 100 900); do
  if [ "$start" -eq 900 ]; then
    end=1055
  else
    end=$((start + 100))
  fi

  run_name="run_qwen_full_${start}_${end}"
  nohup python src/code_solver/run.py --model qwen7b --api-base http://localhost:8000/v1 --no-difficulty --no-fault-localizer --no-adversarial --start-index "$start" --end-index "$end" --run-name "$run_name" > "${run_name}.log" 2>&1 &
done
