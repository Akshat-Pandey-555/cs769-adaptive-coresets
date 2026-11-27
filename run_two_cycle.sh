#!/usr/bin/env bash
set -u
cd "$(dirname "$0")"
mkdir -p logs_two_cycle
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 MALLOC_ARENA_MAX=2

if pgrep -f "^python3 -u (main|run_experiment)\.py" > /dev/null; then
  echo "Other training runs are still going. Wait for them to finish, then run this again."
  exit 1
fi

FREE_MB=$(free -m | awk 'NR==2{print $7}')
if [ "$FREE_MB" -lt 9000 ]; then
  echo "Only ${FREE_MB} MB of memory is free; the five runs need about 8000 MB. Close other programs first."
  exit 1
fi

for ARM in FULL B C C20 D1B; do
  if [ -f "results_two_cycle/${ARM}_seed1.npz" ]; then
    echo "skip $ARM (already finished)"
    continue
  fi
  nohup python3 -u main.py --arm "$ARM" --seed 1 --schedule two_cycle > "logs_two_cycle/${ARM}.log" 2>&1 &
  echo "started $ARM"
  sleep 20
done

echo "Five runs started. Progress: tail -n 1 logs_two_cycle/*.log"
echo "When all reach round 179: python3 analyze.py --schedule two_cycle"
