#!/usr/bin/env bash
set -u
cd "$(dirname "$0")"
mkdir -p logs_memory
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 MALLOC_ARENA_MAX=2

if pgrep -f "^python3 -u (main|run_experiment)\.py" > /dev/null; then
  echo "Other training runs are still going. Wait for them to finish, then run this again."
  exit 1
fi

FREE_MB=$(free -m | awk 'NR==2{print $7}')
if [ "$FREE_MB" -lt 7500 ]; then
  echo "Only ${FREE_MB} MB of memory is free; the four runs need about 6500 MB. Close other programs first."
  exit 1
fi

for SCHEDULE in one_cycle two_cycle; do
  FOLDER=results
  [ "$SCHEDULE" = two_cycle ] && FOLDER=results_two_cycle
  for ARM in BAL RES; do
    if [ -f "$FOLDER/${ARM}_seed1.npz" ]; then
      echo "skip $ARM $SCHEDULE (already finished)"
      continue
    fi
    nohup python3 -u main.py --arm "$ARM" --seed 1 --schedule "$SCHEDULE" > "logs_memory/${ARM}_${SCHEDULE}.log" 2>&1 &
    echo "started $ARM $SCHEDULE"
    sleep 20
  done
done

echo "Four runs started. Progress: tail -n 1 logs_memory/*.log"
echo "One-cycle runs end at round 104, two-cycle runs at round 179."
echo "Then: python3 analyze.py   and   python3 analyze.py --schedule two_cycle"
