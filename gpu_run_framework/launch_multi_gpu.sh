#!/usr/bin/env bash
set -euo pipefail

# How many GPUs do we have?
NGPU=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
echo "Detected $NGPU GPUs"

# Shard config: 100 shards of 1000 runs each = 100k
TOTAL_SHARDS=100
RUNS_PER_SHARD=1000

# Assign contiguous shard ranges to each GPU
# Example: 4 GPUs → GPU0 gets 1-25, GPU1 26-50, GPU2 51-75, GPU3 76-100
PER_GPU=$(( (TOTAL_SHARDS + NGPU - 1) / NGPU ))

run_on_gpu () {
  local gpu=$1
  local start=$2
  local end=$3

  echo "[GPU ${gpu}] shards ${start}-${end}"
  for i in $(seq -w ${start} ${end}); do
    (
      cd run_shard_${i}
      export CUDA_VISIBLE_DEVICES=${gpu}
      # Unique seed per shard for reproducibility & variety
      SEED=$(( 1000 + 10#${i} ))
      echo "[GPU ${gpu}] Starting shard ${i} with seed ${SEED}"
      python -m cli autogen --runs ${RUNS_PER_SHARD} --task-split 50,30,20 --seed ${SEED}
      echo "[GPU ${gpu}] Finished shard ${i}"
    )
  done
}

pids=()
for g in $(seq 0 $((NGPU-1))); do
  S=$(( g*PER_GPU + 1 ))
  E=$(( (g+1)*PER_GPU ))
  if [ $E -gt $TOTAL_SHARDS ]; then E=$TOTAL_SHARDS; fi
  [ $S -le $E ] || continue
  run_on_gpu $g $(printf "%03d" $S) $(printf "%03d" $E) &
  pids+=($!)
done

# Wait for all GPU workers
for p in "${pids[@]}"; do wait $p; done

echo "All shards complete."
