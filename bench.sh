#!/usr/bin/env bash
# Benchmark the required shapes; writes results/results.csv and results/env.txt.
#   bash bench.sh
#   IMPLS="cpu naive tiled" TILES="8 16 32" TRIALS=20 bash bench.sh
set -eu
cd "$(dirname "$0")"
IMPLS=${IMPLS:-"cpu naive tiled"}
TILES=${TILES:-"8 16 32"}
TRIALS=${TRIALS:-20}
WARMUP=${WARMUP:-3}
SEED=42
mkdir -p results data

{
  echo "== date";  date
  echo "== CPU";   lscpu | grep -E 'Model name|^CPU\(s\)'
  echo "== GPU";   nvidia-smi --query-gpu=name,driver_version,clocks.max.sm,power.limit --format=csv
  echo "== nvcc";  nvcc --version | tail -n 2
  echo "== flags"; grep -E '^(NVFLAGS|ARCH)' Makefile
} >results/env.txt 2>&1 || true

SHAPES=(
  "256 256 256"
  "1024 1024 1024"
  "4096 4096 4096"
  "2048 512 4096"
  "1000 513 769"
  # add your instructor-approved shape here
)

echo "impl,M,K,N,tile,trials,kernel_ms,e2e_ms,kernel_gflops,e2e_gflops" >results/results.csv
for s in "${SHAPES[@]}"; do
  read -r M K N <<<"$s"
  f="data/in_${M}_${K}_${N}.bin"
  if [ ! -f "$f" ]; then python3 gen.py "$M" "$K" "$N" "$SEED" "$f"; fi
  work=$((M * K * N))
  cpu_trials=3
  if [ "$work" -gt 2000000000 ]; then cpu_trials=1; fi   # big CPU runs take minutes
  for impl in $IMPLS; do
    tiles="16"   # ignored by cpu / naive
    case "$impl" in tiled|padded) tiles="$TILES" ;; esac
    for t in $tiles; do
      ./matmul --impl "$impl" --tile "$t" --input "$f" --bench --trials "$TRIALS" \
               --warmup "$WARMUP" --cpu-trials "$cpu_trials" | tee -a results/results.csv
    done
  done
done
