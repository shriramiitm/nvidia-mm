#!/usr/bin/env bash
# Performance across matrix sizes: square M = N = K for every step of the
# optimization ladder plus cuBLAS. Writes results/sweep.csv (same columns as results.csv).
#   bash scripts/sweep.sh
#   SIZES="256 1024 4096" IMPLS="tiled tile2d cublas" bash scripts/sweep.sh
# CPU versions only run up to CPU_MAX (the i-j-k loop takes minutes beyond that).
set -eu
cd "$(dirname "$0")/.."   # run from the repo root
SIZES=${SIZES:-"128 256 512 1024 2048"}   # 4096 comes from bench.sh (results.csv); plot_results merges it
IMPLS=${IMPLS:-"naive_uncoal naive tiled padded tile1d tile2d regtile vec vec128 cublas"}
CPU_IMPLS=${CPU_IMPLS:-"cpu_ijk cpu"}
CPU_MAX=${CPU_MAX:-512}
TILE=${TILE:-32}
TRIALS=${TRIALS:-10}   # supplementary curves; the 4096 ladder table uses bench.sh (20 trials)
WARMUP=${WARMUP:-3}
SEED=42
mkdir -p results data

echo "impl,M,K,N,tile,trials,kernel_ms,e2e_ms,kernel_gflops,e2e_gflops" >results/sweep.csv
for n in $SIZES; do
  f="data/in_${n}_${n}_${n}.bin"
  if [ ! -f "$f" ]; then python3 scripts/gen.py "$n" "$n" "$n" "$SEED" "$f"; fi
  impls="$IMPLS"
  if [ "$n" -le "$CPU_MAX" ]; then impls="$CPU_IMPLS $IMPLS"; fi
  for impl in $impls; do
    ./matmul --impl "$impl" --tile "$TILE" --input "$f" --bench --trials "$TRIALS" --warmup "$WARMUP" \
             --cpu-trials 1 | tee -a results/sweep.csv
  done
done
