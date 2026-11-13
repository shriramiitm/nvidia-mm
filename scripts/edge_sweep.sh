#!/usr/bin/env bash
# Edge-effect sweep: square shapes n-1, n, n+1 around tile multiples.
# n = 1024 and 4096 are multiples of every tile size (8, 16, 32) and of the
# 64x64 regtile block, so n+1 adds a whole extra row and column of mostly-empty
# tiles, and n-1 leaves one idle thread row/column in every edge block.
# Writes results/edge_sweep.csv (same columns as results.csv).
#   bash scripts/edge_sweep.sh
#   NS="1024 2048" IMPLS="tiled regtile" bash scripts/edge_sweep.sh
set -eu
cd "$(dirname "$0")/.."   # run from the repo root
NS=${NS:-"1024 2048"}   # add 4096 for the large-size edge (slow)
IMPLS=${IMPLS:-"naive tiled padded tile1d tile2d regtile vec vec128 cublas"}
TILE=${TILE:-32}
TRIALS=${TRIALS:-20}
WARMUP=${WARMUP:-3}
SEED=42
mkdir -p results data

echo "impl,M,K,N,tile,trials,kernel_ms,e2e_ms,kernel_gflops,e2e_gflops" >results/edge_sweep.csv
for n in $NS; do
  for m in $((n - 1)) "$n" $((n + 1)); do
    f="data/in_${m}_${m}_${m}.bin"
    if [ ! -f "$f" ]; then python3 scripts/gen.py "$m" "$m" "$m" "$SEED" "$f"; fi
    for impl in $IMPLS; do
      ./matmul --impl "$impl" --tile "$TILE" --input "$f" --bench --trials "$TRIALS" --warmup "$WARMUP" \
        | tee -a results/edge_sweep.csv
    done
  done
done
