#!/usr/bin/env bash
# Nsight Compute profiling: one large divisible shape + one irregular shape.
# For every (impl, tile, shape) this writes
#   results/profile/<name>.ncu-rep   full report (open in the Nsight Compute GUI)
#   results/profile/<name>.csv       selected metrics (for the report tables)
#
#   bash scripts/profile.sh
#   IMPLS="tiled padded" TILES="8 16 32" bash scripts/profile.sh
#
# Metric names can differ between CUDA / Nsight Compute versions. If the CSV step
# complains about an unknown metric, list valid ones with:  ncu --query-metrics
# (and look for bank_conflicts, sectors, warps_active, ...), then edit METRICS below.
set -u
cd "$(dirname "$0")/.."   # run from the repo root
command -v ncu >/dev/null || { echo "error: ncu (Nsight Compute) not found in PATH"; exit 1; }
IMPLS=${IMPLS:-"naive tiled padded tiled_ta padded_ta regtile vec vec128 cublas"}
TILES=${TILES:-"8 16 32"}
SEED=42
OUT=results/profile
mkdir -p "$OUT" data

SHAPES=("1024 1024 1024" "1000 513 769")   # large divisible, irregular

METRIC_LIST=(
  gpu__time_duration.sum                                           # kernel time
  sm__throughput.avg.pct_of_peak_sustained_elapsed                 # compute throughput %
  gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed # memory throughput %
  sm__warps_active.avg.pct_of_peak_sustained_active                # achieved occupancy %
  l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum                  # global load requests
  l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum                   # global load sectors
  smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct    # global load efficiency
  l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum                   # global store sectors
  l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum         # shared load bank conflicts
  l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum         # shared store bank conflicts
  l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum             # shared load wavefronts
  l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum             # shared store wavefronts
  smsp__inst_executed_op_shared_ld.sum                             # shared load instructions (vectorized?)
  smsp__inst_executed_op_shared_st.sum                             # shared store instructions
  dram__bytes_read.sum
  dram__bytes_write.sum
)
METRICS=$(IFS=,; echo "${METRIC_LIST[*]%% *}")

for s in "${SHAPES[@]}"; do
  read -r M K N <<<"$s"
  f="data/in_${M}_${K}_${N}.bin"
  if [ ! -f "$f" ]; then python3 scripts/gen.py "$M" "$K" "$N" "$SEED" "$f"; fi
  for impl in $IMPLS; do
    tiles="16"
    case "$impl" in tiled|padded|tiled_ta|padded_ta) tiles="$TILES" ;; esac
    for t in $tiles; do
      name="${impl}_T${t}_${M}x${K}x${N}"
      case "$impl" in naive|regtile|vec|vec128|cublas) name="${impl}_${M}x${K}x${N}" ;; esac
      # our kernels are all named matmul_*; cuBLAS kernel names contain "gemm"
      kfilter="regex:matmul_"; [ "$impl" = cublas ] && kfilter="regex:gemm"
      echo "== profiling $name"
      # one normal (non --bench) run = exactly one kernel launch
      ncu --set full --force-overwrite -k "$kfilter" -c 1 -o "$OUT/$name" \
          ./matmul --impl "$impl" --tile "$t" --input "$f" --output data/profile_out.bin >/dev/null
      # text export of the details page (all sections + Nsight's rule messages)
      ncu --import "$OUT/$name.ncu-rep" --page details >"$OUT/$name.details.txt" 2>&1
      ncu --csv --log-file "$OUT/$name.csv" --metrics "$METRICS" -k "$kfilter" -c 1 \
          ./matmul --impl "$impl" --tile "$t" --input "$f" --output data/profile_out.bin >/dev/null
    done
  done
done
# SASS of all kernels: shows whether the compiler merged the As[ty][k] reads into
# 128-bit LDS.128 loads (possible without padding, impossible with an odd row length).
command -v cuobjdump >/dev/null && cuobjdump -sass ./matmul >"$OUT/matmul.sass" \
  && grep -c "LDS" "$OUT/matmul.sass" >/dev/null && echo "wrote $OUT/matmul.sass"
echo "done: see $OUT/"
