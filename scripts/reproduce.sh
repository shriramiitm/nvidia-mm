#!/usr/bin/env bash
# Reproduce every result in the report from a clean checkout, in one command:
#   build -> hardware specs -> correctness tests -> benchmarks -> size sweep -> edge sweep
#   -> profiling -> tables -> plots
#
#   bash scripts/reproduce.sh
#   ARCH=sm_75 SHAPE6="256 16384 256" bash scripts/reproduce.sh
#   SKIP_PROFILE=1 bash scripts/reproduce.sh          # no Nsight Compute available
#   SKIP_CPU=1 bash scripts/reproduce.sh              # skip the slow CPU baseline in the benchmarks
#
# Every step logs to results/logs/<step>.txt. A failing step is reported and the
# script carries on, so one problem does not lose the rest of the run; the exit
# status is nonzero if any step failed.
set -u
cd "$(dirname "$0")/.."   # run from the repo root
ARCH=${ARCH:-native}
export SHAPE6=${SHAPE6:-}
mkdir -p results/logs
failed=()

step() {  # step NAME COMMAND...
  local name=$1; shift
  echo "==> $name"
  local t0=$SECONDS
  if "$@" >"results/logs/$name.txt" 2>&1; then
    echo "    ok ($((SECONDS - t0)) s) - log: results/logs/$name.txt"
  else
    echo "    FAILED ($((SECONDS - t0)) s) - last lines of results/logs/$name.txt:"
    tail -n 15 "results/logs/$name.txt" | sed 's/^/      /'
    failed+=("$name")
  fi
}

step build make ARCH="$ARCH"
if [ ! -x ./matmul ] || [ ! -x ./compare ]; then
  echo "build failed; nothing else can run"; exit 1
fi

step hw_specs sh -c 'python3 scripts/hw_specs.py > results/hw_specs.md && python3 scripts/hw_specs.py --json > results/hw_specs.json'
cat results/hw_specs.md 2>/dev/null | grep -E 'GPU \||FP32 peak|DRAM bandwidth|Ridge' | sed 's/^/    /'

step tests bash scripts/run_tests.sh
cp results/logs/tests.txt results/tests.txt
grep -E '^passed:' results/tests.txt | sed 's/^/    /'

if [ "${SKIP_CPU:-0}" = 1 ]; then
  step bench env IMPLS="naive_uncoal naive tiled padded tiled_ta padded_ta tile1d tile2d regtile vec vec128 cublas" bash scripts/bench.sh
else
  step bench bash scripts/bench.sh
fi
if [ "${SKIP_CPU:-0}" = 1 ]; then
  step sweep env CPU_IMPLS="" bash scripts/sweep.sh
else
  step sweep bash scripts/sweep.sh
fi
step edge_sweep bash scripts/edge_sweep.sh

if [ "${SKIP_PROFILE:-0}" != 1 ] && command -v ncu >/dev/null; then
  step profile bash scripts/profile.sh
  step profile_summary sh -c 'python3 scripts/summarize_profile.py results/profile > results/profile_summary.csv'
  step profile_rules sh -c 'python3 scripts/extract_rules.py results/profile > results/profile/rules_summary.md'
else
  echo "==> profile skipped (SKIP_PROFILE=1 or ncu not found)"
fi

step bank_model sh -c 'python3 scripts/bank_model.py > results/bank_model.txt'
step plots python3 scripts/plot_results.py results

echo
if [ ${#failed[@]} -eq 0 ]; then
  echo "all steps succeeded; results are in results/"
else
  echo "failed steps: ${failed[*]} (see results/logs/)"
  exit 1
fi
