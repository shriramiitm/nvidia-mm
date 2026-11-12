#!/usr/bin/env bash
# Automated correctness suite: for every shape, generate input (fixed seed),
# run the CPU reference, check the CPU against NumPy, then compare every GPU
# implementation (and every tile size) against the CPU output.
#   bash scripts/run_tests.sh   # default: naive tiled padded (+ experiments tiled_ta padded_ta)
#   IMPLS="naive" bash scripts/run_tests.sh             # Week 1 only
#   TILES="8 16 32" bash scripts/run_tests.sh           # tile sizes for tiled/padded
set -u
cd "$(dirname "$0")/.."   # run from the repo root
IMPLS=${IMPLS:-"naive tiled padded tiled_ta padded_ta regtile"}
TILES=${TILES:-"8 16 32"}
SEED=42
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

SHAPES=(
  "64 64 64"      # square
  "200 16 8"      # tall
  "8 16 200"      # wide
  "5 3 7"         # smaller than a CUDA block
  "16 16 16"      # divisible by tile 16
  "32 32 32"      # divisible by tile 32
  "7 9 9"         # around tile 8
  "15 17 16"      # one less / one more than a multiple of 16
  "31 31 31"      # one less than a multiple of 32
  "33 33 33"      # one more than a multiple of 32
  "1 64 64"       # M = 1
  "64 1 64"       # K = 1
  "64 64 1"       # N = 1
  "1 1 1"
  "1000 513 769"  # irregular
  "65 63 129"     # one more / less than the 64x64 regtile block; K not a multiple of 8
)

pass=0
fail=0
report() {  # report STATUS LABEL DETAIL
  if [ "$1" = PASS ]; then pass=$((pass + 1)); else fail=$((fail + 1)); fi
  printf '[%s] %-40s %s\n' "$1" "$2" "$3"
}

for s in "${SHAPES[@]}"; do
  read -r M K N <<<"$s"
  shape="(M=$M K=$K N=$N)"
  if ! python3 scripts/gen.py "$M" "$K" "$N" "$SEED" "$TMP/in.bin"; then
    report FAIL "generate $shape" ""; continue
  fi
  if ! ./matmul --impl cpu --input "$TMP/in.bin" --output "$TMP/ref.bin" >/dev/null; then
    report FAIL "cpu run $shape" ""; continue
  fi
  out=$(python3 scripts/check_cpu.py "$TMP/in.bin" "$TMP/ref.bin") \
    && report PASS "cpu vs numpy $shape" "$out" || report FAIL "cpu vs numpy $shape" "$out"

  for impl in $IMPLS; do
    tiles="16"   # ignored by naive
    case "$impl" in tiled|padded|tiled_ta|padded_ta) tiles="$TILES" ;; esac
    for t in $tiles; do
      label="$impl"
      case "$impl" in tiled|padded|tiled_ta|padded_ta) label="$impl T=$t" ;; esac
      if ! ./matmul --impl "$impl" --tile "$t" --input "$TMP/in.bin" --output "$TMP/out.bin" >/dev/null; then
        report FAIL "$label run $shape" "program failed"; continue
      fi
      out=$(./compare "$TMP/in.bin" "$TMP/ref.bin" "$TMP/out.bin") \
        && report PASS "$label vs cpu $shape" "$out" || report FAIL "$label vs cpu $shape" "$out"
    done
  done
done

# Argument / error handling must give a nonzero exit status.
if ./matmul --impl bogus --input "$TMP/in.bin" --output "$TMP/x.bin" 2>/dev/null; then
  report FAIL "reject bad --impl" ""; else report PASS "reject bad --impl" ""; fi
if ./matmul --impl cpu --input /nonexistent.bin --output "$TMP/x.bin" 2>/dev/null; then
  report FAIL "reject missing input file" ""; else report PASS "reject missing input file" ""; fi
if ./matmul --impl tiled --input "$TMP/in.bin" --output "$TMP/x.bin" --tile 7 2>/dev/null; then
  report FAIL "reject bad --tile" ""; else report PASS "reject bad --tile" ""; fi

echo
echo "passed: $pass   failed: $fail"
[ "$fail" -eq 0 ]
