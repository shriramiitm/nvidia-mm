# Matrix multiplication: CPU to optimized CUDA

Status: all four required versions are implemented: `cpu` and `naive` (Week 1), `tiled` (Week 2),
`padded` (Week 3), with tile sizes 8/16/32. Two Week 3 experiments, `tiled_ta` and `padded_ta`
(A tile stored transposed in shared memory), show the access pattern padding fixes.

Weekly write-ups: [docs/week1.md](docs/week1.md) (CPU baseline, naive kernel, tests),
[docs/week2.md](docs/week2.md) (tiling, tile sizes, profiling plan) and
[docs/week3.md](docs/week3.md) (padding, bank-conflict analysis, experiments). Git tags `week1`
and `week2` mark the code as it was at the end of each week.

## Layout
    Makefile         builds ./matmul and ./compare in the repo root
    src/             matmul.cu (all implementations + CLI), compare.cpp
    scripts/         gen.py, check_cpu.py, run_tests.sh, bench.sh, profile.sh,
                     bank_model.py (predicted shared-memory bank conflicts, no GPU needed)
    docs/            weekly notes (week1.md, week2.md, ...)
    results/         benchmark CSVs and profiler output (created by make bench / make profile)

## Input / output format
Input (native little-endian, no padding): `int32 M, K, N`, then `float32 A[M*K]`
and `float32 B[K*N]`, both row-major. Output: exactly `M*N` float32 values,
row-major, no header. Element `A[i][j]` lives at index `i*K + j`.

## Build
    make              # builds ./matmul (nvcc) and ./compare (g++)
    make ARCH=sm_75   # if -arch=native is not supported by your CUDA version

## Use
    python3 scripts/gen.py 1000 513 769 42 in.bin          # M K N SEED OUT
    ./matmul --impl cpu   --input in.bin --output ref.bin
    ./matmul --impl naive --input in.bin --output out.bin
    ./matmul --impl tiled  --tile 16 --input in.bin --output out2.bin  # tile = 8, 16 or 32
    ./matmul --impl padded --tile 32 --input in.bin --output out3.bin  # shared arrays [T][T+1]
    ./compare in.bin ref.bin out.bin               # exit 0 = PASS

## Tests and benchmarks
    make test      # correctness suite (CPU vs NumPy, GPU vs CPU)
    make bench     # results/results.csv + results/env.txt
    make profile   # Nsight Compute reports + metric CSVs + SASS in results/profile/ (needs ncu)
    python3 scripts/bank_model.py   # predicted bank conflicts for every variant

CPU loop order is i-k-j (see [docs/week1.md](docs/week1.md) for the locality argument).
Tolerance: `abs_err <= 1e-4*K + 1e-4*|ref|`.
