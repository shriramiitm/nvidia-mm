# Matrix multiplication: CPU to optimized CUDA

Status: all four required versions are implemented: `cpu` and `naive` (Week 1), `tiled` (Week 2),
`padded` (Week 3), with tile sizes 8/16/32. Two Week 3 experiments, `tiled_ta` and `padded_ta`
(A tile stored transposed in shared memory), show the access pattern padding fixes.
Optional fifth version: `regtile` (register tiling, 4×4 outputs per thread), the Week 3 second
profiler-guided improvement; implemented, not yet measured.

Weekly write-ups: [docs/week1.md](docs/week1.md) (CPU baseline, naive kernel, tests),
[docs/week2.md](docs/week2.md) (tiling, tile sizes, profiling plan) and
[docs/week3.md](docs/week3.md) (padding, bank-conflict analysis, experiments). Git tags `week1`
and `week2` mark the code as it was at the end of each week.

## Layout
    Makefile         builds ./matmul and ./compare in the repo root
    src/             matmul.cu (all implementations + CLI), compare.cpp
    scripts/         gen.py, check_cpu.py, run_tests.sh, bench.sh, profile.sh,
                     bank_model.py (predicted bank conflicts), summarize_profile.py, plot_results.py
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
    ./matmul --impl regtile --input in.bin --output out4.bin           # register tiling (--tile ignored)
    ./compare in.bin ref.bin out.bin               # exit 0 = PASS

## Results so far (Tesla T4, Colab, 12 Nov 2026)
Best kernel: `tiled` with T = 32, 681–1042 GFLOP/s, 1.2–2.9× faster than `naive` and 59–170× faster than the
single-thread CPU (kernel-only). `padded` is 1.20–1.32× *slower* than `tiled` at T = 32: the tiled kernel
has no bank conflicts to remove, and the odd row length disables 128-bit shared loads. Details in
[docs/week2.md](docs/week2.md) and [docs/week3.md](docs/week3.md); plots in `results/plots/`.

## Tests and benchmarks
    make test      # correctness suite (CPU vs NumPy, GPU vs CPU)
    make bench     # results/results.csv + results/env.txt
    make profile   # Nsight Compute reports + metric CSVs + SASS in results/profile/ (needs ncu)
    python3 scripts/bank_model.py   # predicted bank conflicts for every variant
    python3 scripts/summarize_profile.py > results/profile_summary.csv   # one table from all ncu CSVs
    python3 scripts/plot_results.py   # SVG plots in results/plots/

CPU loop order is i-k-j (see [docs/week1.md](docs/week1.md) for the locality argument).
Tolerance: `abs_err <= 1e-4*K + 1e-4*|ref|`.
