# Matrix multiplication: CPU to optimized CUDA

Status: all four required versions are implemented: `cpu` and `naive` (Week 1), `tiled` (Week 2),
`padded` (Week 3), with tile sizes 8/16/32. Two Week 3 experiments, `tiled_ta` and `padded_ta`
(A tile stored transposed in shared memory), show the access pattern padding fixes.
Optional fifth version: `regtile` (register tiling, 4×4 outputs per thread), the Week 3 second
profiler-guided improvement; implemented, not yet measured. `cublas` (cuBLAS SGEMM) is included only as an
external performance reference; it is not one of the implementations. Extension: `vec` / `vec128`
(regtile + float4 global and shared-memory access; 64×64 / 128×128 block tiles), see
[docs/week4.md](docs/week4.md) §4e.

Weekly write-ups: [docs/week1.md](docs/week1.md) (CPU baseline, naive kernel, tests),
[docs/week2.md](docs/week2.md) (tiling, tile sizes, profiling plan) and
[docs/week3.md](docs/week3.md) (padding, bank-conflict analysis, experiments) and
[docs/week4.md](docs/week4.md) (final runs, cuBLAS comparison, report plan). Git tags `week1`
and `week2` mark the code as it was at the end of each week.

## Reproduce everything (one command)
    bash scripts/reproduce.sh                              # or: make reproduce
    ARCH=sm_75 SHAPE6="M K N" bash scripts/reproduce.sh    # pin the GPU arch; add the 6th shape

Build → correctness tests → benchmarks (all versions + cuBLAS) → edge sweep → Nsight Compute → tables → plots.
Each step logs to `results/logs/`; `SKIP_CPU=1` / `SKIP_PROFILE=1` skip the slow CPU baseline / the profiler.
On Google Colab, open [notebooks/colab_reproduce.ipynb](notebooks/colab_reproduce.ipynb) and run all cells:
it clones a clean copy, saves results to Google Drive as they are written, and downloads a zip at the end.

## Layout
    Makefile         builds ./matmul and ./compare in the repo root
    src/             matmul.cu (all implementations + CLI), compare.cpp
    scripts/         reproduce.sh (everything), gen.py, check_cpu.py, run_tests.sh, bench.sh,
                     edge_sweep.sh, profile.sh, bank_model.py (predicted bank conflicts),
                     summarize_profile.py, extract_rules.py, plot_results.py
    notebooks/       colab_reproduce.ipynb (the full run on Google Colab)
    docs/            weekly notes (week1.md ... week4.md)
    results/         benchmark CSVs, profiler output, plots (created by the scripts)

## Input / output format
Input (native little-endian, no padding): `int32 M, K, N`, then `float32 A[M*K]`
and `float32 B[K*N]`, both row-major. Output: exactly `M*N` float32 values,
row-major, no header. Element `A[i][j]` lives at index `i*K + j`.

## Build
    make              # builds ./matmul (nvcc, links cuBLAS for the reference) and ./compare (g++)
    make ARCH=sm_75   # if -arch=native is not supported by your CUDA version

## Use
    python3 scripts/gen.py 1000 513 769 42 in.bin          # M K N SEED OUT
    ./matmul --impl cpu   --input in.bin --output ref.bin
    ./matmul --impl naive --input in.bin --output out.bin
    ./matmul --impl tiled  --tile 16 --input in.bin --output out2.bin  # tile = 8, 16 or 32
    ./matmul --impl padded --tile 32 --input in.bin --output out3.bin  # shared arrays [T][T+1]
    ./matmul --impl regtile --input in.bin --output out4.bin           # register tiling (--tile ignored)
    ./matmul --impl vec     --input in.bin --output out6.bin           # + float4 access (vec128: bigger tiles)
    ./matmul --impl cublas  --input in.bin --output out5.bin           # cuBLAS SGEMM reference
    ./compare in.bin ref.bin out.bin               # exit 0 = PASS

## Results (Tesla T4, Colab, final run 13 Nov 2026)
All 327 correctness checks pass (`results/tests.txt`). Kernel-only, best of each family:

| | 1024³ | 4096³ | 2048×512×4096 | 1000×513×769 |
|---|---|---|---|---|
| naive | 305 GFLOP/s | 415 | 475 | 773 |
| tiled (T = 32) | 1041 | 816 | 1029 | 673 |
| padded (T = 32) | 804 | 634 | 755 | 532 |
| regtile (4×4 per thread) | 3532 | 2879 | 3829 | 2044 |
| vec128 (8×8 per thread, float4) | **4661** | **5396** | **5815** | **2088** |
| cuBLAS (reference) | 6162 | 6593 | 7039 | 3681 |

- Register tiling is the biggest single step: **3.0–3.7×** over `tiled`.
- `vec128` reaches **76–83% of cuBLAS** on the large shapes.
- `padded` is 21–31% *slower* than `tiled`: the tiled kernel has no bank conflicts to remove, and the odd
  row length disables 128-bit shared loads.

Full analysis: [docs/week4.md](docs/week4.md). Plots: `results/plots/`. Nsight rule messages for every
kernel: `results/profile/rules_summary.md`.

## Tests and benchmarks
    make test      # correctness suite (CPU vs NumPy, GPU vs CPU)
    make bench     # results/results.csv + results/env.txt   (SHAPE6="M K N" adds the 6th shape)
    bash scripts/edge_sweep.sh   # results/edge_sweep.csv: n-1, n, n+1 for n = 1024, 4096
    make profile   # Nsight Compute reports + metric CSVs + SASS in results/profile/ (needs ncu)
    python3 scripts/bank_model.py   # predicted bank conflicts for every variant
    python3 scripts/summarize_profile.py > results/profile_summary.csv   # one table from all ncu CSVs
    python3 scripts/extract_rules.py > results/profile/rules_summary.md  # Nsight rule messages (needs *.details.txt)
    python3 scripts/plot_results.py   # SVG plots in results/plots/

CPU loop order is i-k-j (see [docs/week1.md](docs/week1.md) for the locality argument).
Tolerance: `abs_err <= 1e-4*K + 1e-4*|ref|`.
