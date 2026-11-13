# Matrix multiplication: CPU to optimized CUDA

C = A·B for row-major FP32 matrices of any size (1 ≤ M, N, K ≤ 16384), implemented in progressively
optimized versions and analysed with Nsight Compute on a Tesla T4.

- **Final report:** [docs/report.md](docs/report.md)
- **Step-by-step optimization log:** [docs/optimization_journey.md](docs/optimization_journey.md)

## Implementations (`src/matmul.cu`)

| `--impl` | Version |
|---|---|
| `cpu` | **Required 1.** Single-threaded CPU, i-k-j loop order (correctness reference) |
| `naive` | **Required 2.** One thread per output; 2D grid and blocks; coalesced (`threadIdx.x` → column) |
| `tiled` | **Required 3.** Shared-memory tiling, `--tile 8 \| 16 \| 32` |
| `padded` | **Required 4.** `tiled` with shared arrays `[T][T+1]` |
| `cpu_ijk`, `naive_uncoal` | Ladder steps: CPU in i-j-k order; naive with uncoalesced mapping |
| `tile1d`, `tile2d` | Ladder steps: 8 outputs per thread (1D); 8×8 outputs per thread (2D) |
| `regtile`, `vec`, `vec128` | Extensions: register tiling 4×4; + `float4` access; 8×8 with `float4` |
| `tiled_ta`, `padded_ta` | Experiment: transposed A tile, showing the access pattern padding fixes |
| `cublas` | cuBLAS SGEMM, external performance reference only |

## Build and run

    make                       # ./matmul (nvcc + cuBLAS), ./compare, ./devinfo;  make ARCH=sm_75 if needed
    python3 scripts/gen.py 1000 513 769 42 in.bin                         # M K N SEED OUT (values in [-1, 1])
    ./matmul --impl cpu   --input in.bin --output ref.bin
    ./matmul --impl tiled --tile 32 --input in.bin --output out.bin
    ./compare in.bin ref.bin out.bin                                      # exit 0 = PASS

Input file: `int32 M, K, N`, then `float32 A[M*K]`, `float32 B[K*N]`, row-major, little-endian.
Output file: `M*N` float32 values, row-major, no header. Bad arguments, unsupported tile sizes and CUDA
errors give a clear message and a nonzero exit status. Tolerance: `|out − ref| ≤ 1e-4·K + 1e-4·|ref|`.

## Reproduce all results

    bash scripts/reproduce.sh                              # or: make reproduce
    ARCH=sm_75 SHAPE6="256 16384 256" bash scripts/reproduce.sh

This runs: hardware specs → correctness tests → benchmarks → size sweep → edge sweep → Nsight Compute →
tables → plots. Each step logs to `results/logs/`. `SKIP_CPU=1` / `SKIP_PROFILE=1` skip the slowest parts.
On Google Colab, run [notebooks/colab_reproduce.ipynb](notebooks/colab_reproduce.ipynb): it clones a clean
copy, saves results to Google Drive as they're written, and downloads a zip.

| Script | Produces |
|---|---|
| `scripts/run_tests.sh` | `results/tests.txt`: every version vs CPU on 18 shapes, plus error handling |
| `scripts/bench.sh` | `results/results.csv`, `results/env.txt`: required shapes, median of 20 |
| `scripts/sweep.sh`, `scripts/edge_sweep.sh` | `results/sweep.csv`, `results/edge_sweep.csv` |
| `scripts/profile.sh` | `results/profile/`: `ncu` reports, metric CSVs, details pages, SASS (command lines inside) |
| `scripts/hw_specs.py` | `results/hw_specs.md`: device limits, FP32 peak, DRAM bandwidth, ridge point |
| `scripts/summarize_profile.py`, `extract_rules.py` | `results/profile_summary.csv`, `results/profile/rules_summary.md` |
| `scripts/plot_results.py`, `make_figures.py` | `results/plots/*.svg`, `docs/figures/*.svg` |
| `scripts/bank_model.py`, `check_cpu.py` | predicted bank conflicts; CPU vs float64 NumPy check |

## Results (Tesla T4, 4096 × 4096 × 4096, kernel-only, median of 20)

| Step | Kernel | Time (ms) | TFLOPS | Bandwidth (GB/s) | vs cuBLAS | Speedup over naive | Over previous step |
|---|---|---|---|---|---|---|---|
| 2 | CPU, i-k-j | 26,470 | 0.005 | 0.008 | 0.1% | 0.043× | – |
| 3 | Naive GPU (uncoalesced) | 1143 | 0.12 | 0.18 | 1.8% | 1.0× | – |
| 4 | Global-memory coalesced (required naive) | 329.9 | 0.42 | 0.61 | 6.2% | 3.5× | 3.46× |
| 5 | Shared memory, T=32 (required tiled) | 163.5 | 0.84 | 1.23 | 12.5% | 7.0× | 2.02× |
| 5b | Shared + padding, T=32 (required padded) | 204.5 | 0.67 | 0.99 | 10.0% | 5.6× | 0.80× |
| 6 | 1D block tiling (8×1) | 73.15 | 1.88 | 2.75 | 27.9% | 15.6× | 2.24× |
| 7 | 2D block tiling (8×8) | 35.40 | 3.88 | 5.69 | 57.7% | 32.3× | 2.07× |
| 8 | Vectorized (`float4`, 8×8) | 24.56 | **5.60** | 8.20 | **83.2%** | **46.5×** | 1.44× |
| – | cuBLAS SGEMM (reference) | 20.44 | 6.72 | 9.85 | 100% | 55.9× | – |

Bandwidth = (A + B + C bytes) / kernel time. All 399 correctness checks pass. Every shape, the size sweep,
the roofline and the profiler evidence are in the [report](docs/report.md) and the
[optimization log](docs/optimization_journey.md).
