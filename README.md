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

| Kernel | Time (ms) | TFLOPS | Bandwidth (GB/s) | vs cuBLAS | Speedup over naive |
|---|---|---|---|---|---|
| CPU (single thread, i-k-j) | 25,850 | 0.005 | 0.01 | 0.1% | 0.013× |
| Naive (coalesced, 16×16) | 331.4 | 0.41 | 0.61 | 6.3% | 1.0× |
| Shared-memory tiled (T=32) | 168.4 | 0.82 | 1.20 | 12.4% | 2.0× |
| Tiled + padding (T=32) | 216.6 | 0.63 | 0.93 | 9.6% | 1.5× |
| 2D register tiling, 4×4 (regtile) | 47.74 | 2.88 | 4.22 | 43.7% | 6.9× |
| + Vectorized float4 (vec) | 44.85 | 3.06 | 4.49 | 46.5% | 7.4× |
| Vectorized, 8×8 per thread (vec128) | 25.47 | 5.40 | 7.90 | **81.8%** | **13.0×** |
| cuBLAS SGEMM (reference) | 20.85 | 6.59 | 9.66 | 100% | 15.9× |

Bandwidth = (A + B + C bytes) / kernel time. All 327 correctness checks pass. All shapes, profiler evidence
and analysis are in the [report](docs/report.md).
