# Week 4: final optimization, benchmarks and analysis

**Week 4 goals (from the spec):**
- Run the complete correctness and benchmark matrix.
- Compare all four implementations with the external cuBLAS reference.
- Analyse performance trends, edge effects and the remaining gap to cuBLAS.
- Finalize code, scripts, plots, profiler evidence and the report.

**Milestone:** all four versions and the reported results are reproducible from a clean checkout.

---

## 1. What was added for Week 4

| Piece | Where | Purpose |
|---|---|---|
| cuBLAS SGEMM reference | `--impl cublas` in `src/matmul.cu` | External performance yardstick only, not an implementation. Row-major C is computed as column-major Cᵀ = Bᵀ·Aᵀ in one `cublasSgemm` call; default (FP32) math mode; handle created before timing |
| 6th benchmark shape | `SHAPE6="M K N"` for `bench.sh` / `reproduce.sh`, `SHAPE6` in the notebook | Instructor-approved shape (section 3) |
| Edge-effect sweep | `scripts/edge_sweep.sh` → `results/edge_sweep.csv` | n − 1, n, n + 1 for n = 1024 and 4096 (multiples of every tile and of the 64×64 regtile block) |
| One-command reproduction | `scripts/reproduce.sh` (`make reproduce`) | Build → tests → bench → edge sweep → profile → tables → plots, with a log per step in `results/logs/` |
| Colab notebook | `notebooks/colab_reproduce.ipynb` | Clean clone (or Drive zip) → `reproduce.sh` → results saved to Drive while they're written → zip download |
| New plots | `scripts/plot_results.py` | `percent_of_cublas.svg` (gap to cuBLAS), `edge_sweep.svg`; the main throughput chart now includes regtile and cuBLAS |

The test suite now also checks `cublas` against the CPU reference, which validates the row-major ↔
column-major mapping.

---

## 2. The complete run (do this once, on the GPU)

1. Push the repo, then open `notebooks/colab_reproduce.ipynb` in Colab (File → Open notebook → GitHub,
   or upload it).
2. Choose **T4 GPU**. In the settings cell, set `CODE_SOURCE` and `SHAPE6`. For a private repo, add a
   Colab secret `GH_TOKEN` (fine-grained, read-only, this repo only).
3. Run all. Send back the downloaded `results.zip`, or the Drive folder `nvidia_mm_results/<date>`.

That one run is also the **clean-checkout reproducibility check** for the milestone: a fresh clone, one
script, every number.

---

## 3. The 6th shape (needs instructor approval)

Choose one hypothesis and get it approved, then set `SHAPE6`.

| Option | Shape (M, K, N) | Hypothesis it tests |
|---|---|---|
| A | 256, 16384, 256 | **Too little parallelism.** Only 64 blocks at T = 32 (16 for regtile) for 40 SMs, with a long K loop. Expect every kernel well below its square-shape GFLOP/s, regtile hit hardest, and cuBLAS using split-K to recover |
| B | 4096, 16, 4096 | **Almost no reuse along K.** Only 1 tile step (T = 32) or 2 (regtile); writing the 64 MB C matrix dominates. Expect tiled ≈ naive and all kernels near the store-bandwidth limit |
| C | 4097, 4097, 4097 | **Edge cost at scale.** One extra, almost-empty row and column of tiles; compare with 4096³ |

Chosen: *fill in* — approved by: *fill in*

---

## 4. Results (fill in from the complete run)

### 4a. Correctness (`results/tests.txt`)

| Implementation | Tile sizes | Shapes | Passed | Worst max_abs_err | Worst max_rel_err |
|---|---|---|---|---|---|
| naive | – | 16 | | | |
| tiled | 8, 16, 32 | 16 | | | |
| padded | 8, 16, 32 | 16 | | | |
| regtile | – | 16 | | | |
| cublas (reference) | – | 16 | | | |

### 4b. All four implementations + cuBLAS (kernel-only, median of 20)

| Shape | CPU (ms) | naive | tiled T=32 | padded T=32 | regtile | cuBLAS | Best ours / cuBLAS |
|---|---|---|---|---|---|---|---|
| 256³ | | | | | | | |
| 1024³ | | | | | | | |
| 4096³ | | | | | | | |
| 2048×512×4096 | | | | | | | |
| 1000×513×769 | | | | | | | |
| 6th shape | | | | | | | |

The end-to-end columns and the speedups are in `results/results.csv`. Plots: `gflops_by_impl.svg`,
`percent_of_cublas.svg`, `kernel_share_e2e.svg`.

### 4c. Edge effects (`results/edge_sweep.csv`, `edge_sweep.svg`)

| Kernel | 1023³ | 1024³ | 1025³ | 4095³ | 4096³ | 4097³ |
|---|---|---|---|---|---|---|
| tiled T=32 (GFLOP/s) | | | | | | |
| regtile | | | | | | |
| cuBLAS | | | | | | |

What to look for: n + 1 launches a whole extra row and column of blocks that are almost all padding.
At 1025 that's 33² vs 32² blocks for T = 32, i.e. 6% more blocks for 0.3% more work. Smaller,
less-occupied grids (1024³ has only 1024 blocks at T = 32 for 40 SMs) also make wave quantization
visible.

### 4d. Gap to cuBLAS: analysis checklist

- How far is the best hand-written kernel from cuBLAS on each shape?
- What does cuBLAS do that ours doesn't? Profile `cublas` and compare in `profile_summary.csv`. Likely:
  - larger register tiles (8×8 or more per thread), so fewer shared loads per FMA;
  - double-buffered shared memory (load the next tile while computing this one);
  - vectorized global loads;
  - a kernel chosen per shape.
- Which of these would help ours most, given the profile? Write it up as future work.

---

## 4e. Extension: vectorized (float4) memory access (`vec`, `vec128`)

Based on a vectorized SGEMM kernel you supplied (`sgemm_vectorize_kernel`), adapted to this
project's requirements. Code: `matmul_vec<BM, BN, BK, TM, TN>` in `src/matmul.cu`.

**What it adds on top of `regtile`:** every memory access is 128-bit where that's legal:
- global loads of A and B;
- the shared-memory store of B;
- the shared-memory reads of A and B;
- the global store of C.

| `--impl` | Block tile | Per thread | BK | Threads | Changes vs regtile |
|---|---|---|---|---|---|
| `vec` | 64 × 64 | 4 × 4 | 8 | 256 | **only** float4 access, so it isolates the effect of vectorization |
| `vec128` | 128 × 128 | 8 × 8 | 8 | 256 | float4 access **and** twice the tile in each direction (the supplied kernel's usual config) |

**Changes from the supplied kernel, and why:**

| Supplied kernel | Problem here | What `matmul_vec` does |
|---|---|---|
| No bounds checks | Reads and writes out of bounds when M, N or K isn't a tile multiple (required: any 1 ≤ M, N, K ≤ 16384) | float4 only when all 4 elements are inside; otherwise element-by-element with zero fill; guarded C stores |
| Always `float4` | Misaligned (illegal) when K or N isn't a multiple of 4, since row r of A starts at r·K floats | `float4` only if `K % 4 == 0` (A rows) / `N % 4 == 0` (B, C rows); scalar path otherwise |
| `C = α·AB + β·C`, reading C | C is uninitialised `cudaMalloc` memory; with β = 0, 0 × NaN = NaN | Plain `C = AB`, C is only written |
| `tile_a[col * BM + row]`, no padding | Transposed store is a **2-way bank conflict** (bank model) | Row padded by 4 → conflict-free and still 16-byte aligned |
| Thread owns 8 adjacent columns (TN = 8) | float4 reads 32 B apart: **8 instead of 4 wavefronts** per read (bank model) | Two groups of 4 columns, BN/2 apart (same for rows) |
| One float4 of A and one of B per thread | Holds only for BM·BK/4 = BK·BN/4 = 256 | One loop over all float4 slots of both tiles, so it works for both configs |

**Checked without a GPU:**
- A Python emulation of the kernel as written matches a reference result for both configurations on
  9 shapes. That covers the all-float4 case (64³), mixed paths (130×132×260, 129×64×130) and all-scalar
  cases (65×63×129, 1×1×1).
- The bank model gives 1-way stores and 4 wavefronts per float4 read.
- clang parses it cleanly.
- `run_tests.sh` now covers `vec` and `vec128` and has two new shapes, 130×132×260 and 129×64×130.

**Predictions:**
- `vec` vs `regtile` on aligned shapes (256³, 1024³, 4096³, 2048×512×4096): 4× fewer global load
  instructions and wider C stores. regtile wasn't global-memory-bound on paper, so expect a **modest gain
  (≈1.0–1.3×)**. Its profile shows which unit limits it.
- **On 1000×513×769, `vec` ≈ `regtile`.** K = 513 and N = 769 are odd, so no row is 16-byte aligned and
  every access takes the scalar path. That's a real limitation of float4 access on unaligned shapes, not a
  bug. A fix (padding the matrices on the device) is future work.
- `vec128`: half the shared loads per FMA of `vec` (4 float4 reads per 64 FMAs), but ~2× the registers per
  thread (lower occupancy), and only 4 blocks at 256³ for 40 SMs. Expect **faster on 4096³ and
  2048×512×4096, slower on 256³**.

---

## 5. Final report plan (5–8 pages, `docs/report.md` → PDF)

| Section | Source material | Status |
|---|---|---|
| 1. Problem, environment, methodology (timing, trials, tolerance) | week1 §4–5, `results/env.txt` | ready |
| 2. CPU baseline: loop order and locality | week1 §2 | ready |
| 3. Naive: thread mapping, coalescing diagram, 56% load efficiency | week1 §3, week2 §7 | ready, needs a diagram |
| 4. Tiled: pseudocode, cooperative loads, sync, edges, tile sizes, profiler evidence | week2 | ready, needs a diagram |
| 5. Padded: bank mapping, measured conflicts, why it is slower | week3 §2–7 | ready, needs a bank diagram |
| 6. Register tiling and vectorized access (regtile, vec, vec128) | week3 §8, §4e here | **needs the GPU run** |
| 7. Results: all versions + cuBLAS, plots, edge effects, 6th shape | §4 here | **needs the GPU run** |
| 8. Unsuccessful experiments and lessons | week3 §7, week2 §8 entry 2 | ready |
| 9. Gap to cuBLAS and future work | §4d | **needs the GPU run** |

---

## 6. Week 4 milestone checklist

1. ☐ 6th shape chosen and approved (section 3).
2. ☐ Complete run with the notebook from a clean clone; `results/` committed.
3. ☐ All tests pass (`results/tests.txt`), including regtile and cuBLAS.
4. ☐ Tables 4a–4c filled in; gap to cuBLAS analysed.
5. ☐ Final report written (5–8 pages) with pseudocode, mapping and memory-access diagrams, plots and profiler evidence.
6. ☐ README reproduction steps checked by following them exactly.
