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

## 3. The 6th shape

| Option | Shape (M, K, N) | Hypothesis it tests |
|---|---|---|
| **A (used)** | **256, 16384, 256** | **Too little parallelism.** Only 64 blocks at T = 32 (16 for regtile, 4 for vec128) for 40 SMs, with a long K loop. Expect every kernel well below its square-shape GFLOP/s, the big-tile kernels hit hardest, and cuBLAS recovering by splitting K |
| B | 4096, 16, 4096 | **Almost no reuse along K.** Only 1 tile step (T = 32) or 2 (regtile); writing the 64 MB C matrix dominates |
| C | 4097, 4097, 4097 | **Edge cost at scale.** Covered by the edge sweep (4c) anyway |

Run with option A. Approved by the instructor: *fill in (if a different shape is required, re-run with
`SHAPE6="M K N" bash scripts/bench.sh`)*.

**Result: hypothesis confirmed** (table 4b, last row):

| Kernel | Blocks launched | GFLOP/s at 1024³ | GFLOP/s at 256×16384×256 | Change |
|---|---|---|---|---|
| tiled T=32 | 64 (1024 threads each) | 1041 | 850 | −18% |
| regtile | 16 | 3532 | 1114 | **−68%** |
| vec128 | **4** | 4661 | 457 | **−90%** |
| cuBLAS | (its own choice) | 6162 | 6359 | +3% |

With 16 or 4 blocks, most of the T4's 40 SMs sit idle for the whole kernel, however efficient each block
is. The bigger the per-block tile, the fewer blocks, so the "best" kernel on square shapes becomes the
worst here. cuBLAS is unaffected. It most likely splits the K = 16384 reduction across extra blocks
(split-K) or picks a different kernel for this shape; we didn't profile it at this shape, so this is not
verified. Our kernels have no such strategy. That's future work (split-K, or a smaller tile when the grid
is small).

---

## 4. Results (final run: Tesla T4, Colab, 13 Nov 2026, CUDA 13.0)

All numbers below come from **one complete run of `notebooks/colab_reproduce.ipynb` from a clean clone**.
Raw data: `results/results.csv`, `results/edge_sweep.csv`, `results/tests.txt`, `results/profile/`
(34 Nsight Compute reports with details pages), `results/profile_summary.csv`,
`results/profile/rules_summary.md` (every Nsight rule message, extracted by `scripts/extract_rules.py`).
The first, partial run (12 Nov) is kept in `results/archive/2026-11-12-first-run/` because Weeks 1–3
quote it; see 4g on run-to-run variation.

### 4a. Correctness (`results/tests.txt`): **327 / 327 passed**

18 shapes × (CPU vs NumPy + 17 GPU configurations) + 3 error-handling checks. The 18 shapes cover square,
tall, wide, smaller than a block, divisible by every tile, ±1 around tile multiples, M/K/N = 1,
1000×513×769, around the 64×64 regtile block, and float4-aligned/unaligned mixes.

| Implementation | Tile sizes | Shapes | Passed | Worst max_abs_err | Worst max_rel_err |
|---|---|---|---|---|---|
| cpu vs NumPy (float64) | – | 18 | 18 | 3.21e-5 | – |
| naive | – | 18 | 18 | 1.14e-5 | 9.6e-2 |
| tiled | 8, 16, 32 | 18 each | 54 | 1.14e-5 | 9.6e-2 |
| padded | 8, 16, 32 | 18 each | 54 | 1.14e-5 | 9.6e-2 |
| tiled_ta / padded_ta | 8, 16, 32 | 18 each | 108 | 1.14e-5 | 9.6e-2 |
| regtile, vec, vec128 | – | 18 each | 54 | 1.14e-5 | 9.6e-2 |
| cublas (reference) | – | 18 | 18 | 1.14e-5 | 9.6e-2 |

- The worst absolute error, at 1000×513×769, is **4500× below the tolerance** `1e-4·K + 1e-4·|ref|`
  (≥ 0.051 at K = 513).
- The large worst *relative* error (9.6%) is on an element whose reference value is almost zero
  (cancellation). That's why the spec's mixed tolerance has an absolute term. It passes comfortably.
- Every GPU version has the **same** worst error. That's consistent with all of them summing along k in
  the same increasing order with fused multiply-adds, so they likely produce identical results, while the
  CPU rounds differently (separate multiply and add). It wasn't verified bitwise, since the outputs aren't
  kept.

### 4b. All implementations + cuBLAS (kernel-only ms, GFLOP/s in brackets, median of 20)

| Shape | CPU | naive | tiled T=32 | padded T=32 | regtile | vec | vec128 | cuBLAS | Best ours as % of cuBLAS |
|---|---|---|---|---|---|---|---|---|---|
| 256³ | 4.909 | 0.0719 (467) | 0.0554 (605) | 0.0805 (417) | **0.0437 (768)** | 0.0471 (712) | 0.0955 (351) | 0.0303 (1107) | regtile 69% |
| 1024³ | 294 | 7.05 (305) | 2.063 (1041) | 2.672 (804) | 0.608 (3532) | 0.608 (3532) | **0.461 (4661)** | 0.349 (6162) | vec128 76% |
| 4096³ | 25,850 | 331.4 (415) | 168.4 (816) | 216.6 (634) | 47.74 (2879) | 44.85 (3064) | **25.47 (5396)** | 20.85 (6593) | vec128 82% |
| 2048×512×4096 | 1661 | 18.09 (475) | 8.345 (1029) | 11.38 (755) | 2.244 (3829) | 2.196 (3913) | **1.477 (5815)** | 1.220 (7039) | vec128 83% |
| 1000×513×769 | 105.9 | 1.021 (773) | 1.173 (673) | 1.482 (532) | 0.386 (2044) | 0.455 (1733) | **0.378 (2088)** | 0.214 (3681) | vec128 57% |
| 256×16384×256 | 279 | 4.684 (459) | 2.526 (850) | 3.260 (659) | **1.928 (1114)** | 2.082 (1032) | 4.704 (457) | 0.338 (6359) | regtile 18% |

Speedups (kernel-only):

| Shape | naive vs CPU | tiled vs CPU | best ours vs CPU | naive → tiled | tiled → padded | tiled → regtile | regtile → vec | vec → vec128 |
|---|---|---|---|---|---|---|---|---|
| 256³ | 68× | 89× | 112× | 1.30× | 0.69× | 1.27× | 0.93× | 0.49× |
| 1024³ | 42× | 143× | 638× | 3.42× | 0.77× | 3.39× | 1.00× | 1.32× |
| 4096³ | 78× | 153× | 1015× | 1.97× | 0.78× | 3.53× | 1.06× | 1.76× |
| 2048×512×4096 | 92× | 199× | 1124× | 2.17× | 0.73× | 3.72× | 1.02× | 1.49× |
| 1000×513×769 | 104× | 90× | 280× | 0.87× | 0.79× | 3.04× | 0.85× | 1.21× |
| 256×16384×256 | 60× | 110× | 145× | 1.85× | 0.77× | 1.31× | 0.93× | 0.44× |

Values below 1× are slowdowns. Plots: `gflops_by_impl.svg`, `extensions.svg`, `percent_of_cublas.svg`.

**End-to-end** (ms: allocation + copies + kernel + copy back + free):

| Shape | naive | tiled | padded | regtile | vec | vec128 | cuBLAS | vec128 kernel share |
|---|---|---|---|---|---|---|---|---|
| 256³ | 0.511 | 0.477 | 0.571 | 0.471 | 0.467 | 0.511 | 0.328 | 19% |
| 1024³ | 10.80 | 5.54 | 6.10 | 4.02 | 4.02 | 3.98 | 4.00 | 12% |
| 4096³ | 382.6 | 218.9 | 266.8 | 98.8 | 94.9 | 75.1 | 71.7 | 34% |
| 2048×512×4096 | 30.87 | 19.51 | 22.71 | 14.01 | 13.69 | 13.63 | 12.97 | 11% |
| 1000×513×769 | 3.13 | 3.28 | 3.67 | 2.50 | 2.59 | 2.49 | 2.21 | 15% |
| 256×16384×256 | 13.67 | 10.89 | 11.57 | 10.22 | 10.36 | 12.97 | 8.37 | 36% |

Once the kernel is fast, **end-to-end time is dominated by allocation and PCIe transfers.** At 1024³,
regtile, vec, vec128 and cuBLAS all finish in 4.0 ms end-to-end even though their kernels differ by 1.7×.
The kernel is only 11–36% of end-to-end time for vec128. For a single multiplication, everything
beyond regtile is invisible to the user except at 4096³. Kernel optimizations pay off when data stays on
the GPU across many operations.

### 4c. Edge effects (`results/edge_sweep.csv`, `edge_sweep.svg`): GFLOP/s

| Kernel | 1023³ | 1024³ | 1025³ | 4095³ | 4096³ | 4097³ | worst n±1 vs n |
|---|---|---|---|---|---|---|---|
| naive | 636 | 406 | 538 | 315 | 365 | 348 | (see note) |
| tiled T=32 | 1034 | 1033 | 847 | 724 | 781 | 730 | −18% |
| padded T=32 | 786 | 801 | 652 | 605 | 636 | 597 | −19% |
| regtile | 3402 | 3542 | 3123 | 2707 | 2893 | 2659 | −12% |
| vec | 2908 | 3541 | 2706 | 2314 | 3106 | 2132 | **−31%** |
| vec128 | 3293 | 4681 | 3004 | 4595 | 5562 | 4388 | **−36%** |
| cuBLAS | 4617 | 5223 | 5341 | 6391 | 6532 | 5732 | −12% |

- **n + 1 costs the tiled kernels up to 18–19%.** 1025 at T = 32 needs 33² blocks instead of 32² (6% more)
  and 33 K-steps instead of 32, while the extra work is 0.3%. The extra row and column of blocks is
  almost entirely zero-filled.
- **regtile loses at most 12%**: the same effect, spread over fewer, bigger blocks.
- **vec and vec128 lose up to 31–36% at both n − 1 and n + 1.** That's the float4 alignment limitation:
  1023, 1025, 4095 and 4097 are odd, so no row is 16-byte aligned and every access takes the scalar
  fallback. It's not the extra tiles: n − 1 has no extra tiles, and the loss is just as large. At n ± 1,
  vec is *slower* than regtile. 4f explains why the fallback is worse than regtile's plain loads.
- **cuBLAS varies ±12%** and is even faster at 1025 than at 1024.
- **Note on naive:** naive at exactly 1024 is 36% *slower* than at 1023, and its 1024³ time differed
  between runs (3.44, 5.29 and 7.05 ms). That suggests sensitivity to the power-of-two row stride
  (accesses aliasing onto the same memory partitions or cache sets). It's a hypothesis, not verified.
  Profiling naive at 1023 vs 1024 would settle it.

### 4d. Gap to cuBLAS: what the profiles show (1024³, `rules_summary.md`)

| | tiled T=32 | regtile | vec | vec128 | cuBLAS (`volta_sgemm_128x64_nn`) |
|---|---|---|---|---|---|
| Time under ncu (base clock 585 MHz) | 5.31 ms | 1.60 ms | 1.60 ms | 1.18 ms | 0.83 ms |
| Registers / thread | 44 | 64 | 54 | 123 | 122 |
| Theoretical / achieved occupancy | 100 / 100% | 100 / 84% | 100 / 84% | 50 / 36% | 50 / 47% |
| Grid (blocks × threads) | 1024 × 1024 | 256 × 256 | 256 × 256 | **64** × 256 | 640 × 128 |
| Waves (Nsight) | 25.6 | **1.6** | **1.6** | **0.8** | 4 |
| Compute (SM) throughput | 74% | 48% | 48% | 62% | **89%** |
| Busiest pipe (Nsight) | MIO (shared memory) | – (latency) | – (latency) | FMA 62% | **FMA 89%** |
| Warp cycles per issued instruction | 36.9 | 17.4 | 17.9 | 6.6 | 6.8 |
| of which waiting on MIO (shared-memory queue) | 24.0 (65%) | 7.6 (44%) | 8.0 (45%) | – | – |
| Executed IPC | 0.87 | 1.54 | 1.50 | 1.77 | 2.20 |
| Shared bank conflicts (ld / st) | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 38.8 K / 257 K |

**cuBLAS is compute-bound**: its FMA pipe is 89% busy, about as close to peak as an FP32 GEMM gets.
**vec128 is the first of ours where FMA is the busiest unit (62%).** The remaining gap has three measurable
causes:

1. **Too few blocks (wave quantization).** At 1024³, vec128 launches 64 blocks; at 2 blocks per SM that's
   0.8 of a wave on 40 SMs. Nsight reports only 1.6 resident blocks per SM and a 26% spread in SM active
   cycles. regtile and vec have 1.6 waves (one full wave and a 97-block tail), and Nsight estimates up
   to 50% loss from the tail. cuBLAS uses 640 smaller 128-thread blocks (4 waves), so the load balances.
2. **Lower issue efficiency.** At the same register count and occupancy limit (≈122 registers, 50%
   theoretical), cuBLAS issues 2.20 instructions per cycle against vec128's 1.77, and spends more of them
   on FMAs. Ours spend more on address arithmetic, bounds checks and the two `__syncthreads()` per K step.
   There's no double buffering, so every K step waits for its tile to load.
3. **Irregular shapes** (57% of cuBLAS at 1000×513×769): the float4 fallback is uncoalesced (4f).

Future work, in order of expected payoff: split-K or a smaller tile for small grids; double-buffered shared
memory (load tile k+1 while computing tile k); a coalesced scalar fallback; padding A and B on the device
to a multiple of 4 so odd shapes can use float4.

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

**Predictions vs measured:**

| Prediction | Measured | Verdict |
|---|---|---|
| `vec` vs `regtile` on aligned shapes: ≈1.0–1.3× | 1.00× (1024³), 1.06× (4096³), 1.02× (2048×512×4096), 0.93× (256³) | Low end, essentially no gain |
| `vec` ≈ `regtile` on 1000×513×769 | **0.85×: slower** | Wrong; explained in 4f |
| `vec128` faster on 4096³ and 2048×512×4096, slower on 256³ | 1.76× and 1.49× faster; **0.49×** at 256³ (and 0.44× at 256×16384×256) | Confirmed |

### 4f. Why vectorization alone didn't help, and why it hurt on odd shapes

**The SASS shows `regtile` was already using 128-bit shared-memory loads.** nvcc turned its
`float4` reads of `As`/`Bs` into `LDS.U.128` (16 per unrolled K step), exactly like `vec`. My clang test
compile had suggested otherwise, but the nvcc SASS is what counts. So the only differences in `vec` are
128-bit **global** loads (`LDG.E.128`) and stores (`STG.E.128`) and one `STS.128` for B:

| 1024³ | regtile | vec |
|---|---|---|
| Global load requests | 1.05 M | **0.26 M** (4× fewer) |
| Global store sectors | 524 K | **131 K** (4× fewer) |
| Nsight: "global stores use only 8.0 of 32 bytes per sector" | yes (est. 67%) | **gone** |
| Kernel time | 0.608 ms | 0.608 ms |

The vectorization worked as intended: 4× fewer load requests and fully coalesced C stores. But
regtile **wasn't limited by global memory** (DRAM 7.8% busy, L2 hit rate 86%). Both kernels are limited by
latency (SM and memory throughput both below 50%, still 44% of stall cycles on the MIO queue) and by
the 1.6-wave tail. Removing a non-bottleneck doesn't change the time. **That's a successful optimization
of the wrong thing, and worth saying so in the report.**

**On odd shapes the scalar fallback is uncoalesced** (1000×513×769, Nsight):

| | regtile | vec |
|---|---|---|
| Global load: bytes used per 32-byte sector | 19.3 | **6.2** |
| Excessive sectors (uncoalesced) | 41% | **80%** |
| Global load efficiency | 60% | **19%** |

In vec's load loop, thread e handles one float4 slot: tile row `e / 2`, 4 consecutive k's. When a
row can't use `float4`, the thread loads its 4 elements one by one. So in a single load instruction the
32 threads of a warp touch 16 different rows, each in a different sector. regtile assigns consecutive
threads to consecutive columns of one row, so its scalar loads stay coalesced. **Fix (not
implemented):** in the fallback, re-map threads like regtile does, or pad the matrices to a multiple of 4
on the device.

**vec128's gain comes from the bigger per-thread tile, not from vectorization.**
- 64 independent accumulators per thread give the scheduler plenty of independent FMAs. Warp cycles per
  issued instruction drop from 17.9 (vec) to **6.6**, and the FMA pipe becomes the busiest unit (62%).
- The price is 123 registers per thread, which limits theoretical occupancy to 50%, and 4× fewer blocks.
  That's why it loses on the small and skinny shapes.

### 4g. Run-to-run variation between Colab sessions

Same GPU model (Tesla T4), same code, two sessions a day apart. Kernel-only time, final / first run:

| Shape | naive | tiled T=32 | padded T=32 | CPU |
|---|---|---|---|---|
| 1024³ | **2.05×** | 1.00× | 1.00× | 1.43× |
| 4096³ | 1.09× | 1.09× | 1.11× | 0.98× |
| 1000×513×769 | 1.00× | **1.42×** | 1.38× | 1.16× |

Most GPU timings agree within about 10%, but some differ by 1.4–2×, and the CPU baseline varies by up to
1.9× (a shared 2-vCPU virtual machine). Colab doesn't allow locking clocks, and the VM, power state and
co-tenants change between sessions. Therefore:
- **every comparison in the final report uses numbers from one run** (13 Nov), never a mix of runs;
- the median of 20 trials removes noise *within* a run, not *between* sessions;
- Weeks 1–3 quote the first run (archived), and their conclusions (ratios) hold in both runs. For example,
  padded vs tiled at T = 32 is 0.76–0.83× in the first run and 0.69–0.79× in the final run.

---

## 5. Final report plan (5–8 pages, `docs/report.md` → PDF)

| Section | Source material | Status |
|---|---|---|
| 1. Problem, environment, methodology (timing, trials, tolerance) | week1 §4–5, `results/env.txt` | ready |
| 2. CPU baseline: loop order and locality | week1 §2 | ready |
| 3. Naive: thread mapping, coalescing diagram, 56% load efficiency | week1 §3, week2 §7 | ready, needs a diagram |
| 4. Tiled: pseudocode, cooperative loads, sync, edges, tile sizes, profiler evidence | week2 | ready, needs a diagram |
| 5. Padded: bank mapping, measured conflicts, why it is slower | week3 §2–7 | ready, needs a bank diagram |
| 6. Register tiling and vectorized access (regtile, vec, vec128) | week3 §8, §4e–4f here | ready |
| 7. Results: all versions + cuBLAS, plots, edge effects, 6th shape | §3–4c here | ready |
| 8. Unsuccessful experiments and lessons | week3 §7, week2 §8 entry 2 | ready |
| 9. Gap to cuBLAS and future work | §4d, 4g | ready |

---

## 6. Week 4 milestone checklist

1. ◐ 6th shape run (option A, 256×16384×256); instructor approval to be recorded (section 3).
2. ☑ Complete run with the notebook from a clean clone (13 Nov 2026); `results/` committed.
3. ☑ All tests pass: 327 / 327 (`results/tests.txt`), including regtile, vec, vec128 and cuBLAS.
4. ☑ Tables 4a–4c filled in; gap to cuBLAS analysed (4d), extensions analysed (4e–4f), run variation (4g).
5. ☐ Final report written (5–8 pages) with pseudocode, mapping and memory-access diagrams, plots and profiler evidence.
6. ☐ README reproduction steps checked by following them exactly.
