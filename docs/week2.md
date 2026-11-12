# Week 2: shared-memory tiling (what changed from Week 1 and why)

**Week 2 goal:** implement the shared-memory tiled kernel, compare at least two tile sizes, profile it, and use the profiler results to justify the optimization.

> **Status of the numbers in this file.**
> Sections 1–6 (counts, formulas, code behaviour) come from the problem sizes and the code.
> Sections 7–9 are **measured** on a Tesla T4 (Google Colab, 12 Nov 2026, CUDA 13.0). Raw data:
> `results/results.csv` (timings), `results/profile/` and `results/profile_summary.csv` (Nsight Compute).
> The correctness suite has not been logged on the GPU yet (see section 10).

---

## 1. Summary of the change

| | Week 1: naive | Week 2: tiled |
|---|---|---|
| Work per thread | 1 output element | 1 output element (unchanged) |
| Where operands come from | Global memory, every time | Global memory once per tile, then shared memory |
| Global loads per output element | 2·K | 2·K / TILE |
| Threads per block | 16×16 (fixed) | TILE×TILE (8, 16, or 32) |
| Shared memory per block | 0 | 2·TILE²·4 bytes |
| Synchronization | none | 2 × `__syncthreads()` per tile step |
| Edge handling | `if (row<M && col<N)` | zero-fill loads + guarded store |
| Code | `matmul_naive` | `matmul_tiled<TILE>` |

The key point: **both kernels use the same parallelism** (one thread per output). Week 2 improves only **memory efficiency**, which is exactly the "parallelism vs. memory-efficiency" distinction from the project's learning objectives.

---

## 2. The Week 1 problem: redundant global reads

In the naive kernel, thread (row, col) reads row `row` of A and column `col` of B in full.

Worked example, shape (1024, 1024, 1024):

| Quantity | Value |
|---|---|
| Output elements | 1024 × 1024 = 1,048,576 |
| Loads per output element | 2 × 1024 = 2,048 floats |
| **Total global loads** | **2,147,483,648 floats (8 GiB if every load reached DRAM)** |
| Floating-point operations | 2·M·N·K = 2,147,483,648 |
| FLOP per byte loaded | 2,147,483,648 / 8 GiB = **0.25** |

Every element of A is loaded by all N threads that share its row, and every element of B by all M threads that share its column. The GPU's caches absorb part of this, but the kernel is still expected to be limited by memory traffic rather than arithmetic. Section 7 explains how to confirm that with the profiler.

---

## 3. The Week 2 fix: cooperative tiling

Step by step, for one thread block computing one TILE×TILE tile of C:

1. Each thread computes its own `row` and `col` in C.
2. For each tile step `t` along K (there are ⌈K/TILE⌉ steps):
   1. Every thread loads **one** element of A's current tile into `As[ty][tx]`.
   2. Every thread loads **one** element of B's current tile into `Bs[ty][tx]`.
   3. `__syncthreads()`, so no thread reads a tile that is only partly loaded.
   4. Each thread does TILE multiply-adds: `acc += As[ty][k] * Bs[k][tx]`.
   5. `__syncthreads()`, so no thread overwrites the tile while others still read it.
3. Each thread stores `acc` to `C[row][col]` if it is inside the matrix.

```
                 K ->                       tile step t
          +-------------------+          +-----------+
   A      |    | A tile (t) |  |          |  As[ty][k]|   read along a ROW
 (M x K)  |    +------------+  |          +-----------+
          +-------------------+                 x
                                         +-----------+
   B      +-------+                      |  Bs[k][tx]|   read down a COLUMN
 (K x N)  |  B    |                      +-----------+
          | tile  |                            =
          |  (t)  |                      acc (one C element per thread)
          +-------+
```

Why this is faster: each element loaded into shared memory is read by **TILE** different threads, instead of each thread fetching its own copy from global memory.

---

## 4. Data reuse and arithmetic intensity (worked example)

Shape (1024, 1024, 1024). Total global loads for the tiled kernel are `2·M·N·K / TILE`.

| Kernel | Global loads (floats) | Global bytes | FLOP per byte | Reduction vs. naive |
|---|---|---|---|---|
| Naive | 2,147,483,648 | 8 GiB | 0.25 | 1× |
| Tiled, TILE = 8 | 268,435,456 | 1 GiB | 2 | 8× |
| Tiled, TILE = 16 | 134,217,728 | 512 MiB | 4 | 16× |
| Tiled, TILE = 32 | 67,108,864 | 256 MiB | 8 | 32× |

Larger tiles mean more reuse. They are not automatically faster, because of the resource trade-offs below.

---

## 5. Comparing tile sizes (required: at least two)

| | TILE = 8 | TILE = 16 | TILE = 32 |
|---|---|---|---|
| Threads per block | 64 | 256 | 1024 |
| Shared memory per block | 512 B | 2 KiB | 8 KiB |
| Blocks for 1024³ | 16,384 | 4,096 | 1,024 |
| Contiguous bytes per tile row loaded | 32 B | 64 B | 128 B |
| Reuse factor | 8 | 16 | 32 |

Things to examine with the profiler (these are hypotheses to test, not results):

1. **Coalescing:** with TILE = 8, each row of a loaded tile is only 32 bytes, so a warp touches several short segments. Larger tiles should give wider contiguous reads.
2. **Occupancy:** TILE = 32 uses the maximum 1024 threads per block, which can limit how many blocks fit on one SM. Check "achieved occupancy" and the occupancy limiter in Nsight Compute.
3. **Synchronization cost:** small tiles mean more tile steps (K/TILE) and therefore more `__syncthreads()` calls.

The program checks that the chosen tile fits the device (threads per block and shared memory) and exits with a clear message if it does not.

---

## 6. Edge handling (worked examples)

Both loads use a conditional, so out-of-range elements become `0.0f` and contribute nothing to the dot product.

**Example 1: (33, 33, 33), TILE = 16**
- Grid = 3 × 3 = 9 blocks; tile steps = ⌈33/16⌉ = 3.
- The last tile step covers only index 32, so 1 of 16 columns of A (and rows of B) is real data and 15 are zero-filled.
- Only 1 thread column and 1 thread row of the last blocks write results.

**Example 2: (1000, 513, 769), TILE = 16**

| Quantity | Value | Why |
|---|---|---|
| Grid x | 49 blocks | ⌈769/16⌉; last block column has 1 valid column |
| Grid y | 63 blocks | ⌈1000/16⌉; last block row has 8 valid rows |
| Tile steps | 33 | ⌈513/16⌉; the last step has 1 valid K element |
| Threads launched | 790,272 | 49 × 63 × 256 |
| Outputs | 769,000 | about 2.7% of launched threads are idle |

**Why there is no early `return`:** a thread outside the matrix must still reach both `__syncthreads()` calls (it still helps load tiles). Only the final store is guarded.

---

## 7. Profiler results (1024×1024×1024)

Commands: `bash scripts/profile.sh`, then `python3 scripts/summarize_profile.py > results/profile_summary.csv`.
`ncu` locks the GPU to its base clock while profiling, so the kernel times in this table are about 2.6×
the benchmark times in section 9. Compare kernels with each other, not with section 9.

| Metric | Naive | Tiled T=8 | Tiled T=16 | Tiled T=32 |
|---|---|---|---|---|
| Kernel time under ncu (ms) | 8.96 | 9.10 | 5.78 | 5.31 |
| Global load requests | 67.1 M | 8.39 M | 4.19 M | 2.10 M |
| Global load sectors (32 B) | 134.2 M | 33.5 M | 16.7 M | 8.39 M |
| Global load efficiency | 56% | 100% | 100% | 100% |
| DRAM bytes read | 212 MB | 164 MB | 193 MB | 148 MB |
| Shared-memory bank conflicts (ld / st) | n/a | 0 / 0 | 0 / 0 | 0 / 0 |
| Shared-memory load instructions | n/a | 41.9 M | 41.9 M | 41.9 M |
| Achieved occupancy | 98.5% | 98.7% | 98.7% | 99.9% |
| SM throughput (% of peak) | 64% | 55% | 75% | 74% |
| Memory throughput (% of peak) | 64% | 55% | 75% | 74% |

Irregular shape 1000×513×769 (same trends; global-load efficiency drops because rows of length 513 and
769 floats don't start on 32-byte sector boundaries):

| Metric | Naive | Tiled T=8 | Tiled T=16 | Tiled T=32 |
|---|---|---|---|---|
| Kernel time under ncu (ms) | 2.70 | 3.49 | 2.27 | 2.19 |
| Global load requests | 25.1 M | 3.14 M | 1.60 M | 0.84 M |
| Global load efficiency | 46% | 49% | 63% | 80% |
| DRAM bytes read | 10.0 MB | 13.2 MB | 9.7 MB | 7.4 MB |
| Achieved occupancy | 97.1% | 98.2% | 98.2% | 99.8% |

### What the evidence shows

1. **Tiling cuts global load requests by exactly TILE×,** as predicted in section 4: 67.1 M → 8.39 M /
   4.19 M / 2.10 M for T = 8 / 16 / 32. Sectors drop 134 M → 8.4 M, and load efficiency goes from 56% to 100%.
2. **DRAM traffic hardly changes** (212 MB → 148–193 MB). The naive kernel asks L1 for 134 M sectors (4.3 GB),
   but only 212 MB reaches DRAM. The caches were already absorbing ~95% of the redundant reads. So the
   16–32× reduction happens at L1/L2, not DRAM, and the speedup is **1.7–2.0×, not 16–32×**. Section 7
   predicted this possibility.
3. **Neither kernel is DRAM-bound or FP32-bound.** DRAM reads are ~33 GB/s for tiled T=16 under ncu (~10% of
   the T4's 320 GB/s). The best kernel reaches 1042 GFLOP/s, ~13% of the T4's 8.1 TFLOP/s FP32 peak.
4. **Occupancy is not the problem:** 98–100% for every kernel, including T = 32 with 1024-thread blocks.
5. **The new limiter is the shared-memory path.** SM and memory throughput report the same percentage in
   every row, meaning both peak on the same unit. The only unit busy enough is the L1/shared-memory pipe.
   Each multiply-add needs 1.25 shared-memory loads (one `Bs` value, plus a quarter of a 128-bit `As`
   load); the instruction count of 41.9 M is the same for every tile size. Week 3 (section 6c there) shows
   kernel time is proportional to shared-memory wavefronts. *To confirm which unit, open a `.ncu-rep`:*
   `ncu --import results/profile/tiled_T32_1024x1024x1024.ncu-rep --page details --section SpeedOfLight --section MemoryWorkloadAnalysis`.

---

## 8. Profiler-guided optimization log

| Field | Entry 1: shared-memory tiling |
|---|---|
| Observation (naive profile) | 67.1 M global load requests (2·K per output); load efficiency 56% (each warp spans 2 rows, so the A load uses 8 of 64 fetched bytes); SM / memory throughput 64% |
| Hypothesis | Redundant global loads limit the naive kernel; reuse through shared memory should cut global traffic by about TILE× |
| Change | `matmul_tiled<TILE>` with cooperative loads, 2 × `__syncthreads()`, zero-filled edge tiles |
| Expected effect | Global loads down by 8× / 16× / 32× for T = 8 / 16 / 32 (section 4) |
| Measured effect | Requests down **exactly** 8× / 16× / 32×; load efficiency 56% → 100%; DRAM reads only 1.1–1.4× lower. Kernel time (benchmark, T = 32): 1024³ 3.44 → 2.06 ms (**1.67×**), 4096³ 304 → 154 ms (**1.97×**), 2048×512×4096 **1.96×**, 1000×513×769 **1.24×** |
| Conclusion | Supported at the L1/L2 level. The gain is smaller than the load reduction because L2 was already catching most of the reuse. The new bottleneck is the shared-memory pipe (point 5 above), not global memory |

| Field | Entry 2: tile size selection |
|---|---|
| Observation | T = 32 is fastest on every shape; T = 16 is 4–14% slower (29% slower at 4096³); T = 8 is **slower than naive** at 4096³ (324 vs 304 ms) and 1000×513×769 (1.30 vs 1.03 ms) |
| Hypothesis | Same shared-instruction count for all T (41.9 M), so the difference is overhead per tile step: T = 8 has 4× the global requests and `__syncthreads()` per output of T = 32, and 16× as many blocks. Occupancy is ~99% for all T, so it's not occupancy |
| Change | Default `--tile` changed from 16 to **32** (`src/matmul.cu`) |
| Measured effect | T = 32 vs T = 16: 1024³ 2.15 → 2.06 ms (4%), 4096³ 199 → 154 ms (22%), 2048×512×4096 9.04 → 8.30 ms (8%), 1000×513×769 0.859 → 0.827 ms (4%) |
| Conclusion | Larger tiles win on this GPU. T = 32 also has the best global-load efficiency on the irregular shape (80% vs 63% / 49%) |

---

## 9. Results (benchmark, kernel-only, median of 20)

| Shape | Naive kernel (ms) | Tiled T=8 | Tiled T=16 | Tiled T=32 | Naive → best tiled speedup | Best tiled GFLOP/s |
|---|---|---|---|---|---|---|
| 256×256×256 | 0.144 | 0.073 | 0.056 | 0.049 | 2.92× (T=32) | 681 |
| 1024×1024×1024 | 3.444 | 3.363 | 2.155 | 2.062 | 1.67× (T=32) | 1042 |
| 4096×4096×4096 | 304.3 | 324.2 | 199.1 | 154.4 | 1.97× (T=32) | 890 |
| 2048×512×4096 | 16.27 | 16.27 | 9.039 | 8.302 | 1.96× (T=32) | 1035 |
| 1000×513×769 | 1.025 | 1.301 | 0.859 | 0.827 | 1.24× (T=32) | 954 |

Environment: Tesla T4 (driver 580.82.07, max SM clock 1590 MHz, 70 W power limit), 2-vCPU Intel Xeon @ 2.00 GHz
(Colab), CUDA 13.0, `nvcc -O3 -std=c++14 -arch=native`. Command: `bash scripts/bench.sh`. Plots:
`results/plots/tile_sizes.svg`, `results/plots/gflops_by_impl.svg`.

---

## 10. Week 2 milestone checklist

1. ☐ `make test` passes for `naive` and `tiled` at tile sizes 8, 16, 32. **Not logged yet: run `bash scripts/run_tests.sh | tee results/tests.txt`.**
2. ☑ `make bench` run, `results/results.csv` saved.
3. ☑ `make profile` run, `.ncu-rep` files kept.
4. ☑ Section 7 table filled from the profiler.
5. ☑ Section 8, entries 1 and 2 completed with measured evidence.
6. ☐ Everything committed and pushed to GitHub.

**Heads-up for Week 3 (padding):** in this kernel the shared-memory reads `As[ty][k]` (same address across a warp row) and `Bs[k][tx]` (consecutive addresses) are expected to be free of bank conflicts already, so padding may show little or no gain. Check the bank-conflict row in section 7 first; the project explicitly asks you to report honestly if padding does not help.
