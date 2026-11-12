# Week 2: shared-memory tiling (what changed from Week 1 and why)

**Week 2 goal:** implement the shared-memory tiled kernel, compare at least two tile sizes, profile it, and use the profiler results to justify the optimization.

> **Status of the numbers in this file.**
> Sections 1–6 (counts, formulas, code behaviour) are derived from the problem sizes and the code, so they are exact.
> Sections 7–9 contain **tables to fill in with your own GPU measurements**. They are left blank on purpose: performance numbers must come from running `make bench` and `make profile` on your GPU, not from guesses.
> The CUDA kernel itself could not be compiled where this file was written (no GPU/nvcc available). Its index and zero-fill logic was verified by a NumPy emulation on 13 shapes × 3 tile sizes, but you must still run `make test` on a real GPU.

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

## 7. Profiling plan (required in Week 2)

Run on your GPU:

```
make bench      # timings  -> results/results.csv, results/env.txt
make profile    # Nsight Compute -> results/profile/*.ncu-rep and *.csv
```

`scripts/profile.sh` profiles one large divisible shape (1024×1024×1024) and one irregular shape (1000×513×769) for the naive kernel and the tiled kernel at TILE = 8, 16, 32.

Fill in this table from the profiler (large divisible shape):

| Metric | Naive | Tiled T=8 | Tiled T=16 | Tiled T=32 |
|---|---|---|---|---|
| Kernel time (ms) | | | | |
| DRAM bytes read | | | | |
| Global load requests / sectors | | | | |
| Global load efficiency (%) | | | | |
| Shared-memory bank conflicts (ld / st) | n/a | | | |
| Achieved occupancy (%) | | | | |
| Compute throughput (% of peak) | | | | |
| Memory throughput (% of peak) | | | | |
| Limiting resource (from "Speed of Light") | | | | |

What the evidence should show if the explanation in sections 2–4 is right:

1. The naive kernel has high memory throughput relative to compute throughput (memory-bound).
2. The tiled kernels move far fewer bytes from global memory (compare the DRAM and global-load rows against the table in section 4).
3. Tiled kernel time drops accordingly.

If the numbers disagree with these expectations, **report that and analyse why** (the project explicitly asks for honest reporting). For example, caches may already hide much of the naive kernel's redundant traffic on your GPU, which would make the improvement smaller than the load-count reduction suggests.

---

## 8. Profiler-guided optimization log (required milestone)

Use one entry per optimization. Entry 1 is pre-filled with the Week 2 change; complete the measured parts.

| Field | Entry 1: shared-memory tiling |
|---|---|
| Observation (naive profile) | *fill in: e.g. memory throughput % vs. compute throughput %, DRAM bytes read* |
| Hypothesis | Redundant global loads limit the naive kernel; reuse through shared memory should cut global traffic by about TILE× |
| Change | `matmul_tiled<TILE>` with cooperative loads, 2 × `__syncthreads()`, zero-filled edge tiles |
| Expected effect | Global bytes down by 8× / 16× / 32× for T = 8 / 16 / 32 (section 4) |
| Measured effect | *fill in: DRAM bytes, kernel time, GFLOP/s before and after* |
| Conclusion | *fill in: did the evidence support the hypothesis? what is the new bottleneck?* |

| Field | Entry 2: tile size selection |
|---|---|
| Observation | *fill in: compare T = 8 / 16 / 32 in the table in section 7 (occupancy, load efficiency, time)* |
| Hypothesis | *fill in, using the trade-offs in section 5* |
| Change | Choose the default `--tile` per the evidence |
| Measured effect | *fill in* |
| Conclusion | *fill in* |

Later ideas to try only if the profile supports them (do not add them blindly): register tiling so each thread computes several outputs, and vectorized loads. Both are listed as optional extensions in the project.

---

## 9. Results to report (fill from `results/results.csv`)

| Shape | Naive kernel (ms) | Tiled T=8 | Tiled T=16 | Tiled T=32 | Naive → best tiled speedup | Best tiled GFLOP/s |
|---|---|---|---|---|---|---|
| 256×256×256 | | | | | | |
| 1024×1024×1024 | | | | | | |
| 4096×4096×4096 | | | | | | |
| 2048×512×4096 | | | | | | |
| 1000×513×769 | | | | | | |

Also record (required): CPU model, GPU model, CUDA version, compiler flags, and benchmark commands. `results/env.txt` captures most of these automatically.

---

## 10. Week 2 milestone checklist

1. ☐ `make test` passes for `naive` and `tiled` at tile sizes 8, 16, 32 (all 15 shapes, including the 1-sized and 31/33 cases).
2. ☐ `make bench` has run and `results/results.csv` is saved.
3. ☐ `make profile` has run; the `.ncu-rep` files are kept.
4. ☐ Section 7 table filled from the profiler.
5. ☐ Section 8, entry 1 completed with measured evidence.
6. ☐ Everything committed and pushed to GitHub.

**Heads-up for Week 3 (padding):** in this kernel the shared-memory reads `As[ty][k]` (same address across a warp row) and `Bs[k][tx]` (consecutive addresses) are expected to be free of bank conflicts already, so padding may show little or no gain. Check the bank-conflict row in section 7 first; the project explicitly asks you to report honestly if padding does not help.
