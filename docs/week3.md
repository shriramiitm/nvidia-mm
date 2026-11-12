# Week 3: shared-memory padding and the profile-optimize loop

**Week 3 goal (from the spec):** add shared-memory padding while keeping the tiled kernel's arithmetic, compare bank-conflict metrics for tiled and padded, repeat the profile-optimize cycle on divisible and irregular shapes, and record what worked and what didn't.

**Milestone:** present a second profiler-guided improvement, and explain whether padding helps on the selected GPU.

> **Status of the numbers in this file.**
> - Sections 1–4 come from the code and from `scripts/bank_model.py`, which counts bank conflicts exactly for the access patterns as written. These are **predictions**.
> - Sections 5–7 are **tables to fill in** with Nsight Compute and benchmark results from the GPU machine.
> - The kernels have **not been compiled with nvcc or run on a GPU yet**. What was checked without a GPU:
>   - clang parses `src/matmul.cu` cleanly in both host and device mode (using a stub CUDA header).
>   - A pure-Python emulation of all 12 kernel variants (3 tile sizes × padding on/off × transposed A on/off) matches a reference result on 10 shapes, including 1×1×1, K = 1, and sizes 31/33.
>
>   `make test` on the GPU is still required.

---

## 1. What changed

All tiled variants now come from **one** template in `src/matmul.cu`:

```cpp
template <int TILE, int PAD, bool TRANSPOSE_A>
__global__ void matmul_tiled(...) {
    __shared__ float As[TILE][TILE + PAD];
    __shared__ float Bs[TILE][TILE + PAD];
    ...
}
```

| `--impl` | PAD | TRANSPOSE_A | Role |
|---|---|---|---|
| `tiled` | 0 | no | Version 3 (Week 2 kernel, unchanged behaviour) |
| `padded` | 1 | no | **Version 4** (required) |
| `tiled_ta` | 0 | yes | Experiment: A tile stored transposed |
| `padded_ta` | 1 | yes | Experiment: transposed and padded |

Because `tiled` and `padded` are the same source with a different `PAD`, the global-memory tiling, loads, edge handling and arithmetic are **exactly** the same. Padding is the only variable, as the spec asks.

The extra column (`[*][TILE]`) is never read or written, so it can't change the result. It only changes where each element lives in memory: the physical row length (the *leading dimension*) is `LD = TILE + PAD`.

`check_tile_resources` now includes the padding when checking shared memory: `2·TILE·(TILE+PAD)·4` bytes.

---

## 2. Shared-memory banks in one paragraph

Shared memory is split into **32 banks**, each 4 bytes wide. The 32-bit word at word address `w` is in bank `w % 32`. When a warp (32 threads) runs one shared-memory instruction:

- threads reading the **same word** are served together (a *broadcast*, which is free);
- threads reading **different words in the same bank** are served one after another (a *bank conflict*).

The instruction takes `max over banks (distinct words in that bank)` passes (*wavefronts*). Ideally that's 1. Nsight Compute's `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_{ld,st}.sum` counts the extra passes.

For a 2D array `X[r][c]` with row length `LD`:

```
bank(X[r][c]) = (r * LD + c) % 32
```

- **Along a row** (`c` varies): consecutive banks. No conflict whatever `LD` is.
- **Down a column** (`r` varies): banks step by `LD`. With `LD = 32`, *every* row is in the same bank, a 32-way conflict. With `LD = 33` (odd), each row moves one bank over, so 32 rows hit 32 different banks: **no conflict**. That's the whole trick behind `[TILE][TILE + 1]`.

---

## 3. Which access patterns does the tiled kernel actually have?

Threads in a block are numbered `tid = ty·TILE + tx`, and a warp is 32 consecutive `tid`s. So:

- TILE = 32: one warp = one `ty` row, `tx = 0..31`
- TILE = 16: one warp = 2 rows, `tx = 0..15`
- TILE = 8: one warp = 4 rows, `tx = 0..7`

| Access | Per tile step | What a warp touches | Conflict at PAD = 0? |
|---|---|---|---|
| store `As[ty][tx]` | 1 | along a row (tx varies) | none |
| store `Bs[ty][tx]` | 1 | along a row | none |
| load `As[ty][k]` | TILE | same `k` for all threads, 1–4 different rows: a broadcast per row; the rows' banks differ by `LD` | none |
| load `Bs[k][tx]` | TILE | along row `k` | none |

**Prediction: the Week 2 tiled kernel has no shared-memory bank conflicts at any tile size.** This is a well-known property of the textbook kernel: it reads `As` by row (broadcast) and `Bs` by row (consecutive). The column reads that padding fixes never happen.

---

## 4. Predicted effect of padding (from `python3 scripts/bank_model.py`)

Extra wavefronts (bank conflicts) per block, per K-tile step:

| Variant | T | Shared memory per block | st As | st Bs | ld As | ld Bs | Worst store |
|---|---|---|---|---|---|---|---|
| tiled | 8 | 512 B | 0 | 0 | 0 | 0 | 1-way |
| tiled | 16 | 2048 B | 0 | 0 | 0 | 0 | 1-way |
| tiled | 32 | 8192 B | 0 | 0 | 0 | 0 | 1-way |
| **padded** | 8 | 576 B | **2** | **2** | 0 | 0 | 2-way |
| **padded** | 16 | 2176 B | **8** | **8** | 0 | 0 | 2-way |
| **padded** | 32 | 8448 B | 0 | 0 | 0 | 0 | 1-way |
| tiled_ta | 8 | 512 B | 2 | 0 | 0 | 0 | 2-way |
| tiled_ta | 16 | 2048 B | 56 | 0 | 0 | 0 | 8-way |
| tiled_ta | 32 | 8192 B | **992** | 0 | 0 | 0 | **32-way** |
| padded_ta | 8 | 576 B | 2 | 2 | 0 | 0 | 2-way |
| padded_ta | 16 | 2176 B | 8 | 8 | 0 | 0 | 2-way |
| padded_ta | 32 | 8448 B | **0** | 0 | 0 | 0 | 1-way |

Whole-kernel totals for 1024×1024×1024 (multiply by `blocks × K/T`). This is what `ncu` should report:

| Variant | T = 8 (st As / st Bs) | T = 16 | T = 32 |
|---|---|---|---|
| tiled | 0 / 0 | 0 / 0 | 0 / 0 |
| padded | 4,194,304 / 4,194,304 | 2,097,152 / 2,097,152 | 0 / 0 |
| tiled_ta | 4,194,304 / 0 | 14,680,064 / 0 | 32,505,856 / 0 |
| padded_ta | 4,194,304 / 4,194,304 | 2,097,152 / 2,097,152 | 0 / 0 |

Load conflicts are predicted to be 0 for every variant.

### 4a. Why padding *adds* conflicts to `padded` at T = 8 and 16

With TILE = 16 a warp spans two rows, `ty` and `ty+1`, each storing 16 consecutive words.

- Without padding, row `ty+1` starts exactly 16 words later, so the two halves fill banks `b..b+15` and `b+16..b+31`: all 32 banks, no conflict.
- With `LD = 17`, row `ty+1` starts 17 words later and covers banks `b+17 .. b+32`. Bank `b+32 = b` is already used by `tx = 0` of row `ty`, so that bank gets 2 words: a **2-way conflict**.

TILE = 8 works the same way across four rows. At TILE = 32 a warp covers one row only, so the row length makes no difference.

These are **store** conflicts: 2 store instructions per tile step versus `2·TILE` load instructions, so the expected slowdown is small. It's still the honest answer the spec asks for. On this kernel, padding has nothing to fix, and at T = 8/16 it makes the stores slightly worse.

### 4b. A second possible cost: losing 128-bit shared loads (hypothesis)

In the inner loop, `As[ty][k], As[ty][k+1], …` are consecutive and the same for every thread in a row. With `LD = TILE`, each row starts on a 16-byte boundary, so the compiler *may* merge four of these reads into one `LDS.128`, cutting the number of As load instructions by 4×.

With `LD = TILE + 1`, rows are no longer 16-byte aligned and that merge is impossible. `scripts/profile.sh` now saves the SASS to `results/profile/matmul.sass`. Check it:

```
grep -E "LDS(\.128|\.64)?" results/profile/matmul.sass | sort | uniq -c
```

Compare `smsp__inst_executed_op_shared_ld.sum` between tiled and padded too. If tiled uses `LDS.128` and padded doesn't, that alone can make padded slower even with zero load conflicts.

### 4c. The experiment: where padding *does* help

`tiled_ta` stores the A tile transposed (`As[tx][ty] = A[row][t·T+tx]`) and reads `As[k][ty]`. The arithmetic is identical, but the store now walks **down a column**:

- T = 32: all 32 threads of a warp write column `ty`, `tx = 0..31` rows apart, at stride 32 → one bank → **32-way conflict** on every A-tile store.
- `padded_ta` (LD = 33): stride 33 → 32 different banks → **0 conflicts**.

This is the pattern from the spec and the CuTe tutorial, isolated with everything else held fixed. It shows that the padding works and the bank model is right, even though `padded` itself has nothing to fix. A transposed shared tile isn't contrived either: register-tiled GEMMs often store A transposed so each thread can read several `A` values for different rows with one vector load.

---

## 5. Profiling commands (run on the GPU machine)

```
make test                                             # all variants, all tile sizes, 15 shapes
make bench                                            # results/results.csv (cpu naive tiled padded tiled_ta padded_ta)
make profile                                          # results/profile/*.ncu-rep, *.csv, matmul.sass
python3 scripts/bank_model.py > results/bank_model.txt   # predictions, to sit next to the measurements
```

`profile.sh` now also collects shared-memory wavefronts and shared load/store instruction counts. Shapes: 1024³ (divisible) and 1000×513×769 (irregular).

---

## 6. Tables to fill in

### 6a. Bank conflicts: predicted vs. measured (1024³)

| Variant | T | Predicted st conflicts (As+Bs) | Measured st | Predicted ld | Measured ld | Shared ld instructions |
|---|---|---|---|---|---|---|
| tiled | 16 | 0 | | 0 | | |
| padded | 16 | 4,194,304 | | 0 | | |
| tiled | 32 | 0 | | 0 | | |
| padded | 32 | 0 | | 0 | | |
| tiled_ta | 32 | 32,505,856 | | 0 | | |
| padded_ta | 32 | 0 | | 0 | | |

If measured and predicted disagree, explain why. Likely causes: 128-bit loads (section 4b), or the irregular shape having fewer full blocks.

### 6b. Timing: tiled vs. padded (kernel-only, median of 20)

| Shape | T | tiled (ms) | padded (ms) | padded / tiled | tiled_ta (ms) | padded_ta (ms) | padded_ta / tiled_ta |
|---|---|---|---|---|---|---|---|
| 1024³ | 16 | | | | | | |
| 1024³ | 32 | | | | | | |
| 4096³ | 32 | | | | | | |
| 2048×512×4096 | 32 | | | | | | |
| 1000×513×769 | 16 | | | | | | |
| 1000×513×769 | 32 | | | | | | |

### 6c. Bottleneck after padding (from the "Speed of Light" and "Memory Workload" sections)

| Kernel (1024³, best T) | Compute % | Memory % | Achieved occupancy % | Shared-memory pipe utilisation | Limiting resource |
|---|---|---|---|---|---|
| tiled | | | | | |
| padded | | | | | |

---

## 7. Experiment log (successful and unsuccessful)

| # | Experiment | Hypothesis | Result | Kept? |
|---|---|---|---|---|
| 1 | Padding `[T][T+1]` on the tiled kernel | Tiled has no bank conflicts (section 3), so padding can't help; it adds 2-way store conflicts at T = 8/16 and may block `LDS.128` | *fill in* | *probably no* |
| 2 | Transposed A tile, no padding (`tiled_ta`) | A 32-way store conflict at T = 32 makes it slower than tiled | *fill in* | no (it's a demonstration) |
| 3 | Transposed A tile + padding (`padded_ta`) | Padding removes all of experiment 2's conflicts and gets back most of the lost time | *fill in* | demonstration |
| 4 | Tile-size retune with padding (T = 8/16/32) | The best T is the same as without padding (padding costs at most 3% more shared memory, so occupancy shouldn't change) | *fill in* | |
| 5 | **Second profiler-guided improvement** | see section 8 | *fill in* | |

---

## 8. The second profiler-guided improvement (milestone)

This has to come **from the profile in section 6c**. Don't choose it in advance. The most likely outcome, and what to try for each:

- **If the shared-memory (LSU/MIO) pipe is the limiter** (likely): each FMA needs two shared loads (`As[ty][k]` and `Bs[k][tx]`), so the kernel is limited by shared-memory bandwidth, not FLOPs. The fix is **register tiling**: each thread computes e.g. a 1×4 or 4×4 block of C, keeping `As` values in registers and reusing them across 4 `Bs` values. That gives 4–16 FMAs per pair of shared loads. (This is the spec's first optional extension.)
- **If occupancy or latency is the limiter** (e.g. T = 32 with 1024-thread blocks): try a smaller T, or non-square blocks.
- **If global memory is still the limiter**: try vectorized `float4` global loads (another optional extension).

Record the observation → hypothesis → change → measured effect in the same format as `docs/week2.md` section 8.

---

## 9. Week 3 milestone checklist

1. ☐ `make test` passes for tiled, padded, tiled_ta and padded_ta at T = 8, 16, 32.
2. ☐ `make profile` run; tables 6a and 6c filled in; predicted and measured conflicts compared.
3. ☐ `make bench` run; table 6b filled in.
4. ☐ Answer "does padding help on this GPU?" with the evidence: expected *no* for `padded`, *yes* for the `_ta` experiment.
5. ☐ Second profiler-guided improvement chosen from 6c, implemented, measured (section 8).
6. ☐ Experiment log (section 7) complete, including the unsuccessful experiments.
