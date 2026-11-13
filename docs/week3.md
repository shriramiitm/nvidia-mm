# Week 3: shared-memory padding and the profile-optimize loop

> **Which run these numbers come from.** Tables in this file quote the **first** T4 run (12 Nov 2026), now
> archived in `results/archive/2026-11-12-first-run/`. The final, complete run (13 Nov) is in `results/` and
> is analysed in [week4.md](week4.md). Most timings agree within ~10% between the two Colab sessions; a few
> differ by 1.4–2× (week4.md §4g). The conclusions here hold in both runs.

**Week 3 goal (from the spec):** add shared-memory padding while keeping the tiled kernel's arithmetic, compare bank-conflict metrics for tiled and padded, repeat the profile-optimize cycle on divisible and irregular shapes, and record what worked and what didn't.

**Milestone:** present a second profiler-guided improvement, and explain whether padding helps on the selected GPU.

> **Status of the numbers in this file.**
> - Sections 1–4 are the analysis and predictions, written before any GPU run. They come from the code
>   and from `scripts/bank_model.py`, which counts bank conflicts for the access patterns as written.
> - Sections 6–8 are **measured** on a Tesla T4 (Google Colab, 12 Nov 2026, CUDA 13.0). Raw data:
>   `results/results.csv`, `results/profile/`, `results/profile_summary.csv`, `results/profile/matmul.sass`.
> - **Every bank-conflict prediction matched the profiler exactly** (section 6a).
> - The correctness suite passes on the GPU (final run: 327 / 327, `results/tests.txt`).

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

### 4b. A second cost: losing 128-bit shared loads (hypothesis, **confirmed** in section 6)

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

## 6. Measured results

### 6a. Bank conflicts: predicted vs. measured

1024×1024×1024. Store conflicts are As + Bs together, because Nsight Compute reports one total.

| Variant | T | Predicted st conflicts | **Measured st** | Predicted ld | **Measured ld** | Shared ld instructions | Shared ld wavefronts |
|---|---|---|---|---|---|---|---|
| tiled | 8 | 0 | **0** | 0 | **0** | 41.9 M | 50.3 M |
| tiled | 16 | 0 | **0** | 0 | **0** | 41.9 M | 50.3 M |
| tiled | 32 | 0 | **0** | 0 | **0** | 41.9 M | 50.3 M |
| padded | 8 | 8,388,608 | **8,388,608** | 0 | **0** | 67.1 M | 67.1 M |
| padded | 16 | 4,194,304 | **4,194,304** | 0 | **0** | 67.1 M | 67.1 M |
| padded | 32 | 0 | **0** | 0 | **0** | 67.1 M | 67.1 M |
| tiled_ta | 8 | 4,194,304 | **4,194,304** | 0 | **0** | 67.1 M | 67.1 M |
| tiled_ta | 16 | 14,680,064 | **14,680,064** | 0 | **0** | 67.1 M | 67.1 M |
| tiled_ta | 32 | 32,505,856 | **32,505,856** | 0 | **0** | 67.1 M | 67.1 M |
| padded_ta | 8 | 8,388,608 | **8,388,608** | 0 | **0** | 67.1 M | 67.1 M |
| padded_ta | 16 | 4,194,304 | **4,194,304** | 0 | **0** | 67.1 M | 67.1 M |
| padded_ta | 32 | 0 | **0** | 0 | **0** | 67.1 M | 67.1 M |

The irregular shape (1000×513×769) matches too. For example, padded T=16: 3087 blocks × 33 tile steps × 16 =
**1,629,936** predicted and measured. tiled_ta T=32: 800 × 17 × 992 = **13,491,200** predicted and measured.

**The SASS confirms section 4b** (`results/profile/matmul.sass`). Shared-memory instructions per unrolled
tile step:

| Kernel (T = 16) | `LDS.U.128` | `LDS.U` | Shared loads per tile step |
|---|---|---|---|
| tiled | 4 (the As row: 16 floats as 4 × 128-bit) | 16 (Bs) | 20 |
| padded, tiled_ta, padded_ta | 0 | 32 | 32 |

32 / 20 = 1.6, which is exactly the measured instruction ratio 67.1 M / 41.9 M. T = 8 and T = 32 show the
same pattern. The odd row length `T + 1` breaks 16-byte alignment, so the compiler can't vectorize the
`As` reads. In `tiled_ta` the `As[k][ty]` reads come from different rows, so they can't be vectorized either.

### 6b. Timing: tiled vs. padded (benchmark, kernel-only, median of 20)

| Shape | T | tiled (ms) | padded (ms) | **padded / tiled** | tiled_ta (ms) | padded_ta (ms) | **tiled_ta / padded_ta** |
|---|---|---|---|---|---|---|---|
| 256×256×256 | 16 | 0.056 | 0.061 | 1.08× slower | 0.064 | 0.060 | 1.06× faster |
| 256×256×256 | 32 | 0.049 | 0.059 | 1.20× slower | 0.079 | 0.059 | 1.33× faster |
| 1024×1024×1024 | 16 | 2.155 | 2.834 | 1.32× slower | 3.170 | 2.831 | 1.12× faster |
| 1024×1024×1024 | 32 | 2.062 | 2.665 | 1.29× slower | 3.656 | 2.657 | 1.38× faster |
| 4096×4096×4096 | 16 | 199.1 | 251.1 | 1.26× slower | 272.2 | 258.9 | 1.05× faster |
| 4096×4096×4096 | 32 | 154.4 | 194.5 | 1.26× slower | 243.3 | 199.6 | 1.22× faster |
| 2048×512×4096 | 16 | 9.039 | 12.40 | 1.37× slower | 14.02 | 12.62 | 1.11× faster |
| 2048×512×4096 | 32 | 8.302 | 10.98 | 1.32× slower | 14.26 | 10.93 | 1.31× faster |
| 1000×513×769 | 16 | 0.859 | 1.147 | 1.34× slower | 1.242 | 1.122 | 1.11× faster |
| 1000×513×769 | 32 | 0.827 | 1.071 | 1.30× slower | 1.478 | 1.065 | 1.39× faster |

In both comparisons, padding is the only change.

### 6c. What limits the tiled kernels: time follows shared-memory wavefronts

Profiler, 1024³ (kernel times at base clock):

| Kernel | T | ncu time (ms) | Shared wavefronts, ld + st | **ns per million wavefronts** | SM / memory throughput | Occupancy |
|---|---|---|---|---|---|---|
| tiled | 16 | 5.78 | 54.5 M | 106 | 75% | 98.7% |
| tiled | 32 | 5.31 | 52.4 M | 101 | 74% | 99.9% |
| padded | 16 | 7.62 | 75.5 M | 101 | 85% | 98.7% |
| padded | 32 | 6.94 | 69.2 M | 100 | 88% | 99.96% |
| tiled_ta | 16 | 8.51 | 86.0 M | 99 | 76% | 98.7% |
| tiled_ta | 32 | 9.68 | 101.7 M | 95 | 63% | 100% |
| padded_ta | 32 | 6.92 | 69.2 M | 100 | 88% | 99.96% |
| tiled | 8 | 9.10 | 58.7 M | 155 | 55% | 98.7% |

For T = 16 and 32, **every variant takes 95–106 ns per million shared-memory wavefronts**. The same holds on
1000×513×769 (93–107). So the kernel time is set by how many passes the shared-memory pipe has to make,
regardless of whether those passes come from bank conflicts (tiled_ta) or from extra un-vectorized
instructions (padded). Plot: `results/plots/time_vs_wavefronts.svg`.

T = 8 sits above the line (135–158 ns per million): with 64-thread blocks, the `__syncthreads()` and
per-tile-step overheads dominate rather than the shared-memory pipe.

Padded runs at a **higher** throughput percentage (88% vs 74%) while being **slower**. It's doing more work
(more instructions and wavefronts), not using the hardware better. A busy pipe is not the same as a fast kernel.

---

## 7. Experiment log (successful and unsuccessful)

| # | Experiment | Hypothesis | Result | Kept? |
|---|---|---|---|---|
| 1 | Padding `[T][T+1]` on the tiled kernel (`padded`) | Tiled has no bank conflicts, so padding can't help; it adds 2-way store conflicts at T = 8/16 and may block `LDS.128` | **Confirmed. Padded is slower on every shape and tile size: 1.08–1.37× (1.20–1.32× at T = 32).** Conflicts match the prediction exactly. The main cost is losing `LDS.128`: 1.6× more shared load instructions, 1.33× more wavefronts. The added store conflicts are a minor share (4.2 M of 75.5 M wavefronts at T = 16) | **No.** Unsuccessful as an optimization; it's version 4 by requirement |
| 2 | Transposed A tile, no padding (`tiled_ta`) | 32-way store conflict at T = 32 makes it slower than tiled | Confirmed: 32.5 M store conflicts; 1.58–1.79× slower than tiled at T = 32 | No (demonstration) |
| 3 | Transposed A tile + padding (`padded_ta`) | Padding removes all of experiment 2's conflicts | **Confirmed. Conflicts 32.5 M → 0; 1.22–1.39× faster than tiled_ta at T = 32.** Timing and profile are identical to `padded`: once conflicts are gone, both are limited by the same 32 scalar loads per step | Demonstration: **padding works when there's a column access to fix** |
| 4 | Tile size with padding | Same best T as without padding | Confirmed: T = 32 is fastest for padded on every shape too | T = 32 default |
| 5 | **Register tiling (`regtile`)**, second profiler-guided improvement | Cut shared loads per FMA 1.25 → 0.125 (section 8) | **3.0–3.7× faster than tiled on the large shapes** (final run) | **Yes** |

**Answer to "does padding help on this GPU?"** Not for this kernel. The Week 2 tiled kernel reads shared
memory only along rows (`As[ty][k]` broadcast, `Bs[k][tx]` consecutive), so it has no bank conflicts to
remove. On a T4, padding makes it 8–37% slower (20–32% at the best tile size, T = 32): the odd row length stops the compiler from using
128-bit shared loads, and at T = 8/16 it adds 2-way store conflicts. Padding *does* work exactly as
designed on a kernel with column-wise shared accesses (`tiled_ta` → `padded_ta`: 32.5 M conflicts → 0,
up to 1.39× faster).

---

## 8. The second profiler-guided improvement (milestone)

**Observation (section 6c and Week 2 section 7):**
- Occupancy is 99–100%.
- DRAM is at ~10% of peak; FP32 at ~13% of peak.
- Kernel time is proportional to shared-memory wavefronts.

**Diagnosis:** the tiled kernel is limited by the shared-memory pipe. Every multiply-add needs 1.25
shared-memory loads (one scalar `Bs[k][tx]`, plus a quarter of an `LDS.128` for `As`).

**Change: register tiling**, `--impl regtile` (the spec's first optional extension, kept as a separate
fifth version). The code is `matmul_regtile` in `src/matmul.cu`:

```
block = 16 x 16 threads, owns a 64 x 64 tile of C;   thread (tx, ty) owns C rows ty*4..+3, cols tx*4..+3
for k0 in 0, 8, 16, ... < K:                          # K step = 8
    cooperatively load A[64 rows][8] -> As[k][m]  (stored TRANSPOSED, row length 64 + 4)
    cooperatively load B[8][64 cols] -> Bs[k][n]  (zero-fill outside the matrix)
    __syncthreads()
    for k in 0..7:
        a[0..3] = As[k][ty*4 .. ty*4+3]      # one float4 (128-bit) shared load
        b[0..3] = Bs[k][tx*4 .. tx*4+3]      # one float4 (128-bit) shared load
        acc[i][j] += a[i] * b[j]  for i, j in 0..3     # 16 FMAs, all in registers
    __syncthreads()
store acc[4][4] to C with a bounds check per element
```

| | tiled (T = 32) | regtile |
|---|---|---|
| Outputs per thread | 1 | 16 (4 × 4) |
| Shared-memory load instructions per FMA | 1.25 (scalar + ¼ `LDS.128`) | 0.125 (2 × 128-bit per 16 FMAs) |
| Global loads per output element | 2K/32 | 2K/64 (block tile is 64 × 64) |
| Threads per block / shared memory per block | 1024 / 8 KiB | 256 / 4.1 KiB |
| Registers per thread | low | higher (16 accumulators + 8 operands); `__launch_bounds__(256)` |

**The Week 3 padding lesson, applied.** Storing A transposed makes each thread's 4 A values contiguous
(one `float4` read). The cost is a column-wise store: with a 64-float row, the 8 threads writing one
A row hit the same bank (**8-way conflict**). Padding the row to 64 **+ 4** removes the conflict *and* keeps
every row 16-byte aligned, so the 128-bit reads stay legal. Padding by 1 would break alignment, exactly
the cost measured for `padded` (section 6a). Checked with the same counting model as `bank_model.py`:
PAD = 0 gives an 8-way As store, PAD = 4 gives 1-way.

**Checked without a GPU:**
- A pure-Python emulation of the kernel (real shared-memory layout, zero-fill, guarded stores) matches a
  reference result on 9 shapes, including 1×1×1, 65×63×129, 63×9×65 and K = 1.
- clang parses it cleanly.
- `run_tests.sh` now covers `regtile` and has a new shape, 65×63×129: one past and one short of the 64×64
  block, with K not a multiple of 8.

**Predictions to check on the GPU:**
- **SASS:** two `LDS.U.128` per k in `matmul_regtile`. A test compile with clang, not nvcc, split them
  into scalar loads, so check `results/profile/matmul.sass`. If nvcc also splits them, it's still 0.5
  scalar shared loads per FMA instead of 1.25.
- **Shared-memory wavefronts at 1024³:** at least ~3× lower than tiled's 52.4 M, and **0 bank conflicts**.
- **Time:** if the ~100 ns-per-million-wavefronts relationship held, ~3× faster. But a different limit will
  probably take over first: FP32 issue rate, global-load latency with only 8 warps per block, or
  registers limiting occupancy. **Expected: 1.5–3× faster than tiled T = 32.** The new profile will show
  which limit it hits.

| Field | Entry: register tiling |
|---|---|
| Observation | Shared-memory pipe is the limiter: time ∝ wavefronts (≈100 ns per million), 1.25 shared loads per FMA, occupancy 99%, DRAM ~10% |
| Hypothesis | Computing a 4×4 block per thread from two 128-bit shared loads per k cuts shared loads per FMA 10×, wavefronts ≥3× |
| Change | `matmul_regtile`: 64×64 block tile, BK = 8, 4×4 per thread, As transposed with row padding 4 |
| Expected effect | Wavefronts ≥3× lower, 0 bank conflicts; kernel time 1.5–3× lower than tiled T = 32 |
| Measured effect | Final run (13 Nov), tiled T=32 → regtile kernel time: 1024³ 2.063 → 0.608 ms (**3.39×**), 4096³ **3.53×**, 2048×512×4096 **3.72×**, 1000×513×769 **3.04×**, 256³ 1.27×. Profile (1024³): shared load instructions 41.9 M → 4.19 M (10×); shared wavefronts 52.4 M → 13.6 M (3.9×); 0 bank conflicts; SASS shows the two `LDS.U.128` per k as designed |
| Conclusion | **Hypothesis confirmed, effect larger than predicted** (3.0–3.7× vs the predicted 1.5–3×). The shared-memory pipe is no longer the limiter: MIO stalls fall from 65% to 44% of cycles, and SM and memory throughput are both below 50%. Nsight now points at latency, a 1.6-wave tail (256 blocks for 160 slots, est. up to 50%), and uncoalesced C stores (8 of 32 bytes per sector), which `vec` later fixes. Full analysis: docs/week4.md §4d–4f |

---

## 9. Week 3 milestone checklist

1. ☑ `make test` passes for tiled, padded, tiled_ta and padded_ta at T = 8, 16, 32 (final run: 327 / 327, `results/tests.txt`).
2. ☑ `make profile` run; tables 6a and 6c filled in; predicted and measured conflicts compared (exact match).
3. ☑ `make bench` run; table 6b filled in.
4. ☑ "Does padding help on this GPU?" answered with evidence (section 7).
5. ☑ Second profiler-guided improvement: diagnosed, implemented and measured (`regtile`, 3.0–3.7×, section 8).
6. ☑ Experiment log (section 7) includes the unsuccessful experiment (padding the tiled kernel).
