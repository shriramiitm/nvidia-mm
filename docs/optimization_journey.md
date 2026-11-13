# The optimization journey: from a CPU loop to vectorized CUDA on a Tesla T4

Written in the style of a step-by-step optimization log. Each step has the same parts:
- **Concept:** the idea, and what changes from the previous step.
- **Code:** where it is in `src/matmul.cu`.
- **Prediction:** what should happen, worked out before measuring.
- **Results:** a 4096³ table that grows by one row per step, plus a graph across matrix sizes.
- **Nsight Compute:** what the profiler says.
- **Analysis:** why, and what to try next.

> **Status.** Steps 3 (uncoalesced naive), 6 (1D tiling) and 7 (plain 2D tiling) are new. So are the
> hardware-spec query, the size sweep and the roofline. Their code has been checked offline: a CPU
> emulation matches a reference on 8 shapes, and clang parses it cleanly. **Their numbers need one more GPU
> run** (`notebooks/colab_reproduce.ipynb`). Cells marked *pending* get filled in from that run. Numbers
> for the other steps come from the final run of 13 Nov 2026 and will be refreshed from the same new run, so
> every number on this page comes from one session.

**How to regenerate everything on this page:** `bash scripts/reproduce.sh` produces:
- `results/hw_specs.md`: §1;
- `results/sweep.csv`: the graphs across sizes (n = 128 … 2048, 10 trials each; the 4096 points come from
  the benchmark, 20 trials);
- `results/results.csv` → `results/ladder_4096.md`: the per-step tables;
- `results/profile/`: Nsight;
- `results/plots/`: every graph named below.

---

## 1. Hardware specifications

All benchmarks run on an **NVIDIA Tesla T4** on Google Colab. `./devinfo` (built from `src/devinfo.cu`)
reads the limits from the CUDA runtime, and `scripts/hw_specs.py` adds clocks from `nvidia-smi` and derives
the peaks. Expected values:

| Specification | Tesla T4 |
|---|---|
| Architecture | Turing, compute capability 7.5 |
| SMs / FP32 cores | 40 SMs × 64 = 2,560 cores |
| Max SM clock (boost) | 1590 MHz |
| **FP32 peak** = 40 × 64 × 2 × 1.59 GHz | **8.14 TFLOP/s** |
| Memory | 15 GiB GDDR6, 256-bit bus, 5001 MHz |
| **DRAM bandwidth** = 5001 MHz × 2 × 256 / 8 | **320 GB/s** |
| **Ridge point** = 8.14 TFLOP/s ÷ 320 GB/s | **≈ 25 FLOP/byte** |
| L2 cache | 4 MiB |
| Shared memory per SM / per block | 64 KiB / 48 KiB |
| Registers per SM | 65,536 (32-bit) |
| Max threads per SM / per block | 1024 / 1024 (so 32 warps per SM) |
| Power limit | 70 W |

The measured table replaces this one in `results/hw_specs.md`.

**What the numbers mean for GEMM.**
- At 4096³ a matmul does 2·4096³ ≈ **137 GFLOP** on 3 × 64 MB = **201 MB** of unique data. That's
  ≈ 680 FLOP per byte if each byte were read once, far above the ridge. So a good GEMM should be
  **compute-bound**; the question at each step is how far we are from that.
- The T4 has only **32 warps per SM** (1024 threads). A 1024-thread block, or a kernel using
  64 registers per thread, can already fill an SM. That matters for occupancy at steps 5–8.

**Roofline** (`results/plots/roofline.svg`): each kernel is a point at (FLOP per DRAM byte measured by
Nsight, achieved GFLOP/s). The diagonal is the 320 GB/s memory roof, and the flat line is the 8.14 TFLOP/s
compute roof.

---

## 2. CPU baseline, and why loop order matters

**Concept.** The same triple loop in two orders (`src/matmul.cu:154`, `:169`):

| | i-j-k (`--impl cpu_ijk`) | i-k-j (`--impl cpu`, the baseline) |
|---|---|---|
| Inner loop walks | a row of A and **a column of B** (stride N) | **a row of B and a row of C** (stride 1) |
| Cache-line use on B | 4 of 64 bytes per miss | 64 of 64 bytes |
| SIMD | no (strided) | yes (contiguous) |

**Prediction.** i-k-j should be several times faster, and the gap should grow with size once a column of B
no longer fits in cache.

**Results** (`results/plots/sweep_cpu.svg`, sizes 128–512): *pending.*
For reference, the final run measured i-k-j alone: 6.8 GFLOP/s at 256³, 7.3 at 1024³, 5.3 at 4096³.

**Analysis.** *Pending the i-j-k numbers.* The i-k-j slowdown at 4096³ already shows its own limit: it
streams all 64 MB of B for every row of C. The CPU equivalent of GPU tiling (cache blocking) would fix that,
but the spec keeps the baseline as plain three loops.

---

## 3. Naive GPU: one thread per output, uncoalesced

**Concept.** One thread computes one C[row][col] with a full length-K dot product from global memory. This
version (`matmul_naive_uncoal`, `src/matmul.cu:198`) maps **`threadIdx.x` → row**, the mapping in the
reference post's first kernel, so a warp's threads are on different rows:

```cpp
int row = blockIdx.x * blockDim.x + threadIdx.x;   // neighbouring threads → neighbouring ROWS
int col = blockIdx.y * blockDim.y + threadIdx.y;
```

**Prediction.** In one load, the 16 threads of a half-warp read `A[row*K + k]` at addresses **K floats
apart**: 16 separate 32-byte sectors, 4 useful bytes each. C is written N floats apart. Expect very low
load efficiency, and the slowest GPU version by a wide margin.

**Results.** *Pending:* the first row of the ladder table, and `results/plots/sweep_1_naive_uncoal.svg`.

| Kernel | Time (ms) | TFLOPS | Bandwidth (GB/s) | vs cuBLAS | Speedup over naive |
|---|---|---|---|---|---|
| Naive (uncoalesced) | *pending* | | | | 1.0× |

**Nsight to check:** global load efficiency and sectors per request; the "uncoalesced global accesses"
rule; warp stalls on the global-memory (LG) queue.

**Bridge.** The arithmetic isn't the problem; the access pattern is. The fix is a one-line swap.

---

## 4. Global-memory coalescing

**Concept.** Swap the mapping so **`threadIdx.x` → column** (`matmul_naive`, `src/matmul.cu:182`). This is
the project's required naive version (the spec asks it to "coalesce where practical"):

```cpp
int col = blockIdx.x * blockDim.x + threadIdx.x;   // neighbouring threads → neighbouring COLUMNS
int row = blockIdx.y * blockDim.y + threadIdx.y;
```

A warp now reads 16 consecutive floats of a B row (2 full sectors) and broadcasts one A value per row.
Predicted load efficiency is (64 + 8)/(64 + 64) = **56.25%**; Nsight measured exactly that in the final run.

**Results** (final run, 4096³; the uncoalesced comparison is pending):

| Kernel | Time (ms) | TFLOPS | Bandwidth (GB/s) | vs cuBLAS | Speedup over naive |
|---|---|---|---|---|---|
| Naive (uncoalesced) | *pending* | | | | 1.0× |
| Coalesced | 331.4 | 0.41 | 0.61 | 6.3% | *pending* |

**Nsight** (final run, 1024³):
- 67.1 M global load requests for 8 MB of data: each element is read 1024 times.
- Warps stall **28.9 of 35.6 cycles (81%)** on the global-memory queue.
- Compute pipelines are under-utilized.
- DRAM is only 7.6% busy: the L2 cache absorbs most of the repeated reads.

**Analysis.** Coalescing fixes *how* each load is served, but every value is still loaded K times. The
bottleneck is now the **number** of load requests.

**Bridge.** Threads in the same block need the same data, so let them load it once and share it.

---

## 5. Shared-memory caching (tiled)

**Concept** (`matmul_tiled<32,0,false>`, `src/matmul.cu:210`, figure `docs/figures/tiling.svg`).
- A 32×32 block owns a 32×32 tile of C and walks along K.
- Each step it cooperatively loads one 32×32 tile of A and one of B into shared memory, synchronizes,
  computes 32 multiply-adds per thread from shared memory, and synchronizes again.
- Zero-fill and a guarded store handle edges.

**Prediction.** Global load requests drop exactly 32×. Shared reads are row-wise or broadcast, so there are
0 bank conflicts.

**Results** (final run, 4096³):

| Kernel | Time (ms) | TFLOPS | Bandwidth (GB/s) | vs cuBLAS | Speedup over naive |
|---|---|---|---|---|---|
| Coalesced | 331.4 | 0.41 | 0.61 | 6.3% | – |
| Shared memory (T=32) | 168.4 | 0.82 | 1.20 | 12.4% | 2.0× over coalesced |

**Nsight** (1024³):
- Load requests 67.1 M → **2.10 M**, exactly 32× fewer; load efficiency 100%; 0 bank conflicts.
- The new limiter: warps stall **24.0 of 36.9 cycles (65%) on the MIO (shared-memory) queue**.
- Across all tiled variants, kernel time is proportional to shared-memory wavefronts, about 100 ns per
  million (`results/plots/time_vs_wavefronts.svg`).

**Occupancy.** 1024-thread blocks fill the T4's 32 warps per SM with one block, so achieved occupancy is
100%. Unlike the reference post's RTX 4090 (48 warps per SM, so 66.7% there), occupancy isn't the problem
here.

**Analysis.** Each multiply-add needs **two shared-memory reads**, so the kernel can't go faster than
shared memory can feed it. Only 1.7–2× faster than coalesced, because L2 was already absorbing much of the
redundancy.

### 5b. Shared memory with padding (project requirement)

**Concept.** The same kernel with `As[32][33]`, `Bs[32][33]`. Padding exists to fix **column** access to
shared memory (`docs/figures/banks.svg`).

**Prediction** (bank model): tiled has no column access, so padding can only cost. Rows of 33 floats aren't
16-byte aligned, so nvcc loses its 128-bit `LDS.128` reads.

**Results** (final run): **21–31% slower** than tiled on every shape. 4096³: 216.6 ms vs 168.4 ms.

**Nsight:** shared-load instructions **41.9 M → 67.1 M** (1.6×); 0 conflicts at T = 32, but 4.19 M new
store conflicts at T = 16, exactly as predicted. When A is deliberately stored transposed (a column write),
padding takes conflicts from **32.5 M to 0** and is 1.37× faster. So padding works, but only where there's a
column access.

**Bridge.** The real limit is shared-memory traffic per multiply-add. Do more math per shared load.

---

## 6. 1D block tiling

**Concept** (`matmul_tile1d`, `src/matmul.cu:325`).
- Each thread computes **TM = 8 outputs down one column** of C.
- Block = 512 threads, owning a 64×64 tile; K step 8.
- Per k, a thread reads **one** B value into a register and reuses it for 8 multiply-adds with 8 A values:
  9 shared loads per 8 FMAs (1.1 per FMA) instead of 2 per FMA.

```cpp
for (int k = 0; k < 8; k++) {
    const float b = Bs[k * 64 + tc];                       // one load, kept in a register
    for (int i = 0; i < 8; i++) acc[i] += As[(tr * 8 + i) * 8 + k] * b;   // reused 8 times
}
```

**Prediction.**
- Bank model: `As[(tr*8+i)*8+k]` is the same address for a whole warp (a broadcast), and `Bs[k*64+tc]` is
  consecutive, so **0 conflicts**.
- Shared-load instructions per FMA fall about 1.8×.
- Each value fetched from global memory is reused 64× within the block (vs 32× for tiled).
- Expect a clear gain over tiled. The reference post saw 3.0× on its GPU; ours will differ because the T4 is
  smaller.

**Results.** *Pending:* the 4096³ row in `results/ladder_4096.md`, and `results/plots/sweep_5_tile1d.svg`.

**Nsight to check:** MIO stall share vs tiled's 65%; the instruction mix (shared loads as a share of all
instructions, `shld_inst / inst_total` in `profile_summary.csv`); occupancy (512-thread blocks, 2 per SM).

**Bridge.** Reuse in one direction helped. Reuse in both directions is an outer product.

---

## 7. 2D block tiling

**Concept** (`matmul_tile2d`, `src/matmul.cu:373`; figure `docs/figures/regtile.svg`).
- Each thread computes an **8×8** block of C.
- Block = 256 threads, owning a 128×128 tile; K step 8.
- Per k: 8 A values and 8 B values go into registers, then 64 multiply-adds. That's 16 shared loads per 64
  FMAs (0.25 per FMA).
- Kept **deliberately plain**, like the reference post's kernel: scalar global loads, A tile row-major, each
  thread owning **8 adjacent columns**.

**Prediction** (bank model):
- the A reads `As[(tr*8+i)*8+k]` are **2-way conflicts**: a warp spans two thread rows, 64 words apart, so
  the same bank;
- the B reads `Bs[k*128+tc*8+j]` are **4-way conflicts**: neighbouring threads 8 words apart.

Even so, the 4× higher arithmetic per load should win clearly over 1D tiling. Nsight should flag the shared
bank conflicts, as it did in the reference post (5.6-way there).

**Results.** *Pending:* the 4096³ row, and `results/plots/sweep_6_tile2d.svg`.

**Related measured result** (final run): `regtile`, a 4×4 version with conflict-free 128-bit shared reads, was
**3.0–3.7× faster than tiled**. Shared-load instructions fell 10× and MIO stalls went from 65% to 44%.

**Bridge.** Nsight will flag the conflicts and the scalar, uncoalesced C stores. Vectorize the accesses and
fix the layout.

---

## 8. Vectorized memory access

**Concept** (`matmul_vec<128,128,8,8,8>` = `--impl vec128`, `src/matmul.cu:432`). The same 8×8 tiling as
step 7, with these changes:
- **`float4` (128-bit) global loads** of A and B (`LDG.E.128`), and `float4` C stores;
- **A stored transposed**, so a thread's 8 A values are contiguous (two `float4` shared reads), with the row
  **padded by 4**: conflict-free *and* 16-byte aligned;
- each thread's 8 columns split into **two groups of 4, 64 apart**, so neighbouring threads read neighbouring
  16-byte chunks: conflict-free.

Alignment: `float4` needs 16-byte-aligned addresses, so it's used only when K % 4 = 0 (A rows) or
N % 4 = 0 (B, C rows) and all 4 elements are in range; otherwise there's a scalar fallback. That's why it
works for any size.

**Results** (final run, 4096³):

| Kernel | Time (ms) | TFLOPS | Bandwidth (GB/s) | vs cuBLAS | Speedup over naive |
|---|---|---|---|---|---|
| Vectorized (8×8) | 25.47 | **5.40** | 7.90 | **81.8%** | *pending* |
| cuBLAS | 20.85 | 6.59 | 9.66 | 100% | |

*Pending:* the speedup over step 7 (tile2d), which isolates vectorization plus the layout fix at the same
tiling.

**Nsight** (final run, 1024³):
- 0 bank conflicts.
- Warp cycles per issued instruction **6.6** (vs 17.9 for the 4×4 kernels).
- The **FMA pipe is the busiest unit (62%)**, the first kernel where compute dominates.
- 123 registers per thread, so 50% theoretical occupancy.
- Only 64 blocks at 1024³ (0.8 waves).

**Analysis.**
- **What vectorization alone does.** Measured separately in the final run (`vec` vs `regtile`, same
  4×4 tiling): 4× fewer global load requests and fully coalesced stores, **but 1.00× the time**. regtile's
  shared reads were already 128-bit (SASS), and global memory wasn't its bottleneck.
- **On odd sizes it's slower.** The scalar fallback is uncoalesced: 6.2 of 32 bytes per sector are used.
- **Where vec128's gain comes from:** the bigger per-thread tile and the conflict-free layout. The pending
  tile2d → vec128 comparison will show how much of the step-7 → step-8 gain that is.

---

## 9. Summary (filled in from the next run)

`results/ladder_4096.md` gives the full cumulative table, one row per step, with columns: time, TFLOPS,
effective bandwidth, % of cuBLAS, speedup over naive, speedup over the previous step.

Graphs:
- `results/plots/ladder_4096.svg`: TFLOPS per step;
- `results/plots/sweep_all.svg`: every step across sizes;
- `results/plots/sweep_<n>_<step>.svg`: one per step, against the previous step and cuBLAS;
- `results/plots/roofline.svg`.

| Step | What changed | Bottleneck removed | New bottleneck (Nsight) |
|---|---|---|---|
| 2 → 3 | CPU → one thread per output | serial execution | uncoalesced global loads (*pending*) |
| 3 → 4 | `threadIdx.x` → column | uncoalesced access | redundant global loads (81% LG stalls) |
| 4 → 5 | shared-memory tiles | redundant global loads (32× fewer) | shared-memory pipe (65% MIO stalls) |
| 5 → 5b | padding | nothing (no conflicts) | lost 128-bit loads |
| 5 → 6 | 8 outputs per thread (1D) | shared loads per FMA 2 → 1.1 | *pending* |
| 6 → 7 | 8×8 outputs per thread (2D) | shared loads per FMA → 0.25 | bank conflicts, latency (*pending*) |
| 7 → 8 | float4, transposed padded A, column groups | conflicts, instruction count | FMA 62%, occupancy, block count |
