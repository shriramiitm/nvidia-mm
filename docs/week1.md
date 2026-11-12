# Week 1: CPU baseline, test infrastructure, naive CUDA

**Week 1 goal (from the spec):** the CPU and naive GPU versions pass the correctness suite and have an initial performance table.

Code for this week is tagged `week1` in git (`git checkout week1` to see it as it was).

---

## 1. What was built

| Piece | File | Notes |
|---|---|---|
| Input generator | `scripts/gen.py` | `int32 M,K,N` + row-major `float32 A, B`; values uniform in [-1, 1], fixed seed |
| Single-threaded CPU baseline | `src/matmul.cu` (`cpu_matmul`) | one host thread, no OpenMP/BLAS; loop order i-k-j (section 2) |
| Naive CUDA kernel | `src/matmul.cu` (`matmul_naive`) | one thread per `C[row][col]`, 16×16 blocks, 2D grid, bounds check, no shared memory |
| GPU memory, transfers, error checks | `src/matmul.cu` (`gpu_run`, `CUDA_CHECK`) | every CUDA call checked; free-memory check before allocating |
| Output comparison | `src/compare.cpp` | max abs / rel error, mixed tolerance, rejects NaN/Inf |
| CPU sanity check | `scripts/check_cpu.py` | CPU output vs an independent float64 NumPy result |
| Correctness suite | `scripts/run_tests.sh` | 15 shapes + bad-argument tests |
| Benchmark | `scripts/bench.sh` | kernel-only (CUDA events) and end-to-end (host monotonic clock), warm-up, median of 20 |

---

## 2. CPU loop order: i-k-j

```
for i in 0..M-1:
    for k in 0..K-1:
        a = A[i][k]                 # loop-invariant scalar in the inner loop
        for j in 0..N-1:
            C[i][j] += a * B[k][j]  # B row k and C row i walked with stride 1
```

All matrices are row-major. In the inner `j` loop, `B[k][*]` and `C[i][*]` are read
contiguously (stride 1), and `A[i][k]` stays in a register. Every cache line that is
fetched is used completely, and the compiler can vectorize the inner loop.

The textbook order i-j-k would read `B[k][j]` down a column (stride N floats). For
large N, each access touches a new cache line and uses only 4 of its 64 bytes.

---

## 3. Naive kernel: thread mapping and coalescing

```
block = (16, 16)        grid = (ceil(N/16), ceil(M/16))
col = blockIdx.x*16 + threadIdx.x
row = blockIdx.y*16 + threadIdx.y
if row < M and col < N:
    C[row][col] = sum_k A[row][k] * B[k][col]
```

- `col` comes from `threadIdx.x`, so threads next to each other in a warp read
  `B[k][col..col+15]` from consecutive addresses (coalesced) and write consecutive
  elements of C.
- `A[row][k]` is the same address for all threads in that row of the warp (a broadcast).
- Cost: each output element needs 2·K global loads, so every element of A is reloaded
  N times and every element of B M times. Week 2's tiling targets this.

---

## 4. Correctness tests

Tolerance per element: `|out - ref| <= 1e-4*K + 1e-4*|ref|`.

| Requirement from the spec | Shapes in `scripts/run_tests.sh` |
|---|---|
| Square / tall / wide | 64³, 200×16×8, 8×16×200 |
| Smaller than a CUDA block | 5×3×7 |
| Exactly divisible by the tile size | 16³, 32³ |
| One less / one more than a tile multiple | 7×9×9, 15×17×16, 31³, 33³ |
| M, K or N = 1 | 1×64×64, 64×1×64, 64×64×1, 1×1×1 |
| Irregular | 1000×513×769 |

Run (on the GPU machine):

```
make && IMPLS="naive" bash scripts/run_tests.sh | tee results/tests_week1.txt
```

Result: *fill in: passed / failed counts, worst max_abs_err*

---

## 5. Initial performance table (fill from `results/results.csv`)

```
IMPLS="cpu naive" bash scripts/bench.sh
```

| Shape (M, K, N) | CPU (ms) | Naive kernel (ms) | Naive end-to-end (ms) | Kernel speedup vs CPU | Naive kernel GFLOP/s |
|---|---|---|---|---|---|
| 256, 256, 256 | | | | | |
| 1024, 1024, 1024 | | | | | |
| 4096, 4096, 4096 | | | | | |
| 2048, 512, 4096 | | | | | |
| 1000, 513, 769 | | | | | |

GFLOP/s = 2·M·N·K / time. Environment (CPU, GPU, CUDA version, flags): see `results/env.txt`.

---

## 6. Week 1 milestone checklist

1. ☐ `run_tests.sh` passes for `cpu` and `naive` on the GPU machine (output saved in `results/`).
2. ☐ `bench.sh` run for `cpu naive`, table in section 5 filled in.
3. ☐ `results/` committed.
