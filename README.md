# Matrix multiplication: CPU to optimized CUDA (Week 1)

Status: `cpu` and `naive` (Week 1) and `tiled` (Week 2, tile sizes 8/16/32) are implemented. `padded` is a placeholder (Week 3).
See `WEEK2_IMPROVEMENTS.md` for what changed from Week 1 to Week 2 and the profiling plan.

## Input / output format
Input (native little-endian, no padding): `int32 M, K, N`, then `float32 A[M*K]`
and `float32 B[K*N]`, both row-major. Output: exactly `M*N` float32 values,
row-major, no header. Element `A[i][j]` lives at index `i*K + j`.

## Build
    make              # builds ./matmul (nvcc) and ./compare (g++)
    make ARCH=sm_75   # if -arch=native is not supported by your CUDA version

## Use
    python3 gen.py 1000 513 769 42 in.bin          # M K N SEED OUT
    ./matmul --impl cpu   --input in.bin --output ref.bin
    ./matmul --impl naive --input in.bin --output out.bin
    ./matmul --impl tiled --tile 16 --input in.bin --output out2.bin   # tile = 8, 16 or 32
    ./compare in.bin ref.bin out.bin               # exit 0 = PASS

## Tests and benchmarks
    make test      # correctness suite (CPU vs NumPy, GPU vs CPU)
    make bench     # results/results.csv + results/env.txt
    make profile   # Nsight Compute reports + metric CSVs in results/profile/ (needs ncu)

CPU loop order is i-k-j (see comment in `matmul.cu` for the locality argument).
Tolerance: `abs_err <= 1e-4*K + 1e-4*|ref|`.
