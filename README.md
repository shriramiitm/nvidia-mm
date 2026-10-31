# Matrix multiplication: CPU to optimized CUDA (Week 1)

Status: `cpu` and `naive` work. `tiled` and `padded` are placeholders (Weeks 2-3).

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
    ./compare in.bin ref.bin out.bin               # exit 0 = PASS

## Tests and benchmarks
    make test      # correctness suite (CPU vs NumPy, GPU vs CPU)
    make bench     # results/results.csv + results/env.txt

CPU loop order is i-k-j (see comment in `matmul.cu` for the locality argument).
Tolerance: `abs_err <= 1e-4*K + 1e-4*|ref|`.
