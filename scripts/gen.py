#!/usr/bin/env python3
"""Input generator for the matmul project.

Usage:  python3 scripts/gen.py M K N SEED OUT.bin

File layout (native little-endian, no padding):
    int32   M, K, N
    float32 A[M*K]   row-major
    float32 B[K*N]   row-major

Values are drawn uniformly from [-1, 1] with a fixed seed, so the same
arguments always produce the identical file.
"""
import sys

import numpy as np


def main():
    if len(sys.argv) != 6:
        sys.exit(__doc__)
    try:
        M, K, N, seed = (int(x) for x in sys.argv[1:5])
    except ValueError:
        sys.exit("M, K, N and SEED must be integers")
    out = sys.argv[5]
    if min(M, K, N) < 1:
        sys.exit("M, K and N must be >= 1")

    rng = np.random.default_rng(seed)
    A = rng.uniform(-1.0, 1.0, size=(M, K)).astype("<f4")
    B = rng.uniform(-1.0, 1.0, size=(K, N)).astype("<f4")

    with open(out, "wb") as f:
        np.array([M, K, N], dtype="<i4").tofile(f)
        A.tofile(f)  # C order == row-major
        B.tofile(f)


if __name__ == "__main__":
    main()
