#!/usr/bin/env python3
"""Check an output file against an independent float64 NumPy reference.

Usage: python3 check_cpu.py INPUT.bin OUT.bin
Used to make sure the CPU baseline itself is right before trusting it as the
reference for every GPU version.  Exit status 0 = pass.
"""
import sys

import numpy as np


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    inp, out = sys.argv[1], sys.argv[2]
    with open(inp, "rb") as f:
        M, K, N = (int(x) for x in np.fromfile(f, dtype="<i4", count=3))
        A = np.fromfile(f, dtype="<f4", count=M * K).reshape(M, K)
        B = np.fromfile(f, dtype="<f4", count=K * N).reshape(K, N)
    C = np.fromfile(out, dtype="<f4")
    if C.size != M * N:
        print(f"FAIL: expected {M * N} values, found {C.size}")
        sys.exit(1)

    ref = A.astype(np.float64) @ B.astype(np.float64)
    C = C.reshape(M, N).astype(np.float64)
    finite = bool(np.isfinite(C).all())
    err = np.abs(C - ref)
    ok = finite and bool((err <= 1e-4 * K + 1e-4 * np.abs(ref)).all())
    print(f"max_abs_err={err.max():.3e} finite={finite} {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
