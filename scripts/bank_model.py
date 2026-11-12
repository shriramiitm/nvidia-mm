#!/usr/bin/env python3
"""Predict shared-memory bank conflicts for the tiled kernels, without a GPU.

Usage:  python3 scripts/bank_model.py

Shared memory has 32 banks, each 4 bytes wide; 32-bit word address w lives in
bank w % 32. When a warp executes one shared-memory instruction, threads that
hit the SAME word are served together (broadcast), but threads that hit
DIFFERENT words in the same bank are serialized. The instruction needs
    wavefronts = max over banks of (number of distinct words in that bank)
and an ideal instruction needs 1. "Extra wavefronts" (wavefronts - 1, summed
over warps) is what Nsight Compute reports as bank conflicts
(l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_{ld,st}.sum).

The model assumes 32-bit accesses, one per thread, as written in the source.
If the compiler merges As[ty][k..k+3] into one 128-bit load (possible only when
rows are 16-byte aligned, i.e. no padding), the real instruction count differs;
check the SASS with  cuobjdump -sass ./matmul | grep LDS .

Variants (all in src/matmul.cu):
    tiled      As[T][T]     Bs[T][T]     store As[ty][tx], read As[ty][k]
    padded     As[T][T+1]   Bs[T][T+1]   same accesses as tiled
    tiled_ta   As[T][T]     ...          store As[tx][ty], read As[k][ty]  (A tile transposed)
    padded_ta  As[T][T+1]   ...          same accesses as tiled_ta
"""

BANKS = 32
WARP = 32


def wavefronts(addrs):
    """addrs: word addresses touched by the active threads of one warp."""
    per_bank = {}
    for w in set(addrs):
        per_bank.setdefault(w % BANKS, set()).add(w)
    return max(len(s) for s in per_bank.values())


def extra_per_block(T, addr_fn):
    """Sum of (wavefronts - 1) over all warps of a T x T block for one instruction."""
    threads = [(ty, tx) for ty in range(T) for tx in range(T)]  # linear id = ty*T + tx
    total = 0
    for w0 in range(0, len(threads), WARP):
        warp = threads[w0:w0 + WARP]
        total += wavefronts([addr_fn(ty, tx) for ty, tx in warp]) - 1
    return total, (len(threads) + WARP - 1) // WARP


def analyse(T, pad, transpose_a):
    LD = T + pad  # physical leading dimension (row length in words)
    if transpose_a:
        st_a = lambda ty, tx: tx * LD + ty
        ld_a = lambda k: (lambda ty, tx: k * LD + ty)
    else:
        st_a = lambda ty, tx: ty * LD + tx
        ld_a = lambda k: (lambda ty, tx: ty * LD + k)
    st_b = lambda ty, tx: ty * LD + tx
    ld_b = lambda k: (lambda ty, tx: k * LD + tx)

    out = {}
    out["st As"], warps = extra_per_block(T, st_a)
    out["st Bs"], _ = extra_per_block(T, st_b)
    out["ld As"] = sum(extra_per_block(T, ld_a(k))[0] for k in range(T))
    out["ld Bs"] = sum(extra_per_block(T, ld_b(k))[0] for k in range(T))
    # per-instruction worst case, to show the "N-way" degree
    out["worst st As"] = max(wavefronts([st_a(ty, tx) for ty in range(T) for tx in range(T)][w:w + WARP])
                             for w in range(0, T * T, WARP))
    return out, warps


def main():
    print("Extra wavefronts (bank conflicts) per block per K-tile step.")
    print("Multiply by blocks * ceil(K/T) for a whole kernel.\n")
    hdr = f"{'variant':<10} {'T':>3} {'smem B':>7} {'warps':>5} | {'st As':>6} {'st Bs':>6} {'ld As':>6} {'ld Bs':>6} | worst st As"
    print(hdr)
    print("-" * len(hdr))
    for name, pad, ta in [("tiled", 0, False), ("padded", 1, False), ("tiled_ta", 0, True), ("padded_ta", 1, True)]:
        for T in (8, 16, 32):
            r, warps = analyse(T, pad, ta)
            smem = 2 * T * (T + pad) * 4
            print(f"{name:<10} {T:>3} {smem:>7} {warps:>5} | {r['st As']:>6} {r['st Bs']:>6} "
                  f"{r['ld As']:>6} {r['ld Bs']:>6} | {r['worst st As']}-way")
        print()


if __name__ == "__main__":
    main()
