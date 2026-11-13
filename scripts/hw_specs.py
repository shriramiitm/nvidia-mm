#!/usr/bin/env python3
"""Hardware specifications and theoretical peaks for the roofline model.

Usage:  python3 scripts/hw_specs.py > results/hw_specs.md        (needs ./devinfo and nvidia-smi)
        python3 scripts/hw_specs.py --json > results/hw_specs.json

Combines ./devinfo (limits from the CUDA runtime) with nvidia-smi (max clocks,
power limit). Derived peaks:
  FP32 peak (FLOP/s)   = SMs x FP32 cores per SM x 2 (an FMA is 2 FLOPs) x max SM clock
  DRAM bandwidth (B/s) = max memory clock x 2 (double data rate) x bus width / 8
  ridge point          = FP32 peak / DRAM bandwidth  (FLOP per byte where the roofline bends)
The FP32-cores-per-SM table covers recent architectures; unknown ones are reported as such.
"""
import json
import subprocess
import sys

FP32_CORES_PER_SM = {(6, 0): 64, (6, 1): 128, (7, 0): 64, (7, 2): 64, (7, 5): 64, (8, 0): 64,
                     (8, 6): 128, (8, 7): 128, (8, 9): 128, (9, 0): 128, (10, 0): 128, (12, 0): 128}
ARCH = {6: "Pascal", 7: "Volta/Turing", 8: "Ampere/Ada", 9: "Hopper", 10: "Blackwell", 12: "Blackwell"}


def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout


def main():
    dev = {}
    for line in run(["./devinfo"]).splitlines():
        k, _, v = line.partition("=")
        dev[k] = int(v) if v.lstrip("-").isdigit() else v
    smi = run(["nvidia-smi", "--query-gpu=clocks.max.sm,clocks.max.mem,power.limit,driver_version",
               "--format=csv,noheader,nounits"]).strip().splitlines()[0]
    sm_mhz, mem_mhz, power_w, driver = [x.strip() for x in smi.split(",")]
    sm_mhz, mem_mhz = float(sm_mhz), float(mem_mhz)

    cc = (dev["cc_major"], dev["cc_minor"])
    cores = FP32_CORES_PER_SM.get(cc)
    peak = dev["sm_count"] * cores * 2 * sm_mhz * 1e6 if cores else None
    bw = mem_mhz * 1e6 * 2 * dev["memory_bus_width_bits"] / 8
    ridge = peak / bw if peak else None
    out = dict(dev, sm_max_mhz=sm_mhz, mem_max_mhz=mem_mhz, power_limit_w=power_w, driver=driver,
               fp32_cores_per_sm=cores, fp32_peak_tflops=peak / 1e12 if peak else None,
               dram_bw_gbs=bw / 1e9, ridge_flop_per_byte=ridge)
    if "--json" in sys.argv:
        print(json.dumps(out, indent=1))
        return

    kib = lambda b: f"{b / 1024:.0f} KiB"
    rows = [
        ("GPU", dev["name"]),
        ("Architecture / compute capability", f"{ARCH.get(cc[0], '?')}, sm_{cc[0]}{cc[1]}"),
        ("Streaming multiprocessors (SMs)", dev["sm_count"]),
        ("FP32 cores per SM / total", f"{cores} / {cores * dev['sm_count']}" if cores else "unknown"),
        ("Max SM clock (boost)", f"{sm_mhz:.0f} MHz"),
        ("**FP32 peak** (SMs × cores × 2 × clock)", f"**{peak / 1e12:.2f} TFLOP/s**" if peak else "unknown"),
        ("Global memory", f"{dev['global_memory_mib'] / 1024:.1f} GiB"),
        ("Memory bus / max memory clock", f"{dev['memory_bus_width_bits']}-bit / {mem_mhz:.0f} MHz"),
        ("**DRAM bandwidth** (clock × 2 × bus / 8)", f"**{bw / 1e9:.0f} GB/s**"),
        ("**Ridge point** (peak / bandwidth)", f"**{ridge:.1f} FLOP/byte**" if ridge else "unknown"),
        ("L2 cache", f"{dev['l2_cache_bytes'] / 2**20:.1f} MiB"),
        ("Shared memory per SM / per block (default, opt-in)",
         f"{kib(dev['smem_per_sm_bytes'])} / {kib(dev['smem_per_block_bytes'])}, {kib(dev['smem_per_block_optin_bytes'])}"),
        ("Registers per SM / per block", f"{dev['regs_per_sm']} / {dev['regs_per_block']} (32-bit)"),
        ("Max threads per SM / per block", f"{dev['max_threads_per_sm']} / {dev['max_threads_per_block']}"),
        ("Max resident blocks per SM", dev["max_blocks_per_sm"]),
        ("Warp size", dev["warp_size"]),
        ("Power limit", f"{power_w} W"),
        ("Driver / CUDA runtime", f"{driver} / {dev['cuda_runtime']}"),
    ]
    print("| Specification | Value |\n|---|---|")
    for k, v in rows:
        print(f"| {k} | {v} |")
    warps = dev["max_threads_per_sm"] // dev["warp_size"]
    print(f"\nMax resident warps per SM = {dev['max_threads_per_sm']} / {dev['warp_size']} = {warps}. "
          f"Occupancy = active warps / {warps}.")
    print(f"A kernel below {ridge:.1f} FLOP/byte (DRAM traffic) can be at most memory-bound; above it, "
          f"compute-bound." if ridge else "")


if __name__ == "__main__":
    main()
