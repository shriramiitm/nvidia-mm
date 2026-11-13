| Specification | Value |
|---|---|
| GPU | Tesla T4 |
| Architecture / compute capability | Turing, sm_75 |
| Streaming multiprocessors (SMs) | 40 |
| FP32 cores per SM / total | 64 / 2560 |
| Max SM clock (boost) | 1590 MHz |
| **FP32 peak** (SMs × cores × 2 × clock) | **8.14 TFLOP/s** |
| Global memory | 14.6 GiB |
| Memory bus / max memory clock | 256-bit / 5001 MHz |
| **DRAM bandwidth** (clock × 2 × bus / 8) | **320 GB/s** |
| **Ridge point** (peak / bandwidth) | **25.4 FLOP/byte** |
| L2 cache | 4.0 MiB |
| Shared memory per SM / per block (default, opt-in) | 64 KiB / 48 KiB, 64 KiB |
| Registers per SM / per block | 65536 / 65536 (32-bit) |
| Max threads per SM / per block | 1024 / 1024 |
| Max resident blocks per SM | 16 |
| Warp size | 32 |
| Power limit | 70.00 W |
| Driver / CUDA runtime | 580.82.07 / 13.0 |

Max resident warps per SM = 1024 / 32 = 32. Occupancy = active warps / 32.
A kernel below 25.4 FLOP/byte (DRAM traffic) can be at most memory-bound; above it, compute-bound.
