Kernel-only, median of 20, 4096×4096×4096. Bandwidth = (A + B + C bytes) / time (effective).

| Step | Kernel | Time (ms) | TFLOPS | Bandwidth (GB/s) | vs cuBLAS | Speedup over naive | Speedup over previous step |
|---|---|---|---|---|---|---|---|
| 2 | CPU, i-k-j | 2.647e+04 | 0.00519 | 0.00761 | 0.1% | 0.043× | – |
| 3 | Naive GPU (uncoalesced) | 1143 | 0.12 | 0.176 | 1.8% | 1.0× | – |
| 4 | Global-memory coalesced | 329.9 | 0.417 | 0.61 | 6.2% | 3.5× | 3.46× |
| 5 | Shared memory (T=32) | 163.5 | 0.841 | 1.23 | 12.5% | 7.0× | 2.02× |
| 5b | Shared + padding (T=32) | 204.5 | 0.672 | 0.985 | 10.0% | 5.6× | – |
| 6 | 1D block tiling (8×1) | 73.15 | 1.88 | 2.75 | 27.9% | 15.6× | 2.24× |
| 7 | 2D block tiling (8×8) | 35.4 | 3.88 | 5.69 | 57.7% | 32.3× | 2.07× |
| 8 | Vectorized (float4, 8×8) | 24.56 | 5.6 | 8.2 | 83.2% | 46.5× | 1.44× |
| – | cuBLAS (reference) | 20.44 | 6.72 | 9.85 | 100.0% | 55.9× | – |
