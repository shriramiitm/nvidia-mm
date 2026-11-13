Kernel-only, median of 20, 4096×4096×4096. Bandwidth = (A + B + C bytes) / time (effective).

| Step | Kernel | Time (ms) | TFLOPS | Bandwidth (GB/s) | vs cuBLAS | Speedup over naive | Speedup over previous step |
|---|---|---|---|---|---|---|---|
| – | CPU, i-k-j | 2.585e+04 | 0.00532 | 0.00779 | 0.1% | 0.0× | – |
| 2 | Global-memory coalesced | 331.4 | 0.415 | 0.607 | 6.3% | 1.0× | – |
| 3 | Shared memory (T=32) | 168.4 | 0.816 | 1.2 | 12.4% | 2.0× | 1.97× |
| 4 | Shared + padding (T=32) | 216.6 | 0.634 | 0.929 | 9.6% | 1.5× | – |
| 7 | Vectorized (float4, 8×8) | 25.47 | 5.4 | 7.9 | 81.8% | 13.0× | 6.61× |
| – | cuBLAS (reference) | 20.85 | 6.59 | 9.66 | 100.0% | 15.9× | – |
