// devinfo.cu -- print the GPU's hardware limits as key=value lines.
//   ./devinfo
// Uses cudaDeviceGetAttribute for every field (stable across CUDA versions;
// some cudaDeviceProp fields were removed in CUDA 13). Clock speeds are read
// by scripts/hw_specs.py from nvidia-smi instead.

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call)                                                                   \
    do {                                                                                   \
        cudaError_t err_ = (call);                                                         \
        if (err_ != cudaSuccess) {                                                         \
            fprintf(stderr, "CUDA error: %s (%s:%d)\n", cudaGetErrorString(err_), __FILE__, \
                    __LINE__);                                                             \
            exit(EXIT_FAILURE);                                                            \
        }                                                                                  \
    } while (0)

int main() {
    int count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&count));
    if (count == 0) {
        fprintf(stderr, "no CUDA device found\n");
        return 1;
    }
    const int dev = 0;
    CUDA_CHECK(cudaSetDevice(dev));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    size_t freeB = 0, totalB = 0;
    CUDA_CHECK(cudaMemGetInfo(&freeB, &totalB));

    struct Attr {
        const char* key;
        cudaDeviceAttr attr;
    };
    const Attr attrs[] = {
        {"cc_major", cudaDevAttrComputeCapabilityMajor},
        {"cc_minor", cudaDevAttrComputeCapabilityMinor},
        {"sm_count", cudaDevAttrMultiProcessorCount},
        {"warp_size", cudaDevAttrWarpSize},
        {"max_threads_per_block", cudaDevAttrMaxThreadsPerBlock},
        {"max_threads_per_sm", cudaDevAttrMaxThreadsPerMultiProcessor},
        {"max_blocks_per_sm", cudaDevAttrMaxBlocksPerMultiprocessor},
        {"regs_per_block", cudaDevAttrMaxRegistersPerBlock},
        {"regs_per_sm", cudaDevAttrMaxRegistersPerMultiprocessor},
        {"smem_per_block_bytes", cudaDevAttrMaxSharedMemoryPerBlock},
        {"smem_per_block_optin_bytes", cudaDevAttrMaxSharedMemoryPerBlockOptin},
        {"smem_per_sm_bytes", cudaDevAttrMaxSharedMemoryPerMultiprocessor},
        {"l2_cache_bytes", cudaDevAttrL2CacheSize},
        {"memory_bus_width_bits", cudaDevAttrGlobalMemoryBusWidth},
        {"max_grid_x", cudaDevAttrMaxGridDimX},
        {"max_grid_y", cudaDevAttrMaxGridDimY},
        {"max_block_x", cudaDevAttrMaxBlockDimX},
        {"max_block_y", cudaDevAttrMaxBlockDimY},
    };
    printf("name=%s\n", prop.name);
    printf("global_memory_mib=%zu\n", totalB >> 20);
    for (const Attr& a : attrs) {
        int v = 0;
        if (cudaDeviceGetAttribute(&v, a.attr, dev) == cudaSuccess) printf("%s=%d\n", a.key, v);
        else cudaGetLastError();  // clear and skip an attribute this CUDA version lacks
    }
    int rt = 0, drv = 0;
    CUDA_CHECK(cudaRuntimeGetVersion(&rt));
    CUDA_CHECK(cudaDriverGetVersion(&drv));
    printf("cuda_runtime=%d.%d\n", rt / 1000, (rt % 1000) / 10);
    printf("cuda_driver_api=%d.%d\n", drv / 1000, (drv % 1000) / 10);
    return 0;
}
