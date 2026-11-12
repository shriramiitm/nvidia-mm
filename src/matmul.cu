// matmul.cu -- single-threaded CPU (Week 1), naive CUDA (Week 1), shared-memory tiled CUDA (Week 2),
//              tiled CUDA with shared-memory padding (Week 3).
//
//   ./matmul --impl {cpu,naive,tiled,padded} --input IN.bin --output OUT.bin [--tile T]
//
// Extra options for benchmarking (prints ONE csv line to stdout):
//   --bench  --trials N (GPU, default 20)  --warmup W (GPU, default 3)
//            --cpu-trials C (default 3)    (--output is optional with --bench)
//
// Week 3 experiment (not one of the four required versions):
//   --impl tiled_ta / padded_ta   same kernel, but the A tile is stored TRANSPOSED in
//   shared memory. This creates a column-wise store, the access pattern padding is
//   designed to fix. See docs/week3.md.
//
// Optional extension (clearly separate fifth version, Week 3 second improvement):
//   --impl regtile   register tiling: 64x64 block tile, 4x4 outputs per thread
//                    (--tile is ignored). See docs/week3.md section 8.

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <climits>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <string>
#include <vector>

// ---------------------------------------------------------------- helpers

// Check every CUDA API call. On failure: clear message + nonzero exit status.
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err_ = (call);                                            \
        if (err_ != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error: %s (%s:%d)\n",                       \
                    cudaGetErrorString(err_), __FILE__, __LINE__);            \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

[[noreturn]] static void die(const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    fprintf(stderr, "error: ");
    vfprintf(stderr, fmt, ap);
    fprintf(stderr, "\n");
    va_end(ap);
    exit(EXIT_FAILURE);
}

static double now_ms() {  // monotonic host clock
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

static double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    return (n % 2) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

// ---------------------------------------------------------------- arguments

struct Args {
    std::string impl, input, output;
    int tile = 32;  // fastest on every benchmarked shape (docs/week2.md section 8)
    bool bench = false;
    int trials = 20, warmup = 3, cpu_trials = 3;
};

static void usage() {
    fprintf(stderr,
            "usage: ./matmul --impl {cpu,naive,tiled,padded} --input IN.bin "
            "--output OUT.bin [--tile 8|16|32]\n"
            "       [--bench] [--trials N] [--warmup W] [--cpu-trials C]\n"
            "       experiments: --impl tiled_ta | padded_ta (transposed A tile, see docs/week3.md)\n"
            "       extension:   --impl regtile (register tiling, 4x4 outputs per thread; --tile ignored)\n");
}

static int parse_int(const char* s, const char* name) {
    char* end = nullptr;
    long v = strtol(s, &end, 10);
    if (end == s || *end != '\0' || v <= 0 || v > INT_MAX)
        die("%s needs a positive integer, got '%s'", name, s);
    return (int)v;
}

static Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; i++) {
        std::string k = argv[i];
        auto need = [&]() -> const char* {
            if (i + 1 >= argc) die("missing value for %s", k.c_str());
            return argv[++i];
        };
        if (k == "--impl") a.impl = need();
        else if (k == "--input") a.input = need();
        else if (k == "--output") a.output = need();
        else if (k == "--tile") a.tile = parse_int(need(), "--tile");
        else if (k == "--trials") a.trials = parse_int(need(), "--trials");
        else if (k == "--warmup") a.warmup = parse_int(need(), "--warmup");
        else if (k == "--cpu-trials") a.cpu_trials = parse_int(need(), "--cpu-trials");
        else if (k == "--bench") a.bench = true;
        else if (k == "--help" || k == "-h") { usage(); exit(0); }
        else { usage(); die("unknown argument '%s'", k.c_str()); }
    }
    if (a.impl != "cpu" && a.impl != "naive" && a.impl != "tiled" && a.impl != "padded" &&
        a.impl != "tiled_ta" && a.impl != "padded_ta" && a.impl != "regtile") {
        usage();
        die("--impl must be one of cpu, naive, tiled, padded (or tiled_ta, padded_ta, regtile)");
    }
    if (a.input.empty()) { usage(); die("--input is required"); }
    if (a.output.empty() && !a.bench) { usage(); die("--output is required"); }
    if (a.tile != 8 && a.tile != 16 && a.tile != 32)
        die("unsupported --tile %d (supported: 8, 16, 32)", a.tile);
    return a;
}

// ---------------------------------------------------------------- file I/O

struct Problem {
    int M = 0, K = 0, N = 0;
    std::vector<float> A, B;
};

static Problem read_input(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) die("cannot open input file '%s'", path.c_str());
    int32_t h[3];
    if (fread(h, sizeof(int32_t), 3, f) != 3) die("input file '%s' has a truncated header", path.c_str());
    Problem p;
    p.M = h[0]; p.K = h[1]; p.N = h[2];
    if (p.M < 1 || p.K < 1 || p.N < 1) die("invalid dimensions M=%d K=%d N=%d in header", p.M, p.K, p.N);
    size_t nA = (size_t)p.M * p.K, nB = (size_t)p.K * p.N;
    p.A.resize(nA);
    p.B.resize(nB);
    if (fread(p.A.data(), sizeof(float), nA, f) != nA) die("input file is too short (matrix A)");
    if (fread(p.B.data(), sizeof(float), nB, f) != nB) die("input file is too short (matrix B)");
    if (fgetc(f) != EOF) die("input file has unexpected trailing data");
    fclose(f);
    return p;
}

static void write_output(const std::string& path, const float* C, size_t n) {
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) die("cannot open output file '%s'", path.c_str());
    if (fwrite(C, sizeof(float), n, f) != n) die("failed writing output file '%s'", path.c_str());
    if (fclose(f) != 0) die("failed closing output file '%s'", path.c_str());
}

//CPU Baseline
static void cpu_matmul(const float* A, const float* B, float* C, int M, int K, int N) {
    std::fill(C, C + (size_t)M * N, 0.0f);
    for (int i = 0; i < M; i++) {
        for (int k = 0; k < K; k++) {
            const float a = A[(size_t)i * K + k];
            const float* b = B + (size_t)k * N;
            float* c = C + (size_t)i * N;
            for (int j = 0; j < N; j++) c[j] += a * b[j];
        }
    }
}

// ---------------------------------------------------------------- GPU kernels

// ---- V1 (Week 1): naive ----------------------------------------------------
// One thread = one output element C[row][col], full K-length dot product
// straight from global memory. col comes from threadIdx.x, so the 32 threads
// of a warp read consecutive B[k][col..col+31] and write consecutive C
// elements -> coalesced. A[row][k] is the same address for the whole warp
// (a broadcast).
constexpr int NAIVE_BLOCK = 16;

__global__ void matmul_naive(const float* A, const float* B, float* C, int M, int K, int N) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < M && col < N) {  // bounds check for arbitrary M, N
        float acc = 0.0f;
        for (int k = 0; k < K; k++)
            acc += A[(size_t)row * K + k] * B[(size_t)k * N + col];
        C[(size_t)row * N + col] = acc;
    }
}

// ---- V2 (Week 2) + V3 (Week 3): shared-memory tiled, optionally padded ------
// A block of TILE x TILE threads owns one TILE x TILE tile of C. It walks along
// K in steps of TILE. In each step the block cooperatively loads ONE tile of A
// and ONE tile of B into shared memory (each thread loads one element of each),
// then every thread does TILE multiply-adds out of shared memory. Each element
// fetched from global memory is therefore used by TILE threads instead of 1.
//
//   global loads per output element:  naive 2*K   ->   tiled 2*K/TILE
//
// Edge handling: out-of-range tile elements are loaded as 0.0f, which adds
// nothing to the sum, so K, M, N need not be multiples of TILE. All threads
// (even ones outside C) must reach both __syncthreads(), so there is no early
// return; only the final store is guarded.
//
// PAD (Week 3): the shared arrays are declared [TILE][TILE + PAD]. The logical
// tile is still TILE x TILE and column TILE..TILE+PAD-1 is never read or
// written, so the result is identical. Only the physical row length (leading
// dimension LD) changes, and with it the bank of each element:
//     bank(As[r][c]) = (r * LD + c) % 32
// With LD = TILE + 1 (odd), walking DOWN a column moves one bank per row
// instead of staying in the same bank. tiled = PAD 0, padded = PAD 1, so
// global-memory tiling and arithmetic are exactly the same in both: padding is
// the only experimental variable.
//
// TRANSPOSE_A (Week 3 experiment only): store the A tile as As[tx][ty] and read
// it as As[k][ty]. The multiply-add is unchanged (As[k][ty] holds A[row][t*TILE+k]),
// but the shared-memory STORE now walks down a column: with LD = TILE = 32
// every thread of a warp hits the same bank (32-way conflict). This is the
// pattern padding is designed to fix; the normal layout does not have it.
// Predicted conflict counts for every variant: python3 scripts/bank_model.py
template <int TILE, int PAD, bool TRANSPOSE_A>
__global__ void matmul_tiled(const float* __restrict__ A, const float* __restrict__ B,
                             float* __restrict__ C, int M, int K, int N) {
    __shared__ float As[TILE][TILE + PAD];
    __shared__ float Bs[TILE][TILE + PAD];

    const int tx = threadIdx.x, ty = threadIdx.y;
    const int row = blockIdx.y * TILE + ty;
    const int col = blockIdx.x * TILE + tx;

    float acc = 0.0f;
    const int numTiles = (K + TILE - 1) / TILE;
    for (int t = 0; t < numTiles; t++) {
        const int aCol = t * TILE + tx;  // column of A this thread loads
        const int bRow = t * TILE + ty;  // row of B this thread loads
        // consecutive tx -> consecutive addresses in both loads (coalesced)
        const float a = (row < M && aCol < K) ? A[(size_t)row * K + aCol] : 0.0f;
        if (TRANSPOSE_A) As[tx][ty] = a;  // column-wise store (experiment)
        else             As[ty][tx] = a;  // row-wise store
        Bs[ty][tx] = (bRow < K && col < N) ? B[(size_t)bRow * N + col] : 0.0f;
        __syncthreads();  // tile fully loaded before anyone reads it

#pragma unroll
        for (int k = 0; k < TILE; k++) {
            const float av = TRANSPOSE_A ? As[k][ty] : As[ty][k];  // both = A[row][t*TILE+k]
            acc += av * Bs[k][tx];
        }
        __syncthreads();  // everyone done reading before the tile is overwritten
    }
    if (row < M && col < N) C[(size_t)row * N + col] = acc;
}

// Tile sizes must fit the device: TILE*TILE threads per block and
// 2*TILE*(TILE+PAD) floats of shared memory.
static void check_tile_resources(int tile, int pad) {
    int dev = 0, maxThreads = 0, maxShared = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&maxThreads, cudaDevAttrMaxThreadsPerBlock, dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&maxShared, cudaDevAttrMaxSharedMemoryPerBlock, dev));
    size_t shared = 2 * (size_t)tile * (tile + pad) * sizeof(float);
    if (tile * tile > maxThreads)
        die("--tile %d needs %d threads per block but the device allows %d", tile, tile * tile, maxThreads);
    if (shared > (size_t)maxShared)
        die("--tile %d needs %zu bytes of shared memory but the device allows %d", tile, shared, maxShared);
}

// ---- V5 (Week 3, second profiler-guided improvement): register tiling ---------
// Profiling showed the tiled kernels are limited by the shared-memory pipe:
// kernel time ~ shared-memory wavefronts, and each multiply-add needs 1.25
// shared loads (docs/week3.md section 6c). Register tiling raises the work done
// per shared load:
//
//   * a 256-thread block (16 x 16) owns a 64 x 64 tile of C (RT_BM x RT_BN);
//   * each thread owns a 4 x 4 sub-block of C, kept in 16 registers;
//   * per k, a thread reads 4 values of A and 4 values of B from shared memory
//     (one 128-bit load each) and does 4 x 4 = 16 multiply-adds with them.
//
//   shared loads per FMA:  tiled 1.25 (scalar)  ->  regtile 2 x 128-bit / 16 = 0.125
//
// Shared-memory layout (the Week 3 padding lesson, applied):
//   As is stored TRANSPOSED, As[k][m], so a thread's 4 A values (rows m..m+3)
//   are contiguous and can be read with one float4. The transposed store walks
//   down a column, which with a 64-float row would put 8 threads in one bank.
//   Padding the row by RT_PAD = 4 floats (not 1) removes that conflict AND keeps
//   every row 16-byte aligned, so the float4 reads stay legal. Padding by 1
//   would break the alignment, as the padded kernel showed.
//
// Edge handling is the same idea as before: out-of-range elements are loaded as
// 0.0f and every output store is guarded, so any M, N, K works.
constexpr int RT_BM = 64, RT_BN = 64, RT_BK = 8;  // block tile of C, K step
constexpr int RT_TM = 4, RT_TN = 4;                // per-thread sub-block of C
constexpr int RT_PAD = 4;                          // keeps As rows 16-byte aligned
constexpr int RT_THREADS = (RT_BM / RT_TM) * (RT_BN / RT_TN);  // 256

__global__ void __launch_bounds__(RT_THREADS)
matmul_regtile(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
               int M, int K, int N) {
    __shared__ __align__(16) float As[RT_BK][RT_BM + RT_PAD];  // transposed: As[k][m]
    __shared__ __align__(16) float Bs[RT_BK][RT_BN];           // Bs[k][n]

    const int tx = threadIdx.x, ty = threadIdx.y;  // 16 x 16
    const int tid = ty * blockDim.x + tx;
    const int rowBase = blockIdx.y * RT_BM, colBase = blockIdx.x * RT_BN;

    float acc[RT_TM][RT_TN] = {};

    for (int k0 = 0; k0 < K; k0 += RT_BK) {
        // Cooperative loads: 64 x 8 A elements and 8 x 64 B elements, 2 each per thread.
#pragma unroll
        for (int i = 0; i < (RT_BM * RT_BK) / RT_THREADS; i++) {
            const int e = tid + i * RT_THREADS;
            const int r = e / RT_BK, c = e % RT_BK;    // 8 consecutive threads read one 32-byte row piece
            const int gr = rowBase + r, gc = k0 + c;
            As[c][r] = (gr < M && gc < K) ? A[(size_t)gr * K + gc] : 0.0f;
        }
#pragma unroll
        for (int i = 0; i < (RT_BK * RT_BN) / RT_THREADS; i++) {
            const int e = tid + i * RT_THREADS;
            const int r = e / RT_BN, c = e % RT_BN;    // a warp reads 32 consecutive floats of one row
            const int gr = k0 + r, gc = colBase + c;
            Bs[r][c] = (gr < K && gc < N) ? B[(size_t)gr * N + gc] : 0.0f;
        }
        __syncthreads();  // tiles fully loaded

#pragma unroll
        for (int k = 0; k < RT_BK; k++) {
            const float4 a4 = *reinterpret_cast<const float4*>(&As[k][ty * RT_TM]);  // A[row..row+3][k]
            const float4 b4 = *reinterpret_cast<const float4*>(&Bs[k][tx * RT_TN]);  // B[k][col..col+3]
            const float a[RT_TM] = {a4.x, a4.y, a4.z, a4.w};
            const float b[RT_TN] = {b4.x, b4.y, b4.z, b4.w};
#pragma unroll
            for (int i = 0; i < RT_TM; i++)
#pragma unroll
                for (int j = 0; j < RT_TN; j++) acc[i][j] += a[i] * b[j];
        }
        __syncthreads();  // everyone done before the next tiles overwrite these
    }

#pragma unroll
    for (int i = 0; i < RT_TM; i++) {
        const int row = rowBase + ty * RT_TM + i;
        if (row >= M) continue;
#pragma unroll
        for (int j = 0; j < RT_TN; j++) {
            const int col = colBase + tx * RT_TN + j;
            if (col < N) C[(size_t)row * N + col] = acc[i][j];
        }
    }
}

// The four tiled variants (tiled, padded, tiled_ta, padded_ta) use --tile.
static bool is_tiled(const std::string& impl) { return impl != "cpu" && impl != "naive" && impl != "regtile"; }
static int pad_of(const std::string& impl) { return (impl == "padded" || impl == "padded_ta") ? 1 : 0; }

template <int PAD, bool TA>
static void launch_tiled(int tile, const float* dA, const float* dB, float* dC, int M, int K, int N) {
    dim3 block(tile, tile);
    dim3 grid((N + tile - 1) / tile, (M + tile - 1) / tile);
    switch (tile) {
        case 8:  matmul_tiled<8, PAD, TA><<<grid, block>>>(dA, dB, dC, M, K, N); break;
        case 16: matmul_tiled<16, PAD, TA><<<grid, block>>>(dA, dB, dC, M, K, N); break;
        case 32: matmul_tiled<32, PAD, TA><<<grid, block>>>(dA, dB, dC, M, K, N); break;
        default: die("unsupported --tile %d", tile);
    }
}

static void launch_kernel(const std::string& impl, int tile, const float* dA, const float* dB, float* dC,
                          int M, int K, int N) {
    if (impl == "naive") {
        dim3 block(NAIVE_BLOCK, NAIVE_BLOCK);
        dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
        matmul_naive<<<grid, block>>>(dA, dB, dC, M, K, N);
    } else if (impl == "tiled") {
        launch_tiled<0, false>(tile, dA, dB, dC, M, K, N);
    } else if (impl == "padded") {
        launch_tiled<1, false>(tile, dA, dB, dC, M, K, N);
    } else if (impl == "tiled_ta") {
        launch_tiled<0, true>(tile, dA, dB, dC, M, K, N);
    } else if (impl == "padded_ta") {
        launch_tiled<1, true>(tile, dA, dB, dC, M, K, N);
    } else if (impl == "regtile") {
        dim3 block(RT_BN / RT_TN, RT_BM / RT_TM);  // 16 x 16
        dim3 grid((N + RT_BN - 1) / RT_BN, (M + RT_BM - 1) / RT_BM);
        matmul_regtile<<<grid, block>>>(dA, dB, dC, M, K, N);
    } else {
        die("implementation '%s' has no GPU kernel", impl.c_str());
    }
}

struct GpuRunTimes {
    double kernel_ms;  // CUDA events around the kernel only
    double e2e_ms;     // host clock: malloc + H2D + kernel + D2H + free
};

// One complete GPU run: allocate, copy in, launch, copy out, free.
static GpuRunTimes gpu_run(const std::string& impl, int tile, const float* hA, const float* hB, float* hC,
                           int M, int K, int N) {
    const size_t bytesA = (size_t)M * K * sizeof(float);
    const size_t bytesB = (size_t)K * N * sizeof(float);
    const size_t bytesC = (size_t)M * N * sizeof(float);

    const int blockY = (impl == "regtile") ? RT_BM : is_tiled(impl) ? tile : NAIVE_BLOCK;
    if ((M + blockY - 1) / blockY > 65535) die("M=%d is too large for this launch configuration", M);

    double t0 = now_ms();

    size_t freeB = 0, totalB = 0;
    CUDA_CHECK(cudaMemGetInfo(&freeB, &totalB));
    if (bytesA + bytesB + bytesC > freeB)
        die("problem needs %.1f MiB of device memory but only %.1f MiB are free",
            (bytesA + bytesB + bytesC) / 1048576.0, freeB / 1048576.0);

    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    CUDA_CHECK(cudaMalloc(&dA, bytesA));
    CUDA_CHECK(cudaMalloc(&dB, bytesB));
    CUDA_CHECK(cudaMalloc(&dC, bytesC));
    CUDA_CHECK(cudaMemcpy(dA, hA, bytesA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB, bytesB, cudaMemcpyHostToDevice));

    cudaEvent_t evStart, evStop;
    CUDA_CHECK(cudaEventCreate(&evStart));
    CUDA_CHECK(cudaEventCreate(&evStop));
    CUDA_CHECK(cudaEventRecord(evStart));
    launch_kernel(impl, tile, dA, dB, dC, M, K, N);
    CUDA_CHECK(cudaGetLastError());          // launch errors (bad config, ...)
    CUDA_CHECK(cudaEventRecord(evStop));
    CUDA_CHECK(cudaEventSynchronize(evStop));  // surfaces errors raised during execution
    float kernel_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, evStart, evStop));

    CUDA_CHECK(cudaMemcpy(hC, dC, bytesC, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaEventDestroy(evStart));
    CUDA_CHECK(cudaEventDestroy(evStop));
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));

    double t1 = now_ms();
    return {(double)kernel_ms, t1 - t0};
}

// ---------------------------------------------------------------- main

int main(int argc, char** argv) {
    Args a = parse_args(argc, argv);

    Problem p;
    std::vector<float> C;
    try {
        p = read_input(a.input);
        C.assign((size_t)p.M * p.N, 0.0f);
    } catch (const std::bad_alloc&) {
        die("not enough host memory for this problem size");
    }
    const int M = p.M, K = p.K, N = p.N;
    const double flops = 2.0 * M * N * K;

    double kernel_ms = 0, e2e_ms = 0;
    int trials_used = 1;

    if (a.impl == "cpu") {
        trials_used = a.bench ? a.cpu_trials : 1;
        std::vector<double> t;
        for (int i = 0; i < trials_used; i++) {
            double t0 = now_ms();
            cpu_matmul(p.A.data(), p.B.data(), C.data(), M, K, N);
            t.push_back(now_ms() - t0);
        }
        kernel_ms = e2e_ms = median(t);  // CPU: multiplication only, no transfers
    } else {  // naive, tiled, padded (and experiments)
        CUDA_CHECK(cudaFree(0));  // force context creation; clear error if no GPU
        if (is_tiled(a.impl)) check_tile_resources(a.tile, pad_of(a.impl));
        trials_used = a.bench ? a.trials : 1;
        if (a.bench)
            for (int i = 0; i < a.warmup; i++) gpu_run(a.impl, a.tile, p.A.data(), p.B.data(), C.data(), M, K, N);
        std::vector<double> tk, te;
        for (int i = 0; i < trials_used; i++) {
            GpuRunTimes r = gpu_run(a.impl, a.tile, p.A.data(), p.B.data(), C.data(), M, K, N);
            tk.push_back(r.kernel_ms);
            te.push_back(r.e2e_ms);
        }
        kernel_ms = median(tk);
        e2e_ms = median(te);
    }

    if (!a.output.empty()) write_output(a.output, C.data(), C.size());

    const double kernel_gflops = flops / (kernel_ms * 1e6);
    const double e2e_gflops = flops / (e2e_ms * 1e6);
    const std::string tileStr = is_tiled(a.impl) ? std::to_string(a.tile) : (a.impl == "regtile") ? "64x64" : "NA";
    if (a.bench) {
        // impl,M,K,N,tile,trials,kernel_ms,e2e_ms,kernel_gflops,e2e_gflops
        printf("%s,%d,%d,%d,%s,%d,%.4f,%.4f,%.2f,%.2f\n", a.impl.c_str(), M, K, N, tileStr.c_str(),
               trials_used, kernel_ms, e2e_ms, kernel_gflops, e2e_gflops);
    } else {
        printf("impl=%s tile=%s M=%d K=%d N=%d kernel_ms=%.4f e2e_ms=%.4f kernel_gflops=%.2f\n",
               a.impl.c_str(), tileStr.c_str(), M, K, N, kernel_ms, e2e_ms, kernel_gflops);
    }
    return 0;
}
