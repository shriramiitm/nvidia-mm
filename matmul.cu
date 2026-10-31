// matmul.cu -- Week 1: single-threaded CPU baseline + naive CUDA kernel.
//
//   ./matmul --impl {cpu,naive,tiled,padded} --input IN.bin --output OUT.bin [--tile T]
//
// Extra options for benchmarking (prints ONE csv line to stdout):
//   --bench  --trials N (GPU, default 20)  --warmup W (GPU, default 3)
//            --cpu-trials C (default 3)    (--output is optional with --bench)
//
// tiled / padded are placeholders until Weeks 2-3.

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
    int tile = 16;
    bool bench = false;
    int trials = 20, warmup = 3, cpu_trials = 3;
};

static void usage() {
    fprintf(stderr,
            "usage: ./matmul --impl {cpu,naive,tiled,padded} --input IN.bin "
            "--output OUT.bin [--tile 8|16|32]\n"
            "       [--bench] [--trials N] [--warmup W] [--cpu-trials C]\n");
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
    if (a.impl != "cpu" && a.impl != "naive" && a.impl != "tiled" && a.impl != "padded") {
        usage();
        die("--impl must be one of cpu, naive, tiled, padded");
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

// ---------------------------------------------------------------- CPU baseline

// Single thread, loop order i-k-j.
// For row-major data the inner j-loop walks B[k][*] and C[i][*] contiguously
// (stride 1) and A[i][k] is a loop-invariant scalar, so every cache line that
// is fetched gets fully used. i-j-k would walk down a column of B (stride N).
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

// ---------------------------------------------------------------- naive GPU

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

struct GpuRunTimes {
    double kernel_ms;  // CUDA events around the kernel only
    double e2e_ms;     // host clock: malloc + H2D + kernel + D2H + free
};

// One complete GPU run: allocate, copy in, launch, copy out, free.
static GpuRunTimes gpu_run_naive(const float* hA, const float* hB, float* hC, int M, int K, int N) {
    const size_t bytesA = (size_t)M * K * sizeof(float);
    const size_t bytesB = (size_t)K * N * sizeof(float);
    const size_t bytesC = (size_t)M * N * sizeof(float);

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

    dim3 block(NAIVE_BLOCK, NAIVE_BLOCK);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    if (grid.y > 65535) die("M=%d is too large for this launch configuration", M);

    cudaEvent_t evStart, evStop;
    CUDA_CHECK(cudaEventCreate(&evStart));
    CUDA_CHECK(cudaEventCreate(&evStop));
    CUDA_CHECK(cudaEventRecord(evStart));
    matmul_naive<<<grid, block>>>(dA, dB, dC, M, K, N);
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

    if (a.impl == "tiled" || a.impl == "padded") {
        fprintf(stderr, "error: implementation '%s' is not implemented yet (planned for Weeks 2-3)\n",
                a.impl.c_str());
        return 2;
    }

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
    } else {  // naive
        CUDA_CHECK(cudaFree(0));  // force context creation; clear error if no GPU
        trials_used = a.bench ? a.trials : 1;
        if (a.bench)
            for (int i = 0; i < a.warmup; i++) gpu_run_naive(p.A.data(), p.B.data(), C.data(), M, K, N);
        std::vector<double> tk, te;
        for (int i = 0; i < trials_used; i++) {
            GpuRunTimes r = gpu_run_naive(p.A.data(), p.B.data(), C.data(), M, K, N);
            tk.push_back(r.kernel_ms);
            te.push_back(r.e2e_ms);
        }
        kernel_ms = median(tk);
        e2e_ms = median(te);
    }

    if (!a.output.empty()) write_output(a.output, C.data(), C.size());

    const double kernel_gflops = flops / (kernel_ms * 1e6);
    const double e2e_gflops = flops / (e2e_ms * 1e6);
    if (a.bench) {
        // impl,M,K,N,tile,trials,kernel_ms,e2e_ms,kernel_gflops,e2e_gflops
        printf("%s,%d,%d,%d,NA,%d,%.4f,%.4f,%.2f,%.2f\n", a.impl.c_str(), M, K, N, trials_used,
               kernel_ms, e2e_ms, kernel_gflops, e2e_gflops);
    } else {
        printf("impl=%s M=%d K=%d N=%d kernel_ms=%.4f e2e_ms=%.4f kernel_gflops=%.2f\n", a.impl.c_str(),
               M, K, N, kernel_ms, e2e_ms, kernel_gflops);
    }
    return 0;
}
