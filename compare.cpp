// compare.cpp -- compare a candidate output against a reference output.
//
//   ./compare INPUT.bin REF.bin CAND.bin
//
// INPUT.bin is only used to read M, K, N (needed for the tolerance and sizes).
// Tolerance per element:  |cand - ref| <= 1e-4*K + 1e-4*|ref|
// Exit status: 0 = pass, 1 = mismatch / NaN / Inf, 2 = usage or I/O error.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

static bool read_floats(const char* path, size_t n, std::vector<float>& v) {
    FILE* f = std::fopen(path, "rb");
    if (!f) {
        std::fprintf(stderr, "error: cannot open '%s'\n", path);
        return false;
    }
    v.resize(n);
    size_t got = std::fread(v.data(), sizeof(float), n, f);
    bool extra = std::fgetc(f) != EOF;
    std::fclose(f);
    if (got != n || extra) {
        std::fprintf(stderr, "error: '%s' must contain exactly %zu float32 values\n", path, n);
        return false;
    }
    return true;
}

int main(int argc, char** argv) {
    if (argc != 4) {
        std::fprintf(stderr, "usage: %s INPUT.bin REF.bin CAND.bin\n", argv[0]);
        return 2;
    }
    FILE* f = std::fopen(argv[1], "rb");
    int32_t h[3];
    if (!f || std::fread(h, sizeof(int32_t), 3, f) != 3) {
        std::fprintf(stderr, "error: cannot read header of '%s'\n", argv[1]);
        return 2;
    }
    std::fclose(f);
    const int K = h[1];
    const size_t n = (size_t)h[0] * h[2];

    std::vector<float> ref, cand;
    if (!read_floats(argv[2], n, ref) || !read_floats(argv[3], n, cand)) return 2;

    double maxAbs = 0.0, maxRel = 0.0;
    size_t violations = 0, nonfinite = 0;
    for (size_t i = 0; i < n; i++) {
        double r = ref[i], c = cand[i];
        if (!std::isfinite(r) || !std::isfinite(c)) {
            nonfinite++;
            continue;
        }
        double err = std::fabs(c - r);
        double rel = err / std::max(std::fabs(r), 1e-30);
        maxAbs = std::max(maxAbs, err);
        maxRel = std::max(maxRel, rel);
        if (err > 1e-4 * K + 1e-4 * std::fabs(r)) violations++;
    }
    bool pass = (violations == 0 && nonfinite == 0);
    std::printf("max_abs_err=%.3e max_rel_err=%.3e violations=%zu nonfinite=%zu %s\n", maxAbs, maxRel,
                violations, nonfinite, pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
