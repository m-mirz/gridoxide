// cuDSS uniform-batch and UBATCH_MASK probe, phase 0 of plans/GPU_BATCH_PLAN.md.
//
// Answers, on real Newton-Raphson Jacobians written by
// `examples/gpu_probe_dump.rs`, the questions the GPU batch design depends on:
//
//   A. Does the uniform batch (one plain CSR matrix over a B*nnz value buffer,
//      CUDSS_CONFIG_UBATCH_SIZE = B, one analysis) solve every system to the
//      CPU's answer?
//   B. Does CUDSS_DATA_UBATCH_MASK do what the documentation says: can the
//      mask change between calls, does REFACTORIZATION honour it, are an
//      unselected system's factors and solution left untouched?
//   C. Does a masked refactorization and solve cost less in proportion to the
//      number of systems selected? (The whole adaptive design rests on this.)
//   D. What does one step's mask update cost: cudssDataSet plus a B-int
//      device-to-host copy plus a stream synchronize?
//
// Plain C++ against the cuDSS and CUDA runtime C APIs, no kernels, so it
// builds with g++ alone. Every cuDSS status is reported, never assumed; a
// wrong number exits non-zero. Usage:
//
//   probe <dump_dir> <B> [reps]
//
// B may exceed the dump's own batch: system b uses dumped system b % B_dump.
// Set CUDSS_THREADING_LIB to cuDSS's libcudss_mtlayer_gomp.so to enable its
// host threading (the old gpu-nvidia branch measured ~3.2x on analysis).

#include <cuda_runtime.h>
#include <cudss.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

namespace {

int g_failures = 0;

#define CUDA_OK(call)                                                                   \
    do {                                                                                \
        cudaError_t e_ = (call);                                                        \
        if (e_ != cudaSuccess) {                                                        \
            std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n", cudaGetErrorString(e_), \
                         __FILE__, __LINE__, #call);                                    \
            std::exit(2);                                                               \
        }                                                                               \
    } while (0)

// A cuDSS status, reported by name. `must` aborts on failure; otherwise the
// status is returned so a test can record an unsupported feature as a result.
const char* status_name(cudssStatus_t s) {
    switch (s) {
        case CUDSS_STATUS_SUCCESS: return "SUCCESS";
        case CUDSS_STATUS_NOT_INITIALIZED: return "NOT_INITIALIZED";
        case CUDSS_STATUS_ALLOC_FAILED: return "ALLOC_FAILED";
        case CUDSS_STATUS_INVALID_VALUE: return "INVALID_VALUE";
        case CUDSS_STATUS_NOT_SUPPORTED: return "NOT_SUPPORTED";
        case CUDSS_STATUS_EXECUTION_FAILED: return "EXECUTION_FAILED";
        case CUDSS_STATUS_INTERNAL_ERROR: return "INTERNAL_ERROR";
        case CUDSS_STATUS_IR_FAILED: return "IR_FAILED";
        default: return "UNKNOWN";
    }
}

cudssStatus_t report(cudssStatus_t s, const char* what) {
    if (s != CUDSS_STATUS_SUCCESS) std::printf("  cuDSS %s -> %s\n", what, status_name(s));
    return s;
}

void must(cudssStatus_t s, const char* what) {
    if (report(s, what) != CUDSS_STATUS_SUCCESS) std::exit(3);
}

template <typename T>
std::vector<T> read_file(const std::string& path, size_t count) {
    std::vector<T> v(count);
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::fprintf(stderr, "cannot open %s\n", path.c_str());
        std::exit(2);
    }
    f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(count * sizeof(T)));
    if (static_cast<size_t>(f.gcount()) != count * sizeof(T)) {
        std::fprintf(stderr, "%s is shorter than expected\n", path.c_str());
        std::exit(2);
    }
    return v;
}

// The dump's per-system arrays, tiled to `batch` systems.
std::vector<double> tiled(const std::vector<double>& src, size_t per_system, size_t dumped,
                          size_t batch) {
    std::vector<double> out(per_system * batch);
    for (size_t b = 0; b < batch; ++b)
        std::copy_n(src.begin() + static_cast<long>((b % dumped) * per_system), per_system,
                    out.begin() + static_cast<long>(b * per_system));
    return out;
}

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

struct Timer {
    cudaEvent_t start, stop;
    Timer() {
        CUDA_OK(cudaEventCreate(&start));
        CUDA_OK(cudaEventCreate(&stop));
    }
    ~Timer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    void begin(cudaStream_t s) { CUDA_OK(cudaEventRecord(start, s)); }
    double end_ms(cudaStream_t s) {
        CUDA_OK(cudaEventRecord(stop, s));
        CUDA_OK(cudaEventSynchronize(stop));
        float ms = 0;
        CUDA_OK(cudaEventElapsedTime(&ms, start, stop));
        return ms;
    }
};

// One cuDSS solver over the batch's device buffers: handle, config, data and
// the three matrix descriptors, with UBATCH_SIZE = batch.
struct Solver {
    cudssHandle_t handle = nullptr;
    cudssConfig_t config = nullptr;
    cudssData_t data = nullptr;
    cudssMatrix_t A = nullptr, x = nullptr, b = nullptr;

    Solver(cudaStream_t stream, int n, int nnz, int batch, int* d_rp, int* d_ci, double* d_vals,
           double* d_b, double* d_x, const char* threading_lib) {
        must(cudssCreate(&handle), "cudssCreate");
        must(cudssSetStream(handle, stream), "cudssSetStream");
        if (threading_lib) must(cudssSetThreadingLayer(handle, threading_lib), "cudssSetThreadingLayer");
        must(cudssConfigCreate(&config), "cudssConfigCreate");
        must(cudssDataCreate(handle, &data), "cudssDataCreate");
        must(cudssMatrixCreateCsr(&A, n, n, nnz, d_rp, nullptr, d_ci, d_vals, CUDSS_R_32I,
                                  CUDSS_R_32I, CUDSS_R_64F, CUDSS_MTYPE_GENERAL, CUDSS_MVIEW_FULL,
                                  CUDSS_BASE_ZERO),
             "cudssMatrixCreateCsr");
        must(cudssMatrixCreateDn(&b, n, 1, n, d_b, CUDSS_R_64F, CUDSS_LAYOUT_COL_MAJOR),
             "cudssMatrixCreateDn(b)");
        must(cudssMatrixCreateDn(&x, n, 1, n, d_x, CUDSS_R_64F, CUDSS_LAYOUT_COL_MAJOR),
             "cudssMatrixCreateDn(x)");
        must(cudssConfigSet(config, CUDSS_CONFIG_UBATCH_SIZE, &batch, sizeof(batch)),
             "cudssConfigSet(UBATCH_SIZE)");
    }
    ~Solver() {
        cudssMatrixDestroy(A);
        cudssMatrixDestroy(b);
        cudssMatrixDestroy(x);
        cudssDataDestroy(handle, data);
        cudssConfigDestroy(config);
        cudssDestroy(handle);
    }
    cudssStatus_t run(int phase) { return cudssExecute(handle, phase, config, data, A, x, b); }
    cudssStatus_t set_mask(const std::vector<int>& mask) {
        return cudssDataSet(handle, data, CUDSS_DATA_UBATCH_MASK, mask.data(),
                            mask.size() * sizeof(int));
    }
};

// Largest |x - ref| over the selected systems, NaN if any is NaN, and whether
// every unselected system still holds `sentinel`. Prints how many selected
// systems are off by more than 1e-8 (NaN included) and the first of them.
// (`std::max` would drop a NaN: the first run of this probe reported 0.8.0's
// NaN solutions as a pass that way.)
std::pair<double, bool> compare(const std::vector<double>& x, const std::vector<double>& ref,
                                const std::vector<int>& selected, int n, double sentinel) {
    double err = 0;
    bool untouched = true;
    size_t bad = 0, first_bad = 0;
    for (size_t s = 0; s < selected.size(); ++s) {
        bool off = false;
        for (int i = 0; i < n; ++i) {
            size_t k = s * static_cast<size_t>(n) + static_cast<size_t>(i);
            if (selected[s]) {
                double d = std::fabs(x[k] - ref[k]);
                if (std::isnan(d) || d > err) err = std::isnan(err) ? err : d;
                if (!(d <= 1e-8)) off = true;
            } else if (x[k] != sentinel) {
                untouched = false;
            }
        }
        if (off && bad++ == 0) first_bad = s;
    }
    if (bad) std::printf("  %zu systems off by more than 1e-8 or NaN, the first %zu\n", bad, first_bad);
    return {err, untouched};
}

void verdict(const char* what, bool ok) {
    std::printf("RESULT %-58s %s\n", what, ok ? "PASS" : "FAIL");
    if (!ok) ++g_failures;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: probe <dump_dir> <B> [reps]\n");
        return 2;
    }
    const std::string dir = argv[1];
    const int batch = std::atoi(argv[2]);
    const int reps = argc > 3 ? std::atoi(argv[3]) : 7;
    const char* threading_lib = std::getenv("CUDSS_THREADING_LIB");

    int n = 0, nnz = 0, dumped = 0;
    {
        std::ifstream meta(dir + "/meta.txt");
        meta >> n >> nnz >> dumped;
    }
    int major = 0, minor = 0, patch = 0;
    cudssGetProperty(MAJOR_VERSION, &major);
    cudssGetProperty(MINOR_VERSION, &minor);
    cudssGetProperty(PATCH_LEVEL, &patch);
    std::printf("cuDSS %d.%d.%d, n=%d nnz=%d B=%d (dump B=%d), threading layer: %s\n", major, minor,
                patch, n, nnz, batch, dumped, threading_lib ? threading_lib : "off");

    const auto row_ptr = read_file<int>(dir + "/row_ptr.i32", static_cast<size_t>(n) + 1);
    const auto col_idx = read_file<int>(dir + "/col_idx.i32", static_cast<size_t>(nnz));
    const size_t N = static_cast<size_t>(n), NNZ = static_cast<size_t>(nnz),
                 D = static_cast<size_t>(dumped), Bs = static_cast<size_t>(batch);
    const auto vals0 = tiled(read_file<double>(dir + "/values0.f64", D * NNZ), NNZ, D, Bs);
    const auto vals1 = tiled(read_file<double>(dir + "/values1.f64", D * NNZ), NNZ, D, Bs);
    const auto rhs0 = tiled(read_file<double>(dir + "/rhs0.f64", D * N), N, D, Bs);
    const auto rhs1 = tiled(read_file<double>(dir + "/rhs1.f64", D * N), N, D, Bs);
    const auto x0 = tiled(read_file<double>(dir + "/x0.f64", D * N), N, D, Bs);
    const auto x1 = tiled(read_file<double>(dir + "/x1.f64", D * N), N, D, Bs);

    cudaStream_t stream;
    CUDA_OK(cudaStreamCreate(&stream));
    int *d_rp, *d_ci, *d_flags;
    double *d_vals, *d_b, *d_x;
    CUDA_OK(cudaMalloc(&d_rp, (N + 1) * sizeof(int)));
    CUDA_OK(cudaMalloc(&d_ci, NNZ * sizeof(int)));
    CUDA_OK(cudaMalloc(&d_vals, Bs * NNZ * sizeof(double)));
    CUDA_OK(cudaMalloc(&d_b, Bs * N * sizeof(double)));
    CUDA_OK(cudaMalloc(&d_x, Bs * N * sizeof(double)));
    CUDA_OK(cudaMalloc(&d_flags, Bs * sizeof(int)));
    CUDA_OK(cudaMemcpy(d_rp, row_ptr.data(), (N + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_ci, col_idx.data(), NNZ * sizeof(int), cudaMemcpyHostToDevice));
    auto upload = [&](double* dst, const std::vector<double>& src) {
        CUDA_OK(cudaMemcpy(dst, src.data(), src.size() * sizeof(double), cudaMemcpyHostToDevice));
    };
    auto download = [&](std::vector<double>& dst, const double* src) {
        CUDA_OK(cudaMemcpy(dst.data(), src, dst.size() * sizeof(double), cudaMemcpyDeviceToHost));
    };
    std::vector<double> x(Bs * N);
    const std::vector<int> all(Bs, 1);
    const double sentinel = 12345.678;

    // ---- A: the uniform batch, unmasked ---------------------------------------
    {
        std::printf("\n[A] uniform batch, no mask\n");
        size_t free0 = 0, total = 0, free1 = 0;
        CUDA_OK(cudaMemGetInfo(&free0, &total));
        upload(d_vals, vals0);
        upload(d_b, rhs0);
        Solver s(stream, n, nnz, batch, d_rp, d_ci, d_vals, d_b, d_x, threading_lib);
        Timer t;
        auto wall = std::chrono::steady_clock::now();
        t.begin(stream);
        must(s.run(CUDSS_PHASE_ANALYSIS), "ANALYSIS");
        double analysis_ms = t.end_ms(stream);
        double analysis_wall =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - wall).count();
        t.begin(stream);
        must(s.run(CUDSS_PHASE_FACTORIZATION), "FACTORIZATION");
        double factor_ms = t.end_ms(stream);
        t.begin(stream);
        must(s.run(CUDSS_PHASE_SOLVE), "SOLVE");
        double solve_ms = t.end_ms(stream);
        CUDA_OK(cudaMemGetInfo(&free1, &total));
        download(x, d_x);
        auto [err, _] = compare(x, x0, all, n, sentinel);
        std::printf("  analysis %.2f ms (wall %.2f), factorization %.2f ms, solve %.2f ms\n",
                    analysis_ms, analysis_wall, factor_ms, solve_ms);
        std::printf("  device memory used by cuDSS: %.1f MB of %.1f MB\n",
                    (static_cast<double>(free0) - static_cast<double>(free1)) / 1e6,
                    static_cast<double>(total) / 1e6);
        std::printf("  max |x - x_cpu| = %.3e\n", err);
        verdict("A uniform batch matches the CPU solutions (1e-8)", err < 1e-8);

        // Refactorization on step 1's values, every system.
        upload(d_vals, vals1);
        upload(d_b, rhs1);
        must(cudssMatrixSetValues(s.A, d_vals), "cudssMatrixSetValues");
        must(s.run(CUDSS_PHASE_REFACTORIZATION), "REFACTORIZATION");
        must(s.run(CUDSS_PHASE_SOLVE), "SOLVE");
        CUDA_OK(cudaStreamSynchronize(stream));
        download(x, d_x);
        err = compare(x, x1, all, n, sentinel).first;
        std::printf("  after REFACTORIZATION on step 1: max |x - x_cpu| = %.3e\n", err);
        verdict("A refactorization on new values matches the CPU (1e-8)", err < 1e-8);
    }

    // ---- B: mask semantics ------------------------------------------------------
    {
        std::printf("\n[B] UBATCH_MASK semantics\n");
        upload(d_vals, vals0);
        upload(d_b, rhs0);
        Solver s(stream, n, nnz, batch, d_rp, d_ci, d_vals, d_b, d_x, threading_lib);
        must(s.set_mask(all), "DataSet(UBATCH_MASK, all ones) before ANALYSIS");
        must(s.run(CUDSS_PHASE_ANALYSIS), "ANALYSIS");
        must(s.run(CUDSS_PHASE_FACTORIZATION), "FACTORIZATION");
        must(s.run(CUDSS_PHASE_SOLVE), "SOLVE");
        CUDA_OK(cudaStreamSynchronize(stream));
        download(x, d_x);
        double err = compare(x, x0, all, n, sentinel).first;
        verdict("B all-ones mask: every system matches step 0 (1e-8)", err < 1e-8);

        // Even systems move to step 1's values; odd ones keep step 0's factors.
        std::vector<int> even(Bs), odd(Bs);
        for (size_t i = 0; i < Bs; ++i) even[i] = (i % 2 == 0), odd[i] = (i % 2 == 1);
        std::vector<double> vals_mixed = vals0, rhs_mixed = rhs0;
        for (size_t i = 0; i < Bs; i += 2) {
            std::copy_n(vals1.begin() + static_cast<long>(i * NNZ), NNZ,
                        vals_mixed.begin() + static_cast<long>(i * NNZ));
            std::copy_n(rhs1.begin() + static_cast<long>(i * N), N,
                        rhs_mixed.begin() + static_cast<long>(i * N));
        }
        // Odd systems' values are overwritten with garbage: a masked-out
        // system must not be refactored from them.
        for (size_t i = 1; i < Bs; i += 2)
            std::fill_n(vals_mixed.begin() + static_cast<long>(i * NNZ), NNZ, 1e300);
        upload(d_vals, vals_mixed);
        upload(d_b, rhs_mixed);
        std::vector<double> sentinel_x(Bs * N, sentinel);
        upload(d_x, sentinel_x);
        must(cudssMatrixSetValues(s.A, d_vals), "cudssMatrixSetValues");

        bool mask_changed = report(s.set_mask(even), "DataSet(UBATCH_MASK, even) after FACTORIZATION") ==
                            CUDSS_STATUS_SUCCESS;
        verdict("B the mask can change after a factorization", mask_changed);
        if (mask_changed) {
            cudssStatus_t rf = report(s.run(CUDSS_PHASE_REFACTORIZATION), "masked REFACTORIZATION");
            verdict("B masked REFACTORIZATION is accepted", rf == CUDSS_STATUS_SUCCESS);
            if (rf != CUDSS_STATUS_SUCCESS) {
                rf = report(s.run(CUDSS_PHASE_FACTORIZATION), "masked FACTORIZATION (fallback)");
                verdict("B masked FACTORIZATION is accepted", rf == CUDSS_STATUS_SUCCESS);
            }
            must(s.run(CUDSS_PHASE_SOLVE), "masked SOLVE (even)");
            CUDA_OK(cudaStreamSynchronize(stream));
            download(x, d_x);
            auto [err_even, odd_untouched] = compare(x, x1, even, n, sentinel);
            std::printf("  even systems vs step 1: %.3e; odd solutions untouched: %s\n", err_even,
                        odd_untouched ? "yes" : "no");
            verdict("B masked systems refactor and solve correctly (1e-8)", err_even < 1e-8);
            verdict("B unselected systems' solutions are left untouched", odd_untouched);

            // The odd systems were never refactored: solving them with step
            // 0's right-hand side must give step 0's answer, from the factors
            // of the first, unmasked factorization (and not from the 1e300s).
            std::vector<double> rhs_odd = rhs0;
            upload(d_b, rhs_odd);
            upload(d_x, sentinel_x);
            if (report(s.set_mask(odd), "DataSet(UBATCH_MASK, odd)") == CUDSS_STATUS_SUCCESS) {
                must(s.run(CUDSS_PHASE_SOLVE), "masked SOLVE (odd)");
                CUDA_OK(cudaStreamSynchronize(stream));
                download(x, d_x);
                auto [err_odd, even_untouched] = compare(x, x0, odd, n, sentinel);
                std::printf("  odd systems vs step 0: %.3e; even solutions untouched: %s\n", err_odd,
                            even_untouched ? "yes" : "no");
                verdict("B unselected systems keep their earlier factors (1e-8)", err_odd < 1e-8);
            }
        }
    }

    // ---- C: does masked work scale with the selection? -----------------------------
    {
        std::printf("\n[C] masked REFACTORIZATION + SOLVE time vs. systems selected (%d reps)\n", reps);
        upload(d_vals, vals1);
        upload(d_b, rhs1);
        Solver s(stream, n, nnz, batch, d_rp, d_ci, d_vals, d_b, d_x, threading_lib);
        must(s.set_mask(all), "DataSet(UBATCH_MASK, all ones)");
        must(s.run(CUDSS_PHASE_ANALYSIS), "ANALYSIS");
        must(s.run(CUDSS_PHASE_FACTORIZATION), "FACTORIZATION");
        CUDA_OK(cudaStreamSynchronize(stream));
        Timer t;
        double full_ms = 0;
        std::printf("  %-10s %-10s %12s %12s %14s\n", "selected", "layout", "refactor ms", "solve ms",
                    "vs. full, per sys");
        for (double frac : {1.0, 0.5, 0.25, 0.1, 0.01, 0.0}) {
            int k = std::max(1, static_cast<int>(std::lround(frac * batch)));
            for (int strided = 0; strided < 2; ++strided) {
                if (frac == 1.0 && strided) continue;
                std::vector<int> mask(Bs, 0);
                if (strided) {
                    for (int i = 0; i < k; ++i) mask[static_cast<size_t>(i) * Bs / static_cast<size_t>(k)] = 1;
                } else {
                    std::fill_n(mask.begin(), k, 1);
                }
                if (report(s.set_mask(mask), "DataSet(UBATCH_MASK)") != CUDSS_STATUS_SUCCESS) continue;
                std::vector<double> rf_ms, sv_ms;
                for (int r = 0; r < reps; ++r) {
                    t.begin(stream);
                    if (report(s.run(CUDSS_PHASE_REFACTORIZATION), "REFACTORIZATION") != CUDSS_STATUS_SUCCESS) break;
                    rf_ms.push_back(t.end_ms(stream));
                    t.begin(stream);
                    if (report(s.run(CUDSS_PHASE_SOLVE), "SOLVE") != CUDSS_STATUS_SUCCESS) break;
                    sv_ms.push_back(t.end_ms(stream));
                }
                if (rf_ms.empty() || sv_ms.empty()) continue;
                double total = median(rf_ms) + median(sv_ms);
                if (frac == 1.0) full_ms = total;
                // 1.0 means "costs exactly its share of the full batch".
                double scaling = full_ms > 0 ? (total / k) / (full_ms / batch) : 0;
                std::printf("  %-10d %-10s %12.3f %12.3f %14.2f\n", k, strided ? "strided" : "contiguous",
                            median(rf_ms), median(sv_ms), scaling);
                std::printf("RESULT C selected=%d layout=%s refactor_ms=%.4f solve_ms=%.4f per_system_vs_full=%.3f\n",
                            k, strided ? "strided" : "contiguous", median(rf_ms), median(sv_ms), scaling);
            }
        }
        std::printf("  (per_system_vs_full near 1: masked work scales with the selection;\n"
                    "   near B/selected: the mask saves nothing)\n");
    }

    // ---- D: the per-step cost of driving the mask from the device ---------------------
    {
        std::printf("\n[D] per-step mask update: D2H of %d flags + stream sync + cudssDataSet\n", batch);
        upload(d_vals, vals1);
        Solver s(stream, n, nnz, batch, d_rp, d_ci, d_vals, d_b, d_x, threading_lib);
        must(s.set_mask(all), "DataSet(UBATCH_MASK, all ones)");
        must(s.run(CUDSS_PHASE_ANALYSIS), "ANALYSIS");
        std::vector<int> flags(Bs, 1);
        CUDA_OK(cudaMemcpy(d_flags, flags.data(), Bs * sizeof(int), cudaMemcpyHostToDevice));
        std::vector<double> step_us;
        for (int r = 0; r < std::max(reps, 50); ++r) {
            auto t0 = std::chrono::steady_clock::now();
            CUDA_OK(cudaMemcpyAsync(flags.data(), d_flags, Bs * sizeof(int), cudaMemcpyDeviceToHost, stream));
            CUDA_OK(cudaStreamSynchronize(stream));
            report(s.set_mask(flags), "DataSet(UBATCH_MASK)");
            step_us.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count());
        }
        std::printf("  median %.1f us per step (pageable host memory)\n", median(step_us));
        std::printf("RESULT D mask_update_us=%.2f\n", median(step_us));
    }

    cudaFree(d_rp);
    cudaFree(d_ci);
    cudaFree(d_vals);
    cudaFree(d_b);
    cudaFree(d_x);
    cudaFree(d_flags);
    cudaStreamDestroy(stream);
    std::printf("\n%s: %d failure(s)\n", g_failures ? "FAIL" : "PASS", g_failures);
    return g_failures ? 1 : 0;
}
