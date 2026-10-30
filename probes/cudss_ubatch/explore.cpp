// Follow-up to probe.cpp, written after its first run on an A100: what
// UBATCH_MASK does to the systems it leaves out, and what a refactorization and
// solve cost against the batch size. Its findings are in
// README.md ("Results").
//
//   explore <dump> <B> mask
//       Numbered experiments, each printing how systems 0-3 compare with
//       the sentinel (777), zero, and the CPU's step-0 and step-1 solutions:
//       masked SOLVE without a refactorization (1), masked REFACTORIZATION
//       and FACTORIZATION with valid and garbage values for the unselected
//       systems (2-4), solves selecting systems the latest factorization
//       left out (5, 7, 8), refactoring such systems back (9), and the mask
//       read back (6).
//   explore <dump> <B> time <reorder 0-5> <factor_alg 0-2> <mt 0/1> [pivot_type] [solve_alg]
//       Analysis, factorization, and the median of 9 refactorizations and
//       solves of all B systems, with the given CUDSS_CONFIG_REORDERING_ALG
//       and FACTORIZATION_ALG (0: cuDSS's default), cuDSS's threading layer
//       on or off (CUDSS_THREADING_LIB), and optionally PIVOT_TYPE and
//       SOLVE_ALG. B = 1 runs without UBATCH_SIZE unless UB1 is set; CFG
//       ("param=value,...", cudssConfigParam_t numbers, int values) sets
//       further knobs.
//
// Exits non-zero on any CUDA or cuDSS error, printing the status.

#include <cuda_runtime.h>
#include <cudss.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#define CK(c) do { cudaError_t e = (c); if (e) { std::printf("CUDA %s line %d\n", cudaGetErrorString(e), __LINE__); std::exit(2);} } while (0)
#define DS(c) do { cudssStatus_t s = (c); if (s) { std::printf("cuDSS status %d at line %d: %s\n", (int)s, __LINE__, #c); std::exit(3);} } while (0)

template <typename T> std::vector<T> rd(const std::string& p, size_t n) {
    std::vector<T> v(n); std::ifstream f(p, std::ios::binary);
    f.read((char*)v.data(), (std::streamsize)(n * sizeof(T))); return v;
}
std::vector<double> tile(const std::vector<double>& s, size_t per, size_t d, size_t B) {
    std::vector<double> o(per * B);
    for (size_t b = 0; b < B; ++b) std::copy_n(s.begin() + (long)((b % d) * per), per, o.begin() + (long)(b * per));
    return o;
}
double med(std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }

int main(int argc, char** argv) {
    std::string dir = argv[1]; int B = std::atoi(argv[2]); std::string mode = argv[3];
    int n, nnz, D; { std::ifstream m(dir + "/meta.txt"); m >> n >> nnz >> D; }
    size_t N = n, NNZ = nnz, Bs = B;
    auto rp = rd<int>(dir + "/row_ptr.i32", N + 1), ci = rd<int>(dir + "/col_idx.i32", NNZ);
    auto v0 = tile(rd<double>(dir + "/values0.f64", D * NNZ), NNZ, D, Bs);
    auto v1 = tile(rd<double>(dir + "/values1.f64", D * NNZ), NNZ, D, Bs);
    auto r0 = tile(rd<double>(dir + "/rhs0.f64", D * N), N, D, Bs);
    auto r1 = tile(rd<double>(dir + "/rhs1.f64", D * N), N, D, Bs);
    auto x0 = tile(rd<double>(dir + "/x0.f64", D * N), N, D, Bs);
    auto x1 = tile(rd<double>(dir + "/x1.f64", D * N), N, D, Bs);
    cudaStream_t st; CK(cudaStreamCreate(&st));
    int *drp, *dci; double *dv, *db, *dx;
    CK(cudaMalloc(&drp, (N + 1) * 4)); CK(cudaMalloc(&dci, NNZ * 4));
    CK(cudaMalloc(&dv, Bs * NNZ * 8)); CK(cudaMalloc(&db, Bs * N * 8)); CK(cudaMalloc(&dx, Bs * N * 8));
    CK(cudaMemcpy(drp, rp.data(), (N + 1) * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dci, ci.data(), NNZ * 4, cudaMemcpyHostToDevice));
    auto up = [&](double* d, const std::vector<double>& h) { CK(cudaMemcpy(d, h.data(), h.size() * 8, cudaMemcpyHostToDevice)); };
    auto down = [&](std::vector<double>& h, double* d) { CK(cudaMemcpy(h.data(), d, h.size() * 8, cudaMemcpyDeviceToHost)); };

    cudssHandle_t h; cudssConfig_t cfg; cudssData_t data; cudssMatrix_t A, xm, bm;
    DS(cudssCreate(&h)); DS(cudssSetStream(h, st));
    bool mt = mode == "time" && argc > 6 && std::atoi(argv[6]);
    if (mt) DS(cudssSetThreadingLayer(h, std::getenv("CUDSS_THREADING_LIB")));
    DS(cudssConfigCreate(&cfg)); DS(cudssDataCreate(h, &data));
    up(dv, v0); up(db, r0);
    DS(cudssMatrixCreateCsr(&A, n, n, nnz, drp, nullptr, dci, dv, CUDSS_R_32I, CUDSS_R_32I, CUDSS_R_64F, CUDSS_MTYPE_GENERAL, CUDSS_MVIEW_FULL, CUDSS_BASE_ZERO));
    DS(cudssMatrixCreateDn(&bm, n, 1, n, db, CUDSS_R_64F, CUDSS_LAYOUT_COL_MAJOR));
    DS(cudssMatrixCreateDn(&xm, n, 1, n, dx, CUDSS_R_64F, CUDSS_LAYOUT_COL_MAJOR));
    if (B > 1 || std::getenv("UB1")) DS(cudssConfigSet(cfg, CUDSS_CONFIG_UBATCH_SIZE, &B, sizeof(B)));
    std::vector<int> all(Bs, 1);
    auto setmask = [&](const std::vector<int>& m) { DS(cudssDataSet(h, data, CUDSS_DATA_UBATCH_MASK, (void*)m.data(), m.size() * 4)); };
    auto run = [&](int ph) { DS(cudssExecute(h, ph, cfg, data, A, xm, bm)); };
    auto maxerr = [&](const std::vector<double>& x, const std::vector<double>& ref, size_t s) {
        double e = 0; for (size_t i = 0; i < N; ++i) e = std::max(e, std::fabs(x[s * N + i] - ref[s * N + i])); return e; };

    if (mode == "time") {
        int ro = std::atoi(argv[4]), fa = std::atoi(argv[5]);
        DS(cudssConfigSet(cfg, CUDSS_CONFIG_REORDERING_ALG, &ro, sizeof(ro)));
        DS(cudssConfigSet(cfg, CUDSS_CONFIG_FACTORIZATION_ALG, &fa, sizeof(fa)));
        if (argc > 7) { int pv = std::atoi(argv[7]); if (pv >= 0) DS(cudssConfigSet(cfg, CUDSS_CONFIG_PIVOT_TYPE, &pv, sizeof(pv))); }
        if (argc > 8) { int sa = std::atoi(argv[8]); DS(cudssConfigSet(cfg, CUDSS_CONFIG_SOLVE_ALG, &sa, sizeof(sa))); }
        if (const char* c = std::getenv("CFG")) {  // "param=value,param=value" (int values)
            std::string spec = c; size_t pos = 0;
            while (pos < spec.size()) {
                size_t comma = spec.find(',', pos); if (comma == std::string::npos) comma = spec.size();
                std::string item = spec.substr(pos, comma - pos); size_t eq = item.find('=');
                int p = std::atoi(item.substr(0, eq).c_str()), v = std::atoi(item.substr(eq + 1).c_str());
                cudssStatus_t s = cudssConfigSet(cfg, (cudssConfigParam_t)p, &v, sizeof(v));
                std::printf("config %d = %d -> %d\n", p, v, (int)s);
                pos = comma + 1;
            }
        }
        if (B > 1 || std::getenv("UB1")) setmask(all);
        cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
        auto timed = [&](int ph) { cudaEventRecord(a, st); run(ph); cudaEventRecord(b, st); cudaEventSynchronize(b); float ms; cudaEventElapsedTime(&ms, a, b); return (double)ms; };
        double an = timed(CUDSS_PHASE_ANALYSIS), fa_ms = timed(CUDSS_PHASE_FACTORIZATION);
        std::vector<double> rf, sv;
        for (int r = 0; r < 9; ++r) { rf.push_back(timed(CUDSS_PHASE_REFACTORIZATION)); sv.push_back(timed(CUDSS_PHASE_SOLVE)); }
        std::vector<double> x(Bs * N); down(x, dx);
        double e = 0; for (size_t s = 0; s < Bs; ++s) e = std::max(e, maxerr(x, x0, s));
        int64_t lu = 0; size_t w; cudssDataGet(h, data, CUDSS_DATA_LU_NNZ, &lu, sizeof(lu), &w);
        std::printf("reorder=%d factor_alg=%d mt=%d B=%d: analysis %.1f factor %.2f refactor %.2f solve %.2f ms (per system %.3f / %.3f), lu_nnz %ld, err %.1e\n",
                    ro, fa, (int)mt, B, an, fa_ms, med(rf), med(sv), med(rf) / B, med(sv) / B, (long)lu, e);
        return 0;
    }

    // mask: what happens to unselected systems?
    setmask(all);
    run(CUDSS_PHASE_ANALYSIS); run(CUDSS_PHASE_FACTORIZATION); run(CUDSS_PHASE_SOLVE);
    std::vector<double> x(Bs * N); CK(cudaStreamSynchronize(st)); down(x, dx);
    std::printf("full solve: sys0 err %.1e sys1 err %.1e\n", maxerr(x, x0, 0), maxerr(x, x0, 1));
    std::vector<int> even(Bs), odd(Bs);
    for (size_t i = 0; i < Bs; ++i) even[i] = i % 2 == 0, odd[i] = i % 2 == 1;
    auto describe = [&](const char* what, size_t s) {
        std::vector<double> sent(N, 777.0), zero(N, 0.0);
        std::printf("  %s sys %zu: vs sentinel %.1e, vs 0 %.1e, vs x0 %.1e, vs x1 %.1e, first %g\n", what, s,
                    maxerr(x, std::vector<double>(Bs * N, 777.0), s), maxerr(x, std::vector<double>(Bs * N, 0.0), s),
                    maxerr(x, x0, s), maxerr(x, x1, s), x[s * N]);
    };
    // 1. Masked SOLVE only (no refactor): even mask, rhs1 everywhere. Do odds get written?
    up(db, r1); up(dx, std::vector<double>(Bs * N, 777.0));
    setmask(even); run(CUDSS_PHASE_SOLVE); CK(cudaStreamSynchronize(st)); down(x, dx);
    std::printf("1. even-masked SOLVE on step-0 factors, rhs1:\n");
    describe("even", 0); describe("odd", 1); describe("even", 2); describe("odd", 3);
    // Reference: what does solving rhs1 with step-0 factors give? solve all.
    // 2. Masked REFACTORIZATION (even) with v1 for evens and v0 for odds (valid values), then full solve rhs0.
    std::vector<double> vm = v0; for (size_t i = 0; i < Bs; i += 2) std::copy_n(v1.begin() + (long)(i * NNZ), NNZ, vm.begin() + (long)(i * NNZ));
    up(dv, vm); DS(cudssMatrixSetValues(A, dv));
    setmask(even); run(CUDSS_PHASE_REFACTORIZATION);
    std::vector<double> rm = r1; for (size_t i = 1; i < Bs; i += 2) std::copy_n(r0.begin() + (long)(i * N), N, rm.begin() + (long)(i * N));
    up(db, rm); up(dx, std::vector<double>(Bs * N, 777.0));
    setmask(all); run(CUDSS_PHASE_SOLVE); CK(cudaStreamSynchronize(st)); down(x, dx);
    std::printf("2. even-masked REFACTOR (odd values unchanged), then unmasked SOLVE (even rhs1, odd rhs0):\n");
    describe("even", 0); describe("odd", 1); describe("even", 2); describe("odd", 3);
    // 3. Same but odd values garbage: does the masked refactor touch odd factors?
    std::vector<double> vg = vm; for (size_t i = 1; i < Bs; i += 2) std::fill_n(vg.begin() + (long)(i * NNZ), NNZ, 1e300);
    up(dv, vg); DS(cudssMatrixSetValues(A, dv));
    setmask(even); run(CUDSS_PHASE_REFACTORIZATION);
    up(db, rm); up(dx, std::vector<double>(Bs * N, 777.0));
    setmask(all); run(CUDSS_PHASE_SOLVE); CK(cudaStreamSynchronize(st)); down(x, dx);
    std::printf("3. even-masked REFACTOR with odd values = 1e300, then unmasked SOLVE:\n");
    describe("even", 0); describe("odd", 1); describe("even", 2); describe("odd", 3);
    // 4. Same with masked FACTORIZATION instead of REFACTORIZATION.
    up(dv, vm); DS(cudssMatrixSetValues(A, dv)); setmask(all); run(CUDSS_PHASE_FACTORIZATION);  // reset: evens v1, odds v0
    up(dv, vg); DS(cudssMatrixSetValues(A, dv));
    setmask(even); run(CUDSS_PHASE_FACTORIZATION);
    up(db, rm); up(dx, std::vector<double>(Bs * N, 777.0));
    setmask(all); run(CUDSS_PHASE_SOLVE); CK(cudaStreamSynchronize(st)); down(x, dx);
    std::printf("4. even-masked FACTORIZATION with odd values = 1e300, then unmasked SOLVE:\n");
    describe("even", 0); describe("odd", 1); describe("even", 2); describe("odd", 3);
    // 5. Masked SOLVE with odd mask after (4): odds solved, evens?
    up(dx, std::vector<double>(Bs * N, 777.0));
    setmask(odd); run(CUDSS_PHASE_SOLVE); CK(cudaStreamSynchronize(st)); down(x, dx);
    std::printf("5. odd-masked SOLVE after (4):\n");
    describe("even", 0); describe("odd", 1); describe("even", 2); describe("odd", 3);
    // 7. Fresh factors: all on vm (even v1, odd v0); refactor odd only on vm; solve all.
    up(dv, vm); DS(cudssMatrixSetValues(A, dv)); setmask(all); run(CUDSS_PHASE_REFACTORIZATION);
    setmask(odd); run(CUDSS_PHASE_REFACTORIZATION);
    up(db, rm); up(dx, std::vector<double>(Bs * N, 777.0));
    setmask(all); run(CUDSS_PHASE_SOLVE); CK(cudaStreamSynchronize(st)); down(x, dx);
    std::printf("7. all REFACTOR, then odd-masked REFACTOR, then unmasked SOLVE:\n");
    describe("even", 0); describe("odd", 1); describe("even", 2); describe("odd", 3);
    // 8. Then even-masked SOLVE (evens not in the last refactor).
    up(dx, std::vector<double>(Bs * N, 777.0));
    setmask(even); run(CUDSS_PHASE_SOLVE); CK(cudaStreamSynchronize(st)); down(x, dx);
    std::printf("8. even-masked SOLVE after (7):\n");
    describe("even", 0); describe("odd", 1);
    // 9. After (7) the evens' factors are gone: an unmasked REFACTOR on vm, then solve all.
    up(dv, vm); DS(cudssMatrixSetValues(A, dv)); setmask(all); run(CUDSS_PHASE_REFACTORIZATION);
    up(db, rm); up(dx, std::vector<double>(Bs * N, 777.0));
    run(CUDSS_PHASE_SOLVE); CK(cudaStreamSynchronize(st)); down(x, dx);
    std::printf("9. unmasked REFACTOR after (7), unmasked SOLVE:\n");
    describe("even", 0); describe("odd", 1); describe("even", 2); describe("odd", 3);
    // 6. Does the mask read back as set?
    std::vector<int> back(Bs, -1); size_t w = 0;
    cudssStatus_t s = cudssDataGet(h, data, CUDSS_DATA_UBATCH_MASK, back.data(), Bs * 4, &w);
    std::printf("6. DataGet(UBATCH_MASK) status %d written %zu: %d %d %d %d\n", (int)s, w, back[0], back[1], back[2], back[3]);
    return 0;
}
