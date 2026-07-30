# GPU acceleration plan for gridoxide's power flow

Status: **proposal**, not implemented. Written 2026-07-25 against `d9b8799`.

> ## Amendment, 2026-07-30 — after Phases 0–3 were built and measured
>
> Phases 0–3 were implemented and benchmarked. The result was a **91x
> regression** against the Phase 0 CPU baseline (case1354pegase, batch 4,096:
> 52.7 GPU solves/s vs. 4,825 on 30 EPYC threads). Instrumenting it located
> ~95% of the time inside cuDSS, refactorizing and solving a single
> 10-million-row stacked matrix. Three things in this document are wrong or
> incomplete as a result, and the corrections drive the current work:
>
> 1. **§3 property 2 ("it needs no batched solver API") is the trap.** Block
>    diagonal embedding is *mathematically* equivalent to B independent solves
>    — the proof in `src/bde.rs` stands — but it is not *computationally*
>    equivalent on a GPU. A general sparse direct solver handed one enormous
>    matrix cannot know it is really B independent 2,450-row problems, and
>    pays scheduling and bookkeeping over the whole assembly forest. On NVIDIA
>    the fix is cuDSS's **uniform batch API**, which this document listed in
>    §4.1's table and then never used. See
>    `sparse_cudss::CudssBatchedSystem`.
> 2. **§2's published-results calibration is against the wrong baseline.**
>    SABLE's 253x and Fraunhofer's 100x are over *pandapower*. gridoxide's own
>    30-thread KLU `BatchSolver` is a far higher bar, and no published number
>    in §2 says anything about clearing it. The current exit criterion is
>    therefore "beat the 30-thread CPU baseline at all", not ">=10x".
> 3. **§5's portability recommendation (one CubeCL kernel codebase) was
>    dropped.** CubeCL hid stream control, allocation lifetime and raw device
>    pointers — all three needed for a batched, device-resident loop. The
>    kernels are now hand-written CUDA (`cuda/gridoxide_kernels.cu`) and
>    NVIDIA-only. The portable kernel is preserved at commit `ff92b66`; see §5.
>
> §1's per-phase CPU measurements and §3's other three properties are
> unaffected and still hold.
>
> ## Amendment 2, 2026-07-30 — Session 3, the batched rewrite, on an A100
>
> `CudssBatchedSystem` (`src/sparse_cudss.rs`) was compiled, run, and
> correctness-verified for the first time this session, on a rented
> **A100-SXM4-40GB** (driver 580.126.20, CUDA 13.0/`nvcc` 13.0.88, cuDSS
> 0.8.0.10 via the `libcudss0-cuda-13` apt package, `CUDA_ARCH=sm_80`, host
> 30-core AMD EPYC 7J13). It had never been compiled before — everything below
> was written blind, on a machine with no NVIDIA GPU.
>
> **Three FFI bugs, all now fixed:**
>
> 1. `build.rs`'s nvcc-not-on-`PATH` fallback called `cc::Build::compiler()`
>    with the nvcc path. `cc-rs`, in CUDA mode, treats an explicit
>    `.compiler()` as the *host* compiler nvcc wraps via `-ccbin`, not nvcc
>    itself — silently disabling the `-Xcompiler` wrapping it normally applies
>    to GNU-family flags, so nvcc saw a raw `-ffunction-sections` and rejected
>    it outright. Fixed by setting the `NVCC` env var instead, which routes
>    through `cc-rs`'s CUDA-aware path.
> 2. `cudssMatrixCreateBatchCsr`'s hand-written signature was missing the
>    `offsetType` parameter (real cuDSS 0.8 splits offset/index/value types
>    three ways, not two) and passed `*mut *mut c_void` where the real
>    signature wants `*const *const c_void` — a single-level `*mut → *const`
>    coerces implicitly in Rust, a double-level one does not, so this was a
>    compile error, not a runtime one. Same fix applied to
>    `cudssMatrixCreateBatchDn` and `cudssMatrixSetBatchValues`.
> 3. Neither bug was silent-wrong-numbers, matching the risk table's
>    prediction for Step 1/2 — both were loud compile errors once `nvcc` and
>    real cuDSS headers existed to check against.
>
> **Correctness: full PASS.** `bde_test` (10/10), `gpu_assembly_test` (4/4),
> `batched_ffi_smoke_test`, and `bde_check` on `case9241pegase`/256 all agree
> with independent CPU solves to ~7e-13 — better than the ~1e-9 the runbook
> expected, well inside the ~1e-6 bar.
>
> **Performance, before the threading fix below:** batched beat the stacked
> control by 1.45–1.58x across batch 64/256/1024, but both were ~20–30x
> *slower* than the 30-thread CPU baseline. Per-phase CUDA-event timers
> (`bde_profile`) isolated why: `PHASE_ANALYSIS` alone — not
> `PHASE_FACTORIZATION`, not the GPU kernels — was 95%+ of wall time and
> scaled near-linearly with batch count (20.7s at batch 256, 100.2s at batch
> 1024), meaning cuDSS's uniform batch API was **not** sharing one analysis
> across the batch as its name implies; it was running full per-matrix
> symbolic analysis, unshared.
>
> **The real fix — cuDSS defaults to single-threaded host execution.**
> `nvidia-smi` showed 0% GPU utilization and `ps` showed ~137% CPU during
> `PHASE_ANALYSIS`, regardless of batch size — one core's worth of work, not
> thirty. cuDSS ships an opt-in OpenMP threading layer
> (`libcudss_mtlayer_gomp.so`, installed alongside `libcudss.so` by the apt
> package but never loaded unless `cudssSetThreadingLayer` is called
> explicitly). Calling it (`sparse_cudss::enable_host_threading`, applied to
> both `CudssBatchedSystem` and the `CudssRealSystem` control) cut analysis
> time by **~3.2x** on this 30-core host, correctness unaffected. This is a
> real, permanent, committed fix, not a workaround — `build.rs` now embeds an
> rpath to cuDSS's lib directory via `CUDSS_LIB_DIR` so the `.so` is found
> without relying on the box having registered it with `ldconfig` (the apt
> package does not do this itself — a second, separate gap this session hit
> and worked around manually before finding the rpath fix).
>
> **Batch-size sweep, after the threading fix (`case9241pegase`, `klu` CPU
> baseline ~230–280 solves/s across all batch sizes):**
>
> | batch | CPU (30 threads) | GPU batched | vs CPU | vs stacked control |
> |---|---|---|---|---|
> | 64 | 143.0 solves/s | 38.4 solves/s | 0.27x | 2.65x |
> | 256 | 233.1 solves/s | 35.7 solves/s | 0.15x | 2.51x |
> | 1024 | 277.5 solves/s | 31.1 solves/s | 0.11x | 2.38x |
>
> The threading fix is real and consistent (~2.4–2.7x over stacked at every
> size), but **the CPU-speedup ratio gets worse, not better, as batch size
> grows** — the opposite of what a shared-analysis batch API should do. Setup
> (dominated by `PHASE_ANALYSIS`) still scales slightly *faster* than linear
> with batch count even after threading, because analysis is still run once
> per matrix rather than once per batch. Batch 4096 was attempted and killed
> after 32+ minutes without finishing (`nvidia-smi`: 26GB device memory
> resident, 0% GPU utilization, 319% CPU) — the stacked control at that size
> is a ~70-million-unknown monolithic matrix and its reordering cost is
> almost certainly the dominant factor, consistent with §3 property 2's
> documented failure mode, not new information worth more GPU time to confirm.
>
> **Unresolved lead, worth a future session with real cuDSS documentation.**
> `cudssConfigParam_t` includes `CUDSS_CONFIG_UBATCH_SIZE`/`_UBATCH_INDEX`
> ("U" for Uniform) — an `int` config, default 1, documented (per NVIDIA's
> own docs site) to be set before `PHASE_ANALYSIS` to activate genuinely
> shared analysis across a batch. Setting it (confirmed accepted,
> `cudssConfigSet` returns success) collapsed a toy 2-system batch's
> analysis+factorization+solve from seconds to milliseconds — a **~294x**
> reduction on the toy case — but every one of 12 combinations tried
> (reordering algorithm ∈ {default, AMD, nested dissection, none}; pivot type
> ∈ {default, none}; matrix type ∈ {general, symmetric, SPD}; aliased vs.
> distinct CSR structure pointers; `batchCount=1` with a `UBATCH_SIZE`-sized
> value buffer vs. `batchCount=B`) failed identically at `PHASE_ANALYSIS`
> with `CUDSS_STATUS_NOT_SUPPORTED`. This is a real, sanctioned, documented
> feature that is currently unreachable from this codebase without NVIDIA's
> own sample code or support — if unlocked, the toy-problem result suggests
> it would flip this session's whole conclusion. The probe tests
> (`sparse_cudss::batched_ffi_smoke_test::ubatch_probe*`) were not committed;
> reproduce from this description if picking the thread back up.
>
> **Bottom line:** the batched device-resident path is now correct, ~2.4–2.7x
> faster than the stacked control at every batch size tried, and the
> threading fix is a genuine, permanent, committed win — but it still does
> not clear the 30-thread CPU `BatchSolver`, and the gap widens with batch
> size rather than closing. `UBATCH_SIZE` is the one lead that could change
> that conclusion; everything else diagnosed this session (assembly,
> injections, Newton update, host sync) is already <5% of wall time and not
> worth further optimization until analysis-sharing is solved.

This document answers: *how much of the AC power flow can realistically move to the
GPU, on which stack, and in what order?* It covers CUDA, cuda-oxide, JAX, AMD/ROCm,
and the portable Rust options, and ends with a phased plan.

The short version:

> **Do not GPU-accelerate a single power flow solve.** Measured on this repo, a
> single 9,242-bus solve is ~5.8 ms/iteration, ~60% of which is sparse LU that GPUs
> are bad at. Amdahl caps a perfect GPU port of everything else at **~1.65×**, which
> PCIe latency would then eat. **The GPU win is in *batches*** — N-1 contingency
> screening, time series, Monte Carlo, ML training loops — where published results
> show 100–250× over CPU baselines. Design for batch from day one, via
> **block-diagonal embedding** onto a batched sparse refactorization solver
> (**cuDSS** on NVIDIA, **rocSOLVER `csrrf_*`** on AMD).

---

## 1. Baseline: where the time actually goes

Measured on this machine by instrumenting `newton_raphson_native_klu_cached`
(`src/solver.rs`) with per-phase `Instant` timers, `--release`, `klu_native`
backend, `warm` mode (symbolic factorization reused), 5 repeats, averaged over
all 12 recorded iterations per case. The instrumentation was throwaway and is
not committed.

| case | buses | mismatch (`power_injections`) | Jacobian assembly (`build_jacobian_triplets`) | sparse LU (refactor + solve) | total/iter |
|---|---|---|---|---|---|
| case1354pegase | 1,355 | 23.8 µs (6.5%) | 150.0 µs (41.1%) | 191.4 µs (52.4%) | 365 µs |
| case2869pegase | 2,870 | 56.9 µs (6.1%) | 361.7 µs (39.0%) | 509.4 µs (54.9%) | 928 µs |
| case9241pegase | 9,242 | 214.4 µs (3.7%) | 2,068.1 µs (35.7%) | 3,517.6 µs (60.6%) | 5,800 µs |

Two things fall straight out of this table.

**(a) The GPU-friendly fraction is 39–47%.** Mismatch evaluation and Jacobian
assembly are embarrassingly parallel — one independent thread per Y-bus nonzero,
pure gather/FMA over `sin`/`cos`, no data dependencies. That is textbook GPU work.

**(b) The remaining 54–61% is sparse LU, which is textbook *anti*-GPU work.**
KLU-style factorization with partial pivoting is sequential along the elimination
tree, branch-heavy, and irregular in its memory access. It is exactly the kernel
GPUs lose at on a *single* matrix of this size. Note this repo already has four
CPU implementations of it (`faer`, vendored KLU, `klu_native`, PARDISO), and the
best of them does 9,242 buses in ~3.5 ms.

**The Amdahl ceiling for single-case acceleration:** moving 100% of (a) to the GPU
and leaving (b) on the CPU gives at most `1 / 0.606 = 1.65×` on case9241pegase and
`1 / 0.524 = 1.91×` on case1354pegase. Against that you must pay a host↔device
round trip *every Newton iteration* (voltages down, triplets up, or triplets down
and Δx up). At case1354pegase's 365 µs/iteration budget, a few tens of microseconds
of PCIe latency and launch overhead per iteration is a double-digit percentage of
the whole thing. Realistically single-case GPU offload is a **wash or a
regression**. This is the single most important conclusion in this document, and
it is why the plan below is organized entirely around batching.

---

## 2. Where a GPU actually wins

| Workload | Scale | GPU verdict |
|---|---|---|
| One solve, ≤10k buses | 29 ms (`klu`, case9241pegase) | ❌ **No.** Amdahl-capped at ~1.65×, then PCIe eats it. |
| One solve, ≥100k buses | not currently benchmarked | 🟡 Maybe. Assembly fraction grows; LU still hostile. Low priority. |
| **N-1 contingency screening** | 10³–10⁴ solves, same topology class | ✅ **Yes — the primary target.** |
| **Time series / QSTS** | 10³–10⁵ solves, identical topology | ✅ **Yes — the easiest target.** Only `p_spec`/`q_spec` vary. |
| **Monte Carlo / stochastic PF** | 10³–10⁶ solves | ✅ Yes, same shape as time series. |
| **Differentiable PF layer for ML** | 10²–10³ per training step | ✅ Yes, and it's where the field is going (see SABLE below). |
| CGMES/PGM parsing, Y-bus build | ~once | ❌ No. Host-side, I/O and pointer-chasing bound. |

Published results for the batched case, for calibration:

- **Fraunhofer** (Zhenqi Wang et al., [arXiv:2101.02270](https://arxiv.org/pdf/2101.02270)) —
  batched Newton-Raphson via batched sparse LU *refactorization* on GPU, reporting
  **>100×** over pandapower for repeated power flows sharing one sparsity pattern.
- **SABLE** ([arXiv:2606.07099](https://arxiv.org/html/2606.07099), June 2026) —
  **up to 253×** over pandapower and **5.7×** over ExaPF (an existing GPU baseline),
  on 1,354–25,000 bus systems at batch sizes 4–256.
- **JAX batched AC PF** ([arXiv:2605.14103](https://arxiv.org/html/2605.14103)) —
  **14.3×** over multithreaded pandapower on a 2,224-node network, and **4,700×**
  for batched-GPU vs. single-scenario-GPU execution. The lower headline number vs.
  the two above is the direct consequence of JAX having no batched sparse *direct*
  solver (see §4.3).

Note the gap between 14× and 253× is almost entirely "does your stack have a batched
sparse direct solver." That drives the architecture in §3.

---

## 3. Target architecture: block-diagonal embedding

The design that both the Fraunhofer and SABLE work converge on, and the one this
plan adopts:

Every scenario in a batch shares **one sparsity pattern** (same topology, different
injections). So instead of *B* separate sparse solves, stack the batch into a single
block-diagonal matrix:

```
        ┌ J₁              ┐   ┌ Δx₁ ┐   ┌ f₁ ┐
        │    J₂           │   │ Δx₂ │   │ f₂ │
  J  =  │       ⋱         │ , │  ⋮  │ = │ ⋮  │
        └             J_B ┘   └ Δx_B┘   └ f_B┘
```

The properties that make this the right choice:

1. **One symbolic analysis for the entire batch, ever.** The block-diagonal pattern
   is fixed for the lifetime of the topology. Ordering/fill-in is computed once and
   reused across every scenario and every Newton iteration — exactly what
   `PersistentSolver` (`src/solver.rs`) already does for one scenario, extended to
   *B*. The README already measures this as worth ~45% on case9241pegase.
2. **It works on *any* sparse direct solver**, batched API or not. You are handing
   the library one ordinary sparse matrix. This decouples the architecture from
   vendor-specific batched entry points and is what makes the AMD path viable
   (§5) even though rocSOLVER's refactorization routines are not batched.

   > **Measured correction (2026-07-30).** True, and the reason the GPU path
   > lost by 91x. "Works on any solver" is a statement about *correctness*; it
   > says nothing about throughput. Handing cuDSS one 10M-row matrix instead of
   > B uniform 2,450-row systems cost ~95% of the runtime — the solver has no
   > way to recover the block structure the embedding threw away. Keep BDE as
   > the *data layout* (the scenario-major values buffer, the shared CSR
   > structure, the identity masking are all still exactly right); stop using
   > it as the *solver interface* wherever a batched entry point exists. On
   > AMD, where rocSOLVER has none, expect this same cost and budget for it
   > rather than assuming BDE makes the batched API optional.
3. **It saturates the GPU.** A single 9,242-bus Jacobian (~18k unknowns) cannot fill
   a modern GPU. 256 of them stacked can.
4. **Assembly becomes one flat kernel.** All *B* scenarios' Jacobian nonzeros are
   written into one preallocated value array at precomputed offsets — one thread per
   (scenario, nonzero) pair, fully coalesced, zero branching.

Additional design commitments:

- **Per-scenario convergence masking.** Scenarios converge at different iteration
  counts. Do *not* early-exit the batch; keep a per-scenario active mask, skip
  converged scenarios' updates, and stop when all are done or `max_iter` is hit.
  Divergent contingency scenarios are normal and must not poison the batch.
- **Mixed precision, following SABLE.** Assemble the Jacobian and evaluate the
  convergence residual in **f64**; run the dominant linear-solve stage in **f32**,
  with an f64 residual check and iterative refinement. This roughly doubles
  effective throughput and is what makes consumer GPUs (with crippled FP64, §5)
  usable at all. It must be validated against the existing 12-case accuracy suite
  before being made the default.
- **Reuse the existing Jacobian math verbatim.** `build_jacobian_triplets`'s H/N/M/L
  formulas are already correct and cross-validated against five independent solvers.
  The GPU kernel is a transliteration, not a rederivation.

---

## 4. Options

### 4.1 Comparison

| # | Option | Vendors | Layer | Rust-native | f64 | Batched sparse direct | Maturity | Verdict |
|---|---|---|---|---|---|---|---|---|
| A | **cuDSS** via `bindgen` | NVIDIA | library | FFI | ✅ | ✅ uniform + non-uniform | production | ✅ **Primary, NVIDIA** |
| B | **rocSOLVER `csrrf_*`** via `bindgen` | AMD | library | FFI | ✅ | ⚠️ not batched — use BDE | production | ✅ **Primary, AMD** |
| C | `cudarc` | NVIDIA | bindings | ✅ safe | ✅ | ❌ no cuDSS binding | actively maintained | 🟡 Host-side glue only |
| D | **CubeCL** | NVIDIA + AMD + wgpu | kernel DSL | ✅ | ⚠️ backend-dep. | ❌ (not a solver) | alpha | ✅ **Assembly kernels** |
| E | `cuda-oxide` | NVIDIA | rustc→PTX | ✅ | ✅ | ❌ (not a solver) | alpha | 🟡 Watch |
| F | `wgpu` / WGSL | everything | runtime + shaders | ✅ | ❌ **no f64** | ❌ | production | 🟡 f32 paths only |
| G | `rust-gpu` (SPIR-V) | everything | rustc→SPIR-V | ✅ | ❌ practically | ❌ | graphics-focused | ❌ Wrong tool |
| H | **JAX** | NVIDIA + ROCm | Python framework | ❌ | ✅ (x64 mode) | ❌ GMRES only | production | ✅ **Prototyping** |
| I | PyTorch + CuPy + cuDSS | NVIDIA | Python stack | ❌ | ✅ | ✅ | production | 🟡 If ML integration is the goal |

### 4.2 The Rust/CUDA options, disentangled

These three get conflated constantly, including by their names. They do different things:

- **`cudarc`** is a *binding library* — safe Rust wrappers over the CUDA driver API
  plus cuBLAS/cuSPARSE/cuSOLVER, for launching kernels written elsewhere. It has
  `cusparse` and `cusolver` feature flags. It does **not** wrap cuDSS.
  → Use it for device memory, streams, and launching. Not for the sparse solve.

- **`cuda-oxide`** is, since NVlabs' May 2026 release, **not** what its crates.io
  history suggests. It is now a `rustc` codegen backend that compiles ordinary Rust
  to PTX — write SIMT kernels in safe(ish) Rust, no DSL, no separate language. It
  supports generics, closures, structs, enums, shared memory, warp intrinsics, and
  atomics. It is explicitly **alpha**: "expect bugs, incomplete features, and API
  breakage." NVIDIA-only.
  → Genuinely interesting for the assembly kernels, but locks out AMD, which the
  hardware situation (§5) makes a poor trade *right now*. Revisit in 6–12 months.

- **Rust-CUDA / `cust`** is the older community rustc→NVVM backend, with the inverse
  priority to cuda-oxide (bring *Rust* to the GPU — `async`, parts of `std` — rather
  than bring *CUDA* to Rust). Not recommended here; cuda-oxide has NVIDIA behind it.

**Important:** none of these three solve the sparse system. The 60% of runtime that
is LU comes from cuDSS/rocSOLVER regardless. Kernel-authoring choice only affects
the 40% that is assembly.

### 4.3 JAX

Worth taking seriously as a **prototyping and validation** track, not as the
production path.

What it buys: `vmap` + `jit` give batching essentially for free, autodiff comes
along for the ride (valuable if a differentiable PF layer is ever wanted), and it
runs on **both** CUDA and ROCm. Turnaround on "does batched Newton converge on our
12 cases" is days, not weeks.

What it costs: `jax.experimental.sparse` (BCOO) is, in the JAX PF paper's own words,
"less mature than conventional sparse linear-algebra stacks," and there is **no
batched sparse direct solver**. That paper therefore uses **GMRES with a
fast-decoupled preconditioner** instead of factorization. That is a real algorithmic
change — iterative Krylov on power flow Jacobians is fragile on ill-conditioned
transmission cases, and three of this repo's own 12 benchmark cases already have
known convergence sensitivity. It also explains the 14× vs. 253× gap in §2. The
paper also notes double precision must be force-enabled, which hurts on
non-datacenter GPUs.

**Recommendation:** use JAX in Phase 1 as an executable specification and numerical
oracle for the batched formulation. Do not ship it.

### 4.4 Options explicitly rejected

- **WGSL f64 emulation (double-single / `vec2<f32>`).** WGSL has no f64 and the
  WebGPU proposal ([gpuweb#2805](https://github.com/gpuweb/gpuweb/issues/2805)) is
  still unresolved. Emulating it costs ~10–20× per operation and gives ~f48
  precision. For a solver whose whole value proposition is agreeing with five other
  implementations to 4+ decimal places, this is not a trade worth making. If the
  portable path is taken, take it as honest **f32 + f64 iterative refinement**
  instead.
- **Porting `klu_native` to the GPU.** Sequential elimination tree, partial pivoting,
  Eisenstat-Liu pruning — every property that makes it fast on a CPU makes it
  unmappable to SIMT. Use vendor libraries.
- **`rust-gpu`.** SPIR-V/graphics oriented; f64 support in compute is not a design
  goal.

---

## 5. AMD

AMD is a **credible primary target**, not a fallback — with one very large caveat
about *which* AMD hardware.

**The case for AMD.** FP64 is where AMD's datacenter parts are strongest. An Instinct
MI300X does **81.7 TFLOPS peak vector FP64** — comfortably ahead of NVIDIA's
comparable-generation datacenter parts, and power flow is an FP64-hungry workload.
For a solver that cares about matching reference implementations to 4 decimal places,
that matters more than tensor-core throughput.

**The software path exists.** rocSOLVER ships
[`csrrf_analysis` / `csrrf_refactlu` / `csrrf_splitlu` / `csrrf_sumlu` /
`csrrf_solve`](https://rocm.docs.amd.com/projects/rocSOLVER/en/develop/api/refact.html)
— a direct analogue of NVIDIA's (now-deprecated) cuSolverRF: analyze once, then
refactorize repeatedly against the same sparsity pattern. That is *precisely* the
access pattern §3 needs. It is not a batched API, but with block-diagonal embedding
it does not need to be — that is the third property listed in §3, and it is why BDE
should be treated as non-negotiable rather than a CUDA-specific trick.

**The caveat: this machine's GPU is the wrong AMD GPU.** `lspci` reports a HawkPoint
APU iGPU — a Radeon 780M-class **gfx1103** part. Two problems:

1. **FP64 is 1/32 rate on RDNA3 consumer silicon** (worse than RDNA2's 1/16).
2. **gfx1103 is not officially supported by ROCm.** It works only via
   `HSA_OVERRIDE_GFX_VERSION=11.0.0`/`11.0.2` pretending to be gfx1100/gfx1102,
   with community-rebuilt libraries; reports include hard system locks on some
   override values, and MIOpen ships no precompiled kernels for it.

So: this machine is usable for **development and correctness testing** of the AMD
path (f32 kernels, small batches, API shakeout), and is **useless for performance
numbers**. Any AMD benchmark claim needs a real Instinct part — cloud-rented is fine.

**Portability recommendation:** write assembly kernels once in **CubeCL** (option D),
which JITs the same `#[cube]` Rust function to CUDA, ROCm/HIP, *and* WGSL. Keep the
sparse solve behind a trait with two FFI implementations (cuDSS, rocSOLVER). That
gives one kernel codebase and two thin solver shims, rather than two full stacks.
CubeCL is alpha and used in production by Burn; budget for API churn.

> **Reversed, 2026-07-30. AMD is deferred; the kernels are NVIDIA-only CUDA.**
>
> The CubeCL assembler was built, validated to f64 exactness, and then removed.
> It is preserved in git history at commit **`ff92b66`** — `src/gpu.rs` there is
> the portable `#[cube]` kernel, generic over `R: Runtime`, and reverting to it
> means reverting the dtype to f32 for the wgpu path as its own doc comment
> explains.
>
> It was dropped for three things its runtime did not expose, each of which the
> batched device-resident loop needs:
>
> - **Streams.** gridoxide's kernels ran on CubeCL's stream and cuDSS on the
>   default one, forcing a `cudaDeviceSynchronize` — a whole-device stall —
>   once per Newton iteration. One shared stream makes the ordering free.
> - **Allocation lifetime.** CubeCL allocated a fresh device buffer per input
>   per launch: ~178 MB of host-gathered upload and five `cudaMalloc`s *per
>   iteration* at case1354pegase/4,096.
> - **Raw device pointers**, stable across launches, for cuDSS's batched matrix
>   to bind against once.
>
> This is a real loss and it should be stated as one: there is now no AMD path,
> and reinstating one means either restoring the CubeCL kernel or writing HIP
> alongside the CUDA. What it bought is the ability to test the central
> hypothesis (§3 property 2's correction) at all. The AMD case in this section —
> MI300X's 81.7 TFLOPS FP64, rocSOLVER's `csrrf_*` — is unchanged and still
> good; it is the *hardware* that was never available, and the local gfx1103
> iGPU could never have produced a performance number either way.
>
> Note also that the correction to §3 property 2 makes AMD *harder*, not
> easier: rocSOLVER has no batched entry point, so the ~95%-in-the-solver cost
> measured here is exactly what an AMD port would inherit. That should be
> designed for up front rather than discovered.

---

## 6. Phased plan

Each phase has an exit criterion. **Phases 0–1 deliver value with no GPU at all**,
and are prerequisites regardless of which backend wins. If the project stops after
Phase 0, it has still gained something real.

### Phase 0 — Batch API + multithreaded CPU baseline *(no GPU)*

The honest first move. Before claiming a GPU speedup you need (a) an API that can
express a batch and (b) a CPU baseline that is not artificially single-threaded.
This repo currently has **no parallelism at all** — no `rayon`, one thread.

- Add `solver::BatchSolver`: one topology, *N* scenarios (varying `p_spec`/`q_spec`/
  setpoints), returning *N* `PowerFlowReport`s.
- Parallelize across scenarios with `rayon`, one `PersistentSolver` per thread.
- Expose it through `src/python.rs` and add `scripts/bench/bench_batch.py`.

*Exit:* near-linear scaling to physical core count on a 256-scenario case9241pegase
batch. **This is also the number every later GPU claim must beat** — beating
single-threaded CPU is not a result.

*Risk:* this phase may capture enough of the win that Phases 2–4 are not worth it.
That is a legitimate outcome and finding it out cheaply is the point.

### Phase 1 — Refactor to a `LinearSolver` trait + JAX oracle

Today each backend is a separately duplicated `newton_raphson_*_cached` function in
`src/solver.rs` (~5 near-identical copies of the same Newton loop), even though
`RealSparseSystem`, `KluRealSystem`, and `KluNativeSystem` all already expose the
identical `factor_and_solve(&entries, &rhs) -> Option<Vec<f64>>` signature. Adding
a GPU backend on top of that structure would mean a sixth copy.

- Extract `trait LinearSolver { fn factor_and_solve(...) }`; collapse the duplicated
  Newton loops into one generic driver. Pure refactor, no behavior change — the
  existing 12-case suite must produce byte-identical voltages.
- In parallel, prototype the batched formulation in JAX (§4.3) as a numerical oracle
  for block-diagonal embedding, validated against the existing accuracy suite.

*Exit:* one Newton loop; all existing backends pass unchanged; JAX prototype agrees
with `klu` on all 12 cases.

### Phase 2 — GPU Jacobian assembly (CubeCL, portable)

Attack the 36–41% first: it is the low-risk half and it is vendor-neutral.

- Precompute, once per topology, the mapping from Y-bus nonzeros to Jacobian value
  offsets (host-side, reusing `build_jacobian_triplets`'s indexing logic).
- CubeCL `#[cube]` kernel: one thread per (scenario, Y-bus nonzero), writing H/N/M/L
  into the batch's flat value array. Same for `power_injections`.
- Keep the LU on the CPU initially; validate the GPU-assembled values against the
  CPU assembler bit-for-bit in f64.

*Exit:* GPU-assembled Jacobian values match CPU to f64 exactness on all 12 cases;
kernel runs on both the local AMD iGPU (via wgpu and/or ROCm) and CUDA.

*Note:* per §1, expect **no end-to-end speedup yet** at batch size 1. The win only
appears once Phase 3 keeps the data resident on the device.

### Phase 3 — Batched sparse solve, device-resident

The payoff phase. Also the riskiest.

- Build the block-diagonal Jacobian on-device; never round-trip per iteration.
- `feature = "cudss"` → cuDSS via `bindgen`. `feature = "rocsolver"` → rocSOLVER
  `csrrf_*` via `bindgen`. Both follow the **existing precedent in `build.rs`**,
  which already does exactly this shape of thing twice (vendored KLU with a static
  build, PARDISO against a system oneMKL discovered via `MKLROOT`). Neither library
  should be vendored; link system installs and document licensing in a
  `PROVENANCE.md` alongside the existing ones.
- Per-scenario convergence masking (§3).

*Exit:* ≥10× over the Phase 0 multithreaded CPU baseline at batch ≥256 on
case9241pegase, with voltages matching `klu` to the accuracy suite's existing
tolerances.

### Phase 4 — Mixed precision + iterative refinement

- f32 linear solve, f64 assembly and residual, f64 iterative refinement (§3).
- Gate behind a flag; validate against the full accuracy suite before defaulting.

*Exit:* ≥1.5× over Phase 3 with no accuracy regression on any of the 12 cases.

### Phase 5 — Integration

- Wire the batch path into `scripts/bench/run_case_suite.py` and the README tables.
- Benchmark against ExaGO/ExaPF and, if it is released, SABLE.
- Document AMD-vs-NVIDIA results on real datacenter hardware, not the local iGPU.

---

## 7. Principal risks

| Risk | Severity | Mitigation |
|---|---|---|
| Phase 0 captures most of the win; GPU never pays for itself | Medium | Deliberately front-loaded — Phase 0 is cheap and answers this before any GPU spend. |
| No local NVIDIA GPU; local AMD GPU is unsupported + 1/32 FP64 | **High** | Rent cloud A100/H100 and MI300X for benchmarking. Use local iGPU for correctness only. Never quote local perf numbers. |
| CubeCL is alpha; breaking changes between minor versions | Medium | Pin exact versions. Keep kernels small and mechanical so a rewrite is days, not weeks. |
| cuDSS/rocSOLVER licensing for redistribution | Medium | Same posture as the existing `pardiso` feature: link system installs, vendor nothing, opt-in feature, document in `PROVENANCE.md`. |
| f32 solve breaks the 4-decimal agreement with 5 other solvers | Medium | Phase 4 is gated and optional; f64 path stays the default until proven. |
| Batched contingency scenarios diverge and destabilize the batch | Medium | Per-scenario masking, designed in from Phase 3, not retrofitted. |
| Two GPU backends (CUDA + ROCm) double maintenance | Medium | One CubeCL kernel codebase; vendor difference confined to the `LinearSolver` impl. |

---

## 8. Recommendation

1. **Do Phase 0 now.** A `rayon` batch API is worth having on its own merits, is a
   prerequisite for everything else, and establishes the only baseline against which
   a GPU claim is meaningful. There is no GPU dependency and no hardware blocker.
2. **Do Phase 1 next.** The `LinearSolver` trait refactor is overdue independently of
   GPU work — five copies of the same Newton loop is already a maintenance cost.
3. **Then decide on hardware before Phase 2.** The choice of primary GPU vendor
   should follow from what hardware the project will actually benchmark and deploy
   on. If that is undecided, CubeCL + block-diagonal embedding is specifically
   chosen to defer the decision as long as possible.
4. **Treat AMD as a first-class target, but get real hardware.** The rocSOLVER
   `csrrf_*` path is sound and MI300X FP64 is genuinely best-in-class for this
   workload. The local gfx1103 iGPU is a development device, not a benchmark device.

**Open question for the maintainer:** what is the actual driving workload? The plan
above optimizes for batch throughput (contingency/time series/Monte Carlo). If the
real goal is instead *single* very large grids (100k+ buses) or a differentiable PF
layer for ML training, the priorities shift materially — the latter would move
option I (PyTorch + cuDSS, SABLE's stack) from "maybe" to "primary."

---

## 9. References

- Wang et al., *Fast parallel Newton-Raphson power flow solver for large number of
  system calculations with CPU and GPU* — [arXiv:2101.02270](https://arxiv.org/pdf/2101.02270)
- *SABLE: GPU-Based Power Flow Accelerator for Sparsity-Aware Batched Learning* —
  [arXiv:2606.07099](https://arxiv.org/html/2606.07099)
- *JAX-Based Batched AC Power Flow for GPU Acceleration and AI Ecosystem Integration* —
  [arXiv:2605.14103](https://arxiv.org/html/2605.14103)
- [NVIDIA cuDSS](https://docs.nvidia.com/cuda/cudss) — direct sparse solver; replaces
  cuSOLVERSp/cuSOLVERRf; uniform and non-uniform batching since 0.4.0
- [rocSOLVER refactorization and direct solvers](https://rocm.docs.amd.com/projects/rocSOLVER/en/develop/api/refact.html)
- [cuda-oxide](https://github.com/NVlabs/cuda-oxide) and its
  [Rust+GPU ecosystem comparison](https://nvlabs.github.io/cuda-oxide/appendix/ecosystem.html)
- [CubeCL](https://github.com/tracel-ai/cubecl)
- [cudarc](https://docs.rs/cudarc/latest/cudarc/)
- [WGSL f64 proposal (unresolved)](https://github.com/gpuweb/gpuweb/issues/2805)
- [AMD Instinct MI300X specifications](https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html)
