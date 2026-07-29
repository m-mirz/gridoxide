# Bounded cuda-oxide spike for GPU Jacobian assembly

Status: **proposal**, not implemented. Written 2026-07-29 against `12e682c` on the `gpu` branch, in the style of `plans/GPU_PLAN.md`.

## Context

This session built a device-resident GPU batch-solve path for gridoxide's Newton-Raphson power flow: `src/gpu.rs` (a CubeCL `#[cube]` kernel assembling Jacobian values on-GPU), `src/sparse_cudss.rs` (a `LinearSolver` backed by NVIDIA cuDSS via raw FFI), and `src/bde.rs` (wiring both into a batched Newton loop, `solve_batch_block_diagonal_device_resident`). It's correctness-verified (~1e-11 agreement with an independent CPU solve, five isolation tests all confirming bit-identical values through the scatter/masking/CSR pipeline) but has two open findings:

1. It takes 1-2 more Newton iterations than a host-resident version of the same solve on identical data. Leading (unconfirmed) hypothesis: CubeCL's `ComputeClient` launches kernels on a stream we don't control, so cuDSS's raw FFI calls (on cuDSS's own default stream) needed a defensive whole-context `cudaDeviceSynchronize()` rather than stream-ordered sequencing — a much heavier barrier than necessary, and one that doesn't fully explain the drift.
2. Full benchmarks (case9241pegase, case1354pegase; batch 256/1024/4096) show the Jacobian assembly kernel itself is only ~100ms per batch, while cuDSS's own refactorize+solve of the stacked matrix is 70-85% of each ~150-180ms iteration. The CPU `rayon` `BatchSolver` beats every GPU configuration tried by 50-90x at every batch size, including a projected 35,040-scenario year-long QSTS batch.

The user asked to design a plan to try **cuda-oxide only** — a rustc-to-PTX codegen backend, as an alternative to CubeCL, specifically because owning the CUDA stream directly (instead of going through CubeCL's abstraction) could plausibly close finding #1. But finding #2 caps the *ceiling* of any such fix: even eliminating all kernel-launch/sync overhead only touches the ~15-30% of iteration time that isn't cuDSS's own factorization — Amdahl-bounded at roughly **1.15-1.4x**, not a multiple-x win. This plan is scoped and gated accordingly: bounded, isolated, biased toward stopping early, and explicit that success here does not change the CPU-vs-GPU recommendation already established.

Research done today (live GitHub/crates.io fetches, not memory) found cuda-oxide is **not a normal Cargo dependency**: it's installed via `cargo install --git https://github.com/NVlabs/cuda-oxide.git cargo-oxide`, then driven by `cargo oxide build/run` on a **pinned nightly toolchain** (its own `rust-toolchain.toml`) with `rust-src`/`rustc-dev`/`llvm-tools` and LLVM 21+. Verified on this box: only `stable-x86_64-unknown-linux-gnu` is installed via rustup, no `clang`/`llvm-config`/`llc` binaries exist (only the `libllvm18` runtime lib bindgen uses), and `nvcc` reports CUDA **13.0.88** against cuda-oxide's documented CUDA 12.x support. None of cuda-oxide's prerequisites exist here yet — standing them up is itself part of the time budget below, not a given.

Kernel-shape research: cuda-oxide's `#[kernel]`/`#[cuda_module]` syntax confirmed to support f64 arithmetic/transcendentals, indexed reads of any array/slice, `thread::index_1d()`, ordinary `if`/`else if`, and per-thread writes into a `DisjointSlice<T>` output — covering `assemble_kernel`'s actual shape. Two open GitHub issues (#397/#398) flag a gap in writes to *runtime-indexed local arrays*, which our kernel doesn't have (no local scratch arrays — only reads from device slices, one scalar write). cuda-oxide has its own self-contained host runtime (`cuda-core`: `CudaContext`/`CudaStream`/`DeviceBuffer<T>`) that does not reference `cudarc` in any doc/example; whether it exposes raw pointer/stream accessors (needed for cuDSS interop) is unconfirmed and must be spiked. `cudarc` 0.19.8 — already a resolved transitive dependency via `cubecl-cuda` in this repo's `Cargo.lock` — is a known-working fallback for exactly that layer if cuda-oxide's own types don't expose what's needed.

## Scope

- **In scope**: an isolated crate porting `gpu::assemble_kernel` to cuda-oxide, obtaining a raw device pointer + stream (from cuda-oxide and/or `cudarc`), binding to `sparse_cudss::CudssRealSystem::new_device_resident`, and re-running the exact benchmarks already measured (case9241pegase, case1354pegase; batch 256/1024/4096).
- **Out of scope**: removing CubeCL/wgpu from the production path, adopting cuda-oxide as a default feature, chasing the iteration-count drift as research in its own right beyond what Phase 0 below settles, anything touching AMD/rocSOLVER.
- **Not a goal**: beating CPU `BatchSolver`. Nothing here can — cuDSS's factorization, not assembly, is the bottleneck.

## Phase 0 — give `CudssRealSystem` an explicit stream (main crate, stable Rust, ships independently)

Do this **first**, regardless of cuda-oxide. `sparse_cudss.rs`'s bindings import list never imports `cudssSetStream` (verified: no call site references a stream anywhere in the file), so every cuDSS call runs on its own default stream — half of why `bde.rs` needs the blunt whole-context `cudaDeviceSynchronize()`.

- Add `cudssSetStream` to the existing bindings import list (already generated by the current `bindgen` invocation — no `build.rs` change needed).
- Give `CudssRealSystem` a way to bind an explicit caller-supplied `CUstream` (e.g. `set_stream(&mut self, stream: u64) -> Option<()>` calling `cudssSetStream`, or a `new_device_resident_with_stream` variant).
- Validate with **existing** test patterns: re-run `ffi_smoke_test::solves_simple_system_via_raw_ffi` and `tests/bde_test.rs::bde_device_resident_matches_independent` with an explicit stream (obtainable via `cudarc::driver::CudaStream::cu_stream()`, already resolved in `Cargo.lock`) bound to both cuDSS and the assembly kernel launch, with `device_synchronize()` **removed** in favor of a narrower `cudaStreamSynchronize` (or none, if same-stream ordering alone suffices).

*Exit criterion*: existing tests pass unchanged with `set_stream` added; narrowed sync reproduces the same ~1e-12 agreement.

*Why this matters standalone*: if this alone closes the iteration-count gap, **that is a complete answer with zero cuda-oxide dependency** — cuda-oxide's main selling point for this specific problem may already be available via `cudarc`. Check this before touching cuda-oxide at all.

## Isolation strategy for anything cuda-oxide-related

`gridoxide`'s root `Cargo.toml` has no `[workspace]` table (verified) — a standalone stable-Rust package. cuda-oxide's toolchain requirement is categorically incompatible with a Cargo feature flag (a flag can't change the toolchain a build uses). Use a **separate, non-member crate directory**, `experiments/cuda-oxide-assembly/`:

- Own `Cargo.toml` with an **explicit empty `[workspace]` table** (prevents Cargo's auto-discovery from treating it as part of the parent package — a known gotcha for nested crates under a non-workspace parent), own `rust-toolchain.toml`, own `Cargo.lock`.
- Depends on `gridoxide` as a path dependency (`{ path = "../..", features = ["cudss"] }`) to reuse `CudssRealSystem`, `JacobianPattern`, `BlockDiagonal` — all needed accessors are already `pub`.
- `build_csr_structure`/`csr_scatter_map` are `pub(crate)` (verified), invisible even via a path dependency. **Duplicate the ~20-line logic in the experiment crate** rather than promoting them to `pub` — matches the existing precedent of `sparse_pardiso.rs` keeping its own independent copy of the same logic, and keeps the main crate's public API surface unchanged for an experiment that may not survive.
- No new `build.rs`/bindgen needed for cuDSS itself — reuse `CudssRealSystem` via the path dependency.
- No CI entry ever touches this directory, matching the existing precedent that `pardiso`/`cudss` are both "not built or tested in CI."

## Phase B0 — toolchain bring-up, hard go/no-go gate (budget: half a day)

Before writing any kernel code:

1. `cargo install --git https://github.com/NVlabs/cuda-oxide.git cargo-oxide`.
2. Install the pinned nightly + `rust-src`/`rustc-dev`/`llvm-tools` + LLVM 21, scoped to `experiments/cuda-oxide-assembly/` only via its own `rust-toolchain.toml` — never touch the repo-root default toolchain (currently `stable-x86_64-unknown-linux-gnu`).
3. `cargo oxide build`/`run` on cuda-oxide's own trivial example (vecadd), unmodified, to confirm this box's CUDA 13.0.88 (vs. cuda-oxide's documented 12.x) doesn't break codegen/linking — the single biggest unknown this plan can't pre-verify.
4. Confirm it runs correctly against the actual A100 (driver 580.126.20) present.

*Stop condition*: if the toolchain doesn't install/build cleanly within the half-day budget, **stop here — do not extend the budget "just a little more."** A toolchain that doesn't stand up cleanly is itself the finding. Report it, park the experiment, and leave `plans/GPU_PLAN.md`'s existing verdict on cuda-oxide ("Watch," "revisit in 6-12 months") as-is. Given the CUDA-version mismatch and the complete absence of nightly/LLVM21 on this box today, budget for a real chance of stopping here with nothing built.

## Phase B1 — confirm the one known gap doesn't apply (budget: 1-2h, only if B0 passes)

Write a minimal kernel reproducing `assemble_kernel`'s actual shape (read several device slices at a computed index, branch on an integer with `if`/`else if`, write one scalar to a `DisjointSlice`) — deliberately smaller than the real port, to check GitHub issues #397/#398 (runtime-indexed *local array* writes) don't apply before investing further. Our kernel has no local scratch arrays, so this should pass, but "should" isn't "confirmed."

*Stop condition*: if this gap (or an adjacent one) applies, stop and report. Do not attempt a workaround architecture — that turns a bounded spike into open-ended adoption work.

## Phase B2 — port the assembly kernel (budget: half a day, only if B1 passes)

Mechanical transliteration, same posture `gpu.rs`'s own doc comment already describes for its CubeCL port of `JacobianPattern::fill`: port the eight `kind`-dispatch branches, the H/N/M/L formulas, the `scenario`/`e` index arithmetic, and the `scatter`/`active`/`identity_value` masking logic verbatim — no re-derivation, no incidental "improvements" while porting. Use `thread::index_1d()` for `ABSOLUTE_POS`, `DisjointSlice<f64>` for the output array.

Validate against the same CPU reference `gpu.rs`'s own tests use (`tests/data/network.json` fixture): single-scenario values vs. `JacobianPattern::fill_into`, and multi-scenario masked/scattered values vs. `BlockDiagonal::fill` reordered through the duplicated `csr_scatter_map` — i.e., recreate the equivalent of `gpu.rs`'s `device_resident_tests::scattered_gpu_output_matches_cpu_csr_ordered_values` and `multiscenario_masked_scattered_output_matches_cpu` in the experiment crate.

*Exit criterion*: ~1e-16 agreement, both single- and multi-scenario masked.

## Phase B3 — raw pointer + stream interop with cuDSS (budget: half a day, riskiest open unknown)

Not yet confirmed whether cuda-oxide's `DeviceBuffer<T>`/`CudaStream` expose raw accessors. Spike in isolation: allocate a `DeviceBuffer<f64>`, launch the B2 kernel into it, extract a raw pointer, hand it to `sparse_cudss::debug_read_f64` (the existing `#[cfg(test)]` raw-`cudaMemcpy` helper — same technique `gpu.rs`'s `raw_cuda_memcpy_matches_cubecl_readback` already uses for CubeCL) to confirm it's valid. Same for the stream: extract the raw handle, call `cudssSetStream` (from Phase 0) with it, confirm cuDSS accepts it.

**Fallback, not a blocker**: if cuda-oxide doesn't expose either, use `cudarc` for just that layer (`CudaSlice<T>::device_ptr()`, `CudaStream::cu_stream()`) — already a resolved dependency, doesn't conflict with cuda-oxide for kernel compilation.

*Stop condition*: if neither cuda-oxide nor the `cudarc` fallback can produce a pointer/stream cuDSS's FFI accepts, stop and report — a hard blocker independent of everything else.

## Phase B4 — wire the full loop (budget: 1 day, only if B0-B3 pass)

Reconstruct `solve_batch_block_diagonal_device_resident`'s shape in the experiment crate: host-side mismatch/rhs (unchanged rationale — 4-6% of iteration time per `plans/GPU_PLAN.md` §1, not worth a GPU kernel), B2 kernel assembly into the buffer `CudssRealSystem::new_device_resident` points at, `solve_device_resident`. Bind the Phase 0+B3 stream to both the kernel launch and `cudssSetStream`; replace the whole-context barrier with a per-stream sync (or none, if ordering alone suffices — test both). Validate exactly like `bde_device_resident_matches_independent` (same fixture, same mixed fast/slow/masked scenario shape, ~1e-6 tolerance) and **record iteration count per scenario** against the CPU reference — the specific number this spike is trying to move.

## Phase B5 — the benchmark (budget: half a day)

Re-run the exact measurements already taken: case9241pegase and case1354pegase, batch 256/1024/4096. Fill in:

| metric | CubeCL (measured) | cuda-oxide (to fill in) |
|---|---|---|
| assembly kernel only, batch 64 | ~100 ms | ? |
| full device-resident iteration | ~150-180 ms | ? |
| iteration-count drift vs. CPU-independent | +1 to +2 | ? |
| CPU `BatchSolver` baseline (same batches) | 50-90x faster than every GPU config | (unchanged; re-quote as context, not a target) |

Use the same ~1e-6 to ~1e-9 agreement checks already established as the correctness gate for any reported number.

## Stop conditions, restated

- **Stop after Phase 0** if the stream fix alone resolves the drift — zero cuda-oxide dependency needed.
- **Stop after Phase B0** if the toolchain doesn't stand up in half a day (most likely stopping point given the CUDA-version mismatch and missing nightly/LLVM21).
- **Stop after Phase B1** if the local-array gap applies to this kernel's actual shape.
- **Stop after Phase B3** if no pointer/stream combination interops with cuDSS.
- **Continue past B5 to a fuller port only if**: full-iteration time improves by at least ~1.15x **and** the iteration-count drift closes/shrinks **and** nothing in B0-B4 revealed a correctness/stability gap. Anything less is "documented, not adopted" — park it in `experiments/`, note results, do not carry forward into production `gpu`/`cudss` features.

## Risk table

| Risk | Severity | Mitigation |
|---|---|---|
| Toolchain isolation leaks into main crate's build | Medium | Non-member crate, explicit empty `[workspace]`, no CI entry, scoped `rust-toolchain.toml` |
| CUDA 13.0.88 here vs. cuda-oxide's documented 12.x | **High** — most likely single blocker | Phase B0's hard half-day gate; no workaround attempted beyond budget |
| cuda-oxide alpha/API churn (created 2026-04-22, 38 open issues, main ahead of last tag) | Medium-High | Pin to a specific commit/tag for the spike; treat breakage as a maturity data point |
| Runtime-indexed local array write gap (#397/#398) | Low for this kernel, unconfirmed | Phase B1 spikes the exact shape first |
| cuda-oxide's own buffer/stream types lack raw accessors | Medium | `cudarc` fallback, already a resolved transitive dependency |
| NVIDIA-only lock-in | Medium, accepted for this spike | Doesn't touch the CubeCL path (`plans/GPU_PLAN.md` §5's AMD story stays intact); stays isolated in `experiments/` |
| Even total success only buys a bounded win | **High — the central framing risk** | Stated up front: ~1.15-1.4x ceiling, not a multiple-x win; every exit criterion calibrated to that, not to beating CPU |
| Spike time overruns into open-ended adoption work | Medium | Explicit per-phase budgets and stop conditions favoring "report and stop" |

## Recommendation

1. **Do Phase 0 first, today, on stable Rust.** No toolchain cost, may on its own answer the stream-identity question.
2. **Gate everything cuda-oxide-related on Phase B0.** Given the verified CUDA-version mismatch and complete absence of nightly/LLVM21 here, budget for a real chance of stopping after half a day with nothing built — a legitimate, cheap outcome.
3. **If B0-B5 all clear, judge against the ~1.15-1.4x ceiling, not against the 50-90x-the-other-direction CPU number.** A clean ~1.2x win on the non-cuDSS fraction is a legitimate "land it" result for this narrow question — it does not change the CPU-first recommendation for gridoxide's actual batch workloads.
4. **Any result here is "documented," not "adopted,"** until a separate decision (outside this plan's scope) addresses the `pub(crate)`→`pub` API question and CI implications of graduating anything out of `experiments/`.

## Verification

- Phase 0: `cargo test --features cudss` (existing suite, unchanged) plus the modified `bde_device_resident_matches_independent` run with an explicit stream and narrowed sync.
- B1-B4: correctness checks as specified per phase, all against `tests/data/network.json`, run via `cargo oxide build`/`run` inside `experiments/cuda-oxide-assembly/` (never `cargo test` from the repo root — that must stay untouched).
- B5: the benchmark table above, filled in from real runs on this A100, using the same case files (`case9241pegase.json`, `case1354pegase.json`) and batch sizes already used this session.

### Critical files

- `src/gpu.rs` — kernel to port, `device_resident_tests` module to mirror
- `src/sparse_cudss.rs` — `CudssRealSystem`, stream/pointer integration points, `debug_read_f64`
- `src/bde.rs` — `solve_batch_block_diagonal_device_resident`, the loop shape to reconstruct
- `Cargo.toml` — confirms no `[workspace]` table; `cudss`/`gpu` feature definitions
- `tests/bde_test.rs` — `bde_device_resident_matches_independent`, the validation pattern to reuse
- `plans/GPU_PLAN.md` — style/tone precedent, §1 (Amdahl baseline), §5 (AMD portability story this spike must not disturb)
