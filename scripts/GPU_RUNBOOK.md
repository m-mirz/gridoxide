# GPU session runbook

Ordered checklist for a **rented, metered** GPU box. Everything that could be
verified without a GPU already has been, so this session should be mostly
mechanical: two `bindgen` shims, an f64 dtype switch, and one honest benchmark.

Budget roughly **2–4 hours** for the NVIDIA path if nothing surprises you.
`plans/GPU_PLAN.md` is the design document; this is the operational one.

---

## Before you start the clock

Read this section on your own machine, not theirs.

**Pick an FP64-capable part.** A100, H100, V100 or MI300X. A 4090 at $0.30/hr
has 1/64-rate FP64 (~1.3 TFLOPS) and would produce numbers that mean nothing
for this workload — see `plans/GPU_PLAN.md` §5.

**Pick a `devel` image, not a runtime one.** `bindgen` compiles against real
vendor headers. A PyTorch-runtime container has `libcudart` but no
`cuda_runtime.h`, and you will discover this ten minutes in. On RunPod/Docker,
something like `nvidia/cuda:12.x.x-devel-ubuntu22.04`.

**cuDSS is a separate download** from the CUDA toolkit
(<https://developer.nvidia.com/cudss>). Grab the link beforehand. Check its
licence terms before building anything redistributable — the repo's precedent
is `vendor/suitesparse/PROVENANCE.md` and the `pardiso` feature's
link-a-system-install posture.

**Have the branch pushed.** You do not want to be resolving a rebase on a
metered box.

---

## Phase 0 — Environment (~15 min)

```bash
git clone <repo> && cd gridoxide
./scripts/gpu-setup.sh --install
```

It verifies vendor, FP64 class, Rust, `libclang`, the toolkit *headers*, and
cuDSS/rocSOLVER, then tells you precisely what is missing. Do not proceed until
it reports `Environment is ready.`

**Sanity-check the box with the CPU suite** before trusting anything it says
about GPUs:

```bash
cargo test                 # expect 161 passed, 0 failed
cargo test --features klu  # expect 189 passed, 0 failed
```

If those fail, the machine or toolchain is broken and nothing downstream is
meaningful.

---

## Phase 1 — Re-measure the CPU baseline **on this host** (~15 min)

Do this **first**, before any GPU work, and do not skip it.

```bash
maturin develop --release --features python,klu
python3 scripts/bench/matpower_to_pgm.py case9241pegase   # if not cached
python3 scripts/bench/bench_batch.py <case9241pegase.json> klu 256
```

The number recorded in `scripts/bench/README.md` (~127 solves/s, 28.0 ms/solve
single-threaded) comes from a **thermally throttled laptop APU** and is a lower
bound, not a fair baseline. A rented A100 box has an EPYC or Xeon host that will
beat it substantially. Quoting a GPU speedup against the laptop number would
inflate the result by an unknown factor and is the single easiest way to publish
something wrong.

Write the host's own numbers down. That is what Phase 4's speedup is measured
against.

---

## Phase 2 — Kernel in f64 (~30 min)

The kernel is already written and validated to ~2.7 ULP of f32 on an integrated
GPU (`scripts/bench/README.md` §4e). This phase only changes precision and
backend.

1. **Switch the CubeCL runtime.** `src/gpu.rs` ends with:

   ```rust
   pub type DefaultRuntime = cubecl::wgpu::WgpuRuntime;
   ```

   Point this at `cubecl::cuda::CudaRuntime` (feature `cubecl/cuda`) or
   `cubecl::hip::HipRuntime` (feature `cubecl/hip`). `GpuAssembler<R: Runtime>`
   is already generic, so nothing else needs touching.

2. **Switch the dtype.** `f32` → `f64` in `assemble_kernel` and in
   `GpuAssembler`'s upload/readback. The formulas are unchanged.

3. **Run the real Phase 2 exit criterion:**

   ```bash
   cargo test --features gpu --test gpu_assembly_test
   cargo run --release --features gpu --example gpu_assembly_check -- \
       <case9241pegase.json> 64
   ```

   **Expected in f64:** operand-scaled error drops from ~3e-7 to ~1e-16, and
   `gpu vs cpu(f64)` result-scaled collapses from ~1e-2 to near zero. That ~1e-2
   was pure f32 cancellation in `G·cos + B·sin`; in f64 it should vanish. If it
   does not, the problem is real and is in the kernel, not in precision.

*This is the point at which `plans/GPU_PLAN.md` Phase 2 is genuinely complete.*

---

## Phase 3 — The vendor FFI shim (~1–2 h, the actual new work)

Everything else in Phase 3 is done and validated on CPU: block-diagonal
assembly is **bit-exact** against independent solves up to 83,664 stacked
unknowns (`scripts/bench/README.md` §4d), masking preserves the sparsity
pattern, and the offset arithmetic is tested. What remains is one trait impl.

1. **`build.rs`**: add a `cudss` (or `rocsolver`) branch. Follow the existing
   two precedents in that file exactly — the `pardiso` one is the closer match,
   since it links a system install and vendors nothing. Discover via
   `CUDSS_ROOT` / `ROCM_PATH`.

2. **Implement `solver::LinearSolver`** for the device solver:

   ```rust
   fn new(n, entries) -> Option<Self>          // symbolic analysis, ONCE
   fn factor_and_solve_values(&mut self, values, rhs) -> Option<Vec<f64>>
   ```

   The split maps directly onto both APIs: `cudssExecute(ANALYSIS)` /
   `rocsolver_dcsrrf_analysis` in `new`, then refactor + solve per iteration.
   `factor_and_solve_values` already takes values only, so nothing has to
   rebuild triplets.

3. **Feed it the block-diagonal matrix.** `bde::solve_batch_block_diagonal<S>`
   is generic over `LinearSolver`; instantiate it with the GPU solver. Nothing
   in `src/bde.rs` should need to change — if it does, that is a finding worth
   recording.

4. **Correctness first, speed second:**

   ```bash
   cargo test --features gpu,cudss --test bde_test
   cargo run --release --features gpu,cudss --example bde_check -- \
       <case1354pegase.json> 16
   ```

   `bde_check` already asserts per-scenario agreement with independent CPU
   solves. On CPU backends this comes back bit-exact; a GPU solver will differ
   in the last bits (different ordering and pivoting), so expect agreement at
   ~1e-9 rather than 0. Anything looser than ~1e-6 means something is wrong.

**Watch for:** device-resident data. The win only materialises if the Jacobian
is built on-device and never round-trips per iteration
(`plans/GPU_PLAN.md` §6, Phase 3). A first implementation that uploads values
each iteration is fine for correctness but will not show a speedup — measure it,
but do not conclude anything from it.

---

## Phase 4 — Benchmark honestly (~30 min)

```bash
python3 scripts/bench/bench_batch.py <case9241pegase.json> klu 256   # host CPU
# then the GPU equivalent at the same batch size
```

Phase 3's exit criterion in `plans/GPU_PLAN.md` is **≥10× over the Phase 0
multithreaded CPU baseline at batch ≥256 on case9241pegase**, with voltages
matching `klu` to the accuracy suite's tolerances.

Report against **this host's** CPU number from Phase 1, at the same batch size,
same case, same tolerance. Note the GPU model and whether data stayed
device-resident. If the CPU baseline was measured with fewer threads than the
box has, say so.

---

## Phase 5 — Before you shut the box down

- [ ] Copy every raw number out — you cannot re-run it later for free.
- [ ] `git push` the branch.
- [ ] Record the exact GPU model, driver, CUDA/ROCm version, and cuDSS/rocSOLVER
      version alongside the results. Without these the numbers are not
      reproducible.
- [ ] Note anything the runbook got wrong, so the next session is cheaper.

---

## If you only have an hour

Do Phase 0, Phase 1, and Phase 2. That completes `plans/GPU_PLAN.md` Phase 2
against its real f64 criterion and gives you a trustworthy CPU baseline — both
permanent results. Phase 3 is the part that genuinely needs unhurried time, and
it is much cheaper to start it with a verified environment already behind you.
