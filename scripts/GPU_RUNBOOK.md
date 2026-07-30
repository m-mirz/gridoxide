# GPU session runbook

Ordered checklist for a **rented, metered** GPU box. Everything that could be
verified without a GPU already has been, so this session should be mostly
mechanical: two `bindgen` shims, an f64 dtype switch, and one honest benchmark.

Budget roughly **2–4 hours** for the NVIDIA path if nothing surprises you.
`plans/GPU_PLAN.md` is the design document; this is the operational one.

> **Sessions 1–2 are done.** Phases 0–3 below were executed; the result was a
> 91x *regression* against the CPU baseline. **If you are starting a session
> now, go straight to [Session 3](#session-3--the-batched-rewrite) at the
> bottom** — Phases 1–4 here are kept as the record of what was measured and
> how, not as the next thing to do.

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
cargo test                 # expect 170 passed, 0 failed
cargo test --features klu  # expect 198 passed, 0 failed
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

---

## Session 3 — the batched rewrite

### Read these first — they are where the answers are

This file is the *procedure*. The **reasoning** lives in four places, and when a
step below fails, the fix is almost always in one of them rather than in this
checklist. Read them before starting the clock, not after something breaks:

| Read | For |
|---|---|
| [`plans/GPU_PLAN.md`](../plans/GPU_PLAN.md) — the amendment at the top | Why the last attempt lost by 91x, and which three claims in the original plan were wrong. Everything else here follows from it. |
| [`src/sparse_cudss.rs`](../src/sparse_cudss.rs) — `CudssBatchedSystem`'s doc comment | What the uniform batch API does differently, why the pointer arrays are members and not temporaries, and the "verify before trusting" note on `cudssMatrixCreateBatchCsr`'s signature. **Step 2 fails here.** |
| [`src/bde.rs`](../src/bde.rs) — `solve_batch_block_diagonal_batched_device`'s doc comment | The batched-vs-stacked comparison table, and why the loop keeps its convergence bookkeeping on the host. |
| [`src/device_layout.rs`](../src/device_layout.rs) — module doc | Every layout decision the kernels depend on, with tests that already pass on CPU. **If a Step 3 test fails, the bug is in the kernel, not here.** |

Also worth knowing before Step 1: [`cuda/gridoxide_kernels.cu`](../cuda/gridoxide_kernels.cu)'s
header comment explains the layout conventions every kernel shares
(`scenario * n_buses + bus`, `scenario * blk + r`, `scenario * nnz + scatter[e]`),
and each kernel names the Rust function it was transliterated from.

**What happened in sessions 1–2.** Phases 0–4 above were executed. The
device-resident path works and is correct (voltages agree with independent CPU
solves to ~1e-12), and it is **91x slower** than the CPU:

| case | batch | GPU device-resident | CPU 30 threads | CPU 1 thread |
|---|---|---|---|---|
| case1354pegase | 4,096 | 52.7 solves/s | 4,825 solves/s | 165.2 solves/s |
| case9241pegase | 256 | 37.0 s | 285.9 solves/s | 19.1 s |

**Why**, per the arithmetic in `plans/GPU_PLAN.md`'s amendment: the serial host
mismatch loop is ~1.5% of an iteration, the Jacobian upload another ~1.5%, and
~95% is inside cuDSS refactorizing a single 10-million-row *stacked* matrix.
Block-diagonal embedding threw away the structure the solver needed.

**What changed in the code, before this session.** All written blind, on a
machine with no NVIDIA GPU, and as much of it as possible verified by ordinary
`cargo test`:

- `cuda/gridoxide_kernels.cu` — five hand-written CUDA kernels replacing the
  CubeCL assembler: assembly, power injections, mismatch + convergence-norm
  reduction, masked-rhs zeroing, Newton update.
- `src/device_layout.rs` — **not feature-gated**, 9 unit tests that run
  anywhere. Every layout decision the kernels depend on (Y-bus CSR, ZIP
  coefficient flattening, unknown index maps, scenario-major strides, the CSR
  scatter map) is checked here against the CPU implementation.
- `src/sparse_cudss.rs::CudssBatchedSystem` — cuDSS's uniform batch API.
- `src/bde.rs::solve_batch_block_diagonal_batched_device` — the fully
  device-resident loop. `solve_batch_block_diagonal_device_resident` is kept as
  the A/B control.
- `examples/bde_profile.rs` — per-phase CUDA-event timers.

### Before you start the clock

- **cuDSS must be >= 0.4.0** — that is when the batch API landed. Check first;
  everything else depends on it.
- `nvcc` is now a **build-time** requirement (it was not before — CubeCL
  JIT-compiled at runtime). A `-devel` image, not a runtime one, as always.
- Set `CUDA_ARCH` to match the card: `sm_80` A100 (the default), `sm_90` H100,
  `sm_70` V100, `sm_89` L40S. A mismatch is a *runtime* "no kernel image is
  available" error, not a build failure.

### Where the risk actually is

All of the code above was written on a machine with **no NVIDIA GPU and no
`nvcc`**, so it has never been compiled, let alone run. That shapes where the
failures will be, and they are not evenly distributed:

| Risk | Where it lands | What it looks like |
|---|---|---|
| `nvcc` flags, toolkit discovery, `bindgen` allowlist gaps | Step 1 | build errors in `build.rs`'s `mod cuda` |
| Kernel launch signature vs. the `extern "C"` declarations in `src/gpu.rs` | Step 1 | link error, or `error code 98/209` at runtime |
| `cudssMatrixCreateBatchCsr` parameter order / host-vs-device pointer arrays | Step 2 | **wrong numbers, not an error** — most args are `void*` |
| Kernel indexing or stride typos | Step 3 | a specific test fails by orders of magnitude, not slightly |
| The hypothesis itself being wrong | Step 4 | everything passes, cuDSS phase doesn't move |

The kernels were syntax-checked with `g++` against a shim header, so expect
*semantic* problems there rather than parse errors. Two deliberate safety nets:
`src/device_layout.rs` has 9 tests covering every layout decision the kernels
depend on and they already pass on CPU, so a Step 3 failure is in the kernel,
not in the flattening; and `batched_ffi_smoke_test` (Step 2) exists purely to
turn the silent-garbage row above into a loud failure.

### Ordered steps, with a checkpoint after each

**Step 0 — environment and the bar (~30 min).**

The work described above is on branch **`gpu-nvidia`**. The benchmark cases are
in a git submodule and the PGM JSON is *generated*, not committed
(`scripts/bench/.case-cache/` is gitignored) — so a fresh clone has neither
until you do this:

```bash
git clone --recurse-submodules https://github.com/m-mirz/gridoxide.git
cd gridoxide && git checkout gpu-nvidia
# If you cloned without --recurse-submodules:
git submodule update --init --recursive

./scripts/gpu-setup.sh --install
cargo test                 # expect 170 passed, 0 failed
cargo test --features klu  # expect 198 passed, 0 failed

# The Python extension, which bench_batch.py drives. Install numpy/scipy with
# plain pip rather than `pip install -e '.[matpower]'`: that would re-run the
# maturin build without --features python,klu and replace what you just built.
maturin develop --release --features python,klu
pip install numpy scipy

# Generate the case. Takes the MATPOWER .m from the submodule, writes PGM JSON.
mkdir -p scripts/bench/.case-cache
python3 scripts/bench/matpower_to_pgm.py \
    tests/data/benchmark-grids/matpower/case9241pegase.m \
    scripts/bench/.case-cache/case9241pegase.json

python3 scripts/bench/bench_batch.py scripts/bench/.case-cache/case9241pegase.json klu 256
```

Record **this host's** CPU number. Everything later is measured against it.
Do not reuse a number from a previous box — the one in
`scripts/bench/README.md` came from a thermally throttled laptop APU and is a
lower bound, not a baseline.

Every `<case9241pegase.json>` below means
`scripts/bench/.case-cache/case9241pegase.json`.

**Step 1 — does anything compile and run? (~45 min).**

```bash
cargo test --features gpu,cudss --test gpu_assembly_test
```

This is where `nvcc`, the `bindgen` allowlists and the kernel launch
signatures all get exercised for the first time. The kernels were syntax-checked
with `g++` against a shim header (no `nvcc` on the dev machine), so expect
argument-order and toolkit-discovery problems here rather than logic errors.

*Checkpoint:* all three assembly/injection tests pass.

**Step 2 — is the batch API wired up right? (~20 min).**

```bash
cargo test --features gpu,cudss --lib sparse_cudss::batched_ffi_smoke_test
```

Run this **before** anything gridoxide-specific. `cudssMatrixCreateBatchCsr`
takes most of its arguments as `void*`/`void**`, so a wrong parameter order
compiles cleanly and returns silent garbage; this solves a known two-system
batch off the raw bindings and will fail loudly instead. If it fails, diff the
call against `OUT_DIR/cudss_bindings.rs` and the cuDSS install's own batched
sample before touching anything else.

*Checkpoint:* the smoke test passes.

**Step 3 — correctness (~30 min).**

```bash
cargo test --features gpu,cudss --test bde_test
cargo run --release --features gpu,cudss --example bde_check -- \
    <case9241pegase.json> 256
```

`bde_batched_and_stacked_device_paths_agree` is the informative one: the two
device paths share the assembly kernel but differ in everything else, so a
disagreement isolates the batched wrapper from anything they have in common.
`bde_check` must print `PASS`.

*Checkpoint:* voltages agree with independent CPU solves to ~1e-9.

**Step 4 — the moment of truth (~30 min).**

```bash
cargo run --release --features gpu,cudss --example bde_profile -- \
    <case9241pegase.json> 256
```

This prints the CPU baseline, both GPU paths, the speedup ratios, and the
per-phase breakdown. **The number that decides everything is whether the cuDSS
phase dropped** relative to the stacked control. If it did not, the hypothesis
was wrong and nothing after this step matters — stop, copy the numbers out, and
write up the negative result.

*Checkpoint:* `batched` beats `stacked` by a large factor, or the session ends
here with a real answer.

**Step 5 — sweeps (~40 min), only if step 4 was positive.**

- Batch size: 64 / 256 / 1024 / 4096. The GPU only wins by having enough
  independent blocks, and the curve is the finding.
- Chunking: `bde_profile` prints free device memory. Above the ceiling, split
  the batch into passes and compare against the CPU at the same *total*
  scenario count, never per chunk.
- cuDSS config, measured one at a time as deltas, not assumed:
  `CUDSS_CONFIG_PIVOT_TYPE = NONE` (power-flow Jacobians rarely need pivoting,
  and skipping the pivot search matters disproportionately on a GPU),
  reordering algorithm, and whether
  `CudssBatchedSystem::set_deterministic(true)` costs anything (it is off by
  default here, unlike the stacked path).
- Re-check the iteration-count drift documented on
  `solve_batch_block_diagonal_device_resident`: does it persist under the
  batched path? Record the answer either way — it has been open since `ff92b66`.

**Step 6 — before you shut the box down.**

- [ ] Copy every raw number out; you cannot re-run it later for free.
- [ ] `git push`.
- [ ] Record GPU model, driver, CUDA version, cuDSS version, `CUDA_ARCH`.
- [ ] Update `plans/GPU_PLAN.md`'s amendment with what was actually measured,
      including if the answer is "still slower than 30 EPYC cores" — that is a
      publishable result and closes the track honestly.

### Honest expectation

To merely *match* the CPU at case1354pegase/4,096, the A100 must do ~4,096
refactor+solves in ~77 ms — roughly 10x a single EPYC core on tiny-front sparse
LU. Plausible; not guaranteed. Parity to a few x is the realistic range, and the
declared success bar is "beats the 30-thread `BatchSolver` at all, at some batch
size". SABLE's 253x and Fraunhofer's 100x are over *pandapower* and say nothing
about clearing a 30-thread KLU batch solver.
