# cuDSS uniform-batch and mask probe

Phase 0 of [`plans/GPU_BATCH_PLAN.md`](../../plans/GPU_BATCH_PLAN.md): before any GPU code is
written for gridoxide, measure on real hardware what the design depends on.

`probe.cpp` runs four experiments on real Newton-Raphson Jacobians, written with CPU
reference solutions by [`examples/gpu_probe_dump.rs`](../../examples/gpu_probe_dump.rs):

| | question | decides |
|---|---|---|
| A | Does the uniform batch (one CSR matrix over a `B·nnz` value buffer, `CUDSS_CONFIG_UBATCH_SIZE = B`, one analysis) match the CPU, before and after a refactorization? | whether the batch shape works at all |
| B | Can `CUDSS_DATA_UBATCH_MASK` change between calls; does `REFACTORIZATION` honour it; are an unselected system's factors and solution left untouched? | per-scenario stopping |
| C | Does a masked refactorization and solve cost less in proportion to the systems selected (contiguous and strided selections)? | whether converged scenarios stop costing work |
| D | What does one step's mask update cost (B flags device to host, a sync, `cudssDataSet`)? | the price of adaptivity over a fixed step count |

Every cuDSS status is printed by name, never assumed; a wrong number makes the probe exit
non-zero. Lines starting `RESULT` are the machine-readable summary.

## Running it

On a machine with an NVIDIA GPU, the CUDA 12 toolkit and Rust:

```bash
git submodule update --init tests/data/benchmark-grids
CUDA_HOME=/usr/local/cuda probes/cudss_ubatch/run.sh          # cuDSS 0.8.0
CUDA_HOME=/usr/local/cuda probes/cudss_ubatch/run.sh 1.0.0    # and 1.0.0
```

`run.sh` downloads cuDSS from NVIDIA's redistributable archive (pinned by SHA-256, as
grid-bench's gpusim2grid image does), builds the probe with `g++`, writes dumps of 64
scenarios for case1354pegase, case2869pegase and case9241pegase, and runs B = 64, 256,
1024 and 2048 (systems tiled from the 64), with cuDSS's OpenMP threading layer on. Then
`explore.cpp`, the follow-up written after the first run: what the mask does to the systems
it leaves out (`explore <dump> <B> mask`) and the cost of a refactorization and solve
against B (`explore <dump> <B> time ...`). Last, it runs `crates/gridoxide-gpu`'s tests
with the `cuda` feature: the cuDSS backend's contract (masks, unfactored systems) and the
adaptive Newton loop on the device against the CPU stand-in, step for step. Results go to
`results/`.

A distribution's CUDA package (Ubuntu's `nvidia-cuda-toolkit`) keeps its headers in
`/usr/include`; `run.sh` then sets `CUDA_INCLUDE_DIR` itself.

## Results

Run on an A100-SXM4-40GB on 2026-10-29; `results/` holds the output, and
[the plan](../../plans/GPU_BATCH_PLAN.md#phase-0-on-an-a100) says what it means. In short:
(A) passes in both versions. (B) holds as documented in 1.0.0, but 0.8.0 drops the factors
of the systems a factorization leaves out and overwrites their solutions, so two of probe
B's lines fail there: that is the finding, not a broken probe. (C)
1.0.0's masked work scales with the selection; 0.8.0's has a fixed cost per step from
B = 16 on. (D) costs about 20 µs. And 0.8.0 computes NaNs when two handles work at once in
one process.

## Checking it without a GPU

`stub/` holds stand-in `cuda_runtime.h` and `library_types.h` with the declarations the
probe and cuDSS's headers use, so the probe can be syntax-checked against the real cuDSS
headers on a machine without CUDA:

```bash
g++ -std=c++17 -fsyntax-only -Wall -Wextra -Iprobes/cudss_ubatch/stub \
    -I<cudss>/include probes/cudss_ubatch/probe.cpp
```

The stub is valid C too, so gridoxide-gpu's bindgen runs against it and its cuDSS backend
can be type-checked the same way (from the repository root):

```bash
CUDSS_DIR=<cudss> CUDA_INCLUDE_DIR=$PWD/probes/cudss_ubatch/stub GRIDOXIDE_GPU_NO_NVCC=1 \
    cargo clippy -p gridoxide-gpu --features cuda --all-targets
```

(`GRIDOXIDE_GPU_NO_NVCC` skips compiling the crate's kernels, which needs nvcc; their
launchers' bindings still come from `cuda/kernels.h`, so their signatures are checked
too.)

The old `gpu-nvidia` branch's three FFI bugs were all signature mismatches that only a
GPU machine surfaced; these checks catch that class before any GPU time is spent. Both
pass against cuDSS 0.8.0 and 1.0.0. (The Rust one already caught a difference between
them: 1.0.0 turned `CUDSS_R_*` into macros over CUDA's `CUDA_R_*`, which bindgen cannot
expand, so the backend uses the `CUDA_R_*` names, valid in both.)
