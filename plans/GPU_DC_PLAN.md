# DC power flow and DC N-1 on NVIDIA GPUs

Status: **plan**, 2026-10-30, against `924737c`. Nothing built yet.

The goal: DC power flow and DC N-1 for gridoxide-gpu, beside the AC batch of
[GPU_BATCH_PLAN.md](GPU_BATCH_PLAN.md), measured in grid-bench's DC tabs against
ToOp (Elia's JAX DC solver, the one GPU tool there) and the CPU tools, with
angles **and** flows for every outage, exact to the oracle's 1e-3 MW. And to
find out first whether the CPU, which already has every piece, makes the GPU
unnecessary.

## What we know

### gridoxide's DC today (CPU)

- `linear::dc_power_flow` and `DcSolver`: MATPOWER's `makeBdc` model (b = 1/(x·tap),
  phase shifts as injections, Gs as load since #28), one reference per island.
  Python: `PowerFlowModel.from_matpower(..., method="dc")`, `solve()`,
  `branch_flow_p()`.
- `linear::DcBatchSolver`: many injection scenarios, **one factorization**: each
  scenario is the base solution plus `DcSensitivity::response` to its injection
  change, exact (`tests/dc_batch_test.rs`, 1e-12 against a loop).
- `linear::DcSensitivity`: PTDF and LODF rows and columns from one factorization
  of B_NN, dense matrices on request; `outage_flows` (N-1, `f + LODF[:, l]·f_l`,
  one triangular solve) and `multi_outage_flows` (N-k by Woodbury, one k × k
  solve on top of k triangular solves); `is_breaking_set` for outages that
  island. Python: `outage_flows`, `multi_outage_flows`, `is_breaking_set`,
  `ptdf_*`, `lodf_column`.
- Missing for grid-bench: **angles after an outage** (only flows are returned),
  and an N-1 call over a list (Python loops one outage per call).
- Measured today (A100 machine, one thread, from Python): `solve()` 0.1 ms on
  case1354pegase, 1.1 ms on case9241pegase; every connected outage through
  `outage_flows` 174 ms (1430 outages, 0.12 ms each) and 15.0 s (14384, 1.04 ms
  each), a 16049-float Python list per call included.

### grid-bench's DC problems

`adapters/dc_adapter.py`: MATPOWER's DC model; `solution` returns angles (degrees,
by bus number) and from-end flows (MW, by branch row), both graded by `oracle.dc`
(the nodal balance with the tool's angles, and the flows against the oracle's
own solution, both to 1e-3 MW). A tool that computes no angles is graded on
flows alone (ToOp). DC N-1 (`DcContingencyAdapter`): the base case and every
outage in one timed call, by a new solve or from sensitivities (both exact),
angles of every outage included. Cases: `#dc` on case14 to case9241pegase and
case3120sp; `#dcn1` (200 outages) and `#dcn1-full` (every connected outage:
1430, 3804, 14384) on the three PEGASE cases, all in default groups. The DC tab
is not a GPU tab: ToOp's results sit with the CPU tools' (`results-docker/`),
measured on the same machine.

Stored results (this A100 machine, AMD EPYC 7J13, 30 cores), ms per call:

| case | ToOp (GPU, flows only) | lightsim2grid LODF, 1 thr | lightsim2grid solves, 1 thr |
|---|---|---|---|
| case9241pegase#dc | 49 | | 1.0 |
| case1354pegase#dcn1-full | 55 | 77 | 308 |
| case2869pegase#dcn1-full | 112 | 651 | 2 458 |
| case9241pegase#dcn1-full | **893** | 8 299 | 39 333 |

(pypowsybl's DC N-1 on 30 threads takes 1.0 and 6.5 s on the first two and
runs out of memory on case9241pegase; MATPOWER's loop of `rundcpf` 103 and
315 s.)

ToOp's time is a fixed ~49 ms per call (JAX's dispatch and the copy to the
host) until case9241pegase#dcn1-full, where it is the 1.85 GB of flows. It is
exact on the PEGASE cases; it leaves shunt conductance out (case2869pegase off by
2.45 MW). In single precision it runs 478 against 891 ms but misses the 1e-3 MW
tolerance.

### The arithmetic, and why a single DC solve is not a GPU problem

A single DC power flow is one sparse solve with B_NN (N − 1 unknowns): 1 ms on
case9241pegase on one CPU core, against ToOp's 49 ms floor. No GPU path will
win `#dc`; it is listed below only as the base of the batch.

Full DC N-1 is one factorization and K right-hand sides. For outage l (branch
from f to t, susceptance b_l, base flow p_l), with Θ[:, l] = B_NN⁻¹ (e_f − e_t)
and ψ[:, l] = B_f Θ[:, l] (B_f has two entries per row, so ψ[m, l] =
b_m (Θ[from_m, l] − Θ[to_m, l])), d_l = b_l (Θ[f, l] − Θ[t, l]):

    c_l = p_l / (1 − d_l)
    flows after l:  p + c_l · ψ[:, l],  then branch l's own set to 0
    angles after l: θ + c_l · Θ[:, l]

(the outage as the transfer c_l from f to t that the removed branch would
carry; the reference bus's row of Θ is zero, so its angle stays.)

So the work is one multi-RHS triangular solve (N × K), one gather kernel for ψ
(L × K), and one scaling kernel; d_l is read off ψ. Sizes on case9241pegase,
K = 14384: Θ 1.06 GB, flows 1.85 GB, together 2.9 GB to copy back. The solve is
about 2·nnz(L+U)·K, a few GFLOP; the copy, at 20-25 GB/s, is 0.12-0.15 s and
likely the floor, as it is ToOp's. The same machinery solves DC sweeps
(`DcBatchSolver`'s shape: K injection changes as right-hand sides), which
grid-bench does not have but gridoxide's API does.

The CPU does the same per outage: one triangular solve (tens of µs on
case9241pegase) and the scaling, independent across outages. On 30 threads
14384 outages are well under a second of solves; filling 2.9 GB of output is
again the floor. **Whether the GPU beats the threaded CPU here is open**, and
phase 1 answers it before any device code.

## Design

- **CPU first, in gridoxide-core.** `DcSensitivity::contingencies(&self, base:
  &DcSolution, outages: &[usize]) -> DcContingencies`: angles and flows of every
  outage into two dense buffers (outage-major), on rayon's pool, an islanding
  outage reported, not solved (`None` in a status vector). Exact by
  construction, tested against `outage_flows` and against a re-solve of each
  outaged network.
- **The device, in gridoxide-gpu**, a module beside the AC batch (`src/dc.rs`,
  kernels in `cuda/kernels.cu`): `GpuDcSolver`, a session per network, holding
  B_NN's cuDSS analysis and factorization (one system, not a uniform batch:
  the matrix is the same for every outage) and buffers for `capacity` columns.
  `solve_contingencies(outages)`: the unit right-hand sides built on the
  device, one cuDSS solve with K columns (a dense `n × K` descriptor), the ψ
  gather and the scaling kernels, outputs left on the device until read.
  `solve_batch(injection changes)`: the same solve with ΔP columns, for sweeps.
  No mask and no loop: DC has no iterations.
- **Readout**: as `voltages()` for AC, two float64 buffers (`bytes`), copied
  from pinned host memory, chunked to the capacity so a 40 GB device and the
  host hold any K.
- **Python**: on `PowerFlowModel` (DC method), `solve_contingencies` for a DC
  model returning the two buffers (CPU, phase 1); `GpuDcSolver(model,
  capacity)` with the same calls (phase 2), behind the `gpu` feature.
- **N-k later**: Woodbury per scenario is a k × k solve after k columns of Θ,
  batched small dense solves on the device; not in the first version.

## Phases

0. **Check the model on grid-bench's DC cases (CPU).** Before anything new:
   gridoxide's existing `solve()` against `oracle.dc` on every `#dc` case, as a
   new grid-bench tool `gridoxide` (DC adapters only, from the gridoxide_gpu
   image, which has the CPU path too). Watch phase shifters (case2848rte by
   name, which ToOp passes) and the reference angle. **Gate:** every `#dc` case
   accepted; a residual is a model difference to fix in gridoxide first.
1. **Threaded CPU DC N-1 with angles** (gridoxide-core, Python, grid-bench's
   `gridoxide` DC N-1 adapter). Measure `#dcn1-full` at 1 thread and 30, the
   output copy apart from the solves. **Gate:** if 30 threads are within about
   1.5x of ToOp on case9241pegase#dcn1-full (that is, under ~1.3 s) and the
   output fill is most of it, the GPU can win little beyond the copy: stop
   here, report, and keep the GPU plan for DC sweeps only if someone needs
   them.
2. **The device path** (`GpuDcSolver`): cuDSS multi-RHS solve, ψ and scaling
   kernels, each checked against phase 1 to 1e-12 on case1354pegase and
   case9241pegase, islanding outages and phase shifters included; device
   memory per column and the chunk size measured on the A100.
3. **grid-bench** `gridoxide_gpu` DC N-1 adapter (and `#dc`, for completeness,
   expected to lose to every CPU tool), against ToOp and lightsim2grid in the
   DC tab, angles graded.
4. **DC sweeps** (`solve_batch`), only if phase 3 shows the device pays: the
   same kernels with ΔP columns, against `DcBatchSolver` on 30 threads.

## Targets

- **Correct:** every `#dc` and `#dcn1(-full)` case accepted, angles and flows,
  including the cases ToOp misses on shunt conductance (case300,
  case2869pegase).
- **Fast:** under ToOp's 893 ms on case9241pegase#dcn1-full while returning the
  angles it does not (2.9 GB against its 1.85 GB). The CPU (phase 1) may be what
  meets this; that is a result, not a failure.

## Risks

- **The copy is the floor.** 2.9 GB of output on case9241pegase#dcn1-full; at
  PCIe speed that is 0.12-0.15 s from pinned memory, more from pageable, plus
  the host-side `bytes` and numpy views. The GPU's solve may be the small part.
- **cuDSS's multi-RHS solve** is untested here: its time for K = 14384 columns
  (and whether it parallelizes across columns or runs them one by one) decides
  phase 2. Probe it first, with probe C's harness, on B_NN of case9241pegase.
- **Memory:** Θ and the flows for all outages of case9241pegase are 2.9 GB on
  the device, fine on 40 GB; larger cases chunk by `capacity`.
- **Precision:** FP64 throughout; ToOp's FP32 misses the tolerance, and cuDSS's
  single precision would too.
- **Not a GPU tab:** grid-bench's DC tab compares every tool on one machine;
  the A100 machine's CPU runs the CPU tools, so the numbers stay comparable,
  but a GPU tool listed there must say so (`dependencies.gpu`), as ToOp does.

## Open questions

- A CPU tool `gridoxide` in grid-bench (phases 0-1) is new: gridoxide has no
  grid-bench adapter on the CPU, for AC or DC. Adding its AC power flow, batch
  and N-1 at the same time would put gridoxide's CPU `BatchSolver` beside the
  GPU tabs' baseline, which today is lightsim2grid alone.
- Whether DC sweeps (phase 4) are worth a grid-bench problem of their own.
