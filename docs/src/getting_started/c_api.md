# C and C++ API

gridoxide exposes a C ABI so a C++ project can embed it. Enable the `capi`
feature; the artifacts are a generated `include/gridoxide.h`, a hand-written
header-only C++ wrapper `include/gridoxide.hpp`, and `libgridoxide.a` /
`libgridoxide.so`.

```cpp
#include "gridoxide.hpp"

auto options = gridoxide::PowerFlow::default_options();
options.s_base_va = 1e8;

auto pf = gridoxide::PowerFlow::from_pgm_file("network.json", options);
pf.solve();

const auto vm = pf.voltage_magnitude();
const auto [p, q] = pf.branch_flow();
```

Unlike `klu`, `pardiso`, `opf-highs` and `opf-ipopt`, this feature needs
**nothing installed** — it is pure Rust plus a build-time crate — so it is
built and tested in CI like `opf`.

## The mirror image of everything else here

Four of gridoxide's features consume a C header through `bindgen` so Rust can
call a library. This one runs `cbindgen` over `src/capi/` so a library can call
*us*. Same shape in `build.rs`, opposite direction.

## Scope

**Power flow.** AC Newton-Raphson, the two outer loops (Q limits and
[distributed slack](../powerflow/distributed_slack.md)), voltages, branch
flows, convergence detail and per-island status.

Not DC, sensitivity, state estimation, short circuit or OPF — all reachable
from [Python](./python.md) and from Rust. The ABI is shaped so they can be
added without breaking it: each would be another opaque handle plus
count-then-fill accessors.

One capability here has **no Python equivalent**: `branch_flow()` returns AC
branch flows. The Python binding's `branch_flow_p()` is DC-only, so a caller
who has just solved an AC power flow has to reimplement the π-model and its tap
handling to get them. That seemed the wrong thing to mirror.

## Getting data in

Two ways, both first-class.

**A power-grid-model document**, by path or from memory:

```cpp
auto pf = gridoxide::PowerFlow::from_pgm_file("network.json", options);
auto pf = gridoxide::PowerFlow::from_pgm_string(json_text, options);
```

**Arrays you already hold**, in per-unit:

```cpp
std::vector<gridoxide_bus> buses = { /* ... */ };
std::vector<gridoxide_line> lines = { /* ... */ };
auto pf = gridoxide::PowerFlow::from_arrays(buses, lines);
```

The structs are plain C layout — no pointers, nothing owned — so a
`std::vector` of them passes straight through with no marshalling step. The C++
wrapper deliberately does *not* mirror them in C++; wrapping a POD struct in
another POD struct would be a copy for nothing.

> **ZIP loads cannot cross the array path.** `gridoxide_bus` has no place for
> voltage-dependent load terms, which are a variable-length list. A network
> with them must come in as a document, where they are read as a matter of
> course. Passed through the array path they are silently absent, and the solve
> converges perfectly well to the wrong answer.

## Two things that will bite you otherwise

### A document yields more buses than it has nodes

Converting a power-grid-model document appends a **virtual slack bus per
`source`**, joined to the physical bus by an impedance branch. That is right
for a power flow, where a source's output is an unknown rather than a
schedule — but it means `case14_ieee` reports **15** buses, not 14.

Index results by what `bus_count()` reports, never by your own numbering.

### Handles are single-thread-affine

A handle owns the cached symbolic factorization, and under the `klu` or
`pardiso` features that is a raw pointer into SuiteSparse or MKL state which is
not safe for concurrent use. Python encodes this as `#[pyclass(unsendable)]`;
C has no equivalent, so it is a documented rule instead. `gridoxide::PowerFlow`
is movable but not copyable, which stops the obvious accident.

Separate handles on separate threads are fine.

## Errors

Every fallible C function returns a `gridoxide_status`; the message is fetched
separately with `gridoxide_last_error_message()`, valid until the next call
**on that thread**. The C++ wrapper turns both into a `gridoxide::Error`
carrying the code and the message.

The codes split the way the Python binding's exceptions do — caller-fault
versus numerics — plus one the Python binding did not need:

| Code | Meaning |
|---|---|
| `GRIDOXIDE_STATUS_OK` | Succeeded. |
| `..._INVALID_ARGUMENT` | Your mistake: null pointer, bad index, wrong buffer length, unavailable backend. |
| `..._IO` / `..._PARSE` | The file could not be read / the document could not be parsed. |
| `..._NOT_CONVERGED` | Ran out of iterations. **The state is still readable.** |
| `..._SINGULAR` | The Jacobian was singular. |
| `..._NO_ANSWER` | The question does not apply here. **Not a failure.** |
| `..._PANIC` | A panic was caught at the boundary — always a gridoxide bug. |

`NO_ANSWER` is the third channel. The Rust API uses `Option::None` to mean *"no
such answer exists"* — asking for the slack shift after a solve that did not
distribute slack, for instance. Folding that into the error codes would force a
caller to parse message strings to tell "your input was wrong" from "the
question does not apply". The C++ wrapper returns an empty vector rather than
throwing.

### Panics are caught, not fatal

A `panic!` unwinding out of an `extern "C"` function **aborts the process** —
it would take your program down from inside a library it merely linked. Every
exported function runs its body inside `catch_unwind` and reports
`GRIDOXIDE_STATUS_PANIC` instead.

That is not hypothetical. `newton_raphson_distributing_slack` asserts on a
wrong-length participation vector, `YBus::finish` expects well-formed triplets,
and the document conversion unwraps in places. The ABI checks what it can up
front and reports `INVALID_ARGUMENT` — a far more useful answer — and the guard
is the backstop for the panics nobody predicted.

## Memory

**Nothing allocated on one side is freed by the other.** Results go into
caller-owned buffers whose length you learn from a `_count` call first:

```c
size_t n = gridoxide_powerflow_bus_count(pf);
double *vm = malloc(n * sizeof(double));
gridoxide_powerflow_voltage_magnitude(pf, vm, n);
```

That removes allocator-mismatch bugs entirely, and costs nothing since every
result is a flat array indexed by bus or by branch. The C++ wrapper does the
count-then-fill once so you get a `std::vector`.

The handle is the exception: opaque, released by `gridoxide_powerflow_free`, or
by the C++ destructor.

Buffer lengths are **checked, not trusted** — passing a bus-sized buffer to a
per-branch accessor returns `INVALID_ARGUMENT` rather than overrunning it.

## Building

### With CMake

```bash
# From source: cargo runs as part of the build. Always in step; needs Rust.
cmake -S capi -B build/capi -DGRIDOXIDE_FROM_SOURCE=ON
cmake --build build/capi

# Prebuilt: no Rust toolchain anywhere.
cmake -S capi -B build/capi \
      -DGRIDOXIDE_LIBRARY=/path/to/libgridoxide.a \
      -DGRIDOXIDE_INCLUDE_DIR=/path/to/include
cmake --build build/capi
```

Either way you get one imported target, so the consuming `CMakeLists.txt` does
not care which was used:

```cmake
find_package(gridoxide REQUIRED)
target_link_libraries(my_app PRIVATE gridoxide::gridoxide)
```

A Rust `staticlib` carries no runtime of its own, so the target also brings the
platform link dependencies (`pthread`, `dl`, `m` on Linux). Omitting those is
the usual first failure — it appears as pages of undefined `std` symbols.

### With cargo directly

```bash
cargo build --release --features capi
```

produces `target/release/libgridoxide.{a,so}` and regenerates
`include/gridoxide.h`.

> **Never combine `capi` with `python`.** `pyo3`'s `extension-module` leaves
> libpython deliberately unlinked — right for something Python `dlopen`s, wrong
> for a library a C++ binary links directly. The two cannot come from one cargo
> invocation. The same trap the `python` feature already documents for tests
> and examples.

## The header is generated and committed

`include/gridoxide.h` is cbindgen's output and is **committed**, so a consumer
of a prebuilt release needs no Rust toolchain to get a header. Nothing then
forces it to match the code, so `tests/capi_test.rs` fails if it has drifted —
the same guard `test_committed_documents_match_a_fresh_conversion` provides for
the committed OPF fixtures.

That guard earned its keep immediately. Two accessors were originally written
as a `macro_rules!`, which is exactly what a macro is for — two near-identical
functions. But **cbindgen does not expand macros**, so they existed in the
library and were absent from the header: a consumer would have met symbols that
link but do not declare. They are written out longhand now, and the test names
any exported function the header does not declare.

`GRIDOXIDE_ABI_VERSION` is injected into the header from the Rust constant
rather than written twice, so the two cannot disagree.
`gridoxide::check_abi_compatibility()` compares them at run time — worth calling
once at startup when linking a prebuilt library, pointless when building from
source.

## How it is tested

`tests/capi_test.rs` drives the exported functions through raw pointers exactly
as C++ would, but needs no C compiler, so it runs wherever `cargo test` does.
The strongest check there is that **the ABI reproduces a direct Rust solve
bit-for-bit** — not to a tolerance, since both paths run the same iteration on
the same matrix, so any difference at all is a marshalling error rather than
numerical weather. The array path is checked the same way against the document
path, which is what makes the `#[repr(C)]` layouts trustworthy.

`capi/examples/powerflow.c` and `.cpp` cover what only a real compiler can: that
the header parses as C11 and C++17, that the struct layouts agree, and that the
symbols link. CI compiles and runs both.
