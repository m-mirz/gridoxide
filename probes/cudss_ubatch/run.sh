#!/usr/bin/env bash
# The cuDSS probe (README.md), on a machine with an NVIDIA GPU, the CUDA
# toolkit and Rust. Fetches cuDSS (pinned, checksummed), builds the probe,
# writes the Jacobian dumps from this checkout, runs the probe over batch
# sizes, then gridoxide-gpu's cuDSS backend tests (`--features cuda`). Results
# go to probes/cudss_ubatch/results/.
#
#   CUDA_HOME=/usr/local/cuda probes/cudss_ubatch/run.sh [cudss_version]
#
# cudss_version: 1.0.0 (default; the one gridoxide-gpu builds against) or
# 0.8.0 (gpusim2grid's), which phase 0 found computing NaN solutions in a
# uniform batch: the probe still runs it, to reproduce that, but not
# gridoxide-gpu's tests, whose build refuses it.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../.." && pwd)
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
# A distribution's CUDA package (Ubuntu's nvidia-cuda-toolkit) puts the headers
# in /usr/include, not under CUDA_HOME.
if [ -z "${CUDA_INCLUDE_DIR:-}" ] && [ ! -f "$CUDA_HOME/include/cuda_runtime.h" ] \
   && [ -f /usr/include/cuda_runtime.h ]; then
  export CUDA_INCLUDE_DIR=/usr/include
fi
version=${1:-1.0.0}
case "$version" in
  0.8.0) archive=libcudss-linux-x86_64-0.8.0.10_cuda12-archive
         sha=29716a05c64b28531309afcb7a9dd5044081a43197e72bdd0fbf29e9b4ee4707 ;;
  1.0.0) archive=libcudss-linux-x86_64-1.0.0.35_cuda12-archive
         sha=b8c882c3ebc8b319886f5a06e74311c5eea09846925ba5a19892de4921201746 ;;
  *) echo "unknown cuDSS version $version" >&2; exit 2 ;;
esac
work="$here/work"; mkdir -p "$work" "$here/results"

# cuDSS from NVIDIA's redistributable archive, as grid-bench's gpusim2grid image takes it.
if [ ! -d "$work/$archive" ]; then
  curl -fsSL -o "$work/$archive.tar.xz" \
    "https://developer.download.nvidia.com/compute/cudss/redist/libcudss/linux-x86_64/$archive.tar.xz"
  echo "$sha  $work/$archive.tar.xz" | sha256sum -c -
  tar -xJf "$work/$archive.tar.xz" -C "$work"
fi
cudss="$work/$archive"

g++ -std=c++17 -O2 -Wall -Wextra -o "$work/probe" "$here/probe.cpp" \
  -I"${CUDA_INCLUDE_DIR:-$CUDA_HOME/include}" -I"$cudss/include" \
  -L"$CUDA_HOME/lib64" -L"$cudss/lib" -lcudss -lcudart \
  -Wl,-rpath,"$cudss/lib" -Wl,-rpath,"$CUDA_HOME/lib64"

(cd "$repo" && cargo build --release --example gpu_probe_dump --features matpower)
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv | tee "$here/results/gpu.txt"
export CUDSS_THREADING_LIB="$cudss/lib/libcudss_mtlayer_gomp.so.0"
[ -f "$CUDSS_THREADING_LIB" ] || CUDSS_THREADING_LIB=$(ls "$cudss"/lib/libcudss_mtlayer_gomp.so* | head -1)

for case in case1354pegase case2869pegase case9241pegase; do
  dump="$work/dump-$case"
  [ -f "$dump/meta.txt" ] || "$repo/target/release/examples/gpu_probe_dump" \
    "$repo/tests/data/benchmark-grids/matpower/$case.m" 64 "$dump"
  for B in 64 256 1024 2048; do
    out="$here/results/$case-B$B-cudss$version.txt"
    echo "== $case B=$B"
    # A run that fails to allocate at a large B is a result too: record and go on.
    "$work/probe" "$dump" "$B" 7 2>&1 | tee "$out" | grep -E "^RESULT|^PASS|^FAIL|cuDSS " || true
  done
done

# explore.cpp's follow-ups: what the mask does to unselected systems, and the
# cost of a refactorization and solve against B (the cliff at B = 16 in 0.8.0).
g++ -std=c++17 -O2 -o "$work/explore" "$here/explore.cpp" \
  -I"${CUDA_INCLUDE_DIR:-$CUDA_HOME/include}" -I"$cudss/include" \
  -L"$CUDA_HOME/lib64" -L"$cudss/lib" -lcudss -lcudart \
  -Wl,-rpath,"$cudss/lib" -Wl,-rpath,"$CUDA_HOME/lib64"
for B in 8 64; do
  echo "== explore mask, case1354pegase B=$B"
  "$work/explore" "$work/dump-case1354pegase" "$B" mask 2>&1 \
    | tee "$here/results/explore-mask-B$B-cudss$version.txt" | grep -E "^[0-9]" || true
done
for case in case1354pegase case9241pegase; do
  out="$here/results/explore-time-$case-cudss$version.txt"; : > "$out"
  echo "== explore time, $case"
  for B in 1 8 15 16 64 256 1024 2048; do
    "$work/explore" "$work/dump-$case" "$B" time 0 0 0 2>&1 | tail -1 | tee -a "$out" || true
  done
done

# gridoxide-gpu's cuDSS backend on the same device: its contract tests and the
# adaptive Newton loop against the CPU stand-in. A failure is a result too.
if [ "${version%%.*}" = 0 ]; then
  echo "== gridoxide-gpu needs cuDSS 1.0 or later: its tests skipped for $version"
  exit 0
fi
echo "== gridoxide-gpu --features cuda"
(cd "$repo" && CUDSS_DIR="$cudss" CUDA_HOME="$CUDA_HOME" \
  cargo test --release -p gridoxide-gpu --features cuda 2>&1) \
  | tee "$here/results/gridoxide-gpu-cudss$version.txt" | grep -E "^test |^test result" || true
