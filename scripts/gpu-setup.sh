#!/usr/bin/env bash
# Environment check / setup for a rented GPU box.
#
# gridoxide's GPU work needs more than a CUDA *runtime*: `bindgen` compiles
# against real vendor headers, so a plain PyTorch-runtime container will not do.
# This script verifies every piece before you start burning metered time, and
# tells you exactly what is missing and how to install it.
#
#   ./scripts/gpu-setup.sh              # check only, changes nothing
#   ./scripts/gpu-setup.sh --install    # additionally install what it can
#
# See scripts/GPU_RUNBOOK.md for what to do once this passes.

set -uo pipefail

INSTALL=0
[[ "${1:-}" == "--install" ]] && INSTALL=1

RED=$'\033[31m'; GRN=$'\033[32m'; YLW=$'\033[33m'; BLD=$'\033[1m'; RST=$'\033[0m'
MISSING=()
NOTES=()

ok()   { printf "  ${GRN}ok${RST}      %s\n" "$1"; }
bad()  { printf "  ${RED}MISSING${RST} %s\n" "$1"; MISSING+=("$1"); }
warn() { printf "  ${YLW}warn${RST}    %s\n" "$1"; }
hdr()  { printf "\n${BLD}%s${RST}\n" "$1"; }

need_cmd() { command -v "$1" >/dev/null 2>&1; }

# Package manager, for the suggested install lines only.
PKG=""
need_cmd apt-get && PKG="apt-get"
[[ -z "$PKG" ]] && need_cmd dnf && PKG="dnf"
[[ -z "$PKG" ]] && need_cmd pacman && PKG="pacman"

# Rented containers usually run as root with no `sudo` binary at all, while a
# rented VM usually needs it. Pick whichever applies rather than assuming.
SUDO=""
if [[ $EUID -ne 0 ]]; then
    if need_cmd sudo; then SUDO="sudo"; else SUDO="__NO_SUDO__"; fi
fi

install_pkgs() {
    local pkgs=("$@")
    if [[ "$SUDO" == "__NO_SUDO__" ]]; then
        NOTES+=("not root and no sudo available; install manually: ${pkgs[*]}")
        return
    fi
    if [[ $INSTALL -eq 0 ]]; then
        NOTES+=("install: ${SUDO:+$SUDO }$PKG install -y ${pkgs[*]}")
        return
    fi
    case "$PKG" in
        apt-get) $SUDO apt-get update -qq && $SUDO apt-get install -y "${pkgs[@]}" ;;
        dnf)     $SUDO dnf install -y "${pkgs[@]}" ;;
        pacman)  $SUDO pacman -S --noconfirm "${pkgs[@]}" ;;
        *)       NOTES+=("no known package manager; install manually: ${pkgs[*]}") ;;
    esac
}

# ---------------------------------------------------------------- vendor ----
hdr "GPU vendor"
VENDOR="none"
if need_cmd nvidia-smi && nvidia-smi -L >/dev/null 2>&1; then
    VENDOR="nvidia"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | sed 's/^/  /'
elif need_cmd rocminfo && rocminfo >/dev/null 2>&1; then
    VENDOR="amd"
    rocminfo 2>/dev/null | grep -m4 -E "Name:|Marketing" | sed 's/^/  /'
else
    bad "no NVIDIA or AMD GPU detected (nvidia-smi / rocminfo both absent or failing)"
fi
printf "  vendor: ${BLD}%s${RST}\n" "$VENDOR"

# FP64 sanity. Consumer parts are 1/32-1/64 rate and will produce meaningless
# numbers for this workload -- see plans/GPU_PLAN.md §5.
if [[ "$VENDOR" == "nvidia" ]]; then
    GPUNAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    case "$GPUNAME" in
        *A100*|*H100*|*H200*|*V100*|*A800*|*GH200*|*B100*|*B200*) ok "FP64-capable part ($GPUNAME)" ;;
        *)  warn "$GPUNAME is likely a consumer part with 1/32-1/64 FP64 rate."
            warn "This workload is FP64-bound; benchmark numbers from it mean nothing."
            warn "Correctness work is still fine." ;;
    esac
fi

# ----------------------------------------------------------------- rust ----
hdr "Rust toolchain"
if need_cmd rustc; then
    ok "rustc $(rustc --version | awk '{print $2}')"
else
    if [[ $INSTALL -eq 1 ]]; then
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --no-modify-path
        # shellcheck disable=SC1091
        source "$HOME/.cargo/env"
        need_cmd rustc && ok "rustc $(rustc --version | awk '{print $2}')" || bad "rustc (rustup install failed)"
    else
        bad "rustc"
        NOTES+=("install: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y")
    fi
fi

# --------------------------------------------------------------- bindgen ----
hdr "bindgen prerequisites"
# build.rs already uses bindgen twice (vendored KLU, PARDISO); the cuDSS and
# rocSOLVER shims follow that same precedent and need the same toolchain.
if need_cmd clang || [[ -n "$(ls /usr/lib/llvm-*/lib/libclang.so* 2>/dev/null | head -1)" ]] \
   || [[ -n "$(ls /usr/lib/x86_64-linux-gnu/libclang* 2>/dev/null | head -1)" ]]; then
    ok "libclang (bindgen)"
else
    bad "libclang"
    [[ "$PKG" == "apt-get" ]] && install_pkgs libclang-dev clang
    [[ "$PKG" == "dnf" ]]     && install_pkgs clang-devel
fi
if need_cmd cc; then ok "C compiler ($(cc --version 2>/dev/null | head -1 | cut -c1-40))"; else
    bad "a C compiler"; [[ "$PKG" == "apt-get" ]] && install_pkgs build-essential
fi
need_cmd pkg-config || { warn "pkg-config absent (usually fine)"; }

# ------------------------------------------------------------ cuda / rocm ----
if [[ "$VENDOR" == "nvidia" ]]; then
    hdr "CUDA toolkit (headers, not just runtime)"
    CUDA_HOME="${CUDA_HOME:-${CUDA_PATH:-/usr/local/cuda}}"
    if need_cmd nvcc; then
        ok "nvcc $(nvcc --version | grep -oP 'release \K[0-9.]+')"
    else
        bad "nvcc -- you are probably on a *runtime* image; you need a *devel* image"
        NOTES+=("RunPod/Docker: use an image tagged -devel- (e.g. nvidia/cuda:12.x.x-devel-ubuntu22.04)")
    fi
    if [[ -f "$CUDA_HOME/include/cuda_runtime.h" ]]; then
        ok "CUDA headers at $CUDA_HOME/include"
    else
        bad "cuda_runtime.h (looked in $CUDA_HOME/include)"
        NOTES+=("set CUDA_HOME if your toolkit lives elsewhere")
    fi

    hdr "cuDSS (ships separately from the CUDA toolkit)"
    CUDSS_ROOT="${CUDSS_ROOT:-/usr/local/cudss}"
    CUDSS_H="$(find "$CUDSS_ROOT" /usr/include /usr/local -maxdepth 4 -name 'cudss.h' 2>/dev/null | head -1)"
    CUDSS_SO="$(find "$CUDSS_ROOT" /usr/lib /usr/local -maxdepth 4 -name 'libcudss.so*' 2>/dev/null | head -1)"
    [[ -n "$CUDSS_H"  ]] && ok "cudss.h at $CUDSS_H"      || bad "cudss.h"
    [[ -n "$CUDSS_SO" ]] && ok "libcudss.so at $CUDSS_SO" || bad "libcudss.so"
    if [[ -z "$CUDSS_H" || -z "$CUDSS_SO" ]]; then
        NOTES+=("cuDSS is a separate NVIDIA download: https://developer.nvidia.com/cudss")
        NOTES+=("  then: export CUDSS_ROOT=/path/to/cudss  (build.rs will discover it there)")
        NOTES+=("  check its licence before shipping anything -- see PROVENANCE.md precedent")
    fi

elif [[ "$VENDOR" == "amd" ]]; then
    hdr "ROCm toolkit (headers, not just runtime)"
    ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
    need_cmd hipcc && ok "hipcc" || bad "hipcc (install rocm-hip-sdk / rocm-dev)"
    if [[ -f "$ROCM_PATH/include/rocsolver/rocsolver.h" ]]; then
        ok "rocsolver.h at $ROCM_PATH/include/rocsolver"
    elif [[ -f "$ROCM_PATH/include/rocsolver.h" ]]; then
        ok "rocsolver.h at $ROCM_PATH/include"
    else
        bad "rocsolver.h (looked under $ROCM_PATH/include)"
        [[ "$PKG" == "apt-get" ]] && install_pkgs rocsolver-dev rocblas-dev
    fi
    ROCSOLVER_SO="$(find "$ROCM_PATH/lib" /usr/lib -maxdepth 3 -name 'librocsolver.so*' 2>/dev/null | head -1)"
    [[ -n "$ROCSOLVER_SO" ]] && ok "librocsolver.so at $ROCSOLVER_SO" || bad "librocsolver.so"
    NOTES+=("rocSOLVER's csrrf_* routines are the target: analysis once, refactor repeatedly")
    NOTES+=("  https://rocm.docs.amd.com/projects/rocSOLVER/en/develop/api/refact.html")
fi

# --------------------------------------------------------- optional extras ----
hdr "Optional (for the full comparison suite)"
need_cmd python3 && ok "python3 $(python3 -V 2>&1 | awk '{print $2}')" || warn "python3 absent (needed for scripts/bench/*.py)"
if [[ -n "${MKLROOT:-}" ]]; then ok "MKLROOT set (pardiso backend available)"; else warn "MKLROOT unset -- pardiso backend unavailable (optional)"; fi

# ----------------------------------------------------------------- build ----
hdr "Build check (CPU paths only -- nothing GPU-specific yet)"
if need_cmd cargo; then
    if cargo build --quiet 2>/dev/null; then
        ok "cargo build (default features)"
    else
        bad "cargo build failed -- run 'cargo build' to see why"
    fi
else
    warn "cargo unavailable; skipping build check"
fi

# --------------------------------------------------------------- summary ----
hdr "Summary"
if [[ ${#MISSING[@]} -eq 0 ]]; then
    printf "  ${GRN}Environment is ready.${RST}\n"
else
    printf "  ${RED}%d item(s) missing:${RST}\n" "${#MISSING[@]}"
    printf "    - %s\n" "${MISSING[@]}"
fi
if [[ ${#NOTES[@]} -gt 0 ]]; then
    printf "\n${BLD}Notes${RST}\n"
    printf "  %s\n" "${NOTES[@]}"
fi
printf "\nNext: ${BLD}scripts/GPU_RUNBOOK.md${RST}\n"

[[ ${#MISSING[@]} -eq 0 ]] || exit 1
