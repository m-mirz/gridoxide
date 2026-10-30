# gridoxide's Python bindings with the `gpu` feature: `GpuBatchSolver`, the
# batched power flow on an NVIDIA GPU (plans/GPU_BATCH_PLAN.md, phase 5), for
# grid-bench's GPU tabs. From the repository root:
#
#   docker build -f docker/gpu/Dockerfile -t gridoxide-gpu .
#   docker run --rm --gpus all -v $PWD:/repo -w /repo gridoxide-gpu \
#       pytest python/tests/test_gpu.py
#
# (The tests read tests/data/benchmark-grids from the mounted checkout.)
# CUDA_ARCH is the GPU the kernels are compiled for: sm_80 for an A100, sm_90
# for an H100.
#
# The runtime stage carries cuDSS's shared libraries from NVIDIA's
# redistributable archive, as grid-bench's gpusim2grid image does; their
# licence (LICENSE in the archive, copied to /opt/cudss) governs passing the
# image on.
ARG CUDA_VERSION=12.8.1

FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu24.04 AS build
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential ca-certificates curl libclang-dev python3 python3-venv xz-utils \
 && rm -rf /var/lib/apt/lists/*

# cuDSS 1.0.0, pinned and checksummed as probes/cudss_ubatch/run.sh takes it;
# gridoxide-gpu refuses anything older.
ARG CUDSS_ARCHIVE=libcudss-linux-x86_64-1.0.0.35_cuda12-archive
ARG CUDSS_SHA256=b8c882c3ebc8b319886f5a06e74311c5eea09846925ba5a19892de4921201746
RUN curl -fsSL -o /tmp/cudss.tar.xz \
      "https://developer.download.nvidia.com/compute/cudss/redist/libcudss/linux-x86_64/${CUDSS_ARCHIVE}.tar.xz" \
 && echo "${CUDSS_SHA256}  /tmp/cudss.tar.xz" | sha256sum -c - \
 && tar -xJf /tmp/cudss.tar.xz -C /opt \
 && mv "/opt/${CUDSS_ARCHIVE}" /opt/cudss \
 && rm /tmp/cudss.tar.xz

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
ENV PATH=/root/.cargo/bin:$PATH \
    CUDSS_DIR=/opt/cudss \
    CUDA_HOME=/usr/local/cuda
ARG CUDA_ARCH=sm_80
ENV CUDA_ARCH=${CUDA_ARCH}

WORKDIR /src
COPY . .
# The run path the build writes into the extension names /opt/cudss/lib and
# /usr/local/cuda/lib64, where the runtime stage has them. `--auditwheel
# skip`: the wheel must not bundle cuDSS or the CUDA runtime.
RUN python3 -m venv /opt/build \
 && /opt/build/bin/pip install "maturin>=1.9,<2.0" \
 && /opt/build/bin/maturin build --release --features python,gpu \
      --compatibility linux --auditwheel skip -o /wheels

FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu24.04
RUN apt-get update \
 && apt-get install -y --no-install-recommends python3 python3-venv libgomp1 \
 && rm -rf /var/lib/apt/lists/*
COPY --from=build /opt/cudss/lib /opt/cudss/lib
COPY --from=build /opt/cudss/LICENSE /opt/cudss/LICENSE
COPY --from=build /wheels /wheels
RUN python3 -m venv /opt/venv \
 && /opt/venv/bin/pip install --no-cache-dir /wheels/*.whl numpy pytest
ENV PATH=/opt/venv/bin:$PATH
CMD ["python3", "-c", "import gridoxide; print(gridoxide.GpuBatchSolver)"]
