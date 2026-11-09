# Reproducible build of the tests and the benchmark.
#
#   docker build -f docker/Dockerfile -t t_convl .
#   docker run --rm --gpus all -v "$PWD/results:/results" t_convl \
#       bench --benchmark --label 4060-docker --out /results/4060.csv
#
# Requirements on the host: an NVIDIA driver new enough for the image's CUDA version
# (`nvidia-smi` shows the highest CUDA version it supports) and the NVIDIA Container Toolkit
# (on WSL2: Docker Desktop with the WSL2 backend, or docker + nvidia-container-toolkit in the distro).
#
# PINNING: CUDA image tags expire (Docker Hub deletes tags after their driver branch reaches
# end of life). For a result you can rebuild in a year, pin by DIGEST, not tag:
#   docker pull nvidia/cuda:13.4.3-cudnn-devel-ubuntu24.04
#   docker inspect --format '{{index .RepoDigests 0}}' nvidia/cuda:13.4.3-cudnn-devel-ubuntu24.04
# then build with --build-arg CUDA_DEVEL_IMAGE=nvidia/cuda@sha256:<digest> (and the same for
# the runtime image). For a CUDA 12 host driver, pass 12.x tags that still exist.
#
# The Jetson Orin needs its own L4T-based image (JetPack ships CUDA and cuDNN); this file is
# for x86_64 machines: the RTX 4060 under WSL2 and cloud GPUs.

ARG CUDA_DEVEL_IMAGE=nvidia/cuda:13.4.3-cudnn-devel-ubuntu24.04
ARG CUDA_RUNTIME_IMAGE=nvidia/cuda:13.4.3-cudnn-runtime-ubuntu24.04

# ---------------------------------------------------------------------------
# Stage 1: full toolchain. Compiles every harness and runs the host-only tests as a gate.
# ---------------------------------------------------------------------------
FROM ${CUDA_DEVEL_IMAGE} AS build

RUN apt-get update \
 && apt-get install -y --no-install-recommends cmake ninja-build g++ \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY CMakeLists.txt ./
COPY include    include
COPY tests      tests
COPY benchmarks benchmarks

RUN cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
 && cmake --build build \
 && ctest --test-dir build -R "test_boundary|test_reference" --output-on-failure

# ---------------------------------------------------------------------------
# Stage 2: runtime only. The binaries plus the CUDA, cuDNN and NPP runtime libraries.
# ---------------------------------------------------------------------------
FROM ${CUDA_RUNTIME_IMAGE}

COPY --from=build /src/build/bench            /usr/local/bin/bench
COPY --from=build /src/build/test_boundary    /usr/local/bin/test_boundary
COPY --from=build /src/build/test_reference   /usr/local/bin/test_reference
COPY --from=build /src/build/test_filter_bank /usr/local/bin/test_filter_bank
COPY --from=build /src/build/test_correctness /usr/local/bin/test_correctness
COPY --from=build /src/build/test_convl2d      /usr/local/bin/test_convl2d
COPY --from=build /src/build/test_baselines   /usr/local/bin/test_baselines

# Default: run every GPU test. Override with `bench --benchmark ...` to run the sweep.
CMD ["sh", "-c", "test_filter_bank && test_correctness && test_convl2d && test_baselines"]