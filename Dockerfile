# syntax=docker/dockerfile:1
# ============================================================================
#  Qwen3.8-Flash-Next NVFP4 on sm120 (RTX PRO 6000 Blackwell) — container build
#
#  *** UNTESTED. This image has never been built or run. ***
#  It encodes the README's manual steps plus the host packages this session
#  found missing on a bare container. Treat it as a starting point: expect to
#  fix at least one thing on the first `docker build`.
#
#  Model weights are NOT baked in (~135 GB). Mount them at run time.
#
#  Build (needs plenty of RAM — the sglang build spawns cicc at ~3GB each):
#     docker build -t qwen-flashnext .
#
#  Run (needs NVIDIA Container Toolkit on the host):
#     docker run --gpus all -p 8001:8001 \
#       -v /path/to/Qwen3.8-Flash-Next-NVFP4:/opt/qwen/models/Qwen3.8-Flash-Next-NVFP4:ro \
#       -v qwen-cache:/opt/qwen/cache \
#       qwen-flashnext ./serve best
#
#  The named cache volume matters: FlashInfer autotune JIT-compiles kernels on
#  first start (~20 min per the README). Without a persistent cache every fresh
#  container pays that again. It cannot be baked into the image because it needs
#  a GPU at build time, which most builders do not have.
# ============================================================================

# CUDA 13.0 to match the wheel index do_build.sh uses (docs.sglang.ai/whl/cu130).
# 'devel' (not 'runtime') is required: nvcc is needed for the runtime kernel JIT,
# not just at build time.
ARG CUDA_IMAGE=nvidia/cuda:13.0.0-devel-ubuntu22.04
FROM ${CUDA_IMAGE}

ARG SGLANG_BRANCH=qwen4-main-squashed
ARG SGLANG_REPO=https://github.com/sgl-project/sglang
ENV DEBIAN_FRONTEND=noninteractive

# python3.10-dev, not python3.10: Triton compiles a CUDA shim at runtime and needs
# Python.h. Its absence is a runtime crash deep in the scheduler, not a build error.
# ninja likewise is only reached at first kernel JIT.
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.10 python3.10-dev python3.10-venv python3-pip \
      build-essential gcc g++ git curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

ENV QWEN_HOME=/opt/qwen
WORKDIR ${QWEN_HOME}

# uv, used by do_build.sh
RUN pip3 install --no-cache-dir uv ninja

# Repo scripts first, so edits to them don't invalidate the expensive build layer.
COPY scripts/ ${QWEN_HOME}/scripts/
COPY patches/ ${QWEN_HOME}/patches/
COPY serve hot_tokens_64k.pt ${QWEN_HOME}/
RUN chmod +x ${QWEN_HOME}/serve ${QWEN_HOME}/scripts/*.sh

# Build sglang, then apply the six sm120 patches — this is the README's order:
# do_build.sh installs the editable tree first, patches land on it afterwards.
# git identity is set because `git am` refuses to run without one.
RUN git clone -q -b ${SGLANG_BRANCH} ${SGLANG_REPO} ${QWEN_HOME}/sglang-official \
 && cd ${QWEN_HOME}/sglang-official \
 && git config user.email build@localhost && git config user.name "container build" \
 && uv venv .venv --python 3.10 \
 && bash ${QWEN_HOME}/scripts/do_build.sh \
 && git apply ${QWEN_HOME}/patches/0002-fp8-qsa-tile-dequant.patch --exclude='test/*' \
 && git apply ${QWEN_HOME}/patches/0003-sm120-fp32-prefill-state.patch \
 && git apply ${QWEN_HOME}/patches/0001b-recoverssm-wy-sm120-PORTED.patch \
 && git am ${QWEN_HOME}/patches/0004*.patch ${QWEN_HOME}/patches/0005*.patch ${QWEN_HOME}/patches/0006*.patch

# Mount points. Weights are read-only; cache must be writable and should be a
# named volume so the first-start autotune is paid once, not per container.
VOLUME ["${QWEN_HOME}/models", "${QWEN_HOME}/cache"]
ENV BASE=${QWEN_HOME} \
    CACHE_BASE=${QWEN_HOME}/cache \
    TARGET_MODEL=${QWEN_HOME}/models/Qwen3.8-Flash-Next-NVFP4 \
    PORT=8001
EXPOSE 8001

# No systemd in a container, so ./serve takes its nohup branch. Run it in the
# foreground instead so the container's lifetime tracks the server's.
ENV FOREGROUND=1
HEALTHCHECK --interval=30s --timeout=5s --start-period=25m --retries=3 \
  CMD curl -fsS http://127.0.0.1:8001/health || exit 1

ENTRYPOINT ["./serve"]
CMD ["best"]
