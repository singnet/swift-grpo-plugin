# ======================
# Build arguments
# ======================
ARG CUDA_VERSION=12.6
ARG CUDA_IMAGE=12.6.0-devel-ubuntu22.04

FROM nvidia/cuda:${CUDA_IMAGE}

ARG CUDA_VERSION
ARG CUDA_IMAGE

ENV HOST=0.0.0.0

ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_VERSION=${CUDA_VERSION} \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /workspace

# ======================
# System dependencies
# ======================
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
        git \
        vim \
        build-essential \
        python3 \
        python3-pip \
        python3-dev \
        python3-venv \
        gcc \
        g++ \
        ninja-build \
        cmake \
        libopenmpi-dev \
        openmpi-bin \
        ffmpeg \
        libsm6 \
        libxext6 \
        libgl1 \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# ======================
# ms-swift
# ======================
RUN git clone https://github.com/modelscope/ms-swift.git /workspace/ms-swift && \
    cd /workspace/ms-swift && \
    python3 -m pip install -e ".[all]"

# ======================
# vLLM / DeepSpeed / Flash-Attention
# ======================
RUN set -eux; \
    if echo "${CUDA_VERSION}" | grep -qE '^13\.'; then \
        echo ">>> Installing packages for CUDA 13.x"; \
        pip install "vllm==0.23.0" \
                    "deepspeed==0.18.9"; \
    else \
        echo ">>> Installing packages for CUDA 12.6"; \
        pip install "vllm==0.19.0" \
                    "deepspeed==0.18.9"; \
    fi

# ======================
# PyTorch
# ======================
#RUN set -eux; \
#    if echo "${CUDA_VERSION}" | grep -qE '^13\.'; then \
#        echo ">>> Installing PyTorch for CUDA 13.x (cu130)"; \
#        pip install torch==2.11.0 torchvision torchaudio \
#            --index-url https://download.pytorch.org/whl/cu130; \
#    else \
#        echo ">>> Installing PyTorch for CUDA 12.6 (cu126)"; \
#        pip install torch==2.8.0 torchvision torchaudio \
#            --index-url https://download.pytorch.org/whl/cu126; \
#    fi

# ======================
# Plugin
# ======================
COPY . /workspace/swift-grpo-plugin

RUN if [ -f /workspace/swift-grpo-plugin/requirements.txt ]; then \
        pip install -r /workspace/plugin/requirements.txt; \
    else \
        echo "No requirements.txt found for plugin, skipping..."; \
    fi
