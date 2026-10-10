# Foley-Omni preview webapp — NVIDIA GPU image
# Checkpoints are NOT baked in; mount ./ckpts at runtime.

FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/app/.cache/huggingface \
    TORCH_HOME=/app/.cache/torch \
    CKPT_DIR=/app/ckpts \
    STORAGE_DIR=/app/storage \
    MODEL_CHECKPOINT=/app/ckpts/Foley-Omni/v2st.pth

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 \
        python3.10-venv \
        python3-pip \
        python3.10-dev \
        git \
        ffmpeg \
        libsndfile1 \
        libgl1 \
        libglib2.0-0 \
        curl \
        build-essential \
        ninja-build \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# PyTorch matching public release (CUDA 12.4 / cu124)
RUN pip install --upgrade pip setuptools wheel \
    && pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
        --index-url https://download.pytorch.org/whl/cu124

COPY requirements.txt requirements-web.txt ./
RUN pip install -r requirements.txt \
    && pip install -r requirements-web.txt

# FlashAttention is preferred but optional (SDPA fallback exists).
# Build can take a while; failure must not break the image.
RUN pip install flash_attn==2.7.4.post1 --no-build-isolation \
    || echo "WARNING: flash_attn install failed; continuing with SDPA fallback"

COPY app ./app
COPY static ./static
COPY foley_omni ./foley_omni
COPY mmaudio ./mmaudio
COPY inference_v2st.py inference_v2st.yaml inference_fusion.yaml config.json ./
COPY scripts ./scripts

RUN mkdir -p /app/ckpts /app/storage /app/.cache/huggingface /app/.cache/torch

# Run as root so host-mounted ./storage is writable without UID gymnastics.
# This is a private GPU preview service, not a multi-tenant public SaaS.

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=5 \
    CMD curl -fsS http://127.0.0.1:8000/api/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
