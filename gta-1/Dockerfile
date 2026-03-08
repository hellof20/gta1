FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3.10-dev \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir torch==2.5.0 torchvision --index-url https://download.pytorch.org/whl/cu124 && \
    pip3 install --no-cache-dir \
        accelerate==1.1.1 \
        Pillow==11.1.0 \
        transformers==4.51.3 \
        qwen-vl-utils==0.0.8 \
        fastapi \
        uvicorn[standard] \
        python-multipart && \
    pip3 install --no-cache-dir \
        https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl && \
    find /usr/local/lib/python3.10 -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true && \
    find /usr/local/lib/python3.10 -type f -name "*.pyc" -delete && \
    find /usr/local/lib/python3.10 -type f -name "*.pyo" -delete

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

COPY --from=builder /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages
COPY --from=builder /usr/local/bin /usr/local/bin

RUN pip3 install --no-cache-dir huggingface_hub && \
    python3 -c "from huggingface_hub import snapshot_download; snapshot_download('HelloKKMe/GTA1-7B')"

COPY api.py .

EXPOSE 8000

CMD ["python3", "api.py"]
