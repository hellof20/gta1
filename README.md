# GTA-1: GUI Grounding Inference Service

GUI element grounding service based on [GTA1-7B](https://huggingface.co/HelloKKMe/GTA1-7B). Given a screenshot and a natural-language description of a UI element, the service returns the (x, y) pixel coordinates of that element.

## Project Structure

```
models/
  gta-1/          # GTA1-7B — primary model (Qwen2.5-VL based)
  mai-ui/         # MAI-UI model
  fara/           # Fara model
  POINTS-GUI-G/   # POINTS-GUI-G model
tools/
  test.py         # Inference test with annotated output
  bench.py        # Concurrency benchmark
  download_model.py
deploy/
  cloudrun.sh     # Cloud Run deployment commands
```

## Quick Start

### 1. Download model

```bash
python3 tools/download_model.py
```

### 2. Start the service locally

Two serving backends are available:

| Backend | File | Description |
|---------|------|-------------|
| Transformers | `api.py` | Single-request serving, uses HuggingFace Transformers + Flash Attention |
| vLLM | `api_vllm.py` | Production serving, supports continuous batching, prefix caching, and multimodal cache |

```bash
cd models/gta-1
pip install -r requirements.txt

# Transformers backend
python3 api.py

# vLLM backend (recommended for production)
python3 api_vllm.py
```

The server starts at `http://localhost:8000`.

#### vLLM Configuration

`api_vllm.py` supports the following environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `HelloKKMe/GTA1-7B` | Model path (local or HuggingFace) |
| `MAX_MODEL_LEN` | `12288` | Maximum sequence length |
| `GPU_MEM_UTIL` | `0.85` | GPU memory utilization ratio |
| `MAX_PIXELS` | `8294400` (3840x2160) | Maximum input image pixels |
| `INFLIGHT_LIMIT` | `4` | Max concurrent `/process` requests, excess returns 503 |
| `ENABLE_PREFIX_CACHE` | `1` | Enable vLLM prefix caching (KV cache reuse across requests) |
| `ENABLE_MM_CACHE` | `1` | Enable multimodal processor cache (4 GB) |

### 3. Test

```bash
python3 tools/test.py tools/test_files/big.jpg "chat button" --output-dir results/
```

## API

### `GET /`

Health check.

```json
{"status": "healthy"}
```

### `POST /process/`

Locate a UI element in the given image.

| Parameter     | Type       | Description                          |
|---------------|------------|--------------------------------------|
| `instruction` | form field | Natural-language element description |
| `image_file`  | file       | Screenshot (PNG / JPEG / WebP / BMP / TIFF) |

**Response:**

```json
{"x": 512.0, "y": 384.0}
```

## Tools

### test.py

Send a single request and save an annotated image with the predicted coordinates.

```bash
python3 tools/test.py <image> <instruction> [--url URL] [--output-dir DIR]
```

- `--url` — service endpoint (default: `http://localhost:8000`)
- `--output-dir` — directory for annotated images (default: same as input image)

### bench.py

Concurrency benchmark for throughput and latency measurement.

```bash
# Single burst
python3 tools/bench.py tools/test_files/mail.jpg "chat button" -c 8 -n 32

# Sweep across concurrency levels
python3 tools/bench.py tools/test_files/mail.jpg "chat button" -n 32 --sweep

# Constant rate mode (0.5 req/s for 60s)
python3 tools/bench.py tools/test_files/mail.jpg "chat button" --rate 0.5 --duration 60
```

Set `BASE_URL` env var to target a remote endpoint.

> **Note:** bench.py sends the same image+instruction for every request. When targeting vLLM with prefix caching enabled (`ENABLE_PREFIX_CACHE=1`), KV cache will hit aggressively, making this a best-case throughput measurement. To test without cache, set `ENABLE_PREFIX_CACHE=0` on the server side, or use different images/instructions per request.

## Deployment

### Cloud Run (GPU)

```bash
# Build and push image
gcloud builds submit --config=cloudbuild.yaml --substitutions=_VERSION="0.12"

# Deploy with L4 GPU
gcloud run deploy gta1 \
  --image <REGION>-docker.pkg.dev/<PROJECT>/myrepo/gta1:<VERSION> \
  --gpu 1 --gpu-type nvidia-l4 \
  --cpu 4 --memory 16Gi \
  --port 8000 --concurrency 1 \
  --allow-unauthenticated
```

See [deploy/cloudrun.sh](deploy/cloudrun.sh) for full deployment commands including RTX 6000 configuration.
