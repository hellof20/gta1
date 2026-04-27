# GTA-1 (Qwen2.5-VL)

Model: `HelloKKMe/GTA1-7B`

## Build

From `models/gta-1/`:

```bash
# Transformers version
docker build -t gta1-7b:v0.1.1 .

# vLLM version (production)
docker buildx build -f Dockerfile.vllm -t gta1-vllm:0.1.0 --load .
```

## Run

```bash
# Transformers
docker run --gpus all --rm -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  gta1-7b:v0.2.0

# vLLM (production)
docker run --rm --gpus all \
  -v /home/pwm/GTA1-7B:/models/GTA1-7B \
  -p 8000:8000 \
  --name gta1-vllm \
  gta1-vllm:0.1.0
```

## Test

From repo root:

```bash
python3 tools/test.py tools/test_files/step.png "pc download button"
```

## Upload model to GCS

```bash
gcloud storage cp -r /home/pwm/GTA1-7B gs://pwmmodel/ 2>&1 | tee /tmp/gta1-upload.log
```
