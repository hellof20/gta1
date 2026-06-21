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

### AWS CodeBuild

First time setup (creates S3 bucket, ECR repo, IAM role, CodeBuild project):

```bash
./aws_build.sh 0.1 setup
```

Submit build:

```bash
./aws_build.sh 0.2.1
```

Resources: 8 vCPU / 15GB memory / 128GB disk (BUILD_GENERAL1_LARGE, ~$0.02/min).
Image pushed to `527432981953.dkr.ecr.us-east-1.amazonaws.com/qira/gta1-vllm`.

### GCP Cloud Build

```bash
cd models/gta-1
gcloud builds submit \
  --config gcp_cloudbuild.yaml \
  --substitutions=_VERSION=0.1,_REGION=us-central1,_REPOSITORY=myrepo
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

## Upload model

```bash
# GCS
gcloud storage cp -r /home/pwm/GTA1-7B gs://pwmmodel/ 2>&1 | tee /tmp/gta1-upload.log

# S3
aws s3 cp /home/pwm/GTA1-7B s3://pwmmodel/GTA1-7B --recursive
```
