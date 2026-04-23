docker build -t gta1-7b:v0.1.1 .
docker run --gpus all --rm -p 8000:8000 -v ~/.cache/huggingface:/root/.cache/huggingface gta1-7b:v0.2.0


python3 test_api.py step.png "pc download button"