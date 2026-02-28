# 构建镜像和打包到镜像仓库
gcloud builds submit --project speedy-victory-336109 --config=cloudbuild.yaml --substitutions=_VERSION="0.12"

# 部署到cloud run
gcloud run deploy gta1 \
  --image asia-southeast1-docker.pkg.dev/speedy-victory-336109/myrepo/gta1:0.12 \
  --region asia-southeast1 \
  --platform managed \
  --gpu 1 \
  --gpu-type nvidia-l4 \
  --no-gpu-zonal-redundancy \
  --cpu 4 \
  --memory 16Gi \
  --port 8000 \
  --min 0 \
  --max 3 \
  --timeout 120 \
  --concurrency 1 \
  --cpu-boost \
  --allow-unauthenticated


gcloud beta run deploy gta1-rtx6000 \
  --image asia-southeast1-docker.pkg.dev/speedy-victory-336109/myrepo/gta1:0.12 \
  --region us-central1 \
  --platform managed \
  --gpu 1 \
  --gpu-type nvidia-rtx-pro-6000 \
  --no-gpu-zonal-redundancy \
  --cpu 20 \
  --memory 80Gi \
  --port 8000 \
  --min 0 \
  --max 1 \
  --timeout 120 \
  --concurrency 4 \
  --cpu-boost \
  --allow-unauthenticated \
  --add-volume name=model-vol,type=cloud-storage,bucket=pwm-models \
  --add-volume-mount volume=model-vol,mount-path=/mnt/models