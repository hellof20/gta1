# GTA1-vLLM on GKE — Setup Guide

Target setup:
- 1 GKE Standard cluster + 1 GPU node pool (L4 × g2-standard-8, autoscale 1–3)
- Model loaded from `gs://pwmmodel/GTA1-7B` via GCS Fuse CSI (Workload Identity)
- KEDA + Prometheus (kube-prometheus-stack) for autoscaling on in-flight requests
- minReplicas=1, maxReplicas=3, scale threshold = 2 inflight/Pod (= measured N)

Replace these placeholders before running:

```
export PROJECT_ID=pwmtest1
export PROJECT_NUMBER=887378088346
export REGION=us-central1
export NODE_ZONES=us-central1-a,us-central1-b,us-central1-c   # L4 is available in a/b/c
export CLUSTER=gta1-cluster
export BUCKET=pwmmodel
export NAMESPACE=gta1
export SA=gta1-vllm
```

---

## 1. Create cluster (Regional, with GCS Fuse + Workload Identity)

Regional control plane (HA across zones) + small system node pool in one zone (cheap).

```bash
gcloud container clusters create $CLUSTER \
  --project $PROJECT_ID \
  --region $REGION \
  --node-locations ${REGION}-a \
  --release-channel regular \
  --machine-type c4-standard-4 \
  --num-nodes 1 \
  --workload-pool=${PROJECT_ID}.svc.id.goog \
  --addons GcsFuseCsiDriver \
  --enable-ip-alias
```

## 2. Add L4 GPU node pool (multi-zone, autoscale 1–3 total)

Spans 3 zones so the autoscaler can pick whichever has L4 capacity. Use `--total-min-nodes` / `--total-max-nodes` (cluster-wide totals) instead of per-zone min/max — otherwise `--min-nodes 1` × 3 zones would force 3 always-on GPU nodes.

```bash
gcloud container node-pools create gpu-l4 \
  --project $PROJECT_ID \
  --cluster $CLUSTER \
  --region $REGION \
  --node-locations $NODE_ZONES \
  --machine-type g2-standard-8 \
  --accelerator type=nvidia-l4,count=1,gpu-driver-version=latest \
  --enable-autoscaling \
  --total-min-nodes 1 \
  --total-max-nodes 3 \
  --num-nodes 0 \
  --disk-size 200 \
  --node-taints nvidia.com/gpu=present:NoSchedule \
  --location-policy ANY \
  --enable-image-streaming
```

> **Notes**:
> - `--location-policy ANY` tells the cluster autoscaler to pick whichever zone currently has L4 stockout-free capacity (instead of trying to balance evenly). Recommended for scarce GPU SKUs.
> - `--total-min-nodes 1` keeps one L4 always warm so the first Pod doesn't pay node-provisioning cost (~3 min). 1→2 Pods still requires a 2nd node. Set to 0 if cost matters more than first-request latency.

## 3. Get kubectl credentials

Install `kubectl` and the GKE auth plugin if not already installed.

If the Google Cloud SDK apt repo isn't configured yet, add it first:

```bash
sudo apt-get install -y apt-transport-https ca-certificates gnupg curl
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg \
  | sudo gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg
echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" \
  | sudo tee /etc/apt/sources.list.d/google-cloud-sdk.list
sudo apt-get update
```

Then install:

```bash
sudo apt-get install -y kubectl google-cloud-cli-gke-gcloud-auth-plugin
```

Then fetch credentials:

```bash
gcloud container clusters get-credentials $CLUSTER --project $PROJECT_ID --region $REGION
```

## 4. Grant Workload Identity access to the model bucket

```bash
# Create namespace + SA first (apply 00-namespace.yaml, then run this)
kubectl apply -f deploy/k8s/00-namespace.yaml

gcloud storage buckets add-iam-policy-binding gs://${BUCKET} \
  --project $PROJECT_ID \
  --role=roles/storage.objectViewer \
  --member=principal://iam.googleapis.com/projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/${PROJECT_ID}.svc.id.goog/subject/ns/${NAMESPACE}/sa/${SA}
```

## 5. Install Prometheus (kube-prometheus-stack)

```bash
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update
helm install prometheus prometheus-community/kube-prometheus-stack \
  --namespace monitoring --create-namespace \
  --set prometheus.prometheusSpec.serviceMonitorSelectorNilUsesHelmValues=false
```

> `serviceMonitorSelectorNilUsesHelmValues=false` lets Prometheus discover ServiceMonitors in all namespaces (including `gta1`), not just ones created by the Helm chart.

## 6. Install KEDA

```bash
helm repo add kedacore https://kedacore.github.io/charts
helm repo update
helm install keda kedacore/keda --namespace keda --create-namespace
```

## 7. Build & push image (GKE specific)

```bash
cd models/gta-1
gcloud builds submit \
  --project $PROJECT_ID \
  --config cloudbuild.yaml \
  --substitutions=_VERSION=0.2 \
  .
```

Update the image tag in `deploy/k8s/20-deployment.yaml` to match `_VERSION`.

## 8. Apply manifests

```bash
kubectl apply -f deploy/k8s/10-servicemonitor.yaml
kubectl apply -f deploy/k8s/20-deployment.yaml
kubectl apply -f deploy/k8s/21-service.yaml
kubectl apply -f deploy/k8s/30-scaledobject.yaml
kubectl apply -f deploy/k8s/40-pdb.yaml
```

## 9. Verify

```bash
# Pod is running and ready
kubectl -n gta1 get pods -w

# ServiceMonitor is discovered by Prometheus
kubectl -n gta1 get servicemonitor

# KEDA picked up the ScaledObject
kubectl -n gta1 get scaledobject
kubectl -n gta1 get hpa            # KEDA creates an HPA under the hood

# Test the endpoint
kubectl -n gta1 port-forward svc/gta1-vllm 8000:8000
curl http://localhost:8000/ready
curl http://localhost:8000/metrics | grep gta1_inflight
```

## 10. Expose externally (optional)

For a quick internal LB:

```bash
kubectl -n gta1 patch svc gta1-vllm -p '{"spec":{"type":"LoadBalancer"}}'
```

For production HTTP(S) ingress, use GKE Gateway API.

---

## Operational notes

- **Cold start**: First Pod takes ~2–4 min (node provisioning if cold + vLLM model load from GCS). `readinessProbe.failureThreshold=30` allows 5 min before marking failed.
- **Scale-up latency**: When KEDA wants a 2nd Pod, GKE cluster autoscaler must spin up a 2nd L4 node (~3 min) before the Pod can schedule.
- **Cost**: L4 g2-standard-8 ≈ $0.70/hr × 1 always-on Pod ≈ $500/mo baseline. Each additional Pod ≈ $500/mo while running.
- **Scale signal**: `gta1_inflight_requests` (app-level counter). vLLM's own metrics (`vllm:num_requests_running/waiting`) are also exposed at `/metrics` if you prefer those.
- **503 behavior**: When a Pod hits `INFLIGHT_LIMIT=4` it returns 503 with `Retry-After: 2`. Configure your client (or LB) to retry on a different backend.
