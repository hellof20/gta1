# GTA1 vLLM - AWS EKS 部署

## 创建 EKS 集群

```bash
eksctl create cluster \
  --name vision \
  --region us-east-1 \
  --version 1.35 \
  --without-nodegroup
```

创建 OIDC provider 和 S3 CSI Driver IAM Role：

```bash
eksctl utils associate-iam-oidc-provider --cluster vision --region us-east-1 --approve

eksctl create iamserviceaccount \
  --name s3-csi-driver-sa \
  --namespace kube-system \
  --cluster vision \
  --attach-policy-arn arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess \
  --approve \
  --role-name AmazonEKS_S3_CSI_DriverRole \
  --region us-east-1 \
  --role-only
```

安装 S3 CSI Driver addon：

```bash
eksctl create addon --name aws-mountpoint-s3-csi-driver --cluster vision \
  --service-account-role-arn arn:aws:iam::527432981953:role/AmazonEKS_S3_CSI_DriverRole \
  --force --region us-east-1
```

## 创建 GPU 节点组

获取集群子网：

```bash
SUBNETS=$(aws eks describe-cluster --name vision --region us-east-1 \
  --query 'cluster.resourcesVpcConfig.subnetIds' --output text | tr '\t' ',')
```

创建节点组（含 Warm Pool）：

```bash
aws eks create-nodegroup \
  --cluster-name vision \
  --nodegroup-name gpu-g5 \
  --node-role arn:aws:iam::527432981953:role/ec2-common-role \
  --subnets $(echo $SUBNETS | tr ',' ' ') \
  --region us-east-1 \
  --scaling-config minSize=1,maxSize=3,desiredSize=1 \
  --ami-type AL2023_x86_64_NVIDIA \
  --instance-types g5.xlarge \
  --disk-size 100 \
  --warm-pool-config enabled=true,minSize=1,poolState=Stopped,reuseOnScaleIn=true
```

## 安装依赖组件

```bash
# NVIDIA Device Plugin
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.17.0/deployments/static/nvidia-device-plugin.yml

# KEDA
helm repo add kedacore https://kedacore.github.io/charts
helm install keda kedacore/keda --namespace keda --create-namespace

# kube-prometheus-stack
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm install prometheus prometheus-community/kube-prometheus-stack --namespace monitoring --create-namespace
```

## 上传模型到 S3

```bash
aws s3 cp ~/.cache/huggingface/hub/models--HelloKKMe--GTA1-7B/snapshots/701bedc80b447863bd60e3318ae44f6cbbfafd78/ s3://pwmmodel/GTA1-7B/ --recursive
```

## 部署

```bash
kubectl apply -f deploy/aws-eks/
```

## 验证

```bash
# 检查 Pod 状态
kubectl -n gta1 get pods -w

# 检查模型是否加载完成
kubectl -n gta1 logs -f deployment/gta1-vllm

# 测试推理
kubectl -n gta1 port-forward svc/gta1-vllm 8000:8000
python3 tools/test.py tools/test_files/mail.jpg "pc chat button"

# 查看 Grafana（kube-prometheus-stack 自带）
kubectl -n monitoring port-forward svc/prometheus-grafana 3000:80
# 浏览器访问 http://localhost:3000 (admin/prom-operator)
```

## 文件说明

| 文件 | 说明 |
|---|---|
| `00-namespace.yaml` | namespace |
| `01-s3-pv.yaml` | S3 PV/PVC（挂载模型 bucket）|
| `10-servicemonitor.yaml` | Prometheus 采集配置 |
| `20-deployment.yaml` | vLLM Deployment |
| `21-service.yaml` | ClusterIP Service |
| `30-scaledobject.yaml` | KEDA 自动扩缩容 |
| `40-pdb.yaml` | Pod Disruption Budget |
| `50-grafana-dashboard.yaml` | Grafana dashboard（自动被 kube-prometheus-stack 的 sidecar 加载）|
