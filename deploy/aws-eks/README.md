# GTA1 vLLM - AWS EKS 部署

## 设置环境变量

```bash
export CLUSTER_NAME=vision
export REGION=us-east-1
export ACCOUNT_ID=527432981953
export NODE_ROLE_ARN=arn:aws:iam::${ACCOUNT_ID}:role/ec2-common-role
export S3_CSI_ROLE_NAME=AmazonEKS_S3_CSI_DriverRole
export NODEGROUP_NAME=gpu-g5
export INSTANCE_TYPE=g5.xlarge
```

## 创建 EKS 集群

```bash
eksctl create cluster \
  --name $CLUSTER_NAME \
  --region $REGION \
  --version 1.35 \
  --without-nodegroup
```

创建 OIDC provider 和 S3 CSI Driver IAM Role：

```bash
eksctl utils associate-iam-oidc-provider --cluster $CLUSTER_NAME --region $REGION --approve

eksctl create iamserviceaccount \
  --name s3-csi-driver-sa \
  --namespace kube-system \
  --cluster $CLUSTER_NAME \
  --attach-policy-arn arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess \
  --approve \
  --role-name $S3_CSI_ROLE_NAME \
  --region $REGION \
  --role-only
```

安装 S3 CSI Driver addon：

```bash
eksctl create addon --name aws-mountpoint-s3-csi-driver --cluster $CLUSTER_NAME \
  --service-account-role-arn arn:aws:iam::${ACCOUNT_ID}:role/${S3_CSI_ROLE_NAME} \
  --force --region $REGION
```

## 创建 GPU 节点组

获取集群子网：

```bash
SUBNETS=$(aws eks describe-cluster --name $CLUSTER_NAME --region $REGION \
  --query 'cluster.resourcesVpcConfig.subnetIds' --output text | tr '\t' ',')
```

创建节点组（含 Warm Pool）：

```bash
aws eks create-nodegroup \
  --cluster-name $CLUSTER_NAME \
  --nodegroup-name $NODEGROUP_NAME \
  --node-role $NODE_ROLE_ARN \
  --subnets $(echo $SUBNETS | tr ',' ' ') \
  --region $REGION \
  --scaling-config minSize=1,maxSize=3,desiredSize=1 \
  --ami-type AL2023_x86_64_NVIDIA \
  --instance-types $INSTANCE_TYPE \
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

# Cluster Autoscaler（根据 Pending Pod 自动扩缩节点）
eksctl create iamserviceaccount \
  --name cluster-autoscaler \
  --namespace kube-system \
  --cluster $CLUSTER_NAME \
  --region $REGION \
  --attach-policy-arn arn:aws:iam::aws:policy/AutoScalingFullAccess \
  --approve \
  --override-existing-serviceaccounts

CA_ROLE_ARN=$(aws cloudformation describe-stacks --region $REGION \
  --query 'Stacks[?contains(StackName, `cluster-autoscaler`)].Outputs[0].OutputValue' --output text)

helm repo add autoscaler https://kubernetes.github.io/autoscaler
helm install cluster-autoscaler autoscaler/cluster-autoscaler \
  --namespace kube-system \
  --set autoDiscovery.clusterName=$CLUSTER_NAME \
  --set awsRegion=$REGION \
  --set rbac.serviceAccount.name=cluster-autoscaler \
  --set rbac.serviceAccount.annotations."eks\.amazonaws\.com/role-arn"=$CA_ROLE_ARN \
  --set extraArgs.skip-nodes-with-local-storage=false \
  --set extraArgs.balance-similar-node-groups=true \
  --set extraArgs.expander=least-waste \
  --set extraArgs.scale-down-unneeded-time=5m \
  --set extraArgs.scale-down-delay-after-add=5m
```

## 上传模型到 S3

```bash
aws s3 cp ~/.cache/huggingface/hub/models--HelloKKMe--GTA1-7B/snapshots/701bedc80b447863bd60e3318ae44f6cbbfafd78/ s3://pwmmodel/GTA1-7B/ --recursive
```

## 部署

```bash
kubectl apply -f deploy/aws-eks/
```

## 预热 Warm Pool 节点

首次部署后 Warm Pool 节点没有镜像缓存，扩容时需要拉取 ~9.6GB 镜像（约 4m30s）。通过临时扩容让新节点拉完镜像，缩容后节点进入 Warm Pool（Stopped），EBS 上的镜像缓存保留，后续扩容跳过拉镜像。

```bash
# 1. 临时调大最小副本数，触发新 Pod 调度到新节点拉取镜像
kubectl -n gta1 patch scaledobject gta1-vllm --type merge \
  -p '{"spec":{"minReplicaCount":2}}'

# 2. 等待新 Pod Ready（镜像拉取 + 模型加载）
kubectl -n gta1 get pods -w

# 3. 改回 1，KEDA 自动缩掉多余 Pod，节点回到 Warm Pool
kubectl -n gta1 patch scaledobject gta1-vllm --type merge \
  -p '{"spec":{"minReplicaCount":1}}'
```

## 验证

```bash
# 检查 Pod 状态
kubectl -n gta1 get pods -w

# 检查模型是否加载完成
kubectl -n gta1 logs -f deployment/gta1-vllm

# 测试推理
kubectl -n gta1 port-forward svc/gta1-vllm 8000:8000
python3 tools/test.py tools/test_files/mail.jpg "chat button"

# 查看 Grafana（kube-prometheus-stack 自带）
kubectl -n monitoring port-forward svc/prometheus-grafana 3000:80
# 浏览器访问 http://localhost:3000 (admin/prom-operator)
```

## 压测与扩缩容观察

扩缩容配置（`30-scaledobject.yaml`）：
- 触发指标：`sum(gta1_inflight_requests) > 2` 时扩容
- 扩容：无稳定窗口，每 30s 最多扩 1 个 Pod
- 缩容：稳定窗口 300s，每 300s 最多缩 1 个 Pod
- 副本范围：1 ~ 3

### 1. 开启监控终端

```bash
# 终端1：观察 Pod 扩缩
kubectl -n gta1 get pods -w

# 终端2：观察节点变化和 Ready 时间
kubectl get nodes -w

# 终端3：观察 HPA 状态
kubectl -n gta1 get hpa -w

# 终端4：观察 Warm Pool 节点启动时间
aws eks describe-nodegroup --cluster-name $CLUSTER_NAME --nodegroup-name $NODEGROUP_NAME \
  --region $REGION --query 'nodegroup.{desired:scalingConfig.desiredSize,min:scalingConfig.minSize,max:scalingConfig.maxSize}'
```

### 2. 执行压测

```bash
kubectl -n gta1 port-forward svc/gta1-vllm 8000:8000
```

恒定速率模式（推荐用于观察扩缩容）：

```bash
# 10 rps 持续 5 分钟，持续产生 inflight 压力触发扩容
python3 tools/bench.py tools/test_files/mail.jpg "chat button" --rate 10 --duration 300
```

突发模式：

```bash
# 8 并发共 32 请求
python3 tools/bench.py tools/test_files/mail.jpg "chat button" -c 8 -n 32

# 并发阶梯扫描
python3 tools/bench.py tools/test_files/mail.jpg "chat button" -n 32 --sweep
```

### 3. 统计新节点 Ready 时间

```bash
# 查看节点从创建到 Ready 的耗时
kubectl get nodes -o json | jq -r '
  .items[] |
  .metadata.name as $name |
  (.metadata.creationTimestamp | fromdateiso8601) as $created |
  (.status.conditions[] | select(.type=="Ready" and .status=="True") |
   .lastTransitionTime | fromdateiso8601) as $ready |
  "\($name)\t\($ready - $created)s"'
```

### 4. 观察缩容

停止压测后等待约 5~10 分钟，观察 Pod 缩回 1 副本、Warm Pool 节点回收：

```bash
# 确认 Pod 已缩容
kubectl -n gta1 get pods

# 确认 Warm Pool 状态
aws autoscaling describe-auto-scaling-groups --region $REGION \
  --query 'AutoScalingGroups[?contains(Tags[?Key==`eks:nodegroup-name`].Value, `'$NODEGROUP_NAME'`)].{Name:AutoScalingGroupName,Desired:DesiredCapacity,WarmPool:WarmPoolConfiguration}' \
  --output table
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
