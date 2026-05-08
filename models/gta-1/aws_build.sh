#!/bin/bash
set -e

IMAGE_TAG="${1:-latest}"
REGION="us-east-1"
PROJECT_NAME="gta1-vllm"
S3_BUCKET="codebuild-source-527432981953"
ECR_REPOSITORY="qira/gta1-vllm"
ACCOUNT_ID="527432981953"
ECR_REGISTRY="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"

setup() {
    echo "==> Creating S3 bucket..."
    aws s3 mb "s3://${S3_BUCKET}" --region "${REGION}" 2>/dev/null || true

    echo "==> Creating ECR repository..."
    aws ecr create-repository --repository-name "${ECR_REPOSITORY}" --region "${REGION}" 2>/dev/null || true

    echo "==> Creating IAM role..."
    aws iam create-role \
        --role-name codebuild-${PROJECT_NAME} \
        --assume-role-policy-document '{
            "Version":"2012-10-17",
            "Statement":[{"Effect":"Allow","Principal":{"Service":"codebuild.amazonaws.com"},"Action":"sts:AssumeRole"}]
        }' 2>/dev/null || true

    for policy in AmazonEC2ContainerRegistryPowerUser CloudWatchLogsFullAccess AmazonS3ReadOnlyAccess; do
        aws iam attach-role-policy \
            --role-name codebuild-${PROJECT_NAME} \
            --policy-arn "arn:aws:iam::aws:policy/${policy}" 2>/dev/null || true
    done

    sleep 5

    echo "==> Creating CodeBuild project..."
    aws codebuild create-project \
        --name "${PROJECT_NAME}" \
        --source '{"type":"S3","location":"'"${S3_BUCKET}"'/source.zip"}' \
        --artifacts '{"type":"NO_ARTIFACTS"}' \
        --environment '{
            "type":"LINUX_CONTAINER",
            "image":"aws/codebuild/standard:7.0",
            "computeType":"BUILD_GENERAL1_LARGE",
            "privilegedMode":true,
            "environmentVariables":[
                {"name":"ECR_REPOSITORY","value":"'"${ECR_REPOSITORY}"'","type":"PLAINTEXT"},
                {"name":"IMAGE_TAG","value":"latest","type":"PLAINTEXT"}
            ]
        }' \
        --service-role "arn:aws:iam::${ACCOUNT_ID}:role/codebuild-${PROJECT_NAME}" \
        --region "${REGION}" 2>/dev/null || true

    echo "==> Setup complete."
}

build() {
    SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

    echo "==> Packaging source..."
    cd "${SCRIPT_DIR}"
    zip -q /tmp/gta1-source.zip Dockerfile.vllm api_vllm.py buildspec.yml

    echo "==> Uploading to S3..."
    aws s3 cp /tmp/gta1-source.zip "s3://${S3_BUCKET}/source.zip" --region "${REGION}"

    echo "==> Starting build (tag: ${IMAGE_TAG})..."
    BUILD_ID=$(aws codebuild start-build \
        --project-name "${PROJECT_NAME}" \
        --environment-variables-override "name=IMAGE_TAG,value=${IMAGE_TAG},type=PLAINTEXT" \
        --region "${REGION}" \
        --query 'build.id' --output text)

    echo "==> Build started: ${BUILD_ID}"
    echo "==> Tailing logs..."
    aws codebuild batch-get-builds --ids "${BUILD_ID}" --region "${REGION}" \
        --query 'builds[0].logs.deepLink' --output text

    echo ""
    echo "Watch in console: https://${REGION}.console.aws.amazon.com/codesuite/codebuild/projects/${PROJECT_NAME}/build/${BUILD_ID}"
    echo ""
    echo "Or tail logs:"
    echo "  aws codebuild batch-get-builds --ids ${BUILD_ID} --query 'builds[0].buildStatus'"
}

case "${2:-build}" in
    setup) setup ;;
    build) build ;;
    *)     echo "Usage: $0 <tag> [setup|build]"; exit 1 ;;
esac
