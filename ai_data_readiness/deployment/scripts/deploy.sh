#!/bin/bash

# AI Data Readiness Platform Deployment Script
set -e

# Configuration
ENVIRONMENT=${ENVIRONMENT:-production}
NAMESPACE="ai-data-readiness"
IMAGE_TAG=${IMAGE_TAG:-latest}
REGISTRY=${REGISTRY:-your-registry.com}

echo "🚀 Starting AI Data Readiness Platform deployment..."
echo "Environment: $ENVIRONMENT"
echo "Namespace: $NAMESPACE"
echo "Image Tag: $IMAGE_TAG"

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check required tools
echo "📋 Checking required tools..."
for tool in docker kubectl helm; do
    if ! command_exists $tool; then
        echo "❌ $tool is not installed. Please install it first."
        exit 1
    fi
done
echo "✅ All required tools are available"

# Build Docker image
echo "🔨 Building Docker image..."
docker build -t $REGISTRY/ai-data-readiness:$IMAGE_TAG -f deployment/Dockerfile ../..

# Push to registry
echo "📤 Pushing image to registry..."
docker push $REGISTRY/ai-data-readiness:$IMAGE_TAG

# Create namespace if it doesn't exist
echo "🏗️ Creating namespace..."
kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -

# Apply Kubernetes manifests
echo "🚀 Deploying to Kubernetes..."
kubectl apply -f kubernetes/configmap.yaml
kubectl apply -f kubernetes/postgres.yaml
kubectl apply -f kubernetes/redis.yaml

# Wait for database to be ready
echo "⏳ Waiting for database to be ready..."
kubectl wait --for=condition=ready pod -l app=postgres -n $NAMESPACE --timeout=300s

# Run database migrations
echo "🗄️ Running database migrations..."
kubectl run migration-job --image=$REGISTRY/ai-data-readiness:$IMAGE_TAG \
    --restart=Never \
    --namespace=$NAMESPACE \
    --command -- python -m alembic upgrade head

# Deploy application
echo "🚀 Deploying application..."
sed "s|IMAGE_PLACEHOLDER|$REGISTRY/ai-data-readiness:$IMAGE_TAG|g" kubernetes/deployment.yaml | kubectl apply -f -

# Wait for deployment to be ready
echo "⏳ Waiting for deployment to be ready..."
kubectl wait --for=condition=available deployment/ai-data-readiness-app -n $NAMESPACE --timeout=300s

# Get service information
echo "📊 Deployment information:"
kubectl get pods,services,ingress -n $NAMESPACE

echo "✅ Deployment completed successfully!"
echo "🌐 Access the application at: https://ai-data-readiness.example.com"