#!/bin/bash

# AI Data Readiness Platform Health Check Script
set -e

NAMESPACE="ai-data-readiness"
SERVICE_URL=${SERVICE_URL:-"http://localhost:8000"}

echo "🏥 Starting health check for AI Data Readiness Platform..."

# Function to check HTTP endpoint
check_http_endpoint() {
    local url=$1
    local expected_status=${2:-200}
    
    echo "🔍 Checking $url..."
    
    response=$(curl -s -o /dev/null -w "%{http_code}" "$url" || echo "000")
    
    if [ "$response" = "$expected_status" ]; then
        echo "✅ $url is healthy (HTTP $response)"
        return 0
    else
        echo "❌ $url is unhealthy (HTTP $response)"
        return 1
    fi
}

# Function to check Kubernetes pods
check_k8s_pods() {
    echo "🔍 Checking Kubernetes pods..."
    
    # Check if all pods are running
    not_running=$(kubectl get pods -n $NAMESPACE --field-selector=status.phase!=Running --no-headers 2>/dev/null | wc -l)
    
    if [ "$not_running" -eq 0 ]; then
        echo "✅ All pods are running"
        kubectl get pods -n $NAMESPACE
        return 0
    else
        echo "❌ Some pods are not running:"
        kubectl get pods -n $NAMESPACE --field-selector=status.phase!=Running
        return 1
    fi
}

# Function to check database connectivity
check_database() {
    echo "🔍 Checking database connectivity..."
    
    # Get database pod
    db_pod=$(kubectl get pods -n $NAMESPACE -l app=postgres -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")
    
    if [ -z "$db_pod" ]; then
        echo "❌ Database pod not found"
        return 1
    fi
    
    # Test database connection
    if kubectl exec -n $NAMESPACE $db_pod -- pg_isready -U postgres >/dev/null 2>&1; then
        echo "✅ Database is accessible"
        return 0
    else
        echo "❌ Database is not accessible"
        return 1
    fi
}

# Function to check Redis connectivity
check_redis() {
    echo "🔍 Checking Redis connectivity..."
    
    # Get Redis pod
    redis_pod=$(kubectl get pods -n $NAMESPACE -l app=redis -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")
    
    if [ -z "$redis_pod" ]; then
        echo "❌ Redis pod not found"
        return 1
    fi
    
    # Test Redis connection
    if kubectl exec -n $NAMESPACE $redis_pod -- redis-cli ping | grep -q PONG; then
        echo "✅ Redis is accessible"
        return 0
    else
        echo "❌ Redis is not accessible"
        return 1
    fi
}

# Main health check
main() {
    local exit_code=0
    
    # Check if running in Kubernetes
    if kubectl cluster-info >/dev/null 2>&1; then
        echo "🎯 Running health checks in Kubernetes mode..."
        
        check_k8s_pods || exit_code=1
        check_database || exit_code=1
        check_redis || exit_code=1
        
        # Get service URL for Kubernetes
        SERVICE_URL=$(kubectl get service ai-data-readiness-service -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].hostname}' 2>/dev/null || echo "")
        if [ -n "$SERVICE_URL" ]; then
            SERVICE_URL="http://$SERVICE_URL"
        else
            echo "⚠️ Service URL not available, skipping HTTP checks"
        fi
    else
        echo "🎯 Running health checks in standalone mode..."
    fi
    
    # HTTP health checks
    if [ -n "$SERVICE_URL" ]; then
        check_http_endpoint "$SERVICE_URL/health" || exit_code=1
        check_http_endpoint "$SERVICE_URL/api/v1/health" || exit_code=1
    fi
    
    # Summary
    if [ $exit_code -eq 0 ]; then
        echo "🎉 All health checks passed!"
    else
        echo "💥 Some health checks failed!"
    fi
    
    return $exit_code
}

# Run main function
main "$@"