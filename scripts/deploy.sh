#!/bin/bash

# Agent Steering System Deployment Script
# This script deploys the complete Agent Steering System to Kubernetes

set -euo pipefail

# Configuration
NAMESPACE="agent-steering-system"
ENVIRONMENT="${ENVIRONMENT:-production}"
KUBECTL_TIMEOUT="600s"
HEALTH_CHECK_RETRIES=30
HEALTH_CHECK_DELAY=10

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if kubectl is installed
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed or not in PATH"
        exit 1
    fi
    
    # Check if helm is installed
    if ! command -v helm &> /dev/null; then
        log_error "helm is not installed or not in PATH"
        exit 1
    fi
    
    # Check kubectl connectivity
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

# Create namespace
create_namespace() {
    log_info "Creating namespace: $NAMESPACE"
    
    if kubectl get namespace "$NAMESPACE" &> /dev/null; then
        log_warning "Namespace $NAMESPACE already exists"
    else
        kubectl apply -f k8s/namespace.yaml
        log_success "Namespace created successfully"
    fi
}

# Deploy secrets (with validation)
deploy_secrets() {
    log_info "Deploying secrets..."
    
    # Check if secrets file exists
    if [[ ! -f "k8s/secrets.yaml" ]]; then
        log_error "Secrets file not found: k8s/secrets.yaml"
        exit 1
    fi
    
    # Validate that secrets don't contain default values
    if grep -q "CHANGE_ME_IN_PRODUCTION" k8s/secrets.yaml; then
        log_error "Secrets file contains default values. Please update with production secrets."
        exit 1
    fi
    
    kubectl apply -f k8s/secrets.yaml
    log_success "Secrets deployed successfully"
}

# Deploy configuration
deploy_config() {
    log_info "Deploying configuration..."
    
    kubectl apply -f k8s/configmap.yaml
    kubectl apply -f monitoring/prometheus-config.yaml
    kubectl apply -f monitoring/grafana-config.yaml
    
    log_success "Configuration deployed successfully"
}

# Deploy RBAC
deploy_rbac() {
    log_info "Deploying RBAC configuration..."
    
    kubectl apply -f k8s/rbac.yaml
    
    log_success "RBAC deployed successfully"
}

# Deploy data services
deploy_data_services() {
    log_info "Deploying data services (PostgreSQL, Redis, Kafka)..."
    
    kubectl apply -f k8s/data-services.yaml
    kubectl apply -f k8s/kafka-cluster.yaml
    
    # Wait for data services to be ready
    log_info "Waiting for PostgreSQL to be ready..."
    kubectl wait --for=condition=ready pod -l app=postgresql -n "$NAMESPACE" --timeout="$KUBECTL_TIMEOUT"
    
    log_info "Waiting for Redis to be ready..."
    kubectl wait --for=condition=ready pod -l app=redis -n "$NAMESPACE" --timeout="$KUBECTL_TIMEOUT"
    
    log_info "Waiting for Kafka to be ready..."
    kubectl wait --for=condition=ready pod -l app=kafka -n "$NAMESPACE" --timeout="$KUBECTL_TIMEOUT"
    
    log_success "Data services deployed successfully"
}

# Deploy monitoring stack
deploy_monitoring() {
    log_info "Deploying monitoring stack (Prometheus, Grafana)..."
    
    kubectl apply -f k8s/monitoring-stack.yaml
    
    # Wait for monitoring services to be ready
    log_info "Waiting for Prometheus to be ready..."
    kubectl wait --for=condition=ready pod -l app=prometheus -n "$NAMESPACE" --timeout="$KUBECTL_TIMEOUT"
    
    log_info "Waiting for Grafana to be ready..."
    kubectl wait --for=condition=ready pod -l app=grafana -n "$NAMESPACE" --timeout="$KUBECTL_TIMEOUT"
    
    log_success "Monitoring stack deployed successfully"
}

# Deploy core applications
deploy_applications() {
    log_info "Deploying core applications..."
    
    # Deploy orchestration engine
    kubectl apply -f k8s/orchestration-deployment.yaml
    
    # Deploy intelligence engine
    kubectl apply -f k8s/intelligence-deployment.yaml
    
    # Wait for applications to be ready
    log_info "Waiting for Orchestration Engine to be ready..."
    kubectl wait --for=condition=ready pod -l app=orchestration-engine -n "$NAMESPACE" --timeout="$KUBECTL_TIMEOUT"
    
    log_info "Waiting for Intelligence Engine to be ready..."
    kubectl wait --for=condition=ready pod -l app=intelligence-engine -n "$NAMESPACE" --timeout="$KUBECTL_TIMEOUT"
    
    log_success "Core applications deployed successfully"
}

# Deploy ingress
deploy_ingress() {
    log_info "Deploying ingress configuration..."
    
    kubectl apply -f k8s/ingress.yaml
    
    log_success "Ingress deployed successfully"
}

# Health checks
run_health_checks() {
    log_info "Running health checks..."
    
    local retries=0
    local max_retries=$HEALTH_CHECK_RETRIES
    
    while [[ $retries -lt $max_retries ]]; do
        log_info "Health check attempt $((retries + 1))/$max_retries"
        
        # Check orchestration engine health
        if kubectl exec -n "$NAMESPACE" deployment/orchestration-engine -- curl -f http://localhost:8080/health/live &> /dev/null; then
            log_success "Orchestration Engine health check passed"
            break
        else
            log_warning "Orchestration Engine health check failed, retrying in ${HEALTH_CHECK_DELAY}s..."
            sleep $HEALTH_CHECK_DELAY
            ((retries++))
        fi
    done
    
    if [[ $retries -eq $max_retries ]]; then
        log_error "Orchestration Engine health checks failed after $max_retries attempts"
        return 1
    fi
    
    # Check intelligence engine health
    retries=0
    while [[ $retries -lt $max_retries ]]; do
        log_info "Intelligence Engine health check attempt $((retries + 1))/$max_retries"
        
        if kubectl exec -n "$NAMESPACE" deployment/intelligence-engine -- curl -f http://localhost:8081/health/live &> /dev/null; then
            log_success "Intelligence Engine health check passed"
            break
        else
            log_warning "Intelligence Engine health check failed, retrying in ${HEALTH_CHECK_DELAY}s..."
            sleep $HEALTH_CHECK_DELAY
            ((retries++))
        fi
    done
    
    if [[ $retries -eq $max_retries ]]; then
        log_error "Intelligence Engine health checks failed after $max_retries attempts"
        return 1
    fi
    
    log_success "All health checks passed"
}

# Smoke tests
run_smoke_tests() {
    log_info "Running smoke tests..."
    
    # Test orchestration API
    if kubectl exec -n "$NAMESPACE" deployment/orchestration-engine -- curl -f http://localhost:8080/api/v1/health &> /dev/null; then
        log_success "Orchestration API smoke test passed"
    else
        log_error "Orchestration API smoke test failed"
        return 1
    fi
    
    # Test intelligence API
    if kubectl exec -n "$NAMESPACE" deployment/intelligence-engine -- curl -f http://localhost:8081/api/v1/health &> /dev/null; then
        log_success "Intelligence API smoke test passed"
    else
        log_error "Intelligence API smoke test failed"
        return 1
    fi
    
    log_success "All smoke tests passed"
}

# Display deployment status
show_deployment_status() {
    log_info "Deployment Status:"
    echo
    
    # Show pod status
    echo "Pod Status:"
    kubectl get pods -n "$NAMESPACE" -o wide
    echo
    
    # Show service status
    echo "Service Status:"
    kubectl get services -n "$NAMESPACE"
    echo
    
    # Show ingress status
    echo "Ingress Status:"
    kubectl get ingress -n "$NAMESPACE"
    echo
    
    # Show HPA status
    echo "HPA Status:"
    kubectl get hpa -n "$NAMESPACE"
    echo
}

# Rollback function
rollback_deployment() {
    log_warning "Rolling back deployment..."
    
    # Rollback deployments
    kubectl rollout undo deployment/orchestration-engine -n "$NAMESPACE" || true
    kubectl rollout undo deployment/intelligence-engine -n "$NAMESPACE" || true
    
    log_info "Rollback completed"
}

# Cleanup function
cleanup() {
    if [[ $? -ne 0 ]]; then
        log_error "Deployment failed. Check the logs above for details."
        
        # Show recent events
        log_info "Recent events:"
        kubectl get events -n "$NAMESPACE" --sort-by='.lastTimestamp' | tail -20
        
        # Ask for rollback
        read -p "Do you want to rollback the deployment? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rollback_deployment
        fi
    fi
}

# Main deployment function
main() {
    log_info "Starting Agent Steering System deployment..."
    log_info "Environment: $ENVIRONMENT"
    log_info "Namespace: $NAMESPACE"
    echo
    
    # Set trap for cleanup
    trap cleanup EXIT
    
    # Run deployment steps
    check_prerequisites
    create_namespace
    deploy_secrets
    deploy_config
    deploy_rbac
    deploy_data_services
    deploy_monitoring
    deploy_applications
    deploy_ingress
    
    # Run validation
    run_health_checks
    run_smoke_tests
    
    # Show status
    show_deployment_status
    
    log_success "Agent Steering System deployment completed successfully!"
    echo
    log_info "Access URLs:"
    log_info "  API: https://api.agent-steering.scrollintel.com"
    log_info "  Monitoring: https://monitoring.agent-steering.scrollintel.com"
    echo
    log_info "To check the status: kubectl get all -n $NAMESPACE"
    log_info "To view logs: kubectl logs -f deployment/orchestration-engine -n $NAMESPACE"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --environment|-e)
            ENVIRONMENT="$2"
            shift 2
            ;;
        --namespace|-n)
            NAMESPACE="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  -e, --environment    Environment (production, staging, development)"
            echo "  -n, --namespace      Kubernetes namespace"
            echo "  -h, --help          Show this help message"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Run main function
main "$@"