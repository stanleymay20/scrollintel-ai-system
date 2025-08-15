#!/bin/bash

# ================================
# ScrollIntel Complete Production Deployment
# Orchestrates the entire production deployment process
# ================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
DEPLOYMENT_ID=$(date +%Y%m%d_%H%M%S)
DEPLOYMENT_TYPE="${DEPLOYMENT_TYPE:-full}"
ENABLE_BLUE_GREEN="${ENABLE_BLUE_GREEN:-true}"
ENABLE_AUTO_SCALING="${ENABLE_AUTO_SCALING:-true}"
ENABLE_DB_REPLICATION="${ENABLE_DB_REPLICATION:-true}"
ENABLE_MONITORING="${ENABLE_MONITORING:-true}"
SKIP_VALIDATION="${SKIP_VALIDATION:-false}"
DRY_RUN="${DRY_RUN:-false}"

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

print_banner() {
    echo ""
    echo "=================================="
    echo "🚀 ScrollIntel Production Deployment"
    echo "=================================="
    echo "Deployment ID: $DEPLOYMENT_ID"
    echo "Type: $DEPLOYMENT_TYPE"
    echo "Blue-Green: $ENABLE_BLUE_GREEN"
    echo "Auto-scaling: $ENABLE_AUTO_SCALING"
    echo "DB Replication: $ENABLE_DB_REPLICATION"
    echo "Monitoring: $ENABLE_MONITORING"
    echo "=================================="
    echo ""
}

check_prerequisites() {
    log_step "Checking prerequisites..."
    
    # Check required commands
    local required_commands=("docker" "docker-compose" "python3" "curl")
    for cmd in "${required_commands[@]}"; do
        if ! command -v "$cmd" &> /dev/null; then
            log_error "Required command '$cmd' not found"
            exit 1
        fi
    done
    
    # Check required environment variables
    local required_vars=("POSTGRES_PASSWORD" "JWT_SECRET_KEY" "OPENAI_API_KEY")
    for var in "${required_vars[@]}"; do
        if [ -z "${!var}" ]; then
            log_error "Required environment variable $var is not set"
            exit 1
        fi
    done
    
    # Check Docker daemon
    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running"
        exit 1
    fi
    
    log_info "✅ Prerequisites check passed"
}

setup_directories() {
    log_step "Setting up directories..."
    
    local directories=(
        "logs"
        "logs/nginx"
        "backups"
        "deployments/reports"
        "deployments/validation"
        "nginx/ssl"
        "postgres-config/master"
        "postgres-config/replica"
        "monitoring"
    )
    
    for dir in "${directories[@]}"; do
        mkdir -p "$dir"
    done
    
    log_info "✅ Directories created"
}

run_pre_deployment_tests() {
    log_step "Running pre-deployment tests..."
    
    if [ "$SKIP_VALIDATION" = "true" ]; then
        log_warn "Skipping pre-deployment tests as requested"
        return 0
    fi
    
    # Run unit tests
    log_info "Running unit tests..."
    if ! python -m pytest tests/ -v --tb=short --maxfail=5; then
        log_error "Unit tests failed"
        return 1
    fi
    
    # Run integration tests
    log_info "Running integration tests..."
    if ! python -m pytest tests/integration/ -v --tb=short --maxfail=3; then
        log_error "Integration tests failed"
        return 1
    fi
    
    log_info "✅ Pre-deployment tests passed"
}

deploy_infrastructure() {
    log_step "Deploying production infrastructure..."
    
    if [ "$DRY_RUN" = "true" ]; then
        log_info "DRY RUN: Would deploy infrastructure with configuration:"
        log_info "  Backend instances: ${BACKEND_INSTANCES:-3}"
        log_info "  Auto-scaling: $ENABLE_AUTO_SCALING"
        log_info "  DB replication: $ENABLE_DB_REPLICATION"
        log_info "  Monitoring: $ENABLE_MONITORING"
        return 0
    fi
    
    # Deploy using the production infrastructure script
    if ! python scripts/production-infrastructure-deploy.py --type "$DEPLOYMENT_TYPE"; then
        log_error "Infrastructure deployment failed"
        return 1
    fi
    
    log_info "✅ Infrastructure deployed successfully"
}

deploy_application() {
    log_step "Deploying application..."
    
    if [ "$DRY_RUN" = "true" ]; then
        log_info "DRY RUN: Would deploy application using strategy: $DEPLOYMENT_TYPE"
        return 0
    fi
    
    if [ "$ENABLE_BLUE_GREEN" = "true" ] && [ "$DEPLOYMENT_TYPE" = "full" ]; then
        log_info "Using blue-green deployment strategy..."
        if ! python scripts/blue-green-deploy.py; then
            log_error "Blue-green deployment failed"
            return 1
        fi
    else
        log_info "Using standard deployment strategy..."
        if ! bash scripts/production-deploy.sh; then
            log_error "Standard deployment failed"
            return 1
        fi
    fi
    
    log_info "✅ Application deployed successfully"
}

setup_auto_scaling() {
    if [ "$ENABLE_AUTO_SCALING" != "true" ]; then
        log_info "Auto-scaling disabled, skipping setup"
        return 0
    fi
    
    log_step "Setting up auto-scaling..."
    
    if [ "$DRY_RUN" = "true" ]; then
        log_info "DRY RUN: Would start auto-scaling manager"
        return 0
    fi
    
    # Start auto-scaling manager in background
    nohup python scripts/auto-scaling-manager.py > logs/auto-scaling.log 2>&1 &
    echo $! > logs/auto-scaling.pid
    
    log_info "✅ Auto-scaling manager started (PID: $(cat logs/auto-scaling.pid))"
}

run_health_checks() {
    log_step "Running health checks..."
    
    if [ "$DRY_RUN" = "true" ]; then
        log_info "DRY RUN: Would run comprehensive health checks"
        return 0
    fi
    
    # Wait for services to stabilize
    log_info "Waiting for services to stabilize..."
    sleep 30
    
    # Run deployment validation
    if ! python scripts/validate-deployment.py --wait 30; then
        log_error "Health checks failed"
        return 1
    fi
    
    log_info "✅ Health checks passed"
}

setup_monitoring_alerts() {
    if [ "$ENABLE_MONITORING" != "true" ]; then
        log_info "Monitoring disabled, skipping alert setup"
        return 0
    fi
    
    log_step "Setting up monitoring and alerts..."
    
    if [ "$DRY_RUN" = "true" ]; then
        log_info "DRY RUN: Would configure monitoring alerts"
        return 0
    fi
    
    # Start health check monitor
    nohup python scripts/health-check-monitor.py > logs/health-monitor.log 2>&1 &
    echo $! > logs/health-monitor.pid
    
    log_info "✅ Monitoring and alerts configured"
}

create_deployment_report() {
    log_step "Creating deployment report..."
    
    local report_file="deployments/reports/deployment_${DEPLOYMENT_ID}.json"
    
    cat > "$report_file" << EOF
{
  "deployment_id": "$DEPLOYMENT_ID",
  "timestamp": "$(date -Iseconds)",
  "deployment_type": "$DEPLOYMENT_TYPE",
  "configuration": {
    "blue_green_enabled": $ENABLE_BLUE_GREEN,
    "auto_scaling_enabled": $ENABLE_AUTO_SCALING,
    "db_replication_enabled": $ENABLE_DB_REPLICATION,
    "monitoring_enabled": $ENABLE_MONITORING,
    "backend_instances": ${BACKEND_INSTANCES:-3}
  },
  "services": {
    "application": "deployed",
    "load_balancer": "deployed",
    "database": "$([ "$ENABLE_DB_REPLICATION" = "true" ] && echo "replicated" || echo "single-instance")",
    "monitoring": "$([ "$ENABLE_MONITORING" = "true" ] && echo "enabled" || echo "disabled")",
    "auto_scaling": "$([ "$ENABLE_AUTO_SCALING" = "true" ] && echo "enabled" || echo "disabled")"
  },
  "endpoints": {
    "application": "http://localhost",
    "health_check": "http://localhost:8080/health",
    "monitoring": "$([ "$ENABLE_MONITORING" = "true" ] && echo "http://localhost:3001" || echo "null")"
  },
  "status": "completed"
}
EOF
    
    log_info "✅ Deployment report created: $report_file"
}

print_deployment_summary() {
    echo ""
    echo "=================================="
    echo "🎉 DEPLOYMENT COMPLETED SUCCESSFULLY"
    echo "=================================="
    echo "Deployment ID: $DEPLOYMENT_ID"
    echo "Type: $DEPLOYMENT_TYPE"
    echo "Timestamp: $(date)"
    echo ""
    echo "🔗 Access Points:"
    echo "  • Application: http://localhost"
    echo "  • Health Check: http://localhost:8080/health"
    if [ "$ENABLE_MONITORING" = "true" ]; then
        echo "  • Monitoring: http://localhost:3001"
    fi
    echo ""
    echo "📊 Infrastructure:"
    echo "  • Load Balancing: ✅ Enabled"
    echo "  • Auto-scaling: $([ "$ENABLE_AUTO_SCALING" = "true" ] && echo "✅ Enabled" || echo "❌ Disabled")"
    echo "  • DB Replication: $([ "$ENABLE_DB_REPLICATION" = "true" ] && echo "✅ Enabled" || echo "❌ Disabled")"
    echo "  • Blue-Green Deploy: $([ "$ENABLE_BLUE_GREEN" = "true" ] && echo "✅ Enabled" || echo "❌ Disabled")"
    echo "  • Monitoring: $([ "$ENABLE_MONITORING" = "true" ] && echo "✅ Enabled" || echo "❌ Disabled")"
    echo ""
    echo "📋 Next Steps:"
    echo "  1. Monitor application logs and metrics"
    echo "  2. Configure DNS to point to production"
    echo "  3. Set up backup schedules"
    echo "  4. Configure alerting notifications"
    echo "  5. Run smoke tests on production"
    echo ""
    echo "🔧 Management Commands:"
    echo "  • View logs: docker-compose -f docker-compose.load-balanced.yml logs -f"
    echo "  • Scale services: docker-compose -f docker-compose.load-balanced.yml up -d --scale backend-1=N"
    echo "  • Health check: python scripts/validate-deployment.py"
    echo "  • Stop auto-scaling: kill \$(cat logs/auto-scaling.pid)"
    echo "=================================="
}

cleanup_on_failure() {
    log_error "Deployment failed, cleaning up..."
    
    # Stop auto-scaling if running
    if [ -f "logs/auto-scaling.pid" ]; then
        kill "$(cat logs/auto-scaling.pid)" 2>/dev/null || true
        rm -f logs/auto-scaling.pid
    fi
    
    # Stop health monitor if running
    if [ -f "logs/health-monitor.pid" ]; then
        kill "$(cat logs/health-monitor.pid)" 2>/dev/null || true
        rm -f logs/health-monitor.pid
    fi
    
    log_info "Cleanup completed"
}

main() {
    # Set up error handling
    trap cleanup_on_failure ERR
    
    print_banner
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --type)
                DEPLOYMENT_TYPE="$2"
                shift 2
                ;;
            --dry-run)
                DRY_RUN="true"
                shift
                ;;
            --skip-validation)
                SKIP_VALIDATION="true"
                shift
                ;;
            --no-blue-green)
                ENABLE_BLUE_GREEN="false"
                shift
                ;;
            --no-auto-scaling)
                ENABLE_AUTO_SCALING="false"
                shift
                ;;
            --no-monitoring)
                ENABLE_MONITORING="false"
                shift
                ;;
            --help)
                echo "Usage: $0 [OPTIONS]"
                echo "Options:"
                echo "  --type TYPE           Deployment type (full, minimal, update)"
                echo "  --dry-run            Show what would be deployed without deploying"
                echo "  --skip-validation    Skip pre-deployment tests"
                echo "  --no-blue-green      Disable blue-green deployment"
                echo "  --no-auto-scaling    Disable auto-scaling"
                echo "  --no-monitoring      Disable monitoring"
                echo "  --help               Show this help message"
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    # Execute deployment steps
    check_prerequisites
    setup_directories
    run_pre_deployment_tests
    deploy_infrastructure
    deploy_application
    setup_auto_scaling
    run_health_checks
    setup_monitoring_alerts
    create_deployment_report
    
    if [ "$DRY_RUN" = "true" ]; then
        echo ""
        echo "=================================="
        echo "🔍 DRY RUN COMPLETED"
        echo "=================================="
        echo "No actual deployment was performed."
        echo "Run without --dry-run to execute the deployment."
    else
        print_deployment_summary
    fi
}

# Execute main function
main "$@"