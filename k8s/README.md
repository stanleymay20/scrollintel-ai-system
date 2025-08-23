# Agent Steering System - Kubernetes Deployment

This directory contains the complete Kubernetes deployment configuration for the Agent Steering System, providing enterprise-grade AI orchestration capabilities that surpass platforms like Palantir.

## Architecture Overview

The Agent Steering System is deployed as a cloud-native, microservices architecture with the following components:

### Core Services
- **Orchestration Engine**: Coordinates AI agents and manages task distribution
- **Intelligence Engine**: Provides decision-making and ML inference capabilities
- **Agent Registry**: Manages agent lifecycle and capabilities
- **Real-time Messaging**: Kafka-based event streaming

### Data Layer
- **PostgreSQL**: Primary database for persistent data
- **Redis**: High-performance caching and session storage
- **Kafka**: Message queuing and event streaming

### Monitoring Stack
- **Prometheus**: Metrics collection and alerting
- **Grafana**: Visualization and dashboards
- **Node Exporter**: System metrics collection

## Prerequisites

### Required Tools
- `kubectl` (v1.27+)
- `helm` (v3.10+)
- `docker` (v20.10+)
- Access to a Kubernetes cluster (v1.27+)

### Cluster Requirements
- **Minimum**: 6 vCPUs, 16GB RAM, 100GB storage
- **Recommended**: 12 vCPUs, 32GB RAM, 500GB storage
- **Production**: 24+ vCPUs, 64GB+ RAM, 1TB+ storage

### Node Requirements
- Core nodes: `m6i.2xlarge` or equivalent (8 vCPU, 32GB RAM)
- Intelligence nodes: `g5.2xlarge` or equivalent (8 vCPU, 32GB RAM, GPU)
- Monitoring nodes: `m6i.xlarge` or equivalent (4 vCPU, 16GB RAM)

## Quick Start

### 1. Clone and Configure

```bash
git clone <repository-url>
cd agent-steering-system
```

### 2. Update Secrets

**CRITICAL**: Update the secrets file with production values:

```bash
# Edit k8s/secrets.yaml and replace all CHANGE_ME_IN_PRODUCTION values
vim k8s/secrets.yaml
```

Required secrets to update:
- Database passwords
- Redis passwords
- JWT keys (generate with `openssl genrsa -out private.pem 2048`)
- API keys for external services
- Encryption keys (32-character random string)

### 3. Deploy the System

```bash
# Make the deployment script executable (Linux/Mac)
chmod +x scripts/deploy.sh

# Run the deployment
./scripts/deploy.sh --environment production
```

### 4. Validate Deployment

```bash
# Run validation checks
python3 scripts/validate-deployment.py --namespace agent-steering-system

# Run health checks
python3 scripts/health-check.py --environment production

# Run smoke tests
python3 scripts/smoke-tests.py --environment production
```

## Manual Deployment Steps

If you prefer to deploy manually or need to troubleshoot:

### 1. Create Namespace
```bash
kubectl apply -f k8s/namespace.yaml
```

### 2. Deploy Secrets and Configuration
```bash
kubectl apply -f k8s/secrets.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f monitoring/prometheus-config.yaml
kubectl apply -f monitoring/grafana-config.yaml
```

### 3. Deploy RBAC
```bash
kubectl apply -f k8s/rbac.yaml
```

### 4. Deploy Data Services
```bash
kubectl apply -f k8s/data-services.yaml
kubectl apply -f k8s/kafka-cluster.yaml

# Wait for data services to be ready
kubectl wait --for=condition=ready pod -l app=postgresql -n agent-steering-system --timeout=600s
kubectl wait --for=condition=ready pod -l app=redis -n agent-steering-system --timeout=600s
kubectl wait --for=condition=ready pod -l app=kafka -n agent-steering-system --timeout=600s
```

### 5. Deploy Monitoring
```bash
kubectl apply -f k8s/monitoring-stack.yaml

# Wait for monitoring to be ready
kubectl wait --for=condition=ready pod -l app=prometheus -n agent-steering-system --timeout=600s
kubectl wait --for=condition=ready pod -l app=grafana -n agent-steering-system --timeout=600s
```

### 6. Deploy Core Applications
```bash
kubectl apply -f k8s/orchestration-deployment.yaml
kubectl apply -f k8s/intelligence-deployment.yaml

# Wait for applications to be ready
kubectl wait --for=condition=ready pod -l app=orchestration-engine -n agent-steering-system --timeout=600s
kubectl wait --for=condition=ready pod -l app=intelligence-engine -n agent-steering-system --timeout=600s
```

### 7. Deploy Ingress
```bash
kubectl apply -f k8s/ingress.yaml
```

## Configuration

### Environment Variables

Key configuration options in `k8s/configmap.yaml`:

- `LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)
- `MAX_WORKERS`: Number of worker processes
- `AGENT_REGISTRY_TTL`: Agent registration timeout
- `CACHE_TTL`: Cache timeout in seconds
- `MAX_CONCURRENT_TASKS`: Maximum concurrent task limit

### Resource Limits

Default resource allocations:

| Component | CPU Request | CPU Limit | Memory Request | Memory Limit |
|-----------|-------------|-----------|----------------|--------------|
| Orchestration | 1000m | 2000m | 2Gi | 4Gi |
| Intelligence | 2000m | 4000m | 4Gi | 8Gi |
| PostgreSQL | 1000m | 2000m | 2Gi | 4Gi |
| Redis | 500m | 1000m | 1Gi | 2Gi |
| Prometheus | 1000m | 2000m | 2Gi | 4Gi |
| Grafana | 500m | 1000m | 1Gi | 2Gi |

### Auto-scaling

Horizontal Pod Autoscaler (HPA) is configured for:
- Orchestration Engine: 3-20 replicas (CPU: 70%, Memory: 80%)
- Intelligence Engine: 2-8 replicas (CPU: 70%, Memory: 80%)

## Monitoring and Observability

### Access URLs

After deployment, access the system via:

- **API**: `https://api.agent-steering.scrollintel.com`
- **Monitoring**: `https://monitoring.agent-steering.scrollintel.com`
- **Internal Admin**: `https://internal.agent-steering.scrollintel.com`

### Key Metrics

Monitor these critical metrics:

- `agent_registry_active_agents_total`: Number of active agents
- `task_executions_total`: Task execution rate and status
- `http_request_duration_seconds`: API response times
- `business_cost_savings_total`: Business impact metrics
- `decision_accuracy_percentage`: AI decision accuracy

### Alerts

Critical alerts configured:
- Service downtime (>1 minute)
- High error rates (>5%)
- Resource exhaustion (CPU >80%, Memory >85%)
- Agent registration failures
- Database connectivity issues

## Security

### Network Policies

Network policies restrict traffic to:
- Allow ingress only on required ports
- Restrict inter-pod communication to necessary services
- Block external access to internal services

### RBAC

Role-based access control provides:
- Service accounts with minimal required permissions
- Cluster roles for monitoring and management
- Namespace-scoped roles for application operations

### Secrets Management

All sensitive data is stored in Kubernetes secrets:
- Database credentials
- API keys
- Encryption keys
- TLS certificates

## Troubleshooting

### Common Issues

1. **Pods stuck in Pending state**
   ```bash
   kubectl describe pod <pod-name> -n agent-steering-system
   # Check for resource constraints or node selector issues
   ```

2. **Services not accessible**
   ```bash
   kubectl get endpoints -n agent-steering-system
   # Verify service endpoints are populated
   ```

3. **Database connection failures**
   ```bash
   kubectl logs deployment/orchestration-engine -n agent-steering-system
   # Check database connectivity and credentials
   ```

### Debugging Commands

```bash
# Check overall system status
kubectl get all -n agent-steering-system

# View recent events
kubectl get events -n agent-steering-system --sort-by='.lastTimestamp'

# Check pod logs
kubectl logs -f deployment/orchestration-engine -n agent-steering-system

# Execute into a pod for debugging
kubectl exec -it deployment/orchestration-engine -n agent-steering-system -- /bin/bash

# Port forward for local access
kubectl port-forward service/orchestration-service 8080:8080 -n agent-steering-system
```

### Performance Tuning

1. **Increase resource limits** if pods are being OOMKilled
2. **Adjust HPA settings** for better scaling behavior
3. **Tune database parameters** in PostgreSQL configuration
4. **Optimize cache settings** in Redis configuration

## Backup and Recovery

### Database Backups

PostgreSQL is configured with:
- Daily automated backups
- Point-in-time recovery capability
- 7-day retention policy

### Configuration Backups

```bash
# Backup all configurations
kubectl get all,configmaps,secrets -n agent-steering-system -o yaml > backup.yaml
```

### Disaster Recovery

1. **Database Recovery**: Restore from automated backups
2. **Configuration Recovery**: Apply backed-up YAML files
3. **Persistent Volume Recovery**: Restore from volume snapshots

## Scaling

### Horizontal Scaling

```bash
# Scale orchestration engine
kubectl scale deployment orchestration-engine --replicas=10 -n agent-steering-system

# Scale intelligence engine
kubectl scale deployment intelligence-engine --replicas=6 -n agent-steering-system
```

### Vertical Scaling

Update resource limits in deployment files and apply:

```bash
kubectl apply -f k8s/orchestration-deployment.yaml
kubectl rollout restart deployment/orchestration-engine -n agent-steering-system
```

## Updates and Maintenance

### Rolling Updates

```bash
# Update image version
kubectl set image deployment/orchestration-engine orchestration-engine=scrollintel/orchestration-engine:v2.0.0 -n agent-steering-system

# Monitor rollout
kubectl rollout status deployment/orchestration-engine -n agent-steering-system
```

### Rollback

```bash
# Rollback to previous version
kubectl rollout undo deployment/orchestration-engine -n agent-steering-system

# Rollback to specific revision
kubectl rollout undo deployment/orchestration-engine --to-revision=2 -n agent-steering-system
```

## Support

For issues and support:

1. Check the troubleshooting section above
2. Review logs and events for error messages
3. Run validation and health check scripts
4. Consult the monitoring dashboards for system health

## Contributing

When modifying the deployment configuration:

1. Test changes in a development environment first
2. Update documentation for any configuration changes
3. Validate with the deployment validation script
4. Follow the CI/CD pipeline for production deployments