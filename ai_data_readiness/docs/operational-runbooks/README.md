# Operational Runbooks

This document provides step-by-step procedures for common operational tasks and incident response.

## Table of Contents

1. [System Health Monitoring](#system-health-monitoring)
2. [Incident Response Procedures](#incident-response-procedures)
3. [Backup and Recovery](#backup-and-recovery)
4. [Performance Troubleshooting](#performance-troubleshooting)
5. [Security Incident Response](#security-incident-response)
6. [Maintenance Procedures](#maintenance-procedures)
7. [Scaling Operations](#scaling-operations)
8. [Data Pipeline Management](#data-pipeline-management)

## System Health Monitoring

### Daily Health Checks

**Procedure:**
1. Check system dashboard for overall health status
2. Review error rates and response times
3. Verify database connectivity and performance
4. Check storage usage and capacity
5. Review recent alerts and notifications

**Commands:**
```bash
# Check application health
curl -f http://localhost:8000/health

# Check database connections
psql $DATABASE_URL -c "SELECT count(*) FROM pg_stat_activity;"

# Check disk usage
df -h

# Check memory usage
free -h

# Check recent errors
tail -n 100 /var/log/ai-data-readiness/error.log
```

### Weekly System Review

**Procedure:**
1. Analyze performance trends
2. Review capacity planning metrics
3. Check backup integrity
4. Update security patches
5. Review and update documentation

## Incident Response Procedures

### High Severity Incident (P1)

**Definition:** Complete system outage or data loss

**Response Steps:**
1. **Immediate Response (0-15 minutes)**
   - Acknowledge incident in monitoring system
   - Notify on-call team and stakeholders
   - Begin initial assessment
   - Implement emergency rollback if applicable

2. **Investigation (15-60 minutes)**
   - Gather system logs and metrics
   - Identify root cause
   - Assess impact scope
   - Communicate status updates

3. **Resolution (1-4 hours)**
   - Implement fix or workaround
   - Verify system recovery
   - Monitor for stability
   - Update stakeholders

4. **Post-Incident (24-48 hours)**
   - Conduct post-mortem review
   - Document lessons learned
   - Implement preventive measures
   - Update runbooks

### Medium Severity Incident (P2)

**Definition:** Degraded performance or partial functionality loss

**Response Steps:**
1. Assess impact and affected users
2. Investigate root cause
3. Implement fix during business hours
4. Monitor system stability
5. Document incident and resolution

### Low Severity Incident (P3)

**Definition:** Minor issues with workarounds available

**Response Steps:**
1. Log incident for tracking
2. Schedule fix during maintenance window
3. Communicate workaround to users
4. Implement permanent fix
5. Verify resolution

## Backup and Recovery

### Database Backup Verification

**Daily Procedure:**
```bash
#!/bin/bash
# verify-backup.sh

BACKUP_FILE="/backups/latest.sql.gz"
TEST_DB="ai_data_readiness_test"

# Check if backup file exists and is recent
if [[ -f "$BACKUP_FILE" && $(find "$BACKUP_FILE" -mtime -1) ]]; then
    echo "Backup file found and is recent"
else
    echo "ERROR: Backup file missing or outdated"
    exit 1
fi

# Test restore to temporary database
createdb $TEST_DB
gunzip -c $BACKUP_FILE | psql $TEST_DB

# Verify data integrity
RECORD_COUNT=$(psql $TEST_DB -t -c "SELECT count(*) FROM datasets;")
if [[ $RECORD_COUNT -gt 0 ]]; then
    echo "Backup verification successful: $RECORD_COUNT records"
else
    echo "ERROR: Backup verification failed"
    exit 1
fi

# Cleanup
dropdb $TEST_DB
```

### Disaster Recovery Procedure

**Complete System Recovery:**
1. **Assessment Phase**
   - Determine extent of data loss
   - Identify last known good backup
   - Assess infrastructure damage
   - Estimate recovery time

2. **Infrastructure Recovery**
   ```bash
   # Provision new infrastructure
   terraform apply -var="environment=disaster-recovery"
   
   # Deploy application
   kubectl apply -f k8s/disaster-recovery/
   ```

3. **Data Recovery**
   ```bash
   # Restore database
   aws s3 cp s3://backup-bucket/database/latest.sql.gz ./
   gunzip latest.sql.gz
   psql $DATABASE_URL < latest.sql
   
   # Restore file storage
   aws s3 sync s3://backup-bucket/files/ /app/uploads/
   ```

4. **Verification**
   - Test critical functionality
   - Verify data integrity
   - Check system performance
   - Validate user access

## Performance Troubleshooting

### High Response Times

**Investigation Steps:**
1. Check system resources (CPU, memory, disk I/O)
2. Analyze database performance
3. Review application logs for errors
4. Check network connectivity
5. Examine cache hit rates

**Commands:**
```bash
# Check system resources
htop
iotop
free -h

# Database performance
psql $DATABASE_URL -c "
SELECT query, mean_time, calls 
FROM pg_stat_statements 
ORDER BY mean_time DESC 
LIMIT 10;"

# Application metrics
curl http://localhost:9090/metrics | grep response_time

# Cache statistics
redis-cli info stats
```

### High Memory Usage

**Investigation Steps:**
1. Identify memory-consuming processes
2. Check for memory leaks
3. Analyze garbage collection patterns
4. Review data processing jobs
5. Check cache usage

**Mitigation Actions:**
```bash
# Restart high-memory processes
systemctl restart ai-data-readiness-worker

# Clear caches if safe
redis-cli FLUSHDB

# Adjust memory limits
kubectl patch deployment ai-data-readiness-app -p '{"spec":{"template":{"spec":{"containers":[{"name":"app","resources":{"limits":{"memory":"8Gi"}}}]}}}}'
```

### Database Performance Issues

**Investigation Steps:**
1. Check slow query log
2. Analyze query execution plans
3. Review index usage
4. Check connection pool status
5. Monitor lock contention

**Optimization Actions:**
```sql
-- Identify slow queries
SELECT query, mean_time, calls, total_time
FROM pg_stat_statements
WHERE mean_time > 1000
ORDER BY mean_time DESC;

-- Check missing indexes
SELECT schemaname, tablename, attname, n_distinct, correlation
FROM pg_stats
WHERE schemaname = 'public'
AND n_distinct > 100
AND correlation < 0.1;

-- Analyze table statistics
ANALYZE;

-- Vacuum tables
VACUUM ANALYZE;
```

## Security Incident Response

### Suspected Data Breach

**Immediate Actions (0-1 hour):**
1. Isolate affected systems
2. Preserve evidence
3. Notify security team
4. Begin forensic analysis
5. Assess data exposure

**Investigation Phase (1-24 hours):**
1. Determine attack vector
2. Assess scope of compromise
3. Identify affected data
4. Document timeline
5. Coordinate with legal team

**Containment and Recovery (1-7 days):**
1. Patch vulnerabilities
2. Reset compromised credentials
3. Implement additional security controls
4. Monitor for continued threats
5. Restore services securely

### Unauthorized Access Attempt

**Response Steps:**
1. Block suspicious IP addresses
2. Review access logs
3. Check for privilege escalation
4. Verify user account integrity
5. Update security policies

**Commands:**
```bash
# Block IP address
iptables -A INPUT -s SUSPICIOUS_IP -j DROP

# Review access logs
grep "SUSPICIOUS_IP" /var/log/nginx/access.log
grep "failed login" /var/log/auth.log

# Check user sessions
who
last -n 20

# Review database access
psql $DATABASE_URL -c "SELECT * FROM pg_stat_activity WHERE client_addr = 'SUSPICIOUS_IP';"
```

## Maintenance Procedures

### Planned Maintenance Window

**Pre-Maintenance (1 week before):**
1. Schedule maintenance window
2. Notify users and stakeholders
3. Prepare rollback procedures
4. Test changes in staging
5. Create maintenance checklist

**During Maintenance:**
1. Enable maintenance mode
2. Stop application services
3. Backup current state
4. Apply updates/changes
5. Verify functionality
6. Disable maintenance mode

**Post-Maintenance:**
1. Monitor system stability
2. Verify all services running
3. Check performance metrics
4. Communicate completion
5. Document any issues

### Database Maintenance

**Monthly Procedure:**
```sql
-- Update table statistics
ANALYZE;

-- Vacuum tables
VACUUM ANALYZE;

-- Reindex if needed
REINDEX DATABASE ai_data_readiness;

-- Check for bloat
SELECT schemaname, tablename, 
       pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
FROM pg_tables 
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
```

## Scaling Operations

### Horizontal Scaling

**Scale Up Procedure:**
```bash
# Kubernetes
kubectl scale deployment ai-data-readiness-app --replicas=10

# Docker Compose
docker-compose up -d --scale app=5 --scale worker=3

# Monitor scaling
kubectl get pods -l app=ai-data-readiness-app
```

**Scale Down Procedure:**
```bash
# Gradual scale down
kubectl scale deployment ai-data-readiness-app --replicas=3

# Monitor for stability
kubectl get pods -w
```

### Database Scaling

**Read Replica Setup:**
```bash
# Create read replica
aws rds create-db-instance-read-replica \
  --db-instance-identifier ai-data-readiness-replica \
  --source-db-instance-identifier ai-data-readiness-primary

# Update application configuration
kubectl patch configmap ai-data-readiness-config \
  -p '{"data":{"READ_DATABASE_URL":"postgresql://replica-endpoint"}}'
```

## Data Pipeline Management

### Pipeline Monitoring

**Daily Checks:**
1. Verify pipeline execution status
2. Check data quality metrics
3. Review processing times
4. Monitor error rates
5. Validate output data

**Commands:**
```bash
# Check pipeline status
curl -H "Authorization: Bearer $TOKEN" \
     "http://localhost:8000/api/v1/pipelines/status"

# Review recent jobs
psql $DATABASE_URL -c "
SELECT id, status, created_at, completed_at 
FROM processing_jobs 
WHERE created_at > NOW() - INTERVAL '24 hours'
ORDER BY created_at DESC;"

# Check data quality trends
curl -H "Authorization: Bearer $TOKEN" \
     "http://localhost:8000/api/v1/quality/trends?days=7"
```

### Pipeline Failure Recovery

**Recovery Steps:**
1. Identify failed pipeline stage
2. Check error logs and messages
3. Determine if data corruption occurred
4. Restart from last successful checkpoint
5. Validate recovered data

**Commands:**
```bash
# Restart failed job
curl -X POST -H "Authorization: Bearer $TOKEN" \
     "http://localhost:8000/api/v1/jobs/$JOB_ID/restart"

# Check job progress
curl -H "Authorization: Bearer $TOKEN" \
     "http://localhost:8000/api/v1/jobs/$JOB_ID/status"

# Validate output
curl -H "Authorization: Bearer $TOKEN" \
     "http://localhost:8000/api/v1/datasets/$DATASET_ID/validate"
```

This runbook provides essential procedures for maintaining and operating the AI Data Readiness Platform. Keep it updated as the system evolves and new procedures are developed.