# ScrollIntel Incident Response Runbooks

## Overview

This document provides step-by-step procedures for responding to various types of incidents in the ScrollIntel platform. These runbooks are designed to help the operations team quickly diagnose and resolve issues to minimize downtime and impact on users.

## General Incident Response Process

### 1. Incident Detection
- **Automated Alerts**: Prometheus/Alertmanager notifications
- **User Reports**: Support tickets or direct reports
- **Monitoring Dashboards**: Grafana dashboard anomalies
- **Health Check Failures**: Uptime monitor alerts

### 2. Initial Response (Within 5 minutes)
1. **Acknowledge the alert** in your monitoring system
2. **Assess severity** using the severity matrix below
3. **Create incident ticket** in your ticketing system
4. **Notify stakeholders** based on severity level
5. **Begin investigation** using appropriate runbook

### 3. Investigation and Resolution
1. **Follow specific runbook** for the incident type
2. **Document all actions** taken during investigation
3. **Implement fix** or workaround
4. **Verify resolution** through monitoring and testing
5. **Update stakeholders** on progress and resolution

### 4. Post-Incident
1. **Conduct post-mortem** for major incidents
2. **Update runbooks** based on lessons learned
3. **Implement preventive measures** to avoid recurrence

## Severity Levels

| Severity | Description | Response Time | Notification |
|----------|-------------|---------------|--------------|
| **Critical** | Complete service outage, data loss, security breach | 5 minutes | All stakeholders, executives |
| **Major** | Significant feature unavailable, performance degraded | 15 minutes | Operations team, product team |
| **Minor** | Limited impact, workaround available | 1 hour | Operations team |
| **Info** | No user impact, informational only | 4 hours | Operations team |

---

## Runbook 1: High CPU Usage

### Symptoms
- CPU usage > 80% for 5+ minutes
- Application response times increasing
- Users reporting slow performance

### Investigation Steps
1. **Check current CPU usage**:
   ```bash
   top -p $(pgrep -f scrollintel)
   htop
   ```

2. **Identify CPU-intensive processes**:
   ```bash
   ps aux --sort=-%cpu | head -20
   ```

3. **Check application logs**:
   ```bash
   tail -f logs/scrollintel.json | grep -i error
   ```

4. **Review Grafana CPU dashboard**:
   - Navigate to System Resources panel
   - Check CPU usage trends over last hour

### Resolution Steps
1. **Immediate relief** (if CPU > 95%):
   ```bash
   # Restart application services
   docker-compose restart backend
   ```

2. **Scale horizontally** (if using container orchestration):
   ```bash
   # Increase replica count
   kubectl scale deployment scrollintel-backend --replicas=3
   ```

3. **Optimize queries** (if database-related):
   - Check slow query logs
   - Add missing indexes
   - Optimize expensive operations

4. **Monitor recovery**:
   - Watch CPU metrics return to normal
   - Verify application response times
   - Check error rates

### Prevention
- Set up auto-scaling based on CPU thresholds
- Regular performance testing and optimization
- Code reviews focusing on performance

---

## Runbook 2: Database Connection Issues

### Symptoms
- Database connection errors in logs
- "Connection pool exhausted" messages
- Application timeouts and 500 errors

### Investigation Steps
1. **Check database connectivity**:
   ```bash
   # Test connection
   psql -h $DB_HOST -U $DB_USER -d $DB_NAME -c "SELECT 1;"
   ```

2. **Check connection pool status**:
   ```bash
   # View active connections
   psql -c "SELECT count(*) FROM pg_stat_activity;"
   ```

3. **Review database logs**:
   ```bash
   tail -f /var/log/postgresql/postgresql.log
   ```

4. **Check application connection pool**:
   - Review connection pool configuration
   - Check for connection leaks in application logs

### Resolution Steps
1. **Immediate relief**:
   ```bash
   # Kill idle connections
   psql -c "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE state = 'idle' AND query_start < now() - interval '1 hour';"
   ```

2. **Restart database connection pool**:
   ```bash
   # Restart application to reset connection pool
   docker-compose restart backend
   ```

3. **Scale database connections**:
   - Increase `max_connections` in PostgreSQL config
   - Adjust application connection pool size
   - Consider connection pooling with PgBouncer

4. **Fix connection leaks**:
   - Review code for unclosed connections
   - Implement proper connection management
   - Add connection monitoring

### Prevention
- Implement connection pool monitoring
- Regular connection leak testing
- Proper database connection management in code

---

## Runbook 3: High Error Rate

### Symptoms
- Error rate > 5% for 3+ minutes
- Increased 500 status codes
- User reports of application failures

### Investigation Steps
1. **Check error logs**:
   ```bash
   tail -f logs/errors.json | jq '.'
   ```

2. **Analyze error patterns**:
   ```bash
   # Count errors by type
   grep -o '"error_type":"[^"]*"' logs/errors.json | sort | uniq -c | sort -nr
   ```

3. **Check specific endpoints**:
   ```bash
   # Find failing endpoints
   grep "status_code\":5" logs/scrollintel.json | jq '.endpoint' | sort | uniq -c
   ```

4. **Review recent deployments**:
   - Check if errors started after recent deployment
   - Review recent code changes

### Resolution Steps
1. **Identify root cause**:
   - Database connectivity issues
   - External API failures
   - Code bugs or regressions
   - Resource exhaustion

2. **Immediate mitigation**:
   ```bash
   # Rollback recent deployment if needed
   git checkout previous-stable-version
   docker-compose up -d --build
   ```

3. **Fix underlying issue**:
   - Apply hotfix for code bugs
   - Restart failed external services
   - Scale resources if needed

4. **Verify resolution**:
   - Monitor error rate return to normal
   - Test affected functionality
   - Check user reports

### Prevention
- Comprehensive testing before deployment
- Gradual rollout strategies
- Better error handling and retry logic

---

## Runbook 4: AI Agent Failures

### Symptoms
- Agent failure rate > 10%
- AI service timeout errors
- Users unable to get AI responses

### Investigation Steps
1. **Check agent status**:
   ```bash
   curl http://localhost:8000/health/agents
   ```

2. **Review agent logs**:
   ```bash
   grep "agent_type" logs/scrollintel.json | tail -50
   ```

3. **Check external AI service status**:
   - OpenAI API status page
   - Network connectivity to AI services
   - API key validity and quotas

4. **Monitor agent processing times**:
   - Check Grafana agent performance dashboard
   - Look for timeout patterns

### Resolution Steps
1. **Restart agent services**:
   ```bash
   # Restart specific agent
   docker-compose restart scrollintel-agents
   ```

2. **Check API quotas and limits**:
   - Verify API key limits
   - Check rate limiting status
   - Switch to backup API keys if needed

3. **Scale agent capacity**:
   ```bash
   # Increase agent worker count
   export AGENT_WORKERS=10
   docker-compose up -d
   ```

4. **Implement fallback strategies**:
   - Use cached responses for common queries
   - Provide degraded service with simpler models
   - Queue requests for later processing

### Prevention
- Monitor API usage and quotas
- Implement circuit breakers for external services
- Multiple API key rotation strategy

---

## Runbook 5: Memory Issues

### Symptoms
- Memory usage > 85%
- Out of memory errors
- Application crashes or restarts

### Investigation Steps
1. **Check memory usage**:
   ```bash
   free -h
   ps aux --sort=-%mem | head -20
   ```

2. **Analyze memory leaks**:
   ```bash
   # Monitor memory growth over time
   while true; do ps -o pid,vsz,rss,comm -p $(pgrep scrollintel); sleep 60; done
   ```

3. **Check application memory usage**:
   ```bash
   # Docker container memory stats
   docker stats scrollintel-backend
   ```

4. **Review memory-intensive operations**:
   - Large file processing
   - Data analysis operations
   - Caching strategies

### Resolution Steps
1. **Immediate relief**:
   ```bash
   # Clear system caches
   echo 3 > /proc/sys/vm/drop_caches
   
   # Restart application
   docker-compose restart backend
   ```

2. **Optimize memory usage**:
   - Implement streaming for large data processing
   - Optimize caching strategies
   - Fix memory leaks in code

3. **Scale memory resources**:
   ```bash
   # Increase container memory limits
   docker-compose up -d --scale backend=2
   ```

4. **Monitor recovery**:
   - Watch memory usage stabilize
   - Verify application functionality
   - Check for recurring issues

### Prevention
- Regular memory profiling
- Implement memory usage monitoring
- Optimize data processing algorithms

---

## Runbook 6: Disk Space Issues

### Symptoms
- Disk usage > 90%
- Application unable to write files
- Database write failures

### Investigation Steps
1. **Check disk usage**:
   ```bash
   df -h
   du -sh /* | sort -hr
   ```

2. **Find large files**:
   ```bash
   find / -type f -size +100M -exec ls -lh {} \; 2>/dev/null
   ```

3. **Check log file sizes**:
   ```bash
   du -sh logs/*
   ls -lah logs/
   ```

4. **Identify growing directories**:
   ```bash
   du -sh /var/lib/docker/
   du -sh data/
   ```

### Resolution Steps
1. **Immediate cleanup**:
   ```bash
   # Clean old log files
   find logs/ -name "*.log.*" -mtime +7 -delete
   
   # Clean Docker images
   docker system prune -f
   ```

2. **Archive old data**:
   ```bash
   # Compress old logs
   gzip logs/*.log.old
   
   # Move to archive storage
   mv old-data/ /archive/
   ```

3. **Increase disk space**:
   - Add additional storage volumes
   - Resize existing volumes
   - Move data to larger partition

4. **Implement log rotation**:
   ```bash
   # Configure logrotate
   sudo logrotate -f /etc/logrotate.conf
   ```

### Prevention
- Implement automated log rotation
- Set up disk usage monitoring and alerts
- Regular cleanup of temporary files

---

## Runbook 7: Network Connectivity Issues

### Symptoms
- External API timeouts
- Database connection failures
- Inter-service communication errors

### Investigation Steps
1. **Test network connectivity**:
   ```bash
   # Test external services
   curl -I https://api.openai.com/v1/models
   
   # Test database connection
   telnet $DB_HOST $DB_PORT
   ```

2. **Check DNS resolution**:
   ```bash
   nslookup api.openai.com
   dig api.openai.com
   ```

3. **Review network logs**:
   ```bash
   # Check system network logs
   journalctl -u networking
   
   # Check firewall logs
   sudo iptables -L -n -v
   ```

4. **Test internal services**:
   ```bash
   # Test service-to-service communication
   curl http://backend:8000/health
   ```

### Resolution Steps
1. **Restart network services**:
   ```bash
   # Restart networking
   sudo systemctl restart networking
   
   # Restart Docker networking
   docker network prune -f
   ```

2. **Check firewall rules**:
   ```bash
   # Verify firewall configuration
   sudo ufw status
   
   # Allow necessary ports
   sudo ufw allow 8000/tcp
   ```

3. **Update DNS configuration**:
   ```bash
   # Update DNS servers
   echo "nameserver 8.8.8.8" >> /etc/resolv.conf
   ```

4. **Verify resolution**:
   - Test all external connections
   - Verify internal service communication
   - Monitor network metrics

### Prevention
- Implement network monitoring
- Regular network configuration backups
- Redundant network paths where possible

---

## Emergency Contacts

### Primary On-Call
- **Operations Team**: ops-team@scrollintel.com
- **Phone**: +1-XXX-XXX-XXXX
- **Slack**: #ops-alerts

### Secondary Contacts
- **Development Team**: dev-team@scrollintel.com
- **Product Team**: product-team@scrollintel.com
- **Executive Team**: executives@scrollintel.com

### External Vendors
- **Cloud Provider**: AWS Support
- **Database Support**: PostgreSQL Support
- **Monitoring**: Grafana Support

---

## Tools and Resources

### Monitoring Dashboards
- **Grafana**: http://grafana.scrollintel.com
- **Prometheus**: http://prometheus.scrollintel.com
- **Status Page**: http://status.scrollintel.com

### Log Analysis
- **Elasticsearch**: http://elasticsearch.scrollintel.com
- **Kibana**: http://kibana.scrollintel.com
- **Log Files**: `/var/log/scrollintel/`

### Documentation
- **API Documentation**: http://docs.scrollintel.com
- **Architecture Diagrams**: `/docs/architecture/`
- **Deployment Guide**: `/docs/deployment.md`

---

## Runbook Maintenance

### Regular Updates
- Review and update runbooks monthly
- Test procedures during maintenance windows
- Incorporate lessons learned from incidents

### Version Control
- All runbooks stored in Git repository
- Changes require peer review
- Version history maintained

### Training
- New team members must complete runbook training
- Regular drills and simulations
- Knowledge sharing sessions

---

*Last Updated: [Current Date]*
*Version: 1.0*
*Next Review: [Date + 1 month]*