# ScrollIntel™ Troubleshooting Guide

## Overview

This guide provides solutions to common issues encountered during ScrollIntel deployment and operation.

## Table of Contents

1. [General Troubleshooting](#general-troubleshooting)
2. [Database Issues](#database-issues)
3. [Redis Issues](#redis-issues)
4. [Docker Issues](#docker-issues)
5. [API Issues](#api-issues)
6. [Frontend Issues](#frontend-issues)
7. [AI Service Issues](#ai-service-issues)
8. [Performance Issues](#performance-issues)
9. [Security Issues](#security-issues)
10. [Monitoring Issues](#monitoring-issues)

## General Troubleshooting

### Check System Health

First, always check the overall system health:

```bash
# Basic health check
curl http://localhost:8000/health

# Detailed health check
curl http://localhost:8000/health/detailed

# Check dependencies
curl http://localhost:8000/health/dependencies

# System metrics
curl http://localhost:8000/health/metrics
```

### Common Commands

```bash
# Check running containers
docker ps

# View container logs
docker logs <container_name>

# Check resource usage
docker stats

# Restart services
docker-compose restart

# Rebuild and restart
docker-compose up -d --build
```

## Database Issues

### Issue: Database Connection Failed

**Symptoms:**
- Health check shows database as unhealthy
- Application fails to start
- Error: "could not connect to server"

**Solutions:**

1. **Check database container status:**
   ```bash
   docker ps | grep postgres
   docker logs scrollintel-postgres
   ```

2. **Verify database credentials:**
   ```bash
   # Test connection manually
   psql postgresql://user:password@localhost:5432/scrollintel
   ```

3. **Check environment variables:**
   ```bash
   echo $POSTGRES_HOST
   echo $POSTGRES_PASSWORD
   ```

4. **Restart database container:**
   ```bash
   docker-compose restart postgres
   ```

### Issue: Migration Failures

**Symptoms:**
- Migration script fails
- Database schema out of sync
- Alembic errors

**Solutions:**

1. **Check migration status:**
   ```bash
   python scripts/migrate-database.py current
   python scripts/migrate-database.py history
   ```

2. **Reset migrations (DANGER - loses data):**
   ```bash
   python scripts/migrate-database.py reset --confirm
   ```

3. **Manual migration fix:**
   ```bash
   # Connect to database
   psql $DATABASE_URL
   
   # Check alembic version table
   SELECT * FROM alembic_version;
   
   # Manually set version if needed
   UPDATE alembic_version SET version_num = 'target_revision';
   ```

### Issue: Database Performance

**Symptoms:**
- Slow query responses
- High CPU usage on database
- Connection timeouts

**Solutions:**

1. **Check database stats:**
   ```sql
   -- Active connections
   SELECT count(*) FROM pg_stat_activity;
   
   -- Slow queries
   SELECT query, mean_exec_time, calls 
   FROM pg_stat_statements 
   ORDER BY mean_exec_time DESC 
   LIMIT 10;
   ```

2. **Optimize database:**
   ```bash
   # Analyze tables
   psql $DATABASE_URL -c "ANALYZE;"
   
   # Vacuum database
   psql $DATABASE_URL -c "VACUUM ANALYZE;"
   ```

3. **Increase connection pool:**
   ```python
   # In database.py
   engine = create_engine(
       DATABASE_URL,
       pool_size=20,
       max_overflow=30
   )
   ```

## Redis Issues

### Issue: Redis Connection Failed

**Symptoms:**
- Cache operations fail
- Session management broken
- Redis health check fails

**Solutions:**

1. **Check Redis container:**
   ```bash
   docker logs scrollintel-redis
   redis-cli ping
   ```

2. **Test Redis connection:**
   ```bash
   # Using redis-cli
   redis-cli -h localhost -p 6379 ping
   
   # Using Python
   python -c "import redis; r=redis.Redis(); print(r.ping())"
   ```

3. **Clear Redis cache:**
   ```bash
   redis-cli FLUSHALL
   ```

### Issue: Redis Memory Issues

**Symptoms:**
- Out of memory errors
- Redis performance degradation
- Cache misses increasing

**Solutions:**

1. **Check Redis memory usage:**
   ```bash
   redis-cli INFO memory
   ```

2. **Configure memory policy:**
   ```bash
   redis-cli CONFIG SET maxmemory-policy allkeys-lru
   ```

3. **Increase Redis memory:**
   ```yaml
   # In docker-compose.yml
   redis:
     command: redis-server --maxmemory 512mb
   ```

## Docker Issues

### Issue: Container Won't Start

**Symptoms:**
- Container exits immediately
- Build failures
- Port conflicts

**Solutions:**

1. **Check container logs:**
   ```bash
   docker logs <container_name>
   docker-compose logs <service_name>
   ```

2. **Check port conflicts:**
   ```bash
   # Check what's using port 8000
   lsof -i :8000
   netstat -tulpn | grep 8000
   ```

3. **Rebuild containers:**
   ```bash
   docker-compose down
   docker-compose build --no-cache
   docker-compose up -d
   ```

### Issue: Docker Build Failures

**Symptoms:**
- Build process fails
- Dependency installation errors
- Out of space errors

**Solutions:**

1. **Clean Docker system:**
   ```bash
   docker system prune -a
   docker volume prune
   ```

2. **Check disk space:**
   ```bash
   df -h
   docker system df
   ```

3. **Build with verbose output:**
   ```bash
   docker build --no-cache --progress=plain .
   ```

### Issue: Container Health Check Failures

**Symptoms:**
- Container marked as unhealthy
- Load balancer removes container
- Service unavailable

**Solutions:**

1. **Check health check endpoint:**
   ```bash
   # Test health check manually
   docker exec <container> curl -f http://localhost:8000/health
   ```

2. **Adjust health check timing:**
   ```dockerfile
   HEALTHCHECK --interval=60s --timeout=30s --start-period=60s --retries=3 \
       CMD curl -f http://localhost:8000/health || exit 1
   ```

## API Issues

### Issue: API Not Responding

**Symptoms:**
- 502 Bad Gateway errors
- Connection timeouts
- No response from API

**Solutions:**

1. **Check API server status:**
   ```bash
   curl -v http://localhost:8000/health
   docker logs scrollintel-backend
   ```

2. **Check process status:**
   ```bash
   # Inside container
   docker exec scrollintel-backend ps aux | grep uvicorn
   ```

3. **Restart API server:**
   ```bash
   docker-compose restart backend
   ```

### Issue: High API Latency

**Symptoms:**
- Slow response times
- Timeouts
- Poor user experience

**Solutions:**

1. **Check API metrics:**
   ```bash
   curl http://localhost:8000/health/metrics
   ```

2. **Enable API caching:**
   ```python
   # Add caching to frequently accessed endpoints
   from fastapi_cache import FastAPICache
   from fastapi_cache.backends.redis import RedisBackend
   ```

3. **Optimize database queries:**
   ```python
   # Use async queries and connection pooling
   # Add database indexes
   # Implement query optimization
   ```

### Issue: Authentication Failures

**Symptoms:**
- Login failures
- JWT token errors
- Permission denied errors

**Solutions:**

1. **Check JWT configuration:**
   ```bash
   echo $JWT_SECRET_KEY
   python -c "from scrollintel.core.config import get_settings; print(get_settings().jwt_secret_key)"
   ```

2. **Test token generation:**
   ```python
   from scrollintel.security.auth import create_access_token
   token = create_access_token({"sub": "test@example.com"})
   print(token)
   ```

3. **Check user permissions:**
   ```sql
   SELECT * FROM users WHERE email = 'user@example.com';
   SELECT * FROM audit_logs WHERE user_id = 'user_id' ORDER BY timestamp DESC LIMIT 10;
   ```

## Frontend Issues

### Issue: Frontend Build Failures

**Symptoms:**
- npm build fails
- TypeScript errors
- Missing dependencies

**Solutions:**

1. **Clear npm cache:**
   ```bash
   cd frontend
   npm cache clean --force
   rm -rf node_modules package-lock.json
   npm install
   ```

2. **Check Node.js version:**
   ```bash
   node --version  # Should be 18+
   npm --version
   ```

3. **Fix TypeScript errors:**
   ```bash
   cd frontend
   npm run lint
   npx tsc --noEmit
   ```

### Issue: Frontend Not Loading

**Symptoms:**
- Blank page
- JavaScript errors
- API connection failures

**Solutions:**

1. **Check browser console:**
   - Open browser developer tools
   - Look for JavaScript errors
   - Check network tab for failed requests

2. **Verify API connection:**
   ```bash
   # Check if backend is accessible from frontend
   curl http://backend:8000/health
   ```

3. **Check environment variables:**
   ```bash
   echo $NEXT_PUBLIC_API_URL
   ```

## AI Service Issues

### Issue: OpenAI API Failures

**Symptoms:**
- AI agent responses fail
- API key errors
- Rate limit exceeded

**Solutions:**

1. **Check API key:**
   ```bash
   curl -H "Authorization: Bearer $OPENAI_API_KEY" \
        https://api.openai.com/v1/models
   ```

2. **Check rate limits:**
   ```python
   # Implement exponential backoff
   import time
   import random
   
   def retry_with_backoff(func, max_retries=3):
       for i in range(max_retries):
           try:
               return func()
           except Exception as e:
               if i == max_retries - 1:
                   raise
               time.sleep(2 ** i + random.uniform(0, 1))
   ```

3. **Monitor usage:**
   ```bash
   # Check OpenAI usage dashboard
   # Implement usage tracking in application
   ```

### Issue: Vector Database Issues

**Symptoms:**
- Embedding failures
- Search not working
- Pinecone connection errors

**Solutions:**

1. **Test Pinecone connection:**
   ```python
   import pinecone
   pinecone.init(api_key="your-key", environment="us-east-1")
   print(pinecone.list_indexes())
   ```

2. **Check index status:**
   ```python
   index = pinecone.Index("your-index")
   print(index.describe_index_stats())
   ```

3. **Recreate index if needed:**
   ```python
   # Delete and recreate index
   pinecone.delete_index("your-index")
   pinecone.create_index("your-index", dimension=1536)
   ```

## Performance Issues

### Issue: High Memory Usage

**Symptoms:**
- Out of memory errors
- Container restarts
- Slow performance

**Solutions:**

1. **Monitor memory usage:**
   ```bash
   docker stats
   free -h
   ```

2. **Optimize Python memory:**
   ```python
   # Use generators instead of lists
   # Implement proper cleanup
   # Use memory profiling tools
   ```

3. **Increase container memory:**
   ```yaml
   # In docker-compose.yml
   backend:
     deploy:
       resources:
         limits:
           memory: 2G
   ```

### Issue: High CPU Usage

**Symptoms:**
- Slow response times
- High load average
- CPU throttling

**Solutions:**

1. **Profile CPU usage:**
   ```bash
   top
   htop
   docker stats
   ```

2. **Optimize code:**
   ```python
   # Use async/await properly
   # Implement caching
   # Optimize algorithms
   ```

3. **Scale horizontally:**
   ```bash
   # Add more backend instances
   docker-compose up -d --scale backend=3
   ```

## Security Issues

### Issue: Security Vulnerabilities

**Symptoms:**
- Security scanner alerts
- Suspicious activity
- Unauthorized access

**Solutions:**

1. **Update dependencies:**
   ```bash
   # Python
   pip-audit
   safety check
   
   # Node.js
   npm audit
   npm audit fix
   ```

2. **Check for secrets in logs:**
   ```bash
   # Scan logs for potential secrets
   grep -r "password\|key\|secret" logs/
   ```

3. **Review access logs:**
   ```bash
   # Check for suspicious patterns
   tail -f /var/log/nginx/access.log | grep -E "(404|500|401)"
   ```

## Monitoring Issues

### Issue: Metrics Not Collecting

**Symptoms:**
- Empty Grafana dashboards
- Prometheus targets down
- No alerts firing

**Solutions:**

1. **Check Prometheus targets:**
   ```bash
   curl http://localhost:9090/api/v1/targets
   ```

2. **Verify metrics endpoints:**
   ```bash
   curl http://localhost:8000/metrics
   curl http://localhost:8000/health/metrics
   ```

3. **Check Prometheus configuration:**
   ```bash
   # Validate prometheus.yml
   promtool check config monitoring/prometheus.yml
   ```

## Getting Help

### Log Collection

When reporting issues, collect these logs:

```bash
# Application logs
docker-compose logs > scrollintel-logs.txt

# System logs
journalctl -u docker > system-logs.txt

# Health check output
curl http://localhost:8000/health/detailed > health-check.json
```

### Debug Mode

Enable debug mode for more detailed logging:

```bash
export DEBUG=true
export LOG_LEVEL=DEBUG
docker-compose restart
```

### Support Channels

1. Check the documentation
2. Search existing issues
3. Create detailed bug report
4. Contact support team

---

**Remember**: Always backup your data before attempting major fixes!

**ScrollIntel™ v4.0+ - The world's most advanced AI-CTO platform**