# Troubleshooting Guide

This guide helps you diagnose and resolve common issues with the AI Data Readiness Platform.

## Table of Contents

1. [Common Issues](#common-issues)
2. [Installation Problems](#installation-problems)
3. [Data Upload Issues](#data-upload-issues)
4. [Quality Assessment Problems](#quality-assessment-problems)
5. [Performance Issues](#performance-issues)
6. [API Errors](#api-errors)
7. [Database Issues](#database-issues)
8. [Authentication Problems](#authentication-problems)
9. [Monitoring and Alerts](#monitoring-and-alerts)
10. [Getting Support](#getting-support)

## Common Issues

### Platform Won't Start

**Symptoms:**
- Application fails to start
- Connection errors
- Service unavailable messages

**Possible Causes:**
1. Database connection issues
2. Missing environment variables
3. Port conflicts
4. Insufficient permissions

**Solutions:**

1. **Check Database Connection**
   ```bash
   # Test database connectivity
   python -c "
   import os
   import psycopg2
   try:
       conn = psycopg2.connect(os.getenv('DATABASE_URL'))
       print('Database connection successful')
   except Exception as e:
       print(f'Database connection failed: {e}')
   "
   ```

2. **Verify Environment Variables**
   ```bash
   # Check required environment variables
   echo "DATABASE_URL: $DATABASE_URL"
   echo "REDIS_URL: $REDIS_URL"
   echo "SECRET_KEY: $SECRET_KEY"
   ```

3. **Check Port Availability**
   ```bash
   # Check if port 8000 is available
   netstat -tulpn | grep :8000
   # If occupied, change port in configuration
   ```

4. **Review Logs**
   ```bash
   # Check application logs
   tail -f logs/app.log
   # Check system logs
   journalctl -u ai-data-readiness -f
   ```

### Slow Performance

**Symptoms:**
- Long response times
- Timeouts
- High resource usage

**Diagnostic Steps:**

1. **Check System Resources**
   ```bash
   # Monitor CPU and memory usage
   htop
   # Check disk usage
   df -h
   # Monitor I/O
   iotop
   ```

2. **Database Performance**
   ```sql
   -- Check slow queries
   SELECT query, mean_time, calls 
   FROM pg_stat_statements 
   ORDER BY mean_time DESC 
   LIMIT 10;
   
   -- Check database connections
   SELECT count(*) FROM pg_stat_activity;
   ```

3. **Application Metrics**
   ```bash
   # Check API response times
   curl -w "@curl-format.txt" -o /dev/null -s "http://localhost:8000/health"
   ```

**Solutions:**
- Scale up resources (CPU, memory)
- Optimize database queries
- Implement caching
- Use connection pooling
- Enable compression

## Installation Problems

### Dependency Conflicts

**Error Messages:**
```
ERROR: pip's dependency resolver does not currently consider all the packages that are installed
ERROR: Could not find a version that satisfies the requirement
```

**Solutions:**

1. **Use Virtual Environment**
   ```bash
   python -m venv ai_data_readiness_env
   source ai_data_readiness_env/bin/activate  # Linux/Mac
   # or
   ai_data_readiness_env\Scripts\activate  # Windows
   pip install -r requirements.txt
   ```

2. **Update pip and setuptools**
   ```bash
   pip install --upgrade pip setuptools wheel
   ```

3. **Install Dependencies Separately**
   ```bash
   # Install core dependencies first
   pip install fastapi uvicorn sqlalchemy psycopg2-binary
   # Then install ML dependencies
   pip install pandas numpy scikit-learn
   # Finally install remaining dependencies
   pip install -r requirements.txt
   ```

### Database Setup Issues

**Error Messages:**
```
psycopg2.OperationalError: could not connect to server
sqlalchemy.exc.OperationalError: (psycopg2.OperationalError)
```

**Solutions:**

1. **Install PostgreSQL**
   ```bash
   # Ubuntu/Debian
   sudo apt-get install postgresql postgresql-contrib
   # CentOS/RHEL
   sudo yum install postgresql-server postgresql-contrib
   # macOS
   brew install postgresql
   ```

2. **Create Database and User**
   ```sql
   -- Connect as postgres user
   sudo -u postgres psql
   
   -- Create database
   CREATE DATABASE ai_data_readiness;
   
   -- Create user
   CREATE USER ai_user WITH PASSWORD 'your_password';
   
   -- Grant privileges
   GRANT ALL PRIVILEGES ON DATABASE ai_data_readiness TO ai_user;
   ```

3. **Configure Connection**
   ```bash
   # Update .env file
   DATABASE_URL=postgresql://ai_user:your_password@localhost:5432/ai_data_readiness
   ```

## Data Upload Issues

### File Format Not Supported

**Error Messages:**
```
Unsupported file format: .xls
File format validation failed
```

**Supported Formats:**
- CSV (.csv)
- JSON (.json)
- Parquet (.parquet)
- Avro (.avro)
- Excel (.xlsx) - converted to CSV

**Solutions:**

1. **Convert File Format**
   ```python
   import pandas as pd
   
   # Convert Excel to CSV
   df = pd.read_excel('data.xls')
   df.to_csv('data.csv', index=False)
   
   # Convert to Parquet
   df.to_parquet('data.parquet')
   ```

2. **Check File Extension**
   ```bash
   # Verify file extension matches content
   file your_data.csv
   ```

### Large File Upload Failures

**Error Messages:**
```
413 Request Entity Too Large
Connection timeout
Memory error during processing
```

**Solutions:**

1. **Increase Upload Limits**
   ```python
   # In configuration
   MAX_FILE_SIZE = 2 * 1024 * 1024 * 1024  # 2GB
   REQUEST_TIMEOUT = 300  # 5 minutes
   ```

2. **Split Large Files**
   ```python
   import pandas as pd
   
   # Split large CSV file
   chunk_size = 100000
   for i, chunk in enumerate(pd.read_csv('large_file.csv', chunksize=chunk_size)):
       chunk.to_csv(f'chunk_{i}.csv', index=False)
   ```

3. **Use Streaming Upload**
   ```python
   # Stream large files
   with open('large_file.csv', 'rb') as f:
       response = requests.post(
           'http://localhost:8000/api/v1/datasets/upload',
           files={'file': f},
           stream=True
       )
   ```

### Encoding Issues

**Error Messages:**
```
UnicodeDecodeError: 'utf-8' codec can't decode
Invalid character encoding
```

**Solutions:**

1. **Detect File Encoding**
   ```python
   import chardet
   
   with open('your_file.csv', 'rb') as f:
       result = chardet.detect(f.read())
       print(f"Encoding: {result['encoding']}")
   ```

2. **Convert to UTF-8**
   ```python
   import pandas as pd
   
   # Read with detected encoding and save as UTF-8
   df = pd.read_csv('your_file.csv', encoding='latin1')
   df.to_csv('your_file_utf8.csv', encoding='utf-8', index=False)
   ```

## Quality Assessment Problems

### Assessment Takes Too Long

**Symptoms:**
- Quality assessment jobs stuck in "running" state
- Timeout errors
- No progress updates

**Diagnostic Steps:**

1. **Check Job Status**
   ```bash
   curl -H "Authorization: Bearer $TOKEN" \
        "http://localhost:8000/api/v1/jobs/$JOB_ID"
   ```

2. **Monitor Resource Usage**
   ```bash
   # Check memory usage
   free -h
   # Check CPU usage
   top -p $(pgrep -f "quality_assessment")
   ```

**Solutions:**

1. **Increase Resources**
   ```yaml
   # docker-compose.yml
   services:
     ai-data-readiness:
       deploy:
         resources:
           limits:
             memory: 8G
             cpus: '4'
   ```

2. **Optimize Assessment Parameters**
   ```python
   # Reduce sample size for large datasets
   assessment_config = {
       "sample_size": 10000,
       "enable_advanced_stats": False,
       "parallel_processing": True
   }
   ```

### Incorrect Quality Scores

**Symptoms:**
- Quality scores don't match expectations
- Missing issues not detected
- False positive quality issues

**Solutions:**

1. **Review Data Types**
   ```python
   # Check inferred data types
   import pandas as pd
   df = pd.read_csv('your_data.csv')
   print(df.dtypes)
   print(df.info())
   ```

2. **Configure Quality Rules**
   ```python
   # Custom quality rules
   quality_config = {
       "completeness_threshold": 0.95,
       "accuracy_rules": {
           "email": r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
           "phone": r'^\+?1?-?\.?\s?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}$'
       },
       "range_checks": {
           "age": {"min": 0, "max": 120},
           "salary": {"min": 0, "max": 1000000}
       }
   }
   ```

## Performance Issues

### High Memory Usage

**Symptoms:**
- Out of memory errors
- System becomes unresponsive
- Swap usage increases

**Solutions:**

1. **Optimize Data Processing**
   ```python
   # Process data in chunks
   chunk_size = 10000
   for chunk in pd.read_csv('large_file.csv', chunksize=chunk_size):
       process_chunk(chunk)
   ```

2. **Configure Memory Limits**
   ```python
   # Limit pandas memory usage
   pd.set_option('mode.chained_assignment', None)
   pd.set_option('compute.use_bottleneck', True)
   ```

3. **Use Memory-Efficient Data Types**
   ```python
   # Optimize data types
   df['category'] = df['category'].astype('category')
   df['integer_col'] = pd.to_numeric(df['integer_col'], downcast='integer')
   df['float_col'] = pd.to_numeric(df['float_col'], downcast='float')
   ```

### Slow Database Queries

**Symptoms:**
- Long response times for data retrieval
- Database connection timeouts
- High database CPU usage

**Solutions:**

1. **Add Database Indexes**
   ```sql
   -- Add indexes for frequently queried columns
   CREATE INDEX idx_dataset_id ON quality_reports(dataset_id);
   CREATE INDEX idx_created_at ON datasets(created_at);
   CREATE INDEX idx_user_id ON datasets(user_id);
   ```

2. **Optimize Queries**
   ```sql
   -- Use EXPLAIN to analyze query performance
   EXPLAIN ANALYZE SELECT * FROM datasets WHERE user_id = 123;
   
   -- Add appropriate WHERE clauses and LIMIT
   SELECT id, name, created_at 
   FROM datasets 
   WHERE user_id = 123 
   ORDER BY created_at DESC 
   LIMIT 20;
   ```

3. **Configure Connection Pooling**
   ```python
   # SQLAlchemy connection pool settings
   engine = create_engine(
       DATABASE_URL,
       pool_size=20,
       max_overflow=30,
       pool_pre_ping=True,
       pool_recycle=3600
   )
   ```

## API Errors

### Authentication Failures

**Error Messages:**
```
401 Unauthorized
Invalid token
Token expired
```

**Solutions:**

1. **Check Token Validity**
   ```bash
   # Decode JWT token (without verification)
   python -c "
   import jwt
   token = 'your_jwt_token'
   decoded = jwt.decode(token, options={'verify_signature': False})
   print(decoded)
   "
   ```

2. **Refresh Token**
   ```python
   import requests
   
   response = requests.post(
       'http://localhost:8000/api/v1/auth/refresh',
       json={'refresh_token': 'your_refresh_token'}
   )
   new_token = response.json()['access_token']
   ```

3. **Check Token Format**
   ```bash
   # Correct format
   Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
   ```

### Rate Limiting

**Error Messages:**
```
429 Too Many Requests
Rate limit exceeded
```

**Solutions:**

1. **Implement Exponential Backoff**
   ```python
   import time
   import requests
   from requests.adapters import HTTPAdapter
   from urllib3.util.retry import Retry
   
   session = requests.Session()
   retry_strategy = Retry(
       total=3,
       backoff_factor=1,
       status_forcelist=[429, 500, 502, 503, 504],
   )
   adapter = HTTPAdapter(max_retries=retry_strategy)
   session.mount("http://", adapter)
   session.mount("https://", adapter)
   ```

2. **Check Rate Limit Headers**
   ```python
   response = requests.get('http://localhost:8000/api/v1/datasets')
   print(f"Rate limit: {response.headers.get('X-RateLimit-Limit')}")
   print(f"Remaining: {response.headers.get('X-RateLimit-Remaining')}")
   print(f"Reset: {response.headers.get('X-RateLimit-Reset')}")
   ```

### Validation Errors

**Error Messages:**
```
422 Unprocessable Entity
Validation failed
Invalid request parameters
```

**Solutions:**

1. **Check Request Format**
   ```python
   # Correct JSON format
   data = {
       "name": "Dataset Name",
       "description": "Dataset description",
       "tags": ["tag1", "tag2"]
   }
   
   response = requests.post(
       'http://localhost:8000/api/v1/datasets',
       json=data,  # Use json parameter, not data
       headers={'Content-Type': 'application/json'}
   )
   ```

2. **Validate Required Fields**
   ```python
   # Check API documentation for required fields
   required_fields = ["name", "file"]
   for field in required_fields:
       if field not in request_data:
           print(f"Missing required field: {field}")
   ```

## Database Issues

### Connection Pool Exhausted

**Error Messages:**
```
QueuePool limit of size 20 overflow 30 reached
Connection pool exhausted
```

**Solutions:**

1. **Increase Pool Size**
   ```python
   # Increase connection pool settings
   engine = create_engine(
       DATABASE_URL,
       pool_size=50,
       max_overflow=100,
       pool_timeout=30
   )
   ```

2. **Fix Connection Leaks**
   ```python
   # Always close connections
   try:
       with engine.connect() as conn:
           result = conn.execute(query)
           return result.fetchall()
   except Exception as e:
       logger.error(f"Database error: {e}")
       raise
   ```

### Migration Failures

**Error Messages:**
```
Migration failed
Duplicate column name
Table already exists
```

**Solutions:**

1. **Check Migration Status**
   ```bash
   # Check current migration version
   python -m alembic current
   
   # Check migration history
   python -m alembic history
   ```

2. **Reset Migrations**
   ```bash
   # Downgrade to base
   python -m alembic downgrade base
   
   # Upgrade to latest
   python -m alembic upgrade head
   ```

3. **Manual Migration Fix**
   ```sql
   -- Check if table exists before creating
   SELECT EXISTS (
       SELECT FROM information_schema.tables 
       WHERE table_name = 'your_table_name'
   );
   ```

## Authentication Problems

### SSO Integration Issues

**Error Messages:**
```
SAML assertion validation failed
OAuth callback error
Invalid redirect URI
```

**Solutions:**

1. **Verify SSO Configuration**
   ```python
   # Check SSO settings
   SSO_CONFIG = {
       "entity_id": "your-entity-id",
       "sso_url": "https://your-idp.com/sso",
       "x509cert": "your-certificate",
       "redirect_uri": "https://your-app.com/auth/callback"
   }
   ```

2. **Test SSO Endpoints**
   ```bash
   # Test IdP metadata endpoint
   curl -v "https://your-idp.com/metadata"
   
   # Test callback URL
   curl -v "https://your-app.com/auth/callback"
   ```

### Permission Denied

**Error Messages:**
```
403 Forbidden
Insufficient permissions
Access denied
```

**Solutions:**

1. **Check User Roles**
   ```sql
   -- Check user permissions
   SELECT u.username, r.name as role, p.name as permission
   FROM users u
   JOIN user_roles ur ON u.id = ur.user_id
   JOIN roles r ON ur.role_id = r.id
   JOIN role_permissions rp ON r.id = rp.role_id
   JOIN permissions p ON rp.permission_id = p.id
   WHERE u.username = 'your_username';
   ```

2. **Update User Permissions**
   ```python
   # Grant permissions via API
   response = requests.post(
       'http://localhost:8000/api/v1/admin/users/123/permissions',
       json={'permissions': ['dataset:read', 'dataset:write']},
       headers={'Authorization': f'Bearer {admin_token}'}
   )
   ```

## Monitoring and Alerts

### Missing Alerts

**Symptoms:**
- Expected alerts not received
- Alert configuration not working
- Notification delivery failures

**Solutions:**

1. **Check Alert Configuration**
   ```python
   # Verify alert rules
   alert_config = {
       "quality_threshold": 0.8,
       "drift_threshold": 0.2,
       "notification_channels": ["email", "slack"],
       "enabled": True
   }
   ```

2. **Test Notification Channels**
   ```bash
   # Test email configuration
   python -c "
   import smtplib
   from email.mime.text import MIMEText
   
   msg = MIMEText('Test message')
   msg['Subject'] = 'Test Alert'
   msg['From'] = 'alerts@example.com'
   msg['To'] = 'admin@example.com'
   
   server = smtplib.SMTP('localhost', 587)
   server.send_message(msg)
   server.quit()
   print('Email sent successfully')
   "
   ```

3. **Check Alert History**
   ```sql
   -- Check recent alerts
   SELECT * FROM alerts 
   WHERE created_at > NOW() - INTERVAL '24 hours'
   ORDER BY created_at DESC;
   ```

### Monitoring Dashboard Issues

**Symptoms:**
- Dashboard not loading
- Missing metrics
- Incorrect data visualization

**Solutions:**

1. **Check Data Sources**
   ```python
   # Verify metrics collection
   from ai_data_readiness.core.monitoring import MetricsCollector
   
   collector = MetricsCollector()
   metrics = collector.get_system_metrics()
   print(f"Available metrics: {list(metrics.keys())}")
   ```

2. **Refresh Dashboard Cache**
   ```bash
   # Clear dashboard cache
   redis-cli FLUSHDB
   
   # Restart dashboard service
   systemctl restart ai-data-readiness-dashboard
   ```

## Getting Support

### Before Contacting Support

1. **Gather Information**
   - Error messages and stack traces
   - System configuration details
   - Steps to reproduce the issue
   - Expected vs. actual behavior

2. **Check Logs**
   ```bash
   # Application logs
   tail -n 100 logs/app.log
   
   # System logs
   journalctl -u ai-data-readiness -n 100
   
   # Database logs
   tail -n 100 /var/log/postgresql/postgresql.log
   ```

3. **System Information**
   ```bash
   # System details
   uname -a
   python --version
   pip list | grep -E "(pandas|numpy|scikit-learn)"
   
   # Resource usage
   free -h
   df -h
   ```

### Support Channels

1. **Documentation**
   - [User Guide](../user-guide/README.md)
   - [Developer Guide](../developer-guide/README.md)
   - [API Documentation](../api/README.md)

2. **Community Support**
   - GitHub Issues: https://github.com/your-org/ai-data-readiness-platform/issues
   - Community Forum: https://community.example.com
   - Stack Overflow: Tag with `ai-data-readiness`

3. **Professional Support**
   - Email: support@example.com
   - Support Portal: https://support.example.com
   - Phone: +1-800-SUPPORT (business hours)

### Creating Effective Bug Reports

Include the following information:

1. **Environment Details**
   - Operating system and version
   - Python version
   - Platform version
   - Deployment method (Docker, bare metal, cloud)

2. **Problem Description**
   - Clear, concise description
   - Steps to reproduce
   - Expected behavior
   - Actual behavior

3. **Error Information**
   - Complete error messages
   - Stack traces
   - Log excerpts
   - Screenshots (if applicable)

4. **Additional Context**
   - Data characteristics (size, format, etc.)
   - Configuration settings
   - Recent changes
   - Workarounds attempted

### Emergency Procedures

For critical production issues:

1. **Immediate Actions**
   - Check system health dashboard
   - Review recent deployments or changes
   - Implement emergency rollback if necessary
   - Notify stakeholders

2. **Escalation Process**
   - Contact on-call support: +1-800-EMERGENCY
   - Create high-priority support ticket
   - Engage incident response team
   - Document incident timeline

3. **Recovery Steps**
   - Identify root cause
   - Implement fix or workaround
   - Validate system recovery
   - Conduct post-incident review

Remember: When in doubt, don't hesitate to reach out for help. Our support team is here to ensure your success with the AI Data Readiness Platform.