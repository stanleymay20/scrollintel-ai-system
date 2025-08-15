# ScrollIntel Security Infrastructure Deployment Guide

This guide covers the deployment and configuration of SSL, CDN, and security infrastructure for ScrollIntel's production launch.

## Overview

The security infrastructure includes:
- SSL/TLS certificates with modern security standards
- Content Delivery Network (CDN) integration
- DDoS protection and rate limiting
- Automated backup and disaster recovery
- Security headers and content security policy
- Firewall and intrusion prevention
- Monitoring and alerting

## Prerequisites

### System Requirements
- Ubuntu 20.04+ or CentOS 8+ server
- Root access
- Domain name configured (scrollintel.com)
- Minimum 4GB RAM, 2 CPU cores
- 100GB+ storage for backups

### Required Environment Variables
```bash
# SSL Configuration
DOMAIN=scrollintel.com
EMAIL=admin@scrollintel.com

# Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=scrollintel
DB_USER=scrollintel
DB_PASSWORD=your_secure_password

# Cloudflare CDN (optional)
CLOUDFLARE_API_TOKEN=your_api_token
CLOUDFLARE_ZONE_ID=your_zone_id
CLOUDFLARE_EMAIL=your_email

# Backup Configuration
BACKUP_S3_BUCKET=scrollintel-backups
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_REGION=us-east-1

# Notification Configuration
BACKUP_EMAIL=admin@scrollintel.com
SLACK_WEBHOOK_URL=your_slack_webhook

# SMTP Configuration (for notifications)
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your_email
SMTP_PASSWORD=your_app_password
```

## Deployment Steps

### 1. Prepare the Server

```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install required packages
sudo apt install -y nginx certbot python3-certbot-nginx fail2ban postgresql-client

# Install Python dependencies
pip3 install boto3 requests
```

### 2. Deploy Security Infrastructure

```bash
# Clone the repository
git clone https://github.com/your-org/scrollintel.git
cd scrollintel

# Set environment variables
export DOMAIN=scrollintel.com
export EMAIL=admin@scrollintel.com
# ... (set other variables as needed)

# Run the deployment script
sudo python3 scripts/deploy-security-infrastructure.py --environment production
```

### 3. Configure DNS Records

Update your DNS records to point to your server:

```
A     @              YOUR_SERVER_IP
A     www            YOUR_SERVER_IP
A     api            YOUR_SERVER_IP
CNAME cdn            scrollintel.com
```

### 4. Verify SSL Certificates

```bash
# Check SSL certificate
openssl s_client -servername scrollintel.com -connect scrollintel.com:443 -showcerts

# Test SSL configuration
curl -I https://scrollintel.com
```

### 5. Test Security Headers

```bash
# Test security headers
curl -I https://scrollintel.com

# Expected headers:
# Strict-Transport-Security: max-age=63072000; includeSubDomains; preload
# X-Frame-Options: DENY
# X-Content-Type-Options: nosniff
# X-XSS-Protection: 1; mode=block
# Content-Security-Policy: ...
```

### 6. Validate Rate Limiting

```bash
# Test API rate limiting
for i in {1..20}; do curl -s -o /dev/null -w "%{http_code}\n" https://scrollintel.com/api/health; done

# Should see some 429 responses after several requests
```

## Configuration Files

### SSL Configuration
- **Location**: `/etc/nginx/ssl-config.conf`
- **Purpose**: SSL/TLS security settings
- **Key Features**:
  - TLS 1.2 and 1.3 support
  - Strong cipher suites
  - OCSP stapling
  - Perfect forward secrecy

### Security Headers
- **Location**: `/etc/nginx/security-headers.conf`
- **Purpose**: HTTP security headers
- **Key Features**:
  - HSTS with preload
  - Content Security Policy
  - XSS protection
  - Clickjacking prevention

### Rate Limiting
- **Location**: `/etc/nginx/conf.d/rate-limiting.conf`
- **Purpose**: DDoS protection and rate limiting
- **Key Features**:
  - API rate limiting (5 req/s)
  - Auth rate limiting (2 req/s)
  - Connection limits
  - Slow loris protection

### Bot Protection
- **Location**: `/etc/nginx/conf.d/bot-protection.conf`
- **Purpose**: Block malicious bots and scrapers
- **Key Features**:
  - Bad bot detection
  - Suspicious request patterns
  - Legitimate bot allowlist

### Geo Blocking
- **Location**: `/etc/nginx/conf.d/geo-blocking.conf`
- **Purpose**: Geographic access control
- **Key Features**:
  - Country-based blocking
  - Configurable allowlist
  - Security-focused defaults

## Backup and Recovery

### Automated Backups
- **Database**: Daily at 2 AM
- **Files**: Weekly on Sunday at 3 AM
- **Cleanup**: Weekly on Sunday at 4 AM
- **Retention**: 30 days for DB, 7 days for files

### Backup Locations
- **Local**: `/var/backups/scrollintel/`
- **Remote**: S3 bucket (if configured)
- **Encryption**: AES-256 for S3 uploads

### Manual Backup Commands
```bash
# Database backup
python3 scripts/setup-backup-recovery.py --database-backup

# File backup
python3 scripts/setup-backup-recovery.py --file-backup

# Cleanup old backups
python3 scripts/setup-backup-recovery.py --cleanup
```

### Restore Procedures
```bash
# Restore database from backup
python3 scripts/setup-backup-recovery.py --restore-db /path/to/backup.sql.gz

# Restore files manually
tar -xzf /var/backups/scrollintel/scrollintel_files_YYYYMMDD_HHMMSS.tar.gz -C /
```

## Monitoring and Alerting

### Security Monitoring
- **SSL certificate expiry**: Checked every 6 hours
- **Failed login attempts**: Daily monitoring
- **Backup disk space**: Continuous monitoring
- **DDoS attacks**: Real-time detection

### Log Locations
- **Nginx access**: `/var/log/nginx/access.log`
- **Nginx error**: `/var/log/nginx/error.log`
- **Security monitor**: `/var/log/security-monitor.log`
- **DDoS monitor**: `/var/log/ddos-monitor.log`
- **Backup logs**: `/var/log/scrollintel-backup.log`

### Alert Channels
- **Email**: Configured SMTP server
- **Slack**: Webhook integration
- **System logs**: Syslog integration

## Cloudflare CDN Integration

### Setup Requirements
1. Cloudflare account with domain added
2. API token with Zone:Edit permissions
3. Environment variables configured

### Features Enabled
- **SSL/TLS**: Full (strict) mode
- **Security**: Medium level
- **Performance**: Brotli compression, minification
- **Caching**: Optimized for static assets
- **Firewall**: Custom rules for API protection

### Page Rules
1. `/api/*` - Bypass cache, high security
2. `/*.js` - Cache everything, 1 year TTL
3. `/*.css` - Cache everything, 1 year TTL

## Firewall Configuration

### Fail2Ban Jails
- **SSH**: 3 attempts, 24-hour ban
- **Nginx HTTP Auth**: 3 attempts, 1-hour ban
- **Nginx Rate Limit**: 10 attempts, 2-hour ban
- **ScrollIntel API**: 10 attempts, 1-hour ban

### Custom Filters
- **API abuse**: Multiple 4xx/5xx responses
- **Auth failures**: Failed login attempts
- **Suspicious patterns**: SQL injection, XSS attempts

## Validation and Testing

### Automated Validation
```bash
# Run security validation
python3 scripts/validate-security-infrastructure.py

# Run SSL tests
python3 tests/test_ssl_security_config.py
```

### Manual Testing Checklist

#### SSL/TLS Security
- [ ] SSL certificate is valid and trusted
- [ ] TLS 1.2 and 1.3 are supported
- [ ] Weak ciphers are disabled
- [ ] HSTS header is present
- [ ] Certificate auto-renewal is working

#### Security Headers
- [ ] All required security headers are present
- [ ] Content Security Policy is configured
- [ ] Server information is not disclosed
- [ ] Error pages don't reveal sensitive info

#### Rate Limiting
- [ ] API endpoints are rate limited
- [ ] Auth endpoints have stricter limits
- [ ] Rate limit responses are proper (429)
- [ ] Legitimate traffic is not blocked

#### Backup System
- [ ] Automated backups are running
- [ ] Backup files are created successfully
- [ ] S3 uploads are working (if configured)
- [ ] Restore procedures work correctly

#### Monitoring
- [ ] Security monitoring is active
- [ ] Alerts are being sent
- [ ] Log files are being written
- [ ] DDoS protection is active

## Troubleshooting

### Common Issues

#### SSL Certificate Problems
```bash
# Check certificate status
certbot certificates

# Renew certificate manually
certbot renew --nginx

# Test certificate renewal
certbot renew --dry-run
```

#### Nginx Configuration Errors
```bash
# Test nginx configuration
nginx -t

# Reload nginx configuration
systemctl reload nginx

# Check nginx status
systemctl status nginx
```

#### Rate Limiting Issues
```bash
# Check rate limiting zones
grep -r "limit_req_zone" /etc/nginx/

# Monitor rate limiting in real-time
tail -f /var/log/nginx/error.log | grep "limiting requests"
```

#### Backup Failures
```bash
# Check backup logs
tail -f /var/log/scrollintel-backup.log

# Test database connection
pg_isready -h $DB_HOST -p $DB_PORT -U $DB_USER

# Check S3 connectivity
aws s3 ls s3://$BACKUP_S3_BUCKET
```

### Performance Optimization

#### Nginx Tuning
```nginx
# Worker processes
worker_processes auto;
worker_connections 2048;

# Buffer sizes
client_body_buffer_size 128k;
client_max_body_size 100m;
large_client_header_buffers 4 256k;

# Timeouts
keepalive_timeout 65;
client_body_timeout 60;
client_header_timeout 60;
send_timeout 60;
```

#### SSL Optimization
```nginx
# SSL session caching
ssl_session_cache shared:SSL:50m;
ssl_session_timeout 1d;
ssl_session_tickets off;

# OCSP stapling
ssl_stapling on;
ssl_stapling_verify on;
```

## Security Best Practices

### Regular Maintenance
1. **Weekly**: Review security logs
2. **Monthly**: Update system packages
3. **Quarterly**: Security audit and penetration testing
4. **Annually**: SSL certificate renewal (automated)

### Security Hardening
1. **SSH**: Key-based authentication only
2. **Firewall**: Minimal open ports
3. **Updates**: Automatic security updates
4. **Monitoring**: 24/7 security monitoring
5. **Backups**: Regular backup testing

### Incident Response
1. **Detection**: Automated monitoring and alerts
2. **Assessment**: Log analysis and impact evaluation
3. **Containment**: Immediate threat mitigation
4. **Recovery**: Service restoration procedures
5. **Lessons Learned**: Post-incident review

## Support and Maintenance

### Emergency Contacts
- **Primary**: admin@scrollintel.com
- **Secondary**: security@scrollintel.com
- **Phone**: +1-XXX-XXX-XXXX

### Maintenance Windows
- **Scheduled**: Sunday 2-4 AM UTC
- **Emergency**: As needed with notifications
- **SSL Renewal**: Automated, no downtime

### Documentation Updates
This documentation should be updated whenever:
- Configuration changes are made
- New security features are added
- Incidents occur and procedures change
- Best practices evolve

---

For additional support or questions, please contact the ScrollIntel security team.