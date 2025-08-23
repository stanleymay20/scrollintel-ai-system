# ScrollIntel.com Production Environment
# Copy this to .env.production and fill in your values

# Domain Configuration
DOMAIN=scrollintel.com
API_DOMAIN=api.scrollintel.com
APP_DOMAIN=app.scrollintel.com

# Application Settings
NODE_ENV=production
DEBUG=false
LOG_LEVEL=INFO

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# Database (Replace with your actual database URL)
DATABASE_URL=postgresql://scrollintel:your_password@postgres:5432/scrollintel
POSTGRES_USER=scrollintel
POSTGRES_PASSWORD=your_secure_password

# Redis
REDIS_URL=redis://redis:6379

# Security (Generate secure keys!)
JWT_SECRET_KEY=your_super_secure_jwt_secret_key_here
CORS_ORIGINS=https://scrollintel.com,https://app.scrollintel.com

# AI Services (Add your OpenAI API key)
OPENAI_API_KEY=sk-your-openai-api-key-here

# Object Storage
MINIO_ENDPOINT=storage.scrollintel.com
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=your_secure_minio_password

# Monitoring
PROMETHEUS_ENABLED=true
GRAFANA_ENABLED=true
GRAFANA_PASSWORD=your_secure_grafana_password

# Email (Optional - for notifications)
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your_email@gmail.com
SMTP_PASS=your_app_password

# Backup Configuration
BACKUP_ENABLED=true
BACKUP_SCHEDULE=0 2 * * *
BACKUP_RETENTION_DAYS=7

# SSL Configuration
SSL_EMAIL=admin@scrollintel.com
CERTBOT_EMAIL=admin@scrollintel.com