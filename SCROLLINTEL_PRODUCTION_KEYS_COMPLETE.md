# ScrollIntel Production Keys - Complete List

This document contains all the API keys, configuration variables, and secrets needed for optimized production deployment of ScrollIntel.

## Core Database & Infrastructure

### PostgreSQL Database
```bash
# Primary Database
DATABASE_URL=postgresql://username:password@host:port/scrollintel_prod
DB_HOST=your-postgres-host
DB_PORT=5432
DB_NAME=scrollintel_prod
DB_USER=scrollintel_user
DB_PASSWORD=your-secure-password

# Read Replicas (for scaling)
DATABASE_READ_REPLICA_URL=postgresql://username:password@read-host:port/scrollintel_prod
DATABASE_ANALYTICS_URL=postgresql://username:password@analytics-host:port/scrollintel_analytics
```

### Redis Cache & Sessions
```bash
REDIS_URL=redis://username:password@redis-host:port/0
REDIS_CACHE_URL=redis://username:password@cache-host:port/1
REDIS_SESSION_URL=redis://username:password@session-host:port/2
REDIS_QUEUE_URL=redis://username:password@queue-host:port/3
```

## AI Model APIs

### OpenAI
```bash
OPENAI_API_KEY=sk-your-openai-api-key
OPENAI_ORG_ID=org-your-organization-id
OPENAI_PROJECT_ID=proj_your-project-id
```

### Anthropic Claude
```bash
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key
```

### Google AI (Gemini)
```bash
GOOGLE_AI_API_KEY=your-google-ai-key
GOOGLE_CLOUD_PROJECT_ID=your-project-id
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
```

### Azure OpenAI
```bash
AZURE_OPENAI_API_KEY=your-azure-openai-key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-02-15-preview
```

### Hugging Face
```bash
HUGGINGFACE_API_TOKEN=hf_your-huggingface-token
```

## Visual Generation APIs

### DALL-E 3
```bash
DALLE3_API_KEY=sk-your-dalle3-key
DALLE3_ENDPOINT=https://api.openai.com/v1/images/generations
```

### Midjourney (via API)
```bash
MIDJOURNEY_API_KEY=your-midjourney-api-key
MIDJOURNEY_WEBHOOK_URL=https://your-domain.com/webhooks/midjourney
```

### Stable Diffusion
```bash
STABILITY_API_KEY=sk-your-stability-key
STABILITY_HOST=https://api.stability.ai
```

### Replicate
```bash
REPLICATE_API_TOKEN=r8_your-replicate-token
```

## Cloud Storage & CDN

### AWS S3
```bash
AWS_ACCESS_KEY_ID=your-aws-access-key
AWS_SECRET_ACCESS_KEY=your-aws-secret-key
AWS_REGION=us-east-1
AWS_S3_BUCKET=scrollintel-production
AWS_CLOUDFRONT_DISTRIBUTION_ID=your-distribution-id
```

### Google Cloud Storage
```bash
GOOGLE_CLOUD_STORAGE_BUCKET=scrollintel-gcs-prod
GCS_CREDENTIALS_PATH=/path/to/gcs-credentials.json
```

### Azure Blob Storage
```bash
AZURE_STORAGE_ACCOUNT_NAME=scrollintelstorage
AZURE_STORAGE_ACCOUNT_KEY=your-azure-storage-key
AZURE_STORAGE_CONTAINER_NAME=production
```

## Security & Authentication

### JWT & Session Management
```bash
JWT_SECRET_KEY=your-super-secure-jwt-secret-256-bit
JWT_ALGORITHM=HS256
JWT_EXPIRATION_HOURS=24
SESSION_SECRET_KEY=your-session-secret-key
```

### OAuth Providers
```bash
# Google OAuth
GOOGLE_CLIENT_ID=your-google-client-id
GOOGLE_CLIENT_SECRET=your-google-client-secret

# GitHub OAuth
GITHUB_CLIENT_ID=your-github-client-id
GITHUB_CLIENT_SECRET=your-github-client-secret

# Microsoft OAuth
MICROSOFT_CLIENT_ID=your-microsoft-client-id
MICROSOFT_CLIENT_SECRET=your-microsoft-client-secret
```

### API Security
```bash
API_RATE_LIMIT_PER_MINUTE=1000
API_RATE_LIMIT_PER_HOUR=10000
CORS_ORIGINS=https://scrollintel.com,https://app.scrollintel.com
ALLOWED_HOSTS=scrollintel.com,app.scrollintel.com,api.scrollintel.com
```

## Monitoring & Analytics

### Application Performance Monitoring
```bash
# Sentry
SENTRY_DSN=https://your-sentry-dsn@sentry.io/project-id
SENTRY_ENVIRONMENT=production

# New Relic
NEW_RELIC_LICENSE_KEY=your-newrelic-license-key
NEW_RELIC_APP_NAME=ScrollIntel-Production

# DataDog
DATADOG_API_KEY=your-datadog-api-key
DATADOG_APP_KEY=your-datadog-app-key
```

### Business Analytics
```bash
# Google Analytics
GOOGLE_ANALYTICS_ID=G-your-ga4-id
GOOGLE_TAG_MANAGER_ID=GTM-your-gtm-id

# Mixpanel
MIXPANEL_TOKEN=your-mixpanel-token

# Amplitude
AMPLITUDE_API_KEY=your-amplitude-key
```

## Communication & Notifications

### Email Services
```bash
# SendGrid
SENDGRID_API_KEY=SG.your-sendgrid-api-key

# Mailgun
MAILGUN_API_KEY=your-mailgun-api-key
MAILGUN_DOMAIN=mg.scrollintel.com

# AWS SES
AWS_SES_REGION=us-east-1
AWS_SES_ACCESS_KEY_ID=your-ses-access-key
AWS_SES_SECRET_ACCESS_KEY=your-ses-secret-key
```

### SMS & Push Notifications
```bash
# Twilio
TWILIO_ACCOUNT_SID=your-twilio-account-sid
TWILIO_AUTH_TOKEN=your-twilio-auth-token
TWILIO_PHONE_NUMBER=+1234567890

# Firebase Cloud Messaging
FCM_SERVER_KEY=your-fcm-server-key
FCM_PROJECT_ID=your-firebase-project-id
```

### Slack Integration
```bash
SLACK_BOT_TOKEN=xoxb-your-slack-bot-token
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/your/webhook/url
```

## Payment Processing

### Stripe
```bash
STRIPE_PUBLISHABLE_KEY=pk_live_your-stripe-publishable-key
STRIPE_SECRET_KEY=sk_live_your-stripe-secret-key
STRIPE_WEBHOOK_SECRET=whsec_your-webhook-secret
```

### PayPal
```bash
PAYPAL_CLIENT_ID=your-paypal-client-id
PAYPAL_CLIENT_SECRET=your-paypal-client-secret
PAYPAL_MODE=live
```

## Search & Indexing

### Elasticsearch
```bash
ELASTICSEARCH_URL=https://your-elasticsearch-cluster.com:9200
ELASTICSEARCH_USERNAME=elastic
ELASTICSEARCH_PASSWORD=your-elastic-password
```

### Algolia
```bash
ALGOLIA_APPLICATION_ID=your-algolia-app-id
ALGOLIA_API_KEY=your-algolia-api-key
ALGOLIA_SEARCH_KEY=your-algolia-search-key
```

## Message Queues & Background Jobs

### Celery with Redis
```bash
CELERY_BROKER_URL=redis://username:password@redis-host:port/4
CELERY_RESULT_BACKEND=redis://username:password@redis-host:port/5
```

### RabbitMQ
```bash
RABBITMQ_URL=amqp://username:password@rabbitmq-host:port/vhost
```

## External Integrations

### CRM Systems
```bash
# Salesforce
SALESFORCE_CLIENT_ID=your-salesforce-client-id
SALESFORCE_CLIENT_SECRET=your-salesforce-client-secret
SALESFORCE_USERNAME=your-salesforce-username
SALESFORCE_PASSWORD=your-salesforce-password
SALESFORCE_SECURITY_TOKEN=your-security-token

# HubSpot
HUBSPOT_API_KEY=your-hubspot-api-key
```

### Business Intelligence
```bash
# Tableau
TABLEAU_SERVER_URL=https://your-tableau-server.com
TABLEAU_USERNAME=your-tableau-username
TABLEAU_PASSWORD=your-tableau-password

# Power BI
POWERBI_CLIENT_ID=your-powerbi-client-id
POWERBI_CLIENT_SECRET=your-powerbi-client-secret
```

## Development & CI/CD

### GitHub Actions
```bash
GITHUB_TOKEN=ghp_your-github-token
DOCKER_HUB_USERNAME=your-dockerhub-username
DOCKER_HUB_TOKEN=your-dockerhub-token
```

### Container Registry
```bash
# Docker Hub
DOCKER_REGISTRY_URL=docker.io
DOCKER_REGISTRY_USERNAME=your-username
DOCKER_REGISTRY_PASSWORD=your-password

# AWS ECR
AWS_ECR_REGISTRY=123456789012.dkr.ecr.us-east-1.amazonaws.com
```

## Environment Configuration

### Application Settings
```bash
# Environment
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO

# Application
APP_NAME=ScrollIntel
APP_VERSION=1.0.0
APP_URL=https://scrollintel.com
API_URL=https://api.scrollintel.com
FRONTEND_URL=https://app.scrollintel.com

# Security
SECURE_SSL_REDIRECT=true
SECURE_HSTS_SECONDS=31536000
SECURE_CONTENT_TYPE_NOSNIFF=true
SECURE_BROWSER_XSS_FILTER=true
```

### Performance & Scaling
```bash
# Worker Configuration
WORKER_PROCESSES=4
WORKER_CONNECTIONS=1000
WORKER_TIMEOUT=30

# Cache Configuration
CACHE_TTL=3600
CACHE_MAX_ENTRIES=10000

# Rate Limiting
RATE_LIMIT_ENABLED=true
RATE_LIMIT_REQUESTS_PER_MINUTE=100
```

## Backup & Disaster Recovery

### Backup Services
```bash
# AWS Backup
AWS_BACKUP_VAULT_NAME=scrollintel-backup-vault
AWS_BACKUP_ROLE_ARN=arn:aws:iam::account:role/BackupRole

# Database Backups
DB_BACKUP_S3_BUCKET=scrollintel-db-backups
DB_BACKUP_RETENTION_DAYS=30
```

## Compliance & Legal

### GDPR & Privacy
```bash
GDPR_COMPLIANCE_ENABLED=true
COOKIE_CONSENT_REQUIRED=true
DATA_RETENTION_DAYS=2555  # 7 years
```

### Audit Logging
```bash
AUDIT_LOG_ENABLED=true
AUDIT_LOG_RETENTION_DAYS=2555
COMPLIANCE_REPORTING_ENABLED=true
```

## Health Checks & Status

### Health Check Endpoints
```bash
HEALTH_CHECK_ENABLED=true
HEALTH_CHECK_PATH=/health
STATUS_PAGE_ENABLED=true
UPTIME_MONITORING_ENABLED=true
```

## Feature Flags

### Feature Toggle Service
```bash
# LaunchDarkly
LAUNCHDARKLY_SDK_KEY=sdk-your-launchdarkly-key

# Split.io
SPLIT_API_KEY=your-split-api-key
```

## Quantum & Advanced AI

### Quantum Computing APIs
```bash
IBM_QUANTUM_TOKEN=your-ibm-quantum-token
GOOGLE_QUANTUM_AI_KEY=your-google-quantum-key
```

### Advanced ML Platforms
```bash
# Weights & Biases
WANDB_API_KEY=your-wandb-api-key

# MLflow
MLFLOW_TRACKING_URI=https://your-mlflow-server.com
MLFLOW_TRACKING_USERNAME=your-username
MLFLOW_TRACKING_PASSWORD=your-password
```

---

## Security Notes

1. **Never commit these keys to version control**
2. **Use environment-specific key management services**:
   - AWS Secrets Manager
   - Azure Key Vault
   - Google Secret Manager
   - HashiCorp Vault

3. **Rotate keys regularly** (recommended every 90 days)
4. **Use least privilege principle** for all API keys
5. **Monitor key usage** and set up alerts for unusual activity

## Key Management Best Practices

### Production Deployment
```bash
# Use secrets management
kubectl create secret generic scrollintel-secrets \
  --from-env-file=.env.production

# Or with Helm
helm install scrollintel ./helm-chart \
  --set-file secrets=.env.production
```

### Environment Variables Loading
```python
# In your application
import os
from dotenv import load_dotenv

# Load environment-specific variables
load_dotenv(f'.env.{os.getenv("ENVIRONMENT", "production")}')
```

---

**Last Updated**: December 2024
**Version**: 1.0.0
**Maintained By**: ScrollIntel DevOps Team