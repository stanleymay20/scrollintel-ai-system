# ScrollIntel.com Production Environment
NODE_ENV=production
ENVIRONMENT=production

# Domain Configuration
DOMAIN=scrollintel.com
API_DOMAIN=api.scrollintel.com
APP_DOMAIN=app.scrollintel.com

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
WORKERS=4

# Database (Update with your actual database URL)
DATABASE_URL=postgresql://scrollintel:your_password@localhost:5432/scrollintel_prod
REDIS_URL=redis://localhost:6379/0

# Security (IMPORTANT: Change these!)
SECRET_KEY=your-super-secure-secret-key-change-this-now
JWT_SECRET_KEY=your-jwt-secret-key-change-this-now

# AI Services (Add your actual API keys)
OPENAI_API_KEY=your-openai-api-key-here
ANTHROPIC_API_KEY=your-anthropic-api-key-here

# CORS
CORS_ORIGINS=https://scrollintel.com,https://app.scrollintel.com

# Features
ENABLE_MONITORING=true
ENABLE_ANALYTICS=true
ENABLE_CACHING=true

# Email (Optional)
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=noreply@scrollintel.com
SMTP_PASSWORD=your-email-password