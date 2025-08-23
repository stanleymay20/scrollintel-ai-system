# ScrollIntel.com Deployment Guide

## Domain Setup Complete!
Your domain: scrollintel.com
API endpoint: api.scrollintel.com

## Quick Deployment Options

### Option 1: Vercel + Render (RECOMMENDED)

#### Deploy Frontend to Vercel:
1. Go to https://vercel.com
2. Import from GitHub: your ScrollIntel repository
3. Set root directory to 'frontend'
4. Add custom domain: scrollintel.com
5. Deploy!

#### Deploy Backend to Render:
1. Go to https://render.com
2. Create new Web Service from GitHub
3. Use the render.yaml configuration
4. Add custom domain: api.scrollintel.com
5. Deploy!

### Option 2: Railway (All-in-One)
1. Go to https://railway.app
2. Deploy from GitHub repository
3. Add custom domain in Railway dashboard
4. Configure environment variables

## DNS Configuration

Add these DNS records to your domain registrar:

```
Type    Name    Value                           TTL
A       @       [Your deployment IP]            300
CNAME   www     scrollintel.com                 300
CNAME   api     [Your API deployment URL]      300
```

## Environment Variables

Set these in your deployment platform:

```
ENVIRONMENT=production
DEBUG=false
OPENAI_API_KEY=your_openai_api_key
JWT_SECRET_KEY=your_jwt_secret_key
ALLOWED_ORIGINS=https://scrollintel.com,https://www.scrollintel.com
```

## Expected URLs After Deployment

- Main Site: https://scrollintel.com
- API: https://api.scrollintel.com
- API Docs: https://api.scrollintel.com/docs
- Health Check: https://api.scrollintel.com/health

## Test Your Deployment

After deployment, test these URLs to ensure everything works:
- https://scrollintel.com (should load the frontend)
- https://api.scrollintel.com/health (should return {"status": "healthy"})
- https://api.scrollintel.com/docs (should show API documentation)

ScrollIntel.com is ready to go live!
