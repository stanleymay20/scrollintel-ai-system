# 🚂 ScrollIntel Railway Deployment Guide

Deploy ScrollIntel to Railway in minutes with this comprehensive guide.

## 🚀 Quick Deploy to Railway

### Option 1: One-Click Deploy (Recommended)

[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/template/scrollintel?referralCode=scrollintel)

### Option 2: Manual Deployment

1. **Go to Railway**: https://railway.app
2. **Sign up/Login** with GitHub
3. **New Project** → **Deploy from GitHub repo**
4. **Select Repository**: `stanleymay20/scrollintel-ai-system`
5. **Configure Environment Variables** (see below)
6. **Deploy!**

## 🔑 Required Environment Variables

Add these environment variables in Railway dashboard:

### Essential Variables
```bash
# AI Service Keys (REQUIRED)
OPENAI_API_KEY=sk-your-openai-api-key-here
ANTHROPIC_API_KEY=your-anthropic-api-key-here

# Security Keys (Generate secure random strings)
SECRET_KEY=your-super-secure-secret-key-32-chars
JWT_SECRET_KEY=your-jwt-secret-key-32-chars

# Environment
NODE_ENV=production
ENVIRONMENT=production
```

### Optional Variables
```bash
# Additional AI Services
GOOGLE_API_KEY=your-google-api-key
HUGGINGFACE_API_KEY=your-huggingface-key

# Email (Optional)
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=noreply@yourdomain.com
SMTP_PASSWORD=your-email-password

# Analytics (Optional)
GOOGLE_ANALYTICS_ID=your-ga-id
SENTRY_DSN=your-sentry-dsn
```

## 🗄️ Database Setup

Railway will automatically provide:
- **PostgreSQL Database** (DATABASE_URL)
- **Redis Cache** (REDIS_URL)

These are automatically configured - no manual setup needed!

## 🌐 Custom Domain Setup

### Using Railway Domain (Immediate)
Your app will be available at: `https://your-app-name.up.railway.app`

### Using Custom Domain (scrollintel.com)
1. **Go to Railway Dashboard** → Your Project → Settings
2. **Add Custom Domain**: `scrollintel.com`
3. **Configure DNS** in your domain provider:
   ```
   Type    Name    Value
   CNAME   @       your-app-name.up.railway.app
   CNAME   www     your-app-name.up.railway.app
   ```
4. **SSL Certificate** is automatically provided by Railway

## 📊 What You Get After Deployment

### 🌐 Access Points
- **Main Application**: https://your-app.up.railway.app
- **API**: https://your-app.up.railway.app/api
- **API Documentation**: https://your-app.up.railway.app/docs
- **Health Check**: https://your-app.up.railway.app/health

### 🤖 AI Agents Available
1. **CTO Agent** - Strategic technology leadership
2. **Data Scientist Agent** - Advanced analytics & ML
3. **ML Engineer Agent** - Model building & deployment
4. **AI Engineer Agent** - AI system architecture
5. **Business Analyst Agent** - Business intelligence
6. **QA Agent** - Quality assurance & testing
7. **AutoDev Agent** - Automated development
8. **Forecast Agent** - Predictive analytics
9. **Visualization Agent** - Data visualization
10. **Ethics Agent** - AI ethics & compliance
11. **Security Agent** - Security analysis
12. **Performance Agent** - System optimization
13. **Compliance Agent** - Regulatory compliance
14. **Innovation Agent** - R&D & innovation
15. **Executive Agent** - Executive reporting

### 📈 Platform Features
- **File Processing**: Up to 50GB files
- **Concurrent Users**: 1000+ simultaneous users
- **Processing Speed**: 770,000+ rows/second
- **Real-time Analytics**: Live business intelligence
- **Visual Generation**: AI-powered image/video creation
- **Advanced Security**: JWT auth, rate limiting, audit logs
- **Monitoring**: Built-in health checks and metrics

## 🔧 Railway Configuration Files

The repository includes Railway-specific configuration:

- `railway.json` - Railway deployment configuration
- `railway.toml` - Railway build and deploy settings
- `.env.railway` - Railway environment template
- `Dockerfile` - Production-ready container

## 🚀 Deployment Steps

### 1. Fork/Clone Repository
```bash
git clone https://github.com/stanleymay20/scrollintel-ai-system.git
cd scrollintel-ai-system
```

### 2. Deploy to Railway
1. Go to https://railway.app
2. Click "New Project"
3. Select "Deploy from GitHub repo"
4. Choose your forked repository
5. Railway will automatically detect the configuration

### 3. Add Environment Variables
In Railway dashboard, add the required environment variables listed above.

### 4. Deploy!
Railway will automatically build and deploy your application.

## 📋 Post-Deployment Checklist

After deployment, verify:
- [ ] Application loads at Railway URL
- [ ] API endpoints respond correctly
- [ ] Health check passes
- [ ] Database connection works
- [ ] AI agents are accessible
- [ ] File upload works
- [ ] Authentication works

## 🔍 Testing Your Deployment

### Health Check
```bash
curl https://your-app.up.railway.app/health
```

### API Test
```bash
curl https://your-app.up.railway.app/api/agents
```

### Upload Test
Visit your app and try uploading a CSV file to test the AI agents.

## 🛠️ Management & Monitoring

### Railway Dashboard
- **Logs**: View real-time application logs
- **Metrics**: Monitor CPU, memory, and network usage
- **Deployments**: Track deployment history
- **Environment**: Manage environment variables

### Application Monitoring
- **Health Endpoint**: `/health`
- **Metrics Endpoint**: `/metrics`
- **API Documentation**: `/docs`

## 🔄 Updates & Redeployment

### Automatic Deployment
Railway automatically redeploys when you push to your main branch.

### Manual Deployment
1. Push changes to GitHub
2. Railway will automatically detect and redeploy
3. Monitor deployment in Railway dashboard

## 💰 Railway Pricing

### Hobby Plan (Free)
- $5/month in usage credits
- Perfect for testing and small projects
- Automatic scaling

### Pro Plan ($20/month)
- $20/month in usage credits
- Priority support
- Advanced features

### Usage-Based Pricing
- Pay only for what you use
- CPU, Memory, Network, Storage
- Transparent pricing

## 🆘 Troubleshooting

### Common Issues

**1. Build Failures**
- Check Railway build logs
- Verify Dockerfile syntax
- Ensure all dependencies are listed

**2. Environment Variables**
- Verify all required variables are set
- Check variable names (case-sensitive)
- Ensure no trailing spaces

**3. Database Connection**
- Railway provides DATABASE_URL automatically
- Check if PostgreSQL service is running
- Verify connection string format

**4. Memory Issues**
- Monitor memory usage in Railway dashboard
- Consider upgrading plan if needed
- Optimize application memory usage

### Getting Help
- **Railway Discord**: https://discord.gg/railway
- **Railway Documentation**: https://docs.railway.app
- **GitHub Issues**: Create issue in repository

## 🎯 Success Metrics

After successful deployment, you should see:
- ✅ Application accessible via Railway URL
- ✅ All 15 AI agents responding
- ✅ File upload and processing working
- ✅ Database queries executing
- ✅ Health checks passing
- ✅ API documentation accessible

## 🌟 Next Steps

1. **Test All Features**: Upload files, try AI agents
2. **Set Up Custom Domain**: Point scrollintel.com to Railway
3. **Configure Monitoring**: Set up alerts and dashboards
4. **Scale as Needed**: Monitor usage and upgrade plan
5. **Integrate APIs**: Connect with your existing systems

---

## 🎉 Congratulations!

Your ScrollIntel platform is now live on Railway! 

**🌐 Access your platform**: https://your-app.up.railway.app

Welcome to the future of AI-powered business intelligence! 🚀

---

*Need help? Check the Railway documentation or create an issue in the GitHub repository.*