# ScrollIntel Railway Deployment Guide

This guide will help you deploy ScrollIntel to Railway with optimized network health.

## 🚀 Quick Deploy

1. **Push to GitHub**:
   ```bash
   python deploy_to_github.py
   ```

2. **Deploy on Railway**:
   - Connect your GitHub repository to Railway
   - Railway will automatically detect the `railway.json` configuration
   - Set the required environment variables (see below)

## 📋 Environment Variables

Set these in your Railway project settings:

### Required Variables
```bash
DATABASE_URL=postgresql://...  # Railway provides this automatically
PORT=8000                      # Railway provides this automatically
RAILWAY_ENVIRONMENT=production
```

### Optional Variables (for full functionality)
```bash
# AI API Keys
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key

# Security
JWT_SECRET_KEY=your_jwt_secret
SESSION_SECRET_KEY=your_session_secret

# CORS (Railway domain will be added automatically)
CORS_ORIGINS=https://yourdomain.com
```

## 🏗️ Railway Configuration

The deployment uses these optimized files:

- `railway.json` - Railway deployment configuration
- `Dockerfile.railway` - Optimized Docker container
- `requirements.railway.txt` - Minimal dependencies for faster builds
- `start.py` - Railway-optimized startup script

## 🔍 Health Monitoring

Railway will monitor your app using:
- **Health Check Path**: `/health`
- **Health Check Timeout**: 300 seconds
- **Restart Policy**: On failure (max 3 retries)

## 🐛 Troubleshooting

### Network Health Errors

If you're getting network health errors:

1. **Check the logs**:
   ```bash
   railway logs
   ```

2. **Verify health endpoint**:
   ```bash
   curl https://your-app.railway.app/health
   ```

3. **Common issues**:
   - Database connection timeout → Check DATABASE_URL
   - Port binding issues → Railway sets PORT automatically
   - Memory/CPU limits → Check Railway resource usage

### Database Issues

1. **PostgreSQL connection**:
   - Railway provides DATABASE_URL automatically
   - The app handles URL format conversion

2. **Migration errors**:
   - Database tables are created automatically on startup
   - Check logs for migration errors

### Performance Issues

1. **Slow startup**:
   - Using minimal requirements in `requirements.railway.txt`
   - Health check timeout set to 300 seconds

2. **Memory usage**:
   - Optimized Docker image with minimal dependencies
   - Health endpoint monitors memory usage

## 📊 Monitoring

Access these endpoints to monitor your deployment:

- `/` - Basic system info
- `/health` - Detailed health check with Railway metadata
- `/docs` - API documentation (disabled in production)

## 🔧 Advanced Configuration

### Custom Domain

1. Add your domain in Railway dashboard
2. Update CORS_ORIGINS environment variable
3. Update APP_URL environment variable

### Scaling

Railway automatically handles:
- Horizontal scaling based on traffic
- Resource allocation
- Load balancing

### Database Scaling

For production workloads:
1. Use Railway's PostgreSQL addon
2. Enable connection pooling
3. Monitor query performance

## 📝 Deployment Checklist

- [ ] Code pushed to GitHub
- [ ] Railway project connected to GitHub repo
- [ ] Environment variables configured
- [ ] Database addon added (if needed)
- [ ] Custom domain configured (if needed)
- [ ] Health check passing
- [ ] API endpoints responding
- [ ] Logs showing no errors

## 🆘 Support

If you encounter issues:

1. Check Railway logs: `railway logs`
2. Verify environment variables
3. Test health endpoint
4. Check database connectivity
5. Review Railway resource usage

## 🎯 Performance Optimization

The Railway deployment is optimized for:

- **Fast startup**: Minimal dependencies
- **Low memory usage**: Efficient Docker image
- **Quick health checks**: Lightweight monitoring
- **Automatic scaling**: Railway handles traffic spikes
- **Database performance**: Connection pooling and optimization

## 🔄 Continuous Deployment

Railway automatically redeploys when you push to your main branch:

1. Make changes locally
2. Commit and push to GitHub
3. Railway automatically builds and deploys
4. Monitor deployment in Railway dashboard

Your ScrollIntel app will be available at: `https://your-app.railway.app`