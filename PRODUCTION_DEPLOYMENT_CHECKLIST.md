# ScrollIntel Production Deployment Checklist

## 🚀 Pre-Deployment Checklist

### ✅ Environment Configuration
- [ ] Set real OpenAI API key: `OPENAI_API_KEY=sk-your-real-key`
- [ ] Set secure JWT secret: `JWT_SECRET_KEY=your-secure-64-char-secret`
- [ ] Set production database URL: `DATABASE_URL=postgresql://...`
- [ ] Set production Redis URL: `REDIS_URL=redis://...` (optional)
- [ ] Set email configuration: `SMTP_SERVER`, `EMAIL_PASSWORD`
- [ ] Set base URL: `BASE_URL=https://yourdomain.com`

### ✅ Security Configuration
- [ ] Generate secure JWT secret: `openssl rand -base64 64`
- [ ] Use strong database passwords
- [ ] Enable HTTPS in production
- [ ] Set CORS origins: `CORS_ORIGINS=https://yourdomain.com`
- [ ] Configure rate limiting

### ✅ Infrastructure
- [ ] Database server ready (PostgreSQL 12+)
- [ ] Redis server ready (optional but recommended)
- [ ] SMTP server configured
- [ ] SSL certificates installed
- [ ] Load balancer configured (if needed)

### ✅ Validation
- [ ] Run: `python scripts/validate-environment.py`
- [ ] Test database connection
- [ ] Test API endpoints
- [ ] Test frontend build
- [ ] Verify all services start correctly

## 🚀 Deployment Commands

### Docker Deployment
```bash
# Set environment variables
export OPENAI_API_KEY="sk-your-real-key"
export JWT_SECRET_KEY="$(openssl rand -base64 64)"
export DATABASE_URL="postgresql://user:pass@host:5432/scrollintel"

# Deploy
./scripts/deploy-docker.sh
```

### Railway Deployment
```bash
railway login
railway variables set OPENAI_API_KEY="sk-your-real-key"
railway variables set JWT_SECRET_KEY="$(openssl rand -base64 64)"
railway up
```

### Render Deployment
```bash
# Set environment variables in Render dashboard
# Deploy using render.yaml configuration
```

## 🔍 Post-Deployment Verification

- [ ] Health check: `curl https://yourdomain.com/health`
- [ ] API test: `curl https://yourdomain.com/api/agents`
- [ ] Frontend loads: Visit `https://yourdomain.com`
- [ ] Database connection works
- [ ] AI features work (test with real API key)
- [ ] Email notifications work
- [ ] File uploads work
- [ ] Authentication works

## 🚨 Troubleshooting

### Common Issues
1. **API Key Issues**: Ensure OPENAI_API_KEY starts with 'sk-'
2. **Database Connection**: Check DATABASE_URL format
3. **CORS Issues**: Verify CORS_ORIGINS includes your domain
4. **SSL Issues**: Ensure certificates are valid
5. **Memory Issues**: Check container/server memory limits

### Debug Commands
```bash
# Check environment variables
python scripts/validate-environment.py

# Test database connection
python -c "from scrollintel.models.database_utils import DatabaseManager; print('DB OK' if DatabaseManager().test_connection() else 'DB FAIL')"

# Check API health
curl https://yourdomain.com/health

# View logs
docker-compose logs -f  # For Docker deployment
```

## 📞 Support

If you encounter issues:
1. Check this checklist
2. Run validation scripts
3. Review error logs
4. Verify environment variables
5. Test individual components

**ScrollIntel is production-ready when all checklist items are complete! ✅**
