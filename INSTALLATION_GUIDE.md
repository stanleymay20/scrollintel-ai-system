# ScrollIntel™ Installation Guide

## 🎯 Prerequisites

### Required Software
- **Docker Desktop** (includes Docker Compose)
  - Windows: https://docs.docker.com/desktop/windows/
  - Mac: https://docs.docker.com/desktop/mac/
  - Linux: https://docs.docker.com/desktop/linux/

### Required API Keys
- **OpenAI API Key** (required for AI features)
  - Get from: https://platform.openai.com/api-keys
  - Format: `sk-...`

### Optional API Keys
- **Anthropic API Key** (for Claude integration)
- **Pinecone API Key** (for vector database)
- **Supabase credentials** (for additional storage)

## 🚀 Installation Methods

### Method 1: One-Click Launch (Easiest)

#### Windows
1. Download or clone ScrollIntel
2. Double-click `launch-scrollintel.bat`
3. Follow the prompts
4. Open http://localhost:3000

#### Linux/Mac
1. Download or clone ScrollIntel
2. Run `./launch-scrollintel.sh`
3. Follow the prompts
4. Open http://localhost:3000

### Method 2: Quick Start Script

```bash
# Make executable (Linux/Mac only)
chmod +x quick-start.sh

# Run quick start
./quick-start.sh
```

### Method 3: Manual Docker Setup

```bash
# 1. Setup environment
python scripts/setup-environment.py

# 2. Add your OpenAI API key to .env
# Edit .env file and add: OPENAI_API_KEY=sk-your-key-here

# 3. Start services
docker-compose up -d

# 4. Check health
python scripts/health-check.py
```

### Method 4: Development Setup

```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Setup database
python init_database.py

# 3. Start backend
uvicorn scrollintel.api.main:app --reload

# 4. Start frontend (new terminal)
cd frontend
npm install
npm run dev
```

## 🔧 Configuration

### Environment Variables

The installation scripts automatically configure:
- `JWT_SECRET_KEY` - Secure authentication token
- `POSTGRES_PASSWORD` - Database password
- `ENVIRONMENT` - Development/production mode
- `DEBUG` - Debug logging level

### Manual Configuration

Edit `.env` file to customize:

```bash
# === Core Settings ===
ENVIRONMENT=development
DEBUG=true
API_PORT=8000

# === Database ===
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=scrollintel
POSTGRES_USER=postgres
POSTGRES_PASSWORD=auto-generated

# === AI Services ===
OPENAI_API_KEY=sk-your-key-here
ANTHROPIC_API_KEY=your-anthropic-key

# === Security ===
JWT_SECRET_KEY=auto-generated
JWT_ALGORITHM=HS256
JWT_EXPIRATION_HOURS=24
```

## 🏥 Health Checks

### Automated Health Check
```bash
python scripts/health-check.py
```

### Manual Verification
1. **Frontend**: http://localhost:3000
2. **API**: http://localhost:8000
3. **API Docs**: http://localhost:8000/docs
4. **Health Endpoint**: http://localhost:8000/health

### Docker Service Status
```bash
docker-compose ps
```

## 🐛 Troubleshooting

### Common Issues

#### Port Already in Use
```bash
# Check what's using the port
netstat -tulpn | grep :3000
netstat -tulpn | grep :8000

# Kill the process or change ports in docker-compose.yml
```

#### Docker Not Running
```bash
# Start Docker Desktop
# Or on Linux:
sudo systemctl start docker
```

#### Permission Denied (Linux/Mac)
```bash
# Make scripts executable
chmod +x launch-scrollintel.sh
chmod +x quick-start.sh

# Or run with bash
bash launch-scrollintel.sh
```

#### API Key Not Working
1. Verify API key format: `sk-...`
2. Check API key permissions
3. Ensure no extra spaces in .env file
4. Restart services: `docker-compose restart`

#### Database Connection Failed
```bash
# Reset database
docker-compose down -v
docker-compose up -d

# Check database logs
docker-compose logs postgres
```

### Getting Help

#### View Logs
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f backend
docker-compose logs -f frontend
docker-compose logs -f postgres
```

#### Reset Everything
```bash
# Complete reset (removes all data)
docker-compose down -v
docker-compose up -d --build
```

#### Service Management
```bash
# Restart services
docker-compose restart

# Stop services
docker-compose down

# Update services
docker-compose pull
docker-compose up -d --build
```

## 🏭 Production Installation

### Production Setup
```bash
# Run production setup script
python scripts/production-setup.py

# Deploy with production configuration
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

### Production Checklist
- [ ] SSL certificates configured
- [ ] Domain name configured
- [ ] Firewall rules set
- [ ] Backup system enabled
- [ ] Monitoring configured
- [ ] Strong passwords set
- [ ] API keys secured

## 📊 Monitoring Setup

### Built-in Monitoring
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3001 (admin/admin123)

### Custom Monitoring
```bash
# Setup monitoring
python scripts/production-setup.py

# View metrics
curl http://localhost:8000/metrics
```

## 🔒 Security Considerations

### Development
- Default passwords are auto-generated
- JWT secrets are cryptographically secure
- Debug mode enabled for development

### Production
- Use strong, unique passwords
- Enable SSL/TLS certificates
- Disable debug mode
- Configure firewall rules
- Regular security updates

## 📈 Performance Optimization

### Resource Requirements
- **Minimum**: 4GB RAM, 2 CPU cores
- **Recommended**: 8GB RAM, 4 CPU cores
- **Storage**: 20GB+ available space

### Docker Resource Limits
Edit `docker-compose.prod.yml` to adjust:
```yaml
deploy:
  resources:
    limits:
      memory: 2G
    reservations:
      memory: 1G
```

## 🎉 Success Verification

After installation, you should see:
- ✅ All Docker containers running
- ✅ Database initialized and connected
- ✅ API responding at http://localhost:8000
- ✅ Frontend accessible at http://localhost:3000
- ✅ Health checks passing
- ✅ AI agents operational

## 📞 Support

If you encounter issues:
1. Check this troubleshooting guide
2. Review the logs: `docker-compose logs -f`
3. Run health check: `python scripts/health-check.py`
4. Reset if needed: `docker-compose down -v && docker-compose up -d`

---

**ScrollIntel™** - Your AI-Powered CTO Platform 🌟