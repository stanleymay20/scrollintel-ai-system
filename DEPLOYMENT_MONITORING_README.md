# ScrollIntel Deployment Monitoring System

A comprehensive monitoring and health checking system for the ScrollIntel AI Platform.

## 🚀 Features

### Enhanced Deployment Checker
- **Multi-service monitoring**: Backend, Frontend, APIs, Database
- **Concurrent health checks**: Fast parallel service verification
- **Docker container monitoring**: Automatic container status detection
- **System resource monitoring**: CPU, Memory, Disk usage
- **File system integrity**: Critical file and directory verification
- **Detailed reporting**: Response times, error details, success rates

### Continuous Monitoring Dashboard
- **Real-time monitoring**: Live status updates
- **Historical trending**: Visual trend indicators
- **Configurable intervals**: Customizable check frequency
- **Quick status overview**: At-a-glance system health

## 📦 Installation

### Quick Setup
```bash
# Install and configure the monitoring system
python setup_deployment_checker.py
```

### Manual Setup
```bash
# Install dependencies
pip install -r deployment_checker_requirements.txt

# Make executable (Linux/Mac)
chmod +x check_deployment_status.py
chmod +x continuous_monitoring.py
```

## 🔧 Usage

### One-time Status Check
```bash
# Run comprehensive deployment check
python check_deployment_status.py

# Or if made executable
./check_deployment_status.py
```

### Continuous Monitoring
```bash
# Start continuous monitoring (30s intervals)
python continuous_monitoring.py

# Custom interval (60s)
python continuous_monitoring.py --interval 60

# Short interval for development (10s)
python continuous_monitoring.py -i 10
```

## 📊 What Gets Monitored

### Core Services
- ✅ Backend Health API (`/health`)
- ✅ API Documentation (`/docs`)
- ✅ Frontend Application
- ✅ Agent APIs
- ✅ Chat Interface

### API Endpoints
- 🌐 Chat API
- 📁 File Upload API
- 📊 Monitoring API
- 📈 Analytics API
- 🤖 Agent Management

### Infrastructure
- 💾 Database connectivity (SQLite/PostgreSQL)
- 🐳 Docker container status
- 📁 Critical file system paths
- 💻 System resources (CPU, Memory, Disk)

### Health Metrics
- ⏱️ Response times
- 📊 Success rates
- 🔄 Service availability
- 📈 Historical trends

## 🎯 Status Indicators

| Icon | Status | Description |
|------|--------|-------------|
| ✅ | HEALTHY | Service is fully operational |
| ⚠️ | DEGRADED | Service responding but with issues |
| ❌ | DOWN | Service is not responding |
| ⏰ | TIMEOUT | Service response timeout |
| 🟢 | GOOD | System health is optimal |
| 🟡 | WARNING | Some services need attention |
| 🔴 | CRITICAL | Multiple services are down |

## ⚙️ Configuration

### Custom Configuration
Edit `deployment_config.json` to customize:
- Service endpoints and timeouts
- Critical file paths
- System resource thresholds
- Monitoring preferences

```json
{
  "core_services": [
    {
      "url": "http://localhost:8000/health",
      "name": "Backend Health API",
      "timeout": 10,
      "expected_status": 200
    }
  ],
  "system_thresholds": {
    "cpu_warning": 80,
    "memory_warning": 85,
    "disk_warning": 90
  }
}
```

## 🔍 Troubleshooting

### Common Issues

#### Services Not Responding
```bash
# Check if services are running
docker-compose ps

# View service logs
docker-compose logs -f

# Restart services
docker-compose restart
```

#### Database Connection Issues
```bash
# Check database file exists
ls -la *.db

# Test database connection
python -c "import sqlite3; sqlite3.connect('scrollintel.db').execute('SELECT 1')"
```

#### High Resource Usage
```bash
# Check system resources
python -c "import psutil; print(f'CPU: {psutil.cpu_percent()}%, Memory: {psutil.virtual_memory().percent}%')"

# Monitor processes
top -p $(pgrep -f scrollintel)
```

### Exit Codes
- `0`: All services healthy
- `1`: Some services degraded or down

## 📈 Advanced Usage

### Integration with CI/CD
```bash
# Use in deployment scripts
if python check_deployment_status.py; then
    echo "Deployment successful"
else
    echo "Deployment failed - rolling back"
    exit 1
fi
```

### Monitoring Alerts
```bash
# Run check and send alerts on failure
python check_deployment_status.py || curl -X POST "https://hooks.slack.com/..." -d '{"text":"ScrollIntel deployment issue detected"}'
```

### Custom Health Checks
Extend the `ScrollIntelDeploymentChecker` class to add custom checks:

```python
def check_custom_service(self):
    # Your custom health check logic
    return ServiceStatus("Custom Service", "url", "✅ HEALTHY", 0)
```

## 🛠️ Development

### Running Tests
```bash
# Test the monitoring system
python -m pytest test_deployment_monitoring.py

# Test with mock services
python test_monitoring_mocks.py
```

### Adding New Checks
1. Extend `ScrollIntelDeploymentChecker` class
2. Add new check method
3. Integrate into `run_comprehensive_check()`
4. Update configuration if needed

## 📝 Logging

Monitoring results are displayed in real-time. For persistent logging:

```bash
# Log to file
python check_deployment_status.py > deployment_status.log 2>&1

# Continuous logging
python continuous_monitoring.py 2>&1 | tee monitoring.log
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add your monitoring enhancements
4. Test thoroughly
5. Submit a pull request

## 📄 License

This monitoring system is part of the ScrollIntel AI Platform.

---

**Need Help?** 
- Check the troubleshooting section above
- Review service logs: `docker-compose logs -f`
- Verify configuration in `deployment_config.json`
- Run setup again: `python setup_deployment_checker.py`