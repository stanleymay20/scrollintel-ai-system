# 🚀 ScrollIntel Demo Guide

Welcome to ScrollIntel™ - the AI-Powered CTO Platform! This guide will help you get the demo up and running quickly.

## 🎯 What is ScrollIntel?

ScrollIntel is an advanced AI platform that provides:
- **AI Agents**: Specialized agents for different roles (CTO, Data Scientist, ML Engineer, etc.)
- **Real-time Analytics**: System monitoring and performance metrics
- **Chat Interface**: Natural language interaction with AI agents
- **Dashboard**: Visual insights and system status
- **API-First**: RESTful API for integration with other systems

## 🚀 Quick Start Options

### Option 1: Full Stack Demo (Recommended)
Starts both backend API and frontend React app:

```bash
# Windows
start_demo.bat

# Mac/Linux
./start_demo.sh

# Or directly
python start_full_demo.py
```

**Access Points:**
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

### Option 2: Backend Only Demo
Perfect when Node.js isn't available:

```bash
python start_backend_demo.py
```

**Access Points:**
- Backend API: http://localhost:8000 (or next available port)
- API Docs: http://localhost:8000/docs
- Simple Web UI: Open `simple_frontend.html` in your browser

### Option 3: Simple API Server
Minimal setup for testing:

```bash
python run_simple.py
```

## 🧪 Testing the Demo

### Automated Testing
Run our test suite to verify everything is working:

```bash
python test_demo_api.py
```

This will:
- Find the running API server
- Test all endpoints
- Verify chat functionality
- Provide a summary report

### Manual Testing

1. **Health Check**
   ```bash
   curl http://localhost:8000/health
   ```

2. **Get Available Agents**
   ```bash
   curl http://localhost:8000/api/agents
   ```

3. **System Metrics**
   ```bash
   curl http://localhost:8000/api/monitoring/metrics
   ```

4. **Chat with Agent**
   ```bash
   curl -X POST http://localhost:8000/api/agents/chat \
     -H "Content-Type: application/json" \
     -d '{"message": "Hello, can you help me?", "agent_id": "cto-agent"}'
   ```

## 🎨 Demo Features

### 1. AI Agents
- **CTO Agent**: Strategic technology leadership
- **Data Scientist**: Advanced analytics and ML
- **ML Engineer**: Model development and deployment
- **AI Engineer**: AI system architecture
- **BI Agent**: Business intelligence and reporting
- **QA Agent**: Quality assurance and testing

### 2. Interactive Chat
- Natural language conversations
- Context-aware responses
- Agent specialization
- Real-time responses

### 3. System Monitoring
- CPU, Memory, Disk usage
- Active connections
- Response times
- Uptime statistics

### 4. Dashboard Analytics
- Active users
- Request statistics
- Success rates
- System status

## 🌐 Web Interface Features

### Simple Frontend (simple_frontend.html)
- **Real-time Metrics**: Live system monitoring
- **Agent Directory**: Browse available AI agents
- **Chat Interface**: Interactive conversations
- **Dashboard Data**: Key performance indicators
- **Auto-Discovery**: Automatically finds the API server

### Full Frontend (React/Next.js)
- **Modern UI**: Professional React-based interface
- **Advanced Components**: Rich interactive elements
- **Real-time Updates**: WebSocket connections
- **Responsive Design**: Works on all devices
- **Advanced Features**: File upload, data visualization

## 🔧 Configuration

### Environment Variables
The demo uses these key settings:

```env
# Database
DATABASE_URL=sqlite:///./scrollintel.db

# API Settings
API_PORT=8000
DEBUG=true

# Optional AI API Keys
OPENAI_API_KEY=your-key-here
ANTHROPIC_API_KEY=your-key-here
```

### Port Configuration
The launchers automatically find available ports:
- Backend: 8000, 8001, 8002, 8003, 8080
- Frontend: 3000, 3001, 3002, 3003

## 🐛 Troubleshooting

### Common Issues

1. **Port Already in Use**
   - The launchers automatically find available ports
   - Check what's running: `netstat -an | findstr :8000`

2. **Python Dependencies Missing**
   ```bash
   pip install -r requirements.txt
   ```

3. **Node.js Not Found**
   - Install from https://nodejs.org/
   - Or use backend-only demo

4. **API Not Responding**
   ```bash
   python test_demo_api.py
   ```

### Debug Mode
For detailed logging, set environment variable:
```bash
export DEBUG=true
python start_backend_demo.py
```

## 📚 API Documentation

### Interactive Docs
Visit http://localhost:8000/docs for:
- Complete API reference
- Interactive testing
- Request/response examples
- Authentication details

### Key Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/api/agents` | GET | List available agents |
| `/api/agents/chat` | POST | Chat with agents |
| `/api/monitoring/metrics` | GET | System metrics |
| `/api/dashboard` | GET | Dashboard data |

## 🎯 Demo Scenarios

### Scenario 1: Business Executive Demo
1. Open the web interface
2. Check system metrics and dashboard
3. Chat with the CTO Agent about strategic planning
4. Show real-time monitoring capabilities

### Scenario 2: Technical Demo
1. Open API documentation at `/docs`
2. Test endpoints interactively
3. Show agent specialization
4. Demonstrate system monitoring

### Scenario 3: Developer Demo
1. Show the simple HTML frontend
2. Explain API-first architecture
3. Test different agents
4. Show error handling and fallbacks

## 🚀 Next Steps

After the demo, users can:

1. **Explore More Features**
   - Try different AI agents
   - Upload data files
   - Create custom dashboards

2. **Integration**
   - Use the REST API
   - Build custom frontends
   - Connect to existing systems

3. **Deployment**
   - Deploy to cloud platforms
   - Set up production databases
   - Configure monitoring

## 📞 Support

For issues or questions:
- Check the troubleshooting section
- Run the test suite
- Review API documentation
- Check system logs

---

**ScrollIntel™** - Where AI meets unlimited potential!

*Happy demoing! 🎉*