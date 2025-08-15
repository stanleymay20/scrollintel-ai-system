# ScrollIntel Quick Start Guide

## Best Option: Docker Compose ✅

After analyzing your system, **Docker Compose is the optimal choice** because:

- ✅ **Complete Integration**: Orchestrates all services automatically
- ✅ **Zero Configuration**: No dependency management needed
- ✅ **Production-Ready**: Mirrors production environment
- ✅ **Built-in Health Checks**: Ensures proper service connectivity
- ✅ **Easy Management**: Single command to start/stop everything

## Quick Start (3 Steps)

### Step 1: Configure Environment
```bash
# Copy environment template
copy .env.example .env

# Edit .env with your API keys (minimum required):
# OPENAI_API_KEY=your_openai_key_here
# JWT_SECRET_KEY=your_jwt_secret_here
```

### Step 2: Start the System
```bash
# Option A: Use the startup script (recommended)
start-scrollintel.bat

# Option B: Manual Docker Compose
docker-compose up -d
```

### Step 3: Access Your System
- **Frontend UI**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

## System Architecture

Your ScrollIntel system includes:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │    Backend      │    │   Database      │
│   Next.js 14   │◄──►│   FastAPI       │◄──►│   PostgreSQL    │
│   Port: 3000    │    │   Port: 8000    │    │   Port: 5432    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │     Redis       │
                    │   Port: 6379    │
                    └─────────────────┘
```

## Key Features Working in Sync

### 1. Real-time Chat Interface
- Frontend: `frontend/src/components/chat/chat-interface.tsx`
- Backend: `scrollintel/api/routes/agent_routes.py`
- **Sync Point**: WebSocket connections for live agent communication

### 2. Agent Status Dashboard
- Frontend: `frontend/src/components/dashboard/agent-status-card.tsx`
- Backend: `scrollintel/api/routes/health_routes.py`
- **Sync Point**: Real-time health monitoring and status updates

### 3. File Upload System
- Frontend: `frontend/src/components/upload/file-upload.tsx`
- Backend: `scrollintel/api/routes/file_routes.py`
- **Sync Point**: Progress tracking and file processing

### 4. System Metrics
- Frontend: `frontend/src/components/dashboard/system-metrics.tsx`
- Backend: `scrollintel/core/monitoring.py`
- **Sync Point**: Live performance data streaming

## Health Check Commands

```bash
# Check all services
docker-compose ps

# View logs
docker-compose logs -f

# Check specific service
docker-compose logs backend
docker-compose logs frontend

# Restart services
docker-compose restart

# Stop everything
docker-compose down
```

## API Integration Points

The frontend communicates with backend through:

```typescript
// frontend/src/lib/api.ts
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

// Key endpoints:
// GET  /health          - System health
// GET  /agents          - List all agents
// POST /agents/execute  - Execute agent tasks
// POST /files/upload    - File upload
// GET  /health/detailed - Comprehensive health check
```

## Testing the Integration

1. **Basic Connectivity**:
   ```bash
   curl http://localhost:8000/health
   curl http://localhost:3000
   ```

2. **Agent Communication**:
   - Open http://localhost:3000
   - Click on an agent card
   - Send a test message
   - Verify response appears

3. **File Upload**:
   - Drag and drop a file
   - Watch progress bar
   - Verify file appears in list

4. **Real-time Updates**:
   - Check system metrics update
   - Verify agent status changes
   - Monitor WebSocket connections in browser dev tools

## Troubleshooting

### Common Issues:

1. **Port Conflicts**:
   ```bash
   # Check what's using ports
   netstat -an | findstr :3000
   netstat -an | findstr :8000
   ```

2. **Docker Issues**:
   ```bash
   # Rebuild containers
   docker-compose down
   docker-compose build --no-cache
   docker-compose up -d
   ```

3. **Environment Variables**:
   - Ensure `.env` file exists
   - Check API keys are set
   - Verify database credentials

### Health Check Endpoints:

- **Basic**: http://localhost:8000/health
- **Detailed**: http://localhost:8000/health/detailed
- **Agents**: http://localhost:8000/health/agents
- **Metrics**: http://localhost:8000/health/metrics

## Why Docker Compose Wins

| Feature | Docker Compose | Manual Setup |
|---------|----------------|--------------|
| Setup Time | 2 minutes | 15+ minutes |
| Dependencies | Automatic | Manual install |
| Service Sync | Built-in | Manual config |
| Health Checks | Included | Manual setup |
| Production Parity | ✅ Perfect | ❌ Different |
| Cleanup | One command | Multiple steps |

## Next Steps

Once running, explore:
- **Agent Dashboard**: Interact with AI agents
- **File Processing**: Upload and process documents
- **System Monitoring**: View real-time metrics
- **API Documentation**: http://localhost:8000/docs

Your ScrollIntel system is now running with perfect frontend-backend synchronization! 🚀