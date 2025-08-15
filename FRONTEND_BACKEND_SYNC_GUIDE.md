# ScrollIntel Frontend-Backend Sync Guide

This guide will help you check and run the ScrollIntel UI where the frontend and backend work in perfect synchronization.

## System Architecture Overview

ScrollIntel is a full-stack AI system with:
- **Frontend**: Next.js 14 with TypeScript, Tailwind CSS, and Radix UI components
- **Backend**: FastAPI with Python, PostgreSQL, Redis, and AI integrations
- **Orchestration**: Docker Compose for seamless service coordination

## Quick Start - Run the Complete System

### Option 1: Docker Compose (Recommended)

1. **Set up environment variables**:
```bash
cp .env.example .env
# Edit .env with your API keys and configuration
```

2. **Start all services**:
```bash
docker-compose up -d
```

3. **Access the application**:
- Frontend UI: http://localhost:3000
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs

### Option 2: Development Mode (Separate Terminals)

1. **Terminal 1 - Backend**:
```bash
# Install Python dependencies
pip install -r requirements.txt

# Start the FastAPI server
uvicorn scrollintel.api.gateway:app --reload --host 0.0.0.0 --port 8000
```

2. **Terminal 2 - Frontend**:
```bash
cd frontend
npm install
npm run dev
```

3. **Terminal 3 - Database (if not using Docker)**:
```bash
# Start PostgreSQL and Redis locally
# Or use Docker for just the databases:
docker-compose up postgres redis -d
```

## Frontend-Backend Integration Points

### 1. API Communication Layer

The frontend communicates with the backend through:
- **Base URL**: Configured in `frontend/src/lib/api.ts`
- **Axios Client**: Handles authentication, error handling, and request/response interceptors
- **Environment Variable**: `NEXT_PUBLIC_API_URL` (defaults to http://localhost:8000)

### 2. Real-time Features

- **WebSocket Connection**: For real-time agent status updates
- **Chat Interface**: Live communication with AI agents
- **System Metrics**: Real-time dashboard updates
- **File Upload Progress**: Live progress tracking

### 3. Key Integration Components

#### Chat Interface (`frontend/src/components/chat/chat-interface.tsx`)
- Connects to `/agents/execute` endpoint
- Real-time message streaming
- Agent selection and routing

#### Agent Status Cards (`frontend/src/components/dashboard/agent-status-card.tsx`)
- Displays live agent health from `/agents/` endpoint
- Shows capabilities and performance metrics
- Interactive agent selection

#### File Upload (`frontend/src/components/upload/file-upload.tsx`)
- Connects to `/files/upload` endpoint
- Progress tracking and error handling
- Multi-file support with drag-and-drop

## Health Check Endpoints

Monitor system health through these endpoints:

### Backend Health Checks
```bash
# Basic health
curl http://localhost:8000/health

# Detailed system health
curl http://localhost:8000/health/detailed

# Agent-specific health
curl http://localhost:8000/health/agents

# System metrics
curl http://localhost:8000/health/metrics
```

### Frontend Health Check
```bash
# Frontend availability
curl http://localhost:3000

# Next.js API health
curl http://localhost:3000/api/health
```

## Testing the Integration

### 1. Basic Connectivity Test

1. Open http://localhost:3000 in your browser
2. Check that the dashboard loads with agent cards
3. Verify system metrics are displaying
4. Test the chat interface by sending a message

### 2. Agent Interaction Test

1. Click on an agent card (e.g., ScrollCTO)
2. Send a test message in the chat interface
3. Verify the response appears with proper formatting
4. Check that agent status updates in real-time

### 3. File Upload Test

1. Use the file upload component
2. Drag and drop a file or click to select
3. Monitor upload progress
4. Verify file appears in the uploaded files list

### 4. API Integration Test

Open browser developer tools and check:
- Network tab shows successful API calls
- WebSocket connections are established
- No CORS errors in console
- Authentication headers are properly set

## Configuration Files

### Frontend Configuration
- `frontend/next.config.js`: Next.js configuration with API proxy
- `frontend/src/lib/api.ts`: Axios configuration and API client
- `frontend/tailwind.config.js`: UI styling configuration

### Backend Configuration
- `scrollintel/api/gateway.py`: Main FastAPI application
- `scrollintel/core/config.py`: Environment and system configuration
- `docker-compose.yml`: Service orchestration

### Environment Variables
Key variables for frontend-backend sync:
```bash
# Frontend
NEXT_PUBLIC_API_URL=http://localhost:8000

# Backend
API_HOST=0.0.0.0
API_PORT=8000
POSTGRES_HOST=localhost
REDIS_HOST=localhost
```

## Troubleshooting Common Issues

### 1. CORS Errors
- Check `CORS_ORIGINS` in backend configuration
- Verify `NEXT_PUBLIC_API_URL` matches backend URL
- Ensure middleware is properly configured

### 2. API Connection Failed
```bash
# Check backend is running
curl http://localhost:8000/health

# Check frontend can reach backend
curl http://localhost:3000/api/health
```

### 3. Database Connection Issues
```bash
# Check database status
docker-compose ps postgres

# Check database logs
docker-compose logs postgres
```

### 4. Agent Not Responding
```bash
# Check agent registry status
curl http://localhost:8000/agents/registry/status

# Check individual agent health
curl http://localhost:8000/agents/{agent_id}/health
```

## Development Workflow

### 1. Making Changes

**Frontend Changes**:
- Edit files in `frontend/src/`
- Hot reload automatically updates the browser
- Check browser console for errors

**Backend Changes**:
- Edit files in `scrollintel/`
- FastAPI auto-reloads with `--reload` flag
- Check terminal output for errors

### 2. Adding New Features

1. **API Endpoint**: Add route in `scrollintel/api/routes/`
2. **Frontend Integration**: Update `frontend/src/lib/api.ts`
3. **UI Component**: Create component in `frontend/src/components/`
4. **Integration**: Connect component to API in page/layout

### 3. Testing Integration

```bash
# Run backend tests
pytest tests/

# Run frontend tests
cd frontend && npm test

# Run integration tests
docker-compose -f docker-compose.test.yml up --abort-on-container-exit
```

## Production Deployment

### Using Docker Compose Production
```bash
# Build and deploy production stack
docker-compose -f docker-compose.prod.yml up -d

# With NGINX reverse proxy
docker-compose --profile production up -d
```

### Environment-Specific Configurations
- Development: `docker-compose.yml`
- Testing: `docker-compose.test.yml`
- Production: `docker-compose.prod.yml`

## Monitoring and Observability

### Real-time Monitoring
- System metrics dashboard at http://localhost:3000
- Agent status monitoring
- Performance metrics and response times
- Error tracking and logging

### Log Files
- Backend logs: `logs/` directory
- Frontend logs: Browser developer console
- Container logs: `docker-compose logs [service]`

## Security Considerations

- JWT authentication between frontend and backend
- CORS properly configured for allowed origins
- API rate limiting and security middleware
- Secure headers and content security policy

---

This guide provides a comprehensive overview of how to check and run the ScrollIntel UI with full frontend-backend synchronization. The system is designed for seamless integration with real-time updates, robust error handling, and production-ready deployment options.