# ScrollIntel Full Stack Demo

This demo launches both the backend API and frontend development server for ScrollIntel.

## Quick Start

### Windows
```bash
# Double-click or run:
start_demo.bat

# Or run directly:
python start_full_demo.py
```

### Mac/Linux
```bash
# Make executable and run:
chmod +x start_demo.sh
./start_demo.sh

# Or run directly:
python3 start_full_demo.py
```

## What Gets Started

1. **Backend API** (Port 8000)
   - FastAPI server with simple routes
   - Health check endpoint
   - Agent chat functionality
   - System metrics API

2. **Frontend** (Port 3000)
   - Next.js development server
   - React-based UI components
   - Real-time dashboard
   - Chat interface

## Access Points

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## Features Available

- 🤖 **AI Agent Chat**: Interact with different AI agents
- 📊 **System Dashboard**: View real-time metrics
- 🔧 **API Testing**: Use the interactive API docs
- 📈 **Monitoring**: Basic system monitoring

## Requirements

- Python 3.8+
- Node.js 16+
- npm or yarn

## Troubleshooting

### Port Issues
If ports 8000 or 3000 are busy, the launcher will automatically find available ports.

### Missing Dependencies
The launcher will attempt to install missing Python packages automatically.

For frontend dependencies:
```bash
cd frontend
npm install
```

### Backend Only
To run just the backend:
```bash
python run_simple.py
```

### Frontend Only
To run just the frontend:
```bash
cd frontend
npm run dev
```

## Stopping the Demo

Press `Ctrl+C` in the terminal to stop both servers gracefully.

## Next Steps

1. Explore the chat interface
2. Check out the dashboard
3. Try the API endpoints at `/docs`
4. Upload data files for processing
5. Experiment with different AI agents

---

**ScrollIntel™** - AI-Powered CTO Platform