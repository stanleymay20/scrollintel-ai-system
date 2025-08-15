# ScrollIntel Core

**Focused AI-CTO Replacement Platform with 7 Core Agents**

ScrollIntel Core is a streamlined version of the ScrollIntel platform, focused on delivering exceptional AI-CTO capabilities through 7 essential agents. This focused approach eliminates complexity while maximizing user value.

## 🎯 Core Mission

Build the world's best AI-CTO replacement platform that helps businesses make data-driven decisions without hiring a full technical team.

## 🤖 The 7 Core Agents

1. **CTO Agent** - Technology architecture decisions and scaling strategies
2. **Data Scientist Agent** - Automated data analysis and insights generation
3. **ML Engineer Agent** - Machine learning model building and deployment
4. **BI Agent** - Business intelligence dashboards and reporting
5. **AI Engineer Agent** - AI strategy and implementation guidance
6. **QA Agent** - Natural language data querying
7. **Forecast Agent** - Time series prediction and trend analysis

## 🚀 Quick Start

### Prerequisites

- Docker and Docker Compose
- At least one AI service API key (OpenAI or Anthropic)

### Installation

1. **Clone and navigate to ScrollIntel Core:**
   ```bash
   cd scrollintel_core
   ```

2. **Set up environment:**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

3. **Start the platform:**
   
   **Linux/Mac:**
   ```bash
   ./start.sh dev
   ```
   
   **Windows:**
   ```cmd
   start.bat dev
   ```

4. **Access the platform:**
   - API: http://localhost:8001
   - API Documentation: http://localhost:8001/docs
   - Health Check: http://localhost:8001/health

## 📁 Project Structure

```
scrollintel_core/
├── agents/                 # 7 Core AI agents
│   ├── base.py            # Base agent class
│   ├── orchestrator.py    # Agent routing
│   ├── cto_agent.py       # CTO Agent
│   ├── data_scientist_agent.py
│   ├── ml_engineer_agent.py
│   ├── bi_agent.py
│   ├── ai_engineer_agent.py
│   ├── qa_agent.py
│   └── forecast_agent.py
├── api/                   # API routes
│   └── routes.py
├── init-scripts/          # Database initialization
├── config.py             # Configuration management
├── database.py           # Database setup
├── models.py             # Database models
├── main.py               # FastAPI application
├── docker-compose.yml    # Docker services
├── Dockerfile            # Container definition
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## 🔧 Configuration

### Environment Variables

Key configuration options in `.env`:

```bash
# Application
DEBUG=false
SECRET_KEY=your-secret-key-here

# Database
DATABASE_URL=postgresql://scrollintel:password@localhost:5433/scrollintel_core

# Redis
REDIS_URL=redis://localhost:6380/0

# AI Services (at least one required)
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key

# File Processing
MAX_FILE_SIZE=104857600  # 100MB
```

### Docker Services

- **PostgreSQL** (port 5433) - Primary database
- **Redis** (port 6380) - Caching and sessions
- **ScrollIntel Core** (port 8001) - Main application

## 📊 Usage Examples

### 1. Upload and Analyze Data

```python
import requests

# Upload a CSV file
files = {'file': open('data.csv', 'rb')}
data = {
    'workspace_id': 'your-workspace-id',
    'user_id': 'your-user-id',
    'name': 'Sales Data',
    'description': 'Monthly sales data'
}

response = requests.post('http://localhost:8001/api/v1/files/upload', 
                        files=files, data=data)
dataset_id = response.json()['dataset_id']
```

### 2. Ask the Data Scientist Agent

```python
# Analyze the data
request = {
    'query': 'Analyze this sales data and find key insights',
    'dataset_id': dataset_id,
    'user_id': 'your-user-id'
}

response = requests.post('http://localhost:8001/api/v1/agents/process', 
                        json=request)
insights = response.json()['result']
```

### 3. Build ML Models

```python
# Create a predictive model
request = {
    'query': 'Build a model to predict sales for next month',
    'dataset_id': dataset_id,
    'user_id': 'your-user-id'
}

response = requests.post('http://localhost:8001/api/v1/agents/process', 
                        json=request)
model_info = response.json()['result']
```

### 4. Create Dashboards

```python
# Generate a BI dashboard
request = {
    'query': 'Create a sales performance dashboard',
    'dataset_id': dataset_id,
    'user_id': 'your-user-id'
}

response = requests.post('http://localhost:8001/api/v1/agents/process', 
                        json=request)
dashboard = response.json()['result']
```

## 🏗️ Architecture

### Agent Orchestrator

The orchestrator routes user requests to the appropriate agent based on intent classification:

- **Keywords-based routing** for simple, reliable agent selection
- **Health monitoring** for all agents
- **Request logging** for analytics
- **Error handling** with graceful fallbacks

### Database Schema

Simplified schema focused on core functionality:

- **Users & Workspaces** - Organization and access control
- **Datasets** - Uploaded data files with metadata
- **Analyses** - Agent processing results
- **Models** - ML models with performance metrics
- **Dashboards** - BI visualizations
- **Audit Logs** - Security and compliance

## 🔒 Security

- JWT-based authentication
- Role-based access control
- Data encryption at rest and in transit
- Audit logging for all operations
- File upload validation and limits

## 📈 Performance

- **Response Times:**
  - File processing: < 30 seconds
  - Agent queries: < 5 seconds
  - Dashboard loading: < 2 seconds

- **Scalability:**
  - 100+ concurrent users (MVP)
  - 100MB file size limit
  - Connection pooling and caching

## 🧪 Development

### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-asyncio httpx

# Run tests
pytest tests/
```

### Adding New Agents

1. Create agent class inheriting from `Agent`
2. Implement `process()` and `get_capabilities()` methods
3. Add to orchestrator initialization
4. Update intent classification keywords

### Database Migrations

```bash
# Generate migration
alembic revision --autogenerate -m "Description"

# Apply migration
alembic upgrade head
```

## 🚀 Deployment

### Production Deployment

1. **Set production environment:**
   ```bash
   export DEBUG=false
   export SECRET_KEY=your-production-secret
   ```

2. **Start in production mode:**
   ```bash
   ./start.sh prod
   ```

3. **Monitor health:**
   ```bash
   curl http://localhost:8001/health
   curl http://localhost:8001/api/v1/agents/health
   ```

### Cloud Deployment

The system is designed for easy deployment to:

- **Render** - Automatic deployment from Git
- **Railway** - Simple container deployment  
- **AWS ECS** - Scalable container orchestration
- **Google Cloud Run** - Serverless containers

## 📊 Monitoring

- Health checks for all services
- Request/response logging
- Performance metrics
- Agent usage statistics
- Error tracking and alerting

## 🤝 Contributing

1. Focus on the 7 core agents
2. Maintain simplicity and reliability
3. Write tests for new features
4. Update documentation

## 📄 License

Copyright © 2024 ScrollIntel. All rights reserved.

---

**ScrollIntel Core** - Focused AI-CTO replacement platform that delivers real business value through 7 essential agents.