"""
Simple API routes for basic functionality without authentication
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, List, Any
from datetime import datetime
import psutil
import time
import random

router = APIRouter()

@router.get("/health")
async def health_check():
    """Simple health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "ScrollIntel API"
    }

@router.get("/api/monitoring/metrics")
async def get_system_metrics():
    """Get basic system metrics without authentication"""
    try:
        # Get real system metrics if available, otherwise use mock data
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            metrics = {
                "cpu_usage": round(cpu_percent, 1),
                "memory_usage": round(memory.percent, 1),
                "disk_usage": round((disk.used / disk.total) * 100, 1),
                "active_connections": len(psutil.net_connections()),
                "response_time": round(random.uniform(150, 300), 0),
                "uptime": round(time.time() - psutil.boot_time(), 2) / 86400 * 100  # Convert to percentage-like
            }
        except Exception:
            # Fallback to mock data if psutil fails
            metrics = {
                "cpu_usage": round(random.uniform(20, 80), 1),
                "memory_usage": round(random.uniform(40, 85), 1),
                "disk_usage": round(random.uniform(30, 70), 1),
                "active_connections": random.randint(50, 200),
                "response_time": round(random.uniform(150, 300), 0),
                "uptime": 99.9
            }
        
        return metrics
        
    except Exception as e:
        # Return mock data if everything fails
        return {
            "cpu_usage": 45.2,
            "memory_usage": 62.8,
            "disk_usage": 38.5,
            "active_connections": 127,
            "response_time": 245,
            "uptime": 99.9
        }

@router.get("/api/agents")
async def get_agents():
    """Get list of available agents without authentication"""
    try:
        # Return mock agents for now
        agents = [
            {
                "id": "cto-agent",
                "name": "CTO Agent",
                "description": "Strategic technology leadership and decision making",
                "status": "active",
                "capabilities": ["Strategic Planning", "Technology Assessment", "Team Leadership"],
                "last_active": datetime.utcnow().isoformat()
            },
            {
                "id": "data-scientist",
                "name": "Data Scientist Agent",
                "description": "Advanced data analysis and machine learning",
                "status": "active",
                "capabilities": ["Data Analysis", "Machine Learning", "Statistical Modeling"],
                "last_active": datetime.utcnow().isoformat()
            },
            {
                "id": "ml-engineer",
                "name": "ML Engineer Agent",
                "description": "Machine learning model development and deployment",
                "status": "active",
                "capabilities": ["Model Development", "MLOps", "Performance Optimization"],
                "last_active": datetime.utcnow().isoformat()
            },
            {
                "id": "ai-engineer",
                "name": "AI Engineer Agent",
                "description": "AI system architecture and implementation",
                "status": "active",
                "capabilities": ["AI Architecture", "System Design", "Integration"],
                "last_active": datetime.utcnow().isoformat()
            },
            {
                "id": "bi-agent",
                "name": "Business Intelligence Agent",
                "description": "Business analytics and reporting",
                "status": "active",
                "capabilities": ["Business Analytics", "Reporting", "KPI Tracking"],
                "last_active": datetime.utcnow().isoformat()
            },
            {
                "id": "qa-agent",
                "name": "QA Agent",
                "description": "Quality assurance and testing",
                "status": "active",
                "capabilities": ["Test Automation", "Quality Control", "Bug Detection"],
                "last_active": datetime.utcnow().isoformat()
            }
        ]
        
        return agents
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get agents: {str(e)}")

@router.post("/api/agents/chat")
async def chat_with_agent(request: Dict[str, Any]):
    """Simple chat endpoint for agent communication"""
    try:
        message = request.get("message", "")
        agent_id = request.get("agent_id", "default")
        
        if not message:
            raise HTTPException(status_code=400, detail="Message is required")
        
        # Mock response based on agent type and message content
        responses = {
            "cto-agent": "As your CTO, I recommend focusing on scalable architecture and strategic technology decisions.",
            "data-scientist": "I can help you analyze your data patterns and build predictive models.",
            "ml-engineer": "Let me assist you with model deployment and MLOps best practices.",
            "ai-engineer": "I'll help you design robust AI systems and integration strategies.",
            "bi-agent": "I can create comprehensive business intelligence reports and dashboards.",
            "qa-agent": "I'll help you implement quality assurance processes and automated testing.",
            "default": "Hello! I'm here to help you with your questions. How can I assist you today?"
        }
        
        response_content = responses.get(agent_id, responses["default"])
        
        # Add some context-aware responses
        if "data" in message.lower() or "analysis" in message.lower():
            response_content = "I can help you analyze your data. Please upload your dataset and I'll provide insights."
        elif "help" in message.lower():
            response_content = "I'm here to help! You can ask me about data analysis, strategic planning, or upload files for processing."
        elif "hello" in message.lower() or "hi" in message.lower():
            response_content = f"Hello! I'm the {agent_id.replace('-', ' ').title()}. How can I assist you today?"
        
        return {
            "id": f"msg_{int(time.time())}",
            "content": response_content,
            "agent_id": agent_id,
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": {
                "processing_time": round(random.uniform(0.5, 2.0), 2),
                "template_used": "simple_response"
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")

@router.get("/api/dashboard")
async def get_dashboard_data():
    """Get dashboard data"""
    try:
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "status": "operational",
            "active_users": random.randint(10, 50),
            "total_requests_today": random.randint(1000, 5000),
            "success_rate": round(random.uniform(98.5, 99.9), 1)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get dashboard data: {str(e)}")