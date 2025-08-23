#!/usr/bin/env python3
"""
Simple ScrollIntel Backend Server
Minimal setup for immediate demo
"""

import os
import sys
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, Any, List
import uvicorn
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="ScrollIntel™ AI Platform",
    description="AI-Powered CTO Platform with intelligent agents",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class ChatMessage(BaseModel):
    message: str
    agent_type: str = "cto"

class FileUpload(BaseModel):
    filename: str
    content_type: str
    size: int

class AnalysisResult(BaseModel):
    insights: List[str]
    recommendations: List[str]
    metrics: Dict[str, Any]

# Mock data for demo
MOCK_AGENTS = {
    "cto": {
        "name": "CTO Agent",
        "description": "Strategic technical decision making",
        "status": "active",
        "capabilities": ["Architecture", "Strategy", "Team Management"]
    },
    "data_scientist": {
        "name": "Data Scientist",
        "description": "Advanced analytics and insights",
        "status": "active",
        "capabilities": ["ML Models", "Statistical Analysis", "Data Visualization"]
    },
    "ml_engineer": {
        "name": "ML Engineer",
        "description": "Model building and deployment",
        "status": "active",
        "capabilities": ["AutoML", "Model Deployment", "Performance Optimization"]
    },
    "ai_engineer": {
        "name": "AI Engineer",
        "description": "AI system architecture",
        "status": "active",
        "capabilities": ["AI Systems", "Neural Networks", "Deep Learning"]
    },
    "business_analyst": {
        "name": "Business Analyst",
        "description": "Business intelligence and reporting",
        "status": "active",
        "capabilities": ["BI Dashboards", "KPI Tracking", "Business Insights"]
    }
}

MOCK_INSIGHTS = [
    "Your data shows strong growth patterns in Q4",
    "Customer retention rate has improved by 15%",
    "Revenue per user is trending upward",
    "Market expansion opportunities identified in 3 regions",
    "Operational efficiency can be improved by 20%"
]

MOCK_RECOMMENDATIONS = [
    "Implement automated data pipeline for real-time analytics",
    "Deploy ML model for customer churn prediction",
    "Optimize database queries to reduce response time by 40%",
    "Scale infrastructure to handle 10x traffic growth",
    "Integrate AI-powered recommendation engine"
]

# API Routes
@app.get("/")
async def root():
    return {
        "message": "Welcome to ScrollIntel™ AI Platform",
        "version": "1.0.0",
        "status": "running",
        "features": [
            "AI Agents",
            "Data Analysis",
            "ML Models",
            "Real-time Insights",
            "Business Intelligence"
        ]
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": "2025-01-23T10:00:00Z",
        "services": {
            "api": "running",
            "database": "connected",
            "ai_agents": "active"
        }
    }

@app.get("/agents")
async def get_agents():
    return {
        "agents": MOCK_AGENTS,
        "total": len(MOCK_AGENTS),
        "active": len([a for a in MOCK_AGENTS.values() if a["status"] == "active"])
    }

@app.get("/agents/{agent_id}")
async def get_agent(agent_id: str):
    if agent_id not in MOCK_AGENTS:
        raise HTTPException(status_code=404, detail="Agent not found")
    return MOCK_AGENTS[agent_id]

@app.post("/chat")
async def chat_with_agent(message: ChatMessage):
    agent = MOCK_AGENTS.get(message.agent_type)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    # Mock AI response based on message content
    if "data" in message.message.lower():
        response = f"As your {agent['name']}, I've analyzed your data and found several key insights. Your metrics show positive trends across all major KPIs."
    elif "model" in message.message.lower():
        response = f"I recommend implementing a machine learning model for this use case. Based on your data patterns, a gradient boosting model would be optimal."
    elif "strategy" in message.message.lower():
        response = f"From a strategic perspective, I suggest focusing on three key areas: scalability, performance optimization, and team development."
    else:
        response = f"Hello! I'm your {agent['name']}. I can help you with {', '.join(agent['capabilities'])}. What would you like to work on today?"
    
    return {
        "response": response,
        "agent": agent["name"],
        "timestamp": "2025-01-23T10:00:00Z",
        "suggestions": [
            "Upload your data for analysis",
            "Build a predictive model",
            "Create a dashboard",
            "Get strategic recommendations"
        ]
    }

@app.post("/upload")
async def upload_file(file_info: FileUpload):
    # Mock file processing
    return {
        "message": f"File '{file_info.filename}' uploaded successfully",
        "file_id": f"file_{hash(file_info.filename) % 10000}",
        "size": file_info.size,
        "status": "processing",
        "estimated_completion": "2-3 minutes"
    }

@app.get("/analysis/{file_id}")
async def get_analysis(file_id: str):
    # Mock analysis results
    return AnalysisResult(
        insights=MOCK_INSIGHTS[:3],
        recommendations=MOCK_RECOMMENDATIONS[:3],
        metrics={
            "rows": 10000,
            "columns": 25,
            "quality_score": 0.92,
            "completeness": 0.98,
            "accuracy": 0.95
        }
    )

@app.get("/dashboard")
async def get_dashboard():
    return {
        "widgets": [
            {
                "id": "revenue",
                "title": "Revenue Growth",
                "type": "line_chart",
                "value": "$2.4M",
                "change": "+15.3%"
            },
            {
                "id": "users",
                "title": "Active Users",
                "type": "metric",
                "value": "45,231",
                "change": "+8.7%"
            },
            {
                "id": "models",
                "title": "ML Models",
                "type": "status",
                "value": "12 Active",
                "change": "+2 this week"
            },
            {
                "id": "insights",
                "title": "AI Insights",
                "type": "list",
                "value": "23 New",
                "change": "+5 today"
            }
        ],
        "alerts": [
            {
                "type": "info",
                "message": "New data available for analysis",
                "timestamp": "2025-01-23T09:45:00Z"
            },
            {
                "type": "success",
                "message": "Model deployment completed successfully",
                "timestamp": "2025-01-23T09:30:00Z"
            }
        ]
    }

@app.get("/models")
async def get_models():
    return {
        "models": [
            {
                "id": "model_1",
                "name": "Customer Churn Prediction",
                "type": "classification",
                "accuracy": 0.94,
                "status": "deployed"
            },
            {
                "id": "model_2",
                "name": "Revenue Forecasting",
                "type": "regression",
                "accuracy": 0.89,
                "status": "training"
            },
            {
                "id": "model_3",
                "name": "Recommendation Engine",
                "type": "collaborative_filtering",
                "accuracy": 0.87,
                "status": "deployed"
            }
        ]
    }

@app.post("/models/build")
async def build_model(model_config: Dict[str, Any]):
    return {
        "message": "Model training started",
        "model_id": f"model_{hash(str(model_config)) % 10000}",
        "estimated_time": "15-20 minutes",
        "status": "training"
    }

if __name__ == "__main__":
    logger.info("🚀 Starting ScrollIntel™ Backend Server...")
    logger.info("📱 Frontend should connect to: http://localhost:8000")
    logger.info("📚 API Documentation: http://localhost:8000/docs")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )