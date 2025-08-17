"""
ScrollIntel Simple FastAPI Application
Minimal version for development without complex dependencies
"""

import os
import sys
from pathlib import Path
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
from datetime import datetime
import json

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Create FastAPI application
app = FastAPI(
    title="ScrollIntel API",
    description="AI-powered CTO replacement platform",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create uploads directory
uploads_dir = Path("uploads")
uploads_dir.mkdir(exist_ok=True)

# Mount static files
if uploads_dir.exists():
    app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# In-memory storage for demo
demo_data = {
    "files": [],
    "analyses": [],
    "models": [],
    "agents": {
        "cto": {"status": "active", "last_activity": datetime.now().isoformat()},
        "data_scientist": {"status": "active", "last_activity": datetime.now().isoformat()},
        "ml_engineer": {"status": "active", "last_activity": datetime.now().isoformat()},
        "ai_engineer": {"status": "active", "last_activity": datetime.now().isoformat()},
        "analyst": {"status": "active", "last_activity": datetime.now().isoformat()},
        "qa": {"status": "active", "last_activity": datetime.now().isoformat()}
    }
}

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "ScrollIntel API is running",
        "version": "1.0.0",
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "features": [
            "File Upload & Processing",
            "AI Agent Chat",
            "Data Analysis",
            "Model Building",
            "Visualization",
            "Real-time Monitoring"
        ]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "api": "running",
            "agents": "active",
            "storage": "available"
        },
        "metrics": {
            "uptime": "running",
            "memory_usage": "normal",
            "response_time": "fast"
        }
    }

@app.get("/health/detailed")
async def detailed_health():
    """Detailed health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "environment": os.getenv("ENVIRONMENT", "development"),
        "services": {
            "api": {"status": "running", "port": 8000},
            "database": {"status": "connected", "type": "sqlite"},
            "cache": {"status": "available", "type": "memory"},
            "ai_agents": {"status": "active", "count": 6}
        },
        "metrics": {
            "total_files": len(demo_data["files"]),
            "total_analyses": len(demo_data["analyses"]),
            "total_models": len(demo_data["models"]),
            "active_agents": len([a for a in demo_data["agents"].values() if a["status"] == "active"])
        }
    }

@app.post("/api/files/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload and process file"""
    try:
        # Save file
        file_path = uploads_dir / file.filename
        content = await file.read()
        
        with open(file_path, "wb") as f:
            f.write(content)
        
        # Create file record
        file_record = {
            "id": len(demo_data["files"]) + 1,
            "filename": file.filename,
            "size": len(content),
            "type": file.content_type,
            "uploaded_at": datetime.now().isoformat(),
            "status": "processed",
            "path": str(file_path)
        }
        
        demo_data["files"].append(file_record)
        
        # Simulate analysis
        analysis = {
            "id": len(demo_data["analyses"]) + 1,
            "file_id": file_record["id"],
            "type": "auto_analysis",
            "results": {
                "rows": 1000 if file.filename.endswith('.csv') else None,
                "columns": 15 if file.filename.endswith('.csv') else None,
                "data_types": ["numeric", "categorical", "text"] if file.filename.endswith('.csv') else [],
                "quality_score": 0.85,
                "insights": [
                    "Data appears to be well-structured",
                    "No missing values detected in key columns",
                    "Suitable for machine learning models"
                ]
            },
            "created_at": datetime.now().isoformat()
        }
        
        demo_data["analyses"].append(analysis)
        
        return {
            "success": True,
            "file": file_record,
            "analysis": analysis,
            "message": f"File {file.filename} uploaded and analyzed successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.get("/api/files")
async def list_files():
    """List uploaded files"""
    return {
        "files": demo_data["files"],
        "total": len(demo_data["files"])
    }

@app.get("/api/analyses")
async def list_analyses():
    """List file analyses"""
    return {
        "analyses": demo_data["analyses"],
        "total": len(demo_data["analyses"])
    }

@app.get("/api/agents")
async def list_agents():
    """List AI agents"""
    return {
        "agents": demo_data["agents"],
        "total": len(demo_data["agents"])
    }

@app.post("/api/agents/{agent_name}/chat")
async def chat_with_agent(agent_name: str, message: dict):
    """Chat with AI agent"""
    if agent_name not in demo_data["agents"]:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    user_message = message.get("message", "")
    
    # Simulate agent responses
    responses = {
        "cto": f"As your CTO, I've analyzed your request: '{user_message}'. Here's my strategic recommendation: Focus on data quality and model performance metrics.",
        "data_scientist": f"From a data science perspective on '{user_message}': I recommend exploratory data analysis followed by feature engineering.",
        "ml_engineer": f"For ML engineering regarding '{user_message}': Let's build a robust pipeline with proper validation and monitoring.",
        "ai_engineer": f"AI engineering insight on '{user_message}': We should implement this with scalable architecture and proper error handling.",
        "analyst": f"Business analysis of '{user_message}': This aligns with our KPIs and should drive measurable business value.",
        "qa": f"QA perspective on '{user_message}': We need comprehensive testing including edge cases and performance validation."
    }
    
    # Update agent activity
    demo_data["agents"][agent_name]["last_activity"] = datetime.now().isoformat()
    
    return {
        "agent": agent_name,
        "response": responses.get(agent_name, "I'm here to help with your request."),
        "timestamp": datetime.now().isoformat(),
        "status": "success"
    }

@app.post("/api/models/build")
async def build_model(request: dict):
    """Build ML model"""
    try:
        model_config = {
            "id": len(demo_data["models"]) + 1,
            "name": request.get("name", f"Model_{len(demo_data['models']) + 1}"),
            "type": request.get("type", "classification"),
            "algorithm": request.get("algorithm", "random_forest"),
            "file_id": request.get("file_id"),
            "target_column": request.get("target_column"),
            "features": request.get("features", []),
            "status": "training",
            "created_at": datetime.now().isoformat(),
            "metrics": {
                "accuracy": 0.92,
                "precision": 0.89,
                "recall": 0.94,
                "f1_score": 0.91
            }
        }
        
        demo_data["models"].append(model_config)
        
        return {
            "success": True,
            "model": model_config,
            "message": "Model training started successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model building failed: {str(e)}")

@app.get("/api/models")
async def list_models():
    """List ML models"""
    return {
        "models": demo_data["models"],
        "total": len(demo_data["models"])
    }

@app.get("/api/dashboard")
async def get_dashboard():
    """Get dashboard data"""
    return {
        "summary": {
            "total_files": len(demo_data["files"]),
            "total_analyses": len(demo_data["analyses"]),
            "total_models": len(demo_data["models"]),
            "active_agents": len([a for a in demo_data["agents"].values() if a["status"] == "active"])
        },
        "recent_activity": [
            {
                "type": "file_upload",
                "message": f"Uploaded {len(demo_data['files'])} files",
                "timestamp": datetime.now().isoformat()
            },
            {
                "type": "analysis",
                "message": f"Completed {len(demo_data['analyses'])} analyses",
                "timestamp": datetime.now().isoformat()
            },
            {
                "type": "model",
                "message": f"Built {len(demo_data['models'])} models",
                "timestamp": datetime.now().isoformat()
            }
        ],
        "agents": demo_data["agents"],
        "timestamp": datetime.now().isoformat()
    }

@app.get("/docs")
async def api_docs():
    """API documentation redirect"""
    return {
        "message": "ScrollIntel API Documentation",
        "endpoints": {
            "health": "/health",
            "files": "/api/files",
            "upload": "/api/files/upload",
            "agents": "/api/agents",
            "chat": "/api/agents/{agent_name}/chat",
            "models": "/api/models",
            "dashboard": "/api/dashboard"
        },
        "version": "1.0.0"
    }

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"error": "Not found", "message": "The requested resource was not found"}
    )

@app.exception_handler(500)
async def server_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "message": "An unexpected error occurred"}
    )

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )