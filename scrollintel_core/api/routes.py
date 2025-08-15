"""
API Routes for ScrollIntel Core
Focused endpoints for 7 core agents with Natural Language Interface
"""
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session
from typing import Dict, Any, List, Optional
import json
import os
import uuid
from datetime import datetime

from ..database import get_db
from ..models import Dataset, Analysis, User, Workspace
from ..agents.orchestrator import AgentOrchestrator
from ..config import settings
from . import nl_routes

# Create router
router = APIRouter(prefix=settings.API_V1_PREFIX)

# Global orchestrator instance (will be injected in main.py)
orchestrator: Optional[AgentOrchestrator] = None


def set_orchestrator(orch: AgentOrchestrator):
    """Set the global orchestrator instance"""
    global orchestrator
    orchestrator = orch
    # Also set orchestrator for NL routes
    nl_routes.set_orchestrator(orch)


# Agent Endpoints
@router.post("/agents/process")
async def process_agent_request(
    request: Dict[str, Any],
    db: Session = Depends(get_db)
):
    """Process request through agent orchestrator"""
    if not orchestrator:
        raise HTTPException(status_code=500, detail="Agent orchestrator not initialized")
    
    try:
        result = await orchestrator.process_request(request)
        
        # Log analysis if dataset_id provided
        if request.get("dataset_id") and request.get("user_id"):
            analysis = Analysis(
                dataset_id=request["dataset_id"],
                user_id=request["user_id"],
                agent_type=result.get("agent", "unknown"),
                query=request.get("query", ""),
                results=result.get("result"),
                metadata=result.get("metadata", {}),
                processing_time=result.get("processing_time", 0),
                success=result.get("success", False),
                error_message=result.get("error")
            )
            db.add(analysis)
            db.commit()
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/agents")
async def get_available_agents():
    """Get list of available agents"""
    if not orchestrator:
        raise HTTPException(status_code=500, detail="Agent orchestrator not initialized")
    
    return await orchestrator.get_available_agents()


@router.get("/agents/health")
async def get_agents_health():
    """Get health status of all agents"""
    if not orchestrator:
        raise HTTPException(status_code=500, detail="Agent orchestrator not initialized")
    
    return await orchestrator.health_check()


@router.get("/agents/stats")
async def get_agent_stats():
    """Get agent usage statistics"""
    if not orchestrator:
        raise HTTPException(status_code=500, detail="Agent orchestrator not initialized")
    
    return orchestrator.get_request_stats()


# File Upload Endpoints
@router.post("/files/upload")
async def upload_file(
    file: UploadFile = File(...),
    workspace_id: str = Form(...),
    user_id: str = Form(...),
    name: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    """Upload and process data file"""
    try:
        # Validate file size
        if file.size > settings.MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size: {settings.MAX_FILE_SIZE} bytes"
            )
        
        # Validate file type
        allowed_types = [".csv", ".xlsx", ".xls", ".json", ".parquet"]
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in allowed_types:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type. Allowed: {', '.join(allowed_types)}"
            )
        
        # Generate unique filename
        file_id = str(uuid.uuid4())
        filename = f"{file_id}{file_ext}"
        file_path = os.path.join(settings.UPLOAD_DIR, filename)
        
        # Ensure upload directory exists
        os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
        
        # Save file
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Basic file analysis
        import pandas as pd
        
        try:
            if file_ext == ".csv":
                df = pd.read_csv(file_path, nrows=5)  # Sample for schema
            elif file_ext in [".xlsx", ".xls"]:
                df = pd.read_excel(file_path, nrows=5)
            elif file_ext == ".json":
                df = pd.read_json(file_path, nrows=5)
            elif file_ext == ".parquet":
                df = pd.read_parquet(file_path)
                df = df.head(5)
            
            # Extract schema information
            schema_info = {
                "columns": list(df.columns),
                "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                "sample_data": df.to_dict("records")
            }
            
            # Get full row count
            if file_ext == ".csv":
                full_df = pd.read_csv(file_path)
            elif file_ext in [".xlsx", ".xls"]:
                full_df = pd.read_excel(file_path)
            elif file_ext == ".json":
                full_df = pd.read_json(file_path)
            elif file_ext == ".parquet":
                full_df = pd.read_parquet(file_path)
            
            row_count = len(full_df)
            column_count = len(full_df.columns)
            
        except Exception as e:
            # If file processing fails, still save the record
            schema_info = {"error": f"Could not process file: {str(e)}"}
            row_count = 0
            column_count = 0
        
        # Create dataset record
        dataset = Dataset(
            name=name or file.filename,
            description=description,
            file_path=file_path,
            file_size=file.size,
            file_type=file_ext[1:],  # Remove the dot
            schema_info=schema_info,
            row_count=row_count,
            column_count=column_count,
            workspace_id=workspace_id,
            owner_id=user_id
        )
        
        db.add(dataset)
        db.commit()
        db.refresh(dataset)
        
        return {
            "dataset_id": str(dataset.id),
            "name": dataset.name,
            "file_type": dataset.file_type,
            "row_count": dataset.row_count,
            "column_count": dataset.column_count,
            "schema_info": dataset.schema_info,
            "created_at": dataset.created_at.isoformat()
        }
        
    except Exception as e:
        # Clean up file if database operation fails
        if 'file_path' in locals() and os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/files/{dataset_id}")
async def get_dataset_info(dataset_id: str, db: Session = Depends(get_db)):
    """Get dataset information"""
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    return {
        "id": str(dataset.id),
        "name": dataset.name,
        "description": dataset.description,
        "file_type": dataset.file_type,
        "row_count": dataset.row_count,
        "column_count": dataset.column_count,
        "schema_info": dataset.schema_info,
        "created_at": dataset.created_at.isoformat()
    }


@router.get("/files")
async def list_datasets(
    workspace_id: Optional[str] = None,
    user_id: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """List datasets"""
    query = db.query(Dataset)
    
    if workspace_id:
        query = query.filter(Dataset.workspace_id == workspace_id)
    if user_id:
        query = query.filter(Dataset.owner_id == user_id)
    
    datasets = query.order_by(Dataset.created_at.desc()).all()
    
    return [
        {
            "id": str(dataset.id),
            "name": dataset.name,
            "description": dataset.description,
            "file_type": dataset.file_type,
            "row_count": dataset.row_count,
            "column_count": dataset.column_count,
            "created_at": dataset.created_at.isoformat()
        }
        for dataset in datasets
    ]


# Analysis Endpoints
@router.get("/analyses/{dataset_id}")
async def get_dataset_analyses(dataset_id: str, db: Session = Depends(get_db)):
    """Get analyses for a dataset"""
    analyses = db.query(Analysis).filter(
        Analysis.dataset_id == dataset_id
    ).order_by(Analysis.created_at.desc()).all()
    
    return [
        {
            "id": str(analysis.id),
            "agent_type": analysis.agent_type,
            "query": analysis.query,
            "results": analysis.results,
            "success": analysis.success,
            "processing_time": analysis.processing_time,
            "created_at": analysis.created_at.isoformat()
        }
        for analysis in analyses
    ]


# Workspace Endpoints
@router.get("/workspaces")
async def list_workspaces(user_id: str, db: Session = Depends(get_db)):
    """List user workspaces"""
    # This is simplified - in production you'd check workspace membership
    workspaces = db.query(Workspace).filter(Workspace.owner_id == user_id).all()
    
    return [
        {
            "id": str(workspace.id),
            "name": workspace.name,
            "description": workspace.description,
            "created_at": workspace.created_at.isoformat()
        }
        for workspace in workspaces
    ]


@router.post("/workspaces")
async def create_workspace(
    workspace_data: Dict[str, Any],
    db: Session = Depends(get_db)
):
    """Create new workspace"""
    workspace = Workspace(
        name=workspace_data["name"],
        description=workspace_data.get("description"),
        owner_id=workspace_data["owner_id"],
        settings=workspace_data.get("settings", {})
    )
    
    db.add(workspace)
    db.commit()
    db.refresh(workspace)
    
    return {
        "id": str(workspace.id),
        "name": workspace.name,
        "description": workspace.description,
        "created_at": workspace.created_at.isoformat()
    }