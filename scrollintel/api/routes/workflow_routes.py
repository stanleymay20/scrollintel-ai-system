"""
API routes for workflow automation system.
"""
import uuid
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session

from ...core.database import SessionLocal
from ...models.workflow_models import (
    WorkflowDefinitionCreate, WorkflowDefinitionResponse,
    WorkflowExecutionCreate, WorkflowExecutionResponse,
    WebhookConfigCreate, WebhookConfigResponse,
    WorkflowTemplateCreate, WorkflowTemplateResponse,
    WorkflowDefinition, WorkflowExecution, WebhookConfig, WorkflowTemplate
)
from ...engines.workflow_engine import WorkflowEngine
from ...core.auth import get_current_user

router = APIRouter(prefix="/api/v1/workflows", tags=["workflows"])

@router.post("/", response_model=Dict[str, str])
async def create_workflow(
    workflow_data: WorkflowDefinitionCreate,
    current_user: dict = Depends(get_current_user)
):
    """Create a new workflow definition."""
    try:
        engine = WorkflowEngine()
        workflow_id = await engine.create_workflow(
            workflow_data.dict(),
            current_user["user_id"]
        )
        return {"workflow_id": workflow_id, "status": "created"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/", response_model=List[WorkflowDefinitionResponse])
async def list_workflows(
    skip: int = 0,
    limit: int = 100,
    current_user: dict = Depends(get_current_user)
):
    """List workflow definitions."""
    db = SessionLocal()
    try:
        workflows = db.query(WorkflowDefinition).offset(skip).limit(limit).all()
        return workflows
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        db.close()

@router.get("/{workflow_id}", response_model=WorkflowDefinitionResponse)
async def get_workflow(
    workflow_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get workflow definition by ID."""
    try:
        with get_db_session() as db:
            workflow = db.query(WorkflowDefinition).filter(
                WorkflowDefinition.id == workflow_id
            ).first()
            if not workflow:
                raise HTTPException(status_code=404, detail="Workflow not found")
            return workflow
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.put("/{workflow_id}", response_model=Dict[str, str])
async def update_workflow(
    workflow_id: str,
    workflow_data: WorkflowDefinitionCreate,
    current_user: dict = Depends(get_current_user)
):
    """Update workflow definition."""
    try:
        with get_db_session() as db:
            workflow = db.query(WorkflowDefinition).filter(
                WorkflowDefinition.id == workflow_id
            ).first()
            if not workflow:
                raise HTTPException(status_code=404, detail="Workflow not found")
            
            # Update workflow fields
            for field, value in workflow_data.dict().items():
                setattr(workflow, field, value)
            
            db.commit()
            return {"workflow_id": workflow_id, "status": "updated"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.delete("/{workflow_id}", response_model=Dict[str, str])
async def delete_workflow(
    workflow_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Delete workflow definition."""
    try:
        with get_db_session() as db:
            workflow = db.query(WorkflowDefinition).filter(
                WorkflowDefinition.id == workflow_id
            ).first()
            if not workflow:
                raise HTTPException(status_code=404, detail="Workflow not found")
            
            db.delete(workflow)
            db.commit()
            return {"workflow_id": workflow_id, "status": "deleted"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/{workflow_id}/execute", response_model=Dict[str, str])
async def execute_workflow(
    workflow_id: str,
    execution_data: Optional[WorkflowExecutionCreate] = None,
    background_tasks: BackgroundTasks = BackgroundTasks(),
    current_user: dict = Depends(get_current_user)
):
    """Execute a workflow."""
    try:
        engine = WorkflowEngine()
        input_data = execution_data.input_data if execution_data else None
        
        # Execute workflow in background
        execution_id = await engine.execute_workflow(workflow_id, input_data)
        
        return {"execution_id": execution_id, "status": "started"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/{workflow_id}/executions", response_model=List[WorkflowExecutionResponse])
async def list_workflow_executions(
    workflow_id: str,
    skip: int = 0,
    limit: int = 100,
    current_user: dict = Depends(get_current_user)
):
    """List workflow executions."""
    try:
        with get_db_session() as db:
            executions = db.query(WorkflowExecution).filter(
                WorkflowExecution.workflow_id == workflow_id
            ).offset(skip).limit(limit).all()
            return executions
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/executions/{execution_id}", response_model=WorkflowExecutionResponse)
async def get_workflow_execution(
    execution_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get workflow execution by ID."""
    try:
        with get_db_session() as db:
            execution = db.query(WorkflowExecution).filter(
                WorkflowExecution.id == execution_id
            ).first()
            if not execution:
                raise HTTPException(status_code=404, detail="Execution not found")
            return execution
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/{workflow_id}/webhooks", response_model=Dict[str, str])
async def create_webhook(
    workflow_id: str,
    webhook_data: WebhookConfigCreate,
    current_user: dict = Depends(get_current_user)
):
    """Create webhook configuration for workflow."""
    try:
        engine = WorkflowEngine()
        webhook_id = await engine.webhook_manager.create_webhook(
            workflow_id,
            webhook_data.dict()
        )
        return {"webhook_id": webhook_id, "status": "created"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/{workflow_id}/webhooks", response_model=List[WebhookConfigResponse])
async def list_webhooks(
    workflow_id: str,
    current_user: dict = Depends(get_current_user)
):
    """List webhook configurations for workflow."""
    try:
        with get_db_session() as db:
            webhooks = db.query(WebhookConfig).filter(
                WebhookConfig.workflow_id == workflow_id
            ).all()
            return webhooks
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/webhooks/{webhook_id}/callback", response_model=Dict[str, str])
async def webhook_callback(
    webhook_id: str,
    payload: Dict[str, Any],
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Handle webhook callback."""
    try:
        engine = WorkflowEngine()
        result = await engine.webhook_manager.handle_webhook_callback(webhook_id, payload)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/templates/", response_model=List[WorkflowTemplateResponse])
async def list_workflow_templates(
    category: Optional[str] = None,
    current_user: dict = Depends(get_current_user)
):
    """List workflow templates."""
    try:
        engine = WorkflowEngine()
        templates = engine.get_workflow_templates(category)
        return templates
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/templates/", response_model=Dict[str, str])
async def create_workflow_template(
    template_data: WorkflowTemplateCreate,
    current_user: dict = Depends(get_current_user)
):
    """Create a new workflow template."""
    try:
        with get_db_session() as db:
            template_id = str(uuid.uuid4())
            
            template = WorkflowTemplate(
                id=template_id,
                name=template_data.name,
                description=template_data.description,
                category=template_data.category,
                integration_type=template_data.integration_type,
                template_config=template_data.template_config,
                is_public=template_data.is_public,
                created_by=current_user["user_id"]
            )
            
            db.add(template)
            db.commit()
            
            return {"template_id": template_id, "status": "created"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/templates/{template_id}/instantiate", response_model=Dict[str, str])
async def instantiate_template(
    template_id: str,
    workflow_name: str,
    customizations: Optional[Dict[str, Any]] = None,
    current_user: dict = Depends(get_current_user)
):
    """Create workflow from template."""
    try:
        with get_db_session() as db:
            template = db.query(WorkflowTemplate).filter(
                WorkflowTemplate.id == template_id
            ).first()
            if not template:
                raise HTTPException(status_code=404, detail="Template not found")
            
            # Create workflow from template
            workflow_config = template.template_config.copy()
            if customizations:
                workflow_config.update(customizations)
            
            workflow_config["name"] = workflow_name
            
            engine = WorkflowEngine()
            workflow_id = await engine.create_workflow(workflow_config, current_user["user_id"])
            
            return {"workflow_id": workflow_id, "status": "created"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/{workflow_id}/status", response_model=Dict[str, Any])
async def get_workflow_status(
    workflow_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get workflow status and recent executions."""
    try:
        with get_db_session() as db:
            workflow = db.query(WorkflowDefinition).filter(
                WorkflowDefinition.id == workflow_id
            ).first()
            if not workflow:
                raise HTTPException(status_code=404, detail="Workflow not found")
            
            # Get recent executions
            recent_executions = db.query(WorkflowExecution).filter(
                WorkflowExecution.workflow_id == workflow_id
            ).order_by(WorkflowExecution.started_at.desc()).limit(10).all()
            
            return {
                "workflow_id": workflow_id,
                "status": workflow.status,
                "recent_executions": [
                    {
                        "id": ex.id,
                        "status": ex.status,
                        "started_at": ex.started_at,
                        "completed_at": ex.completed_at
                    }
                    for ex in recent_executions
                ]
            }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))