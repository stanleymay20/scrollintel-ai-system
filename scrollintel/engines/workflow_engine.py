"""
Workflow automation engine with support for Zapier, Power Automate, and Airflow integrations.
"""
import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
import logging
from enum import Enum

import httpx
import requests
from sqlalchemy.orm import Session

from ..models.workflow_models import (
    WorkflowDefinition, WorkflowExecution, WorkflowStepExecution,
    WebhookConfig, WorkflowTemplate, WorkflowStatus, ProcessingMode,
    IntegrationType, TriggerType
)
from ..core.database import SessionLocal

logger = logging.getLogger(__name__)

class WorkflowEngine:
    """Main workflow automation engine."""
    
    def __init__(self):
        self.integrations = {
            IntegrationType.ZAPIER: ZapierIntegration(),
            IntegrationType.POWER_AUTOMATE: PowerAutomateIntegration(),
            IntegrationType.AIRFLOW: AirflowIntegration(),
            IntegrationType.CUSTOM: CustomIntegration()
        }
        self.webhook_manager = WebhookManager()
        self.retry_manager = RetryManager()
        
    async def create_workflow(self, workflow_data: Dict[str, Any], user_id: str) -> str:
        """Create a new workflow definition."""
        db = SessionLocal()
        try:
            workflow_id = str(uuid.uuid4())
            
            workflow = WorkflowDefinition(
                id=workflow_id,
                name=workflow_data["name"],
                description=workflow_data.get("description"),
                integration_type=workflow_data["integration_type"],
                trigger_config=workflow_data["trigger_config"],
                steps=workflow_data["steps"],
                processing_mode=workflow_data.get("processing_mode", ProcessingMode.REAL_TIME),
                created_by=user_id
            )
            
            db.add(workflow)
            db.commit()
            
            # Set up webhooks if needed
            if workflow_data["trigger_config"]["type"] == TriggerType.WEBHOOK:
                await self.webhook_manager.create_webhook(workflow_id, workflow_data["trigger_config"]["config"])
            
            logger.info(f"Created workflow {workflow_id}")
            return workflow_id
            
        except Exception as e:
            logger.error(f"Error creating workflow: {str(e)}")
            raise
        finally:
            db.close()
    
    async def execute_workflow(self, workflow_id: str, input_data: Optional[Dict[str, Any]] = None) -> str:
        """Execute a workflow."""
        db = SessionLocal()
        execution_id = str(uuid.uuid4())
        try:
            workflow = db.query(WorkflowDefinition).filter(WorkflowDefinition.id == workflow_id).first()
            if not workflow:
                raise ValueError(f"Workflow {workflow_id} not found")
            
            execution = WorkflowExecution(
                id=execution_id,
                workflow_id=workflow_id,
                input_data=input_data or {},
                status=WorkflowStatus.ACTIVE
            )
            
            db.add(execution)
            db.commit()
            
            # Execute based on processing mode
            if workflow.processing_mode == ProcessingMode.BATCH:
                await self._execute_batch_workflow(execution_id, workflow, input_data)
            else:
                await self._execute_realtime_workflow(execution_id, workflow, input_data)
            
            return execution_id
            
        except Exception as e:
            logger.error(f"Error executing workflow {workflow_id}: {str(e)}")
            await self._mark_execution_failed(execution_id, str(e))
            raise
        finally:
            db.close()
    
    async def _execute_realtime_workflow(self, execution_id: str, workflow: WorkflowDefinition, input_data: Dict[str, Any]):
        """Execute workflow in real-time mode."""
        try:
            integration = self.integrations[IntegrationType(workflow.integration_type)]
            
            current_data = input_data or {}
            
            for step_config in workflow.steps:
                step_id = str(uuid.uuid4())
                
                db = SessionLocal()
                try:
                    step_execution = WorkflowStepExecution(
                        id=step_id,
                        execution_id=execution_id,
                        step_name=step_config["name"],
                        step_type=step_config["type"],
                        input_data=current_data,
                        status=WorkflowStatus.ACTIVE
                    )
                    db.add(step_execution)
                    db.commit()
                finally:
                    db.close()
                
                try:
                    # Execute step with retry logic
                    result = await self.retry_manager.execute_with_retry(
                        integration.execute_step,
                        step_config,
                        current_data
                    )
                    
                    current_data = result
                    
                    db = SessionLocal()
                    try:
                        step_execution = db.query(WorkflowStepExecution).filter(
                            WorkflowStepExecution.id == step_id
                        ).first()
                        step_execution.output_data = result
                        step_execution.status = WorkflowStatus.COMPLETED
                        step_execution.completed_at = datetime.utcnow()
                        db.commit()
                    finally:
                        db.close()
                        
                except Exception as step_error:
                    db = SessionLocal()
                    try:
                        step_execution = db.query(WorkflowStepExecution).filter(
                            WorkflowStepExecution.id == step_id
                        ).first()
                        step_execution.error_message = str(step_error)
                        step_execution.status = WorkflowStatus.FAILED
                        step_execution.completed_at = datetime.utcnow()
                        db.commit()
                    finally:
                        db.close()
                    raise
            
            # Mark execution as completed
            db = SessionLocal()
            try:
                execution = db.query(WorkflowExecution).filter(
                    WorkflowExecution.id == execution_id
                ).first()
                execution.output_data = current_data
                execution.status = WorkflowStatus.COMPLETED
                execution.completed_at = datetime.utcnow()
                db.commit()
            finally:
                db.close()
                
        except Exception as e:
            await self._mark_execution_failed(execution_id, str(e))
            raise
    
    async def _execute_batch_workflow(self, execution_id: str, workflow: WorkflowDefinition, input_data: Dict[str, Any]):
        """Execute workflow in batch mode."""
        # For batch processing, we might queue the workflow for later execution
        # or process multiple items at once
        await self._execute_realtime_workflow(execution_id, workflow, input_data)
    
    async def _mark_execution_failed(self, execution_id: str, error_message: str):
        """Mark workflow execution as failed."""
        db = SessionLocal()
        try:
            execution = db.query(WorkflowExecution).filter(
                WorkflowExecution.id == execution_id
            ).first()
            if execution:
                execution.error_message = error_message
                execution.status = WorkflowStatus.FAILED
                execution.completed_at = datetime.utcnow()
                db.commit()
        except Exception as e:
            logger.error(f"Error marking execution as failed: {str(e)}")
        finally:
            db.close()
    
    def get_workflow_templates(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get available workflow templates."""
        db = SessionLocal()
        try:
            query = db.query(WorkflowTemplate)
            if category:
                query = query.filter(WorkflowTemplate.category == category)
            
            templates = query.all()
            return [
                {
                    "id": t.id,
                    "name": t.name,
                    "description": t.description,
                    "category": t.category,
                    "integration_type": t.integration_type,
                    "template_config": t.template_config
                }
                for t in templates
            ]
        except Exception as e:
            logger.error(f"Error getting workflow templates: {str(e)}")
            return []
        finally:
            db.close()

class ZapierIntegration:
    """Integration with Zapier workflows."""
    
    async def execute_step(self, step_config: Dict[str, Any], input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a Zapier workflow step."""
        try:
            # Zapier webhook URL from step config
            webhook_url = step_config["config"].get("webhook_url")
            if not webhook_url:
                raise ValueError("Zapier webhook URL not configured")
            
            # Prepare payload
            payload = {
                "input_data": input_data,
                "step_config": step_config["config"]
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    webhook_url,
                    json=payload,
                    timeout=30.0
                )
                response.raise_for_status()
                
                return response.json()
                
        except Exception as e:
            logger.error(f"Zapier integration error: {str(e)}")
            raise

class PowerAutomateIntegration:
    """Integration with Microsoft Power Automate."""
    
    async def execute_step(self, step_config: Dict[str, Any], input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a Power Automate flow step."""
        try:
            # Power Automate HTTP trigger URL
            flow_url = step_config["config"].get("flow_url")
            if not flow_url:
                raise ValueError("Power Automate flow URL not configured")
            
            # Prepare payload
            payload = {
                "input_data": input_data,
                "step_config": step_config["config"]
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    flow_url,
                    json=payload,
                    timeout=60.0
                )
                response.raise_for_status()
                
                return response.json()
                
        except Exception as e:
            logger.error(f"Power Automate integration error: {str(e)}")
            raise

class AirflowIntegration:
    """Integration with Apache Airflow."""
    
    async def execute_step(self, step_config: Dict[str, Any], input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an Airflow DAG step."""
        try:
            # Airflow API configuration
            airflow_config = step_config["config"]
            base_url = airflow_config.get("base_url")
            dag_id = airflow_config.get("dag_id")
            
            if not base_url or not dag_id:
                raise ValueError("Airflow configuration incomplete")
            
            # Trigger DAG run
            trigger_url = f"{base_url}/api/v1/dags/{dag_id}/dagRuns"
            
            payload = {
                "conf": {
                    "input_data": input_data,
                    "step_config": step_config["config"]
                }
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    trigger_url,
                    json=payload,
                    timeout=30.0,
                    auth=(airflow_config.get("username"), airflow_config.get("password"))
                )
                response.raise_for_status()
                
                dag_run = response.json()
                
                # Poll for completion (simplified)
                return await self._poll_dag_run(base_url, dag_id, dag_run["dag_run_id"], airflow_config)
                
        except Exception as e:
            logger.error(f"Airflow integration error: {str(e)}")
            raise
    
    async def _poll_dag_run(self, base_url: str, dag_id: str, dag_run_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Poll Airflow DAG run for completion."""
        max_polls = 60  # 5 minutes with 5-second intervals
        poll_count = 0
        
        async with httpx.AsyncClient() as client:
            while poll_count < max_polls:
                status_url = f"{base_url}/api/v1/dags/{dag_id}/dagRuns/{dag_run_id}"
                
                response = await client.get(
                    status_url,
                    auth=(config.get("username"), config.get("password"))
                )
                response.raise_for_status()
                
                dag_run = response.json()
                state = dag_run.get("state")
                
                if state == "success":
                    return {"status": "completed", "dag_run": dag_run}
                elif state == "failed":
                    raise Exception(f"Airflow DAG run failed: {dag_run}")
                
                await asyncio.sleep(5)
                poll_count += 1
        
        raise Exception("Airflow DAG run timeout")

class CustomIntegration:
    """Custom workflow integration."""
    
    async def execute_step(self, step_config: Dict[str, Any], input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a custom workflow step."""
        try:
            step_type = step_config["type"]
            
            if step_type == "http_request":
                return await self._execute_http_request(step_config["config"], input_data)
            elif step_type == "data_transformation":
                return await self._execute_data_transformation(step_config["config"], input_data)
            elif step_type == "condition":
                return await self._execute_condition(step_config["config"], input_data)
            else:
                raise ValueError(f"Unknown custom step type: {step_type}")
                
        except Exception as e:
            logger.error(f"Custom integration error: {str(e)}")
            raise
    
    async def _execute_http_request(self, config: Dict[str, Any], input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute HTTP request step."""
        async with httpx.AsyncClient() as client:
            response = await client.request(
                method=config.get("method", "POST"),
                url=config["url"],
                json=input_data,
                headers=config.get("headers", {}),
                timeout=config.get("timeout", 30.0)
            )
            response.raise_for_status()
            return response.json()
    
    async def _execute_data_transformation(self, config: Dict[str, Any], input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute data transformation step."""
        # Simple transformation logic
        transformation_rules = config.get("rules", [])
        result = input_data.copy()
        
        for rule in transformation_rules:
            if rule["type"] == "map_field":
                result[rule["target"]] = input_data.get(rule["source"])
            elif rule["type"] == "filter":
                # Apply filter logic
                pass
        
        return result
    
    async def _execute_condition(self, config: Dict[str, Any], input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute conditional step."""
        condition = config["condition"]
        # Simple condition evaluation (in production, use a proper expression evaluator)
        if eval(condition, {"data": input_data}):
            return {"condition_met": True, "data": input_data}
        else:
            return {"condition_met": False, "data": input_data}

class WebhookManager:
    """Manages webhook configurations and callbacks."""
    
    async def create_webhook(self, workflow_id: str, webhook_config: Dict[str, Any]) -> str:
        """Create webhook configuration for workflow."""
        db = SessionLocal()
        try:
            webhook_id = str(uuid.uuid4())
            
            webhook = WebhookConfig(
                id=webhook_id,
                workflow_id=workflow_id,
                url=webhook_config["url"],
                method=webhook_config.get("method", "POST"),
                headers=webhook_config.get("headers", {}),
                secret=webhook_config.get("secret")
            )
            
            db.add(webhook)
            db.commit()
            
            return webhook_id
            
        except Exception as e:
            logger.error(f"Error creating webhook: {str(e)}")
            raise
        finally:
            db.close()
    
    async def handle_webhook_callback(self, webhook_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle incoming webhook callback."""
        db = SessionLocal()
        try:
            webhook = db.query(WebhookConfig).filter(WebhookConfig.id == webhook_id).first()
            if not webhook or not webhook.is_active:
                raise ValueError("Webhook not found or inactive")
            
            # Trigger workflow execution
            workflow_engine = WorkflowEngine()
            execution_id = await workflow_engine.execute_workflow(webhook.workflow_id, payload)
            
            return {"execution_id": execution_id, "status": "triggered"}
            
        except Exception as e:
            logger.error(f"Error handling webhook callback: {str(e)}")
            raise
        finally:
            db.close()

class RetryManager:
    """Manages retry logic for workflow steps."""
    
    async def execute_with_retry(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with retry logic."""
        max_retries = 3
        retry_delay = 1  # seconds
        
        for attempt in range(max_retries + 1):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                if attempt == max_retries:
                    raise
                
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}. Retrying in {retry_delay} seconds...")
                await asyncio.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff