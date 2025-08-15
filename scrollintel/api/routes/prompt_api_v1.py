"""
Advanced Prompt Management API v1 - Enhanced API endpoints with versioning, rate limiting, and webhook support.
"""
from typing import List, Optional, Dict, Any, Union
from fastapi import APIRouter, Depends, HTTPException, Request, BackgroundTasks, Header
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field
from datetime import datetime
import asyncio
import json
import uuid

from ...models.database_utils import get_db
from ...core.prompt_manager import PromptManager, SearchQuery, PromptChanges
from ...core.prompt_import_export import PromptImportExport
from ...models.prompt_models import AdvancedPromptTemplate, AdvancedPromptVersion
from ...core.rate_limiter import AdvancedRateLimiter, UsageMonitor
from ...core.webhook_system import webhook_manager, trigger_webhook_event, WebhookEventType
from .prompt_routes import (
    PromptTemplateCreate, PromptTemplateUpdate, PromptTemplateResponse,
    PromptVersionResponse, PromptSearchRequest, VariableSubstitutionRequest
)


# Enhanced API models with additional metadata
class APIResponse(BaseModel):
    """Standard API response wrapper."""
    success: bool
    data: Optional[Any] = None
    message: Optional[str] = None
    timestamp: datetime
    version: str = "1.0"
    request_id: Optional[str] = None
    rate_limit: Optional[Dict[str, Any]] = None


class PaginatedResponse(BaseModel):
    """Paginated response wrapper."""
    items: List[Any]
    total: int
    page: int
    page_size: int
    has_next: bool
    has_previous: bool


class PromptUsageMetrics(BaseModel):
    """Prompt usage metrics."""
    prompt_id: str
    total_uses: int
    unique_users: int
    avg_response_time: float
    success_rate: float
    last_used: Optional[datetime]


class BatchOperation(BaseModel):
    """Batch operation request."""
    type: str = Field(..., description="Operation type: create, update, delete")
    prompt_id: Optional[str] = Field(None, description="Prompt ID for update/delete operations")
    data: Optional[Dict[str, Any]] = Field(None, description="Operation data")


class WebhookConfig(BaseModel):
    """Webhook configuration."""
    url: str = Field(..., description="Webhook URL")
    events: List[str] = Field(..., description="List of events to subscribe to")
    secret: Optional[str] = Field(None, description="Secret for signature verification")
    active: bool = Field(True, description="Whether webhook is active")


class RateLimitInfo(BaseModel):
    """Rate limit information."""
    limit: int
    remaining: int
    reset_time: datetime
    retry_after: Optional[int] = None


# Create versioned router
router = APIRouter(prefix="/api/v1/prompts", tags=["Prompts API v1"])


def get_prompt_manager(db: Session = Depends(get_db)) -> PromptManager:
    """Get PromptManager instance."""
    return PromptManager(db)


def get_import_export(prompt_manager: PromptManager = Depends(get_prompt_manager)) -> PromptImportExport:
    """Get PromptImportExport instance."""
    return PromptImportExport(prompt_manager)


def get_rate_limiter(db: Session = Depends(get_db)) -> AdvancedRateLimiter:
    """Get rate limiter instance."""
    return AdvancedRateLimiter(db_session=db)


def get_usage_monitor(db: Session = Depends(get_db)) -> UsageMonitor:
    """Get usage monitor instance."""
    return UsageMonitor(db_session=db)


async def check_rate_limits(
    request: Request,
    rate_limiter: AdvancedRateLimiter = Depends(get_rate_limiter)
) -> Dict[str, Any]:
    """Check rate limits for the request."""
    # Extract user info from request
    user_id = getattr(request.state, "user_id", None)
    api_key_id = getattr(request.state, "api_key_id", None)
    ip_address = request.client.host if request.client else None
    endpoint = request.url.path
    
    # Check rate limits
    allowed, violated_rule, retry_after = await rate_limiter.check_rate_limit(
        user_id=user_id,
        api_key_id=api_key_id,
        ip_address=ip_address,
        endpoint=endpoint
    )
    
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded: {violated_rule.name}",
            headers={"Retry-After": str(retry_after)}
        )
    
    # Record the request
    await rate_limiter.record_request(
        user_id=user_id,
        api_key_id=api_key_id,
        ip_address=ip_address,
        endpoint=endpoint
    )
    
    # Get current usage stats
    if user_id:
        from ...core.rate_limiter import RateLimitScope
        stats = await rate_limiter.get_usage_stats(RateLimitScope.USER, user_id, endpoint)
        return stats
    
    return {}


async def log_request_usage(
    request: Request,
    response_time: float,
    status_code: int,
    usage_monitor: UsageMonitor = Depends(get_usage_monitor)
):
    """Log request usage for monitoring."""
    user_id = getattr(request.state, "user_id", None)
    api_key_id = getattr(request.state, "api_key_id", None)
    ip_address = request.client.host if request.client else None
    endpoint = request.url.path
    method = request.method
    
    await usage_monitor.log_request(
        user_id=user_id,
        api_key_id=api_key_id,
        ip_address=ip_address,
        endpoint=endpoint,
        method=method,
        status_code=status_code,
        response_time=response_time
    )


@router.post("/", response_model=APIResponse)
async def create_prompt_v1(
    prompt: PromptTemplateCreate,
    request: Request,
    background_tasks: BackgroundTasks,
    rate_limit_info: Dict[str, Any] = Depends(check_rate_limits),
    prompt_manager: PromptManager = Depends(get_prompt_manager),
    usage_monitor: UsageMonitor = Depends(get_usage_monitor)
):
    """Create a new prompt template with enhanced API response."""
    start_time = datetime.utcnow()
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    
    try:
        # Get user context (simplified for now)
        user_id = getattr(request.state, "user_id", "system")
        
        variables_dict = [var.dict() for var in prompt.variables] if prompt.variables else []
        
        prompt_id = prompt_manager.create_prompt(
            name=prompt.name,
            content=prompt.content,
            category=prompt.category,
            created_by=user_id,
            tags=prompt.tags,
            variables=variables_dict,
            description=prompt.description
        )
        
        # Trigger webhook
        await trigger_webhook_event(
            event_type=WebhookEventType.PROMPT_CREATED,
            resource_type="prompt",
            resource_id=prompt_id,
            action="create",
            user_id=user_id,
            data={"name": prompt.name, "category": prompt.category}
        )
        
        # Log usage
        response_time = (datetime.utcnow() - start_time).total_seconds()
        background_tasks.add_task(
            log_request_usage,
            request,
            response_time,
            200,
            usage_monitor
        )
        
        return APIResponse(
            success=True,
            data={"id": prompt_id},
            message="Prompt created successfully",
            timestamp=datetime.utcnow(),
            request_id=request_id,
            rate_limit=rate_limit_info
        )
    except Exception as e:
        # Log error usage
        response_time = (datetime.utcnow() - start_time).total_seconds()
        background_tasks.add_task(
            log_request_usage,
            request,
            response_time,
            400,
            usage_monitor
        )
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/{prompt_id}", response_model=APIResponse)
async def get_prompt_v1(
    prompt_id: str,
    request: Request,
    context: SecurityContext = Depends(require_resource_access(ResourceType.PROMPT, Action.READ)),
    prompt_manager: PromptManager = Depends(get_prompt_manager)
):
    """Get a prompt template by ID with enhanced API response."""
    prompt = prompt_manager.get_prompt(prompt_id)
    if not prompt:
        raise HTTPException(status_code=404, detail="Prompt not found")
    
    return APIResponse(
        success=True,
        data=PromptTemplateResponse.from_orm(prompt).dict(),
        message="Prompt retrieved successfully",
        timestamp=datetime.utcnow(),
        request_id=getattr(request.state, "request_id", None)
    )


@router.put("/{prompt_id}", response_model=APIResponse)
async def update_prompt_v1(
    prompt_id: str,
    updates: PromptTemplateUpdate,
    request: Request,
    background_tasks: BackgroundTasks,
    context: SecurityContext = Depends(require_resource_access(ResourceType.PROMPT, Action.UPDATE)),
    prompt_manager: PromptManager = Depends(get_prompt_manager)
):
    """Update a prompt template with enhanced API response."""
    try:
        variables_dict = None
        if updates.variables is not None:
            variables_dict = [var.dict() for var in updates.variables]
        
        changes = PromptChanges(
            name=updates.name,
            content=updates.content,
            category=updates.category,
            tags=updates.tags,
            variables=variables_dict,
            description=updates.description,
            changes_description=updates.changes_description
        )
        
        new_version = prompt_manager.update_prompt(prompt_id, changes, context.user_id)
        
        # Trigger webhook
        webhook_event = WebhookEvent(
            event_type="prompt.updated",
            resource_type="prompt",
            resource_id=prompt_id,
            action="update",
            timestamp=datetime.utcnow(),
            user_id=context.user_id,
            data={"version": new_version.version, "changes": updates.changes_description}
        )
        await trigger_webhook(webhook_event, background_tasks)
        
        return APIResponse(
            success=True,
            data=PromptVersionResponse.from_orm(new_version).dict(),
            message="Prompt updated successfully",
            timestamp=datetime.utcnow(),
            request_id=getattr(request.state, "request_id", None)
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/{prompt_id}", response_model=APIResponse)
async def delete_prompt_v1(
    prompt_id: str,
    request: Request,
    background_tasks: BackgroundTasks,
    context: SecurityContext = Depends(require_resource_access(ResourceType.PROMPT, Action.DELETE)),
    prompt_manager: PromptManager = Depends(get_prompt_manager)
):
    """Delete (deactivate) a prompt template with enhanced API response."""
    success = prompt_manager.delete_prompt(prompt_id)
    if not success:
        raise HTTPException(status_code=404, detail="Prompt not found")
    
    # Trigger webhook
    webhook_event = WebhookEvent(
        event_type="prompt.deleted",
        resource_type="prompt",
        resource_id=prompt_id,
        action="delete",
        timestamp=datetime.utcnow(),
        user_id=context.user_id,
        data={}
    )
    await trigger_webhook(webhook_event, background_tasks)
    
    return APIResponse(
        success=True,
        data=None,
        message="Prompt deleted successfully",
        timestamp=datetime.utcnow(),
        request_id=getattr(request.state, "request_id", None)
    )


@router.post("/search", response_model=APIResponse)
async def search_prompts_v1(
    search_request: PromptSearchRequest,
    request: Request,
    context: SecurityContext = Depends(require_resource_access(ResourceType.PROMPT, Action.READ)),
    prompt_manager: PromptManager = Depends(get_prompt_manager)
):
    """Search prompt templates with enhanced pagination."""
    query = SearchQuery(
        text=search_request.text,
        category=search_request.category,
        tags=search_request.tags,
        created_by=search_request.created_by,
        date_from=search_request.date_from,
        date_to=search_request.date_to,
        limit=search_request.limit,
        offset=search_request.offset
    )
    
    prompts = prompt_manager.search_prompts(query)
    
    # Calculate pagination info
    total_count = len(prompts)  # This would be more efficient with a count query
    page = (search_request.offset // search_request.limit) + 1
    has_next = len(prompts) == search_request.limit
    has_previous = search_request.offset > 0
    
    paginated_response = PaginatedResponse(
        items=[PromptTemplateResponse.from_orm(prompt).dict() for prompt in prompts],
        total=total_count,
        page=page,
        page_size=search_request.limit,
        has_next=has_next,
        has_previous=has_previous
    )
    
    return APIResponse(
        success=True,
        data=paginated_response.dict(),
        message="Search completed successfully",
        timestamp=datetime.utcnow(),
        request_id=getattr(request.state, "request_id", None)
    )


@router.get("/{prompt_id}/history", response_model=APIResponse)
async def get_prompt_history_v1(
    prompt_id: str,
    request: Request,
    context: SecurityContext = Depends(require_resource_access(ResourceType.PROMPT, Action.READ)),
    prompt_manager: PromptManager = Depends(get_prompt_manager)
):
    """Get version history for a prompt template with enhanced API response."""
    versions = prompt_manager.get_prompt_history(prompt_id)
    
    return APIResponse(
        success=True,
        data=[PromptVersionResponse.from_orm(version).dict() for version in versions],
        message="History retrieved successfully",
        timestamp=datetime.utcnow(),
        request_id=getattr(request.state, "request_id", None)
    )


@router.get("/{prompt_id}/metrics", response_model=APIResponse)
async def get_prompt_metrics_v1(
    prompt_id: str,
    request: Request,
    context: SecurityContext = Depends(require_resource_access(ResourceType.PROMPT, Action.READ)),
    prompt_manager: PromptManager = Depends(get_prompt_manager)
):
    """Get usage metrics for a prompt template."""
    # TODO: Implement actual metrics collection
    # This would typically involve querying usage logs, performance data, etc.
    
    metrics = PromptUsageMetrics(
        prompt_id=prompt_id,
        total_uses=0,
        unique_users=0,
        avg_response_time=0.0,
        success_rate=0.0,
        last_used=None
    )
    
    return APIResponse(
        success=True,
        data=metrics.dict(),
        message="Metrics retrieved successfully",
        timestamp=datetime.utcnow(),
        request_id=getattr(request.state, "request_id", None)
    )


@router.post("/batch", response_model=APIResponse)
async def batch_operations_v1(
    operations: List[Dict[str, Any]],
    request: Request,
    background_tasks: BackgroundTasks,
    context: SecurityContext = Depends(require_resource_access(ResourceType.PROMPT, Action.UPDATE)),
    prompt_manager: PromptManager = Depends(get_prompt_manager)
):
    """Perform batch operations on multiple prompts."""
    results = []
    errors = []
    
    for i, operation in enumerate(operations):
        try:
            op_type = operation.get("type")
            prompt_id = operation.get("prompt_id")
            
            if op_type == "delete":
                success = prompt_manager.delete_prompt(prompt_id)
                results.append({"operation": i, "type": op_type, "prompt_id": prompt_id, "success": success})
            elif op_type == "update":
                # Handle batch updates
                changes_data = operation.get("changes", {})
                changes = PromptChanges(**changes_data)
                new_version = prompt_manager.update_prompt(prompt_id, changes, context.user_id)
                results.append({"operation": i, "type": op_type, "prompt_id": prompt_id, "version": new_version.version})
            else:
                errors.append({"operation": i, "error": f"Unknown operation type: {op_type}"})
                
        except Exception as e:
            errors.append({"operation": i, "error": str(e)})
    
    return APIResponse(
        success=len(errors) == 0,
        data={"results": results, "errors": errors},
        message=f"Batch operation completed. {len(results)} successful, {len(errors)} errors",
        timestamp=datetime.utcnow(),
        request_id=getattr(request.state, "request_id", None)
    )


@router.get("/", response_model=APIResponse)
async def list_prompts_v1(
    request: Request,
    page: int = 1,
    page_size: int = 50,
    category: Optional[str] = None,
    tags: Optional[str] = None,
    context: SecurityContext = Depends(require_resource_access(ResourceType.PROMPT, Action.READ)),
    prompt_manager: PromptManager = Depends(get_prompt_manager)
):
    """List prompts with pagination and filtering."""
    offset = (page - 1) * page_size
    tag_list = tags.split(",") if tags else None
    
    query = SearchQuery(
        category=category,
        tags=tag_list,
        limit=page_size,
        offset=offset
    )
    
    prompts = prompt_manager.search_prompts(query)
    
    paginated_response = PaginatedResponse(
        items=[PromptTemplateResponse.from_orm(prompt).dict() for prompt in prompts],
        total=len(prompts),  # This would be more efficient with a count query
        page=page,
        page_size=page_size,
        has_next=len(prompts) == page_size,
        has_previous=page > 1
    )
    
    return APIResponse(
        success=True,
        data=paginated_response.dict(),
        message="Prompts listed successfully",
        timestamp=datetime.utcnow(),
        request_id=getattr(request.state, "request_id", None)
    )


# Webhook management endpoints
@router.post("/webhooks", response_model=APIResponse)
async def register_webhook_v1(
    webhook_config: WebhookConfig,
    request: Request,
    rate_limit_info: Dict[str, Any] = Depends(check_rate_limits)
):
    """Register a new webhook endpoint."""
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    user_id = getattr(request.state, "user_id", "system")
    
    try:
        webhook_id = await webhook_manager.register_endpoint(
            user_id=user_id,
            name=f"webhook_{int(datetime.utcnow().timestamp())}",
            url=webhook_config.url,
            events=webhook_config.events,
            secret=webhook_config.secret
        )
        
        return APIResponse(
            success=True,
            data={"webhook_id": webhook_id},
            message="Webhook registered successfully",
            timestamp=datetime.utcnow(),
            request_id=request_id,
            rate_limit=rate_limit_info
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/webhooks", response_model=APIResponse)
async def list_webhooks_v1(
    request: Request,
    rate_limit_info: Dict[str, Any] = Depends(check_rate_limits)
):
    """List all webhook endpoints for the user."""
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    user_id = getattr(request.state, "user_id", "system")
    
    try:
        webhooks = await webhook_manager.get_endpoints(user_id)
        
        return APIResponse(
            success=True,
            data=webhooks,
            message="Webhooks retrieved successfully",
            timestamp=datetime.utcnow(),
            request_id=request_id,
            rate_limit=rate_limit_info
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.put("/webhooks/{webhook_id}", response_model=APIResponse)
async def update_webhook_v1(
    webhook_id: str,
    webhook_config: WebhookConfig,
    request: Request,
    rate_limit_info: Dict[str, Any] = Depends(check_rate_limits)
):
    """Update a webhook endpoint."""
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    user_id = getattr(request.state, "user_id", "system")
    
    try:
        success = await webhook_manager.update_endpoint(
            endpoint_id=webhook_id,
            user_id=user_id,
            url=webhook_config.url,
            events=webhook_config.events,
            secret=webhook_config.secret,
            active=webhook_config.active
        )
        
        if not success:
            raise HTTPException(status_code=404, detail="Webhook not found")
        
        return APIResponse(
            success=True,
            data=None,
            message="Webhook updated successfully",
            timestamp=datetime.utcnow(),
            request_id=request_id,
            rate_limit=rate_limit_info
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/webhooks/{webhook_id}", response_model=APIResponse)
async def delete_webhook_v1(
    webhook_id: str,
    request: Request,
    rate_limit_info: Dict[str, Any] = Depends(check_rate_limits)
):
    """Delete a webhook endpoint."""
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    user_id = getattr(request.state, "user_id", "system")
    
    try:
        success = await webhook_manager.delete_endpoint(webhook_id, user_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Webhook not found")
        
        return APIResponse(
            success=True,
            data=None,
            message="Webhook deleted successfully",
            timestamp=datetime.utcnow(),
            request_id=request_id,
            rate_limit=rate_limit_info
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/webhooks/{webhook_id}/test", response_model=APIResponse)
async def test_webhook_v1(
    webhook_id: str,
    request: Request,
    rate_limit_info: Dict[str, Any] = Depends(check_rate_limits)
):
    """Test a webhook endpoint."""
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    user_id = getattr(request.state, "user_id", "system")
    
    try:
        result = await webhook_manager.test_endpoint(webhook_id, user_id)
        
        return APIResponse(
            success=result["success"],
            data=result,
            message="Webhook test completed",
            timestamp=datetime.utcnow(),
            request_id=request_id,
            rate_limit=rate_limit_info
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# Usage analytics endpoints
@router.get("/usage/summary", response_model=APIResponse)
async def get_usage_summary_v1(
    request: Request,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    rate_limit_info: Dict[str, Any] = Depends(check_rate_limits),
    usage_monitor: UsageMonitor = Depends(get_usage_monitor)
):
    """Get usage summary for the current user."""
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    user_id = getattr(request.state, "user_id", "system")
    
    try:
        from ...core.rate_limiter import RateLimitScope
        summary = await usage_monitor.get_usage_summary(
            scope=RateLimitScope.USER,
            identifier=user_id,
            start_date=start_date,
            end_date=end_date
        )
        
        return APIResponse(
            success=True,
            data=summary,
            message="Usage summary retrieved successfully",
            timestamp=datetime.utcnow(),
            request_id=request_id,
            rate_limit=rate_limit_info
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))