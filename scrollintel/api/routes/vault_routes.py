"""
FastAPI routes for ScrollIntel Vault operations.
Handles secure insight storage, retrieval, and management.
"""

from datetime import datetime
from typing import Dict, Any, List, Optional
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer
from sqlalchemy.orm import Session

from ...models.database_utils import get_db
from ...models.schemas import (
    VaultInsightCreate, VaultInsightUpdate, VaultInsightResponse,
    VaultInsightSummary, VaultSearchQuery, VaultSearchResponse,
    VaultAccessLogResponse, VaultStatsResponse, PaginationParams
)
from ...security.auth import get_current_user
from ...security.middleware import require_admin
from ...engines.vault_engine import ScrollVaultEngine
from ...core.registry import get_engine_registry
from ...models.database import User

router = APIRouter(prefix="/vault", tags=["vault"])
security = HTTPBearer()


def get_vault_engine() -> ScrollVaultEngine:
    """Get the vault engine instance."""
    registry = get_engine_registry()
    engine = registry.get_engine("scroll-vault-engine")
    if not engine:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vault engine not available"
        )
    return engine


@router.post("/insights", response_model=Dict[str, Any])
async def store_insight(
    insight_data: VaultInsightCreate,
    request: Request,
    current_user: User = Depends(get_current_user),
    vault_engine: ScrollVaultEngine = Depends(get_vault_engine),
    db: Session = Depends(get_db)
):
    """Store a new insight in the vault."""
    try:
        # Prepare insight data
        input_data = {
            "title": insight_data.title,
            "content": insight_data.content,
            "type": insight_data.insight_type,
            "access_level": insight_data.access_level,
            "retention_policy": insight_data.retention_policy,
            "tags": insight_data.tags,
            "metadata": insight_data.metadata
        }
        
        # Process through vault engine
        result = await vault_engine.process(
            input_data=input_data,
            parameters={
                "operation": "store_insight",
                "user_id": str(current_user.id),
                "organization_id": insight_data.organization_id,
                "ip_address": request.client.host,
                "parent_id": str(insight_data.parent_id) if insight_data.parent_id else None
            }
        )
        
        return result
        
    except PermissionError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e)
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to store insight: {str(e)}"
        )


@router.get("/insights/{insight_id}", response_model=Dict[str, Any])
async def get_insight(
    insight_id: UUID,
    request: Request,
    current_user: User = Depends(get_current_user),
    vault_engine: ScrollVaultEngine = Depends(get_vault_engine),
    db: Session = Depends(get_db)
):
    """Retrieve and decrypt an insight."""
    try:
        result = await vault_engine.process(
            input_data=None,
            parameters={
                "operation": "retrieve_insight",
                "insight_id": str(insight_id),
                "user_id": str(current_user.id),
                "ip_address": request.client.host
            }
        )
        
        return result
        
    except PermissionError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e)
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve insight: {str(e)}"
        )


@router.put("/insights/{insight_id}", response_model=Dict[str, Any])
async def update_insight(
    insight_id: UUID,
    update_data: VaultInsightUpdate,
    request: Request,
    current_user: User = Depends(get_current_user),
    vault_engine: ScrollVaultEngine = Depends(get_vault_engine),
    db: Session = Depends(get_db)
):
    """Update an existing insight with version control."""
    try:
        # Prepare update data
        input_data = {}
        if update_data.title is not None:
            input_data["title"] = update_data.title
        if update_data.content is not None:
            input_data["content"] = update_data.content
        if update_data.tags is not None:
            input_data["tags"] = update_data.tags
        if update_data.metadata is not None:
            input_data["metadata"] = update_data.metadata
        if update_data.access_level is not None:
            input_data["access_level"] = update_data.access_level
        if update_data.retention_policy is not None:
            input_data["retention_policy"] = update_data.retention_policy
        
        result = await vault_engine.process(
            input_data=input_data,
            parameters={
                "operation": "update_insight",
                "insight_id": str(insight_id),
                "user_id": str(current_user.id),
                "ip_address": request.client.host
            }
        )
        
        return result
        
    except PermissionError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e)
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update insight: {str(e)}"
        )


@router.delete("/insights/{insight_id}", response_model=Dict[str, Any])
async def delete_insight(
    insight_id: UUID,
    request: Request,
    current_user: User = Depends(get_current_user),
    vault_engine: ScrollVaultEngine = Depends(get_vault_engine),
    db: Session = Depends(get_db)
):
    """Delete an insight."""
    try:
        result = await vault_engine.process(
            input_data=None,
            parameters={
                "operation": "delete_insight",
                "insight_id": str(insight_id),
                "user_id": str(current_user.id),
                "ip_address": request.client.host
            }
        )
        
        return result
        
    except PermissionError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e)
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete insight: {str(e)}"
        )


@router.post("/search", response_model=Dict[str, Any])
async def search_insights(
    search_query: VaultSearchQuery,
    request: Request,
    current_user: User = Depends(get_current_user),
    vault_engine: ScrollVaultEngine = Depends(get_vault_engine),
    db: Session = Depends(get_db)
):
    """Search insights using semantic search and filters."""
    try:
        # Prepare search data
        search_data = {
            "query": search_query.query,
            "filters": search_query.filters,
            "access_levels": search_query.access_levels,
            "insight_types": search_query.insight_types,
            "date_range": search_query.date_range,
            "limit": search_query.limit,
            "offset": search_query.offset
        }
        
        # Add tag filter if provided
        if search_query.tags:
            search_data["filters"]["tags"] = search_query.tags
        
        # Add creator filter if provided
        if search_query.creator_id:
            search_data["filters"]["creator_id"] = str(search_query.creator_id)
        
        result = await vault_engine.process(
            input_data=search_data,
            parameters={
                "operation": "search_insights",
                "user_id": str(current_user.id),
                "ip_address": request.client.host
            }
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}"
        )


@router.get("/insights/{insight_id}/history", response_model=Dict[str, Any])
async def get_insight_history(
    insight_id: UUID,
    current_user: User = Depends(get_current_user),
    vault_engine: ScrollVaultEngine = Depends(get_vault_engine),
    db: Session = Depends(get_db)
):
    """Get version history for an insight."""
    try:
        result = await vault_engine.process(
            input_data=None,
            parameters={
                "operation": "get_insight_history",
                "insight_id": str(insight_id),
                "user_id": str(current_user.id)
            }
        )
        
        return result
        
    except PermissionError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e)
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get insight history: {str(e)}"
        )


@router.get("/audit", response_model=Dict[str, Any])
async def get_access_audit(
    insight_id: Optional[UUID] = None,
    action: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    current_user: User = Depends(get_current_user),
    vault_engine: ScrollVaultEngine = Depends(get_vault_engine),
    db: Session = Depends(get_db)
):
    """Get access audit logs for insights."""
    try:
        params = {
            "operation": "audit_access",
            "user_id": str(current_user.id),
            "limit": limit,
            "offset": offset
        }
        
        if insight_id:
            params["insight_id"] = str(insight_id)
        if action:
            params["action"] = action
        
        result = await vault_engine.process(
            input_data=None,
            parameters=params
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get audit logs: {str(e)}"
        )


@router.post("/cleanup", response_model=Dict[str, Any])
async def cleanup_expired_insights(
    current_user: User = Depends(get_current_user),
    vault_engine: ScrollVaultEngine = Depends(get_vault_engine),
    db: Session = Depends(get_db),
    _: None = Depends(require_admin)
):
    """Clean up expired insights (admin only)."""
    try:
        result = await vault_engine.process(
            input_data=None,
            parameters={
                "operation": "cleanup_expired",
                "user_id": str(current_user.id)
            }
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Cleanup failed: {str(e)}"
        )


@router.get("/stats", response_model=Dict[str, Any])
async def get_vault_stats(
    current_user: User = Depends(get_current_user),
    vault_engine: ScrollVaultEngine = Depends(get_vault_engine),
    db: Session = Depends(get_db)
):
    """Get vault statistics and status."""
    try:
        # Get engine status
        status_info = vault_engine.get_status()
        
        # Get recent audit logs
        audit_result = await vault_engine.process(
            input_data=None,
            parameters={
                "operation": "audit_access",
                "user_id": str(current_user.id),
                "limit": 10,
                "offset": 0
            }
        )
        
        return {
            "success": True,
            "vault_status": status_info,
            "recent_activity": audit_result.get("audit_logs", []),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get vault stats: {str(e)}"
        )