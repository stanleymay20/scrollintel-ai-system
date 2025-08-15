"""
Admin routes for ScrollIntel API.
Handles system administration, user management, and monitoring.
"""

import time
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

from fastapi import APIRouter, HTTPException, status, Depends, Query
from pydantic import BaseModel, EmailStr, Field

from ...core.interfaces import SecurityContext, UserRole
from ...core.registry import AgentRegistry
from ...security.middleware import require_admin, require_permission
from ...security.permissions import Permission
from ...security.audit import audit_logger, AuditAction


# Request/Response models
class UserCreateRequest(BaseModel):
    """Request model for creating a user."""
    email: EmailStr
    password: str = Field(..., min_length=8)
    role: UserRole
    permissions: List[str] = []


class UserUpdateRequest(BaseModel):
    """Request model for updating a user."""
    email: Optional[EmailStr] = None
    role: Optional[UserRole] = None
    permissions: Optional[List[str]] = None
    is_active: Optional[bool] = None


class UserResponse(BaseModel):
    """Response model for user information."""
    id: str
    email: str
    role: str
    permissions: List[str]
    is_active: bool
    created_at: float
    last_login: Optional[float] = None


class SystemStatsResponse(BaseModel):
    """Response model for system statistics."""
    total_users: int
    active_users: int
    total_agents: int
    active_agents: int
    total_requests_today: int
    successful_requests_today: int
    error_rate_today: float
    uptime_seconds: float


class AuditLogResponse(BaseModel):
    """Response model for audit log entries."""
    id: str
    timestamp: float
    user_id: Optional[str]
    action: str
    resource_type: str
    resource_id: str
    ip_address: str
    success: bool
    error_message: Optional[str] = None
    details: Dict[str, Any] = {}


def create_admin_router(agent_registry: AgentRegistry) -> APIRouter:
    """Create admin router with dependencies."""
    
    router = APIRouter()
    
    # User Management Routes
    @router.get("/users", response_model=List[UserResponse])
    async def list_users(
        skip: int = Query(0, ge=0),
        limit: int = Query(100, ge=1, le=1000),
        context: SecurityContext = Depends(require_permission(Permission.USER_LIST))
    ):
        """List all users with pagination."""
        try:
            # TODO: Implement actual user listing from database
            # For now, return mock data
            mock_users = [
                UserResponse(
                    id="admin-user-id",
                    email="admin@scrollintel.com",
                    role=UserRole.ADMIN.value,
                    permissions=["*"],
                    is_active=True,
                    created_at=time.time() - 86400,  # 1 day ago
                    last_login=time.time() - 3600    # 1 hour ago
                )
            ]
            
            return mock_users[skip:skip + limit]
            
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to list users: {str(e)}"
            )
    
    @router.post("/users", response_model=UserResponse)
    async def create_user(
        request: UserCreateRequest,
        context: SecurityContext = Depends(require_permission(Permission.USER_CREATE))
    ):
        """Create a new user."""
        try:
            # TODO: Implement actual user creation with database
            # For now, return not implemented
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail="User creation is not yet implemented"
            )
            
        except HTTPException:
            raise
        except Exception as e:
            await audit_logger.log(
                action=AuditAction.USER_CREATE,
                resource_type="user",
                resource_id=request.email,
                user_id=context.user_id,
                session_id=context.session_id,
                details={
                    "email": request.email,
                    "role": request.role.value,
                    "permissions_count": len(request.permissions)
                },
                success=False,
                error_message=str(e)
            )
            
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to create user: {str(e)}"
            )
    
    @router.get("/users/{user_id}", response_model=UserResponse)
    async def get_user(
        user_id: str,
        context: SecurityContext = Depends(require_permission(Permission.USER_READ))
    ):
        """Get user information by ID."""
        try:
            # TODO: Implement actual user retrieval from database
            # For now, return mock data for admin user
            if user_id == "admin-user-id":
                return UserResponse(
                    id=user_id,
                    email="admin@scrollintel.com",
                    role=UserRole.ADMIN.value,
                    permissions=["*"],
                    is_active=True,
                    created_at=time.time() - 86400,
                    last_login=time.time() - 3600
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"User with ID {user_id} not found"
                )
                
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to get user: {str(e)}"
            )
    
    @router.put("/users/{user_id}", response_model=UserResponse)
    async def update_user(
        user_id: str,
        request: UserUpdateRequest,
        context: SecurityContext = Depends(require_permission(Permission.USER_UPDATE))
    ):
        """Update user information."""
        try:
            # TODO: Implement actual user update with database
            # For now, return not implemented
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail="User update is not yet implemented"
            )
            
        except HTTPException:
            raise
        except Exception as e:
            await audit_logger.log(
                action=AuditAction.USER_UPDATE,
                resource_type="user",
                resource_id=user_id,
                user_id=context.user_id,
                session_id=context.session_id,
                details={"updated_fields": list(request.dict(exclude_unset=True).keys())},
                success=False,
                error_message=str(e)
            )
            
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to update user: {str(e)}"
            )
    
    @router.delete("/users/{user_id}")
    async def delete_user(
        user_id: str,
        context: SecurityContext = Depends(require_permission(Permission.USER_DELETE))
    ):
        """Delete a user."""
        try:
            # Prevent self-deletion
            if user_id == context.user_id:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Cannot delete your own account"
                )
            
            # TODO: Implement actual user deletion with database
            # For now, return not implemented
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail="User deletion is not yet implemented"
            )
            
        except HTTPException:
            raise
        except Exception as e:
            await audit_logger.log(
                action=AuditAction.USER_DELETE,
                resource_type="user",
                resource_id=user_id,
                user_id=context.user_id,
                session_id=context.session_id,
                details={},
                success=False,
                error_message=str(e)
            )
            
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to delete user: {str(e)}"
            )
    
    # System Statistics Routes
    @router.get("/stats", response_model=SystemStatsResponse)
    async def get_system_stats(
        context: SecurityContext = Depends(require_permission(Permission.SYSTEM_METRICS))
    ):
        """Get system statistics and metrics."""
        try:
            registry_status = agent_registry.get_registry_status()
            
            # TODO: Implement actual statistics from database
            # For now, return mock data
            return SystemStatsResponse(
                total_users=1,
                active_users=1,
                total_agents=registry_status["total_agents"],
                active_agents=registry_status["agent_status"].get("active", 0),
                total_requests_today=150,
                successful_requests_today=145,
                error_rate_today=3.33,  # (5/150) * 100
                uptime_seconds=86400  # 1 day
            )
            
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to get system statistics: {str(e)}"
            )
    
    # Audit Log Routes
    @router.get("/audit-logs", response_model=List[AuditLogResponse])
    async def get_audit_logs(
        skip: int = Query(0, ge=0),
        limit: int = Query(100, ge=1, le=1000),
        user_id: Optional[str] = Query(None),
        action: Optional[str] = Query(None),
        start_date: Optional[datetime] = Query(None),
        end_date: Optional[datetime] = Query(None),
        context: SecurityContext = Depends(require_permission(Permission.AUDIT_READ))
    ):
        """Get audit logs with filtering and pagination."""
        try:
            # TODO: Implement actual audit log retrieval from database
            # For now, return mock data
            mock_logs = [
                AuditLogResponse(
                    id="audit-1",
                    timestamp=time.time() - 3600,
                    user_id="admin-user-id",
                    action="user.login",
                    resource_type="auth",
                    resource_id="admin-user-id",
                    ip_address="127.0.0.1",
                    success=True,
                    details={"email": "admin@scrollintel.com"}
                ),
                AuditLogResponse(
                    id="audit-2",
                    timestamp=time.time() - 1800,
                    user_id="admin-user-id",
                    action="agent.execute",
                    resource_type="agent",
                    resource_id="cto-agent-1",
                    ip_address="127.0.0.1",
                    success=True,
                    details={"prompt_length": 150, "execution_time": 2.5}
                )
            ]
            
            # Apply filters (mock implementation)
            filtered_logs = mock_logs
            if user_id:
                filtered_logs = [log for log in filtered_logs if log.user_id == user_id]
            if action:
                filtered_logs = [log for log in filtered_logs if log.action == action]
            
            return filtered_logs[skip:skip + limit]
            
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to get audit logs: {str(e)}"
            )
    
    # System Configuration Routes
    @router.get("/config")
    async def get_system_config(
        context: SecurityContext = Depends(require_permission(Permission.SYSTEM_CONFIG))
    ):
        """Get system configuration (non-sensitive parts)."""
        try:
            from ...core.config import get_config
            config = get_config()
            
            return {
                "environment": config.environment,
                "debug": config.debug,
                "api_host": config.system.api_host,
                "api_port": config.system.api_port,
                "max_concurrent_agents": config.system.max_concurrent_agents,
                "agent_timeout_seconds": config.system.agent_timeout_seconds,
                "upload_max_size_mb": config.system.upload_max_size_mb,
                "rate_limit_requests": config.security.rate_limit_requests,
                "session_timeout_minutes": config.security.session_timeout_minutes,
                "jwt_expiration_hours": config.security.jwt_expiration_hours
            }
            
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to get system configuration: {str(e)}"
            )
    
    # Agent Management Routes
    @router.post("/agents/{agent_id}/restart")
    async def restart_agent(
        agent_id: str,
        context: SecurityContext = Depends(require_permission(Permission.AGENT_UPDATE))
    ):
        """Restart a specific agent."""
        try:
            agent = agent_registry.get_agent(agent_id)
            
            if not agent:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Agent with ID {agent_id} not found"
                )
            
            # Stop and start the agent
            agent.stop()
            agent.start()
            
            await audit_logger.log(
                action=AuditAction.AGENT_RESTART,
                resource_type="agent",
                resource_id=agent_id,
                user_id=context.user_id,
                session_id=context.session_id,
                details={"agent_name": agent.name, "agent_type": agent.agent_type.value},
                success=True
            )
            
            return {
                "message": f"Agent {agent_id} restarted successfully",
                "agent_id": agent_id,
                "status": agent.status.value,
                "timestamp": time.time()
            }
            
        except HTTPException:
            raise
        except Exception as e:
            await audit_logger.log(
                action=AuditAction.AGENT_RESTART,
                resource_type="agent",
                resource_id=agent_id,
                user_id=context.user_id,
                session_id=context.session_id,
                details={},
                success=False,
                error_message=str(e)
            )
            
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to restart agent: {str(e)}"
            )
    
    @router.post("/system/health-check")
    async def perform_system_health_check(
        context: SecurityContext = Depends(require_permission(Permission.SYSTEM_HEALTH))
    ):
        """Perform comprehensive system health check."""
        try:
            # Check all agents
            agent_health = await agent_registry.health_check_all()
            
            # TODO: Check database, Redis, external services
            
            health_summary = {
                "timestamp": time.time(),
                "overall_healthy": all(agent_health.values()) if agent_health else True,
                "components": {
                    "agents": {
                        "healthy": all(agent_health.values()) if agent_health else True,
                        "total": len(agent_health),
                        "healthy_count": sum(1 for h in agent_health.values() if h),
                        "details": agent_health
                    },
                    "database": {"healthy": True, "response_time_ms": 10},  # Mock
                    "redis": {"healthy": True, "response_time_ms": 5},      # Mock
                    "external_services": {"healthy": True}                  # Mock
                }
            }
            
            await audit_logger.log(
                action=AuditAction.SYSTEM_HEALTH_CHECK,
                resource_type="system",
                resource_id="health-check",
                user_id=context.user_id,
                session_id=context.session_id,
                details=health_summary,
                success=True
            )
            
            return health_summary
            
        except Exception as e:
            await audit_logger.log(
                action=AuditAction.SYSTEM_HEALTH_CHECK,
                resource_type="system",
                resource_id="health-check",
                user_id=context.user_id,
                session_id=context.session_id,
                details={},
                success=False,
                error_message=str(e)
            )
            
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"System health check failed: {str(e)}"
            )
    
    return router