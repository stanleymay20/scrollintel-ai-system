"""
API routes for user management and role-based access control.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from pydantic import BaseModel, EmailStr, validator

from ...core.database import get_db
from ...core.user_management import UserManagementService
from ...core.interfaces import UserRole, SecurityError, ValidationError
from ...models.database import User
from ...models.user_management_models import APIKey, Workspace


# Pydantic models for request/response
class OrganizationCreate(BaseModel):
    name: str
    display_name: str
    description: Optional[str] = None
    domain: Optional[str] = None
    subscription_plan: str = "free"
    
    @validator('name')
    def validate_name(cls, v):
        if not v or len(v.strip()) < 2:
            raise ValueError('Organization name must be at least 2 characters')
        return v.strip().lower().replace(' ', '-')


class OrganizationUpdate(BaseModel):
    display_name: Optional[str] = None
    description: Optional[str] = None
    domain: Optional[str] = None
    logo_url: Optional[str] = None
    website: Optional[str] = None
    industry: Optional[str] = None
    size: Optional[str] = None
    settings: Optional[Dict] = None
    features: Optional[List[str]] = None


class UserInvite(BaseModel):
    email: EmailStr
    role: UserRole = UserRole.VIEWER
    permissions: Optional[List[str]] = None
    message: Optional[str] = None


class UserRoleUpdate(BaseModel):
    role: UserRole
    permissions: Optional[List[str]] = None


class WorkspaceCreate(BaseModel):
    name: str
    description: Optional[str] = None
    visibility: str = "private"
    
    @validator('visibility')
    def validate_visibility(cls, v):
        if v not in ["private", "organization", "public"]:
            raise ValueError('Visibility must be private, organization, or public')
        return v


class WorkspaceMemberAdd(BaseModel):
    user_id: str
    role: str = "member"
    permissions: Optional[List[str]] = None
    
    @validator('role')
    def validate_role(cls, v):
        if v not in ["owner", "admin", "member", "viewer"]:
            raise ValueError('Role must be owner, admin, member, or viewer')
        return v


class APIKeyCreate(BaseModel):
    name: str
    description: Optional[str] = None
    permissions: Optional[List[str]] = None
    expires_days: Optional[int] = None


class OrganizationResponse(BaseModel):
    id: str
    name: str
    display_name: str
    description: Optional[str]
    subscription_plan: str
    subscription_status: str
    max_users: int
    max_workspaces: int
    created_at: datetime
    
    class Config:
        from_attributes = True


class WorkspaceResponse(BaseModel):
    id: str
    name: str
    description: Optional[str]
    organization_id: str
    owner_id: str
    visibility: str
    created_at: datetime
    
    class Config:
        from_attributes = True


class UserResponse(BaseModel):
    user_id: str
    email: str
    full_name: Optional[str]
    role: str
    permissions: List[str]
    joined_at: datetime
    last_active: Optional[datetime]


class APIKeyResponse(BaseModel):
    id: str
    name: str
    description: Optional[str]
    key_prefix: str
    permissions: List[str]
    usage_count: int
    last_used: Optional[datetime]
    expires_at: Optional[datetime]
    created_at: datetime
    
    class Config:
        from_attributes = True


# Security
security = HTTPBearer()

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
) -> User:
    """Get current authenticated user."""
    user_service = UserManagementService(db)
    user = await user_service.validate_session(credentials.credentials)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired session token"
        )
    
    return user


# Router
router = APIRouter(prefix="/api/v1/user-management", tags=["User Management"])


# Organization Management Routes

@router.post("/organizations", response_model=OrganizationResponse)
async def create_organization(
    org_data: OrganizationCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create a new organization."""
    try:
        user_service = UserManagementService(db)
        organization = await user_service.create_organization(
            name=org_data.name,
            display_name=org_data.display_name,
            creator_user_id=str(current_user.id),
            description=org_data.description,
            domain=org_data.domain,
            subscription_plan=org_data.subscription_plan
        )
        
        return OrganizationResponse.from_orm(organization)
        
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to create organization")


@router.get("/organizations/{organization_id}", response_model=OrganizationResponse)
async def get_organization(
    organization_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get organization details."""
    try:
        user_service = UserManagementService(db)
        
        # Check if user has access to organization
        if not user_service._check_organization_permission(
            str(current_user.id), organization_id, "view_organization"
        ):
            raise HTTPException(status_code=403, detail="Access denied")
        
        organization = await user_service.get_organization(organization_id)
        if not organization:
            raise HTTPException(status_code=404, detail="Organization not found")
        
        return OrganizationResponse.from_orm(organization)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to get organization")


@router.put("/organizations/{organization_id}", response_model=OrganizationResponse)
async def update_organization(
    organization_id: str,
    updates: OrganizationUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update organization settings."""
    try:
        user_service = UserManagementService(db)
        organization = await user_service.update_organization(
            organization_id=organization_id,
            user_id=str(current_user.id),
            updates=updates.dict(exclude_unset=True)
        )
        
        return OrganizationResponse.from_orm(organization)
        
    except SecurityError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to update organization")


@router.get("/organizations/{organization_id}/users", response_model=List[UserResponse])
async def get_organization_users(
    organization_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get all users in an organization."""
    try:
        user_service = UserManagementService(db)
        
        # Check permissions
        if not user_service._check_organization_permission(
            str(current_user.id), organization_id, "view_users"
        ):
            raise HTTPException(status_code=403, detail="Access denied")
        
        users = await user_service.get_organization_users(organization_id)
        return [UserResponse(**user) for user in users]
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to get organization users")


# User Invitation Routes

@router.post("/organizations/{organization_id}/invitations")
async def invite_user(
    organization_id: str,
    invite_data: UserInvite,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Invite a user to join an organization."""
    try:
        user_service = UserManagementService(db)
        invitation = await user_service.invite_user(
            email=invite_data.email,
            organization_id=organization_id,
            invited_by=str(current_user.id),
            role=invite_data.role,
            permissions=invite_data.permissions,
            message=invite_data.message
        )
        
        return {
            "message": "Invitation sent successfully",
            "invitation_id": str(invitation.id),
            "expires_at": invitation.expires_at
        }
        
    except SecurityError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to send invitation")


@router.post("/invitations/{token}/accept")
async def accept_invitation(
    token: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Accept a user invitation."""
    try:
        user_service = UserManagementService(db)
        org_user = await user_service.accept_invitation(token, str(current_user.id))
        
        return {
            "message": "Invitation accepted successfully",
            "organization_id": str(org_user.organization_id),
            "role": org_user.role.value
        }
        
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to accept invitation")


# User Role Management Routes

@router.put("/organizations/{organization_id}/users/{user_id}/role")
async def update_user_role(
    organization_id: str,
    user_id: str,
    role_update: UserRoleUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update a user's role in an organization."""
    try:
        user_service = UserManagementService(db)
        org_user = await user_service.update_user_role(
            organization_id=organization_id,
            target_user_id=user_id,
            new_role=role_update.role,
            updated_by=str(current_user.id),
            permissions=role_update.permissions
        )
        
        return {
            "message": "User role updated successfully",
            "user_id": user_id,
            "new_role": org_user.role.value
        }
        
    except SecurityError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to update user role")


@router.delete("/organizations/{organization_id}/users/{user_id}")
async def remove_user_from_organization(
    organization_id: str,
    user_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Remove a user from an organization."""
    try:
        user_service = UserManagementService(db)
        success = await user_service.remove_user_from_organization(
            organization_id=organization_id,
            target_user_id=user_id,
            removed_by=str(current_user.id)
        )
        
        return {"message": "User removed from organization successfully"}
        
    except SecurityError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to remove user")


# Workspace Management Routes

@router.post("/organizations/{organization_id}/workspaces", response_model=WorkspaceResponse)
async def create_workspace(
    organization_id: str,
    workspace_data: WorkspaceCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create a new workspace."""
    try:
        user_service = UserManagementService(db)
        workspace = await user_service.create_workspace(
            name=workspace_data.name,
            organization_id=organization_id,
            owner_id=str(current_user.id),
            description=workspace_data.description,
            visibility=workspace_data.visibility
        )
        
        return WorkspaceResponse.from_orm(workspace)
        
    except SecurityError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to create workspace")


@router.get("/workspaces", response_model=List[WorkspaceResponse])
async def get_user_workspaces(
    organization_id: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get all workspaces accessible to the current user."""
    try:
        user_service = UserManagementService(db)
        workspaces = await user_service.get_user_workspaces(
            str(current_user.id), organization_id
        )
        
        return [WorkspaceResponse.from_orm(workspace) for workspace in workspaces]
        
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to get workspaces")


@router.get("/workspaces/{workspace_id}", response_model=WorkspaceResponse)
async def get_workspace(
    workspace_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get workspace details."""
    try:
        user_service = UserManagementService(db)
        
        # Check if user has access to workspace
        if not user_service._check_workspace_permission(
            str(current_user.id), workspace_id, "view_workspace"
        ):
            raise HTTPException(status_code=403, detail="Access denied")
        
        workspace = db.query(Workspace).filter(Workspace.id == workspace_id).first()
        if not workspace:
            raise HTTPException(status_code=404, detail="Workspace not found")
        
        return WorkspaceResponse.from_orm(workspace)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to get workspace")


@router.put("/workspaces/{workspace_id}", response_model=WorkspaceResponse)
async def update_workspace(
    workspace_id: str,
    updates: WorkspaceCreate,  # Reusing the same model for updates
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update workspace settings."""
    try:
        user_service = UserManagementService(db)
        workspace = await user_service.update_workspace(
            workspace_id=workspace_id,
            user_id=str(current_user.id),
            updates=updates.dict(exclude_unset=True)
        )
        
        return WorkspaceResponse.from_orm(workspace)
        
    except SecurityError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to update workspace")


@router.get("/workspaces/{workspace_id}/members", response_model=List[UserResponse])
async def get_workspace_members(
    workspace_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get all members of a workspace."""
    try:
        user_service = UserManagementService(db)
        
        # Check permissions
        if not user_service._check_workspace_permission(
            str(current_user.id), workspace_id, "view_members"
        ):
            raise HTTPException(status_code=403, detail="Access denied")
        
        members = await user_service.get_workspace_members(workspace_id)
        return [UserResponse(**member) for member in members]
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to get workspace members")


@router.post("/workspaces/{workspace_id}/members")
async def add_workspace_member(
    workspace_id: str,
    member_data: WorkspaceMemberAdd,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Add a member to a workspace."""
    try:
        user_service = UserManagementService(db)
        workspace_member = await user_service.add_workspace_member(
            workspace_id=workspace_id,
            user_id=member_data.user_id,
            role=member_data.role,
            added_by=str(current_user.id),
            permissions=member_data.permissions
        )
        
        return {
            "message": "Member added to workspace successfully",
            "member_id": str(workspace_member.id)
        }
        
    except SecurityError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to add workspace member")


@router.delete("/workspaces/{workspace_id}/members/{user_id}")
async def remove_workspace_member(
    workspace_id: str,
    user_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Remove a member from a workspace."""
    try:
        user_service = UserManagementService(db)
        success = await user_service.remove_workspace_member(
            workspace_id=workspace_id,
            target_user_id=user_id,
            removed_by=str(current_user.id)
        )
        
        return {"message": "Member removed from workspace successfully"}
        
    except SecurityError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to remove workspace member")


# API Key Management Routes

@router.post("/organizations/{organization_id}/api-keys")
async def create_api_key(
    organization_id: str,
    key_data: APIKeyCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create a new API key."""
    try:
        user_service = UserManagementService(db)
        
        expires_at = None
        if key_data.expires_days:
            expires_at = datetime.utcnow() + timedelta(days=key_data.expires_days)
        
        api_key_record, api_key = await user_service.create_api_key(
            name=key_data.name,
            user_id=str(current_user.id),
            organization_id=organization_id,
            description=key_data.description,
            permissions=key_data.permissions,
            expires_at=expires_at
        )
        
        return {
            "message": "API key created successfully",
            "api_key": api_key,  # Only returned once
            "key_id": str(api_key_record.id),
            "key_prefix": api_key_record.key_prefix
        }
        
    except SecurityError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to create API key")


@router.get("/organizations/{organization_id}/api-keys", response_model=List[APIKeyResponse])
async def get_api_keys(
    organization_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get all API keys for an organization."""
    try:
        user_service = UserManagementService(db)
        
        # Check permissions
        if not user_service._check_organization_permission(
            str(current_user.id), organization_id, "view_api_keys"
        ):
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Get API keys
        api_keys = db.query(APIKey).filter(
            APIKey.organization_id == organization_id,
            APIKey.is_active == True
        ).all()
        
        return [APIKeyResponse.from_orm(key) for key in api_keys]
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to get API keys")


# User Profile Routes

@router.get("/profile/organizations", response_model=List[OrganizationResponse])
async def get_user_organizations(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get all organizations the current user belongs to."""
    try:
        user_service = UserManagementService(db)
        organizations = await user_service.get_user_organizations(str(current_user.id))
        
        return [OrganizationResponse.from_orm(org) for org in organizations]
        
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to get user organizations")


# Session Management Routes

@router.post("/sessions/revoke")
async def revoke_session(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
):
    """Revoke the current session."""
    try:
        user_service = UserManagementService(db)
        success = await user_service.revoke_session(credentials.credentials)
        
        return {"message": "Session revoked successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to revoke session")