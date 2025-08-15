"""
User management service for multi-user and role-based access control.
"""

import secrets
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_
import bcrypt
import jwt
from email_validator import validate_email, EmailNotValidError

from ..models.database import User
from ..models.user_management_models import (
    Organization, OrganizationUser, Workspace, WorkspaceMember,
    Project, UserInvitation, UserSession, UserAuditLog, APIKey
)
from ..core.interfaces import UserRole, SecurityError, ValidationError
from ..core.config import get_settings


class UserManagementService:
    """Service for managing users, organizations, and access control."""
    
    def __init__(self, db: Session):
        self.db = db
        self.settings = get_settings()
    
    # Organization Management
    
    async def create_organization(
        self,
        name: str,
        display_name: str,
        creator_user_id: str,
        description: Optional[str] = None,
        domain: Optional[str] = None,
        subscription_plan: str = "free"
    ) -> Organization:
        """Create a new organization."""
        try:
            # Validate organization name is unique
            existing = self.db.query(Organization).filter(Organization.name == name).first()
            if existing:
                raise ValidationError(f"Organization name '{name}' already exists")
            
            # Create organization
            organization = Organization(
                name=name,
                display_name=display_name,
                description=description,
                domain=domain,
                subscription_plan=subscription_plan,
                settings={
                    "allow_user_registration": False,
                    "require_email_verification": True,
                    "session_timeout_hours": 24,
                    "password_policy": {
                        "min_length": 8,
                        "require_uppercase": True,
                        "require_lowercase": True,
                        "require_numbers": True,
                        "require_symbols": False
                    }
                },
                features=["basic_analytics", "file_upload", "api_access"]
            )
            
            self.db.add(organization)
            self.db.flush()  # Get the ID
            
            # Add creator as admin
            org_user = OrganizationUser(
                organization_id=organization.id,
                user_id=creator_user_id,
                role=UserRole.ADMIN,
                permissions=["*"],  # Full permissions
                status="active"
            )
            
            self.db.add(org_user)
            
            # Create default workspace
            workspace = Workspace(
                name="Default Workspace",
                description="Default workspace for the organization",
                organization_id=organization.id,
                owner_id=creator_user_id,
                visibility="organization",
                settings={"default_workspace": True}
            )
            
            self.db.add(workspace)
            self.db.flush()
            
            # Add creator as workspace owner
            workspace_member = WorkspaceMember(
                workspace_id=workspace.id,
                user_id=creator_user_id,
                role="owner",
                permissions=["*"]
            )
            
            self.db.add(workspace_member)
            
            # Log audit event
            self._log_audit_event(
                user_id=creator_user_id,
                organization_id=organization.id,
                action="create_organization",
                resource_type="organization",
                resource_id=str(organization.id),
                details={"name": name, "display_name": display_name}
            )
            
            self.db.commit()
            return organization
            
        except Exception as e:
            self.db.rollback()
            raise e
    
    def get_organization(self, organization_id: str) -> Optional[Organization]:
        """Get organization by ID."""
        return self.db.query(Organization).filter(
            Organization.id == organization_id,
            Organization.is_active == True
        ).first()
    
    async def update_organization(
        self,
        organization_id: str,
        user_id: str,
        updates: Dict
    ) -> Organization:
        """Update organization settings."""
        # Check permissions
        if not await self._check_organization_permission(user_id, organization_id, "manage_organization"):
            raise SecurityError("Insufficient permissions to update organization")
        
        organization = await self.get_organization(organization_id)
        if not organization:
            raise ValidationError("Organization not found")
        
        # Update allowed fields
        allowed_fields = [
            "display_name", "description", "domain", "logo_url", 
            "website", "industry", "size", "settings", "features"
        ]
        
        for field, value in updates.items():
            if field in allowed_fields and hasattr(organization, field):
                setattr(organization, field, value)
        
        organization.updated_at = datetime.utcnow()
        
        # Log audit event
        self._log_audit_event(
            user_id=user_id,
            organization_id=organization_id,
            action="update_organization",
            resource_type="organization",
            resource_id=organization_id,
            details=updates
        )
        
        self.db.commit()
        return organization
    
    # User Management
    
    async def invite_user(
        self,
        email: str,
        organization_id: str,
        invited_by: str,
        role: UserRole = UserRole.VIEWER,
        permissions: List[str] = None,
        message: Optional[str] = None
    ) -> UserInvitation:
        """Invite a user to join an organization."""
        try:
            # Check permissions first
            if not self._check_organization_permission(invited_by, organization_id, "invite_users"):
                raise SecurityError("Insufficient permissions to invite users")
            
            # Validate email
            try:
                validated_email = validate_email(email)
                email = validated_email.email
            except EmailNotValidError:
                raise ValidationError("Invalid email address")
            
            # Check if user is already in organization
            existing_user = self.db.query(User).filter(User.email == email).first()
            if existing_user:
                existing_org_user = self.db.query(OrganizationUser).filter(
                    and_(
                        OrganizationUser.user_id == existing_user.id,
                        OrganizationUser.organization_id == organization_id
                    )
                ).first()
                if existing_org_user:
                    raise ValidationError("User is already a member of this organization")
            
            # Check for existing pending invitation
            existing_invitation = self.db.query(UserInvitation).filter(
                and_(
                    UserInvitation.email == email,
                    UserInvitation.organization_id == organization_id,
                    UserInvitation.status == "pending"
                )
            ).first()
            if existing_invitation:
                raise ValidationError("User already has a pending invitation")
            
            # Generate invitation token
            token = secrets.token_urlsafe(32)
            
            # Create invitation
            invitation = UserInvitation(
                email=email,
                organization_id=organization_id,
                invited_by=invited_by,
                role=role,
                permissions=permissions or [],
                token=token,
                message=message,
                expires_at=datetime.utcnow() + timedelta(days=7)  # 7 days to accept
            )
            
            self.db.add(invitation)
            
            # Log audit event
            self._log_audit_event(
                user_id=invited_by,
                organization_id=organization_id,
                action="invite_user",
                resource_type="user_invitation",
                resource_id=str(invitation.id),
                details={"email": email, "role": role.value}
            )
            
            self.db.commit()
            
            # TODO: Send invitation email
            # await self._send_invitation_email(invitation)
            
            return invitation
            
        except Exception as e:
            self.db.rollback()
            raise e
    
    async def accept_invitation(self, token: str, user_id: str) -> OrganizationUser:
        """Accept a user invitation."""
        try:
            # Find invitation
            invitation = self.db.query(UserInvitation).filter(
                and_(
                    UserInvitation.token == token,
                    UserInvitation.status == "pending",
                    UserInvitation.expires_at > datetime.utcnow()
                )
            ).first()
            
            if not invitation:
                raise ValidationError("Invalid or expired invitation")
            
            # Get user
            user = self.db.query(User).filter(User.id == user_id).first()
            if not user:
                raise ValidationError("User not found")
            
            # Verify email matches
            if user.email != invitation.email:
                raise ValidationError("Email address does not match invitation")
            
            # Check if user is already in organization
            existing_org_user = self.db.query(OrganizationUser).filter(
                and_(
                    OrganizationUser.user_id == user_id,
                    OrganizationUser.organization_id == invitation.organization_id
                )
            ).first()
            if existing_org_user:
                raise ValidationError("User is already a member of this organization")
            
            # Create organization user
            org_user = OrganizationUser(
                organization_id=invitation.organization_id,
                user_id=user_id,
                role=invitation.role,
                permissions=invitation.permissions or [],
                invited_by=invitation.invited_by,
                status="active"
            )
            
            self.db.add(org_user)
            
            # Update invitation
            invitation.status = "accepted"
            invitation.accepted_at = datetime.utcnow()
            
            # Log audit event
            self._log_audit_event(
                user_id=user_id,
                organization_id=invitation.organization_id,
                action="accept_invitation",
                resource_type="organization_user",
                resource_id=str(org_user.id),
                details={"invitation_id": str(invitation.id)}
            )
            
            self.db.commit()
            return org_user
            
        except Exception as e:
            self.db.rollback()
            raise e
    
    async def update_user_role(
        self,
        organization_id: str,
        target_user_id: str,
        new_role: UserRole,
        updated_by: str,
        permissions: List[str] = None
    ) -> OrganizationUser:
        """Update a user's role in an organization."""
        # Check permissions
        if not self._check_organization_permission(updated_by, organization_id, "manage_users"):
            raise SecurityError("Insufficient permissions to update user roles")
        
        # Get organization user
        org_user = self.db.query(OrganizationUser).filter(
            and_(
                OrganizationUser.organization_id == organization_id,
                OrganizationUser.user_id == target_user_id,
                OrganizationUser.status == "active"
            )
        ).first()
        
        if not org_user:
            raise ValidationError("User not found in organization")
        
        # Update role and permissions
        old_role = org_user.role
        org_user.role = new_role
        if permissions is not None:
            org_user.permissions = permissions
        
        # Log audit event
        self._log_audit_event(
            user_id=updated_by,
            organization_id=organization_id,
            action="update_user_role",
            resource_type="organization_user",
            resource_id=str(org_user.id),
            details={
                "target_user_id": target_user_id,
                "old_role": old_role.value,
                "new_role": new_role.value
            }
        )
        
        self.db.commit()
        return org_user
    
    async def remove_user_from_organization(
        self,
        organization_id: str,
        target_user_id: str,
        removed_by: str
    ) -> bool:
        """Remove a user from an organization."""
        # Check permissions
        if not self._check_organization_permission(removed_by, organization_id, "manage_users"):
            raise SecurityError("Insufficient permissions to remove users")
        
        # Get organization user
        org_user = self.db.query(OrganizationUser).filter(
            and_(
                OrganizationUser.organization_id == organization_id,
                OrganizationUser.user_id == target_user_id
            )
        ).first()
        
        if not org_user:
            raise ValidationError("User not found in organization")
        
        # Remove from all workspaces in the organization
        workspace_members = self.db.query(WorkspaceMember).join(Workspace).filter(
            and_(
                Workspace.organization_id == organization_id,
                WorkspaceMember.user_id == target_user_id
            )
        ).all()
        
        for member in workspace_members:
            self.db.delete(member)
        
        # Remove organization user
        self.db.delete(org_user)
        
        # Log audit event
        self._log_audit_event(
            user_id=removed_by,
            organization_id=organization_id,
            action="remove_user",
            resource_type="organization_user",
            resource_id=str(org_user.id),
            details={"target_user_id": target_user_id}
        )
        
        self.db.commit()
        return True
    
    # Workspace Management
    
    async def create_workspace(
        self,
        name: str,
        organization_id: str,
        owner_id: str,
        description: Optional[str] = None,
        visibility: str = "private"
    ) -> Workspace:
        """Create a new workspace."""
        try:
            # Check permissions
            if not self._check_organization_permission(owner_id, organization_id, "create_workspaces"):
                raise SecurityError("Insufficient permissions to create workspaces")
            
            # Check workspace limit
            organization = self.get_organization(organization_id)
            if not organization:
                raise ValidationError("Organization not found")
            
            workspace_count = self.db.query(Workspace).filter(
                and_(
                    Workspace.organization_id == organization_id,
                    Workspace.is_active == True
                )
            ).count()
            
            if workspace_count >= organization.max_workspaces:
                raise ValidationError(f"Maximum workspace limit ({organization.max_workspaces}) reached")
            
            # Create workspace
            workspace = Workspace(
                name=name,
                description=description,
                organization_id=organization_id,
                owner_id=owner_id,
                visibility=visibility,
                settings={"created_by_user": True}
            )
            
            self.db.add(workspace)
            self.db.flush()
            
            # Add owner as workspace member
            workspace_member = WorkspaceMember(
                workspace_id=workspace.id,
                user_id=owner_id,
                role="owner",
                permissions=["*"]
            )
            
            self.db.add(workspace_member)
            
            # Log audit event
            self._log_audit_event(
                user_id=owner_id,
                organization_id=organization_id,
                action="create_workspace",
                resource_type="workspace",
                resource_id=str(workspace.id),
                details={"name": name, "visibility": visibility}
            )
            
            self.db.commit()
            return workspace
            
        except Exception as e:
            self.db.rollback()
            raise e
    
    async def add_workspace_member(
        self,
        workspace_id: str,
        user_id: str,
        role: str,
        added_by: str,
        permissions: List[str] = None
    ) -> WorkspaceMember:
        """Add a member to a workspace."""
        try:
            # Get workspace
            workspace = self.db.query(Workspace).filter(Workspace.id == workspace_id).first()
            if not workspace:
                raise ValidationError("Workspace not found")
            
            # Check permissions
            if not self._check_workspace_permission(added_by, workspace_id, "manage_members"):
                raise SecurityError("Insufficient permissions to add workspace members")
            
            # Check if user is in organization
            org_user = self.db.query(OrganizationUser).filter(
                and_(
                    OrganizationUser.organization_id == workspace.organization_id,
                    OrganizationUser.user_id == user_id,
                    OrganizationUser.status == "active"
                )
            ).first()
            
            if not org_user:
                raise ValidationError("User is not a member of the organization")
            
            # Check if already a member
            existing_member = self.db.query(WorkspaceMember).filter(
                and_(
                    WorkspaceMember.workspace_id == workspace_id,
                    WorkspaceMember.user_id == user_id
                )
            ).first()
            
            if existing_member:
                raise ValidationError("User is already a member of this workspace")
            
            # Create workspace member
            workspace_member = WorkspaceMember(
                workspace_id=workspace_id,
                user_id=user_id,
                role=role,
                permissions=permissions or [],
                added_by=added_by
            )
            
            self.db.add(workspace_member)
            
            # Log audit event
            self._log_audit_event(
                user_id=added_by,
                organization_id=workspace.organization_id,
                action="add_workspace_member",
                resource_type="workspace_member",
                resource_id=str(workspace_member.id),
                details={"workspace_id": workspace_id, "target_user_id": user_id, "role": role}
            )
            
            self.db.commit()
            return workspace_member
            
        except Exception as e:
            self.db.rollback()
            raise e
    
    # Session Management
    
    async def create_session(
        self,
        user_id: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        device_info: Dict = None
    ) -> Tuple[str, str]:
        """Create a new user session."""
        try:
            # Generate tokens
            session_token = secrets.token_urlsafe(32)
            refresh_token = secrets.token_urlsafe(32)
            
            # Create session
            session = UserSession(
                user_id=user_id,
                session_token=session_token,
                refresh_token=refresh_token,
                ip_address=ip_address,
                user_agent=user_agent,
                device_info=device_info or {},
                expires_at=datetime.utcnow() + timedelta(hours=24)
            )
            
            self.db.add(session)
            
            # Update user last login
            user = self.db.query(User).filter(User.id == user_id).first()
            if user:
                user.last_login = datetime.utcnow()
            
            self.db.commit()
            return session_token, refresh_token
            
        except Exception as e:
            self.db.rollback()
            raise e
    
    async def validate_session(self, session_token: str) -> Optional[User]:
        """Validate a session token and return the user."""
        session = self.db.query(UserSession).filter(
            and_(
                UserSession.session_token == session_token,
                UserSession.is_active == True,
                UserSession.expires_at > datetime.utcnow()
            )
        ).first()
        
        if not session:
            return None
        
        # Update last activity
        session.last_activity = datetime.utcnow()
        self.db.commit()
        
        # Get user
        user = self.db.query(User).filter(
            and_(
                User.id == session.user_id,
                User.is_active == True
            )
        ).first()
        
        return user
    
    async def revoke_session(self, session_token: str) -> bool:
        """Revoke a user session."""
        session = self.db.query(UserSession).filter(
            UserSession.session_token == session_token
        ).first()
        
        if session:
            session.is_active = False
            self.db.commit()
            return True
        
        return False
    
    # API Key Management
    
    async def create_api_key(
        self,
        name: str,
        user_id: str,
        organization_id: str,
        description: Optional[str] = None,
        permissions: List[str] = None,
        expires_at: Optional[datetime] = None
    ) -> Tuple[APIKey, str]:
        """Create a new API key."""
        try:
            # Check permissions
            if not self._check_organization_permission(user_id, organization_id, "manage_api_keys"):
                raise SecurityError("Insufficient permissions to create API keys")
            
            # Generate API key
            api_key = secrets.token_urlsafe(32)
            key_hash = hashlib.sha256(api_key.encode()).hexdigest()
            key_prefix = api_key[:8]
            
            # Create API key record
            api_key_record = APIKey(
                name=name,
                description=description,
                user_id=user_id,
                organization_id=organization_id,
                key_hash=key_hash,
                key_prefix=key_prefix,
                permissions=permissions or [],
                expires_at=expires_at
            )
            
            self.db.add(api_key_record)
            
            # Log audit event
            self._log_audit_event(
                user_id=user_id,
                organization_id=organization_id,
                action="create_api_key",
                resource_type="api_key",
                resource_id=str(api_key_record.id),
                details={"name": name, "permissions": permissions or []}
            )
            
            self.db.commit()
            return api_key_record, api_key
            
        except Exception as e:
            self.db.rollback()
            raise e
    
    async def validate_api_key(self, api_key: str) -> Optional[APIKey]:
        """Validate an API key."""
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        api_key_record = self.db.query(APIKey).filter(
            and_(
                APIKey.key_hash == key_hash,
                APIKey.is_active == True,
                or_(APIKey.expires_at.is_(None), APIKey.expires_at > datetime.utcnow())
            )
        ).first()
        
        if api_key_record:
            # Update usage
            api_key_record.usage_count += 1
            api_key_record.last_used = datetime.utcnow()
            self.db.commit()
        
        return api_key_record
    
    # Permission Checking
    
    def _check_organization_permission(
        self,
        user_id: str,
        organization_id: str,
        permission: str
    ) -> bool:
        """Check if user has permission in organization."""
        org_user = self.db.query(OrganizationUser).filter(
            and_(
                OrganizationUser.user_id == user_id,
                OrganizationUser.organization_id == organization_id,
                OrganizationUser.status == "active"
            )
        ).first()
        
        if not org_user:
            return False
        
        # Admin has all permissions
        if org_user.role == UserRole.ADMIN:
            return True
        
        # Check specific permissions
        if "*" in org_user.permissions or permission in org_user.permissions:
            return True
        
        # Role-based permissions
        role_permissions = {
            UserRole.ADMIN: ["*"],
            UserRole.ANALYST: [
                "view_data", "create_workspaces", "invite_users", "manage_api_keys"
            ],
            UserRole.VIEWER: ["view_data"]
        }
        
        return permission in role_permissions.get(org_user.role, [])
    
    def _check_workspace_permission(
        self,
        user_id: str,
        workspace_id: str,
        permission: str
    ) -> bool:
        """Check if user has permission in workspace."""
        workspace_member = self.db.query(WorkspaceMember).filter(
            and_(
                WorkspaceMember.user_id == user_id,
                WorkspaceMember.workspace_id == workspace_id,
                WorkspaceMember.status == "active"
            )
        ).first()
        
        if not workspace_member:
            return False
        
        # Owner and admin have all permissions
        if workspace_member.role in ["owner", "admin"]:
            return True
        
        # Check specific permissions
        if "*" in workspace_member.permissions or permission in workspace_member.permissions:
            return True
        
        # Role-based permissions
        role_permissions = {
            "owner": ["*"],
            "admin": ["*"],
            "member": ["view_data", "create_projects"],
            "viewer": ["view_data"]
        }
        
        return permission in role_permissions.get(workspace_member.role, [])
    
    # Audit Logging
    
    def _log_audit_event(
        self,
        user_id: str,
        action: str,
        resource_type: str,
        resource_id: str,
        details: Dict,
        organization_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        success: bool = True,
        error_message: Optional[str] = None
    ) -> None:
        """Log an audit event."""
        audit_log = UserAuditLog(
            user_id=user_id,
            organization_id=organization_id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            details=details,
            ip_address=ip_address,
            user_agent=user_agent,
            success=success,
            error_message=error_message
        )
        
        self.db.add(audit_log)
        # Note: Commit is handled by the calling method
    
    # Utility Methods
    
    async def get_user_organizations(self, user_id: str) -> List[Organization]:
        """Get all organizations a user belongs to."""
        return self.db.query(Organization).join(OrganizationUser).filter(
            and_(
                OrganizationUser.user_id == user_id,
                OrganizationUser.status == "active",
                Organization.is_active == True
            )
        ).all()
    
    async def get_user_workspaces(self, user_id: str, organization_id: Optional[str] = None) -> List[Workspace]:
        """Get all workspaces accessible to a user."""
        query = self.db.query(Workspace).join(WorkspaceMember).filter(
            and_(
                WorkspaceMember.user_id == user_id,
                WorkspaceMember.status == "active",
                Workspace.is_active == True
            )
        )
        
        if organization_id:
            query = query.filter(Workspace.organization_id == organization_id)
        
        return query.all()
    
    async def get_workspace_members(self, workspace_id: str) -> List[Dict]:
        """Get all members of a workspace."""
        workspace_members = self.db.query(WorkspaceMember, User).join(User).filter(
            and_(
                WorkspaceMember.workspace_id == workspace_id,
                WorkspaceMember.status == "active",
                User.is_active == True
            )
        ).all()
        
        members = []
        for workspace_member, user in workspace_members:
            members.append({
                "user_id": str(user.id),
                "email": user.email,
                "full_name": user.full_name,
                "role": workspace_member.role,
                "permissions": workspace_member.permissions,
                "added_at": workspace_member.added_at,
                "last_active": workspace_member.last_active
            })
        
        return members
    
    async def update_workspace(
        self,
        workspace_id: str,
        user_id: str,
        updates: Dict
    ) -> Workspace:
        """Update workspace settings."""
        # Check permissions
        if not self._check_workspace_permission(user_id, workspace_id, "manage_workspace"):
            raise SecurityError("Insufficient permissions to update workspace")
        
        workspace = self.db.query(Workspace).filter(Workspace.id == workspace_id).first()
        if not workspace:
            raise ValidationError("Workspace not found")
        
        # Update allowed fields
        allowed_fields = ["name", "description", "visibility", "settings", "tags"]
        
        for field, value in updates.items():
            if field in allowed_fields and hasattr(workspace, field):
                setattr(workspace, field, value)
        
        workspace.updated_at = datetime.utcnow()
        
        # Log audit event
        self._log_audit_event(
            user_id=user_id,
            organization_id=str(workspace.organization_id),
            action="update_workspace",
            resource_type="workspace",
            resource_id=workspace_id,
            details=updates
        )
        
        self.db.commit()
        return workspace
    
    async def remove_workspace_member(
        self,
        workspace_id: str,
        target_user_id: str,
        removed_by: str
    ) -> bool:
        """Remove a member from a workspace."""
        # Check permissions
        if not self._check_workspace_permission(removed_by, workspace_id, "manage_members"):
            raise SecurityError("Insufficient permissions to remove workspace members")
        
        # Get workspace member
        workspace_member = self.db.query(WorkspaceMember).filter(
            and_(
                WorkspaceMember.workspace_id == workspace_id,
                WorkspaceMember.user_id == target_user_id
            )
        ).first()
        
        if not workspace_member:
            raise ValidationError("User not found in workspace")
        
        # Don't allow removing the owner
        if workspace_member.role == "owner":
            raise ValidationError("Cannot remove workspace owner")
        
        # Get workspace for audit logging
        workspace = self.db.query(Workspace).filter(Workspace.id == workspace_id).first()
        
        # Remove workspace member
        self.db.delete(workspace_member)
        
        # Log audit event
        self._log_audit_event(
            user_id=removed_by,
            organization_id=str(workspace.organization_id) if workspace else None,
            action="remove_workspace_member",
            resource_type="workspace_member",
            resource_id=str(workspace_member.id),
            details={"workspace_id": workspace_id, "target_user_id": target_user_id}
        )
        
        self.db.commit()
        return True
    
    async def get_organization_users(self, organization_id: str) -> List[Dict]:
        """Get all users in an organization."""
        org_users = self.db.query(OrganizationUser, User).join(User).filter(
            and_(
                OrganizationUser.organization_id == organization_id,
                OrganizationUser.status == "active",
                User.is_active == True
            )
        ).all()
        
        users = []
        for org_user, user in org_users:
            users.append({
                "user_id": str(user.id),
                "email": user.email,
                "full_name": user.full_name,
                "role": org_user.role.value,
                "permissions": org_user.permissions,
                "joined_at": org_user.joined_at.isoformat(),
                "last_active": org_user.last_active.isoformat() if org_user.last_active else None
            })
        
        return users

    async def get_user_workspaces(self, user_id: str, organization_id: Optional[str] = None) -> List[Workspace]:
        """Get all workspaces a user has access to."""
        query = self.db.query(Workspace).join(WorkspaceMember).filter(
            and_(
                WorkspaceMember.user_id == user_id,
                WorkspaceMember.status == "active",
                Workspace.is_active == True
            )
        )
        
        if organization_id:
            query = query.filter(Workspace.organization_id == organization_id)
        
        return query.all()
    
    async def get_organization_users(self, organization_id: str) -> List[Dict]:
        """Get all users in an organization with their roles."""
        org_users = self.db.query(OrganizationUser, User).join(User).filter(
            and_(
                OrganizationUser.organization_id == organization_id,
                OrganizationUser.status == "active",
                User.is_active == True
            )
        ).all()
        
        return [
            {
                "user_id": str(user.id),
                "email": user.email,
                "full_name": user.full_name,
                "role": org_user.role.value,
                "permissions": org_user.permissions,
                "joined_at": org_user.joined_at,
                "last_active": org_user.last_active
            }
            for org_user, user in org_users
        ]