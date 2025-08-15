"""
User management models for multi-user and role-based access control.
"""

from datetime import datetime
from typing import Dict, Any, Optional, List
from sqlalchemy import (
    Column, String, Integer, DateTime, Boolean, Text, JSON, 
    ForeignKey, Float, Enum as SQLEnum, Index, UniqueConstraint
)
from sqlalchemy.orm import relationship, validates
from sqlalchemy.dialects.postgresql import UUID, JSONB
import uuid

from .database import Base, JSONType
from ..core.interfaces import UserRole


class Organization(Base):
    """Organization model for multi-tenant support."""
    
    __tablename__ = "organizations"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    display_name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    domain = Column(String(255), nullable=True)  # Email domain for auto-assignment
    logo_url = Column(String(500), nullable=True)
    website = Column(String(500), nullable=True)
    industry = Column(String(100), nullable=True)
    size = Column(String(50), nullable=True)  # startup, small, medium, large, enterprise
    
    # Subscription and billing
    subscription_plan = Column(String(50), nullable=False, default="free")  # free, pro, enterprise
    subscription_status = Column(String(50), nullable=False, default="active")  # active, suspended, cancelled
    billing_email = Column(String(255), nullable=True)
    max_users = Column(Integer, nullable=False, default=5)
    max_workspaces = Column(Integer, nullable=False, default=3)
    max_storage_gb = Column(Integer, nullable=False, default=10)
    
    # Configuration
    settings = Column(JSONType, nullable=False, default=dict)
    features = Column(JSONType, nullable=False, default=list)  # List of enabled features
    integrations = Column(JSONType, nullable=False, default=dict)  # External integrations config
    
    # Metadata
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    users = relationship("OrganizationUser", back_populates="organization", cascade="all, delete-orphan")
    workspaces = relationship("Workspace", back_populates="organization", cascade="all, delete-orphan")
    invitations = relationship("UserInvitation", back_populates="organization", cascade="all, delete-orphan")
    user_audit_logs = relationship("UserAuditLog", back_populates="organization", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index('idx_organization_name', 'name'),
        Index('idx_organization_domain', 'domain'),
        Index('idx_organization_active', 'is_active'),
        Index('idx_organization_plan', 'subscription_plan'),
        UniqueConstraint('name', name='uq_organization_name'),
    )
    
    @validates('settings')
    def validate_settings(self, key, settings):
        """Validate settings is a dict."""
        if not isinstance(settings, dict):
            raise ValueError("Settings must be a dictionary")
        return settings
    
    @validates('features')
    def validate_features(self, key, features):
        """Validate features is a list."""
        if not isinstance(features, list):
            raise ValueError("Features must be a list")
        return features
    
    @validates('integrations')
    def validate_integrations(self, key, integrations):
        """Validate integrations is a dict."""
        if not isinstance(integrations, dict):
            raise ValueError("Integrations must be a dictionary")
        return integrations
    
    def __repr__(self):
        return f"<Organization(id={self.id}, name={self.name}, plan={self.subscription_plan})>"


class OrganizationUser(Base):
    """Association model for users in organizations with roles."""
    
    __tablename__ = "organization_users"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    organization_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id"), nullable=False)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    role = Column(SQLEnum(UserRole), nullable=False, default=UserRole.VIEWER)
    permissions = Column(JSONType, nullable=False, default=list)
    
    # Status and metadata
    status = Column(String(50), nullable=False, default="active")  # active, suspended, pending
    invited_by = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    joined_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_active = Column(DateTime, nullable=True)
    
    # Relationships
    organization = relationship("Organization", back_populates="users")
    user = relationship("User", foreign_keys=[user_id])
    inviter = relationship("User", foreign_keys=[invited_by])
    
    # Indexes
    __table_args__ = (
        Index('idx_org_user_organization', 'organization_id'),
        Index('idx_org_user_user', 'user_id'),
        Index('idx_org_user_role', 'role'),
        Index('idx_org_user_status', 'status'),
        UniqueConstraint('organization_id', 'user_id', name='uq_org_user'),
    )
    
    @validates('permissions')
    def validate_permissions(self, key, permissions):
        """Validate permissions is a list."""
        if not isinstance(permissions, list):
            raise ValueError("Permissions must be a list")
        return permissions
    
    def __repr__(self):
        return f"<OrganizationUser(org_id={self.organization_id}, user_id={self.user_id}, role={self.role})>"


class Workspace(Base):
    """Workspace model for project organization."""
    
    __tablename__ = "workspaces"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    organization_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id"), nullable=False)
    owner_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    
    # Configuration
    settings = Column(JSONType, nullable=False, default=dict)
    tags = Column(JSONType, nullable=False, default=list)
    
    # Access control
    visibility = Column(String(50), nullable=False, default="private")  # private, organization, public
    
    # Metadata
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    organization = relationship("Organization", back_populates="workspaces")
    owner = relationship("User")
    members = relationship("WorkspaceMember", back_populates="workspace", cascade="all, delete-orphan")
    projects = relationship("Project", back_populates="workspace", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index('idx_workspace_organization', 'organization_id'),
        Index('idx_workspace_owner', 'owner_id'),
        Index('idx_workspace_visibility', 'visibility'),
        Index('idx_workspace_active', 'is_active'),
        Index('idx_workspace_name', 'name'),
    )
    
    @validates('settings')
    def validate_settings(self, key, settings):
        """Validate settings is a dict."""
        if not isinstance(settings, dict):
            raise ValueError("Settings must be a dictionary")
        return settings
    
    @validates('tags')
    def validate_tags(self, key, tags):
        """Validate tags is a list."""
        if not isinstance(tags, list):
            raise ValueError("Tags must be a list")
        return tags
    
    def __repr__(self):
        return f"<Workspace(id={self.id}, name={self.name}, org_id={self.organization_id})>"


class WorkspaceMember(Base):
    """Association model for workspace members with permissions."""
    
    __tablename__ = "workspace_members"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    workspace_id = Column(UUID(as_uuid=True), ForeignKey("workspaces.id"), nullable=False)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    role = Column(String(50), nullable=False, default="member")  # owner, admin, member, viewer
    permissions = Column(JSONType, nullable=False, default=list)
    
    # Status and metadata
    status = Column(String(50), nullable=False, default="active")  # active, suspended
    added_by = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    added_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_active = Column(DateTime, nullable=True)
    
    # Relationships
    workspace = relationship("Workspace", back_populates="members")
    user = relationship("User", foreign_keys=[user_id])
    added_by_user = relationship("User", foreign_keys=[added_by])
    
    # Indexes
    __table_args__ = (
        Index('idx_workspace_member_workspace', 'workspace_id'),
        Index('idx_workspace_member_user', 'user_id'),
        Index('idx_workspace_member_role', 'role'),
        Index('idx_workspace_member_status', 'status'),
        UniqueConstraint('workspace_id', 'user_id', name='uq_workspace_member'),
    )
    
    @validates('permissions')
    def validate_permissions(self, key, permissions):
        """Validate permissions is a list."""
        if not isinstance(permissions, list):
            raise ValueError("Permissions must be a list")
        return permissions
    
    def __repr__(self):
        return f"<WorkspaceMember(workspace_id={self.workspace_id}, user_id={self.user_id}, role={self.role})>"


class Project(Base):
    """Project model for organizing work within workspaces."""
    
    __tablename__ = "projects"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    workspace_id = Column(UUID(as_uuid=True), ForeignKey("workspaces.id"), nullable=False)
    owner_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    
    # Configuration
    settings = Column(JSONType, nullable=False, default=dict)
    tags = Column(JSONType, nullable=False, default=list)
    
    # Status
    status = Column(String(50), nullable=False, default="active")  # active, archived, completed
    
    # Metadata
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    workspace = relationship("Workspace", back_populates="projects")
    owner = relationship("User")
    
    # Indexes
    __table_args__ = (
        Index('idx_project_workspace', 'workspace_id'),
        Index('idx_project_owner', 'owner_id'),
        Index('idx_project_status', 'status'),
        Index('idx_project_active', 'is_active'),
        Index('idx_project_name', 'name'),
    )
    
    @validates('settings')
    def validate_settings(self, key, settings):
        """Validate settings is a dict."""
        if not isinstance(settings, dict):
            raise ValueError("Settings must be a dictionary")
        return settings
    
    @validates('tags')
    def validate_tags(self, key, tags):
        """Validate tags is a list."""
        if not isinstance(tags, list):
            raise ValueError("Tags must be a list")
        return tags
    
    def __repr__(self):
        return f"<Project(id={self.id}, name={self.name}, workspace_id={self.workspace_id})>"


class UserInvitation(Base):
    """User invitation model for inviting users to organizations."""
    
    __tablename__ = "user_invitations"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), nullable=False)
    organization_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id"), nullable=False)
    invited_by = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    role = Column(SQLEnum(UserRole), nullable=False, default=UserRole.VIEWER)
    permissions = Column(JSONType, nullable=False, default=list)
    
    # Invitation details
    token = Column(String(255), unique=True, nullable=False, index=True)
    message = Column(Text, nullable=True)
    
    # Status and timing
    status = Column(String(50), nullable=False, default="pending")  # pending, accepted, expired, cancelled
    expires_at = Column(DateTime, nullable=False)
    accepted_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    organization = relationship("Organization", back_populates="invitations")
    inviter = relationship("User")
    
    # Indexes
    __table_args__ = (
        Index('idx_invitation_email', 'email'),
        Index('idx_invitation_organization', 'organization_id'),
        Index('idx_invitation_status', 'status'),
        Index('idx_invitation_expires', 'expires_at'),
        Index('idx_invitation_token', 'token'),
    )
    
    @validates('permissions')
    def validate_permissions(self, key, permissions):
        """Validate permissions is a list."""
        if not isinstance(permissions, list):
            raise ValueError("Permissions must be a list")
        return permissions
    
    def __repr__(self):
        return f"<UserInvitation(id={self.id}, email={self.email}, status={self.status})>"


class UserSession(Base):
    """User session model for session management."""
    
    __tablename__ = "user_sessions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    session_token = Column(String(255), unique=True, nullable=False, index=True)
    refresh_token = Column(String(255), unique=True, nullable=True, index=True)
    
    # Session details
    ip_address = Column(String(45), nullable=True)  # IPv6 support
    user_agent = Column(String(500), nullable=True)
    device_info = Column(JSONType, nullable=False, default=dict)
    
    # Status and timing
    is_active = Column(Boolean, default=True, nullable=False)
    expires_at = Column(DateTime, nullable=False)
    last_activity = Column(DateTime, default=datetime.utcnow, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    user = relationship("User")
    
    # Indexes
    __table_args__ = (
        Index('idx_session_user', 'user_id'),
        Index('idx_session_active', 'is_active'),
        Index('idx_session_expires', 'expires_at'),
        Index('idx_session_last_activity', 'last_activity'),
    )
    
    @validates('device_info')
    def validate_device_info(self, key, device_info):
        """Validate device_info is a dict."""
        if not isinstance(device_info, dict):
            raise ValueError("Device info must be a dictionary")
        return device_info
    
    def __repr__(self):
        return f"<UserSession(id={self.id}, user_id={self.user_id}, active={self.is_active})>"


class UserAuditLog(Base):
    """User audit log model for tracking user actions."""
    
    __tablename__ = "user_audit_logs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    organization_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id"), nullable=True)
    
    # Action details
    action = Column(String(100), nullable=False)
    resource_type = Column(String(100), nullable=False)
    resource_id = Column(String(255), nullable=True)
    details = Column(JSONType, nullable=False, default=dict)
    
    # Request context
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(String(500), nullable=True)
    session_id = Column(UUID(as_uuid=True), nullable=True)
    
    # Status
    success = Column(Boolean, nullable=False, default=True)
    error_message = Column(Text, nullable=True)
    
    # Timing
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="user_audit_logs")
    organization = relationship("Organization", back_populates="user_audit_logs")
    
    # Indexes
    __table_args__ = (
        Index('idx_audit_user', 'user_id'),
        Index('idx_audit_organization', 'organization_id'),
        Index('idx_audit_action', 'action'),
        Index('idx_audit_resource_type', 'resource_type'),
        Index('idx_audit_timestamp', 'timestamp'),
        Index('idx_audit_success', 'success'),
    )
    
    @validates('details')
    def validate_details(self, key, details):
        """Validate details is a dict."""
        if not isinstance(details, dict):
            raise ValueError("Details must be a dictionary")
        return details
    
    def __repr__(self):
        return f"<UserAuditLog(id={self.id}, user_id={self.user_id}, action={self.action})>"


class APIKey(Base):
    """API key model for API access management."""
    
    __tablename__ = "api_keys"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    organization_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id"), nullable=False)
    
    # Key details
    key_hash = Column(String(255), unique=True, nullable=False, index=True)
    key_prefix = Column(String(20), nullable=False)  # First few characters for identification
    
    # Permissions and limits
    permissions = Column(JSONType, nullable=False, default=list)
    rate_limit_per_minute = Column(Integer, nullable=False, default=60)
    rate_limit_per_hour = Column(Integer, nullable=False, default=1000)
    rate_limit_per_day = Column(Integer, nullable=False, default=10000)
    
    # Usage tracking
    usage_count = Column(Integer, nullable=False, default=0)
    last_used = Column(DateTime, nullable=True)
    
    # Status and timing
    is_active = Column(Boolean, default=True, nullable=False)
    expires_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    user = relationship("User")
    organization = relationship("Organization")
    
    # Indexes
    __table_args__ = (
        Index('idx_api_key_user', 'user_id'),
        Index('idx_api_key_organization', 'organization_id'),
        Index('idx_api_key_active', 'is_active'),
        Index('idx_api_key_expires', 'expires_at'),
        Index('idx_api_key_prefix', 'key_prefix'),
    )
    
    @validates('permissions')
    def validate_permissions(self, key, permissions):
        """Validate permissions is a list."""
        if not isinstance(permissions, list):
            raise ValueError("Permissions must be a list")
        return permissions
    
    def __repr__(self):
        return f"<APIKey(id={self.id}, name={self.name}, prefix={self.key_prefix})>"