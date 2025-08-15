"""
Data models for collaboration and sharing functionality.
"""

from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from sqlalchemy import Column, Integer, String, DateTime, Text, Boolean, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class ProjectStatus(Enum):
    """Project status enumeration."""
    ACTIVE = "active"
    ARCHIVED = "archived"
    DELETED = "deleted"


class SharePermission(Enum):
    """Share permission levels."""
    VIEW = "view"
    COMMENT = "comment"
    EDIT = "edit"
    ADMIN = "admin"


class ApprovalStatus(Enum):
    """Approval workflow status."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    NEEDS_REVISION = "needs_revision"


class Project(Base):
    """Project model for organizing generated content."""
    __tablename__ = "projects"
    
    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    owner_id = Column(String(255), nullable=False)
    status = Column(String(50), default=ProjectStatus.ACTIVE.value)
    tags = Column(JSON)
    project_metadata = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    content_items = relationship("ContentItem", back_populates="project")
    shares = relationship("ProjectShare", back_populates="project")
    comments = relationship("ProjectComment", back_populates="project")
    approvals = relationship("ApprovalWorkflow", back_populates="project")


class ContentItem(Base):
    """Content item within a project."""
    __tablename__ = "content_items"
    
    id = Column(Integer, primary_key=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False)
    name = Column(String(255), nullable=False)
    content_type = Column(String(50), nullable=False)  # image, video, etc.
    file_path = Column(String(500))
    thumbnail_path = Column(String(500))
    generation_params = Column(JSON)
    quality_metrics = Column(JSON)
    tags = Column(JSON)
    content_metadata = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    project = relationship("Project", back_populates="content_items")
    versions = relationship("ContentVersion", back_populates="content_item")
    comments = relationship("ContentComment", back_populates="content_item")


class ContentVersion(Base):
    """Version control for content items."""
    __tablename__ = "content_versions"
    
    id = Column(Integer, primary_key=True)
    content_item_id = Column(Integer, ForeignKey("content_items.id"), nullable=False)
    version_number = Column(Integer, nullable=False)
    file_path = Column(String(500), nullable=False)
    thumbnail_path = Column(String(500))
    generation_params = Column(JSON)
    quality_metrics = Column(JSON)
    change_description = Column(Text)
    created_by = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    content_item = relationship("ContentItem", back_populates="versions")


class ProjectShare(Base):
    """Project sharing permissions."""
    __tablename__ = "project_shares"
    
    id = Column(Integer, primary_key=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False)
    shared_with_user_id = Column(String(255), nullable=False)
    permission_level = Column(String(50), nullable=False)
    shared_by = Column(String(255), nullable=False)
    expires_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    project = relationship("Project", back_populates="shares")


class ProjectComment(Base):
    """Comments on projects."""
    __tablename__ = "project_comments"
    
    id = Column(Integer, primary_key=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False)
    user_id = Column(String(255), nullable=False)
    content = Column(Text, nullable=False)
    parent_comment_id = Column(Integer, ForeignKey("project_comments.id"))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    project = relationship("Project", back_populates="comments")
    parent_comment = relationship("ProjectComment", remote_side=[id])


class ContentComment(Base):
    """Comments on specific content items."""
    __tablename__ = "content_comments"
    
    id = Column(Integer, primary_key=True)
    content_item_id = Column(Integer, ForeignKey("content_items.id"), nullable=False)
    user_id = Column(String(255), nullable=False)
    content = Column(Text, nullable=False)
    position_x = Column(Integer)  # For positioning comments on images/videos
    position_y = Column(Integer)
    timestamp = Column(Integer)  # For video comments
    parent_comment_id = Column(Integer, ForeignKey("content_comments.id"))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    content_item = relationship("ContentItem", back_populates="comments")
    parent_comment = relationship("ContentComment", remote_side=[id])


class ApprovalWorkflow(Base):
    """Approval workflows for projects or content."""
    __tablename__ = "approval_workflows"
    
    id = Column(Integer, primary_key=True)
    project_id = Column(Integer, ForeignKey("projects.id"))
    content_item_id = Column(Integer, ForeignKey("content_items.id"))
    workflow_name = Column(String(255), nullable=False)
    status = Column(String(50), default=ApprovalStatus.PENDING.value)
    requested_by = Column(String(255), nullable=False)
    assigned_to = Column(String(255), nullable=False)
    due_date = Column(DateTime)
    feedback = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    project = relationship("Project", back_populates="approvals")


@dataclass
class ProjectCreateRequest:
    """Request model for creating a project."""
    name: str
    description: Optional[str] = None
    tags: Optional[List[str]] = None
    project_metadata: Optional[Dict[str, Any]] = None


@dataclass
class ProjectUpdateRequest:
    """Request model for updating a project."""
    name: Optional[str] = None
    description: Optional[str] = None
    tags: Optional[List[str]] = None
    project_metadata: Optional[Dict[str, Any]] = None
    status: Optional[ProjectStatus] = None


@dataclass
class ShareRequest:
    """Request model for sharing a project."""
    user_id: str
    permission_level: SharePermission
    expires_at: Optional[datetime] = None


@dataclass
class CommentRequest:
    """Request model for creating a comment."""
    content: str
    parent_comment_id: Optional[int] = None
    position_x: Optional[int] = None
    position_y: Optional[int] = None
    timestamp: Optional[int] = None


@dataclass
class ApprovalRequest:
    """Request model for creating an approval workflow."""
    workflow_name: str
    assigned_to: str
    due_date: Optional[datetime] = None


@dataclass
class ApprovalResponse:
    """Response model for approval decisions."""
    status: ApprovalStatus
    feedback: Optional[str] = None


@dataclass
class ProjectResponse:
    """Response model for project data."""
    id: int
    name: str
    description: Optional[str]
    owner_id: str
    status: str
    tags: Optional[List[str]]
    project_metadata: Optional[Dict[str, Any]]
    created_at: datetime
    updated_at: datetime
    content_count: int
    collaborator_count: int


@dataclass
class ContentItemResponse:
    """Response model for content item data."""
    id: int
    project_id: int
    name: str
    content_type: str
    file_path: Optional[str]
    thumbnail_path: Optional[str]
    generation_params: Optional[Dict[str, Any]]
    quality_metrics: Optional[Dict[str, Any]]
    tags: Optional[List[str]]
    content_metadata: Optional[Dict[str, Any]]
    created_at: datetime
    updated_at: datetime
    version_count: int
    comment_count: int


@dataclass
class CollaborationMetrics:
    """Metrics for collaboration activity."""
    total_projects: int
    active_projects: int
    total_collaborators: int
    total_comments: int
    pending_approvals: int
    recent_activity_count: int