"""
Data models for the Advanced Prompt Management System.
"""
from datetime import datetime
from typing import List, Optional, Dict, Any
from sqlalchemy import Column, String, Text, DateTime, JSON, ForeignKey, Boolean, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
import uuid
from enum import Enum

from .database import Base


class VersionControlAction(Enum):
    """Types of version control actions."""
    CREATE = "create"
    UPDATE = "update"
    BRANCH = "branch"
    MERGE = "merge"
    TAG = "tag"
    ROLLBACK = "rollback"


class ConflictResolutionStrategy(Enum):
    """Strategies for resolving merge conflicts."""
    MANUAL = "manual"
    ACCEPT_CURRENT = "accept_current"
    ACCEPT_INCOMING = "accept_incoming"
    AUTO_MERGE = "auto_merge"


class AdvancedPromptTemplate(Base):
    """Advanced prompt template model with versioning support."""
    __tablename__ = "advanced_prompt_templates"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(255), nullable=False, index=True)
    content = Column(Text, nullable=False)
    category = Column(String(100), nullable=False, index=True)
    tags = Column(JSON, default=list)  # List of string tags
    variables = Column(JSON, default=list)  # List of PromptVariable dicts
    description = Column(Text)
    is_active = Column(Boolean, default=True)
    created_by = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    versions = relationship("AdvancedPromptVersion", back_populates="template", cascade="all, delete-orphan")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "name": self.name,
            "content": self.content,
            "category": self.category,
            "tags": self.tags or [],
            "variables": self.variables or [],
            "description": self.description,
            "is_active": self.is_active,
            "created_by": self.created_by,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }


class AdvancedPromptVersion(Base):
    """Version history for advanced prompt templates."""
    __tablename__ = "advanced_prompt_versions"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    prompt_id = Column(String, ForeignKey("advanced_prompt_templates.id"), nullable=False)
    version = Column(String(50), nullable=False)  # e.g., "1.0.0", "1.1.0"
    content = Column(Text, nullable=False)
    changes = Column(Text)  # Description of changes made
    variables = Column(JSON, default=list)  # Variables at this version
    tags = Column(JSON, default=list)  # Tags at this version
    created_by = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    template = relationship("AdvancedPromptTemplate", back_populates="versions")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "prompt_id": self.prompt_id,
            "version": self.version,
            "content": self.content,
            "changes": self.changes,
            "variables": self.variables or [],
            "tags": self.tags or [],
            "created_by": self.created_by,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }


class PromptVariable:
    """Represents a variable in a prompt template."""
    
    def __init__(self, name: str, type: str = "string", default: Optional[str] = None, 
                 description: Optional[str] = None, required: bool = True):
        self.name = name
        self.type = type  # string, number, boolean, list
        self.default = default
        self.description = description
        self.required = required
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "type": self.type,
            "default": self.default,
            "description": self.description,
            "required": self.required
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PromptVariable":
        """Create from dictionary representation."""
        return cls(
            name=data["name"],
            type=data.get("type", "string"),
            default=data.get("default"),
            description=data.get("description"),
            required=data.get("required", True)
        )


class AdvancedPromptCategory(Base):
    """Categories for organizing advanced prompt templates."""
    __tablename__ = "advanced_prompt_categories"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(100), nullable=False, unique=True, index=True)
    description = Column(Text)
    parent_id = Column(String, ForeignKey("advanced_prompt_categories.id"))
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Self-referential relationship for hierarchical categories
    parent = relationship("AdvancedPromptCategory", remote_side=[id])
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "parent_id": self.parent_id,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }


class AdvancedPromptTag(Base):
    """Tags for advanced prompt templates."""
    __tablename__ = "advanced_prompt_tags"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(50), nullable=False, unique=True, index=True)
    description = Column(Text)
    color = Column(String(7))  # Hex color code
    usage_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "color": self.color,
            "usage_count": self.usage_count,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }


class PromptBranch(Base):
    """Git-like branches for prompt development."""
    __tablename__ = "prompt_branches"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    prompt_id = Column(String, ForeignKey("advanced_prompt_templates.id"), nullable=False)
    name = Column(String(100), nullable=False)  # e.g., "main", "feature/optimization"
    description = Column(Text)
    parent_branch_id = Column(String, ForeignKey("prompt_branches.id"))
    head_version_id = Column(String, ForeignKey("advanced_prompt_versions.id"))
    is_active = Column(Boolean, default=True)
    is_protected = Column(Boolean, default=False)  # Prevent direct pushes
    created_by = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    template = relationship("AdvancedPromptTemplate")
    parent_branch = relationship("PromptBranch", remote_side=[id])
    head_version = relationship("AdvancedPromptVersion")
    commits = relationship("PromptCommit", back_populates="branch")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "prompt_id": self.prompt_id,
            "name": self.name,
            "description": self.description,
            "parent_branch_id": self.parent_branch_id,
            "head_version_id": self.head_version_id,
            "is_active": self.is_active,
            "is_protected": self.is_protected,
            "created_by": self.created_by,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }


class PromptCommit(Base):
    """Git-like commits for prompt changes."""
    __tablename__ = "prompt_commits"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    branch_id = Column(String, ForeignKey("prompt_branches.id"), nullable=False)
    version_id = Column(String, ForeignKey("advanced_prompt_versions.id"), nullable=False)
    parent_commit_id = Column(String, ForeignKey("prompt_commits.id"))
    commit_hash = Column(String(40), nullable=False, unique=True)  # SHA-like hash
    message = Column(Text, nullable=False)
    author = Column(String(255), nullable=False)
    committer = Column(String(255), nullable=False)
    committed_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    branch = relationship("PromptBranch", back_populates="commits")
    version = relationship("AdvancedPromptVersion")
    parent_commit = relationship("PromptCommit", remote_side=[id])
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "branch_id": self.branch_id,
            "version_id": self.version_id,
            "parent_commit_id": self.parent_commit_id,
            "commit_hash": self.commit_hash,
            "message": self.message,
            "author": self.author,
            "committer": self.committer,
            "committed_at": self.committed_at.isoformat() if self.committed_at else None
        }


class PromptMergeRequest(Base):
    """Pull/merge requests for prompt changes."""
    __tablename__ = "prompt_merge_requests"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    source_branch_id = Column(String, ForeignKey("prompt_branches.id"), nullable=False)
    target_branch_id = Column(String, ForeignKey("prompt_branches.id"), nullable=False)
    title = Column(String(255), nullable=False)
    description = Column(Text)
    status = Column(String(20), default="open")  # open, merged, closed, draft
    author = Column(String(255), nullable=False)
    assignee = Column(String(255))
    reviewers = Column(JSON, default=list)  # List of reviewer usernames
    conflicts = Column(JSON, default=list)  # List of conflict descriptions
    auto_merge_enabled = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    merged_at = Column(DateTime)
    merged_by = Column(String(255))
    
    # Relationships
    source_branch = relationship("PromptBranch", foreign_keys=[source_branch_id])
    target_branch = relationship("PromptBranch", foreign_keys=[target_branch_id])
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "source_branch_id": self.source_branch_id,
            "target_branch_id": self.target_branch_id,
            "title": self.title,
            "description": self.description,
            "status": self.status,
            "author": self.author,
            "assignee": self.assignee,
            "reviewers": self.reviewers or [],
            "conflicts": self.conflicts or [],
            "auto_merge_enabled": self.auto_merge_enabled,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "merged_at": self.merged_at.isoformat() if self.merged_at else None,
            "merged_by": self.merged_by
        }


class PromptVersionTag(Base):
    """Tags for marking specific prompt versions."""
    __tablename__ = "prompt_version_tags"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    version_id = Column(String, ForeignKey("advanced_prompt_versions.id"), nullable=False)
    name = Column(String(100), nullable=False)  # e.g., "v1.0", "stable", "production"
    description = Column(Text)
    tag_type = Column(String(20), default="release")  # release, hotfix, feature
    created_by = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    version = relationship("AdvancedPromptVersion")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "version_id": self.version_id,
            "name": self.name,
            "description": self.description,
            "tag_type": self.tag_type,
            "created_by": self.created_by,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }