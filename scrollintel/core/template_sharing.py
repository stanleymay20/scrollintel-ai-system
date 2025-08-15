"""
Template sharing and collaboration system for dashboard templates.
"""
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import json
import uuid
from sqlalchemy import Column, String, DateTime, Text, Boolean, Integer
from sqlalchemy.orm import Session
from scrollintel.models.database import Base, get_db


class SharePermission(Enum):
    VIEW = "view"
    EDIT = "edit"
    ADMIN = "admin"


class CollaborationRole(Enum):
    VIEWER = "viewer"
    EDITOR = "editor"
    MAINTAINER = "maintainer"
    OWNER = "owner"


class TemplateShare(Base):
    """Database model for template sharing."""
    __tablename__ = "template_shares"
    
    id = Column(String, primary_key=True)
    template_id = Column(String, nullable=False, index=True)
    share_token = Column(String, unique=True, nullable=False)
    created_by = Column(String, nullable=False)
    permission = Column(String, nullable=False)
    expires_at = Column(DateTime, nullable=True)
    is_public = Column(Boolean, default=False)
    access_count = Column(Integer, default=0)
    max_access_count = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)


class TemplateCollaborator(Base):
    """Database model for template collaborators."""
    __tablename__ = "template_collaborators"
    
    id = Column(String, primary_key=True)
    template_id = Column(String, nullable=False, index=True)
    user_id = Column(String, nullable=False)
    role = Column(String, nullable=False)
    invited_by = Column(String, nullable=False)
    joined_at = Column(DateTime, default=datetime.utcnow)
    last_activity = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)


class TemplateComment(Base):
    """Database model for template comments."""
    __tablename__ = "template_comments"
    
    id = Column(String, primary_key=True)
    template_id = Column(String, nullable=False, index=True)
    user_id = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    parent_comment_id = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)
    is_resolved = Column(Boolean, default=False)
    is_active = Column(Boolean, default=True)


class TemplateSharingManager:
    """Manager for template sharing and collaboration."""
    
    def __init__(self):
        self.db_session = None
    
    def _get_db_session(self) -> Session:
        """Get database session."""
        if not self.db_session:
            self.db_session = next(get_db())
        return self.db_session
    
    def create_share_link(
        self,
        template_id: str,
        created_by: str,
        permission: SharePermission,
        expires_at: Optional[datetime] = None,
        is_public: bool = False,
        max_access_count: Optional[int] = None
    ) -> Dict[str, Any]:
        """Create a shareable link for a template."""
        db = self._get_db_session()
        
        share_token = f"share_{uuid.uuid4().hex}"
        share_id = f"sharelink_{uuid.uuid4().hex[:12]}"
        
        share_record = TemplateShare(
            id=share_id,
            template_id=template_id,
            share_token=share_token,
            created_by=created_by,
            permission=permission.value,
            expires_at=expires_at,
            is_public=is_public,
            max_access_count=max_access_count
        )
        
        db.add(share_record)
        db.commit()
        
        return {
            "id": share_id,
            "share_token": share_token,
            "template_id": template_id,
            "permission": permission.value,
            "expires_at": expires_at.isoformat() if expires_at else None,
            "is_public": is_public,
            "created_by": created_by
        }
    
    def get_share_link(self, share_token: str) -> Optional[Dict[str, Any]]:
        """Get share link by token."""
        db = self._get_db_session()
        
        share_record = db.query(TemplateShare)\
            .filter(
                TemplateShare.share_token == share_token,
                TemplateShare.is_active == True
            )\
            .first()
        
        if not share_record:
            return None
        
        # Check if expired
        if share_record.expires_at and share_record.expires_at < datetime.utcnow():
            return None
        
        # Check access count limit
        if (share_record.max_access_count and 
            share_record.access_count >= share_record.max_access_count):
            return None
        
        return {
            "id": share_record.id,
            "template_id": share_record.template_id,
            "permission": share_record.permission,
            "access_count": share_record.access_count,
            "created_by": share_record.created_by
        }
    
    def access_shared_template(self, share_token: str, accessed_by: str) -> Optional[Dict[str, Any]]:
        """Access a shared template and increment access count."""
        db = self._get_db_session()
        
        share_link = self.get_share_link(share_token)
        if not share_link:
            return None
        
        # Increment access count
        share_record = db.query(TemplateShare)\
            .filter(TemplateShare.share_token == share_token)\
            .first()
        
        if share_record:
            share_record.access_count += 1
            db.commit()
        
        return {
            "template_id": share_link["template_id"],
            "permission": share_link["permission"],
            "accessed_by": accessed_by,
            "access_count": share_link["access_count"] + 1
        }
    
    def add_collaborator(
        self,
        template_id: str,
        user_id: str,
        role: CollaborationRole,
        invited_by: str
    ) -> bool:
        """Add a collaborator to a template."""
        db = self._get_db_session()
        
        # Check if already a collaborator
        existing = db.query(TemplateCollaborator)\
            .filter(
                TemplateCollaborator.template_id == template_id,
                TemplateCollaborator.user_id == user_id,
                TemplateCollaborator.is_active == True
            )\
            .first()
        
        if existing:
            # Update role
            existing.role = role.value
            existing.last_activity = datetime.utcnow()
        else:
            # Add new collaborator
            collaborator = TemplateCollaborator(
                id=f"collab_{uuid.uuid4().hex[:12]}",
                template_id=template_id,
                user_id=user_id,
                role=role.value,
                invited_by=invited_by
            )
            db.add(collaborator)
        
        db.commit()
        return True
    
    def get_template_collaborators(self, template_id: str) -> List[Dict[str, Any]]:
        """Get all collaborators for a template."""
        db = self._get_db_session()
        
        collaborators = db.query(TemplateCollaborator)\
            .filter(
                TemplateCollaborator.template_id == template_id,
                TemplateCollaborator.is_active == True
            )\
            .all()
        
        return [
            {
                "id": collab.id,
                "user_id": collab.user_id,
                "role": collab.role,
                "invited_by": collab.invited_by,
                "joined_at": collab.joined_at.isoformat(),
                "last_activity": collab.last_activity.isoformat()
            }
            for collab in collaborators
        ]
    
    def add_comment(
        self,
        template_id: str,
        user_id: str,
        content: str,
        parent_comment_id: Optional[str] = None
    ) -> str:
        """Add a comment to a template."""
        db = self._get_db_session()
        
        comment_id = f"comment_{uuid.uuid4().hex[:12]}"
        
        comment = TemplateComment(
            id=comment_id,
            template_id=template_id,
            user_id=user_id,
            content=content,
            parent_comment_id=parent_comment_id
        )
        
        db.add(comment)
        db.commit()
        
        return comment_id
    
    def get_template_comments(self, template_id: str) -> List[Dict[str, Any]]:
        """Get all comments for a template."""
        db = self._get_db_session()
        
        comments = db.query(TemplateComment)\
            .filter(
                TemplateComment.template_id == template_id,
                TemplateComment.is_active == True
            )\
            .order_by(TemplateComment.created_at.asc())\
            .all()
        
        return [
            {
                "id": comment.id,
                "user_id": comment.user_id,
                "content": comment.content,
                "parent_comment_id": comment.parent_comment_id,
                "created_at": comment.created_at.isoformat(),
                "is_resolved": comment.is_resolved
            }
            for comment in comments
        ]
    
    def get_public_templates(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get publicly shared templates."""
        db = self._get_db_session()
        
        public_shares = db.query(TemplateShare)\
            .filter(
                TemplateShare.is_public == True,
                TemplateShare.is_active == True
            )\
            .limit(limit)\
            .all()
        
        return [
            {
                "template_id": share.template_id,
                "share_token": share.share_token,
                "permission": share.permission,
                "created_by": share.created_by,
                "access_count": share.access_count,
                "created_at": share.created_at.isoformat()
            }
            for share in public_shares
        ]


# Global sharing manager instance
sharing_manager = TemplateSharingManager()