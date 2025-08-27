"""
Template Sharing and Collaboration System

Provides sharing, collaboration, and permission management for dashboard templates.
"""

from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import uuid
from enum import Enum

class SharePermission(Enum):
    VIEW = "view"
    EDIT = "edit"
    ADMIN = "admin"
    COMMENT = "comment"

class ShareScope(Enum):
    PRIVATE = "private"
    ORGANIZATION = "organization"
    PUBLIC = "public"
    CUSTOM = "custom"

@dataclass
class ShareLink:
    """Shareable link for templates"""
    link_id: str
    template_id: str
    created_by: str
    permission: SharePermission
    scope: ShareScope
    expires_at: Optional[datetime]
    access_count: int = 0
    max_access_count: Optional[int] = None
    created_at: datetime = None
    is_active: bool = True
    allowed_users: Optional[List[str]] = None
    allowed_domains: Optional[List[str]] = None

@dataclass
class TemplateComment:
    """Comment on template"""
    comment_id: str
    template_id: str
    author: str
    content: str
    widget_id: Optional[str] = None  # Comment on specific widget
    position: Optional[Dict[str, int]] = None  # Position for annotation
    created_at: datetime = None
    updated_at: Optional[datetime] = None
    is_resolved: bool = False
    parent_comment_id: Optional[str] = None
    reactions: Dict[str, List[str]] = None  # reaction_type -> list of users

@dataclass
class CollaborationSession:
    """Real-time collaboration session"""
    session_id: str
    template_id: str
    participants: List[str]
    created_by: str
    created_at: datetime
    last_activity: datetime
    is_active: bool = True
    current_editors: Set[str] = None
    locked_widgets: Dict[str, str] = None  # widget_id -> user_id

@dataclass
class TemplatePermission:
    """User permissions for template"""
    template_id: str
    user_id: str
    permission: SharePermission
    granted_by: str
    granted_at: datetime
    expires_at: Optional[datetime] = None

class TemplateSharingManager:
    """Manager for template sharing and collaboration"""
    
    def __init__(self):
        self.share_links: Dict[str, ShareLink] = {}
        self.comments: Dict[str, List[TemplateComment]] = {}  # template_id -> comments
        self.permissions: Dict[str, List[TemplatePermission]] = {}  # template_id -> permissions
        self.collaboration_sessions: Dict[str, CollaborationSession] = {}
        self.user_favorites: Dict[str, Set[str]] = {}  # user_id -> template_ids
    
    def create_share_link(self, 
                         template_id: str, 
                         created_by: str,
                         permission: SharePermission = SharePermission.VIEW,
                         scope: ShareScope = ShareScope.ORGANIZATION,
                         expires_in_days: Optional[int] = None,
                         max_access_count: Optional[int] = None,
                         allowed_users: Optional[List[str]] = None,
                         allowed_domains: Optional[List[str]] = None) -> str:
        """Create shareable link for template"""
        
        link_id = str(uuid.uuid4())
        expires_at = None
        
        if expires_in_days:
            expires_at = datetime.now() + timedelta(days=expires_in_days)
        
        share_link = ShareLink(
            link_id=link_id,
            template_id=template_id,
            created_by=created_by,
            permission=permission,
            scope=scope,
            expires_at=expires_at,
            max_access_count=max_access_count,
            created_at=datetime.now(),
            allowed_users=allowed_users or [],
            allowed_domains=allowed_domains or []
        )
        
        self.share_links[link_id] = share_link
        return link_id
    
    def access_shared_template(self, link_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        """Access template via share link"""
        
        share_link = self.share_links.get(link_id)
        if not share_link or not share_link.is_active:
            return None
        
        # Check expiration
        if share_link.expires_at and datetime.now() > share_link.expires_at:
            share_link.is_active = False
            return None
        
        # Check access count limit
        if (share_link.max_access_count and 
            share_link.access_count >= share_link.max_access_count):
            return None
        
        # Check user permissions
        if not self._check_share_access(share_link, user_id):
            return None
        
        # Increment access count
        share_link.access_count += 1
        
        return {
            "template_id": share_link.template_id,
            "permission": share_link.permission.value,
            "link_id": link_id
        }
    
    def grant_template_permission(self, 
                                 template_id: str,
                                 user_id: str,
                                 permission: SharePermission,
                                 granted_by: str,
                                 expires_in_days: Optional[int] = None) -> bool:
        """Grant direct permission to user for template"""
        
        expires_at = None
        if expires_in_days:
            expires_at = datetime.now() + timedelta(days=expires_in_days)
        
        template_permission = TemplatePermission(
            template_id=template_id,
            user_id=user_id,
            permission=permission,
            granted_by=granted_by,
            granted_at=datetime.now(),
            expires_at=expires_at
        )
        
        if template_id not in self.permissions:
            self.permissions[template_id] = []
        
        # Remove existing permission for user
        self.permissions[template_id] = [
            p for p in self.permissions[template_id] if p.user_id != user_id
        ]
        
        self.permissions[template_id].append(template_permission)
        return True
    
    def revoke_template_permission(self, template_id: str, user_id: str) -> bool:
        """Revoke user permission for template"""
        
        if template_id not in self.permissions:
            return False
        
        original_count = len(self.permissions[template_id])
        self.permissions[template_id] = [
            p for p in self.permissions[template_id] if p.user_id != user_id
        ]
        
        return len(self.permissions[template_id]) < original_count
    
    def get_user_permission(self, template_id: str, user_id: str) -> Optional[SharePermission]:
        """Get user's permission level for template"""
        
        if template_id not in self.permissions:
            return None
        
        for permission in self.permissions[template_id]:
            if permission.user_id == user_id:
                # Check if permission is expired
                if (permission.expires_at and 
                    datetime.now() > permission.expires_at):
                    return None
                return permission.permission
        
        return None
    
    def add_comment(self, 
                   template_id: str,
                   author: str,
                   content: str,
                   widget_id: Optional[str] = None,
                   position: Optional[Dict[str, int]] = None,
                   parent_comment_id: Optional[str] = None) -> str:
        """Add comment to template"""
        
        comment_id = str(uuid.uuid4())
        
        comment = TemplateComment(
            comment_id=comment_id,
            template_id=template_id,
            author=author,
            content=content,
            widget_id=widget_id,
            position=position,
            created_at=datetime.now(),
            parent_comment_id=parent_comment_id,
            reactions={}
        )
        
        if template_id not in self.comments:
            self.comments[template_id] = []
        
        self.comments[template_id].append(comment)
        return comment_id
    
    def update_comment(self, comment_id: str, content: str) -> bool:
        """Update comment content"""
        
        for template_comments in self.comments.values():
            for comment in template_comments:
                if comment.comment_id == comment_id:
                    comment.content = content
                    comment.updated_at = datetime.now()
                    return True
        
        return False
    
    def resolve_comment(self, comment_id: str) -> bool:
        """Mark comment as resolved"""
        
        for template_comments in self.comments.values():
            for comment in template_comments:
                if comment.comment_id == comment_id:
                    comment.is_resolved = True
                    return True
        
        return False
    
    def add_reaction(self, comment_id: str, user_id: str, reaction_type: str) -> bool:
        """Add reaction to comment"""
        
        for template_comments in self.comments.values():
            for comment in template_comments:
                if comment.comment_id == comment_id:
                    if reaction_type not in comment.reactions:
                        comment.reactions[reaction_type] = []
                    
                    if user_id not in comment.reactions[reaction_type]:
                        comment.reactions[reaction_type].append(user_id)
                    
                    return True
        
        return False
    
    def get_template_comments(self, template_id: str) -> List[TemplateComment]:
        """Get all comments for template"""
        return self.comments.get(template_id, [])
    
    def start_collaboration_session(self, template_id: str, created_by: str) -> str:
        """Start real-time collaboration session"""
        
        session_id = str(uuid.uuid4())
        
        session = CollaborationSession(
            session_id=session_id,
            template_id=template_id,
            participants=[created_by],
            created_by=created_by,
            created_at=datetime.now(),
            last_activity=datetime.now(),
            current_editors=set(),
            locked_widgets={}
        )
        
        self.collaboration_sessions[session_id] = session
        return session_id
    
    def join_collaboration_session(self, session_id: str, user_id: str) -> bool:
        """Join existing collaboration session"""
        
        session = self.collaboration_sessions.get(session_id)
        if not session or not session.is_active:
            return False
        
        if user_id not in session.participants:
            session.participants.append(user_id)
        
        session.last_activity = datetime.now()
        return True
    
    def lock_widget_for_editing(self, session_id: str, widget_id: str, user_id: str) -> bool:
        """Lock widget for exclusive editing"""
        
        session = self.collaboration_sessions.get(session_id)
        if not session or not session.is_active:
            return False
        
        # Check if widget is already locked
        if widget_id in session.locked_widgets:
            return session.locked_widgets[widget_id] == user_id
        
        session.locked_widgets[widget_id] = user_id
        session.current_editors.add(user_id)
        session.last_activity = datetime.now()
        
        return True
    
    def unlock_widget(self, session_id: str, widget_id: str, user_id: str) -> bool:
        """Unlock widget after editing"""
        
        session = self.collaboration_sessions.get(session_id)
        if not session or not session.is_active:
            return False
        
        if (widget_id in session.locked_widgets and 
            session.locked_widgets[widget_id] == user_id):
            del session.locked_widgets[widget_id]
            
            # Remove from current editors if no more locked widgets
            user_locked_widgets = [w for w, u in session.locked_widgets.items() if u == user_id]
            if not user_locked_widgets:
                session.current_editors.discard(user_id)
            
            session.last_activity = datetime.now()
            return True
        
        return False
    
    def end_collaboration_session(self, session_id: str) -> bool:
        """End collaboration session"""
        
        session = self.collaboration_sessions.get(session_id)
        if not session:
            return False
        
        session.is_active = False
        session.locked_widgets.clear()
        session.current_editors.clear()
        
        return True
    
    def add_to_favorites(self, user_id: str, template_id: str) -> bool:
        """Add template to user's favorites"""
        
        if user_id not in self.user_favorites:
            self.user_favorites[user_id] = set()
        
        self.user_favorites[user_id].add(template_id)
        return True
    
    def remove_from_favorites(self, user_id: str, template_id: str) -> bool:
        """Remove template from user's favorites"""
        
        if user_id in self.user_favorites:
            self.user_favorites[user_id].discard(template_id)
            return True
        
        return False
    
    def get_user_favorites(self, user_id: str) -> List[str]:
        """Get user's favorite templates"""
        return list(self.user_favorites.get(user_id, set()))
    
    def get_template_share_links(self, template_id: str) -> List[ShareLink]:
        """Get all share links for template"""
        return [link for link in self.share_links.values() 
                if link.template_id == template_id and link.is_active]
    
    def get_template_permissions(self, template_id: str) -> List[TemplatePermission]:
        """Get all permissions for template"""
        return self.permissions.get(template_id, [])
    
    def _check_share_access(self, share_link: ShareLink, user_id: str) -> bool:
        """Check if user can access shared template"""
        
        if share_link.scope == ShareScope.PUBLIC:
            return True
        
        if share_link.scope == ShareScope.PRIVATE:
            return user_id in (share_link.allowed_users or [])
        
        if share_link.scope == ShareScope.CUSTOM:
            # Check allowed users
            if share_link.allowed_users and user_id in share_link.allowed_users:
                return True
            
            # Check allowed domains (assuming user_id contains email)
            if share_link.allowed_domains and '@' in user_id:
                user_domain = user_id.split('@')[1]
                return user_domain in share_link.allowed_domains
        
        # For organization scope, would need organization membership check
        # This would be implemented based on your organization structure
        
        return False