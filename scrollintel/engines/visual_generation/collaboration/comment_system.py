"""
Comment System for project and content collaboration.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import and_, desc, func

from scrollintel.models.collaboration_models import (
    Project, ContentItem, ProjectComment, ContentComment, CommentRequest
)
from scrollintel.models.database_utils import get_sync_db

def get_db_session():
    """Wrapper for database session."""
    return get_sync_db()


class CommentSystem:
    """Manages comments on projects and content items."""
    
    def __init__(self):
        pass
    
    async def add_project_comment(self, project_id: int, comment_request: CommentRequest,
                                user_id: str) -> Optional[Dict[str, Any]]:
        """Add a comment to a project."""
        with get_db_session() as db:
            # Check if user has access to the project
            if not await self._has_project_access(project_id, user_id, db):
                return None
            
            # Validate parent comment if specified
            if comment_request.parent_comment_id:
                parent_comment = db.query(ProjectComment).filter(
                    and_(
                        ProjectComment.id == comment_request.parent_comment_id,
                        ProjectComment.project_id == project_id
                    )
                ).first()
                
                if not parent_comment:
                    return None
            
            # Create comment
            comment = ProjectComment(
                project_id=project_id,
                user_id=user_id,
                content=comment_request.content,
                parent_comment_id=comment_request.parent_comment_id
            )
            
            db.add(comment)
            db.commit()
            db.refresh(comment)
            
            # Send notifications to project collaborators
            await self._notify_project_collaborators(project_id, user_id, "comment_added", db)
            
            return self._comment_to_dict(comment)
    
    async def add_content_comment(self, content_item_id: int, comment_request: CommentRequest,
                                user_id: str) -> Optional[Dict[str, Any]]:
        """Add a comment to a content item."""
        with get_db_session() as db:
            # Get content item and check project access
            content_item = db.query(ContentItem).filter(
                ContentItem.id == content_item_id
            ).first()
            
            if not content_item:
                return None
            
            if not await self._has_project_access(content_item.project_id, user_id, db):
                return None
            
            # Validate parent comment if specified
            if comment_request.parent_comment_id:
                parent_comment = db.query(ContentComment).filter(
                    and_(
                        ContentComment.id == comment_request.parent_comment_id,
                        ContentComment.content_item_id == content_item_id
                    )
                ).first()
                
                if not parent_comment:
                    return None
            
            # Create comment
            comment = ContentComment(
                content_item_id=content_item_id,
                user_id=user_id,
                content=comment_request.content,
                position_x=comment_request.position_x,
                position_y=comment_request.position_y,
                timestamp=comment_request.timestamp,
                parent_comment_id=comment_request.parent_comment_id
            )
            
            db.add(comment)
            db.commit()
            db.refresh(comment)
            
            # Send notifications to project collaborators
            await self._notify_project_collaborators(content_item.project_id, user_id, "content_comment_added", db)
            
            return self._content_comment_to_dict(comment)
    
    async def get_project_comments(self, project_id: int, user_id: str,
                                 page: int = 1, page_size: int = 50) -> List[Dict[str, Any]]:
        """Get comments for a project."""
        with get_db_session() as db:
            # Check if user has access to the project
            if not await self._has_project_access(project_id, user_id, db):
                return []
            
            # Get comments with pagination
            offset = (page - 1) * page_size
            comments = db.query(ProjectComment).filter(
                ProjectComment.project_id == project_id
            ).order_by(desc(ProjectComment.created_at)).offset(offset).limit(page_size).all()
            
            # Build comment tree
            comment_dict = {}
            root_comments = []
            
            for comment in comments:
                comment_data = self._comment_to_dict(comment)
                comment_data["replies"] = []
                comment_dict[comment.id] = comment_data
                
                if comment.parent_comment_id:
                    # This is a reply
                    if comment.parent_comment_id in comment_dict:
                        comment_dict[comment.parent_comment_id]["replies"].append(comment_data)
                else:
                    # This is a root comment
                    root_comments.append(comment_data)
            
            return root_comments
    
    async def get_content_comments(self, content_item_id: int, user_id: str) -> List[Dict[str, Any]]:
        """Get comments for a content item."""
        with get_db_session() as db:
            # Get content item and check project access
            content_item = db.query(ContentItem).filter(
                ContentItem.id == content_item_id
            ).first()
            
            if not content_item:
                return []
            
            if not await self._has_project_access(content_item.project_id, user_id, db):
                return []
            
            # Get comments
            comments = db.query(ContentComment).filter(
                ContentComment.content_item_id == content_item_id
            ).order_by(desc(ContentComment.created_at)).all()
            
            # Build comment tree
            comment_dict = {}
            root_comments = []
            
            for comment in comments:
                comment_data = self._content_comment_to_dict(comment)
                comment_data["replies"] = []
                comment_dict[comment.id] = comment_data
                
                if comment.parent_comment_id:
                    # This is a reply
                    if comment.parent_comment_id in comment_dict:
                        comment_dict[comment.parent_comment_id]["replies"].append(comment_data)
                else:
                    # This is a root comment
                    root_comments.append(comment_data)
            
            return root_comments
    
    async def update_comment(self, comment_id: int, new_content: str, user_id: str,
                           comment_type: str = "project") -> bool:
        """Update a comment."""
        with get_db_session() as db:
            if comment_type == "project":
                comment = db.query(ProjectComment).filter(
                    and_(
                        ProjectComment.id == comment_id,
                        ProjectComment.user_id == user_id
                    )
                ).first()
            else:
                comment = db.query(ContentComment).filter(
                    and_(
                        ContentComment.id == comment_id,
                        ContentComment.user_id == user_id
                    )
                ).first()
            
            if not comment:
                return False
            
            comment.content = new_content
            comment.updated_at = datetime.utcnow()
            
            db.commit()
            return True
    
    async def delete_comment(self, comment_id: int, user_id: str,
                           comment_type: str = "project") -> bool:
        """Delete a comment."""
        with get_db_session() as db:
            if comment_type == "project":
                comment = db.query(ProjectComment).filter(
                    and_(
                        ProjectComment.id == comment_id,
                        ProjectComment.user_id == user_id
                    )
                ).first()
            else:
                comment = db.query(ContentComment).filter(
                    and_(
                        ContentComment.id == comment_id,
                        ContentComment.user_id == user_id
                    )
                ).first()
            
            if not comment:
                return False
            
            # Delete comment and all replies
            if comment_type == "project":
                # Delete replies first
                db.query(ProjectComment).filter(
                    ProjectComment.parent_comment_id == comment_id
                ).delete()
                
                # Delete the comment
                db.delete(comment)
            else:
                # Delete replies first
                db.query(ContentComment).filter(
                    ContentComment.parent_comment_id == comment_id
                ).delete()
                
                # Delete the comment
                db.delete(comment)
            
            db.commit()
            return True
    
    async def get_comment_statistics(self, project_id: int, user_id: str) -> Dict[str, Any]:
        """Get comment statistics for a project."""
        with get_db_session() as db:
            # Check if user has access to the project
            if not await self._has_project_access(project_id, user_id, db):
                return {}
            
            # Get project comment count
            project_comment_count = db.query(func.count(ProjectComment.id)).filter(
                ProjectComment.project_id == project_id
            ).scalar()
            
            # Get content comment count
            content_comment_count = db.query(func.count(ContentComment.id)).join(
                ContentItem, ContentComment.content_item_id == ContentItem.id
            ).filter(
                ContentItem.project_id == project_id
            ).scalar()
            
            # Get recent activity (last 7 days)
            recent_date = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
            recent_date = recent_date.replace(day=recent_date.day - 7)
            
            recent_project_comments = db.query(func.count(ProjectComment.id)).filter(
                and_(
                    ProjectComment.project_id == project_id,
                    ProjectComment.created_at >= recent_date
                )
            ).scalar()
            
            recent_content_comments = db.query(func.count(ContentComment.id)).join(
                ContentItem, ContentComment.content_item_id == ContentItem.id
            ).filter(
                and_(
                    ContentItem.project_id == project_id,
                    ContentComment.created_at >= recent_date
                )
            ).scalar()
            
            # Get top commenters
            top_commenters = db.query(
                ProjectComment.user_id,
                func.count(ProjectComment.id).label('comment_count')
            ).filter(
                ProjectComment.project_id == project_id
            ).group_by(ProjectComment.user_id).order_by(
                desc('comment_count')
            ).limit(5).all()
            
            return {
                "total_project_comments": project_comment_count or 0,
                "total_content_comments": content_comment_count or 0,
                "total_comments": (project_comment_count or 0) + (content_comment_count or 0),
                "recent_project_comments": recent_project_comments or 0,
                "recent_content_comments": recent_content_comments or 0,
                "recent_total_comments": (recent_project_comments or 0) + (recent_content_comments or 0),
                "top_commenters": [
                    {"user_id": user_id, "comment_count": count}
                    for user_id, count in top_commenters
                ]
            }
    
    async def search_comments(self, project_id: int, query: str, user_id: str,
                            content_item_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """Search comments within a project."""
        with get_db_session() as db:
            # Check if user has access to the project
            if not await self._has_project_access(project_id, user_id, db):
                return []
            
            results = []
            
            # Search project comments
            if not content_item_id:
                project_comments = db.query(ProjectComment).filter(
                    and_(
                        ProjectComment.project_id == project_id,
                        ProjectComment.content.ilike(f"%{query}%")
                    )
                ).order_by(desc(ProjectComment.created_at)).all()
                
                for comment in project_comments:
                    comment_data = self._comment_to_dict(comment)
                    comment_data["type"] = "project"
                    results.append(comment_data)
            
            # Search content comments
            content_query = db.query(ContentComment).join(
                ContentItem, ContentComment.content_item_id == ContentItem.id
            ).filter(
                and_(
                    ContentItem.project_id == project_id,
                    ContentComment.content.ilike(f"%{query}%")
                )
            )
            
            if content_item_id:
                content_query = content_query.filter(
                    ContentComment.content_item_id == content_item_id
                )
            
            content_comments = content_query.order_by(desc(ContentComment.created_at)).all()
            
            for comment in content_comments:
                comment_data = self._content_comment_to_dict(comment)
                comment_data["type"] = "content"
                results.append(comment_data)
            
            # Sort all results by creation date
            results.sort(key=lambda x: x["created_at"], reverse=True)
            
            return results
    
    async def get_user_mentions(self, user_id: str) -> List[Dict[str, Any]]:
        """Get comments that mention the user."""
        with get_db_session() as db:
            mentions = []
            
            # Search for mentions in project comments
            project_comments = db.query(ProjectComment).filter(
                ProjectComment.content.ilike(f"%@{user_id}%")
            ).order_by(desc(ProjectComment.created_at)).limit(50).all()
            
            for comment in project_comments:
                # Check if user has access to the project
                if await self._has_project_access(comment.project_id, user_id, db):
                    comment_data = self._comment_to_dict(comment)
                    comment_data["type"] = "project"
                    comment_data["project_id"] = comment.project_id
                    mentions.append(comment_data)
            
            # Search for mentions in content comments
            content_comments = db.query(ContentComment).filter(
                ContentComment.content.ilike(f"%@{user_id}%")
            ).order_by(desc(ContentComment.created_at)).limit(50).all()
            
            for comment in content_comments:
                content_item = db.query(ContentItem).filter(
                    ContentItem.id == comment.content_item_id
                ).first()
                
                if content_item and await self._has_project_access(content_item.project_id, user_id, db):
                    comment_data = self._content_comment_to_dict(comment)
                    comment_data["type"] = "content"
                    comment_data["project_id"] = content_item.project_id
                    comment_data["content_item_name"] = content_item.name
                    mentions.append(comment_data)
            
            # Sort by creation date
            mentions.sort(key=lambda x: x["created_at"], reverse=True)
            
            return mentions[:50]  # Limit to 50 most recent mentions
    
    def _comment_to_dict(self, comment: ProjectComment) -> Dict[str, Any]:
        """Convert ProjectComment to dictionary."""
        return {
            "id": comment.id,
            "user_id": comment.user_id,
            "content": comment.content,
            "parent_comment_id": comment.parent_comment_id,
            "created_at": comment.created_at,
            "updated_at": comment.updated_at
        }
    
    def _content_comment_to_dict(self, comment: ContentComment) -> Dict[str, Any]:
        """Convert ContentComment to dictionary."""
        return {
            "id": comment.id,
            "content_item_id": comment.content_item_id,
            "user_id": comment.user_id,
            "content": comment.content,
            "position_x": comment.position_x,
            "position_y": comment.position_y,
            "timestamp": comment.timestamp,
            "parent_comment_id": comment.parent_comment_id,
            "created_at": comment.created_at,
            "updated_at": comment.updated_at
        }
    
    async def _has_project_access(self, project_id: int, user_id: str, db: Session) -> bool:
        """Check if user has access to project."""
        from scrollintel.engines.visual_generation.collaboration.sharing_manager import SharingManager
        sharing_manager = SharingManager()
        return await sharing_manager._has_project_access(project_id, user_id, db)
    
    async def _notify_project_collaborators(self, project_id: int, commenter_id: str, 
                                          event_type: str, db: Session):
        """Send notifications to project collaborators (placeholder)."""
        # In a real implementation, this would send notifications to all collaborators
        print(f"Notification: {event_type} by {commenter_id} in project {project_id}")