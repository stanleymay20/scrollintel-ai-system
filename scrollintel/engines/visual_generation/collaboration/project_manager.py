"""
Project Manager for organizing generated visual content.
"""

import os
import shutil
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc, func

from scrollintel.models.collaboration_models import (
    Project, ContentItem, ContentVersion, ProjectStatus,
    ProjectCreateRequest, ProjectUpdateRequest, ProjectResponse,
    ContentItemResponse, CollaborationMetrics
)
from scrollintel.models.database_utils import get_sync_db

def get_db_session():
    """Wrapper for database session."""
    return get_sync_db()
from scrollintel.core.config import get_config


class ProjectManager:
    """Manages projects and content organization for visual generation."""
    
    def __init__(self):
        self.config = get_config()
        # Use a default storage path since it's not in the config
        self.storage_path = "storage/projects"
        os.makedirs(self.storage_path, exist_ok=True)
    
    async def create_project(self, request: ProjectCreateRequest, owner_id: str) -> ProjectResponse:
        """Create a new project."""
        with get_db_session() as db:
            # Create project directory
            project_dir = os.path.join(self.storage_path, f"project_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}")
            os.makedirs(project_dir, exist_ok=True)
            
            # Create database record
            project = Project(
                name=request.name,
                description=request.description,
                owner_id=owner_id,
                tags=request.tags or [],
                project_metadata={
                    **(request.project_metadata or {}),
                    "storage_path": project_dir
                }
            )
            
            db.add(project)
            db.commit()
            db.refresh(project)
            
            return self._project_to_response(project, db)
    
    async def get_project(self, project_id: int, user_id: str) -> Optional[ProjectResponse]:
        """Get a project by ID if user has access."""
        with get_db_session() as db:
            project = db.query(Project).filter(
                and_(
                    Project.id == project_id,
                    Project.status != ProjectStatus.DELETED.value
                )
            ).first()
            
            if not project:
                return None
            
            # Check access permissions
            if not await self._has_project_access(project_id, user_id, db):
                return None
            
            return self._project_to_response(project, db)
    
    async def update_project(self, project_id: int, request: ProjectUpdateRequest, user_id: str) -> Optional[ProjectResponse]:
        """Update a project."""
        with get_db_session() as db:
            project = db.query(Project).filter(
                and_(
                    Project.id == project_id,
                    Project.status != ProjectStatus.DELETED.value
                )
            ).first()
            
            if not project:
                return None
            
            # Check edit permissions
            if not await self._has_project_edit_access(project_id, user_id, db):
                return None
            
            # Update fields
            if request.name is not None:
                project.name = request.name
            if request.description is not None:
                project.description = request.description
            if request.tags is not None:
                project.tags = request.tags
            if request.project_metadata is not None:
                project.project_metadata = {**(project.project_metadata or {}), **request.project_metadata}
            if request.status is not None:
                project.status = request.status.value
            
            project.updated_at = datetime.utcnow()
            
            db.commit()
            db.refresh(project)
            
            return self._project_to_response(project, db)
    
    async def delete_project(self, project_id: int, user_id: str) -> bool:
        """Delete a project (soft delete)."""
        with get_db_session() as db:
            project = db.query(Project).filter(
                and_(
                    Project.id == project_id,
                    Project.owner_id == user_id,
                    Project.status != ProjectStatus.DELETED.value
                )
            ).first()
            
            if not project:
                return False
            
            project.status = ProjectStatus.DELETED.value
            project.updated_at = datetime.utcnow()
            
            db.commit()
            return True
    
    async def list_projects(self, user_id: str, page: int = 1, page_size: int = 20, 
                          search: Optional[str] = None, tags: Optional[List[str]] = None) -> Tuple[List[ProjectResponse], int]:
        """List projects accessible to the user."""
        with get_db_session() as db:
            query = db.query(Project).filter(
                and_(
                    Project.status != ProjectStatus.DELETED.value,
                    or_(
                        Project.owner_id == user_id,
                        Project.id.in_(
                            db.query(Project.id)
                            .join(Project.shares)
                            .filter(Project.shares.any(shared_with_user_id=user_id))
                        )
                    )
                )
            )
            
            # Apply search filter
            if search:
                query = query.filter(
                    or_(
                        Project.name.ilike(f"%{search}%"),
                        Project.description.ilike(f"%{search}%")
                    )
                )
            
            # Apply tag filter
            if tags:
                for tag in tags:
                    query = query.filter(Project.tags.contains([tag]))
            
            # Get total count
            total_count = query.count()
            
            # Apply pagination
            offset = (page - 1) * page_size
            projects = query.order_by(desc(Project.updated_at)).offset(offset).limit(page_size).all()
            
            # Convert to response objects
            project_responses = []
            for project in projects:
                project_responses.append(self._project_to_response(project, db))
            
            return project_responses, total_count
    
    async def add_content_item(self, project_id: int, name: str, content_type: str,
                             file_path: str, user_id: str, **kwargs) -> Optional[ContentItemResponse]:
        """Add a content item to a project."""
        with get_db_session() as db:
            # Check project access
            if not await self._has_project_edit_access(project_id, user_id, db):
                return None
            
            # Create content item
            content_item = ContentItem(
                project_id=project_id,
                name=name,
                content_type=content_type,
                file_path=file_path,
                thumbnail_path=kwargs.get("thumbnail_path"),
                generation_params=kwargs.get("generation_params"),
                quality_metrics=kwargs.get("quality_metrics"),
                tags=kwargs.get("tags", []),
                content_metadata=kwargs.get("content_metadata", {})
            )
            
            db.add(content_item)
            db.commit()
            db.refresh(content_item)
            
            # Create initial version
            version = ContentVersion(
                content_item_id=content_item.id,
                version_number=1,
                file_path=file_path,
                thumbnail_path=kwargs.get("thumbnail_path"),
                generation_params=kwargs.get("generation_params"),
                quality_metrics=kwargs.get("quality_metrics"),
                change_description="Initial version",
                created_by=user_id
            )
            
            db.add(version)
            db.commit()
            
            return self._content_item_to_response(content_item, db)
    
    async def get_content_items(self, project_id: int, user_id: str) -> List[ContentItemResponse]:
        """Get all content items in a project."""
        with get_db_session() as db:
            # Check project access
            if not await self._has_project_access(project_id, user_id, db):
                return []
            
            content_items = db.query(ContentItem).filter(
                ContentItem.project_id == project_id
            ).order_by(desc(ContentItem.created_at)).all()
            
            return [self._content_item_to_response(item, db) for item in content_items]
    
    async def organize_content(self, project_id: int, organization_rules: Dict[str, Any], user_id: str) -> bool:
        """Organize content items based on rules (tags, folders, etc.)."""
        with get_db_session() as db:
            # Check project access
            if not await self._has_project_edit_access(project_id, user_id, db):
                return False
            
            content_items = db.query(ContentItem).filter(
                ContentItem.project_id == project_id
            ).all()
            
            for item in content_items:
                # Apply organization rules
                if "auto_tag" in organization_rules:
                    auto_tags = self._generate_auto_tags(item)
                    item.tags = list(set((item.tags or []) + auto_tags))
                
                if "folder_structure" in organization_rules:
                    folder_path = self._determine_folder_path(item, organization_rules["folder_structure"])
                    if not item.content_metadata:
                        item.content_metadata = {}
                    item.content_metadata["folder_path"] = folder_path
                
                item.updated_at = datetime.utcnow()
            
            db.commit()
            return True
    
    async def search_content(self, project_id: int, query: str, user_id: str, 
                           filters: Optional[Dict[str, Any]] = None) -> List[ContentItemResponse]:
        """Search content items within a project."""
        with get_db_session() as db:
            # Check project access
            if not await self._has_project_access(project_id, user_id, db):
                return []
            
            search_query = db.query(ContentItem).filter(
                ContentItem.project_id == project_id
            )
            
            # Apply text search
            if query:
                search_query = search_query.filter(
                    or_(
                        ContentItem.name.ilike(f"%{query}%"),
                        ContentItem.tags.contains([query])
                    )
                )
            
            # Apply filters
            if filters:
                if "content_type" in filters:
                    search_query = search_query.filter(
                        ContentItem.content_type == filters["content_type"]
                    )
                
                if "tags" in filters:
                    for tag in filters["tags"]:
                        search_query = search_query.filter(
                            ContentItem.tags.contains([tag])
                        )
                
                if "date_range" in filters:
                    date_range = filters["date_range"]
                    if "start" in date_range:
                        search_query = search_query.filter(
                            ContentItem.created_at >= date_range["start"]
                        )
                    if "end" in date_range:
                        search_query = search_query.filter(
                            ContentItem.created_at <= date_range["end"]
                        )
            
            content_items = search_query.order_by(desc(ContentItem.created_at)).all()
            return [self._content_item_to_response(item, db) for item in content_items]
    
    async def get_collaboration_metrics(self, user_id: str) -> CollaborationMetrics:
        """Get collaboration metrics for the user."""
        with get_db_session() as db:
            # Get projects accessible to user
            accessible_projects = db.query(Project.id).filter(
                and_(
                    Project.status != ProjectStatus.DELETED.value,
                    or_(
                        Project.owner_id == user_id,
                        Project.id.in_(
                            db.query(Project.id)
                            .join(Project.shares)
                            .filter(Project.shares.any(shared_with_user_id=user_id))
                        )
                    )
                )
            ).subquery()
            
            # Calculate metrics
            total_projects = db.query(func.count(accessible_projects.c.id)).scalar()
            
            active_projects = db.query(func.count(Project.id)).filter(
                and_(
                    Project.id.in_(accessible_projects),
                    Project.status == ProjectStatus.ACTIVE.value
                )
            ).scalar()
            
            total_collaborators = db.query(func.count(func.distinct(Project.shares.any().shared_with_user_id))).filter(
                Project.id.in_(accessible_projects)
            ).scalar()
            
            total_comments = db.query(func.count(Project.comments.any().id)).filter(
                Project.id.in_(accessible_projects)
            ).scalar()
            
            pending_approvals = db.query(func.count(Project.approvals.any().id)).filter(
                and_(
                    Project.id.in_(accessible_projects),
                    Project.approvals.any().status == "pending"
                )
            ).scalar()
            
            # Recent activity (last 7 days)
            recent_date = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
            recent_date = recent_date.replace(day=recent_date.day - 7)
            
            recent_activity_count = db.query(func.count(Project.id)).filter(
                and_(
                    Project.id.in_(accessible_projects),
                    Project.updated_at >= recent_date
                )
            ).scalar()
            
            return CollaborationMetrics(
                total_projects=total_projects or 0,
                active_projects=active_projects or 0,
                total_collaborators=total_collaborators or 0,
                total_comments=total_comments or 0,
                pending_approvals=pending_approvals or 0,
                recent_activity_count=recent_activity_count or 0
            )
    
    def _project_to_response(self, project: Project, db: Session) -> ProjectResponse:
        """Convert Project model to response."""
        content_count = db.query(func.count(ContentItem.id)).filter(
            ContentItem.project_id == project.id
        ).scalar()
        
        collaborator_count = db.query(func.count(func.distinct(Project.shares.any().shared_with_user_id))).filter(
            Project.id == project.id
        ).scalar()
        
        return ProjectResponse(
            id=project.id,
            name=project.name,
            description=project.description,
            owner_id=project.owner_id,
            status=project.status,
            tags=project.tags,
            project_metadata=project.project_metadata,
            created_at=project.created_at,
            updated_at=project.updated_at,
            content_count=content_count or 0,
            collaborator_count=collaborator_count or 0
        )
    
    def _content_item_to_response(self, content_item: ContentItem, db: Session) -> ContentItemResponse:
        """Convert ContentItem model to response."""
        version_count = db.query(func.count(ContentVersion.id)).filter(
            ContentVersion.content_item_id == content_item.id
        ).scalar()
        
        comment_count = db.query(func.count(ContentComment.id)).filter(
            ContentComment.content_item_id == content_item.id
        ).scalar()
        
        return ContentItemResponse(
            id=content_item.id,
            project_id=content_item.project_id,
            name=content_item.name,
            content_type=content_item.content_type,
            file_path=content_item.file_path,
            thumbnail_path=content_item.thumbnail_path,
            generation_params=content_item.generation_params,
            quality_metrics=content_item.quality_metrics,
            tags=content_item.tags,
            content_metadata=content_item.content_metadata,
            created_at=content_item.created_at,
            updated_at=content_item.updated_at,
            version_count=version_count or 0,
            comment_count=comment_count or 0
        )
    
    async def _has_project_access(self, project_id: int, user_id: str, db: Session) -> bool:
        """Check if user has access to project."""
        project = db.query(Project).filter(Project.id == project_id).first()
        if not project:
            return False
        
        # Owner has access
        if project.owner_id == user_id:
            return True
        
        # Check shared access
        share = db.query(Project.shares).filter(
            and_(
                Project.shares.any().project_id == project_id,
                Project.shares.any().shared_with_user_id == user_id,
                or_(
                    Project.shares.any().expires_at.is_(None),
                    Project.shares.any().expires_at > datetime.utcnow()
                )
            )
        ).first()
        
        return share is not None
    
    async def _has_project_edit_access(self, project_id: int, user_id: str, db: Session) -> bool:
        """Check if user has edit access to project."""
        project = db.query(Project).filter(Project.id == project_id).first()
        if not project:
            return False
        
        # Owner has edit access
        if project.owner_id == user_id:
            return True
        
        # Check shared edit access
        share = db.query(Project.shares).filter(
            and_(
                Project.shares.any().project_id == project_id,
                Project.shares.any().shared_with_user_id == user_id,
                Project.shares.any().permission_level.in_(["edit", "admin"]),
                or_(
                    Project.shares.any().expires_at.is_(None),
                    Project.shares.any().expires_at > datetime.utcnow()
                )
            )
        ).first()
        
        return share is not None
    
    def _generate_auto_tags(self, content_item: ContentItem) -> List[str]:
        """Generate automatic tags based on content analysis."""
        auto_tags = []
        
        # Add content type tag
        auto_tags.append(content_item.content_type)
        
        # Add quality-based tags
        if content_item.quality_metrics:
            quality_score = content_item.quality_metrics.get("overall_score", 0)
            if quality_score > 0.8:
                auto_tags.append("high-quality")
            elif quality_score > 0.6:
                auto_tags.append("medium-quality")
            else:
                auto_tags.append("needs-improvement")
        
        # Add generation parameter tags
        if content_item.generation_params:
            style = content_item.generation_params.get("style")
            if style:
                auto_tags.append(f"style-{style}")
        
        return auto_tags
    
    def _determine_folder_path(self, content_item: ContentItem, folder_rules: Dict[str, Any]) -> str:
        """Determine folder path based on organization rules."""
        folder_path = "/"
        
        # Organize by content type
        if folder_rules.get("by_content_type"):
            folder_path = os.path.join(folder_path, content_item.content_type)
        
        # Organize by date
        if folder_rules.get("by_date"):
            date_folder = content_item.created_at.strftime("%Y/%m")
            folder_path = os.path.join(folder_path, date_folder)
        
        # Organize by quality
        if folder_rules.get("by_quality") and content_item.quality_metrics:
            quality_score = content_item.quality_metrics.get("overall_score", 0)
            if quality_score > 0.8:
                quality_folder = "high-quality"
            elif quality_score > 0.6:
                quality_folder = "medium-quality"
            else:
                quality_folder = "needs-improvement"
            folder_path = os.path.join(folder_path, quality_folder)
        
        return folder_path.replace("\\", "/")  # Normalize path separators