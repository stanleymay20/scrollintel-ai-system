"""
Version Control System for content items and revision history.
"""

import os
import shutil
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import and_, desc, func

from scrollintel.models.collaboration_models import (
    ContentItem, ContentVersion
)
from scrollintel.models.database_utils import get_sync_db

def get_db_session():
    """Wrapper for database session."""
    return get_sync_db()
from scrollintel.core.config import get_config


class VersionControl:
    """Manages version control and revision history for content items."""
    
    def __init__(self):
        self.config = get_config()
        # Use a default storage path since it's not in the config
        self.storage_path = "storage/versions"
        os.makedirs(self.storage_path, exist_ok=True)
    
    async def create_version(self, content_item_id: int, file_path: str, user_id: str,
                           change_description: str, **kwargs) -> Optional[Dict[str, Any]]:
        """Create a new version of a content item."""
        with get_db_session() as db:
            # Get content item
            content_item = db.query(ContentItem).filter(
                ContentItem.id == content_item_id
            ).first()
            
            if not content_item:
                return None
            
            # Check if user has edit access to the project
            if not await self._has_project_edit_access(content_item.project_id, user_id, db):
                return None
            
            # Get next version number
            latest_version = db.query(func.max(ContentVersion.version_number)).filter(
                ContentVersion.content_item_id == content_item_id
            ).scalar()
            
            next_version = (latest_version or 0) + 1
            
            # Create version directory
            version_dir = os.path.join(
                self.storage_path,
                f"content_{content_item_id}",
                f"v{next_version}"
            )
            os.makedirs(version_dir, exist_ok=True)
            
            # Copy file to version directory
            version_file_path = os.path.join(version_dir, os.path.basename(file_path))
            shutil.copy2(file_path, version_file_path)
            
            # Copy thumbnail if provided
            thumbnail_path = kwargs.get("thumbnail_path")
            version_thumbnail_path = None
            if thumbnail_path and os.path.exists(thumbnail_path):
                version_thumbnail_path = os.path.join(version_dir, f"thumb_{os.path.basename(thumbnail_path)}")
                shutil.copy2(thumbnail_path, version_thumbnail_path)
            
            # Create version record
            version = ContentVersion(
                content_item_id=content_item_id,
                version_number=next_version,
                file_path=version_file_path,
                thumbnail_path=version_thumbnail_path,
                generation_params=kwargs.get("generation_params"),
                quality_metrics=kwargs.get("quality_metrics"),
                change_description=change_description,
                created_by=user_id
            )
            
            db.add(version)
            db.commit()
            db.refresh(version)
            
            # Update content item to point to latest version
            content_item.file_path = version_file_path
            content_item.thumbnail_path = version_thumbnail_path
            content_item.generation_params = kwargs.get("generation_params", content_item.generation_params)
            content_item.quality_metrics = kwargs.get("quality_metrics", content_item.quality_metrics)
            content_item.updated_at = datetime.utcnow()
            
            db.commit()
            
            return self._version_to_dict(version)
    
    async def get_version_history(self, content_item_id: int, user_id: str) -> List[Dict[str, Any]]:
        """Get version history for a content item."""
        with get_db_session() as db:
            # Get content item and check access
            content_item = db.query(ContentItem).filter(
                ContentItem.id == content_item_id
            ).first()
            
            if not content_item:
                return []
            
            if not await self._has_project_access(content_item.project_id, user_id, db):
                return []
            
            # Get all versions
            versions = db.query(ContentVersion).filter(
                ContentVersion.content_item_id == content_item_id
            ).order_by(desc(ContentVersion.version_number)).all()
            
            return [self._version_to_dict(version) for version in versions]
    
    async def get_version(self, content_item_id: int, version_number: int, user_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific version of a content item."""
        with get_db_session() as db:
            # Get content item and check access
            content_item = db.query(ContentItem).filter(
                ContentItem.id == content_item_id
            ).first()
            
            if not content_item:
                return None
            
            if not await self._has_project_access(content_item.project_id, user_id, db):
                return None
            
            # Get specific version
            version = db.query(ContentVersion).filter(
                and_(
                    ContentVersion.content_item_id == content_item_id,
                    ContentVersion.version_number == version_number
                )
            ).first()
            
            if not version:
                return None
            
            return self._version_to_dict(version)
    
    async def revert_to_version(self, content_item_id: int, version_number: int, 
                              user_id: str, revert_description: str = None) -> bool:
        """Revert content item to a specific version."""
        with get_db_session() as db:
            # Get content item and check access
            content_item = db.query(ContentItem).filter(
                ContentItem.id == content_item_id
            ).first()
            
            if not content_item:
                return False
            
            if not await self._has_project_edit_access(content_item.project_id, user_id, db):
                return False
            
            # Get target version
            target_version = db.query(ContentVersion).filter(
                and_(
                    ContentVersion.content_item_id == content_item_id,
                    ContentVersion.version_number == version_number
                )
            ).first()
            
            if not target_version:
                return False
            
            # Create new version from the target version
            change_description = revert_description or f"Reverted to version {version_number}"
            
            new_version_result = await self.create_version(
                content_item_id=content_item_id,
                file_path=target_version.file_path,
                user_id=user_id,
                change_description=change_description,
                thumbnail_path=target_version.thumbnail_path,
                generation_params=target_version.generation_params,
                quality_metrics=target_version.quality_metrics
            )
            
            return new_version_result is not None
    
    async def compare_versions(self, content_item_id: int, version1: int, version2: int,
                             user_id: str) -> Optional[Dict[str, Any]]:
        """Compare two versions of a content item."""
        with get_db_session() as db:
            # Get content item and check access
            content_item = db.query(ContentItem).filter(
                ContentItem.id == content_item_id
            ).first()
            
            if not content_item:
                return None
            
            if not await self._has_project_access(content_item.project_id, user_id, db):
                return None
            
            # Get both versions
            v1 = db.query(ContentVersion).filter(
                and_(
                    ContentVersion.content_item_id == content_item_id,
                    ContentVersion.version_number == version1
                )
            ).first()
            
            v2 = db.query(ContentVersion).filter(
                and_(
                    ContentVersion.content_item_id == content_item_id,
                    ContentVersion.version_number == version2
                )
            ).first()
            
            if not v1 or not v2:
                return None
            
            # Compare versions
            comparison = {
                "version1": self._version_to_dict(v1),
                "version2": self._version_to_dict(v2),
                "differences": self._calculate_differences(v1, v2),
                "quality_comparison": self._compare_quality_metrics(v1, v2)
            }
            
            return comparison
    
    async def get_version_statistics(self, content_item_id: int, user_id: str) -> Optional[Dict[str, Any]]:
        """Get statistics for version history."""
        with get_db_session() as db:
            # Get content item and check access
            content_item = db.query(ContentItem).filter(
                ContentItem.id == content_item_id
            ).first()
            
            if not content_item:
                return None
            
            if not await self._has_project_access(content_item.project_id, user_id, db):
                return None
            
            # Get version statistics
            total_versions = db.query(func.count(ContentVersion.id)).filter(
                ContentVersion.content_item_id == content_item_id
            ).scalar()
            
            # Get contributors
            contributors = db.query(
                ContentVersion.created_by,
                func.count(ContentVersion.id).label('version_count')
            ).filter(
                ContentVersion.content_item_id == content_item_id
            ).group_by(ContentVersion.created_by).all()
            
            # Get creation timeline
            first_version = db.query(ContentVersion).filter(
                ContentVersion.content_item_id == content_item_id
            ).order_by(ContentVersion.version_number).first()
            
            latest_version = db.query(ContentVersion).filter(
                ContentVersion.content_item_id == content_item_id
            ).order_by(desc(ContentVersion.version_number)).first()
            
            # Calculate average time between versions
            versions = db.query(ContentVersion).filter(
                ContentVersion.content_item_id == content_item_id
            ).order_by(ContentVersion.version_number).all()
            
            time_deltas = []
            for i in range(1, len(versions)):
                delta = versions[i].created_at - versions[i-1].created_at
                time_deltas.append(delta.total_seconds())
            
            avg_time_between_versions = sum(time_deltas) / len(time_deltas) if time_deltas else 0
            
            return {
                "total_versions": total_versions or 0,
                "contributors": [
                    {"user_id": user_id, "version_count": count}
                    for user_id, count in contributors
                ],
                "first_created": first_version.created_at if first_version else None,
                "last_updated": latest_version.created_at if latest_version else None,
                "avg_time_between_versions_seconds": avg_time_between_versions,
                "current_version": latest_version.version_number if latest_version else 0
            }
    
    async def delete_version(self, content_item_id: int, version_number: int, user_id: str) -> bool:
        """Delete a specific version (if not the only version)."""
        with get_db_session() as db:
            # Get content item and check access
            content_item = db.query(ContentItem).filter(
                ContentItem.id == content_item_id
            ).first()
            
            if not content_item:
                return False
            
            if not await self._has_project_edit_access(content_item.project_id, user_id, db):
                return False
            
            # Check if this is the only version
            version_count = db.query(func.count(ContentVersion.id)).filter(
                ContentVersion.content_item_id == content_item_id
            ).scalar()
            
            if version_count <= 1:
                return False  # Cannot delete the only version
            
            # Get version to delete
            version = db.query(ContentVersion).filter(
                and_(
                    ContentVersion.content_item_id == content_item_id,
                    ContentVersion.version_number == version_number
                )
            ).first()
            
            if not version:
                return False
            
            # Delete version files
            try:
                if os.path.exists(version.file_path):
                    os.remove(version.file_path)
                if version.thumbnail_path and os.path.exists(version.thumbnail_path):
                    os.remove(version.thumbnail_path)
                
                # Remove version directory if empty
                version_dir = os.path.dirname(version.file_path)
                if os.path.exists(version_dir) and not os.listdir(version_dir):
                    os.rmdir(version_dir)
            except OSError:
                pass  # File deletion failed, but continue with database cleanup
            
            # Delete version record
            db.delete(version)
            db.commit()
            
            # If this was the current version, update content item to latest version
            if content_item.file_path == version.file_path:
                latest_version = db.query(ContentVersion).filter(
                    ContentVersion.content_item_id == content_item_id
                ).order_by(desc(ContentVersion.version_number)).first()
                
                if latest_version:
                    content_item.file_path = latest_version.file_path
                    content_item.thumbnail_path = latest_version.thumbnail_path
                    content_item.generation_params = latest_version.generation_params
                    content_item.quality_metrics = latest_version.quality_metrics
                    content_item.updated_at = datetime.utcnow()
                    db.commit()
            
            return True
    
    async def create_branch(self, content_item_id: int, branch_name: str, 
                          from_version: int, user_id: str) -> Optional[Dict[str, Any]]:
        """Create a branch from a specific version (advanced feature)."""
        with get_db_session() as db:
            # Get content item and check access
            content_item = db.query(ContentItem).filter(
                ContentItem.id == content_item_id
            ).first()
            
            if not content_item:
                return None
            
            if not await self._has_project_edit_access(content_item.project_id, user_id, db):
                return None
            
            # Get source version
            source_version = db.query(ContentVersion).filter(
                and_(
                    ContentVersion.content_item_id == content_item_id,
                    ContentVersion.version_number == from_version
                )
            ).first()
            
            if not source_version:
                return None
            
            # Store branch information in content item metadata
            if not content_item.content_metadata:
                content_item.content_metadata = {}
            
            if "branches" not in content_item.content_metadata:
                content_item.content_metadata["branches"] = {}
            
            content_item.content_metadata["branches"][branch_name] = {
                "created_by": user_id,
                "created_at": datetime.utcnow().isoformat(),
                "source_version": from_version,
                "current_version": from_version
            }
            
            db.commit()
            
            return {
                "branch_name": branch_name,
                "created_by": user_id,
                "source_version": from_version,
                "created_at": datetime.utcnow()
            }
    
    def _version_to_dict(self, version: ContentVersion) -> Dict[str, Any]:
        """Convert ContentVersion to dictionary."""
        return {
            "id": version.id,
            "content_item_id": version.content_item_id,
            "version_number": version.version_number,
            "file_path": version.file_path,
            "thumbnail_path": version.thumbnail_path,
            "generation_params": version.generation_params,
            "quality_metrics": version.quality_metrics,
            "change_description": version.change_description,
            "created_by": version.created_by,
            "created_at": version.created_at
        }
    
    def _calculate_differences(self, v1: ContentVersion, v2: ContentVersion) -> Dict[str, Any]:
        """Calculate differences between two versions."""
        differences = {
            "file_size_change": 0,
            "parameter_changes": [],
            "quality_changes": []
        }
        
        # File size comparison
        try:
            if os.path.exists(v1.file_path) and os.path.exists(v2.file_path):
                size1 = os.path.getsize(v1.file_path)
                size2 = os.path.getsize(v2.file_path)
                differences["file_size_change"] = size2 - size1
        except OSError:
            pass
        
        # Parameter changes
        params1 = v1.generation_params or {}
        params2 = v2.generation_params or {}
        
        all_keys = set(params1.keys()) | set(params2.keys())
        for key in all_keys:
            val1 = params1.get(key)
            val2 = params2.get(key)
            
            if val1 != val2:
                differences["parameter_changes"].append({
                    "parameter": key,
                    "old_value": val1,
                    "new_value": val2
                })
        
        # Quality changes
        quality1 = v1.quality_metrics or {}
        quality2 = v2.quality_metrics or {}
        
        all_quality_keys = set(quality1.keys()) | set(quality2.keys())
        for key in all_quality_keys:
            val1 = quality1.get(key)
            val2 = quality2.get(key)
            
            if val1 != val2 and isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                differences["quality_changes"].append({
                    "metric": key,
                    "old_value": val1,
                    "new_value": val2,
                    "change": val2 - val1
                })
        
        return differences
    
    def _compare_quality_metrics(self, v1: ContentVersion, v2: ContentVersion) -> Dict[str, Any]:
        """Compare quality metrics between versions."""
        quality1 = v1.quality_metrics or {}
        quality2 = v2.quality_metrics or {}
        
        comparison = {
            "overall_improvement": 0,
            "metric_comparisons": []
        }
        
        # Compare overall scores
        overall1 = quality1.get("overall_score", 0)
        overall2 = quality2.get("overall_score", 0)
        comparison["overall_improvement"] = overall2 - overall1
        
        # Compare individual metrics
        all_metrics = set(quality1.keys()) | set(quality2.keys())
        for metric in all_metrics:
            val1 = quality1.get(metric, 0)
            val2 = quality2.get(metric, 0)
            
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                comparison["metric_comparisons"].append({
                    "metric": metric,
                    "v1_score": val1,
                    "v2_score": val2,
                    "improvement": val2 - val1,
                    "improvement_percentage": ((val2 - val1) / val1 * 100) if val1 > 0 else 0
                })
        
        return comparison
    
    async def _has_project_access(self, project_id: int, user_id: str, db: Session) -> bool:
        """Check if user has access to project."""
        from scrollintel.engines.visual_generation.collaboration.sharing_manager import SharingManager
        sharing_manager = SharingManager()
        return await sharing_manager._has_project_access(project_id, user_id, db)
    
    async def _has_project_edit_access(self, project_id: int, user_id: str, db: Session) -> bool:
        """Check if user has edit access to project."""
        from scrollintel.engines.visual_generation.collaboration.sharing_manager import SharingManager
        sharing_manager = SharingManager()
        return await sharing_manager._has_project_edit_access(project_id, user_id, db)