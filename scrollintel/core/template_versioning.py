"""
Template versioning system for dashboard templates with rollback capabilities.
"""
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
import json
import uuid
from sqlalchemy import Column, String, DateTime, Text, Boolean, Integer, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, Session
from scrollintel.models.database import Base, get_db


class VersionAction(Enum):
    CREATE = "create"
    UPDATE = "update"
    CLONE = "clone"
    IMPORT = "import"
    ROLLBACK = "rollback"


@dataclass
class TemplateChange:
    field: str
    old_value: Any
    new_value: Any
    change_type: str  # "added", "modified", "removed"


@dataclass
class VersionMetadata:
    version: str
    action: VersionAction
    changes: List[TemplateChange]
    created_by: str
    created_at: datetime
    description: str
    is_major: bool = False
    parent_version: Optional[str] = None


class TemplateVersion(Base):
    """Database model for template versions."""
    __tablename__ = "template_versions"
    
    id = Column(String, primary_key=True)
    template_id = Column(String, nullable=False, index=True)
    version = Column(String, nullable=False)
    template_data = Column(Text, nullable=False)  # JSON serialized template
    metadata = Column(Text, nullable=False)  # JSON serialized metadata
    created_by = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    is_major = Column(Boolean, default=False)
    parent_version = Column(String, nullable=True)
    
    def __repr__(self):
        return f"<TemplateVersion(id={self.id}, template_id={self.template_id}, version={self.version})>"


class TemplateVersionManager:
    """Manager for template versioning and rollback operations."""
    
    def __init__(self):
        self.db_session = None
    
    def _get_db_session(self) -> Session:
        """Get database session."""
        if not self.db_session:
            self.db_session = next(get_db())
        return self.db_session
    
    def _generate_version_number(self, template_id: str, is_major: bool = False) -> str:
        """Generate next version number for template."""
        db = self._get_db_session()
        
        # Get latest version
        latest_version = db.query(TemplateVersion)\
            .filter(TemplateVersion.template_id == template_id)\
            .order_by(TemplateVersion.created_at.desc())\
            .first()
        
        if not latest_version:
            return "1.0.0"
        
        # Parse current version
        try:
            major, minor, patch = map(int, latest_version.version.split('.'))
        except ValueError:
            return "1.0.0"
        
        # Increment version
        if is_major:
            return f"{major + 1}.0.0"
        else:
            return f"{major}.{minor}.{patch + 1}"
    
    def _calculate_changes(self, old_template: Dict[str, Any], new_template: Dict[str, Any]) -> List[TemplateChange]:
        """Calculate changes between template versions."""
        changes = []
        
        # Compare basic fields
        basic_fields = ['name', 'description', 'industry', 'category', 'permissions', 'tags']
        for field in basic_fields:
            old_val = old_template.get(field)
            new_val = new_template.get(field)
            
            if old_val != new_val:
                if old_val is None:
                    change_type = "added"
                elif new_val is None:
                    change_type = "removed"
                else:
                    change_type = "modified"
                
                changes.append(TemplateChange(
                    field=field,
                    old_value=old_val,
                    new_value=new_val,
                    change_type=change_type
                ))
        
        # Compare widgets
        old_widgets = {w['id']: w for w in old_template.get('widgets', [])}
        new_widgets = {w['id']: w for w in new_template.get('widgets', [])}
        
        # Added widgets
        for widget_id, widget in new_widgets.items():
            if widget_id not in old_widgets:
                changes.append(TemplateChange(
                    field=f"widgets.{widget_id}",
                    old_value=None,
                    new_value=widget,
                    change_type="added"
                ))
        
        # Removed widgets
        for widget_id, widget in old_widgets.items():
            if widget_id not in new_widgets:
                changes.append(TemplateChange(
                    field=f"widgets.{widget_id}",
                    old_value=widget,
                    new_value=None,
                    change_type="removed"
                ))
        
        # Modified widgets
        for widget_id in set(old_widgets.keys()) & set(new_widgets.keys()):
            old_widget = old_widgets[widget_id]
            new_widget = new_widgets[widget_id]
            
            if old_widget != new_widget:
                changes.append(TemplateChange(
                    field=f"widgets.{widget_id}",
                    old_value=old_widget,
                    new_value=new_widget,
                    change_type="modified"
                ))
        
        # Compare layout config
        old_layout = old_template.get('layout_config', {})
        new_layout = new_template.get('layout_config', {})
        
        if old_layout != new_layout:
            changes.append(TemplateChange(
                field="layout_config",
                old_value=old_layout,
                new_value=new_layout,
                change_type="modified"
            ))
        
        return changes
    
    def create_version(
        self,
        template_id: str,
        template_data: Dict[str, Any],
        action: VersionAction,
        created_by: str,
        description: str = "",
        is_major: bool = False,
        parent_version: Optional[str] = None
    ) -> TemplateVersion:
        """Create a new version of a template."""
        db = self._get_db_session()
        
        # Generate version number
        version_number = self._generate_version_number(template_id, is_major)
        
        # Calculate changes if updating
        changes = []
        if action == VersionAction.UPDATE and parent_version:
            parent = self.get_version(template_id, parent_version)
            if parent:
                old_data = json.loads(parent.template_data)
                changes = self._calculate_changes(old_data, template_data)
        
        # Create metadata
        metadata = VersionMetadata(
            version=version_number,
            action=action,
            changes=changes,
            created_by=created_by,
            created_at=datetime.utcnow(),
            description=description,
            is_major=is_major,
            parent_version=parent_version
        )
        
        # Create version record
        version = TemplateVersion(
            id=f"version_{uuid.uuid4().hex[:12]}",
            template_id=template_id,
            version=version_number,
            template_data=json.dumps(template_data),
            metadata=json.dumps(asdict(metadata)),
            created_by=created_by,
            is_major=is_major,
            parent_version=parent_version
        )
        
        db.add(version)
        db.commit()
        
        return version
    
    def get_version(self, template_id: str, version: str) -> Optional[TemplateVersion]:
        """Get a specific version of a template."""
        db = self._get_db_session()
        
        return db.query(TemplateVersion)\
            .filter(
                TemplateVersion.template_id == template_id,
                TemplateVersion.version == version
            )\
            .first()
    
    def get_latest_version(self, template_id: str) -> Optional[TemplateVersion]:
        """Get the latest version of a template."""
        db = self._get_db_session()
        
        return db.query(TemplateVersion)\
            .filter(TemplateVersion.template_id == template_id)\
            .order_by(TemplateVersion.created_at.desc())\
            .first()
    
    def get_version_history(self, template_id: str, limit: int = 50) -> List[TemplateVersion]:
        """Get version history for a template."""
        db = self._get_db_session()
        
        return db.query(TemplateVersion)\
            .filter(TemplateVersion.template_id == template_id)\
            .order_by(TemplateVersion.created_at.desc())\
            .limit(limit)\
            .all()
    
    def get_major_versions(self, template_id: str) -> List[TemplateVersion]:
        """Get only major versions of a template."""
        db = self._get_db_session()
        
        return db.query(TemplateVersion)\
            .filter(
                TemplateVersion.template_id == template_id,
                TemplateVersion.is_major == True
            )\
            .order_by(TemplateVersion.created_at.desc())\
            .all()
    
    def rollback_to_version(
        self,
        template_id: str,
        target_version: str,
        rolled_back_by: str,
        description: str = ""
    ) -> Optional[TemplateVersion]:
        """Rollback template to a specific version."""
        db = self._get_db_session()
        
        # Get target version
        target = self.get_version(template_id, target_version)
        if not target:
            return None
        
        # Get current version for comparison
        current = self.get_latest_version(template_id)
        current_version = current.version if current else "0.0.0"
        
        # Create rollback version
        template_data = json.loads(target.template_data)
        rollback_description = description or f"Rollback to version {target_version}"
        
        rollback_version = self.create_version(
            template_id=template_id,
            template_data=template_data,
            action=VersionAction.ROLLBACK,
            created_by=rolled_back_by,
            description=rollback_description,
            is_major=False,
            parent_version=current_version
        )
        
        return rollback_version
    
    def compare_versions(
        self,
        template_id: str,
        version1: str,
        version2: str
    ) -> Dict[str, Any]:
        """Compare two versions of a template."""
        v1 = self.get_version(template_id, version1)
        v2 = self.get_version(template_id, version2)
        
        if not v1 or not v2:
            return {"error": "One or both versions not found"}
        
        v1_data = json.loads(v1.template_data)
        v2_data = json.loads(v2.template_data)
        
        changes = self._calculate_changes(v1_data, v2_data)
        
        return {
            "version1": {
                "version": v1.version,
                "created_at": v1.created_at.isoformat(),
                "created_by": v1.created_by
            },
            "version2": {
                "version": v2.version,
                "created_at": v2.created_at.isoformat(),
                "created_by": v2.created_by
            },
            "changes": [asdict(change) for change in changes],
            "change_count": len(changes)
        }
    
    def get_version_metadata(self, template_id: str, version: str) -> Optional[VersionMetadata]:
        """Get metadata for a specific version."""
        version_record = self.get_version(template_id, version)
        if not version_record:
            return None
        
        metadata_dict = json.loads(version_record.metadata)
        
        # Convert changes back to TemplateChange objects
        changes = []
        for change_dict in metadata_dict.get('changes', []):
            changes.append(TemplateChange(**change_dict))
        
        metadata_dict['changes'] = changes
        metadata_dict['action'] = VersionAction(metadata_dict['action'])
        metadata_dict['created_at'] = datetime.fromisoformat(metadata_dict['created_at'])
        
        return VersionMetadata(**metadata_dict)
    
    def delete_version(self, template_id: str, version: str) -> bool:
        """Delete a specific version (soft delete)."""
        db = self._get_db_session()
        
        version_record = self.get_version(template_id, version)
        if not version_record:
            return False
        
        # Don't allow deletion of the only version
        version_count = db.query(TemplateVersion)\
            .filter(
                TemplateVersion.template_id == template_id,
                TemplateVersion.is_active == True
            )\
            .count()
        
        if version_count <= 1:
            return False
        
        version_record.is_active = False
        db.commit()
        
        return True
    
    def cleanup_old_versions(self, template_id: str, keep_count: int = 10) -> int:
        """Clean up old versions, keeping only the most recent ones."""
        db = self._get_db_session()
        
        # Get all versions ordered by creation date
        versions = db.query(TemplateVersion)\
            .filter(
                TemplateVersion.template_id == template_id,
                TemplateVersion.is_active == True
            )\
            .order_by(TemplateVersion.created_at.desc())\
            .all()
        
        if len(versions) <= keep_count:
            return 0
        
        # Keep major versions and recent versions
        to_keep = set()
        
        # Always keep major versions
        for version in versions:
            if version.is_major:
                to_keep.add(version.id)
        
        # Keep most recent versions
        for version in versions[:keep_count]:
            to_keep.add(version.id)
        
        # Mark others for deletion
        deleted_count = 0
        for version in versions:
            if version.id not in to_keep:
                version.is_active = False
                deleted_count += 1
        
        db.commit()
        return deleted_count


# Global version manager instance
version_manager = TemplateVersionManager()