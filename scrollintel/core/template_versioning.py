"""
Template Versioning System for Dashboard Templates

Provides version control and rollback capabilities for dashboard templates.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import json
import uuid
from enum import Enum

from .template_engine import DashboardTemplate

class VersionAction(Enum):
    CREATE = "create"
    UPDATE = "update"
    ROLLBACK = "rollback"
    BRANCH = "branch"
    MERGE = "merge"

@dataclass
class TemplateVersion:
    """Version information for a template"""
    version_id: str
    template_id: str
    version_number: str
    action: VersionAction
    changes: Dict[str, Any]
    author: str
    commit_message: str
    created_at: datetime
    parent_version: Optional[str] = None
    is_active: bool = False
    template_snapshot: Optional[Dict[str, Any]] = None

@dataclass
class VersionBranch:
    """Branch information for template versions"""
    branch_id: str
    template_id: str
    branch_name: str
    base_version: str
    created_by: str
    created_at: datetime
    is_merged: bool = False
    merged_at: Optional[datetime] = None

class TemplateVersioning:
    """Version control system for dashboard templates"""
    
    def __init__(self):
        self.versions: Dict[str, List[TemplateVersion]] = {}  # template_id -> versions
        self.branches: Dict[str, List[VersionBranch]] = {}   # template_id -> branches
        self.active_versions: Dict[str, str] = {}            # template_id -> active_version_id
    
    def create_initial_version(self, template: DashboardTemplate, author: str) -> str:
        """Create initial version for a new template"""
        version_id = str(uuid.uuid4())
        
        version = TemplateVersion(
            version_id=version_id,
            template_id=template.id,
            version_number="1.0.0",
            action=VersionAction.CREATE,
            changes={"action": "initial_creation"},
            author=author,
            commit_message="Initial template creation",
            created_at=datetime.now(),
            is_active=True,
            template_snapshot=self._create_template_snapshot(template)
        )
        
        if template.id not in self.versions:
            self.versions[template.id] = []
        
        self.versions[template.id].append(version)
        self.active_versions[template.id] = version_id
        
        return version_id
    
    def create_version(self, 
                      template: DashboardTemplate, 
                      author: str, 
                      commit_message: str,
                      changes: Dict[str, Any]) -> str:
        """Create new version of existing template"""
        
        if template.id not in self.versions:
            return self.create_initial_version(template, author)
        
        # Get current active version
        current_versions = self.versions[template.id]
        current_active = next((v for v in current_versions if v.is_active), None)
        
        # Generate new version number
        new_version_number = self._generate_version_number(template.id, changes)
        
        version_id = str(uuid.uuid4())
        
        version = TemplateVersion(
            version_id=version_id,
            template_id=template.id,
            version_number=new_version_number,
            action=VersionAction.UPDATE,
            changes=changes,
            author=author,
            commit_message=commit_message,
            created_at=datetime.now(),
            parent_version=current_active.version_id if current_active else None,
            is_active=True,
            template_snapshot=self._create_template_snapshot(template)
        )
        
        # Deactivate current version
        if current_active:
            current_active.is_active = False
        
        self.versions[template.id].append(version)
        self.active_versions[template.id] = version_id
        
        return version_id
    
    def rollback_to_version(self, template_id: str, version_id: str, author: str) -> Optional[DashboardTemplate]:
        """Rollback template to specific version"""
        
        if template_id not in self.versions:
            return None
        
        target_version = None
        for version in self.versions[template_id]:
            if version.version_id == version_id:
                target_version = version
                break
        
        if not target_version or not target_version.template_snapshot:
            return None
        
        # Create rollback version
        rollback_version_id = str(uuid.uuid4())
        current_active = self.active_versions.get(template_id)
        
        rollback_version = TemplateVersion(
            version_id=rollback_version_id,
            template_id=template_id,
            version_number=self._generate_rollback_version_number(template_id),
            action=VersionAction.ROLLBACK,
            changes={"rollback_to": version_id, "rollback_from": current_active},
            author=author,
            commit_message=f"Rollback to version {target_version.version_number}",
            created_at=datetime.now(),
            parent_version=current_active,
            is_active=True,
            template_snapshot=target_version.template_snapshot.copy()
        )
        
        # Deactivate current version
        for version in self.versions[template_id]:
            if version.is_active:
                version.is_active = False
        
        self.versions[template_id].append(rollback_version)
        self.active_versions[template_id] = rollback_version_id
        
        # Reconstruct template from snapshot
        return self._reconstruct_template_from_snapshot(target_version.template_snapshot)
    
    def create_branch(self, template_id: str, branch_name: str, base_version: str, author: str) -> str:
        """Create new branch from specific version"""
        
        branch_id = str(uuid.uuid4())
        
        branch = VersionBranch(
            branch_id=branch_id,
            template_id=template_id,
            branch_name=branch_name,
            base_version=base_version,
            created_by=author,
            created_at=datetime.now()
        )
        
        if template_id not in self.branches:
            self.branches[template_id] = []
        
        self.branches[template_id].append(branch)
        return branch_id
    
    def merge_branch(self, template_id: str, branch_id: str, author: str) -> bool:
        """Merge branch back to main template"""
        
        if template_id not in self.branches:
            return False
        
        branch = None
        for b in self.branches[template_id]:
            if b.branch_id == branch_id:
                branch = b
                break
        
        if not branch or branch.is_merged:
            return False
        
        # Mark branch as merged
        branch.is_merged = True
        branch.merged_at = datetime.now()
        
        # Create merge version
        merge_version_id = str(uuid.uuid4())
        current_active = self.active_versions.get(template_id)
        
        merge_version = TemplateVersion(
            version_id=merge_version_id,
            template_id=template_id,
            version_number=self._generate_merge_version_number(template_id),
            action=VersionAction.MERGE,
            changes={"merged_branch": branch_id, "branch_name": branch.branch_name},
            author=author,
            commit_message=f"Merge branch '{branch.branch_name}'",
            created_at=datetime.now(),
            parent_version=current_active,
            is_active=True
        )
        
        # Deactivate current version
        for version in self.versions[template_id]:
            if version.is_active:
                version.is_active = False
        
        self.versions[template_id].append(merge_version)
        self.active_versions[template_id] = merge_version_id
        
        return True
    
    def get_version_history(self, template_id: str) -> List[TemplateVersion]:
        """Get version history for template"""
        return self.versions.get(template_id, [])
    
    def get_active_version(self, template_id: str) -> Optional[TemplateVersion]:
        """Get currently active version"""
        if template_id not in self.versions:
            return None
        
        active_version_id = self.active_versions.get(template_id)
        if not active_version_id:
            return None
        
        for version in self.versions[template_id]:
            if version.version_id == active_version_id:
                return version
        
        return None
    
    def compare_versions(self, template_id: str, version1_id: str, version2_id: str) -> Dict[str, Any]:
        """Compare two versions and return differences"""
        
        version1 = None
        version2 = None
        
        for version in self.versions.get(template_id, []):
            if version.version_id == version1_id:
                version1 = version
            elif version.version_id == version2_id:
                version2 = version
        
        if not version1 or not version2:
            return {"error": "One or both versions not found"}
        
        # Compare snapshots
        snapshot1 = version1.template_snapshot or {}
        snapshot2 = version2.template_snapshot or {}
        
        differences = self._compare_snapshots(snapshot1, snapshot2)
        
        return {
            "version1": {
                "id": version1.version_id,
                "number": version1.version_number,
                "created_at": version1.created_at.isoformat()
            },
            "version2": {
                "id": version2.version_id,
                "number": version2.version_number,
                "created_at": version2.created_at.isoformat()
            },
            "differences": differences
        }
    
    def get_branches(self, template_id: str) -> List[VersionBranch]:
        """Get all branches for template"""
        return self.branches.get(template_id, [])
    
    def _create_template_snapshot(self, template: DashboardTemplate) -> Dict[str, Any]:
        """Create snapshot of template state"""
        return {
            **asdict(template),
            'created_at': template.created_at.isoformat(),
            'updated_at': template.updated_at.isoformat(),
            'industry': template.industry.value,
            'category': template.category.value
        }
    
    def _reconstruct_template_from_snapshot(self, snapshot: Dict[str, Any]) -> DashboardTemplate:
        """Reconstruct template from snapshot"""
        from .template_engine import IndustryType, TemplateCategory, WidgetConfig
        
        # Parse dates
        snapshot['created_at'] = datetime.fromisoformat(snapshot['created_at'])
        snapshot['updated_at'] = datetime.fromisoformat(snapshot['updated_at'])
        
        # Convert enums
        snapshot['industry'] = IndustryType(snapshot['industry'])
        snapshot['category'] = TemplateCategory(snapshot['category'])
        
        # Convert widgets
        widgets = []
        for widget_data in snapshot['widgets']:
            widget = WidgetConfig(**widget_data)
            widgets.append(widget)
        snapshot['widgets'] = widgets
        
        return DashboardTemplate(**snapshot)
    
    def _generate_version_number(self, template_id: str, changes: Dict[str, Any]) -> str:
        """Generate next version number based on changes"""
        versions = self.versions.get(template_id, [])
        if not versions:
            return "1.0.0"
        
        # Get latest version number
        latest_version = max(versions, key=lambda v: v.created_at)
        parts = latest_version.version_number.split('.')
        major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])
        
        # Determine version increment based on changes
        if self._is_breaking_change(changes):
            major += 1
            minor = 0
            patch = 0
        elif self._is_feature_change(changes):
            minor += 1
            patch = 0
        else:
            patch += 1
        
        return f"{major}.{minor}.{patch}"
    
    def _generate_rollback_version_number(self, template_id: str) -> str:
        """Generate version number for rollback"""
        versions = self.versions.get(template_id, [])
        if not versions:
            return "1.0.0"
        
        latest_version = max(versions, key=lambda v: v.created_at)
        parts = latest_version.version_number.split('.')
        major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])
        
        return f"{major}.{minor}.{patch + 1}"
    
    def _generate_merge_version_number(self, template_id: str) -> str:
        """Generate version number for merge"""
        return self._generate_rollback_version_number(template_id)
    
    def _is_breaking_change(self, changes: Dict[str, Any]) -> bool:
        """Determine if changes are breaking"""
        breaking_changes = [
            'layout_config_major_change',
            'widget_removal',
            'data_source_change'
        ]
        return any(change in changes for change in breaking_changes)
    
    def _is_feature_change(self, changes: Dict[str, Any]) -> bool:
        """Determine if changes add new features"""
        feature_changes = [
            'widget_addition',
            'new_visualization',
            'filter_addition'
        ]
        return any(change in changes for change in feature_changes)
    
    def _compare_snapshots(self, snapshot1: Dict[str, Any], snapshot2: Dict[str, Any]) -> Dict[str, Any]:
        """Compare two template snapshots"""
        differences = {}
        
        # Compare basic fields
        for key in ['name', 'description', 'version']:
            if snapshot1.get(key) != snapshot2.get(key):
                differences[key] = {
                    'old': snapshot1.get(key),
                    'new': snapshot2.get(key)
                }
        
        # Compare widgets
        widgets1 = {w['id']: w for w in snapshot1.get('widgets', [])}
        widgets2 = {w['id']: w for w in snapshot2.get('widgets', [])}
        
        widget_changes = {}
        
        # Added widgets
        added_widgets = set(widgets2.keys()) - set(widgets1.keys())
        if added_widgets:
            widget_changes['added'] = [widgets2[wid] for wid in added_widgets]
        
        # Removed widgets
        removed_widgets = set(widgets1.keys()) - set(widgets2.keys())
        if removed_widgets:
            widget_changes['removed'] = [widgets1[wid] for wid in removed_widgets]
        
        # Modified widgets
        modified_widgets = []
        for wid in set(widgets1.keys()) & set(widgets2.keys()):
            if widgets1[wid] != widgets2[wid]:
                modified_widgets.append({
                    'id': wid,
                    'old': widgets1[wid],
                    'new': widgets2[wid]
                })
        
        if modified_widgets:
            widget_changes['modified'] = modified_widgets
        
        if widget_changes:
            differences['widgets'] = widget_changes
        
        return differences