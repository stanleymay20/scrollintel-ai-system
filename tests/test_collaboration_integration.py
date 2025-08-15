"""
Integration tests for collaboration features.
"""

import pytest
import tempfile
import os
from datetime import datetime

from scrollintel.engines.visual_generation.collaboration.project_manager import ProjectManager
from scrollintel.engines.visual_generation.collaboration.sharing_manager import SharingManager
from scrollintel.engines.visual_generation.collaboration.comment_system import CommentSystem
from scrollintel.engines.visual_generation.collaboration.version_control import VersionControl
from scrollintel.models.collaboration_models import (
    ProjectCreateRequest, ShareRequest, CommentRequest, SharePermission
)


class TestCollaborationIntegration:
    """Integration tests for collaboration features."""
    
    def test_project_manager_initialization(self):
        """Test that ProjectManager can be initialized."""
        pm = ProjectManager()
        assert pm is not None
        assert hasattr(pm, 'storage_path')
        assert os.path.exists(pm.storage_path)
    
    def test_sharing_manager_initialization(self):
        """Test that SharingManager can be initialized."""
        sm = SharingManager()
        assert sm is not None
        assert hasattr(sm, 'default_share_expiry_days')
        assert sm.default_share_expiry_days == 30
    
    def test_comment_system_initialization(self):
        """Test that CommentSystem can be initialized."""
        cs = CommentSystem()
        assert cs is not None
    
    def test_version_control_initialization(self):
        """Test that VersionControl can be initialized."""
        vc = VersionControl()
        assert vc is not None
        assert hasattr(vc, 'storage_path')
        assert os.path.exists(vc.storage_path)
    
    def test_project_create_request_model(self):
        """Test ProjectCreateRequest model."""
        request = ProjectCreateRequest(
            name="Test Project",
            description="A test project",
            tags=["test", "demo"],
            project_metadata={"key": "value"}
        )
        
        assert request.name == "Test Project"
        assert request.description == "A test project"
        assert request.tags == ["test", "demo"]
        assert request.project_metadata == {"key": "value"}
    
    def test_share_request_model(self):
        """Test ShareRequest model."""
        request = ShareRequest(
            user_id="user123",
            permission_level=SharePermission.EDIT
        )
        
        assert request.user_id == "user123"
        assert request.permission_level == SharePermission.EDIT
    
    def test_comment_request_model(self):
        """Test CommentRequest model."""
        request = CommentRequest(
            content="This is a test comment",
            position_x=100,
            position_y=200
        )
        
        assert request.content == "This is a test comment"
        assert request.position_x == 100
        assert request.position_y == 200
    
    def test_storage_directories_created(self):
        """Test that storage directories are created properly."""
        pm = ProjectManager()
        vc = VersionControl()
        
        # Check that directories exist
        assert os.path.exists(pm.storage_path)
        assert os.path.exists(vc.storage_path)
        
        # Check that they are different directories
        assert pm.storage_path != vc.storage_path
    
    def test_enum_values(self):
        """Test that enum values are correct."""
        # Test SharePermission enum
        assert SharePermission.VIEW.value == "view"
        assert SharePermission.COMMENT.value == "comment"
        assert SharePermission.EDIT.value == "edit"
        assert SharePermission.ADMIN.value == "admin"
    
    def test_data_model_imports(self):
        """Test that all data models can be imported."""
        from scrollintel.models.collaboration_models import (
            Project, ContentItem, ContentVersion, ProjectShare,
            ProjectComment, ContentComment, ApprovalWorkflow,
            ProjectStatus, SharePermission, ApprovalStatus
        )
        
        # Test that classes exist
        assert Project is not None
        assert ContentItem is not None
        assert ContentVersion is not None
        assert ProjectShare is not None
        assert ProjectComment is not None
        assert ContentComment is not None
        assert ApprovalWorkflow is not None
        
        # Test that enums exist
        assert ProjectStatus is not None
        assert SharePermission is not None
        assert ApprovalStatus is not None


if __name__ == "__main__":
    pytest.main([__file__])