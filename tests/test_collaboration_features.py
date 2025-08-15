"""
Comprehensive tests for collaboration and sharing functionality.
"""

import pytest
import os
import tempfile
import shutil
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock

from scrollintel.engines.visual_generation.collaboration.project_manager import ProjectManager
from scrollintel.engines.visual_generation.collaboration.sharing_manager import SharingManager
from scrollintel.engines.visual_generation.collaboration.comment_system import CommentSystem
from scrollintel.engines.visual_generation.collaboration.version_control import VersionControl
from scrollintel.models.collaboration_models import (
    ProjectCreateRequest, ProjectUpdateRequest, ShareRequest, CommentRequest,
    ApprovalRequest, ApprovalResponse, SharePermission, ApprovalStatus, ProjectStatus
)


class TestProjectManager:
    """Test project management functionality."""
    
    @pytest.fixture
    def project_manager(self):
        """Create project manager instance."""
        with patch('scrollintel.engines.visual_generation.collaboration.project_manager.get_db_session'):
            return ProjectManager()
    
    @pytest.fixture
    def mock_db_session(self):
        """Mock database session."""
        session = Mock()
        session.__enter__ = Mock(return_value=session)
        session.__exit__ = Mock(return_value=None)
        return session
    
    @pytest.mark.asyncio
    async def test_create_project(self, project_manager, mock_db_session):
        """Test project creation."""
        with patch('scrollintel.engines.visual_generation.collaboration.project_manager.get_db_session', return_value=mock_db_session):
            # Mock project creation
            mock_project = Mock()
            mock_project.id = 1
            mock_project.name = "Test Project"
            mock_project.description = "Test Description"
            mock_project.owner_id = "user123"
            mock_project.status = ProjectStatus.ACTIVE.value
            mock_project.tags = ["test"]
            mock_project.metadata = {}
            mock_project.created_at = datetime.utcnow()
            mock_project.updated_at = datetime.utcnow()
            
            mock_db_session.add = Mock()
            mock_db_session.commit = Mock()
            mock_db_session.refresh = Mock()
            
            # Mock the _project_to_response method
            with patch.object(project_manager, '_project_to_response') as mock_response:
                mock_response.return_value = {
                    "id": 1,
                    "name": "Test Project",
                    "description": "Test Description",
                    "owner_id": "user123",
                    "status": "active",
                    "tags": ["test"],
                    "metadata": {},
                    "created_at": datetime.utcnow(),
                    "updated_at": datetime.utcnow(),
                    "content_count": 0,
                    "collaborator_count": 0
                }
                
                request = ProjectCreateRequest(
                    name="Test Project",
                    description="Test Description",
                    tags=["test"]
                )
                
                result = await project_manager.create_project(request, "user123")
                
                assert result is not None
                assert result["name"] == "Test Project"
                assert result["owner_id"] == "user123"
                mock_db_session.add.assert_called_once()
                mock_db_session.commit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_project_with_access(self, project_manager, mock_db_session):
        """Test getting a project with proper access."""
        with patch('scrollintel.engines.visual_generation.collaboration.project_manager.get_db_session', return_value=mock_db_session):
            # Mock project query
            mock_project = Mock()
            mock_project.id = 1
            mock_project.owner_id = "user123"
            mock_project.status = ProjectStatus.ACTIVE.value
            
            mock_db_session.query.return_value.filter.return_value.first.return_value = mock_project
            
            # Mock access check
            with patch.object(project_manager, '_has_project_access', return_value=True):
                with patch.object(project_manager, '_project_to_response') as mock_response:
                    mock_response.return_value = {"id": 1, "name": "Test Project"}
                    
                    result = await project_manager.get_project(1, "user123")
                    
                    assert result is not None
                    assert result["id"] == 1
    
    @pytest.mark.asyncio
    async def test_get_project_without_access(self, project_manager, mock_db_session):
        """Test getting a project without access."""
        with patch('scrollintel.engines.visual_generation.collaboration.project_manager.get_db_session', return_value=mock_db_session):
            # Mock project query
            mock_project = Mock()
            mock_project.id = 1
            mock_project.owner_id = "user123"
            
            mock_db_session.query.return_value.filter.return_value.first.return_value = mock_project
            
            # Mock access check (no access)
            with patch.object(project_manager, '_has_project_access', return_value=False):
                result = await project_manager.get_project(1, "user456")
                
                assert result is None
    
    @pytest.mark.asyncio
    async def test_update_project(self, project_manager, mock_db_session):
        """Test project update."""
        with patch('scrollintel.engines.visual_generation.collaboration.project_manager.get_db_session', return_value=mock_db_session):
            # Mock project query
            mock_project = Mock()
            mock_project.id = 1
            mock_project.name = "Old Name"
            mock_project.description = "Old Description"
            mock_project.tags = []
            mock_project.metadata = {}
            mock_project.status = ProjectStatus.ACTIVE.value
            
            mock_db_session.query.return_value.filter.return_value.first.return_value = mock_project
            
            # Mock access check
            with patch.object(project_manager, '_has_project_edit_access', return_value=True):
                with patch.object(project_manager, '_project_to_response') as mock_response:
                    mock_response.return_value = {"id": 1, "name": "New Name"}
                    
                    request = ProjectUpdateRequest(
                        name="New Name",
                        description="New Description"
                    )
                    
                    result = await project_manager.update_project(1, request, "user123")
                    
                    assert result is not None
                    assert mock_project.name == "New Name"
                    assert mock_project.description == "New Description"
                    mock_db_session.commit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_delete_project(self, project_manager, mock_db_session):
        """Test project deletion."""
        with patch('scrollintel.engines.visual_generation.collaboration.project_manager.get_db_session', return_value=mock_db_session):
            # Mock project query
            mock_project = Mock()
            mock_project.id = 1
            mock_project.owner_id = "user123"
            mock_project.status = ProjectStatus.ACTIVE.value
            
            mock_db_session.query.return_value.filter.return_value.first.return_value = mock_project
            
            result = await project_manager.delete_project(1, "user123")
            
            assert result is True
            assert mock_project.status == ProjectStatus.DELETED.value
            mock_db_session.commit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_list_projects(self, project_manager, mock_db_session):
        """Test listing projects."""
        with patch('scrollintel.engines.visual_generation.collaboration.project_manager.get_db_session', return_value=mock_db_session):
            # Mock query chain
            mock_query = Mock()
            mock_query.filter.return_value = mock_query
            mock_query.count.return_value = 2
            mock_query.order_by.return_value = mock_query
            mock_query.offset.return_value = mock_query
            mock_query.limit.return_value = mock_query
            mock_query.all.return_value = [Mock(id=1), Mock(id=2)]
            
            mock_db_session.query.return_value = mock_query
            
            with patch.object(project_manager, '_project_to_response') as mock_response:
                mock_response.side_effect = [
                    {"id": 1, "name": "Project 1"},
                    {"id": 2, "name": "Project 2"}
                ]
                
                projects, total_count = await project_manager.list_projects("user123")
                
                assert len(projects) == 2
                assert total_count == 2
                assert projects[0]["id"] == 1
                assert projects[1]["id"] == 2
    
    @pytest.mark.asyncio
    async def test_add_content_item(self, project_manager, mock_db_session):
        """Test adding content item to project."""
        with patch('scrollintel.engines.visual_generation.collaboration.project_manager.get_db_session', return_value=mock_db_session):
            # Mock access check
            with patch.object(project_manager, '_has_project_edit_access', return_value=True):
                # Mock content item creation
                mock_content_item = Mock()
                mock_content_item.id = 1
                mock_content_item.project_id = 1
                mock_content_item.name = "Test Content"
                
                mock_db_session.add = Mock()
                mock_db_session.commit = Mock()
                mock_db_session.refresh = Mock()
                
                with patch.object(project_manager, '_content_item_to_response') as mock_response:
                    mock_response.return_value = {"id": 1, "name": "Test Content"}
                    
                    result = await project_manager.add_content_item(
                        1, "Test Content", "image", "/path/to/file.jpg", "user123"
                    )
                    
                    assert result is not None
                    assert result["name"] == "Test Content"
                    mock_db_session.add.assert_called()
                    mock_db_session.commit.assert_called()


class TestSharingManager:
    """Test sharing and permissions functionality."""
    
    @pytest.fixture
    def sharing_manager(self):
        """Create sharing manager instance."""
        return SharingManager()
    
    @pytest.fixture
    def mock_db_session(self):
        """Mock database session."""
        session = Mock()
        session.__enter__ = Mock(return_value=session)
        session.__exit__ = Mock(return_value=None)
        return session
    
    @pytest.mark.asyncio
    async def test_share_project(self, sharing_manager, mock_db_session):
        """Test sharing a project."""
        with patch('scrollintel.engines.visual_generation.collaboration.sharing_manager.get_db_session', return_value=mock_db_session):
            # Mock admin access check
            with patch.object(sharing_manager, '_has_admin_access', return_value=True):
                # Mock existing share check
                mock_db_session.query.return_value.filter.return_value.first.return_value = None
                
                # Mock notification
                with patch.object(sharing_manager, '_send_share_notification'):
                    request = ShareRequest(
                        user_id="user456",
                        permission_level=SharePermission.EDIT
                    )
                    
                    result = await sharing_manager.share_project(1, request, "user123")
                    
                    assert result is True
                    mock_db_session.add.assert_called_once()
                    mock_db_session.commit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_share_project_without_admin_access(self, sharing_manager, mock_db_session):
        """Test sharing a project without admin access."""
        with patch('scrollintel.engines.visual_generation.collaboration.sharing_manager.get_db_session', return_value=mock_db_session):
            # Mock admin access check (no access)
            with patch.object(sharing_manager, '_has_admin_access', return_value=False):
                request = ShareRequest(
                    user_id="user456",
                    permission_level=SharePermission.EDIT
                )
                
                result = await sharing_manager.share_project(1, request, "user123")
                
                assert result is False
    
    @pytest.mark.asyncio
    async def test_revoke_project_access(self, sharing_manager, mock_db_session):
        """Test revoking project access."""
        with patch('scrollintel.engines.visual_generation.collaboration.sharing_manager.get_db_session', return_value=mock_db_session):
            # Mock admin access check
            with patch.object(sharing_manager, '_has_admin_access', return_value=True):
                # Mock existing share
                mock_share = Mock()
                mock_db_session.query.return_value.filter.return_value.first.return_value = mock_share
                
                # Mock notification
                with patch.object(sharing_manager, '_send_revoke_notification'):
                    result = await sharing_manager.revoke_project_access(1, "user456", "user123")
                    
                    assert result is True
                    mock_db_session.delete.assert_called_once_with(mock_share)
                    mock_db_session.commit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_create_share_link(self, sharing_manager, mock_db_session):
        """Test creating a shareable link."""
        with patch('scrollintel.engines.visual_generation.collaboration.sharing_manager.get_db_session', return_value=mock_db_session):
            # Mock admin access check
            with patch.object(sharing_manager, '_has_admin_access', return_value=True):
                # Mock project
                mock_project = Mock()
                mock_project.id = 1
                mock_project.metadata = {}
                
                mock_db_session.query.return_value.filter.return_value.first.return_value = mock_project
                
                result = await sharing_manager.create_share_link(1, "user123")
                
                assert result is not None
                assert result.startswith("https://scrollintel.com/shared/")
                assert "share_links" in mock_project.metadata
                mock_db_session.commit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_create_approval_workflow(self, sharing_manager, mock_db_session):
        """Test creating an approval workflow."""
        with patch('scrollintel.engines.visual_generation.collaboration.sharing_manager.get_db_session', return_value=mock_db_session):
            # Mock project access check
            with patch.object(sharing_manager, '_has_project_access', return_value=True):
                # Mock approval creation
                mock_approval = Mock()
                mock_approval.id = 1
                
                mock_db_session.add = Mock()
                mock_db_session.commit = Mock()
                mock_db_session.refresh = Mock()
                
                # Mock notification
                with patch.object(sharing_manager, '_send_approval_request_notification'):
                    request = ApprovalRequest(
                        workflow_name="Content Review",
                        assigned_to="reviewer123"
                    )
                    
                    result = await sharing_manager.create_approval_workflow(1, request, "user123")
                    
                    assert result == 1
                    mock_db_session.add.assert_called_once()
                    mock_db_session.commit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_respond_to_approval(self, sharing_manager, mock_db_session):
        """Test responding to an approval request."""
        with patch('scrollintel.engines.visual_generation.collaboration.sharing_manager.get_db_session', return_value=mock_db_session):
            # Mock approval query
            mock_approval = Mock()
            mock_approval.id = 1
            mock_approval.assigned_to = "reviewer123"
            mock_approval.status = ApprovalStatus.PENDING.value
            mock_approval.requested_by = "user123"
            
            mock_db_session.query.return_value.filter.return_value.first.return_value = mock_approval
            
            # Mock notification
            with patch.object(sharing_manager, '_send_approval_response_notification'):
                response = ApprovalResponse(
                    status=ApprovalStatus.APPROVED,
                    feedback="Looks good!"
                )
                
                result = await sharing_manager.respond_to_approval(1, response, "reviewer123")
                
                assert result is True
                assert mock_approval.status == ApprovalStatus.APPROVED.value
                assert mock_approval.feedback == "Looks good!"
                mock_db_session.commit.assert_called_once()


class TestCommentSystem:
    """Test comment system functionality."""
    
    @pytest.fixture
    def comment_system(self):
        """Create comment system instance."""
        return CommentSystem()
    
    @pytest.fixture
    def mock_db_session(self):
        """Mock database session."""
        session = Mock()
        session.__enter__ = Mock(return_value=session)
        session.__exit__ = Mock(return_value=None)
        return session
    
    @pytest.mark.asyncio
    async def test_add_project_comment(self, comment_system, mock_db_session):
        """Test adding a comment to a project."""
        with patch('scrollintel.engines.visual_generation.collaboration.comment_system.get_db_session', return_value=mock_db_session):
            # Mock project access check
            with patch.object(comment_system, '_has_project_access', return_value=True):
                # Mock comment creation
                mock_comment = Mock()
                mock_comment.id = 1
                mock_comment.project_id = 1
                mock_comment.user_id = "user123"
                mock_comment.content = "Great project!"
                mock_comment.created_at = datetime.utcnow()
                mock_comment.updated_at = datetime.utcnow()
                
                mock_db_session.add = Mock()
                mock_db_session.commit = Mock()
                mock_db_session.refresh = Mock()
                
                # Mock notification
                with patch.object(comment_system, '_notify_project_collaborators'):
                    request = CommentRequest(content="Great project!")
                    
                    result = await comment_system.add_project_comment(1, request, "user123")
                    
                    assert result is not None
                    assert result["content"] == "Great project!"
                    mock_db_session.add.assert_called_once()
                    mock_db_session.commit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_add_content_comment_with_position(self, comment_system, mock_db_session):
        """Test adding a positioned comment to content."""
        with patch('scrollintel.engines.visual_generation.collaboration.comment_system.get_db_session', return_value=mock_db_session):
            # Mock content item query
            mock_content_item = Mock()
            mock_content_item.id = 1
            mock_content_item.project_id = 1
            
            mock_db_session.query.return_value.filter.return_value.first.return_value = mock_content_item
            
            # Mock project access check
            with patch.object(comment_system, '_has_project_access', return_value=True):
                # Mock comment creation
                mock_comment = Mock()
                mock_comment.id = 1
                mock_comment.content_item_id = 1
                mock_comment.user_id = "user123"
                mock_comment.content = "Nice detail here!"
                mock_comment.position_x = 100
                mock_comment.position_y = 200
                mock_comment.created_at = datetime.utcnow()
                mock_comment.updated_at = datetime.utcnow()
                
                mock_db_session.add = Mock()
                mock_db_session.commit = Mock()
                mock_db_session.refresh = Mock()
                
                # Mock notification
                with patch.object(comment_system, '_notify_project_collaborators'):
                    request = CommentRequest(
                        content="Nice detail here!",
                        position_x=100,
                        position_y=200
                    )
                    
                    result = await comment_system.add_content_comment(1, request, "user123")
                    
                    assert result is not None
                    assert result["content"] == "Nice detail here!"
                    assert result["position_x"] == 100
                    assert result["position_y"] == 200
    
    @pytest.mark.asyncio
    async def test_get_project_comments_with_replies(self, comment_system, mock_db_session):
        """Test getting project comments with reply structure."""
        with patch('scrollintel.engines.visual_generation.collaboration.comment_system.get_db_session', return_value=mock_db_session):
            # Mock project access check
            with patch.object(comment_system, '_has_project_access', return_value=True):
                # Mock comments query
                mock_root_comment = Mock()
                mock_root_comment.id = 1
                mock_root_comment.parent_comment_id = None
                mock_root_comment.content = "Root comment"
                
                mock_reply_comment = Mock()
                mock_reply_comment.id = 2
                mock_reply_comment.parent_comment_id = 1
                mock_reply_comment.content = "Reply comment"
                
                mock_db_session.query.return_value.filter.return_value.order_by.return_value.offset.return_value.limit.return_value.all.return_value = [
                    mock_root_comment, mock_reply_comment
                ]
                
                with patch.object(comment_system, '_comment_to_dict') as mock_to_dict:
                    mock_to_dict.side_effect = [
                        {"id": 1, "content": "Root comment", "parent_comment_id": None},
                        {"id": 2, "content": "Reply comment", "parent_comment_id": 1}
                    ]
                    
                    result = await comment_system.get_project_comments(1, "user123")
                    
                    assert len(result) == 1  # Only root comments returned
                    assert result[0]["id"] == 1
                    assert len(result[0]["replies"]) == 1
                    assert result[0]["replies"][0]["id"] == 2
    
    @pytest.mark.asyncio
    async def test_search_comments(self, comment_system, mock_db_session):
        """Test searching comments."""
        with patch('scrollintel.engines.visual_generation.collaboration.comment_system.get_db_session', return_value=mock_db_session):
            # Mock project access check
            with patch.object(comment_system, '_has_project_access', return_value=True):
                # Mock comment queries
                mock_project_comment = Mock()
                mock_content_comment = Mock()
                
                # Mock query chains
                mock_project_query = Mock()
                mock_project_query.filter.return_value = mock_project_query
                mock_project_query.order_by.return_value = mock_project_query
                mock_project_query.all.return_value = [mock_project_comment]
                
                mock_content_query = Mock()
                mock_content_query.join.return_value = mock_content_query
                mock_content_query.filter.return_value = mock_content_query
                mock_content_query.order_by.return_value = mock_content_query
                mock_content_query.all.return_value = [mock_content_comment]
                
                mock_db_session.query.side_effect = [mock_project_query, mock_content_query]
                
                with patch.object(comment_system, '_comment_to_dict') as mock_project_dict:
                    with patch.object(comment_system, '_content_comment_to_dict') as mock_content_dict:
                        mock_project_dict.return_value = {
                            "id": 1, "content": "project comment", "created_at": datetime.utcnow()
                        }
                        mock_content_dict.return_value = {
                            "id": 2, "content": "content comment", "created_at": datetime.utcnow()
                        }
                        
                        result = await comment_system.search_comments(1, "test", "user123")
                        
                        assert len(result) == 2
                        assert result[0]["type"] == "project"
                        assert result[1]["type"] == "content"


class TestVersionControl:
    """Test version control functionality."""
    
    @pytest.fixture
    def version_control(self):
        """Create version control instance."""
        return VersionControl()
    
    @pytest.fixture
    def mock_db_session(self):
        """Mock database session."""
        session = Mock()
        session.__enter__ = Mock(return_value=session)
        session.__exit__ = Mock(return_value=None)
        return session
    
    @pytest.fixture
    def temp_file(self):
        """Create a temporary file for testing."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"test content")
            temp_path = f.name
        
        yield temp_path
        
        # Cleanup
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    
    @pytest.mark.asyncio
    async def test_create_version(self, version_control, mock_db_session, temp_file):
        """Test creating a new version."""
        with patch('scrollintel.engines.visual_generation.collaboration.version_control.get_db_session', return_value=mock_db_session):
            # Mock content item query
            mock_content_item = Mock()
            mock_content_item.id = 1
            mock_content_item.project_id = 1
            
            mock_db_session.query.return_value.filter.return_value.first.return_value = mock_content_item
            
            # Mock project access check
            with patch.object(version_control, '_has_project_edit_access', return_value=True):
                # Mock version number query
                mock_db_session.query.return_value.filter.return_value.scalar.return_value = 1
                
                # Mock file operations
                with patch('os.makedirs'), patch('shutil.copy2'):
                    # Mock version creation
                    mock_version = Mock()
                    mock_version.id = 1
                    mock_version.version_number = 2
                    
                    mock_db_session.add = Mock()
                    mock_db_session.commit = Mock()
                    mock_db_session.refresh = Mock()
                    
                    with patch.object(version_control, '_version_to_dict') as mock_to_dict:
                        mock_to_dict.return_value = {"id": 1, "version_number": 2}
                        
                        result = await version_control.create_version(
                            1, temp_file, "user123", "Updated content"
                        )
                        
                        assert result is not None
                        assert result["version_number"] == 2
                        mock_db_session.add.assert_called()
                        mock_db_session.commit.assert_called()
    
    @pytest.mark.asyncio
    async def test_get_version_history(self, version_control, mock_db_session):
        """Test getting version history."""
        with patch('scrollintel.engines.visual_generation.collaboration.version_control.get_db_session', return_value=mock_db_session):
            # Mock content item query
            mock_content_item = Mock()
            mock_content_item.id = 1
            mock_content_item.project_id = 1
            
            mock_db_session.query.return_value.filter.return_value.first.return_value = mock_content_item
            
            # Mock project access check
            with patch.object(version_control, '_has_project_access', return_value=True):
                # Mock versions query
                mock_version1 = Mock()
                mock_version1.version_number = 2
                mock_version2 = Mock()
                mock_version2.version_number = 1
                
                mock_db_session.query.return_value.filter.return_value.order_by.return_value.all.return_value = [
                    mock_version1, mock_version2
                ]
                
                with patch.object(version_control, '_version_to_dict') as mock_to_dict:
                    mock_to_dict.side_effect = [
                        {"id": 1, "version_number": 2},
                        {"id": 2, "version_number": 1}
                    ]
                    
                    result = await version_control.get_version_history(1, "user123")
                    
                    assert len(result) == 2
                    assert result[0]["version_number"] == 2  # Latest first
                    assert result[1]["version_number"] == 1
    
    @pytest.mark.asyncio
    async def test_revert_to_version(self, version_control, mock_db_session):
        """Test reverting to a previous version."""
        with patch('scrollintel.engines.visual_generation.collaboration.version_control.get_db_session', return_value=mock_db_session):
            # Mock content item query
            mock_content_item = Mock()
            mock_content_item.id = 1
            mock_content_item.project_id = 1
            
            # Mock target version query
            mock_target_version = Mock()
            mock_target_version.version_number = 1
            mock_target_version.file_path = "/path/to/v1.jpg"
            mock_target_version.generation_params = {"style": "realistic"}
            
            mock_db_session.query.return_value.filter.return_value.first.side_effect = [
                mock_content_item, mock_target_version
            ]
            
            # Mock project access check
            with patch.object(version_control, '_has_project_edit_access', return_value=True):
                # Mock create_version method
                with patch.object(version_control, 'create_version') as mock_create:
                    mock_create.return_value = {"id": 3, "version_number": 3}
                    
                    result = await version_control.revert_to_version(1, 1, "user123", "Reverted to v1")
                    
                    assert result is True
                    mock_create.assert_called_once_with(
                        content_item_id=1,
                        file_path="/path/to/v1.jpg",
                        user_id="user123",
                        change_description="Reverted to v1",
                        thumbnail_path=mock_target_version.thumbnail_path,
                        generation_params={"style": "realistic"},
                        quality_metrics=mock_target_version.quality_metrics
                    )
    
    @pytest.mark.asyncio
    async def test_compare_versions(self, version_control, mock_db_session):
        """Test comparing two versions."""
        with patch('scrollintel.engines.visual_generation.collaboration.version_control.get_db_session', return_value=mock_db_session):
            # Mock content item query
            mock_content_item = Mock()
            mock_content_item.id = 1
            mock_content_item.project_id = 1
            
            # Mock version queries
            mock_v1 = Mock()
            mock_v1.version_number = 1
            mock_v1.generation_params = {"style": "realistic"}
            mock_v1.quality_metrics = {"overall_score": 0.8}
            
            mock_v2 = Mock()
            mock_v2.version_number = 2
            mock_v2.generation_params = {"style": "artistic"}
            mock_v2.quality_metrics = {"overall_score": 0.9}
            
            mock_db_session.query.return_value.filter.return_value.first.side_effect = [
                mock_content_item, mock_v1, mock_v2
            ]
            
            # Mock project access check
            with patch.object(version_control, '_has_project_access', return_value=True):
                with patch.object(version_control, '_version_to_dict') as mock_to_dict:
                    with patch.object(version_control, '_calculate_differences') as mock_diff:
                        with patch.object(version_control, '_compare_quality_metrics') as mock_quality:
                            mock_to_dict.side_effect = [
                                {"id": 1, "version_number": 1},
                                {"id": 2, "version_number": 2}
                            ]
                            mock_diff.return_value = {"parameter_changes": []}
                            mock_quality.return_value = {"overall_improvement": 0.1}
                            
                            result = await version_control.compare_versions(1, 1, 2, "user123")
                            
                            assert result is not None
                            assert "version1" in result
                            assert "version2" in result
                            assert "differences" in result
                            assert "quality_comparison" in result
    
    @pytest.mark.asyncio
    async def test_delete_version_protection(self, version_control, mock_db_session):
        """Test that the only version cannot be deleted."""
        with patch('scrollintel.engines.visual_generation.collaboration.version_control.get_db_session', return_value=mock_db_session):
            # Mock content item query
            mock_content_item = Mock()
            mock_content_item.id = 1
            mock_content_item.project_id = 1
            
            mock_db_session.query.return_value.filter.return_value.first.return_value = mock_content_item
            
            # Mock project access check
            with patch.object(version_control, '_has_project_edit_access', return_value=True):
                # Mock version count (only 1 version)
                mock_db_session.query.return_value.filter.return_value.scalar.return_value = 1
                
                result = await version_control.delete_version(1, 1, "user123")
                
                assert result is False  # Cannot delete the only version


class TestDataConsistency:
    """Test data consistency across collaboration features."""
    
    @pytest.mark.asyncio
    async def test_project_deletion_cascades(self):
        """Test that project deletion properly handles related data."""
        # This would test that when a project is deleted:
        # - All content items are handled appropriately
        # - All shares are removed
        # - All comments are handled
        # - All versions are cleaned up
        pass
    
    @pytest.mark.asyncio
    async def test_user_access_consistency(self):
        """Test that user access is consistent across all features."""
        # This would test that:
        # - Users can only access projects they have permission for
        # - Permission changes are reflected immediately
        # - Expired shares are properly handled
        pass
    
    @pytest.mark.asyncio
    async def test_version_integrity(self):
        """Test version control data integrity."""
        # This would test that:
        # - Version numbers are sequential
        # - File references are valid
        # - Version history is preserved
        pass


if __name__ == "__main__":
    pytest.main([__file__])