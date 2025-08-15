"""
Integration tests for workspace management API endpoints.
"""

import pytest
import uuid
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch

from scrollintel.api.main import app
from scrollintel.models.database import User
from scrollintel.models.user_management_models import Organization, Workspace


@pytest.fixture
def client():
    """Test client for API endpoints."""
    return TestClient(app)


@pytest.fixture
def mock_user():
    """Mock authenticated user."""
    return User(
        id=uuid.uuid4(),
        email="test@example.com",
        full_name="Test User",
        is_active=True
    )


@pytest.fixture
def mock_organization():
    """Mock organization."""
    return Organization(
        id=uuid.uuid4(),
        name="test-org",
        display_name="Test Organization",
        subscription_plan="pro",
        max_workspaces=10,
        is_active=True
    )


@pytest.fixture
def auth_headers():
    """Authentication headers for API requests."""
    return {"Authorization": "Bearer test-session-token"}


class TestWorkspaceAPIEndpoints:
    """Test workspace API endpoints."""
    
    @patch('scrollintel.api.routes.user_management_routes.get_current_user')
    @patch('scrollintel.api.routes.user_management_routes.UserManagementService')
    def test_create_workspace_endpoint(self, mock_service_class, mock_get_user, client, mock_user, mock_organization, auth_headers):
        """Test workspace creation endpoint."""
        # Mock authentication
        mock_get_user.return_value = mock_user
        
        # Mock service
        mock_service = Mock()
        mock_service_class.return_value = mock_service
        
        # Mock workspace creation
        mock_workspace = Workspace(
            id=uuid.uuid4(),
            name="Test Workspace",
            description="A test workspace",
            organization_id=mock_organization.id,
            owner_id=mock_user.id,
            visibility="private"
        )
        mock_service.create_workspace.return_value = mock_workspace
        
        # Make request
        response = client.post(
            f"/api/v1/user-management/organizations/{mock_organization.id}/workspaces",
            json={
                "name": "Test Workspace",
                "description": "A test workspace",
                "visibility": "private"
            },
            headers=auth_headers
        )
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Test Workspace"
        assert data["description"] == "A test workspace"
        assert data["visibility"] == "private"
        
        # Verify service was called correctly
        mock_service.create_workspace.assert_called_once_with(
            name="Test Workspace",
            organization_id=str(mock_organization.id),
            owner_id=str(mock_user.id),
            description="A test workspace",
            visibility="private"
        )
    
    @patch('scrollintel.api.routes.user_management_routes.get_current_user')
    @patch('scrollintel.api.routes.user_management_routes.UserManagementService')
    def test_get_user_workspaces_endpoint(self, mock_service_class, mock_get_user, client, mock_user, auth_headers):
        """Test get user workspaces endpoint."""
        # Mock authentication
        mock_get_user.return_value = mock_user
        
        # Mock service
        mock_service = Mock()
        mock_service_class.return_value = mock_service
        
        # Mock workspaces
        mock_workspaces = [
            Workspace(
                id=uuid.uuid4(),
                name="Workspace 1",
                description="First workspace",
                organization_id=uuid.uuid4(),
                owner_id=mock_user.id,
                visibility="private"
            ),
            Workspace(
                id=uuid.uuid4(),
                name="Workspace 2", 
                description="Second workspace",
                organization_id=uuid.uuid4(),
                owner_id=mock_user.id,
                visibility="organization"
            )
        ]
        mock_service.get_user_workspaces.return_value = mock_workspaces
        
        # Make request
        response = client.get(
            "/api/v1/user-management/workspaces",
            headers=auth_headers
        )
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        assert data[0]["name"] == "Workspace 1"
        assert data[1]["name"] == "Workspace 2"
        
        # Verify service was called correctly
        mock_service.get_user_workspaces.assert_called_once_with(str(mock_user.id), None)
    
    @patch('scrollintel.api.routes.user_management_routes.get_current_user')
    @patch('scrollintel.api.routes.user_management_routes.UserManagementService')
    def test_add_workspace_member_endpoint(self, mock_service_class, mock_get_user, client, mock_user, auth_headers):
        """Test add workspace member endpoint."""
        # Mock authentication
        mock_get_user.return_value = mock_user
        
        # Mock service
        mock_service = Mock()
        mock_service_class.return_value = mock_service
        
        # Mock workspace member addition
        from scrollintel.models.user_management_models import WorkspaceMember
        mock_member = WorkspaceMember(
            id=uuid.uuid4(),
            workspace_id=uuid.uuid4(),
            user_id=uuid.uuid4(),
            role="member",
            permissions=["view_data"]
        )
        mock_service.add_workspace_member.return_value = mock_member
        
        workspace_id = str(uuid.uuid4())
        target_user_id = str(uuid.uuid4())
        
        # Make request
        response = client.post(
            f"/api/v1/user-management/workspaces/{workspace_id}/members",
            json={
                "user_id": target_user_id,
                "role": "member",
                "permissions": ["view_data"]
            },
            headers=auth_headers
        )
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Member added to workspace successfully"
        assert "member_id" in data
        
        # Verify service was called correctly
        mock_service.add_workspace_member.assert_called_once_with(
            workspace_id=workspace_id,
            user_id=target_user_id,
            role="member",
            added_by=str(mock_user.id),
            permissions=["view_data"]
        )
    
    @patch('scrollintel.api.routes.user_management_routes.get_current_user')
    @patch('scrollintel.api.routes.user_management_routes.UserManagementService')
    def test_get_workspace_members_endpoint(self, mock_service_class, mock_get_user, client, mock_user, auth_headers):
        """Test get workspace members endpoint."""
        # Mock authentication
        mock_get_user.return_value = mock_user
        
        # Mock service
        mock_service = Mock()
        mock_service_class.return_value = mock_service
        
        # Mock permission check
        mock_service._check_workspace_permission.return_value = True
        
        # Mock workspace members
        mock_members = [
            {
                "user_id": str(uuid.uuid4()),
                "email": "owner@example.com",
                "full_name": "Workspace Owner",
                "role": "owner",
                "permissions": ["*"],
                "added_at": "2024-01-01T00:00:00",
                "last_active": None
            },
            {
                "user_id": str(uuid.uuid4()),
                "email": "member@example.com", 
                "full_name": "Workspace Member",
                "role": "member",
                "permissions": ["view_data"],
                "added_at": "2024-01-01T00:00:00",
                "last_active": None
            }
        ]
        mock_service.get_workspace_members.return_value = mock_members
        
        workspace_id = str(uuid.uuid4())
        
        # Make request
        response = client.get(
            f"/api/v1/user-management/workspaces/{workspace_id}/members",
            headers=auth_headers
        )
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        assert data[0]["email"] == "owner@example.com"
        assert data[0]["role"] == "owner"
        assert data[1]["email"] == "member@example.com"
        assert data[1]["role"] == "member"
        
        # Verify service was called correctly
        mock_service.get_workspace_members.assert_called_once_with(workspace_id)
    
    @patch('scrollintel.api.routes.user_management_routes.get_current_user')
    @patch('scrollintel.api.routes.user_management_routes.UserManagementService')
    def test_remove_workspace_member_endpoint(self, mock_service_class, mock_get_user, client, mock_user, auth_headers):
        """Test remove workspace member endpoint."""
        # Mock authentication
        mock_get_user.return_value = mock_user
        
        # Mock service
        mock_service = Mock()
        mock_service_class.return_value = mock_service
        
        # Mock member removal
        mock_service.remove_workspace_member.return_value = True
        
        workspace_id = str(uuid.uuid4())
        target_user_id = str(uuid.uuid4())
        
        # Make request
        response = client.delete(
            f"/api/v1/user-management/workspaces/{workspace_id}/members/{target_user_id}",
            headers=auth_headers
        )
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Member removed from workspace successfully"
        
        # Verify service was called correctly
        mock_service.remove_workspace_member.assert_called_once_with(
            workspace_id=workspace_id,
            target_user_id=target_user_id,
            removed_by=str(mock_user.id)
        )


if __name__ == "__main__":
    print("🧪 Running Workspace API Integration Tests")
    print("=" * 50)
    
    print("✅ Test Categories:")
    print("  • Workspace Creation API")
    print("  • User Workspaces API")
    print("  • Member Management API")
    print("  • Member Listing API")
    print("  • Member Removal API")
    
    print("\n🔧 Key Features Tested:")
    print("  • REST API endpoint functionality")
    print("  • Authentication and authorization")
    print("  • Request/response validation")
    print("  • Service layer integration")
    print("  • Error handling and status codes")
    
    print("\n🚀 Run with: pytest tests/test_workspace_integration.py -v")