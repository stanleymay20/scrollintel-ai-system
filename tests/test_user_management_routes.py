"""
Tests for user management API routes.
"""

import pytest
import uuid
from datetime import datetime, timedelta
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch

from scrollintel.api.routes.user_management_routes import router
from scrollintel.models.database import User
from scrollintel.models.user_management_models import Organization, Workspace
from scrollintel.core.interfaces import UserRole


class TestUserManagementRoutes:
    """Test cases for user management API routes."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        from fastapi import FastAPI
        app = FastAPI()
        app.include_router(router)
        return TestClient(app)
    
    @pytest.fixture
    def mock_user(self):
        """Create mock user."""
        return User(
            id=uuid.uuid4(),
            email="test@example.com",
            full_name="Test User",
            role=UserRole.ADMIN,
            is_active=True
        )
    
    @pytest.fixture
    def mock_organization(self):
        """Create mock organization."""
        return Organization(
            id=uuid.uuid4(),
            name="test-org",
            display_name="Test Organization",
            subscription_plan="free",
            subscription_status="active",
            max_users=10,
            max_workspaces=5,
            created_at=datetime.utcnow()
        )
    
    @pytest.fixture
    def auth_headers(self):
        """Create authentication headers."""
        return {"Authorization": "Bearer test-token"}
    
    # Organization Management Route Tests
    
    @patch('scrollintel.api.routes.user_management_routes.get_current_user')
    @patch('scrollintel.api.routes.user_management_routes.UserManagementService')
    def test_create_organization_success(self, mock_service_class, mock_get_user, client, mock_user, mock_organization, auth_headers):
        """Test successful organization creation."""
        # Mock dependencies
        mock_get_user.return_value = mock_user
        mock_service = Mock()
        mock_service_class.return_value = mock_service
        mock_service.create_organization.return_value = mock_organization
        
        # Make request
        response = client.post(
            "/api/v1/user-management/organizations",
            json={
                "name": "test-org",
                "display_name": "Test Organization",
                "description": "Test description"
            },
            headers=auth_headers
        )
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "test-org"
        assert data["display_name"] == "Test Organization"
        
        # Verify service was called
        mock_service.create_organization.assert_called_once()
    
    @patch('scrollintel.api.routes.user_management_routes.get_current_user')
    @patch('scrollintel.api.routes.user_management_routes.UserManagementService')
    def test_create_organization_validation_error(self, mock_service_class, mock_get_user, client, mock_user, auth_headers):
        """Test organization creation with validation error."""
        # Mock dependencies
        mock_get_user.return_value = mock_user
        mock_service = Mock()
        mock_service_class.return_value = mock_service
        mock_service.create_organization.side_effect = Exception("Organization name already exists")
        
        # Make request
        response = client.post(
            "/api/v1/user-management/organizations",
            json={
                "name": "existing-org",
                "display_name": "Existing Organization"
            },
            headers=auth_headers
        )
        
        # Verify response
        assert response.status_code == 500
    
    @patch('scrollintel.api.routes.user_management_routes.get_current_user')
    @patch('scrollintel.api.routes.user_management_routes.UserManagementService')
    def test_get_organization_success(self, mock_service_class, mock_get_user, client, mock_user, mock_organization, auth_headers):
        """Test successful organization retrieval."""
        # Mock dependencies
        mock_get_user.return_value = mock_user
        mock_service = Mock()
        mock_service_class.return_value = mock_service
        mock_service._check_organization_permission.return_value = True
        mock_service.get_organization.return_value = mock_organization
        
        # Make request
        response = client.get(
            f"/api/v1/user-management/organizations/{mock_organization.id}",
            headers=auth_headers
        )
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "test-org"
    
    @patch('scrollintel.api.routes.user_management_routes.get_current_user')
    @patch('scrollintel.api.routes.user_management_routes.UserManagementService')
    def test_get_organization_access_denied(self, mock_service_class, mock_get_user, client, mock_user, auth_headers):
        """Test organization retrieval with access denied."""
        # Mock dependencies
        mock_get_user.return_value = mock_user
        mock_service = Mock()
        mock_service_class.return_value = mock_service
        mock_service._check_organization_permission.return_value = False
        
        # Make request
        response = client.get(
            f"/api/v1/user-management/organizations/{uuid.uuid4()}",
            headers=auth_headers
        )
        
        # Verify response
        assert response.status_code == 403
        assert "Access denied" in response.json()["detail"]
    
    # User Invitation Route Tests
    
    @patch('scrollintel.api.routes.user_management_routes.get_current_user')
    @patch('scrollintel.api.routes.user_management_routes.UserManagementService')
    def test_invite_user_success(self, mock_service_class, mock_get_user, client, mock_user, auth_headers):
        """Test successful user invitation."""
        # Mock dependencies
        mock_get_user.return_value = mock_user
        mock_service = Mock()
        mock_service_class.return_value = mock_service
        
        mock_invitation = Mock()
        mock_invitation.id = uuid.uuid4()
        mock_invitation.expires_at = datetime.utcnow() + timedelta(days=7)
        mock_service.invite_user.return_value = mock_invitation
        
        # Make request
        response = client.post(
            f"/api/v1/user-management/organizations/{uuid.uuid4()}/invitations",
            json={
                "email": "newuser@example.com",
                "role": "viewer",
                "message": "Welcome to our team!"
            },
            headers=auth_headers
        )
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert "Invitation sent successfully" in data["message"]
        assert "invitation_id" in data
    
    @patch('scrollintel.api.routes.user_management_routes.get_current_user')
    @patch('scrollintel.api.routes.user_management_routes.UserManagementService')
    def test_invite_user_invalid_email(self, mock_service_class, mock_get_user, client, mock_user, auth_headers):
        """Test user invitation with invalid email."""
        # Mock dependencies
        mock_get_user.return_value = mock_user
        
        # Make request with invalid email
        response = client.post(
            f"/api/v1/user-management/organizations/{uuid.uuid4()}/invitations",
            json={
                "email": "invalid-email",
                "role": "viewer"
            },
            headers=auth_headers
        )
        
        # Verify response
        assert response.status_code == 422  # Validation error
    
    @patch('scrollintel.api.routes.user_management_routes.get_current_user')
    @patch('scrollintel.api.routes.user_management_routes.UserManagementService')
    def test_accept_invitation_success(self, mock_service_class, mock_get_user, client, mock_user, auth_headers):
        """Test successful invitation acceptance."""
        # Mock dependencies
        mock_get_user.return_value = mock_user
        mock_service = Mock()
        mock_service_class.return_value = mock_service
        
        mock_org_user = Mock()
        mock_org_user.organization_id = uuid.uuid4()
        mock_org_user.role = UserRole.VIEWER
        mock_service.accept_invitation.return_value = mock_org_user
        
        # Make request
        response = client.post(
            "/api/v1/user-management/invitations/test-token/accept",
            headers=auth_headers
        )
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert "Invitation accepted successfully" in data["message"]
        assert "organization_id" in data
    
    # User Role Management Route Tests
    
    @patch('scrollintel.api.routes.user_management_routes.get_current_user')
    @patch('scrollintel.api.routes.user_management_routes.UserManagementService')
    def test_update_user_role_success(self, mock_service_class, mock_get_user, client, mock_user, auth_headers):
        """Test successful user role update."""
        # Mock dependencies
        mock_get_user.return_value = mock_user
        mock_service = Mock()
        mock_service_class.return_value = mock_service
        
        mock_org_user = Mock()
        mock_org_user.role = UserRole.ANALYST
        mock_service.update_user_role.return_value = mock_org_user
        
        # Make request
        org_id = uuid.uuid4()
        user_id = uuid.uuid4()
        response = client.put(
            f"/api/v1/user-management/organizations/{org_id}/users/{user_id}/role",
            json={
                "role": "analyst",
                "permissions": ["view_data", "create_projects"]
            },
            headers=auth_headers
        )
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert "User role updated successfully" in data["message"]
        assert data["new_role"] == "analyst"
    
    @patch('scrollintel.api.routes.user_management_routes.get_current_user')
    @patch('scrollintel.api.routes.user_management_routes.UserManagementService')
    def test_remove_user_success(self, mock_service_class, mock_get_user, client, mock_user, auth_headers):
        """Test successful user removal."""
        # Mock dependencies
        mock_get_user.return_value = mock_user
        mock_service = Mock()
        mock_service_class.return_value = mock_service
        mock_service.remove_user_from_organization.return_value = True
        
        # Make request
        org_id = uuid.uuid4()
        user_id = uuid.uuid4()
        response = client.delete(
            f"/api/v1/user-management/organizations/{org_id}/users/{user_id}",
            headers=auth_headers
        )
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert "User removed from organization successfully" in data["message"]
    
    # Workspace Management Route Tests
    
    @patch('scrollintel.api.routes.user_management_routes.get_current_user')
    @patch('scrollintel.api.routes.user_management_routes.UserManagementService')
    def test_create_workspace_success(self, mock_service_class, mock_get_user, client, mock_user, auth_headers):
        """Test successful workspace creation."""
        # Mock dependencies
        mock_get_user.return_value = mock_user
        mock_service = Mock()
        mock_service_class.return_value = mock_service
        
        mock_workspace = Workspace(
            id=uuid.uuid4(),
            name="Test Workspace",
            description="Test description",
            organization_id=uuid.uuid4(),
            owner_id=mock_user.id,
            visibility="private",
            created_at=datetime.utcnow()
        )
        mock_service.create_workspace.return_value = mock_workspace
        
        # Make request
        org_id = uuid.uuid4()
        response = client.post(
            f"/api/v1/user-management/organizations/{org_id}/workspaces",
            json={
                "name": "Test Workspace",
                "description": "Test description",
                "visibility": "private"
            },
            headers=auth_headers
        )
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Test Workspace"
        assert data["visibility"] == "private"
    
    @patch('scrollintel.api.routes.user_management_routes.get_current_user')
    @patch('scrollintel.api.routes.user_management_routes.UserManagementService')
    def test_get_user_workspaces_success(self, mock_service_class, mock_get_user, client, mock_user, auth_headers):
        """Test successful workspace retrieval."""
        # Mock dependencies
        mock_get_user.return_value = mock_user
        mock_service = Mock()
        mock_service_class.return_value = mock_service
        
        mock_workspaces = [
            Workspace(
                id=uuid.uuid4(),
                name="Workspace 1",
                organization_id=uuid.uuid4(),
                owner_id=mock_user.id,
                visibility="private",
                created_at=datetime.utcnow()
            ),
            Workspace(
                id=uuid.uuid4(),
                name="Workspace 2",
                organization_id=uuid.uuid4(),
                owner_id=mock_user.id,
                visibility="organization",
                created_at=datetime.utcnow()
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
    
    # API Key Management Route Tests
    
    @patch('scrollintel.api.routes.user_management_routes.get_current_user')
    @patch('scrollintel.api.routes.user_management_routes.UserManagementService')
    def test_create_api_key_success(self, mock_service_class, mock_get_user, client, mock_user, auth_headers):
        """Test successful API key creation."""
        # Mock dependencies
        mock_get_user.return_value = mock_user
        mock_service = Mock()
        mock_service_class.return_value = mock_service
        
        mock_api_key_record = Mock()
        mock_api_key_record.id = uuid.uuid4()
        mock_api_key_record.key_prefix = "sk_test_"
        mock_api_key = "sk_test_1234567890abcdef"
        
        mock_service.create_api_key.return_value = (mock_api_key_record, mock_api_key)
        
        # Make request
        org_id = uuid.uuid4()
        response = client.post(
            f"/api/v1/user-management/organizations/{org_id}/api-keys",
            json={
                "name": "Test API Key",
                "description": "Test description",
                "permissions": ["read_data"],
                "expires_days": 30
            },
            headers=auth_headers
        )
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert "API key created successfully" in data["message"]
        assert data["api_key"] == mock_api_key
        assert "key_prefix" in data
    
    # Authentication Tests
    
    def test_unauthorized_request(self, client):
        """Test request without authentication."""
        response = client.get("/api/v1/user-management/organizations/123")
        assert response.status_code == 403  # No authorization header
    
    @patch('scrollintel.api.routes.user_management_routes.UserManagementService')
    def test_invalid_token(self, mock_service_class, client):
        """Test request with invalid token."""
        # Mock service
        mock_service = Mock()
        mock_service_class.return_value = mock_service
        mock_service.validate_session.return_value = None
        
        # Make request with invalid token
        response = client.get(
            "/api/v1/user-management/organizations/123",
            headers={"Authorization": "Bearer invalid-token"}
        )
        
        assert response.status_code == 401
    
    # Input Validation Tests
    
    def test_create_organization_invalid_input(self, client, auth_headers):
        """Test organization creation with invalid input."""
        response = client.post(
            "/api/v1/user-management/organizations",
            json={
                "name": "",  # Empty name
                "display_name": "Test Organization"
            },
            headers=auth_headers
        )
        
        assert response.status_code == 422  # Validation error
    
    def test_create_workspace_invalid_visibility(self, client, auth_headers):
        """Test workspace creation with invalid visibility."""
        response = client.post(
            f"/api/v1/user-management/organizations/{uuid.uuid4()}/workspaces",
            json={
                "name": "Test Workspace",
                "visibility": "invalid"  # Invalid visibility
            },
            headers=auth_headers
        )
        
        assert response.status_code == 422  # Validation error
    
    def test_update_user_role_invalid_role(self, client, auth_headers):
        """Test user role update with invalid role."""
        org_id = uuid.uuid4()
        user_id = uuid.uuid4()
        response = client.put(
            f"/api/v1/user-management/organizations/{org_id}/users/{user_id}/role",
            json={
                "role": "invalid_role"  # Invalid role
            },
            headers=auth_headers
        )
        
        assert response.status_code == 422  # Validation error


if __name__ == "__main__":
    pytest.main([__file__])