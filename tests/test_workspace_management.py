"""
Tests for workspace management functionality.
"""

import pytest
import uuid
from datetime import datetime
from unittest.mock import Mock, AsyncMock
from sqlalchemy.orm import Session

from scrollintel.core.user_management import UserManagementService
from scrollintel.models.user_management_models import (
    Organization, OrganizationUser, Workspace, WorkspaceMember
)
from scrollintel.models.database import User
from scrollintel.core.interfaces import UserRole, SecurityError, ValidationError


@pytest.fixture
def mock_db():
    """Mock database session."""
    return Mock(spec=Session)


@pytest.fixture
def user_service(mock_db):
    """User management service with mocked database."""
    return UserManagementService(mock_db)


@pytest.fixture
def sample_organization():
    """Sample organization for testing."""
    return Organization(
        id=uuid.uuid4(),
        name="test-org",
        display_name="Test Organization",
        subscription_plan="pro",
        max_workspaces=10,
        is_active=True
    )


@pytest.fixture
def sample_user():
    """Sample user for testing."""
    return User(
        id=uuid.uuid4(),
        email="test@example.com",
        full_name="Test User",
        is_active=True
    )


@pytest.fixture
def sample_workspace(sample_organization, sample_user):
    """Sample workspace for testing."""
    return Workspace(
        id=uuid.uuid4(),
        name="Test Workspace",
        description="A test workspace",
        organization_id=sample_organization.id,
        owner_id=sample_user.id,
        visibility="private",
        is_active=True
    )


class TestWorkspaceCreation:
    """Test workspace creation functionality."""
    
    @pytest.mark.asyncio
    async def test_create_workspace_success(self, user_service, mock_db, sample_organization, sample_user):
        """Test successful workspace creation."""
        # Mock database queries
        mock_db.query.return_value.filter.return_value.first.return_value = sample_organization
        mock_db.query.return_value.filter.return_value.count.return_value = 2  # Under limit
        
        # Mock permission check
        user_service._check_organization_permission = Mock(return_value=True)
        user_service._log_audit_event = Mock()
        
        # Create workspace
        result = await user_service.create_workspace(
            name="New Workspace",
            organization_id=str(sample_organization.id),
            owner_id=str(sample_user.id),
            description="A new workspace",
            visibility="organization"
        )
        
        # Verify workspace creation
        assert result.name == "New Workspace"
        assert result.description == "A new workspace"
        assert result.visibility == "organization"
        assert str(result.organization_id) == str(sample_organization.id)
        assert str(result.owner_id) == str(sample_user.id)
        
        # Verify database operations
        assert mock_db.add.call_count == 2  # Workspace + WorkspaceMember
        mock_db.flush.assert_called_once()
        mock_db.commit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_create_workspace_no_permission(self, user_service, mock_db, sample_organization, sample_user):
        """Test workspace creation without permission."""
        # Mock permission check to fail
        user_service._check_organization_permission = Mock(return_value=False)
        
        # Attempt to create workspace
        with pytest.raises(SecurityError, match="Insufficient permissions"):
            await user_service.create_workspace(
                name="New Workspace",
                organization_id=str(sample_organization.id),
                owner_id=str(sample_user.id)
            )
    
    @pytest.mark.asyncio
    async def test_create_workspace_limit_exceeded(self, user_service, mock_db, sample_organization, sample_user):
        """Test workspace creation when limit is exceeded."""
        # Mock database queries
        mock_db.query.return_value.filter.return_value.first.return_value = sample_organization
        mock_db.query.return_value.filter.return_value.count.return_value = 10  # At limit
        
        # Mock permission check
        user_service._check_organization_permission = Mock(return_value=True)
        
        # Attempt to create workspace
        with pytest.raises(ValidationError, match="Maximum workspace limit"):
            await user_service.create_workspace(
                name="New Workspace",
                organization_id=str(sample_organization.id),
                owner_id=str(sample_user.id)
            )


class TestWorkspaceMemberManagement:
    """Test workspace member management functionality."""
    
    @pytest.mark.asyncio
    async def test_add_workspace_member_success(self, user_service, mock_db, sample_workspace, sample_user):
        """Test successful workspace member addition."""
        # Create another user to add
        new_user = User(id=uuid.uuid4(), email="newuser@example.com", is_active=True)
        
        # Mock database queries
        mock_db.query.return_value.filter.return_value.first.side_effect = [
            sample_workspace,  # Workspace lookup
            OrganizationUser(  # Organization user lookup
                organization_id=sample_workspace.organization_id,
                user_id=new_user.id,
                status="active"
            ),
            None  # No existing workspace member
        ]
        
        # Mock permission check
        user_service._check_workspace_permission = Mock(return_value=True)
        user_service._log_audit_event = Mock()
        
        # Add workspace member
        result = await user_service.add_workspace_member(
            workspace_id=str(sample_workspace.id),
            user_id=str(new_user.id),
            role="member",
            added_by=str(sample_user.id),
            permissions=["view_data"]
        )
        
        # Verify member addition
        assert str(result.workspace_id) == str(sample_workspace.id)
        assert str(result.user_id) == str(new_user.id)
        assert result.role == "member"
        assert result.permissions == ["view_data"]
        assert str(result.added_by) == str(sample_user.id)
        
        # Verify database operations
        mock_db.add.assert_called_once()
        mock_db.commit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_add_workspace_member_not_in_org(self, user_service, mock_db, sample_workspace, sample_user):
        """Test adding workspace member who is not in organization."""
        new_user = User(id=uuid.uuid4(), email="newuser@example.com", is_active=True)
        
        # Mock database queries
        mock_db.query.return_value.filter.return_value.first.side_effect = [
            sample_workspace,  # Workspace lookup
            None  # No organization user
        ]
        
        # Mock permission check
        user_service._check_workspace_permission = Mock(return_value=True)
        
        # Attempt to add workspace member
        with pytest.raises(ValidationError, match="not a member of the organization"):
            await user_service.add_workspace_member(
                workspace_id=str(sample_workspace.id),
                user_id=str(new_user.id),
                role="member",
                added_by=str(sample_user.id)
            )
    
    @pytest.mark.asyncio
    async def test_remove_workspace_member_success(self, user_service, mock_db, sample_workspace, sample_user):
        """Test successful workspace member removal."""
        # Create workspace member to remove
        member_to_remove = WorkspaceMember(
            id=uuid.uuid4(),
            workspace_id=sample_workspace.id,
            user_id=uuid.uuid4(),
            role="member"
        )
        
        # Mock database queries
        mock_db.query.return_value.filter.return_value.first.side_effect = [
            member_to_remove,  # Workspace member lookup
            sample_workspace   # Workspace lookup for audit
        ]
        
        # Mock permission check
        user_service._check_workspace_permission = Mock(return_value=True)
        user_service._log_audit_event = Mock()
        
        # Remove workspace member
        result = await user_service.remove_workspace_member(
            workspace_id=str(sample_workspace.id),
            target_user_id=str(member_to_remove.user_id),
            removed_by=str(sample_user.id)
        )
        
        # Verify member removal
        assert result is True
        
        # Verify database operations
        mock_db.delete.assert_called_once_with(member_to_remove)
        mock_db.commit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_remove_workspace_owner_fails(self, user_service, mock_db, sample_workspace, sample_user):
        """Test that removing workspace owner fails."""
        # Create workspace owner to remove
        owner_member = WorkspaceMember(
            id=uuid.uuid4(),
            workspace_id=sample_workspace.id,
            user_id=sample_user.id,
            role="owner"
        )
        
        # Mock database queries
        mock_db.query.return_value.filter.return_value.first.return_value = owner_member
        
        # Mock permission check
        user_service._check_workspace_permission = Mock(return_value=True)
        
        # Attempt to remove workspace owner
        with pytest.raises(ValidationError, match="Cannot remove workspace owner"):
            await user_service.remove_workspace_member(
                workspace_id=str(sample_workspace.id),
                target_user_id=str(sample_user.id),
                removed_by=str(sample_user.id)
            )


class TestWorkspaceQueries:
    """Test workspace query functionality."""
    
    @pytest.mark.asyncio
    async def test_get_user_workspaces(self, user_service, mock_db, sample_user):
        """Test getting user workspaces."""
        # Mock workspace query
        mock_workspaces = [
            Workspace(id=uuid.uuid4(), name="Workspace 1", organization_id=uuid.uuid4()),
            Workspace(id=uuid.uuid4(), name="Workspace 2", organization_id=uuid.uuid4())
        ]
        
        mock_db.query.return_value.join.return_value.filter.return_value.all.return_value = mock_workspaces
        
        # Get user workspaces
        result = await user_service.get_user_workspaces(str(sample_user.id))
        
        # Verify results
        assert len(result) == 2
        assert result[0].name == "Workspace 1"
        assert result[1].name == "Workspace 2"
    
    @pytest.mark.asyncio
    async def test_get_workspace_members(self, user_service, mock_db, sample_workspace):
        """Test getting workspace members."""
        # Mock member query
        mock_members = [
            (
                WorkspaceMember(
                    workspace_id=sample_workspace.id,
                    user_id=uuid.uuid4(),
                    role="owner",
                    permissions=["*"],
                    added_at=datetime.utcnow()
                ),
                User(
                    id=uuid.uuid4(),
                    email="owner@example.com",
                    full_name="Workspace Owner",
                    is_active=True
                )
            ),
            (
                WorkspaceMember(
                    workspace_id=sample_workspace.id,
                    user_id=uuid.uuid4(),
                    role="member",
                    permissions=["view_data"],
                    added_at=datetime.utcnow()
                ),
                User(
                    id=uuid.uuid4(),
                    email="member@example.com",
                    full_name="Workspace Member",
                    is_active=True
                )
            )
        ]
        
        mock_db.query.return_value.join.return_value.filter.return_value.all.return_value = mock_members
        
        # Get workspace members
        result = await user_service.get_workspace_members(str(sample_workspace.id))
        
        # Verify results
        assert len(result) == 2
        assert result[0]["email"] == "owner@example.com"
        assert result[0]["role"] == "owner"
        assert result[1]["email"] == "member@example.com"
        assert result[1]["role"] == "member"


class TestWorkspacePermissions:
    """Test workspace permission checking."""
    
    def test_check_workspace_permission_owner(self, user_service, mock_db):
        """Test workspace permission check for owner."""
        # Mock workspace member query
        workspace_member = WorkspaceMember(
            workspace_id=uuid.uuid4(),
            user_id=uuid.uuid4(),
            role="owner",
            permissions=[],
            status="active"
        )
        
        mock_db.query.return_value.filter.return_value.first.return_value = workspace_member
        
        # Check permission
        result = user_service._check_workspace_permission(
            str(workspace_member.user_id),
            str(workspace_member.workspace_id),
            "manage_members"
        )
        
        # Owner should have all permissions
        assert result is True
    
    def test_check_workspace_permission_member_with_permission(self, user_service, mock_db):
        """Test workspace permission check for member with specific permission."""
        # Mock workspace member query
        workspace_member = WorkspaceMember(
            workspace_id=uuid.uuid4(),
            user_id=uuid.uuid4(),
            role="member",
            permissions=["view_data", "create_projects"],
            status="active"
        )
        
        mock_db.query.return_value.filter.return_value.first.return_value = workspace_member
        
        # Check permission that user has
        result = user_service._check_workspace_permission(
            str(workspace_member.user_id),
            str(workspace_member.workspace_id),
            "view_data"
        )
        
        assert result is True
        
        # Check permission that user doesn't have
        result = user_service._check_workspace_permission(
            str(workspace_member.user_id),
            str(workspace_member.workspace_id),
            "manage_members"
        )
        
        assert result is False
    
    def test_check_workspace_permission_no_membership(self, user_service, mock_db):
        """Test workspace permission check for non-member."""
        # Mock no workspace member found
        mock_db.query.return_value.filter.return_value.first.return_value = None
        
        # Check permission
        result = user_service._check_workspace_permission(
            str(uuid.uuid4()),
            str(uuid.uuid4()),
            "view_data"
        )
        
        # Non-member should have no permissions
        assert result is False


if __name__ == "__main__":
    print("🧪 Running Workspace Management Tests")
    print("=" * 50)
    
    # This would normally be run with pytest
    # For demo purposes, we'll just show the test structure
    
    print("✅ Test Categories:")
    print("  • Workspace Creation")
    print("  • Member Management") 
    print("  • Workspace Queries")
    print("  • Permission Checking")
    
    print("\n🔧 Key Features Tested:")
    print("  • Create workspaces with proper validation")
    print("  • Add/remove workspace members")
    print("  • Check workspace limits and permissions")
    print("  • Query user workspaces and members")
    print("  • Enforce role-based access control")
    
    print("\n🚀 Run with: pytest tests/test_workspace_management.py -v")