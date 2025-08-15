"""
Tests for user management and role-based access control.
"""

import pytest
import uuid
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from unittest.mock import Mock, patch

from scrollintel.core.user_management import UserManagementService
from scrollintel.models.database import User
from scrollintel.models.user_management_models import (
    Organization, OrganizationUser, Workspace, WorkspaceMember,
    UserInvitation, UserSession, UserAuditLog, APIKey
)
from scrollintel.core.interfaces import UserRole, SecurityError, ValidationError


class TestUserManagementService:
    """Test cases for UserManagementService."""
    
    @pytest.fixture
    def db_session(self):
        """Mock database session."""
        return Mock(spec=Session)
    
    @pytest.fixture
    def user_service(self, db_session):
        """Create UserManagementService instance."""
        return UserManagementService(db_session)
    
    @pytest.fixture
    def sample_user(self):
        """Create a sample user."""
        return User(
            id=uuid.uuid4(),
            email="test@example.com",
            full_name="Test User",
            role=UserRole.ADMIN,
            is_active=True
        )
    
    @pytest.fixture
    def sample_organization(self):
        """Create a sample organization."""
        return Organization(
            id=uuid.uuid4(),
            name="test-org",
            display_name="Test Organization",
            subscription_plan="free",
            max_users=10,
            max_workspaces=5
        )
    
    # Organization Management Tests
    
    @pytest.mark.asyncio
    async def test_create_organization_success(self, user_service, db_session, sample_user):
        """Test successful organization creation."""
        # Mock database queries
        db_session.query.return_value.filter.return_value.first.return_value = None
        db_session.flush.return_value = None
        db_session.commit.return_value = None
        
        # Create organization
        org = await user_service.create_organization(
            name="test-org",
            display_name="Test Organization",
            creator_user_id=str(sample_user.id),
            description="Test description"
        )
        
        # Verify organization was created
        assert db_session.add.call_count >= 3  # Organization, OrganizationUser, Workspace
        assert db_session.commit.called
    
    @pytest.mark.asyncio
    async def test_create_organization_duplicate_name(self, user_service, db_session):
        """Test organization creation with duplicate name."""
        # Mock existing organization
        existing_org = Organization(name="test-org", display_name="Existing Org")
        db_session.query.return_value.filter.return_value.first.return_value = existing_org
        
        # Attempt to create organization with same name
        with pytest.raises(ValidationError, match="Organization name 'test-org' already exists"):
            await user_service.create_organization(
                name="test-org",
                display_name="Test Organization",
                creator_user_id=str(uuid.uuid4())
            )
    
    @pytest.mark.asyncio
    async def test_get_organization_success(self, user_service, db_session, sample_organization):
        """Test successful organization retrieval."""
        # Mock database query
        db_session.query.return_value.filter.return_value.first.return_value = sample_organization
        
        # Get organization
        org = user_service.get_organization(str(sample_organization.id))
        
        # Verify result
        assert org == sample_organization
        assert db_session.query.called
    
    @pytest.mark.asyncio
    async def test_get_organization_not_found(self, user_service, db_session):
        """Test organization retrieval when not found."""
        # Mock database query
        db_session.query.return_value.filter.return_value.first.return_value = None
        
        # Get organization
        org = user_service.get_organization(str(uuid.uuid4()))
        
        # Verify result
        assert org is None
    
    # User Invitation Tests
    
    @pytest.mark.asyncio
    async def test_invite_user_success(self, user_service, db_session, sample_organization):
        """Test successful user invitation."""
        # Mock permission check
        user_service._check_organization_permission = Mock(return_value=True)
        
        # Mock database queries
        db_session.query.return_value.filter.return_value.first.return_value = None
        db_session.add.return_value = None
        db_session.commit.return_value = None
        
        # Mock audit logging
        user_service._log_audit_event = Mock()
        
        # Mock email validation
        with patch('scrollintel.core.user_management.validate_email') as mock_validate:
            mock_validate.return_value.email = "newuser@test.com"
            
            # Invite user
            invitation = await user_service.invite_user(
                email="newuser@test.com",
                organization_id=str(sample_organization.id),
                invited_by=str(uuid.uuid4()),
                role=UserRole.VIEWER
            )
        
            # Verify invitation was created
            assert db_session.add.called
            assert db_session.commit.called
            assert user_service._log_audit_event.called
    
    @pytest.mark.asyncio
    async def test_invite_user_insufficient_permissions(self, user_service, db_session):
        """Test user invitation with insufficient permissions."""
        # Mock permission check
        user_service._check_organization_permission = Mock(return_value=False)
        
        # Attempt to invite user
        with pytest.raises(SecurityError, match="Insufficient permissions to invite users"):
            await user_service.invite_user(
                email="newuser@example.com",
                organization_id=str(uuid.uuid4()),
                invited_by=str(uuid.uuid4()),
                role=UserRole.VIEWER
            )
    
    @pytest.mark.asyncio
    async def test_invite_user_invalid_email(self, user_service, db_session):
        """Test user invitation with invalid email."""
        # Mock permission check
        user_service._check_organization_permission = Mock(return_value=True)
        
        # Attempt to invite user with invalid email
        with pytest.raises(ValidationError, match="Invalid email address"):
            await user_service.invite_user(
                email="invalid-email",
                organization_id=str(uuid.uuid4()),
                invited_by=str(uuid.uuid4()),
                role=UserRole.VIEWER
            )
    
    @pytest.mark.asyncio
    async def test_accept_invitation_success(self, user_service, db_session, sample_user):
        """Test successful invitation acceptance."""
        # Mock invitation
        invitation = UserInvitation(
            id=uuid.uuid4(),
            email=sample_user.email,
            organization_id=uuid.uuid4(),
            invited_by=uuid.uuid4(),
            role=UserRole.VIEWER,
            token="test-token",
            status="pending",
            expires_at=datetime.utcnow() + timedelta(days=1)
        )
        
        # Mock database queries
        db_session.query.return_value.filter.return_value.first.side_effect = [
            invitation,  # Find invitation
            sample_user,  # Find user
            None  # Check existing org user
        ]
        db_session.add.return_value = None
        db_session.commit.return_value = None
        
        # Mock audit logging
        user_service._log_audit_event = Mock()
        
        # Accept invitation
        org_user = await user_service.accept_invitation("test-token", str(sample_user.id))
        
        # Verify organization user was created
        assert db_session.add.called
        assert db_session.commit.called
        assert invitation.status == "accepted"
        assert user_service._log_audit_event.called
    
    @pytest.mark.asyncio
    async def test_accept_invitation_invalid_token(self, user_service, db_session):
        """Test invitation acceptance with invalid token."""
        # Mock database query
        db_session.query.return_value.filter.return_value.first.return_value = None
        
        # Attempt to accept invitation with invalid token
        with pytest.raises(ValidationError, match="Invalid or expired invitation"):
            await user_service.accept_invitation("invalid-token", str(uuid.uuid4()))
    
    # Role Management Tests
    
    @pytest.mark.asyncio
    async def test_update_user_role_success(self, user_service, db_session):
        """Test successful user role update."""
        # Mock permission check
        user_service._check_organization_permission = Mock(return_value=True)
        
        # Mock organization user
        org_user = OrganizationUser(
            id=uuid.uuid4(),
            organization_id=uuid.uuid4(),
            user_id=uuid.uuid4(),
            role=UserRole.VIEWER,
            status="active"
        )
        
        # Mock database queries
        db_session.query.return_value.filter.return_value.first.return_value = org_user
        db_session.commit.return_value = None
        
        # Mock audit logging
        user_service._log_audit_event = Mock()
        
        # Update user role
        updated_org_user = await user_service.update_user_role(
            organization_id=str(org_user.organization_id),
            target_user_id=str(org_user.user_id),
            new_role=UserRole.ANALYST,
            updated_by=str(uuid.uuid4())
        )
        
        # Verify role was updated
        assert updated_org_user.role == UserRole.ANALYST
        assert db_session.commit.called
        assert user_service._log_audit_event.called
    
    @pytest.mark.asyncio
    async def test_update_user_role_insufficient_permissions(self, user_service, db_session):
        """Test user role update with insufficient permissions."""
        # Mock permission check
        user_service._check_organization_permission = Mock(return_value=False)
        
        # Attempt to update user role
        with pytest.raises(SecurityError, match="Insufficient permissions to update user roles"):
            await user_service.update_user_role(
                organization_id=str(uuid.uuid4()),
                target_user_id=str(uuid.uuid4()),
                new_role=UserRole.ANALYST,
                updated_by=str(uuid.uuid4())
            )
    
    # Workspace Management Tests
    
    @pytest.mark.asyncio
    async def test_create_workspace_success(self, user_service, db_session, sample_organization):
        """Test successful workspace creation."""
        # Mock permission check
        user_service._check_organization_permission = Mock(return_value=True)
        
        # Mock get organization
        user_service.get_organization = Mock(return_value=sample_organization)
        
        # Mock database queries
        db_session.query.return_value.filter.return_value.count.return_value = 0
        db_session.add.return_value = None
        db_session.flush.return_value = None
        db_session.commit.return_value = None
        
        # Mock audit logging
        user_service._log_audit_event = Mock()
        
        # Create workspace
        workspace = await user_service.create_workspace(
            name="Test Workspace",
            organization_id=str(sample_organization.id),
            owner_id=str(uuid.uuid4()),
            description="Test description"
        )
        
        # Verify workspace was created
        assert db_session.add.call_count >= 2  # Workspace and WorkspaceMember
        assert db_session.commit.called
        assert user_service._log_audit_event.called
    
    @pytest.mark.asyncio
    async def test_create_workspace_limit_exceeded(self, user_service, db_session, sample_organization):
        """Test workspace creation when limit is exceeded."""
        # Mock permission check
        user_service._check_organization_permission = Mock(return_value=True)
        
        # Mock get organization
        user_service.get_organization = Mock(return_value=sample_organization)
        
        # Mock workspace count at limit
        db_session.query.return_value.filter.return_value.count.return_value = sample_organization.max_workspaces
        
        # Attempt to create workspace
        with pytest.raises(ValidationError, match="Maximum workspace limit"):
            await user_service.create_workspace(
                name="Test Workspace",
                organization_id=str(sample_organization.id),
                owner_id=str(uuid.uuid4())
            )
    
    # Session Management Tests
    
    @pytest.mark.asyncio
    async def test_create_session_success(self, user_service, db_session, sample_user):
        """Test successful session creation."""
        # Mock database operations
        db_session.add.return_value = None
        db_session.query.return_value.filter.return_value.first.return_value = sample_user
        db_session.commit.return_value = None
        
        # Create session
        session_token, refresh_token = await user_service.create_session(
            user_id=str(sample_user.id),
            ip_address="127.0.0.1",
            user_agent="Test Agent"
        )
        
        # Verify session was created
        assert session_token is not None
        assert refresh_token is not None
        assert db_session.add.called
        assert db_session.commit.called
    
    @pytest.mark.asyncio
    async def test_validate_session_success(self, user_service, db_session, sample_user):
        """Test successful session validation."""
        # Mock session
        session = UserSession(
            id=uuid.uuid4(),
            user_id=sample_user.id,
            session_token="test-token",
            is_active=True,
            expires_at=datetime.utcnow() + timedelta(hours=1)
        )
        
        # Mock database queries
        db_session.query.return_value.filter.return_value.first.side_effect = [
            session,  # Find session
            sample_user  # Find user
        ]
        db_session.commit.return_value = None
        
        # Validate session
        user = await user_service.validate_session("test-token")
        
        # Verify user was returned
        assert user == sample_user
        assert db_session.commit.called  # Last activity updated
    
    @pytest.mark.asyncio
    async def test_validate_session_expired(self, user_service, db_session):
        """Test session validation with expired session."""
        # Mock database query
        db_session.query.return_value.filter.return_value.first.return_value = None
        
        # Validate expired session
        user = await user_service.validate_session("expired-token")
        
        # Verify no user was returned
        assert user is None
    
    # API Key Management Tests
    
    @pytest.mark.asyncio
    async def test_create_api_key_success(self, user_service, db_session):
        """Test successful API key creation."""
        # Mock permission check
        user_service._check_organization_permission = Mock(return_value=True)
        
        # Mock database operations
        db_session.add.return_value = None
        db_session.commit.return_value = None
        
        # Mock audit logging
        user_service._log_audit_event = Mock()
        
        # Create API key
        api_key_record, api_key = await user_service.create_api_key(
            name="Test API Key",
            user_id=str(uuid.uuid4()),
            organization_id=str(uuid.uuid4()),
            description="Test description"
        )
        
        # Verify API key was created
        assert api_key is not None
        assert len(api_key) > 20  # Should be a long random string
        assert db_session.add.called
        assert db_session.commit.called
        assert user_service._log_audit_event.called
    
    @pytest.mark.asyncio
    async def test_validate_api_key_success(self, user_service, db_session):
        """Test successful API key validation."""
        # Mock API key record
        api_key_record = APIKey(
            id=uuid.uuid4(),
            name="Test Key",
            key_hash="test-hash",
            is_active=True,
            expires_at=None,
            usage_count=0
        )
        
        # Mock database query
        db_session.query.return_value.filter.return_value.first.return_value = api_key_record
        db_session.commit.return_value = None
        
        # Mock hash calculation
        with patch('hashlib.sha256') as mock_hash:
            mock_hash.return_value.hexdigest.return_value = "test-hash"
            
            # Validate API key
            result = await user_service.validate_api_key("test-api-key")
        
        # Verify API key was validated
        assert result == api_key_record
        assert api_key_record.usage_count == 1
        assert db_session.commit.called
    
    @pytest.mark.asyncio
    async def test_validate_api_key_invalid(self, user_service, db_session):
        """Test API key validation with invalid key."""
        # Mock database query
        db_session.query.return_value.filter.return_value.first.return_value = None
        
        # Validate invalid API key
        result = await user_service.validate_api_key("invalid-key")
        
        # Verify no API key was returned
        assert result is None
    
    # Permission Checking Tests
    
    @pytest.mark.asyncio
    async def test_check_organization_permission_admin(self, user_service, db_session):
        """Test organization permission check for admin user."""
        # Mock organization user with admin role
        org_user = OrganizationUser(
            user_id=uuid.uuid4(),
            organization_id=uuid.uuid4(),
            role=UserRole.ADMIN,
            status="active"
        )
        
        # Mock database query
        db_session.query.return_value.filter.return_value.first.return_value = org_user
        
        # Check permission
        has_permission = user_service._check_organization_permission(
            str(org_user.user_id),
            str(org_user.organization_id),
            "any_permission"
        )
        
        # Verify admin has permission
        assert has_permission is True
    
    @pytest.mark.asyncio
    async def test_check_organization_permission_specific(self, user_service, db_session):
        """Test organization permission check for specific permission."""
        # Mock organization user with specific permissions
        org_user = OrganizationUser(
            user_id=uuid.uuid4(),
            organization_id=uuid.uuid4(),
            role=UserRole.VIEWER,
            permissions=["view_data", "create_projects"],
            status="active"
        )
        
        # Mock database query
        db_session.query.return_value.filter.return_value.first.return_value = org_user
        
        # Check allowed permission
        has_permission = user_service._check_organization_permission(
            str(org_user.user_id),
            str(org_user.organization_id),
            "view_data"
        )
        assert has_permission is True
        
        # Check denied permission
        has_permission = user_service._check_organization_permission(
            str(org_user.user_id),
            str(org_user.organization_id),
            "manage_users"
        )
        assert has_permission is False
    
    @pytest.mark.asyncio
    async def test_check_organization_permission_not_member(self, user_service, db_session):
        """Test organization permission check for non-member."""
        # Mock database query
        db_session.query.return_value.filter.return_value.first.return_value = None
        
        # Check permission
        has_permission = user_service._check_organization_permission(
            str(uuid.uuid4()),
            str(uuid.uuid4()),
            "any_permission"
        )
        
        # Verify non-member has no permission
        assert has_permission is False
    
    # Utility Method Tests
    
    @pytest.mark.asyncio
    async def test_get_user_organizations(self, user_service, db_session, sample_organization):
        """Test getting user organizations."""
        # Mock database query
        db_session.query.return_value.join.return_value.filter.return_value.all.return_value = [sample_organization]
        
        # Get user organizations
        organizations = await user_service.get_user_organizations(str(uuid.uuid4()))
        
        # Verify organizations were returned
        assert organizations == [sample_organization]
        assert db_session.query.called
    
    @pytest.mark.asyncio
    async def test_get_organization_users(self, user_service, db_session, sample_user):
        """Test getting organization users."""
        # Mock organization user and user data
        org_user = OrganizationUser(
            user_id=sample_user.id,
            organization_id=uuid.uuid4(),
            role=UserRole.ADMIN,
            permissions=["*"],
            joined_at=datetime.utcnow(),
            status="active"
        )
        
        # Mock database query
        db_session.query.return_value.join.return_value.filter.return_value.all.return_value = [
            (org_user, sample_user)
        ]
        
        # Get organization users
        users = await user_service.get_organization_users(str(org_user.organization_id))
        
        # Verify users were returned
        assert len(users) == 1
        assert users[0]["user_id"] == str(sample_user.id)
        assert users[0]["email"] == sample_user.email
        assert users[0]["role"] == UserRole.ADMIN.value


if __name__ == "__main__":
    pytest.main([__file__])