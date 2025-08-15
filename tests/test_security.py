"""
Unit tests for EXOUSIA security system components.
Tests authentication, authorization, audit logging, and session management.
"""

import pytest
import asyncio
import json
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, AsyncMock, patch
from uuid import uuid4, UUID

from scrollintel.core.interfaces import UserRole, SecurityContext, SecurityError
from scrollintel.security.auth import JWTAuthenticator, PasswordManager
from scrollintel.security.permissions import (
    Permission, ResourceType, Action, RolePermissionManager, PermissionChecker
)
from scrollintel.security.audit import AuditLogger, AuditAction
from scrollintel.security.session import SessionManager, SessionData
from scrollintel.models.database import User


class TestJWTAuthenticator:
    """Test JWT authentication functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.authenticator = JWTAuthenticator()
        self.test_user = Mock(spec=User)
        self.test_user.id = uuid4()
        self.test_user.email = "test@example.com"
        self.test_user.role = UserRole.ANALYST
        self.test_user.permissions = ["dataset:read", "model:create"]
    
    def test_hash_password(self):
        """Test password hashing."""
        password = "TestPassword123!"
        hashed = self.authenticator.hash_password(password)
        
        assert hashed != password
        assert len(hashed) > 0
        assert self.authenticator.verify_password(password, hashed)
    
    def test_verify_password_success(self):
        """Test successful password verification."""
        password = "TestPassword123!"
        hashed = self.authenticator.hash_password(password)
        
        assert self.authenticator.verify_password(password, hashed)
    
    def test_verify_password_failure(self):
        """Test failed password verification."""
        password = "TestPassword123!"
        wrong_password = "WrongPassword123!"
        hashed = self.authenticator.hash_password(password)
        
        assert not self.authenticator.verify_password(wrong_password, hashed)
    
    def test_create_access_token(self):
        """Test access token creation."""
        token = self.authenticator.create_access_token(self.test_user)
        
        assert isinstance(token, str)
        assert len(token) > 0
        
        # Verify token contents
        payload = self.authenticator.verify_token(token)
        assert payload["sub"] == str(self.test_user.id)
        assert payload["email"] == self.test_user.email
        assert payload["role"] == self.test_user.role.value
        assert payload["permissions"] == self.test_user.permissions
        assert payload["type"] == "access"
    
    def test_create_refresh_token(self):
        """Test refresh token creation."""
        token = self.authenticator.create_refresh_token(self.test_user)
        
        assert isinstance(token, str)
        assert len(token) > 0
        
        # Verify token contents
        payload = self.authenticator.verify_token(token)
        assert payload["sub"] == str(self.test_user.id)
        assert payload["type"] == "refresh"
    
    def test_verify_token_success(self):
        """Test successful token verification."""
        token = self.authenticator.create_access_token(self.test_user)
        payload = self.authenticator.verify_token(token)
        
        assert payload is not None
        assert payload["sub"] == str(self.test_user.id)
    
    def test_verify_token_invalid(self):
        """Test invalid token verification."""
        with pytest.raises(SecurityError, match="Invalid token"):
            self.authenticator.verify_token("invalid.token.here")
    
    def test_get_user_from_token(self):
        """Test extracting user from token."""
        token = self.authenticator.create_access_token(self.test_user)
        context = self.authenticator.get_user_from_token(token)
        
        assert context is not None
        assert context.user_id == str(self.test_user.id)
        assert context.role == self.test_user.role
        assert context.permissions == self.test_user.permissions
    
    def test_get_user_from_refresh_token_fails(self):
        """Test that refresh tokens can't be used for user extraction."""
        refresh_token = self.authenticator.create_refresh_token(self.test_user)
        
        with pytest.raises(SecurityError, match="Invalid token type"):
            self.authenticator.get_user_from_token(refresh_token)


class TestPasswordManager:
    """Test password management utilities."""
    
    def test_validate_password_strength_valid(self):
        """Test valid password validation."""
        valid_passwords = [
            "TestPass123!",
            "MySecure@Password1",
            "Complex#Pass99"
        ]
        
        for password in valid_passwords:
            assert PasswordManager.validate_password_strength(password)
    
    def test_validate_password_strength_invalid(self):
        """Test invalid password validation."""
        invalid_passwords = [
            "short",  # Too short
            "nouppercase123!",  # No uppercase
            "NOLOWERCASE123!",  # No lowercase
            "NoDigits!",  # No digits
            "NoSpecialChars123"  # No special characters
        ]
        
        for password in invalid_passwords:
            assert not PasswordManager.validate_password_strength(password)
    
    def test_generate_password_requirements(self):
        """Test password requirements generation."""
        requirements = PasswordManager.generate_password_requirements()
        
        assert isinstance(requirements, dict)
        assert "min_length" in requirements
        assert "uppercase" in requirements
        assert "lowercase" in requirements
        assert "digit" in requirements
        assert "special" in requirements


class TestRolePermissionManager:
    """Test role-based permission management."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.manager = RolePermissionManager()
    
    def test_get_role_permissions(self):
        """Test getting permissions for a role."""
        admin_permissions = self.manager.get_role_permissions(UserRole.ADMIN)
        viewer_permissions = self.manager.get_role_permissions(UserRole.VIEWER)
        
        assert len(admin_permissions) > len(viewer_permissions)
        assert Permission.USER_CREATE in admin_permissions
        assert Permission.USER_CREATE not in viewer_permissions
    
    def test_has_permission(self):
        """Test permission checking."""
        assert self.manager.has_permission(UserRole.ADMIN, Permission.USER_CREATE)
        assert not self.manager.has_permission(UserRole.VIEWER, Permission.USER_CREATE)
        assert self.manager.has_permission(UserRole.VIEWER, Permission.USER_READ)
    
    def test_can_perform_action(self):
        """Test action permission checking."""
        assert self.manager.can_perform_action(UserRole.ADMIN, ResourceType.USER, Action.CREATE)
        assert not self.manager.can_perform_action(UserRole.VIEWER, ResourceType.USER, Action.CREATE)
        assert self.manager.can_perform_action(UserRole.ANALYST, ResourceType.DATASET, Action.CREATE)
    
    def test_add_permission_to_role(self):
        """Test adding permission to role."""
        original_permissions = self.manager.get_role_permissions(UserRole.VIEWER).copy()
        
        self.manager.add_permission_to_role(UserRole.VIEWER, Permission.USER_CREATE)
        
        new_permissions = self.manager.get_role_permissions(UserRole.VIEWER)
        assert Permission.USER_CREATE in new_permissions
        assert len(new_permissions) == len(original_permissions) + 1
    
    def test_remove_permission_from_role(self):
        """Test removing permission from role."""
        self.manager.remove_permission_from_role(UserRole.ADMIN, Permission.USER_CREATE)
        
        admin_permissions = self.manager.get_role_permissions(UserRole.ADMIN)
        assert Permission.USER_CREATE not in admin_permissions


class TestPermissionChecker:
    """Test permission checking logic."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.checker = PermissionChecker()
        self.admin_context = SecurityContext(
            user_id=str(uuid4()),
            role=UserRole.ADMIN,
            permissions=[],
            session_id="test-session",
            ip_address="127.0.0.1"
        )
        self.viewer_context = SecurityContext(
            user_id=str(uuid4()),
            role=UserRole.VIEWER,
            permissions=["custom:permission"],
            session_id="test-session",
            ip_address="127.0.0.1"
        )
    
    def test_check_permission_role_based(self):
        """Test role-based permission checking."""
        assert self.checker.check_permission(self.admin_context, Permission.USER_CREATE)
        assert not self.checker.check_permission(self.viewer_context, Permission.USER_CREATE)
    
    def test_check_permission_user_specific(self):
        """Test user-specific permission checking."""
        # Test with a permission that exists in user's custom permissions
        # Since the viewer context has "custom:permission" in permissions list
        # We need to check if the permission checker handles string permissions correctly
        
        # Create a context with a permission that matches an existing enum value
        context_with_custom = SecurityContext(
            user_id=str(uuid4()),
            role=UserRole.VIEWER,
            permissions=["user:create"],  # This matches Permission.USER_CREATE
            session_id="test-session",
            ip_address="127.0.0.1"
        )
        
        # Should pass because user has explicit permission
        assert self.checker.check_permission(context_with_custom, Permission.USER_CREATE)
    
    def test_check_resource_access(self):
        """Test resource access checking."""
        assert self.checker.check_resource_access(
            self.admin_context, ResourceType.USER, Action.CREATE
        )
        assert not self.checker.check_resource_access(
            self.viewer_context, ResourceType.USER, Action.CREATE
        )
    
    def test_require_permission_success(self):
        """Test successful permission requirement."""
        # Should not raise exception
        self.checker.require_permission(self.admin_context, Permission.USER_CREATE)
    
    def test_require_permission_failure(self):
        """Test failed permission requirement."""
        with pytest.raises(SecurityError, match="Permission denied"):
            self.checker.require_permission(self.viewer_context, Permission.USER_CREATE)
    
    def test_require_resource_access_success(self):
        """Test successful resource access requirement."""
        # Should not raise exception
        self.checker.require_resource_access(
            self.admin_context, ResourceType.USER, Action.CREATE
        )
    
    def test_require_resource_access_failure(self):
        """Test failed resource access requirement."""
        with pytest.raises(SecurityError, match="Access denied"):
            self.checker.require_resource_access(
                self.viewer_context, ResourceType.USER, Action.CREATE
            )
    
    def test_get_user_permissions(self):
        """Test getting all user permissions."""
        permissions = self.checker.get_user_permissions(self.admin_context)
        assert isinstance(permissions, list)
        assert len(permissions) > 0
        assert "user:create" in permissions
    
    def test_can_access_admin_panel(self):
        """Test admin panel access checking."""
        assert self.checker.can_access_admin_panel(self.admin_context)
        assert not self.checker.can_access_admin_panel(self.viewer_context)
    
    def test_can_manage_users(self):
        """Test user management permission checking."""
        assert self.checker.can_manage_users(self.admin_context)
        assert not self.checker.can_manage_users(self.viewer_context)
    
    def test_can_execute_agents(self):
        """Test agent execution permission checking."""
        analyst_context = SecurityContext(
            user_id=str(uuid4()),
            role=UserRole.ANALYST,
            permissions=[],
            session_id="test-session",
            ip_address="127.0.0.1"
        )
        
        assert self.checker.can_execute_agents(analyst_context)
        assert not self.checker.can_execute_agents(self.viewer_context)


class TestAuditLogger:
    """Test audit logging functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.logger = AuditLogger()
        self.test_context = SecurityContext(
            user_id=str(uuid4()),
            role=UserRole.ANALYST,
            permissions=[],
            session_id="test-session",
            ip_address="127.0.0.1"
        )
    
    @pytest.mark.asyncio
    async def test_log_basic(self):
        """Test basic audit logging."""
        with patch.object(self.logger, '_save_audit_log', new_callable=AsyncMock) as mock_save:
            await self.logger.log(
                action=AuditAction.USER_CREATE,
                resource_type="user",
                resource_id="test-user-id",
                user_id=UUID(self.test_context.user_id),
                ip_address=self.test_context.ip_address
            )
            
            # Wait for background processing
            await asyncio.sleep(0.1)
            
            # Verify log was queued
            assert not self.logger._log_queue.empty()
    
    @pytest.mark.asyncio
    async def test_log_with_context(self):
        """Test audit logging with security context."""
        with patch.object(self.logger, '_save_audit_log', new_callable=AsyncMock) as mock_save:
            await self.logger.log_with_context(
                context=self.test_context,
                action=AuditAction.DATASET_CREATE,
                resource_type="dataset",
                resource_id="test-dataset-id",
                details={"name": "Test Dataset"}
            )
            
            # Wait for background processing
            await asyncio.sleep(0.1)
            
            # Verify log was queued
            assert not self.logger._log_queue.empty()
    
    @pytest.mark.asyncio
    async def test_log_authentication(self):
        """Test authentication event logging."""
        with patch.object(self.logger, '_save_audit_log', new_callable=AsyncMock) as mock_save:
            await self.logger.log_authentication(
                action=AuditAction.LOGIN_SUCCESS,
                email="test@example.com",
                ip_address="127.0.0.1",
                user_agent="Test Browser",
                user_id=UUID(self.test_context.user_id)
            )
            
            # Wait for background processing
            await asyncio.sleep(0.1)
            
            # Verify log was queued
            assert not self.logger._log_queue.empty()
    
    @pytest.mark.asyncio
    async def test_log_permission_denied(self):
        """Test permission denied logging."""
        with patch.object(self.logger, '_save_audit_log', new_callable=AsyncMock) as mock_save:
            await self.logger.log_permission_denied(
                context=self.test_context,
                action="user:create",
                resource_type="user",
                resource_id="test-user-id"
            )
            
            # Wait for background processing
            await asyncio.sleep(0.1)
            
            # Verify log was queued
            assert not self.logger._log_queue.empty()
    
    @pytest.mark.asyncio
    async def test_log_suspicious_activity(self):
        """Test suspicious activity logging."""
        with patch.object(self.logger, '_save_audit_log', new_callable=AsyncMock) as mock_save:
            await self.logger.log_suspicious_activity(
                context=self.test_context,
                activity_type="multiple_failed_logins",
                details={"attempts": 5, "timeframe": "5 minutes"}
            )
            
            # Wait for background processing
            await asyncio.sleep(0.1)
            
            # Verify log was queued
            assert not self.logger._log_queue.empty()


class TestSessionData:
    """Test session data structure."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.now = datetime.now(timezone.utc)
        self.session_data = SessionData(
            user_id=str(uuid4()),
            role=UserRole.ANALYST,
            permissions=["dataset:read", "model:create"],
            ip_address="127.0.0.1",
            user_agent="Test Browser",
            created_at=self.now,
            last_activity=self.now,
            expires_at=self.now + timedelta(hours=1)
        )
    
    def test_to_dict(self):
        """Test converting session data to dictionary."""
        data_dict = self.session_data.to_dict()
        
        assert isinstance(data_dict, dict)
        assert data_dict["user_id"] == self.session_data.user_id
        assert data_dict["role"] == self.session_data.role.value
        assert data_dict["permissions"] == self.session_data.permissions
        assert data_dict["ip_address"] == self.session_data.ip_address
    
    def test_from_dict(self):
        """Test creating session data from dictionary."""
        data_dict = self.session_data.to_dict()
        restored_session = SessionData.from_dict(data_dict)
        
        assert restored_session.user_id == self.session_data.user_id
        assert restored_session.role == self.session_data.role
        assert restored_session.permissions == self.session_data.permissions
        assert restored_session.ip_address == self.session_data.ip_address
    
    def test_is_expired_false(self):
        """Test session not expired."""
        assert not self.session_data.is_expired()
    
    def test_is_expired_true(self):
        """Test session expired."""
        expired_session = SessionData(
            user_id=str(uuid4()),
            role=UserRole.ANALYST,
            permissions=[],
            ip_address="127.0.0.1",
            user_agent="Test Browser",
            created_at=self.now - timedelta(hours=2),
            last_activity=self.now - timedelta(hours=2),
            expires_at=self.now - timedelta(hours=1)
        )
        
        assert expired_session.is_expired()
    
    def test_update_activity(self):
        """Test updating last activity."""
        original_activity = self.session_data.last_activity
        
        # Wait a bit to ensure time difference
        import time
        time.sleep(0.01)
        
        self.session_data.update_activity()
        
        assert self.session_data.last_activity > original_activity


class TestSessionManager:
    """Test session management functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.manager = SessionManager()
        self.test_user_id = str(uuid4())
    
    @pytest.mark.asyncio
    async def test_initialize_success(self):
        """Test successful session manager initialization."""
        with patch('redis.asyncio.Redis') as mock_redis:
            mock_client = AsyncMock()
            mock_redis.return_value = mock_client
            mock_client.ping = AsyncMock()
            
            await self.manager.initialize()
            
            assert self.manager.redis_client is not None
            mock_client.ping.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_initialize_failure(self):
        """Test failed session manager initialization."""
        with patch('redis.asyncio.Redis') as mock_redis:
            mock_client = AsyncMock()
            mock_redis.return_value = mock_client
            mock_client.ping = AsyncMock(side_effect=Exception("Connection failed"))
            
            with pytest.raises(SecurityError, match="Failed to initialize session manager"):
                await self.manager.initialize()
    
    @pytest.mark.asyncio
    async def test_create_session(self):
        """Test session creation."""
        with patch('redis.asyncio.Redis') as mock_redis:
            mock_client = AsyncMock()
            mock_redis.return_value = mock_client
            self.manager.redis_client = mock_client
            
            session_id = await self.manager.create_session(
                user_id=self.test_user_id,
                role=UserRole.ANALYST,
                permissions=["dataset:read"],
                ip_address="127.0.0.1",
                user_agent="Test Browser"
            )
            
            assert isinstance(session_id, str)
            assert len(session_id) > 0
            mock_client.setex.assert_called()
            mock_client.sadd.assert_called()
    
    @pytest.mark.asyncio
    async def test_get_session_success(self):
        """Test successful session retrieval."""
        with patch('redis.asyncio.Redis') as mock_redis:
            mock_client = AsyncMock()
            mock_redis.return_value = mock_client
            self.manager.redis_client = mock_client
            
            # Mock session data
            session_data = SessionData(
                user_id=self.test_user_id,
                role=UserRole.ANALYST,
                permissions=["dataset:read"],
                ip_address="127.0.0.1",
                user_agent="Test Browser",
                created_at=datetime.now(timezone.utc),
                last_activity=datetime.now(timezone.utc),
                expires_at=datetime.now(timezone.utc) + timedelta(hours=1)
            )
            
            mock_client.get.return_value = json.dumps(session_data.to_dict())
            
            result = await self.manager.get_session("test-session-id")
            
            assert result is not None
            assert result.user_id == self.test_user_id
            assert result.role == UserRole.ANALYST
    
    @pytest.mark.asyncio
    async def test_get_session_not_found(self):
        """Test session not found."""
        with patch('redis.asyncio.Redis') as mock_redis:
            mock_client = AsyncMock()
            mock_redis.return_value = mock_client
            self.manager.redis_client = mock_client
            
            mock_client.get.return_value = None
            
            result = await self.manager.get_session("nonexistent-session")
            
            assert result is None
    
    @pytest.mark.asyncio
    async def test_validate_session_success(self):
        """Test successful session validation."""
        with patch('redis.asyncio.Redis') as mock_redis:
            mock_client = AsyncMock()
            mock_redis.return_value = mock_client
            self.manager.redis_client = mock_client
            
            # Mock session data
            session_data = SessionData(
                user_id=self.test_user_id,
                role=UserRole.ANALYST,
                permissions=["dataset:read"],
                ip_address="127.0.0.1",
                user_agent="Test Browser",
                created_at=datetime.now(timezone.utc),
                last_activity=datetime.now(timezone.utc),
                expires_at=datetime.now(timezone.utc) + timedelta(hours=1)
            )
            
            mock_client.get.return_value = json.dumps(session_data.to_dict())
            
            context = await self.manager.validate_session("test-session-id", "127.0.0.1")
            
            assert context is not None
            assert context.user_id == self.test_user_id
            assert context.role == UserRole.ANALYST
            assert context.session_id == "test-session-id"
    
    @pytest.mark.asyncio
    async def test_validate_session_not_found(self):
        """Test session validation when session not found."""
        with patch('redis.asyncio.Redis') as mock_redis:
            mock_client = AsyncMock()
            mock_redis.return_value = mock_client
            self.manager.redis_client = mock_client
            
            mock_client.get.return_value = None
            
            context = await self.manager.validate_session("nonexistent-session", "127.0.0.1")
            
            assert context is None
    
    @pytest.mark.asyncio
    async def test_delete_session(self):
        """Test session deletion."""
        with patch('redis.asyncio.Redis') as mock_redis:
            mock_client = AsyncMock()
            mock_redis.return_value = mock_client
            self.manager.redis_client = mock_client
            
            # Mock session data for logging
            session_data = SessionData(
                user_id=self.test_user_id,
                role=UserRole.ANALYST,
                permissions=["dataset:read"],
                ip_address="127.0.0.1",
                user_agent="Test Browser",
                created_at=datetime.now(timezone.utc),
                last_activity=datetime.now(timezone.utc),
                expires_at=datetime.now(timezone.utc) + timedelta(hours=1)
            )
            
            mock_client.get.return_value = json.dumps(session_data.to_dict())
            
            await self.manager.delete_session("test-session-id")
            
            mock_client.delete.assert_called()
            mock_client.srem.assert_called()


if __name__ == "__main__":
    pytest.main([__file__])