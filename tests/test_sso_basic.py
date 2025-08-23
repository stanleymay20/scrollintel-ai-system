"""
Basic tests for SSO system without external dependencies
"""
import pytest
import uuid
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from scrollintel.models.sso_models import (
    SSOConfigurationCreate, SSOProviderType, MFAType, AuthResult, UserProfile
)

class TestSSOBasic:
    """Basic SSO functionality tests"""
    
    def test_sso_configuration_create_model(self):
        """Test SSO configuration creation model"""
        config = SSOConfigurationCreate(
            name="Test Provider",
            provider_type=SSOProviderType.OAUTH2,
            config={
                "client_id": "test-client",
                "client_secret": "test-secret"
            },
            user_mapping={
                "email": "email",
                "name": "display_name"
            }
        )
        
        assert config.name == "Test Provider"
        assert config.provider_type == SSOProviderType.OAUTH2
        assert config.config["client_id"] == "test-client"
        assert config.user_mapping["email"] == "email"
    
    def test_auth_result_model(self):
        """Test authentication result model"""
        result = AuthResult(
            success=True,
            user_id="test_user",
            session_token="test_token",
            expires_at=datetime.utcnow() + timedelta(hours=1)
        )
        
        assert result.success is True
        assert result.user_id == "test_user"
        assert result.session_token == "test_token"
        assert result.expires_at > datetime.utcnow()
    
    def test_user_profile_model(self):
        """Test user profile model"""
        profile = UserProfile(
            user_id="test_user",
            email="test@example.com",
            first_name="Test",
            last_name="User",
            display_name="Test User",
            groups=["group1", "group2"],
            roles=["role1", "role2"],
            attributes={"department": "Engineering"}
        )
        
        assert profile.user_id == "test_user"
        assert profile.email == "test@example.com"
        assert profile.display_name == "Test User"
        assert len(profile.groups) == 2
        assert len(profile.roles) == 2
        assert profile.attributes["department"] == "Engineering"
    
    def test_sso_provider_types(self):
        """Test SSO provider type enumeration"""
        assert SSOProviderType.SAML == "saml"
        assert SSOProviderType.OAUTH2 == "oauth2"
        assert SSOProviderType.OIDC == "oidc"
        assert SSOProviderType.LDAP == "ldap"
        assert SSOProviderType.ACTIVE_DIRECTORY == "active_directory"
    
    def test_mfa_types(self):
        """Test MFA type enumeration"""
        assert MFAType.SMS == "sms"
        assert MFAType.EMAIL == "email"
        assert MFAType.TOTP == "totp"
        assert MFAType.PUSH == "push"
        assert MFAType.HARDWARE_TOKEN == "hardware_token"
    
    @patch('scrollintel.core.sso_manager.get_database_session')
    def test_sso_manager_import(self, mock_db):
        """Test SSO manager can be imported and instantiated"""
        from scrollintel.core.sso_manager import SSOManager
        
        # Mock database session
        mock_session = Mock()
        mock_db.return_value.__enter__.return_value = mock_session
        
        manager = SSOManager()
        assert manager is not None
        assert hasattr(manager, 'create_sso_configuration')
        assert hasattr(manager, 'authenticate_user')
        assert hasattr(manager, 'sync_user_attributes')
    
    def test_sso_provider_factory_import(self):
        """Test SSO provider factory can be imported"""
        from scrollintel.core.sso_providers import SSOProviderFactory
        
        assert SSOProviderFactory is not None
        assert hasattr(SSOProviderFactory, 'create_provider')
    
    def test_auth_result_failure(self):
        """Test authentication failure result"""
        result = AuthResult(
            success=False,
            error_message="Invalid credentials"
        )
        
        assert result.success is False
        assert result.user_id is None
        assert result.session_token is None
        assert result.error_message == "Invalid credentials"
    
    def test_auth_result_with_mfa(self):
        """Test authentication result requiring MFA"""
        result = AuthResult(
            success=True,
            user_id="test_user",
            requires_mfa=True,
            mfa_methods=[MFAType.TOTP, MFAType.SMS]
        )
        
        assert result.success is True
        assert result.requires_mfa is True
        assert len(result.mfa_methods) == 2
        assert MFAType.TOTP in result.mfa_methods
        assert MFAType.SMS in result.mfa_methods

class TestSSORoutes:
    """Test SSO API routes"""
    
    def test_sso_routes_import(self):
        """Test SSO routes can be imported"""
        from scrollintel.api.routes.sso_routes import router
        
        assert router is not None
        assert router.prefix == "/api/v1/sso"
        assert "SSO" in router.tags

if __name__ == "__main__":
    pytest.main([__file__])