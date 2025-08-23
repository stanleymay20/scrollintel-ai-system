"""
Integration tests for SSO and Authentication System
"""
import pytest
import uuid
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from scrollintel.core.sso_manager import SSOManager
from scrollintel.core.sso_providers import SAMLProvider, OAuth2Provider, LDAPProvider, MFAManager
from scrollintel.models.sso_models import (
    SSOConfigurationCreate, SSOProviderType, MFAType, AuthResult, UserProfile
)

class TestSSOIntegration:
    """Test SSO integration functionality"""
    
    def setup_method(self):
        """Setup test environment"""
        self.sso_manager = SSOManager()
        self.test_user_id = "test_user_123"
        self.test_provider_id = str(uuid.uuid4())
    
    @patch('scrollintel.core.sso_manager.get_database_session')
    def test_create_sso_configuration(self, mock_db_session):
        """Test creating SSO configuration"""
        # Mock database session
        mock_session = Mock()
        mock_db_session.return_value.__enter__.return_value = mock_session
        
        # Create test configuration
        config_data = SSOConfigurationCreate(
            name="Test Azure AD",
            provider_type=SSOProviderType.OAUTH2,
            config={
                "client_id": "test-client-id",
                "client_secret": "test-secret",
                "auth_url": "https://login.microsoftonline.com/common/oauth2/v2.0/authorize",
                "token_url": "https://login.microsoftonline.com/common/oauth2/v2.0/token",
                "userinfo_url": "https://graph.microsoft.com/v1.0/me"
            },
            user_mapping={
                "email": "mail",
                "first_name": "givenName",
                "last_name": "surname"
            }
        )
        
        # Create configuration
        config_id = self.sso_manager.create_sso_configuration(config_data, "admin_user")
        
        # Verify configuration was created
        assert config_id is not None
        assert len(config_id) > 0
        mock_session.add.assert_called_once()
        mock_session.commit.assert_called_once()
    
    @patch('scrollintel.core.sso_manager.get_database_session')
    def test_oauth2_authentication_success(self, mock_db_session):
        """Test successful OAuth2 authentication"""
        # Mock database session
        mock_session = Mock()
        mock_db_session.return_value.__enter__.return_value = mock_session
        
        # Mock SSO configuration
        mock_config = Mock()
        mock_config.id = self.test_provider_id
        mock_config.provider_type = "oauth2"
        mock_config.is_active = True
        mock_config.config = {
            "client_id": "test-client-id",
            "client_secret": "test-secret",
            "token_url": "https://example.com/token",
            "userinfo_url": "https://example.com/userinfo"
        }
        mock_config.user_mapping = {"email": "email"}
        
        # Mock provider authentication
        with patch.object(self.sso_manager, 'get_sso_configuration', return_value=mock_config):
            with patch('requests.post') as mock_post, patch('requests.get') as mock_get:
                # Mock token exchange
                mock_post.return_value.status_code = 200
                mock_post.return_value.json.return_value = {
                    "access_token": "test-access-token",
                    "refresh_token": "test-refresh-token",
                    "expires_in": 3600
                }
                
                # Mock user info
                mock_get.return_value.status_code = 200
                mock_get.return_value.json.return_value = {
                    "sub": self.test_user_id,
                    "email": "test@example.com",
                    "name": "Test User"
                }
                
                # Authenticate user
                credentials = {
                    "code": "test-auth-code",
                    "ip_address": "127.0.0.1",
                    "user_agent": "Test Browser"
                }
                
                result = self.sso_manager.authenticate_user(self.test_provider_id, credentials)
                
                # Verify authentication success
                assert result.success is True
                assert result.user_id == self.test_user_id
                assert result.session_token == "test-access-token"
                assert result.refresh_token == "test-refresh-token"
    
    @patch('scrollintel.core.sso_manager.get_database_session')
    def test_saml_authentication_success(self, mock_db_session):
        """Test successful SAML authentication"""
        # Mock database session
        mock_session = Mock()
        mock_db_session.return_value.__enter__.return_value = mock_session
        
        # Mock SSO configuration
        mock_config = Mock()
        mock_config.id = self.test_provider_id
        mock_config.provider_type = "saml"
        mock_config.is_active = True
        mock_config.config = {
            "idp_url": "https://example.com/saml/sso",
            "sp_entity_id": "scrollintel",
            "certificate": "test-cert",
            "private_key": "test-key"
        }
        
        with patch.object(self.sso_manager, 'get_sso_configuration', return_value=mock_config):
            # Mock SAML response validation
            with patch('xml.etree.ElementTree.fromstring') as mock_xml:
                mock_element = Mock()
                mock_element.find.return_value.text = self.test_user_id
                mock_xml.return_value = mock_element
                
                with patch('jwt.encode', return_value="test-jwt-token"):
                    credentials = {
                        "saml_response": "base64-encoded-saml-response"
                    }
                    
                    result = self.sso_manager.authenticate_user(self.test_provider_id, credentials)
                    
                    # Verify authentication success
                    assert result.success is True
                    assert result.user_id == self.test_user_id
                    assert result.session_token == "test-jwt-token"
    
    @patch('scrollintel.core.sso_manager.get_database_session')
    def test_ldap_authentication_success(self, mock_db_session):
        """Test successful LDAP authentication"""
        # Mock database session
        mock_session = Mock()
        mock_db_session.return_value.__enter__.return_value = mock_session
        
        # Mock SSO configuration
        mock_config = Mock()
        mock_config.id = self.test_provider_id
        mock_config.provider_type = "ldap"
        mock_config.is_active = True
        mock_config.config = {
            "server_url": "ldap://example.com",
            "bind_dn": "cn=admin,dc=example,dc=com",
            "bind_password": "admin-password",
            "user_search_base": "ou=users,dc=example,dc=com",
            "user_search_filter": "(uid={username})"
        }
        
        with patch.object(self.sso_manager, 'get_sso_configuration', return_value=mock_config):
            with patch('ldap3.Server'), patch('ldap3.Connection') as mock_conn_class:
                # Mock LDAP connections
                mock_bind_conn = Mock()
                mock_bind_conn.bind.return_value = True
                mock_bind_conn.entries = [Mock(entry_dn=f"uid={self.test_user_id},ou=users,dc=example,dc=com")]
                
                mock_user_conn = Mock()
                mock_user_conn.bind.return_value = True
                
                mock_conn_class.side_effect = [mock_bind_conn, mock_user_conn]
                
                with patch('jwt.encode', return_value="test-ldap-token"):
                    credentials = {
                        "username": self.test_user_id,
                        "password": "test-password"
                    }
                    
                    result = self.sso_manager.authenticate_user(self.test_provider_id, credentials)
                    
                    # Verify authentication success
                    assert result.success is True
                    assert result.user_id == self.test_user_id
                    assert result.session_token == "test-ldap-token"
    
    @patch('scrollintel.core.sso_manager.get_database_session')
    def test_mfa_setup_and_verification(self, mock_db_session):
        """Test MFA setup and verification"""
        # Mock database session
        mock_session = Mock()
        mock_db_session.return_value.__enter__.return_value = mock_session
        
        # Setup MFA
        mfa_setup = self.sso_manager.setup_mfa(self.test_user_id, MFAType.TOTP)
        
        # Verify MFA setup
        assert 'mfa_id' in mfa_setup
        assert 'secret_key' in mfa_setup
        assert 'backup_codes' in mfa_setup
        assert 'qr_code_url' in mfa_setup
        assert len(mfa_setup['backup_codes']) == 10
        
        # Test MFA challenge creation
        challenge = self.sso_manager.create_mfa_challenge(self.test_user_id)
        
        assert challenge.challenge_id is not None
        assert challenge.mfa_type == MFAType.TOTP
        assert challenge.expires_at > datetime.utcnow()
        
        # Test MFA verification
        is_valid = self.sso_manager.verify_mfa_challenge(
            challenge.challenge_id,
            "123456"  # Test code
        )
        
        assert isinstance(is_valid, bool)
    
    @patch('scrollintel.core.sso_manager.get_database_session')
    def test_user_synchronization(self, mock_db_session):
        """Test user attribute synchronization"""
        # Mock database session
        mock_session = Mock()
        mock_db_session.return_value.__enter__.return_value = mock_session
        
        # Mock SSO configuration
        mock_config = Mock()
        mock_config.id = self.test_provider_id
        mock_config.provider_type = "oauth2"
        mock_config.user_mapping = {
            "local_email": "email",
            "local_name": "name"
        }
        
        # Mock active session
        mock_session_obj = Mock()
        mock_session_obj.session_token = "test-token"
        
        with patch.object(self.sso_manager, 'get_sso_configuration', return_value=mock_config):
            with patch.object(self.sso_manager, '_get_active_user_session', return_value=mock_session_obj):
                with patch('requests.get') as mock_get:
                    # Mock user info response
                    mock_get.return_value.status_code = 200
                    mock_get.return_value.json.return_value = {
                        "sub": self.test_user_id,
                        "email": "test@example.com",
                        "name": "Test User"
                    }
                    
                    # Sync user attributes
                    user_profile = self.sso_manager.sync_user_attributes(
                        self.test_user_id,
                        self.test_provider_id
                    )
                    
                    # Verify user profile
                    assert user_profile.user_id == self.test_user_id
                    assert user_profile.email == "test@example.com"
                    assert user_profile.display_name == "Test User"
    
    def test_permission_validation(self):
        """Test user permission validation"""
        # Test permission validation
        has_permission = self.sso_manager.validate_permissions(
            self.test_user_id,
            "dashboard:read"
        )
        
        # Should return boolean
        assert isinstance(has_permission, bool)
    
    @patch('scrollintel.core.sso_manager.get_database_session')
    def test_user_logout(self, mock_db_session):
        """Test user logout functionality"""
        # Mock database session
        mock_session = Mock()
        mock_db_session.return_value.__enter__.return_value = mock_session
        
        # Mock user sessions
        mock_sessions = [Mock(), Mock()]
        mock_session.query.return_value.filter.return_value.all.return_value = mock_sessions
        
        # Logout user
        success = self.sso_manager.logout_user(self.test_user_id)
        
        # Verify logout
        assert success is True
        assert mock_session.delete.call_count == 2
        mock_session.commit.assert_called_once()

class TestSSOProviders:
    """Test individual SSO providers"""
    
    def test_oauth2_provider_token_exchange(self):
        """Test OAuth2 provider token exchange"""
        config = {
            "client_id": "test-client",
            "client_secret": "test-secret",
            "token_url": "https://example.com/token",
            "userinfo_url": "https://example.com/userinfo",
            "redirect_uri": "https://app.com/callback"
        }
        
        provider = OAuth2Provider(config)
        
        with patch('requests.post') as mock_post:
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = {
                "access_token": "test-token",
                "expires_in": 3600
            }
            
            # Test token exchange
            tokens = provider._exchange_code_for_tokens("test-code")
            
            assert tokens["access_token"] == "test-token"
            assert tokens["expires_in"] == 3600
    
    def test_saml_provider_assertion_validation(self):
        """Test SAML provider assertion validation"""
        config = {
            "idp_url": "https://example.com/saml",
            "sp_entity_id": "test-sp",
            "certificate": "test-cert",
            "private_key": "test-key"
        }
        
        provider = SAMLProvider(config)
        
        # Test with valid XML
        valid_xml = b"<saml:Assertion xmlns:saml='urn:oasis:names:tc:SAML:2.0:assertion'></saml:Assertion>"
        
        with patch('xml.etree.ElementTree.fromstring') as mock_parse:
            mock_element = Mock()
            mock_parse.return_value = mock_element
            
            assertion = provider._validate_saml_assertion(valid_xml)
            assert assertion == mock_element
    
    def test_ldap_provider_connection(self):
        """Test LDAP provider connection"""
        config = {
            "server_url": "ldap://example.com",
            "bind_dn": "cn=admin,dc=example,dc=com",
            "bind_password": "password",
            "user_search_base": "ou=users,dc=example,dc=com",
            "user_search_filter": "(uid={username})"
        }
        
        provider = LDAPProvider(config)
        
        with patch('ldap3.Server'), patch('ldap3.Connection') as mock_conn:
            mock_connection = Mock()
            mock_connection.bind.return_value = True
            mock_connection.entries = [
                Mock(entry_dn="uid=testuser,ou=users,dc=example,dc=com")
            ]
            mock_conn.return_value = mock_connection
            
            # Test user search
            credentials = {"username": "testuser", "password": "password"}
            result = provider.authenticate(credentials)
            
            # Should attempt authentication
            assert mock_conn.call_count >= 2  # Bind connection + user connection

class TestMFAManager:
    """Test MFA manager functionality"""
    
    def setup_method(self):
        """Setup test environment"""
        self.mfa_manager = MFAManager("test-encryption-key-32-bytes-long")
    
    def test_totp_secret_generation(self):
        """Test TOTP secret generation"""
        secret = self.mfa_manager.generate_totp_secret()
        
        assert len(secret) == 32  # Base32 encoded 20 bytes
        assert secret.isalnum()
        assert secret.isupper()
    
    def test_backup_codes_generation(self):
        """Test backup codes generation"""
        codes = self.mfa_manager.generate_backup_codes(5)
        
        assert len(codes) == 5
        for code in codes:
            assert len(code) == 8  # 4 bytes hex = 8 characters
            assert code.isupper()
    
    def test_mfa_challenge_creation(self):
        """Test MFA challenge creation"""
        challenge = self.mfa_manager.create_mfa_challenge("test_user", MFAType.TOTP)
        
        assert challenge.challenge_id is not None
        assert challenge.mfa_type == MFAType.TOTP
        assert challenge.expires_at > datetime.utcnow()
    
    def test_mfa_code_verification(self):
        """Test MFA code verification"""
        # Test valid code format
        is_valid = self.mfa_manager.verify_mfa_code("challenge_id", "123456", "secret")
        assert isinstance(is_valid, bool)
        
        # Test invalid code format
        is_valid = self.mfa_manager.verify_mfa_code("challenge_id", "invalid", "secret")
        assert is_valid is False

if __name__ == "__main__":
    pytest.main([__file__])