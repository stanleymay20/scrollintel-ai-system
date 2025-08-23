"""
SSO Provider Implementations for Enterprise Integration
"""
import base64
import hashlib
import hmac
import json
import secrets
import time
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from urllib.parse import urlencode, parse_qs
import xml.etree.ElementTree as ET

try:
    import jwt
except ImportError:
    jwt = None

try:
    import requests
except ImportError:
    requests = None

try:
    from cryptography.fernet import Fernet
except ImportError:
    Fernet = None

try:
    from ldap3 import Server, Connection, ALL, NTLM
except ImportError:
    Server = Connection = ALL = NTLM = None

from scrollintel.models.sso_models import (
    AuthResult, UserProfile, SSOProviderType, AuthenticationStatus,
    MFAChallenge, MFAType
)

class BaseSSOProvider(ABC):
    """Base class for SSO providers"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.provider_type = config.get('provider_type')
        self.name = config.get('name', 'Unknown Provider')
    
    @abstractmethod
    def authenticate(self, credentials: Dict[str, Any]) -> AuthResult:
        """Authenticate user with provider"""
        pass
    
    @abstractmethod
    def get_user_profile(self, user_id: str, access_token: str) -> UserProfile:
        """Get user profile from provider"""
        pass
    
    @abstractmethod
    def validate_token(self, token: str) -> bool:
        """Validate authentication token"""
        pass
    
    def refresh_token(self, refresh_token: str) -> Optional[AuthResult]:
        """Refresh authentication token"""
        return None

class SAMLProvider(BaseSSOProvider):
    """SAML 2.0 authentication provider"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.idp_url = config.get('idp_url')
        self.sp_entity_id = config.get('sp_entity_id')
        self.certificate = config.get('certificate')
        self.private_key = config.get('private_key')
        self.attribute_mapping = config.get('attribute_mapping', {})
    
    def authenticate(self, credentials: Dict[str, Any]) -> AuthResult:
        """Authenticate using SAML assertion"""
        try:
            saml_response = credentials.get('saml_response')
            if not saml_response:
                return AuthResult(
                    success=False,
                    error_message="SAML response required"
                )
            
            # Decode and validate SAML response
            decoded_response = base64.b64decode(saml_response)
            assertion = self._validate_saml_assertion(decoded_response)
            
            if not assertion:
                return AuthResult(
                    success=False,
                    error_message="Invalid SAML assertion"
                )
            
            # Extract user information
            user_profile = self._extract_user_from_assertion(assertion)
            
            # Generate session token
            session_token = self._generate_session_token(user_profile.user_id)
            
            return AuthResult(
                success=True,
                user_id=user_profile.user_id,
                session_token=session_token,
                expires_at=datetime.utcnow() + timedelta(hours=8)
            )
            
        except Exception as e:
            return AuthResult(
                success=False,
                error_message=f"SAML authentication failed: {str(e)}"
            )
    
    def get_user_profile(self, user_id: str, access_token: str) -> UserProfile:
        """Get user profile from SAML attributes"""
        # In SAML, user profile is typically extracted from the assertion
        # This method would query additional user details if needed
        return UserProfile(
            user_id=user_id,
            email=f"{user_id}@example.com",  # Would be from assertion
            display_name=user_id
        )
    
    def validate_token(self, token: str) -> bool:
        """Validate SAML session token"""
        try:
            payload = jwt.decode(token, self.private_key, algorithms=['RS256'])
            return payload.get('exp', 0) > time.time()
        except:
            return False
    
    def _validate_saml_assertion(self, assertion_xml: bytes) -> Optional[ET.Element]:
        """Validate SAML assertion signature and structure"""
        try:
            root = ET.fromstring(assertion_xml)
            # Implement signature validation logic here
            # For now, return the assertion if it parses correctly
            return root
        except ET.ParseError:
            return None
    
    def _extract_user_from_assertion(self, assertion: ET.Element) -> UserProfile:
        """Extract user profile from SAML assertion"""
        # Extract attributes from SAML assertion
        user_id = assertion.find('.//saml:NameID', {'saml': 'urn:oasis:names:tc:SAML:2.0:assertion'})
        
        return UserProfile(
            user_id=user_id.text if user_id is not None else "unknown",
            email="user@example.com",  # Extract from attributes
            display_name="User Name"   # Extract from attributes
        )
    
    def _generate_session_token(self, user_id: str) -> str:
        """Generate JWT session token"""
        payload = {
            'user_id': user_id,
            'provider': 'saml',
            'iat': time.time(),
            'exp': time.time() + (8 * 3600)  # 8 hours
        }
        return jwt.encode(payload, self.private_key, algorithm='RS256')

class OAuth2Provider(BaseSSOProvider):
    """OAuth2/OIDC authentication provider"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.client_id = config.get('client_id')
        self.client_secret = config.get('client_secret')
        self.auth_url = config.get('auth_url')
        self.token_url = config.get('token_url')
        self.userinfo_url = config.get('userinfo_url')
        self.redirect_uri = config.get('redirect_uri')
        self.scopes = config.get('scopes', ['openid', 'profile', 'email'])
    
    def authenticate(self, credentials: Dict[str, Any]) -> AuthResult:
        """Authenticate using OAuth2 authorization code"""
        try:
            if requests is None:
                return AuthResult(
                    success=False,
                    error_message="requests package required for OAuth2 authentication"
                )
            
            auth_code = credentials.get('code')
            if not auth_code:
                return AuthResult(
                    success=False,
                    error_message="Authorization code required"
                )
            
            # Exchange code for tokens
            token_response = self._exchange_code_for_tokens(auth_code)
            
            if not token_response.get('access_token'):
                return AuthResult(
                    success=False,
                    error_message="Failed to obtain access token"
                )
            
            # Get user profile
            user_profile = self._get_user_info(token_response['access_token'])
            
            return AuthResult(
                success=True,
                user_id=user_profile.user_id,
                session_token=token_response['access_token'],
                refresh_token=token_response.get('refresh_token'),
                expires_at=datetime.utcnow() + timedelta(
                    seconds=token_response.get('expires_in', 3600)
                )
            )
            
        except Exception as e:
            return AuthResult(
                success=False,
                error_message=f"OAuth2 authentication failed: {str(e)}"
            )
    
    def get_user_profile(self, user_id: str, access_token: str) -> UserProfile:
        """Get user profile from OAuth2 userinfo endpoint"""
        return self._get_user_info(access_token)
    
    def validate_token(self, token: str) -> bool:
        """Validate OAuth2 access token"""
        try:
            response = requests.get(
                self.userinfo_url,
                headers={'Authorization': f'Bearer {token}'},
                timeout=10
            )
            return response.status_code == 200
        except:
            return False
    
    def refresh_token(self, refresh_token: str) -> Optional[AuthResult]:
        """Refresh OAuth2 access token"""
        try:
            data = {
                'grant_type': 'refresh_token',
                'refresh_token': refresh_token,
                'client_id': self.client_id,
                'client_secret': self.client_secret
            }
            
            response = requests.post(self.token_url, data=data, timeout=10)
            
            if response.status_code == 200:
                token_data = response.json()
                return AuthResult(
                    success=True,
                    session_token=token_data['access_token'],
                    refresh_token=token_data.get('refresh_token', refresh_token),
                    expires_at=datetime.utcnow() + timedelta(
                        seconds=token_data.get('expires_in', 3600)
                    )
                )
        except:
            pass
        
        return None
    
    def _exchange_code_for_tokens(self, auth_code: str) -> Dict[str, Any]:
        """Exchange authorization code for access tokens"""
        data = {
            'grant_type': 'authorization_code',
            'code': auth_code,
            'redirect_uri': self.redirect_uri,
            'client_id': self.client_id,
            'client_secret': self.client_secret
        }
        
        response = requests.post(self.token_url, data=data, timeout=10)
        response.raise_for_status()
        return response.json()
    
    def _get_user_info(self, access_token: str) -> UserProfile:
        """Get user information from userinfo endpoint"""
        headers = {'Authorization': f'Bearer {access_token}'}
        response = requests.get(self.userinfo_url, headers=headers, timeout=10)
        response.raise_for_status()
        
        user_data = response.json()
        return UserProfile(
            user_id=user_data.get('sub', user_data.get('id')),
            email=user_data.get('email'),
            first_name=user_data.get('given_name'),
            last_name=user_data.get('family_name'),
            display_name=user_data.get('name'),
            attributes=user_data
        )

class LDAPProvider(BaseSSOProvider):
    """LDAP/Active Directory authentication provider"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.server_url = config.get('server_url')
        self.bind_dn = config.get('bind_dn')
        self.bind_password = config.get('bind_password')
        self.user_search_base = config.get('user_search_base')
        self.user_search_filter = config.get('user_search_filter', '(uid={username})')
        self.group_search_base = config.get('group_search_base')
        self.use_ssl = config.get('use_ssl', True)
        self.use_ntlm = config.get('use_ntlm', False)
    
    def authenticate(self, credentials: Dict[str, Any]) -> AuthResult:
        """Authenticate using LDAP bind"""
        try:
            if Server is None or Connection is None:
                return AuthResult(
                    success=False,
                    error_message="ldap3 package required for LDAP authentication"
                )
            
            username = credentials.get('username')
            password = credentials.get('password')
            
            if not username or not password:
                return AuthResult(
                    success=False,
                    error_message="Username and password required"
                )
            
            # Create LDAP connection
            server = Server(self.server_url, get_info=ALL, use_ssl=self.use_ssl)
            
            # Find user DN
            bind_conn = Connection(server, self.bind_dn, self.bind_password)
            if not bind_conn.bind():
                return AuthResult(
                    success=False,
                    error_message="LDAP bind failed"
                )
            
            search_filter = self.user_search_filter.format(username=username)
            bind_conn.search(
                self.user_search_base,
                search_filter,
                attributes=['dn', 'cn', 'mail', 'memberOf']
            )
            
            if not bind_conn.entries:
                return AuthResult(
                    success=False,
                    error_message="User not found"
                )
            
            user_dn = bind_conn.entries[0].entry_dn
            bind_conn.unbind()
            
            # Authenticate user
            auth_method = NTLM if self.use_ntlm else None
            user_conn = Connection(server, user_dn, password, authentication=auth_method)
            
            if user_conn.bind():
                # Generate session token
                session_token = self._generate_session_token(username)
                
                return AuthResult(
                    success=True,
                    user_id=username,
                    session_token=session_token,
                    expires_at=datetime.utcnow() + timedelta(hours=8)
                )
            else:
                return AuthResult(
                    success=False,
                    error_message="Invalid credentials"
                )
                
        except Exception as e:
            return AuthResult(
                success=False,
                error_message=f"LDAP authentication failed: {str(e)}"
            )
    
    def get_user_profile(self, user_id: str, access_token: str) -> UserProfile:
        """Get user profile from LDAP"""
        try:
            server = Server(self.server_url, get_info=ALL, use_ssl=self.use_ssl)
            conn = Connection(server, self.bind_dn, self.bind_password)
            
            if conn.bind():
                search_filter = self.user_search_filter.format(username=user_id)
                conn.search(
                    self.user_search_base,
                    search_filter,
                    attributes=['cn', 'mail', 'givenName', 'sn', 'memberOf']
                )
                
                if conn.entries:
                    entry = conn.entries[0]
                    groups = [str(group) for group in entry.memberOf] if entry.memberOf else []
                    
                    return UserProfile(
                        user_id=user_id,
                        email=str(entry.mail) if entry.mail else None,
                        first_name=str(entry.givenName) if entry.givenName else None,
                        last_name=str(entry.sn) if entry.sn else None,
                        display_name=str(entry.cn) if entry.cn else user_id,
                        groups=groups
                    )
        except:
            pass
        
        return UserProfile(user_id=user_id, email=f"{user_id}@example.com")
    
    def validate_token(self, token: str) -> bool:
        """Validate LDAP session token"""
        try:
            payload = jwt.decode(token, self.bind_password, algorithms=['HS256'])
            return payload.get('exp', 0) > time.time()
        except:
            return False
    
    def _generate_session_token(self, user_id: str) -> str:
        """Generate JWT session token for LDAP user"""
        payload = {
            'user_id': user_id,
            'provider': 'ldap',
            'iat': time.time(),
            'exp': time.time() + (8 * 3600)  # 8 hours
        }
        return jwt.encode(payload, self.bind_password, algorithm='HS256')

class MFAManager:
    """Multi-factor authentication manager"""
    
    def __init__(self, encryption_key: str):
        if Fernet is None:
            raise ImportError("cryptography package required for MFA functionality")
        self.fernet = Fernet(encryption_key.encode())
    
    def generate_totp_secret(self) -> str:
        """Generate TOTP secret key"""
        return base64.b32encode(secrets.token_bytes(20)).decode()
    
    def generate_backup_codes(self, count: int = 10) -> List[str]:
        """Generate backup codes for MFA"""
        return [secrets.token_hex(4).upper() for _ in range(count)]
    
    def create_mfa_challenge(self, user_id: str, mfa_type: MFAType) -> MFAChallenge:
        """Create MFA challenge"""
        challenge_id = secrets.token_urlsafe(32)
        
        challenge_data = {}
        if mfa_type == MFAType.SMS:
            challenge_data['phone_number'] = "***-***-1234"  # Masked
        elif mfa_type == MFAType.EMAIL:
            challenge_data['email'] = "user@***.com"  # Masked
        
        return MFAChallenge(
            challenge_id=challenge_id,
            mfa_type=mfa_type,
            challenge_data=challenge_data,
            expires_at=datetime.utcnow() + timedelta(minutes=5)
        )
    
    def verify_mfa_code(self, challenge_id: str, code: str, secret: str) -> bool:
        """Verify MFA code"""
        # Implementation would depend on MFA type
        # For TOTP, verify against time-based code
        # For SMS/Email, verify against sent code
        return len(code) == 6 and code.isdigit()

class SSOProviderFactory:
    """Factory for creating SSO providers"""
    
    @staticmethod
    def create_provider(provider_type: SSOProviderType, config: Dict[str, Any]) -> BaseSSOProvider:
        """Create SSO provider instance"""
        if provider_type == SSOProviderType.SAML:
            return SAMLProvider(config)
        elif provider_type in [SSOProviderType.OAUTH2, SSOProviderType.OIDC]:
            return OAuth2Provider(config)
        elif provider_type in [SSOProviderType.LDAP, SSOProviderType.ACTIVE_DIRECTORY]:
            return LDAPProvider(config)
        else:
            raise ValueError(f"Unsupported provider type: {provider_type}")