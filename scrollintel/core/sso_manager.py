"""
SSO Manager for Enterprise Integration
Handles SSO configuration, authentication, and user synchronization
"""
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from sqlalchemy.orm import Session

from scrollintel.models.sso_models import (
    SSOConfiguration, AuthProvider, UserSession, MFAConfiguration,
    SSOConfigurationCreate, AuthResult, UserProfile, MFAChallenge,
    SSOProviderType, MFAType
)
from scrollintel.core.sso_providers import SSOProviderFactory, MFAManager
from scrollintel.core.database_connection_manager import get_sync_session

class SSOManager:
    """Main SSO management class"""
    
    def __init__(self):
        # Generate a proper Fernet key for MFA encryption
        import base64
        import os
        encryption_key = base64.urlsafe_b64encode(os.urandom(32)).decode()
        
        try:
            self.mfa_manager = MFAManager(encryption_key)
        except ImportError:
            # If cryptography is not available, use a mock MFA manager
            self.mfa_manager = None
        
        self._provider_cache = {}
    
    def create_sso_configuration(self, config_data: SSOConfigurationCreate, created_by: str) -> str:
        """Create new SSO configuration"""
        with get_sync_session() as db:
            config_id = str(uuid.uuid4())
            
            sso_config = SSOConfiguration(
                id=config_id,
                name=config_data.name,
                provider_type=config_data.provider_type.value,
                config=config_data.config,
                user_mapping=config_data.user_mapping,
                created_by=created_by
            )
            
            db.add(sso_config)
            db.commit()
            
            return config_id
    
    def get_sso_configuration(self, config_id: str) -> Optional[SSOConfiguration]:
        """Get SSO configuration by ID"""
        with get_sync_session() as db:
            return db.query(SSOConfiguration).filter(
                SSOConfiguration.id == config_id
            ).first()
    
    def list_sso_configurations(self, active_only: bool = True) -> List[SSOConfiguration]:
        """List all SSO configurations"""
        with get_sync_session() as db:
            query = db.query(SSOConfiguration)
            if active_only:
                query = query.filter(SSOConfiguration.is_active == True)
            return query.all()
    
    def authenticate_user(self, provider_id: str, credentials: Dict[str, Any]) -> AuthResult:
        """Authenticate user with specified provider"""
        try:
            # Get provider configuration
            config = self.get_sso_configuration(provider_id)
            if not config or not config.is_active:
                return AuthResult(
                    success=False,
                    error_message="Provider not found or inactive"
                )
            
            # Create provider instance
            provider = self._get_provider_instance(config)
            
            # Authenticate with provider
            auth_result = provider.authenticate(credentials)
            
            if auth_result.success:
                # Create user session
                session_id = self._create_user_session(
                    auth_result.user_id,
                    provider_id,
                    auth_result.session_token,
                    auth_result.refresh_token,
                    auth_result.expires_at,
                    credentials.get('ip_address'),
                    credentials.get('user_agent')
                )
                
                # Check if MFA is required
                if self._is_mfa_required(auth_result.user_id):
                    mfa_methods = self._get_user_mfa_methods(auth_result.user_id)
                    auth_result.requires_mfa = True
                    auth_result.mfa_methods = mfa_methods
                
                # Sync user profile
                self._sync_user_profile(auth_result.user_id, provider, auth_result.session_token)
            
            return auth_result
            
        except Exception as e:
            return AuthResult(
                success=False,
                error_message=f"Authentication failed: {str(e)}"
            )
    
    def sync_user_attributes(self, user_id: str, provider_id: str) -> UserProfile:
        """Synchronize user attributes from provider"""
        config = self.get_sso_configuration(provider_id)
        if not config:
            raise ValueError("Provider configuration not found")
        
        provider = self._get_provider_instance(config)
        
        # Get current session for access token
        session = self._get_active_user_session(user_id, provider_id)
        if not session:
            raise ValueError("No active session found")
        
        # Get user profile from provider
        user_profile = provider.get_user_profile(user_id, session.session_token)
        
        # Apply user mapping
        mapped_profile = self._apply_user_mapping(user_profile, config.user_mapping)
        
        return mapped_profile
    
    def validate_permissions(self, user_id: str, resource: str) -> bool:
        """Validate user permissions for resource"""
        # Get user profile with roles and groups
        user_profile = self._get_cached_user_profile(user_id)
        if not user_profile:
            return False
        
        # Check permissions based on roles and groups
        # This would integrate with your authorization system
        return self._check_resource_permissions(user_profile, resource)
    
    def logout_user(self, user_id: str, session_id: Optional[str] = None) -> bool:
        """Logout user and invalidate sessions"""
        try:
            with get_sync_session() as db:
                query = db.query(UserSession).filter(UserSession.user_id == user_id)
                
                if session_id:
                    query = query.filter(UserSession.id == session_id)
                
                sessions = query.all()
                for session in sessions:
                    db.delete(session)
                
                db.commit()
                return True
                
        except Exception:
            return False
    
    def setup_mfa(self, user_id: str, mfa_type: MFAType) -> Dict[str, Any]:
        """Setup multi-factor authentication for user"""
        if self.mfa_manager is None:
            raise ImportError("MFA functionality requires cryptography package")
        
        with get_sync_session() as db:
            # Generate MFA configuration
            mfa_id = str(uuid.uuid4())
            secret_key = self.mfa_manager.generate_totp_secret()
            backup_codes = self.mfa_manager.generate_backup_codes()
            
            # Encrypt sensitive data
            encrypted_secret = self.mfa_manager.fernet.encrypt(secret_key.encode()).decode()
            encrypted_codes = [
                self.mfa_manager.fernet.encrypt(code.encode()).decode()
                for code in backup_codes
            ]
            
            mfa_config = MFAConfiguration(
                id=mfa_id,
                user_id=user_id,
                mfa_type=mfa_type.value,
                secret_key=encrypted_secret,
                backup_codes=encrypted_codes
            )
            
            db.add(mfa_config)
            db.commit()
            
            return {
                'mfa_id': mfa_id,
                'secret_key': secret_key,
                'backup_codes': backup_codes,
                'qr_code_url': self._generate_qr_code_url(user_id, secret_key)
            }
    
    def create_mfa_challenge(self, user_id: str) -> MFAChallenge:
        """Create MFA challenge for user"""
        if self.mfa_manager is None:
            raise ImportError("MFA functionality requires cryptography package")
        
        # Get user's MFA methods
        mfa_methods = self._get_user_mfa_methods(user_id)
        
        if not mfa_methods:
            raise ValueError("No MFA methods configured")
        
        # Use first available method
        mfa_type = mfa_methods[0]
        
        return self.mfa_manager.create_mfa_challenge(user_id, mfa_type)
    
    def verify_mfa_challenge(self, challenge_id: str, verification_code: str) -> bool:
        """Verify MFA challenge"""
        if self.mfa_manager is None:
            raise ImportError("MFA functionality requires cryptography package")
        
        # This would retrieve the challenge from cache/database
        # and verify the code
        return self.mfa_manager.verify_mfa_code(challenge_id, verification_code, "secret")
    
    def _get_provider_instance(self, config: SSOConfiguration):
        """Get cached provider instance"""
        if config.id not in self._provider_cache:
            provider_type = SSOProviderType(config.provider_type)
            self._provider_cache[config.id] = SSOProviderFactory.create_provider(
                provider_type, config.config
            )
        
        return self._provider_cache[config.id]
    
    def _create_user_session(self, user_id: str, provider_id: str, session_token: str,
                           refresh_token: Optional[str], expires_at: datetime,
                           ip_address: Optional[str], user_agent: Optional[str]) -> str:
        """Create user session record"""
        with get_sync_session() as db:
            session_id = str(uuid.uuid4())
            
            user_session = UserSession(
                id=session_id,
                user_id=user_id,
                provider_id=provider_id,
                session_token=session_token,
                refresh_token=refresh_token,
                expires_at=expires_at,
                ip_address=ip_address,
                user_agent=user_agent
            )
            
            db.add(user_session)
            db.commit()
            
            return session_id
    
    def _get_active_user_session(self, user_id: str, provider_id: str) -> Optional[UserSession]:
        """Get active user session"""
        with get_sync_session() as db:
            return db.query(UserSession).filter(
                UserSession.user_id == user_id,
                UserSession.provider_id == provider_id,
                UserSession.expires_at > datetime.utcnow()
            ).first()
    
    def _is_mfa_required(self, user_id: str) -> bool:
        """Check if MFA is required for user"""
        with get_sync_session() as db:
            mfa_config = db.query(MFAConfiguration).filter(
                MFAConfiguration.user_id == user_id,
                MFAConfiguration.is_active == True
            ).first()
            
            return mfa_config is not None
    
    def _get_user_mfa_methods(self, user_id: str) -> List[MFAType]:
        """Get user's configured MFA methods"""
        with get_sync_session() as db:
            mfa_configs = db.query(MFAConfiguration).filter(
                MFAConfiguration.user_id == user_id,
                MFAConfiguration.is_active == True
            ).all()
            
            return [MFAType(config.mfa_type) for config in mfa_configs]
    
    def _sync_user_profile(self, user_id: str, provider, access_token: str):
        """Sync user profile from provider"""
        try:
            user_profile = provider.get_user_profile(user_id, access_token)
            # Cache user profile for quick access
            # This would typically be stored in Redis or database
            self._cache_user_profile(user_id, user_profile)
        except Exception:
            # Log error but don't fail authentication
            pass
    
    def _apply_user_mapping(self, user_profile: UserProfile, mapping: Dict[str, str]) -> UserProfile:
        """Apply user attribute mapping"""
        # Apply configured attribute mapping
        mapped_attributes = {}
        for local_attr, remote_attr in mapping.items():
            if hasattr(user_profile, remote_attr):
                mapped_attributes[local_attr] = getattr(user_profile, remote_attr)
        
        # Update user profile with mapped attributes
        user_profile.attributes.update(mapped_attributes)
        return user_profile
    
    def _get_cached_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """Get cached user profile"""
        # This would retrieve from cache (Redis, etc.)
        # For now, return a basic profile
        return UserProfile(user_id=user_id, email=f"{user_id}@example.com")
    
    def _cache_user_profile(self, user_id: str, profile: UserProfile):
        """Cache user profile"""
        # This would store in cache (Redis, etc.)
        pass
    
    def _check_resource_permissions(self, user_profile: UserProfile, resource: str) -> bool:
        """Check if user has permission for resource"""
        # Implement your authorization logic here
        # This could check roles, groups, or specific permissions
        return True  # Placeholder
    
    def _generate_qr_code_url(self, user_id: str, secret_key: str) -> str:
        """Generate QR code URL for TOTP setup"""
        issuer = "ScrollIntel"
        return f"otpauth://totp/{issuer}:{user_id}?secret={secret_key}&issuer={issuer}"