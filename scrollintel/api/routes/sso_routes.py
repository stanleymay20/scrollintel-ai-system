"""
SSO API Routes for Enterprise Integration
"""
from datetime import datetime
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Depends, Request, Response
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

from scrollintel.core.sso_manager import SSOManager
from scrollintel.models.sso_models import (
    SSOConfigurationCreate, SSOConfigurationResponse, AuthResult,
    UserProfile, MFAChallenge, MFAVerification, SSOProviderType, MFAType
)

router = APIRouter(prefix="/api/v1/sso", tags=["SSO"])
security = HTTPBearer()

# Initialize SSO manager lazily to avoid import issues
_sso_manager = None

def get_sso_manager() -> SSOManager:
    """Get SSO manager instance"""
    global _sso_manager
    if _sso_manager is None:
        _sso_manager = SSOManager()
    return _sso_manager

# Request/Response Models
class AuthenticationRequest(BaseModel):
    """Authentication request"""
    provider_id: str
    credentials: Dict[str, Any]

class MFASetupRequest(BaseModel):
    """MFA setup request"""
    mfa_type: MFAType

class MFASetupResponse(BaseModel):
    """MFA setup response"""
    mfa_id: str
    secret_key: str
    backup_codes: List[str]
    qr_code_url: str

class SessionInfo(BaseModel):
    """Session information"""
    user_id: str
    provider_id: str
    expires_at: datetime
    last_activity: datetime

# Dependency to get current user
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """Get current authenticated user"""
    # This would validate the JWT token and return user ID
    # For now, return a placeholder
    return "current_user_id"

@router.post("/configurations", response_model=str)
async def create_sso_configuration(
    config: SSOConfigurationCreate,
    current_user: str = Depends(get_current_user)
):
    """Create new SSO configuration"""
    try:
        sso_manager = get_sso_manager()
        config_id = sso_manager.create_sso_configuration(config, current_user)
        return config_id
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/configurations", response_model=List[SSOConfigurationResponse])
async def list_sso_configurations(
    active_only: bool = True,
    current_user: str = Depends(get_current_user)
):
    """List SSO configurations"""
    try:
        sso_manager = get_sso_manager()
        configurations = sso_manager.list_sso_configurations(active_only)
        return [
            SSOConfigurationResponse(
                id=config.id,
                name=config.name,
                provider_type=SSOProviderType(config.provider_type),
                is_active=config.is_active,
                created_at=config.created_at,
                updated_at=config.updated_at
            )
            for config in configurations
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/configurations/{config_id}", response_model=SSOConfigurationResponse)
async def get_sso_configuration(
    config_id: str,
    current_user: str = Depends(get_current_user)
):
    """Get SSO configuration by ID"""
    try:
        sso_manager = get_sso_manager()
        config = sso_manager.get_sso_configuration(config_id)
        if not config:
            raise HTTPException(status_code=404, detail="Configuration not found")
        
        return SSOConfigurationResponse(
            id=config.id,
            name=config.name,
            provider_type=SSOProviderType(config.provider_type),
            is_active=config.is_active,
            created_at=config.created_at,
            updated_at=config.updated_at
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/authenticate", response_model=AuthResult)
async def authenticate(
    auth_request: AuthenticationRequest,
    request: Request
):
    """Authenticate user with SSO provider"""
    try:
        # Add request metadata to credentials
        credentials = auth_request.credentials.copy()
        credentials['ip_address'] = request.client.host
        credentials['user_agent'] = request.headers.get('user-agent')
        
        sso_manager = get_sso_manager()
        auth_result = sso_manager.authenticate_user(
            auth_request.provider_id,
            credentials
        )
        
        return auth_result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/sync-user/{user_id}", response_model=UserProfile)
async def sync_user_attributes(
    user_id: str,
    provider_id: str,
    current_user: str = Depends(get_current_user)
):
    """Synchronize user attributes from SSO provider"""
    try:
        sso_manager = get_sso_manager()
        user_profile = sso_manager.sync_user_attributes(user_id, provider_id)
        return user_profile
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/validate-permissions/{resource}")
async def validate_permissions(
    resource: str,
    current_user: str = Depends(get_current_user)
):
    """Validate user permissions for resource"""
    try:
        sso_manager = get_sso_manager()
        has_permission = sso_manager.validate_permissions(current_user, resource)
        return {"has_permission": has_permission}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/logout")
async def logout(
    session_id: Optional[str] = None,
    current_user: str = Depends(get_current_user)
):
    """Logout user and invalidate sessions"""
    try:
        sso_manager = get_sso_manager()
        success = sso_manager.logout_user(current_user, session_id)
        return {"success": success}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/mfa/setup", response_model=MFASetupResponse)
async def setup_mfa(
    mfa_request: MFASetupRequest,
    current_user: str = Depends(get_current_user)
):
    """Setup multi-factor authentication"""
    try:
        sso_manager = get_sso_manager()
        mfa_setup = sso_manager.setup_mfa(current_user, mfa_request.mfa_type)
        return MFASetupResponse(**mfa_setup)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/mfa/challenge", response_model=MFAChallenge)
async def create_mfa_challenge(
    current_user: str = Depends(get_current_user)
):
    """Create MFA challenge"""
    try:
        sso_manager = get_sso_manager()
        challenge = sso_manager.create_mfa_challenge(current_user)
        return challenge
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/mfa/verify")
async def verify_mfa(
    verification: MFAVerification,
    current_user: str = Depends(get_current_user)
):
    """Verify MFA challenge"""
    try:
        sso_manager = get_sso_manager()
        is_valid = sso_manager.verify_mfa_challenge(
            verification.challenge_id,
            verification.verification_code
        )
        return {"valid": is_valid}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Provider-specific endpoints
@router.get("/providers/azure-ad/auth-url")
async def get_azure_ad_auth_url(
    redirect_uri: str,
    state: Optional[str] = None
):
    """Get Azure AD authorization URL"""
    # This would generate the OAuth2 authorization URL for Azure AD
    auth_url = f"https://login.microsoftonline.com/common/oauth2/v2.0/authorize"
    params = {
        "client_id": "your-client-id",
        "response_type": "code",
        "redirect_uri": redirect_uri,
        "scope": "openid profile email",
        "state": state or "random-state"
    }
    
    query_string = "&".join([f"{k}={v}" for k, v in params.items()])
    return {"auth_url": f"{auth_url}?{query_string}"}

@router.get("/providers/okta/auth-url")
async def get_okta_auth_url(
    redirect_uri: str,
    okta_domain: str,
    state: Optional[str] = None
):
    """Get Okta authorization URL"""
    auth_url = f"https://{okta_domain}/oauth2/default/v1/authorize"
    params = {
        "client_id": "your-client-id",
        "response_type": "code",
        "redirect_uri": redirect_uri,
        "scope": "openid profile email",
        "state": state or "random-state"
    }
    
    query_string = "&".join([f"{k}={v}" for k, v in params.items()])
    return {"auth_url": f"{auth_url}?{query_string}"}

@router.post("/providers/saml/acs")
async def saml_acs_endpoint(
    request: Request
):
    """SAML Assertion Consumer Service endpoint"""
    try:
        # Get SAML response from form data
        form_data = await request.form()
        saml_response = form_data.get("SAMLResponse")
        relay_state = form_data.get("RelayState")
        
        if not saml_response:
            raise HTTPException(status_code=400, detail="SAML response required")
        
        # Process SAML response
        # This would typically validate the SAML assertion and create a session
        
        return {"message": "SAML authentication processed", "relay_state": relay_state}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/session/info", response_model=SessionInfo)
async def get_session_info(
    current_user: str = Depends(get_current_user)
):
    """Get current session information"""
    try:
        # This would retrieve actual session information
        return SessionInfo(
            user_id=current_user,
            provider_id="example-provider",
            expires_at=datetime.utcnow(),
            last_activity=datetime.utcnow()
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/session/refresh")
async def refresh_session(
    refresh_token: str,
    current_user: str = Depends(get_current_user)
):
    """Refresh authentication session"""
    try:
        # This would use the refresh token to get new access tokens
        return {"message": "Session refreshed successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))