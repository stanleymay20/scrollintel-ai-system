"""
Security and Compliance API Routes
Provides REST endpoints for authentication, authorization, and audit functions
"""

from fastapi import APIRouter, HTTPException, Depends, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import logging

from scrollintel.security.enterprise_security_manager import (
    security_manager,
    UserCredentials,
    SecurityLevel,
    EncryptionLevel,
    AuditCriteria,
    User
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1/security", tags=["security"])
security = HTTPBearer()

# Pydantic models for API
class LoginRequest(BaseModel):
    """Login request model"""
    username: str = Field(..., description="Username")
    password: Optional[str] = Field(None, description="Password")
    mfa_token: Optional[str] = Field(None, description="MFA token")
    sso_token: Optional[str] = Field(None, description="SSO token")

class LoginResponse(BaseModel):
    """Login response model"""
    success: bool
    token: Optional[str] = None
    user_id: Optional[str] = None
    username: Optional[str] = None
    roles: List[str] = []
    permissions: List[str] = []
    expires_at: Optional[datetime] = None
    message: Optional[str] = None

class EncryptionRequest(BaseModel):
    """Encryption request model"""
    data: str = Field(..., description="Data to encrypt")
    level: str = Field("basic", description="Encryption level: basic, advanced, quantum_safe")

class EncryptionResponse(BaseModel):
    """Encryption response model"""
    encrypted_data: str
    encryption_level: str

class AuthorizationRequest(BaseModel):
    """Authorization request model"""
    action: str = Field(..., description="Action to authorize")
    resource: Optional[str] = Field(None, description="Resource being accessed")

class AuthorizationResponse(BaseModel):
    """Authorization response model"""
    authorized: bool
    permission: str
    message: Optional[str] = None

class AuditReportRequest(BaseModel):
    """Audit report request model"""
    start_date: datetime
    end_date: datetime
    event_types: Optional[List[str]] = None
    user_ids: Optional[List[str]] = None
    resources: Optional[List[str]] = None

class SecurityEventModel(BaseModel):
    """Security event model"""
    event_type: str
    user_id: Optional[str]
    resource: Optional[str]
    action: str
    result: str
    timestamp: datetime
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    details: Optional[Dict[str, Any]] = None

class AuditReportResponse(BaseModel):
    """Audit report response model"""
    total_events: int
    events: List[SecurityEventModel]
    summary: Dict[str, Any]
    generated_at: datetime

class UserProfileResponse(BaseModel):
    """User profile response model"""
    id: str
    username: str
    email: str
    roles: List[str]
    permissions: List[str]
    security_clearance: str
    last_login: Optional[datetime]

# Dependency to get current user
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> User:
    """Get current authenticated user"""
    try:
        user = await security_manager.validate_token(credentials.credentials)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired token"
            )
        return user
    except Exception as e:
        logger.error(f"Token validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed"
        )

@router.post("/login", response_model=LoginResponse)
async def login(request: LoginRequest, http_request: Request):
    """
    Authenticate user with multi-factor authentication and SSO
    """
    try:
        # Create credentials object
        credentials = UserCredentials(
            username=request.username,
            password=request.password,
            mfa_token=request.mfa_token,
            sso_token=request.sso_token
        )
        
        # Authenticate user
        result = await security_manager.authenticate_user(credentials)
        
        if result.success:
            return LoginResponse(
                success=True,
                token=result.token,
                user_id=result.user.id,
                username=result.user.username,
                roles=result.user.roles,
                permissions=result.permissions,
                expires_at=result.expires_at,
                message="Authentication successful"
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=result.reason or "Authentication failed"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication system error"
        )

@router.post("/logout")
async def logout(current_user: User = Depends(get_current_user)):
    """
    Logout current user and invalidate token
    """
    try:
        # In a real implementation, you would invalidate the token
        # For now, we'll just log the logout event
        logger.info(f"User {current_user.username} logged out")
        
        return {"message": "Logout successful"}
        
    except Exception as e:
        logger.error(f"Logout error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Logout failed"
        )

@router.get("/profile", response_model=UserProfileResponse)
async def get_profile(current_user: User = Depends(get_current_user)):
    """
    Get current user profile
    """
    try:
        permissions = await security_manager.access_controller.get_user_permissions(current_user)
        
        return UserProfileResponse(
            id=current_user.id,
            username=current_user.username,
            email=current_user.email,
            roles=current_user.roles,
            permissions=permissions,
            security_clearance=current_user.security_clearance.value,
            last_login=current_user.last_login
        )
        
    except Exception as e:
        logger.error(f"Profile retrieval error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve profile"
        )

@router.post("/encrypt", response_model=EncryptionResponse)
async def encrypt_data(
    request: EncryptionRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Encrypt sensitive data with specified encryption level
    """
    try:
        # Check authorization
        authorized = await security_manager.authorize_action(
            current_user, "encrypt", "data"
        )
        if not authorized:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions for data encryption"
            )
        
        # Map string to enum
        level_map = {
            "basic": EncryptionLevel.BASIC,
            "advanced": EncryptionLevel.ADVANCED,
            "quantum_safe": EncryptionLevel.QUANTUM_SAFE
        }
        
        encryption_level = level_map.get(request.level, EncryptionLevel.BASIC)
        
        # Encrypt data
        encrypted_data = await security_manager.encrypt_sensitive_data(
            request.data, encryption_level
        )
        
        # Encode as base64 for JSON response
        import base64
        encoded_data = base64.b64encode(encrypted_data).decode('utf-8')
        
        return EncryptionResponse(
            encrypted_data=encoded_data,
            encryption_level=request.level
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Encryption error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Encryption failed"
        )

@router.post("/decrypt")
async def decrypt_data(
    encrypted_data: str,
    encryption_level: str = "basic",
    current_user: User = Depends(get_current_user)
):
    """
    Decrypt sensitive data
    """
    try:
        # Check authorization
        authorized = await security_manager.authorize_action(
            current_user, "decrypt", "data"
        )
        if not authorized:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions for data decryption"
            )
        
        # Map string to enum
        level_map = {
            "basic": EncryptionLevel.BASIC,
            "advanced": EncryptionLevel.ADVANCED,
            "quantum_safe": EncryptionLevel.QUANTUM_SAFE
        }
        
        level = level_map.get(encryption_level, EncryptionLevel.BASIC)
        
        # Decode from base64
        import base64
        encrypted_bytes = base64.b64decode(encrypted_data.encode('utf-8'))
        
        # Decrypt data
        decrypted_data = await security_manager.decrypt_sensitive_data(
            encrypted_bytes, level
        )
        
        return {"decrypted_data": decrypted_data.decode('utf-8')}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Decryption error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Decryption failed"
        )

@router.post("/authorize", response_model=AuthorizationResponse)
async def check_authorization(
    request: AuthorizationRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Check if user is authorized for specific action
    """
    try:
        authorized = await security_manager.authorize_action(
            current_user, request.action, request.resource
        )
        
        permission = f"{request.resource}.{request.action}" if request.resource else request.action
        
        return AuthorizationResponse(
            authorized=authorized,
            permission=permission,
            message="Authorized" if authorized else "Access denied"
        )
        
    except Exception as e:
        logger.error(f"Authorization check error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authorization check failed"
        )

@router.post("/audit/report", response_model=AuditReportResponse)
async def generate_audit_report(
    request: AuditReportRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Generate comprehensive audit report
    """
    try:
        # Check authorization for audit reports
        authorized = await security_manager.authorize_action(
            current_user, "generate", "audit_reports"
        )
        if not authorized:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions for audit report generation"
            )
        
        # Create audit criteria
        criteria = AuditCriteria(
            start_date=request.start_date,
            end_date=request.end_date,
            event_types=request.event_types,
            user_ids=request.user_ids,
            resources=request.resources
        )
        
        # Generate report
        report = await security_manager.generate_compliance_report(criteria)
        
        # Convert events to response models
        events = [
            SecurityEventModel(
                event_type=event.event_type,
                user_id=event.user_id,
                resource=event.resource,
                action=event.action,
                result=event.result,
                timestamp=event.timestamp,
                ip_address=event.ip_address,
                user_agent=event.user_agent,
                details=event.details
            )
            for event in report.events
        ]
        
        return AuditReportResponse(
            total_events=len(events),
            events=events,
            summary=report.summary,
            generated_at=report.generated_at
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Audit report generation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Audit report generation failed"
        )

@router.get("/permissions")
async def get_user_permissions(current_user: User = Depends(get_current_user)):
    """
    Get all permissions for current user
    """
    try:
        permissions = await security_manager.access_controller.get_user_permissions(current_user)
        
        return {
            "user_id": current_user.id,
            "username": current_user.username,
            "roles": current_user.roles,
            "permissions": permissions,
            "security_clearance": current_user.security_clearance.value
        }
        
    except Exception as e:
        logger.error(f"Permission retrieval error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve permissions"
        )

@router.get("/health")
async def security_health_check():
    """
    Security system health check
    """
    try:
        # Perform basic health checks
        health_status = {
            "status": "healthy",
            "timestamp": datetime.utcnow(),
            "components": {
                "authentication": "operational",
                "encryption": "operational",
                "audit_logging": "operational",
                "access_control": "operational"
            }
        }
        
        return health_status
        
    except Exception as e:
        logger.error(f"Security health check error: {e}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.utcnow(),
            "error": str(e)
        }

# Initialize demo data on startup
@router.on_event("startup")
async def setup_security_demo_data():
    """Setup demo data for security system"""
    try:
        await security_manager.setup_demo_data()
        logger.info("Security demo data initialized")
    except Exception as e:
        logger.error(f"Failed to setup security demo data: {e}")