"""
IAM API Routes
REST API endpoints for Identity and Access Management
"""

from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

from ...iam.iam_integration import IAMSystem
from ...iam.mfa_system import MFAMethod
from ...iam.session_manager import SessionType
from ...iam.ueba_system import RiskLevel

logger = logging.getLogger(__name__)

# Initialize IAM system (in production, use dependency injection)
iam_config = {
    "mfa": {},
    "jit": {},
    "rbac": {},
    "ueba": {"learning_window_days": 30},
    "session": {
        "max_concurrent_sessions": 5,
        "session_timeout_minutes": 30,
        "absolute_timeout_hours": 8
    }
}

iam_system = IAMSystem(iam_config)
security = HTTPBearer()

router = APIRouter(prefix="/api/v1/iam", tags=["Identity and Access Management"])

# Request/Response Models
class AuthenticationRequest(BaseModel):
    username: str
    password: str
    device_fingerprint: Optional[str] = None
    session_type: str = "web"

class MFARequest(BaseModel):
    user_id: str
    mfa_method: str
    mfa_token: str
    challenge_id: Optional[str] = None

class AuthorizationRequest(BaseModel):
    resource_id: str
    resource_type: str
    action: str
    additional_context: Optional[Dict[str, Any]] = None

class JITAccessRequest(BaseModel):
    resource_id: str
    permissions: List[str]
    justification: str
    duration_hours: int = Field(default=8, ge=1, le=24)

class AuthenticationResponse(BaseModel):
    success: bool
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    session_token: Optional[str] = None
    requires_mfa: bool = False
    mfa_challenge_id: Optional[str] = None
    available_mfa_methods: Optional[List[str]] = None
    error_message: Optional[str] = None
    risk_score: float = 0.0

class AuthorizationResponse(BaseModel):
    allowed: bool
    reason: Optional[str] = None
    temporary_access: bool = False
    access_expires_at: Optional[datetime] = None
    risk_factors: Optional[List[str]] = None

# Helper functions
def get_client_ip(request: Request) -> str:
    """Extract client IP address"""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"

def get_user_agent(request: Request) -> str:
    """Extract user agent"""
    return request.headers.get("User-Agent", "unknown")

async def get_current_session(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Dependency to validate session token"""
    try:
        # Extract session_id and token from Authorization header
        # Format: Bearer session_id:session_token
        auth_parts = credentials.credentials.split(":")
        if len(auth_parts) != 2:
            raise HTTPException(status_code=401, detail="Invalid authorization format")
        
        session_id, session_token = auth_parts
        
        # Validate session
        session_info = iam_system.session_manager.validate_session(session_id, session_token)
        if not session_info:
            raise HTTPException(status_code=401, detail="Invalid or expired session")
        
        return session_info
        
    except Exception as e:
        logger.error(f"Session validation error: {str(e)}")
        raise HTTPException(status_code=401, detail="Authentication required")

# Authentication endpoints
@router.post("/authenticate", response_model=AuthenticationResponse)
async def authenticate(request: AuthenticationRequest, http_request: Request):
    """Authenticate user with username and password"""
    try:
        ip_address = get_client_ip(http_request)
        user_agent = get_user_agent(http_request)
        
        # Map session type string to enum
        session_type_map = {
            "web": SessionType.WEB,
            "api": SessionType.API,
            "mobile": SessionType.MOBILE,
            "desktop": SessionType.DESKTOP
        }
        session_type = session_type_map.get(request.session_type.lower(), SessionType.WEB)
        
        result = await iam_system.authenticate_user(
            username=request.username,
            password=request.password,
            ip_address=ip_address,
            user_agent=user_agent,
            device_fingerprint=request.device_fingerprint,
            session_type=session_type
        )
        
        # Convert MFAMethod enums to strings
        available_methods = None
        if result.available_mfa_methods:
            available_methods = [method.value for method in result.available_mfa_methods]
        
        return AuthenticationResponse(
            success=result.success,
            user_id=result.user_id,
            session_id=result.session_id,
            session_token=result.session_token,
            requires_mfa=result.requires_mfa,
            mfa_challenge_id=result.mfa_challenge_id,
            available_mfa_methods=available_methods,
            error_message=result.error_message,
            risk_score=result.risk_score
        )
        
    except Exception as e:
        logger.error(f"Authentication endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail="Authentication service error")

@router.post("/mfa/complete", response_model=AuthenticationResponse)
async def complete_mfa(request: MFARequest, http_request: Request):
    """Complete MFA authentication"""
    try:
        ip_address = get_client_ip(http_request)
        user_agent = get_user_agent(http_request)
        
        # Map MFA method string to enum
        mfa_method_map = {
            "totp": MFAMethod.TOTP,
            "sms": MFAMethod.SMS,
            "biometric": MFAMethod.BIOMETRIC,
            "backup_codes": MFAMethod.BACKUP_CODES
        }
        
        mfa_method = mfa_method_map.get(request.mfa_method.lower())
        if not mfa_method:
            raise HTTPException(status_code=400, detail="Invalid MFA method")
        
        result = await iam_system.complete_mfa_authentication(
            user_id=request.user_id,
            mfa_method=mfa_method,
            mfa_token=request.mfa_token,
            challenge_id=request.challenge_id,
            ip_address=ip_address,
            user_agent=user_agent,
            session_type=SessionType.WEB
        )
        
        return AuthenticationResponse(
            success=result.success,
            user_id=result.user_id,
            session_id=result.session_id,
            session_token=result.session_token,
            error_message=result.error_message,
            risk_score=result.risk_score
        )
        
    except Exception as e:
        logger.error(f"MFA completion endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail="MFA service error")

@router.post("/logout")
async def logout(session_info = Depends(get_current_session)):
    """Logout and terminate session"""
    try:
        success = iam_system.terminate_session(session_info.session_id, "user_logout")
        
        if success:
            return {"message": "Logged out successfully"}
        else:
            raise HTTPException(status_code=400, detail="Logout failed")
            
    except Exception as e:
        logger.error(f"Logout endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail="Logout service error")

# Authorization endpoints
@router.post("/authorize", response_model=AuthorizationResponse)
async def authorize(request: AuthorizationRequest, http_request: Request,
                   session_info = Depends(get_current_session)):
    """Authorize access to resource"""
    try:
        ip_address = get_client_ip(http_request)
        
        result = await iam_system.authorize_access(
            session_id=session_info.session_id,
            session_token="dummy",  # Token already validated in dependency
            resource_id=request.resource_id,
            resource_type=request.resource_type,
            action=request.action,
            ip_address=ip_address,
            additional_context=request.additional_context
        )
        
        return AuthorizationResponse(
            allowed=result.allowed,
            reason=result.reason,
            temporary_access=result.temporary_access,
            access_expires_at=result.access_expires_at,
            risk_factors=result.risk_factors
        )
        
    except Exception as e:
        logger.error(f"Authorization endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail="Authorization service error")

# JIT Access endpoints
@router.post("/jit/request")
async def request_jit_access(request: JITAccessRequest,
                           session_info = Depends(get_current_session)):
    """Request just-in-time access"""
    try:
        request_id = await iam_system.request_jit_access(
            session_id=session_info.session_id,
            session_token="dummy",  # Token already validated
            resource_id=request.resource_id,
            permissions=request.permissions,
            justification=request.justification,
            duration_hours=request.duration_hours
        )
        
        return {"request_id": request_id, "message": "Access request submitted"}
        
    except Exception as e:
        logger.error(f"JIT access request endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail="JIT access service error")

@router.get("/jit/pending")
async def get_pending_approvals(session_info = Depends(get_current_session)):
    """Get pending approval requests for current user"""
    try:
        # In production, check if user is an approver
        pending_requests = iam_system.jit_system.get_pending_approvals(session_info.user_id)
        
        return {
            "pending_requests": [
                {
                    "request_id": req.request_id,
                    "user_id": req.user_id,
                    "resource_id": req.resource_id,
                    "permissions": req.permissions,
                    "justification": req.justification,
                    "requested_at": req.requested_at.isoformat(),
                    "requested_duration": str(req.requested_duration)
                }
                for req in pending_requests
            ]
        }
        
    except Exception as e:
        logger.error(f"Pending approvals endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail="JIT access service error")

@router.post("/jit/approve/{request_id}")
async def approve_jit_request(request_id: str, comments: Optional[str] = None,
                            session_info = Depends(get_current_session)):
    """Approve JIT access request"""
    try:
        success = iam_system.jit_system.approve_request(
            request_id=request_id,
            approver_id=session_info.user_id,
            comments=comments
        )
        
        if success:
            return {"message": "Request approved successfully"}
        else:
            raise HTTPException(status_code=400, detail="Approval failed")
            
    except Exception as e:
        logger.error(f"JIT approval endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail="JIT access service error")

@router.post("/jit/deny/{request_id}")
async def deny_jit_request(request_id: str, reason: str,
                         session_info = Depends(get_current_session)):
    """Deny JIT access request"""
    try:
        success = iam_system.jit_system.deny_request(
            request_id=request_id,
            approver_id=session_info.user_id,
            reason=reason
        )
        
        if success:
            return {"message": "Request denied successfully"}
        else:
            raise HTTPException(status_code=400, detail="Denial failed")
            
    except Exception as e:
        logger.error(f"JIT denial endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail="JIT access service error")

# Security monitoring endpoints
@router.get("/security/alerts")
async def get_security_alerts(user_id: Optional[str] = None,
                            session_info = Depends(get_current_session)):
    """Get security alerts"""
    try:
        # Only allow users to see their own alerts unless they're admin
        if user_id and user_id != session_info.user_id:
            # In production, check admin permissions
            pass
        
        alerts = iam_system.get_security_alerts(user_id or session_info.user_id)
        
        return {
            "alerts": [
                {
                    "alert_id": alert.alert_id,
                    "anomaly_type": alert.anomaly_type.value,
                    "risk_level": alert.risk_level.value,
                    "confidence_score": alert.confidence_score,
                    "description": alert.description,
                    "detected_at": alert.detected_at.isoformat(),
                    "is_acknowledged": alert.is_acknowledged
                }
                for alert in alerts
            ]
        }
        
    except Exception as e:
        logger.error(f"Security alerts endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail="Security monitoring service error")

@router.post("/security/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(alert_id: str, session_info = Depends(get_current_session)):
    """Acknowledge security alert"""
    try:
        success = iam_system.ueba_system.acknowledge_alert(alert_id, session_info.user_id)
        
        if success:
            return {"message": "Alert acknowledged successfully"}
        else:
            raise HTTPException(status_code=404, detail="Alert not found")
            
    except Exception as e:
        logger.error(f"Alert acknowledgment endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail="Security monitoring service error")

# Session management endpoints
@router.get("/sessions")
async def get_user_sessions(session_info = Depends(get_current_session)):
    """Get user's active sessions"""
    try:
        sessions = iam_system.get_user_sessions(session_info.user_id)
        
        return {
            "sessions": [
                {
                    "session_id": session.session_id,
                    "session_type": session.session_type.value,
                    "created_at": session.created_at.isoformat(),
                    "last_activity": session.last_activity.isoformat(),
                    "expires_at": session.expires_at.isoformat(),
                    "ip_address": session.ip_address,
                    "user_agent": session.user_agent,
                    "is_current": session.session_id == session_info.session_id
                }
                for session in sessions
            ]
        }
        
    except Exception as e:
        logger.error(f"User sessions endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail="Session management service error")

@router.delete("/sessions/{session_id}")
async def terminate_session(session_id: str, session_info = Depends(get_current_session)):
    """Terminate a specific session"""
    try:
        # Verify session belongs to current user
        user_sessions = iam_system.get_user_sessions(session_info.user_id)
        session_ids = [s.session_id for s in user_sessions]
        
        if session_id not in session_ids:
            raise HTTPException(status_code=403, detail="Session does not belong to user")
        
        success = iam_system.terminate_session(session_id, "user_termination")
        
        if success:
            return {"message": "Session terminated successfully"}
        else:
            raise HTTPException(status_code=400, detail="Session termination failed")
            
    except Exception as e:
        logger.error(f"Session termination endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail="Session management service error")

# MFA setup endpoints
@router.post("/mfa/setup/totp")
async def setup_totp(session_info = Depends(get_current_session)):
    """Setup TOTP MFA for user"""
    try:
        secret, qr_code = iam_system.mfa_system.setup_totp(session_info.user_id)
        
        return {
            "secret": secret,
            "qr_code": qr_code,
            "message": "TOTP setup completed"
        }
        
    except Exception as e:
        logger.error(f"TOTP setup endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail="MFA setup service error")

@router.post("/mfa/setup/backup-codes")
async def generate_backup_codes(session_info = Depends(get_current_session)):
    """Generate backup codes for user"""
    try:
        codes = iam_system.mfa_system.generate_backup_codes(session_info.user_id)
        
        return {
            "backup_codes": codes,
            "message": "Backup codes generated"
        }
        
    except Exception as e:
        logger.error(f"Backup codes endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail="MFA setup service error")

@router.post("/mfa/challenge/sms")
async def initiate_sms_challenge(phone_number: str, session_info = Depends(get_current_session)):
    """Initiate SMS MFA challenge"""
    try:
        challenge_id = iam_system.mfa_system.initiate_sms_challenge(
            session_info.user_id, phone_number
        )
        
        return {
            "challenge_id": challenge_id,
            "message": "SMS challenge initiated"
        }
        
    except Exception as e:
        logger.error(f"SMS challenge endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail="MFA service error")

# System status endpoint
@router.get("/status")
async def get_system_status():
    """Get IAM system status"""
    try:
        status = iam_system.get_system_status()
        return status
        
    except Exception as e:
        logger.error(f"System status endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail="System status service error")

# Test endpoint for setting up demo users
@router.post("/test/setup-user/{user_id}")
async def setup_test_user(user_id: str, phone_number: Optional[str] = None):
    """Setup test user (development only)"""
    try:
        result = iam_system.setup_test_user(user_id, phone_number)
        return result
        
    except Exception as e:
        logger.error(f"Test user setup error: {str(e)}")
        raise HTTPException(status_code=500, detail="Test user setup failed")