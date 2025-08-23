"""
Security and Compliance API Routes for Agent Steering System
Provides enterprise-grade security management endpoints
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, Depends, HTTPException, Request, Response, status, Query
from fastapi.security import HTTPBearer
from sqlalchemy.orm import Session
from pydantic import BaseModel
import logging

from ...core.database import get_db
from ...models.security_compliance_models import (
    User, Role, Permission, UserRole, UserSession, AuditLog,
    UserCreate, UserUpdate, UserResponse, AuthenticationRequest, AuthenticationResponse,
    RoleCreate, RoleResponse, PermissionCreate, PermissionResponse,
    AuditLogResponse, ComplianceReport, SecurityMetrics,
    SecurityLevel, PermissionType, AuditEventType, ComplianceFramework
)
from ...security.enterprise_security_framework import (
    EnterpriseSecurityFramework, SecurityException, AuthenticationException, AuthorizationException
)
from ...api.middleware.security_middleware import (
    JWTBearer, require_permission, require_security_level
)

logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter(prefix="/security", tags=["Security & Compliance"])

# Security framework instance (should be injected via dependency)
def get_security_framework() -> EnterpriseSecurityFramework:
    # This should be properly injected in production
    from ...core.config import get_security_config
    config = get_security_config()
    return EnterpriseSecurityFramework(config)

# Authentication endpoints
@router.post("/auth/login", response_model=AuthenticationResponse)
async def login(
    auth_request: AuthenticationRequest,
    request: Request,
    db: Session = Depends(get_db),
    security: EnterpriseSecurityFramework = Depends(get_security_framework)
):
    """Authenticate user with multiple methods (password, MFA, SSO)"""
    
    try:
        client_ip = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("user-agent", "unknown")
        
        user, session_token = security.authenticate_user(
            db=db,
            username=auth_request.username,
            password=auth_request.password,
            mfa_code=auth_request.mfa_code,
            sso_token=auth_request.sso_token,
            ip_address=client_ip,
            user_agent=user_agent
        )
        
        # Get user permissions
        permissions = security.get_user_permissions(db, user.id)
        
        # Get user roles
        user_roles = db.query(Role).join(UserRole).filter(
            UserRole.user_id == user.id,
            UserRole.valid_from <= datetime.utcnow(),
            (UserRole.valid_until.is_(None) | (UserRole.valid_until > datetime.utcnow()))
        ).all()
        
        user_response = UserResponse(
            id=user.id,
            username=user.username,
            email=user.email,
            is_active=user.is_active,
            is_verified=user.is_verified,
            security_level=SecurityLevel(user.security_level),
            mfa_enabled=user.mfa_enabled,
            last_login=user.last_login,
            created_at=user.created_at,
            roles=[role.name for role in user_roles]
        )
        
        return AuthenticationResponse(
            access_token=session_token,
            refresh_token="refresh_token_placeholder",  # Implement refresh token logic
            expires_in=security.config.session_timeout_minutes * 60,
            user=user_response,
            permissions=permissions
        )
        
    except AuthenticationException as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication service error"
        )

@router.post("/auth/logout")
async def logout(
    request: Request,
    db: Session = Depends(get_db),
    current_user: Dict = Depends(JWTBearer(get_security_framework()))
):
    """Logout user and invalidate session"""
    
    try:
        # Invalidate session
        session = db.query(UserSession).filter(
            UserSession.id == current_user['session_id']
        ).first()
        
        if session:
            session.is_active = False
            db.commit()
        
        # Log logout event
        security = get_security_framework()
        security._log_audit_event(
            db=db,
            user_id=current_user['user_id'],
            event_type=AuditEventType.AUTHENTICATION,
            event_category="logout",
            event_description="User logged out",
            action="logout",
            success=True
        )
        
        return {"message": "Logged out successfully"}
        
    except Exception as e:
        logger.error(f"Logout error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Logout service error"
        )

@router.post("/auth/mfa/setup")
@require_permission("security", PermissionType.WRITE)
async def setup_mfa(
    request: Request,
    db: Session = Depends(get_db),
    current_user: Dict = Depends(JWTBearer(get_security_framework())),
    security: EnterpriseSecurityFramework = Depends(get_security_framework)
):
    """Setup multi-factor authentication for user"""
    
    try:
        user = db.query(User).filter(User.id == current_user['user_id']).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Generate MFA secret
        mfa_secret = security.generate_mfa_secret()
        qr_url = security.generate_mfa_qr_url(user.email, mfa_secret)
        backup_codes = security.generate_backup_codes()
        
        # Store encrypted secret and backup codes
        user.mfa_secret = security.encrypt_sensitive_field(mfa_secret)
        user.backup_codes = [security.encrypt_sensitive_field(code) for code in backup_codes]
        
        db.commit()
        
        # Log MFA setup
        security._log_audit_event(
            db=db,
            user_id=user.id,
            event_type=AuditEventType.CONFIGURATION_CHANGE,
            event_category="mfa_setup",
            event_description="MFA setup initiated",
            action="setup_mfa",
            success=True
        )
        
        return {
            "qr_url": qr_url,
            "backup_codes": backup_codes,
            "message": "MFA setup initiated. Scan QR code with authenticator app."
        }
        
    except Exception as e:
        logger.error(f"MFA setup error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="MFA setup service error"
        )

@router.post("/auth/mfa/verify")
@require_permission("security", PermissionType.WRITE)
async def verify_mfa_setup(
    mfa_code: str,
    request: Request,
    db: Session = Depends(get_db),
    current_user: Dict = Depends(JWTBearer(get_security_framework())),
    security: EnterpriseSecurityFramework = Depends(get_security_framework)
):
    """Verify and enable MFA for user"""
    
    try:
        user = db.query(User).filter(User.id == current_user['user_id']).first()
        if not user or not user.mfa_secret:
            raise HTTPException(status_code=404, detail="MFA not set up")
        
        # Decrypt and verify MFA code
        decrypted_secret = security.decrypt_sensitive_field(user.mfa_secret)
        
        if not security.verify_mfa_code(decrypted_secret, mfa_code):
            raise HTTPException(status_code=400, detail="Invalid MFA code")
        
        # Enable MFA
        user.mfa_enabled = True
        db.commit()
        
        # Log MFA verification
        security._log_audit_event(
            db=db,
            user_id=user.id,
            event_type=AuditEventType.CONFIGURATION_CHANGE,
            event_category="mfa_enabled",
            event_description="MFA enabled for user",
            action="enable_mfa",
            success=True
        )
        
        return {"message": "MFA enabled successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"MFA verification error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="MFA verification service error"
        )

# User Management
@router.post("/users", response_model=UserResponse)
@require_permission("users", PermissionType.WRITE)
@require_security_level(SecurityLevel.CONFIDENTIAL)
async def create_user(
    user_data: UserCreate,
    request: Request,
    db: Session = Depends(get_db),
    current_user: Dict = Depends(JWTBearer(get_security_framework())),
    security: EnterpriseSecurityFramework = Depends(get_security_framework)
):
    """Create new user with security validation"""
    
    try:
        # Check if user already exists
        existing_user = db.query(User).filter(
            (User.username == user_data.username) | (User.email == user_data.email)
        ).first()
        
        if existing_user:
            raise HTTPException(status_code=400, detail="User already exists")
        
        # Create user
        password_hash = None
        if user_data.password:
            password_hash = security.hash_password(user_data.password)
        
        user = User(
            username=user_data.username,
            email=user_data.email,
            password_hash=password_hash,
            security_level=user_data.security_level.value,
            created_by=current_user['user_id']
        )
        
        db.add(user)
        db.flush()  # Get user ID
        
        # Assign roles
        for role_name in user_data.roles:
            role = db.query(Role).filter(Role.name == role_name).first()
            if role:
                user_role = UserRole(
                    user_id=user.id,
                    role_id=role.id,
                    assigned_by=current_user['user_id']
                )
                db.add(user_role)
        
        db.commit()
        
        # Log user creation
        security._log_audit_event(
            db=db,
            user_id=current_user['user_id'],
            event_type=AuditEventType.CONFIGURATION_CHANGE,
            event_category="user_created",
            event_description=f"User created: {user.username}",
            resource_type="user",
            resource_id=user.id,
            action="create_user",
            success=True
        )
        
        return UserResponse(
            id=user.id,
            username=user.username,
            email=user.email,
            is_active=user.is_active,
            is_verified=user.is_verified,
            security_level=SecurityLevel(user.security_level),
            mfa_enabled=user.mfa_enabled,
            last_login=user.last_login,
            created_at=user.created_at,
            roles=user_data.roles
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"User creation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="User creation service error"
        )

@router.get("/users", response_model=List[UserResponse])
@require_permission("users", PermissionType.READ)
async def list_users(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    security_level: Optional[SecurityLevel] = None,
    db: Session = Depends(get_db),
    current_user: Dict = Depends(JWTBearer(get_security_framework()))
):
    """List users with filtering"""
    
    try:
        query = db.query(User)
        
        if security_level:
            query = query.filter(User.security_level == security_level.value)
        
        users = query.offset(skip).limit(limit).all()
        
        result = []
        for user in users:
            # Get user roles
            user_roles = db.query(Role).join(UserRole).filter(
                UserRole.user_id == user.id,
                UserRole.valid_from <= datetime.utcnow(),
                (UserRole.valid_until.is_(None) | (UserRole.valid_until > datetime.utcnow()))
            ).all()
            
            result.append(UserResponse(
                id=user.id,
                username=user.username,
                email=user.email,
                is_active=user.is_active,
                is_verified=user.is_verified,
                security_level=SecurityLevel(user.security_level),
                mfa_enabled=user.mfa_enabled,
                last_login=user.last_login,
                created_at=user.created_at,
                roles=[role.name for role in user_roles]
            ))
        
        return result
        
    except Exception as e:
        logger.error(f"User listing error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="User listing service error"
        )

# Role Management
@router.post("/roles", response_model=RoleResponse)
@require_permission("roles", PermissionType.WRITE)
@require_security_level(SecurityLevel.SECRET)
async def create_role(
    role_data: RoleCreate,
    request: Request,
    db: Session = Depends(get_db),
    current_user: Dict = Depends(JWTBearer(get_security_framework())),
    security: EnterpriseSecurityFramework = Depends(get_security_framework)
):
    """Create new role with permissions"""
    
    try:
        # Check if role already exists
        existing_role = db.query(Role).filter(Role.name == role_data.name).first()
        if existing_role:
            raise HTTPException(status_code=400, detail="Role already exists")
        
        # Create role
        role = Role(
            name=role_data.name,
            description=role_data.description,
            security_level=role_data.security_level.value
        )
        
        db.add(role)
        db.flush()  # Get role ID
        
        # Assign permissions
        for permission_name in role_data.permissions:
            permission = db.query(Permission).filter(Permission.name == permission_name).first()
            if permission:
                from ...models.security_compliance_models import RolePermission
                role_permission = RolePermission(
                    role_id=role.id,
                    permission_id=permission.id,
                    granted_by=current_user['user_id']
                )
                db.add(role_permission)
        
        db.commit()
        
        # Log role creation
        security._log_audit_event(
            db=db,
            user_id=current_user['user_id'],
            event_type=AuditEventType.CONFIGURATION_CHANGE,
            event_category="role_created",
            event_description=f"Role created: {role.name}",
            resource_type="role",
            resource_id=role.id,
            action="create_role",
            success=True
        )
        
        return RoleResponse(
            id=role.id,
            name=role.name,
            description=role.description,
            security_level=SecurityLevel(role.security_level),
            is_system_role=role.is_system_role,
            created_at=role.created_at,
            permissions=role_data.permissions
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Role creation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Role creation service error"
        )

# Audit and Compliance
@router.get("/audit/logs", response_model=List[AuditLogResponse])
@require_permission("audit", PermissionType.READ)
@require_security_level(SecurityLevel.SECRET)
async def get_audit_logs(
    user_id: Optional[str] = None,
    event_type: Optional[AuditEventType] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(get_db),
    current_user: Dict = Depends(JWTBearer(get_security_framework())),
    security: EnterpriseSecurityFramework = Depends(get_security_framework)
):
    """Retrieve audit logs with decryption"""
    
    try:
        logs = security.get_audit_logs(
            db=db,
            user_id=user_id,
            event_type=event_type,
            start_date=start_date,
            end_date=end_date,
            limit=limit
        )
        
        result = []
        for log in logs:
            result.append(AuditLogResponse(
                id=log.id,
                user_id=log.user_id,
                event_type=log.event_type,
                event_category=log.event_category,
                event_description=log.event_description,
                resource_type=log.resource_type,
                resource_id=log.resource_id,
                action=log.action,
                success=log.success,
                timestamp=log.timestamp,
                ip_address=log.ip_address,
                metadata=log.event_metadata
            ))
        
        return result
        
    except Exception as e:
        logger.error(f"Audit log retrieval error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Audit log service error"
        )

@router.get("/compliance/report", response_model=Dict[str, Any])
@require_permission("compliance", PermissionType.READ)
@require_security_level(SecurityLevel.SECRET)
async def generate_compliance_report(
    framework: ComplianceFramework,
    start_date: datetime,
    end_date: datetime,
    db: Session = Depends(get_db),
    current_user: Dict = Depends(JWTBearer(get_security_framework())),
    security: EnterpriseSecurityFramework = Depends(get_security_framework)
):
    """Generate compliance report for specified framework"""
    
    try:
        report = security.generate_compliance_report(
            db=db,
            framework=framework,
            start_date=start_date,
            end_date=end_date
        )
        
        # Log compliance report generation
        security._log_audit_event(
            db=db,
            user_id=current_user['user_id'],
            event_type=AuditEventType.COMPLIANCE_VIOLATION,
            event_category="compliance_report",
            event_description=f"Compliance report generated: {framework.value}",
            action="generate_report",
            success=True,
            compliance_frameworks=[framework]
        )
        
        return report
        
    except Exception as e:
        logger.error(f"Compliance report error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Compliance report service error"
        )

@router.get("/metrics", response_model=SecurityMetrics)
@require_permission("security", PermissionType.READ)
async def get_security_metrics(
    db: Session = Depends(get_db),
    current_user: Dict = Depends(JWTBearer(get_security_framework()))
):
    """Get current security metrics"""
    
    try:
        # Calculate metrics
        active_users = db.query(User).filter(User.is_active == True).count()
        active_sessions = db.query(UserSession).filter(
            UserSession.is_active == True,
            UserSession.expires_at > datetime.utcnow()
        ).count()
        
        failed_attempts_today = db.query(AuditLog).filter(
            AuditLog.event_type == AuditEventType.AUTHENTICATION.value,
            AuditLog.success == False,
            AuditLog.timestamp >= datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        ).count()
        
        security_incidents_today = db.query(AuditLog).filter(
            AuditLog.event_type == AuditEventType.SECURITY_INCIDENT.value,
            AuditLog.timestamp >= datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        ).count()
        
        audit_events_today = db.query(AuditLog).filter(
            AuditLog.timestamp >= datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        ).count()
        
        # Calculate compliance score (simplified)
        total_events_week = db.query(AuditLog).filter(
            AuditLog.timestamp >= datetime.utcnow() - timedelta(days=7)
        ).count()
        
        successful_events_week = db.query(AuditLog).filter(
            AuditLog.timestamp >= datetime.utcnow() - timedelta(days=7),
            AuditLog.success == True
        ).count()
        
        compliance_score = (successful_events_week / total_events_week * 100) if total_events_week > 0 else 100
        
        return SecurityMetrics(
            active_users=active_users,
            active_sessions=active_sessions,
            failed_login_attempts=failed_attempts_today,
            security_incidents=security_incidents_today,
            compliance_score=compliance_score,
            encryption_coverage=100.0,  # Assuming full encryption
            audit_events_today=audit_events_today
        )
        
    except Exception as e:
        logger.error(f"Security metrics error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Security metrics service error"
        )

@router.post("/encryption/rotate-keys")
@require_permission("security", PermissionType.ADMIN)
@require_security_level(SecurityLevel.TOP_SECRET)
async def rotate_encryption_keys(
    request: Request,
    db: Session = Depends(get_db),
    current_user: Dict = Depends(JWTBearer(get_security_framework())),
    security: EnterpriseSecurityFramework = Depends(get_security_framework)
):
    """Rotate encryption keys (admin only)"""
    
    try:
        security.rotate_encryption_keys(db)
        
        # Log key rotation
        security._log_audit_event(
            db=db,
            user_id=current_user['user_id'],
            event_type=AuditEventType.CONFIGURATION_CHANGE,
            event_category="key_rotation",
            event_description="Encryption keys rotated",
            action="rotate_keys",
            success=True
        )
        
        return {"message": "Encryption keys rotated successfully"}
        
    except Exception as e:
        logger.error(f"Key rotation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Key rotation service error"
        )