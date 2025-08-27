"""
Security and Authentication Layer with JWT Tokens and RBAC
"""
import jwt
import bcrypt
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Set
from dataclasses import dataclass
from enum import Enum
import logging
from functools import wraps
from flask import request, jsonify, g
import secrets
import hashlib

logger = logging.getLogger(__name__)

class Permission(Enum):
    """System permissions"""
    READ_DATA_PRODUCTS = "read_data_products"
    WRITE_DATA_PRODUCTS = "write_data_products"
    DELETE_DATA_PRODUCTS = "delete_data_products"
    MANAGE_USERS = "manage_users"
    MANAGE_ROLES = "manage_roles"
    EXECUTE_AGENTS = "execute_agents"
    VIEW_ANALYTICS = "view_analytics"
    MANAGE_SYSTEM = "manage_system"
    AUDIT_ACCESS = "audit_access"

class Role(Enum):
    """System roles"""
    ADMIN = "admin"
    DATA_ENGINEER = "data_engineer"
    DATA_SCIENTIST = "data_scientist"
    ANALYST = "analyst"
    VIEWER = "viewer"
    AGENT_OPERATOR = "agent_operator"

@dataclass
class User:
    """User model"""
    id: str
    username: str
    email: str
    password_hash: str
    roles: Set[Role]
    is_active: bool = True
    created_at: datetime = None
    last_login: datetime = None

@dataclass
class JWTConfig:
    """JWT configuration"""
    secret_key: str
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7

class RBACManager:
    """Role-Based Access Control Manager"""
    
    # Role-Permission mapping
    ROLE_PERMISSIONS = {
        Role.ADMIN: {
            Permission.READ_DATA_PRODUCTS,
            Permission.WRITE_DATA_PRODUCTS,
            Permission.DELETE_DATA_PRODUCTS,
            Permission.MANAGE_USERS,
            Permission.MANAGE_ROLES,
            Permission.EXECUTE_AGENTS,
            Permission.VIEW_ANALYTICS,
            Permission.MANAGE_SYSTEM,
            Permission.AUDIT_ACCESS
        },
        Role.DATA_ENGINEER: {
            Permission.READ_DATA_PRODUCTS,
            Permission.WRITE_DATA_PRODUCTS,
            Permission.DELETE_DATA_PRODUCTS,
            Permission.EXECUTE_AGENTS,
            Permission.VIEW_ANALYTICS
        },
        Role.DATA_SCIENTIST: {
            Permission.READ_DATA_PRODUCTS,
            Permission.WRITE_DATA_PRODUCTS,
            Permission.EXECUTE_AGENTS,
            Permission.VIEW_ANALYTICS
        },
        Role.ANALYST: {
            Permission.READ_DATA_PRODUCTS,
            Permission.VIEW_ANALYTICS
        },
        Role.VIEWER: {
            Permission.READ_DATA_PRODUCTS
        },
        Role.AGENT_OPERATOR: {
            Permission.EXECUTE_AGENTS,
            Permission.VIEW_ANALYTICS
        }
    }
    
    @classmethod
    def get_permissions(cls, roles: Set[Role]) -> Set[Permission]:
        """Get all permissions for given roles"""
        permissions = set()
        for role in roles:
            permissions.update(cls.ROLE_PERMISSIONS.get(role, set()))
        return permissions
    
    @classmethod
    def has_permission(cls, user_roles: Set[Role], required_permission: Permission) -> bool:
        """Check if user roles have required permission"""
        user_permissions = cls.get_permissions(user_roles)
        return required_permission in user_permissions

class AuthenticationManager:
    """Manages authentication with JWT tokens"""
    
    def __init__(self, jwt_config: JWTConfig):
        self.jwt_config = jwt_config
        self.users: Dict[str, User] = {}  # In production, use database
        self.refresh_tokens: Set[str] = set()  # In production, use Redis
    
    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt"""
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')
    
    def verify_password(self, password: str, password_hash: str) -> bool:
        """Verify password against hash"""
        return bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8'))
    
    def create_user(self, username: str, email: str, password: str, roles: Set[Role]) -> User:
        """Create new user"""
        user_id = hashlib.sha256(f"{username}{email}".encode()).hexdigest()[:16]
        password_hash = self.hash_password(password)
        
        user = User(
            id=user_id,
            username=username,
            email=email,
            password_hash=password_hash,
            roles=roles,
            created_at=datetime.utcnow()
        )
        
        self.users[user_id] = user
        logger.info(f"Created user {username} with roles {[r.value for r in roles]}")
        return user
    
    def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """Authenticate user with username/password"""
        user = next((u for u in self.users.values() if u.username == username), None)
        
        if user and user.is_active and self.verify_password(password, user.password_hash):
            user.last_login = datetime.utcnow()
            logger.info(f"User {username} authenticated successfully")
            return user
        
        logger.warning(f"Authentication failed for user {username}")
        return None
    
    def generate_tokens(self, user: User) -> Dict[str, str]:
        """Generate access and refresh tokens"""
        now = datetime.utcnow()
        
        # Access token payload
        access_payload = {
            "user_id": user.id,
            "username": user.username,
            "roles": [role.value for role in user.roles],
            "iat": now,
            "exp": now + timedelta(minutes=self.jwt_config.access_token_expire_minutes),
            "type": "access"
        }
        
        # Refresh token payload
        refresh_payload = {
            "user_id": user.id,
            "iat": now,
            "exp": now + timedelta(days=self.jwt_config.refresh_token_expire_days),
            "type": "refresh",
            "jti": secrets.token_urlsafe(32)  # JWT ID for revocation
        }
        
        access_token = jwt.encode(
            access_payload,
            self.jwt_config.secret_key,
            algorithm=self.jwt_config.algorithm
        )
        
        refresh_token = jwt.encode(
            refresh_payload,
            self.jwt_config.secret_key,
            algorithm=self.jwt_config.algorithm
        )
        
        # Store refresh token for validation
        self.refresh_tokens.add(refresh_payload["jti"])
        
        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer",
            "expires_in": self.jwt_config.access_token_expire_minutes * 60
        }
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(
                token,
                self.jwt_config.secret_key,
                algorithms=[self.jwt_config.algorithm]
            )
            
            # Check if refresh token is still valid
            if payload.get("type") == "refresh":
                jti = payload.get("jti")
                if jti not in self.refresh_tokens:
                    logger.warning("Refresh token has been revoked")
                    return None
            
            return payload
            
        except jwt.ExpiredSignatureError:
            logger.warning("Token has expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token: {e}")
            return None
    
    def refresh_access_token(self, refresh_token: str) -> Optional[Dict[str, str]]:
        """Generate new access token using refresh token"""
        payload = self.verify_token(refresh_token)
        
        if not payload or payload.get("type") != "refresh":
            return None
        
        user_id = payload.get("user_id")
        user = self.users.get(user_id)
        
        if not user or not user.is_active:
            return None
        
        # Generate new access token
        return self.generate_tokens(user)
    
    def revoke_refresh_token(self, refresh_token: str) -> bool:
        """Revoke refresh token"""
        payload = self.verify_token(refresh_token)
        
        if payload and payload.get("type") == "refresh":
            jti = payload.get("jti")
            if jti in self.refresh_tokens:
                self.refresh_tokens.remove(jti)
                logger.info(f"Revoked refresh token for user {payload.get('user_id')}")
                return True
        
        return False

class SecurityMiddleware:
    """Security middleware for request authentication and authorization"""
    
    def __init__(self, auth_manager: AuthenticationManager):
        self.auth_manager = auth_manager
    
    def require_auth(self, required_permission: Optional[Permission] = None):
        """Decorator to require authentication and optional permission"""
        def decorator(f):
            @wraps(f)
            def decorated_function(*args, **kwargs):
                # Get token from Authorization header
                auth_header = request.headers.get('Authorization')
                if not auth_header or not auth_header.startswith('Bearer '):
                    return jsonify({"error": "Missing or invalid authorization header"}), 401
                
                token = auth_header.split(' ')[1]
                payload = self.auth_manager.verify_token(token)
                
                if not payload or payload.get("type") != "access":
                    return jsonify({"error": "Invalid or expired token"}), 401
                
                # Get user
                user_id = payload.get("user_id")
                user = self.auth_manager.users.get(user_id)
                
                if not user or not user.is_active:
                    return jsonify({"error": "User not found or inactive"}), 401
                
                # Check permission if required
                if required_permission:
                    user_roles = {Role(role) for role in payload.get("roles", [])}
                    if not RBACManager.has_permission(user_roles, required_permission):
                        return jsonify({"error": "Insufficient permissions"}), 403
                
                # Store user info in request context
                g.current_user = user
                g.user_permissions = RBACManager.get_permissions(user.roles)
                
                return f(*args, **kwargs)
            
            return decorated_function
        return decorator
    
    def require_permission(self, permission: Permission):
        """Decorator to require specific permission"""
        return self.require_auth(permission)

# Global authentication manager
_auth_manager: Optional[AuthenticationManager] = None
_security_middleware: Optional[SecurityMiddleware] = None

def get_auth_manager() -> AuthenticationManager:
    """Get global authentication manager"""
    global _auth_manager
    
    if _auth_manager is None:
        # In production, load from config
        jwt_config = JWTConfig(
            secret_key=secrets.token_urlsafe(32),  # Generate secure key
            access_token_expire_minutes=30,
            refresh_token_expire_days=7
        )
        _auth_manager = AuthenticationManager(jwt_config)
        
        # Create default admin user
        _auth_manager.create_user(
            username="admin",
            email="admin@scrollintel.com",
            password=os.getenv("ADMIN_PASSWORD", ""),  # Must be set via environment
            roles={Role.ADMIN}
        )
    
    return _auth_manager

def get_security_middleware() -> SecurityMiddleware:
    """Get global security middleware"""
    global _security_middleware
    
    if _security_middleware is None:
        _security_middleware = SecurityMiddleware(get_auth_manager())
    
    return _security_middleware

def audit_log(action: str, resource: str, user_id: str, details: Optional[Dict[str, Any]] = None):
    """Log security audit event"""
    audit_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "action": action,
        "resource": resource,
        "user_id": user_id,
        "details": details or {},
        "ip_address": request.remote_addr if request else None,
        "user_agent": request.headers.get('User-Agent') if request else None
    }
    
    # In production, store in secure audit log
    logger.info(f"AUDIT: {audit_entry}")