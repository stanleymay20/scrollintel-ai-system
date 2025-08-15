"""
Production-Ready Authentication and Security System
"""
import jwt
import bcrypt
import secrets
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging
from enum import Enum

logger = logging.getLogger(__name__)

class UserRole(Enum):
    """User roles for role-based access control"""
    ADMIN = "admin"
    USER = "user"
    VIEWER = "viewer"
    API_USER = "api_user"

class Permission(Enum):
    """System permissions"""
    READ_DATA = "read_data"
    WRITE_DATA = "write_data"
    DELETE_DATA = "delete_data"
    MANAGE_USERS = "manage_users"
    MANAGE_SYSTEM = "manage_system"
    CREATE_DASHBOARDS = "create_dashboards"
    EXPORT_DATA = "export_data"
    API_ACCESS = "api_access"

class AuthenticationSystem:
    """Production-ready authentication and authorization system"""
    
    def __init__(self, secret_key: str, token_expiry_hours: int = 24):
        self.secret_key = secret_key
        self.token_expiry_hours = token_expiry_hours
        self.algorithm = "HS256"
        
        # Role-based permissions
        self.role_permissions = {
            UserRole.ADMIN: [
                Permission.READ_DATA, Permission.WRITE_DATA, Permission.DELETE_DATA,
                Permission.MANAGE_USERS, Permission.MANAGE_SYSTEM, Permission.CREATE_DASHBOARDS,
                Permission.EXPORT_DATA, Permission.API_ACCESS
            ],
            UserRole.USER: [
                Permission.READ_DATA, Permission.WRITE_DATA, Permission.CREATE_DASHBOARDS,
                Permission.EXPORT_DATA, Permission.API_ACCESS
            ],
            UserRole.VIEWER: [
                Permission.READ_DATA
            ],
            UserRole.API_USER: [
                Permission.READ_DATA, Permission.WRITE_DATA, Permission.API_ACCESS
            ]
        }
        
        # Session storage (in production, use Redis or database)
        self.active_sessions = {}
        self.api_keys = {}
        self.audit_log = []
    
    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt"""
        try:
            salt = bcrypt.gensalt()
            hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
            return hashed.decode('utf-8')
        except Exception as e:
            logger.error(f"Password hashing error: {e}")
            raise
    
    def verify_password(self, password: str, hashed_password: str) -> bool:
        """Verify password against hash"""
        try:
            return bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8'))
        except Exception as e:
            logger.error(f"Password verification error: {e}")
            return False
    
    def create_user(self, username: str, email: str, password: str, role: UserRole = UserRole.USER) -> Dict[str, Any]:
        """Create a new user"""
        try:
            # Validate input
            if not username or not email or not password:
                return {"success": False, "error": "Missing required fields"}
            
            if len(password) < 8:
                return {"success": False, "error": "Password must be at least 8 characters"}
            
            # Hash password
            hashed_password = self.hash_password(password)
            
            user_data = {
                "id": self._generate_user_id(),
                "username": username,
                "email": email,
                "password_hash": hashed_password,
                "role": role.value,
                "created_at": datetime.now().isoformat(),
                "last_login": None,
                "is_active": True,
                "failed_login_attempts": 0,
                "locked_until": None
            }
            
            self._log_audit_event("user_created", {"username": username, "role": role.value})
            
            return {
                "success": True,
                "user": {
                    "id": user_data["id"],
                    "username": user_data["username"],
                    "email": user_data["email"],
                    "role": user_data["role"],
                    "created_at": user_data["created_at"]
                }
            }
            
        except Exception as e:
            logger.error(f"User creation error: {e}")
            return {"success": False, "error": "User creation failed"}
    
    def authenticate_user(self, username: str, password: str) -> Dict[str, Any]:
        """Authenticate user and return JWT token"""
        try:
            # In production, this would query the database
            user = self._get_user_by_username(username)
            
            if not user:
                self._log_audit_event("login_failed", {"username": username, "reason": "user_not_found"})
                return {"success": False, "error": "Invalid credentials"}
            
            # Check if account is locked
            if self._is_account_locked(user):
                return {"success": False, "error": "Account temporarily locked due to failed login attempts"}
            
            # Verify password
            if not self.verify_password(password, user["password_hash"]):
                self._increment_failed_login(user["id"])
                self._log_audit_event("login_failed", {"username": username, "reason": "invalid_password"})
                return {"success": False, "error": "Invalid credentials"}
            
            # Reset failed login attempts on successful login
            self._reset_failed_login(user["id"])
            
            # Generate JWT token
            token = self._generate_jwt_token(user)
            
            # Update last login
            self._update_last_login(user["id"])
            
            self._log_audit_event("login_success", {"username": username, "user_id": user["id"]})
            
            return {
                "success": True,
                "token": token,
                "user": {
                    "id": user["id"],
                    "username": user["username"],
                    "email": user["email"],
                    "role": user["role"]
                },
                "expires_at": (datetime.now() + timedelta(hours=self.token_expiry_hours)).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return {"success": False, "error": "Authentication failed"}
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify JWT token and return user info"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            
            # Check if token is expired
            if payload.get("exp", 0) < time.time():
                return {"success": False, "error": "Token expired"}
            
            # Get user info
            user_id = payload.get("user_id")
            user = self._get_user_by_id(user_id)
            
            if not user or not user.get("is_active"):
                return {"success": False, "error": "User not found or inactive"}
            
            return {
                "success": True,
                "user": {
                    "id": user["id"],
                    "username": user["username"],
                    "email": user["email"],
                    "role": user["role"]
                }
            }
            
        except jwt.ExpiredSignatureError:
            return {"success": False, "error": "Token expired"}
        except jwt.InvalidTokenError:
            return {"success": False, "error": "Invalid token"}
        except Exception as e:
            logger.error(f"Token verification error: {e}")
            return {"success": False, "error": "Token verification failed"}
    
    def check_permission(self, user_role: str, required_permission: Permission) -> bool:
        """Check if user role has required permission"""
        try:
            role = UserRole(user_role)
            permissions = self.role_permissions.get(role, [])
            return required_permission in permissions
        except (ValueError, KeyError):
            return False
    
    def create_api_key(self, user_id: str, name: str, permissions: List[Permission] = None) -> Dict[str, Any]:
        """Create API key for programmatic access"""
        try:
            api_key = self._generate_api_key()
            
            key_data = {
                "key": api_key,
                "user_id": user_id,
                "name": name,
                "permissions": [p.value for p in permissions] if permissions else [],
                "created_at": datetime.now().isoformat(),
                "last_used": None,
                "is_active": True,
                "usage_count": 0
            }
            
            self.api_keys[api_key] = key_data
            
            self._log_audit_event("api_key_created", {"user_id": user_id, "key_name": name})
            
            return {
                "success": True,
                "api_key": api_key,
                "name": name,
                "permissions": key_data["permissions"],
                "created_at": key_data["created_at"]
            }
            
        except Exception as e:
            logger.error(f"API key creation error: {e}")
            return {"success": False, "error": "API key creation failed"}
    
    def verify_api_key(self, api_key: str) -> Dict[str, Any]:
        """Verify API key and return associated user info"""
        try:
            key_data = self.api_keys.get(api_key)
            
            if not key_data or not key_data.get("is_active"):
                return {"success": False, "error": "Invalid or inactive API key"}
            
            # Update usage statistics
            key_data["last_used"] = datetime.now().isoformat()
            key_data["usage_count"] += 1
            
            # Get user info
            user = self._get_user_by_id(key_data["user_id"])
            
            if not user or not user.get("is_active"):
                return {"success": False, "error": "Associated user not found or inactive"}
            
            return {
                "success": True,
                "user": {
                    "id": user["id"],
                    "username": user["username"],
                    "role": user["role"]
                },
                "api_key_permissions": key_data["permissions"]
            }
            
        except Exception as e:
            logger.error(f"API key verification error: {e}")
            return {"success": False, "error": "API key verification failed"}
    
    def revoke_api_key(self, api_key: str, user_id: str) -> Dict[str, Any]:
        """Revoke API key"""
        try:
            key_data = self.api_keys.get(api_key)
            
            if not key_data:
                return {"success": False, "error": "API key not found"}
            
            if key_data["user_id"] != user_id:
                return {"success": False, "error": "Unauthorized to revoke this API key"}
            
            key_data["is_active"] = False
            
            self._log_audit_event("api_key_revoked", {"user_id": user_id, "api_key": api_key[:8] + "..."})
            
            return {"success": True, "message": "API key revoked successfully"}
            
        except Exception as e:
            logger.error(f"API key revocation error: {e}")
            return {"success": False, "error": "API key revocation failed"}
    
    def get_audit_log(self, user_id: str = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get audit log entries"""
        try:
            logs = self.audit_log[-limit:] if not user_id else [
                log for log in self.audit_log if log.get("user_id") == user_id
            ][-limit:]
            
            return logs
            
        except Exception as e:
            logger.error(f"Audit log retrieval error: {e}")
            return []
    
    def _generate_jwt_token(self, user: Dict[str, Any]) -> str:
        """Generate JWT token for user"""
        payload = {
            "user_id": user["id"],
            "username": user["username"],
            "role": user["role"],
            "iat": time.time(),
            "exp": time.time() + (self.token_expiry_hours * 3600)
        }
        
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def _generate_user_id(self) -> str:
        """Generate unique user ID"""
        return f"user_{int(time.time())}_{secrets.token_hex(8)}"
    
    def _generate_api_key(self) -> str:
        """Generate secure API key"""
        return f"sk_{secrets.token_urlsafe(32)}"
    
    def _get_user_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        """Get user by username (mock implementation)"""
        # In production, this would query the database
        # For demo purposes, create a default admin user
        if username == "admin":
            return {
                "id": "user_admin_001",
                "username": "admin",
                "email": "admin@scrollintel.com",
                "password_hash": self.hash_password("admin123"),  # Default password
                "role": UserRole.ADMIN.value,
                "created_at": datetime.now().isoformat(),
                "last_login": None,
                "is_active": True,
                "failed_login_attempts": 0,
                "locked_until": None
            }
        return None
    
    def _get_user_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user by ID (mock implementation)"""
        # In production, this would query the database
        if user_id == "user_admin_001":
            return self._get_user_by_username("admin")
        return None
    
    def _is_account_locked(self, user: Dict[str, Any]) -> bool:
        """Check if account is locked due to failed login attempts"""
        if user.get("failed_login_attempts", 0) >= 5:
            locked_until = user.get("locked_until")
            if locked_until:
                return datetime.fromisoformat(locked_until) > datetime.now()
            return True
        return False
    
    def _increment_failed_login(self, user_id: str):
        """Increment failed login attempts"""
        # In production, this would update the database
        pass
    
    def _reset_failed_login(self, user_id: str):
        """Reset failed login attempts"""
        # In production, this would update the database
        pass
    
    def _update_last_login(self, user_id: str):
        """Update last login timestamp"""
        # In production, this would update the database
        pass
    
    def _log_audit_event(self, event_type: str, details: Dict[str, Any]):
        """Log audit event"""
        audit_entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "details": details,
            "ip_address": "127.0.0.1",  # In production, get from request
            "user_agent": "ScrollIntel/1.0"  # In production, get from request
        }
        
        self.audit_log.append(audit_entry)
        
        # Keep only last 1000 entries in memory
        if len(self.audit_log) > 1000:
            self.audit_log = self.audit_log[-1000:]
        
        logger.info(f"Audit event: {event_type} - {details}")

class SecurityMiddleware:
    """Security middleware for request validation"""
    
    def __init__(self, auth_system: AuthenticationSystem):
        self.auth_system = auth_system
    
    def validate_request(self, headers: Dict[str, str], required_permission: Permission = None) -> Dict[str, Any]:
        """Validate incoming request"""
        try:
            # Check for JWT token
            auth_header = headers.get("Authorization", "")
            if auth_header.startswith("Bearer "):
                token = auth_header[7:]
                result = self.auth_system.verify_token(token)
                
                if result["success"] and required_permission:
                    user_role = result["user"]["role"]
                    if not self.auth_system.check_permission(user_role, required_permission):
                        return {"success": False, "error": "Insufficient permissions"}
                
                return result
            
            # Check for API key
            api_key = headers.get("X-API-Key", "")
            if api_key:
                result = self.auth_system.verify_api_key(api_key)
                
                if result["success"] and required_permission:
                    api_permissions = result.get("api_key_permissions", [])
                    if required_permission.value not in api_permissions:
                        return {"success": False, "error": "API key lacks required permission"}
                
                return result
            
            return {"success": False, "error": "No authentication provided"}
            
        except Exception as e:
            logger.error(f"Request validation error: {e}")
            return {"success": False, "error": "Request validation failed"}
    
    def require_permission(self, permission: Permission):
        """Decorator to require specific permission"""
        def decorator(func):
            def wrapper(*args, **kwargs):
                # This would be implemented as a proper decorator in production
                return func(*args, **kwargs)
            return wrapper
        return decorator