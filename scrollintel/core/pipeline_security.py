"""
Pipeline Security and Validation System
Provides comprehensive security controls, input validation, and access management.
"""
import re
import hashlib
import secrets
import jwt
from typing import Dict, Any, List, Optional, Set, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import logging
from functools import wraps
import json
from sqlalchemy.sql import text
import bleach

logger = logging.getLogger(__name__)

class SecurityLevel(Enum):
    """Security levels for operations"""
    PUBLIC = "public"
    AUTHENTICATED = "authenticated"
    AUTHORIZED = "authorized"
    ADMIN = "admin"
    SYSTEM = "system"

class ValidationSeverity(Enum):
    """Validation error severity"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class ValidationResult:
    """Validation result"""
    is_valid: bool
    severity: ValidationSeverity
    message: str
    field: Optional[str] = None
    code: Optional[str] = None

@dataclass
class SecurityContext:
    """Security context for operations"""
    user_id: str
    roles: List[str]
    permissions: Set[str]
    ip_address: str
    user_agent: str
    session_id: str
    authenticated_at: datetime
    expires_at: datetime

class InputValidator:
    """Comprehensive input validation system"""
    
    def __init__(self):
        # SQL injection patterns
        self.sql_injection_patterns = [
            r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION)\b)",
            r"(--|#|/\*|\*/)",
            r"(\b(OR|AND)\s+\d+\s*=\s*\d+)",
            r"(\b(OR|AND)\s+['\"]?\w+['\"]?\s*=\s*['\"]?\w+['\"]?)",
            r"(INFORMATION_SCHEMA|SYSOBJECTS|SYSCOLUMNS)",
        ]
        
        # XSS patterns
        self.xss_patterns = [
            r"<script[^>]*>.*?</script>",
            r"javascript:",
            r"on\w+\s*=",
            r"<iframe[^>]*>.*?</iframe>",
            r"<object[^>]*>.*?</object>",
        ]
        
        # Path traversal patterns
        self.path_traversal_patterns = [
            r"\.\./",
            r"\.\.\\",
            r"%2e%2e%2f",
            r"%2e%2e\\",
        ]
        
        # Command injection patterns
        self.command_injection_patterns = [
            r"[;&|`$(){}[\]<>]",
            r"\b(cat|ls|pwd|whoami|id|uname|ps|netstat|ifconfig)\b",
        ]
    
    def validate_sql_injection(self, value: str) -> ValidationResult:
        """Check for SQL injection attempts"""
        if not isinstance(value, str):
            return ValidationResult(True, ValidationSeverity.INFO, "Not a string value")
        
        value_lower = value.lower()
        
        for pattern in self.sql_injection_patterns:
            if re.search(pattern, value_lower, re.IGNORECASE):
                return ValidationResult(
                    False, 
                    ValidationSeverity.CRITICAL,
                    f"Potential SQL injection detected: {pattern}",
                    code="SQL_INJECTION"
                )
        
        return ValidationResult(True, ValidationSeverity.INFO, "No SQL injection detected")
    
    def validate_xss(self, value: str) -> ValidationResult:
        """Check for XSS attempts"""
        if not isinstance(value, str):
            return ValidationResult(True, ValidationSeverity.INFO, "Not a string value")
        
        for pattern in self.xss_patterns:
            if re.search(pattern, value, re.IGNORECASE):
                return ValidationResult(
                    False,
                    ValidationSeverity.ERROR,
                    f"Potential XSS detected: {pattern}",
                    code="XSS_ATTEMPT"
                )
        
        return ValidationResult(True, ValidationSeverity.INFO, "No XSS detected")
    
    def validate_path_traversal(self, value: str) -> ValidationResult:
        """Check for path traversal attempts"""
        if not isinstance(value, str):
            return ValidationResult(True, ValidationSeverity.INFO, "Not a string value")
        
        for pattern in self.path_traversal_patterns:
            if re.search(pattern, value, re.IGNORECASE):
                return ValidationResult(
                    False,
                    ValidationSeverity.ERROR,
                    f"Potential path traversal detected: {pattern}",
                    code="PATH_TRAVERSAL"
                )
        
        return ValidationResult(True, ValidationSeverity.INFO, "No path traversal detected")
    
    def validate_command_injection(self, value: str) -> ValidationResult:
        """Check for command injection attempts"""
        if not isinstance(value, str):
            return ValidationResult(True, ValidationSeverity.INFO, "Not a string value")
        
        for pattern in self.command_injection_patterns:
            if re.search(pattern, value):
                return ValidationResult(
                    False,
                    ValidationSeverity.CRITICAL,
                    f"Potential command injection detected: {pattern}",
                    code="COMMAND_INJECTION"
                )
        
        return ValidationResult(True, ValidationSeverity.INFO, "No command injection detected")
    
    def sanitize_html(self, value: str) -> str:
        """Sanitize HTML content"""
        if not isinstance(value, str):
            return str(value)
        
        # Allow only safe HTML tags and attributes
        allowed_tags = ['p', 'br', 'strong', 'em', 'u', 'ol', 'ul', 'li', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']
        allowed_attributes = {}
        
        return bleach.clean(value, tags=allowed_tags, attributes=allowed_attributes, strip=True)
    
    def validate_pipeline_config(self, config: Dict[str, Any]) -> List[ValidationResult]:
        """Validate pipeline configuration"""
        results = []
        
        # Validate required fields
        required_fields = ['name', 'nodes', 'connections']
        for field in required_fields:
            if field not in config:
                results.append(ValidationResult(
                    False,
                    ValidationSeverity.ERROR,
                    f"Required field '{field}' is missing",
                    field=field,
                    code="MISSING_FIELD"
                ))
        
        # Validate pipeline name
        if 'name' in config:
            name = config['name']
            if not isinstance(name, str) or len(name.strip()) == 0:
                results.append(ValidationResult(
                    False,
                    ValidationSeverity.ERROR,
                    "Pipeline name must be a non-empty string",
                    field="name",
                    code="INVALID_NAME"
                ))
            elif len(name) > 255:
                results.append(ValidationResult(
                    False,
                    ValidationSeverity.ERROR,
                    "Pipeline name must be less than 255 characters",
                    field="name",
                    code="NAME_TOO_LONG"
                ))
            else:
                # Check for security issues in name
                xss_result = self.validate_xss(name)
                if not xss_result.is_valid:
                    results.append(ValidationResult(
                        False,
                        ValidationSeverity.ERROR,
                        f"Pipeline name contains unsafe content: {xss_result.message}",
                        field="name",
                        code="UNSAFE_NAME"
                    ))
        
        # Validate nodes
        if 'nodes' in config:
            nodes = config['nodes']
            if not isinstance(nodes, list):
                results.append(ValidationResult(
                    False,
                    ValidationSeverity.ERROR,
                    "Nodes must be a list",
                    field="nodes",
                    code="INVALID_NODES_TYPE"
                ))
            else:
                for i, node in enumerate(nodes):
                    node_results = self._validate_node(node, f"nodes[{i}]")
                    results.extend(node_results)
        
        # Validate connections
        if 'connections' in config:
            connections = config['connections']
            if not isinstance(connections, list):
                results.append(ValidationResult(
                    False,
                    ValidationSeverity.ERROR,
                    "Connections must be a list",
                    field="connections",
                    code="INVALID_CONNECTIONS_TYPE"
                ))
            else:
                for i, connection in enumerate(connections):
                    conn_results = self._validate_connection(connection, f"connections[{i}]")
                    results.extend(conn_results)
        
        return results
    
    def _validate_node(self, node: Dict[str, Any], field_prefix: str) -> List[ValidationResult]:
        """Validate individual pipeline node"""
        results = []
        
        # Required node fields
        required_fields = ['id', 'type', 'config']
        for field in required_fields:
            if field not in node:
                results.append(ValidationResult(
                    False,
                    ValidationSeverity.ERROR,
                    f"Required field '{field}' is missing from node",
                    field=f"{field_prefix}.{field}",
                    code="MISSING_NODE_FIELD"
                ))
        
        # Validate node ID
        if 'id' in node:
            node_id = node['id']
            if not isinstance(node_id, str) or not re.match(r'^[a-zA-Z0-9_-]+$', node_id):
                results.append(ValidationResult(
                    False,
                    ValidationSeverity.ERROR,
                    "Node ID must contain only alphanumeric characters, underscores, and hyphens",
                    field=f"{field_prefix}.id",
                    code="INVALID_NODE_ID"
                ))
        
        # Validate node configuration
        if 'config' in node:
            config = node['config']
            if isinstance(config, dict):
                # Check for dangerous configuration values
                for key, value in config.items():
                    if isinstance(value, str):
                        # Check for SQL injection in config values
                        sql_result = self.validate_sql_injection(value)
                        if not sql_result.is_valid:
                            results.append(ValidationResult(
                                False,
                                ValidationSeverity.CRITICAL,
                                f"Unsafe configuration value in {key}: {sql_result.message}",
                                field=f"{field_prefix}.config.{key}",
                                code="UNSAFE_CONFIG_VALUE"
                            ))
        
        return results
    
    def _validate_connection(self, connection: Dict[str, Any], field_prefix: str) -> List[ValidationResult]:
        """Validate individual pipeline connection"""
        results = []
        
        # Required connection fields
        required_fields = ['source', 'target']
        for field in required_fields:
            if field not in connection:
                results.append(ValidationResult(
                    False,
                    ValidationSeverity.ERROR,
                    f"Required field '{field}' is missing from connection",
                    field=f"{field_prefix}.{field}",
                    code="MISSING_CONNECTION_FIELD"
                ))
        
        return results

class AccessController:
    """Access control and authorization system"""
    
    def __init__(self):
        self.permissions = {
            'pipeline.create': 'Create new pipelines',
            'pipeline.read': 'View pipelines',
            'pipeline.update': 'Modify pipelines',
            'pipeline.delete': 'Delete pipelines',
            'pipeline.execute': 'Execute pipelines',
            'pipeline.admin': 'Full pipeline administration',
            'system.admin': 'System administration'
        }
        
        self.role_permissions = {
            'viewer': ['pipeline.read'],
            'editor': ['pipeline.read', 'pipeline.create', 'pipeline.update'],
            'executor': ['pipeline.read', 'pipeline.execute'],
            'admin': ['pipeline.read', 'pipeline.create', 'pipeline.update', 
                     'pipeline.delete', 'pipeline.execute', 'pipeline.admin'],
            'system_admin': list(self.permissions.keys())
        }
    
    def check_permission(self, security_context: SecurityContext, 
                        required_permission: str) -> bool:
        """Check if user has required permission"""
        # System users have all permissions
        if 'system' in security_context.roles:
            return True
        
        # Check direct permissions
        if required_permission in security_context.permissions:
            return True
        
        # Check role-based permissions
        for role in security_context.roles:
            role_perms = self.role_permissions.get(role, [])
            if required_permission in role_perms:
                return True
        
        return False
    
    def get_user_permissions(self, roles: List[str]) -> Set[str]:
        """Get all permissions for user roles"""
        permissions = set()
        
        for role in roles:
            role_perms = self.role_permissions.get(role, [])
            permissions.update(role_perms)
        
        return permissions

class SecurityAuditor:
    """Security audit and logging system"""
    
    def __init__(self):
        self.audit_log = []
        self.max_log_entries = 10000
    
    def log_security_event(self, event_type: str, user_id: str, 
                          resource: str, action: str, 
                          success: bool, details: Dict[str, Any] = None):
        """Log security-related events"""
        event = {
            'timestamp': datetime.utcnow().isoformat(),
            'event_type': event_type,
            'user_id': user_id,
            'resource': resource,
            'action': action,
            'success': success,
            'details': details or {}
        }
        
        self.audit_log.append(event)
        
        # Keep only recent entries
        if len(self.audit_log) > self.max_log_entries:
            self.audit_log = self.audit_log[-self.max_log_entries:]
        
        # Log to system logger
        level = logging.INFO if success else logging.WARNING
        logger.log(level, f"Security event: {event_type} - {action} on {resource} by {user_id}")
    
    def get_audit_log(self, user_id: str = None, hours: int = 24) -> List[Dict[str, Any]]:
        """Get audit log entries"""
        since = datetime.utcnow() - timedelta(hours=hours)
        
        filtered_log = []
        for entry in self.audit_log:
            entry_time = datetime.fromisoformat(entry['timestamp'])
            if entry_time >= since:
                if not user_id or entry['user_id'] == user_id:
                    filtered_log.append(entry)
        
        return sorted(filtered_log, key=lambda x: x['timestamp'], reverse=True)

def require_permission(permission: str):
    """Decorator to require specific permission"""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Extract security context from kwargs
            security_context = kwargs.get('security_context')
            if not security_context:
                raise PermissionError("No security context provided")
            
            if not access_controller.check_permission(security_context, permission):
                security_auditor.log_security_event(
                    'access_denied',
                    security_context.user_id,
                    func.__name__,
                    permission,
                    False,
                    {'required_permission': permission}
                )
                raise PermissionError(f"Permission '{permission}' required")
            
            # Log successful access
            security_auditor.log_security_event(
                'access_granted',
                security_context.user_id,
                func.__name__,
                permission,
                True
            )
            
            return await func(*args, **kwargs)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            security_context = kwargs.get('security_context')
            if not security_context:
                raise PermissionError("No security context provided")
            
            if not access_controller.check_permission(security_context, permission):
                security_auditor.log_security_event(
                    'access_denied',
                    security_context.user_id,
                    func.__name__,
                    permission,
                    False,
                    {'required_permission': permission}
                )
                raise PermissionError(f"Permission '{permission}' required")
            
            security_auditor.log_security_event(
                'access_granted',
                security_context.user_id,
                func.__name__,
                permission,
                True
            )
            
            return func(*args, **kwargs)
        
        # Return appropriate wrapper
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

def validate_input(validation_func: Callable = None):
    """Decorator for input validation"""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Apply default validation if no custom function provided
            if validation_func:
                validation_results = validation_func(*args, **kwargs)
            else:
                validation_results = []
                
                # Apply default validations to string arguments
                for key, value in kwargs.items():
                    if isinstance(value, str):
                        # Check for common security issues
                        sql_result = input_validator.validate_sql_injection(value)
                        if not sql_result.is_valid:
                            validation_results.append(sql_result)
                        
                        xss_result = input_validator.validate_xss(value)
                        if not xss_result.is_valid:
                            validation_results.append(xss_result)
            
            # Check for validation errors
            errors = [r for r in validation_results if not r.is_valid and r.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]]
            if errors:
                error_messages = [f"{r.field}: {r.message}" if r.field else r.message for r in errors]
                raise ValueError(f"Validation failed: {'; '.join(error_messages)}")
            
            return await func(*args, **kwargs)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Similar validation for sync functions
            if validation_func:
                validation_results = validation_func(*args, **kwargs)
            else:
                validation_results = []
                
                for key, value in kwargs.items():
                    if isinstance(value, str):
                        sql_result = input_validator.validate_sql_injection(value)
                        if not sql_result.is_valid:
                            validation_results.append(sql_result)
                        
                        xss_result = input_validator.validate_xss(value)
                        if not xss_result.is_valid:
                            validation_results.append(xss_result)
            
            errors = [r for r in validation_results if not r.is_valid and r.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]]
            if errors:
                error_messages = [f"{r.field}: {r.message}" if r.field else r.message for r in errors]
                raise ValueError(f"Validation failed: {'; '.join(error_messages)}")
            
            return func(*args, **kwargs)
        
        # Return appropriate wrapper
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

# Global instances
input_validator = InputValidator()
access_controller = AccessController()
security_auditor = SecurityAuditor()

# Convenience functions
def create_security_context(user_id: str, roles: List[str], 
                          ip_address: str = "unknown", 
                          user_agent: str = "unknown") -> SecurityContext:
    """Create security context for user"""
    permissions = access_controller.get_user_permissions(roles)
    
    return SecurityContext(
        user_id=user_id,
        roles=roles,
        permissions=permissions,
        ip_address=ip_address,
        user_agent=user_agent,
        session_id=secrets.token_urlsafe(32),
        authenticated_at=datetime.utcnow(),
        expires_at=datetime.utcnow() + timedelta(hours=8)
    )

def validate_pipeline_security(config: Dict[str, Any]) -> List[ValidationResult]:
    """Validate pipeline configuration for security issues"""
    return input_validator.validate_pipeline_config(config)

def get_security_summary() -> Dict[str, Any]:
    """Get security system summary"""
    recent_events = security_auditor.get_audit_log(hours=24)
    
    return {
        'total_audit_events': len(security_auditor.audit_log),
        'recent_events_24h': len(recent_events),
        'failed_access_attempts': len([e for e in recent_events if not e['success']]),
        'permissions_available': len(access_controller.permissions),
        'roles_configured': len(access_controller.role_permissions)
    }