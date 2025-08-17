"""
Simplified Input Validation Framework for Demo
"""

import re
import html
import json
from datetime import datetime
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum

class ValidationResult(Enum):
    VALID = "valid"
    INVALID = "invalid"
    SANITIZED = "sanitized"
    BLOCKED = "blocked"

class InputType(Enum):
    TEXT = "text"
    EMAIL = "email"
    URL = "url"
    JSON = "json"
    HTML = "html"
    NUMBER = "number"
    BOOLEAN = "boolean"

@dataclass
class ValidationRule:
    name: str
    input_type: InputType
    required: bool = False
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    pattern: Optional[str] = None
    allowed_values: Optional[List[Any]] = None

@dataclass
class ValidationError:
    field: str
    rule: str
    message: str
    input_value: Any

@dataclass
class ValidationReport:
    is_valid: bool
    result: ValidationResult
    sanitized_data: Dict[str, Any]
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    blocked_fields: List[str] = field(default_factory=list)

class InputValidator:
    """Simplified input validation framework"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.strict_mode = self.config.get('strict_mode', True)
        self.auto_sanitize = self.config.get('auto_sanitize', True)
        
        # Security patterns
        self.security_patterns = {
            'sql_injection': [
                r"(?i)(union\s+select|union\s+all\s+select)",
                r"(?i)(\'\s*or\s*\'\s*=\s*\')",
                r"(?i)(\'\s*or\s*1\s*=\s*1)",
                r"(?i)(drop\s+table|delete\s+from)",
            ],
            'xss': [
                r"(?i)<script[^>]*>.*?</script>",
                r"(?i)javascript:",
                r"(?i)on\w+\s*=",
            ],
            'command_injection': [
                r"(?i)(;|\||\&)\s*(cat|ls|pwd|whoami)",
                r"(?i)/bin/(sh|bash)",
            ]
        }
    
    def validate_input(self, data: Dict[str, Any], rules: Dict[str, ValidationRule]) -> ValidationReport:
        """Validate input data against rules"""
        
        sanitized_data = {}
        errors = []
        warnings = []
        blocked_fields = []
        overall_valid = True
        result = ValidationResult.VALID
        
        for field_name, rule in rules.items():
            field_value = data.get(field_name)
            
            # Check if required
            if rule.required and (field_value is None or field_value == ""):
                overall_valid = False
                errors.append(ValidationError(
                    field=field_name,
                    rule="required",
                    message=f"Field '{field_name}' is required",
                    input_value=field_value
                ))
                continue
            
            if field_value is None or field_value == "":
                sanitized_data[field_name] = field_value
                continue
            
            str_value = str(field_value)
            
            # Check for security threats
            security_check = self._check_security_threats(str_value)
            if security_check['is_threat']:
                overall_valid = False
                blocked_fields.append(field_name)
                result = ValidationResult.BLOCKED
                errors.append(ValidationError(
                    field=field_name,
                    rule="security_threat",
                    message=f"Security threat detected: {security_check['threat_type']}",
                    input_value=field_value
                ))
                continue
            
            # Length validation
            if rule.min_length and len(str_value) < rule.min_length:
                overall_valid = False
                errors.append(ValidationError(
                    field=field_name,
                    rule="min_length",
                    message=f"Minimum length is {rule.min_length}",
                    input_value=field_value
                ))
                continue
            
            if rule.max_length and len(str_value) > rule.max_length:
                if self.auto_sanitize:
                    str_value = str_value[:rule.max_length]
                    result = ValidationResult.SANITIZED
                    warnings.append(f"Field '{field_name}' was truncated")
                else:
                    overall_valid = False
                    errors.append(ValidationError(
                        field=field_name,
                        rule="max_length",
                        message=f"Maximum length is {rule.max_length}",
                        input_value=field_value
                    ))
                    continue
            
            # Pattern validation
            if rule.pattern and not re.match(rule.pattern, str_value):
                overall_valid = False
                errors.append(ValidationError(
                    field=field_name,
                    rule="pattern",
                    message=f"Value does not match required pattern",
                    input_value=field_value
                ))
                continue
            
            # Type-specific validation
            validated_value = self._validate_by_type(str_value, rule.input_type)
            if validated_value is None:
                overall_valid = False
                errors.append(ValidationError(
                    field=field_name,
                    rule=f"type_{rule.input_type.value}",
                    message=f"Invalid {rule.input_type.value} format",
                    input_value=field_value
                ))
                continue
            
            sanitized_data[field_name] = validated_value
        
        return ValidationReport(
            is_valid=overall_valid,
            result=result,
            sanitized_data=sanitized_data,
            errors=errors,
            warnings=warnings,
            blocked_fields=blocked_fields
        )
    
    def _check_security_threats(self, value: str) -> Dict[str, Any]:
        """Check for security threats"""
        for threat_type, patterns in self.security_patterns.items():
            for pattern in patterns:
                if re.search(pattern, value):
                    return {'is_threat': True, 'threat_type': threat_type}
        return {'is_threat': False}
    
    def _validate_by_type(self, value: str, input_type: InputType):
        """Validate by input type"""
        try:
            if input_type == InputType.EMAIL:
                if not re.match(r'^[^@]+@[^@]+\.[^@]+$', value):
                    return None
                return value
            
            elif input_type == InputType.NUMBER:
                if '.' in value:
                    return float(value)
                else:
                    return int(value)
            
            elif input_type == InputType.BOOLEAN:
                if value.lower() in ['true', '1', 'yes']:
                    return True
                elif value.lower() in ['false', '0', 'no']:
                    return False
                return None
            
            elif input_type == InputType.JSON:
                return json.loads(value)
            
            elif input_type == InputType.HTML:
                # Simple HTML sanitization
                return html.escape(value)
            
            else:  # TEXT
                return value
                
        except Exception:
            return None

class CommonValidationRules:
    """Common validation rules"""
    
    @staticmethod
    def user_registration() -> Dict[str, ValidationRule]:
        return {
            'username': ValidationRule(
                name='username',
                input_type=InputType.TEXT,
                required=True,
                min_length=3,
                max_length=50,
                pattern=r'^[a-zA-Z0-9_-]+$'
            ),
            'email': ValidationRule(
                name='email',
                input_type=InputType.EMAIL,
                required=True
            ),
            'password': ValidationRule(
                name='password',
                input_type=InputType.TEXT,
                required=True,
                min_length=8
            )
        }
    
    @staticmethod
    def api_request() -> Dict[str, ValidationRule]:
        return {
            'data': ValidationRule(
                name='data',
                input_type=InputType.JSON,
                required=True
            ),
            'format': ValidationRule(
                name='format',
                input_type=InputType.TEXT,
                allowed_values=['json', 'xml', 'csv']
            )
        }

def create_input_validator(config: Dict[str, Any] = None) -> InputValidator:
    """Factory function"""
    return InputValidator(config or {})