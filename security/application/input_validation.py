"""
Input Validation and Sanitization Framework
Comprehensive input validation for all user inputs
"""

import re
import html
import json
import base64
import urllib.parse
from datetime import datetime
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
# import bleach
# from email_validator import validate_email, EmailNotValidError
import ipaddress
# import phonenumbers
# from phonenumbers import NumberParseException

class ValidationResult(Enum):
    VALID = "valid"
    INVALID = "invalid"
    SANITIZED = "sanitized"
    BLOCKED = "blocked"

class InputType(Enum):
    TEXT = "text"
    EMAIL = "email"
    URL = "url"
    PHONE = "phone"
    IP_ADDRESS = "ip_address"
    JSON = "json"
    HTML = "html"
    SQL = "sql"
    FILENAME = "filename"
    PATH = "path"
    CREDIT_CARD = "credit_card"
    SSN = "ssn"
    DATE = "date"
    NUMBER = "number"
    BOOLEAN = "boolean"
    UUID = "uuid"
    BASE64 = "base64"
    REGEX = "regex"

@dataclass
class ValidationRule:
    name: str
    input_type: InputType
    required: bool = False
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    pattern: Optional[str] = None
    allowed_values: Optional[List[Any]] = None
    custom_validator: Optional[Callable] = None
    sanitizer: Optional[Callable] = None
    error_message: Optional[str] = None

@dataclass
class ValidationError:
    field: str
    rule: str
    message: str
    input_value: Any
    suggested_value: Optional[Any] = None

@dataclass
class ValidationReport:
    is_valid: bool
    result: ValidationResult
    sanitized_data: Dict[str, Any]
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    blocked_fields: List[str] = field(default_factory=list)

class InputValidator:
    """Comprehensive input validation and sanitization framework"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.strict_mode = self.config.get('strict_mode', True)
        self.auto_sanitize = self.config.get('auto_sanitize', True)
        
        # Security patterns for detection
        self.security_patterns = {
            'sql_injection': [
                r"(?i)(union\s+select|union\s+all\s+select)",
                r"(?i)(select.*from.*information_schema)",
                r"(?i)(\'\s*or\s*\'\s*=\s*\')",
                r"(?i)(\'\s*or\s*1\s*=\s*1)",
                r"(?i)(drop\s+table|delete\s+from|insert\s+into)",
                r"(?i)(exec\s*\(|execute\s*\()",
            ],
            'xss': [
                r"(?i)<script[^>]*>.*?</script>",
                r"(?i)javascript:",
                r"(?i)on\w+\s*=",
                r"(?i)<iframe[^>]*>",
                r"(?i)eval\s*\(",
                r"(?i)<object[^>]*>",
                r"(?i)<embed[^>]*>",
            ],
            'command_injection': [
                r"(?i)(;|\||\&)\s*(cat|ls|pwd|whoami|id|uname)",
                r"(?i)(wget|curl)\s+http",
                r"(?i)(nc|netcat)\s+-",
                r"(?i)/bin/(sh|bash|csh)",
                r"(?i)cmd\.exe",
            ],
            'path_traversal': [
                r"\.\.\/",
                r"\.\.\\",
                r"%2e%2e%2f",
                r"%2e%2e%5c",
                r"\/etc\/passwd",
                r"\/windows\/system32",
            ],
            'ldap_injection': [
                r"(?i)(\*|\(|\)|\||&)",
                r"(?i)(objectclass=\*)",
            ],
            'xml_injection': [
                r"(?i)<!entity",
                r"(?i)<!doctype",
                r"(?i)<\?xml",
            ]
        }
        
        # HTML sanitization configuration
        self.allowed_html_tags = self.config.get('allowed_html_tags', [
            'p', 'br', 'strong', 'em', 'u', 'ol', 'ul', 'li', 'a', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'
        ])
        
        self.allowed_html_attributes = self.config.get('allowed_html_attributes', {
            'a': ['href', 'title'],
            '*': ['class']
        })

class InputValidator:
    """Comprehensive input validation and sanitization framework"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.strict_mode = self.config.get('strict_mode', True)
        self.auto_sanitize = self.config.get('auto_sanitize', True)
        
        # Security patterns for detection
        self.security_patterns = {
            'sql_injection': [
                r"(?i)(union\s+select|union\s+all\s+select)",
                r"(?i)(select.*from.*information_schema)",
                r"(?i)(\'\s*or\s*\'\s*=\s*\')",
                r"(?i)(\'\s*or\s*1\s*=\s*1)",
                r"(?i)(drop\s+table|delete\s+from|insert\s+into)",
                r"(?i)(exec\s*\(|execute\s*\()",
            ],
            'xss': [
                r"(?i)<script[^>]*>.*?</script>",
                r"(?i)javascript:",
                r"(?i)on\w+\s*=",
                r"(?i)<iframe[^>]*>",
                r"(?i)eval\s*\(",
                r"(?i)<object[^>]*>",
                r"(?i)<embed[^>]*>",
            ],
            'command_injection': [
                r"(?i)(;|\||\&)\s*(cat|ls|pwd|whoami|id|uname)",
                r"(?i)(wget|curl)\s+http",
                r"(?i)(nc|netcat)\s+-",
                r"(?i)/bin/(sh|bash|csh)",
                r"(?i)cmd\.exe",
            ],
            'path_traversal': [
                r"\.\.\/",
                r"\.\.\\",
                r"%2e%2e%2f",
                r"%2e%2e%5c",
                r"\/etc\/passwd",
                r"\/windows\/system32",
            ],
            'ldap_injection': [
                r"(?i)(\*|\(|\)|\||&)",
                r"(?i)(objectclass=\*)",
            ],
            'xml_injection': [
                r"(?i)<!entity",
                r"(?i)<!doctype",
                r"(?i)<\?xml",
            ]
        }
        
        # HTML sanitization configuration
        self.allowed_html_tags = self.config.get('allowed_html_tags', [
            'p', 'br', 'strong', 'em', 'u', 'ol', 'ul', 'li', 'a', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'
        ])
        
        self.allowed_html_attributes = self.config.get('allowed_html_attributes', {
            'a': ['href', 'title'],
            '*': ['class']
        })
    
    def validate_input(self, data: Dict[str, Any], rules: Dict[str, ValidationRule]) -> ValidationReport:
        """Validate input data against rules"""
        
        sanitized_data = {}
        errors = []
        warnings = []
        blocked_fields = []
        overall_valid = True
        result = ValidationResult.VALID
        
        # Validate each field
        for field_name, rule in rules.items():
            field_value = data.get(field_name)
            
            try:
                field_result = self._validate_field(field_name, field_value, rule)
                
                if field_result['is_valid']:
                    sanitized_data[field_name] = field_result['sanitized_value']
                    if field_result['was_sanitized']:
                        result = ValidationResult.SANITIZED
                        warnings.append(f"Field '{field_name}' was sanitized")
                else:
                    overall_valid = False
                    errors.extend(field_result['errors'])
                    
                    if field_result['is_blocked']:
                        blocked_fields.append(field_name)
                        result = ValidationResult.BLOCKED
                    elif result != ValidationResult.BLOCKED:
                        result = ValidationResult.INVALID
                
            except Exception as e:
                overall_valid = False
                errors.append(ValidationError(
                    field=field_name,
                    rule="validation_error",
                    message=f"Validation failed: {str(e)}",
                    input_value=field_value
                ))
        
        # Check for extra fields not in rules
        extra_fields = set(data.keys()) - set(rules.keys())
        if extra_fields and self.strict_mode:
            overall_valid = False
            for field in extra_fields:
                errors.append(ValidationError(
                    field=field,
                    rule="unexpected_field",
                    message=f"Unexpected field '{field}' not allowed",
                    input_value=data[field]
                ))
        
        return ValidationReport(
            is_valid=overall_valid,
            result=result,
            sanitized_data=sanitized_data,
            errors=errors,
            warnings=warnings,
            blocked_fields=blocked_fields
        )
    
    def _validate_field(self, field_name: str, value: Any, rule: ValidationRule) -> Dict[str, Any]:
        """Validate a single field"""
        
        result = {
            'is_valid': True,
            'sanitized_value': value,
            'was_sanitized': False,
            'is_blocked': False,
            'errors': []
        }
        
        # Check if required
        if rule.required and (value is None or value == ""):
            result['is_valid'] = False
            result['errors'].append(ValidationError(
                field=field_name,
                rule="required",
                message=rule.error_message or f"Field '{field_name}' is required",
                input_value=value
            ))
            return result
        
        # Skip validation if value is None/empty and not required
        if value is None or value == "":
            return result
        
        # Convert to string for most validations
        str_value = str(value)
        
        # Check for security threats first
        security_check = self._check_security_threats(str_value)
        if security_check['is_threat']:
            result['is_valid'] = False
            result['is_blocked'] = True
            result['errors'].append(ValidationError(
                field=field_name,
                rule="security_threat",
                message=f"Security threat detected: {security_check['threat_type']}",
                input_value=value
            ))
            return result
        
        # Length validation
        if rule.min_length is not None and len(str_value) < rule.min_length:
            result['is_valid'] = False
            result['errors'].append(ValidationError(
                field=field_name,
                rule="min_length",
                message=rule.error_message or f"Minimum length is {rule.min_length}",
                input_value=value
            ))
        
        if rule.max_length is not None and len(str_value) > rule.max_length:
            if self.auto_sanitize:
                result['sanitized_value'] = str_value[:rule.max_length]
                result['was_sanitized'] = True
            else:
                result['is_valid'] = False
                result['errors'].append(ValidationError(
                    field=field_name,
                    rule="max_length",
                    message=rule.error_message or f"Maximum length is {rule.max_length}",
                    input_value=value
                ))
        
        # Pattern validation
        if rule.pattern and not re.match(rule.pattern, str_value):
            result['is_valid'] = False
            result['errors'].append(ValidationError(
                field=field_name,
                rule="pattern",
                message=rule.error_message or f"Value does not match required pattern",
                input_value=value
            ))
        
        # Allowed values validation
        if rule.allowed_values and value not in rule.allowed_values:
            result['is_valid'] = False
            result['errors'].append(ValidationError(
                field=field_name,
                rule="allowed_values",
                message=rule.error_message or f"Value must be one of: {rule.allowed_values}",
                input_value=value
            ))
        
        # Type-specific validation
        type_result = self._validate_by_type(value, rule.input_type)
        if not type_result['is_valid']:
            result['is_valid'] = False
            result['errors'].extend([
                ValidationError(
                    field=field_name,
                    rule=f"type_{rule.input_type.value}",
                    message=error,
                    input_value=value
                ) for error in type_result['errors']
            ])
        elif type_result['sanitized_value'] != value:
            result['sanitized_value'] = type_result['sanitized_value']
            result['was_sanitized'] = True
        
        # Custom validator
        if rule.custom_validator:
            try:
                custom_result = rule.custom_validator(value)
                if not custom_result:
                    result['is_valid'] = False
                    result['errors'].append(ValidationError(
                        field=field_name,
                        rule="custom_validation",
                        message=rule.error_message or "Custom validation failed",
                        input_value=value
                    ))
            except Exception as e:
                result['is_valid'] = False
                result['errors'].append(ValidationError(
                    field=field_name,
                    rule="custom_validation_error",
                    message=f"Custom validation error: {str(e)}",
                    input_value=value
                ))
        
        # Custom sanitizer
        if rule.sanitizer and result['is_valid']:
            try:
                sanitized = rule.sanitizer(result['sanitized_value'])
                if sanitized != result['sanitized_value']:
                    result['sanitized_value'] = sanitized
                    result['was_sanitized'] = True
            except Exception as e:
                result['is_valid'] = False
                result['errors'].append(ValidationError(
                    field=field_name,
                    rule="sanitization_error",
                    message=f"Sanitization error: {str(e)}",
                    input_value=value
                ))
        
        return result
    
    def _check_security_threats(self, value: str) -> Dict[str, Any]:
        """Check for security threats in input"""
        
        for threat_type, patterns in self.security_patterns.items():
            for pattern in patterns:
                if re.search(pattern, value):
                    return {
                        'is_threat': True,
                        'threat_type': threat_type,
                        'pattern': pattern
                    }
        
        return {'is_threat': False}
    
    def _validate_by_type(self, value: Any, input_type: InputType) -> Dict[str, Any]:
        """Validate value based on input type"""
        
        result = {
            'is_valid': True,
            'sanitized_value': value,
            'errors': []
        }
        
        try:
            if input_type == InputType.EMAIL:
                try:
                    validated_email = validate_email(str(value))
                    result['sanitized_value'] = validated_email.email
                except EmailNotValidError as e:
                    result['is_valid'] = False
                    result['errors'].append(f"Invalid email: {str(e)}")
            
            elif input_type == InputType.URL:
                if not self._is_valid_url(str(value)):
                    result['is_valid'] = False
                    result['errors'].append("Invalid URL format")
                else:
                    # Sanitize URL
                    result['sanitized_value'] = self._sanitize_url(str(value))
            
            elif input_type == InputType.PHONE:
                try:
                    parsed_number = phonenumbers.parse(str(value), None)
                    if phonenumbers.is_valid_number(parsed_number):
                        result['sanitized_value'] = phonenumbers.format_number(
                            parsed_number, phonenumbers.PhoneNumberFormat.E164
                        )
                    else:
                        result['is_valid'] = False
                        result['errors'].append("Invalid phone number")
                except NumberParseException:
                    result['is_valid'] = False
                    result['errors'].append("Invalid phone number format")
            
            elif input_type == InputType.IP_ADDRESS:
                try:
                    ip = ipaddress.ip_address(str(value))
                    result['sanitized_value'] = str(ip)
                except ValueError:
                    result['is_valid'] = False
                    result['errors'].append("Invalid IP address")
            
            elif input_type == InputType.JSON:
                try:
                    if isinstance(value, str):
                        parsed = json.loads(value)
                        result['sanitized_value'] = parsed
                except json.JSONDecodeError:
                    result['is_valid'] = False
                    result['errors'].append("Invalid JSON format")
            
            elif input_type == InputType.HTML:
                # Sanitize HTML
                result['sanitized_value'] = self._sanitize_html(str(value))
            
            elif input_type == InputType.FILENAME:
                sanitized_filename = self._sanitize_filename(str(value))
                if not sanitized_filename:
                    result['is_valid'] = False
                    result['errors'].append("Invalid filename")
                else:
                    result['sanitized_value'] = sanitized_filename
            
            elif input_type == InputType.PATH:
                if not self._is_safe_path(str(value)):
                    result['is_valid'] = False
                    result['errors'].append("Unsafe path detected")
            
            elif input_type == InputType.CREDIT_CARD:
                if not self._is_valid_credit_card(str(value)):
                    result['is_valid'] = False
                    result['errors'].append("Invalid credit card number")
                else:
                    # Mask credit card for security
                    result['sanitized_value'] = self._mask_credit_card(str(value))
            
            elif input_type == InputType.SSN:
                if not self._is_valid_ssn(str(value)):
                    result['is_valid'] = False
                    result['errors'].append("Invalid SSN format")
                else:
                    # Mask SSN for security
                    result['sanitized_value'] = self._mask_ssn(str(value))
            
            elif input_type == InputType.DATE:
                try:
                    if isinstance(value, str):
                        parsed_date = datetime.fromisoformat(value.replace('Z', '+00:00'))
                        result['sanitized_value'] = parsed_date.isoformat()
                except ValueError:
                    result['is_valid'] = False
                    result['errors'].append("Invalid date format")
            
            elif input_type == InputType.NUMBER:
                try:
                    if isinstance(value, str):
                        if '.' in value:
                            result['sanitized_value'] = float(value)
                        else:
                            result['sanitized_value'] = int(value)
                except ValueError:
                    result['is_valid'] = False
                    result['errors'].append("Invalid number format")
            
            elif input_type == InputType.BOOLEAN:
                if isinstance(value, str):
                    lower_value = value.lower()
                    if lower_value in ['true', '1', 'yes', 'on']:
                        result['sanitized_value'] = True
                    elif lower_value in ['false', '0', 'no', 'off']:
                        result['sanitized_value'] = False
                    else:
                        result['is_valid'] = False
                        result['errors'].append("Invalid boolean value")
            
            elif input_type == InputType.UUID:
                import uuid
                try:
                    uuid_obj = uuid.UUID(str(value))
                    result['sanitized_value'] = str(uuid_obj)
                except ValueError:
                    result['is_valid'] = False
                    result['errors'].append("Invalid UUID format")
            
            elif input_type == InputType.BASE64:
                try:
                    decoded = base64.b64decode(str(value), validate=True)
                    # Re-encode to ensure proper format
                    result['sanitized_value'] = base64.b64encode(decoded).decode()
                except Exception:
                    result['is_valid'] = False
                    result['errors'].append("Invalid Base64 encoding")
            
            elif input_type == InputType.TEXT:
                # Basic text sanitization
                result['sanitized_value'] = self._sanitize_text(str(value))
        
        except Exception as e:
            result['is_valid'] = False
            result['errors'].append(f"Validation error: {str(e)}")
        
        return result
    
    def _is_valid_url(self, url: str) -> bool:
        """Validate URL format"""
        url_pattern = re.compile(
            r'^https?://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
            r'localhost|'  # localhost...
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)
        return url_pattern.match(url) is not None
    
    def _sanitize_url(self, url: str) -> str:
        """Sanitize URL"""
        # Parse and reconstruct URL to remove dangerous components
        try:
            parsed = urllib.parse.urlparse(url)
            # Only allow http and https schemes
            if parsed.scheme not in ['http', 'https']:
                return ""
            
            # Reconstruct clean URL
            clean_url = urllib.parse.urlunparse((
                parsed.scheme,
                parsed.netloc,
                parsed.path,
                parsed.params,
                parsed.query,
                ""  # Remove fragment for security
            ))
            return clean_url
        except Exception:
            return ""
    
    def _sanitize_html(self, html_content: str) -> str:
        """Sanitize HTML content"""
        return bleach.clean(
            html_content,
            tags=self.allowed_html_tags,
            attributes=self.allowed_html_attributes,
            strip=True
        )
    
    def _sanitize_text(self, text: str) -> str:
        """Basic text sanitization"""
        # HTML escape
        sanitized = html.escape(text)
        
        # Remove null bytes
        sanitized = sanitized.replace('\x00', '')
        
        # Normalize whitespace
        sanitized = re.sub(r'\s+', ' ', sanitized).strip()
        
        return sanitized
    
    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename"""
        # Remove path separators and dangerous characters
        sanitized = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '', filename)
        
        # Remove leading/trailing dots and spaces
        sanitized = sanitized.strip('. ')
        
        # Limit length
        if len(sanitized) > 255:
            sanitized = sanitized[:255]
        
        # Don't allow empty or reserved names
        reserved_names = ['CON', 'PRN', 'AUX', 'NUL'] + [f'COM{i}' for i in range(1, 10)] + [f'LPT{i}' for i in range(1, 10)]
        if sanitized.upper() in reserved_names:
            sanitized = f"file_{sanitized}"
        
        return sanitized if sanitized else None
    
    def _is_safe_path(self, path: str) -> bool:
        """Check if path is safe (no directory traversal)"""
        # Check for directory traversal patterns
        dangerous_patterns = ['../', '..\\', '%2e%2e%2f', '%2e%2e%5c']
        for pattern in dangerous_patterns:
            if pattern in path.lower():
                return False
        
        # Check for absolute paths (depending on use case)
        if path.startswith('/') or (len(path) > 1 and path[1] == ':'):
            return False
        
        return True
    
    def _is_valid_credit_card(self, cc_number: str) -> bool:
        """Validate credit card using Luhn algorithm"""
        # Remove spaces and dashes
        cc_number = re.sub(r'[\s-]', '', cc_number)
        
        # Check if all digits
        if not cc_number.isdigit():
            return False
        
        # Check length
        if len(cc_number) < 13 or len(cc_number) > 19:
            return False
        
        # Luhn algorithm
        def luhn_check(card_num):
            def digits_of(n):
                return [int(d) for d in str(n)]
            
            digits = digits_of(card_num)
            odd_digits = digits[-1::-2]
            even_digits = digits[-2::-2]
            checksum = sum(odd_digits)
            for d in even_digits:
                checksum += sum(digits_of(d * 2))
            return checksum % 10 == 0
        
        return luhn_check(cc_number)
    
    def _mask_credit_card(self, cc_number: str) -> str:
        """Mask credit card number"""
        cc_number = re.sub(r'[\s-]', '', cc_number)
        if len(cc_number) >= 4:
            return '*' * (len(cc_number) - 4) + cc_number[-4:]
        return '*' * len(cc_number)
    
    def _is_valid_ssn(self, ssn: str) -> bool:
        """Validate SSN format"""
        # Remove dashes
        ssn = ssn.replace('-', '')
        
        # Check format: 9 digits, not all same, not sequential
        if not re.match(r'^\d{9}$', ssn):
            return False
        
        # Check for invalid patterns
        if ssn == '000000000' or ssn[0:3] == '000' or ssn[3:5] == '00' or ssn[5:9] == '0000':
            return False
        
        return True
    
    def _mask_ssn(self, ssn: str) -> str:
        """Mask SSN"""
        ssn = ssn.replace('-', '')
        if len(ssn) == 9:
            return f"***-**-{ssn[-4:]}"
        return '*' * len(ssn)

# Predefined validation rules for common use cases
class CommonValidationRules:
    """Common validation rules for typical use cases"""
    
    @staticmethod
    def user_registration() -> Dict[str, ValidationRule]:
        """Validation rules for user registration"""
        return {
            'username': ValidationRule(
                name='username',
                input_type=InputType.TEXT,
                required=True,
                min_length=3,
                max_length=50,
                pattern=r'^[a-zA-Z0-9_-]+$',
                error_message="Username must be 3-50 characters, alphanumeric, underscore, or dash only"
            ),
            'email': ValidationRule(
                name='email',
                input_type=InputType.EMAIL,
                required=True,
                max_length=254
            ),
            'password': ValidationRule(
                name='password',
                input_type=InputType.TEXT,
                required=True,
                min_length=8,
                max_length=128,
                pattern=r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]',
                error_message="Password must be 8+ characters with uppercase, lowercase, number, and special character"
            ),
            'phone': ValidationRule(
                name='phone',
                input_type=InputType.PHONE,
                required=False
            )
        }
    
    @staticmethod
    def api_request() -> Dict[str, ValidationRule]:
        """Validation rules for API requests"""
        return {
            'data': ValidationRule(
                name='data',
                input_type=InputType.JSON,
                required=True,
                max_length=1024 * 1024  # 1MB limit
            ),
            'format': ValidationRule(
                name='format',
                input_type=InputType.TEXT,
                required=False,
                allowed_values=['json', 'xml', 'csv']
            ),
            'callback_url': ValidationRule(
                name='callback_url',
                input_type=InputType.URL,
                required=False
            )
        }
    
    @staticmethod
    def file_upload() -> Dict[str, ValidationRule]:
        """Validation rules for file uploads"""
        return {
            'filename': ValidationRule(
                name='filename',
                input_type=InputType.FILENAME,
                required=True,
                max_length=255
            ),
            'content_type': ValidationRule(
                name='content_type',
                input_type=InputType.TEXT,
                required=True,
                allowed_values=[
                    'image/jpeg', 'image/png', 'image/gif',
                    'application/pdf', 'text/plain', 'text/csv',
                    'application/json', 'application/xml'
                ]
            ),
            'file_size': ValidationRule(
                name='file_size',
                input_type=InputType.NUMBER,
                required=True,
                custom_validator=lambda x: 0 < x <= 10 * 1024 * 1024,  # 10MB limit
                error_message="File size must be between 1 byte and 10MB"
            )
        }

def create_input_validator(config: Dict[str, Any] = None) -> InputValidator:
    """Factory function to create configured input validator"""
    return InputValidator(config or {})