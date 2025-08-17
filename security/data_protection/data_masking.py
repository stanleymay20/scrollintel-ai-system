"""
Dynamic Data Masking System

Context-aware user-based access controls with dynamic masking.
"""

import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
import re
import random
import string
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class MaskingLevel(Enum):
    NONE = "none"
    PARTIAL = "partial"
    FULL = "full"
    REDACTED = "redacted"

class UserRole(Enum):
    ADMIN = "admin"
    ANALYST = "analyst"
    DEVELOPER = "developer"
    VIEWER = "viewer"
    EXTERNAL = "external"

@dataclass
class UserContext:
    user_id: str
    role: UserRole
    department: str
    clearance_level: int
    access_purpose: str
    session_risk_score: float

@dataclass
class MaskingRule:
    field_pattern: str
    role_permissions: Dict[UserRole, MaskingLevel]
    context_conditions: Dict[str, Any]
    custom_masker: Optional[Callable] = None

class DynamicDataMasking:
    """Context-aware dynamic data masking system"""
    
    def __init__(self):
        self.masking_rules = self._initialize_default_rules()
        self.custom_maskers = self._initialize_custom_maskers()
        
    def _initialize_default_rules(self) -> Dict[str, MaskingRule]:
        """Initialize default masking rules"""
        return {
            'ssn': MaskingRule(
                field_pattern=r'\d{3}-\d{2}-\d{4}',
                role_permissions={
                    UserRole.ADMIN: MaskingLevel.NONE,
                    UserRole.ANALYST: MaskingLevel.PARTIAL,
                    UserRole.DEVELOPER: MaskingLevel.FULL,
                    UserRole.VIEWER: MaskingLevel.FULL,
                    UserRole.EXTERNAL: MaskingLevel.REDACTED
                },
                context_conditions={'clearance_level': 3}
            ),
            'credit_card': MaskingRule(
                field_pattern=r'\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}',
                role_permissions={
                    UserRole.ADMIN: MaskingLevel.NONE,
                    UserRole.ANALYST: MaskingLevel.PARTIAL,
                    UserRole.DEVELOPER: MaskingLevel.FULL,
                    UserRole.VIEWER: MaskingLevel.FULL,
                    UserRole.EXTERNAL: MaskingLevel.REDACTED
                },
                context_conditions={'clearance_level': 4}
            ),
            'email': MaskingRule(
                field_pattern=r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
                role_permissions={
                    UserRole.ADMIN: MaskingLevel.NONE,
                    UserRole.ANALYST: MaskingLevel.PARTIAL,
                    UserRole.DEVELOPER: MaskingLevel.PARTIAL,
                    UserRole.VIEWER: MaskingLevel.FULL,
                    UserRole.EXTERNAL: MaskingLevel.REDACTED
                },
                context_conditions={'clearance_level': 2}
            ),
            'phone': MaskingRule(
                field_pattern=r'\d{3}-\d{3}-\d{4}',
                role_permissions={
                    UserRole.ADMIN: MaskingLevel.NONE,
                    UserRole.ANALYST: MaskingLevel.PARTIAL,
                    UserRole.DEVELOPER: MaskingLevel.PARTIAL,
                    UserRole.VIEWER: MaskingLevel.FULL,
                    UserRole.EXTERNAL: MaskingLevel.REDACTED
                },
                context_conditions={'clearance_level': 2}
            ),
            'salary': MaskingRule(
                field_pattern=r'\$\d+(?:,\d{3})*(?:\.\d{2})?',
                role_permissions={
                    UserRole.ADMIN: MaskingLevel.NONE,
                    UserRole.ANALYST: MaskingLevel.PARTIAL,
                    UserRole.DEVELOPER: MaskingLevel.FULL,
                    UserRole.VIEWER: MaskingLevel.FULL,
                    UserRole.EXTERNAL: MaskingLevel.REDACTED
                },
                context_conditions={'clearance_level': 3, 'department': 'HR'}
            ),
            'medical_record': MaskingRule(
                field_pattern=r'(patient|diagnosis|treatment|prescription)',
                role_permissions={
                    UserRole.ADMIN: MaskingLevel.NONE,
                    UserRole.ANALYST: MaskingLevel.FULL,
                    UserRole.DEVELOPER: MaskingLevel.FULL,
                    UserRole.VIEWER: MaskingLevel.REDACTED,
                    UserRole.EXTERNAL: MaskingLevel.REDACTED
                },
                context_conditions={'clearance_level': 5, 'department': 'Healthcare'}
            )
        }
    
    def _initialize_custom_maskers(self) -> Dict[str, Callable]:
        """Initialize custom masking functions"""
        return {
            'preserve_analytics': self._analytics_preserving_mask,
            'consistent_hash': self._consistent_hash_mask,
            'format_preserving': self._format_preserving_mask,
            'statistical_noise': self._statistical_noise_mask
        }
    
    def mask_data(self, data: str, field_name: str, user_context: UserContext) -> str:
        """Apply dynamic masking based on user context"""
        try:
            # Find applicable masking rule
            masking_rule = self._find_masking_rule(field_name, data)
            
            if not masking_rule:
                return data  # No masking rule found
            
            # Check context conditions
            if not self._check_context_conditions(masking_rule, user_context):
                return "[ACCESS DENIED]"
            
            # Determine masking level
            masking_level = self._determine_masking_level(masking_rule, user_context)
            
            # Apply masking
            return self._apply_masking(data, field_name, masking_level, masking_rule)
            
        except Exception as e:
            logger.error(f"Error masking data: {e}")
            return "[MASKING ERROR]"
    
    def _find_masking_rule(self, field_name: str, data: str) -> Optional[MaskingRule]:
        """Find applicable masking rule for field"""
        # Check field name first
        if field_name.lower() in self.masking_rules:
            return self.masking_rules[field_name.lower()]
        
        # Check data patterns
        for rule_name, rule in self.masking_rules.items():
            if re.search(rule.field_pattern, data, re.IGNORECASE):
                return rule
        
        return None
    
    def _check_context_conditions(self, rule: MaskingRule, context: UserContext) -> bool:
        """Check if user context meets rule conditions"""
        conditions = rule.context_conditions
        
        # Check clearance level
        if 'clearance_level' in conditions:
            if context.clearance_level < conditions['clearance_level']:
                return False
        
        # Check department
        if 'department' in conditions:
            if context.department != conditions['department']:
                return False
        
        # Check session risk score
        if 'max_risk_score' in conditions:
            if context.session_risk_score > conditions['max_risk_score']:
                return False
        
        # Check access purpose
        if 'allowed_purposes' in conditions:
            if context.access_purpose not in conditions['allowed_purposes']:
                return False
        
        return True
    
    def _determine_masking_level(self, rule: MaskingRule, context: UserContext) -> MaskingLevel:
        """Determine appropriate masking level"""
        base_level = rule.role_permissions.get(context.role, MaskingLevel.FULL)
        
        # Adjust based on risk score
        if context.session_risk_score > 0.8:
            # High risk - increase masking
            if base_level == MaskingLevel.NONE:
                return MaskingLevel.PARTIAL
            elif base_level == MaskingLevel.PARTIAL:
                return MaskingLevel.FULL
        elif context.session_risk_score < 0.2:
            # Low risk - potentially reduce masking
            if base_level == MaskingLevel.FULL and context.clearance_level >= 4:
                return MaskingLevel.PARTIAL
        
        return base_level
    
    def _apply_masking(self, data: str, field_name: str, level: MaskingLevel, 
                      rule: MaskingRule) -> str:
        """Apply the specified masking level"""
        if level == MaskingLevel.NONE:
            return data
        elif level == MaskingLevel.REDACTED:
            return "[REDACTED]"
        
        # Use custom masker if available
        if rule.custom_masker:
            return rule.custom_masker(data, level)
        
        # Apply standard masking based on field type
        if 'ssn' in field_name.lower() or re.match(r'\d{3}-\d{2}-\d{4}', data):
            return self._mask_ssn(data, level)
        elif 'credit_card' in field_name.lower() or re.match(r'\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}', data):
            return self._mask_credit_card(data, level)
        elif 'email' in field_name.lower() or '@' in data:
            return self._mask_email(data, level)
        elif 'phone' in field_name.lower() or re.match(r'\d{3}-\d{3}-\d{4}', data):
            return self._mask_phone(data, level)
        elif 'salary' in field_name.lower() or data.startswith('$'):
            return self._mask_currency(data, level)
        else:
            return self._mask_generic(data, level)
    
    def _mask_ssn(self, ssn: str, level: MaskingLevel) -> str:
        """Mask SSN based on level"""
        if level == MaskingLevel.PARTIAL:
            return f"XXX-XX-{ssn[-4:]}"
        else:  # FULL
            return "XXX-XX-XXXX"
    
    def _mask_credit_card(self, cc: str, level: MaskingLevel) -> str:
        """Mask credit card based on level"""
        clean_cc = re.sub(r'[-\s]', '', cc)
        if level == MaskingLevel.PARTIAL:
            masked = f"{clean_cc[:4]}{'*' * (len(clean_cc) - 8)}{clean_cc[-4:]}"
        else:  # FULL
            masked = '*' * len(clean_cc)
        
        # Restore original format
        if '-' in cc:
            return f"{masked[:4]}-{masked[4:8]}-{masked[8:12]}-{masked[12:]}"
        elif ' ' in cc:
            return f"{masked[:4]} {masked[4:8]} {masked[8:12]} {masked[12:]}"
        return masked
    
    def _mask_email(self, email: str, level: MaskingLevel) -> str:
        """Mask email based on level"""
        local, domain = email.split('@', 1)
        
        if level == MaskingLevel.PARTIAL:
            if len(local) > 2:
                masked_local = local[0] + '*' * (len(local) - 2) + local[-1]
            else:
                masked_local = '*' * len(local)
            return f"{masked_local}@{domain}"
        else:  # FULL
            return f"{'*' * len(local)}@{domain}"
    
    def _mask_phone(self, phone: str, level: MaskingLevel) -> str:
        """Mask phone based on level"""
        if level == MaskingLevel.PARTIAL:
            return f"XXX-XXX-{phone[-4:]}"
        else:  # FULL
            return "XXX-XXX-XXXX"
    
    def _mask_currency(self, amount: str, level: MaskingLevel) -> str:
        """Mask currency amount based on level"""
        if level == MaskingLevel.PARTIAL:
            # Show order of magnitude
            try:
                value = float(amount.replace('$', '').replace(',', ''))
                if value >= 1000000:
                    return "$1M+"
                elif value >= 100000:
                    return "$100K+"
                elif value >= 10000:
                    return "$10K+"
                else:
                    return "$X,XXX"
            except:
                return "$X,XXX"
        else:  # FULL
            return "$XXXXX"
    
    def _mask_generic(self, data: str, level: MaskingLevel) -> str:
        """Generic masking for unknown data types"""
        if level == MaskingLevel.PARTIAL:
            if len(data) <= 4:
                return '*' * len(data)
            else:
                return data[:2] + '*' * (len(data) - 4) + data[-2:]
        else:  # FULL
            return '*' * len(data)
    
    def _analytics_preserving_mask(self, data: str, level: MaskingLevel) -> str:
        """Mask data while preserving analytical properties"""
        if data.isdigit():
            # For numeric data, preserve statistical properties
            num_val = int(data)
            # Add consistent noise based on hash
            hash_val = hash(data) % 1000
            masked_val = num_val + hash_val
            return str(masked_val)
        
        # For text, use consistent substitution
        return self._consistent_hash_mask(data, level)
    
    def _consistent_hash_mask(self, data: str, level: MaskingLevel) -> str:
        """Create consistent hash-based mask"""
        hash_val = abs(hash(data))
        
        if data.isdigit():
            # Preserve numeric format
            masked_digits = str(hash_val)[:len(data)].zfill(len(data))
            return masked_digits
        elif data.isalpha():
            # Preserve alphabetic format
            chars = string.ascii_letters
            masked_chars = ''.join(chars[hash_val % len(chars)] for _ in range(len(data)))
            return masked_chars
        else:
            # Mixed format
            return f"HASH_{hash_val % 10000:04d}"
    
    def _format_preserving_mask(self, data: str, level: MaskingLevel) -> str:
        """Format-preserving masking"""
        result = ""
        for char in data:
            if char.isdigit():
                result += str(random.randint(0, 9))
            elif char.isalpha():
                if char.isupper():
                    result += random.choice(string.ascii_uppercase)
                else:
                    result += random.choice(string.ascii_lowercase)
            else:
                result += char
        return result
    
    def _statistical_noise_mask(self, data: str, level: MaskingLevel) -> str:
        """Add statistical noise while preserving data utility"""
        if data.isdigit():
            num_val = int(data)
            # Add noise proportional to value
            noise_factor = 0.1 if level == MaskingLevel.PARTIAL else 0.3
            noise = int(num_val * noise_factor * random.uniform(-1, 1))
            return str(max(0, num_val + noise))
        
        return self._mask_generic(data, level)
    
    def add_masking_rule(self, rule_name: str, rule: MaskingRule) -> None:
        """Add custom masking rule"""
        self.masking_rules[rule_name] = rule
        logger.info(f"Added masking rule: {rule_name}")
    
    def batch_mask_data(self, data_dict: Dict[str, str], 
                       user_context: UserContext) -> Dict[str, str]:
        """Batch mask multiple fields"""
        masked_data = {}
        
        for field_name, field_value in data_dict.items():
            masked_data[field_name] = self.mask_data(field_value, field_name, user_context)
        
        return masked_data
    
    def get_masking_preview(self, data: str, field_name: str) -> Dict[UserRole, str]:
        """Preview masking for different user roles"""
        preview = {}
        
        for role in UserRole:
            context = UserContext(
                user_id="preview",
                role=role,
                department="General",
                clearance_level=3,
                access_purpose="preview",
                session_risk_score=0.5
            )
            preview[role] = self.mask_data(data, field_name, context)
        
        return preview
    
    def audit_masking_access(self, user_context: UserContext, field_name: str, 
                           original_data: str, masked_data: str) -> None:
        """Audit masking access for compliance"""
        audit_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'user_id': user_context.user_id,
            'role': user_context.role.value,
            'field_name': field_name,
            'masking_applied': original_data != masked_data,
            'session_risk_score': user_context.session_risk_score,
            'access_purpose': user_context.access_purpose
        }
        
        logger.info(f"Masking audit: {audit_entry}")
        # In production, this would be sent to audit system