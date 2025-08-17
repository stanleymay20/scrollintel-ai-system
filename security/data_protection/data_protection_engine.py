"""
Advanced Data Protection Engine

Orchestrates ML-based classification, format-preserving encryption,
dynamic masking, and secure deletion capabilities.
"""

import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import asyncio
from datetime import datetime

from .data_classifier import MLDataClassifier, ClassificationResult, DataSensitivityLevel
from .encryption_engine import FormatPreservingEncryption, EncryptionMode
from .data_masking import DynamicDataMasking, UserContext, MaskingLevel
from .key_manager import HSMKeyManager, KeyType
from .secure_deletion import CryptographicErasure, DeletionMethod

logger = logging.getLogger(__name__)

class ProtectionPolicy(Enum):
    MINIMAL = "minimal"
    STANDARD = "standard"
    HIGH_SECURITY = "high_security"
    MAXIMUM = "maximum"

@dataclass
class DataProtectionConfig:
    auto_classify: bool = True
    auto_encrypt: bool = True
    auto_mask: bool = True
    classification_threshold: float = 0.8
    encryption_key_rotation_days: int = 90
    audit_all_access: bool = True
    policy: ProtectionPolicy = ProtectionPolicy.STANDARD

@dataclass
class ProtectionResult:
    original_data: str
    classified_data: ClassificationResult
    encrypted_data: Optional[str]
    masked_data: Optional[str]
    key_id: Optional[str]
    protection_applied: List[str]
    audit_trail: List[Dict[str, Any]]

class DataProtectionEngine:
    """Advanced Data Protection Engine orchestrating all protection mechanisms"""
    
    def __init__(self, config: Optional[DataProtectionConfig] = None):
        self.config = config or DataProtectionConfig()
        
        # Initialize components
        self.key_manager = HSMKeyManager()
        self.classifier = MLDataClassifier()
        self.encryption_engine = FormatPreservingEncryption(self.key_manager)
        self.masking_engine = DynamicDataMasking()
        self.deletion_engine = CryptographicErasure(self.key_manager)
        
        # Protection policies
        self.protection_policies = self._initialize_protection_policies()
        
        # Audit trail
        self.audit_trail = []
        
        logger.info("Data Protection Engine initialized")
    
    def _initialize_protection_policies(self) -> Dict[ProtectionPolicy, Dict[str, Any]]:
        """Initialize protection policies"""
        return {
            ProtectionPolicy.MINIMAL: {
                'encrypt_restricted': True,
                'encrypt_confidential': False,
                'mask_for_external': True,
                'mask_for_viewers': False,
                'audit_access': False,
                'key_rotation_days': 365
            },
            ProtectionPolicy.STANDARD: {
                'encrypt_restricted': True,
                'encrypt_confidential': True,
                'mask_for_external': True,
                'mask_for_viewers': True,
                'audit_access': True,
                'key_rotation_days': 90
            },
            ProtectionPolicy.HIGH_SECURITY: {
                'encrypt_restricted': True,
                'encrypt_confidential': True,
                'encrypt_internal': True,
                'mask_for_external': True,
                'mask_for_viewers': True,
                'mask_for_developers': True,
                'audit_access': True,
                'key_rotation_days': 30
            },
            ProtectionPolicy.MAXIMUM: {
                'encrypt_all': True,
                'mask_for_all_except_admin': True,
                'audit_all_operations': True,
                'key_rotation_days': 7,
                'require_mfa': True,
                'require_approval': True
            }
        }
    
    def protect_data(self, data: str, field_name: str, 
                    user_context: Optional[UserContext] = None,
                    custom_config: Optional[Dict[str, Any]] = None) -> ProtectionResult:
        """Apply comprehensive data protection"""
        try:
            audit_trail = []
            protection_applied = []
            
            # Step 1: Classify data
            classification_result = None
            if self.config.auto_classify:
                classification_result = self.classifier.classify_data(data)
                audit_trail.append({
                    'timestamp': datetime.utcnow().isoformat(),
                    'operation': 'classification',
                    'sensitivity_level': classification_result.sensitivity_level.value,
                    'confidence': classification_result.confidence_score,
                    'categories': [cat.value for cat in classification_result.categories]
                })
                protection_applied.append('classification')
            
            # Step 2: Determine protection requirements
            policy_config = self.protection_policies[self.config.policy]
            if custom_config:
                policy_config.update(custom_config)
            
            # Step 3: Apply encryption if needed
            encrypted_data = None
            key_id = None
            
            if self.config.auto_encrypt and classification_result:
                should_encrypt = self._should_encrypt(classification_result, policy_config)
                
                if should_encrypt:
                    key_id = self._get_or_create_key(field_name, classification_result)
                    field_type = self._determine_field_type(field_name, classification_result)
                    
                    encrypted_data = self.encryption_engine.encrypt_field(
                        data, field_type, key_id, preserve_format=True
                    )
                    
                    audit_trail.append({
                        'timestamp': datetime.utcnow().isoformat(),
                        'operation': 'encryption',
                        'key_id': key_id,
                        'field_type': field_type,
                        'format_preserving': True
                    })
                    protection_applied.append('encryption')
            
            # Step 4: Apply masking if needed and user context provided
            masked_data = None
            if self.config.auto_mask and user_context:
                masked_data = self.masking_engine.mask_data(data, field_name, user_context)
                
                if masked_data != data:
                    audit_trail.append({
                        'timestamp': datetime.utcnow().isoformat(),
                        'operation': 'masking',
                        'user_id': user_context.user_id,
                        'user_role': user_context.role.value,
                        'masking_applied': True
                    })
                    protection_applied.append('masking')
            
            # Create result
            result = ProtectionResult(
                original_data=data,
                classified_data=classification_result,
                encrypted_data=encrypted_data,
                masked_data=masked_data,
                key_id=key_id,
                protection_applied=protection_applied,
                audit_trail=audit_trail
            )
            
            # Add to global audit trail
            if self.config.audit_all_access:
                self.audit_trail.extend(audit_trail)
            
            return result
            
        except Exception as e:
            logger.error(f"Error protecting data: {e}")
            raise
    
    def unprotect_data(self, protected_data: str, field_name: str, key_id: str,
                      user_context: UserContext) -> str:
        """Decrypt and unmask protected data"""
        try:
            # Audit access attempt
            audit_entry = {
                'timestamp': datetime.utcnow().isoformat(),
                'operation': 'data_access',
                'user_id': user_context.user_id,
                'user_role': user_context.role.value,
                'field_name': field_name,
                'key_id': key_id
            }
            
            # Decrypt data
            field_type = self._determine_field_type(field_name)
            decrypted_data = self.encryption_engine.decrypt_field(
                protected_data, field_type, key_id, preserve_format=True
            )
            
            audit_entry['decryption_successful'] = True
            
            # Apply appropriate masking based on user context
            final_data = self.masking_engine.mask_data(decrypted_data, field_name, user_context)
            
            audit_entry['masking_applied'] = final_data != decrypted_data
            
            if self.config.audit_all_access:
                self.audit_trail.append(audit_entry)
            
            return final_data
            
        except Exception as e:
            logger.error(f"Error unprotecting data: {e}")
            if self.config.audit_all_access:
                self.audit_trail.append({
                    'timestamp': datetime.utcnow().isoformat(),
                    'operation': 'data_access_failed',
                    'user_id': user_context.user_id,
                    'error': str(e)
                })
            raise
    
    def batch_protect_data(self, data_dict: Dict[str, str],
                          user_context: Optional[UserContext] = None) -> Dict[str, ProtectionResult]:
        """Batch protect multiple data fields"""
        results = {}
        
        for field_name, field_value in data_dict.items():
            try:
                result = self.protect_data(field_value, field_name, user_context)
                results[field_name] = result
            except Exception as e:
                logger.error(f"Error protecting field {field_name}: {e}")
                # Continue with other fields
        
        return results
    
    def schedule_secure_deletion(self, data_identifiers: List[str],
                               encryption_key_ids: List[str],
                               delay_hours: int = 0) -> str:
        """Schedule secure deletion of data using cryptographic erasure"""
        try:
            # Combine data identifiers with their encryption keys
            deletion_items = [
                f"{data_id}:{key_id}" 
                for data_id, key_id in zip(data_identifiers, encryption_key_ids)
            ]
            
            request_id = self.deletion_engine.schedule_deletion(
                data_identifiers=deletion_items,
                method=DeletionMethod.CRYPTOGRAPHIC_ERASURE,
                delay_hours=delay_hours,
                verification_required=True
            )
            
            # Audit deletion request
            if self.config.audit_all_access:
                self.audit_trail.append({
                    'timestamp': datetime.utcnow().isoformat(),
                    'operation': 'deletion_scheduled',
                    'request_id': request_id,
                    'data_count': len(data_identifiers),
                    'method': DeletionMethod.CRYPTOGRAPHIC_ERASURE.value
                })
            
            return request_id
            
        except Exception as e:
            logger.error(f"Error scheduling secure deletion: {e}")
            raise
    
    def train_classifier(self, training_data: List[tuple]) -> float:
        """Train the ML classifier with labeled data"""
        try:
            accuracy = self.classifier.train_model(training_data)
            
            if self.config.audit_all_access:
                self.audit_trail.append({
                    'timestamp': datetime.utcnow().isoformat(),
                    'operation': 'classifier_training',
                    'training_samples': len(training_data),
                    'accuracy': accuracy
                })
            
            return accuracy
            
        except Exception as e:
            logger.error(f"Error training classifier: {e}")
            raise
    
    def _should_encrypt(self, classification: ClassificationResult, 
                       policy_config: Dict[str, Any]) -> bool:
        """Determine if data should be encrypted based on classification and policy"""
        sensitivity = classification.sensitivity_level
        
        if policy_config.get('encrypt_all'):
            return True
        
        if sensitivity == DataSensitivityLevel.RESTRICTED and policy_config.get('encrypt_restricted'):
            return True
        
        if sensitivity == DataSensitivityLevel.CONFIDENTIAL and policy_config.get('encrypt_confidential'):
            return True
        
        if sensitivity == DataSensitivityLevel.INTERNAL and policy_config.get('encrypt_internal'):
            return True
        
        return False
    
    def _get_or_create_key(self, field_name: str, classification: ClassificationResult) -> str:
        """Get existing key or create new one for field"""
        key_id = f"field_{field_name}_{classification.sensitivity_level.value}"
        
        try:
            # Try to get existing key
            self.key_manager.get_key(key_id)
            return key_id
        except:
            # Create new key
            self.key_manager.generate_key(
                key_type=KeyType.AES_256,
                key_id=key_id,
                purpose=f"Field encryption for {field_name}",
                owner="data_protection_engine",
                expires_in_days=self.config.encryption_key_rotation_days
            )
            return key_id
    
    def _determine_field_type(self, field_name: str, 
                             classification: Optional[ClassificationResult] = None) -> str:
        """Determine field type for format-preserving encryption"""
        field_name_lower = field_name.lower()
        
        if 'credit_card' in field_name_lower or 'cc' in field_name_lower:
            return 'credit_card'
        elif 'ssn' in field_name_lower or 'social_security' in field_name_lower:
            return 'ssn'
        elif 'phone' in field_name_lower:
            return 'phone'
        elif 'email' in field_name_lower:
            return 'email'
        elif 'date' in field_name_lower:
            return 'date'
        elif any(word in field_name_lower for word in ['amount', 'salary', 'price', 'cost']):
            return 'numeric'
        else:
            return 'generic'
    
    def get_protection_statistics(self) -> Dict[str, Any]:
        """Get comprehensive protection statistics"""
        try:
            # Classification stats
            classification_stats = {
                'model_trained': self.classifier.is_trained,
                'feature_count': len(self.classifier.get_feature_importance()) if self.classifier.is_trained else 0
            }
            
            # Key management stats
            key_stats = self.key_manager.get_key_usage_stats()
            
            # Deletion stats
            deletion_stats = self.deletion_engine.get_deletion_statistics()
            
            # Audit stats
            audit_stats = {
                'total_audit_entries': len(self.audit_trail),
                'operations_audited': len(set(entry.get('operation', 'unknown') for entry in self.audit_trail))
            }
            
            return {
                'classification': classification_stats,
                'key_management': key_stats,
                'secure_deletion': deletion_stats,
                'audit': audit_stats,
                'policy': self.config.policy.value,
                'auto_protection_enabled': {
                    'classification': self.config.auto_classify,
                    'encryption': self.config.auto_encrypt,
                    'masking': self.config.auto_mask
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting protection statistics: {e}")
            return {}
    
    def export_audit_trail(self, start_date: Optional[datetime] = None,
                          end_date: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Export audit trail for compliance reporting"""
        try:
            filtered_trail = self.audit_trail
            
            if start_date or end_date:
                filtered_trail = []
                for entry in self.audit_trail:
                    entry_time = datetime.fromisoformat(entry['timestamp'])
                    
                    if start_date and entry_time < start_date:
                        continue
                    if end_date and entry_time > end_date:
                        continue
                    
                    filtered_trail.append(entry)
            
            return filtered_trail
            
        except Exception as e:
            logger.error(f"Error exporting audit trail: {e}")
            return []
    
    def validate_data_protection(self, test_data: Dict[str, str]) -> Dict[str, Any]:
        """Validate data protection implementation"""
        try:
            validation_results = {
                'classification_accuracy': 0.0,
                'encryption_success_rate': 0.0,
                'masking_effectiveness': 0.0,
                'overall_score': 0.0,
                'issues': []
            }
            
            total_tests = len(test_data)
            classification_correct = 0
            encryption_successful = 0
            masking_effective = 0
            
            for field_name, field_value in test_data.items():
                try:
                    # Test classification
                    classification = self.classifier.classify_data(field_value)
                    if classification.confidence_score >= self.config.classification_threshold:
                        classification_correct += 1
                    
                    # Test encryption/decryption
                    key_id = self._get_or_create_key(field_name, classification)
                    field_type = self._determine_field_type(field_name, classification)
                    
                    encrypted = self.encryption_engine.encrypt_field(field_value, field_type, key_id)
                    decrypted = self.encryption_engine.decrypt_field(encrypted, field_type, key_id)
                    
                    if decrypted == field_value:
                        encryption_successful += 1
                    else:
                        validation_results['issues'].append(f"Encryption/decryption failed for {field_name}")
                    
                    # Test masking
                    from .data_masking import UserContext, UserRole
                    test_context = UserContext(
                        user_id="test_user",
                        role=UserRole.VIEWER,
                        department="Test",
                        clearance_level=1,
                        access_purpose="validation",
                        session_risk_score=0.5
                    )
                    
                    masked = self.masking_engine.mask_data(field_value, field_name, test_context)
                    if masked != field_value:  # Masking should change the data
                        masking_effective += 1
                    
                except Exception as e:
                    validation_results['issues'].append(f"Validation failed for {field_name}: {str(e)}")
            
            # Calculate rates
            if total_tests > 0:
                validation_results['classification_accuracy'] = classification_correct / total_tests
                validation_results['encryption_success_rate'] = encryption_successful / total_tests
                validation_results['masking_effectiveness'] = masking_effective / total_tests
                
                validation_results['overall_score'] = (
                    validation_results['classification_accuracy'] +
                    validation_results['encryption_success_rate'] +
                    validation_results['masking_effectiveness']
                ) / 3
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Error validating data protection: {e}")
            return {'error': str(e)}
    
    def update_protection_policy(self, policy: ProtectionPolicy) -> None:
        """Update protection policy"""
        self.config.policy = policy
        
        if self.config.audit_all_access:
            self.audit_trail.append({
                'timestamp': datetime.utcnow().isoformat(),
                'operation': 'policy_update',
                'new_policy': policy.value
            })
        
        logger.info(f"Updated protection policy to {policy.value}")
    
    def cleanup_expired_keys(self) -> int:
        """Clean up expired encryption keys"""
        try:
            expired_keys = [
                key.key_id for key in self.key_manager.list_keys()
                if key.expires_at and datetime.utcnow() > key.expires_at
            ]
            
            cleaned_count = 0
            for key_id in expired_keys:
                try:
                    self.key_manager.delete_key(key_id)
                    cleaned_count += 1
                except Exception as e:
                    logger.warning(f"Failed to delete expired key {key_id}: {e}")
            
            if self.config.audit_all_access and cleaned_count > 0:
                self.audit_trail.append({
                    'timestamp': datetime.utcnow().isoformat(),
                    'operation': 'key_cleanup',
                    'keys_deleted': cleaned_count
                })
            
            return cleaned_count
            
        except Exception as e:
            logger.error(f"Error cleaning up expired keys: {e}")
            return 0