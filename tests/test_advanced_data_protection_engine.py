"""
Tests for Advanced Data Protection Engine

Comprehensive test suite for ML classification, encryption, masking, and secure deletion.
"""

import pytest
import tempfile
import os
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from security.data_protection.data_protection_engine import (
    DataProtectionEngine, DataProtectionConfig, ProtectionPolicy
)
from security.data_protection.data_classifier import (
    MLDataClassifier, DataSensitivityLevel, DataCategory
)
from security.data_protection.encryption_engine import FormatPreservingEncryption
from security.data_protection.data_masking import (
    DynamicDataMasking, UserContext, UserRole, MaskingLevel
)
from security.data_protection.key_manager import HSMKeyManager, KeyType
from security.data_protection.secure_deletion import CryptographicErasure, DeletionMethod

class TestMLDataClassifier:
    """Test ML-based data classification"""
    
    def test_classifier_initialization(self):
        """Test classifier initializes correctly"""
        classifier = MLDataClassifier()
        assert classifier is not None
        assert not classifier.is_trained
        assert len(classifier.pattern_matchers) > 0
    
    def test_pattern_matching(self):
        """Test pattern-based classification"""
        classifier = MLDataClassifier()
        
        # Test SSN pattern
        result = classifier.classify_data("123-45-6789")
        assert DataCategory.PII in result.categories
        assert result.sensitivity_level == DataSensitivityLevel.RESTRICTED
        
        # Test credit card pattern
        result = classifier.classify_data("4532-1234-5678-9012")
        assert DataCategory.PCI in result.categories
        assert result.sensitivity_level == DataSensitivityLevel.RESTRICTED
        
        # Test email pattern
        result = classifier.classify_data("john.doe@company.com")
        assert DataCategory.PII in result.categories
    
    def test_ml_training(self):
        """Test ML model training"""
        classifier = MLDataClassifier()
        
        training_data = [
            ("john.doe@email.com", DataSensitivityLevel.INTERNAL),
            ("123-45-6789", DataSensitivityLevel.RESTRICTED),
            ("public information", DataSensitivityLevel.PUBLIC),
            ("confidential data", DataSensitivityLevel.CONFIDENTIAL),
            ("restricted access", DataSensitivityLevel.RESTRICTED)
        ] * 10  # Repeat to have enough samples
        
        accuracy = classifier.train_model(training_data)
        assert accuracy > 0.0
        assert classifier.is_trained
    
    def test_batch_classification(self):
        """Test batch classification"""
        classifier = MLDataClassifier()
        
        data_samples = [
            "test@email.com",
            "555-123-4567",
            "public announcement"
        ]
        
        results = classifier.batch_classify(data_samples)
        assert len(results) == len(data_samples)
        
        for result in results:
            assert result.confidence_score >= 0.0
            assert result.sensitivity_level in DataSensitivityLevel

class TestFormatPreservingEncryption:
    """Test format-preserving encryption"""
    
    def setup_method(self):
        """Setup test environment"""
        self.key_manager = HSMKeyManager()
        self.encryption_engine = FormatPreservingEncryption(self.key_manager)
        
        # Generate test key
        self.test_key_id = self.key_manager.generate_key(
            key_type=KeyType.AES_256,
            key_id="test_encryption_key",
            purpose="Testing",
            owner="test"
        )
    
    def test_credit_card_encryption(self):
        """Test credit card format-preserving encryption"""
        original = "4532-1234-5678-9012"
        
        encrypted = self.encryption_engine.encrypt_field(
            original, "credit_card", self.test_key_id, preserve_format=True
        )
        
        decrypted = self.encryption_engine.decrypt_field(
            encrypted, "credit_card", self.test_key_id, preserve_format=True
        )
        
        # Format should be preserved
        assert len(encrypted) == len(original)
        assert encrypted.count('-') == original.count('-')
        
        # Roundtrip should work
        assert decrypted == original
        
        # Encrypted should be different
        assert encrypted != original
    
    def test_ssn_encryption(self):
        """Test SSN format-preserving encryption"""
        original = "123-45-6789"
        
        encrypted = self.encryption_engine.encrypt_field(
            original, "ssn", self.test_key_id, preserve_format=True
        )
        
        decrypted = self.encryption_engine.decrypt_field(
            encrypted, "ssn", self.test_key_id, preserve_format=True
        )
        
        assert len(encrypted) == len(original)
        assert encrypted.count('-') == original.count('-')
        assert decrypted == original
        assert encrypted != original
    
    def test_email_encryption(self):
        """Test email format-preserving encryption"""
        original = "john.doe@company.com"
        
        encrypted = self.encryption_engine.encrypt_field(
            original, "email", self.test_key_id, preserve_format=True
        )
        
        decrypted = self.encryption_engine.decrypt_field(
            encrypted, "email", self.test_key_id, preserve_format=True
        )
        
        # Should preserve @ symbol and domain
        assert '@' in encrypted
        assert encrypted.endswith('@company.com')
        assert decrypted == original
    
    def test_batch_encryption(self):
        """Test batch encryption/decryption"""
        data_dict = {
            "ssn": "123-45-6789",
            "credit_card": "4532-1234-5678-9012",
            "phone": "555-123-4567"
        }
        
        field_types = {
            "ssn": "ssn",
            "credit_card": "credit_card", 
            "phone": "phone"
        }
        
        encrypted_dict = self.encryption_engine.batch_encrypt(
            data_dict, field_types, self.test_key_id
        )
        
        decrypted_dict = self.encryption_engine.batch_decrypt(
            encrypted_dict, field_types, self.test_key_id
        )
        
        assert decrypted_dict == data_dict
        
        # All fields should be encrypted (different from original)
        for field_name in data_dict:
            assert encrypted_dict[field_name] != data_dict[field_name]

class TestDynamicDataMasking:
    """Test dynamic data masking"""
    
    def setup_method(self):
        """Setup test environment"""
        self.masking_engine = DynamicDataMasking()
    
    def test_role_based_masking(self):
        """Test masking based on user roles"""
        data = "123-45-6789"
        field_name = "ssn"
        
        # Admin should see unmasked data
        admin_context = UserContext(
            user_id="admin",
            role=UserRole.ADMIN,
            department="Security",
            clearance_level=5,
            access_purpose="administration",
            session_risk_score=0.1
        )
        
        admin_result = self.masking_engine.mask_data(data, field_name, admin_context)
        assert admin_result == data  # No masking for admin
        
        # External user should see heavily masked data
        external_context = UserContext(
            user_id="external",
            role=UserRole.EXTERNAL,
            department="External",
            clearance_level=0,
            access_purpose="external_access",
            session_risk_score=0.8
        )
        
        external_result = self.masking_engine.mask_data(data, field_name, external_context)
        assert external_result != data  # Should be masked
        # External users with low clearance get access denied
        assert external_result in ["[REDACTED]", "[ACCESS DENIED]"]
    
    def test_context_conditions(self):
        """Test context-based masking conditions"""
        data = "123-45-6789"
        field_name = "ssn"
        
        # Low clearance should be denied access
        low_clearance_context = UserContext(
            user_id="user",
            role=UserRole.VIEWER,
            department="General",
            clearance_level=1,  # Below required level
            access_purpose="viewing",
            session_risk_score=0.3
        )
        
        result = self.masking_engine.mask_data(data, field_name, low_clearance_context)
        assert result == "[ACCESS DENIED]"
    
    def test_risk_score_adjustment(self):
        """Test masking adjustment based on session risk score"""
        data = "john.doe@company.com"
        field_name = "email"
        
        # Low risk context
        low_risk_context = UserContext(
            user_id="analyst",
            role=UserRole.ANALYST,
            department="Analytics",
            clearance_level=3,
            access_purpose="analysis",
            session_risk_score=0.1  # Low risk
        )
        
        # High risk context
        high_risk_context = UserContext(
            user_id="analyst",
            role=UserRole.ANALYST,
            department="Analytics", 
            clearance_level=3,
            access_purpose="analysis",
            session_risk_score=0.9  # High risk
        )
        
        low_risk_result = self.masking_engine.mask_data(data, field_name, low_risk_context)
        high_risk_result = self.masking_engine.mask_data(data, field_name, high_risk_context)
        
        # High risk should result in more masking
        assert len(low_risk_result.replace('*', '')) >= len(high_risk_result.replace('*', ''))
    
    def test_batch_masking(self):
        """Test batch data masking"""
        data_dict = {
            "ssn": "123-45-6789",
            "email": "john.doe@company.com",
            "phone": "555-123-4567"
        }
        
        context = UserContext(
            user_id="viewer",
            role=UserRole.VIEWER,
            department="General",
            clearance_level=2,
            access_purpose="viewing",
            session_risk_score=0.5
        )
        
        masked_dict = self.masking_engine.batch_mask_data(data_dict, context)
        
        assert len(masked_dict) == len(data_dict)
        
        # All fields should be masked for viewer
        for field_name in data_dict:
            assert masked_dict[field_name] != data_dict[field_name]

class TestHSMKeyManager:
    """Test HSM key management"""
    
    def setup_method(self):
        """Setup test environment"""
        self.key_manager = HSMKeyManager()
    
    def test_key_generation(self):
        """Test key generation"""
        key_id = self.key_manager.generate_key(
            key_type=KeyType.AES_256,
            key_id="test_key_001",
            purpose="Testing",
            owner="test_user"
        )
        
        assert key_id == "test_key_001"
        
        # Should be able to retrieve the key
        key_material = self.key_manager.get_key(key_id)
        assert len(key_material) == 32  # 256 bits = 32 bytes
    
    def test_key_rotation(self):
        """Test key rotation"""
        original_key_id = self.key_manager.generate_key(
            key_type=KeyType.AES_256,
            key_id="rotation_test_key",
            purpose="Rotation testing",
            owner="test_user"
        )
        
        # Rotate the key
        new_key_id = self.key_manager.rotate_key(original_key_id)
        
        assert new_key_id != original_key_id
        
        # Original key should be inactive
        original_metadata = self.key_manager.get_key_metadata(original_key_id)
        assert original_metadata.status.value == "inactive"
        
        # New key should be active
        new_metadata = self.key_manager.get_key_metadata(new_key_id)
        assert new_metadata.status.value == "active"
    
    def test_key_deletion(self):
        """Test key deletion"""
        key_id = self.key_manager.generate_key(
            key_type=KeyType.AES_256,
            key_id="deletion_test_key",
            purpose="Deletion testing",
            owner="test_user"
        )
        
        # Delete the key
        self.key_manager.delete_key(key_id, force=True)
        
        # Should not be able to retrieve deleted key
        with pytest.raises(ValueError):
            self.key_manager.get_key(key_id)
    
    def test_key_backup_restore(self):
        """Test key backup and restore"""
        # Generate test keys
        key_ids = []
        for i in range(3):
            key_id = self.key_manager.generate_key(
                key_type=KeyType.AES_256,
                key_id=f"backup_test_key_{i}",
                purpose="Backup testing",
                owner="test_user"
            )
            key_ids.append(key_id)
        
        # Backup keys
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            backup_path = f.name
        
        try:
            self.key_manager.backup_keys(backup_path)
            
            # Clear current keys
            for key_id in key_ids:
                self.key_manager.delete_key(key_id, force=True)
            
            # Restore from backup
            self.key_manager.restore_keys(backup_path)
            
            # Verify keys are restored
            for key_id in key_ids:
                key_material = self.key_manager.get_key(key_id)
                assert len(key_material) == 32
                
        finally:
            os.unlink(backup_path)

class TestCryptographicErasure:
    """Test secure deletion with cryptographic erasure"""
    
    def setup_method(self):
        """Setup test environment"""
        self.key_manager = HSMKeyManager()
        self.deletion_engine = CryptographicErasure(self.key_manager)
    
    def test_cryptographic_erasure(self):
        """Test cryptographic erasure"""
        # Generate test key
        key_id = self.key_manager.generate_key(
            key_type=KeyType.AES_256,
            key_id="erasure_test_key",
            purpose="Erasure testing",
            owner="test_user"
        )
        
        # Verify key exists
        key_material = self.key_manager.get_key(key_id)
        assert len(key_material) == 32
        
        # Perform cryptographic erasure
        result = self.deletion_engine.execute_cryptographic_erasure(
            "test_data", key_id
        )
        
        assert result.status.value == "completed"
        assert result.verification_hash is not None
        
        # Key should no longer exist
        with pytest.raises(ValueError):
            self.key_manager.get_key(key_id)
    
    def test_secure_file_overwrite(self):
        """Test secure file overwrite"""
        # Create temporary test file
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("Sensitive data that must be securely deleted")
            test_file_path = f.name
        
        try:
            # Verify file exists
            assert os.path.exists(test_file_path)
            
            # Perform secure overwrite
            result = self.deletion_engine.execute_secure_overwrite(test_file_path)
            
            assert result.status.value == "completed"
            assert result.verification_hash is not None
            
            # File should be deleted
            assert not os.path.exists(test_file_path)
            
        finally:
            # Cleanup if test failed
            if os.path.exists(test_file_path):
                os.unlink(test_file_path)
    
    def test_deletion_scheduling(self):
        """Test deletion scheduling"""
        # Generate test keys
        key_ids = []
        for i in range(2):
            key_id = self.key_manager.generate_key(
                key_type=KeyType.AES_256,
                key_id=f"scheduled_deletion_key_{i}",
                purpose="Scheduled deletion testing",
                owner="test_user"
            )
            key_ids.append(key_id)
        
        # Schedule deletion
        data_identifiers = [f"data_{i}:{key_ids[i]}" for i in range(2)]
        
        request_id = self.deletion_engine.schedule_deletion(
            data_identifiers=data_identifiers,
            method=DeletionMethod.CRYPTOGRAPHIC_ERASURE,
            delay_hours=0,  # Immediate
            verification_required=True
        )
        
        assert request_id is not None
        
        # Check status
        import time
        time.sleep(1)  # Wait for processing
        
        status = self.deletion_engine.get_deletion_status(request_id)
        assert status is not None
        assert status.status.value in ["completed", "in_progress"]

class TestDataProtectionEngine:
    """Test comprehensive data protection engine"""
    
    def setup_method(self):
        """Setup test environment"""
        config = DataProtectionConfig(
            auto_classify=True,
            auto_encrypt=True,
            auto_mask=True,
            policy=ProtectionPolicy.STANDARD
        )
        self.engine = DataProtectionEngine(config)
        
        # Train classifier with sufficient data for each class
        training_data = [
            ("test@email.com", DataSensitivityLevel.INTERNAL),
            ("user@company.com", DataSensitivityLevel.INTERNAL),
            ("123-45-6789", DataSensitivityLevel.RESTRICTED),
            ("987-65-4321", DataSensitivityLevel.RESTRICTED),
            ("public info", DataSensitivityLevel.PUBLIC),
            ("general announcement", DataSensitivityLevel.PUBLIC),
            ("confidential data", DataSensitivityLevel.CONFIDENTIAL),
            ("secret information", DataSensitivityLevel.CONFIDENTIAL)
        ]
        self.engine.train_classifier(training_data)
    
    def test_comprehensive_protection(self):
        """Test comprehensive data protection workflow"""
        test_data = "123-45-6789"  # SSN
        field_name = "customer_ssn"
        
        user_context = UserContext(
            user_id="test_user",
            role=UserRole.ANALYST,
            department="Analytics",
            clearance_level=3,
            access_purpose="analysis",
            session_risk_score=0.3
        )
        
        result = self.engine.protect_data(test_data, field_name, user_context)
        
        assert result.original_data == test_data
        assert result.classified_data is not None
        assert result.classified_data.sensitivity_level == DataSensitivityLevel.RESTRICTED
        assert "classification" in result.protection_applied
        
        # Should be encrypted due to high security policy
        if result.encrypted_data:
            assert "encryption" in result.protection_applied
            assert result.key_id is not None
        
        # Should be masked for analyst role
        if result.masked_data:
            assert "masking" in result.protection_applied
            assert result.masked_data != test_data
    
    def test_batch_protection(self):
        """Test batch data protection"""
        test_data = {
            "ssn": "123-45-6789",
            "email": "john.doe@company.com",
            "public_info": "General information"
        }
        
        user_context = UserContext(
            user_id="test_user",
            role=UserRole.VIEWER,
            department="General",
            clearance_level=2,
            access_purpose="viewing",
            session_risk_score=0.4
        )
        
        results = self.engine.batch_protect_data(test_data, user_context)
        
        assert len(results) == len(test_data)
        
        for field_name, result in results.items():
            assert result.original_data == test_data[field_name]
            assert len(result.protection_applied) > 0
    
    def test_data_unprotection(self):
        """Test data unprotection (decryption + masking)"""
        # First protect some data
        test_data = "4532-1234-5678-9012"  # Credit card
        field_name = "credit_card"
        
        protection_result = self.engine.protect_data(test_data, field_name)
        
        if protection_result.encrypted_data and protection_result.key_id:
            # Now unprotect it
            user_context = UserContext(
                user_id="test_user",
                role=UserRole.ADMIN,  # Admin should see unmasked
                department="Security",
                clearance_level=5,
                access_purpose="administration",
                session_risk_score=0.1
            )
            
            unprotected = self.engine.unprotect_data(
                protection_result.encrypted_data,
                field_name,
                protection_result.key_id,
                user_context
            )
            
            # Admin should see original data
            assert unprotected == test_data
    
    def test_secure_deletion_scheduling(self):
        """Test secure deletion scheduling"""
        # Protect some data first
        test_data = ["sensitive_data_1", "sensitive_data_2"]
        field_names = ["field_1", "field_2"]
        
        protected_results = []
        for data, field_name in zip(test_data, field_names):
            result = self.engine.protect_data(data, field_name)
            protected_results.append(result)
        
        # Extract data identifiers and key IDs
        data_identifiers = [f"record_{i}" for i in range(len(test_data))]
        key_ids = [result.key_id for result in protected_results if result.key_id]
        
        if key_ids:
            # Schedule deletion
            request_id = self.engine.schedule_secure_deletion(
                data_identifiers=data_identifiers[:len(key_ids)],
                encryption_key_ids=key_ids,
                delay_hours=0
            )
            
            assert request_id is not None
    
    def test_protection_statistics(self):
        """Test protection statistics"""
        stats = self.engine.get_protection_statistics()
        
        assert 'classification' in stats
        assert 'key_management' in stats
        assert 'secure_deletion' in stats
        assert 'audit' in stats
        assert 'policy' in stats
        assert stats['policy'] == 'standard'
    
    def test_validation(self):
        """Test data protection validation"""
        test_data = {
            "test_ssn": "111-22-3333",
            "test_email": "test@validation.com",
            "test_public": "Public information"
        }
        
        validation_results = self.engine.validate_data_protection(test_data)
        
        assert 'classification_accuracy' in validation_results
        assert 'encryption_success_rate' in validation_results
        assert 'masking_effectiveness' in validation_results
        assert 'overall_score' in validation_results
        
        # Should have reasonable scores
        assert validation_results['overall_score'] >= 0.0
        assert validation_results['overall_score'] <= 1.0
    
    def test_audit_trail(self):
        """Test audit trail functionality"""
        # Perform some operations to generate audit entries
        test_data = "123-45-6789"
        field_name = "test_ssn"
        
        user_context = UserContext(
            user_id="audit_test_user",
            role=UserRole.ANALYST,
            department="Test",
            clearance_level=3,
            access_purpose="testing",
            session_risk_score=0.2
        )
        
        # This should generate audit entries
        self.engine.protect_data(test_data, field_name, user_context)
        
        # Export audit trail
        audit_trail = self.engine.export_audit_trail()
        
        assert len(audit_trail) > 0
        
        # Check audit entry structure
        for entry in audit_trail:
            assert 'timestamp' in entry
            assert 'operation' in entry

if __name__ == "__main__":
    pytest.main([__file__, "-v"])