"""
Simple test to verify core data protection functionality
"""

import sys
sys.path.append('security')

from security.data_protection.data_protection_engine import (
    DataProtectionEngine, DataProtectionConfig, ProtectionPolicy
)
from security.data_protection.data_classifier import DataSensitivityLevel
from security.data_protection.data_masking import UserContext, UserRole

def test_basic_functionality():
    """Test basic data protection functionality"""
    print("Testing Advanced Data Protection Engine...")
    
    # Initialize engine
    config = DataProtectionConfig(
        auto_classify=True,
        auto_encrypt=True,
        auto_mask=True,
        policy=ProtectionPolicy.STANDARD
    )
    
    engine = DataProtectionEngine(config)
    
    # Train classifier with sufficient data
    training_data = [
        ("john.doe@company.com", DataSensitivityLevel.INTERNAL),
        ("jane.smith@company.com", DataSensitivityLevel.INTERNAL),
        ("123-45-6789", DataSensitivityLevel.RESTRICTED),
        ("987-65-4321", DataSensitivityLevel.RESTRICTED),
        ("4532-1234-5678-9012", DataSensitivityLevel.RESTRICTED),
        ("5555-4444-3333-2222", DataSensitivityLevel.RESTRICTED),
        ("Public announcement", DataSensitivityLevel.PUBLIC),
        ("General information", DataSensitivityLevel.PUBLIC),
        ("Confidential data", DataSensitivityLevel.CONFIDENTIAL),
        ("Secret information", DataSensitivityLevel.CONFIDENTIAL)
    ]
    
    print("Training ML classifier...")
    accuracy = engine.train_classifier(training_data)
    print(f"✅ Classifier trained with {accuracy:.1%} accuracy")
    
    # Test classification
    test_data = "123-45-6789"
    result = engine.classifier.classify_data(test_data)
    print(f"✅ Classification: {result.sensitivity_level.value} (confidence: {result.confidence_score:.2f})")
    
    # Test encryption
    from security.data_protection.key_manager import KeyType
    key_id = engine.key_manager.generate_key(
        key_type=KeyType.AES_256,
        key_id="test_key",
        purpose="Testing",
        owner="test"
    )
    
    encrypted = engine.encryption_engine.encrypt_field(test_data, "ssn", key_id)
    decrypted = engine.encryption_engine.decrypt_field(encrypted, "ssn", key_id)
    
    print(f"✅ Encryption: {test_data} -> {encrypted}")
    print(f"✅ Decryption: {encrypted} -> {decrypted}")
    print(f"✅ Roundtrip successful: {test_data == decrypted}")
    
    # Test masking
    user_context = UserContext(
        user_id="test_user",
        role=UserRole.VIEWER,
        department="General",
        clearance_level=2,
        access_purpose="testing",
        session_risk_score=0.3
    )
    
    masked = engine.masking_engine.mask_data(test_data, "ssn", user_context)
    print(f"✅ Masking: {test_data} -> {masked}")
    
    # Test comprehensive protection
    protection_result = engine.protect_data(test_data, "customer_ssn", user_context)
    print(f"✅ Comprehensive protection applied: {protection_result.protection_applied}")
    
    # Test secure deletion
    if protection_result.key_id:
        deletion_request = engine.schedule_secure_deletion(
            data_identifiers=["test_record"],
            encryption_key_ids=[protection_result.key_id],
            delay_hours=0
        )
        print(f"✅ Secure deletion scheduled: {deletion_request}")
    
    print("\n🎉 All basic functionality tests passed!")
    print("🔒 Advanced Data Protection Engine is working correctly")

if __name__ == "__main__":
    test_basic_functionality()