"""
Test suite for AnonymizationEngine

Tests data anonymization techniques, privacy risk assessment,
and privacy protection functionality.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import Mock, patch

from ai_data_readiness.engines.anonymization_engine import (
    AnonymizationEngine, AnonymizationTechnique, PrivacyRiskLevel,
    AnonymizationConfig, PrivacyRiskAssessment, AnonymizationResult,
    create_anonymization_engine
)
from ai_data_readiness.models.compliance_models import SensitiveDataType


class TestAnonymizationEngine:
    """Test cases for AnonymizationEngine"""
    
    @pytest.fixture
    def engine(self):
        """Create AnonymizationEngine instance for testing"""
        return AnonymizationEngine()
    
    @pytest.fixture
    def sample_data(self):
        """Create sample dataset for testing"""
        return pd.DataFrame({
            'user_id': ['U001', 'U002', 'U003', 'U004', 'U005'],
            'email': ['john@example.com', 'jane@test.com', 'bob@demo.org', 'alice@sample.net', 'charlie@test.co'],
            'age': [25, 34, 45, 28, 52],
            'salary': [50000, 75000, 90000, 65000, 120000],
            'department': ['Engineering', 'Marketing', 'Sales', 'Engineering', 'Executive']
        })
    
    @pytest.fixture
    def sensitive_data(self):
        """Create dataset with sensitive information"""
        return pd.DataFrame({
            'ssn': ['123-45-6789', '987-65-4321', '555-44-3333', '111-22-3333'],
            'phone': ['555-123-4567', '555-987-6543', '555-555-5555', '555-111-2222'],
            'medical_record': ['MR001', 'MR002', 'MR003', 'MR004'],
            'diagnosis': ['Diabetes', 'Hypertension', 'Asthma', 'Arthritis']
        })
    
    def test_engine_initialization(self, engine):
        """Test engine initializes correctly"""
        assert engine is not None
        assert engine.salt is not None
        assert len(engine.salt) == 32  # 16 bytes hex = 32 chars
        assert isinstance(engine.anonymization_cache, dict)
    
    def test_factory_function(self):
        """Test factory function creates engine"""
        engine = create_anonymization_engine()
        assert isinstance(engine, AnonymizationEngine)
    
    def test_pseudonymization(self, engine, sample_data):
        """Test pseudonymization technique"""
        config = AnonymizationConfig(
            technique=AnonymizationTechnique.PSEUDONYMIZATION,
            parameters={},
            target_columns=['user_id', 'email']
        )
        
        result = engine.anonymize_data(sample_data, config)
        
        assert isinstance(result, AnonymizationResult)
        assert result.technique_applied == AnonymizationTechnique.PSEUDONYMIZATION
        assert result.anonymized_data.shape == sample_data.shape
        
        # Check that values are pseudonymized (different from original)
        assert not result.anonymized_data['user_id'].equals(sample_data['user_id'])
        assert not result.anonymized_data['email'].equals(sample_data['email'])
        
        # Check that non-target columns are unchanged
        assert result.anonymized_data['age'].equals(sample_data['age'])
        assert result.anonymized_data['department'].equals(sample_data['department'])
    
    def test_masking(self, engine, sample_data):
        """Test data masking technique"""
        config = AnonymizationConfig(
            technique=AnonymizationTechnique.MASKING,
            parameters={
                'mask_char': '*',
                'preserve_length': True,
                'preserve_format': True
            },
            target_columns=['email']
        )
        
        result = engine.anonymize_data(sample_data, config)
        
        assert result.technique_applied == AnonymizationTechnique.MASKING
        
        # Check that emails are masked but format is preserved
        masked_emails = result.anonymized_data['email']
        for email in masked_emails:
            assert '@' in email  # Format preserved
            assert '*' in email  # Masking applied
    
    def test_k_anonymity(self, engine, sample_data):
        """Test k-anonymity technique"""
        config = AnonymizationConfig(
            technique=AnonymizationTechnique.K_ANONYMITY,
            parameters={'k': 3},
            target_columns=['age', 'salary']
        )
        
        result = engine.anonymize_data(sample_data, config)
        
        assert result.technique_applied == AnonymizationTechnique.K_ANONYMITY
        
        # Check that numerical values are generalized
        age_values = result.anonymized_data['age']
        assert all('-' in str(val) for val in age_values if pd.notna(val))
    
    def test_generalization(self, engine, sample_data):
        """Test data generalization technique"""
        config = AnonymizationConfig(
            technique=AnonymizationTechnique.GENERALIZATION,
            parameters={'generalization_levels': 3},
            target_columns=['age', 'department']
        )
        
        result = engine.anonymize_data(sample_data, config)
        
        assert result.technique_applied == AnonymizationTechnique.GENERALIZATION
        assert result.anonymized_data.shape == sample_data.shape
    
    def test_suppression(self, engine, sample_data):
        """Test data suppression technique"""
        config = AnonymizationConfig(
            technique=AnonymizationTechnique.SUPPRESSION,
            parameters={'suppression_rate': 0.4},
            target_columns=['user_id']
        )
        
        result = engine.anonymize_data(sample_data, config)
        
        assert result.technique_applied == AnonymizationTechnique.SUPPRESSION
        
        # Check that some values are suppressed (None)
        suppressed_count = result.anonymized_data['user_id'].isna().sum()
        assert suppressed_count > 0
    
    def test_differential_privacy(self, engine, sample_data):
        """Test differential privacy technique"""
        config = AnonymizationConfig(
            technique=AnonymizationTechnique.DIFFERENTIAL_PRIVACY,
            parameters={'epsilon': 1.0, 'sensitivity': 1.0},
            target_columns=['age', 'salary']
        )
        
        result = engine.anonymize_data(sample_data, config)
        
        assert result.technique_applied == AnonymizationTechnique.DIFFERENTIAL_PRIVACY
        
        # Check that noise has been added (values should be different)
        assert not result.anonymized_data['age'].equals(sample_data['age'])
        assert not result.anonymized_data['salary'].equals(sample_data['salary'])
    
    def test_synthetic_data_generation(self, engine, sample_data):
        """Test synthetic data generation"""
        config = AnonymizationConfig(
            technique=AnonymizationTechnique.SYNTHETIC_DATA,
            parameters={},
            target_columns=list(sample_data.columns)
        )
        
        result = engine.anonymize_data(sample_data, config)
        
        assert result.technique_applied == AnonymizationTechnique.SYNTHETIC_DATA
        assert result.anonymized_data.shape == sample_data.shape
        assert list(result.anonymized_data.columns) == list(sample_data.columns)
        
        # Synthetic data should be different from original
        assert not result.anonymized_data.equals(sample_data)
    
    def test_privacy_risk_assessment(self, engine, sensitive_data):
        """Test privacy risk assessment"""
        assessments = engine.assess_privacy_risk(sensitive_data)
        
        assert len(assessments) == len(sensitive_data.columns)
        
        for assessment in assessments:
            assert isinstance(assessment, PrivacyRiskAssessment)
            assert assessment.column_name in sensitive_data.columns
            assert isinstance(assessment.risk_level, PrivacyRiskLevel)
            assert 0 <= assessment.risk_score <= 1
            assert isinstance(assessment.vulnerability_factors, list)
            assert isinstance(assessment.recommended_techniques, list)
            assert isinstance(assessment.assessment_timestamp, datetime)
    
    def test_high_risk_column_detection(self, engine):
        """Test detection of high-risk columns"""
        # Create data with high uniqueness (high risk)
        high_risk_data = pd.DataFrame({
            'unique_id': [f'ID_{i:06d}' for i in range(1000)],  # Highly unique
            'common_category': ['A', 'B', 'C'] * 334  # Low uniqueness
        })
        
        assessments = engine.assess_privacy_risk(high_risk_data)
        
        # unique_id should have higher risk than common_category
        unique_id_assessment = next(a for a in assessments if a.column_name == 'unique_id')
        category_assessment = next(a for a in assessments if a.column_name == 'common_category')
        
        assert unique_id_assessment.risk_score > category_assessment.risk_score
    
    def test_sensitive_pattern_detection(self, engine):
        """Test detection of sensitive data patterns"""
        pattern_data = pd.DataFrame({
            'emails': ['user1@example.com', 'user2@test.org', 'user3@demo.net'],
            'ssns': ['123-45-6789', '987-65-4321', '555-44-3333'],
            'normal_text': ['hello', 'world', 'test']
        })
        
        assessments = engine.assess_privacy_risk(pattern_data)
        
        # Email and SSN columns should have higher risk
        email_assessment = next(a for a in assessments if a.column_name == 'emails')
        ssn_assessment = next(a for a in assessments if a.column_name == 'ssns')
        text_assessment = next(a for a in assessments if a.column_name == 'normal_text')
        
        assert email_assessment.risk_score > text_assessment.risk_score
        assert ssn_assessment.risk_score > text_assessment.risk_score
        assert 'sensitive_patterns' in email_assessment.vulnerability_factors
        assert 'sensitive_patterns' in ssn_assessment.vulnerability_factors
    
    def test_anonymization_strategy_recommendation(self, engine, sensitive_data):
        """Test anonymization strategy recommendation"""
        assessments = engine.assess_privacy_risk(sensitive_data)
        recommendations = engine.recommend_anonymization_strategy(assessments)
        
        assert len(recommendations) > 0
        
        for config in recommendations:
            assert isinstance(config, AnonymizationConfig)
            assert isinstance(config.technique, AnonymizationTechnique)
            assert len(config.target_columns) > 0
            assert isinstance(config.parameters, dict)
    
    def test_critical_risk_recommendations(self, engine):
        """Test recommendations for critical risk data"""
        # Create assessment with critical risk
        critical_assessment = PrivacyRiskAssessment(
            column_name='critical_data',
            risk_level=PrivacyRiskLevel.CRITICAL,
            risk_score=0.9,
            vulnerability_factors=['high_uniqueness', 'sensitive_patterns'],
            recommended_techniques=[
                AnonymizationTechnique.SYNTHETIC_DATA,
                AnonymizationTechnique.SUPPRESSION
            ],
            assessment_timestamp=datetime.now()
        )
        
        config = engine._generate_anonymization_config(critical_assessment)
        
        assert config is not None
        assert config.technique in [
            AnonymizationTechnique.SYNTHETIC_DATA,
            AnonymizationTechnique.SUPPRESSION,
            AnonymizationTechnique.K_ANONYMITY
        ]
    
    def test_anonymization_metrics(self, engine, sample_data):
        """Test anonymization result metrics"""
        config = AnonymizationConfig(
            technique=AnonymizationTechnique.PSEUDONYMIZATION,
            parameters={},
            target_columns=['user_id']
        )
        
        result = engine.anonymize_data(sample_data, config)
        
        # Check metrics are calculated
        assert 0 <= result.privacy_gain <= 1
        assert 0 <= result.utility_loss <= 1
        assert result.processing_time > 0
        assert result.original_data_shape == sample_data.shape
        assert 'config' in result.metadata
        assert 'columns_processed' in result.metadata
        assert 'records_processed' in result.metadata
    
    def test_pseudonymization_consistency(self, engine):
        """Test that pseudonymization is consistent for same values"""
        data = pd.DataFrame({'id': ['A', 'B', 'A', 'C', 'B']})
        
        config = AnonymizationConfig(
            technique=AnonymizationTechnique.PSEUDONYMIZATION,
            parameters={},
            target_columns=['id']
        )
        
        result = engine.anonymize_data(data, config)
        
        # Same original values should have same pseudonyms
        pseudonyms = result.anonymized_data['id']
        assert pseudonyms.iloc[0] == pseudonyms.iloc[2]  # Both 'A'
        assert pseudonyms.iloc[1] == pseudonyms.iloc[4]  # Both 'B'
        assert pseudonyms.iloc[0] != pseudonyms.iloc[1]  # 'A' != 'B'
    
    def test_masking_format_preservation(self, engine):
        """Test that masking preserves format when requested"""
        data = pd.DataFrame({
            'phone': ['555-123-4567', '555-987-6543'],
            'email': ['user@domain.com', 'test@example.org']
        })
        
        config = AnonymizationConfig(
            technique=AnonymizationTechnique.MASKING,
            parameters={
                'mask_char': '*',
                'preserve_format': True
            },
            target_columns=['phone', 'email']
        )
        
        result = engine.anonymize_data(data, config)
        
        # Check format preservation
        masked_phones = result.anonymized_data['phone']
        masked_emails = result.anonymized_data['email']
        
        for phone in masked_phones:
            assert phone.count('-') == 2  # Dashes preserved
        
        for email in masked_emails:
            assert '@' in email and '.' in email  # Email format preserved
    
    def test_unsupported_technique_error(self, engine, sample_data):
        """Test error handling for unsupported techniques"""
        # Create config with invalid technique (simulate by using string)
        config = AnonymizationConfig(
            technique="invalid_technique",  # This will cause an error
            parameters={},
            target_columns=['user_id']
        )
        
        with pytest.raises(ValueError):
            engine.anonymize_data(sample_data, config)
    
    def test_empty_data_handling(self, engine):
        """Test handling of empty datasets"""
        empty_data = pd.DataFrame()
        
        config = AnonymizationConfig(
            technique=AnonymizationTechnique.PSEUDONYMIZATION,
            parameters={},
            target_columns=[]
        )
        
        result = engine.anonymize_data(empty_data, config)
        
        assert result.anonymized_data.empty
        assert result.original_data_shape == (0, 0)
    
    def test_missing_values_handling(self, engine):
        """Test handling of missing values in data"""
        data_with_nulls = pd.DataFrame({
            'id': ['A', None, 'C', None, 'E'],
            'value': [1, 2, None, 4, None]
        })
        
        config = AnonymizationConfig(
            technique=AnonymizationTechnique.MASKING,
            parameters={'mask_char': '*'},
            target_columns=['id']
        )
        
        result = engine.anonymize_data(data_with_nulls, config)
        
        # Should handle nulls gracefully
        assert result.anonymized_data.shape == data_with_nulls.shape
    
    def test_large_dataset_performance(self, engine):
        """Test performance with larger datasets"""
        # Create larger dataset
        large_data = pd.DataFrame({
            'id': [f'ID_{i}' for i in range(10000)],
            'value': np.random.randint(1, 100, 10000),
            'category': np.random.choice(['A', 'B', 'C'], 10000)
        })
        
        config = AnonymizationConfig(
            technique=AnonymizationTechnique.PSEUDONYMIZATION,
            parameters={},
            target_columns=['id']
        )
        
        result = engine.anonymize_data(large_data, config)
        
        assert result.anonymized_data.shape == large_data.shape
        assert result.processing_time < 10.0  # Should complete within 10 seconds
    
    @pytest.mark.parametrize("technique", [
        AnonymizationTechnique.PSEUDONYMIZATION,
        AnonymizationTechnique.MASKING,
        AnonymizationTechnique.GENERALIZATION,
        AnonymizationTechnique.SUPPRESSION
    ])
    def test_all_techniques_work(self, engine, sample_data, technique):
        """Test that all anonymization techniques work without errors"""
        config = AnonymizationConfig(
            technique=technique,
            parameters={},
            target_columns=['user_id']
        )
        
        result = engine.anonymize_data(sample_data, config)
        
        assert isinstance(result, AnonymizationResult)
        assert result.technique_applied == technique
        assert result.anonymized_data.shape == sample_data.shape


class TestPrivacyRiskAssessment:
    """Test privacy risk assessment functionality"""
    
    def test_risk_level_enum(self):
        """Test PrivacyRiskLevel enum values"""
        assert PrivacyRiskLevel.LOW.value == "low"
        assert PrivacyRiskLevel.MEDIUM.value == "medium"
        assert PrivacyRiskLevel.HIGH.value == "high"
        assert PrivacyRiskLevel.CRITICAL.value == "critical"
    
    def test_anonymization_technique_enum(self):
        """Test AnonymizationTechnique enum values"""
        assert AnonymizationTechnique.K_ANONYMITY.value == "k_anonymity"
        assert AnonymizationTechnique.PSEUDONYMIZATION.value == "pseudonymization"
        assert AnonymizationTechnique.MASKING.value == "masking"
        assert AnonymizationTechnique.SYNTHETIC_DATA.value == "synthetic_data"


if __name__ == '__main__':
    pytest.main([__file__])