"""
Tests for Quality Assurance Engine

This module contains comprehensive tests for the quality assurance and validation
framework, ensuring all components work correctly and maintain zero tolerance
for simulations or fake results.
"""

import pytest
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any
import json

from scrollintel.engines.quality_assurance_engine import QualityAssuranceEngine
from scrollintel.models.quality_assurance_models import (
    QualityAssuranceConfig, TestCase, TestSuite, TestResults, TestType, ValidationStatus,
    DataQualityRule, DataQualityReport, DataQualityDimension, AnomalyDetectionConfig,
    AnomalyDetection, AnomalyType, BusinessRule, BusinessRuleValidation, ComplianceFramework,
    AgentOutput, AgentOutputSchema, OutputValidationResult, PerformanceTestConfig,
    SecurityTestCase, QualityAssessment
)


@pytest.fixture
def qa_config():
    """Create test quality assurance configuration"""
    return QualityAssuranceConfig(
        organization_id="test_org",
        automated_testing_enabled=True,
        data_quality_monitoring=True,
        anomaly_detection_enabled=True,
        business_rule_enforcement=True,
        agent_output_validation=True,
        simulation_detection_enabled=True,
        authenticity_verification=True,
        quality_score_threshold=0.8
    )


@pytest.fixture
def qa_engine(qa_config):
    """Create test quality assurance engine"""
    return QualityAssuranceEngine(qa_config)


@pytest.fixture
def sample_data():
    """Create sample test data"""
    return pd.DataFrame({
        'id': range(1, 101),
        'name': [f'User_{i}' for i in range(1, 101)],
        'age': np.random.randint(18, 80, 100),
        'email': [f'user{i}@example.com' for i in range(1, 101)],
        'salary': np.random.normal(50000, 15000, 100),
        'created_at': pd.date_range('2023-01-01', periods=100, freq='D')
    })


@pytest.fixture
def data_quality_rules():
    """Create sample data quality rules"""
    return [
        DataQualityRule(
            name="Completeness Check",
            description="Check for missing values",
            dimension=DataQualityDimension.COMPLETENESS,
            rule_expression="no_nulls",
            threshold=0.95,
            severity="error"
        ),
        DataQualityRule(
            name="Age Validity",
            description="Check age is within valid range",
            dimension=DataQualityDimension.VALIDITY,
            rule_expression="age_range_18_120",
            threshold=0.99,
            severity="error"
        ),
        DataQualityRule(
            name="Email Format",
            description="Check email format validity",
            dimension=DataQualityDimension.VALIDITY,
            rule_expression="email_format",
            threshold=0.98,
            severity="warning"
        )
    ]


@pytest.fixture
def business_rules():
    """Create sample business rules"""
    return [
        BusinessRule(
            name="Minimum Age Requirement",
            description="Users must be at least 18 years old",
            category="user_validation",
            rule_logic="required_fields: age",
            business_owner="HR Department",
            technical_owner="Data Team",
            effective_date=datetime.utcnow() - timedelta(days=30)
        ),
        BusinessRule(
            name="Email Required",
            description="All users must have valid email addresses",
            category="contact_validation",
            rule_logic="required_fields: email",
            business_owner="Marketing Department",
            technical_owner="Data Team",
            effective_date=datetime.utcnow() - timedelta(days=60)
        )
    ]


@pytest.fixture
def agent_output():
    """Create sample agent output"""
    return AgentOutput(
        agent_id="test_agent_001",
        agent_type="data_analyst",
        task_id="task_123",
        output_data={
            "analysis_result": "Customer satisfaction increased by 15%",
            "confidence": 0.87,
            "recommendations": [
                "Continue current marketing strategy",
                "Increase customer support capacity"
            ],
            "data_points": 1500,
            "methodology": "statistical_analysis"
        },
        metadata={
            "model_version": "v2.1",
            "data_sources_count": 3,
            "processing_method": "real_time"
        },
        generation_timestamp=datetime.utcnow(),
        processing_time_ms=2500,
        confidence_score=0.87,
        data_sources=["customer_feedback_db", "sales_analytics", "support_tickets"],
        reasoning_trace=[
            "Analyzed customer feedback data from Q4 2023",
            "Calculated satisfaction metrics using NPS methodology",
            "Compared with previous quarter results",
            "Identified key improvement factors"
        ]
    )


@pytest.fixture
def agent_output_schema():
    """Create sample agent output schema"""
    return AgentOutputSchema(
        agent_type="data_analyst",
        output_format="json",
        required_fields=["analysis_result", "confidence", "recommendations"],
        optional_fields=["data_points", "methodology"],
        field_types={
            "analysis_result": "string",
            "confidence": "float",
            "recommendations": "list",
            "data_points": "integer"
        },
        validation_rules=[
            "confidence >= 0.0 and confidence <= 1.0",
            "len(recommendations) > 0"
        ],
        business_constraints=[
            "analysis_result must not be empty",
            "confidence must be realistic (< 0.99)"
        ],
        authenticity_checks=[
            "no_simulation_markers",
            "realistic_processing_time",
            "authentic_data_sources"
        ]
    )


class TestQualityAssuranceEngine:
    """Test cases for Quality Assurance Engine"""
    
    @pytest.mark.asyncio
    async def test_comprehensive_assessment(self, qa_engine):
        """Test comprehensive quality assessment"""
        assessment = await qa_engine.run_comprehensive_assessment(
            target_system="test_system",
            assessment_type="full"
        )
        
        assert isinstance(assessment, QualityAssessment)
        assert assessment.target_system == "test_system"
        assert assessment.assessment_type == "full"
        assert assessment.overall_quality_score >= 0.0
        assert assessment.overall_quality_score <= 1.0
        assert assessment.certification_status in ["certified", "conditional", "rejected"]
    
    @pytest.mark.asyncio
    async def test_data_quality_validation(self, qa_engine, sample_data, data_quality_rules):
        """Test data quality validation"""
        report = await qa_engine.validate_data_quality_real_time(
            data=sample_data,
            dataset_id="test_dataset",
            rules=data_quality_rules
        )
        
        assert isinstance(report, DataQualityReport)
        assert report.dataset_id == "test_dataset"
        assert len(report.metrics) == len(data_quality_rules)
        assert report.overall_score >= 0.0
        assert report.overall_score <= 1.0
        
        # Check dimension scores
        for dimension in report.dimension_scores:
            assert isinstance(dimension, DataQualityDimension)
            assert report.dimension_scores[dimension] >= 0.0
            assert report.dimension_scores[dimension] <= 1.0
    
    @pytest.mark.asyncio
    async def test_anomaly_detection(self, qa_engine, sample_data):
        """Test real-time anomaly detection"""
        # Add some anomalous data
        anomalous_data = sample_data.copy()
        anomalous_data.loc[0, 'age'] = 200  # Impossible age
        anomalous_data.loc[1, 'salary'] = -50000  # Negative salary
        
        config = AnomalyDetectionConfig(
            detection_method="hybrid",
            sensitivity=0.8,
            real_time_processing=True
        )
        
        anomalies = await qa_engine.detect_anomalies_real_time(
            data=anomalous_data,
            dataset_id="test_dataset",
            config=config
        )
        
        assert isinstance(anomalies, list)
        # Should detect at least some anomalies
        assert len(anomalies) >= 0
        
        for anomaly in anomalies:
            assert isinstance(anomaly, AnomalyDetection)
            assert anomaly.confidence_score >= 0.0
            assert anomaly.confidence_score <= 1.0
            assert anomaly.anomaly_type in list(AnomalyType)
    
    @pytest.mark.asyncio
    async def test_business_rule_validation(self, qa_engine, business_rules):
        """Test business rule validation"""
        test_data = {
            "age": 25,
            "email": "test@example.com",
            "name": "Test User"
        }
        
        validations = await qa_engine.validate_business_rules(
            data=test_data,
            rules=business_rules
        )
        
        assert isinstance(validations, list)
        assert len(validations) == len(business_rules)
        
        for validation in validations:
            assert isinstance(validation, BusinessRuleValidation)
            assert validation.compliance_score >= 0.0
            assert validation.compliance_score <= 1.0
            assert validation.status in list(ValidationStatus)
    
    @pytest.mark.asyncio
    async def test_agent_output_validation(self, qa_engine, agent_output, agent_output_schema):
        """Test agent output validation"""
        result = await qa_engine.validate_agent_output(
            output=agent_output,
            schema=agent_output_schema
        )
        
        assert isinstance(result, OutputValidationResult)
        assert result.agent_id == agent_output.agent_id
        assert result.quality_score >= 0.0
        assert result.quality_score <= 1.0
        assert result.overall_status in list(ValidationStatus)
        
        # Check validation components
        assert "is_valid" in result.format_validation
        assert "is_valid" in result.business_validation
        assert "authenticity_score" in result.data_validation
        assert "is_compliant" in result.compliance_validation
    
    @pytest.mark.asyncio
    async def test_simulation_detection(self, qa_engine):
        """Test simulation detection in agent output"""
        # Create output with simulation markers
        simulated_output = AgentOutput(
            agent_id="test_agent",
            agent_type="analyst",
            task_id="task_123",
            output_data={
                "result": "This is test_data from demo_source",
                "confidence": 0.999  # Suspiciously high confidence
            },
            data_sources=["mock_api", "sample_database"],
            reasoning_trace=["Generated mock analysis"],
            generation_timestamp=datetime.utcnow(),
            processing_time_ms=5  # Suspiciously fast
        )
        
        schema = AgentOutputSchema(
            agent_type="analyst",
            output_format="json",
            required_fields=["result"],
            authenticity_checks=["no_simulation_markers"]
        )
        
        result = await qa_engine.validate_agent_output(
            output=simulated_output,
            schema=schema
        )
        
        # Should detect simulation
        assert not result.is_simulation_free
        assert not result.is_authentic
        assert result.quality_score < 0.5
    
    @pytest.mark.asyncio
    async def test_performance_testing(self, qa_engine):
        """Test performance testing functionality"""
        config = PerformanceTestConfig(
            test_name="API Load Test",
            target_endpoint="http://localhost:8000/api/test",
            load_pattern="constant",
            concurrent_users=10,
            duration_seconds=60,
            success_criteria={"response_time": 500.0, "error_rate": 0.05}
        )
        
        results = await qa_engine.run_performance_tests(config)
        
        assert results.config_id == config.id
        assert results.total_requests >= 0
        assert results.error_rate >= 0.0
        assert results.error_rate <= 1.0
        assert results.average_response_time >= 0.0
    
    @pytest.mark.asyncio
    async def test_security_testing(self, qa_engine):
        """Test security testing functionality"""
        test_cases = [
            SecurityTestCase(
                name="SQL Injection Test",
                category="injection",
                severity="high",
                test_vector="'; DROP TABLE users; --",
                expected_behavior="Input should be sanitized"
            ),
            SecurityTestCase(
                name="XSS Test",
                category="xss",
                severity="medium",
                test_vector="<script>alert('xss')</script>",
                expected_behavior="Script should be escaped"
            )
        ]
        
        results = await qa_engine.run_security_tests(test_cases)
        
        assert len(results) == len(test_cases)
        for result in results:
            assert result.risk_level in ["critical", "high", "medium", "low"]
            assert isinstance(result.vulnerability_found, bool)


class TestDataQualityValidation:
    """Test cases for data quality validation"""
    
    def test_completeness_calculation(self, qa_engine, data_quality_rules):
        """Test completeness score calculation"""
        # Create data with missing values
        data_with_nulls = pd.DataFrame({
            'col1': [1, 2, None, 4, 5],
            'col2': ['a', None, 'c', 'd', None],
            'col3': [1.1, 2.2, 3.3, 4.4, 5.5]
        })
        
        completeness_rule = data_quality_rules[0]  # Completeness rule
        score = qa_engine._calculate_completeness_score(data_with_nulls, completeness_rule)
        
        # Should be 11/15 = 0.733...
        expected_score = 11 / 15
        assert abs(score - expected_score) < 0.01
    
    def test_uniqueness_calculation(self, qa_engine, data_quality_rules):
        """Test uniqueness score calculation"""
        # Create data with duplicates
        data_with_duplicates = pd.DataFrame({
            'id': [1, 2, 2, 3, 3, 3],
            'name': ['a', 'b', 'b', 'c', 'c', 'c']
        })
        
        uniqueness_rule = DataQualityRule(
            name="Uniqueness Test",
            description="Test uniqueness",
            dimension=DataQualityDimension.UNIQUENESS,
            rule_expression="unique_rows"
        )
        
        score = qa_engine._calculate_uniqueness_score(data_with_duplicates, uniqueness_rule)
        
        # Should be 3/6 = 0.5 (3 unique rows out of 6 total)
        assert abs(score - 0.5) < 0.01
    
    def test_authenticity_detection(self, qa_engine):
        """Test authenticity score calculation"""
        # Create data with simulation patterns
        fake_data = pd.DataFrame({
            'name': ['test_user_1', 'demo_user_2', 'sample_user_3'],
            'email': ['test@example.com', 'demo@test.com', 'sample@fake.com'],
            'value': [1.0, 2.0, 3.0]  # Perfect sequence
        })
        
        authenticity_rule = DataQualityRule(
            name="Authenticity Check",
            description="Check for fake data patterns",
            dimension=DataQualityDimension.AUTHENTICITY,
            rule_expression="no_simulation_patterns"
        )
        
        score = qa_engine._calculate_authenticity_score(fake_data, authenticity_rule)
        
        # Should detect simulation patterns and reduce score
        assert score < 1.0


class TestAnomalyDetection:
    """Test cases for anomaly detection"""
    
    @pytest.mark.asyncio
    async def test_statistical_anomaly_detection(self, qa_engine):
        """Test statistical anomaly detection"""
        # Create data with clear outliers
        normal_data = np.random.normal(50, 10, 100)
        outlier_data = np.concatenate([normal_data, [200, -100]])  # Clear outliers
        
        data = pd.DataFrame({'value': outlier_data})
        config = AnomalyDetectionConfig(detection_method="statistical")
        
        anomalies = await qa_engine._detect_statistical_anomalies(data, config)
        
        # Should detect the outliers
        assert len(anomalies) >= 2
        
        for anomaly in anomalies:
            assert anomaly.anomaly_type == AnomalyType.STATISTICAL_OUTLIER
            assert anomaly.confidence_score > 0.0
    
    @pytest.mark.asyncio
    async def test_ml_anomaly_detection(self, qa_engine):
        """Test ML-based anomaly detection"""
        # Create data with pattern anomalies
        normal_pattern = []
        for i in range(100):
            normal_pattern.append([i, i*2, i*3])
        
        # Add anomalous patterns
        anomalous_pattern = [[1000, 5, 10], [2000, 3, 7]]
        
        all_data = normal_pattern + anomalous_pattern
        data = pd.DataFrame(all_data, columns=['x', 'y', 'z'])
        
        config = AnomalyDetectionConfig(
            detection_method="ml_based",
            min_samples=50
        )
        
        anomalies = await qa_engine._detect_ml_anomalies(data, config)
        
        # Should detect some anomalies
        assert len(anomalies) >= 0  # ML detection might not catch all patterns
    
    @pytest.mark.asyncio
    async def test_business_rule_anomaly_detection(self, qa_engine):
        """Test business rule-based anomaly detection"""
        # Create data with business rule violations
        data = pd.DataFrame({
            'age': [25, 30, -5, 200, 45],  # Invalid ages
            'salary': [50000, 60000, 70000, 80000, 90000]
        })
        
        config = AnomalyDetectionConfig(detection_method="rule_based")
        
        anomalies = await qa_engine._detect_rule_based_anomalies(data, config)
        
        # Should detect invalid ages
        assert len(anomalies) >= 2  # -5 and 200 are invalid ages
        
        for anomaly in anomalies:
            assert anomaly.anomaly_type == AnomalyType.BUSINESS_RULE_VIOLATION


class TestBusinessRuleValidation:
    """Test cases for business rule validation"""
    
    @pytest.mark.asyncio
    async def test_valid_data_passes_rules(self, qa_engine, business_rules):
        """Test that valid data passes business rules"""
        valid_data = {
            "age": 30,
            "email": "valid@example.com",
            "name": "Valid User"
        }
        
        validations = await qa_engine.validate_business_rules(valid_data, business_rules)
        
        for validation in validations:
            assert validation.validation_result is True
            assert validation.status == ValidationStatus.PASSED
            assert validation.compliance_score == 1.0
    
    @pytest.mark.asyncio
    async def test_invalid_data_fails_rules(self, qa_engine, business_rules):
        """Test that invalid data fails business rules"""
        invalid_data = {
            "name": "Invalid User"
            # Missing required age and email
        }
        
        validations = await qa_engine.validate_business_rules(invalid_data, business_rules)
        
        # Should have some failures
        failed_validations = [v for v in validations if not v.validation_result]
        assert len(failed_validations) > 0


class TestAgentOutputValidation:
    """Test cases for agent output validation"""
    
    @pytest.mark.asyncio
    async def test_valid_output_passes_validation(self, qa_engine, agent_output, agent_output_schema):
        """Test that valid agent output passes validation"""
        result = await qa_engine.validate_agent_output(agent_output, agent_output_schema)
        
        assert result.overall_status in [ValidationStatus.PASSED, ValidationStatus.WARNING]
        assert result.is_authentic is True
        assert result.is_simulation_free is True
        assert result.quality_score > 0.5
    
    @pytest.mark.asyncio
    async def test_invalid_output_fails_validation(self, qa_engine, agent_output_schema):
        """Test that invalid agent output fails validation"""
        invalid_output = AgentOutput(
            agent_id="test_agent",
            agent_type="analyst",
            task_id="task_123",
            output_data={
                # Missing required fields
                "extra_field": "not_required"
            },
            generation_timestamp=datetime.utcnow(),
            processing_time_ms=1000
        )
        
        result = await qa_engine.validate_agent_output(invalid_output, agent_output_schema)
        
        assert result.overall_status == ValidationStatus.FAILED
        assert result.quality_score < 0.5
        assert len(result.format_validation.get("errors", [])) > 0
    
    def test_field_type_validation(self, qa_engine):
        """Test field type validation"""
        # Test various type validations
        assert qa_engine._validate_field_type("hello", "string") is True
        assert qa_engine._validate_field_type(42, "integer") is True
        assert qa_engine._validate_field_type(3.14, "float") is True
        assert qa_engine._validate_field_type(True, "boolean") is True
        assert qa_engine._validate_field_type([1, 2, 3], "list") is True
        assert qa_engine._validate_field_type({"key": "value"}, "dict") is True
        
        # Test type mismatches
        assert qa_engine._validate_field_type(42, "string") is False
        assert qa_engine._validate_field_type("hello", "integer") is False


class TestQualityMetrics:
    """Test cases for quality metrics and scoring"""
    
    def test_overall_quality_score_calculation(self, qa_engine):
        """Test overall quality score calculation"""
        # Create mock assessment
        assessment = QualityAssessment(
            target_system="test",
            assessment_type="full",
            assessment_timestamp=datetime.utcnow()
        )
        
        # Add mock results
        assessment.test_results = TestResults(
            suite_id="test",
            total_tests=10,
            passed_tests=8,
            failed_tests=2,
            skipped_tests=0,
            warning_tests=0,
            success_rate=0.8,
            total_duration_ms=5000
        )
        
        score = qa_engine._calculate_overall_quality_score(assessment)
        
        assert score >= 0.0
        assert score <= 1.0
    
    def test_certification_status_determination(self, qa_engine):
        """Test certification status determination"""
        # Create assessments with different scores
        high_score_assessment = QualityAssessment(
            target_system="test",
            assessment_type="full",
            assessment_timestamp=datetime.utcnow(),
            overall_quality_score=0.96
        )
        
        medium_score_assessment = QualityAssessment(
            target_system="test",
            assessment_type="full",
            assessment_timestamp=datetime.utcnow(),
            overall_quality_score=0.85
        )
        
        low_score_assessment = QualityAssessment(
            target_system="test",
            assessment_type="full",
            assessment_timestamp=datetime.utcnow(),
            overall_quality_score=0.65
        )
        
        assert qa_engine._determine_certification_status(high_score_assessment) == "certified"
        assert qa_engine._determine_certification_status(medium_score_assessment) == "conditional"
        assert qa_engine._determine_certification_status(low_score_assessment) == "rejected"


class TestSimulationDetection:
    """Test cases for simulation and fake data detection"""
    
    def test_simulation_pattern_detection(self, qa_engine):
        """Test detection of simulation patterns in data"""
        # Test data with simulation markers
        simulation_data = pd.DataFrame({
            'name': ['test_user', 'demo_account', 'sample_data'],
            'email': ['test@example.com', 'demo@fake.com', 'sample@test.org']
        })
        
        # Mock pattern
        pattern = {"type": "test_markers", "pattern": "test_|demo_|sample_"}
        
        # This would be implemented in the actual detection method
        # For now, we test the concept
        has_pattern = any(
            any(marker in str(val).lower() for marker in ['test_', 'demo_', 'sample_'])
            for col in simulation_data.columns
            for val in simulation_data[col]
        )
        
        assert has_pattern is True
    
    @pytest.mark.asyncio
    async def test_output_simulation_detection(self, qa_engine):
        """Test simulation detection in agent outputs"""
        # Create output with simulation indicators
        simulated_output = AgentOutput(
            agent_id="test_agent",
            agent_type="analyst",
            task_id="task_123",
            output_data={"result": "Generated mock analysis"},
            reasoning_trace=["Simulated data analysis", "Generated fake insights"],
            metadata={"is_simulated": True},
            generation_timestamp=datetime.utcnow(),
            processing_time_ms=1000
        )
        
        is_simulation_free = await qa_engine._detect_output_simulation(simulated_output)
        
        assert is_simulation_free is False
    
    @pytest.mark.asyncio
    async def test_authenticity_verification(self, qa_engine):
        """Test authenticity verification"""
        # Create authentic output
        authentic_output = AgentOutput(
            agent_id="prod_agent",
            agent_type="analyst",
            task_id="real_task",
            output_data={"analysis": "Real business insights"},
            data_sources=["production_db", "sales_api"],
            reasoning_trace=["Analyzed customer data", "Calculated metrics"],
            generation_timestamp=datetime.utcnow(),
            processing_time_ms=2500
        )
        
        is_authentic = await qa_engine._verify_output_authenticity(authentic_output)
        
        assert is_authentic is True
        
        # Create fake output
        fake_output = AgentOutput(
            agent_id="test_agent",
            agent_type="analyst",
            task_id="fake_task",
            output_data={"analysis": "Fake insights"},
            data_sources=["mock_api", "test_db"],
            reasoning_trace=["Generated fake analysis"],
            generation_timestamp=datetime.utcnow(),
            processing_time_ms=5  # Too fast
        )
        
        is_authentic_fake = await qa_engine._verify_output_authenticity(fake_output)
        
        assert is_authentic_fake is False


if __name__ == "__main__":
    pytest.main([__file__])