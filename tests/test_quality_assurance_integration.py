"""
Integration Tests for Quality Assurance API Routes

This module contains integration tests for the quality assurance API endpoints,
testing the complete flow from API requests to engine responses.
"""

import pytest
import asyncio
from fastapi.testclient import TestClient
from fastapi import FastAPI
import pandas as pd
import json
import io
from datetime import datetime
from unittest.mock import Mock, patch

from scrollintel.api.routes.quality_assurance_routes import router
from scrollintel.models.quality_assurance_models import (
    QualityAssuranceConfig, DataQualityRule, DataQualityDimension,
    AnomalyDetectionConfig, BusinessRule, AgentOutput, AgentOutputSchema,
    PerformanceTestConfig, SecurityTestCase, ValidationStatus
)


# Create test app
app = FastAPI()
app.include_router(router)
client = TestClient(app)


@pytest.fixture
def mock_qa_engine():
    """Create mock quality assurance engine"""
    with patch('scrollintel.api.routes.quality_assurance_routes.get_qa_engine') as mock:
        engine = Mock()
        mock.return_value = engine
        yield engine


@pytest.fixture
def mock_auth():
    """Mock authentication"""
    with patch('scrollintel.api.routes.quality_assurance_routes.get_current_user') as mock:
        mock.return_value = {"user_id": "test_user", "role": "admin"}
        yield mock


@pytest.fixture
def sample_csv_data():
    """Create sample CSV data for testing"""
    data = {
        'id': [1, 2, 3, 4, 5],
        'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
        'age': [25, 30, 35, 28, 32],
        'email': ['alice@test.com', 'bob@test.com', 'charlie@test.com', 'diana@test.com', 'eve@test.com']
    }
    df = pd.DataFrame(data)
    return df.to_csv(index=False)


class TestQualityAssuranceAPI:
    """Test cases for Quality Assurance API endpoints"""
    
    def test_health_check(self):
        """Test health check endpoint"""
        response = client.get("/api/v1/quality-assurance/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "components" in data
        
        # Check all components are operational
        components = data["components"]
        expected_components = [
            "automated_testing", "data_quality_validation", "anomaly_detection",
            "business_rule_validation", "agent_output_validation",
            "performance_testing", "security_testing"
        ]
        
        for component in expected_components:
            assert component in components
            assert components[component] == "operational"
    
    def test_get_statistics(self, mock_auth):
        """Test statistics endpoint"""
        response = client.get("/api/v1/quality-assurance/statistics")
        
        assert response.status_code == 200
        data = response.json()
        
        expected_fields = [
            "assessments_completed", "data_quality_checks", "anomalies_detected",
            "business_rules_validated", "agent_outputs_validated",
            "performance_tests_run", "security_tests_run",
            "average_quality_score", "time_range"
        ]
        
        for field in expected_fields:
            assert field in data
    
    def test_get_qa_config(self, mock_auth):
        """Test get QA configuration endpoint"""
        response = client.get("/api/v1/quality-assurance/config")
        
        assert response.status_code == 200
        data = response.json()
        
        # Check required configuration fields
        assert "organization_id" in data
        assert "automated_testing_enabled" in data
        assert "data_quality_monitoring" in data
        assert "anomaly_detection_enabled" in data
        assert "business_rule_enforcement" in data
        assert "agent_output_validation" in data
        assert "simulation_detection_enabled" in data
        assert "authenticity_verification" in data
    
    def test_update_qa_config(self, mock_auth):
        """Test update QA configuration endpoint"""
        config_data = {
            "organization_id": "test_org",
            "automated_testing_enabled": True,
            "data_quality_monitoring": True,
            "anomaly_detection_enabled": True,
            "business_rule_enforcement": True,
            "agent_output_validation": True,
            "simulation_detection_enabled": True,
            "authenticity_verification": True,
            "quality_score_threshold": 0.85
        }
        
        response = client.put(
            "/api/v1/quality-assurance/config",
            json=config_data
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["organization_id"] == "test_org"
        assert data["quality_score_threshold"] == 0.85


class TestComprehensiveAssessment:
    """Test cases for comprehensive assessment endpoints"""
    
    def test_run_comprehensive_assessment(self, mock_qa_engine, mock_auth):
        """Test running comprehensive assessment"""
        # Mock assessment result
        mock_assessment = {
            "assessment_id": "test_assessment_123",
            "target_system": "test_system",
            "assessment_type": "full",
            "assessment_timestamp": datetime.utcnow().isoformat(),
            "overall_quality_score": 0.85,
            "certification_status": "conditional",
            "production_readiness": True,
            "critical_issues": [],
            "recommendations": ["Improve data quality monitoring"]
        }
        
        mock_qa_engine.run_comprehensive_assessment.return_value = mock_assessment
        
        response = client.post(
            "/api/v1/quality-assurance/assessments",
            params={"target_system": "test_system", "assessment_type": "full"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["target_system"] == "test_system"
        assert data["overall_quality_score"] == 0.85
        assert data["certification_status"] == "conditional"
    
    def test_list_assessments(self, mock_auth):
        """Test listing assessments"""
        response = client.get("/api/v1/quality-assurance/assessments")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
    
    def test_get_assessment_not_found(self, mock_auth):
        """Test getting non-existent assessment"""
        response = client.get("/api/v1/quality-assurance/assessments/nonexistent")
        
        assert response.status_code == 404


class TestDataQualityValidation:
    """Test cases for data quality validation endpoints"""
    
    def test_validate_data_quality_csv(self, mock_qa_engine, mock_auth, sample_csv_data):
        """Test data quality validation with CSV file"""
        # Mock validation result
        mock_report = {
            "dataset_id": "test_dataset",
            "dataset_name": "Dataset_test_dataset",
            "assessment_timestamp": datetime.utcnow().isoformat(),
            "overall_score": 0.92,
            "dimension_scores": {
                "completeness": 0.95,
                "accuracy": 0.90,
                "validity": 0.91
            },
            "metrics": [],
            "critical_issues": [],
            "recommendations": [],
            "is_production_ready": True,
            "certification_status": "certified"
        }
        
        mock_qa_engine.validate_data_quality_real_time.return_value = mock_report
        
        # Create rules
        rules_data = [
            {
                "name": "Completeness Check",
                "description": "Check for missing values",
                "dimension": "completeness",
                "rule_expression": "no_nulls",
                "threshold": 0.95,
                "severity": "error"
            }
        ]
        
        # Create file upload
        files = {"file": ("test.csv", io.StringIO(sample_csv_data), "text/csv")}
        data = {
            "dataset_id": "test_dataset",
            "rules": json.dumps(rules_data)
        }
        
        response = client.post(
            "/api/v1/quality-assurance/data-quality/validate",
            files=files,
            data=data
        )
        
        assert response.status_code == 200
        result = response.json()
        assert result["dataset_id"] == "test_dataset"
        assert result["overall_score"] == 0.92
        assert result["is_production_ready"] is True
    
    def test_validate_data_quality_json(self, mock_qa_engine, mock_auth):
        """Test data quality validation with JSON file"""
        # Mock validation result
        mock_report = {
            "dataset_id": "json_dataset",
            "overall_score": 0.88,
            "is_production_ready": True
        }
        
        mock_qa_engine.validate_data_quality_real_time.return_value = mock_report
        
        # Create JSON data
        json_data = [
            {"id": 1, "name": "Alice", "age": 25},
            {"id": 2, "name": "Bob", "age": 30}
        ]
        
        files = {"file": ("test.json", io.StringIO(json.dumps(json_data)), "application/json")}
        data = {
            "dataset_id": "json_dataset",
            "rules": json.dumps([])
        }
        
        response = client.post(
            "/api/v1/quality-assurance/data-quality/validate",
            files=files,
            data=data
        )
        
        assert response.status_code == 200
        result = response.json()
        assert result["dataset_id"] == "json_dataset"
    
    def test_validate_unsupported_file_format(self, mock_auth):
        """Test validation with unsupported file format"""
        files = {"file": ("test.txt", io.StringIO("some text"), "text/plain")}
        data = {
            "dataset_id": "test_dataset",
            "rules": json.dumps([])
        }
        
        response = client.post(
            "/api/v1/quality-assurance/data-quality/validate",
            files=files,
            data=data
        )
        
        assert response.status_code == 400
        assert "Unsupported file format" in response.json()["detail"]
    
    def test_create_data_quality_rule(self, mock_auth):
        """Test creating data quality rule"""
        rule_data = {
            "name": "Age Validity",
            "description": "Check age is within valid range",
            "dimension": "validity",
            "rule_expression": "age >= 0 AND age <= 120",
            "threshold": 0.99,
            "severity": "error",
            "business_impact": "high"
        }
        
        response = client.post(
            "/api/v1/quality-assurance/data-quality/rules",
            json=rule_data
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Age Validity"
        assert data["dimension"] == "validity"
    
    def test_list_data_quality_rules(self, mock_auth):
        """Test listing data quality rules"""
        response = client.get("/api/v1/quality-assurance/data-quality/rules")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)


class TestAnomalyDetection:
    """Test cases for anomaly detection endpoints"""
    
    def test_detect_anomalies(self, mock_qa_engine, mock_auth, sample_csv_data):
        """Test anomaly detection"""
        # Mock anomaly detection result
        mock_anomalies = [
            {
                "detection_id": "anomaly_123",
                "anomaly_type": "statistical_outlier",
                "severity": "medium",
                "confidence_score": 0.85,
                "affected_records": 1,
                "detection_timestamp": datetime.utcnow().isoformat(),
                "data_source": "test_dataset",
                "recommended_actions": ["Review data point"]
            }
        ]
        
        mock_qa_engine.detect_anomalies_real_time.return_value = mock_anomalies
        
        config_data = {
            "detection_method": "hybrid",
            "sensitivity": 0.8,
            "real_time_processing": True
        }
        
        files = {"file": ("test.csv", io.StringIO(sample_csv_data), "text/csv")}
        data = {
            "dataset_id": "test_dataset",
            "config": json.dumps(config_data)
        }
        
        response = client.post(
            "/api/v1/quality-assurance/anomaly-detection/detect",
            files=files,
            data=data
        )
        
        assert response.status_code == 200
        result = response.json()
        assert isinstance(result, list)
        if result:  # If anomalies detected
            assert result[0]["anomaly_type"] == "statistical_outlier"
            assert result[0]["confidence_score"] == 0.85
    
    def test_list_anomalies(self, mock_auth):
        """Test listing anomalies"""
        response = client.get("/api/v1/quality-assurance/anomaly-detection/anomalies")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)


class TestBusinessRuleValidation:
    """Test cases for business rule validation endpoints"""
    
    def test_validate_business_rules(self, mock_qa_engine, mock_auth):
        """Test business rule validation"""
        # Mock validation result
        mock_validations = [
            {
                "rule_id": "rule_123",
                "validation_result": True,
                "status": "passed",
                "compliance_score": 1.0
            }
        ]
        
        mock_qa_engine.validate_business_rules.return_value = mock_validations
        
        request_data = {
            "data": {
                "age": 25,
                "email": "test@example.com"
            },
            "rules": [
                {
                    "name": "Age Requirement",
                    "description": "Must be 18 or older",
                    "category": "validation",
                    "rule_logic": "age >= 18",
                    "business_owner": "HR",
                    "technical_owner": "Data Team",
                    "effective_date": datetime.utcnow().isoformat()
                }
            ]
        }
        
        response = client.post(
            "/api/v1/quality-assurance/business-rules/validate",
            json=request_data
        )
        
        assert response.status_code == 200
        result = response.json()
        assert isinstance(result, list)
        if result:
            assert result[0]["validation_result"] is True
            assert result[0]["status"] == "passed"
    
    def test_create_business_rule(self, mock_auth):
        """Test creating business rule"""
        rule_data = {
            "name": "Email Required",
            "description": "All users must have email",
            "category": "validation",
            "rule_logic": "email IS NOT NULL",
            "business_owner": "Marketing",
            "technical_owner": "Data Team",
            "priority": 3,
            "is_mandatory": True,
            "effective_date": datetime.utcnow().isoformat()
        }
        
        response = client.post(
            "/api/v1/quality-assurance/business-rules",
            json=rule_data
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Email Required"
        assert data["is_mandatory"] is True
    
    def test_list_business_rules(self, mock_auth):
        """Test listing business rules"""
        response = client.get("/api/v1/quality-assurance/business-rules")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)


class TestAgentOutputValidation:
    """Test cases for agent output validation endpoints"""
    
    def test_validate_agent_output(self, mock_qa_engine, mock_auth):
        """Test agent output validation"""
        # Mock validation result
        mock_result = {
            "output_id": "output_123",
            "agent_id": "agent_456",
            "validation_timestamp": datetime.utcnow().isoformat(),
            "overall_status": "passed",
            "is_authentic": True,
            "is_simulation_free": True,
            "quality_score": 0.92,
            "format_validation": {"is_valid": True},
            "business_validation": {"is_valid": True},
            "data_validation": {"authenticity_score": 0.95},
            "compliance_validation": {"is_compliant": True}
        }
        
        mock_qa_engine.validate_agent_output.return_value = mock_result
        
        request_data = {
            "output": {
                "agent_id": "agent_456",
                "agent_type": "analyst",
                "task_id": "task_789",
                "output_data": {
                    "analysis": "Customer satisfaction improved",
                    "confidence": 0.87
                },
                "generation_timestamp": datetime.utcnow().isoformat(),
                "processing_time_ms": 2500,
                "data_sources": ["customer_db", "feedback_api"]
            },
            "schema": {
                "agent_type": "analyst",
                "output_format": "json",
                "required_fields": ["analysis", "confidence"],
                "field_types": {
                    "analysis": "string",
                    "confidence": "float"
                }
            }
        }
        
        response = client.post(
            "/api/v1/quality-assurance/agent-output/validate",
            json=request_data
        )
        
        assert response.status_code == 200
        result = response.json()
        assert result["agent_id"] == "agent_456"
        assert result["is_authentic"] is True
        assert result["is_simulation_free"] is True
        assert result["quality_score"] == 0.92
    
    def test_create_output_schema(self, mock_auth):
        """Test creating agent output schema"""
        schema_data = {
            "agent_type": "data_scientist",
            "output_format": "json",
            "required_fields": ["model_accuracy", "feature_importance"],
            "optional_fields": ["model_parameters"],
            "field_types": {
                "model_accuracy": "float",
                "feature_importance": "list"
            },
            "validation_rules": [
                "model_accuracy >= 0.0 and model_accuracy <= 1.0"
            ]
        }
        
        response = client.post(
            "/api/v1/quality-assurance/agent-output/schemas",
            json=schema_data
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["agent_type"] == "data_scientist"
        assert "model_accuracy" in data["required_fields"]
    
    def test_list_output_validations(self, mock_auth):
        """Test listing output validations"""
        response = client.get("/api/v1/quality-assurance/agent-output/validations")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)


class TestPerformanceTesting:
    """Test cases for performance testing endpoints"""
    
    def test_run_performance_test(self, mock_qa_engine, mock_auth):
        """Test running performance test"""
        # Mock performance test result
        mock_result = {
            "config_id": "config_123",
            "execution_timestamp": datetime.utcnow().isoformat(),
            "total_requests": 1000,
            "successful_requests": 950,
            "failed_requests": 50,
            "average_response_time": 150.0,
            "p95_response_time": 300.0,
            "p99_response_time": 500.0,
            "throughput_rps": 100.0,
            "error_rate": 0.05,
            "performance_grade": "good"
        }
        
        mock_qa_engine.run_performance_tests.return_value = mock_result
        
        config_data = {
            "test_name": "API Load Test",
            "target_endpoint": "http://localhost:8000/api/test",
            "load_pattern": "constant",
            "concurrent_users": 10,
            "duration_seconds": 60,
            "success_criteria": {
                "response_time": 500.0,
                "error_rate": 0.05
            }
        }
        
        response = client.post(
            "/api/v1/quality-assurance/performance-tests/run",
            json=config_data
        )
        
        assert response.status_code == 200
        result = response.json()
        assert result["total_requests"] == 1000
        assert result["error_rate"] == 0.05
        assert result["performance_grade"] == "good"
    
    def test_list_performance_test_results(self, mock_auth):
        """Test listing performance test results"""
        response = client.get("/api/v1/quality-assurance/performance-tests/results")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)


class TestSecurityTesting:
    """Test cases for security testing endpoints"""
    
    def test_run_security_tests(self, mock_qa_engine, mock_auth):
        """Test running security tests"""
        # Mock security test results
        mock_results = [
            {
                "test_case_id": "test_123",
                "execution_timestamp": datetime.utcnow().isoformat(),
                "vulnerability_found": False,
                "risk_level": "low"
            }
        ]
        
        mock_qa_engine.run_security_tests.return_value = mock_results
        
        test_cases_data = [
            {
                "name": "SQL Injection Test",
                "category": "injection",
                "severity": "high",
                "test_vector": "'; DROP TABLE users; --",
                "expected_behavior": "Input should be sanitized",
                "automated": True
            }
        ]
        
        response = client.post(
            "/api/v1/quality-assurance/security-tests/run",
            json=test_cases_data
        )
        
        assert response.status_code == 200
        result = response.json()
        assert isinstance(result, list)
        if result:
            assert result[0]["vulnerability_found"] is False
            assert result[0]["risk_level"] == "low"
    
    def test_list_security_test_results(self, mock_auth):
        """Test listing security test results"""
        response = client.get("/api/v1/quality-assurance/security-tests/results")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)


class TestMonitoringEndpoints:
    """Test cases for monitoring endpoints"""
    
    def test_get_quality_metrics(self, mock_auth):
        """Test getting quality metrics"""
        response = client.get("/api/v1/quality-assurance/metrics")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
    
    def test_get_quality_alerts(self, mock_auth):
        """Test getting quality alerts"""
        response = client.get("/api/v1/quality-assurance/alerts")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
    
    def test_acknowledge_alert(self, mock_auth):
        """Test acknowledging quality alert"""
        response = client.post("/api/v1/quality-assurance/alerts/alert_123/acknowledge")
        
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Alert acknowledged"
        assert data["alert_id"] == "alert_123"


class TestErrorHandling:
    """Test cases for error handling"""
    
    def test_assessment_failure(self, mock_qa_engine, mock_auth):
        """Test handling of assessment failures"""
        mock_qa_engine.run_comprehensive_assessment.side_effect = Exception("Assessment failed")
        
        response = client.post(
            "/api/v1/quality-assurance/assessments",
            params={"target_system": "failing_system"}
        )
        
        assert response.status_code == 500
        assert "Assessment failed" in response.json()["detail"]
    
    def test_unauthorized_access(self):
        """Test unauthorized access to protected endpoints"""
        with patch('scrollintel.api.routes.quality_assurance_routes.get_current_user') as mock_auth:
            mock_auth.side_effect = Exception("Unauthorized")
            
            response = client.get("/api/v1/quality-assurance/config")
            
            # The actual behavior depends on how authentication is implemented
            # This test ensures the endpoint is protected
            assert response.status_code in [401, 403, 500]


if __name__ == "__main__":
    pytest.main([__file__])