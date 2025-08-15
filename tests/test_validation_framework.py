"""
Tests for Innovation Validation Framework.
"""

import pytest
import pytest_asyncio
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from scrollintel.engines.validation_framework import ValidationFramework
from scrollintel.models.validation_models import (
    Innovation, ValidationRequest, ValidationReport, ValidationScore,
    ValidationType, ValidationStatus, ValidationResult, ImpactLevel
)


@pytest_asyncio.fixture
async def validation_framework():
    """Create and initialize validation framework for testing."""
    framework = ValidationFramework()
    await framework.start()
    yield framework
    await framework.stop()


@pytest.fixture
def sample_innovation():
    """Create sample innovation for testing."""
    return Innovation(
        title="AI-Powered Healthcare Assistant",
        description="An AI system that assists healthcare professionals with diagnosis and treatment recommendations",
        category="Healthcare Technology",
        domain="Healthcare",
        technology_stack=["Python", "TensorFlow", "React", "PostgreSQL", "AWS"],
        target_market="Healthcare providers",
        problem_statement="Healthcare professionals need better tools for accurate diagnosis",
        proposed_solution="AI-powered assistant that analyzes patient data and provides recommendations",
        unique_value_proposition="Reduces diagnostic errors by 40% and improves treatment outcomes",
        competitive_advantages=["Advanced AI algorithms", "Integration with existing systems"],
        estimated_timeline="18 months",
        estimated_cost=2000000.0,
        potential_revenue=10000000.0,
        risk_factors=["Regulatory approval", "Data privacy concerns"],
        success_metrics=["Diagnostic accuracy", "User adoption rate", "Patient outcomes"]
    )


@pytest.fixture
def sample_validation_request(sample_innovation):
    """Create sample validation request for testing."""
    return ValidationRequest(
        innovation_id=sample_innovation.id,
        validation_types=[
            ValidationType.TECHNICAL_FEASIBILITY,
            ValidationType.MARKET_VIABILITY,
            ValidationType.RISK_ASSESSMENT
        ],
        priority="high",
        requester="test_user"
    )


class TestValidationFramework:
    """Test cases for ValidationFramework class."""
    
    @pytest.mark.asyncio
    async def test_initialization(self, validation_framework):
        """Test validation framework initialization."""
        assert validation_framework.engine_id == "validation_framework"
        assert validation_framework.name == "Innovation Validation Framework"
        assert len(validation_framework.validation_methodologies) > 0
        assert len(validation_framework.validation_criteria) > 0
        assert validation_framework.validation_context is not None
    
    @pytest.mark.asyncio
    async def test_validate_innovation(self, validation_framework, sample_innovation):
        """Test innovation validation process."""
        # Validate innovation
        report = await validation_framework.validate_innovation(sample_innovation)
        
        # Verify report structure
        assert isinstance(report, ValidationReport)
        assert report.innovation_id == sample_innovation.id
        assert 0.0 <= report.overall_score <= 1.0
        assert report.overall_result in ValidationResult
        assert 0.0 <= report.confidence_level <= 1.0
        assert len(report.validation_scores) > 0
        assert report.completed_at is not None
        
        # Verify validation scores
        for score in report.validation_scores:
            assert isinstance(score, ValidationScore)
            assert 0.0 <= score.score <= 1.0
            assert 0.0 <= score.confidence <= 1.0
            assert score.reasoning != ""
    
    @pytest.mark.asyncio
    async def test_select_validation_methodology(self, validation_framework, sample_innovation):
        """Test validation methodology selection."""
        validation_types = [ValidationType.TECHNICAL_FEASIBILITY, ValidationType.MARKET_VIABILITY]
        
        methodology = await validation_framework.select_validation_methodology(
            sample_innovation, validation_types
        )
        
        assert methodology is not None
        assert methodology.name != ""
        assert len(methodology.validation_types) > 0
        assert methodology.accuracy_rate > 0.0
        assert methodology.confidence_level > 0.0
    
    @pytest.mark.asyncio
    async def test_execute_validation(self, validation_framework, sample_innovation):
        """Test validation execution."""
        # Get methodology
        methodology = await validation_framework.select_validation_methodology(
            sample_innovation, [ValidationType.TECHNICAL_FEASIBILITY]
        )
        
        # Execute validation
        validation_scores = await validation_framework.execute_validation(
            sample_innovation, methodology
        )
        
        assert len(validation_scores) > 0
        for score in validation_scores:
            assert isinstance(score, ValidationScore)
            assert 0.0 <= score.score <= 1.0
            assert score.reasoning != ""
    
    @pytest.mark.asyncio
    async def test_analyze_validation_results(self, validation_framework, sample_innovation):
        """Test validation results analysis."""
        # Create sample validation scores
        validation_scores = [
            ValidationScore(
                criteria_id="test_criteria_1",
                score=0.8,
                confidence=0.9,
                reasoning="High technical feasibility"
            ),
            ValidationScore(
                criteria_id="test_criteria_2",
                score=0.6,
                confidence=0.7,
                reasoning="Moderate market viability"
            )
        ]
        
        # Analyze results
        report = await validation_framework.analyze_validation_results(
            sample_innovation, validation_scores
        )
        
        assert isinstance(report, ValidationReport)
        assert report.innovation_id == sample_innovation.id
        assert 0.0 <= report.overall_score <= 1.0
        assert len(report.strengths) >= 0
        assert len(report.weaknesses) >= 0
        assert len(report.recommendations) >= 0
        assert len(report.next_steps) >= 0
    
    @pytest.mark.asyncio
    async def test_process_validation_request(self, validation_framework, sample_validation_request, sample_innovation):
        """Test validation request processing."""
        # Process request
        report = await validation_framework.process(
            sample_validation_request,
            {"innovation": sample_innovation}
        )
        
        assert isinstance(report, ValidationReport)
        assert report.innovation_id == sample_innovation.id
        assert report.request_id == sample_validation_request.id
        assert report.completed_at is not None
    
    @pytest.mark.asyncio
    async def test_validation_with_different_types(self, validation_framework, sample_innovation):
        """Test validation with different validation types."""
        validation_types = [
            ValidationType.TECHNICAL_FEASIBILITY,
            ValidationType.MARKET_VIABILITY,
            ValidationType.RISK_ASSESSMENT
        ]
        
        for validation_type in validation_types:
            report = await validation_framework.validate_innovation(
                sample_innovation, [validation_type]
            )
            
            assert isinstance(report, ValidationReport)
            assert report.overall_score >= 0.0
            assert len(report.validation_scores) > 0
    
    @pytest.mark.asyncio
    async def test_historical_validations_tracking(self, validation_framework, sample_innovation):
        """Test that validations are tracked in history."""
        initial_count = len(validation_framework.historical_validations)
        
        # Perform validation
        await validation_framework.validate_innovation(sample_innovation)
        
        # Check that validation was added to history
        assert len(validation_framework.historical_validations) == initial_count + 1
    
    @pytest.mark.asyncio
    async def test_get_status(self, validation_framework):
        """Test getting framework status."""
        status = validation_framework.get_status()
        
        assert isinstance(status, dict)
        assert "healthy" in status
        assert "methodologies_loaded" in status
        assert "criteria_loaded" in status
        assert "historical_validations" in status
        assert "context_initialized" in status
        assert status["healthy"] is True
    
    @pytest.mark.asyncio
    async def test_health_check(self, validation_framework):
        """Test framework health check."""
        is_healthy = await validation_framework.health_check()
        assert is_healthy is True
    
    @pytest.mark.asyncio
    async def test_cleanup(self, validation_framework):
        """Test framework cleanup."""
        # Add some data
        await validation_framework.validate_innovation(Innovation(title="Test"))
        
        # Verify data exists
        assert len(validation_framework.historical_validations) > 0
        
        # Cleanup
        await validation_framework.cleanup()
        
        # Verify cleanup
        assert len(validation_framework.validation_methodologies) == 0
        assert len(validation_framework.validation_criteria) == 0
        assert len(validation_framework.historical_validations) == 0


class TestValidationModels:
    """Test cases for validation data models."""
    
    def test_innovation_creation(self):
        """Test Innovation model creation."""
        innovation = Innovation(
            title="Test Innovation",
            description="Test description",
            category="Technology"
        )
        
        assert innovation.title == "Test Innovation"
        assert innovation.description == "Test description"
        assert innovation.category == "Technology"
        assert innovation.id != ""
        assert isinstance(innovation.created_at, datetime)
    
    def test_validation_request_creation(self):
        """Test ValidationRequest model creation."""
        request = ValidationRequest(
            innovation_id="test_innovation",
            validation_types=[ValidationType.TECHNICAL_FEASIBILITY],
            requester="test_user"
        )
        
        assert request.innovation_id == "test_innovation"
        assert ValidationType.TECHNICAL_FEASIBILITY in request.validation_types
        assert request.requester == "test_user"
        assert request.status == ValidationStatus.PENDING
    
    def test_validation_score_creation(self):
        """Test ValidationScore model creation."""
        score = ValidationScore(
            criteria_id="test_criteria",
            score=0.8,
            confidence=0.9,
            reasoning="Test reasoning"
        )
        
        assert score.criteria_id == "test_criteria"
        assert score.score == 0.8
        assert score.confidence == 0.9
        assert score.reasoning == "Test reasoning"
        assert isinstance(score.timestamp, datetime)
    
    def test_validation_report_creation(self):
        """Test ValidationReport model creation."""
        report = ValidationReport(
            innovation_id="test_innovation",
            overall_score=0.75,
            overall_result=ValidationResult.APPROVED
        )
        
        assert report.innovation_id == "test_innovation"
        assert report.overall_score == 0.75
        assert report.overall_result == ValidationResult.APPROVED
        assert isinstance(report.created_at, datetime)


class TestValidationIntegration:
    """Integration tests for validation framework."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_validation_flow(self, validation_framework, sample_innovation):
        """Test complete validation flow from innovation to report."""
        # Step 1: Create validation request
        request = ValidationRequest(
            innovation_id=sample_innovation.id,
            validation_types=[ValidationType.TECHNICAL_FEASIBILITY, ValidationType.MARKET_VIABILITY],
            requester="integration_test"
        )
        
        # Step 2: Process validation
        report = await validation_framework.process(request, {"innovation": sample_innovation})
        
        # Step 3: Verify complete flow
        assert isinstance(report, ValidationReport)
        assert report.innovation_id == sample_innovation.id
        assert report.request_id == request.id
        assert report.overall_score >= 0.0
        assert report.overall_result in ValidationResult
        assert len(report.validation_scores) > 0
        assert len(report.recommendations) >= 0
        assert report.completed_at is not None
        
        # Step 4: Verify report is in history
        assert report in validation_framework.historical_validations
    
    @pytest.mark.asyncio
    async def test_multiple_concurrent_validations(self, validation_framework):
        """Test handling multiple concurrent validations."""
        innovations = [
            Innovation(title=f"Innovation {i}", domain="Technology")
            for i in range(3)
        ]
        
        # Start concurrent validations
        tasks = [
            validation_framework.validate_innovation(innovation)
            for innovation in innovations
        ]
        
        # Wait for all to complete
        reports = await asyncio.gather(*tasks)
        
        # Verify all completed successfully
        assert len(reports) == 3
        for report in reports:
            assert isinstance(report, ValidationReport)
            assert report.completed_at is not None
    
    @pytest.mark.asyncio
    async def test_validation_with_missing_data(self, validation_framework):
        """Test validation with incomplete innovation data."""
        incomplete_innovation = Innovation(
            title="Incomplete Innovation",
            # Missing many fields
        )
        
        # Should still complete validation
        report = await validation_framework.validate_innovation(incomplete_innovation)
        
        assert isinstance(report, ValidationReport)
        assert report.innovation_id == incomplete_innovation.id
        # May have lower confidence due to missing data
        assert 0.0 <= report.confidence_level <= 1.0


if __name__ == "__main__":
    pytest.main([__file__])