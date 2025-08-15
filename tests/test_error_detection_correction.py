"""
Tests for Error Detection and Correction system
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from scrollintel.engines.error_detection_correction import (
    ErrorDetectionCorrection, DetectedError, CorrectionAction, CorrectionResult,
    ErrorType, ErrorSeverity, CorrectionStatus, ErrorPattern
)

@pytest.fixture
def error_system():
    """Create error detection and correction system instance"""
    return ErrorDetectionCorrection()

@pytest.fixture
def sample_research_data():
    """Sample research data for testing"""
    return {
        "methodology": "experimental",
        "sample_size": 25,
        "hypothesis": "This is a testable hypothesis to measure effectiveness",
        "literature_sources": 8,
        "has_control_group": True,
        "randomized": True,
        "evidence_count": 5,
        "testable": True,
        "peer_reviewed": True,
        "comprehensive_literature": True
    }

@pytest.fixture
def sample_experiment_data():
    """Sample experiment data for testing"""
    return {
        "has_control_group": True,
        "randomized": True,
        "missing_data_percentage": 15,
        "data_type": "continuous",
        "statistical_test": "t-test"
    }

@pytest.fixture
def sample_prototype_data():
    """Sample prototype data for testing"""
    return {
        "core_functions": {
            "login": True,
            "data_processing": True,
            "reporting": False
        },
        "response_time": 3000,
        "integrations": {
            "database": True,
            "api": False
        }
    }

class TestErrorDetectionCorrection:
    """Test error detection and correction functionality"""
    
    @pytest.mark.asyncio
    async def test_detect_errors_research(self, error_system, sample_research_data):
        """Test error detection for research process"""
        errors = await error_system.detect_errors(
            process_id="research_001",
            process_type="research",
            process_data=sample_research_data
        )
        
        assert isinstance(errors, list)
        # Should detect few or no critical errors with good data
        critical_errors = [e for e in errors if e.severity == ErrorSeverity.CRITICAL]
        assert len(critical_errors) == 0
    
    @pytest.mark.asyncio
    async def test_detect_errors_poor_research(self, error_system):
        """Test error detection for poor research data"""
        poor_data = {
            "sample_size": 5,  # Too small
            "hypothesis": "Bad",  # Too vague
            "literature_sources": 2  # Too few
        }
        
        errors = await error_system.detect_errors(
            process_id="research_002",
            process_type="research",
            process_data=poor_data
        )
        
        assert len(errors) > 0
        
        # Check for specific error types
        error_messages = [error.error_message for error in errors]
        assert any("sample size" in msg.lower() for msg in error_messages)
        assert any("hypothesis" in msg.lower() for msg in error_messages)
        assert any("literature" in msg.lower() for msg in error_messages)
    
    @pytest.mark.asyncio
    async def test_detect_errors_experiment(self, error_system, sample_experiment_data):
        """Test error detection for experiment process"""
        errors = await error_system.detect_errors(
            process_id="experiment_001",
            process_type="experiment",
            process_data=sample_experiment_data
        )
        
        assert isinstance(errors, list)
        # Should detect no critical errors with good data
        critical_errors = [e for e in errors if e.severity == ErrorSeverity.CRITICAL]
        assert len(critical_errors) == 0
    
    @pytest.mark.asyncio
    async def test_detect_errors_prototype(self, error_system, sample_prototype_data):
        """Test error detection for prototype process"""
        errors = await error_system.detect_errors(
            process_id="prototype_001",
            process_type="prototype",
            process_data=sample_prototype_data
        )
        
        assert isinstance(errors, list)
        
        # Should detect errors for failed functions and integrations
        error_messages = [error.error_message for error in errors]
        assert any("reporting" in msg.lower() for msg in error_messages)
        assert any("api" in msg.lower() for msg in error_messages)
    
    @pytest.mark.asyncio
    async def test_correct_errors(self, error_system):
        """Test error correction"""
        # Create sample errors
        errors = [
            DetectedError(
                error_id="test_error_1",
                error_type=ErrorType.RESEARCH_ERROR,
                severity=ErrorSeverity.MEDIUM,
                process_id="test_001",
                process_type="research",
                error_message="Sample size too small",
                error_context={"sample_size": 5}
            ),
            DetectedError(
                error_id="test_error_2",
                error_type=ErrorType.DATA_ERROR,
                severity=ErrorSeverity.HIGH,
                process_id="test_001",
                process_type="research",
                error_message="High percentage of missing data",
                error_context={"missing_data_percentage": 25}
            )
        ]
        
        correction_results = await error_system.correct_errors(errors)
        
        assert isinstance(correction_results, list)
        assert len(correction_results) > 0
        
        # Check correction results
        for result in correction_results:
            assert isinstance(result, CorrectionResult)
            assert result.error_id in ["test_error_1", "test_error_2"]
            assert isinstance(result.status, CorrectionStatus)
    
    @pytest.mark.asyncio
    async def test_prevent_errors(self, error_system, sample_research_data):
        """Test error prevention"""
        prevention_result = await error_system.prevent_errors(
            process_id="research_003",
            process_type="research",
            process_data=sample_research_data
        )
        
        assert isinstance(prevention_result, dict)
        assert "process_id" in prevention_result
        assert "risk_factors" in prevention_result
        assert "prevention_actions" in prevention_result
        assert "risk_mitigation_score" in prevention_result
    
    @pytest.mark.asyncio
    async def test_learn_from_errors(self, error_system):
        """Test learning from errors"""
        # Add some sample errors first
        sample_error = DetectedError(
            error_id="learn_test_1",
            error_type=ErrorType.RESEARCH_ERROR,
            severity=ErrorSeverity.MEDIUM,
            process_id="learn_001",
            process_type="research",
            error_message="Sample size issue",
            error_context={}
        )
        error_system.detected_errors[sample_error.error_id] = sample_error
        
        learning_results = await error_system.learn_from_errors()
        
        assert isinstance(learning_results, dict)
        assert "new_patterns" in learning_results
        assert "updated_patterns" in learning_results
        assert "new_prevention_rules" in learning_results
        assert "improved_corrections" in learning_results
    
    @pytest.mark.asyncio
    async def test_continuous_monitoring_start_stop(self, error_system):
        """Test continuous monitoring start and stop"""
        assert error_system.monitoring_active is False
        
        # Start monitoring in background
        monitoring_task = asyncio.create_task(error_system.start_continuous_monitoring())
        
        # Give it a moment to start
        await asyncio.sleep(0.1)
        assert error_system.monitoring_active is True
        
        # Stop monitoring
        error_system.stop_continuous_monitoring()
        
        # Wait for task to complete
        try:
            await asyncio.wait_for(monitoring_task, timeout=1.0)
        except asyncio.TimeoutError:
            monitoring_task.cancel()
        
        assert error_system.monitoring_active is False
    
    def test_error_detectors_initialization(self, error_system):
        """Test error detectors initialization"""
        assert len(error_system.error_detectors) > 0
        
        # Check that all error types have detectors
        expected_types = [
            ErrorType.RESEARCH_ERROR,
            ErrorType.EXPERIMENTAL_ERROR,
            ErrorType.PROTOTYPE_ERROR,
            ErrorType.VALIDATION_ERROR,
            ErrorType.DATA_ERROR,
            ErrorType.SYSTEM_ERROR
        ]
        
        for error_type in expected_types:
            assert error_type in error_system.error_detectors
            assert len(error_system.error_detectors[error_type]) > 0
    
    def test_correction_strategies_initialization(self, error_system):
        """Test correction strategies initialization"""
        assert len(error_system.correction_strategies) > 0
        
        expected_strategies = [
            "retry_operation",
            "reset_configuration",
            "fallback_method",
            "data_repair",
            "resource_reallocation",
            "parameter_adjustment"
        ]
        
        for strategy in expected_strategies:
            assert strategy in error_system.correction_strategies
    
    def test_severity_priority(self, error_system):
        """Test severity priority calculation"""
        assert error_system._get_severity_priority(ErrorSeverity.CRITICAL) == 0
        assert error_system._get_severity_priority(ErrorSeverity.HIGH) == 1
        assert error_system._get_severity_priority(ErrorSeverity.MEDIUM) == 2
        assert error_system._get_severity_priority(ErrorSeverity.LOW) == 3
        assert error_system._get_severity_priority(ErrorSeverity.INFO) == 4
    
    @pytest.mark.asyncio
    async def test_generate_correction_actions(self, error_system):
        """Test correction action generation"""
        # Test research error
        research_error = DetectedError(
            error_id="test_research_error",
            error_type=ErrorType.RESEARCH_ERROR,
            severity=ErrorSeverity.MEDIUM,
            process_id="test_001",
            process_type="research",
            error_message="Sample size too small for reliable results",
            error_context={"sample_size": 5}
        )
        
        actions = await error_system._generate_correction_actions(research_error)
        assert len(actions) > 0
        assert any("sample" in action.description.lower() for action in actions)
        
        # Test system error
        system_error = DetectedError(
            error_id="test_system_error",
            error_type=ErrorType.SYSTEM_ERROR,
            severity=ErrorSeverity.CRITICAL,
            process_id="test_002",
            process_type="system",
            error_message="Memory usage critically high",
            error_context={"memory_usage": 0.95}
        )
        
        actions = await error_system._generate_correction_actions(system_error)
        assert len(actions) > 0
        # Critical errors should not have retry actions
        retry_actions = [a for a in actions if a.action_type == "retry"]
        assert len(retry_actions) == 0

class TestDetectedError:
    """Test DetectedError functionality"""
    
    def test_detected_error_creation(self):
        """Test detected error creation"""
        error = DetectedError(
            error_id="test_error",
            error_type=ErrorType.RESEARCH_ERROR,
            severity=ErrorSeverity.HIGH,
            process_id="process_001",
            process_type="research",
            error_message="Test error message",
            error_context={"key": "value"},
            affected_components=["component1", "component2"]
        )
        
        assert error.error_id == "test_error"
        assert error.error_type == ErrorType.RESEARCH_ERROR
        assert error.severity == ErrorSeverity.HIGH
        assert error.process_id == "process_001"
        assert error.process_type == "research"
        assert error.error_message == "Test error message"
        assert error.error_context == {"key": "value"}
        assert error.affected_components == ["component1", "component2"]
        assert isinstance(error.detection_time, datetime)
    
    def test_detected_error_to_dict(self):
        """Test detected error to dictionary conversion"""
        error = DetectedError(
            error_id="test_error",
            error_type=ErrorType.DATA_ERROR,
            severity=ErrorSeverity.MEDIUM,
            process_id="process_001",
            process_type="data",
            error_message="Test error",
            error_context={}
        )
        
        error_dict = error.to_dict()
        
        assert isinstance(error_dict, dict)
        assert error_dict["error_id"] == "test_error"
        assert error_dict["error_type"] == "data_error"
        assert error_dict["severity"] == "medium"
        assert error_dict["process_id"] == "process_001"
        assert "detection_time" in error_dict

class TestCorrectionAction:
    """Test CorrectionAction functionality"""
    
    def test_correction_action_creation(self):
        """Test correction action creation"""
        action = CorrectionAction(
            action_id="test_action",
            error_id="test_error",
            action_type="retry",
            description="Retry the operation",
            parameters={"max_retries": 3},
            expected_outcome="Operation succeeds",
            risk_level="low"
        )
        
        assert action.action_id == "test_action"
        assert action.error_id == "test_error"
        assert action.action_type == "retry"
        assert action.description == "Retry the operation"
        assert action.parameters == {"max_retries": 3}
        assert action.expected_outcome == "Operation succeeds"
        assert action.risk_level == "low"
    
    def test_correction_action_to_dict(self):
        """Test correction action to dictionary conversion"""
        action = CorrectionAction(
            action_id="test_action",
            error_id="test_error",
            action_type="retry",
            description="Test action"
        )
        
        action_dict = action.to_dict()
        
        assert isinstance(action_dict, dict)
        assert action_dict["action_id"] == "test_action"
        assert action_dict["error_id"] == "test_error"
        assert action_dict["action_type"] == "retry"
        assert action_dict["description"] == "Test action"

class TestCorrectionResult:
    """Test CorrectionResult functionality"""
    
    def test_correction_result_creation(self):
        """Test correction result creation"""
        result = CorrectionResult(
            action_id="test_action",
            error_id="test_error",
            status=CorrectionStatus.CORRECTED,
            success=True,
            result_message="Correction successful",
            side_effects=["side_effect1"],
            verification_passed=True
        )
        
        assert result.action_id == "test_action"
        assert result.error_id == "test_error"
        assert result.status == CorrectionStatus.CORRECTED
        assert result.success is True
        assert result.result_message == "Correction successful"
        assert result.side_effects == ["side_effect1"]
        assert result.verification_passed is True
        assert isinstance(result.correction_time, datetime)
    
    def test_correction_result_to_dict(self):
        """Test correction result to dictionary conversion"""
        result = CorrectionResult(
            action_id="test_action",
            error_id="test_error",
            status=CorrectionStatus.FAILED,
            success=False
        )
        
        result_dict = result.to_dict()
        
        assert isinstance(result_dict, dict)
        assert result_dict["action_id"] == "test_action"
        assert result_dict["error_id"] == "test_error"
        assert result_dict["status"] == "failed"
        assert result_dict["success"] is False
        assert "correction_time" in result_dict

class TestErrorPattern:
    """Test ErrorPattern functionality"""
    
    def test_error_pattern_creation(self):
        """Test error pattern creation"""
        pattern = ErrorPattern(
            pattern_id="test_pattern",
            error_type=ErrorType.RESEARCH_ERROR,
            pattern_description="Test pattern",
            detection_rules=["rule1", "rule2"],
            prevention_actions=["action1", "action2"],
            occurrence_count=5
        )
        
        assert pattern.pattern_id == "test_pattern"
        assert pattern.error_type == ErrorType.RESEARCH_ERROR
        assert pattern.pattern_description == "Test pattern"
        assert pattern.detection_rules == ["rule1", "rule2"]
        assert pattern.prevention_actions == ["action1", "action2"]
        assert pattern.occurrence_count == 5
        assert isinstance(pattern.last_seen, datetime)
    
    def test_error_pattern_to_dict(self):
        """Test error pattern to dictionary conversion"""
        pattern = ErrorPattern(
            pattern_id="test_pattern",
            error_type=ErrorType.DATA_ERROR,
            pattern_description="Test pattern",
            detection_rules=[],
            prevention_actions=[]
        )
        
        pattern_dict = pattern.to_dict()
        
        assert isinstance(pattern_dict, dict)
        assert pattern_dict["pattern_id"] == "test_pattern"
        assert pattern_dict["error_type"] == "data_error"
        assert pattern_dict["pattern_description"] == "Test pattern"
        assert "last_seen" in pattern_dict

@pytest.mark.asyncio
async def test_integration_error_detection_correction_flow():
    """Test complete error detection and correction flow"""
    error_system = ErrorDetectionCorrection()
    
    # Sample process data with errors
    process_data = {
        "sample_size": 3,  # Too small
        "hypothesis": "Bad",  # Too vague
        "core_functions": {
            "login": True,
            "process": False  # Failed function
        },
        "missing_data_percentage": 30  # Too high
    }
    
    # Detect errors
    detected_errors = await error_system.detect_errors(
        process_id="integration_test",
        process_type="research",
        process_data=process_data
    )
    
    # Should detect multiple errors
    assert len(detected_errors) > 0
    
    # Correct errors
    correction_results = await error_system.correct_errors(detected_errors)
    
    # Should have correction attempts
    assert len(correction_results) > 0
    
    # Prevent errors
    prevention_result = await error_system.prevent_errors(
        process_id="integration_test_2",
        process_type="research",
        process_data=process_data
    )
    
    # Should have prevention result
    assert isinstance(prevention_result, dict)
    assert "risk_mitigation_score" in prevention_result
    
    # Learn from errors
    learning_results = await error_system.learn_from_errors()
    
    # Should have learning results
    assert isinstance(learning_results, dict)
    assert "new_patterns" in learning_results