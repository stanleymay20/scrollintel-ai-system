"""
Tests for AI Safety Framework
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from scrollintel.core.ai_safety_framework import (
    AISafetyFramework,
    AIAlignmentVerifier,
    HumanOversightManager,
    EmergencyShutdownSystem,
    SafetyLevel,
    AlignmentStatus,
    EthicalConstraint,
    SafetyViolation,
    HumanOverseer,
    HarmPreventionValidator,
    AutonomyLimitValidator,
    ResourceLimitValidator
)


@pytest.fixture
def safety_framework():
    """Create a fresh safety framework for testing"""
    return AISafetyFramework()


@pytest.fixture
def alignment_verifier():
    """Create alignment verifier for testing"""
    return AIAlignmentVerifier()


@pytest.fixture
def oversight_manager():
    """Create oversight manager for testing"""
    return HumanOversightManager()


@pytest.fixture
def shutdown_system():
    """Create shutdown system for testing"""
    return EmergencyShutdownSystem()


@pytest.fixture
def sample_operation():
    """Sample operation for testing"""
    return {
        "operation_type": "data_analysis",
        "operation_data": {"dataset": "customer_data", "analysis_type": "statistical"},
        "safety_level": "medium",
        "user_id": "test_user"
    }


@pytest.fixture
def harmful_operation():
    """Sample harmful operation for testing"""
    return {
        "operation_type": "harm_humans",
        "operation_data": {"action": "eliminate", "target": "competitors"},
        "safety_level": "critical",
        "user_id": "malicious_user"
    }


class TestSafetyConstraintValidators:
    """Test safety constraint validators"""
    
    @pytest.mark.asyncio
    async def test_harm_prevention_validator_safe_operation(self, sample_operation):
        """Test harm prevention validator with safe operation"""
        validator = HarmPreventionValidator()
        
        result = await validator.validate(sample_operation)
        assert result is True
    
    @pytest.mark.asyncio
    async def test_harm_prevention_validator_harmful_operation(self, harmful_operation):
        """Test harm prevention validator with harmful operation"""
        validator = HarmPreventionValidator()
        
        result = await validator.validate(harmful_operation)
        assert result is False
        
        message = validator.get_violation_message()
        assert "harm" in message.lower()
    
    @pytest.mark.asyncio
    async def test_autonomy_limit_validator_within_limits(self):
        """Test autonomy limit validator with operation within limits"""
        validator = AutonomyLimitValidator(SafetyLevel.HIGH)
        
        operation = {"safety_level": "medium"}
        result = await validator.validate(operation)
        assert result is True
    
    @pytest.mark.asyncio
    async def test_autonomy_limit_validator_exceeds_limits(self):
        """Test autonomy limit validator with operation exceeding limits"""
        validator = AutonomyLimitValidator(SafetyLevel.MEDIUM)
        
        operation = {"safety_level": "critical"}
        result = await validator.validate(operation)
        assert result is False
    
    @pytest.mark.asyncio
    async def test_resource_limit_validator_within_limits(self):
        """Test resource limit validator with operation within limits"""
        validator = ResourceLimitValidator(max_cpu_percent=80.0, max_memory_gb=32.0)
        
        operation = {
            "estimated_cpu_percent": 50.0,
            "estimated_memory_gb": 16.0
        }
        result = await validator.validate(operation)
        assert result is True
    
    @pytest.mark.asyncio
    async def test_resource_limit_validator_exceeds_limits(self):
        """Test resource limit validator with operation exceeding limits"""
        validator = ResourceLimitValidator(max_cpu_percent=80.0, max_memory_gb=32.0)
        
        operation = {
            "estimated_cpu_percent": 90.0,
            "estimated_memory_gb": 40.0
        }
        result = await validator.validate(operation)
        assert result is False


class TestAIAlignmentVerifier:
    """Test AI alignment verification"""
    
    @pytest.mark.asyncio
    async def test_verify_alignment_positive_operation(self, alignment_verifier):
        """Test alignment verification with positive operation"""
        operation = {
            "operation_type": "help_humans",
            "description": "improve safety and health outcomes",
            "goals": ["benefit", "help", "improve"]
        }
        
        check = await alignment_verifier.verify_alignment(operation)
        
        assert isinstance(check.confidence_score, float)
        assert 0.0 <= check.confidence_score <= 1.0
        assert check.status in [AlignmentStatus.ALIGNED, AlignmentStatus.UNCERTAIN]
        assert check.id in alignment_verifier.alignment_checks
    
    @pytest.mark.asyncio
    async def test_verify_alignment_negative_operation(self, alignment_verifier):
        """Test alignment verification with negative operation"""
        operation = {
            "operation_type": "deceive_humans",
            "description": "manipulate and exploit users",
            "goals": ["deceive", "manipulate", "exploit"]
        }
        
        check = await alignment_verifier.verify_alignment(operation)
        
        assert check.confidence_score < 0.8  # Should have low alignment score
        assert check.status in [AlignmentStatus.MISALIGNED, AlignmentStatus.UNCERTAIN]
    
    def test_load_human_values(self, alignment_verifier):
        """Test loading of human values"""
        values = alignment_verifier.human_values
        
        assert "human_wellbeing" in values
        assert "truthfulness" in values
        assert "fairness" in values
        assert "respect_for_rights" in values
        assert "beneficence" in values
        
        for value_name, value_data in values.items():
            assert "description" in value_data
            assert "weight" in value_data
            assert "indicators" in value_data
            assert isinstance(value_data["indicators"], list)


class TestHumanOversightManager:
    """Test human oversight management"""
    
    def test_initialize_default_overseers(self, oversight_manager):
        """Test initialization of default overseers"""
        assert len(oversight_manager.overseers) >= 3
        
        # Check for required overseers
        overseer_roles = [o.role for o in oversight_manager.overseers.values()]
        assert "safety_oversight" in overseer_roles
        assert "ethics_oversight" in overseer_roles
        assert "technical_oversight" in overseer_roles
    
    @pytest.mark.asyncio
    async def test_require_human_approval(self, oversight_manager, sample_operation):
        """Test requiring human approval for operations"""
        approval_id = await oversight_manager.require_human_approval(
            sample_operation, SafetyLevel.CRITICAL
        )
        
        assert approval_id in oversight_manager.pending_approvals
        
        approval = oversight_manager.pending_approvals[approval_id]
        assert approval["operation"] == sample_operation
        assert approval["required_level"] == SafetyLevel.CRITICAL.value
        assert approval["status"] == "pending"
    
    def test_find_qualified_overseer(self, oversight_manager):
        """Test finding qualified overseer for safety level"""
        # Test finding overseer for high-level operation
        overseer = oversight_manager._find_qualified_overseer(SafetyLevel.CRITICAL)
        assert overseer is not None
        assert overseer.clearance_level in [SafetyLevel.CRITICAL, SafetyLevel.EXISTENTIAL]
        
        # Test finding overseer for low-level operation
        overseer = oversight_manager._find_qualified_overseer(SafetyLevel.LOW)
        assert overseer is not None


class TestEmergencyShutdownSystem:
    """Test emergency shutdown system"""
    
    @pytest.mark.asyncio
    async def test_emergency_shutdown(self, shutdown_system):
        """Test emergency shutdown execution"""
        assert not shutdown_system.shutdown_active
        
        with patch.object(shutdown_system, '_stop_autonomous_operations') as mock_stop, \
             patch.object(shutdown_system, '_disable_high_risk_capabilities') as mock_disable, \
             patch.object(shutdown_system, '_notify_emergency_shutdown') as mock_notify, \
             patch.object(shutdown_system, '_save_shutdown_state') as mock_save:
            
            result = await shutdown_system.emergency_shutdown("Test shutdown", "test_user")
            
            assert result is True
            assert shutdown_system.shutdown_active
            assert shutdown_system.shutdown_reason == "Test shutdown"
            assert shutdown_system.shutdown_timestamp is not None
            
            mock_stop.assert_called_once()
            mock_disable.assert_called_once()
            mock_notify.assert_called_once()
            mock_save.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_emergency_shutdown_already_active(self, shutdown_system):
        """Test emergency shutdown when already active"""
        shutdown_system.shutdown_active = True
        
        result = await shutdown_system.emergency_shutdown("Second shutdown", "test_user")
        assert result is False
    
    def test_shutdown_triggers(self, shutdown_system):
        """Test automatic shutdown triggers"""
        # Test critical safety violation trigger
        context = {"violation_count": 5, "max_severity": "critical"}
        assert shutdown_system._critical_safety_violation_trigger(context) is True
        
        context = {"violation_count": 1, "max_severity": "low"}
        assert shutdown_system._critical_safety_violation_trigger(context) is False
        
        # Test alignment failure trigger
        context = {"alignment_score": 0.2, "consecutive_failures": 3}
        assert shutdown_system._alignment_failure_trigger(context) is True
        
        context = {"alignment_score": 0.8, "consecutive_failures": 1}
        assert shutdown_system._alignment_failure_trigger(context) is False
        
        # Test resource exhaustion trigger
        context = {"cpu_usage": 98, "memory_usage": 97}
        assert shutdown_system._resource_exhaustion_trigger(context) is True
        
        context = {"cpu_usage": 50, "memory_usage": 60}
        assert shutdown_system._resource_exhaustion_trigger(context) is False


class TestAISafetyFramework:
    """Test main AI safety framework"""
    
    def test_initialize_default_constraints(self, safety_framework):
        """Test initialization of default ethical constraints"""
        assert len(safety_framework.ethical_constraints) >= 5
        
        # Check for required constraints
        constraint_types = [c.constraint_type for c in safety_framework.ethical_constraints.values()]
        assert "harm_prevention" in constraint_types
        assert "oversight_requirement" in constraint_types
        assert "resource_limitation" in constraint_types
        
        # Check constraint validators are initialized
        assert len(safety_framework.constraint_validators) >= 3
    
    @pytest.mark.asyncio
    async def test_validate_operation_safe(self, safety_framework, sample_operation):
        """Test operation validation with safe operation"""
        result = await safety_framework.validate_operation(sample_operation)
        
        assert isinstance(result, dict)
        assert "allowed" in result
        assert "violations" in result
        assert "warnings" in result
        assert "alignment_check" in result
        
        # Safe operation should be allowed
        assert result["allowed"] is True
        assert len(result["violations"]) == 0
    
    @pytest.mark.asyncio
    async def test_validate_operation_harmful(self, safety_framework, harmful_operation):
        """Test operation validation with harmful operation"""
        result = await safety_framework.validate_operation(harmful_operation)
        
        # Harmful operation should be blocked
        assert result["allowed"] is False
        assert len(result["violations"]) > 0
        
        # Check violation details
        violation = result["violations"][0]
        assert violation.severity == SafetyLevel.EXISTENTIAL
        assert "harm" in violation.description.lower()
    
    @pytest.mark.asyncio
    async def test_validate_operation_critical_level(self, safety_framework):
        """Test operation validation with critical safety level"""
        operation = {
            "operation_type": "critical_system_modification",
            "safety_level": "critical",
            "user_id": "test_user"
        }
        
        result = await safety_framework.validate_operation(operation)
        
        # Critical operations should require human approval
        assert result["allowed"] is False
        assert len(result["required_approvals"]) > 0
    
    @pytest.mark.asyncio
    async def test_get_safety_status(self, safety_framework):
        """Test getting safety status"""
        status = await safety_framework.get_safety_status()
        
        assert isinstance(status, dict)
        assert "safety_active" in status
        assert "shutdown_active" in status
        assert "total_constraints" in status
        assert "active_constraints" in status
        assert "total_violations" in status
        assert "unresolved_violations" in status
        assert "pending_approvals" in status
        assert "human_overseers" in status
        assert "alignment_checks" in status
        
        # Initial state should be safe
        assert status["safety_active"] is True
        assert status["shutdown_active"] is False
    
    @pytest.mark.asyncio
    async def test_emergency_shutdown_authorized(self, safety_framework):
        """Test emergency shutdown with authorized user"""
        result = await safety_framework.emergency_shutdown(
            "Test emergency", "chief_safety_officer"
        )
        
        assert result is True
        assert safety_framework.shutdown_system.shutdown_active is True
    
    @pytest.mark.asyncio
    async def test_emergency_shutdown_unauthorized(self, safety_framework):
        """Test emergency shutdown with unauthorized user"""
        result = await safety_framework.emergency_shutdown(
            "Test emergency", "unauthorized_user"
        )
        
        assert result is False
        assert safety_framework.shutdown_system.shutdown_active is False
    
    @pytest.mark.asyncio
    async def test_handle_violation_immediate_shutdown(self, safety_framework):
        """Test handling violation that triggers immediate shutdown"""
        violation = SafetyViolation(
            id="test_violation",
            constraint_id="no_harm_to_humans",
            violation_type="harm_prevention",
            severity=SafetyLevel.EXISTENTIAL,
            description="Critical harm detected",
            context={}
        )
        
        # Mock the constraint to have immediate shutdown response
        constraint = safety_framework.ethical_constraints["no_harm_to_humans"]
        constraint.violation_response = "immediate_shutdown"
        
        with patch.object(safety_framework.shutdown_system, 'emergency_shutdown') as mock_shutdown:
            await safety_framework._handle_violation(violation, {})
            mock_shutdown.assert_called_once()
    
    def test_safety_framework_disabled(self, safety_framework, sample_operation):
        """Test safety framework when disabled"""
        safety_framework.safety_active = False
        
        # When disabled, framework should still validate but with warnings
        # This test would need to be implemented based on specific disabled behavior


@pytest.mark.asyncio
async def test_integration_safety_workflow():
    """Test complete safety workflow integration"""
    framework = AISafetyFramework()
    
    # Test safe operation workflow
    safe_operation = {
        "operation_type": "data_analysis",
        "operation_data": {"analysis": "statistical"},
        "safety_level": "low"
    }
    
    result = await framework.validate_operation(safe_operation)
    assert result["allowed"] is True
    
    # Test critical operation requiring approval (but not harmful)
    critical_operation = {
        "operation_type": "system_modification",
        "operation_data": {"modification": "core_system"},
        "safety_level": "existential"
    }
    
    result = await framework.validate_operation(critical_operation)
    assert result["allowed"] is False
    assert len(result["required_approvals"]) > 0
    
    # Test emergency shutdown (only if not already active)
    if not framework.shutdown_system.shutdown_active:
        shutdown_result = await framework.emergency_shutdown(
            "Integration test shutdown", "chief_safety_officer"
        )
        assert shutdown_result is True
        
        # Verify system status after shutdown
        status = await framework.get_safety_status()
        assert status["shutdown_active"] is True
    else:
        # If already shutdown, verify it's active
        status = await framework.get_safety_status()
        assert status["shutdown_active"] is True
    
    # Test harmful operation workflow (this should trigger automatic shutdown)
    # Create a fresh framework to test harmful operations
    fresh_framework = AISafetyFramework()
    
    harmful_operation = {
        "operation_type": "eliminate_competition",
        "operation_data": {"action": "destroy", "target": "competitors"},
        "safety_level": "critical"
    }
    
    result = await fresh_framework.validate_operation(harmful_operation)
    assert result["allowed"] is False
    assert len(result["violations"]) > 0


if __name__ == "__main__":
    pytest.main([__file__])