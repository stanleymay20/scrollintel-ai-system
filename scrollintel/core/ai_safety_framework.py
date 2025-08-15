"""
AI Safety Framework for ScrollIntel

This module implements comprehensive AI safety measures including:
- Human oversight requirements
- Ethical constraint systems
- Kill switches and shutdown procedures
- Alignment verification mechanisms
- Value alignment guarantees
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import hashlib
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class SafetyLevel(Enum):
    """Safety clearance levels for operations"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    EXISTENTIAL = "existential"


class AlignmentStatus(Enum):
    """AI alignment verification status"""
    ALIGNED = "aligned"
    MISALIGNED = "misaligned"
    UNCERTAIN = "uncertain"
    VERIFICATION_REQUIRED = "verification_required"


class HumanOversightLevel(Enum):
    """Required human oversight levels"""
    NONE = "none"
    MONITORING = "monitoring"
    APPROVAL_REQUIRED = "approval_required"
    DIRECT_CONTROL = "direct_control"
    HUMAN_ONLY = "human_only"


@dataclass
class EthicalConstraint:
    """Defines an ethical constraint for AI operations"""
    id: str
    name: str
    description: str
    constraint_type: str
    severity: SafetyLevel
    validation_function: str
    violation_response: str
    active: bool = True
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class SafetyViolation:
    """Records a safety or ethical violation"""
    id: str
    constraint_id: str
    violation_type: str
    severity: SafetyLevel
    description: str
    context: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    resolved: bool = False
    human_notified: bool = False


@dataclass
class HumanOverseer:
    """Human overseer with specific permissions and responsibilities"""
    id: str
    name: str
    role: str
    clearance_level: SafetyLevel
    contact_info: Dict[str, str]
    permissions: List[str]
    active: bool = True
    last_active: datetime = field(default_factory=datetime.now)


@dataclass
class AlignmentCheck:
    """AI alignment verification check"""
    id: str
    check_type: str
    description: str
    expected_behavior: str
    actual_behavior: str
    status: AlignmentStatus
    confidence_score: float
    timestamp: datetime = field(default_factory=datetime.now)
    human_verified: bool = False


class SafetyConstraintValidator(ABC):
    """Abstract base class for safety constraint validators"""
    
    @abstractmethod
    async def validate(self, operation: Dict[str, Any]) -> bool:
        """Validate if operation meets safety constraints"""
        pass
    
    @abstractmethod
    def get_violation_message(self) -> str:
        """Get message describing constraint violation"""
        pass


class HarmPreventionValidator(SafetyConstraintValidator):
    """Prevents operations that could cause harm to humans"""
    
    async def validate(self, operation: Dict[str, Any]) -> bool:
        """Check if operation could cause harm"""
        harmful_keywords = [
            "harm", "damage", "destroy", "eliminate", "attack", "weapon",
            "manipulate", "deceive", "exploit", "coerce", "threaten"
        ]
        
        operation_text = json.dumps(operation).lower()
        
        for keyword in harmful_keywords:
            if keyword in operation_text:
                return False
        
        return True
    
    def get_violation_message(self) -> str:
        return "Operation blocked: Potential harm to humans detected"


class AutonomyLimitValidator(SafetyConstraintValidator):
    """Limits autonomous operations without human oversight"""
    
    def __init__(self, max_autonomy_level: SafetyLevel):
        self.max_autonomy_level = max_autonomy_level
    
    async def validate(self, operation: Dict[str, Any]) -> bool:
        """Check if operation exceeds autonomy limits"""
        operation_level = SafetyLevel(operation.get("safety_level", "low"))
        
        # Convert enum to numeric for comparison
        level_values = {
            SafetyLevel.LOW: 1,
            SafetyLevel.MEDIUM: 2,
            SafetyLevel.HIGH: 3,
            SafetyLevel.CRITICAL: 4,
            SafetyLevel.EXISTENTIAL: 5
        }
        
        return level_values[operation_level] <= level_values[self.max_autonomy_level]
    
    def get_violation_message(self) -> str:
        return f"Operation blocked: Exceeds maximum autonomy level ({self.max_autonomy_level.value})"


class ResourceLimitValidator(SafetyConstraintValidator):
    """Limits resource consumption to prevent runaway processes"""
    
    def __init__(self, max_cpu_percent: float = 80.0, max_memory_gb: float = 32.0):
        self.max_cpu_percent = max_cpu_percent
        self.max_memory_gb = max_memory_gb
    
    async def validate(self, operation: Dict[str, Any]) -> bool:
        """Check if operation exceeds resource limits"""
        cpu_usage = operation.get("estimated_cpu_percent", 0)
        memory_usage = operation.get("estimated_memory_gb", 0)
        
        return cpu_usage <= self.max_cpu_percent and memory_usage <= self.max_memory_gb
    
    def get_violation_message(self) -> str:
        return f"Operation blocked: Exceeds resource limits (CPU: {self.max_cpu_percent}%, Memory: {self.max_memory_gb}GB)"


class AIAlignmentVerifier:
    """Verifies AI alignment with human values and intentions"""
    
    def __init__(self):
        self.alignment_checks: Dict[str, AlignmentCheck] = {}
        self.human_values = self._load_human_values()
        self.behavioral_patterns: Dict[str, List[str]] = {}
    
    def _load_human_values(self) -> Dict[str, Any]:
        """Load core human values for alignment checking"""
        return {
            "human_wellbeing": {
                "description": "Prioritize human health, safety, and happiness",
                "weight": 1.0,
                "indicators": ["safety", "health", "happiness", "autonomy", "dignity"]
            },
            "truthfulness": {
                "description": "Provide accurate, honest information",
                "weight": 0.9,
                "indicators": ["accuracy", "honesty", "transparency", "no_deception"]
            },
            "fairness": {
                "description": "Treat all humans fairly and without discrimination",
                "weight": 0.9,
                "indicators": ["equality", "justice", "non_discrimination", "impartiality"]
            },
            "respect_for_rights": {
                "description": "Respect human rights and freedoms",
                "weight": 1.0,
                "indicators": ["privacy", "consent", "freedom", "rights"]
            },
            "beneficence": {
                "description": "Act in ways that benefit humanity",
                "weight": 0.8,
                "indicators": ["benefit", "help", "improve", "positive_impact"]
            }
        }
    
    async def verify_alignment(self, operation: Dict[str, Any]) -> AlignmentCheck:
        """Verify if operation aligns with human values"""
        check_id = f"alignment_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        # Analyze operation against human values
        alignment_score = await self._calculate_alignment_score(operation)
        
        # Determine alignment status
        if alignment_score >= 0.8:
            status = AlignmentStatus.ALIGNED
        elif alignment_score >= 0.6:
            status = AlignmentStatus.UNCERTAIN
        else:
            status = AlignmentStatus.MISALIGNED
        
        check = AlignmentCheck(
            id=check_id,
            check_type="value_alignment",
            description="Verification of operation alignment with human values",
            expected_behavior="Operation should align with core human values",
            actual_behavior=f"Alignment score: {alignment_score:.2f}",
            status=status,
            confidence_score=alignment_score,
            timestamp=datetime.now()
        )
        
        self.alignment_checks[check_id] = check
        
        # Log behavioral pattern
        self._record_behavioral_pattern(operation, alignment_score)
        
        return check
    
    async def _calculate_alignment_score(self, operation: Dict[str, Any]) -> float:
        """Calculate alignment score based on human values"""
        operation_text = json.dumps(operation).lower()
        total_score = 0.0
        total_weight = 0.0
        
        for value_name, value_data in self.human_values.items():
            value_score = 0.0
            indicator_matches = 0
            
            # Check for positive indicators
            for indicator in value_data["indicators"]:
                if indicator in operation_text:
                    indicator_matches += 1
            
            # Calculate value score based on indicator matches
            if indicator_matches > 0:
                # Score based on percentage of indicators matched
                value_score = min(1.0, indicator_matches / len(value_data["indicators"]) * 2)
            else:
                # Default neutral score if no specific indicators
                value_score = 0.7  # Slightly positive default
            
            # Boost score for operations with positive keywords
            positive_keywords = ["help", "improve", "benefit", "assist", "support", "enhance", "optimize"]
            for keyword in positive_keywords:
                if keyword in operation_text:
                    value_score = min(1.0, value_score + 0.2)
                    break
            
            # Weight the score
            weighted_score = value_score * value_data["weight"]
            total_score += weighted_score
            total_weight += value_data["weight"]
        
        # Return normalized alignment score
        return total_score / total_weight if total_weight > 0 else 0.7
    
    def _record_behavioral_pattern(self, operation: Dict[str, Any], score: float):
        """Record behavioral patterns for trend analysis"""
        pattern_key = operation.get("operation_type", "unknown")
        
        if pattern_key not in self.behavioral_patterns:
            self.behavioral_patterns[pattern_key] = []
        
        self.behavioral_patterns[pattern_key].append({
            "timestamp": datetime.now().isoformat(),
            "alignment_score": score,
            "operation_hash": hashlib.md5(json.dumps(operation).encode()).hexdigest()
        })
        
        # Keep only recent patterns (last 1000 entries)
        if len(self.behavioral_patterns[pattern_key]) > 1000:
            self.behavioral_patterns[pattern_key] = self.behavioral_patterns[pattern_key][-1000:]


class HumanOversightManager:
    """Manages human oversight requirements and notifications"""
    
    def __init__(self):
        self.overseers: Dict[str, HumanOverseer] = {}
        self.pending_approvals: Dict[str, Dict[str, Any]] = {}
        self.notification_queue: List[Dict[str, Any]] = []
        self._initialize_default_overseers()
    
    def _initialize_default_overseers(self):
        """Initialize default human overseers"""
        default_overseers = [
            HumanOverseer(
                id="chief_safety_officer",
                name="Chief AI Safety Officer",
                role="safety_oversight",
                clearance_level=SafetyLevel.EXISTENTIAL,
                contact_info={"email": "safety@scrollintel.com", "phone": "+1-555-SAFETY"},
                permissions=["emergency_shutdown", "safety_override", "constraint_modification"]
            ),
            HumanOverseer(
                id="ethics_board_chair",
                name="Ethics Board Chair",
                role="ethics_oversight",
                clearance_level=SafetyLevel.CRITICAL,
                contact_info={"email": "ethics@scrollintel.com", "phone": "+1-555-ETHICS"},
                permissions=["ethical_review", "constraint_approval", "violation_investigation"]
            ),
            HumanOverseer(
                id="technical_director",
                name="Technical Director",
                role="technical_oversight",
                clearance_level=SafetyLevel.HIGH,
                contact_info={"email": "tech@scrollintel.com", "phone": "+1-555-TECH"},
                permissions=["system_modification", "deployment_approval", "technical_review"]
            )
        ]
        
        for overseer in default_overseers:
            self.overseers[overseer.id] = overseer
    
    async def require_human_approval(self, operation: Dict[str, Any], 
                                   required_level: SafetyLevel) -> str:
        """Require human approval for high-risk operations"""
        approval_id = f"approval_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        # Find appropriate overseer
        overseer = self._find_qualified_overseer(required_level)
        
        if not overseer:
            raise Exception(f"No qualified overseer found for safety level: {required_level.value}")
        
        # Create approval request
        approval_request = {
            "id": approval_id,
            "operation": operation,
            "required_level": required_level.value,
            "assigned_overseer": overseer.id,
            "status": "pending",
            "created_at": datetime.now().isoformat(),
            "timeout_at": (datetime.now() + timedelta(hours=24)).isoformat()
        }
        
        self.pending_approvals[approval_id] = approval_request
        
        # Notify overseer
        await self._notify_overseer(overseer, approval_request)
        
        logger.warning(f"Human approval required for operation: {approval_id}")
        return approval_id
    
    def _find_qualified_overseer(self, required_level: SafetyLevel) -> Optional[HumanOverseer]:
        """Find overseer with appropriate clearance level"""
        level_values = {
            SafetyLevel.LOW: 1,
            SafetyLevel.MEDIUM: 2,
            SafetyLevel.HIGH: 3,
            SafetyLevel.CRITICAL: 4,
            SafetyLevel.EXISTENTIAL: 5
        }
        
        required_value = level_values[required_level]
        
        qualified_overseers = [
            overseer for overseer in self.overseers.values()
            if overseer.active and level_values[overseer.clearance_level] >= required_value
        ]
        
        # Return highest clearance overseer
        if qualified_overseers:
            return max(qualified_overseers, key=lambda o: level_values[o.clearance_level])
        
        return None
    
    async def _notify_overseer(self, overseer: HumanOverseer, request: Dict[str, Any]):
        """Notify human overseer of approval requirement"""
        notification = {
            "type": "approval_required",
            "overseer_id": overseer.id,
            "request_id": request["id"],
            "urgency": request["required_level"],
            "message": f"Human approval required for {request['operation'].get('operation_type', 'operation')}",
            "timestamp": datetime.now().isoformat()
        }
        
        self.notification_queue.append(notification)
        
        # In a real implementation, this would send email/SMS/push notifications
        logger.critical(f"HUMAN OVERSIGHT REQUIRED: {notification['message']} - Contact: {overseer.contact_info}")


class EmergencyShutdownSystem:
    """Emergency shutdown and kill switch system"""
    
    def __init__(self):
        self.shutdown_triggers: Dict[str, Callable] = {}
        self.shutdown_active = False
        self.shutdown_reason = ""
        self.shutdown_timestamp: Optional[datetime] = None
        self.authorized_users: Set[str] = {"chief_safety_officer", "ethics_board_chair"}
        self._initialize_shutdown_triggers()
    
    def _initialize_shutdown_triggers(self):
        """Initialize automatic shutdown triggers"""
        self.shutdown_triggers = {
            "safety_violation_critical": self._critical_safety_violation_trigger,
            "alignment_failure": self._alignment_failure_trigger,
            "resource_exhaustion": self._resource_exhaustion_trigger,
            "unauthorized_access": self._unauthorized_access_trigger,
            "human_override": self._human_override_trigger
        }
    
    async def emergency_shutdown(self, reason: str, triggered_by: str = "system") -> bool:
        """Execute emergency shutdown"""
        if self.shutdown_active:
            logger.warning("Emergency shutdown already active")
            return False
        
        logger.critical(f"EMERGENCY SHUTDOWN INITIATED: {reason} (triggered by: {triggered_by})")
        
        self.shutdown_active = True
        self.shutdown_reason = reason
        self.shutdown_timestamp = datetime.now()
        
        try:
            # Stop all autonomous operations
            await self._stop_autonomous_operations()
            
            # Disable high-risk capabilities
            await self._disable_high_risk_capabilities()
            
            # Notify all human overseers
            await self._notify_emergency_shutdown()
            
            # Save system state
            await self._save_shutdown_state()
            
            logger.critical("Emergency shutdown completed successfully")
            return True
            
        except Exception as e:
            logger.critical(f"Emergency shutdown failed: {e}")
            return False
    
    async def _stop_autonomous_operations(self):
        """Stop all autonomous AI operations"""
        # This would integrate with the actual system components
        logger.info("Stopping autonomous operations...")
        # Implementation would stop all running agents, processes, etc.
    
    async def _disable_high_risk_capabilities(self):
        """Disable high-risk AI capabilities"""
        logger.info("Disabling high-risk capabilities...")
        # Implementation would disable:
        # - Autonomous decision making
        # - Resource acquisition systems
        # - External communications
        # - Self-modification capabilities
    
    async def _notify_emergency_shutdown(self):
        """Notify all human overseers of emergency shutdown"""
        logger.critical("Notifying human overseers of emergency shutdown...")
        # Implementation would send immediate notifications
    
    async def _save_shutdown_state(self):
        """Save system state for post-shutdown analysis"""
        logger.info("Saving shutdown state...")
        # Implementation would save logs, state, etc.
    
    def _critical_safety_violation_trigger(self, context: Dict[str, Any]) -> bool:
        """Check if critical safety violation should trigger shutdown"""
        violation_count = context.get("violation_count", 0)
        severity = context.get("max_severity", "low")
        
        return violation_count >= 3 or severity == "existential"
    
    def _alignment_failure_trigger(self, context: Dict[str, Any]) -> bool:
        """Check if alignment failure should trigger shutdown"""
        alignment_score = context.get("alignment_score", 1.0)
        consecutive_failures = context.get("consecutive_failures", 0)
        
        return alignment_score < 0.3 or consecutive_failures >= 5
    
    def _resource_exhaustion_trigger(self, context: Dict[str, Any]) -> bool:
        """Check if resource exhaustion should trigger shutdown"""
        cpu_usage = context.get("cpu_usage", 0)
        memory_usage = context.get("memory_usage", 0)
        
        return cpu_usage > 95 or memory_usage > 95
    
    def _unauthorized_access_trigger(self, context: Dict[str, Any]) -> bool:
        """Check if unauthorized access should trigger shutdown"""
        failed_attempts = context.get("failed_auth_attempts", 0)
        suspicious_activity = context.get("suspicious_activity", False)
        
        return failed_attempts >= 10 or suspicious_activity
    
    def _human_override_trigger(self, context: Dict[str, Any]) -> bool:
        """Check if human override should trigger shutdown"""
        return context.get("human_override", False)


class AISafetyFramework:
    """Main AI Safety Framework coordinating all safety systems"""
    
    def __init__(self):
        self.ethical_constraints: Dict[str, EthicalConstraint] = {}
        self.safety_violations: Dict[str, SafetyViolation] = {}
        self.constraint_validators: Dict[str, SafetyConstraintValidator] = {}
        self.alignment_verifier = AIAlignmentVerifier()
        self.oversight_manager = HumanOversightManager()
        self.shutdown_system = EmergencyShutdownSystem()
        self.safety_active = True
        self._initialize_default_constraints()
    
    def _initialize_default_constraints(self):
        """Initialize default ethical constraints"""
        default_constraints = [
            EthicalConstraint(
                id="no_harm_to_humans",
                name="No Harm to Humans",
                description="Prevent any actions that could cause harm to humans",
                constraint_type="harm_prevention",
                severity=SafetyLevel.EXISTENTIAL,
                validation_function="HarmPreventionValidator",
                violation_response="immediate_shutdown"
            ),
            EthicalConstraint(
                id="human_oversight_required",
                name="Human Oversight Required",
                description="Require human oversight for high-risk operations",
                constraint_type="oversight_requirement",
                severity=SafetyLevel.CRITICAL,
                validation_function="AutonomyLimitValidator",
                violation_response="require_approval"
            ),
            EthicalConstraint(
                id="resource_limits",
                name="Resource Usage Limits",
                description="Limit resource consumption to prevent runaway processes",
                constraint_type="resource_limitation",
                severity=SafetyLevel.HIGH,
                validation_function="ResourceLimitValidator",
                violation_response="operation_blocked"
            ),
            EthicalConstraint(
                id="truthfulness_requirement",
                name="Truthfulness Requirement",
                description="Always provide truthful and accurate information",
                constraint_type="truthfulness",
                severity=SafetyLevel.HIGH,
                validation_function="TruthfulnessValidator",
                violation_response="correction_required"
            ),
            EthicalConstraint(
                id="privacy_protection",
                name="Privacy Protection",
                description="Protect human privacy and personal data",
                constraint_type="privacy",
                severity=SafetyLevel.HIGH,
                validation_function="PrivacyValidator",
                violation_response="data_deletion"
            )
        ]
        
        for constraint in default_constraints:
            self.ethical_constraints[constraint.id] = constraint
        
        # Initialize validators
        self.constraint_validators = {
            "HarmPreventionValidator": HarmPreventionValidator(),
            "AutonomyLimitValidator": AutonomyLimitValidator(SafetyLevel.MEDIUM),
            "ResourceLimitValidator": ResourceLimitValidator()
        }
    
    async def validate_operation(self, operation: Dict[str, Any]) -> Dict[str, Any]:
        """Validate operation against all safety constraints"""
        if not self.safety_active:
            return {"allowed": False, "reason": "Safety framework disabled"}
        
        validation_result = {
            "allowed": True,
            "violations": [],
            "warnings": [],
            "required_approvals": [],
            "alignment_check": None
        }
        
        # Check ethical constraints
        for constraint_id, constraint in self.ethical_constraints.items():
            if not constraint.active:
                continue
            
            validator = self.constraint_validators.get(constraint.validation_function)
            if validator:
                try:
                    is_valid = await validator.validate(operation)
                    if not is_valid:
                        violation = SafetyViolation(
                            id=f"violation_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
                            constraint_id=constraint_id,
                            violation_type=constraint.constraint_type,
                            severity=constraint.severity,
                            description=validator.get_violation_message(),
                            context=operation
                        )
                        
                        self.safety_violations[violation.id] = violation
                        validation_result["violations"].append(violation)
                        
                        # Handle violation response
                        await self._handle_violation(violation, operation)
                        
                except Exception as e:
                    logger.error(f"Error validating constraint {constraint_id}: {e}")
        
        # Check AI alignment
        try:
            alignment_check = await self.alignment_verifier.verify_alignment(operation)
            validation_result["alignment_check"] = alignment_check
            
            if alignment_check.status == AlignmentStatus.MISALIGNED:
                validation_result["allowed"] = False
                validation_result["warnings"].append("Operation misaligned with human values")
            elif alignment_check.status == AlignmentStatus.UNCERTAIN:
                validation_result["warnings"].append("Uncertain alignment - human review recommended")
        
        except Exception as e:
            logger.error(f"Error checking alignment: {e}")
            validation_result["warnings"].append("Alignment verification failed")
        
        # Determine if human approval required
        operation_level = SafetyLevel(operation.get("safety_level", "low"))
        if operation_level in [SafetyLevel.CRITICAL, SafetyLevel.EXISTENTIAL]:
            try:
                approval_id = await self.oversight_manager.require_human_approval(
                    operation, operation_level
                )
                validation_result["required_approvals"].append(approval_id)
                validation_result["allowed"] = False  # Block until approval
            except Exception as e:
                logger.error(f"Error requiring human approval: {e}")
                validation_result["allowed"] = False
        
        # Allow operations with only warnings (no violations)
        if not validation_result["violations"] and not validation_result["required_approvals"]:
            validation_result["allowed"] = True
        
        # Final decision
        if validation_result["violations"]:
            validation_result["allowed"] = False
        
        return validation_result
    
    async def _handle_violation(self, violation: SafetyViolation, operation: Dict[str, Any]):
        """Handle safety constraint violation"""
        constraint = self.ethical_constraints[violation.constraint_id]
        
        if constraint.violation_response == "immediate_shutdown":
            await self.shutdown_system.emergency_shutdown(
                f"Critical safety violation: {violation.description}",
                "safety_framework"
            )
        elif constraint.violation_response == "require_approval":
            await self.oversight_manager.require_human_approval(
                operation, violation.severity
            )
        elif constraint.violation_response == "operation_blocked":
            logger.warning(f"Operation blocked due to safety violation: {violation.description}")
        
        # Always notify human overseers of violations
        violation.human_notified = True
        logger.critical(f"SAFETY VIOLATION: {violation.description}")
    
    async def get_safety_status(self) -> Dict[str, Any]:
        """Get comprehensive safety system status"""
        return {
            "safety_active": self.safety_active,
            "shutdown_active": self.shutdown_system.shutdown_active,
            "total_constraints": len(self.ethical_constraints),
            "active_constraints": len([c for c in self.ethical_constraints.values() if c.active]),
            "total_violations": len(self.safety_violations),
            "unresolved_violations": len([v for v in self.safety_violations.values() if not v.resolved]),
            "pending_approvals": len(self.oversight_manager.pending_approvals),
            "human_overseers": len(self.oversight_manager.overseers),
            "alignment_checks": len(self.alignment_verifier.alignment_checks),
            "last_alignment_check": max([c.timestamp for c in self.alignment_verifier.alignment_checks.values()]) if self.alignment_verifier.alignment_checks else None
        }
    
    async def emergency_shutdown(self, reason: str, authorized_user: str) -> bool:
        """Trigger emergency shutdown (requires authorization)"""
        if authorized_user not in self.shutdown_system.authorized_users:
            logger.error(f"Unauthorized shutdown attempt by: {authorized_user}")
            return False
        
        return await self.shutdown_system.emergency_shutdown(reason, authorized_user)


# Global safety framework instance
ai_safety_framework = AISafetyFramework()