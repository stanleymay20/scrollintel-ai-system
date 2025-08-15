"""
Requirements validation and completeness checking for automated code generation.
Validates requirements quality and completeness for successful code generation.
"""
import re
from typing import List, Dict, Any, Optional, Tuple, Set
from enum import Enum

from scrollintel.models.code_generation_models import (
    Requirements, ParsedRequirement, Entity, Relationship,
    RequirementType, EntityType, ConfidenceLevel, Intent
)


class ValidationSeverity(str, Enum):
    """Severity levels for validation issues."""
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"


class ValidationIssue:
    """Represents a validation issue."""
    
    def __init__(self, severity: ValidationSeverity, message: str, 
                 requirement_id: Optional[str] = None, entity_id: Optional[str] = None):
        self.severity = severity
        self.message = message
        self.requirement_id = requirement_id
        self.entity_id = entity_id


class RequirementsValidator:
    """Validates requirements for completeness and quality."""
    
    def __init__(self):
        """Initialize the requirements validator."""
        self.min_requirements_count = 1
        self.min_entities_count = 2
        self.min_completeness_score = 0.6
        
        # Quality patterns
        self.quality_patterns = {
            'too_vague': [
                r'\b(?:somehow|maybe|perhaps|possibly|probably)\b',
                r'\b(?:good|bad|nice|better|worse)\b(?!\s+(?:than|practice|performance))',
                r'\b(?:stuff|things|items|elements)\b'
            ],
            'too_technical': [
                r'\b(?:algorithm|implementation|code|programming|technical)\b',
                r'\b(?:class|method|function|variable|parameter)\b',
                r'\b(?:database|table|column|index|query)\b'
            ],
            'missing_context': [
                r'^(?:it|this|that|they|we)\s+(?:should|must|will|can)\b',
                r'\b(?:the\s+system|the\s+application|the\s+platform)\b(?!\s+\w+)',
                r'\b(?:users?|customers?)\b(?!\s+(?:can|should|will|must|who|that))'
            ]
        }
    
    def validate_requirements(self, requirements: Requirements) -> List[ValidationIssue]:
        """
        Validate requirements for completeness and quality.
        
        Args:
            requirements: The requirements object to validate
            
        Returns:
            List of validation issues
        """
        issues = []
        
        # Validate overall structure
        issues.extend(self._validate_structure(requirements))
        
        # Validate individual requirements
        for req in requirements.parsed_requirements:
            issues.extend(self._validate_requirement(req))
        
        # Validate entities
        issues.extend(self._validate_entities(requirements.entities))
        
        # Validate relationships
        issues.extend(self._validate_relationships(requirements.relationships, requirements.entities))
        
        # Validate completeness
        issues.extend(self._validate_completeness(requirements))
        
        # Validate consistency
        issues.extend(self._validate_consistency(requirements))
        
        return issues
    
    def calculate_quality_score(self, requirements: Requirements) -> float:
        """
        Calculate overall quality score for requirements.
        
        Args:
            requirements: The requirements object to score
            
        Returns:
            Quality score from 0.0 to 1.0
        """
        issues = self.validate_requirements(requirements)
        
        # Count issues by severity
        critical_count = sum(1 for issue in issues if issue.severity == ValidationSeverity.CRITICAL)
        warning_count = sum(1 for issue in issues if issue.severity == ValidationSeverity.WARNING)
        info_count = sum(1 for issue in issues if issue.severity == ValidationSeverity.INFO)
        
        # Calculate penalty based on issues
        total_requirements = len(requirements.parsed_requirements)
        if total_requirements == 0:
            return 0.0
        
        # Base score from completeness
        base_score = requirements.completeness_score
        
        # Apply penalties
        critical_penalty = min(0.5, critical_count * 0.1)
        warning_penalty = min(0.3, warning_count * 0.05)
        info_penalty = min(0.1, info_count * 0.02)
        
        quality_score = base_score - critical_penalty - warning_penalty - info_penalty
        
        return max(0.0, min(1.0, quality_score))
    
    def get_improvement_suggestions(self, requirements: Requirements) -> List[str]:
        """
        Get suggestions for improving requirements quality.
        
        Args:
            requirements: The requirements object to analyze
            
        Returns:
            List of improvement suggestions
        """
        suggestions = []
        issues = self.validate_requirements(requirements)
        
        # Group issues by type
        critical_issues = [i for i in issues if i.severity == ValidationSeverity.CRITICAL]
        warning_issues = [i for i in issues if i.severity == ValidationSeverity.WARNING]
        
        if critical_issues:
            suggestions.append("Address critical issues first - these prevent successful code generation")
        
        if len(requirements.parsed_requirements) < 3:
            suggestions.append("Consider breaking down complex requirements into smaller, more specific ones")
        
        if requirements.completeness_score < 0.7:
            suggestions.append("Add more details about user roles, data entities, and system components")
        
        # Check for missing requirement types
        req_types = {req.requirement_type for req in requirements.parsed_requirements}
        if RequirementType.FUNCTIONAL not in req_types:
            suggestions.append("Add functional requirements describing what the system should do")
        
        if RequirementType.NON_FUNCTIONAL not in req_types:
            suggestions.append("Consider adding non-functional requirements (performance, security, etc.)")
        
        # Check entity coverage
        entity_types = {entity.type for entity in requirements.entities}
        if EntityType.USER_ROLE not in entity_types:
            suggestions.append("Specify who will use the system (user roles)")
        
        if EntityType.DATA_ENTITY not in entity_types:
            suggestions.append("Describe what data the system will manage")
        
        # Check for acceptance criteria
        reqs_without_criteria = [req for req in requirements.parsed_requirements if not req.acceptance_criteria]
        if len(reqs_without_criteria) > len(requirements.parsed_requirements) * 0.5:
            suggestions.append("Add acceptance criteria to help validate when requirements are met")
        
        return suggestions
    
    def _validate_structure(self, requirements: Requirements) -> List[ValidationIssue]:
        """Validate overall requirements structure."""
        issues = []
        
        if not requirements.parsed_requirements:
            issues.append(ValidationIssue(
                ValidationSeverity.CRITICAL,
                "No requirements found - cannot generate code without requirements"
            ))
        
        if len(requirements.parsed_requirements) < self.min_requirements_count:
            issues.append(ValidationIssue(
                ValidationSeverity.WARNING,
                f"Only {len(requirements.parsed_requirements)} requirement(s) found - consider adding more detail"
            ))
        
        if len(requirements.entities) < self.min_entities_count:
            issues.append(ValidationIssue(
                ValidationSeverity.WARNING,
                f"Only {len(requirements.entities)} entities found - may need more domain modeling"
            ))
        
        if requirements.completeness_score < self.min_completeness_score:
            issues.append(ValidationIssue(
                ValidationSeverity.WARNING,
                f"Completeness score is {requirements.completeness_score:.1%} - consider adding more details"
            ))
        
        return issues
    
    def _validate_requirement(self, requirement: ParsedRequirement) -> List[ValidationIssue]:
        """Validate an individual requirement."""
        issues = []
        
        # Check for empty or very short requirements
        if not requirement.structured_text.strip():
            issues.append(ValidationIssue(
                ValidationSeverity.CRITICAL,
                "Requirement has empty text",
                requirement_id=requirement.id
            ))
        elif len(requirement.structured_text.strip()) < 10:
            issues.append(ValidationIssue(
                ValidationSeverity.WARNING,
                "Requirement is very short and may lack detail",
                requirement_id=requirement.id
            ))
        
        # Check confidence level
        if requirement.confidence == ConfidenceLevel.LOW:
            issues.append(ValidationIssue(
                ValidationSeverity.WARNING,
                "Low confidence in requirement parsing - may need clarification",
                requirement_id=requirement.id
            ))
        
        # Check for acceptance criteria
        if not requirement.acceptance_criteria:
            issues.append(ValidationIssue(
                ValidationSeverity.INFO,
                "Requirement lacks acceptance criteria",
                requirement_id=requirement.id
            ))
        
        # Check for quality issues
        text_lower = requirement.structured_text.lower()
        
        for pattern_type, patterns in self.quality_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    if pattern_type == 'too_vague':
                        issues.append(ValidationIssue(
                            ValidationSeverity.WARNING,
                            f"Requirement contains vague language: '{requirement.structured_text[:50]}...'",
                            requirement_id=requirement.id
                        ))
                    elif pattern_type == 'too_technical':
                        issues.append(ValidationIssue(
                            ValidationSeverity.INFO,
                            f"Requirement may be too technical/implementation-focused",
                            requirement_id=requirement.id
                        ))
                    elif pattern_type == 'missing_context':
                        issues.append(ValidationIssue(
                            ValidationSeverity.WARNING,
                            f"Requirement lacks context or clear subject",
                            requirement_id=requirement.id
                        ))
                    break  # Only report one issue per pattern type per requirement
        
        # Check for measurable criteria
        if requirement.requirement_type == RequirementType.PERFORMANCE:
            if not re.search(r'\b\d+\s*(?:ms|seconds?|minutes?|users?|requests?|%)\b', requirement.structured_text):
                issues.append(ValidationIssue(
                    ValidationSeverity.WARNING,
                    "Performance requirement lacks measurable criteria",
                    requirement_id=requirement.id
                ))
        
        return issues
    
    def _validate_entities(self, entities: List[Entity]) -> List[ValidationIssue]:
        """Validate extracted entities."""
        issues = []
        
        # Check for essential entity types
        entity_types = {entity.type for entity in entities}
        
        if EntityType.USER_ROLE not in entity_types:
            issues.append(ValidationIssue(
                ValidationSeverity.WARNING,
                "No user roles identified - who will use the system?"
            ))
        
        if EntityType.DATA_ENTITY not in entity_types:
            issues.append(ValidationIssue(
                ValidationSeverity.INFO,
                "No data entities identified - what data will the system manage?"
            ))
        
        # Check for low confidence entities
        low_confidence_entities = [e for e in entities if e.confidence < 0.4]
        if low_confidence_entities:
            issues.append(ValidationIssue(
                ValidationSeverity.INFO,
                f"{len(low_confidence_entities)} entities have low confidence scores"
            ))
        
        # Check for duplicate entity names
        entity_names = [e.name.lower() for e in entities]
        duplicates = set([name for name in entity_names if entity_names.count(name) > 1])
        if duplicates:
            issues.append(ValidationIssue(
                ValidationSeverity.WARNING,
                f"Duplicate entity names found: {', '.join(duplicates)}"
            ))
        
        return issues
    
    def _validate_relationships(self, relationships: List[Relationship], entities: List[Entity]) -> List[ValidationIssue]:
        """Validate entity relationships."""
        issues = []
        
        if not relationships and len(entities) > 2:
            issues.append(ValidationIssue(
                ValidationSeverity.INFO,
                "No relationships identified between entities - consider how they interact"
            ))
        
        # Validate relationship references
        entity_ids = {e.id for e in entities}
        for relationship in relationships:
            if relationship.source_entity_id not in entity_ids:
                issues.append(ValidationIssue(
                    ValidationSeverity.CRITICAL,
                    f"Relationship references non-existent source entity: {relationship.source_entity_id}"
                ))
            
            if relationship.target_entity_id not in entity_ids:
                issues.append(ValidationIssue(
                    ValidationSeverity.CRITICAL,
                    f"Relationship references non-existent target entity: {relationship.target_entity_id}"
                ))
        
        return issues
    
    def _validate_completeness(self, requirements: Requirements) -> List[ValidationIssue]:
        """Validate requirements completeness."""
        issues = []
        
        # Check requirement type coverage
        req_types = {req.requirement_type for req in requirements.parsed_requirements}
        
        if RequirementType.FUNCTIONAL not in req_types:
            issues.append(ValidationIssue(
                ValidationSeverity.WARNING,
                "No functional requirements found - what should the system do?"
            ))
        
        # Check intent coverage
        intents = {req.intent for req in requirements.parsed_requirements}
        
        if Intent.CREATE_APPLICATION in intents:
            # For new applications, check for essential components
            if Intent.DESIGN_DATABASE not in intents and EntityType.DATA_ENTITY not in {e.type for e in requirements.entities}:
                issues.append(ValidationIssue(
                    ValidationSeverity.INFO,
                    "Application creation without clear data requirements - consider data needs"
                ))
            
            if Intent.BUILD_UI not in intents and RequirementType.UI_UX not in req_types:
                issues.append(ValidationIssue(
                    ValidationSeverity.INFO,
                    "Application creation without UI requirements - consider user interface needs"
                ))
        
        return issues
    
    def _validate_consistency(self, requirements: Requirements) -> List[ValidationIssue]:
        """Validate internal consistency of requirements."""
        issues = []
        
        # Check for conflicting requirements
        priorities = [req.priority for req in requirements.parsed_requirements]
        if priorities and max(priorities) - min(priorities) > 3:
            issues.append(ValidationIssue(
                ValidationSeverity.INFO,
                "Wide range of requirement priorities - consider reviewing priority assignments"
            ))
        
        # Check for complexity vs. detail mismatch
        for req in requirements.parsed_requirements:
            if req.complexity >= 4 and len(req.acceptance_criteria) < 2:
                issues.append(ValidationIssue(
                    ValidationSeverity.WARNING,
                    f"High complexity requirement lacks detailed acceptance criteria",
                    requirement_id=req.id
                ))
        
        return issues
    
    def is_ready_for_code_generation(self, requirements: Requirements) -> Tuple[bool, List[str]]:
        """
        Check if requirements are ready for code generation.
        
        Args:
            requirements: The requirements to check
            
        Returns:
            Tuple of (is_ready, list_of_blocking_issues)
        """
        issues = self.validate_requirements(requirements)
        
        # Find critical issues that block code generation
        blocking_issues = []
        for issue in issues:
            if issue.severity == ValidationSeverity.CRITICAL:
                blocking_issues.append(issue.message)
        
        # Additional readiness checks
        if not requirements.parsed_requirements:
            blocking_issues.append("No requirements available")
        
        if requirements.completeness_score < 0.4:
            blocking_issues.append("Requirements completeness too low for code generation")
        
        # Check for minimum viable requirements
        entity_types = {e.type for e in requirements.entities}
        if EntityType.USER_ROLE not in entity_types and EntityType.DATA_ENTITY not in entity_types:
            blocking_issues.append("Need at least user roles or data entities to generate meaningful code")
        
        is_ready = len(blocking_issues) == 0
        return is_ready, blocking_issues