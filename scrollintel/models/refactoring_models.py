"""
Refactoring Models for Refactor Genius Agent

Data models for automatic legacy modernization with superhuman capabilities.
"""

from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from datetime import datetime
from enum import Enum
import uuid


class RefactoringType(Enum):
    MODERNIZATION = "modernization"
    PERFORMANCE_OPTIMIZATION = "performance_optimization"
    SECURITY_HARDENING = "security_hardening"
    TECHNICAL_DEBT_ELIMINATION = "technical_debt_elimination"
    ARCHITECTURE_IMPROVEMENT = "architecture_improvement"
    CODE_QUALITY_ENHANCEMENT = "code_quality_enhancement"
    DEPENDENCY_UPGRADE = "dependency_upgrade"
    PATTERN_MODERNIZATION = "pattern_modernization"


class ModernizationLevel(Enum):
    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    CUTTING_EDGE = "cutting_edge"
    SUPERHUMAN = "superhuman"


class CompatibilityLevel(Enum):
    BREAKING = "breaking"
    PARTIAL = "partial"
    BACKWARD_COMPATIBLE = "backward_compatible"
    PERFECT = "perfect"


class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    ZERO = "zero"


@dataclass
class RefactoringRequest:
    """Request for legacy code refactoring and modernization"""
    id: str
    name: Optional[str]
    description: Optional[str]
    legacy_code: str
    language: str
    refactoring_types: List[RefactoringType]
    target_modernization_level: ModernizationLevel
    compatibility_requirements: List[str]
    performance_targets: Optional[Dict[str, Any]] = None
    security_requirements: Optional[Dict[str, Any]] = None
    constraints: Optional[Dict[str, Any]] = None
    existing_tests: Optional[str] = None
    documentation: Optional[str] = None
    dependencies: Optional[List[str]] = None
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class TechnicalDebtAnalysis:
    """Analysis of technical debt in legacy code"""
    id: str
    code_id: str
    debt_score: float
    debt_categories: Dict[str, float]
    code_smells: List[Dict[str, Any]]
    complexity_metrics: Dict[str, Any]
    maintainability_index: float
    duplication_percentage: float
    test_coverage: float
    documentation_coverage: float
    security_vulnerabilities: List[str]
    performance_issues: List[str]
    refactoring_recommendations: List[str]
    estimated_effort: str
    priority_level: str
    analyzed_at: datetime


@dataclass
class RefactoringStrategy:
    """Strategy for refactoring legacy code"""
    id: str
    name: str
    type: RefactoringType
    description: str
    implementation_steps: List[str]
    risk_level: RiskLevel
    effort_estimate: str
    impact_assessment: Dict[str, Any]
    compatibility_requirements: List[str]
    rollback_plan: Dict[str, Any]
    prerequisites: Optional[List[str]] = None
    success_criteria: Optional[List[str]] = None
    testing_strategy: Optional[str] = None
    
    def __post_init__(self):
        if self.prerequisites is None:
            self.prerequisites = []
        if self.success_criteria is None:
            self.success_criteria = []


@dataclass
class ModernizationPlan:
    """Comprehensive plan for legacy code modernization"""
    id: str
    target_modernization_level: ModernizationLevel
    strategies: List[RefactoringStrategy]
    execution_order: List[RefactoringStrategy]
    estimated_improvement: float
    risk_assessment: Dict[str, Any]
    compatibility_requirements: List[str]
    rollback_strategy: Dict[str, Any]
    timeline: Optional[Dict[str, Any]] = None
    resource_requirements: Optional[Dict[str, Any]] = None
    success_metrics: Optional[Dict[str, Any]] = None
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class CompatibilityReport:
    """Report on compatibility preservation during refactoring"""
    id: str
    api_compatibility_score: float
    data_compatibility_score: float
    integration_compatibility_score: float
    compatibility_adapters: List[Dict[str, Any]]
    breaking_changes: List[Dict[str, Any]]
    migration_requirements: List[str]
    rollback_feasibility: float
    compatibility_level: Optional[CompatibilityLevel] = None
    validation_results: Optional[Dict[str, Any]] = None
    validated_at: datetime = None
    
    def __post_init__(self):
        if self.validated_at is None:
            self.validated_at = datetime.now()
        if self.compatibility_level is None:
            # Determine compatibility level based on scores
            avg_score = (
                self.api_compatibility_score + 
                self.data_compatibility_score + 
                self.integration_compatibility_score
            ) / 3
            if avg_score >= 1.0:
                self.compatibility_level = CompatibilityLevel.PERFECT
            elif avg_score >= 0.9:
                self.compatibility_level = CompatibilityLevel.BACKWARD_COMPATIBLE
            elif avg_score >= 0.7:
                self.compatibility_level = CompatibilityLevel.PARTIAL
            else:
                self.compatibility_level = CompatibilityLevel.BREAKING


@dataclass
class RefactoredCode:
    """Result of refactoring operation"""
    id: str
    original_code: str
    refactored_code: str
    language: str
    refactoring_type: RefactoringType
    strategies_applied: List[RefactoringStrategy]
    improvements: Dict[str, Any]
    quality_metrics: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    security_enhancements: List[str]
    compatibility_report: CompatibilityReport
    test_results: Optional[Dict[str, Any]] = None
    documentation: Optional[str] = None
    migration_guide: Optional[str] = None
    refactored_at: datetime = None
    
    def __post_init__(self):
        if self.refactored_at is None:
            self.refactored_at = datetime.now()


@dataclass
class MigrationPlan:
    """Plan for migrating legacy systems"""
    id: str
    source_system: Dict[str, Any]
    target_architecture: str
    migration_strategy: str
    phases: List[Dict[str, Any]]
    risk_mitigation: Dict[str, Any]
    rollback_procedures: List[str]
    testing_strategy: Dict[str, Any]
    timeline: Dict[str, Any]
    resource_requirements: Dict[str, Any]
    success_criteria: List[str]
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class LegacySystemAnalysis:
    """Analysis of legacy system for modernization"""
    id: str
    system_name: str
    technology_stack: List[str]
    architecture_type: str
    complexity_assessment: Dict[str, Any]
    technical_debt_analysis: TechnicalDebtAnalysis
    modernization_opportunities: List[str]
    migration_challenges: List[str]
    business_impact_assessment: Dict[str, Any]
    recommended_approach: str
    estimated_effort: str
    analyzed_at: datetime


@dataclass
class RefactoringMetrics:
    """Metrics for measuring refactoring success"""
    id: str
    refactoring_id: str
    before_metrics: Dict[str, Any]
    after_metrics: Dict[str, Any]
    improvement_percentages: Dict[str, float]
    quality_improvements: Dict[str, float]
    performance_improvements: Dict[str, float]
    maintainability_improvements: Dict[str, float]
    security_improvements: Dict[str, float]
    technical_debt_reduction: float
    roi_analysis: Dict[str, Any]
    measured_at: datetime


@dataclass
class AutomationReport:
    """Report on automation level achieved during refactoring"""
    id: str
    refactoring_id: str
    automation_percentage: float
    manual_interventions: List[str]
    automated_tasks: List[str]
    automation_failures: List[str]
    human_review_points: List[str]
    quality_assurance_results: Dict[str, Any]
    automation_effectiveness: float
    generated_at: datetime


@dataclass
class RollbackPlan:
    """Plan for rolling back refactoring changes"""
    id: str
    refactoring_id: str
    rollback_strategy: str
    rollback_steps: List[str]
    data_restoration_plan: Dict[str, Any]
    system_restoration_plan: Dict[str, Any]
    validation_procedures: List[str]
    estimated_rollback_time: str
    risk_assessment: Dict[str, Any]
    success_criteria: List[str]
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


# Utility functions for refactoring models

def create_refactoring_request(
    legacy_code: str,
    language: str,
    refactoring_types: List[RefactoringType],
    target_level: ModernizationLevel = ModernizationLevel.ADVANCED,
    name: Optional[str] = None,
    description: Optional[str] = None
) -> RefactoringRequest:
    """Create a new refactoring request"""
    return RefactoringRequest(
        id=str(uuid.uuid4()),
        name=name,
        description=description,
        legacy_code=legacy_code,
        language=language,
        refactoring_types=refactoring_types,
        target_modernization_level=target_level,
        compatibility_requirements=["backward_compatible"]
    )


def create_technical_debt_analysis(
    code_id: str,
    debt_score: float,
    debt_categories: Dict[str, float],
    code_smells: List[Dict[str, Any]],
    maintainability_index: float
) -> TechnicalDebtAnalysis:
    """Create technical debt analysis"""
    return TechnicalDebtAnalysis(
        id=str(uuid.uuid4()),
        code_id=code_id,
        debt_score=debt_score,
        debt_categories=debt_categories,
        code_smells=code_smells,
        complexity_metrics={},
        maintainability_index=maintainability_index,
        duplication_percentage=0.0,
        test_coverage=0.0,
        documentation_coverage=0.0,
        security_vulnerabilities=[],
        performance_issues=[],
        refactoring_recommendations=[],
        estimated_effort="medium",
        priority_level="high",
        analyzed_at=datetime.now()
    )


def create_refactoring_strategy(
    name: str,
    strategy_type: RefactoringType,
    description: str,
    implementation_steps: List[str],
    risk_level: RiskLevel = RiskLevel.LOW
) -> RefactoringStrategy:
    """Create refactoring strategy"""
    return RefactoringStrategy(
        id=str(uuid.uuid4()),
        name=name,
        type=strategy_type,
        description=description,
        implementation_steps=implementation_steps,
        risk_level=risk_level,
        effort_estimate="medium",
        impact_assessment={},
        compatibility_requirements=[],
        rollback_plan={}
    )


def create_modernization_plan(
    target_level: ModernizationLevel,
    strategies: List[RefactoringStrategy],
    estimated_improvement: float
) -> ModernizationPlan:
    """Create modernization plan"""
    return ModernizationPlan(
        id=str(uuid.uuid4()),
        target_modernization_level=target_level,
        strategies=strategies,
        execution_order=strategies,  # Default to same order
        estimated_improvement=estimated_improvement,
        risk_assessment={},
        compatibility_requirements=[],
        rollback_strategy={}
    )


def create_compatibility_report(
    api_score: float = 1.0,
    data_score: float = 1.0,
    integration_score: float = 1.0
) -> CompatibilityReport:
    """Create compatibility report"""
    return CompatibilityReport(
        id=str(uuid.uuid4()),
        api_compatibility_score=api_score,
        data_compatibility_score=data_score,
        integration_compatibility_score=integration_score,
        compatibility_adapters=[],
        breaking_changes=[],
        migration_requirements=[],
        rollback_feasibility=1.0
    )


def create_refactored_code(
    original_code: str,
    refactored_code: str,
    language: str,
    refactoring_type: RefactoringType,
    strategies: List[RefactoringStrategy],
    compatibility_report: CompatibilityReport
) -> RefactoredCode:
    """Create refactored code result"""
    return RefactoredCode(
        id=str(uuid.uuid4()),
        original_code=original_code,
        refactored_code=refactored_code,
        language=language,
        refactoring_type=refactoring_type,
        strategies_applied=strategies,
        improvements={},
        quality_metrics={},
        performance_metrics={},
        security_enhancements=[],
        compatibility_report=compatibility_report
    )