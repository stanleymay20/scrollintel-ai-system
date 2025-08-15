"""
Code Generation Models for Code Master Agent

Data models for perfect code generation with superhuman capabilities.
"""

from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from datetime import datetime
from enum import Enum
import uuid


class ProgrammingLanguage(Enum):
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    JAVA = "java"
    CSHARP = "csharp"
    CPP = "cpp"
    GO = "go"
    RUST = "rust"
    KOTLIN = "kotlin"
    SWIFT = "swift"
    PHP = "php"
    RUBY = "ruby"
    SCALA = "scala"
    CLOJURE = "clojure"
    HASKELL = "haskell"


class CodeType(Enum):
    FUNCTION = "function"
    CLASS = "class"
    MODULE = "module"
    SCRIPT = "script"
    LIBRARY = "library"
    APPLICATION = "application"
    MICROSERVICE = "microservice"
    API = "api"


class OptimizationType(Enum):
    PERFORMANCE = "performance"
    MEMORY = "memory"
    SECURITY = "security"
    READABILITY = "readability"
    MAINTAINABILITY = "maintainability"
    SCALABILITY = "scalability"


class QualityLevel(Enum):
    BASIC = "basic"
    GOOD = "good"
    EXCELLENT = "excellent"
    SUPERHUMAN = "superhuman"


@dataclass
class CodeGenerationRequest:
    """Request for code generation"""
    id: str
    name: str
    description: str
    language: ProgrammingLanguage
    code_type: CodeType
    requirements: List[str]
    specifications: Dict[str, Any]
    performance_requirements: Optional[Dict[str, Any]] = None
    security_requirements: Optional[Dict[str, Any]] = None
    quality_requirements: Optional[Dict[str, Any]] = None
    optimization_preferences: Optional[List[OptimizationType]] = None
    existing_code: Optional[str] = None
    dependencies: Optional[List[str]] = None
    constraints: Optional[Dict[str, Any]] = None
    target_platform: Optional[str] = None
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.optimization_preferences is None:
            self.optimization_preferences = [OptimizationType.PERFORMANCE]


@dataclass
class CodeOptimization:
    """Code optimization applied to generated code"""
    id: str
    name: str
    type: OptimizationType
    description: str
    implementation: str
    performance_impact: float
    memory_impact: float
    readability_impact: float
    security_impact: float
    applied_at: datetime
    metrics: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = {}


@dataclass
class CodeQualityMetrics:
    """Code quality metrics for generated code"""
    id: str
    complexity_score: float
    maintainability_score: float
    readability_score: float
    test_coverage: float
    documentation_coverage: float
    security_score: float
    performance_score: float
    bug_probability: float
    code_smells: int
    technical_debt: float
    cyclomatic_complexity: Optional[int] = None
    lines_of_code: Optional[int] = None
    duplication_percentage: Optional[float] = None
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class PerformanceImprovement:
    """Performance improvement metrics"""
    id: str
    optimization_type: str
    baseline_performance: Dict[str, Any]
    optimized_performance: Dict[str, Any]
    improvement_percentage: float
    execution_time_improvement: float
    memory_usage_improvement: float
    throughput_improvement: float
    latency_improvement: float
    benchmarks: Dict[str, Any]
    measured_at: datetime
    
    def __post_init__(self):
        if not hasattr(self, 'measured_at') or self.measured_at is None:
            self.measured_at = datetime.now()


@dataclass
class GeneratedCode:
    """Generated code with metadata"""
    id: str
    language: ProgrammingLanguage
    code_type: CodeType
    source_code: str
    documentation: str
    tests: str
    dependencies: List[str]
    performance_metrics: Dict[str, Any]
    quality_metrics: CodeQualityMetrics
    optimizations: List[CodeOptimization]
    security_features: List[str]
    generated_at: datetime
    generator_version: str
    
    def __post_init__(self):
        if not hasattr(self, 'generated_at') or self.generated_at is None:
            self.generated_at = datetime.now()


@dataclass
class CodeValidationResult:
    """Result of code validation"""
    id: str
    code_id: str
    validation_type: str
    status: str
    score: float
    issues: List[Dict[str, Any]]
    warnings: List[str]
    suggestions: List[str]
    performance_analysis: Dict[str, Any]
    security_analysis: Dict[str, Any]
    quality_analysis: Dict[str, Any]
    validated_at: datetime
    validator: str


@dataclass
class CodeTemplate:
    """Code template for generation"""
    id: str
    name: str
    description: str
    language: ProgrammingLanguage
    code_type: CodeType
    template_code: str
    parameters: List[Dict[str, Any]]
    optimizations: List[str]
    best_practices: List[str]
    security_patterns: List[str]
    performance_patterns: List[str]
    created_at: datetime
    updated_at: Optional[datetime] = None


@dataclass
class CodeReview:
    """Code review results"""
    id: str
    code_id: str
    reviewer: str
    review_type: str
    overall_score: float
    code_quality_score: float
    performance_score: float
    security_score: float
    maintainability_score: float
    comments: List[Dict[str, Any]]
    suggestions: List[str]
    approved: bool
    reviewed_at: datetime


@dataclass
class CodeBenchmark:
    """Code performance benchmark"""
    id: str
    code_id: str
    benchmark_type: str
    execution_time: float
    memory_usage: float
    cpu_usage: float
    throughput: float
    latency: float
    scalability_metrics: Dict[str, Any]
    comparison_baseline: Dict[str, Any]
    environment: Dict[str, Any]
    benchmarked_at: datetime


@dataclass
class CodeEvolution:
    """Code evolution tracking"""
    id: str
    original_code_id: str
    evolved_code_id: str
    evolution_type: str
    changes: List[Dict[str, Any]]
    performance_impact: Dict[str, Any]
    quality_impact: Dict[str, Any]
    security_impact: Dict[str, Any]
    evolution_reason: str
    evolved_at: datetime
    evolved_by: str


@dataclass
class SuperhumanCodeFeatures:
    """Superhuman features of generated code"""
    bug_free_guarantee: bool
    performance_optimization_level: float
    security_hardening_level: float
    maintainability_score: float
    scalability_factor: str
    documentation_completeness: float
    test_coverage: float
    optimization_techniques: List[str]
    security_features: List[str]
    performance_features: List[str]


# Utility functions for code generation models

def create_code_generation_request(
    name: str,
    description: str,
    language: ProgrammingLanguage,
    code_type: CodeType,
    requirements: List[str],
    specifications: Dict[str, Any]
) -> CodeGenerationRequest:
    """Create a new code generation request"""
    return CodeGenerationRequest(
        id=str(uuid.uuid4()),
        name=name,
        description=description,
        language=language,
        code_type=code_type,
        requirements=requirements,
        specifications=specifications
    )


def create_code_optimization(
    name: str,
    optimization_type: OptimizationType,
    description: str,
    implementation: str,
    performance_impact: float,
    memory_impact: float = 0.0,
    readability_impact: float = 0.0,
    security_impact: float = 0.0
) -> CodeOptimization:
    """Create a new code optimization"""
    return CodeOptimization(
        id=str(uuid.uuid4()),
        name=name,
        type=optimization_type,
        description=description,
        implementation=implementation,
        performance_impact=performance_impact,
        memory_impact=memory_impact,
        readability_impact=readability_impact,
        security_impact=security_impact,
        applied_at=datetime.now()
    )


def create_code_quality_metrics(
    complexity_score: float,
    maintainability_score: float,
    readability_score: float,
    test_coverage: float,
    documentation_coverage: float,
    security_score: float,
    performance_score: float,
    bug_probability: float,
    code_smells: int = 0,
    technical_debt: float = 0.0
) -> CodeQualityMetrics:
    """Create code quality metrics"""
    return CodeQualityMetrics(
        id=str(uuid.uuid4()),
        complexity_score=complexity_score,
        maintainability_score=maintainability_score,
        readability_score=readability_score,
        test_coverage=test_coverage,
        documentation_coverage=documentation_coverage,
        security_score=security_score,
        performance_score=performance_score,
        bug_probability=bug_probability,
        code_smells=code_smells,
        technical_debt=technical_debt
    )


def create_performance_improvement(
    optimization_type: str,
    baseline_performance: Dict[str, Any],
    optimized_performance: Dict[str, Any],
    improvement_percentage: float
) -> PerformanceImprovement:
    """Create performance improvement metrics"""
    return PerformanceImprovement(
        id=str(uuid.uuid4()),
        optimization_type=optimization_type,
        baseline_performance=baseline_performance,
        optimized_performance=optimized_performance,
        improvement_percentage=improvement_percentage,
        execution_time_improvement=0.0,
        memory_usage_improvement=0.0,
        throughput_improvement=0.0,
        latency_improvement=0.0,
        benchmarks={},
        measured_at=datetime.now()
    )


def create_generated_code(
    language: ProgrammingLanguage,
    code_type: CodeType,
    source_code: str,
    documentation: str,
    tests: str,
    quality_metrics: CodeQualityMetrics,
    generator_version: str = "1.0.0"
) -> GeneratedCode:
    """Create generated code object"""
    return GeneratedCode(
        id=str(uuid.uuid4()),
        language=language,
        code_type=code_type,
        source_code=source_code,
        documentation=documentation,
        tests=tests,
        dependencies=[],
        performance_metrics={},
        quality_metrics=quality_metrics,
        optimizations=[],
        security_features=[],
        generated_at=datetime.now(),
        generator_version=generator_version
    )