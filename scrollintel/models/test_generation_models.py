"""
Test Generation Models for Automated Code Generation System

This module defines the data models for automated test generation,
including test suites, test cases, and test execution results.
"""

from datetime import datetime
from enum import Enum
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class TestType(str, Enum):
    """Types of tests that can be generated"""
    UNIT = "unit"
    INTEGRATION = "integration"
    END_TO_END = "end_to_end"
    PERFORMANCE = "performance"
    LOAD = "load"
    SECURITY = "security"


class TestFramework(str, Enum):
    """Supported test frameworks"""
    PYTEST = "pytest"
    JEST = "jest"
    MOCHA = "mocha"
    JUNIT = "junit"
    CYPRESS = "cypress"
    SELENIUM = "selenium"


class TestStatus(str, Enum):
    """Test execution status"""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


class TestAssertion(BaseModel):
    """Individual test assertion"""
    assertion_type: str = Field(..., description="Type of assertion (equals, contains, etc.)")
    expected_value: Any = Field(..., description="Expected value for assertion")
    actual_expression: str = Field(..., description="Code expression to evaluate")
    description: str = Field(..., description="Human-readable assertion description")


class TestCase(BaseModel):
    """Individual test case model"""
    id: str = Field(..., description="Unique test case identifier")
    name: str = Field(..., description="Test case name")
    description: str = Field(..., description="Test case description")
    test_type: TestType = Field(..., description="Type of test")
    framework: TestFramework = Field(..., description="Test framework to use")
    
    # Test structure
    setup_code: Optional[str] = Field(None, description="Setup/arrange code")
    test_code: str = Field(..., description="Main test execution code")
    teardown_code: Optional[str] = Field(None, description="Cleanup code")
    
    # Test assertions
    assertions: List[TestAssertion] = Field(default_factory=list, description="Test assertions")
    
    # Test metadata
    tags: List[str] = Field(default_factory=list, description="Test tags for categorization")
    priority: int = Field(1, description="Test priority (1=high, 5=low)")
    timeout: Optional[int] = Field(None, description="Test timeout in seconds")
    
    # Dependencies
    dependencies: List[str] = Field(default_factory=list, description="Required dependencies/imports")
    fixtures: List[str] = Field(default_factory=list, description="Required test fixtures")
    
    # Execution results
    status: TestStatus = Field(TestStatus.PENDING, description="Test execution status")
    execution_time: Optional[float] = Field(None, description="Execution time in seconds")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class TestSuite(BaseModel):
    """Test suite containing multiple test cases"""
    id: str = Field(..., description="Unique test suite identifier")
    name: str = Field(..., description="Test suite name")
    description: str = Field(..., description="Test suite description")
    
    # Test organization
    test_cases: List[TestCase] = Field(default_factory=list, description="Test cases in suite")
    sub_suites: List['TestSuite'] = Field(default_factory=list, description="Nested test suites")
    
    # Suite configuration
    framework: TestFramework = Field(..., description="Primary test framework")
    parallel_execution: bool = Field(False, description="Whether tests can run in parallel")
    max_parallel_tests: Optional[int] = Field(None, description="Maximum parallel test count")
    
    # Coverage requirements
    target_coverage: float = Field(0.8, description="Target code coverage percentage")
    coverage_threshold: float = Field(0.7, description="Minimum acceptable coverage")
    
    # Execution configuration
    setup_script: Optional[str] = Field(None, description="Suite-level setup script")
    teardown_script: Optional[str] = Field(None, description="Suite-level teardown script")
    environment_config: Dict[str, Any] = Field(default_factory=dict, description="Test environment configuration")
    
    # Results tracking
    total_tests: int = Field(0, description="Total number of tests")
    passed_tests: int = Field(0, description="Number of passed tests")
    failed_tests: int = Field(0, description="Number of failed tests")
    skipped_tests: int = Field(0, description="Number of skipped tests")
    coverage_percentage: Optional[float] = Field(None, description="Actual coverage percentage")
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class TestGenerationRequest(BaseModel):
    """Request for generating tests"""
    target_code: str = Field(..., description="Code to generate tests for")
    code_type: str = Field(..., description="Type of code (function, class, module, api)")
    test_types: List[TestType] = Field(..., description="Types of tests to generate")
    framework: TestFramework = Field(..., description="Preferred test framework")
    
    # Generation options
    coverage_target: float = Field(0.8, description="Target coverage percentage")
    include_edge_cases: bool = Field(True, description="Include edge case testing")
    include_error_cases: bool = Field(True, description="Include error condition testing")
    include_performance_tests: bool = Field(False, description="Include performance testing")
    
    # Context information
    dependencies: List[str] = Field(default_factory=list, description="Code dependencies")
    existing_tests: List[str] = Field(default_factory=list, description="Existing test files")
    requirements: List[str] = Field(default_factory=list, description="Functional requirements")


class TestGenerationResult(BaseModel):
    """Result of test generation"""
    request_id: str = Field(..., description="Original request identifier")
    generated_suite: TestSuite = Field(..., description="Generated test suite")
    
    # Generation metadata
    generation_time: float = Field(..., description="Time taken to generate tests")
    estimated_coverage: float = Field(..., description="Estimated code coverage")
    test_count: int = Field(..., description="Number of tests generated")
    
    # Quality metrics
    complexity_score: float = Field(..., description="Test complexity score")
    maintainability_score: float = Field(..., description="Test maintainability score")
    
    # Recommendations
    recommendations: List[str] = Field(default_factory=list, description="Test improvement recommendations")
    warnings: List[str] = Field(default_factory=list, description="Generation warnings")
    
    created_at: datetime = Field(default_factory=datetime.utcnow)


class PerformanceTestConfig(BaseModel):
    """Configuration for performance tests"""
    load_pattern: str = Field("constant", description="Load pattern (constant, ramp, spike)")
    concurrent_users: int = Field(10, description="Number of concurrent users")
    duration_seconds: int = Field(60, description="Test duration in seconds")
    ramp_up_time: int = Field(10, description="Ramp up time in seconds")
    
    # Performance thresholds
    max_response_time: float = Field(2.0, description="Maximum acceptable response time")
    max_error_rate: float = Field(0.01, description="Maximum acceptable error rate")
    min_throughput: float = Field(100.0, description="Minimum required throughput")


class LoadTestScenario(BaseModel):
    """Load test scenario definition"""
    name: str = Field(..., description="Scenario name")
    description: str = Field(..., description="Scenario description")
    
    # Test steps
    test_steps: List[Dict[str, Any]] = Field(..., description="Test execution steps")
    
    # Load configuration
    config: PerformanceTestConfig = Field(..., description="Performance test configuration")
    
    # Data requirements
    test_data: Dict[str, Any] = Field(default_factory=dict, description="Test data requirements")
    data_generation_rules: List[str] = Field(default_factory=list, description="Data generation rules")


# Update TestSuite to handle forward reference
TestSuite.model_rebuild()