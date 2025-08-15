"""
API Routes for Automated Test Generation System

This module provides REST API endpoints for generating automated tests
for code components, including unit tests, integration tests, and
performance tests.
"""

from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from pydantic import BaseModel

from ...models.test_generation_models import (
    TestSuite, TestCase, TestGenerationRequest, TestGenerationResult,
    TestType, TestFramework, PerformanceTestConfig, LoadTestScenario
)
from ...engines.test_generator import TestGenerator
from ...core.auth import get_current_user
from ...core.logging_config import get_logger

router = APIRouter(prefix="/api/v1/test-generation", tags=["test-generation"])
logger = get_logger(__name__)


# Request/Response Models
class GenerateTestsRequest(BaseModel):
    """Request model for test generation"""
    target_code: str
    code_type: str = "function"  # function, class, module, api
    test_types: List[TestType] = [TestType.UNIT]
    framework: TestFramework = TestFramework.PYTEST
    coverage_target: float = 0.8
    include_edge_cases: bool = True
    include_error_cases: bool = True
    include_performance_tests: bool = False
    dependencies: List[str] = []
    existing_tests: List[str] = []
    requirements: List[str] = []


class TestExecutionRequest(BaseModel):
    """Request model for test execution"""
    test_suite_id: str
    environment: str = "test"
    parallel: bool = False
    max_workers: Optional[int] = None


class TestExecutionResult(BaseModel):
    """Result of test execution"""
    execution_id: str
    test_suite_id: str
    status: str
    total_tests: int
    passed_tests: int
    failed_tests: int
    skipped_tests: int
    execution_time: float
    coverage_percentage: Optional[float]
    detailed_results: List[Dict[str, Any]]


# Initialize test generator
test_generator = TestGenerator()


@router.post("/generate", response_model=TestGenerationResult)
async def generate_tests(
    request: GenerateTestsRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    """
    Generate comprehensive test suite for provided code
    
    This endpoint analyzes the provided code and generates appropriate
    tests based on the specified test types and configuration.
    """
    try:
        logger.info(f"Generating tests for code type: {request.code_type}")
        
        # Convert request to internal model
        generation_request = TestGenerationRequest(
            target_code=request.target_code,
            code_type=request.code_type,
            test_types=request.test_types,
            framework=request.framework,
            coverage_target=request.coverage_target,
            include_edge_cases=request.include_edge_cases,
            include_error_cases=request.include_error_cases,
            include_performance_tests=request.include_performance_tests,
            dependencies=request.dependencies,
            existing_tests=request.existing_tests,
            requirements=request.requirements
        )
        
        # Generate tests
        result = test_generator.generate_comprehensive_tests(generation_request)
        
        # Log generation metrics
        logger.info(f"Generated {result.test_count} tests with {result.estimated_coverage:.2%} coverage")
        
        # Store test suite in background (if needed)
        background_tasks.add_task(store_test_suite, result.generated_suite)
        
        return result
        
    except Exception as e:
        logger.error(f"Error generating tests: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Test generation failed: {str(e)}")


@router.get("/suites", response_model=List[TestSuite])
async def list_test_suites(
    skip: int = 0,
    limit: int = 100,
    test_type: Optional[TestType] = None,
    framework: Optional[TestFramework] = None,
    current_user: dict = Depends(get_current_user)
):
    """
    List all test suites with optional filtering
    """
    try:
        # This would typically query a database
        # For now, return empty list as placeholder
        return []
        
    except Exception as e:
        logger.error(f"Error listing test suites: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list test suites: {str(e)}")


@router.get("/suites/{suite_id}", response_model=TestSuite)
async def get_test_suite(
    suite_id: str,
    current_user: dict = Depends(get_current_user)
):
    """
    Get specific test suite by ID
    """
    try:
        # This would typically query a database
        # For now, raise not found
        raise HTTPException(status_code=404, detail="Test suite not found")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving test suite {suite_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve test suite: {str(e)}")


@router.post("/suites/{suite_id}/execute", response_model=TestExecutionResult)
async def execute_test_suite(
    suite_id: str,
    request: TestExecutionRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    """
    Execute a test suite and return results
    """
    try:
        logger.info(f"Executing test suite {suite_id}")
        
        # This would typically:
        # 1. Retrieve the test suite
        # 2. Set up test environment
        # 3. Execute tests
        # 4. Collect results
        
        # For now, return mock result
        result = TestExecutionResult(
            execution_id="exec_123",
            test_suite_id=suite_id,
            status="completed",
            total_tests=10,
            passed_tests=8,
            failed_tests=2,
            skipped_tests=0,
            execution_time=45.5,
            coverage_percentage=0.85,
            detailed_results=[]
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error executing test suite {suite_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Test execution failed: {str(e)}")


@router.post("/generate-unit-tests", response_model=List[TestCase])
async def generate_unit_tests_only(
    request: GenerateTestsRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Generate only unit tests for provided code
    """
    try:
        logger.info("Generating unit tests only")
        
        # Force unit tests only
        generation_request = TestGenerationRequest(
            target_code=request.target_code,
            code_type=request.code_type,
            test_types=[TestType.UNIT],
            framework=request.framework,
            coverage_target=request.coverage_target,
            include_edge_cases=request.include_edge_cases,
            include_error_cases=request.include_error_cases,
            dependencies=request.dependencies,
            requirements=request.requirements
        )
        
        result = test_generator.generate_comprehensive_tests(generation_request)
        return result.generated_suite.test_cases
        
    except Exception as e:
        logger.error(f"Error generating unit tests: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unit test generation failed: {str(e)}")


@router.post("/generate-integration-tests", response_model=List[TestCase])
async def generate_integration_tests_only(
    request: GenerateTestsRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Generate only integration tests for provided code
    """
    try:
        logger.info("Generating integration tests only")
        
        # Force integration tests only
        generation_request = TestGenerationRequest(
            target_code=request.target_code,
            code_type=request.code_type,
            test_types=[TestType.INTEGRATION],
            framework=request.framework,
            coverage_target=request.coverage_target,
            dependencies=request.dependencies,
            requirements=request.requirements
        )
        
        result = test_generator.generate_comprehensive_tests(generation_request)
        return result.generated_suite.test_cases
        
    except Exception as e:
        logger.error(f"Error generating integration tests: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Integration test generation failed: {str(e)}")


@router.post("/generate-e2e-tests", response_model=List[TestCase])
async def generate_e2e_tests_only(
    request: GenerateTestsRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Generate only end-to-end tests for provided code
    """
    try:
        logger.info("Generating end-to-end tests only")
        
        # Force E2E tests only
        generation_request = TestGenerationRequest(
            target_code=request.target_code,
            code_type=request.code_type,
            test_types=[TestType.END_TO_END],
            framework=TestFramework.CYPRESS,  # Default to Cypress for E2E
            coverage_target=request.coverage_target,
            dependencies=request.dependencies,
            requirements=request.requirements
        )
        
        result = test_generator.generate_comprehensive_tests(generation_request)
        return result.generated_suite.test_cases
        
    except Exception as e:
        logger.error(f"Error generating E2E tests: {str(e)}")
        raise HTTPException(status_code=500, detail=f"E2E test generation failed: {str(e)}")


@router.post("/generate-performance-tests", response_model=List[TestCase])
async def generate_performance_tests_only(
    request: GenerateTestsRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Generate only performance tests for provided code
    """
    try:
        logger.info("Generating performance tests only")
        
        # Force performance tests only
        generation_request = TestGenerationRequest(
            target_code=request.target_code,
            code_type=request.code_type,
            test_types=[TestType.PERFORMANCE, TestType.LOAD],
            framework=request.framework,
            include_performance_tests=True,
            dependencies=request.dependencies,
            requirements=request.requirements
        )
        
        result = test_generator.generate_comprehensive_tests(generation_request)
        return result.generated_suite.test_cases
        
    except Exception as e:
        logger.error(f"Error generating performance tests: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Performance test generation failed: {str(e)}")


@router.post("/validate-tests", response_model=Dict[str, Any])
async def validate_test_suite_quality(
    test_suite: TestSuite,
    current_user: dict = Depends(get_current_user)
):
    """
    Validate test suite quality and provide recommendations
    """
    try:
        logger.info(f"Validating test suite: {test_suite.name}")
        
        validation_result = test_generator.test_validator.validate_test_suite(test_suite)
        
        return {
            "suite_id": test_suite.id,
            "validation_result": validation_result,
            "quality_score": (validation_result['maintainability_score'] + 
                            (1 - validation_result['complexity_score'])) / 2,
            "timestamp": test_suite.updated_at
        }
        
    except Exception as e:
        logger.error(f"Error validating test suite: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Test validation failed: {str(e)}")


@router.get("/frameworks", response_model=List[str])
async def get_supported_frameworks():
    """
    Get list of supported test frameworks
    """
    return [framework.value for framework in TestFramework]


@router.get("/test-types", response_model=List[str])
async def get_supported_test_types():
    """
    Get list of supported test types
    """
    return [test_type.value for test_type in TestType]


@router.post("/export-tests/{suite_id}")
async def export_test_suite(
    suite_id: str,
    format: str = "pytest",
    current_user: dict = Depends(get_current_user)
):
    """
    Export test suite to executable test files
    """
    try:
        logger.info(f"Exporting test suite {suite_id} in {format} format")
        
        # This would typically:
        # 1. Retrieve the test suite
        # 2. Generate executable test files
        # 3. Return downloadable archive
        
        return {"message": f"Test suite exported in {format} format", "download_url": "/downloads/tests.zip"}
        
    except Exception as e:
        logger.error(f"Error exporting test suite {suite_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Test export failed: {str(e)}")


# Background task functions
async def store_test_suite(test_suite: TestSuite):
    """Store test suite in database (background task)"""
    try:
        # This would typically store in database
        logger.info(f"Storing test suite: {test_suite.name}")
        
    except Exception as e:
        logger.error(f"Error storing test suite: {str(e)}")


# Health check endpoint
@router.get("/health")
async def health_check():
    """Health check for test generation service"""
    return {
        "status": "healthy",
        "service": "test-generation",
        "timestamp": "2024-01-01T00:00:00Z"
    }