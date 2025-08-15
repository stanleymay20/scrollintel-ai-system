"""
Autonomous Lab Testing API Routes

This module provides REST API endpoints for the autonomous innovation lab testing suite,
enabling comprehensive testing and validation of lab components.

Requirements: 1.1, 2.1, 3.1, 4.1, 5.1
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict, Any, List, Optional
from datetime import datetime
import asyncio

from tests.test_autonomous_lab_testing_suite import AutonomousLabTestingSuite
from scrollintel.core.logging_config import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/api/v1/autonomous-lab-testing", tags=["Autonomous Lab Testing"])

# Global testing suite instance
testing_suite = AutonomousLabTestingSuite()


@router.post("/run-research-engine-test")
async def run_research_engine_test() -> Dict[str, Any]:
    """
    Run research engine effectiveness testing
    
    Tests:
    - Research topic generation quality
    - Literature analysis accuracy
    - Hypothesis formation validity
    - Research planning effectiveness
    """
    try:
        logger.info("Starting research engine effectiveness test")
        
        results = await testing_suite.test_research_engine_effectiveness()
        
        logger.info(f"Research engine test completed with score: {results['overall_effectiveness']}")
        
        return {
            "success": True,
            "test_type": "research_engine_effectiveness",
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Research engine test failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Research engine test failed: {str(e)}")


@router.post("/run-experimental-design-test")
async def run_experimental_design_test() -> Dict[str, Any]:
    """
    Run experimental design quality validation
    
    Tests:
    - Experiment planning rigor
    - Protocol completeness
    - Resource allocation efficiency
    - Quality control effectiveness
    """
    try:
        logger.info("Starting experimental design quality test")
        
        results = await testing_suite.test_experimental_design_quality()
        
        logger.info(f"Experimental design test completed with score: {results['overall_quality']}")
        
        return {
            "success": True,
            "test_type": "experimental_design_quality",
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Experimental design test failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Experimental design test failed: {str(e)}")


@router.post("/run-prototype-development-test")
async def run_prototype_development_test() -> Dict[str, Any]:
    """
    Run prototype development success measurement
    
    Tests:
    - Rapid prototyping effectiveness
    - Design iteration quality
    - Testing automation accuracy
    - Performance evaluation precision
    """
    try:
        logger.info("Starting prototype development success test")
        
        results = await testing_suite.test_prototype_development_success()
        
        logger.info(f"Prototype development test completed with score: {results['overall_success']}")
        
        return {
            "success": True,
            "test_type": "prototype_development_success",
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Prototype development test failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prototype development test failed: {str(e)}")


@router.post("/run-comprehensive-test-suite")
async def run_comprehensive_test_suite(background_tasks: BackgroundTasks) -> Dict[str, Any]:
    """
    Run the complete autonomous lab testing suite
    
    Executes all test categories and provides comprehensive lab effectiveness assessment
    """
    try:
        logger.info("Starting comprehensive autonomous lab testing suite")
        
        # Run comprehensive test suite
        results = await testing_suite.run_comprehensive_test_suite()
        
        logger.info(f"Comprehensive test suite completed with overall score: {results['overall_lab_effectiveness']}")
        
        return {
            "success": True,
            "test_type": "comprehensive_suite",
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Comprehensive test suite failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Comprehensive test suite failed: {str(e)}")


@router.get("/test-results")
async def get_test_results() -> Dict[str, Any]:
    """
    Get stored test results from previous test runs
    """
    try:
        return {
            "success": True,
            "test_results": testing_suite.test_results,
            "performance_metrics": testing_suite.performance_metrics,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to retrieve test results: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve test results: {str(e)}")


@router.get("/test-status")
async def get_test_status() -> Dict[str, Any]:
    """
    Get current testing status and health metrics
    """
    try:
        # Calculate current lab health based on recent test results
        lab_health = "unknown"
        last_test_score = 0.0
        
        if testing_suite.test_results:
            latest_result = max(testing_suite.test_results.values(), 
                              key=lambda x: x.get('test_timestamp', datetime.min))
            
            if 'overall_lab_effectiveness' in latest_result:
                last_test_score = latest_result['overall_lab_effectiveness']
                
                if last_test_score >= 0.9:
                    lab_health = "excellent"
                elif last_test_score >= 0.8:
                    lab_health = "good"
                elif last_test_score >= 0.6:
                    lab_health = "fair"
                else:
                    lab_health = "poor"
        
        return {
            "success": True,
            "lab_health": lab_health,
            "last_test_score": last_test_score,
            "total_tests_run": len(testing_suite.test_results),
            "testing_suite_status": "active",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get test status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get test status: {str(e)}")


@router.post("/validate-component/{component_name}")
async def validate_specific_component(component_name: str) -> Dict[str, Any]:
    """
    Validate a specific autonomous lab component
    
    Args:
        component_name: Name of component to validate (research_engine, experiment_planner, etc.)
    """
    try:
        logger.info(f"Validating component: {component_name}")
        
        component_tests = {
            "research_engine": testing_suite.test_research_engine_effectiveness,
            "experiment_planner": testing_suite.test_experimental_design_quality,
            "rapid_prototyper": testing_suite.test_prototype_development_success
        }
        
        if component_name not in component_tests:
            raise HTTPException(
                status_code=400, 
                detail=f"Unknown component: {component_name}. Available: {list(component_tests.keys())}"
            )
        
        # Run specific component test
        results = await component_tests[component_name]()
        
        logger.info(f"Component {component_name} validation completed")
        
        return {
            "success": True,
            "component": component_name,
            "validation_results": results,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Component validation failed for {component_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Component validation failed: {str(e)}")


@router.post("/benchmark-performance")
async def benchmark_lab_performance() -> Dict[str, Any]:
    """
    Benchmark autonomous lab performance against baseline metrics
    """
    try:
        logger.info("Starting lab performance benchmarking")
        
        # Run comprehensive test suite for benchmarking
        results = await testing_suite.run_comprehensive_test_suite()
        
        # Define baseline performance metrics
        baseline_metrics = {
            "research_effectiveness": 0.85,
            "experimental_quality": 0.80,
            "prototype_success": 0.82,
            "overall_effectiveness": 0.82
        }
        
        # Compare against baselines
        performance_comparison = {
            "research_effectiveness": {
                "current": results['research_engine_effectiveness']['overall_effectiveness'],
                "baseline": baseline_metrics["research_effectiveness"],
                "improvement": results['research_engine_effectiveness']['overall_effectiveness'] - baseline_metrics["research_effectiveness"]
            },
            "experimental_quality": {
                "current": results['experimental_design_quality']['overall_quality'],
                "baseline": baseline_metrics["experimental_quality"],
                "improvement": results['experimental_design_quality']['overall_quality'] - baseline_metrics["experimental_quality"]
            },
            "prototype_success": {
                "current": results['prototype_development_success']['overall_success'],
                "baseline": baseline_metrics["prototype_success"],
                "improvement": results['prototype_development_success']['overall_success'] - baseline_metrics["prototype_success"]
            },
            "overall_effectiveness": {
                "current": results['overall_lab_effectiveness'],
                "baseline": baseline_metrics["overall_effectiveness"],
                "improvement": results['overall_lab_effectiveness'] - baseline_metrics["overall_effectiveness"]
            }
        }
        
        # Calculate benchmark score
        benchmark_score = sum(
            1 if comparison["improvement"] >= 0 else 0 
            for comparison in performance_comparison.values()
        ) / len(performance_comparison)
        
        logger.info(f"Lab performance benchmarking completed with score: {benchmark_score}")
        
        return {
            "success": True,
            "benchmark_score": benchmark_score,
            "performance_comparison": performance_comparison,
            "baseline_metrics": baseline_metrics,
            "current_results": results,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Lab performance benchmarking failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Benchmarking failed: {str(e)}")


@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """
    Health check endpoint for autonomous lab testing service
    """
    return {
        "status": "healthy",
        "service": "autonomous_lab_testing",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }