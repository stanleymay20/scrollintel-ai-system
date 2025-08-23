"""
Security Testing API Routes
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from typing import Dict, List, Any, Optional
import asyncio
import logging
from datetime import datetime

from ...testing import (
    SecurityTestFramework,
    SecurityTestResult,
    SecurityTestType,
    SecuritySeverity
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/security/testing", tags=["security-testing"])

# Global test framework instance
security_test_framework = None

def get_security_test_framework():
    """Get or create security test framework instance"""
    global security_test_framework
    if security_test_framework is None:
        config = {
            'penetration': {'enabled': True},
            'vulnerability': {'enabled': True},
            'chaos': {'enabled': True},
            'performance': {'enabled': True},
            'regression': {'enabled': True},
            'metrics': {'enabled': True}
        }
        security_test_framework = SecurityTestFramework(config)
    return security_test_framework

@router.post("/run-comprehensive-tests")
async def run_comprehensive_security_tests(
    target_config: Dict[str, Any],
    background_tasks: BackgroundTasks,
    framework: SecurityTestFramework = Depends(get_security_test_framework)
):
    """Run comprehensive security testing suite"""
    try:
        logger.info("Starting comprehensive security tests")
        
        # Validate target configuration
        if not target_config.get('base_url'):
            raise HTTPException(status_code=400, detail="base_url is required in target_config")
        
        # Run tests
        results = await framework.run_comprehensive_security_tests(target_config)
        
        # Convert results to serializable format
        serialized_results = []
        for result in results:
            serialized_results.append({
                "test_id": result.test_id,
                "test_type": result.test_type.value,
                "test_name": result.test_name,
                "status": result.status,
                "severity": result.severity.value,
                "findings": result.findings,
                "execution_time": result.execution_time,
                "timestamp": result.timestamp.isoformat(),
                "metadata": result.metadata,
                "recommendations": result.recommendations
            })
        
        return {
            "status": "completed",
            "total_tests": len(results),
            "passed_tests": len([r for r in results if r.status == "passed"]),
            "failed_tests": len([r for r in results if r.status == "failed"]),
            "results": serialized_results,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Comprehensive security tests failed: {e}")
        raise HTTPException(status_code=500, detail=f"Security testing failed: {str(e)}")

@router.post("/run-penetration-tests")
async def run_penetration_tests(
    target_config: Dict[str, Any],
    framework: SecurityTestFramework = Depends(get_security_test_framework)
):
    """Run penetration testing suite"""
    try:
        logger.info("Starting penetration tests")
        
        results = await framework._run_penetration_tests(target_config)
        
        serialized_results = []
        for result in results:
            serialized_results.append({
                "test_id": result.test_id,
                "test_name": result.test_name,
                "status": result.status,
                "severity": result.severity.value,
                "findings": result.findings,
                "execution_time": result.execution_time,
                "recommendations": result.recommendations
            })
        
        return {
            "status": "completed",
            "test_type": "penetration",
            "results": serialized_results,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Penetration tests failed: {e}")
        raise HTTPException(status_code=500, detail=f"Penetration testing failed: {str(e)}")

@router.post("/run-vulnerability-scan")
async def run_vulnerability_scan(
    target_config: Dict[str, Any],
    framework: SecurityTestFramework = Depends(get_security_test_framework)
):
    """Run vulnerability scanning"""
    try:
        logger.info("Starting vulnerability scan")
        
        results = await framework._run_vulnerability_scans(target_config)
        
        serialized_results = []
        for result in results:
            serialized_results.append({
                "test_id": result.test_id,
                "test_name": result.test_name,
                "status": result.status,
                "severity": result.severity.value,
                "findings": result.findings,
                "execution_time": result.execution_time,
                "recommendations": result.recommendations
            })
        
        return {
            "status": "completed",
            "test_type": "vulnerability",
            "results": serialized_results,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Vulnerability scan failed: {e}")
        raise HTTPException(status_code=500, detail=f"Vulnerability scanning failed: {str(e)}")

@router.post("/run-chaos-tests")
async def run_chaos_tests(
    target_config: Dict[str, Any],
    framework: SecurityTestFramework = Depends(get_security_test_framework)
):
    """Run security chaos engineering tests"""
    try:
        logger.info("Starting chaos engineering tests")
        
        results = await framework._run_chaos_tests(target_config)
        
        serialized_results = []
        for result in results:
            serialized_results.append({
                "test_id": result.test_id,
                "test_name": result.test_name,
                "status": result.status,
                "severity": result.severity.value,
                "findings": result.findings,
                "execution_time": result.execution_time,
                "metadata": result.metadata,
                "recommendations": result.recommendations
            })
        
        return {
            "status": "completed",
            "test_type": "chaos",
            "results": serialized_results,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Chaos tests failed: {e}")
        raise HTTPException(status_code=500, detail=f"Chaos testing failed: {str(e)}")

@router.post("/run-performance-tests")
async def run_performance_tests(
    target_config: Dict[str, Any],
    framework: SecurityTestFramework = Depends(get_security_test_framework)
):
    """Run security performance tests"""
    try:
        logger.info("Starting security performance tests")
        
        results = await framework._run_performance_tests(target_config)
        
        serialized_results = []
        for result in results:
            serialized_results.append({
                "test_id": result.test_id,
                "test_name": result.test_name,
                "status": result.status,
                "severity": result.severity.value,
                "findings": result.findings,
                "execution_time": result.execution_time,
                "metadata": result.metadata,
                "recommendations": result.recommendations
            })
        
        return {
            "status": "completed",
            "test_type": "performance",
            "results": serialized_results,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Performance tests failed: {e}")
        raise HTTPException(status_code=500, detail=f"Performance testing failed: {str(e)}")

@router.post("/run-regression-tests")
async def run_regression_tests(
    target_config: Dict[str, Any],
    framework: SecurityTestFramework = Depends(get_security_test_framework)
):
    """Run security regression tests"""
    try:
        logger.info("Starting security regression tests")
        
        results = await framework._run_regression_tests(target_config)
        
        serialized_results = []
        for result in results:
            serialized_results.append({
                "test_id": result.test_id,
                "test_name": result.test_name,
                "status": result.status,
                "severity": result.severity.value,
                "findings": result.findings,
                "execution_time": result.execution_time,
                "metadata": result.metadata,
                "recommendations": result.recommendations
            })
        
        return {
            "status": "completed",
            "test_type": "regression",
            "results": serialized_results,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Regression tests failed: {e}")
        raise HTTPException(status_code=500, detail=f"Regression testing failed: {str(e)}")

@router.get("/metrics")
async def get_security_metrics(
    framework: SecurityTestFramework = Depends(get_security_test_framework)
):
    """Get security testing metrics"""
    try:
        # Get metrics from the framework
        if hasattr(framework, 'metrics_collector') and framework.metrics_collector:
            dashboard_data = framework.metrics_collector.get_metrics_dashboard_data()
            return {
                "status": "success",
                "metrics": dashboard_data,
                "timestamp": datetime.now().isoformat()
            }
        else:
            return {
                "status": "no_data",
                "message": "No metrics available yet",
                "timestamp": datetime.now().isoformat()
            }
    
    except Exception as e:
        logger.error(f"Failed to get security metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")

@router.get("/test-history")
async def get_test_history(
    limit: int = 100,
    test_type: Optional[str] = None,
    framework: SecurityTestFramework = Depends(get_security_test_framework)
):
    """Get security test history"""
    try:
        # Filter test results from framework history
        test_results = framework.test_results[-limit:] if hasattr(framework, 'test_results') else []
        
        if test_type:
            test_results = [r for r in test_results if r.test_type.value == test_type]
        
        serialized_results = []
        for result in test_results:
            serialized_results.append({
                "test_id": result.test_id,
                "test_type": result.test_type.value,
                "test_name": result.test_name,
                "status": result.status,
                "severity": result.severity.value,
                "execution_time": result.execution_time,
                "timestamp": result.timestamp.isoformat()
            })
        
        return {
            "status": "success",
            "total_results": len(serialized_results),
            "results": serialized_results,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Failed to get test history: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get test history: {str(e)}")

@router.get("/test-status/{test_id}")
async def get_test_status(
    test_id: str,
    framework: SecurityTestFramework = Depends(get_security_test_framework)
):
    """Get status of specific test"""
    try:
        # Check if test is currently running
        if hasattr(framework, 'active_tests') and test_id in framework.active_tests:
            return {
                "test_id": test_id,
                "status": "running",
                "start_time": framework.active_tests[test_id].get('start_time'),
                "timestamp": datetime.now().isoformat()
            }
        
        # Check test results history
        if hasattr(framework, 'test_results'):
            for result in reversed(framework.test_results):
                if result.test_id == test_id:
                    return {
                        "test_id": test_id,
                        "status": result.status,
                        "severity": result.severity.value,
                        "execution_time": result.execution_time,
                        "timestamp": result.timestamp.isoformat(),
                        "findings_count": len(result.findings)
                    }
        
        return {
            "test_id": test_id,
            "status": "not_found",
            "message": "Test not found",
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Failed to get test status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get test status: {str(e)}")

@router.post("/create-baseline")
async def create_security_baseline(
    version: str,
    description: str = "",
    framework: SecurityTestFramework = Depends(get_security_test_framework)
):
    """Create security testing baseline"""
    try:
        if hasattr(framework, 'regression_tester') and framework.regression_tester:
            baseline_id = framework.regression_tester.create_baseline(version, description)
            
            return {
                "status": "success",
                "baseline_id": baseline_id,
                "version": version,
                "description": description,
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(status_code=400, detail="Regression tester not available")
    
    except Exception as e:
        logger.error(f"Failed to create baseline: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create baseline: {str(e)}")

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "security-testing",
        "timestamp": datetime.now().isoformat()
    }