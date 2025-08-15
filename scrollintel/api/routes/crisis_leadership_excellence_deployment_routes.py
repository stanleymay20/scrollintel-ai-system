"""
API Routes for Crisis Leadership Excellence Deployment

Provides REST API endpoints for deploying, validating, and monitoring
the complete crisis leadership excellence system.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

from ...core.crisis_leadership_excellence_deployment import (
    CrisisLeadershipExcellenceDeployment,
    ValidationLevel,
    DeploymentStatus
)
from ...models.crisis_leadership_deployment_models import (
    DeploymentRequest,
    ValidationRequest,
    DeploymentResponse,
    ValidationResponse,
    SystemHealthResponse,
    LearningInsightsResponse
)

router = APIRouter(prefix="/api/v1/crisis-leadership-deployment", tags=["Crisis Leadership Deployment"])
logger = logging.getLogger(__name__)

# Global deployment system instance
deployment_system = CrisisLeadershipExcellenceDeployment()


@router.post("/deploy", response_model=DeploymentResponse)
async def deploy_crisis_leadership_system(
    request: DeploymentRequest,
    background_tasks: BackgroundTasks
):
    """
    Deploy complete crisis leadership excellence system
    """
    try:
        logger.info(f"Starting crisis leadership system deployment with validation level: {request.validation_level}")
        
        # Start deployment process
        deployment_metrics = await deployment_system.deploy_complete_system(
            validation_level=ValidationLevel(request.validation_level)
        )
        
        return DeploymentResponse(
            deployment_id=f"deployment_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            status=deployment_system.deployment_status.value,
            deployment_timestamp=deployment_metrics.deployment_timestamp,
            validation_level=deployment_metrics.validation_level.value,
            overall_readiness_score=deployment_metrics.overall_readiness_score,
            deployment_success=deployment_metrics.deployment_success,
            component_health=deployment_metrics.component_health,
            integration_scores=deployment_metrics.integration_scores,
            crisis_response_capabilities=deployment_metrics.crisis_response_capabilities,
            performance_benchmarks=deployment_metrics.performance_benchmarks,
            continuous_learning_metrics=deployment_metrics.continuous_learning_metrics,
            message="Crisis leadership excellence system deployment completed successfully" if deployment_metrics.deployment_success else "Deployment failed"
        )
        
    except Exception as e:
        logger.error(f"Crisis leadership deployment failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Deployment failed: {str(e)}")


@router.post("/validate", response_model=ValidationResponse)
async def validate_crisis_leadership_excellence(
    request: ValidationRequest
):
    """
    Validate crisis leadership excellence across all crisis types
    """
    try:
        logger.info("Starting crisis leadership excellence validation")
        
        # Perform validation
        validation_results = await deployment_system.validate_crisis_leadership_excellence(
            validation_scenarios=request.validation_scenarios
        )
        
        return ValidationResponse(
            validation_id=f"validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            validation_timestamp=validation_results['validation_timestamp'],
            scenarios_tested=validation_results['scenarios_tested'],
            overall_success_rate=validation_results['overall_success_rate'],
            average_response_time=validation_results['average_response_time'],
            leadership_effectiveness=validation_results['leadership_effectiveness'],
            stakeholder_satisfaction=validation_results['stakeholder_satisfaction'],
            crisis_type_performance=validation_results['crisis_type_performance'],
            detailed_results=validation_results['detailed_results'],
            validation_success=validation_results['overall_success_rate'] >= 0.8,
            message="Crisis leadership excellence validation completed successfully"
        )
        
    except Exception as e:
        logger.error(f"Crisis leadership validation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")


@router.get("/status", response_model=SystemHealthResponse)
async def get_deployment_status():
    """
    Get current deployment status and system health
    """
    try:
        status_info = await deployment_system.get_deployment_status()
        
        return SystemHealthResponse(
            deployment_status=status_info['deployment_status'],
            deployment_metrics=status_info['deployment_metrics'],
            validation_history_count=status_info['validation_history_count'],
            learning_data_points=status_info['learning_data_points'],
            system_health=status_info['system_health'],
            last_updated=datetime.now(),
            message="System status retrieved successfully"
        )
        
    except Exception as e:
        logger.error(f"Failed to get deployment status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Status retrieval failed: {str(e)}")


@router.get("/learning-insights", response_model=LearningInsightsResponse)
async def get_continuous_learning_insights():
    """
    Get insights from continuous learning system
    """
    try:
        insights = deployment_system.get_continuous_learning_insights()
        
        return LearningInsightsResponse(
            insights_timestamp=datetime.now(),
            total_crises_handled=insights.get('total_crises_handled', 0),
            average_response_time=insights.get('average_response_time', 0.0),
            average_effectiveness=insights.get('average_effectiveness', 0.0),
            improvement_trend=insights.get('improvement_trend', 'unknown'),
            best_performing_scenarios=insights.get('best_performing_scenarios', []),
            improvement_opportunities=insights.get('improvement_opportunities', []),
            learning_system_active=True,
            message="Continuous learning insights retrieved successfully"
        )
        
    except Exception as e:
        logger.error(f"Failed to get learning insights: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Learning insights retrieval failed: {str(e)}")


@router.post("/test-crisis-response")
async def test_crisis_response(
    crisis_signals: List[Dict[str, Any]]
):
    """
    Test crisis response with provided signals
    """
    try:
        logger.info("Testing crisis response with provided signals")
        
        # Execute crisis response
        start_time = datetime.now()
        response = await deployment_system.crisis_system.handle_crisis(crisis_signals)
        end_time = datetime.now()
        
        response_time = (end_time - start_time).total_seconds()
        
        return {
            "test_id": f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "response_time": response_time,
            "crisis_id": response.crisis_id,
            "response_plan": response.response_plan,
            "team_formation": response.team_formation,
            "resource_allocation": response.resource_allocation,
            "communication_strategy": response.communication_strategy,
            "success_metrics": response.success_metrics,
            "message": "Crisis response test completed successfully"
        }
        
    except Exception as e:
        logger.error(f"Crisis response test failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Crisis response test failed: {str(e)}")


@router.post("/stress-test")
async def run_stress_test():
    """
    Run comprehensive stress test of crisis leadership system
    """
    try:
        logger.info("Starting crisis leadership stress test")
        
        # Run stress test with production-ready validation
        deployment_metrics = await deployment_system.deploy_complete_system(
            validation_level=ValidationLevel.STRESS_TEST
        )
        
        # Additional stress testing
        stress_scenarios = [
            {
                "signals": [
                    {"type": "system_alert", "severity": "catastrophic", "affected_services": ["all"]},
                    {"type": "security_alert", "severity": "critical", "breach_type": "data_exposure"},
                    {"type": "financial_alert", "metric": "revenue_loss", "impact": "severe"}
                ]
            }
        ] * 10  # 10 concurrent extreme scenarios
        
        validation_results = await deployment_system.validate_crisis_leadership_excellence(
            validation_scenarios=stress_scenarios
        )
        
        return {
            "stress_test_id": f"stress_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "deployment_readiness": deployment_metrics.overall_readiness_score,
            "stress_test_success_rate": validation_results['overall_success_rate'],
            "average_response_time_under_stress": validation_results['average_response_time'],
            "leadership_effectiveness_under_stress": validation_results['leadership_effectiveness'],
            "scenarios_tested": validation_results['scenarios_tested'],
            "stress_test_passed": validation_results['overall_success_rate'] >= 0.7,
            "message": "Stress test completed successfully"
        }
        
    except Exception as e:
        logger.error(f"Stress test failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Stress test failed: {str(e)}")


@router.get("/deployment-history")
async def get_deployment_history():
    """
    Get deployment and validation history
    """
    try:
        history = [
            {
                "deployment_timestamp": metrics.deployment_timestamp,
                "validation_level": metrics.validation_level.value,
                "overall_readiness_score": metrics.overall_readiness_score,
                "deployment_success": metrics.deployment_success,
                "component_health_avg": sum(metrics.component_health.values()) / len(metrics.component_health) if metrics.component_health else 0,
                "crisis_capabilities_avg": sum(metrics.crisis_response_capabilities.values()) / len(metrics.crisis_response_capabilities) if metrics.crisis_response_capabilities else 0
            }
            for metrics in deployment_system.validation_history
        ]
        
        return {
            "deployment_history": history,
            "total_deployments": len(history),
            "latest_deployment": history[-1] if history else None,
            "message": "Deployment history retrieved successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to get deployment history: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Deployment history retrieval failed: {str(e)}")


@router.post("/emergency-deployment")
async def emergency_deployment():
    """
    Emergency deployment with minimal validation for critical situations
    """
    try:
        logger.warning("Starting emergency deployment with minimal validation")
        
        # Deploy with basic validation only
        deployment_metrics = await deployment_system.deploy_complete_system(
            validation_level=ValidationLevel.BASIC
        )
        
        return {
            "emergency_deployment_id": f"emergency_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "deployment_status": deployment_system.deployment_status.value,
            "readiness_score": deployment_metrics.overall_readiness_score,
            "deployment_success": deployment_metrics.deployment_success,
            "warning": "Emergency deployment completed with minimal validation",
            "recommendation": "Run full validation as soon as possible",
            "message": "Emergency deployment completed"
        }
        
    except Exception as e:
        logger.error(f"Emergency deployment failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Emergency deployment failed: {str(e)}")


@router.get("/health")
async def health_check():
    """
    Health check endpoint for crisis leadership deployment system
    """
    try:
        system_health = await deployment_system._get_system_health_summary()
        
        return {
            "status": "healthy",
            "timestamp": datetime.now(),
            "deployment_system_status": deployment_system.deployment_status.value,
            "system_health": system_health,
            "message": "Crisis leadership deployment system is healthy"
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.now(),
            "error": str(e),
            "message": "Crisis leadership deployment system health check failed"
        }