"""
Crisis Response Effectiveness Testing API Routes

API endpoints for crisis response effectiveness testing and validation.
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from pydantic import BaseModel, Field

from scrollintel.engines.crisis_response_effectiveness_testing import (
    CrisisResponseEffectivenessTesting,
    EffectivenessMetric,
    TestingPhase
)

router = APIRouter(prefix="/api/crisis-response-effectiveness", tags=["Crisis Response Effectiveness"])

# Initialize the testing engine
effectiveness_testing = CrisisResponseEffectivenessTesting()

# Pydantic models for request/response
class StartTestRequest(BaseModel):
    crisis_scenario: str = Field(..., description="Description of the crisis scenario to test")
    test_type: str = Field(default="comprehensive", description="Type of effectiveness test")

class SpeedMeasurementRequest(BaseModel):
    test_id: str = Field(..., description="ID of the effectiveness test")
    detection_time: datetime = Field(..., description="Time when crisis was detected")
    first_response_time: datetime = Field(..., description="Time of first response action")
    full_response_time: datetime = Field(..., description="Time when full response was deployed")

class DecisionQualityRequest(BaseModel):
    test_id: str = Field(..., description="ID of the effectiveness test")
    decisions_made: List[Dict[str, Any]] = Field(..., description="List of decisions made during crisis")
    decision_outcomes: List[Dict[str, Any]] = Field(..., description="Outcomes and evaluations of decisions")

class CommunicationEffectivenessRequest(BaseModel):
    test_id: str = Field(..., description="ID of the effectiveness test")
    communications_sent: List[Dict[str, Any]] = Field(..., description="List of communications sent")
    stakeholder_feedback: List[Dict[str, Any]] = Field(..., description="Feedback from stakeholders")

class OutcomeSuccessRequest(BaseModel):
    test_id: str = Field(..., description="ID of the effectiveness test")
    crisis_objectives: List[str] = Field(..., description="Original crisis response objectives")
    achieved_outcomes: List[Dict[str, Any]] = Field(..., description="Actual outcomes achieved")

class LeadershipEffectivenessRequest(BaseModel):
    test_id: str = Field(..., description="ID of the effectiveness test")
    leadership_actions: List[Dict[str, Any]] = Field(..., description="Leadership actions taken")
    team_feedback: List[Dict[str, Any]] = Field(..., description="Feedback from team members")
    stakeholder_confidence: Dict[str, float] = Field(..., description="Stakeholder confidence ratings")

class TrendsRequest(BaseModel):
    metric: Optional[str] = Field(None, description="Specific metric to analyze")
    time_period_days: Optional[int] = Field(30, description="Time period in days")

class BaselineUpdateRequest(BaseModel):
    baselines: Dict[str, float] = Field(..., description="New baseline metrics")

@router.post("/start-test")
async def start_effectiveness_test(request: StartTestRequest) -> Dict[str, Any]:
    """Start a new crisis response effectiveness test"""
    try:
        test_id = await effectiveness_testing.start_effectiveness_test(
            crisis_scenario=request.crisis_scenario,
            test_type=request.test_type
        )
        
        return {
            "success": True,
            "test_id": test_id,
            "message": "Crisis response effectiveness test started successfully",
            "start_time": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start effectiveness test: {str(e)}")

@router.post("/measure-speed")
async def measure_response_speed(request: SpeedMeasurementRequest) -> Dict[str, Any]:
    """Measure crisis response speed effectiveness"""
    try:
        score = await effectiveness_testing.measure_response_speed(
            test_id=request.test_id,
            detection_time=request.detection_time,
            first_response_time=request.first_response_time,
            full_response_time=request.full_response_time
        )
        
        return {
            "success": True,
            "metric": score.metric.value,
            "score": score.score,
            "details": score.details,
            "confidence_level": score.confidence_level,
            "measurement_time": score.measurement_time.isoformat()
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to measure response speed: {str(e)}")

@router.post("/measure-decision-quality")
async def measure_decision_quality(request: DecisionQualityRequest) -> Dict[str, Any]:
    """Measure quality of decisions made during crisis"""
    try:
        score = await effectiveness_testing.measure_decision_quality(
            test_id=request.test_id,
            decisions_made=request.decisions_made,
            decision_outcomes=request.decision_outcomes
        )
        
        return {
            "success": True,
            "metric": score.metric.value,
            "score": score.score,
            "details": score.details,
            "confidence_level": score.confidence_level,
            "measurement_time": score.measurement_time.isoformat()
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to measure decision quality: {str(e)}")

@router.post("/measure-communication")
async def measure_communication_effectiveness(request: CommunicationEffectivenessRequest) -> Dict[str, Any]:
    """Measure effectiveness of crisis communications"""
    try:
        score = await effectiveness_testing.measure_communication_effectiveness(
            test_id=request.test_id,
            communications_sent=request.communications_sent,
            stakeholder_feedback=request.stakeholder_feedback
        )
        
        return {
            "success": True,
            "metric": score.metric.value,
            "score": score.score,
            "details": score.details,
            "confidence_level": score.confidence_level,
            "measurement_time": score.measurement_time.isoformat()
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to measure communication effectiveness: {str(e)}")

@router.post("/measure-outcomes")
async def measure_outcome_success(request: OutcomeSuccessRequest) -> Dict[str, Any]:
    """Measure success of crisis outcomes against objectives"""
    try:
        score = await effectiveness_testing.measure_outcome_success(
            test_id=request.test_id,
            crisis_objectives=request.crisis_objectives,
            achieved_outcomes=request.achieved_outcomes
        )
        
        return {
            "success": True,
            "metric": score.metric.value,
            "score": score.score,
            "details": score.details,
            "confidence_level": score.confidence_level,
            "measurement_time": score.measurement_time.isoformat()
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to measure outcome success: {str(e)}")

@router.post("/measure-leadership")
async def measure_leadership_effectiveness(request: LeadershipEffectivenessRequest) -> Dict[str, Any]:
    """Measure effectiveness of crisis leadership"""
    try:
        score = await effectiveness_testing.measure_leadership_effectiveness(
            test_id=request.test_id,
            leadership_actions=request.leadership_actions,
            team_feedback=request.team_feedback,
            stakeholder_confidence=request.stakeholder_confidence
        )
        
        return {
            "success": True,
            "metric": score.metric.value,
            "score": score.score,
            "details": score.details,
            "confidence_level": score.confidence_level,
            "measurement_time": score.measurement_time.isoformat()
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to measure leadership effectiveness: {str(e)}")

@router.post("/complete-test/{test_id}")
async def complete_effectiveness_test(test_id: str) -> Dict[str, Any]:
    """Complete an effectiveness test and calculate overall results"""
    try:
        test = await effectiveness_testing.complete_effectiveness_test(test_id)
        
        return {
            "success": True,
            "test_id": test.test_id,
            "overall_score": test.overall_score,
            "duration_seconds": (test.end_time - test.start_time).total_seconds(),
            "effectiveness_scores": [
                {
                    "metric": score.metric.value,
                    "score": score.score,
                    "confidence_level": score.confidence_level
                }
                for score in test.effectiveness_scores
            ],
            "recommendations": test.recommendations,
            "end_time": test.end_time.isoformat()
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to complete effectiveness test: {str(e)}")

@router.get("/trends")
async def get_effectiveness_trends(
    metric: Optional[str] = None,
    time_period_days: int = 30
) -> Dict[str, Any]:
    """Get effectiveness trends over time"""
    try:
        effectiveness_metric = None
        if metric:
            try:
                effectiveness_metric = EffectivenessMetric(metric)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid metric: {metric}")
        
        time_period = timedelta(days=time_period_days)
        trends = await effectiveness_testing.get_effectiveness_trends(
            metric=effectiveness_metric,
            time_period=time_period
        )
        
        return {
            "success": True,
            "trends": trends,
            "time_period_days": time_period_days
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get effectiveness trends: {str(e)}")

@router.get("/benchmark/{test_id}")
async def benchmark_against_baseline(test_id: str) -> Dict[str, Any]:
    """Compare test results against established baselines"""
    try:
        comparison = await effectiveness_testing.benchmark_against_baseline(test_id)
        
        return {
            "success": True,
            "benchmark_comparison": comparison
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to benchmark test results: {str(e)}")

@router.post("/update-baselines")
async def update_baseline_metrics(request: BaselineUpdateRequest) -> Dict[str, Any]:
    """Update baseline metrics for comparison"""
    try:
        # Convert string keys to EffectivenessMetric enum
        new_baselines = {}
        for metric_str, value in request.baselines.items():
            try:
                metric = EffectivenessMetric(metric_str)
                new_baselines[metric] = value
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid metric: {metric_str}")
        
        effectiveness_testing.update_baseline_metrics(new_baselines)
        
        return {
            "success": True,
            "message": "Baseline metrics updated successfully",
            "updated_metrics": list(request.baselines.keys())
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update baseline metrics: {str(e)}")

@router.get("/export/{test_id}")
async def export_test_results(test_id: str) -> Dict[str, Any]:
    """Export comprehensive test results"""
    try:
        results = await effectiveness_testing.export_test_results(test_id)
        
        return {
            "success": True,
            "test_results": results
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to export test results: {str(e)}")

@router.get("/active-tests")
async def get_active_tests() -> Dict[str, Any]:
    """Get list of currently active effectiveness tests"""
    try:
        active_tests = [
            {
                "test_id": test.test_id,
                "crisis_scenario": test.crisis_scenario,
                "test_type": test.test_type,
                "start_time": test.start_time.isoformat(),
                "scores_count": len(test.effectiveness_scores)
            }
            for test in effectiveness_testing.active_tests.values()
        ]
        
        return {
            "success": True,
            "active_tests": active_tests,
            "count": len(active_tests)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get active tests: {str(e)}")

@router.get("/test-history")
async def get_test_history(limit: int = 50) -> Dict[str, Any]:
    """Get history of completed effectiveness tests"""
    try:
        history = effectiveness_testing.test_history[-limit:] if limit > 0 else effectiveness_testing.test_history
        
        test_history = [
            {
                "test_id": test.test_id,
                "crisis_scenario": test.crisis_scenario,
                "test_type": test.test_type,
                "start_time": test.start_time.isoformat(),
                "end_time": test.end_time.isoformat() if test.end_time else None,
                "overall_score": test.overall_score,
                "recommendations_count": len(test.recommendations) if test.recommendations else 0
            }
            for test in history
        ]
        
        return {
            "success": True,
            "test_history": test_history,
            "count": len(test_history)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get test history: {str(e)}")

@router.get("/metrics")
async def get_available_metrics() -> Dict[str, Any]:
    """Get list of available effectiveness metrics"""
    try:
        metrics = [
            {
                "metric": metric.value,
                "description": metric.name.replace("_", " ").title()
            }
            for metric in EffectivenessMetric
        ]
        
        return {
            "success": True,
            "available_metrics": metrics
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get available metrics: {str(e)}")

@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """Health check endpoint for crisis response effectiveness testing"""
    return {
        "status": "healthy",
        "service": "Crisis Response Effectiveness Testing",
        "active_tests": len(effectiveness_testing.active_tests),
        "completed_tests": len(effectiveness_testing.test_history),
        "timestamp": datetime.now().isoformat()
    }