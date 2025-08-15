"""
API routes for Quality Control Automation system
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging

from scrollintel.engines.quality_control_automation import QualityControlAutomation
from scrollintel.models.quality_control_models import (
    QualityAssessmentData, QualityMetricData, QualityStandardConfig,
    QualityLevelType, QualityStandardType
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/quality-control", tags=["quality-control"])

# Global quality control automation instance
quality_control = QualityControlAutomation()

@router.post("/assess")
async def assess_quality(
    process_id: str,
    process_type: str,
    process_data: Dict[str, Any]
) -> Dict[str, Any]:
    """Perform quality assessment for a process"""
    try:
        assessment = await quality_control.assess_quality(
            process_id=process_id,
            process_type=process_type,
            process_data=process_data
        )
        
        return {
            "success": True,
            "assessment": {
                "process_id": assessment.process_id,
                "process_type": assessment.process_type,
                "overall_score": assessment.overall_score,
                "quality_level": assessment.quality_level.value,
                "metrics": [
                    {
                        "name": metric.name,
                        "value": metric.value,
                        "threshold": metric.threshold,
                        "weight": metric.weight,
                        "description": metric.description,
                        "passing": metric.value >= metric.threshold
                    }
                    for metric in assessment.metrics
                ],
                "issues": assessment.issues,
                "recommendations": assessment.recommendations,
                "assessment_time": assessment.assessment_time.isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Error in quality assessment: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/enforce-standards")
async def enforce_quality_standards(
    process_id: str,
    assessment_data: Dict[str, Any]
) -> Dict[str, Any]:
    """Enforce quality standards for a process"""
    try:
        # Convert assessment data to QualityAssessmentData
        metrics = [
            QualityMetricData(
                name=m["name"],
                value=m["value"],
                threshold=m["threshold"],
                weight=m["weight"],
                description=m["description"]
            )
            for m in assessment_data.get("metrics", [])
        ]
        
        assessment = QualityAssessmentData(
            process_id=process_id,
            process_type=assessment_data["process_type"],
            overall_score=assessment_data["overall_score"],
            quality_level=QualityLevelType(assessment_data["quality_level"]),
            metrics=metrics,
            issues=assessment_data.get("issues", []),
            recommendations=assessment_data.get("recommendations", [])
        )
        
        allowed = await quality_control.enforce_quality_standards(process_id, assessment)
        
        return {
            "success": True,
            "process_allowed": allowed,
            "enforcement_actions": {
                "blocked": not allowed,
                "enhanced_monitoring": assessment.quality_level == QualityLevelType.ACCEPTABLE,
                "corrective_actions_triggered": assessment.quality_level in [
                    QualityLevelType.POOR, QualityLevelType.UNACCEPTABLE
                ]
            }
        }
        
    except Exception as e:
        logger.error(f"Error enforcing quality standards: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/start-monitoring")
async def start_continuous_monitoring(background_tasks: BackgroundTasks) -> Dict[str, Any]:
    """Start continuous quality monitoring"""
    try:
        background_tasks.add_task(quality_control.start_continuous_monitoring)
        
        return {
            "success": True,
            "message": "Continuous quality monitoring started",
            "monitoring_active": True
        }
        
    except Exception as e:
        logger.error(f"Error starting monitoring: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/stop-monitoring")
async def stop_continuous_monitoring() -> Dict[str, Any]:
    """Stop continuous quality monitoring"""
    try:
        quality_control.stop_continuous_monitoring()
        
        return {
            "success": True,
            "message": "Continuous quality monitoring stopped",
            "monitoring_active": False
        }
        
    except Exception as e:
        logger.error(f"Error stopping monitoring: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/monitoring-status")
async def get_monitoring_status() -> Dict[str, Any]:
    """Get current monitoring status"""
    try:
        return {
            "success": True,
            "monitoring_active": quality_control.monitoring_active,
            "quality_history_count": len(quality_control.quality_history),
            "improvement_suggestions_count": len(quality_control.improvement_suggestions)
        }
        
    except Exception as e:
        logger.error(f"Error getting monitoring status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/optimize-processes")
async def optimize_quality_processes() -> Dict[str, Any]:
    """Optimize quality control processes"""
    try:
        optimization_results = await quality_control.optimize_quality_processes()
        
        return {
            "success": True,
            "optimization_results": optimization_results,
            "optimizations_applied": len(optimization_results.get("threshold_adjustments", {})) > 0
        }
        
    except Exception as e:
        logger.error(f"Error optimizing quality processes: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/quality-standards")
async def get_quality_standards() -> Dict[str, Any]:
    """Get all quality standards"""
    try:
        standards = {}
        for standard_type, standard_def in quality_control.quality_standards.items():
            standards[standard_type.value] = {
                "metrics": standard_def.metrics,
                "thresholds": standard_def.thresholds,
                "weights": standard_def.weights,
                "validation_rules": standard_def.validation_rules
            }
        
        return {
            "success": True,
            "quality_standards": standards
        }
        
    except Exception as e:
        logger.error(f"Error getting quality standards: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/quality-history/{process_id}")
async def get_quality_history(process_id: str, limit: int = 10) -> Dict[str, Any]:
    """Get quality assessment history for a process"""
    try:
        history = quality_control.quality_history.get(process_id, [])
        
        # Limit results
        limited_history = history[-limit:] if len(history) > limit else history
        
        history_data = []
        for assessment in limited_history:
            history_data.append({
                "process_id": assessment.process_id,
                "process_type": assessment.process_type,
                "overall_score": assessment.overall_score,
                "quality_level": assessment.quality_level.value,
                "metrics_count": len(assessment.metrics),
                "issues_count": len(assessment.issues),
                "recommendations_count": len(assessment.recommendations),
                "assessment_time": assessment.assessment_time.isoformat()
            })
        
        return {
            "success": True,
            "process_id": process_id,
            "history": history_data,
            "total_assessments": len(history)
        }
        
    except Exception as e:
        logger.error(f"Error getting quality history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/improvement-suggestions/{process_id}")
async def get_improvement_suggestions(process_id: str) -> Dict[str, Any]:
    """Get improvement suggestions for a process"""
    try:
        suggestions = quality_control.improvement_suggestions.get(process_id, [])
        
        return {
            "success": True,
            "process_id": process_id,
            "suggestions": suggestions,
            "suggestions_count": len(suggestions)
        }
        
    except Exception as e:
        logger.error(f"Error getting improvement suggestions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/quality-metrics")
async def get_quality_metrics(
    process_type: Optional[str] = None,
    time_range: Optional[str] = "24h"
) -> Dict[str, Any]:
    """Get quality metrics summary"""
    try:
        # Calculate time range
        if time_range == "1h":
            since = datetime.now() - timedelta(hours=1)
        elif time_range == "24h":
            since = datetime.now() - timedelta(days=1)
        elif time_range == "7d":
            since = datetime.now() - timedelta(days=7)
        elif time_range == "30d":
            since = datetime.now() - timedelta(days=30)
        else:
            since = datetime.now() - timedelta(days=1)
        
        # Collect metrics from history
        all_assessments = []
        for process_history in quality_control.quality_history.values():
            for assessment in process_history:
                if assessment.assessment_time >= since:
                    if not process_type or assessment.process_type == process_type:
                        all_assessments.append(assessment)
        
        # Calculate summary metrics
        if all_assessments:
            avg_score = sum(a.overall_score for a in all_assessments) / len(all_assessments)
            quality_distribution = {}
            for assessment in all_assessments:
                level = assessment.quality_level.value
                quality_distribution[level] = quality_distribution.get(level, 0) + 1
        else:
            avg_score = 0.0
            quality_distribution = {}
        
        return {
            "success": True,
            "time_range": time_range,
            "process_type": process_type,
            "metrics": {
                "total_assessments": len(all_assessments),
                "average_score": round(avg_score, 3),
                "quality_distribution": quality_distribution
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting quality metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/validate-process")
async def validate_process_quality(
    process_id: str,
    process_type: str,
    validation_data: Dict[str, Any]
) -> Dict[str, Any]:
    """Validate process quality against standards"""
    try:
        # Perform assessment
        assessment = await quality_control.assess_quality(
            process_id=process_id,
            process_type=process_type,
            process_data=validation_data
        )
        
        # Enforce standards
        allowed = await quality_control.enforce_quality_standards(process_id, assessment)
        
        # Determine validation result
        validation_result = {
            "process_id": process_id,
            "process_type": process_type,
            "validation_passed": allowed,
            "quality_score": assessment.overall_score,
            "quality_level": assessment.quality_level.value,
            "critical_issues": [issue for issue in assessment.issues if "critical" in issue.lower()],
            "blocking_issues": assessment.issues if not allowed else [],
            "recommendations": assessment.recommendations,
            "next_steps": []
        }
        
        # Add next steps based on quality level
        if assessment.quality_level == QualityLevelType.UNACCEPTABLE:
            validation_result["next_steps"].append("Process blocked - address critical issues")
        elif assessment.quality_level == QualityLevelType.POOR:
            validation_result["next_steps"].append("Process blocked - implement improvements")
        elif assessment.quality_level == QualityLevelType.ACCEPTABLE:
            validation_result["next_steps"].append("Process allowed with enhanced monitoring")
        else:
            validation_result["next_steps"].append("Process approved - continue normally")
        
        return {
            "success": True,
            "validation_result": validation_result
        }
        
    except Exception as e:
        logger.error(f"Error validating process quality: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """Health check for quality control system"""
    try:
        return {
            "success": True,
            "status": "healthy",
            "monitoring_active": quality_control.monitoring_active,
            "quality_standards_count": len(quality_control.quality_standards),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))