"""
API routes for Error Detection and Correction system
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging

from scrollintel.engines.error_detection_correction import (
    ErrorDetectionCorrection, DetectedError, CorrectionResult,
    ErrorType, ErrorSeverity, CorrectionStatus
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/error-detection", tags=["error-detection"])

# Global error detection and correction instance
error_system = ErrorDetectionCorrection()

@router.post("/detect")
async def detect_errors(
    process_id: str,
    process_type: str,
    process_data: Dict[str, Any]
) -> Dict[str, Any]:
    """Detect errors in a process"""
    try:
        detected_errors = await error_system.detect_errors(
            process_id=process_id,
            process_type=process_type,
            process_data=process_data
        )
        
        return {
            "success": True,
            "process_id": process_id,
            "errors_detected": len(detected_errors),
            "errors": [error.to_dict() for error in detected_errors],
            "detection_time": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in error detection: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/correct")
async def correct_errors(
    error_data: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Correct detected errors"""
    try:
        # Convert error data to DetectedError objects
        errors = []
        for error_dict in error_data:
            error = DetectedError(
                error_id=error_dict["error_id"],
                error_type=ErrorType(error_dict["error_type"]),
                severity=ErrorSeverity(error_dict["severity"]),
                process_id=error_dict["process_id"],
                process_type=error_dict["process_type"],
                error_message=error_dict["error_message"],
                error_context=error_dict.get("error_context", {}),
                stack_trace=error_dict.get("stack_trace"),
                affected_components=error_dict.get("affected_components", [])
            )
            errors.append(error)
        
        correction_results = await error_system.correct_errors(errors)
        
        return {
            "success": True,
            "errors_processed": len(errors),
            "corrections_attempted": len(correction_results),
            "corrections": [result.to_dict() for result in correction_results],
            "correction_time": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in error correction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/detect-and-correct")
async def detect_and_correct_errors(
    process_id: str,
    process_type: str,
    process_data: Dict[str, Any]
) -> Dict[str, Any]:
    """Detect and correct errors in one operation"""
    try:
        # Detect errors
        detected_errors = await error_system.detect_errors(
            process_id=process_id,
            process_type=process_type,
            process_data=process_data
        )
        
        # Correct errors if any found
        correction_results = []
        if detected_errors:
            correction_results = await error_system.correct_errors(detected_errors)
        
        return {
            "success": True,
            "process_id": process_id,
            "errors_detected": len(detected_errors),
            "corrections_attempted": len(correction_results),
            "errors": [error.to_dict() for error in detected_errors],
            "corrections": [result.to_dict() for result in correction_results],
            "processing_time": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in detect and correct: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/prevent")
async def prevent_errors(
    process_id: str,
    process_type: str,
    process_data: Dict[str, Any]
) -> Dict[str, Any]:
    """Prevent errors based on learned patterns"""
    try:
        prevention_result = await error_system.prevent_errors(
            process_id=process_id,
            process_type=process_type,
            process_data=process_data
        )
        
        return {
            "success": True,
            "prevention_result": prevention_result,
            "prevention_time": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in error prevention: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/learn")
async def learn_from_errors() -> Dict[str, Any]:
    """Learn from error history to improve detection and correction"""
    try:
        learning_results = await error_system.learn_from_errors()
        
        return {
            "success": True,
            "learning_results": learning_results,
            "learning_time": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in learning from errors: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/start-monitoring")
async def start_continuous_monitoring(background_tasks: BackgroundTasks) -> Dict[str, Any]:
    """Start continuous error monitoring"""
    try:
        background_tasks.add_task(error_system.start_continuous_monitoring)
        
        return {
            "success": True,
            "message": "Continuous error monitoring started",
            "monitoring_active": True
        }
        
    except Exception as e:
        logger.error(f"Error starting monitoring: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/stop-monitoring")
async def stop_continuous_monitoring() -> Dict[str, Any]:
    """Stop continuous error monitoring"""
    try:
        error_system.stop_continuous_monitoring()
        
        return {
            "success": True,
            "message": "Continuous error monitoring stopped",
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
            "monitoring_active": error_system.monitoring_active,
            "detected_errors_count": len(error_system.detected_errors),
            "error_patterns_count": len(error_system.error_patterns),
            "prevention_rules_count": len(error_system.prevention_rules)
        }
        
    except Exception as e:
        logger.error(f"Error getting monitoring status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/error-history")
async def get_error_history(
    process_id: Optional[str] = None,
    error_type: Optional[str] = None,
    limit: int = 50
) -> Dict[str, Any]:
    """Get error detection history"""
    try:
        errors = []
        
        for error_id, error in error_system.detected_errors.items():
            # Filter by process_id if specified
            if process_id and error.process_id != process_id:
                continue
            
            # Filter by error_type if specified
            if error_type and error.error_type.value != error_type:
                continue
            
            errors.append(error.to_dict())
        
        # Sort by detection time (most recent first)
        errors.sort(key=lambda x: x["detection_time"], reverse=True)
        
        # Limit results
        limited_errors = errors[:limit]
        
        return {
            "success": True,
            "errors": limited_errors,
            "total_errors": len(errors),
            "filters": {
                "process_id": process_id,
                "error_type": error_type,
                "limit": limit
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting error history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/correction-history")
async def get_correction_history(
    error_id: Optional[str] = None,
    limit: int = 50
) -> Dict[str, Any]:
    """Get error correction history"""
    try:
        corrections = []
        
        for err_id, correction_list in error_system.correction_history.items():
            # Filter by error_id if specified
            if error_id and err_id != error_id:
                continue
            
            for correction in correction_list:
                corrections.append(correction.to_dict())
        
        # Sort by correction time (most recent first)
        corrections.sort(key=lambda x: x["correction_time"], reverse=True)
        
        # Limit results
        limited_corrections = corrections[:limit]
        
        return {
            "success": True,
            "corrections": limited_corrections,
            "total_corrections": len(corrections),
            "filters": {
                "error_id": error_id,
                "limit": limit
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting correction history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/error-patterns")
async def get_error_patterns() -> Dict[str, Any]:
    """Get learned error patterns"""
    try:
        patterns = []
        
        for pattern_id, pattern in error_system.error_patterns.items():
            patterns.append(pattern.to_dict())
        
        # Sort by occurrence count (most frequent first)
        patterns.sort(key=lambda x: x["occurrence_count"], reverse=True)
        
        return {
            "success": True,
            "patterns": patterns,
            "total_patterns": len(patterns)
        }
        
    except Exception as e:
        logger.error(f"Error getting error patterns: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/prevention-rules")
async def get_prevention_rules() -> Dict[str, Any]:
    """Get current prevention rules"""
    try:
        return {
            "success": True,
            "prevention_rules": error_system.prevention_rules,
            "rules_count": len(error_system.prevention_rules)
        }
        
    except Exception as e:
        logger.error(f"Error getting prevention rules: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/error-statistics")
async def get_error_statistics(
    time_range: Optional[str] = "24h"
) -> Dict[str, Any]:
    """Get error statistics"""
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
        
        # Collect statistics
        total_errors = 0
        error_types = {}
        severity_distribution = {}
        process_types = {}
        
        for error in error_system.detected_errors.values():
            if error.detection_time >= since:
                total_errors += 1
                
                # Count by error type
                error_type = error.error_type.value
                error_types[error_type] = error_types.get(error_type, 0) + 1
                
                # Count by severity
                severity = error.severity.value
                severity_distribution[severity] = severity_distribution.get(severity, 0) + 1
                
                # Count by process type
                process_type = error.process_type
                process_types[process_type] = process_types.get(process_type, 0) + 1
        
        # Calculate correction success rate
        total_corrections = 0
        successful_corrections = 0
        
        for correction_list in error_system.correction_history.values():
            for correction in correction_list:
                if correction.correction_time >= since:
                    total_corrections += 1
                    if correction.success:
                        successful_corrections += 1
        
        success_rate = (successful_corrections / total_corrections * 100) if total_corrections > 0 else 0
        
        return {
            "success": True,
            "time_range": time_range,
            "statistics": {
                "total_errors": total_errors,
                "error_types": error_types,
                "severity_distribution": severity_distribution,
                "process_types": process_types,
                "total_corrections": total_corrections,
                "successful_corrections": successful_corrections,
                "correction_success_rate": round(success_rate, 2)
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting error statistics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """Health check for error detection system"""
    try:
        return {
            "success": True,
            "status": "healthy",
            "monitoring_active": error_system.monitoring_active,
            "error_detectors_count": sum(len(detectors) for detectors in error_system.error_detectors.values()),
            "correction_strategies_count": len(error_system.correction_strategies),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))