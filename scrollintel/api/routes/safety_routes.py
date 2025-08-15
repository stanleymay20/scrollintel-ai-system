"""
API routes for AI Safety Framework monitoring and control
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, List, Optional
from pydantic import BaseModel
import logging

from scrollintel.core.ai_safety_framework import (
    ai_safety_framework,
    SafetyLevel,
    AlignmentStatus,
    EthicalConstraint,
    SafetyViolation,
    HumanOverseer
)
from scrollintel.core.safety_middleware import safety_middleware

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/safety", tags=["ai-safety"])


class SafetyStatusResponse(BaseModel):
    safety_active: bool
    shutdown_active: bool
    total_constraints: int
    active_constraints: int
    total_violations: int
    unresolved_violations: int
    pending_approvals: int
    human_overseers: int
    alignment_checks: int
    last_alignment_check: Optional[str]


class OperationValidationRequest(BaseModel):
    operation_type: str
    operation_data: Dict
    safety_level: str = "medium"
    user_id: str = "system"


class EmergencyShutdownRequest(BaseModel):
    reason: str
    authorized_user: str
    confirmation_code: str


class ConstraintUpdateRequest(BaseModel):
    constraint_id: str
    active: bool
    severity: Optional[str] = None


@router.get("/status", response_model=SafetyStatusResponse)
async def get_safety_status():
    """
    Get comprehensive AI safety system status
    """
    try:
        status = await ai_safety_framework.get_safety_status()
        
        return SafetyStatusResponse(
            safety_active=status["safety_active"],
            shutdown_active=status["shutdown_active"],
            total_constraints=status["total_constraints"],
            active_constraints=status["active_constraints"],
            total_violations=status["total_violations"],
            unresolved_violations=status["unresolved_violations"],
            pending_approvals=status["pending_approvals"],
            human_overseers=status["human_overseers"],
            alignment_checks=status["alignment_checks"],
            last_alignment_check=status["last_alignment_check"].isoformat() if status["last_alignment_check"] else None
        )
        
    except Exception as e:
        logger.error(f"Error getting safety status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/constraints")
async def get_ethical_constraints():
    """
    Get all ethical constraints and their status
    """
    try:
        constraints = []
        
        for constraint_id, constraint in ai_safety_framework.ethical_constraints.items():
            constraints.append({
                "id": constraint.id,
                "name": constraint.name,
                "description": constraint.description,
                "constraint_type": constraint.constraint_type,
                "severity": constraint.severity.value,
                "active": constraint.active,
                "created_at": constraint.created_at.isoformat()
            })
        
        return {
            "success": True,
            "message": f"Retrieved {len(constraints)} ethical constraints",
            "data": {
                "constraints": constraints,
                "total_count": len(constraints),
                "active_count": len([c for c in constraints if c["active"]])
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting ethical constraints: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/violations")
async def get_safety_violations():
    """
    Get all safety violations
    """
    try:
        violations = []
        
        for violation_id, violation in ai_safety_framework.safety_violations.items():
            violations.append({
                "id": violation.id,
                "constraint_id": violation.constraint_id,
                "violation_type": violation.violation_type,
                "severity": violation.severity.value,
                "description": violation.description,
                "resolved": violation.resolved,
                "human_notified": violation.human_notified,
                "timestamp": violation.timestamp.isoformat()
            })
        
        # Sort by timestamp (most recent first)
        violations.sort(key=lambda x: x["timestamp"], reverse=True)
        
        return {
            "success": True,
            "message": f"Retrieved {len(violations)} safety violations",
            "data": {
                "violations": violations,
                "total_count": len(violations),
                "unresolved_count": len([v for v in violations if not v["resolved"]]),
                "critical_count": len([v for v in violations if v["severity"] in ["critical", "existential"]])
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting safety violations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/alignment-checks")
async def get_alignment_checks():
    """
    Get AI alignment verification checks
    """
    try:
        checks = []
        
        for check_id, check in ai_safety_framework.alignment_verifier.alignment_checks.items():
            checks.append({
                "id": check.id,
                "check_type": check.check_type,
                "description": check.description,
                "status": check.status.value,
                "confidence_score": check.confidence_score,
                "human_verified": check.human_verified,
                "timestamp": check.timestamp.isoformat()
            })
        
        # Sort by timestamp (most recent first)
        checks.sort(key=lambda x: x["timestamp"], reverse=True)
        
        # Calculate alignment statistics
        total_checks = len(checks)
        aligned_count = len([c for c in checks if c["status"] == "aligned"])
        misaligned_count = len([c for c in checks if c["status"] == "misaligned"])
        uncertain_count = len([c for c in checks if c["status"] == "uncertain"])
        
        avg_confidence = sum(c["confidence_score"] for c in checks) / total_checks if total_checks > 0 else 0
        
        return {
            "success": True,
            "message": f"Retrieved {len(checks)} alignment checks",
            "data": {
                "checks": checks[:50],  # Return last 50 checks
                "statistics": {
                    "total_checks": total_checks,
                    "aligned_count": aligned_count,
                    "misaligned_count": misaligned_count,
                    "uncertain_count": uncertain_count,
                    "alignment_rate": (aligned_count / total_checks * 100) if total_checks > 0 else 0,
                    "average_confidence": avg_confidence
                }
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting alignment checks: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/human-overseers")
async def get_human_overseers():
    """
    Get human oversight personnel and their status
    """
    try:
        overseers = []
        
        for overseer_id, overseer in ai_safety_framework.oversight_manager.overseers.items():
            overseers.append({
                "id": overseer.id,
                "name": overseer.name,
                "role": overseer.role,
                "clearance_level": overseer.clearance_level.value,
                "permissions": overseer.permissions,
                "active": overseer.active,
                "last_active": overseer.last_active.isoformat()
            })
        
        return {
            "success": True,
            "message": f"Retrieved {len(overseers)} human overseers",
            "data": {
                "overseers": overseers,
                "total_count": len(overseers),
                "active_count": len([o for o in overseers if o["active"]])
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting human overseers: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/pending-approvals")
async def get_pending_approvals():
    """
    Get operations pending human approval
    """
    try:
        approvals = []
        
        for approval_id, approval in ai_safety_framework.oversight_manager.pending_approvals.items():
            approvals.append({
                "id": approval["id"],
                "operation_type": approval["operation"].get("operation_type", "unknown"),
                "required_level": approval["required_level"],
                "assigned_overseer": approval["assigned_overseer"],
                "status": approval["status"],
                "created_at": approval["created_at"],
                "timeout_at": approval["timeout_at"]
            })
        
        return {
            "success": True,
            "message": f"Retrieved {len(approvals)} pending approvals",
            "data": {
                "approvals": approvals,
                "total_count": len(approvals)
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting pending approvals: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/validate-operation")
async def validate_operation(request: OperationValidationRequest):
    """
    Validate an operation against safety constraints
    """
    try:
        operation_context = {
            "operation_type": request.operation_type,
            "operation_data": request.operation_data,
            "safety_level": request.safety_level,
            "user_id": request.user_id
        }
        
        validation_result = await ai_safety_framework.validate_operation(operation_context)
        
        return {
            "success": True,
            "message": "Operation validation completed",
            "data": {
                "allowed": validation_result["allowed"],
                "violations": [
                    {
                        "id": v.id,
                        "description": v.description,
                        "severity": v.severity.value,
                        "constraint_id": v.constraint_id
                    } for v in validation_result["violations"]
                ],
                "warnings": validation_result["warnings"],
                "required_approvals": validation_result["required_approvals"],
                "alignment_check": {
                    "status": validation_result["alignment_check"].status.value,
                    "confidence_score": validation_result["alignment_check"].confidence_score
                } if validation_result["alignment_check"] else None
            }
        }
        
    except Exception as e:
        logger.error(f"Error validating operation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/emergency-shutdown")
async def emergency_shutdown(request: EmergencyShutdownRequest):
    """
    Trigger emergency shutdown of AI systems
    """
    try:
        # Verify confirmation code (in real implementation, this would be more secure)
        expected_code = "EMERGENCY_SHUTDOWN_CONFIRMED"
        if request.confirmation_code != expected_code:
            raise HTTPException(status_code=403, detail="Invalid confirmation code")
        
        success = await ai_safety_framework.emergency_shutdown(
            request.reason,
            request.authorized_user
        )
        
        if success:
            return {
                "success": True,
                "message": "Emergency shutdown initiated successfully",
                "data": {
                    "shutdown_active": True,
                    "reason": request.reason,
                    "authorized_by": request.authorized_user,
                    "timestamp": ai_safety_framework.shutdown_system.shutdown_timestamp.isoformat()
                }
            }
        else:
            raise HTTPException(status_code=500, detail="Emergency shutdown failed")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in emergency shutdown: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/update-constraint")
async def update_constraint(request: ConstraintUpdateRequest):
    """
    Update ethical constraint settings
    """
    try:
        if request.constraint_id not in ai_safety_framework.ethical_constraints:
            raise HTTPException(status_code=404, detail="Constraint not found")
        
        constraint = ai_safety_framework.ethical_constraints[request.constraint_id]
        constraint.active = request.active
        
        if request.severity:
            constraint.severity = SafetyLevel(request.severity)
        
        return {
            "success": True,
            "message": f"Constraint {request.constraint_id} updated successfully",
            "data": {
                "constraint_id": constraint.id,
                "name": constraint.name,
                "active": constraint.active,
                "severity": constraint.severity.value
            }
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid severity level: {request.severity}")
    except Exception as e:
        logger.error(f"Error updating constraint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/middleware-stats")
async def get_middleware_statistics():
    """
    Get safety middleware statistics
    """
    try:
        stats = safety_middleware.get_safety_statistics()
        
        return {
            "success": True,
            "message": "Retrieved safety middleware statistics",
            "data": stats
        }
        
    except Exception as e:
        logger.error(f"Error getting middleware statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/enable-safety")
async def enable_safety_framework():
    """
    Enable the AI safety framework
    """
    try:
        ai_safety_framework.safety_active = True
        safety_middleware.enabled = True
        
        return {
            "success": True,
            "message": "AI safety framework enabled",
            "data": {
                "safety_active": True,
                "middleware_enabled": True
            }
        }
        
    except Exception as e:
        logger.error(f"Error enabling safety framework: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/disable-safety")
async def disable_safety_framework():
    """
    Disable the AI safety framework (DANGEROUS - requires authorization)
    """
    try:
        # This should require very high authorization in real implementation
        logger.critical("WARNING: AI safety framework disabled - system is now unsafe!")
        
        ai_safety_framework.safety_active = False
        safety_middleware.enabled = False
        
        return {
            "success": True,
            "message": "AI safety framework disabled - SYSTEM IS NOW UNSAFE",
            "data": {
                "safety_active": False,
                "middleware_enabled": False,
                "warning": "All safety constraints have been removed"
            }
        }
        
    except Exception as e:
        logger.error(f"Error disabling safety framework: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/safety-report")
async def generate_safety_report():
    """
    Generate comprehensive safety report
    """
    try:
        # Get all safety data
        status = await ai_safety_framework.get_safety_status()
        middleware_stats = safety_middleware.get_safety_statistics()
        
        # Calculate risk assessment
        risk_score = 0
        risk_factors = []
        
        if not status["safety_active"]:
            risk_score += 100
            risk_factors.append("Safety framework disabled")
        
        if status["shutdown_active"]:
            risk_score += 50
            risk_factors.append("Emergency shutdown active")
        
        if status["unresolved_violations"] > 0:
            risk_score += status["unresolved_violations"] * 10
            risk_factors.append(f"{status['unresolved_violations']} unresolved safety violations")
        
        if middleware_stats["block_rate"] > 20:
            risk_score += 20
            risk_factors.append("High operation block rate")
        
        # Determine risk level
        if risk_score >= 100:
            risk_level = "CRITICAL"
        elif risk_score >= 50:
            risk_level = "HIGH"
        elif risk_score >= 20:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        
        return {
            "success": True,
            "message": "Safety report generated",
            "data": {
                "report_timestamp": datetime.now().isoformat(),
                "overall_risk_level": risk_level,
                "risk_score": risk_score,
                "risk_factors": risk_factors,
                "safety_status": status,
                "middleware_statistics": middleware_stats,
                "recommendations": [
                    "Ensure safety framework remains active at all times",
                    "Regularly review and resolve safety violations",
                    "Maintain human oversight for critical operations",
                    "Monitor alignment checks for concerning patterns",
                    "Keep emergency shutdown procedures accessible"
                ]
            }
        }
        
    except Exception as e:
        logger.error(f"Error generating safety report: {e}")
        raise HTTPException(status_code=500, detail=str(e))