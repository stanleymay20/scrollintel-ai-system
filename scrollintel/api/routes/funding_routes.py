"""
API Routes for Unlimited Funding Access System

Provides REST API endpoints for funding management, coordination,
and monitoring within the guaranteed success framework.
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, List, Optional
from datetime import datetime
from pydantic import BaseModel, Field

from ...engines.funding_engine import funding_engine, FundingRequest
from ...core.funding_coordinator import funding_coordinator, CoordinationStrategy
from ...security.auth import get_current_user

router = APIRouter(prefix="/api/funding", tags=["funding"])


# Request/Response Models
class FundingRequestModel(BaseModel):
    amount: float = Field(..., gt=0, description="Funding amount requested")
    purpose: str = Field(..., min_length=1, description="Purpose of funding")
    urgency: int = Field(5, ge=1, le=10, description="Urgency level (1-10)")
    required_by: Optional[datetime] = Field(None, description="Required completion date")


class FundingPlanModel(BaseModel):
    amount: float = Field(..., gt=0, description="Total funding amount")
    purpose: str = Field(..., min_length=1, description="Purpose of funding plan")
    strategy: CoordinationStrategy = Field(CoordinationStrategy.DIVERSIFIED, description="Coordination strategy")
    timeline_days: int = Field(30, ge=1, le=365, description="Timeline in days")


class FundingStatusResponse(BaseModel):
    total_commitment: float
    total_available: float
    total_deployed: float
    utilization_rate: float
    source_count: int
    active_requests: int
    target_achievement: float
    last_updated: str


class FundingSourceResponse(BaseModel):
    id: str
    name: str
    source_type: str
    total_commitment: float
    available_amount: float
    deployed_amount: float
    status: str
    security_level: int
    response_time_hours: int


@router.post("/initialize")
async def initialize_funding_system(current_user: dict = Depends(get_current_user)):
    """Initialize the unlimited funding access system"""
    try:
        success = await funding_engine.initialize_funding_sources()
        if success:
            # Start monitoring
            import asyncio
            asyncio.create_task(funding_engine.start_monitoring())
            
            return {
                "success": True,
                "message": "Funding system initialized successfully",
                "status": await funding_engine.get_funding_status()
            }
        else:
            raise HTTPException(
                status_code=500,
                detail="Failed to initialize funding system - insufficient commitments"
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Initialization error: {str(e)}")


@router.get("/status", response_model=FundingStatusResponse)
async def get_funding_status(current_user: dict = Depends(get_current_user)):
    """Get comprehensive funding system status"""
    try:
        status = await funding_engine.get_funding_status()
        return FundingStatusResponse(**status)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving status: {str(e)}")


@router.get("/sources")
async def get_funding_sources(current_user: dict = Depends(get_current_user)) -> List[FundingSourceResponse]:
    """Get all funding sources with their current status"""
    try:
        sources = []
        for source_id, source in funding_engine.funding_sources.items():
            sources.append(FundingSourceResponse(
                id=source.id,
                name=source.name,
                source_type=source.source_type.value,
                total_commitment=source.total_commitment,
                available_amount=source.available_amount,
                deployed_amount=source.deployed_amount,
                status=source.status.value,
                security_level=source.security_level,
                response_time_hours=source.response_time_hours
            ))
        
        return sources
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving sources: {str(e)}")


@router.get("/availability")
async def get_funding_availability(current_user: dict = Depends(get_current_user)):
    """Get real-time funding availability across all sources"""
    try:
        availability = await funding_engine.monitor_funding_availability()
        return {
            "availability_report": availability,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error monitoring availability: {str(e)}")


@router.post("/request")
async def request_funding(
    request: FundingRequestModel,
    current_user: dict = Depends(get_current_user)
):
    """Submit a funding request"""
    try:
        request_id = await funding_engine.request_funding(
            amount=request.amount,
            purpose=request.purpose,
            urgency=request.urgency,
            required_by=request.required_by
        )
        
        return {
            "success": True,
            "request_id": request_id,
            "message": f"Funding request submitted for ${request.amount:,.0f}",
            "estimated_processing": "24-72 hours" if request.urgency < 8 else "Immediate"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error submitting request: {str(e)}")


@router.get("/request/{request_id}")
async def get_funding_request(
    request_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get status of a specific funding request"""
    try:
        if request_id not in funding_engine.funding_requests:
            raise HTTPException(status_code=404, detail="Funding request not found")
        
        request = funding_engine.funding_requests[request_id]
        return {
            "id": request.id,
            "amount_requested": request.amount_requested,
            "allocated_amount": request.allocated_amount,
            "purpose": request.purpose,
            "urgency_level": request.urgency_level,
            "status": request.status,
            "required_by": request.required_by.isoformat(),
            "created_at": request.created_at.isoformat(),
            "approved_sources": request.approved_sources
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving request: {str(e)}")


@router.post("/validate/{source_id}")
async def validate_funding_source(
    source_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Validate security and availability of a funding source"""
    try:
        is_valid = await funding_engine.validate_funding_security(source_id)
        
        if source_id not in funding_engine.funding_sources:
            raise HTTPException(status_code=404, detail="Funding source not found")
        
        source = funding_engine.funding_sources[source_id]
        
        return {
            "source_id": source_id,
            "source_name": source.name,
            "is_valid": is_valid,
            "security_level": source.security_level,
            "last_validated": source.last_validated.isoformat(),
            "status": source.status.value,
            "available_amount": source.available_amount
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error validating source: {str(e)}")


@router.post("/backup/{source_id}")
async def activate_backup_sources(
    source_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Activate backup sources for a primary funding source"""
    try:
        activated_backups = await funding_engine.activate_backup_sources(source_id)
        
        return {
            "primary_source_id": source_id,
            "activated_backups": activated_backups,
            "backup_count": len(activated_backups),
            "message": f"Activated {len(activated_backups)} backup sources"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error activating backups: {str(e)}")


# Funding Coordination Endpoints

@router.post("/plan")
async def create_funding_plan(
    plan: FundingPlanModel,
    current_user: dict = Depends(get_current_user)
):
    """Create a comprehensive funding plan"""
    try:
        plan_id = await funding_coordinator.create_funding_plan(
            amount=plan.amount,
            purpose=plan.purpose,
            strategy=plan.strategy,
            timeline_days=plan.timeline_days
        )
        
        return {
            "success": True,
            "plan_id": plan_id,
            "message": f"Funding plan created for ${plan.amount:,.0f}",
            "strategy": plan.strategy.value,
            "timeline_days": plan.timeline_days
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating plan: {str(e)}")


@router.get("/plan/{plan_id}")
async def get_funding_plan(
    plan_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get details of a specific funding plan"""
    try:
        if plan_id not in funding_coordinator.active_plans:
            raise HTTPException(status_code=404, detail="Funding plan not found")
        
        plan = funding_coordinator.active_plans[plan_id]
        
        return {
            "id": plan.id,
            "total_amount": plan.total_amount,
            "purpose": plan.purpose,
            "strategy": plan.strategy.value,
            "source_allocations": plan.source_allocations,
            "backup_sources": plan.backup_sources,
            "timeline": {k: v.isoformat() for k, v in plan.timeline.items()},
            "risk_assessment": plan.risk_assessment,
            "created_at": plan.created_at.isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving plan: {str(e)}")


@router.post("/plan/{plan_id}/execute")
async def execute_funding_plan(
    plan_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Execute a funding plan"""
    try:
        success = await funding_coordinator.execute_funding_plan(plan_id)
        
        if success:
            return {
                "success": True,
                "plan_id": plan_id,
                "message": "Funding plan executed successfully"
            }
        else:
            return {
                "success": False,
                "plan_id": plan_id,
                "message": "Funding plan execution failed - check backup sources"
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error executing plan: {str(e)}")


@router.get("/coordination/status")
async def get_coordination_status(current_user: dict = Depends(get_current_user)):
    """Get funding coordination system status"""
    try:
        status = await funding_coordinator.get_coordination_status()
        return status
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving coordination status: {str(e)}")


@router.get("/plans")
async def get_all_funding_plans(current_user: dict = Depends(get_current_user)):
    """Get all active funding plans"""
    try:
        plans = []
        for plan_id, plan in funding_coordinator.active_plans.items():
            plans.append({
                "id": plan.id,
                "total_amount": plan.total_amount,
                "purpose": plan.purpose,
                "strategy": plan.strategy.value,
                "overall_risk": plan.risk_assessment["overall_risk"],
                "created_at": plan.created_at.isoformat()
            })
        
        return {
            "plans": plans,
            "total_plans": len(plans),
            "total_planned_amount": sum(plan.total_amount for plan in funding_coordinator.active_plans.values())
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving plans: {str(e)}")


# Emergency and Monitoring Endpoints

@router.post("/emergency/activate")
async def activate_emergency_funding(
    amount: float,
    purpose: str,
    current_user: dict = Depends(get_current_user)
):
    """Activate emergency funding with maximum urgency"""
    try:
        request_id = await funding_engine.request_funding(
            amount=amount,
            purpose=f"EMERGENCY: {purpose}",
            urgency=10,  # Maximum urgency
            required_by=datetime.now() + timedelta(hours=1)  # 1 hour deadline
        )
        
        # Immediately process the request
        request = funding_engine.funding_requests[request_id]
        success, sources = await funding_engine.allocate_funding(request)
        
        return {
            "success": success,
            "request_id": request_id,
            "allocated_amount": request.allocated_amount,
            "sources_used": len(sources),
            "message": "Emergency funding activated" if success else "Emergency funding partially allocated"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error activating emergency funding: {str(e)}")


@router.get("/monitoring/health")
async def get_funding_system_health(current_user: dict = Depends(get_current_user)):
    """Get comprehensive funding system health status"""
    try:
        status = await funding_engine.get_funding_status()
        availability = await funding_engine.monitor_funding_availability()
        coordination_status = await funding_coordinator.get_coordination_status()
        
        # Calculate health score
        health_score = 100.0
        
        # Deduct for low availability
        if status["target_achievement"] < 100:
            health_score -= (100 - status["target_achievement"]) * 0.5
        
        # Deduct for high utilization
        if status["utilization_rate"] > 80:
            health_score -= (status["utilization_rate"] - 80) * 0.5
        
        # Deduct for high risk
        if coordination_status["average_risk_level"] > 0.5:
            health_score -= (coordination_status["average_risk_level"] - 0.5) * 20
        
        health_status = "EXCELLENT" if health_score >= 90 else \
                       "GOOD" if health_score >= 75 else \
                       "WARNING" if health_score >= 60 else "CRITICAL"
        
        return {
            "health_score": max(0, health_score),
            "health_status": health_status,
            "funding_status": status,
            "availability": availability["TOTAL_AVAILABLE"],
            "coordination_status": coordination_status,
            "monitoring_active": funding_engine.monitoring_active,
            "last_updated": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving health status: {str(e)}")