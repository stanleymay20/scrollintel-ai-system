"""
Innovation Lab Integration API Routes

This module provides REST API endpoints for managing integrations between
the Autonomous Innovation Lab and other ScrollIntel systems.
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime
import logging

from scrollintel.engines.innovation_lab_integration import (
    InnovationLabIntegrationEngine,
    IntegrationType,
    SynergyLevel,
    InnovationCrossPollination,
    SystemIntegrationPoint,
    InnovationSynergy
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/innovation-lab-integration", tags=["Innovation Lab Integration"])

# Global integration engine instance
integration_engine = InnovationLabIntegrationEngine()


# Pydantic models for API
class IntegrationConfigRequest(BaseModel):
    """Request model for integration configuration"""
    breakthrough_innovation: Optional[Dict[str, Any]] = Field(None, description="Breakthrough innovation system config")
    quantum_ai_research: Optional[Dict[str, Any]] = Field(None, description="Quantum AI research system config")


class InnovationRequest(BaseModel):
    """Request model for innovation processing"""
    id: str = Field(..., description="Innovation ID")
    name: str = Field(..., description="Innovation name")
    description: str = Field(..., description="Innovation description")
    domain: str = Field(..., description="Innovation domain")
    complexity: float = Field(0.5, ge=0.0, le=1.0, description="Innovation complexity")
    novelty: float = Field(0.5, ge=0.0, le=1.0, description="Innovation novelty")
    potential: float = Field(0.5, ge=0.0, le=1.0, description="Innovation potential")
    domains: List[str] = Field(default_factory=list, description="Related domains")
    capabilities: List[str] = Field(default_factory=list, description="Innovation capabilities")
    computational_bottlenecks: List[str] = Field(default_factory=list, description="Computational bottlenecks")


class CrossPollinationResponse(BaseModel):
    """Response model for cross-pollination results"""
    id: str
    source_system: str
    target_system: str
    innovation_concept: str
    synergy_level: str
    enhancement_potential: float
    implementation_complexity: float
    expected_impact: float
    created_at: datetime


class SynergyResponse(BaseModel):
    """Response model for innovation synergy"""
    id: str
    innovation_ids: List[str]
    synergy_type: str
    synergy_description: str
    combined_potential: float
    exploitation_strategy: str
    resource_requirements: Dict[str, Any]


class IntegrationStatusResponse(BaseModel):
    """Response model for integration status"""
    integration_status: Dict[str, str]
    breakthrough_integrations: int
    quantum_integrations: int
    active_cross_pollinations: int
    quantum_enhanced_innovations: int


@router.post("/initialize", response_model=Dict[str, Any])
async def initialize_integrations(
    config: IntegrationConfigRequest,
    background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    """Initialize all system integrations"""
    try:
        config_dict = config.dict(exclude_none=True)
        
        # Initialize integrations in background
        integrations = await integration_engine.initialize_all_integrations(config_dict)
        
        return {
            "status": "success",
            "message": f"Successfully initialized {len(integrations)} integrations",
            "integrations": {k: v.dict() if hasattr(v, 'dict') else str(v) for k, v in integrations.items()}
        }
        
    except Exception as e:
        logger.error(f"Failed to initialize integrations: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Integration initialization failed: {str(e)}")


@router.post("/breakthrough/cross-pollinate", response_model=CrossPollinationResponse)
async def create_cross_pollination(
    innovation: InnovationRequest
) -> CrossPollinationResponse:
    """Create cross-pollination with breakthrough innovation systems"""
    try:
        innovation_dict = innovation.dict()
        
        # Get breakthrough integration point
        if not integration_engine.breakthrough_integrator.integration_points:
            raise HTTPException(status_code=400, detail="Breakthrough innovation integration not initialized")
        
        breakthrough_system = list(integration_engine.breakthrough_integrator.integration_points.values())[0]
        
        # Implement cross-pollination
        cross_pollination = await integration_engine.breakthrough_integrator.implement_innovation_cross_pollination(
            innovation_dict, breakthrough_system
        )
        
        return CrossPollinationResponse(
            id=cross_pollination.id,
            source_system=cross_pollination.source_system,
            target_system=cross_pollination.target_system,
            innovation_concept=cross_pollination.innovation_concept,
            synergy_level=cross_pollination.synergy_level.value,
            enhancement_potential=cross_pollination.enhancement_potential,
            implementation_complexity=cross_pollination.implementation_complexity,
            expected_impact=cross_pollination.expected_impact,
            created_at=cross_pollination.created_at
        )
        
    except Exception as e:
        logger.error(f"Failed to create cross-pollination: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Cross-pollination failed: {str(e)}")


@router.post("/breakthrough/synergies", response_model=List[SynergyResponse])
async def identify_innovation_synergies(
    innovations: List[InnovationRequest]
) -> List[SynergyResponse]:
    """Identify synergies between innovations"""
    try:
        innovations_dict = [innovation.dict() for innovation in innovations]
        
        # Identify synergies
        synergies = await integration_engine.breakthrough_integrator.build_innovation_synergy_identification(
            innovations_dict
        )
        
        return [
            SynergyResponse(
                id=synergy.id,
                innovation_ids=synergy.innovation_ids,
                synergy_type=synergy.synergy_type,
                synergy_description=synergy.synergy_description,
                combined_potential=synergy.combined_potential,
                exploitation_strategy=synergy.exploitation_strategy,
                resource_requirements=synergy.resource_requirements
            )
            for synergy in synergies
        ]
        
    except Exception as e:
        logger.error(f"Failed to identify synergies: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Synergy identification failed: {str(e)}")


@router.post("/quantum/enhance", response_model=Dict[str, Any])
async def quantum_enhance_innovation(
    innovation: InnovationRequest
) -> Dict[str, Any]:
    """Enhance innovation with quantum AI capabilities"""
    try:
        innovation_dict = innovation.dict()
        
        # Get quantum integration point
        if not integration_engine.quantum_integrator.quantum_integration_points:
            raise HTTPException(status_code=400, detail="Quantum AI integration not initialized")
        
        quantum_system = list(integration_engine.quantum_integrator.quantum_integration_points.values())[0]
        
        # Apply quantum enhancement
        enhanced_innovation = await integration_engine.quantum_integrator.build_quantum_enhanced_innovation_research(
            innovation_dict, quantum_system
        )
        
        return {
            "status": "success",
            "enhanced_innovation": enhanced_innovation,
            "quantum_enhanced": enhanced_innovation.get("quantum_enhanced", False),
            "expected_speedup": enhanced_innovation.get("expected_speedup", 1.0)
        }
        
    except Exception as e:
        logger.error(f"Failed to quantum enhance innovation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Quantum enhancement failed: {str(e)}")


@router.post("/quantum/accelerate", response_model=List[Dict[str, Any]])
async def quantum_accelerate_innovations(
    innovations: List[InnovationRequest]
) -> List[Dict[str, Any]]:
    """Accelerate innovations using quantum AI"""
    try:
        innovations_dict = [innovation.dict() for innovation in innovations]
        
        # Get quantum integration point
        if not integration_engine.quantum_integrator.quantum_integration_points:
            raise HTTPException(status_code=400, detail="Quantum AI integration not initialized")
        
        quantum_system = list(integration_engine.quantum_integrator.quantum_integration_points.values())[0]
        
        # Apply quantum acceleration
        accelerated_innovations = await integration_engine.quantum_integrator.implement_quantum_innovation_acceleration(
            innovations_dict, quantum_system
        )
        
        return accelerated_innovations
        
    except Exception as e:
        logger.error(f"Failed to quantum accelerate innovations: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Quantum acceleration failed: {str(e)}")


@router.post("/process", response_model=Dict[str, Any])
async def process_innovation_with_all_integrations(
    innovation: InnovationRequest
) -> Dict[str, Any]:
    """Process innovation through all available integrations"""
    try:
        innovation_dict = innovation.dict()
        
        # Process through all integrations
        processed_innovation = await integration_engine.process_innovation_with_all_integrations(innovation_dict)
        
        return {
            "status": "success",
            "processed_innovation": processed_innovation,
            "integrations_applied": [
                key for key in ["cross_pollination", "quantum_enhanced"] 
                if key in processed_innovation
            ]
        }
        
    except Exception as e:
        logger.error(f"Failed to process innovation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Innovation processing failed: {str(e)}")


@router.get("/status", response_model=IntegrationStatusResponse)
async def get_integration_status() -> IntegrationStatusResponse:
    """Get status of all integrations"""
    try:
        status = await integration_engine.get_integration_status()
        
        return IntegrationStatusResponse(
            integration_status=status["integration_status"],
            breakthrough_integrations=status["breakthrough_integrations"],
            quantum_integrations=status["quantum_integrations"],
            active_cross_pollinations=status["active_cross_pollinations"],
            quantum_enhanced_innovations=status["quantum_enhanced_innovations"]
        )
        
    except Exception as e:
        logger.error(f"Failed to get integration status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Status retrieval failed: {str(e)}")


@router.get("/breakthrough/cross-pollinations", response_model=List[CrossPollinationResponse])
async def get_active_cross_pollinations() -> List[CrossPollinationResponse]:
    """Get all active cross-pollinations"""
    try:
        cross_pollinations = integration_engine.breakthrough_integrator.active_cross_pollinations
        
        return [
            CrossPollinationResponse(
                id=cp.id,
                source_system=cp.source_system,
                target_system=cp.target_system,
                innovation_concept=cp.innovation_concept,
                synergy_level=cp.synergy_level.value,
                enhancement_potential=cp.enhancement_potential,
                implementation_complexity=cp.implementation_complexity,
                expected_impact=cp.expected_impact,
                created_at=cp.created_at
            )
            for cp in cross_pollinations
        ]
        
    except Exception as e:
        logger.error(f"Failed to get cross-pollinations: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Cross-pollination retrieval failed: {str(e)}")


@router.get("/quantum/enhanced-innovations", response_model=List[Dict[str, Any]])
async def get_quantum_enhanced_innovations() -> List[Dict[str, Any]]:
    """Get all quantum-enhanced innovations"""
    try:
        enhanced_innovations = integration_engine.quantum_integrator.quantum_enhanced_innovations
        
        return enhanced_innovations
        
    except Exception as e:
        logger.error(f"Failed to get quantum-enhanced innovations: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Quantum innovation retrieval failed: {str(e)}")


@router.delete("/reset")
async def reset_integrations() -> Dict[str, str]:
    """Reset all integrations"""
    try:
        # Clear all integration data
        integration_engine.breakthrough_integrator.integration_points.clear()
        integration_engine.breakthrough_integrator.active_cross_pollinations.clear()
        integration_engine.breakthrough_integrator.synergy_cache.clear()
        
        integration_engine.quantum_integrator.quantum_integration_points.clear()
        integration_engine.quantum_integrator.quantum_enhanced_innovations.clear()
        
        integration_engine.integration_status.clear()
        
        return {"status": "success", "message": "All integrations reset successfully"}
        
    except Exception as e:
        logger.error(f"Failed to reset integrations: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Integration reset failed: {str(e)}")


# Health check endpoint
@router.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check for integration system"""
    return {
        "status": "healthy",
        "service": "innovation_lab_integration",
        "timestamp": datetime.now().isoformat()
    }