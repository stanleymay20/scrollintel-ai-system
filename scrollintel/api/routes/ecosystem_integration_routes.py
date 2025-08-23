"""
API Routes for Ecosystem Integration Engine

Provides endpoints for integrating partnerships with influence networks
and managing competitive ecosystem development.
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

from ...engines.ecosystem_integration_engine import EcosystemIntegrationEngine
from ...engines.influence_mapping_engine import InfluenceMappingEngine
from ...engines.partnership_analysis_engine import PartnershipAnalysisEngine
from ...models.ecosystem_models import PartnershipOpportunity
from ...models.influence_network_models import InfluenceNetwork

router = APIRouter(prefix="/api/ecosystem-integration", tags=["ecosystem-integration"])
logger = logging.getLogger(__name__)

# Initialize engines
ecosystem_engine = EcosystemIntegrationEngine()
influence_engine = InfluenceMappingEngine()
partnership_engine = PartnershipAnalysisEngine()


@router.post("/integrate-partnerships")
async def integrate_partnerships_with_network(
    network_id: str,
    partnership_ids: List[str],
    integration_strategy: Dict[str, Any]
):
    """Integrate partnership opportunities with influence network"""
    try:
        # Get network (would typically fetch from database)
        if network_id not in influence_engine.networks:
            raise HTTPException(status_code=404, detail="Network not found")
        
        network = influence_engine.networks[network_id]
        
        # Get partnerships (would typically fetch from database)
        partnerships = []
        for partnership_id in partnership_ids:
            if partnership_id in partnership_engine.opportunities:
                partnerships.append(partnership_engine.opportunities[partnership_id])
        
        if not partnerships:
            raise HTTPException(status_code=404, detail="No valid partnerships found")
        
        # Create integration
        integration = await ecosystem_engine.integrate_partnership_with_influence_network(
            network=network,
            partnerships=partnerships,
            integration_strategy=integration_strategy
        )
        
        return {
            "integration_id": integration.integration_id,
            "network_id": integration.network_id,
            "partnership_count": len(integration.partnership_opportunities),
            "influence_amplification": integration.influence_amplification,
            "competitive_advantages": integration.competitive_advantages,
            "ecosystem_growth_potential": integration.ecosystem_growth_potential,
            "created_at": integration.created_at.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error integrating partnerships with network: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/orchestrate-network-effects")
async def orchestrate_network_effects(
    integration_id: str,
    orchestration_config: Dict[str, Any]
):
    """Orchestrate network effects across partnerships and influence"""
    try:
        # Get integration
        if integration_id not in ecosystem_engine.ecosystem_integrations:
            raise HTTPException(status_code=404, detail="Integration not found")
        
        integration = ecosystem_engine.ecosystem_integrations[integration_id]
        
        # Create orchestration
        orchestration = await ecosystem_engine.orchestrate_network_effects(
            integration=integration,
            orchestration_config=orchestration_config
        )
        
        return {
            "orchestration_id": orchestration.orchestration_id,
            "network_id": orchestration.network_id,
            "partnership_synergies": orchestration.partnership_synergies,
            "influence_multipliers": orchestration.influence_multipliers,
            "leverage_points": len(orchestration.ecosystem_leverage_points),
            "competitive_moats": orchestration.competitive_moats,
            "expected_outcomes": orchestration.expected_outcomes
        }
        
    except Exception as e:
        logger.error(f"Error orchestrating network effects: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/competitive-ecosystem-map")
async def create_competitive_ecosystem_map(
    industry: str,
    network_id: str,
    partnership_ids: List[str],
    competitor_data: Dict[str, Any]
):
    """Create comprehensive competitive ecosystem mapping"""
    try:
        # Get network
        if network_id not in influence_engine.networks:
            raise HTTPException(status_code=404, detail="Network not found")
        
        network = influence_engine.networks[network_id]
        
        # Get partnerships
        partnerships = []
        for partnership_id in partnership_ids:
            if partnership_id in partnership_engine.opportunities:
                partnerships.append(partnership_engine.opportunities[partnership_id])
        
        # Create competitive map
        competitive_map = await ecosystem_engine.create_competitive_ecosystem_map(
            industry=industry,
            our_network=network,
            our_partnerships=partnerships,
            competitor_data=competitor_data
        )
        
        return {
            "map_id": competitive_map.map_id,
            "industry": competitive_map.industry,
            "our_ecosystem_position": competitive_map.our_ecosystem_position,
            "competitor_count": len(competitive_map.competitor_ecosystems),
            "disruption_opportunities": competitive_map.disruption_opportunities,
            "alternative_strategies": competitive_map.alternative_strategies,
            "market_gaps": competitive_map.market_gaps,
            "strategic_recommendations": competitive_map.strategic_recommendations,
            "created_at": competitive_map.created_at.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error creating competitive ecosystem map: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/optimize-ecosystem-growth")
async def optimize_ecosystem_growth(
    integration_id: str,
    orchestration_id: str,
    optimization_goals: Dict[str, float]
):
    """Optimize ecosystem growth based on integration and orchestration"""
    try:
        # Get integration
        if integration_id not in ecosystem_engine.ecosystem_integrations:
            raise HTTPException(status_code=404, detail="Integration not found")
        
        integration = ecosystem_engine.ecosystem_integrations[integration_id]
        
        # Get orchestration
        if orchestration_id not in ecosystem_engine.network_orchestrations:
            raise HTTPException(status_code=404, detail="Orchestration not found")
        
        orchestration = ecosystem_engine.network_orchestrations[orchestration_id]
        
        # Optimize growth
        optimization_result = await ecosystem_engine.optimize_ecosystem_growth(
            integration=integration,
            orchestration=orchestration,
            optimization_goals=optimization_goals
        )
        
        return optimization_result
        
    except Exception as e:
        logger.error(f"Error optimizing ecosystem growth: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/integration/{integration_id}/status")
async def get_integration_status(integration_id: str):
    """Get status of ecosystem integration"""
    try:
        status = ecosystem_engine.get_integration_status(integration_id)
        
        if not status:
            raise HTTPException(status_code=404, detail="Integration not found")
        
        return status
        
    except Exception as e:
        logger.error(f"Error getting integration status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/competitive-landscape/{industry}")
async def get_competitive_landscape(industry: str):
    """Get competitive landscape for industry"""
    try:
        landscape = ecosystem_engine.get_competitive_landscape(industry)
        
        return {
            "industry": industry,
            "competitive_maps": landscape,
            "total_maps": len(landscape)
        }
        
    except Exception as e:
        logger.error(f"Error getting competitive landscape: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/integrations")
async def list_ecosystem_integrations(
    network_id: Optional[str] = None,
    limit: int = 50
):
    """List ecosystem integrations"""
    try:
        integrations = []
        
        for integration in ecosystem_engine.ecosystem_integrations.values():
            if network_id and integration.network_id != network_id:
                continue
            
            integrations.append({
                "integration_id": integration.integration_id,
                "network_id": integration.network_id,
                "partnership_count": len(integration.partnership_opportunities),
                "growth_potential": integration.ecosystem_growth_potential,
                "competitive_advantages": len(integration.competitive_advantages),
                "created_at": integration.created_at.isoformat()
            })
        
        # Sort by creation date (newest first) and limit
        integrations.sort(key=lambda x: x["created_at"], reverse=True)
        integrations = integrations[:limit]
        
        return {
            "integrations": integrations,
            "total_count": len(integrations)
        }
        
    except Exception as e:
        logger.error(f"Error listing ecosystem integrations: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/orchestrations")
async def list_network_orchestrations(
    network_id: Optional[str] = None,
    limit: int = 50
):
    """List network effect orchestrations"""
    try:
        orchestrations = []
        
        for orchestration in ecosystem_engine.network_orchestrations.values():
            if network_id and orchestration.network_id != network_id:
                continue
            
            orchestrations.append({
                "orchestration_id": orchestration.orchestration_id,
                "network_id": orchestration.network_id,
                "leverage_points": len(orchestration.ecosystem_leverage_points),
                "competitive_moats": len(orchestration.competitive_moats),
                "expected_outcomes": orchestration.expected_outcomes
            })
        
        # Limit results
        orchestrations = orchestrations[:limit]
        
        return {
            "orchestrations": orchestrations,
            "total_count": len(orchestrations)
        }
        
    except Exception as e:
        logger.error(f"Error listing network orchestrations: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/integration/{integration_id}")
async def delete_ecosystem_integration(integration_id: str):
    """Delete ecosystem integration"""
    try:
        if integration_id not in ecosystem_engine.ecosystem_integrations:
            raise HTTPException(status_code=404, detail="Integration not found")
        
        del ecosystem_engine.ecosystem_integrations[integration_id]
        
        return {"message": "Integration deleted successfully"}
        
    except Exception as e:
        logger.error(f"Error deleting ecosystem integration: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "ecosystem-integration",
        "timestamp": datetime.now().isoformat(),
        "integrations_count": len(ecosystem_engine.ecosystem_integrations),
        "orchestrations_count": len(ecosystem_engine.network_orchestrations),
        "competitive_maps_count": len(ecosystem_engine.competitive_maps)
    }