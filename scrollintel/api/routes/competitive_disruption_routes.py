"""
API Routes for Competitive Disruption Engine

Provides endpoints for competitive ecosystem mapping, disruption opportunity detection,
and alternative ecosystem strategies.
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

from ...engines.competitive_disruption_engine import CompetitiveDisruptionEngine
from ...models.ecosystem_integration_models import DisruptionOpportunity, MarketGap

router = APIRouter(prefix="/api/competitive-disruption", tags=["competitive-disruption"])
logger = logging.getLogger(__name__)

# Initialize engine
disruption_engine = CompetitiveDisruptionEngine()


@router.post("/map-competitors")
async def map_competitor_ecosystems(
    industry: str,
    competitor_data: Dict[str, Any],
    market_intelligence: Dict[str, Any]
):
    """Map competitive ecosystem landscape"""
    try:
        intelligence = await disruption_engine.map_competitor_ecosystems(
            industry=industry,
            competitor_data=competitor_data,
            market_intelligence=market_intelligence
        )
        
        return {
            "intelligence_id": intelligence.intelligence_id,
            "industry": intelligence.industry,
            "competitor_count": len(intelligence.competitor_profiles),
            "disruption_vectors": intelligence.disruption_vectors,
            "strategic_insights": intelligence.strategic_insights,
            "opportunity_matrix": intelligence.opportunity_matrix,
            "created_at": intelligence.created_at.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error mapping competitor ecosystems: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/detect-disruption-opportunities")
async def detect_disruption_opportunities(
    intelligence_id: str,
    our_capabilities: Dict[str, Any],
    strategic_objectives: List[str]
):
    """Detect opportunities for competitive ecosystem disruption"""
    try:
        # Get intelligence
        if intelligence_id not in disruption_engine.competitive_intelligence:
            raise HTTPException(status_code=404, detail="Intelligence not found")
        
        intelligence = disruption_engine.competitive_intelligence[intelligence_id]
        
        # Detect opportunities
        opportunities = await disruption_engine.detect_disruption_opportunities(
            intelligence=intelligence,
            our_capabilities=our_capabilities,
            strategic_objectives=strategic_objectives
        )
        
        return {
            "intelligence_id": intelligence_id,
            "opportunities_count": len(opportunities),
            "opportunities": [
                {
                    "opportunity_id": opp.opportunity_id,
                    "target_competitor": opp.target_competitor,
                    "disruption_type": opp.disruption_type,
                    "market_segment": opp.market_segment,
                    "impact_assessment": opp.impact_assessment,
                    "success_probability": opp.success_probability,
                    "timeline": {k: v.isoformat() for k, v in opp.timeline.items()}
                }
                for opp in opportunities
            ]
        }
        
    except Exception as e:
        logger.error(f"Error detecting disruption opportunities: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate-alternative-strategies")
async def generate_alternative_strategies(
    intelligence_id: str,
    market_gaps: List[Dict[str, Any]],
    innovation_capabilities: Dict[str, Any]
):
    """Generate alternative ecosystem development strategies"""
    try:
        # Get intelligence
        if intelligence_id not in disruption_engine.competitive_intelligence:
            raise HTTPException(status_code=404, detail="Intelligence not found")
        
        intelligence = disruption_engine.competitive_intelligence[intelligence_id]
        
        # Convert market gaps data to MarketGap objects
        gap_objects = []
        for gap_data in market_gaps:
            gap = MarketGap(
                gap_id=gap_data.get('gap_id', f"gap_{len(gap_objects)}"),
                gap_type=gap_data.get('gap_type', 'market'),
                market_segment=gap_data.get('market_segment', 'unknown'),
                gap_description=gap_data.get('description', ''),
                opportunity_size=gap_data.get('opportunity_size', 0.5),
                competitive_intensity=gap_data.get('competitive_intensity', 0.5),
                entry_barriers=gap_data.get('entry_barriers', []),
                success_factors=gap_data.get('success_factors', []),
                recommended_approach=gap_data.get('recommended_approach', 'standard'),
                timeline_to_capture=gap_data.get('timeline', '12_months')
            )
            gap_objects.append(gap)
        
        # Generate alternatives
        alternatives = await disruption_engine.generate_alternative_strategies(
            intelligence=intelligence,
            market_gaps=gap_objects,
            innovation_capabilities=innovation_capabilities
        )
        
        return {
            "intelligence_id": intelligence_id,
            "alternatives_count": len(alternatives),
            "alternatives": [
                {
                    "alternative_id": alt.alternative_id,
                    "strategy_name": alt.strategy_name,
                    "market_approach": alt.market_approach,
                    "value_proposition": alt.value_proposition,
                    "competitive_differentiation": alt.competitive_differentiation,
                    "resource_allocation": alt.resource_allocation,
                    "success_metrics": alt.success_metrics
                }
                for alt in alternatives
            ]
        }
        
    except Exception as e:
        logger.error(f"Error generating alternative strategies: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/create-disruption-strategy")
async def create_disruption_strategy(
    opportunity_data: Dict[str, Any],
    our_capabilities: Dict[str, Any],
    resource_constraints: Dict[str, Any]
):
    """Create detailed disruption strategy"""
    try:
        # Create DisruptionOpportunity object
        opportunity = DisruptionOpportunity(
            opportunity_id=opportunity_data.get('opportunity_id', f"opp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
            disruption_type=opportunity_data.get('disruption_type', 'direct'),
            target_competitor=opportunity_data.get('target_competitor', ''),
            market_segment=opportunity_data.get('market_segment', ''),
            disruption_strategy=opportunity_data.get('strategy', ''),
            impact_assessment=opportunity_data.get('impact_assessment', {}),
            implementation_plan=opportunity_data.get('implementation_plan', []),
            resource_requirements=opportunity_data.get('resource_requirements', {}),
            success_probability=opportunity_data.get('success_probability', 0.5),
            timeline={}  # Will be populated by the engine
        )
        
        # Create strategy
        strategy = await disruption_engine.create_disruption_strategy(
            opportunity=opportunity,
            our_capabilities=our_capabilities,
            resource_constraints=resource_constraints
        )
        
        return {
            "strategy_id": strategy.strategy_id,
            "target_competitor": strategy.target_competitor,
            "disruption_type": strategy.disruption_type,
            "attack_vectors": len(strategy.attack_vectors),
            "resource_requirements": strategy.resource_requirements,
            "success_probability": strategy.success_probability,
            "expected_impact": strategy.expected_impact,
            "risk_mitigation": len(strategy.risk_mitigation),
            "contingency_plans": len(strategy.contingency_plans)
        }
        
    except Exception as e:
        logger.error(f"Error creating disruption strategy: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/intelligence/{intelligence_id}")
async def get_competitive_intelligence(intelligence_id: str):
    """Get competitive intelligence summary"""
    try:
        intelligence_summary = disruption_engine.get_competitive_intelligence(intelligence_id)
        
        if not intelligence_summary:
            raise HTTPException(status_code=404, detail="Intelligence not found")
        
        return intelligence_summary
        
    except Exception as e:
        logger.error(f"Error getting competitive intelligence: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/intelligence")
async def list_competitive_intelligence(
    industry: Optional[str] = None,
    limit: int = 50
):
    """List competitive intelligence reports"""
    try:
        intelligence_list = []
        
        for intelligence in disruption_engine.competitive_intelligence.values():
            if industry and intelligence.industry != industry:
                continue
            
            intelligence_list.append({
                "intelligence_id": intelligence.intelligence_id,
                "industry": intelligence.industry,
                "competitor_count": len(intelligence.competitor_profiles),
                "disruption_vectors": len(intelligence.disruption_vectors),
                "created_at": intelligence.created_at.isoformat()
            })
        
        # Sort by creation date (newest first) and limit
        intelligence_list.sort(key=lambda x: x["created_at"], reverse=True)
        intelligence_list = intelligence_list[:limit]
        
        return {
            "intelligence_reports": intelligence_list,
            "total_count": len(intelligence_list)
        }
        
    except Exception as e:
        logger.error(f"Error listing competitive intelligence: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/strategies")
async def list_disruption_strategies(
    target_competitor: Optional[str] = None,
    limit: int = 50
):
    """List disruption strategies"""
    try:
        strategies_list = []
        
        for strategy in disruption_engine.disruption_strategies.values():
            if target_competitor and strategy.target_competitor != target_competitor:
                continue
            
            strategies_list.append({
                "strategy_id": strategy.strategy_id,
                "target_competitor": strategy.target_competitor,
                "disruption_type": strategy.disruption_type,
                "success_probability": strategy.success_probability,
                "expected_impact": strategy.expected_impact
            })
        
        # Limit results
        strategies_list = strategies_list[:limit]
        
        return {
            "disruption_strategies": strategies_list,
            "total_count": len(strategies_list)
        }
        
    except Exception as e:
        logger.error(f"Error listing disruption strategies: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/intelligence/{intelligence_id}")
async def delete_competitive_intelligence(intelligence_id: str):
    """Delete competitive intelligence"""
    try:
        if intelligence_id not in disruption_engine.competitive_intelligence:
            raise HTTPException(status_code=404, detail="Intelligence not found")
        
        del disruption_engine.competitive_intelligence[intelligence_id]
        
        return {"message": "Intelligence deleted successfully"}
        
    except Exception as e:
        logger.error(f"Error deleting competitive intelligence: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "competitive-disruption",
        "timestamp": datetime.now().isoformat(),
        "intelligence_reports": len(disruption_engine.competitive_intelligence),
        "disruption_strategies": len(disruption_engine.disruption_strategies),
        "ecosystem_alternatives": len(disruption_engine.ecosystem_alternatives)
    }