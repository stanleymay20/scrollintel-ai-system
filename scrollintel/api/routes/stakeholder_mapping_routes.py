"""
API Routes for Stakeholder Mapping System
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime

from ...engines.stakeholder_mapping_engine import StakeholderMappingEngine
from ...models.stakeholder_influence_models import (
    Stakeholder, StakeholderMap, InfluenceAssessment,
    RelationshipOptimization, StakeholderAnalysis
)

router = APIRouter(prefix="/api/v1/stakeholder-mapping", tags=["stakeholder-mapping"])
logger = logging.getLogger(__name__)

# Initialize the stakeholder mapping engine
stakeholder_engine = StakeholderMappingEngine()


@router.post("/identify-stakeholders")
async def identify_stakeholders(organization_context: Dict[str, Any]):
    """
    Identify key board and executive stakeholders
    """
    try:
        stakeholders = stakeholder_engine.identify_key_stakeholders(organization_context)
        
        return {
            "success": True,
            "stakeholders": [
                {
                    "id": s.id,
                    "name": s.name,
                    "title": s.title,
                    "organization": s.organization,
                    "stakeholder_type": s.stakeholder_type.value,
                    "influence_level": s.influence_level.value,
                    "communication_style": s.communication_style.value,
                    "priorities": [
                        {
                            "name": p.name,
                            "description": p.description,
                            "importance": p.importance,
                            "category": p.category
                        } for p in s.priorities
                    ]
                } for s in stakeholders
            ],
            "total_count": len(stakeholders),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error identifying stakeholders: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to identify stakeholders: {str(e)}")


@router.post("/assess-influence/{stakeholder_id}")
async def assess_stakeholder_influence(stakeholder_id: str, context: Dict[str, Any]):
    """
    Assess stakeholder influence level and factors
    """
    try:
        # Get stakeholder from context or database
        stakeholder_data = context.get('stakeholder')
        if not stakeholder_data:
            raise HTTPException(status_code=404, detail="Stakeholder not found")
        
        # Convert dict to Stakeholder object (simplified)
        stakeholder = Stakeholder(**stakeholder_data)
        
        influence_assessment = stakeholder_engine.assess_stakeholder_influence(stakeholder, context)
        
        return {
            "success": True,
            "stakeholder_id": stakeholder_id,
            "influence_assessment": {
                "formal_authority": influence_assessment.formal_authority,
                "informal_influence": influence_assessment.informal_influence,
                "network_centrality": influence_assessment.network_centrality,
                "expertise_credibility": influence_assessment.expertise_credibility,
                "resource_control": influence_assessment.resource_control,
                "overall_influence": influence_assessment.overall_influence,
                "assessment_date": influence_assessment.assessment_date.isoformat()
            },
            "influence_level": "critical" if influence_assessment.overall_influence >= 0.8 else
                             "high" if influence_assessment.overall_influence >= 0.6 else
                             "medium" if influence_assessment.overall_influence >= 0.4 else "low",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error assessing stakeholder influence: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to assess influence: {str(e)}")


@router.post("/map-relationships")
async def map_stakeholder_relationships(request_data: Dict[str, Any]):
    """
    Create comprehensive stakeholder relationship mapping
    """
    try:
        stakeholders_data = request_data.get('stakeholders', [])
        
        # Convert stakeholder data to Stakeholder objects (simplified)
        stakeholders = []
        for s_data in stakeholders_data:
            stakeholder = Stakeholder(**s_data)
            stakeholders.append(stakeholder)
        
        stakeholder_map = stakeholder_engine.map_stakeholder_relationships(stakeholders)
        
        return {
            "success": True,
            "stakeholder_map": {
                "id": stakeholder_map.id,
                "organization_id": stakeholder_map.organization_id,
                "stakeholder_count": len(stakeholder_map.stakeholders),
                "network_count": len(stakeholder_map.influence_networks),
                "key_relationships": len(stakeholder_map.key_relationships),
                "power_dynamics": stakeholder_map.power_dynamics,
                "created_at": stakeholder_map.created_at.isoformat(),
                "updated_at": stakeholder_map.updated_at.isoformat()
            },
            "influence_networks": [
                {
                    "id": network.id,
                    "name": network.name,
                    "stakeholder_count": len(network.stakeholders),
                    "power_centers": network.power_centers
                } for network in stakeholder_map.influence_networks
            ],
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error mapping stakeholder relationships: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to map relationships: {str(e)}")


@router.post("/optimize-relationships")
async def optimize_stakeholder_relationships(request_data: Dict[str, Any]):
    """
    Generate relationship optimization strategies
    """
    try:
        stakeholder_map_data = request_data.get('stakeholder_map')
        objectives = request_data.get('objectives', [])
        
        if not stakeholder_map_data:
            raise HTTPException(status_code=400, detail="Stakeholder map data required")
        
        # Convert to StakeholderMap object (simplified)
        stakeholder_map = StakeholderMap(**stakeholder_map_data)
        
        optimizations = stakeholder_engine.optimize_stakeholder_relationships(stakeholder_map, objectives)
        
        return {
            "success": True,
            "optimizations": [
                {
                    "stakeholder_id": opt.stakeholder_id,
                    "current_strength": opt.current_relationship_strength,
                    "target_strength": opt.target_relationship_strength,
                    "improvement_potential": opt.target_relationship_strength - opt.current_relationship_strength,
                    "strategies": opt.optimization_strategies,
                    "action_items": opt.action_items,
                    "timeline": {
                        "short_term": opt.timeline["short_term"].isoformat(),
                        "medium_term": opt.timeline["medium_term"].isoformat(),
                        "long_term": opt.timeline["long_term"].isoformat()
                    },
                    "success_metrics": opt.success_metrics
                } for opt in optimizations
            ],
            "total_optimizations": len(optimizations),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error optimizing relationships: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to optimize relationships: {str(e)}")


@router.post("/analyze-comprehensive/{stakeholder_id}")
async def analyze_stakeholder_comprehensive(stakeholder_id: str, context: Dict[str, Any]):
    """
    Perform comprehensive stakeholder analysis
    """
    try:
        stakeholder_data = context.get('stakeholder')
        if not stakeholder_data:
            raise HTTPException(status_code=404, detail="Stakeholder not found")
        
        # Convert dict to Stakeholder object (simplified)
        stakeholder = Stakeholder(**stakeholder_data)
        
        analysis = stakeholder_engine.analyze_stakeholder_comprehensive(stakeholder, context)
        
        return {
            "success": True,
            "stakeholder_id": stakeholder_id,
            "comprehensive_analysis": {
                "influence_assessment": {
                    "overall_influence": analysis.influence_assessment.overall_influence,
                    "formal_authority": analysis.influence_assessment.formal_authority,
                    "informal_influence": analysis.influence_assessment.informal_influence,
                    "network_centrality": analysis.influence_assessment.network_centrality,
                    "expertise_credibility": analysis.influence_assessment.expertise_credibility,
                    "resource_control": analysis.influence_assessment.resource_control
                },
                "relationship_optimization": {
                    "current_strength": analysis.relationship_optimization.current_relationship_strength,
                    "target_strength": analysis.relationship_optimization.target_relationship_strength,
                    "strategies": analysis.relationship_optimization.optimization_strategies,
                    "action_items": analysis.relationship_optimization.action_items
                },
                "engagement_history": analysis.engagement_history,
                "predicted_positions": analysis.predicted_positions,
                "engagement_recommendations": analysis.engagement_recommendations,
                "analysis_date": analysis.analysis_date.isoformat()
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in comprehensive analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to perform comprehensive analysis: {str(e)}")


@router.get("/stakeholder/{stakeholder_id}/profile")
async def get_stakeholder_profile(stakeholder_id: str):
    """
    Get detailed stakeholder profile
    """
    try:
        # In a real implementation, this would fetch from database
        # For now, return a sample profile
        
        return {
            "success": True,
            "stakeholder_id": stakeholder_id,
            "profile": {
                "basic_info": {
                    "name": "Sample Stakeholder",
                    "title": "Board Member",
                    "organization": "Current Organization"
                },
                "influence_metrics": {
                    "overall_influence": 0.85,
                    "influence_level": "high",
                    "key_strengths": ["formal_authority", "expertise_credibility"]
                },
                "communication_preferences": {
                    "style": "analytical",
                    "preferred_channels": ["email", "formal_meetings"],
                    "frequency": "weekly"
                },
                "relationship_status": {
                    "current_strength": 0.7,
                    "trend": "improving",
                    "last_interaction": "2024-01-15"
                }
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting stakeholder profile: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get profile: {str(e)}")


@router.get("/network-analysis")
async def get_network_analysis():
    """
    Get stakeholder network analysis
    """
    try:
        return {
            "success": True,
            "network_analysis": {
                "total_stakeholders": 12,
                "influence_distribution": {
                    "critical": 2,
                    "high": 4,
                    "medium": 4,
                    "low": 2
                },
                "relationship_strength": {
                    "strong": 8,
                    "neutral": 15,
                    "weak": 5,
                    "adversarial": 0
                },
                "power_centers": [
                    "board_chair_001",
                    "ceo_001",
                    "lead_investor_001"
                ],
                "influence_clusters": [
                    ["board_chair_001", "board_member_002", "board_member_003"],
                    ["ceo_001", "cfo_001", "cto_001"]
                ],
                "coalition_opportunities": [
                    {
                        "issue": "ai_investment",
                        "potential_supporters": ["cto_001", "lead_investor_001", "board_member_tech_001"],
                        "success_probability": 0.8
                    }
                ]
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting network analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get network analysis: {str(e)}")


@router.post("/relationship-tracking")
async def track_relationship_changes(tracking_data: Dict[str, Any]):
    """
    Track changes in stakeholder relationships over time
    """
    try:
        stakeholder_id = tracking_data.get('stakeholder_id')
        interaction_data = tracking_data.get('interaction_data', {})
        
        if not stakeholder_id:
            raise HTTPException(status_code=400, detail="Stakeholder ID required")
        
        # Process relationship tracking (simplified)
        relationship_update = {
            "stakeholder_id": stakeholder_id,
            "previous_strength": interaction_data.get('previous_strength', 0.5),
            "current_strength": interaction_data.get('current_strength', 0.6),
            "change": interaction_data.get('current_strength', 0.6) - interaction_data.get('previous_strength', 0.5),
            "interaction_type": interaction_data.get('type', 'meeting'),
            "interaction_outcome": interaction_data.get('outcome', 'positive'),
            "updated_at": datetime.now().isoformat()
        }
        
        return {
            "success": True,
            "relationship_update": relationship_update,
            "recommendations": [
                "Continue current engagement strategy" if relationship_update["change"] >= 0 
                else "Review and adjust engagement approach",
                "Schedule follow-up interaction within 2 weeks",
                "Monitor relationship strength trends"
            ],
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error tracking relationship changes: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to track relationship: {str(e)}")