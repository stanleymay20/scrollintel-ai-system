"""
API routes for Influence Mapping System
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Dict, Optional, Any
from datetime import datetime
import logging

from ...engines.influence_mapping_engine import InfluenceMappingEngine
from ...engines.relationship_building_engine import RelationshipBuildingEngine
from ...engines.partnership_analysis_engine import PartnershipAnalysisEngine
from ...models.influence_network_models import (
    InfluenceNetwork, NetworkGap, CompetitivePosition, 
    NetworkExpansionPlan, InfluenceMetrics
)
from ...models.relationship_models import RelationshipProfile
from ...models.ecosystem_models import PartnershipOpportunity

router = APIRouter(prefix="/api/v1/influence-mapping", tags=["influence-mapping"])
logger = logging.getLogger(__name__)

# Initialize engines
influence_engine = InfluenceMappingEngine()
relationship_engine = RelationshipBuildingEngine()
partnership_engine = PartnershipAnalysisEngine()


@router.post("/networks", response_model=Dict[str, Any])
async def create_influence_network(
    industry: str,
    stakeholder_ids: List[str],
    include_partnerships: bool = True,
    external_data: Optional[Dict[str, Any]] = None
):
    """Create a new influence network mapping"""
    try:
        logger.info(f"Creating influence network for industry: {industry}")
        
        # Get stakeholder profiles (simplified - would integrate with relationship engine)
        stakeholders = []
        for stakeholder_id in stakeholder_ids:
            # In practice, would fetch from relationship engine
            stakeholder = RelationshipProfile(
                stakeholder_id=stakeholder_id,
                name=f"Stakeholder {stakeholder_id}",
                title="Executive",
                organization=f"Company {stakeholder_id}",
                relationship_type="executive",
                relationship_status="active",
                personality_profile=None,
                influence_level=0.7,
                decision_making_power=0.6,
                network_connections=[],
                trust_metrics=None,
                relationship_strength=0.6,
                engagement_frequency=0.5,
                response_rate=0.8,
                relationship_start_date=datetime.now(),
                last_interaction_date=None,
                interaction_history=[],
                relationship_goals=[],
                development_strategy="",
                next_planned_interaction=None,
                key_interests=["technology", "innovation"],
                business_priorities=[],
                personal_interests=[],
                communication_cadence="monthly"
            )
            stakeholders.append(stakeholder)
        
        # Get partnerships if requested
        partnerships = []
        if include_partnerships:
            # In practice, would fetch from partnership engine
            partnerships = []
        
        # Create influence network
        network = await influence_engine.create_influence_network(
            industry=industry,
            stakeholders=stakeholders,
            partnerships=partnerships,
            external_data=external_data
        )
        
        return {
            "network_id": network.id,
            "name": network.name,
            "industry": network.industry,
            "node_count": len(network.nodes),
            "edge_count": len(network.edges),
            "network_metrics": network.network_metrics,
            "competitive_position": network.competitive_position,
            "created_at": network.created_at.isoformat(),
            "status": "created"
        }
        
    except Exception as e:
        logger.error(f"Error creating influence network: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/networks/{network_id}", response_model=Dict[str, Any])
async def get_influence_network(network_id: str):
    """Get influence network details"""
    try:
        network = influence_engine.networks.get(network_id)
        if not network:
            raise HTTPException(status_code=404, detail="Network not found")
        
        return {
            "network_id": network.id,
            "name": network.name,
            "industry": network.industry,
            "nodes": [
                {
                    "id": node.id,
                    "name": node.name,
                    "title": node.title,
                    "organization": node.organization,
                    "influence_score": node.influence_score,
                    "centrality_score": node.centrality_score,
                    "influence_type": node.influence_type,
                    "connections_count": len(node.connections),
                    "expertise_areas": node.expertise_areas
                }
                for node in network.nodes
            ],
            "edges": [
                {
                    "source": edge.source_id,
                    "target": edge.target_id,
                    "relationship_type": edge.relationship_type,
                    "strength": edge.strength,
                    "direction": edge.direction,
                    "influence_flow": edge.influence_flow
                }
                for edge in network.edges
            ],
            "network_metrics": network.network_metrics,
            "competitive_position": network.competitive_position,
            "last_updated": network.last_updated.isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting influence network: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/networks/{network_id}/analyze-centrality", response_model=Dict[str, Any])
async def analyze_network_centrality(
    network_id: str,
    algorithms: Optional[List[str]] = Query(default=None)
):
    """Analyze network centrality using multiple algorithms"""
    try:
        network = influence_engine.networks.get(network_id)
        if not network:
            raise HTTPException(status_code=404, detail="Network not found")
        
        centrality_results = await influence_engine.analyze_network_centrality(
            network=network,
            algorithms=algorithms
        )
        
        return {
            "network_id": network_id,
            "centrality_results": centrality_results,
            "top_central_nodes": [
                {
                    "node_id": node.id,
                    "name": node.name,
                    "centrality_score": node.centrality_score,
                    "influence_score": node.influence_score
                }
                for node in sorted(network.nodes, key=lambda n: n.centrality_score, reverse=True)[:10]
            ],
            "analysis_date": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing network centrality: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/networks/{network_id}/identify-gaps", response_model=Dict[str, Any])
async def identify_network_gaps(
    network_id: str,
    target_objectives: List[str]
):
    """Identify gaps in the influence network"""
    try:
        network = influence_engine.networks.get(network_id)
        if not network:
            raise HTTPException(status_code=404, detail="Network not found")
        
        gaps = await influence_engine.identify_network_gaps(
            network=network,
            target_objectives=target_objectives
        )
        
        return {
            "network_id": network_id,
            "total_gaps": len(gaps),
            "high_priority_gaps": len([g for g in gaps if g.priority == 'high']),
            "gaps": [
                {
                    "gap_id": gap.gap_id,
                    "gap_type": gap.gap_type,
                    "description": gap.description,
                    "priority": gap.priority,
                    "potential_impact": gap.potential_impact,
                    "effort_required": gap.effort_required,
                    "recommended_actions": gap.recommended_actions,
                    "target_nodes": gap.target_nodes
                }
                for gap in gaps
            ],
            "analysis_date": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error identifying network gaps: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/networks/{network_id}/expansion-strategy", response_model=Dict[str, Any])
async def generate_expansion_strategy(
    network_id: str,
    gap_ids: List[str],
    resources: Dict[str, Any]
):
    """Generate network expansion strategy"""
    try:
        network = influence_engine.networks.get(network_id)
        if not network:
            raise HTTPException(status_code=404, detail="Network not found")
        
        # Get gaps by IDs (simplified - would have proper gap storage)
        all_gaps = await influence_engine.identify_network_gaps(
            network=network,
            target_objectives=["technology", "innovation", "leadership"]
        )
        selected_gaps = [gap for gap in all_gaps if gap.gap_id in gap_ids]
        
        strategy = await influence_engine.generate_network_expansion_strategy(
            network=network,
            gaps=selected_gaps,
            resources=resources
        )
        
        return {
            "network_id": network_id,
            "strategy": strategy,
            "generated_at": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating expansion strategy: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/networks/{network_id}/competitive-position", response_model=Dict[str, Any])
async def get_competitive_position(network_id: str):
    """Get competitive position analysis"""
    try:
        network = influence_engine.networks.get(network_id)
        if not network:
            raise HTTPException(status_code=404, detail="Network not found")
        
        return {
            "network_id": network_id,
            "competitive_position": network.competitive_position,
            "network_strength": network.network_metrics.get('network_cohesion', 0.0),
            "influence_distribution": {
                "avg_influence": network.network_metrics.get('avg_influence_score', 0.0),
                "max_influence": network.network_metrics.get('max_influence_score', 0.0),
                "influence_std": network.network_metrics.get('influence_std', 0.0)
            },
            "connectivity_metrics": {
                "density": network.network_metrics.get('density', 0.0),
                "avg_connections": network.network_metrics.get('avg_connections', 0.0),
                "max_connections": network.network_metrics.get('max_connections', 0.0)
            },
            "analysis_date": network.last_updated.isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting competitive position: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/networks/{network_id}/key-influencers", response_model=Dict[str, Any])
async def get_key_influencers(
    network_id: str,
    top_n: int = Query(default=10, ge=1, le=50)
):
    """Get key influencers in the network"""
    try:
        network = influence_engine.networks.get(network_id)
        if not network:
            raise HTTPException(status_code=404, detail="Network not found")
        
        # Sort by combined influence and centrality
        sorted_nodes = sorted(
            network.nodes,
            key=lambda n: (n.influence_score * 0.6 + n.centrality_score * 0.4),
            reverse=True
        )
        
        key_influencers = sorted_nodes[:top_n]
        
        return {
            "network_id": network_id,
            "total_nodes": len(network.nodes),
            "key_influencers": [
                {
                    "node_id": node.id,
                    "name": node.name,
                    "title": node.title,
                    "organization": node.organization,
                    "influence_score": node.influence_score,
                    "centrality_score": node.centrality_score,
                    "combined_score": node.influence_score * 0.6 + node.centrality_score * 0.4,
                    "influence_type": node.influence_type,
                    "connections_count": len(node.connections),
                    "expertise_areas": node.expertise_areas,
                    "geographic_reach": node.geographic_reach
                }
                for node in key_influencers
            ],
            "analysis_date": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting key influencers: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/networks/{network_id}/influence-paths", response_model=Dict[str, Any])
async def find_influence_paths(
    network_id: str,
    source_node_id: str,
    target_node_id: str,
    max_hops: int = Query(default=3, ge=1, le=6)
):
    """Find influence paths between two nodes"""
    try:
        network = influence_engine.networks.get(network_id)
        if not network:
            raise HTTPException(status_code=404, detail="Network not found")
        
        # Build adjacency list for path finding
        adj_list = {}
        for node in network.nodes:
            adj_list[node.id] = []
        
        for edge in network.edges:
            adj_list[edge.source_id].append({
                'target': edge.target_id,
                'strength': edge.strength,
                'influence_flow': edge.influence_flow
            })
            if edge.direction == 'bidirectional':
                adj_list[edge.target_id].append({
                    'target': edge.source_id,
                    'strength': edge.strength,
                    'influence_flow': edge.influence_flow
                })
        
        # Find paths using BFS with limited depth
        paths = []
        queue = [(source_node_id, [source_node_id], 1.0)]
        visited_paths = set()
        
        while queue and len(paths) < 10:  # Limit to top 10 paths
            current_node, path, path_strength = queue.pop(0)
            
            if len(path) > max_hops:
                continue
            
            if current_node == target_node_id and len(path) > 1:
                path_key = tuple(path)
                if path_key not in visited_paths:
                    visited_paths.add(path_key)
                    paths.append({
                        'path': path,
                        'length': len(path) - 1,
                        'strength': path_strength,
                        'influence_flow': path_strength
                    })
                continue
            
            for neighbor in adj_list.get(current_node, []):
                if neighbor['target'] not in path:  # Avoid cycles
                    new_path = path + [neighbor['target']]
                    new_strength = path_strength * neighbor['strength']
                    queue.append((neighbor['target'], new_path, new_strength))
        
        # Sort paths by strength
        paths.sort(key=lambda p: p['strength'], reverse=True)
        
        return {
            "network_id": network_id,
            "source_node_id": source_node_id,
            "target_node_id": target_node_id,
            "max_hops": max_hops,
            "paths_found": len(paths),
            "paths": paths,
            "analysis_date": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error finding influence paths: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/networks", response_model=Dict[str, Any])
async def list_influence_networks(
    industry: Optional[str] = None,
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0)
):
    """List all influence networks"""
    try:
        networks = list(influence_engine.networks.values())
        
        # Filter by industry if specified
        if industry:
            networks = [n for n in networks if n.industry.lower() == industry.lower()]
        
        # Apply pagination
        total_count = len(networks)
        networks = networks[offset:offset + limit]
        
        return {
            "total_count": total_count,
            "limit": limit,
            "offset": offset,
            "networks": [
                {
                    "network_id": network.id,
                    "name": network.name,
                    "industry": network.industry,
                    "node_count": len(network.nodes),
                    "edge_count": len(network.edges),
                    "network_health": network.network_metrics.get('network_cohesion', 0.0),
                    "created_at": network.created_at.isoformat(),
                    "last_updated": network.last_updated.isoformat()
                }
                for network in networks
            ]
        }
        
    except Exception as e:
        logger.error(f"Error listing influence networks: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/networks/{network_id}", response_model=Dict[str, str])
async def delete_influence_network(network_id: str):
    """Delete an influence network"""
    try:
        if network_id not in influence_engine.networks:
            raise HTTPException(status_code=404, detail="Network not found")
        
        del influence_engine.networks[network_id]
        
        return {
            "message": f"Network {network_id} deleted successfully",
            "deleted_at": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting influence network: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/networks/{network_id}/monitor", response_model=Dict[str, Any])
async def start_network_monitoring(
    network_id: str,
    monitoring_config: Dict[str, Any]
):
    """Start real-time monitoring of influence network changes"""
    try:
        network = influence_engine.networks.get(network_id)
        if not network:
            raise HTTPException(status_code=404, detail="Network not found")
        
        # In practice, would set up real-time monitoring
        monitoring_id = f"monitor_{network_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        return {
            "monitoring_id": monitoring_id,
            "network_id": network_id,
            "monitoring_config": monitoring_config,
            "status": "monitoring_started",
            "started_at": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting network monitoring: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health", response_model=Dict[str, str])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "influence-mapping",
        "timestamp": datetime.now().isoformat()
    }