"""
Comprehensive Demo for Ecosystem Integration Engine

This script demonstrates the integration of partnerships with influence networks
and competitive ecosystem development capabilities.
"""

import asyncio
import json
from datetime import datetime, timedelta

from scrollintel.engines.ecosystem_integration_engine import EcosystemIntegrationEngine
from scrollintel.engines.influence_mapping_engine import InfluenceMappingEngine
from scrollintel.engines.partnership_analysis_engine import PartnershipAnalysisEngine
from scrollintel.models.ecosystem_models import PartnershipOpportunity
from scrollintel.models.influence_network_models import InfluenceNetwork, InfluenceNode
from scrollintel.models.relationship_models import RelationshipProfile, RelationshipType


def create_sample_influence_network():
    """Create sample influence network for demo"""
    nodes = [
        InfluenceNode(
            id="ceo_techcorp",
            name="Elena Rodriguez",
            title="CEO",
            organization="TechCorp Global",
            industry="technology",
            influence_score=0.92,
            centrality_score=0.85,
            connections=["cto_innovate", "vp_strategy", "board_chair"],
            influence_type="decision_maker",
            geographic_reach=["north_america", "europe", "asia_pacific"],
            expertise_areas=["ai", "strategy", "leadership"],
            last_updated=datetime.now()
        ),
        InfluenceNode(
            id="cto_innovate",
            name="Dr. James Chen",
            title="CTO",
            organization="InnovateCorp",
            industry="technology",
            influence_score=0.88,
            centrality_score=0.78,
            connections=["ceo_techcorp", "research_lead", "vp_strategy"],
            influence_type="thought_leader",
            geographic_reach=["global"],
            expertise_areas=["artificial_intelligence", "quantum_computing", "innovation"],
            last_updated=datetime.now()
        )
    ]
    
    return InfluenceNetwork(
        id="global_tech_network",
        name="Global Technology Influence Network",
        industry="technology",
        nodes=nodes,
        edges=[],
        network_metrics={"density": 0.85, "avg_influence_score": 0.9},
        competitive_position={"market_position": "dominant"},
        created_at=datetime.now(),
        last_updated=datetime.now()
    )


async def demonstrate_ecosystem_integration():
    """Demonstrate comprehensive ecosystem integration capabilities"""
    print("🌐 ECOSYSTEM INTEGRATION ENGINE - COMPREHENSIVE DEMO")
    print("=" * 60)
    
    # Initialize engines
    ecosystem_engine = EcosystemIntegrationEngine()
    
    # Create sample data
    network = create_sample_influence_network()
    
    partnerships = [
        PartnershipOpportunity(
            opportunity_id="ai_alliance_partnership",
            partner_name="AI Innovations Alliance",
            partnership_type="strategic_alliance",
            industry="artificial_intelligence",
            strategic_value=0.95,
            market_expansion_potential=0.85,
            key_stakeholders=["ceo_techcorp", "cto_innovate"],
            timeline="9_months",
            investment_required=2000000,
            expected_roi=4.5,
            risk_assessment={"overall_risk": 0.25},
            competitive_advantages=["ai_leadership", "market_dominance"],
            created_at=datetime.now()
        )
    ]
    
    print(f"\n📊 PHASE 1: PARTNERSHIP-INFLUENCE INTEGRATION")
    print("-" * 50)
    
    # Integrate partnerships with influence network
    integration_strategy = {
        "focus": "synergy_maximization",
        "timeline": "aggressive",
        "resource_allocation": "growth_focused"
    }
    
    integration = await ecosystem_engine.integrate_partnership_with_influence_network(
        network=network,
        partnerships=partnerships,
        integration_strategy=integration_strategy
    )
    
    print(f"✅ Created Ecosystem Integration")
    print(f"   • Integration ID: {integration.integration_id}")
    print(f"   • Network: {integration.network_id}")
    print(f"   • Partnerships: {len(integration.partnership_opportunities)}")
    print(f"   • Growth Potential: {integration.ecosystem_growth_potential:.2f}x")
    print(f"   • Competitive Advantages: {len(integration.competitive_advantages)}")
    
    return integration


if __name__ == "__main__":
    asyncio.run(demonstrate_ecosystem_integration())