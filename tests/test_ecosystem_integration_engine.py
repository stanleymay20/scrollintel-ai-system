"""
Tests for Ecosystem Integration Engine

Tests the integration of partnerships with influence networks
and competitive ecosystem development capabilities.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from scrollintel.engines.ecosystem_integration_engine import (
    EcosystemIntegrationEngine,
    EcosystemIntegration,
    CompetitiveEcosystemMap,
    NetworkEffectOrchestration
)
from scrollintel.models.ecosystem_models import PartnershipOpportunity
from scrollintel.models.influence_network_models import InfluenceNetwork, InfluenceNode
from scrollintel.models.relationship_models import RelationshipProfile, RelationshipType


@pytest.fixture
def ecosystem_engine():
    """Create ecosystem integration engine for testing"""
    return EcosystemIntegrationEngine()


@pytest.fixture
def sample_influence_network():
    """Create sample influence network for testing"""
    nodes = [
        InfluenceNode(
            id="node_1",
            name="John Smith",
            title="CEO",
            organization="TechCorp",
            industry="technology",
            influence_score=0.85,
            centrality_score=0.7,
            connections=["node_2", "node_3"],
            influence_type="decision_maker",
            geographic_reach=["north_america"],
            expertise_areas=["ai", "strategy"],
            last_updated=datetime.now()
        ),
        InfluenceNode(
            id="node_2",
            name="Jane Doe",
            title="CTO",
            organization="InnovateCorp",
            industry="technology",
            influence_score=0.75,
            centrality_score=0.6,
            connections=["node_1", "node_3"],
            influence_type="thought_leader",
            geographic_reach=["global"],
            expertise_areas=["technology", "innovation"],
            last_updated=datetime.now()
        ),
        InfluenceNode(
            id="node_3",
            name="Bob Johnson",
            title="VP Strategy",
            organization="StrategyCorp",
            industry="consulting",
            influence_score=0.65,
            centrality_score=0.5,
            connections=["node_1", "node_2"],
            influence_type="connector",
            geographic_reach=["north_america", "europe"],
            expertise_areas=["strategy", "partnerships"],
            last_updated=datetime.now()
        )
    ]
    
    return InfluenceNetwork(
        id="test_network",
        name="Test Influence Network",
        industry="technology",
        nodes=nodes,
        edges=[],
        network_metrics={"density": 0.8, "avg_influence_score": 0.75},
        competitive_position={"market_position": "strong"},
        created_at=datetime.now(),
        last_updated=datetime.now()
    )


@pytest.fixture
def sample_partnerships():
    """Create sample partnerships for testing"""
    return [
        PartnershipOpportunity(
            opportunity_id="partnership_1",
            partner_name="AI Innovations Inc",
            partnership_type="strategic_alliance",
            industry="artificial_intelligence",
            strategic_value=0.85,
            market_expansion_potential=0.75,
            key_stakeholders=["node_1", "node_2"],
            timeline="6_months",
            investment_required=500000,
            expected_roi=2.5,
            risk_assessment={"overall_risk": 0.3},
            competitive_advantages=["ai_expertise", "market_access"],
            created_at=datetime.now()
        ),
        PartnershipOpportunity(
            opportunity_id="partnership_2",
            partner_name="Global Consulting Partners",
            partnership_type="joint_venture",
            industry="consulting",
            strategic_value=0.7,
            market_expansion_potential=0.8,
            key_stakeholders=["node_3"],
            timeline="12_months",
            investment_required=750000,
            expected_roi=3.0,
            risk_assessment={"overall_risk": 0.4},
            competitive_advantages=["global_reach", "consulting_expertise"],
            created_at=datetime.now()
        )
    ]


class TestEcosystemIntegrationEngine:
    """Test ecosystem integration engine functionality"""
    
    @pytest.mark.asyncio
    async def test_integrate_partnership_with_influence_network(
        self, ecosystem_engine, sample_influence_network, sample_partnerships
    ):
        """Test integrating partnerships with influence network"""
        integration_strategy = {
            "focus": "synergy_maximization",
            "timeline": "aggressive",
            "resource_allocation": "balanced"
        }
        
        integration = await ecosystem_engine.integrate_partnership_with_influence_network(
            network=sample_influence_network,
            partnerships=sample_partnerships,
            integration_strategy=integration_strategy
        )
        
        assert isinstance(integration, EcosystemIntegration)
        assert integration.network_id == sample_influence_network.id
        assert len(integration.partnership_opportunities) == len(sample_partnerships)
        assert integration.ecosystem_growth_potential > 0
        assert len(integration.competitive_advantages) > 0
        assert integration.integration_id in ecosystem_engine.ecosystem_integrations
    
    @pytest.mark.asyncio
    async def test_orchestrate_network_effects(
        self, ecosystem_engine, sample_influence_network, sample_partnerships
    ):
        """Test orchestrating network effects"""
        # First create integration
        integration_strategy = {"focus": "network_effects"}
        integration = await ecosystem_engine.integrate_partnership_with_influence_network(
            network=sample_influence_network,
            partnerships=sample_partnerships,
            integration_strategy=integration_strategy
        )
        
        # Then orchestrate network effects
        orchestration_config = {
            "priority": "growth_acceleration",
            "timeline": "6_months",
            "resource_constraints": {"budget": 1000000}
        }
        
        orchestration = await ecosystem_engine.orchestrate_network_effects(
            integration=integration,
            orchestration_config=orchestration_config
        )
        
        assert isinstance(orchestration, NetworkEffectOrchestration)
        assert orchestration.network_id == sample_influence_network.id
        assert len(orchestration.partnership_synergies) > 0
        assert len(orchestration.influence_multipliers) > 0
        assert len(orchestration.ecosystem_leverage_points) > 0
        assert len(orchestration.competitive_moats) > 0
    
    @pytest.mark.asyncio
    async def test_create_competitive_ecosystem_map(
        self, ecosystem_engine, sample_influence_network, sample_partnerships
    ):
        """Test creating competitive ecosystem map"""
        competitor_data = {
            "TechRival Corp": {
                "ecosystem_strength": 0.8,
                "partnerships": ["AI Partner 1", "Consulting Partner 1"],
                "market_position": "strong",
                "recent_moves": ["acquisition", "new_partnership"]
            },
            "Innovation Leaders": {
                "ecosystem_strength": 0.75,
                "partnerships": ["Tech Partner 2", "Strategy Partner 2"],
                "market_position": "competitive",
                "recent_moves": ["market_expansion"]
            }
        }
        
        competitive_map = await ecosystem_engine.create_competitive_ecosystem_map(
            industry="technology",
            our_network=sample_influence_network,
            our_partnerships=sample_partnerships,
            competitor_data=competitor_data
        )
        
        assert isinstance(competitive_map, CompetitiveEcosystemMap)
        assert competitive_map.industry == "technology"
        assert len(competitive_map.competitor_ecosystems) == len(competitor_data)
        assert len(competitive_map.disruption_opportunities) > 0
        assert len(competitive_map.alternative_strategies) > 0
        assert len(competitive_map.strategic_recommendations) > 0
    
    @pytest.mark.asyncio
    async def test_optimize_ecosystem_growth(
        self, ecosystem_engine, sample_influence_network, sample_partnerships
    ):
        """Test optimizing ecosystem growth"""
        # Create integration and orchestration first
        integration_strategy = {"focus": "optimization"}
        integration = await ecosystem_engine.integrate_partnership_with_influence_network(
            network=sample_influence_network,
            partnerships=sample_partnerships,
            integration_strategy=integration_strategy
        )
        
        orchestration_config = {"priority": "optimization"}
        orchestration = await ecosystem_engine.orchestrate_network_effects(
            integration=integration,
            orchestration_config=orchestration_config
        )
        
        # Optimize growth
        optimization_goals = {
            "growth_rate": 2.0,
            "market_share": 0.25,
            "competitive_advantage": 0.8
        }
        
        optimization_result = await ecosystem_engine.optimize_ecosystem_growth(
            integration=integration,
            orchestration=orchestration,
            optimization_goals=optimization_goals
        )
        
        assert "optimization_strategy" in optimization_result
        assert "growth_projections" in optimization_result
        assert "resource_allocation" in optimization_result
        assert "timeline_optimization" in optimization_result
        assert "success_metrics" in optimization_result
    
    @pytest.mark.asyncio
    async def test_analyze_partnership_influence_synergies(
        self, ecosystem_engine, sample_influence_network, sample_partnerships
    ):
        """Test analyzing partnership-influence synergies"""
        synergies = await ecosystem_engine._analyze_partnership_influence_synergies(
            network=sample_influence_network,
            partnerships=sample_partnerships
        )
        
        assert isinstance(synergies, dict)
        assert len(synergies) == len(sample_partnerships)
        
        for partnership in sample_partnerships:
            assert partnership.opportunity_id in synergies
            assert 0 <= synergies[partnership.opportunity_id] <= 1
    
    @pytest.mark.asyncio
    async def test_calculate_influence_amplification(
        self, ecosystem_engine, sample_influence_network, sample_partnerships
    ):
        """Test calculating influence amplification"""
        synergies = {
            "partnership_1": 0.8,
            "partnership_2": 0.6
        }
        
        amplification = await ecosystem_engine._calculate_influence_amplification(
            network=sample_influence_network,
            partnerships=sample_partnerships,
            synergies=synergies
        )
        
        assert isinstance(amplification, dict)
        assert len(amplification) == len(sample_partnerships)
        
        for partnership in sample_partnerships:
            assert partnership.opportunity_id in amplification
            assert amplification[partnership.opportunity_id] > 0
    
    @pytest.mark.asyncio
    async def test_identify_network_effects(
        self, ecosystem_engine, sample_influence_network, sample_partnerships
    ):
        """Test identifying network effects"""
        synergies = {
            "partnership_1": 0.8,
            "partnership_2": 0.7
        }
        
        network_effects = await ecosystem_engine._identify_network_effects(
            network=sample_influence_network,
            partnerships=sample_partnerships,
            synergies=synergies
        )
        
        assert isinstance(network_effects, dict)
        assert "direct_effects" in network_effects
        assert "indirect_effects" in network_effects
        assert "multiplier_effects" in network_effects
        assert "ecosystem_effects" in network_effects
        
        # Should have some effects given high synergies
        total_effects = (
            len(network_effects["direct_effects"]) +
            len(network_effects["indirect_effects"]) +
            len(network_effects["multiplier_effects"]) +
            len(network_effects["ecosystem_effects"])
        )
        assert total_effects > 0
    
    @pytest.mark.asyncio
    async def test_calculate_ecosystem_growth_potential(
        self, ecosystem_engine
    ):
        """Test calculating ecosystem growth potential"""
        synergies = {"partnership_1": 0.8, "partnership_2": 0.7}
        amplification = {"partnership_1": 1.5, "partnership_2": 1.3}
        network_effects = {
            "direct_effects": [{"effect": "market_access"}],
            "indirect_effects": [{"effect": "influence_cascade"}],
            "multiplier_effects": [{"effect": "partnership_synergy"}],
            "ecosystem_effects": [{"effect": "market_positioning"}]
        }
        
        growth_potential = await ecosystem_engine._calculate_ecosystem_growth_potential(
            synergies=synergies,
            amplification=amplification,
            network_effects=network_effects
        )
        
        assert isinstance(growth_potential, float)
        assert growth_potential > 0
        assert growth_potential <= 5.0  # Should be capped at 5x
    
    def test_get_integration_status(
        self, ecosystem_engine, sample_influence_network, sample_partnerships
    ):
        """Test getting integration status"""
        # Create a mock integration
        integration = EcosystemIntegration(
            integration_id="test_integration",
            network_id=sample_influence_network.id,
            partnership_opportunities=sample_partnerships,
            influence_amplification={"partnership_1": 1.5},
            network_effects={"direct_effects": []},
            competitive_advantages=["advantage_1"],
            ecosystem_growth_potential=2.5,
            integration_strategy={"focus": "test"},
            created_at=datetime.now(),
            last_updated=datetime.now()
        )
        
        ecosystem_engine.ecosystem_integrations["test_integration"] = integration
        
        status = ecosystem_engine.get_integration_status("test_integration")
        
        assert status is not None
        assert status["integration_id"] == "test_integration"
        assert status["network_id"] == sample_influence_network.id
        assert status["partnership_count"] == len(sample_partnerships)
        assert status["growth_potential"] == 2.5
    
    def test_get_integration_status_not_found(self, ecosystem_engine):
        """Test getting status for non-existent integration"""
        status = ecosystem_engine.get_integration_status("non_existent")
        assert status is None
    
    def test_get_competitive_landscape(self, ecosystem_engine):
        """Test getting competitive landscape"""
        # Create mock competitive maps
        competitive_map = CompetitiveEcosystemMap(
            map_id="test_map",
            industry="technology",
            our_ecosystem_position={"strength": 0.8},
            competitor_ecosystems={"competitor_1": {"strength": 0.7}},
            disruption_opportunities=[{"opportunity": "test"}],
            alternative_strategies=[{"strategy": "test"}],
            market_gaps=[{"gap": "test"}],
            strategic_recommendations=["recommendation_1"],
            created_at=datetime.now()
        )
        
        ecosystem_engine.competitive_maps["test_map"] = competitive_map
        
        landscape = ecosystem_engine.get_competitive_landscape("technology")
        
        assert isinstance(landscape, list)
        assert len(landscape) == 1
        assert landscape[0]["industry"] == "technology"
        assert landscape[0]["map_id"] == "test_map"


class TestEcosystemIntegrationEdgeCases:
    """Test edge cases and error conditions"""
    
    @pytest.mark.asyncio
    async def test_integration_with_empty_partnerships(
        self, ecosystem_engine, sample_influence_network
    ):
        """Test integration with empty partnerships list"""
        integration = await ecosystem_engine.integrate_partnership_with_influence_network(
            network=sample_influence_network,
            partnerships=[],
            integration_strategy={"focus": "minimal"}
        )
        
        assert integration.ecosystem_growth_potential >= 0
        assert len(integration.partnership_opportunities) == 0
    
    @pytest.mark.asyncio
    async def test_integration_with_low_synergy_partnerships(
        self, ecosystem_engine, sample_influence_network
    ):
        """Test integration with low synergy partnerships"""
        low_synergy_partnership = PartnershipOpportunity(
            opportunity_id="low_synergy",
            partner_name="Unrelated Corp",
            partnership_type="supplier",
            industry="manufacturing",  # Different industry
            strategic_value=0.2,  # Low strategic value
            market_expansion_potential=0.1,  # Low expansion potential
            key_stakeholders=[],  # No overlapping stakeholders
            timeline="24_months",
            investment_required=100000,
            expected_roi=1.1,
            risk_assessment={"overall_risk": 0.8},
            competitive_advantages=[],
            created_at=datetime.now()
        )
        
        integration = await ecosystem_engine.integrate_partnership_with_influence_network(
            network=sample_influence_network,
            partnerships=[low_synergy_partnership],
            integration_strategy={"focus": "minimal"}
        )
        
        # Should still create integration but with low growth potential
        assert integration.ecosystem_growth_potential < 1.0
        assert len(integration.competitive_advantages) >= 0
    
    @pytest.mark.asyncio
    async def test_competitive_map_with_no_competitors(
        self, ecosystem_engine, sample_influence_network, sample_partnerships
    ):
        """Test creating competitive map with no competitor data"""
        competitive_map = await ecosystem_engine.create_competitive_ecosystem_map(
            industry="niche_technology",
            our_network=sample_influence_network,
            our_partnerships=sample_partnerships,
            competitor_data={}
        )
        
        assert len(competitive_map.competitor_ecosystems) == 0
        assert len(competitive_map.disruption_opportunities) >= 0
        assert len(competitive_map.strategic_recommendations) > 0


if __name__ == "__main__":
    pytest.main([__file__])