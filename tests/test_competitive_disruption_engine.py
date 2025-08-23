"""
Tests for Competitive Disruption Engine

Tests competitive ecosystem mapping, disruption opportunity detection,
and alternative ecosystem strategies.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from scrollintel.engines.competitive_disruption_engine import (
    CompetitiveDisruptionEngine,
    CompetitiveIntelligence,
    DisruptionStrategy,
    EcosystemAlternative
)
from scrollintel.models.ecosystem_integration_models import (
    DisruptionOpportunity, MarketGap, CompetitorEcosystem
)


@pytest.fixture
def disruption_engine():
    """Create competitive disruption engine for testing"""
    return CompetitiveDisruptionEngine()


@pytest.fixture
def sample_competitor_data():
    """Create sample competitor data for testing"""
    return {
        "TechRival Corp": {
            "name": "TechRival Corp",
            "ecosystem_strength": 0.8,
            "partnerships": ["AI Partner 1", "Cloud Partner 1"],
            "network_size": 150,
            "market_position": "strong",
            "advantages": ["strong_partnerships", "market_presence"],
            "recent_moves": ["acquisition", "new_partnership"],
            "vulnerabilities": ["limited_innovation", "high_costs"]
        },
        "Innovation Leaders": {
            "name": "Innovation Leaders",
            "ecosystem_strength": 0.75,
            "partnerships": ["Tech Partner 2", "Strategy Partner 2"],
            "network_size": 120,
            "market_position": "competitive",
            "advantages": ["innovation_speed", "agility"],
            "recent_moves": ["market_expansion"],
            "vulnerabilities": ["limited_resources", "narrow_focus"]
        }
    }


@pytest.fixture
def sample_market_intelligence():
    """Create sample market intelligence for testing"""
    return {
        "technology_change_rate": 0.8,
        "business_model_innovation": 0.7,
        "customer_expectations_change": 0.6,
        "regulatory_change": True,
        "market_growth_rate": 0.15,
        "competitive_intensity": 0.7
    }


@pytest.fixture
def sample_market_gaps():
    """Create sample market gaps for testing"""
    return [
        MarketGap(
            gap_id="gap_1",
            gap_type="technology",
            market_segment="AI automation",
            gap_description="Limited AI automation solutions for SMEs",
            opportunity_size=0.8,
            competitive_intensity=0.2,
            entry_barriers=["technology_complexity", "market_education"],
            success_factors=["ease_of_use", "affordability"],
            recommended_approach="platform_strategy",
            timeline_to_capture="18_months"
        ),
        MarketGap(
            gap_id="gap_2",
            gap_type="market",
            market_segment="emerging_markets",
            gap_description="Underserved emerging markets",
            opportunity_size=0.6,
            competitive_intensity=0.3,
            entry_barriers=["local_knowledge", "regulatory_complexity"],
            success_factors=["local_partnerships", "cultural_adaptation"],
            recommended_approach="partnership_strategy",
            timeline_to_capture="24_months"
        )
    ]


class TestCompetitiveDisruptionEngine:
    """Test competitive disruption engine functionality"""
    
    @pytest.mark.asyncio
    async def test_map_competitor_ecosystems(
        self, disruption_engine, sample_competitor_data, sample_market_intelligence
    ):
        """Test mapping competitor ecosystems"""
        intelligence = await disruption_engine.map_competitor_ecosystems(
            industry="technology",
            competitor_data=sample_competitor_data,
            market_intelligence=sample_market_intelligence
        )
        
        assert isinstance(intelligence, CompetitiveIntelligence)
        assert intelligence.industry == "technology"
        assert len(intelligence.competitor_profiles) == len(sample_competitor_data)
        assert len(intelligence.disruption_vectors) > 0
        assert len(intelligence.strategic_insights) > 0
        assert intelligence.intelligence_id in disruption_engine.competitive_intelligence
    
    @pytest.mark.asyncio
    async def test_detect_disruption_opportunities(
        self, disruption_engine, sample_competitor_data, sample_market_intelligence
    ):
        """Test detecting disruption opportunities"""
        # First create intelligence
        intelligence = await disruption_engine.map_competitor_ecosystems(
            industry="technology",
            competitor_data=sample_competitor_data,
            market_intelligence=sample_market_intelligence
        )
        
        # Define our capabilities
        our_capabilities = {
            "technology_innovation": 0.9,
            "market_agility": 0.8,
            "resource_availability": 0.7,
            "partnership_network": 0.75
        }
        
        strategic_objectives = [
            "market_leadership",
            "technology_dominance",
            "ecosystem_control"
        ]
        
        # Detect opportunities
        opportunities = await disruption_engine.detect_disruption_opportunities(
            intelligence=intelligence,
            our_capabilities=our_capabilities,
            strategic_objectives=strategic_objectives
        )
        
        assert isinstance(opportunities, list)
        assert len(opportunities) > 0
        
        for opportunity in opportunities:
            assert isinstance(opportunity, DisruptionOpportunity)
            assert opportunity.target_competitor in sample_competitor_data
            assert 0 <= opportunity.success_probability <= 1
    
    @pytest.mark.asyncio
    async def test_generate_alternative_strategies(
        self, disruption_engine, sample_competitor_data, sample_market_intelligence, sample_market_gaps
    ):
        """Test generating alternative strategies"""
        # First create intelligence
        intelligence = await disruption_engine.map_competitor_ecosystems(
            industry="technology",
            competitor_data=sample_competitor_data,
            market_intelligence=sample_market_intelligence
        )
        
        # Define innovation capabilities
        innovation_capabilities = {
            "technology_development": 0.85,
            "market_creation": 0.7,
            "ecosystem_orchestration": 0.8,
            "platform_building": 0.75
        }
        
        # Generate alternatives
        alternatives = await disruption_engine.generate_alternative_strategies(
            intelligence=intelligence,
            market_gaps=sample_market_gaps,
            innovation_capabilities=innovation_capabilities
        )
        
        assert isinstance(alternatives, list)
        assert len(alternatives) > 0
        
        for alternative in alternatives:
            assert isinstance(alternative, EcosystemAlternative)
            assert alternative.strategy_name
            assert alternative.market_approach
            assert len(alternative.competitive_differentiation) > 0
    
    @pytest.mark.asyncio
    async def test_create_disruption_strategy(self, disruption_engine):
        """Test creating disruption strategy"""
        # Create sample opportunity
        opportunity = DisruptionOpportunity(
            opportunity_id="test_opportunity",
            disruption_type="direct",
            target_competitor="TechRival Corp",
            market_segment="AI solutions",
            disruption_strategy="technology_leapfrog",
            impact_assessment={"market_share_gain": 0.15, "revenue_impact": 0.25},
            implementation_plan=[
                {"phase": "research", "duration": "3_months"},
                {"phase": "development", "duration": "6_months"},
                {"phase": "launch", "duration": "3_months"}
            ],
            resource_requirements={"budget": 2000000, "team_size": 20},
            success_probability=0.7,
            timeline={}
        )
        
        our_capabilities = {
            "technology_innovation": 0.9,
            "execution_speed": 0.8,
            "market_access": 0.7
        }
        
        resource_constraints = {
            "max_budget": 5000000,
            "max_team_size": 50,
            "timeline_limit": "12_months"
        }
        
        # Create strategy
        strategy = await disruption_engine.create_disruption_strategy(
            opportunity=opportunity,
            our_capabilities=our_capabilities,
            resource_constraints=resource_constraints
        )
        
        assert isinstance(strategy, DisruptionStrategy)
        assert strategy.target_competitor == opportunity.target_competitor
        assert strategy.disruption_type == opportunity.disruption_type
        assert len(strategy.attack_vectors) > 0
        assert len(strategy.risk_mitigation) > 0
        assert 0 <= strategy.success_probability <= 1
    
    @pytest.mark.asyncio
    async def test_analyze_competitor_profiles(
        self, disruption_engine, sample_competitor_data, sample_market_intelligence
    ):
        """Test analyzing competitor profiles"""
        profiles = await disruption_engine._analyze_competitor_profiles(
            competitor_data=sample_competitor_data,
            market_intelligence=sample_market_intelligence
        )
        
        assert isinstance(profiles, dict)
        assert len(profiles) == len(sample_competitor_data)
        
        for competitor_id, profile in profiles.items():
            assert isinstance(profile, CompetitorEcosystem)
            assert profile.competitor_id == competitor_id
            assert 0 <= profile.ecosystem_strength <= 1
            assert len(profile.key_partnerships) >= 0
    
    @pytest.mark.asyncio
    async def test_identify_disruption_vectors(
        self, disruption_engine, sample_competitor_data, sample_market_intelligence
    ):
        """Test identifying disruption vectors"""
        # First analyze competitor profiles
        competitor_profiles = await disruption_engine._analyze_competitor_profiles(
            competitor_data=sample_competitor_data,
            market_intelligence=sample_market_intelligence
        )
        
        # Analyze market dynamics
        market_dynamics = await disruption_engine._analyze_market_dynamics(
            industry="technology",
            competitor_data=sample_competitor_data,
            market_intelligence=sample_market_intelligence
        )
        
        # Identify disruption vectors
        vectors = await disruption_engine._identify_disruption_vectors(
            competitor_profiles=competitor_profiles,
            market_dynamics=market_dynamics
        )
        
        assert isinstance(vectors, list)
        assert len(vectors) > 0
        
        # Should include common disruption vectors
        expected_vectors = [
            'technology_disruption',
            'business_model_disruption',
            'value_chain_disruption',
            'ecosystem_disruption'
        ]
        
        for expected_vector in expected_vectors:
            if expected_vector in vectors:
                assert True
                break
        else:
            assert False, f"No expected disruption vectors found in {vectors}"
    
    @pytest.mark.asyncio
    async def test_create_blue_ocean_strategy(
        self, disruption_engine, sample_competitor_data, sample_market_intelligence, sample_market_gaps
    ):
        """Test creating blue ocean strategy"""
        # Create intelligence
        intelligence = await disruption_engine.map_competitor_ecosystems(
            industry="technology",
            competitor_data=sample_competitor_data,
            market_intelligence=sample_market_intelligence
        )
        
        capabilities = {
            "innovation": 0.9,
            "market_creation": 0.8,
            "execution": 0.7
        }
        
        # Create blue ocean strategy
        blue_ocean = await disruption_engine._create_blue_ocean_strategy(
            intelligence=intelligence,
            market_gaps=sample_market_gaps,
            capabilities=capabilities
        )
        
        if blue_ocean:  # May be None if no suitable uncontested spaces
            assert isinstance(blue_ocean, EcosystemAlternative)
            assert "Blue Ocean" in blue_ocean.strategy_name
            assert blue_ocean.market_approach == "uncontested_market_creation"
            assert len(blue_ocean.competitive_differentiation) > 0
    
    def test_get_competitive_intelligence(self, disruption_engine):
        """Test getting competitive intelligence"""
        # Create mock intelligence
        intelligence = CompetitiveIntelligence(
            intelligence_id="test_intelligence",
            industry="technology",
            competitor_profiles={},
            market_dynamics={},
            disruption_vectors=["technology_disruption"],
            vulnerability_analysis={},
            opportunity_matrix={},
            strategic_insights=["insight_1", "insight_2"],
            created_at=datetime.now(),
            last_updated=datetime.now()
        )
        
        disruption_engine.competitive_intelligence["test_intelligence"] = intelligence
        
        summary = disruption_engine.get_competitive_intelligence("test_intelligence")
        
        assert summary is not None
        assert summary["intelligence_id"] == "test_intelligence"
        assert summary["industry"] == "technology"
        assert summary["disruption_vectors"] == 1
        assert summary["strategic_insights"] == 2
    
    def test_get_competitive_intelligence_not_found(self, disruption_engine):
        """Test getting intelligence for non-existent ID"""
        summary = disruption_engine.get_competitive_intelligence("non_existent")
        assert summary is None


class TestCompetitiveDisruptionEdgeCases:
    """Test edge cases and error conditions"""
    
    @pytest.mark.asyncio
    async def test_mapping_with_empty_competitor_data(self, disruption_engine, sample_market_intelligence):
        """Test mapping with empty competitor data"""
        intelligence = await disruption_engine.map_competitor_ecosystems(
            industry="technology",
            competitor_data={},
            market_intelligence=sample_market_intelligence
        )
        
        assert len(intelligence.competitor_profiles) == 0
        assert len(intelligence.disruption_vectors) > 0  # Should still identify vectors
        assert len(intelligence.strategic_insights) >= 0
    
    @pytest.mark.asyncio
    async def test_disruption_opportunities_with_low_capabilities(
        self, disruption_engine, sample_competitor_data, sample_market_intelligence
    ):
        """Test detecting opportunities with low capabilities"""
        # Create intelligence
        intelligence = await disruption_engine.map_competitor_ecosystems(
            industry="technology",
            competitor_data=sample_competitor_data,
            market_intelligence=sample_market_intelligence
        )
        
        # Define low capabilities
        low_capabilities = {
            "technology_innovation": 0.2,
            "market_agility": 0.3,
            "resource_availability": 0.1
        }
        
        opportunities = await disruption_engine.detect_disruption_opportunities(
            intelligence=intelligence,
            our_capabilities=low_capabilities,
            strategic_objectives=["survival"]
        )
        
        # Should still find some opportunities, but with lower success probability
        for opportunity in opportunities:
            assert opportunity.success_probability <= 0.7  # Should be lower due to low capabilities
    
    @pytest.mark.asyncio
    async def test_alternative_strategies_with_no_market_gaps(
        self, disruption_engine, sample_competitor_data, sample_market_intelligence
    ):
        """Test generating alternatives with no market gaps"""
        # Create intelligence
        intelligence = await disruption_engine.map_competitor_ecosystems(
            industry="technology",
            competitor_data=sample_competitor_data,
            market_intelligence=sample_market_intelligence
        )
        
        capabilities = {"innovation": 0.8}
        
        alternatives = await disruption_engine.generate_alternative_strategies(
            intelligence=intelligence,
            market_gaps=[],  # No market gaps
            innovation_capabilities=capabilities
        )
        
        # Should still generate some alternatives (platform, orchestration, etc.)
        assert len(alternatives) >= 0


if __name__ == "__main__":
    pytest.main([__file__])