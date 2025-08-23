"""
Tests for Influence Mapping Engine - Global Influence Network System
Tests comprehensive influence network mapping, power structure analysis,
and competitive positioning capabilities.
"""
import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock

from scrollintel.engines.influence_mapping_engine import (
    InfluenceMappingEngine,
    InfluenceNode,
    InfluenceEdge,
    PowerStructure,
    CompetitivePosition,
    NetworkGap,
    InfluenceLevel,
    NetworkPosition
)
from scrollintel.models.relationship_models import Relationship, RelationshipType
from scrollintel.models.influence_strategy_models import InfluenceTarget, InfluenceType


class TestInfluenceMappingEngine:
    """Test suite for InfluenceMappingEngine"""

    @pytest.fixture
    def engine(self):
        """Create test engine instance"""
        return InfluenceMappingEngine()

    @pytest.fixture
    def sample_relationships(self):
        """Create sample relationships for testing"""
        return [
            Relationship(
                id="rel_1",
                source_id="person_1",
                target_id="person_2",
                relationship_type=RelationshipType.PROFESSIONAL,
                strength=0.8,
                interaction_frequency=0.7,
                trust_level=0.9
            ),
            Relationship(
                id="rel_2",
                source_id="person_2",
                target_id="person_3",
                relationship_type=RelationshipType.BUSINESS,
                strength=0.9,
                interaction_frequency=0.8,
                trust_level=0.8
            )
        ]

    @pytest.fixture
    def sample_targets(self):
        """Create sample influence targets for testing"""
        return [
            InfluenceTarget(
                id="person_1",
                name="John Smith",
                title="CEO",
                organization="TechCorp",
                industry="Technology",
                influence_score=0.9,
                expertise_areas=["AI", "Strategy"],
                geographic_reach=["US", "Europe"]
            ),
            InfluenceTarget(
                id="person_2",
                name="Jane Doe",
                title="CTO",
                organization="InnovateInc",
                industry="Technology",
                influence_score=0.8,
                expertise_areas=["Engineering", "Innovation"],
                geographic_reach=["US", "Asia"]
            ),
            InfluenceTarget(
                id="person_3",
                name="Bob Johnson",
                title="VP Marketing",
                organization="MarketLeader",
                industry="Marketing",
                influence_score=0.7,
                expertise_areas=["Marketing", "Brand"],
                geographic_reach=["Global"]
            )
        ]

    @pytest.mark.asyncio
    async def test_build_influence_network(self, engine, sample_relationships, sample_targets):
        """Test building influence network from relationships and targets"""
        result = await engine.build_influence_network(
            relationships=sample_relationships,
            influence_targets=sample_targets
        )

        assert "network_id" in result
        assert "nodes" in result
        assert "edges" in result
        assert "metrics" in result
        assert result["network_size"] == len(sample_targets)
        assert result["connection_count"] == len(sample_relationships)

    @pytest.mark.asyncio
    async def test_analyze_power_structures(self, engine, sample_relationships, sample_targets):
        """Test power structure analysis"""
        # First build the network
        await engine.build_influence_network(
            relationships=sample_relationships,
            influence_targets=sample_targets
        )

        # Then analyze power structures
        power_structure = await engine.analyze_power_structures()

        assert isinstance(power_structure, PowerStructure)
        assert "hierarchy_levels" in power_structure.__dict__
        assert "decision_makers" in power_structure.__dict__
        assert "influence_clusters" in power_structure.__dict__
        assert "power_brokers" in power_structure.__dict__

    @pytest.mark.asyncio
    async def test_assess_competitive_position(self, engine, sample_relationships, sample_targets):
        """Test competitive position assessment"""
        # First build the network
        await engine.build_influence_network(
            relationships=sample_relationships,
            influence_targets=sample_targets
        )

        # Then assess competitive position
        competitive_position = await engine.assess_competitive_position(
            our_organization="TechCorp",
            competitors=["InnovateInc", "MarketLeader"]
        )

        assert isinstance(competitive_position, CompetitivePosition)
        assert "our_position" in competitive_position.__dict__
        assert "competitor_positions" in competitive_position.__dict__
        assert "market_gaps" in competitive_position.__dict__
        assert "strategic_opportunities" in competitive_position.__dict__

    @pytest.mark.asyncio
    async def test_identify_network_gaps(self, engine, sample_relationships, sample_targets):
        """Test network gap identification"""
        # First build the network
        await engine.build_influence_network(
            relationships=sample_relationships,
            influence_targets=sample_targets
        )

        # Then identify gaps
        gaps = await engine.identify_network_gaps()

        assert isinstance(gaps, list)
        for gap in gaps:
            assert isinstance(gap, NetworkGap)
            assert hasattr(gap, 'gap_type')
            assert hasattr(gap, 'impact_score')
            assert hasattr(gap, 'priority_level')

    @pytest.mark.asyncio
    async def test_monitor_influence_shifts(self, engine, sample_relationships, sample_targets):
        """Test influence shift monitoring"""
        # First build the network
        await engine.build_influence_network(
            relationships=sample_relationships,
            influence_targets=sample_targets
        )

        # Then monitor shifts
        shifts = await engine.monitor_influence_shifts(time_window_days=30)

        assert "analysis_period" in shifts
        assert "influence_changes" in shifts
        assert "emerging_influencers" in shifts
        assert "declining_influence" in shifts
        assert "relationship_changes" in shifts
        assert "alerts" in shifts

    def test_influence_node_creation(self):
        """Test InfluenceNode creation and properties"""
        node = InfluenceNode(
            id="test_node",
            name="Test Person",
            title="CEO",
            organization="TestCorp",
            industry="Technology",
            influence_score=0.8,
            centrality_score=0.7,
            betweenness_score=0.6,
            closeness_score=0.5,
            network_position=NetworkPosition.CENTRAL_HUB,
            influence_level=InfluenceLevel.HIGH,
            expertise_areas=["AI", "Strategy"],
            geographic_reach=["US", "Europe"]
        )

        assert node.id == "test_node"
        assert node.name == "Test Person"
        assert node.influence_score == 0.8
        assert node.network_position == NetworkPosition.CENTRAL_HUB
        assert node.influence_level == InfluenceLevel.HIGH

    def test_influence_edge_creation(self):
        """Test InfluenceEdge creation and properties"""
        edge = InfluenceEdge(
            source_id="person_1",
            target_id="person_2",
            relationship_type=RelationshipType.PROFESSIONAL,
            strength=0.8,
            direction="bidirectional",
            interaction_frequency=0.7,
            trust_level=0.9
        )

        assert edge.source_id == "person_1"
        assert edge.target_id == "person_2"
        assert edge.relationship_type == RelationshipType.PROFESSIONAL
        assert edge.strength == 0.8
        assert edge.direction == "bidirectional"

    @pytest.mark.asyncio
    async def test_network_metrics_calculation(self, engine, sample_relationships, sample_targets):
        """Test network metrics calculation"""
        # Build network
        result = await engine.build_influence_network(
            relationships=sample_relationships,
            influence_targets=sample_targets
        )

        metrics = result["metrics"]
        assert "basic_metrics" in metrics
        assert "centrality_measures" in metrics
        assert "connectivity" in metrics

        basic_metrics = metrics["basic_metrics"]
        assert "num_nodes" in basic_metrics
        assert "num_edges" in basic_metrics
        assert "density" in basic_metrics

    def test_determine_influence_level(self, engine):
        """Test influence level determination"""
        assert engine._determine_influence_level(0.9) == InfluenceLevel.CRITICAL
        assert engine._determine_influence_level(0.7) == InfluenceLevel.HIGH
        assert engine._determine_influence_level(0.5) == InfluenceLevel.MODERATE
        assert engine._determine_influence_level(0.3) == InfluenceLevel.LOW
        assert engine._determine_influence_level(0.1) == InfluenceLevel.MINIMAL

    def test_determine_network_position(self, engine):
        """Test network position determination"""
        assert engine._determine_network_position(0.9, 0.8, 0.7) == NetworkPosition.CENTRAL_HUB
        assert engine._determine_network_position(0.5, 0.8, 0.6) == NetworkPosition.BRIDGE
        assert engine._determine_network_position(0.7, 0.5, 0.7) == NetworkPosition.GATEKEEPER
        assert engine._determine_network_position(0.4, 0.3, 0.4) == NetworkPosition.PERIPHERAL
        assert engine._determine_network_position(0.2, 0.1, 0.2) == NetworkPosition.ISOLATED

    @pytest.mark.asyncio
    async def test_error_handling_empty_network(self, engine):
        """Test error handling with empty network data"""
        result = await engine.build_influence_network(
            relationships=[],
            influence_targets=[]
        )

        assert result["network_size"] == 0
        assert result["connection_count"] == 0

    @pytest.mark.asyncio
    async def test_error_handling_invalid_data(self, engine):
        """Test error handling with invalid data"""
        with pytest.raises(Exception):
            await engine.build_influence_network(
                relationships=None,
                influence_targets=None
            )

    @pytest.mark.asyncio
    async def test_node_to_dict_conversion(self, engine):
        """Test node to dictionary conversion"""
        node = InfluenceNode(
            id="test_node",
            name="Test Person",
            title="CEO",
            organization="TestCorp",
            industry="Technology",
            influence_score=0.8,
            centrality_score=0.7,
            betweenness_score=0.6,
            closeness_score=0.5,
            network_position=NetworkPosition.CENTRAL_HUB,
            influence_level=InfluenceLevel.HIGH,
            expertise_areas=["AI", "Strategy"],
            geographic_reach=["US", "Europe"]
        )

        node_dict = engine._node_to_dict(node)

        assert node_dict["id"] == "test_node"
        assert node_dict["name"] == "Test Person"
        assert node_dict["influence_score"] == 0.8
        assert node_dict["network_position"] == "central_hub"
        assert node_dict["influence_level"] == "high"

    @pytest.mark.asyncio
    async def test_edge_to_dict_conversion(self, engine):
        """Test edge to dictionary conversion"""
        edge = InfluenceEdge(
            source_id="person_1",
            target_id="person_2",
            relationship_type=RelationshipType.PROFESSIONAL,
            strength=0.8,
            direction="bidirectional",
            interaction_frequency=0.7,
            trust_level=0.9
        )

        edge_dict = engine._edge_to_dict(edge)

        assert edge_dict["source_id"] == "person_1"
        assert edge_dict["target_id"] == "person_2"
        assert edge_dict["relationship_type"] == "professional"
        assert edge_dict["strength"] == 0.8
        assert edge_dict["direction"] == "bidirectional"

    @pytest.mark.asyncio
    async def test_cache_functionality(self, engine, sample_relationships, sample_targets):
        """Test caching functionality"""
        # Build network
        await engine.build_influence_network(
            relationships=sample_relationships,
            influence_targets=sample_targets
        )

        # Check that caches are populated
        assert len(engine.node_cache) > 0
        assert len(engine.edge_cache) > 0

        # Verify cache contents
        for target in sample_targets:
            assert target.id in engine.node_cache

    @pytest.mark.asyncio
    async def test_performance_with_large_network(self, engine):
        """Test performance with larger network"""
        # Create larger dataset
        large_targets = []
        large_relationships = []

        for i in range(100):
            target = InfluenceTarget(
                id=f"person_{i}",
                name=f"Person {i}",
                title="Executive",
                organization=f"Company {i}",
                industry="Technology",
                influence_score=0.5 + (i % 5) * 0.1,
                expertise_areas=["Tech"],
                geographic_reach=["US"]
            )
            large_targets.append(target)

            if i > 0:
                relationship = Relationship(
                    id=f"rel_{i}",
                    source_id=f"person_{i-1}",
                    target_id=f"person_{i}",
                    relationship_type=RelationshipType.PROFESSIONAL,
                    strength=0.7,
                    interaction_frequency=0.6,
                    trust_level=0.8
                )
                large_relationships.append(relationship)

        # Build network and measure time
        start_time = datetime.now()
        result = await engine.build_influence_network(
            relationships=large_relationships,
            influence_targets=large_targets
        )
        end_time = datetime.now()

        # Verify results
        assert result["network_size"] == 100
        assert result["connection_count"] == 99

        # Check performance (should complete within reasonable time)
        processing_time = (end_time - start_time).total_seconds()
        assert processing_time < 10  # Should complete within 10 seconds


class TestInfluenceMappingIntegration:
    """Integration tests for influence mapping with other systems"""

    @pytest.fixture
    def engine(self):
        return InfluenceMappingEngine()

    @pytest.mark.asyncio
    async def test_integration_with_relationship_engine(self, engine):
        """Test integration with relationship building engine"""
        # This would test integration with the existing relationship engine
        # For now, we'll test the interface compatibility
        
        # Mock relationship engine data
        mock_relationships = [
            Relationship(
                id="rel_integration",
                source_id="person_a",
                target_id="person_b",
                relationship_type=RelationshipType.BUSINESS,
                strength=0.8
            )
        ]

        mock_targets = [
            InfluenceTarget(
                id="person_a",
                name="Alice",
                title="CEO",
                organization="AliceCorp",
                industry="Tech",
                influence_score=0.9
            ),
            InfluenceTarget(
                id="person_b",
                name="Bob",
                title="CTO",
                organization="BobTech",
                industry="Tech",
                influence_score=0.8
            )
        ]

        # Test that the engine can process this data
        result = await engine.build_influence_network(
            relationships=mock_relationships,
            influence_targets=mock_targets
        )

        assert result["network_size"] == 2
        assert result["connection_count"] == 1

    @pytest.mark.asyncio
    async def test_integration_with_influence_strategy_engine(self, engine):
        """Test integration with influence strategy engine"""
        # This would test how influence mapping integrates with strategy planning
        # For now, we'll test the data compatibility
        
        mock_targets = [
            InfluenceTarget(
                id="strategic_target",
                name="Strategic Person",
                title="Industry Leader",
                organization="LeaderCorp",
                industry="Finance",
                influence_score=0.95,
                influence_type=InfluenceType.INDUSTRY_LEADER
            )
        ]

        result = await engine.build_influence_network(
            relationships=[],
            influence_targets=mock_targets
        )

        # Verify strategic targets are properly processed
        nodes = result["nodes"]
        assert len(nodes) == 1
        assert nodes[0]["influence_score"] == 0.95


if __name__ == "__main__":
    pytest.main([__file__])