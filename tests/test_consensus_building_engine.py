"""
Tests for Consensus Building Engine

This module tests the board consensus building strategy development,
tracking, and facilitation capabilities.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from scrollintel.engines.consensus_building_engine import ConsensusBuildingEngine
from scrollintel.models.consensus_building_models import (
    ConsensusStatus, StakeholderPosition, InfluenceLevel, ConsensusStrategy as StrategyType
)


class TestConsensusBuildingEngine:
    """Test cases for ConsensusBuildingEngine"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.engine = ConsensusBuildingEngine()
    
    def test_create_consensus_building(self):
        """Test creating a new consensus building process"""
        deadline = datetime.now() + timedelta(days=30)
        
        process = self.engine.create_consensus_building(
            title="Strategic Investment Decision",
            description="Build consensus on cloud infrastructure investment",
            decision_topic="Cloud Infrastructure Investment",
            target_consensus_level=ConsensusStatus.STRONG_CONSENSUS,
            deadline=deadline,
            facilitator_id="cto-001"
        )
        
        assert process.title == "Strategic Investment Decision"
        assert process.decision_topic == "Cloud Infrastructure Investment"
        assert process.target_consensus_level == ConsensusStatus.STRONG_CONSENSUS
        assert process.current_consensus_level == ConsensusStatus.NOT_STARTED
        assert process.deadline == deadline
        assert process.facilitator_id == "cto-001"
        assert process.consensus_score == 0.0
        assert process.id in self.engine.consensus_processes
    
    def test_add_board_member(self):
        """Test adding board members to consensus process"""
        process = self.engine.create_consensus_building(
            title="Test Process",
            description="Test description",
            decision_topic="Test Topic"
        )
        
        member = self.engine.add_board_member(
            process_id=process.id,
            name="John Smith",
            role="Board Chair",
            influence_level=InfluenceLevel.CRITICAL,
            decision_making_style="analytical",
            key_concerns=["Budget impact", "Timeline"],
            motivations=["Company growth", "Risk management"],
            communication_preferences=["Data-driven presentations", "One-on-one meetings"]
        )
        
        assert member.name == "John Smith"
        assert member.role == "Board Chair"
        assert member.influence_level == InfluenceLevel.CRITICAL
        assert member.decision_making_style == "analytical"
        assert len(member.key_concerns) == 2
        assert len(member.motivations) == 2
        assert len(member.communication_preferences) == 2
        assert len(process.board_members) == 1
    
    def test_update_stakeholder_position(self):
        """Test updating stakeholder positions"""
        process = self.engine.create_consensus_building(
            title="Test Process",
            description="Test description",
            decision_topic="Test Topic"
        )
        
        # Add board member first
        member = self.engine.add_board_member(
            process_id=process.id,
            name="Jane Doe",
            role="Board Member",
            influence_level=InfluenceLevel.HIGH
        )
        
        # Update position
        position = self.engine.update_stakeholder_position(
            process_id=process.id,
            stakeholder_id=member.id,
            stakeholder_name="Jane Doe",
            position=StakeholderPosition.SUPPORT,
            confidence_level=0.8,
            key_concerns=["Implementation complexity"],
            requirements_for_support=["Detailed implementation plan"],
            deal_breakers=["Budget overrun"]
        )
        
        assert position.stakeholder_name == "Jane Doe"
        assert position.current_position == StakeholderPosition.SUPPORT
        assert position.confidence_level == 0.8
        assert len(position.key_concerns) == 1
        assert len(position.requirements_for_support) == 1
        assert len(position.deal_breakers) == 1
        assert len(process.stakeholder_positions) == 1
        assert process.consensus_score > 0  # Should be updated
    
    def test_identify_consensus_barriers(self):
        """Test identifying barriers to consensus"""
        process = self.engine.create_consensus_building(
            title="Test Process",
            description="Test description",
            decision_topic="Test Topic"
        )
        
        # Add stakeholders with different positions
        member1 = self.engine.add_board_member(
            process_id=process.id,
            name="Supporter",
            role="Board Member",
            influence_level=InfluenceLevel.HIGH
        )
        
        member2 = self.engine.add_board_member(
            process_id=process.id,
            name="Opposer",
            role="Board Member",
            influence_level=InfluenceLevel.MODERATE
        )
        
        # Set positions
        self.engine.update_stakeholder_position(
            process_id=process.id,
            stakeholder_id=member1.id,
            stakeholder_name="Supporter",
            position=StakeholderPosition.SUPPORT,
            key_concerns=["Need more information"]
        )
        
        self.engine.update_stakeholder_position(
            process_id=process.id,
            stakeholder_id=member2.id,
            stakeholder_name="Opposer",
            position=StakeholderPosition.OPPOSE,
            key_concerns=["Trust issues", "Budget concerns"]
        )
        
        barriers = self.engine.identify_consensus_barriers(
            process_id=process.id,
            decision_type="strategic_decision"
        )
        
        assert len(barriers) > 0
        barrier_types = [b.barrier_type for b in barriers]
        assert "information" in barrier_types  # Due to "need more information" concern
        assert "trust" in barrier_types        # Due to "trust issues" concern
        assert "interests" in barrier_types    # Due to opposition
        
        # Check that barriers were added to process
        assert len(process.barriers) == len(barriers)
    
    def test_develop_consensus_strategy(self):
        """Test developing consensus strategies"""
        process = self.engine.create_consensus_building(
            title="Test Process",
            description="Test description",
            decision_topic="Test Topic"
        )
        
        # Add stakeholders with mixed positions
        for i, (name, position) in enumerate([
            ("Supporter1", StakeholderPosition.SUPPORT),
            ("Supporter2", StakeholderPosition.SUPPORT),
            ("Neutral1", StakeholderPosition.NEUTRAL),
            ("Opposer1", StakeholderPosition.OPPOSE)
        ]):
            member = self.engine.add_board_member(
                process_id=process.id,
                name=name,
                role="Board Member",
                influence_level=InfluenceLevel.HIGH if i == 3 else InfluenceLevel.MODERATE
            )
            
            self.engine.update_stakeholder_position(
                process_id=process.id,
                stakeholder_id=member.id,
                stakeholder_name=name,
                position=position
            )
        
        strategies = self.engine.develop_consensus_strategy(
            process_id=process.id,
            decision_type="strategic_decision"
        )
        
        assert len(strategies) > 0
        strategy_types = [s.strategy_type for s in strategies]
        
        # Should include information sharing for neutral stakeholders
        assert StrategyType.INFORMATION_SHARING in strategy_types
        
        # Should include coalition building since there are supporters
        assert StrategyType.COALITION_BUILDING in strategy_types
        
        # Should include direct persuasion for high-influence opposer
        assert StrategyType.DIRECT_PERSUASION in strategy_types
        
        # Check that strategies were added to process
        assert len(process.strategies) == len(strategies)
        
        # Verify strategy details
        info_strategy = next(s for s in strategies if s.strategy_type == StrategyType.INFORMATION_SHARING)
        assert len(info_strategy.tactics) > 0
        assert len(info_strategy.expected_outcomes) > 0
        assert info_strategy.success_probability > 0
    
    def test_create_consensus_action(self):
        """Test creating consensus building actions"""
        process = self.engine.create_consensus_building(
            title="Test Process",
            description="Test description",
            decision_topic="Test Topic"
        )
        
        deadline = datetime.now() + timedelta(days=7)
        
        action = self.engine.create_consensus_action(
            process_id=process.id,
            title="Board Presentation",
            description="Present detailed analysis to board",
            action_type="presentation",
            target_stakeholders=["member-001", "member-002"],
            responsible_party="cto-001",
            deadline=deadline,
            expected_impact="Increase support by providing comprehensive information"
        )
        
        assert action.title == "Board Presentation"
        assert action.action_type == "presentation"
        assert len(action.target_stakeholders) == 2
        assert action.responsible_party == "cto-001"
        assert action.deadline == deadline
        assert action.status == "planned"
        assert len(process.actions) == 1
    
    def test_track_consensus_progress(self):
        """Test tracking consensus building progress"""
        process = self.engine.create_consensus_building(
            title="Test Process",
            description="Test description",
            decision_topic="Test Topic"
        )
        
        # Add stakeholders with different positions and influence levels
        stakeholder_data = [
            ("High Supporter", InfluenceLevel.CRITICAL, StakeholderPosition.STRONGLY_SUPPORT),
            ("Moderate Supporter", InfluenceLevel.MODERATE, StakeholderPosition.SUPPORT),
            ("Neutral", InfluenceLevel.HIGH, StakeholderPosition.NEUTRAL),
            ("Opposer", InfluenceLevel.LOW, StakeholderPosition.OPPOSE)
        ]
        
        for name, influence, position in stakeholder_data:
            member = self.engine.add_board_member(
                process_id=process.id,
                name=name,
                role="Board Member",
                influence_level=influence
            )
            
            self.engine.update_stakeholder_position(
                process_id=process.id,
                stakeholder_id=member.id,
                stakeholder_name=name,
                position=position,
                confidence_level=0.8
            )
        
        # Add some actions
        self.engine.create_consensus_action(
            process_id=process.id,
            title="Completed Action",
            description="Test",
            action_type="meeting",
            target_stakeholders=["member-001"],
            responsible_party="facilitator",
            deadline=datetime.now() + timedelta(days=1),
            expected_impact="Test impact"
        )
        
        # Manually set action as completed for testing
        process.actions[0].status = "completed"
        
        metrics = self.engine.track_consensus_progress(process.id)
        
        assert metrics.support_percentage == 50.0  # 2 out of 4 support
        assert metrics.opposition_percentage == 25.0  # 1 out of 4 oppose
        assert metrics.neutral_percentage == 25.0  # 1 out of 4 neutral
        assert metrics.weighted_support_score > 0.5  # Should be higher due to critical supporter
        assert metrics.stakeholder_engagement_level == 1.0  # 1 completed out of 1 total action
        assert metrics.communication_effectiveness == 0.8  # Average confidence level
        
        # Check that process status was updated
        assert process.current_consensus_level in [ConsensusStatus.IN_PROGRESS, ConsensusStatus.PARTIAL_CONSENSUS]
    
    def test_generate_consensus_recommendation(self):
        """Test generating consensus recommendations"""
        process = self.engine.create_consensus_building(
            title="Test Process",
            description="Test description",
            decision_topic="Test Topic",
            deadline=datetime.now() + timedelta(days=21)
        )
        
        # Add stakeholders
        member1 = self.engine.add_board_member(
            process_id=process.id,
            name="Neutral Member",
            role="Board Member",
            influence_level=InfluenceLevel.HIGH,
            decision_making_style="analytical"
        )
        
        member2 = self.engine.add_board_member(
            process_id=process.id,
            name="Supporter",
            role="Board Member",
            influence_level=InfluenceLevel.MODERATE
        )
        
        # Set positions
        self.engine.update_stakeholder_position(
            process_id=process.id,
            stakeholder_id=member1.id,
            stakeholder_name="Neutral Member",
            position=StakeholderPosition.NEUTRAL,
            requirements_for_support=["More detailed analysis", "Risk mitigation plan"]
        )
        
        self.engine.update_stakeholder_position(
            process_id=process.id,
            stakeholder_id=member2.id,
            stakeholder_name="Supporter",
            position=StakeholderPosition.SUPPORT
        )
        
        # Add barriers and strategies
        self.engine.identify_consensus_barriers(process.id)
        self.engine.develop_consensus_strategy(process.id)
        
        recommendation = self.engine.generate_consensus_recommendation(process.id)
        
        assert recommendation.title.startswith("Consensus Building Recommendation:")
        assert recommendation.recommended_approach in [
            StrategyType.INFORMATION_SHARING,  # Due to neutral stakeholder
            StrategyType.STAKEHOLDER_ENGAGEMENT
        ]
        assert len(recommendation.priority_actions) > 0
        assert len(recommendation.key_stakeholders_to_focus) > 0
        assert "Neutral Member" in recommendation.key_stakeholders_to_focus  # High influence neutral
        assert recommendation.timeline_recommendation  # Should have timeline recommendation
        assert len(recommendation.meeting_recommendations) > 0
        assert len(recommendation.negotiation_points) > 0
        assert recommendation.success_probability > 0
        assert len(recommendation.critical_success_factors) > 0
    
    def test_optimize_consensus_process(self):
        """Test consensus process optimization"""
        process = self.engine.create_consensus_building(
            title="Test Process",
            description="Test description",
            decision_topic="Test Topic"
        )
        
        # Add stakeholders with different styles
        analytical_member = self.engine.add_board_member(
            process_id=process.id,
            name="Analytical Member",
            role="Board Member",
            influence_level=InfluenceLevel.HIGH,
            decision_making_style="analytical"
        )
        
        collaborative_member = self.engine.add_board_member(
            process_id=process.id,
            name="Collaborative Member",
            role="Board Member",
            influence_level=InfluenceLevel.MODERATE,
            decision_making_style="collaborative"
        )
        
        # Set positions
        self.engine.update_stakeholder_position(
            process_id=process.id,
            stakeholder_id=analytical_member.id,
            stakeholder_name="Analytical Member",
            position=StakeholderPosition.NEUTRAL,
            key_concerns=["Data quality", "ROI analysis"]
        )
        
        self.engine.update_stakeholder_position(
            process_id=process.id,
            stakeholder_id=collaborative_member.id,
            stakeholder_name="Collaborative Member",
            position=StakeholderPosition.SUPPORT,
            key_concerns=["Team alignment"]
        )
        
        optimization = self.engine.optimize_consensus_process(process.id)
        
        assert len(optimization.process_improvements) > 0
        assert len(optimization.communication_enhancements) > 0
        assert len(optimization.engagement_strategies) > 0
        
        # Check stakeholder-specific approaches
        assert analytical_member.id in optimization.stakeholder_specific_approaches
        analytical_approaches = optimization.stakeholder_specific_approaches[analytical_member.id]
        assert any("data" in approach.lower() for approach in analytical_approaches)
        
        assert collaborative_member.id in optimization.stakeholder_specific_approaches
        collaborative_approaches = optimization.stakeholder_specific_approaches[collaborative_member.id]
        assert any("involve" in approach.lower() or "input" in approach.lower() for approach in collaborative_approaches)
        
        assert len(optimization.quick_wins) > 0
        assert len(optimization.accelerated_timeline_options) > 0
    
    def test_create_consensus_visualization(self):
        """Test creating consensus visualizations"""
        process = self.engine.create_consensus_building(
            title="Test Process",
            description="Test description",
            decision_topic="Test Topic"
        )
        
        # Add stakeholders for visualization
        for i, (name, influence, position) in enumerate([
            ("High Supporter", InfluenceLevel.HIGH, StakeholderPosition.SUPPORT),
            ("Critical Neutral", InfluenceLevel.CRITICAL, StakeholderPosition.NEUTRAL),
            ("Low Opposer", InfluenceLevel.LOW, StakeholderPosition.OPPOSE)
        ]):
            member = self.engine.add_board_member(
                process_id=process.id,
                name=name,
                role="Board Member",
                influence_level=influence
            )
            
            self.engine.update_stakeholder_position(
                process_id=process.id,
                stakeholder_id=member.id,
                stakeholder_name=name,
                position=position,
                confidence_level=0.7 + i * 0.1,
                key_concerns=[f"Concern {i+1}", f"Concern {i+2}"]
            )
        
        # Test stakeholder map visualization
        viz = self.engine.create_consensus_visualization(
            process_id=process.id,
            visualization_type="stakeholder_map"
        )
        
        assert viz.visualization_type == "stakeholder_map"
        assert viz.title == "Stakeholder Position and Influence Map"
        assert viz.chart_config["type"] == "scatter"
        assert len(viz.chart_config["data"]) == 3  # 3 stakeholders
        assert "Stakeholder analysis of 3 board members" in viz.executive_summary
        
        # Verify data structure
        stakeholder_data = viz.chart_config["data"]
        assert all("name" in item for item in stakeholder_data)
        assert all("position" in item for item in stakeholder_data)
        assert all("influence" in item for item in stakeholder_data)
        assert all("confidence" in item for item in stakeholder_data)
    
    def test_consensus_timeline_visualization(self):
        """Test consensus timeline visualization"""
        process = self.engine.create_consensus_building(
            title="Test Process",
            description="Test description",
            decision_topic="Test Topic"
        )
        
        # Add a stakeholder to have some consensus score
        member = self.engine.add_board_member(
            process_id=process.id,
            name="Test Member",
            role="Board Member",
            influence_level=InfluenceLevel.MODERATE
        )
        
        self.engine.update_stakeholder_position(
            process_id=process.id,
            stakeholder_id=member.id,
            stakeholder_name="Test Member",
            position=StakeholderPosition.SUPPORT
        )
        
        viz = self.engine.create_consensus_visualization(
            process_id=process.id,
            visualization_type="consensus_timeline"
        )
        
        assert viz.visualization_type == "consensus_timeline"
        assert viz.title == "Consensus Building Timeline"
        assert viz.chart_config["type"] == "line"
        assert len(viz.chart_config["data"]) >= 1  # At least initial data point
        assert "Consensus progress from" in viz.executive_summary
    
    def test_influence_network_visualization(self):
        """Test influence network visualization"""
        process = self.engine.create_consensus_building(
            title="Test Process",
            description="Test description",
            decision_topic="Test Topic"
        )
        
        # Add board members
        member1 = self.engine.add_board_member(
            process_id=process.id,
            name="Influencer",
            role="Board Chair",
            influence_level=InfluenceLevel.CRITICAL
        )
        
        member2 = self.engine.add_board_member(
            process_id=process.id,
            name="Member",
            role="Board Member",
            influence_level=InfluenceLevel.MODERATE
        )
        
        # Set up relationship network
        member1.relationship_network = [member2.id]
        
        viz = self.engine.create_consensus_visualization(
            process_id=process.id,
            visualization_type="influence_network"
        )
        
        assert viz.visualization_type == "influence_network"
        assert viz.title == "Board Member Influence Network"
        assert viz.chart_config["type"] == "network"
        assert len(viz.chart_config["data"]) == 2  # 2 board members
        assert "Influence network analysis of 2 board members" in viz.executive_summary
    
    def test_consensus_score_calculation(self):
        """Test consensus score calculation with different influence levels"""
        process = self.engine.create_consensus_building(
            title="Test Process",
            description="Test description",
            decision_topic="Test Topic"
        )
        
        # Add stakeholders with different influence levels and positions
        test_cases = [
            ("Critical Supporter", InfluenceLevel.CRITICAL, StakeholderPosition.STRONGLY_SUPPORT),  # Weight 3.0, Score 1.0
            ("High Neutral", InfluenceLevel.HIGH, StakeholderPosition.NEUTRAL),                     # Weight 2.0, Score 0.5
            ("Moderate Opposer", InfluenceLevel.MODERATE, StakeholderPosition.OPPOSE),              # Weight 1.0, Score 0.25
            ("Low Supporter", InfluenceLevel.LOW, StakeholderPosition.SUPPORT)                      # Weight 0.5, Score 0.75
        ]
        
        for name, influence, position in test_cases:
            member = self.engine.add_board_member(
                process_id=process.id,
                name=name,
                role="Board Member",
                influence_level=influence
            )
            
            self.engine.update_stakeholder_position(
                process_id=process.id,
                stakeholder_id=member.id,
                stakeholder_name=name,
                position=position
            )
        
        # Expected calculation:
        # (3.0 * 1.0 + 2.0 * 0.5 + 1.0 * 0.25 + 0.5 * 0.75) / (3.0 + 2.0 + 1.0 + 0.5)
        # = (3.0 + 1.0 + 0.25 + 0.375) / 6.5 = 4.625 / 6.5 ≈ 0.711
        
        expected_score = (3.0 * 1.0 + 2.0 * 0.5 + 1.0 * 0.25 + 0.5 * 0.75) / (3.0 + 2.0 + 1.0 + 0.5)
        
        assert abs(process.consensus_score - expected_score) < 0.01
    
    def test_error_handling(self):
        """Test error handling in consensus building"""
        # Test invalid process ID
        with pytest.raises(ValueError):
            self.engine.add_board_member(
                process_id="invalid-id",
                name="Test",
                role="Test",
                influence_level=InfluenceLevel.MODERATE
            )
        
        with pytest.raises(ValueError):
            self.engine.update_stakeholder_position(
                process_id="invalid-id",
                stakeholder_id="test",
                stakeholder_name="Test",
                position=StakeholderPosition.NEUTRAL
            )
        
        with pytest.raises(ValueError):
            self.engine.identify_consensus_barriers("invalid-id")
        
        with pytest.raises(ValueError):
            self.engine.develop_consensus_strategy("invalid-id")
        
        with pytest.raises(ValueError):
            self.engine.create_consensus_action(
                process_id="invalid-id",
                title="Test",
                description="Test",
                action_type="test",
                target_stakeholders=[],
                responsible_party="test",
                deadline=datetime.now(),
                expected_impact="test"
            )
        
        with pytest.raises(ValueError):
            self.engine.track_consensus_progress("invalid-id")
        
        with pytest.raises(ValueError):
            self.engine.generate_consensus_recommendation("invalid-id")
        
        with pytest.raises(ValueError):
            self.engine.optimize_consensus_process("invalid-id")
        
        with pytest.raises(ValueError):
            self.engine.create_consensus_visualization("invalid-id")
    
    def test_consensus_templates(self):
        """Test consensus building templates"""
        # Test that templates are properly initialized
        assert "strategic_decision" in self.engine.consensus_templates
        assert "budget_allocation" in self.engine.consensus_templates
        assert "personnel_decision" in self.engine.consensus_templates
        
        # Test strategic decision template
        strategic_template = self.engine.consensus_templates["strategic_decision"]
        assert "typical_barriers" in strategic_template
        assert "recommended_strategies" in strategic_template
        assert len(strategic_template["typical_barriers"]) > 0
        assert len(strategic_template["recommended_strategies"]) > 0
        
        # Test that template barriers are applied
        process = self.engine.create_consensus_building(
            title="Strategic Test",
            description="Test strategic template",
            decision_topic="Strategic Decision"
        )
        
        barriers = self.engine.identify_consensus_barriers(
            process_id=process.id,
            decision_type="strategic_decision"
        )
        
        # Should include template barriers
        barrier_descriptions = [b.description for b in barriers]
        template_descriptions = [b["description"] for b in strategic_template["typical_barriers"]]
        
        # At least some template barriers should be included
        assert any(template_desc in barrier_descriptions for template_desc in template_descriptions)
    
    def test_coalition_map_update(self):
        """Test coalition mapping functionality"""
        process = self.engine.create_consensus_building(
            title="Test Process",
            description="Test description",
            decision_topic="Test Topic"
        )
        
        # Add stakeholders with different positions
        positions = [
            ("Supporter1", StakeholderPosition.SUPPORT),
            ("Supporter2", StakeholderPosition.STRONGLY_SUPPORT),
            ("Neutral1", StakeholderPosition.NEUTRAL),
            ("Opposer1", StakeholderPosition.OPPOSE)
        ]
        
        for name, position in positions:
            member = self.engine.add_board_member(
                process_id=process.id,
                name=name,
                role="Board Member",
                influence_level=InfluenceLevel.MODERATE
            )
            
            self.engine.update_stakeholder_position(
                process_id=process.id,
                stakeholder_id=member.id,
                stakeholder_name=name,
                position=position
            )
        
        # Check coalition map
        coalition_map = process.coalition_map
        
        assert "support" in coalition_map
        assert "strongly_support" in coalition_map
        assert "neutral" in coalition_map
        assert "oppose" in coalition_map
        
        assert len(coalition_map["support"]) == 1
        assert len(coalition_map["strongly_support"]) == 1
        assert len(coalition_map["neutral"]) == 1
        assert len(coalition_map["oppose"]) == 1