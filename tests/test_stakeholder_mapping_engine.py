"""
Tests for Stakeholder Mapping Engine
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from scrollintel.engines.stakeholder_mapping_engine import StakeholderMappingEngine
from scrollintel.models.stakeholder_influence_models import (
    Stakeholder, StakeholderType, InfluenceLevel, CommunicationStyle,
    Background, Priority, Relationship, DecisionPattern
)


class TestStakeholderMappingEngine:
    
    def setup_method(self):
        """Set up test fixtures"""
        self.engine = StakeholderMappingEngine()
        
        # Sample organization context
        self.organization_context = {
            'organization_name': 'Test Corp',
            'board_members': [
                {
                    'id': 'board_001',
                    'name': 'John Smith',
                    'title': 'Board Chair',
                    'industry_experience': ['technology', 'finance'],
                    'expertise': ['strategy', 'governance'],
                    'education': ['MBA Harvard'],
                    'previous_roles': ['CEO TechCorp'],
                    'achievements': ['IPO Success'],
                    'contact_preferences': {'email': 'preferred'}
                }
            ],
            'executives': [
                {
                    'id': 'exec_001',
                    'name': 'Jane Doe',
                    'title': 'CEO',
                    'industry_experience': ['technology'],
                    'expertise': ['operations', 'strategy'],
                    'education': ['BS Engineering'],
                    'previous_roles': ['VP Operations'],
                    'achievements': ['Revenue Growth'],
                    'contact_preferences': {'phone': 'preferred'}
                }
            ],
            'investors': [
                {
                    'id': 'inv_001',
                    'name': 'Investment Fund',
                    'title': 'Lead Investor',
                    'organization': 'VC Fund',
                    'industry_experience': ['technology'],
                    'expertise': ['venture_capital'],
                    'education': ['MBA'],
                    'previous_roles': ['Partner'],
                    'achievements': ['Successful Exits'],
                    'contact_preferences': {'email': 'preferred'}
                }
            ],
            'advisors': []
        }
    
    def test_identify_key_stakeholders(self):
        """Test stakeholder identification functionality"""
        stakeholders = self.engine.identify_key_stakeholders(self.organization_context)
        
        assert len(stakeholders) == 3  # 1 board member + 1 executive + 1 investor
        
        # Check board member
        board_member = next(s for s in stakeholders if s.stakeholder_type == StakeholderType.BOARD_MEMBER)
        assert board_member.name == 'John Smith'
        assert board_member.title == 'Board Chair'
        assert board_member.influence_level == InfluenceLevel.HIGH
        assert len(board_member.priorities) == 3
        
        # Check executive
        executive = next(s for s in stakeholders if s.stakeholder_type == StakeholderType.EXECUTIVE)
        assert executive.name == 'Jane Doe'
        assert executive.title == 'CEO'
        assert executive.communication_style == CommunicationStyle.RESULTS_ORIENTED
        
        # Check investor
        investor = next(s for s in stakeholders if s.stakeholder_type == StakeholderType.INVESTOR)
        assert investor.name == 'Investment Fund'
        assert investor.influence_level == InfluenceLevel.CRITICAL
    
    def test_assess_stakeholder_influence(self):
        """Test stakeholder influence assessment"""
        # Create test stakeholder
        stakeholder = Stakeholder(
            id="test_001",
            name="Test Stakeholder",
            title="Board Chair",
            organization="Test Corp",
            stakeholder_type=StakeholderType.BOARD_MEMBER,
            background=Background(
                industry_experience=['technology', 'finance'],
                functional_expertise=['strategy', 'governance'],
                education=['MBA'],
                previous_roles=['CEO'],
                achievements=['IPO Success']
            ),
            influence_level=InfluenceLevel.HIGH,
            communication_style=CommunicationStyle.ANALYTICAL,
            decision_pattern=DecisionPattern(
                decision_style="consensus_building",
                key_factors=["financial_impact"],
                typical_concerns=["governance"],
                influence_tactics=["data_presentation"]
            ),
            priorities=[],
            relationships=[
                Relationship(
                    stakeholder_id="other_001",
                    relationship_type="professional",
                    strength=0.8,
                    history=["positive_interaction"]
                )
            ],
            contact_preferences={}
        )
        
        context = {'max_relationships': 20}
        assessment = self.engine.assess_stakeholder_influence(stakeholder, context)
        
        assert assessment.stakeholder_id == "test_001"
        assert 0.0 <= assessment.formal_authority <= 1.0
        assert 0.0 <= assessment.informal_influence <= 1.0
        assert 0.0 <= assessment.network_centrality <= 1.0
        assert 0.0 <= assessment.expertise_credibility <= 1.0
        assert 0.0 <= assessment.resource_control <= 1.0
        assert 0.0 <= assessment.overall_influence <= 1.0
        
        # Board chair should have high formal authority
        assert assessment.formal_authority >= 0.9
    
    def test_map_stakeholder_relationships(self):
        """Test stakeholder relationship mapping"""
        stakeholders = self.engine.identify_key_stakeholders(self.organization_context)
        
        # Add some relationships
        stakeholders[0].relationships = [
            Relationship(
                stakeholder_id=stakeholders[1].id,
                relationship_type="professional",
                strength=0.8,
                history=["positive_collaboration"]
            )
        ]
        
        stakeholder_map = self.engine.map_stakeholder_relationships(stakeholders)
        
        assert stakeholder_map.id.startswith("map_")
        assert stakeholder_map.organization_id == "current_org"
        assert len(stakeholder_map.stakeholders) == 3
        assert len(stakeholder_map.influence_networks) >= 1
        
        # Check main network
        main_network = stakeholder_map.influence_networks[0]
        assert main_network.name == "Main Organizational Network"
        assert len(main_network.stakeholders) == 3
        assert len(main_network.power_centers) >= 1  # Should have high influence stakeholders
    
    def test_optimize_stakeholder_relationships(self):
        """Test relationship optimization"""
        stakeholders = self.engine.identify_key_stakeholders(self.organization_context)
        stakeholder_map = self.engine.map_stakeholder_relationships(stakeholders)
        
        objectives = ["growth", "innovation"]
        optimizations = self.engine.optimize_stakeholder_relationships(stakeholder_map, objectives)
        
        assert len(optimizations) == len(stakeholders)
        
        for optimization in optimizations:
            assert optimization.stakeholder_id in [s.id for s in stakeholders]
            assert 0.0 <= optimization.current_relationship_strength <= 1.0
            assert 0.0 <= optimization.target_relationship_strength <= 1.0
            assert optimization.target_relationship_strength >= optimization.current_relationship_strength
            assert len(optimization.optimization_strategies) > 0
            assert len(optimization.action_items) > 0
            assert len(optimization.success_metrics) > 0
            assert "short_term" in optimization.timeline
            assert "medium_term" in optimization.timeline
            assert "long_term" in optimization.timeline
    
    def test_analyze_stakeholder_comprehensive(self):
        """Test comprehensive stakeholder analysis"""
        stakeholders = self.engine.identify_key_stakeholders(self.organization_context)
        stakeholder = stakeholders[0]  # Board member
        
        context = {
            'max_relationships': 20,
            'stakeholder_map': None,
            'objectives': ['growth', 'governance']
        }
        
        analysis = self.engine.analyze_stakeholder_comprehensive(stakeholder, context)
        
        assert analysis.stakeholder_id == stakeholder.id
        assert analysis.influence_assessment is not None
        assert analysis.relationship_optimization is not None
        assert isinstance(analysis.engagement_history, list)
        assert isinstance(analysis.predicted_positions, dict)
        assert isinstance(analysis.engagement_recommendations, list)
        
        # Check influence assessment
        assert 0.0 <= analysis.influence_assessment.overall_influence <= 1.0
        
        # Check predicted positions
        assert len(analysis.predicted_positions) > 0
        
        # Check engagement recommendations
        assert len(analysis.engagement_recommendations) > 0
    
    def test_assess_formal_authority(self):
        """Test formal authority assessment"""
        # Test board member
        board_stakeholder = Stakeholder(
            id="board_001", name="Board Chair", title="Board Chair",
            organization="Test", stakeholder_type=StakeholderType.BOARD_MEMBER,
            background=Background([], [], [], [], []),
            influence_level=InfluenceLevel.HIGH,
            communication_style=CommunicationStyle.ANALYTICAL,
            decision_pattern=DecisionPattern("", [], [], []),
            priorities=[], relationships=[], contact_preferences={}
        )
        
        authority = self.engine._assess_formal_authority(board_stakeholder, {})
        assert authority >= 0.9  # Board chair should have high authority
        
        # Test advisor
        advisor_stakeholder = Stakeholder(
            id="advisor_001", name="Advisor", title="Advisor",
            organization="Test", stakeholder_type=StakeholderType.ADVISOR,
            background=Background([], [], [], [], []),
            influence_level=InfluenceLevel.MEDIUM,
            communication_style=CommunicationStyle.ANALYTICAL,
            decision_pattern=DecisionPattern("", [], [], []),
            priorities=[], relationships=[], contact_preferences={}
        )
        
        authority = self.engine._assess_formal_authority(advisor_stakeholder, {})
        assert authority <= 0.5  # Advisor should have lower authority
    
    def test_calculate_overall_influence(self):
        """Test overall influence calculation"""
        overall = self.engine._calculate_overall_influence(
            formal_authority=0.9,
            informal_influence=0.7,
            network_centrality=0.6,
            expertise_credibility=0.8,
            resource_control=0.8
        )
        
        assert 0.0 <= overall <= 1.0
        assert overall > 0.7  # Should be high given high input values
    
    def test_build_influence_networks(self):
        """Test influence network building"""
        stakeholders = self.engine.identify_key_stakeholders(self.organization_context)
        
        # Add relationships
        stakeholders[0].relationships = [
            Relationship(
                stakeholder_id=stakeholders[1].id,
                relationship_type="professional",
                strength=0.8,
                history=[]
            )
        ]
        
        networks = self.engine._build_influence_networks(stakeholders)
        
        assert len(networks) >= 1
        main_network = networks[0]
        assert main_network.name == "Main Organizational Network"
        assert len(main_network.stakeholders) == len(stakeholders)
        assert len(main_network.power_centers) > 0
    
    def test_identify_influence_clusters(self):
        """Test influence cluster identification"""
        stakeholders = self.engine.identify_key_stakeholders(self.organization_context)
        
        # Create strong relationships to form clusters
        stakeholders[0].relationships = [
            Relationship(
                stakeholder_id=stakeholders[1].id,
                relationship_type="professional",
                strength=0.8,
                history=[]
            )
        ]
        
        clusters = self.engine._identify_influence_clusters(stakeholders)
        
        assert isinstance(clusters, list)
        # Should identify at least one cluster if relationships are strong enough
        if clusters:
            assert all(len(cluster) > 1 for cluster in clusters)
    
    def test_generate_optimization_strategies(self):
        """Test optimization strategy generation"""
        stakeholder = Stakeholder(
            id="test_001", name="Test", title="Test",
            organization="Test", stakeholder_type=StakeholderType.BOARD_MEMBER,
            background=Background([], [], [], [], []),
            influence_level=InfluenceLevel.HIGH,
            communication_style=CommunicationStyle.ANALYTICAL,
            decision_pattern=DecisionPattern("", [], [], []),
            priorities=[], relationships=[], contact_preferences={}
        )
        
        objectives = ["growth", "innovation"]
        strategies = self.engine._generate_optimization_strategies(stakeholder, objectives)
        
        assert len(strategies) > 0
        assert any("data" in strategy.lower() or "analysis" in strategy.lower() for strategy in strategies)
        assert any("growth" in strategy.lower() for strategy in strategies)
    
    def test_predict_stakeholder_positions(self):
        """Test stakeholder position prediction"""
        stakeholder = Stakeholder(
            id="test_001", name="Test", title="Test",
            organization="Test", stakeholder_type=StakeholderType.INVESTOR,
            background=Background([], [], [], [], []),
            influence_level=InfluenceLevel.HIGH,
            communication_style=CommunicationStyle.ANALYTICAL,
            decision_pattern=DecisionPattern("", [], [], []),
            priorities=[], relationships=[], contact_preferences={}
        )
        
        positions = self.engine._predict_stakeholder_positions(stakeholder, {})
        
        assert isinstance(positions, dict)
        assert len(positions) > 0
        # Investor should have positions on financial matters
        assert any("investment" in key or "cost" in key for key in positions.keys())
    
    def test_generate_engagement_recommendations(self):
        """Test engagement recommendation generation"""
        stakeholder = Stakeholder(
            id="test_001", name="Test", title="Test",
            organization="Test", stakeholder_type=StakeholderType.BOARD_MEMBER,
            background=Background([], [], [], [], []),
            influence_level=InfluenceLevel.HIGH,
            communication_style=CommunicationStyle.VISIONARY,
            decision_pattern=DecisionPattern("", [], [], []),
            priorities=[], relationships=[], contact_preferences={}
        )
        
        from scrollintel.models.stakeholder_influence_models import InfluenceAssessment
        influence_assessment = InfluenceAssessment(
            stakeholder_id="test_001",
            formal_authority=0.9,
            informal_influence=0.8,
            network_centrality=0.7,
            expertise_credibility=0.8,
            resource_control=0.8,
            overall_influence=0.8
        )
        
        recommendations = self.engine._generate_engagement_recommendations(
            stakeholder, influence_assessment, {}
        )
        
        assert len(recommendations) > 0
        assert any("high-frequency" in rec.lower() or "strategic" in rec.lower() for rec in recommendations)
        assert any("vision" in rec.lower() for rec in recommendations)  # Visionary communication style


if __name__ == "__main__":
    pytest.main([__file__])