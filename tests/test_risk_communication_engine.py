"""
Tests for Risk Communication Engine

Tests for risk communication, visualization, and effectiveness measurement.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from scrollintel.engines.risk_communication_engine import (
    RiskCommunicationEngine,
    Risk,
    RiskCategory,
    RiskLevel,
    ImpactType,
    CommunicationAudience,
    RiskImpact,
    MitigationStrategy,
    RiskScenario,
    RiskCommunication,
    RiskVisualization
)

class TestRiskCommunicationEngine:
    
    @pytest.fixture
    def engine(self):
        """Create a risk communication engine instance"""
        return RiskCommunicationEngine()
    
    @pytest.fixture
    def sample_risk_impact(self):
        """Create a sample risk impact"""
        return RiskImpact(
            impact_type=ImpactType.REVENUE_LOSS,
            probability=0.3,
            financial_impact=500000,
            timeline="Q2 2024",
            description="Potential revenue loss from market disruption",
            affected_stakeholders=["customers", "investors", "employees"]
        )
    
    @pytest.fixture
    def sample_mitigation_strategy(self):
        """Create a sample mitigation strategy"""
        return MitigationStrategy(
            strategy_id="mit_001",
            title="Market Diversification",
            description="Expand into new market segments to reduce dependency",
            implementation_cost=200000,
            implementation_timeline="6 months",
            effectiveness_rating=0.8,
            responsible_party="VP Marketing",
            success_metrics=["Market share increase", "Revenue diversification"],
            dependencies=["Budget approval", "Team hiring"]
        )
    
    @pytest.fixture
    def sample_risk_scenario(self):
        """Create a sample risk scenario"""
        return RiskScenario(
            scenario_id="scenario_001",
            title="Market Downturn Scenario",
            description="Economic recession impacts customer demand",
            probability=0.25,
            potential_impacts=[],  # Will be populated in tests
            trigger_events=["Economic indicators decline", "Customer spending drops"],
            early_warning_indicators=["Leading economic indicators", "Customer survey results"]
        )
    
    @pytest.fixture
    def sample_risk(self, sample_risk_impact, sample_mitigation_strategy, sample_risk_scenario):
        """Create a sample risk"""
        sample_risk_scenario.potential_impacts = [sample_risk_impact]
        
        return Risk(
            risk_id="risk_001",
            title="Market Disruption Risk",
            description="Risk of market disruption affecting business operations",
            category=RiskCategory.MARKET,
            risk_level=RiskLevel.HIGH,
            probability=0.4,
            potential_impacts=[sample_risk_impact],
            current_controls=["Market monitoring", "Competitive analysis"],
            mitigation_strategies=[sample_mitigation_strategy],
            risk_scenarios=[sample_risk_scenario],
            risk_owner="Chief Strategy Officer",
            last_assessed=datetime.now(),
            next_review_date=datetime.now() + timedelta(days=90)
        )
    
    def test_engine_initialization(self, engine):
        """Test engine initialization"""
        assert engine is not None
        assert len(engine.risks) == 0
        assert len(engine.communications) == 0
        assert len(engine.visualizations) == 0
        assert len(engine.communication_templates) > 0
        assert CommunicationAudience.BOARD.value in engine.communication_templates
    
    def test_add_risk(self, engine, sample_risk):
        """Test adding risk to engine"""
        engine.add_risk(sample_risk)
        
        assert len(engine.risks) == 1
        assert engine.risks[0].risk_id == "risk_001"
        assert engine.risks[0].title == "Market Disruption Risk"
    
    def test_create_risk_communication_board(self, engine, sample_risk):
        """Test creating risk communication for board"""
        engine.add_risk(sample_risk)
        
        communication = engine.create_risk_communication(
            risk_id="risk_001",
            audience=CommunicationAudience.BOARD,
            communication_type="presentation"
        )
        
        assert communication is not None
        assert communication.risk_id == "risk_001"
        assert communication.audience == CommunicationAudience.BOARD
        assert communication.communication_type == "presentation"
        assert len(communication.key_messages) > 0
        assert len(communication.visual_elements) > 0
        assert len(communication.action_items) > 0
        assert len(engine.communications) == 1
    
    def test_create_risk_communication_executive(self, engine, sample_risk):
        """Test creating risk communication for executive team"""
        engine.add_risk(sample_risk)
        
        communication = engine.create_risk_communication(
            risk_id="risk_001",
            audience=CommunicationAudience.EXECUTIVE_TEAM,
            communication_type="briefing"
        )
        
        assert communication.audience == CommunicationAudience.EXECUTIVE_TEAM
        assert len(communication.key_messages) > 0
        
        # Executive messages should focus on operational details
        messages_text = " ".join(communication.key_messages).lower()
        assert any(keyword in messages_text for keyword in ["operational", "implementation", "resource"])
    
    def test_create_risk_communication_investors(self, engine, sample_risk):
        """Test creating risk communication for investors"""
        engine.add_risk(sample_risk)
        
        communication = engine.create_risk_communication(
            risk_id="risk_001",
            audience=CommunicationAudience.INVESTORS,
            communication_type="report"
        )
        
        assert communication.audience == CommunicationAudience.INVESTORS
        
        # Investor messages should focus on financial impact
        messages_text = " ".join(communication.key_messages).lower()
        assert any(keyword in messages_text for keyword in ["financial", "revenue", "value", "market"])
    
    def test_create_risk_communication_regulators(self, engine, sample_risk):
        """Test creating risk communication for regulators"""
        # Create regulatory risk
        regulatory_risk = Risk(
            risk_id="reg_risk_001",
            title="Compliance Risk",
            description="Risk of regulatory non-compliance",
            category=RiskCategory.REGULATORY,
            risk_level=RiskLevel.MEDIUM,
            probability=0.2,
            potential_impacts=[],
            current_controls=["Compliance monitoring", "Legal review"],
            mitigation_strategies=[],
            risk_scenarios=[],
            risk_owner="Chief Compliance Officer",
            last_assessed=datetime.now(),
            next_review_date=datetime.now() + timedelta(days=60)
        )
        
        engine.add_risk(regulatory_risk)
        
        communication = engine.create_risk_communication(
            risk_id="reg_risk_001",
            audience=CommunicationAudience.REGULATORS,
            communication_type="compliance_report"
        )
        
        assert communication.audience == CommunicationAudience.REGULATORS
        
        # Regulator messages should focus on compliance
        messages_text = " ".join(communication.key_messages).lower()
        assert any(keyword in messages_text for keyword in ["compliance", "regulatory", "control"])
    
    def test_board_message_generation(self, engine, sample_risk):
        """Test board-specific message generation"""
        engine.add_risk(sample_risk)
        
        communication = engine.create_risk_communication(
            risk_id="risk_001",
            audience=CommunicationAudience.BOARD,
            communication_type="presentation"
        )
        
        messages = communication.key_messages
        assert len(messages) >= 3
        
        # Should include strategic impact
        assert any("strategic" in msg.lower() or "risk alert" in msg.lower() for msg in messages)
        
        # Should include financial impact
        assert any("$500,000" in msg or "financial impact" in msg.lower() for msg in messages)
        
        # Should include mitigation overview
        assert any("mitigation" in msg.lower() for msg in messages)
        
        # Should include board action
        assert any("board action" in msg.lower() or "review" in msg.lower() for msg in messages)
    
    def test_visual_elements_creation(self, engine, sample_risk):
        """Test visual elements creation"""
        engine.add_risk(sample_risk)
        
        communication = engine.create_risk_communication(
            risk_id="risk_001",
            audience=CommunicationAudience.BOARD,
            communication_type="presentation"
        )
        
        visual_elements = communication.visual_elements
        assert len(visual_elements) > 0
        
        # Should include risk indicator
        risk_indicator = next((v for v in visual_elements if v['type'] == 'risk_indicator'), None)
        assert risk_indicator is not None
        assert risk_indicator['data']['risk_level'] == 'high'
        assert risk_indicator['data']['probability'] == 0.4
        
        # Should include impact chart
        impact_chart = next((v for v in visual_elements if v['type'] == 'impact_chart'), None)
        assert impact_chart is not None
        assert len(impact_chart['data']['impacts']) > 0
        
        # Should include mitigation timeline
        mitigation_timeline = next((v for v in visual_elements if v['type'] == 'mitigation_timeline'), None)
        assert mitigation_timeline is not None
        assert len(mitigation_timeline['data']['strategies']) > 0
    
    def test_action_items_generation(self, engine, sample_risk):
        """Test action items generation"""
        engine.add_risk(sample_risk)
        
        # Test board action items
        board_comm = engine.create_risk_communication(
            risk_id="risk_001",
            audience=CommunicationAudience.BOARD,
            communication_type="presentation"
        )
        
        board_actions = board_comm.action_items
        assert len(board_actions) > 0
        assert any("review" in action.lower() and "approve" in action.lower() for action in board_actions)
        assert any("budget" in action.lower() for action in board_actions)
        
        # Test executive action items
        exec_comm = engine.create_risk_communication(
            risk_id="risk_001",
            audience=CommunicationAudience.EXECUTIVE_TEAM,
            communication_type="briefing"
        )
        
        exec_actions = exec_comm.action_items
        assert any("implement" in action.lower() for action in exec_actions)
        assert any("monitor" in action.lower() for action in exec_actions)
    
    def test_create_risk_visualization_heatmap(self, engine):
        """Test creating risk heatmap visualization"""
        risk_data = {
            'risks': [
                {'id': 'r1', 'probability': 0.8, 'impact': 0.9, 'title': 'High Risk'},
                {'id': 'r2', 'probability': 0.3, 'impact': 0.4, 'title': 'Low Risk'},
                {'id': 'r3', 'probability': 0.6, 'impact': 0.7, 'title': 'Medium Risk'}
            ]
        }
        
        visualization = engine.create_risk_visualization(
            visualization_type="heatmap",
            risk_data=risk_data,
            audience=CommunicationAudience.BOARD
        )
        
        assert visualization is not None
        assert visualization.visualization_type == "heatmap"
        assert visualization.audience == CommunicationAudience.BOARD
        assert visualization.effectiveness_score > 0
        assert 'x_axis' in visualization.visual_config
        assert 'y_axis' in visualization.visual_config
        assert len(engine.visualizations) == 1
    
    def test_create_risk_visualization_timeline(self, engine):
        """Test creating risk timeline visualization"""
        risk_data = {
            'mitigation_strategies': [
                {'title': 'Strategy 1', 'timeline': '3 months', 'milestones': ['M1', 'M2']},
                {'title': 'Strategy 2', 'timeline': '6 months', 'milestones': ['M3', 'M4']}
            ]
        }
        
        visualization = engine.create_risk_visualization(
            visualization_type="timeline",
            risk_data=risk_data,
            audience=CommunicationAudience.EXECUTIVE_TEAM
        )
        
        assert visualization.visualization_type == "timeline"
        assert visualization.audience == CommunicationAudience.EXECUTIVE_TEAM
        assert 'time_axis' in visualization.visual_config
        assert 'events' in visualization.visual_config
    
    def test_visualization_effectiveness_calculation(self, engine):
        """Test visualization effectiveness calculation"""
        risk_data = {'risks': [{'id': 'r1'}, {'id': 'r2'}, {'id': 'r3'}]}
        
        # Test different visualization types for board
        heatmap_viz = engine.create_risk_visualization(
            "heatmap", risk_data, CommunicationAudience.BOARD
        )
        
        timeline_viz = engine.create_risk_visualization(
            "timeline", risk_data, CommunicationAudience.EXECUTIVE_TEAM
        )
        
        # Heatmap should be more effective for board
        assert heatmap_viz.effectiveness_score >= 0.8
        
        # Timeline should be very effective for executive team
        assert timeline_viz.effectiveness_score >= 0.9
    
    def test_measure_communication_effectiveness(self, engine, sample_risk):
        """Test measuring communication effectiveness"""
        engine.add_risk(sample_risk)
        
        communication = engine.create_risk_communication(
            risk_id="risk_001",
            audience=CommunicationAudience.BOARD,
            communication_type="presentation"
        )
        
        # Simulate feedback
        feedback_data = {
            'clarity_rating': 4,  # 1-5 scale
            'usefulness_rating': 5,
            'action_taken': True,
            'understanding_rating': 4,
            'decision_time_hours': 2
        }
        
        effectiveness_score = engine.measure_communication_effectiveness(
            communication.communication_id,
            feedback_data
        )
        
        assert 0 <= effectiveness_score <= 1
        assert effectiveness_score > 0.7  # Should be high with good feedback
        assert communication.communication_effectiveness == effectiveness_score
        assert communication.communication_id in engine.effectiveness_metrics
    
    def test_measure_communication_effectiveness_poor_feedback(self, engine, sample_risk):
        """Test measuring communication effectiveness with poor feedback"""
        engine.add_risk(sample_risk)
        
        communication = engine.create_risk_communication(
            risk_id="risk_001",
            audience=CommunicationAudience.BOARD,
            communication_type="presentation"
        )
        
        # Simulate poor feedback
        feedback_data = {
            'clarity_rating': 2,
            'usefulness_rating': 2,
            'action_taken': False,
            'understanding_rating': 2,
            'decision_time_hours': 48
        }
        
        effectiveness_score = engine.measure_communication_effectiveness(
            communication.communication_id,
            feedback_data
        )
        
        assert effectiveness_score < 0.5  # Should be low with poor feedback
    
    def test_get_risk_communication_analytics_empty(self, engine):
        """Test analytics with no communications"""
        analytics = engine.get_risk_communication_analytics()
        
        assert analytics['total_communications'] == 0
        assert analytics['average_effectiveness'] == 0.0
        assert analytics['effectiveness_by_audience'] == {}
        assert analytics['effectiveness_by_type'] == {}
        assert len(analytics['improvement_recommendations']) == 0
    
    def test_get_risk_communication_analytics_with_data(self, engine, sample_risk):
        """Test analytics with communication data"""
        engine.add_risk(sample_risk)
        
        # Create multiple communications
        board_comm = engine.create_risk_communication(
            "risk_001", CommunicationAudience.BOARD, "presentation"
        )
        exec_comm = engine.create_risk_communication(
            "risk_001", CommunicationAudience.EXECUTIVE_TEAM, "briefing"
        )
        
        # Add effectiveness scores
        engine.measure_communication_effectiveness(
            board_comm.communication_id,
            {'clarity_rating': 4, 'usefulness_rating': 4, 'action_taken': True}
        )
        engine.measure_communication_effectiveness(
            exec_comm.communication_id,
            {'clarity_rating': 3, 'usefulness_rating': 3, 'action_taken': False}
        )
        
        analytics = engine.get_risk_communication_analytics()
        
        assert analytics['total_communications'] == 2
        assert analytics['communications_measured'] == 2
        assert analytics['average_effectiveness'] > 0
        assert CommunicationAudience.BOARD.value in analytics['effectiveness_by_audience']
        assert CommunicationAudience.EXECUTIVE_TEAM.value in analytics['effectiveness_by_audience']
        assert 'presentation' in analytics['effectiveness_by_type']
        assert 'briefing' in analytics['effectiveness_by_type']
    
    def test_generate_risk_communication_report(self, engine, sample_risk):
        """Test generating comprehensive risk communication report"""
        engine.add_risk(sample_risk)
        
        report = engine.generate_risk_communication_report(
            risk_id="risk_001",
            audience=CommunicationAudience.BOARD
        )
        
        assert 'risk_overview' in report
        assert 'impact_analysis' in report
        assert 'mitigation_strategies' in report
        assert 'communication_details' in report
        assert 'recommendations' in report
        assert 'next_steps' in report
        
        # Check risk overview
        risk_overview = report['risk_overview']
        assert risk_overview['title'] == "Market Disruption Risk"
        assert risk_overview['category'] == "market"
        assert risk_overview['level'] == "high"
        assert risk_overview['owner'] == "Chief Strategy Officer"
        
        # Check impact analysis
        impact_analysis = report['impact_analysis']
        assert len(impact_analysis) > 0
        assert impact_analysis[0]['type'] == "revenue_loss"
        assert "$500,000" in impact_analysis[0]['financial_impact']
        
        # Check mitigation strategies
        mitigation_strategies = report['mitigation_strategies']
        assert len(mitigation_strategies) > 0
        assert mitigation_strategies[0]['title'] == "Market Diversification"
        assert "$200,000" in mitigation_strategies[0]['cost']
    
    def test_invalid_risk_id_error(self, engine):
        """Test error handling for invalid risk ID"""
        with pytest.raises(ValueError, match="Risk invalid_id not found"):
            engine.create_risk_communication(
                risk_id="invalid_id",
                audience=CommunicationAudience.BOARD,
                communication_type="presentation"
            )
        
        with pytest.raises(ValueError, match="Risk invalid_id not found"):
            engine.generate_risk_communication_report(
                risk_id="invalid_id",
                audience=CommunicationAudience.BOARD
            )
    
    def test_invalid_communication_id_error(self, engine):
        """Test error handling for invalid communication ID"""
        with pytest.raises(ValueError, match="Communication invalid_id not found"):
            engine.measure_communication_effectiveness(
                communication_id="invalid_id",
                feedback_data={'clarity_rating': 4}
            )
    
    def test_custom_messages_communication(self, engine, sample_risk):
        """Test creating communication with custom messages"""
        engine.add_risk(sample_risk)
        
        custom_messages = [
            "Custom message 1 about the risk",
            "Custom message 2 with specific details",
            "Custom message 3 for action items"
        ]
        
        communication = engine.create_risk_communication(
            risk_id="risk_001",
            audience=CommunicationAudience.BOARD,
            communication_type="presentation",
            custom_messages=custom_messages
        )
        
        assert communication.key_messages == custom_messages
    
    def test_risk_color_scheme(self, engine):
        """Test risk color scheme generation"""
        # Test different risk levels
        critical_colors = engine._get_risk_color_scheme(RiskLevel.CRITICAL)
        assert critical_colors['primary'] == '#DC2626'  # Red
        
        low_colors = engine._get_risk_color_scheme(RiskLevel.LOW)
        assert low_colors['primary'] == '#65A30D'  # Green
        
        medium_colors = engine._get_risk_color_scheme(RiskLevel.MEDIUM)
        assert medium_colors['primary'] == '#D97706'  # Orange
    
    def test_communication_trends_analysis(self, engine, sample_risk):
        """Test communication trends analysis"""
        engine.add_risk(sample_risk)
        
        # Create communications with different effectiveness scores
        comm1 = engine.create_risk_communication("risk_001", CommunicationAudience.BOARD, "presentation")
        comm2 = engine.create_risk_communication("risk_001", CommunicationAudience.EXECUTIVE_TEAM, "briefing")
        
        # Simulate different effectiveness scores over time
        engine.measure_communication_effectiveness(comm1.communication_id, {'clarity_rating': 3, 'action_taken': False})
        engine.measure_communication_effectiveness(comm2.communication_id, {'clarity_rating': 5, 'action_taken': True})
        
        analytics = engine.get_risk_communication_analytics()
        trends = analytics['communication_trends']
        
        assert 'trend' in trends
        assert 'analysis' in trends
        assert trends['trend'] in ['improving', 'declining', 'stable', 'insufficient_data', 'insufficient_scored_data']
    
    def test_improvement_recommendations_generation(self, engine, sample_risk):
        """Test improvement recommendations generation"""
        engine.add_risk(sample_risk)
        
        # Create communications with poor effectiveness
        comm = engine.create_risk_communication("risk_001", CommunicationAudience.BOARD, "presentation")
        engine.measure_communication_effectiveness(
            comm.communication_id,
            {'clarity_rating': 2, 'usefulness_rating': 2, 'action_taken': False}
        )
        
        analytics = engine.get_risk_communication_analytics()
        recommendations = analytics['improvement_recommendations']
        
        assert len(recommendations) > 0
        assert any("effectiveness" in rec.lower() for rec in recommendations)
    
    def test_multiple_risk_scenarios(self, engine):
        """Test handling multiple risk scenarios"""
        # Create risk with multiple scenarios
        scenario1 = RiskScenario(
            scenario_id="s1",
            title="Scenario 1",
            description="First scenario",
            probability=0.3,
            potential_impacts=[],
            trigger_events=["Event 1"],
            early_warning_indicators=["Indicator 1"]
        )
        
        scenario2 = RiskScenario(
            scenario_id="s2",
            title="Scenario 2", 
            description="Second scenario",
            probability=0.2,
            potential_impacts=[],
            trigger_events=["Event 2"],
            early_warning_indicators=["Indicator 2"]
        )
        
        risk = Risk(
            risk_id="multi_scenario_risk",
            title="Multi-Scenario Risk",
            description="Risk with multiple scenarios",
            category=RiskCategory.STRATEGIC,
            risk_level=RiskLevel.HIGH,
            probability=0.5,
            potential_impacts=[],
            current_controls=[],
            mitigation_strategies=[],
            risk_scenarios=[scenario1, scenario2],
            risk_owner="Risk Manager",
            last_assessed=datetime.now(),
            next_review_date=datetime.now() + timedelta(days=30)
        )
        
        engine.add_risk(risk)
        
        communication = engine.create_risk_communication(
            "multi_scenario_risk",
            CommunicationAudience.EXECUTIVE_TEAM,
            "briefing"
        )
        
        # Should include scenario analysis in visual elements
        scenario_viz = next(
            (v for v in communication.visual_elements if v['type'] == 'scenario_analysis'),
            None
        )
        assert scenario_viz is not None
        assert len(scenario_viz['data']['scenarios']) == 2