"""
Tests for Decision Analysis Engine

This module tests the decision analysis and recommendation capabilities
for executive-level decision making.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from scrollintel.engines.decision_analysis_engine import DecisionAnalysisEngine
from scrollintel.models.decision_analysis_models import (
    DecisionType, DecisionUrgency, DecisionComplexity, ImpactLevel
)


class TestDecisionAnalysisEngine:
    """Test cases for DecisionAnalysisEngine"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.engine = DecisionAnalysisEngine()
    
    def test_create_decision_analysis(self):
        """Test creating a new decision analysis"""
        analysis = self.engine.create_decision_analysis(
            title="Strategic Technology Investment",
            description="Evaluate cloud infrastructure options",
            decision_type=DecisionType.STRATEGIC,
            urgency=DecisionUrgency.HIGH,
            complexity=DecisionComplexity.COMPLEX,
            background="Current infrastructure reaching capacity",
            decision_drivers=["Cost reduction", "Scalability", "Performance"],
            constraints=["Budget limit", "Timeline constraints"]
        )
        
        assert analysis.title == "Strategic Technology Investment"
        assert analysis.decision_type == DecisionType.STRATEGIC
        assert analysis.urgency == DecisionUrgency.HIGH
        assert analysis.complexity == DecisionComplexity.COMPLEX
        assert len(analysis.decision_drivers) == 3
        assert len(analysis.constraints) == 2
        assert len(analysis.criteria) > 0  # Should have template criteria
        assert analysis.id in self.engine.decision_analyses
    
    def test_add_decision_option(self):
        """Test adding decision options"""
        analysis = self.engine.create_decision_analysis(
            title="Test Decision",
            description="Test description",
            decision_type=DecisionType.TECHNOLOGY
        )
        
        option = self.engine.add_decision_option(
            analysis_id=analysis.id,
            title="Option A: Cloud Migration",
            description="Migrate to AWS cloud infrastructure",
            pros=["Cost savings", "Scalability", "Reliability"],
            cons=["Migration complexity", "Learning curve"],
            estimated_cost=500000.0,
            estimated_timeline="6 months",
            success_probability=0.8
        )
        
        assert option.title == "Option A: Cloud Migration"
        assert len(option.pros) == 3
        assert len(option.cons) == 2
        assert option.estimated_cost == 500000.0
        assert option.success_probability == 0.8
        assert len(analysis.options) == 1
    
    def test_score_option_criteria(self):
        """Test scoring options against criteria"""
        analysis = self.engine.create_decision_analysis(
            title="Test Decision",
            description="Test description",
            decision_type=DecisionType.TECHNOLOGY
        )
        
        option = self.engine.add_decision_option(
            analysis_id=analysis.id,
            title="Test Option",
            description="Test option description",
            pros=["Benefit 1"],
            cons=["Drawback 1"]
        )
        
        # Score the option
        criteria_scores = {}
        for criteria in analysis.criteria:
            criteria_scores[criteria.id] = 0.8
        
        success = self.engine.score_option_criteria(
            analysis_id=analysis.id,
            option_id=option.id,
            criteria_scores=criteria_scores
        )
        
        assert success
        for criteria_id, score in criteria_scores.items():
            assert option.criteria_scores[criteria_id] == score
    
    def test_add_stakeholder_impact(self):
        """Test adding stakeholder impact analysis"""
        analysis = self.engine.create_decision_analysis(
            title="Test Decision",
            description="Test description",
            decision_type=DecisionType.STRATEGIC
        )
        
        impact = self.engine.add_stakeholder_impact(
            analysis_id=analysis.id,
            stakeholder_id="board-001",
            stakeholder_name="Board of Directors",
            impact_level=ImpactLevel.HIGH,
            impact_description="Significant strategic implications",
            support_likelihood=0.7,
            concerns=["Budget impact", "Timeline concerns"],
            mitigation_strategies=["Phased implementation", "Regular updates"]
        )
        
        assert impact.stakeholder_name == "Board of Directors"
        assert impact.impact_level == ImpactLevel.HIGH
        assert impact.support_likelihood == 0.7
        assert len(impact.concerns) == 2
        assert len(impact.mitigation_strategies) == 2
        assert len(analysis.stakeholder_impacts) == 1
    
    def test_add_risk_assessment(self):
        """Test adding risk assessments"""
        analysis = self.engine.create_decision_analysis(
            title="Test Decision",
            description="Test description",
            decision_type=DecisionType.TECHNOLOGY
        )
        
        risk = self.engine.add_risk_assessment(
            analysis_id=analysis.id,
            risk_category="Technical Risk",
            probability=0.3,
            impact=ImpactLevel.HIGH,
            description="Potential system integration failures",
            mitigation_strategies=["Thorough testing", "Rollback plan"],
            contingency_plans=["Alternative solution", "Extended timeline"]
        )
        
        assert risk.risk_category == "Technical Risk"
        assert risk.probability == 0.3
        assert risk.impact == ImpactLevel.HIGH
        assert len(risk.mitigation_strategies) == 2
        assert len(risk.contingency_plans) == 2
        assert len(analysis.risk_assessments) == 1
    
    def test_calculate_option_scores(self):
        """Test calculating weighted option scores"""
        analysis = self.engine.create_decision_analysis(
            title="Test Decision",
            description="Test description",
            decision_type=DecisionType.TECHNOLOGY
        )
        
        # Add two options
        option1 = self.engine.add_decision_option(
            analysis_id=analysis.id,
            title="Option 1",
            description="First option",
            pros=["Pro 1"],
            cons=["Con 1"]
        )
        
        option2 = self.engine.add_decision_option(
            analysis_id=analysis.id,
            title="Option 2",
            description="Second option",
            pros=["Pro 2"],
            cons=["Con 2"]
        )
        
        # Score both options
        for criteria in analysis.criteria:
            self.engine.score_option_criteria(
                analysis_id=analysis.id,
                option_id=option1.id,
                criteria_scores={criteria.id: 0.8}
            )
            self.engine.score_option_criteria(
                analysis_id=analysis.id,
                option_id=option2.id,
                criteria_scores={criteria.id: 0.6}
            )
        
        scores = self.engine.calculate_option_scores(analysis.id)
        
        assert len(scores) == 2
        assert option1.id in scores
        assert option2.id in scores
        assert scores[option1.id] > scores[option2.id]  # Option 1 should score higher
    
    def test_generate_recommendation(self):
        """Test generating executive recommendations"""
        analysis = self.engine.create_decision_analysis(
            title="Strategic Investment Decision",
            description="Choose investment strategy",
            decision_type=DecisionType.STRATEGIC,
            decision_drivers=["Growth", "Profitability", "Market position"]
        )
        
        # Add option
        option = self.engine.add_decision_option(
            analysis_id=analysis.id,
            title="Recommended Option",
            description="Best strategic choice",
            pros=["High ROI", "Market advantage", "Scalable"],
            cons=["High initial cost"],
            estimated_cost=1000000.0,
            estimated_timeline="12 months",
            success_probability=0.85,
            expected_outcome="Market leadership position"
        )
        
        # Score the option
        for criteria in analysis.criteria:
            self.engine.score_option_criteria(
                analysis_id=analysis.id,
                option_id=option.id,
                criteria_scores={criteria.id: 0.9}
            )
        
        # Add stakeholder and risk for comprehensive recommendation
        self.engine.add_stakeholder_impact(
            analysis_id=analysis.id,
            stakeholder_id="board-001",
            stakeholder_name="Board",
            impact_level=ImpactLevel.HIGH,
            impact_description="Strategic impact",
            support_likelihood=0.8
        )
        
        self.engine.add_risk_assessment(
            analysis_id=analysis.id,
            risk_category="Financial Risk",
            probability=0.2,
            impact=ImpactLevel.MODERATE,
            description="Budget overrun risk"
        )
        
        recommendation = self.engine.generate_recommendation(analysis.id)
        
        assert recommendation.recommended_action == "Recommended Option"
        assert recommendation.success_probability == 0.85
        assert recommendation.confidence_level > 0.8  # Should be high due to good scores
        assert len(recommendation.key_benefits) > 0
        assert len(recommendation.next_steps) > 0
        assert len(recommendation.approval_requirements) > 0
        assert "Executive Summary" in recommendation.executive_summary
    
    def test_assess_decision_impact(self):
        """Test comprehensive decision impact assessment"""
        analysis = self.engine.create_decision_analysis(
            title="Impact Assessment Test",
            description="Test impact assessment",
            decision_type=DecisionType.STRATEGIC
        )
        
        option = self.engine.add_decision_option(
            analysis_id=analysis.id,
            title="Test Option",
            description="Option for impact assessment",
            pros=["Benefit"],
            cons=["Cost"],
            estimated_cost=2000000.0,
            success_probability=0.75
        )
        
        # Score and recommend
        for criteria in analysis.criteria:
            self.engine.score_option_criteria(
                analysis_id=analysis.id,
                option_id=option.id,
                criteria_scores={criteria.id: 0.8}
            )
        
        self.engine.generate_recommendation(analysis.id)
        
        # Add stakeholder for board support calculation
        self.engine.add_stakeholder_impact(
            analysis_id=analysis.id,
            stakeholder_id="board-001",
            stakeholder_name="Board Members",
            impact_level=ImpactLevel.HIGH,
            impact_description="Strategic decision impact",
            support_likelihood=0.9
        )
        
        impact_assessment = self.engine.assess_decision_impact(analysis.id)
        
        assert impact_assessment.financial_impact["cost"] == 2000000.0
        assert impact_assessment.strategic_alignment > 0
        assert impact_assessment.competitive_advantage == 0.75  # Same as success probability
        assert impact_assessment.board_support_likelihood == 0.9
        assert impact_assessment.overall_risk_level in [ImpactLevel.MODERATE, ImpactLevel.HIGH]
    
    def test_optimize_decision(self):
        """Test decision optimization recommendations"""
        analysis = self.engine.create_decision_analysis(
            title="Optimization Test",
            description="Test optimization",
            decision_type=DecisionType.TECHNOLOGY
        )
        
        option = self.engine.add_decision_option(
            analysis_id=analysis.id,
            title="Optimization Target",
            description="Option to optimize",
            pros=["Good performance", "Reliable"],
            cons=["High cost", "Complex implementation", "Long timeline"],
            success_probability=0.6
        )
        
        # Add risks for optimization
        self.engine.add_risk_assessment(
            analysis_id=analysis.id,
            risk_category="Implementation Risk",
            probability=0.4,
            impact=ImpactLevel.HIGH,
            description="Complex implementation challenges",
            mitigation_strategies=["Phased approach"]
        )
        
        optimization = self.engine.optimize_decision(analysis.id, option.id)
        
        assert len(optimization.optimization_suggestions) > 0
        assert len(optimization.enhanced_benefits) > 0
        assert len(optimization.risk_reductions) > 0
        assert len(optimization.cost_optimizations) > 0
        assert optimization.optimized_success_probability > option.success_probability
        assert optimization.optimization_confidence > 0
    
    def test_create_decision_visualization(self):
        """Test creating decision visualizations"""
        analysis = self.engine.create_decision_analysis(
            title="Visualization Test",
            description="Test visualization creation",
            decision_type=DecisionType.STRATEGIC
        )
        
        # Add options for visualization
        for i in range(3):
            option = self.engine.add_decision_option(
                analysis_id=analysis.id,
                title=f"Option {i+1}",
                description=f"Option {i+1} description",
                pros=[f"Pro {i+1}"],
                cons=[f"Con {i+1}"]
            )
            
            # Score options
            for criteria in analysis.criteria:
                self.engine.score_option_criteria(
                    analysis_id=analysis.id,
                    option_id=option.id,
                    criteria_scores={criteria.id: 0.5 + i * 0.2}
                )
        
        # Test comparison matrix visualization
        viz = self.engine.create_decision_visualization(
            analysis_id=analysis.id,
            visualization_type="comparison_matrix"
        )
        
        assert viz.visualization_type == "comparison_matrix"
        assert viz.title == "Decision Options Comparison Matrix"
        assert viz.chart_config["type"] == "heatmap"
        assert len(viz.chart_config["data"]) == 3  # 3 options
        assert "Comparison of 3 options" in viz.executive_summary
    
    def test_risk_impact_visualization(self):
        """Test risk impact visualization"""
        analysis = self.engine.create_decision_analysis(
            title="Risk Visualization Test",
            description="Test risk visualization",
            decision_type=DecisionType.STRATEGIC
        )
        
        # Add risks
        self.engine.add_risk_assessment(
            analysis_id=analysis.id,
            risk_category="Financial Risk",
            probability=0.3,
            impact=ImpactLevel.HIGH,
            description="Budget risk"
        )
        
        self.engine.add_risk_assessment(
            analysis_id=analysis.id,
            risk_category="Technical Risk",
            probability=0.6,
            impact=ImpactLevel.MODERATE,
            description="Implementation risk"
        )
        
        viz = self.engine.create_decision_visualization(
            analysis_id=analysis.id,
            visualization_type="risk_impact"
        )
        
        assert viz.visualization_type == "risk_impact"
        assert viz.title == "Risk Impact Assessment"
        assert viz.chart_config["type"] == "scatter"
        assert len(viz.chart_config["data"]) == 2  # 2 risks
        assert "Analysis of 2 identified risks" in viz.executive_summary
    
    def test_stakeholder_map_visualization(self):
        """Test stakeholder mapping visualization"""
        analysis = self.engine.create_decision_analysis(
            title="Stakeholder Visualization Test",
            description="Test stakeholder visualization",
            decision_type=DecisionType.STRATEGIC
        )
        
        # Add stakeholders
        self.engine.add_stakeholder_impact(
            analysis_id=analysis.id,
            stakeholder_id="board-001",
            stakeholder_name="Board",
            impact_level=ImpactLevel.HIGH,
            impact_description="Strategic impact",
            support_likelihood=0.8,
            concerns=["Budget", "Timeline"]
        )
        
        self.engine.add_stakeholder_impact(
            analysis_id=analysis.id,
            stakeholder_id="team-001",
            stakeholder_name="Engineering Team",
            impact_level=ImpactLevel.MODERATE,
            impact_description="Implementation impact",
            support_likelihood=0.9,
            concerns=["Workload"]
        )
        
        viz = self.engine.create_decision_visualization(
            analysis_id=analysis.id,
            visualization_type="stakeholder_map"
        )
        
        assert viz.visualization_type == "stakeholder_map"
        assert viz.title == "Stakeholder Impact and Support Analysis"
        assert viz.chart_config["type"] == "scatter"
        assert len(viz.chart_config["data"]) == 2  # 2 stakeholders
        assert "Analysis of 2 key stakeholders" in viz.executive_summary
    
    def test_update_analysis_quality_score(self):
        """Test analysis quality score calculation"""
        analysis = self.engine.create_decision_analysis(
            title="Quality Test",
            description="Test quality scoring",
            decision_type=DecisionType.STRATEGIC
        )
        
        # Initially should have low quality (only criteria from template)
        initial_score = self.engine.update_analysis_quality_score(analysis.id)
        assert initial_score == 0.2  # Only criteria present
        
        # Add option
        self.engine.add_decision_option(
            analysis_id=analysis.id,
            title="Test Option",
            description="Test",
            pros=["Pro"],
            cons=["Con"]
        )
        score_after_option = self.engine.update_analysis_quality_score(analysis.id)
        assert score_after_option > initial_score
        
        # Add stakeholder
        self.engine.add_stakeholder_impact(
            analysis_id=analysis.id,
            stakeholder_id="test-001",
            stakeholder_name="Test Stakeholder",
            impact_level=ImpactLevel.MODERATE,
            impact_description="Test impact"
        )
        score_after_stakeholder = self.engine.update_analysis_quality_score(analysis.id)
        assert score_after_stakeholder > score_after_option
        
        # Add risk
        self.engine.add_risk_assessment(
            analysis_id=analysis.id,
            risk_category="Test Risk",
            probability=0.3,
            impact=ImpactLevel.MODERATE,
            description="Test risk"
        )
        score_after_risk = self.engine.update_analysis_quality_score(analysis.id)
        assert score_after_risk > score_after_stakeholder
        
        # Generate recommendation
        for criteria in analysis.criteria:
            self.engine.score_option_criteria(
                analysis_id=analysis.id,
                option_id=analysis.options[0].id,
                criteria_scores={criteria.id: 0.8}
            )
        
        self.engine.generate_recommendation(analysis.id)
        final_score = self.engine.update_analysis_quality_score(analysis.id)
        assert final_score > score_after_risk
        assert final_score >= 1.0  # Should be at maximum
    
    def test_decision_templates(self):
        """Test decision analysis templates"""
        # Test strategic investment template
        strategic_analysis = self.engine.create_decision_analysis(
            title="Strategic Test",
            description="Test strategic template",
            decision_type=DecisionType.STRATEGIC
        )
        
        strategic_criteria_names = [c.name for c in strategic_analysis.criteria]
        assert "Strategic Alignment" in strategic_criteria_names
        assert "Financial Return" in strategic_criteria_names
        
        # Test technology decision template
        tech_analysis = self.engine.create_decision_analysis(
            title="Technology Test",
            description="Test technology template",
            decision_type=DecisionType.TECHNOLOGY
        )
        
        tech_criteria_names = [c.name for c in tech_analysis.criteria]
        assert "Technical Capability" in tech_criteria_names
        assert "Scalability" in tech_criteria_names
        assert "Security" in tech_criteria_names
        
        # Test personnel decision template
        personnel_analysis = self.engine.create_decision_analysis(
            title="Personnel Test",
            description="Test personnel template",
            decision_type=DecisionType.PERSONNEL
        )
        
        personnel_criteria_names = [c.name for c in personnel_analysis.criteria]
        assert "Capability Match" in personnel_criteria_names
        assert "Cultural Fit" in personnel_criteria_names
    
    def test_error_handling(self):
        """Test error handling in decision analysis"""
        # Test invalid analysis ID
        with pytest.raises(ValueError):
            self.engine.add_decision_option(
                analysis_id="invalid-id",
                title="Test",
                description="Test",
                pros=[],
                cons=[]
            )
        
        # Test invalid option ID for scoring
        analysis = self.engine.create_decision_analysis(
            title="Error Test",
            description="Test errors",
            decision_type=DecisionType.STRATEGIC
        )
        
        success = self.engine.score_option_criteria(
            analysis_id=analysis.id,
            option_id="invalid-option-id",
            criteria_scores={}
        )
        assert not success
        
        # Test recommendation without options
        with pytest.raises(ValueError):
            self.engine.generate_recommendation(analysis.id)
        
        # Test impact assessment without recommendation
        with pytest.raises(ValueError):
            self.engine.assess_decision_impact(analysis.id)