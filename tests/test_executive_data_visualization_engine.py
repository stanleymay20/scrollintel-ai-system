"""
Tests for Executive Data Visualization Engine

This module contains comprehensive tests for the executive data visualization system.
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch

from scrollintel.engines.executive_data_visualization_engine import (
    ExecutiveDataVisualizationEngine, ExecutiveDataVisualizer,
    VisualizationOptimizer, VisualizationImpactMeasurer,
    ExecutiveVisualization, VisualizationType, ChartType
)


class TestExecutiveDataVisualizer:
    """Test cases for ExecutiveDataVisualizer"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.visualizer = ExecutiveDataVisualizer()
        self.test_data = {
            "revenue": 15000000,
            "growth_rate": 18.5,
            "profit_margin": 22.3,
            "customer_count": 1250,
            "market_share": 15.8,
            "performance_score": 87,
            "strategic_initiatives": [
                "Digital transformation",
                "Market expansion",
                "Product innovation"
            ],
            "financial_metrics": {
                "operating_margin": 18.2,
                "cash_flow": 5000000,
                "debt_ratio": 0.3
            },
            "quarterly_trends": [12.5, 15.2, 18.5, 22.1]
        }
        self.board_context = {
            "board_id": "test_board",
            "board_members": ["member_1", "member_2", "member_3"],
            "meeting_type": "quarterly_review"
        }
    
    def test_create_executive_visualizations_success(self):
        """Test successful creation of executive visualizations"""
        visualizations = self.visualizer.create_executive_visualizations(
            data=self.test_data,
            board_context=self.board_context
        )
        
        assert isinstance(visualizations, list)
        assert len(visualizations) > 0
        assert len(visualizations) <= 5  # Should limit to 5 visualizations
        
        for viz in visualizations:
            assert isinstance(viz, ExecutiveVisualization)
            assert viz.id is not None
            assert viz.title is not None
            assert isinstance(viz.visualization_type, VisualizationType)
            assert isinstance(viz.chart_type, ChartType)
            assert isinstance(viz.insights, list)
            assert len(viz.insights) > 0
            assert viz.executive_summary is not None
            assert 0.0 <= viz.impact_score <= 1.0
            assert isinstance(viz.board_relevance, dict)
    
    def test_analyze_data_structure(self):
        """Test data structure analysis"""
        analysis = self.visualizer._analyze_data_structure(self.test_data)
        
        assert isinstance(analysis, dict)
        assert "data_types" in analysis
        assert "metrics_count" in analysis
        assert "financial_data" in analysis
        assert "performance_data" in analysis
        assert "strategic_data" in analysis
        
        # Should detect financial data
        assert analysis["financial_data"] is True
        # Should detect performance data
        assert analysis["performance_data"] is True
        # Should detect strategic data
        assert analysis["strategic_data"] is True
        # Should detect time series data
        assert analysis["time_series"] is True
        # Should count numeric metrics
        assert analysis["metrics_count"] > 0
    
    def test_recommend_visualizations(self):
        """Test visualization recommendations"""
        data_analysis = self.visualizer._analyze_data_structure(self.test_data)
        recommendations = self.visualizer._recommend_visualizations(data_analysis, self.board_context)
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        assert len(recommendations) <= 5
        
        for rec in recommendations:
            assert "type" in rec
            assert "chart_type" in rec
            assert "focus_area" in rec
            assert "priority" in rec
            assert isinstance(rec["type"], VisualizationType)
            assert isinstance(rec["chart_type"], ChartType)
    
    def test_create_visualization(self):
        """Test individual visualization creation"""
        viz = self.visualizer._create_visualization(
            data=self.test_data,
            viz_type=VisualizationType.FINANCIAL_SUMMARY,
            chart_type=ChartType.EXECUTIVE_SUMMARY_CHART,
            focus_area="financial_performance",
            board_context=self.board_context
        )
        
        assert isinstance(viz, ExecutiveVisualization)
        assert viz.visualization_type == VisualizationType.FINANCIAL_SUMMARY
        assert viz.chart_type == ChartType.EXECUTIVE_SUMMARY_CHART
        assert len(viz.insights) > 0
        assert viz.impact_score > 0.0
        assert len(viz.board_relevance) > 0
    
    def test_extract_focused_data(self):
        """Test focused data extraction"""
        financial_data = self.visualizer._extract_focused_data(
            self.test_data, "financial_performance"
        )
        
        assert isinstance(financial_data, dict)
        assert len(financial_data) > 0
        # Should include financial-related keys
        financial_keys = [k for k in financial_data.keys() 
                         if any(term in k.lower() for term in ['revenue', 'profit', 'financial'])]
        assert len(financial_keys) > 0
    
    def test_generate_financial_insights(self):
        """Test financial insights generation"""
        financial_data = {
            "revenue": 15000000,
            "growth_rate": 18.5,
            "profit_margin": 22.3
        }
        
        insights = self.visualizer._generate_financial_insights(financial_data)
        
        assert isinstance(insights, list)
        assert len(insights) > 0
        # Should mention revenue
        assert any("revenue" in insight.lower() for insight in insights)
        # Should mention growth
        assert any("growth" in insight.lower() for insight in insights)
    
    def test_generate_performance_insights(self):
        """Test performance insights generation"""
        performance_data = {
            "performance_score": 87,
            "efficiency_rating": 92,
            "quality_score": 78
        }
        
        insights = self.visualizer._generate_performance_insights(performance_data)
        
        assert isinstance(insights, list)
        assert len(insights) > 0
        # Should evaluate performance levels
        assert any("excellent" in insight.lower() or "good" in insight.lower() 
                  for insight in insights)
    
    def test_generate_trend_insights(self):
        """Test trend insights generation"""
        trend_data = {
            "quarterly_revenue": [10000000, 12000000, 14000000, 15000000],
            "growth_trajectory": [10.5, 15.2, 18.5, 22.1]
        }
        
        insights = self.visualizer._generate_trend_insights(trend_data)
        
        assert isinstance(insights, list)
        assert len(insights) > 0
        # Should identify positive trends
        assert any("positive" in insight.lower() or "trend" in insight.lower() 
                  for insight in insights)
    
    def test_calculate_impact_score(self):
        """Test impact score calculation"""
        score = self.visualizer._calculate_impact_score(
            data=self.test_data,
            viz_type=VisualizationType.FINANCIAL_SUMMARY,
            board_context=self.board_context
        )
        
        assert 0.0 <= score <= 1.0
        # Financial summary should have high impact
        assert score >= 0.7
    
    def test_error_handling_empty_data(self):
        """Test error handling with empty data"""
        visualizations = self.visualizer.create_executive_visualizations(
            data={},
            board_context=self.board_context
        )
        
        # Should handle gracefully and return empty list or minimal visualizations
        assert isinstance(visualizations, list)
    
    def test_error_handling_invalid_data(self):
        """Test error handling with invalid data"""
        with pytest.raises(Exception):
            self.visualizer.create_executive_visualizations(
                data=None,
                board_context=self.board_context
            )


class TestVisualizationOptimizer:
    """Test cases for VisualizationOptimizer"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.optimizer = VisualizationOptimizer()
        
        # Create mock visualizations
        self.mock_visualizations = [
            ExecutiveVisualization(
                id="viz_1",
                title="Financial Summary",
                visualization_type=VisualizationType.FINANCIAL_SUMMARY,
                chart_type=ChartType.EXECUTIVE_SUMMARY_CHART,
                data={"revenue": 15000000, "profit": 3000000},
                insights=["Strong revenue growth", "Healthy profit margins", "Market leadership"],
                executive_summary="Financial performance exceeds expectations",
                impact_score=0.9,
                board_relevance={"member_1": 0.8, "member_2": 0.9}
            ),
            ExecutiveVisualization(
                id="viz_2",
                title="Performance Metrics",
                visualization_type=VisualizationType.KPI_SCORECARD,
                chart_type=ChartType.KPI_GAUGE,
                data={"performance": 87, "efficiency": 92},
                insights=["Excellent performance", "High efficiency"],
                executive_summary="Performance metrics show strong results",
                impact_score=0.7,
                board_relevance={"member_1": 0.7, "member_2": 0.8}
            )
        ]
    
    def test_optimize_for_board_consumption_success(self):
        """Test successful optimization for board consumption"""
        board_preferences = {
            "detail_level": "medium",
            "visual_emphasis": True,
            "time_constraints": 30
        }
        
        optimized = self.optimizer.optimize_for_board_consumption(
            self.mock_visualizations,
            board_preferences
        )
        
        assert isinstance(optimized, list)
        assert len(optimized) <= len(self.mock_visualizations)
        # Should be sorted by impact score
        if len(optimized) > 1:
            assert optimized[0].impact_score >= optimized[1].impact_score
    
    def test_apply_board_preferences(self):
        """Test application of board preferences"""
        preferences = {"detail_level": "low"}
        
        adjusted = self.optimizer._apply_board_preferences(
            self.mock_visualizations,
            preferences
        )
        
        # Low detail should limit insights
        for viz in adjusted:
            assert len(viz.insights) <= 3
    
    def test_optimize_data_density(self):
        """Test data density optimization"""
        # Create visualization with high data density
        high_density_viz = ExecutiveVisualization(
            id="dense_viz",
            title="Dense Data",
            visualization_type=VisualizationType.EXECUTIVE_DASHBOARD,
            chart_type=ChartType.EXECUTIVE_SUMMARY_CHART,
            data={f"metric_{i}": i for i in range(15)},  # 15 data points
            insights=["Dense data insight"],
            executive_summary="Dense data summary",
            impact_score=0.8,
            board_relevance={"member_1": 0.8}
        )
        
        optimized = self.optimizer._optimize_data_density([high_density_viz])
        
        # Should limit data points
        assert len(optimized[0].data) <= 8
    
    def test_optimize_for_time_constraints(self):
        """Test optimization for time constraints"""
        # Create many visualizations
        many_visualizations = self.mock_visualizations * 3  # 6 visualizations
        
        optimized = self.optimizer._optimize_for_time_constraints(many_visualizations)
        
        # Should limit to optimal number
        assert len(optimized) <= 4


class TestVisualizationImpactMeasurer:
    """Test cases for VisualizationImpactMeasurer"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.impact_measurer = VisualizationImpactMeasurer()
        
        # Create mock visualizations with varying impact
        self.mock_visualizations = [
            ExecutiveVisualization(
                id="high_impact_viz",
                title="High Impact Visualization",
                visualization_type=VisualizationType.FINANCIAL_SUMMARY,
                chart_type=ChartType.EXECUTIVE_SUMMARY_CHART,
                data={"revenue": 15000000},
                insights=["Critical insight", "Strategic finding", "Key recommendation"],
                executive_summary="High impact summary",
                impact_score=0.9,
                board_relevance={"member_1": 0.9, "member_2": 0.8}
            ),
            ExecutiveVisualization(
                id="medium_impact_viz",
                title="Medium Impact Visualization",
                visualization_type=VisualizationType.KPI_SCORECARD,
                chart_type=ChartType.KPI_GAUGE,
                data={"performance": 75},
                insights=["Good insight", "Useful finding"],
                executive_summary="Medium impact summary",
                impact_score=0.6,
                board_relevance={"member_1": 0.7, "member_2": 0.6}
            )
        ]
    
    def test_measure_visualization_impact_success(self):
        """Test successful impact measurement"""
        impact_metrics = self.impact_measurer.measure_visualization_impact(
            self.mock_visualizations
        )
        
        assert isinstance(impact_metrics, dict)
        assert "total_visualizations" in impact_metrics
        assert "average_impact_score" in impact_metrics
        assert "high_impact_count" in impact_metrics
        assert "board_relevance_score" in impact_metrics
        assert "insight_quality_score" in impact_metrics
        assert "recommendations" in impact_metrics
        
        assert impact_metrics["total_visualizations"] == len(self.mock_visualizations)
        assert 0.0 <= impact_metrics["average_impact_score"] <= 1.0
        assert impact_metrics["high_impact_count"] >= 0
        assert isinstance(impact_metrics["recommendations"], list)
    
    def test_improve_visualization_impact(self):
        """Test visualization impact improvement"""
        # Create low-impact metrics to trigger improvement
        low_impact_metrics = {
            "average_impact_score": 0.5,
            "high_impact_count": 0,
            "board_relevance_score": 0.6,
            "insight_quality_score": 0.5
        }
        
        improved = self.impact_measurer.improve_visualization_impact(
            self.mock_visualizations,
            low_impact_metrics
        )
        
        assert isinstance(improved, list)
        assert len(improved) == len(self.mock_visualizations)
        
        # Should improve low-impact visualizations
        for viz in improved:
            if viz.id == "medium_impact_viz":
                # Should have enhanced impact score
                assert viz.impact_score > 0.6
    
    def test_generate_improvement_recommendations(self):
        """Test improvement recommendations generation"""
        low_metrics = {
            "average_impact_score": 0.5,
            "high_impact_count": 0,
            "board_relevance_score": 0.6,
            "insight_quality_score": 0.5
        }
        
        recommendations = self.impact_measurer._generate_improvement_recommendations(
            self.mock_visualizations,
            low_metrics
        )
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        # Should suggest improvements for low scores
        assert any("relevance" in rec.lower() for rec in recommendations)
        assert any("impact" in rec.lower() for rec in recommendations)
    
    def test_enhance_low_impact_visualization(self):
        """Test enhancement of low-impact visualization"""
        low_impact_viz = self.mock_visualizations[1]  # Medium impact viz
        original_score = low_impact_viz.impact_score
        
        enhanced = self.impact_measurer._enhance_low_impact_visualization(low_impact_viz)
        
        # Should improve impact score
        assert enhanced.impact_score > original_score
        # Should enhance summary
        assert "strategic" in enhanced.executive_summary.lower()
    
    def test_measure_empty_visualizations(self):
        """Test impact measurement with empty visualizations"""
        impact_metrics = self.impact_measurer.measure_visualization_impact([])
        
        assert impact_metrics["total_visualizations"] == 0
        assert impact_metrics["average_impact_score"] == 0.0


class TestExecutiveDataVisualizationEngine:
    """Test cases for ExecutiveDataVisualizationEngine"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.engine = ExecutiveDataVisualizationEngine()
        self.test_data = {
            "financial_performance": {
                "revenue": 25000000,
                "profit_margin": 18.5,
                "growth_rate": 22.3,
                "operating_margin": 15.8
            },
            "operational_metrics": {
                "customer_satisfaction": 92,
                "employee_engagement": 87,
                "operational_efficiency": 89,
                "quality_score": 94
            },
            "strategic_initiatives": {
                "digital_transformation": "85% complete",
                "market_expansion": "3 new regions",
                "product_innovation": "5 new features launched"
            },
            "market_position": {
                "market_share": 15.8,
                "competitive_ranking": 2,
                "brand_recognition": 78
            }
        }
        self.board_context = {
            "board_id": "executive_board",
            "board_members": ["ceo", "cfo", "independent_1", "independent_2"],
            "meeting_type": "quarterly_review",
            "focus_areas": ["financial_performance", "strategic_progress"]
        }
        self.board_preferences = {
            "detail_level": "high",
            "visual_emphasis": True,
            "interaction_style": "formal",
            "time_constraints": 45
        }
    
    def test_create_optimized_visualizations_success(self):
        """Test successful creation of optimized visualizations"""
        visualizations, impact_metrics = self.engine.create_optimized_visualizations(
            data=self.test_data,
            board_context=self.board_context,
            board_preferences=self.board_preferences
        )
        
        assert isinstance(visualizations, list)
        assert len(visualizations) > 0
        assert len(visualizations) <= 4  # Should limit for board attention
        
        assert isinstance(impact_metrics, dict)
        assert "total_visualizations" in impact_metrics
        assert "average_impact_score" in impact_metrics
        
        # Validate visualization quality
        for viz in visualizations:
            assert isinstance(viz, ExecutiveVisualization)
            assert viz.impact_score > 0.0
            assert len(viz.insights) > 0
            assert viz.executive_summary is not None
            assert len(viz.board_relevance) > 0
    
    def test_financial_data_visualization(self):
        """Test visualization of financial data"""
        financial_data = {
            "quarterly_revenue": [20000000, 22000000, 24000000, 25000000],
            "profit_margins": [16.2, 17.1, 17.8, 18.5],
            "operating_expenses": [15000000, 16000000, 17000000, 18000000],
            "cash_flow": 8000000,
            "debt_to_equity": 0.25
        }
        
        visualizations, impact_metrics = self.engine.create_optimized_visualizations(
            data=financial_data,
            board_context=self.board_context
        )
        
        # Should create financial-focused visualizations
        financial_viz = [v for v in visualizations 
                        if v.visualization_type == VisualizationType.FINANCIAL_SUMMARY]
        assert len(financial_viz) > 0
        
        # Should have high impact for financial data
        assert impact_metrics["average_impact_score"] >= 0.7
    
    def test_performance_data_visualization(self):
        """Test visualization of performance data"""
        performance_data = {
            "kpi_dashboard": {
                "customer_acquisition_cost": 150,
                "customer_lifetime_value": 2500,
                "monthly_recurring_revenue": 2000000,
                "churn_rate": 2.5
            },
            "performance_scores": {
                "product_quality": 94,
                "customer_service": 89,
                "delivery_performance": 96,
                "innovation_index": 87
            }
        }
        
        visualizations, impact_metrics = self.engine.create_optimized_visualizations(
            data=performance_data,
            board_context=self.board_context
        )
        
        # Should create KPI-focused visualizations
        kpi_viz = [v for v in visualizations 
                  if v.visualization_type == VisualizationType.KPI_SCORECARD]
        assert len(kpi_viz) > 0
        
        # Should generate performance insights
        performance_insights = []
        for viz in visualizations:
            performance_insights.extend([i for i in viz.insights 
                                       if "performance" in i.lower()])
        assert len(performance_insights) > 0
    
    def test_strategic_data_visualization(self):
        """Test visualization of strategic data"""
        strategic_data = {
            "strategic_objectives": {
                "market_leadership": "Target: #1 position by 2025",
                "innovation_pipeline": "12 products in development",
                "global_expansion": "Enter 5 new markets",
                "sustainability_goals": "Carbon neutral by 2030"
            },
            "initiative_progress": {
                "digital_transformation": 85,
                "operational_excellence": 78,
                "talent_development": 92,
                "customer_experience": 88
            }
        }
        
        visualizations, impact_metrics = self.engine.create_optimized_visualizations(
            data=strategic_data,
            board_context=self.board_context
        )
        
        # Should create strategic-focused visualizations
        strategic_viz = [v for v in visualizations 
                        if v.visualization_type == VisualizationType.STRATEGIC_METRICS]
        assert len(strategic_viz) > 0
        
        # Should include strategic insights
        strategic_insights = []
        for viz in visualizations:
            strategic_insights.extend([i for i in viz.insights 
                                     if "strategic" in i.lower()])
        assert len(strategic_insights) > 0
    
    def test_board_context_optimization(self):
        """Test optimization based on board context"""
        # Test with different board contexts
        investor_context = {
            "board_id": "investor_board",
            "board_members": ["lead_investor", "partner_1", "partner_2"],
            "meeting_type": "investor_update",
            "focus_areas": ["financial_performance", "growth_metrics"]
        }
        
        visualizations, impact_metrics = self.engine.create_optimized_visualizations(
            data=self.test_data,
            board_context=investor_context
        )
        
        # Should optimize for investor interests
        assert len(visualizations) > 0
        
        # Should have high board relevance
        for viz in visualizations:
            avg_relevance = sum(viz.board_relevance.values()) / len(viz.board_relevance)
            assert avg_relevance >= 0.6
    
    def test_board_preferences_application(self):
        """Test application of board preferences"""
        high_detail_preferences = {
            "detail_level": "high",
            "visual_emphasis": True,
            "interaction_style": "detailed"
        }
        
        low_detail_preferences = {
            "detail_level": "low",
            "visual_emphasis": False,
            "interaction_style": "summary"
        }
        
        # Test high detail
        high_detail_viz, _ = self.engine.create_optimized_visualizations(
            data=self.test_data,
            board_context=self.board_context,
            board_preferences=high_detail_preferences
        )
        
        # Test low detail
        low_detail_viz, _ = self.engine.create_optimized_visualizations(
            data=self.test_data,
            board_context=self.board_context,
            board_preferences=low_detail_preferences
        )
        
        # Low detail should have fewer insights per visualization
        if low_detail_viz:
            avg_insights_low = sum(len(v.insights) for v in low_detail_viz) / len(low_detail_viz)
            if high_detail_viz:
                avg_insights_high = sum(len(v.insights) for v in high_detail_viz) / len(high_detail_viz)
                assert avg_insights_low <= avg_insights_high
    
    def test_impact_improvement_cycle(self):
        """Test automatic impact improvement cycle"""
        # Create data that should trigger improvement
        minimal_data = {
            "basic_metric": 100,
            "simple_value": 50
        }
        
        visualizations, impact_metrics = self.engine.create_optimized_visualizations(
            data=minimal_data,
            board_context=self.board_context
        )
        
        # Should still create visualizations even with minimal data
        assert len(visualizations) > 0
        
        # Should have improvement recommendations if impact is low
        if impact_metrics["average_impact_score"] < 0.8:
            assert len(impact_metrics["recommendations"]) > 0
    
    def test_error_handling_invalid_data(self):
        """Test error handling with invalid data"""
        with pytest.raises(Exception):
            self.engine.create_optimized_visualizations(
                data=None,
                board_context=self.board_context
            )
    
    def test_error_handling_empty_context(self):
        """Test handling of empty board context"""
        visualizations, impact_metrics = self.engine.create_optimized_visualizations(
            data=self.test_data,
            board_context=None
        )
        
        # Should handle gracefully
        assert isinstance(visualizations, list)
        assert isinstance(impact_metrics, dict)
    
    @patch('scrollintel.engines.executive_data_visualization_engine.logging')
    def test_logging_integration(self, mock_logging):
        """Test logging integration"""
        visualizations, impact_metrics = self.engine.create_optimized_visualizations(
            data=self.test_data,
            board_context=self.board_context
        )
        
        # Should log successful creation
        assert len(visualizations) > 0


class TestIntegrationScenarios:
    """Integration test scenarios for executive data visualization"""
    
    def setup_method(self):
        """Set up integration test fixtures"""
        self.engine = ExecutiveDataVisualizationEngine()
    
    def test_quarterly_board_review_scenario(self):
        """Test complete quarterly board review visualization scenario"""
        quarterly_data = {
            "financial_summary": {
                "q4_revenue": 28000000,
                "annual_revenue": 95000000,
                "revenue_growth": 23.5,
                "gross_margin": 68.2,
                "operating_margin": 18.7,
                "net_income": 15000000,
                "cash_position": 45000000,
                "debt_level": 12000000
            },
            "operational_performance": {
                "customer_acquisition": 2150,
                "customer_retention": 94.8,
                "product_uptime": 99.97,
                "support_satisfaction": 91,
                "employee_satisfaction": 87,
                "operational_efficiency": 89
            },
            "strategic_progress": {
                "product_roadmap_completion": 92,
                "market_expansion_progress": 78,
                "partnership_development": 85,
                "innovation_pipeline": 15,  # number of projects
                "competitive_positioning": "Strong"
            },
            "market_metrics": {
                "market_share": 18.5,
                "brand_recognition": 82,
                "customer_satisfaction": 93,
                "net_promoter_score": 67
            },
            "risk_indicators": {
                "cybersecurity_score": 95,
                "regulatory_compliance": 100,
                "financial_risk_rating": "Low",
                "operational_risk_level": "Medium"
            }
        }
        
        board_context = {
            "board_id": "public_company_board",
            "board_members": [
                "chairman", "ceo", "cfo", "independent_1", 
                "independent_2", "investor_rep", "industry_expert"
            ],
            "meeting_type": "quarterly_board_meeting",
            "focus_areas": [
                "financial_performance", "strategic_execution", 
                "risk_management", "market_position"
            ],
            "time_allocation": 60  # minutes for presentations
        }
        
        board_preferences = {
            "detail_level": "high",
            "visual_emphasis": True,
            "interaction_style": "formal",
            "decision_focus": True
        }
        
        # Create comprehensive visualizations
        visualizations, impact_metrics = self.engine.create_optimized_visualizations(
            data=quarterly_data,
            board_context=board_context,
            board_preferences=board_preferences
        )
        
        # Validate comprehensive coverage
        assert len(visualizations) >= 3
        assert len(visualizations) <= 4  # Optimal for board attention
        
        # Should cover key areas
        viz_types = [v.visualization_type for v in visualizations]
        assert VisualizationType.FINANCIAL_SUMMARY in viz_types
        assert VisualizationType.EXECUTIVE_DASHBOARD in viz_types or VisualizationType.KPI_SCORECARD in viz_types
        
        # Should have high impact scores
        assert impact_metrics["average_impact_score"] >= 0.7
        assert impact_metrics["high_impact_count"] >= 2
        
        # Should have board-relevant insights
        total_insights = sum(len(v.insights) for v in visualizations)
        assert total_insights >= 8  # Sufficient insights for board discussion
        
        # Should have strategic focus
        strategic_content = []
        for viz in visualizations:
            strategic_content.extend([
                insight for insight in viz.insights 
                if any(term in insight.lower() for term in ['strategic', 'growth', 'market', 'competitive'])
            ])
        assert len(strategic_content) >= 3
    
    def test_crisis_communication_visualization(self):
        """Test visualization for crisis communication scenario"""
        crisis_data = {
            "incident_overview": {
                "incident_type": "Data Security Breach",
                "severity_level": "High",
                "customers_affected": 15000,
                "data_types_compromised": ["email", "phone"],
                "discovery_time": "2024-01-15 14:30",
                "containment_time": "2024-01-15 18:45"
            },
            "immediate_response": {
                "response_team_activated": True,
                "systems_isolated": True,
                "customers_notified": 98.5,  # percentage
                "regulators_informed": True,
                "media_response_prepared": True
            },
            "impact_assessment": {
                "financial_impact_estimate": 2500000,
                "reputation_risk_level": "Medium",
                "legal_exposure": "Limited",
                "operational_disruption": "Minimal",
                "customer_churn_risk": 5.2  # percentage
            },
            "remediation_status": {
                "security_patches_deployed": 100,
                "monitoring_enhanced": True,
                "third_party_audit_scheduled": True,
                "customer_support_scaled": 300,  # percentage increase
                "legal_review_complete": 85  # percentage
            }
        }
        
        crisis_board_context = {
            "board_id": "crisis_response_board",
            "board_members": ["chairman", "ceo", "cfo", "legal_counsel", "security_expert"],
            "meeting_type": "emergency_board_meeting",
            "urgency_level": "high",
            "decision_required": True
        }
        
        # Create crisis visualizations
        visualizations, impact_metrics = self.engine.create_optimized_visualizations(
            data=crisis_data,
            board_context=crisis_board_context
        )
        
        # Crisis visualizations should be focused and actionable
        assert len(visualizations) <= 3  # Focused for urgent decision-making
        
        # Should have high impact for urgent attention
        assert impact_metrics["average_impact_score"] >= 0.8
        
        # Should include risk-focused content
        risk_content = []
        for viz in visualizations:
            risk_content.extend([
                insight for insight in viz.insights 
                if any(term in insight.lower() for term in ['risk', 'impact', 'response', 'critical'])
            ])
        assert len(risk_content) >= 2
        
        # Should have clear executive summaries for quick understanding
        for viz in visualizations:
            assert len(viz.executive_summary) > 0
            assert len(viz.executive_summary.split()) <= 20  # Concise for crisis
    
    def test_investor_presentation_visualization(self):
        """Test visualization optimized for investor presentation"""
        investor_data = {
            "growth_metrics": {
                "arr": 45000000,  # Annual Recurring Revenue
                "arr_growth_rate": 125,
                "monthly_growth_rate": 8.5,
                "customer_growth": 89,
                "revenue_per_customer": 3600
            },
            "unit_economics": {
                "customer_acquisition_cost": 1200,
                "customer_lifetime_value": 15000,
                "ltv_cac_ratio": 12.5,
                "payback_period": 14,  # months
                "gross_margin": 78.5
            },
            "market_traction": {
                "total_customers": 12500,
                "enterprise_customers": 450,
                "fortune_500_customers": 25,
                "market_penetration": 8.5,
                "competitive_wins": 67
            },
            "financial_health": {
                "burn_rate": 2800000,
                "runway_months": 24,
                "cash_position": 67000000,
                "revenue_multiple": 8.2,
                "growth_efficiency": 1.8
            },
            "product_metrics": {
                "product_market_fit_score": 85,
                "feature_adoption_rate": 73,
                "user_engagement_score": 89,
                "platform_reliability": 99.95
            }
        }
        
        investor_board_context = {
            "board_id": "investor_board",
            "board_members": ["lead_investor", "partner_1", "partner_2", "ceo", "cfo"],
            "meeting_type": "investor_board_meeting",
            "focus_areas": ["growth_metrics", "unit_economics", "market_position"],
            "investment_stage": "series_b"
        }
        
        investor_preferences = {
            "detail_level": "high",
            "metrics_focus": True,
            "growth_emphasis": True,
            "financial_rigor": True
        }
        
        # Create investor-focused visualizations
        visualizations, impact_metrics = self.engine.create_optimized_visualizations(
            data=investor_data,
            board_context=investor_board_context,
            board_preferences=investor_preferences
        )
        
        # Should create growth and financial focused visualizations
        assert len(visualizations) >= 3
        
        # Should have very high impact for investor interests
        assert impact_metrics["average_impact_score"] >= 0.8
        
        # Should include growth-focused insights
        growth_insights = []
        for viz in visualizations:
            growth_insights.extend([
                insight for insight in viz.insights 
                if any(term in insight.lower() for term in ['growth', 'revenue', 'customer', 'market'])
            ])
        assert len(growth_insights) >= 5
        
        # Should include financial metrics visualization
        financial_viz = [v for v in visualizations 
                        if v.visualization_type == VisualizationType.FINANCIAL_SUMMARY]
        assert len(financial_viz) >= 1
        
        # Should have high board relevance for all investor members
        for viz in visualizations:
            investor_relevance = [score for member_id, score in viz.board_relevance.items() 
                                if "investor" in member_id or "partner" in member_id]
            if investor_relevance:
                assert all(score >= 0.8 for score in investor_relevance)