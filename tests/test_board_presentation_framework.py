"""
Tests for Board Presentation Framework

This module contains comprehensive tests for the board presentation design system.
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch

from scrollintel.engines.presentation_design_engine import (
    BoardPresentationFramework, PresentationDesigner, 
    PresentationFormatOptimizer, PresentationQualityAssessor
)
from scrollintel.models.board_presentation_models import (
    BoardPresentation, PresentationFormat, QualityMetrics,
    BoardMemberProfile, BoardMemberType
)
from scrollintel.models.board_dynamics_models import Board


class TestPresentationDesigner:
    """Test cases for PresentationDesigner"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.designer = PresentationDesigner()
        self.mock_board = Board(
            id="test_board_1",
            name="Test Board",
            composition={},
            governance_structure={},
            meeting_patterns={},
            decision_processes={}
        )
        self.test_content = {
            "overview": "Strategic overview of company performance",
            "key_metrics": {
                "revenue": 10000000,
                "growth_rate": 15.5,
                "market_share": 25
            },
            "recommendations": [
                "Increase R&D investment",
                "Expand into new markets",
                "Optimize operational efficiency"
            ],
            "next_steps": "Implementation timeline and resource allocation"
        }
    
    def test_create_board_presentation_success(self):
        """Test successful board presentation creation"""
        presentation = self.designer.create_board_presentation(
            content=self.test_content,
            board=self.mock_board,
            format_type=PresentationFormat.STRATEGIC_OVERVIEW
        )
        
        assert isinstance(presentation, BoardPresentation)
        assert presentation.title == "Board Presentation"
        assert presentation.board_id == "test_board_1"
        assert presentation.format_type == PresentationFormat.STRATEGIC_OVERVIEW
        assert len(presentation.slides) > 0
        assert presentation.executive_summary is not None
        assert len(presentation.key_messages) > 0
        assert len(presentation.success_metrics) > 0
    
    def test_create_presentation_with_custom_title(self):
        """Test presentation creation with custom title"""
        content_with_title = {**self.test_content, "title": "Q4 Strategic Review"}
        
        presentation = self.designer.create_board_presentation(
            content=content_with_title,
            board=self.mock_board
        )
        
        assert presentation.title == "Q4 Strategic Review"
    
    def test_select_optimal_template(self):
        """Test template selection logic"""
        template = self.designer._select_optimal_template(
            self.mock_board, 
            PresentationFormat.EXECUTIVE_SUMMARY
        )
        
        assert template is not None
        assert template.format_type == PresentationFormat.EXECUTIVE_SUMMARY
    
    def test_generate_slides(self):
        """Test slide generation from content"""
        template = self.designer._get_default_template(PresentationFormat.STRATEGIC_OVERVIEW)
        slides = self.designer._generate_slides(self.test_content, template, self.mock_board)
        
        assert len(slides) > 0
        for slide in slides:
            assert slide.slide_number > 0
            assert slide.title is not None
            assert len(slide.sections) > 0
            assert slide.estimated_duration > 0
    
    def test_create_executive_summary(self):
        """Test executive summary creation"""
        summary = self.designer._create_executive_summary(self.test_content, self.mock_board)
        
        assert isinstance(summary, str)
        assert len(summary) > 0
        assert "overview" in summary.lower() or "metrics" in summary.lower()
    
    def test_extract_key_messages(self):
        """Test key message extraction"""
        messages = self.designer._extract_key_messages(self.test_content, self.mock_board)
        
        assert isinstance(messages, list)
        assert len(messages) > 0
        assert len(messages) <= 5  # Should limit to 5 messages
    
    def test_content_type_determination(self):
        """Test content type determination logic"""
        assert self.designer._determine_content_type(42) == "metric"
        assert self.designer._determine_content_type([1, 2, 3]) == "chart"
        assert self.designer._determine_content_type(["item1", "item2"]) == "list"
        assert self.designer._determine_content_type({"key": "value"}) == "table"
        assert self.designer._determine_content_type("text content") == "text"
    
    def test_importance_assessment(self):
        """Test content importance assessment"""
        assert self.designer._assess_importance("revenue", 1000000) == "critical"
        assert self.designer._assess_importance("performance", "good") == "important"
        assert self.designer._assess_importance("notes", "misc") == "supporting"
    
    def test_error_handling(self):
        """Test error handling in presentation creation"""
        with pytest.raises(Exception):
            self.designer.create_board_presentation(
                content=None,  # Invalid content
                board=self.mock_board
            )


class TestPresentationFormatOptimizer:
    """Test cases for PresentationFormatOptimizer"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.optimizer = PresentationFormatOptimizer()
        self.mock_board = Board(
            id="test_board_1",
            name="Test Board",
            composition={},
            governance_structure={},
            meeting_patterns={},
            decision_processes={}
        )
        
        # Create mock presentation
        designer = PresentationDesigner()
        test_content = {
            "overview": "Test overview",
            "metrics": {"revenue": 1000000},
            "recommendations": ["Test recommendation"]
        }
        self.mock_presentation = designer.create_board_presentation(
            content=test_content,
            board=self.mock_board
        )
    
    def test_optimize_for_board_success(self):
        """Test successful presentation optimization"""
        original_slide_count = len(self.mock_presentation.slides)
        
        optimized = self.optimizer.optimize_for_board(
            self.mock_presentation, 
            self.mock_board
        )
        
        assert isinstance(optimized, BoardPresentation)
        assert optimized.id == self.mock_presentation.id
        assert len(optimized.slides) == original_slide_count
    
    def test_analyze_board_preferences(self):
        """Test board preference analysis"""
        preferences = self.optimizer._analyze_board_preferences(self.mock_board)
        
        assert isinstance(preferences, dict)
        assert "attention_span" in preferences
        assert "detail_preference" in preferences
        assert "interaction_frequency" in preferences
        assert "visual_emphasis" in preferences
    
    def test_optimize_slide_order(self):
        """Test slide order optimization"""
        original_slides = self.mock_presentation.slides.copy()
        preferences = {"detail_preference": "high"}
        
        optimized_slides = self.optimizer._optimize_slide_order(original_slides, preferences)
        
        assert len(optimized_slides) == len(original_slides)
        # Slides should be reordered by priority
        assert optimized_slides[0].slide_number >= 1
    
    def test_adjust_content_density(self):
        """Test content density adjustment"""
        preferences = {"detail_preference": "low"}
        
        adjusted_slides = self.optimizer._adjust_content_density(
            self.mock_presentation.slides, 
            preferences
        )
        
        # Low detail preference should reduce sections
        for slide in adjusted_slides:
            assert len(slide.sections) <= 3
    
    def test_optimize_timing(self):
        """Test timing optimization"""
        preferences = {"attention_span": 10}  # 10 minutes
        
        optimized_slides = self.optimizer._optimize_timing(
            self.mock_presentation.slides, 
            preferences
        )
        
        total_time = sum(slide.estimated_duration for slide in optimized_slides)
        assert total_time <= 600  # Should be within 10 minutes


class TestPresentationQualityAssessor:
    """Test cases for PresentationQualityAssessor"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.assessor = PresentationQualityAssessor()
        self.mock_board = Board(
            id="test_board_1",
            name="Test Board",
            composition={},
            governance_structure={},
            meeting_patterns={},
            decision_processes={}
        )
        
        # Create mock presentation
        designer = PresentationDesigner()
        test_content = {
            "overview": "Strategic overview with key strategic initiatives",
            "performance": {"revenue": 1000000, "growth": 15},
            "recommendations": ["Increase investment", "Expand market"],
            "risk": "Market volatility assessment"
        }
        self.mock_presentation = designer.create_board_presentation(
            content=test_content,
            board=self.mock_board
        )
    
    def test_assess_quality_success(self):
        """Test successful quality assessment"""
        metrics = self.assessor.assess_quality(self.mock_presentation, self.mock_board)
        
        assert isinstance(metrics, QualityMetrics)
        assert 0.0 <= metrics.clarity_score <= 1.0
        assert 0.0 <= metrics.relevance_score <= 1.0
        assert 0.0 <= metrics.engagement_score <= 1.0
        assert 0.0 <= metrics.professional_score <= 1.0
        assert 0.0 <= metrics.time_efficiency_score <= 1.0
        assert 0.0 <= metrics.overall_score <= 1.0
        assert isinstance(metrics.improvement_suggestions, list)
    
    def test_assess_clarity(self):
        """Test clarity assessment"""
        score = self.assessor._assess_clarity(self.mock_presentation)
        
        assert 0.0 <= score <= 1.0
        # Should have good clarity with slides and summary
        assert score >= 0.8
    
    def test_assess_relevance(self):
        """Test relevance assessment"""
        score = self.assessor._assess_relevance(self.mock_presentation, self.mock_board)
        
        assert 0.0 <= score <= 1.0
        # Should have good relevance with strategic content
        assert score >= 0.7
    
    def test_assess_engagement_potential(self):
        """Test engagement potential assessment"""
        score = self.assessor._assess_engagement_potential(self.mock_presentation)
        
        assert 0.0 <= score <= 1.0
    
    def test_assess_professionalism(self):
        """Test professionalism assessment"""
        score = self.assessor._assess_professionalism(self.mock_presentation)
        
        assert 0.0 <= score <= 1.0
        # Should have good professionalism with proper structure
        assert score >= 0.8
    
    def test_assess_time_efficiency(self):
        """Test time efficiency assessment"""
        score = self.assessor._assess_time_efficiency(self.mock_presentation)
        
        assert 0.0 <= score <= 1.0
    
    def test_enhance_presentation(self):
        """Test presentation enhancement"""
        # Create low-quality metrics to trigger enhancement
        low_metrics = QualityMetrics(
            clarity_score=0.5,
            relevance_score=0.6,
            engagement_score=0.4,
            professional_score=0.7,
            time_efficiency_score=0.5,
            overall_score=0.54,
            improvement_suggestions=["Improve clarity", "Add engagement"]
        )
        
        enhanced = self.assessor.enhance_presentation(self.mock_presentation, low_metrics)
        
        assert isinstance(enhanced, BoardPresentation)
        assert enhanced.id == self.mock_presentation.id
    
    def test_improvement_suggestions_generation(self):
        """Test improvement suggestions generation"""
        suggestions = self.assessor._generate_improvement_suggestions(
            0.5, 0.6, 0.4, 0.8, 0.7  # Low scores for first three metrics
        )
        
        assert isinstance(suggestions, list)
        assert len(suggestions) >= 3  # Should suggest improvements for low scores
        assert any("clarity" in suggestion.lower() for suggestion in suggestions)
        assert any("relevance" in suggestion.lower() for suggestion in suggestions)
        assert any("engagement" in suggestion.lower() for suggestion in suggestions)


class TestBoardPresentationFramework:
    """Test cases for BoardPresentationFramework"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.framework = BoardPresentationFramework()
        self.mock_board = Board(
            id="test_board_1",
            name="Test Board",
            composition={},
            governance_structure={},
            meeting_patterns={},
            decision_processes={}
        )
        self.test_content = {
            "title": "Q4 Board Review",
            "overview": "Comprehensive quarterly performance review",
            "financial_performance": {
                "revenue": 15000000,
                "profit_margin": 18.5,
                "growth_rate": 12.3
            },
            "strategic_initiatives": [
                "Digital transformation program",
                "Market expansion into APAC",
                "Sustainability initiative launch"
            ],
            "risk_assessment": {
                "market_risks": "High competition in core markets",
                "operational_risks": "Supply chain vulnerabilities",
                "financial_risks": "Currency fluctuation exposure"
            },
            "recommendations": [
                "Accelerate digital transformation timeline",
                "Establish strategic partnerships in APAC",
                "Implement comprehensive risk management framework"
            ],
            "next_steps": "Board approval for strategic initiatives and budget allocation"
        }
    
    def test_create_optimized_presentation_success(self):
        """Test successful creation of optimized presentation"""
        presentation = self.framework.create_optimized_presentation(
            content=self.test_content,
            board=self.mock_board,
            format_type=PresentationFormat.STRATEGIC_OVERVIEW
        )
        
        assert isinstance(presentation, BoardPresentation)
        assert presentation.title == "Q4 Board Review"
        assert presentation.board_id == "test_board_1"
        assert presentation.format_type == PresentationFormat.STRATEGIC_OVERVIEW
        assert len(presentation.slides) > 0
        assert presentation.quality_score is not None
        assert presentation.quality_score > 0.0
        assert len(presentation.key_messages) > 0
        assert len(presentation.success_metrics) > 0
    
    def test_create_executive_summary_format(self):
        """Test creation with executive summary format"""
        presentation = self.framework.create_optimized_presentation(
            content=self.test_content,
            board=self.mock_board,
            format_type=PresentationFormat.EXECUTIVE_SUMMARY
        )
        
        assert presentation.format_type == PresentationFormat.EXECUTIVE_SUMMARY
        # Executive summary should be more concise
        total_duration = sum(slide.estimated_duration for slide in presentation.slides)
        assert total_duration <= 1200  # Should be under 20 minutes
    
    def test_create_financial_report_format(self):
        """Test creation with financial report format"""
        financial_content = {
            **self.test_content,
            "financial_statements": {
                "income_statement": {"revenue": 15000000, "expenses": 12000000},
                "balance_sheet": {"assets": 50000000, "liabilities": 20000000},
                "cash_flow": {"operating": 5000000, "investing": -2000000}
            }
        }
        
        presentation = self.framework.create_optimized_presentation(
            content=financial_content,
            board=self.mock_board,
            format_type=PresentationFormat.FINANCIAL_REPORT
        )
        
        assert presentation.format_type == PresentationFormat.FINANCIAL_REPORT
        # Should include financial-specific success metrics
        assert any("financial" in metric.lower() for metric in presentation.success_metrics)
    
    def test_quality_score_calculation(self):
        """Test quality score calculation and optimization"""
        presentation = self.framework.create_optimized_presentation(
            content=self.test_content,
            board=self.mock_board
        )
        
        assert presentation.quality_score is not None
        assert 0.0 <= presentation.quality_score <= 1.0
        
        # High-quality content should result in good score
        assert presentation.quality_score >= 0.6
    
    def test_enhancement_for_low_quality(self):
        """Test automatic enhancement for low-quality presentations"""
        # Create minimal content that should trigger enhancement
        minimal_content = {
            "title": "Brief Update",
            "content": "Short update"
        }
        
        presentation = self.framework.create_optimized_presentation(
            content=minimal_content,
            board=self.mock_board
        )
        
        # Should still create a valid presentation
        assert isinstance(presentation, BoardPresentation)
        assert len(presentation.slides) > 0
        assert presentation.quality_score is not None
    
    def test_board_specific_optimization(self):
        """Test optimization based on board characteristics"""
        # Create board with specific characteristics
        board_with_members = Board(
            id="specialized_board",
            name="Tech-Focused Board",
            composition={"tech_experts": 3, "financial_experts": 2},
            governance_structure={"committees": ["audit", "technology"]},
            meeting_patterns={"frequency": "monthly", "duration": 120},
            decision_processes={"consensus_required": True}
        )
        
        presentation = self.framework.create_optimized_presentation(
            content=self.test_content,
            board=board_with_members
        )
        
        assert presentation.board_id == "specialized_board"
        # Should be optimized for the specific board
        assert len(presentation.slides) > 0
    
    def test_error_handling_invalid_content(self):
        """Test error handling with invalid content"""
        with pytest.raises(Exception):
            self.framework.create_optimized_presentation(
                content=None,
                board=self.mock_board
            )
    
    def test_error_handling_invalid_board(self):
        """Test error handling with invalid board"""
        with pytest.raises(Exception):
            self.framework.create_optimized_presentation(
                content=self.test_content,
                board=None
            )
    
    @patch('scrollintel.engines.presentation_design_engine.logging')
    def test_logging_integration(self, mock_logging):
        """Test logging integration"""
        presentation = self.framework.create_optimized_presentation(
            content=self.test_content,
            board=self.mock_board
        )
        
        # Should log successful creation
        assert presentation is not None


class TestIntegrationScenarios:
    """Integration test scenarios for board presentation framework"""
    
    def setup_method(self):
        """Set up integration test fixtures"""
        self.framework = BoardPresentationFramework()
    
    def test_complete_board_presentation_workflow(self):
        """Test complete workflow from content to optimized presentation"""
        # Simulate real board scenario
        board = Board(
            id="public_company_board",
            name="Public Company Board of Directors",
            composition={
                "independent_directors": 5,
                "executive_directors": 2,
                "investor_representatives": 2
            },
            governance_structure={
                "committees": ["audit", "compensation", "nominating"],
                "meeting_frequency": "quarterly",
                "governance_standards": "NYSE"
            },
            meeting_patterns={
                "typical_duration": 180,  # 3 hours
                "presentation_time": 45,  # 45 minutes for presentations
                "q_and_a_time": 30      # 30 minutes for Q&A
            },
            decision_processes={
                "voting_threshold": "majority",
                "consensus_building": True,
                "documentation_required": True
            }
        )
        
        # Comprehensive quarterly content
        quarterly_content = {
            "title": "Q4 2024 Board Presentation",
            "presenter_id": "cto_scrollintel",
            "executive_summary": "Strong Q4 performance with strategic initiatives on track",
            "financial_highlights": {
                "revenue": 25000000,
                "revenue_growth": 18.5,
                "gross_margin": 72.3,
                "operating_margin": 15.8,
                "cash_position": 45000000
            },
            "operational_metrics": {
                "customer_acquisition": 1250,
                "customer_retention": 94.2,
                "employee_satisfaction": 87,
                "product_uptime": 99.97
            },
            "strategic_progress": {
                "ai_platform_development": "On track - 85% complete",
                "market_expansion": "Ahead of schedule - 3 new regions",
                "partnership_program": "Exceeded targets - 15 new partners"
            },
            "risk_management": {
                "cybersecurity": "Enhanced protocols implemented",
                "regulatory_compliance": "Full compliance maintained",
                "market_competition": "Competitive positioning strengthened",
                "talent_retention": "Retention programs showing positive results"
            },
            "forward_looking": {
                "2025_outlook": "Continued growth trajectory expected",
                "investment_priorities": ["R&D expansion", "International growth", "Talent acquisition"],
                "market_opportunities": ["Enterprise AI adoption", "Vertical market expansion"]
            },
            "board_decisions_required": [
                "Approval of 2025 budget allocation",
                "Authorization of international expansion",
                "Endorsement of strategic partnership framework"
            ]
        }
        
        # Create optimized presentation
        presentation = self.framework.create_optimized_presentation(
            content=quarterly_content,
            board=board,
            format_type=PresentationFormat.STRATEGIC_OVERVIEW
        )
        
        # Validate comprehensive presentation
        assert presentation.title == "Q4 2024 Board Presentation"
        assert presentation.board_id == "public_company_board"
        assert len(presentation.slides) >= 5  # Should have multiple slides for comprehensive content
        assert presentation.quality_score >= 0.7  # Should be high quality
        
        # Validate timing fits board constraints
        total_duration = sum(slide.estimated_duration for slide in presentation.slides)
        assert total_duration <= 2700  # Should fit within 45-minute presentation slot
        
        # Validate key messages are strategic
        assert len(presentation.key_messages) > 0
        assert any("strategic" in message.lower() or "growth" in message.lower() 
                  for message in presentation.key_messages)
        
        # Validate success metrics are board-appropriate
        assert len(presentation.success_metrics) > 0
        assert any("board" in metric.lower() or "decision" in metric.lower() 
                  for metric in presentation.success_metrics)
        
        # Validate slides have interaction points for board engagement
        interaction_slides = [slide for slide in presentation.slides if slide.interaction_points]
        assert len(interaction_slides) > 0
    
    def test_crisis_communication_presentation(self):
        """Test presentation creation for crisis communication scenario"""
        crisis_board = Board(
            id="crisis_board",
            name="Crisis Management Board",
            composition={"crisis_experts": 3, "legal_experts": 2, "pr_experts": 1},
            governance_structure={"emergency_protocols": True},
            meeting_patterns={"emergency_meeting": True, "duration": 60},
            decision_processes={"rapid_decision": True}
        )
        
        crisis_content = {
            "title": "Emergency Board Communication - Security Incident",
            "situation_overview": "Data security incident detected and contained",
            "immediate_actions": [
                "Incident response team activated",
                "Affected systems isolated",
                "Customers notified within 24 hours",
                "Regulatory authorities informed"
            ],
            "impact_assessment": {
                "customers_affected": 1500,
                "data_compromised": "Email addresses only",
                "financial_impact": "Estimated $2M in response costs",
                "reputation_risk": "Moderate - proactive response"
            },
            "remediation_plan": {
                "technical_fixes": "Security patches deployed",
                "process_improvements": "Enhanced monitoring implemented",
                "customer_support": "Dedicated support team assigned",
                "legal_compliance": "Full regulatory compliance maintained"
            },
            "board_actions_needed": [
                "Approve additional security investment",
                "Authorize external security audit",
                "Endorse customer communication strategy"
            ]
        }
        
        presentation = self.framework.create_optimized_presentation(
            content=crisis_content,
            board=crisis_board,
            format_type=PresentationFormat.RISK_ASSESSMENT
        )
        
        # Crisis presentations should be concise and action-oriented
        assert presentation.format_type == PresentationFormat.RISK_ASSESSMENT
        total_duration = sum(slide.estimated_duration for slide in presentation.slides)
        assert total_duration <= 1800  # Should fit within 30 minutes for urgent decision
        
        # Should emphasize critical information
        critical_sections = []
        for slide in presentation.slides:
            critical_sections.extend([s for s in slide.sections if s.importance_level == "critical"])
        assert len(critical_sections) > 0
    
    def test_investor_presentation_scenario(self):
        """Test presentation optimized for investor audience"""
        investor_board = Board(
            id="investor_board",
            name="Investor-Heavy Board",
            composition={"venture_partners": 4, "independent_directors": 3},
            governance_structure={"investor_rights": True, "board_observer_seats": 2},
            meeting_patterns={"quarterly_reviews": True, "duration": 120},
            decision_processes={"investor_approval_required": True}
        )
        
        investor_content = {
            "title": "Series B Performance Review",
            "financial_performance": {
                "arr": 12000000,  # Annual Recurring Revenue
                "growth_rate": 150,
                "burn_rate": 800000,
                "runway": 18,  # months
                "ltv_cac_ratio": 4.2
            },
            "market_traction": {
                "customer_count": 450,
                "enterprise_customers": 85,
                "market_penetration": 12,
                "competitive_wins": 23
            },
            "product_milestones": {
                "feature_releases": 8,
                "platform_stability": 99.95,
                "customer_satisfaction": 92,
                "product_market_fit_score": 78
            },
            "team_scaling": {
                "headcount": 125,
                "engineering_team": 45,
                "sales_team": 25,
                "key_hires": ["VP Engineering", "Head of Sales"]
            },
            "funding_utilization": {
                "series_b_deployed": 65,  # percentage
                "key_investments": ["Product development", "Sales expansion", "Market entry"],
                "efficiency_metrics": "Improved CAC by 30%"
            },
            "next_milestones": {
                "revenue_targets": "Reach $20M ARR by Q4",
                "market_expansion": "Enter European market",
                "product_roadmap": "Launch enterprise platform"
            }
        }
        
        presentation = self.framework.create_optimized_presentation(
            content=investor_content,
            board=investor_board,
            format_type=PresentationFormat.FINANCIAL_REPORT
        )
        
        # Investor presentations should focus on metrics and growth
        assert presentation.format_type == PresentationFormat.FINANCIAL_REPORT
        
        # Should include financial-focused key messages
        financial_messages = [msg for msg in presentation.key_messages 
                            if any(term in msg.lower() for term in ['revenue', 'growth', 'market', 'customer'])]
        assert len(financial_messages) > 0
        
        # Should have appropriate success metrics for investors
        investor_metrics = [metric for metric in presentation.success_metrics 
                          if any(term in metric.lower() for term in ['financial', 'growth', 'performance'])]
        assert len(investor_metrics) > 0