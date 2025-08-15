"""
Tests for Q&A Preparation Engine

This module contains comprehensive tests for the Q&A preparation system.
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch

from scrollintel.engines.qa_preparation_engine import (
    QAPreparationEngine, BoardQuestionAnticipator, ResponseDeveloper,
    QAEffectivenessTracker, AnticipatedQuestion, PreparedResponse,
    QAPreparation, QuestionCategory, QuestionDifficulty, ResponseStrategy
)
from scrollintel.engines.presentation_design_engine import PresentationDesigner
from scrollintel.models.board_dynamics_models import Board


class TestBoardQuestionAnticipator:
    """Test cases for BoardQuestionAnticipator"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.anticipator = BoardQuestionAnticipator()
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
            "financial_performance": {
                "revenue": 15000000,
                "growth_rate": 18.5,
                "profit_margin": 22.3
            },
            "strategic_initiatives": [
                "Digital transformation",
                "Market expansion",
                "Product innovation"
            ],
            "operational_metrics": {
                "efficiency": 89,
                "customer_satisfaction": 92
            }
        }
        self.mock_presentation = designer.create_board_presentation(
            content=test_content,
            board=self.mock_board
        )
    
    def test_anticipate_board_questions_success(self):
        """Test successful board question anticipation"""
        questions = self.anticipator.anticipate_board_questions(
            presentation=self.mock_presentation,
            board=self.mock_board
        )
        
        assert isinstance(questions, list)
        assert len(questions) > 0
        assert len(questions) <= 15  # Should limit to manageable number
        
        for question in questions:
            assert isinstance(question, AnticipatedQuestion)
            assert question.id is not None
            assert question.question_text is not None
            assert isinstance(question.category, QuestionCategory)
            assert isinstance(question.difficulty, QuestionDifficulty)
            assert 0.0 <= question.likelihood <= 1.0
            assert question.context is not None
    
    def test_analyze_presentation_content(self):
        """Test presentation content analysis for questions"""
        content_questions = self.anticipator._analyze_presentation_content(self.mock_presentation)
        
        assert isinstance(content_questions, list)
        # Should generate questions from financial and strategic content
        categories = [q.category for q in content_questions]
        assert any(cat in [QuestionCategory.FINANCIAL_PERFORMANCE, QuestionCategory.STRATEGIC_DIRECTION] 
                  for cat in categories)
    
    def test_create_financial_question(self):
        """Test financial question creation"""
        # Create mock section with revenue data
        mock_section = Mock()
        mock_section.content = "Revenue: $15,000,000"
        mock_section.content_type = "metric"
        mock_section.importance_level = "critical"
        
        question = self.anticipator._create_financial_question(mock_section, "Financial Performance")
        
        assert isinstance(question, AnticipatedQuestion)
        assert question.category == QuestionCategory.FINANCIAL_PERFORMANCE
        assert "revenue" in question.question_text.lower()
        assert question.likelihood > 0.5
        assert len(question.potential_follow_ups) > 0
    
    def test_create_strategic_question(self):
        """Test strategic question creation"""
        question = self.anticipator._create_strategic_question("Strategic Growth Initiative")
        
        assert isinstance(question, AnticipatedQuestion)
        assert question.category == QuestionCategory.STRATEGIC_DIRECTION
        assert "strategic" in question.question_text.lower()
        assert question.likelihood > 0.5
        assert len(question.potential_follow_ups) > 0
    
    def test_generate_member_specific_questions(self):
        """Test generation of member-specific questions"""
        member_questions = self.anticipator._generate_member_specific_questions(
            self.mock_board, self.mock_presentation
        )
        
        assert isinstance(member_questions, list)
        assert len(member_questions) > 0
        
        # Should cover different member types
        member_sources = [q.board_member_source for q in member_questions]
        assert len(set(member_sources)) > 1  # Multiple member types
    
    def test_generate_contextual_questions(self):
        """Test contextual question generation"""
        board_context = {
            "meeting_type": "quarterly_review",
            "focus_areas": ["risk_management", "strategic_planning"]
        }
        
        contextual_questions = self.anticipator._generate_contextual_questions(
            board_context, self.mock_presentation
        )
        
        assert isinstance(contextual_questions, list)
        # Should generate questions based on context
        if contextual_questions:
            categories = [q.category for q in contextual_questions]
            assert any(cat in [QuestionCategory.RISK_MANAGEMENT, QuestionCategory.STRATEGIC_DIRECTION] 
                      for cat in categories)
    
    def test_generate_governance_questions(self):
        """Test governance question generation"""
        governance_questions = self.anticipator._generate_governance_questions(self.mock_presentation)
        
        assert isinstance(governance_questions, list)
        assert len(governance_questions) > 0
        
        for question in governance_questions:
            assert question.category == QuestionCategory.GOVERNANCE_COMPLIANCE
            assert "oversight" in question.question_text.lower() or "governance" in question.question_text.lower()
    
    def test_prioritize_questions(self):
        """Test question prioritization"""
        # Create questions with different likelihoods
        questions = [
            AnticipatedQuestion(
                id="q1", question_text="High likelihood question",
                category=QuestionCategory.FINANCIAL_PERFORMANCE,
                difficulty=QuestionDifficulty.MODERATE,
                likelihood=0.9, board_member_source="test", context="test"
            ),
            AnticipatedQuestion(
                id="q2", question_text="Low likelihood question",
                category=QuestionCategory.OPERATIONAL_EFFICIENCY,
                difficulty=QuestionDifficulty.STRAIGHTFORWARD,
                likelihood=0.3, board_member_source="test", context="test"
            )
        ]
        
        prioritized = self.anticipator._prioritize_questions(questions, self.mock_board)
        
        assert len(prioritized) <= 15
        # Should be sorted by likelihood
        if len(prioritized) > 1:
            assert prioritized[0].likelihood >= prioritized[1].likelihood
    
    def test_identify_summary_gaps(self):
        """Test identification of gaps in executive summary"""
        # Summary without risk discussion
        summary_without_risk = "Strong financial performance with strategic growth initiatives"
        
        gap_questions = self.anticipator._identify_summary_gaps(summary_without_risk)
        
        assert isinstance(gap_questions, list)
        # Should identify risk gap
        risk_questions = [q for q in gap_questions if q.category == QuestionCategory.RISK_MANAGEMENT]
        assert len(risk_questions) > 0


class TestResponseDeveloper:
    """Test cases for ResponseDeveloper"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.response_developer = ResponseDeveloper()
        
        # Create mock questions
        self.mock_questions = [
            AnticipatedQuestion(
                id="q1",
                question_text="What are the key drivers behind our revenue growth?",
                category=QuestionCategory.FINANCIAL_PERFORMANCE,
                difficulty=QuestionDifficulty.MODERATE,
                likelihood=0.8,
                board_member_source="financial_expert",
                context="Financial performance inquiry"
            ),
            AnticipatedQuestion(
                id="q2",
                question_text="How does our strategic direction align with market trends?",
                category=QuestionCategory.STRATEGIC_DIRECTION,
                difficulty=QuestionDifficulty.CHALLENGING,
                likelihood=0.7,
                board_member_source="independent_director",
                context="Strategic alignment question"
            )
        ]
        
        # Create mock presentation
        designer = PresentationDesigner()
        test_content = {
            "revenue_analysis": "Strong revenue growth driven by market expansion",
            "strategic_overview": "Strategic direction focused on innovation and growth"
        }
        self.mock_presentation = designer.create_board_presentation(
            content=test_content,
            board=Board(id="test", name="Test", composition={}, governance_structure={}, 
                       meeting_patterns={}, decision_processes={})
        )
    
    def test_develop_comprehensive_responses_success(self):
        """Test successful response development"""
        responses = self.response_developer.develop_comprehensive_responses(
            questions=self.mock_questions,
            presentation=self.mock_presentation
        )
        
        assert isinstance(responses, list)
        assert len(responses) == len(self.mock_questions)
        
        for response in responses:
            assert isinstance(response, PreparedResponse)
            assert response.id is not None
            assert response.question_id in [q.id for q in self.mock_questions]
            assert response.primary_response is not None
            assert isinstance(response.response_strategy, ResponseStrategy)
            assert isinstance(response.key_messages, list)
            assert isinstance(response.potential_challenges, list)
            assert isinstance(response.backup_responses, list)
            assert 0.0 <= response.confidence_level <= 1.0
            assert response.preparation_notes is not None
    
    def test_determine_response_strategy(self):
        """Test response strategy determination"""
        financial_question = self.mock_questions[0]  # Financial performance question
        strategy = self.response_developer._determine_response_strategy(financial_question)
        assert strategy == ResponseStrategy.DATA_DRIVEN
        
        strategic_question = self.mock_questions[1]  # Strategic direction question
        strategy = self.response_developer._determine_response_strategy(strategic_question)
        assert strategy == ResponseStrategy.STRATEGIC_CONTEXT
    
    def test_create_data_driven_response(self):
        """Test data-driven response creation"""
        financial_question = self.mock_questions[0]
        response = self.response_developer._create_data_driven_response(
            financial_question, self.mock_presentation
        )
        
        assert isinstance(response, str)
        assert len(response) > 0
        assert "data" in response.lower() or "analysis" in response.lower()
    
    def test_create_strategic_response(self):
        """Test strategic response creation"""
        strategic_question = self.mock_questions[1]
        response = self.response_developer._create_strategic_response(
            strategic_question, self.mock_presentation
        )
        
        assert isinstance(response, str)
        assert len(response) > 0
        assert "strategic" in response.lower()
    
    def test_gather_supporting_data(self):
        """Test supporting data gathering"""
        question = self.mock_questions[0]
        supporting_data = self.response_developer._gather_supporting_data(
            question, self.mock_presentation
        )
        
        assert isinstance(supporting_data, dict)
        # Should include contextual data
        assert len(supporting_data) > 0
    
    def test_calculate_confidence_level(self):
        """Test confidence level calculation"""
        question = self.mock_questions[0]
        supporting_data = {"revenue_data": "Strong performance"}
        
        confidence = self.response_developer._calculate_confidence_level(
            question, supporting_data
        )
        
        assert 0.0 <= confidence <= 1.0
    
    def test_create_backup_responses(self):
        """Test backup response creation"""
        primary_response = "Our revenue growth is driven by strategic market expansion and product innovation."
        backup_responses = self.response_developer._create_backup_responses(
            self.mock_questions[0], primary_response
        )
        
        assert isinstance(backup_responses, list)
        assert len(backup_responses) > 0
        # Should include shorter and more detailed versions
        assert any(len(response) < len(primary_response) for response in backup_responses)


class TestQAEffectivenessTracker:
    """Test cases for QAEffectivenessTracker"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.effectiveness_tracker = QAEffectivenessTracker()
        
        # Create mock Q&A preparation
        self.mock_qa_preparation = QAPreparation(
            id="qa_prep_1",
            presentation_id="pres_1",
            board_id="board_1",
            anticipated_questions=[
                AnticipatedQuestion(
                    id="q1", question_text="What drives revenue growth?",
                    category=QuestionCategory.FINANCIAL_PERFORMANCE,
                    difficulty=QuestionDifficulty.MODERATE,
                    likelihood=0.8, board_member_source="cfo", context="financial"
                ),
                AnticipatedQuestion(
                    id="q2", question_text="How do we manage strategic risks?",
                    category=QuestionCategory.RISK_MANAGEMENT,
                    difficulty=QuestionDifficulty.CHALLENGING,
                    likelihood=0.7, board_member_source="risk_committee", context="risk"
                )
            ],
            prepared_responses=[
                PreparedResponse(
                    id="r1", question_id="q1", primary_response="Revenue growth is driven by...",
                    response_strategy=ResponseStrategy.DATA_DRIVEN,
                    key_messages=["Strong fundamentals", "Market expansion"],
                    potential_challenges=[], backup_responses=[],
                    confidence_level=0.8, preparation_notes="Well prepared"
                ),
                PreparedResponse(
                    id="r2", question_id="q2", primary_response="Risk management approach includes...",
                    response_strategy=ResponseStrategy.ACKNOWLEDGE_AND_PLAN,
                    key_messages=["Comprehensive framework", "Continuous monitoring"],
                    potential_challenges=[], backup_responses=[],
                    confidence_level=0.7, preparation_notes="Moderate preparation"
                )
            ],
            question_categories_covered=[QuestionCategory.FINANCIAL_PERFORMANCE, QuestionCategory.RISK_MANAGEMENT],
            overall_preparedness_score=0.75,
            high_risk_questions=["How do we manage strategic risks?"],
            key_talking_points=["Strong fundamentals", "Market expansion"]
        )
    
    def test_track_qa_effectiveness_success(self):
        """Test successful Q&A effectiveness tracking"""
        effectiveness_metrics = self.effectiveness_tracker.track_qa_effectiveness(
            qa_preparation=self.mock_qa_preparation
        )
        
        assert isinstance(effectiveness_metrics, dict)
        assert "preparation_completeness" in effectiveness_metrics
        assert "question_prediction_accuracy" in effectiveness_metrics
        assert "response_quality_score" in effectiveness_metrics
        assert "board_satisfaction_score" in effectiveness_metrics
        assert "areas_for_improvement" in effectiveness_metrics
        assert "success_indicators" in effectiveness_metrics
        
        # All scores should be between 0 and 1
        for key in ["preparation_completeness", "response_quality_score"]:
            assert 0.0 <= effectiveness_metrics[key] <= 1.0
    
    def test_calculate_preparation_completeness(self):
        """Test preparation completeness calculation"""
        completeness = self.effectiveness_tracker._calculate_preparation_completeness(
            self.mock_qa_preparation
        )
        
        assert 0.0 <= completeness <= 1.0
        # Should be reasonably high with 2 questions and responses
        assert completeness > 0.3
    
    def test_calculate_prediction_accuracy(self):
        """Test question prediction accuracy calculation"""
        actual_questions = [
            "What drives our revenue growth this quarter?",  # Similar to anticipated
            "How do we handle competitive threats?"  # Different from anticipated
        ]
        
        accuracy = self.effectiveness_tracker._calculate_prediction_accuracy(
            self.mock_qa_preparation.anticipated_questions,
            actual_questions
        )
        
        assert 0.0 <= accuracy <= 1.0
        # Should have some accuracy due to similar question
        assert accuracy > 0.0
    
    def test_questions_match(self):
        """Test question matching logic"""
        actual = "What are the key drivers of revenue growth?"
        anticipated = "What drives revenue growth?"
        
        match = self.effectiveness_tracker._questions_match(actual, anticipated)
        assert match is True
        
        # Test non-matching questions
        actual_different = "How do we manage employee satisfaction?"
        match_different = self.effectiveness_tracker._questions_match(actual_different, anticipated)
        assert match_different is False
    
    def test_calculate_response_quality(self):
        """Test response quality calculation"""
        quality = self.effectiveness_tracker._calculate_response_quality(
            self.mock_qa_preparation.prepared_responses
        )
        
        assert 0.0 <= quality <= 1.0
        # Should be reasonably high with prepared responses
        assert quality > 0.5
    
    def test_incorporate_board_feedback(self):
        """Test board feedback incorporation"""
        metrics = {"preparation_completeness": 0.8}
        board_feedback = {
            "satisfaction_score": 0.9,
            "response_clarity": 0.85,
            "improvement_suggestions": ["More detailed financial analysis"]
        }
        
        updated_metrics = self.effectiveness_tracker._incorporate_board_feedback(
            metrics, board_feedback
        )
        
        assert "response_clarity_score" in updated_metrics
        assert "board_improvement_suggestions" in updated_metrics
        assert updated_metrics["response_clarity_score"] == 0.85
    
    def test_identify_improvement_areas(self):
        """Test improvement area identification"""
        low_metrics = {
            "preparation_completeness": 0.5,
            "question_prediction_accuracy": 0.4,
            "response_quality_score": 0.6,
            "board_satisfaction_score": 0.7
        }
        
        improvements = self.effectiveness_tracker._identify_improvement_areas(low_metrics)
        
        assert isinstance(improvements, list)
        assert len(improvements) > 0
        # Should identify areas with low scores
        assert any("preparation" in imp.lower() for imp in improvements)
        assert any("prediction" in imp.lower() for imp in improvements)
    
    def test_identify_success_indicators(self):
        """Test success indicator identification"""
        high_metrics = {
            "preparation_completeness": 0.9,
            "question_prediction_accuracy": 0.8,
            "response_quality_score": 0.85,
            "board_satisfaction_score": 0.9
        }
        
        successes = self.effectiveness_tracker._identify_success_indicators(high_metrics)
        
        assert isinstance(successes, list)
        assert len(successes) > 0
        # Should identify successful areas
        assert any("preparation" in success.lower() for success in successes)
        assert any("satisfaction" in success.lower() for success in successes)


class TestQAPreparationEngine:
    """Test cases for QAPreparationEngine"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.qa_engine = QAPreparationEngine()
        self.mock_board = Board(
            id="test_board_1",
            name="Test Board",
            composition={"independent_directors": 3, "executive_directors": 2},
            governance_structure={"committees": ["audit", "compensation"]},
            meeting_patterns={"frequency": "quarterly"},
            decision_processes={"consensus_required": True}
        )
        
        # Create comprehensive presentation
        designer = PresentationDesigner()
        comprehensive_content = {
            "executive_summary": "Strong Q4 performance with strategic initiatives on track",
            "financial_highlights": {
                "revenue": 25000000,
                "revenue_growth": 18.5,
                "profit_margin": 22.3,
                "cash_position": 45000000
            },
            "strategic_progress": {
                "digital_transformation": "85% complete",
                "market_expansion": "3 new regions launched",
                "product_innovation": "5 new features released"
            },
            "operational_metrics": {
                "customer_satisfaction": 92,
                "employee_engagement": 87,
                "operational_efficiency": 89
            },
            "risk_assessment": {
                "market_risks": "Competitive pressure increasing",
                "operational_risks": "Supply chain vulnerabilities",
                "financial_risks": "Currency fluctuation exposure"
            },
            "forward_outlook": {
                "2024_targets": "Achieve $30M revenue",
                "strategic_priorities": ["Scale operations", "International expansion"],
                "investment_areas": ["R&D", "Talent acquisition"]
            }
        }
        
        self.mock_presentation = designer.create_board_presentation(
            content=comprehensive_content,
            board=self.mock_board
        )
    
    def test_create_comprehensive_qa_preparation_success(self):
        """Test successful comprehensive Q&A preparation creation"""
        board_context = {
            "meeting_type": "quarterly_board_meeting",
            "focus_areas": ["financial_performance", "strategic_execution", "risk_management"],
            "time_allocation": 60
        }
        
        qa_preparation = self.qa_engine.create_comprehensive_qa_preparation(
            presentation=self.mock_presentation,
            board=self.mock_board,
            board_context=board_context
        )
        
        assert isinstance(qa_preparation, QAPreparation)
        assert qa_preparation.id is not None
        assert qa_preparation.presentation_id == self.mock_presentation.id
        assert qa_preparation.board_id == self.mock_board.id
        
        # Should have anticipated questions
        assert len(qa_preparation.anticipated_questions) > 0
        assert len(qa_preparation.anticipated_questions) <= 15
        
        # Should have prepared responses
        assert len(qa_preparation.prepared_responses) > 0
        assert len(qa_preparation.prepared_responses) <= len(qa_preparation.anticipated_questions)
        
        # Should cover multiple categories
        assert len(qa_preparation.question_categories_covered) > 1
        
        # Should have reasonable preparedness score
        assert 0.0 <= qa_preparation.overall_preparedness_score <= 1.0
        
        # Should identify high-risk questions
        assert isinstance(qa_preparation.high_risk_questions, list)
        
        # Should have key talking points
        assert len(qa_preparation.key_talking_points) > 0
    
    def test_financial_focus_qa_preparation(self):
        """Test Q&A preparation with financial focus"""
        financial_context = {
            "meeting_type": "financial_review",
            "focus_areas": ["financial_performance", "profitability", "cash_management"],
            "board_composition": {"financial_experts": 4, "independent_directors": 3}
        }
        
        qa_preparation = self.qa_engine.create_comprehensive_qa_preparation(
            presentation=self.mock_presentation,
            board=self.mock_board,
            board_context=financial_context
        )
        
        # Should have financial-focused questions
        financial_questions = [q for q in qa_preparation.anticipated_questions 
                             if q.category == QuestionCategory.FINANCIAL_PERFORMANCE]
        assert len(financial_questions) > 0
        
        # Should have data-driven responses
        data_driven_responses = [r for r in qa_preparation.prepared_responses 
                               if r.response_strategy == ResponseStrategy.DATA_DRIVEN]
        assert len(data_driven_responses) > 0
    
    def test_strategic_focus_qa_preparation(self):
        """Test Q&A preparation with strategic focus"""
        strategic_context = {
            "meeting_type": "strategic_planning_session",
            "focus_areas": ["strategic_direction", "competitive_position", "market_expansion"],
            "board_composition": {"industry_experts": 3, "independent_directors": 4}
        }
        
        qa_preparation = self.qa_engine.create_comprehensive_qa_preparation(
            presentation=self.mock_presentation,
            board=self.mock_board,
            board_context=strategic_context
        )
        
        # Should have strategic-focused questions
        strategic_questions = [q for q in qa_preparation.anticipated_questions 
                             if q.category == QuestionCategory.STRATEGIC_DIRECTION]
        assert len(strategic_questions) > 0
        
        # Should have strategic context responses
        strategic_responses = [r for r in qa_preparation.prepared_responses 
                             if r.response_strategy == ResponseStrategy.STRATEGIC_CONTEXT]
        assert len(strategic_responses) > 0
    
    def test_risk_management_qa_preparation(self):
        """Test Q&A preparation with risk management focus"""
        risk_context = {
            "meeting_type": "risk_committee_meeting",
            "focus_areas": ["risk_management", "compliance", "governance"],
            "urgency_level": "high"
        }
        
        qa_preparation = self.qa_engine.create_comprehensive_qa_preparation(
            presentation=self.mock_presentation,
            board=self.mock_board,
            board_context=risk_context
        )
        
        # Should have risk-focused questions
        risk_questions = [q for q in qa_preparation.anticipated_questions 
                         if q.category == QuestionCategory.RISK_MANAGEMENT]
        assert len(risk_questions) > 0
        
        # Should identify risk questions as high-risk
        assert len(qa_preparation.high_risk_questions) > 0
        
        # Should have acknowledge and plan responses
        risk_responses = [r for r in qa_preparation.prepared_responses 
                         if r.response_strategy == ResponseStrategy.ACKNOWLEDGE_AND_PLAN]
        assert len(risk_responses) > 0
    
    def test_calculate_preparedness_score(self):
        """Test preparedness score calculation"""
        # Create questions and responses
        questions = [
            AnticipatedQuestion(
                id="q1", question_text="Test question 1",
                category=QuestionCategory.FINANCIAL_PERFORMANCE,
                difficulty=QuestionDifficulty.MODERATE,
                likelihood=0.8, board_member_source="test", context="test"
            ),
            AnticipatedQuestion(
                id="q2", question_text="Test question 2",
                category=QuestionCategory.STRATEGIC_DIRECTION,
                difficulty=QuestionDifficulty.CRITICAL,
                likelihood=0.9, board_member_source="test", context="test"
            )
        ]
        
        responses = [
            PreparedResponse(
                id="r1", question_id="q1", primary_response="Response 1",
                response_strategy=ResponseStrategy.DATA_DRIVEN,
                key_messages=[], potential_challenges=[], backup_responses=[],
                confidence_level=0.8, preparation_notes=""
            ),
            PreparedResponse(
                id="r2", question_id="q2", primary_response="Response 2",
                response_strategy=ResponseStrategy.STRATEGIC_CONTEXT,
                key_messages=[], potential_challenges=[], backup_responses=[],
                confidence_level=0.9, preparation_notes=""
            )
        ]
        
        score = self.qa_engine._calculate_preparedness_score(questions, responses)
        
        assert 0.0 <= score <= 1.0
        # Should be high with full response coverage and high confidence
        assert score > 0.7
    
    def test_identify_high_risk_questions(self):
        """Test high-risk question identification"""
        questions = [
            AnticipatedQuestion(
                id="q1", question_text="Normal question",
                category=QuestionCategory.OPERATIONAL_EFFICIENCY,
                difficulty=QuestionDifficulty.MODERATE,
                likelihood=0.6, board_member_source="test", context="test"
            ),
            AnticipatedQuestion(
                id="q2", question_text="Critical risk question",
                category=QuestionCategory.RISK_MANAGEMENT,
                difficulty=QuestionDifficulty.CRITICAL,
                likelihood=0.9, board_member_source="test", context="test"
            ),
            AnticipatedQuestion(
                id="q3", question_text="High likelihood question",
                category=QuestionCategory.FINANCIAL_PERFORMANCE,
                difficulty=QuestionDifficulty.MODERATE,
                likelihood=0.95, board_member_source="test", context="test"
            )
        ]
        
        high_risk = self.qa_engine._identify_high_risk_questions(questions)
        
        assert isinstance(high_risk, list)
        assert len(high_risk) > 0
        # Should include critical and high-likelihood questions
        assert "Critical risk question" in high_risk
        assert "High likelihood question" in high_risk
    
    def test_extract_key_talking_points(self):
        """Test key talking points extraction"""
        responses = [
            PreparedResponse(
                id="r1", question_id="q1", primary_response="Response 1",
                response_strategy=ResponseStrategy.DATA_DRIVEN,
                key_messages=["Strong performance", "Market leadership"],
                potential_challenges=[], backup_responses=[],
                confidence_level=0.8, preparation_notes=""
            ),
            PreparedResponse(
                id="r2", question_id="q2", primary_response="Response 2",
                response_strategy=ResponseStrategy.STRATEGIC_CONTEXT,
                key_messages=["Strategic alignment", "Growth trajectory"],
                potential_challenges=[], backup_responses=[],
                confidence_level=0.9, preparation_notes=""
            )
        ]
        
        talking_points = self.qa_engine._extract_key_talking_points(responses)
        
        assert isinstance(talking_points, list)
        assert len(talking_points) > 0
        assert len(talking_points) <= 10  # Should limit to 10
        # Should include messages from responses
        assert "Strong performance" in talking_points
        assert "Strategic alignment" in talking_points
    
    def test_error_handling_invalid_presentation(self):
        """Test error handling with invalid presentation"""
        with pytest.raises(Exception):
            self.qa_engine.create_comprehensive_qa_preparation(
                presentation=None,
                board=self.mock_board
            )
    
    def test_error_handling_invalid_board(self):
        """Test error handling with invalid board"""
        with pytest.raises(Exception):
            self.qa_engine.create_comprehensive_qa_preparation(
                presentation=self.mock_presentation,
                board=None
            )
    
    @patch('scrollintel.engines.qa_preparation_engine.logging')
    def test_logging_integration(self, mock_logging):
        """Test logging integration"""
        qa_preparation = self.qa_engine.create_comprehensive_qa_preparation(
            presentation=self.mock_presentation,
            board=self.mock_board
        )
        
        # Should log successful creation
        assert qa_preparation is not None


class TestIntegrationScenarios:
    """Integration test scenarios for Q&A preparation"""
    
    def setup_method(self):
        """Set up integration test fixtures"""
        self.qa_engine = QAPreparationEngine()
    
    def test_complete_board_meeting_qa_workflow(self):
        """Test complete Q&A preparation workflow for board meeting"""
        # Create comprehensive board scenario
        board = Board(
            id="public_company_board",
            name="Public Company Board of Directors",
            composition={
                "independent_directors": 5,
                "executive_directors": 2,
                "investor_representatives": 2,
                "industry_experts": 2
            },
            governance_structure={
                "committees": ["audit", "compensation", "nominating", "risk"],
                "governance_standards": "NYSE_listed",
                "board_size": 11
            },
            meeting_patterns={
                "frequency": "quarterly",
                "duration": 240,  # 4 hours
                "presentation_time": 60,  # 1 hour for presentations
                "qa_time": 45  # 45 minutes for Q&A
            },
            decision_processes={
                "voting_threshold": "majority",
                "consensus_building": True,
                "executive_sessions": True
            }
        )
        
        # Create comprehensive quarterly presentation
        designer = PresentationDesigner()
        quarterly_content = {
            "executive_summary": "Strong Q4 performance with record revenue and strategic milestones achieved",
            "financial_performance": {
                "q4_revenue": 32000000,
                "annual_revenue": 115000000,
                "revenue_growth": 28.5,
                "gross_margin": 71.2,
                "operating_margin": 19.8,
                "net_income": 18500000,
                "eps": 2.45,
                "cash_position": 52000000
            },
            "operational_excellence": {
                "customer_acquisition": 2850,
                "customer_retention": 96.2,
                "nps_score": 72,
                "employee_satisfaction": 89,
                "operational_efficiency": 92,
                "product_uptime": 99.98
            },
            "strategic_achievements": {
                "market_expansion": "Successfully entered 4 new markets",
                "product_innovation": "Launched 3 major product updates",
                "partnership_development": "Secured 12 strategic partnerships",
                "digital_transformation": "95% completion of digital initiatives"
            },
            "risk_management": {
                "cybersecurity_posture": "Enhanced with zero incidents",
                "regulatory_compliance": "100% compliance across all jurisdictions",
                "financial_risk_controls": "Strengthened with new frameworks",
                "operational_risk_mitigation": "Comprehensive business continuity plans"
            },
            "governance_updates": {
                "board_effectiveness": "Annual evaluation completed",
                "executive_compensation": "Aligned with performance metrics",
                "stakeholder_engagement": "Enhanced investor relations program",
                "esg_initiatives": "Sustainability goals on track"
            },
            "forward_outlook": {
                "2024_guidance": "Revenue target of $140M with 22% growth",
                "strategic_priorities": ["International expansion", "AI integration", "Sustainability"],
                "investment_focus": ["R&D scaling", "Talent acquisition", "Infrastructure"],
                "market_opportunities": ["Enterprise AI adoption", "Vertical expansion"]
            }
        }
        
        presentation = designer.create_board_presentation(
            content=quarterly_content,
            board=board
        )
        
        # Comprehensive board context
        board_context = {
            "meeting_type": "quarterly_board_meeting",
            "quarter": "Q4_2024",
            "focus_areas": [
                "financial_performance", "strategic_execution", "risk_management",
                "governance_oversight", "market_position", "operational_excellence"
            ],
            "special_topics": ["CEO succession planning", "ESG strategy", "Competitive response"],
            "regulatory_environment": "Increased scrutiny on AI governance",
            "market_conditions": "Economic uncertainty with growth opportunities",
            "stakeholder_priorities": ["Sustainable growth", "Risk management", "Innovation leadership"]
        }
        
        # Create comprehensive Q&A preparation
        qa_preparation = self.qa_engine.create_comprehensive_qa_preparation(
            presentation=presentation,
            board=board,
            board_context=board_context
        )
        
        # Validate comprehensive preparation
        assert len(qa_preparation.anticipated_questions) >= 10
        assert len(qa_preparation.anticipated_questions) <= 15
        
        # Should cover all major categories
        categories = set(q.category for q in qa_preparation.anticipated_questions)
        expected_categories = {
            QuestionCategory.FINANCIAL_PERFORMANCE,
            QuestionCategory.STRATEGIC_DIRECTION,
            QuestionCategory.RISK_MANAGEMENT,
            QuestionCategory.GOVERNANCE_COMPLIANCE
        }
        assert len(categories & expected_categories) >= 3
        
        # Should have high preparedness score
        assert qa_preparation.overall_preparedness_score >= 0.7
        
        # Should identify appropriate high-risk questions
        assert len(qa_preparation.high_risk_questions) >= 2
        assert len(qa_preparation.high_risk_questions) <= 5
        
        # Should have comprehensive talking points
        assert len(qa_preparation.key_talking_points) >= 5
        assert len(qa_preparation.key_talking_points) <= 10
        
        # Should have responses for all anticipated questions
        assert len(qa_preparation.prepared_responses) == len(qa_preparation.anticipated_questions)
        
        # Validate response quality
        high_confidence_responses = [r for r in qa_preparation.prepared_responses if r.confidence_level >= 0.7]
        assert len(high_confidence_responses) >= len(qa_preparation.prepared_responses) * 0.6  # At least 60% high confidence
        
        # Should have diverse response strategies
        strategies = set(r.response_strategy for r in qa_preparation.prepared_responses)
        assert len(strategies) >= 3  # Multiple strategies used
        
        # Test effectiveness tracking
        # Simulate actual board meeting questions
        actual_questions = [
            "What are the key drivers behind our exceptional Q4 revenue performance?",
            "How sustainable is our current growth trajectory given market conditions?",
            "What specific measures are we taking to address cybersecurity risks?",
            "How does our strategic direction position us against emerging competitors?",
            "What governance enhancements are we implementing for AI oversight?"
        ]
        
        board_feedback = {
            "satisfaction_score": 0.85,
            "response_clarity": 0.88,
            "preparation_thoroughness": 0.82,
            "improvement_suggestions": [
                "Provide more detailed competitive analysis",
                "Include specific risk mitigation timelines"
            ]
        }
        
        effectiveness_metrics = self.qa_engine.effectiveness_tracker.track_qa_effectiveness(
            qa_preparation=qa_preparation,
            actual_questions=actual_questions,
            board_feedback=board_feedback
        )
        
        # Validate effectiveness tracking
        assert effectiveness_metrics["preparation_completeness"] >= 0.8
        assert effectiveness_metrics["question_prediction_accuracy"] >= 0.4  # Some questions matched
        assert effectiveness_metrics["response_quality_score"] >= 0.7
        assert effectiveness_metrics["board_satisfaction_score"] == 0.85
        
        # Should have actionable improvement areas
        assert len(effectiveness_metrics["areas_for_improvement"]) >= 0
        
        # Should identify success indicators
        assert len(effectiveness_metrics["success_indicators"]) >= 2
    
    def test_crisis_communication_qa_preparation(self):
        """Test Q&A preparation for crisis communication scenario"""
        # Crisis scenario setup
        crisis_board = Board(
            id="crisis_response_board",
            name="Crisis Response Board",
            composition={"independent_directors": 4, "executive_directors": 2, "crisis_experts": 2},
            governance_structure={"emergency_protocols": True, "crisis_committee": True},
            meeting_patterns={"emergency_meeting": True, "duration": 90},
            decision_processes={"rapid_decision": True, "unanimous_consent": False}
        )
        
        # Crisis presentation content
        designer = PresentationDesigner()
        crisis_content = {
            "incident_summary": "Data security incident detected and contained within 4 hours",
            "immediate_response": {
                "incident_response_team": "Activated immediately",
                "affected_systems": "Isolated and secured",
                "customer_notification": "Completed within 24 hours",
                "regulatory_reporting": "Filed within required timeframes"
            },
            "impact_assessment": {
                "customers_affected": 8500,
                "data_types": ["Email addresses", "Account preferences"],
                "financial_impact": "$1.8M estimated response costs",
                "reputation_impact": "Managed through proactive communication"
            },
            "remediation_actions": {
                "security_enhancements": "Advanced monitoring deployed",
                "process_improvements": "Incident response procedures updated",
                "third_party_validation": "External security audit initiated",
                "customer_support": "Dedicated support team established"
            },
            "governance_response": {
                "board_notification": "Immediate notification provided",
                "regulatory_compliance": "Full compliance maintained",
                "stakeholder_communication": "Transparent communication strategy",
                "lessons_learned": "Comprehensive review initiated"
            }
        }
        
        crisis_presentation = designer.create_board_presentation(
            content=crisis_content,
            board=crisis_board
        )
        
        crisis_context = {
            "meeting_type": "emergency_board_meeting",
            "urgency_level": "critical",
            "focus_areas": ["incident_response", "risk_management", "stakeholder_communication"],
            "decision_required": True,
            "media_attention": "High",
            "regulatory_scrutiny": "Active"
        }
        
        # Create crisis Q&A preparation
        qa_preparation = self.qa_engine.create_comprehensive_qa_preparation(
            presentation=crisis_presentation,
            board=crisis_board,
            board_context=crisis_context
        )
        
        # Crisis preparation should be focused and actionable
        assert len(qa_preparation.anticipated_questions) <= 12  # Focused for urgent decision-making
        
        # Should heavily focus on risk and governance
        risk_governance_questions = [
            q for q in qa_preparation.anticipated_questions 
            if q.category in [QuestionCategory.RISK_MANAGEMENT, QuestionCategory.GOVERNANCE_COMPLIANCE]
        ]
        assert len(risk_governance_questions) >= len(qa_preparation.anticipated_questions) * 0.5
        
        # Should identify most questions as high-risk
        assert len(qa_preparation.high_risk_questions) >= 3
        
        # Should have high-confidence responses for crisis management
        crisis_responses = [
            r for r in qa_preparation.prepared_responses 
            if r.response_strategy == ResponseStrategy.ACKNOWLEDGE_AND_PLAN
        ]
        assert len(crisis_responses) >= 2
        
        # Should have preparedness score appropriate for crisis
        assert qa_preparation.overall_preparedness_score >= 0.6  # Reasonable given crisis urgency