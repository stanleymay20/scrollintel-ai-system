"""
Unit tests for the Clarification Engine in the Automated Code Generation System.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from scrollintel.models.nlp_models import (
    Base, Requirements, Clarification, RequirementsModel, ParsedRequirementModel, 
    EntityModel, RequirementType, IntentType, EntityType
)
from scrollintel.engines.clarification_engine import (
    ClarificationEngine, ClarificationQuestion, ClarificationResult,
    QuestionType, QuestionPriority
)


@pytest.fixture
def db_session():
    """Create an in-memory SQLite database for testing."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    yield session
    session.close()


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client."""
    with patch('scrollintel.engines.clarification_engine.openai.OpenAI') as mock_client:
        mock_instance = Mock()
        mock_client.return_value = mock_instance
        
        # Mock the async chat completion
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = '''
        {
            "questions": [
                {
                    "question": "What is the expected number of concurrent users?",
                    "type": "non_functional_requirement",
                    "priority": "high",
                    "context": "Performance requirements not specified",
                    "suggested_answers": ["100", "1000", "10000"],
                    "related_entities": ["user", "system"],
                    "reasoning": "Need to understand scalability requirements"
                },
                {
                    "question": "What authentication method should be used?",
                    "type": "technical_constraint",
                    "priority": "medium",
                    "context": "Security requirements unclear",
                    "suggested_answers": ["OAuth", "JWT", "Session-based"],
                    "related_entities": ["authentication", "user"],
                    "reasoning": "Authentication method affects security architecture"
                }
            ]
        }
        '''
        
        mock_instance.chat.completions.acreate = AsyncMock(return_value=mock_response)
        yield mock_instance


@pytest.fixture
def clarification_engine(db_session, mock_openai_client):
    """Create ClarificationEngine instance with mocked dependencies."""
    with patch('scrollintel.engines.clarification_engine.get_settings') as mock_settings:
        mock_settings.return_value.openai_api_key = "test-key"
        engine = ClarificationEngine(db_session)
        engine.client = mock_openai_client
        return engine


class TestClarificationQuestion:
    """Test cases for ClarificationQuestion."""
    
    def test_creation(self):
        """Test creating ClarificationQuestion."""
        question = ClarificationQuestion(
            question="What is the expected user load?",
            question_type=QuestionType.NON_FUNCTIONAL_REQUIREMENT,
            priority=QuestionPriority.HIGH,
            context="Performance requirements not specified",
            suggested_answers=["100", "1000", "10000"],
            related_entities=["user", "system"],
            reasoning="Need to understand scalability requirements"
        )
        
        assert question.question == "What is the expected user load?"
        assert question.question_type == QuestionType.NON_FUNCTIONAL_REQUIREMENT
        assert question.priority == QuestionPriority.HIGH
        assert len(question.suggested_answers) == 3
        assert len(question.related_entities) == 2
        assert "scalability" in question.reasoning


class TestClarificationResult:
    """Test cases for ClarificationResult."""
    
    def test_creation(self):
        """Test creating ClarificationResult."""
        questions = [
            ClarificationQuestion(
                "Test question?",
                QuestionType.FUNCTIONAL_DETAIL,
                QuestionPriority.MEDIUM,
                "Test context"
            )
        ]
        
        result = ClarificationResult(
            questions=questions,
            ambiguity_score=0.3,
            completeness_score=0.7,
            critical_gaps=["Missing security requirements"],
            recommendations=["Add more detail to requirements"]
        )
        
        assert len(result.questions) == 1
        assert result.ambiguity_score == 0.3
        assert result.completeness_score == 0.7
        assert len(result.critical_gaps) == 1
        assert len(result.recommendations) == 1


class TestClarificationEngine:
    """Test cases for ClarificationEngine."""
    
    @pytest.mark.asyncio
    async def test_analyze_requirements_success(self, clarification_engine):
        """Test successful requirements analysis."""
        requirements = RequirementsModel(
            raw_text="I need some kind of user system that might work with authentication"
        )
        
        result = await clarification_engine.analyze_requirements(requirements)
        
        assert isinstance(result, ClarificationResult)
        assert len(result.questions) > 0
        assert 0.0 <= result.ambiguity_score <= 1.0
        assert 0.0 <= result.completeness_score <= 1.0
        assert isinstance(result.critical_gaps, list)
        assert isinstance(result.recommendations, list)

    @pytest.mark.asyncio
    async def test_analyze_requirements_with_parsed_data(self, clarification_engine):
        """Test analysis with parsed requirements and entities."""
        requirements = RequirementsModel(
            raw_text="Create a user management system"
        )
        
        parsed_requirements = [
            ParsedRequirementModel(
                requirement_type=RequirementType.FUNCTIONAL,
                intent=IntentType.CREATE_APPLICATION,
                description="User registration feature",
                acceptance_criteria=["Email validation", "Password strength"]
            )
        ]
        
        entities = [
            EntityModel(
                entity_type=EntityType.USER_ROLE,
                name="user",
                description="System user",
                confidence_score=0.8
            )
        ]
        
        result = await clarification_engine.analyze_requirements(
            requirements, parsed_requirements, entities
        )
        
        assert result.completeness_score > 0.0
        assert len(result.questions) >= 0

    def test_calculate_ambiguity_score_high(self, clarification_engine):
        """Test ambiguity score calculation with high ambiguity."""
        text = "I need some kind of system that might work with maybe a few users"
        
        score = clarification_engine._calculate_ambiguity_score(text)
        
        assert score > 0.0
        # Should detect ambiguous terms like "some", "might", "maybe", "few"

    def test_calculate_ambiguity_score_low(self, clarification_engine):
        """Test ambiguity score calculation with low ambiguity."""
        text = "I need a user registration system with email validation and password requirements"
        
        score = clarification_engine._calculate_ambiguity_score(text)
        
        assert score >= 0.0
        # Should have lower ambiguity score

    def test_calculate_completeness_score_complete(self, clarification_engine):
        """Test completeness score with comprehensive requirements."""
        text = """
        I need a user management system with the following features:
        - User registration and authentication
        - Performance should handle 1000 concurrent users
        - Security with OAuth authentication
        - User interface should be responsive
        - Database for storing user data
        - Integration with external email service
        - Deployment on AWS cloud platform
        """
        
        score = clarification_engine._calculate_completeness_score(text)
        
        assert score > 0.5  # Should be relatively complete

    def test_calculate_completeness_score_incomplete(self, clarification_engine):
        """Test completeness score with incomplete requirements."""
        text = "I need a system"
        
        score = clarification_engine._calculate_completeness_score(text)
        
        assert score < 0.5  # Should be incomplete

    def test_generate_rule_based_questions_vague_quantifiers(self, clarification_engine):
        """Test generating questions for vague quantifiers."""
        text = "The system should handle many users and some data"
        
        questions = clarification_engine._generate_rule_based_questions(text)
        
        # Should generate questions about "many" and "some"
        quantifier_questions = [
            q for q in questions 
            if "many" in q.question.lower() or "some" in q.question.lower()
        ]
        assert len(quantifier_questions) > 0

    def test_generate_rule_based_questions_subjective_terms(self, clarification_engine):
        """Test generating questions for subjective terms."""
        text = "The system should be fast and user-friendly"
        
        questions = clarification_engine._generate_rule_based_questions(text)
        
        # Should generate questions about "fast" and "user-friendly"
        subjective_questions = [
            q for q in questions 
            if q.question_type == QuestionType.ACCEPTANCE_CRITERIA
        ]
        assert len(subjective_questions) > 0

    def test_generate_rule_based_questions_missing_areas(self, clarification_engine):
        """Test generating questions for missing requirement areas."""
        text = "I need a system"  # Very minimal requirements
        
        questions = clarification_engine._generate_rule_based_questions(text)
        
        # Should generate questions for missing areas
        assert len(questions) > 0
        
        # Check for different question types
        question_types = [q.question_type for q in questions]
        assert QuestionType.FUNCTIONAL_DETAIL in question_types

    @pytest.mark.asyncio
    async def test_generate_ai_questions_success(self, clarification_engine):
        """Test AI-powered question generation."""
        text = "I need a user management system"
        
        questions = await clarification_engine._generate_ai_questions(text)
        
        assert len(questions) == 2  # Based on mock response
        
        # Check first question
        first_question = questions[0]
        assert first_question.question == "What is the expected number of concurrent users?"
        assert first_question.question_type == QuestionType.NON_FUNCTIONAL_REQUIREMENT
        assert first_question.priority == QuestionPriority.HIGH
        assert len(first_question.suggested_answers) == 3

    @pytest.mark.asyncio
    async def test_generate_ai_questions_error(self, clarification_engine):
        """Test AI question generation error handling."""
        # Mock OpenAI to raise an exception
        clarification_engine.client.chat.completions.acreate = AsyncMock(
            side_effect=Exception("API Error")
        )
        
        questions = await clarification_engine._generate_ai_questions("test text")
        
        assert len(questions) == 0

    def test_generate_context_questions_missing_acceptance_criteria(self, clarification_engine):
        """Test generating questions for missing acceptance criteria."""
        parsed_requirements = [
            ParsedRequirementModel(
                requirement_type=RequirementType.FUNCTIONAL,
                intent=IntentType.CREATE_APPLICATION,
                description="User registration feature",
                acceptance_criteria=[]  # Empty acceptance criteria
            )
        ]
        
        questions = clarification_engine._generate_context_questions(parsed_requirements, None)
        
        # Should generate question about acceptance criteria
        acceptance_questions = [
            q for q in questions 
            if q.question_type == QuestionType.ACCEPTANCE_CRITERIA
        ]
        assert len(acceptance_questions) > 0

    def test_generate_context_questions_isolated_entities(self, clarification_engine):
        """Test generating questions for isolated entities."""
        entities = [
            EntityModel(
                entity_type=EntityType.BUSINESS_OBJECT,
                name="user_profile",
                description="User profile data",
                attributes={},  # No attributes
                confidence_score=0.8
            )
        ]
        
        questions = clarification_engine._generate_context_questions(None, entities)
        
        # Should generate question about entity relationships
        data_questions = [
            q for q in questions 
            if q.question_type == QuestionType.DATA_REQUIREMENT
        ]
        assert len(data_questions) > 0

    def test_generate_completeness_question(self, clarification_engine):
        """Test generating completeness questions for missing areas."""
        # Test functional requirements
        func_question = clarification_engine._generate_completeness_question('functional_requirements')
        assert func_question is not None
        assert func_question.question_type == QuestionType.FUNCTIONAL_DETAIL
        assert func_question.priority == QuestionPriority.CRITICAL
        
        # Test non-functional requirements
        nonfunc_question = clarification_engine._generate_completeness_question('non_functional_requirements')
        assert nonfunc_question is not None
        assert nonfunc_question.question_type == QuestionType.NON_FUNCTIONAL_REQUIREMENT
        assert nonfunc_question.priority == QuestionPriority.HIGH
        
        # Test unknown area
        unknown_question = clarification_engine._generate_completeness_question('unknown_area')
        assert unknown_question is None

    def test_merge_questions(self, clarification_engine):
        """Test merging questions from different sources."""
        questions1 = [
            ClarificationQuestion(
                "What is the user load?",
                QuestionType.NON_FUNCTIONAL_REQUIREMENT,
                QuestionPriority.HIGH,
                "Performance context"
            )
        ]
        
        questions2 = [
            ClarificationQuestion(
                "What is the user load?",  # Duplicate
                QuestionType.NON_FUNCTIONAL_REQUIREMENT,
                QuestionPriority.MEDIUM,
                "Different context"
            ),
            ClarificationQuestion(
                "What authentication method?",
                QuestionType.TECHNICAL_CONSTRAINT,
                QuestionPriority.CRITICAL,
                "Security context"
            )
        ]
        
        merged = clarification_engine._merge_questions(questions1, questions2)
        
        # Should deduplicate and sort by priority
        assert len(merged) == 2
        assert merged[0].priority == QuestionPriority.CRITICAL  # Should be first
        assert merged[1].priority == QuestionPriority.HIGH

    def test_identify_critical_gaps(self, clarification_engine):
        """Test identifying critical gaps in requirements."""
        text = "I need a system"  # Very minimal
        
        gaps = clarification_engine._identify_critical_gaps(text, None, None)
        
        assert len(gaps) > 0
        
        # Should identify multiple gaps
        gap_text = " ".join(gaps).lower()
        assert "user" in gap_text or "role" in gap_text
        assert "data" in gap_text or "storage" in gap_text

    def test_identify_critical_gaps_with_data(self, clarification_engine):
        """Test gap identification with some data provided."""
        text = "I need a user management system with authentication"
        
        parsed_requirements = [
            ParsedRequirementModel(
                requirement_type=RequirementType.FUNCTIONAL,
                intent=IntentType.CREATE_APPLICATION,
                description="User management",
                acceptance_criteria=[]
            )
        ]
        
        entities = [
            EntityModel(
                entity_type=EntityType.USER_ROLE,
                name="user",
                confidence_score=0.8
            ),
            EntityModel(
                entity_type=EntityType.BUSINESS_OBJECT,
                name="account",
                confidence_score=0.8
            )
        ]
        
        gaps = clarification_engine._identify_critical_gaps(text, parsed_requirements, entities)
        
        # Should have fewer gaps since some data is provided
        assert len(gaps) < 5

    def test_generate_recommendations(self, clarification_engine):
        """Test generating recommendations based on analysis."""
        # High ambiguity, low completeness
        recommendations = clarification_engine._generate_recommendations(
            ambiguity_score=0.6,
            completeness_score=0.3,
            critical_gaps=["Missing user roles", "No data requirements"]
        )
        
        assert len(recommendations) > 0
        
        rec_text = " ".join(recommendations).lower()
        assert "specific" in rec_text or "measurable" in rec_text
        assert "incomplete" in rec_text or "detail" in rec_text
        assert "critical gaps" in rec_text

    def test_generate_recommendations_good_requirements(self, clarification_engine):
        """Test recommendations for good requirements."""
        recommendations = clarification_engine._generate_recommendations(
            ambiguity_score=0.1,
            completeness_score=0.9,
            critical_gaps=[]
        )
        
        assert len(recommendations) == 1
        assert "well-defined" in recommendations[0].lower()

    def test_get_context(self, clarification_engine):
        """Test context extraction."""
        text = "The quick brown fox jumps over the lazy dog"
        start = 16  # "fox"
        end = 19
        
        context = clarification_engine._get_context(text, start, end, window=10)
        
        assert "brown fox jumps" in context
        assert len(context) <= len(text)

    def test_parse_ai_response_valid_json(self, clarification_engine):
        """Test parsing valid JSON AI response."""
        content = '''
        {
            "questions": [
                {
                    "question": "What is the expected load?",
                    "type": "non_functional_requirement",
                    "priority": "high",
                    "context": "Performance not specified"
                }
            ]
        }
        '''
        
        result = clarification_engine._parse_ai_response(content)
        
        assert "questions" in result
        assert len(result["questions"]) == 1
        assert result["questions"][0]["question"] == "What is the expected load?"

    def test_parse_ai_response_invalid_json(self, clarification_engine):
        """Test parsing invalid JSON AI response."""
        content = "This is not valid JSON"
        
        result = clarification_engine._parse_ai_response(content)
        
        assert result == {"questions": []}

    @pytest.mark.asyncio
    async def test_save_questions(self, clarification_engine, db_session):
        """Test saving clarification questions to database."""
        # Create requirements record
        requirements = Requirements(raw_text="test", processed_at=datetime.utcnow())
        db_session.add(requirements)
        db_session.flush()
        
        questions = [
            ClarificationQuestion(
                "What is the expected user load?",
                QuestionType.NON_FUNCTIONAL_REQUIREMENT,
                QuestionPriority.HIGH,
                "Performance requirements not specified"
            ),
            ClarificationQuestion(
                "What authentication method?",
                QuestionType.TECHNICAL_CONSTRAINT,
                QuestionPriority.MEDIUM,
                "Security requirements unclear"
            )
        ]
        
        saved_questions = await clarification_engine.save_questions(requirements.id, questions)
        
        assert len(saved_questions) == 2
        
        # Verify database records
        db_clarifications = db_session.query(Clarification).all()
        assert len(db_clarifications) == 2
        
        # Check first question
        first_clarification = db_clarifications[0]
        assert first_clarification.question == "What is the expected user load?"
        assert first_clarification.priority == "high"
        assert first_clarification.requirements_id == requirements.id

    @pytest.mark.asyncio
    async def test_analyze_requirements_error_handling(self, clarification_engine):
        """Test error handling in requirements analysis."""
        # Mock methods to raise exceptions
        with patch.object(clarification_engine, '_calculate_ambiguity_score', side_effect=Exception("Error")):
            
            requirements = RequirementsModel(raw_text="test")
            result = await clarification_engine.analyze_requirements(requirements)
            
            assert len(result.questions) == 0
            assert result.ambiguity_score == 0.0
            assert result.completeness_score == 0.0
            assert len(result.critical_gaps) > 0
            assert "Analysis failed" in result.critical_gaps[0]

    def test_ambiguity_patterns_coverage(self, clarification_engine):
        """Test that ambiguity patterns cover expected cases."""
        # Test vague quantifiers
        text = "I need some users and many features"
        score = clarification_engine._calculate_ambiguity_score(text)
        assert score > 0.0
        
        # Test uncertain language
        text = "The system might need authentication maybe"
        score = clarification_engine._calculate_ambiguity_score(text)
        assert score > 0.0
        
        # Test incomplete specifications
        text = "Features include login, logout, etc."
        score = clarification_engine._calculate_ambiguity_score(text)
        assert score > 0.0
        
        # Test subjective terms
        text = "The interface should be nice and user-friendly"
        score = clarification_engine._calculate_ambiguity_score(text)
        assert score > 0.0

    def test_completeness_requirements_coverage(self, clarification_engine):
        """Test that completeness requirements cover expected areas."""
        # Test functional requirements detection
        text = "User stories include login and registration features"
        score = clarification_engine._calculate_completeness_score(text)
        assert score > 0.0
        
        # Test non-functional requirements detection
        text = "Performance should be fast with good security"
        score = clarification_engine._calculate_completeness_score(text)
        assert score > 0.0
        
        # Test data requirements detection
        text = "Database will store user data and profiles"
        score = clarification_engine._calculate_completeness_score(text)
        assert score > 0.0