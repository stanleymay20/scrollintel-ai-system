"""
Unit tests for the Intent Classifier in the Automated Code Generation System.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from scrollintel.models.nlp_models import Base, IntentType, ConfidenceLevel
from scrollintel.engines.intent_classifier import (
    IntentClassifier, IntentClassificationResult
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
    with patch('scrollintel.engines.intent_classifier.openai.OpenAI') as mock_client:
        mock_instance = Mock()
        mock_client.return_value = mock_instance
        
        # Mock the async chat completion
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = '''
        {
            "primary_intent": "create_application",
            "confidence": 0.9,
            "secondary_intents": [
                {"intent": "add_functionality", "confidence": 0.3}
            ],
            "reasoning": "User wants to create a new application from scratch"
        }
        '''
        
        mock_instance.chat.completions.acreate = AsyncMock(return_value=mock_response)
        yield mock_instance


@pytest.fixture
def intent_classifier(db_session, mock_openai_client):
    """Create IntentClassifier instance with mocked dependencies."""
    with patch('scrollintel.engines.intent_classifier.get_settings') as mock_settings:
        mock_settings.return_value.openai_api_key = "test-key"
        classifier = IntentClassifier(db_session)
        classifier.client = mock_openai_client
        return classifier


class TestIntentClassificationResult:
    """Test cases for IntentClassificationResult."""
    
    def test_creation_with_high_confidence(self):
        """Test creating result with high confidence."""
        result = IntentClassificationResult(
            primary_intent=IntentType.CREATE_APPLICATION,
            confidence=0.9,
            reasoning="Clear intent to create application"
        )
        
        assert result.primary_intent == IntentType.CREATE_APPLICATION
        assert result.confidence == 0.9
        assert result.confidence_level == ConfidenceLevel.HIGH
        assert result.reasoning == "Clear intent to create application"
        assert len(result.secondary_intents) == 0

    def test_creation_with_secondary_intents(self):
        """Test creating result with secondary intents."""
        secondary_intents = [
            (IntentType.ADD_FUNCTIONALITY, 0.4),
            (IntentType.INTEGRATE_SYSTEM, 0.3)
        ]
        
        result = IntentClassificationResult(
            primary_intent=IntentType.CREATE_APPLICATION,
            confidence=0.8,
            secondary_intents=secondary_intents,
            reasoning="Primary intent with alternatives"
        )
        
        assert len(result.secondary_intents) == 2
        assert result.secondary_intents[0][0] == IntentType.ADD_FUNCTIONALITY
        assert result.secondary_intents[0][1] == 0.4

    def test_confidence_level_mapping(self):
        """Test confidence level mapping."""
        high_result = IntentClassificationResult(IntentType.CREATE_APPLICATION, 0.85)
        assert high_result.confidence_level == ConfidenceLevel.HIGH
        
        medium_result = IntentClassificationResult(IntentType.CREATE_APPLICATION, 0.7)
        assert medium_result.confidence_level == ConfidenceLevel.MEDIUM
        
        low_result = IntentClassificationResult(IntentType.CREATE_APPLICATION, 0.4)
        assert low_result.confidence_level == ConfidenceLevel.LOW


class TestIntentClassifier:
    """Test cases for IntentClassifier."""
    
    @pytest.mark.asyncio
    async def test_classify_intent_create_application(self, intent_classifier):
        """Test classifying CREATE_APPLICATION intent."""
        text = "I want to create a new web application for managing users"
        
        result = await intent_classifier.classify_intent(text)
        
        assert result.primary_intent == IntentType.CREATE_APPLICATION
        assert result.confidence > 0.0
        assert result.confidence_level in [ConfidenceLevel.HIGH, ConfidenceLevel.MEDIUM, ConfidenceLevel.LOW]

    @pytest.mark.asyncio
    async def test_classify_intent_modify_feature(self, intent_classifier):
        """Test classifying MODIFY_FEATURE intent."""
        text = "I need to modify the existing user registration feature"
        
        result = await intent_classifier.classify_intent(text)
        
        # Should detect modify intent through rule-based classification
        assert result.primary_intent == IntentType.MODIFY_FEATURE or \
               any(intent == IntentType.MODIFY_FEATURE for intent, _ in result.secondary_intents)

    @pytest.mark.asyncio
    async def test_classify_intent_add_functionality(self, intent_classifier):
        """Test classifying ADD_FUNCTIONALITY intent."""
        text = "I want to add new search functionality to the existing system"
        
        result = await intent_classifier.classify_intent(text)
        
        # Should detect add functionality intent
        assert result.primary_intent == IntentType.ADD_FUNCTIONALITY or \
               any(intent == IntentType.ADD_FUNCTIONALITY for intent, _ in result.secondary_intents)

    @pytest.mark.asyncio
    async def test_classify_intent_integrate_system(self, intent_classifier):
        """Test classifying INTEGRATE_SYSTEM intent."""
        text = "I need to integrate with external payment API"
        
        result = await intent_classifier.classify_intent(text)
        
        # Should detect integration intent
        assert result.primary_intent == IntentType.INTEGRATE_SYSTEM or \
               any(intent == IntentType.INTEGRATE_SYSTEM for intent, _ in result.secondary_intents)

    @pytest.mark.asyncio
    async def test_classify_intent_optimize_performance(self, intent_classifier):
        """Test classifying OPTIMIZE_PERFORMANCE intent."""
        text = "The system is too slow, I need to optimize performance"
        
        result = await intent_classifier.classify_intent(text)
        
        # Should detect performance optimization intent
        assert result.primary_intent == IntentType.OPTIMIZE_PERFORMANCE or \
               any(intent == IntentType.OPTIMIZE_PERFORMANCE for intent, _ in result.secondary_intents)

    @pytest.mark.asyncio
    async def test_classify_intent_enhance_security(self, intent_classifier):
        """Test classifying ENHANCE_SECURITY intent."""
        text = "I need to add authentication and authorization features"
        
        result = await intent_classifier.classify_intent(text)
        
        # Should detect security enhancement intent
        assert result.primary_intent == IntentType.ENHANCE_SECURITY or \
               any(intent == IntentType.ENHANCE_SECURITY for intent, _ in result.secondary_intents)

    @pytest.mark.asyncio
    async def test_classify_intent_improve_ui(self, intent_classifier):
        """Test classifying IMPROVE_UI intent."""
        text = "The user interface needs to be more user-friendly and responsive"
        
        result = await intent_classifier.classify_intent(text)
        
        # Should detect UI improvement intent
        assert result.primary_intent == IntentType.IMPROVE_UI or \
               any(intent == IntentType.IMPROVE_UI for intent, _ in result.secondary_intents)

    @pytest.mark.asyncio
    async def test_classify_intent_manage_data(self, intent_classifier):
        """Test classifying MANAGE_DATA intent."""
        text = "I need to design the database schema for storing user data"
        
        result = await intent_classifier.classify_intent(text)
        
        # Should detect data management intent
        assert result.primary_intent == IntentType.MANAGE_DATA or \
               any(intent == IntentType.MANAGE_DATA for intent, _ in result.secondary_intents)

    @pytest.mark.asyncio
    async def test_classify_intent_deploy_application(self, intent_classifier):
        """Test classifying DEPLOY_APPLICATION intent."""
        text = "I need to deploy this application to the cloud"
        
        result = await intent_classifier.classify_intent(text)
        
        # Should detect deployment intent
        assert result.primary_intent == IntentType.DEPLOY_APPLICATION or \
               any(intent == IntentType.DEPLOY_APPLICATION for intent, _ in result.secondary_intents)

    @pytest.mark.asyncio
    async def test_classify_intent_clarify_requirement(self, intent_classifier):
        """Test classifying CLARIFY_REQUIREMENT intent."""
        text = "I don't understand what you mean by user roles"
        
        result = await intent_classifier.classify_intent(text)
        
        # Should detect clarification intent
        assert result.primary_intent == IntentType.CLARIFY_REQUIREMENT or \
               any(intent == IntentType.CLARIFY_REQUIREMENT for intent, _ in result.secondary_intents)

    def test_classify_with_rules_create_application(self, intent_classifier):
        """Test rule-based classification for CREATE_APPLICATION."""
        text = "I want to build a new application from scratch"
        
        result = intent_classifier._classify_with_rules(text)
        
        assert result.primary_intent == IntentType.CREATE_APPLICATION
        assert result.confidence > 0.0

    def test_classify_with_rules_no_matches(self, intent_classifier):
        """Test rule-based classification with no pattern matches."""
        text = "This is just random text with no clear intent"
        
        result = intent_classifier._classify_with_rules(text)
        
        # Should default to CREATE_APPLICATION with low confidence
        assert result.primary_intent == IntentType.CREATE_APPLICATION
        assert result.confidence == 0.0  # No matches means 0.0 confidence

    def test_classify_with_rules_multiple_matches(self, intent_classifier):
        """Test rule-based classification with multiple pattern matches."""
        text = "I want to create a new application and also modify existing features"
        
        result = intent_classifier._classify_with_rules(text)
        
        # Should have primary intent and may have secondary intents
        assert result.primary_intent in [IntentType.CREATE_APPLICATION, IntentType.MODIFY_FEATURE]
        # Secondary intents depend on the scoring threshold, so we don't assert on count

    @pytest.mark.asyncio
    async def test_classify_with_ai_success(self, intent_classifier):
        """Test AI-powered classification success."""
        text = "I need a user management system"
        
        result = await intent_classifier._classify_with_ai(text)
        
        assert result.primary_intent == IntentType.CREATE_APPLICATION
        assert result.confidence == 0.9
        assert "User wants to create" in result.reasoning

    @pytest.mark.asyncio
    async def test_classify_with_ai_error(self, intent_classifier):
        """Test AI-powered classification error handling."""
        # Mock OpenAI to raise an exception
        intent_classifier.client.chat.completions.acreate = AsyncMock(
            side_effect=Exception("API Error")
        )
        
        result = await intent_classifier._classify_with_ai("test text")
        
        assert result.primary_intent == IntentType.CREATE_APPLICATION
        assert result.confidence == 0.1
        assert "AI classification failed" in result.reasoning

    def test_parse_ai_response_valid_json(self, intent_classifier):
        """Test parsing valid JSON AI response."""
        content = '''
        {
            "primary_intent": "ADD_FUNCTIONALITY",
            "confidence": 0.8,
            "secondary_intents": [{"intent": "MODIFY_FEATURE", "confidence": 0.4}],
            "reasoning": "User wants to add features"
        }
        '''
        
        result = intent_classifier._parse_ai_response(content)
        
        assert result["primary_intent"] == "ADD_FUNCTIONALITY"
        assert result["confidence"] == 0.8
        assert len(result["secondary_intents"]) == 1

    def test_parse_ai_response_invalid_json(self, intent_classifier):
        """Test parsing invalid JSON AI response."""
        content = "This is not JSON but mentions CREATE_APPLICATION and confidence: 0.7"
        
        result = intent_classifier._parse_ai_response(content)
        
        # Should fall back to manual parsing
        assert result["primary_intent"] == "create_application"
        assert result["confidence"] == 0.7

    def test_manual_parse_response(self, intent_classifier):
        """Test manual parsing of AI response."""
        content = "The intent is MODIFY_FEATURE with confidence: 0.85"
        
        result = intent_classifier._manual_parse_response(content)
        
        assert result["primary_intent"] == "modify_feature"
        assert result["confidence"] == 0.85

    def test_combine_results_agreement(self, intent_classifier):
        """Test combining results when both methods agree."""
        rule_result = IntentClassificationResult(
            primary_intent=IntentType.CREATE_APPLICATION,
            confidence=0.7,
            reasoning="Rule-based classification"
        )
        
        ai_result = IntentClassificationResult(
            primary_intent=IntentType.CREATE_APPLICATION,
            confidence=0.8,
            reasoning="AI classification"
        )
        
        combined = intent_classifier._combine_results(rule_result, ai_result)
        
        assert combined.primary_intent == IntentType.CREATE_APPLICATION
        assert combined.confidence > 0.7  # Should be boosted due to agreement
        assert "agreement" in combined.reasoning

    def test_combine_results_disagreement(self, intent_classifier):
        """Test combining results when methods disagree."""
        rule_result = IntentClassificationResult(
            primary_intent=IntentType.CREATE_APPLICATION,
            confidence=0.6,
            reasoning="Rule-based classification"
        )
        
        ai_result = IntentClassificationResult(
            primary_intent=IntentType.MODIFY_FEATURE,
            confidence=0.8,
            reasoning="AI classification"
        )
        
        combined = intent_classifier._combine_results(rule_result, ai_result)
        
        # Should choose AI result (higher confidence)
        assert combined.primary_intent == IntentType.MODIFY_FEATURE
        assert combined.confidence < 0.8  # Should be penalized for disagreement
        assert "disagreement" in combined.reasoning
        
        # Rule result should be in secondary intents
        assert any(intent == IntentType.CREATE_APPLICATION for intent, _ in combined.secondary_intents)

    @pytest.mark.asyncio
    async def test_classify_multiple_intents(self, intent_classifier):
        """Test classifying multiple texts."""
        texts = [
            "I want to create a new application",
            "I need to modify existing features",
            "Add search functionality"
        ]
        
        results = await intent_classifier.classify_multiple_intents(texts)
        
        assert len(results) == 3
        assert all(isinstance(result, IntentClassificationResult) for result in results)

    def test_get_intent_statistics(self, intent_classifier):
        """Test getting statistics from classification results."""
        results = [
            IntentClassificationResult(IntentType.CREATE_APPLICATION, 0.9),
            IntentClassificationResult(IntentType.CREATE_APPLICATION, 0.8),
            IntentClassificationResult(IntentType.MODIFY_FEATURE, 0.7),
            IntentClassificationResult(IntentType.ADD_FUNCTIONALITY, 0.5)
        ]
        
        stats = intent_classifier.get_intent_statistics(results)
        
        assert stats["total_classifications"] == 4
        assert stats["intent_distribution"]["create_application"] == 2
        assert stats["intent_distribution"]["modify_feature"] == 1
        assert stats["intent_distribution"]["add_functionality"] == 1
        assert stats["average_confidence"] == 0.725
        assert stats["high_confidence_count"] == 2  # >= 0.8
        assert stats["low_confidence_count"] == 1   # < 0.6

    def test_get_intent_statistics_empty(self, intent_classifier):
        """Test getting statistics from empty results."""
        results = []
        
        stats = intent_classifier.get_intent_statistics(results)
        
        assert stats == {}

    @pytest.mark.asyncio
    async def test_classify_intent_high_rule_confidence(self, intent_classifier):
        """Test that high rule-based confidence is used directly."""
        # Mock rule-based classification to return high confidence
        with patch.object(intent_classifier, '_classify_with_rules') as mock_rules:
            mock_rules.return_value = IntentClassificationResult(
                primary_intent=IntentType.CREATE_APPLICATION,
                confidence=0.9,
                reasoning="High confidence rule match"
            )
            
            result = await intent_classifier.classify_intent("test text")
            
            # Should use rule-based result directly without calling AI
            assert result.primary_intent == IntentType.CREATE_APPLICATION
            assert result.confidence == 0.9

    @pytest.mark.asyncio
    async def test_classify_intent_error_handling(self, intent_classifier):
        """Test error handling in intent classification."""
        # Mock both methods to raise exceptions
        with patch.object(intent_classifier, '_classify_with_rules', side_effect=Exception("Rule error")):
            with patch.object(intent_classifier, '_classify_with_ai', side_effect=Exception("AI error")):
                
                result = await intent_classifier.classify_intent("test text")
                
                # Should return default result
                assert result.primary_intent == IntentType.CREATE_APPLICATION
                assert result.confidence == 0.1
                assert "Classification failed" in result.reasoning


class TestIntentPatterns:
    """Test cases for intent pattern matching."""
    
    def test_create_application_patterns(self, intent_classifier):
        """Test CREATE_APPLICATION pattern matching."""
        test_cases = [
            "I want to create a new application",
            "I need to build an app from scratch",
            "I want to develop a system",
            "I need to make a platform"
        ]
        
        for text in test_cases:
            result = intent_classifier._classify_with_rules(text)
            assert result.primary_intent == IntentType.CREATE_APPLICATION or \
                   any(intent == IntentType.CREATE_APPLICATION for intent, _ in result.secondary_intents)

    def test_modify_feature_patterns(self, intent_classifier):
        """Test MODIFY_FEATURE pattern matching."""
        test_cases = [
            "I need to modify the existing feature",
            "I want to change how the system works",
            "I need to update the functionality",
            "I want to alter the current behavior"
        ]
        
        for text in test_cases:
            result = intent_classifier._classify_with_rules(text)
            assert result.primary_intent == IntentType.MODIFY_FEATURE or \
                   any(intent == IntentType.MODIFY_FEATURE for intent, _ in result.secondary_intents)

    def test_integration_patterns(self, intent_classifier):
        """Test INTEGRATE_SYSTEM pattern matching."""
        test_cases = [
            "I need to integrate with external API",
            "I want to connect to third-party service",
            "I need API integration",
            "I want to sync with external system"
        ]
        
        for text in test_cases:
            result = intent_classifier._classify_with_rules(text)
            assert result.primary_intent == IntentType.INTEGRATE_SYSTEM or \
                   any(intent == IntentType.INTEGRATE_SYSTEM for intent, _ in result.secondary_intents)

    def test_performance_patterns(self, intent_classifier):
        """Test OPTIMIZE_PERFORMANCE pattern matching."""
        test_cases = [
            "I need to optimize performance",
            "The system is too slow",
            "I want to improve efficiency",
            "I need to speed up the application"
        ]
        
        for text in test_cases:
            result = intent_classifier._classify_with_rules(text)
            assert result.primary_intent == IntentType.OPTIMIZE_PERFORMANCE or \
                   any(intent == IntentType.OPTIMIZE_PERFORMANCE for intent, _ in result.secondary_intents)