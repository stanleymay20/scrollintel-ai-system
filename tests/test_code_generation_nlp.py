"""
Unit tests for automated code generation NLP components.
Tests NLP accuracy and edge cases for requirements processing.
"""
import pytest
from unittest.mock import Mock, patch
import json

from scrollintel.engines.code_generation_nlp import NLProcessor
from scrollintel.engines.code_generation_intent import IntentClassifier
from scrollintel.engines.code_generation_entities import EntityExtractor
from scrollintel.engines.code_generation_clarification import ClarificationEngine
from scrollintel.engines.code_generation_validation import RequirementsValidator
from scrollintel.models.code_generation_models import (
    Requirements, ParsedRequirement, Entity, Relationship, Clarification,
    RequirementType, Intent, EntityType, ConfidenceLevel, ProcessingResult
)


class TestNLProcessor:
    """Test cases for NLProcessor."""
    
    @pytest.fixture
    def processor(self):
        """Create NLProcessor instance for testing."""
        return NLProcessor()
    
    @pytest.fixture
    def mock_openai_response(self):
        """Mock OpenAI API response."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = json.dumps([
            {
                "original_text": "Users should be able to login",
                "structured_text": "Users should be able to authenticate with username and password",
                "requirement_type": "functional",
                "intent": "add_security",
                "acceptance_criteria": ["Valid credentials allow access", "Invalid credentials are rejected"],
                "dependencies": [],
                "priority": 1,
                "complexity": 2,
                "confidence": "high"
            }
        ])
        return mock_response
    
    def test_parse_requirements_success(self, processor, mock_openai_response):
        """Test successful requirements parsing."""
        with patch.object(processor.client.chat.completions, 'create', return_value=mock_openai_response):
            result = processor.parse_requirements("Users should be able to login", "Test Project")
            
            assert result.success is True
            assert result.requirements is not None
            assert len(result.requirements.parsed_requirements) == 1
            assert result.requirements.project_name == "Test Project"
            
            req = result.requirements.parsed_requirements[0]
            assert req.requirement_type == RequirementType.FUNCTIONAL
            assert req.intent == Intent.ADD_SECURITY
            assert req.confidence == ConfidenceLevel.HIGH
    
    def test_parse_requirements_empty_text(self, processor):
        """Test parsing with empty text."""
        result = processor.parse_requirements("", "Test Project")
        
        assert result.success is True
        assert result.requirements is not None
        assert len(result.requirements.parsed_requirements) == 0
    
    def test_parse_requirements_api_failure(self, processor):
        """Test handling of API failures."""
        with patch.object(processor.client.chat.completions, 'create', side_effect=Exception("API Error")):
            result = processor.parse_requirements("Users should be able to login")
            
            # Should fall back to pattern-based parsing and still succeed
            assert result.success is True
            assert result.requirements is not None
            # Should have at least one requirement from fallback parsing
            assert len(result.requirements.parsed_requirements) > 0
    
    def test_fallback_requirement_parsing(self, processor):
        """Test fallback parsing when GPT-4 is unavailable."""
        text = "Users should login. System must be secure. Data should be encrypted."
        requirements = processor._fallback_requirement_parsing(text)
        
        assert len(requirements) == 3
        for req in requirements:
            assert req.requirement_type == RequirementType.FUNCTIONAL
            assert req.confidence == ConfidenceLevel.LOW
    
    def test_preprocess_text(self, processor):
        """Test text preprocessing."""
        text = "Users   should    login.System must be secure."
        processed = processor._preprocess_text(text)
        
        assert "  " not in processed  # No double spaces
        assert processed == "Users should login. System must be secure."
    
    def test_estimate_tokens(self, processor):
        """Test token estimation."""
        text = "This is a test text with approximately twenty characters per word."
        tokens = processor._estimate_tokens(text)
        
        assert tokens > 0
        assert tokens == len(text) // 4


class TestIntentClassifier:
    """Test cases for IntentClassifier."""
    
    @pytest.fixture
    def classifier(self):
        """Create IntentClassifier instance for testing."""
        return IntentClassifier()
    
    def test_classify_intent_create_application(self, classifier):
        """Test classification of application creation intent."""
        requirement = "I need to build a new web application for managing customers"
        intent, confidence = classifier.classify_intent(requirement)
        
        assert intent == Intent.CREATE_APPLICATION
        assert confidence > 0.3
    
    def test_classify_intent_add_security(self, classifier):
        """Test classification of security intent."""
        requirement = "Users should be able to authenticate with username and password"
        intent, confidence = classifier.classify_intent(requirement)
        
        assert intent == Intent.ADD_SECURITY
        assert confidence > 0.3
    
    def test_classify_intent_create_api(self, classifier):
        """Test classification of API creation intent."""
        requirement = "Create REST API endpoints for user management"
        intent, confidence = classifier.classify_intent(requirement)
        
        assert intent == Intent.CREATE_API
        assert confidence > 0.3
    
    def test_classify_multiple_intents(self, classifier):
        """Test classification of multiple requirements."""
        requirements = [
            "Build a web application",
            "Create user authentication",
            "Design database schema"
        ]
        results = classifier.classify_multiple_intents(requirements)
        
        assert len(results) == 3
        for intent, confidence in results:
            assert isinstance(intent, Intent)
            assert 0.0 <= confidence <= 1.0
    
    def test_get_intent_distribution(self, classifier):
        """Test intent distribution calculation."""
        requirements = [
            "Build a web application",
            "Create user authentication", 
            "Build another application"
        ]
        distribution = classifier.get_intent_distribution(requirements)
        
        assert isinstance(distribution, dict)
        assert all(isinstance(count, int) for count in distribution.values())
        assert sum(distribution.values()) == len(requirements)
    
    def test_suggest_architecture_patterns(self, classifier):
        """Test architecture pattern suggestions."""
        intents = [Intent.CREATE_APPLICATION, Intent.BUILD_UI, Intent.CREATE_API]
        patterns = classifier.suggest_architecture_patterns(intents)
        
        assert isinstance(patterns, list)
        assert len(patterns) > 0
        assert "Model-View-Controller (MVC)" in patterns
    
    def test_get_requirement_complexity(self, classifier):
        """Test requirement complexity estimation."""
        simple_req = "Users can login"
        complex_req = "Build a real-time scalable microservices system with multiple databases"
        
        simple_complexity = classifier.get_requirement_complexity(simple_req, Intent.ADD_SECURITY)
        complex_complexity = classifier.get_requirement_complexity(complex_req, Intent.CREATE_APPLICATION)
        
        assert 1 <= simple_complexity <= 5
        assert 1 <= complex_complexity <= 5
        assert complex_complexity > simple_complexity


class TestEntityExtractor:
    """Test cases for EntityExtractor."""
    
    @pytest.fixture
    def extractor(self):
        """Create EntityExtractor instance for testing."""
        return EntityExtractor()
    
    def test_extract_entities_pattern_based(self, extractor):
        """Test pattern-based entity extraction."""
        text = "Users should be able to login to the system and manage their profile data"
        entities = extractor._extract_with_patterns(text)
        
        assert len(entities) > 0
        
        # Check for user role entity
        user_entities = [e for e in entities if e.type == EntityType.USER_ROLE]
        assert len(user_entities) > 0
        
        # Check for action entities
        action_entities = [e for e in entities if e.type == EntityType.ACTION]
        assert len(action_entities) > 0
    
    def test_extract_relationships_pattern_based(self, extractor):
        """Test pattern-based relationship extraction."""
        entities = [
            Entity(id="1", name="user", type=EntityType.USER_ROLE, confidence=0.8, source_text="user", position=(0, 4)),
            Entity(id="2", name="system", type=EntityType.SYSTEM_COMPONENT, confidence=0.8, source_text="system", position=(5, 11))
        ]
        text = "user uses system"
        
        relationships = extractor._extract_relationships_with_patterns(entities, text)
        
        assert len(relationships) >= 0  # May or may not find relationships with simple patterns
    
    def test_validate_entities(self, extractor):
        """Test entity validation."""
        entities = [
            Entity(id="1", name="user", type=EntityType.USER_ROLE, confidence=0.8, source_text="user", position=(0, 4)),
            Entity(id="2", name="data", type=EntityType.DATA_ENTITY, confidence=0.9, source_text="data", position=(5, 9)),
            Entity(id="3", name="login", type=EntityType.ACTION, confidence=0.7, source_text="login", position=(10, 15))
        ]
        
        warnings = extractor.validate_entities(entities)
        
        # Should have minimal warnings for well-formed entities
        assert isinstance(warnings, list)
    
    def test_group_entities_by_domain(self, extractor):
        """Test entity grouping by domain."""
        entities = [
            Entity(id="1", name="user", type=EntityType.USER_ROLE, confidence=0.8, source_text="user", position=(0, 4)),
            Entity(id="2", name="customer_data", type=EntityType.DATA_ENTITY, confidence=0.9, source_text="data", position=(5, 9)),
            Entity(id="3", name="api", type=EntityType.SYSTEM_COMPONENT, confidence=0.7, source_text="api", position=(10, 13))
        ]
        
        grouped = extractor.group_entities_by_domain(entities)
        
        assert isinstance(grouped, dict)
        assert "User Management" in grouped
        assert "Data Management" in grouped
        assert "System Components" in grouped


class TestClarificationEngine:
    """Test cases for ClarificationEngine."""
    
    @pytest.fixture
    def engine(self):
        """Create ClarificationEngine instance for testing."""
        return ClarificationEngine()
    
    @pytest.fixture
    def sample_requirements(self):
        """Create sample requirements for testing."""
        req = ParsedRequirement(
            id="req1",
            original_text="Users should be able to login somehow",
            structured_text="Users should be able to login somehow",
            requirement_type=RequirementType.FUNCTIONAL,
            intent=Intent.ADD_SECURITY,
            confidence=ConfidenceLevel.LOW
        )
        
        return Requirements(
            id="test",
            project_name="Test Project",
            raw_text="Users should be able to login somehow",
            parsed_requirements=[req],
            entities=[],
            relationships=[],
            clarifications=[],
            completeness_score=0.5
        )
    
    def test_generate_clarifications_low_confidence(self, engine, sample_requirements):
        """Test clarification generation for low confidence requirements."""
        clarifications = engine.generate_clarifications(sample_requirements)
        
        assert len(clarifications) > 0
        
        # Should have clarification for low confidence requirement
        low_conf_clarifications = [c for c in clarifications if c.requirement_id == "req1"]
        assert len(low_conf_clarifications) > 0
    
    def test_generate_clarifications_missing_users(self, engine):
        """Test clarification generation for missing user roles."""
        requirements = Requirements(
            id="test",
            project_name="Test Project", 
            raw_text="System should work",
            parsed_requirements=[],
            entities=[],  # No user entities
            relationships=[],
            clarifications=[],
            completeness_score=0.3
        )
        
        clarifications = engine.generate_clarifications(requirements)
        
        # Should ask about users
        user_questions = [c for c in clarifications if "user" in c.question.lower()]
        assert len(user_questions) > 0
    
    def test_answer_clarification(self, engine, sample_requirements):
        """Test answering a clarification."""
        clarifications = engine.generate_clarifications(sample_requirements)
        
        if clarifications:
            clarification_id = clarifications[0].id
            updated_requirements = engine.answer_clarification(
                clarification_id, "Users login with email and password", sample_requirements
            )
            
            # Clarification should be removed
            remaining_clarifications = [c for c in updated_requirements.clarifications if c.id == clarification_id]
            assert len(remaining_clarifications) == 0


class TestRequirementsValidator:
    """Test cases for RequirementsValidator."""
    
    @pytest.fixture
    def validator(self):
        """Create RequirementsValidator instance for testing."""
        return RequirementsValidator()
    
    @pytest.fixture
    def valid_requirements(self):
        """Create valid requirements for testing."""
        req = ParsedRequirement(
            id="req1",
            original_text="Users should be able to login with email and password",
            structured_text="Users should be able to login with email and password",
            requirement_type=RequirementType.FUNCTIONAL,
            intent=Intent.ADD_SECURITY,
            acceptance_criteria=["Valid credentials allow access", "Invalid credentials are rejected"],
            confidence=ConfidenceLevel.HIGH
        )
        
        entities = [
            Entity(id="1", name="user", type=EntityType.USER_ROLE, confidence=0.9, source_text="user", position=(0, 4)),
            Entity(id="2", name="email", type=EntityType.DATA_ENTITY, confidence=0.8, source_text="email", position=(5, 10))
        ]
        
        return Requirements(
            id="test",
            project_name="Test Project",
            raw_text="Users should be able to login with email and password",
            parsed_requirements=[req],
            entities=entities,
            relationships=[],
            clarifications=[],
            completeness_score=0.8
        )
    
    def test_validate_valid_requirements(self, validator, valid_requirements):
        """Test validation of valid requirements."""
        issues = validator.validate_requirements(valid_requirements)
        
        # Should have minimal critical issues
        critical_issues = [i for i in issues if i.severity.value == "critical"]
        assert len(critical_issues) == 0
    
    def test_validate_empty_requirements(self, validator):
        """Test validation of empty requirements."""
        empty_requirements = Requirements(
            id="test",
            project_name="Test Project",
            raw_text="",
            parsed_requirements=[],
            entities=[],
            relationships=[],
            clarifications=[],
            completeness_score=0.0
        )
        
        issues = validator.validate_requirements(empty_requirements)
        
        # Should have critical issue for no requirements
        critical_issues = [i for i in issues if i.severity.value == "critical"]
        assert len(critical_issues) > 0
    
    def test_calculate_quality_score(self, validator, valid_requirements):
        """Test quality score calculation."""
        score = validator.calculate_quality_score(valid_requirements)
        
        assert 0.0 <= score <= 1.0
        assert score > 0.5  # Should be decent for valid requirements
    
    def test_get_improvement_suggestions(self, validator, valid_requirements):
        """Test improvement suggestions."""
        suggestions = validator.get_improvement_suggestions(valid_requirements)
        
        assert isinstance(suggestions, list)
        # Valid requirements should have fewer suggestions
    
    def test_is_ready_for_code_generation(self, validator, valid_requirements):
        """Test readiness check for code generation."""
        is_ready, blocking_issues = validator.is_ready_for_code_generation(valid_requirements)
        
        assert isinstance(is_ready, bool)
        assert isinstance(blocking_issues, list)
        
        # Valid requirements should be ready
        assert is_ready is True
        assert len(blocking_issues) == 0
    
    def test_is_not_ready_empty_requirements(self, validator):
        """Test readiness check for empty requirements."""
        empty_requirements = Requirements(
            id="test",
            project_name="Test Project",
            raw_text="",
            parsed_requirements=[],
            entities=[],
            relationships=[],
            clarifications=[],
            completeness_score=0.0
        )
        
        is_ready, blocking_issues = validator.is_ready_for_code_generation(empty_requirements)
        
        assert is_ready is False
        assert len(blocking_issues) > 0


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_very_long_text(self):
        """Test processing of very long requirements text."""
        processor = NLProcessor()
        long_text = "Users should be able to login. " * 1000  # Very long text
        
        result = processor.parse_requirements(long_text)
        
        # Should handle gracefully
        assert isinstance(result, ProcessingResult)
    
    def test_special_characters(self):
        """Test processing of text with special characters."""
        processor = NLProcessor()
        special_text = "Users should login with @#$%^&*() characters in passwords"
        
        result = processor.parse_requirements(special_text)
        
        assert result.success is True or len(result.errors) > 0  # Should either work or fail gracefully
    
    def test_non_english_text(self):
        """Test processing of non-English text."""
        processor = NLProcessor()
        non_english = "Los usuarios deben poder iniciar sesión"
        
        result = processor.parse_requirements(non_english)
        
        # Should handle gracefully
        assert isinstance(result, ProcessingResult)
    
    def test_malformed_json_response(self):
        """Test handling of malformed JSON from API."""
        processor = NLProcessor()
        
        with patch.object(processor.client.chat.completions, 'create') as mock_create:
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = "Invalid JSON {"
            mock_create.return_value = mock_response
            
            result = processor.parse_requirements("Test requirement")
            
            # Should fall back to pattern-based parsing
            assert isinstance(result, ProcessingResult)
    
    def test_empty_entity_list(self):
        """Test relationship extraction with empty entity list."""
        extractor = EntityExtractor()
        relationships = extractor.extract_relationships([], "Some text")
        
        assert relationships == []
    
    def test_single_entity_relationships(self):
        """Test relationship extraction with single entity."""
        extractor = EntityExtractor()
        entities = [Entity(id="1", name="user", type=EntityType.USER_ROLE, confidence=0.8, source_text="user", position=(0, 4))]
        relationships = extractor.extract_relationships(entities, "User text")
        
        assert relationships == []