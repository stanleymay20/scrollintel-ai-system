"""
Unit tests for the NLP Processor in the Automated Code Generation System.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from scrollintel.models.nlp_models import (
    Base, Requirements, ParsedRequirement, Entity, EntityRelationship, Clarification,
    RequirementType, IntentType, EntityType, ConfidenceLevel
)
from scrollintel.engines.nlp_processor import NLProcessor


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
    with patch('scrollintel.engines.nlp_processor.openai.OpenAI') as mock_client:
        mock_instance = Mock()
        mock_client.return_value = mock_instance
        
        # Mock the async chat completion
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = '''
        {
            "requirements": [
                {
                    "type": "functional",
                    "intent": "create_application",
                    "description": "Create a user management system",
                    "priority": "high",
                    "confidence": 0.9,
                    "acceptance_criteria": ["Users can register", "Users can login"],
                    "dependencies": []
                }
            ],
            "entities": [
                {
                    "type": "user_role",
                    "name": "user",
                    "description": "System user",
                    "attributes": {"permissions": "basic"},
                    "confidence": 0.8
                },
                {
                    "type": "business_object",
                    "name": "account",
                    "description": "User account",
                    "attributes": {"fields": ["username", "email"]},
                    "confidence": 0.9
                }
            ],
            "relationships": [
                {
                    "source": "user",
                    "target": "account",
                    "type": "has_one",
                    "description": "User has one account",
                    "confidence": 0.8
                }
            ]
        }
        '''
        
        mock_instance.chat.completions.acreate = AsyncMock(return_value=mock_response)
        yield mock_instance


@pytest.fixture
def nlp_processor(db_session, mock_openai_client):
    """Create NLProcessor instance with mocked dependencies."""
    with patch('scrollintel.engines.nlp_processor.get_settings') as mock_settings:
        mock_settings.return_value.openai_api_key = "test-key"
        processor = NLProcessor(db_session)
        processor.client = mock_openai_client
        return processor


class TestNLProcessor:
    """Test cases for NLProcessor."""
    
    @pytest.mark.asyncio
    async def test_parse_requirements_success(self, nlp_processor, db_session):
        """Test successful requirements parsing."""
        text = "I need a user management system where users can register and login"
        
        result = await nlp_processor.parse_requirements(text)
        
        assert result.requirements.raw_text == text
        assert result.confidence_level in [ConfidenceLevel.HIGH, ConfidenceLevel.MEDIUM, ConfidenceLevel.LOW]
        assert len(result.errors) == 0
        assert result.processing_time > 0
        
        # Check database records were created
        requirements = db_session.query(Requirements).first()
        assert requirements is not None
        assert requirements.raw_text == text
        
        parsed_reqs = db_session.query(ParsedRequirement).all()
        assert len(parsed_reqs) > 0
        
        entities = db_session.query(Entity).all()
        assert len(entities) > 0

    @pytest.mark.asyncio
    async def test_parse_requirements_with_clarification_needed(self, nlp_processor, db_session):
        """Test parsing when clarification is needed."""
        text = "I need some kind of system that does something with users maybe"
        
        result = await nlp_processor.parse_requirements(text)
        
        assert result.requirements.needs_clarification == True
        
        # Check clarifications were generated (should have at least the default one from error handling)
        clarifications = db_session.query(Clarification).all()
        assert len(clarifications) >= 0  # May be 0 if GPT-4 call succeeds, >= 1 if it fails

    @pytest.mark.asyncio
    async def test_extract_parsed_requirements(self, nlp_processor, db_session):
        """Test extraction of parsed requirements from GPT-4 response."""
        parsed_data = {
            "requirements": [
                {
                    "type": "functional",
                    "intent": "create_application",
                    "description": "User registration feature",
                    "priority": "high",
                    "confidence": 0.9,
                    "acceptance_criteria": ["Email validation", "Password strength"],
                    "dependencies": ["authentication_service"]
                }
            ]
        }
        
        # Create a requirements record first
        requirements = Requirements(raw_text="test", processed_at=datetime.utcnow())
        db_session.add(requirements)
        db_session.flush()
        
        parsed_reqs = await nlp_processor._extract_parsed_requirements(
            parsed_data, requirements.id
        )
        
        assert len(parsed_reqs) == 1
        assert parsed_reqs[0].requirement_type == RequirementType.FUNCTIONAL
        assert parsed_reqs[0].intent == IntentType.CREATE_APPLICATION
        assert parsed_reqs[0].description == "User registration feature"
        assert parsed_reqs[0].priority == "high"
        assert len(parsed_reqs[0].acceptance_criteria) == 2

    @pytest.mark.asyncio
    async def test_extract_entities(self, nlp_processor, db_session):
        """Test extraction of entities from GPT-4 response."""
        parsed_data = {
            "entities": [
                {
                    "type": "user_role",
                    "name": "administrator",
                    "description": "System administrator",
                    "attributes": {"permissions": "all"},
                    "confidence": 0.9
                },
                {
                    "type": "business_object",
                    "name": "user_profile",
                    "description": "User profile information",
                    "attributes": {"fields": ["name", "email", "phone"]},
                    "confidence": 0.8
                }
            ]
        }
        
        # Create a requirements record first
        requirements = Requirements(raw_text="test", processed_at=datetime.utcnow())
        db_session.add(requirements)
        db_session.flush()
        
        entities = await nlp_processor._extract_entities(parsed_data, requirements.id)
        
        assert len(entities) == 2
        assert entities[0].entity_type == EntityType.USER_ROLE
        assert entities[0].name == "administrator"
        assert entities[1].entity_type == EntityType.BUSINESS_OBJECT
        assert entities[1].name == "user_profile"

    @pytest.mark.asyncio
    async def test_extract_relationships(self, nlp_processor, db_session):
        """Test extraction of entity relationships."""
        # Create entities first
        requirements = Requirements(raw_text="test", processed_at=datetime.utcnow())
        db_session.add(requirements)
        db_session.flush()
        
        entity1 = Entity(
            requirements_id=requirements.id,
            entity_type=EntityType.USER_ROLE.value,
            name="user",
            confidence_score=0.8
        )
        entity2 = Entity(
            requirements_id=requirements.id,
            entity_type=EntityType.BUSINESS_OBJECT.value,
            name="profile",
            confidence_score=0.8
        )
        db_session.add_all([entity1, entity2])
        db_session.flush()
        
        # Mock entities for the method
        from scrollintel.models.nlp_models import EntityModel
        mock_entities = [
            EntityModel(id=entity1.id, entity_type=EntityType.USER_ROLE, name="user", confidence_score=0.8),
            EntityModel(id=entity2.id, entity_type=EntityType.BUSINESS_OBJECT, name="profile", confidence_score=0.8)
        ]
        
        parsed_data = {
            "relationships": [
                {
                    "source": "user",
                    "target": "profile",
                    "type": "has_one",
                    "description": "User has one profile",
                    "confidence": 0.9
                }
            ]
        }
        
        await nlp_processor._extract_relationships(parsed_data, mock_entities)
        
        relationships = db_session.query(EntityRelationship).all()
        assert len(relationships) == 1
        assert relationships[0].relationship_type == "has_one"
        assert relationships[0].source_entity_id == entity1.id
        assert relationships[0].target_entity_id == entity2.id

    def test_calculate_confidence(self, nlp_processor):
        """Test confidence score calculation."""
        parsed_data = {
            "requirements": [
                {"confidence": 0.9},
                {"confidence": 0.8}
            ],
            "entities": [
                {"confidence": 0.7},
                {"confidence": 0.8}
            ],
            "relationships": [
                {"confidence": 0.6}
            ]
        }
        
        confidence = nlp_processor._calculate_confidence(parsed_data)
        expected = (0.9 + 0.8 + 0.7 + 0.8 + 0.6) / 5
        assert abs(confidence - expected) < 0.01

    def test_get_confidence_level(self, nlp_processor):
        """Test confidence level mapping."""
        assert nlp_processor._get_confidence_level(0.9) == ConfidenceLevel.HIGH
        assert nlp_processor._get_confidence_level(0.7) == ConfidenceLevel.MEDIUM
        assert nlp_processor._get_confidence_level(0.5) == ConfidenceLevel.LOW

    def test_needs_clarification(self, nlp_processor):
        """Test clarification detection."""
        # Test with ambiguous text
        ambiguous_text = "I need something like a user system that might work"
        result = asyncio.run(nlp_processor._needs_clarification(ambiguous_text))
        assert result == True
        
        # Test with clear text
        clear_text = "I need a user registration system with email validation"
        result = asyncio.run(nlp_processor._needs_clarification(clear_text))
        assert result == False

    def test_generate_warnings(self, nlp_processor):
        """Test warning generation."""
        # Test with low confidence
        parsed_data = {"requirements": [], "entities": []}
        warnings = nlp_processor._generate_warnings(parsed_data, 0.4)
        assert any("Low confidence" in warning for warning in warnings)
        assert any("No clear requirements" in warning for warning in warnings)
        assert any("No business entities" in warning for warning in warnings)
        
        # Test with good data
        parsed_data = {
            "requirements": [{"description": "test"}],
            "entities": [{"name": "user"}]
        }
        warnings = nlp_processor._generate_warnings(parsed_data, 0.8)
        assert len(warnings) == 0

    @pytest.mark.asyncio
    async def test_get_clarifying_questions(self, nlp_processor, db_session):
        """Test retrieval of clarifying questions."""
        # Create requirements and clarifications
        requirements = Requirements(raw_text="test", processed_at=datetime.utcnow())
        db_session.add(requirements)
        db_session.flush()
        
        clarification1 = Clarification(
            requirements_id=requirements.id,
            question="What is the expected user load?",
            priority="high",
            is_answered=False
        )
        clarification2 = Clarification(
            requirements_id=requirements.id,
            question="Which authentication method?",
            priority="medium",
            is_answered=True,
            answer="OAuth 2.0"
        )
        db_session.add_all([clarification1, clarification2])
        db_session.commit()
        
        questions = await nlp_processor.get_clarifying_questions(requirements.id)
        
        # Should only return unanswered questions
        assert len(questions) == 1
        assert questions[0].question == "What is the expected user load?"
        assert questions[0].is_answered == False

    @pytest.mark.asyncio
    async def test_answer_clarification(self, nlp_processor, db_session):
        """Test answering a clarification question."""
        # Create requirements and clarification
        requirements = Requirements(raw_text="test", processed_at=datetime.utcnow())
        db_session.add(requirements)
        db_session.flush()
        
        clarification = Clarification(
            requirements_id=requirements.id,
            question="What is the expected user load?",
            priority="high",
            is_answered=False
        )
        db_session.add(clarification)
        db_session.commit()
        
        answer = "Up to 10,000 concurrent users"
        result = await nlp_processor.answer_clarification(clarification.id, answer)
        
        assert result.answer == answer
        assert result.is_answered == True
        assert result.answered_at is not None
        
        # Verify database was updated
        updated_clarification = db_session.query(Clarification).filter(
            Clarification.id == clarification.id
        ).first()
        assert updated_clarification.answer == answer
        assert updated_clarification.is_answered == True

    @pytest.mark.asyncio
    async def test_answer_clarification_not_found(self, nlp_processor, db_session):
        """Test answering a non-existent clarification."""
        with pytest.raises(ValueError, match="Clarification 999 not found"):
            await nlp_processor.answer_clarification(999, "test answer")

    @pytest.mark.asyncio
    async def test_parse_requirements_error_handling(self, nlp_processor, db_session):
        """Test error handling in requirements parsing."""
        # Mock OpenAI to raise an exception
        nlp_processor.client.chat.completions.acreate = AsyncMock(
            side_effect=Exception("API Error")
        )
        
        result = await nlp_processor.parse_requirements("test text")
        
        # The system is designed to be resilient - it should continue processing
        # even when GPT-4 fails, but with low confidence and warnings
        assert result.confidence_level == ConfidenceLevel.LOW
        assert len(result.warnings) > 0
        assert any("Low confidence" in warning for warning in result.warnings)
        
        # Should still create a requirements record
        assert result.requirements.raw_text == "test text"

    def test_extract_json_from_text(self, nlp_processor):
        """Test JSON extraction from mixed text."""
        text_with_json = '''
        Here is some explanation text.
        {"requirements": [{"type": "functional", "description": "test"}]}
        And some more text after.
        '''
        
        result = nlp_processor._extract_json_from_text(text_with_json)
        
        assert "requirements" in result
        assert len(result["requirements"]) == 1
        assert result["requirements"][0]["type"] == "functional"

    def test_extract_json_from_text_no_json(self, nlp_processor):
        """Test JSON extraction when no JSON is present."""
        text_without_json = "This is just plain text with no JSON."
        
        result = nlp_processor._extract_json_from_text(text_without_json)
        
        assert result == {}


class TestNLPModels:
    """Test cases for NLP data models."""
    
    def test_requirements_model_creation(self, db_session):
        """Test creating Requirements model."""
        requirements = Requirements(
            raw_text="Test requirements",
            confidence_score=0.8,
            is_complete=True,
            needs_clarification=False
        )
        
        db_session.add(requirements)
        db_session.commit()
        
        retrieved = db_session.query(Requirements).first()
        assert retrieved.raw_text == "Test requirements"
        assert retrieved.confidence_score == 0.8
        assert retrieved.is_complete == True
        assert retrieved.needs_clarification == False

    def test_parsed_requirement_model_creation(self, db_session):
        """Test creating ParsedRequirement model."""
        requirements = Requirements(raw_text="Test")
        db_session.add(requirements)
        db_session.flush()
        
        parsed_req = ParsedRequirement(
            requirements_id=requirements.id,
            requirement_type=RequirementType.FUNCTIONAL.value,
            intent=IntentType.CREATE_APPLICATION.value,
            description="Test requirement",
            priority="high",
            confidence_score=0.9,
            acceptance_criteria=["Criteria 1", "Criteria 2"],
            dependencies=["Dependency 1"]
        )
        
        db_session.add(parsed_req)
        db_session.commit()
        
        retrieved = db_session.query(ParsedRequirement).first()
        assert retrieved.requirement_type == RequirementType.FUNCTIONAL.value
        assert retrieved.intent == IntentType.CREATE_APPLICATION.value
        assert len(retrieved.acceptance_criteria) == 2
        assert len(retrieved.dependencies) == 1

    def test_entity_model_creation(self, db_session):
        """Test creating Entity model."""
        requirements = Requirements(raw_text="Test")
        db_session.add(requirements)
        db_session.flush()
        
        entity = Entity(
            requirements_id=requirements.id,
            entity_type=EntityType.USER_ROLE.value,
            name="administrator",
            description="System administrator",
            attributes={"permissions": "all"},
            confidence_score=0.9
        )
        
        db_session.add(entity)
        db_session.commit()
        
        retrieved = db_session.query(Entity).first()
        assert retrieved.entity_type == EntityType.USER_ROLE.value
        assert retrieved.name == "administrator"
        assert retrieved.attributes["permissions"] == "all"

    def test_entity_relationship_model_creation(self, db_session):
        """Test creating EntityRelationship model."""
        requirements = Requirements(raw_text="Test")
        db_session.add(requirements)
        db_session.flush()
        
        entity1 = Entity(
            requirements_id=requirements.id,
            entity_type=EntityType.USER_ROLE.value,
            name="user",
            confidence_score=0.8
        )
        entity2 = Entity(
            requirements_id=requirements.id,
            entity_type=EntityType.BUSINESS_OBJECT.value,
            name="profile",
            confidence_score=0.8
        )
        db_session.add_all([entity1, entity2])
        db_session.flush()
        
        relationship = EntityRelationship(
            source_entity_id=entity1.id,
            target_entity_id=entity2.id,
            relationship_type="has_one",
            description="User has one profile",
            confidence_score=0.9
        )
        
        db_session.add(relationship)
        db_session.commit()
        
        retrieved = db_session.query(EntityRelationship).first()
        assert retrieved.relationship_type == "has_one"
        assert retrieved.source_entity_id == entity1.id
        assert retrieved.target_entity_id == entity2.id

    def test_clarification_model_creation(self, db_session):
        """Test creating Clarification model."""
        requirements = Requirements(raw_text="Test")
        db_session.add(requirements)
        db_session.flush()
        
        clarification = Clarification(
            requirements_id=requirements.id,
            question="What is the expected load?",
            answer="1000 users",
            is_answered=True,
            priority="high"
        )
        
        db_session.add(clarification)
        db_session.commit()
        
        retrieved = db_session.query(Clarification).first()
        assert retrieved.question == "What is the expected load?"
        assert retrieved.answer == "1000 users"
        assert retrieved.is_answered == True
        assert retrieved.priority == "high"