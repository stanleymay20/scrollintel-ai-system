"""
Unit tests for the Entity Extractor in the Automated Code Generation System.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from scrollintel.models.nlp_models import Base, EntityType
from scrollintel.engines.entity_extractor import (
    EntityExtractor, ExtractedEntity, ExtractedRelationship, EntityExtractionResult
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
    with patch('scrollintel.engines.entity_extractor.openai.OpenAI') as mock_client:
        mock_instance = Mock()
        mock_client.return_value = mock_instance
        
        # Mock the async chat completion
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = '''
        {
            "entities": [
                {
                    "type": "USER_ROLE",
                    "name": "administrator",
                    "description": "System administrator with full access",
                    "attributes": {"permissions": "all", "level": "admin"},
                    "confidence": 0.9
                },
                {
                    "type": "BUSINESS_OBJECT",
                    "name": "user_account",
                    "description": "User account information",
                    "attributes": {"fields": ["username", "email", "password"]},
                    "confidence": 0.8
                },
                {
                    "type": "SYSTEM_COMPONENT",
                    "name": "authentication_service",
                    "description": "Service handling user authentication",
                    "attributes": {"type": "microservice"},
                    "confidence": 0.85
                }
            ],
            "relationships": [
                {
                    "source": "administrator",
                    "target": "user_account",
                    "type": "manages",
                    "description": "Administrator manages user accounts",
                    "confidence": 0.8
                },
                {
                    "source": "authentication_service",
                    "target": "user_account",
                    "type": "processes",
                    "description": "Authentication service processes user accounts",
                    "confidence": 0.9
                }
            ]
        }
        '''
        
        mock_instance.chat.completions.acreate = AsyncMock(return_value=mock_response)
        yield mock_instance


@pytest.fixture
def mock_spacy():
    """Mock spaCy NLP model."""
    with patch('scrollintel.engines.entity_extractor.spacy.load') as mock_load:
        mock_nlp = Mock()
        mock_load.return_value = mock_nlp
        
        # Mock document and entities
        mock_doc = Mock()
        mock_ent1 = Mock()
        mock_ent1.text = "John Doe"
        mock_ent1.label_ = "PERSON"
        mock_ent1.start_char = 0
        mock_ent1.end_char = 8
        
        mock_ent2 = Mock()
        mock_ent2.text = "Acme Corp"
        mock_ent2.label_ = "ORG"
        mock_ent2.start_char = 20
        mock_ent2.end_char = 29
        
        mock_doc.ents = [mock_ent1, mock_ent2]
        mock_nlp.return_value = mock_doc
        
        yield mock_nlp


@pytest.fixture
def entity_extractor(db_session, mock_openai_client, mock_spacy):
    """Create EntityExtractor instance with mocked dependencies."""
    with patch('scrollintel.engines.entity_extractor.get_settings') as mock_settings:
        mock_settings.return_value.openai_api_key = "test-key"
        extractor = EntityExtractor(db_session)
        extractor.client = mock_openai_client
        extractor.nlp = mock_spacy
        return extractor


class TestExtractedEntity:
    """Test cases for ExtractedEntity dataclass."""
    
    def test_creation(self):
        """Test creating ExtractedEntity."""
        entity = ExtractedEntity(
            name="user",
            entity_type=EntityType.USER_ROLE,
            description="System user",
            attributes={"permissions": "read"},
            confidence=0.8,
            mentions=["user", "users"],
            context="The user can access the system"
        )
        
        assert entity.name == "user"
        assert entity.entity_type == EntityType.USER_ROLE
        assert entity.description == "System user"
        assert entity.attributes["permissions"] == "read"
        assert entity.confidence == 0.8
        assert len(entity.mentions) == 2
        assert "system" in entity.context


class TestExtractedRelationship:
    """Test cases for ExtractedRelationship dataclass."""
    
    def test_creation(self):
        """Test creating ExtractedRelationship."""
        relationship = ExtractedRelationship(
            source_entity="user",
            target_entity="profile",
            relationship_type="has_one",
            description="User has one profile",
            confidence=0.9,
            context="Each user has one profile"
        )
        
        assert relationship.source_entity == "user"
        assert relationship.target_entity == "profile"
        assert relationship.relationship_type == "has_one"
        assert relationship.description == "User has one profile"
        assert relationship.confidence == 0.9
        assert "profile" in relationship.context


class TestEntityExtractionResult:
    """Test cases for EntityExtractionResult."""
    
    def test_creation(self):
        """Test creating EntityExtractionResult."""
        entities = [
            ExtractedEntity("user", EntityType.USER_ROLE, "User", {}, 0.8, [], "")
        ]
        relationships = [
            ExtractedRelationship("user", "profile", "has_one", "User has profile", 0.9, "")
        ]
        
        result = EntityExtractionResult(
            entities=entities,
            relationships=relationships,
            processing_time=1.5,
            confidence_score=0.85,
            warnings=["Low confidence entity detected"]
        )
        
        assert len(result.entities) == 1
        assert len(result.relationships) == 1
        assert result.processing_time == 1.5
        assert result.confidence_score == 0.85
        assert len(result.warnings) == 1


class TestEntityExtractor:
    """Test cases for EntityExtractor."""
    
    @pytest.mark.asyncio
    async def test_extract_entities_success(self, entity_extractor):
        """Test successful entity extraction."""
        text = "The administrator manages user accounts through the authentication service"
        
        result = await entity_extractor.extract_entities(text)
        
        assert isinstance(result, EntityExtractionResult)
        assert len(result.entities) > 0
        assert result.processing_time > 0
        assert result.confidence_score > 0
        
        # Check that entities were extracted
        entity_names = [entity.name for entity in result.entities]
        assert any("administrator" in name.lower() for name in entity_names)

    @pytest.mark.asyncio
    async def test_extract_entities_with_relationships(self, entity_extractor):
        """Test entity extraction with relationships."""
        text = "Users have profiles and administrators manage user accounts"
        
        result = await entity_extractor.extract_entities(text)
        
        assert len(result.entities) > 0
        assert len(result.relationships) >= 0  # May or may not find relationships
        
        # Check entity types
        entity_types = [entity.entity_type for entity in result.entities]
        assert EntityType.USER_ROLE in entity_types or EntityType.BUSINESS_OBJECT in entity_types

    def test_extract_with_rules_user_roles(self, entity_extractor):
        """Test rule-based extraction of user roles."""
        text = "The administrator and regular user can access the system"
        
        entities = entity_extractor._extract_with_rules(text)
        
        user_role_entities = [e for e in entities if e.entity_type == EntityType.USER_ROLE]
        assert len(user_role_entities) > 0
        
        entity_names = [e.name for e in user_role_entities]
        assert "administrator" in entity_names or "user" in entity_names

    def test_extract_with_rules_business_objects(self, entity_extractor):
        """Test rule-based extraction of business objects."""
        text = "The system manages orders, products, and customer information"
        
        entities = entity_extractor._extract_with_rules(text)
        
        business_entities = [e for e in entities if e.entity_type == EntityType.BUSINESS_OBJECT]
        assert len(business_entities) > 0
        
        entity_names = [e.name for e in business_entities]
        expected_names = ["order", "product", "customer"]
        assert any(name in entity_names for name in expected_names)

    def test_extract_with_rules_system_components(self, entity_extractor):
        """Test rule-based extraction of system components."""
        text = "The database stores data and the API provides access to services"
        
        entities = entity_extractor._extract_with_rules(text)
        
        system_entities = [e for e in entities if e.entity_type == EntityType.SYSTEM_COMPONENT]
        assert len(system_entities) > 0
        
        entity_names = [e.name for e in system_entities]
        expected_names = ["database", "api", "service"]
        assert any(name in entity_names for name in expected_names)

    def test_extract_with_rules_technologies(self, entity_extractor):
        """Test rule-based extraction of technologies."""
        text = "The application uses React frontend with Node.js backend and PostgreSQL database"
        
        entities = entity_extractor._extract_with_rules(text)
        
        tech_entities = [e for e in entities if e.entity_type == EntityType.TECHNOLOGY]
        assert len(tech_entities) > 0
        
        entity_names = [e.name for e in tech_entities]
        expected_names = ["react", "node", "postgresql"]
        assert any(name in entity_names for name in expected_names)

    def test_extract_with_spacy(self, entity_extractor):
        """Test spaCy-based entity extraction."""
        text = "John Doe works at Acme Corp"
        
        entities = entity_extractor._extract_with_spacy(text)
        
        assert len(entities) == 2
        
        # Check person entity
        person_entities = [e for e in entities if e.entity_type == EntityType.USER_ROLE]
        assert len(person_entities) == 1
        assert person_entities[0].name == "John Doe"
        
        # Check organization entity
        org_entities = [e for e in entities if e.entity_type == EntityType.BUSINESS_OBJECT]
        assert len(org_entities) == 1
        assert org_entities[0].name == "Acme Corp"

    def test_extract_with_spacy_no_model(self, entity_extractor):
        """Test spaCy extraction when model is not available."""
        entity_extractor.nlp = None
        
        entities = entity_extractor._extract_with_spacy("test text")
        
        assert len(entities) == 0

    def test_map_spacy_label_to_entity_type(self, entity_extractor):
        """Test mapping spaCy labels to entity types."""
        assert entity_extractor._map_spacy_label_to_entity_type("PERSON") == EntityType.USER_ROLE
        assert entity_extractor._map_spacy_label_to_entity_type("ORG") == EntityType.BUSINESS_OBJECT
        assert entity_extractor._map_spacy_label_to_entity_type("PRODUCT") == EntityType.BUSINESS_OBJECT
        assert entity_extractor._map_spacy_label_to_entity_type("UNKNOWN") is None

    @pytest.mark.asyncio
    async def test_extract_with_ai_success(self, entity_extractor):
        """Test AI-powered entity extraction."""
        text = "The administrator manages user accounts"
        
        entities, relationships = await entity_extractor._extract_with_ai(text)
        
        assert len(entities) == 3  # Based on mock response
        assert len(relationships) == 2  # Based on mock response
        
        # Check entity details
        admin_entity = next((e for e in entities if e.name == "administrator"), None)
        assert admin_entity is not None
        assert admin_entity.entity_type == EntityType.USER_ROLE
        assert admin_entity.confidence == 0.9

    @pytest.mark.asyncio
    async def test_extract_with_ai_error(self, entity_extractor):
        """Test AI extraction error handling."""
        # Mock OpenAI to raise an exception
        entity_extractor.client.chat.completions.acreate = AsyncMock(
            side_effect=Exception("API Error")
        )
        
        entities, relationships = await entity_extractor._extract_with_ai("test text")
        
        assert len(entities) == 0
        assert len(relationships) == 0

    def test_parse_ai_response_valid_json(self, entity_extractor):
        """Test parsing valid JSON AI response."""
        content = '''
        {
            "entities": [
                {
                    "type": "USER_ROLE",
                    "name": "user",
                    "description": "System user",
                    "attributes": {"role": "basic"},
                    "confidence": 0.8
                }
            ],
            "relationships": [
                {
                    "source": "user",
                    "target": "profile",
                    "type": "has_one",
                    "description": "User has profile",
                    "confidence": 0.9
                }
            ]
        }
        '''
        
        result = entity_extractor._parse_ai_response(content)
        
        assert "entities" in result
        assert "relationships" in result
        assert len(result["entities"]) == 1
        assert len(result["relationships"]) == 1

    def test_parse_ai_response_invalid_json(self, entity_extractor):
        """Test parsing invalid JSON AI response."""
        content = "This is not valid JSON"
        
        result = entity_extractor._parse_ai_response(content)
        
        assert result == {"entities": [], "relationships": []}

    def test_extract_relationships_with_rules(self, entity_extractor):
        """Test rule-based relationship extraction."""
        text = "User has many orders and belongs to organization"
        
        # Create mock entities
        entities = [
            ExtractedEntity("user", EntityType.USER_ROLE, "", {}, 0.8, [], ""),
            ExtractedEntity("order", EntityType.BUSINESS_OBJECT, "", {}, 0.8, [], ""),
            ExtractedEntity("organization", EntityType.BUSINESS_OBJECT, "", {}, 0.8, [], "")
        ]
        
        relationships = entity_extractor._extract_relationships_with_rules(text, entities)
        
        assert len(relationships) > 0
        
        # Check for "has_many" relationship
        has_many_rels = [r for r in relationships if r.relationship_type == "has_many"]
        assert len(has_many_rels) > 0
        
        # Check for "belongs_to" relationship
        belongs_to_rels = [r for r in relationships if r.relationship_type == "belongs_to"]
        assert len(belongs_to_rels) > 0

    def test_merge_entities(self, entity_extractor):
        """Test merging entities from multiple sources."""
        entities1 = [
            ExtractedEntity("user", EntityType.USER_ROLE, "User role", {"perm": "read"}, 0.7, ["user"], "")
        ]
        entities2 = [
            ExtractedEntity("user", EntityType.USER_ROLE, "System user", {"level": "basic"}, 0.8, ["users"], ""),
            ExtractedEntity("admin", EntityType.USER_ROLE, "Administrator", {}, 0.9, ["admin"], "")
        ]
        
        merged = entity_extractor._merge_entities(entities1, entities2)
        
        assert len(merged) == 2  # user (merged) and admin
        
        # Check merged user entity
        user_entity = next((e for e in merged if e.name == "user"), None)
        assert user_entity is not None
        assert user_entity.confidence == 0.8  # Should take max confidence
        assert len(user_entity.mentions) == 2  # Should merge mentions
        assert "perm" in user_entity.attributes and "level" in user_entity.attributes

    def test_merge_relationships(self, entity_extractor):
        """Test merging relationships from multiple sources."""
        relationships1 = [
            ExtractedRelationship("user", "profile", "has_one", "User has profile", 0.7, "")
        ]
        relationships2 = [
            ExtractedRelationship("user", "profile", "has_one", "User owns profile", 0.8, ""),
            ExtractedRelationship("user", "order", "has_many", "User has orders", 0.9, "")
        ]
        
        merged = entity_extractor._merge_relationships(relationships1, relationships2)
        
        assert len(merged) == 2  # user-profile (merged) and user-order
        
        # Check merged relationship
        user_profile_rel = next(
            (r for r in merged if r.source_entity == "user" and r.target_entity == "profile"), 
            None
        )
        assert user_profile_rel is not None
        assert user_profile_rel.confidence == 0.8  # Should take max confidence

    def test_deduplicate_entities(self, entity_extractor):
        """Test entity deduplication."""
        entities = [
            ExtractedEntity("user", EntityType.USER_ROLE, "User", {}, 0.8, [], ""),
            ExtractedEntity("User", EntityType.USER_ROLE, "User role", {}, 0.7, [], ""),
            ExtractedEntity("admin", EntityType.USER_ROLE, "Admin", {}, 0.9, [], "")
        ]
        
        deduplicated = entity_extractor._deduplicate_entities(entities)
        
        assert len(deduplicated) == 2  # user and admin (User should be deduplicated)
        
        entity_names = [e.name.lower() for e in deduplicated]
        assert "user" in entity_names
        assert "admin" in entity_names

    def test_get_context(self, entity_extractor):
        """Test context extraction around text matches."""
        text = "The quick brown fox jumps over the lazy dog"
        start = 16  # "fox"
        end = 19
        
        context = entity_extractor._get_context(text, start, end, window=10)
        
        assert "brown fox jumps" in context
        assert len(context) <= len(text)

    def test_calculate_confidence(self, entity_extractor):
        """Test confidence score calculation."""
        entities = [
            ExtractedEntity("user", EntityType.USER_ROLE, "", {}, 0.8, [], ""),
            ExtractedEntity("admin", EntityType.USER_ROLE, "", {}, 0.9, [], "")
        ]
        relationships = [
            ExtractedRelationship("user", "admin", "reports_to", "", 0.7, "")
        ]
        
        confidence = entity_extractor._calculate_confidence(entities, relationships)
        expected = (0.8 + 0.9 + 0.7) / 3
        assert abs(confidence - expected) < 0.01

    def test_calculate_confidence_empty(self, entity_extractor):
        """Test confidence calculation with empty lists."""
        confidence = entity_extractor._calculate_confidence([], [])
        assert confidence == 0.0

    def test_generate_warnings(self, entity_extractor):
        """Test warning generation."""
        # Test with no entities
        warnings = entity_extractor._generate_warnings([], [])
        assert any("No entities extracted" in warning for warning in warnings)
        
        # Test with few entities
        entities = [ExtractedEntity("user", EntityType.USER_ROLE, "", {}, 0.8, [], "")]
        warnings = entity_extractor._generate_warnings(entities, [])
        assert any("Few entities extracted" in warning for warning in warnings)
        assert any("No relationships identified" in warning for warning in warnings)
        
        # Test with low confidence entities
        low_conf_entities = [ExtractedEntity("user", EntityType.USER_ROLE, "", {}, 0.4, [], "")]
        warnings = entity_extractor._generate_warnings(low_conf_entities, [])
        assert any("low confidence scores" in warning for warning in warnings)

    def test_convert_to_models(self, entity_extractor):
        """Test conversion to Pydantic models."""
        extraction_result = EntityExtractionResult(
            entities=[
                ExtractedEntity("user", EntityType.USER_ROLE, "User role", {"perm": "read"}, 0.8, [], ""),
                ExtractedEntity("profile", EntityType.BUSINESS_OBJECT, "User profile", {}, 0.9, [], "")
            ],
            relationships=[
                ExtractedRelationship("user", "profile", "has_one", "User has profile", 0.8, "")
            ],
            processing_time=1.0,
            confidence_score=0.85,
            warnings=[]
        )
        
        entity_models, relationship_models = entity_extractor.convert_to_models(extraction_result)
        
        assert len(entity_models) == 2
        assert len(relationship_models) == 1
        
        # Check entity models
        user_model = next((e for e in entity_models if e.name == "user"), None)
        assert user_model is not None
        assert user_model.entity_type == EntityType.USER_ROLE
        assert user_model.confidence_score == 0.8
        
        # Check relationship model
        rel_model = relationship_models[0]
        assert rel_model.relationship_type == "has_one"
        assert rel_model.confidence_score == 0.8

    @pytest.mark.asyncio
    async def test_extract_entities_error_handling(self, entity_extractor):
        """Test error handling in entity extraction."""
        # Mock all extraction methods to raise exceptions
        with patch.object(entity_extractor, '_extract_with_rules', side_effect=Exception("Rule error")):
            with patch.object(entity_extractor, '_extract_with_spacy', side_effect=Exception("spaCy error")):
                with patch.object(entity_extractor, '_extract_with_ai', side_effect=Exception("AI error")):
                    
                    result = await entity_extractor.extract_entities("test text")
                    
                    assert len(result.entities) == 0
                    assert len(result.relationships) == 0
                    assert result.confidence_score == 0.0
                    assert len(result.warnings) > 0
                    assert any("Extraction failed" in warning for warning in result.warnings)