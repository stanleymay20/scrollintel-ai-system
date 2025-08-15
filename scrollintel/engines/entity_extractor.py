"""
Entity Extraction Engine for the Automated Code Generation System.
Extracts business entities and their relationships from natural language requirements.
"""

import logging
import re
import json
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass
import openai
from sqlalchemy.orm import Session
import spacy
from collections import defaultdict

from scrollintel.models.nlp_models import EntityType, EntityModel, EntityRelationshipModel
from scrollintel.core.config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class ExtractedEntity:
    """Represents an extracted entity with metadata."""
    name: str
    entity_type: EntityType
    description: str
    attributes: Dict[str, Any]
    confidence: float
    mentions: List[str]
    context: str


@dataclass
class ExtractedRelationship:
    """Represents a relationship between entities."""
    source_entity: str
    target_entity: str
    relationship_type: str
    description: str
    confidence: float
    context: str


class EntityExtractionResult:
    """Result of entity extraction process."""
    
    def __init__(
        self,
        entities: List[ExtractedEntity],
        relationships: List[ExtractedRelationship],
        processing_time: float,
        confidence_score: float,
        warnings: List[str] = None
    ):
        self.entities = entities
        self.relationships = relationships
        self.processing_time = processing_time
        self.confidence_score = confidence_score
        self.warnings = warnings or []


class EntityExtractor:
    """
    Advanced Entity Extractor using NLP and AI techniques.
    Identifies business entities, technical components, and their relationships.
    """
    
    def __init__(self, db_session: Session):
        self.db_session = db_session
        self.settings = get_settings()
        self.client = openai.OpenAI(api_key=self.settings.openai_api_key)
        
        # Load spaCy model for NER
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found, using basic extraction only")
            self.nlp = None
        
        # Entity type patterns for rule-based extraction
        self.entity_patterns = {
            EntityType.USER_ROLE: [
                r'\b(user|admin|administrator|customer|client|manager|employee|staff|operator)\b',
                r'\b(role|actor|persona)\b',
                r'\bas\s+a\s+(\w+)\b'
            ],
            EntityType.BUSINESS_OBJECT: [
                r'\b(order|product|customer|invoice|payment|account|profile|document|report)\b',
                r'\b(item|record|entry|transaction|booking|reservation|appointment)\b',
                r'\b(contract|agreement|policy|rule|regulation)\b'
            ],
            EntityType.SYSTEM_COMPONENT: [
                r'\b(database|server|api|service|module|component|interface|gateway)\b',
                r'\b(microservice|endpoint|controller|repository|model|view)\b',
                r'\b(authentication|authorization|logging|monitoring|caching)\b'
            ],
            EntityType.DATA_ENTITY: [
                r'\b(table|collection|schema|field|column|attribute|property)\b',
                r'\b(data|information|metadata|dataset|record|entity)\b',
                r'\b(json|xml|csv|file|document|blob)\b'
            ],
            EntityType.TECHNOLOGY: [
                r'\b(react|angular|vue|node|python|java|javascript|typescript)\b',
                r'\b(mysql|postgresql|mongodb|redis|elasticsearch|kafka)\b',
                r'\b(aws|azure|gcp|docker|kubernetes|jenkins|github)\b'
            ],
            EntityType.FEATURE: [
                r'\b(feature|functionality|capability|function|operation|action)\b',
                r'\b(search|filter|sort|export|import|upload|download)\b',
                r'\b(notification|alert|reminder|email|sms|push)\b'
            ],
            EntityType.CONSTRAINT: [
                r'\b(constraint|limitation|restriction|requirement|rule|policy)\b',
                r'\b(must|should|shall|cannot|forbidden|prohibited)\b',
                r'\b(maximum|minimum|limit|threshold|boundary)\b'
            ],
            EntityType.METRIC: [
                r'\b(metric|measurement|kpi|performance|speed|latency|throughput)\b',
                r'\b(count|number|quantity|amount|percentage|ratio|rate)\b',
                r'\b(time|duration|frequency|interval|period)\b'
            ],
            EntityType.INTEGRATION_POINT: [
                r'\b(integration|connector|webhook|api\s+call|external\s+service)\b',
                r'\b(third.?party|external|remote|cloud\s+service)\b',
                r'\b(sync|synchronization|import|export|feed|stream)\b'
            ]
        }
        
        # Relationship patterns
        self.relationship_patterns = {
            'has_many': [r'(\w+)\s+has\s+(many|multiple)\s+(\w+)', r'(\w+)\s+contains\s+(many|multiple)\s+(\w+)'],
            'belongs_to': [r'(\w+)\s+belongs\s+to\s+(\w+)', r'(\w+)\s+is\s+part\s+of\s+(\w+)'],
            'uses': [r'(\w+)\s+uses\s+(\w+)', r'(\w+)\s+utilizes\s+(\w+)', r'(\w+)\s+depends\s+on\s+(\w+)'],
            'manages': [r'(\w+)\s+manages\s+(\w+)', r'(\w+)\s+controls\s+(\w+)', r'(\w+)\s+handles\s+(\w+)'],
            'connects_to': [r'(\w+)\s+connects\s+to\s+(\w+)', r'(\w+)\s+integrates\s+with\s+(\w+)'],
            'inherits_from': [r'(\w+)\s+inherits\s+from\s+(\w+)', r'(\w+)\s+extends\s+(\w+)'],
            'implements': [r'(\w+)\s+implements\s+(\w+)', r'(\w+)\s+realizes\s+(\w+)'],
            'processes': [r'(\w+)\s+processes\s+(\w+)', r'(\w+)\s+handles\s+(\w+)\s+data']
        }
        
        # System prompt for AI-powered extraction
        self.extraction_prompt = """
        You are an expert business analyst and software architect. Extract entities and their relationships from the following requirements text.
        
        Entity Types to identify:
        - USER_ROLE: Users, roles, actors, personas
        - BUSINESS_OBJECT: Business domain objects, concepts, items
        - SYSTEM_COMPONENT: Technical components, services, modules
        - DATA_ENTITY: Data structures, tables, schemas, fields
        - TECHNOLOGY: Technologies, frameworks, tools, platforms
        - FEATURE: Features, functionalities, capabilities
        - CONSTRAINT: Constraints, limitations, rules, policies
        - METRIC: Metrics, measurements, KPIs, performance indicators
        - INTEGRATION_POINT: External integrations, APIs, connectors
        
        For each entity, provide:
        - name: Clear, concise name
        - type: One of the entity types above
        - description: Brief description of the entity
        - attributes: Key properties or characteristics
        - confidence: Confidence score (0.0 to 1.0)
        
        For relationships, provide:
        - source: Source entity name
        - target: Target entity name
        - type: Relationship type (has_many, belongs_to, uses, manages, etc.)
        - description: Description of the relationship
        - confidence: Confidence score (0.0 to 1.0)
        
        Return as JSON with 'entities' and 'relationships' arrays.
        """

    async def extract_entities(self, text: str) -> EntityExtractionResult:
        """
        Extract entities and relationships from the given text.
        
        Args:
            text: Natural language requirements text
            
        Returns:
            EntityExtractionResult with extracted entities and relationships
        """
        import time
        start_time = time.time()
        
        try:
            # Extract using multiple approaches
            rule_based_entities = self._extract_with_rules(text)
            spacy_entities = self._extract_with_spacy(text) if self.nlp else []
            ai_entities, ai_relationships = await self._extract_with_ai(text)
            
            # Merge and deduplicate entities
            all_entities = self._merge_entities(rule_based_entities, spacy_entities, ai_entities)
            
            # Extract relationships using rules
            rule_based_relationships = self._extract_relationships_with_rules(text, all_entities)
            
            # Merge relationships
            all_relationships = self._merge_relationships(rule_based_relationships, ai_relationships)
            
            # Calculate overall confidence
            confidence_score = self._calculate_confidence(all_entities, all_relationships)
            
            # Generate warnings
            warnings = self._generate_warnings(all_entities, all_relationships)
            
            processing_time = time.time() - start_time
            
            return EntityExtractionResult(
                entities=all_entities,
                relationships=all_relationships,
                processing_time=processing_time,
                confidence_score=confidence_score,
                warnings=warnings
            )
            
        except Exception as e:
            logger.error(f"Error extracting entities: {str(e)}")
            return EntityExtractionResult(
                entities=[],
                relationships=[],
                processing_time=time.time() - start_time,
                confidence_score=0.0,
                warnings=[f"Extraction failed: {str(e)}"]
            )

    def _extract_with_rules(self, text: str) -> List[ExtractedEntity]:
        """Extract entities using rule-based patterns."""
        entities = []
        text_lower = text.lower()
        
        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text_lower)
                for match in matches:
                    entity_name = match.group(1) if match.groups() else match.group(0)
                    entity_name = entity_name.strip()
                    
                    if len(entity_name) > 2:  # Filter out very short matches
                        entities.append(ExtractedEntity(
                            name=entity_name,
                            entity_type=entity_type,
                            description=f"{entity_type.value} identified by pattern matching",
                            attributes={},
                            confidence=0.7,
                            mentions=[entity_name],
                            context=self._get_context(text, match.start(), match.end())
                        ))
        
        return self._deduplicate_entities(entities)

    def _extract_with_spacy(self, text: str) -> List[ExtractedEntity]:
        """Extract entities using spaCy NER."""
        if not self.nlp:
            return []
        
        entities = []
        doc = self.nlp(text)
        
        for ent in doc.ents:
            entity_type = self._map_spacy_label_to_entity_type(ent.label_)
            if entity_type:
                entities.append(ExtractedEntity(
                    name=ent.text,
                    entity_type=entity_type,
                    description=f"{entity_type.value} identified by spaCy NER",
                    attributes={'spacy_label': ent.label_},
                    confidence=0.8,
                    mentions=[ent.text],
                    context=self._get_context(text, ent.start_char, ent.end_char)
                ))
        
        return entities

    def _map_spacy_label_to_entity_type(self, spacy_label: str) -> Optional[EntityType]:
        """Map spaCy entity labels to our entity types."""
        mapping = {
            'PERSON': EntityType.USER_ROLE,
            'ORG': EntityType.BUSINESS_OBJECT,
            'PRODUCT': EntityType.BUSINESS_OBJECT,
            'EVENT': EntityType.FEATURE,
            'WORK_OF_ART': EntityType.SYSTEM_COMPONENT,
            'LAW': EntityType.CONSTRAINT,
            'LANGUAGE': EntityType.TECHNOLOGY,
            'PERCENT': EntityType.METRIC,
            'MONEY': EntityType.METRIC,
            'QUANTITY': EntityType.METRIC,
            'ORDINAL': EntityType.METRIC,
            'CARDINAL': EntityType.METRIC
        }
        return mapping.get(spacy_label)

    async def _extract_with_ai(self, text: str) -> Tuple[List[ExtractedEntity], List[ExtractedRelationship]]:
        """Extract entities and relationships using AI (GPT-4)."""
        try:
            response = await self.client.chat.completions.acreate(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": self.extraction_prompt},
                    {"role": "user", "content": text}
                ],
                temperature=0.1,
                max_tokens=2000
            )
            
            content = response.choices[0].message.content
            data = self._parse_ai_response(content)
            
            entities = []
            for entity_data in data.get('entities', []):
                try:
                    entity_type = EntityType(entity_data.get('type', 'BUSINESS_OBJECT'))
                    entities.append(ExtractedEntity(
                        name=entity_data.get('name', ''),
                        entity_type=entity_type,
                        description=entity_data.get('description', ''),
                        attributes=entity_data.get('attributes', {}),
                        confidence=float(entity_data.get('confidence', 0.8)),
                        mentions=[entity_data.get('name', '')],
                        context=text[:200]  # First 200 chars as context
                    ))
                except (ValueError, KeyError) as e:
                    logger.warning(f"Error parsing AI entity: {str(e)}")
                    continue
            
            relationships = []
            for rel_data in data.get('relationships', []):
                try:
                    relationships.append(ExtractedRelationship(
                        source_entity=rel_data.get('source', ''),
                        target_entity=rel_data.get('target', ''),
                        relationship_type=rel_data.get('type', 'related_to'),
                        description=rel_data.get('description', ''),
                        confidence=float(rel_data.get('confidence', 0.8)),
                        context=text[:200]
                    ))
                except (ValueError, KeyError) as e:
                    logger.warning(f"Error parsing AI relationship: {str(e)}")
                    continue
            
            return entities, relationships
            
        except Exception as e:
            logger.error(f"Error in AI extraction: {str(e)}")
            return [], []

    def _parse_ai_response(self, content: str) -> Dict[str, Any]:
        """Parse AI response content into structured data."""
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # Try to extract JSON from text
            json_pattern = r'\{.*\}'
            matches = re.findall(json_pattern, content, re.DOTALL)
            
            for match in matches:
                try:
                    return json.loads(match)
                except json.JSONDecodeError:
                    continue
            
            return {'entities': [], 'relationships': []}

    def _extract_relationships_with_rules(
        self, text: str, entities: List[ExtractedEntity]
    ) -> List[ExtractedRelationship]:
        """Extract relationships using rule-based patterns."""
        relationships = []
        text_lower = text.lower()
        entity_names = {entity.name.lower(): entity.name for entity in entities}
        
        for rel_type, patterns in self.relationship_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text_lower)
                for match in matches:
                    groups = match.groups()
                    if len(groups) >= 2:
                        source = groups[0].strip()
                        target = groups[-1].strip()  # Last group for patterns with 3 groups
                        
                        # Check if entities exist
                        if source in entity_names and target in entity_names:
                            relationships.append(ExtractedRelationship(
                                source_entity=entity_names[source],
                                target_entity=entity_names[target],
                                relationship_type=rel_type,
                                description=f"{rel_type} relationship identified by pattern",
                                confidence=0.7,
                                context=self._get_context(text, match.start(), match.end())
                            ))
        
        return relationships

    def _merge_entities(self, *entity_lists: List[ExtractedEntity]) -> List[ExtractedEntity]:
        """Merge entities from multiple extraction methods."""
        entity_map = {}
        
        for entity_list in entity_lists:
            for entity in entity_list:
                key = entity.name.lower()
                
                if key in entity_map:
                    # Merge with existing entity
                    existing = entity_map[key]
                    existing.confidence = max(existing.confidence, entity.confidence)
                    existing.mentions.extend(entity.mentions)
                    existing.attributes.update(entity.attributes)
                    
                    # Use more specific description if available
                    if len(entity.description) > len(existing.description):
                        existing.description = entity.description
                else:
                    entity_map[key] = entity
        
        return list(entity_map.values())

    def _merge_relationships(self, *relationship_lists: List[ExtractedRelationship]) -> List[ExtractedRelationship]:
        """Merge relationships from multiple extraction methods."""
        relationship_map = {}
        
        for relationship_list in relationship_lists:
            for relationship in relationship_list:
                key = f"{relationship.source_entity.lower()}_{relationship.target_entity.lower()}_{relationship.relationship_type}"
                
                if key in relationship_map:
                    # Update confidence to maximum
                    existing = relationship_map[key]
                    existing.confidence = max(existing.confidence, relationship.confidence)
                else:
                    relationship_map[key] = relationship
        
        return list(relationship_map.values())

    def _deduplicate_entities(self, entities: List[ExtractedEntity]) -> List[ExtractedEntity]:
        """Remove duplicate entities based on name similarity."""
        unique_entities = []
        seen_names = set()
        
        for entity in entities:
            name_lower = entity.name.lower()
            if name_lower not in seen_names:
                unique_entities.append(entity)
                seen_names.add(name_lower)
        
        return unique_entities

    def _get_context(self, text: str, start: int, end: int, window: int = 50) -> str:
        """Get context around a text match."""
        context_start = max(0, start - window)
        context_end = min(len(text), end + window)
        return text[context_start:context_end].strip()

    def _calculate_confidence(
        self, entities: List[ExtractedEntity], relationships: List[ExtractedRelationship]
    ) -> float:
        """Calculate overall confidence score."""
        all_scores = []
        
        for entity in entities:
            all_scores.append(entity.confidence)
        
        for relationship in relationships:
            all_scores.append(relationship.confidence)
        
        return sum(all_scores) / len(all_scores) if all_scores else 0.0

    def _generate_warnings(
        self, entities: List[ExtractedEntity], relationships: List[ExtractedRelationship]
    ) -> List[str]:
        """Generate warnings based on extraction results."""
        warnings = []
        
        if not entities:
            warnings.append("No entities extracted - requirements may be too vague")
        
        if len(entities) < 3:
            warnings.append("Few entities extracted - consider providing more detailed requirements")
        
        if not relationships:
            warnings.append("No relationships identified - entity interactions may be unclear")
        
        low_confidence_entities = [e for e in entities if e.confidence < 0.6]
        if low_confidence_entities:
            warnings.append(f"{len(low_confidence_entities)} entities have low confidence scores")
        
        return warnings

    def convert_to_models(
        self, extraction_result: EntityExtractionResult
    ) -> Tuple[List[EntityModel], List[EntityRelationshipModel]]:
        """Convert extraction results to Pydantic models."""
        entity_models = []
        entity_name_to_id = {}
        
        # Convert entities
        for i, entity in enumerate(extraction_result.entities):
            entity_model = EntityModel(
                id=i + 1,  # Temporary ID
                entity_type=entity.entity_type,
                name=entity.name,
                description=entity.description,
                attributes=entity.attributes,
                confidence_score=entity.confidence
            )
            entity_models.append(entity_model)
            entity_name_to_id[entity.name] = i + 1
        
        # Convert relationships
        relationship_models = []
        for relationship in extraction_result.relationships:
            if (relationship.source_entity in entity_name_to_id and 
                relationship.target_entity in entity_name_to_id):
                
                relationship_model = EntityRelationshipModel(
                    source_entity_id=entity_name_to_id[relationship.source_entity],
                    target_entity_id=entity_name_to_id[relationship.target_entity],
                    relationship_type=relationship.relationship_type,
                    description=relationship.description,
                    confidence_score=relationship.confidence
                )
                relationship_models.append(relationship_model)
        
        return entity_models, relationship_models