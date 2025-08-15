"""
Entity extraction system for automated code generation.
Extracts and classifies entities from requirements text.
"""
import json
import re
import uuid
from typing import List, Dict, Any, Optional, Tuple, Set
from openai import OpenAI

from scrollintel.models.code_generation_models import Entity, EntityType, Relationship
from scrollintel.core.config import get_settings


class EntityExtractor:
    """Extracts entities from requirements text for code generation."""
    
    def __init__(self):
        """Initialize the entity extractor."""
        self.settings = get_settings()
        self.client = OpenAI(api_key=self.settings.openai_api_key)
        self.model = "gpt-4-turbo-preview"
        
        # Entity patterns for fallback extraction
        self.entity_patterns = {
            EntityType.USER_ROLE: [
                r'\b(?:user|admin|administrator|customer|client|manager|operator|developer|analyst|owner|stakeholder)s?\b',
                r'\b(?:end[- ]user|super[- ]user|power[- ]user)\b',
                r'\bas\s+an?\s+([a-zA-Z\s]+?)(?:,|\s+I\s+want|\s+should)\b'
            ],
            EntityType.FEATURE: [
                r'\b(?:feature|functionality|capability|function|module|component|service)\b',
                r'\b(?:login|authentication|registration|dashboard|report|search|filter|export|import)\b',
                r'\b(?:notification|alert|messaging|chat|email|sms)\b'
            ],
            EntityType.DATA_ENTITY: [
                r'\b(?:user|customer|product|order|invoice|payment|account|profile|record|data|information)\b',
                r'\b(?:database|table|entity|model|schema|field|attribute|property)\b',
                r'\b(?:file|document|image|video|attachment|upload)\b'
            ],
            EntityType.SYSTEM_COMPONENT: [
                r'\b(?:api|endpoint|service|microservice|database|server|client|frontend|backend)\b',
                r'\b(?:web\s+application|mobile\s+app|desktop\s+application)\b',
                r'\b(?:authentication\s+system|payment\s+gateway|notification\s+service)\b'
            ],
            EntityType.ACTION: [
                r'\b(?:create|read|update|delete|add|remove|edit|modify|save|load|send|receive)\b',
                r'\b(?:login|logout|register|authenticate|authorize|validate|verify|process|generate)\b',
                r'\b(?:search|filter|sort|export|import|upload|download|share|publish)\b'
            ],
            EntityType.CONDITION: [
                r'\bif\s+([^,]+?)(?:,|\s+then)\b',
                r'\bwhen\s+([^,]+?)(?:,|\s+then)\b',
                r'\bunless\s+([^,]+?)(?:,|\s+then)\b',
                r'\bprovided\s+that\s+([^,]+?)(?:,|\s+then)\b'
            ],
            EntityType.CONSTRAINT: [
                r'\b(?:must|should|shall|required|mandatory|optional|forbidden|prohibited)\b',
                r'\b(?:within\s+\d+\s+(?:seconds|minutes|hours|days))\b',
                r'\b(?:maximum|minimum|at\s+least|no\s+more\s+than)\s+\d+\b',
                r'\b(?:secure|encrypted|authenticated|authorized|validated)\b'
            ],
            EntityType.TECHNOLOGY: [
                r'\b(?:react|angular|vue|javascript|typescript|python|java|c#|php|ruby)\b',
                r'\b(?:mysql|postgresql|mongodb|redis|elasticsearch|sqlite)\b',
                r'\b(?:aws|azure|gcp|docker|kubernetes|jenkins|github)\b',
                r'\b(?:rest|graphql|soap|json|xml|html|css|bootstrap|tailwind)\b'
            ]
        }
    
    def extract_entities(self, text: str) -> List[Entity]:
        """
        Extract entities from requirements text.
        
        Args:
            text: The requirements text to analyze
            
        Returns:
            List of extracted entities
        """
        try:
            # Try GPT-4 extraction first
            return self._extract_with_gpt4(text)
        except Exception:
            # Fallback to pattern-based extraction
            return self._extract_with_patterns(text)
    
    def extract_relationships(self, entities: List[Entity], text: str) -> List[Relationship]:
        """
        Extract relationships between entities.
        
        Args:
            entities: List of extracted entities
            text: Original requirements text
            
        Returns:
            List of relationships between entities
        """
        if len(entities) < 2:
            return []
        
        try:
            return self._extract_relationships_with_gpt4(entities, text)
        except Exception:
            return self._extract_relationships_with_patterns(entities, text)
    
    def validate_entities(self, entities: List[Entity]) -> List[str]:
        """
        Validate extracted entities for completeness and accuracy.
        
        Args:
            entities: List of entities to validate
            
        Returns:
            List of validation warnings/errors
        """
        warnings = []
        
        # Check for essential entity types
        entity_types = {e.type for e in entities}
        
        if EntityType.USER_ROLE not in entity_types:
            warnings.append("No user roles identified - consider specifying who will use the system")
        
        if EntityType.DATA_ENTITY not in entity_types:
            warnings.append("No data entities identified - consider specifying what data the system will handle")
        
        if EntityType.ACTION not in entity_types:
            warnings.append("No actions identified - consider specifying what users can do")
        
        # Check for duplicate entities
        entity_names = [e.name.lower() for e in entities]
        duplicates = set([name for name in entity_names if entity_names.count(name) > 1])
        if duplicates:
            warnings.append(f"Duplicate entities found: {', '.join(duplicates)}")
        
        # Check for very low confidence entities
        low_confidence = [e.name for e in entities if e.confidence < 0.3]
        if low_confidence:
            warnings.append(f"Low confidence entities (may need clarification): {', '.join(low_confidence)}")
        
        return warnings
    
    def group_entities_by_domain(self, entities: List[Entity]) -> Dict[str, List[Entity]]:
        """
        Group entities by domain/context for better organization.
        
        Args:
            entities: List of entities to group
            
        Returns:
            Dictionary mapping domain names to entity lists
        """
        domains = {
            "User Management": [],
            "Data Management": [],
            "System Components": [],
            "Business Logic": [],
            "Technical Infrastructure": [],
            "Security": [],
            "Other": []
        }
        
        for entity in entities:
            if entity.type == EntityType.USER_ROLE:
                domains["User Management"].append(entity)
            elif entity.type == EntityType.DATA_ENTITY:
                domains["Data Management"].append(entity)
            elif entity.type == EntityType.SYSTEM_COMPONENT:
                domains["System Components"].append(entity)
            elif entity.type in [EntityType.ACTION, EntityType.FEATURE]:
                domains["Business Logic"].append(entity)
            elif entity.type == EntityType.TECHNOLOGY:
                domains["Technical Infrastructure"].append(entity)
            elif "security" in entity.name.lower() or "auth" in entity.name.lower():
                domains["Security"].append(entity)
            else:
                domains["Other"].append(entity)
        
        # Remove empty domains
        return {k: v for k, v in domains.items() if v}
    
    def _extract_with_gpt4(self, text: str) -> List[Entity]:
        """Extract entities using GPT-4."""
        prompt = f"""
        Extract all relevant entities from the following requirements text.
        For each entity, provide:
        1. name: The entity name
        2. type: One of (user_role, feature, data_entity, system_component, action, condition, constraint, technology)
        3. description: Brief description of the entity
        4. confidence: Confidence score from 0.0 to 1.0
        5. source_text: The exact text where this entity was mentioned
        6. attributes: Any additional attributes as key-value pairs
        
        Requirements text:
        {text}
        
        Return as a JSON array of entity objects.
        Focus on entities that are relevant for code generation and system design.
        """
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an expert at extracting entities from software requirements for code generation."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )
        
        content = response.choices[0].message.content
        entities_data = json.loads(content)
        
        entities = []
        for i, entity_data in enumerate(entities_data):
            # Find position in text
            source_text = entity_data.get('source_text', entity_data.get('name', ''))
            start_pos = text.lower().find(source_text.lower())
            end_pos = start_pos + len(source_text) if start_pos >= 0 else 0
            
            entity = Entity(
                id=str(uuid.uuid4()),
                name=entity_data.get('name', f'Entity_{i}'),
                type=EntityType(entity_data.get('type', 'feature')),
                description=entity_data.get('description'),
                confidence=entity_data.get('confidence', 0.8),
                source_text=source_text,
                position=(max(0, start_pos), max(0, end_pos)),
                attributes=entity_data.get('attributes', {})
            )
            entities.append(entity)
        
        return entities
    
    def _extract_with_patterns(self, text: str) -> List[Entity]:
        """Extract entities using pattern matching."""
        entities = []
        text_lower = text.lower()
        
        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text_lower, re.IGNORECASE)
                for match in matches:
                    # Extract the entity name
                    if match.groups():
                        name = match.group(1).strip()
                    else:
                        name = match.group(0).strip()
                    
                    # Skip very short or common words
                    if len(name) < 2 or name in ['a', 'an', 'the', 'and', 'or', 'but']:
                        continue
                    
                    entity = Entity(
                        id=str(uuid.uuid4()),
                        name=name,
                        type=entity_type,
                        confidence=0.6,  # Lower confidence for pattern matching
                        source_text=match.group(0),
                        position=(match.start(), match.end())
                    )
                    entities.append(entity)
        
        # Remove duplicates based on name and type
        unique_entities = []
        seen = set()
        for entity in entities:
            key = (entity.name.lower(), entity.type)
            if key not in seen:
                seen.add(key)
                unique_entities.append(entity)
        
        return unique_entities
    
    def _extract_relationships_with_gpt4(self, entities: List[Entity], text: str) -> List[Relationship]:
        """Extract relationships using GPT-4."""
        entity_info = [{"name": e.name, "type": e.type.value, "id": e.id} for e in entities]
        
        prompt = f"""
        Given these entities extracted from requirements:
        {json.dumps(entity_info, indent=2)}
        
        And this requirements text:
        {text}
        
        Identify relationships between entities. For each relationship provide:
        1. source_entity_id: ID of the source entity
        2. target_entity_id: ID of the target entity
        3. relationship_type: Type of relationship (uses, contains, depends_on, implements, manages, processes, etc.)
        4. description: Brief description of the relationship
        5. confidence: Confidence score from 0.0 to 1.0
        
        Return as JSON array of relationship objects.
        Only include relationships that are clearly indicated in the text.
        """
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an expert at identifying relationships between entities in software requirements."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )
        
        content = response.choices[0].message.content
        relationships_data = json.loads(content)
        
        relationships = []
        entity_ids = {e.id for e in entities}
        
        for rel_data in relationships_data:
            source_id = rel_data.get('source_entity_id')
            target_id = rel_data.get('target_entity_id')
            
            # Validate entity IDs exist
            if source_id in entity_ids and target_id in entity_ids and source_id != target_id:
                relationship = Relationship(
                    id=str(uuid.uuid4()),
                    source_entity_id=source_id,
                    target_entity_id=target_id,
                    relationship_type=rel_data.get('relationship_type', 'related_to'),
                    description=rel_data.get('description'),
                    confidence=rel_data.get('confidence', 0.7)
                )
                relationships.append(relationship)
        
        return relationships
    
    def _extract_relationships_with_patterns(self, entities: List[Entity], text: str) -> List[Relationship]:
        """Extract relationships using pattern matching."""
        relationships = []
        text_lower = text.lower()
        
        # Create entity name to ID mapping
        entity_map = {e.name.lower(): e.id for e in entities}
        entity_names = list(entity_map.keys())
        
        # Look for common relationship patterns
        relationship_patterns = [
            (r'(\w+)\s+(?:uses|utilizes|employs)\s+(\w+)', 'uses'),
            (r'(\w+)\s+(?:contains|includes|has)\s+(\w+)', 'contains'),
            (r'(\w+)\s+(?:depends\s+on|requires)\s+(\w+)', 'depends_on'),
            (r'(\w+)\s+(?:manages|controls|handles)\s+(\w+)', 'manages'),
            (r'(\w+)\s+(?:processes|handles|works\s+with)\s+(\w+)', 'processes'),
            (r'(\w+)\s+(?:implements|provides)\s+(\w+)', 'implements')
        ]
        
        for pattern, rel_type in relationship_patterns:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                source_name = match.group(1).strip()
                target_name = match.group(2).strip()
                
                # Check if both entities exist
                if source_name in entity_map and target_name in entity_map:
                    relationship = Relationship(
                        id=str(uuid.uuid4()),
                        source_entity_id=entity_map[source_name],
                        target_entity_id=entity_map[target_name],
                        relationship_type=rel_type,
                        confidence=0.5  # Lower confidence for pattern matching
                    )
                    relationships.append(relationship)
        
        return relationships