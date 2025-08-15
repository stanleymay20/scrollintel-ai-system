"""
Natural Language Processing engine for automated code generation.
Provides advanced language understanding using GPT-4 for requirements analysis.
"""
import json
import re
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import openai
from openai import OpenAI

from scrollintel.models.code_generation_models import (
    Requirements, ParsedRequirement, Entity, Relationship, Clarification,
    RequirementType, Intent, EntityType, ConfidenceLevel, ProcessingResult
)
from scrollintel.core.config import get_settings


class NLProcessor:
    """Advanced Natural Language Processor for code generation requirements."""
    
    def __init__(self):
        """Initialize the NL processor with GPT-4 client."""
        self.settings = get_settings()
        self.client = OpenAI(api_key=self.settings.openai_api_key)
        self.model = "gpt-4-turbo-preview"
        
    def parse_requirements(self, text: str, project_name: str = "Untitled Project") -> ProcessingResult:
        """
        Parse natural language requirements into structured format.
        
        Args:
            text: Raw requirements text
            project_name: Name of the project
            
        Returns:
            ProcessingResult with parsed requirements
        """
        start_time = datetime.utcnow()
        
        try:
            # Clean and preprocess text
            cleaned_text = self._preprocess_text(text)
            
            # Extract structured requirements using GPT-4
            parsed_reqs = self._extract_requirements(cleaned_text)
            
            # Extract entities and relationships
            entities = self._extract_entities(cleaned_text)
            relationships = self._extract_relationships(entities, cleaned_text)
            
            # Generate clarifications for ambiguous parts
            clarifications = self._generate_clarifications(parsed_reqs, entities)
            
            # Calculate completeness score
            completeness_score = self._calculate_completeness(parsed_reqs, entities)
            
            # Validate requirements
            validation_errors = self._validate_requirements(parsed_reqs)
            
            # Create requirements container
            requirements = Requirements(
                id=str(uuid.uuid4()),
                project_name=project_name,
                raw_text=text,
                parsed_requirements=parsed_reqs,
                entities=entities,
                relationships=relationships,
                clarifications=clarifications,
                completeness_score=completeness_score,
                validation_errors=validation_errors,
                processing_status="completed"
            )
            
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            return ProcessingResult(
                success=True,
                requirements=requirements,
                processing_time=processing_time,
                tokens_used=self._estimate_tokens(text)
            )
            
        except Exception as e:
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            return ProcessingResult(
                success=False,
                errors=[str(e)],
                processing_time=processing_time,
                tokens_used=self._estimate_tokens(text)
            )
    
    def _preprocess_text(self, text: str) -> str:
        """Clean and preprocess the input text."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Fix common formatting issues
        text = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', text)
        
        # Normalize bullet points
        text = re.sub(r'[•·▪▫‣⁃]\s*', '- ', text)
        
        return text
    
    def _extract_requirements(self, text: str) -> List[ParsedRequirement]:
        """Extract structured requirements using GPT-4."""
        prompt = f"""
        Analyze the following requirements text and extract individual requirements.
        For each requirement, provide:
        1. The original text
        2. A structured version
        3. The requirement type (functional, non_functional, business, technical, ui_ux, data, integration, security, performance)
        4. The primary intent (create_application, modify_feature, add_integration, etc.)
        5. Acceptance criteria
        6. Dependencies
        7. Priority (1-5, where 1 is highest)
        8. Complexity (1-5, where 1 is simplest)
        9. Confidence level (high, medium, low)
        
        Requirements text:
        {text}
        
        Return the response as a JSON array of requirements objects.
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert requirements analyst. Extract and structure requirements with high precision."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            
            content = response.choices[0].message.content
            requirements_data = json.loads(content)
            
            parsed_requirements = []
            for req_data in requirements_data:
                parsed_req = ParsedRequirement(
                    id=str(uuid.uuid4()),
                    original_text=req_data.get('original_text', ''),
                    structured_text=req_data.get('structured_text', ''),
                    requirement_type=RequirementType(req_data.get('requirement_type', 'functional')),
                    intent=Intent(req_data.get('intent', 'create_application')),
                    acceptance_criteria=req_data.get('acceptance_criteria', []),
                    dependencies=req_data.get('dependencies', []),
                    priority=req_data.get('priority', 3),
                    complexity=req_data.get('complexity', 3),
                    confidence=ConfidenceLevel(req_data.get('confidence', 'medium'))
                )
                parsed_requirements.append(parsed_req)
            
            return parsed_requirements
            
        except Exception as e:
            # Fallback to basic parsing if GPT-4 fails
            return self._fallback_requirement_parsing(text)
    
    def _extract_entities(self, text: str) -> List[Entity]:
        """Extract entities from the requirements text."""
        prompt = f"""
        Extract all relevant entities from the following requirements text.
        For each entity, provide:
        1. Name
        2. Type (user_role, feature, data_entity, system_component, action, condition, constraint, technology)
        3. Description
        4. Confidence score (0.0-1.0)
        5. Source text where it was found
        6. Position in text (start, end)
        
        Requirements text:
        {text}
        
        Return as JSON array of entity objects.
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert at entity extraction from technical requirements."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            
            content = response.choices[0].message.content
            entities_data = json.loads(content)
            
            entities = []
            for entity_data in entities_data:
                entity = Entity(
                    id=str(uuid.uuid4()),
                    name=entity_data.get('name', ''),
                    type=EntityType(entity_data.get('type', 'feature')),
                    description=entity_data.get('description'),
                    confidence=entity_data.get('confidence', 0.8),
                    source_text=entity_data.get('source_text', ''),
                    position=(entity_data.get('start_pos', 0), entity_data.get('end_pos', 0))
                )
                entities.append(entity)
            
            return entities
            
        except Exception as e:
            # Fallback to basic entity extraction
            return self._fallback_entity_extraction(text)
    
    def _extract_relationships(self, entities: List[Entity], text: str) -> List[Relationship]:
        """Extract relationships between entities."""
        if len(entities) < 2:
            return []
        
        entity_names = [e.name for e in entities]
        entity_map = {e.name: e.id for e in entities}
        
        prompt = f"""
        Given these entities: {entity_names}
        And this requirements text: {text}
        
        Identify relationships between entities. For each relationship provide:
        1. Source entity name
        2. Target entity name  
        3. Relationship type (depends_on, contains, uses, implements, etc.)
        4. Description
        5. Confidence score (0.0-1.0)
        
        Return as JSON array of relationship objects.
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert at identifying relationships in technical requirements."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            
            content = response.choices[0].message.content
            relationships_data = json.loads(content)
            
            relationships = []
            for rel_data in relationships_data:
                source_name = rel_data.get('source_entity')
                target_name = rel_data.get('target_entity')
                
                if source_name in entity_map and target_name in entity_map:
                    relationship = Relationship(
                        id=str(uuid.uuid4()),
                        source_entity_id=entity_map[source_name],
                        target_entity_id=entity_map[target_name],
                        relationship_type=rel_data.get('relationship_type', 'related_to'),
                        description=rel_data.get('description'),
                        confidence=rel_data.get('confidence', 0.7)
                    )
                    relationships.append(relationship)
            
            return relationships
            
        except Exception:
            return []
    
    def _generate_clarifications(self, requirements: List[ParsedRequirement], entities: List[Entity]) -> List[Clarification]:
        """Generate clarification questions for ambiguous requirements."""
        clarifications = []
        
        # Check for low confidence requirements
        for req in requirements:
            if req.confidence == ConfidenceLevel.LOW:
                clarification = Clarification(
                    id=str(uuid.uuid4()),
                    question=f"Could you provide more details about: {req.structured_text}?",
                    context=req.original_text,
                    priority=2,
                    requirement_id=req.id
                )
                clarifications.append(clarification)
        
        # Check for missing critical information
        has_user_roles = any(e.type == EntityType.USER_ROLE for e in entities)
        if not has_user_roles:
            clarifications.append(Clarification(
                id=str(uuid.uuid4()),
                question="Who are the primary users of this system?",
                context="No user roles were identified in the requirements",
                priority=1
            ))
        
        return clarifications
    
    def _calculate_completeness(self, requirements: List[ParsedRequirement], entities: List[Entity]) -> float:
        """Calculate completeness score for requirements."""
        score = 0.0
        total_checks = 6
        
        # Check for functional requirements
        if any(req.requirement_type == RequirementType.FUNCTIONAL for req in requirements):
            score += 1
        
        # Check for user roles
        if any(e.type == EntityType.USER_ROLE for e in entities):
            score += 1
        
        # Check for data entities
        if any(e.type == EntityType.DATA_ENTITY for e in entities):
            score += 1
        
        # Check for system components
        if any(e.type == EntityType.SYSTEM_COMPONENT for e in entities):
            score += 1
        
        # Check for acceptance criteria
        if any(req.acceptance_criteria for req in requirements):
            score += 1
        
        # Check for non-functional requirements
        if any(req.requirement_type in [RequirementType.PERFORMANCE, RequirementType.SECURITY] for req in requirements):
            score += 1
        
        return score / total_checks
    
    def _validate_requirements(self, requirements: List[ParsedRequirement]) -> List[str]:
        """Validate requirements for common issues."""
        errors = []
        
        if not requirements:
            errors.append("No requirements were extracted from the input text")
        
        for req in requirements:
            if not req.structured_text.strip():
                errors.append(f"Requirement {req.id} has empty structured text")
            
            if not req.acceptance_criteria:
                errors.append(f"Requirement {req.id} lacks acceptance criteria")
        
        return errors
    
    def _fallback_requirement_parsing(self, text: str) -> List[ParsedRequirement]:
        """Fallback parsing when GPT-4 is unavailable."""
        sentences = re.split(r'[.!?]+', text)
        requirements = []
        
        for i, sentence in enumerate(sentences):
            if sentence.strip():
                req = ParsedRequirement(
                    id=str(uuid.uuid4()),
                    original_text=sentence.strip(),
                    structured_text=sentence.strip(),
                    requirement_type=RequirementType.FUNCTIONAL,
                    intent=Intent.CREATE_APPLICATION,
                    confidence=ConfidenceLevel.LOW
                )
                requirements.append(req)
        
        return requirements
    
    def _fallback_entity_extraction(self, text: str) -> List[Entity]:
        """Fallback entity extraction using simple patterns."""
        entities = []
        
        # Extract potential user roles
        user_patterns = r'\b(?:user|admin|customer|manager|operator)\b'
        for match in re.finditer(user_patterns, text, re.IGNORECASE):
            entity = Entity(
                id=str(uuid.uuid4()),
                name=match.group(),
                type=EntityType.USER_ROLE,
                confidence=0.6,
                source_text=match.group(),
                position=(match.start(), match.end())
            )
            entities.append(entity)
        
        return entities
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for the text."""
        # Rough estimation: ~4 characters per token
        return len(text) // 4