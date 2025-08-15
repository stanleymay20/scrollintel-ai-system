"""
Natural Language Processor for the Automated Code Generation System.
Handles parsing and understanding of natural language requirements using GPT-4.
"""

import re
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import openai
from sqlalchemy.orm import Session

from scrollintel.models.nlp_models import (
    Requirements, ParsedRequirement, Entity, EntityRelationship, Clarification,
    RequirementsModel, ParsedRequirementModel, EntityModel, ClarificationModel,
    ProcessingResult, ValidationResult, RequirementType, IntentType, EntityType,
    ConfidenceLevel
)
from scrollintel.core.config import get_settings

logger = logging.getLogger(__name__)


class NLProcessor:
    """
    Advanced Natural Language Processor for requirements analysis.
    Uses GPT-4 for sophisticated language understanding and entity extraction.
    """
    
    def __init__(self, db_session: Session):
        self.db_session = db_session
        self.settings = get_settings()
        self.client = openai.OpenAI(api_key=self.settings.openai_api_key)
        
        # System prompts for different NLP tasks
        self.requirements_prompt = """
        You are an expert business analyst and software architect. Analyze the following natural language requirements and extract structured information.
        
        Extract:
        1. Individual requirements with type, intent, description, and priority
        2. Business entities and their relationships
        3. Technical constraints and non-functional requirements
        4. User roles and their interactions
        
        Return a JSON structure with parsed requirements, entities, and relationships.
        Be precise and comprehensive in your analysis.
        """
        
        self.clarification_prompt = """
        You are a requirements analyst. Review the following requirements and identify areas that need clarification.
        
        Generate specific, actionable questions that would help clarify:
        1. Ambiguous or vague requirements
        2. Missing technical details
        3. Unclear business rules
        4. Incomplete user stories
        
        Return a JSON list of clarification questions with priority levels.
        """
        
        self.validation_prompt = """
        You are a quality assurance expert for requirements analysis. Evaluate the completeness and quality of these requirements.
        
        Assess:
        1. Completeness of functional requirements
        2. Presence of non-functional requirements
        3. Clarity and specificity
        4. Testability and measurability
        5. Consistency and coherence
        
        Return a JSON structure with validation results, completeness score, and improvement suggestions.
        """

    async def parse_requirements(self, text: str) -> ProcessingResult:
        """
        Parse natural language requirements into structured format.
        
        Args:
            text: Raw requirements text
            
        Returns:
            ProcessingResult with parsed requirements and metadata
        """
        start_time = datetime.utcnow()
        
        try:
            # Create requirements record
            requirements = Requirements(
                raw_text=text,
                processed_at=start_time
            )
            self.db_session.add(requirements)
            self.db_session.flush()  # Get ID without committing
            
            # Process with GPT-4
            parsed_data = await self._process_with_gpt4(text, self.requirements_prompt)
            
            # Extract and save parsed requirements
            parsed_reqs = await self._extract_parsed_requirements(
                parsed_data, requirements.id
            )
            
            # Extract and save entities
            entities = await self._extract_entities(parsed_data, requirements.id)
            
            # Extract and save relationships
            await self._extract_relationships(parsed_data, entities)
            
            # Calculate confidence and completeness
            confidence_score = self._calculate_confidence(parsed_data)
            requirements.confidence_score = confidence_score
            
            # Check if clarification is needed
            needs_clarification = await self._needs_clarification(text)
            requirements.needs_clarification = needs_clarification
            
            if needs_clarification:
                await self._generate_clarifications(text, requirements.id)
            
            # Validate completeness
            validation_result = await self._validate_requirements(text, parsed_reqs)
            requirements.is_complete = validation_result.is_valid
            
            self.db_session.commit()
            
            # Build response
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            requirements_model = RequirementsModel(
                id=requirements.id,
                raw_text=requirements.raw_text,
                processed_at=requirements.processed_at,
                confidence_score=requirements.confidence_score,
                is_complete=requirements.is_complete,
                needs_clarification=requirements.needs_clarification,
                parsed_requirements=parsed_reqs,
                entities=entities
            )
            
            confidence_level = self._get_confidence_level(confidence_score)
            
            return ProcessingResult(
                requirements=requirements_model,
                processing_time=processing_time,
                confidence_level=confidence_level,
                warnings=self._generate_warnings(parsed_data, confidence_score),
                errors=[]
            )
            
        except Exception as e:
            logger.error(f"Error processing requirements: {str(e)}")
            self.db_session.rollback()
            
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            return ProcessingResult(
                requirements=RequirementsModel(raw_text=text),
                processing_time=processing_time,
                confidence_level=ConfidenceLevel.LOW,
                warnings=[],
                errors=[f"Processing failed: {str(e)}"]
            )

    async def _process_with_gpt4(self, text: str, system_prompt: str) -> Dict[str, Any]:
        """Process text with GPT-4 using the specified system prompt."""
        try:
            response = await self.client.chat.completions.acreate(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text}
                ],
                temperature=0.1,
                max_tokens=4000
            )
            
            content = response.choices[0].message.content
            return json.loads(content)
            
        except json.JSONDecodeError:
            logger.warning("GPT-4 response was not valid JSON, attempting to extract")
            return self._extract_json_from_text(content)
        except Exception as e:
            logger.error(f"Error calling GPT-4: {str(e)}")
            return {}

    def _extract_json_from_text(self, text: str) -> Dict[str, Any]:
        """Extract JSON from text that might contain additional content."""
        json_pattern = r'\{.*\}'
        matches = re.findall(json_pattern, text, re.DOTALL)
        
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
        
        return {}

    async def _extract_parsed_requirements(
        self, parsed_data: Dict[str, Any], requirements_id: int
    ) -> List[ParsedRequirementModel]:
        """Extract and save parsed requirements from GPT-4 response."""
        parsed_reqs = []
        requirements_data = parsed_data.get('requirements', [])
        
        for req_data in requirements_data:
            try:
                req_type = RequirementType(req_data.get('type', 'functional'))
                intent = IntentType(req_data.get('intent', 'create_application'))
                
                parsed_req = ParsedRequirement(
                    requirements_id=requirements_id,
                    requirement_type=req_type.value,
                    intent=intent.value,
                    description=req_data.get('description', ''),
                    priority=req_data.get('priority', 'medium'),
                    confidence_score=req_data.get('confidence', 0.8),
                    acceptance_criteria=req_data.get('acceptance_criteria', []),
                    dependencies=req_data.get('dependencies', [])
                )
                
                self.db_session.add(parsed_req)
                
                parsed_reqs.append(ParsedRequirementModel(
                    requirement_type=req_type,
                    intent=intent,
                    description=parsed_req.description,
                    priority=parsed_req.priority,
                    confidence_score=parsed_req.confidence_score,
                    acceptance_criteria=parsed_req.acceptance_criteria,
                    dependencies=parsed_req.dependencies
                ))
                
            except (ValueError, KeyError) as e:
                logger.warning(f"Error parsing requirement: {str(e)}")
                continue
        
        return parsed_reqs

    async def _extract_entities(
        self, parsed_data: Dict[str, Any], requirements_id: int
    ) -> List[EntityModel]:
        """Extract and save entities from GPT-4 response."""
        entities = []
        entities_data = parsed_data.get('entities', [])
        
        for entity_data in entities_data:
            try:
                entity_type = EntityType(entity_data.get('type', 'business_object'))
                
                entity = Entity(
                    requirements_id=requirements_id,
                    entity_type=entity_type.value,
                    name=entity_data.get('name', ''),
                    description=entity_data.get('description', ''),
                    attributes=entity_data.get('attributes', {}),
                    confidence_score=entity_data.get('confidence', 0.8)
                )
                
                self.db_session.add(entity)
                self.db_session.flush()  # Get ID
                
                entities.append(EntityModel(
                    id=entity.id,
                    entity_type=entity_type,
                    name=entity.name,
                    description=entity.description,
                    attributes=entity.attributes,
                    confidence_score=entity.confidence_score
                ))
                
            except (ValueError, KeyError) as e:
                logger.warning(f"Error parsing entity: {str(e)}")
                continue
        
        return entities

    async def _extract_relationships(
        self, parsed_data: Dict[str, Any], entities: List[EntityModel]
    ) -> None:
        """Extract and save entity relationships from GPT-4 response."""
        relationships_data = parsed_data.get('relationships', [])
        entity_name_to_id = {entity.name: entity.id for entity in entities}
        
        for rel_data in relationships_data:
            try:
                source_name = rel_data.get('source')
                target_name = rel_data.get('target')
                
                if source_name in entity_name_to_id and target_name in entity_name_to_id:
                    relationship = EntityRelationship(
                        source_entity_id=entity_name_to_id[source_name],
                        target_entity_id=entity_name_to_id[target_name],
                        relationship_type=rel_data.get('type', 'related_to'),
                        description=rel_data.get('description', ''),
                        confidence_score=rel_data.get('confidence', 0.8)
                    )
                    
                    self.db_session.add(relationship)
                    
            except KeyError as e:
                logger.warning(f"Error parsing relationship: {str(e)}")
                continue

    async def _needs_clarification(self, text: str) -> bool:
        """Determine if requirements need clarification."""
        # Simple heuristics for now - can be enhanced with ML
        ambiguous_indicators = [
            'maybe', 'possibly', 'might', 'could', 'should probably',
            'something like', 'similar to', 'kind of', 'sort of'
        ]
        
        text_lower = text.lower()
        return any(indicator in text_lower for indicator in ambiguous_indicators)

    async def _generate_clarifications(self, text: str, requirements_id: int) -> None:
        """Generate clarification questions using GPT-4."""
        try:
            clarification_data = await self._process_with_gpt4(text, self.clarification_prompt)
            questions = clarification_data.get('questions', [])
            
            for question_data in questions:
                clarification = Clarification(
                    requirements_id=requirements_id,
                    question=question_data.get('question', ''),
                    priority=question_data.get('priority', 'medium')
                )
                self.db_session.add(clarification)
                
        except Exception as e:
            logger.error(f"Error generating clarifications: {str(e)}")
            # Add a default clarification question when generation fails
            clarification = Clarification(
                requirements_id=requirements_id,
                question="Could you provide more specific details about your requirements?",
                priority="medium"
            )
            self.db_session.add(clarification)

    async def _validate_requirements(
        self, text: str, parsed_reqs: List[ParsedRequirementModel]
    ) -> ValidationResult:
        """Validate requirements completeness and quality."""
        try:
            validation_data = await self._process_with_gpt4(text, self.validation_prompt)
            
            return ValidationResult(
                is_valid=validation_data.get('is_complete', False),
                completeness_score=validation_data.get('completeness_score', 0.0),
                missing_elements=validation_data.get('missing_elements', []),
                suggestions=validation_data.get('suggestions', []),
                confidence_score=validation_data.get('confidence', 0.0)
            )
            
        except Exception as e:
            logger.error(f"Error validating requirements: {str(e)}")
            return ValidationResult(
                is_valid=False,
                completeness_score=0.0,
                missing_elements=['Validation failed'],
                suggestions=['Please review requirements manually'],
                confidence_score=0.0
            )

    def _calculate_confidence(self, parsed_data: Dict[str, Any]) -> float:
        """Calculate overall confidence score for the parsing results."""
        scores = []
        
        # Collect confidence scores from requirements
        for req in parsed_data.get('requirements', []):
            scores.append(req.get('confidence', 0.5))
        
        # Collect confidence scores from entities
        for entity in parsed_data.get('entities', []):
            scores.append(entity.get('confidence', 0.5))
        
        # Collect confidence scores from relationships
        for rel in parsed_data.get('relationships', []):
            scores.append(rel.get('confidence', 0.5))
        
        return sum(scores) / len(scores) if scores else 0.5

    def _get_confidence_level(self, score: float) -> ConfidenceLevel:
        """Convert confidence score to confidence level."""
        if score >= 0.8:
            return ConfidenceLevel.HIGH
        elif score >= 0.6:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW

    def _generate_warnings(self, parsed_data: Dict[str, Any], confidence_score: float) -> List[str]:
        """Generate warnings based on parsing results."""
        warnings = []
        
        if confidence_score < 0.6:
            warnings.append("Low confidence in parsing results - manual review recommended")
        
        if not parsed_data.get('requirements'):
            warnings.append("No clear requirements identified - consider providing more detail")
        
        if not parsed_data.get('entities'):
            warnings.append("No business entities identified - may need domain clarification")
        
        return warnings

    async def get_clarifying_questions(self, requirements_id: int) -> List[ClarificationModel]:
        """Get clarifying questions for specific requirements."""
        clarifications = self.db_session.query(Clarification).filter(
            Clarification.requirements_id == requirements_id,
            Clarification.is_answered == False
        ).all()
        
        return [
            ClarificationModel(
                id=c.id,
                question=c.question,
                answer=c.answer,
                is_answered=c.is_answered,
                priority=c.priority,
                created_at=c.created_at,
                answered_at=c.answered_at
            )
            for c in clarifications
        ]

    async def answer_clarification(
        self, clarification_id: int, answer: str
    ) -> ClarificationModel:
        """Answer a clarification question."""
        clarification = self.db_session.query(Clarification).filter(
            Clarification.id == clarification_id
        ).first()
        
        if not clarification:
            raise ValueError(f"Clarification {clarification_id} not found")
        
        clarification.answer = answer
        clarification.is_answered = True
        clarification.answered_at = datetime.utcnow()
        
        self.db_session.commit()
        
        return ClarificationModel(
            id=clarification.id,
            question=clarification.question,
            answer=clarification.answer,
            is_answered=clarification.is_answered,
            priority=clarification.priority,
            created_at=clarification.created_at,
            answered_at=clarification.answered_at
        )