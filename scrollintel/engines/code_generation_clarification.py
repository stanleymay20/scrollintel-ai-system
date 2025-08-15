"""
Clarification engine for automated code generation.
Identifies ambiguous requirements and generates clarifying questions.
"""
import json
import re
import uuid
from typing import List, Dict, Any, Optional, Tuple
from openai import OpenAI

from scrollintel.models.code_generation_models import (
    Clarification, ParsedRequirement, Entity, Requirements,
    RequirementType, EntityType, ConfidenceLevel
)
from scrollintel.core.config import get_settings


class ClarificationEngine:
    """Generates clarifying questions for ambiguous requirements."""
    
    def __init__(self):
        """Initialize the clarification engine."""
        self.settings = get_settings()
        self.client = OpenAI(api_key=self.settings.openai_api_key)
        self.model = "gpt-4-turbo-preview"
        
        # Ambiguity patterns that trigger clarifications
        self.ambiguity_patterns = {
            'vague_quantities': [
                r'\b(?:many|few|several|some|multiple|various)\b',
                r'\b(?:large|small|big|little|huge|tiny)\b',
                r'\b(?:fast|slow|quick|rapid)\b'
            ],
            'unclear_scope': [
                r'\b(?:system|application|platform|solution)\b(?!\s+\w+)',
                r'\b(?:users?|customers?|people)\b(?!\s+(?:can|should|will|must))',
                r'\b(?:data|information|content)\b(?!\s+\w+)'
            ],
            'missing_details': [
                r'\b(?:somehow|automatically|appropriately|properly)\b',
                r'\b(?:etc|and\s+so\s+on|among\s+others)\b',
                r'\b(?:relevant|appropriate|suitable|necessary)\b'
            ],
            'conditional_ambiguity': [
                r'\bif\s+(?:needed|necessary|required|possible)\b',
                r'\bwhen\s+(?:appropriate|suitable|ready)\b',
                r'\bunless\s+(?:otherwise|specified)\b'
            ],
            'technical_ambiguity': [
                r'\b(?:integrate|connection|interface)\b(?!\s+with\s+\w+)',
                r'\b(?:secure|encrypted|protected)\b(?!\s+(?:using|with|by))',
                r'\b(?:optimized|efficient|performant)\b(?!\s+for\s+\w+)'
            ]
        }
    
    def generate_clarifications(self, requirements: Requirements) -> List[Clarification]:
        """
        Generate clarifying questions for ambiguous requirements.
        
        Args:
            requirements: The requirements object to analyze
            
        Returns:
            List of clarification questions
        """
        clarifications = []
        
        # Analyze individual requirements
        for req in requirements.parsed_requirements:
            req_clarifications = self._analyze_requirement(req)
            clarifications.extend(req_clarifications)
        
        # Analyze overall requirements completeness
        completeness_clarifications = self._analyze_completeness(requirements)
        clarifications.extend(completeness_clarifications)
        
        # Analyze entity relationships
        relationship_clarifications = self._analyze_relationships(requirements)
        clarifications.extend(relationship_clarifications)
        
        # Remove duplicates and prioritize
        clarifications = self._deduplicate_and_prioritize(clarifications)
        
        return clarifications
    
    def _analyze_requirement(self, requirement: ParsedRequirement) -> List[Clarification]:
        """Analyze a single requirement for ambiguities."""
        clarifications = []
        
        # Check confidence level
        if requirement.confidence == ConfidenceLevel.LOW:
            clarifications.append(Clarification(
                id=str(uuid.uuid4()),
                question=f"Could you provide more specific details about: '{requirement.structured_text}'?",
                context=requirement.original_text,
                priority=2,
                requirement_id=requirement.id
            ))
        
        # Check for ambiguous patterns
        text_lower = requirement.structured_text.lower()
        
        for category, patterns in self.ambiguity_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    question = self._generate_pattern_question(category, pattern, requirement)
                    if question:
                        clarifications.append(question)
        
        # Check for missing acceptance criteria
        if not requirement.acceptance_criteria:
            clarifications.append(Clarification(
                id=str(uuid.uuid4()),
                question=f"What are the specific acceptance criteria for: '{requirement.structured_text}'?",
                context=requirement.original_text,
                suggested_answers=[
                    "Define measurable success criteria",
                    "Specify expected behavior",
                    "List validation requirements"
                ],
                priority=2,
                requirement_id=requirement.id
            ))
        
        # Use GPT-4 for advanced ambiguity detection
        try:
            gpt4_clarifications = self._generate_gpt4_clarifications(requirement)
            clarifications.extend(gpt4_clarifications)
        except Exception:
            pass  # Continue with pattern-based clarifications
        
        return clarifications
    
    def _analyze_completeness(self, requirements: Requirements) -> List[Clarification]:
        """Analyze overall requirements completeness."""
        clarifications = []
        
        # Check for missing user roles
        user_entities = [e for e in requirements.entities if e.type == EntityType.USER_ROLE]
        if not user_entities:
            clarifications.append(Clarification(
                id=str(uuid.uuid4()),
                question="Who are the primary users of this system?",
                context="No user roles were identified in the requirements",
                suggested_answers=[
                    "End users/customers",
                    "Administrators",
                    "System operators",
                    "External integrators"
                ],
                priority=1
            ))
        
        # Check for missing data entities
        data_entities = [e for e in requirements.entities if e.type == EntityType.DATA_ENTITY]
        if not data_entities:
            clarifications.append(Clarification(
                id=str(uuid.uuid4()),
                question="What types of data will the system store and manage?",
                context="No data entities were identified in the requirements",
                suggested_answers=[
                    "User profiles and accounts",
                    "Business transactions",
                    "Configuration data",
                    "Audit logs"
                ],
                priority=1
            ))
        
        # Check for missing non-functional requirements
        non_functional = [r for r in requirements.parsed_requirements 
                         if r.requirement_type in [RequirementType.PERFORMANCE, RequirementType.SECURITY]]
        if not non_functional:
            clarifications.append(Clarification(
                id=str(uuid.uuid4()),
                question="What are the performance and security requirements for the system?",
                context="No non-functional requirements were specified",
                suggested_answers=[
                    "Expected number of concurrent users",
                    "Response time requirements",
                    "Security and compliance needs",
                    "Availability requirements"
                ],
                priority=2
            ))
        
        # Check for missing integration requirements
        integration_reqs = [r for r in requirements.parsed_requirements 
                           if r.requirement_type == RequirementType.INTEGRATION]
        tech_entities = [e for e in requirements.entities if e.type == EntityType.TECHNOLOGY]
        
        if not integration_reqs and not tech_entities:
            clarifications.append(Clarification(
                id=str(uuid.uuid4()),
                question="Does the system need to integrate with any external services or systems?",
                context="No integration requirements were identified",
                suggested_answers=[
                    "Third-party APIs",
                    "Existing databases",
                    "Authentication services",
                    "Payment gateways",
                    "No external integrations needed"
                ],
                priority=3
            ))
        
        return clarifications
    
    def _analyze_relationships(self, requirements: Requirements) -> List[Clarification]:
        """Analyze entity relationships for missing connections."""
        clarifications = []
        
        # Check for isolated entities
        entity_ids_in_relationships = set()
        for rel in requirements.relationships:
            entity_ids_in_relationships.add(rel.source_entity_id)
            entity_ids_in_relationships.add(rel.target_entity_id)
        
        isolated_entities = [e for e in requirements.entities 
                           if e.id not in entity_ids_in_relationships and e.type != EntityType.CONSTRAINT]
        
        if len(isolated_entities) > 2:
            entity_names = [e.name for e in isolated_entities[:3]]
            clarifications.append(Clarification(
                id=str(uuid.uuid4()),
                question=f"How do these entities relate to each other: {', '.join(entity_names)}?",
                context="Some entities appear to be isolated without clear relationships",
                priority=3
            ))
        
        return clarifications
    
    def _generate_pattern_question(self, category: str, pattern: str, requirement: ParsedRequirement) -> Optional[Clarification]:
        """Generate a clarification question based on ambiguity pattern."""
        questions = {
            'vague_quantities': "Could you specify exact numbers or ranges instead of vague quantities?",
            'unclear_scope': "Could you provide more specific details about the scope and boundaries?",
            'missing_details': "What specific details or examples can you provide?",
            'conditional_ambiguity': "Under what specific conditions should this behavior occur?",
            'technical_ambiguity': "Could you specify the technical requirements and constraints?"
        }
        
        if category in questions:
            return Clarification(
                id=str(uuid.uuid4()),
                question=questions[category],
                context=requirement.structured_text,
                priority=3,
                requirement_id=requirement.id
            )
        
        return None
    
    def _generate_gpt4_clarifications(self, requirement: ParsedRequirement) -> List[Clarification]:
        """Generate clarifications using GPT-4."""
        prompt = f"""
        Analyze this software requirement for ambiguities and missing information:
        
        Requirement: "{requirement.structured_text}"
        Type: {requirement.requirement_type.value}
        Intent: {requirement.intent.value}
        
        Identify specific ambiguities and generate clarifying questions that would help a developer implement this requirement.
        
        For each ambiguity, provide:
        1. question: A specific clarifying question
        2. context: What part of the requirement is ambiguous
        3. suggested_answers: 2-3 possible answers or approaches
        4. priority: 1 (critical), 2 (important), or 3 (nice to have)
        
        Return as JSON array of clarification objects.
        Only include questions that are genuinely needed for implementation.
        """
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an expert at identifying ambiguities in software requirements."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )
        
        content = response.choices[0].message.content
        clarifications_data = json.loads(content)
        
        clarifications = []
        for clarif_data in clarifications_data:
            clarification = Clarification(
                id=str(uuid.uuid4()),
                question=clarif_data.get('question', ''),
                context=clarif_data.get('context', requirement.structured_text),
                suggested_answers=clarif_data.get('suggested_answers', []),
                priority=clarif_data.get('priority', 3),
                requirement_id=requirement.id
            )
            clarifications.append(clarification)
        
        return clarifications
    
    def _deduplicate_and_prioritize(self, clarifications: List[Clarification]) -> List[Clarification]:
        """Remove duplicate clarifications and sort by priority."""
        # Remove duplicates based on similar questions
        unique_clarifications = []
        seen_questions = set()
        
        for clarif in clarifications:
            # Normalize question for comparison
            normalized = re.sub(r'\s+', ' ', clarif.question.lower().strip())
            if normalized not in seen_questions:
                seen_questions.add(normalized)
                unique_clarifications.append(clarif)
        
        # Sort by priority (1 = highest priority)
        unique_clarifications.sort(key=lambda x: (x.priority, x.question))
        
        return unique_clarifications
    
    def answer_clarification(self, clarification_id: str, answer: str, requirements: Requirements) -> Requirements:
        """
        Process an answer to a clarification and update requirements.
        
        Args:
            clarification_id: ID of the clarification being answered
            answer: The provided answer
            requirements: The requirements object to update
            
        Returns:
            Updated requirements object
        """
        # Find the clarification
        clarification = None
        for clarif in requirements.clarifications:
            if clarif.id == clarification_id:
                clarification = clarif
                break
        
        if not clarification:
            return requirements
        
        # Update the related requirement if applicable
        if clarification.requirement_id:
            for req in requirements.parsed_requirements:
                if req.id == clarification.requirement_id:
                    # Add the answer as additional context
                    if 'clarification_answers' not in req.metadata:
                        req.metadata['clarification_answers'] = {}
                    req.metadata['clarification_answers'][clarification.question] = answer
                    
                    # Improve confidence if answer provides clarity
                    if req.confidence == ConfidenceLevel.LOW:
                        req.confidence = ConfidenceLevel.MEDIUM
                    break
        
        # Remove the answered clarification
        requirements.clarifications = [c for c in requirements.clarifications if c.id != clarification_id]
        
        # Recalculate completeness score
        requirements.completeness_score = self._recalculate_completeness(requirements)
        
        return requirements
    
    def _recalculate_completeness(self, requirements: Requirements) -> float:
        """Recalculate completeness score after clarifications are answered."""
        score = 0.0
        total_checks = 8
        
        # Check for functional requirements
        if any(req.requirement_type == RequirementType.FUNCTIONAL for req in requirements.parsed_requirements):
            score += 1
        
        # Check for user roles
        if any(e.type == EntityType.USER_ROLE for e in requirements.entities):
            score += 1
        
        # Check for data entities
        if any(e.type == EntityType.DATA_ENTITY for e in requirements.entities):
            score += 1
        
        # Check for system components
        if any(e.type == EntityType.SYSTEM_COMPONENT for e in requirements.entities):
            score += 1
        
        # Check for acceptance criteria
        if any(req.acceptance_criteria for req in requirements.parsed_requirements):
            score += 1
        
        # Check for non-functional requirements
        if any(req.requirement_type in [RequirementType.PERFORMANCE, RequirementType.SECURITY] 
               for req in requirements.parsed_requirements):
            score += 1
        
        # Check for high confidence requirements
        high_confidence_reqs = [req for req in requirements.parsed_requirements 
                               if req.confidence == ConfidenceLevel.HIGH]
        if len(high_confidence_reqs) >= len(requirements.parsed_requirements) * 0.7:
            score += 1
        
        # Check for answered clarifications
        answered_clarifications = 0
        for req in requirements.parsed_requirements:
            if req.metadata.get('clarification_answers'):
                answered_clarifications += len(req.metadata['clarification_answers'])
        
        if answered_clarifications > 0:
            score += 1
        
        return score / total_checks