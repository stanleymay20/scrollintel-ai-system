"""
Clarification Engine for the Automated Code Generation System.
Generates clarifying questions for ambiguous or incomplete requirements.
"""

import logging
import json
import re
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from enum import Enum
import openai
from sqlalchemy.orm import Session

from scrollintel.models.nlp_models import (
    Requirements, Clarification, ClarificationModel,
    RequirementsModel, ParsedRequirementModel, EntityModel
)
from scrollintel.core.config import get_settings

logger = logging.getLogger(__name__)


class QuestionType(str, Enum):
    """Types of clarification questions."""
    FUNCTIONAL_DETAIL = "functional_detail"
    NON_FUNCTIONAL_REQUIREMENT = "non_functional_requirement"
    BUSINESS_RULE = "business_rule"
    USER_INTERACTION = "user_interaction"
    DATA_REQUIREMENT = "data_requirement"
    INTEGRATION_DETAIL = "integration_detail"
    TECHNICAL_CONSTRAINT = "technical_constraint"
    ACCEPTANCE_CRITERIA = "acceptance_criteria"
    EDGE_CASE = "edge_case"
    PRIORITY_CLARIFICATION = "priority_clarification"


class QuestionPriority(str, Enum):
    """Priority levels for clarification questions."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ClarificationQuestion:
    """Represents a clarification question with metadata."""
    
    def __init__(
        self,
        question: str,
        question_type: QuestionType,
        priority: QuestionPriority,
        context: str,
        suggested_answers: List[str] = None,
        related_entities: List[str] = None,
        reasoning: str = ""
    ):
        self.question = question
        self.question_type = question_type
        self.priority = priority
        self.context = context
        self.suggested_answers = suggested_answers or []
        self.related_entities = related_entities or []
        self.reasoning = reasoning


class ClarificationResult:
    """Result of clarification analysis."""
    
    def __init__(
        self,
        questions: List[ClarificationQuestion],
        ambiguity_score: float,
        completeness_score: float,
        critical_gaps: List[str],
        recommendations: List[str]
    ):
        self.questions = questions
        self.ambiguity_score = ambiguity_score
        self.completeness_score = completeness_score
        self.critical_gaps = critical_gaps
        self.recommendations = recommendations


class ClarificationEngine:
    """
    Advanced Clarification Engine that identifies ambiguous requirements
    and generates targeted questions to improve requirement quality.
    """
    
    def __init__(self, db_session: Session):
        self.db_session = db_session
        self.settings = get_settings()
        self.client = openai.OpenAI(api_key=self.settings.openai_api_key)
        
        # Ambiguity indicators
        self.ambiguity_patterns = {
            'vague_quantifiers': [
                r'\b(some|many|few|several|various|multiple|numerous)\b',
                r'\b(a lot of|lots of|plenty of|bunch of)\b',
                r'\b(around|about|approximately|roughly|nearly)\b'
            ],
            'uncertain_language': [
                r'\b(maybe|perhaps|possibly|probably|likely|might|could|should)\b',
                r'\b(seems|appears|looks like|sounds like)\b',
                r'\b(kind of|sort of|something like|similar to)\b'
            ],
            'incomplete_specifications': [
                r'\b(etc|and so on|and more|among others)\b',
                r'\b(to be determined|tbd|to be decided)\b',
                r'\b(as needed|as required|as appropriate)\b'
            ],
            'subjective_terms': [
                r'\b(good|bad|better|best|nice|great|awesome|cool)\b',
                r'\b(fast|slow|quick|efficient|user.?friendly|intuitive)\b',
                r'\b(simple|easy|complex|difficult|hard)\b'
            ],
            'missing_details': [
                r'\bwithout\s+(specifying|defining|explaining)\b',
                r'\b(somehow|someway|in some way)\b',
                r'\b(details|specifics|information)\s+(missing|needed|required)\b'
            ]
        }
        
        # Completeness checkers
        self.completeness_requirements = {
            'functional_requirements': [
                'user stories', 'use cases', 'features', 'functionality',
                'what the system should do', 'business logic'
            ],
            'non_functional_requirements': [
                'performance', 'scalability', 'security', 'usability',
                'reliability', 'availability', 'maintainability'
            ],
            'data_requirements': [
                'data models', 'database', 'storage', 'data flow',
                'data validation', 'data integrity'
            ],
            'user_interface': [
                'ui', 'user interface', 'frontend', 'user experience',
                'screens', 'forms', 'navigation'
            ],
            'integration_requirements': [
                'apis', 'external systems', 'third party', 'integrations',
                'data exchange', 'connectors'
            ],
            'technical_constraints': [
                'technology stack', 'platform', 'architecture',
                'deployment', 'infrastructure', 'environment'
            ]
        }
        
        # System prompt for AI-powered clarification
        self.clarification_prompt = """
        You are an expert business analyst and requirements engineer. Analyze the following requirements and identify areas that need clarification.
        
        Focus on:
        1. Ambiguous or vague statements that could be interpreted multiple ways
        2. Missing functional requirements or incomplete user stories
        3. Unclear business rules or logic
        4. Missing non-functional requirements (performance, security, etc.)
        5. Incomplete data requirements or unclear data relationships
        6. Missing integration details or external system interactions
        7. Unclear user roles or permissions
        8. Missing acceptance criteria or success metrics
        9. Edge cases or error handling scenarios not addressed
        10. Priority or timeline ambiguities
        
        For each area needing clarification, generate:
        - A specific, actionable question
        - The type of clarification needed
        - Priority level (critical, high, medium, low)
        - Context explaining why this clarification is important
        - Suggested answer options when applicable
        - Related entities or components affected
        
        Return as JSON with detailed clarification questions.
        """

    async def analyze_requirements(
        self, 
        requirements: RequirementsModel,
        parsed_requirements: List[ParsedRequirementModel] = None,
        entities: List[EntityModel] = None
    ) -> ClarificationResult:
        """
        Analyze requirements and generate clarification questions.
        
        Args:
            requirements: The requirements to analyze
            parsed_requirements: Parsed requirements for additional context
            entities: Extracted entities for additional context
            
        Returns:
            ClarificationResult with questions and analysis
        """
        try:
            # Analyze ambiguity
            ambiguity_score = self._calculate_ambiguity_score(requirements.raw_text)
            
            # Analyze completeness
            completeness_score = self._calculate_completeness_score(
                requirements.raw_text, parsed_requirements, entities
            )
            
            # Generate questions using multiple approaches
            rule_based_questions = self._generate_rule_based_questions(requirements.raw_text)
            ai_questions = await self._generate_ai_questions(requirements.raw_text)
            context_questions = self._generate_context_questions(
                parsed_requirements, entities
            )
            
            # Merge and prioritize questions
            all_questions = self._merge_questions(
                rule_based_questions, ai_questions, context_questions
            )
            
            # Identify critical gaps
            critical_gaps = self._identify_critical_gaps(
                requirements.raw_text, parsed_requirements, entities
            )
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                ambiguity_score, completeness_score, critical_gaps
            )
            
            return ClarificationResult(
                questions=all_questions,
                ambiguity_score=ambiguity_score,
                completeness_score=completeness_score,
                critical_gaps=critical_gaps,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Error analyzing requirements: {str(e)}")
            return ClarificationResult(
                questions=[],
                ambiguity_score=0.0,
                completeness_score=0.0,
                critical_gaps=[f"Analysis failed: {str(e)}"],
                recommendations=["Manual review recommended due to analysis failure"]
            )

    def _calculate_ambiguity_score(self, text: str) -> float:
        """Calculate ambiguity score based on linguistic patterns."""
        text_lower = text.lower()
        total_indicators = 0
        word_count = len(text.split())
        
        for category, patterns in self.ambiguity_patterns.items():
            for pattern in patterns:
                matches = len(re.findall(pattern, text_lower))
                total_indicators += matches
        
        # Normalize by text length
        if word_count > 0:
            ambiguity_score = min(total_indicators / word_count * 10, 1.0)
        else:
            ambiguity_score = 0.0
        
        return ambiguity_score

    def _calculate_completeness_score(
        self,
        text: str,
        parsed_requirements: List[ParsedRequirementModel] = None,
        entities: List[EntityModel] = None
    ) -> float:
        """Calculate completeness score based on requirement coverage."""
        text_lower = text.lower()
        covered_areas = 0
        total_areas = len(self.completeness_requirements)
        
        for area, keywords in self.completeness_requirements.items():
            area_covered = any(keyword in text_lower for keyword in keywords)
            if area_covered:
                covered_areas += 1
        
        base_score = covered_areas / total_areas
        
        # Bonus for having parsed requirements and entities
        if parsed_requirements and len(parsed_requirements) > 0:
            base_score += 0.1
        
        if entities and len(entities) > 0:
            base_score += 0.1
        
        return min(base_score, 1.0)

    def _generate_rule_based_questions(self, text: str) -> List[ClarificationQuestion]:
        """Generate questions based on rule-based analysis."""
        questions = []
        text_lower = text.lower()
        
        # Check for ambiguous quantifiers
        for pattern in self.ambiguity_patterns['vague_quantifiers']:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                context = self._get_context(text, match.start(), match.end())
                questions.append(ClarificationQuestion(
                    question=f"Can you specify the exact number or range instead of '{match.group()}'?",
                    question_type=QuestionType.FUNCTIONAL_DETAIL,
                    priority=QuestionPriority.MEDIUM,
                    context=context,
                    reasoning="Vague quantifier detected that could lead to implementation ambiguity"
                ))
        
        # Check for subjective terms
        for pattern in self.ambiguity_patterns['subjective_terms']:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                context = self._get_context(text, match.start(), match.end())
                questions.append(ClarificationQuestion(
                    question=f"What specific criteria define '{match.group()}' in this context?",
                    question_type=QuestionType.ACCEPTANCE_CRITERIA,
                    priority=QuestionPriority.HIGH,
                    context=context,
                    reasoning="Subjective term needs objective criteria for implementation"
                ))
        
        # Check for missing completeness areas
        for area, keywords in self.completeness_requirements.items():
            area_mentioned = any(keyword in text_lower for keyword in keywords)
            if not area_mentioned:
                question = self._generate_completeness_question(area)
                if question:
                    questions.append(question)
        
        return questions

    async def _generate_ai_questions(self, text: str) -> List[ClarificationQuestion]:
        """Generate questions using AI analysis."""
        try:
            response = await self.client.chat.completions.acreate(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": self.clarification_prompt},
                    {"role": "user", "content": text}
                ],
                temperature=0.2,
                max_tokens=2000
            )
            
            content = response.choices[0].message.content
            data = self._parse_ai_response(content)
            
            questions = []
            for question_data in data.get('questions', []):
                try:
                    question_type = QuestionType(question_data.get('type', 'functional_detail'))
                    priority = QuestionPriority(question_data.get('priority', 'medium'))
                    
                    questions.append(ClarificationQuestion(
                        question=question_data.get('question', ''),
                        question_type=question_type,
                        priority=priority,
                        context=question_data.get('context', ''),
                        suggested_answers=question_data.get('suggested_answers', []),
                        related_entities=question_data.get('related_entities', []),
                        reasoning=question_data.get('reasoning', '')
                    ))
                except (ValueError, KeyError) as e:
                    logger.warning(f"Error parsing AI question: {str(e)}")
                    continue
            
            return questions
            
        except Exception as e:
            logger.error(f"Error generating AI questions: {str(e)}")
            return []

    def _generate_context_questions(
        self,
        parsed_requirements: List[ParsedRequirementModel] = None,
        entities: List[EntityModel] = None
    ) -> List[ClarificationQuestion]:
        """Generate questions based on parsed requirements and entities context."""
        questions = []
        
        if parsed_requirements:
            # Check for requirements without acceptance criteria
            for req in parsed_requirements:
                if not req.acceptance_criteria or len(req.acceptance_criteria) == 0:
                    questions.append(ClarificationQuestion(
                        question=f"What are the specific acceptance criteria for: '{req.description}'?",
                        question_type=QuestionType.ACCEPTANCE_CRITERIA,
                        priority=QuestionPriority.HIGH,
                        context=req.description,
                        reasoning="Requirement lacks specific acceptance criteria"
                    ))
        
        if entities:
            # Check for entities without clear relationships
            entity_names = [entity.name for entity in entities]
            isolated_entities = []
            
            for entity in entities:
                # This is a simplified check - in a real implementation,
                # you'd check actual relationships from the database
                if len(entity.attributes) == 0:
                    isolated_entities.append(entity.name)
            
            if isolated_entities:
                questions.append(ClarificationQuestion(
                    question=f"How do these entities relate to other parts of the system: {', '.join(isolated_entities)}?",
                    question_type=QuestionType.DATA_REQUIREMENT,
                    priority=QuestionPriority.MEDIUM,
                    context="Entities without clear relationships identified",
                    related_entities=isolated_entities,
                    reasoning="Entities need clear relationships for proper system design"
                ))
        
        return questions

    def _generate_completeness_question(self, area: str) -> Optional[ClarificationQuestion]:
        """Generate a question for a missing completeness area."""
        question_templates = {
            'functional_requirements': ClarificationQuestion(
                question="What are the main functional requirements and user stories for this system?",
                question_type=QuestionType.FUNCTIONAL_DETAIL,
                priority=QuestionPriority.CRITICAL,
                context="No clear functional requirements identified",
                reasoning="Functional requirements are essential for system development"
            ),
            'non_functional_requirements': ClarificationQuestion(
                question="What are the performance, security, and scalability requirements?",
                question_type=QuestionType.NON_FUNCTIONAL_REQUIREMENT,
                priority=QuestionPriority.HIGH,
                context="Non-functional requirements not specified",
                reasoning="Non-functional requirements affect architecture decisions"
            ),
            'data_requirements': ClarificationQuestion(
                question="What data needs to be stored, processed, and how should it be structured?",
                question_type=QuestionType.DATA_REQUIREMENT,
                priority=QuestionPriority.HIGH,
                context="Data requirements not clearly specified",
                reasoning="Data requirements are crucial for database design"
            ),
            'user_interface': ClarificationQuestion(
                question="What are the user interface requirements and user experience expectations?",
                question_type=QuestionType.USER_INTERACTION,
                priority=QuestionPriority.MEDIUM,
                context="User interface requirements not specified",
                reasoning="UI requirements affect user adoption and satisfaction"
            ),
            'integration_requirements': ClarificationQuestion(
                question="What external systems or APIs need to be integrated with?",
                question_type=QuestionType.INTEGRATION_DETAIL,
                priority=QuestionPriority.MEDIUM,
                context="Integration requirements not specified",
                reasoning="Integration requirements affect system architecture"
            ),
            'technical_constraints': ClarificationQuestion(
                question="What are the technical constraints, preferred technologies, and deployment requirements?",
                question_type=QuestionType.TECHNICAL_CONSTRAINT,
                priority=QuestionPriority.MEDIUM,
                context="Technical constraints not specified",
                reasoning="Technical constraints guide technology choices"
            )
        }
        
        return question_templates.get(area)

    def _merge_questions(self, *question_lists: List[ClarificationQuestion]) -> List[ClarificationQuestion]:
        """Merge questions from different sources and remove duplicates."""
        all_questions = []
        seen_questions = set()
        
        for question_list in question_lists:
            for question in question_list:
                # Use question text as deduplication key
                question_key = question.question.lower().strip()
                if question_key not in seen_questions:
                    all_questions.append(question)
                    seen_questions.add(question_key)
        
        # Sort by priority
        priority_order = {
            QuestionPriority.CRITICAL: 0,
            QuestionPriority.HIGH: 1,
            QuestionPriority.MEDIUM: 2,
            QuestionPriority.LOW: 3
        }
        
        all_questions.sort(key=lambda q: priority_order.get(q.priority, 3))
        
        return all_questions

    def _identify_critical_gaps(
        self,
        text: str,
        parsed_requirements: List[ParsedRequirementModel] = None,
        entities: List[EntityModel] = None
    ) -> List[str]:
        """Identify critical gaps in requirements."""
        gaps = []
        text_lower = text.lower()
        
        # Check for missing critical elements
        if 'user' not in text_lower and 'role' not in text_lower:
            gaps.append("No user roles or personas identified")
        
        if not any(word in text_lower for word in ['data', 'database', 'store', 'save']):
            gaps.append("No data storage requirements specified")
        
        if not any(word in text_lower for word in ['security', 'authentication', 'authorization']):
            gaps.append("No security requirements mentioned")
        
        if not parsed_requirements or len(parsed_requirements) == 0:
            gaps.append("No clear functional requirements identified")
        
        if not entities or len(entities) < 2:
            gaps.append("Insufficient business entities identified")
        
        return gaps

    def _generate_recommendations(
        self,
        ambiguity_score: float,
        completeness_score: float,
        critical_gaps: List[str]
    ) -> List[str]:
        """Generate recommendations based on analysis results."""
        recommendations = []
        
        if ambiguity_score > 0.3:
            recommendations.append("Consider using more specific and measurable language")
        
        if completeness_score < 0.5:
            recommendations.append("Requirements appear incomplete - consider adding more detail")
        
        if critical_gaps:
            recommendations.append("Address critical gaps before proceeding with development")
        
        if ambiguity_score > 0.5 or completeness_score < 0.3:
            recommendations.append("Consider conducting stakeholder interviews for clarification")
        
        if not recommendations:
            recommendations.append("Requirements appear well-defined and ready for development")
        
        return recommendations

    def _get_context(self, text: str, start: int, end: int, window: int = 50) -> str:
        """Get context around a text match."""
        context_start = max(0, start - window)
        context_end = min(len(text), end + window)
        return text[context_start:context_end].strip()

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
            
            return {'questions': []}

    async def save_questions(
        self, 
        requirements_id: int, 
        questions: List[ClarificationQuestion]
    ) -> List[ClarificationModel]:
        """Save clarification questions to database."""
        saved_questions = []
        
        for question in questions:
            clarification = Clarification(
                requirements_id=requirements_id,
                question=question.question,
                priority=question.priority.value,
                created_at=datetime.utcnow()
            )
            
            self.db_session.add(clarification)
            self.db_session.flush()  # Get ID
            
            saved_questions.append(ClarificationModel(
                id=clarification.id,
                question=clarification.question,
                priority=clarification.priority,
                created_at=clarification.created_at
            ))
        
        self.db_session.commit()
        return saved_questions