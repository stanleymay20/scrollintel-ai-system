"""
Strategic Narrative Development Engine for Board Executive Mastery System
"""

import logging
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime
import re
from dataclasses import dataclass
from enum import Enum

from ..models.executive_communication_models import (
    ExecutiveAudience, ExecutiveLevel, CommunicationStyle
)


class NarrativeType(Enum):
    """Types of strategic narratives"""
    VISION_STORY = "vision_story"
    TRANSFORMATION_JOURNEY = "transformation_journey"
    COMPETITIVE_ADVANTAGE = "competitive_advantage"
    GROWTH_STRATEGY = "growth_strategy"
    CRISIS_RESPONSE = "crisis_response"
    INNOVATION_SHOWCASE = "innovation_showcase"
    MARKET_OPPORTUNITY = "market_opportunity"


class NarrativeStructure(Enum):
    """Narrative structure patterns"""
    PROBLEM_SOLUTION = "problem_solution"
    HERO_JOURNEY = "hero_journey"
    BEFORE_AFTER = "before_after"
    THREE_ACT = "three_act"
    PYRAMID = "pyramid"
    CHRONOLOGICAL = "chronological"


@dataclass
class StrategicContext:
    """Strategic context for narrative development"""
    company_position: str
    market_conditions: str
    competitive_landscape: str
    key_challenges: List[str]
    opportunities: List[str]
    stakeholder_concerns: List[str]
    success_metrics: List[str]
    timeline: str


@dataclass
class NarrativeElement:
    """Individual narrative element"""
    element_type: str  # opening, conflict, resolution, call_to_action
    content: str
    emotional_tone: str
    supporting_data: Dict[str, Any]
    audience_relevance: float


@dataclass
class StrategicNarrative:
    """Complete strategic narrative"""
    id: str
    title: str
    narrative_type: NarrativeType
    structure: NarrativeStructure
    audience_id: str
    elements: List[NarrativeElement]
    key_messages: List[str]
    emotional_arc: str
    call_to_action: str
    supporting_visuals: List[str]
    impact_score: float
    personalization_notes: str
    created_at: datetime


@dataclass
class NarrativeImpact:
    """Assessment of narrative impact"""
    id: str
    narrative_id: str
    audience_id: str
    engagement_level: float
    emotional_resonance: float
    message_retention: float
    action_likelihood: float
    credibility_score: float
    overall_impact: float
    feedback: Optional[str]
    measured_at: datetime


class NarrativeDeveloper:
    """Develops compelling strategic narratives for board presentations"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.narrative_templates = self._load_narrative_templates()
        self.emotional_frameworks = self._load_emotional_frameworks()
        
    def develop_strategic_narrative(self, strategy: Dict[str, Any], 
                                  audience: ExecutiveAudience,
                                  context: StrategicContext) -> StrategicNarrative:
        """Create compelling strategic story and presentation development"""
        try:
            # Determine optimal narrative type and structure
            narrative_type = self._select_narrative_type(strategy, audience, context)
            structure = self._select_narrative_structure(narrative_type, audience)
            
            # Develop narrative elements
            elements = self._create_narrative_elements(strategy, audience, context, structure)
            
            # Extract key messages
            key_messages = self._extract_key_messages(strategy, audience)
            
            # Design emotional arc
            emotional_arc = self._design_emotional_arc(narrative_type, audience)
            
            # Create call to action
            call_to_action = self._create_call_to_action(strategy, audience)
            
            # Generate supporting visuals recommendations
            supporting_visuals = self._recommend_supporting_visuals(narrative_type, elements)
            
            # Calculate impact prediction
            impact_score = self._predict_narrative_impact(elements, audience, context)
            
            # Generate personalization notes
            personalization_notes = self._generate_personalization_notes(audience, context)
            
            narrative = StrategicNarrative(
                id=f"narrative_{strategy.get('id', 'unknown')}_{audience.id}",
                title=self._generate_narrative_title(strategy, narrative_type),
                narrative_type=narrative_type,
                structure=structure,
                audience_id=audience.id,
                elements=elements,
                key_messages=key_messages,
                emotional_arc=emotional_arc,
                call_to_action=call_to_action,
                supporting_visuals=supporting_visuals,
                impact_score=impact_score,
                personalization_notes=personalization_notes,
                created_at=datetime.now()
            )
            
            self.logger.info(f"Successfully developed strategic narrative: {narrative.title}")
            return narrative
            
        except Exception as e:
            self.logger.error(f"Error developing strategic narrative: {str(e)}")
            raise
    
    def personalize_narrative(self, narrative: StrategicNarrative, 
                            audience: ExecutiveAudience) -> StrategicNarrative:
        """Implement narrative personalization for different board audiences"""
        try:
            # Adjust narrative elements based on audience preferences
            personalized_elements = []
            
            for element in narrative.elements:
                personalized_element = self._personalize_element(element, audience)
                personalized_elements.append(personalized_element)
            
            # Adjust key messages for audience
            personalized_messages = self._personalize_key_messages(narrative.key_messages, audience)
            
            # Adjust emotional arc
            personalized_emotional_arc = self._personalize_emotional_arc(narrative.emotional_arc, audience)
            
            # Adjust call to action
            personalized_cta = self._personalize_call_to_action(narrative.call_to_action, audience)
            
            # Update personalization notes
            updated_notes = f"{narrative.personalization_notes} | Personalized for {audience.communication_style.value} style"
            
            # Create personalized narrative
            personalized_narrative = StrategicNarrative(
                id=f"{narrative.id}_personalized",
                title=narrative.title,
                narrative_type=narrative.narrative_type,
                structure=narrative.structure,
                audience_id=audience.id,
                elements=personalized_elements,
                key_messages=personalized_messages,
                emotional_arc=personalized_emotional_arc,
                call_to_action=personalized_cta,
                supporting_visuals=narrative.supporting_visuals,
                impact_score=self._recalculate_impact_score(personalized_elements, audience),
                personalization_notes=updated_notes,
                created_at=datetime.now()
            )
            
            self.logger.info(f"Successfully personalized narrative for {audience.name}")
            return personalized_narrative
            
        except Exception as e:
            self.logger.error(f"Error personalizing narrative: {str(e)}")
            raise
    
    def _select_narrative_type(self, strategy: Dict[str, Any], 
                              audience: ExecutiveAudience,
                              context: StrategicContext) -> NarrativeType:
        """Select optimal narrative type based on strategy and audience"""
        strategy_focus = strategy.get('focus', '').lower()
        
        if 'transformation' in strategy_focus or 'change' in strategy_focus:
            return NarrativeType.TRANSFORMATION_JOURNEY
        elif 'growth' in strategy_focus or 'expansion' in strategy_focus:
            return NarrativeType.GROWTH_STRATEGY
        elif 'competitive' in strategy_focus or 'advantage' in strategy_focus:
            return NarrativeType.COMPETITIVE_ADVANTAGE
        elif 'innovation' in strategy_focus or 'technology' in strategy_focus:
            return NarrativeType.INNOVATION_SHOWCASE
        elif 'crisis' in strategy_focus or 'urgent' in strategy_focus:
            return NarrativeType.CRISIS_RESPONSE
        elif 'market' in strategy_focus or 'opportunity' in strategy_focus:
            return NarrativeType.MARKET_OPPORTUNITY
        else:
            return NarrativeType.VISION_STORY
    
    def _select_narrative_structure(self, narrative_type: NarrativeType, 
                                   audience: ExecutiveAudience) -> NarrativeStructure:
        """Select narrative structure based on type and audience"""
        if audience.communication_style == CommunicationStyle.ANALYTICAL:
            return NarrativeStructure.PROBLEM_SOLUTION
        elif audience.communication_style == CommunicationStyle.STRATEGIC:
            return NarrativeStructure.THREE_ACT
        elif narrative_type == NarrativeType.TRANSFORMATION_JOURNEY:
            return NarrativeStructure.HERO_JOURNEY
        elif narrative_type == NarrativeType.CRISIS_RESPONSE:
            return NarrativeStructure.PROBLEM_SOLUTION
        else:
            return NarrativeStructure.BEFORE_AFTER
    
    def _create_narrative_elements(self, strategy: Dict[str, Any],
                                  audience: ExecutiveAudience,
                                  context: StrategicContext,
                                  structure: NarrativeStructure) -> List[NarrativeElement]:
        """Create narrative elements based on structure"""
        elements = []
        
        if structure == NarrativeStructure.PROBLEM_SOLUTION:
            elements = [
                NarrativeElement(
                    element_type="problem_statement",
                    content=self._create_problem_statement(context),
                    emotional_tone="concern_urgency",
                    supporting_data={"challenges": context.key_challenges},
                    audience_relevance=0.9
                ),
                NarrativeElement(
                    element_type="solution_proposal",
                    content=self._create_solution_proposal(strategy),
                    emotional_tone="confidence_optimism",
                    supporting_data=strategy,
                    audience_relevance=0.95
                ),
                NarrativeElement(
                    element_type="implementation_path",
                    content=self._create_implementation_path(strategy, context),
                    emotional_tone="determination_clarity",
                    supporting_data={"timeline": context.timeline},
                    audience_relevance=0.8
                ),
                NarrativeElement(
                    element_type="expected_outcomes",
                    content=self._create_expected_outcomes(strategy, context),
                    emotional_tone="excitement_achievement",
                    supporting_data={"metrics": context.success_metrics},
                    audience_relevance=0.9
                )
            ]
        
        elif structure == NarrativeStructure.THREE_ACT:
            elements = [
                NarrativeElement(
                    element_type="act1_setup",
                    content=self._create_strategic_setup(context),
                    emotional_tone="anticipation_focus",
                    supporting_data={"market_conditions": context.market_conditions},
                    audience_relevance=0.8
                ),
                NarrativeElement(
                    element_type="act2_conflict",
                    content=self._create_strategic_conflict(context),
                    emotional_tone="tension_challenge",
                    supporting_data={"challenges": context.key_challenges},
                    audience_relevance=0.9
                ),
                NarrativeElement(
                    element_type="act3_resolution",
                    content=self._create_strategic_resolution(strategy),
                    emotional_tone="triumph_confidence",
                    supporting_data=strategy,
                    audience_relevance=0.95
                )
            ]
        
        elif structure == NarrativeStructure.HERO_JOURNEY:
            elements = [
                NarrativeElement(
                    element_type="ordinary_world",
                    content=self._create_current_state(context),
                    emotional_tone="stability_comfort",
                    supporting_data={"position": context.company_position},
                    audience_relevance=0.7
                ),
                NarrativeElement(
                    element_type="call_to_adventure",
                    content=self._create_transformation_call(context),
                    emotional_tone="excitement_uncertainty",
                    supporting_data={"opportunities": context.opportunities},
                    audience_relevance=0.9
                ),
                NarrativeElement(
                    element_type="journey_trials",
                    content=self._create_transformation_challenges(context),
                    emotional_tone="struggle_determination",
                    supporting_data={"challenges": context.key_challenges},
                    audience_relevance=0.8
                ),
                NarrativeElement(
                    element_type="transformation",
                    content=self._create_transformation_outcome(strategy),
                    emotional_tone="achievement_pride",
                    supporting_data=strategy,
                    audience_relevance=0.95
                )
            ]
        
        return elements
    
    def _create_problem_statement(self, context: StrategicContext) -> str:
        """Create compelling problem statement"""
        challenges = ", ".join(context.key_challenges[:3])
        return f"We face critical challenges in {context.market_conditions}: {challenges}. " \
               f"Our current position of {context.company_position} requires immediate strategic action."
    
    def _create_solution_proposal(self, strategy: Dict[str, Any]) -> str:
        """Create solution proposal"""
        return f"Our strategic response centers on {strategy.get('core_approach', 'comprehensive transformation')}. " \
               f"This approach will {strategy.get('primary_benefit', 'drive sustainable growth')} " \
               f"while {strategy.get('secondary_benefit', 'strengthening our competitive position')}."
    
    def _create_implementation_path(self, strategy: Dict[str, Any], context: StrategicContext) -> str:
        """Create implementation path description"""
        return f"Implementation will occur over {context.timeline} through " \
               f"{strategy.get('implementation_phases', '3 strategic phases')}. " \
               f"Each phase builds on the previous, ensuring {strategy.get('risk_mitigation', 'controlled risk')} " \
               f"and {strategy.get('value_delivery', 'continuous value delivery')}."
    
    def _create_expected_outcomes(self, strategy: Dict[str, Any], context: StrategicContext) -> str:
        """Create expected outcomes description"""
        metrics = ", ".join(context.success_metrics[:3])
        return f"Success will be measured through {metrics}. " \
               f"We anticipate {strategy.get('projected_impact', 'significant positive impact')} " \
               f"with {strategy.get('roi_timeline', 'ROI visible within 12 months')}."
    
    def _extract_key_messages(self, strategy: Dict[str, Any], audience: ExecutiveAudience) -> List[str]:
        """Extract key messages for the narrative"""
        messages = []
        
        # Core strategic message
        messages.append(f"Strategic initiative: {strategy.get('name', 'Transformation Strategy')}")
        
        # Value proposition
        if 'value_proposition' in strategy:
            messages.append(f"Value: {strategy['value_proposition']}")
        
        # Competitive advantage
        if 'competitive_advantage' in strategy:
            messages.append(f"Advantage: {strategy['competitive_advantage']}")
        
        # Risk mitigation
        if 'risk_mitigation' in strategy:
            messages.append(f"Risk management: {strategy['risk_mitigation']}")
        
        # Audience-specific message
        if audience.executive_level == ExecutiveLevel.CEO:
            messages.append("CEO focus: Market leadership and growth acceleration")
        elif audience.executive_level == ExecutiveLevel.BOARD_CHAIR:
            messages.append("Board focus: Governance excellence and stakeholder value")
        elif audience.executive_level == ExecutiveLevel.CFO:
            messages.append("Financial focus: ROI optimization and cost efficiency")
        
        return messages[:5]  # Limit to 5 key messages
    
    def _design_emotional_arc(self, narrative_type: NarrativeType, audience: ExecutiveAudience) -> str:
        """Design emotional arc for the narrative"""
        if narrative_type == NarrativeType.CRISIS_RESPONSE:
            return "urgency → determination → confidence → relief"
        elif narrative_type == NarrativeType.TRANSFORMATION_JOURNEY:
            return "anticipation → challenge → struggle → triumph"
        elif narrative_type == NarrativeType.GROWTH_STRATEGY:
            return "opportunity → excitement → commitment → achievement"
        elif audience.communication_style == CommunicationStyle.ANALYTICAL:
            return "curiosity → analysis → understanding → conviction"
        else:
            return "attention → interest → desire → action"
    
    def _create_call_to_action(self, strategy: Dict[str, Any], audience: ExecutiveAudience) -> str:
        """Create compelling call to action"""
        if audience.executive_level == ExecutiveLevel.BOARD_CHAIR:
            return f"Board approval requested for {strategy.get('name', 'strategic initiative')} " \
                   f"with {strategy.get('budget_request', 'proposed budget allocation')}."
        elif audience.executive_level == ExecutiveLevel.CEO:
            return f"Executive leadership commitment needed to drive {strategy.get('name', 'transformation')} " \
                   f"across all business units."
        else:
            return f"Strategic alignment and resource commitment required for " \
                   f"{strategy.get('name', 'initiative')} success."
    
    def _recommend_supporting_visuals(self, narrative_type: NarrativeType, 
                                    elements: List[NarrativeElement]) -> List[str]:
        """Recommend supporting visuals for the narrative"""
        visuals = []
        
        if narrative_type == NarrativeType.GROWTH_STRATEGY:
            visuals.extend(["market_growth_chart", "revenue_projection", "market_share_comparison"])
        elif narrative_type == NarrativeType.COMPETITIVE_ADVANTAGE:
            visuals.extend(["competitive_matrix", "differentiation_chart", "market_positioning"])
        elif narrative_type == NarrativeType.TRANSFORMATION_JOURNEY:
            visuals.extend(["transformation_roadmap", "before_after_comparison", "milestone_timeline"])
        
        # Add element-specific visuals
        for element in elements:
            if "data" in element.supporting_data:
                visuals.append("supporting_data_visualization")
            if "metrics" in element.supporting_data:
                visuals.append("metrics_dashboard")
        
        return list(set(visuals))  # Remove duplicates
    
    def _predict_narrative_impact(self, elements: List[NarrativeElement],
                                 audience: ExecutiveAudience,
                                 context: StrategicContext) -> float:
        """Predict narrative impact score"""
        base_score = 0.7
        
        # Adjust based on element relevance
        avg_relevance = sum(e.audience_relevance for e in elements) / len(elements)
        base_score += (avg_relevance - 0.5) * 0.3
        
        # Adjust based on audience alignment
        if audience.communication_style == CommunicationStyle.STRATEGIC:
            base_score += 0.1
        
        # Adjust based on context urgency
        if any("urgent" in challenge.lower() for challenge in context.key_challenges):
            base_score += 0.1
        
        return min(1.0, max(0.0, base_score))
    
    def _generate_personalization_notes(self, audience: ExecutiveAudience, 
                                       context: StrategicContext) -> str:
        """Generate personalization notes"""
        notes = []
        
        notes.append(f"Tailored for {audience.executive_level.value}")
        notes.append(f"Communication style: {audience.communication_style.value}")
        notes.append(f"Detail preference: {audience.detail_preference}")
        notes.append(f"Attention span: {audience.attention_span} minutes")
        
        if audience.risk_tolerance == "low":
            notes.append("Emphasize risk mitigation")
        elif audience.risk_tolerance == "high":
            notes.append("Highlight growth opportunities")
        
        return " | ".join(notes)
    
    def _personalize_element(self, element: NarrativeElement, 
                           audience: ExecutiveAudience) -> NarrativeElement:
        """Personalize individual narrative element"""
        personalized_content = element.content
        
        # Adjust based on communication style
        if audience.communication_style == CommunicationStyle.DIRECT:
            personalized_content = self._make_more_direct(personalized_content)
        elif audience.communication_style == CommunicationStyle.DIPLOMATIC:
            personalized_content = self._make_more_diplomatic(personalized_content)
        elif audience.communication_style == CommunicationStyle.ANALYTICAL:
            personalized_content = self._add_analytical_depth(personalized_content)
        
        # Adjust based on detail preference
        if audience.detail_preference == "low":
            personalized_content = self._simplify_content(personalized_content)
        elif audience.detail_preference == "high":
            personalized_content = self._add_detail(personalized_content, element.supporting_data)
        
        return NarrativeElement(
            element_type=element.element_type,
            content=personalized_content,
            emotional_tone=element.emotional_tone,
            supporting_data=element.supporting_data,
            audience_relevance=min(1.0, element.audience_relevance + 0.1)  # Boost relevance
        )
    
    def _personalize_key_messages(self, messages: List[str], 
                                 audience: ExecutiveAudience) -> List[str]:
        """Personalize key messages for audience"""
        personalized = []
        
        for message in messages:
            if audience.executive_level == ExecutiveLevel.CEO and "growth" not in message.lower():
                message = f"{message} - driving organizational growth"
            elif audience.executive_level == ExecutiveLevel.BOARD_CHAIR and "governance" not in message.lower():
                message = f"{message} - ensuring governance excellence"
            
            personalized.append(message)
        
        return personalized
    
    def _personalize_emotional_arc(self, emotional_arc: str, audience: ExecutiveAudience) -> str:
        """Personalize emotional arc for audience"""
        if audience.communication_style == CommunicationStyle.ANALYTICAL:
            return emotional_arc.replace("excitement", "interest").replace("triumph", "satisfaction")
        elif audience.risk_tolerance == "low":
            return emotional_arc.replace("excitement", "confidence").replace("challenge", "opportunity")
        
        return emotional_arc
    
    def _personalize_call_to_action(self, cta: str, audience: ExecutiveAudience) -> str:
        """Personalize call to action"""
        if audience.communication_style == CommunicationStyle.DIRECT:
            return cta.replace("requested", "required").replace("needed", "essential")
        elif audience.communication_style == CommunicationStyle.DIPLOMATIC:
            return cta.replace("required", "would benefit from").replace("must", "should consider")
        
        return cta
    
    def _recalculate_impact_score(self, elements: List[NarrativeElement], 
                                 audience: ExecutiveAudience) -> float:
        """Recalculate impact score after personalization"""
        base_score = sum(e.audience_relevance for e in elements) / len(elements)
        personalization_boost = 0.1  # Boost for personalization
        
        return min(1.0, base_score + personalization_boost)
    
    # Helper methods for content adaptation
    def _make_more_direct(self, content: str) -> str:
        """Make content more direct"""
        content = content.replace("we should consider", "we will")
        content = content.replace("it might be beneficial", "we must")
        content = content.replace("potentially", "")
        return content
    
    def _make_more_diplomatic(self, content: str) -> str:
        """Make content more diplomatic"""
        content = content.replace("we must", "we should consider")
        content = content.replace("will fail", "may face challenges")
        content = content.replace("critical", "important")
        return content
    
    def _add_analytical_depth(self, content: str) -> str:
        """Add analytical depth to content"""
        if not content.startswith("Analysis:"):
            content = f"Analysis: {content}\n\nData supports this approach based on market research and competitive analysis."
        return content
    
    def _simplify_content(self, content: str) -> str:
        """Simplify content for low detail preference"""
        sentences = content.split('.')
        simplified = [s for s in sentences if len(s.split()) <= 15]  # Keep shorter sentences
        return '. '.join(simplified[:3])  # Limit to 3 sentences
    
    def _add_detail(self, content: str, supporting_data: Dict[str, Any]) -> str:
        """Add detail for high detail preference"""
        details = []
        for key, value in supporting_data.items():
            if isinstance(value, (list, tuple)):
                details.append(f"{key}: {', '.join(map(str, value[:3]))}")
            else:
                details.append(f"{key}: {value}")
        
        if details:
            content += f"\n\nSupporting details: {'; '.join(details)}"
        
        return content
    
    # Template and framework loading methods
    def _load_narrative_templates(self) -> Dict[str, Any]:
        """Load narrative templates"""
        return {
            "vision_story": {
                "opening": "Imagine a future where...",
                "development": "To achieve this vision...",
                "climax": "The transformation will...",
                "resolution": "Our success will be measured by..."
            },
            "problem_solution": {
                "problem": "We face the challenge of...",
                "analysis": "Root cause analysis reveals...",
                "solution": "Our strategic response involves...",
                "implementation": "We will execute through..."
            }
        }
    
    def _load_emotional_frameworks(self) -> Dict[str, List[str]]:
        """Load emotional frameworks"""
        return {
            "executive_emotions": ["confidence", "urgency", "optimism", "determination"],
            "board_emotions": ["trust", "stability", "growth", "governance"],
            "crisis_emotions": ["concern", "resolve", "action", "recovery"]
        }
    
    # Placeholder methods for narrative element creation
    def _create_strategic_setup(self, context: StrategicContext) -> str:
        return f"In today's {context.market_conditions} environment, {context.company_position} positions us uniquely."
    
    def _create_strategic_conflict(self, context: StrategicContext) -> str:
        challenges = ", ".join(context.key_challenges[:2])
        return f"However, we must navigate {challenges} to realize our full potential."
    
    def _create_strategic_resolution(self, strategy: Dict[str, Any]) -> str:
        return f"Our {strategy.get('name', 'strategic approach')} provides the pathway to overcome these challenges."
    
    def _create_current_state(self, context: StrategicContext) -> str:
        return f"Currently, we operate from a position of {context.company_position} in {context.market_conditions}."
    
    def _create_transformation_call(self, context: StrategicContext) -> str:
        opportunities = ", ".join(context.opportunities[:2])
        return f"Market opportunities in {opportunities} call us to transform our approach."
    
    def _create_transformation_challenges(self, context: StrategicContext) -> str:
        return f"This transformation requires us to address {', '.join(context.key_challenges[:2])}."
    
    def _create_transformation_outcome(self, strategy: Dict[str, Any]) -> str:
        return f"Through {strategy.get('name', 'our strategy')}, we will emerge as {strategy.get('outcome', 'market leaders')}."
    
    def _generate_narrative_title(self, strategy: Dict[str, Any], narrative_type: NarrativeType) -> str:
        """Generate compelling narrative title"""
        strategy_name = strategy.get('name', 'Strategic Initiative')
        
        if narrative_type == NarrativeType.TRANSFORMATION_JOURNEY:
            return f"Transforming {strategy_name}: Our Path to Excellence"
        elif narrative_type == NarrativeType.GROWTH_STRATEGY:
            return f"Accelerating Growth: The {strategy_name} Opportunity"
        elif narrative_type == NarrativeType.COMPETITIVE_ADVANTAGE:
            return f"Winning in the Market: {strategy_name} Advantage"
        elif narrative_type == NarrativeType.CRISIS_RESPONSE:
            return f"Navigating Challenge: {strategy_name} Response"
        else:
            return f"Strategic Vision: {strategy_name}"


class NarrativeImpactAssessor:
    """Assesses and optimizes narrative impact"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.impact_history = []
    
    def assess_narrative_impact(self, narrative: StrategicNarrative,
                               audience: ExecutiveAudience,
                               engagement_data: Dict[str, Any]) -> NarrativeImpact:
        """Build narrative impact assessment and optimization"""
        try:
            impact = NarrativeImpact(
                id=f"impact_{narrative.id}_{audience.id}",
                narrative_id=narrative.id,
                audience_id=audience.id,
                engagement_level=engagement_data.get('engagement_level', 0.0),
                emotional_resonance=engagement_data.get('emotional_resonance', 0.0),
                message_retention=engagement_data.get('message_retention', 0.0),
                action_likelihood=engagement_data.get('action_likelihood', 0.0),
                credibility_score=engagement_data.get('credibility_score', 0.0),
                overall_impact=self._calculate_overall_impact(engagement_data),
                feedback=engagement_data.get('feedback'),
                measured_at=datetime.now()
            )
            
            self.impact_history.append(impact)
            self.logger.info(f"Assessed narrative impact: {impact.overall_impact:.2f}")
            
            return impact
            
        except Exception as e:
            self.logger.error(f"Error assessing narrative impact: {str(e)}")
            raise
    
    def optimize_narrative(self, narrative: StrategicNarrative,
                          impact: NarrativeImpact) -> Dict[str, str]:
        """Generate optimization recommendations"""
        recommendations = {}
        
        if impact.engagement_level < 0.7:
            recommendations['engagement'] = "Strengthen opening hook and emotional connection"
        
        if impact.emotional_resonance < 0.6:
            recommendations['emotion'] = "Enhance emotional arc and personal relevance"
        
        if impact.message_retention < 0.7:
            recommendations['clarity'] = "Simplify key messages and add memorable elements"
        
        if impact.action_likelihood < 0.8:
            recommendations['action'] = "Strengthen call-to-action and create urgency"
        
        if impact.credibility_score < 0.8:
            recommendations['credibility'] = "Add more supporting data and expert validation"
        
        return recommendations
    
    def _calculate_overall_impact(self, engagement_data: Dict[str, Any]) -> float:
        """Calculate overall impact score"""
        scores = [
            engagement_data.get('engagement_level', 0.0),
            engagement_data.get('emotional_resonance', 0.0),
            engagement_data.get('message_retention', 0.0),
            engagement_data.get('action_likelihood', 0.0),
            engagement_data.get('credibility_score', 0.0)
        ]
        
        return sum(scores) / len(scores)


class StrategicNarrativeSystem:
    """Main strategic narrative development system"""
    
    def __init__(self):
        self.narrative_developer = NarrativeDeveloper()
        self.impact_assessor = NarrativeImpactAssessor()
        self.logger = logging.getLogger(__name__)
    
    def create_strategic_narrative(self, strategy: Dict[str, Any],
                                  audience: ExecutiveAudience,
                                  context: StrategicContext) -> StrategicNarrative:
        """Create and personalize strategic narrative"""
        try:
            # Develop base narrative
            narrative = self.narrative_developer.develop_strategic_narrative(strategy, audience, context)
            
            # Personalize for audience
            personalized_narrative = self.narrative_developer.personalize_narrative(narrative, audience)
            
            self.logger.info(f"Created strategic narrative: {personalized_narrative.title}")
            return personalized_narrative
            
        except Exception as e:
            self.logger.error(f"Error creating strategic narrative: {str(e)}")
            raise
    
    def assess_and_optimize(self, narrative: StrategicNarrative,
                           audience: ExecutiveAudience,
                           engagement_data: Dict[str, Any]) -> Tuple[NarrativeImpact, Dict[str, str]]:
        """Assess narrative impact and provide optimization recommendations"""
        try:
            impact = self.impact_assessor.assess_narrative_impact(narrative, audience, engagement_data)
            recommendations = self.impact_assessor.optimize_narrative(narrative, impact)
            
            return impact, recommendations
            
        except Exception as e:
            self.logger.error(f"Error in assessment and optimization: {str(e)}")
            raise