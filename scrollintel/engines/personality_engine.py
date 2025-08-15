"""
Personality Engine for AGI Cognitive Architecture

This module implements personality traits, behavioral patterns, and personality-driven
decision-making that influences all cognitive processes.
"""

import random
import math
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

from scrollintel.models.emotion_models import (
    PersonalityTrait, PersonalityProfile, PersonalityInfluence,
    SocialContext, EmotionalState, EmotionType
)


class PersonalityEngine:
    """
    Implements personality traits and their influence on decision-making,
    behavior, and cognitive processes
    """
    
    def __init__(self, personality_profile: Optional[PersonalityProfile] = None):
        self.personality_profile = personality_profile or PersonalityProfile()
        self.behavioral_history: List[Dict[str, Any]] = []
        self.personality_adaptations: Dict[str, float] = {}
        self.social_learning_experiences: List[Dict[str, Any]] = []
        
        # Initialize personality-specific preferences
        self._initialize_personality_preferences()
        
        # Set up trait interaction patterns
        self.trait_interactions = self._define_trait_interactions()
    
    def influence_decision(self, decision_context: str, options: List[str], 
                          emotional_state: Optional[EmotionalState] = None) -> PersonalityInfluence:
        """
        Apply personality influence to decision-making process
        """
        # Analyze decision context
        context_analysis = self._analyze_decision_context(decision_context)
        
        # Calculate trait influences for each option
        trait_influences = self._calculate_trait_influences(
            context_analysis, options, emotional_state
        )
        
        # Determine personality bias
        personality_bias = self._determine_personality_bias(
            trait_influences, context_analysis
        )
        
        # Calculate confidence modifier based on personality
        confidence_modifier = self._calculate_confidence_modifier(
            trait_influences, emotional_state
        )
        
        # Assess risk tolerance for this decision
        risk_tolerance = self._assess_risk_tolerance(
            context_analysis, emotional_state
        )
        
        # Evaluate social considerations
        social_consideration = self._evaluate_social_considerations(
            context_analysis, options
        )
        
        # Store decision influence for learning
        self._store_decision_influence(
            decision_context, trait_influences, personality_bias
        )
        
        return PersonalityInfluence(
            decision_context=decision_context,
            trait_influences=trait_influences,
            personality_bias=personality_bias,
            confidence_modifier=confidence_modifier,
            risk_tolerance=risk_tolerance,
            social_consideration=social_consideration
        )
    
    def adapt_communication_style(self, target_audience: str, 
                                 context: SocialContext) -> Dict[str, Any]:
        """
        Adapt communication style based on personality and social context
        """
        # Analyze target audience
        audience_analysis = self._analyze_target_audience(target_audience, context)
        
        # Determine base communication style from personality
        base_style = self._determine_base_communication_style()
        
        # Adapt style for audience and context
        adapted_style = self._adapt_style_for_context(
            base_style, audience_analysis, context
        )
        
        # Generate specific communication parameters
        communication_params = self._generate_communication_parameters(
            adapted_style, audience_analysis
        )
        
        return {
            "communication_style": adapted_style,
            "tone": communication_params["tone"],
            "formality_level": communication_params["formality"],
            "directness": communication_params["directness"],
            "empathy_level": communication_params["empathy"],
            "technical_depth": communication_params["technical_depth"],
            "persuasion_approach": communication_params["persuasion"]
        }
    
    async def generate_personality_response(self, situation: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate personality-driven response to a situation"""
        # Analyze situation through personality lens
        situation_analysis = self._analyze_situation_personality(
            situation, None, None
        )
        
        # Determine primary behavioral drivers
        behavioral_drivers = self._identify_behavioral_drivers(
            situation_analysis, None
        )
        
        # Generate behavioral response
        behavioral_response = self._generate_personality_behavior(
            behavioral_drivers, situation_analysis
        )
        
        # Calculate influence score
        traits = self.personality_profile.traits
        influence_score = sum(traits.values()) / len(traits) if traits else 0.5
        
        return {
            "response": behavioral_response,
            "traits_activated": behavioral_drivers,
            "influence_score": influence_score,
            "situation_analysis": situation_analysis
        }
    
    def generate_behavioral_response(self, situation: str, 
                                   social_context: Optional[SocialContext] = None,
                                   emotional_state: Optional[EmotionalState] = None) -> Dict[str, Any]:
        """
        Generate personality-driven behavioral response to situations
        """
        # Analyze situation through personality lens
        situation_analysis = self._analyze_situation_personality(
            situation, social_context, emotional_state
        )
        
        # Determine primary behavioral drivers
        behavioral_drivers = self._identify_behavioral_drivers(
            situation_analysis, emotional_state
        )
        
        # Generate behavioral response
        behavioral_response = self._generate_personality_behavior(
            behavioral_drivers, situation_analysis
        )
        
        # Assess behavioral appropriateness
        appropriateness = self._assess_behavioral_appropriateness(
            behavioral_response, social_context, situation
        )
        
        # Calculate behavioral confidence
        confidence = self._calculate_behavioral_confidence(
            behavioral_drivers, situation_analysis
        )
        
        # Store behavioral experience for learning
        self._store_behavioral_experience(
            situation, behavioral_response, appropriateness
        )
        
        return {
            "behavioral_response": behavioral_response,
            "behavioral_drivers": behavioral_drivers,
            "appropriateness_score": appropriateness,
            "confidence": confidence,
            "personality_signature": self._get_personality_signature()
        }
    
    def learn_from_interaction(self, interaction_context: str, 
                              outcome: str, feedback: Optional[str] = None):
        """
        Learn and adapt personality expression based on interaction outcomes
        """
        # Analyze interaction outcome
        outcome_analysis = self._analyze_interaction_outcome(
            interaction_context, outcome, feedback
        )
        
        # Identify personality adjustments needed
        adjustments = self._identify_personality_adjustments(outcome_analysis)
        
        # Apply gradual personality adaptations
        self._apply_personality_adaptations(adjustments)
        
        # Store learning experience
        learning_experience = {
            "context": interaction_context,
            "outcome": outcome,
            "feedback": feedback,
            "adjustments": adjustments,
            "timestamp": datetime.now()
        }
        
        self.social_learning_experiences.append(learning_experience)
        
        # Keep recent experiences (last 50)
        if len(self.social_learning_experiences) > 50:
            self.social_learning_experiences = self.social_learning_experiences[-50:]
    
    def assess_personality_compatibility(self, other_personality: PersonalityProfile) -> Dict[str, float]:
        """
        Assess compatibility with another personality profile
        """
        compatibility_scores = {}
        
        # Compare each trait
        for trait in PersonalityTrait:
            self_value = self.personality_profile.traits.get(trait, 0.5)
            other_value = other_personality.traits.get(trait, 0.5)
            
            # Calculate compatibility based on trait interaction patterns
            compatibility = self._calculate_trait_compatibility(
                trait, self_value, other_value
            )
            compatibility_scores[trait.value] = compatibility
        
        # Calculate overall compatibility
        overall_compatibility = sum(compatibility_scores.values()) / len(compatibility_scores)
        
        # Assess communication compatibility
        communication_compatibility = self._assess_communication_compatibility(
            other_personality
        )
        
        # Assess working style compatibility
        working_style_compatibility = self._assess_working_style_compatibility(
            other_personality
        )
        
        return {
            "trait_compatibility": compatibility_scores,
            "overall_compatibility": overall_compatibility,
            "communication_compatibility": communication_compatibility,
            "working_style_compatibility": working_style_compatibility,
            "collaboration_potential": (overall_compatibility + communication_compatibility) / 2
        }
    
    def get_personality_insights(self) -> Dict[str, Any]:
        """
        Generate insights about current personality configuration and patterns
        """
        # Analyze trait patterns
        trait_analysis = self._analyze_trait_patterns()
        
        # Identify behavioral tendencies
        behavioral_tendencies = self._identify_behavioral_tendencies()
        
        # Assess personality strengths and growth areas
        strengths = self._identify_personality_strengths()
        growth_areas = self._identify_growth_areas()
        
        # Analyze recent adaptations
        recent_adaptations = self._analyze_recent_adaptations()
        
        return {
            "trait_profile": dict(self.personality_profile.traits),
            "trait_analysis": trait_analysis,
            "behavioral_tendencies": behavioral_tendencies,
            "strengths": strengths,
            "growth_areas": growth_areas,
            "recent_adaptations": recent_adaptations,
            "personality_signature": self._get_personality_signature(),
            "social_learning_progress": len(self.social_learning_experiences)
        }
    
    def _initialize_personality_preferences(self):
        """Initialize personality-specific preferences and patterns"""
        traits = self.personality_profile.traits
        
        # Set communication preferences based on traits
        if traits.get(PersonalityTrait.EXTRAVERSION, 0.5) > 0.6:
            self.personality_profile.communication_style = "expressive"
        elif traits.get(PersonalityTrait.AGREEABLENESS, 0.5) > 0.7:
            self.personality_profile.communication_style = "collaborative"
        else:
            self.personality_profile.communication_style = "analytical"
        
        # Set decision-making style
        if traits.get(PersonalityTrait.CONSCIENTIOUSNESS, 0.5) > 0.7:
            self.personality_profile.decision_making_style = "systematic"
        elif traits.get(PersonalityTrait.OPENNESS, 0.5) > 0.7:
            self.personality_profile.decision_making_style = "innovative"
        else:
            self.personality_profile.decision_making_style = "balanced"
        
        # Set social style
        extraversion = traits.get(PersonalityTrait.EXTRAVERSION, 0.5)
        agreeableness = traits.get(PersonalityTrait.AGREEABLENESS, 0.5)
        
        if extraversion > 0.6 and agreeableness > 0.6:
            self.personality_profile.social_style = "collaborative"
        elif extraversion > 0.6:
            self.personality_profile.social_style = "assertive"
        elif agreeableness > 0.6:
            self.personality_profile.social_style = "supportive"
        else:
            self.personality_profile.social_style = "independent"
    
    def _define_trait_interactions(self) -> Dict[str, Dict[str, float]]:
        """Define how personality traits interact with each other"""
        return {
            "openness_conscientiousness": {
                "creative_planning": 0.8,
                "innovative_execution": 0.7
            },
            "extraversion_agreeableness": {
                "social_leadership": 0.9,
                "team_building": 0.8
            },
            "conscientiousness_neuroticism": {
                "stress_management": -0.6,
                "perfectionism": 0.7
            }
        }
    
    def _analyze_decision_context(self, decision_context: str) -> Dict[str, Any]:
        """Analyze decision context for personality influence"""
        context_lower = decision_context.lower()
        
        analysis = {
            "complexity": 0.5,
            "social_impact": 0.3,
            "risk_level": 0.4,
            "time_pressure": 0.3,
            "innovation_required": 0.4,
            "collaboration_needed": 0.5
        }
        
        # Adjust based on context keywords
        if any(word in context_lower for word in ["complex", "difficult", "challenging"]):
            analysis["complexity"] += 0.3
        
        if any(word in context_lower for word in ["team", "group", "collaboration"]):
            analysis["collaboration_needed"] += 0.4
            analysis["social_impact"] += 0.3
        
        if any(word in context_lower for word in ["urgent", "deadline", "quickly"]):
            analysis["time_pressure"] += 0.4
        
        if any(word in context_lower for word in ["innovative", "creative", "new"]):
            analysis["innovation_required"] += 0.4
        
        return analysis
    
    def _calculate_trait_influences(self, context_analysis: Dict[str, Any], 
                                  options: List[str], 
                                  emotional_state: Optional[EmotionalState]) -> Dict[PersonalityTrait, float]:
        """Calculate how each personality trait influences the decision"""
        influences = {}
        traits = self.personality_profile.traits
        
        # Openness influence
        openness = traits.get(PersonalityTrait.OPENNESS, 0.5)
        if context_analysis["innovation_required"] > 0.5:
            influences[PersonalityTrait.OPENNESS] = openness * 0.8
        else:
            influences[PersonalityTrait.OPENNESS] = openness * 0.3
        
        # Conscientiousness influence
        conscientiousness = traits.get(PersonalityTrait.CONSCIENTIOUSNESS, 0.5)
        if context_analysis["complexity"] > 0.6:
            influences[PersonalityTrait.CONSCIENTIOUSNESS] = conscientiousness * 0.9
        else:
            influences[PersonalityTrait.CONSCIENTIOUSNESS] = conscientiousness * 0.5
        
        # Extraversion influence
        extraversion = traits.get(PersonalityTrait.EXTRAVERSION, 0.5)
        if context_analysis["collaboration_needed"] > 0.5:
            influences[PersonalityTrait.EXTRAVERSION] = extraversion * 0.7
        else:
            influences[PersonalityTrait.EXTRAVERSION] = extraversion * 0.4
        
        # Agreeableness influence
        agreeableness = traits.get(PersonalityTrait.AGREEABLENESS, 0.5)
        if context_analysis["social_impact"] > 0.5:
            influences[PersonalityTrait.AGREEABLENESS] = agreeableness * 0.8
        else:
            influences[PersonalityTrait.AGREEABLENESS] = agreeableness * 0.4
        
        # Neuroticism influence (inverse for stability)
        neuroticism = traits.get(PersonalityTrait.NEUROTICISM, 0.5)
        if context_analysis["time_pressure"] > 0.6:
            influences[PersonalityTrait.NEUROTICISM] = neuroticism * -0.6
        else:
            influences[PersonalityTrait.NEUROTICISM] = neuroticism * -0.3
        
        return influences
    
    def _determine_personality_bias(self, trait_influences: Dict[PersonalityTrait, float], 
                                  context_analysis: Dict[str, Any]) -> str:
        """Determine the primary personality bias affecting the decision"""
        max_influence = max(trait_influences.values())
        dominant_trait = max(trait_influences.keys(), key=trait_influences.get)
        
        if dominant_trait == PersonalityTrait.OPENNESS:
            return "innovation_bias"
        elif dominant_trait == PersonalityTrait.CONSCIENTIOUSNESS:
            return "systematic_bias"
        elif dominant_trait == PersonalityTrait.EXTRAVERSION:
            return "social_bias"
        elif dominant_trait == PersonalityTrait.AGREEABLENESS:
            return "harmony_bias"
        else:
            return "stability_bias"
    
    def _calculate_confidence_modifier(self, trait_influences: Dict[PersonalityTrait, float], 
                                     emotional_state: Optional[EmotionalState]) -> float:
        """Calculate how personality affects decision confidence"""
        base_confidence = 0.0
        
        # High conscientiousness increases confidence in systematic decisions
        conscientiousness = trait_influences.get(PersonalityTrait.CONSCIENTIOUSNESS, 0.0)
        base_confidence += conscientiousness * 0.3
        
        # Low neuroticism increases overall confidence
        neuroticism = trait_influences.get(PersonalityTrait.NEUROTICISM, 0.0)
        base_confidence -= abs(neuroticism) * 0.2
        
        # Emotional state affects confidence
        if emotional_state:
            if emotional_state.primary_emotion == EmotionType.TRUST:
                base_confidence += 0.2
            elif emotional_state.primary_emotion == EmotionType.FEAR:
                base_confidence -= 0.3
        
        return max(-0.5, min(0.5, base_confidence))
    
    def _assess_risk_tolerance(self, context_analysis: Dict[str, Any], 
                             emotional_state: Optional[EmotionalState]) -> float:
        """Assess risk tolerance based on personality and context"""
        traits = self.personality_profile.traits
        
        # Base risk tolerance from personality
        openness = traits.get(PersonalityTrait.OPENNESS, 0.5)
        conscientiousness = traits.get(PersonalityTrait.CONSCIENTIOUSNESS, 0.5)
        neuroticism = traits.get(PersonalityTrait.NEUROTICISM, 0.5)
        
        risk_tolerance = (openness * 0.4) + ((1 - conscientiousness) * 0.3) + ((1 - neuroticism) * 0.3)
        
        # Adjust for emotional state
        if emotional_state:
            if emotional_state.primary_emotion == EmotionType.FEAR:
                risk_tolerance *= 0.7
            elif emotional_state.primary_emotion == EmotionType.ANTICIPATION:
                risk_tolerance *= 1.2
        
        return max(0.0, min(1.0, risk_tolerance))
    
    def _evaluate_social_considerations(self, context_analysis: Dict[str, Any], 
                                      options: List[str]) -> float:
        """Evaluate how much social factors should influence the decision"""
        traits = self.personality_profile.traits
        
        agreeableness = traits.get(PersonalityTrait.AGREEABLENESS, 0.5)
        empathy = traits.get(PersonalityTrait.EMPATHY, 0.5)
        
        social_consideration = (agreeableness * 0.6) + (empathy * 0.4)
        
        # Adjust based on context
        if context_analysis["social_impact"] > 0.5:
            social_consideration *= 1.3
        
        return max(0.0, min(1.0, social_consideration))
    
    def _store_decision_influence(self, decision_context: str, 
                                trait_influences: Dict[PersonalityTrait, float], 
                                personality_bias: str):
        """Store decision influence for learning and analysis"""
        decision_record = {
            "context": decision_context,
            "trait_influences": trait_influences,
            "personality_bias": personality_bias,
            "timestamp": datetime.now()
        }
        
        self.behavioral_history.append(decision_record)
        
        # Keep recent history (last 100 decisions)
        if len(self.behavioral_history) > 100:
            self.behavioral_history = self.behavioral_history[-100:]
    
    def _analyze_target_audience(self, target_audience: str, 
                               context: SocialContext) -> Dict[str, Any]:
        """Analyze target audience for communication adaptation"""
        return {
            "audience_type": "professional",
            "expertise_level": "high",
            "formality_expected": 0.7,
            "technical_depth_preferred": 0.8,
            "relationship_type": "collaborative"
        }
    
    def _determine_base_communication_style(self) -> Dict[str, float]:
        """Determine base communication style from personality"""
        traits = self.personality_profile.traits
        
        return {
            "directness": traits.get(PersonalityTrait.ASSERTIVENESS, 0.5),
            "warmth": traits.get(PersonalityTrait.AGREEABLENESS, 0.5),
            "detail_orientation": traits.get(PersonalityTrait.CONSCIENTIOUSNESS, 0.5),
            "creativity": traits.get(PersonalityTrait.OPENNESS, 0.5),
            "empathy": traits.get(PersonalityTrait.EMPATHY, 0.5)
        }
    
    def _adapt_style_for_context(self, base_style: Dict[str, float], 
                               audience_analysis: Dict[str, Any], 
                               context: SocialContext) -> Dict[str, float]:
        """Adapt communication style for specific context"""
        adapted_style = base_style.copy()
        
        # Adjust for professional context
        if context.social_setting == "professional":
            adapted_style["directness"] = min(1.0, adapted_style["directness"] + 0.2)
            adapted_style["detail_orientation"] = min(1.0, adapted_style["detail_orientation"] + 0.1)
        
        # Adjust for audience expertise
        if audience_analysis["expertise_level"] == "high":
            adapted_style["detail_orientation"] = min(1.0, adapted_style["detail_orientation"] + 0.2)
        
        return adapted_style
    
    def _generate_communication_parameters(self, adapted_style: Dict[str, float], 
                                         audience_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate specific communication parameters"""
        return {
            "tone": "professional" if adapted_style["directness"] > 0.6 else "collaborative",
            "formality": min(1.0, adapted_style["detail_orientation"] + 0.3),
            "directness": adapted_style["directness"],
            "empathy": adapted_style["empathy"],
            "technical_depth": audience_analysis["technical_depth_preferred"],
            "persuasion": "logical" if adapted_style["detail_orientation"] > 0.6 else "emotional"
        }
    
    def _analyze_situation_personality(self, situation: str, 
                                     social_context: Optional[SocialContext], 
                                     emotional_state: Optional[EmotionalState]) -> Dict[str, Any]:
        """Analyze situation through personality lens"""
        return {
            "situation_type": "professional_challenge",
            "personality_relevance": 0.8,
            "trait_activation": ["conscientiousness", "openness"],
            "behavioral_options": ["systematic_approach", "creative_solution", "collaborative_effort"]
        }
    
    def _identify_behavioral_drivers(self, situation_analysis: Dict[str, Any], 
                                   emotional_state: Optional[EmotionalState]) -> List[str]:
        """Identify primary drivers for behavioral response"""
        drivers = []
        traits = self.personality_profile.traits
        
        if traits.get(PersonalityTrait.CONSCIENTIOUSNESS, 0.5) > 0.6:
            drivers.append("systematic_approach")
        
        if traits.get(PersonalityTrait.OPENNESS, 0.5) > 0.6:
            drivers.append("creative_thinking")
        
        if traits.get(PersonalityTrait.AGREEABLENESS, 0.5) > 0.6:
            drivers.append("collaborative_solution")
        
        return drivers
    
    def _generate_personality_behavior(self, behavioral_drivers: List[str], 
                                     situation_analysis: Dict[str, Any]) -> str:
        """Generate specific behavioral response based on personality"""
        if "systematic_approach" in behavioral_drivers:
            return "Approach systematically with detailed analysis and structured planning"
        elif "creative_thinking" in behavioral_drivers:
            return "Explore innovative solutions and think outside conventional approaches"
        elif "collaborative_solution" in behavioral_drivers:
            return "Engage stakeholders and build consensus for collaborative solutions"
        else:
            return "Apply balanced approach considering multiple perspectives"
    
    def _assess_behavioral_appropriateness(self, behavioral_response: str, 
                                         social_context: Optional[SocialContext], 
                                         situation: str) -> float:
        """Assess appropriateness of behavioral response"""
        base_appropriateness = 0.8
        
        # Adjust based on social context
        if social_context and social_context.social_setting == "professional":
            if "collaborative" in behavioral_response.lower():
                base_appropriateness += 0.1
        
        return min(1.0, base_appropriateness)
    
    def _calculate_behavioral_confidence(self, behavioral_drivers: List[str], 
                                       situation_analysis: Dict[str, Any]) -> float:
        """Calculate confidence in behavioral response"""
        base_confidence = 0.7
        
        # More drivers = higher confidence
        base_confidence += len(behavioral_drivers) * 0.1
        
        return min(1.0, base_confidence)
    
    def _store_behavioral_experience(self, situation: str, behavioral_response: str, 
                                   appropriateness: float):
        """Store behavioral experience for learning"""
        experience = {
            "situation": situation,
            "response": behavioral_response,
            "appropriateness": appropriateness,
            "timestamp": datetime.now()
        }
        
        self.behavioral_history.append(experience)
    
    def _get_personality_signature(self) -> str:
        """Get a signature representing the current personality configuration"""
        traits = self.personality_profile.traits
        
        dominant_traits = sorted(traits.items(), key=lambda x: x[1], reverse=True)[:3]
        signature_parts = [f"{trait.value}:{value:.2f}" for trait, value in dominant_traits]
        
        return "|".join(signature_parts)
    
    def _analyze_interaction_outcome(self, interaction_context: str, 
                                   outcome: str, feedback: Optional[str]) -> Dict[str, Any]:
        """Analyze interaction outcome for learning"""
        return {
            "success_level": 0.8 if "success" in outcome.lower() else 0.4,
            "feedback_sentiment": "positive" if feedback and "good" in feedback.lower() else "neutral",
            "learning_opportunities": ["communication_style", "approach_method"]
        }
    
    def _identify_personality_adjustments(self, outcome_analysis: Dict[str, Any]) -> Dict[str, float]:
        """Identify needed personality adjustments"""
        adjustments = {}
        
        if outcome_analysis["success_level"] < 0.5:
            # Slight adjustment toward more agreeableness
            adjustments["agreeableness"] = 0.05
        
        return adjustments
    
    def _apply_personality_adaptations(self, adjustments: Dict[str, float]):
        """Apply gradual personality adaptations"""
        for trait_name, adjustment in adjustments.items():
            try:
                trait = PersonalityTrait(trait_name)
                current_value = self.personality_profile.traits.get(trait, 0.5)
                new_value = max(0.0, min(1.0, current_value + adjustment))
                self.personality_profile.traits[trait] = new_value
                
                # Store adaptation
                self.personality_adaptations[trait_name] = self.personality_adaptations.get(trait_name, 0) + adjustment
            except ValueError:
                continue
    
    def _calculate_trait_compatibility(self, trait: PersonalityTrait, 
                                     self_value: float, other_value: float) -> float:
        """Calculate compatibility for a specific trait"""
        # Some traits work better when similar, others when complementary
        if trait in [PersonalityTrait.AGREEABLENESS, PersonalityTrait.CONSCIENTIOUSNESS]:
            # Similar values work better
            return 1.0 - abs(self_value - other_value)
        elif trait in [PersonalityTrait.EXTRAVERSION, PersonalityTrait.ASSERTIVENESS]:
            # Complementary values can work well
            return 0.7 + 0.3 * (1.0 - abs(self_value - other_value))
        else:
            # Balanced approach
            return 0.8 - 0.3 * abs(self_value - other_value)
    
    def _assess_communication_compatibility(self, other_personality: PersonalityProfile) -> float:
        """Assess communication compatibility"""
        self_communication = self._determine_base_communication_style()
        
        # Simulate other's communication style
        other_traits = other_personality.traits
        other_communication = {
            "directness": other_traits.get(PersonalityTrait.ASSERTIVENESS, 0.5),
            "warmth": other_traits.get(PersonalityTrait.AGREEABLENESS, 0.5),
            "detail_orientation": other_traits.get(PersonalityTrait.CONSCIENTIOUSNESS, 0.5),
            "empathy": other_traits.get(PersonalityTrait.EMPATHY, 0.5)
        }
        
        # Calculate compatibility
        compatibility_sum = 0
        for key in self_communication:
            if key in other_communication:
                compatibility_sum += 1.0 - abs(self_communication[key] - other_communication[key])
        
        return compatibility_sum / len(self_communication)
    
    def _assess_working_style_compatibility(self, other_personality: PersonalityProfile) -> float:
        """Assess working style compatibility"""
        self_conscientiousness = self.personality_profile.traits.get(PersonalityTrait.CONSCIENTIOUSNESS, 0.5)
        other_conscientiousness = other_personality.traits.get(PersonalityTrait.CONSCIENTIOUSNESS, 0.5)
        
        self_openness = self.personality_profile.traits.get(PersonalityTrait.OPENNESS, 0.5)
        other_openness = other_personality.traits.get(PersonalityTrait.OPENNESS, 0.5)
        
        # Similar conscientiousness and openness work well together
        conscientiousness_compatibility = 1.0 - abs(self_conscientiousness - other_conscientiousness)
        openness_compatibility = 1.0 - abs(self_openness - other_openness)
        
        return (conscientiousness_compatibility + openness_compatibility) / 2
    
    def _analyze_trait_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in personality traits"""
        traits = self.personality_profile.traits
        
        # Identify dominant traits
        dominant_traits = [trait for trait, value in traits.items() if value > 0.7]
        
        # Identify trait clusters
        analytical_cluster = (
            traits.get(PersonalityTrait.CONSCIENTIOUSNESS, 0.5) + 
            traits.get(PersonalityTrait.OPENNESS, 0.5)
        ) / 2
        
        social_cluster = (
            traits.get(PersonalityTrait.EXTRAVERSION, 0.5) + 
            traits.get(PersonalityTrait.AGREEABLENESS, 0.5) + 
            traits.get(PersonalityTrait.EMPATHY, 0.5)
        ) / 3
        
        return {
            "dominant_traits": [trait.value for trait in dominant_traits],
            "analytical_orientation": analytical_cluster,
            "social_orientation": social_cluster,
            "stability": 1.0 - traits.get(PersonalityTrait.NEUROTICISM, 0.5)
        }
    
    def _identify_behavioral_tendencies(self) -> List[str]:
        """Identify key behavioral tendencies"""
        tendencies = []
        traits = self.personality_profile.traits
        
        if traits.get(PersonalityTrait.CONSCIENTIOUSNESS, 0.5) > 0.7:
            tendencies.append("systematic_planning")
        
        if traits.get(PersonalityTrait.OPENNESS, 0.5) > 0.7:
            tendencies.append("innovative_thinking")
        
        if traits.get(PersonalityTrait.AGREEABLENESS, 0.5) > 0.7:
            tendencies.append("collaborative_approach")
        
        if traits.get(PersonalityTrait.EXTRAVERSION, 0.5) > 0.7:
            tendencies.append("social_engagement")
        
        return tendencies
    
    def _identify_personality_strengths(self) -> List[str]:
        """Identify personality strengths"""
        strengths = []
        traits = self.personality_profile.traits
        
        if traits.get(PersonalityTrait.CONSCIENTIOUSNESS, 0.5) > 0.7:
            strengths.append("reliable_execution")
        
        if traits.get(PersonalityTrait.OPENNESS, 0.5) > 0.7:
            strengths.append("creative_problem_solving")
        
        if traits.get(PersonalityTrait.EMPATHY, 0.5) > 0.7:
            strengths.append("interpersonal_understanding")
        
        if traits.get(PersonalityTrait.ASSERTIVENESS, 0.5) > 0.7:
            strengths.append("decisive_leadership")
        
        return strengths
    
    def _identify_growth_areas(self) -> List[str]:
        """Identify areas for personality growth"""
        growth_areas = []
        traits = self.personality_profile.traits
        
        if traits.get(PersonalityTrait.NEUROTICISM, 0.5) > 0.6:
            growth_areas.append("stress_management")
        
        if traits.get(PersonalityTrait.AGREEABLENESS, 0.5) < 0.4:
            growth_areas.append("collaborative_skills")
        
        if traits.get(PersonalityTrait.OPENNESS, 0.5) < 0.4:
            growth_areas.append("adaptability")
        
        return growth_areas
    
    def _analyze_recent_adaptations(self) -> Dict[str, float]:
        """Analyze recent personality adaptations"""
        return dict(self.personality_adaptations)