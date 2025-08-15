"""
Consciousness-Decision Integration System for AGI cognitive architecture.
Integrates consciousness simulation with decision-making processes.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from enum import Enum

from ..models.consciousness_models import (
    ConsciousnessState, CognitiveContext, Goal, Thought, Experience,
    AwarenessType, ConsciousnessLevel
)
from ..engines.consciousness_engine import ConsciousnessEngine, AwarenessEngine
from ..engines.intentionality_engine import IntentionalityEngine


logger = logging.getLogger(__name__)


class DecisionType(Enum):
    """Types of decisions that can be made"""
    STRATEGIC = "strategic"
    TACTICAL = "tactical"
    OPERATIONAL = "operational"
    CREATIVE = "creative"
    ANALYTICAL = "analytical"
    ETHICAL = "ethical"


class DecisionUrgency(Enum):
    """Urgency levels for decisions"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ConsciousnessDecisionIntegrator:
    """Integrates consciousness simulation with decision-making"""
    
    def __init__(self):
        self.consciousness_engine = ConsciousnessEngine()
        self.awareness_engine = AwarenessEngine(self.consciousness_engine)
        self.intentionality_engine = IntentionalityEngine()
        self.decision_history: List[Dict[str, Any]] = []
        self.consciousness_decision_mappings: Dict[str, Any] = {}
        
    async def make_conscious_decision(self, decision_context: Dict[str, Any],
                                    options: List[Dict[str, Any]],
                                    decision_type: DecisionType = DecisionType.STRATEGIC) -> Dict[str, Any]:
        """Make a decision using consciousness-guided process"""
        logger.info(f"Making conscious decision: {decision_type.value}")
        
        # Create cognitive context
        cognitive_context = self._create_cognitive_context(decision_context)
        
        # Simulate awareness for decision context
        awareness_state = await self.consciousness_engine.simulate_awareness(cognitive_context)
        
        # Generate intentional state if goal-directed
        intentional_state = None
        if "goal" in decision_context:
            goal = self._extract_goal_from_context(decision_context)
            intentional_state = await self.intentionality_engine.generate_intentional_state(
                goal, self.consciousness_engine.current_state
            )
        
        # Evaluate options using consciousness
        option_evaluations = await self._evaluate_options_consciously(
            options, awareness_state, intentional_state, decision_type
        )
        
        # Make decision based on conscious evaluation
        decision = await self._make_final_decision(
            option_evaluations, awareness_state, intentional_state
        )
        
        # Reflect on decision process
        decision_reflection = await self._reflect_on_decision_process(
            decision, decision_context, awareness_state
        )
        
        # Store decision in history
        decision_record = {
            "decision_id": decision["decision_id"],
            "decision_type": decision_type.value,
            "context": decision_context,
            "options_evaluated": len(options),
            "chosen_option": decision["chosen_option"],
            "confidence": decision["confidence"],
            "consciousness_level": awareness_state.level.value,
            "awareness_types": [at.value for at in awareness_state.awareness_types],
            "reflection": decision_reflection,
            "timestamp": datetime.now()
        }
        
        self.decision_history.append(decision_record)
        
        return decision
    
    async def integrate_consciousness_with_planning(self, planning_context: Dict[str, Any]) -> Dict[str, Any]:
        """Integrate consciousness simulation with strategic planning"""
        logger.info("Integrating consciousness with planning")
        
        # Create cognitive context for planning
        cognitive_context = self._create_cognitive_context(planning_context)
        
        # Elevate consciousness level for complex planning
        cognitive_context.complexity_level = max(0.8, cognitive_context.complexity_level)
        
        # Simulate meta-conscious awareness
        awareness_state = await self.consciousness_engine.simulate_awareness(cognitive_context)
        
        # Generate planning intentions
        planning_goals = self._extract_planning_goals(planning_context)
        planning_intentions = []
        
        for goal in planning_goals:
            intention = await self.intentionality_engine.generate_intentional_state(
                goal, self.consciousness_engine.current_state
            )
            planning_intentions.append(intention)
        
        # Perform conscious planning analysis
        planning_analysis = await self._perform_conscious_planning_analysis(
            planning_context, awareness_state, planning_intentions
        )
        
        # Generate conscious insights
        conscious_insights = await self._generate_conscious_planning_insights(
            planning_analysis, awareness_state
        )
        
        return {
            "planning_analysis": planning_analysis,
            "conscious_insights": conscious_insights,
            "awareness_level": awareness_state.level.value,
            "consciousness_coherence": self.consciousness_engine.current_state.consciousness_coherence,
            "recommended_actions": await self._recommend_conscious_actions(
                planning_analysis, conscious_insights
            )
        }
    
    async def consciousness_guided_problem_solving(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """Use consciousness to guide problem-solving process"""
        logger.info("Applying consciousness-guided problem solving")
        
        # Create problem-solving context
        cognitive_context = CognitiveContext(
            situation=problem.get("description", ""),
            complexity_level=problem.get("complexity", 0.7),
            time_pressure=problem.get("urgency", 0.5),
            available_resources=problem.get("resources", []),
            constraints=problem.get("constraints", [])
        )
        
        # Simulate consciousness for problem context
        awareness_state = await self.consciousness_engine.simulate_awareness(cognitive_context)
        
        # Generate problem-solving thoughts
        problem_thoughts = await self._generate_problem_solving_thoughts(problem, awareness_state)
        
        # Apply meta-cognitive analysis to thoughts
        meta_insights = []
        for thought in problem_thoughts:
            insight = await self.consciousness_engine.process_meta_cognition(thought)
            meta_insights.append(insight)
        
        # Synthesize solution using conscious processing
        solution = await self._synthesize_conscious_solution(
            problem, problem_thoughts, meta_insights, awareness_state
        )
        
        # Validate solution through self-reflection
        solution_reflection = await self._reflect_on_solution(solution, problem, awareness_state)
        
        return {
            "problem_id": problem.get("id", "unknown"),
            "solution": solution,
            "problem_thoughts": [t.content for t in problem_thoughts],
            "meta_insights": [i.description for i in meta_insights],
            "consciousness_level": awareness_state.level.value,
            "solution_confidence": solution.get("confidence", 0.0),
            "reflection": solution_reflection,
            "recommended_validation": await self._recommend_solution_validation(solution)
        }
    
    async def adaptive_consciousness_response(self, stimulus: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt consciousness response to different stimuli"""
        logger.info("Generating adaptive consciousness response")
        
        # Analyze stimulus characteristics
        stimulus_analysis = self._analyze_stimulus(stimulus)
        
        # Adapt consciousness level based on stimulus
        adapted_consciousness_level = self._determine_consciousness_level(stimulus_analysis)
        
        # Create adaptive cognitive context
        adaptive_context = CognitiveContext(
            situation=stimulus.get("description", ""),
            complexity_level=stimulus_analysis["complexity"],
            time_pressure=stimulus_analysis["urgency"],
            available_resources=stimulus.get("available_resources", []),
            constraints=stimulus.get("constraints", [])
        )
        
        # Simulate adaptive awareness
        adaptive_awareness = await self.consciousness_engine.simulate_awareness(adaptive_context)
        adaptive_awareness.level = adapted_consciousness_level
        
        # Generate adaptive response
        adaptive_response = await self._generate_adaptive_response(
            stimulus, adaptive_awareness, stimulus_analysis
        )
        
        # Monitor consciousness adaptation
        adaptation_monitoring = await self.consciousness_engine.recursive_self_monitor()
        
        return {
            "stimulus_id": stimulus.get("id", "unknown"),
            "stimulus_analysis": stimulus_analysis,
            "adapted_consciousness_level": adapted_consciousness_level.value,
            "adaptive_response": adaptive_response,
            "consciousness_monitoring": adaptation_monitoring,
            "adaptation_effectiveness": await self._assess_adaptation_effectiveness(
                stimulus, adaptive_response, adaptation_monitoring
            )
        }
    
    def _create_cognitive_context(self, context_data: Dict[str, Any]) -> CognitiveContext:
        """Create cognitive context from context data"""
        return CognitiveContext(
            situation=context_data.get("situation", ""),
            environment=context_data.get("environment", {}),
            available_resources=context_data.get("resources", []),
            constraints=context_data.get("constraints", []),
            time_pressure=context_data.get("urgency", 0.5),
            complexity_level=context_data.get("complexity", 0.5),
            stakeholders=context_data.get("stakeholders", [])
        )
    
    def _extract_goal_from_context(self, context: Dict[str, Any]) -> Goal:
        """Extract goal from decision context"""
        goal_data = context.get("goal", {})
        
        return Goal(
            description=goal_data.get("description", "Achieve decision objective"),
            priority=goal_data.get("priority", 0.7),
            context=goal_data.get("context", {})
        )
    
    async def _evaluate_options_consciously(self, options: List[Dict[str, Any]],
                                          awareness_state: Any,
                                          intentional_state: Optional[Any],
                                          decision_type: DecisionType) -> List[Dict[str, Any]]:
        """Evaluate decision options using consciousness"""
        evaluations = []
        
        for i, option in enumerate(options):
            # Create thought about option
            option_thought = Thought(
                content=f"Evaluating option: {option.get('description', f'Option {i+1}')}",
                thought_type=decision_type.value,
                confidence=0.7,
                source="conscious_evaluation"
            )
            
            # Process meta-cognitively
            meta_insight = await self.consciousness_engine.process_meta_cognition(option_thought)
            
            # Evaluate against awareness and intentions
            awareness_score = self._score_option_against_awareness(option, awareness_state)
            intention_score = self._score_option_against_intentions(option, intentional_state)
            
            # Calculate overall evaluation
            overall_score = (awareness_score + intention_score + meta_insight.effectiveness_score) / 3
            
            evaluation = {
                "option_index": i,
                "option": option,
                "awareness_score": awareness_score,
                "intention_score": intention_score,
                "meta_cognitive_score": meta_insight.effectiveness_score,
                "overall_score": overall_score,
                "evaluation_rationale": self._generate_evaluation_rationale(
                    option, awareness_score, intention_score, meta_insight
                )
            }
            
            evaluations.append(evaluation)
        
        return evaluations
    
    async def _make_final_decision(self, evaluations: List[Dict[str, Any]],
                                 awareness_state: Any,
                                 intentional_state: Optional[Any]) -> Dict[str, Any]:
        """Make final decision based on evaluations"""
        # Sort by overall score
        sorted_evaluations = sorted(evaluations, key=lambda e: e["overall_score"], reverse=True)
        
        best_option = sorted_evaluations[0]
        
        # Calculate decision confidence
        confidence = self._calculate_decision_confidence(sorted_evaluations, awareness_state)
        
        decision = {
            "decision_id": f"decision_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "chosen_option": best_option["option"],
            "option_index": best_option["option_index"],
            "confidence": confidence,
            "evaluation_scores": {
                "awareness": best_option["awareness_score"],
                "intention": best_option["intention_score"],
                "meta_cognitive": best_option["meta_cognitive_score"],
                "overall": best_option["overall_score"]
            },
            "decision_rationale": best_option["evaluation_rationale"],
            "alternative_options": [e["option"] for e in sorted_evaluations[1:3]]  # Top 2 alternatives
        }
        
        return decision
    
    async def _reflect_on_decision_process(self, decision: Dict[str, Any],
                                         context: Dict[str, Any],
                                         awareness_state: Any) -> Dict[str, Any]:
        """Reflect on the decision-making process"""
        # Create experience from decision
        decision_experience = Experience(
            description=f"Made decision: {decision['chosen_option'].get('description', 'Unknown')}",
            experience_type="decision_making",
            emotional_valence=decision["confidence"] - 0.5,  # Convert confidence to valence
            significance=context.get("importance", 0.5),
            context={"decision_context": context, "awareness_level": awareness_state.level.value}
        )
        
        # Reflect on experience
        reflection = await self.consciousness_engine.reflect_on_experience(decision_experience)
        
        return {
            "reflection_insights": reflection.insights,
            "self_assessment": reflection.self_assessment,
            "improvement_areas": reflection.areas_for_improvement,
            "decision_learning": self._extract_decision_learning(decision, reflection)
        }
    
    def _extract_planning_goals(self, planning_context: Dict[str, Any]) -> List[Goal]:
        """Extract goals from planning context"""
        goals = []
        
        # Extract explicit goals
        if "goals" in planning_context:
            for goal_data in planning_context["goals"]:
                goal = Goal(
                    description=goal_data.get("description", ""),
                    priority=goal_data.get("priority", 0.5),
                    context=goal_data.get("context", {})
                )
                goals.append(goal)
        
        # Create implicit goal from planning objective
        if "objective" in planning_context:
            implicit_goal = Goal(
                description=f"Achieve planning objective: {planning_context['objective']}",
                priority=0.8,
                context={"implicit": True, "source": "planning_objective"}
            )
            goals.append(implicit_goal)
        
        return goals
    
    async def _perform_conscious_planning_analysis(self, planning_context: Dict[str, Any],
                                                 awareness_state: Any,
                                                 planning_intentions: List[Any]) -> Dict[str, Any]:
        """Perform conscious analysis of planning situation"""
        analysis = {
            "situational_assessment": await self._assess_planning_situation(planning_context, awareness_state),
            "goal_analysis": self._analyze_planning_goals(planning_intentions),
            "resource_evaluation": self._evaluate_planning_resources(planning_context),
            "constraint_analysis": self._analyze_planning_constraints(planning_context),
            "opportunity_identification": await self._identify_planning_opportunities(
                planning_context, awareness_state
            ),
            "risk_assessment": self._assess_planning_risks(planning_context)
        }
        
        return analysis
    
    async def _generate_conscious_planning_insights(self, planning_analysis: Dict[str, Any],
                                                  awareness_state: Any) -> List[str]:
        """Generate insights from conscious planning analysis"""
        insights = []
        
        # Situational insights
        if planning_analysis["situational_assessment"]["complexity"] > 0.8:
            insights.append("High complexity situation requires careful decomposition")
        
        # Goal insights
        if len(planning_analysis["goal_analysis"]["conflicting_goals"]) > 0:
            insights.append("Goal conflicts detected - prioritization needed")
        
        # Resource insights
        if planning_analysis["resource_evaluation"]["adequacy"] < 0.6:
            insights.append("Resource constraints may limit plan execution")
        
        # Opportunity insights
        if len(planning_analysis["opportunity_identification"]) > 3:
            insights.append("Multiple opportunities available for strategic advantage")
        
        # Meta-cognitive insights
        if awareness_state.level == ConsciousnessLevel.META_CONSCIOUS:
            insights.append("Meta-conscious analysis reveals deeper strategic patterns")
        
        return insights
    
    async def _recommend_conscious_actions(self, planning_analysis: Dict[str, Any],
                                         conscious_insights: List[str]) -> List[str]:
        """Recommend actions based on conscious analysis"""
        actions = []
        
        # Actions based on analysis
        if planning_analysis["resource_evaluation"]["adequacy"] < 0.7:
            actions.append("Secure additional resources before execution")
        
        if len(planning_analysis["constraint_analysis"]["critical_constraints"]) > 0:
            actions.append("Address critical constraints first")
        
        # Actions based on insights
        if "Goal conflicts detected" in " ".join(conscious_insights):
            actions.append("Resolve goal conflicts through stakeholder alignment")
        
        if "Meta-conscious analysis" in " ".join(conscious_insights):
            actions.append("Leverage meta-cognitive insights for strategic advantage")
        
        # Default strategic actions
        actions.extend([
            "Develop detailed implementation roadmap",
            "Establish monitoring and feedback mechanisms",
            "Create contingency plans for identified risks"
        ])
        
        return actions
    
    async def _generate_problem_solving_thoughts(self, problem: Dict[str, Any],
                                               awareness_state: Any) -> List[Thought]:
        """Generate thoughts for problem-solving"""
        thoughts = []
        
        # Problem analysis thought
        analysis_thought = Thought(
            content=f"Analyzing problem: {problem.get('description', 'Unknown problem')}",
            thought_type="analytical",
            confidence=0.8,
            source="problem_analysis"
        )
        thoughts.append(analysis_thought)
        
        # Solution exploration thought
        exploration_thought = Thought(
            content="Exploring potential solution approaches",
            thought_type="creative",
            confidence=0.7,
            source="solution_exploration"
        )
        thoughts.append(exploration_thought)
        
        # Constraint consideration thought
        constraint_thought = Thought(
            content="Considering constraints and limitations",
            thought_type="analytical",
            confidence=0.8,
            source="constraint_analysis"
        )
        thoughts.append(constraint_thought)
        
        # Meta-cognitive thought if awareness level is high
        if awareness_state.level in [ConsciousnessLevel.SELF_AWARE, ConsciousnessLevel.META_CONSCIOUS]:
            meta_thought = Thought(
                content="Reflecting on problem-solving approach effectiveness",
                thought_type="meta_cognitive",
                confidence=0.6,
                source="meta_analysis"
            )
            thoughts.append(meta_thought)
        
        return thoughts
    
    async def _synthesize_conscious_solution(self, problem: Dict[str, Any],
                                           thoughts: List[Thought],
                                           meta_insights: List[Any],
                                           awareness_state: Any) -> Dict[str, Any]:
        """Synthesize solution using conscious processing"""
        # Combine insights from thoughts and meta-cognition
        solution_elements = []
        
        for thought in thoughts:
            if thought.thought_type == "analytical":
                solution_elements.append(f"Analysis: {thought.content}")
            elif thought.thought_type == "creative":
                solution_elements.append(f"Creative approach: {thought.content}")
        
        for insight in meta_insights:
            if insight.effectiveness_score > 0.7:
                solution_elements.extend(insight.improvement_suggestions)
        
        # Generate solution based on consciousness level
        if awareness_state.level == ConsciousnessLevel.META_CONSCIOUS:
            solution_approach = "meta_cognitive_synthesis"
            confidence_boost = 0.2
        elif awareness_state.level == ConsciousnessLevel.SELF_AWARE:
            solution_approach = "self_aware_integration"
            confidence_boost = 0.1
        else:
            solution_approach = "conscious_analysis"
            confidence_boost = 0.0
        
        base_confidence = sum(t.confidence for t in thoughts) / len(thoughts) if thoughts else 0.5
        
        solution = {
            "approach": solution_approach,
            "solution_elements": solution_elements,
            "confidence": min(1.0, base_confidence + confidence_boost),
            "consciousness_level": awareness_state.level.value,
            "synthesis_method": "conscious_integration",
            "validation_needed": base_confidence < 0.8
        }
        
        return solution
    
    async def _reflect_on_solution(self, solution: Dict[str, Any],
                                 problem: Dict[str, Any],
                                 awareness_state: Any) -> Dict[str, Any]:
        """Reflect on the generated solution"""
        # Create experience from solution generation
        solution_experience = Experience(
            description=f"Generated solution using {solution['approach']}",
            experience_type="problem_solving",
            emotional_valence=solution["confidence"] - 0.5,
            significance=problem.get("importance", 0.5),
            context={"solution_confidence": solution["confidence"]}
        )
        
        # Reflect on experience
        reflection = await self.consciousness_engine.reflect_on_experience(solution_experience)
        
        return {
            "solution_quality_assessment": reflection.self_assessment,
            "solution_insights": reflection.insights,
            "improvement_opportunities": reflection.areas_for_improvement,
            "solution_strengths": reflection.strengths_identified
        }
    
    async def _recommend_solution_validation(self, solution: Dict[str, Any]) -> List[str]:
        """Recommend validation steps for solution"""
        validations = []
        
        if solution["confidence"] < 0.7:
            validations.append("Conduct thorough solution review")
            validations.append("Seek expert validation")
        
        if solution["validation_needed"]:
            validations.append("Test solution with pilot implementation")
            validations.append("Validate assumptions with stakeholders")
        
        validations.extend([
            "Monitor solution effectiveness",
            "Gather feedback for continuous improvement",
            "Document lessons learned"
        ])
        
        return validations
    
    def _analyze_stimulus(self, stimulus: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze stimulus characteristics"""
        return {
            "complexity": stimulus.get("complexity", 0.5),
            "urgency": stimulus.get("urgency", 0.5),
            "novelty": stimulus.get("novelty", 0.5),
            "emotional_impact": stimulus.get("emotional_impact", 0.5),
            "strategic_importance": stimulus.get("importance", 0.5)
        }
    
    def _determine_consciousness_level(self, stimulus_analysis: Dict[str, Any]) -> ConsciousnessLevel:
        """Determine appropriate consciousness level for stimulus"""
        complexity = stimulus_analysis["complexity"]
        novelty = stimulus_analysis["novelty"]
        importance = stimulus_analysis["strategic_importance"]
        
        consciousness_score = (complexity + novelty + importance) / 3
        
        if consciousness_score > 0.8:
            return ConsciousnessLevel.META_CONSCIOUS
        elif consciousness_score > 0.6:
            return ConsciousnessLevel.SELF_AWARE
        else:
            return ConsciousnessLevel.CONSCIOUS
    
    async def _generate_adaptive_response(self, stimulus: Dict[str, Any],
                                        adaptive_awareness: Any,
                                        stimulus_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate adaptive response to stimulus"""
        response_type = self._determine_response_type(stimulus_analysis)
        
        response = {
            "response_type": response_type,
            "consciousness_level": adaptive_awareness.level.value,
            "response_elements": await self._generate_response_elements(
                stimulus, adaptive_awareness, response_type
            ),
            "adaptation_strategy": self._determine_adaptation_strategy(stimulus_analysis),
            "confidence": self._calculate_response_confidence(stimulus_analysis, adaptive_awareness)
        }
        
        return response
    
    async def _assess_adaptation_effectiveness(self, stimulus: Dict[str, Any],
                                             response: Dict[str, Any],
                                             monitoring: Dict[str, Any]) -> float:
        """Assess effectiveness of consciousness adaptation"""
        effectiveness_factors = []
        
        # Response appropriateness
        if response["confidence"] > 0.7:
            effectiveness_factors.append(0.8)
        
        # Consciousness coherence
        effectiveness_factors.append(monitoring.get("consciousness_coherence", 0.5))
        
        # Adaptation quality
        if response["response_type"] == "meta_cognitive" and stimulus.get("complexity", 0) > 0.8:
            effectiveness_factors.append(0.9)  # Good match
        
        return sum(effectiveness_factors) / len(effectiveness_factors) if effectiveness_factors else 0.5
    
    def _score_option_against_awareness(self, option: Dict[str, Any], awareness_state: Any) -> float:
        """Score option against current awareness state"""
        score = 0.5  # Base score
        
        # Adjust based on awareness types
        if AwarenessType.GOAL_AWARENESS in awareness_state.awareness_types:
            if "goal_alignment" in option:
                score += option["goal_alignment"] * 0.3
        
        if AwarenessType.SITUATIONAL_AWARENESS in awareness_state.awareness_types:
            if "situational_fit" in option:
                score += option["situational_fit"] * 0.2
        
        # Adjust based on awareness intensity
        score *= (0.5 + awareness_state.awareness_intensity * 0.5)
        
        return min(1.0, score)
    
    def _score_option_against_intentions(self, option: Dict[str, Any], intentional_state: Optional[Any]) -> float:
        """Score option against intentional state"""
        if not intentional_state:
            return 0.5
        
        score = 0.5
        
        # Score against primary goal
        if intentional_state.primary_goal:
            goal_alignment = option.get("goal_alignment", 0.5)
            score += goal_alignment * intentional_state.intention_strength * 0.4
        
        # Score against commitment level
        score += intentional_state.commitment_level * 0.3
        
        return min(1.0, score)
    
    def _generate_evaluation_rationale(self, option: Dict[str, Any], awareness_score: float,
                                     intention_score: float, meta_insight: Any) -> str:
        """Generate rationale for option evaluation"""
        rationale_parts = []
        
        if awareness_score > 0.7:
            rationale_parts.append("Strong alignment with current awareness")
        
        if intention_score > 0.7:
            rationale_parts.append("Good fit with intentional goals")
        
        if meta_insight.effectiveness_score > 0.7:
            rationale_parts.append("Meta-cognitive analysis supports this option")
        
        if not rationale_parts:
            rationale_parts.append("Moderate alignment across evaluation criteria")
        
        return "; ".join(rationale_parts)
    
    def _calculate_decision_confidence(self, evaluations: List[Dict[str, Any]], awareness_state: Any) -> float:
        """Calculate confidence in decision"""
        if not evaluations:
            return 0.5
        
        best_score = evaluations[0]["overall_score"]
        
        # Base confidence on best option score
        confidence = best_score
        
        # Adjust based on score separation
        if len(evaluations) > 1:
            second_best = evaluations[1]["overall_score"]
            score_separation = best_score - second_best
            confidence += score_separation * 0.2  # More separation = more confidence
        
        # Adjust based on awareness intensity
        confidence *= (0.7 + awareness_state.awareness_intensity * 0.3)
        
        return min(1.0, confidence)
    
    def _extract_decision_learning(self, decision: Dict[str, Any], reflection: Any) -> List[str]:
        """Extract learning from decision process"""
        learning = []
        
        if decision["confidence"] > 0.8:
            learning.append("High-confidence decisions benefit from thorough evaluation")
        
        if "meta_cognitive" in str(reflection.insights):
            learning.append("Meta-cognitive processing enhances decision quality")
        
        learning.extend([
            "Consciousness integration improves decision rationale",
            "Awareness state influences option evaluation",
            "Intentional alignment is crucial for goal achievement"
        ])
        
        return learning
    
    # Additional helper methods for planning analysis
    async def _assess_planning_situation(self, context: Dict[str, Any], awareness_state: Any) -> Dict[str, Any]:
        """Assess planning situation"""
        return {
            "complexity": context.get("complexity", 0.5),
            "urgency": context.get("urgency", 0.5),
            "stakeholder_count": len(context.get("stakeholders", [])),
            "resource_availability": len(context.get("resources", [])),
            "awareness_level": awareness_state.level.value
        }
    
    def _analyze_planning_goals(self, intentions: List[Any]) -> Dict[str, Any]:
        """Analyze planning goals"""
        if not intentions:
            return {"goal_count": 0, "conflicting_goals": []}
        
        goal_priorities = [i.intention_strength for i in intentions]
        
        return {
            "goal_count": len(intentions),
            "average_priority": sum(goal_priorities) / len(goal_priorities),
            "conflicting_goals": [],  # Simplified - would need more complex analysis
            "goal_coherence": min(goal_priorities) / max(goal_priorities) if goal_priorities else 0
        }
    
    def _evaluate_planning_resources(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate planning resources"""
        resources = context.get("resources", [])
        complexity = context.get("complexity", 0.5)
        
        return {
            "resource_count": len(resources),
            "adequacy": min(1.0, len(resources) / max(1, complexity * 5)),
            "resource_types": resources
        }
    
    def _analyze_planning_constraints(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze planning constraints"""
        constraints = context.get("constraints", [])
        
        return {
            "constraint_count": len(constraints),
            "critical_constraints": [c for c in constraints if "critical" in str(c).lower()],
            "constraint_severity": len(constraints) / 10.0  # Simplified severity calculation
        }
    
    async def _identify_planning_opportunities(self, context: Dict[str, Any], awareness_state: Any) -> List[str]:
        """Identify planning opportunities"""
        opportunities = []
        
        if len(context.get("resources", [])) > 5:
            opportunities.append("Resource abundance enables ambitious planning")
        
        if awareness_state.level == ConsciousnessLevel.META_CONSCIOUS:
            opportunities.append("Meta-conscious analysis reveals strategic opportunities")
        
        if context.get("complexity", 0) > 0.7:
            opportunities.append("Complex situation offers differentiation potential")
        
        return opportunities
    
    def _assess_planning_risks(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Assess planning risks"""
        risks = []
        
        if context.get("urgency", 0) > 0.8:
            risks.append("High urgency increases execution risk")
        
        if len(context.get("constraints", [])) > 3:
            risks.append("Multiple constraints may limit flexibility")
        
        return {
            "identified_risks": risks,
            "risk_level": min(1.0, (context.get("urgency", 0) + len(context.get("constraints", [])) / 5) / 2)
        }
    
    def _determine_response_type(self, stimulus_analysis: Dict[str, Any]) -> str:
        """Determine type of response needed"""
        if stimulus_analysis["urgency"] > 0.8:
            return "immediate_action"
        elif stimulus_analysis["complexity"] > 0.8:
            return "analytical_deep_dive"
        elif stimulus_analysis["novelty"] > 0.8:
            return "exploratory_investigation"
        else:
            return "balanced_response"
    
    async def _generate_response_elements(self, stimulus: Dict[str, Any],
                                        awareness: Any, response_type: str) -> List[str]:
        """Generate elements of adaptive response"""
        elements = []
        
        if response_type == "immediate_action":
            elements.extend([
                "Rapid situation assessment",
                "Quick decision making",
                "Immediate implementation"
            ])
        elif response_type == "analytical_deep_dive":
            elements.extend([
                "Comprehensive analysis",
                "Multi-perspective evaluation",
                "Systematic solution development"
            ])
        elif response_type == "exploratory_investigation":
            elements.extend([
                "Novel approach exploration",
                "Creative solution generation",
                "Experimental validation"
            ])
        else:
            elements.extend([
                "Balanced assessment",
                "Measured response",
                "Adaptive implementation"
            ])
        
        return elements
    
    def _determine_adaptation_strategy(self, stimulus_analysis: Dict[str, Any]) -> str:
        """Determine adaptation strategy"""
        if stimulus_analysis["complexity"] > 0.8:
            return "complexity_adaptation"
        elif stimulus_analysis["urgency"] > 0.8:
            return "urgency_adaptation"
        elif stimulus_analysis["novelty"] > 0.8:
            return "novelty_adaptation"
        else:
            return "balanced_adaptation"
    
    def _calculate_response_confidence(self, stimulus_analysis: Dict[str, Any], awareness: Any) -> float:
        """Calculate confidence in adaptive response"""
        base_confidence = 0.5
        
        # Adjust based on awareness intensity
        base_confidence += awareness.awareness_intensity * 0.3
        
        # Adjust based on stimulus familiarity (inverse of novelty)
        familiarity = 1.0 - stimulus_analysis["novelty"]
        base_confidence += familiarity * 0.2
        
        return min(1.0, base_confidence)