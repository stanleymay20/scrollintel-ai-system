"""
Influence Strategy Development Engine for Board Executive Mastery
"""

import logging
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime, timedelta
import json

from ..models.influence_strategy_models import (
    InfluenceStrategy, InfluenceTactic, InfluenceTarget, InfluenceType,
    InfluenceContext, InfluenceObjective, InfluenceExecution,
    InfluenceEffectivenessMetrics, InfluenceOptimization
)
from ..models.stakeholder_influence_models import Stakeholder


class InfluenceStrategyEngine:
    """Engine for developing and optimizing influence strategies"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.tactic_library = self._initialize_tactic_library()
        self.effectiveness_history = {}
        self.optimization_rules = self._initialize_optimization_rules()
    
    def develop_influence_strategy(
        self,
        objective: InfluenceObjective,
        target_stakeholders: List[Stakeholder],
        context: InfluenceContext,
        constraints: Optional[Dict[str, Any]] = None
    ) -> InfluenceStrategy:
        """Develop targeted influence strategy for building support and consensus"""
        try:
            # Convert stakeholders to influence targets
            influence_targets = self._convert_to_influence_targets(target_stakeholders)
            
            # Analyze stakeholder characteristics
            stakeholder_analysis = self._analyze_stakeholder_characteristics(influence_targets)
            
            # Select optimal tactics
            primary_tactics = self._select_primary_tactics(
                objective, influence_targets, context, stakeholder_analysis
            )
            secondary_tactics = self._select_secondary_tactics(
                objective, influence_targets, context, primary_tactics
            )
            
            # Calculate expected effectiveness
            expected_effectiveness = self._calculate_expected_effectiveness(
                primary_tactics, secondary_tactics, influence_targets, context
            )
            
            # Create timeline
            timeline = self._create_influence_timeline(primary_tactics, secondary_tactics)
            
            # Identify risks and mitigation
            risk_mitigation = self._identify_risk_mitigation(
                primary_tactics, influence_targets, context
            )
            
            # Determine resource requirements
            resource_requirements = self._determine_resource_requirements(
                primary_tactics, secondary_tactics, influence_targets
            )
            
            strategy = InfluenceStrategy(
                id=f"influence_strategy_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                name=f"{objective.value.title()} Strategy - {context.value.title()}",
                objective=objective,
                target_stakeholders=influence_targets,
                primary_tactics=primary_tactics,
                secondary_tactics=secondary_tactics,
                context=context,
                timeline=timeline,
                success_metrics=self._define_success_metrics(objective, influence_targets),
                risk_mitigation=risk_mitigation,
                resource_requirements=resource_requirements,
                expected_effectiveness=expected_effectiveness
            )
            
            self.logger.info(f"Developed influence strategy: {strategy.id}")
            return strategy
            
        except Exception as e:
            self.logger.error(f"Error developing influence strategy: {str(e)}")
            raise
    
    def select_optimal_tactics(
        self,
        objective: InfluenceObjective,
        targets: List[InfluenceTarget],
        context: InfluenceContext,
        constraints: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[InfluenceTactic], List[InfluenceTactic]]:
        """Select and optimize influence tactics"""
        try:
            # Filter tactics by context suitability
            suitable_tactics = [
                tactic for tactic in self.tactic_library
                if context in tactic.context_suitability
            ]
            
            # Score tactics for each target
            tactic_scores = {}
            for tactic in suitable_tactics:
                total_score = 0
                for target in targets:
                    score = self._score_tactic_for_target(tactic, target, objective)
                    total_score += score
                
                tactic_scores[tactic.id] = {
                    'tactic': tactic,
                    'score': total_score / len(targets),
                    'individual_scores': {target.stakeholder_id: 
                        self._score_tactic_for_target(tactic, target, objective) 
                        for target in targets}
                }
            
            # Select primary tactics (top performers)
            sorted_tactics = sorted(
                tactic_scores.items(), 
                key=lambda x: x[1]['score'], 
                reverse=True
            )
            
            primary_tactics = [item[1]['tactic'] for item in sorted_tactics[:3]]
            secondary_tactics = [item[1]['tactic'] for item in sorted_tactics[3:6]]
            
            # Apply constraints if provided
            if constraints:
                primary_tactics = self._apply_constraints(primary_tactics, constraints)
                secondary_tactics = self._apply_constraints(secondary_tactics, constraints)
            
            self.logger.info(f"Selected {len(primary_tactics)} primary and {len(secondary_tactics)} secondary tactics")
            return primary_tactics, secondary_tactics
            
        except Exception as e:
            self.logger.error(f"Error selecting tactics: {str(e)}")
            raise
    
    def measure_influence_effectiveness(
        self,
        execution: InfluenceExecution,
        strategy: InfluenceStrategy
    ) -> InfluenceEffectivenessMetrics:
        """Measure and track influence effectiveness"""
        try:
            # Calculate objective achievement
            objective_achievement = self._calculate_objective_achievement(
                execution, strategy
            )
            
            # Assess stakeholder satisfaction
            stakeholder_satisfaction = self._assess_stakeholder_satisfaction(
                execution, strategy.target_stakeholders
            )
            
            # Measure relationship impact
            relationship_impact = self._measure_relationship_impact(
                execution, strategy.target_stakeholders
            )
            
            # Calculate consensus level
            consensus_level = self._calculate_consensus_level(execution)
            
            # Measure support gained
            support_gained = self._measure_support_gained(execution)
            
            # Assess opposition reduction
            opposition_reduced = self._assess_opposition_reduction(execution)
            
            # Evaluate long-term relationship health
            long_term_health = self._evaluate_long_term_relationship_health(
                execution, strategy.target_stakeholders
            )
            
            metrics = InfluenceEffectivenessMetrics(
                strategy_id=strategy.id,
                execution_id=execution.id,
                objective_achievement=objective_achievement,
                stakeholder_satisfaction=stakeholder_satisfaction,
                relationship_impact=relationship_impact,
                consensus_level=consensus_level,
                support_gained=support_gained,
                opposition_reduced=opposition_reduced,
                long_term_relationship_health=long_term_health
            )
            
            # Store for future optimization
            self.effectiveness_history[execution.id] = metrics
            
            self.logger.info(f"Measured influence effectiveness for execution: {execution.id}")
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error measuring influence effectiveness: {str(e)}")
            raise
    
    def optimize_influence_strategy(
        self,
        strategy: InfluenceStrategy,
        effectiveness_metrics: List[InfluenceEffectivenessMetrics]
    ) -> InfluenceOptimization:
        """Optimize influence strategy based on effectiveness data"""
        try:
            # Calculate current effectiveness
            current_effectiveness = sum(
                metrics.objective_achievement for metrics in effectiveness_metrics
            ) / len(effectiveness_metrics)
            
            # Identify optimization opportunities
            optimization_opportunities = self._identify_optimization_opportunities(
                strategy, effectiveness_metrics
            )
            
            # Recommend tactic changes
            tactic_changes = self._recommend_tactic_changes(
                strategy, effectiveness_metrics
            )
            
            # Suggest timing adjustments
            timing_adjustments = self._suggest_timing_adjustments(
                strategy, effectiveness_metrics
            )
            
            # Recommend context modifications
            context_modifications = self._recommend_context_modifications(
                strategy, effectiveness_metrics
            )
            
            # Refine target approaches
            target_refinements = self._refine_target_approaches(
                strategy, effectiveness_metrics
            )
            
            # Calculate expected improvement
            expected_improvement = self._calculate_expected_improvement(
                current_effectiveness, optimization_opportunities
            )
            
            # Assess confidence level
            confidence_level = self._assess_optimization_confidence(
                effectiveness_metrics, optimization_opportunities
            )
            
            optimization = InfluenceOptimization(
                strategy_id=strategy.id,
                current_effectiveness=current_effectiveness,
                optimization_opportunities=optimization_opportunities,
                recommended_tactic_changes=tactic_changes,
                timing_adjustments=timing_adjustments,
                context_modifications=context_modifications,
                target_approach_refinements=target_refinements,
                expected_improvement=expected_improvement,
                confidence_level=confidence_level
            )
            
            self.logger.info(f"Generated optimization recommendations for strategy: {strategy.id}")
            return optimization
            
        except Exception as e:
            self.logger.error(f"Error optimizing influence strategy: {str(e)}")
            raise
    
    def _initialize_tactic_library(self) -> List[InfluenceTactic]:
        """Initialize library of influence tactics"""
        return [
            InfluenceTactic(
                id="rational_persuasion_data",
                name="Data-Driven Rational Persuasion",
                influence_type=InfluenceType.RATIONAL_PERSUASION,
                description="Present logical arguments supported by data and evidence",
                effectiveness_score=0.85,
                context_suitability=[InfluenceContext.BOARD_MEETING, InfluenceContext.COMMITTEE_MEETING],
                target_personality_types=["analytical", "detail_oriented", "skeptical"],
                required_preparation=["data_analysis", "evidence_compilation", "logical_structure"],
                success_indicators=["questions_about_data", "request_for_details", "analytical_engagement"],
                risk_factors=["data_quality_concerns", "analysis_paralysis", "over_complexity"]
            ),
            InfluenceTactic(
                id="inspirational_vision",
                name="Inspirational Vision Appeal",
                influence_type=InfluenceType.INSPIRATIONAL_APPEALS,
                description="Appeal to values, ideals, and aspirations",
                effectiveness_score=0.78,
                context_suitability=[InfluenceContext.BOARD_MEETING, InfluenceContext.STRATEGIC_PLANNING],
                target_personality_types=["visionary", "values_driven", "big_picture"],
                required_preparation=["vision_articulation", "value_alignment", "emotional_connection"],
                success_indicators=["emotional_engagement", "vision_questions", "enthusiasm_increase"],
                risk_factors=["skepticism", "practicality_concerns", "implementation_doubts"]
            ),
            InfluenceTactic(
                id="collaborative_consultation",
                name="Collaborative Consultation",
                influence_type=InfluenceType.CONSULTATION,
                description="Seek input and involve stakeholders in decision-making",
                effectiveness_score=0.82,
                context_suitability=[InfluenceContext.ONE_ON_ONE, InfluenceContext.COMMITTEE_MEETING],
                target_personality_types=["collaborative", "participative", "consensus_seeking"],
                required_preparation=["question_preparation", "listening_skills", "synthesis_ability"],
                success_indicators=["active_participation", "idea_contribution", "ownership_feeling"],
                risk_factors=["decision_delays", "conflicting_opinions", "analysis_paralysis"]
            )
        ]
    
    def _initialize_optimization_rules(self) -> Dict[str, Any]:
        """Initialize optimization rules"""
        return {
            "effectiveness_threshold": 0.7,
            "relationship_impact_weight": 0.3,
            "objective_achievement_weight": 0.4,
            "consensus_weight": 0.3,
            "minimum_confidence": 0.6
        }
    
    def _convert_to_influence_targets(self, stakeholders: List[Stakeholder]) -> List[InfluenceTarget]:
        """Convert stakeholders to influence targets"""
        targets = []
        for stakeholder in stakeholders:
            # Convert influence level enum to float
            influence_level_map = {
                "low": 0.3,
                "medium": 0.6,
                "high": 0.8,
                "critical": 0.95
            }
            influence_level = influence_level_map.get(stakeholder.influence_level.value, 0.5)
            
            # Extract communication preferences from communication style
            comm_prefs = [stakeholder.communication_style.value]
            
            # Extract motivators from priorities
            key_motivators = [priority.name for priority in stakeholder.priorities]
            
            # Extract concerns from decision pattern
            concerns = stakeholder.decision_pattern.typical_concerns
            
            # Calculate relationship strength (simplified - would be more complex in practice)
            relationship_strength = 0.7  # Default value, would be calculated from relationships
            
            target = InfluenceTarget(
                stakeholder_id=stakeholder.id,
                name=stakeholder.name,
                role=stakeholder.title,
                influence_level=influence_level,
                decision_making_style=stakeholder.decision_pattern.decision_style,
                communication_preferences=comm_prefs,
                key_motivators=key_motivators,
                concerns=concerns,
                relationship_strength=relationship_strength
            )
            targets.append(target)
        return targets
    
    def _analyze_stakeholder_characteristics(self, targets: List[InfluenceTarget]) -> Dict[str, Any]:
        """Analyze stakeholder characteristics for strategy development"""
        analysis = {
            "dominant_decision_styles": {},
            "common_motivators": [],
            "shared_concerns": [],
            "relationship_strengths": {},
            "influence_distribution": {}
        }
        
        # Analyze decision-making styles
        styles = [target.decision_making_style for target in targets]
        for style in set(styles):
            analysis["dominant_decision_styles"][style] = styles.count(style) / len(styles)
        
        # Find common motivators
        all_motivators = []
        for target in targets:
            all_motivators.extend(target.key_motivators)
        
        motivator_counts = {}
        for motivator in all_motivators:
            motivator_counts[motivator] = motivator_counts.get(motivator, 0) + 1
        
        analysis["common_motivators"] = [
            motivator for motivator, count in motivator_counts.items()
            if count >= len(targets) * 0.5
        ]
        
        return analysis
    
    def _select_primary_tactics(
        self,
        objective: InfluenceObjective,
        targets: List[InfluenceTarget],
        context: InfluenceContext,
        analysis: Dict[str, Any]
    ) -> List[InfluenceTactic]:
        """Select primary influence tactics"""
        # This is a simplified selection - in practice would be more sophisticated
        suitable_tactics = [
            tactic for tactic in self.tactic_library
            if context in tactic.context_suitability
        ]
        
        # Score and select top tactics
        scored_tactics = []
        for tactic in suitable_tactics:
            score = tactic.effectiveness_score
            # Adjust score based on target characteristics
            for target in targets:
                if any(ptype in target.decision_making_style.lower() 
                      for ptype in tactic.target_personality_types):
                    score += 0.1
            
            scored_tactics.append((tactic, score))
        
        # Sort by score and return top 2-3
        scored_tactics.sort(key=lambda x: x[1], reverse=True)
        return [tactic for tactic, _ in scored_tactics[:2]]
    
    def _select_secondary_tactics(
        self,
        objective: InfluenceObjective,
        targets: List[InfluenceTarget],
        context: InfluenceContext,
        primary_tactics: List[InfluenceTactic]
    ) -> List[InfluenceTactic]:
        """Select secondary/backup influence tactics"""
        # Select tactics that complement primary tactics
        used_types = {tactic.influence_type for tactic in primary_tactics}
        
        secondary_tactics = []
        for tactic in self.tactic_library:
            if (tactic.influence_type not in used_types and 
                context in tactic.context_suitability and
                len(secondary_tactics) < 2):
                secondary_tactics.append(tactic)
        
        return secondary_tactics
    
    def _calculate_expected_effectiveness(
        self,
        primary_tactics: List[InfluenceTactic],
        secondary_tactics: List[InfluenceTactic],
        targets: List[InfluenceTarget],
        context: InfluenceContext
    ) -> float:
        """Calculate expected effectiveness of strategy"""
        if not primary_tactics:
            return 0.0
        
        # Weight primary tactics more heavily
        primary_score = sum(tactic.effectiveness_score for tactic in primary_tactics) / len(primary_tactics)
        secondary_score = sum(tactic.effectiveness_score for tactic in secondary_tactics) / max(len(secondary_tactics), 1)
        
        # Combine scores with weighting
        combined_score = (primary_score * 0.7) + (secondary_score * 0.3)
        
        # Adjust for target relationship strength
        avg_relationship = sum(target.relationship_strength for target in targets) / len(targets)
        relationship_factor = 0.8 + (avg_relationship * 0.2)
        
        return min(combined_score * relationship_factor, 1.0)
    
    def _create_influence_timeline(
        self,
        primary_tactics: List[InfluenceTactic],
        secondary_tactics: List[InfluenceTactic]
    ) -> Dict[str, datetime]:
        """Create timeline for influence strategy execution"""
        now = datetime.now()
        return {
            "preparation_start": now,
            "preparation_complete": now + timedelta(days=3),
            "primary_execution_start": now + timedelta(days=4),
            "primary_execution_complete": now + timedelta(days=10),
            "assessment_date": now + timedelta(days=12),
            "secondary_execution_start": now + timedelta(days=14),
            "final_assessment": now + timedelta(days=21)
        }
    
    def _identify_risk_mitigation(
        self,
        tactics: List[InfluenceTactic],
        targets: List[InfluenceTarget],
        context: InfluenceContext
    ) -> Dict[str, str]:
        """Identify risks and mitigation strategies"""
        risk_mitigation = {}
        
        for tactic in tactics:
            for risk in tactic.risk_factors:
                if risk not in risk_mitigation:
                    risk_mitigation[risk] = self._get_mitigation_strategy(risk, context)
        
        return risk_mitigation
    
    def _get_mitigation_strategy(self, risk: str, context: InfluenceContext) -> str:
        """Get mitigation strategy for specific risk"""
        mitigation_map = {
            "data_quality_concerns": "Prepare data validation and source documentation",
            "analysis_paralysis": "Set clear decision timelines and criteria",
            "over_complexity": "Prepare simplified executive summaries",
            "skepticism": "Include third-party validation and references",
            "practicality_concerns": "Develop detailed implementation roadmap",
            "decision_delays": "Establish clear decision-making process and timeline"
        }
        return mitigation_map.get(risk, "Monitor and adapt approach as needed")
    
    def _determine_resource_requirements(
        self,
        primary_tactics: List[InfluenceTactic],
        secondary_tactics: List[InfluenceTactic],
        targets: List[InfluenceTarget]
    ) -> List[str]:
        """Determine resource requirements for strategy"""
        resources = set()
        
        for tactic in primary_tactics + secondary_tactics:
            resources.update(tactic.required_preparation)
        
        # Add target-specific resources
        resources.add("stakeholder_research")
        resources.add("presentation_materials")
        resources.add("follow_up_system")
        
        return list(resources)
    
    def _define_success_metrics(
        self,
        objective: InfluenceObjective,
        targets: List[InfluenceTarget]
    ) -> List[str]:
        """Define success metrics for influence strategy"""
        base_metrics = [
            "objective_achievement_rate",
            "stakeholder_satisfaction_score",
            "relationship_strength_change",
            "consensus_level_achieved"
        ]
        
        objective_specific = {
            InfluenceObjective.BUILD_SUPPORT: ["support_level_increase", "advocacy_behaviors"],
            InfluenceObjective.GAIN_CONSENSUS: ["consensus_percentage", "dissent_reduction"],
            InfluenceObjective.CHANGE_OPINION: ["opinion_shift_measurement", "attitude_change"],
            InfluenceObjective.SECURE_APPROVAL: ["approval_rate", "approval_timeline"],
            InfluenceObjective.PREVENT_OPPOSITION: ["opposition_level", "neutral_maintenance"],
            InfluenceObjective.BUILD_COALITION: ["coalition_size", "coalition_strength"]
        }
        
        return base_metrics + objective_specific.get(objective, [])
    
    def _score_tactic_for_target(
        self,
        tactic: InfluenceTactic,
        target: InfluenceTarget,
        objective: InfluenceObjective
    ) -> float:
        """Score how well a tactic matches a specific target"""
        base_score = tactic.effectiveness_score
        
        # Adjust for personality match
        personality_match = any(
            ptype in target.decision_making_style.lower()
            for ptype in tactic.target_personality_types
        )
        if personality_match:
            base_score += 0.1
        
        # Adjust for relationship strength
        relationship_factor = 0.8 + (target.relationship_strength * 0.2)
        
        return min(base_score * relationship_factor, 1.0)
    
    def _apply_constraints(
        self,
        tactics: List[InfluenceTactic],
        constraints: Dict[str, Any]
    ) -> List[InfluenceTactic]:
        """Apply constraints to tactic selection"""
        # This would filter tactics based on constraints like time, resources, etc.
        return tactics  # Simplified for now
    
    def _calculate_objective_achievement(
        self,
        execution: InfluenceExecution,
        strategy: InfluenceStrategy
    ) -> float:
        """Calculate how well the objective was achieved"""
        # Simplified calculation based on execution effectiveness rating
        return execution.effectiveness_rating
    
    def _assess_stakeholder_satisfaction(
        self,
        execution: InfluenceExecution,
        targets: List[InfluenceTarget]
    ) -> Dict[str, float]:
        """Assess stakeholder satisfaction with influence approach"""
        satisfaction = {}
        for target in targets:
            # In practice, this would be based on feedback or behavioral indicators
            satisfaction[target.stakeholder_id] = execution.effectiveness_rating * 0.9
        return satisfaction
    
    def _measure_relationship_impact(
        self,
        execution: InfluenceExecution,
        targets: List[InfluenceTarget]
    ) -> Dict[str, float]:
        """Measure impact on relationships"""
        impact = {}
        for target in targets:
            # Simplified - would measure actual relationship changes
            impact[target.stakeholder_id] = 0.05 if execution.effectiveness_rating > 0.7 else -0.02
        return impact
    
    def _calculate_consensus_level(self, execution: InfluenceExecution) -> float:
        """Calculate level of consensus achieved"""
        # Simplified calculation
        return execution.effectiveness_rating * 0.8
    
    def _measure_support_gained(self, execution: InfluenceExecution) -> float:
        """Measure support gained through influence"""
        return execution.effectiveness_rating * 0.75
    
    def _assess_opposition_reduction(self, execution: InfluenceExecution) -> float:
        """Assess reduction in opposition"""
        return execution.effectiveness_rating * 0.6
    
    def _evaluate_long_term_relationship_health(
        self,
        execution: InfluenceExecution,
        targets: List[InfluenceTarget]
    ) -> float:
        """Evaluate long-term relationship health impact"""
        # Consider whether tactics were relationship-preserving
        return execution.effectiveness_rating * 0.85
    
    def _identify_optimization_opportunities(
        self,
        strategy: InfluenceStrategy,
        metrics: List[InfluenceEffectivenessMetrics]
    ) -> List[str]:
        """Identify opportunities for strategy optimization"""
        opportunities = []
        
        avg_effectiveness = sum(m.objective_achievement for m in metrics) / len(metrics)
        if avg_effectiveness < 0.7:
            opportunities.append("Improve tactic selection for target personalities")
        
        avg_consensus = sum(m.consensus_level for m in metrics) / len(metrics)
        if avg_consensus < 0.6:
            opportunities.append("Enhance consensus-building approaches")
        
        return opportunities
    
    def _recommend_tactic_changes(
        self,
        strategy: InfluenceStrategy,
        metrics: List[InfluenceEffectivenessMetrics]
    ) -> List[Dict[str, Any]]:
        """Recommend changes to tactics"""
        changes = []
        
        # Analyze which tactics performed poorly
        avg_effectiveness = sum(m.objective_achievement for m in metrics) / len(metrics)
        if avg_effectiveness < 0.7:
            changes.append({
                "change_type": "replace_tactic",
                "current_tactic": strategy.primary_tactics[0].id if strategy.primary_tactics else None,
                "recommended_tactic": "collaborative_consultation",
                "reason": "Improve stakeholder engagement"
            })
        
        return changes
    
    def _suggest_timing_adjustments(
        self,
        strategy: InfluenceStrategy,
        metrics: List[InfluenceEffectivenessMetrics]
    ) -> List[str]:
        """Suggest timing adjustments"""
        adjustments = []
        
        # Analyze timing effectiveness
        if any(m.objective_achievement < 0.6 for m in metrics):
            adjustments.append("Allow more preparation time before primary execution")
            adjustments.append("Extend relationship building phase")
        
        return adjustments
    
    def _recommend_context_modifications(
        self,
        strategy: InfluenceStrategy,
        metrics: List[InfluenceEffectivenessMetrics]
    ) -> List[str]:
        """Recommend context modifications"""
        modifications = []
        
        avg_satisfaction = sum(
            sum(m.stakeholder_satisfaction.values()) / len(m.stakeholder_satisfaction)
            for m in metrics
        ) / len(metrics)
        
        if avg_satisfaction < 0.7:
            modifications.append("Consider more informal settings for initial discussions")
            modifications.append("Use one-on-one meetings before group sessions")
        
        return modifications
    
    def _refine_target_approaches(
        self,
        strategy: InfluenceStrategy,
        metrics: List[InfluenceEffectivenessMetrics]
    ) -> Dict[str, List[str]]:
        """Refine approaches for specific targets"""
        refinements = {}
        
        for target in strategy.target_stakeholders:
            target_refinements = []
            
            # Analyze target-specific effectiveness
            target_satisfaction = sum(
                m.stakeholder_satisfaction.get(target.stakeholder_id, 0.5)
                for m in metrics
            ) / len(metrics)
            
            if target_satisfaction < 0.6:
                target_refinements.append("Increase personalization of approach")
                target_refinements.append("Focus more on target's key motivators")
            
            if target_refinements:
                refinements[target.stakeholder_id] = target_refinements
        
        return refinements
    
    def _calculate_expected_improvement(
        self,
        current_effectiveness: float,
        opportunities: List[str]
    ) -> float:
        """Calculate expected improvement from optimizations"""
        # Simplified calculation
        improvement_per_opportunity = 0.05
        max_improvement = 0.3
        
        potential_improvement = min(
            len(opportunities) * improvement_per_opportunity,
            max_improvement
        )
        
        return min(current_effectiveness + potential_improvement, 1.0)
    
    def _assess_optimization_confidence(
        self,
        metrics: List[InfluenceEffectivenessMetrics],
        opportunities: List[str]
    ) -> float:
        """Assess confidence in optimization recommendations"""
        # Base confidence on data quality and consistency
        if len(metrics) < 3:
            return 0.5  # Low confidence with limited data
        
        # Calculate consistency of results
        effectiveness_values = [m.objective_achievement for m in metrics]
        variance = sum((x - sum(effectiveness_values)/len(effectiveness_values))**2 
                      for x in effectiveness_values) / len(effectiveness_values)
        
        # Higher variance = lower confidence
        consistency_factor = max(0.3, 1.0 - variance)
        
        # More opportunities identified = higher confidence in need for optimization
        opportunity_factor = min(1.0, len(opportunities) * 0.2 + 0.4)
        
        return (consistency_factor + opportunity_factor) / 2