"""
Behavior Modification Engine for Cultural Transformation Leadership

This engine develops systematic behavior change strategies, selects appropriate
modification techniques, and tracks behavior change progress with optimization.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from uuid import uuid4

from ..models.behavior_modification_models import (
    BehaviorModificationStrategy, ModificationIntervention, BehaviorChangeProgress,
    ModificationPlan, TechniqueEffectiveness, ModificationOptimization,
    BehaviorChangeMetrics, ModificationTechnique, ModificationStatus, ProgressLevel
)
from ..models.behavioral_analysis_models import BehaviorPattern, BehavioralNorm

logger = logging.getLogger(__name__)


class BehaviorModificationEngine:
    """Engine for systematic behavior modification"""
    
    def __init__(self):
        self.modification_plans = {}
        self.strategies = {}
        self.interventions = {}
        self.progress_tracking = {}
        self.technique_effectiveness = {}
        
    def develop_modification_strategy(
        self,
        target_behavior: str,
        current_state: Dict[str, Any],
        desired_outcome: str,
        constraints: Dict[str, Any] = None,
        stakeholders: List[str] = None
    ) -> BehaviorModificationStrategy:
        """
        Develop systematic behavior change strategy
        
        Requirements: 3.2, 3.3 - Create systematic behavior change strategy development
        """
        try:
            strategy_id = str(uuid4())
            constraints = constraints or {}
            stakeholders = stakeholders or []
            
            # Analyze current behavior and determine modification approach
            techniques = self._select_modification_techniques(
                target_behavior, current_state, desired_outcome, constraints
            )
            
            # Estimate timeline based on behavior complexity
            timeline_weeks = self._estimate_modification_timeline(
                target_behavior, current_state, techniques
            )
            
            # Define success criteria
            success_criteria = self._define_success_criteria(
                target_behavior, desired_outcome, current_state
            )
            
            # Identify required resources
            resources_required = self._identify_required_resources(techniques, stakeholders)
            
            # Assess risks and mitigation strategies
            risk_factors, mitigation_strategies = self._assess_modification_risks(
                target_behavior, techniques, constraints
            )
            
            strategy = BehaviorModificationStrategy(
                id=strategy_id,
                name=f"Modification Strategy: {target_behavior}",
                description=f"Strategy to change {target_behavior} to achieve {desired_outcome}",
                target_behavior=target_behavior,
                desired_outcome=desired_outcome,
                techniques=techniques,
                timeline_weeks=timeline_weeks,
                success_criteria=success_criteria,
                resources_required=resources_required,
                stakeholders=stakeholders,
                risk_factors=risk_factors,
                mitigation_strategies=mitigation_strategies,
                created_date=datetime.now(),
                created_by="ScrollIntel"
            )
            
            self.strategies[strategy_id] = strategy
            logger.info(f"Developed modification strategy for {target_behavior}")
            
            return strategy
            
        except Exception as e:
            logger.error(f"Error developing modification strategy: {str(e)}")
            raise
    
    def _select_modification_techniques(
        self,
        target_behavior: str,
        current_state: Dict[str, Any],
        desired_outcome: str,
        constraints: Dict[str, Any]
    ) -> List[ModificationTechnique]:
        """Select appropriate modification techniques based on context"""
        
        techniques = []
        behavior_complexity = current_state.get('complexity', 'medium')
        participant_count = current_state.get('participant_count', 10)
        urgency = constraints.get('urgency', 'medium')
        budget = constraints.get('budget', 'medium')
        
        # Always include feedback as a foundational technique
        techniques.append(ModificationTechnique.FEEDBACK)
        
        # Select techniques based on behavior characteristics
        if 'communication' in target_behavior.lower():
            techniques.extend([
                ModificationTechnique.MODELING,
                ModificationTechnique.COACHING,
                ModificationTechnique.TRAINING
            ])
        
        if 'collaboration' in target_behavior.lower():
            techniques.extend([
                ModificationTechnique.PEER_INFLUENCE,
                ModificationTechnique.ENVIRONMENTAL_DESIGN,
                ModificationTechnique.GOAL_SETTING
            ])
        
        if 'leadership' in target_behavior.lower():
            techniques.extend([
                ModificationTechnique.COACHING,
                ModificationTechnique.MODELING,
                ModificationTechnique.GOAL_SETTING
            ])
        
        if 'innovation' in target_behavior.lower():
            techniques.extend([
                ModificationTechnique.POSITIVE_REINFORCEMENT,
                ModificationTechnique.INCENTIVE_SYSTEMS,
                ModificationTechnique.ENVIRONMENTAL_DESIGN
            ])
        
        # Adjust based on constraints
        if urgency == 'high':
            techniques.append(ModificationTechnique.INCENTIVE_SYSTEMS)
        
        if budget == 'high':
            techniques.append(ModificationTechnique.TRAINING)
        
        if participant_count > 50:
            techniques.append(ModificationTechnique.ENVIRONMENTAL_DESIGN)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_techniques = []
        for technique in techniques:
            if technique not in seen:
                seen.add(technique)
                unique_techniques.append(technique)
        
        return unique_techniques[:5]  # Limit to 5 techniques for manageability
    
    def _estimate_modification_timeline(
        self,
        target_behavior: str,
        current_state: Dict[str, Any],
        techniques: List[ModificationTechnique]
    ) -> int:
        """Estimate timeline for behavior modification in weeks"""
        
        base_timeline = 8  # Base 8 weeks
        
        # Adjust based on behavior complexity
        complexity = current_state.get('complexity', 'medium')
        complexity_multipliers = {
            'low': 0.7,
            'medium': 1.0,
            'high': 1.5
        }
        
        # Adjust based on current behavior strength
        current_strength = current_state.get('strength', 0.5)
        if current_strength > 0.8:  # Strong existing behavior is harder to change
            base_timeline *= 1.3
        elif current_strength < 0.3:  # Weak behavior is easier to change
            base_timeline *= 0.8
        
        # Adjust based on techniques used
        technique_adjustments = {
            ModificationTechnique.TRAINING: 1.2,
            ModificationTechnique.COACHING: 1.1,
            ModificationTechnique.ENVIRONMENTAL_DESIGN: 0.9,
            ModificationTechnique.INCENTIVE_SYSTEMS: 0.8
        }
        
        technique_factor = 1.0
        for technique in techniques:
            if technique in technique_adjustments:
                technique_factor *= technique_adjustments[technique]
        
        final_timeline = int(base_timeline * complexity_multipliers[complexity] * technique_factor)
        
        return max(4, min(52, final_timeline))  # Between 4 and 52 weeks
    
    def _define_success_criteria(
        self,
        target_behavior: str,
        desired_outcome: str,
        current_state: Dict[str, Any]
    ) -> List[str]:
        """Define measurable success criteria for behavior modification"""
        
        criteria = []
        
        # Quantitative criteria
        current_frequency = current_state.get('frequency', 0.3)
        target_frequency = min(1.0, current_frequency + 0.4)
        criteria.append(f"Increase behavior frequency to {target_frequency:.1f}")
        
        current_quality = current_state.get('quality', 0.5)
        target_quality = min(1.0, current_quality + 0.3)
        criteria.append(f"Improve behavior quality to {target_quality:.1f}")
        
        # Qualitative criteria based on behavior type
        if 'communication' in target_behavior.lower():
            criteria.extend([
                "Achieve clear and effective communication",
                "Reduce communication-related conflicts by 50%",
                "Increase stakeholder satisfaction with communication"
            ])
        
        if 'collaboration' in target_behavior.lower():
            criteria.extend([
                "Increase cross-team collaboration instances",
                "Improve team cohesion scores",
                "Reduce project completion time through better collaboration"
            ])
        
        if 'leadership' in target_behavior.lower():
            criteria.extend([
                "Demonstrate consistent leadership behaviors",
                "Increase team engagement and motivation",
                "Achieve leadership effectiveness rating above 4.0/5.0"
            ])
        
        # Sustainability criteria
        criteria.extend([
            "Maintain behavior change for at least 3 months",
            "Achieve 80% participant satisfaction with change process",
            "Demonstrate behavior integration into daily routines"
        ])
        
        return criteria
    
    def _identify_required_resources(
        self,
        techniques: List[ModificationTechnique],
        stakeholders: List[str]
    ) -> List[str]:
        """Identify resources required for modification techniques"""
        
        resources = set()
        
        # Resources based on techniques
        technique_resources = {
            ModificationTechnique.TRAINING: ["training_materials", "facilitator", "training_venue"],
            ModificationTechnique.COACHING: ["certified_coach", "coaching_sessions", "assessment_tools"],
            ModificationTechnique.MODELING: ["role_models", "demonstration_opportunities", "observation_tools"],
            ModificationTechnique.FEEDBACK: ["feedback_system", "measurement_tools", "regular_check_ins"],
            ModificationTechnique.GOAL_SETTING: ["goal_tracking_system", "progress_dashboards", "review_meetings"],
            ModificationTechnique.ENVIRONMENTAL_DESIGN: ["workspace_modifications", "process_changes", "system_updates"],
            ModificationTechnique.INCENTIVE_SYSTEMS: ["reward_budget", "recognition_program", "performance_metrics"],
            ModificationTechnique.PEER_INFLUENCE: ["peer_champions", "group_activities", "collaboration_tools"]
        }
        
        for technique in techniques:
            if technique in technique_resources:
                resources.update(technique_resources[technique])
        
        # General resources
        resources.update([
            "project_manager",
            "communication_plan",
            "progress_tracking_system",
            "participant_time_allocation"
        ])
        
        return list(resources)
    
    def _assess_modification_risks(
        self,
        target_behavior: str,
        techniques: List[ModificationTechnique],
        constraints: Dict[str, Any]
    ) -> tuple[List[str], List[str]]:
        """Assess risks and develop mitigation strategies"""
        
        risk_factors = []
        mitigation_strategies = []
        
        # Common risks
        risk_factors.extend([
            "Participant resistance to change",
            "Insufficient time allocation",
            "Competing priorities",
            "Lack of management support",
            "Resource constraints"
        ])
        
        # Technique-specific risks
        if ModificationTechnique.TRAINING in techniques:
            risk_factors.append("Training effectiveness varies by participant")
            mitigation_strategies.append("Provide multiple training formats and follow-up support")
        
        if ModificationTechnique.INCENTIVE_SYSTEMS in techniques:
            risk_factors.append("Over-reliance on external motivation")
            mitigation_strategies.append("Gradually transition to intrinsic motivation")
        
        if ModificationTechnique.ENVIRONMENTAL_DESIGN in techniques:
            risk_factors.append("Environmental changes may face organizational resistance")
            mitigation_strategies.append("Implement changes gradually with stakeholder involvement")
        
        # General mitigation strategies
        mitigation_strategies.extend([
            "Conduct thorough stakeholder engagement and communication",
            "Implement change management best practices",
            "Provide continuous support and coaching",
            "Monitor progress closely and adjust approach as needed",
            "Celebrate early wins to build momentum"
        ])
        
        return risk_factors, mitigation_strategies
    
    def create_modification_interventions(
        self,
        strategy: BehaviorModificationStrategy,
        participants: List[str],
        facilitators: List[str] = None
    ) -> List[ModificationIntervention]:
        """
        Create specific interventions for behavior modification strategy
        
        Requirements: 3.2, 3.3 - Implement behavior modification technique selection and application
        """
        try:
            interventions = []
            facilitators = facilitators or ["ScrollIntel"]
            
            for i, technique in enumerate(strategy.techniques):
                intervention = self._create_intervention_for_technique(
                    strategy, technique, participants, facilitators, i
                )
                interventions.append(intervention)
                self.interventions[intervention.id] = intervention
            
            logger.info(f"Created {len(interventions)} interventions for strategy {strategy.id}")
            return interventions
            
        except Exception as e:
            logger.error(f"Error creating interventions: {str(e)}")
            raise
    
    def _create_intervention_for_technique(
        self,
        strategy: BehaviorModificationStrategy,
        technique: ModificationTechnique,
        participants: List[str],
        facilitators: List[str],
        sequence: int
    ) -> ModificationIntervention:
        """Create a specific intervention for a modification technique"""
        
        intervention_id = str(uuid4())
        
        # Define intervention details based on technique
        intervention_details = self._get_intervention_details(technique, strategy.target_behavior)
        
        # Calculate timing
        start_offset_days = sequence * 7  # Start each intervention a week apart
        duration_days = intervention_details['duration_days']
        
        intervention = ModificationIntervention(
            id=intervention_id,
            strategy_id=strategy.id,
            technique=technique,
            intervention_name=intervention_details['name'],
            description=intervention_details['description'],
            target_participants=participants,
            implementation_steps=intervention_details['steps'],
            duration_days=duration_days,
            frequency=intervention_details['frequency'],
            resources_needed=intervention_details['resources'],
            success_metrics=intervention_details['metrics'],
            status=ModificationStatus.PLANNED,
            assigned_facilitator=facilitators[0] if facilitators else None
        )
        
        return intervention
    
    def _get_intervention_details(
        self,
        technique: ModificationTechnique,
        target_behavior: str
    ) -> Dict[str, Any]:
        """Get detailed intervention specifications for a technique"""
        
        details = {
            ModificationTechnique.POSITIVE_REINFORCEMENT: {
                'name': f'Positive Reinforcement for {target_behavior}',
                'description': 'Reinforce desired behaviors through recognition and rewards',
                'duration_days': 21,
                'frequency': 'daily',
                'steps': [
                    'Identify specific behaviors to reinforce',
                    'Establish recognition criteria',
                    'Implement immediate positive feedback',
                    'Track and celebrate improvements',
                    'Adjust reinforcement schedule'
                ],
                'resources': ['recognition_system', 'feedback_tools', 'reward_budget'],
                'metrics': ['behavior_frequency', 'participant_satisfaction', 'improvement_rate']
            },
            ModificationTechnique.COACHING: {
                'name': f'Coaching Program for {target_behavior}',
                'description': 'One-on-one coaching to develop desired behaviors',
                'duration_days': 42,
                'frequency': 'weekly',
                'steps': [
                    'Conduct initial assessment',
                    'Set coaching goals and milestones',
                    'Provide regular coaching sessions',
                    'Practice new behaviors',
                    'Review progress and adjust approach'
                ],
                'resources': ['certified_coach', 'coaching_materials', 'assessment_tools'],
                'metrics': ['skill_development', 'behavior_consistency', 'goal_achievement']
            },
            ModificationTechnique.TRAINING: {
                'name': f'Training Program for {target_behavior}',
                'description': 'Structured training to build required skills and knowledge',
                'duration_days': 14,
                'frequency': 'intensive',
                'steps': [
                    'Design training curriculum',
                    'Deliver interactive training sessions',
                    'Provide hands-on practice opportunities',
                    'Conduct knowledge assessments',
                    'Offer follow-up support'
                ],
                'resources': ['training_materials', 'facilitator', 'training_venue', 'assessment_tools'],
                'metrics': ['knowledge_acquisition', 'skill_demonstration', 'application_rate']
            },
            ModificationTechnique.MODELING: {
                'name': f'Behavioral Modeling for {target_behavior}',
                'description': 'Demonstrate desired behaviors through role models',
                'duration_days': 28,
                'frequency': 'ongoing',
                'steps': [
                    'Identify effective role models',
                    'Create observation opportunities',
                    'Facilitate modeling sessions',
                    'Encourage practice and imitation',
                    'Provide feedback on attempts'
                ],
                'resources': ['role_models', 'observation_tools', 'practice_opportunities'],
                'metrics': ['observation_frequency', 'imitation_accuracy', 'behavior_adoption']
            },
            ModificationTechnique.ENVIRONMENTAL_DESIGN: {
                'name': f'Environment Design for {target_behavior}',
                'description': 'Modify environment to support desired behaviors',
                'duration_days': 35,
                'frequency': 'permanent',
                'steps': [
                    'Analyze current environment',
                    'Design behavior-supporting changes',
                    'Implement environmental modifications',
                    'Monitor behavior changes',
                    'Refine environment as needed'
                ],
                'resources': ['design_expertise', 'modification_budget', 'implementation_team'],
                'metrics': ['environment_utilization', 'behavior_frequency', 'user_satisfaction']
            }
        }
        
        # Default details for techniques not explicitly defined
        default_details = {
            'name': f'{technique.value.replace("_", " ").title()} for {target_behavior}',
            'description': f'Apply {technique.value.replace("_", " ")} technique to modify {target_behavior}',
            'duration_days': 21,
            'frequency': 'weekly',
            'steps': [
                'Plan intervention approach',
                'Implement technique',
                'Monitor progress',
                'Adjust as needed',
                'Evaluate results'
            ],
            'resources': ['facilitator', 'tracking_tools'],
            'metrics': ['behavior_change', 'participant_engagement']
        }
        
        return details.get(technique, default_details)
    
    def track_behavior_change_progress(
        self,
        strategy_id: str,
        participant_id: str,
        current_measurement: float,
        baseline_measurement: float = None,
        target_measurement: float = None
    ) -> BehaviorChangeProgress:
        """
        Track progress of behavior change efforts
        
        Requirements: 3.2, 3.3 - Build behavior change progress tracking and optimization
        """
        try:
            progress_id = str(uuid4())
            
            # Use provided values or defaults
            baseline = baseline_measurement if baseline_measurement is not None else 0.3
            target = target_measurement if target_measurement is not None else 0.8
            
            # Determine progress level
            progress_level = self._determine_progress_level(baseline, current_measurement, target)
            
            # Calculate improvement rate (simplified - would need historical data)
            improvement_rate = max(0.0, current_measurement - baseline)
            
            # Generate milestones and challenges (simplified)
            milestones = self._generate_milestones(baseline, current_measurement, target)
            challenges = self._identify_challenges(progress_level, current_measurement)
            adjustments = self._suggest_adjustments(progress_level, strategy_id)
            
            progress = BehaviorChangeProgress(
                id=progress_id,
                strategy_id=strategy_id,
                participant_id=participant_id,
                baseline_measurement=baseline,
                current_measurement=current_measurement,
                target_measurement=target,
                progress_level=progress_level,
                improvement_rate=improvement_rate,
                milestones_achieved=milestones,
                challenges_encountered=challenges,
                adjustments_made=adjustments,
                last_updated=datetime.now(),
                next_review_date=datetime.now() + timedelta(weeks=1)
            )
            
            self.progress_tracking[progress_id] = progress
            logger.info(f"Tracked progress for participant {participant_id} in strategy {strategy_id}")
            
            return progress
            
        except Exception as e:
            logger.error(f"Error tracking behavior change progress: {str(e)}")
            raise
    
    def _determine_progress_level(
        self,
        baseline: float,
        current: float,
        target: float
    ) -> ProgressLevel:
        """Determine the level of progress made"""
        
        if current <= baseline:
            return ProgressLevel.NO_CHANGE
        
        progress_ratio = (current - baseline) / (target - baseline) if target > baseline else 0
        
        if progress_ratio < 0.2:
            return ProgressLevel.MINIMAL_PROGRESS
        elif progress_ratio < 0.5:
            return ProgressLevel.MODERATE_PROGRESS
        elif progress_ratio < 0.9:
            return ProgressLevel.SIGNIFICANT_PROGRESS
        else:
            return ProgressLevel.COMPLETE_CHANGE
    
    def _generate_milestones(
        self,
        baseline: float,
        current: float,
        target: float
    ) -> List[str]:
        """Generate achieved milestones based on progress"""
        
        milestones = []
        progress_ratio = (current - baseline) / (target - baseline) if target > baseline else 0
        
        if progress_ratio >= 0.1:
            milestones.append("Initial behavior change observed")
        if progress_ratio >= 0.25:
            milestones.append("Quarter progress milestone reached")
        if progress_ratio >= 0.5:
            milestones.append("Halfway milestone achieved")
        if progress_ratio >= 0.75:
            milestones.append("Three-quarter milestone reached")
        if progress_ratio >= 0.9:
            milestones.append("Near-complete behavior change achieved")
        
        return milestones
    
    def _identify_challenges(
        self,
        progress_level: ProgressLevel,
        current_measurement: float
    ) -> List[str]:
        """Identify challenges based on progress level"""
        
        challenges = []
        
        if progress_level == ProgressLevel.NO_CHANGE:
            challenges.extend([
                "No measurable behavior change observed",
                "Possible resistance to change",
                "Intervention may not be effective"
            ])
        elif progress_level == ProgressLevel.MINIMAL_PROGRESS:
            challenges.extend([
                "Slow progress rate",
                "May need additional support",
                "Motivation may be low"
            ])
        elif progress_level in [ProgressLevel.MODERATE_PROGRESS, ProgressLevel.SIGNIFICANT_PROGRESS]:
            challenges.extend([
                "Maintaining momentum",
                "Avoiding regression",
                "Sustaining motivation"
            ])
        
        return challenges
    
    def _suggest_adjustments(
        self,
        progress_level: ProgressLevel,
        strategy_id: str
    ) -> List[str]:
        """Suggest adjustments based on progress level"""
        
        adjustments = []
        
        if progress_level == ProgressLevel.NO_CHANGE:
            adjustments.extend([
                "Review and modify intervention approach",
                "Increase support and coaching",
                "Address barriers to change",
                "Consider alternative techniques"
            ])
        elif progress_level == ProgressLevel.MINIMAL_PROGRESS:
            adjustments.extend([
                "Increase intervention intensity",
                "Provide additional motivation",
                "Address specific challenges",
                "Enhance feedback frequency"
            ])
        elif progress_level in [ProgressLevel.MODERATE_PROGRESS, ProgressLevel.SIGNIFICANT_PROGRESS]:
            adjustments.extend([
                "Maintain current approach",
                "Focus on sustainability",
                "Prepare for maintenance phase",
                "Celebrate achievements"
            ])
        
        return adjustments
    
    def optimize_modification_strategy(
        self,
        strategy_id: str,
        progress_data: List[BehaviorChangeProgress],
        effectiveness_data: Dict[str, Any] = None
    ) -> ModificationOptimization:
        """
        Optimize behavior modification strategy based on progress and effectiveness data
        
        Requirements: 3.2, 3.3 - Build behavior change progress tracking and optimization
        """
        try:
            strategy = self.strategies.get(strategy_id)
            if not strategy:
                raise ValueError(f"Strategy {strategy_id} not found")
            
            effectiveness_data = effectiveness_data or {}
            
            # Calculate current effectiveness
            current_effectiveness = self._calculate_strategy_effectiveness(progress_data)
            
            # Identify optimization opportunities
            opportunities = self._identify_optimization_opportunities(
                strategy, progress_data, effectiveness_data
            )
            
            # Generate recommendations
            recommendations = self._generate_optimization_recommendations(
                strategy, opportunities, current_effectiveness
            )
            
            # Estimate expected improvement
            expected_improvement = self._estimate_optimization_impact(
                current_effectiveness, recommendations
            )
            
            # Assess implementation effort and risk
            effort_level, risk_level = self._assess_optimization_complexity(recommendations)
            
            # Calculate priority score
            priority_score = self._calculate_optimization_priority(
                expected_improvement, effort_level, risk_level, current_effectiveness
            )
            
            optimization = ModificationOptimization(
                strategy_id=strategy_id,
                current_effectiveness=current_effectiveness,
                optimization_opportunities=opportunities,
                recommended_adjustments=recommendations,
                expected_improvement=expected_improvement,
                implementation_effort=effort_level,
                risk_level=risk_level,
                priority_score=priority_score,
                analysis_date=datetime.now()
            )
            
            logger.info(f"Generated optimization recommendations for strategy {strategy_id}")
            return optimization
            
        except Exception as e:
            logger.error(f"Error optimizing modification strategy: {str(e)}")
            raise
    
    def _calculate_strategy_effectiveness(
        self,
        progress_data: List[BehaviorChangeProgress]
    ) -> float:
        """Calculate overall strategy effectiveness from progress data"""
        
        if not progress_data:
            return 0.5  # Neutral effectiveness if no data
        
        # Calculate average progress ratio
        progress_ratios = []
        for progress in progress_data:
            if progress.target_measurement > progress.baseline_measurement:
                ratio = (progress.current_measurement - progress.baseline_measurement) / \
                       (progress.target_measurement - progress.baseline_measurement)
                progress_ratios.append(max(0.0, min(1.0, ratio)))
        
        if not progress_ratios:
            return 0.5
        
        return sum(progress_ratios) / len(progress_ratios)
    
    def _identify_optimization_opportunities(
        self,
        strategy: BehaviorModificationStrategy,
        progress_data: List[BehaviorChangeProgress],
        effectiveness_data: Dict[str, Any]
    ) -> List[str]:
        """Identify opportunities for strategy optimization"""
        
        opportunities = []
        
        # Analyze progress patterns
        if progress_data:
            slow_progress = [p for p in progress_data if p.progress_level in [
                ProgressLevel.NO_CHANGE, ProgressLevel.MINIMAL_PROGRESS
            ]]
            
            if len(slow_progress) > len(progress_data) * 0.3:
                opportunities.append("High percentage of participants showing slow progress")
            
            # Check for common challenges
            all_challenges = [challenge for p in progress_data for challenge in p.challenges_encountered]
            if all_challenges:
                common_challenges = max(set(all_challenges), key=all_challenges.count)
                opportunities.append(f"Address common challenge: {common_challenges}")
        
        # Analyze technique effectiveness
        if len(strategy.techniques) > 3:
            opportunities.append("Consider reducing number of techniques for better focus")
        
        # Timeline optimization
        if strategy.timeline_weeks > 26:
            opportunities.append("Consider breaking into shorter phases")
        
        # Resource optimization
        if len(strategy.resources_required) > 10:
            opportunities.append("Optimize resource requirements")
        
        return opportunities
    
    def _generate_optimization_recommendations(
        self,
        strategy: BehaviorModificationStrategy,
        opportunities: List[str],
        current_effectiveness: float
    ) -> List[str]:
        """Generate specific optimization recommendations"""
        
        recommendations = []
        
        # Effectiveness-based recommendations
        if current_effectiveness < 0.4:
            recommendations.extend([
                "Conduct thorough review of intervention approach",
                "Increase participant engagement activities",
                "Provide additional coaching and support",
                "Consider alternative modification techniques"
            ])
        elif current_effectiveness < 0.7:
            recommendations.extend([
                "Enhance feedback mechanisms",
                "Increase intervention frequency",
                "Address identified barriers",
                "Strengthen reinforcement systems"
            ])
        
        # Opportunity-based recommendations
        for opportunity in opportunities:
            if "slow progress" in opportunity.lower():
                recommendations.append("Implement accelerated intervention protocols")
            elif "common challenge" in opportunity.lower():
                recommendations.append("Develop targeted solutions for common challenges")
            elif "techniques" in opportunity.lower():
                recommendations.append("Focus on most effective techniques")
            elif "phases" in opportunity.lower():
                recommendations.append("Implement phased approach with shorter cycles")
        
        # General optimization recommendations
        recommendations.extend([
            "Increase measurement frequency for better tracking",
            "Implement peer support networks",
            "Enhance communication about progress and benefits",
            "Provide more personalized interventions"
        ])
        
        return list(set(recommendations))  # Remove duplicates
    
    def _estimate_optimization_impact(
        self,
        current_effectiveness: float,
        recommendations: List[str]
    ) -> float:
        """Estimate expected improvement from optimization"""
        
        base_improvement = 0.1  # Base 10% improvement
        
        # Adjust based on current effectiveness (more room for improvement if low)
        if current_effectiveness < 0.3:
            base_improvement = 0.3
        elif current_effectiveness < 0.6:
            base_improvement = 0.2
        
        # Adjust based on number and type of recommendations
        recommendation_factor = min(1.5, 1.0 + (len(recommendations) * 0.05))
        
        expected_improvement = base_improvement * recommendation_factor
        
        # Ensure we don't exceed 1.0 total effectiveness
        max_possible = 1.0 - current_effectiveness
        return min(expected_improvement, max_possible)
    
    def _assess_optimization_complexity(
        self,
        recommendations: List[str]
    ) -> tuple[str, str]:
        """Assess implementation effort and risk level"""
        
        # Simple heuristic based on recommendation types
        high_effort_keywords = ["thorough review", "alternative techniques", "phased approach"]
        high_risk_keywords = ["alternative", "major change", "restructure"]
        
        effort_score = 0
        risk_score = 0
        
        for rec in recommendations:
            rec_lower = rec.lower()
            if any(keyword in rec_lower for keyword in high_effort_keywords):
                effort_score += 2
            else:
                effort_score += 1
            
            if any(keyword in rec_lower for keyword in high_risk_keywords):
                risk_score += 2
            else:
                risk_score += 1
        
        # Determine levels
        avg_effort = effort_score / len(recommendations) if recommendations else 1
        avg_risk = risk_score / len(recommendations) if recommendations else 1
        
        effort_level = "high" if avg_effort > 1.5 else "medium" if avg_effort > 1.2 else "low"
        risk_level = "high" if avg_risk > 1.5 else "medium" if avg_risk > 1.2 else "low"
        
        return effort_level, risk_level
    
    def _calculate_optimization_priority(
        self,
        expected_improvement: float,
        effort_level: str,
        risk_level: str,
        current_effectiveness: float
    ) -> float:
        """Calculate priority score for optimization"""
        
        # Base score from expected improvement
        priority_score = expected_improvement
        
        # Adjust for effort (lower effort = higher priority)
        effort_multipliers = {"low": 1.2, "medium": 1.0, "high": 0.8}
        priority_score *= effort_multipliers[effort_level]
        
        # Adjust for risk (lower risk = higher priority)
        risk_multipliers = {"low": 1.1, "medium": 1.0, "high": 0.9}
        priority_score *= risk_multipliers[risk_level]
        
        # Boost priority for very low current effectiveness
        if current_effectiveness < 0.3:
            priority_score *= 1.3
        
        return min(1.0, max(0.0, priority_score))
    
    def calculate_modification_metrics(
        self,
        organization_id: str
    ) -> BehaviorChangeMetrics:
        """Calculate comprehensive behavior modification metrics"""
        
        # Get all strategies and progress for organization
        org_strategies = [s for s in self.strategies.values() if organization_id in s.stakeholders]
        org_progress = [p for p in self.progress_tracking.values() 
                       if p.strategy_id in [s.id for s in org_strategies]]
        org_interventions = [i for i in self.interventions.values() 
                           if i.strategy_id in [s.id for s in org_strategies]]
        
        # Calculate metrics
        total_strategies = len(org_strategies)
        active_interventions = len([i for i in org_interventions 
                                  if i.status == ModificationStatus.IN_PROGRESS])
        participants_engaged = len(set(p.participant_id for p in org_progress))
        
        # Success rate calculation
        successful_progress = [p for p in org_progress 
                             if p.progress_level in [ProgressLevel.SIGNIFICANT_PROGRESS, 
                                                   ProgressLevel.COMPLETE_CHANGE]]
        success_rate = len(successful_progress) / len(org_progress) if org_progress else 0.0
        
        # Average improvement rate
        avg_improvement = sum(p.improvement_rate for p in org_progress) / len(org_progress) if org_progress else 0.0
        
        # Simplified metrics (would need more data in real implementation)
        time_to_change_avg = 8.0  # weeks
        satisfaction_avg = 0.75
        cost_per_participant = 1000.0  # dollars
        roi_achieved = 2.5  # 250% ROI
        sustainability_index = 0.8
        
        return BehaviorChangeMetrics(
            organization_id=organization_id,
            total_strategies=total_strategies,
            active_interventions=active_interventions,
            participants_engaged=participants_engaged,
            overall_success_rate=success_rate,
            average_improvement_rate=avg_improvement,
            time_to_change_average=time_to_change_avg,
            participant_satisfaction_average=satisfaction_avg,
            cost_per_participant=cost_per_participant,
            roi_achieved=roi_achieved,
            sustainability_index=sustainability_index,
            calculated_date=datetime.now()
        )
    
    def get_strategy(self, strategy_id: str) -> Optional[BehaviorModificationStrategy]:
        """Get behavior modification strategy by ID"""
        return self.strategies.get(strategy_id)
    
    def get_organization_strategies(self, organization_id: str) -> List[BehaviorModificationStrategy]:
        """Get all strategies for an organization"""
        return [
            strategy for strategy in self.strategies.values()
            if organization_id in strategy.stakeholders
        ]