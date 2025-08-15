"""
Trust Management Engine for Board Executive Mastery

This engine provides trust building and maintenance strategies for board relationships,
trust measurement and tracking system, and trust recovery and enhancement mechanisms.
"""

import logging
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
import numpy as np
from dataclasses import asdict

from ..models.credibility_models import (
    TrustLevel, TrustMetric, TrustAssessment, TrustBuildingStrategy,
    TrustRecoveryPlan, StakeholderProfile, RelationshipEvent
)

logger = logging.getLogger(__name__)


class TrustManagementEngine:
    """Engine for building, maintaining, and recovering trust in board relationships"""
    
    def __init__(self):
        self.trust_dimensions = {
            "reliability": 0.30,  # Consistency in actions and commitments
            "competence": 0.25,   # Technical and professional capability
            "benevolence": 0.25,  # Care for stakeholder interests
            "integrity": 0.20     # Honesty and ethical behavior
        }
        
        self.trust_building_strategies = {
            "relationship_focused": ["Regular communication", "Personal connection", "Shared experiences"],
            "performance_focused": ["Deliver results", "Exceed expectations", "Demonstrate competence"],
            "transparency_focused": ["Open communication", "Share challenges", "Admit mistakes"],
            "value_alignment": ["Understand values", "Align actions", "Demonstrate commitment"]
        }
    
    def assess_trust(self, stakeholder_id: str, relationship_data: Dict[str, Any]) -> TrustAssessment:
        """Assess current trust level with stakeholder"""
        try:
            metrics = []
            total_weighted_score = 0.0
            
            for dimension, weight in self.trust_dimensions.items():
                score = self._calculate_trust_dimension_score(dimension, relationship_data)
                evidence = self._extract_trust_evidence(dimension, relationship_data)
                last_interaction = self._get_last_interaction_date(dimension, relationship_data)
                trend = self._analyze_trust_trend(dimension, relationship_data)
                
                metric = TrustMetric(
                    dimension=dimension,
                    score=score,
                    evidence=evidence,
                    last_interaction=last_interaction,
                    trend=trend
                )
                metrics.append(metric)
                total_weighted_score += score * weight
            
            level = self._determine_trust_level(total_weighted_score)
            trust_drivers = self._identify_trust_drivers(metrics)
            trust_barriers = self._identify_trust_barriers(metrics)
            relationship_history = self._compile_relationship_history(relationship_data)
            
            assessment = TrustAssessment(
                stakeholder_id=stakeholder_id,
                overall_score=total_weighted_score,
                level=level,
                metrics=metrics,
                trust_drivers=trust_drivers,
                trust_barriers=trust_barriers,
                assessment_date=datetime.now(),
                relationship_history=relationship_history
            )
            
            logger.info(f"Trust assessment completed for stakeholder {stakeholder_id}: {level.value}")
            return assessment
            
        except Exception as e:
            logger.error(f"Error assessing trust: {str(e)}")
            raise
    
    def develop_trust_building_strategy(self, assessment: TrustAssessment, target_level: TrustLevel) -> TrustBuildingStrategy:
        """Develop comprehensive strategy for building trust"""
        try:
            # Determine primary strategy approach based on current trust barriers
            primary_approach = self._determine_primary_approach(assessment)
            
            # Generate key actions based on trust barriers and target level
            key_actions = self._generate_trust_building_actions(assessment, target_level, primary_approach)
            
            # Create timeline based on current and target trust levels
            timeline = self._calculate_trust_timeline(assessment.level, target_level)
            
            # Define milestones for trust building
            milestones = self._create_trust_milestones(assessment, target_level)
            
            # Identify risk factors that could undermine trust building
            risk_factors = self._identify_trust_risks(assessment)
            
            # Define success indicators
            success_indicators = self._define_success_indicators(target_level)
            
            strategy = TrustBuildingStrategy(
                id=f"trust_strategy_{assessment.stakeholder_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                stakeholder_id=assessment.stakeholder_id,
                current_trust_level=assessment.level,
                target_trust_level=target_level,
                key_actions=key_actions,
                timeline=timeline,
                milestones=milestones,
                risk_factors=risk_factors,
                success_indicators=success_indicators
            )
            
            logger.info(f"Trust building strategy developed for stakeholder {assessment.stakeholder_id}")
            return strategy
            
        except Exception as e:
            logger.error(f"Error developing trust building strategy: {str(e)}")
            raise
    
    def track_trust_progress(self, strategy: TrustBuildingStrategy, recent_events: List[RelationshipEvent]) -> Dict[str, Any]:
        """Track progress on trust building strategy"""
        try:
            progress_data = {
                'strategy_id': strategy.id,
                'stakeholder_id': strategy.stakeholder_id,
                'tracking_date': datetime.now(),
                'current_trust_level': strategy.current_trust_level.value,
                'target_trust_level': strategy.target_trust_level.value,
                'actions_completed': 0,
                'actions_in_progress': 0,
                'milestones_achieved': 0,
                'trust_trend': 'stable',
                'recent_trust_impacts': [],
                'relationship_quality_indicators': {},
                'recommendations': []
            }
            
            # Analyze recent events for trust impact
            trust_impacts = []
            for event in recent_events:
                if event.trust_impact != 0:
                    impact_data = {
                        'event_type': event.event_type,
                        'date': event.date,
                        'impact': event.trust_impact,
                        'description': event.description,
                        'lessons_learned': event.lessons_learned
                    }
                    trust_impacts.append(impact_data)
            
            progress_data['recent_trust_impacts'] = trust_impacts
            
            # Calculate trust trend based on recent impacts
            if trust_impacts:
                avg_impact = np.mean([impact['impact'] for impact in trust_impacts])
                if avg_impact > 0.05:
                    progress_data['trust_trend'] = 'improving'
                elif avg_impact < -0.05:
                    progress_data['trust_trend'] = 'declining'
            
            # Assess relationship quality indicators
            progress_data['relationship_quality_indicators'] = self._assess_relationship_quality(recent_events)
            
            # Generate progress recommendations
            progress_data['recommendations'] = self._generate_trust_progress_recommendations(strategy, progress_data)
            
            logger.info(f"Trust progress tracked for stakeholder {strategy.stakeholder_id}")
            return progress_data
            
        except Exception as e:
            logger.error(f"Error tracking trust progress: {str(e)}")
            raise
    
    def create_trust_recovery_plan(self, stakeholder_id: str, trust_breach_description: str, 
                                 current_assessment: TrustAssessment, target_level: TrustLevel) -> TrustRecoveryPlan:
        """Create plan for recovering damaged trust"""
        try:
            # Determine recovery strategy based on type of trust breach
            recovery_strategy = self._determine_recovery_strategy(trust_breach_description, current_assessment)
            
            # Generate immediate actions to address the breach
            immediate_actions = self._generate_immediate_recovery_actions(trust_breach_description, current_assessment)
            
            # Generate long-term actions to rebuild trust
            long_term_actions = self._generate_long_term_recovery_actions(current_assessment, target_level)
            
            # Calculate recovery timeline
            timeline = self._calculate_recovery_timeline(current_assessment.level, target_level, trust_breach_description)
            
            # Define success metrics for recovery
            success_metrics = self._define_recovery_success_metrics(target_level)
            
            # Create monitoring plan
            monitoring_plan = self._create_recovery_monitoring_plan()
            
            recovery_plan = TrustRecoveryPlan(
                id=f"recovery_plan_{stakeholder_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                stakeholder_id=stakeholder_id,
                trust_breach_description=trust_breach_description,
                current_trust_level=current_assessment.level,
                target_trust_level=target_level,
                recovery_strategy=recovery_strategy,
                immediate_actions=immediate_actions,
                long_term_actions=long_term_actions,
                timeline=timeline,
                success_metrics=success_metrics,
                monitoring_plan=monitoring_plan
            )
            
            logger.info(f"Trust recovery plan created for stakeholder {stakeholder_id}")
            return recovery_plan
            
        except Exception as e:
            logger.error(f"Error creating trust recovery plan: {str(e)}")
            raise
    
    def measure_trust_effectiveness(self, stakeholder_profiles: List[StakeholderProfile]) -> Dict[str, Any]:
        """Measure overall trust management effectiveness"""
        try:
            effectiveness_data = {
                'measurement_date': datetime.now(),
                'total_stakeholders': len(stakeholder_profiles),
                'trust_distribution': {},
                'average_trust_score': 0.0,
                'trust_trends': {},
                'high_trust_relationships': 0,
                'at_risk_relationships': 0,
                'improvement_opportunities': [],
                'success_stories': []
            }
            
            if not stakeholder_profiles:
                return effectiveness_data
            
            # Calculate trust distribution
            trust_levels = [profile.trust_assessment.level.value for profile in stakeholder_profiles 
                          if profile.trust_assessment]
            
            for level in TrustLevel:
                effectiveness_data['trust_distribution'][level.value] = trust_levels.count(level.value)
            
            # Calculate average trust score
            trust_scores = [profile.trust_assessment.overall_score for profile in stakeholder_profiles 
                          if profile.trust_assessment]
            if trust_scores:
                effectiveness_data['average_trust_score'] = np.mean(trust_scores)
            
            # Identify high trust and at-risk relationships
            for profile in stakeholder_profiles:
                if profile.trust_assessment:
                    if profile.trust_assessment.level in [TrustLevel.TRUSTING, TrustLevel.COMPLETE_TRUST]:
                        effectiveness_data['high_trust_relationships'] += 1
                    elif profile.trust_assessment.level in [TrustLevel.DISTRUST, TrustLevel.CAUTIOUS]:
                        effectiveness_data['at_risk_relationships'] += 1
            
            # Generate improvement opportunities
            effectiveness_data['improvement_opportunities'] = self._identify_trust_improvement_opportunities(stakeholder_profiles)
            
            # Identify success stories
            effectiveness_data['success_stories'] = self._identify_trust_success_stories(stakeholder_profiles)
            
            logger.info(f"Trust effectiveness measured for {len(stakeholder_profiles)} stakeholders")
            return effectiveness_data
            
        except Exception as e:
            logger.error(f"Error measuring trust effectiveness: {str(e)}")
            raise
    
    def _calculate_trust_dimension_score(self, dimension: str, relationship_data: Dict[str, Any]) -> float:
        """Calculate score for specific trust dimension"""
        dimension_data = relationship_data.get(dimension, {})
        
        if dimension == "reliability":
            commitment_fulfillment = dimension_data.get('commitment_fulfillment_rate', 0.5)
            consistency_score = dimension_data.get('consistency_score', 0.5)
            punctuality_score = dimension_data.get('punctuality_score', 0.5)
            return (commitment_fulfillment + consistency_score + punctuality_score) / 3
        
        elif dimension == "competence":
            technical_competence = dimension_data.get('technical_competence', 0.5)
            problem_solving_ability = dimension_data.get('problem_solving_ability', 0.5)
            decision_quality = dimension_data.get('decision_quality', 0.5)
            return (technical_competence + problem_solving_ability + decision_quality) / 3
        
        elif dimension == "benevolence":
            stakeholder_focus = dimension_data.get('stakeholder_focus', 0.5)
            support_provided = dimension_data.get('support_provided', 0.5)
            consideration_shown = dimension_data.get('consideration_shown', 0.5)
            return (stakeholder_focus + support_provided + consideration_shown) / 3
        
        elif dimension == "integrity":
            honesty_score = dimension_data.get('honesty_score', 0.5)
            ethical_behavior = dimension_data.get('ethical_behavior', 0.5)
            transparency_level = dimension_data.get('transparency_level', 0.5)
            return (honesty_score + ethical_behavior + transparency_level) / 3
        
        return 0.5  # Default neutral score
    
    def _extract_trust_evidence(self, dimension: str, relationship_data: Dict[str, Any]) -> List[str]:
        """Extract evidence for specific trust dimension"""
        dimension_data = relationship_data.get(dimension, {})
        return dimension_data.get('evidence', [])
    
    def _get_last_interaction_date(self, dimension: str, relationship_data: Dict[str, Any]) -> datetime:
        """Get last interaction date for trust dimension"""
        dimension_data = relationship_data.get(dimension, {})
        last_interaction_str = dimension_data.get('last_interaction', datetime.now().isoformat())
        
        if isinstance(last_interaction_str, str):
            try:
                return datetime.fromisoformat(last_interaction_str.replace('Z', '+00:00'))
            except:
                return datetime.now()
        return last_interaction_str if isinstance(last_interaction_str, datetime) else datetime.now()
    
    def _analyze_trust_trend(self, dimension: str, relationship_data: Dict[str, Any]) -> str:
        """Analyze trend for specific trust dimension"""
        dimension_data = relationship_data.get(dimension, {})
        historical_scores = dimension_data.get('historical_scores', [])
        
        if len(historical_scores) < 2:
            return "stable"
        
        recent_trend = np.mean(historical_scores[-3:]) - np.mean(historical_scores[-6:-3]) if len(historical_scores) >= 6 else 0
        
        if recent_trend > 0.1:
            return "improving"
        elif recent_trend < -0.1:
            return "declining"
        else:
            return "stable"
    
    def _determine_trust_level(self, score: float) -> TrustLevel:
        """Determine trust level based on score"""
        if score >= 0.85:
            return TrustLevel.COMPLETE_TRUST
        elif score >= 0.70:
            return TrustLevel.TRUSTING
        elif score >= 0.50:
            return TrustLevel.NEUTRAL
        elif score >= 0.30:
            return TrustLevel.CAUTIOUS
        else:
            return TrustLevel.DISTRUST
    
    def _identify_trust_drivers(self, metrics: List[TrustMetric]) -> List[str]:
        """Identify factors driving trust"""
        drivers = []
        for metric in metrics:
            if metric.score >= 0.75:
                drivers.append(f"Strong {metric.dimension}")
        return drivers
    
    def _identify_trust_barriers(self, metrics: List[TrustMetric]) -> List[str]:
        """Identify barriers to trust"""
        barriers = []
        for metric in metrics:
            if metric.score < 0.50:
                barriers.append(f"Weak {metric.dimension}")
        return barriers
    
    def _compile_relationship_history(self, relationship_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Compile relationship history"""
        return relationship_data.get('relationship_history', [])
    
    def _determine_primary_approach(self, assessment: TrustAssessment) -> str:
        """Determine primary approach for trust building"""
        if "competence" in [barrier.lower() for barrier in assessment.trust_barriers]:
            return "performance_focused"
        elif "integrity" in [barrier.lower() for barrier in assessment.trust_barriers]:
            return "transparency_focused"
        elif "benevolence" in [barrier.lower() for barrier in assessment.trust_barriers]:
            return "relationship_focused"
        else:
            return "value_alignment"
    
    def _generate_trust_building_actions(self, assessment: TrustAssessment, target_level: TrustLevel, approach: str) -> List[str]:
        """Generate trust building actions"""
        base_actions = self.trust_building_strategies.get(approach, [])
        
        # Add specific actions based on trust barriers
        specific_actions = []
        for barrier in assessment.trust_barriers:
            if "reliability" in barrier.lower():
                specific_actions.extend(["Meet all commitments", "Provide regular updates", "Be punctual"])
            elif "competence" in barrier.lower():
                specific_actions.extend(["Demonstrate expertise", "Deliver quality results", "Seek continuous improvement"])
            elif "benevolence" in barrier.lower():
                specific_actions.extend(["Show genuine interest", "Provide support", "Consider stakeholder needs"])
            elif "integrity" in barrier.lower():
                specific_actions.extend(["Be transparent", "Admit mistakes", "Act ethically"])
        
        return list(set(base_actions + specific_actions))
    
    def _calculate_trust_timeline(self, current_level: TrustLevel, target_level: TrustLevel) -> str:
        """Calculate timeline for trust building"""
        level_order = [TrustLevel.DISTRUST, TrustLevel.CAUTIOUS, TrustLevel.NEUTRAL, TrustLevel.TRUSTING, TrustLevel.COMPLETE_TRUST]
        
        current_index = level_order.index(current_level)
        target_index = level_order.index(target_level)
        level_gap = target_index - current_index
        
        if level_gap <= 0:
            return "1-2 months"
        elif level_gap == 1:
            return "3-6 months"
        elif level_gap == 2:
            return "6-12 months"
        else:
            return "12-18 months"
    
    def _create_trust_milestones(self, assessment: TrustAssessment, target_level: TrustLevel) -> List[Dict[str, Any]]:
        """Create trust building milestones"""
        milestones = []
        current_score = assessment.overall_score
        target_score = self._get_trust_target_score(target_level)
        
        score_increment = (target_score - current_score) / 3
        
        for i in range(1, 4):
            milestone_score = current_score + (score_increment * i)
            milestones.append({
                'milestone': f'Trust Milestone {i}',
                'target_score': milestone_score,
                'target_date': datetime.now() + timedelta(weeks=i*6),
                'achieved': False,
                'description': f'Achieve trust score of {milestone_score:.2f}',
                'key_indicators': self._get_milestone_indicators(milestone_score)
            })
        
        return milestones
    
    def _get_trust_target_score(self, level: TrustLevel) -> float:
        """Get target score for trust level"""
        level_scores = {
            TrustLevel.DISTRUST: 0.25,
            TrustLevel.CAUTIOUS: 0.40,
            TrustLevel.NEUTRAL: 0.60,
            TrustLevel.TRUSTING: 0.80,
            TrustLevel.COMPLETE_TRUST: 0.95
        }
        return level_scores.get(level, 0.60)
    
    def _get_milestone_indicators(self, score: float) -> List[str]:
        """Get indicators for trust milestone"""
        if score >= 0.8:
            return ["Proactive communication", "Seeking advice", "Collaborative decision-making"]
        elif score >= 0.6:
            return ["Regular interaction", "Positive feedback", "Increased engagement"]
        else:
            return ["Basic communication", "Meeting attendance", "Neutral responses"]
    
    def _identify_trust_risks(self, assessment: TrustAssessment) -> List[str]:
        """Identify risk factors for trust building"""
        risks = []
        
        if assessment.level == TrustLevel.DISTRUST:
            risks.append("Previous trust breach may create skepticism")
        
        if "reliability" in [barrier.lower() for barrier in assessment.trust_barriers]:
            risks.append("Failure to meet commitments could worsen trust")
        
        if "integrity" in [barrier.lower() for barrier in assessment.trust_barriers]:
            risks.append("Any perceived dishonesty could be devastating")
        
        risks.extend([
            "External pressures may affect relationship",
            "Competing priorities may limit attention",
            "Organizational changes may disrupt progress"
        ])
        
        return risks
    
    def _define_success_indicators(self, target_level: TrustLevel) -> List[str]:
        """Define success indicators for trust level"""
        if target_level == TrustLevel.COMPLETE_TRUST:
            return [
                "Stakeholder seeks advice proactively",
                "Confidential information is shared",
                "Stakeholder advocates for your positions",
                "Informal communication increases"
            ]
        elif target_level == TrustLevel.TRUSTING:
            return [
                "Positive feedback in meetings",
                "Increased collaboration requests",
                "Support for initiatives",
                "Regular check-ins initiated by stakeholder"
            ]
        else:
            return [
                "Neutral to positive interactions",
                "Meeting attendance and participation",
                "Responsive to communications",
                "No negative feedback"
            ]
    
    def _determine_recovery_strategy(self, breach_description: str, assessment: TrustAssessment) -> str:
        """Determine recovery strategy based on breach type"""
        breach_lower = breach_description.lower()
        
        if any(word in breach_lower for word in ["lie", "dishonest", "mislead", "deceive"]):
            return "transparency_and_accountability"
        elif any(word in breach_lower for word in ["failed", "missed", "late", "unreliable"]):
            return "reliability_rebuilding"
        elif any(word in breach_lower for word in ["incompetent", "mistake", "error", "poor_decision"]):
            return "competence_demonstration"
        elif any(word in breach_lower for word in ["selfish", "uncaring", "ignored", "dismissed"]):
            return "relationship_repair"
        else:
            return "comprehensive_recovery"
    
    def _generate_immediate_recovery_actions(self, breach_description: str, assessment: TrustAssessment) -> List[str]:
        """Generate immediate actions for trust recovery"""
        actions = [
            "Acknowledge the issue directly and take responsibility",
            "Apologize sincerely without making excuses",
            "Explain what went wrong and why",
            "Outline immediate corrective measures"
        ]
        
        breach_lower = breach_description.lower()
        
        if "communication" in breach_lower:
            actions.append("Establish regular communication schedule")
        if "deadline" in breach_lower or "late" in breach_lower:
            actions.append("Provide realistic timeline for resolution")
        if "quality" in breach_lower:
            actions.append("Implement quality assurance measures")
        
        return actions
    
    def _generate_long_term_recovery_actions(self, assessment: TrustAssessment, target_level: TrustLevel) -> List[str]:
        """Generate long-term actions for trust recovery"""
        actions = [
            "Consistently deliver on all commitments",
            "Provide regular progress updates",
            "Seek feedback and act on it",
            "Demonstrate continuous improvement",
            "Build personal relationship through regular interaction"
        ]
        
        # Add specific actions based on trust barriers
        for barrier in assessment.trust_barriers:
            if "reliability" in barrier.lower():
                actions.append("Implement robust project management processes")
            elif "competence" in barrier.lower():
                actions.append("Invest in skill development and training")
            elif "integrity" in barrier.lower():
                actions.append("Establish transparent reporting mechanisms")
        
        return list(set(actions))
    
    def _calculate_recovery_timeline(self, current_level: TrustLevel, target_level: TrustLevel, breach_description: str) -> str:
        """Calculate timeline for trust recovery"""
        base_timeline = self._calculate_trust_timeline(current_level, target_level)
        
        # Recovery typically takes longer than initial trust building
        if "severe" in breach_description.lower() or current_level == TrustLevel.DISTRUST:
            return f"{base_timeline} (extended due to trust breach)"
        else:
            return base_timeline
    
    def _define_recovery_success_metrics(self, target_level: TrustLevel) -> List[str]:
        """Define success metrics for trust recovery"""
        base_metrics = self._define_success_indicators(target_level)
        recovery_metrics = [
            "No mention of past breach in recent interactions",
            "Stakeholder comfort level visibly improved",
            "Willingness to engage in new initiatives",
            "Positive references to recent performance"
        ]
        return base_metrics + recovery_metrics
    
    def _create_recovery_monitoring_plan(self) -> str:
        """Create monitoring plan for trust recovery"""
        return """
        Weekly: Monitor stakeholder interactions and feedback
        Bi-weekly: Assess progress against recovery milestones
        Monthly: Conduct formal trust assessment
        Quarterly: Review and adjust recovery strategy
        """
    
    def _assess_relationship_quality(self, recent_events: List[RelationshipEvent]) -> Dict[str, Any]:
        """Assess relationship quality indicators"""
        quality_indicators = {
            'interaction_frequency': 0,
            'positive_interactions': 0,
            'negative_interactions': 0,
            'collaboration_instances': 0,
            'conflict_instances': 0,
            'communication_quality': 'neutral'
        }
        
        for event in recent_events:
            quality_indicators['interaction_frequency'] += 1
            
            if event.trust_impact > 0:
                quality_indicators['positive_interactions'] += 1
            elif event.trust_impact < 0:
                quality_indicators['negative_interactions'] += 1
            
            if event.event_type in ['collaboration', 'joint_decision', 'partnership']:
                quality_indicators['collaboration_instances'] += 1
            elif event.event_type in ['conflict', 'disagreement', 'tension']:
                quality_indicators['conflict_instances'] += 1
        
        # Determine overall communication quality
        if quality_indicators['positive_interactions'] > quality_indicators['negative_interactions']:
            quality_indicators['communication_quality'] = 'positive'
        elif quality_indicators['negative_interactions'] > quality_indicators['positive_interactions']:
            quality_indicators['communication_quality'] = 'negative'
        
        return quality_indicators
    
    def _generate_trust_progress_recommendations(self, strategy: TrustBuildingStrategy, progress_data: Dict[str, Any]) -> List[str]:
        """Generate recommendations for trust progress"""
        recommendations = []
        
        if progress_data['trust_trend'] == 'declining':
            recommendations.append("Immediate intervention needed - trust is declining")
            recommendations.append("Review recent interactions for potential issues")
        
        if progress_data['relationship_quality_indicators']['negative_interactions'] > 0:
            recommendations.append("Address negative interactions with targeted recovery actions")
        
        if progress_data['milestones_achieved'] == 0:
            recommendations.append("Focus on achieving first trust milestone")
        
        if len(progress_data['recent_trust_impacts']) == 0:
            recommendations.append("Increase interaction frequency to build trust momentum")
        
        return recommendations
    
    def _identify_trust_improvement_opportunities(self, stakeholder_profiles: List[StakeholderProfile]) -> List[str]:
        """Identify trust improvement opportunities"""
        opportunities = []
        
        low_trust_count = sum(1 for profile in stakeholder_profiles 
                            if profile.trust_assessment and profile.trust_assessment.level in [TrustLevel.DISTRUST, TrustLevel.CAUTIOUS])
        
        if low_trust_count > 0:
            opportunities.append(f"Focus on {low_trust_count} low-trust relationships")
        
        # Identify common trust barriers
        all_barriers = []
        for profile in stakeholder_profiles:
            if profile.trust_assessment:
                all_barriers.extend(profile.trust_assessment.trust_barriers)
        
        if all_barriers:
            from collections import Counter
            common_barriers = Counter(all_barriers).most_common(3)
            for barrier, count in common_barriers:
                opportunities.append(f"Address common barrier: {barrier} (affects {count} relationships)")
        
        return opportunities
    
    def _identify_trust_success_stories(self, stakeholder_profiles: List[StakeholderProfile]) -> List[str]:
        """Identify trust success stories"""
        success_stories = []
        
        for profile in stakeholder_profiles:
            if profile.trust_assessment and profile.trust_assessment.level in [TrustLevel.TRUSTING, TrustLevel.COMPLETE_TRUST]:
                success_stories.append(f"High trust relationship with {profile.name} ({profile.role})")
        
        return success_stories