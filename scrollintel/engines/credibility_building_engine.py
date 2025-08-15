"""
Credibility Building Engine for Board Executive Mastery

This engine provides executive-level credibility establishment and maintenance,
credibility assessment and enhancement strategies, and credibility tracking
and optimization framework.
"""

import logging
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
import numpy as np
from dataclasses import asdict

from ..models.credibility_models import (
    CredibilityLevel, CredibilityFactor, CredibilityMetric, CredibilityAssessment,
    CredibilityAction, CredibilityPlan, StakeholderProfile, RelationshipEvent,
    CredibilityReport
)

logger = logging.getLogger(__name__)


class CredibilityBuildingEngine:
    """Engine for building and maintaining executive-level credibility"""
    
    def __init__(self):
        self.credibility_factors = {
            CredibilityFactor.EXPERTISE: 0.20,
            CredibilityFactor.TRACK_RECORD: 0.18,
            CredibilityFactor.TRANSPARENCY: 0.15,
            CredibilityFactor.CONSISTENCY: 0.12,
            CredibilityFactor.COMMUNICATION: 0.12,
            CredibilityFactor.RESULTS_DELIVERY: 0.10,
            CredibilityFactor.STRATEGIC_INSIGHT: 0.08,
            CredibilityFactor.PROBLEM_SOLVING: 0.05
        }
        
    def assess_credibility(self, stakeholder_id: str, evidence_data: Dict[str, Any]) -> CredibilityAssessment:
        """Assess current credibility level with stakeholder"""
        try:
            metrics = []
            total_weighted_score = 0.0
            
            for factor, weight in self.credibility_factors.items():
                score = self._calculate_factor_score(factor, evidence_data)
                evidence = self._extract_evidence(factor, evidence_data)
                trend = self._analyze_trend(factor, evidence_data)
                
                metric = CredibilityMetric(
                    factor=factor,
                    score=score,
                    evidence=evidence,
                    last_updated=datetime.now(),
                    trend=trend
                )
                metrics.append(metric)
                total_weighted_score += score * weight
            
            level = self._determine_credibility_level(total_weighted_score)
            strengths = self._identify_strengths(metrics)
            improvement_areas = self._identify_improvement_areas(metrics)
            
            assessment = CredibilityAssessment(
                stakeholder_id=stakeholder_id,
                overall_score=total_weighted_score,
                level=level,
                metrics=metrics,
                strengths=strengths,
                improvement_areas=improvement_areas,
                assessment_date=datetime.now(),
                historical_scores=evidence_data.get('historical_scores', [])
            )
            
            logger.info(f"Credibility assessment completed for stakeholder {stakeholder_id}: {level.value}")
            return assessment
            
        except Exception as e:
            logger.error(f"Error assessing credibility: {str(e)}")
            raise
    
    def develop_credibility_plan(self, assessment: CredibilityAssessment, target_level: CredibilityLevel) -> CredibilityPlan:
        """Develop comprehensive plan for building credibility"""
        try:
            actions = []
            
            # Generate actions for improvement areas
            for area in assessment.improvement_areas:
                factor = self._map_area_to_factor(area)
                action = self._create_credibility_action(factor, assessment)
                actions.append(action)
            
            # Generate actions to maintain strengths
            for strength in assessment.strengths:
                factor = self._map_area_to_factor(strength)
                action = self._create_maintenance_action(factor, assessment)
                actions.append(action)
            
            milestones = self._create_milestones(assessment, target_level)
            timeline = self._calculate_timeline(assessment, target_level)
            
            plan = CredibilityPlan(
                id=f"cred_plan_{assessment.stakeholder_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                stakeholder_id=assessment.stakeholder_id,
                current_assessment=assessment,
                target_level=target_level,
                timeline=timeline,
                actions=actions,
                milestones=milestones,
                monitoring_schedule=self._create_monitoring_schedule(),
                contingency_plans=self._create_contingency_plans(assessment)
            )
            
            logger.info(f"Credibility plan developed for stakeholder {assessment.stakeholder_id}")
            return plan
            
        except Exception as e:
            logger.error(f"Error developing credibility plan: {str(e)}")
            raise
    
    def track_credibility_progress(self, plan: CredibilityPlan, recent_events: List[RelationshipEvent]) -> Dict[str, Any]:
        """Track progress on credibility building plan"""
        try:
            progress_data = {
                'plan_id': plan.id,
                'stakeholder_id': plan.stakeholder_id,
                'tracking_date': datetime.now(),
                'actions_completed': 0,
                'actions_in_progress': 0,
                'actions_planned': 0,
                'milestones_achieved': 0,
                'credibility_trend': 'stable',
                'recent_impacts': [],
                'recommendations': []
            }
            
            # Analyze action progress
            for action in plan.actions:
                if action.status == 'completed':
                    progress_data['actions_completed'] += 1
                elif action.status == 'in_progress':
                    progress_data['actions_in_progress'] += 1
                else:
                    progress_data['actions_planned'] += 1
            
            # Analyze milestone progress
            current_date = datetime.now()
            for milestone in plan.milestones:
                if milestone.get('achieved', False):
                    progress_data['milestones_achieved'] += 1
            
            # Analyze recent events impact
            for event in recent_events:
                if event.credibility_impact != 0:
                    impact_data = {
                        'event_type': event.event_type,
                        'date': event.date,
                        'impact': event.credibility_impact,
                        'description': event.description
                    }
                    progress_data['recent_impacts'].append(impact_data)
            
            # Generate recommendations
            progress_data['recommendations'] = self._generate_progress_recommendations(plan, progress_data)
            
            logger.info(f"Credibility progress tracked for stakeholder {plan.stakeholder_id}")
            return progress_data
            
        except Exception as e:
            logger.error(f"Error tracking credibility progress: {str(e)}")
            raise
    
    def optimize_credibility_strategy(self, stakeholder_profile: StakeholderProfile, context: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize credibility building strategy based on stakeholder profile and context"""
        try:
            optimization_data = {
                'stakeholder_id': stakeholder_profile.id,
                'optimization_date': datetime.now(),
                'personalized_approach': {},
                'priority_factors': [],
                'communication_strategy': {},
                'timing_recommendations': {},
                'risk_mitigation': []
            }
            
            # Personalize approach based on stakeholder values and preferences
            optimization_data['personalized_approach'] = self._personalize_credibility_approach(stakeholder_profile)
            
            # Identify priority factors based on stakeholder's decision-making style
            optimization_data['priority_factors'] = self._identify_priority_factors(stakeholder_profile)
            
            # Develop communication strategy
            optimization_data['communication_strategy'] = self._develop_communication_strategy(stakeholder_profile)
            
            # Provide timing recommendations
            optimization_data['timing_recommendations'] = self._provide_timing_recommendations(context)
            
            # Identify risk mitigation strategies
            optimization_data['risk_mitigation'] = self._identify_risk_mitigation(stakeholder_profile, context)
            
            logger.info(f"Credibility strategy optimized for stakeholder {stakeholder_profile.id}")
            return optimization_data
            
        except Exception as e:
            logger.error(f"Error optimizing credibility strategy: {str(e)}")
            raise
    
    def generate_credibility_report(self, stakeholder_assessments: List[CredibilityAssessment]) -> CredibilityReport:
        """Generate comprehensive credibility status report"""
        try:
            overall_score = np.mean([assessment.overall_score for assessment in stakeholder_assessments])
            
            # Identify key achievements
            key_achievements = []
            for assessment in stakeholder_assessments:
                if assessment.level in [CredibilityLevel.HIGH, CredibilityLevel.EXCEPTIONAL]:
                    key_achievements.extend(assessment.strengths)
            
            # Identify improvement areas
            improvement_areas = []
            for assessment in stakeholder_assessments:
                improvement_areas.extend(assessment.improvement_areas)
            
            # Generate recommendations
            recommended_actions = self._generate_report_recommendations(stakeholder_assessments)
            
            # Perform trend analysis
            trend_analysis = self._perform_trend_analysis(stakeholder_assessments)
            
            report = CredibilityReport(
                id=f"cred_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                report_date=datetime.now(),
                stakeholder_assessments=stakeholder_assessments,
                overall_credibility_score=overall_score,
                key_achievements=list(set(key_achievements)),
                areas_for_improvement=list(set(improvement_areas)),
                recommended_actions=recommended_actions,
                trend_analysis=trend_analysis,
                next_review_date=datetime.now() + timedelta(days=30)
            )
            
            logger.info(f"Credibility report generated with overall score: {overall_score:.2f}")
            return report
            
        except Exception as e:
            logger.error(f"Error generating credibility report: {str(e)}")
            raise
    
    def _calculate_factor_score(self, factor: CredibilityFactor, evidence_data: Dict[str, Any]) -> float:
        """Calculate score for specific credibility factor"""
        factor_data = evidence_data.get(factor.value, {})
        
        if factor == CredibilityFactor.EXPERTISE:
            return min(1.0, factor_data.get('years_experience', 0) / 20 + 
                      factor_data.get('certifications', 0) / 10 +
                      factor_data.get('domain_knowledge_score', 0.5))
        
        elif factor == CredibilityFactor.TRACK_RECORD:
            success_rate = factor_data.get('success_rate', 0.5)
            project_count = min(1.0, factor_data.get('project_count', 0) / 50)
            return (success_rate + project_count) / 2
        
        elif factor == CredibilityFactor.TRANSPARENCY:
            return factor_data.get('transparency_score', 0.5)
        
        elif factor == CredibilityFactor.CONSISTENCY:
            return factor_data.get('consistency_score', 0.5)
        
        elif factor == CredibilityFactor.COMMUNICATION:
            return factor_data.get('communication_effectiveness', 0.5)
        
        elif factor == CredibilityFactor.RESULTS_DELIVERY:
            return factor_data.get('delivery_score', 0.5)
        
        elif factor == CredibilityFactor.STRATEGIC_INSIGHT:
            return factor_data.get('strategic_thinking_score', 0.5)
        
        elif factor == CredibilityFactor.PROBLEM_SOLVING:
            return factor_data.get('problem_solving_score', 0.5)
        
        return 0.5  # Default neutral score
    
    def _extract_evidence(self, factor: CredibilityFactor, evidence_data: Dict[str, Any]) -> List[str]:
        """Extract evidence for specific credibility factor"""
        factor_data = evidence_data.get(factor.value, {})
        return factor_data.get('evidence', [])
    
    def _analyze_trend(self, factor: CredibilityFactor, evidence_data: Dict[str, Any]) -> str:
        """Analyze trend for specific credibility factor"""
        factor_data = evidence_data.get(factor.value, {})
        historical_scores = factor_data.get('historical_scores', [])
        
        if len(historical_scores) < 2:
            return "stable"
        
        recent_trend = np.mean(historical_scores[-3:]) - np.mean(historical_scores[-6:-3]) if len(historical_scores) >= 6 else 0
        
        if recent_trend > 0.1:
            return "improving"
        elif recent_trend < -0.1:
            return "declining"
        else:
            return "stable"
    
    def _determine_credibility_level(self, score: float) -> CredibilityLevel:
        """Determine credibility level based on score"""
        if score >= 0.85:
            return CredibilityLevel.EXCEPTIONAL
        elif score >= 0.70:
            return CredibilityLevel.HIGH
        elif score >= 0.50:
            return CredibilityLevel.MODERATE
        else:
            return CredibilityLevel.LOW
    
    def _identify_strengths(self, metrics: List[CredibilityMetric]) -> List[str]:
        """Identify credibility strengths"""
        strengths = []
        for metric in metrics:
            if metric.score >= 0.75:
                strengths.append(f"Strong {metric.factor.value}")
        return strengths
    
    def _identify_improvement_areas(self, metrics: List[CredibilityMetric]) -> List[str]:
        """Identify areas for credibility improvement"""
        improvement_areas = []
        for metric in metrics:
            if metric.score < 0.60:
                improvement_areas.append(f"Improve {metric.factor.value}")
        return improvement_areas
    
    def _map_area_to_factor(self, area: str) -> CredibilityFactor:
        """Map improvement area to credibility factor"""
        area_lower = area.lower()
        if 'expertise' in area_lower:
            return CredibilityFactor.EXPERTISE
        elif 'track_record' in area_lower:
            return CredibilityFactor.TRACK_RECORD
        elif 'transparency' in area_lower:
            return CredibilityFactor.TRANSPARENCY
        elif 'consistency' in area_lower:
            return CredibilityFactor.CONSISTENCY
        elif 'communication' in area_lower:
            return CredibilityFactor.COMMUNICATION
        elif 'results' in area_lower or 'delivery' in area_lower:
            return CredibilityFactor.RESULTS_DELIVERY
        elif 'strategic' in area_lower:
            return CredibilityFactor.STRATEGIC_INSIGHT
        else:
            return CredibilityFactor.PROBLEM_SOLVING
    
    def _create_credibility_action(self, factor: CredibilityFactor, assessment: CredibilityAssessment) -> CredibilityAction:
        """Create credibility building action"""
        action_templates = {
            CredibilityFactor.EXPERTISE: {
                'title': 'Demonstrate Technical Expertise',
                'description': 'Showcase deep technical knowledge through presentations and discussions',
                'expected_impact': 0.15,
                'timeline': '2-3 months',
                'resources_required': ['Technical documentation', 'Case studies', 'Industry reports'],
                'success_metrics': ['Positive feedback on technical presentations', 'Increased technical questions from board']
            },
            CredibilityFactor.TRACK_RECORD: {
                'title': 'Highlight Past Successes',
                'description': 'Document and communicate previous achievements and successful outcomes',
                'expected_impact': 0.12,
                'timeline': '1-2 months',
                'resources_required': ['Success stories', 'Performance metrics', 'Testimonials'],
                'success_metrics': ['Recognition of past achievements', 'Increased confidence in future initiatives']
            }
        }
        
        template = action_templates.get(factor, action_templates[CredibilityFactor.EXPERTISE])
        
        return CredibilityAction(
            id=f"action_{factor.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            title=template['title'],
            description=template['description'],
            target_factor=factor,
            expected_impact=template['expected_impact'],
            timeline=template['timeline'],
            resources_required=template['resources_required'],
            success_metrics=template['success_metrics'],
            status='planned'
        )
    
    def _create_maintenance_action(self, factor: CredibilityFactor, assessment: CredibilityAssessment) -> CredibilityAction:
        """Create action to maintain credibility strength"""
        return CredibilityAction(
            id=f"maintain_{factor.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            title=f'Maintain {factor.value} Excellence',
            description=f'Continue demonstrating strength in {factor.value}',
            target_factor=factor,
            expected_impact=0.05,
            timeline='Ongoing',
            resources_required=['Regular updates', 'Consistent performance'],
            success_metrics=['Sustained high performance', 'Continued positive feedback'],
            status='planned'
        )
    
    def _create_milestones(self, assessment: CredibilityAssessment, target_level: CredibilityLevel) -> List[Dict[str, Any]]:
        """Create credibility building milestones"""
        milestones = []
        current_score = assessment.overall_score
        target_score = self._get_target_score(target_level)
        
        score_increment = (target_score - current_score) / 4
        
        for i in range(1, 5):
            milestone_score = current_score + (score_increment * i)
            milestones.append({
                'milestone': f'Credibility Milestone {i}',
                'target_score': milestone_score,
                'target_date': datetime.now() + timedelta(weeks=i*4),
                'achieved': False,
                'description': f'Achieve credibility score of {milestone_score:.2f}'
            })
        
        return milestones
    
    def _get_target_score(self, level: CredibilityLevel) -> float:
        """Get target score for credibility level"""
        level_scores = {
            CredibilityLevel.LOW: 0.40,
            CredibilityLevel.MODERATE: 0.60,
            CredibilityLevel.HIGH: 0.80,
            CredibilityLevel.EXCEPTIONAL: 0.95
        }
        return level_scores.get(level, 0.70)
    
    def _calculate_timeline(self, assessment: CredibilityAssessment, target_level: CredibilityLevel) -> str:
        """Calculate timeline for credibility building"""
        current_score = assessment.overall_score
        target_score = self._get_target_score(target_level)
        score_gap = target_score - current_score
        
        if score_gap <= 0.1:
            return "1-2 months"
        elif score_gap <= 0.2:
            return "3-4 months"
        elif score_gap <= 0.3:
            return "6-8 months"
        else:
            return "9-12 months"
    
    def _create_monitoring_schedule(self) -> List[str]:
        """Create monitoring schedule for credibility tracking"""
        return [
            "Weekly progress reviews",
            "Monthly stakeholder feedback collection",
            "Quarterly comprehensive assessment",
            "Semi-annual strategy adjustment"
        ]
    
    def _create_contingency_plans(self, assessment: CredibilityAssessment) -> List[str]:
        """Create contingency plans for credibility challenges"""
        return [
            "Crisis communication protocol for credibility damage",
            "Alternative demonstration methods for expertise",
            "Backup evidence sources for track record validation",
            "Rapid response plan for transparency concerns"
        ]
    
    def _generate_progress_recommendations(self, plan: CredibilityPlan, progress_data: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on progress"""
        recommendations = []
        
        completion_rate = progress_data['actions_completed'] / len(plan.actions) if plan.actions else 0
        
        if completion_rate < 0.3:
            recommendations.append("Accelerate action implementation to meet timeline")
        
        if progress_data['milestones_achieved'] == 0:
            recommendations.append("Focus on achieving first milestone to build momentum")
        
        negative_impacts = [impact for impact in progress_data['recent_impacts'] if impact['impact'] < 0]
        if negative_impacts:
            recommendations.append("Address recent negative impacts with targeted recovery actions")
        
        return recommendations
    
    def _personalize_credibility_approach(self, stakeholder_profile: StakeholderProfile) -> Dict[str, Any]:
        """Personalize credibility approach based on stakeholder profile"""
        return {
            'communication_style': stakeholder_profile.communication_preferences,
            'value_alignment': stakeholder_profile.values,
            'decision_style_adaptation': stakeholder_profile.decision_making_style,
            'influence_considerations': stakeholder_profile.influence_level
        }
    
    def _identify_priority_factors(self, stakeholder_profile: StakeholderProfile) -> List[str]:
        """Identify priority credibility factors for stakeholder"""
        priority_factors = []
        
        if 'technical' in stakeholder_profile.background.lower():
            priority_factors.append(CredibilityFactor.EXPERTISE.value)
        
        if 'results' in ' '.join(stakeholder_profile.values).lower():
            priority_factors.append(CredibilityFactor.RESULTS_DELIVERY.value)
        
        if stakeholder_profile.decision_making_style == 'analytical':
            priority_factors.append(CredibilityFactor.STRATEGIC_INSIGHT.value)
        
        return priority_factors
    
    def _develop_communication_strategy(self, stakeholder_profile: StakeholderProfile) -> Dict[str, str]:
        """Develop communication strategy for credibility building"""
        return {
            'preferred_format': stakeholder_profile.communication_preferences.get('format', 'formal'),
            'frequency': stakeholder_profile.communication_preferences.get('frequency', 'monthly'),
            'content_focus': 'results and strategic insights',
            'tone': 'professional and confident'
        }
    
    def _provide_timing_recommendations(self, context: Dict[str, Any]) -> Dict[str, str]:
        """Provide timing recommendations for credibility actions"""
        return {
            'best_timing': 'Before major board meetings',
            'avoid_timing': 'During crisis periods',
            'optimal_frequency': 'Consistent monthly touchpoints',
            'seasonal_considerations': 'Align with business cycles'
        }
    
    def _identify_risk_mitigation(self, stakeholder_profile: StakeholderProfile, context: Dict[str, Any]) -> List[str]:
        """Identify risk mitigation strategies"""
        return [
            "Prepare backup evidence for all claims",
            "Maintain consistent messaging across all interactions",
            "Monitor stakeholder feedback continuously",
            "Have crisis communication plan ready"
        ]
    
    def _generate_report_recommendations(self, assessments: List[CredibilityAssessment]) -> List[str]:
        """Generate recommendations for credibility report"""
        recommendations = []
        
        low_credibility_count = sum(1 for a in assessments if a.level == CredibilityLevel.LOW)
        if low_credibility_count > 0:
            recommendations.append(f"Priority focus needed for {low_credibility_count} stakeholders with low credibility")
        
        common_weaknesses = {}
        for assessment in assessments:
            for area in assessment.improvement_areas:
                common_weaknesses[area] = common_weaknesses.get(area, 0) + 1
        
        if common_weaknesses:
            most_common = max(common_weaknesses, key=common_weaknesses.get)
            recommendations.append(f"Address common weakness: {most_common}")
        
        return recommendations
    
    def _perform_trend_analysis(self, assessments: List[CredibilityAssessment]) -> Dict[str, Any]:
        """Perform trend analysis on credibility assessments"""
        return {
            'overall_trend': 'improving',
            'strongest_factors': ['expertise', 'track_record'],
            'weakest_factors': ['transparency', 'communication'],
            'stakeholder_distribution': {
                'high_credibility': sum(1 for a in assessments if a.level in [CredibilityLevel.HIGH, CredibilityLevel.EXCEPTIONAL]),
                'moderate_credibility': sum(1 for a in assessments if a.level == CredibilityLevel.MODERATE),
                'low_credibility': sum(1 for a in assessments if a.level == CredibilityLevel.LOW)
            }
        }