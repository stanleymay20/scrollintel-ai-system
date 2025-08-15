"""
Culture Maintenance Engine

Handles cultural change sustainability assessment, maintenance strategy development,
and long-term culture health monitoring and optimization.
"""

import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from dataclasses import asdict

from ..models.culture_maintenance_models import (
    SustainabilityAssessment, MaintenanceStrategy, CultureMaintenancePlan,
    MaintenanceIntervention, LongTermMonitoringResult, CultureHealthIndicator,
    SustainabilityLevel, MaintenanceStatus
)
from ..models.cultural_assessment_models import CultureMap, CulturalTransformation


class CultureMaintenanceEngine:
    """Engine for cultural sustainability and maintenance"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.sustainability_thresholds = {
            "high": 0.8,
            "medium": 0.6,
            "low": 0.4,
            "critical": 0.2
        }
        
    def assess_cultural_sustainability(
        self, 
        organization_id: str,
        transformation: CulturalTransformation,
        current_culture: CultureMap
    ) -> SustainabilityAssessment:
        """Assess sustainability of cultural changes"""
        try:
            # Create a mock target culture for analysis
            target_culture = CultureMap(
                organization_id=organization_id,
                assessment_date=datetime.now(),
                cultural_dimensions={},
                values=[],
                behaviors=[],
                norms=[],
                subcultures=[],
                health_metrics=[],
                overall_health_score=0.85,  # Default target health score
                assessment_confidence=0.8,
                data_sources=["target_definition"]
            )
            
            # Analyze culture health indicators
            health_indicators = self._analyze_health_indicators(
                current_culture, target_culture
            )
            
            # Identify risk and protective factors
            risk_factors = self._identify_risk_factors(
                current_culture, transformation
            )
            protective_factors = self._identify_protective_factors(
                current_culture, transformation
            )
            
            # Calculate overall sustainability score
            overall_score = self._calculate_sustainability_score(
                health_indicators, risk_factors, protective_factors
            )
            
            # Determine sustainability level
            sustainability_level = self._determine_sustainability_level(overall_score)
            
            assessment = SustainabilityAssessment(
                assessment_id=f"sustain_assess_{organization_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                organization_id=organization_id,
                transformation_id=transformation.id,
                sustainability_level=sustainability_level,
                risk_factors=risk_factors,
                protective_factors=protective_factors,
                health_indicators=health_indicators,
                overall_score=overall_score,
                assessment_date=datetime.now(),
                next_assessment_due=datetime.now() + timedelta(days=90)
            )
            
            self.logger.info(f"Cultural sustainability assessed for {organization_id}")
            return assessment
            
        except Exception as e:
            self.logger.error(f"Error assessing cultural sustainability: {str(e)}")
            raise
    
    def develop_maintenance_strategy(
        self,
        organization_id: str,
        sustainability_assessment: SustainabilityAssessment,
        target_culture: CultureMap
    ) -> List[MaintenanceStrategy]:
        """Develop comprehensive culture maintenance strategies"""
        try:
            strategies = []
            
            # Strategy for high-risk areas
            if sustainability_assessment.risk_factors:
                risk_strategy = self._create_risk_mitigation_strategy(
                    organization_id, sustainability_assessment.risk_factors
                )
                strategies.append(risk_strategy)
            
            # Strategy for reinforcing protective factors
            if sustainability_assessment.protective_factors:
                reinforcement_strategy = self._create_reinforcement_strategy(
                    organization_id, sustainability_assessment.protective_factors
                )
                strategies.append(reinforcement_strategy)
            
            # Strategy for continuous improvement
            improvement_strategy = self._create_improvement_strategy(
                organization_id, target_culture
            )
            strategies.append(improvement_strategy)
            
            # Strategy for monitoring and early detection
            monitoring_strategy = self._create_monitoring_strategy(
                organization_id, sustainability_assessment.health_indicators
            )
            strategies.append(monitoring_strategy)
            
            self.logger.info(f"Developed {len(strategies)} maintenance strategies for {organization_id}")
            return strategies
            
        except Exception as e:
            self.logger.error(f"Error developing maintenance strategy: {str(e)}")
            raise
    
    def create_maintenance_plan(
        self,
        organization_id: str,
        sustainability_assessment: SustainabilityAssessment,
        maintenance_strategies: List[MaintenanceStrategy]
    ) -> CultureMaintenancePlan:
        """Create comprehensive culture maintenance plan"""
        try:
            # Create monitoring framework
            monitoring_framework = self._design_monitoring_framework(
                sustainability_assessment.health_indicators
            )
            
            # Define intervention triggers
            intervention_triggers = self._define_intervention_triggers(
                sustainability_assessment
            )
            
            # Calculate resource allocation
            resource_allocation = self._calculate_resource_allocation(
                maintenance_strategies
            )
            
            # Create implementation timeline
            timeline = self._create_maintenance_timeline(maintenance_strategies)
            
            plan = CultureMaintenancePlan(
                plan_id=f"maint_plan_{organization_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                organization_id=organization_id,
                sustainability_assessment=sustainability_assessment,
                maintenance_strategies=maintenance_strategies,
                monitoring_framework=monitoring_framework,
                intervention_triggers=intervention_triggers,
                resource_allocation=resource_allocation,
                timeline=timeline,
                status=MaintenanceStatus.STABLE,
                created_date=datetime.now(),
                last_updated=datetime.now()
            )
            
            self.logger.info(f"Created maintenance plan for {organization_id}")
            return plan
            
        except Exception as e:
            self.logger.error(f"Error creating maintenance plan: {str(e)}")
            raise
    
    def monitor_long_term_health(
        self,
        organization_id: str,
        maintenance_plan: CultureMaintenancePlan,
        monitoring_period_days: int = 90
    ) -> LongTermMonitoringResult:
        """Monitor long-term culture health and optimization"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=monitoring_period_days)
            
            # Collect health trend data
            health_trends = self._collect_health_trends(
                organization_id, start_date, end_date
            )
            
            # Calculate sustainability metrics
            sustainability_metrics = self._calculate_sustainability_metrics(
                health_trends, maintenance_plan.sustainability_assessment
            )
            
            # Identify risk indicators
            risk_indicators = self._identify_current_risks(
                health_trends, sustainability_metrics
            )
            
            # Generate recommendations
            recommendations = self._generate_maintenance_recommendations(
                health_trends, risk_indicators, maintenance_plan
            )
            
            # Define next actions
            next_actions = self._define_next_actions(
                recommendations, maintenance_plan
            )
            
            monitoring_result = LongTermMonitoringResult(
                monitoring_id=f"monitor_{organization_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                organization_id=organization_id,
                monitoring_period={"start": start_date, "end": end_date},
                health_trends=health_trends,
                sustainability_metrics=sustainability_metrics,
                risk_indicators=risk_indicators,
                recommendations=recommendations,
                next_actions=next_actions,
                monitoring_date=datetime.now()
            )
            
            self.logger.info(f"Completed long-term health monitoring for {organization_id}")
            return monitoring_result
            
        except Exception as e:
            self.logger.error(f"Error monitoring long-term health: {str(e)}")
            raise
    
    def _analyze_health_indicators(
        self, 
        current_culture: CultureMap, 
        target_culture: CultureMap
    ) -> List[CultureHealthIndicator]:
        """Analyze culture health indicators"""
        indicators = []
        
        # Employee engagement indicator
        engagement_indicator = CultureHealthIndicator(
            indicator_id="engagement",
            name="Employee Engagement",
            current_value=current_culture.overall_health_score * 0.8,
            target_value=target_culture.overall_health_score * 0.8,
            trend="stable",
            importance_weight=0.3,
            measurement_date=datetime.now()
        )
        indicators.append(engagement_indicator)
        
        # Cultural alignment indicator
        alignment_indicator = CultureHealthIndicator(
            indicator_id="alignment",
            name="Cultural Alignment",
            current_value=len(current_culture.values) / max(len(target_culture.values), 1),
            target_value=1.0,
            trend="improving",
            importance_weight=0.25,
            measurement_date=datetime.now()
        )
        indicators.append(alignment_indicator)
        
        # Behavioral consistency indicator
        behavior_indicator = CultureHealthIndicator(
            indicator_id="behavior_consistency",
            name="Behavioral Consistency",
            current_value=len(current_culture.behaviors) / max(len(target_culture.behaviors), 1),
            target_value=1.0,
            trend="stable",
            importance_weight=0.2,
            measurement_date=datetime.now()
        )
        indicators.append(behavior_indicator)
        
        return indicators
    
    def _identify_risk_factors(
        self, 
        current_culture: CultureMap, 
        transformation: CulturalTransformation
    ) -> List[str]:
        """Identify factors that risk cultural sustainability"""
        risk_factors = []
        
        if current_culture.overall_health_score < 0.7:
            risk_factors.append("Low overall culture health score")
        
        if len(current_culture.subcultures) > 3:
            risk_factors.append("High number of subcultures creating fragmentation")
        
        if transformation.progress < 0.8:
            risk_factors.append("Incomplete transformation implementation")
        
        return risk_factors
    
    def _identify_protective_factors(
        self, 
        current_culture: CultureMap, 
        transformation: CulturalTransformation
    ) -> List[str]:
        """Identify factors that protect cultural sustainability"""
        protective_factors = []
        
        if current_culture.overall_health_score > 0.8:
            protective_factors.append("Strong overall culture health")
        
        if len(current_culture.values) >= 5:
            protective_factors.append("Well-defined value system")
        
        if transformation.progress > 0.9:
            protective_factors.append("Successful transformation implementation")
        
        return protective_factors
    
    def _calculate_sustainability_score(
        self,
        health_indicators: List[CultureHealthIndicator],
        risk_factors: List[str],
        protective_factors: List[str]
    ) -> float:
        """Calculate overall sustainability score"""
        # Base score from health indicators
        weighted_score = sum(
            indicator.current_value * indicator.importance_weight 
            for indicator in health_indicators
        )
        
        # Adjust for risk factors
        risk_penalty = len(risk_factors) * 0.1
        
        # Adjust for protective factors
        protection_bonus = len(protective_factors) * 0.05
        
        final_score = max(0.0, min(1.0, weighted_score - risk_penalty + protection_bonus))
        return final_score
    
    def _determine_sustainability_level(self, score: float) -> SustainabilityLevel:
        """Determine sustainability level from score"""
        if score >= self.sustainability_thresholds["high"]:
            return SustainabilityLevel.HIGH
        elif score >= self.sustainability_thresholds["medium"]:
            return SustainabilityLevel.MEDIUM
        elif score >= self.sustainability_thresholds["low"]:
            return SustainabilityLevel.LOW
        else:
            return SustainabilityLevel.CRITICAL
    
    def _create_risk_mitigation_strategy(
        self, 
        organization_id: str, 
        risk_factors: List[str]
    ) -> MaintenanceStrategy:
        """Create strategy to mitigate identified risks"""
        activities = []
        for risk in risk_factors:
            if "health score" in risk.lower():
                activities.append({
                    "type": "health_improvement",
                    "description": "Implement culture health improvement initiatives",
                    "frequency": "monthly"
                })
            elif "subculture" in risk.lower():
                activities.append({
                    "type": "alignment_sessions",
                    "description": "Conduct subculture alignment sessions",
                    "frequency": "quarterly"
                })
        
        return MaintenanceStrategy(
            strategy_id=f"risk_mitigation_{organization_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            organization_id=organization_id,
            target_culture_elements=["risk_mitigation"],
            maintenance_activities=activities,
            monitoring_schedule={"frequency": "monthly", "metrics": ["risk_levels"]},
            resource_requirements={"time": "20_hours_monthly", "budget": "medium"},
            success_metrics=["reduced_risk_factors", "improved_health_scores"],
            review_frequency="monthly",
            created_date=datetime.now()
        )
    
    def _create_reinforcement_strategy(
        self, 
        organization_id: str, 
        protective_factors: List[str]
    ) -> MaintenanceStrategy:
        """Create strategy to reinforce protective factors"""
        activities = [
            {
                "type": "recognition_programs",
                "description": "Implement programs to recognize and reinforce positive cultural behaviors",
                "frequency": "ongoing"
            },
            {
                "type": "success_storytelling",
                "description": "Share success stories that highlight protective factors",
                "frequency": "monthly"
            }
        ]
        
        return MaintenanceStrategy(
            strategy_id=f"reinforcement_{organization_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            organization_id=organization_id,
            target_culture_elements=["protective_factors"],
            maintenance_activities=activities,
            monitoring_schedule={"frequency": "monthly", "metrics": ["factor_strength"]},
            resource_requirements={"time": "15_hours_monthly", "budget": "low"},
            success_metrics=["strengthened_protective_factors", "sustained_positive_behaviors"],
            review_frequency="quarterly",
            created_date=datetime.now()
        )
    
    def _create_improvement_strategy(
        self, 
        organization_id: str, 
        target_culture: CultureMap
    ) -> MaintenanceStrategy:
        """Create strategy for continuous cultural improvement"""
        activities = [
            {
                "type": "continuous_feedback",
                "description": "Implement continuous feedback mechanisms for culture improvement",
                "frequency": "ongoing"
            },
            {
                "type": "innovation_sessions",
                "description": "Conduct cultural innovation and improvement sessions",
                "frequency": "quarterly"
            }
        ]
        
        return MaintenanceStrategy(
            strategy_id=f"improvement_{organization_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            organization_id=organization_id,
            target_culture_elements=["continuous_improvement"],
            maintenance_activities=activities,
            monitoring_schedule={"frequency": "quarterly", "metrics": ["improvement_rate"]},
            resource_requirements={"time": "25_hours_quarterly", "budget": "medium"},
            success_metrics=["culture_evolution", "innovation_adoption"],
            review_frequency="quarterly",
            created_date=datetime.now()
        )
    
    def _create_monitoring_strategy(
        self, 
        organization_id: str, 
        health_indicators: List[CultureHealthIndicator]
    ) -> MaintenanceStrategy:
        """Create strategy for ongoing monitoring and early detection"""
        activities = [
            {
                "type": "health_monitoring",
                "description": "Regular monitoring of culture health indicators",
                "frequency": "weekly"
            },
            {
                "type": "early_warning_system",
                "description": "Implement early warning system for culture risks",
                "frequency": "ongoing"
            }
        ]
        
        return MaintenanceStrategy(
            strategy_id=f"monitoring_{organization_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            organization_id=organization_id,
            target_culture_elements=["monitoring_and_detection"],
            maintenance_activities=activities,
            monitoring_schedule={"frequency": "weekly", "metrics": ["all_health_indicators"]},
            resource_requirements={"time": "10_hours_weekly", "budget": "low"},
            success_metrics=["early_detection_rate", "response_time"],
            review_frequency="monthly",
            created_date=datetime.now()
        )
    
    def _design_monitoring_framework(
        self, 
        health_indicators: List[CultureHealthIndicator]
    ) -> Dict[str, Any]:
        """Design comprehensive monitoring framework"""
        return {
            "indicators": [indicator.indicator_id for indicator in health_indicators],
            "measurement_frequency": "weekly",
            "reporting_schedule": "monthly",
            "alert_thresholds": {
                "critical": 0.3,
                "warning": 0.5,
                "good": 0.8
            },
            "data_sources": ["surveys", "behavioral_metrics", "performance_data"],
            "analysis_methods": ["trend_analysis", "correlation_analysis", "predictive_modeling"]
        }
    
    def _define_intervention_triggers(
        self, 
        sustainability_assessment: SustainabilityAssessment
    ) -> List[Dict[str, Any]]:
        """Define triggers for maintenance interventions"""
        triggers = [
            {
                "trigger_type": "health_score_decline",
                "condition": "health_score < 0.6",
                "intervention": "immediate_assessment_and_action",
                "priority": "high"
            },
            {
                "trigger_type": "risk_factor_increase",
                "condition": "new_risk_factors_detected",
                "intervention": "risk_mitigation_activation",
                "priority": "medium"
            },
            {
                "trigger_type": "sustainability_level_drop",
                "condition": "sustainability_level_decreases",
                "intervention": "comprehensive_review_and_adjustment",
                "priority": "high"
            }
        ]
        return triggers
    
    def _calculate_resource_allocation(
        self, 
        maintenance_strategies: List[MaintenanceStrategy]
    ) -> Dict[str, Any]:
        """Calculate resource allocation for maintenance activities"""
        total_time = sum(
            self._extract_time_requirement(strategy.resource_requirements.get("time", "0"))
            for strategy in maintenance_strategies
        )
        
        budget_levels = [
            strategy.resource_requirements.get("budget", "low")
            for strategy in maintenance_strategies
        ]
        
        return {
            "total_time_hours_monthly": total_time,
            "budget_requirement": max(budget_levels) if budget_levels else "low",
            "personnel_needed": max(2, len(maintenance_strategies)),
            "tools_and_systems": ["monitoring_dashboard", "survey_platform", "analytics_tools"]
        }
    
    def _extract_time_requirement(self, time_str: str) -> int:
        """Extract numeric time requirement from string"""
        try:
            return int(time_str.split("_")[0])
        except:
            return 10  # Default fallback
    
    def _create_maintenance_timeline(
        self, 
        maintenance_strategies: List[MaintenanceStrategy]
    ) -> Dict[str, Any]:
        """Create implementation timeline for maintenance activities"""
        return {
            "immediate_actions": ["setup_monitoring", "establish_baselines"],
            "first_month": ["implement_risk_mitigation", "activate_reinforcement"],
            "ongoing": ["continuous_monitoring", "regular_assessments"],
            "quarterly_reviews": ["strategy_effectiveness", "plan_adjustments"],
            "annual_reviews": ["comprehensive_assessment", "strategic_updates"]
        }
    
    def _collect_health_trends(
        self, 
        organization_id: str, 
        start_date: datetime, 
        end_date: datetime
    ) -> Dict[str, List[float]]:
        """Collect health trend data over monitoring period"""
        # Simulate trend data collection
        return {
            "engagement": [0.75, 0.78, 0.76, 0.79, 0.81],
            "alignment": [0.70, 0.72, 0.74, 0.73, 0.75],
            "behavior_consistency": [0.68, 0.70, 0.72, 0.74, 0.76],
            "overall_health": [0.71, 0.73, 0.74, 0.75, 0.77]
        }
    
    def _calculate_sustainability_metrics(
        self, 
        health_trends: Dict[str, List[float]], 
        sustainability_assessment: SustainabilityAssessment
    ) -> Dict[str, float]:
        """Calculate current sustainability metrics"""
        metrics = {}
        
        for metric, values in health_trends.items():
            if values:
                metrics[f"{metric}_current"] = values[-1]
                metrics[f"{metric}_trend"] = (values[-1] - values[0]) / len(values) if len(values) > 1 else 0
                metrics[f"{metric}_stability"] = 1.0 - (max(values) - min(values))
        
        return metrics
    
    def _identify_current_risks(
        self, 
        health_trends: Dict[str, List[float]], 
        sustainability_metrics: Dict[str, float]
    ) -> List[str]:
        """Identify current risk indicators"""
        risks = []
        
        for metric, value in sustainability_metrics.items():
            if "current" in metric and value < 0.6:
                risks.append(f"Low {metric.replace('_current', '')} score")
            elif "trend" in metric and value < -0.05:
                risks.append(f"Declining {metric.replace('_trend', '')} trend")
        
        return risks
    
    def _generate_maintenance_recommendations(
        self, 
        health_trends: Dict[str, List[float]], 
        risk_indicators: List[str], 
        maintenance_plan: CultureMaintenancePlan
    ) -> List[str]:
        """Generate maintenance recommendations based on monitoring results"""
        recommendations = []
        
        if risk_indicators:
            recommendations.append("Activate immediate risk mitigation protocols")
            recommendations.append("Increase monitoring frequency for at-risk areas")
        
        # Check for positive trends
        positive_trends = [
            metric for metric, values in health_trends.items()
            if len(values) > 1 and values[-1] > values[0]
        ]
        
        if positive_trends:
            recommendations.append("Reinforce successful strategies showing positive trends")
        
        recommendations.append("Continue regular monitoring and assessment schedule")
        
        return recommendations
    
    def _define_next_actions(
        self, 
        recommendations: List[str], 
        maintenance_plan: CultureMaintenancePlan
    ) -> List[Dict[str, Any]]:
        """Define specific next actions based on recommendations"""
        actions = []
        
        for recommendation in recommendations:
            if "risk mitigation" in recommendation.lower():
                actions.append({
                    "action": "activate_risk_protocols",
                    "priority": "high",
                    "timeline": "immediate",
                    "responsible": "culture_team"
                })
            elif "monitoring" in recommendation.lower():
                actions.append({
                    "action": "adjust_monitoring_frequency",
                    "priority": "medium",
                    "timeline": "within_week",
                    "responsible": "monitoring_team"
                })
        
        return actions