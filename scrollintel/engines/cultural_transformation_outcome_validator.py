"""
Cultural Transformation Outcome Validator

This engine validates transformation outcomes, measures success against goals,
and assesses the overall impact and quality of cultural transformations.
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
from scipy import stats
import logging

logger = logging.getLogger(__name__)


@dataclass
class TransformationSuccessAssessment:
    """Assessment of transformation success"""
    overall_success_score: float
    criteria_met_percentage: float
    success_level: str
    exceeded_targets: List[str]
    underperformed_areas: List[str]
    success_factors: List[str]
    improvement_recommendations: List[str]


@dataclass
class GoalAchievementAssessment:
    """Assessment of goal achievement"""
    achievement_rate: float
    fully_achieved_goals: List[str]
    partially_achieved_goals: List[str]
    unmet_goals: List[str]
    weighted_success_score: float
    goal_results: List[Dict[str, Any]]


@dataclass
class TransformationImpactAssessment:
    """Assessment of transformation impact"""
    net_impact_score: float
    attribution_confidence: float
    significant_improvements: List[str]
    impact_breakdown: List[Dict[str, Any]]
    external_factor_adjustments: Dict[str, float]


@dataclass
class BenchmarkAssessment:
    """Assessment against benchmarks"""
    overall_ranking: str
    percentile_score: float
    top_quartile_metrics: List[str]
    below_average_metrics: List[str]
    competitive_position: str


@dataclass
class QualityAssessment:
    """Assessment of transformation quality"""
    overall_quality_score: float
    depth_score: float
    integration_score: float
    quality_level: str
    quality_indicators: Dict[str, float]


class CulturalTransformationOutcomeValidator:
    """Validates cultural transformation outcomes and measures success"""
    
    def __init__(self):
        self.success_thresholds = {
            "exceptional": 0.9,
            "high": 0.8,
            "moderate": 0.7,
            "low": 0.6
        }
        self.quality_thresholds = {
            "exceptional": 0.9,
            "high": 0.8,
            "moderate": 0.7,
            "basic": 0.6
        }
    
    def measure_transformation_success(self, transformation_outcome) -> TransformationSuccessAssessment:
        """
        Measure overall transformation success
        
        Args:
            transformation_outcome: TransformationOutcome object with results
        
        Returns:
            TransformationSuccessAssessment with success metrics
        """
        try:
            # Calculate overall success score
            success_scores = []
            exceeded_targets = []
            underperformed_areas = []
            
            for criterion in transformation_outcome.success_criteria:
                metric_name = criterion["metric"]
                threshold = criterion["threshold"]
                weight = criterion.get("weight", 1.0)
                
                achieved_value = transformation_outcome.achieved_metrics.get(metric_name, 0)
                target_value = transformation_outcome.target_metrics.get(metric_name, threshold)
                
                # Calculate success for this criterion
                if achieved_value >= threshold:
                    criterion_success = min(1.0, achieved_value / target_value)
                    if achieved_value > target_value:
                        exceeded_targets.append(metric_name)
                else:
                    criterion_success = achieved_value / threshold
                    underperformed_areas.append(metric_name)
                
                success_scores.append(criterion_success * weight)
            
            overall_success_score = np.mean(success_scores) if success_scores else 0
            criteria_met_percentage = len([s for s in success_scores if s >= 1.0]) / len(success_scores) if success_scores else 0
            
            # Determine success level
            success_level = self._determine_success_level(overall_success_score)
            
            # Identify success factors
            success_factors = self._identify_success_factors(transformation_outcome, exceeded_targets)
            
            # Generate improvement recommendations
            improvement_recommendations = self._generate_improvement_recommendations(
                underperformed_areas, transformation_outcome
            )
            
            return TransformationSuccessAssessment(
                overall_success_score=overall_success_score,
                criteria_met_percentage=criteria_met_percentage,
                success_level=success_level,
                exceeded_targets=exceeded_targets,
                underperformed_areas=underperformed_areas,
                success_factors=success_factors,
                improvement_recommendations=improvement_recommendations
            )
            
        except Exception as e:
            logger.error(f"Error measuring transformation success: {str(e)}")
            return TransformationSuccessAssessment(
                overall_success_score=0.0,
                criteria_met_percentage=0.0,
                success_level="failed",
                exceeded_targets=[],
                underperformed_areas=[],
                success_factors=[],
                improvement_recommendations=[]
            )
    
    def validate_goal_achievement(self,
                                target_metrics: Dict[str, float],
                                achieved_metrics: Dict[str, float],
                                success_criteria: List[Dict[str, Any]]) -> GoalAchievementAssessment:
        """
        Validate achievement of specific goals
        
        Args:
            target_metrics: Target values for metrics
            achieved_metrics: Achieved values for metrics
            success_criteria: Success criteria with thresholds and weights
        
        Returns:
            GoalAchievementAssessment with goal achievement details
        """
        try:
            fully_achieved_goals = []
            partially_achieved_goals = []
            unmet_goals = []
            goal_results = []
            weighted_scores = []
            
            for criterion in success_criteria:
                metric_name = criterion["metric"]
                threshold = criterion["threshold"]
                weight = criterion.get("weight", 1.0)
                
                target_value = target_metrics.get(metric_name, threshold)
                achieved_value = achieved_metrics.get(metric_name, 0)
                
                # Calculate achievement level
                achievement_ratio = achieved_value / target_value if target_value > 0 else 0
                success_margin = achieved_value - target_value
                
                goal_result = {
                    "metric_name": metric_name,
                    "target_value": target_value,
                    "achieved_value": achieved_value,
                    "threshold": threshold,
                    "achievement_ratio": achievement_ratio,
                    "success_margin": success_margin,
                    "weight": weight
                }
                
                # Categorize achievement
                if achieved_value >= target_value:
                    fully_achieved_goals.append(metric_name)
                    weighted_scores.append(weight * 1.0)
                elif achieved_value >= threshold:
                    partially_achieved_goals.append(metric_name)
                    weighted_scores.append(weight * (achieved_value / target_value))
                else:
                    unmet_goals.append(metric_name)
                    weighted_scores.append(weight * (achieved_value / threshold))
                
                goal_results.append(goal_result)
            
            achievement_rate = len(fully_achieved_goals) / len(success_criteria) if success_criteria else 0
            weighted_success_score = np.mean(weighted_scores) if weighted_scores else 0
            
            return GoalAchievementAssessment(
                achievement_rate=achievement_rate,
                fully_achieved_goals=fully_achieved_goals,
                partially_achieved_goals=partially_achieved_goals,
                unmet_goals=unmet_goals,
                weighted_success_score=weighted_success_score,
                goal_results=goal_results
            )
            
        except Exception as e:
            logger.error(f"Error validating goal achievement: {str(e)}")
            return GoalAchievementAssessment(
                achievement_rate=0.0,
                fully_achieved_goals=[],
                partially_achieved_goals=[],
                unmet_goals=[],
                weighted_success_score=0.0,
                goal_results=[]
            )
    
    def measure_transformation_impact(self,
                                    transformation_outcome,
                                    impact_data: Dict[str, Any]) -> TransformationImpactAssessment:
        """
        Measure the impact of transformation on organizational performance
        
        Args:
            transformation_outcome: TransformationOutcome object
            impact_data: Data about baseline and post-transformation performance
        
        Returns:
            TransformationImpactAssessment with impact analysis
        """
        try:
            baseline_performance = impact_data.get("baseline_performance", {})
            post_transformation_performance = impact_data.get("post_transformation_performance", {})
            external_factors = impact_data.get("external_factors", [])
            
            impact_breakdown = []
            significant_improvements = []
            net_impacts = []
            
            # Calculate impact for each metric
            for metric, post_value in post_transformation_performance.items():
                baseline_value = baseline_performance.get(metric, 0)
                raw_improvement = post_value - baseline_value
                improvement_percentage = (raw_improvement / baseline_value) if baseline_value > 0 else 0
                
                # Adjust for external factors
                external_adjustment = sum(
                    factor.get("impact", 0) for factor in external_factors 
                    if factor.get("factor") == metric
                )
                
                net_improvement = raw_improvement - external_adjustment
                transformation_attribution = net_improvement / raw_improvement if raw_improvement > 0 else 1.0
                transformation_attribution = max(0, min(1, transformation_attribution))
                
                impact_breakdown.append({
                    "metric": metric,
                    "baseline_value": baseline_value,
                    "post_value": post_value,
                    "raw_improvement": raw_improvement,
                    "improvement_percentage": improvement_percentage,
                    "external_adjustment": external_adjustment,
                    "net_improvement": net_improvement,
                    "transformation_attribution": transformation_attribution
                })
                
                if improvement_percentage >= 0.1:  # 10% improvement threshold
                    significant_improvements.append(metric)
                
                net_impacts.append(net_improvement / baseline_value if baseline_value > 0 else 0)
            
            net_impact_score = np.mean(net_impacts) if net_impacts else 0
            attribution_confidence = np.mean([
                item["transformation_attribution"] for item in impact_breakdown
            ]) if impact_breakdown else 0
            
            external_factor_adjustments = {
                factor["factor"]: factor["impact"] for factor in external_factors
            }
            
            return TransformationImpactAssessment(
                net_impact_score=net_impact_score,
                attribution_confidence=attribution_confidence,
                significant_improvements=significant_improvements,
                impact_breakdown=impact_breakdown,
                external_factor_adjustments=external_factor_adjustments
            )
            
        except Exception as e:
            logger.error(f"Error measuring transformation impact: {str(e)}")
            return TransformationImpactAssessment(
                net_impact_score=0.0,
                attribution_confidence=0.0,
                significant_improvements=[],
                impact_breakdown=[],
                external_factor_adjustments={}
            )
    
    def validate_against_benchmarks(self,
                                  transformation_results: Dict[str, float],
                                  benchmark_data: Dict[str, Any]) -> BenchmarkAssessment:
        """
        Validate transformation results against industry benchmarks
        
        Args:
            transformation_results: Results achieved by the transformation
            benchmark_data: Industry benchmark data
        
        Returns:
            BenchmarkAssessment with benchmark comparison
        """
        try:
            industry_averages = benchmark_data.get("industry_averages", {})
            top_quartile = benchmark_data.get("top_quartile", {})
            
            above_average_count = 0
            top_quartile_metrics = []
            below_average_metrics = []
            percentile_scores = []
            
            for metric, result in transformation_results.items():
                industry_avg = industry_averages.get(metric, 0)
                top_quartile_value = top_quartile.get(metric, industry_avg * 1.5)
                
                if result >= top_quartile_value:
                    top_quartile_metrics.append(metric)
                    percentile_scores.append(90)
                elif result >= industry_avg:
                    above_average_count += 1
                    percentile_scores.append(70)
                else:
                    below_average_metrics.append(metric)
                    percentile_scores.append(30)
            
            overall_percentile = np.mean(percentile_scores) if percentile_scores else 50
            
            # Determine overall ranking
            if len(top_quartile_metrics) >= len(transformation_results) * 0.5:
                overall_ranking = "top_quartile"
            elif above_average_count >= len(transformation_results) * 0.7:
                overall_ranking = "above_average"
            elif len(below_average_metrics) <= len(transformation_results) * 0.3:
                overall_ranking = "average"
            else:
                overall_ranking = "below_average"
            
            # Determine competitive position
            if overall_percentile >= 80:
                competitive_position = "market_leader"
            elif overall_percentile >= 60:
                competitive_position = "strong_performer"
            elif overall_percentile >= 40:
                competitive_position = "average_performer"
            else:
                competitive_position = "underperformer"
            
            return BenchmarkAssessment(
                overall_ranking=overall_ranking,
                percentile_score=overall_percentile,
                top_quartile_metrics=top_quartile_metrics,
                below_average_metrics=below_average_metrics,
                competitive_position=competitive_position
            )
            
        except Exception as e:
            logger.error(f"Error validating against benchmarks: {str(e)}")
            return BenchmarkAssessment(
                overall_ranking="unknown",
                percentile_score=50.0,
                top_quartile_metrics=[],
                below_average_metrics=[],
                competitive_position="unknown"
            )
    
    def assess_transformation_quality(self,
                                    transformation_outcome,
                                    quality_indicators: Dict[str, float]) -> QualityAssessment:
        """
        Assess the quality and depth of transformation
        
        Args:
            transformation_outcome: TransformationOutcome object
            quality_indicators: Quality indicators and their scores
        
        Returns:
            QualityAssessment with quality analysis
        """
        try:
            # Calculate depth score (how deep the changes go)
            depth_indicators = [
                "behavior_change_depth",
                "value_internalization",
                "process_embedding"
            ]
            depth_scores = [quality_indicators.get(indicator, 0) for indicator in depth_indicators]
            depth_score = np.mean(depth_scores) if depth_scores else 0
            
            # Calculate integration score (how well integrated the changes are)
            integration_indicators = [
                "cultural_integration",
                "leadership_alignment",
                "employee_adoption"
            ]
            integration_scores = [quality_indicators.get(indicator, 0) for indicator in integration_indicators]
            integration_score = np.mean(integration_scores) if integration_scores else 0
            
            # Calculate overall quality score
            all_scores = list(quality_indicators.values())
            overall_quality_score = np.mean(all_scores) if all_scores else 0
            
            # Determine quality level
            quality_level = self._determine_quality_level(overall_quality_score)
            
            return QualityAssessment(
                overall_quality_score=overall_quality_score,
                depth_score=depth_score,
                integration_score=integration_score,
                quality_level=quality_level,
                quality_indicators=quality_indicators
            )
            
        except Exception as e:
            logger.error(f"Error assessing transformation quality: {str(e)}")
            return QualityAssessment(
                overall_quality_score=0.0,
                depth_score=0.0,
                integration_score=0.0,
                quality_level="poor",
                quality_indicators={}
            )
    
    def _determine_success_level(self, success_score: float) -> str:
        """Determine success level based on score"""
        for level, threshold in sorted(self.success_thresholds.items(), 
                                     key=lambda x: x[1], reverse=True):
            if success_score >= threshold:
                return level
        return "failed"
    
    def _determine_quality_level(self, quality_score: float) -> str:
        """Determine quality level based on score"""
        for level, threshold in sorted(self.quality_thresholds.items(), 
                                     key=lambda x: x[1], reverse=True):
            if quality_score >= threshold:
                return level
        return "poor"
    
    def _identify_success_factors(self, transformation_outcome, exceeded_targets: List[str]) -> List[str]:
        """Identify key success factors"""
        success_factors = []
        
        if len(exceeded_targets) >= 3:
            success_factors.append("comprehensive_goal_achievement")
        
        if transformation_outcome.achieved_metrics.get("employee_engagement", 0) >= 0.85:
            success_factors.append("strong_employee_engagement")
        
        if transformation_outcome.achieved_metrics.get("cultural_health", 0) >= 0.8:
            success_factors.append("healthy_cultural_foundation")
        
        # Add more success factor identification logic
        return success_factors
    
    def _generate_improvement_recommendations(self, 
                                            underperformed_areas: List[str],
                                            transformation_outcome) -> List[str]:
        """Generate improvement recommendations"""
        recommendations = []
        
        for area in underperformed_areas:
            if area == "employee_engagement":
                recommendations.append("Enhance communication and involvement strategies")
            elif area == "cultural_health":
                recommendations.append("Strengthen cultural reinforcement mechanisms")
            elif area == "performance_improvement":
                recommendations.append("Focus on performance-culture alignment")
            elif area == "innovation_index":
                recommendations.append("Implement innovation-supporting cultural practices")
        
        return recommendations