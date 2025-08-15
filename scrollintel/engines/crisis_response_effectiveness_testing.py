"""
Crisis Response Effectiveness Testing Engine

This module provides comprehensive testing and validation of crisis response effectiveness,
measuring speed, quality, outcomes, and leadership performance during crisis situations.
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import json
import statistics
from abc import ABC, abstractmethod

class EffectivenessMetric(Enum):
    RESPONSE_SPEED = "response_speed"
    DECISION_QUALITY = "decision_quality"
    COMMUNICATION_CLARITY = "communication_clarity"
    RESOURCE_UTILIZATION = "resource_utilization"
    STAKEHOLDER_SATISFACTION = "stakeholder_satisfaction"
    OUTCOME_SUCCESS = "outcome_success"
    LEADERSHIP_EFFECTIVENESS = "leadership_effectiveness"

class TestingPhase(Enum):
    DETECTION = "detection"
    RESPONSE = "response"
    COMMUNICATION = "communication"
    RESOLUTION = "resolution"
    RECOVERY = "recovery"

@dataclass
class EffectivenessScore:
    metric: EffectivenessMetric
    score: float  # 0.0 to 1.0
    details: Dict[str, Any]
    measurement_time: datetime
    confidence_level: float

@dataclass
class CrisisResponseTest:
    test_id: str
    crisis_scenario: str
    test_type: str
    start_time: datetime
    end_time: Optional[datetime] = None
    phases_tested: List[TestingPhase] = field(default_factory=list)
    effectiveness_scores: List[EffectivenessScore] = field(default_factory=list)
    overall_score: Optional[float] = None
    recommendations: List[str] = field(default_factory=list)

class CrisisResponseEffectivenessTesting:
    """
    Comprehensive testing system for crisis response effectiveness
    """
    
    def __init__(self):
        self.active_tests: Dict[str, CrisisResponseTest] = {}
        self.test_history: List[CrisisResponseTest] = []
        self.baseline_metrics: Dict[EffectivenessMetric, float] = {}
        self.performance_thresholds = self._initialize_thresholds()
    
    def _initialize_thresholds(self) -> Dict[EffectivenessMetric, Dict[str, float]]:
        """Initialize performance thresholds for different effectiveness metrics"""
        return {
            EffectivenessMetric.RESPONSE_SPEED: {
                "excellent": 0.9,
                "good": 0.7,
                "acceptable": 0.5,
                "poor": 0.3
            },
            EffectivenessMetric.DECISION_QUALITY: {
                "excellent": 0.95,
                "good": 0.8,
                "acceptable": 0.6,
                "poor": 0.4
            },
            EffectivenessMetric.COMMUNICATION_CLARITY: {
                "excellent": 0.9,
                "good": 0.75,
                "acceptable": 0.6,
                "poor": 0.4
            },
            EffectivenessMetric.RESOURCE_UTILIZATION: {
                "excellent": 0.85,
                "good": 0.7,
                "acceptable": 0.55,
                "poor": 0.35
            },
            EffectivenessMetric.STAKEHOLDER_SATISFACTION: {
                "excellent": 0.9,
                "good": 0.75,
                "acceptable": 0.6,
                "poor": 0.4
            },
            EffectivenessMetric.OUTCOME_SUCCESS: {
                "excellent": 0.95,
                "good": 0.8,
                "acceptable": 0.65,
                "poor": 0.4
            },
            EffectivenessMetric.LEADERSHIP_EFFECTIVENESS: {
                "excellent": 0.9,
                "good": 0.75,
                "acceptable": 0.6,
                "poor": 0.4
            }
        }
    
    async def start_effectiveness_test(
        self,
        crisis_scenario: str,
        test_type: str = "comprehensive"
    ) -> str:
        """Start a new crisis response effectiveness test"""
        test_id = f"effectiveness_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        test = CrisisResponseTest(
            test_id=test_id,
            crisis_scenario=crisis_scenario,
            test_type=test_type,
            start_time=datetime.now()
        )
        
        self.active_tests[test_id] = test
        return test_id
    
    async def measure_response_speed(
        self,
        test_id: str,
        detection_time: datetime,
        first_response_time: datetime,
        full_response_time: datetime
    ) -> EffectivenessScore:
        """Measure crisis response speed effectiveness"""
        if test_id not in self.active_tests:
            raise ValueError(f"Test {test_id} not found")
        
        # Calculate speed metrics
        detection_to_first_response = (first_response_time - detection_time).total_seconds()
        detection_to_full_response = (full_response_time - detection_time).total_seconds()
        
        # Define target response times (in seconds)
        target_first_response = 300  # 5 minutes
        target_full_response = 1800  # 30 minutes
        
        # Calculate speed scores
        first_response_score = max(0, 1 - (detection_to_first_response / target_first_response))
        full_response_score = max(0, 1 - (detection_to_full_response / target_full_response))
        
        overall_speed_score = (first_response_score + full_response_score) / 2
        
        score = EffectivenessScore(
            metric=EffectivenessMetric.RESPONSE_SPEED,
            score=overall_speed_score,
            details={
                "detection_to_first_response_seconds": detection_to_first_response,
                "detection_to_full_response_seconds": detection_to_full_response,
                "first_response_score": first_response_score,
                "full_response_score": full_response_score,
                "target_first_response": target_first_response,
                "target_full_response": target_full_response
            },
            measurement_time=datetime.now(),
            confidence_level=0.95
        )
        
        self.active_tests[test_id].effectiveness_scores.append(score)
        return score
    
    async def measure_decision_quality(
        self,
        test_id: str,
        decisions_made: List[Dict[str, Any]],
        decision_outcomes: List[Dict[str, Any]]
    ) -> EffectivenessScore:
        """Measure quality of decisions made during crisis"""
        if test_id not in self.active_tests:
            raise ValueError(f"Test {test_id} not found")
        
        quality_scores = []
        decision_details = []
        
        for decision, outcome in zip(decisions_made, decision_outcomes):
            # Evaluate decision quality based on multiple factors
            factors = {
                "information_completeness": outcome.get("information_completeness", 0.5),
                "stakeholder_consideration": outcome.get("stakeholder_consideration", 0.5),
                "risk_assessment_accuracy": outcome.get("risk_assessment_accuracy", 0.5),
                "implementation_feasibility": outcome.get("implementation_feasibility", 0.5),
                "outcome_effectiveness": outcome.get("outcome_effectiveness", 0.5)
            }
            
            decision_score = sum(factors.values()) / len(factors)
            quality_scores.append(decision_score)
            
            decision_details.append({
                "decision_id": decision.get("id"),
                "decision_type": decision.get("type"),
                "quality_factors": factors,
                "overall_score": decision_score
            })
        
        overall_quality_score = statistics.mean(quality_scores) if quality_scores else 0.0
        
        score = EffectivenessScore(
            metric=EffectivenessMetric.DECISION_QUALITY,
            score=overall_quality_score,
            details={
                "decisions_evaluated": len(decisions_made),
                "individual_scores": quality_scores,
                "decision_details": decision_details,
                "average_score": overall_quality_score
            },
            measurement_time=datetime.now(),
            confidence_level=0.9
        )
        
        self.active_tests[test_id].effectiveness_scores.append(score)
        return score
    
    async def measure_communication_effectiveness(
        self,
        test_id: str,
        communications_sent: List[Dict[str, Any]],
        stakeholder_feedback: List[Dict[str, Any]]
    ) -> EffectivenessScore:
        """Measure effectiveness of crisis communications"""
        if test_id not in self.active_tests:
            raise ValueError(f"Test {test_id} not found")
        
        communication_scores = []
        communication_details = []
        
        for comm in communications_sent:
            # Find corresponding feedback
            feedback = next(
                (f for f in stakeholder_feedback if f.get("communication_id") == comm.get("id")),
                {}
            )
            
            # Evaluate communication effectiveness
            clarity_score = feedback.get("clarity_rating", 0.5)
            timeliness_score = feedback.get("timeliness_rating", 0.5)
            completeness_score = feedback.get("completeness_rating", 0.5)
            appropriateness_score = feedback.get("appropriateness_rating", 0.5)
            
            comm_score = (clarity_score + timeliness_score + completeness_score + appropriateness_score) / 4
            communication_scores.append(comm_score)
            
            communication_details.append({
                "communication_id": comm.get("id"),
                "channel": comm.get("channel"),
                "audience": comm.get("audience"),
                "clarity_score": clarity_score,
                "timeliness_score": timeliness_score,
                "completeness_score": completeness_score,
                "appropriateness_score": appropriateness_score,
                "overall_score": comm_score
            })
        
        overall_comm_score = statistics.mean(communication_scores) if communication_scores else 0.0
        
        score = EffectivenessScore(
            metric=EffectivenessMetric.COMMUNICATION_CLARITY,
            score=overall_comm_score,
            details={
                "communications_evaluated": len(communications_sent),
                "individual_scores": communication_scores,
                "communication_details": communication_details,
                "average_score": overall_comm_score
            },
            measurement_time=datetime.now(),
            confidence_level=0.85
        )
        
        self.active_tests[test_id].effectiveness_scores.append(score)
        return score
    
    async def measure_outcome_success(
        self,
        test_id: str,
        crisis_objectives: List[str],
        achieved_outcomes: List[Dict[str, Any]]
    ) -> EffectivenessScore:
        """Measure success of crisis outcomes against objectives"""
        if test_id not in self.active_tests:
            raise ValueError(f"Test {test_id} not found")
        
        objective_scores = []
        outcome_details = []
        
        for i, objective in enumerate(crisis_objectives):
            outcome = achieved_outcomes[i] if i < len(achieved_outcomes) else {}
            
            # Evaluate outcome success
            completion_rate = outcome.get("completion_rate", 0.0)
            quality_rating = outcome.get("quality_rating", 0.5)
            stakeholder_satisfaction = outcome.get("stakeholder_satisfaction", 0.5)
            long_term_impact = outcome.get("long_term_impact_score", 0.5)
            
            objective_score = (completion_rate + quality_rating + stakeholder_satisfaction + long_term_impact) / 4
            objective_scores.append(objective_score)
            
            outcome_details.append({
                "objective": objective,
                "completion_rate": completion_rate,
                "quality_rating": quality_rating,
                "stakeholder_satisfaction": stakeholder_satisfaction,
                "long_term_impact": long_term_impact,
                "objective_score": objective_score
            })
        
        overall_outcome_score = statistics.mean(objective_scores) if objective_scores else 0.0
        
        score = EffectivenessScore(
            metric=EffectivenessMetric.OUTCOME_SUCCESS,
            score=overall_outcome_score,
            details={
                "objectives_evaluated": len(crisis_objectives),
                "individual_scores": objective_scores,
                "outcome_details": outcome_details,
                "average_score": overall_outcome_score
            },
            measurement_time=datetime.now(),
            confidence_level=0.9
        )
        
        self.active_tests[test_id].effectiveness_scores.append(score)
        return score
    
    async def measure_leadership_effectiveness(
        self,
        test_id: str,
        leadership_actions: List[Dict[str, Any]],
        team_feedback: List[Dict[str, Any]],
        stakeholder_confidence: Dict[str, float]
    ) -> EffectivenessScore:
        """Measure effectiveness of crisis leadership"""
        if test_id not in self.active_tests:
            raise ValueError(f"Test {test_id} not found")
        
        leadership_dimensions = {
            "decision_making": 0.0,
            "communication": 0.0,
            "team_coordination": 0.0,
            "stakeholder_management": 0.0,
            "crisis_composure": 0.0
        }
        
        # Evaluate leadership actions
        for action in leadership_actions:
            action_type = action.get("type", "")
            effectiveness = action.get("effectiveness_rating", 0.5)
            
            if "decision" in action_type.lower():
                leadership_dimensions["decision_making"] += effectiveness
            elif "communication" in action_type.lower():
                leadership_dimensions["communication"] += effectiveness
            elif "coordination" in action_type.lower():
                leadership_dimensions["team_coordination"] += effectiveness
            elif "stakeholder" in action_type.lower():
                leadership_dimensions["stakeholder_management"] += effectiveness
            else:
                leadership_dimensions["crisis_composure"] += effectiveness
        
        # Normalize scores
        action_counts = {dim: max(1, sum(1 for a in leadership_actions if dim.replace("_", "") in a.get("type", "").lower())) for dim in leadership_dimensions}
        for dim in leadership_dimensions:
            leadership_dimensions[dim] /= action_counts[dim]
        
        # Incorporate team feedback
        team_ratings = {
            "leadership_clarity": statistics.mean([f.get("leadership_clarity", 0.5) for f in team_feedback]) if team_feedback else 0.5,
            "decision_confidence": statistics.mean([f.get("decision_confidence", 0.5) for f in team_feedback]) if team_feedback else 0.5,
            "communication_effectiveness": statistics.mean([f.get("communication_effectiveness", 0.5) for f in team_feedback]) if team_feedback else 0.5
        }
        
        # Calculate overall leadership score
        leadership_score = (
            sum(leadership_dimensions.values()) / len(leadership_dimensions) * 0.6 +
            sum(team_ratings.values()) / len(team_ratings) * 0.3 +
            statistics.mean(stakeholder_confidence.values()) * 0.1
        ) if stakeholder_confidence else (
            sum(leadership_dimensions.values()) / len(leadership_dimensions) * 0.7 +
            sum(team_ratings.values()) / len(team_ratings) * 0.3
        )
        
        score = EffectivenessScore(
            metric=EffectivenessMetric.LEADERSHIP_EFFECTIVENESS,
            score=leadership_score,
            details={
                "leadership_dimensions": leadership_dimensions,
                "team_ratings": team_ratings,
                "stakeholder_confidence": stakeholder_confidence,
                "actions_evaluated": len(leadership_actions),
                "team_feedback_count": len(team_feedback),
                "overall_score": leadership_score
            },
            measurement_time=datetime.now(),
            confidence_level=0.85
        )
        
        self.active_tests[test_id].effectiveness_scores.append(score)
        return score
    
    async def complete_effectiveness_test(self, test_id: str) -> CrisisResponseTest:
        """Complete an effectiveness test and calculate overall results"""
        if test_id not in self.active_tests:
            raise ValueError(f"Test {test_id} not found")
        
        test = self.active_tests[test_id]
        test.end_time = datetime.now()
        
        # Calculate overall effectiveness score
        if test.effectiveness_scores:
            # Weight different metrics based on importance
            metric_weights = {
                EffectivenessMetric.RESPONSE_SPEED: 0.2,
                EffectivenessMetric.DECISION_QUALITY: 0.25,
                EffectivenessMetric.COMMUNICATION_CLARITY: 0.15,
                EffectivenessMetric.RESOURCE_UTILIZATION: 0.1,
                EffectivenessMetric.STAKEHOLDER_SATISFACTION: 0.1,
                EffectivenessMetric.OUTCOME_SUCCESS: 0.15,
                EffectivenessMetric.LEADERSHIP_EFFECTIVENESS: 0.05
            }
            
            weighted_scores = []
            for score in test.effectiveness_scores:
                weight = metric_weights.get(score.metric, 0.1)
                weighted_scores.append(score.score * weight)
            
            test.overall_score = sum(weighted_scores) / sum(metric_weights.values())
        else:
            test.overall_score = 0.0
        
        # Generate recommendations
        test.recommendations = self._generate_recommendations(test)
        
        # Move to history
        self.test_history.append(test)
        del self.active_tests[test_id]
        
        return test
    
    def _generate_recommendations(self, test: CrisisResponseTest) -> List[str]:
        """Generate improvement recommendations based on test results"""
        recommendations = []
        
        for score in test.effectiveness_scores:
            thresholds = self.performance_thresholds.get(score.metric, {})
            
            if score.score < thresholds.get("acceptable", 0.5):
                if score.metric == EffectivenessMetric.RESPONSE_SPEED:
                    recommendations.append("Improve crisis detection and early warning systems")
                    recommendations.append("Streamline crisis response protocols")
                elif score.metric == EffectivenessMetric.DECISION_QUALITY:
                    recommendations.append("Enhance decision-making frameworks and information synthesis")
                    recommendations.append("Improve risk assessment capabilities")
                elif score.metric == EffectivenessMetric.COMMUNICATION_CLARITY:
                    recommendations.append("Develop clearer communication templates and protocols")
                    recommendations.append("Improve stakeholder notification systems")
                elif score.metric == EffectivenessMetric.LEADERSHIP_EFFECTIVENESS:
                    recommendations.append("Enhance crisis leadership training and preparation")
                    recommendations.append("Improve team coordination and morale management")
        
        return list(set(recommendations))  # Remove duplicates
    
    async def get_effectiveness_trends(
        self,
        metric: Optional[EffectivenessMetric] = None,
        time_period: Optional[timedelta] = None
    ) -> Dict[str, Any]:
        """Get effectiveness trends over time"""
        cutoff_time = datetime.now() - (time_period or timedelta(days=30))
        
        relevant_tests = [
            test for test in self.test_history
            if test.end_time and test.end_time >= cutoff_time
        ]
        
        if metric:
            scores = []
            for test in relevant_tests:
                metric_scores = [s for s in test.effectiveness_scores if s.metric == metric]
                if metric_scores:
                    scores.append(statistics.mean([s.score for s in metric_scores]))
            
            return {
                "metric": metric.value,
                "scores": scores,
                "trend": "improving" if len(scores) > 1 and scores[-1] > scores[0] else "stable",
                "average_score": statistics.mean(scores) if scores else 0.0,
                "tests_analyzed": len(relevant_tests)
            }
        else:
            overall_scores = [test.overall_score for test in relevant_tests if test.overall_score is not None]
            
            return {
                "overall_scores": overall_scores,
                "trend": "improving" if len(overall_scores) > 1 and overall_scores[-1] > overall_scores[0] else "stable",
                "average_score": statistics.mean(overall_scores) if overall_scores else 0.0,
                "tests_analyzed": len(relevant_tests)
            }
    
    async def benchmark_against_baseline(self, test_id: str) -> Dict[str, Any]:
        """Compare test results against established baselines"""
        test = None
        
        # First check if it's in history
        for t in self.test_history:
            if t.test_id == test_id:
                test = t
                break
        
        # If not in history, check if it's still active
        if not test:
            test = self.active_tests.get(test_id)
        
        if not test:
            raise ValueError(f"Test {test_id} not found")
        
        comparisons = {}
        
        for score in test.effectiveness_scores:
            baseline = self.baseline_metrics.get(score.metric, 0.5)
            improvement = score.score - baseline
            
            comparisons[score.metric.value] = {
                "current_score": score.score,
                "baseline_score": baseline,
                "improvement": improvement,
                "improvement_percentage": (improvement / baseline * 100) if baseline > 0 else 0,
                "performance_level": self._get_performance_level(score.metric, score.score)
            }
        
        return {
            "test_id": test_id,
            "overall_score": test.overall_score,
            "metric_comparisons": comparisons,
            "test_duration": (test.end_time - test.start_time).total_seconds() if test.end_time else None
        }
    
    def _get_performance_level(self, metric: EffectivenessMetric, score: float) -> str:
        """Determine performance level based on score and thresholds"""
        thresholds = self.performance_thresholds.get(metric, {})
        
        if score >= thresholds.get("excellent", 0.9):
            return "excellent"
        elif score >= thresholds.get("good", 0.7):
            return "good"
        elif score >= thresholds.get("acceptable", 0.5):
            return "acceptable"
        else:
            return "poor"
    
    def update_baseline_metrics(self, new_baselines: Dict[EffectivenessMetric, float]):
        """Update baseline metrics for comparison"""
        self.baseline_metrics.update(new_baselines)
    
    async def export_test_results(self, test_id: str) -> Dict[str, Any]:
        """Export comprehensive test results"""
        test = None
        if test_id in self.active_tests:
            test = self.active_tests[test_id]
        else:
            test = next((t for t in self.test_history if t.test_id == test_id), None)
        
        if not test:
            raise ValueError(f"Test {test_id} not found")
        
        return {
            "test_id": test.test_id,
            "crisis_scenario": test.crisis_scenario,
            "test_type": test.test_type,
            "start_time": test.start_time.isoformat(),
            "end_time": test.end_time.isoformat() if test.end_time else None,
            "duration_seconds": (test.end_time - test.start_time).total_seconds() if test.end_time else None,
            "phases_tested": [phase.value for phase in test.phases_tested],
            "effectiveness_scores": [
                {
                    "metric": score.metric.value,
                    "score": score.score,
                    "details": score.details,
                    "measurement_time": score.measurement_time.isoformat(),
                    "confidence_level": score.confidence_level
                }
                for score in test.effectiveness_scores
            ],
            "overall_score": test.overall_score,
            "recommendations": test.recommendations
        }