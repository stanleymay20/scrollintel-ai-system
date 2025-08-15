"""
Cultural Sustainability Tester

This engine tests and validates the sustainability of cultural transformations,
including long-term stability, reinforcement mechanisms, and drift detection.
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
from scipy import stats
import logging

logger = logging.getLogger(__name__)


@dataclass
class SustainabilityMetrics:
    """Metrics for cultural sustainability"""
    stability_score: float
    trend_consistency: float
    reinforcement_strength: float
    decay_resistance: float
    sustainability_level: str
    risk_factors: List[str]


@dataclass
class StabilityAssessment:
    """Assessment of long-term stability"""
    stability_coefficient: float
    variance_within_acceptable_range: bool
    trend_direction: str
    confidence_level: float
    stability_by_period: Dict[str, float]


@dataclass
class ReinforcementAnalysis:
    """Analysis of reinforcement mechanism effectiveness"""
    overall_effectiveness: float
    strong_mechanisms: List[str]
    weak_mechanisms: List[str]
    consistency_score: float
    mechanism_scores: List[Dict[str, Any]]


@dataclass
class DriftAssessment:
    """Assessment of cultural drift"""
    drift_detected: bool
    drift_severity: str
    affected_dimensions: List[str]
    drift_rate: float
    recommended_interventions: List[str]


@dataclass
class SustainabilityRiskAssessment:
    """Assessment of sustainability risks"""
    overall_risk_level: str
    risk_score: float
    high_risk_factors: List[str]
    mitigation_strategies: List[str]
    monitoring_recommendations: Dict[str, Any]


@dataclass
class ResilienceAssessment:
    """Assessment of cultural resilience"""
    resilience_score: float
    average_recovery_time: float
    recovery_completeness: float
    resilience_level: str
    resilience_factors: List[str]


class CulturalSustainabilityTester:
    """Tests and validates cultural transformation sustainability"""
    
    def __init__(self):
        self.stability_thresholds = {
            "excellent": 0.9,
            "good": 0.8,
            "acceptable": 0.7,
            "concerning": 0.6
        }
        self.drift_thresholds = {
            "minimal": 0.05,
            "moderate": 0.15,
            "significant": 0.25,
            "severe": 0.35
        }
    
    def measure_sustainability(self, sustainability_data: Dict[str, Any]) -> SustainabilityMetrics:
        """
        Measure overall cultural sustainability
        
        Args:
            sustainability_data: Data about cultural measurements over time
        
        Returns:
            SustainabilityMetrics with sustainability assessment
        """
        try:
            measurement_points = sustainability_data.get("measurement_points", [])
            reinforcement_mechanisms = sustainability_data.get("reinforcement_mechanisms", [])
            
            # Calculate stability score
            stability_score = self._calculate_stability_score(measurement_points)
            
            # Calculate trend consistency
            trend_consistency = self._calculate_trend_consistency(measurement_points)
            
            # Calculate reinforcement strength
            reinforcement_strength = self._calculate_reinforcement_strength(reinforcement_mechanisms)
            
            # Calculate decay resistance
            decay_resistance = self._calculate_decay_resistance(measurement_points)
            
            # Determine sustainability level
            overall_score = np.mean([stability_score, trend_consistency, reinforcement_strength, decay_resistance])
            sustainability_level = self._determine_sustainability_level(overall_score)
            
            # Identify risk factors
            risk_factors = self._identify_risk_factors(sustainability_data)
            
            return SustainabilityMetrics(
                stability_score=stability_score,
                trend_consistency=trend_consistency,
                reinforcement_strength=reinforcement_strength,
                decay_resistance=decay_resistance,
                sustainability_level=sustainability_level,
                risk_factors=risk_factors
            )
            
        except Exception as e:
            logger.error(f"Error measuring sustainability: {str(e)}")
            return SustainabilityMetrics(
                stability_score=0.0,
                trend_consistency=0.0,
                reinforcement_strength=0.0,
                decay_resistance=0.0,
                sustainability_level="poor",
                risk_factors=[]
            )
    
    def validate_long_term_stability(self, measurement_points: List[Dict[str, Any]]) -> StabilityAssessment:
        """
        Validate long-term cultural stability
        
        Args:
            measurement_points: Time series of cultural measurements
        
        Returns:
            StabilityAssessment with stability analysis
        """
        try:
            if len(measurement_points) < 3:
                return StabilityAssessment(
                    stability_coefficient=0.5,
                    variance_within_acceptable_range=False,
                    trend_direction="insufficient_data",
                    confidence_level=0.0,
                    stability_by_period={}
                )
            
            # Extract time series data
            dates = [point["date"] for point in measurement_points]
            cultural_health_values = [point.get("cultural_health", 0) for point in measurement_points]
            engagement_values = [point.get("engagement", 0) for point in measurement_points]
            
            # Calculate stability coefficient
            stability_coefficient = self._calculate_stability_coefficient(cultural_health_values, engagement_values)
            
            # Check variance
            cultural_variance = np.var(cultural_health_values)
            engagement_variance = np.var(engagement_values)
            variance_within_acceptable_range = (cultural_variance <= 0.01 and engagement_variance <= 0.01)
            
            # Determine trend direction
            trend_direction = self._determine_trend_direction(cultural_health_values, engagement_values)
            
            # Calculate confidence level
            confidence_level = self._calculate_stability_confidence(measurement_points)
            
            # Calculate stability by different periods
            stability_by_period = self._calculate_stability_by_period(measurement_points)
            
            return StabilityAssessment(
                stability_coefficient=stability_coefficient,
                variance_within_acceptable_range=variance_within_acceptable_range,
                trend_direction=trend_direction,
                confidence_level=confidence_level,
                stability_by_period=stability_by_period
            )
            
        except Exception as e:
            logger.error(f"Error validating long-term stability: {str(e)}")
            return StabilityAssessment(
                stability_coefficient=0.0,
                variance_within_acceptable_range=False,
                trend_direction="error",
                confidence_level=0.0,
                stability_by_period={}
            )
    
    def analyze_reinforcement_effectiveness(self, 
                                          reinforcement_mechanisms: List[Dict[str, Any]]) -> ReinforcementAnalysis:
        """
        Analyze effectiveness of cultural reinforcement mechanisms
        
        Args:
            reinforcement_mechanisms: List of reinforcement mechanisms with metrics
        
        Returns:
            ReinforcementAnalysis with effectiveness assessment
        """
        try:
            mechanism_scores = []
            strong_mechanisms = []
            weak_mechanisms = []
            effectiveness_scores = []
            consistency_scores = []
            
            for mechanism in reinforcement_mechanisms:
                mechanism_type = mechanism.get("type", "unknown")
                strength = mechanism.get("strength", 0)
                consistency = mechanism.get("consistency", 0)
                
                # Calculate effectiveness score
                effectiveness_score = (strength * 0.6) + (consistency * 0.4)
                
                mechanism_score = {
                    "type": mechanism_type,
                    "strength": strength,
                    "consistency": consistency,
                    "effectiveness_score": effectiveness_score
                }
                mechanism_scores.append(mechanism_score)
                effectiveness_scores.append(effectiveness_score)
                consistency_scores.append(consistency)
                
                # Categorize mechanisms
                if effectiveness_score >= 0.8:
                    strong_mechanisms.append(mechanism_type)
                elif effectiveness_score < 0.6:
                    weak_mechanisms.append(mechanism_type)
            
            overall_effectiveness = np.mean(effectiveness_scores) if effectiveness_scores else 0
            consistency_score = np.mean(consistency_scores) if consistency_scores else 0
            
            return ReinforcementAnalysis(
                overall_effectiveness=overall_effectiveness,
                strong_mechanisms=strong_mechanisms,
                weak_mechanisms=weak_mechanisms,
                consistency_score=consistency_score,
                mechanism_scores=mechanism_scores
            )
            
        except Exception as e:
            logger.error(f"Error analyzing reinforcement effectiveness: {str(e)}")
            return ReinforcementAnalysis(
                overall_effectiveness=0.0,
                strong_mechanisms=[],
                weak_mechanisms=[],
                consistency_score=0.0,
                mechanism_scores=[]
            )
    
    def detect_cultural_drift(self, drift_data: Dict[str, Any]) -> DriftAssessment:
        """
        Detect cultural drift and degradation
        
        Args:
            drift_data: Data comparing baseline, current, and historical culture
        
        Returns:
            DriftAssessment with drift analysis
        """
        try:
            baseline_culture = drift_data.get("baseline_culture", {})
            current_culture = drift_data.get("current_culture", {})
            measurement_history = drift_data.get("measurement_history", [])
            
            affected_dimensions = []
            drift_magnitudes = []
            
            # Analyze drift for each cultural dimension
            for dimension, baseline_value in baseline_culture.items():
                current_value = current_culture.get(dimension, 0)
                drift_magnitude = abs(baseline_value - current_value) / baseline_value if baseline_value > 0 else 0
                
                drift_magnitudes.append(drift_magnitude)
                
                # Check if drift is significant
                if drift_magnitude >= self.drift_thresholds["moderate"]:
                    affected_dimensions.append(dimension)
            
            # Calculate overall drift rate
            overall_drift_rate = np.mean(drift_magnitudes) if drift_magnitudes else 0
            
            # Determine drift severity
            drift_severity = self._determine_drift_severity(overall_drift_rate)
            
            # Check if drift is detected
            drift_detected = overall_drift_rate >= self.drift_thresholds["moderate"]
            
            # Generate intervention recommendations
            recommended_interventions = self._generate_drift_interventions(
                affected_dimensions, drift_severity, measurement_history
            )
            
            return DriftAssessment(
                drift_detected=drift_detected,
                drift_severity=drift_severity,
                affected_dimensions=affected_dimensions,
                drift_rate=overall_drift_rate,
                recommended_interventions=recommended_interventions
            )
            
        except Exception as e:
            logger.error(f"Error detecting cultural drift: {str(e)}")
            return DriftAssessment(
                drift_detected=False,
                drift_severity="unknown",
                affected_dimensions=[],
                drift_rate=0.0,
                recommended_interventions=[]
            )
    
    def assess_sustainability_risks(self,
                                  sustainability_data: Dict[str, Any],
                                  risk_factors: Dict[str, Any]) -> SustainabilityRiskAssessment:
        """
        Assess risks to cultural sustainability
        
        Args:
            sustainability_data: Current sustainability data
            risk_factors: Potential risk factors with probabilities and impacts
        
        Returns:
            SustainabilityRiskAssessment with risk analysis
        """
        try:
            risk_scores = []
            high_risk_factors = []
            
            # Calculate risk scores
            for risk_name, risk_info in risk_factors.items():
                probability = risk_info.get("probability", 0)
                impact = risk_info.get("impact", 0)
                risk_score = probability * impact
                
                risk_scores.append(risk_score)
                
                if risk_score >= 0.6:  # High risk threshold
                    high_risk_factors.append(risk_name)
            
            overall_risk_score = np.mean(risk_scores) if risk_scores else 0
            
            # Determine risk level
            if overall_risk_score >= 0.7:
                overall_risk_level = "high"
            elif overall_risk_score >= 0.5:
                overall_risk_level = "medium"
            elif overall_risk_score >= 0.3:
                overall_risk_level = "low"
            else:
                overall_risk_level = "minimal"
            
            # Generate mitigation strategies
            mitigation_strategies = self._generate_mitigation_strategies(high_risk_factors, risk_factors)
            
            # Generate monitoring recommendations
            monitoring_recommendations = self._generate_monitoring_recommendations(
                sustainability_data, high_risk_factors
            )
            
            return SustainabilityRiskAssessment(
                overall_risk_level=overall_risk_level,
                risk_score=overall_risk_score,
                high_risk_factors=high_risk_factors,
                mitigation_strategies=mitigation_strategies,
                monitoring_recommendations=monitoring_recommendations
            )
            
        except Exception as e:
            logger.error(f"Error assessing sustainability risks: {str(e)}")
            return SustainabilityRiskAssessment(
                overall_risk_level="unknown",
                risk_score=0.0,
                high_risk_factors=[],
                mitigation_strategies=[],
                monitoring_recommendations={}
            )
    
    def measure_cultural_resilience(self, resilience_data: Dict[str, Any]) -> ResilienceAssessment:
        """
        Measure cultural resilience to challenges
        
        Args:
            resilience_data: Data about challenges and recovery
        
        Returns:
            ResilienceAssessment with resilience analysis
        """
        try:
            challenge_events = resilience_data.get("challenge_events", [])
            recovery_metrics = resilience_data.get("recovery_metrics", [])
            
            if not challenge_events or not recovery_metrics:
                return ResilienceAssessment(
                    resilience_score=0.5,
                    average_recovery_time=0,
                    recovery_completeness=0,
                    resilience_level="unknown",
                    resilience_factors=[]
                )
            
            # Calculate recovery metrics
            recovery_times = [metric.get("recovery_time", 0) for metric in recovery_metrics]
            recovery_completeness_scores = [metric.get("recovery_completeness", 0) for metric in recovery_metrics]
            
            average_recovery_time = np.mean(recovery_times) if recovery_times else 0
            average_recovery_completeness = np.mean(recovery_completeness_scores) if recovery_completeness_scores else 0
            
            # Calculate resilience score
            # Lower recovery time and higher completeness = higher resilience
            time_score = max(0, 1 - (average_recovery_time / 60))  # Normalize against 60 days
            completeness_score = average_recovery_completeness
            
            resilience_score = (time_score * 0.4) + (completeness_score * 0.6)
            
            # Determine resilience level
            if resilience_score >= 0.9:
                resilience_level = "exceptional"
            elif resilience_score >= 0.8:
                resilience_level = "high"
            elif resilience_score >= 0.7:
                resilience_level = "moderate"
            elif resilience_score >= 0.6:
                resilience_level = "low"
            else:
                resilience_level = "poor"
            
            # Identify resilience factors
            resilience_factors = self._identify_resilience_factors(
                challenge_events, recovery_metrics, resilience_score
            )
            
            return ResilienceAssessment(
                resilience_score=resilience_score,
                average_recovery_time=average_recovery_time,
                recovery_completeness=average_recovery_completeness,
                resilience_level=resilience_level,
                resilience_factors=resilience_factors
            )
            
        except Exception as e:
            logger.error(f"Error measuring cultural resilience: {str(e)}")
            return ResilienceAssessment(
                resilience_score=0.0,
                average_recovery_time=0,
                recovery_completeness=0,
                resilience_level="error",
                resilience_factors=[]
            )
    
    # Helper methods
    def _calculate_stability_score(self, measurement_points: List[Dict[str, Any]]) -> float:
        """Calculate stability score from measurement points"""
        if len(measurement_points) < 2:
            return 0.5
        
        cultural_health_values = [point.get("cultural_health", 0) for point in measurement_points]
        engagement_values = [point.get("engagement", 0) for point in measurement_points]
        
        # Calculate coefficient of variation (lower = more stable)
        cultural_cv = np.std(cultural_health_values) / np.mean(cultural_health_values) if np.mean(cultural_health_values) > 0 else 1
        engagement_cv = np.std(engagement_values) / np.mean(engagement_values) if np.mean(engagement_values) > 0 else 1
        
        # Convert to stability score (higher = more stable)
        stability_score = 1 - min(1, (cultural_cv + engagement_cv) / 2)
        return max(0, stability_score)
    
    def _calculate_trend_consistency(self, measurement_points: List[Dict[str, Any]]) -> float:
        """Calculate trend consistency"""
        if len(measurement_points) < 3:
            return 0.5
        
        cultural_health_values = [point.get("cultural_health", 0) for point in measurement_points]
        
        # Calculate differences between consecutive measurements
        differences = [cultural_health_values[i+1] - cultural_health_values[i] 
                      for i in range(len(cultural_health_values)-1)]
        
        # Consistency is high when differences are small and consistent in direction
        if len(differences) == 0:
            return 0.5
        
        avg_difference = np.mean(differences)
        std_difference = np.std(differences)
        
        # Normalize consistency score
        consistency = 1 - min(1, std_difference / (abs(avg_difference) + 0.1))
        return max(0, consistency)
    
    def _calculate_reinforcement_strength(self, reinforcement_mechanisms: List[Dict[str, Any]]) -> float:
        """Calculate overall reinforcement strength"""
        if not reinforcement_mechanisms:
            return 0.3  # Default low strength if no mechanisms
        
        strengths = [mechanism.get("strength", 0) for mechanism in reinforcement_mechanisms]
        consistencies = [mechanism.get("consistency", 0) for mechanism in reinforcement_mechanisms]
        
        # Weight strength and consistency
        overall_strength = np.mean([(s * 0.6) + (c * 0.4) for s, c in zip(strengths, consistencies)])
        return overall_strength
    
    def _calculate_decay_resistance(self, measurement_points: List[Dict[str, Any]]) -> float:
        """Calculate resistance to cultural decay"""
        if len(measurement_points) < 3:
            return 0.5
        
        cultural_health_values = [point.get("cultural_health", 0) for point in measurement_points]
        
        # Check if there's a declining trend
        recent_values = cultural_health_values[-3:]  # Last 3 measurements
        if len(recent_values) < 3:
            return 0.5
        
        # Calculate trend slope
        x = np.arange(len(recent_values))
        slope, _, r_value, _, _ = stats.linregress(x, recent_values)
        
        # Decay resistance is higher when slope is positive or stable
        if slope >= 0:
            decay_resistance = 0.8 + (slope * 0.2)  # Bonus for positive trend
        else:
            decay_resistance = 0.8 + slope  # Penalty for negative trend
        
        return max(0, min(1, decay_resistance))
    
    def _determine_sustainability_level(self, overall_score: float) -> str:
        """Determine sustainability level from overall score"""
        for level, threshold in sorted(self.stability_thresholds.items(), 
                                     key=lambda x: x[1], reverse=True):
            if overall_score >= threshold:
                return level
        return "poor"
    
    def _identify_risk_factors(self, sustainability_data: Dict[str, Any]) -> List[str]:
        """Identify sustainability risk factors"""
        risk_factors = []
        
        measurement_points = sustainability_data.get("measurement_points", [])
        if len(measurement_points) > 0:
            latest_measurement = measurement_points[-1]
            
            if latest_measurement.get("cultural_health", 0) < 0.7:
                risk_factors.append("declining_cultural_health")
            
            if latest_measurement.get("engagement", 0) < 0.75:
                risk_factors.append("low_employee_engagement")
        
        reinforcement_mechanisms = sustainability_data.get("reinforcement_mechanisms", [])
        weak_mechanisms = [m for m in reinforcement_mechanisms if m.get("strength", 0) < 0.6]
        if len(weak_mechanisms) > len(reinforcement_mechanisms) * 0.5:
            risk_factors.append("weak_reinforcement_mechanisms")
        
        return risk_factors
    
    def _calculate_stability_coefficient(self, cultural_values: List[float], engagement_values: List[float]) -> float:
        """Calculate stability coefficient"""
        if len(cultural_values) < 2 or len(engagement_values) < 2:
            return 0.5
        
        cultural_stability = 1 - (np.std(cultural_values) / np.mean(cultural_values)) if np.mean(cultural_values) > 0 else 0
        engagement_stability = 1 - (np.std(engagement_values) / np.mean(engagement_values)) if np.mean(engagement_values) > 0 else 0
        
        return (cultural_stability + engagement_stability) / 2
    
    def _determine_trend_direction(self, cultural_values: List[float], engagement_values: List[float]) -> str:
        """Determine overall trend direction"""
        if len(cultural_values) < 2:
            return "insufficient_data"
        
        cultural_trend = cultural_values[-1] - cultural_values[0]
        engagement_trend = engagement_values[-1] - engagement_values[0]
        
        if cultural_trend > 0.05 and engagement_trend > 0.05:
            return "improving"
        elif cultural_trend < -0.05 or engagement_trend < -0.05:
            return "declining"
        else:
            return "stable"
    
    def _calculate_stability_confidence(self, measurement_points: List[Dict[str, Any]]) -> float:
        """Calculate confidence in stability assessment"""
        # Confidence increases with more data points and consistent measurements
        data_points_factor = min(1.0, len(measurement_points) / 6)  # Optimal at 6+ points
        
        # Calculate measurement consistency
        if len(measurement_points) < 2:
            consistency_factor = 0.5
        else:
            cultural_values = [point.get("cultural_health", 0) for point in measurement_points]
            consistency_factor = 1 - min(1, np.std(cultural_values))
        
        return (data_points_factor * 0.6) + (consistency_factor * 0.4)
    
    def _calculate_stability_by_period(self, measurement_points: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate stability for different time periods"""
        stability_by_period = {}
        
        if len(measurement_points) >= 2:
            # Last 30 days (assuming recent measurements)
            recent_points = measurement_points[-2:]
            stability_by_period["30_days"] = self._calculate_stability_score(recent_points)
        
        if len(measurement_points) >= 3:
            # Last 90 days
            medium_points = measurement_points[-3:]
            stability_by_period["90_days"] = self._calculate_stability_score(medium_points)
        
        if len(measurement_points) >= 4:
            # Last 180 days
            long_points = measurement_points[-4:]
            stability_by_period["180_days"] = self._calculate_stability_score(long_points)
        
        return stability_by_period
    
    def _determine_drift_severity(self, drift_rate: float) -> str:
        """Determine drift severity level"""
        for severity, threshold in sorted(self.drift_thresholds.items(), 
                                        key=lambda x: x[1], reverse=True):
            if drift_rate >= threshold:
                return severity
        return "minimal"
    
    def _generate_drift_interventions(self, 
                                    affected_dimensions: List[str],
                                    drift_severity: str,
                                    measurement_history: List[Dict[str, Any]]) -> List[str]:
        """Generate intervention recommendations for cultural drift"""
        interventions = []
        
        if "innovation" in affected_dimensions:
            interventions.append("Reinforce innovation practices and recognition")
        
        if "collaboration" in affected_dimensions:
            interventions.append("Strengthen team collaboration mechanisms")
        
        if "accountability" in affected_dimensions:
            interventions.append("Clarify accountability structures and expectations")
        
        if drift_severity in ["significant", "severe"]:
            interventions.append("Conduct comprehensive culture refresh initiative")
            interventions.append("Increase leadership visibility and modeling")
        
        return interventions
    
    def _generate_mitigation_strategies(self, 
                                      high_risk_factors: List[str],
                                      risk_factors: Dict[str, Any]) -> List[str]:
        """Generate risk mitigation strategies"""
        strategies = []
        
        for risk_factor in high_risk_factors:
            if risk_factor == "leadership_changes":
                strategies.append("Develop leadership succession and continuity plans")
            elif risk_factor == "organizational_restructuring":
                strategies.append("Create culture preservation protocols during restructuring")
            elif risk_factor == "market_pressures":
                strategies.append("Build cultural resilience and adaptability capabilities")
            elif risk_factor == "resource_constraints":
                strategies.append("Prioritize high-impact, low-cost cultural reinforcement activities")
        
        return strategies
    
    def _generate_monitoring_recommendations(self,
                                           sustainability_data: Dict[str, Any],
                                           high_risk_factors: List[str]) -> Dict[str, Any]:
        """Generate monitoring recommendations"""
        recommendations = {
            "frequency": "monthly" if high_risk_factors else "quarterly",
            "key_metrics": ["cultural_health", "employee_engagement", "behavioral_indicators"],
            "early_warning_indicators": [],
            "escalation_triggers": []
        }
        
        if "leadership_changes" in high_risk_factors:
            recommendations["early_warning_indicators"].append("leadership_alignment_score")
        
        if "resource_constraints" in high_risk_factors:
            recommendations["early_warning_indicators"].append("reinforcement_mechanism_strength")
        
        recommendations["escalation_triggers"] = [
            "cultural_health_drop_below_0.7",
            "engagement_drop_below_0.75",
            "drift_rate_exceeds_0.15"
        ]
        
        return recommendations
    
    def _identify_resilience_factors(self,
                                   challenge_events: List[Dict[str, Any]],
                                   recovery_metrics: List[Dict[str, Any]],
                                   resilience_score: float) -> List[str]:
        """Identify factors contributing to cultural resilience"""
        factors = []
        
        if resilience_score >= 0.8:
            factors.append("strong_cultural_foundation")
            factors.append("effective_leadership_response")
        
        # Analyze recovery patterns
        quick_recoveries = [m for m in recovery_metrics if m.get("recovery_time", 100) <= 30]
        if len(quick_recoveries) >= len(recovery_metrics) * 0.7:
            factors.append("rapid_recovery_capability")
        
        complete_recoveries = [m for m in recovery_metrics if m.get("recovery_completeness", 0) >= 0.9]
        if len(complete_recoveries) >= len(recovery_metrics) * 0.8:
            factors.append("comprehensive_recovery_ability")
        
        return factors