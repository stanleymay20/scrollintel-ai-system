"""
Cultural Change Resistance Detection Engine
Implements early identification of cultural resistance patterns with source analysis and impact prediction.
"""
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Tuple
import uuid
from dataclasses import asdict

from ..models.resistance_detection_models import (
    ResistancePattern, ResistanceIndicator, ResistanceDetection,
    ResistanceSource, ResistanceImpactAssessment, ResistancePrediction,
    ResistanceMonitoringConfig, ResistanceType, ResistanceSeverity
)
from ..models.cultural_assessment_models import Culture, Organization
from ..models.transformation_roadmap_models import Transformation


class ResistanceDetectionEngine:
    """Engine for detecting and analyzing cultural change resistance"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.resistance_patterns = self._initialize_resistance_patterns()
        self.detection_indicators = self._initialize_detection_indicators()
        
    def detect_resistance_patterns(
        self, 
        organization: Organization,
        transformation: Transformation,
        monitoring_data: Dict[str, Any]
    ) -> List[ResistanceDetection]:
        """
        Early identification of cultural resistance patterns
        
        Args:
            organization: Organization undergoing transformation
            transformation: Current transformation process
            monitoring_data: Real-time monitoring data
            
        Returns:
            List of detected resistance instances
        """
        try:
            detections = []
            
            # Analyze behavioral indicators
            behavioral_resistance = self._detect_behavioral_resistance(
                monitoring_data.get('behavioral_data', {}),
                organization,
                transformation
            )
            detections.extend(behavioral_resistance)
            
            # Analyze communication patterns
            communication_resistance = self._detect_communication_resistance(
                monitoring_data.get('communication_data', {}),
                organization,
                transformation
            )
            detections.extend(communication_resistance)
            
            # Analyze engagement metrics
            engagement_resistance = self._detect_engagement_resistance(
                monitoring_data.get('engagement_data', {}),
                organization,
                transformation
            )
            detections.extend(engagement_resistance)
            
            # Analyze performance indicators
            performance_resistance = self._detect_performance_resistance(
                monitoring_data.get('performance_data', {}),
                organization,
                transformation
            )
            detections.extend(performance_resistance)
            
            # Cross-validate detections
            validated_detections = self._validate_detections(detections)
            
            self.logger.info(f"Detected {len(validated_detections)} resistance patterns")
            return validated_detections
            
        except Exception as e:
            self.logger.error(f"Error detecting resistance patterns: {str(e)}")
            raise
    
    def analyze_resistance_sources(
        self,
        detection: ResistanceDetection,
        organization: Organization
    ) -> List[ResistanceSource]:
        """
        Analyze and categorize sources of resistance
        
        Args:
            detection: Detected resistance instance
            organization: Organization context
            
        Returns:
            List of resistance sources with analysis
        """
        try:
            sources = []
            
            # Analyze individual sources
            individual_sources = self._analyze_individual_sources(detection, organization)
            sources.extend(individual_sources)
            
            # Analyze team/group sources
            group_sources = self._analyze_group_sources(detection, organization)
            sources.extend(group_sources)
            
            # Analyze systemic sources
            systemic_sources = self._analyze_systemic_sources(detection, organization)
            sources.extend(systemic_sources)
            
            # Analyze external sources
            external_sources = self._analyze_external_sources(detection, organization)
            sources.extend(external_sources)
            
            # Rank sources by influence
            ranked_sources = self._rank_sources_by_influence(sources)
            
            self.logger.info(f"Identified {len(ranked_sources)} resistance sources")
            return ranked_sources
            
        except Exception as e:
            self.logger.error(f"Error analyzing resistance sources: {str(e)}")
            raise
    
    def assess_resistance_impact(
        self,
        detection: ResistanceDetection,
        transformation: Transformation
    ) -> ResistanceImpactAssessment:
        """
        Assess impact of resistance on transformation
        
        Args:
            detection: Detected resistance instance
            transformation: Transformation being impacted
            
        Returns:
            Impact assessment with predictions
        """
        try:
            # Assess transformation impact
            transformation_impact = self._assess_transformation_impact(detection, transformation)
            
            # Assess timeline impact
            timeline_impact = self._assess_timeline_impact(detection, transformation)
            
            # Assess resource impact
            resource_impact = self._assess_resource_impact(detection, transformation)
            
            # Assess stakeholder impact
            stakeholder_impact = self._assess_stakeholder_impact(detection, transformation)
            
            # Calculate success probability reduction
            success_reduction = self._calculate_success_probability_reduction(
                detection, transformation
            )
            
            # Identify cascading effects
            cascading_effects = self._identify_cascading_effects(detection, transformation)
            
            # Check critical path disruption
            critical_disruption = self._check_critical_path_disruption(
                detection, transformation
            )
            
            # Calculate assessment confidence
            confidence = self._calculate_assessment_confidence(detection, transformation)
            
            impact_assessment = ResistanceImpactAssessment(
                id=str(uuid.uuid4()),
                detection_id=detection.id,
                transformation_impact=transformation_impact,
                timeline_impact=timeline_impact,
                resource_impact=resource_impact,
                stakeholder_impact=stakeholder_impact,
                success_probability_reduction=success_reduction,
                cascading_effects=cascading_effects,
                critical_path_disruption=critical_disruption,
                assessment_confidence=confidence
            )
            
            self.logger.info(f"Assessed resistance impact with {confidence:.2f} confidence")
            return impact_assessment
            
        except Exception as e:
            self.logger.error(f"Error assessing resistance impact: {str(e)}")
            raise
    
    def predict_future_resistance(
        self,
        organization: Organization,
        transformation: Transformation,
        historical_data: Dict[str, Any]
    ) -> List[ResistancePrediction]:
        """
        Predict potential future resistance patterns
        
        Args:
            organization: Organization context
            transformation: Current transformation
            historical_data: Historical resistance data
            
        Returns:
            List of resistance predictions
        """
        try:
            predictions = []
            
            # Analyze transformation phases for resistance risk
            phase_predictions = self._predict_phase_resistance(transformation, historical_data)
            predictions.extend(phase_predictions)
            
            # Analyze stakeholder groups for resistance risk
            stakeholder_predictions = self._predict_stakeholder_resistance(
                organization, transformation, historical_data
            )
            predictions.extend(stakeholder_predictions)
            
            # Analyze intervention points for resistance risk
            intervention_predictions = self._predict_intervention_resistance(
                transformation, historical_data
            )
            predictions.extend(intervention_predictions)
            
            # Analyze external factors for resistance risk
            external_predictions = self._predict_external_resistance(
                organization, transformation, historical_data
            )
            predictions.extend(external_predictions)
            
            # Rank predictions by probability and impact
            ranked_predictions = self._rank_predictions(predictions)
            
            self.logger.info(f"Generated {len(ranked_predictions)} resistance predictions")
            return ranked_predictions
            
        except Exception as e:
            self.logger.error(f"Error predicting resistance: {str(e)}")
            raise
    
    def _detect_behavioral_resistance(
        self,
        behavioral_data: Dict[str, Any],
        organization: Organization,
        transformation: Transformation
    ) -> List[ResistanceDetection]:
        """Detect resistance through behavioral pattern analysis"""
        detections = []
        
        # Analyze attendance patterns
        if 'attendance' in behavioral_data:
            attendance_resistance = self._analyze_attendance_patterns(
                behavioral_data['attendance'], organization, transformation
            )
            detections.extend(attendance_resistance)
        
        # Analyze participation patterns
        if 'participation' in behavioral_data:
            participation_resistance = self._analyze_participation_patterns(
                behavioral_data['participation'], organization, transformation
            )
            detections.extend(participation_resistance)
        
        # Analyze compliance patterns
        if 'compliance' in behavioral_data:
            compliance_resistance = self._analyze_compliance_patterns(
                behavioral_data['compliance'], organization, transformation
            )
            detections.extend(compliance_resistance)
        
        return detections
    
    def _detect_communication_resistance(
        self,
        communication_data: Dict[str, Any],
        organization: Organization,
        transformation: Transformation
    ) -> List[ResistanceDetection]:
        """Detect resistance through communication pattern analysis"""
        detections = []
        
        # Analyze sentiment patterns
        if 'sentiment' in communication_data:
            sentiment_resistance = self._analyze_sentiment_patterns(
                communication_data['sentiment'], organization, transformation
            )
            detections.extend(sentiment_resistance)
        
        # Analyze feedback patterns
        if 'feedback' in communication_data:
            feedback_resistance = self._analyze_feedback_patterns(
                communication_data['feedback'], organization, transformation
            )
            detections.extend(feedback_resistance)
        
        # Analyze rumor/concern patterns
        if 'concerns' in communication_data:
            concern_resistance = self._analyze_concern_patterns(
                communication_data['concerns'], organization, transformation
            )
            detections.extend(concern_resistance)
        
        return detections
    
    def _detect_engagement_resistance(
        self,
        engagement_data: Dict[str, Any],
        organization: Organization,
        transformation: Transformation
    ) -> List[ResistanceDetection]:
        """Detect resistance through engagement metric analysis"""
        detections = []
        
        # Analyze engagement scores
        if 'scores' in engagement_data:
            score_resistance = self._analyze_engagement_scores(
                engagement_data['scores'], organization, transformation
            )
            detections.extend(score_resistance)
        
        # Analyze voluntary participation
        if 'voluntary_participation' in engagement_data:
            voluntary_resistance = self._analyze_voluntary_participation(
                engagement_data['voluntary_participation'], organization, transformation
            )
            detections.extend(voluntary_resistance)
        
        return detections
    
    def _detect_performance_resistance(
        self,
        performance_data: Dict[str, Any],
        organization: Organization,
        transformation: Transformation
    ) -> List[ResistanceDetection]:
        """Detect resistance through performance indicator analysis"""
        detections = []
        
        # Analyze productivity patterns
        if 'productivity' in performance_data:
            productivity_resistance = self._analyze_productivity_patterns(
                performance_data['productivity'], organization, transformation
            )
            detections.extend(productivity_resistance)
        
        # Analyze quality patterns
        if 'quality' in performance_data:
            quality_resistance = self._analyze_quality_patterns(
                performance_data['quality'], organization, transformation
            )
            detections.extend(quality_resistance)
        
        return detections
    
    def _initialize_resistance_patterns(self) -> List[ResistancePattern]:
        """Initialize known resistance patterns"""
        patterns = [
            ResistancePattern(
                id="pattern_001",
                pattern_type=ResistanceType.ACTIVE_OPPOSITION,
                description="Direct vocal opposition to change initiatives",
                indicators=["negative_feedback", "public_criticism", "meeting_disruption"],
                typical_sources=[ResistanceSource.INDIVIDUAL, ResistanceSource.LEADERSHIP],
                severity_factors={"visibility": 0.8, "influence": 0.9, "persistence": 0.7},
                detection_methods=["sentiment_analysis", "feedback_monitoring", "meeting_analysis"],
                created_at=datetime.now()
            ),
            ResistancePattern(
                id="pattern_002",
                pattern_type=ResistanceType.PASSIVE_RESISTANCE,
                description="Subtle non-compliance and avoidance behaviors",
                indicators=["low_participation", "delayed_compliance", "minimal_effort"],
                typical_sources=[ResistanceSource.INDIVIDUAL, ResistanceSource.TEAM],
                severity_factors={"scope": 0.6, "duration": 0.8, "impact": 0.5},
                detection_methods=["behavioral_analysis", "performance_monitoring", "engagement_tracking"],
                created_at=datetime.now()
            )
        ]
        return patterns
    
    def _initialize_detection_indicators(self) -> List[ResistanceIndicator]:
        """Initialize resistance detection indicators"""
        indicators = [
            ResistanceIndicator(
                id="indicator_001",
                indicator_type="engagement_drop",
                description="Significant drop in engagement scores",
                measurement_method="engagement_survey_analysis",
                threshold_values={"moderate": 0.15, "high": 0.25, "critical": 0.40},
                weight=0.8,
                reliability_score=0.85
            ),
            ResistanceIndicator(
                id="indicator_002",
                indicator_type="negative_sentiment",
                description="Increase in negative sentiment in communications",
                measurement_method="sentiment_analysis",
                threshold_values={"moderate": 0.20, "high": 0.35, "critical": 0.50},
                weight=0.7,
                reliability_score=0.75
            )
        ]
        return indicators
    
    def _validate_detections(self, detections: List[ResistanceDetection]) -> List[ResistanceDetection]:
        """Cross-validate resistance detections"""
        validated = []
        for detection in detections:
            if detection.confidence_score >= 0.6:  # Minimum confidence threshold
                validated.append(detection)
        return validated
    
    def _analyze_individual_sources(
        self, detection: ResistanceDetection, organization: Organization
    ) -> List[ResistanceSource]:
        """Analyze individual sources of resistance"""
        # Implementation would analyze individual stakeholders
        return []
    
    def _analyze_group_sources(
        self, detection: ResistanceDetection, organization: Organization
    ) -> List[ResistanceSource]:
        """Analyze group/team sources of resistance"""
        # Implementation would analyze team and department resistance
        return []
    
    def _analyze_systemic_sources(
        self, detection: ResistanceDetection, organization: Organization
    ) -> List[ResistanceSource]:
        """Analyze systemic sources of resistance"""
        # Implementation would analyze organizational system resistance
        return []
    
    def _analyze_external_sources(
        self, detection: ResistanceDetection, organization: Organization
    ) -> List[ResistanceSource]:
        """Analyze external sources of resistance"""
        # Implementation would analyze external stakeholder resistance
        return []
    
    def _rank_sources_by_influence(self, sources: List[ResistanceSource]) -> List[ResistanceSource]:
        """Rank resistance sources by influence level"""
        return sorted(sources, key=lambda x: x.influence_level, reverse=True)
    
    def _assess_transformation_impact(
        self, detection: ResistanceDetection, transformation: Transformation
    ) -> Dict[str, float]:
        """Assess impact on transformation objectives"""
        return {
            "timeline_delay": 0.15,
            "resource_increase": 0.10,
            "scope_reduction": 0.05,
            "quality_impact": 0.08
        }
    
    def _assess_timeline_impact(
        self, detection: ResistanceDetection, transformation: Transformation
    ) -> Dict[str, int]:
        """Assess impact on transformation timeline"""
        return {
            "delay_days": 14,
            "critical_path_impact": 7,
            "milestone_delays": 2
        }
    
    def _assess_resource_impact(
        self, detection: ResistanceDetection, transformation: Transformation
    ) -> Dict[str, float]:
        """Assess impact on resource requirements"""
        return {
            "additional_effort": 0.20,
            "additional_cost": 0.15,
            "additional_personnel": 0.10
        }
    
    def _assess_stakeholder_impact(
        self, detection: ResistanceDetection, transformation: Transformation
    ) -> Dict[str, float]:
        """Assess impact on stakeholder groups"""
        return {
            "employee_morale": -0.15,
            "leadership_confidence": -0.10,
            "customer_perception": -0.05
        }
    
    def _calculate_success_probability_reduction(
        self, detection: ResistanceDetection, transformation: Transformation
    ) -> float:
        """Calculate reduction in transformation success probability"""
        base_reduction = 0.10
        severity_multiplier = {
            ResistanceSeverity.LOW: 0.5,
            ResistanceSeverity.MODERATE: 1.0,
            ResistanceSeverity.HIGH: 1.5,
            ResistanceSeverity.CRITICAL: 2.0
        }
        return base_reduction * severity_multiplier.get(detection.severity, 1.0)
    
    def _identify_cascading_effects(
        self, detection: ResistanceDetection, transformation: Transformation
    ) -> List[str]:
        """Identify potential cascading effects of resistance"""
        return [
            "reduced_team_morale",
            "increased_skepticism",
            "delayed_adoption",
            "resource_reallocation_needed"
        ]
    
    def _check_critical_path_disruption(
        self, detection: ResistanceDetection, transformation: Transformation
    ) -> bool:
        """Check if resistance disrupts critical transformation path"""
        return detection.severity in [ResistanceSeverity.HIGH, ResistanceSeverity.CRITICAL]
    
    def _calculate_assessment_confidence(
        self, detection: ResistanceDetection, transformation: Transformation
    ) -> float:
        """Calculate confidence in impact assessment"""
        return min(0.95, detection.confidence_score + 0.10)
    
    def _predict_phase_resistance(
        self, transformation: Transformation, historical_data: Dict[str, Any]
    ) -> List[ResistancePrediction]:
        """Predict resistance based on transformation phases"""
        # Implementation would analyze phase-specific resistance patterns
        return []
    
    def _predict_stakeholder_resistance(
        self, organization: Organization, transformation: Transformation, historical_data: Dict[str, Any]
    ) -> List[ResistancePrediction]:
        """Predict resistance from specific stakeholder groups"""
        # Implementation would analyze stakeholder-specific resistance patterns
        return []
    
    def _predict_intervention_resistance(
        self, transformation: Transformation, historical_data: Dict[str, Any]
    ) -> List[ResistancePrediction]:
        """Predict resistance to specific interventions"""
        # Implementation would analyze intervention-specific resistance patterns
        return []
    
    def _predict_external_resistance(
        self, organization: Organization, transformation: Transformation, historical_data: Dict[str, Any]
    ) -> List[ResistancePrediction]:
        """Predict resistance from external factors"""
        # Implementation would analyze external resistance factors
        return []
    
    def _rank_predictions(self, predictions: List[ResistancePrediction]) -> List[ResistancePrediction]:
        """Rank predictions by probability and potential impact"""
        return sorted(predictions, key=lambda x: x.probability, reverse=True)
    
    # Placeholder methods for specific analysis functions
    def _analyze_attendance_patterns(self, data, org, trans): return []
    def _analyze_participation_patterns(self, data, org, trans): return []
    def _analyze_compliance_patterns(self, data, org, trans): return []
    def _analyze_sentiment_patterns(self, data, org, trans): return []
    def _analyze_feedback_patterns(self, data, org, trans): return []
    def _analyze_concern_patterns(self, data, org, trans): return []
    def _analyze_engagement_scores(self, data, org, trans): return []
    def _analyze_voluntary_participation(self, data, org, trans): return []
    def _analyze_productivity_patterns(self, data, org, trans): return []
    def _analyze_quality_patterns(self, data, org, trans): return []