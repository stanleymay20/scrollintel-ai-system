"""
Behavioral Analysis Engine for Cultural Transformation Leadership

This engine analyzes organizational behaviors, identifies patterns,
assesses behavioral norms, and evaluates behavior-culture alignment.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from uuid import uuid4

from ..models.behavioral_analysis_models import (
    BehaviorPattern, BehavioralNorm, BehaviorCultureAlignment,
    BehaviorAnalysisResult, BehaviorObservation, BehaviorMetrics,
    BehaviorType, BehaviorFrequency, AlignmentLevel
)

logger = logging.getLogger(__name__)


class BehavioralAnalysisEngine:
    """Engine for comprehensive behavioral analysis"""
    
    def __init__(self):
        self.behavior_patterns = {}
        self.behavioral_norms = {}
        self.observations = {}
        self.analysis_results = {}
        
    def analyze_organizational_behaviors(
        self, 
        organization_id: str,
        observations: List[BehaviorObservation],
        cultural_values: List[str],
        analyst: str = "ScrollIntel"
    ) -> BehaviorAnalysisResult:
        """
        Perform comprehensive behavioral analysis
        
        Requirements: 3.1, 3.2 - Build current behavioral pattern identification and analysis
        """
        try:
            analysis_id = str(uuid4())
            
            # Identify behavior patterns
            behavior_patterns = self._identify_behavior_patterns(observations)
            
            # Assess behavioral norms
            behavioral_norms = self._assess_behavioral_norms(observations, behavior_patterns)
            
            # Analyze behavior-culture alignment
            culture_alignments = self._analyze_culture_alignment(
                behavior_patterns, cultural_values
            )
            
            # Calculate overall health score
            health_score = self._calculate_behavioral_health_score(
                behavior_patterns, behavioral_norms, culture_alignments
            )
            
            # Generate insights and recommendations
            insights = self._generate_behavioral_insights(
                behavior_patterns, behavioral_norms, culture_alignments
            )
            recommendations = self._generate_behavioral_recommendations(
                behavior_patterns, behavioral_norms, culture_alignments
            )
            
            result = BehaviorAnalysisResult(
                organization_id=organization_id,
                analysis_id=analysis_id,
                behavior_patterns=behavior_patterns,
                behavioral_norms=behavioral_norms,
                culture_alignments=culture_alignments,
                overall_health_score=health_score,
                key_insights=insights,
                recommendations=recommendations,
                analysis_date=datetime.now(),
                analyst=analyst
            )
            
            self.analysis_results[analysis_id] = result
            logger.info(f"Completed behavioral analysis for organization {organization_id}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in behavioral analysis: {str(e)}")
            raise
    
    def _identify_behavior_patterns(
        self, 
        observations: List[BehaviorObservation]
    ) -> List[BehaviorPattern]:
        """Identify recurring behavioral patterns from observations"""
        patterns = []
        
        # Group observations by behavior type and context
        behavior_groups = {}
        for obs in observations:
            key = f"{obs.behavior_type.value}_{obs.observed_behavior}"
            if key not in behavior_groups:
                behavior_groups[key] = []
            behavior_groups[key].append(obs)
        
        # Analyze each group for patterns
        for group_key, group_obs in behavior_groups.items():
            if len(group_obs) >= 3:  # Minimum occurrences for pattern
                pattern = self._create_behavior_pattern(group_key, group_obs)
                patterns.append(pattern)
        
        return patterns
    
    def _create_behavior_pattern(
        self, 
        group_key: str, 
        observations: List[BehaviorObservation]
    ) -> BehaviorPattern:
        """Create a behavior pattern from grouped observations"""
        behavior_type = observations[0].behavior_type
        
        # Determine frequency based on observation count
        freq_mapping = {
            (3, 5): BehaviorFrequency.OCCASIONAL,
            (6, 10): BehaviorFrequency.FREQUENT,
            (11, float('inf')): BehaviorFrequency.CONSTANT
        }
        
        frequency = BehaviorFrequency.RARE
        for (min_count, max_count), freq in freq_mapping.items():
            if min_count <= len(observations) <= max_count:
                frequency = freq
                break
        
        # Extract common triggers and outcomes
        triggers = list(set([
            trigger for obs in observations 
            for trigger in obs.context.get('triggers', [])
        ]))
        
        outcomes = list(set([
            outcome for obs in observations 
            for outcome in obs.context.get('outcomes', [])
        ]))
        
        # Get all participants
        participants = list(set([
            participant for obs in observations 
            for participant in obs.participants
        ]))
        
        # Calculate pattern strength based on consistency
        strength = min(1.0, len(observations) / 10.0)
        
        return BehaviorPattern(
            id=str(uuid4()),
            name=f"Pattern: {observations[0].observed_behavior}",
            description=f"Recurring {behavior_type.value} behavior pattern",
            behavior_type=behavior_type,
            frequency=frequency,
            triggers=triggers,
            outcomes=outcomes,
            participants=participants,
            context={'observation_count': len(observations)},
            strength=strength,
            identified_date=datetime.now()
        )
    
    def _assess_behavioral_norms(
        self, 
        observations: List[BehaviorObservation],
        patterns: List[BehaviorPattern]
    ) -> List[BehavioralNorm]:
        """Assess and document behavioral norms"""
        norms = []
        
        # Group patterns by behavior type
        type_patterns = {}
        for pattern in patterns:
            if pattern.behavior_type not in type_patterns:
                type_patterns[pattern.behavior_type] = []
            type_patterns[pattern.behavior_type].append(pattern)
        
        # Create norms for each behavior type
        for behavior_type, type_patterns_list in type_patterns.items():
            norm = self._create_behavioral_norm(behavior_type, type_patterns_list, observations)
            norms.append(norm)
        
        return norms
    
    def _create_behavioral_norm(
        self, 
        behavior_type: BehaviorType,
        patterns: List[BehaviorPattern],
        observations: List[BehaviorObservation]
    ) -> BehavioralNorm:
        """Create a behavioral norm from patterns and observations"""
        
        # Identify expected behaviors from strong patterns
        expected_behaviors = [
            pattern.name for pattern in patterns 
            if pattern.strength > 0.6
        ]
        
        # Identify discouraged behaviors from weak or negative patterns
        discouraged_behaviors = [
            pattern.name for pattern in patterns 
            if pattern.strength < 0.3
        ]
        
        # Basic enforcement mechanisms
        enforcement_mechanisms = [
            "peer_feedback",
            "management_oversight",
            "cultural_reinforcement"
        ]
        
        # Calculate compliance rate based on pattern strengths
        compliance_rate = sum(p.strength for p in patterns) / len(patterns) if patterns else 0.5
        
        # Assign cultural importance based on behavior type
        importance_mapping = {
            BehaviorType.LEADERSHIP: 0.9,
            BehaviorType.COLLABORATION: 0.8,
            BehaviorType.COMMUNICATION: 0.8,
            BehaviorType.INNOVATION: 0.7,
            BehaviorType.PERFORMANCE: 0.7,
            BehaviorType.LEARNING: 0.6,
            BehaviorType.DECISION_MAKING: 0.8,
            BehaviorType.CONFLICT_RESOLUTION: 0.7
        }
        
        cultural_importance = importance_mapping.get(behavior_type, 0.5)
        
        return BehavioralNorm(
            id=str(uuid4()),
            name=f"{behavior_type.value.title()} Norm",
            description=f"Behavioral norm for {behavior_type.value}",
            behavior_type=behavior_type,
            expected_behaviors=expected_behaviors,
            discouraged_behaviors=discouraged_behaviors,
            enforcement_mechanisms=enforcement_mechanisms,
            compliance_rate=compliance_rate,
            cultural_importance=cultural_importance,
            established_date=datetime.now(),
            last_updated=datetime.now()
        )
    
    def _analyze_culture_alignment(
        self, 
        patterns: List[BehaviorPattern],
        cultural_values: List[str]
    ) -> List[BehaviorCultureAlignment]:
        """Analyze alignment between behaviors and cultural values"""
        alignments = []
        
        for pattern in patterns:
            for value in cultural_values:
                alignment = self._assess_pattern_value_alignment(pattern, value)
                alignments.append(alignment)
        
        return alignments
    
    def _assess_pattern_value_alignment(
        self, 
        pattern: BehaviorPattern, 
        cultural_value: str
    ) -> BehaviorCultureAlignment:
        """Assess alignment between a behavior pattern and cultural value"""
        
        # Simple alignment scoring based on keywords and behavior type
        alignment_keywords = {
            'collaboration': ['teamwork', 'cooperation', 'partnership'],
            'innovation': ['creativity', 'experimentation', 'risk-taking'],
            'excellence': ['quality', 'performance', 'achievement'],
            'integrity': ['honesty', 'transparency', 'ethics'],
            'respect': ['diversity', 'inclusion', 'dignity']
        }
        
        # Calculate alignment score
        alignment_score = 0.5  # Default neutral alignment
        supporting_evidence = []
        conflicting_evidence = []
        
        value_lower = cultural_value.lower()
        for keyword_category, keywords in alignment_keywords.items():
            if any(keyword in value_lower for keyword in keywords):
                if pattern.behavior_type.value in keywords or any(
                    keyword in pattern.name.lower() for keyword in keywords
                ):
                    alignment_score = min(1.0, alignment_score + 0.3)
                    supporting_evidence.append(f"Pattern aligns with {keyword_category}")
        
        # Determine alignment level
        if alignment_score >= 0.8:
            alignment_level = AlignmentLevel.PERFECTLY_ALIGNED
        elif alignment_score >= 0.6:
            alignment_level = AlignmentLevel.WELL_ALIGNED
        elif alignment_score >= 0.4:
            alignment_level = AlignmentLevel.PARTIALLY_ALIGNED
        else:
            alignment_level = AlignmentLevel.MISALIGNED
            conflicting_evidence.append("Low alignment with cultural value")
        
        return BehaviorCultureAlignment(
            id=str(uuid4()),
            behavior_pattern_id=pattern.id,
            cultural_value=cultural_value,
            alignment_level=alignment_level,
            alignment_score=alignment_score,
            supporting_evidence=supporting_evidence,
            conflicting_evidence=conflicting_evidence,
            impact_on_culture=f"Pattern {alignment_level.value} with {cultural_value}",
            recommendations=self._generate_alignment_recommendations(alignment_level, pattern, cultural_value),
            analysis_date=datetime.now()
        )
    
    def _generate_alignment_recommendations(
        self, 
        alignment_level: AlignmentLevel,
        pattern: BehaviorPattern,
        cultural_value: str
    ) -> List[str]:
        """Generate recommendations based on alignment level"""
        recommendations = []
        
        if alignment_level == AlignmentLevel.MISALIGNED:
            recommendations.extend([
                f"Address misalignment between {pattern.name} and {cultural_value}",
                "Implement corrective interventions",
                "Provide targeted training and coaching"
            ])
        elif alignment_level == AlignmentLevel.PARTIALLY_ALIGNED:
            recommendations.extend([
                f"Strengthen alignment between {pattern.name} and {cultural_value}",
                "Reinforce positive aspects of the behavior",
                "Address areas of misalignment"
            ])
        elif alignment_level == AlignmentLevel.WELL_ALIGNED:
            recommendations.extend([
                f"Maintain and reinforce {pattern.name}",
                "Use as positive example for others",
                "Monitor for consistency"
            ])
        else:  # PERFECTLY_ALIGNED
            recommendations.extend([
                f"Celebrate and showcase {pattern.name}",
                "Use as cultural exemplar",
                "Scale across organization"
            ])
        
        return recommendations
    
    def _calculate_behavioral_health_score(
        self, 
        patterns: List[BehaviorPattern],
        norms: List[BehavioralNorm],
        alignments: List[BehaviorCultureAlignment]
    ) -> float:
        """Calculate overall behavioral health score"""
        
        if not patterns and not norms and not alignments:
            return 0.5  # Neutral score if no data
        
        # Pattern strength component (40% weight)
        pattern_score = sum(p.strength for p in patterns) / len(patterns) if patterns else 0.5
        
        # Norm compliance component (30% weight)
        norm_score = sum(n.compliance_rate for n in norms) / len(norms) if norms else 0.5
        
        # Culture alignment component (30% weight)
        alignment_score = sum(a.alignment_score for a in alignments) / len(alignments) if alignments else 0.5
        
        health_score = (pattern_score * 0.4) + (norm_score * 0.3) + (alignment_score * 0.3)
        
        return min(1.0, max(0.0, health_score))
    
    def _generate_behavioral_insights(
        self, 
        patterns: List[BehaviorPattern],
        norms: List[BehavioralNorm],
        alignments: List[BehaviorCultureAlignment]
    ) -> List[str]:
        """Generate key behavioral insights"""
        insights = []
        
        if patterns:
            strong_patterns = [p for p in patterns if p.strength > 0.7]
            weak_patterns = [p for p in patterns if p.strength < 0.3]
            
            if strong_patterns:
                insights.append(f"Identified {len(strong_patterns)} strong behavioral patterns")
            if weak_patterns:
                insights.append(f"Found {len(weak_patterns)} weak patterns requiring attention")
        
        if norms:
            high_compliance = [n for n in norms if n.compliance_rate > 0.8]
            low_compliance = [n for n in norms if n.compliance_rate < 0.5]
            
            if high_compliance:
                insights.append(f"{len(high_compliance)} behavioral norms show high compliance")
            if low_compliance:
                insights.append(f"{len(low_compliance)} norms need compliance improvement")
        
        if alignments:
            well_aligned = [a for a in alignments if a.alignment_level in [
                AlignmentLevel.WELL_ALIGNED, AlignmentLevel.PERFECTLY_ALIGNED
            ]]
            misaligned = [a for a in alignments if a.alignment_level == AlignmentLevel.MISALIGNED]
            
            if well_aligned:
                insights.append(f"{len(well_aligned)} behaviors are well-aligned with culture")
            if misaligned:
                insights.append(f"{len(misaligned)} behaviors are misaligned and need intervention")
        
        return insights
    
    def _generate_behavioral_recommendations(
        self, 
        patterns: List[BehaviorPattern],
        norms: List[BehavioralNorm],
        alignments: List[BehaviorCultureAlignment]
    ) -> List[str]:
        """Generate behavioral improvement recommendations"""
        recommendations = []
        
        # Pattern-based recommendations
        weak_patterns = [p for p in patterns if p.strength < 0.5]
        if weak_patterns:
            recommendations.append("Strengthen weak behavioral patterns through targeted interventions")
        
        # Norm-based recommendations
        low_compliance_norms = [n for n in norms if n.compliance_rate < 0.6]
        if low_compliance_norms:
            recommendations.append("Improve compliance with behavioral norms through reinforcement")
        
        # Alignment-based recommendations
        misaligned = [a for a in alignments if a.alignment_level == AlignmentLevel.MISALIGNED]
        if misaligned:
            recommendations.append("Address behavior-culture misalignments through coaching and training")
        
        # General recommendations
        recommendations.extend([
            "Implement regular behavioral monitoring and feedback",
            "Establish clear behavioral expectations and consequences",
            "Recognize and reward positive behavioral patterns",
            "Provide ongoing behavioral development opportunities"
        ])
        
        return recommendations
    
    def calculate_behavior_metrics(
        self, 
        analysis_result: BehaviorAnalysisResult
    ) -> BehaviorMetrics:
        """Calculate comprehensive behavioral metrics"""
        
        patterns = analysis_result.behavior_patterns
        norms = analysis_result.behavioral_norms
        alignments = analysis_result.culture_alignments
        
        # Behavior diversity index
        behavior_types = set(p.behavior_type for p in patterns)
        diversity_index = len(behavior_types) / len(BehaviorType) if patterns else 0.0
        
        # Norm compliance average
        compliance_avg = sum(n.compliance_rate for n in norms) / len(norms) if norms else 0.0
        
        # Culture alignment score
        alignment_score = sum(a.alignment_score for a in alignments) / len(alignments) if alignments else 0.0
        
        # Behavior consistency index (based on pattern strengths)
        consistency_index = sum(p.strength for p in patterns) / len(patterns) if patterns else 0.0
        
        # Positive behavior ratio
        positive_patterns = [p for p in patterns if p.strength > 0.6]
        positive_ratio = len(positive_patterns) / len(patterns) if patterns else 0.0
        
        # Improvement trend (simplified - would need historical data)
        improvement_trend = 0.0  # Placeholder
        
        return BehaviorMetrics(
            behavior_diversity_index=diversity_index,
            norm_compliance_average=compliance_avg,
            culture_alignment_score=alignment_score,
            behavior_consistency_index=consistency_index,
            positive_behavior_ratio=positive_ratio,
            improvement_trend=improvement_trend,
            calculated_date=datetime.now()
        )
    
    def get_analysis_result(self, analysis_id: str) -> Optional[BehaviorAnalysisResult]:
        """Retrieve behavioral analysis result by ID"""
        return self.analysis_results.get(analysis_id)
    
    def get_organization_analyses(self, organization_id: str) -> List[BehaviorAnalysisResult]:
        """Get all analyses for an organization"""
        return [
            result for result in self.analysis_results.values()
            if result.organization_id == organization_id
        ]