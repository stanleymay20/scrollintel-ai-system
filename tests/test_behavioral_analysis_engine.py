"""
Tests for Behavioral Analysis Engine

Tests behavioral pattern identification, norm assessment,
and behavior-culture alignment analysis.
"""

import pytest
from datetime import datetime
from uuid import uuid4

from scrollintel.engines.behavioral_analysis_engine import BehavioralAnalysisEngine
from scrollintel.models.behavioral_analysis_models import (
    BehaviorObservation, BehaviorType, AlignmentLevel
)


class TestBehavioralAnalysisEngine:
    """Test suite for BehavioralAnalysisEngine"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.engine = BehavioralAnalysisEngine()
        self.organization_id = "test-org-123"
        self.cultural_values = ["collaboration", "innovation", "excellence", "integrity"]
        
        # Create sample observations
        self.sample_observations = [
            BehaviorObservation(
                id=str(uuid4()),
                observer_id="observer1",
                observed_behavior="Team members actively share knowledge",
                behavior_type=BehaviorType.COLLABORATION,
                context={"triggers": ["project_start"], "outcomes": ["improved_efficiency"]},
                participants=["alice", "bob", "charlie"],
                timestamp=datetime.now(),
                impact_assessment="Positive impact on team productivity",
                cultural_relevance=0.8
            ),
            BehaviorObservation(
                id=str(uuid4()),
                observer_id="observer1",
                observed_behavior="Team members actively share knowledge",
                behavior_type=BehaviorType.COLLABORATION,
                context={"triggers": ["project_start"], "outcomes": ["improved_efficiency"]},
                participants=["alice", "bob", "david"],
                timestamp=datetime.now(),
                impact_assessment="Positive impact on team productivity",
                cultural_relevance=0.8
            ),
            BehaviorObservation(
                id=str(uuid4()),
                observer_id="observer1",
                observed_behavior="Team members actively share knowledge",
                behavior_type=BehaviorType.COLLABORATION,
                context={"triggers": ["project_start"], "outcomes": ["improved_efficiency"]},
                participants=["alice", "charlie", "eve"],
                timestamp=datetime.now(),
                impact_assessment="Positive impact on team productivity",
                cultural_relevance=0.8
            ),
            BehaviorObservation(
                id=str(uuid4()),
                observer_id="observer2",
                observed_behavior="Manager provides clear direction",
                behavior_type=BehaviorType.LEADERSHIP,
                context={"triggers": ["team_confusion"], "outcomes": ["clarity_achieved"]},
                participants=["manager1", "team"],
                timestamp=datetime.now(),
                impact_assessment="Resolved team confusion",
                cultural_relevance=0.9
            ),
            BehaviorObservation(
                id=str(uuid4()),
                observer_id="observer2",
                observed_behavior="Manager provides clear direction",
                behavior_type=BehaviorType.LEADERSHIP,
                context={"triggers": ["team_confusion"], "outcomes": ["clarity_achieved"]},
                participants=["manager2", "team"],
                timestamp=datetime.now(),
                impact_assessment="Resolved team confusion",
                cultural_relevance=0.9
            ),
            BehaviorObservation(
                id=str(uuid4()),
                observer_id="observer2",
                observed_behavior="Manager provides clear direction",
                behavior_type=BehaviorType.LEADERSHIP,
                context={"triggers": ["team_confusion"], "outcomes": ["clarity_achieved"]},
                participants=["manager3", "team"],
                timestamp=datetime.now(),
                impact_assessment="Resolved team confusion",
                cultural_relevance=0.9
            ),
            BehaviorObservation(
                id=str(uuid4()),
                observer_id="observer3",
                observed_behavior="Employee suggests process improvement",
                behavior_type=BehaviorType.INNOVATION,
                context={"triggers": ["inefficiency_noticed"], "outcomes": ["process_improved"]},
                participants=["employee1"],
                timestamp=datetime.now(),
                impact_assessment="Led to process optimization",
                cultural_relevance=0.7
            ),
            BehaviorObservation(
                id=str(uuid4()),
                observer_id="observer3",
                observed_behavior="Employee suggests process improvement",
                behavior_type=BehaviorType.INNOVATION,
                context={"triggers": ["inefficiency_noticed"], "outcomes": ["process_improved"]},
                participants=["employee2"],
                timestamp=datetime.now(),
                impact_assessment="Led to process optimization",
                cultural_relevance=0.7
            ),
            BehaviorObservation(
                id=str(uuid4()),
                observer_id="observer3",
                observed_behavior="Employee suggests process improvement",
                behavior_type=BehaviorType.INNOVATION,
                context={"triggers": ["inefficiency_noticed"], "outcomes": ["process_improved"]},
                participants=["employee3"],
                timestamp=datetime.now(),
                impact_assessment="Led to process optimization",
                cultural_relevance=0.7
            )
        ]
    
    def test_analyze_organizational_behaviors(self):
        """Test comprehensive behavioral analysis"""
        result = self.engine.analyze_organizational_behaviors(
            organization_id=self.organization_id,
            observations=self.sample_observations,
            cultural_values=self.cultural_values,
            analyst="Test Analyst"
        )
        
        # Verify result structure
        assert result.organization_id == self.organization_id
        assert result.analysis_id is not None
        assert result.analyst == "Test Analyst"
        assert isinstance(result.behavior_patterns, list)
        assert isinstance(result.behavioral_norms, list)
        assert isinstance(result.culture_alignments, list)
        assert 0.0 <= result.overall_health_score <= 1.0
        assert isinstance(result.key_insights, list)
        assert isinstance(result.recommendations, list)
        
        # Verify patterns were identified
        assert len(result.behavior_patterns) > 0
        
        # Verify norms were assessed
        assert len(result.behavioral_norms) > 0
        
        # Verify culture alignments were analyzed
        assert len(result.culture_alignments) > 0
    
    def test_identify_behavior_patterns(self):
        """Test behavior pattern identification"""
        patterns = self.engine._identify_behavior_patterns(self.sample_observations)
        
        # Should identify patterns from repeated behaviors
        assert len(patterns) >= 2  # At least collaboration and leadership patterns
        
        # Verify pattern structure
        for pattern in patterns:
            assert pattern.id is not None
            assert pattern.name is not None
            assert pattern.behavior_type in BehaviorType
            assert 0.0 <= pattern.strength <= 1.0
            assert isinstance(pattern.triggers, list)
            assert isinstance(pattern.outcomes, list)
            assert isinstance(pattern.participants, list)
    
    def test_assess_behavioral_norms(self):
        """Test behavioral norm assessment"""
        patterns = self.engine._identify_behavior_patterns(self.sample_observations)
        norms = self.engine._assess_behavioral_norms(self.sample_observations, patterns)
        
        # Should create norms for different behavior types
        assert len(norms) > 0
        
        # Verify norm structure
        for norm in norms:
            assert norm.id is not None
            assert norm.behavior_type in BehaviorType
            assert 0.0 <= norm.compliance_rate <= 1.0
            assert 0.0 <= norm.cultural_importance <= 1.0
            assert isinstance(norm.expected_behaviors, list)
            assert isinstance(norm.discouraged_behaviors, list)
            assert isinstance(norm.enforcement_mechanisms, list)
    
    def test_analyze_culture_alignment(self):
        """Test behavior-culture alignment analysis"""
        patterns = self.engine._identify_behavior_patterns(self.sample_observations)
        alignments = self.engine._analyze_culture_alignment(patterns, self.cultural_values)
        
        # Should create alignments for each pattern-value combination
        expected_alignments = len(patterns) * len(self.cultural_values)
        assert len(alignments) == expected_alignments
        
        # Verify alignment structure
        for alignment in alignments:
            assert alignment.id is not None
            assert alignment.behavior_pattern_id in [p.id for p in patterns]
            assert alignment.cultural_value in self.cultural_values
            assert alignment.alignment_level in AlignmentLevel
            assert 0.0 <= alignment.alignment_score <= 1.0
            assert isinstance(alignment.supporting_evidence, list)
            assert isinstance(alignment.conflicting_evidence, list)
            assert isinstance(alignment.recommendations, list)
    
    def test_calculate_behavioral_health_score(self):
        """Test behavioral health score calculation"""
        patterns = self.engine._identify_behavior_patterns(self.sample_observations)
        norms = self.engine._assess_behavioral_norms(self.sample_observations, patterns)
        alignments = self.engine._analyze_culture_alignment(patterns, self.cultural_values)
        
        health_score = self.engine._calculate_behavioral_health_score(patterns, norms, alignments)
        
        # Should return valid score
        assert 0.0 <= health_score <= 1.0
        
        # Test with empty inputs
        empty_score = self.engine._calculate_behavioral_health_score([], [], [])
        assert empty_score == 0.5  # Neutral score
    
    def test_generate_behavioral_insights(self):
        """Test behavioral insight generation"""
        patterns = self.engine._identify_behavior_patterns(self.sample_observations)
        norms = self.engine._assess_behavioral_norms(self.sample_observations, patterns)
        alignments = self.engine._analyze_culture_alignment(patterns, self.cultural_values)
        
        insights = self.engine._generate_behavioral_insights(patterns, norms, alignments)
        
        # Should generate meaningful insights
        assert len(insights) > 0
        assert all(isinstance(insight, str) for insight in insights)
        assert all(len(insight) > 0 for insight in insights)
    
    def test_generate_behavioral_recommendations(self):
        """Test behavioral recommendation generation"""
        patterns = self.engine._identify_behavior_patterns(self.sample_observations)
        norms = self.engine._assess_behavioral_norms(self.sample_observations, patterns)
        alignments = self.engine._analyze_culture_alignment(patterns, self.cultural_values)
        
        recommendations = self.engine._generate_behavioral_recommendations(patterns, norms, alignments)
        
        # Should generate actionable recommendations
        assert len(recommendations) > 0
        assert all(isinstance(rec, str) for rec in recommendations)
        assert all(len(rec) > 0 for rec in recommendations)
    
    def test_calculate_behavior_metrics(self):
        """Test behavioral metrics calculation"""
        result = self.engine.analyze_organizational_behaviors(
            organization_id=self.organization_id,
            observations=self.sample_observations,
            cultural_values=self.cultural_values
        )
        
        metrics = self.engine.calculate_behavior_metrics(result)
        
        # Verify metrics structure
        assert 0.0 <= metrics.behavior_diversity_index <= 1.0
        assert 0.0 <= metrics.norm_compliance_average <= 1.0
        assert 0.0 <= metrics.culture_alignment_score <= 1.0
        assert 0.0 <= metrics.behavior_consistency_index <= 1.0
        assert 0.0 <= metrics.positive_behavior_ratio <= 1.0
        assert metrics.calculated_date is not None
    
    def test_get_analysis_result(self):
        """Test analysis result retrieval"""
        result = self.engine.analyze_organizational_behaviors(
            organization_id=self.organization_id,
            observations=self.sample_observations,
            cultural_values=self.cultural_values
        )
        
        # Should retrieve the same result
        retrieved = self.engine.get_analysis_result(result.analysis_id)
        assert retrieved is not None
        assert retrieved.analysis_id == result.analysis_id
        assert retrieved.organization_id == result.organization_id
        
        # Should return None for non-existent ID
        non_existent = self.engine.get_analysis_result("non-existent-id")
        assert non_existent is None
    
    def test_get_organization_analyses(self):
        """Test organization analyses retrieval"""
        # Create multiple analyses
        result1 = self.engine.analyze_organizational_behaviors(
            organization_id=self.organization_id,
            observations=self.sample_observations,
            cultural_values=self.cultural_values
        )
        
        result2 = self.engine.analyze_organizational_behaviors(
            organization_id=self.organization_id,
            observations=self.sample_observations[:2],
            cultural_values=self.cultural_values
        )
        
        # Should retrieve both analyses
        analyses = self.engine.get_organization_analyses(self.organization_id)
        assert len(analyses) == 2
        assert result1.analysis_id in [a.analysis_id for a in analyses]
        assert result2.analysis_id in [a.analysis_id for a in analyses]
        
        # Should return empty list for non-existent organization
        empty_analyses = self.engine.get_organization_analyses("non-existent-org")
        assert len(empty_analyses) == 0
    
    def test_behavior_observation_validation(self):
        """Test behavior observation validation"""
        # Valid observation should not raise exception
        valid_obs = BehaviorObservation(
            id="test-id",
            observer_id="observer1",
            observed_behavior="Test behavior",
            behavior_type=BehaviorType.COMMUNICATION,
            context={},
            participants=["person1"],
            timestamp=datetime.now(),
            impact_assessment="Test impact",
            cultural_relevance=0.5
        )
        
        # Should not raise exception
        assert valid_obs.cultural_relevance == 0.5
        
        # Invalid cultural relevance should raise exception
        with pytest.raises(ValueError):
            BehaviorObservation(
                id="test-id",
                observer_id="observer1",
                observed_behavior="Test behavior",
                behavior_type=BehaviorType.COMMUNICATION,
                context={},
                participants=["person1"],
                timestamp=datetime.now(),
                impact_assessment="Test impact",
                cultural_relevance=1.5  # Invalid: > 1.0
            )
    
    def test_empty_observations_handling(self):
        """Test handling of empty observations list"""
        result = self.engine.analyze_organizational_behaviors(
            organization_id=self.organization_id,
            observations=[],
            cultural_values=self.cultural_values
        )
        
        # Should handle empty observations gracefully
        assert result.organization_id == self.organization_id
        assert len(result.behavior_patterns) == 0
        assert len(result.behavioral_norms) == 0
        assert len(result.culture_alignments) == 0
        assert result.overall_health_score == 0.5  # Neutral score
        assert isinstance(result.key_insights, list)
        assert isinstance(result.recommendations, list)


if __name__ == "__main__":
    pytest.main([__file__])