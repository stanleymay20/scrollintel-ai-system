"""
Tests for Trust Management Engine

This module contains comprehensive tests for the trust management
functionality in the Board Executive Mastery system.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import numpy as np

from scrollintel.engines.trust_management_engine import TrustManagementEngine
from scrollintel.models.credibility_models import (
    TrustLevel, TrustAssessment, TrustBuildingStrategy,
    TrustRecoveryPlan, StakeholderProfile, RelationshipEvent
)


class TestTrustManagementEngine:
    """Test cases for TrustManagementEngine"""
    
    @pytest.fixture
    def engine(self):
        """Create TrustManagementEngine instance for testing"""
        return TrustManagementEngine()
    
    @pytest.fixture
    def sample_relationship_data(self):
        """Sample relationship data for testing"""
        return {
            "reliability": {
                "commitment_fulfillment_rate": 0.8,
                "consistency_score": 0.75,
                "punctuality_score": 0.9,
                "evidence": ["Met all project deadlines", "Consistent communication"],
                "historical_scores": [0.7, 0.75, 0.8],
                "last_interaction": datetime.now().isoformat()
            },
            "competence": {
                "technical_competence": 0.85,
                "problem_solving_ability": 0.8,
                "decision_quality": 0.75,
                "evidence": ["Solved complex technical issues", "Made sound decisions"],
                "historical_scores": [0.75, 0.78, 0.8],
                "last_interaction": datetime.now().isoformat()
            },
            "benevolence": {
                "stakeholder_focus": 0.7,
                "support_provided": 0.8,
                "consideration_shown": 0.75,
                "evidence": ["Supported team initiatives", "Considered stakeholder needs"],
                "historical_scores": [0.7, 0.72, 0.75],
                "last_interaction": datetime.now().isoformat()
            },
            "integrity": {
                "honesty_score": 0.9,
                "ethical_behavior": 0.85,
                "transparency_level": 0.8,
                "evidence": ["Transparent communication", "Ethical decision-making"],
                "historical_scores": [0.8, 0.82, 0.85],
                "last_interaction": datetime.now().isoformat()
            },
            "relationship_history": [
                {
                    "date": "2024-01-01",
                    "event": "Initial meeting",
                    "outcome": "positive"
                }
            ]
        }
    
    @pytest.fixture
    def sample_stakeholder_profile(self):
        """Sample stakeholder profile for testing"""
        return StakeholderProfile(
            id="stakeholder_1",
            name="Jane Executive",
            role="Board Member",
            background="Finance executive with 15 years experience",
            values=["integrity", "results", "collaboration"],
            communication_preferences={"format": "direct", "frequency": "weekly"},
            decision_making_style="data-driven",
            influence_level=0.8,
            credibility_assessment=None,
            trust_assessment=None,
            relationship_events=[]
        )
    
    def test_assess_trust_success(self, engine, sample_relationship_data):
        """Test successful trust assessment"""
        stakeholder_id = "test_stakeholder"
        
        assessment = engine.assess_trust(stakeholder_id, sample_relationship_data)
        
        assert assessment.stakeholder_id == stakeholder_id
        assert isinstance(assessment.overall_score, float)
        assert 0.0 <= assessment.overall_score <= 1.0
        assert isinstance(assessment.level, TrustLevel)
        assert len(assessment.metrics) == 4  # Four trust dimensions
        assert isinstance(assessment.trust_drivers, list)
        assert isinstance(assessment.trust_barriers, list)
        assert isinstance(assessment.assessment_date, datetime)
        assert isinstance(assessment.relationship_history, list)
    
    def test_assess_trust_high_scores(self, engine):
        """Test trust assessment with high scores"""
        relationship_data = {
            dimension: {
                f"{dimension}_score": 0.9,
                "evidence": ["High performance evidence"],
                "historical_scores": [0.85, 0.87, 0.9],
                "last_interaction": datetime.now().isoformat()
            }
            for dimension in ["reliability", "competence", "benevolence", "integrity"]
        }
        
        # Add specific sub-scores for each dimension
        relationship_data["reliability"].update({
            "commitment_fulfillment_rate": 0.9,
            "consistency_score": 0.9,
            "punctuality_score": 0.9
        })
        relationship_data["competence"].update({
            "technical_competence": 0.9,
            "problem_solving_ability": 0.9,
            "decision_quality": 0.9
        })
        relationship_data["benevolence"].update({
            "stakeholder_focus": 0.9,
            "support_provided": 0.9,
            "consideration_shown": 0.9
        })
        relationship_data["integrity"].update({
            "honesty_score": 0.9,
            "ethical_behavior": 0.9,
            "transparency_level": 0.9
        })
        
        assessment = engine.assess_trust("high_trust_stakeholder", relationship_data)
        
        assert assessment.level in [TrustLevel.TRUSTING, TrustLevel.COMPLETE_TRUST]
        assert assessment.overall_score > 0.8
        assert len(assessment.trust_drivers) > 0
    
    def test_assess_trust_low_scores(self, engine):
        """Test trust assessment with low scores"""
        relationship_data = {
            dimension: {
                f"{dimension}_score": 0.3,
                "evidence": ["Limited evidence"],
                "historical_scores": [0.3, 0.3, 0.3],
                "last_interaction": datetime.now().isoformat()
            }
            for dimension in ["reliability", "competence", "benevolence", "integrity"]
        }
        
        # Add specific sub-scores for each dimension
        for dimension in relationship_data:
            if dimension == "reliability":
                relationship_data[dimension].update({
                    "commitment_fulfillment_rate": 0.3,
                    "consistency_score": 0.3,
                    "punctuality_score": 0.3
                })
            elif dimension == "competence":
                relationship_data[dimension].update({
                    "technical_competence": 0.3,
                    "problem_solving_ability": 0.3,
                    "decision_quality": 0.3
                })
            elif dimension == "benevolence":
                relationship_data[dimension].update({
                    "stakeholder_focus": 0.3,
                    "support_provided": 0.3,
                    "consideration_shown": 0.3
                })
            elif dimension == "integrity":
                relationship_data[dimension].update({
                    "honesty_score": 0.3,
                    "ethical_behavior": 0.3,
                    "transparency_level": 0.3
                })
        
        assessment = engine.assess_trust("low_trust_stakeholder", relationship_data)
        
        assert assessment.level in [TrustLevel.DISTRUST, TrustLevel.CAUTIOUS]
        assert assessment.overall_score < 0.5
        assert len(assessment.trust_barriers) > 0
    
    def test_develop_trust_building_strategy(self, engine, sample_relationship_data):
        """Test trust building strategy development"""
        assessment = engine.assess_trust("test_stakeholder", sample_relationship_data)
        target_level = TrustLevel.COMPLETE_TRUST
        
        strategy = engine.develop_trust_building_strategy(assessment, target_level)
        
        assert strategy.stakeholder_id == assessment.stakeholder_id
        assert strategy.current_trust_level == assessment.level
        assert strategy.target_trust_level == target_level
        assert len(strategy.key_actions) > 0
        assert isinstance(strategy.timeline, str)
        assert len(strategy.milestones) > 0
        assert len(strategy.risk_factors) > 0
        assert len(strategy.success_indicators) > 0
    
    def test_track_trust_progress(self, engine, sample_relationship_data):
        """Test trust progress tracking"""
        assessment = engine.assess_trust("test_stakeholder", sample_relationship_data)
        strategy = engine.develop_trust_building_strategy(assessment, TrustLevel.TRUSTING)
        
        # Create sample recent events
        recent_events = [
            RelationshipEvent(
                id="event_1",
                stakeholder_id="test_stakeholder",
                event_type="meeting",
                description="Productive board meeting",
                date=datetime.now(),
                credibility_impact=0.05,
                trust_impact=0.1,
                lessons_learned=["Good preparation pays off"],
                follow_up_actions=["Schedule follow-up"]
            ),
            RelationshipEvent(
                id="event_2",
                stakeholder_id="test_stakeholder",
                event_type="collaboration",
                description="Joint project success",
                date=datetime.now() - timedelta(days=7),
                credibility_impact=0.1,
                trust_impact=0.15,
                lessons_learned=["Collaboration builds trust"],
                follow_up_actions=["Plan next collaboration"]
            )
        ]
        
        progress = engine.track_trust_progress(strategy, recent_events)
        
        assert "strategy_id" in progress
        assert "stakeholder_id" in progress
        assert "tracking_date" in progress
        assert "current_trust_level" in progress
        assert "target_trust_level" in progress
        assert "recent_trust_impacts" in progress
        assert "relationship_quality_indicators" in progress
        assert "recommendations" in progress
        assert len(progress["recent_trust_impacts"]) == 2
    
    def test_create_trust_recovery_plan(self, engine, sample_relationship_data):
        """Test trust recovery plan creation"""
        assessment = engine.assess_trust("test_stakeholder", sample_relationship_data)
        breach_description = "Failed to deliver on major commitment"
        target_level = TrustLevel.TRUSTING
        
        recovery_plan = engine.create_trust_recovery_plan(
            "test_stakeholder", breach_description, assessment, target_level
        )
        
        assert recovery_plan.stakeholder_id == "test_stakeholder"
        assert recovery_plan.trust_breach_description == breach_description
        assert recovery_plan.target_trust_level == target_level
        assert isinstance(recovery_plan.recovery_strategy, str)
        assert len(recovery_plan.immediate_actions) > 0
        assert len(recovery_plan.long_term_actions) > 0
        assert isinstance(recovery_plan.timeline, str)
        assert len(recovery_plan.success_metrics) > 0
        assert isinstance(recovery_plan.monitoring_plan, str)
    
    def test_measure_trust_effectiveness(self, engine, sample_stakeholder_profile):
        """Test trust effectiveness measurement"""
        # Create multiple stakeholder profiles with trust assessments
        profiles = []
        for i in range(3):
            profile = StakeholderProfile(
                id=f"stakeholder_{i}",
                name=f"Stakeholder {i}",
                role="Board Member",
                background="Executive background",
                values=["integrity", "results"],
                communication_preferences={},
                decision_making_style="analytical",
                influence_level=0.7,
                credibility_assessment=None,
                trust_assessment=None,  # Would be populated in real implementation
                relationship_events=[]
            )
            profiles.append(profile)
        
        effectiveness = engine.measure_trust_effectiveness(profiles)
        
        assert "measurement_date" in effectiveness
        assert "total_stakeholders" in effectiveness
        assert "trust_distribution" in effectiveness
        assert "average_trust_score" in effectiveness
        assert "high_trust_relationships" in effectiveness
        assert "at_risk_relationships" in effectiveness
        assert "improvement_opportunities" in effectiveness
        assert "success_stories" in effectiveness
        assert effectiveness["total_stakeholders"] == 3
    
    def test_calculate_trust_dimension_score_reliability(self, engine):
        """Test reliability dimension score calculation"""
        relationship_data = {
            "reliability": {
                "commitment_fulfillment_rate": 0.9,
                "consistency_score": 0.8,
                "punctuality_score": 0.85
            }
        }
        
        score = engine._calculate_trust_dimension_score("reliability", relationship_data)
        
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        expected_score = (0.9 + 0.8 + 0.85) / 3
        assert abs(score - expected_score) < 0.01
    
    def test_calculate_trust_dimension_score_competence(self, engine):
        """Test competence dimension score calculation"""
        relationship_data = {
            "competence": {
                "technical_competence": 0.85,
                "problem_solving_ability": 0.8,
                "decision_quality": 0.75
            }
        }
        
        score = engine._calculate_trust_dimension_score("competence", relationship_data)
        
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        expected_score = (0.85 + 0.8 + 0.75) / 3
        assert abs(score - expected_score) < 0.01
    
    def test_determine_trust_level(self, engine):
        """Test trust level determination"""
        assert engine._determine_trust_level(0.95) == TrustLevel.COMPLETE_TRUST
        assert engine._determine_trust_level(0.80) == TrustLevel.TRUSTING
        assert engine._determine_trust_level(0.60) == TrustLevel.NEUTRAL
        assert engine._determine_trust_level(0.40) == TrustLevel.CAUTIOUS
        assert engine._determine_trust_level(0.20) == TrustLevel.DISTRUST
    
    def test_identify_trust_drivers(self, engine):
        """Test trust driver identification"""
        from scrollintel.models.credibility_models import TrustMetric
        
        metrics = [
            TrustMetric(
                dimension="reliability",
                score=0.8,
                evidence=["Consistent performance"],
                last_interaction=datetime.now(),
                trend="improving"
            ),
            TrustMetric(
                dimension="competence",
                score=0.6,
                evidence=["Good technical skills"],
                last_interaction=datetime.now(),
                trend="stable"
            )
        ]
        
        drivers = engine._identify_trust_drivers(metrics)
        
        assert len(drivers) == 1
        assert "reliability" in drivers[0].lower()
    
    def test_identify_trust_barriers(self, engine):
        """Test trust barrier identification"""
        from scrollintel.models.credibility_models import TrustMetric
        
        metrics = [
            TrustMetric(
                dimension="integrity",
                score=0.4,
                evidence=["Some transparency issues"],
                last_interaction=datetime.now(),
                trend="stable"
            ),
            TrustMetric(
                dimension="benevolence",
                score=0.8,
                evidence=["Shows care for stakeholders"],
                last_interaction=datetime.now(),
                trend="improving"
            )
        ]
        
        barriers = engine._identify_trust_barriers(metrics)
        
        assert len(barriers) == 1
        assert "integrity" in barriers[0].lower()
    
    def test_determine_primary_approach(self, engine, sample_relationship_data):
        """Test primary approach determination"""
        assessment = engine.assess_trust("test_stakeholder", sample_relationship_data)
        
        # Modify assessment to have specific barriers
        assessment.trust_barriers = ["Weak competence"]
        approach = engine._determine_primary_approach(assessment)
        assert approach == "performance_focused"
        
        assessment.trust_barriers = ["Weak integrity"]
        approach = engine._determine_primary_approach(assessment)
        assert approach == "transparency_focused"
        
        assessment.trust_barriers = ["Weak benevolence"]
        approach = engine._determine_primary_approach(assessment)
        assert approach == "relationship_focused"
    
    def test_calculate_trust_timeline(self, engine):
        """Test trust timeline calculation"""
        timeline = engine._calculate_trust_timeline(TrustLevel.DISTRUST, TrustLevel.COMPLETE_TRUST)
        assert "12-18 months" in timeline
        
        timeline = engine._calculate_trust_timeline(TrustLevel.NEUTRAL, TrustLevel.TRUSTING)
        assert "3-6 months" in timeline
        
        timeline = engine._calculate_trust_timeline(TrustLevel.TRUSTING, TrustLevel.TRUSTING)
        assert "1-2 months" in timeline
    
    def test_create_trust_milestones(self, engine, sample_relationship_data):
        """Test trust milestone creation"""
        assessment = engine.assess_trust("test_stakeholder", sample_relationship_data)
        target_level = TrustLevel.COMPLETE_TRUST
        
        milestones = engine._create_trust_milestones(assessment, target_level)
        
        assert len(milestones) == 3
        for milestone in milestones:
            assert "milestone" in milestone
            assert "target_score" in milestone
            assert "target_date" in milestone
            assert "achieved" in milestone
            assert "description" in milestone
            assert "key_indicators" in milestone
    
    def test_get_trust_target_score(self, engine):
        """Test trust target score calculation"""
        assert engine._get_trust_target_score(TrustLevel.DISTRUST) == 0.25
        assert engine._get_trust_target_score(TrustLevel.CAUTIOUS) == 0.40
        assert engine._get_trust_target_score(TrustLevel.NEUTRAL) == 0.60
        assert engine._get_trust_target_score(TrustLevel.TRUSTING) == 0.80
        assert engine._get_trust_target_score(TrustLevel.COMPLETE_TRUST) == 0.95
    
    def test_determine_recovery_strategy(self, engine, sample_relationship_data):
        """Test recovery strategy determination"""
        assessment = engine.assess_trust("test_stakeholder", sample_relationship_data)
        
        strategy = engine._determine_recovery_strategy("Failed to deliver on time", assessment)
        assert strategy == "reliability_rebuilding"
        
        strategy = engine._determine_recovery_strategy("Made a dishonest statement", assessment)
        assert strategy == "transparency_and_accountability"
        
        strategy = engine._determine_recovery_strategy("Poor technical decision", assessment)
        assert strategy == "competence_demonstration"
        
        strategy = engine._determine_recovery_strategy("Ignored stakeholder concerns", assessment)
        assert strategy == "relationship_repair"
    
    def test_generate_immediate_recovery_actions(self, engine, sample_relationship_data):
        """Test immediate recovery action generation"""
        assessment = engine.assess_trust("test_stakeholder", sample_relationship_data)
        breach_description = "Missed important deadline"
        
        actions = engine._generate_immediate_recovery_actions(breach_description, assessment)
        
        assert len(actions) >= 4  # Base actions
        assert any("acknowledge" in action.lower() for action in actions)
        assert any("apologize" in action.lower() for action in actions)
        assert any("timeline" in action.lower() for action in actions)  # Should be added for deadline issues
    
    def test_assess_relationship_quality(self, engine):
        """Test relationship quality assessment"""
        recent_events = [
            RelationshipEvent(
                id="event_1",
                stakeholder_id="test_stakeholder",
                event_type="collaboration",
                description="Successful collaboration",
                date=datetime.now(),
                credibility_impact=0.1,
                trust_impact=0.15,
                lessons_learned=[],
                follow_up_actions=[]
            ),
            RelationshipEvent(
                id="event_2",
                stakeholder_id="test_stakeholder",
                event_type="conflict",
                description="Minor disagreement",
                date=datetime.now() - timedelta(days=3),
                credibility_impact=-0.05,
                trust_impact=-0.1,
                lessons_learned=[],
                follow_up_actions=[]
            )
        ]
        
        quality = engine._assess_relationship_quality(recent_events)
        
        assert quality["interaction_frequency"] == 2
        assert quality["positive_interactions"] == 1
        assert quality["negative_interactions"] == 1
        assert quality["collaboration_instances"] == 1
        assert quality["conflict_instances"] == 1
        assert quality["communication_quality"] == "neutral"  # Equal positive and negative
    
    def test_analyze_trust_trend_improving(self, engine):
        """Test trust trend analysis for improving scores"""
        relationship_data = {
            "reliability": {
                "historical_scores": [0.5, 0.55, 0.6, 0.65, 0.7, 0.75]
            }
        }
        
        trend = engine._analyze_trust_trend("reliability", relationship_data)
        
        assert trend == "improving"
    
    def test_analyze_trust_trend_declining(self, engine):
        """Test trust trend analysis for declining scores"""
        relationship_data = {
            "reliability": {
                "historical_scores": [0.8, 0.75, 0.7, 0.65, 0.6, 0.55]
            }
        }
        
        trend = engine._analyze_trust_trend("reliability", relationship_data)
        
        assert trend == "declining"
    
    def test_analyze_trust_trend_stable(self, engine):
        """Test trust trend analysis for stable scores"""
        relationship_data = {
            "reliability": {
                "historical_scores": [0.7, 0.71, 0.69, 0.7, 0.72, 0.71]
            }
        }
        
        trend = engine._analyze_trust_trend("reliability", relationship_data)
        
        assert trend == "stable"
    
    def test_trust_dimensions_weights(self, engine):
        """Test that trust dimension weights sum to 1.0"""
        total_weight = sum(engine.trust_dimensions.values())
        assert abs(total_weight - 1.0) < 0.01  # Allow for small floating point errors
    
    def test_error_handling_invalid_data(self, engine):
        """Test error handling with invalid data"""
        with pytest.raises(Exception):
            engine.assess_trust("", {})
    
    def test_error_handling_missing_relationship_data(self, engine):
        """Test handling of missing relationship data"""
        minimal_data = {"reliability": {}}
        
        # Should not raise exception, should use defaults
        assessment = engine.assess_trust("test_stakeholder", minimal_data)
        
        assert assessment is not None
        assert assessment.overall_score >= 0.0
    
    @pytest.mark.asyncio
    async def test_concurrent_trust_assessments(self, engine, sample_relationship_data):
        """Test concurrent trust assessments"""
        import asyncio
        
        async def assess_stakeholder_trust(stakeholder_id):
            return engine.assess_trust(stakeholder_id, sample_relationship_data)
        
        tasks = [assess_stakeholder_trust(f"stakeholder_{i}") for i in range(5)]
        assessments = await asyncio.gather(*tasks)
        
        assert len(assessments) == 5
        for assessment in assessments:
            assert assessment is not None
            assert isinstance(assessment.overall_score, float)
    
    def test_get_milestone_indicators(self, engine):
        """Test milestone indicator generation"""
        high_score_indicators = engine._get_milestone_indicators(0.85)
        assert "Proactive communication" in high_score_indicators
        
        medium_score_indicators = engine._get_milestone_indicators(0.65)
        assert "Regular interaction" in medium_score_indicators
        
        low_score_indicators = engine._get_milestone_indicators(0.45)
        assert "Basic communication" in low_score_indicators
    
    def test_define_success_indicators(self, engine):
        """Test success indicator definition"""
        complete_trust_indicators = engine._define_success_indicators(TrustLevel.COMPLETE_TRUST)
        assert any("seeks advice" in indicator.lower() for indicator in complete_trust_indicators)
        
        trusting_indicators = engine._define_success_indicators(TrustLevel.TRUSTING)
        assert any("positive feedback" in indicator.lower() for indicator in trusting_indicators)
        
        neutral_indicators = engine._define_success_indicators(TrustLevel.NEUTRAL)
        assert any("neutral to positive" in indicator.lower() for indicator in neutral_indicators)


if __name__ == "__main__":
    pytest.main([__file__])