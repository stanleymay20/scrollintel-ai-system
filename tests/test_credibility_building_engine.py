"""
Tests for Credibility Building Engine

This module contains comprehensive tests for the credibility building
functionality in the Board Executive Mastery system.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import numpy as np

from scrollintel.engines.credibility_building_engine import CredibilityBuildingEngine
from scrollintel.models.credibility_models import (
    CredibilityLevel, CredibilityFactor, CredibilityAssessment,
    CredibilityPlan, StakeholderProfile, RelationshipEvent
)


class TestCredibilityBuildingEngine:
    """Test cases for CredibilityBuildingEngine"""
    
    @pytest.fixture
    def engine(self):
        """Create CredibilityBuildingEngine instance for testing"""
        return CredibilityBuildingEngine()
    
    @pytest.fixture
    def sample_evidence_data(self):
        """Sample evidence data for testing"""
        return {
            "expertise": {
                "years_experience": 15,
                "certifications": 5,
                "domain_knowledge_score": 0.8,
                "evidence": ["Led major technical initiatives", "Industry recognition"],
                "historical_scores": [0.7, 0.75, 0.8]
            },
            "track_record": {
                "success_rate": 0.85,
                "project_count": 25,
                "evidence": ["Successful project deliveries", "Positive outcomes"],
                "historical_scores": [0.8, 0.82, 0.85]
            },
            "transparency": {
                "transparency_score": 0.7,
                "evidence": ["Open communication", "Regular updates"],
                "historical_scores": [0.6, 0.65, 0.7]
            },
            "consistency": {
                "consistency_score": 0.75,
                "evidence": ["Consistent messaging", "Reliable performance"],
                "historical_scores": [0.7, 0.72, 0.75]
            },
            "communication": {
                "communication_effectiveness": 0.8,
                "evidence": ["Clear presentations", "Effective meetings"],
                "historical_scores": [0.75, 0.77, 0.8]
            },
            "results_delivery": {
                "delivery_score": 0.9,
                "evidence": ["On-time deliveries", "Quality outcomes"],
                "historical_scores": [0.85, 0.87, 0.9]
            },
            "strategic_insight": {
                "strategic_thinking_score": 0.85,
                "evidence": ["Strategic recommendations", "Vision development"],
                "historical_scores": [0.8, 0.82, 0.85]
            },
            "problem_solving": {
                "problem_solving_score": 0.8,
                "evidence": ["Complex problem resolution", "Innovative solutions"],
                "historical_scores": [0.75, 0.77, 0.8]
            }
        }
    
    @pytest.fixture
    def sample_stakeholder_profile(self):
        """Sample stakeholder profile for testing"""
        return StakeholderProfile(
            id="stakeholder_1",
            name="John Board",
            role="Board Chair",
            background="Technology executive with 20 years experience",
            values=["innovation", "results", "transparency"],
            communication_preferences={"format": "formal", "frequency": "monthly"},
            decision_making_style="analytical",
            influence_level=0.9,
            credibility_assessment=None,
            trust_assessment=None,
            relationship_events=[]
        )
    
    def test_assess_credibility_success(self, engine, sample_evidence_data):
        """Test successful credibility assessment"""
        stakeholder_id = "test_stakeholder"
        
        assessment = engine.assess_credibility(stakeholder_id, sample_evidence_data)
        
        assert assessment.stakeholder_id == stakeholder_id
        assert isinstance(assessment.overall_score, float)
        assert 0.0 <= assessment.overall_score <= 1.0
        assert isinstance(assessment.level, CredibilityLevel)
        assert len(assessment.metrics) == len(CredibilityFactor)
        assert isinstance(assessment.strengths, list)
        assert isinstance(assessment.improvement_areas, list)
        assert isinstance(assessment.assessment_date, datetime)
    
    def test_assess_credibility_high_scores(self, engine):
        """Test credibility assessment with high scores"""
        evidence_data = {
            factor.value: {
                f"{factor.value}_score": 0.9,
                "evidence": ["High performance evidence"],
                "historical_scores": [0.85, 0.87, 0.9]
            }
            for factor in CredibilityFactor
        }
        
        assessment = engine.assess_credibility("high_performer", evidence_data)
        
        assert assessment.level in [CredibilityLevel.HIGH, CredibilityLevel.EXCEPTIONAL]
        assert assessment.overall_score > 0.8
        assert len(assessment.strengths) > 0
    
    def test_assess_credibility_low_scores(self, engine):
        """Test credibility assessment with low scores"""
        evidence_data = {
            factor.value: {
                f"{factor.value}_score": 0.3,
                "evidence": ["Limited evidence"],
                "historical_scores": [0.3, 0.3, 0.3]
            }
            for factor in CredibilityFactor
        }
        
        assessment = engine.assess_credibility("low_performer", evidence_data)
        
        assert assessment.level in [CredibilityLevel.LOW, CredibilityLevel.MODERATE]
        assert assessment.overall_score < 0.6
        assert len(assessment.improvement_areas) > 0
    
    def test_develop_credibility_plan(self, engine, sample_evidence_data):
        """Test credibility plan development"""
        assessment = engine.assess_credibility("test_stakeholder", sample_evidence_data)
        target_level = CredibilityLevel.EXCEPTIONAL
        
        plan = engine.develop_credibility_plan(assessment, target_level)
        
        assert plan.stakeholder_id == assessment.stakeholder_id
        assert plan.target_level == target_level
        assert len(plan.actions) > 0
        assert len(plan.milestones) > 0
        assert isinstance(plan.timeline, str)
        assert len(plan.monitoring_schedule) > 0
        assert len(plan.contingency_plans) > 0
    
    def test_track_credibility_progress(self, engine, sample_evidence_data):
        """Test credibility progress tracking"""
        assessment = engine.assess_credibility("test_stakeholder", sample_evidence_data)
        plan = engine.develop_credibility_plan(assessment, CredibilityLevel.HIGH)
        
        # Create sample recent events
        recent_events = [
            RelationshipEvent(
                id="event_1",
                stakeholder_id="test_stakeholder",
                event_type="presentation",
                description="Successful board presentation",
                date=datetime.now(),
                credibility_impact=0.1,
                trust_impact=0.05,
                lessons_learned=["Clear communication works"],
                follow_up_actions=["Continue this approach"]
            )
        ]
        
        progress = engine.track_credibility_progress(plan, recent_events)
        
        assert "plan_id" in progress
        assert "stakeholder_id" in progress
        assert "tracking_date" in progress
        assert "actions_completed" in progress
        assert "actions_in_progress" in progress
        assert "actions_planned" in progress
        assert "recent_impacts" in progress
        assert "recommendations" in progress
        assert len(progress["recent_impacts"]) == 1
    
    def test_optimize_credibility_strategy(self, engine, sample_stakeholder_profile):
        """Test credibility strategy optimization"""
        context = {
            "current_situation": "preparing for board meeting",
            "timeline": "2 months",
            "priorities": ["technical expertise", "strategic vision"]
        }
        
        optimization = engine.optimize_credibility_strategy(sample_stakeholder_profile, context)
        
        assert "stakeholder_id" in optimization
        assert "optimization_date" in optimization
        assert "personalized_approach" in optimization
        assert "priority_factors" in optimization
        assert "communication_strategy" in optimization
        assert "timing_recommendations" in optimization
        assert "risk_mitigation" in optimization
    
    def test_generate_credibility_report(self, engine, sample_evidence_data):
        """Test credibility report generation"""
        # Create multiple assessments
        assessments = []
        for i in range(3):
            assessment = engine.assess_credibility(f"stakeholder_{i}", sample_evidence_data)
            assessments.append(assessment)
        
        report = engine.generate_credibility_report(assessments)
        
        assert len(report.stakeholder_assessments) == 3
        assert isinstance(report.overall_credibility_score, float)
        assert isinstance(report.key_achievements, list)
        assert isinstance(report.areas_for_improvement, list)
        assert isinstance(report.recommended_actions, list)
        assert isinstance(report.trend_analysis, dict)
        assert isinstance(report.next_review_date, datetime)
    
    def test_calculate_factor_score_expertise(self, engine):
        """Test expertise factor score calculation"""
        evidence_data = {
            "expertise": {
                "years_experience": 20,
                "certifications": 8,
                "domain_knowledge_score": 0.9
            }
        }
        
        score = engine._calculate_factor_score(CredibilityFactor.EXPERTISE, evidence_data)
        
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        assert score > 0.5  # Should be high given the input
    
    def test_calculate_factor_score_track_record(self, engine):
        """Test track record factor score calculation"""
        evidence_data = {
            "track_record": {
                "success_rate": 0.9,
                "project_count": 30
            }
        }
        
        score = engine._calculate_factor_score(CredibilityFactor.TRACK_RECORD, evidence_data)
        
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        assert score > 0.7  # Should be high given the input
    
    def test_determine_credibility_level(self, engine):
        """Test credibility level determination"""
        assert engine._determine_credibility_level(0.95) == CredibilityLevel.EXCEPTIONAL
        assert engine._determine_credibility_level(0.80) == CredibilityLevel.HIGH
        assert engine._determine_credibility_level(0.60) == CredibilityLevel.MODERATE
        assert engine._determine_credibility_level(0.40) == CredibilityLevel.LOW
    
    def test_identify_strengths(self, engine):
        """Test strength identification"""
        from scrollintel.models.credibility_models import CredibilityMetric
        
        metrics = [
            CredibilityMetric(
                factor=CredibilityFactor.EXPERTISE,
                score=0.8,
                evidence=["Strong technical background"],
                last_updated=datetime.now(),
                trend="improving"
            ),
            CredibilityMetric(
                factor=CredibilityFactor.COMMUNICATION,
                score=0.6,
                evidence=["Good presentation skills"],
                last_updated=datetime.now(),
                trend="stable"
            )
        ]
        
        strengths = engine._identify_strengths(metrics)
        
        assert len(strengths) == 1
        assert "expertise" in strengths[0].lower()
    
    def test_identify_improvement_areas(self, engine):
        """Test improvement area identification"""
        from scrollintel.models.credibility_models import CredibilityMetric
        
        metrics = [
            CredibilityMetric(
                factor=CredibilityFactor.TRANSPARENCY,
                score=0.5,
                evidence=["Limited transparency"],
                last_updated=datetime.now(),
                trend="stable"
            ),
            CredibilityMetric(
                factor=CredibilityFactor.CONSISTENCY,
                score=0.8,
                evidence=["Consistent performance"],
                last_updated=datetime.now(),
                trend="improving"
            )
        ]
        
        improvement_areas = engine._identify_improvement_areas(metrics)
        
        assert len(improvement_areas) == 1
        assert "transparency" in improvement_areas[0].lower()
    
    def test_analyze_trend_improving(self, engine):
        """Test trend analysis for improving scores"""
        evidence_data = {
            "expertise": {
                "historical_scores": [0.6, 0.65, 0.7, 0.75, 0.8, 0.85]
            }
        }
        
        trend = engine._analyze_trend(CredibilityFactor.EXPERTISE, evidence_data)
        
        assert trend == "improving"
    
    def test_analyze_trend_declining(self, engine):
        """Test trend analysis for declining scores"""
        evidence_data = {
            "expertise": {
                "historical_scores": [0.8, 0.75, 0.7, 0.65, 0.6, 0.55]
            }
        }
        
        trend = engine._analyze_trend(CredibilityFactor.EXPERTISE, evidence_data)
        
        assert trend == "declining"
    
    def test_analyze_trend_stable(self, engine):
        """Test trend analysis for stable scores"""
        evidence_data = {
            "expertise": {
                "historical_scores": [0.7, 0.71, 0.69, 0.7, 0.72, 0.71]
            }
        }
        
        trend = engine._analyze_trend(CredibilityFactor.EXPERTISE, evidence_data)
        
        assert trend == "stable"
    
    def test_create_credibility_action(self, engine, sample_evidence_data):
        """Test credibility action creation"""
        assessment = engine.assess_credibility("test_stakeholder", sample_evidence_data)
        
        action = engine._create_credibility_action(CredibilityFactor.EXPERTISE, assessment)
        
        assert action.target_factor == CredibilityFactor.EXPERTISE
        assert isinstance(action.title, str)
        assert isinstance(action.description, str)
        assert isinstance(action.expected_impact, float)
        assert isinstance(action.timeline, str)
        assert isinstance(action.resources_required, list)
        assert isinstance(action.success_metrics, list)
        assert action.status == "planned"
    
    def test_create_milestones(self, engine, sample_evidence_data):
        """Test milestone creation"""
        assessment = engine.assess_credibility("test_stakeholder", sample_evidence_data)
        target_level = CredibilityLevel.EXCEPTIONAL
        
        milestones = engine._create_milestones(assessment, target_level)
        
        assert len(milestones) == 4
        for milestone in milestones:
            assert "milestone" in milestone
            assert "target_score" in milestone
            assert "target_date" in milestone
            assert "achieved" in milestone
            assert "description" in milestone
    
    def test_personalize_credibility_approach(self, engine, sample_stakeholder_profile):
        """Test credibility approach personalization"""
        approach = engine._personalize_credibility_approach(sample_stakeholder_profile)
        
        assert "communication_style" in approach
        assert "value_alignment" in approach
        assert "decision_style_adaptation" in approach
        assert "influence_considerations" in approach
    
    def test_identify_priority_factors(self, engine, sample_stakeholder_profile):
        """Test priority factor identification"""
        priority_factors = engine._identify_priority_factors(sample_stakeholder_profile)
        
        assert isinstance(priority_factors, list)
        # Should include expertise for technical background
        assert any("expertise" in factor for factor in priority_factors)
    
    def test_error_handling_invalid_data(self, engine):
        """Test error handling with invalid data"""
        with pytest.raises(Exception):
            engine.assess_credibility("", {})
    
    def test_error_handling_missing_evidence(self, engine):
        """Test handling of missing evidence data"""
        minimal_data = {"expertise": {}}
        
        # Should not raise exception, should use defaults
        assessment = engine.assess_credibility("test_stakeholder", minimal_data)
        
        assert assessment is not None
        assert assessment.overall_score >= 0.0
    
    @pytest.mark.asyncio
    async def test_concurrent_assessments(self, engine, sample_evidence_data):
        """Test concurrent credibility assessments"""
        import asyncio
        
        async def assess_stakeholder(stakeholder_id):
            return engine.assess_credibility(stakeholder_id, sample_evidence_data)
        
        tasks = [assess_stakeholder(f"stakeholder_{i}") for i in range(5)]
        assessments = await asyncio.gather(*tasks)
        
        assert len(assessments) == 5
        for assessment in assessments:
            assert assessment is not None
            assert isinstance(assessment.overall_score, float)
    
    def test_credibility_factors_weights(self, engine):
        """Test that credibility factor weights sum to 1.0"""
        total_weight = sum(engine.credibility_factors.values())
        assert abs(total_weight - 1.0) < 0.01  # Allow for small floating point errors
    
    def test_get_target_score(self, engine):
        """Test target score calculation for different levels"""
        assert engine._get_target_score(CredibilityLevel.LOW) == 0.40
        assert engine._get_target_score(CredibilityLevel.MODERATE) == 0.60
        assert engine._get_target_score(CredibilityLevel.HIGH) == 0.80
        assert engine._get_target_score(CredibilityLevel.EXCEPTIONAL) == 0.95
    
    def test_calculate_timeline(self, engine, sample_evidence_data):
        """Test timeline calculation based on score gap"""
        assessment = engine.assess_credibility("test_stakeholder", sample_evidence_data)
        
        # Test different target levels
        timeline_low = engine._calculate_timeline(assessment, CredibilityLevel.LOW)
        timeline_exceptional = engine._calculate_timeline(assessment, CredibilityLevel.EXCEPTIONAL)
        
        assert isinstance(timeline_low, str)
        assert isinstance(timeline_exceptional, str)
        # Exceptional should take longer if current score is not already high
        if assessment.overall_score < 0.8:
            assert "month" in timeline_exceptional.lower()


if __name__ == "__main__":
    pytest.main([__file__])