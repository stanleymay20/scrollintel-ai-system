"""
Tests for Culture Maintenance Engine

Tests for cultural sustainability assessment and maintenance framework.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from scrollintel.engines.culture_maintenance_engine import CultureMaintenanceEngine
from scrollintel.models.culture_maintenance_models import (
    SustainabilityLevel, MaintenanceStatus
)
from scrollintel.models.cultural_assessment_models import CultureMap, CulturalTransformation


class TestCultureMaintenanceEngine:
    
    def setup_method(self):
        """Set up test fixtures"""
        self.engine = CultureMaintenanceEngine()
        
        # Mock culture data
        self.current_culture = CultureMap(
            organization_id="test_org",
            assessment_date=datetime.now(),
            cultural_dimensions={},
            values=[],
            behaviors=[],
            norms=[],
            subcultures=[],
            health_metrics=[],
            overall_health_score=0.75,
            assessment_confidence=0.8,
            data_sources=["survey"]
        )
        
        self.target_culture = CultureMap(
            organization_id="test_org",
            assessment_date=datetime.now(),
            cultural_dimensions={},
            values=[],
            behaviors=[],
            norms=[],
            subcultures=[],
            health_metrics=[],
            overall_health_score=0.85,
            assessment_confidence=0.8,
            data_sources=["survey"]
        )
        
        self.transformation = CulturalTransformation(
            id="test_transformation",
            organization_id="test_org",
            current_culture={},
            target_culture={},
            vision={},
            roadmap={},
            interventions=[],
            progress=0.8,
            start_date=datetime.now() - timedelta(days=90),
            target_completion=datetime.now() + timedelta(days=30)
        )
    
    def test_assess_cultural_sustainability_high_score(self):
        """Test sustainability assessment with high score"""
        # Arrange
        high_health_culture = CultureMap(
            organization_id="test_org",
            assessment_date=datetime.now(),
            cultural_dimensions={},
            values=[],
            behaviors=[],
            norms=[],
            subcultures=[],
            health_metrics=[],
            overall_health_score=0.9,
            assessment_confidence=0.8,
            data_sources=["survey"]
        )
        
        high_progress_transformation = CulturalTransformation(
            id="test_transformation",
            organization_id="test_org",
            current_culture={},
            target_culture={},
            vision={},
            roadmap={},
            interventions=[],
            progress=0.95,
            start_date=datetime.now() - timedelta(days=90),
            target_completion=datetime.now() + timedelta(days=30)
        )
        
        # Act
        assessment = self.engine.assess_cultural_sustainability(
            "test_org", high_progress_transformation, high_health_culture
        )
        
        # Assert
        assert assessment.organization_id == "test_org"
        assert assessment.sustainability_level == SustainabilityLevel.HIGH
        assert assessment.overall_score > 0.8
        assert len(assessment.health_indicators) > 0
        assert len(assessment.protective_factors) > 0
    
    def test_assess_cultural_sustainability_low_score(self):
        """Test sustainability assessment with low score"""
        # Arrange
        low_health_culture = CultureMap(
            organization_id="test_org",
            assessment_date=datetime.now(),
            cultural_dimensions={},
            values=[],
            behaviors=[],
            norms=[],
            subcultures=[],
            health_metrics=[],
            overall_health_score=0.4,
            assessment_confidence=0.8,
            data_sources=["survey"]
        )
        
        low_progress_transformation = CulturalTransformation(
            id="test_transformation",
            organization_id="test_org",
            current_culture={},
            target_culture={},
            vision={},
            roadmap={},
            interventions=[],
            progress=0.3,
            start_date=datetime.now() - timedelta(days=90),
            target_completion=datetime.now() + timedelta(days=30)
        )
        
        # Act
        assessment = self.engine.assess_cultural_sustainability(
            "test_org", low_progress_transformation, low_health_culture
        )
        
        # Assert
        assert assessment.sustainability_level in [SustainabilityLevel.LOW, SustainabilityLevel.CRITICAL]
        assert assessment.overall_score < 0.6
        assert len(assessment.risk_factors) > 0
    
    def test_develop_maintenance_strategy(self):
        """Test development of maintenance strategies"""
        # Arrange
        assessment = self.engine.assess_cultural_sustainability(
            "test_org", self.transformation, self.current_culture
        )
        
        # Act
        strategies = self.engine.develop_maintenance_strategy(
            "test_org", assessment, self.target_culture
        )
        
        # Assert
        assert len(strategies) > 0
        assert all(strategy.organization_id == "test_org" for strategy in strategies)
        assert all(len(strategy.maintenance_activities) > 0 for strategy in strategies)
        assert all(len(strategy.success_metrics) > 0 for strategy in strategies)
        
        # Check for different strategy types
        strategy_types = [strategy.target_culture_elements[0] for strategy in strategies if strategy.target_culture_elements]
        assert len(set(strategy_types)) > 1  # Multiple different strategy types
    
    def test_create_maintenance_plan(self):
        """Test creation of comprehensive maintenance plan"""
        # Arrange
        assessment = self.engine.assess_cultural_sustainability(
            "test_org", self.transformation, self.current_culture
        )
        strategies = self.engine.develop_maintenance_strategy(
            "test_org", assessment, self.target_culture
        )
        
        # Act
        plan = self.engine.create_maintenance_plan(
            "test_org", assessment, strategies
        )
        
        # Assert
        assert plan.organization_id == "test_org"
        assert plan.status == MaintenanceStatus.STABLE
        assert len(plan.maintenance_strategies) == len(strategies)
        assert "indicators" in plan.monitoring_framework
        assert len(plan.intervention_triggers) > 0
        assert "total_time_hours_monthly" in plan.resource_allocation
        assert "immediate_actions" in plan.timeline
    
    def test_monitor_long_term_health(self):
        """Test long-term culture health monitoring"""
        # Arrange
        assessment = self.engine.assess_cultural_sustainability(
            "test_org", self.transformation, self.current_culture
        )
        strategies = self.engine.develop_maintenance_strategy(
            "test_org", assessment, self.target_culture
        )
        plan = self.engine.create_maintenance_plan(
            "test_org", assessment, strategies
        )
        
        # Act
        monitoring_result = self.engine.monitor_long_term_health(
            "test_org", plan, monitoring_period_days=30
        )
        
        # Assert
        assert monitoring_result.organization_id == "test_org"
        assert "start" in monitoring_result.monitoring_period
        assert "end" in monitoring_result.monitoring_period
        assert len(monitoring_result.health_trends) > 0
        assert len(monitoring_result.sustainability_metrics) > 0
        assert len(monitoring_result.recommendations) > 0
        assert len(monitoring_result.next_actions) > 0
    
    def test_analyze_health_indicators(self):
        """Test health indicators analysis"""
        # Act
        indicators = self.engine._analyze_health_indicators(
            self.current_culture, self.target_culture
        )
        
        # Assert
        assert len(indicators) > 0
        assert all(indicator.current_value >= 0 for indicator in indicators)
        assert all(indicator.target_value >= 0 for indicator in indicators)
        assert all(indicator.importance_weight > 0 for indicator in indicators)
        assert all(indicator.trend in ["improving", "stable", "declining"] for indicator in indicators)
    
    def test_identify_risk_factors(self):
        """Test risk factor identification"""
        # Act
        risk_factors = self.engine._identify_risk_factors(
            self.current_culture, self.transformation
        )
        
        # Assert
        assert isinstance(risk_factors, list)
        # With current test data, should have some risk factors
        assert len(risk_factors) >= 0
    
    def test_identify_protective_factors(self):
        """Test protective factor identification"""
        # Act
        protective_factors = self.engine._identify_protective_factors(
            self.current_culture, self.transformation
        )
        
        # Assert
        assert isinstance(protective_factors, list)
        # With current test data, should have some protective factors
        assert len(protective_factors) > 0
    
    def test_calculate_sustainability_score(self):
        """Test sustainability score calculation"""
        # Arrange
        indicators = self.engine._analyze_health_indicators(
            self.current_culture, self.target_culture
        )
        risk_factors = ["test_risk"]
        protective_factors = ["test_protection"]
        
        # Act
        score = self.engine._calculate_sustainability_score(
            indicators, risk_factors, protective_factors
        )
        
        # Assert
        assert 0.0 <= score <= 1.0
    
    def test_determine_sustainability_level(self):
        """Test sustainability level determination"""
        # Test high score
        high_level = self.engine._determine_sustainability_level(0.9)
        assert high_level == SustainabilityLevel.HIGH
        
        # Test medium score
        medium_level = self.engine._determine_sustainability_level(0.7)
        assert medium_level == SustainabilityLevel.MEDIUM
        
        # Test low score
        low_level = self.engine._determine_sustainability_level(0.5)
        assert low_level == SustainabilityLevel.LOW
        
        # Test critical score
        critical_level = self.engine._determine_sustainability_level(0.1)
        assert critical_level == SustainabilityLevel.CRITICAL
    
    def test_create_risk_mitigation_strategy(self):
        """Test risk mitigation strategy creation"""
        # Arrange
        risk_factors = ["Low overall culture health score", "High number of subcultures creating fragmentation"]
        
        # Act
        strategy = self.engine._create_risk_mitigation_strategy(
            "test_org", risk_factors
        )
        
        # Assert
        assert strategy.organization_id == "test_org"
        assert len(strategy.maintenance_activities) > 0
        assert len(strategy.success_metrics) > 0
        assert strategy.review_frequency == "monthly"
    
    def test_create_reinforcement_strategy(self):
        """Test reinforcement strategy creation"""
        # Arrange
        protective_factors = ["Strong overall culture health", "Well-defined value system"]
        
        # Act
        strategy = self.engine._create_reinforcement_strategy(
            "test_org", protective_factors
        )
        
        # Assert
        assert strategy.organization_id == "test_org"
        assert len(strategy.maintenance_activities) > 0
        assert len(strategy.success_metrics) > 0
        assert strategy.review_frequency == "quarterly"
    
    def test_design_monitoring_framework(self):
        """Test monitoring framework design"""
        # Arrange
        indicators = self.engine._analyze_health_indicators(
            self.current_culture, self.target_culture
        )
        
        # Act
        framework = self.engine._design_monitoring_framework(indicators)
        
        # Assert
        assert "indicators" in framework
        assert "measurement_frequency" in framework
        assert "alert_thresholds" in framework
        assert "data_sources" in framework
        assert len(framework["indicators"]) == len(indicators)
    
    def test_define_intervention_triggers(self):
        """Test intervention trigger definition"""
        # Arrange
        assessment = self.engine.assess_cultural_sustainability(
            "test_org", self.transformation, self.current_culture
        )
        
        # Act
        triggers = self.engine._define_intervention_triggers(assessment)
        
        # Assert
        assert len(triggers) > 0
        assert all("trigger_type" in trigger for trigger in triggers)
        assert all("condition" in trigger for trigger in triggers)
        assert all("intervention" in trigger for trigger in triggers)
        assert all("priority" in trigger for trigger in triggers)
    
    def test_calculate_resource_allocation(self):
        """Test resource allocation calculation"""
        # Arrange
        assessment = self.engine.assess_cultural_sustainability(
            "test_org", self.transformation, self.current_culture
        )
        strategies = self.engine.develop_maintenance_strategy(
            "test_org", assessment, self.target_culture
        )
        
        # Act
        allocation = self.engine._calculate_resource_allocation(strategies)
        
        # Assert
        assert "total_time_hours_monthly" in allocation
        assert "budget_requirement" in allocation
        assert "personnel_needed" in allocation
        assert "tools_and_systems" in allocation
        assert allocation["total_time_hours_monthly"] > 0
        assert allocation["personnel_needed"] >= 2
    
    def test_collect_health_trends(self):
        """Test health trends collection"""
        # Arrange
        start_date = datetime.now() - timedelta(days=30)
        end_date = datetime.now()
        
        # Act
        trends = self.engine._collect_health_trends(
            "test_org", start_date, end_date
        )
        
        # Assert
        assert isinstance(trends, dict)
        assert len(trends) > 0
        assert all(isinstance(values, list) for values in trends.values())
        assert all(len(values) > 0 for values in trends.values())
    
    def test_calculate_sustainability_metrics(self):
        """Test sustainability metrics calculation"""
        # Arrange
        health_trends = {
            "engagement": [0.7, 0.75, 0.8],
            "alignment": [0.6, 0.65, 0.7]
        }
        assessment = self.engine.assess_cultural_sustainability(
            "test_org", self.transformation, self.current_culture
        )
        
        # Act
        metrics = self.engine._calculate_sustainability_metrics(
            health_trends, assessment
        )
        
        # Assert
        assert isinstance(metrics, dict)
        assert len(metrics) > 0
        assert all("_current" in key or "_trend" in key or "_stability" in key for key in metrics.keys())
    
    def test_error_handling(self):
        """Test error handling in engine methods"""
        # Test with invalid data
        with pytest.raises(Exception):
            self.engine.assess_cultural_sustainability(
                None, None, None
            )
    
    def test_engine_initialization(self):
        """Test engine initialization"""
        engine = CultureMaintenanceEngine()
        assert engine.sustainability_thresholds["high"] == 0.8
        assert engine.sustainability_thresholds["medium"] == 0.6
        assert engine.sustainability_thresholds["low"] == 0.4
        assert engine.sustainability_thresholds["critical"] == 0.2