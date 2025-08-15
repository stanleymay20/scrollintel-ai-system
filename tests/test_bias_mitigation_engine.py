"""Tests for the Bias Mitigation Engine."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from datetime import datetime

from ai_data_readiness.engines.bias_mitigation_engine import (
    BiasMitigationEngine, FairnessConstraint, MitigationResult,
    DataBalancingStrategy, FeatureEngineeringStrategy, AlgorithmicFairnessStrategy,
    PreprocessingStrategy, DataCollectionStrategy, MitigationTechnique, MitigationComplexity
)
from ai_data_readiness.engines.bias_analysis_engine import BiasAnalysisEngine
from ai_data_readiness.models.base_models import (
    BiasReport, FairnessViolation, MitigationStrategy, BiasType
)
from ai_data_readiness.core.exceptions import BiasDetectionError


class TestFairnessConstraint:
    """Test cases for FairnessConstraint."""
    
    def test_constraint_initialization(self):
        """Test constraint initialization."""
        constraint = FairnessConstraint(
            metric_name="demographic_parity",
            threshold=0.1,
            operator="less_than",
            protected_attribute="gender"
        )
        
        assert constraint.metric_name == "demographic_parity"
        assert constraint.threshold == 0.1
        assert constraint.operator == "less_than"
        assert constraint.protected_attribute == "gender"
        assert constraint.priority == "medium"
    
    def test_constraint_satisfaction_less_than(self):
        """Test constraint satisfaction for less_than operator."""
        constraint = FairnessConstraint(
            metric_name="demographic_parity",
            threshold=0.1,
            operator="less_than",
            protected_attribute="gender"
        )
        
        assert constraint.is_satisfied(0.05) == True
        assert constraint.is_satisfied(0.15) == False
        assert constraint.is_satisfied(0.1) == False
    
    def test_constraint_satisfaction_greater_than(self):
        """Test constraint satisfaction for greater_than operator."""
        constraint = FairnessConstraint(
            metric_name="disparate_impact",
            threshold=0.8,
            operator="greater_than",
            protected_attribute="race"
        )
        
        assert constraint.is_satisfied(0.9) == True
        assert constraint.is_satisfied(0.7) == False
        assert constraint.is_satisfied(0.8) == False
    
    def test_constraint_satisfaction_equal_to(self):
        """Test constraint satisfaction for equal_to operator."""
        constraint = FairnessConstraint(
            metric_name="statistical_parity",
            threshold=0.0,
            operator="equal_to",
            protected_attribute="age"
        )
        
        assert constraint.is_satisfied(0.005) == True  # Within tolerance
        assert constraint.is_satisfied(0.02) == False  # Outside tolerance


class TestMitigationStrategies:
    """Test cases for individual mitigation strategies."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample biased data."""
        np.random.seed(42)
        return pd.DataFrame({
            'gender': ['male'] * 700 + ['female'] * 300,
            'race': ['white'] * 600 + ['black'] * 200 + ['asian'] * 150 + ['hispanic'] * 50,
            'target': np.random.choice([0, 1], 1000, p=[0.6, 0.4])
        })
    
    @pytest.fixture
    def sample_violation(self):
        """Create sample fairness violation."""
        return FairnessViolation(
            bias_type=BiasType.DEMOGRAPHIC_PARITY,
            protected_attribute='gender',
            severity='high',
            description='Demographic parity violation',
            metric_value=0.4,
            threshold=0.1,
            affected_groups=['female']
        )
    
    def test_data_balancing_strategy(self, sample_data, sample_violation):
        """Test data balancing strategy."""
        strategy = DataBalancingStrategy()
        
        # Test can_apply
        assert strategy.can_apply(sample_violation, sample_data) == True
        
        # Test generate_strategy
        mitigation = strategy.generate_strategy(sample_violation, sample_data)
        assert isinstance(mitigation, MitigationStrategy)
        assert mitigation.strategy_type == "data_balancing"
        assert "gender" in mitigation.description
        assert len(mitigation.implementation_steps) > 0
        
        # Test estimate_impact
        impact = strategy.estimate_impact(sample_violation, sample_data)
        assert 0 <= impact <= 1
    
    def test_feature_engineering_strategy(self, sample_data, sample_violation):
        """Test feature engineering strategy."""
        strategy = FeatureEngineeringStrategy()
        
        # Test can_apply
        assert strategy.can_apply(sample_violation, sample_data) == True
        
        # Test generate_strategy
        mitigation = strategy.generate_strategy(sample_violation, sample_data)
        assert isinstance(mitigation, MitigationStrategy)
        assert mitigation.strategy_type == "feature_engineering"
        assert mitigation.complexity == "high"
        
        # Test estimate_impact
        impact = strategy.estimate_impact(sample_violation, sample_data)
        assert impact == 0.7
    
    def test_algorithmic_fairness_strategy(self, sample_data):
        """Test algorithmic fairness strategy."""
        strategy = AlgorithmicFairnessStrategy()
        
        # Test with equalized odds violation
        eq_odds_violation = FairnessViolation(
            bias_type=BiasType.EQUALIZED_ODDS,
            protected_attribute='gender',
            severity='medium',
            description='Equalized odds violation',
            metric_value=0.2,
            threshold=0.1,
            affected_groups=['female']
        )
        
        assert strategy.can_apply(eq_odds_violation, sample_data) == True
        
        mitigation = strategy.generate_strategy(eq_odds_violation, sample_data)
        assert "equalized odds" in mitigation.description
        assert "fairness-aware" in mitigation.implementation_steps[0]
    
    def test_preprocessing_strategy(self, sample_data, sample_violation):
        """Test preprocessing strategy."""
        strategy = PreprocessingStrategy()
        
        assert strategy.can_apply(sample_violation, sample_data) == True
        
        mitigation = strategy.generate_strategy(sample_violation, sample_data)
        assert mitigation.strategy_type == "preprocessing"
        assert "disparate impact remover" in mitigation.implementation_steps[0]
    
    def test_data_collection_strategy(self, sample_data, sample_violation):
        """Test data collection strategy."""
        strategy = DataCollectionStrategy()
        
        assert strategy.can_apply(sample_violation, sample_data) == True
        
        mitigation = strategy.generate_strategy(sample_violation, sample_data)
        assert mitigation.strategy_type == "data_collection"
        assert mitigation.complexity == "high"
        assert mitigation.expected_impact == 0.9


class TestBiasMitigationEngine:
    """Test cases for BiasMitigationEngine."""
    
    @pytest.fixture
    def engine(self):
        """Create BiasMitigationEngine instance."""
        return BiasMitigationEngine()
    
    @pytest.fixture
    def sample_data(self):
        """Create sample biased data."""
        np.random.seed(42)
        return pd.DataFrame({
            'gender': ['male'] * 700 + ['female'] * 300,
            'race': ['white'] * 600 + ['black'] * 200 + ['asian'] * 150 + ['hispanic'] * 50,
            'age': np.random.randint(18, 80, 1000),
            'target': np.random.choice([0, 1], 1000, p=[0.6, 0.4])
        })
    
    @pytest.fixture
    def sample_violations(self):
        """Create sample fairness violations."""
        return [
            FairnessViolation(
                bias_type=BiasType.DEMOGRAPHIC_PARITY,
                protected_attribute='gender',
                severity='high',
                description='Gender demographic parity violation',
                metric_value=0.4,
                threshold=0.1,
                affected_groups=['female']
            ),
            FairnessViolation(
                bias_type=BiasType.EQUALIZED_ODDS,
                protected_attribute='race',
                severity='medium',
                description='Race equalized odds violation',
                metric_value=0.15,
                threshold=0.1,
                affected_groups=['minority']
            )
        ]
    
    @pytest.fixture
    def sample_bias_report(self, sample_violations):
        """Create sample bias report."""
        return BiasReport(
            dataset_id="test_dataset",
            protected_attributes=['gender', 'race'],
            bias_metrics={
                'gender': {'demographic_parity': 0.4, 'disparate_impact': 0.6},
                'race': {'equalized_odds': 0.15, 'disparate_impact': 0.7}
            },
            fairness_violations=sample_violations,
            mitigation_strategies=[]
        )
    
    def test_engine_initialization(self):
        """Test engine initialization."""
        engine = BiasMitigationEngine()
        assert isinstance(engine.bias_engine, BiasAnalysisEngine)
        assert len(engine.strategies) == 5
        
        # Test with custom bias engine
        custom_bias_engine = BiasAnalysisEngine()
        engine = BiasMitigationEngine(custom_bias_engine)
        assert engine.bias_engine is custom_bias_engine
    
    def test_generate_mitigation_strategies(self, engine, sample_violations, sample_data):
        """Test mitigation strategy generation."""
        strategies = engine.generate_mitigation_strategies(sample_violations, sample_data)
        
        assert isinstance(strategies, list)
        assert len(strategies) > 0
        
        # Check that strategies are MitigationStrategy instances
        for strategy in strategies:
            assert isinstance(strategy, MitigationStrategy)
            assert hasattr(strategy, 'strategy_type')
            assert hasattr(strategy, 'expected_impact')
            assert hasattr(strategy, 'complexity')
    
    def test_group_violations(self, engine, sample_violations):
        """Test violation grouping."""
        groups = engine._group_violations(sample_violations)
        
        assert isinstance(groups, dict)
        assert len(groups) == 2  # Two different violation types
        assert 'gender_demographic_parity' in groups
        assert 'race_equalized_odds' in groups
    
    def test_rank_strategies(self, engine):
        """Test strategy ranking."""
        strategies = [
            MitigationStrategy(
                strategy_type="low_impact_high_complexity",
                description="Test strategy 1",
                implementation_steps=["Step 1"],
                expected_impact=0.3,
                complexity="high"
            ),
            MitigationStrategy(
                strategy_type="high_impact_low_complexity",
                description="Test strategy 2",
                implementation_steps=["Step 1"],
                expected_impact=0.8,
                complexity="low"
            ),
            MitigationStrategy(
                strategy_type="medium_impact_medium_complexity",
                description="Test strategy 3",
                implementation_steps=["Step 1"],
                expected_impact=0.6,
                complexity="medium"
            )
        ]
        
        ranked = engine._rank_strategies(strategies)
        
        # High impact + low complexity should be first
        assert ranked[0].strategy_type == "high_impact_low_complexity"
        # Low impact + high complexity should be last
        assert ranked[-1].strategy_type == "low_impact_high_complexity"
    
    def test_validate_fairness_constraints(self, engine, sample_data):
        """Test fairness constraint validation."""
        constraints = [
            FairnessConstraint(
                metric_name="demographic_parity",
                threshold=0.1,
                operator="less_than",
                protected_attribute="gender"
            ),
            FairnessConstraint(
                metric_name="disparate_impact",
                threshold=0.8,
                operator="greater_than",
                protected_attribute="race"
            )
        ]
        
        results = engine.validate_fairness_constraints(sample_data, constraints)
        
        assert isinstance(results, dict)
        assert len(results) == 2
        
        # Results should be boolean values
        for result in results.values():
            assert isinstance(result, bool)
    
    def test_recommend_mitigation_approach(self, engine, sample_bias_report, sample_data):
        """Test comprehensive mitigation approach recommendation."""
        constraints = [
            FairnessConstraint(
                metric_name="demographic_parity",
                threshold=0.1,
                operator="less_than",
                protected_attribute="gender"
            )
        ]
        
        recommendation = engine.recommend_mitigation_approach(
            sample_bias_report, sample_data, constraints
        )
        
        assert isinstance(recommendation, dict)
        assert 'recommended_strategies' in recommendation
        assert 'implementation_roadmap' in recommendation
        assert 'constraint_validation' in recommendation
        assert 'estimated_timeline' in recommendation
        assert 'resource_requirements' in recommendation
        
        # Check recommended strategies
        strategies = recommendation['recommended_strategies']
        assert isinstance(strategies, list)
        assert len(strategies) > 0
        
        # Check implementation roadmap
        roadmap = recommendation['implementation_roadmap']
        assert isinstance(roadmap, list)
        
        # Check constraint validation
        constraint_validation = recommendation['constraint_validation']
        assert isinstance(constraint_validation, dict)
        
        # Check timeline and resources
        assert isinstance(recommendation['estimated_timeline'], str)
        assert isinstance(recommendation['resource_requirements'], dict)
    
    def test_generate_implementation_roadmap(self, engine):
        """Test implementation roadmap generation."""
        strategies = [
            MitigationStrategy(
                strategy_type="quick_win",
                description="Quick win strategy",
                implementation_steps=["Step 1"],
                expected_impact=0.5,
                complexity="low"
            ),
            MitigationStrategy(
                strategy_type="medium_term",
                description="Medium term strategy",
                implementation_steps=["Step 1"],
                expected_impact=0.7,
                complexity="medium"
            ),
            MitigationStrategy(
                strategy_type="long_term",
                description="Long term strategy",
                implementation_steps=["Step 1"],
                expected_impact=0.9,
                complexity="high"
            )
        ]
        
        roadmap = engine._generate_implementation_roadmap(strategies)
        
        assert isinstance(roadmap, list)
        assert len(roadmap) >= 1
        
        # Check roadmap structure
        for phase in roadmap:
            assert 'phase' in phase
            assert 'name' in phase
            assert 'duration' in phase
            assert 'strategies' in phase
            assert 'description' in phase
    
    def test_estimate_timeline(self, engine):
        """Test timeline estimation."""
        # Test with low complexity strategies
        low_strategies = [
            MitigationStrategy("test", "test", ["step"], 0.5, "low")
            for _ in range(2)
        ]
        timeline = engine._estimate_timeline(low_strategies)
        assert timeline == "1 month"
        
        # Test with mixed complexity strategies
        mixed_strategies = [
            MitigationStrategy("test", "test", ["step"], 0.5, "low"),
            MitigationStrategy("test", "test", ["step"], 0.7, "medium"),
            MitigationStrategy("test", "test", ["step"], 0.9, "high")
        ]
        timeline = engine._estimate_timeline(mixed_strategies)
        assert timeline in ["2-3 months", "3-6 months", "6+ months"]
    
    def test_estimate_resources(self, engine):
        """Test resource estimation."""
        strategies = [
            MitigationStrategy("data_collection", "test", ["step"], 0.9, "high"),
            MitigationStrategy("algorithmic_fairness", "test", ["step"], 0.8, "high"),
            MitigationStrategy("preprocessing", "test", ["step"], 0.6, "medium")
        ]
        
        resources = engine._estimate_resources(strategies)
        
        assert isinstance(resources, dict)
        assert 'team_size' in resources
        assert 'skills_required' in resources
        assert 'tools_needed' in resources
        assert 'budget_estimate' in resources
        
        # Should include additional skills for data collection and algorithmic fairness
        assert 'Data Engineering' in resources['skills_required']
        assert 'Fairness-aware ML' in resources['skills_required']
        assert resources['budget_estimate'] == 'High'
    
    def test_prioritize_strategies(self, engine, sample_violations):
        """Test strategy prioritization."""
        strategies = [
            MitigationStrategy("low_priority", "test", ["step"], 0.3, "medium"),
            MitigationStrategy("high_priority", "gender bias", ["step"], 0.7, "low"),
            MitigationStrategy("medium_priority", "test", ["step"], 0.5, "high")
        ]
        
        constraint_results = {"gender constraint": False}  # Failed constraint
        
        prioritized = engine._prioritize_strategies(
            strategies, sample_violations, constraint_results
        )
        
        # Strategy addressing gender (high severity violation) should be prioritized
        assert "gender" in prioritized[0].description
    
    def test_error_handling(self, engine):
        """Test error handling in mitigation engine."""
        # Test with invalid data
        invalid_data = pd.DataFrame()
        violations = []
        
        # Should handle empty violations gracefully
        strategies = engine.generate_mitigation_strategies(violations, invalid_data)
        assert isinstance(strategies, list)
    
    @patch('ai_data_readiness.engines.bias_mitigation_engine.logger')
    def test_logging(self, mock_logger, engine, sample_violations, sample_data):
        """Test that appropriate logging occurs."""
        engine.generate_mitigation_strategies(sample_violations, sample_data)
        
        # Should not have error logs for successful operation
        mock_logger.error.assert_not_called()
    
    def test_constraint_validation_missing_attribute(self, engine, sample_data):
        """Test constraint validation with missing protected attribute."""
        constraints = [
            FairnessConstraint(
                metric_name="demographic_parity",
                threshold=0.1,
                operator="less_than",
                protected_attribute="nonexistent_attribute"
            )
        ]
        
        results = engine.validate_fairness_constraints(sample_data, constraints)
        
        # Should handle missing attribute gracefully
        assert len(results) == 1
        assert list(results.values())[0] == False
    
    def test_constraint_validation_unknown_metric(self, engine, sample_data):
        """Test constraint validation with unknown metric."""
        constraints = [
            FairnessConstraint(
                metric_name="unknown_metric",
                threshold=0.1,
                operator="less_than",
                protected_attribute="gender"
            )
        ]
        
        results = engine.validate_fairness_constraints(sample_data, constraints)
        
        # Should handle unknown metric gracefully
        assert len(results) == 1
        assert list(results.values())[0] == False


if __name__ == "__main__":
    pytest.main([__file__])