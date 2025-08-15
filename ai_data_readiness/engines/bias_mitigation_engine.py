"""Bias Mitigation Recommendation System for AI Data Readiness Platform."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from abc import ABC, abstractmethod

from ..models.base_models import (
    BiasReport, FairnessViolation, MitigationStrategy, BiasType
)
from ..core.exceptions import BiasDetectionError
from .bias_analysis_engine import BiasAnalysisEngine, FairnessMetrics


logger = logging.getLogger(__name__)


class MitigationTechnique(Enum):
    """Types of bias mitigation techniques."""
    PREPROCESSING = "preprocessing"
    IN_PROCESSING = "in_processing"
    POST_PROCESSING = "post_processing"
    DATA_COLLECTION = "data_collection"
    FEATURE_ENGINEERING = "feature_engineering"


class MitigationComplexity(Enum):
    """Complexity levels for mitigation strategies."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class FairnessConstraint:
    """Fairness constraint definition."""
    metric_name: str
    threshold: float
    operator: str  # 'less_than', 'greater_than', 'equal_to'
    protected_attribute: str
    priority: str = "medium"  # low, medium, high
    
    def is_satisfied(self, metric_value: float) -> bool:
        """Check if the constraint is satisfied."""
        if self.operator == 'less_than':
            return metric_value < self.threshold
        elif self.operator == 'greater_than':
            return metric_value > self.threshold
        elif self.operator == 'equal_to':
            return abs(metric_value - self.threshold) < 0.01
        return False


@dataclass
class MitigationResult:
    """Result of applying a mitigation strategy."""
    strategy: MitigationStrategy
    success: bool
    original_metrics: Dict[str, float]
    improved_metrics: Dict[str, float]
    improvement_score: float
    implementation_notes: List[str] = field(default_factory=list)


class BaseMitigationStrategy(ABC):
    """Abstract base class for mitigation strategies."""
    
    def __init__(self, name: str, complexity: MitigationComplexity):
        self.name = name
        self.complexity = complexity
    
    @abstractmethod
    def can_apply(self, violation: FairnessViolation, data: pd.DataFrame) -> bool:
        """Check if this strategy can be applied to the given violation."""
        pass
    
    @abstractmethod
    def generate_strategy(self, violation: FairnessViolation, 
                         data: pd.DataFrame) -> MitigationStrategy:
        """Generate a mitigation strategy for the violation."""
        pass
    
    @abstractmethod
    def estimate_impact(self, violation: FairnessViolation, 
                       data: pd.DataFrame) -> float:
        """Estimate the expected impact of this strategy."""
        pass


class DataBalancingStrategy(BaseMitigationStrategy):
    """Strategy for balancing data representation."""
    
    def __init__(self):
        super().__init__("Data Balancing", MitigationComplexity.MEDIUM)
    
    def can_apply(self, violation: FairnessViolation, data: pd.DataFrame) -> bool:
        """Check if data balancing can address this violation."""
        return violation.bias_type in [
            BiasType.DEMOGRAPHIC_PARITY,
            BiasType.STATISTICAL_PARITY
        ]
    
    def generate_strategy(self, violation: FairnessViolation, 
                         data: pd.DataFrame) -> MitigationStrategy:
        """Generate data balancing strategy."""
        protected_attr = violation.protected_attribute
        
        # Analyze current distribution
        distribution = data[protected_attr].value_counts()
        minority_groups = distribution[distribution < distribution.median()].index.tolist()
        
        implementation_steps = [
            f"Analyze current distribution of {protected_attr}",
            f"Identify underrepresented groups: {minority_groups}",
            "Apply oversampling techniques (SMOTE, ADASYN) for minority groups",
            "Consider undersampling majority groups if dataset is large",
            "Validate balanced representation across all groups",
            "Monitor impact on model performance"
        ]
        
        return MitigationStrategy(
            strategy_type="data_balancing",
            description=f"Balance representation of {protected_attr} through sampling techniques",
            implementation_steps=implementation_steps,
            expected_impact=self.estimate_impact(violation, data),
            complexity=self.complexity.value
        )
    
    def estimate_impact(self, violation: FairnessViolation, 
                       data: pd.DataFrame) -> float:
        """Estimate impact based on current imbalance severity."""
        # Higher imbalance = higher potential impact from balancing
        return min(0.9, 0.5 + (violation.metric_value * 0.4))


class FeatureEngineeringStrategy(BaseMitigationStrategy):
    """Strategy for feature engineering to reduce bias."""
    
    def __init__(self):
        super().__init__("Feature Engineering", MitigationComplexity.HIGH)
    
    def can_apply(self, violation: FairnessViolation, data: pd.DataFrame) -> bool:
        """Check if feature engineering can address this violation."""
        return True  # Feature engineering can help with most bias types
    
    def generate_strategy(self, violation: FairnessViolation, 
                         data: pd.DataFrame) -> MitigationStrategy:
        """Generate feature engineering strategy."""
        protected_attr = violation.protected_attribute
        
        implementation_steps = [
            f"Analyze correlation between {protected_attr} and other features",
            "Identify proxy variables that may encode protected attribute information",
            "Apply feature transformation techniques to reduce correlation",
            "Create fairness-aware feature representations",
            "Use adversarial debiasing techniques if appropriate",
            "Validate that essential predictive power is maintained"
        ]
        
        return MitigationStrategy(
            strategy_type="feature_engineering",
            description=f"Engineer features to reduce bias related to {protected_attr}",
            implementation_steps=implementation_steps,
            expected_impact=self.estimate_impact(violation, data),
            complexity=self.complexity.value
        )
    
    def estimate_impact(self, violation: FairnessViolation, 
                       data: pd.DataFrame) -> float:
        """Estimate impact based on feature correlation."""
        # Simplified estimation - in practice would analyze actual correlations
        return 0.7


class AlgorithmicFairnessStrategy(BaseMitigationStrategy):
    """Strategy for algorithmic fairness constraints."""
    
    def __init__(self):
        super().__init__("Algorithmic Fairness", MitigationComplexity.HIGH)
    
    def can_apply(self, violation: FairnessViolation, data: pd.DataFrame) -> bool:
        """Check if algorithmic fairness can address this violation."""
        return violation.bias_type in [
            BiasType.EQUALIZED_ODDS,
            BiasType.DEMOGRAPHIC_PARITY
        ]
    
    def generate_strategy(self, violation: FairnessViolation, 
                         data: pd.DataFrame) -> MitigationStrategy:
        """Generate algorithmic fairness strategy."""
        bias_type = violation.bias_type
        
        if bias_type == BiasType.EQUALIZED_ODDS:
            focus = "equalized odds constraints"
            techniques = [
                "Implement fairness-aware machine learning algorithms",
                "Add equalized odds constraints to model optimization",
                "Use post-processing calibration techniques",
                "Apply threshold optimization for different groups"
            ]
        else:
            focus = "demographic parity constraints"
            techniques = [
                "Implement demographic parity constraints in model training",
                "Use fairness-aware loss functions",
                "Apply group-specific regularization",
                "Use adversarial fairness techniques"
            ]
        
        implementation_steps = [
            f"Select appropriate fairness-aware algorithm for {focus}",
            *techniques,
            "Monitor fairness metrics during training",
            "Validate fairness improvements on test data"
        ]
        
        return MitigationStrategy(
            strategy_type="algorithmic_fairness",
            description=f"Apply {focus} during model training",
            implementation_steps=implementation_steps,
            expected_impact=self.estimate_impact(violation, data),
            complexity=self.complexity.value
        )
    
    def estimate_impact(self, violation: FairnessViolation, 
                       data: pd.DataFrame) -> float:
        """Estimate impact of algorithmic fairness techniques."""
        return 0.8


class PreprocessingStrategy(BaseMitigationStrategy):
    """Strategy for preprocessing-based bias mitigation."""
    
    def __init__(self):
        super().__init__("Preprocessing", MitigationComplexity.MEDIUM)
    
    def can_apply(self, violation: FairnessViolation, data: pd.DataFrame) -> bool:
        """Check if preprocessing can address this violation."""
        return True  # Preprocessing can help with most bias types
    
    def generate_strategy(self, violation: FairnessViolation, 
                         data: pd.DataFrame) -> MitigationStrategy:
        """Generate preprocessing strategy."""
        protected_attr = violation.protected_attribute
        
        implementation_steps = [
            f"Apply disparate impact remover for {protected_attr}",
            "Use reweighting techniques to balance group representation",
            "Apply data transformation to reduce statistical disparities",
            "Implement fairness-aware data preprocessing pipelines",
            "Validate preprocessing effectiveness on fairness metrics"
        ]
        
        return MitigationStrategy(
            strategy_type="preprocessing",
            description=f"Preprocess data to reduce bias in {protected_attr}",
            implementation_steps=implementation_steps,
            expected_impact=self.estimate_impact(violation, data),
            complexity=self.complexity.value
        )
    
    def estimate_impact(self, violation: FairnessViolation, 
                       data: pd.DataFrame) -> float:
        """Estimate impact of preprocessing techniques."""
        return 0.6


class DataCollectionStrategy(BaseMitigationStrategy):
    """Strategy for collecting additional data."""
    
    def __init__(self):
        super().__init__("Data Collection", MitigationComplexity.HIGH)
    
    def can_apply(self, violation: FairnessViolation, data: pd.DataFrame) -> bool:
        """Check if data collection can address this violation."""
        return True  # More data can always help
    
    def generate_strategy(self, violation: FairnessViolation, 
                         data: pd.DataFrame) -> MitigationStrategy:
        """Generate data collection strategy."""
        protected_attr = violation.protected_attribute
        
        # Identify underrepresented groups
        distribution = data[protected_attr].value_counts()
        underrepresented = distribution[distribution < distribution.median()].index.tolist()
        
        implementation_steps = [
            f"Identify data gaps for {protected_attr}",
            f"Design targeted collection for underrepresented groups: {underrepresented}",
            "Establish partnerships with diverse data sources",
            "Implement quality controls for new data collection",
            "Ensure ethical data collection practices",
            "Validate improved representation and fairness"
        ]
        
        return MitigationStrategy(
            strategy_type="data_collection",
            description=f"Collect additional data to improve {protected_attr} representation",
            implementation_steps=implementation_steps,
            expected_impact=self.estimate_impact(violation, data),
            complexity=self.complexity.value
        )
    
    def estimate_impact(self, violation: FairnessViolation, 
                       data: pd.DataFrame) -> float:
        """Estimate impact of additional data collection."""
        return 0.9  # High impact but high complexity


class BiasMitigationEngine:
    """Engine for generating bias mitigation recommendations."""
    
    def __init__(self, bias_engine: Optional[BiasAnalysisEngine] = None):
        """
        Initialize the bias mitigation engine.
        
        Args:
            bias_engine: BiasAnalysisEngine instance for bias detection
        """
        self.bias_engine = bias_engine or BiasAnalysisEngine()
        
        # Initialize mitigation strategies
        self.strategies = [
            DataBalancingStrategy(),
            FeatureEngineeringStrategy(),
            AlgorithmicFairnessStrategy(),
            PreprocessingStrategy(),
            DataCollectionStrategy()
        ]
    
    def generate_mitigation_strategies(self, 
                                     violations: List[FairnessViolation],
                                     data: pd.DataFrame) -> List[MitigationStrategy]:
        """
        Generate mitigation strategies for detected violations.
        
        Args:
            violations: List of fairness violations
            data: The dataset being analyzed
            
        Returns:
            List of recommended mitigation strategies
        """
        try:
            strategies = []
            
            # Group violations by protected attribute and bias type
            violation_groups = self._group_violations(violations)
            
            for group_key, group_violations in violation_groups.items():
                # Generate strategies for this group
                group_strategies = self._generate_strategies_for_group(
                    group_violations, data
                )
                strategies.extend(group_strategies)
            
            # Rank strategies by expected impact and complexity
            ranked_strategies = self._rank_strategies(strategies)
            
            return ranked_strategies
            
        except Exception as e:
            logger.error(f"Failed to generate mitigation strategies: {str(e)}")
            raise BiasDetectionError(f"Strategy generation failed: {str(e)}")
    
    def _group_violations(self, violations: List[FairnessViolation]) -> Dict[str, List[FairnessViolation]]:
        """Group violations by protected attribute and bias type."""
        groups = {}
        
        for violation in violations:
            key = f"{violation.protected_attribute}_{violation.bias_type.value}"
            if key not in groups:
                groups[key] = []
            groups[key].append(violation)
        
        return groups
    
    def _generate_strategies_for_group(self, 
                                     violations: List[FairnessViolation],
                                     data: pd.DataFrame) -> List[MitigationStrategy]:
        """Generate strategies for a group of related violations."""
        strategies = []
        
        for violation in violations:
            for strategy_generator in self.strategies:
                if strategy_generator.can_apply(violation, data):
                    try:
                        strategy = strategy_generator.generate_strategy(violation, data)
                        strategies.append(strategy)
                    except Exception as e:
                        logger.warning(f"Failed to generate {strategy_generator.name} strategy: {str(e)}")
                        continue
        
        return strategies
    
    def _rank_strategies(self, strategies: List[MitigationStrategy]) -> List[MitigationStrategy]:
        """Rank strategies by expected impact and complexity."""
        def strategy_score(strategy: MitigationStrategy) -> float:
            # Higher impact is better, lower complexity is better
            complexity_weights = {'low': 1.0, 'medium': 0.7, 'high': 0.4}
            complexity_weight = complexity_weights.get(strategy.complexity, 0.5)
            
            return strategy.expected_impact * complexity_weight
        
        return sorted(strategies, key=strategy_score, reverse=True)
    
    def validate_fairness_constraints(self, 
                                    data: pd.DataFrame,
                                    constraints: List[FairnessConstraint]) -> Dict[str, bool]:
        """
        Validate that data meets fairness constraints.
        
        Args:
            data: The dataset to validate
            constraints: List of fairness constraints
            
        Returns:
            Dictionary mapping constraint descriptions to satisfaction status
        """
        try:
            results = {}
            
            for constraint in constraints:
                # Calculate the relevant metric
                if constraint.protected_attribute in data.columns:
                    metrics = self.bias_engine._calculate_fairness_metrics(
                        data, constraint.protected_attribute
                    )
                    
                    if hasattr(metrics, constraint.metric_name):
                        metric_value = getattr(metrics, constraint.metric_name)
                        is_satisfied = constraint.is_satisfied(metric_value)
                        
                        constraint_desc = (
                            f"{constraint.metric_name} for {constraint.protected_attribute} "
                            f"{constraint.operator} {constraint.threshold}"
                        )
                        results[constraint_desc] = is_satisfied
                    else:
                        logger.warning(f"Unknown metric: {constraint.metric_name}")
                        results[f"Unknown metric: {constraint.metric_name}"] = False
                else:
                    logger.warning(f"Protected attribute not found: {constraint.protected_attribute}")
                    results[f"Missing attribute: {constraint.protected_attribute}"] = False
            
            return results
            
        except Exception as e:
            logger.error(f"Fairness constraint validation failed: {str(e)}")
            raise BiasDetectionError(f"Constraint validation failed: {str(e)}")
    
    def recommend_mitigation_approach(self, 
                                    bias_report: BiasReport,
                                    data: pd.DataFrame,
                                    constraints: Optional[List[FairnessConstraint]] = None) -> Dict[str, Any]:
        """
        Recommend comprehensive mitigation approach.
        
        Args:
            bias_report: Bias analysis report
            data: The dataset
            constraints: Optional fairness constraints
            
        Returns:
            Dictionary with comprehensive mitigation recommendations
        """
        try:
            # Generate strategies for violations
            strategies = self.generate_mitigation_strategies(
                bias_report.fairness_violations, data
            )
            
            # Validate constraints if provided
            constraint_results = {}
            if constraints:
                constraint_results = self.validate_fairness_constraints(data, constraints)
            
            # Prioritize strategies based on violations and constraints
            prioritized_strategies = self._prioritize_strategies(
                strategies, bias_report.fairness_violations, constraint_results
            )
            
            # Generate implementation roadmap
            roadmap = self._generate_implementation_roadmap(prioritized_strategies)
            
            return {
                'recommended_strategies': prioritized_strategies,
                'implementation_roadmap': roadmap,
                'constraint_validation': constraint_results,
                'estimated_timeline': self._estimate_timeline(prioritized_strategies),
                'resource_requirements': self._estimate_resources(prioritized_strategies)
            }
            
        except Exception as e:
            logger.error(f"Failed to recommend mitigation approach: {str(e)}")
            raise BiasDetectionError(f"Mitigation recommendation failed: {str(e)}")
    
    def _prioritize_strategies(self, 
                             strategies: List[MitigationStrategy],
                             violations: List[FairnessViolation],
                             constraint_results: Dict[str, bool]) -> List[MitigationStrategy]:
        """Prioritize strategies based on violations and constraints."""
        # Create priority scores based on violation severity and constraint failures
        violation_severity = {v.protected_attribute: v.severity for v in violations}
        failed_constraints = [k for k, v in constraint_results.items() if not v]
        
        def priority_score(strategy: MitigationStrategy) -> float:
            base_score = strategy.expected_impact
            
            # Boost score for high-severity violations
            if any(attr in strategy.description for attr in violation_severity):
                if any(violation_severity[attr] == 'high' for attr in violation_severity):
                    base_score *= 1.5
            
            # Boost score if strategy addresses failed constraints
            if any(constraint in strategy.description for constraint in failed_constraints):
                base_score *= 1.3
            
            return base_score
        
        return sorted(strategies, key=priority_score, reverse=True)
    
    def _generate_implementation_roadmap(self, 
                                       strategies: List[MitigationStrategy]) -> List[Dict[str, Any]]:
        """Generate implementation roadmap for strategies."""
        roadmap = []
        
        # Group strategies by complexity and dependencies
        low_complexity = [s for s in strategies if s.complexity == 'low']
        medium_complexity = [s for s in strategies if s.complexity == 'medium']
        high_complexity = [s for s in strategies if s.complexity == 'high']
        
        # Phase 1: Quick wins (low complexity)
        if low_complexity:
            roadmap.append({
                'phase': 1,
                'name': 'Quick Wins',
                'duration': '2-4 weeks',
                'strategies': low_complexity[:3],  # Top 3 low complexity
                'description': 'Implement low-complexity strategies for immediate impact'
            })
        
        # Phase 2: Medium-term improvements (medium complexity)
        if medium_complexity:
            roadmap.append({
                'phase': 2,
                'name': 'Medium-term Improvements',
                'duration': '1-3 months',
                'strategies': medium_complexity[:2],  # Top 2 medium complexity
                'description': 'Implement preprocessing and data balancing strategies'
            })
        
        # Phase 3: Long-term solutions (high complexity)
        if high_complexity:
            roadmap.append({
                'phase': 3,
                'name': 'Long-term Solutions',
                'duration': '3-6 months',
                'strategies': high_complexity[:2],  # Top 2 high complexity
                'description': 'Implement algorithmic fairness and data collection strategies'
            })
        
        return roadmap
    
    def _estimate_timeline(self, strategies: List[MitigationStrategy]) -> str:
        """Estimate overall implementation timeline."""
        complexity_counts = {
            'low': len([s for s in strategies if s.complexity == 'low']),
            'medium': len([s for s in strategies if s.complexity == 'medium']),
            'high': len([s for s in strategies if s.complexity == 'high'])
        }
        
        # Rough timeline estimation
        weeks = (complexity_counts['low'] * 2 + 
                complexity_counts['medium'] * 8 + 
                complexity_counts['high'] * 20)
        
        if weeks <= 4:
            return "1 month"
        elif weeks <= 12:
            return "2-3 months"
        elif weeks <= 24:
            return "3-6 months"
        else:
            return "6+ months"
    
    def _estimate_resources(self, strategies: List[MitigationStrategy]) -> Dict[str, str]:
        """Estimate resource requirements."""
        has_data_collection = any('data_collection' in s.strategy_type for s in strategies)
        has_algorithmic = any('algorithmic' in s.strategy_type for s in strategies)
        has_preprocessing = any('preprocessing' in s.strategy_type for s in strategies)
        
        resources = {
            'team_size': '2-4 people',
            'skills_required': ['Data Science', 'Machine Learning'],
            'tools_needed': ['Python/R', 'ML Libraries'],
            'budget_estimate': 'Medium'
        }
        
        if has_data_collection:
            resources['skills_required'].append('Data Engineering')
            resources['budget_estimate'] = 'High'
        
        if has_algorithmic:
            resources['skills_required'].append('Fairness-aware ML')
            resources['tools_needed'].append('Fairness Libraries')
        
        if has_preprocessing:
            resources['tools_needed'].append('Data Processing Tools')
        
        return resources