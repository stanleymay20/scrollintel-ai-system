"""Bias Analysis Engine for detecting and quantifying bias in datasets."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from scipy import stats
from sklearn.metrics import confusion_matrix
import logging

from ..models.base_models import (
    BiasReport, FairnessViolation, MitigationStrategy, BiasType
)
from ..core.exceptions import BiasDetectionError, InsufficientDataError


logger = logging.getLogger(__name__)


@dataclass
class FairnessMetrics:
    """Container for fairness metrics."""
    demographic_parity: float
    equalized_odds: float
    statistical_parity: float
    individual_fairness: float
    disparate_impact: float
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            'demographic_parity': self.demographic_parity,
            'equalized_odds': self.equalized_odds,
            'statistical_parity': self.statistical_parity,
            'individual_fairness': self.individual_fairness,
            'disparate_impact': self.disparate_impact
        }


@dataclass
class ProtectedAttribute:
    """Information about a protected attribute."""
    name: str
    values: List[str]
    is_binary: bool
    privileged_group: Optional[str] = None
    unprivileged_group: Optional[str] = None


class BiasAnalysisEngine:
    """Engine for detecting and analyzing bias in datasets."""
    
    def __init__(self, fairness_thresholds: Optional[Dict[str, float]] = None):
        """
        Initialize the bias analysis engine.
        
        Args:
            fairness_thresholds: Thresholds for fairness metrics
        """
        self.fairness_thresholds = fairness_thresholds or {
            'demographic_parity': 0.1,
            'equalized_odds': 0.1,
            'statistical_parity': 0.1,
            'individual_fairness': 0.1,
            'disparate_impact': 0.8
        }
        
        # Common protected attributes
        self.common_protected_attributes = [
            'gender', 'race', 'ethnicity', 'age', 'religion', 
            'nationality', 'sexual_orientation', 'disability',
            'marital_status', 'income_level', 'education_level'
        ]
    
    def detect_bias(self, dataset_id: str, data: pd.DataFrame, 
                   protected_attributes: List[str],
                   target_column: Optional[str] = None) -> BiasReport:
        """
        Detect bias in the dataset.
        
        Args:
            dataset_id: Unique identifier for the dataset
            data: The dataset to analyze
            protected_attributes: List of protected attribute column names
            target_column: Target variable column name (if available)
            
        Returns:
            BiasReport containing bias analysis results
        """
        try:
            logger.info(f"Starting bias detection for dataset {dataset_id}")
            
            if data.empty:
                raise InsufficientDataError("Dataset is empty")
            
            # Validate protected attributes exist in data
            missing_attrs = [attr for attr in protected_attributes if attr not in data.columns]
            if missing_attrs:
                raise BiasDetectionError(f"Protected attributes not found in data: {missing_attrs}")
            
            # Identify protected attributes automatically if not provided
            if not protected_attributes:
                protected_attributes = self._identify_protected_attributes(data)
            
            # Calculate bias metrics
            bias_metrics = {}
            fairness_violations = []
            
            for attr in protected_attributes:
                try:
                    # Calculate fairness metrics for this attribute
                    metrics = self._calculate_fairness_metrics(data, attr, target_column)
                    bias_metrics[attr] = metrics.to_dict()
                    
                    # Check for violations
                    violations = self._check_fairness_violations(attr, metrics)
                    fairness_violations.extend(violations)
                    
                except Exception as e:
                    logger.warning(f"Failed to analyze attribute {attr}: {str(e)}")
                    continue
            
            # Generate mitigation strategies
            mitigation_strategies = self._generate_mitigation_strategies(fairness_violations)
            
            return BiasReport(
                dataset_id=dataset_id,
                protected_attributes=protected_attributes,
                bias_metrics=bias_metrics,
                fairness_violations=fairness_violations,
                mitigation_strategies=mitigation_strategies
            )
            
        except Exception as e:
            logger.error(f"Bias detection failed for dataset {dataset_id}: {str(e)}")
            raise BiasDetectionError(f"Bias detection failed: {str(e)}")
    
    def _identify_protected_attributes(self, data: pd.DataFrame) -> List[str]:
        """
        Automatically identify potential protected attributes in the dataset.
        
        Args:
            data: The dataset to analyze
            
        Returns:
            List of potential protected attribute column names
        """
        identified_attrs = []
        
        for col in data.columns:
            col_lower = col.lower()
            
            # Check if column name matches common protected attributes
            for protected_attr in self.common_protected_attributes:
                if protected_attr in col_lower or col_lower in protected_attr:
                    identified_attrs.append(col)
                    break
            
            # Check for binary categorical variables that might be protected
            if data[col].dtype == 'object' or data[col].dtype.name == 'category':
                unique_values = data[col].nunique()
                if 2 <= unique_values <= 10:  # Reasonable range for protected attributes
                    # Check if values suggest protected attributes
                    values = data[col].unique()
                    value_strings = [str(v).lower() for v in values if pd.notna(v)]
                    
                    # Gender indicators
                    if any(gender in ' '.join(value_strings) for gender in ['male', 'female', 'm', 'f']):
                        if col not in identified_attrs:
                            identified_attrs.append(col)
                    
                    # Race/ethnicity indicators
                    if any(race in ' '.join(value_strings) for race in ['white', 'black', 'asian', 'hispanic', 'latino']):
                        if col not in identified_attrs:
                            identified_attrs.append(col)
        
        logger.info(f"Identified potential protected attributes: {identified_attrs}")
        return identified_attrs
    
    def _calculate_fairness_metrics(self, data: pd.DataFrame, 
                                  protected_attr: str,
                                  target_column: Optional[str] = None) -> FairnessMetrics:
        """
        Calculate fairness metrics for a protected attribute.
        
        Args:
            data: The dataset
            protected_attr: Protected attribute column name
            target_column: Target variable column name
            
        Returns:
            FairnessMetrics object
        """
        # Get protected attribute information
        protected_info = self._analyze_protected_attribute(data, protected_attr)
        
        # Calculate demographic parity
        demographic_parity = self._calculate_demographic_parity(data, protected_attr)
        
        # Calculate statistical parity (similar to demographic parity for representation)
        statistical_parity = self._calculate_statistical_parity(data, protected_attr)
        
        # Calculate disparate impact
        disparate_impact = self._calculate_disparate_impact(data, protected_attr, target_column)
        
        # Calculate equalized odds (requires target variable)
        equalized_odds = 0.0
        if target_column and target_column in data.columns:
            equalized_odds = self._calculate_equalized_odds(data, protected_attr, target_column)
        
        # Calculate individual fairness (simplified version)
        individual_fairness = self._calculate_individual_fairness(data, protected_attr)
        
        return FairnessMetrics(
            demographic_parity=demographic_parity,
            equalized_odds=equalized_odds,
            statistical_parity=statistical_parity,
            individual_fairness=individual_fairness,
            disparate_impact=disparate_impact
        )
    
    def _analyze_protected_attribute(self, data: pd.DataFrame, attr: str) -> ProtectedAttribute:
        """Analyze a protected attribute to understand its characteristics."""
        values = data[attr].dropna().unique().tolist()
        is_binary = len(values) == 2
        
        # For binary attributes, try to identify privileged/unprivileged groups
        privileged_group = None
        unprivileged_group = None
        
        if is_binary:
            value_strings = [str(v).lower() for v in values]
            
            # Common privileged group indicators
            privileged_indicators = ['male', 'white', 'majority', 'high', 'yes', '1', 'true']
            unprivileged_indicators = ['female', 'minority', 'low', 'no', '0', 'false']
            
            for i, val_str in enumerate(value_strings):
                if any(indicator in val_str for indicator in privileged_indicators):
                    privileged_group = values[i]
                elif any(indicator in val_str for indicator in unprivileged_indicators):
                    unprivileged_group = values[i]
        
        return ProtectedAttribute(
            name=attr,
            values=values,
            is_binary=is_binary,
            privileged_group=privileged_group,
            unprivileged_group=unprivileged_group
        )
    
    def _calculate_demographic_parity(self, data: pd.DataFrame, protected_attr: str) -> float:
        """Calculate demographic parity difference."""
        try:
            # Calculate representation of each group
            group_counts = data[protected_attr].value_counts()
            total_count = len(data)
            
            if len(group_counts) < 2:
                return 0.0
            
            # Calculate proportions
            proportions = group_counts / total_count
            
            # Demographic parity difference (max - min proportion)
            return float(proportions.max() - proportions.min())
            
        except Exception as e:
            logger.warning(f"Failed to calculate demographic parity: {str(e)}")
            return 0.0
    
    def _calculate_statistical_parity(self, data: pd.DataFrame, protected_attr: str) -> float:
        """Calculate statistical parity difference."""
        # For this implementation, statistical parity is similar to demographic parity
        # In practice, this would measure outcome differences across groups
        return self._calculate_demographic_parity(data, protected_attr)
    
    def _calculate_disparate_impact(self, data: pd.DataFrame, 
                                  protected_attr: str,
                                  target_column: Optional[str] = None) -> float:
        """Calculate disparate impact ratio."""
        try:
            if not target_column or target_column not in data.columns:
                # Without target, use representation ratios
                group_counts = data[protected_attr].value_counts()
                if len(group_counts) < 2:
                    return 1.0
                
                # Calculate ratio of smallest to largest group
                return float(group_counts.min() / group_counts.max())
            
            # With target variable, calculate outcome ratios
            grouped = data.groupby(protected_attr)[target_column].mean()
            if len(grouped) < 2:
                return 1.0
            
            # Disparate impact ratio (min rate / max rate)
            return float(grouped.min() / grouped.max()) if grouped.max() > 0 else 1.0
            
        except Exception as e:
            logger.warning(f"Failed to calculate disparate impact: {str(e)}")
            return 1.0
    
    def _calculate_equalized_odds(self, data: pd.DataFrame, 
                                protected_attr: str, 
                                target_column: str) -> float:
        """Calculate equalized odds difference."""
        try:
            # This is a simplified version - in practice would need predicted outcomes
            # For now, calculate difference in target variable means across groups
            grouped_means = data.groupby(protected_attr)[target_column].mean()
            
            if len(grouped_means) < 2:
                return 0.0
            
            return float(grouped_means.max() - grouped_means.min())
            
        except Exception as e:
            logger.warning(f"Failed to calculate equalized odds: {str(e)}")
            return 0.0
    
    def _calculate_individual_fairness(self, data: pd.DataFrame, protected_attr: str) -> float:
        """Calculate individual fairness metric (simplified)."""
        try:
            # Simplified individual fairness based on within-group variance
            grouped_var = data.groupby(protected_attr).var().mean(axis=1)
            
            if len(grouped_var) < 2:
                return 0.0
            
            # Higher variance difference indicates less individual fairness
            return float(grouped_var.max() - grouped_var.min())
            
        except Exception as e:
            logger.warning(f"Failed to calculate individual fairness: {str(e)}")
            return 0.0
    
    def _check_fairness_violations(self, attr: str, metrics: FairnessMetrics) -> List[FairnessViolation]:
        """Check for fairness violations based on thresholds."""
        violations = []
        
        # Check demographic parity
        if metrics.demographic_parity > self.fairness_thresholds['demographic_parity']:
            violations.append(FairnessViolation(
                bias_type=BiasType.DEMOGRAPHIC_PARITY,
                protected_attribute=attr,
                severity="high" if metrics.demographic_parity > 0.2 else "medium",
                description=f"Demographic parity violation: {metrics.demographic_parity:.3f} > {self.fairness_thresholds['demographic_parity']}",
                metric_value=metrics.demographic_parity,
                threshold=self.fairness_thresholds['demographic_parity'],
                affected_groups=["all"]
            ))
        
        # Check equalized odds
        if metrics.equalized_odds > self.fairness_thresholds['equalized_odds']:
            violations.append(FairnessViolation(
                bias_type=BiasType.EQUALIZED_ODDS,
                protected_attribute=attr,
                severity="high" if metrics.equalized_odds > 0.2 else "medium",
                description=f"Equalized odds violation: {metrics.equalized_odds:.3f} > {self.fairness_thresholds['equalized_odds']}",
                metric_value=metrics.equalized_odds,
                threshold=self.fairness_thresholds['equalized_odds'],
                affected_groups=["all"]
            ))
        
        # Check disparate impact
        if metrics.disparate_impact < self.fairness_thresholds['disparate_impact']:
            violations.append(FairnessViolation(
                bias_type=BiasType.STATISTICAL_PARITY,
                protected_attribute=attr,
                severity="high" if metrics.disparate_impact < 0.6 else "medium",
                description=f"Disparate impact violation: {metrics.disparate_impact:.3f} < {self.fairness_thresholds['disparate_impact']}",
                metric_value=metrics.disparate_impact,
                threshold=self.fairness_thresholds['disparate_impact'],
                affected_groups=["minority"]
            ))
        
        return violations
    
    def _generate_mitigation_strategies(self, violations: List[FairnessViolation]) -> List[MitigationStrategy]:
        """Generate mitigation strategies for detected violations."""
        strategies = []
        
        # Group violations by type
        violation_types = {}
        for violation in violations:
            if violation.bias_type not in violation_types:
                violation_types[violation.bias_type] = []
            violation_types[violation.bias_type].append(violation)
        
        # Generate strategies for each violation type
        for bias_type, type_violations in violation_types.items():
            if bias_type == BiasType.DEMOGRAPHIC_PARITY:
                strategies.append(MitigationStrategy(
                    strategy_type="data_balancing",
                    description="Balance representation across protected groups through sampling techniques",
                    implementation_steps=[
                        "Identify underrepresented groups",
                        "Apply oversampling techniques (SMOTE, ADASYN)",
                        "Consider undersampling majority groups if appropriate",
                        "Validate balanced representation"
                    ],
                    expected_impact=0.7,
                    complexity="medium"
                ))
            
            elif bias_type == BiasType.EQUALIZED_ODDS:
                strategies.append(MitigationStrategy(
                    strategy_type="algorithmic_fairness",
                    description="Apply fairness constraints during model training",
                    implementation_steps=[
                        "Implement fairness-aware algorithms",
                        "Add fairness constraints to optimization",
                        "Use post-processing fairness techniques",
                        "Monitor fairness metrics during training"
                    ],
                    expected_impact=0.8,
                    complexity="high"
                ))
            
            elif bias_type == BiasType.STATISTICAL_PARITY:
                strategies.append(MitigationStrategy(
                    strategy_type="preprocessing",
                    description="Preprocess data to reduce statistical disparities",
                    implementation_steps=[
                        "Apply disparate impact remover",
                        "Use reweighting techniques",
                        "Transform features to reduce correlation with protected attributes",
                        "Validate statistical parity improvements"
                    ],
                    expected_impact=0.6,
                    complexity="medium"
                ))
        
        # Add general strategies
        if violations:
            strategies.append(MitigationStrategy(
                strategy_type="data_collection",
                description="Collect additional data to improve representation",
                implementation_steps=[
                    "Identify data gaps for underrepresented groups",
                    "Design targeted data collection strategies",
                    "Ensure data quality across all groups",
                    "Validate improved representation"
                ],
                expected_impact=0.9,
                complexity="high"
            ))
        
        return strategies
    
    def calculate_fairness_metrics(self, dataset_id: str, 
                                 data: pd.DataFrame,
                                 protected_attributes: List[str],
                                 target_column: str) -> Dict[str, FairnessMetrics]:
        """
        Calculate fairness metrics for multiple protected attributes.
        
        Args:
            dataset_id: Dataset identifier
            data: The dataset
            protected_attributes: List of protected attributes
            target_column: Target variable column
            
        Returns:
            Dictionary mapping attribute names to fairness metrics
        """
        try:
            metrics_dict = {}
            
            for attr in protected_attributes:
                if attr in data.columns:
                    metrics = self._calculate_fairness_metrics(data, attr, target_column)
                    metrics_dict[attr] = metrics
                else:
                    logger.warning(f"Protected attribute {attr} not found in dataset")
            
            return metrics_dict
            
        except Exception as e:
            logger.error(f"Failed to calculate fairness metrics: {str(e)}")
            raise BiasDetectionError(f"Fairness metrics calculation failed: {str(e)}")
    
    def validate_fairness(self, data: pd.DataFrame,
                         protected_attributes: List[str],
                         fairness_constraints: Dict[str, float]) -> bool:
        """
        Validate that data meets fairness constraints.
        
        Args:
            data: The dataset to validate
            protected_attributes: List of protected attributes
            fairness_constraints: Fairness constraints to validate against
            
        Returns:
            True if all constraints are met, False otherwise
        """
        try:
            for attr in protected_attributes:
                if attr not in data.columns:
                    continue
                
                metrics = self._calculate_fairness_metrics(data, attr)
                
                # Check each constraint
                for constraint_name, threshold in fairness_constraints.items():
                    if hasattr(metrics, constraint_name):
                        metric_value = getattr(metrics, constraint_name)
                        
                        # Different constraints have different validation logic
                        if constraint_name == 'disparate_impact':
                            if metric_value < threshold:
                                return False
                        else:
                            if metric_value > threshold:
                                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Fairness validation failed: {str(e)}")
            return False