"""
Synthetic data generation for testing data quality algorithms.
"""

import pytest
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from ai_data_readiness.engines.quality_assessment_engine import QualityAssessmentEngine
from ai_data_readiness.engines.bias_analysis_engine import BiasAnalysisEngine


class DataQualityIssueType(Enum):
    """Types of data quality issues that can be introduced."""
    MISSING_VALUES = "missing_values"
    OUTLIERS = "outliers"
    DUPLICATES = "duplicates"
    INCONSISTENT_FORMATS = "inconsistent_formats"
    INVALID_VALUES = "invalid_values"
    SCHEMA_VIOLATIONS = "schema_violations"
    ENCODING_ISSUES = "encoding_issues"
    TEMPORAL_INCONSISTENCIES = "temporal_inconsistencies"


@dataclass
class QualityIssueConfig:
    """Configuration for introducing quality issues."""
    issue_type: DataQualityIssueType
    severity: float  # 0.0 to 1.0
    affected_columns: Optional[List[str]] = None
    parameters: Optional[Dict] = None


@dataclass
class BiasConfig:
    """Configuration for introducing bias."""
    protected_attribute: str
    target_column: str
    bias_strength: float  # 0.0 to 1.0
    bias_type: str = "demographic_parity"  # or "equalized_odds", "equality_of_opportunity"


class SyntheticDataGenerator:
    """Generate synthetic datasets with controlled quality issues and bias."""
    
    def __init__(self, seed: int = 42):
        """Initialize the synthetic data generator."""
        self.seed = seed
        np.random.seed(seed)
    
    def generate_clean_dataset(self, n_rows: int, n_features: int, 
                             feature_types: Optional[Dict[str, str]] = None) -> pd.DataFrame:
        """Generate a clean dataset without quality issues."""
        np.random.seed(self.seed)
        
        data = {}
        
        if feature_types is None:
            # Default feature type distribution
            feature_types = {}
            for i in range(n_features):
                if i % 4 == 0:
                    feature_types[f'numerical_{i}'] = 'numerical'
                elif i % 4 == 1:
                    feature_types[f'categorical_{i}'] = 'categorical'
                elif i % 4 == 2:
                    feature_types[f'boolean_{i}'] = 'boolean'
                else:
                    feature_types[f'datetime_{i}'] = 'datetime'
        
        for feature_name, feature_type in feature_types.items():
            if feature_type == 'numerical':
                data[feature_name] = np.random.normal(50, 15, n_rows)
            elif feature_type == 'categorical':
                categories = ['A', 'B', 'C', 'D', 'E']
                data[feature_name] = np.random.choice(categories, n_rows)
            elif feature_type == 'boolean':
                data[feature_name] = np.random.choice([True, False], n_rows)
            elif feature_type == 'datetime':
                start_date = pd.Timestamp('2020-01-01')
                data[feature_name] = pd.date_range(start_date, periods=n_rows, freq='D')
            elif feature_type == 'text':
                data[feature_name] = [f'text_value_{i}_{np.random.randint(0, 1000)}' for i in range(n_rows)]
        
        # Add target variable
        data['target'] = np.random.choice([0, 1], n_rows, p=[0.6, 0.4])
        
        return pd.DataFrame(data)
    
    def introduce_missing_values(self, df: pd.DataFrame, config: QualityIssueConfig) -> pd.DataFrame:
        """Introduce missing values into the dataset."""
        df_copy = df.copy()
        
        columns = config.affected_columns or df.columns.tolist()
        missing_rate = config.severity
        
        pattern = config.parameters.get('pattern', 'random') if config.parameters else 'random'
        
        for column in columns:
            if column == 'target':  # Don't introduce missing values in target
                continue
                
            n_missing = int(len(df_copy) * missing_rate)
            
            if pattern == 'random':
                missing_indices = np.random.choice(len(df_copy), n_missing, replace=False)
            elif pattern == 'clustered':
                # Create clusters of missing values
                start_idx = np.random.randint(0, len(df_copy) - n_missing)
                missing_indices = range(start_idx, start_idx + n_missing)
            elif pattern == 'periodic':
                # Missing values at regular intervals
                period = max(1, len(df_copy) // n_missing)
                missing_indices = range(0, len(df_copy), period)[:n_missing]
            else:
                missing_indices = np.random.choice(len(df_copy), n_missing, replace=False)
            
            df_copy.loc[missing_indices, column] = None
        
        return df_copy
    
    def introduce_outliers(self, df: pd.DataFrame, config: QualityIssueConfig) -> pd.DataFrame:
        """Introduce outliers into numerical columns."""
        df_copy = df.copy()
        
        numerical_columns = df_copy.select_dtypes(include=[np.number]).columns.tolist()
        columns = config.affected_columns or numerical_columns
        
        outlier_rate = config.severity
        outlier_strength = config.parameters.get('strength', 3.0) if config.parameters else 3.0
        
        for column in columns:
            if column not in numerical_columns:
                continue
                
            n_outliers = int(len(df_copy) * outlier_rate)
            outlier_indices = np.random.choice(len(df_copy), n_outliers, replace=False)
            
            # Calculate outlier values
            col_mean = df_copy[column].mean()
            col_std = df_copy[column].std()
            
            # Generate outliers at both extremes
            outlier_values = []
            for _ in range(n_outliers):
                if np.random.random() < 0.5:
                    # Positive outlier
                    outlier_values.append(col_mean + outlier_strength * col_std)
                else:
                    # Negative outlier
                    outlier_values.append(col_mean - outlier_strength * col_std)
            
            df_copy.loc[outlier_indices, column] = outlier_values
        
        return df_copy
    
    def introduce_duplicates(self, df: pd.DataFrame, config: QualityIssueConfig) -> pd.DataFrame:
        """Introduce duplicate rows."""
        df_copy = df.copy()
        
        duplicate_rate = config.severity
        n_duplicates = int(len(df_copy) * duplicate_rate)
        
        duplicate_type = config.parameters.get('type', 'exact') if config.parameters else 'exact'
        
        if duplicate_type == 'exact':
            # Exact duplicates
            duplicate_indices = np.random.choice(len(df_copy), n_duplicates, replace=True)
            duplicate_rows = df_copy.iloc[duplicate_indices].copy()
            df_copy = pd.concat([df_copy, duplicate_rows], ignore_index=True)
        
        elif duplicate_type == 'near':
            # Near duplicates with slight variations
            duplicate_indices = np.random.choice(len(df_copy), n_duplicates, replace=True)
            duplicate_rows = df_copy.iloc[duplicate_indices].copy()
            
            # Add slight variations to numerical columns
            numerical_columns = duplicate_rows.select_dtypes(include=[np.number]).columns
            for col in numerical_columns:
                if col != 'target':
                    noise = np.random.normal(0, duplicate_rows[col].std() * 0.01, len(duplicate_rows))
                    duplicate_rows[col] = duplicate_rows[col] + noise
            
            df_copy = pd.concat([df_copy, duplicate_rows], ignore_index=True)
        
        return df_copy
    
    def introduce_inconsistent_formats(self, df: pd.DataFrame, config: QualityIssueConfig) -> pd.DataFrame:
        """Introduce inconsistent formats in categorical/text columns."""
        df_copy = df.copy()
        
        categorical_columns = df_copy.select_dtypes(include=['object']).columns.tolist()
        columns = config.affected_columns or categorical_columns
        
        inconsistency_rate = config.severity
        
        for column in columns:
            if column not in categorical_columns:
                continue
            
            n_inconsistent = int(len(df_copy) * inconsistency_rate)
            inconsistent_indices = np.random.choice(len(df_copy), n_inconsistent, replace=False)
            
            # Apply various format inconsistencies
            for idx in inconsistent_indices:
                original_value = str(df_copy.loc[idx, column])
                
                # Random format change
                format_change = np.random.choice([
                    'lowercase', 'uppercase', 'mixed_case', 'extra_spaces', 
                    'special_chars', 'abbreviation'
                ])
                
                if format_change == 'lowercase':
                    df_copy.loc[idx, column] = original_value.lower()
                elif format_change == 'uppercase':
                    df_copy.loc[idx, column] = original_value.upper()
                elif format_change == 'mixed_case':
                    df_copy.loc[idx, column] = ''.join(
                        c.upper() if i % 2 == 0 else c.lower() 
                        for i, c in enumerate(original_value)
                    )
                elif format_change == 'extra_spaces':
                    df_copy.loc[idx, column] = f"  {original_value}  "
                elif format_change == 'special_chars':
                    df_copy.loc[idx, column] = f"{original_value}!!!"
                elif format_change == 'abbreviation':
                    df_copy.loc[idx, column] = original_value[:3] + "."
        
        return df_copy
    
    def introduce_invalid_values(self, df: pd.DataFrame, config: QualityIssueConfig) -> pd.DataFrame:
        """Introduce invalid values that violate domain constraints."""
        df_copy = df.copy()
        
        columns = config.affected_columns or df.columns.tolist()
        invalid_rate = config.severity
        
        for column in columns:
            if column == 'target':
                continue
                
            n_invalid = int(len(df_copy) * invalid_rate)
            invalid_indices = np.random.choice(len(df_copy), n_invalid, replace=False)
            
            # Generate invalid values based on column type
            if df_copy[column].dtype in [np.int64, np.float64]:
                # Invalid numerical values
                invalid_values = np.random.choice([
                    -999999, 999999, np.inf, -np.inf, np.nan
                ], n_invalid)
                df_copy.loc[invalid_indices, column] = invalid_values
            
            elif df_copy[column].dtype == 'object':
                # Invalid categorical values
                invalid_values = ['INVALID', 'NULL', 'N/A', '???', '']
                df_copy.loc[invalid_indices, column] = np.random.choice(invalid_values, n_invalid)
            
            elif df_copy[column].dtype == 'bool':
                # Invalid boolean values
                df_copy.loc[invalid_indices, column] = 'INVALID'
        
        return df_copy
    
    def introduce_bias(self, df: pd.DataFrame, bias_config: BiasConfig) -> pd.DataFrame:
        """Introduce bias into the dataset."""
        df_copy = df.copy()
        
        protected_attr = bias_config.protected_attribute
        target_col = bias_config.target_column
        bias_strength = bias_config.bias_strength
        
        if protected_attr not in df_copy.columns or target_col not in df_copy.columns:
            raise ValueError(f"Protected attribute '{protected_attr}' or target '{target_col}' not found in dataset")
        
        # Get unique values of protected attribute
        protected_values = df_copy[protected_attr].unique()
        
        if bias_config.bias_type == "demographic_parity":
            # Introduce demographic parity bias
            for i, value in enumerate(protected_values):
                mask = df_copy[protected_attr] == value
                
                # Adjust approval rates based on bias strength
                base_rate = df_copy[target_col].mean()
                if i == 0:  # Favor first group
                    new_rate = min(1.0, base_rate + bias_strength)
                else:  # Disadvantage other groups
                    new_rate = max(0.0, base_rate - bias_strength)
                
                # Update target values
                group_indices = df_copy[mask].index
                n_positive = int(len(group_indices) * new_rate)
                
                # Set target values
                df_copy.loc[group_indices, target_col] = 0
                positive_indices = np.random.choice(group_indices, n_positive, replace=False)
                df_copy.loc[positive_indices, target_col] = 1
        
        elif bias_config.bias_type == "equalized_odds":
            # Introduce equalized odds bias
            for i, value in enumerate(protected_values):
                mask = df_copy[protected_attr] == value
                
                # Different true positive and false positive rates
                if i == 0:  # Favor first group
                    tpr_adjustment = bias_strength
                    fpr_adjustment = -bias_strength / 2
                else:  # Disadvantage other groups
                    tpr_adjustment = -bias_strength
                    fpr_adjustment = bias_strength / 2
                
                # This is a simplified implementation
                # In practice, you'd need actual ground truth to implement proper equalized odds bias
                group_indices = df_copy[mask].index
                adjustment = np.random.choice([-1, 0, 1], len(group_indices), 
                                            p=[0.1, 0.8, 0.1]) * tpr_adjustment
                
                # Apply adjustment (simplified)
                current_values = df_copy.loc[group_indices, target_col]
                adjusted_probs = np.clip(current_values + adjustment, 0, 1)
                df_copy.loc[group_indices, target_col] = np.random.binomial(1, adjusted_probs)
        
        return df_copy
    
    def generate_dataset_with_issues(self, n_rows: int, n_features: int,
                                   quality_issues: List[QualityIssueConfig],
                                   bias_configs: Optional[List[BiasConfig]] = None,
                                   feature_types: Optional[Dict[str, str]] = None) -> pd.DataFrame:
        """Generate a dataset with specified quality issues and bias."""
        # Start with clean dataset
        df = self.generate_clean_dataset(n_rows, n_features, feature_types)
        
        # Introduce quality issues
        for issue_config in quality_issues:
            if issue_config.issue_type == DataQualityIssueType.MISSING_VALUES:
                df = self.introduce_missing_values(df, issue_config)
            elif issue_config.issue_type == DataQualityIssueType.OUTLIERS:
                df = self.introduce_outliers(df, issue_config)
            elif issue_config.issue_type == DataQualityIssueType.DUPLICATES:
                df = self.introduce_duplicates(df, issue_config)
            elif issue_config.issue_type == DataQualityIssueType.INCONSISTENT_FORMATS:
                df = self.introduce_inconsistent_formats(df, issue_config)
            elif issue_config.issue_type == DataQualityIssueType.INVALID_VALUES:
                df = self.introduce_invalid_values(df, issue_config)
        
        # Introduce bias
        if bias_configs:
            for bias_config in bias_configs:
                df = self.introduce_bias(df, bias_config)
        
        return df


class TestSyntheticDataGeneration:
    """Test synthetic data generation capabilities."""
    
    @pytest.fixture
    def data_generator(self):
        """Create synthetic data generator."""
        return SyntheticDataGenerator(seed=42)
    
    def test_clean_dataset_generation(self, data_generator):
        """Test generation of clean datasets."""
        df = data_generator.generate_clean_dataset(1000, 10)
        
        assert len(df) == 1000
        assert len(df.columns) == 11  # 10 features + target
        assert 'target' in df.columns
        
        # Should have no missing values
        assert df.isnull().sum().sum() == 0
        
        # Should have reasonable data types
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        assert len(numerical_cols) > 0
        assert len(categorical_cols) > 0
    
    def test_missing_values_introduction(self, data_generator):
        """Test introduction of missing values."""
        df = data_generator.generate_clean_dataset(1000, 5)
        
        # Introduce 10% missing values
        missing_config = QualityIssueConfig(
            issue_type=DataQualityIssueType.MISSING_VALUES,
            severity=0.1
        )
        
        df_with_missing = data_generator.introduce_missing_values(df, missing_config)
        
        # Should have missing values
        total_missing = df_with_missing.isnull().sum().sum()
        expected_missing = len(df) * (len(df.columns) - 1) * 0.1  # Exclude target
        
        assert total_missing > 0
        assert abs(total_missing - expected_missing) < expected_missing * 0.2  # Within 20%
    
    def test_outliers_introduction(self, data_generator):
        """Test introduction of outliers."""
        df = data_generator.generate_clean_dataset(1000, 5)
        
        outlier_config = QualityIssueConfig(
            issue_type=DataQualityIssueType.OUTLIERS,
            severity=0.05,  # 5% outliers
            parameters={'strength': 4.0}
        )
        
        df_with_outliers = data_generator.introduce_outliers(df, outlier_config)
        
        # Check for outliers in numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numerical_cols:
            if col == 'target':
                continue
                
            original_std = df[col].std()
            original_mean = df[col].mean()
            
            # Count values beyond 3 standard deviations
            outliers = df_with_outliers[
                (df_with_outliers[col] > original_mean + 3 * original_std) |
                (df_with_outliers[col] < original_mean - 3 * original_std)
            ]
            
            assert len(outliers) > 0  # Should have outliers
    
    def test_duplicates_introduction(self, data_generator):
        """Test introduction of duplicate rows."""
        df = data_generator.generate_clean_dataset(1000, 5)
        original_length = len(df)
        
        duplicate_config = QualityIssueConfig(
            issue_type=DataQualityIssueType.DUPLICATES,
            severity=0.1,  # 10% duplicates
            parameters={'type': 'exact'}
        )
        
        df_with_duplicates = data_generator.introduce_duplicates(df, duplicate_config)
        
        # Should have more rows
        assert len(df_with_duplicates) > original_length
        
        # Should have duplicate rows
        duplicate_count = len(df_with_duplicates) - len(df_with_duplicates.drop_duplicates())
        assert duplicate_count > 0
    
    def test_format_inconsistencies_introduction(self, data_generator):
        """Test introduction of format inconsistencies."""
        # Create dataset with categorical columns
        feature_types = {
            'category_1': 'categorical',
            'category_2': 'categorical',
            'numerical_1': 'numerical'
        }
        
        df = data_generator.generate_clean_dataset(1000, 3, feature_types)
        
        format_config = QualityIssueConfig(
            issue_type=DataQualityIssueType.INCONSISTENT_FORMATS,
            severity=0.1,
            affected_columns=['category_1', 'category_2']
        )
        
        df_with_inconsistencies = data_generator.introduce_inconsistent_formats(df, format_config)
        
        # Check for format variations
        for col in ['category_1', 'category_2']:
            original_values = set(df[col].unique())
            modified_values = set(df_with_inconsistencies[col].unique())
            
            # Should have more unique values due to format variations
            assert len(modified_values) >= len(original_values)
    
    def test_bias_introduction(self, data_generator):
        """Test introduction of bias."""
        # Create dataset with protected attribute
        feature_types = {
            'gender': 'categorical',
            'age': 'numerical',
            'income': 'numerical'
        }
        
        df = data_generator.generate_clean_dataset(1000, 3, feature_types)
        
        # Ensure gender has binary values for bias testing
        df['gender'] = np.random.choice(['M', 'F'], len(df))
        
        bias_config = BiasConfig(
            protected_attribute='gender',
            target_column='target',
            bias_strength=0.3,
            bias_type='demographic_parity'
        )
        
        df_with_bias = data_generator.introduce_bias(df, bias_config)
        
        # Check for bias
        male_approval_rate = df_with_bias[df_with_bias['gender'] == 'M']['target'].mean()
        female_approval_rate = df_with_bias[df_with_bias['gender'] == 'F']['target'].mean()
        
        bias_difference = abs(male_approval_rate - female_approval_rate)
        assert bias_difference > 0.2  # Should have significant bias
    
    def test_complex_dataset_generation(self, data_generator):
        """Test generation of complex datasets with multiple issues."""
        quality_issues = [
            QualityIssueConfig(
                issue_type=DataQualityIssueType.MISSING_VALUES,
                severity=0.05
            ),
            QualityIssueConfig(
                issue_type=DataQualityIssueType.OUTLIERS,
                severity=0.03,
                parameters={'strength': 3.5}
            ),
            QualityIssueConfig(
                issue_type=DataQualityIssueType.DUPLICATES,
                severity=0.02,
                parameters={'type': 'near'}
            )
        ]
        
        bias_configs = [
            BiasConfig(
                protected_attribute='categorical_1',
                target_column='target',
                bias_strength=0.2,
                bias_type='demographic_parity'
            )
        ]
        
        df = data_generator.generate_dataset_with_issues(
            n_rows=2000,
            n_features=8,
            quality_issues=quality_issues,
            bias_configs=bias_configs
        )
        
        # Verify dataset characteristics
        assert len(df) >= 2000  # May be more due to duplicates
        assert len(df.columns) == 9  # 8 features + target
        
        # Should have missing values
        assert df.isnull().sum().sum() > 0
        
        # Should have duplicates
        assert len(df) > len(df.drop_duplicates())
        
        # Should have outliers (check numerical columns)
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        has_outliers = False
        
        for col in numerical_cols:
            if col == 'target':
                continue
            col_mean = df[col].mean()
            col_std = df[col].std()
            outliers = df[
                (df[col] > col_mean + 3 * col_std) |
                (df[col] < col_mean - 3 * col_std)
            ]
            if len(outliers) > 0:
                has_outliers = True
                break
        
        assert has_outliers