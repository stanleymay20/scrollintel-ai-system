"""AI-specific quality metrics and scoring functionality."""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging
import re
from scipy import stats
from sklearn.ensemble import IsolationForest
import warnings
warnings.filterwarnings('ignore')

from ..models.base_models import (
    Dataset, Schema, AIReadinessScore, DimensionScore, ImprovementArea
)


logger = logging.getLogger(__name__)


class AIQualityMetrics:
    """
    AI-specific quality metrics and scoring functionality.
    
    Implements AI readiness scoring algorithm, feature correlation analysis,
    target leakage detection, and statistical anomaly detection as specified
    in requirements 1.3, 1.4, and 5.1.
    """
    
    def __init__(self):
        """Initialize the AI quality metrics engine."""
        self.logger = logging.getLogger(__name__)
    
    def calculate_ai_readiness_score(self, dataset: Dataset, data: pd.DataFrame, quality_report) -> AIReadinessScore:
        """
        Calculate comprehensive AI readiness score with AI-specific metrics.
        
        Implements AI readiness scoring algorithm as specified in requirements 1.3, 1.4, and 5.1.
        
        Args:
            dataset: Dataset metadata and schema information
            data: The actual data to assess
            quality_report: Base quality assessment report
            
        Returns:
            AIReadinessScore with detailed scoring across multiple dimensions
        """
        try:
            self.logger.info(f"Calculating AI readiness score for dataset {dataset.id}")
            
            # Calculate AI-specific metrics
            feature_quality_score = self._calculate_feature_quality_score(data)
            bias_score = self._calculate_bias_score(data)
            compliance_score = self._calculate_compliance_score(data, dataset.schema)
            scalability_score = self._calculate_scalability_score(data)
            
            # Calculate dimension scores with details
            dimensions = {
                "data_quality": DimensionScore(
                    dimension="data_quality",
                    score=quality_report.overall_score,
                    weight=0.3,
                    details={
                        "completeness": quality_report.completeness_score,
                        "accuracy": quality_report.accuracy_score,
                        "consistency": quality_report.consistency_score,
                        "validity": quality_report.validity_score
                    }
                ),
                "feature_quality": DimensionScore(
                    dimension="feature_quality",
                    score=feature_quality_score,
                    weight=0.25,
                    details=self._get_feature_quality_details(data)
                ),
                "bias_fairness": DimensionScore(
                    dimension="bias_fairness",
                    score=bias_score,
                    weight=0.2,
                    details=self._get_bias_details(data)
                ),
                "compliance": DimensionScore(
                    dimension="compliance",
                    score=compliance_score,
                    weight=0.15,
                    details=self._get_compliance_details(data, dataset.schema)
                ),
                "scalability": DimensionScore(
                    dimension="scalability",
                    score=scalability_score,
                    weight=0.1,
                    details=self._get_scalability_details(data)
                )
            }
            
            # Calculate overall AI readiness score
            overall_score = sum(
                dim.score * dim.weight for dim in dimensions.values()
            )
            
            # Identify improvement areas
            improvement_areas = self._identify_improvement_areas(dimensions)
            
            ai_readiness_score = AIReadinessScore(
                overall_score=overall_score,
                data_quality_score=quality_report.overall_score,
                feature_quality_score=feature_quality_score,
                bias_score=bias_score,
                compliance_score=compliance_score,
                scalability_score=scalability_score,
                dimensions=dimensions,
                improvement_areas=improvement_areas
            )
            
            self.logger.info(f"AI readiness score calculated: {overall_score:.3f}")
            return ai_readiness_score
            
        except Exception as e:
            self.logger.error(f"Error calculating AI readiness score: {str(e)}")
            raise
    
    def _calculate_feature_quality_score(self, data: pd.DataFrame) -> float:
        """
        Calculate feature quality score including correlation analysis and target leakage detection.
        
        Implements feature correlation and target leakage detection as specified in requirement 1.3.
        """
        if data.empty:
            return 0.0
        
        scores = []
        
        # Feature correlation analysis
        correlation_score = self._analyze_feature_correlations(data)
        scores.append(correlation_score)
        
        # Target leakage detection
        leakage_score = self._detect_target_leakage(data)
        scores.append(leakage_score)
        
        # Feature distribution analysis
        distribution_score = self._analyze_feature_distributions(data)
        scores.append(distribution_score)
        
        # Feature variance analysis
        variance_score = self._analyze_feature_variance(data)
        scores.append(variance_score)
        
        return np.mean(scores) if scores else 0.0
    
    def _analyze_feature_correlations(self, data: pd.DataFrame) -> float:
        """Analyze feature correlations to detect multicollinearity issues."""
        numeric_data = data.select_dtypes(include=[np.number])
        
        if numeric_data.shape[1] < 2:
            return 1.0  # No correlation issues with less than 2 numeric features
        
        try:
            # Calculate correlation matrix
            corr_matrix = numeric_data.corr().abs()
            
            # Remove diagonal elements
            np.fill_diagonal(corr_matrix.values, 0)
            
            # Count high correlations (> 0.9)
            high_correlations = (corr_matrix > 0.9).sum().sum() / 2  # Divide by 2 for symmetry
            total_pairs = (corr_matrix.shape[0] * (corr_matrix.shape[0] - 1)) / 2
            
            if total_pairs == 0:
                return 1.0
            
            # Score decreases with more high correlations
            correlation_ratio = high_correlations / total_pairs
            return max(0.0, 1.0 - correlation_ratio)
            
        except Exception:
            return 0.5  # Default score if correlation calculation fails
    
    def _detect_target_leakage(self, data: pd.DataFrame) -> float:
        """
        Detect potential target leakage by analyzing feature relationships.
        
        This is a simplified implementation that looks for suspicious patterns.
        """
        if data.empty:
            return 1.0
        
        # Look for columns that might be targets or derived from targets
        suspicious_columns = []
        
        for column in data.columns:
            column_lower = column.lower()
            
            # Check for common target-like column names
            target_indicators = [
                'target', 'label', 'outcome', 'result', 'prediction',
                'score', 'rating', 'class', 'category'
            ]
            
            if any(indicator in column_lower for indicator in target_indicators):
                suspicious_columns.append(column)
                continue
            
            # Check for perfect correlations with other columns (potential leakage)
            if data[column].dtype in [np.number]:
                for other_column in data.select_dtypes(include=[np.number]).columns:
                    if column != other_column:
                        try:
                            correlation = data[column].corr(data[other_column])
                            if abs(correlation) > 0.99:  # Near-perfect correlation
                                suspicious_columns.append(column)
                                break
                        except:
                            continue
        
        # Score decreases with more suspicious columns
        if len(data.columns) == 0:
            return 1.0
        
        leakage_ratio = len(suspicious_columns) / len(data.columns)
        return max(0.0, 1.0 - leakage_ratio)
    
    def _analyze_feature_distributions(self, data: pd.DataFrame) -> float:
        """Analyze feature distributions for AI suitability."""
        if data.empty:
            return 0.0
        
        scores = []
        
        # Analyze numeric features
        numeric_data = data.select_dtypes(include=[np.number])
        for column in numeric_data.columns:
            series = numeric_data[column].dropna()
            if len(series) > 0:
                # Check for reasonable variance
                if series.std() == 0:
                    scores.append(0.0)  # No variance is bad for ML
                else:
                    # Check for extreme skewness
                    skewness = abs(stats.skew(series))
                    skew_score = max(0.0, 1.0 - min(1.0, skewness / 3.0))
                    scores.append(skew_score)
        
        # Analyze categorical features
        categorical_data = data.select_dtypes(include=['object'])
        for column in categorical_data.columns:
            series = categorical_data[column].dropna()
            if len(series) > 0:
                # Check for reasonable number of categories
                unique_count = series.nunique()
                total_count = len(series)
                
                if unique_count == 1:
                    scores.append(0.0)  # Single category is not useful
                elif unique_count == total_count:
                    scores.append(0.3)  # Too many unique values (might need encoding)
                else:
                    # Optimal range is 2-20 categories
                    if 2 <= unique_count <= 20:
                        scores.append(1.0)
                    else:
                        scores.append(0.7)
        
        return np.mean(scores) if scores else 0.5
    
    def _analyze_feature_variance(self, data: pd.DataFrame) -> float:
        """Analyze feature variance to identify low-variance features."""
        numeric_data = data.select_dtypes(include=[np.number])
        
        if numeric_data.empty:
            return 1.0
        
        low_variance_count = 0
        total_features = len(numeric_data.columns)
        
        for column in numeric_data.columns:
            series = numeric_data[column].dropna()
            if len(series) > 1:
                # Normalize variance by mean to handle different scales
                if series.mean() != 0:
                    cv = series.std() / abs(series.mean())  # Coefficient of variation
                    if cv < 0.01:  # Very low variance
                        low_variance_count += 1
                elif series.std() == 0:  # Zero variance
                    low_variance_count += 1
        
        if total_features == 0:
            return 1.0
        
        # Score decreases with more low-variance features
        low_variance_ratio = low_variance_count / total_features
        return max(0.0, 1.0 - low_variance_ratio)
    
    def _calculate_bias_score(self, data: pd.DataFrame) -> float:
        """
        Calculate bias score by detecting potential bias in the dataset.
        
        This is a simplified implementation that looks for statistical imbalances.
        """
        if data.empty:
            return 1.0
        
        bias_indicators = []
        
        # Check for class imbalance in categorical columns
        categorical_data = data.select_dtypes(include=['object'])
        for column in categorical_data.columns:
            series = categorical_data[column].dropna()
            if len(series) > 0:
                value_counts = series.value_counts()
                if len(value_counts) > 1:
                    # Calculate imbalance ratio
                    max_count = value_counts.max()
                    min_count = value_counts.min()
                    imbalance_ratio = max_count / min_count
                    
                    # High imbalance indicates potential bias
                    if imbalance_ratio > 10:  # 10:1 ratio threshold
                        bias_indicators.append(0.3)  # Significant bias
                    elif imbalance_ratio > 5:  # 5:1 ratio threshold
                        bias_indicators.append(0.6)  # Moderate bias
                    else:
                        bias_indicators.append(1.0)  # Acceptable balance
        
        # Check for outliers in numeric columns (can indicate bias)
        numeric_data = data.select_dtypes(include=[np.number])
        for column in numeric_data.columns:
            series = numeric_data[column].dropna()
            if len(series) > 0:
                # Use IQR method to detect outliers
                Q1 = series.quantile(0.25)
                Q3 = series.quantile(0.75)
                IQR = Q3 - Q1
                
                if IQR > 0:
                    outlier_count = sum((series < Q1 - 1.5 * IQR) | (series > Q3 + 1.5 * IQR))
                    outlier_ratio = outlier_count / len(series)
                    
                    # High outlier ratio might indicate bias
                    if outlier_ratio > 0.1:  # More than 10% outliers
                        bias_indicators.append(0.7)
                    else:
                        bias_indicators.append(1.0)
        
        return np.mean(bias_indicators) if bias_indicators else 1.0
    
    def _calculate_compliance_score(self, data: pd.DataFrame, schema: Optional[Schema]) -> float:
        """Calculate compliance score based on data governance requirements."""
        if data.empty:
            return 0.0
        
        compliance_scores = []
        
        # Check for potential PII (simplified detection)
        pii_score = self._detect_pii_compliance(data)
        compliance_scores.append(pii_score)
        
        # Check schema compliance
        if schema:
            schema_compliance = self._check_schema_compliance(data, schema)
            compliance_scores.append(schema_compliance)
        
        # Check data format compliance
        format_compliance = self._check_format_compliance(data)
        compliance_scores.append(format_compliance)
        
        return np.mean(compliance_scores) if compliance_scores else 0.5
    
    def _detect_pii_compliance(self, data: pd.DataFrame) -> float:
        """Detect potential PII and assess compliance risk."""
        pii_indicators = [
            'email', 'phone', 'ssn', 'social', 'address', 'name',
            'firstname', 'lastname', 'dob', 'birthdate', 'id'
        ]
        
        potential_pii_columns = []
        
        for column in data.columns:
            column_lower = column.lower()
            if any(indicator in column_lower for indicator in pii_indicators):
                potential_pii_columns.append(column)
                continue
            
            # Check data patterns for PII
            if data[column].dtype == 'object':
                sample_values = data[column].dropna().head(100).astype(str)
                
                # Email pattern
                email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
                if any(re.match(email_pattern, str(val)) for val in sample_values):
                    potential_pii_columns.append(column)
                    continue
                
                # Phone pattern
                phone_pattern = r'^\+?1?-?\.?\s?\(?(\d{3})\)?[\s.-]?(\d{3})[\s.-]?(\d{4})$'
                if any(re.match(phone_pattern, str(val)) for val in sample_values):
                    potential_pii_columns.append(column)
        
        # Score decreases with more potential PII
        if len(data.columns) == 0:
            return 1.0
        
        pii_ratio = len(potential_pii_columns) / len(data.columns)
        return max(0.0, 1.0 - pii_ratio)
    
    def _check_schema_compliance(self, data: pd.DataFrame, schema: Schema) -> float:
        """Check compliance with defined schema."""
        if not schema.validate_data_types():
            return 0.0
        
        compliance_scores = []
        
        for column, expected_type in schema.columns.items():
            if column in data.columns:
                # Check type compliance
                actual_data = data[column].dropna()
                if not actual_data.empty:
                    type_matches = sum(1 for val in actual_data if self._value_matches_type(val, expected_type))
                    type_compliance = type_matches / len(actual_data)
                    compliance_scores.append(type_compliance)
        
        return np.mean(compliance_scores) if compliance_scores else 1.0
    
    def _value_matches_type(self, value: Any, expected_type: str) -> bool:
        """Check if a value matches the expected type."""
        try:
            if expected_type == 'integer':
                return isinstance(value, (int, np.integer)) or (isinstance(value, str) and value.isdigit())
            elif expected_type == 'float':
                float(value)
                return True
            elif expected_type == 'string':
                return isinstance(value, str)
            elif expected_type == 'boolean':
                return isinstance(value, (bool, np.bool_)) or str(value).lower() in ['true', 'false', '1', '0']
            elif expected_type == 'datetime':
                pd.to_datetime(value)
                return True
            elif expected_type == 'categorical':
                return isinstance(value, str)
            else:
                return True  # Unknown type, assume valid
        except (ValueError, TypeError):
            return False
    
    def _check_format_compliance(self, data: pd.DataFrame) -> float:
        """Check general format compliance."""
        # This is a simplified check for common format issues
        format_scores = []
        
        # Check for consistent encoding
        for column in data.select_dtypes(include=['object']).columns:
            series = data[column].dropna().astype(str)
            if len(series) > 0:
                # Check for encoding issues (simplified)
                encoding_issues = sum(1 for val in series if any(ord(char) > 127 for char in str(val)))
                encoding_score = 1.0 - (encoding_issues / len(series))
                format_scores.append(encoding_score)
        
        return np.mean(format_scores) if format_scores else 1.0
    
    def _calculate_scalability_score(self, data: pd.DataFrame) -> float:
        """Calculate scalability score based on data characteristics."""
        if data.empty:
            return 0.0
        
        scalability_factors = []
        
        # Data size factor
        size_score = self._assess_data_size_scalability(data)
        scalability_factors.append(size_score)
        
        # Memory efficiency factor
        memory_score = self._assess_memory_efficiency(data)
        scalability_factors.append(memory_score)
        
        # Processing complexity factor
        complexity_score = self._assess_processing_complexity(data)
        scalability_factors.append(complexity_score)
        
        return np.mean(scalability_factors) if scalability_factors else 0.5
    
    def _assess_data_size_scalability(self, data: pd.DataFrame) -> float:
        """Assess scalability based on data size."""
        row_count = len(data)
        col_count = len(data.columns)
        
        # Optimal size ranges for different ML scenarios
        if row_count < 1000:
            size_score = 0.3  # Too small for robust ML
        elif row_count < 10000:
            size_score = 0.7  # Adequate for simple models
        elif row_count < 1000000:
            size_score = 1.0  # Good size for most ML tasks
        else:
            size_score = 0.8  # Large but manageable
        
        # Adjust for feature count
        if col_count > 1000:
            size_score *= 0.8  # High dimensionality penalty
        elif col_count > 100:
            size_score *= 0.9  # Moderate dimensionality penalty
        
        return size_score
    
    def _assess_memory_efficiency(self, data: pd.DataFrame) -> float:
        """Assess memory efficiency of the dataset."""
        try:
            # Calculate memory usage
            memory_usage = data.memory_usage(deep=True).sum()
            
            # Assess efficiency based on data types
            efficiency_scores = []
            
            for column in data.columns:
                if data[column].dtype == 'object':
                    # String columns are less memory efficient
                    unique_ratio = data[column].nunique() / len(data)
                    if unique_ratio > 0.5:
                        efficiency_scores.append(0.6)  # High cardinality strings
                    else:
                        efficiency_scores.append(0.8)  # Could be categorical
                elif data[column].dtype in ['int64', 'float64']:
                    # Check if smaller types could be used
                    if data[column].dtype == 'int64':
                        max_val = data[column].max()
                        min_val = data[column].min()
                        if pd.isna(max_val) or pd.isna(min_val):
                            efficiency_scores.append(0.8)
                        elif -32768 <= min_val and max_val <= 32767:
                            efficiency_scores.append(0.7)  # Could use int16
                        else:
                            efficiency_scores.append(1.0)  # int64 needed
                    else:
                        efficiency_scores.append(0.9)  # float64 is reasonable
                else:
                    efficiency_scores.append(1.0)  # Other types are fine
            
            return np.mean(efficiency_scores) if efficiency_scores else 0.5
            
        except Exception:
            return 0.5  # Default score if assessment fails
    
    def _assess_processing_complexity(self, data: pd.DataFrame) -> float:
        """Assess processing complexity for ML algorithms."""
        complexity_factors = []
        
        # Feature count complexity
        feature_count = len(data.columns)
        if feature_count < 10:
            complexity_factors.append(1.0)  # Low complexity
        elif feature_count < 100:
            complexity_factors.append(0.9)  # Moderate complexity
        elif feature_count < 1000:
            complexity_factors.append(0.7)  # High complexity
        else:
            complexity_factors.append(0.5)  # Very high complexity
        
        # Data type complexity
        object_columns = len(data.select_dtypes(include=['object']).columns)
        if object_columns == 0:
            complexity_factors.append(1.0)  # No text processing needed
        else:
            text_ratio = object_columns / len(data.columns)
            complexity_factors.append(max(0.3, 1.0 - text_ratio))
        
        # Missing data complexity
        missing_ratio = data.isnull().sum().sum() / data.size
        complexity_factors.append(max(0.5, 1.0 - missing_ratio))
        
        return np.mean(complexity_factors) if complexity_factors else 0.5
    
    def detect_anomalies(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect statistical anomalies in the dataset.
        
        Implements statistical anomaly detection capabilities as specified in requirement 1.4.
        """
        if data.empty:
            return {"anomalies": [], "anomaly_score": 1.0}
        
        anomalies = []
        
        # Detect outliers in numeric columns using Isolation Forest
        numeric_data = data.select_dtypes(include=[np.number])
        if not numeric_data.empty:
            try:
                # Use Isolation Forest for anomaly detection
                iso_forest = IsolationForest(contamination=0.1, random_state=42)
                outlier_predictions = iso_forest.fit_predict(numeric_data.fillna(numeric_data.mean()))
                
                outlier_indices = np.where(outlier_predictions == -1)[0]
                
                for idx in outlier_indices:
                    anomalies.append({
                        "type": "statistical_outlier",
                        "row_index": int(idx),
                        "description": f"Statistical outlier detected in row {idx}",
                        "severity": "medium",
                        "affected_columns": numeric_data.columns.tolist()
                    })
            except Exception as e:
                self.logger.warning(f"Isolation Forest anomaly detection failed: {str(e)}")
        
        # Detect anomalies in categorical columns
        categorical_data = data.select_dtypes(include=['object'])
        for column in categorical_data.columns:
            series = categorical_data[column].dropna()
            if len(series) > 0:
                value_counts = series.value_counts()
                
                # Detect rare categories (less than 1% of data)
                rare_threshold = len(series) * 0.01
                rare_values = value_counts[value_counts < rare_threshold]
                
                if not rare_values.empty:
                    anomalies.append({
                        "type": "rare_category",
                        "column": column,
                        "description": f"Rare categories detected in column '{column}': {list(rare_values.index)}",
                        "severity": "low",
                        "affected_rows": int(rare_values.sum()),
                        "rare_values": list(rare_values.index)
                    })
        
        # Calculate overall anomaly score
        total_rows = len(data)
        anomalous_rows = sum(
            anomaly.get("affected_rows", 1) for anomaly in anomalies
            if anomaly["type"] != "statistical_outlier"
        ) + len([a for a in anomalies if a["type"] == "statistical_outlier"])
        
        anomaly_score = max(0.0, 1.0 - (anomalous_rows / total_rows)) if total_rows > 0 else 1.0
        
        return {
            "anomalies": anomalies,
            "anomaly_score": anomaly_score,
            "total_anomalies": len(anomalies),
            "anomalous_rows": anomalous_rows
        }
    
    def _get_feature_quality_details(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Get detailed feature quality information."""
        return {
            "correlation_analysis": self._analyze_feature_correlations(data),
            "target_leakage_score": self._detect_target_leakage(data),
            "distribution_score": self._analyze_feature_distributions(data),
            "variance_score": self._analyze_feature_variance(data),
            "numeric_features": len(data.select_dtypes(include=[np.number]).columns),
            "categorical_features": len(data.select_dtypes(include=['object']).columns)
        }
    
    def _get_bias_details(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Get detailed bias analysis information."""
        categorical_data = data.select_dtypes(include=['object'])
        imbalance_info = {}
        
        for column in categorical_data.columns:
            series = categorical_data[column].dropna()
            if len(series) > 0:
                value_counts = series.value_counts()
                if len(value_counts) > 1:
                    max_count = value_counts.max()
                    min_count = value_counts.min()
                    imbalance_ratio = max_count / min_count
                    imbalance_info[column] = {
                        "imbalance_ratio": float(imbalance_ratio),
                        "categories": len(value_counts),
                        "most_common": value_counts.index[0],
                        "least_common": value_counts.index[-1]
                    }
        
        return {
            "overall_bias_score": self._calculate_bias_score(data),
            "class_imbalance": imbalance_info,
            "categorical_columns_analyzed": len(categorical_data.columns)
        }
    
    def _get_compliance_details(self, data: pd.DataFrame, schema: Optional[Schema]) -> Dict[str, Any]:
        """Get detailed compliance information."""
        return {
            "pii_compliance_score": self._detect_pii_compliance(data),
            "schema_compliance_score": self._check_schema_compliance(data, schema) if schema else None,
            "format_compliance_score": self._check_format_compliance(data),
            "potential_pii_columns": self._identify_potential_pii_columns(data)
        }
    
    def _identify_potential_pii_columns(self, data: pd.DataFrame) -> List[str]:
        """Identify columns that might contain PII."""
        pii_indicators = [
            'email', 'phone', 'ssn', 'social', 'address', 'name',
            'firstname', 'lastname', 'dob', 'birthdate', 'id'
        ]
        
        potential_pii = []
        for column in data.columns:
            column_lower = column.lower()
            if any(indicator in column_lower for indicator in pii_indicators):
                potential_pii.append(column)
        
        return potential_pii
    
    def _get_scalability_details(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Get detailed scalability information."""
        return {
            "data_size_score": self._assess_data_size_scalability(data),
            "memory_efficiency_score": self._assess_memory_efficiency(data),
            "processing_complexity_score": self._assess_processing_complexity(data),
            "row_count": len(data),
            "column_count": len(data.columns),
            "memory_usage_mb": data.memory_usage(deep=True).sum() / (1024 * 1024)
        }
    
    def _identify_improvement_areas(self, dimensions: Dict[str, DimensionScore]) -> List[ImprovementArea]:
        """Identify areas for improvement based on dimension scores."""
        improvement_areas = []
        
        for dim_name, dim_score in dimensions.items():
            if dim_score.score < 0.8:  # Threshold for improvement
                priority = "high" if dim_score.score < 0.5 else "medium"
                target_score = min(1.0, dim_score.score + 0.2)
                
                actions = self._get_improvement_actions(dim_name, dim_score.score)
                
                improvement_areas.append(ImprovementArea(
                    area=dim_name,
                    current_score=dim_score.score,
                    target_score=target_score,
                    priority=priority,
                    actions=actions
                ))
        
        return improvement_areas
    
    def _get_improvement_actions(self, dimension: str, score: float) -> List[str]:
        """Get specific improvement actions for a dimension."""
        actions = []
        
        if dimension == "data_quality":
            if score < 0.5:
                actions.extend([
                    "Address missing values through imputation or collection",
                    "Validate and correct data accuracy issues",
                    "Standardize data formats for consistency"
                ])
            else:
                actions.extend([
                    "Fine-tune data validation rules",
                    "Implement automated quality monitoring"
                ])
        
        elif dimension == "feature_quality":
            if score < 0.5:
                actions.extend([
                    "Remove highly correlated features",
                    "Address potential target leakage",
                    "Transform skewed distributions"
                ])
            else:
                actions.extend([
                    "Optimize feature selection",
                    "Consider feature engineering techniques"
                ])
        
        elif dimension == "bias_fairness":
            if score < 0.5:
                actions.extend([
                    "Address class imbalance through sampling techniques",
                    "Investigate potential bias sources",
                    "Implement fairness constraints"
                ])
            else:
                actions.extend([
                    "Monitor for emerging bias patterns",
                    "Validate fairness metrics"
                ])
        
        elif dimension == "compliance":
            if score < 0.5:
                actions.extend([
                    "Implement PII detection and anonymization",
                    "Ensure schema compliance",
                    "Address data governance requirements"
                ])
            else:
                actions.extend([
                    "Maintain compliance monitoring",
                    "Update governance policies as needed"
                ])
        
        elif dimension == "scalability":
            if score < 0.5:
                actions.extend([
                    "Optimize data types for memory efficiency",
                    "Consider data sampling strategies",
                    "Implement distributed processing"
                ])
            else:
                actions.extend([
                    "Monitor performance metrics",
                    "Plan for data growth"
                ])
        
        return actions