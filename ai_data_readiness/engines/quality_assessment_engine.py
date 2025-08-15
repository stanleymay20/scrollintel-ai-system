"""Core quality assessment engine for AI Data Readiness Platform."""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
import re
from scipy import stats
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

from ..models.base_models import (
    QualityReport, QualityIssue, Recommendation, QualityDimension,
    Dataset, DatasetMetadata, Schema, AIReadinessScore, DimensionScore,
    ImprovementArea
)
from ..core.config import Config
from .ai_quality_metrics import AIQualityMetrics


logger = logging.getLogger(__name__)


@dataclass
class QualityMetrics:
    """Container for quality assessment metrics."""
    completeness: float = 0.0
    accuracy: float = 0.0
    consistency: float = 0.0
    validity: float = 0.0
    uniqueness: float = 0.0
    timeliness: float = 0.0


class QualityAssessmentEngine:
    """
    Core quality assessment engine with multi-dimensional scoring.
    
    Implements completeness, accuracy, consistency, and validity metrics
    as specified in requirements 1.1, 1.2, and 5.1.
    """
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize the quality assessment engine."""
        self.config = config or Config()
        self.logger = logging.getLogger(__name__)
        
        # Initialize AI-specific quality metrics
        self.ai_metrics = AIQualityMetrics()
        
        # Quality dimension weights for overall score calculation
        self.dimension_weights = {
            QualityDimension.COMPLETENESS: 0.25,
            QualityDimension.ACCURACY: 0.20,
            QualityDimension.CONSISTENCY: 0.20,
            QualityDimension.VALIDITY: 0.20,
            QualityDimension.UNIQUENESS: 0.10,
            QualityDimension.TIMELINESS: 0.05
        }
    
    def assess_quality(self, dataset: Dataset, data: pd.DataFrame) -> QualityReport:
        """
        Perform comprehensive quality assessment on a dataset.
        
        Args:
            dataset: Dataset metadata and schema information
            data: The actual data to assess
            
        Returns:
            QualityReport with scores and recommendations
        """
        try:
            self.logger.info(f"Starting quality assessment for dataset {dataset.id}")
            
            # Calculate individual quality metrics
            metrics = self._calculate_quality_metrics(data, dataset.schema)
            
            # Generate quality issues and recommendations
            issues = self._identify_quality_issues(data, metrics, dataset.schema)
            recommendations = self._generate_recommendations(issues, metrics)
            
            # Calculate overall score
            overall_score = self._calculate_overall_score(metrics)
            
            # Create quality report
            report = QualityReport(
                dataset_id=dataset.id,
                overall_score=overall_score,
                completeness_score=metrics.completeness,
                accuracy_score=metrics.accuracy,
                consistency_score=metrics.consistency,
                validity_score=metrics.validity,
                uniqueness_score=metrics.uniqueness,
                timeliness_score=metrics.timeliness,
                issues=issues,
                recommendations=recommendations,
                generated_at=datetime.utcnow()
            )
            
            self.logger.info(f"Quality assessment completed. Overall score: {overall_score:.3f}")
            return report
            
        except Exception as e:
            self.logger.error(f"Error during quality assessment: {str(e)}")
            raise
    
    def assess_quality_dataframe(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Assess quality of a DataFrame for streaming processing.
        
        This method provides a simplified quality assessment suitable for
        real-time streaming data processing.
        
        Args:
            data: DataFrame to assess
            
        Returns:
            Dictionary containing quality metrics and overall score
        """
        try:
            if data.empty:
                return {
                    'overall_score': 0.0,
                    'completeness_score': 0.0,
                    'accuracy_score': 0.0,
                    'consistency_score': 0.0,
                    'validity_score': 0.0,
                    'record_count': 0
                }
            
            # Calculate basic quality metrics
            completeness = self._calculate_completeness(data)
            accuracy = self._calculate_accuracy(data, None)  # No schema for streaming
            consistency = self._calculate_consistency(data, None)
            validity = self._calculate_validity(data, None)
            
            # Calculate overall score
            overall_score = (completeness + accuracy + consistency + validity) / 4
            
            return {
                'overall_score': overall_score,
                'completeness_score': completeness,
                'accuracy_score': accuracy,
                'consistency_score': consistency,
                'validity_score': validity,
                'record_count': len(data),
                'column_count': len(data.columns),
                'missing_values': data.isnull().sum().sum(),
                'duplicate_rows': data.duplicated().sum()
            }
            
        except Exception as e:
            self.logger.error(f"Error in streaming quality assessment: {str(e)}")
            return {
                'overall_score': 0.0,
                'completeness_score': 0.0,
                'accuracy_score': 0.0,
                'consistency_score': 0.0,
                'validity_score': 0.0,
                'record_count': len(data) if not data.empty else 0,
                'error': str(e)
            }
    
    def _calculate_quality_metrics(self, data: pd.DataFrame, schema: Optional[Schema]) -> QualityMetrics:
        """Calculate all quality dimension scores."""
        metrics = QualityMetrics()
        
        # Completeness: Measure missing data
        metrics.completeness = self._calculate_completeness(data)
        
        # Accuracy: Measure data correctness based on patterns and constraints
        metrics.accuracy = self._calculate_accuracy(data, schema)
        
        # Consistency: Measure data consistency across columns and formats
        metrics.consistency = self._calculate_consistency(data, schema)
        
        # Validity: Measure adherence to business rules and constraints
        metrics.validity = self._calculate_validity(data, schema)
        
        # Uniqueness: Measure duplicate records
        metrics.uniqueness = self._calculate_uniqueness(data)
        
        # Timeliness: Measure data freshness (if temporal columns exist)
        metrics.timeliness = self._calculate_timeliness(data)
        
        return metrics
    
    def _calculate_completeness(self, data: pd.DataFrame) -> float:
        """Calculate completeness score based on missing values."""
        if data.empty:
            return 0.0
        
        total_cells = data.size
        missing_cells = data.isnull().sum().sum()
        
        if total_cells == 0:
            return 1.0
        
        completeness = (total_cells - missing_cells) / total_cells
        return max(0.0, min(1.0, completeness))
    
    def _calculate_accuracy(self, data: pd.DataFrame, schema: Optional[Schema]) -> float:
        """Calculate accuracy score based on data type conformity."""
        if data.empty:
            return 0.0
        
        # Simple accuracy check based on data types
        accuracy_scores = []
        
        for column in data.columns:
            series = data[column].dropna()
            if len(series) == 0:
                accuracy_scores.append(1.0)
                continue
            
            # Check for mixed types
            types = set(type(x).__name__ for x in series)
            if len(types) == 1:
                accuracy_scores.append(1.0)
            else:
                # Penalize mixed types
                accuracy_scores.append(0.7)
        
        return np.mean(accuracy_scores) if accuracy_scores else 0.0
    
    def _calculate_consistency(self, data: pd.DataFrame, schema: Optional[Schema]) -> float:
        """Calculate consistency score based on format uniformity."""
        if data.empty:
            return 0.0
        
        consistency_scores = []
        
        for column in data.columns:
            series = data[column].dropna()
            if len(series) <= 1:
                consistency_scores.append(1.0)
                continue
            
            if series.dtype == 'object':
                # Check string length consistency
                lengths = series.astype(str).str.len()
                if lengths.std() == 0:
                    consistency_scores.append(1.0)
                else:
                    cv = lengths.std() / lengths.mean() if lengths.mean() > 0 else 0
                    consistency_scores.append(max(0.0, 1.0 - min(1.0, cv)))
            else:
                # For numeric data, use coefficient of variation
                if series.std() == 0:
                    consistency_scores.append(1.0)
                else:
                    mean_val = series.mean()
                    if mean_val == 0:
                        consistency_scores.append(1.0)
                    else:
                        cv = series.std() / abs(mean_val)
                        consistency_scores.append(max(0.0, 1.0 - min(1.0, cv)))
        
        return np.mean(consistency_scores) if consistency_scores else 0.0
    
    def _calculate_validity(self, data: pd.DataFrame, schema: Optional[Schema]) -> float:
        """Calculate validity score based on outlier detection."""
        if data.empty:
            return 0.0
        
        validity_scores = []
        
        # Check numeric columns for outliers
        for column in data.select_dtypes(include=[np.number]).columns:
            series = data[column].dropna()
            if len(series) == 0:
                validity_scores.append(1.0)
                continue
            
            # Use IQR method for outlier detection
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            
            if IQR == 0:
                validity_scores.append(1.0)
            else:
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                valid_values = sum((series >= lower_bound) & (series <= upper_bound))
                validity_scores.append(valid_values / len(series))
        
        # For non-numeric columns, assume valid
        for column in data.select_dtypes(exclude=[np.number]).columns:
            validity_scores.append(1.0)
        
        return np.mean(validity_scores) if validity_scores else 1.0
    
    def _calculate_uniqueness(self, data: pd.DataFrame) -> float:
        """Calculate uniqueness score based on duplicate records."""
        if data.empty:
            return 1.0
        
        total_rows = len(data)
        unique_rows = len(data.drop_duplicates())
        
        return unique_rows / total_rows if total_rows > 0 else 1.0
    
    def _calculate_timeliness(self, data: pd.DataFrame) -> float:
        """Calculate timeliness score based on data freshness."""
        # For streaming data, assume timely unless we can detect timestamps
        return 1.0
    
    def _calculate_overall_score(self, metrics: QualityMetrics) -> float:
        """Calculate weighted overall quality score."""
        scores = {
            QualityDimension.COMPLETENESS: metrics.completeness,
            QualityDimension.ACCURACY: metrics.accuracy,
            QualityDimension.CONSISTENCY: metrics.consistency,
            QualityDimension.VALIDITY: metrics.validity,
            QualityDimension.UNIQUENESS: metrics.uniqueness,
            QualityDimension.TIMELINESS: metrics.timeliness
        }
        
        # If completeness is 0 (empty data), overall score should be 0
        if metrics.completeness == 0.0:
            return 0.0
        
        weighted_sum = sum(
            scores[dimension] * weight 
            for dimension, weight in self.dimension_weights.items()
        )
        
        return max(0.0, min(1.0, weighted_sum))
    
    def _identify_quality_issues(self, data: pd.DataFrame, metrics: QualityMetrics, schema: Optional[Schema]) -> List[QualityIssue]:
        """Identify specific quality issues based on assessment results."""
        issues = []
        
        # Simplified issue identification for core functionality
        if metrics.completeness < 0.8:
            issues.append(QualityIssue(
                dimension=QualityDimension.COMPLETENESS,
                severity="high",
                description="High percentage of missing values detected",
                affected_columns=[],
                affected_rows=0,
                recommendation="Consider data imputation or collection strategies"
            ))
        
        return issues
    
    def _generate_recommendations(self, issues: List[QualityIssue], metrics: QualityMetrics) -> List[Recommendation]:
        """Generate actionable recommendations based on quality issues."""
        recommendations = []
        
        for issue in issues:
            if issue.dimension == QualityDimension.COMPLETENESS:
                recommendations.append(Recommendation(
                    type="data_imputation",
                    priority="high",
                    description="Implement data imputation strategy",
                    implementation="Use statistical methods to fill missing values",
                    estimated_impact=0.8,
                    estimated_effort="medium"
                ))
        
        return recommendations
    
    def calculate_ai_readiness_score(self, dataset: Dataset, data: pd.DataFrame) -> AIReadinessScore:
        """
        Calculate comprehensive AI readiness score with AI-specific metrics.
        
        Implements AI readiness scoring algorithm as specified in requirements 1.3, 1.4, and 5.1.
        """
        # Get base quality report first
        quality_report = self.assess_quality(dataset, data)
        
        # Use AI metrics engine to calculate AI readiness score
        return self.ai_metrics.calculate_ai_readiness_score(dataset, data, quality_report)