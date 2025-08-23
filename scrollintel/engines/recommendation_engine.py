"""
AI Recommendation Engine for Data Pipeline Automation

This module provides intelligent recommendations for pipeline optimization,
transformations, schema mapping, and performance improvements using ML algorithms.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
import pandas as pd

logger = logging.getLogger(__name__)


class RecommendationType(Enum):
    TRANSFORMATION = "transformation"
    PERFORMANCE = "performance"
    SCHEMA_MAPPING = "schema_mapping"
    JOIN_STRATEGY = "join_strategy"


@dataclass
class Schema:
    """Data schema representation"""
    name: str
    columns: List[Dict[str, Any]]
    data_types: Dict[str, str]
    sample_data: Optional[pd.DataFrame] = None
    
    def get_column_names(self) -> List[str]:
        return [col['name'] for col in self.columns]
    
    def get_numeric_columns(self) -> List[str]:
        return [name for name, dtype in self.data_types.items() 
                if dtype in ['int64', 'float64', 'int32', 'float32']]
    
    def get_categorical_columns(self) -> List[str]:
        return [name for name, dtype in self.data_types.items() 
                if dtype in ['object', 'category', 'string']]


@dataclass
class Dataset:
    """Dataset representation for analysis"""
    name: str
    schema: Schema
    row_count: int
    size_mb: float
    quality_score: float = 0.0
    
    def get_complexity_score(self) -> float:
        """Calculate dataset complexity based on size and column count"""
        column_count = len(self.schema.columns)
        return (self.row_count * column_count) / 1000000  # Normalize


@dataclass
class Transformation:
    """Transformation recommendation"""
    name: str
    type: str
    description: str
    confidence: float
    parameters: Dict[str, Any]
    estimated_performance_impact: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'type': self.type,
            'description': self.description,
            'confidence': self.confidence,
            'parameters': self.parameters,
            'estimated_performance_impact': self.estimated_performance_impact
        }


@dataclass
class JoinRecommendation:
    """Join strategy recommendation"""
    join_type: str
    left_key: str
    right_key: str
    confidence: float
    estimated_rows: int
    performance_score: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'join_type': self.join_type,
            'left_key': self.left_key,
            'right_key': self.right_key,
            'confidence': self.confidence,
            'estimated_rows': self.estimated_rows,
            'performance_score': self.performance_score
        }


@dataclass
class Optimization:
    """Performance optimization recommendation"""
    category: str
    description: str
    impact: str
    implementation_effort: str
    estimated_improvement: float
    priority: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'category': self.category,
            'description': self.description,
            'impact': self.impact,
            'implementation_effort': self.implementation_effort,
            'estimated_improvement': self.estimated_improvement,
            'priority': self.priority
        }


@dataclass
class DataPatternAnalysis:
    """Data pattern analysis results"""
    patterns: List[str]
    anomalies: List[str]
    recommendations: List[str]
    quality_issues: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'patterns': self.patterns,
            'anomalies': self.anomalies,
            'recommendations': self.recommendations,
            'quality_issues': self.quality_issues
        }


class RecommendationEngine:
    """
    AI-powered recommendation engine for data pipeline automation
    
    Provides intelligent suggestions for:
    - Data transformations
    - Performance optimizations
    - Schema mappings
    - Join strategies
    """
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.performance_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.clustering_model = KMeans(n_clusters=5, random_state=42)
        self.transformation_patterns = self._load_transformation_patterns()
        self.performance_history = []
        self.user_feedback = {}
        
    def _load_transformation_patterns(self) -> Dict[str, List[Dict]]:
        """Load common transformation patterns"""
        return {
            'data_cleaning': [
                {
                    'name': 'remove_nulls',
                    'description': 'Remove rows with null values',
                    'conditions': ['high_null_percentage'],
                    'confidence_base': 0.8
                },
                {
                    'name': 'fill_missing_values',
                    'description': 'Fill missing values with mean/mode',
                    'conditions': ['moderate_null_percentage'],
                    'confidence_base': 0.7
                }
            ],
            'data_type_conversion': [
                {
                    'name': 'string_to_datetime',
                    'description': 'Convert string columns to datetime',
                    'conditions': ['datetime_pattern'],
                    'confidence_base': 0.9
                },
                {
                    'name': 'string_to_numeric',
                    'description': 'Convert string columns to numeric',
                    'conditions': ['numeric_pattern'],
                    'confidence_base': 0.85
                }
            ],
            'aggregation': [
                {
                    'name': 'group_by_aggregate',
                    'description': 'Group by categorical columns and aggregate',
                    'conditions': ['high_cardinality'],
                    'confidence_base': 0.75
                }
            ]
        }
    
    def recommend_transformations(self, source_schema: Schema, target_schema: Schema) -> List[Transformation]:
        """
        Recommend appropriate transformations between source and target schemas
        
        Args:
            source_schema: Source data schema
            target_schema: Target data schema
            
        Returns:
            List of transformation recommendations
        """
        try:
            recommendations = []
            
            # Analyze schema differences
            source_cols = set(source_schema.get_column_names())
            target_cols = set(target_schema.get_column_names())
            
            # Missing columns in target
            missing_cols = target_cols - source_cols
            if missing_cols:
                recommendations.append(Transformation(
                    name='add_missing_columns',
                    type='schema_alignment',
                    description=f'Add missing columns: {", ".join(missing_cols)}',
                    confidence=0.9,
                    parameters={'columns': list(missing_cols)},
                    estimated_performance_impact=0.1
                ))
            
            # Data type conversions
            for col_name in source_cols.intersection(target_cols):
                source_type = source_schema.data_types.get(col_name)
                target_type = target_schema.data_types.get(col_name)
                
                if source_type != target_type and source_type and target_type:
                    recommendations.append(Transformation(
                        name='convert_data_type',
                        type='data_type_conversion',
                        description=f'Convert {col_name} from {source_type} to {target_type}',
                        confidence=0.85,
                        parameters={
                            'column': col_name,
                            'from_type': source_type,
                            'to_type': target_type
                        },
                        estimated_performance_impact=0.05
                    ))
            
            # Pattern-based recommendations
            if source_schema.sample_data is not None:
                pattern_recommendations = self._analyze_data_patterns(source_schema.sample_data)
                recommendations.extend(pattern_recommendations)
            
            # Sort by confidence
            recommendations.sort(key=lambda x: x.confidence, reverse=True)
            
            logger.info(f"Generated {len(recommendations)} transformation recommendations")
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating transformation recommendations: {e}")
            return []
    
    def suggest_optimizations(self, pipeline: Dict[str, Any], metrics: Dict[str, Any]) -> List[Optimization]:
        """
        Suggest performance optimizations based on pipeline configuration and metrics
        
        Args:
            pipeline: Pipeline configuration
            metrics: Performance metrics
            
        Returns:
            List of optimization recommendations
        """
        try:
            optimizations = []
            
            # Analyze execution time
            execution_time = metrics.get('execution_time_seconds', 0)
            if execution_time > 300:  # 5 minutes
                optimizations.append(Optimization(
                    category='performance',
                    description='Consider parallel processing for long-running transformations',
                    impact='high',
                    implementation_effort='medium',
                    estimated_improvement=0.4,
                    priority=1
                ))
            
            # Analyze memory usage
            memory_usage = metrics.get('memory_usage_mb', 0)
            if memory_usage > 1000:  # 1GB
                optimizations.append(Optimization(
                    category='memory',
                    description='Implement streaming processing to reduce memory footprint',
                    impact='high',
                    implementation_effort='high',
                    estimated_improvement=0.6,
                    priority=2
                ))
            
            # Analyze data volume
            rows_processed = metrics.get('rows_processed', 0)
            if rows_processed > 1000000:  # 1M rows
                optimizations.append(Optimization(
                    category='scalability',
                    description='Add data partitioning for better scalability',
                    impact='medium',
                    implementation_effort='medium',
                    estimated_improvement=0.3,
                    priority=3
                ))
            
            # Analyze error rate
            error_rate = metrics.get('error_rate', 0)
            if error_rate > 0.01:  # 1%
                optimizations.append(Optimization(
                    category='reliability',
                    description='Add data validation and error handling',
                    impact='high',
                    implementation_effort='low',
                    estimated_improvement=0.8,
                    priority=1
                ))
            
            # Sort by priority and impact
            optimizations.sort(key=lambda x: (x.priority, -x.estimated_improvement))
            
            logger.info(f"Generated {len(optimizations)} optimization recommendations")
            return optimizations
            
        except Exception as e:
            logger.error(f"Error generating optimization recommendations: {e}")
            return []
    
    def recommend_join_strategy(self, left_dataset: Dataset, right_dataset: Dataset) -> JoinRecommendation:
        """
        Recommend optimal join strategy for two datasets
        
        Args:
            left_dataset: Left dataset for join
            right_dataset: Right dataset for join
            
        Returns:
            Join strategy recommendation
        """
        try:
            # Find potential join keys
            left_cols = set(left_dataset.schema.get_column_names())
            right_cols = set(right_dataset.schema.get_column_names())
            common_cols = left_cols.intersection(right_cols)
            
            if not common_cols:
                # No common columns, suggest cross join with caution
                return JoinRecommendation(
                    join_type='cross',
                    left_key='',
                    right_key='',
                    confidence=0.3,
                    estimated_rows=left_dataset.row_count * right_dataset.row_count,
                    performance_score=0.1
                )
            
            # Select best join key based on data types and cardinality
            best_key = self._select_best_join_key(common_cols, left_dataset, right_dataset)
            
            # Determine join type based on dataset sizes
            size_ratio = left_dataset.row_count / right_dataset.row_count
            
            if 0.8 <= size_ratio <= 1.2:
                join_type = 'inner'
                confidence = 0.9
            elif size_ratio > 2:
                join_type = 'left'
                confidence = 0.8
            else:
                join_type = 'inner'
                confidence = 0.7
            
            # Estimate result size
            estimated_rows = min(left_dataset.row_count, right_dataset.row_count)
            
            # Calculate performance score
            performance_score = self._calculate_join_performance_score(
                left_dataset, right_dataset, join_type
            )
            
            return JoinRecommendation(
                join_type=join_type,
                left_key=best_key,
                right_key=best_key,
                confidence=confidence,
                estimated_rows=estimated_rows,
                performance_score=performance_score
            )
            
        except Exception as e:
            logger.error(f"Error generating join recommendation: {e}")
            return JoinRecommendation(
                join_type='inner',
                left_key='',
                right_key='',
                confidence=0.5,
                estimated_rows=0,
                performance_score=0.5
            )
    
    def analyze_data_patterns(self, data_sample: pd.DataFrame) -> DataPatternAnalysis:
        """
        Analyze data patterns and provide insights
        
        Args:
            data_sample: Sample of the data to analyze
            
        Returns:
            Data pattern analysis results
        """
        try:
            patterns = []
            anomalies = []
            recommendations = []
            quality_issues = []
            
            # Analyze null patterns
            null_percentages = data_sample.isnull().sum() / len(data_sample)
            high_null_cols = null_percentages[null_percentages > 0.5].index.tolist()
            
            if high_null_cols:
                quality_issues.append(f"High null percentage in columns: {', '.join(high_null_cols)}")
                recommendations.append("Consider removing or imputing high-null columns")
            
            # Analyze data types
            for col in data_sample.columns:
                if data_sample[col].dtype == 'object':
                    # Check if it's actually numeric
                    try:
                        pd.to_numeric(data_sample[col], errors='raise')
                        patterns.append(f"Column '{col}' appears to be numeric but stored as string")
                        recommendations.append(f"Convert column '{col}' to numeric type")
                    except:
                        pass
                    
                    # Check for datetime patterns
                    if self._is_datetime_pattern(data_sample[col]):
                        patterns.append(f"Column '{col}' appears to contain datetime values")
                        recommendations.append(f"Convert column '{col}' to datetime type")
            
            # Analyze duplicates
            duplicate_count = data_sample.duplicated().sum()
            if duplicate_count > 0:
                quality_issues.append(f"Found {duplicate_count} duplicate rows")
                recommendations.append("Remove duplicate rows")
            
            # Analyze outliers in numeric columns
            numeric_cols = data_sample.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                q1 = data_sample[col].quantile(0.25)
                q3 = data_sample[col].quantile(0.75)
                iqr = q3 - q1
                outliers = data_sample[
                    (data_sample[col] < q1 - 1.5 * iqr) | 
                    (data_sample[col] > q3 + 1.5 * iqr)
                ]
                
                if len(outliers) > len(data_sample) * 0.05:  # More than 5% outliers
                    anomalies.append(f"High number of outliers in column '{col}'")
                    recommendations.append(f"Consider outlier treatment for column '{col}'")
            
            return DataPatternAnalysis(
                patterns=patterns,
                anomalies=anomalies,
                recommendations=recommendations,
                quality_issues=quality_issues
            )
            
        except Exception as e:
            logger.error(f"Error analyzing data patterns: {e}")
            return DataPatternAnalysis([], [], [], [])
    
    def learn_from_feedback(self, recommendation_id: str, feedback: Dict[str, Any]):
        """
        Learn from user feedback to improve future recommendations
        
        Args:
            recommendation_id: ID of the recommendation
            feedback: User feedback data
        """
        try:
            self.user_feedback[recommendation_id] = feedback
            
            # Update confidence scores based on feedback
            if feedback.get('helpful', False):
                # Positive feedback - increase confidence for similar patterns
                self._update_pattern_confidence(recommendation_id, 0.1)
            else:
                # Negative feedback - decrease confidence
                self._update_pattern_confidence(recommendation_id, -0.1)
            
            logger.info(f"Learned from feedback for recommendation {recommendation_id}")
            
        except Exception as e:
            logger.error(f"Error learning from feedback: {e}")
    
    def _analyze_data_patterns(self, data: pd.DataFrame) -> List[Transformation]:
        """Analyze data patterns and suggest transformations"""
        recommendations = []
        
        # Check for null values
        null_percentages = data.isnull().sum() / len(data)
        for col, null_pct in null_percentages.items():
            if null_pct > 0.5:
                recommendations.append(Transformation(
                    name='remove_high_null_column',
                    type='data_cleaning',
                    description=f'Remove column {col} with {null_pct:.1%} null values',
                    confidence=0.8,
                    parameters={'column': col, 'null_percentage': null_pct},
                    estimated_performance_impact=0.05
                ))
            elif null_pct > 0.1:
                recommendations.append(Transformation(
                    name='impute_missing_values',
                    type='data_cleaning',
                    description=f'Impute missing values in column {col}',
                    confidence=0.7,
                    parameters={'column': col, 'null_percentage': null_pct},
                    estimated_performance_impact=0.02
                ))
        
        return recommendations
    
    def _select_best_join_key(self, common_cols: set, left_dataset: Dataset, right_dataset: Dataset) -> str:
        """Select the best join key from common columns"""
        if not common_cols:
            return ''
        
        # Prefer columns with matching data types and reasonable cardinality
        best_key = list(common_cols)[0]  # Default to first common column
        
        for col in common_cols:
            left_type = left_dataset.schema.data_types.get(col)
            right_type = right_dataset.schema.data_types.get(col)
            
            # Prefer exact type matches
            if left_type == right_type:
                best_key = col
                break
        
        return best_key
    
    def _calculate_join_performance_score(self, left_dataset: Dataset, right_dataset: Dataset, join_type: str) -> float:
        """Calculate expected performance score for join operation"""
        # Simple heuristic based on dataset sizes
        total_size = left_dataset.size_mb + right_dataset.size_mb
        
        if total_size < 100:  # Small datasets
            return 0.9
        elif total_size < 1000:  # Medium datasets
            return 0.7
        else:  # Large datasets
            return 0.5
    
    def _is_datetime_pattern(self, series: pd.Series) -> bool:
        """Check if a string series contains datetime patterns"""
        try:
            # Try to parse a sample of values
            sample = series.dropna().head(10)
            if len(sample) == 0:
                return False
            
            parsed_count = 0
            for value in sample:
                try:
                    pd.to_datetime(str(value))
                    parsed_count += 1
                except:
                    pass
            
            return parsed_count / len(sample) > 0.8
        except:
            return False
    
    def _update_pattern_confidence(self, recommendation_id: str, adjustment: float):
        """Update confidence scores for similar patterns based on feedback"""
        # This would update the internal models based on feedback
        # For now, just log the adjustment
        logger.info(f"Adjusting confidence for pattern {recommendation_id} by {adjustment}")