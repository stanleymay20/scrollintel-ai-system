"""
Metadata Extractor for AI Data Readiness Platform

This module provides comprehensive metadata extraction and dataset profiling
capabilities with automatic cataloging and versioning support.
"""

print("DEBUG: Starting metadata_extractor.py execution")

import logging
import hashlib
import json
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import pandas as pd
import numpy as np
from sqlalchemy.orm import Session

print("DEBUG: Starting imports...")

# Simple fallback definitions
def get_settings():
    return None

class MetadataExtractionError(Exception):
    pass

@dataclass
class ColumnSchema:
    name: str
    data_type: str
    nullable: bool = True
    unique: bool = False
    primary_key: bool = False
    foreign_key: Optional[str] = None
    constraints: List[str] = None
    description: Optional[str] = None
    
    def __post_init__(self):
        if self.constraints is None:
            self.constraints = []

@dataclass
class Schema:
    dataset_id: str
    columns: List[ColumnSchema]
    primary_keys: List[str] = None
    foreign_keys: Dict[str, str] = None
    indexes: List[Dict[str, Any]] = None
    constraints: List[Dict[str, Any]] = None
    description: Optional[str] = None
    
    def __post_init__(self):
        if self.primary_keys is None:
            self.primary_keys = []
        if self.foreign_keys is None:
            self.foreign_keys = {}
        if self.indexes is None:
            self.indexes = []
        if self.constraints is None:
            self.constraints = []

class DatasetMetadata:
    pass

async def get_db_session():
    return None

print("DEBUG: Imports complete")


print("DEBUG: Defining ProfileLevel enum")

class ProfileLevel(Enum):
    """Levels of dataset profiling"""
    BASIC = "basic"
    STANDARD = "standard"
    COMPREHENSIVE = "comprehensive"

print("DEBUG: ProfileLevel defined successfully")


@dataclass
class DatasetProfile:
    """Comprehensive dataset profile"""
    dataset_id: str
    profile_level: ProfileLevel
    row_count: int
    column_count: int
    memory_usage: int
    missing_values_total: int
    missing_values_percentage: float
    duplicate_rows: int
    duplicate_rows_percentage: float
    data_types_distribution: Dict[str, int]
    column_profiles: List['ColumnProfile']
    correlations: Optional[Dict[str, Dict[str, float]]] = None
    statistical_summary: Optional[Dict[str, Any]] = None
    data_quality_score: Optional[float] = None
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()


@dataclass
class ColumnProfile:
    """Detailed column profile"""
    name: str
    data_type: str
    null_count: int
    null_percentage: float
    unique_count: int
    unique_percentage: float
    most_frequent_value: Any
    most_frequent_count: int
    min_value: Any = None
    max_value: Any = None
    mean_value: Any = None
    median_value: Any = None
    std_deviation: Any = None
    quartiles: Optional[Dict[str, Any]] = None
    value_distribution: Optional[Dict[str, int]] = None
    pattern_analysis: Optional[Dict[str, Any]] = None


@dataclass
class SchemaVersion:
    """Schema version information"""
    version_id: str
    dataset_id: str
    schema: Schema
    version_number: int
    created_at: datetime
    created_by: str
    change_summary: str
    parent_version_id: Optional[str] = None
    is_active: bool = True


class MetadataExtractor:
    """
    Comprehensive metadata extraction and dataset profiling engine
    with automatic cataloging and versioning capabilities.
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.logger = logging.getLogger(__name__)
        
    async def extract_comprehensive_metadata(
        self, 
        df: pd.DataFrame, 
        dataset_id: str,
        profile_level: ProfileLevel = ProfileLevel.STANDARD
    ) -> DatasetProfile:
        """
        Extract comprehensive metadata and profile from a dataset.
        
        Args:
            df: DataFrame to profile
            dataset_id: Unique identifier for the dataset
            profile_level: Level of profiling detail
            
        Returns:
            DatasetProfile: Comprehensive dataset profile
            
        Raises:
            MetadataExtractionError: If extraction fails
        """
        try:
            self.logger.info(f"Extracting metadata for dataset {dataset_id} at {profile_level.value} level")
            
            # Basic dataset statistics
            row_count = len(df)
            column_count = len(df.columns)
            memory_usage = df.memory_usage(deep=True).sum()
            
            # Missing values analysis
            missing_values_total = df.isnull().sum().sum()
            missing_values_percentage = (missing_values_total / (row_count * column_count)) * 100 if (row_count * column_count) > 0 else 0
            
            # Duplicate rows analysis
            duplicate_rows = df.duplicated().sum()
            duplicate_rows_percentage = (duplicate_rows / row_count) * 100 if row_count > 0 else 0
            
            # Data types distribution
            data_types_distribution = df.dtypes.value_counts().to_dict()
            data_types_distribution = {str(k): int(v) for k, v in data_types_distribution.items()}
            
            # Column profiles
            column_profiles = []
            for column in df.columns:
                column_profile = await self._profile_column(df[column], column, profile_level)
                column_profiles.append(column_profile)
            
            # Advanced analysis based on profile level
            correlations = None
            statistical_summary = None
            data_quality_score = None
            
            if profile_level in [ProfileLevel.STANDARD, ProfileLevel.COMPREHENSIVE]:
                correlations = await self._calculate_correlations(df)
                statistical_summary = await self._generate_statistical_summary(df)
                
            if profile_level == ProfileLevel.COMPREHENSIVE:
                data_quality_score = await self._calculate_data_quality_score(df, column_profiles)
            
            profile = DatasetProfile(
                dataset_id=dataset_id,
                profile_level=profile_level,
                row_count=row_count,
                column_count=column_count,
                memory_usage=memory_usage,
                missing_values_total=missing_values_total,
                missing_values_percentage=missing_values_percentage,
                duplicate_rows=duplicate_rows,
                duplicate_rows_percentage=duplicate_rows_percentage,
                data_types_distribution=data_types_distribution,
                column_profiles=column_profiles,
                correlations=correlations,
                statistical_summary=statistical_summary,
                data_quality_score=data_quality_score
            )
            
            self.logger.info(f"Successfully extracted metadata for dataset {dataset_id}")
            return profile
            
        except Exception as e:
            self.logger.error(f"Metadata extraction failed for dataset {dataset_id}: {str(e)}")
            raise MetadataExtractionError(f"Failed to extract metadata: {str(e)}")
    
    async def _profile_column(self, series: pd.Series, column_name: str, profile_level: ProfileLevel) -> ColumnProfile:
        """Profile individual column"""
        # Basic statistics
        null_count = series.isnull().sum()
        null_percentage = (null_count / len(series)) * 100 if len(series) > 0 else 0
        unique_count = series.nunique()
        unique_percentage = (unique_count / len(series)) * 100 if len(series) > 0 else 0
        
        # Most frequent value
        value_counts = series.value_counts()
        most_frequent_value = value_counts.index[0] if len(value_counts) > 0 else None
        most_frequent_count = value_counts.iloc[0] if len(value_counts) > 0 else 0
        
        # Initialize optional fields
        min_value = max_value = mean_value = median_value = std_deviation = None
        quartiles = None
        value_distribution = None
        pattern_analysis = None
        
        # Numeric column analysis
        if pd.api.types.is_numeric_dtype(series):
            min_value = series.min()
            max_value = series.max()
            mean_value = series.mean()
            median_value = series.median()
            std_deviation = series.std()
            
            if profile_level in [ProfileLevel.STANDARD, ProfileLevel.COMPREHENSIVE]:
                quartiles = {
                    'q1': series.quantile(0.25),
                    'q2': series.quantile(0.5),
                    'q3': series.quantile(0.75)
                }
        
        # Categorical/text column analysis
        elif pd.api.types.is_string_dtype(series) or pd.api.types.is_categorical_dtype(series):
            if profile_level == ProfileLevel.COMPREHENSIVE:
                # Value distribution for categorical data
                if unique_count <= 100:  # Only for reasonable number of categories
                    value_distribution = value_counts.head(20).to_dict()
                
                # Pattern analysis for text data
                if pd.api.types.is_string_dtype(series):
                    pattern_analysis = await self._analyze_text_patterns(series)
        
        return ColumnProfile(
            name=column_name,
            data_type=str(series.dtype),
            null_count=null_count,
            null_percentage=null_percentage,
            unique_count=unique_count,
            unique_percentage=unique_percentage,
            most_frequent_value=most_frequent_value,
            most_frequent_count=most_frequent_count,
            min_value=min_value,
            max_value=max_value,
            mean_value=mean_value,
            median_value=median_value,
            std_deviation=std_deviation,
            quartiles=quartiles,
            value_distribution=value_distribution,
            pattern_analysis=pattern_analysis
        )
    
    async def _analyze_text_patterns(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze patterns in text data"""
        import re
        
        non_null_series = series.dropna()
        if len(non_null_series) == 0:
            return {}
        
        # Length statistics
        lengths = non_null_series.str.len()
        
        # Pattern detection
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        phone_pattern = r'^\+?1?[-.\s]?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}$'
        url_pattern = r'^https?://[^\s/$.?#].[^\s]*$'
        
        email_matches = non_null_series.str.match(email_pattern).sum()
        phone_matches = non_null_series.str.match(phone_pattern).sum()
        url_matches = non_null_series.str.match(url_pattern).sum()
        
        return {
            'avg_length': lengths.mean(),
            'min_length': lengths.min(),
            'max_length': lengths.max(),
            'std_length': lengths.std(),
            'email_pattern_matches': email_matches,
            'phone_pattern_matches': phone_matches,
            'url_pattern_matches': url_matches,
            'contains_numbers': non_null_series.str.contains(r'\d').sum(),
            'contains_special_chars': non_null_series.str.contains(r'[^a-zA-Z0-9\s]').sum()
        }
    
    async def _calculate_correlations(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Calculate correlations between numeric columns"""
        try:
            numeric_df = df.select_dtypes(include=[np.number])
            if len(numeric_df.columns) < 2:
                return {}
            
            corr_matrix = numeric_df.corr()
            
            # Convert to nested dictionary format
            correlations = {}
            for col1 in corr_matrix.columns:
                correlations[col1] = {}
                for col2 in corr_matrix.columns:
                    if not pd.isna(corr_matrix.loc[col1, col2]):
                        correlations[col1][col2] = float(corr_matrix.loc[col1, col2])
            
            return correlations
        except Exception as e:
            self.logger.warning(f"Failed to calculate correlations: {str(e)}")
            return {}
    
    async def _generate_statistical_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive statistical summary"""
        try:
            numeric_df = df.select_dtypes(include=[np.number])
            categorical_df = df.select_dtypes(include=['object', 'category'])
            
            summary = {
                'numeric_columns': len(numeric_df.columns),
                'categorical_columns': len(categorical_df.columns),
                'datetime_columns': len(df.select_dtypes(include=['datetime']).columns),
                'boolean_columns': len(df.select_dtypes(include=['bool']).columns)
            }
            
            if len(numeric_df.columns) > 0:
                summary['numeric_summary'] = {
                    'mean_values': numeric_df.mean().to_dict(),
                    'median_values': numeric_df.median().to_dict(),
                    'std_values': numeric_df.std().to_dict(),
                    'skewness': numeric_df.skew().to_dict(),
                    'kurtosis': numeric_df.kurtosis().to_dict()
                }
            
            if len(categorical_df.columns) > 0:
                summary['categorical_summary'] = {}
                for col in categorical_df.columns:
                    summary['categorical_summary'][col] = {
                        'unique_count': categorical_df[col].nunique(),
                        'most_frequent': categorical_df[col].mode().iloc[0] if len(categorical_df[col].mode()) > 0 else None,
                        'frequency_distribution': categorical_df[col].value_counts().head(10).to_dict()
                    }
            
            return summary
        except Exception as e:
            self.logger.warning(f"Failed to generate statistical summary: {str(e)}")
            return {}
    
    async def _calculate_data_quality_score(self, df: pd.DataFrame, column_profiles: List[ColumnProfile]) -> float:
        """Calculate overall data quality score"""
        try:
            scores = []
            
            # Completeness score (based on missing values)
            total_cells = len(df) * len(df.columns)
            missing_cells = sum(profile.null_count for profile in column_profiles)
            completeness_score = (1 - missing_cells / total_cells) * 100 if total_cells > 0 else 0
            scores.append(completeness_score)
            
            # Uniqueness score (based on duplicate rows)
            duplicate_rows = df.duplicated().sum()
            uniqueness_score = (1 - duplicate_rows / len(df)) * 100 if len(df) > 0 else 100
            scores.append(uniqueness_score)
            
            # Consistency score (based on data type consistency)
            consistency_score = 100  # Start with perfect score
            for profile in column_profiles:
                # Penalize columns with mixed types or inconsistent patterns
                if profile.data_type == 'object' and profile.unique_percentage > 90:
                    consistency_score -= 5  # Potential data type inconsistency
            
            consistency_score = max(0, consistency_score)
            scores.append(consistency_score)
            
            # Validity score (based on pattern analysis)
            validity_score = 100
            for profile in column_profiles:
                if profile.pattern_analysis:
                    # Check for potential data validity issues
                    if profile.pattern_analysis.get('contains_special_chars', 0) > len(df) * 0.5:
                        validity_score -= 10
            
            validity_score = max(0, validity_score)
            scores.append(validity_score)
            
            # Overall score is weighted average
            overall_score = sum(scores) / len(scores)
            return round(overall_score, 2)
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate data quality score: {str(e)}")
            return 0.0