"""
Data Profiling and Baseline Establishment Engine
Creates comprehensive data profiles and establishes quality baselines
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import func, and_
import logging
import json
from dataclasses import dataclass
import re
from collections import Counter

from ..models.data_quality_models import DataProfile, QualityRule, QualityRuleType, Severity

logger = logging.getLogger(__name__)

@dataclass
class ProfilingConfig:
    """Configuration for data profiling"""
    sample_size: int = 10000
    pattern_detection: bool = True
    statistical_analysis: bool = True
    categorical_threshold: float = 0.5  # If unique values < 50% of total, treat as categorical
    min_frequency_threshold: int = 5  # Minimum frequency for pattern detection

@dataclass
class DataQualityBaseline:
    """Comprehensive data quality baseline"""
    completeness_baseline: float
    validity_baseline: float
    consistency_baseline: float
    uniqueness_baseline: float
    statistical_baseline: Dict
    pattern_baseline: Dict
    recommended_rules: List[Dict]

class DataProfiler:
    """Comprehensive data profiling engine"""
    
    def __init__(self, db_session: Session, config: ProfilingConfig = None):
        self.db_session = db_session
        self.config = config or ProfilingConfig()
        self.logger = logging.getLogger(__name__)
    
    def create_comprehensive_profile(self, data: pd.DataFrame, table_name: str, 
                                   pipeline_id: str = None) -> List[DataProfile]:
        """
        Create comprehensive profiles for all columns in a dataset
        
        Args:
            data: DataFrame to profile
            table_name: Name of the table/dataset
            pipeline_id: Associated pipeline ID
            
        Returns:
            List of DataProfile objects for each column
        """
        profiles = []
        
        # Sample data if too large
        if len(data) > self.config.sample_size:
            sampled_data = data.sample(n=self.config.sample_size, random_state=42)
            self.logger.info(f"Sampling {self.config.sample_size} rows from {len(data)} total rows")
        else:
            sampled_data = data
        
        for column in data.columns:
            try:
                profile = self._profile_column(sampled_data, table_name, column)
                profiles.append(profile)
                
                # Save to database
                self.db_session.add(profile)
                
            except Exception as e:
                self.logger.error(f"Error profiling column {column}: {str(e)}")
        
        self.db_session.commit()
        return profiles
    
    def _profile_column(self, data: pd.DataFrame, table_name: str, column_name: str) -> DataProfile:
        """Profile a single column comprehensively"""
        column_data = data[column_name]
        
        # Basic statistics
        record_count = len(column_data)
        null_count = column_data.isnull().sum()
        unique_count = column_data.nunique()
        
        profile = DataProfile(
            table_name=table_name,
            column_name=column_name,
            data_type=str(column_data.dtype),
            record_count=record_count,
            null_count=null_count,
            unique_count=unique_count
        )
        
        # Completeness score
        profile.completeness_score = (record_count - null_count) / record_count * 100 if record_count > 0 else 0
        
        # Data type specific profiling
        if self._is_numeric_column(column_data):
            self._profile_numeric_column(column_data, profile)
        elif self._is_datetime_column(column_data):
            self._profile_datetime_column(column_data, profile)
        else:
            self._profile_text_column(column_data, profile)
        
        # Pattern analysis
        if self.config.pattern_detection:
            self._analyze_patterns(column_data, profile)
        
        # Quality scores
        self._calculate_quality_scores(column_data, profile)
        
        return profile
    
    def _is_numeric_column(self, column_data: pd.Series) -> bool:
        """Check if column contains numeric data"""
        if column_data.dtype in ['int64', 'float64', 'int32', 'float32']:
            return True
        
        # Try to convert to numeric
        numeric_data = pd.to_numeric(column_data, errors='coerce')
        non_null_count = column_data.dropna().shape[0]
        numeric_count = numeric_data.dropna().shape[0]
        
        # If >80% can be converted to numeric, consider it numeric
        return numeric_count / non_null_count > 0.8 if non_null_count > 0 else False
    
    def _is_datetime_column(self, column_data: pd.Series) -> bool:
        """Check if column contains datetime data"""
        if pd.api.types.is_datetime64_any_dtype(column_data):
            return True
        
        # Try to parse as datetime
        try:
            pd.to_datetime(column_data.dropna().head(100))
            return True
        except:
            return False
    
    def _profile_numeric_column(self, column_data: pd.Series, profile: DataProfile):
        """Profile numeric column"""
        numeric_data = pd.to_numeric(column_data, errors='coerce').dropna()
        
        if len(numeric_data) > 0:
            profile.min_value = float(numeric_data.min())
            profile.max_value = float(numeric_data.max())
            profile.mean_value = float(numeric_data.mean())
            profile.median_value = float(numeric_data.median())
            profile.std_deviation = float(numeric_data.std())
            
            # Additional statistics
            percentiles = numeric_data.quantile([0.25, 0.75]).to_dict()
            profile.value_distribution = {
                "q25": percentiles[0.25],
                "q75": percentiles[0.75],
                "iqr": percentiles[0.75] - percentiles[0.25],
                "skewness": float(numeric_data.skew()),
                "kurtosis": float(numeric_data.kurtosis())
            }
    
    def _profile_datetime_column(self, column_data: pd.Series, profile: DataProfile):
        """Profile datetime column"""
        try:
            datetime_data = pd.to_datetime(column_data, errors='coerce').dropna()
            
            if len(datetime_data) > 0:
                profile.min_value = datetime_data.min().timestamp()
                profile.max_value = datetime_data.max().timestamp()
                
                # Calculate frequency patterns
                date_range = datetime_data.max() - datetime_data.min()
                profile.common_patterns = {
                    "date_range_days": date_range.days,
                    "min_date": datetime_data.min().isoformat(),
                    "max_date": datetime_data.max().isoformat(),
                    "frequency_analysis": self._analyze_datetime_frequency(datetime_data)
                }
                
        except Exception as e:
            self.logger.warning(f"Error profiling datetime column: {str(e)}")
    
    def _profile_text_column(self, column_data: pd.Series, profile: DataProfile):
        """Profile text/categorical column"""
        text_data = column_data.dropna().astype(str)
        
        if len(text_data) > 0:
            # Length statistics
            lengths = text_data.str.len()
            profile.common_patterns = {
                "avg_length": float(lengths.mean()),
                "min_length": int(lengths.min()),
                "max_length": int(lengths.max()),
                "length_std": float(lengths.std())
            }
            
            # Value frequency analysis
            value_counts = text_data.value_counts()
            
            # Store top values if not too many unique values
            if profile.unique_count < len(text_data) * self.config.categorical_threshold:
                profile.most_frequent_values = value_counts.head(20).to_dict()
                
                # Value distribution
                distribution = (value_counts / len(text_data)).to_dict()
                profile.value_distribution = distribution
    
    def _analyze_patterns(self, column_data: pd.Series, profile: DataProfile):
        """Analyze data patterns in the column"""
        text_data = column_data.dropna().astype(str)
        
        if len(text_data) == 0:
            return
        
        patterns = {}
        
        # Common format patterns
        format_patterns = {
            "email": r'^[\w\.-]+@[\w\.-]+\.\w+$',
            "phone": r'^\+?[\d\s\-\(\)]{10,}$',
            "url": r'^https?://[\w\.-]+',
            "ip_address": r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$',
            "credit_card": r'^\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}$',
            "ssn": r'^\d{3}-?\d{2}-?\d{4}$',
            "uuid": r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
            "date_iso": r'^\d{4}-\d{2}-\d{2}',
            "numeric_string": r'^\d+$',
            "alphanumeric": r'^[a-zA-Z0-9]+$',
            "alphabetic": r'^[a-zA-Z\s]+$'
        }
        
        for pattern_name, pattern_regex in format_patterns.items():
            matches = text_data.str.match(pattern_regex, case=False, na=False).sum()
            if matches > self.config.min_frequency_threshold:
                patterns[pattern_name] = {
                    "count": int(matches),
                    "percentage": float(matches / len(text_data) * 100)
                }
        
        # Character set analysis
        char_analysis = self._analyze_character_sets(text_data)
        patterns.update(char_analysis)
        
        # Length pattern analysis
        length_patterns = self._analyze_length_patterns(text_data)
        patterns.update(length_patterns)
        
        profile.format_patterns = patterns
    
    def _analyze_character_sets(self, text_data: pd.Series) -> Dict:
        """Analyze character set usage in text data"""
        char_patterns = {}
        
        # Count different character types
        has_digits = text_data.str.contains(r'\d', na=False).sum()
        has_letters = text_data.str.contains(r'[a-zA-Z]', na=False).sum()
        has_special = text_data.str.contains(r'[^a-zA-Z0-9\s]', na=False).sum()
        has_spaces = text_data.str.contains(r'\s', na=False).sum()
        
        total_count = len(text_data)
        
        char_patterns["character_analysis"] = {
            "contains_digits": {"count": int(has_digits), "percentage": float(has_digits / total_count * 100)},
            "contains_letters": {"count": int(has_letters), "percentage": float(has_letters / total_count * 100)},
            "contains_special": {"count": int(has_special), "percentage": float(has_special / total_count * 100)},
            "contains_spaces": {"count": int(has_spaces), "percentage": float(has_spaces / total_count * 100)}
        }
        
        return char_patterns
    
    def _analyze_length_patterns(self, text_data: pd.Series) -> Dict:
        """Analyze length patterns in text data"""
        lengths = text_data.str.len()
        length_counts = lengths.value_counts().head(10)
        
        return {
            "common_lengths": length_counts.to_dict(),
            "length_distribution": {
                "mean": float(lengths.mean()),
                "std": float(lengths.std()),
                "mode": int(lengths.mode().iloc[0]) if not lengths.mode().empty else 0
            }
        }
    
    def _analyze_datetime_frequency(self, datetime_data: pd.Series) -> Dict:
        """Analyze frequency patterns in datetime data"""
        # Day of week analysis
        dow_counts = datetime_data.dt.dayofweek.value_counts().sort_index()
        
        # Hour analysis (if time component exists)
        hour_counts = datetime_data.dt.hour.value_counts().sort_index()
        
        # Month analysis
        month_counts = datetime_data.dt.month.value_counts().sort_index()
        
        return {
            "day_of_week_distribution": dow_counts.to_dict(),
            "hour_distribution": hour_counts.to_dict() if hour_counts.sum() > 0 else {},
            "month_distribution": month_counts.to_dict()
        }
    
    def _calculate_quality_scores(self, column_data: pd.Series, profile: DataProfile):
        """Calculate quality scores for the column"""
        # Completeness already calculated
        
        # Validity score (based on pattern consistency)
        validity_score = 100.0
        if profile.format_patterns:
            # If we found strong patterns, check consistency
            max_pattern_pct = max([
                p.get("percentage", 0) for p in profile.format_patterns.values() 
                if isinstance(p, dict) and "percentage" in p
            ] + [0])
            
            if max_pattern_pct > 80:
                validity_score = max_pattern_pct
        
        profile.validity_score = validity_score
        
        # Consistency score (placeholder - would need reference data)
        profile.consistency_score = 100.0
    
    def establish_quality_baseline(self, profiles: List[DataProfile], 
                                 table_name: str, pipeline_id: str = None) -> DataQualityBaseline:
        """
        Establish quality baselines and recommend quality rules
        
        Args:
            profiles: List of data profiles for the dataset
            table_name: Name of the table
            pipeline_id: Associated pipeline ID
            
        Returns:
            DataQualityBaseline with recommended rules
        """
        # Calculate overall baselines
        completeness_scores = [p.completeness_score for p in profiles if p.completeness_score is not None]
        validity_scores = [p.validity_score for p in profiles if p.validity_score is not None]
        consistency_scores = [p.consistency_score for p in profiles if p.consistency_score is not None]
        
        completeness_baseline = np.mean(completeness_scores) if completeness_scores else 100.0
        validity_baseline = np.mean(validity_scores) if validity_scores else 100.0
        consistency_baseline = np.mean(consistency_scores) if consistency_scores else 100.0
        
        # Calculate uniqueness baseline
        uniqueness_scores = []
        for profile in profiles:
            if profile.record_count > 0:
                uniqueness_ratio = profile.unique_count / profile.record_count
                uniqueness_scores.append(uniqueness_ratio * 100)
        
        uniqueness_baseline = np.mean(uniqueness_scores) if uniqueness_scores else 100.0
        
        # Statistical baseline
        statistical_baseline = self._create_statistical_baseline(profiles)
        
        # Pattern baseline
        pattern_baseline = self._create_pattern_baseline(profiles)
        
        # Generate recommended rules
        recommended_rules = self._generate_recommended_rules(profiles, table_name, pipeline_id)
        
        return DataQualityBaseline(
            completeness_baseline=completeness_baseline,
            validity_baseline=validity_baseline,
            consistency_baseline=consistency_baseline,
            uniqueness_baseline=uniqueness_baseline,
            statistical_baseline=statistical_baseline,
            pattern_baseline=pattern_baseline,
            recommended_rules=recommended_rules
        )
    
    def _create_statistical_baseline(self, profiles: List[DataProfile]) -> Dict:
        """Create statistical baseline from profiles"""
        baseline = {}
        
        for profile in profiles:
            if profile.mean_value is not None and profile.std_deviation is not None:
                baseline[profile.column_name] = {
                    "mean": profile.mean_value,
                    "std": profile.std_deviation,
                    "min": profile.min_value,
                    "max": profile.max_value,
                    "median": profile.median_value
                }
        
        return baseline
    
    def _create_pattern_baseline(self, profiles: List[DataProfile]) -> Dict:
        """Create pattern baseline from profiles"""
        baseline = {}
        
        for profile in profiles:
            if profile.format_patterns:
                # Find the dominant pattern
                dominant_patterns = {}
                for pattern_name, pattern_data in profile.format_patterns.items():
                    if isinstance(pattern_data, dict) and pattern_data.get("percentage", 0) > 70:
                        dominant_patterns[pattern_name] = pattern_data
                
                if dominant_patterns:
                    baseline[profile.column_name] = dominant_patterns
        
        return baseline
    
    def _generate_recommended_rules(self, profiles: List[DataProfile], 
                                  table_name: str, pipeline_id: str = None) -> List[Dict]:
        """Generate recommended quality rules based on profiles"""
        recommended_rules = []
        
        for profile in profiles:
            column_name = profile.column_name
            
            # Completeness rule
            if profile.completeness_score < 100:
                threshold = max(0.8, profile.completeness_score / 100 * 0.9)  # 90% of current completeness
                recommended_rules.append({
                    "name": f"Completeness check for {column_name}",
                    "rule_type": QualityRuleType.COMPLETENESS.value,
                    "target_table": table_name,
                    "target_column": column_name,
                    "target_pipeline_id": pipeline_id,
                    "threshold_value": threshold,
                    "severity": Severity.MEDIUM.value,
                    "description": f"Ensure {column_name} has at least {threshold*100:.1f}% non-null values"
                })
            
            # Uniqueness rule for high-cardinality columns
            if profile.record_count > 0:
                uniqueness_ratio = profile.unique_count / profile.record_count
                if uniqueness_ratio > 0.95:  # Likely a unique identifier
                    recommended_rules.append({
                        "name": f"Uniqueness check for {column_name}",
                        "rule_type": QualityRuleType.UNIQUENESS.value,
                        "target_table": table_name,
                        "target_column": column_name,
                        "target_pipeline_id": pipeline_id,
                        "threshold_value": 1.0,
                        "severity": Severity.HIGH.value,
                        "description": f"Ensure {column_name} values are unique"
                    })
            
            # Statistical rules for numeric columns
            if profile.mean_value is not None and profile.std_deviation is not None:
                recommended_rules.append({
                    "name": f"Statistical validation for {column_name}",
                    "rule_type": QualityRuleType.STATISTICAL.value,
                    "target_table": table_name,
                    "target_column": column_name,
                    "target_pipeline_id": pipeline_id,
                    "conditions": {
                        "expected_mean": profile.mean_value,
                        "expected_std": profile.std_deviation,
                        "tolerance": 0.2  # 20% tolerance
                    },
                    "severity": Severity.MEDIUM.value,
                    "description": f"Validate statistical properties of {column_name}"
                })
            
            # Pattern validation rules
            if profile.format_patterns:
                for pattern_name, pattern_data in profile.format_patterns.items():
                    if isinstance(pattern_data, dict) and pattern_data.get("percentage", 0) > 80:
                        # Strong pattern detected
                        if pattern_name in ["email", "phone", "url", "ip_address"]:
                            recommended_rules.append({
                                "name": f"{pattern_name.title()} format validation for {column_name}",
                                "rule_type": QualityRuleType.VALIDITY.value,
                                "target_table": table_name,
                                "target_column": column_name,
                                "target_pipeline_id": pipeline_id,
                                "conditions": {"format_type": pattern_name},
                                "threshold_value": 0.95,
                                "severity": Severity.MEDIUM.value,
                                "description": f"Validate {pattern_name} format in {column_name}"
                            })
            
            # Range validation for numeric columns
            if profile.min_value is not None and profile.max_value is not None:
                # Add some buffer to the range
                range_buffer = (profile.max_value - profile.min_value) * 0.1
                min_allowed = profile.min_value - range_buffer
                max_allowed = profile.max_value + range_buffer
                
                recommended_rules.append({
                    "name": f"Range validation for {column_name}",
                    "rule_type": QualityRuleType.VALIDITY.value,
                    "target_table": table_name,
                    "target_column": column_name,
                    "target_pipeline_id": pipeline_id,
                    "conditions": {
                        "min_value": min_allowed,
                        "max_value": max_allowed
                    },
                    "threshold_value": 0.95,
                    "severity": Severity.MEDIUM.value,
                    "description": f"Validate {column_name} is within expected range"
                })
        
        return recommended_rules
    
    def update_profile_baseline(self, table_name: str, column_name: str, 
                              new_data: pd.DataFrame) -> DataProfile:
        """Update an existing profile with new data"""
        # Get existing profile
        existing_profile = self.db_session.query(DataProfile).filter_by(
            table_name=table_name,
            column_name=column_name
        ).order_by(DataProfile.created_at.desc()).first()
        
        if not existing_profile:
            # Create new profile if none exists
            return self._profile_column(new_data, table_name, column_name)
        
        # Create new profile with updated data
        new_profile = self._profile_column(new_data, table_name, column_name)
        
        # Blend with existing profile (weighted average)
        blended_profile = self._blend_profiles(existing_profile, new_profile)
        
        # Save blended profile
        self.db_session.add(blended_profile)
        self.db_session.commit()
        
        return blended_profile
    
    def _blend_profiles(self, existing: DataProfile, new: DataProfile, 
                       weight: float = 0.7) -> DataProfile:
        """Blend existing and new profiles with weighted average"""
        # Create blended profile
        blended = DataProfile(
            table_name=existing.table_name,
            column_name=existing.column_name,
            data_type=new.data_type,
            record_count=new.record_count,
            null_count=new.null_count,
            unique_count=new.unique_count
        )
        
        # Blend numeric statistics
        if existing.mean_value is not None and new.mean_value is not None:
            blended.mean_value = existing.mean_value * weight + new.mean_value * (1 - weight)
            blended.std_deviation = existing.std_deviation * weight + new.std_deviation * (1 - weight)
            blended.min_value = min(existing.min_value or float('inf'), new.min_value or float('inf'))
            blended.max_value = max(existing.max_value or float('-inf'), new.max_value or float('-inf'))
        else:
            blended.mean_value = new.mean_value
            blended.std_deviation = new.std_deviation
            blended.min_value = new.min_value
            blended.max_value = new.max_value
        
        # Blend quality scores
        blended.completeness_score = (existing.completeness_score or 0) * weight + (new.completeness_score or 0) * (1 - weight)
        blended.validity_score = (existing.validity_score or 0) * weight + (new.validity_score or 0) * (1 - weight)
        blended.consistency_score = (existing.consistency_score or 0) * weight + (new.consistency_score or 0) * (1 - weight)
        
        # Use new patterns and distributions (they represent current state)
        blended.format_patterns = new.format_patterns
        blended.value_distribution = new.value_distribution
        blended.common_patterns = new.common_patterns
        
        return blended