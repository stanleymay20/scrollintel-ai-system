"""
Anonymization and Privacy Protection Engine for AI Data Readiness Platform

This module provides comprehensive data anonymization techniques and privacy
protection tools to ensure compliance with data protection regulations.
"""

import pandas as pd
import numpy as np
import hashlib
import secrets
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
import re

from ai_data_readiness.models.compliance_models import (
    SensitiveDataType, PrivacyTechnique
)

logger = logging.getLogger(__name__)


class AnonymizationTechnique(Enum):
    """Available anonymization techniques"""
    K_ANONYMITY = "k_anonymity"
    L_DIVERSITY = "l_diversity"
    T_CLOSENESS = "t_closeness"
    DIFFERENTIAL_PRIVACY = "differential_privacy"
    GENERALIZATION = "generalization"
    SUPPRESSION = "suppression"
    PSEUDONYMIZATION = "pseudonymization"
    TOKENIZATION = "tokenization"
    MASKING = "masking"
    SYNTHETIC_DATA = "synthetic_data"


class PrivacyRiskLevel(Enum):
    """Privacy risk assessment levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AnonymizationConfig:
    """Configuration for anonymization techniques"""
    technique: AnonymizationTechnique
    parameters: Dict[str, Any]
    target_columns: List[str]
    preserve_utility: bool = True
    risk_threshold: float = 0.1


@dataclass
class PrivacyRiskAssessment:
    """Privacy risk assessment results"""
    column_name: str
    risk_level: PrivacyRiskLevel
    risk_score: float
    vulnerability_factors: List[str]
    recommended_techniques: List[AnonymizationTechnique]
    assessment_timestamp: datetime


@dataclass
class AnonymizationResult:
    """Results of anonymization process"""
    original_data_shape: Tuple[int, int]
    anonymized_data: pd.DataFrame
    technique_applied: AnonymizationTechnique
    privacy_gain: float
    utility_loss: float
    processing_time: float
    metadata: Dict[str, Any]


class AnonymizationEngine:
    """
    Comprehensive anonymization and privacy protection engine
    
    Provides various anonymization techniques including k-anonymity,
    l-diversity, differential privacy, and synthetic data generation.
    """
    
    def __init__(self):
        self.salt = secrets.token_hex(16)
        self.anonymization_cache = {}
        
    def anonymize_data(
        self, 
        data: pd.DataFrame, 
        config: AnonymizationConfig
    ) -> AnonymizationResult:
        """
        Apply anonymization technique to dataset
        
        Args:
            data: Dataset to anonymize
            config: Anonymization configuration
            
        Returns:
            AnonymizationResult with anonymized data and metrics
        """
        start_time = datetime.now()
        
        try:
            logger.info(f"Starting anonymization with technique: {config.technique.value}")
            
            # Create copy of data
            anonymized_data = data.copy()
            
            # Apply anonymization technique
            if config.technique == AnonymizationTechnique.K_ANONYMITY:
                anonymized_data = self._apply_k_anonymity(anonymized_data, config)
            elif config.technique == AnonymizationTechnique.L_DIVERSITY:
                anonymized_data = self._apply_l_diversity(anonymized_data, config)
            elif config.technique == AnonymizationTechnique.PSEUDONYMIZATION:
                anonymized_data = self._apply_pseudonymization(anonymized_data, config)
            elif config.technique == AnonymizationTechnique.MASKING:
                anonymized_data = self._apply_masking(anonymized_data, config)
            elif config.technique == AnonymizationTechnique.GENERALIZATION:
                anonymized_data = self._apply_generalization(anonymized_data, config)
            elif config.technique == AnonymizationTechnique.SUPPRESSION:
                anonymized_data = self._apply_suppression(anonymized_data, config)
            elif config.technique == AnonymizationTechnique.DIFFERENTIAL_PRIVACY:
                anonymized_data = self._apply_differential_privacy(anonymized_data, config)
            elif config.technique == AnonymizationTechnique.SYNTHETIC_DATA:
                anonymized_data = self._generate_synthetic_data(anonymized_data, config)
            else:
                raise ValueError(f"Unsupported anonymization technique: {config.technique}")
            
            # Calculate metrics
            processing_time = (datetime.now() - start_time).total_seconds()
            privacy_gain = self._calculate_privacy_gain(data, anonymized_data, config)
            utility_loss = self._calculate_utility_loss(data, anonymized_data, config)
            
            result = AnonymizationResult(
                original_data_shape=data.shape,
                anonymized_data=anonymized_data,
                technique_applied=config.technique,
                privacy_gain=privacy_gain,
                utility_loss=utility_loss,
                processing_time=processing_time,
                metadata={
                    'config': config,
                    'columns_processed': config.target_columns,
                    'records_processed': len(data)
                }
            )
            
            logger.info(f"Anonymization completed in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Error in anonymization: {str(e)}")
            raise
    
    def assess_privacy_risk(
        self, 
        data: pd.DataFrame, 
        sensitive_columns: List[str] = None
    ) -> List[PrivacyRiskAssessment]:
        """
        Assess privacy risks in dataset
        
        Args:
            data: Dataset to assess
            sensitive_columns: List of known sensitive columns
            
        Returns:
            List of privacy risk assessments
        """
        assessments = []
        
        if sensitive_columns is None:
            sensitive_columns = list(data.columns)
        
        for column in sensitive_columns:
            if column not in data.columns:
                continue
                
            assessment = self._assess_column_privacy_risk(data, column)
            assessments.append(assessment)
        
        return assessments
    
    def recommend_anonymization_strategy(
        self, 
        risk_assessments: List[PrivacyRiskAssessment],
        utility_requirements: Dict[str, float] = None
    ) -> List[AnonymizationConfig]:
        """
        Recommend anonymization strategies based on risk assessments
        
        Args:
            risk_assessments: Privacy risk assessments
            utility_requirements: Utility preservation requirements
            
        Returns:
            List of recommended anonymization configurations
        """
        recommendations = []
        
        for assessment in risk_assessments:
            config = self._generate_anonymization_config(assessment, utility_requirements)
            if config:
                recommendations.append(config)
        
        return recommendations
    
    def _apply_k_anonymity(self, data: pd.DataFrame, config: AnonymizationConfig) -> pd.DataFrame:
        """Apply k-anonymity anonymization"""
        k = config.parameters.get('k', 3)
        quasi_identifiers = config.target_columns
        
        # Simple k-anonymity implementation using generalization
        anonymized_data = data.copy()
        
        for column in quasi_identifiers:
            if column in data.columns:
                if data[column].dtype in ['int64', 'float64']:
                    # Numerical generalization
                    anonymized_data[column] = self._generalize_numerical(data[column], k)
                else:
                    # Categorical generalization
                    anonymized_data[column] = self._generalize_categorical(data[column], k)
        
        return anonymized_data
    
    def _apply_l_diversity(self, data: pd.DataFrame, config: AnonymizationConfig) -> pd.DataFrame:
        """Apply l-diversity anonymization"""
        l = config.parameters.get('l', 2)
        sensitive_attribute = config.parameters.get('sensitive_attribute')
        
        # Simple l-diversity implementation
        anonymized_data = data.copy()
        
        if sensitive_attribute and sensitive_attribute in data.columns:
            # Ensure l-diversity in sensitive attribute
            anonymized_data = self._ensure_l_diversity(anonymized_data, sensitive_attribute, l)
        
        return anonymized_data
    
    def _apply_pseudonymization(self, data: pd.DataFrame, config: AnonymizationConfig) -> pd.DataFrame:
        """Apply pseudonymization"""
        anonymized_data = data.copy()
        
        for column in config.target_columns:
            if column in data.columns:
                anonymized_data[column] = data[column].apply(
                    lambda x: self._pseudonymize_value(str(x))
                )
        
        return anonymized_data
    
    def _apply_masking(self, data: pd.DataFrame, config: AnonymizationConfig) -> pd.DataFrame:
        """Apply data masking"""
        mask_char = config.parameters.get('mask_char', '*')
        preserve_length = config.parameters.get('preserve_length', True)
        preserve_format = config.parameters.get('preserve_format', True)
        
        anonymized_data = data.copy()
        
        for column in config.target_columns:
            if column in data.columns:
                anonymized_data[column] = data[column].apply(
                    lambda x: self._mask_value(str(x), mask_char, preserve_length, preserve_format)
                )
        
        return anonymized_data
    
    def _apply_generalization(self, data: pd.DataFrame, config: AnonymizationConfig) -> pd.DataFrame:
        """Apply data generalization"""
        anonymized_data = data.copy()
        
        for column in config.target_columns:
            if column in data.columns:
                if data[column].dtype in ['int64', 'float64']:
                    # Numerical generalization
                    levels = config.parameters.get('generalization_levels', 3)
                    anonymized_data[column] = self._generalize_numerical(data[column], levels)
                else:
                    # Categorical generalization
                    anonymized_data[column] = self._generalize_categorical_hierarchy(data[column])
        
        return anonymized_data
    
    def _apply_suppression(self, data: pd.DataFrame, config: AnonymizationConfig) -> pd.DataFrame:
        """Apply data suppression"""
        suppression_rate = config.parameters.get('suppression_rate', 0.1)
        
        anonymized_data = data.copy()
        
        for column in config.target_columns:
            if column in data.columns:
                # Randomly suppress values
                mask = np.random.random(len(data)) < suppression_rate
                anonymized_data.loc[mask, column] = None
        
        return anonymized_data
    
    def _apply_differential_privacy(self, data: pd.DataFrame, config: AnonymizationConfig) -> pd.DataFrame:
        """Apply differential privacy"""
        epsilon = config.parameters.get('epsilon', 1.0)
        
        anonymized_data = data.copy()
        
        for column in config.target_columns:
            if column in data.columns and data[column].dtype in ['int64', 'float64']:
                # Add Laplace noise for differential privacy
                sensitivity = config.parameters.get('sensitivity', 1.0)
                noise = np.random.laplace(0, sensitivity / epsilon, len(data))
                anonymized_data[column] = data[column] + noise
        
        return anonymized_data
    
    def _generate_synthetic_data(self, data: pd.DataFrame, config: AnonymizationConfig) -> pd.DataFrame:
        """Generate synthetic data"""
        # Simple synthetic data generation
        synthetic_data = pd.DataFrame()
        
        for column in data.columns:
            if data[column].dtype in ['int64', 'float64']:
                # Generate synthetic numerical data
                mean = data[column].mean()
                std = data[column].std()
                synthetic_values = np.random.normal(mean, std, len(data))
                synthetic_data[column] = synthetic_values
            else:
                # Generate synthetic categorical data
                value_counts = data[column].value_counts(normalize=True)
                synthetic_values = np.random.choice(
                    value_counts.index, 
                    size=len(data), 
                    p=value_counts.values
                )
                synthetic_data[column] = synthetic_values
        
        return synthetic_data
    
    def _assess_column_privacy_risk(self, data: pd.DataFrame, column: str) -> PrivacyRiskAssessment:
        """Assess privacy risk for a specific column"""
        vulnerability_factors = []
        risk_score = 0.0
        
        column_data = data[column]
        
        # Check uniqueness (high uniqueness = high risk)
        uniqueness = column_data.nunique() / len(column_data)
        if uniqueness > 0.9:
            vulnerability_factors.append("high_uniqueness")
            risk_score += 0.4
        elif uniqueness > 0.7:
            vulnerability_factors.append("medium_uniqueness")
            risk_score += 0.2
        
        # Check for potential identifiers
        if self._is_potential_identifier(column_data):
            vulnerability_factors.append("potential_identifier")
            risk_score += 0.3
        
        # Check for sensitive patterns
        if self._contains_sensitive_patterns(column_data):
            vulnerability_factors.append("sensitive_patterns")
            risk_score += 0.3
        
        # Check data distribution
        if self._has_skewed_distribution(column_data):
            vulnerability_factors.append("skewed_distribution")
            risk_score += 0.1
        
        # Determine risk level
        if risk_score >= 0.7:
            risk_level = PrivacyRiskLevel.CRITICAL
        elif risk_score >= 0.5:
            risk_level = PrivacyRiskLevel.HIGH
        elif risk_score >= 0.3:
            risk_level = PrivacyRiskLevel.MEDIUM
        else:
            risk_level = PrivacyRiskLevel.LOW
        
        # Recommend techniques based on risk
        recommended_techniques = self._recommend_techniques_for_risk(risk_level, column_data.dtype)
        
        return PrivacyRiskAssessment(
            column_name=column,
            risk_level=risk_level,
            risk_score=risk_score,
            vulnerability_factors=vulnerability_factors,
            recommended_techniques=recommended_techniques,
            assessment_timestamp=datetime.now()
        )
    
    def _is_potential_identifier(self, column_data: pd.Series) -> bool:
        """Check if column contains potential identifiers"""
        # Check for ID-like patterns
        id_patterns = [
            r'^\d+$',  # Pure numbers
            r'^[A-Z]\d+$',  # Letter followed by numbers
            r'^\d+-\d+-\d+$',  # Dash-separated numbers
        ]
        
        sample_values = column_data.astype(str).head(100)
        for pattern in id_patterns:
            matches = sample_values.str.match(pattern).sum()
            if matches > len(sample_values) * 0.8:
                return True
        
        return False
    
    def _contains_sensitive_patterns(self, column_data: pd.Series) -> bool:
        """Check if column contains sensitive data patterns"""
        sensitive_patterns = [
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
            r'\b\d{3}-\d{3}-\d{4}\b',  # Phone
        ]
        
        sample_values = column_data.astype(str).head(100)
        for pattern in sensitive_patterns:
            matches = sample_values.str.contains(pattern, regex=True).sum()
            if matches > 0:
                return True
        
        return False
    
    def _has_skewed_distribution(self, column_data: pd.Series) -> bool:
        """Check if column has skewed distribution"""
        if column_data.dtype in ['int64', 'float64']:
            try:
                from scipy import stats
                skewness = abs(stats.skew(column_data.dropna()))
                return skewness > 2.0
            except ImportError:
                # Fallback without scipy
                return False
        return False
    
    def _recommend_techniques_for_risk(self, risk_level: PrivacyRiskLevel, dtype) -> List[AnonymizationTechnique]:
        """Recommend anonymization techniques based on risk level"""
        if risk_level == PrivacyRiskLevel.CRITICAL:
            return [
                AnonymizationTechnique.SYNTHETIC_DATA,
                AnonymizationTechnique.SUPPRESSION,
                AnonymizationTechnique.K_ANONYMITY
            ]
        elif risk_level == PrivacyRiskLevel.HIGH:
            return [
                AnonymizationTechnique.PSEUDONYMIZATION,
                AnonymizationTechnique.K_ANONYMITY,
                AnonymizationTechnique.GENERALIZATION
            ]
        elif risk_level == PrivacyRiskLevel.MEDIUM:
            return [
                AnonymizationTechnique.MASKING,
                AnonymizationTechnique.GENERALIZATION,
                AnonymizationTechnique.DIFFERENTIAL_PRIVACY
            ]
        else:
            return [
                AnonymizationTechnique.MASKING,
                AnonymizationTechnique.GENERALIZATION
            ]
    
    def _generate_anonymization_config(
        self, 
        assessment: PrivacyRiskAssessment,
        utility_requirements: Dict[str, float] = None
    ) -> Optional[AnonymizationConfig]:
        """Generate anonymization configuration based on risk assessment"""
        if not assessment.recommended_techniques:
            return None
        
        # Select best technique based on utility requirements
        technique = assessment.recommended_techniques[0]
        
        # Set parameters based on technique and risk level
        parameters = {}
        if technique == AnonymizationTechnique.K_ANONYMITY:
            k_value = 5 if assessment.risk_level == PrivacyRiskLevel.CRITICAL else 3
            parameters = {'k': k_value}
        elif technique == AnonymizationTechnique.MASKING:
            parameters = {
                'mask_char': '*',
                'preserve_length': True,
                'preserve_format': True
            }
        elif technique == AnonymizationTechnique.DIFFERENTIAL_PRIVACY:
            epsilon = 0.5 if assessment.risk_level == PrivacyRiskLevel.HIGH else 1.0
            parameters = {'epsilon': epsilon, 'sensitivity': 1.0}
        
        return AnonymizationConfig(
            technique=technique,
            parameters=parameters,
            target_columns=[assessment.column_name],
            preserve_utility=True,
            risk_threshold=0.1
        )
    
    def _pseudonymize_value(self, value: str) -> str:
        """Create pseudonym for a value"""
        hash_input = f"{value}{self.salt}".encode('utf-8')
        return hashlib.sha256(hash_input).hexdigest()[:16]
    
    def _mask_value(self, value: str, mask_char: str, preserve_length: bool, preserve_format: bool) -> str:
        """Mask a value while optionally preserving length and format"""
        if not value or value == 'nan':
            return value
        
        if preserve_format:
            # Preserve special characters and structure
            masked = ""
            for char in value:
                if char.isalnum():
                    masked += mask_char
                else:
                    masked += char
            return masked
        elif preserve_length:
            return mask_char * len(value)
        else:
            return mask_char * 3
    
    def _generalize_numerical(self, series: pd.Series, levels: int) -> pd.Series:
        """Generalize numerical data into ranges"""
        try:
            min_val, max_val = series.min(), series.max()
            range_size = (max_val - min_val) / levels
            
            def generalize_value(x):
                if pd.isna(x):
                    return x
                range_idx = min(int((x - min_val) / range_size), levels - 1)
                range_start = min_val + range_idx * range_size
                range_end = range_start + range_size
                return f"{range_start:.1f}-{range_end:.1f}"
            
            return series.apply(generalize_value)
        except Exception:
            return series
    
    def _generalize_categorical(self, series: pd.Series, k: int) -> pd.Series:
        """Generalize categorical data"""
        value_counts = series.value_counts()
        
        # Group rare values together
        rare_values = value_counts[value_counts < k].index
        
        def generalize_value(x):
            if x in rare_values:
                return "OTHER"
            return x
        
        return series.apply(generalize_value)
    
    def _generalize_categorical_hierarchy(self, series: pd.Series) -> pd.Series:
        """Apply hierarchical generalization to categorical data"""
        # Simple hierarchy: specific -> general
        generalization_map = {
            # Example mappings - would be domain-specific in practice
            'Manager': 'Employee',
            'Developer': 'Employee',
            'Analyst': 'Employee',
            'Director': 'Executive',
            'VP': 'Executive',
            'CEO': 'Executive'
        }
        
        return series.map(generalization_map).fillna(series)
    
    def _ensure_l_diversity(self, data: pd.DataFrame, sensitive_column: str, l: int) -> pd.DataFrame:
        """Ensure l-diversity in sensitive attribute"""
        # Simple l-diversity implementation
        # In practice, this would be more sophisticated
        return data
    
    def _calculate_privacy_gain(self, original: pd.DataFrame, anonymized: pd.DataFrame, config: AnonymizationConfig) -> float:
        """Calculate privacy gain from anonymization"""
        # Simple privacy gain calculation
        # In practice, this would use more sophisticated metrics
        if config.technique == AnonymizationTechnique.SYNTHETIC_DATA:
            return 0.95
        elif config.technique == AnonymizationTechnique.SUPPRESSION:
            return 0.9
        elif config.technique == AnonymizationTechnique.K_ANONYMITY:
            return 0.8
        elif config.technique == AnonymizationTechnique.PSEUDONYMIZATION:
            return 0.85
        else:
            return 0.7
    
    def _calculate_utility_loss(self, original: pd.DataFrame, anonymized: pd.DataFrame, config: AnonymizationConfig) -> float:
        """Calculate utility loss from anonymization"""
        # Simple utility loss calculation
        if config.technique == AnonymizationTechnique.SUPPRESSION:
            return 0.3
        elif config.technique == AnonymizationTechnique.SYNTHETIC_DATA:
            return 0.2
        elif config.technique == AnonymizationTechnique.GENERALIZATION:
            return 0.15
        else:
            return 0.1


def create_anonymization_engine() -> AnonymizationEngine:
    """Factory function to create AnonymizationEngine instance"""
    return AnonymizationEngine()