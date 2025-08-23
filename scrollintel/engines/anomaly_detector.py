"""
Statistical Anomaly Detection Engine
Implements various anomaly detection algorithms for data quality monitoring
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy import stats
import logging
from dataclasses import dataclass

from ..models.data_quality_models import DataAnomaly, DataProfile, Severity

logger = logging.getLogger(__name__)

@dataclass
class AnomalyDetectionResult:
    """Result of anomaly detection analysis"""
    anomalies: List[Dict]
    total_records: int
    anomaly_count: int
    anomaly_rate: float
    detection_method: str
    confidence_threshold: float

class AnomalyDetector:
    """Statistical anomaly detection engine"""
    
    def __init__(self, db_session: Session):
        self.db_session = db_session
        self.logger = logging.getLogger(__name__)
        
    def detect_anomalies(self, data: pd.DataFrame, baseline: DataProfile, 
                        detection_methods: List[str] = None) -> List[DataAnomaly]:
        """
        Detect anomalies in data using multiple statistical methods
        
        Args:
            data: DataFrame to analyze
            baseline: Data profile baseline for comparison
            detection_methods: List of methods to use ['statistical', 'isolation_forest', 'pca']
            
        Returns:
            List of detected anomalies
        """
        if detection_methods is None:
            detection_methods = ['statistical', 'isolation_forest']
        
        all_anomalies = []
        column = baseline.column_name
        
        if column not in data.columns:
            self.logger.warning(f"Column '{column}' not found in data")
            return []
        
        # Get column data
        column_data = data[column].dropna()
        
        if len(column_data) == 0:
            return []
        
        # Apply each detection method
        for method in detection_methods:
            try:
                if method == 'statistical':
                    anomalies = self._detect_statistical_anomalies(data, column, baseline)
                elif method == 'isolation_forest':
                    anomalies = self._detect_isolation_forest_anomalies(data, column, baseline)
                elif method == 'pca':
                    anomalies = self._detect_pca_anomalies(data, column, baseline)
                else:
                    self.logger.warning(f"Unknown detection method: {method}")
                    continue
                
                all_anomalies.extend(anomalies)
                
            except Exception as e:
                self.logger.error(f"Error in {method} anomaly detection: {str(e)}")
        
        # Remove duplicates and save to database
        unique_anomalies = self._deduplicate_anomalies(all_anomalies)
        
        for anomaly_data in unique_anomalies:
            anomaly = DataAnomaly(**anomaly_data)
            self.db_session.add(anomaly)
        
        self.db_session.commit()
        return unique_anomalies
    
    def _detect_statistical_anomalies(self, data: pd.DataFrame, column: str, 
                                    baseline: DataProfile) -> List[Dict]:
        """Detect anomalies using statistical methods (Z-score, IQR)"""
        anomalies = []
        column_data = pd.to_numeric(data[column], errors='coerce').dropna()
        
        if len(column_data) == 0:
            return anomalies
        
        # Z-score based detection
        if baseline.mean_value is not None and baseline.std_deviation is not None:
            z_scores = np.abs((column_data - baseline.mean_value) / baseline.std_deviation)
            z_threshold = 3.0  # Standard 3-sigma rule
            
            outlier_mask = z_scores > z_threshold
            outlier_indices = column_data[outlier_mask].index
            
            for idx in outlier_indices:
                z_score = z_scores.loc[idx]
                anomalies.append({
                    'table_name': baseline.table_name,
                    'column_name': column,
                    'record_id': str(idx),
                    'anomaly_type': 'statistical_outlier',
                    'confidence_score': min(0.99, z_score / 10.0),  # Scale confidence
                    'severity': self._calculate_severity(z_score, 'z_score'),
                    'expected_value': str(baseline.mean_value),
                    'actual_value': str(column_data.loc[idx]),
                    'deviation_score': float(z_score),
                    'baseline_mean': baseline.mean_value,
                    'baseline_std': baseline.std_deviation,
                    'z_score': float(z_score)
                })
        
        # IQR based detection
        q1 = column_data.quantile(0.25)
        q3 = column_data.quantile(0.75)
        iqr = q3 - q1
        
        if iqr > 0:
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            iqr_outliers = column_data[(column_data < lower_bound) | (column_data > upper_bound)]
            
            for idx, value in iqr_outliers.items():
                if idx not in [a['record_id'] for a in anomalies]:  # Avoid duplicates
                    deviation = min(abs(value - lower_bound), abs(value - upper_bound)) / iqr
                    anomalies.append({
                        'table_name': baseline.table_name,
                        'column_name': column,
                        'record_id': str(idx),
                        'anomaly_type': 'iqr_outlier',
                        'confidence_score': min(0.95, deviation / 5.0),
                        'severity': self._calculate_severity(deviation, 'iqr'),
                        'expected_value': f"[{lower_bound:.2f}, {upper_bound:.2f}]",
                        'actual_value': str(value),
                        'deviation_score': float(deviation)
                    })
        
        return anomalies
    
    def _detect_isolation_forest_anomalies(self, data: pd.DataFrame, column: str, 
                                         baseline: DataProfile) -> List[Dict]:
        """Detect anomalies using Isolation Forest algorithm"""
        anomalies = []
        
        # Prepare numeric data
        numeric_data = pd.to_numeric(data[column], errors='coerce').dropna()
        
        if len(numeric_data) < 10:  # Need minimum samples
            return anomalies
        
        # Reshape for sklearn
        X = numeric_data.values.reshape(-1, 1)
        
        # Apply Isolation Forest
        iso_forest = IsolationForest(
            contamination=0.1,  # Expect 10% anomalies
            random_state=42,
            n_estimators=100
        )
        
        outlier_labels = iso_forest.fit_predict(X)
        anomaly_scores = iso_forest.decision_function(X)
        
        # Find anomalies (labeled as -1)
        anomaly_indices = numeric_data.index[outlier_labels == -1]
        
        for idx in anomaly_indices:
            score_idx = numeric_data.index.get_loc(idx)
            anomaly_score = anomaly_scores[score_idx]
            confidence = 1 - (anomaly_score + 0.5)  # Convert to 0-1 range
            
            anomalies.append({
                'table_name': baseline.table_name,
                'column_name': column,
                'record_id': str(idx),
                'anomaly_type': 'isolation_forest',
                'confidence_score': max(0.5, min(0.99, confidence)),
                'severity': self._calculate_severity(abs(anomaly_score), 'isolation'),
                'expected_value': 'normal_range',
                'actual_value': str(numeric_data.loc[idx]),
                'deviation_score': float(abs(anomaly_score))
            })
        
        return anomalies
    
    def _detect_pca_anomalies(self, data: pd.DataFrame, column: str, 
                            baseline: DataProfile) -> List[Dict]:
        """Detect anomalies using PCA reconstruction error"""
        anomalies = []
        
        # This method works better with multiple numeric columns
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_columns) < 2:
            return anomalies  # Need multiple features for PCA
        
        # Prepare data
        numeric_data = data[numeric_columns].dropna()
        
        if len(numeric_data) < 10:
            return anomalies
        
        # Standardize data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_data)
        
        # Apply PCA
        n_components = min(len(numeric_columns) - 1, 5)  # Use fewer components
        pca = PCA(n_components=n_components)
        transformed = pca.fit_transform(scaled_data)
        reconstructed = pca.inverse_transform(transformed)
        
        # Calculate reconstruction error
        reconstruction_errors = np.sum((scaled_data - reconstructed) ** 2, axis=1)
        
        # Find anomalies using threshold (95th percentile)
        threshold = np.percentile(reconstruction_errors, 95)
        anomaly_mask = reconstruction_errors > threshold
        
        anomaly_indices = numeric_data.index[anomaly_mask]
        
        for idx in anomaly_indices:
            if column in numeric_data.columns:  # Only report for the target column
                error_idx = numeric_data.index.get_loc(idx)
                error_score = reconstruction_errors[error_idx]
                confidence = min(0.99, error_score / (threshold * 2))
                
                anomalies.append({
                    'table_name': baseline.table_name,
                    'column_name': column,
                    'record_id': str(idx),
                    'anomaly_type': 'pca_reconstruction',
                    'confidence_score': max(0.5, confidence),
                    'severity': self._calculate_severity(error_score, 'pca'),
                    'expected_value': 'normal_pattern',
                    'actual_value': str(data.loc[idx, column]),
                    'deviation_score': float(error_score)
                })
        
        return anomalies
    
    def _calculate_severity(self, score: float, method: str) -> Severity:
        """Calculate severity based on anomaly score and method"""
        if method == 'z_score':
            if score > 5:
                return Severity.CRITICAL
            elif score > 4:
                return Severity.HIGH
            elif score > 3:
                return Severity.MEDIUM
            else:
                return Severity.LOW
        
        elif method == 'iqr':
            if score > 3:
                return Severity.CRITICAL
            elif score > 2:
                return Severity.HIGH
            elif score > 1.5:
                return Severity.MEDIUM
            else:
                return Severity.LOW
        
        elif method in ['isolation', 'pca']:
            if score > 0.8:
                return Severity.CRITICAL
            elif score > 0.6:
                return Severity.HIGH
            elif score > 0.4:
                return Severity.MEDIUM
            else:
                return Severity.LOW
        
        return Severity.MEDIUM
    
    def _deduplicate_anomalies(self, anomalies: List[Dict]) -> List[Dict]:
        """Remove duplicate anomalies based on record_id and column"""
        seen = set()
        unique_anomalies = []
        
        for anomaly in anomalies:
            key = (anomaly['record_id'], anomaly['column_name'])
            if key not in seen:
                seen.add(key)
                unique_anomalies.append(anomaly)
        
        return unique_anomalies
    
    def detect_data_drift(self, current_data: pd.DataFrame, baseline: DataProfile, 
                         column: str) -> Dict:
        """
        Detect data drift by comparing current data distribution to baseline
        
        Args:
            current_data: Current data to analyze
            baseline: Historical baseline profile
            column: Column to analyze for drift
            
        Returns:
            Drift detection results
        """
        if column not in current_data.columns:
            return {"drift_detected": False, "error": f"Column '{column}' not found"}
        
        column_data = pd.to_numeric(current_data[column], errors='coerce').dropna()
        
        if len(column_data) == 0:
            return {"drift_detected": False, "error": "No numeric data found"}
        
        # Calculate current statistics
        current_mean = column_data.mean()
        current_std = column_data.std()
        
        # Compare with baseline
        drift_results = {
            "drift_detected": False,
            "drift_score": 0.0,
            "drift_type": None,
            "current_stats": {
                "mean": current_mean,
                "std": current_std,
                "count": len(column_data)
            },
            "baseline_stats": {
                "mean": baseline.mean_value,
                "std": baseline.std_deviation,
                "count": baseline.record_count
            }
        }
        
        # Mean drift detection
        if baseline.mean_value is not None:
            mean_drift = abs(current_mean - baseline.mean_value) / baseline.mean_value
            if mean_drift > 0.1:  # 10% threshold
                drift_results["drift_detected"] = True
                drift_results["drift_type"] = "mean_shift"
                drift_results["drift_score"] = mean_drift
        
        # Standard deviation drift detection
        if baseline.std_deviation is not None and baseline.std_deviation > 0:
            std_drift = abs(current_std - baseline.std_deviation) / baseline.std_deviation
            if std_drift > 0.2:  # 20% threshold
                drift_results["drift_detected"] = True
                if drift_results["drift_type"] is None:
                    drift_results["drift_type"] = "variance_change"
                else:
                    drift_results["drift_type"] = "distribution_shift"
                drift_results["drift_score"] = max(drift_results["drift_score"], std_drift)
        
        # Kolmogorov-Smirnov test for distribution drift
        if baseline.value_distribution and len(column_data) > 30:
            try:
                # Create baseline distribution sample
                baseline_sample = np.random.normal(
                    baseline.mean_value, 
                    baseline.std_deviation, 
                    min(1000, len(column_data))
                )
                
                # Perform KS test
                ks_statistic, p_value = stats.ks_2samp(baseline_sample, column_data)
                
                if p_value < 0.05:  # Significant difference
                    drift_results["drift_detected"] = True
                    drift_results["drift_type"] = "distribution_drift"
                    drift_results["drift_score"] = max(drift_results["drift_score"], ks_statistic)
                    drift_results["ks_test"] = {
                        "statistic": ks_statistic,
                        "p_value": p_value
                    }
            except Exception as e:
                self.logger.warning(f"KS test failed: {str(e)}")
        
        return drift_results
    
    def create_data_profile(self, data: pd.DataFrame, table_name: str, 
                          column_name: str) -> DataProfile:
        """
        Create a data profile baseline for anomaly detection
        
        Args:
            data: DataFrame to profile
            table_name: Name of the table
            column_name: Name of the column to profile
            
        Returns:
            DataProfile object
        """
        if column_name not in data.columns:
            raise ValueError(f"Column '{column_name}' not found in data")
        
        column_data = data[column_name]
        
        # Basic statistics
        record_count = len(column_data)
        null_count = column_data.isnull().sum()
        unique_count = column_data.nunique()
        
        # Initialize profile
        profile = DataProfile(
            table_name=table_name,
            column_name=column_name,
            data_type=str(column_data.dtype),
            record_count=record_count,
            null_count=null_count,
            unique_count=unique_count
        )
        
        # Numeric statistics
        numeric_data = pd.to_numeric(column_data, errors='coerce').dropna()
        if len(numeric_data) > 0:
            profile.min_value = float(numeric_data.min())
            profile.max_value = float(numeric_data.max())
            profile.mean_value = float(numeric_data.mean())
            profile.median_value = float(numeric_data.median())
            profile.std_deviation = float(numeric_data.std())
        
        # Categorical statistics
        if column_data.dtype == 'object' or unique_count < record_count * 0.5:
            value_counts = column_data.value_counts().head(10)
            profile.most_frequent_values = value_counts.to_dict()
            
            # Value distribution
            distribution = (value_counts / record_count).to_dict()
            profile.value_distribution = distribution
        
        # Pattern analysis for string data
        if column_data.dtype == 'object':
            string_data = column_data.dropna().astype(str)
            if len(string_data) > 0:
                # Common patterns (length, format)
                lengths = string_data.str.len()
                profile.common_patterns = {
                    "avg_length": float(lengths.mean()),
                    "min_length": int(lengths.min()),
                    "max_length": int(lengths.max()),
                    "length_std": float(lengths.std())
                }
                
                # Format patterns (basic)
                format_patterns = {}
                if string_data.str.match(r'^\d+$').any():
                    format_patterns["numeric_strings"] = string_data.str.match(r'^\d+$').sum()
                if string_data.str.match(r'^[a-zA-Z\s]+$').any():
                    format_patterns["alphabetic_strings"] = string_data.str.match(r'^[a-zA-Z\s]+$').sum()
                if string_data.str.match(r'^[\w\.-]+@[\w\.-]+\.\w+$').any():
                    format_patterns["email_format"] = string_data.str.match(r'^[\w\.-]+@[\w\.-]+\.\w+$').sum()
                
                profile.format_patterns = format_patterns
        
        # Quality scores
        profile.completeness_score = (record_count - null_count) / record_count * 100 if record_count > 0 else 0
        profile.consistency_score = 100.0  # Would need reference data to calculate
        profile.validity_score = 100.0  # Would need validation rules to calculate
        
        # Save to database
        self.db_session.add(profile)
        self.db_session.commit()
        
        return profile
    
    def update_baseline(self, profile_id: str, new_data: pd.DataFrame) -> DataProfile:
        """Update an existing data profile with new data"""
        profile = self.db_session.query(DataProfile).filter_by(id=profile_id).first()
        
        if not profile:
            raise ValueError(f"Profile with id '{profile_id}' not found")
        
        # Create new profile with updated data
        updated_profile = self.create_data_profile(
            new_data, 
            profile.table_name, 
            profile.column_name
        )
        
        # Update the existing profile
        profile.record_count = updated_profile.record_count
        profile.null_count = updated_profile.null_count
        profile.unique_count = updated_profile.unique_count
        profile.min_value = updated_profile.min_value
        profile.max_value = updated_profile.max_value
        profile.mean_value = updated_profile.mean_value
        profile.median_value = updated_profile.median_value
        profile.std_deviation = updated_profile.std_deviation
        profile.most_frequent_values = updated_profile.most_frequent_values
        profile.value_distribution = updated_profile.value_distribution
        profile.common_patterns = updated_profile.common_patterns
        profile.format_patterns = updated_profile.format_patterns
        profile.completeness_score = updated_profile.completeness_score
        profile.updated_at = datetime.utcnow()
        
        # Remove the temporary profile
        self.db_session.delete(updated_profile)
        self.db_session.commit()
        
        return profile