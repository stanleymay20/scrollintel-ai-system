"""
Data Quality Monitor Engine
Implements rule-based validation and anomaly detection for data pipelines
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import create_engine
import logging
import json
from dataclasses import dataclass

from ..models.data_quality_models import (
    QualityRule, QualityReport, DataAnomaly, DataProfile, QualityAlert,
    QualityRuleType, Severity, QualityStatus
)

logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """Result of a quality validation check"""
    passed: bool
    score: float
    records_checked: int
    records_failed: int
    error_message: Optional[str] = None
    error_details: Optional[Dict] = None
    sample_failures: Optional[List] = None

@dataclass
class AnomalyResult:
    """Result of anomaly detection"""
    is_anomaly: bool
    confidence_score: float
    anomaly_type: str
    deviation_score: float
    expected_value: Any
    actual_value: Any
    z_score: Optional[float] = None

class DataQualityMonitor:
    """Main data quality monitoring engine"""
    
    def __init__(self, db_session: Session):
        self.db_session = db_session
        self.logger = logging.getLogger(__name__)
        
    def validate_data_batch(self, data: pd.DataFrame, rules: List[QualityRule], 
                          pipeline_execution_id: str = None) -> List[QualityReport]:
        """
        Validate a data batch against quality rules
        
        Args:
            data: DataFrame to validate
            rules: List of quality rules to apply
            pipeline_execution_id: ID of the pipeline execution
            
        Returns:
            List of quality reports
        """
        reports = []
        
        for rule in rules:
            if not rule.is_active:
                continue
                
            try:
                start_time = datetime.utcnow()
                
                # Apply the appropriate validation based on rule type
                result = self._apply_rule(data, rule)
                
                execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
                
                # Create quality report
                report = QualityReport(
                    rule_id=rule.id,
                    pipeline_execution_id=pipeline_execution_id,
                    status=QualityStatus.PASSED if result.passed else QualityStatus.FAILED,
                    score=result.score,
                    records_checked=result.records_checked,
                    records_failed=result.records_failed,
                    error_message=result.error_message,
                    error_details=result.error_details,
                    sample_failures=result.sample_failures,
                    execution_time_ms=int(execution_time),
                    data_volume_mb=data.memory_usage(deep=True).sum() / (1024 * 1024)
                )
                
                self.db_session.add(report)
                reports.append(report)
                
                # Handle rule violations
                if not result.passed:
                    self._handle_rule_violation(rule, report, result)
                    
            except Exception as e:
                self.logger.error(f"Error validating rule {rule.name}: {str(e)}")
                
                # Create error report
                error_report = QualityReport(
                    rule_id=rule.id,
                    pipeline_execution_id=pipeline_execution_id,
                    status=QualityStatus.FAILED,
                    score=0.0,
                    records_checked=len(data),
                    records_failed=len(data),
                    error_message=str(e),
                    error_details={"exception_type": type(e).__name__}
                )
                
                self.db_session.add(error_report)
                reports.append(error_report)
        
        self.db_session.commit()
        return reports
    
    def _apply_rule(self, data: pd.DataFrame, rule: QualityRule) -> ValidationResult:
        """Apply a specific quality rule to data"""
        
        if rule.rule_type == QualityRuleType.COMPLETENESS:
            return self._check_completeness(data, rule)
        elif rule.rule_type == QualityRuleType.ACCURACY:
            return self._check_accuracy(data, rule)
        elif rule.rule_type == QualityRuleType.CONSISTENCY:
            return self._check_consistency(data, rule)
        elif rule.rule_type == QualityRuleType.VALIDITY:
            return self._check_validity(data, rule)
        elif rule.rule_type == QualityRuleType.UNIQUENESS:
            return self._check_uniqueness(data, rule)
        elif rule.rule_type == QualityRuleType.TIMELINESS:
            return self._check_timeliness(data, rule)
        elif rule.rule_type == QualityRuleType.STATISTICAL:
            return self._check_statistical(data, rule)
        else:
            raise ValueError(f"Unknown rule type: {rule.rule_type}")
    
    def _check_completeness(self, data: pd.DataFrame, rule: QualityRule) -> ValidationResult:
        """Check data completeness (null values)"""
        column = rule.target_column
        threshold = rule.threshold_value or 0.95  # Default 95% completeness
        
        if column not in data.columns:
            return ValidationResult(
                passed=False,
                score=0.0,
                records_checked=len(data),
                records_failed=len(data),
                error_message=f"Column '{column}' not found in data"
            )
        
        null_count = data[column].isnull().sum()
        total_count = len(data)
        completeness_ratio = (total_count - null_count) / total_count if total_count > 0 else 0
        
        passed = completeness_ratio >= threshold
        score = completeness_ratio * 100
        
        sample_failures = []
        if not passed and null_count > 0:
            null_indices = data[data[column].isnull()].index.tolist()[:10]
            sample_failures = [{"row_index": idx, "issue": "null_value"} for idx in null_indices]
        
        return ValidationResult(
            passed=passed,
            score=score,
            records_checked=total_count,
            records_failed=null_count,
            sample_failures=sample_failures,
            error_details={
                "completeness_ratio": completeness_ratio,
                "threshold": threshold,
                "null_count": null_count
            }
        )
    
    def _check_validity(self, data: pd.DataFrame, rule: QualityRule) -> ValidationResult:
        """Check data validity against patterns or ranges"""
        column = rule.target_column
        conditions = rule.conditions or {}
        
        if column not in data.columns:
            return ValidationResult(
                passed=False,
                score=0.0,
                records_checked=len(data),
                records_failed=len(data),
                error_message=f"Column '{column}' not found in data"
            )
        
        total_count = len(data)
        failed_records = []
        
        # Check different validity conditions
        if 'min_value' in conditions and 'max_value' in conditions:
            # Range validation
            min_val = conditions['min_value']
            max_val = conditions['max_value']
            invalid_mask = ~data[column].between(min_val, max_val, inclusive='both')
            failed_records.extend(data[invalid_mask].index.tolist())
            
        elif 'pattern' in conditions:
            # Pattern validation (regex)
            pattern = conditions['pattern']
            invalid_mask = ~data[column].astype(str).str.match(pattern, na=False)
            failed_records.extend(data[invalid_mask].index.tolist())
            
        elif 'allowed_values' in conditions:
            # Allowed values validation
            allowed_values = conditions['allowed_values']
            invalid_mask = ~data[column].isin(allowed_values)
            failed_records.extend(data[invalid_mask].index.tolist())
        
        failed_count = len(set(failed_records))
        validity_ratio = (total_count - failed_count) / total_count if total_count > 0 else 0
        passed = validity_ratio >= (rule.threshold_value or 0.95)
        
        sample_failures = []
        if failed_count > 0:
            sample_indices = list(set(failed_records))[:10]
            sample_failures = [
                {
                    "row_index": idx,
                    "value": str(data.loc[idx, column]),
                    "issue": "invalid_value"
                }
                for idx in sample_indices
            ]
        
        return ValidationResult(
            passed=passed,
            score=validity_ratio * 100,
            records_checked=total_count,
            records_failed=failed_count,
            sample_failures=sample_failures,
            error_details={
                "validity_ratio": validity_ratio,
                "conditions": conditions
            }
        )
    
    def _check_uniqueness(self, data: pd.DataFrame, rule: QualityRule) -> ValidationResult:
        """Check data uniqueness"""
        column = rule.target_column
        
        if column not in data.columns:
            return ValidationResult(
                passed=False,
                score=0.0,
                records_checked=len(data),
                records_failed=len(data),
                error_message=f"Column '{column}' not found in data"
            )
        
        total_count = len(data)
        unique_count = data[column].nunique()
        duplicate_count = total_count - unique_count
        
        uniqueness_ratio = unique_count / total_count if total_count > 0 else 0
        passed = uniqueness_ratio >= (rule.threshold_value or 1.0)
        
        sample_failures = []
        if duplicate_count > 0:
            duplicates = data[data.duplicated(subset=[column], keep=False)]
            sample_failures = [
                {
                    "row_index": idx,
                    "value": str(row[column]),
                    "issue": "duplicate_value"
                }
                for idx, row in duplicates.head(10).iterrows()
            ]
        
        return ValidationResult(
            passed=passed,
            score=uniqueness_ratio * 100,
            records_checked=total_count,
            records_failed=duplicate_count,
            sample_failures=sample_failures,
            error_details={
                "uniqueness_ratio": uniqueness_ratio,
                "unique_count": unique_count,
                "duplicate_count": duplicate_count
            }
        )
    
    def _check_consistency(self, data: pd.DataFrame, rule: QualityRule) -> ValidationResult:
        """Check data consistency across columns"""
        conditions = rule.conditions or {}
        
        if 'reference_column' not in conditions:
            return ValidationResult(
                passed=False,
                score=0.0,
                records_checked=len(data),
                records_failed=len(data),
                error_message="Reference column not specified for consistency check"
            )
        
        column = rule.target_column
        ref_column = conditions['reference_column']
        
        if column not in data.columns or ref_column not in data.columns:
            return ValidationResult(
                passed=False,
                score=0.0,
                records_checked=len(data),
                records_failed=len(data),
                error_message=f"Required columns not found in data"
            )
        
        # Example: Check if date_end >= date_start
        if conditions.get('check_type') == 'date_order':
            inconsistent_mask = data[column] < data[ref_column]
            failed_indices = data[inconsistent_mask].index.tolist()
        else:
            # Default: check if values are equal
            inconsistent_mask = data[column] != data[ref_column]
            failed_indices = data[inconsistent_mask].index.tolist()
        
        failed_count = len(failed_indices)
        total_count = len(data)
        consistency_ratio = (total_count - failed_count) / total_count if total_count > 0 else 0
        passed = consistency_ratio >= (rule.threshold_value or 0.95)
        
        sample_failures = []
        if failed_count > 0:
            sample_indices = failed_indices[:10]
            sample_failures = [
                {
                    "row_index": idx,
                    "target_value": str(data.loc[idx, column]),
                    "reference_value": str(data.loc[idx, ref_column]),
                    "issue": "inconsistent_values"
                }
                for idx in sample_indices
            ]
        
        return ValidationResult(
            passed=passed,
            score=consistency_ratio * 100,
            records_checked=total_count,
            records_failed=failed_count,
            sample_failures=sample_failures,
            error_details={
                "consistency_ratio": consistency_ratio,
                "check_type": conditions.get('check_type', 'equality')
            }
        )
    
    def _check_accuracy(self, data: pd.DataFrame, rule: QualityRule) -> ValidationResult:
        """Check data accuracy against reference data"""
        # This would typically involve comparing against a reference dataset
        # For now, implement a basic format accuracy check
        column = rule.target_column
        conditions = rule.conditions or {}
        
        if column not in data.columns:
            return ValidationResult(
                passed=False,
                score=0.0,
                records_checked=len(data),
                records_failed=len(data),
                error_message=f"Column '{column}' not found in data"
            )
        
        # Example: Email format accuracy
        if conditions.get('format_type') == 'email':
            email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            invalid_mask = ~data[column].astype(str).str.match(email_pattern, na=False)
            failed_indices = data[invalid_mask].index.tolist()
        else:
            # Default: assume all records are accurate
            failed_indices = []
        
        failed_count = len(failed_indices)
        total_count = len(data)
        accuracy_ratio = (total_count - failed_count) / total_count if total_count > 0 else 0
        passed = accuracy_ratio >= (rule.threshold_value or 0.95)
        
        return ValidationResult(
            passed=passed,
            score=accuracy_ratio * 100,
            records_checked=total_count,
            records_failed=failed_count,
            error_details={"accuracy_ratio": accuracy_ratio}
        )
    
    def _check_timeliness(self, data: pd.DataFrame, rule: QualityRule) -> ValidationResult:
        """Check data timeliness"""
        column = rule.target_column
        conditions = rule.conditions or {}
        
        if column not in data.columns:
            return ValidationResult(
                passed=False,
                score=0.0,
                records_checked=len(data),
                records_failed=len(data),
                error_message=f"Column '{column}' not found in data"
            )
        
        # Convert to datetime if needed
        try:
            date_column = pd.to_datetime(data[column])
        except:
            return ValidationResult(
                passed=False,
                score=0.0,
                records_checked=len(data),
                records_failed=len(data),
                error_message=f"Cannot convert column '{column}' to datetime"
            )
        
        # Check if data is within acceptable time window
        max_age_hours = conditions.get('max_age_hours', 24)
        cutoff_time = datetime.utcnow() - timedelta(hours=max_age_hours)
        
        stale_mask = date_column < cutoff_time
        failed_count = stale_mask.sum()
        total_count = len(data)
        
        timeliness_ratio = (total_count - failed_count) / total_count if total_count > 0 else 0
        passed = timeliness_ratio >= (rule.threshold_value or 0.95)
        
        return ValidationResult(
            passed=passed,
            score=timeliness_ratio * 100,
            records_checked=total_count,
            records_failed=failed_count,
            error_details={
                "timeliness_ratio": timeliness_ratio,
                "max_age_hours": max_age_hours,
                "cutoff_time": cutoff_time.isoformat()
            }
        )
    
    def _check_statistical(self, data: pd.DataFrame, rule: QualityRule) -> ValidationResult:
        """Check statistical properties of data"""
        column = rule.target_column
        conditions = rule.conditions or {}
        
        if column not in data.columns:
            return ValidationResult(
                passed=False,
                score=0.0,
                records_checked=len(data),
                records_failed=len(data),
                error_message=f"Column '{column}' not found in data"
            )
        
        # Get numeric data only
        numeric_data = pd.to_numeric(data[column], errors='coerce').dropna()
        
        if len(numeric_data) == 0:
            return ValidationResult(
                passed=False,
                score=0.0,
                records_checked=len(data),
                records_failed=len(data),
                error_message=f"No numeric data found in column '{column}'"
            )
        
        # Check statistical properties
        mean_val = numeric_data.mean()
        std_val = numeric_data.std()
        
        expected_mean = conditions.get('expected_mean')
        expected_std = conditions.get('expected_std')
        tolerance = conditions.get('tolerance', 0.1)  # 10% tolerance
        
        passed = True
        issues = []
        
        if expected_mean is not None:
            mean_diff = abs(mean_val - expected_mean) / expected_mean
            if mean_diff > tolerance:
                passed = False
                issues.append(f"Mean deviation: {mean_diff:.2%}")
        
        if expected_std is not None:
            std_diff = abs(std_val - expected_std) / expected_std
            if std_diff > tolerance:
                passed = False
                issues.append(f"Std deviation: {std_diff:.2%}")
        
        # Calculate score based on how close we are to expected values
        score = 100.0
        if not passed:
            score = max(0, 100 - (len(issues) * 25))  # Reduce score for each issue
        
        return ValidationResult(
            passed=passed,
            score=score,
            records_checked=len(data),
            records_failed=0 if passed else len(data),
            error_message="; ".join(issues) if issues else None,
            error_details={
                "actual_mean": mean_val,
                "actual_std": std_val,
                "expected_mean": expected_mean,
                "expected_std": expected_std,
                "tolerance": tolerance
            }
        )
    
    def _handle_rule_violation(self, rule: QualityRule, report: QualityReport, result: ValidationResult):
        """Handle quality rule violations"""
        # Create alert
        alert = QualityAlert(
            rule_id=rule.id,
            quality_report_id=report.id,
            alert_type="rule_violation",
            severity=rule.severity,
            message=f"Quality rule '{rule.name}' failed: {result.error_message or 'Threshold not met'}",
            pipeline_id=rule.target_pipeline_id,
            table_name=rule.target_table,
            column_name=rule.target_column
        )
        
        self.db_session.add(alert)
        
        # Execute configured actions
        actions = rule.actions or []
        for action in actions:
            self._execute_action(action, rule, report, alert)
    
    def _execute_action(self, action: Dict, rule: QualityRule, report: QualityReport, alert: QualityAlert):
        """Execute a quality rule action"""
        action_type = action.get('type')
        
        if action_type == 'alert':
            # Send notification (would integrate with notification system)
            self.logger.warning(f"Quality alert: {alert.message}")
            
        elif action_type == 'stop_pipeline':
            # Signal to stop the pipeline (would integrate with orchestrator)
            self.logger.error(f"Stopping pipeline due to quality failure: {alert.message}")
            
        elif action_type == 'quarantine_data':
            # Mark data for quarantine (would move to quarantine table)
            self.logger.info(f"Quarantining data due to quality failure: {alert.message}")
    
    def create_quality_rule(self, rule_config: Dict) -> str:
        """Create a new quality rule"""
        rule = QualityRule(
            name=rule_config['name'],
            description=rule_config.get('description'),
            rule_type=QualityRuleType(rule_config['rule_type']),
            severity=Severity(rule_config.get('severity', 'medium')),
            conditions=rule_config.get('conditions'),
            threshold_value=rule_config.get('threshold_value'),
            expected_value=rule_config.get('expected_value'),
            target_table=rule_config.get('target_table'),
            target_column=rule_config.get('target_column'),
            target_pipeline_id=rule_config.get('target_pipeline_id'),
            actions=rule_config.get('actions', []),
            created_by=rule_config.get('created_by')
        )
        
        self.db_session.add(rule)
        self.db_session.commit()
        
        return rule.id
    
    def get_quality_metrics(self, pipeline_id: str, time_range: Tuple[datetime, datetime]) -> Dict:
        """Get quality metrics for a pipeline within a time range"""
        start_time, end_time = time_range
        
        # Query quality reports for the pipeline
        reports = self.db_session.query(QualityReport).join(QualityRule).filter(
            QualityRule.target_pipeline_id == pipeline_id,
            QualityReport.check_timestamp >= start_time,
            QualityReport.check_timestamp <= end_time
        ).all()
        
        if not reports:
            return {
                "total_checks": 0,
                "passed_checks": 0,
                "failed_checks": 0,
                "average_score": 0.0,
                "quality_trend": []
            }
        
        total_checks = len(reports)
        passed_checks = sum(1 for r in reports if r.status == QualityStatus.PASSED)
        failed_checks = total_checks - passed_checks
        average_score = sum(r.score or 0 for r in reports) / total_checks
        
        # Calculate quality trend (daily averages)
        daily_scores = {}
        for report in reports:
            day = report.check_timestamp.date()
            if day not in daily_scores:
                daily_scores[day] = []
            daily_scores[day].append(report.score or 0)
        
        quality_trend = [
            {
                "date": day.isoformat(),
                "average_score": sum(scores) / len(scores)
            }
            for day, scores in sorted(daily_scores.items())
        ]
        
        return {
            "total_checks": total_checks,
            "passed_checks": passed_checks,
            "failed_checks": failed_checks,
            "average_score": average_score,
            "quality_trend": quality_trend
        }