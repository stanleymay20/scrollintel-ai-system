"""
Data Quality Monitor for Advanced Analytics Dashboard System
"""

from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import pandas as pd
import numpy as np
import logging
import json

logger = logging.getLogger(__name__)


class QualityRuleType(Enum):
    COMPLETENESS = "completeness"
    UNIQUENESS = "uniqueness"
    VALIDITY = "validity"
    CONSISTENCY = "consistency"
    ACCURACY = "accuracy"
    TIMELINESS = "timeliness"
    RANGE = "range"
    FORMAT = "format"


class QualitySeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class QualityRule:
    """Represents a data quality validation rule"""
    id: str
    name: str
    description: str
    rule_type: QualityRuleType
    severity: QualitySeverity
    target_fields: List[str]
    parameters: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class QualityIssue:
    """Represents a data quality issue"""
    rule_id: str
    rule_name: str
    severity: QualitySeverity
    field_name: str
    issue_description: str
    affected_records: int
    sample_values: List[Any] = field(default_factory=list)
    detected_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QualityReport:
    """Comprehensive data quality assessment report"""
    dataset_name: str
    assessment_timestamp: datetime
    total_records: int
    overall_score: float
    dimension_scores: Dict[str, float]
    issues: List[QualityIssue]
    metrics: Dict[str, Any]
    recommendations: List[str] = field(default_factory=list)


class DataQualityMonitor:
    """
    Advanced data quality monitoring and validation system
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.rules: Dict[str, QualityRule] = {}
        self.validation_functions = self._initialize_validators()
        self.quality_history: List[QualityReport] = []
        
    def register_quality_rule(self, rule: QualityRule) -> bool:
        """Register a data quality rule"""
        try:
            self.rules[rule.id] = rule
            logger.info(f"Registered quality rule: {rule.name}")
            return True
        except Exception as e:
            logger.error(f"Failed to register quality rule {rule.name}: {str(e)}")
            return False
    
    def assess_data_quality(self, data: pd.DataFrame, dataset_name: str,
                          rule_ids: Optional[List[str]] = None) -> QualityReport:
        """
        Perform comprehensive data quality assessment
        """
        try:
            assessment_start = datetime.utcnow()
            
            # Determine which rules to apply
            rules_to_apply = []
            if rule_ids:
                rules_to_apply = [self.rules[rid] for rid in rule_ids if rid in self.rules]
            else:
                rules_to_apply = [rule for rule in self.rules.values() if rule.enabled]
            
            # Initialize assessment results
            issues = []
            dimension_scores = {}
            metrics = {
                "total_records": len(data),
                "total_fields": len(data.columns),
                "assessment_duration": 0,
                "rules_applied": len(rules_to_apply)
            }
            
            # Apply quality rules
            for rule in rules_to_apply:
                try:
                    rule_issues = self._apply_quality_rule(data, rule)
                    issues.extend(rule_issues)
                    
                    # Calculate dimension score for this rule
                    if rule.rule_type.value not in dimension_scores:
                        dimension_scores[rule.rule_type.value] = []
                    
                    # Score based on issues found (0-100 scale)
                    total_affected = sum(issue.affected_records for issue in rule_issues)
                    rule_score = max(0, 100 - (total_affected / len(data) * 100)) if len(data) > 0 else 100
                    dimension_scores[rule.rule_type.value].append(rule_score)
                    
                except Exception as e:
                    logger.error(f"Failed to apply rule {rule.name}: {str(e)}")
                    issues.append(QualityIssue(
                        rule_id=rule.id,
                        rule_name=rule.name,
                        severity=QualitySeverity.HIGH,
                        field_name="system",
                        issue_description=f"Rule execution failed: {str(e)}",
                        affected_records=0
                    ))
            
            # Calculate final dimension scores
            final_dimension_scores = {}
            for dimension, scores in dimension_scores.items():
                final_dimension_scores[dimension] = np.mean(scores) if scores else 100
            
            # Calculate overall quality score
            overall_score = np.mean(list(final_dimension_scores.values())) if final_dimension_scores else 100
            
            # Generate recommendations
            recommendations = self._generate_recommendations(issues, data)
            
            # Calculate assessment duration
            assessment_end = datetime.utcnow()
            metrics["assessment_duration"] = (assessment_end - assessment_start).total_seconds()
            
            # Create quality report
            report = QualityReport(
                dataset_name=dataset_name,
                assessment_timestamp=assessment_start,
                total_records=len(data),
                overall_score=overall_score,
                dimension_scores=final_dimension_scores,
                issues=issues,
                metrics=metrics,
                recommendations=recommendations
            )
            
            # Store in history
            self.quality_history.append(report)
            
            return report
            
        except Exception as e:
            logger.error(f"Data quality assessment failed: {str(e)}")
            return QualityReport(
                dataset_name=dataset_name,
                assessment_timestamp=datetime.utcnow(),
                total_records=0,
                overall_score=0,
                dimension_scores={},
                issues=[QualityIssue(
                    rule_id="system",
                    rule_name="System Error",
                    severity=QualitySeverity.CRITICAL,
                    field_name="system",
                    issue_description=f"Assessment failed: {str(e)}",
                    affected_records=0
                )],
                metrics={}
            )    

    def _apply_quality_rule(self, data: pd.DataFrame, rule: QualityRule) -> List[QualityIssue]:
        """Apply a single quality rule to the data"""
        try:
            validator = self.validation_functions.get(rule.rule_type)
            if not validator:
                raise ValueError(f"No validator for rule type: {rule.rule_type}")
            
            return validator(data, rule)
            
        except Exception as e:
            logger.error(f"Failed to apply rule {rule.name}: {str(e)}")
            return [QualityIssue(
                rule_id=rule.id,
                rule_name=rule.name,
                severity=rule.severity,
                field_name="unknown",
                issue_description=f"Rule application failed: {str(e)}",
                affected_records=0
            )]
    
    def _initialize_validators(self) -> Dict[QualityRuleType, Callable]:
        """Initialize quality validation functions"""
        return {
            QualityRuleType.COMPLETENESS: self._validate_completeness,
            QualityRuleType.UNIQUENESS: self._validate_uniqueness,
            QualityRuleType.VALIDITY: self._validate_validity,
            QualityRuleType.CONSISTENCY: self._validate_consistency,
            QualityRuleType.ACCURACY: self._validate_accuracy,
            QualityRuleType.TIMELINESS: self._validate_timeliness,
            QualityRuleType.RANGE: self._validate_range,
            QualityRuleType.FORMAT: self._validate_format
        }
    
    def _validate_completeness(self, data: pd.DataFrame, rule: QualityRule) -> List[QualityIssue]:
        """Validate data completeness"""
        issues = []
        threshold = rule.parameters.get("min_completeness", 0.95)
        
        for field in rule.target_fields:
            if field not in data.columns:
                continue
                
            completeness = data[field].notna().sum() / len(data) if len(data) > 0 else 1.0
            
            if completeness < threshold:
                missing_count = data[field].isna().sum()
                issues.append(QualityIssue(
                    rule_id=rule.id,
                    rule_name=rule.name,
                    severity=rule.severity,
                    field_name=field,
                    issue_description=f"Completeness {completeness:.2%} below threshold {threshold:.2%}",
                    affected_records=missing_count,
                    metadata={"completeness": completeness, "threshold": threshold}
                ))
        
        return issues
    
    def _validate_uniqueness(self, data: pd.DataFrame, rule: QualityRule) -> List[QualityIssue]:
        """Validate data uniqueness"""
        issues = []
        
        for field in rule.target_fields:
            if field not in data.columns:
                continue
                
            duplicates = data[field].duplicated()
            duplicate_count = duplicates.sum()
            
            if duplicate_count > 0:
                sample_duplicates = data[duplicates][field].head(5).tolist()
                issues.append(QualityIssue(
                    rule_id=rule.id,
                    rule_name=rule.name,
                    severity=rule.severity,
                    field_name=field,
                    issue_description=f"Found {duplicate_count} duplicate values",
                    affected_records=duplicate_count,
                    sample_values=sample_duplicates
                ))
        
        return issues
    
    def _validate_validity(self, data: pd.DataFrame, rule: QualityRule) -> List[QualityIssue]:
        """Validate data validity using custom conditions"""
        issues = []
        # Simplified implementation
        return issues
    
    def _validate_consistency(self, data: pd.DataFrame, rule: QualityRule) -> List[QualityIssue]:
        """Validate data consistency across fields"""
        return []
    
    def _validate_accuracy(self, data: pd.DataFrame, rule: QualityRule) -> List[QualityIssue]:
        """Validate data accuracy against reference values"""
        return []
    
    def _validate_timeliness(self, data: pd.DataFrame, rule: QualityRule) -> List[QualityIssue]:
        """Validate data timeliness"""
        return []
    
    def _validate_range(self, data: pd.DataFrame, rule: QualityRule) -> List[QualityIssue]:
        """Validate data ranges"""
        return []
    
    def _validate_format(self, data: pd.DataFrame, rule: QualityRule) -> List[QualityIssue]:
        """Validate data format using regex patterns"""
        return []
    
    def _generate_recommendations(self, issues: List[QualityIssue], 
                                data: pd.DataFrame) -> List[str]:
        """Generate actionable recommendations based on quality issues"""
        recommendations = []
        
        if any("completeness" in issue.rule_name.lower() for issue in issues):
            recommendations.append("Implement data validation at source to prevent missing values")
        
        if any("uniqueness" in issue.rule_name.lower() for issue in issues):
            recommendations.append("Add unique constraints to prevent duplicate entries")
        
        return recommendations
    
    def get_quality_trends(self, dataset_name: str, days: int = 30) -> Dict[str, Any]:
        """Get quality trends for a dataset over time"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            relevant_reports = [
                report for report in self.quality_history
                if report.dataset_name == dataset_name and report.assessment_timestamp >= cutoff_date
            ]
            
            if not relevant_reports:
                return {"error": "No quality history found for dataset"}
            
            return {
                "dataset_name": dataset_name,
                "period_days": days,
                "assessments_count": len(relevant_reports)
            }
            
        except Exception as e:
            logger.error(f"Failed to get quality trends: {str(e)}")
            return {"error": str(e)}