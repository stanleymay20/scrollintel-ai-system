"""
Data Quality Monitoring System with Automated Alerts
Monitors data quality metrics and triggers alerts for quality issues
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Callable, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import pandas as pd
import numpy as np
from sqlalchemy.orm import Session
import redis

from ..core.intelligent_alerting_system import IntelligentAlertingSystem, AlertSeverity, ThresholdRule
from ..core.notification_system import NotificationSystem, NotificationPriority

logger = logging.getLogger(__name__)

class DataQualityDimension(Enum):
    COMPLETENESS = "completeness"
    ACCURACY = "accuracy"
    CONSISTENCY = "consistency"
    VALIDITY = "validity"
    UNIQUENESS = "uniqueness"
    TIMELINESS = "timeliness"
    INTEGRITY = "integrity"

class QualityCheckType(Enum):
    NULL_CHECK = "null_check"
    RANGE_CHECK = "range_check"
    FORMAT_CHECK = "format_check"
    UNIQUENESS_CHECK = "uniqueness_check"
    REFERENTIAL_INTEGRITY = "referential_integrity"
    STATISTICAL_OUTLIER = "statistical_outlier"
    FRESHNESS_CHECK = "freshness_check"
    SCHEMA_VALIDATION = "schema_validation"
    BUSINESS_RULE = "business_rule"

class QualityStatus(Enum):
    EXCELLENT = "excellent"  # 95-100%
    GOOD = "good"           # 85-94%
    FAIR = "fair"           # 70-84%
    POOR = "poor"           # 50-69%
    CRITICAL = "critical"   # <50%

@dataclass
class QualityRule:
    """Data quality rule configuration"""
    id: str
    name: str
    description: str
    dimension: DataQualityDimension
    check_type: QualityCheckType
    table_name: str
    column_name: Optional[str] = None
    rule_config: Dict[str, Any] = None
    threshold_warning: float = 0.85  # 85%
    threshold_critical: float = 0.70  # 70%
    enabled: bool = True
    schedule_minutes: int = 60  # Run every hour
    
    def __post_init__(self):
        if self.rule_config is None:
            self.rule_config = {}
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            'dimension': self.dimension.value,
            'check_type': self.check_type.value
        }

@dataclass
class QualityMetric:
    """Data quality metric result"""
    id: str
    rule_id: str
    table_name: str
    column_name: Optional[str]
    dimension: DataQualityDimension
    check_type: QualityCheckType
    score: float  # 0.0 to 1.0
    status: QualityStatus
    total_records: int
    failed_records: int
    timestamp: datetime
    details: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.details is None:
            self.details = {}
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            'dimension': self.dimension.value,
            'check_type': self.check_type.value,
            'status': self.status.value,
            'timestamp': self.timestamp.isoformat()
        }

@dataclass
class QualityIssue:
    """Data quality issue detected"""
    id: str
    rule_id: str
    metric_id: str
    severity: AlertSeverity
    title: str
    description: str
    affected_records: int
    sample_records: List[Dict[str, Any]]
    detected_at: datetime
    resolved_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            'severity': self.severity.value,
            'detected_at': self.detected_at.isoformat(),
            'resolved_at': self.resolved_at.isoformat() if self.resolved_at else None
        }

class DataQualityMonitor:
    """
    Comprehensive data quality monitoring system with automated alerts
    """
    
    def __init__(self, redis_client: redis.Redis, 
                 alerting_system: IntelligentAlertingSystem,
                 notification_system: NotificationSystem,
                 database_session: Session = None):
        self.redis_client = redis_client
        self.alerting_system = alerting_system
        self.notification_system = notification_system
        self.db_session = database_session
        
        # Quality rules and metrics storage
        self.quality_rules: Dict[str, QualityRule] = {}
        self.quality_metrics: Dict[str, QualityMetric] = {}
        self.quality_issues: Dict[str, QualityIssue] = {}
        
        # Quality check implementations
        self.quality_checks: Dict[QualityCheckType, Callable] = {}
        
        # Monitoring state
        self.last_check_times: Dict[str, datetime] = {}
        self.quality_trends: Dict[str, List[float]] = {}
        
        # Background tasks
        self._running = False
        self._tasks = []
        
        # Statistics
        self.stats = {
            'total_checks': 0,
            'passed_checks': 0,
            'failed_checks': 0,
            'issues_detected': 0,
            'issues_resolved': 0
        }
        
        # Initialize quality check implementations
        self._setup_quality_checks()
    
    async def start(self):
        """Start the data quality monitoring system"""
        if self._running:
            return
            
        self._running = True
        logger.info("Starting data quality monitoring system")
        
        # Load existing rules and metrics
        await self._load_quality_rules()
        await self._load_quality_metrics()
        
        # Start background tasks
        monitor_task = asyncio.create_task(self._monitor_data_quality())
        self._tasks.append(monitor_task)
        
        trend_task = asyncio.create_task(self._analyze_quality_trends())
        self._tasks.append(trend_task)
        
        cleanup_task = asyncio.create_task(self._cleanup_old_metrics())
        self._tasks.append(cleanup_task)
        
        stats_task = asyncio.create_task(self._update_statistics())
        self._tasks.append(stats_task)
    
    async def stop(self):
        """Stop the data quality monitoring system"""
        self._running = False
        
        # Cancel all tasks
        for task in self._tasks:
            task.cancel()
        
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        
        logger.info("Data quality monitoring system stopped")
    
    # Quality Rule Management
    
    async def add_quality_rule(self, rule: QualityRule) -> bool:
        """Add a new data quality rule"""
        try:
            self.quality_rules[rule.id] = rule
            
            # Store in Redis
            await self.redis_client.hset(
                "data_quality:rules",
                rule.id,
                json.dumps(rule.to_dict())
            )
            
            # Create corresponding alert thresholds
            await self._create_alert_thresholds(rule)
            
            logger.info(f"Added data quality rule: {rule.id}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding quality rule: {str(e)}")
            return False
    
    async def remove_quality_rule(self, rule_id: str) -> bool:
        """Remove a data quality rule"""
        try:
            if rule_id in self.quality_rules:
                del self.quality_rules[rule_id]
                
                # Remove from Redis
                await self.redis_client.hdel("data_quality:rules", rule_id)
                
                # Remove alert thresholds
                await self.alerting_system.remove_threshold_rule(f"dq_warning_{rule_id}")
                await self.alerting_system.remove_threshold_rule(f"dq_critical_{rule_id}")
                
                logger.info(f"Removed data quality rule: {rule_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error removing quality rule: {str(e)}")
            return False
    
    # Quality Monitoring
    
    async def run_quality_check(self, rule_id: str) -> Optional[QualityMetric]:
        """Run a specific quality check"""
        try:
            rule = self.quality_rules.get(rule_id)
            if not rule or not rule.enabled:
                return None
            
            logger.debug(f"Running quality check: {rule_id}")
            
            # Get quality check implementation
            check_function = self.quality_checks.get(rule.check_type)
            if not check_function:
                logger.error(f"No implementation for check type: {rule.check_type}")
                return None
            
            # Run the quality check
            metric = await check_function(rule)
            
            if metric:
                # Store metric
                self.quality_metrics[metric.id] = metric
                
                # Persist to Redis
                await self.redis_client.hset(
                    "data_quality:metrics",
                    metric.id,
                    json.dumps(metric.to_dict())
                )
                
                # Update last check time
                self.last_check_times[rule_id] = datetime.now()
                
                # Check for quality issues
                await self._check_quality_thresholds(metric)
                
                # Update trends
                await self._update_quality_trends(rule_id, metric.score)
                
                # Update statistics
                self.stats['total_checks'] += 1
                if metric.score >= rule.threshold_warning:
                    self.stats['passed_checks'] += 1
                else:
                    self.stats['failed_checks'] += 1
                
                logger.debug(f"Quality check completed: {rule_id}, Score: {metric.score:.3f}")
                return metric
            
        except Exception as e:
            logger.error(f"Error running quality check {rule_id}: {str(e)}")
            return None
    
    async def run_all_quality_checks(self) -> List[QualityMetric]:
        """Run all enabled quality checks"""
        results = []
        
        for rule_id, rule in self.quality_rules.items():
            if rule.enabled:
                metric = await self.run_quality_check(rule_id)
                if metric:
                    results.append(metric)
        
        return results
    
    # Quality Check Implementations
    
    def _setup_quality_checks(self):
        """Setup quality check implementations"""
        self.quality_checks[QualityCheckType.NULL_CHECK] = self._check_completeness
        self.quality_checks[QualityCheckType.RANGE_CHECK] = self._check_range_validity
        self.quality_checks[QualityCheckType.FORMAT_CHECK] = self._check_format_validity
        self.quality_checks[QualityCheckType.UNIQUENESS_CHECK] = self._check_uniqueness
        self.quality_checks[QualityCheckType.REFERENTIAL_INTEGRITY] = self._check_referential_integrity
        self.quality_checks[QualityCheckType.STATISTICAL_OUTLIER] = self._check_statistical_outliers
        self.quality_checks[QualityCheckType.FRESHNESS_CHECK] = self._check_data_freshness
        self.quality_checks[QualityCheckType.SCHEMA_VALIDATION] = self._check_schema_validation
        self.quality_checks[QualityCheckType.BUSINESS_RULE] = self._check_business_rules
    
    async def _check_completeness(self, rule: QualityRule) -> Optional[QualityMetric]:
        """Check data completeness (null values)"""
        try:
            # This would typically query the actual database
            # For demo purposes, we'll simulate the check
            
            total_records = 1000  # Simulated
            null_records = 50     # Simulated
            
            score = (total_records - null_records) / total_records if total_records > 0 else 0
            status = self._calculate_quality_status(score, rule)
            
            metric = QualityMetric(
                id=f"metric_{datetime.now().timestamp()}_{rule.id}",
                rule_id=rule.id,
                table_name=rule.table_name,
                column_name=rule.column_name,
                dimension=rule.dimension,
                check_type=rule.check_type,
                score=score,
                status=status,
                total_records=total_records,
                failed_records=null_records,
                timestamp=datetime.now(),
                details={
                    'null_count': null_records,
                    'null_percentage': (null_records / total_records) * 100
                }
            )
            
            return metric
            
        except Exception as e:
            logger.error(f"Error in completeness check: {str(e)}")
            return None
    
    async def _check_range_validity(self, rule: QualityRule) -> Optional[QualityMetric]:
        """Check if values are within expected ranges"""
        try:
            config = rule.rule_config
            min_value = config.get('min_value')
            max_value = config.get('max_value')
            
            # Simulated data check
            total_records = 1000
            out_of_range_records = 25
            
            score = (total_records - out_of_range_records) / total_records if total_records > 0 else 0
            status = self._calculate_quality_status(score, rule)
            
            metric = QualityMetric(
                id=f"metric_{datetime.now().timestamp()}_{rule.id}",
                rule_id=rule.id,
                table_name=rule.table_name,
                column_name=rule.column_name,
                dimension=rule.dimension,
                check_type=rule.check_type,
                score=score,
                status=status,
                total_records=total_records,
                failed_records=out_of_range_records,
                timestamp=datetime.now(),
                details={
                    'min_value': min_value,
                    'max_value': max_value,
                    'out_of_range_count': out_of_range_records
                }
            )
            
            return metric
            
        except Exception as e:
            logger.error(f"Error in range validity check: {str(e)}")
            return None
    
    async def _check_format_validity(self, rule: QualityRule) -> Optional[QualityMetric]:
        """Check if values match expected format patterns"""
        try:
            config = rule.rule_config
            pattern = config.get('pattern')
            
            # Simulated format validation
            total_records = 1000
            invalid_format_records = 15
            
            score = (total_records - invalid_format_records) / total_records if total_records > 0 else 0
            status = self._calculate_quality_status(score, rule)
            
            metric = QualityMetric(
                id=f"metric_{datetime.now().timestamp()}_{rule.id}",
                rule_id=rule.id,
                table_name=rule.table_name,
                column_name=rule.column_name,
                dimension=rule.dimension,
                check_type=rule.check_type,
                score=score,
                status=status,
                total_records=total_records,
                failed_records=invalid_format_records,
                timestamp=datetime.now(),
                details={
                    'pattern': pattern,
                    'invalid_format_count': invalid_format_records
                }
            )
            
            return metric
            
        except Exception as e:
            logger.error(f"Error in format validity check: {str(e)}")
            return None
    
    async def _check_uniqueness(self, rule: QualityRule) -> Optional[QualityMetric]:
        """Check for duplicate values"""
        try:
            # Simulated uniqueness check
            total_records = 1000
            duplicate_records = 8
            
            score = (total_records - duplicate_records) / total_records if total_records > 0 else 0
            status = self._calculate_quality_status(score, rule)
            
            metric = QualityMetric(
                id=f"metric_{datetime.now().timestamp()}_{rule.id}",
                rule_id=rule.id,
                table_name=rule.table_name,
                column_name=rule.column_name,
                dimension=rule.dimension,
                check_type=rule.check_type,
                score=score,
                status=status,
                total_records=total_records,
                failed_records=duplicate_records,
                timestamp=datetime.now(),
                details={
                    'duplicate_count': duplicate_records,
                    'unique_values': total_records - duplicate_records
                }
            )
            
            return metric
            
        except Exception as e:
            logger.error(f"Error in uniqueness check: {str(e)}")
            return None
    
    async def _check_referential_integrity(self, rule: QualityRule) -> Optional[QualityMetric]:
        """Check referential integrity between tables"""
        try:
            config = rule.rule_config
            reference_table = config.get('reference_table')
            reference_column = config.get('reference_column')
            
            # Simulated referential integrity check
            total_records = 1000
            orphaned_records = 5
            
            score = (total_records - orphaned_records) / total_records if total_records > 0 else 0
            status = self._calculate_quality_status(score, rule)
            
            metric = QualityMetric(
                id=f"metric_{datetime.now().timestamp()}_{rule.id}",
                rule_id=rule.id,
                table_name=rule.table_name,
                column_name=rule.column_name,
                dimension=rule.dimension,
                check_type=rule.check_type,
                score=score,
                status=status,
                total_records=total_records,
                failed_records=orphaned_records,
                timestamp=datetime.now(),
                details={
                    'reference_table': reference_table,
                    'reference_column': reference_column,
                    'orphaned_records': orphaned_records
                }
            )
            
            return metric
            
        except Exception as e:
            logger.error(f"Error in referential integrity check: {str(e)}")
            return None
    
    async def _check_statistical_outliers(self, rule: QualityRule) -> Optional[QualityMetric]:
        """Check for statistical outliers"""
        try:
            config = rule.rule_config
            std_dev_threshold = config.get('std_dev_threshold', 3)
            
            # Simulated outlier detection
            total_records = 1000
            outlier_records = 12
            
            score = (total_records - outlier_records) / total_records if total_records > 0 else 0
            status = self._calculate_quality_status(score, rule)
            
            metric = QualityMetric(
                id=f"metric_{datetime.now().timestamp()}_{rule.id}",
                rule_id=rule.id,
                table_name=rule.table_name,
                column_name=rule.column_name,
                dimension=rule.dimension,
                check_type=rule.check_type,
                score=score,
                status=status,
                total_records=total_records,
                failed_records=outlier_records,
                timestamp=datetime.now(),
                details={
                    'std_dev_threshold': std_dev_threshold,
                    'outlier_count': outlier_records
                }
            )
            
            return metric
            
        except Exception as e:
            logger.error(f"Error in statistical outlier check: {str(e)}")
            return None
    
    async def _check_data_freshness(self, rule: QualityRule) -> Optional[QualityMetric]:
        """Check data freshness/timeliness"""
        try:
            config = rule.rule_config
            max_age_hours = config.get('max_age_hours', 24)
            timestamp_column = config.get('timestamp_column', 'created_at')
            
            # Simulated freshness check
            total_records = 1000
            stale_records = 30
            
            score = (total_records - stale_records) / total_records if total_records > 0 else 0
            status = self._calculate_quality_status(score, rule)
            
            metric = QualityMetric(
                id=f"metric_{datetime.now().timestamp()}_{rule.id}",
                rule_id=rule.id,
                table_name=rule.table_name,
                column_name=timestamp_column,
                dimension=rule.dimension,
                check_type=rule.check_type,
                score=score,
                status=status,
                total_records=total_records,
                failed_records=stale_records,
                timestamp=datetime.now(),
                details={
                    'max_age_hours': max_age_hours,
                    'stale_records': stale_records,
                    'timestamp_column': timestamp_column
                }
            )
            
            return metric
            
        except Exception as e:
            logger.error(f"Error in data freshness check: {str(e)}")
            return None
    
    async def _check_schema_validation(self, rule: QualityRule) -> Optional[QualityMetric]:
        """Check schema validation"""
        try:
            # Simulated schema validation
            total_records = 1000
            schema_violations = 3
            
            score = (total_records - schema_violations) / total_records if total_records > 0 else 0
            status = self._calculate_quality_status(score, rule)
            
            metric = QualityMetric(
                id=f"metric_{datetime.now().timestamp()}_{rule.id}",
                rule_id=rule.id,
                table_name=rule.table_name,
                column_name=rule.column_name,
                dimension=rule.dimension,
                check_type=rule.check_type,
                score=score,
                status=status,
                total_records=total_records,
                failed_records=schema_violations,
                timestamp=datetime.now(),
                details={
                    'schema_violations': schema_violations
                }
            )
            
            return metric
            
        except Exception as e:
            logger.error(f"Error in schema validation check: {str(e)}")
            return None
    
    async def _check_business_rules(self, rule: QualityRule) -> Optional[QualityMetric]:
        """Check custom business rules"""
        try:
            config = rule.rule_config
            business_rule = config.get('rule_expression')
            
            # Simulated business rule validation
            total_records = 1000
            rule_violations = 18
            
            score = (total_records - rule_violations) / total_records if total_records > 0 else 0
            status = self._calculate_quality_status(score, rule)
            
            metric = QualityMetric(
                id=f"metric_{datetime.now().timestamp()}_{rule.id}",
                rule_id=rule.id,
                table_name=rule.table_name,
                column_name=rule.column_name,
                dimension=rule.dimension,
                check_type=rule.check_type,
                score=score,
                status=status,
                total_records=total_records,
                failed_records=rule_violations,
                timestamp=datetime.now(),
                details={
                    'business_rule': business_rule,
                    'rule_violations': rule_violations
                }
            )
            
            return metric
            
        except Exception as e:
            logger.error(f"Error in business rule check: {str(e)}")
            return None
    
    # Helper Methods
    
    def _calculate_quality_status(self, score: float, rule: QualityRule) -> QualityStatus:
        """Calculate quality status based on score and thresholds"""
        if score >= 0.95:
            return QualityStatus.EXCELLENT
        elif score >= rule.threshold_warning:
            return QualityStatus.GOOD
        elif score >= rule.threshold_critical:
            return QualityStatus.FAIR
        elif score >= 0.50:
            return QualityStatus.POOR
        else:
            return QualityStatus.CRITICAL
    
    async def _create_alert_thresholds(self, rule: QualityRule):
        """Create alert thresholds for quality rule"""
        try:
            # Warning threshold
            warning_rule = ThresholdRule(
                id=f"dq_warning_{rule.id}",
                metric_name=f"data_quality_{rule.id}",
                operator="<",
                value=rule.threshold_warning,
                severity=AlertSeverity.MEDIUM,
                description=f"Data quality warning for {rule.name}",
                cooldown_minutes=30
            )
            
            await self.alerting_system.add_threshold_rule(warning_rule)
            
            # Critical threshold
            critical_rule = ThresholdRule(
                id=f"dq_critical_{rule.id}",
                metric_name=f"data_quality_{rule.id}",
                operator="<",
                value=rule.threshold_critical,
                severity=AlertSeverity.HIGH,
                description=f"Critical data quality issue for {rule.name}",
                cooldown_minutes=15
            )
            
            await self.alerting_system.add_threshold_rule(critical_rule)
            
        except Exception as e:
            logger.error(f"Error creating alert thresholds: {str(e)}")
    
    async def _check_quality_thresholds(self, metric: QualityMetric):
        """Check quality metric against thresholds and trigger alerts"""
        try:
            rule = self.quality_rules.get(metric.rule_id)
            if not rule:
                return
            
            # Check thresholds using the alerting system
            await self.alerting_system.check_thresholds(
                metric_name=f"data_quality_{rule.id}",
                value=metric.score,
                timestamp=metric.timestamp,
                context={
                    'table_name': metric.table_name,
                    'column_name': metric.column_name,
                    'dimension': metric.dimension.value,
                    'check_type': metric.check_type.value,
                    'total_records': metric.total_records,
                    'failed_records': metric.failed_records,
                    'details': metric.details
                }
            )
            
            # Create quality issue if score is below critical threshold
            if metric.score < rule.threshold_critical:
                await self._create_quality_issue(metric, rule)
            
        except Exception as e:
            logger.error(f"Error checking quality thresholds: {str(e)}")
    
    async def _create_quality_issue(self, metric: QualityMetric, rule: QualityRule):
        """Create a quality issue record"""
        try:
            severity = AlertSeverity.CRITICAL if metric.score < 0.5 else AlertSeverity.HIGH
            
            issue = QualityIssue(
                id=f"issue_{datetime.now().timestamp()}_{metric.id}",
                rule_id=rule.id,
                metric_id=metric.id,
                severity=severity,
                title=f"Data Quality Issue: {rule.name}",
                description=f"Quality score ({metric.score:.3f}) below threshold ({rule.threshold_critical}) for {rule.table_name}.{rule.column_name or 'table'}",
                affected_records=metric.failed_records,
                sample_records=[],  # Would contain sample problematic records
                detected_at=datetime.now()
            )
            
            self.quality_issues[issue.id] = issue
            
            # Store in Redis
            await self.redis_client.hset(
                "data_quality:issues",
                issue.id,
                json.dumps(issue.to_dict())
            )
            
            # Send notification
            await self.notification_system.send_notification(
                'data_quality_issue',
                {
                    'issue': issue.to_dict(),
                    'metric': metric.to_dict(),
                    'rule': rule.to_dict()
                },
                NotificationPriority.HIGH if severity == AlertSeverity.HIGH else NotificationPriority.CRITICAL
            )
            
            # Update statistics
            self.stats['issues_detected'] += 1
            
            logger.warning(f"Data quality issue detected: {issue.id}")
            
        except Exception as e:
            logger.error(f"Error creating quality issue: {str(e)}")
    
    async def _update_quality_trends(self, rule_id: str, score: float):
        """Update quality trends for analysis"""
        try:
            if rule_id not in self.quality_trends:
                self.quality_trends[rule_id] = []
            
            # Keep last 100 scores for trend analysis
            self.quality_trends[rule_id].append(score)
            if len(self.quality_trends[rule_id]) > 100:
                self.quality_trends[rule_id].pop(0)
            
            # Store trends in Redis
            await self.redis_client.setex(
                f"data_quality:trends:{rule_id}",
                86400,  # 24 hours TTL
                json.dumps(self.quality_trends[rule_id])
            )
            
        except Exception as e:
            logger.error(f"Error updating quality trends: {str(e)}")
    
    # Background Tasks
    
    async def _monitor_data_quality(self):
        """Background task to monitor data quality"""
        while self._running:
            try:
                current_time = datetime.now()
                
                # Check which rules need to be executed
                for rule_id, rule in self.quality_rules.items():
                    if not rule.enabled:
                        continue
                    
                    last_check = self.last_check_times.get(rule_id)
                    
                    # Check if it's time to run this rule
                    if (not last_check or 
                        (current_time - last_check).total_seconds() >= rule.schedule_minutes * 60):
                        
                        await self.run_quality_check(rule_id)
                
                # Sleep for 1 minute before next check
                await asyncio.sleep(60)
                
            except Exception as e:
                logger.error(f"Error in quality monitoring task: {str(e)}")
                await asyncio.sleep(60)
    
    async def _analyze_quality_trends(self):
        """Analyze quality trends and detect degradation"""
        while self._running:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                
                for rule_id, scores in self.quality_trends.items():
                    if len(scores) >= 10:  # Need at least 10 data points
                        # Simple trend analysis
                        recent_scores = scores[-5:]  # Last 5 scores
                        older_scores = scores[-10:-5]  # Previous 5 scores
                        
                        recent_avg = sum(recent_scores) / len(recent_scores)
                        older_avg = sum(older_scores) / len(older_scores)
                        
                        # Check for significant degradation (>5% drop)
                        if recent_avg < older_avg - 0.05:
                            await self._alert_quality_degradation(rule_id, recent_avg, older_avg)
                
            except Exception as e:
                logger.error(f"Error in trend analysis task: {str(e)}")
    
    async def _alert_quality_degradation(self, rule_id: str, recent_avg: float, older_avg: float):
        """Alert on quality degradation trend"""
        try:
            rule = self.quality_rules.get(rule_id)
            if not rule:
                return
            
            degradation_pct = ((older_avg - recent_avg) / older_avg) * 100
            
            await self.notification_system.send_notification(
                'quality_degradation',
                {
                    'rule_name': rule.name,
                    'table_name': rule.table_name,
                    'column_name': rule.column_name,
                    'recent_average': recent_avg,
                    'previous_average': older_avg,
                    'degradation_percentage': degradation_pct
                },
                NotificationPriority.MEDIUM
            )
            
            logger.warning(f"Quality degradation detected for {rule_id}: {degradation_pct:.1f}% drop")
            
        except Exception as e:
            logger.error(f"Error alerting quality degradation: {str(e)}")
    
    async def _cleanup_old_metrics(self):
        """Clean up old quality metrics"""
        while self._running:
            try:
                await asyncio.sleep(3600)  # Run every hour
                
                # Clean up metrics older than 30 days
                cutoff_time = datetime.now() - timedelta(days=30)
                
                metrics_to_remove = []
                for metric_id, metric in self.quality_metrics.items():
                    if metric.timestamp < cutoff_time:
                        metrics_to_remove.append(metric_id)
                
                # Remove old metrics
                for metric_id in metrics_to_remove:
                    del self.quality_metrics[metric_id]
                    await self.redis_client.hdel("data_quality:metrics", metric_id)
                
                if metrics_to_remove:
                    logger.info(f"Cleaned up {len(metrics_to_remove)} old quality metrics")
                
            except Exception as e:
                logger.error(f"Error in cleanup task: {str(e)}")
    
    async def _update_statistics(self):
        """Update quality monitoring statistics"""
        while self._running:
            try:
                await asyncio.sleep(60)  # Update every minute
                
                # Calculate current statistics
                current_stats = {
                    **self.stats,
                    'active_rules': len([r for r in self.quality_rules.values() if r.enabled]),
                    'total_rules': len(self.quality_rules),
                    'active_issues': len([i for i in self.quality_issues.values() if not i.resolved_at]),
                    'last_updated': datetime.now().isoformat()
                }
                
                # Store in Redis
                await self.redis_client.setex(
                    "data_quality:statistics",
                    300,  # 5 minutes TTL
                    json.dumps(current_stats)
                )
                
            except Exception as e:
                logger.error(f"Error updating statistics: {str(e)}")
    
    async def _load_quality_rules(self):
        """Load quality rules from Redis"""
        try:
            rules = await self.redis_client.hgetall("data_quality:rules")
            
            for rule_id, rule_data in rules.items():
                try:
                    rule_dict = json.loads(rule_data)
                    rule_dict['dimension'] = DataQualityDimension(rule_dict['dimension'])
                    rule_dict['check_type'] = QualityCheckType(rule_dict['check_type'])
                    rule = QualityRule(**rule_dict)
                    self.quality_rules[rule_id] = rule
                except Exception as e:
                    logger.error(f"Error loading quality rule {rule_id}: {str(e)}")
            
            logger.info(f"Loaded {len(self.quality_rules)} quality rules")
            
        except Exception as e:
            logger.error(f"Error loading quality rules: {str(e)}")
    
    async def _load_quality_metrics(self):
        """Load recent quality metrics from Redis"""
        try:
            metrics = await self.redis_client.hgetall("data_quality:metrics")
            
            # Load only recent metrics (last 7 days)
            cutoff_time = datetime.now() - timedelta(days=7)
            
            for metric_id, metric_data in metrics.items():
                try:
                    metric_dict = json.loads(metric_data)
                    metric_dict['timestamp'] = datetime.fromisoformat(metric_dict['timestamp'])
                    
                    if metric_dict['timestamp'] >= cutoff_time:
                        metric_dict['dimension'] = DataQualityDimension(metric_dict['dimension'])
                        metric_dict['check_type'] = QualityCheckType(metric_dict['check_type'])
                        metric_dict['status'] = QualityStatus(metric_dict['status'])
                        metric = QualityMetric(**metric_dict)
                        self.quality_metrics[metric_id] = metric
                        
                except Exception as e:
                    logger.error(f"Error loading quality metric {metric_id}: {str(e)}")
            
            logger.info(f"Loaded {len(self.quality_metrics)} recent quality metrics")
            
        except Exception as e:
            logger.error(f"Error loading quality metrics: {str(e)}")
    
    # Public API Methods
    
    async def get_quality_summary(self) -> Dict[str, Any]:
        """Get overall data quality summary"""
        try:
            # Calculate overall quality score
            if not self.quality_metrics:
                return {'overall_score': 0, 'status': 'no_data'}
            
            recent_metrics = [
                m for m in self.quality_metrics.values()
                if (datetime.now() - m.timestamp).total_seconds() <= 86400  # Last 24 hours
            ]
            
            if not recent_metrics:
                return {'overall_score': 0, 'status': 'stale_data'}
            
            overall_score = sum(m.score for m in recent_metrics) / len(recent_metrics)
            
            # Count by status
            status_counts = {}
            for metric in recent_metrics:
                status = metric.status.value
                status_counts[status] = status_counts.get(status, 0) + 1
            
            return {
                'overall_score': overall_score,
                'status': self._calculate_quality_status(overall_score, None).value,
                'total_checks': len(recent_metrics),
                'status_distribution': status_counts,
                'active_issues': len([i for i in self.quality_issues.values() if not i.resolved_at]),
                'last_updated': max(m.timestamp for m in recent_metrics).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting quality summary: {str(e)}")
            return {'error': str(e)}
    
    async def get_quality_metrics(self, rule_id: str = None, 
                                hours: int = 24) -> List[Dict[str, Any]]:
        """Get quality metrics for a specific rule or all rules"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            metrics = []
            for metric in self.quality_metrics.values():
                if metric.timestamp >= cutoff_time:
                    if rule_id is None or metric.rule_id == rule_id:
                        metrics.append(metric.to_dict())
            
            return sorted(metrics, key=lambda m: m['timestamp'], reverse=True)
            
        except Exception as e:
            logger.error(f"Error getting quality metrics: {str(e)}")
            return []
    
    async def get_quality_issues(self, resolved: bool = False) -> List[Dict[str, Any]]:
        """Get quality issues"""
        try:
            issues = []
            for issue in self.quality_issues.values():
                if resolved or not issue.resolved_at:
                    issues.append(issue.to_dict())
            
            return sorted(issues, key=lambda i: i['detected_at'], reverse=True)
            
        except Exception as e:
            logger.error(f"Error getting quality issues: {str(e)}")
            return []
    
    async def resolve_quality_issue(self, issue_id: str, resolved_by: str = None) -> bool:
        """Resolve a quality issue"""
        try:
            if issue_id in self.quality_issues:
                issue = self.quality_issues[issue_id]
                issue.resolved_at = datetime.now()
                
                # Update in Redis
                await self.redis_client.hset(
                    "data_quality:issues",
                    issue_id,
                    json.dumps(issue.to_dict())
                )
                
                # Update statistics
                self.stats['issues_resolved'] += 1
                
                logger.info(f"Quality issue resolved: {issue_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error resolving quality issue: {str(e)}")
            return False