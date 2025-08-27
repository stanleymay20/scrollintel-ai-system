"""
Data Quality Monitoring with Automated Alerts
Real-time monitoring of data quality metrics with intelligent alerting
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, Column, String, Float, DateTime, Boolean, Text, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import aioredis
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import great_expectations as ge
from great_expectations.core import ExpectationSuite
from great_expectations.dataset import PandasDataset

logger = logging.getLogger(__name__)

Base = declarative_base()

class DataQualityIssueType(Enum):
    MISSING_VALUES = "missing_values"
    DUPLICATE_RECORDS = "duplicate_records"
    OUTLIERS = "outliers"
    SCHEMA_VIOLATION = "schema_violation"
    FRESHNESS = "freshness"
    COMPLETENESS = "completeness"
    ACCURACY = "accuracy"
    CONSISTENCY = "consistency"
    VALIDITY = "validity"

class QualityCheckSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class DataQualityRule:
    name: str
    description: str
    rule_type: DataQualityIssueType
    severity: QualityCheckSeverity
    threshold: float
    enabled: bool = True
    metadata: Dict[str, Any] = None

@dataclass
class DataQualityIssue:
    id: str
    rule_name: str
    issue_type: DataQualityIssueType
    severity: QualityCheckSeverity
    description: str
    affected_records: int
    total_records: int
    quality_score: float
    detected_at: datetime
    source_table: str
    source_column: Optional[str] = None
    metadata: Dict[str, Any] = None

@dataclass
class DataQualityMetrics:
    source_name: str
    table_name: str
    total_records: int
    completeness_score: float
    accuracy_score: float
    consistency_score: float
    validity_score: float
    freshness_score: float
    overall_score: float
    issues_count: int
    critical_issues_count: int
    measured_at: datetime

class DataQualityIssueModel(Base):
    __tablename__ = 'data_quality_issues'
    
    id = Column(String, primary_key=True)
    rule_name = Column(String, nullable=False)
    issue_type = Column(String, nullable=False)
    severity = Column(String, nullable=False)
    description = Column(Text)
    affected_records = Column(Integer)
    total_records = Column(Integer)
    quality_score = Column(Float)
    detected_at = Column(DateTime, nullable=False)
    source_table = Column(String, nullable=False)
    source_column = Column(String)
    metadata = Column(Text)  # JSON string

class DataQualityMetricsModel(Base):
    __tablename__ = 'data_quality_metrics'
    
    id = Column(String, primary_key=True)
    source_name = Column(String, nullable=False)
    table_name = Column(String, nullable=False)
    total_records = Column(Integer)
    completeness_score = Column(Float)
    accuracy_score = Column(Float)
    consistency_score = Column(Float)
    validity_score = Column(Float)
    freshness_score = Column(Float)
    overall_score = Column(Float)
    issues_count = Column(Integer)
    critical_issues_count = Column(Integer)
    measured_at = Column(DateTime, nullable=False)

class DataQualityMonitor:
    """Real-time data quality monitoring system"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379", 
                 db_url: str = None):
        self.redis_url = redis_url
        self.db_url = db_url
        self.redis_client = None
        self.db_engine = None
        self.Session = None
        
        # Quality rules and configuration
        self.quality_rules: Dict[str, List[DataQualityRule]] = {}  # table_name -> rules
        self.expectation_suites: Dict[str, ExpectationSuite] = {}
        
        # Monitoring state
        self.active_issues: Dict[str, DataQualityIssue] = {}
        self.quality_metrics: Dict[str, DataQualityMetrics] = {}
        
        # ML models for anomaly detection
        self.anomaly_models: Dict[str, IsolationForest] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        
        # Callbacks for alerts
        self.alert_callbacks: List[Callable] = []
        
        self.running = False
    
    async def initialize(self):
        """Initialize the data quality monitor"""
        try:
            self.redis_client = await aioredis.from_url(self.redis_url)
            
            if self.db_url:
                self.db_engine = create_engine(self.db_url)
                Base.metadata.create_all(self.db_engine)
                self.Session = sessionmaker(bind=self.db_engine)
            
            # Load default quality rules
            self._load_default_rules()
            
            logger.info("Data quality monitor initialized")
        except Exception as e:
            logger.error(f"Failed to initialize data quality monitor: {e}")
            raise
    
    def _load_default_rules(self):
        """Load default data quality rules"""
        default_rules = [
            DataQualityRule(
                name="missing_values_check",
                description="Check for missing values in critical columns",
                rule_type=DataQualityIssueType.MISSING_VALUES,
                severity=QualityCheckSeverity.HIGH,
                threshold=0.05,  # 5% threshold
                metadata={"columns": ["id", "created_at", "user_id"]}
            ),
            DataQualityRule(
                name="duplicate_records_check",
                description="Check for duplicate records",
                rule_type=DataQualityIssueType.DUPLICATE_RECORDS,
                severity=QualityCheckSeverity.MEDIUM,
                threshold=0.01,  # 1% threshold
                metadata={"key_columns": ["id"]}
            ),
            DataQualityRule(
                name="freshness_check",
                description="Check data freshness",
                rule_type=DataQualityIssueType.FRESHNESS,
                severity=QualityCheckSeverity.CRITICAL,
                threshold=24,  # 24 hours
                metadata={"timestamp_column": "created_at"}
            ),
            DataQualityRule(
                name="outlier_detection",
                description="Detect statistical outliers",
                rule_type=DataQualityIssueType.OUTLIERS,
                severity=QualityCheckSeverity.LOW,
                threshold=0.1,  # 10% threshold
                metadata={"numeric_columns": ["amount", "quantity", "score"]}
            )
        ]
        
        # Apply default rules to all tables (can be overridden)
        for rule in default_rules:
            self.add_quality_rule("*", rule)
    
    async def start_monitoring(self):
        """Start the data quality monitoring"""
        self.running = True
        logger.info("Starting data quality monitoring")
        
        tasks = [
            asyncio.create_task(self._monitor_data_streams()),
            asyncio.create_task(self._periodic_quality_checks()),
            asyncio.create_task(self._train_anomaly_models()),
            asyncio.create_task(self._cleanup_old_issues())
        ]
        
        await asyncio.gather(*tasks)
    
    async def stop_monitoring(self):
        """Stop the data quality monitoring"""
        self.running = False
        logger.info("Stopping data quality monitoring")
    
    async def _monitor_data_streams(self):
        """Monitor incoming data streams for quality issues"""
        while self.running:
            try:
                # Listen for data ingestion events
                messages = await self.redis_client.xread(
                    {'data_ingestion': '$'}, 
                    count=10, 
                    block=1000
                )
                
                for stream, msgs in messages:
                    for msg_id, fields in msgs:
                        await self._process_data_event(msg_id, fields)
                        
            except Exception as e:
                logger.error(f"Error monitoring data streams: {e}")
                await asyncio.sleep(1)
    
    async def _process_data_event(self, msg_id: bytes, fields: Dict[bytes, bytes]):
        """Process data ingestion event"""
        try:
            decoded_fields = {k.decode(): v.decode() for k, v in fields.items()}
            
            table_name = decoded_fields.get('table_name')
            operation = decoded_fields.get('operation')  # insert, update, delete
            record_count = int(decoded_fields.get('record_count', 0))
            
            if table_name and operation == 'insert' and record_count > 0:
                # Trigger quality check for the table
                await self._check_table_quality(table_name)
                
        except Exception as e:
            logger.error(f"Error processing data event: {e}")
    
    async def _periodic_quality_checks(self):
        """Run periodic quality checks on all monitored tables"""
        while self.running:
            try:
                # Get list of tables to monitor
                tables_to_check = set()
                for table_pattern in self.quality_rules.keys():
                    if table_pattern != "*":
                        tables_to_check.add(table_pattern)
                
                # Also get tables from Redis
                table_keys = await self.redis_client.keys("table:*")
                for key in table_keys:
                    table_name = key.decode().split(":")[1]
                    tables_to_check.add(table_name)
                
                # Run quality checks
                for table_name in tables_to_check:
                    await self._check_table_quality(table_name)
                    await asyncio.sleep(1)  # Avoid overwhelming the system
                
                # Wait before next round
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in periodic quality checks: {e}")
                await asyncio.sleep(60)
    
    async def _check_table_quality(self, table_name: str):
        """Run quality checks on a specific table"""
        try:
            # Get table data (this would connect to your actual data source)
            data = await self._get_table_data(table_name)
            
            if data is None or data.empty:
                return
            
            # Get applicable rules
            rules = self._get_applicable_rules(table_name)
            
            # Run quality checks
            issues = []
            for rule in rules:
                if rule.enabled:
                    issue = await self._run_quality_check(table_name, data, rule)
                    if issue:
                        issues.append(issue)
            
            # Calculate overall quality metrics
            metrics = self._calculate_quality_metrics(table_name, data, issues)
            
            # Store metrics
            await self._store_quality_metrics(metrics)
            
            # Process issues
            for issue in issues:
                await self._process_quality_issue(issue)
            
            # Update Redis with latest metrics
            await self._update_redis_metrics(table_name, metrics)
            
        except Exception as e:
            logger.error(f"Error checking quality for table {table_name}: {e}")
    
    async def _get_table_data(self, table_name: str) -> Optional[pd.DataFrame]:
        """Get table data for quality checking"""
        try:
            # This is a placeholder - implement based on your data source
            # For example, if using SQL database:
            if self.db_engine:
                query = f"SELECT * FROM {table_name} ORDER BY created_at DESC LIMIT 10000"
                return pd.read_sql(query, self.db_engine)
            
            # For Redis/NoSQL sources, implement appropriate data retrieval
            return None
            
        except Exception as e:
            logger.error(f"Error getting data for table {table_name}: {e}")
            return None
    
    def _get_applicable_rules(self, table_name: str) -> List[DataQualityRule]:
        """Get quality rules applicable to a table"""
        rules = []
        
        # Add table-specific rules
        if table_name in self.quality_rules:
            rules.extend(self.quality_rules[table_name])
        
        # Add global rules
        if "*" in self.quality_rules:
            rules.extend(self.quality_rules["*"])
        
        return rules
    
    async def _run_quality_check(self, table_name: str, data: pd.DataFrame, 
                                rule: DataQualityRule) -> Optional[DataQualityIssue]:
        """Run a specific quality check"""
        try:
            if rule.rule_type == DataQualityIssueType.MISSING_VALUES:
                return await self._check_missing_values(table_name, data, rule)
            
            elif rule.rule_type == DataQualityIssueType.DUPLICATE_RECORDS:
                return await self._check_duplicate_records(table_name, data, rule)
            
            elif rule.rule_type == DataQualityIssueType.FRESHNESS:
                return await self._check_data_freshness(table_name, data, rule)
            
            elif rule.rule_type == DataQualityIssueType.OUTLIERS:
                return await self._check_outliers(table_name, data, rule)
            
            elif rule.rule_type == DataQualityIssueType.SCHEMA_VIOLATION:
                return await self._check_schema_violations(table_name, data, rule)
            
            elif rule.rule_type == DataQualityIssueType.VALIDITY:
                return await self._check_data_validity(table_name, data, rule)
            
            return None
            
        except Exception as e:
            logger.error(f"Error running quality check {rule.name}: {e}")
            return None
    
    async def _check_missing_values(self, table_name: str, data: pd.DataFrame, 
                                  rule: DataQualityRule) -> Optional[DataQualityIssue]:
        """Check for missing values"""
        columns = rule.metadata.get("columns", data.columns.tolist())
        
        total_records = len(data)
        missing_count = 0
        
        for column in columns:
            if column in data.columns:
                missing_count += data[column].isnull().sum()
        
        missing_rate = missing_count / (total_records * len(columns)) if total_records > 0 else 0
        
        if missing_rate > rule.threshold:
            return DataQualityIssue(
                id=f"{table_name}_{rule.name}_{int(datetime.now().timestamp())}",
                rule_name=rule.name,
                issue_type=rule.rule_type,
                severity=rule.severity,
                description=f"Missing values rate ({missing_rate:.2%}) exceeds threshold ({rule.threshold:.2%})",
                affected_records=missing_count,
                total_records=total_records,
                quality_score=1 - missing_rate,
                detected_at=datetime.now(),
                source_table=table_name,
                metadata={"missing_rate": missing_rate, "columns": columns}
            )
        
        return None
    
    async def _check_duplicate_records(self, table_name: str, data: pd.DataFrame, 
                                     rule: DataQualityRule) -> Optional[DataQualityIssue]:
        """Check for duplicate records"""
        key_columns = rule.metadata.get("key_columns", ["id"])
        
        # Filter to existing columns
        existing_key_columns = [col for col in key_columns if col in data.columns]
        
        if not existing_key_columns:
            return None
        
        total_records = len(data)
        duplicate_count = data.duplicated(subset=existing_key_columns).sum()
        duplicate_rate = duplicate_count / total_records if total_records > 0 else 0
        
        if duplicate_rate > rule.threshold:
            return DataQualityIssue(
                id=f"{table_name}_{rule.name}_{int(datetime.now().timestamp())}",
                rule_name=rule.name,
                issue_type=rule.rule_type,
                severity=rule.severity,
                description=f"Duplicate records rate ({duplicate_rate:.2%}) exceeds threshold ({rule.threshold:.2%})",
                affected_records=duplicate_count,
                total_records=total_records,
                quality_score=1 - duplicate_rate,
                detected_at=datetime.now(),
                source_table=table_name,
                metadata={"duplicate_rate": duplicate_rate, "key_columns": existing_key_columns}
            )
        
        return None
    
    async def _check_data_freshness(self, table_name: str, data: pd.DataFrame, 
                                  rule: DataQualityRule) -> Optional[DataQualityIssue]:
        """Check data freshness"""
        timestamp_column = rule.metadata.get("timestamp_column", "created_at")
        
        if timestamp_column not in data.columns:
            return None
        
        # Convert to datetime if needed
        try:
            data[timestamp_column] = pd.to_datetime(data[timestamp_column])
        except:
            return None
        
        latest_timestamp = data[timestamp_column].max()
        current_time = datetime.now()
        
        # Handle timezone-naive datetime
        if latest_timestamp.tzinfo is None:
            latest_timestamp = latest_timestamp.replace(tzinfo=None)
            current_time = current_time.replace(tzinfo=None)
        
        hours_old = (current_time - latest_timestamp).total_seconds() / 3600
        
        if hours_old > rule.threshold:
            return DataQualityIssue(
                id=f"{table_name}_{rule.name}_{int(datetime.now().timestamp())}",
                rule_name=rule.name,
                issue_type=rule.rule_type,
                severity=rule.severity,
                description=f"Data is {hours_old:.1f} hours old, exceeds threshold of {rule.threshold} hours",
                affected_records=len(data),
                total_records=len(data),
                quality_score=max(0, 1 - (hours_old / (rule.threshold * 2))),
                detected_at=datetime.now(),
                source_table=table_name,
                metadata={"hours_old": hours_old, "latest_timestamp": latest_timestamp.isoformat()}
            )
        
        return None
    
    async def _check_outliers(self, table_name: str, data: pd.DataFrame, 
                            rule: DataQualityRule) -> Optional[DataQualityIssue]:
        """Check for statistical outliers"""
        numeric_columns = rule.metadata.get("numeric_columns", [])
        
        # Get numeric columns that exist in data
        existing_numeric_columns = [col for col in numeric_columns if col in data.columns and data[col].dtype in ['int64', 'float64']]
        
        if not existing_numeric_columns:
            return None
        
        total_records = len(data)
        outlier_count = 0
        
        for column in existing_numeric_columns:
            # Use IQR method for outlier detection
            Q1 = data[column].quantile(0.25)
            Q3 = data[column].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
            outlier_count += len(outliers)
        
        outlier_rate = outlier_count / (total_records * len(existing_numeric_columns)) if total_records > 0 else 0
        
        if outlier_rate > rule.threshold:
            return DataQualityIssue(
                id=f"{table_name}_{rule.name}_{int(datetime.now().timestamp())}",
                rule_name=rule.name,
                issue_type=rule.rule_type,
                severity=rule.severity,
                description=f"Outlier rate ({outlier_rate:.2%}) exceeds threshold ({rule.threshold:.2%})",
                affected_records=outlier_count,
                total_records=total_records,
                quality_score=1 - outlier_rate,
                detected_at=datetime.now(),
                source_table=table_name,
                metadata={"outlier_rate": outlier_rate, "columns": existing_numeric_columns}
            )
        
        return None
    
    async def _check_schema_violations(self, table_name: str, data: pd.DataFrame, 
                                     rule: DataQualityRule) -> Optional[DataQualityIssue]:
        """Check for schema violations"""
        # Implement schema validation logic
        return None
    
    async def _check_data_validity(self, table_name: str, data: pd.DataFrame, 
                                 rule: DataQualityRule) -> Optional[DataQualityIssue]:
        """Check data validity using Great Expectations"""
        try:
            # Convert to Great Expectations dataset
            ge_data = PandasDataset(data)
            
            # Get expectation suite for table
            suite = self.expectation_suites.get(table_name)
            if not suite:
                return None
            
            # Run expectations
            results = ge_data.validate(expectation_suite=suite)
            
            if not results.success:
                failed_expectations = len([r for r in results.results if not r.success])
                total_expectations = len(results.results)
                
                failure_rate = failed_expectations / total_expectations if total_expectations > 0 else 0
                
                if failure_rate > rule.threshold:
                    return DataQualityIssue(
                        id=f"{table_name}_{rule.name}_{int(datetime.now().timestamp())}",
                        rule_name=rule.name,
                        issue_type=rule.rule_type,
                        severity=rule.severity,
                        description=f"Data validation failure rate ({failure_rate:.2%}) exceeds threshold",
                        affected_records=len(data),
                        total_records=len(data),
                        quality_score=1 - failure_rate,
                        detected_at=datetime.now(),
                        source_table=table_name,
                        metadata={"failed_expectations": failed_expectations, "total_expectations": total_expectations}
                    )
            
            return None
            
        except Exception as e:
            logger.error(f"Error in data validity check: {e}")
            return None
    
    def _calculate_quality_metrics(self, table_name: str, data: pd.DataFrame, 
                                 issues: List[DataQualityIssue]) -> DataQualityMetrics:
        """Calculate overall quality metrics"""
        total_records = len(data)
        
        # Calculate individual scores
        completeness_score = 1.0 - (data.isnull().sum().sum() / (total_records * len(data.columns)) if total_records > 0 else 0)
        
        # Accuracy score (based on outliers and validation issues)
        accuracy_issues = [i for i in issues if i.issue_type in [DataQualityIssueType.OUTLIERS, DataQualityIssueType.VALIDITY]]
        accuracy_score = 1.0 - (sum(i.affected_records for i in accuracy_issues) / total_records if total_records > 0 else 0)
        
        # Consistency score (based on duplicates and schema violations)
        consistency_issues = [i for i in issues if i.issue_type in [DataQualityIssueType.DUPLICATE_RECORDS, DataQualityIssueType.SCHEMA_VIOLATION]]
        consistency_score = 1.0 - (sum(i.affected_records for i in consistency_issues) / total_records if total_records > 0 else 0)
        
        # Validity score (based on validation issues)
        validity_issues = [i for i in issues if i.issue_type == DataQualityIssueType.VALIDITY]
        validity_score = 1.0 - (sum(i.affected_records for i in validity_issues) / total_records if total_records > 0 else 0)
        
        # Freshness score (based on freshness issues)
        freshness_issues = [i for i in issues if i.issue_type == DataQualityIssueType.FRESHNESS]
        freshness_score = 1.0 if not freshness_issues else min(i.quality_score for i in freshness_issues)
        
        # Overall score (weighted average)
        overall_score = (
            completeness_score * 0.25 +
            accuracy_score * 0.25 +
            consistency_score * 0.2 +
            validity_score * 0.2 +
            freshness_score * 0.1
        )
        
        critical_issues_count = len([i for i in issues if i.severity == QualityCheckSeverity.CRITICAL])
        
        return DataQualityMetrics(
            source_name="default",  # Configure based on your setup
            table_name=table_name,
            total_records=total_records,
            completeness_score=completeness_score,
            accuracy_score=accuracy_score,
            consistency_score=consistency_score,
            validity_score=validity_score,
            freshness_score=freshness_score,
            overall_score=overall_score,
            issues_count=len(issues),
            critical_issues_count=critical_issues_count,
            measured_at=datetime.now()
        )
    
    async def _store_quality_metrics(self, metrics: DataQualityMetrics):
        """Store quality metrics in database"""
        if not self.Session:
            return
        
        session = self.Session()
        try:
            metrics_model = DataQualityMetricsModel(
                id=f"{metrics.table_name}_{int(metrics.measured_at.timestamp())}",
                source_name=metrics.source_name,
                table_name=metrics.table_name,
                total_records=metrics.total_records,
                completeness_score=metrics.completeness_score,
                accuracy_score=metrics.accuracy_score,
                consistency_score=metrics.consistency_score,
                validity_score=metrics.validity_score,
                freshness_score=metrics.freshness_score,
                overall_score=metrics.overall_score,
                issues_count=metrics.issues_count,
                critical_issues_count=metrics.critical_issues_count,
                measured_at=metrics.measured_at
            )
            
            session.add(metrics_model)
            session.commit()
            
        finally:
            session.close()
    
    async def _process_quality_issue(self, issue: DataQualityIssue):
        """Process and store quality issue"""
        try:
            # Store in memory
            self.active_issues[issue.id] = issue
            
            # Store in database
            if self.Session:
                session = self.Session()
                try:
                    issue_model = DataQualityIssueModel(
                        id=issue.id,
                        rule_name=issue.rule_name,
                        issue_type=issue.issue_type.value,
                        severity=issue.severity.value,
                        description=issue.description,
                        affected_records=issue.affected_records,
                        total_records=issue.total_records,
                        quality_score=issue.quality_score,
                        detected_at=issue.detected_at,
                        source_table=issue.source_table,
                        source_column=issue.source_column,
                        metadata=json.dumps(issue.metadata or {})
                    )
                    
                    session.add(issue_model)
                    session.commit()
                    
                finally:
                    session.close()
            
            # Trigger alerts
            await self._trigger_quality_alert(issue)
            
            logger.warning(f"Data quality issue detected: {issue.description}")
            
        except Exception as e:
            logger.error(f"Error processing quality issue: {e}")
    
    async def _trigger_quality_alert(self, issue: DataQualityIssue):
        """Trigger alert for quality issue"""
        try:
            # Call registered alert callbacks
            for callback in self.alert_callbacks:
                try:
                    await callback(issue)
                except Exception as e:
                    logger.error(f"Error in alert callback: {e}")
            
            # Publish to Redis for other systems
            await self.redis_client.xadd('quality_alerts', {
                'issue_id': issue.id,
                'issue_type': issue.issue_type.value,
                'severity': issue.severity.value,
                'table_name': issue.source_table,
                'description': issue.description,
                'data': json.dumps(asdict(issue))
            })
            
        except Exception as e:
            logger.error(f"Error triggering quality alert: {e}")
    
    async def _update_redis_metrics(self, table_name: str, metrics: DataQualityMetrics):
        """Update Redis with latest quality metrics"""
        try:
            metrics_key = f"quality_metrics:{table_name}"
            
            await self.redis_client.hset(metrics_key, mapping={
                'overall_score': str(metrics.overall_score),
                'completeness_score': str(metrics.completeness_score),
                'accuracy_score': str(metrics.accuracy_score),
                'consistency_score': str(metrics.consistency_score),
                'validity_score': str(metrics.validity_score),
                'freshness_score': str(metrics.freshness_score),
                'issues_count': str(metrics.issues_count),
                'critical_issues_count': str(metrics.critical_issues_count),
                'total_records': str(metrics.total_records),
                'measured_at': metrics.measured_at.isoformat()
            })
            
            # Set expiration
            await self.redis_client.expire(metrics_key, 86400)  # 24 hours
            
        except Exception as e:
            logger.error(f"Error updating Redis metrics: {e}")
    
    async def _train_anomaly_models(self):
        """Train anomaly detection models for quality metrics"""
        while self.running:
            try:
                # This would train ML models based on historical quality data
                # Implementation depends on your specific requirements
                await asyncio.sleep(3600)  # Train every hour
                
            except Exception as e:
                logger.error(f"Error training anomaly models: {e}")
                await asyncio.sleep(300)
    
    async def _cleanup_old_issues(self):
        """Clean up old quality issues"""
        while self.running:
            try:
                cutoff_time = datetime.now() - timedelta(days=30)
                
                # Clean up from memory
                issues_to_remove = []
                for issue_id, issue in self.active_issues.items():
                    if issue.detected_at < cutoff_time:
                        issues_to_remove.append(issue_id)
                
                for issue_id in issues_to_remove:
                    del self.active_issues[issue_id]
                
                # Clean up from database
                if self.Session:
                    session = self.Session()
                    try:
                        session.query(DataQualityIssueModel).filter(
                            DataQualityIssueModel.detected_at < cutoff_time
                        ).delete()
                        session.commit()
                    finally:
                        session.close()
                
                await asyncio.sleep(86400)  # Clean up daily
                
            except Exception as e:
                logger.error(f"Error cleaning up old issues: {e}")
                await asyncio.sleep(3600)
    
    # Public API methods
    
    def add_quality_rule(self, table_name: str, rule: DataQualityRule):
        """Add quality rule for a table"""
        if table_name not in self.quality_rules:
            self.quality_rules[table_name] = []
        
        self.quality_rules[table_name].append(rule)
        logger.info(f"Added quality rule {rule.name} for table {table_name}")
    
    def add_expectation_suite(self, table_name: str, suite: ExpectationSuite):
        """Add Great Expectations suite for a table"""
        self.expectation_suites[table_name] = suite
        logger.info(f"Added expectation suite for table {table_name}")
    
    def add_alert_callback(self, callback: Callable):
        """Add callback for quality alerts"""
        self.alert_callbacks.append(callback)
    
    async def get_quality_metrics(self, table_name: str) -> Optional[DataQualityMetrics]:
        """Get latest quality metrics for a table"""
        return self.quality_metrics.get(table_name)
    
    async def get_active_issues(self, table_name: Optional[str] = None) -> List[DataQualityIssue]:
        """Get active quality issues"""
        if table_name:
            return [issue for issue in self.active_issues.values() 
                   if issue.source_table == table_name]
        else:
            return list(self.active_issues.values())
    
    async def resolve_issue(self, issue_id: str):
        """Mark quality issue as resolved"""
        if issue_id in self.active_issues:
            del self.active_issues[issue_id]
            logger.info(f"Quality issue resolved: {issue_id}")

# Example usage
async def main():
    """Example usage of data quality monitor"""
    monitor = DataQualityMonitor()
    await monitor.initialize()
    
    # Add custom quality rules
    monitor.add_quality_rule("users", DataQualityRule(
        name="email_format_check",
        description="Check email format validity",
        rule_type=DataQualityIssueType.VALIDITY,
        severity=QualityCheckSeverity.HIGH,
        threshold=0.02,
        metadata={"email_column": "email"}
    ))
    
    # Add alert callback
    async def quality_alert_handler(issue: DataQualityIssue):
        print(f"Quality Alert: {issue.description}")
    
    monitor.add_alert_callback(quality_alert_handler)
    
    # Start monitoring
    await monitor.start_monitoring()

if __name__ == "__main__":
    asyncio.run(main())