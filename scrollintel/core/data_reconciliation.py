"""
Data Reconciliation System for Advanced Analytics Dashboard

Handles data inconsistencies across multiple sources with automated
conflict resolution and data lineage tracking.
"""

from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import pandas as pd
import numpy as np
import logging
import json
from collections import defaultdict

logger = logging.getLogger(__name__)


class ConflictResolutionStrategy(Enum):
    LATEST_TIMESTAMP = "latest_timestamp"
    HIGHEST_PRIORITY = "highest_priority"
    MAJORITY_VOTE = "majority_vote"
    AVERAGE_VALUE = "average_value"
    MANUAL_REVIEW = "manual_review"
    CUSTOM_RULE = "custom_rule"


class ReconciliationStatus(Enum):
    PENDING = "pending"
    RESOLVED = "resolved"
    FAILED = "failed"
    MANUAL_REQUIRED = "manual_required"


@dataclass
class DataSource:
    """Represents a data source with metadata"""
    id: str
    name: str
    priority: int
    reliability_score: float
    last_updated: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataConflict:
    """Represents a conflict between data sources"""
    id: str
    field_name: str
    record_key: str
    conflicting_values: Dict[str, Any]  # source_id -> value
    source_metadata: Dict[str, Dict[str, Any]]
    detected_at: datetime = field(default_factory=datetime.utcnow)
    resolution_strategy: Optional[ConflictResolutionStrategy] = None
    resolved_value: Any = None
    status: ReconciliationStatus = ReconciliationStatus.PENDING
    confidence_score: float = 0.0


@dataclass
class ReconciliationRule:
    """Defines how to reconcile conflicts for specific fields"""
    field_name: str
    strategy: ConflictResolutionStrategy
    parameters: Dict[str, Any] = field(default_factory=dict)
    priority: int = 1
    enabled: bool = True


@dataclass
class ReconciliationResult:
    """Result of data reconciliation process"""
    success: bool
    reconciled_data: pd.DataFrame
    conflicts_detected: int
    conflicts_resolved: int
    unresolved_conflicts: List[DataConflict]
    processing_time: float
    quality_metrics: Dict[str, Any] = field(default_factory=dict)
    lineage_info: Dict[str, Any] = field(default_factory=dict)


class DataReconciliationEngine:
    """
    Advanced data reconciliation system for handling multi-source conflicts
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.data_sources: Dict[str, DataSource] = {}
        self.reconciliation_rules: Dict[str, ReconciliationRule] = {}
        self.conflict_history: List[DataConflict] = []
        self.resolution_strategies = self._initialize_strategies()
        
    def register_data_source(self, source: DataSource) -> bool:
        """Register a data source for reconciliation"""
        try:
            self.data_sources[source.id] = source
            logger.info(f"Registered data source: {source.name}")
            return True
        except Exception as e:
            logger.error(f"Failed to register data source {source.name}: {str(e)}")
            return False
    
    def register_reconciliation_rule(self, rule: ReconciliationRule) -> bool:
        """Register a reconciliation rule for a field"""
        try:
            self.reconciliation_rules[rule.field_name] = rule
            logger.info(f"Registered reconciliation rule for field: {rule.field_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to register rule for {rule.field_name}: {str(e)}")
            return False
    
    def reconcile_data(self, datasets: Dict[str, pd.DataFrame], 
                      key_field: str) -> ReconciliationResult:
        """
        Reconcile data from multiple sources
        
        Args:
            datasets: Dictionary mapping source_id to DataFrame
            key_field: Field to use as record identifier
        """
        start_time = datetime.utcnow()
        
        try:
            # Validate inputs
            if not datasets or len(datasets) < 2:
                return ReconciliationResult(
                    success=False,
                    reconciled_data=pd.DataFrame(),
                    conflicts_detected=0,
                    conflicts_resolved=0,
                    unresolved_conflicts=[],
                    processing_time=0
                )
            
            # Detect conflicts
            conflicts = self._detect_conflicts(datasets, key_field)
            logger.info(f"Detected {len(conflicts)} conflicts")
            
            # Resolve conflicts
            resolved_conflicts = []
            unresolved_conflicts = []
            
            for conflict in conflicts:
                try:
                    resolved_conflict = self._resolve_conflict(conflict)
                    if resolved_conflict.status == ReconciliationStatus.RESOLVED:
                        resolved_conflicts.append(resolved_conflict)
                    else:
                        unresolved_conflicts.append(resolved_conflict)
                except Exception as e:
                    logger.error(f"Failed to resolve conflict {conflict.id}: {str(e)}")
                    conflict.status = ReconciliationStatus.FAILED
                    unresolved_conflicts.append(conflict)
            
            # Build reconciled dataset
            reconciled_data = self._build_reconciled_dataset(
                datasets, key_field, resolved_conflicts
            )
            
            # Calculate quality metrics
            quality_metrics = self._calculate_reconciliation_metrics(
                datasets, reconciled_data, conflicts, resolved_conflicts
            )
            
            # Generate lineage information
            lineage_info = self._generate_lineage_info(
                datasets, resolved_conflicts, key_field
            )
            
            # Calculate processing time
            end_time = datetime.utcnow()
            processing_time = (end_time - start_time).total_seconds()
            
            # Store conflicts in history
            self.conflict_history.extend(conflicts)
            
            return ReconciliationResult(
                success=True,
                reconciled_data=reconciled_data,
                conflicts_detected=len(conflicts),
                conflicts_resolved=len(resolved_conflicts),
                unresolved_conflicts=unresolved_conflicts,
                processing_time=processing_time,
                quality_metrics=quality_metrics,
                lineage_info=lineage_info
            )
            
        except Exception as e:
            logger.error(f"Data reconciliation failed: {str(e)}")
            end_time = datetime.utcnow()
            processing_time = (end_time - start_time).total_seconds()
            
            return ReconciliationResult(
                success=False,
                reconciled_data=pd.DataFrame(),
                conflicts_detected=0,
                conflicts_resolved=0,
                unresolved_conflicts=[],
                processing_time=processing_time
            )
    
    def _detect_conflicts(self, datasets: Dict[str, pd.DataFrame], 
                         key_field: str) -> List[DataConflict]:
        """Detect conflicts between datasets"""
        conflicts = []
        
        try:
            # Get all unique keys across datasets
            all_keys = set()
            for df in datasets.values():
                if key_field in df.columns:
                    all_keys.update(df[key_field].unique())
            
            # Check each key for conflicts
            for key in all_keys:
                key_conflicts = self._detect_key_conflicts(datasets, key_field, key)
                conflicts.extend(key_conflicts)
            
            return conflicts
            
        except Exception as e:
            logger.error(f"Conflict detection failed: {str(e)}")
            return []
    
    def _detect_key_conflicts(self, datasets: Dict[str, pd.DataFrame], 
                            key_field: str, key_value: Any) -> List[DataConflict]:
        """Detect conflicts for a specific key across datasets"""
        conflicts = []
        
        try:
            # Get records for this key from each dataset
            key_records = {}
            for source_id, df in datasets.items():
                if key_field in df.columns:
                    matching_records = df[df[key_field] == key_value]
                    if not matching_records.empty:
                        # Take the first record if multiple exist
                        key_records[source_id] = matching_records.iloc[0].to_dict()
            
            if len(key_records) < 2:
                return conflicts  # No conflict if only one source has this key
            
            # Compare fields across sources
            all_fields = set()
            for record in key_records.values():
                all_fields.update(record.keys())
            
            for field in all_fields:
                if field == key_field:
                    continue  # Skip the key field itself
                
                # Get values for this field from each source
                field_values = {}
                source_metadata = {}
                
                for source_id, record in key_records.items():
                    if field in record:
                        field_values[source_id] = record[field]
                        source_metadata[source_id] = {
                            "source_name": self.data_sources.get(source_id, DataSource(
                                id=source_id, name=source_id, priority=0, reliability_score=0.5,
                                last_updated=datetime.utcnow()
                            )).name,
                            "timestamp": datetime.utcnow().isoformat(),
                            "reliability": self.data_sources.get(source_id, DataSource(
                                id=source_id, name=source_id, priority=0, reliability_score=0.5,
                                last_updated=datetime.utcnow()
                            )).reliability_score
                        }
                
                # Check if values conflict
                if len(set(str(v) for v in field_values.values() if pd.notna(v))) > 1:
                    conflict_id = f"{key_value}_{field}_{datetime.utcnow().timestamp()}"
                    
                    conflict = DataConflict(
                        id=conflict_id,
                        field_name=field,
                        record_key=str(key_value),
                        conflicting_values=field_values,
                        source_metadata=source_metadata
                    )
                    
                    conflicts.append(conflict)
            
            return conflicts
            
        except Exception as e:
            logger.error(f"Key conflict detection failed for {key_value}: {str(e)}")
            return []
    
    def _resolve_conflict(self, conflict: DataConflict) -> DataConflict:
        """Resolve a single data conflict"""
        try:
            # Get reconciliation rule for this field
            rule = self.reconciliation_rules.get(conflict.field_name)
            
            if not rule or not rule.enabled:
                # Use default strategy
                strategy = ConflictResolutionStrategy.HIGHEST_PRIORITY
            else:
                strategy = rule.strategy
            
            # Apply resolution strategy
            resolver = self.resolution_strategies.get(strategy)
            if not resolver:
                raise ValueError(f"Unknown resolution strategy: {strategy}")
            
            resolved_value, confidence = resolver(conflict, rule)
            
            # Update conflict with resolution
            conflict.resolution_strategy = strategy
            conflict.resolved_value = resolved_value
            conflict.confidence_score = confidence
            conflict.status = ReconciliationStatus.RESOLVED
            
            return conflict
            
        except Exception as e:
            logger.error(f"Conflict resolution failed for {conflict.id}: {str(e)}")
            conflict.status = ReconciliationStatus.FAILED
            return conflict
    
    def _initialize_strategies(self) -> Dict[ConflictResolutionStrategy, callable]:
        """Initialize conflict resolution strategies"""
        return {
            ConflictResolutionStrategy.LATEST_TIMESTAMP: self._resolve_by_timestamp,
            ConflictResolutionStrategy.HIGHEST_PRIORITY: self._resolve_by_priority,
            ConflictResolutionStrategy.MAJORITY_VOTE: self._resolve_by_majority,
            ConflictResolutionStrategy.AVERAGE_VALUE: self._resolve_by_average,
            ConflictResolutionStrategy.MANUAL_REVIEW: self._resolve_manual,
            ConflictResolutionStrategy.CUSTOM_RULE: self._resolve_by_custom_rule
        }
    
    def _resolve_by_timestamp(self, conflict: DataConflict, 
                            rule: Optional[ReconciliationRule]) -> Tuple[Any, float]:
        """Resolve conflict by selecting value from most recent source"""
        try:
            latest_source = None
            latest_time = None
            
            for source_id in conflict.conflicting_values.keys():
                source = self.data_sources.get(source_id)
                if source and (latest_time is None or source.last_updated > latest_time):
                    latest_time = source.last_updated
                    latest_source = source_id
            
            if latest_source:
                return conflict.conflicting_values[latest_source], 0.8
            else:
                # Fallback to first available value
                return list(conflict.conflicting_values.values())[0], 0.3
                
        except Exception as e:
            logger.error(f"Timestamp resolution failed: {str(e)}")
            return list(conflict.conflicting_values.values())[0], 0.1
    
    def _resolve_by_priority(self, conflict: DataConflict, 
                           rule: Optional[ReconciliationRule]) -> Tuple[Any, float]:
        """Resolve conflict by selecting value from highest priority source"""
        try:
            highest_priority_source = None
            highest_priority = -1
            
            for source_id in conflict.conflicting_values.keys():
                source = self.data_sources.get(source_id)
                if source and source.priority > highest_priority:
                    highest_priority = source.priority
                    highest_priority_source = source_id
            
            if highest_priority_source:
                reliability = self.data_sources[highest_priority_source].reliability_score
                return conflict.conflicting_values[highest_priority_source], reliability
            else:
                return list(conflict.conflicting_values.values())[0], 0.3
                
        except Exception as e:
            logger.error(f"Priority resolution failed: {str(e)}")
            return list(conflict.conflicting_values.values())[0], 0.1
    
    def _resolve_by_majority(self, conflict: DataConflict, 
                           rule: Optional[ReconciliationRule]) -> Tuple[Any, float]:
        """Resolve conflict by majority vote"""
        try:
            value_counts = defaultdict(int)
            
            for value in conflict.conflicting_values.values():
                if pd.notna(value):
                    value_counts[str(value)] += 1
            
            if value_counts:
                majority_value_str = max(value_counts, key=value_counts.get)
                majority_count = value_counts[majority_value_str]
                total_count = len(conflict.conflicting_values)
                
                # Find original value (not string representation)
                majority_value = None
                for value in conflict.conflicting_values.values():
                    if str(value) == majority_value_str:
                        majority_value = value
                        break
                
                confidence = majority_count / total_count
                return majority_value, confidence
            else:
                return list(conflict.conflicting_values.values())[0], 0.1
                
        except Exception as e:
            logger.error(f"Majority resolution failed: {str(e)}")
            return list(conflict.conflicting_values.values())[0], 0.1
    
    def _resolve_by_average(self, conflict: DataConflict, 
                          rule: Optional[ReconciliationRule]) -> Tuple[Any, float]:
        """Resolve conflict by averaging numeric values"""
        try:
            numeric_values = []
            
            for value in conflict.conflicting_values.values():
                if pd.notna(value):
                    try:
                        numeric_values.append(float(value))
                    except (ValueError, TypeError):
                        continue
            
            if numeric_values:
                average_value = np.mean(numeric_values)
                confidence = 0.7 if len(numeric_values) > 1 else 0.3
                return average_value, confidence
            else:
                # Fallback to first non-null value
                for value in conflict.conflicting_values.values():
                    if pd.notna(value):
                        return value, 0.2
                return None, 0.0
                
        except Exception as e:
            logger.error(f"Average resolution failed: {str(e)}")
            return list(conflict.conflicting_values.values())[0], 0.1
    
    def _resolve_manual(self, conflict: DataConflict, 
                       rule: Optional[ReconciliationRule]) -> Tuple[Any, float]:
        """Mark conflict for manual review"""
        conflict.status = ReconciliationStatus.MANUAL_REQUIRED
        return None, 0.0
    
    def _resolve_by_custom_rule(self, conflict: DataConflict, 
                              rule: Optional[ReconciliationRule]) -> Tuple[Any, float]:
        """Resolve conflict using custom rule logic"""
        try:
            if not rule or "custom_logic" not in rule.parameters:
                return self._resolve_by_priority(conflict, rule)
            
            custom_logic = rule.parameters["custom_logic"]
            
            # Simple custom rule implementation
            # In production, this would be more sophisticated
            if custom_logic == "prefer_non_null":
                for source_id, value in conflict.conflicting_values.items():
                    if pd.notna(value) and value != "":
                        source = self.data_sources.get(source_id)
                        reliability = source.reliability_score if source else 0.5
                        return value, reliability
            
            elif custom_logic == "prefer_longer_string":
                longest_value = None
                max_length = 0
                
                for value in conflict.conflicting_values.values():
                    if pd.notna(value):
                        length = len(str(value))
                        if length > max_length:
                            max_length = length
                            longest_value = value
                
                return longest_value, 0.6 if longest_value else 0.1
            
            # Fallback to priority resolution
            return self._resolve_by_priority(conflict, rule)
            
        except Exception as e:
            logger.error(f"Custom rule resolution failed: {str(e)}")
            return self._resolve_by_priority(conflict, rule)
    
    def _build_reconciled_dataset(self, datasets: Dict[str, pd.DataFrame], 
                                key_field: str, 
                                resolved_conflicts: List[DataConflict]) -> pd.DataFrame:
        """Build the final reconciled dataset"""
        try:
            # Start with the dataset from the highest priority source
            primary_source_id = None
            highest_priority = -1
            
            for source_id in datasets.keys():
                source = self.data_sources.get(source_id)
                if source and source.priority > highest_priority:
                    highest_priority = source.priority
                    primary_source_id = source_id
            
            if not primary_source_id:
                primary_source_id = list(datasets.keys())[0]
            
            reconciled_df = datasets[primary_source_id].copy()
            
            # Apply conflict resolutions
            conflict_map = {}
            for conflict in resolved_conflicts:
                key = (conflict.record_key, conflict.field_name)
                conflict_map[key] = conflict.resolved_value
            
            # Update values based on conflict resolutions
            for index, row in reconciled_df.iterrows():
                key_value = row[key_field]
                
                for field in reconciled_df.columns:
                    conflict_key = (str(key_value), field)
                    if conflict_key in conflict_map:
                        reconciled_df.at[index, field] = conflict_map[conflict_key]
            
            # Add records from other sources that don't exist in primary source
            all_keys = set(reconciled_df[key_field].unique())
            
            for source_id, df in datasets.items():
                if source_id == primary_source_id:
                    continue
                
                if key_field not in df.columns:
                    continue
                
                # Find records not in primary source
                missing_keys = set(df[key_field].unique()) - all_keys
                
                for missing_key in missing_keys:
                    missing_records = df[df[key_field] == missing_key]
                    if not missing_records.empty:
                        # Add the first record (in case of duplicates)
                        new_record = missing_records.iloc[0].to_dict()
                        
                        # Apply any relevant conflict resolutions
                        for field, value in new_record.items():
                            conflict_key = (str(missing_key), field)
                            if conflict_key in conflict_map:
                                new_record[field] = conflict_map[conflict_key]
                        
                        # Add to reconciled dataset
                        reconciled_df = pd.concat([
                            reconciled_df, 
                            pd.DataFrame([new_record])
                        ], ignore_index=True)
                        
                        all_keys.add(missing_key)
            
            return reconciled_df
            
        except Exception as e:
            logger.error(f"Failed to build reconciled dataset: {str(e)}")
            # Return the first available dataset as fallback
            return list(datasets.values())[0].copy() if datasets else pd.DataFrame()
    
    def _calculate_reconciliation_metrics(self, datasets: Dict[str, pd.DataFrame],
                                        reconciled_data: pd.DataFrame,
                                        all_conflicts: List[DataConflict],
                                        resolved_conflicts: List[DataConflict]) -> Dict[str, Any]:
        """Calculate metrics for the reconciliation process"""
        try:
            total_records = sum(len(df) for df in datasets.values())
            reconciled_records = len(reconciled_data)
            
            metrics = {
                "source_count": len(datasets),
                "total_source_records": total_records,
                "reconciled_records": reconciled_records,
                "conflicts_detected": len(all_conflicts),
                "conflicts_resolved": len(resolved_conflicts),
                "resolution_rate": len(resolved_conflicts) / len(all_conflicts) if all_conflicts else 1.0,
                "data_completeness": reconciled_records / max(len(df) for df in datasets.values()) if datasets else 0,
                "average_confidence": np.mean([c.confidence_score for c in resolved_conflicts]) if resolved_conflicts else 0,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Strategy usage statistics
            strategy_usage = defaultdict(int)
            for conflict in resolved_conflicts:
                if conflict.resolution_strategy:
                    strategy_usage[conflict.resolution_strategy.value] += 1
            
            metrics["strategy_usage"] = dict(strategy_usage)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to calculate reconciliation metrics: {str(e)}")
            return {"error": str(e)}
    
    def _generate_lineage_info(self, datasets: Dict[str, pd.DataFrame],
                             resolved_conflicts: List[DataConflict],
                             key_field: str) -> Dict[str, Any]:
        """Generate data lineage information"""
        try:
            lineage = {
                "sources": {},
                "transformations": [],
                "conflicts": {},
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Source information
            for source_id, df in datasets.items():
                source = self.data_sources.get(source_id)
                lineage["sources"][source_id] = {
                    "name": source.name if source else source_id,
                    "record_count": len(df),
                    "fields": list(df.columns),
                    "priority": source.priority if source else 0,
                    "reliability": source.reliability_score if source else 0.5,
                    "last_updated": source.last_updated.isoformat() if source else datetime.utcnow().isoformat()
                }
            
            # Conflict resolution information
            for conflict in resolved_conflicts:
                conflict_info = {
                    "field": conflict.field_name,
                    "record_key": conflict.record_key,
                    "strategy": conflict.resolution_strategy.value if conflict.resolution_strategy else "unknown",
                    "confidence": conflict.confidence_score,
                    "sources_involved": list(conflict.conflicting_values.keys()),
                    "resolved_value": str(conflict.resolved_value) if conflict.resolved_value is not None else None
                }
                
                lineage["conflicts"][conflict.id] = conflict_info
            
            return lineage
            
        except Exception as e:
            logger.error(f"Failed to generate lineage info: {str(e)}")
            return {"error": str(e)}
    
    def get_conflict_summary(self, days: int = 30) -> Dict[str, Any]:
        """Get summary of conflicts over specified time period"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            recent_conflicts = [
                conflict for conflict in self.conflict_history
                if conflict.detected_at >= cutoff_date
            ]
            
            if not recent_conflicts:
                return {"message": "No conflicts found in specified period"}
            
            # Analyze conflicts
            field_conflicts = defaultdict(int)
            strategy_usage = defaultdict(int)
            resolution_rates = defaultdict(lambda: {"resolved": 0, "total": 0})
            
            for conflict in recent_conflicts:
                field_conflicts[conflict.field_name] += 1
                
                if conflict.resolution_strategy:
                    strategy_usage[conflict.resolution_strategy.value] += 1
                
                status_key = "resolved" if conflict.status == ReconciliationStatus.RESOLVED else "total"
                resolution_rates[conflict.field_name][status_key] += 1
                resolution_rates[conflict.field_name]["total"] += 1
            
            # Calculate resolution rates by field
            field_resolution_rates = {}
            for field, rates in resolution_rates.items():
                field_resolution_rates[field] = rates["resolved"] / rates["total"] if rates["total"] > 0 else 0
            
            return {
                "period_days": days,
                "total_conflicts": len(recent_conflicts),
                "conflicts_by_field": dict(field_conflicts),
                "strategy_usage": dict(strategy_usage),
                "resolution_rates_by_field": field_resolution_rates,
                "overall_resolution_rate": sum(1 for c in recent_conflicts if c.status == ReconciliationStatus.RESOLVED) / len(recent_conflicts)
            }
            
        except Exception as e:
            logger.error(f"Failed to get conflict summary: {str(e)}")
            return {"error": str(e)}