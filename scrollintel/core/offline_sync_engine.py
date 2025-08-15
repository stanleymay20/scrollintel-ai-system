"""
Intelligent sync conflict resolution engine for ScrollIntel offline capabilities.
Provides advanced conflict detection, resolution strategies, and sync orchestration.
"""

import asyncio
import logging
import json
import time
import uuid
from typing import Dict, List, Optional, Any, Union, Callable, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict, deque
import hashlib
import difflib
import copy

from .offline_data_manager import (
    OfflineDataEntry, SyncConflict, ConflictResolutionStrategy, 
    SyncStatus, DataOperation
)

logger = logging.getLogger(__name__)


class ConflictType(Enum):
    """Types of synchronization conflicts."""
    DATA_MISMATCH = "data_mismatch"
    VERSION_CONFLICT = "version_conflict"
    CONCURRENT_EDIT = "concurrent_edit"
    DELETE_MODIFY = "delete_modify"
    SCHEMA_CHANGE = "schema_change"
    PERMISSION_CONFLICT = "permission_conflict"
    DEPENDENCY_CONFLICT = "dependency_conflict"


class SyncPriority(Enum):
    """Priority levels for sync operations."""
    CRITICAL = "critical"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"
    BACKGROUND = "background"


@dataclass
class ConflictAnalysis:
    """Analysis of a synchronization conflict."""
    conflict_type: ConflictType
    severity: float  # 0-1 scale
    affected_fields: List[str]
    resolution_confidence: float  # 0-1 scale
    recommended_strategy: ConflictResolutionStrategy
    merge_complexity: float  # 0-1 scale
    user_impact: float  # 0-1 scale
    auto_resolvable: bool
    analysis_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SyncBatch:
    """Batch of sync operations for efficient processing."""
    id: str
    operations: List[Dict[str, Any]]
    priority: SyncPriority
    created_at: datetime
    estimated_duration: float
    dependencies: List[str] = field(default_factory=list)
    status: str = "pending"
    progress: float = 0.0


@dataclass
class SyncMetrics:
    """Metrics for sync performance monitoring."""
    total_syncs: int = 0
    successful_syncs: int = 0
    failed_syncs: int = 0
    conflicts_detected: int = 0
    conflicts_resolved: int = 0
    auto_resolved_conflicts: int = 0
    manual_resolved_conflicts: int = 0
    avg_sync_time: float = 0.0
    avg_conflict_resolution_time: float = 0.0
    data_transferred_mb: float = 0.0
    last_sync_timestamp: Optional[datetime] = None


class IntelligentSyncEngine:
    """Advanced synchronization engine with intelligent conflict resolution."""
    
    def __init__(self, offline_manager):
        self.offline_manager = offline_manager
        
        # Conflict analysis and resolution
        self.conflict_analyzers: Dict[ConflictType, Callable] = {}
        self.resolution_strategies: Dict[ConflictResolutionStrategy, Callable] = {}
        self.field_merge_handlers: Dict[str, Callable] = {}
        
        # Sync orchestration
        self.sync_batches: Dict[str, SyncBatch] = {}
        self.sync_queue: deque = deque()
        self.active_syncs: Dict[str, asyncio.Task] = {}
        self.max_concurrent_syncs = 5
        
        # Performance and learning
        self.metrics = SyncMetrics()
        self.conflict_patterns: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.resolution_success_rates: Dict[str, float] = {}
        self.user_preferences: Dict[str, Dict[str, Any]] = {}
        
        # Dependency tracking
        self.dependency_graph: Dict[str, Set[str]] = defaultdict(set)
        self.reverse_dependencies: Dict[str, Set[str]] = defaultdict(set)
        
        # Initialize components
        self._setup_conflict_analyzers()
        self._setup_resolution_strategies()
        self._setup_field_merge_handlers()
        self._load_learning_data()
    
    def _setup_conflict_analyzers(self):
        """Setup conflict analysis functions."""
        self.conflict_analyzers = {
            ConflictType.DATA_MISMATCH: self._analyze_data_mismatch,
            ConflictType.VERSION_CONFLICT: self._analyze_version_conflict,
            ConflictType.CONCURRENT_EDIT: self._analyze_concurrent_edit,
            ConflictType.DELETE_MODIFY: self._analyze_delete_modify,
            ConflictType.SCHEMA_CHANGE: self._analyze_schema_change,
            ConflictType.PERMISSION_CONFLICT: self._analyze_permission_conflict,
            ConflictType.DEPENDENCY_CONFLICT: self._analyze_dependency_conflict
        }
    
    def _setup_resolution_strategies(self):
        """Setup advanced conflict resolution strategies."""
        self.resolution_strategies = {
            ConflictResolutionStrategy.INTELLIGENT_MERGE: self._intelligent_merge_resolution,
            ConflictResolutionStrategy.TIMESTAMP_BASED: self._timestamp_based_resolution,
            ConflictResolutionStrategy.CLIENT_WINS: self._client_wins_resolution,
            ConflictResolutionStrategy.SERVER_WINS: self._server_wins_resolution,
            ConflictResolutionStrategy.MERGE: self._field_level_merge_resolution,
            ConflictResolutionStrategy.USER_CHOICE: self._user_choice_resolution
        }
    
    def _setup_field_merge_handlers(self):
        """Setup field-specific merge handlers."""
        self.field_merge_handlers = {
            'list': self._merge_list_field,
            'dict': self._merge_dict_field,
            'string': self._merge_string_field,
            'number': self._merge_number_field,
            'boolean': self._merge_boolean_field,
            'timestamp': self._merge_timestamp_field,
            'array': self._merge_array_field,
            'object': self._merge_object_field
        }
    
    def _load_learning_data(self):
        """Load learning data from previous conflict resolutions."""
        try:
            # This would load from persistent storage
            # For now, initialize with defaults
            self.resolution_success_rates = {
                ConflictResolutionStrategy.INTELLIGENT_MERGE.value: 0.85,
                ConflictResolutionStrategy.TIMESTAMP_BASED.value: 0.75,
                ConflictResolutionStrategy.MERGE.value: 0.70,
                ConflictResolutionStrategy.CLIENT_WINS.value: 0.60,
                ConflictResolutionStrategy.SERVER_WINS.value: 0.60
            }
        except Exception as e:
            logger.warning(f"Could not load learning data: {e}")
    
    async def analyze_conflict(self, conflict: SyncConflict) -> ConflictAnalysis:
        """Perform comprehensive conflict analysis."""
        # Determine conflict type
        conflict_type = await self._determine_conflict_type(conflict)
        
        # Get analyzer for this conflict type
        analyzer = self.conflict_analyzers.get(conflict_type, self._analyze_generic_conflict)
        
        # Perform detailed analysis
        analysis = await analyzer(conflict)
        
        # Enhance analysis with learning data
        await self._enhance_analysis_with_learning(analysis, conflict)
        
        # Record conflict pattern for learning
        self._record_conflict_pattern(conflict, analysis)
        
        return analysis
    
    async def _determine_conflict_type(self, conflict: SyncConflict) -> ConflictType:
        """Determine the type of conflict based on data analysis."""
        local_data = conflict.local_entry.data
        remote_data = conflict.remote_entry.get('data', {})
        
        # Check for deletion conflicts
        if local_data.get('_deleted') or remote_data.get('_deleted'):
            return ConflictType.DELETE_MODIFY
        
        # Check for version conflicts
        local_version = conflict.local_entry.version
        remote_version = conflict.remote_entry.get('version', 0)
        if abs(local_version - remote_version) > 1:
            return ConflictType.VERSION_CONFLICT
        
        # Check for schema changes
        local_keys = set(local_data.keys())
        remote_keys = set(remote_data.keys())
        if local_keys != remote_keys:
            key_diff_ratio = len(local_keys.symmetric_difference(remote_keys)) / max(len(local_keys), len(remote_keys), 1)
            if key_diff_ratio > 0.3:  # More than 30% key difference
                return ConflictType.SCHEMA_CHANGE
        
        # Check for concurrent edits (timestamps close together)
        local_time = conflict.local_entry.timestamp
        remote_time_str = conflict.remote_entry.get('timestamp')
        if remote_time_str:
            remote_time = datetime.fromisoformat(remote_time_str.replace('Z', '+00:00'))
            time_diff = abs((local_time - remote_time).total_seconds())
            if time_diff < 300:  # Within 5 minutes
                return ConflictType.CONCURRENT_EDIT
        
        # Check for permission conflicts
        local_user = conflict.local_entry.user_id
        remote_user = conflict.remote_entry.get('user_id')
        if local_user and remote_user and local_user != remote_user:
            return ConflictType.PERMISSION_CONFLICT
        
        # Default to data mismatch
        return ConflictType.DATA_MISMATCH
    
    async def _analyze_data_mismatch(self, conflict: SyncConflict) -> ConflictAnalysis:
        """Analyze data mismatch conflicts."""
        local_data = conflict.local_entry.data
        remote_data = conflict.remote_entry.get('data', {})
        
        # Find affected fields
        affected_fields = []
        for key in set(local_data.keys()) | set(remote_data.keys()):
            if local_data.get(key) != remote_data.get(key):
                affected_fields.append(key)
        
        # Calculate severity based on number of affected fields
        total_fields = len(set(local_data.keys()) | set(remote_data.keys()))
        severity = len(affected_fields) / max(total_fields, 1)
        
        # Determine merge complexity
        merge_complexity = self._calculate_merge_complexity(local_data, remote_data)
        
        # Assess user impact
        user_impact = self._assess_user_impact(affected_fields, local_data, remote_data)
        
        # Determine if auto-resolvable
        auto_resolvable = merge_complexity < 0.5 and user_impact < 0.3
        
        # Recommend strategy
        if auto_resolvable:
            recommended_strategy = ConflictResolutionStrategy.INTELLIGENT_MERGE
        elif merge_complexity < 0.7:
            recommended_strategy = ConflictResolutionStrategy.MERGE
        else:
            recommended_strategy = ConflictResolutionStrategy.USER_CHOICE
        
        return ConflictAnalysis(
            conflict_type=ConflictType.DATA_MISMATCH,
            severity=severity,
            affected_fields=affected_fields,
            resolution_confidence=0.8 if auto_resolvable else 0.6,
            recommended_strategy=recommended_strategy,
            merge_complexity=merge_complexity,
            user_impact=user_impact,
            auto_resolvable=auto_resolvable,
            analysis_metadata={
                'total_fields': total_fields,
                'changed_fields': len(affected_fields),
                'field_types': {k: type(v).__name__ for k, v in local_data.items()}
            }
        )
    
    async def _analyze_version_conflict(self, conflict: SyncConflict) -> ConflictAnalysis:
        """Analyze version conflicts."""
        local_version = conflict.local_entry.version
        remote_version = conflict.remote_entry.get('version', 0)
        version_gap = abs(local_version - remote_version)
        
        # Higher version gaps are more severe
        severity = min(version_gap / 10.0, 1.0)
        
        # Version conflicts are complex to merge
        merge_complexity = 0.8
        user_impact = 0.7  # Usually high impact
        
        # Not auto-resolvable due to complexity
        auto_resolvable = False
        
        return ConflictAnalysis(
            conflict_type=ConflictType.VERSION_CONFLICT,
            severity=severity,
            affected_fields=['version'],
            resolution_confidence=0.4,
            recommended_strategy=ConflictResolutionStrategy.USER_CHOICE,
            merge_complexity=merge_complexity,
            user_impact=user_impact,
            auto_resolvable=auto_resolvable,
            analysis_metadata={
                'local_version': local_version,
                'remote_version': remote_version,
                'version_gap': version_gap
            }
        )
    
    async def _analyze_concurrent_edit(self, conflict: SyncConflict) -> ConflictAnalysis:
        """Analyze concurrent edit conflicts."""
        local_time = conflict.local_entry.timestamp
        remote_time_str = conflict.remote_entry.get('timestamp')
        
        time_diff = 0
        if remote_time_str:
            remote_time = datetime.fromisoformat(remote_time_str.replace('Z', '+00:00'))
            time_diff = abs((local_time - remote_time).total_seconds())
        
        # Closer timestamps indicate higher concurrency
        severity = max(0.1, 1.0 - (time_diff / 3600))  # Normalize to 1 hour
        
        # Find affected fields
        local_data = conflict.local_entry.data
        remote_data = conflict.remote_entry.get('data', {})
        affected_fields = [
            key for key in set(local_data.keys()) | set(remote_data.keys())
            if local_data.get(key) != remote_data.get(key)
        ]
        
        merge_complexity = self._calculate_merge_complexity(local_data, remote_data)
        user_impact = 0.6  # Moderate impact
        
        # Can be auto-resolved if merge complexity is low
        auto_resolvable = merge_complexity < 0.4
        
        return ConflictAnalysis(
            conflict_type=ConflictType.CONCURRENT_EDIT,
            severity=severity,
            affected_fields=affected_fields,
            resolution_confidence=0.7 if auto_resolvable else 0.5,
            recommended_strategy=ConflictResolutionStrategy.INTELLIGENT_MERGE if auto_resolvable else ConflictResolutionStrategy.TIMESTAMP_BASED,
            merge_complexity=merge_complexity,
            user_impact=user_impact,
            auto_resolvable=auto_resolvable,
            analysis_metadata={
                'time_diff_seconds': time_diff,
                'local_timestamp': local_time.isoformat(),
                'remote_timestamp': remote_time_str
            }
        )
    
    async def _analyze_delete_modify(self, conflict: SyncConflict) -> ConflictAnalysis:
        """Analyze delete-modify conflicts."""
        local_data = conflict.local_entry.data
        remote_data = conflict.remote_entry.get('data', {})
        
        local_deleted = local_data.get('_deleted', False)
        remote_deleted = remote_data.get('_deleted', False)
        
        # High severity for delete conflicts
        severity = 0.9
        user_impact = 0.9  # High user impact
        merge_complexity = 0.9  # Complex to resolve
        
        # Not auto-resolvable
        auto_resolvable = False
        
        affected_fields = ['_deleted']
        if not local_deleted:
            affected_fields.extend(local_data.keys())
        if not remote_deleted:
            affected_fields.extend(remote_data.keys())
        
        return ConflictAnalysis(
            conflict_type=ConflictType.DELETE_MODIFY,
            severity=severity,
            affected_fields=list(set(affected_fields)),
            resolution_confidence=0.3,
            recommended_strategy=ConflictResolutionStrategy.USER_CHOICE,
            merge_complexity=merge_complexity,
            user_impact=user_impact,
            auto_resolvable=auto_resolvable,
            analysis_metadata={
                'local_deleted': local_deleted,
                'remote_deleted': remote_deleted
            }
        )
    
    async def _analyze_schema_change(self, conflict: SyncConflict) -> ConflictAnalysis:
        """Analyze schema change conflicts."""
        local_data = conflict.local_entry.data
        remote_data = conflict.remote_entry.get('data', {})
        
        local_keys = set(local_data.keys())
        remote_keys = set(remote_data.keys())
        
        added_fields = remote_keys - local_keys
        removed_fields = local_keys - remote_keys
        common_fields = local_keys & remote_keys
        
        # Calculate severity based on schema changes
        total_fields = len(local_keys | remote_keys)
        changed_fields = len(added_fields) + len(removed_fields)
        severity = changed_fields / max(total_fields, 1)
        
        merge_complexity = 0.6  # Moderate complexity
        user_impact = 0.5  # Moderate impact
        
        # Can be auto-resolved if changes are additive
        auto_resolvable = len(removed_fields) == 0 and len(added_fields) > 0
        
        return ConflictAnalysis(
            conflict_type=ConflictType.SCHEMA_CHANGE,
            severity=severity,
            affected_fields=list(added_fields | removed_fields),
            resolution_confidence=0.7 if auto_resolvable else 0.4,
            recommended_strategy=ConflictResolutionStrategy.INTELLIGENT_MERGE if auto_resolvable else ConflictResolutionStrategy.USER_CHOICE,
            merge_complexity=merge_complexity,
            user_impact=user_impact,
            auto_resolvable=auto_resolvable,
            analysis_metadata={
                'added_fields': list(added_fields),
                'removed_fields': list(removed_fields),
                'common_fields': list(common_fields)
            }
        )
    
    async def _analyze_permission_conflict(self, conflict: SyncConflict) -> ConflictAnalysis:
        """Analyze permission conflicts."""
        local_user = conflict.local_entry.user_id
        remote_user = conflict.remote_entry.get('user_id')
        
        # High severity for permission conflicts
        severity = 0.8
        user_impact = 0.8
        merge_complexity = 0.7
        
        # Not auto-resolvable
        auto_resolvable = False
        
        return ConflictAnalysis(
            conflict_type=ConflictType.PERMISSION_CONFLICT,
            severity=severity,
            affected_fields=['user_id'],
            resolution_confidence=0.3,
            recommended_strategy=ConflictResolutionStrategy.USER_CHOICE,
            merge_complexity=merge_complexity,
            user_impact=user_impact,
            auto_resolvable=auto_resolvable,
            analysis_metadata={
                'local_user': local_user,
                'remote_user': remote_user
            }
        )
    
    async def _analyze_dependency_conflict(self, conflict: SyncConflict) -> ConflictAnalysis:
        """Analyze dependency conflicts."""
        # This would analyze conflicts based on data dependencies
        # For now, provide a basic implementation
        
        severity = 0.7
        user_impact = 0.6
        merge_complexity = 0.8
        auto_resolvable = False
        
        return ConflictAnalysis(
            conflict_type=ConflictType.DEPENDENCY_CONFLICT,
            severity=severity,
            affected_fields=[],
            resolution_confidence=0.4,
            recommended_strategy=ConflictResolutionStrategy.USER_CHOICE,
            merge_complexity=merge_complexity,
            user_impact=user_impact,
            auto_resolvable=auto_resolvable
        )
    
    async def _analyze_generic_conflict(self, conflict: SyncConflict) -> ConflictAnalysis:
        """Generic conflict analysis fallback."""
        return ConflictAnalysis(
            conflict_type=ConflictType.DATA_MISMATCH,
            severity=0.5,
            affected_fields=[],
            resolution_confidence=0.5,
            recommended_strategy=ConflictResolutionStrategy.USER_CHOICE,
            merge_complexity=0.5,
            user_impact=0.5,
            auto_resolvable=False
        )
    
    def _calculate_merge_complexity(self, local_data: Dict[str, Any], 
                                  remote_data: Dict[str, Any]) -> float:
        """Calculate complexity of merging two data objects."""
        complexity_score = 0.0
        total_fields = len(set(local_data.keys()) | set(remote_data.keys()))
        
        if total_fields == 0:
            return 0.0
        
        for key in set(local_data.keys()) | set(remote_data.keys()):
            local_val = local_data.get(key)
            remote_val = remote_data.get(key)
            
            if local_val is None or remote_val is None:
                complexity_score += 0.2  # Missing field
            elif type(local_val) != type(remote_val):
                complexity_score += 0.8  # Type mismatch
            elif isinstance(local_val, dict):
                complexity_score += 0.6  # Nested object
            elif isinstance(local_val, list):
                complexity_score += 0.5  # Array
            elif local_val != remote_val:
                complexity_score += 0.3  # Simple value difference
        
        return min(complexity_score / total_fields, 1.0)
    
    def _assess_user_impact(self, affected_fields: List[str], 
                          local_data: Dict[str, Any], 
                          remote_data: Dict[str, Any]) -> float:
        """Assess the impact of conflicts on user experience."""
        if not affected_fields:
            return 0.0
        
        # Define field importance weights
        field_importance = {
            'id': 0.9,
            'name': 0.8,
            'title': 0.8,
            'email': 0.7,
            'status': 0.7,
            'created_at': 0.3,
            'updated_at': 0.2,
            '_metadata': 0.1
        }
        
        total_impact = 0.0
        for field in affected_fields:
            # Get field importance (default to 0.5 for unknown fields)
            importance = field_importance.get(field, 0.5)
            
            # Increase impact for complex data types
            local_val = local_data.get(field)
            if isinstance(local_val, (dict, list)):
                importance *= 1.2
            
            total_impact += importance
        
        # Normalize by number of fields
        return min(total_impact / len(affected_fields), 1.0)
    
    async def _enhance_analysis_with_learning(self, analysis: ConflictAnalysis, 
                                            conflict: SyncConflict):
        """Enhance analysis with learning from previous conflicts."""
        # Get historical success rates for this strategy
        strategy_key = analysis.recommended_strategy.value
        success_rate = self.resolution_success_rates.get(strategy_key, 0.5)
        
        # Adjust confidence based on historical success
        analysis.resolution_confidence *= success_rate
        
        # Check for similar conflict patterns
        similar_patterns = self._find_similar_conflict_patterns(conflict, analysis)
        if similar_patterns:
            # Adjust recommendations based on similar patterns
            successful_strategies = [
                p['resolution_strategy'] for p in similar_patterns 
                if p.get('resolution_successful', False)
            ]
            
            if successful_strategies:
                # Find most successful strategy for similar conflicts
                strategy_counts = {}
                for strategy in successful_strategies:
                    strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
                
                best_strategy = max(strategy_counts.items(), key=lambda x: x[1])[0]
                if strategy_counts[best_strategy] >= 2:  # At least 2 successes
                    try:
                        analysis.recommended_strategy = ConflictResolutionStrategy(best_strategy)
                        analysis.resolution_confidence = min(analysis.resolution_confidence * 1.2, 1.0)
                    except ValueError:
                        pass  # Invalid strategy value
    
    def _find_similar_conflict_patterns(self, conflict: SyncConflict, 
                                      analysis: ConflictAnalysis) -> List[Dict[str, Any]]:
        """Find similar conflict patterns from history."""
        collection_patterns = self.conflict_patterns.get(conflict.collection, [])
        
        similar_patterns = []
        for pattern in collection_patterns:
            # Check similarity based on conflict type and affected fields
            if (pattern.get('conflict_type') == analysis.conflict_type.value and
                len(set(pattern.get('affected_fields', [])) & set(analysis.affected_fields)) > 0):
                similar_patterns.append(pattern)
        
        return similar_patterns[-10:]  # Return last 10 similar patterns
    
    def _record_conflict_pattern(self, conflict: SyncConflict, analysis: ConflictAnalysis):
        """Record conflict pattern for learning."""
        pattern = {
            'timestamp': datetime.now().isoformat(),
            'conflict_type': analysis.conflict_type.value,
            'affected_fields': analysis.affected_fields,
            'severity': analysis.severity,
            'merge_complexity': analysis.merge_complexity,
            'user_impact': analysis.user_impact,
            'recommended_strategy': analysis.recommended_strategy.value,
            'resolution_confidence': analysis.resolution_confidence
        }
        
        self.conflict_patterns[conflict.collection].append(pattern)
        
        # Keep only recent patterns (last 100 per collection)
        if len(self.conflict_patterns[conflict.collection]) > 100:
            self.conflict_patterns[conflict.collection] = self.conflict_patterns[conflict.collection][-100:]
    
    # Resolution strategy implementations
    async def _intelligent_merge_resolution(self, conflict: SyncConflict, 
                                          analysis: ConflictAnalysis) -> Dict[str, Any]:
        """Intelligent merge resolution using field-level analysis."""
        local_data = conflict.local_entry.data
        remote_data = conflict.remote_entry.get('data', {})
        
        merged_data = {}
        
        # Start with remote data as base
        merged_data.update(remote_data)
        
        # Intelligently merge each field
        for field in analysis.affected_fields:
            local_val = local_data.get(field)
            remote_val = remote_data.get(field)
            
            # Determine field type and use appropriate merge handler
            field_type = self._determine_field_type(local_val, remote_val)
            merge_handler = self.field_merge_handlers.get(field_type, self._merge_default_field)
            
            try:
                merged_val = await merge_handler(field, local_val, remote_val, conflict, analysis)
                if merged_val is not None:
                    merged_data[field] = merged_val
            except Exception as e:
                logger.warning(f"Field merge failed for {field}: {e}")
                # Fall back to timestamp-based resolution for this field
                merged_data[field] = self._resolve_field_by_timestamp(
                    local_val, remote_val, conflict
                )
        
        # Add merge metadata
        merged_data['_merge_metadata'] = {
            'strategy': 'intelligent_merge',
            'merged_at': datetime.now().isoformat(),
            'conflict_id': conflict.id,
            'merged_fields': analysis.affected_fields
        }
        
        return merged_data
    
    async def _timestamp_based_resolution(self, conflict: SyncConflict, 
                                        analysis: ConflictAnalysis) -> Dict[str, Any]:
        """Timestamp-based conflict resolution."""
        local_time = conflict.local_entry.timestamp
        remote_time_str = conflict.remote_entry.get('timestamp')
        
        if remote_time_str:
            remote_time = datetime.fromisoformat(remote_time_str.replace('Z', '+00:00'))
            if remote_time > local_time:
                return conflict.remote_entry.get('data', {})
        
        return conflict.local_entry.data
    
    async def _client_wins_resolution(self, conflict: SyncConflict, 
                                    analysis: ConflictAnalysis) -> Dict[str, Any]:
        """Client wins resolution strategy."""
        return conflict.local_entry.data
    
    async def _server_wins_resolution(self, conflict: SyncConflict, 
                                    analysis: ConflictAnalysis) -> Dict[str, Any]:
        """Server wins resolution strategy."""
        return conflict.remote_entry.get('data', {})
    
    async def _field_level_merge_resolution(self, conflict: SyncConflict, 
                                          analysis: ConflictAnalysis) -> Dict[str, Any]:
        """Field-level merge resolution."""
        local_data = conflict.local_entry.data
        remote_data = conflict.remote_entry.get('data', {})
        
        # Start with remote data
        merged_data = copy.deepcopy(remote_data)
        
        # Merge each affected field
        for field in analysis.affected_fields:
            local_val = local_data.get(field)
            remote_val = remote_data.get(field)
            
            # Use timestamp to decide which value to keep
            merged_data[field] = self._resolve_field_by_timestamp(
                local_val, remote_val, conflict
            )
        
        return merged_data
    
    async def _user_choice_resolution(self, conflict: SyncConflict, 
                                    analysis: ConflictAnalysis) -> Dict[str, Any]:
        """User choice resolution (requires external input)."""
        # This would typically wait for user input
        # For now, fall back to intelligent merge
        return await self._intelligent_merge_resolution(conflict, analysis)
    
    def _determine_field_type(self, local_val: Any, remote_val: Any) -> str:
        """Determine the type of field for merge handling."""
        if isinstance(local_val, list) or isinstance(remote_val, list):
            return 'list'
        elif isinstance(local_val, dict) or isinstance(remote_val, dict):
            return 'dict'
        elif isinstance(local_val, str) or isinstance(remote_val, str):
            return 'string'
        elif isinstance(local_val, (int, float)) or isinstance(remote_val, (int, float)):
            return 'number'
        elif isinstance(local_val, bool) or isinstance(remote_val, bool):
            return 'boolean'
        else:
            return 'object'
    
    def _resolve_field_by_timestamp(self, local_val: Any, remote_val: Any, 
                                   conflict: SyncConflict) -> Any:
        """Resolve field value based on timestamp."""
        local_time = conflict.local_entry.timestamp
        remote_time_str = conflict.remote_entry.get('timestamp')
        
        if remote_time_str:
            remote_time = datetime.fromisoformat(remote_time_str.replace('Z', '+00:00'))
            if remote_time > local_time:
                return remote_val
        
        return local_val
    
    # Field merge handlers
    async def _merge_list_field(self, field_name: str, local_val: Any, remote_val: Any,
                              conflict: SyncConflict, analysis: ConflictAnalysis) -> Any:
        """Merge list fields intelligently."""
        if not isinstance(local_val, list):
            local_val = []
        if not isinstance(remote_val, list):
            remote_val = []
        
        # Combine lists and remove duplicates while preserving order
        merged_list = []
        seen = set()
        
        # Add remote items first
        for item in remote_val:
            item_key = json.dumps(item, sort_keys=True) if isinstance(item, dict) else str(item)
            if item_key not in seen:
                merged_list.append(item)
                seen.add(item_key)
        
        # Add local items that aren't already present
        for item in local_val:
            item_key = json.dumps(item, sort_keys=True) if isinstance(item, dict) else str(item)
            if item_key not in seen:
                merged_list.append(item)
                seen.add(item_key)
        
        return merged_list
    
    async def _merge_dict_field(self, field_name: str, local_val: Any, remote_val: Any,
                              conflict: SyncConflict, analysis: ConflictAnalysis) -> Any:
        """Merge dictionary fields recursively."""
        if not isinstance(local_val, dict):
            local_val = {}
        if not isinstance(remote_val, dict):
            remote_val = {}
        
        merged_dict = copy.deepcopy(remote_val)
        
        # Merge local changes
        for key, value in local_val.items():
            if key not in remote_val:
                merged_dict[key] = value
            elif isinstance(value, dict) and isinstance(remote_val[key], dict):
                # Recursively merge nested dictionaries
                merged_dict[key] = {**remote_val[key], **value}
            else:
                # Use timestamp-based resolution for conflicting values
                merged_dict[key] = self._resolve_field_by_timestamp(
                    value, remote_val[key], conflict
                )
        
        return merged_dict
    
    async def _merge_string_field(self, field_name: str, local_val: Any, remote_val: Any,
                                conflict: SyncConflict, analysis: ConflictAnalysis) -> Any:
        """Merge string fields using diff-based approach."""
        if not isinstance(local_val, str):
            local_val = str(local_val) if local_val is not None else ""
        if not isinstance(remote_val, str):
            remote_val = str(remote_val) if remote_val is not None else ""
        
        # For short strings, use timestamp-based resolution
        if len(local_val) < 100 and len(remote_val) < 100:
            return self._resolve_field_by_timestamp(local_val, remote_val, conflict)
        
        # For longer strings, attempt intelligent merge
        # This is a simplified implementation - in practice, you might use
        # more sophisticated text merging algorithms
        if local_val == remote_val:
            return local_val
        
        # Use timestamp-based resolution as fallback
        return self._resolve_field_by_timestamp(local_val, remote_val, conflict)
    
    async def _merge_number_field(self, field_name: str, local_val: Any, remote_val: Any,
                                conflict: SyncConflict, analysis: ConflictAnalysis) -> Any:
        """Merge numeric fields."""
        # For numbers, use timestamp-based resolution
        return self._resolve_field_by_timestamp(local_val, remote_val, conflict)
    
    async def _merge_boolean_field(self, field_name: str, local_val: Any, remote_val: Any,
                                 conflict: SyncConflict, analysis: ConflictAnalysis) -> Any:
        """Merge boolean fields."""
        # For booleans, use timestamp-based resolution
        return self._resolve_field_by_timestamp(local_val, remote_val, conflict)
    
    async def _merge_timestamp_field(self, field_name: str, local_val: Any, remote_val: Any,
                                   conflict: SyncConflict, analysis: ConflictAnalysis) -> Any:
        """Merge timestamp fields."""
        # For timestamps, use the more recent one
        try:
            if isinstance(local_val, str):
                local_time = datetime.fromisoformat(local_val.replace('Z', '+00:00'))
            else:
                local_time = local_val
            
            if isinstance(remote_val, str):
                remote_time = datetime.fromisoformat(remote_val.replace('Z', '+00:00'))
            else:
                remote_time = remote_val
            
            return remote_val if remote_time > local_time else local_val
        except:
            # Fall back to timestamp-based resolution
            return self._resolve_field_by_timestamp(local_val, remote_val, conflict)
    
    async def _merge_array_field(self, field_name: str, local_val: Any, remote_val: Any,
                               conflict: SyncConflict, analysis: ConflictAnalysis) -> Any:
        """Merge array fields (alias for list merge)."""
        return await self._merge_list_field(field_name, local_val, remote_val, conflict, analysis)
    
    async def _merge_object_field(self, field_name: str, local_val: Any, remote_val: Any,
                                conflict: SyncConflict, analysis: ConflictAnalysis) -> Any:
        """Merge object fields (alias for dict merge)."""
        return await self._merge_dict_field(field_name, local_val, remote_val, conflict, analysis)
    
    async def _merge_default_field(self, field_name: str, local_val: Any, remote_val: Any,
                                 conflict: SyncConflict, analysis: ConflictAnalysis) -> Any:
        """Default field merge handler."""
        return self._resolve_field_by_timestamp(local_val, remote_val, conflict)
    
    # Public API methods
    async def resolve_conflict_intelligently(self, conflict: SyncConflict) -> Dict[str, Any]:
        """Resolve conflict using intelligent analysis and resolution."""
        start_time = time.time()
        
        try:
            # Analyze the conflict
            analysis = await self.analyze_conflict(conflict)
            
            # Get resolution strategy
            strategy = analysis.recommended_strategy
            resolver = self.resolution_strategies.get(strategy)
            
            if not resolver:
                logger.warning(f"No resolver found for strategy {strategy}, using fallback")
                resolver = self.resolution_strategies[ConflictResolutionStrategy.TIMESTAMP_BASED]
            
            # Resolve the conflict
            resolved_data = await resolver(conflict, analysis)
            
            # Record resolution metrics
            resolution_time = time.time() - start_time
            self.metrics.conflicts_resolved += 1
            self.metrics.avg_conflict_resolution_time = (
                (self.metrics.avg_conflict_resolution_time * (self.metrics.conflicts_resolved - 1) + resolution_time) /
                self.metrics.conflicts_resolved
            )
            
            if analysis.auto_resolvable:
                self.metrics.auto_resolved_conflicts += 1
            else:
                self.metrics.manual_resolved_conflicts += 1
            
            # Update learning data
            await self._update_resolution_success_rate(strategy.value, True)
            
            # Record successful resolution pattern
            self._record_resolution_success(conflict, analysis, strategy)
            
            return resolved_data
            
        except Exception as e:
            logger.error(f"Conflict resolution failed: {e}")
            
            # Update failure metrics
            await self._update_resolution_success_rate(
                analysis.recommended_strategy.value if 'analysis' in locals() else 'unknown', 
                False
            )
            
            # Fall back to timestamp-based resolution
            return await self._timestamp_based_resolution(conflict, None)
    
    async def _update_resolution_success_rate(self, strategy: str, success: bool):
        """Update success rate for resolution strategy."""
        current_rate = self.resolution_success_rates.get(strategy, 0.5)
        
        # Simple exponential moving average
        alpha = 0.1  # Learning rate
        new_rate = current_rate * (1 - alpha) + (1.0 if success else 0.0) * alpha
        
        self.resolution_success_rates[strategy] = new_rate
    
    def _record_resolution_success(self, conflict: SyncConflict, 
                                 analysis: ConflictAnalysis, 
                                 strategy: ConflictResolutionStrategy):
        """Record successful resolution for learning."""
        # Update the conflict pattern with resolution information
        collection_patterns = self.conflict_patterns.get(conflict.collection, [])
        
        if collection_patterns:
            # Find the most recent pattern for this conflict
            for pattern in reversed(collection_patterns):
                if (pattern.get('conflict_type') == analysis.conflict_type.value and
                    set(pattern.get('affected_fields', [])) == set(analysis.affected_fields)):
                    pattern['resolution_successful'] = True
                    pattern['resolution_strategy'] = strategy.value
                    pattern['resolution_time'] = datetime.now().isoformat()
                    break
    
    def get_sync_metrics(self) -> Dict[str, Any]:
        """Get synchronization metrics."""
        return {
            'total_syncs': self.metrics.total_syncs,
            'successful_syncs': self.metrics.successful_syncs,
            'failed_syncs': self.metrics.failed_syncs,
            'success_rate': self.metrics.successful_syncs / max(self.metrics.total_syncs, 1),
            'conflicts_detected': self.metrics.conflicts_detected,
            'conflicts_resolved': self.metrics.conflicts_resolved,
            'auto_resolved_conflicts': self.metrics.auto_resolved_conflicts,
            'manual_resolved_conflicts': self.metrics.manual_resolved_conflicts,
            'auto_resolution_rate': self.metrics.auto_resolved_conflicts / max(self.metrics.conflicts_resolved, 1),
            'avg_sync_time': self.metrics.avg_sync_time,
            'avg_conflict_resolution_time': self.metrics.avg_conflict_resolution_time,
            'data_transferred_mb': self.metrics.data_transferred_mb,
            'last_sync_timestamp': self.metrics.last_sync_timestamp.isoformat() if self.metrics.last_sync_timestamp else None,
            'resolution_success_rates': self.resolution_success_rates.copy(),
            'active_syncs': len(self.active_syncs),
            'queued_batches': len(self.sync_batches)
        }
    
    def get_conflict_insights(self) -> Dict[str, Any]:
        """Get insights about conflict patterns."""
        total_patterns = sum(len(patterns) for patterns in self.conflict_patterns.values())
        
        if total_patterns == 0:
            return {'total_patterns': 0}
        
        # Analyze conflict types
        conflict_type_counts = defaultdict(int)
        resolution_strategy_counts = defaultdict(int)
        
        for collection, patterns in self.conflict_patterns.items():
            for pattern in patterns:
                conflict_type_counts[pattern.get('conflict_type', 'unknown')] += 1
                if pattern.get('resolution_strategy'):
                    resolution_strategy_counts[pattern['resolution_strategy']] += 1
        
        return {
            'total_patterns': total_patterns,
            'collections_with_conflicts': len(self.conflict_patterns),
            'conflict_type_distribution': dict(conflict_type_counts),
            'resolution_strategy_distribution': dict(resolution_strategy_counts),
            'most_common_conflict_type': max(conflict_type_counts.items(), key=lambda x: x[1])[0] if conflict_type_counts else None,
            'most_successful_strategy': max(self.resolution_success_rates.items(), key=lambda x: x[1])[0] if self.resolution_success_rates else None
        }


# Global instance
intelligent_sync_engine = None

def get_sync_engine(offline_manager=None):
    """Get or create the global sync engine instance."""
    global intelligent_sync_engine
    if intelligent_sync_engine is None and offline_manager:
        intelligent_sync_engine = IntelligentSyncEngine(offline_manager)
    return intelligent_sync_engine