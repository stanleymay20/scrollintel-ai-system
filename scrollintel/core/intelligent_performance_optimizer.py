"""
Intelligent Performance Optimization Engine for ScrollIntel
Implements adaptive performance optimization based on device capabilities,
dynamic resource allocation with load prediction, intelligent caching with
predictive pre-loading, and progressive enhancement with automatic adaptation.
"""

import asyncio
import logging
import time
import json
import hashlib
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
from collections import defaultdict, deque
import psutil
import threading
from concurrent.futures import ThreadPoolExecutor
import pickle
import os

from .performance_optimizer import ScrollIntelPerformanceOptimizer, OptimizationType, PerformanceMetrics
from .never_fail_decorators import never_fail_api_endpoint
from .graceful_degradation import degradation_manager, DegradationLevel

logger = logging.getLogger(__name__)


class DeviceCapability(Enum):
    """Device capability classifications."""
    HIGH_END = "high_end"
    MEDIUM_END = "medium_end"
    LOW_END = "low_end"
    MOBILE = "mobile"
    TABLET = "tablet"
    DESKTOP = "desktop"


class OptimizationStrategy(Enum):
    """Performance optimization strategies."""
    AGGRESSIVE = "aggressive"
    BALANCED = "balanced"
    CONSERVATIVE = "conservative"
    ADAPTIVE = "adaptive"


@dataclass
class DeviceProfile:
    """Device capability profile for optimization decisions."""
    device_id: str
    device_type: DeviceCapability
    cpu_cores: int
    memory_gb: float
    network_speed: float  # Mbps
    screen_resolution: Tuple[int, int]
    supports_webgl: bool
    supports_webworkers: bool
    battery_level: Optional[float] = None
    is_mobile: bool = False
    performance_score: float = 1.0
    last_updated: datetime = field(default_factory=datetime.utcnow)


@dataclass
class LoadPrediction:
    """Load prediction for resource allocation."""
    timestamp: datetime
    predicted_cpu_usage: float
    predicted_memory_usage: float
    predicted_network_usage: float
    predicted_concurrent_users: int
    confidence_score: float
    prediction_horizon: timedelta
    factors: Dict[str, float] = field(default_factory=dict)


@dataclass
class CacheItem:
    """Intelligent cache item with predictive metadata."""
    key: str
    data: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int
    size_bytes: int
    priority_score: float
    predicted_next_access: Optional[datetime] = None
    user_context: Optional[Dict[str, Any]] = None
    expiry_time: Optional[datetime] = None


@dataclass
class ProgressiveEnhancement:
    """Progressive enhancement configuration."""
    base_features: List[str]
    enhanced_features: Dict[DeviceCapability, List[str]]
    fallback_features: List[str]
    adaptation_rules: Dict[str, Any]


class DeviceCapabilityDetector:
    """Detects and profiles device capabilities for optimization."""
    
    def __init__(self):
        self.device_profiles: Dict[str, DeviceProfile] = {}
        self.capability_thresholds = {
            DeviceCapability.HIGH_END: {
                'cpu_cores': 8,
                'memory_gb': 16,
                'performance_score': 0.8
            },
            DeviceCapability.MEDIUM_END: {
                'cpu_cores': 4,
                'memory_gb': 8,
                'performance_score': 0.5
            },
            DeviceCapability.LOW_END: {
                'cpu_cores': 2,
                'memory_gb': 4,
                'performance_score': 0.3
            }
        }
    
    async def detect_device_capabilities(self, user_agent: str, 
                                       client_hints: Dict[str, Any] = None) -> DeviceProfile:
        """Detect device capabilities from user agent and client hints."""
        device_id = hashlib.md5(user_agent.encode()).hexdigest()
        
        # Check if we have cached profile
        if device_id in self.device_profiles:
            profile = self.device_profiles[device_id]
            if (datetime.utcnow() - profile.last_updated).total_seconds() < 3600:  # 1 hour cache
                return profile
        
        # Parse user agent for basic device info
        is_mobile = any(term in user_agent.lower() for term in ['mobile', 'android', 'iphone'])
        is_tablet = any(term in user_agent.lower() for term in ['tablet', 'ipad'])
        
        # Determine device type
        if is_mobile:
            device_type = DeviceCapability.MOBILE
        elif is_tablet:
            device_type = DeviceCapability.TABLET
        else:
            device_type = DeviceCapability.DESKTOP
        
        # Extract capabilities from client hints or defaults
        client_hints = client_hints or {}
        cpu_cores = client_hints.get('cpu_cores', self._estimate_cpu_cores(user_agent))
        memory_gb = client_hints.get('memory_gb', self._estimate_memory(user_agent))
        network_speed = client_hints.get('network_speed', self._estimate_network_speed(user_agent))
        screen_resolution = client_hints.get('screen_resolution', (1920, 1080))
        supports_webgl = client_hints.get('webgl', True)
        supports_webworkers = client_hints.get('webworkers', True)
        
        # Calculate performance score
        performance_score = self._calculate_performance_score(
            cpu_cores, memory_gb, network_speed, device_type
        )
        
        # Create device profile
        profile = DeviceProfile(
            device_id=device_id,
            device_type=device_type,
            cpu_cores=cpu_cores,
            memory_gb=memory_gb,
            network_speed=network_speed,
            screen_resolution=screen_resolution,
            supports_webgl=supports_webgl,
            supports_webworkers=supports_webworkers,
            is_mobile=is_mobile,
            performance_score=performance_score
        )
        
        # Cache profile
        self.device_profiles[device_id] = profile
        
        logger.info(f"Detected device capabilities: {device_type.value}, "
                   f"performance score: {performance_score:.2f}")
        
        return profile
    
    def _estimate_cpu_cores(self, user_agent: str) -> int:
        """Estimate CPU cores from user agent."""
        if 'mobile' in user_agent.lower():
            return 4  # Modern mobile devices
        elif 'tablet' in user_agent.lower():
            return 6  # Tablets typically have more cores
        else:
            return 8  # Desktop default
    
    def _estimate_memory(self, user_agent: str) -> float:
        """Estimate memory from user agent."""
        if 'mobile' in user_agent.lower():
            return 4.0  # Mobile devices
        elif 'tablet' in user_agent.lower():
            return 6.0  # Tablets
        else:
            return 16.0  # Desktop default
    
    def _estimate_network_speed(self, user_agent: str) -> float:
        """Estimate network speed from user agent."""
        if 'mobile' in user_agent.lower():
            return 50.0  # Mobile network
        else:
            return 100.0  # Broadband default
    
    def _calculate_performance_score(self, cpu_cores: int, memory_gb: float, 
                                   network_speed: float, device_type: DeviceCapability) -> float:
        """Calculate overall performance score."""
        # Normalize metrics
        cpu_score = min(cpu_cores / 16.0, 1.0)  # Max 16 cores
        memory_score = min(memory_gb / 32.0, 1.0)  # Max 32GB
        network_score = min(network_speed / 1000.0, 1.0)  # Max 1Gbps
        
        # Weight factors
        weights = {
            DeviceCapability.MOBILE: {'cpu': 0.4, 'memory': 0.4, 'network': 0.2},
            DeviceCapability.TABLET: {'cpu': 0.4, 'memory': 0.3, 'network': 0.3},
            DeviceCapability.DESKTOP: {'cpu': 0.3, 'memory': 0.3, 'network': 0.4}
        }
        
        device_weights = weights.get(device_type, weights[DeviceCapability.DESKTOP])
        
        score = (cpu_score * device_weights['cpu'] + 
                memory_score * device_weights['memory'] + 
                network_score * device_weights['network'])
        
        return score


class LoadPredictor:
    """Predicts system load for dynamic resource allocation."""
    
    def __init__(self):
        self.historical_metrics: deque = deque(maxlen=1000)
        self.prediction_models: Dict[str, Any] = {}
        self.seasonal_patterns: Dict[str, List[float]] = {}
        self.trend_coefficients: Dict[str, float] = {}
        
        # Initialize prediction models
        self._initialize_prediction_models()
    
    def _initialize_prediction_models(self):
        """Initialize simple prediction models."""
        # Simple moving average models for different metrics
        self.prediction_models = {
            'cpu_usage': {'window': 10, 'weights': np.array([0.1, 0.15, 0.2, 0.25, 0.3])},
            'memory_usage': {'window': 15, 'weights': np.array([0.1, 0.1, 0.15, 0.2, 0.25, 0.2])},
            'network_usage': {'window': 5, 'weights': np.array([0.2, 0.3, 0.5])},
            'concurrent_users': {'window': 20, 'weights': np.linspace(0.1, 0.5, 10)}
        }
        
        # Initialize seasonal patterns (hourly patterns)
        self.seasonal_patterns = {
            'cpu_usage': [0.3, 0.2, 0.2, 0.2, 0.3, 0.4, 0.6, 0.8, 0.9, 1.0, 
                         1.0, 0.9, 0.8, 0.9, 1.0, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.3],
            'concurrent_users': [0.1, 0.1, 0.1, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0,
                               1.0, 0.9, 0.8, 0.9, 1.0, 1.0, 0.9, 0.8, 0.6, 0.4, 0.3, 0.2, 0.1, 0.1]
        }
    
    async def predict_load(self, prediction_horizon: timedelta = timedelta(minutes=30)) -> LoadPrediction:
        """Predict system load for the given time horizon."""
        current_time = datetime.utcnow()
        
        # Collect recent metrics
        recent_metrics = list(self.historical_metrics)[-50:]  # Last 50 samples
        
        if len(recent_metrics) < 5:
            # Not enough data, return conservative prediction
            return LoadPrediction(
                timestamp=current_time,
                predicted_cpu_usage=50.0,
                predicted_memory_usage=50.0,
                predicted_network_usage=50.0,
                predicted_concurrent_users=10,
                confidence_score=0.3,
                prediction_horizon=prediction_horizon,
                factors={'insufficient_data': 1.0}
            )
        
        # Predict each metric
        cpu_prediction = self._predict_metric('cpu_usage', recent_metrics, prediction_horizon)
        memory_prediction = self._predict_metric('memory_usage', recent_metrics, prediction_horizon)
        network_prediction = self._predict_metric('network_usage', recent_metrics, prediction_horizon)
        users_prediction = self._predict_metric('concurrent_users', recent_metrics, prediction_horizon)
        
        # Calculate confidence based on prediction variance
        confidence = self._calculate_prediction_confidence(recent_metrics)
        
        # Identify key factors affecting the prediction
        factors = self._identify_prediction_factors(current_time, recent_metrics)
        
        prediction = LoadPrediction(
            timestamp=current_time,
            predicted_cpu_usage=cpu_prediction,
            predicted_memory_usage=memory_prediction,
            predicted_network_usage=network_prediction,
            predicted_concurrent_users=int(users_prediction),
            confidence_score=confidence,
            prediction_horizon=prediction_horizon,
            factors=factors
        )
        
        logger.debug(f"Load prediction: CPU {cpu_prediction:.1f}%, "
                    f"Memory {memory_prediction:.1f}%, "
                    f"Users {int(users_prediction)}, "
                    f"Confidence {confidence:.2f}")
        
        return prediction
    
    def _predict_metric(self, metric_name: str, recent_metrics: List[Any], 
                       horizon: timedelta) -> float:
        """Predict a specific metric using trend and seasonal analysis."""
        if not recent_metrics:
            return 50.0  # Default prediction
        
        # Extract metric values
        values = []
        for metric in recent_metrics:
            if hasattr(metric, metric_name):
                values.append(getattr(metric, metric_name))
            elif isinstance(metric, dict) and metric_name in metric:
                values.append(metric[metric_name])
        
        if not values:
            return 50.0
        
        # Calculate trend
        if len(values) >= 3:
            trend = (values[-1] - values[-3]) / 2.0  # Simple trend calculation
        else:
            trend = 0.0
        
        # Apply seasonal adjustment
        current_hour = datetime.utcnow().hour
        seasonal_factor = self.seasonal_patterns.get(metric_name, [1.0] * 24)[current_hour]
        
        # Predict future value
        base_value = np.mean(values[-5:])  # Average of last 5 values
        trend_adjustment = trend * (horizon.total_seconds() / 3600.0)  # Scale by hours
        seasonal_adjustment = base_value * (seasonal_factor - 1.0) * 0.2  # 20% seasonal impact
        
        predicted_value = base_value + trend_adjustment + seasonal_adjustment
        
        # Clamp to reasonable bounds
        if metric_name in ['cpu_usage', 'memory_usage', 'network_usage']:
            predicted_value = max(0.0, min(100.0, predicted_value))
        elif metric_name == 'concurrent_users':
            predicted_value = max(0.0, predicted_value)
        
        return predicted_value
    
    def _calculate_prediction_confidence(self, recent_metrics: List[Any]) -> float:
        """Calculate confidence in predictions based on data stability."""
        if len(recent_metrics) < 5:
            return 0.3
        
        # Calculate variance in recent metrics
        cpu_values = [getattr(m, 'cpu_usage', 50) for m in recent_metrics[-10:]]
        cpu_variance = np.var(cpu_values) if cpu_values else 100
        
        # Lower variance = higher confidence
        confidence = max(0.1, min(1.0, 1.0 - (cpu_variance / 1000.0)))
        
        return confidence
    
    def _identify_prediction_factors(self, current_time: datetime, 
                                   recent_metrics: List[Any]) -> Dict[str, float]:
        """Identify factors affecting the prediction."""
        factors = {}
        
        # Time-based factors
        hour = current_time.hour
        if 9 <= hour <= 17:
            factors['business_hours'] = 0.8
        elif 18 <= hour <= 22:
            factors['evening_peak'] = 0.6
        else:
            factors['off_hours'] = 0.3
        
        # Day of week factor
        weekday = current_time.weekday()
        if weekday < 5:  # Monday to Friday
            factors['weekday'] = 0.7
        else:
            factors['weekend'] = 0.4
        
        # Recent trend factor
        if len(recent_metrics) >= 5:
            recent_cpu = [getattr(m, 'cpu_usage', 50) for m in recent_metrics[-5:]]
            if recent_cpu[-1] > recent_cpu[0]:
                factors['increasing_load'] = 0.8
            else:
                factors['decreasing_load'] = 0.4
        
        return factors
    
    def record_metrics(self, metrics: PerformanceMetrics):
        """Record metrics for prediction learning."""
        self.historical_metrics.append(metrics)


class IntelligentCacheManager:
    """Intelligent caching system with predictive pre-loading."""
    
    def __init__(self, max_cache_size: int = 1000):
        self.cache: Dict[str, CacheItem] = {}
        self.max_cache_size = max_cache_size
        self.access_patterns: Dict[str, List[datetime]] = defaultdict(list)
        self.user_patterns: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.preload_queue: asyncio.Queue = asyncio.Queue()
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'preload_hits': 0,
            'evictions': 0
        }
        
        # Start background preloading task (only if event loop is running)
        try:
            asyncio.create_task(self._preload_worker())
        except RuntimeError:
            # No event loop running, will start worker later
            pass
    
    async def get(self, key: str, user_id: str = None) -> Optional[Any]:
        """Get item from cache with intelligent access tracking."""
        if key in self.cache:
            item = self.cache[key]
            
            # Update access statistics
            item.last_accessed = datetime.utcnow()
            item.access_count += 1
            self.cache_stats['hits'] += 1
            
            # Record access pattern
            self.access_patterns[key].append(datetime.utcnow())
            if len(self.access_patterns[key]) > 100:
                self.access_patterns[key] = self.access_patterns[key][-100:]
            
            # Update user patterns
            if user_id:
                self._update_user_patterns(user_id, key)
            
            # Check if item is expired
            if item.expiry_time and datetime.utcnow() > item.expiry_time:
                del self.cache[key]
                self.cache_stats['misses'] += 1
                return None
            
            # Predict next access and schedule preloading of related items
            await self._schedule_predictive_preloading(key, user_id)
            
            return item.data
        
        self.cache_stats['misses'] += 1
        return None
    
    async def set(self, key: str, data: Any, user_id: str = None, 
                 ttl: Optional[timedelta] = None, priority: float = 1.0) -> None:
        """Set item in cache with intelligent priority management."""
        # Calculate item size (rough estimate)
        size_bytes = len(str(data)) if data else 0
        
        # Create cache item
        expiry_time = datetime.utcnow() + ttl if ttl else None
        item = CacheItem(
            key=key,
            data=data,
            created_at=datetime.utcnow(),
            last_accessed=datetime.utcnow(),
            access_count=1,
            size_bytes=size_bytes,
            priority_score=priority,
            expiry_time=expiry_time,
            user_context={'user_id': user_id} if user_id else None
        )
        
        # Check if cache is full and evict if necessary
        if len(self.cache) >= self.max_cache_size:
            await self._evict_items()
        
        self.cache[key] = item
        
        # Update user patterns
        if user_id:
            self._update_user_patterns(user_id, key)
    
    async def _evict_items(self):
        """Intelligently evict cache items based on priority and access patterns."""
        if not self.cache:
            return
        
        # Calculate eviction scores for all items
        eviction_candidates = []
        current_time = datetime.utcnow()
        
        for key, item in self.cache.items():
            # Factors for eviction score (lower score = more likely to evict)
            time_since_access = (current_time - item.last_accessed).total_seconds()
            access_frequency = item.access_count / max(1, time_since_access / 3600)  # accesses per hour
            
            # Predict next access time
            predicted_next_access = self._predict_next_access(key)
            time_to_next_access = float('inf')
            if predicted_next_access:
                time_to_next_access = (predicted_next_access - current_time).total_seconds()
            
            eviction_score = (
                item.priority_score * 0.3 +
                access_frequency * 0.3 +
                (1.0 / max(1, time_since_access / 3600)) * 0.2 +
                (1.0 / max(1, time_to_next_access / 3600)) * 0.2
            )
            
            eviction_candidates.append((key, eviction_score))
        
        # Sort by eviction score (lowest first) and evict bottom 10%
        eviction_candidates.sort(key=lambda x: x[1])
        items_to_evict = max(1, len(eviction_candidates) // 10)
        
        for key, _ in eviction_candidates[:items_to_evict]:
            del self.cache[key]
            self.cache_stats['evictions'] += 1
        
        logger.debug(f"Evicted {items_to_evict} cache items")
    
    def _predict_next_access(self, key: str) -> Optional[datetime]:
        """Predict when a cache item will be accessed next."""
        if key not in self.access_patterns or len(self.access_patterns[key]) < 3:
            return None
        
        accesses = self.access_patterns[key]
        
        # Calculate average time between accesses
        intervals = []
        for i in range(1, len(accesses)):
            interval = (accesses[i] - accesses[i-1]).total_seconds()
            intervals.append(interval)
        
        if not intervals:
            return None
        
        # Use median interval for prediction (more robust than mean)
        median_interval = np.median(intervals)
        predicted_next = accesses[-1] + timedelta(seconds=median_interval)
        
        return predicted_next
    
    def _update_user_patterns(self, user_id: str, key: str):
        """Update user access patterns for personalized caching."""
        if 'access_sequence' not in self.user_patterns[user_id]:
            self.user_patterns[user_id]['access_sequence'] = deque(maxlen=50)
        
        self.user_patterns[user_id]['access_sequence'].append({
            'key': key,
            'timestamp': datetime.utcnow()
        })
        
        # Update key preferences
        if 'key_preferences' not in self.user_patterns[user_id]:
            self.user_patterns[user_id]['key_preferences'] = defaultdict(int)
        
        self.user_patterns[user_id]['key_preferences'][key] += 1
    
    async def _schedule_predictive_preloading(self, accessed_key: str, user_id: str = None):
        """Schedule predictive preloading of related items."""
        preload_candidates = []
        
        # Find items commonly accessed after this key
        if user_id and user_id in self.user_patterns:
            sequence = list(self.user_patterns[user_id]['access_sequence'])
            
            # Look for patterns in access sequence
            for i, access in enumerate(sequence):
                if access['key'] == accessed_key and i < len(sequence) - 1:
                    next_key = sequence[i + 1]['key']
                    if next_key not in self.cache:
                        preload_candidates.append(next_key)
        
        # Add to preload queue
        for candidate in preload_candidates[:3]:  # Limit to 3 candidates
            try:
                await self.preload_queue.put(candidate)
            except asyncio.QueueFull:
                break
    
    async def _preload_worker(self):
        """Background worker for predictive preloading."""
        while True:
            try:
                key_to_preload = await asyncio.wait_for(
                    self.preload_queue.get(), timeout=1.0
                )
                
                # Simulate preloading (in real implementation, this would
                # call the appropriate data loading function)
                logger.debug(f"Preloading cache key: {key_to_preload}")
                
                # Mark task as done
                self.preload_queue.task_done()
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Preload worker error: {e}")
                await asyncio.sleep(1)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        total_requests = self.cache_stats['hits'] + self.cache_stats['misses']
        hit_rate = self.cache_stats['hits'] / max(1, total_requests)
        
        return {
            'hit_rate': hit_rate,
            'total_items': len(self.cache),
            'total_requests': total_requests,
            'preload_hits': self.cache_stats['preload_hits'],
            'evictions': self.cache_stats['evictions'],
            'cache_size_mb': sum(item.size_bytes for item in self.cache.values()) / (1024 * 1024)
        }


class ProgressiveEnhancementManager:
    """Manages progressive enhancement with automatic adaptation."""
    
    def __init__(self):
        self.enhancement_configs: Dict[str, ProgressiveEnhancement] = {}
        self.device_adaptations: Dict[str, Dict[str, Any]] = {}
        self.feature_performance: Dict[str, Dict[DeviceCapability, float]] = defaultdict(dict)
        
        # Initialize default enhancement configurations
        self._setup_default_enhancements()
    
    def _setup_default_enhancements(self):
        """Setup default progressive enhancement configurations."""
        
        # Dashboard enhancements
        self.enhancement_configs['dashboard'] = ProgressiveEnhancement(
            base_features=['basic_charts', 'data_tables', 'simple_filters'],
            enhanced_features={
                DeviceCapability.HIGH_END: [
                    'interactive_charts', 'real_time_updates', 'advanced_filters',
                    'animations', 'webgl_visualizations', 'parallel_processing'
                ],
                DeviceCapability.MEDIUM_END: [
                    'interactive_charts', 'periodic_updates', 'standard_filters',
                    'basic_animations'
                ],
                DeviceCapability.LOW_END: [
                    'static_charts', 'manual_refresh', 'basic_filters'
                ],
                DeviceCapability.MOBILE: [
                    'mobile_optimized_charts', 'touch_interactions', 'simplified_ui'
                ]
            },
            fallback_features=['data_tables', 'text_summaries', 'basic_navigation'],
            adaptation_rules={
                'memory_threshold': 0.8,
                'cpu_threshold': 0.9,
                'network_threshold': 1.0  # Mbps
            }
        )
        
        # Data processing enhancements
        self.enhancement_configs['data_processing'] = ProgressiveEnhancement(
            base_features=['basic_analysis', 'simple_aggregations'],
            enhanced_features={
                DeviceCapability.HIGH_END: [
                    'advanced_analytics', 'machine_learning', 'parallel_processing',
                    'large_dataset_support', 'real_time_processing'
                ],
                DeviceCapability.MEDIUM_END: [
                    'standard_analytics', 'batch_processing', 'medium_datasets'
                ],
                DeviceCapability.LOW_END: [
                    'basic_analytics', 'small_datasets', 'simplified_processing'
                ],
                DeviceCapability.MOBILE: [
                    'mobile_optimized_processing', 'cloud_processing', 'result_caching'
                ]
            },
            fallback_features=['cached_results', 'pre_computed_summaries'],
            adaptation_rules={
                'dataset_size_threshold': 1000000,  # 1M records
                'processing_time_threshold': 30.0   # 30 seconds
            }
        )
        
        # Visualization enhancements
        self.enhancement_configs['visualization'] = ProgressiveEnhancement(
            base_features=['bar_charts', 'line_charts', 'pie_charts'],
            enhanced_features={
                DeviceCapability.HIGH_END: [
                    '3d_visualizations', 'webgl_rendering', 'interactive_animations',
                    'complex_charts', 'real_time_streaming', 'custom_shaders'
                ],
                DeviceCapability.MEDIUM_END: [
                    'svg_charts', 'basic_interactions', 'standard_animations',
                    'medium_complexity_charts'
                ],
                DeviceCapability.LOW_END: [
                    'canvas_charts', 'static_images', 'simple_charts'
                ],
                DeviceCapability.MOBILE: [
                    'mobile_charts', 'touch_optimized', 'responsive_design'
                ]
            },
            fallback_features=['static_images', 'data_tables', 'text_descriptions'],
            adaptation_rules={
                'render_time_threshold': 2.0,
                'frame_rate_threshold': 30
            }
        )
    
    async def adapt_features_for_device(self, feature_set: str, device_profile: DeviceProfile,
                                      current_performance: Dict[str, float] = None) -> List[str]:
        """Adapt features based on device capabilities and current performance."""
        if feature_set not in self.enhancement_configs:
            logger.warning(f"Unknown feature set: {feature_set}")
            return []
        
        config = self.enhancement_configs[feature_set]
        
        # Start with base features
        enabled_features = config.base_features.copy()
        
        # Determine device capability level
        device_capability = self._determine_device_capability(device_profile)
        
        # Get enhanced features for this device capability
        enhanced_features = config.enhanced_features.get(device_capability, [])
        
        # Apply performance-based filtering
        if current_performance:
            enhanced_features = self._filter_features_by_performance(
                enhanced_features, current_performance, config.adaptation_rules
            )
        
        # Add enhanced features
        enabled_features.extend(enhanced_features)
        
        # Record adaptation decision
        self.device_adaptations[device_profile.device_id] = {
            'feature_set': feature_set,
            'device_capability': device_capability.value,
            'enabled_features': enabled_features,
            'timestamp': datetime.utcnow(),
            'performance_factors': current_performance or {}
        }
        
        logger.info(f"Adapted {feature_set} for {device_capability.value}: "
                   f"{len(enabled_features)} features enabled")
        
        return enabled_features
    
    def _determine_device_capability(self, device_profile: DeviceProfile) -> DeviceCapability:
        """Determine device capability level from profile."""
        # Mobile devices have their own category
        if device_profile.is_mobile:
            return DeviceCapability.MOBILE
        
        # Use performance score to determine capability
        if device_profile.performance_score >= 0.8:
            return DeviceCapability.HIGH_END
        elif device_profile.performance_score >= 0.5:
            return DeviceCapability.MEDIUM_END
        else:
            return DeviceCapability.LOW_END
    
    def _filter_features_by_performance(self, features: List[str], 
                                      performance: Dict[str, float],
                                      rules: Dict[str, Any]) -> List[str]:
        """Filter features based on current performance metrics."""
        filtered_features = []
        
        # Check performance thresholds
        memory_usage = performance.get('memory_usage', 0.0)
        cpu_usage = performance.get('cpu_usage', 0.0)
        network_speed = performance.get('network_speed', 100.0)
        
        memory_threshold = rules.get('memory_threshold', 0.8)
        cpu_threshold = rules.get('cpu_threshold', 0.9)
        network_threshold = rules.get('network_threshold', 1.0)
        
        for feature in features:
            # Skip resource-intensive features if performance is poor
            if self._is_resource_intensive_feature(feature):
                if (memory_usage > memory_threshold or 
                    cpu_usage > cpu_threshold or 
                    network_speed < network_threshold):
                    logger.debug(f"Skipping resource-intensive feature: {feature}")
                    continue
            
            filtered_features.append(feature)
        
        return filtered_features
    
    def _is_resource_intensive_feature(self, feature: str) -> bool:
        """Check if a feature is resource-intensive."""
        intensive_features = [
            'webgl_visualizations', '3d_visualizations', 'real_time_updates',
            'parallel_processing', 'machine_learning', 'advanced_analytics',
            'interactive_animations', 'custom_shaders', 'real_time_streaming'
        ]
        return feature in intensive_features
    
    async def record_feature_performance(self, feature_set: str, feature: str,
                                       device_capability: DeviceCapability,
                                       performance_score: float):
        """Record performance of a feature for learning."""
        self.feature_performance[feature_set][device_capability] = performance_score
        
        # Adapt thresholds based on performance feedback
        if performance_score < 0.5:  # Poor performance
            config = self.enhancement_configs.get(feature_set)
            if config and device_capability in config.enhanced_features:
                # Remove poorly performing feature
                if feature in config.enhanced_features[device_capability]:
                    config.enhanced_features[device_capability].remove(feature)
                    logger.info(f"Removed poorly performing feature: {feature}")
    
    def get_adaptation_stats(self) -> Dict[str, Any]:
        """Get adaptation statistics."""
        stats = {
            'total_adaptations': len(self.device_adaptations),
            'feature_sets': list(self.enhancement_configs.keys()),
            'device_capabilities': {},
            'feature_performance': dict(self.feature_performance)
        }
        
        # Count adaptations by device capability
        for adaptation in self.device_adaptations.values():
            capability = adaptation['device_capability']
            stats['device_capabilities'][capability] = stats['device_capabilities'].get(capability, 0) + 1
        
        return stats


class IntelligentPerformanceOptimizer:
    """Main intelligent performance optimization engine."""
    
    def __init__(self, base_optimizer: ScrollIntelPerformanceOptimizer = None):
        self.base_optimizer = base_optimizer
        self.device_detector = DeviceCapabilityDetector()
        self.load_predictor = LoadPredictor()
        self.cache_manager = IntelligentCacheManager()
        self.enhancement_manager = ProgressiveEnhancementManager()
        
        # Optimization state
        self.current_strategy = OptimizationStrategy.ADAPTIVE
        self.optimization_history: deque = deque(maxlen=1000)
        self.resource_allocations: Dict[str, Dict[str, float]] = {}
        self.performance_targets = {
            'response_time': 2.0,  # seconds
            'memory_usage': 0.8,   # 80%
            'cpu_usage': 0.8,      # 80%
            'cache_hit_rate': 0.8  # 80%
        }
        
        # Start background optimization tasks (only if event loop is running)
        try:
            asyncio.create_task(self._continuous_optimization_loop())
            asyncio.create_task(self._resource_allocation_loop())
        except RuntimeError:
            # No event loop running, will start tasks later
            pass
    
    async def optimize_for_device(self, user_agent: str, client_hints: Dict[str, Any] = None,
                                user_id: str = None) -> Dict[str, Any]:
        """Optimize performance for specific device capabilities."""
        # Detect device capabilities
        device_profile = await self.device_detector.detect_device_capabilities(
            user_agent, client_hints
        )
        
        # Get current system performance
        current_performance = await self._get_current_performance()
        
        # Adapt features for device
        dashboard_features = await self.enhancement_manager.adapt_features_for_device(
            'dashboard', device_profile, current_performance
        )
        
        processing_features = await self.enhancement_manager.adapt_features_for_device(
            'data_processing', device_profile, current_performance
        )
        
        visualization_features = await self.enhancement_manager.adapt_features_for_device(
            'visualization', device_profile, current_performance
        )
        
        # Generate optimization configuration
        optimization_config = {
            'device_profile': {
                'device_id': device_profile.device_id,
                'device_type': device_profile.device_type.value,
                'performance_score': device_profile.performance_score,
                'is_mobile': device_profile.is_mobile
            },
            'enabled_features': {
                'dashboard': dashboard_features,
                'data_processing': processing_features,
                'visualization': visualization_features
            },
            'performance_settings': {
                'chunk_size': self._calculate_optimal_chunk_size(device_profile),
                'cache_size': self._calculate_optimal_cache_size(device_profile),
                'update_frequency': self._calculate_optimal_update_frequency(device_profile),
                'quality_settings': self._calculate_quality_settings(device_profile)
            },
            'resource_limits': {
                'max_memory_usage': device_profile.memory_gb * 0.7,  # 70% of available
                'max_cpu_usage': device_profile.cpu_cores * 0.8,    # 80% of cores
                'network_optimization': device_profile.network_speed < 50  # Optimize for slow networks
            }
        }
        
        logger.info(f"Generated optimization config for device {device_profile.device_id}")
        
        return optimization_config
    
    async def predict_and_allocate_resources(self, prediction_horizon: timedelta = timedelta(minutes=30)) -> Dict[str, Any]:
        """Predict load and dynamically allocate resources."""
        # Get load prediction
        load_prediction = await self.load_predictor.predict_load(prediction_horizon)
        
        # Calculate resource allocation based on prediction
        resource_allocation = {
            'cpu_allocation': {
                'dashboard': self._calculate_cpu_allocation('dashboard', load_prediction),
                'data_processing': self._calculate_cpu_allocation('data_processing', load_prediction),
                'ai_services': self._calculate_cpu_allocation('ai_services', load_prediction),
                'background_tasks': self._calculate_cpu_allocation('background_tasks', load_prediction)
            },
            'memory_allocation': {
                'cache': self._calculate_memory_allocation('cache', load_prediction),
                'processing': self._calculate_memory_allocation('processing', load_prediction),
                'buffers': self._calculate_memory_allocation('buffers', load_prediction)
            },
            'network_allocation': {
                'real_time_updates': self._calculate_network_allocation('real_time_updates', load_prediction),
                'file_transfers': self._calculate_network_allocation('file_transfers', load_prediction),
                'api_calls': self._calculate_network_allocation('api_calls', load_prediction)
            }
        }
        
        # Store allocation for monitoring
        self.resource_allocations[datetime.utcnow().isoformat()] = resource_allocation
        
        # Apply resource allocation
        await self._apply_resource_allocation(resource_allocation)
        
        allocation_result = {
            'prediction': {
                'timestamp': load_prediction.timestamp.isoformat(),
                'predicted_cpu_usage': load_prediction.predicted_cpu_usage,
                'predicted_memory_usage': load_prediction.predicted_memory_usage,
                'predicted_users': load_prediction.predicted_concurrent_users,
                'confidence': load_prediction.confidence_score,
                'factors': load_prediction.factors
            },
            'allocation': resource_allocation,
            'optimization_strategy': self.current_strategy.value
        }
        
        logger.info(f"Allocated resources based on prediction: "
                   f"CPU {load_prediction.predicted_cpu_usage:.1f}%, "
                   f"Memory {load_prediction.predicted_memory_usage:.1f}%, "
                   f"Users {load_prediction.predicted_concurrent_users}")
        
        return allocation_result
    
    async def optimize_caching_strategy(self, user_id: str = None, 
                                      context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Optimize caching with predictive pre-loading."""
        # Get cache statistics
        cache_stats = self.cache_manager.get_cache_stats()
        
        # Analyze cache performance
        cache_optimization = {
            'current_performance': cache_stats,
            'recommendations': [],
            'preload_candidates': [],
            'eviction_candidates': []
        }
        
        # Generate recommendations based on performance
        if cache_stats['hit_rate'] < 0.7:
            cache_optimization['recommendations'].append({
                'type': 'increase_cache_size',
                'reason': 'Low hit rate indicates insufficient cache size',
                'suggested_action': 'Increase cache size by 50%'
            })
        
        if cache_stats['cache_size_mb'] > 500:  # 500MB threshold
            cache_optimization['recommendations'].append({
                'type': 'optimize_eviction',
                'reason': 'Large cache size may impact memory usage',
                'suggested_action': 'Implement more aggressive eviction policy'
            })
        
        # Identify preload candidates based on user patterns
        if user_id:
            preload_candidates = await self._identify_preload_candidates(user_id, context)
            cache_optimization['preload_candidates'] = preload_candidates
        
        logger.info(f"Cache optimization analysis: hit rate {cache_stats['hit_rate']:.2f}, "
                   f"size {cache_stats['cache_size_mb']:.1f}MB")
        
        return cache_optimization
    
    async def _get_current_performance(self) -> Dict[str, float]:
        """Get current system performance metrics."""
        try:
            cpu_usage = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            memory_usage = memory.percent / 100.0
            
            return {
                'cpu_usage': cpu_usage / 100.0,
                'memory_usage': memory_usage,
                'network_speed': 100.0,  # Default network speed
                'response_time': 1.0     # Default response time
            }
        except ImportError:
            # Fallback if psutil not available
            return {
                'cpu_usage': 0.5,
                'memory_usage': 0.6,
                'network_speed': 100.0,
                'response_time': 1.0
            }
    
    def _calculate_optimal_chunk_size(self, device_profile: DeviceProfile) -> int:
        """Calculate optimal chunk size for data processing."""
        base_chunk_size = 1000
        
        # Adjust based on device capabilities
        if device_profile.device_type == DeviceCapability.HIGH_END:
            return base_chunk_size * 10
        elif device_profile.device_type == DeviceCapability.MEDIUM_END:
            return base_chunk_size * 5
        elif device_profile.device_type == DeviceCapability.MOBILE:
            return base_chunk_size // 2
        else:
            return base_chunk_size
    
    def _calculate_optimal_cache_size(self, device_profile: DeviceProfile) -> int:
        """Calculate optimal cache size based on device memory."""
        # Use 10% of available memory for cache
        cache_size_mb = device_profile.memory_gb * 1024 * 0.1
        
        # Convert to number of items (assuming 1KB per item)
        cache_items = int(cache_size_mb * 1024)
        
        return max(100, min(10000, cache_items))  # Clamp between 100 and 10k items
    
    def _calculate_optimal_update_frequency(self, device_profile: DeviceProfile) -> float:
        """Calculate optimal update frequency in seconds."""
        if device_profile.device_type == DeviceCapability.HIGH_END:
            return 1.0  # 1 second updates
        elif device_profile.device_type == DeviceCapability.MEDIUM_END:
            return 5.0  # 5 second updates
        elif device_profile.device_type == DeviceCapability.MOBILE:
            return 30.0  # 30 second updates
        else:
            return 10.0  # 10 second updates
    
    def _calculate_quality_settings(self, device_profile: DeviceProfile) -> Dict[str, Any]:
        """Calculate quality settings based on device capabilities."""
        if device_profile.device_type == DeviceCapability.HIGH_END:
            return {
                'chart_quality': 'high',
                'animation_quality': 'high',
                'image_quality': 'high',
                'enable_effects': True
            }
        elif device_profile.device_type == DeviceCapability.MEDIUM_END:
            return {
                'chart_quality': 'medium',
                'animation_quality': 'medium',
                'image_quality': 'medium',
                'enable_effects': True
            }
        else:
            return {
                'chart_quality': 'low',
                'animation_quality': 'low',
                'image_quality': 'low',
                'enable_effects': False
            }
    
    def _calculate_cpu_allocation(self, service: str, prediction: LoadPrediction) -> float:
        """Calculate CPU allocation percentage for a service."""
        base_allocations = {
            'dashboard': 0.3,
            'data_processing': 0.4,
            'ai_services': 0.2,
            'background_tasks': 0.1
        }
        
        base_allocation = base_allocations.get(service, 0.1)
        
        # Adjust based on predicted load
        load_factor = prediction.predicted_cpu_usage / 100.0
        adjusted_allocation = base_allocation * (1.0 + load_factor)
        
        return min(0.8, adjusted_allocation)  # Cap at 80%
    
    def _calculate_memory_allocation(self, component: str, prediction: LoadPrediction) -> float:
        """Calculate memory allocation percentage for a component."""
        base_allocations = {
            'cache': 0.3,
            'processing': 0.5,
            'buffers': 0.2
        }
        
        base_allocation = base_allocations.get(component, 0.1)
        
        # Adjust based on predicted memory usage
        memory_factor = prediction.predicted_memory_usage / 100.0
        adjusted_allocation = base_allocation * (1.0 + memory_factor)
        
        return min(0.8, adjusted_allocation)  # Cap at 80%
    
    def _calculate_network_allocation(self, service: str, prediction: LoadPrediction) -> float:
        """Calculate network bandwidth allocation for a service."""
        base_allocations = {
            'real_time_updates': 0.4,
            'file_transfers': 0.3,
            'api_calls': 0.3
        }
        
        base_allocation = base_allocations.get(service, 0.1)
        
        # Adjust based on predicted user count
        user_factor = min(prediction.predicted_concurrent_users / 100.0, 2.0)
        adjusted_allocation = base_allocation * (1.0 + user_factor)
        
        return min(0.9, adjusted_allocation)  # Cap at 90%
    
    async def _apply_resource_allocation(self, allocation: Dict[str, Any]):
        """Apply resource allocation to system components."""
        # In a real implementation, this would configure actual resource limits
        # For now, we'll just log the allocation
        logger.info(f"Applied resource allocation: {allocation}")
    
    async def _identify_preload_candidates(self, user_id: str, 
                                         context: Dict[str, Any] = None) -> List[str]:
        """Identify candidates for predictive preloading."""
        candidates = []
        
        # Analyze user patterns to predict next actions
        if user_id in self.cache_manager.user_patterns:
            user_pattern = self.cache_manager.user_patterns[user_id]
            
            # Get recent access sequence
            if 'access_sequence' in user_pattern:
                recent_accesses = list(user_pattern['access_sequence'])[-10:]
                
                # Simple pattern matching: if user accessed A then B frequently,
                # preload B when A is accessed
                for i in range(len(recent_accesses) - 1):
                    current_key = recent_accesses[i]['key']
                    next_key = recent_accesses[i + 1]['key']
                    
                    # Add to candidates if pattern is strong
                    if next_key not in candidates:
                        candidates.append(next_key)
        
        return candidates[:5]  # Limit to 5 candidates
    
    async def _continuous_optimization_loop(self):
        """Continuous optimization loop that runs in the background."""
        while True:
            try:
                # Collect current performance metrics
                current_performance = await self._get_current_performance()
                
                # Record metrics for prediction learning
                metrics = PerformanceMetrics(
                    operation_type="system_monitoring",
                    execution_time=0.0,
                    memory_usage=current_performance['memory_usage'],
                    cpu_usage=current_performance['cpu_usage'],
                    throughput=1.0,
                    error_rate=0.0,
                    timestamp=time.time()
                )
                
                self.load_predictor.record_metrics(metrics)
                
                # Adjust optimization strategy based on performance
                await self._adjust_optimization_strategy(current_performance)
                
                # Sleep before next optimization cycle
                await asyncio.sleep(60)  # Run every minute
                
            except Exception as e:
                logger.error(f"Optimization loop error: {e}")
                await asyncio.sleep(60)
    
    async def _resource_allocation_loop(self):
        """Resource allocation loop that runs periodically."""
        while True:
            try:
                # Predict and allocate resources every 5 minutes
                await self.predict_and_allocate_resources()
                await asyncio.sleep(300)  # 5 minutes
                
            except Exception as e:
                logger.error(f"Resource allocation loop error: {e}")
                await asyncio.sleep(300)
    
    async def _adjust_optimization_strategy(self, performance: Dict[str, float]):
        """Adjust optimization strategy based on current performance."""
        cpu_usage = performance['cpu_usage']
        memory_usage = performance['memory_usage']
        
        # Switch to more aggressive optimization if performance is poor
        if cpu_usage > 0.9 or memory_usage > 0.9:
            if self.current_strategy != OptimizationStrategy.AGGRESSIVE:
                self.current_strategy = OptimizationStrategy.AGGRESSIVE
                logger.info("Switched to aggressive optimization strategy")
        
        # Switch to balanced optimization if performance is moderate
        elif cpu_usage > 0.7 or memory_usage > 0.7:
            if self.current_strategy != OptimizationStrategy.BALANCED:
                self.current_strategy = OptimizationStrategy.BALANCED
                logger.info("Switched to balanced optimization strategy")
        
        # Switch to conservative optimization if performance is good
        else:
            if self.current_strategy != OptimizationStrategy.CONSERVATIVE:
                self.current_strategy = OptimizationStrategy.CONSERVATIVE
                logger.info("Switched to conservative optimization strategy")
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get comprehensive optimization statistics."""
        cache_stats = self.cache_manager.get_cache_stats()
        adaptation_stats = self.enhancement_manager.get_adaptation_stats()
        
        return {
            'current_strategy': self.current_strategy.value,
            'cache_performance': cache_stats,
            'device_adaptations': adaptation_stats,
            'resource_allocations': len(self.resource_allocations),
            'optimization_history': len(self.optimization_history),
            'performance_targets': self.performance_targets
        }


# Global optimizer instance
_intelligent_optimizer = None

def get_intelligent_optimizer() -> IntelligentPerformanceOptimizer:
    """Get global intelligent performance optimizer instance."""
    global _intelligent_optimizer
    
    if _intelligent_optimizer is None:
        _intelligent_optimizer = IntelligentPerformanceOptimizer()
    
    return _intelligent_optimizer


@never_fail_api_endpoint(
    fallback_response={"optimized": False, "message": "Optimization temporarily unavailable"},
    degradation_service="performance_optimization",
    user_action="performance_optimization",
    context_aware=True
)
async def optimize_performance_for_request(user_agent: str, client_hints: Dict[str, Any] = None,
                                         user_id: str = None) -> Dict[str, Any]:
    """API endpoint to optimize performance for a specific request."""
    optimizer = get_intelligent_optimizer()
    
    # Optimize for device
    optimization_config = await optimizer.optimize_for_device(
        user_agent, client_hints, user_id
    )
    
    # Get caching optimization
    cache_optimization = await optimizer.optimize_caching_strategy(user_id)
    
    return {
        "optimized": True,
        "device_optimization": optimization_config,
        "cache_optimization": cache_optimization,
        "timestamp": datetime.utcnow().isoformat()
    }


@never_fail_api_endpoint(
    fallback_response={"predicted": False, "message": "Prediction temporarily unavailable"},
    degradation_service="performance_optimization",
    user_action="load_prediction",
    context_aware=True
)
async def predict_system_load(horizon_minutes: int = 30) -> Dict[str, Any]:
    """API endpoint to predict system load and allocate resources."""
    optimizer = get_intelligent_optimizer()
    
    prediction_horizon = timedelta(minutes=horizon_minutes)
    allocation_result = await optimizer.predict_and_allocate_resources(prediction_horizon)
    
    return {
        "predicted": True,
        "result": allocation_result,
        "timestamp": datetime.utcnow().isoformat()
    }


async def initialize_intelligent_performance_optimization():
    """Initialize the intelligent performance optimization system."""
    logger.info("Initializing intelligent performance optimization system")
    
    # Get optimizer instance (this will create it if it doesn't exist)
    optimizer = get_intelligent_optimizer()
    
    logger.info("Intelligent performance optimization system initialized successfully")
    
    return optimizer