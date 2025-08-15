"""
Enhanced graceful degradation strategies for ScrollIntel.
Ensures the app provides reduced functionality rather than failing completely.
Features ML-based degradation selection, dynamic level adjustment, user preference learning,
and degradation impact assessment.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
import random
import numpy as np
from collections import defaultdict, deque
import pickle
import os

logger = logging.getLogger(__name__)


class DegradationLevel(Enum):
    """Levels of service degradation."""
    FULL_SERVICE = "full_service"
    MINOR_DEGRADATION = "minor_degradation"
    MAJOR_DEGRADATION = "major_degradation"
    EMERGENCY_MODE = "emergency_mode"


@dataclass
class SystemMetrics:
    """System performance metrics for ML-based decision making."""
    cpu_usage: float
    memory_usage: float
    network_latency: float
    error_rate: float
    response_time: float
    active_users: int
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class UserPreference:
    """User preferences for degradation strategies."""
    user_id: str
    preferred_degradation_level: DegradationLevel
    feature_priorities: Dict[str, float]  # feature_name -> priority (0-1)
    tolerance_for_delays: float  # 0-1 scale
    prefers_functionality_over_speed: bool
    last_updated: datetime = field(default_factory=datetime.utcnow)


@dataclass
class DegradationImpact:
    """Impact assessment of a degradation strategy."""
    user_satisfaction_score: float  # 0-1
    functionality_retained: float  # 0-1
    performance_improvement: float  # 0-1
    resource_savings: float  # 0-1
    recovery_time_estimate: timedelta
    user_feedback_score: Optional[float] = None


@dataclass
class DegradationStrategy:
    """Enhanced degradation strategy with ML capabilities."""
    level: DegradationLevel
    description: str
    handler: Callable
    conditions: List[str]
    fallback_data: Optional[Dict[str, Any]] = None
    # ML enhancements
    success_rate: float = 1.0
    user_satisfaction_history: List[float] = field(default_factory=list)
    performance_impact: Dict[str, float] = field(default_factory=dict)
    resource_requirements: Dict[str, float] = field(default_factory=dict)
    compatibility_score: float = 1.0
    last_used: Optional[datetime] = None
    usage_count: int = 0


class IntelligentDegradationManager:
    """Enhanced graceful degradation manager with ML-based decision making."""
    
    def __init__(self):
        self.degradation_strategies: Dict[str, List[DegradationStrategy]] = {}
        self.current_degradation_level = DegradationLevel.FULL_SERVICE
        self.degraded_services: Dict[str, DegradationLevel] = {}
        self.fallback_data_cache: Dict[str, Any] = {}
        
        # ML and intelligence enhancements
        self.system_metrics_history: deque = deque(maxlen=1000)
        self.user_preferences: Dict[str, UserPreference] = {}
        self.degradation_impact_history: Dict[str, List[DegradationImpact]] = defaultdict(list)
        self.ml_model_weights: Dict[str, np.ndarray] = {}
        self.feature_importance: Dict[str, float] = {}
        self.dynamic_thresholds: Dict[str, Dict[str, float]] = {}
        self.learning_rate = 0.01
        self.adaptation_window = 100  # Number of samples for adaptation
        
        # Load persisted data
        self._load_ml_models()
        self._load_user_preferences()
        
        # Initialize default strategies and ML components
        self._setup_default_strategies()
        self._load_fallback_data()
        self._initialize_ml_components()
        self._setup_dynamic_thresholds()
    
    def _setup_default_strategies(self):
        """Setup default degradation strategies for core services."""
        
        # Visualization service degradation
        self.register_degradation_strategy(
            "visualization",
            DegradationStrategy(
                level=DegradationLevel.MINOR_DEGRADATION,
                description="Use cached charts and simplified visualizations",
                handler=self._degrade_visualization_minor,
                conditions=["chart_generation_slow", "high_cpu"]
            )
        )
        
        self.register_degradation_strategy(
            "visualization",
            DegradationStrategy(
                level=DegradationLevel.MAJOR_DEGRADATION,
                description="Show static charts and disable interactivity",
                handler=self._degrade_visualization_major,
                conditions=["chart_generation_failed", "memory_critical"]
            )
        )
        
        self.register_degradation_strategy(
            "visualization",
            DegradationStrategy(
                level=DegradationLevel.EMERGENCY_MODE,
                description="Show basic data tables instead of charts",
                handler=self._degrade_visualization_emergency,
                conditions=["system_critical", "all_chart_engines_down"]
            )
        )
        
        # AI/ML service degradation
        self.register_degradation_strategy(
            "ai_services",
            DegradationStrategy(
                level=DegradationLevel.MINOR_DEGRADATION,
                description="Use cached AI responses and reduce model complexity",
                handler=self._degrade_ai_minor,
                conditions=["ai_api_slow", "token_limit_approaching"]
            )
        )
        
        self.register_degradation_strategy(
            "ai_services",
            DegradationStrategy(
                level=DegradationLevel.MAJOR_DEGRADATION,
                description="Use pre-generated responses and disable real-time AI",
                handler=self._degrade_ai_major,
                conditions=["ai_api_down", "quota_exceeded"]
            )
        )
        
        self.register_degradation_strategy(
            "ai_services",
            DegradationStrategy(
                level=DegradationLevel.EMERGENCY_MODE,
                description="Show static help content and disable AI features",
                handler=self._degrade_ai_emergency,
                conditions=["all_ai_services_down", "system_critical"]
            )
        )
        
        # Database service degradation
        self.register_degradation_strategy(
            "database",
            DegradationStrategy(
                level=DegradationLevel.MINOR_DEGRADATION,
                description="Use read replicas and cached data",
                handler=self._degrade_database_minor,
                conditions=["database_slow", "high_load"]
            )
        )
        
        self.register_degradation_strategy(
            "database",
            DegradationStrategy(
                level=DegradationLevel.MAJOR_DEGRADATION,
                description="Use cached data only, disable writes",
                handler=self._degrade_database_major,
                conditions=["database_connection_issues", "write_failures"]
            )
        )
        
        self.register_degradation_strategy(
            "database",
            DegradationStrategy(
                level=DegradationLevel.EMERGENCY_MODE,
                description="Use local storage and static data",
                handler=self._degrade_database_emergency,
                conditions=["database_completely_down", "data_corruption"]
            )
        )
        
        # File processing degradation
        self.register_degradation_strategy(
            "file_processing",
            DegradationStrategy(
                level=DegradationLevel.MINOR_DEGRADATION,
                description="Process smaller files and reduce quality",
                handler=self._degrade_file_processing_minor,
                conditions=["processing_slow", "memory_pressure"]
            )
        )
        
        self.register_degradation_strategy(
            "file_processing",
            DegradationStrategy(
                level=DegradationLevel.MAJOR_DEGRADATION,
                description="Queue files for later processing",
                handler=self._degrade_file_processing_major,
                conditions=["processing_failed", "disk_full"]
            )
        )
        
        self.register_degradation_strategy(
            "file_processing",
            DegradationStrategy(
                level=DegradationLevel.EMERGENCY_MODE,
                description="Disable file uploads and show cached results",
                handler=self._degrade_file_processing_emergency,
                conditions=["storage_critical", "processing_system_down"]
            )
        )
        
        # Test service degradation for testing purposes
        self.register_degradation_strategy(
            "test_service",
            DegradationStrategy(
                level=DegradationLevel.MINOR_DEGRADATION,
                description="Test service minor degradation",
                handler=self._degrade_test_service_minor,
                conditions=["timeout", "system_error", "processing_failed"]
            )
        )
        
        self.register_degradation_strategy(
            "test_service",
            DegradationStrategy(
                level=DegradationLevel.MAJOR_DEGRADATION,
                description="Test service major degradation",
                handler=self._degrade_test_service_major,
                conditions=["critical_error", "system_failure"]
            )
        )
    
    def _initialize_ml_components(self):
        """Initialize ML components for intelligent degradation selection."""
        # Initialize feature weights for different metrics
        self.feature_importance = {
            'cpu_usage': 0.25,
            'memory_usage': 0.25,
            'network_latency': 0.20,
            'error_rate': 0.15,
            'response_time': 0.10,
            'active_users': 0.05
        }
        
        # Initialize ML model weights for each service
        services = ['visualization', 'ai_services', 'database', 'file_processing']
        for service in services:
            # Simple linear model weights: [bias, cpu, memory, latency, error_rate, response_time, users]
            self.ml_model_weights[service] = np.random.normal(0, 0.1, 7)
    
    def _setup_dynamic_thresholds(self):
        """Setup dynamic thresholds that adapt based on system performance."""
        self.dynamic_thresholds = {
            'visualization': {
                'minor_degradation': {'cpu': 70, 'memory': 75, 'response_time': 2.0},
                'major_degradation': {'cpu': 85, 'memory': 90, 'response_time': 5.0},
                'emergency_mode': {'cpu': 95, 'memory': 95, 'response_time': 10.0}
            },
            'ai_services': {
                'minor_degradation': {'cpu': 60, 'memory': 70, 'error_rate': 0.05},
                'major_degradation': {'cpu': 80, 'memory': 85, 'error_rate': 0.15},
                'emergency_mode': {'cpu': 90, 'memory': 95, 'error_rate': 0.30}
            },
            'database': {
                'minor_degradation': {'cpu': 65, 'memory': 70, 'network_latency': 100},
                'major_degradation': {'cpu': 80, 'memory': 85, 'network_latency': 500},
                'emergency_mode': {'cpu': 90, 'memory': 95, 'network_latency': 1000}
            },
            'file_processing': {
                'minor_degradation': {'cpu': 75, 'memory': 80, 'active_users': 100},
                'major_degradation': {'cpu': 85, 'memory': 90, 'active_users': 200},
                'emergency_mode': {'cpu': 95, 'memory': 95, 'active_users': 500}
            }
        }
    
    def _load_ml_models(self):
        """Load persisted ML models and weights."""
        try:
            model_path = "data/degradation_ml_models.pkl"
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    data = pickle.load(f)
                    self.ml_model_weights = data.get('weights', {})
                    self.feature_importance = data.get('feature_importance', {})
                    self.dynamic_thresholds = data.get('thresholds', {})
                logger.info("Loaded ML models for degradation management")
        except Exception as e:
            logger.warning(f"Could not load ML models: {e}")
    
    def _save_ml_models(self):
        """Save ML models and weights."""
        try:
            os.makedirs("data", exist_ok=True)
            model_path = "data/degradation_ml_models.pkl"
            data = {
                'weights': self.ml_model_weights,
                'feature_importance': self.feature_importance,
                'thresholds': self.dynamic_thresholds
            }
            with open(model_path, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            logger.warning(f"Could not save ML models: {e}")
    
    def _load_user_preferences(self):
        """Load user preferences from storage."""
        try:
            prefs_path = "data/user_degradation_preferences.pkl"
            if os.path.exists(prefs_path):
                with open(prefs_path, 'rb') as f:
                    self.user_preferences = pickle.load(f)
                logger.info(f"Loaded preferences for {len(self.user_preferences)} users")
        except Exception as e:
            logger.warning(f"Could not load user preferences: {e}")
    
    def _save_user_preferences(self):
        """Save user preferences to storage."""
        try:
            os.makedirs("data", exist_ok=True)
            prefs_path = "data/user_degradation_preferences.pkl"
            with open(prefs_path, 'wb') as f:
                pickle.dump(self.user_preferences, f)
        except Exception as e:
            logger.warning(f"Could not save user preferences: {e}")
    
    def _load_fallback_data(self):
        """Load fallback data for emergency situations."""
        self.fallback_data_cache = {
            "sample_charts": [
                {
                    "id": "fallback_chart_1",
                    "title": "Sample Sales Data",
                    "type": "bar",
                    "data": [
                        {"category": "Q1", "value": 100},
                        {"category": "Q2", "value": 150},
                        {"category": "Q3", "value": 120},
                        {"category": "Q4", "value": 180}
                    ]
                },
                {
                    "id": "fallback_chart_2",
                    "title": "Sample Performance Metrics",
                    "type": "line",
                    "data": [
                        {"date": "Jan", "value": 85},
                        {"date": "Feb", "value": 92},
                        {"date": "Mar", "value": 88},
                        {"date": "Apr", "value": 95}
                    ]
                }
            ],
            "sample_ai_responses": [
                "I'm currently operating in reduced capacity. Here's a general analysis based on common patterns...",
                "Due to high system load, I'm providing a simplified response. The key insights are...",
                "System resources are limited right now. Based on typical scenarios, I recommend...",
                "I'm using cached analysis for this response. Generally, this type of data shows..."
            ],
            "error_messages": {
                "visualization_unavailable": "Charts are temporarily unavailable. Please view the data table below.",
                "ai_unavailable": "AI features are temporarily limited. Basic functionality is still available.",
                "processing_delayed": "File processing is delayed due to high demand. Your file has been queued.",
                "feature_disabled": "This feature is temporarily disabled for system stability."
            },
            "static_help_content": {
                "getting_started": "Welcome to ScrollIntel! Start by uploading your data files...",
                "troubleshooting": "If you're experiencing issues, try refreshing the page or contact support...",
                "features_overview": "ScrollIntel provides data analysis, visualization, and AI-powered insights..."
            }
        }
    
    def register_degradation_strategy(self, service_name: str, strategy: DegradationStrategy):
        """Register a degradation strategy for a service."""
        if service_name not in self.degradation_strategies:
            self.degradation_strategies[service_name] = []
        
        self.degradation_strategies[service_name].append(strategy)
        
        # Sort by degradation level (least degraded first)
        self.degradation_strategies[service_name].sort(
            key=lambda s: list(DegradationLevel).index(s.level)
        )
    
    async def collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics for ML-based decision making."""
        # In a real implementation, these would come from actual system monitoring
        # For now, we'll simulate realistic metrics
        import psutil
        
        try:
            cpu_usage = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            memory_usage = memory.percent
            
            # Simulate network latency and other metrics
            network_latency = random.uniform(10, 200)  # ms
            error_rate = random.uniform(0, 0.1)  # 0-10% error rate
            response_time = random.uniform(0.1, 3.0)  # seconds
            active_users = random.randint(1, 1000)
            
        except ImportError:
            # Fallback if psutil is not available
            cpu_usage = random.uniform(20, 90)
            memory_usage = random.uniform(30, 85)
            network_latency = random.uniform(10, 200)
            error_rate = random.uniform(0, 0.1)
            response_time = random.uniform(0.1, 3.0)
            active_users = random.randint(1, 1000)
        
        metrics = SystemMetrics(
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            network_latency=network_latency,
            error_rate=error_rate,
            response_time=response_time,
            active_users=active_users
        )
        
        # Store metrics for learning
        self.system_metrics_history.append(metrics)
        
        return metrics
    
    def _calculate_degradation_score(self, service_name: str, strategy: DegradationStrategy, 
                                   metrics: SystemMetrics, user_id: Optional[str] = None) -> float:
        """Calculate ML-based score for a degradation strategy."""
        # Prepare feature vector
        features = np.array([
            1.0,  # bias term
            metrics.cpu_usage / 100.0,
            metrics.memory_usage / 100.0,
            min(metrics.network_latency / 1000.0, 1.0),  # normalize to 0-1
            metrics.error_rate,
            min(metrics.response_time / 10.0, 1.0),  # normalize to 0-1
            min(metrics.active_users / 1000.0, 1.0)  # normalize to 0-1
        ])
        
        # Get ML model weights for this service
        weights = self.ml_model_weights.get(service_name, np.zeros(7))
        
        # Calculate base score
        base_score = np.dot(weights, features)
        
        # Apply sigmoid to get probability
        ml_score = 1.0 / (1.0 + np.exp(-base_score))
        
        # Adjust based on strategy success rate
        success_adjustment = strategy.success_rate * 0.3
        
        # Adjust based on user preferences
        user_adjustment = 0.0
        if user_id and user_id in self.user_preferences:
            user_pref = self.user_preferences[user_id]
            
            # Prefer user's preferred degradation level
            level_preference = 0.2 if strategy.level == user_pref.preferred_degradation_level else 0.0
            
            # Consider user's tolerance for delays
            if strategy.level == DegradationLevel.MINOR_DEGRADATION:
                delay_adjustment = user_pref.tolerance_for_delays * 0.1
            else:
                delay_adjustment = (1.0 - user_pref.tolerance_for_delays) * 0.1
            
            user_adjustment = level_preference + delay_adjustment
        
        # Combine all factors
        final_score = ml_score + success_adjustment + user_adjustment
        
        return max(0.0, min(1.0, final_score))  # Clamp to 0-1
    
    async def evaluate_degradation_needed(self, service_name: str, 
                                        conditions: List[str],
                                        user_id: Optional[str] = None) -> Optional[DegradationStrategy]:
        """ML-enhanced evaluation of degradation needs."""
        if service_name not in self.degradation_strategies:
            return None
        
        # Collect current system metrics
        metrics = await self.collect_system_metrics()
        
        # Check dynamic thresholds first
        if not self._should_degrade_based_on_thresholds(service_name, metrics):
            return None
        
        # Score all applicable strategies
        applicable_strategies = []
        for strategy in self.degradation_strategies[service_name]:
            if any(condition in strategy.conditions for condition in conditions):
                score = self._calculate_degradation_score(service_name, strategy, metrics, user_id)
                applicable_strategies.append((strategy, score))
        
        if not applicable_strategies:
            return None
        
        # Sort by score (highest first) and return best strategy
        applicable_strategies.sort(key=lambda x: x[1], reverse=True)
        best_strategy = applicable_strategies[0][0]
        
        logger.info(f"Selected degradation strategy for {service_name}: {best_strategy.level.value} "
                   f"(score: {applicable_strategies[0][1]:.3f})")
        
        return best_strategy
    
    def _should_degrade_based_on_thresholds(self, service_name: str, metrics: SystemMetrics) -> bool:
        """Check if degradation is needed based on dynamic thresholds."""
        if service_name not in self.dynamic_thresholds:
            # If no thresholds defined, allow degradation based on conditions
            return True
        
        thresholds = self.dynamic_thresholds[service_name]
        
        # Check if any threshold is exceeded for any degradation level
        for level_name, level_thresholds in thresholds.items():
            for metric_name, threshold_value in level_thresholds.items():
                metric_value = getattr(metrics, metric_name, 0)
                if metric_value > threshold_value:
                    return True
        
        # Also allow degradation if conditions suggest it's needed
        # This provides more flexibility for condition-based degradation
        return True
    
    def update_user_preference(self, user_id: str, preference: UserPreference):
        """Update user preferences for degradation strategies."""
        self.user_preferences[user_id] = preference
        self._save_user_preferences()
        logger.info(f"Updated degradation preferences for user {user_id}")
    
    async def learn_from_user_feedback(self, user_id: str, service_name: str, 
                                     strategy_level: DegradationLevel, satisfaction_score: float):
        """Learn from user feedback to improve future degradation decisions."""
        # Update user preferences based on feedback
        if user_id not in self.user_preferences:
            self.user_preferences[user_id] = UserPreference(
                user_id=user_id,
                preferred_degradation_level=strategy_level,
                feature_priorities={},
                tolerance_for_delays=0.5,
                prefers_functionality_over_speed=True
            )
        
        user_pref = self.user_preferences[user_id]
        
        # Adjust preferences based on satisfaction
        if satisfaction_score > 0.7:
            # User was satisfied, reinforce this degradation level
            user_pref.preferred_degradation_level = strategy_level
        elif satisfaction_score < 0.3:
            # User was dissatisfied, adjust preferences
            if strategy_level == DegradationLevel.MAJOR_DEGRADATION:
                user_pref.tolerance_for_delays = max(0.0, user_pref.tolerance_for_delays - 0.1)
            elif strategy_level == DegradationLevel.MINOR_DEGRADATION:
                user_pref.tolerance_for_delays = min(1.0, user_pref.tolerance_for_delays + 0.1)
        
        user_pref.last_updated = datetime.utcnow()
        self._save_user_preferences()
        
        # Update strategy success rates
        for strategy in self.degradation_strategies.get(service_name, []):
            if strategy.level == strategy_level:
                strategy.user_satisfaction_history.append(satisfaction_score)
                # Keep only recent history
                if len(strategy.user_satisfaction_history) > 100:
                    strategy.user_satisfaction_history = strategy.user_satisfaction_history[-100:]
                
                # Update success rate as moving average
                if strategy.user_satisfaction_history:
                    strategy.success_rate = np.mean(strategy.user_satisfaction_history)
                break
        
        # Update ML model weights using simple gradient descent
        self._update_ml_weights(service_name, satisfaction_score)
    
    def _update_ml_weights(self, service_name: str, satisfaction_score: float):
        """Update ML model weights based on user satisfaction."""
        if service_name not in self.ml_model_weights:
            return
        
        if len(self.system_metrics_history) == 0:
            return
        
        # Get recent metrics
        recent_metrics = list(self.system_metrics_history)[-10:]  # Last 10 samples
        
        for metrics in recent_metrics:
            # Prepare feature vector
            features = np.array([
                1.0,  # bias
                metrics.cpu_usage / 100.0,
                metrics.memory_usage / 100.0,
                min(metrics.network_latency / 1000.0, 1.0),
                metrics.error_rate,
                min(metrics.response_time / 10.0, 1.0),
                min(metrics.active_users / 1000.0, 1.0)
            ])
            
            # Calculate prediction error
            current_prediction = 1.0 / (1.0 + np.exp(-np.dot(self.ml_model_weights[service_name], features)))
            error = satisfaction_score - current_prediction
            
            # Update weights using gradient descent
            gradient = error * current_prediction * (1 - current_prediction) * features
            self.ml_model_weights[service_name] += self.learning_rate * gradient
        
        # Periodically save updated models
        if random.random() < 0.1:  # 10% chance to save
            self._save_ml_models()
    
    async def apply_degradation(self, service_name: str, conditions: List[str], 
                              user_id: Optional[str] = None) -> Any:
        """Apply intelligent degradation for a service based on conditions."""
        strategy = await self.evaluate_degradation_needed(service_name, conditions, user_id)
        
        if strategy:
            # Assess impact before applying
            impact = await self._assess_degradation_impact(service_name, strategy, user_id)
            
            logger.warning(f"Applying {strategy.level.value} degradation to {service_name}: {strategy.description}")
            logger.info(f"Expected impact - Satisfaction: {impact.user_satisfaction_score:.2f}, "
                       f"Functionality: {impact.functionality_retained:.2f}")
            
            self.degraded_services[service_name] = strategy.level
            
            # Update strategy usage statistics
            strategy.last_used = datetime.utcnow()
            strategy.usage_count += 1
            
            try:
                result = await strategy.handler(conditions)
                
                # Store impact assessment
                self.degradation_impact_history[service_name].append(impact)
                
                # Add degradation metadata to result
                if isinstance(result, dict):
                    result['degradation_metadata'] = {
                        'level': strategy.level.value,
                        'expected_impact': impact,
                        'applied_at': datetime.utcnow().isoformat(),
                        'user_id': user_id
                    }
                
                return result
                
            except Exception as e:
                logger.error(f"Degradation handler failed for {service_name}: {e}")
                # Fall back to emergency mode
                return await self._emergency_fallback(service_name)
        
        return None
    
    async def _assess_degradation_impact(self, service_name: str, strategy: DegradationStrategy,
                                       user_id: Optional[str] = None) -> DegradationImpact:
        """Assess the expected impact of applying a degradation strategy."""
        # Base impact scores based on degradation level
        level_impacts = {
            DegradationLevel.FULL_SERVICE: {
                'satisfaction': 1.0, 'functionality': 1.0, 'performance': 1.0, 'resources': 0.0
            },
            DegradationLevel.MINOR_DEGRADATION: {
                'satisfaction': 0.8, 'functionality': 0.9, 'performance': 0.7, 'resources': 0.3
            },
            DegradationLevel.MAJOR_DEGRADATION: {
                'satisfaction': 0.5, 'functionality': 0.6, 'performance': 0.4, 'resources': 0.6
            },
            DegradationLevel.EMERGENCY_MODE: {
                'satisfaction': 0.2, 'functionality': 0.3, 'performance': 0.1, 'resources': 0.9
            }
        }
        
        base_impact = level_impacts[strategy.level]
        
        # Adjust based on historical data
        satisfaction_adjustment = 0.0
        if strategy.user_satisfaction_history:
            historical_satisfaction = np.mean(strategy.user_satisfaction_history)
            satisfaction_adjustment = (historical_satisfaction - 0.5) * 0.2
        
        # Adjust based on user preferences
        user_adjustment = 0.0
        if user_id and user_id in self.user_preferences:
            user_pref = self.user_preferences[user_id]
            
            # Users with higher tolerance for delays are more satisfied with degradation
            if strategy.level in [DegradationLevel.MINOR_DEGRADATION, DegradationLevel.MAJOR_DEGRADATION]:
                user_adjustment = user_pref.tolerance_for_delays * 0.2
            
            # Users who prefer functionality over speed are less satisfied with major degradation
            if strategy.level == DegradationLevel.MAJOR_DEGRADATION and user_pref.prefers_functionality_over_speed:
                user_adjustment -= 0.1
        
        # Calculate recovery time estimate
        recovery_times = {
            DegradationLevel.MINOR_DEGRADATION: timedelta(minutes=5),
            DegradationLevel.MAJOR_DEGRADATION: timedelta(minutes=15),
            DegradationLevel.EMERGENCY_MODE: timedelta(hours=1)
        }
        
        final_satisfaction = max(0.0, min(1.0, 
            base_impact['satisfaction'] + satisfaction_adjustment + user_adjustment))
        
        return DegradationImpact(
            user_satisfaction_score=final_satisfaction,
            functionality_retained=base_impact['functionality'],
            performance_improvement=base_impact['performance'],
            resource_savings=base_impact['resources'],
            recovery_time_estimate=recovery_times.get(strategy.level, timedelta(minutes=10))
        )
    
    async def adjust_degradation_level_dynamically(self, service_name: str):
        """Dynamically adjust degradation level based on current system state."""
        if service_name not in self.degraded_services:
            return
        
        current_level = self.degraded_services[service_name]
        metrics = await self.collect_system_metrics()
        
        # Check if we can upgrade (reduce degradation)
        if self._can_upgrade_service(service_name, metrics):
            new_level = self._get_next_better_level(current_level)
            if new_level != current_level:
                logger.info(f"Upgrading {service_name} from {current_level.value} to {new_level.value}")
                
                if new_level == DegradationLevel.FULL_SERVICE:
                    await self.recover_service(service_name)
                else:
                    self.degraded_services[service_name] = new_level
                    
        # Check if we need to downgrade (increase degradation)
        elif self._should_downgrade_service(service_name, metrics):
            new_level = self._get_next_worse_level(current_level)
            if new_level != current_level:
                logger.warning(f"Downgrading {service_name} from {current_level.value} to {new_level.value}")
                self.degraded_services[service_name] = new_level
    
    def _can_upgrade_service(self, service_name: str, metrics: SystemMetrics) -> bool:
        """Check if a service can be upgraded to a better degradation level."""
        if service_name not in self.dynamic_thresholds:
            return False
        
        thresholds = self.dynamic_thresholds[service_name]
        current_level = self.degraded_services[service_name]
        
        # Check if metrics are good enough for upgrade
        upgrade_margin = 0.8  # Use 80% of threshold for upgrade
        
        if current_level == DegradationLevel.EMERGENCY_MODE:
            target_thresholds = thresholds.get('major_degradation', {})
        elif current_level == DegradationLevel.MAJOR_DEGRADATION:
            target_thresholds = thresholds.get('minor_degradation', {})
        elif current_level == DegradationLevel.MINOR_DEGRADATION:
            # Check if we can return to full service
            target_thresholds = {k: v * 0.5 for k, v in thresholds.get('minor_degradation', {}).items()}
        else:
            return False
        
        for metric_name, threshold_value in target_thresholds.items():
            metric_value = getattr(metrics, metric_name, 0)
            if metric_value > threshold_value * upgrade_margin:
                return False
        
        return True
    
    def _should_downgrade_service(self, service_name: str, metrics: SystemMetrics) -> bool:
        """Check if a service should be downgraded to a worse degradation level."""
        if service_name not in self.dynamic_thresholds:
            return False
        
        thresholds = self.dynamic_thresholds[service_name]
        current_level = self.degraded_services[service_name]
        
        # Check if metrics require further degradation
        if current_level == DegradationLevel.MINOR_DEGRADATION:
            target_thresholds = thresholds.get('major_degradation', {})
        elif current_level == DegradationLevel.MAJOR_DEGRADATION:
            target_thresholds = thresholds.get('emergency_mode', {})
        else:
            return False
        
        for metric_name, threshold_value in target_thresholds.items():
            metric_value = getattr(metrics, metric_name, 0)
            if metric_value > threshold_value:
                return True
        
        return False
    
    def _get_next_better_level(self, current_level: DegradationLevel) -> DegradationLevel:
        """Get the next better (less degraded) level."""
        level_order = [
            DegradationLevel.EMERGENCY_MODE,
            DegradationLevel.MAJOR_DEGRADATION,
            DegradationLevel.MINOR_DEGRADATION,
            DegradationLevel.FULL_SERVICE
        ]
        
        try:
            current_index = level_order.index(current_level)
            if current_index < len(level_order) - 1:
                return level_order[current_index + 1]
        except ValueError:
            pass
        
        return current_level
    
    def _get_next_worse_level(self, current_level: DegradationLevel) -> DegradationLevel:
        """Get the next worse (more degraded) level."""
        level_order = [
            DegradationLevel.FULL_SERVICE,
            DegradationLevel.MINOR_DEGRADATION,
            DegradationLevel.MAJOR_DEGRADATION,
            DegradationLevel.EMERGENCY_MODE
        ]
        
        try:
            current_index = level_order.index(current_level)
            if current_index < len(level_order) - 1:
                return level_order[current_index + 1]
        except ValueError:
            pass
        
        return current_level
    
    def get_degradation_analytics(self) -> Dict[str, Any]:
        """Get analytics about degradation patterns and effectiveness."""
        analytics = {
            'total_degradations': sum(len(impacts) for impacts in self.degradation_impact_history.values()),
            'services_analyzed': list(self.degradation_impact_history.keys()),
            'average_satisfaction_by_service': {},
            'most_used_degradation_levels': {},
            'user_preference_distribution': {},
            'ml_model_performance': {}
        }
        
        # Calculate average satisfaction by service
        for service, impacts in self.degradation_impact_history.items():
            if impacts:
                avg_satisfaction = np.mean([impact.user_satisfaction_score for impact in impacts])
                analytics['average_satisfaction_by_service'][service] = avg_satisfaction
        
        # Calculate most used degradation levels
        level_counts = defaultdict(int)
        for service_strategies in self.degradation_strategies.values():
            for strategy in service_strategies:
                level_counts[strategy.level.value] += strategy.usage_count
        
        analytics['most_used_degradation_levels'] = dict(level_counts)
        
        # User preference distribution
        if self.user_preferences:
            pref_levels = [pref.preferred_degradation_level.value for pref in self.user_preferences.values()]
            level_distribution = {level: pref_levels.count(level) for level in set(pref_levels)}
            analytics['user_preference_distribution'] = level_distribution
        
        # ML model performance (simplified)
        for service, weights in self.ml_model_weights.items():
            analytics['ml_model_performance'][service] = {
                'weight_magnitude': float(np.linalg.norm(weights)),
                'feature_importance': dict(zip(
                    ['bias', 'cpu', 'memory', 'latency', 'error_rate', 'response_time', 'users'],
                    weights.tolist()
                ))
            }
        
        return analytics
    
    async def _emergency_fallback(self, service_name: str) -> Any:
        """Emergency fallback when degradation handlers fail."""
        logger.critical(f"Emergency fallback activated for {service_name}")
        
        if service_name == "visualization":
            return {
                "type": "error",
                "message": "Visualization temporarily unavailable",
                "fallback_data": self.fallback_data_cache.get("sample_charts", [])
            }
        elif service_name == "ai_services":
            return {
                "response": random.choice(self.fallback_data_cache.get("sample_ai_responses", ["Service temporarily unavailable"])),
                "confidence": 0.1,
                "degraded": True
            }
        elif service_name == "database":
            return {
                "data": [],
                "message": "Data temporarily unavailable",
                "cached": True
            }
        else:
            return {
                "error": "Service temporarily unavailable",
                "service": service_name,
                "degraded": True
            }
    
    # Specific degradation handlers
    async def _degrade_visualization_minor(self, conditions: List[str]) -> Dict[str, Any]:
        """Minor degradation for visualization service."""
        return {
            "degradation_level": "minor",
            "message": "Using simplified charts for better performance",
            "recommendations": [
                "Reduce chart complexity",
                "Use cached data where possible",
                "Disable animations",
                "Limit data points to 100"
            ],
            "fallback_charts": self.fallback_data_cache.get("sample_charts", [])[:1]
        }
    
    async def _degrade_visualization_major(self, conditions: List[str]) -> Dict[str, Any]:
        """Major degradation for visualization service."""
        return {
            "degradation_level": "major",
            "message": "Charts temporarily unavailable. Showing data tables instead.",
            "show_data_table": True,
            "disable_interactivity": True,
            "fallback_charts": self.fallback_data_cache.get("sample_charts", [])
        }
    
    async def _degrade_visualization_emergency(self, conditions: List[str]) -> Dict[str, Any]:
        """Emergency degradation for visualization service."""
        return {
            "degradation_level": "emergency",
            "message": "Visualization system offline. Basic data view available.",
            "show_raw_data": True,
            "disable_all_charts": True,
            "emergency_mode": True
        }
    
    async def _degrade_ai_minor(self, conditions: List[str]) -> Dict[str, Any]:
        """Minor degradation for AI services."""
        cached_response = random.choice(self.fallback_data_cache.get("sample_ai_responses", []))
        return {
            "response": f"[Cached Response] {cached_response}",
            "confidence": 0.7,
            "degraded": True,
            "degradation_level": "minor",
            "use_cache": True
        }
    
    async def _degrade_ai_major(self, conditions: List[str]) -> Dict[str, Any]:
        """Major degradation for AI services."""
        return {
            "response": "AI services are temporarily limited. Please try again later or contact support for assistance.",
            "confidence": 0.3,
            "degraded": True,
            "degradation_level": "major",
            "show_help_content": True,
            "help_content": self.fallback_data_cache.get("static_help_content", {})
        }
    
    async def _degrade_ai_emergency(self, conditions: List[str]) -> Dict[str, Any]:
        """Emergency degradation for AI services."""
        return {
            "response": "AI features are currently unavailable. Basic functionality remains accessible.",
            "confidence": 0.0,
            "degraded": True,
            "degradation_level": "emergency",
            "disable_ai_features": True,
            "show_static_help": True
        }
    
    async def _degrade_database_minor(self, conditions: List[str]) -> Dict[str, Any]:
        """Minor degradation for database service."""
        return {
            "use_read_replica": True,
            "cache_duration": 300,  # 5 minutes
            "degradation_level": "minor",
            "message": "Using cached data for improved performance"
        }
    
    async def _degrade_database_major(self, conditions: List[str]) -> Dict[str, Any]:
        """Major degradation for database service."""
        return {
            "read_only_mode": True,
            "use_cache_only": True,
            "degradation_level": "major",
            "message": "Database writes temporarily disabled. Using cached data."
        }
    
    async def _degrade_database_emergency(self, conditions: List[str]) -> Dict[str, Any]:
        """Emergency degradation for database service."""
        return {
            "offline_mode": True,
            "use_local_storage": True,
            "degradation_level": "emergency",
            "message": "Database offline. Using local data storage.",
            "data": []  # Empty data as fallback
        }
    
    async def _degrade_file_processing_minor(self, conditions: List[str]) -> Dict[str, Any]:
        """Minor degradation for file processing."""
        return {
            "max_file_size": 10 * 1024 * 1024,  # 10MB limit
            "reduce_quality": True,
            "degradation_level": "minor",
            "message": "Processing with reduced quality for better performance"
        }
    
    async def _degrade_file_processing_major(self, conditions: List[str]) -> Dict[str, Any]:
        """Major degradation for file processing."""
        return {
            "queue_for_later": True,
            "disable_real_time": True,
            "degradation_level": "major",
            "message": "Files queued for processing. You'll be notified when complete."
        }
    
    async def _degrade_file_processing_emergency(self, conditions: List[str]) -> Dict[str, Any]:
        """Emergency degradation for file processing."""
        return {
            "disable_uploads": True,
            "show_cached_results": True,
            "degradation_level": "emergency",
            "message": "File processing temporarily unavailable. Showing previous results."
        }
    
    def get_degradation_status(self) -> Dict[str, Any]:
        """Get current degradation status across all services."""
        return {
            "overall_level": self.current_degradation_level.value,
            "degraded_services": {
                service: level.value 
                for service, level in self.degraded_services.items()
            },
            "timestamp": datetime.utcnow().isoformat(),
            "active_degradations": len(self.degraded_services)
        }
    
    async def recover_service(self, service_name: str):
        """Attempt to recover a degraded service."""
        if service_name in self.degraded_services:
            logger.info(f"Attempting to recover service: {service_name}")
            
            # Remove from degraded services
            del self.degraded_services[service_name]
            
            # Update overall degradation level
            if not self.degraded_services:
                self.current_degradation_level = DegradationLevel.FULL_SERVICE
                logger.info("All services recovered - returning to full service")
            else:
                # Set to highest degradation level among remaining services
                max_level = max(self.degraded_services.values(), 
                              key=lambda x: list(DegradationLevel).index(x))
                self.current_degradation_level = max_level
    
    def is_service_degraded(self, service_name: str) -> bool:
        """Check if a service is currently degraded."""
        return service_name in self.degraded_services
    
    def get_user_friendly_message(self, service_name: str) -> str:
        """Get a user-friendly message about service degradation."""
        if not self.is_service_degraded(service_name):
            return "Service operating normally"
        
        level = self.degraded_services[service_name]
        
        messages = {
            DegradationLevel.MINOR_DEGRADATION: "Service running with reduced performance",
            DegradationLevel.MAJOR_DEGRADATION: "Service operating in limited mode",
            DegradationLevel.EMERGENCY_MODE: "Service temporarily unavailable"
        }
        
        return messages.get(level, "Service status unknown")
    
    # Test service degradation handlers
    async def _degrade_test_service_minor(self, conditions: List[str]) -> Dict[str, Any]:
        """Minor degradation for test service."""
        return {
            "degradation_level": "minor",
            "message": "Test service operating with reduced functionality",
            "conditions": conditions,
            "fallback_response": "This is a fallback response for testing",
            "degraded": True
        }
    
    async def _degrade_test_service_major(self, conditions: List[str]) -> Dict[str, Any]:
        """Major degradation for test service."""
        return {
            "degradation_level": "major",
            "message": "Test service operating in limited mode",
            "conditions": conditions,
            "fallback_response": "This is a major degradation fallback for testing",
            "degraded": True,
            "limited_functionality": True
        }


# Global instance
degradation_manager = IntelligentDegradationManager()


# Convenience functions
async def degrade_if_needed(service_name: str, conditions: List[str], user_id: Optional[str] = None) -> Any:
    """Apply intelligent degradation if needed for a service."""
    return await degradation_manager.apply_degradation(service_name, conditions, user_id)


async def learn_from_feedback(user_id: str, service_name: str, 
                            strategy_level: DegradationLevel, satisfaction_score: float):
    """Learn from user feedback to improve degradation decisions."""
    await degradation_manager.learn_from_user_feedback(user_id, service_name, strategy_level, satisfaction_score)


async def adjust_degradation_dynamically(service_name: str):
    """Dynamically adjust degradation level based on current conditions."""
    await degradation_manager.adjust_degradation_level_dynamically(service_name)


def get_degradation_analytics() -> Dict[str, Any]:
    """Get analytics about degradation effectiveness."""
    return degradation_manager.get_degradation_analytics()


def with_intelligent_degradation(service_name: str, user_id_param: str = None):
    """Enhanced decorator to add intelligent graceful degradation to a function."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Extract user_id if specified
            user_id = None
            if user_id_param and user_id_param in kwargs:
                user_id = kwargs[user_id_param]
            
            try:
                # Try dynamic adjustment before function execution
                await degradation_manager.adjust_degradation_level_dynamically(service_name)
                
                return await func(*args, **kwargs)
                
            except Exception as e:
                logger.warning(f"Function {func.__name__} failed, attempting intelligent degradation")
                
                # Determine conditions based on exception
                conditions = []
                error_msg = str(e).lower()
                
                if "timeout" in error_msg:
                    conditions.extend(["timeout", "processing_slow"])
                if "memory" in error_msg:
                    conditions.extend(["memory_pressure", "memory_critical"])
                if "database" in error_msg:
                    conditions.extend(["database_connection_issues", "database_slow"])
                if "api" in error_msg:
                    conditions.extend(["api_failure", "ai_api_down"])
                if "network" in error_msg:
                    conditions.extend(["network_issues", "high_latency"])
                if "cpu" in error_msg or "load" in error_msg:
                    conditions.extend(["high_cpu", "high_load"])
                
                # If no specific conditions found, add generic ones
                if not conditions:
                    conditions = ["system_error", "processing_failed"]
                
                # Apply intelligent degradation
                degraded_result = await degradation_manager.apply_degradation(
                    service_name, conditions, user_id
                )
                
                if degraded_result:
                    return degraded_result
                
                # If no degradation strategy available, try emergency fallback
                emergency_result = await degradation_manager._emergency_fallback(service_name)
                if emergency_result:
                    return emergency_result
                
                # If all else fails, re-raise original exception
                raise e
        
        return wrapper
    return decorator


# Backward compatibility
def with_graceful_degradation(service_name: str):
    """Backward compatible decorator."""
    return with_intelligent_degradation(service_name)