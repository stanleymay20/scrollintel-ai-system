"""
AI-Powered User Experience Optimization Engine

This module implements machine learning models for failure prediction and prevention,
intelligent user behavior analysis for proactive assistance, personalized degradation
strategies, and adaptive interface optimization based on usage patterns.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os

logger = logging.getLogger(__name__)

class PredictionType(Enum):
    FAILURE_RISK = "failure_risk"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    USER_FRUSTRATION = "user_frustration"
    SYSTEM_OVERLOAD = "system_overload"

class UserBehaviorPattern(Enum):
    POWER_USER = "power_user"
    CASUAL_USER = "casual_user"
    STRUGGLING_USER = "struggling_user"
    NEW_USER = "new_user"

class DegradationStrategy(Enum):
    MINIMAL = "minimal"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"
    CUSTOM = "custom"

@dataclass
class FailurePrediction:
    prediction_type: PredictionType
    probability: float
    confidence: float
    time_to_failure: Optional[int]  # minutes
    contributing_factors: List[str]
    recommended_actions: List[str]
    timestamp: datetime

@dataclass
class UserBehaviorAnalysis:
    user_id: str
    behavior_pattern: UserBehaviorPattern
    engagement_score: float
    frustration_indicators: List[str]
    preferred_features: List[str]
    usage_patterns: Dict[str, Any]
    assistance_needs: List[str]
    timestamp: datetime

@dataclass
class PersonalizedDegradation:
    user_id: str
    strategy: DegradationStrategy
    feature_priorities: Dict[str, int]
    acceptable_delays: Dict[str, float]
    fallback_preferences: Dict[str, str]
    communication_style: str
    timestamp: datetime

@dataclass
class InterfaceOptimization:
    user_id: str
    layout_preferences: Dict[str, Any]
    interaction_patterns: Dict[str, float]
    performance_requirements: Dict[str, float]
    accessibility_needs: List[str]
    optimization_suggestions: List[str]
    timestamp: datetime

class AIUXOptimizer:
    """AI-powered user experience optimization engine"""
    
    def __init__(self):
        self.failure_predictor = None
        self.behavior_analyzer = None
        self.degradation_personalizer = None
        self.interface_optimizer = None
        self.scaler = StandardScaler()
        self.models_path = "models/ai_ux"
        self.user_profiles = {}
        self.system_metrics_history = []
        self.user_interaction_history = {}
        self._initialized = False
        
        # Initialize models synchronously for testing
        try:
            # Try to initialize in async context if available
            loop = asyncio.get_running_loop()
            asyncio.create_task(self._initialize_models())
        except RuntimeError:
            # No event loop running, initialize synchronously
            self._initialize_models_sync()
    
    def _initialize_models_sync(self):
        """Initialize ML models synchronously for testing"""
        try:
            os.makedirs(self.models_path, exist_ok=True)
            
            # Initialize failure prediction model
            self.failure_predictor = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            
            # Initialize behavior analysis model
            self.behavior_analyzer = KMeans(
                n_clusters=4,  # 4 behavior patterns
                random_state=42
            )
            
            # Initialize anomaly detection for user frustration
            self.frustration_detector = IsolationForest(
                contamination=0.1,
                random_state=42
            )
            
            self._initialized = True
            logger.info("AI UX Optimizer models initialized successfully (sync)")
            
        except Exception as e:
            logger.error(f"Error initializing AI UX models: {e}")

    async def _initialize_models(self):
        """Initialize ML models for UX optimization"""
        try:
            os.makedirs(self.models_path, exist_ok=True)
            
            # Initialize failure prediction model
            self.failure_predictor = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            
            # Initialize behavior analysis model
            self.behavior_analyzer = KMeans(
                n_clusters=4,  # 4 behavior patterns
                random_state=42
            )
            
            # Initialize anomaly detection for user frustration
            self.frustration_detector = IsolationForest(
                contamination=0.1,
                random_state=42
            )
            
            # Load existing models if available
            await self._load_models()
            
            self._initialized = True
            logger.info("AI UX Optimizer models initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing AI UX models: {e}")
    
    async def _load_models(self):
        """Load pre-trained models from disk"""
        try:
            model_files = {
                'failure_predictor': f"{self.models_path}/failure_predictor.pkl",
                'behavior_analyzer': f"{self.models_path}/behavior_analyzer.pkl",
                'frustration_detector': f"{self.models_path}/frustration_detector.pkl",
                'scaler': f"{self.models_path}/scaler.pkl"
            }
            
            for model_name, file_path in model_files.items():
                if os.path.exists(file_path):
                    setattr(self, model_name, joblib.load(file_path))
                    logger.info(f"Loaded {model_name} model")
                    
        except Exception as e:
            logger.error(f"Error loading models: {e}")
    
    async def _save_models(self):
        """Save trained models to disk"""
        try:
            model_files = {
                'failure_predictor': self.failure_predictor,
                'behavior_analyzer': self.behavior_analyzer,
                'frustration_detector': self.frustration_detector,
                'scaler': self.scaler
            }
            
            for model_name, model in model_files.items():
                if model is not None:
                    file_path = f"{self.models_path}/{model_name}.pkl"
                    joblib.dump(model, file_path)
                    logger.info(f"Saved {model_name} model")
                    
        except Exception as e:
            logger.error(f"Error saving models: {e}")
    
    async def predict_failures(self, system_metrics: Dict[str, Any]) -> List[FailurePrediction]:
        """Predict potential system failures using ML models"""
        try:
            predictions = []
            
            # Extract features from system metrics
            features = self._extract_failure_features(system_metrics)
            
            if self.failure_predictor is not None and len(features) > 0:
                # Scale features
                features_scaled = self.scaler.transform([features])
                
                # Predict failure probability
                failure_prob = self.failure_predictor.predict_proba(features_scaled)[0]
                
                # Generate predictions for different failure types
                for i, pred_type in enumerate(PredictionType):
                    if i < len(failure_prob):
                        prob = failure_prob[i]
                        
                        if prob > 0.3:  # Threshold for concern
                            prediction = FailurePrediction(
                                prediction_type=pred_type,
                                probability=prob,
                                confidence=min(prob * 1.2, 1.0),
                                time_to_failure=self._estimate_time_to_failure(pred_type, prob),
                                contributing_factors=self._identify_contributing_factors(features, pred_type),
                                recommended_actions=self._get_failure_recommendations(pred_type, prob),
                                timestamp=datetime.now()
                            )
                            predictions.append(prediction)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error predicting failures: {e}")
            return []
    
    def _extract_failure_features(self, metrics: Dict[str, Any]) -> List[float]:
        """Extract features for failure prediction"""
        features = []
        
        try:
            # System performance features
            features.extend([
                metrics.get('cpu_usage', 0.0),
                metrics.get('memory_usage', 0.0),
                metrics.get('disk_usage', 0.0),
                metrics.get('network_latency', 0.0),
                metrics.get('error_rate', 0.0),
                metrics.get('response_time', 0.0),
                metrics.get('active_users', 0),
                metrics.get('request_rate', 0.0)
            ])
            
            # Historical trend features
            if len(self.system_metrics_history) > 0:
                recent_metrics = self.system_metrics_history[-10:]  # Last 10 data points
                
                # Calculate trends
                cpu_trend = np.mean([m.get('cpu_usage', 0) for m in recent_metrics])
                memory_trend = np.mean([m.get('memory_usage', 0) for m in recent_metrics])
                error_trend = np.mean([m.get('error_rate', 0) for m in recent_metrics])
                
                features.extend([cpu_trend, memory_trend, error_trend])
            else:
                features.extend([0.0, 0.0, 0.0])
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting failure features: {e}")
            return []
    
    def _estimate_time_to_failure(self, pred_type: PredictionType, probability: float) -> Optional[int]:
        """Estimate time until potential failure"""
        base_times = {
            PredictionType.FAILURE_RISK: 30,
            PredictionType.PERFORMANCE_DEGRADATION: 15,
            PredictionType.USER_FRUSTRATION: 10,
            PredictionType.SYSTEM_OVERLOAD: 5
        }
        
        base_time = base_times.get(pred_type, 20)
        # Higher probability means shorter time to failure
        time_factor = max(0.1, 1.0 - probability)
        
        return int(base_time * time_factor)
    
    def _identify_contributing_factors(self, features: List[float], pred_type: PredictionType) -> List[str]:
        """Identify factors contributing to potential failure"""
        factors = []
        
        if len(features) >= 8:
            cpu_usage, memory_usage, disk_usage, network_latency, error_rate, response_time, active_users, request_rate = features[:8]
            
            if cpu_usage > 0.8:
                factors.append("High CPU usage")
            if memory_usage > 0.8:
                factors.append("High memory usage")
            if disk_usage > 0.9:
                factors.append("Low disk space")
            if network_latency > 1000:
                factors.append("High network latency")
            if error_rate > 0.05:
                factors.append("Elevated error rate")
            if response_time > 5000:
                factors.append("Slow response times")
            if active_users > 1000:
                factors.append("High user load")
            if request_rate > 100:
                factors.append("High request rate")
        
        return factors
    
    def _get_failure_recommendations(self, pred_type: PredictionType, probability: float) -> List[str]:
        """Get recommendations to prevent predicted failures"""
        recommendations = []
        
        base_recommendations = {
            PredictionType.FAILURE_RISK: [
                "Scale up resources proactively",
                "Enable circuit breakers",
                "Prepare fallback systems"
            ],
            PredictionType.PERFORMANCE_DEGRADATION: [
                "Optimize database queries",
                "Enable caching",
                "Reduce feature complexity"
            ],
            PredictionType.USER_FRUSTRATION: [
                "Show progress indicators",
                "Provide helpful messages",
                "Enable graceful degradation"
            ],
            PredictionType.SYSTEM_OVERLOAD: [
                "Implement rate limiting",
                "Scale horizontally",
                "Defer non-critical operations"
            ]
        }
        
        recommendations = base_recommendations.get(pred_type, [])
        
        if probability > 0.7:
            recommendations.append("Take immediate action")
        elif probability > 0.5:
            recommendations.append("Monitor closely")
        
        return recommendations   
 
    async def analyze_user_behavior(self, user_id: str, interaction_data: Dict[str, Any]) -> UserBehaviorAnalysis:
        """Analyze user behavior patterns for proactive assistance"""
        try:
            # Store interaction data
            if user_id not in self.user_interaction_history:
                self.user_interaction_history[user_id] = []
            
            self.user_interaction_history[user_id].append({
                **interaction_data,
                'timestamp': datetime.now()
            })
            
            # Keep only recent interactions (last 100)
            self.user_interaction_history[user_id] = self.user_interaction_history[user_id][-100:]
            
            # Extract behavior features
            behavior_features = self._extract_behavior_features(user_id, interaction_data)
            
            # Classify behavior pattern
            behavior_pattern = self._classify_behavior_pattern(behavior_features)
            
            # Calculate engagement score
            engagement_score = self._calculate_engagement_score(user_id, interaction_data)
            
            # Detect frustration indicators
            frustration_indicators = self._detect_frustration_indicators(user_id, interaction_data)
            
            # Identify preferred features
            preferred_features = self._identify_preferred_features(user_id)
            
            # Analyze usage patterns
            usage_patterns = self._analyze_usage_patterns(user_id)
            
            # Determine assistance needs
            assistance_needs = self._determine_assistance_needs(behavior_pattern, frustration_indicators)
            
            analysis = UserBehaviorAnalysis(
                user_id=user_id,
                behavior_pattern=behavior_pattern,
                engagement_score=engagement_score,
                frustration_indicators=frustration_indicators,
                preferred_features=preferred_features,
                usage_patterns=usage_patterns,
                assistance_needs=assistance_needs,
                timestamp=datetime.now()
            )
            
            # Update user profile
            self.user_profiles[user_id] = analysis
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing user behavior for {user_id}: {e}")
            return UserBehaviorAnalysis(
                user_id=user_id,
                behavior_pattern=UserBehaviorPattern.CASUAL_USER,
                engagement_score=0.5,
                frustration_indicators=[],
                preferred_features=[],
                usage_patterns={},
                assistance_needs=[],
                timestamp=datetime.now()
            )
    
    def _extract_behavior_features(self, user_id: str, interaction_data: Dict[str, Any]) -> List[float]:
        """Extract features for behavior analysis"""
        features = []
        
        try:
            user_history = self.user_interaction_history.get(user_id, [])
            
            if len(user_history) == 0:
                return [0.0] * 10  # Default features for new users
            
            # Session-based features
            session_duration = interaction_data.get('session_duration', 0)
            clicks_per_minute = interaction_data.get('clicks_per_minute', 0)
            pages_visited = interaction_data.get('pages_visited', 0)
            errors_encountered = interaction_data.get('errors_encountered', 0)
            help_requests = interaction_data.get('help_requests', 0)
            
            # Historical features
            avg_session_duration = np.mean([h.get('session_duration', 0) for h in user_history[-10:]])
            total_sessions = len(user_history)
            error_rate = np.mean([h.get('errors_encountered', 0) for h in user_history[-10:]])
            help_frequency = np.mean([h.get('help_requests', 0) for h in user_history[-10:]])
            
            # Feature complexity usage
            advanced_features_used = interaction_data.get('advanced_features_used', 0)
            
            features = [
                session_duration,
                clicks_per_minute,
                pages_visited,
                errors_encountered,
                help_requests,
                avg_session_duration,
                total_sessions,
                error_rate,
                help_frequency,
                advanced_features_used
            ]
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting behavior features: {e}")
            return [0.0] * 10
    
    def _classify_behavior_pattern(self, features: List[float]) -> UserBehaviorPattern:
        """Classify user behavior pattern"""
        try:
            if len(features) < 10:
                return UserBehaviorPattern.CASUAL_USER
            
            session_duration, clicks_per_minute, pages_visited, errors_encountered, help_requests, \
            avg_session_duration, total_sessions, error_rate, help_frequency, advanced_features_used = features
            
            # Power user indicators
            if (avg_session_duration > 30 and 
                advanced_features_used > 5 and 
                total_sessions > 50 and 
                error_rate < 0.1):
                return UserBehaviorPattern.POWER_USER
            
            # Struggling user indicators
            if (error_rate > 0.3 or 
                help_frequency > 2 or 
                (session_duration < 5 and errors_encountered > 2)):
                return UserBehaviorPattern.STRUGGLING_USER
            
            # New user indicators
            if total_sessions < 5:
                return UserBehaviorPattern.NEW_USER
            
            # Default to casual user
            return UserBehaviorPattern.CASUAL_USER
            
        except Exception as e:
            logger.error(f"Error classifying behavior pattern: {e}")
            return UserBehaviorPattern.CASUAL_USER
    
    def _calculate_engagement_score(self, user_id: str, interaction_data: Dict[str, Any]) -> float:
        """Calculate user engagement score"""
        try:
            user_history = self.user_interaction_history.get(user_id, [])
            
            if len(user_history) == 0:
                return 0.5  # Neutral score for new users
            
            # Engagement factors
            session_duration = interaction_data.get('session_duration', 0)
            feature_usage = interaction_data.get('features_used', 0)
            return_frequency = len(user_history) / max(1, (datetime.now() - datetime.fromisoformat(user_history[0].get('timestamp', datetime.now().isoformat()))).days)
            
            # Calculate weighted score
            duration_score = min(1.0, session_duration / 60)  # Normalize to 1 hour
            feature_score = min(1.0, feature_usage / 10)  # Normalize to 10 features
            frequency_score = min(1.0, return_frequency / 2)  # Normalize to 2 visits per day
            
            engagement_score = (duration_score * 0.4 + feature_score * 0.3 + frequency_score * 0.3)
            
            return max(0.0, min(1.0, engagement_score))
            
        except Exception as e:
            logger.error(f"Error calculating engagement score: {e}")
            return 0.5
    
    def _detect_frustration_indicators(self, user_id: str, interaction_data: Dict[str, Any]) -> List[str]:
        """Detect indicators of user frustration"""
        indicators = []
        
        try:
            # Current session indicators
            if interaction_data.get('errors_encountered', 0) > 3:
                indicators.append("Multiple errors in session")
            
            if interaction_data.get('help_requests', 0) > 2:
                indicators.append("Frequent help requests")
            
            if interaction_data.get('session_duration', 0) < 2 and interaction_data.get('pages_visited', 0) > 5:
                indicators.append("Rapid page switching")
            
            if interaction_data.get('back_button_usage', 0) > 5:
                indicators.append("Excessive back button usage")
            
            # Historical indicators
            user_history = self.user_interaction_history.get(user_id, [])
            if len(user_history) > 3:
                recent_sessions = user_history[-3:]
                avg_errors = np.mean([s.get('errors_encountered', 0) for s in recent_sessions])
                
                if avg_errors > 2:
                    indicators.append("Consistent error patterns")
                
                session_durations = [s.get('session_duration', 0) for s in recent_sessions]
                if all(d < 5 for d in session_durations):
                    indicators.append("Consistently short sessions")
            
            return indicators
            
        except Exception as e:
            logger.error(f"Error detecting frustration indicators: {e}")
            return []
    
    def _identify_preferred_features(self, user_id: str) -> List[str]:
        """Identify user's preferred features"""
        try:
            user_history = self.user_interaction_history.get(user_id, [])
            
            if len(user_history) == 0:
                return []
            
            # Count feature usage
            feature_usage = {}
            for session in user_history:
                features_used = session.get('features_used_list', [])
                for feature in features_used:
                    feature_usage[feature] = feature_usage.get(feature, 0) + 1
            
            # Sort by usage frequency
            sorted_features = sorted(feature_usage.items(), key=lambda x: x[1], reverse=True)
            
            # Return top 5 preferred features
            return [feature for feature, count in sorted_features[:5]]
            
        except Exception as e:
            logger.error(f"Error identifying preferred features: {e}")
            return []
    
    def _analyze_usage_patterns(self, user_id: str) -> Dict[str, Any]:
        """Analyze user's usage patterns"""
        try:
            user_history = self.user_interaction_history.get(user_id, [])
            
            if len(user_history) == 0:
                return {}
            
            patterns = {}
            
            # Time-based patterns
            session_times = []
            for session in user_history:
                timestamp = session.get('timestamp')
                if isinstance(timestamp, str):
                    timestamp = datetime.fromisoformat(timestamp)
                session_times.append(timestamp.hour)
            
            if session_times:
                patterns['peak_usage_hour'] = max(set(session_times), key=session_times.count)
                patterns['usage_distribution'] = {
                    'morning': sum(1 for h in session_times if 6 <= h < 12) / len(session_times),
                    'afternoon': sum(1 for h in session_times if 12 <= h < 18) / len(session_times),
                    'evening': sum(1 for h in session_times if 18 <= h < 24) / len(session_times),
                    'night': sum(1 for h in session_times if 0 <= h < 6) / len(session_times)
                }
            
            # Session patterns
            session_durations = [s.get('session_duration', 0) for s in user_history]
            if session_durations:
                patterns['avg_session_duration'] = np.mean(session_durations)
                patterns['session_consistency'] = 1.0 - (np.std(session_durations) / max(1, np.mean(session_durations)))
            
            # Feature usage patterns
            all_features = []
            for session in user_history:
                all_features.extend(session.get('features_used_list', []))
            
            if all_features:
                unique_features = set(all_features)
                patterns['feature_diversity'] = len(unique_features) / max(1, len(all_features))
                patterns['most_used_feature'] = max(set(all_features), key=all_features.count)
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error analyzing usage patterns: {e}")
            return {}
    
    def _determine_assistance_needs(self, behavior_pattern: UserBehaviorPattern, frustration_indicators: List[str]) -> List[str]:
        """Determine what assistance the user needs"""
        needs = []
        
        # Pattern-based needs
        if behavior_pattern == UserBehaviorPattern.NEW_USER:
            needs.extend([
                "Onboarding tutorial",
                "Feature introduction",
                "Basic navigation help"
            ])
        elif behavior_pattern == UserBehaviorPattern.STRUGGLING_USER:
            needs.extend([
                "Error prevention guidance",
                "Simplified interface",
                "Step-by-step instructions"
            ])
        elif behavior_pattern == UserBehaviorPattern.POWER_USER:
            needs.extend([
                "Advanced features",
                "Keyboard shortcuts",
                "Customization options"
            ])
        
        # Frustration-based needs
        if "Multiple errors in session" in frustration_indicators:
            needs.append("Error recovery assistance")
        
        if "Frequent help requests" in frustration_indicators:
            needs.append("Contextual help improvement")
        
        if "Rapid page switching" in frustration_indicators:
            needs.append("Navigation optimization")
        
        return list(set(needs))  # Remove duplicates 
   
    async def create_personalized_degradation(self, user_id: str, system_conditions: Dict[str, Any]) -> PersonalizedDegradation:
        """Create personalized degradation strategy based on user preferences"""
        try:
            user_profile = self.user_profiles.get(user_id)
            
            if not user_profile:
                # Create default degradation for unknown users
                return PersonalizedDegradation(
                    user_id=user_id,
                    strategy=DegradationStrategy.MODERATE,
                    feature_priorities={},
                    acceptable_delays={},
                    fallback_preferences={},
                    communication_style="informative",
                    timestamp=datetime.now()
                )
            
            # Determine strategy based on user behavior
            strategy = self._determine_degradation_strategy(user_profile, system_conditions)
            
            # Set feature priorities based on user preferences
            feature_priorities = self._calculate_feature_priorities(user_profile)
            
            # Determine acceptable delays based on user patience
            acceptable_delays = self._calculate_acceptable_delays(user_profile)
            
            # Set fallback preferences
            fallback_preferences = self._determine_fallback_preferences(user_profile)
            
            # Determine communication style
            communication_style = self._determine_communication_style(user_profile)
            
            degradation = PersonalizedDegradation(
                user_id=user_id,
                strategy=strategy,
                feature_priorities=feature_priorities,
                acceptable_delays=acceptable_delays,
                fallback_preferences=fallback_preferences,
                communication_style=communication_style,
                timestamp=datetime.now()
            )
            
            return degradation
            
        except Exception as e:
            logger.error(f"Error creating personalized degradation for {user_id}: {e}")
            return PersonalizedDegradation(
                user_id=user_id,
                strategy=DegradationStrategy.MODERATE,
                feature_priorities={},
                acceptable_delays={},
                fallback_preferences={},
                communication_style="informative",
                timestamp=datetime.now()
            )
    
    def _determine_degradation_strategy(self, user_profile: UserBehaviorAnalysis, system_conditions: Dict[str, Any]) -> DegradationStrategy:
        """Determine appropriate degradation strategy"""
        try:
            # Power users prefer minimal degradation
            if user_profile.behavior_pattern == UserBehaviorPattern.POWER_USER:
                return DegradationStrategy.MINIMAL
            
            # Struggling users need aggressive degradation to simplify experience
            if user_profile.behavior_pattern == UserBehaviorPattern.STRUGGLING_USER:
                return DegradationStrategy.AGGRESSIVE
            
            # New users benefit from moderate degradation with guidance
            if user_profile.behavior_pattern == UserBehaviorPattern.NEW_USER:
                return DegradationStrategy.MODERATE
            
            # Consider system conditions
            system_load = system_conditions.get('system_load', 0.5)
            if system_load > 0.8:
                return DegradationStrategy.AGGRESSIVE
            elif system_load > 0.6:
                return DegradationStrategy.MODERATE
            
            return DegradationStrategy.MINIMAL
            
        except Exception as e:
            logger.error(f"Error determining degradation strategy: {e}")
            return DegradationStrategy.MODERATE
    
    def _calculate_feature_priorities(self, user_profile: UserBehaviorAnalysis) -> Dict[str, int]:
        """Calculate feature priorities based on user preferences"""
        priorities = {}
        
        try:
            # High priority for preferred features
            for i, feature in enumerate(user_profile.preferred_features):
                priorities[feature] = 10 - i  # Decreasing priority
            
            # Default priorities for common features
            default_priorities = {
                'dashboard': 8,
                'search': 7,
                'navigation': 9,
                'data_visualization': 6,
                'export': 4,
                'advanced_analytics': 3,
                'collaboration': 5,
                'notifications': 6
            }
            
            # Merge with user preferences, keeping higher values
            for feature, priority in default_priorities.items():
                if feature not in priorities:
                    priorities[feature] = priority
                else:
                    priorities[feature] = max(priorities[feature], priority)
            
            return priorities
            
        except Exception as e:
            logger.error(f"Error calculating feature priorities: {e}")
            return {}
    
    def _calculate_acceptable_delays(self, user_profile: UserBehaviorAnalysis) -> Dict[str, float]:
        """Calculate acceptable delays based on user patience"""
        delays = {}
        
        try:
            # Base delays in seconds
            base_delays = {
                'page_load': 3.0,
                'search': 2.0,
                'data_processing': 10.0,
                'export': 30.0,
                'visualization': 5.0
            }
            
            # Adjust based on user behavior
            patience_multiplier = 1.0
            
            if user_profile.behavior_pattern == UserBehaviorPattern.POWER_USER:
                patience_multiplier = 1.5  # Power users are more patient
            elif user_profile.behavior_pattern == UserBehaviorPattern.STRUGGLING_USER:
                patience_multiplier = 0.7  # Struggling users are less patient
            elif user_profile.behavior_pattern == UserBehaviorPattern.NEW_USER:
                patience_multiplier = 0.8  # New users expect quick responses
            
            # Adjust based on engagement score
            engagement_factor = 0.5 + (user_profile.engagement_score * 0.5)
            patience_multiplier *= engagement_factor
            
            # Apply multiplier to base delays
            for operation, base_delay in base_delays.items():
                delays[operation] = base_delay * patience_multiplier
            
            return delays
            
        except Exception as e:
            logger.error(f"Error calculating acceptable delays: {e}")
            return {}
    
    def _determine_fallback_preferences(self, user_profile: UserBehaviorAnalysis) -> Dict[str, str]:
        """Determine fallback preferences based on user behavior"""
        preferences = {}
        
        try:
            if user_profile.behavior_pattern == UserBehaviorPattern.POWER_USER:
                preferences.update({
                    'data_visualization': 'simplified_charts',
                    'search': 'cached_results',
                    'analytics': 'basic_metrics',
                    'export': 'csv_format'
                })
            elif user_profile.behavior_pattern == UserBehaviorPattern.STRUGGLING_USER:
                preferences.update({
                    'data_visualization': 'simple_tables',
                    'search': 'guided_search',
                    'analytics': 'summary_only',
                    'export': 'basic_export'
                })
            elif user_profile.behavior_pattern == UserBehaviorPattern.NEW_USER:
                preferences.update({
                    'data_visualization': 'tutorial_mode',
                    'search': 'suggested_searches',
                    'analytics': 'explained_metrics',
                    'export': 'guided_export'
                })
            else:  # Casual user
                preferences.update({
                    'data_visualization': 'standard_charts',
                    'search': 'recent_results',
                    'analytics': 'key_metrics',
                    'export': 'standard_format'
                })
            
            return preferences
            
        except Exception as e:
            logger.error(f"Error determining fallback preferences: {e}")
            return {}
    
    def _determine_communication_style(self, user_profile: UserBehaviorAnalysis) -> str:
        """Determine appropriate communication style"""
        try:
            if user_profile.behavior_pattern == UserBehaviorPattern.POWER_USER:
                return "technical"  # Detailed, technical explanations
            elif user_profile.behavior_pattern == UserBehaviorPattern.STRUGGLING_USER:
                return "supportive"  # Encouraging, step-by-step guidance
            elif user_profile.behavior_pattern == UserBehaviorPattern.NEW_USER:
                return "educational"  # Explanatory, helpful
            else:
                return "informative"  # Clear, concise information
                
        except Exception as e:
            logger.error(f"Error determining communication style: {e}")
            return "informative"
    
    async def optimize_interface(self, user_id: str, current_interface: Dict[str, Any]) -> InterfaceOptimization:
        """Optimize interface based on user usage patterns"""
        try:
            user_profile = self.user_profiles.get(user_id)
            
            if not user_profile:
                # Return default optimization for unknown users
                return InterfaceOptimization(
                    user_id=user_id,
                    layout_preferences={},
                    interaction_patterns={},
                    performance_requirements={},
                    accessibility_needs=[],
                    optimization_suggestions=[],
                    timestamp=datetime.now()
                )
            
            # Analyze layout preferences
            layout_preferences = self._analyze_layout_preferences(user_profile, current_interface)
            
            # Analyze interaction patterns
            interaction_patterns = self._analyze_interaction_patterns(user_profile)
            
            # Determine performance requirements
            performance_requirements = self._determine_performance_requirements(user_profile)
            
            # Identify accessibility needs
            accessibility_needs = self._identify_accessibility_needs(user_profile)
            
            # Generate optimization suggestions
            optimization_suggestions = self._generate_optimization_suggestions(
                user_profile, layout_preferences, interaction_patterns
            )
            
            optimization = InterfaceOptimization(
                user_id=user_id,
                layout_preferences=layout_preferences,
                interaction_patterns=interaction_patterns,
                performance_requirements=performance_requirements,
                accessibility_needs=accessibility_needs,
                optimization_suggestions=optimization_suggestions,
                timestamp=datetime.now()
            )
            
            return optimization
            
        except Exception as e:
            logger.error(f"Error optimizing interface for {user_id}: {e}")
            return InterfaceOptimization(
                user_id=user_id,
                layout_preferences={},
                interaction_patterns={},
                performance_requirements={},
                accessibility_needs=[],
                optimization_suggestions=[],
                timestamp=datetime.now()
            )
    
    def _analyze_layout_preferences(self, user_profile: UserBehaviorAnalysis, current_interface: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze user's layout preferences"""
        preferences = {}
        
        try:
            # Based on behavior pattern
            if user_profile.behavior_pattern == UserBehaviorPattern.POWER_USER:
                preferences.update({
                    'density': 'compact',
                    'sidebar': 'expanded',
                    'toolbar': 'full',
                    'panels': 'multiple'
                })
            elif user_profile.behavior_pattern == UserBehaviorPattern.STRUGGLING_USER:
                preferences.update({
                    'density': 'spacious',
                    'sidebar': 'simplified',
                    'toolbar': 'essential',
                    'panels': 'single'
                })
            elif user_profile.behavior_pattern == UserBehaviorPattern.NEW_USER:
                preferences.update({
                    'density': 'comfortable',
                    'sidebar': 'guided',
                    'toolbar': 'contextual',
                    'panels': 'progressive'
                })
            
            # Adjust based on preferred features
            if 'data_visualization' in user_profile.preferred_features:
                preferences['chart_area'] = 'expanded'
            
            if 'search' in user_profile.preferred_features:
                preferences['search_prominence'] = 'high'
            
            return preferences
            
        except Exception as e:
            logger.error(f"Error analyzing layout preferences: {e}")
            return {}
    
    def _analyze_interaction_patterns(self, user_profile: UserBehaviorAnalysis) -> Dict[str, float]:
        """Analyze user's interaction patterns"""
        patterns = {}
        
        try:
            usage_patterns = user_profile.usage_patterns
            
            # Extract interaction metrics
            patterns['click_frequency'] = usage_patterns.get('avg_clicks_per_session', 10.0)
            patterns['navigation_speed'] = usage_patterns.get('pages_per_minute', 2.0)
            patterns['feature_switching'] = usage_patterns.get('feature_diversity', 0.5)
            patterns['session_depth'] = usage_patterns.get('avg_session_duration', 15.0) / 60.0  # Convert to hours
            
            # Behavioral adjustments
            if user_profile.behavior_pattern == UserBehaviorPattern.POWER_USER:
                patterns['keyboard_usage'] = 0.8  # High keyboard usage
                patterns['shortcut_preference'] = 0.9
            else:
                patterns['keyboard_usage'] = 0.3  # Lower keyboard usage
                patterns['shortcut_preference'] = 0.4
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error analyzing interaction patterns: {e}")
            return {}
    
    def _determine_performance_requirements(self, user_profile: UserBehaviorAnalysis) -> Dict[str, float]:
        """Determine performance requirements based on user behavior"""
        requirements = {}
        
        try:
            # Base requirements
            requirements = {
                'page_load_time': 3.0,
                'interaction_response': 0.5,
                'data_refresh': 5.0,
                'search_response': 2.0
            }
            
            # Adjust based on user patience (from engagement score)
            patience_factor = 0.5 + (user_profile.engagement_score * 0.5)
            
            for metric in requirements:
                requirements[metric] *= (2.0 - patience_factor)  # Higher engagement = more patience
            
            # Behavior-specific adjustments
            if user_profile.behavior_pattern == UserBehaviorPattern.POWER_USER:
                requirements['data_refresh'] *= 0.8  # Power users want faster data
            elif user_profile.behavior_pattern == UserBehaviorPattern.STRUGGLING_USER:
                requirements['interaction_response'] *= 0.7  # Need faster feedback
            
            return requirements
            
        except Exception as e:
            logger.error(f"Error determining performance requirements: {e}")
            return {}
    
    def _identify_accessibility_needs(self, user_profile: UserBehaviorAnalysis) -> List[str]:
        """Identify accessibility needs based on user behavior"""
        needs = []
        
        try:
            # Infer needs from frustration indicators
            if "Rapid page switching" in user_profile.frustration_indicators:
                needs.append("Improved navigation clarity")
            
            if "Multiple errors in session" in user_profile.frustration_indicators:
                needs.append("Better error messaging")
            
            if "Frequent help requests" in user_profile.frustration_indicators:
                needs.append("Contextual help")
            
            # Behavior-based needs
            if user_profile.behavior_pattern == UserBehaviorPattern.STRUGGLING_USER:
                needs.extend([
                    "Larger click targets",
                    "High contrast mode",
                    "Simplified language"
                ])
            
            if user_profile.engagement_score < 0.3:
                needs.extend([
                    "Progress indicators",
                    "Motivational feedback"
                ])
            
            return needs
            
        except Exception as e:
            logger.error(f"Error identifying accessibility needs: {e}")
            return []
    
    def _generate_optimization_suggestions(self, user_profile: UserBehaviorAnalysis, 
                                         layout_preferences: Dict[str, Any], 
                                         interaction_patterns: Dict[str, float]) -> List[str]:
        """Generate interface optimization suggestions"""
        suggestions = []
        
        try:
            # Layout-based suggestions
            if layout_preferences.get('density') == 'compact':
                suggestions.append("Enable compact view mode")
            elif layout_preferences.get('density') == 'spacious':
                suggestions.append("Increase spacing between elements")
            
            # Interaction-based suggestions
            if interaction_patterns.get('keyboard_usage', 0) > 0.7:
                suggestions.append("Show keyboard shortcuts")
                suggestions.append("Enable keyboard navigation hints")
            
            if interaction_patterns.get('feature_switching', 0) > 0.7:
                suggestions.append("Add quick feature switcher")
            
            # Behavior-specific suggestions
            if user_profile.behavior_pattern == UserBehaviorPattern.POWER_USER:
                suggestions.extend([
                    "Enable advanced mode",
                    "Show detailed tooltips",
                    "Add customization options"
                ])
            elif user_profile.behavior_pattern == UserBehaviorPattern.NEW_USER:
                suggestions.extend([
                    "Show onboarding hints",
                    "Enable guided tour",
                    "Highlight key features"
                ])
            elif user_profile.behavior_pattern == UserBehaviorPattern.STRUGGLING_USER:
                suggestions.extend([
                    "Simplify interface",
                    "Add step-by-step guidance",
                    "Enable beginner mode"
                ])
            
            # Performance-based suggestions
            if user_profile.engagement_score < 0.4:
                suggestions.extend([
                    "Reduce loading times",
                    "Add progress feedback",
                    "Optimize critical path"
                ])
            
            return suggestions
            
        except Exception as e:
            logger.error(f"Error generating optimization suggestions: {e}")
            return []
    
    async def train_models(self, training_data: Dict[str, Any]):
        """Train ML models with new data"""
        try:
            # Train failure prediction model
            if 'failure_data' in training_data:
                await self._train_failure_predictor(training_data['failure_data'])
            
            # Train behavior analysis model
            if 'behavior_data' in training_data:
                await self._train_behavior_analyzer(training_data['behavior_data'])
            
            # Save updated models
            await self._save_models()
            
            logger.info("AI UX models trained successfully")
            
        except Exception as e:
            logger.error(f"Error training models: {e}")
    
    async def _train_failure_predictor(self, failure_data: List[Dict[str, Any]]):
        """Train the failure prediction model"""
        try:
            if len(failure_data) < 10:
                logger.warning("Insufficient data for training failure predictor")
                return
            
            # Prepare training data
            X = []
            y = []
            
            for data_point in failure_data:
                features = self._extract_failure_features(data_point.get('metrics', {}))
                if len(features) > 0:
                    X.append(features)
                    y.append(data_point.get('failure_occurred', 0))
            
            if len(X) > 0:
                X = np.array(X)
                y = np.array(y)
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                # Scale features
                self.scaler.fit(X_train)
                X_train_scaled = self.scaler.transform(X_train)
                
                # Train model
                self.failure_predictor.fit(X_train_scaled, y_train)
                
                logger.info(f"Failure predictor trained on {len(X_train)} samples")
                
        except Exception as e:
            logger.error(f"Error training failure predictor: {e}")
    
    async def _train_behavior_analyzer(self, behavior_data: List[Dict[str, Any]]):
        """Train the behavior analysis model"""
        try:
            if len(behavior_data) < 20:
                logger.warning("Insufficient data for training behavior analyzer")
                return
            
            # Prepare training data
            X = []
            
            for data_point in behavior_data:
                features = self._extract_behavior_features(
                    data_point.get('user_id', ''),
                    data_point.get('interaction_data', {})
                )
                if len(features) > 0:
                    X.append(features)
            
            if len(X) > 0:
                X = np.array(X)
                
                # Train clustering model
                self.behavior_analyzer.fit(X)
                
                logger.info(f"Behavior analyzer trained on {len(X)} samples")
                
        except Exception as e:
            logger.error(f"Error training behavior analyzer: {e}")
    
    async def get_optimization_metrics(self) -> Dict[str, Any]:
        """Get metrics about AI UX optimization performance"""
        try:
            metrics = {
                'total_users_analyzed': len(self.user_profiles),
                'behavior_patterns': {},
                'average_engagement_score': 0.0,
                'common_frustration_indicators': {},
                'optimization_suggestions_generated': 0,
                'model_performance': {}
            }
            
            if len(self.user_profiles) > 0:
                # Analyze behavior patterns
                pattern_counts = {}
                engagement_scores = []
                frustration_counts = {}
                
                for profile in self.user_profiles.values():
                    # Count behavior patterns
                    pattern = profile.behavior_pattern.value
                    pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
                    
                    # Collect engagement scores
                    engagement_scores.append(profile.engagement_score)
                    
                    # Count frustration indicators
                    for indicator in profile.frustration_indicators:
                        frustration_counts[indicator] = frustration_counts.get(indicator, 0) + 1
                
                metrics['behavior_patterns'] = pattern_counts
                metrics['average_engagement_score'] = np.mean(engagement_scores)
                metrics['common_frustration_indicators'] = dict(
                    sorted(frustration_counts.items(), key=lambda x: x[1], reverse=True)[:5]
                )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting optimization metrics: {e}")
            return {}