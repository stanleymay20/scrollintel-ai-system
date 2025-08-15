"""
Tests for AI UX Optimization Engine

This module contains comprehensive tests for the AI-powered user experience
optimization system, including failure prediction, behavior analysis,
personalized degradation, and interface optimization.
"""

import pytest
import asyncio
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any

from scrollintel.engines.ai_ux_optimizer import (
    AIUXOptimizer, PredictionType, UserBehaviorPattern, DegradationStrategy,
    FailurePrediction, UserBehaviorAnalysis, PersonalizedDegradation, InterfaceOptimization
)

class TestAIUXOptimizer:
    """Test cases for AI UX Optimizer"""
    
    @pytest.fixture
    def optimizer(self):
        """Create AI UX Optimizer instance for testing"""
        return AIUXOptimizer()
    
    @pytest.fixture
    def sample_system_metrics(self):
        """Sample system metrics for testing"""
        return {
            'cpu_usage': 0.7,
            'memory_usage': 0.6,
            'disk_usage': 0.4,
            'network_latency': 150,
            'error_rate': 0.02,
            'response_time': 800,
            'active_users': 150,
            'request_rate': 25
        }
    
    @pytest.fixture
    def sample_interaction_data(self):
        """Sample user interaction data for testing"""
        return {
            'session_duration': 25,
            'clicks_per_minute': 8,
            'pages_visited': 5,
            'errors_encountered': 1,
            'help_requests': 0,
            'features_used': 3,
            'features_used_list': ['dashboard', 'search', 'export'],
            'advanced_features_used': 1,
            'back_button_usage': 2
        }
    
    @pytest.mark.asyncio
    async def test_initialization(self, optimizer):
        """Test optimizer initialization"""
        assert optimizer is not None
        assert optimizer.user_profiles == {}
        assert optimizer.system_metrics_history == []
        assert optimizer.user_interaction_history == {}
        
        # Wait for models to initialize
        await asyncio.sleep(0.1)
        
        assert optimizer.failure_predictor is not None
        assert optimizer.behavior_analyzer is not None
        assert optimizer.scaler is not None
    
    @pytest.mark.asyncio
    async def test_predict_failures(self, optimizer, sample_system_metrics):
        """Test failure prediction functionality"""
        # Wait for initialization
        await asyncio.sleep(0.1)
        
        predictions = await optimizer.predict_failures(sample_system_metrics)
        
        assert isinstance(predictions, list)
        # Should have predictions for high-risk metrics
        assert len(predictions) >= 0
        
        for prediction in predictions:
            assert isinstance(prediction, FailurePrediction)
            assert prediction.probability >= 0.0
            assert prediction.probability <= 1.0
            assert prediction.confidence >= 0.0
            assert prediction.confidence <= 1.0
            assert isinstance(prediction.contributing_factors, list)
            assert isinstance(prediction.recommended_actions, list)
    
    def test_extract_failure_features(self, optimizer, sample_system_metrics):
        """Test failure feature extraction"""
        features = optimizer._extract_failure_features(sample_system_metrics)
        
        assert isinstance(features, list)
        assert len(features) >= 8  # At least basic features
        
        # Check that features are numeric
        for feature in features:
            assert isinstance(feature, (int, float))
    
    def test_estimate_time_to_failure(self, optimizer):
        """Test time to failure estimation"""
        time_estimate = optimizer._estimate_time_to_failure(
            PredictionType.FAILURE_RISK, 0.8
        )
        
        assert isinstance(time_estimate, int)
        assert time_estimate > 0
        
        # Higher probability should mean shorter time
        high_prob_time = optimizer._estimate_time_to_failure(
            PredictionType.FAILURE_RISK, 0.9
        )
        low_prob_time = optimizer._estimate_time_to_failure(
            PredictionType.FAILURE_RISK, 0.4
        )
        
        assert high_prob_time <= low_prob_time
    
    def test_identify_contributing_factors(self, optimizer):
        """Test contributing factor identification"""
        # High resource usage features
        high_usage_features = [0.9, 0.8, 0.95, 2000, 0.1, 8000, 2000, 200]
        factors = optimizer._identify_contributing_factors(
            high_usage_features, PredictionType.FAILURE_RISK
        )
        
        assert isinstance(factors, list)
        assert len(factors) > 0
        assert any('High CPU usage' in factor for factor in factors)
        assert any('High memory usage' in factor for factor in factors)
        assert any('Low disk space' in factor for factor in factors)
    
    def test_get_failure_recommendations(self, optimizer):
        """Test failure recommendation generation"""
        recommendations = optimizer._get_failure_recommendations(
            PredictionType.SYSTEM_OVERLOAD, 0.8
        )
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        assert any('rate limiting' in rec.lower() for rec in recommendations)
        assert 'Take immediate action' in recommendations  # High probability
    
    @pytest.mark.asyncio
    async def test_analyze_user_behavior(self, optimizer, sample_interaction_data):
        """Test user behavior analysis"""
        user_id = "test_user_123"
        
        analysis = await optimizer.analyze_user_behavior(user_id, sample_interaction_data)
        
        assert isinstance(analysis, UserBehaviorAnalysis)
        assert analysis.user_id == user_id
        assert isinstance(analysis.behavior_pattern, UserBehaviorPattern)
        assert 0.0 <= analysis.engagement_score <= 1.0
        assert isinstance(analysis.frustration_indicators, list)
        assert isinstance(analysis.preferred_features, list)
        assert isinstance(analysis.usage_patterns, dict)
        assert isinstance(analysis.assistance_needs, list)
        
        # Check that user profile is stored
        assert user_id in optimizer.user_profiles
        assert optimizer.user_profiles[user_id] == analysis
    
    def test_extract_behavior_features(self, optimizer, sample_interaction_data):
        """Test behavior feature extraction"""
        user_id = "test_user_123"
        features = optimizer._extract_behavior_features(user_id, sample_interaction_data)
        
        assert isinstance(features, list)
        assert len(features) == 10  # Expected number of features
        
        # Check that features are numeric
        for feature in features:
            assert isinstance(feature, (int, float))
    
    def test_classify_behavior_pattern(self, optimizer):
        """Test behavior pattern classification"""
        # Power user features
        power_user_features = [45, 12, 8, 0, 0, 40, 100, 0.05, 0.1, 8]
        pattern = optimizer._classify_behavior_pattern(power_user_features)
        assert pattern == UserBehaviorPattern.POWER_USER
        
        # Struggling user features
        struggling_user_features = [3, 5, 2, 5, 3, 5, 10, 0.4, 3, 0]
        pattern = optimizer._classify_behavior_pattern(struggling_user_features)
        assert pattern == UserBehaviorPattern.STRUGGLING_USER
        
        # New user features
        new_user_features = [10, 6, 3, 1, 1, 8, 2, 0.1, 0.5, 1]
        pattern = optimizer._classify_behavior_pattern(new_user_features)
        assert pattern == UserBehaviorPattern.NEW_USER
    
    def test_calculate_engagement_score(self, optimizer, sample_interaction_data):
        """Test engagement score calculation"""
        user_id = "test_user_123"
        score = optimizer._calculate_engagement_score(user_id, sample_interaction_data)
        
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
    
    def test_detect_frustration_indicators(self, optimizer):
        """Test frustration indicator detection"""
        # High frustration interaction data
        frustrating_data = {
            'errors_encountered': 5,
            'help_requests': 3,
            'session_duration': 1,
            'pages_visited': 8,
            'back_button_usage': 10
        }
        
        user_id = "frustrated_user"
        indicators = optimizer._detect_frustration_indicators(user_id, frustrating_data)
        
        assert isinstance(indicators, list)
        assert len(indicators) > 0
        assert any('Multiple errors' in indicator for indicator in indicators)
        assert any('help requests' in indicator for indicator in indicators)
        assert any('back button' in indicator for indicator in indicators)
    
    def test_identify_preferred_features(self, optimizer):
        """Test preferred feature identification"""
        user_id = "test_user_123"
        
        # Add some interaction history
        optimizer.user_interaction_history[user_id] = [
            {'features_used_list': ['dashboard', 'search', 'export']},
            {'features_used_list': ['dashboard', 'analytics']},
            {'features_used_list': ['dashboard', 'search', 'visualization']}
        ]
        
        preferred = optimizer._identify_preferred_features(user_id)
        
        assert isinstance(preferred, list)
        assert 'dashboard' in preferred  # Most used feature
        assert len(preferred) <= 5  # Top 5 features
    
    def test_analyze_usage_patterns(self, optimizer):
        """Test usage pattern analysis"""
        user_id = "test_user_123"
        
        # Add interaction history with timestamps
        now = datetime.now()
        optimizer.user_interaction_history[user_id] = [
            {
                'timestamp': now - timedelta(hours=2),
                'session_duration': 20,
                'features_used_list': ['dashboard', 'search']
            },
            {
                'timestamp': now - timedelta(hours=1),
                'session_duration': 30,
                'features_used_list': ['analytics', 'export']
            }
        ]
        
        patterns = optimizer._analyze_usage_patterns(user_id)
        
        assert isinstance(patterns, dict)
        assert 'avg_session_duration' in patterns
        assert 'feature_diversity' in patterns
        assert 'most_used_feature' in patterns
    
    def test_determine_assistance_needs(self, optimizer):
        """Test assistance need determination"""
        # New user with frustration
        needs = optimizer._determine_assistance_needs(
            UserBehaviorPattern.NEW_USER,
            ['Multiple errors in session', 'Frequent help requests']
        )
        
        assert isinstance(needs, list)
        assert any('tutorial' in need.lower() for need in needs)
        assert any('error recovery' in need.lower() for need in needs)
        
        # Power user
        power_needs = optimizer._determine_assistance_needs(
            UserBehaviorPattern.POWER_USER,
            []
        )
        
        assert any('advanced' in need.lower() for need in power_needs)
        assert any('shortcut' in need.lower() for need in power_needs)
    
    @pytest.mark.asyncio
    async def test_create_personalized_degradation(self, optimizer):
        """Test personalized degradation strategy creation"""
        user_id = "test_user_123"
        system_conditions = {'system_load': 0.8}
        
        # Create a user profile first
        sample_profile = UserBehaviorAnalysis(
            user_id=user_id,
            behavior_pattern=UserBehaviorPattern.POWER_USER,
            engagement_score=0.8,
            frustration_indicators=[],
            preferred_features=['dashboard', 'analytics'],
            usage_patterns={'avg_session_duration': 30},
            assistance_needs=[],
            timestamp=datetime.now()
        )
        optimizer.user_profiles[user_id] = sample_profile
        
        degradation = await optimizer.create_personalized_degradation(
            user_id, system_conditions
        )
        
        assert isinstance(degradation, PersonalizedDegradation)
        assert degradation.user_id == user_id
        assert isinstance(degradation.strategy, DegradationStrategy)
        assert isinstance(degradation.feature_priorities, dict)
        assert isinstance(degradation.acceptable_delays, dict)
        assert isinstance(degradation.fallback_preferences, dict)
        assert isinstance(degradation.communication_style, str)
    
    def test_determine_degradation_strategy(self, optimizer):
        """Test degradation strategy determination"""
        # Power user profile
        power_profile = UserBehaviorAnalysis(
            user_id="power_user",
            behavior_pattern=UserBehaviorPattern.POWER_USER,
            engagement_score=0.9,
            frustration_indicators=[],
            preferred_features=[],
            usage_patterns={},
            assistance_needs=[],
            timestamp=datetime.now()
        )
        
        strategy = optimizer._determine_degradation_strategy(
            power_profile, {'system_load': 0.5}
        )
        assert strategy == DegradationStrategy.MINIMAL
        
        # Struggling user profile
        struggling_profile = UserBehaviorAnalysis(
            user_id="struggling_user",
            behavior_pattern=UserBehaviorPattern.STRUGGLING_USER,
            engagement_score=0.3,
            frustration_indicators=['Multiple errors'],
            preferred_features=[],
            usage_patterns={},
            assistance_needs=[],
            timestamp=datetime.now()
        )
        
        strategy = optimizer._determine_degradation_strategy(
            struggling_profile, {'system_load': 0.5}
        )
        assert strategy == DegradationStrategy.AGGRESSIVE
    
    def test_calculate_feature_priorities(self, optimizer):
        """Test feature priority calculation"""
        profile = UserBehaviorAnalysis(
            user_id="test_user",
            behavior_pattern=UserBehaviorPattern.CASUAL_USER,
            engagement_score=0.6,
            frustration_indicators=[],
            preferred_features=['dashboard', 'search', 'export'],
            usage_patterns={},
            assistance_needs=[],
            timestamp=datetime.now()
        )
        
        priorities = optimizer._calculate_feature_priorities(profile)
        
        assert isinstance(priorities, dict)
        assert 'dashboard' in priorities
        assert 'search' in priorities
        assert priorities['dashboard'] >= priorities['search']  # First preferred has higher priority
    
    def test_calculate_acceptable_delays(self, optimizer):
        """Test acceptable delay calculation"""
        profile = UserBehaviorAnalysis(
            user_id="test_user",
            behavior_pattern=UserBehaviorPattern.POWER_USER,
            engagement_score=0.8,
            frustration_indicators=[],
            preferred_features=[],
            usage_patterns={},
            assistance_needs=[],
            timestamp=datetime.now()
        )
        
        delays = optimizer._calculate_acceptable_delays(profile)
        
        assert isinstance(delays, dict)
        assert 'page_load' in delays
        assert 'search' in delays
        assert all(isinstance(delay, float) for delay in delays.values())
        assert all(delay > 0 for delay in delays.values())
    
    def test_determine_fallback_preferences(self, optimizer):
        """Test fallback preference determination"""
        profile = UserBehaviorAnalysis(
            user_id="test_user",
            behavior_pattern=UserBehaviorPattern.NEW_USER,
            engagement_score=0.5,
            frustration_indicators=[],
            preferred_features=[],
            usage_patterns={},
            assistance_needs=[],
            timestamp=datetime.now()
        )
        
        preferences = optimizer._determine_fallback_preferences(profile)
        
        assert isinstance(preferences, dict)
        assert 'data_visualization' in preferences
        assert 'search' in preferences
        assert preferences['data_visualization'] == 'tutorial_mode'  # New user preference
    
    def test_determine_communication_style(self, optimizer):
        """Test communication style determination"""
        # Power user
        power_profile = UserBehaviorAnalysis(
            user_id="power_user",
            behavior_pattern=UserBehaviorPattern.POWER_USER,
            engagement_score=0.9,
            frustration_indicators=[],
            preferred_features=[],
            usage_patterns={},
            assistance_needs=[],
            timestamp=datetime.now()
        )
        
        style = optimizer._determine_communication_style(power_profile)
        assert style == "technical"
        
        # Struggling user
        struggling_profile = UserBehaviorAnalysis(
            user_id="struggling_user",
            behavior_pattern=UserBehaviorPattern.STRUGGLING_USER,
            engagement_score=0.3,
            frustration_indicators=[],
            preferred_features=[],
            usage_patterns={},
            assistance_needs=[],
            timestamp=datetime.now()
        )
        
        style = optimizer._determine_communication_style(struggling_profile)
        assert style == "supportive"
    
    @pytest.mark.asyncio
    async def test_optimize_interface(self, optimizer):
        """Test interface optimization"""
        user_id = "test_user_123"
        current_interface = {'layout': 'standard', 'theme': 'light'}
        
        # Create a user profile first
        sample_profile = UserBehaviorAnalysis(
            user_id=user_id,
            behavior_pattern=UserBehaviorPattern.CASUAL_USER,
            engagement_score=0.6,
            frustration_indicators=[],
            preferred_features=['dashboard', 'search'],
            usage_patterns={'avg_session_duration': 15},
            assistance_needs=[],
            timestamp=datetime.now()
        )
        optimizer.user_profiles[user_id] = sample_profile
        
        optimization = await optimizer.optimize_interface(user_id, current_interface)
        
        assert isinstance(optimization, InterfaceOptimization)
        assert optimization.user_id == user_id
        assert isinstance(optimization.layout_preferences, dict)
        assert isinstance(optimization.interaction_patterns, dict)
        assert isinstance(optimization.performance_requirements, dict)
        assert isinstance(optimization.accessibility_needs, list)
        assert isinstance(optimization.optimization_suggestions, list)
    
    def test_analyze_layout_preferences(self, optimizer):
        """Test layout preference analysis"""
        profile = UserBehaviorAnalysis(
            user_id="test_user",
            behavior_pattern=UserBehaviorPattern.POWER_USER,
            engagement_score=0.8,
            frustration_indicators=[],
            preferred_features=['data_visualization', 'search'],
            usage_patterns={},
            assistance_needs=[],
            timestamp=datetime.now()
        )
        
        preferences = optimizer._analyze_layout_preferences(profile, {})
        
        assert isinstance(preferences, dict)
        assert preferences['density'] == 'compact'  # Power user preference
        assert preferences['chart_area'] == 'expanded'  # Has data_visualization preference
        assert preferences['search_prominence'] == 'high'  # Has search preference
    
    def test_analyze_interaction_patterns(self, optimizer):
        """Test interaction pattern analysis"""
        profile = UserBehaviorAnalysis(
            user_id="test_user",
            behavior_pattern=UserBehaviorPattern.POWER_USER,
            engagement_score=0.8,
            frustration_indicators=[],
            preferred_features=[],
            usage_patterns={
                'avg_clicks_per_session': 15,
                'pages_per_minute': 3,
                'feature_diversity': 0.7,
                'avg_session_duration': 25
            },
            assistance_needs=[],
            timestamp=datetime.now()
        )
        
        patterns = optimizer._analyze_interaction_patterns(profile)
        
        assert isinstance(patterns, dict)
        assert 'click_frequency' in patterns
        assert 'navigation_speed' in patterns
        assert 'keyboard_usage' in patterns
        assert patterns['keyboard_usage'] == 0.8  # Power user has high keyboard usage
    
    def test_determine_performance_requirements(self, optimizer):
        """Test performance requirement determination"""
        profile = UserBehaviorAnalysis(
            user_id="test_user",
            behavior_pattern=UserBehaviorPattern.STRUGGLING_USER,
            engagement_score=0.3,
            frustration_indicators=[],
            preferred_features=[],
            usage_patterns={},
            assistance_needs=[],
            timestamp=datetime.now()
        )
        
        requirements = optimizer._determine_performance_requirements(profile)
        
        assert isinstance(requirements, dict)
        assert 'page_load_time' in requirements
        assert 'interaction_response' in requirements
        assert all(isinstance(req, float) for req in requirements.values())
        assert all(req > 0 for req in requirements.values())
    
    def test_identify_accessibility_needs(self, optimizer):
        """Test accessibility need identification"""
        profile = UserBehaviorAnalysis(
            user_id="test_user",
            behavior_pattern=UserBehaviorPattern.STRUGGLING_USER,
            engagement_score=0.2,
            frustration_indicators=['Rapid page switching', 'Multiple errors in session'],
            preferred_features=[],
            usage_patterns={},
            assistance_needs=[],
            timestamp=datetime.now()
        )
        
        needs = optimizer._identify_accessibility_needs(profile)
        
        assert isinstance(needs, list)
        assert any('navigation clarity' in need.lower() for need in needs)
        assert any('error messaging' in need.lower() for need in needs)
        assert any('larger click targets' in need.lower() for need in needs)
    
    def test_generate_optimization_suggestions(self, optimizer):
        """Test optimization suggestion generation"""
        profile = UserBehaviorAnalysis(
            user_id="test_user",
            behavior_pattern=UserBehaviorPattern.NEW_USER,
            engagement_score=0.4,
            frustration_indicators=[],
            preferred_features=[],
            usage_patterns={},
            assistance_needs=[],
            timestamp=datetime.now()
        )
        
        layout_preferences = {'density': 'comfortable'}
        interaction_patterns = {'keyboard_usage': 0.3, 'feature_switching': 0.8}
        
        suggestions = optimizer._generate_optimization_suggestions(
            profile, layout_preferences, interaction_patterns
        )
        
        assert isinstance(suggestions, list)
        assert any('onboarding' in suggestion.lower() for suggestion in suggestions)
        assert any('guided tour' in suggestion.lower() for suggestion in suggestions)
        assert any('quick feature switcher' in suggestion.lower() for suggestion in suggestions)
    
    @pytest.mark.asyncio
    async def test_get_optimization_metrics(self, optimizer):
        """Test optimization metrics retrieval"""
        # Add some test data
        optimizer.user_profiles['user1'] = UserBehaviorAnalysis(
            user_id="user1",
            behavior_pattern=UserBehaviorPattern.POWER_USER,
            engagement_score=0.8,
            frustration_indicators=[],
            preferred_features=[],
            usage_patterns={},
            assistance_needs=[],
            timestamp=datetime.now()
        )
        
        optimizer.user_profiles['user2'] = UserBehaviorAnalysis(
            user_id="user2",
            behavior_pattern=UserBehaviorPattern.NEW_USER,
            engagement_score=0.5,
            frustration_indicators=['Multiple errors'],
            preferred_features=[],
            usage_patterns={},
            assistance_needs=[],
            timestamp=datetime.now()
        )
        
        metrics = await optimizer.get_optimization_metrics()
        
        assert isinstance(metrics, dict)
        assert 'total_users_analyzed' in metrics
        assert 'behavior_patterns' in metrics
        assert 'average_engagement_score' in metrics
        assert 'common_frustration_indicators' in metrics
        
        assert metrics['total_users_analyzed'] == 2
        assert 'power_user' in metrics['behavior_patterns']
        assert 'new_user' in metrics['behavior_patterns']
        assert metrics['average_engagement_score'] == 0.65  # (0.8 + 0.5) / 2

@pytest.mark.asyncio
async def test_integration_workflow():
    """Test complete AI UX optimization workflow"""
    optimizer = AIUXOptimizer()
    await asyncio.sleep(0.1)  # Wait for initialization
    
    user_id = "integration_test_user"
    
    # Step 1: Record system metrics and predict failures
    system_metrics = {
        'cpu_usage': 0.8,
        'memory_usage': 0.7,
        'error_rate': 0.03,
        'response_time': 1200,
        'active_users': 200
    }
    
    predictions = await optimizer.predict_failures(system_metrics)
    assert isinstance(predictions, list)
    
    # Step 2: Analyze user behavior
    interaction_data = {
        'session_duration': 20,
        'clicks_per_minute': 10,
        'pages_visited': 4,
        'errors_encountered': 2,
        'help_requests': 1,
        'features_used': 3,
        'features_used_list': ['dashboard', 'search', 'analytics']
    }
    
    behavior_analysis = await optimizer.analyze_user_behavior(user_id, interaction_data)
    assert behavior_analysis.user_id == user_id
    assert isinstance(behavior_analysis.behavior_pattern, UserBehaviorPattern)
    
    # Step 3: Create personalized degradation strategy
    system_conditions = {'system_load': 0.7}
    degradation = await optimizer.create_personalized_degradation(user_id, system_conditions)
    assert degradation.user_id == user_id
    assert isinstance(degradation.strategy, DegradationStrategy)
    
    # Step 4: Optimize interface
    current_interface = {'layout': 'standard', 'theme': 'light'}
    optimization = await optimizer.optimize_interface(user_id, current_interface)
    assert optimization.user_id == user_id
    assert len(optimization.optimization_suggestions) > 0
    
    # Step 5: Get overall metrics
    metrics = await optimizer.get_optimization_metrics()
    assert metrics['total_users_analyzed'] >= 1
    assert user_id in optimizer.user_profiles

if __name__ == "__main__":
    pytest.main([__file__])