"""
Tests for prompt analytics system.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import json

from scrollintel.core.prompt_analytics import (
    PromptPerformanceTracker,
    AnalyticsEngine,
    PromptUsageEvent,
    PromptPerformanceSummary
)

class TestPromptPerformanceTracker:
    """Test prompt performance tracking functionality."""
    
    @pytest.fixture
    def tracker(self):
        """Create a fresh tracker instance for each test."""
        return PromptPerformanceTracker()
    
    @pytest.mark.asyncio
    async def test_record_prompt_usage_basic(self, tracker):
        """Test basic prompt usage recording."""
        event_id = await tracker.record_prompt_usage(
            prompt_id="test_prompt_001",
            user_id="user_123",
            team_id="team_456"
        )
        
        assert event_id is not None
        assert len(tracker.events_buffer) == 1
        
        event = tracker.events_buffer[0]
        assert event.prompt_id == "test_prompt_001"
        assert event.user_id == "user_123"
        assert event.team_id == "team_456"
        assert event.success is True
    
    @pytest.mark.asyncio
    async def test_record_prompt_usage_with_metrics(self, tracker):
        """Test recording prompt usage with performance metrics."""
        performance_metrics = {
            'accuracy_score': 0.95,
            'relevance_score': 0.88,
            'response_time_ms': 1500,
            'token_usage': 250,
            'cost_per_request': 0.005
        }
        
        context = {
            'use_case': 'content_generation',
            'model_used': 'gpt-4',
            'success': True
        }
        
        event_id = await tracker.record_prompt_usage(
            prompt_id="test_prompt_002",
            performance_metrics=performance_metrics,
            context=context
        )
        
        assert event_id is not None
        event = tracker.events_buffer[0]
        
        assert event.accuracy_score == 0.95
        assert event.relevance_score == 0.88
        assert event.response_time_ms == 1500
        assert event.token_usage == 250
        assert event.cost_per_request == 0.005
        assert event.use_case == 'content_generation'
        assert event.model_used == 'gpt-4'
    
    @pytest.mark.asyncio
    async def test_real_time_metrics_update(self, tracker):
        """Test real-time metrics updates."""
        # Record multiple events for the same prompt
        for i in range(5):
            await tracker.record_prompt_usage(
                prompt_id="test_prompt_003",
                performance_metrics={
                    'accuracy_score': 0.9 + (i * 0.01),
                    'response_time_ms': 1000 + (i * 100)
                }
            )
        
        # Get real-time metrics
        metrics = await tracker.get_real_time_metrics("test_prompt_003")
        
        assert metrics['total_requests'] == 5
        assert metrics['successful_requests'] == 5
        assert metrics['success_rate'] == 100.0
        assert metrics['avg_response_time'] is not None
        assert metrics['avg_accuracy_score'] is not None
    
    @pytest.mark.asyncio
    async def test_performance_summary_generation(self, tracker):
        """Test performance summary generation."""
        # Add sample data
        base_date = datetime.utcnow() - timedelta(days=5)
        
        for i in range(10):
            event = PromptUsageEvent(
                event_id=f"event_{i}",
                prompt_id="test_prompt_004",
                version_id="v1",
                user_id=f"user_{i % 3}",
                team_id="team_test",
                accuracy_score=0.85 + (i * 0.01),
                response_time_ms=1200 + (i * 50),
                token_usage=200 + (i * 10),
                cost_per_request=0.004 + (i * 0.0001),
                timestamp=base_date + timedelta(hours=i * 2)
            )
            tracker.events_buffer.append(event)
        
        # Generate summary
        summary = await tracker.get_prompt_performance_summary(
            prompt_id="test_prompt_004",
            days=7
        )
        
        assert 'error' not in summary
        assert summary['total_requests'] == 10
        assert summary['successful_requests'] == 10
        assert summary['unique_users'] == 3
        assert summary['avg_accuracy_score'] is not None
        assert summary['avg_response_time_ms'] is not None
        assert summary['total_cost'] is not None
    
    @pytest.mark.asyncio
    async def test_team_analytics_generation(self, tracker):
        """Test team analytics generation."""
        # Add sample data for team
        base_date = datetime.utcnow() - timedelta(days=3)
        
        prompts = ["prompt_a", "prompt_b", "prompt_c"]
        users = ["user_1", "user_2", "user_3", "user_4"]
        
        for i in range(20):
            event = PromptUsageEvent(
                event_id=f"team_event_{i}",
                prompt_id=prompts[i % len(prompts)],
                version_id="v1",
                user_id=users[i % len(users)],
                team_id="analytics_team",
                accuracy_score=0.8 + (i % 10) * 0.02,
                response_time_ms=1000 + (i % 5) * 200,
                cost_per_request=0.003 + (i % 3) * 0.001,
                timestamp=base_date + timedelta(hours=i)
            )
            tracker.events_buffer.append(event)
        
        # Generate team analytics
        analytics = await tracker.get_team_analytics(
            team_id="analytics_team",
            days=5
        )
        
        assert 'error' not in analytics
        assert analytics['summary']['total_requests'] == 20
        assert analytics['summary']['unique_prompts'] == 3
        assert analytics['summary']['unique_users'] == 4
        assert len(analytics['top_prompts']) > 0
        assert len(analytics['top_users']) > 0
    
    @pytest.mark.asyncio
    async def test_trend_calculation(self, tracker):
        """Test trend calculation functionality."""
        # Test improving trend
        improving_values = [10, 12, 15, 18, 20, 22, 25]
        trend = tracker._calculate_trend(improving_values)
        assert trend == 'improving'
        
        # Test declining trend
        declining_values = [25, 22, 20, 18, 15, 12, 10]
        trend = tracker._calculate_trend(declining_values)
        assert trend == 'declining'
        
        # Test stable trend
        stable_values = [20, 19, 21, 20, 19, 20, 21]
        trend = tracker._calculate_trend(stable_values)
        assert trend == 'stable'
    
    @pytest.mark.asyncio
    async def test_error_handling(self, tracker):
        """Test error handling in analytics functions."""
        # Test with non-existent prompt
        summary = await tracker.get_prompt_performance_summary(
            prompt_id="non_existent_prompt",
            days=30
        )
        
        assert summary['total_requests'] == 0
        assert 'message' in summary
        
        # Test with non-existent team
        analytics = await tracker.get_team_analytics(
            team_id="non_existent_team",
            days=30
        )
        
        assert analytics.get('summary', {}).get('total_requests', 0) == 0
        assert 'message' in analytics

class TestAnalyticsEngine:
    """Test analytics engine functionality."""
    
    @pytest.fixture
    def engine(self):
        """Create a fresh analytics engine for each test."""
        return AnalyticsEngine()
    
    @pytest.mark.asyncio
    async def test_performance_insights_generation(self, engine):
        """Test performance insights generation."""
        # Mock the performance tracker
        with patch.object(engine.performance_tracker, 'get_prompt_performance_summary') as mock_summary:
            mock_summary.return_value = {
                'prompt_id': 'test_prompt',
                'total_requests': 100,
                'success_rate': 85.0,  # Below optimal
                'avg_response_time_ms': 6000,  # High response time
                'avg_accuracy_score': 0.75,  # Declining accuracy
                'accuracy_trend': 'declining',
                'total_cost': 15.50,
                'avg_cost_per_request': 0.155
            }
            
            insights = await engine.generate_performance_insights(
                prompt_id="test_prompt",
                days=30
            )
            
            assert 'error' not in insights
            assert len(insights['insights']) > 0
            assert len(insights['recommendations']) > 0
            
            # Check for specific insights based on mock data
            insight_texts = [insight for insight in insights['insights']]
            assert any('Success rate is 85.0%' in str(insight) for insight in insight_texts)
    
    @pytest.mark.asyncio
    async def test_prompt_version_comparison(self, engine):
        """Test prompt version comparison functionality."""
        # Mock performance summaries for different versions
        with patch.object(engine.performance_tracker, 'get_prompt_performance_summary') as mock_summary:
            def mock_summary_side_effect(prompt_id, version_id, days):
                if version_id == "v1":
                    return {
                        'prompt_id': prompt_id,
                        'version_id': version_id,
                        'avg_accuracy_score': 0.80,
                        'success_rate': 90.0,
                        'avg_response_time_ms': 2000
                    }
                elif version_id == "v2":
                    return {
                        'prompt_id': prompt_id,
                        'version_id': version_id,
                        'avg_accuracy_score': 0.85,
                        'success_rate': 95.0,
                        'avg_response_time_ms': 1800
                    }
                else:
                    return {'error': 'Version not found'}
            
            mock_summary.side_effect = mock_summary_side_effect
            
            comparison = await engine.compare_prompt_versions(
                prompt_id="test_prompt",
                version_ids=["v1", "v2"],
                days=30
            )
            
            assert 'error' not in comparison
            assert comparison['best_version'] == "v2"  # v2 should score higher
            assert len(comparison['version_comparisons']) == 2
    
    @pytest.mark.asyncio
    async def test_insights_with_no_data(self, engine):
        """Test insights generation when no data is available."""
        with patch.object(engine.performance_tracker, 'get_prompt_performance_summary') as mock_summary:
            mock_summary.return_value = {'error': 'No data found'}
            
            insights = await engine.generate_performance_insights(
                prompt_id="empty_prompt",
                days=30
            )
            
            assert 'error' in insights

class TestPromptUsageEvent:
    """Test PromptUsageEvent data class."""
    
    def test_event_creation_basic(self):
        """Test basic event creation."""
        event = PromptUsageEvent(
            event_id="test_001",
            prompt_id="prompt_123",
            version_id="v1",
            user_id="user_456",
            team_id="team_789"
        )
        
        assert event.event_id == "test_001"
        assert event.prompt_id == "prompt_123"
        assert event.version_id == "v1"
        assert event.user_id == "user_456"
        assert event.team_id == "team_789"
        assert event.success is True  # Default value
        assert event.timestamp is not None  # Auto-generated
    
    def test_event_creation_with_metrics(self):
        """Test event creation with performance metrics."""
        event = PromptUsageEvent(
            event_id="test_002",
            prompt_id="prompt_456",
            version_id="v1",
            user_id="user_123",
            team_id="team_456",
            accuracy_score=0.92,
            relevance_score=0.88,
            efficiency_score=0.95,
            user_satisfaction=0.90,
            response_time_ms=1200,
            token_usage=180,
            cost_per_request=0.0045
        )
        
        assert event.accuracy_score == 0.92
        assert event.relevance_score == 0.88
        assert event.efficiency_score == 0.95
        assert event.user_satisfaction == 0.90
        assert event.response_time_ms == 1200
        assert event.token_usage == 180
        assert event.cost_per_request == 0.0045
    
    def test_event_timestamp_auto_generation(self):
        """Test automatic timestamp generation."""
        before = datetime.utcnow()
        
        event = PromptUsageEvent(
            event_id="test_003",
            prompt_id="prompt_789",
            version_id="v1",
            user_id="user_123",
            team_id="team_456"
        )
        
        after = datetime.utcnow()
        
        assert before <= event.timestamp <= after

class TestIntegration:
    """Integration tests for the analytics system."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_analytics_workflow(self):
        """Test complete analytics workflow from recording to insights."""
        tracker = PromptPerformanceTracker()
        engine = AnalyticsEngine()
        
        # Record multiple usage events
        for i in range(10):
            await tracker.record_prompt_usage(
                prompt_id="integration_test_prompt",
                user_id=f"user_{i % 3}",
                team_id="integration_team",
                performance_metrics={
                    'accuracy_score': 0.85 + (i * 0.01),
                    'response_time_ms': 1500 + (i * 50),
                    'cost_per_request': 0.005
                }
            )
        
        # Get performance summary
        summary = await tracker.get_prompt_performance_summary(
            prompt_id="integration_test_prompt",
            days=1
        )
        
        assert summary['total_requests'] == 10
        assert summary['avg_accuracy_score'] > 0.85
        
        # Get team analytics
        team_analytics = await tracker.get_team_analytics(
            team_id="integration_team",
            days=1
        )
        
        assert team_analytics['summary']['total_requests'] == 10
        assert team_analytics['summary']['unique_users'] == 3
        
        # Generate insights
        insights = await engine.generate_performance_insights(
            prompt_id="integration_test_prompt",
            days=1
        )
        
        assert 'insights' in insights
        assert 'recommendations' in insights
    
    @pytest.mark.asyncio
    async def test_concurrent_usage_recording(self):
        """Test concurrent usage recording."""
        tracker = PromptPerformanceTracker()
        
        # Create multiple concurrent recording tasks
        tasks = []
        for i in range(20):
            task = tracker.record_prompt_usage(
                prompt_id=f"concurrent_prompt_{i % 5}",
                user_id=f"user_{i}",
                team_id="concurrent_team"
            )
            tasks.append(task)
        
        # Wait for all tasks to complete
        event_ids = await asyncio.gather(*tasks)
        
        # Verify all events were recorded
        assert len(event_ids) == 20
        assert len(tracker.events_buffer) == 20
        assert all(event_id is not None for event_id in event_ids)
    
    @pytest.mark.asyncio
    async def test_large_dataset_performance(self):
        """Test performance with large datasets."""
        tracker = PromptPerformanceTracker()
        
        # Record a large number of events
        start_time = datetime.utcnow()
        
        for i in range(1000):
            await tracker.record_prompt_usage(
                prompt_id=f"perf_prompt_{i % 10}",
                user_id=f"user_{i % 50}",
                team_id=f"team_{i % 5}",
                performance_metrics={
                    'accuracy_score': 0.8 + (i % 20) * 0.01,
                    'response_time_ms': 1000 + (i % 100) * 10
                }
            )
        
        recording_time = (datetime.utcnow() - start_time).total_seconds()
        
        # Generate analytics
        analytics_start = datetime.utcnow()
        
        summary = await tracker.get_prompt_performance_summary(
            prompt_id="perf_prompt_0",
            days=1
        )
        
        analytics_time = (datetime.utcnow() - analytics_start).total_seconds()
        
        # Verify performance is reasonable
        assert recording_time < 10.0  # Should complete within 10 seconds
        assert analytics_time < 5.0   # Analytics should be fast
        assert summary['total_requests'] == 100  # 1000 events / 10 prompts

if __name__ == "__main__":
    pytest.main([__file__])