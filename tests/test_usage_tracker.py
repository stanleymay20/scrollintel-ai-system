"""
Tests for usage tracking system.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from scrollintel.engines.usage_tracker import UsageTracker
from scrollintel.models.usage_tracking_models import (
    GenerationType, ResourceType, GenerationUsage, UserUsageSummary,
    BudgetAlert, UsageForecast, CostOptimizationRecommendation
)


class TestUsageTracker:
    """Test suite for UsageTracker."""
    
    @pytest.fixture
    def tracker(self):
        """Create a UsageTracker instance for testing."""
        return UsageTracker()
    
    @pytest.fixture
    def sample_user_id(self):
        """Sample user ID for testing."""
        return "test_user_123"
    
    @pytest.mark.asyncio
    async def test_start_generation_tracking(self, tracker, sample_user_id):
        """Test starting generation tracking."""
        session_id = await tracker.start_generation_tracking(
            user_id=sample_user_id,
            generation_type=GenerationType.IMAGE,
            model_used="stable_diffusion_xl",
            prompt="A beautiful landscape",
            parameters={"resolution": (1024, 1024), "steps": 50}
        )
        
        assert session_id is not None
        assert session_id in tracker.active_sessions
        
        session = tracker.active_sessions[session_id]
        assert session.user_id == sample_user_id
        assert session.generation_type == GenerationType.IMAGE
        assert session.model_used == "stable_diffusion_xl"
        assert session.prompt == "A beautiful landscape"
        assert session.parameters["resolution"] == (1024, 1024)
        assert session.start_time is not None
    
    @pytest.mark.asyncio
    async def test_track_resource_usage(self, tracker, sample_user_id):
        """Test tracking resource usage."""
        # Start tracking
        session_id = await tracker.start_generation_tracking(
            user_id=sample_user_id,
            generation_type=GenerationType.IMAGE,
            model_used="stable_diffusion_xl",
            prompt="Test prompt"
        )
        
        # Track GPU usage
        await tracker.track_resource_usage(
            session_id=session_id,
            resource_type=ResourceType.GPU_SECONDS,
            amount=10.5,
            metadata={"gpu_type": "A100"}
        )
        
        # Track API calls
        await tracker.track_resource_usage(
            session_id=session_id,
            resource_type=ResourceType.API_CALLS,
            amount=1,
            metadata={"endpoint": "/generate"}
        )
        
        session = tracker.active_sessions[session_id]
        assert len(session.resources) == 2
        
        gpu_resource = session.resources[0]
        assert gpu_resource.resource_type == ResourceType.GPU_SECONDS
        assert gpu_resource.amount == 10.5
        assert gpu_resource.unit_cost == 0.05  # From cost_rates
        assert gpu_resource.total_cost == 10.5 * 0.05
        
        api_resource = session.resources[1]
        assert api_resource.resource_type == ResourceType.API_CALLS
        assert api_resource.amount == 1
        assert api_resource.unit_cost == 0.001
        
        # Check total cost calculation
        expected_total = (10.5 * 0.05) + (1 * 0.001)
        assert abs(session.total_cost - expected_total) < 0.0001
    
    @pytest.mark.asyncio
    async def test_end_generation_tracking(self, tracker, sample_user_id):
        """Test ending generation tracking."""
        # Start tracking
        session_id = await tracker.start_generation_tracking(
            user_id=sample_user_id,
            generation_type=GenerationType.IMAGE,
            model_used="dalle3",
            prompt="Test prompt"
        )
        
        # Add some resource usage
        await tracker.track_resource_usage(
            session_id=session_id,
            resource_type=ResourceType.GPU_SECONDS,
            amount=5.0
        )
        
        # End tracking
        completed_session = await tracker.end_generation_tracking(
            session_id=session_id,
            success=True,
            quality_score=0.85,
            error_message=None
        )
        
        assert completed_session.id == session_id
        assert completed_session.success is True
        assert completed_session.quality_score == 0.85
        assert completed_session.end_time is not None
        assert completed_session.duration_seconds > 0
        
        # Session should be moved to storage
        assert session_id not in tracker.active_sessions
        assert sample_user_id in tracker.storage
        assert len(tracker.storage[sample_user_id]) == 1
        assert tracker.storage[sample_user_id][0].id == session_id
    
    @pytest.mark.asyncio
    async def test_end_tracking_nonexistent_session(self, tracker):
        """Test ending tracking for non-existent session."""
        with pytest.raises(ValueError, match="Session .* not found"):
            await tracker.end_generation_tracking("nonexistent_session")
    
    @pytest.mark.asyncio
    async def test_get_user_usage_summary_empty(self, tracker, sample_user_id):
        """Test getting usage summary for user with no data."""
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=30)
        
        summary = await tracker.get_user_usage_summary(
            user_id=sample_user_id,
            start_date=start_date,
            end_date=end_date
        )
        
        assert summary.user_id == sample_user_id
        assert summary.total_generations == 0
        assert summary.total_cost == 0.0
        assert summary.average_cost_per_generation == 0.0
    
    @pytest.mark.asyncio
    async def test_get_user_usage_summary_with_data(self, tracker, sample_user_id):
        """Test getting usage summary with actual data."""
        # Create multiple completed sessions
        sessions_data = [
            (GenerationType.IMAGE, "stable_diffusion_xl", 5.0, True, 0.8),
            (GenerationType.VIDEO, "runway_ml", 15.0, True, 0.9),
            (GenerationType.IMAGE, "dalle3", 3.0, False, None),
            (GenerationType.ENHANCEMENT, "real_esrgan", 2.0, True, 0.7)
        ]
        
        for gen_type, model, gpu_time, success, quality in sessions_data:
            session_id = await tracker.start_generation_tracking(
                user_id=sample_user_id,
                generation_type=gen_type,
                model_used=model,
                prompt="Test prompt"
            )
            
            await tracker.track_resource_usage(
                session_id=session_id,
                resource_type=ResourceType.GPU_SECONDS,
                amount=gpu_time
            )
            
            await tracker.end_generation_tracking(
                session_id=session_id,
                success=success,
                quality_score=quality
            )
        
        # Get summary
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=30)
        
        summary = await tracker.get_user_usage_summary(
            user_id=sample_user_id,
            start_date=start_date,
            end_date=end_date
        )
        
        assert summary.total_generations == 4
        assert summary.successful_generations == 3
        assert summary.failed_generations == 1
        assert summary.image_generations == 2
        assert summary.video_generations == 1
        assert summary.enhancement_operations == 1
        assert summary.total_gpu_seconds == 25.0
        assert summary.total_cost == 25.0 * 0.05  # GPU cost
        assert summary.average_cost_per_generation == (25.0 * 0.05) / 4
        assert abs(summary.average_quality_score - 0.8) < 0.01  # (0.8 + 0.9 + 0.7) / 3
    
    @pytest.mark.asyncio
    async def test_check_budget_alerts(self, tracker, sample_user_id):
        """Test budget alert checking."""
        # Create a session that costs $1.00
        session_id = await tracker.start_generation_tracking(
            user_id=sample_user_id,
            generation_type=GenerationType.IMAGE,
            model_used="stable_diffusion_xl",
            prompt="Test prompt"
        )
        
        await tracker.track_resource_usage(
            session_id=session_id,
            resource_type=ResourceType.GPU_SECONDS,
            amount=20.0  # $1.00 at $0.05/second
        )
        
        await tracker.end_generation_tracking(session_id=session_id)
        
        # Check alerts with $1.50 budget (should trigger 50% and 75% alerts)
        alerts = await tracker.check_budget_alerts(
            user_id=sample_user_id,
            budget_limit=1.50,
            period_days=30
        )
        
        # Should have alerts for 50% and 75% thresholds
        assert len(alerts) >= 2
        
        alert_types = [alert.alert_type for alert in alerts]
        assert "budget_50.0percent" in alert_types
        assert "budget_75.0percent" in alert_types
        
        for alert in alerts:
            assert alert.user_id == sample_user_id
            assert alert.current_usage == 1.0
            assert alert.budget_limit == 1.50
    
    @pytest.mark.asyncio
    async def test_generate_usage_forecast_insufficient_data(self, tracker, sample_user_id):
        """Test forecast generation with insufficient data."""
        forecast = await tracker.generate_usage_forecast(
            user_id=sample_user_id,
            forecast_days=30,
            historical_days=90
        )
        
        assert forecast.user_id == sample_user_id
        assert forecast.forecast_period_days == 30
        assert forecast.predicted_cost == 0.0
        assert forecast.usage_trend == "insufficient_data"
    
    @pytest.mark.asyncio
    async def test_generate_cost_optimization_recommendations(self, tracker, sample_user_id):
        """Test cost optimization recommendations."""
        # Create sessions with high failure rate
        for i in range(10):
            session_id = await tracker.start_generation_tracking(
                user_id=sample_user_id,
                generation_type=GenerationType.IMAGE,
                model_used="unstable_model",
                prompt=f"Test prompt {i}"
            )
            
            await tracker.track_resource_usage(
                session_id=session_id,
                resource_type=ResourceType.GPU_SECONDS,
                amount=2.0
            )
            
            # Make 40% of them fail
            success = i < 6
            await tracker.end_generation_tracking(
                session_id=session_id,
                success=success,
                quality_score=0.5 if success else None
            )
        
        recommendations = await tracker.generate_cost_optimization_recommendations(
            user_id=sample_user_id,
            analysis_days=30
        )
        
        assert len(recommendations) > 0
        
        # Should have recommendation about reducing failures
        failure_rec = next(
            (rec for rec in recommendations if rec.recommendation_type == "reduce_failures"),
            None
        )
        assert failure_rec is not None
        assert failure_rec.potential_savings > 0
        assert failure_rec.priority == "high"
    
    @pytest.mark.asyncio
    async def test_real_time_cost_calculation_image(self, tracker):
        """Test real-time cost calculation for images."""
        cost_estimate = await tracker.get_real_time_cost_calculation(
            generation_type=GenerationType.IMAGE,
            model_name="stable_diffusion_xl",
            parameters={
                "resolution": (2048, 2048),  # 4x resolution
                "steps": 100  # 2x steps
            }
        )
        
        assert cost_estimate["base_cost"] == 0.02
        assert cost_estimate["multiplier"] == 4.0 * 2.0  # 4x resolution * 2x steps
        assert cost_estimate["estimated_cost"] == 0.02 * 8.0
        assert cost_estimate["currency"] == "USD"
    
    @pytest.mark.asyncio
    async def test_real_time_cost_calculation_video(self, tracker):
        """Test real-time cost calculation for videos."""
        cost_estimate = await tracker.get_real_time_cost_calculation(
            generation_type=GenerationType.VIDEO,
            model_name="runway_ml",
            parameters={
                "duration": 10.0,  # 2x duration
                "fps": 30,  # 1.25x fps
                "resolution": (1920, 1080)  # ~1.5x resolution
            }
        )
        
        assert cost_estimate["base_cost"] == 0.50
        # Multiplier should account for increased computation
        assert cost_estimate["multiplier"] > 1.0
        assert cost_estimate["estimated_cost"] > 0.50
    
    @pytest.mark.asyncio
    async def test_get_usage_analytics(self, tracker, sample_user_id):
        """Test comprehensive usage analytics."""
        # Create some test data
        session_id = await tracker.start_generation_tracking(
            user_id=sample_user_id,
            generation_type=GenerationType.IMAGE,
            model_used="stable_diffusion_xl",
            prompt="Test prompt"
        )
        
        await tracker.track_resource_usage(
            session_id=session_id,
            resource_type=ResourceType.GPU_SECONDS,
            amount=5.0
        )
        
        await tracker.end_generation_tracking(
            session_id=session_id,
            success=True,
            quality_score=0.8
        )
        
        analytics = await tracker.get_usage_analytics(
            user_id=sample_user_id,
            days=30
        )
        
        assert "summary" in analytics
        assert "forecast" in analytics
        assert "recommendations" in analytics
        assert "period" in analytics
        
        assert analytics["summary"].user_id == sample_user_id
        assert analytics["summary"].total_generations == 1
        assert analytics["summary"].total_cost == 0.25  # 5.0 * 0.05
    
    @pytest.mark.asyncio
    async def test_track_resource_nonexistent_session(self, tracker):
        """Test tracking resource for non-existent session."""
        # Should not raise exception, just log warning
        await tracker.track_resource_usage(
            session_id="nonexistent",
            resource_type=ResourceType.GPU_SECONDS,
            amount=1.0
        )
        # Test passes if no exception is raised
    
    @pytest.mark.asyncio
    async def test_cost_rates_initialization(self, tracker):
        """Test that cost rates are properly initialized."""
        assert ResourceType.GPU_SECONDS in tracker.cost_rates
        assert ResourceType.CPU_SECONDS in tracker.cost_rates
        assert ResourceType.STORAGE_GB in tracker.cost_rates
        assert ResourceType.BANDWIDTH_GB in tracker.cost_rates
        assert ResourceType.API_CALLS in tracker.cost_rates
        
        # All rates should be positive
        for rate in tracker.cost_rates.values():
            assert rate > 0
    
    @pytest.mark.asyncio
    async def test_multiple_resource_types_tracking(self, tracker, sample_user_id):
        """Test tracking multiple resource types in one session."""
        session_id = await tracker.start_generation_tracking(
            user_id=sample_user_id,
            generation_type=GenerationType.VIDEO,
            model_used="custom_video",
            prompt="Test video"
        )
        
        # Track multiple resource types
        resources = [
            (ResourceType.GPU_SECONDS, 30.0),
            (ResourceType.CPU_SECONDS, 120.0),
            (ResourceType.STORAGE_GB, 2.5),
            (ResourceType.BANDWIDTH_GB, 1.0),
            (ResourceType.API_CALLS, 5)
        ]
        
        for resource_type, amount in resources:
            await tracker.track_resource_usage(
                session_id=session_id,
                resource_type=resource_type,
                amount=amount
            )
        
        session = tracker.active_sessions[session_id]
        assert len(session.resources) == 5
        
        # Calculate expected total cost
        expected_cost = (
            30.0 * 0.05 +    # GPU
            120.0 * 0.001 +  # CPU
            2.5 * 0.02 +     # Storage
            1.0 * 0.01 +     # Bandwidth
            5 * 0.001        # API calls
        )
        
        assert abs(session.total_cost - expected_cost) < 0.0001


class TestUsageTrackerIntegration:
    """Integration tests for usage tracking system."""
    
    @pytest.mark.asyncio
    async def test_complete_generation_workflow(self):
        """Test complete generation workflow with usage tracking."""
        tracker = UsageTracker()
        user_id = "integration_test_user"
        
        # Start tracking
        session_id = await tracker.start_generation_tracking(
            user_id=user_id,
            generation_type=GenerationType.IMAGE,
            model_used="stable_diffusion_xl",
            prompt="A photorealistic portrait",
            parameters={"resolution": (1024, 1024), "steps": 50}
        )
        
        # Simulate generation process with resource tracking
        await tracker.track_resource_usage(
            session_id=session_id,
            resource_type=ResourceType.API_CALLS,
            amount=1,
            metadata={"endpoint": "/generate"}
        )
        
        await tracker.track_resource_usage(
            session_id=session_id,
            resource_type=ResourceType.GPU_SECONDS,
            amount=8.5,
            metadata={"gpu_type": "A100", "model_loading_time": 2.0}
        )
        
        await tracker.track_resource_usage(
            session_id=session_id,
            resource_type=ResourceType.STORAGE_GB,
            amount=0.1,
            metadata={"file_format": "png", "compression": "lossless"}
        )
        
        # End tracking
        completed_session = await tracker.end_generation_tracking(
            session_id=session_id,
            success=True,
            quality_score=0.92
        )
        
        # Verify session data
        assert completed_session.success is True
        assert completed_session.quality_score == 0.92
        assert len(completed_session.resources) == 3
        assert completed_session.total_cost > 0
        
        # Get usage summary
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=1)
        
        summary = await tracker.get_user_usage_summary(
            user_id=user_id,
            start_date=start_date,
            end_date=end_date
        )
        
        assert summary.total_generations == 1
        assert summary.successful_generations == 1
        assert summary.image_generations == 1
        assert summary.total_cost == completed_session.total_cost
        assert summary.average_quality_score == 0.92
    
    @pytest.mark.asyncio
    async def test_budget_monitoring_workflow(self):
        """Test budget monitoring and alerting workflow."""
        tracker = UsageTracker()
        user_id = "budget_test_user"
        budget_limit = 5.00  # $5 budget
        
        # Create multiple sessions to approach budget limit
        total_cost = 0
        for i in range(3):
            session_id = await tracker.start_generation_tracking(
                user_id=user_id,
                generation_type=GenerationType.IMAGE,
                model_used="dalle3",
                prompt=f"Test image {i}"
            )
            
            # Each session costs ~$1.50
            await tracker.track_resource_usage(
                session_id=session_id,
                resource_type=ResourceType.GPU_SECONDS,
                amount=30.0  # $1.50 at $0.05/second
            )
            
            session = await tracker.end_generation_tracking(session_id=session_id)
            total_cost += session.total_cost
        
        # Check budget alerts
        alerts = await tracker.check_budget_alerts(
            user_id=user_id,
            budget_limit=budget_limit,
            period_days=30
        )
        
        # Should have multiple alerts as we're at 90% of budget
        assert len(alerts) >= 3  # 50%, 75%, 90%
        
        # Verify alert details
        for alert in alerts:
            assert alert.user_id == user_id
            assert alert.budget_limit == budget_limit
            assert abs(alert.current_usage - total_cost) < 0.01
    
    @pytest.mark.asyncio
    async def test_cost_optimization_analysis(self):
        """Test cost optimization analysis workflow."""
        tracker = UsageTracker()
        user_id = "optimization_test_user"
        
        # Create sessions with different patterns for optimization analysis
        
        # High-cost, low-quality model usage
        for i in range(8):
            session_id = await tracker.start_generation_tracking(
                user_id=user_id,
                generation_type=GenerationType.IMAGE,
                model_used="expensive_low_quality_model",
                prompt=f"Test {i}"
            )
            
            await tracker.track_resource_usage(
                session_id=session_id,
                resource_type=ResourceType.GPU_SECONDS,
                amount=20.0  # High cost
            )
            
            await tracker.end_generation_tracking(
                session_id=session_id,
                success=True,
                quality_score=0.4  # Low quality
            )
        
        # Some failed generations
        for i in range(3):
            session_id = await tracker.start_generation_tracking(
                user_id=user_id,
                generation_type=GenerationType.IMAGE,
                model_used="unstable_model",
                prompt=f"Failed test {i}"
            )
            
            await tracker.track_resource_usage(
                session_id=session_id,
                resource_type=ResourceType.GPU_SECONDS,
                amount=5.0
            )
            
            await tracker.end_generation_tracking(
                session_id=session_id,
                success=False,
                error_message="Generation failed"
            )
        
        # Get optimization recommendations
        recommendations = await tracker.generate_cost_optimization_recommendations(
            user_id=user_id,
            analysis_days=30
        )
        
        assert len(recommendations) >= 2
        
        # Should have recommendations for both issues
        rec_types = [rec.recommendation_type for rec in recommendations]
        assert "reduce_failures" in rec_types
        assert "optimize_model_selection" in rec_types
        
        # Verify recommendation details
        for rec in recommendations:
            assert rec.user_id == user_id
            assert rec.potential_savings > 0
            assert rec.priority in ["low", "medium", "high"]