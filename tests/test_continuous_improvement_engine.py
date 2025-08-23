"""
Tests for Continuous Improvement Engine

This module tests the continuous improvement system including feedback collection,
A/B testing, model retraining, and feature enhancement processes.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
from sqlalchemy.orm import Session

from scrollintel.engines.continuous_improvement_engine import ContinuousImprovementEngine
from scrollintel.models.continuous_improvement_models import (
    UserFeedback, ABTest, ABTestResult, ModelRetrainingJob, FeatureEnhancement,
    FeedbackType, FeedbackPriority, ABTestStatus, ModelRetrainingStatus,
    FeatureEnhancementStatus
)

class TestContinuousImprovementEngine:
    """Test suite for continuous improvement engine"""
    
    @pytest.fixture
    def engine(self):
        """Create improvement engine instance"""
        return ContinuousImprovementEngine()
    
    @pytest.fixture
    def mock_db(self):
        """Create mock database session"""
        return Mock(spec=Session)
    
    @pytest.fixture
    def sample_feedback_data(self):
        """Sample feedback data for testing"""
        return {
            "feedback_type": FeedbackType.USER_SATISFACTION,
            "priority": FeedbackPriority.MEDIUM,
            "title": "Improve dashboard loading speed",
            "description": "The dashboard takes too long to load with large datasets",
            "context": {"page": "dashboard", "dataset_size": "large"},
            "satisfaction_rating": 6,
            "feature_area": "dashboard"
        }
    
    @pytest.fixture
    def sample_ab_test_config(self):
        """Sample A/B test configuration"""
        return {
            "name": "dashboard_optimization_test",
            "description": "Test dashboard loading optimizations",
            "hypothesis": "Optimized dashboard will improve user satisfaction",
            "feature_area": "dashboard",
            "control_config": {"optimization": False},
            "variant_configs": [{"optimization": True}],
            "traffic_allocation": {"control": 0.5, "variant_1": 0.5},
            "primary_metric": "satisfaction_rating",
            "secondary_metrics": ["session_duration", "bounce_rate"],
            "minimum_sample_size": 1000,
            "confidence_level": 0.95
        }
    
    @pytest.fixture
    def sample_model_config(self):
        """Sample model retraining configuration"""
        return {
            "model_name": "user_satisfaction_predictor",
            "model_version": "v2.0",
            "agent_type": "bi_agent",
            "training_config": {
                "algorithm": "random_forest",
                "hyperparameters": {"n_estimators": 100, "max_depth": 10}
            },
            "data_sources": [
                {"type": "feedback", "table": "user_feedback"},
                {"type": "metrics", "table": "user_metrics"}
            ],
            "performance_threshold": 0.85,
            "scheduled_at": datetime.utcnow() + timedelta(hours=1)
        }
    
    @pytest.mark.asyncio
    async def test_collect_user_feedback(self, engine, mock_db, sample_feedback_data):
        """Test user feedback collection"""
        # Mock database operations
        mock_feedback = Mock(spec=UserFeedback)
        mock_feedback.id = 1
        mock_db.add.return_value = None
        mock_db.commit.return_value = None
        mock_db.refresh.return_value = None
        
        # Mock business impact calculation
        with patch.object(engine, '_calculate_business_impact', return_value=7.5):
            with patch.object(engine, '_analyze_feedback_patterns', return_value=None):
                result = await engine.collect_user_feedback(
                    user_id="user123",
                    feedback_data=sample_feedback_data,
                    db=mock_db
                )
        
        # Verify database operations
        mock_db.add.assert_called_once()
        mock_db.commit.assert_called_once()
        
        assert result is not None
    
    @pytest.mark.asyncio
    async def test_create_ab_test(self, engine, mock_db, sample_ab_test_config):
        """Test A/B test creation"""
        # Mock database operations
        mock_test = Mock(spec=ABTest)
        mock_test.id = 1
        mock_db.add.return_value = None
        mock_db.commit.return_value = None
        mock_db.refresh.return_value = None
        
        # Mock validation
        with patch.object(engine, '_validate_ab_test_config', return_value=None):
            result = await engine.create_ab_test(
                test_config=sample_ab_test_config,
                db=mock_db
            )
        
        # Verify database operations
        mock_db.add.assert_called_once()
        mock_db.commit.assert_called_once()
        
        assert result is not None
    
    @pytest.mark.asyncio
    async def test_start_ab_test(self, engine, mock_db):
        """Test A/B test startup"""
        # Mock existing test
        mock_test = Mock(spec=ABTest)
        mock_test.id = 1
        mock_test.status = ABTestStatus.DRAFT
        mock_test.feature_area = "dashboard"
        mock_test.minimum_sample_size = 1000
        
        mock_db.query.return_value.filter.return_value.first.return_value = mock_test
        mock_db.commit.return_value = None
        
        # Mock daily users estimation
        with patch.object(engine, '_estimate_daily_users', return_value=500):
            with patch.object(engine, '_initialize_ab_test_monitoring', return_value=None):
                result = await engine.start_ab_test(test_id=1, db=mock_db)
        
        assert result is True
        assert mock_test.status == ABTestStatus.RUNNING
        assert mock_test.start_date is not None
    
    @pytest.mark.asyncio
    async def test_record_ab_test_result(self, engine, mock_db):
        """Test A/B test result recording"""
        # Mock active test
        mock_test = Mock(spec=ABTest)
        mock_test.id = 1
        mock_test.primary_metric = "satisfaction_rating"
        
        mock_db.query.return_value.filter.return_value.first.return_value = mock_test
        
        # Mock result creation
        mock_result = Mock(spec=ABTestResult)
        mock_result.id = 1
        mock_db.add.return_value = None
        mock_db.commit.return_value = None
        mock_db.refresh.return_value = None
        
        metrics = {
            "satisfaction_rating": 8.5,
            "session_duration": 300,
            "conversion_event": True,
            "business_value_generated": 150.0
        }
        
        with patch.object(engine, '_check_ab_test_completion', return_value=None):
            result = await engine.record_ab_test_result(
                test_id=1,
                user_id="user123",
                variant_name="variant_1",
                metrics=metrics,
                db=mock_db
            )
        
        # Verify database operations
        mock_db.add.assert_called_once()
        mock_db.commit.assert_called_once()
        
        assert result is not None
    
    @pytest.mark.asyncio
    async def test_analyze_ab_test_results(self, engine, mock_db):
        """Test A/B test results analysis"""
        # Mock test and results
        mock_test = Mock(spec=ABTest)
        mock_test.id = 1
        mock_test.minimum_sample_size = 100
        mock_test.primary_metric = "satisfaction_rating"
        mock_test.confidence_level = 0.95
        
        mock_results = []
        for i in range(150):  # Sufficient sample size
            result = Mock(spec=ABTestResult)
            result.variant_name = "control" if i < 75 else "variant_1"
            result.primary_metric_value = 7.0 if i < 75 else 8.0
            result.conversion_event = i % 3 == 0
            result.user_satisfaction = 7 if i < 75 else 8
            result.business_value_generated = 100.0 if i < 75 else 120.0
            mock_results.append(result)
        
        mock_db.query.return_value.filter.return_value.first.return_value = mock_test
        mock_db.query.return_value.filter.return_value.all.return_value = mock_results
        
        # Mock analysis methods
        with patch.object(engine, '_perform_statistical_analysis') as mock_stats:
            mock_stats.return_value = {
                "variant_1": {
                    "sample_size": 75,
                    "mean": 8.0,
                    "control_mean": 7.0,
                    "lift": 0.143,
                    "p_value": 0.01,
                    "significant": True
                }
            }
            
            with patch.object(engine, '_calculate_ab_test_business_impact') as mock_impact:
                mock_impact.return_value = {"revenue_increase": 15000}
                
                with patch.object(engine, '_generate_ab_test_recommendations') as mock_recs:
                    mock_recs.return_value = ["Deploy variant_1 to all users"]
                    
                    result = await engine.analyze_ab_test_results(test_id=1, db=mock_db)
        
        assert result["status"] == "completed"
        assert result["sample_size"] == 150
        assert "statistical_results" in result
        assert "business_impact" in result
        assert "recommendations" in result
    
    @pytest.mark.asyncio
    async def test_schedule_model_retraining(self, engine, mock_db, sample_model_config):
        """Test model retraining scheduling"""
        # Mock database operations
        mock_job = Mock(spec=ModelRetrainingJob)
        mock_job.id = 1
        mock_db.add.return_value = None
        mock_db.commit.return_value = None
        mock_db.refresh.return_value = None
        
        # Mock validation and baseline metrics
        with patch.object(engine, '_validate_model_config', return_value=None):
            with patch.object(engine, '_get_model_baseline_metrics') as mock_baseline:
                mock_baseline.return_value = {"accuracy": 0.82, "f1_score": 0.78}
                
                with patch.object(engine, '_schedule_retraining_execution', return_value=None):
                    result = await engine.schedule_model_retraining(
                        model_config=sample_model_config,
                        db=mock_db
                    )
        
        # Verify database operations
        mock_db.add.assert_called_once()
        mock_db.commit.assert_called_once()
        
        assert result is not None
    
    @pytest.mark.asyncio
    async def test_execute_model_retraining(self, engine, mock_db):
        """Test model retraining execution"""
        # Mock retraining job
        mock_job = Mock(spec=ModelRetrainingJob)
        mock_job.id = 1
        mock_job.model_name = "test_model"
        mock_job.baseline_metrics = {"accuracy": 0.82}
        
        mock_db.query.return_value.filter.return_value.first.return_value = mock_job
        mock_db.commit.return_value = None
        
        # Mock training pipeline
        with patch.object(engine, '_prepare_training_data') as mock_data:
            mock_data.return_value = {"features": [], "labels": []}
            
            with patch.object(engine, '_train_model_with_business_feedback') as mock_train:
                mock_train.return_value = {"artifacts_path": "/models/test_model_v2"}
                
                with patch.object(engine, '_evaluate_model_performance') as mock_eval:
                    mock_eval.return_value = {
                        "metrics": {"accuracy": 0.87, "f1_score": 0.84},
                        "business_impact": {"cost_savings": 25000}
                    }
                    
                    with patch.object(engine, '_calculate_improvement_percentage') as mock_imp:
                        mock_imp.return_value = 6.1  # 6.1% improvement
                        
                        result = await engine.execute_model_retraining(job_id=1, db=mock_db)
        
        assert result["job_id"] == 1
        assert result["improvement_percentage"] == 6.1
        assert "performance_metrics" in result
        assert "business_impact" in result
        assert mock_job.status == ModelRetrainingStatus.COMPLETED
    
    @pytest.mark.asyncio
    async def test_create_feature_enhancement(self, engine, mock_db):
        """Test feature enhancement creation"""
        enhancement_data = {
            "title": "Add real-time notifications",
            "description": "Implement real-time notifications for critical alerts",
            "feature_area": "notifications",
            "enhancement_type": "new_feature",
            "priority": FeedbackPriority.HIGH,
            "complexity_score": 7,
            "business_value_score": 8.5,
            "user_impact_score": 9.0,
            "technical_feasibility_score": 7.5,
            "estimated_effort_hours": 120,
            "expected_roi": 2.5,
            "requirements": ["Real-time push notifications", "Email notifications"],
            "acceptance_criteria": ["Notifications delivered within 5 seconds"]
        }
        
        # Mock database operations
        mock_enhancement = Mock(spec=FeatureEnhancement)
        mock_enhancement.id = 1
        mock_db.add.return_value = None
        mock_db.commit.return_value = None
        mock_db.refresh.return_value = None
        
        # Mock priority calculation and review trigger
        with patch.object(engine, '_calculate_enhancement_priority', return_value=8.0):
            with patch.object(engine, '_trigger_enhancement_review', return_value=None):
                result = await engine.create_feature_enhancement(
                    requester_id="user123",
                    enhancement_data=enhancement_data,
                    db=mock_db
                )
        
        # Verify database operations
        mock_db.add.assert_called_once()
        mock_db.commit.assert_called_once()
        
        assert result is not None
    
    @pytest.mark.asyncio
    async def test_generate_improvement_recommendations(self, engine, mock_db):
        """Test improvement recommendations generation"""
        # Mock analysis methods
        with patch.object(engine, '_analyze_feedback_patterns_comprehensive') as mock_feedback:
            mock_feedback.return_value = {"satisfaction_trend": "declining"}
            
            with patch.object(engine, '_analyze_ab_test_trends') as mock_ab:
                mock_ab.return_value = {"successful_tests": 3}
                
                with patch.object(engine, '_analyze_model_performance_trends') as mock_model:
                    mock_model.return_value = {"accuracy_trend": "stable"}
                    
                    with patch.object(engine, '_analyze_feature_adoption') as mock_feature:
                        mock_feature.return_value = {"adoption_rate": 0.75}
                        
                        # Mock recommendation generators
                        with patch.object(engine, '_generate_performance_recommendations') as mock_perf:
                            mock_perf.return_value = []
                            
                            with patch.object(engine, '_generate_ux_recommendations') as mock_ux:
                                mock_ux.return_value = []
                                
                                with patch.object(engine, '_generate_business_recommendations') as mock_biz:
                                    mock_biz.return_value = []
                                    
                                    result = await engine.generate_improvement_recommendations(
                                        db=mock_db,
                                        time_window_days=30
                                    )
        
        assert isinstance(result, list)
    
    @pytest.mark.asyncio
    async def test_get_improvement_metrics(self, engine, mock_db):
        """Test improvement metrics calculation"""
        # Mock all metric calculation methods
        with patch.object(engine, '_calculate_feedback_metrics') as mock_feedback:
            mock_feedback.return_value = {"total_feedback": 150}
            
            with patch.object(engine, '_calculate_ab_test_metrics') as mock_ab:
                mock_ab.return_value = {"active_tests": 3}
                
                with patch.object(engine, '_calculate_model_performance_metrics') as mock_model:
                    mock_model.return_value = {"avg_accuracy": 0.85}
                    
                    with patch.object(engine, '_calculate_feature_adoption_metrics') as mock_feature:
                        mock_feature.return_value = {"adoption_rate": 0.78}
                        
                        with patch.object(engine, '_calculate_business_impact_metrics') as mock_business:
                            mock_business.return_value = {"total_savings": 50000}
                            
                            with patch.object(engine, '_calculate_satisfaction_trends') as mock_satisfaction:
                                mock_satisfaction.return_value = []
                                
                                with patch.object(engine, '_calculate_reliability_trends') as mock_reliability:
                                    mock_reliability.return_value = []
                                    
                                    result = await engine.get_improvement_metrics(
                                        db=mock_db,
                                        time_window_days=30
                                    )
        
        assert result.feedback_metrics["total_feedback"] == 150
        assert result.ab_test_metrics["active_tests"] == 3
        assert result.model_performance_metrics["avg_accuracy"] == 0.85
    
    @pytest.mark.asyncio
    async def test_calculate_business_impact(self, engine, mock_db):
        """Test business impact calculation"""
        feedback_data = {
            "feedback_type": FeedbackType.BUSINESS_IMPACT,
            "satisfaction_rating": 8,
            "user_id": "user123"
        }
        
        with patch.object(engine, '_get_user_impact_factor', return_value=1.5):
            result = await engine._calculate_business_impact(feedback_data, mock_db)
        
        # Business impact feedback with high satisfaction should have high score
        assert result > 5.0
        assert result <= 10.0
    
    def test_validate_ab_test_config(self, engine):
        """Test A/B test configuration validation"""
        # Valid configuration
        valid_config = {
            "name": "test",
            "hypothesis": "Test hypothesis",
            "feature_area": "dashboard",
            "control_config": {},
            "variant_configs": [{}],
            "traffic_allocation": {"control": 0.5, "variant_1": 0.5},
            "primary_metric": "satisfaction"
        }
        
        # Should not raise exception
        asyncio.run(engine._validate_ab_test_config(valid_config))
        
        # Invalid configuration - missing field
        invalid_config = valid_config.copy()
        del invalid_config["name"]
        
        with pytest.raises(ValueError, match="Missing required field: name"):
            asyncio.run(engine._validate_ab_test_config(invalid_config))
        
        # Invalid configuration - traffic allocation doesn't sum to 1.0
        invalid_allocation = valid_config.copy()
        invalid_allocation["traffic_allocation"] = {"control": 0.3, "variant_1": 0.5}
        
        with pytest.raises(ValueError, match="Traffic allocation must sum to 1.0"):
            asyncio.run(engine._validate_ab_test_config(invalid_allocation))

if __name__ == "__main__":
    pytest.main([__file__])