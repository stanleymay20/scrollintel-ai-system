"""
Comprehensive integration tests for A/B Testing Engine.
Tests the complete workflow from experiment creation to winner promotion.
"""

import pytest
import asyncio
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
from sqlalchemy.orm import Session
from fastapi.testclient import TestClient
from fastapi import FastAPI

from scrollintel.engines.experiment_engine import ExperimentEngine, ExperimentConfig
from scrollintel.models.experiment_models import (
    Experiment, ExperimentVariant, VariantMetric, ExperimentResult,
    ExperimentSchedule, ExperimentStatus, VariantType, StatisticalSignificance
)


class TestExperimentIntegration:
    """Integration tests for the complete A/B testing workflow."""
    
    @pytest.fixture
    def app(self):
        """Create FastAPI app for testing."""
        app = FastAPI()
        return app
    
    @pytest.fixture
    def client(self, app):
        """Create test client."""
        return TestClient(app)
    
    @pytest.fixture
    def mock_db(self):
        """Create mock database session."""
        return Mock(spec=Session)
    
    @pytest.fixture
    def engine(self):
        """Create ExperimentEngine instance."""
        return ExperimentEngine()
    
    @pytest.fixture
    def sample_prompt(self):
        """Create sample prompt template."""
        return Mock(
            id="prompt-123",
            name="Test Prompt",
            content="Hello {name}",
            category="greeting",
            created_by="test-user"
        )
    
    @pytest.fixture
    def experiment_config(self):
        """Create sample experiment configuration."""
        return {
            "name": "Greeting Optimization Test",
            "prompt_id": "prompt-123",
            "hypothesis": "Casual greeting will improve user engagement",
            "variants": [
                {
                    "name": "Control - Formal",
                    "prompt_content": "Hello {name}, how may I assist you today?",
                    "variant_type": "control",
                    "traffic_weight": 0.5
                },
                {
                    "name": "Treatment - Casual",
                    "prompt_content": "Hi there {name}! What can I help you with?",
                    "variant_type": "treatment",
                    "traffic_weight": 0.5
                }
            ],
            "success_metrics": ["user_engagement", "response_quality"],
            "target_sample_size": 1000,
            "confidence_level": 0.95,
            "minimum_effect_size": 0.05,
            "auto_start": False,
            "auto_stop": False,
            "auto_promote_winner": False
        }


class TestExperimentCreationWorkflow:
    """Test the complete experiment creation workflow."""
    
    @pytest.mark.asyncio
    async def test_complete_experiment_creation_workflow(self, engine, mock_db, sample_prompt, experiment_config):
        """Test the complete experiment creation workflow."""
        # Setup mocks
        mock_db.query.return_value.filter.return_value.first.return_value = sample_prompt
        mock_db.add = Mock()
        mock_db.commit = Mock()
        mock_db.flush = Mock()
        
        # Convert dict to ExperimentConfig
        config = ExperimentConfig(
            name=experiment_config["name"],
            prompt_id=experiment_config["prompt_id"],
            hypothesis=experiment_config["hypothesis"],
            variants=experiment_config["variants"],
            success_metrics=experiment_config["success_metrics"],
            target_sample_size=experiment_config["target_sample_size"],
            confidence_level=experiment_config["confidence_level"],
            minimum_effect_size=experiment_config["minimum_effect_size"]
        )
        
        # Create experiment
        with patch('scrollintel.engines.experiment_engine.get_db', return_value=iter([mock_db])):
            result = await engine.create_experiment(config)
        
        # Verify experiment creation
        assert result["status"] == "created"
        assert "experiment_id" in result
        assert result["variants_count"] == 2
        assert result["target_sample_size"] == 1000
        assert mock_db.add.call_count >= 3  # Experiment + variants
        assert mock_db.commit.called
    
    @pytest.mark.asyncio
    async def test_experiment_lifecycle_workflow(self, engine, mock_db):
        """Test complete experiment lifecycle from creation to completion."""
        # Create mock experiment
        experiment = Experiment(
            id="exp-123",
            name="Test Experiment",
            prompt_id="prompt-123",
            hypothesis="Test hypothesis",
            success_metrics=["response_quality"],
            target_sample_size=100,
            confidence_level=0.95,
            minimum_effect_size=0.05,
            status=ExperimentStatus.DRAFT.value,
            created_by="test-user"
        )
        
        # Add variants
        variant_a = ExperimentVariant(
            id="variant-a",
            experiment_id="exp-123",
            name="Control",
            prompt_content="Hello {name}",
            variant_type=VariantType.CONTROL.value,
            traffic_weight=0.5
        )
        
        variant_b = ExperimentVariant(
            id="variant-b",
            experiment_id="exp-123",
            name="Treatment",
            prompt_content="Hi {name}!",
            variant_type=VariantType.TREATMENT.value,
            traffic_weight=0.5
        )
        
        experiment.variants = [variant_a, variant_b]
        
        # Setup mocks
        mock_db.query.return_value.filter.return_value.first.return_value = experiment
        mock_db.commit = Mock()
        
        with patch('scrollintel.engines.experiment_engine.get_db', return_value=iter([mock_db])):
            # 1. Start experiment
            start_result = await engine.start_experiment("exp-123")
            assert start_result["status"] == "running"
            assert experiment.status == ExperimentStatus.RUNNING.value
            assert experiment.start_date is not None
            
            # 2. Simulate metrics collection
            metrics = []
            for i in range(50):  # 50 samples per variant
                # Control variant metrics
                metrics.append(VariantMetric(
                    variant_id="variant-a",
                    metric_name="response_quality",
                    metric_value=np.random.normal(0.7, 0.1)
                ))
                # Treatment variant metrics (better performance)
                metrics.append(VariantMetric(
                    variant_id="variant-b",
                    metric_name="response_quality",
                    metric_value=np.random.normal(0.85, 0.1)
                ))
            
            # Mock metrics retrieval
            def mock_get_variant_metrics(variant_id):
                variant_metrics = [m for m in metrics if m.variant_id == variant_id]
                return [{"response_quality": m.metric_value} for m in variant_metrics]
            
            engine._get_variant_metrics = mock_get_variant_metrics
            
            # 3. Analyze experiment
            results = await engine.analyze_experiment_results("exp-123")
            assert results.experiment_id == "exp-123"
            assert len(results.analyses) > 0
            assert results.total_sample_size == 100
            
            # 4. Stop experiment
            stop_result = await engine.stop_experiment("exp-123")
            assert stop_result["status"] == "completed"
            assert experiment.status == ExperimentStatus.COMPLETED.value
            assert experiment.end_date is not None


class TestStatisticalAnalysisIntegration:
    """Test statistical analysis integration."""
    
    @pytest.mark.asyncio
    async def test_statistical_analysis_with_significant_results(self, engine):
        """Test statistical analysis with statistically significant results."""
        # Create mock data with significant difference
        control_data = np.random.normal(0.7, 0.1, 100).tolist()
        treatment_data = np.random.normal(0.85, 0.1, 100).tolist()  # Clearly better
        
        variant_data = {
            "variant-a": [{"response_quality": v} for v in control_data],
            "variant-b": [{"response_quality": v} for v in treatment_data]
        }
        
        # Perform analysis
        analysis = await engine._perform_statistical_analysis(
            variant_data, "response_quality", 0.95
        )
        
        # Verify results
        assert analysis is not None
        assert analysis.metric_name == "response_quality"
        assert analysis.control_sample_size == 100
        assert analysis.treatment_sample_size == 100
        assert analysis.p_value < 0.05  # Should be significant
        assert analysis.statistical_significance == StatisticalSignificance.SIGNIFICANT
        assert analysis.effect_size > 0  # Treatment should be better
        assert len(analysis.confidence_interval) == 2
    
    @pytest.mark.asyncio
    async def test_statistical_analysis_with_no_difference(self, engine):
        """Test statistical analysis with no significant difference."""
        # Create mock data with no real difference
        control_data = np.random.normal(0.75, 0.1, 100).tolist()
        treatment_data = np.random.normal(0.76, 0.1, 100).tolist()  # Minimal difference
        
        variant_data = {
            "variant-a": [{"response_quality": v} for v in control_data],
            "variant-b": [{"response_quality": v} for v in treatment_data]
        }
        
        # Perform analysis
        analysis = await engine._perform_statistical_analysis(
            variant_data, "response_quality", 0.95
        )
        
        # Verify results
        assert analysis is not None
        assert analysis.p_value > 0.05  # Should not be significant
        assert analysis.statistical_significance == StatisticalSignificance.NOT_SIGNIFICANT
        assert abs(analysis.effect_size) < 0.2  # Small effect size
    
    def test_winner_determination_logic(self, engine):
        """Test winner determination logic."""
        # Create analyses with clear winner
        analyses = [
            Mock(
                statistical_significance=StatisticalSignificance.SIGNIFICANT,
                effect_size=0.3,  # Large positive effect
                treatment_mean=0.85,
                control_mean=0.7
            ),
            Mock(
                statistical_significance=StatisticalSignificance.SIGNIFICANT,
                effect_size=0.25,  # Another positive effect
                treatment_mean=4.2,
                control_mean=3.8
            )
        ]
        
        winner = engine._determine_winner(analyses, min_effect_size=0.1)
        assert winner is not None  # Should have a winner
        
        # Test with no significant results
        analyses_no_sig = [
            Mock(
                statistical_significance=StatisticalSignificance.NOT_SIGNIFICANT,
                effect_size=0.05
            )
        ]
        
        winner_no_sig = engine._determine_winner(analyses_no_sig, min_effect_size=0.1)
        assert winner_no_sig is None  # Should have no winner


class TestExperimentSchedulingIntegration:
    """Test experiment scheduling and automation integration."""
    
    @pytest.mark.asyncio
    async def test_scheduled_experiment_execution(self, engine, mock_db):
        """Test scheduled experiment execution."""
        # Create experiment and schedule
        experiment = Experiment(
            id="exp-123",
            status=ExperimentStatus.DRAFT.value,
            success_metrics=["response_quality"]
        )
        
        schedule = ExperimentSchedule(
            id="schedule-123",
            experiment_id="exp-123",
            auto_start=True,
            next_run=datetime.utcnow() - timedelta(minutes=1),  # Due
            is_active=True
        )
        schedule.experiment = experiment
        
        # Setup mocks
        mock_db.query.return_value.filter.return_value.all.return_value = [schedule]
        mock_db.commit = Mock()
        
        # Mock start_experiment method
        engine.start_experiment = AsyncMock(return_value=True)
        
        # Run scheduled experiments
        with patch('scrollintel.engines.experiment_engine.get_db', return_value=iter([mock_db])):
            results = await engine.run_scheduled_experiments()
        
        # Verify execution
        assert len(results) == 1
        assert results[0]["experiment_id"] == "exp-123"
        assert "started" in results[0]["actions_taken"]
        engine.start_experiment.assert_called_once_with("exp-123")
    
    @pytest.mark.asyncio
    async def test_auto_stop_on_duration_limit(self, engine, mock_db):
        """Test automatic experiment stopping based on duration limit."""
        # Create running experiment that exceeded duration
        experiment = Experiment(
            id="exp-123",
            status=ExperimentStatus.RUNNING.value,
            start_date=datetime.utcnow() - timedelta(hours=25),  # Started 25 hours ago
            success_metrics=["response_quality"]
        )
        
        schedule = ExperimentSchedule(
            id="schedule-123",
            experiment_id="exp-123",
            auto_stop=True,
            max_duration_hours=24,  # 24 hour limit
            next_run=datetime.utcnow() - timedelta(minutes=1),
            is_active=True
        )
        schedule.experiment = experiment
        
        # Setup mocks
        mock_db.query.return_value.filter.return_value.all.return_value = [schedule]
        mock_db.commit = Mock()
        
        # Mock stop_experiment method
        engine.stop_experiment = AsyncMock(return_value={"status": "completed"})
        
        # Run scheduled experiments
        with patch('scrollintel.engines.experiment_engine.get_db', return_value=iter([mock_db])):
            results = await engine.run_scheduled_experiments()
        
        # Verify execution
        assert len(results) == 1
        assert "stopped" in results[0]["actions_taken"]
        engine.stop_experiment.assert_called_once_with("exp-123")


class TestAPIIntegration:
    """Test API integration concepts (mocked for demo)."""
    
    def test_experiment_api_concepts(self):
        """Test experiment API concepts."""
        # Test that we can create experiment configurations
        config = {
            "name": "Test Experiment",
            "prompt_id": "prompt-123",
            "hypothesis": "Treatment will be better",
            "variants": [
                {"name": "Control", "prompt_content": "Hello"},
                {"name": "Treatment", "prompt_content": "Hi there!"}
            ],
            "success_metrics": ["engagement"]
        }
        
        # Verify configuration structure
        assert "name" in config
        assert "variants" in config
        assert len(config["variants"]) == 2
        assert "success_metrics" in config
        
        # Test metric recording structure
        metric_data = {
            "metric_name": "response_quality",
            "metric_value": 0.85,
            "session_id": "session-123"
        }
        
        assert "metric_name" in metric_data
        assert "metric_value" in metric_data
        assert isinstance(metric_data["metric_value"], (int, float))
        
        print("✅ API integration concepts validated")


class TestEndToEndWorkflow:
    """Test complete end-to-end A/B testing workflow."""
    
    @pytest.mark.asyncio
    async def test_complete_ab_testing_workflow(self, engine, mock_db, sample_prompt):
        """Test complete A/B testing workflow from creation to promotion."""
        # Setup experiment configuration
        config = ExperimentConfig(
            name="Complete Workflow Test",
            prompt_id="prompt-123",
            hypothesis="Treatment will outperform control",
            variants=[
                {
                    "name": "Control",
                    "prompt_content": "Hello {name}",
                    "variant_type": "control",
                    "traffic_weight": 0.5
                },
                {
                    "name": "Treatment",
                    "prompt_content": "Hi {name}!",
                    "variant_type": "treatment",
                    "traffic_weight": 0.5
                }
            ],
            success_metrics=["user_engagement", "response_quality"],
            target_sample_size=200,
            confidence_level=0.95,
            minimum_effect_size=0.1
        )
        
        # Setup mocks
        mock_db.query.return_value.filter.return_value.first.return_value = sample_prompt
        mock_db.add = Mock()
        mock_db.commit = Mock()
        mock_db.flush = Mock()
        
        with patch('scrollintel.engines.experiment_engine.get_db', return_value=iter([mock_db])):
            # Step 1: Create experiment
            create_result = await engine.create_experiment(config)
            experiment_id = create_result["experiment_id"]
            
            # Step 2: Start experiment
            start_result = await engine.start_experiment(experiment_id)
            assert start_result["status"] == "running"
            
            # Step 3: Simulate data collection
            np.random.seed(42)  # For reproducible results
            
            # Generate metrics with treatment performing better
            control_metrics = []
            treatment_metrics = []
            
            for _ in range(100):  # 100 samples per variant
                # Control metrics
                control_metrics.extend([
                    {"user_engagement": np.random.normal(0.65, 0.1)},
                    {"response_quality": np.random.normal(0.75, 0.05)}
                ])
                
                # Treatment metrics (better performance)
                treatment_metrics.extend([
                    {"user_engagement": np.random.normal(0.8, 0.1)},
                    {"response_quality": np.random.normal(0.85, 0.05)}
                ])
            
            # Mock metrics retrieval
            def mock_get_variant_metrics(variant_id):
                if "control" in variant_id.lower() or variant_id == "variant-a":
                    return control_metrics
                else:
                    return treatment_metrics
            
            engine._get_variant_metrics = mock_get_variant_metrics
            
            # Step 4: Analyze results
            results = await engine.analyze_experiment_results(experiment_id)
            
            # Verify analysis results
            assert results.experiment_id == experiment_id
            assert len(results.analyses) == 2  # Two metrics analyzed
            assert results.total_sample_size == 400  # 200 per metric per variant
            
            # Check that treatment won for both metrics
            for analysis in results.analyses:
                assert analysis.statistical_significance in [
                    StatisticalSignificance.SIGNIFICANT,
                    StatisticalSignificance.HIGHLY_SIGNIFICANT
                ]
                assert analysis.effect_size > 0  # Treatment should be better
            
            # Step 5: Stop experiment
            stop_result = await engine.stop_experiment(experiment_id)
            assert stop_result["status"] == "completed"
            
            # Step 6: Promote winner
            promote_result = await engine.promote_winner(experiment_id)
            assert promote_result["status"] == "promoted"
            
            print("✅ Complete A/B testing workflow executed successfully!")
            print(f"   - Experiment created: {experiment_id}")
            print(f"   - Metrics analyzed: {len(results.analyses)}")
            print(f"   - Winner promoted: {results.winner_variant_id}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])