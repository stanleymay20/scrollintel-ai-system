"""
Unit tests for the A/B Testing Engine in the Advanced Prompt Management System.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from sqlalchemy.orm import Session

from scrollintel.engines.experiment_engine import ExperimentEngine, ExperimentConfig
from scrollintel.models.experiment_models import (
    Experiment, ExperimentVariant, VariantMetric, ExperimentResult,
    ExperimentSchedule, ExperimentStatus, VariantType, StatisticalSignificance
)
from scrollintel.models.prompt_models import AdvancedPromptTemplate


class TestExperimentEngine:
    """Test cases for ExperimentEngine."""
    
    @pytest.fixture
    def engine(self):
        """Create ExperimentEngine instance."""
        return ExperimentEngine()
    
    @pytest.fixture
    def mock_db(self):
        """Create mock database session."""
        return Mock(spec=Session)
    
    @pytest.fixture
    def sample_prompt(self):
        """Create sample prompt template."""
        return AdvancedPromptTemplate(
            id="prompt-123",
            name="Test Prompt",
            content="Hello {name}",
            category="greeting",
            created_by="test-user"
        )
    
    @pytest.fixture
    def sample_experiment_config(self):
        """Create sample experiment configuration."""
        return ExperimentConfig(
            name="Test Experiment",
            prompt_id="prompt-123",
            hypothesis="Variant B will perform better than Variant A",
            variants=[
                {
                    'name': 'Control',
                    'prompt_content': 'Hello {name}',
                    'variant_type': VariantType.CONTROL.value,
                    'traffic_weight': 0.5
                },
                {
                    'name': 'Treatment',
                    'prompt_content': 'Hi there {name}!',
                    'variant_type': VariantType.TREATMENT.value,
                    'traffic_weight': 0.5
                }
            ],
            success_metrics=['response_quality', 'user_satisfaction'],
            target_sample_size=1000,
            confidence_level=0.95,
            minimum_effect_size=0.05
        )
    
    @pytest.fixture
    def sample_experiment(self):
        """Create sample experiment."""
        experiment = Experiment(
            id="exp-123",
            name="Test Experiment",
            prompt_id="prompt-123",
            hypothesis="Test hypothesis",
            success_metrics=['response_quality'],
            target_sample_size=1000,
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
            prompt_content="Hi there {name}!",
            variant_type=VariantType.TREATMENT.value,
            traffic_weight=0.5
        )
        
        experiment.variants = [variant_a, variant_b]
        return experiment


class TestExperimentCreation:
    """Test experiment creation functionality."""
    
    @pytest.mark.asyncio
    async def test_create_experiment_success(self, engine, mock_db, sample_prompt, sample_experiment_config):
        """Test successful experiment creation."""
        # Setup mocks
        mock_db.query.return_value.filter.return_value.first.return_value = sample_prompt
        mock_db.flush = Mock()
        mock_db.commit = Mock()
        mock_db.add = Mock()
        
        # Create experiment
        with patch('scrollintel.engines.experiment_engine.get_db', return_value=iter([mock_db])):
            experiment = await engine.create_experiment(
                sample_experiment_config, 
                "test-user"
            )
        
        # Verify experiment creation
        assert mock_db.add.call_count == 3  # 1 experiment + 2 variants
        assert mock_db.commit.called
        assert experiment.name == "Test Experiment"
        assert experiment.prompt_id == "prompt-123"
        assert experiment.created_by == "test-user"
    
    @pytest.mark.asyncio
    async def test_create_experiment_prompt_not_found(self, engine, mock_db, sample_experiment_config):
        """Test experiment creation with non-existent prompt."""
        # Setup mocks
        mock_db.query.return_value.filter.return_value.first.return_value = None
        mock_db.rollback = Mock()
        
        # Test experiment creation failure
        with patch('scrollintel.engines.experiment_engine.get_db', return_value=iter([mock_db])):
            with pytest.raises(ValueError, match="Prompt prompt-123 not found"):
                await engine.create_experiment(sample_experiment_config, "test-user")
        
        assert mock_db.rollback.called
    
    @pytest.mark.asyncio
    async def test_create_experiment_database_error(self, engine, mock_db, sample_prompt, sample_experiment_config):
        """Test experiment creation with database error."""
        # Setup mocks
        mock_db.query.return_value.filter.return_value.first.return_value = sample_prompt
        mock_db.add.side_effect = Exception("Database error")
        mock_db.rollback = Mock()
        
        # Test experiment creation failure
        with patch('scrollintel.engines.experiment_engine.get_db', return_value=iter([mock_db])):
            with pytest.raises(Exception, match="Database error"):
                await engine.create_experiment(sample_experiment_config, "test-user")
        
        assert mock_db.rollback.called


class TestExperimentExecution:
    """Test experiment execution functionality."""
    
    @pytest.mark.asyncio
    async def test_start_experiment_success(self, engine, mock_db, sample_experiment):
        """Test successful experiment start."""
        # Setup mocks
        mock_db.query.return_value.filter.return_value.first.return_value = sample_experiment
        mock_db.commit = Mock()
        
        # Start experiment
        with patch('scrollintel.engines.experiment_engine.get_db', return_value=iter([mock_db])):
            result = await engine.start_experiment("exp-123")
        
        # Verify experiment started
        assert result is True
        assert sample_experiment.status == ExperimentStatus.RUNNING.value
        assert sample_experiment.start_date is not None
        assert mock_db.commit.called
    
    @pytest.mark.asyncio
    async def test_start_experiment_not_found(self, engine, mock_db):
        """Test starting non-existent experiment."""
        # Setup mocks
        mock_db.query.return_value.filter.return_value.first.return_value = None
        mock_db.rollback = Mock()
        
        # Test experiment start failure
        with patch('scrollintel.engines.experiment_engine.get_db', return_value=iter([mock_db])):
            with pytest.raises(ValueError, match="Experiment exp-123 not found"):
                await engine.start_experiment("exp-123")
        
        assert mock_db.rollback.called
    
    @pytest.mark.asyncio
    async def test_start_experiment_wrong_status(self, engine, mock_db, sample_experiment):
        """Test starting experiment in wrong status."""
        # Setup experiment in running status
        sample_experiment.status = ExperimentStatus.RUNNING.value
        mock_db.query.return_value.filter.return_value.first.return_value = sample_experiment
        mock_db.rollback = Mock()
        
        # Test experiment start failure
        with patch('scrollintel.engines.experiment_engine.get_db', return_value=iter([mock_db])):
            with pytest.raises(ValueError, match="Cannot start experiment in running status"):
                await engine.start_experiment("exp-123")
        
        assert mock_db.rollback.called
    
    @pytest.mark.asyncio
    async def test_stop_experiment_success(self, engine, mock_db, sample_experiment):
        """Test successful experiment stop."""
        # Setup experiment in running status
        sample_experiment.status = ExperimentStatus.RUNNING.value
        mock_db.query.return_value.filter.return_value.first.return_value = sample_experiment
        mock_db.commit = Mock()
        
        # Stop experiment
        with patch('scrollintel.engines.experiment_engine.get_db', return_value=iter([mock_db])):
            result = await engine.stop_experiment("exp-123")
        
        # Verify experiment stopped
        assert result is True
        assert sample_experiment.status == ExperimentStatus.COMPLETED.value
        assert sample_experiment.end_date is not None
        assert mock_db.commit.called


class TestVariantAssignment:
    """Test variant assignment functionality."""
    
    def test_assign_variant_success(self, engine, mock_db, sample_experiment):
        """Test successful variant assignment."""
        # Setup experiment in running status
        sample_experiment.status = ExperimentStatus.RUNNING.value
        mock_db.query.return_value.filter.return_value.first.return_value = sample_experiment
        
        # Test variant assignment
        with patch('scrollintel.engines.experiment_engine.get_db', return_value=iter([mock_db])):
            variant = engine.assign_variant("exp-123", "user-123")
        
        # Verify variant assigned
        assert variant is not None
        assert variant.id in ["variant-a", "variant-b"]
    
    def test_assign_variant_experiment_not_running(self, engine, mock_db, sample_experiment):
        """Test variant assignment for non-running experiment."""
        # Setup experiment in draft status
        sample_experiment.status = ExperimentStatus.DRAFT.value
        mock_db.query.return_value.filter.return_value.first.return_value = None
        
        # Test variant assignment
        with patch('scrollintel.engines.experiment_engine.get_db', return_value=iter([mock_db])):
            variant = engine.assign_variant("exp-123", "user-123")
        
        # Verify no variant assigned
        assert variant is None
    
    def test_assign_variant_traffic_allocation(self, engine, mock_db, sample_experiment):
        """Test variant assignment with traffic allocation."""
        # Setup experiment with low traffic allocation
        sample_experiment.status = ExperimentStatus.RUNNING.value
        sample_experiment.traffic_allocation = 0.1  # Only 10% of users
        mock_db.query.return_value.filter.return_value.first.return_value = sample_experiment
        
        # Test variant assignment for many users
        assigned_count = 0
        with patch('scrollintel.engines.experiment_engine.get_db', return_value=iter([mock_db])):
            for i in range(100):
                variant = engine.assign_variant("exp-123", f"user-{i}")
                if variant:
                    assigned_count += 1
        
        # Verify traffic allocation respected (approximately)
        assert assigned_count < 50  # Should be much less than 50% due to 10% allocation


class TestMetricsCollection:
    """Test metrics collection functionality."""
    
    @pytest.mark.asyncio
    async def test_record_metric_success(self, engine, mock_db):
        """Test successful metric recording."""
        # Setup mocks
        variant = ExperimentVariant(
            id="variant-123",
            experiment=Experiment(status=ExperimentStatus.RUNNING.value)
        )
        mock_db.query.return_value.filter.return_value.first.return_value = variant
        mock_db.add = Mock()
        mock_db.commit = Mock()
        
        # Record metric
        with patch('scrollintel.engines.experiment_engine.get_db', return_value=iter([mock_db])):
            result = await engine.record_metric(
                "variant-123", 
                "response_quality", 
                0.85,
                session_id="session-123",
                user_feedback={"rating": 4}
            )
        
        # Verify metric recorded
        assert result is True
        assert mock_db.add.called
        assert mock_db.commit.called
    
    @pytest.mark.asyncio
    async def test_record_metric_variant_not_found(self, engine, mock_db):
        """Test metric recording for non-existent variant."""
        # Setup mocks
        mock_db.query.return_value.filter.return_value.first.return_value = None
        mock_db.rollback = Mock()
        
        # Test metric recording failure
        with patch('scrollintel.engines.experiment_engine.get_db', return_value=iter([mock_db])):
            with pytest.raises(ValueError, match="Variant variant-123 not found"):
                await engine.record_metric("variant-123", "response_quality", 0.85)
        
        assert mock_db.rollback.called


class TestStatisticalAnalysis:
    """Test statistical analysis functionality."""
    
    @pytest.mark.asyncio
    async def test_analyze_experiment_success(self, engine, mock_db, sample_experiment):
        """Test successful experiment analysis."""
        # Setup experiment with metrics
        mock_db.query.return_value.filter.return_value.first.return_value = sample_experiment
        
        # Mock metrics data
        control_metrics = [
            VariantMetric(metric_value=0.7),
            VariantMetric(metric_value=0.8),
            VariantMetric(metric_value=0.75)
        ]
        treatment_metrics = [
            VariantMetric(metric_value=0.85),
            VariantMetric(metric_value=0.9),
            VariantMetric(metric_value=0.88)
        ]
        
        def mock_query_filter(query_obj):
            if "variant-a" in str(query_obj):
                return Mock(all=Mock(return_value=control_metrics))
            else:
                return Mock(all=Mock(return_value=treatment_metrics))
        
        mock_db.query.return_value.filter.side_effect = mock_query_filter
        mock_db.query.return_value.delete = Mock()
        mock_db.add = Mock()
        mock_db.commit = Mock()
        
        # Analyze experiment
        with patch('scrollintel.engines.experiment_engine.get_db', return_value=iter([mock_db])):
            results = await engine.analyze_experiment("exp-123")
        
        # Verify analysis results
        assert 'response_quality' in results
        metric_results = results['response_quality']
        assert 'variant_data' in metric_results
        assert 'comparisons' in metric_results
        assert 'winner' in metric_results
        assert len(metric_results['comparisons']) == 1  # One pairwise comparison
    
    def test_compare_variants_statistical_significance(self, engine):
        """Test variant comparison with statistical significance."""
        # Create variant data with significant difference
        variant_a = {
            'variant': Mock(id="variant-a"),
            'values': [0.7, 0.72, 0.68, 0.71, 0.69],
            'mean': 0.7,
            'std': 0.015,
            'count': 5
        }
        
        variant_b = {
            'variant': Mock(id="variant-b"),
            'values': [0.85, 0.87, 0.83, 0.86, 0.84],
            'mean': 0.85,
            'std': 0.015,
            'count': 5
        }
        
        # Compare variants
        comparison = engine._compare_variants(variant_a, variant_b, 0.95)
        
        # Verify comparison results
        assert 'p_value' in comparison
        assert 'effect_size' in comparison
        assert 'statistical_significance' in comparison
        assert 'confidence_interval' in comparison
        assert comparison['winner'] == "variant-b"  # Higher mean
    
    def test_determine_winner_high_confidence(self, engine):
        """Test winner determination with high confidence."""
        variant_data = {
            'variant-a': {'variant': Mock(id="variant-a", name="Control"), 'mean': 0.7},
            'variant-b': {'variant': Mock(id="variant-b", name="Treatment"), 'mean': 0.85}
        }
        
        comparisons = [{
            'winner': 'variant-b',
            'statistical_significance': StatisticalSignificance.SIGNIFICANT.value
        }]
        
        # Determine winner
        winner = engine._determine_winner(variant_data, comparisons)
        
        # Verify winner
        assert winner is not None
        assert winner['variant_id'] == 'variant-b'
        assert winner['confidence'] == 'high'
    
    def test_calculate_power(self, engine):
        """Test statistical power calculation."""
        # Test power calculation
        power = engine._calculate_power(
            effect_size=0.5,  # Medium effect size
            n1=100,
            n2=100,
            alpha=0.05
        )
        
        # Verify power is reasonable
        assert 0.0 <= power <= 1.0
        assert power > 0.5  # Should have decent power with medium effect size


class TestWinnerPromotion:
    """Test winner promotion functionality."""
    
    @pytest.mark.asyncio
    async def test_promote_winner_success(self, engine, mock_db, sample_experiment, sample_prompt):
        """Test successful winner promotion."""
        # Setup mocks
        mock_db.query.return_value.filter.return_value.first.side_effect = [
            sample_experiment,  # First call for experiment
            sample_experiment.variants[1],  # Second call for winning variant
            sample_prompt  # Third call for original prompt
        ]
        mock_db.commit = Mock()
        
        # Mock analysis results with clear winner
        analysis_results = {
            'response_quality': {
                'winner': {
                    'variant_id': 'variant-b',
                    'confidence': 'high',
                    'mean_performance': 0.85
                }
            }
        }
        
        with patch.object(engine, 'analyze_experiment', return_value=analysis_results):
            with patch('scrollintel.engines.experiment_engine.get_db', return_value=iter([mock_db])):
                result = await engine.promote_winner("exp-123", "response_quality")
        
        # Verify promotion success
        assert result['success'] is True
        assert 'Successfully promoted' in result['message']
        assert sample_experiment.status == ExperimentStatus.COMPLETED.value
        assert mock_db.commit.called
    
    @pytest.mark.asyncio
    async def test_promote_winner_no_significance(self, engine, mock_db, sample_experiment):
        """Test winner promotion without statistical significance."""
        # Setup mocks
        mock_db.query.return_value.filter.return_value.first.return_value = sample_experiment
        
        # Mock analysis results with low confidence winner
        analysis_results = {
            'response_quality': {
                'winner': {
                    'variant_id': 'variant-b',
                    'confidence': 'low',
                    'mean_performance': 0.75
                }
            }
        }
        
        with patch.object(engine, 'analyze_experiment', return_value=analysis_results):
            with patch('scrollintel.engines.experiment_engine.get_db', return_value=iter([mock_db])):
                result = await engine.promote_winner("exp-123", "response_quality")
        
        # Verify promotion rejected
        assert result['success'] is False
        assert 'not statistically significant' in result['message']


class TestExperimentScheduling:
    """Test experiment scheduling functionality."""
    
    @pytest.mark.asyncio
    async def test_schedule_experiment_success(self, engine, mock_db, sample_experiment):
        """Test successful experiment scheduling."""
        # Setup mocks
        mock_db.query.return_value.filter.return_value.first.return_value = sample_experiment
        mock_db.add = Mock()
        mock_db.commit = Mock()
        
        schedule_config = {
            'schedule_type': 'daily',
            'auto_start': True,
            'auto_stop': True,
            'auto_promote_winner': True,
            'promotion_threshold': 0.05,
            'max_duration_hours': 24
        }
        
        # Schedule experiment
        with patch('scrollintel.engines.experiment_engine.get_db', return_value=iter([mock_db])):
            schedule = await engine.schedule_experiment(
                "exp-123", 
                schedule_config, 
                "test-user"
            )
        
        # Verify schedule created
        assert mock_db.add.called
        assert mock_db.commit.called
        assert schedule.experiment_id == "exp-123"
        assert schedule.auto_start is True
        assert schedule.auto_stop is True
    
    def test_calculate_next_run_daily(self, engine):
        """Test next run calculation for daily schedule."""
        schedule = ExperimentSchedule(schedule_type='daily')
        
        # Calculate next run
        next_run = engine._calculate_next_run(schedule)
        
        # Verify next run is approximately 1 day from now
        now = datetime.utcnow()
        expected = now + timedelta(days=1)
        assert abs((next_run - expected).total_seconds()) < 3600  # Within 1 hour
    
    @pytest.mark.asyncio
    async def test_run_scheduled_experiments(self, engine, mock_db):
        """Test running scheduled experiments."""
        # Setup due schedule
        now = datetime.utcnow()
        schedule = ExperimentSchedule(
            id="schedule-123",
            experiment_id="exp-123",
            auto_start=True,
            next_run=now - timedelta(minutes=1),  # Due 1 minute ago
            is_active=True
        )
        
        experiment = Experiment(
            id="exp-123",
            status=ExperimentStatus.DRAFT.value,
            success_metrics=['response_quality']
        )
        schedule.experiment = experiment
        
        mock_db.query.return_value.filter.return_value.all.return_value = [schedule]
        mock_db.commit = Mock()
        
        with patch.object(engine, 'start_experiment', return_value=True):
            with patch('scrollintel.engines.experiment_engine.get_db', return_value=iter([mock_db])):
                results = await engine.run_scheduled_experiments()
        
        # Verify scheduled execution
        assert len(results) == 1
        assert results[0]['experiment_id'] == "exp-123"
        assert 'started' in results[0]['actions_taken']
        assert mock_db.commit.called


class TestExperimentStatus:
    """Test experiment status functionality."""
    
    @pytest.mark.asyncio
    async def test_get_experiment_status_success(self, engine, mock_db, sample_experiment):
        """Test getting experiment status."""
        # Setup mocks
        mock_db.query.return_value.filter.return_value.first.return_value = sample_experiment
        mock_db.query.return_value.filter.return_value.scalar.return_value = 50  # Metrics count
        
        # Get experiment status
        with patch('scrollintel.engines.experiment_engine.get_db', return_value=iter([mock_db])):
            status = await engine.get_experiment_status("exp-123")
        
        # Verify status information
        assert status['experiment_id'] == "exp-123"
        assert status['name'] == "Test Experiment"
        assert status['status'] == ExperimentStatus.DRAFT.value
        assert status['target_sample_size'] == 1000
        assert status['total_metrics_collected'] == 100  # 50 per variant
        assert status['progress'] == 0.1  # 100/1000
        assert len(status['variants']) == 2


if __name__ == "__main__":
    pytest.main([__file__])