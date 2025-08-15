"""
Tests for PromptTemplateManager.
Tests template storage, retrieval, A/B testing, and analytics functionality.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from sqlalchemy.orm import Session

from scrollintel.engines.visual_generation.utils.prompt_template_manager import PromptTemplateManager
from scrollintel.models.prompt_enhancement_models import (
    VisualPromptTemplate, VisualPromptPattern, VisualPromptVariation, 
    VisualABTestExperiment, VisualABTestResult, VisualPromptCategory,
    VisualPromptUsageLog, VisualPromptOptimizationSuggestion
)

class TestPromptTemplateManager:
    """Test suite for PromptTemplateManager."""
    
    @pytest.fixture
    def mock_db(self):
        """Mock database session."""
        return Mock(spec=Session)
    
    @pytest.fixture
    def template_manager(self, mock_db):
        """Create PromptTemplateManager with mocked database."""
        with patch('scrollintel.engines.visual_generation.utils.prompt_template_manager.get_sync_db') as mock_get_db:
            mock_get_db.return_value = iter([mock_db])
            manager = PromptTemplateManager()
            manager.db = mock_db
            return manager
    
    @pytest.fixture
    def sample_category(self):
        """Sample prompt category."""
        return VisualPromptCategory(
            id=1,
            name="Photorealistic",
            description="Templates for photorealistic image generation",
            is_active=True
        )
    
    @pytest.fixture
    def sample_template(self, sample_category):
        """Sample prompt template."""
        return VisualPromptTemplate(
            id=1,
            name="Professional Portrait",
            template="professional portrait of {subject}, {lighting} lighting, high resolution",
            description="Template for professional portraits",
            category_id=sample_category.id,
            parameters=["subject", "lighting"],
            success_rate=0.85,
            usage_count=10,
            average_quality_score=8.5,
            is_active=True,
            created_by="test_user"
        )
    
    @pytest.fixture
    def sample_pattern(self):
        """Sample prompt pattern."""
        return VisualPromptPattern(
            id=1,
            pattern_text="high resolution, detailed, photorealistic",
            pattern_type="quality_enhancer",
            success_rate=0.92,
            usage_count=50,
            context="photorealistic images",
            effectiveness_score=9.2,
            is_active=True
        )
    
    def test_get_template_by_id_success(self, template_manager, sample_template):
        """Test successful template retrieval by ID."""
        template_manager.db.query.return_value.filter.return_value.first.return_value = sample_template
        
        result = template_manager.get_template_by_id(1)
        
        assert result == sample_template
        template_manager.db.query.assert_called_once_with(VisualPromptTemplate)
    
    def test_get_template_by_id_not_found(self, template_manager):
        """Test template retrieval when template doesn't exist."""
        template_manager.db.query.return_value.filter.return_value.first.return_value = None
        
        result = template_manager.get_template_by_id(999)
        
        assert result is None
    
    def test_get_templates_by_category(self, template_manager, sample_template):
        """Test retrieving templates by category."""
        template_manager.db.query.return_value.join.return_value.filter.return_value.order_by.return_value.limit.return_value.all.return_value = [sample_template]
        
        result = template_manager.get_templates_by_category("Photorealistic")
        
        assert len(result) == 1
        assert result[0] == sample_template
    
    def test_search_templates(self, template_manager, sample_template):
        """Test template search functionality."""
        template_manager.db.query.return_value.filter.return_value.order_by.return_value.limit.return_value.all.return_value = [sample_template]
        
        result = template_manager.search_templates("portrait")
        
        assert len(result) == 1
        assert result[0] == sample_template
    
    def test_search_templates_with_category(self, template_manager, sample_template):
        """Test template search with category filter."""
        template_manager.db.query.return_value.filter.return_value.join.return_value.filter.return_value.order_by.return_value.limit.return_value.all.return_value = [sample_template]
        
        result = template_manager.search_templates("portrait", category="Photorealistic")
        
        assert len(result) == 1
        assert result[0] == sample_template
    
    def test_get_top_templates(self, template_manager, sample_template):
        """Test retrieving top-performing templates."""
        template_manager.db.query.return_value.filter.return_value.order_by.return_value.limit.return_value.all.return_value = [sample_template]
        
        result = template_manager.get_top_templates()
        
        assert len(result) == 1
        assert result[0] == sample_template
    
    def test_create_template(self, template_manager):
        """Test creating a new template."""
        template_manager.db.add = Mock()
        template_manager.db.commit = Mock()
        template_manager.db.refresh = Mock()
        
        result = template_manager.create_template(
            name="Test Template",
            template="test {subject}",
            description="Test description",
            category_id=1,
            parameters=["subject"],
            created_by="test_user",
            tags=["test"]
        )
        
        assert result.name == "Test Template"
        assert result.template == "test {subject}"
        assert result.parameters == ["subject"]
        template_manager.db.add.assert_called_once()
        template_manager.db.commit.assert_called_once()
        template_manager.db.refresh.assert_called_once()
    
    def test_update_template_performance_success(self, template_manager, sample_template):
        """Test updating template performance metrics for successful generation."""
        template_manager.get_template_by_id = Mock(return_value=sample_template)
        template_manager.db.commit = Mock()
        
        original_success_rate = sample_template.success_rate
        original_usage_count = sample_template.usage_count
        
        template_manager.update_template_performance(1, 9.0, True, 2.5)
        
        assert sample_template.usage_count == original_usage_count + 1
        assert sample_template.success_rate > original_success_rate
        template_manager.db.commit.assert_called_once()
    
    def test_update_template_performance_failure(self, template_manager, sample_template):
        """Test updating template performance metrics for failed generation."""
        template_manager.get_template_by_id = Mock(return_value=sample_template)
        template_manager.db.commit = Mock()
        
        original_success_rate = sample_template.success_rate
        original_usage_count = sample_template.usage_count
        
        template_manager.update_template_performance(1, 3.0, False, 2.5)
        
        assert sample_template.usage_count == original_usage_count + 1
        assert sample_template.success_rate < original_success_rate
        template_manager.db.commit.assert_called_once()
    
    def test_get_successful_patterns(self, template_manager, sample_pattern):
        """Test retrieving successful prompt patterns."""
        template_manager.db.query.return_value.filter.return_value.order_by.return_value.limit.return_value.all.return_value = [sample_pattern]
        
        result = template_manager.get_successful_patterns()
        
        assert len(result) == 1
        assert result[0] == sample_pattern
    
    def test_get_successful_patterns_with_filters(self, template_manager, sample_pattern):
        """Test retrieving patterns with type and context filters."""
        template_manager.db.query.return_value.filter.return_value.filter.return_value.filter.return_value.order_by.return_value.limit.return_value.all.return_value = [sample_pattern]
        
        result = template_manager.get_successful_patterns(
            pattern_type="quality_enhancer",
            context="photorealistic"
        )
        
        assert len(result) == 1
        assert result[0] == sample_pattern
    
    def test_create_pattern(self, template_manager):
        """Test creating a new prompt pattern."""
        template_manager.db.add = Mock()
        template_manager.db.commit = Mock()
        template_manager.db.refresh = Mock()
        
        result = template_manager.create_pattern(
            pattern_text="ultra high quality",
            pattern_type="quality_enhancer",
            context="general",
            effectiveness_score=8.5
        )
        
        assert result.pattern_text == "ultra high quality"
        assert result.pattern_type == "quality_enhancer"
        assert result.effectiveness_score == 8.5
        template_manager.db.add.assert_called_once()
        template_manager.db.commit.assert_called_once()
    
    @patch('random.sample')
    def test_generate_template_variations(self, mock_sample, template_manager, sample_template, sample_pattern):
        """Test generating template variations."""
        template_manager.get_template_by_id = Mock(return_value=sample_template)
        template_manager.get_successful_patterns = Mock(return_value=[sample_pattern])
        template_manager.db.add = Mock()
        template_manager.db.commit = Mock()
        
        mock_sample.return_value = [sample_pattern]
        
        result = template_manager.generate_template_variations(1, 2)
        
        assert len(result) == 2
        assert all(isinstance(var, VisualPromptVariation) for var in result)
        assert template_manager.db.add.call_count == 2
        template_manager.db.commit.assert_called_once()
    
    def test_create_ab_test(self, template_manager):
        """Test creating an A/B test experiment."""
        template_manager.db.query.return_value.filter.return_value.count.return_value = 3  # Existing variations
        template_manager.db.add = Mock()
        template_manager.db.commit = Mock()
        template_manager.db.refresh = Mock()
        
        result = template_manager.create_ab_test(
            name="Test Experiment",
            description="Testing variations",
            template_id=1,
            created_by="test_user"
        )
        
        assert result.name == "Test Experiment"
        assert result.template_id == 1
        assert result.target_sample_size == 100
        template_manager.db.add.assert_called_once()
        template_manager.db.commit.assert_called_once()
    
    def test_get_active_ab_tests(self, template_manager):
        """Test retrieving active A/B tests."""
        mock_experiment = VisualABTestExperiment(
            id=1,
            name="Test Experiment",
            status="active",
            template_id=1,
            created_by="test_user"
        )
        
        template_manager.db.query.return_value.filter.return_value.order_by.return_value.limit.return_value.all.return_value = [mock_experiment]
        
        result = template_manager.get_active_ab_tests()
        
        assert len(result) == 1
        assert result[0] == mock_experiment
    
    def test_record_ab_test_result(self, template_manager):
        """Test recording A/B test results."""
        mock_experiment = VisualABTestExperiment(
            id=1,
            current_sample_size=50,
            target_sample_size=100
        )
        
        template_manager.db.query.return_value.filter.return_value.first.return_value = mock_experiment
        template_manager.db.add = Mock()
        template_manager.db.commit = Mock()
        template_manager.db.refresh = Mock()
        
        result = template_manager.record_ab_test_result(
            experiment_id=1,
            variation_id=1,
            user_id="test_user",
            quality_score=8.5,
            user_rating=4,
            generation_time=2.5,
            success=True
        )
        
        assert result.experiment_id == 1
        assert result.quality_score == 8.5
        assert mock_experiment.current_sample_size == 51
        template_manager.db.add.assert_called_once()
        template_manager.db.commit.assert_called_once()
    
    def test_log_prompt_usage(self, template_manager):
        """Test logging prompt usage."""
        template_manager.db.add = Mock()
        template_manager.db.commit = Mock()
        
        template_manager.log_prompt_usage(
            template_id=1,
            user_id="test_user",
            prompt_text="test prompt",
            parameters_used={"subject": "person"},
            quality_score=8.0,
            user_rating=4,
            generation_time=2.0,
            success=True,
            model_used="stable_diffusion_xl"
        )
        
        template_manager.db.add.assert_called_once()
        template_manager.db.commit.assert_called_once()
    
    def test_get_usage_analytics(self, template_manager):
        """Test retrieving usage analytics."""
        # Mock database queries
        template_manager.db.query.return_value.filter.return_value.count.return_value = 100
        template_manager.db.query.return_value.filter.return_value.scalar.return_value = 8.5
        
        mock_template_usage = Mock()
        mock_template_usage.name = "Test Template"
        mock_template_usage.usage_count = 25
        
        template_manager.db.query.return_value.join.return_value.filter.return_value.group_by.return_value.order_by.return_value.limit.return_value.all.return_value = [mock_template_usage]
        
        result = template_manager.get_usage_analytics(30)
        
        assert result['period_days'] == 30
        assert result['total_usage'] == 100
        assert result['average_quality_score'] == 8.5
        assert len(result['top_templates']) == 1
        assert result['top_templates'][0]['name'] == "Test Template"
    
    def test_suggest_prompt_optimization(self, template_manager, sample_pattern):
        """Test AI-powered prompt optimization suggestions."""
        template_manager.get_successful_patterns = Mock(return_value=[sample_pattern])
        template_manager.db.add = Mock()
        template_manager.db.commit = Mock()
        
        original_prompt = "portrait of a person"
        result = template_manager.suggest_prompt_optimization(original_prompt)
        
        assert result is not None
        assert "high resolution, detailed, photorealistic" in result
        assert original_prompt in result
        template_manager.db.add.assert_called_once()
        template_manager.db.commit.assert_called_once()
    
    def test_suggest_prompt_optimization_no_patterns(self, template_manager):
        """Test optimization when no patterns are available."""
        template_manager.get_successful_patterns = Mock(return_value=[])
        
        result = template_manager.suggest_prompt_optimization("test prompt")
        
        assert result is None
    
    def test_get_categories(self, template_manager, sample_category):
        """Test retrieving prompt categories."""
        template_manager.db.query.return_value.filter.return_value.order_by.return_value.all.return_value = [sample_category]
        
        result = template_manager.get_categories()
        
        assert len(result) == 1
        assert result[0] == sample_category
    
    @patch('random.choice')
    def test_get_template_for_ab_test_new_user(self, mock_choice, template_manager):
        """Test getting template for A/B test for new user."""
        mock_experiment = VisualABTestExperiment(id=1, template_id=1, status="active")
        mock_variation = VisualPromptVariation(id=1, template_id=1, variation_name="Test Variation")
        
        template_manager.db.query.return_value.filter.return_value.first.side_effect = [
            mock_experiment,  # First call for experiment
            None,  # Second call for existing result (none found)
        ]
        template_manager.db.query.return_value.filter.return_value.all.return_value = [mock_variation]
        
        mock_choice.return_value = mock_variation
        
        result = template_manager.get_template_for_ab_test(1, "new_user")
        
        assert result is not None
        assert result[0] == mock_variation
        assert result[1] is True  # New assignment
    
    def test_get_template_for_ab_test_existing_user(self, template_manager):
        """Test getting template for A/B test for existing user."""
        mock_experiment = VisualABTestExperiment(id=1, template_id=1, status="active")
        mock_result = VisualABTestResult(variation_id=1)
        mock_variation = VisualPromptVariation(id=1, template_id=1, variation_name="Test Variation")
        
        template_manager.db.query.return_value.filter.return_value.first.side_effect = [
            mock_experiment,  # First call for experiment
            mock_result,  # Second call for existing result
            mock_variation  # Third call for variation
        ]
        
        result = template_manager.get_template_for_ab_test(1, "existing_user")
        
        assert result is not None
        assert result[0] == mock_variation
        assert result[1] is False  # Existing assignment
    
    def test_close_connection(self, template_manager):
        """Test closing database connection."""
        template_manager.db.close = Mock()
        
        template_manager.close()
        
        template_manager.db.close.assert_called_once()

class TestPromptTemplateManagerIntegration:
    """Integration tests for PromptTemplateManager."""
    
    def test_full_ab_test_workflow(self):
        """Test complete A/B testing workflow."""
        # This would be an integration test with a real database
        # For now, we'll skip it as it requires database setup
        pass
    
    def test_template_performance_tracking(self):
        """Test template performance tracking over time."""
        # This would test the exponential moving average updates
        # and long-term performance tracking
        pass
    
    def test_pattern_effectiveness_analysis(self):
        """Test analysis of pattern effectiveness across different contexts."""
        # This would test the pattern recommendation system
        # and effectiveness scoring
        pass