"""
Tests for the PromptEnhancer class.
Tests ML-based improvement logic, context-aware suggestions, and feedback learning.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
from sqlalchemy.orm import Session

from scrollintel.engines.visual_generation.utils.prompt_enhancer import (
    PromptEnhancer, EnhancementStrategy, SuggestionType, EnhancementSuggestion,
    PromptEnhancementResult
)
from scrollintel.engines.visual_generation.utils.prompt_analyzer import (
    PromptAnalysisResult, PromptComplexity, ContentType, ArtisticStyle,
    TechnicalParameters, PromptQualityMetrics
)
from scrollintel.engines.visual_generation.exceptions import PromptEnhancementError
from scrollintel.models.prompt_enhancement_models import (
    VisualPromptTemplate, VisualPromptUsageLog, VisualPromptOptimizationSuggestion
)


class TestPromptEnhancer:
    """Test suite for PromptEnhancer class."""
    
    @pytest.fixture
    def mock_db_session(self):
        """Mock database session."""
        session = Mock(spec=Session)
        session.query.return_value.filter.return_value.limit.return_value.all.return_value = []
        session.query.return_value.filter.return_value.first.return_value = None
        session.add = Mock()
        session.commit = Mock()
        return session
    
    @pytest.fixture
    def mock_analysis_result(self):
        """Mock prompt analysis result."""
        return PromptAnalysisResult(
            original_prompt="a beautiful landscape",
            word_count=3,
            complexity=PromptComplexity.SIMPLE,
            content_type=ContentType.LANDSCAPE,
            artistic_style=ArtisticStyle.PHOTOREALISTIC,
            technical_parameters=TechnicalParameters(),
            quality_metrics=PromptQualityMetrics(
                overall_score=0.6,
                specificity_score=0.5,
                technical_completeness=0.4,
                style_clarity=0.7,
                structure_score=0.6,
                improvement_potential=0.4
            ),
            detected_subjects=[],
            detected_objects=["landscape"],
            detected_emotions=[],
            missing_elements=["resolution specification", "lighting description"],
            improvement_suggestions=["Add more descriptive details"],
            confidence=0.8,
            analysis_timestamp=datetime.now().isoformat()
        )
    
    @pytest.fixture
    def prompt_enhancer(self, mock_db_session):
        """Create PromptEnhancer instance with mocked dependencies."""
        with patch('scrollintel.engines.visual_generation.utils.prompt_enhancer.PromptAnalyzer') as mock_analyzer, \
             patch('scrollintel.engines.visual_generation.utils.prompt_enhancer.PromptTemplateManager') as mock_template_manager:
            
            enhancer = PromptEnhancer(mock_db_session)
            enhancer.analyzer = mock_analyzer.return_value
            enhancer.template_manager = mock_template_manager.return_value
            enhancer.template_manager.db = mock_db_session
            
            return enhancer
    
    @pytest.mark.asyncio
    async def test_enhance_prompt_basic(self, prompt_enhancer, mock_analysis_result):
        """Test basic prompt enhancement functionality."""
        # Setup
        prompt_enhancer.analyzer.analyze_prompt = AsyncMock(return_value=mock_analysis_result)
        prompt_enhancer.template_manager.search_templates = Mock(return_value=[])
        
        # Test
        result = await prompt_enhancer.enhance_prompt(
            "a beautiful landscape",
            strategy=EnhancementStrategy.MODERATE
        )
        
        # Assertions
        assert isinstance(result, PromptEnhancementResult)
        assert result.original_prompt == "a beautiful landscape"
        assert len(result.enhanced_prompt) > len(result.original_prompt)
        assert result.strategy_used == EnhancementStrategy.MODERATE
        assert result.overall_confidence > 0
        assert result.improvement_score > mock_analysis_result.quality_metrics.overall_score
        assert len(result.suggestions) > 0
    
    @pytest.mark.asyncio
    async def test_enhance_prompt_conservative_strategy(self, prompt_enhancer, mock_analysis_result):
        """Test conservative enhancement strategy."""
        # Setup
        prompt_enhancer.analyzer.analyze_prompt = AsyncMock(return_value=mock_analysis_result)
        prompt_enhancer.template_manager.search_templates = Mock(return_value=[])
        
        # Test
        result = await prompt_enhancer.enhance_prompt(
            "a beautiful landscape",
            strategy=EnhancementStrategy.CONSERVATIVE
        )
        
        # Assertions
        assert result.strategy_used == EnhancementStrategy.CONSERVATIVE
        # Conservative strategy should apply fewer suggestions
        assert len(result.suggestions) <= 3
        # Should prefer high-confidence, low-impact suggestions
        for suggestion in result.suggestions:
            assert suggestion.confidence > 0.8
    
    @pytest.mark.asyncio
    async def test_enhance_prompt_aggressive_strategy(self, prompt_enhancer, mock_analysis_result):
        """Test aggressive enhancement strategy."""
        # Setup
        prompt_enhancer.analyzer.analyze_prompt = AsyncMock(return_value=mock_analysis_result)
        prompt_enhancer.template_manager.search_templates = Mock(return_value=[])
        
        # Test
        result = await prompt_enhancer.enhance_prompt(
            "a beautiful landscape",
            strategy=EnhancementStrategy.AGGRESSIVE
        )
        
        # Assertions
        assert result.strategy_used == EnhancementStrategy.AGGRESSIVE
        # Aggressive strategy should apply more suggestions
        assert len(result.suggestions) >= 3
        # Should prioritize high-impact suggestions
        if len(result.suggestions) > 1:
            assert result.suggestions[0].impact_score >= result.suggestions[-1].impact_score
    
    @pytest.mark.asyncio
    async def test_enhance_prompt_with_context(self, prompt_enhancer, mock_analysis_result):
        """Test prompt enhancement with context information."""
        # Setup
        prompt_enhancer.analyzer.analyze_prompt = AsyncMock(return_value=mock_analysis_result)
        prompt_enhancer.template_manager.search_templates = Mock(return_value=[])
        
        context = {
            'time_of_day': 'evening',
            'mood': 'dramatic',
            'purpose': 'artistic'
        }
        
        # Test
        result = await prompt_enhancer.enhance_prompt(
            "a beautiful landscape",
            context=context,
            strategy=EnhancementStrategy.MODERATE
        )
        
        # Assertions
        assert result.enhancement_metadata['context'] == context
        # Should include context-aware suggestions or context should influence other suggestions
        context_suggestions = [s for s in result.suggestions if s.suggestion_type == SuggestionType.CONTEXT_ENRICHMENT]
        
        # Check if context-appropriate enhancements are applied (either through context suggestions or other types)
        enhanced_lower = result.enhanced_prompt.lower()
        context_influenced = any(word in enhanced_lower for word in ['sunset', 'warm', 'dramatic', 'artistic', 'evening', 'lighting'])
        
        # Either we have context suggestions OR the context influenced the enhancement
        assert len(context_suggestions) > 0 or context_influenced
    
    @pytest.mark.asyncio
    async def test_enhance_prompt_with_user_preferences(self, prompt_enhancer, mock_analysis_result, mock_db_session):
        """Test prompt enhancement with user preferences."""
        # Setup mock user usage history
        mock_usage = [
            Mock(
                prompt_text="cinematic landscape photo",
                quality_score=0.9,
                success=True,
                created_at=datetime.utcnow() - timedelta(days=5)
            ),
            Mock(
                prompt_text="dramatic lighting portrait",
                quality_score=0.85,
                success=True,
                created_at=datetime.utcnow() - timedelta(days=10)
            )
        ]
        
        mock_db_session.query.return_value.filter.return_value.limit.return_value.all.return_value = mock_usage
        
        prompt_enhancer.analyzer.analyze_prompt = AsyncMock(return_value=mock_analysis_result)
        prompt_enhancer.template_manager.search_templates = Mock(return_value=[])
        
        # Test
        result = await prompt_enhancer.enhance_prompt(
            "a beautiful landscape",
            user_id="test_user",
            strategy=EnhancementStrategy.MODERATE
        )
        
        # Assertions
        assert result.enhancement_metadata['user_id'] == "test_user"
        # Should include personalized suggestions based on user history
        enhanced_lower = result.enhanced_prompt.lower()
        assert 'cinematic' in enhanced_lower or 'dramatic' in enhanced_lower
    
    @pytest.mark.asyncio
    async def test_quality_improvement_suggestions(self, prompt_enhancer, mock_analysis_result):
        """Test generation of quality improvement suggestions."""
        # Setup - low quality prompt
        mock_analysis_result.quality_metrics.technical_completeness = 0.3
        mock_analysis_result.technical_parameters.resolution = None
        
        prompt_enhancer.analyzer.analyze_prompt = AsyncMock(return_value=mock_analysis_result)
        prompt_enhancer.template_manager.search_templates = Mock(return_value=[])
        
        # Test
        result = await prompt_enhancer.enhance_prompt(
            "landscape",
            strategy=EnhancementStrategy.MODERATE
        )
        
        # Assertions
        quality_suggestions = [s for s in result.suggestions if s.suggestion_type == SuggestionType.QUALITY_IMPROVEMENT]
        assert len(quality_suggestions) > 0
        
        # Should suggest quality modifiers
        enhanced_lower = result.enhanced_prompt.lower()
        quality_terms = ['high quality', 'masterpiece', 'detailed', '4k', '8k']
        assert any(term in enhanced_lower for term in quality_terms)
    
    @pytest.mark.asyncio
    async def test_style_enhancement_suggestions(self, prompt_enhancer, mock_analysis_result):
        """Test generation of style enhancement suggestions."""
        # Setup - photorealistic style
        mock_analysis_result.artistic_style = ArtisticStyle.PHOTOREALISTIC
        
        prompt_enhancer.analyzer.analyze_prompt = AsyncMock(return_value=mock_analysis_result)
        prompt_enhancer.template_manager.search_templates = Mock(return_value=[])
        
        # Test
        result = await prompt_enhancer.enhance_prompt(
            "a portrait",
            strategy=EnhancementStrategy.MODERATE
        )
        
        # Assertions
        style_suggestions = [s for s in result.suggestions if s.suggestion_type == SuggestionType.STYLE_ENHANCEMENT]
        assert len(style_suggestions) > 0
        
        # Should suggest photorealistic enhancements
        enhanced_lower = result.enhanced_prompt.lower()
        photo_terms = ['natural lighting', 'depth of field', 'bokeh', 'professional camera']
        assert any(term in enhanced_lower for term in photo_terms)
    
    @pytest.mark.asyncio
    async def test_technical_optimization_suggestions(self, prompt_enhancer, mock_analysis_result):
        """Test generation of technical optimization suggestions."""
        # Setup - missing technical parameters
        mock_analysis_result.technical_parameters.lighting = []
        mock_analysis_result.technical_parameters.camera_settings = []
        
        prompt_enhancer.analyzer.analyze_prompt = AsyncMock(return_value=mock_analysis_result)
        prompt_enhancer.template_manager.search_templates = Mock(return_value=[])
        
        # Test
        result = await prompt_enhancer.enhance_prompt(
            "a portrait",
            strategy=EnhancementStrategy.MODERATE
        )
        
        # Assertions
        tech_suggestions = [s for s in result.suggestions if s.suggestion_type == SuggestionType.TECHNICAL_OPTIMIZATION]
        assert len(tech_suggestions) > 0
        
        # Should suggest lighting and camera settings
        enhanced_lower = result.enhanced_prompt.lower()
        tech_terms = ['lighting', 'depth of field', 'lens', 'camera']
        assert any(term in enhanced_lower for term in tech_terms)
    
    @pytest.mark.asyncio
    async def test_template_based_suggestions(self, prompt_enhancer, mock_analysis_result):
        """Test suggestions based on successful templates."""
        # Setup mock successful templates
        mock_template = Mock()
        mock_template.template = "beautiful landscape, high quality, cinematic lighting, 4k"
        mock_template.success_rate = 0.9
        mock_template.name = "Landscape Template"
        mock_template.id = 1
        
        prompt_enhancer.analyzer.analyze_prompt = AsyncMock(return_value=mock_analysis_result)
        prompt_enhancer.template_manager.search_templates = Mock(return_value=[mock_template])
        
        # Test
        result = await prompt_enhancer.enhance_prompt(
            "landscape",
            strategy=EnhancementStrategy.MODERATE
        )
        
        # Assertions
        # Should incorporate patterns from successful templates
        enhanced_lower = result.enhanced_prompt.lower()
        template_terms = ['high quality', 'cinematic lighting', '4k']
        assert any(term in enhanced_lower for term in template_terms)
    
    @pytest.mark.asyncio
    async def test_suggestion_filtering_and_ranking(self, prompt_enhancer, mock_analysis_result):
        """Test suggestion filtering and ranking logic."""
        prompt_enhancer.analyzer.analyze_prompt = AsyncMock(return_value=mock_analysis_result)
        prompt_enhancer.template_manager.search_templates = Mock(return_value=[])
        
        # Test
        result = await prompt_enhancer.enhance_prompt(
            "landscape",
            strategy=EnhancementStrategy.MODERATE
        )
        
        # Assertions
        # Suggestions should be ranked by confidence and impact
        if len(result.suggestions) > 1:
            for i in range(len(result.suggestions) - 1):
                current_score = result.suggestions[i].confidence * result.suggestions[i].impact_score
                next_score = result.suggestions[i + 1].confidence * result.suggestions[i + 1].impact_score
                assert current_score >= next_score
        
        # Should not have duplicate suggestions
        suggestion_texts = [s.enhanced_text for s in result.suggestions]
        assert len(suggestion_texts) == len(set(suggestion_texts))
    
    @pytest.mark.asyncio
    async def test_learn_from_feedback_positive(self, prompt_enhancer, mock_db_session):
        """Test learning from positive user feedback."""
        # Setup mock suggestion
        mock_suggestion = Mock()
        mock_suggestion.suggestion_type = "quality_improvement"
        mock_db_session.query.return_value.filter.return_value.first.return_value = mock_suggestion
        
        # Test
        await prompt_enhancer.learn_from_feedback(
            enhancement_id=1,
            feedback="accepted",
            quality_score=0.9,
            user_rating=5
        )
        
        # Assertions
        assert mock_suggestion.user_feedback == "accepted"
        assert mock_suggestion.applied == True
        mock_db_session.commit.assert_called_once()
        
        # Should increase feedback weights for positive feedback
        assert abs(prompt_enhancer.feedback_weights['quality_modifiers'] - 0.95) < 0.001
    
    @pytest.mark.asyncio
    async def test_learn_from_feedback_negative(self, prompt_enhancer, mock_db_session):
        """Test learning from negative user feedback."""
        # Setup mock suggestion
        mock_suggestion = Mock()
        mock_suggestion.suggestion_type = "quality_improvement"
        mock_db_session.query.return_value.filter.return_value.first.return_value = mock_suggestion
        
        initial_weight = prompt_enhancer.feedback_weights['quality_modifiers']
        
        # Test
        await prompt_enhancer.learn_from_feedback(
            enhancement_id=1,
            feedback="rejected",
            quality_score=0.3,
            user_rating=2
        )
        
        # Assertions
        assert mock_suggestion.user_feedback == "rejected"
        assert mock_suggestion.applied == False
        mock_db_session.commit.assert_called_once()
        
        # Should decrease feedback weights for negative feedback
        assert prompt_enhancer.feedback_weights['quality_modifiers'] < initial_weight
    
    @pytest.mark.asyncio
    async def test_batch_enhance_prompts(self, prompt_enhancer, mock_analysis_result):
        """Test batch enhancement of multiple prompts."""
        # Setup
        prompt_enhancer.analyzer.analyze_prompt = AsyncMock(return_value=mock_analysis_result)
        prompt_enhancer.template_manager.search_templates = Mock(return_value=[])
        
        prompts = [
            "a beautiful landscape",
            "portrait of a person",
            "abstract art composition"
        ]
        
        # Test
        results = await prompt_enhancer.batch_enhance_prompts(
            prompts,
            strategy=EnhancementStrategy.MODERATE
        )
        
        # Assertions
        assert len(results) == len(prompts)
        for i, result in enumerate(results):
            assert isinstance(result, PromptEnhancementResult)
            assert result.original_prompt == prompts[i]
            assert len(result.enhanced_prompt) > len(prompts[i])
    
    @pytest.mark.asyncio
    async def test_get_enhancement_analytics(self, prompt_enhancer):
        """Test enhancement analytics functionality."""
        # Setup enhancement history
        prompt_enhancer.enhancement_history = [
            {
                'timestamp': (datetime.now() - timedelta(days=1)).isoformat(),
                'improvement_score': 0.8,
                'confidence': 0.9,
                'suggestions_count': 3,
                'strategy': 'moderate'
            },
            {
                'timestamp': (datetime.now() - timedelta(days=2)).isoformat(),
                'improvement_score': 0.75,
                'confidence': 0.85,
                'suggestions_count': 2,
                'strategy': 'conservative'
            }
        ]
        
        # Mock database query for feedback stats
        mock_feedback_stats = [
            Mock(user_feedback='accepted', count=10),
            Mock(user_feedback='rejected', count=2)
        ]
        prompt_enhancer.template_manager.db.query.return_value.filter.return_value.group_by.return_value.all.return_value = mock_feedback_stats
        
        # Test
        analytics = await prompt_enhancer.get_enhancement_analytics(days=7)
        
        # Assertions
        assert analytics['total_enhancements'] == 2
        assert analytics['average_improvement_score'] == 0.775
        assert analytics['average_confidence'] == 0.875
        assert analytics['average_suggestions_per_enhancement'] == 2.5
        assert 'strategy_distribution' in analytics
        assert 'feedback_statistics' in analytics
    
    @pytest.mark.asyncio
    async def test_error_handling(self, prompt_enhancer):
        """Test error handling in prompt enhancement."""
        # Setup analyzer to raise exception
        prompt_enhancer.analyzer.analyze_prompt = AsyncMock(side_effect=Exception("Analysis failed"))
        
        # Test
        with pytest.raises(PromptEnhancementError):
            await prompt_enhancer.enhance_prompt("test prompt")
    
    def test_export_enhancement_result(self, prompt_enhancer, mock_analysis_result):
        """Test exporting enhancement result to dictionary."""
        # Create a sample result
        suggestions = [
            EnhancementSuggestion(
                suggestion_type=SuggestionType.QUALITY_IMPROVEMENT,
                original_text="test",
                enhanced_text="test, high quality",
                confidence=0.9,
                reasoning="Adding quality modifier",
                impact_score=0.8
            )
        ]
        
        result = PromptEnhancementResult(
            original_prompt="test",
            enhanced_prompt="test, high quality",
            suggestions=suggestions,
            overall_confidence=0.9,
            improvement_score=0.8,
            strategy_used=EnhancementStrategy.MODERATE,
            analysis_result=mock_analysis_result,
            enhancement_metadata={'test': 'data'},
            timestamp=datetime.now().isoformat()
        )
        
        # Test
        exported = prompt_enhancer.export_enhancement_result(result)
        
        # Assertions
        assert isinstance(exported, dict)
        assert exported['original_prompt'] == "test"
        assert exported['enhanced_prompt'] == "test, high quality"
        assert len(exported['suggestions']) == 1
    
    def test_get_enhancement_summary(self, prompt_enhancer, mock_analysis_result):
        """Test getting human-readable enhancement summary."""
        # Create a sample result
        suggestions = [
            EnhancementSuggestion(
                suggestion_type=SuggestionType.QUALITY_IMPROVEMENT,
                original_text="test",
                enhanced_text="test, high quality",
                confidence=0.9,
                reasoning="Adding quality modifier for better results",
                impact_score=0.8
            )
        ]
        
        result = PromptEnhancementResult(
            original_prompt="test",
            enhanced_prompt="test, high quality",
            suggestions=suggestions,
            overall_confidence=0.9,
            improvement_score=0.8,
            strategy_used=EnhancementStrategy.MODERATE,
            analysis_result=mock_analysis_result,
            enhancement_metadata={},
            timestamp=datetime.now().isoformat()
        )
        
        # Test
        summary = prompt_enhancer.get_enhancement_summary(result)
        
        # Assertions
        assert isinstance(summary, str)
        assert "Prompt Enhancement Summary" in summary
        assert "moderate" in summary
        assert "test" in summary
        assert "high quality" in summary
        assert "Adding quality modifier" in summary
    
    @pytest.mark.asyncio
    async def test_content_type_specific_suggestions(self, prompt_enhancer, mock_analysis_result):
        """Test content type specific enhancement suggestions."""
        # Test portrait-specific suggestions
        mock_analysis_result.content_type = ContentType.PORTRAIT
        prompt_enhancer.analyzer.analyze_prompt = AsyncMock(return_value=mock_analysis_result)
        prompt_enhancer.template_manager.search_templates = Mock(return_value=[])
        
        result = await prompt_enhancer.enhance_prompt(
            "portrait of a person",
            strategy=EnhancementStrategy.MODERATE
        )
        
        # Should include portrait-specific enhancements
        enhanced_lower = result.enhanced_prompt.lower()
        portrait_terms = ['shallow depth', 'portrait', 'eye contact', 'professional headshot']
        assert any(term in enhanced_lower for term in portrait_terms)
    
    @pytest.mark.asyncio
    async def test_improvement_score_calculation(self, prompt_enhancer, mock_analysis_result):
        """Test improvement score calculation logic."""
        # Setup low quality analysis
        mock_analysis_result.quality_metrics.overall_score = 0.4
        
        prompt_enhancer.analyzer.analyze_prompt = AsyncMock(return_value=mock_analysis_result)
        prompt_enhancer.template_manager.search_templates = Mock(return_value=[])
        
        # Test
        result = await prompt_enhancer.enhance_prompt(
            "simple prompt",
            strategy=EnhancementStrategy.MODERATE
        )
        
        # Assertions
        # Improvement score should be higher than original quality
        assert result.improvement_score > mock_analysis_result.quality_metrics.overall_score
        # Should not exceed 1.0
        assert result.improvement_score <= 1.0
        # Should be reasonable improvement
        assert result.improvement_score < 1.0  # Perfect scores are unrealistic


if __name__ == "__main__":
    pytest.main([__file__])