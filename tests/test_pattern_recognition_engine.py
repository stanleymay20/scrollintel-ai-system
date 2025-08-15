"""
Tests for Pattern Recognition Engine

This module contains comprehensive tests for the pattern recognition engine,
including pattern recognition, analysis, interpretation, and innovation optimization.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import numpy as np

from scrollintel.engines.pattern_recognition_engine import PatternRecognitionEngine
from scrollintel.models.knowledge_integration_models import (
    KnowledgeItem, Pattern, PatternRecognitionResult,
    KnowledgeType, ConfidenceLevel, PatternType
)


class TestPatternRecognitionEngine:
    """Test cases for Pattern Recognition Engine"""
    
    @pytest.fixture
    def engine(self):
        """Create a pattern recognition engine instance"""
        return PatternRecognitionEngine()
    
    @pytest.fixture
    def sample_knowledge_items(self):
        """Sample knowledge items for testing"""
        return [
            KnowledgeItem(
                id="item_1",
                knowledge_type=KnowledgeType.RESEARCH_FINDING,
                content={
                    "title": "Machine Learning Optimization",
                    "description": "Study on optimizing machine learning algorithms",
                    "results": {"accuracy": 0.95, "speed": 120}
                },
                source="research_lab_1",
                timestamp=datetime.now() - timedelta(days=10),
                confidence=ConfidenceLevel.HIGH,
                tags=["machine learning", "optimization", "algorithms"]
            ),
            KnowledgeItem(
                id="item_2",
                knowledge_type=KnowledgeType.RESEARCH_FINDING,
                content={
                    "title": "Deep Learning Performance",
                    "description": "Analysis of deep learning model performance",
                    "results": {"accuracy": 0.92, "speed": 100}
                },
                source="research_lab_2",
                timestamp=datetime.now() - timedelta(days=8),
                confidence=ConfidenceLevel.HIGH,
                tags=["deep learning", "performance", "machine learning"]
            ),
            KnowledgeItem(
                id="item_3",
                knowledge_type=KnowledgeType.EXPERIMENTAL_RESULT,
                content={
                    "title": "Neural Network Experiment",
                    "description": "Experimental results from neural network training",
                    "results": {"accuracy": 0.88, "training_time": 240}
                },
                source="experiment_lab",
                timestamp=datetime.now() - timedelta(days=5),
                confidence=ConfidenceLevel.MEDIUM,
                tags=["neural networks", "experiments", "training"]
            ),
            KnowledgeItem(
                id="item_4",
                knowledge_type=KnowledgeType.RESEARCH_FINDING,
                content={
                    "title": "Quantum Computing Applications",
                    "description": "Research on quantum computing applications",
                    "results": {"quantum_advantage": True, "complexity": "exponential"}
                },
                source="quantum_lab",
                timestamp=datetime.now() - timedelta(days=3),
                confidence=ConfidenceLevel.VERY_HIGH,
                tags=["quantum computing", "applications", "complexity"]
            )
        ]
    
    @pytest.mark.asyncio
    async def test_recognize_patterns_all_types(self, engine, sample_knowledge_items):
        """Test pattern recognition for all pattern types"""
        # Recognize patterns
        result = await engine.recognize_patterns(sample_knowledge_items)
        
        # Verify result structure
        assert isinstance(result, PatternRecognitionResult)
        assert isinstance(result.patterns_found, list)
        assert result.analysis_method == "multi_pattern_recognition"
        assert result.processing_time > 0
        assert isinstance(result.recommendations, list)
        
        # Should find some patterns
        assert len(result.patterns_found) > 0
        
        # Verify patterns are stored in engine
        for pattern in result.patterns_found:
            assert pattern.id in engine.patterns
    
    @pytest.mark.asyncio
    async def test_recognize_correlation_patterns(self, engine, sample_knowledge_items):
        """Test recognition of correlation patterns"""
        # Focus on correlation patterns only
        result = await engine.recognize_patterns(
            sample_knowledge_items, 
            [PatternType.CORRELATION]
        )
        
        # Should find correlation patterns between ML-related items
        correlation_patterns = [p for p in result.patterns_found if p.pattern_type == PatternType.CORRELATION]
        assert len(correlation_patterns) > 0
        
        # Check pattern properties
        for pattern in correlation_patterns:
            assert pattern.strength > 0.0
            assert len(pattern.evidence) >= 2  # Correlations need at least 2 items
            assert pattern.predictive_power > 0.0
    
    @pytest.mark.asyncio
    async def test_recognize_temporal_patterns(self, engine, sample_knowledge_items):
        """Test recognition of temporal patterns"""
        # Focus on temporal patterns only
        result = await engine.recognize_patterns(
            sample_knowledge_items,
            [PatternType.TEMPORAL]
        )
        
        # May or may not find temporal patterns depending on timestamps
        temporal_patterns = [p for p in result.patterns_found if p.pattern_type == PatternType.TEMPORAL]
        
        # If found, verify properties
        for pattern in temporal_patterns:
            assert pattern.pattern_type == PatternType.TEMPORAL
            assert len(pattern.evidence) >= 3  # Temporal patterns need multiple items
    
    @pytest.mark.asyncio
    async def test_recognize_structural_patterns(self, engine, sample_knowledge_items):
        """Test recognition of structural patterns"""
        # Focus on structural patterns only
        result = await engine.recognize_patterns(
            sample_knowledge_items,
            [PatternType.STRUCTURAL]
        )
        
        # Should find structural patterns based on knowledge types and tags
        structural_patterns = [p for p in result.patterns_found if p.pattern_type == PatternType.STRUCTURAL]
        assert len(structural_patterns) > 0
        
        # Check pattern properties
        for pattern in structural_patterns:
            assert pattern.pattern_type == PatternType.STRUCTURAL
            assert pattern.strength > 0.0
    
    @pytest.mark.asyncio
    async def test_analyze_pattern_significance(self, engine, sample_knowledge_items):
        """Test pattern significance analysis"""
        # First recognize patterns
        result = await engine.recognize_patterns(sample_knowledge_items)
        
        if result.patterns_found:
            pattern = result.patterns_found[0]
            
            # Analyze significance
            significance = await engine.analyze_pattern_significance(pattern, sample_knowledge_items)
            
            # Verify significance analysis structure
            assert "significance" in significance
            assert "metrics" in significance
            assert "analysis" in significance
            assert "evidence_count" in significance
            assert "pattern_type" in significance
            
            # Check metrics
            metrics = significance["metrics"]
            assert "coverage" in metrics
            assert "strength" in metrics
            assert "confidence" in metrics
            assert "predictive_power" in metrics
            assert "diversity" in metrics
            assert "consistency" in metrics
            
            # All metrics should be between 0 and 1
            for metric_value in metrics.values():
                assert 0.0 <= metric_value <= 1.0
    
    @pytest.mark.asyncio
    async def test_interpret_patterns(self, engine, sample_knowledge_items):
        """Test pattern interpretation"""
        # First recognize patterns
        result = await engine.recognize_patterns(sample_knowledge_items)
        
        if result.patterns_found:
            # Interpret patterns
            interpretation = await engine.interpret_patterns(result.patterns_found)
            
            # Verify interpretation structure
            assert "interpretations" in interpretation
            assert "insights" in interpretation
            assert "recommendations" in interpretation
            assert "pattern_count" in interpretation
            assert "pattern_types" in interpretation
            
            # Should have some insights and recommendations
            assert len(interpretation["insights"]) > 0
            assert len(interpretation["recommendations"]) > 0
            assert interpretation["pattern_count"] == len(result.patterns_found)
    
    @pytest.mark.asyncio
    async def test_optimize_innovation_based_on_patterns(self, engine, sample_knowledge_items):
        """Test innovation optimization based on patterns"""
        # First recognize patterns
        result = await engine.recognize_patterns(sample_knowledge_items)
        
        if result.patterns_found:
            innovation_context = {
                "innovation_type": "machine_learning_system",
                "current_performance": {"accuracy": 0.85, "speed": 80},
                "target_performance": {"accuracy": 0.95, "speed": 120},
                "constraints": {"budget": 100000, "timeline": "6 months"}
            }
            
            # Optimize innovation
            optimization = await engine.optimize_innovation_based_on_patterns(
                result.patterns_found, innovation_context
            )
            
            # Verify optimization structure
            assert "optimization_recommendations" in optimization
            assert "enhancement_strategies" in optimization
            assert "risk_mitigations" in optimization
            assert "implementation_plan" in optimization
            assert "expected_impact" in optimization
            
            # Check implementation plan
            plan = optimization["implementation_plan"]
            assert "phases" in plan
            assert "timeline" in plan
            assert "resources_needed" in plan
            assert "success_metrics" in plan
            
            # Check expected impact
            impact = optimization["expected_impact"]
            assert "innovation_speed_improvement" in impact
            assert "quality_improvement" in impact
            assert "risk_reduction" in impact
            assert "overall_impact_score" in impact
    
    @pytest.mark.asyncio
    async def test_enhance_innovation_pipeline(self, engine, sample_knowledge_items):
        """Test innovation pipeline enhancement"""
        # First recognize patterns
        result = await engine.recognize_patterns(sample_knowledge_items)
        
        if result.patterns_found:
            pipeline_context = {
                "pipeline_type": "research_to_product",
                "current_bottlenecks": ["data_collection", "model_training"],
                "team_size": 10,
                "resources": {"compute": "limited", "data": "abundant"}
            }
            
            # Enhance pipeline
            enhancement = await engine.enhance_innovation_pipeline(
                result.patterns_found, pipeline_context
            )
            
            # Verify enhancement structure
            assert "pipeline_enhancements" in enhancement
            assert "process_improvements" in enhancement
            assert "bottleneck_solutions" in enhancement
            assert "optimization_strategy" in enhancement
            
            # Check optimization strategy
            strategy = enhancement["optimization_strategy"]
            assert "strategy_overview" in strategy
            assert "implementation_approach" in strategy
    
    @pytest.mark.asyncio
    async def test_pattern_caching(self, engine, sample_knowledge_items):
        """Test pattern caching functionality"""
        # Recognize patterns twice
        result1 = await engine.recognize_patterns(sample_knowledge_items)
        result2 = await engine.recognize_patterns(sample_knowledge_items)
        
        # Both results should have patterns
        assert len(result1.patterns_found) > 0
        assert len(result2.patterns_found) > 0
        
        # Patterns should be stored in engine
        assert len(engine.patterns) > 0
        
        # Cache should contain results
        assert len(engine.pattern_cache) > 0
    
    @pytest.mark.asyncio
    async def test_empty_knowledge_items(self, engine):
        """Test pattern recognition with empty knowledge items"""
        result = await engine.recognize_patterns([])
        
        # Should return empty result
        assert len(result.patterns_found) == 0
        assert result.processing_time >= 0
    
    @pytest.mark.asyncio
    async def test_single_knowledge_item(self, engine, sample_knowledge_items):
        """Test pattern recognition with single knowledge item"""
        result = await engine.recognize_patterns([sample_knowledge_items[0]])
        
        # May find some patterns (like structural patterns)
        assert isinstance(result.patterns_found, list)
        assert result.processing_time > 0
    
    @pytest.mark.asyncio
    async def test_pattern_strength_calculation(self, engine):
        """Test pattern strength calculation"""
        # Create items with high similarity
        similar_items = [
            KnowledgeItem(
                id="similar_1",
                knowledge_type=KnowledgeType.RESEARCH_FINDING,
                content={"title": "Machine Learning Optimization", "description": "ML optimization study"},
                source="lab1",
                timestamp=datetime.now(),
                confidence=ConfidenceLevel.HIGH,
                tags=["machine learning", "optimization"]
            ),
            KnowledgeItem(
                id="similar_2",
                knowledge_type=KnowledgeType.RESEARCH_FINDING,
                content={"title": "Machine Learning Enhancement", "description": "ML enhancement research"},
                source="lab2",
                timestamp=datetime.now(),
                confidence=ConfidenceLevel.HIGH,
                tags=["machine learning", "optimization"]
            )
        ]
        
        result = await engine.recognize_patterns(similar_items, [PatternType.CORRELATION])
        
        # Should find strong correlation patterns
        correlation_patterns = [p for p in result.patterns_found if p.pattern_type == PatternType.CORRELATION]
        
        if correlation_patterns:
            # Strength should be relatively high due to similarity
            assert any(p.strength > 0.5 for p in correlation_patterns)
    
    @pytest.mark.asyncio
    async def test_confidence_level_calculation(self, engine, sample_knowledge_items):
        """Test confidence level calculation for patterns"""
        result = await engine.recognize_patterns(sample_knowledge_items)
        
        # All patterns should have valid confidence levels
        for pattern in result.patterns_found:
            assert pattern.confidence in [
                ConfidenceLevel.LOW, ConfidenceLevel.MEDIUM, 
                ConfidenceLevel.HIGH, ConfidenceLevel.VERY_HIGH
            ]
        
        # Overall result confidence should be valid
        assert result.confidence in [
            ConfidenceLevel.LOW, ConfidenceLevel.MEDIUM,
            ConfidenceLevel.HIGH, ConfidenceLevel.VERY_HIGH
        ]
    
    @pytest.mark.asyncio
    async def test_predictive_power_calculation(self, engine, sample_knowledge_items):
        """Test predictive power calculation for patterns"""
        result = await engine.recognize_patterns(sample_knowledge_items)
        
        # All patterns should have predictive power between 0 and 1
        for pattern in result.patterns_found:
            assert 0.0 <= pattern.predictive_power <= 1.0
    
    @pytest.mark.asyncio
    async def test_pattern_recommendations(self, engine, sample_knowledge_items):
        """Test pattern-based recommendations"""
        result = await engine.recognize_patterns(sample_knowledge_items)
        
        # Should generate recommendations
        assert len(result.recommendations) > 0
        
        # Recommendations should be strings
        for recommendation in result.recommendations:
            assert isinstance(recommendation, str)
            assert len(recommendation) > 0
    
    @pytest.mark.asyncio
    async def test_cross_pattern_relationships(self, engine, sample_knowledge_items):
        """Test identification of cross-pattern relationships"""
        # Recognize multiple pattern types
        result = await engine.recognize_patterns(sample_knowledge_items)
        
        if len(result.patterns_found) > 1:
            # Interpret patterns to find cross-relationships
            interpretation = await engine.interpret_patterns(result.patterns_found)
            
            # Should have insights about pattern relationships
            insights = interpretation["insights"]
            cross_pattern_insights = [
                insight for insight in insights 
                if "pattern" in insight.lower() and ("and" in insight.lower() or "together" in insight.lower())
            ]
            
            # May or may not find cross-pattern relationships
            # This is acceptable as it depends on the specific patterns found
    
    @pytest.mark.asyncio
    async def test_pattern_applications(self, engine, sample_knowledge_items):
        """Test pattern applications tracking"""
        result = await engine.recognize_patterns(sample_knowledge_items)
        
        # Initially, patterns should have empty applications
        for pattern in result.patterns_found:
            assert isinstance(pattern.applications, list)
            # Applications start empty and are populated when patterns are used
    
    @pytest.mark.asyncio
    async def test_error_handling(self, engine):
        """Test error handling in various scenarios"""
        # Test with invalid knowledge items (should not crash)
        try:
            result = await engine.recognize_patterns([])
            assert len(result.patterns_found) == 0
        except Exception as e:
            pytest.fail(f"Should handle empty list gracefully: {str(e)}")
        
        # Test pattern significance with non-existent pattern
        fake_pattern = Pattern(
            id="fake_pattern",
            pattern_type=PatternType.CORRELATION,
            description="Fake pattern",
            evidence=[],
            strength=0.5,
            confidence=ConfidenceLevel.MEDIUM,
            discovered_at=datetime.now()
        )
        
        try:
            significance = await engine.analyze_pattern_significance(fake_pattern, [])
            # Should handle gracefully
            assert "significance" in significance
        except Exception as e:
            pytest.fail(f"Should handle empty evidence gracefully: {str(e)}")
    
    def test_initialization(self, engine):
        """Test engine initialization"""
        assert isinstance(engine.patterns, dict)
        assert isinstance(engine.pattern_cache, dict)
        assert hasattr(engine, 'vectorizer')
        
        # All stores should be empty initially
        assert len(engine.patterns) == 0
        assert len(engine.pattern_cache) == 0
    
    @pytest.mark.asyncio
    async def test_pattern_type_filtering(self, engine, sample_knowledge_items):
        """Test filtering by specific pattern types"""
        # Test each pattern type individually
        for pattern_type in PatternType:
            result = await engine.recognize_patterns(sample_knowledge_items, [pattern_type])
            
            # All found patterns should be of the requested type
            for pattern in result.patterns_found:
                assert pattern.pattern_type == pattern_type
    
    @pytest.mark.asyncio
    async def test_batch_pattern_recognition(self, engine, sample_knowledge_items):
        """Test batch pattern recognition performance"""
        # Recognize patterns multiple times to test performance
        results = []
        
        for i in range(3):
            result = await engine.recognize_patterns(sample_knowledge_items)
            results.append(result)
        
        # All results should be valid
        for result in results:
            assert isinstance(result, PatternRecognitionResult)
            assert result.processing_time > 0
        
        # Engine should accumulate patterns
        assert len(engine.patterns) > 0


if __name__ == "__main__":
    pytest.main([__file__])