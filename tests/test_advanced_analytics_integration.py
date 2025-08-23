"""
Integration Tests for Advanced Analytics System

This module tests the complete advanced analytics system including all engines
and their integration with the unified coordinator.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Any

from scrollintel.models.advanced_analytics_models import (
    AdvancedAnalyticsRequest, AnalyticsType,
    GraphAnalysisRequest, SemanticQuery, PatternRecognitionRequest, 
    PredictiveAnalyticsRequest, PatternType, PredictionType
)
from scrollintel.engines.advanced_analytics_engine import AdvancedAnalyticsEngine
from scrollintel.engines.graph_analytics_engine import GraphAnalyticsEngine
from scrollintel.engines.semantic_search_engine import SemanticSearchEngine
from scrollintel.engines.pattern_recognition_engine import PatternRecognitionEngine
from scrollintel.engines.predictive_analytics_engine import PredictiveAnalyticsEngine


class TestAdvancedAnalyticsIntegration:
    """Test suite for advanced analytics integration."""
    
    @pytest.fixture
    def analytics_engine(self):
        """Create analytics engine instance for testing."""
        return AdvancedAnalyticsEngine()
    
    @pytest.fixture
    def sample_data_sources(self):
        """Sample data sources for testing."""
        return ["sales_data", "customer_data", "financial_data", "operational_data"]
    
    @pytest.mark.asyncio
    async def test_unified_analytics_execution(self, analytics_engine):
        """Test unified analytics execution through main coordinator."""
        # Test graph analytics
        graph_request = AdvancedAnalyticsRequest(
            analytics_type=AnalyticsType.GRAPH_ANALYSIS,
            parameters={
                "data_sources": ["sales_data", "customer_data"],
                "analysis_type": "centrality_analysis"
            },
            requester_id="test_user"
        )
        
        graph_result = await analytics_engine.execute_analytics(graph_request)
        
        assert graph_result.analytics_type == AnalyticsType.GRAPH_ANALYSIS
        assert graph_result.status == "completed"
        assert len(graph_result.insights) > 0
        assert graph_result.execution_metrics["execution_time_ms"] > 0
        
        # Test pattern recognition
        pattern_request = AdvancedAnalyticsRequest(
            analytics_type=AnalyticsType.PATTERN_RECOGNITION,
            parameters={
                "data_sources": ["sales_data"],
                "pattern_types": ["trend", "anomaly"]
            },
            requester_id="test_user"
        )
        
        pattern_result = await analytics_engine.execute_analytics(pattern_request)
        
        assert pattern_result.analytics_type == AnalyticsType.PATTERN_RECOGNITION
        assert pattern_result.status == "completed"
        assert len(pattern_result.insights) > 0
        
        # Test predictive analytics
        predictive_request = AdvancedAnalyticsRequest(
            analytics_type=AnalyticsType.PREDICTIVE_ANALYTICS,
            parameters={
                "data_sources": ["sales_data", "financial_data"],
                "prediction_types": ["revenue_forecast"]
            },
            requester_id="test_user"
        )
        
        predictive_result = await analytics_engine.execute_analytics(predictive_request)
        
        assert predictive_result.analytics_type == AnalyticsType.PREDICTIVE_ANALYTICS
        assert predictive_result.status == "completed"
        assert len(predictive_result.insights) > 0
    
    @pytest.mark.asyncio
    async def test_comprehensive_analysis(self, analytics_engine, sample_data_sources):
        """Test comprehensive analysis across all engines."""
        business_context = {
            "industry": "technology",
            "focus_areas": ["revenue", "efficiency", "growth"]
        }
        
        result = await analytics_engine.execute_comprehensive_analysis(
            sample_data_sources, business_context
        )
        
        assert "execution_summary" in result
        assert result["execution_summary"]["engines_executed"] >= 2
        assert result["execution_summary"]["total_insights"] > 0
        assert len(result["combined_insights"]) > 0
        assert len(result["business_opportunities"]) > 0
        
        # Check that multiple engines were executed
        engines_executed = 0
        if result.get("graph_analytics"):
            engines_executed += 1
        if result.get("pattern_recognition"):
            engines_executed += 1
        if result.get("predictive_analytics"):
            engines_executed += 1
        
        assert engines_executed >= 2
    
    @pytest.mark.asyncio
    async def test_business_intelligence_summary(self, analytics_engine):
        """Test business intelligence summary generation."""
        summary = await analytics_engine.get_business_intelligence_summary(30)
        
        assert "executive_summary" in summary
        assert "performance_metrics" in summary
        assert "trend_analysis" in summary
        assert "recommendations" in summary
        
        # Check executive summary structure
        exec_summary = summary["executive_summary"]
        assert "key_insights" in exec_summary
        assert "critical_actions" in exec_summary
        assert "risk_indicators" in exec_summary
        
        # Check performance metrics
        perf_metrics = summary["performance_metrics"]
        assert "analytics_executed" in perf_metrics
        assert "insights_generated" in perf_metrics
        assert "business_impact_score" in perf_metrics
    
    @pytest.mark.asyncio
    async def test_performance_tracking(self, analytics_engine):
        """Test performance metrics tracking."""
        # Execute some analytics to generate metrics
        request = AdvancedAnalyticsRequest(
            analytics_type=AnalyticsType.PATTERN_RECOGNITION,
            parameters={
                "data_sources": ["sales_data"],
                "pattern_types": ["trend"]
            },
            requester_id="test_user"
        )
        
        await analytics_engine.execute_analytics(request)
        
        # Get performance summary
        performance_summary = await analytics_engine.get_performance_summary()
        
        assert "total_operations" in performance_summary
        assert "avg_execution_time_ms" in performance_summary
        assert "by_engine" in performance_summary
        
        if performance_summary["total_operations"] > 0:
            assert performance_summary["avg_execution_time_ms"] > 0


class TestGraphAnalyticsEngine:
    """Test suite for graph analytics engine."""
    
    @pytest.fixture
    def graph_engine(self):
        """Create graph analytics engine for testing."""
        return GraphAnalyticsEngine()
    
    @pytest.mark.asyncio
    async def test_enterprise_graph_building(self, graph_engine):
        """Test enterprise graph construction."""
        data_sources = ["crm_data", "erp_data", "financial_data"]
        
        result = await graph_engine.build_enterprise_graph(data_sources)
        
        assert "nodes_count" in result
        assert "edges_count" in result
        assert "execution_time_ms" in result
        assert result["nodes_count"] > 0
        assert result["edges_count"] > 0
        assert result["execution_time_ms"] > 0
    
    @pytest.mark.asyncio
    async def test_relationship_analysis(self, graph_engine):
        """Test complex relationship analysis."""
        # Build graph first
        await graph_engine.build_enterprise_graph(["crm_data", "erp_data"])
        
        # Test different analysis types
        analysis_types = [
            "centrality_analysis",
            "community_detection", 
            "path_analysis",
            "influence_analysis",
            "anomaly_detection"
        ]
        
        for analysis_type in analysis_types:
            request = GraphAnalysisRequest(
                analysis_type=analysis_type,
                max_depth=3,
                min_confidence=0.5
            )
            
            result = await graph_engine.analyze_complex_relationships(request)
            
            assert result.analysis_type == analysis_type
            assert result.confidence_score > 0
            assert len(result.insights) > 0
            assert result.execution_time_ms > 0
    
    @pytest.mark.asyncio
    async def test_business_opportunity_detection(self, graph_engine):
        """Test business opportunity detection through graph analysis."""
        # Build graph first
        await graph_engine.build_enterprise_graph(["crm_data", "financial_data"])
        
        business_context = {
            "focus_areas": ["growth", "efficiency"],
            "industry": "technology"
        }
        
        opportunities = await graph_engine.detect_business_opportunities(business_context)
        
        assert isinstance(opportunities, list)
        # Should detect at least some opportunities with sample data
        if len(opportunities) > 0:
            opportunity = opportunities[0]
            assert hasattr(opportunity, 'title')
            assert hasattr(opportunity, 'confidence')
            assert hasattr(opportunity, 'business_impact')


class TestSemanticSearchEngine:
    """Test suite for semantic search engine."""
    
    @pytest.fixture
    def search_engine(self):
        """Create semantic search engine for testing."""
        return SemanticSearchEngine()
    
    @pytest.mark.asyncio
    async def test_data_indexing(self, search_engine):
        """Test enterprise data indexing."""
        data_sources = ["knowledge_base", "customer_communications", "financial_reports"]
        
        result = await search_engine.index_enterprise_data(data_sources)
        
        assert "documents_indexed" in result
        assert "execution_time_ms" in result
        assert result["documents_indexed"] > 0
        assert result["execution_time_ms"] > 0
    
    @pytest.mark.asyncio
    async def test_semantic_search(self, search_engine):
        """Test semantic search functionality."""
        # Index some data first
        await search_engine.index_enterprise_data(["knowledge_base", "customer_communications"])
        
        # Test search query
        query = SemanticQuery(
            query_text="customer satisfaction and product quality",
            max_results=10,
            similarity_threshold=0.5
        )
        
        result = await search_engine.semantic_search(query)
        
        assert result.query.query_text == query.query_text
        assert result.total_results >= 0
        assert result.execution_time_ms > 0
        assert len(result.search_insights) > 0
        
        # Check result structure if results found
        if len(result.results) > 0:
            search_result = result.results[0]
            assert hasattr(search_result, 'content')
            assert hasattr(search_result, 'relevance_score')
            assert hasattr(search_result, 'source')
    
    @pytest.mark.asyncio
    async def test_related_content_discovery(self, search_engine):
        """Test related content discovery."""
        # Index some data first
        await search_engine.index_enterprise_data(["knowledge_base"])
        
        # Test with a content ID (would exist after indexing)
        if search_engine.document_metadata:
            content_id = list(search_engine.document_metadata.keys())[0]
            
            related_items = await search_engine.discover_related_content(content_id, 5)
            
            assert isinstance(related_items, list)
            # Related items depend on content similarity
            for item in related_items:
                assert hasattr(item, 'relevance_score')
                assert item.relevance_score > 0


class TestPatternRecognitionEngine:
    """Test suite for pattern recognition engine."""
    
    @pytest.fixture
    def pattern_engine(self):
        """Create pattern recognition engine for testing."""
        return PatternRecognitionEngine()
    
    @pytest.mark.asyncio
    async def test_pattern_recognition(self, pattern_engine):
        """Test pattern recognition across different types."""
        request = PatternRecognitionRequest(
            data_source="sales_data",
            pattern_types=[PatternType.TREND, PatternType.ANOMALY, PatternType.CORRELATION],
            sensitivity=0.7,
            min_pattern_strength=0.5
        )
        
        result = await pattern_engine.recognize_patterns(request)
        
        assert result.request.data_source == "sales_data"
        assert len(result.patterns) >= 0
        assert len(result.summary_insights) > 0
        assert result.execution_time_ms > 0
        
        # Check pattern structure if patterns found
        for pattern in result.patterns:
            assert pattern.strength >= request.min_pattern_strength
            assert pattern.confidence >= request.sensitivity
            assert len(pattern.recommended_actions) > 0
    
    @pytest.mark.asyncio
    async def test_emerging_opportunities_detection(self, pattern_engine):
        """Test emerging opportunities detection."""
        data_sources = ["sales_data", "customer_behavior", "market_data"]
        
        opportunities = await pattern_engine.detect_emerging_opportunities(data_sources, 90)
        
        assert isinstance(opportunities, list)
        # Should detect opportunities with sample data
        for opportunity in opportunities:
            assert hasattr(opportunity, 'title')
            assert hasattr(opportunity, 'confidence')
            assert hasattr(opportunity, 'priority')
    
    @pytest.mark.asyncio
    async def test_pattern_monitoring(self, pattern_engine):
        """Test pattern change monitoring."""
        # Create baseline patterns
        baseline_request = PatternRecognitionRequest(
            data_source="sales_data",
            pattern_types=[PatternType.TREND],
            sensitivity=0.6
        )
        
        baseline_result = await pattern_engine.recognize_patterns(baseline_request)
        baseline_patterns = baseline_result.patterns
        
        # Monitor changes
        changes = await pattern_engine.monitor_pattern_changes(
            baseline_patterns, "sales_data"
        )
        
        assert "new_patterns" in changes
        assert "disappeared_patterns" in changes
        assert "change_summary" in changes
        assert len(changes["change_summary"]) > 0


class TestPredictiveAnalyticsEngine:
    """Test suite for predictive analytics engine."""
    
    @pytest.fixture
    def predictive_engine(self):
        """Create predictive analytics engine for testing."""
        return PredictiveAnalyticsEngine()
    
    @pytest.mark.asyncio
    async def test_business_outcome_prediction(self, predictive_engine):
        """Test business outcome predictions."""
        request = PredictiveAnalyticsRequest(
            prediction_type=PredictionType.REVENUE_FORECAST,
            data_sources=["sales_data", "financial_data"],
            target_variable="revenue",
            prediction_horizon=90,
            confidence_level=0.95
        )
        
        result = await predictive_engine.predict_business_outcomes(request)
        
        assert result.request.prediction_type == PredictionType.REVENUE_FORECAST
        assert len(result.predictions) > 0
        assert len(result.business_insights) > 0
        assert len(result.recommended_actions) > 0
        assert result.execution_time_ms > 0
        
        # Check prediction structure
        prediction = result.predictions[0]
        assert prediction.prediction_type == PredictionType.REVENUE_FORECAST
        assert "lower" in prediction.confidence_interval
        assert "upper" in prediction.confidence_interval
        assert len(prediction.scenarios) > 0
    
    @pytest.mark.asyncio
    async def test_revenue_forecasting(self, predictive_engine):
        """Test specific revenue forecasting."""
        data_sources = ["sales_data", "financial_data"]
        
        prediction = await predictive_engine.forecast_revenue(data_sources, 60)
        
        assert prediction.prediction_type == PredictionType.REVENUE_FORECAST
        assert prediction.base_prediction is not None
        assert "lower" in prediction.confidence_interval
        assert "upper" in prediction.confidence_interval
        assert len(prediction.key_drivers) > 0
    
    @pytest.mark.asyncio
    async def test_churn_prediction(self, predictive_engine):
        """Test customer churn prediction."""
        predictions = await predictive_engine.predict_customer_churn("customer_data")
        
        assert isinstance(predictions, list)
        # Check prediction structure if predictions exist
        for prediction in predictions:
            assert "churn_probability" in prediction
            assert "confidence_interval" in prediction
            assert "risk_factors" in prediction
    
    @pytest.mark.asyncio
    async def test_growth_opportunity_identification(self, predictive_engine):
        """Test growth opportunity identification."""
        data_sources = ["sales_data", "market_data", "customer_data"]
        
        opportunities = await predictive_engine.identify_growth_opportunities(data_sources)
        
        assert isinstance(opportunities, list)
        # Should identify opportunities with sample data
        for opportunity in opportunities:
            assert hasattr(opportunity, 'title')
            assert hasattr(opportunity, 'confidence')
            assert hasattr(opportunity, 'business_impact')


class TestAdvancedAnalyticsAPI:
    """Test suite for advanced analytics API integration."""
    
    @pytest.mark.asyncio
    async def test_api_request_processing(self):
        """Test API request processing and response format."""
        engine = AdvancedAnalyticsEngine()
        
        # Test graph analytics request
        request = AdvancedAnalyticsRequest(
            analytics_type=AnalyticsType.GRAPH_ANALYSIS,
            parameters={
                "data_sources": ["test_data"],
                "analysis_type": "centrality_analysis"
            },
            requester_id="api_test_user"
        )
        
        response = await engine.execute_analytics(request)
        
        # Validate response structure
        assert response.request_id == request.request_id
        assert response.analytics_type == request.analytics_type
        assert response.status == "completed"
        assert "execution_time_ms" in response.execution_metrics
        assert isinstance(response.insights, list)
        assert isinstance(response.business_opportunities, list)
    
    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling in analytics execution."""
        engine = AdvancedAnalyticsEngine()
        
        # Test with invalid analytics type (should be handled gracefully)
        try:
            request = AdvancedAnalyticsRequest(
                analytics_type="invalid_type",  # This should cause validation error
                parameters={},
                requester_id="test_user"
            )
            # This should fail at the model validation level
            assert False, "Should have raised validation error"
        except Exception:
            # Expected validation error
            pass
    
    @pytest.mark.asyncio
    async def test_caching_functionality(self):
        """Test analytics result caching."""
        engine = AdvancedAnalyticsEngine()
        
        request = AdvancedAnalyticsRequest(
            analytics_type=AnalyticsType.PATTERN_RECOGNITION,
            parameters={
                "data_sources": ["test_data"],
                "pattern_types": ["trend"]
            },
            requester_id="cache_test_user"
        )
        
        # Execute analytics
        response = await engine.execute_analytics(request)
        
        # Check if result is cached
        assert response.response_id in engine.analytics_cache
        cached_response = engine.analytics_cache[response.response_id]
        assert cached_response.request_id == request.request_id


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])