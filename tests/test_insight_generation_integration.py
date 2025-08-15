"""
Integration tests for AI Insight Generation Engine workflows.

Tests the complete insight generation pipeline including pattern detection,
insight generation, recommendation creation, and anomaly explanation.
"""

import pytest
import pytest_asyncio
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Any
import numpy as np

from scrollintel.engines.insight_generator import (
    InsightGenerator, AnalyticsData, PatternDetector, InsightEngine, RecommendationEngine
)
from scrollintel.models.insight_models import (
    Pattern, Insight, ActionRecommendation, BusinessContext, Anomaly,
    InsightType, PatternType, SignificanceLevel, ActionPriority
)
from scrollintel.models.dashboard_models import BusinessMetric


class TestInsightGenerationIntegration:
    """Integration tests for insight generation workflows."""
    
    @pytest_asyncio.fixture
    async def insight_generator(self):
        """Create and initialize insight generator."""
        generator = InsightGenerator()
        await generator.start()
        try:
            yield generator
        finally:
            await generator.stop()
    
    @pytest.fixture
    def sample_metrics(self) -> List[BusinessMetric]:
        """Create sample business metrics for testing."""
        base_time = datetime.utcnow() - timedelta(days=30)
        metrics = []
        
        # Revenue trend (increasing)
        for i in range(30):
            timestamp = base_time + timedelta(days=i)
            value = 100000 + (i * 2000) + np.random.normal(0, 1000)  # Upward trend with noise
            metrics.append(BusinessMetric(
                name="revenue",
                category="financial",
                value=value,
                unit="USD",
                timestamp=timestamp,
                source="test",
                context={"department": "sales"}
            ))
        
        # Cost trend (stable with anomaly)
        for i in range(30):
            timestamp = base_time + timedelta(days=i)
            if i == 20:  # Anomaly on day 20
                value = 80000  # Spike
            else:
                value = 50000 + np.random.normal(0, 2000)  # Stable with noise
            metrics.append(BusinessMetric(
                name="operational_cost",
                category="financial",
                value=value,
                unit="USD",
                timestamp=timestamp,
                source="test",
                context={"department": "operations"}
            ))
        
        # Customer satisfaction (decreasing)
        for i in range(30):
            timestamp = base_time + timedelta(days=i)
            value = 95 - (i * 0.5) + np.random.normal(0, 2)  # Downward trend
            metrics.append(BusinessMetric(
                name="customer_satisfaction",
                category="quality",
                value=max(0, min(100, value)),  # Clamp to 0-100
                unit="percentage",
                timestamp=timestamp,
                source="test",
                context={"department": "customer_success"}
            ))
        
        # Performance metric (correlated with satisfaction)
        for i, satisfaction_metric in enumerate([m for m in metrics if m.name == "customer_satisfaction"]):
            timestamp = satisfaction_metric.timestamp
            # Performance correlates with satisfaction
            value = satisfaction_metric.value * 0.8 + np.random.normal(0, 5)
            metrics.append(BusinessMetric(
                name="system_performance",
                category="technical",
                value=max(0, min(100, value)),
                unit="percentage",
                timestamp=timestamp,
                source="test",
                context={"department": "engineering"}
            ))
        
        return metrics
    
    @pytest.fixture
    def analytics_data(self, sample_metrics) -> AnalyticsData:
        """Create analytics data from sample metrics."""
        start_time = datetime.utcnow() - timedelta(days=30)
        end_time = datetime.utcnow()
        
        return AnalyticsData(
            metrics=sample_metrics,
            time_range=(start_time, end_time),
            context={"analysis_type": "integration_test"}
        )
    
    @pytest.mark.asyncio
    async def test_complete_insight_generation_workflow(self, insight_generator, analytics_data):
        """Test the complete insight generation workflow."""
        # Process analytics data
        result = await insight_generator.process(analytics_data)
        
        # Verify result structure
        assert "patterns" in result
        assert "insights" in result
        assert "recommendations" in result
        assert "anomalies" in result
        assert "summary" in result
        assert "metadata" in result
        
        # Verify patterns were detected
        patterns = result["patterns"]
        assert len(patterns) > 0
        
        # Should detect trends in revenue and customer satisfaction
        pattern_metrics = [p["metric_name"] for p in patterns]
        assert any("revenue" in metric for metric in pattern_metrics)
        assert any("customer_satisfaction" in metric for metric in pattern_metrics)
        
        # Verify insights were generated
        insights = result["insights"]
        assert len(insights) > 0
        
        # Insights should have required fields
        for insight in insights:
            assert "id" in insight
            assert "type" in insight
            assert "title" in insight
            assert "description" in insight
            assert "confidence" in insight
            assert "significance" in insight
            assert "priority" in insight
        
        # Verify recommendations were created
        recommendations = result["recommendations"]
        assert len(recommendations) > 0
        
        # Recommendations should have implementation details
        for rec in recommendations:
            assert "title" in rec
            assert "description" in rec
            assert "priority" in rec
            assert "implementation_steps" in rec
            assert "timeline" in rec
        
        # Verify anomalies were detected
        anomalies = result["anomalies"]
        assert len(anomalies) > 0
        
        # Should detect the cost spike anomaly
        cost_anomalies = [a for a in anomalies if "cost" in a["metric_name"]]
        assert len(cost_anomalies) > 0
        
        # Verify summary
        summary = result["summary"]
        assert "overview" in summary
        assert "key_findings" in summary
        assert "recommendations_count" in summary
        
        # Verify metadata
        metadata = result["metadata"]
        assert "processed_at" in metadata
        assert "metrics_count" in metadata
        assert metadata["metrics_count"] == len(analytics_data.metrics)
    
    @pytest.mark.asyncio
    async def test_trend_pattern_detection(self, insight_generator, sample_metrics):
        """Test trend pattern detection specifically."""
        # Filter to revenue metrics only
        revenue_metrics = [m for m in sample_metrics if m.name == "revenue"]
        
        analytics_data = AnalyticsData(
            metrics=revenue_metrics,
            time_range=(datetime.utcnow() - timedelta(days=30), datetime.utcnow()),
            context={}
        )
        
        # Detect patterns
        patterns = await insight_generator.analyze_data_patterns(analytics_data)
        
        # Should detect increasing trend in revenue
        trend_patterns = [p for p in patterns if p.type == PatternType.INCREASING_TREND.value]
        assert len(trend_patterns) > 0
        
        revenue_trend = trend_patterns[0]
        assert revenue_trend.metric_name == "revenue"
        assert revenue_trend.confidence > 0.7  # High confidence for clear trend
        assert revenue_trend.statistical_measures["slope"] > 0  # Positive slope
    
    @pytest.mark.asyncio
    async def test_correlation_detection(self, insight_generator, sample_metrics):
        """Test correlation detection between metrics."""
        # Use satisfaction and performance metrics (designed to be correlated)
        correlated_metrics = [
            m for m in sample_metrics 
            if m.name in ["customer_satisfaction", "system_performance"]
        ]
        
        analytics_data = AnalyticsData(
            metrics=correlated_metrics,
            time_range=(datetime.utcnow() - timedelta(days=30), datetime.utcnow()),
            context={}
        )
        
        # Detect patterns
        patterns = await insight_generator.analyze_data_patterns(analytics_data)
        
        # Should detect correlation
        correlation_patterns = [p for p in patterns if p.type == PatternType.CORRELATION.value]
        assert len(correlation_patterns) > 0
        
        correlation = correlation_patterns[0]
        assert "customer_satisfaction" in correlation.metric_name
        assert "system_performance" in correlation.metric_name
        assert correlation.confidence > 0.7  # Strong correlation
    
    @pytest.mark.asyncio
    async def test_anomaly_detection(self, insight_generator, sample_metrics):
        """Test anomaly detection in metrics."""
        # Filter to cost metrics (contains anomaly)
        cost_metrics = [m for m in sample_metrics if m.name == "operational_cost"]
        
        pattern_detector = PatternDetector()
        anomalies = await pattern_detector.detect_anomalies(cost_metrics)
        
        # Should detect the cost spike anomaly
        assert len(anomalies) > 0
        
        cost_anomaly = anomalies[0]
        assert cost_anomaly.metric_name == "operational_cost"
        assert cost_anomaly.anomaly_type in ["spike", "drop"]
        assert cost_anomaly.deviation_score > 1.0  # Significant deviation
    
    @pytest.mark.asyncio
    async def test_insight_generation_from_patterns(self, insight_generator):
        """Test insight generation from detected patterns."""
        # Create a sample pattern
        pattern = Pattern(
            type=PatternType.INCREASING_TREND.value,
            metric_name="revenue",
            metric_category="financial",
            description="Strong upward trend detected",
            confidence=0.85,
            significance=SignificanceLevel.HIGH.value,
            start_time=datetime.utcnow() - timedelta(days=30),
            end_time=datetime.utcnow(),
            statistical_measures={
                "slope": 2000,
                "r_squared": 0.85,
                "p_value": 0.01,
                "trend_strength": "strong"
            }
        )
        
        # Generate insights
        insights = await insight_generator.generate_insights([pattern])
        
        assert len(insights) > 0
        
        insight = insights[0]
        assert insight.type == InsightType.TREND.value
        assert "revenue" in insight.title.lower()
        assert "increasing" in insight.description.lower()
        assert insight.confidence == 0.85
        assert insight.priority in [p.value for p in ActionPriority]
    
    @pytest.mark.asyncio
    async def test_recommendation_generation(self, insight_generator):
        """Test recommendation generation from insights."""
        # Create a sample insight
        insight = Insight(
            type=InsightType.TREND.value,
            title="Increasing trend in revenue",
            description="Revenue has shown strong upward trend",
            explanation="Statistical analysis shows significant growth",
            business_impact="Positive business performance",
            confidence=0.85,
            significance=0.8,
            priority=ActionPriority.HIGH.value,
            tags=["trend", "increasing", "financial"],
            affected_metrics=["revenue"]
        )
        
        # Generate recommendations
        recommendations = await insight_generator.suggest_actions([insight])
        
        assert len(recommendations) > 0
        
        for rec in recommendations:
            assert rec.insight_id == insight.id
            assert rec.title
            assert rec.description
            assert rec.priority in [p.value for p in ActionPriority]
            assert rec.implementation_steps
            assert len(rec.implementation_steps) > 0
    
    @pytest.mark.asyncio
    async def test_anomaly_explanation(self, insight_generator):
        """Test anomaly explanation with business context."""
        # Create sample anomaly
        anomaly = Anomaly(
            metric_name="operational_cost",
            metric_value=80000,
            expected_value=50000,
            deviation_score=3.0,
            anomaly_type="spike",
            severity=SignificanceLevel.HIGH.value,
            context={"department": "operations"}
        )
        
        # Create business context
        business_context = BusinessContext(
            context_type="department",
            name="Operations",
            description="Operations department context",
            thresholds={"operational_cost": {"critical": 0.3, "high": 0.2}},
            kpis=["operational_cost", "efficiency"]
        )
        
        # Generate explanation
        explanation = await insight_generator.explain_anomaly(anomaly, business_context)
        
        assert explanation
        assert "operational_cost" in explanation
        assert "anomaly" in explanation.lower()
        assert "operations" in explanation.lower()
    
    @pytest.mark.asyncio
    async def test_insight_ranking_and_prioritization(self, insight_generator, analytics_data):
        """Test that insights are properly ranked and prioritized."""
        result = await insight_generator.process(analytics_data)
        insights = result["insights"]
        
        if len(insights) > 1:
            # Insights should be ranked by significance and priority
            for i in range(len(insights) - 1):
                current = insights[i]
                next_insight = insights[i + 1]
                
                # Higher priority insights should come first
                priority_order = {
                    ActionPriority.URGENT.value: 4,
                    ActionPriority.HIGH.value: 3,
                    ActionPriority.MEDIUM.value: 2,
                    ActionPriority.LOW.value: 1
                }
                
                current_priority = priority_order.get(current["priority"], 0)
                next_priority = priority_order.get(next_insight["priority"], 0)
                
                # If same priority, higher significance should come first
                if current_priority == next_priority:
                    assert current["significance"] >= next_insight["significance"]
                else:
                    assert current_priority >= next_priority
    
    @pytest.mark.asyncio
    async def test_business_context_integration(self, insight_generator, sample_metrics):
        """Test integration with business context for better insights."""
        # Create analytics data with business context
        analytics_data = AnalyticsData(
            metrics=sample_metrics,
            time_range=(datetime.utcnow() - timedelta(days=30), datetime.utcnow()),
            context={
                "industry": "technology",
                "company_size": "enterprise",
                "business_model": "saas"
            }
        )
        
        result = await insight_generator.process(analytics_data)
        
        # Insights should be generated with business context
        insights = result["insights"]
        assert len(insights) > 0
        
        # Business impact should be contextual
        for insight in insights:
            assert insight["business_impact"]
            assert len(insight["business_impact"]) > 0
    
    @pytest.mark.asyncio
    async def test_multi_metric_correlation_analysis(self, insight_generator):
        """Test correlation analysis across multiple metrics."""
        # Create metrics with known correlations
        base_time = datetime.utcnow() - timedelta(days=20)
        metrics = []
        
        for i in range(20):
            timestamp = base_time + timedelta(days=i)
            
            # Base value that others correlate with
            base_value = 100 + i * 2 + np.random.normal(0, 5)
            
            # Positively correlated metric
            pos_corr_value = base_value * 0.8 + np.random.normal(0, 3)
            
            # Negatively correlated metric
            neg_corr_value = 200 - base_value * 0.6 + np.random.normal(0, 4)
            
            metrics.extend([
                BusinessMetric(
                    name="base_metric",
                    category="primary",
                    value=base_value,
                    timestamp=timestamp,
                    source="test"
                ),
                BusinessMetric(
                    name="positive_corr_metric",
                    category="secondary",
                    value=pos_corr_value,
                    timestamp=timestamp,
                    source="test"
                ),
                BusinessMetric(
                    name="negative_corr_metric",
                    category="secondary",
                    value=neg_corr_value,
                    timestamp=timestamp,
                    source="test"
                )
            ])
        
        analytics_data = AnalyticsData(
            metrics=metrics,
            time_range=(base_time, base_time + timedelta(days=20)),
            context={}
        )
        
        # Detect patterns
        patterns = await insight_generator.analyze_data_patterns(analytics_data)
        
        # Should detect correlations
        correlation_patterns = [p for p in patterns if p.type == PatternType.CORRELATION.value]
        assert len(correlation_patterns) >= 1
        
        # Verify correlation strengths
        for pattern in correlation_patterns:
            correlation_coeff = pattern.statistical_measures.get("correlation_coefficient", 0)
            assert abs(correlation_coeff) > 0.7  # Strong correlation
    
    @pytest.mark.asyncio
    async def test_time_series_pattern_detection(self, insight_generator):
        """Test detection of various time series patterns."""
        base_time = datetime.utcnow() - timedelta(days=30)
        metrics = []
        
        # Create seasonal pattern
        for i in range(30):
            timestamp = base_time + timedelta(days=i)
            # Seasonal pattern with weekly cycle
            seasonal_value = 100 + 20 * np.sin(2 * np.pi * i / 7) + np.random.normal(0, 5)
            
            metrics.append(BusinessMetric(
                name="seasonal_metric",
                category="operational",
                value=seasonal_value,
                timestamp=timestamp,
                source="test"
            ))
        
        analytics_data = AnalyticsData(
            metrics=metrics,
            time_range=(base_time, base_time + timedelta(days=30)),
            context={}
        )
        
        # Process data
        result = await insight_generator.process(analytics_data)
        
        # Should detect patterns in seasonal data
        patterns = result["patterns"]
        assert len(patterns) > 0
        
        # Verify pattern detection worked
        seasonal_patterns = [p for p in patterns if p["metric_name"] == "seasonal_metric"]
        assert len(seasonal_patterns) > 0
    
    @pytest.mark.asyncio
    async def test_insight_confidence_scoring(self, insight_generator, analytics_data):
        """Test that insight confidence scores are properly calculated."""
        result = await insight_generator.process(analytics_data)
        insights = result["insights"]
        
        for insight in insights:
            # Confidence should be between 0 and 1
            assert 0 <= insight["confidence"] <= 1
            
            # Significance should be between 0 and 1
            assert 0 <= insight["significance"] <= 1
            
            # Higher confidence insights should have more detailed explanations
            if insight["confidence"] > 0.8:
                assert len(insight["explanation"]) > 50  # Detailed explanation
    
    @pytest.mark.asyncio
    async def test_recommendation_implementation_details(self, insight_generator, analytics_data):
        """Test that recommendations include proper implementation details."""
        result = await insight_generator.process(analytics_data)
        recommendations = result["recommendations"]
        
        for rec in recommendations:
            # Should have implementation steps
            assert "implementation_steps" in rec
            assert isinstance(rec["implementation_steps"], list)
            assert len(rec["implementation_steps"]) > 0
            
            # Should have timeline
            assert "timeline" in rec
            assert rec["timeline"]
            
            # Should have effort estimate
            assert "effort_required" in rec
            assert rec["effort_required"] in ["low", "medium", "high"]
            
            # Should have responsible role
            assert "responsible_role" in rec
            assert rec["responsible_role"]
            
            # Should have success metrics
            assert "success_metrics" in rec
            assert isinstance(rec["success_metrics"], list)
    
    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self, insight_generator):
        """Test error handling in insight generation pipeline."""
        # Test with invalid data
        invalid_metrics = [
            BusinessMetric(
                name="",  # Empty name
                category="test",
                value=float('nan'),  # Invalid value
                timestamp=datetime.utcnow(),
                source="test"
            )
        ]
        
        analytics_data = AnalyticsData(
            metrics=invalid_metrics,
            time_range=(datetime.utcnow() - timedelta(days=1), datetime.utcnow()),
            context={}
        )
        
        # Should handle errors gracefully
        try:
            result = await insight_generator.process(analytics_data)
            # Should return empty results rather than crash
            assert "patterns" in result
            assert "insights" in result
            assert "recommendations" in result
        except Exception as e:
            # If it does raise an exception, it should be informative
            assert str(e)
    
    @pytest.mark.asyncio
    async def test_performance_with_large_dataset(self, insight_generator):
        """Test performance with larger datasets."""
        # Create large dataset
        base_time = datetime.utcnow() - timedelta(days=90)
        metrics = []
        
        metric_names = ["revenue", "cost", "users", "engagement", "performance"]
        
        for metric_name in metric_names:
            for i in range(90):  # 90 days of data
                timestamp = base_time + timedelta(days=i)
                value = 1000 + i * 10 + np.random.normal(0, 50)
                
                metrics.append(BusinessMetric(
                    name=metric_name,
                    category="test",
                    value=value,
                    timestamp=timestamp,
                    source="test"
                ))
        
        analytics_data = AnalyticsData(
            metrics=metrics,  # 450 data points
            time_range=(base_time, base_time + timedelta(days=90)),
            context={}
        )
        
        # Measure processing time
        start_time = datetime.utcnow()
        result = await insight_generator.process(analytics_data)
        end_time = datetime.utcnow()
        
        processing_time = (end_time - start_time).total_seconds()
        
        # Should complete within reasonable time (adjust threshold as needed)
        assert processing_time < 30  # 30 seconds max
        
        # Should still produce meaningful results
        assert len(result["patterns"]) > 0
        assert len(result["insights"]) > 0
        
        # Verify metadata
        assert result["metadata"]["metrics_count"] == len(metrics)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])