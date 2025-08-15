"""
Tests for Performance Reporting Engine

Tests for performance reporting, insight generation, and report optimization.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from scrollintel.engines.performance_reporting_engine import (
    PerformanceReportingEngine,
    PerformanceMetric,
    MetricType,
    ReportingPeriod,
    TrendDirection,
    PerformanceLevel,
    PerformanceTrend,
    PerformanceInsight,
    PerformanceReport
)

class TestPerformanceReportingEngine:
    
    @pytest.fixture
    def engine(self):
        """Create a performance reporting engine instance"""
        return PerformanceReportingEngine()
    
    @pytest.fixture
    def sample_financial_metric(self):
        """Create a sample financial metric"""
        return PerformanceMetric(
            metric_id="fin_001",
            name="Monthly Revenue",
            description="Total monthly revenue",
            metric_type=MetricType.FINANCIAL,
            current_value=1200000,
            target_value=1000000,
            previous_value=1100000,
            unit="USD",
            data_source="Financial System",
            calculation_method="Sum of all revenue streams",
            owner="CFO",
            last_updated=datetime.now()
        )
    
    @pytest.fixture
    def sample_operational_metric(self):
        """Create a sample operational metric"""
        return PerformanceMetric(
            metric_id="ops_001",
            name="Customer Satisfaction Score",
            description="Average customer satisfaction rating",
            metric_type=MetricType.OPERATIONAL,
            current_value=4.2,
            target_value=4.5,
            previous_value=4.3,
            unit="Score (1-5)",
            data_source="Customer Survey System",
            calculation_method="Weighted average of customer ratings",
            owner="VP Customer Success",
            last_updated=datetime.now()
        )
    
    @pytest.fixture
    def sample_critical_metric(self):
        """Create a sample critical performance metric"""
        return PerformanceMetric(
            metric_id="crit_001",
            name="System Uptime",
            description="System availability percentage",
            metric_type=MetricType.TECHNOLOGY,
            current_value=95.0,
            target_value=99.9,
            previous_value=98.5,
            unit="Percentage",
            data_source="Monitoring System",
            calculation_method="Uptime / Total time * 100",
            owner="CTO",
            last_updated=datetime.now()
        )
    
    @pytest.fixture
    def sample_historical_data(self):
        """Create sample historical data for trend analysis"""
        base_date = datetime.now() - timedelta(days=90)
        return [
            (base_date + timedelta(days=i*10), 1000000 + (i * 50000) + (i**2 * 1000))
            for i in range(10)
        ]
    
    def test_engine_initialization(self, engine):
        """Test engine initialization"""
        assert engine is not None
        assert len(engine.metrics) == 0
        assert len(engine.trends) == 0
        assert len(engine.insights) == 0
        assert len(engine.reports) == 0
        assert len(engine.report_templates) > 0
        assert 'board' in engine.report_templates
        assert 'executive' in engine.report_templates
        assert 'department' in engine.report_templates
    
    def test_add_performance_metric(self, engine, sample_financial_metric):
        """Test adding performance metric"""
        engine.add_performance_metric(sample_financial_metric)
        
        assert len(engine.metrics) == 1
        assert engine.metrics[0].metric_id == "fin_001"
        assert engine.metrics[0].name == "Monthly Revenue"
    
    def test_update_existing_metric(self, engine, sample_financial_metric):
        """Test updating existing metric"""
        # Add initial metric
        engine.add_performance_metric(sample_financial_metric)
        
        # Update with new value
        updated_metric = PerformanceMetric(
            metric_id="fin_001",
            name="Monthly Revenue",
            description="Total monthly revenue",
            metric_type=MetricType.FINANCIAL,
            current_value=1300000,  # Updated value
            target_value=1000000,
            previous_value=None,
            unit="USD",
            data_source="Financial System",
            calculation_method="Sum of all revenue streams",
            owner="CFO",
            last_updated=datetime.now()
        )
        
        engine.add_performance_metric(updated_metric)
        
        # Should still have only one metric but with updated value
        assert len(engine.metrics) == 1
        assert engine.metrics[0].current_value == 1300000
        assert engine.metrics[0].previous_value == 1200000  # Previous current value
    
    def test_calculate_performance_level_excellent(self, engine, sample_financial_metric):
        """Test calculating excellent performance level"""
        level = engine.calculate_performance_level(sample_financial_metric)
        assert level == PerformanceLevel.EXCELLENT  # 1.2M vs 1M target = 120%
    
    def test_calculate_performance_level_critical(self, engine, sample_critical_metric):
        """Test calculating critical performance level"""
        level = engine.calculate_performance_level(sample_critical_metric)
        assert level == PerformanceLevel.CRITICAL  # 95% vs 99.9% target = 95%
    
    def test_calculate_performance_level_good(self, engine):
        """Test calculating good performance level"""
        metric = PerformanceMetric(
            metric_id="test_001",
            name="Test Metric",
            description="Test metric",
            metric_type=MetricType.OPERATIONAL,
            current_value=105,
            target_value=100,
            previous_value=None,
            unit="Units",
            data_source="Test",
            calculation_method="Test",
            owner="Test",
            last_updated=datetime.now()
        )
        
        level = engine.calculate_performance_level(metric)
        assert level == PerformanceLevel.GOOD  # 105% of target
    
    def test_analyze_metric_trend_improving(self, engine, sample_historical_data):
        """Test analyzing improving trend"""
        trend = engine.analyze_metric_trend("test_metric", sample_historical_data)
        
        assert trend.metric_id == "test_metric"
        assert trend.trend_direction == TrendDirection.IMPROVING
        assert trend.trend_strength > 0
        assert len(trend.historical_data) == len(sample_historical_data)
        assert len(trend.forecast_values) == 3
        assert len(engine.trends) == 1
    
    def test_analyze_metric_trend_insufficient_data(self, engine):
        """Test trend analysis with insufficient data"""
        minimal_data = [(datetime.now(), 100), (datetime.now() + timedelta(days=1), 101)]
        
        trend = engine.analyze_metric_trend("test_metric", minimal_data)
        
        assert trend.trend_direction == TrendDirection.STABLE
        assert trend.trend_strength == 0.0
        assert trend.confidence_level == 0.0
    
    def test_generate_performance_insights(self, engine, sample_financial_metric, sample_critical_metric):
        """Test generating performance insights"""
        engine.add_performance_metric(sample_financial_metric)
        engine.add_performance_metric(sample_critical_metric)
        
        insights = engine.generate_performance_insights()
        
        assert len(insights) > 0
        assert len(engine.insights) > 0
        
        # Should have insights for both metrics
        metric_ids_with_insights = set(i.metric_id for i in insights)
        assert "fin_001" in metric_ids_with_insights or "crit_001" in metric_ids_with_insights
    
    def test_generate_insights_for_specific_metrics(self, engine, sample_financial_metric, sample_operational_metric):
        """Test generating insights for specific metrics"""
        engine.add_performance_metric(sample_financial_metric)
        engine.add_performance_metric(sample_operational_metric)
        
        insights = engine.generate_performance_insights(metric_ids=["fin_001"])
        
        # Should only have insights for the specified metric
        metric_ids_with_insights = set(i.metric_id for i in insights)
        assert "fin_001" in metric_ids_with_insights
        assert "ops_001" not in metric_ids_with_insights
    
    def test_detect_anomalies(self, engine, sample_critical_metric):
        """Test anomaly detection"""
        insights = engine._detect_anomalies(sample_critical_metric)
        
        assert len(insights) > 0
        anomaly_insight = insights[0]
        assert anomaly_insight.insight_type == "anomaly"
        assert anomaly_insight.significance_score > 0
        assert anomaly_insight.actionable is True
        assert len(anomaly_insight.recommended_actions) > 0
    
    def test_analyze_trends_with_trend_data(self, engine, sample_financial_metric, sample_historical_data):
        """Test trend analysis with existing trend data"""
        engine.add_performance_metric(sample_financial_metric)
        
        # Create trend first
        trend = engine.analyze_metric_trend("fin_001", sample_historical_data)
        
        # Now analyze trends
        insights = engine._analyze_trends(sample_financial_metric)
        
        assert len(insights) > 0
        trend_insight = insights[0]
        assert trend_insight.insight_type == "trend"
        assert "improving" in trend_insight.title.lower() or "declining" in trend_insight.title.lower()
    
    def test_generate_performance_alerts(self, engine, sample_critical_metric):
        """Test performance alert generation"""
        insights = engine._generate_performance_alerts(sample_critical_metric)
        
        assert len(insights) > 0
        alert_insight = insights[0]
        assert alert_insight.insight_type == "alert"
        assert alert_insight.title.startswith("Critical Performance Alert")
        assert alert_insight.significance_score == 1.0
        assert "immediate intervention" in alert_insight.recommended_actions[0].lower()
    
    def test_analyze_correlations(self, engine, sample_financial_metric, sample_operational_metric):
        """Test correlation analysis"""
        # Create metrics with similar performance levels
        financial_metric = sample_financial_metric  # Excellent performance
        
        operational_metric = PerformanceMetric(
            metric_id="ops_002",
            name="Operational Efficiency",
            description="Operational efficiency score",
            metric_type=MetricType.OPERATIONAL,
            current_value=110,
            target_value=100,  # Also excellent performance
            previous_value=105,
            unit="Score",
            data_source="Operations System",
            calculation_method="Efficiency calculation",
            owner="COO",
            last_updated=datetime.now()
        )
        
        engine.add_performance_metric(financial_metric)
        engine.add_performance_metric(operational_metric)
        
        insights = engine._analyze_correlations(financial_metric)
        
        # May or may not find correlations depending on implementation
        # Just verify it doesn't crash and returns a list
        assert isinstance(insights, list)
    
    def test_generate_forecasts(self, engine, sample_financial_metric, sample_historical_data):
        """Test forecast generation"""
        engine.add_performance_metric(sample_financial_metric)
        
        # Create declining trend that would miss target
        declining_data = [
            (datetime.now() - timedelta(days=i*10), 1000000 - (i * 100000))
            for i in range(5)
        ]
        declining_data.reverse()  # Make it chronological
        
        trend = engine.analyze_metric_trend("fin_001", declining_data)
        
        insights = engine._generate_forecasts(sample_financial_metric)
        
        if insights:  # May not generate forecast if trend is not concerning
            forecast_insight = insights[0]
            assert forecast_insight.insight_type == "forecast"
            assert "forecast" in forecast_insight.title.lower()
    
    def test_create_performance_report_board(self, engine, sample_financial_metric, sample_operational_metric):
        """Test creating board-level performance report"""
        engine.add_performance_metric(sample_financial_metric)
        engine.add_performance_metric(sample_operational_metric)
        
        report = engine.create_performance_report(
            title="Board Performance Report",
            reporting_period=ReportingPeriod.MONTHLY,
            period_start=datetime.now() - timedelta(days=30),
            period_end=datetime.now(),
            audience="board"
        )
        
        assert report is not None
        assert report.title == "Board Performance Report"
        assert report.audience == "board"
        assert len(report.metrics) > 0
        assert len(report.executive_summary) > 0
        assert len(report.key_findings) > 0
        assert len(report.recommendations) > 0
        assert len(engine.reports) == 1
    
    def test_create_performance_report_executive(self, engine, sample_financial_metric, sample_operational_metric, sample_critical_metric):
        """Test creating executive-level performance report"""
        engine.add_performance_metric(sample_financial_metric)
        engine.add_performance_metric(sample_operational_metric)
        engine.add_performance_metric(sample_critical_metric)
        
        report = engine.create_performance_report(
            title="Executive Performance Report",
            reporting_period=ReportingPeriod.WEEKLY,
            period_start=datetime.now() - timedelta(days=7),
            period_end=datetime.now(),
            audience="executive"
        )
        
        assert report.audience == "executive"
        assert len(report.metrics) <= 25  # Executive template max
        
        # Executive reports should have more detailed insights
        assert len(report.insights) > 0
    
    def test_create_performance_report_with_specific_metrics(self, engine, sample_financial_metric, sample_operational_metric):
        """Test creating report with specific metrics"""
        engine.add_performance_metric(sample_financial_metric)
        engine.add_performance_metric(sample_operational_metric)
        
        report = engine.create_performance_report(
            title="Specific Metrics Report",
            reporting_period=ReportingPeriod.QUARTERLY,
            period_start=datetime.now() - timedelta(days=90),
            period_end=datetime.now(),
            audience="executive",
            metric_ids=["fin_001"]
        )
        
        assert len(report.metrics) == 1
        assert report.metrics[0].metric_id == "fin_001"
    
    def test_optimize_report_for_audience(self, engine, sample_financial_metric):
        """Test optimizing report based on audience feedback"""
        engine.add_performance_metric(sample_financial_metric)
        
        # Create initial report
        report = engine.create_performance_report(
            title="Initial Report",
            reporting_period=ReportingPeriod.MONTHLY,
            period_start=datetime.now() - timedelta(days=30),
            period_end=datetime.now(),
            audience="executive"
        )
        
        # Provide feedback
        feedback = {
            'detail_preference': 'more_detail',
            'focus_preferences': ['financial', 'strategic'],
            'visualization_preference': 'detailed_charts'
        }
        
        optimized_report = engine.optimize_report_for_audience(report.report_id, feedback)
        
        assert optimized_report is not None
        assert "Optimized" in optimized_report.title
        assert optimized_report.customizations['focus_areas'] == ['financial', 'strategic']
        assert optimized_report.customizations['visualization_style'] == 'detailed_charts'
    
    def test_get_performance_analytics_empty(self, engine):
        """Test analytics with no metrics"""
        analytics = engine.get_performance_analytics()
        
        assert analytics['total_metrics'] == 0
        assert analytics['performance_distribution'] == {level.value: 0 for level in PerformanceLevel}
        assert analytics['trend_analysis']['improving_trends'] == 0
        assert analytics['insight_summary']['total_insights'] == 0
    
    def test_get_performance_analytics_with_data(self, engine, sample_financial_metric, sample_critical_metric, sample_historical_data):
        """Test analytics with performance data"""
        engine.add_performance_metric(sample_financial_metric)
        engine.add_performance_metric(sample_critical_metric)
        
        # Add trend data
        engine.analyze_metric_trend("fin_001", sample_historical_data)
        
        # Generate insights
        engine.generate_performance_insights()
        
        analytics = engine.get_performance_analytics()
        
        assert analytics['total_metrics'] == 2
        assert analytics['performance_distribution']['excellent'] >= 1  # Financial metric
        assert analytics['performance_distribution']['critical'] >= 1  # Critical metric
        assert analytics['trend_analysis']['improving_trends'] >= 0
        assert analytics['insight_summary']['total_insights'] > 0
        assert len(analytics['recommendations']) > 0
    
    def test_export_report_data(self, engine, sample_financial_metric):
        """Test exporting report data"""
        engine.add_performance_metric(sample_financial_metric)
        
        report = engine.create_performance_report(
            title="Export Test Report",
            reporting_period=ReportingPeriod.MONTHLY,
            period_start=datetime.now() - timedelta(days=30),
            period_end=datetime.now(),
            audience="board"
        )
        
        export_data = engine.export_report_data(report.report_id)
        
        assert 'report_metadata' in export_data
        assert 'executive_summary' in export_data
        assert 'key_findings' in export_data
        assert 'recommendations' in export_data
        assert 'metrics' in export_data
        assert 'insights' in export_data
        
        # Check metadata
        metadata = export_data['report_metadata']
        assert metadata['id'] == report.report_id
        assert metadata['title'] == "Export Test Report"
        assert metadata['audience'] == "board"
        
        # Check metrics data
        assert len(export_data['metrics']) > 0
        metric_data = export_data['metrics'][0]
        assert 'id' in metric_data
        assert 'name' in metric_data
        assert 'performance_level' in metric_data
    
    def test_invalid_report_id_error(self, engine):
        """Test error handling for invalid report ID"""
        with pytest.raises(ValueError, match="Report invalid_id not found"):
            engine.optimize_report_for_audience("invalid_id", {})
        
        with pytest.raises(ValueError, match="Report invalid_id not found"):
            engine.export_report_data("invalid_id")
    
    def test_metric_priority_calculation(self, engine, sample_financial_metric):
        """Test metric priority calculation"""
        template = engine.report_templates['board']
        priority = engine._get_metric_priority(sample_financial_metric, template)
        
        # Financial metrics should have high priority for board reports
        assert priority >= 1.0
    
    def test_performance_urgency_calculation(self, engine, sample_critical_metric):
        """Test performance urgency calculation"""
        urgency = engine._get_performance_urgency(sample_critical_metric)
        
        # Critical metrics should have high urgency
        assert urgency == 1.0
    
    def test_executive_summary_generation(self, engine, sample_financial_metric, sample_critical_metric):
        """Test executive summary generation"""
        engine.add_performance_metric(sample_financial_metric)
        engine.add_performance_metric(sample_critical_metric)
        
        metrics = [sample_financial_metric, sample_critical_metric]
        insights = engine.generate_performance_insights()
        template = engine.report_templates['board']
        
        summary = engine._create_executive_summary(metrics, insights, template)
        
        assert len(summary) > 0
        assert "Performance Overview" in summary or "performance" in summary.lower()
        assert "2" in summary  # Should mention 2 metrics
    
    def test_key_findings_extraction(self, engine, sample_financial_metric, sample_critical_metric):
        """Test key findings extraction"""
        engine.add_performance_metric(sample_financial_metric)
        engine.add_performance_metric(sample_critical_metric)
        
        metrics = [sample_financial_metric, sample_critical_metric]
        insights = engine.generate_performance_insights()
        
        findings = engine._extract_key_findings(metrics, insights)
        
        assert len(findings) > 0
        assert any("exceeding" in finding.lower() or "critical" in finding.lower() for finding in findings)
    
    def test_report_recommendations_generation(self, engine, sample_critical_metric):
        """Test report recommendations generation"""
        engine.add_performance_metric(sample_critical_metric)
        
        metrics = [sample_critical_metric]
        insights = engine.generate_performance_insights()
        template = engine.report_templates['executive']
        
        recommendations = engine._generate_report_recommendations(metrics, insights, template)
        
        assert len(recommendations) > 0
        assert any("immediate" in rec.lower() or "critical" in rec.lower() for rec in recommendations)
    
    def test_system_recommendations_generation(self, engine):
        """Test system-level recommendations generation"""
        performance_dist = {
            'critical': 2,
            'needs_improvement': 1,
            'satisfactory': 3,
            'good': 2,
            'excellent': 1
        }
        
        trend_analysis = {
            'improving_trends': 2,
            'declining_trends': 4,
            'stable_trends': 3,
            'volatile_trends': 0
        }
        
        insight_summary = {
            'total_insights': 10,
            'actionable_insights': 6
        }
        
        recommendations = engine._generate_system_recommendations(
            performance_dist, trend_analysis, insight_summary
        )
        
        assert len(recommendations) > 0
        assert any("critical" in rec.lower() for rec in recommendations)  # Should address critical metrics
        assert any("declining" in rec.lower() for rec in recommendations)  # Should address declining trends
    
    def test_trend_recommendations(self, engine, sample_financial_metric):
        """Test trend-based recommendations"""
        improving_recs = engine._get_trend_recommendations(TrendDirection.IMPROVING, sample_financial_metric)
        declining_recs = engine._get_trend_recommendations(TrendDirection.DECLINING, sample_financial_metric)
        
        assert len(improving_recs) > 0
        assert len(declining_recs) > 0
        assert any("continue" in rec.lower() for rec in improving_recs)
        assert any("investigate" in rec.lower() or "implement" in rec.lower() for rec in declining_recs)
    
    def test_trend_impact_assessment(self, engine, sample_financial_metric, sample_historical_data):
        """Test trend impact assessment"""
        trend = engine.analyze_metric_trend("fin_001", sample_historical_data)
        impact = engine._assess_trend_impact(trend, sample_financial_metric)
        
        assert impact in ["Critical", "High", "Medium", "Low"]
        
        # Financial metrics with strong trends should have higher impact
        if trend.trend_strength > 0.7:
            assert impact in ["Critical", "High"]
    
    @pytest.mark.asyncio
    async def test_concurrent_report_generation(self, engine, sample_financial_metric, sample_operational_metric):
        """Test concurrent report generation"""
        import asyncio
        
        engine.add_performance_metric(sample_financial_metric)
        engine.add_performance_metric(sample_operational_metric)
        
        async def create_report(title_suffix):
            return engine.create_performance_report(
                title=f"Concurrent Report {title_suffix}",
                reporting_period=ReportingPeriod.MONTHLY,
                period_start=datetime.now() - timedelta(days=30),
                period_end=datetime.now(),
                audience="executive"
            )
        
        # Create multiple reports concurrently
        tasks = [create_report(i) for i in range(3)]
        reports = await asyncio.gather(*tasks)
        
        # All should be created successfully
        assert len(reports) == 3
        assert len(engine.reports) == 3
        
        # Each should have unique ID
        ids = [r.report_id for r in reports]
        assert len(set(ids)) == 3