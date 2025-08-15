"""
Integration tests for Advanced Analytics Dashboard System
Tests complete workflows from data input to report generation
"""

import pytest
import asyncio
import json
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
import tempfile
import os

from scrollintel.engines.comprehensive_reporting_engine import (
    ComprehensiveReportingEngine, ReportConfig, ReportFormat, ReportType
)
from scrollintel.core.automated_report_scheduler import (
    AutomatedReportScheduler, ScheduledReportConfig, ScheduleFrequency, DeliveryMethod
)
from scrollintel.engines.advanced_statistical_analytics import AdvancedStatisticalAnalytics
from scrollintel.engines.executive_summary_generator import (
    ExecutiveSummaryGenerator, SummaryType
)
from scrollintel.api.routes.advanced_analytics_routes import router

class TestAdvancedAnalyticsIntegration:
    """Integration tests for advanced analytics system"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        from fastapi import FastAPI
        app = FastAPI()
        app.include_router(router)
        return TestClient(app)
    
    @pytest.fixture
    def sample_data(self):
        """Sample data for testing"""
        return {
            "metrics": {
                "revenue": 2400000,
                "revenue_growth": "12.5%",
                "active_users": 45200,
                "user_growth": "8.3%",
                "roi": "24.8%"
            },
            "system_metrics": {
                "uptime": "98.7%",
                "response_time": "150ms",
                "throughput": "1000 req/s",
                "error_rate": "0.1%"
            },
            "financial_metrics": {
                "revenue": 2400000,
                "revenue_growth": "12.5%",
                "profit_margin": "18.5%",
                "customer_acquisition_cost": 125
            },
            "engagement_metrics": {
                "active_users": 45200,
                "user_growth": "8.3%",
                "session_duration": "8.5 minutes",
                "bounce_rate": "32%"
            }
        }
    
    @pytest.fixture
    def sample_dataframe(self):
        """Sample DataFrame for statistical analysis"""
        return pd.DataFrame({
            'revenue': [100000, 120000, 110000, 130000, 125000, 140000, 135000, 150000],
            'users': [1000, 1200, 1100, 1300, 1250, 1400, 1350, 1500],
            'conversion_rate': [0.12, 0.15, 0.13, 0.16, 0.14, 0.17, 0.15, 0.18],
            'cost_per_acquisition': [50, 45, 48, 42, 46, 40, 43, 38]
        })
    
    def test_comprehensive_reporting_engine_initialization(self):
        """Test reporting engine initialization"""
        engine = ComprehensiveReportingEngine()
        assert engine is not None
        assert hasattr(engine, 'templates')
        assert hasattr(engine, 'report_cache')
        assert hasattr(engine, 'styles')
    
    def test_report_generation_pdf(self, sample_data):
        """Test PDF report generation"""
        engine = ComprehensiveReportingEngine()
        
        config = ReportConfig(
            report_type=ReportType.EXECUTIVE_SUMMARY,
            format=ReportFormat.PDF,
            title="Test Executive Report",
            description="Test report for integration testing",
            date_range={
                'start_date': datetime.now() - timedelta(days=30),
                'end_date': datetime.now()
            },
            filters={},
            sections=['overview', 'metrics', 'recommendations'],
            visualizations=['charts', 'tables'],
            recipients=['test@example.com']
        )
        
        report = engine.generate_report(config, sample_data)
        
        assert report is not None
        assert report.id is not None
        assert report.config.title == "Test Executive Report"
        assert report.file_data is not None
        assert len(report.sections) > 0
        assert report.generated_at is not None
    
    def test_report_generation_excel(self, sample_data):
        """Test Excel report generation"""
        engine = ComprehensiveReportingEngine()
        
        config = ReportConfig(
            report_type=ReportType.DETAILED_ANALYTICS,
            format=ReportFormat.EXCEL,
            title="Test Analytics Report",
            description="Detailed analytics report",
            date_range={
                'start_date': datetime.now() - timedelta(days=30),
                'end_date': datetime.now()
            },
            filters={},
            sections=[],
            visualizations=[],
            recipients=[]
        )
        
        report = engine.generate_report(config, sample_data)
        
        assert report is not None
        assert report.file_data is not None
        assert report.config.format == ReportFormat.EXCEL
    
    def test_report_generation_json(self, sample_data):
        """Test JSON report generation"""
        engine = ComprehensiveReportingEngine()
        
        config = ReportConfig(
            report_type=ReportType.ROI_ANALYSIS,
            format=ReportFormat.JSON,
            title="Test ROI Report",
            description="ROI analysis report",
            date_range={
                'start_date': datetime.now() - timedelta(days=30),
                'end_date': datetime.now()
            },
            filters={},
            sections=[],
            visualizations=[],
            recipients=[]
        )
        
        report = engine.generate_report(config, sample_data)
        
        assert report is not None
        assert report.file_data is not None
        
        # Verify JSON is valid
        json_data = json.loads(report.file_data.decode('utf-8'))
        assert 'id' in json_data
        assert 'config' in json_data
        assert 'sections' in json_data
    
    def test_automated_scheduler_initialization(self):
        """Test automated scheduler initialization"""
        engine = ComprehensiveReportingEngine()
        scheduler = AutomatedReportScheduler(engine)
        
        assert scheduler is not None
        assert scheduler.reporting_engine == engine
        assert scheduler.scheduled_reports == {}
        assert scheduler.is_running == False
    
    def test_scheduled_report_creation(self, sample_data):
        """Test creating scheduled reports"""
        engine = ComprehensiveReportingEngine()
        scheduler = AutomatedReportScheduler(engine)
        
        report_config = ReportConfig(
            report_type=ReportType.PERFORMANCE_METRICS,
            format=ReportFormat.PDF,
            title="Daily Performance Report",
            description="Automated daily performance metrics",
            date_range={
                'start_date': datetime.now() - timedelta(days=1),
                'end_date': datetime.now()
            },
            filters={},
            sections=[],
            visualizations=[],
            recipients=['manager@example.com']
        )
        
        scheduled_config = ScheduledReportConfig(
            id="test_daily_report",
            name="Daily Performance Report",
            description="Automated daily report",
            report_config=report_config,
            frequency=ScheduleFrequency.DAILY,
            schedule_time="09:00",
            delivery_methods=[DeliveryMethod.EMAIL],
            delivery_config={'recipients': ['manager@example.com']},
            data_source_config={'metrics': {}}
        )
        
        success = scheduler.add_scheduled_report(scheduled_config)
        
        assert success == True
        assert "test_daily_report" in scheduler.scheduled_reports
        assert scheduler.scheduled_reports["test_daily_report"].name == "Daily Performance Report"
    
    def test_statistical_analysis_comprehensive(self, sample_dataframe):
        """Test comprehensive statistical analysis"""
        engine = AdvancedStatisticalAnalytics()
        
        results = engine.comprehensive_analysis(sample_dataframe, target_column='revenue')
        
        assert 'descriptive_statistics' in results
        assert 'correlation_analysis' in results
        assert 'distribution_analysis' in results
        assert 'outlier_detection' in results
        assert 'normality_tests' in results
        assert 'cluster_analysis' in results
        assert 'trend_analysis' in results
        assert 'feature_importance' in results
        
        # Verify descriptive statistics
        desc_stats = results['descriptive_statistics']
        assert 'column_statistics' in desc_stats
        assert 'revenue' in desc_stats['column_statistics']
        assert 'mean' in desc_stats['column_statistics']['revenue']
        assert 'std' in desc_stats['column_statistics']['revenue']
    
    def test_outlier_detection_methods(self, sample_dataframe):
        """Test different outlier detection methods"""
        engine = AdvancedStatisticalAnalytics()
        
        # Test isolation forest
        result_if = engine.detect_outliers(sample_dataframe, method='isolation_forest')
        assert result_if.method == 'isolation_forest'
        assert isinstance(result_if.anomalies, list)
        assert isinstance(result_if.anomaly_scores, list)
        
        # Test statistical method
        result_stat = engine.detect_outliers(sample_dataframe, method='statistical')
        assert result_stat.method == 'statistical'
        assert isinstance(result_stat.anomalies, list)
        
        # Test IQR method
        result_iqr = engine.detect_outliers(sample_dataframe, method='iqr')
        assert result_iqr.method == 'iqr'
        assert isinstance(result_iqr.anomalies, list)
    
    def test_cluster_analysis_methods(self, sample_dataframe):
        """Test different clustering methods"""
        engine = AdvancedStatisticalAnalytics()
        
        # Test K-means
        result_kmeans = engine.cluster_analysis(sample_dataframe, method='kmeans', n_clusters=3)
        assert result_kmeans.method == 'kmeans'
        assert result_kmeans.n_clusters == 3
        assert len(result_kmeans.cluster_labels) == len(sample_dataframe)
        assert len(result_kmeans.cluster_centers) == 3
        
        # Test DBSCAN
        result_dbscan = engine.cluster_analysis(sample_dataframe, method='dbscan')
        assert result_dbscan.method == 'dbscan'
        assert len(result_dbscan.cluster_labels) == len(sample_dataframe)
    
    def test_trend_analysis(self, sample_dataframe):
        """Test trend analysis functionality"""
        engine = AdvancedStatisticalAnalytics()
        
        result = engine.trend_analysis(sample_dataframe, target_column='revenue')
        
        assert result.trend_direction in ['increasing', 'decreasing', 'stable']
        assert isinstance(result.trend_strength, float)
        assert isinstance(result.residuals, list)
        assert isinstance(result.forecast, list)
        assert isinstance(result.confidence_intervals, list)
        assert len(result.forecast) > 0
    
    def test_executive_summary_generation(self, sample_data):
        """Test executive summary generation"""
        generator = ExecutiveSummaryGenerator()
        
        summary = generator.generate_executive_summary(
            sample_data,
            SummaryType.COMPREHENSIVE
        )
        
        assert summary is not None
        assert summary.title is not None
        assert summary.summary_type == SummaryType.COMPREHENSIVE
        assert summary.executive_overview is not None
        assert len(summary.key_findings) > 0
        assert len(summary.recommendations) > 0
        assert isinstance(summary.confidence_score, float)
        assert 0 <= summary.confidence_score <= 1
    
    def test_executive_summary_different_types(self, sample_data):
        """Test different types of executive summaries"""
        generator = ExecutiveSummaryGenerator()
        
        # Test performance summary
        perf_summary = generator.generate_executive_summary(
            sample_data,
            SummaryType.PERFORMANCE
        )
        assert perf_summary.summary_type == SummaryType.PERFORMANCE
        
        # Test financial summary
        fin_summary = generator.generate_executive_summary(
            sample_data,
            SummaryType.FINANCIAL
        )
        assert fin_summary.summary_type == SummaryType.FINANCIAL
        
        # Test risk summary
        risk_summary = generator.generate_executive_summary(
            sample_data,
            SummaryType.RISK
        )
        assert risk_summary.summary_type == SummaryType.RISK
    
    def test_api_report_generation(self, client):
        """Test API endpoint for report generation"""
        request_data = {
            "report_type": "comprehensive",
            "format": "pdf",
            "title": "Test API Report",
            "description": "Test report via API",
            "date_range": {
                "start_date": (datetime.now() - timedelta(days=30)).isoformat(),
                "end_date": datetime.now().isoformat()
            },
            "filters": {},
            "sections": ["overview", "metrics"],
            "visualizations": ["charts"],
            "recipients": ["test@example.com"]
        }
        
        response = client.post("/api/advanced-analytics/reports/generate", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert "report_id" in data
        assert data["status"] == "completed"
        assert "download_url" in data
    
    def test_api_statistical_analysis(self, client, sample_dataframe):
        """Test API endpoint for statistical analysis"""
        request_data = {
            "data": sample_dataframe.to_dict('records'),
            "target_column": "revenue",
            "analysis_types": ["comprehensive"]
        }
        
        response = client.post("/api/advanced-analytics/analysis/statistical", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert "analysis_results" in data
        assert "insights" in data
        assert "data_summary" in data
    
    def test_api_executive_summary(self, client, sample_data):
        """Test API endpoint for executive summary"""
        request_data = {
            "data": sample_data,
            "summary_type": "comprehensive",
            "focus_areas": ["financial", "performance"]
        }
        
        response = client.post("/api/advanced-analytics/summaries/executive", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert "summary" in data
        assert "key_findings" in data["summary"]
        assert "recommendations" in data["summary"]
        assert "confidence_score" in data["summary"]
    
    def test_api_mobile_dashboard(self, client):
        """Test API endpoint for mobile dashboard"""
        response = client.get("/api/advanced-analytics/dashboard/mobile")
        
        assert response.status_code == 200
        data = response.json()
        assert "metrics" in data
        assert "key_findings" in data
        assert "recommendations" in data
        assert "confidence_score" in data
        assert isinstance(data["metrics"], list)
        assert len(data["metrics"]) > 0
    
    def test_api_scheduled_reports(self, client):
        """Test API endpoints for scheduled reports"""
        # Test creating scheduled report
        request_data = {
            "name": "Test Scheduled Report",
            "description": "Test scheduled report via API",
            "report_config": {
                "report_type": "performance",
                "format": "pdf",
                "title": "Scheduled Performance Report",
                "description": "Automated performance report",
                "date_range": {
                    "start_date": (datetime.now() - timedelta(days=1)).isoformat(),
                    "end_date": datetime.now().isoformat()
                },
                "filters": {},
                "sections": [],
                "visualizations": [],
                "recipients": ["test@example.com"]
            },
            "frequency": "daily",
            "schedule_time": "09:00",
            "delivery_methods": ["email"],
            "delivery_config": {"recipients": ["test@example.com"]},
            "data_source_config": {"metrics": {}}
        }
        
        response = client.post("/api/advanced-analytics/reports/schedule", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert "scheduled_report_id" in data
        assert data["status"] == "scheduled"
        
        # Test getting scheduled reports
        response = client.get("/api/advanced-analytics/reports/scheduled")
        
        assert response.status_code == 200
        data = response.json()
        assert "scheduled_reports" in data
        assert isinstance(data["scheduled_reports"], list)
    
    def test_api_health_check(self, client):
        """Test API health check endpoint"""
        response = client.get("/api/advanced-analytics/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "services" in data
        assert "timestamp" in data
    
    def test_end_to_end_workflow(self, sample_data, sample_dataframe):
        """Test complete end-to-end analytics workflow"""
        # 1. Initialize engines
        reporting_engine = ComprehensiveReportingEngine()
        statistical_engine = AdvancedStatisticalAnalytics()
        summary_generator = ExecutiveSummaryGenerator()
        scheduler = AutomatedReportScheduler(reporting_engine)
        
        # 2. Perform statistical analysis
        statistical_results = statistical_engine.comprehensive_analysis(
            sample_dataframe, 
            target_column='revenue'
        )
        
        assert statistical_results is not None
        assert 'descriptive_statistics' in statistical_results
        
        # 3. Generate executive summary
        enhanced_data = {**sample_data, 'statistical_analysis': statistical_results}
        
        summary = summary_generator.generate_executive_summary(
            enhanced_data,
            SummaryType.COMPREHENSIVE
        )
        
        assert summary is not None
        assert len(summary.key_findings) > 0
        assert len(summary.recommendations) > 0
        
        # 4. Generate comprehensive report
        config = ReportConfig(
            report_type=ReportType.COMPREHENSIVE,
            format=ReportFormat.PDF,
            title="End-to-End Test Report",
            description="Complete analytics workflow test",
            date_range={
                'start_date': datetime.now() - timedelta(days=30),
                'end_date': datetime.now()
            },
            filters={},
            sections=['overview', 'analytics', 'summary'],
            visualizations=['charts', 'tables'],
            recipients=['test@example.com']
        )
        
        report_data = {
            **enhanced_data,
            'executive_summary': {
                'key_findings': [
                    {
                        'title': finding.title,
                        'description': finding.description,
                        'priority': finding.priority.value
                    }
                    for finding in summary.key_findings
                ],
                'recommendations': [
                    {
                        'title': rec.title,
                        'description': rec.description,
                        'priority': rec.priority.value
                    }
                    for rec in summary.recommendations
                ]
            }
        }
        
        report = reporting_engine.generate_report(config, report_data)
        
        assert report is not None
        assert report.file_data is not None
        assert len(report.sections) > 0
        
        # 5. Schedule automated report
        scheduled_config = ScheduledReportConfig(
            id="e2e_test_report",
            name="End-to-End Test Report",
            description="Automated end-to-end test report",
            report_config=config,
            frequency=ScheduleFrequency.DAILY,
            schedule_time="09:00",
            delivery_methods=[DeliveryMethod.EMAIL],
            delivery_config={'recipients': ['test@example.com']},
            data_source_config={'test': {}}
        )
        
        success = scheduler.add_scheduled_report(scheduled_config)
        assert success == True
        
        # 6. Verify complete workflow
        scheduled_reports = scheduler.get_scheduled_reports()
        assert len(scheduled_reports) > 0
        assert any(r.id == "e2e_test_report" for r in scheduled_reports)
    
    def test_error_handling(self):
        """Test error handling in various scenarios"""
        engine = ComprehensiveReportingEngine()
        
        # Test with invalid report type
        with pytest.raises(Exception):
            config = ReportConfig(
                report_type="invalid_type",  # This should cause an error
                format=ReportFormat.PDF,
                title="Error Test",
                description="Test error handling",
                date_range={
                    'start_date': datetime.now() - timedelta(days=30),
                    'end_date': datetime.now()
                },
                filters={},
                sections=[],
                visualizations=[],
                recipients=[]
            )
        
        # Test statistical analysis with empty data
        statistical_engine = AdvancedStatisticalAnalytics()
        empty_df = pd.DataFrame()
        
        results = statistical_engine.comprehensive_analysis(empty_df)
        assert 'error' in results['descriptive_statistics']
    
    def test_performance_with_large_dataset(self):
        """Test performance with larger datasets"""
        # Create larger dataset
        large_df = pd.DataFrame({
            'metric_1': np.random.normal(100, 15, 1000),
            'metric_2': np.random.normal(50, 10, 1000),
            'metric_3': np.random.exponential(2, 1000),
            'metric_4': np.random.uniform(0, 100, 1000)
        })
        
        statistical_engine = AdvancedStatisticalAnalytics()
        
        # Measure analysis time
        start_time = datetime.now()
        results = statistical_engine.comprehensive_analysis(large_df, target_column='metric_1')
        end_time = datetime.now()
        
        analysis_time = (end_time - start_time).total_seconds()
        
        # Should complete within reasonable time (adjust threshold as needed)
        assert analysis_time < 30  # 30 seconds max
        assert results is not None
        assert 'descriptive_statistics' in results
    
    @pytest.mark.asyncio
    async def test_concurrent_report_generation(self, sample_data):
        """Test concurrent report generation"""
        engine = ComprehensiveReportingEngine()
        
        async def generate_report(report_id):
            config = ReportConfig(
                report_type=ReportType.PERFORMANCE_METRICS,
                format=ReportFormat.JSON,
                title=f"Concurrent Test Report {report_id}",
                description=f"Test concurrent generation {report_id}",
                date_range={
                    'start_date': datetime.now() - timedelta(days=30),
                    'end_date': datetime.now()
                },
                filters={},
                sections=[],
                visualizations=[],
                recipients=[]
            )
            
            return engine.generate_report(config, sample_data)
        
        # Generate multiple reports concurrently
        tasks = [generate_report(i) for i in range(5)]
        reports = await asyncio.gather(*tasks)
        
        # Verify all reports were generated successfully
        assert len(reports) == 5
        for report in reports:
            assert report is not None
            assert report.file_data is not None
            assert report.id is not None

if __name__ == "__main__":
    pytest.main([__file__, "-v"])