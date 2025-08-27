"""
End-to-End Integration Tests for Advanced Analytics and Reporting

This module provides comprehensive tests for the complete reporting workflows
including report generation, scheduling, statistical analysis, and interactive reports.
"""

import pytest
import asyncio
import json
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any
import tempfile
import os

from scrollintel.engines.comprehensive_reporting_engine import (
    ComprehensiveReportingEngine, ReportConfig, ReportType, ReportFormat
)
from scrollintel.core.automated_report_scheduler import (
    AutomatedReportScheduler, ReportSchedule, ScheduleConfig, DeliveryConfig,
    ScheduleFrequency, DeliveryMethod, ScheduleStatus
)
from scrollintel.engines.advanced_statistical_analytics import (
    AdvancedStatisticalAnalytics, AnalysisConfig, AnalysisType
)
from scrollintel.engines.executive_summary_generator import (
    ExecutiveSummaryGenerator, SummaryType
)
from scrollintel.engines.interactive_report_builder import (
    InteractiveReportBuilder, ComponentType, ChartType, LayoutType,
    ComponentConfig, ChartConfig
)


class TestAdvancedAnalyticsIntegration:
    """Integration tests for advanced analytics and reporting system"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        return pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=100, freq='D'),
            'revenue': [1000 + i * 10 + (i % 7) * 50 for i in range(100)],
            'users': [500 + i * 5 + (i % 3) * 20 for i in range(100)],
            'conversion_rate': [0.02 + (i % 10) * 0.001 for i in range(100)],
            'cost': [800 + i * 8 + (i % 5) * 30 for i in range(100)]
        })
    
    @pytest.fixture
    def reporting_engine(self):
        """Create reporting engine instance"""
        return ComprehensiveReportingEngine()
    
    @pytest.fixture
    def scheduler(self, reporting_engine):
        """Create scheduler instance"""
        return AutomatedReportScheduler(reporting_engine)
    
    @pytest.fixture
    def analytics_engine(self):
        """Create analytics engine instance"""
        return AdvancedStatisticalAnalytics()
    
    @pytest.fixture
    def summary_generator(self):
        """Create summary generator instance"""
        return ExecutiveSummaryGenerator()
    
    @pytest.fixture
    def report_builder(self):
        """Create report builder instance"""
        return InteractiveReportBuilder()
    
    @pytest.mark.asyncio
    async def test_comprehensive_report_generation_workflow(self, reporting_engine, sample_data):
        """Test complete report generation workflow"""
        
        # Test PDF report generation
        pdf_config = ReportConfig(
            report_type=ReportType.EXECUTIVE_SUMMARY,
            format=ReportFormat.PDF,
            title="Executive Summary Report",
            description="Comprehensive executive summary with key metrics",
            data_sources=["sample_data"],
            filters={"date_range": "last_30_days"},
            date_range={
                "start": datetime.utcnow() - timedelta(days=30),
                "end": datetime.utcnow()
            }
        )
        
        pdf_report = await reporting_engine.generate_report(pdf_config)
        
        assert pdf_report is not None
        assert pdf_report.report_id is not None
        assert pdf_report.format == ReportFormat.PDF
        assert len(pdf_report.content) > 0
        assert pdf_report.file_size > 0
        
        # Test Excel report generation
        excel_config = ReportConfig(
            report_type=ReportType.DETAILED_ANALYTICS,
            format=ReportFormat.EXCEL,
            title="Detailed Analytics Report",
            description="Comprehensive analytics with detailed breakdowns",
            data_sources=["sample_data"],
            filters={},
            date_range={
                "start": datetime.utcnow() - timedelta(days=30),
                "end": datetime.utcnow()
            }
        )
        
        excel_report = await reporting_engine.generate_report(excel_config)
        
        assert excel_report is not None
        assert excel_report.format == ReportFormat.EXCEL
        assert len(excel_report.content) > 0
        
        # Test Web report generation
        web_config = ReportConfig(
            report_type=ReportType.PERFORMANCE_METRICS,
            format=ReportFormat.WEB,
            title="Performance Metrics Dashboard",
            description="Interactive web-based performance metrics",
            data_sources=["sample_data"],
            filters={},
            date_range={
                "start": datetime.utcnow() - timedelta(days=7),
                "end": datetime.utcnow()
            }
        )
        
        web_report = await reporting_engine.generate_report(web_config)
        
        assert web_report is not None
        assert web_report.format == ReportFormat.WEB
        assert b"<!DOCTYPE html>" in web_report.content
        
        # Test JSON report generation
        json_config = ReportConfig(
            report_type=ReportType.ROI_ANALYSIS,
            format=ReportFormat.JSON,
            title="ROI Analysis Report",
            description="Return on investment analysis in JSON format",
            data_sources=["sample_data"],
            filters={},
            date_range={
                "start": datetime.utcnow() - timedelta(days=90),
                "end": datetime.utcnow()
            }
        )
        
        json_report = await reporting_engine.generate_report(json_config)
        
        assert json_report is not None
        assert json_report.format == ReportFormat.JSON
        
        # Validate JSON content
        json_content = json.loads(json_report.content.decode('utf-8'))
        assert "title" in json_content
        assert "sections" in json_content
        
        # Test report retrieval
        retrieved_report = await reporting_engine.get_report(pdf_report.report_id)
        assert retrieved_report is not None
        assert retrieved_report.report_id == pdf_report.report_id
        
        # Test report listing
        reports = await reporting_engine.list_reports()
        assert len(reports) >= 4  # At least the 4 reports we created
    
    @pytest.mark.asyncio
    async def test_automated_scheduling_workflow(self, scheduler):
        """Test complete automated scheduling workflow"""
        
        # Configure SMTP for testing (mock)
        scheduler.configure_smtp(
            host="localhost",
            port=587,
            username="test@example.com",
            password="password",
            use_tls=True
        )
        
        # Create schedule configuration
        schedule_config = ScheduleConfig(
            frequency=ScheduleFrequency.DAILY,
            start_date=datetime.utcnow(),
            end_date=datetime.utcnow() + timedelta(days=30),
            time_of_day="09:00",
            timezone="UTC"
        )
        
        # Create delivery configuration
        delivery_config = DeliveryConfig(
            method=DeliveryMethod.EMAIL,
            recipients=["test@example.com", "manager@example.com"],
            settings={
                "subject_template": "Daily Report - {date}",
                "body_template": "Please find attached the daily report."
            }
        )
        
        # Create report schedule
        schedule = ReportSchedule(
            schedule_id="test_schedule_1",
            name="Daily Executive Report",
            description="Automated daily executive summary report",
            report_config={
                "report_type": "executive_summary",
                "format": "pdf",
                "title": "Daily Executive Summary",
                "description": "Automated daily summary of key metrics",
                "data_sources": ["daily_metrics"],
                "filters": {"date_range": "yesterday"},
                "date_range": {
                    "start": datetime.utcnow() - timedelta(days=1),
                    "end": datetime.utcnow()
                }
            },
            schedule_config=schedule_config,
            delivery_config=delivery_config,
            status=ScheduleStatus.ACTIVE,
            created_at=datetime.utcnow()
        )
        
        # Create schedule
        schedule_id = await scheduler.create_schedule(schedule)
        assert schedule_id is not None
        
        # Test schedule retrieval
        retrieved_schedule = await scheduler.get_schedule(schedule_id)
        assert retrieved_schedule is not None
        assert retrieved_schedule.name == "Daily Executive Report"
        assert retrieved_schedule.status == ScheduleStatus.ACTIVE
        
        # Test schedule listing
        schedules = await scheduler.list_schedules()
        assert len(schedules) >= 1
        
        # Test schedule pause/resume
        pause_result = await scheduler.pause_schedule(schedule_id)
        assert pause_result is True
        
        paused_schedule = await scheduler.get_schedule(schedule_id)
        assert paused_schedule.status == ScheduleStatus.PAUSED
        
        resume_result = await scheduler.resume_schedule(schedule_id)
        assert resume_result is True
        
        resumed_schedule = await scheduler.get_schedule(schedule_id)
        assert resumed_schedule.status == ScheduleStatus.ACTIVE
        
        # Test execution history
        executions = await scheduler.get_execution_history(schedule_id)
        assert isinstance(executions, list)
        
        # Test schedule deletion
        delete_result = await scheduler.delete_schedule(schedule_id)
        assert delete_result is True
        
        deleted_schedule = await scheduler.get_schedule(schedule_id)
        assert deleted_schedule is None
    
    @pytest.mark.asyncio
    async def test_statistical_analysis_workflow(self, analytics_engine, sample_data):
        """Test complete statistical analysis workflow"""
        
        # Create analysis configuration
        config = AnalysisConfig(
            analysis_types=[
                AnalysisType.DESCRIPTIVE,
                AnalysisType.CORRELATION,
                AnalysisType.TREND_ANALYSIS,
                AnalysisType.ANOMALY_DETECTION
            ],
            confidence_level=0.95,
            significance_threshold=0.05,
            min_data_points=30
        )
        
        # Perform comprehensive analysis
        results = await analytics_engine.perform_comprehensive_analysis(sample_data, config)
        
        # Validate results structure
        assert isinstance(results, dict)
        assert "descriptive" in results
        assert "correlation" in results
        assert "trend_analysis" in results
        
        # Validate descriptive analysis
        descriptive_results = results["descriptive"]
        assert len(descriptive_results) > 0
        
        for result in descriptive_results:
            assert result.analysis_type == AnalysisType.DESCRIPTIVE
            assert "mean" in result.result
            assert "std" in result.result
            assert "min" in result.result
            assert "max" in result.result
            assert result.confidence_level == 0.95
        
        # Validate correlation analysis
        correlation_results = results["correlation"]
        assert len(correlation_results) > 0
        
        for result in correlation_results:
            assert result.analysis_type == AnalysisType.CORRELATION
            assert "correlation" in result.result
            assert abs(result.result["correlation"]) <= 1.0
        
        # Generate ML insights
        ml_insights = await analytics_engine.generate_ml_insights(sample_data, results)
        
        assert isinstance(ml_insights, list)
        assert len(ml_insights) > 0
        
        for insight in ml_insights:
            assert hasattr(insight, 'title')
            assert hasattr(insight, 'description')
            assert hasattr(insight, 'confidence_score')
            assert hasattr(insight, 'impact_score')
            assert 0 <= insight.confidence_score <= 1
            assert 0 <= insight.impact_score <= 1
    
    @pytest.mark.asyncio
    async def test_executive_summary_workflow(self, summary_generator):
        """Test complete executive summary generation workflow"""
        
        # Create sample analytics data
        analytics_data = {
            "statistical_analysis": {
                "correlation": [
                    {
                        "metric_name": "Revenue vs Users",
                        "result": {"correlation": 0.85, "strength": "strong"},
                        "confidence_level": 0.95,
                        "p_value": 0.001,
                        "interpretation": "Strong positive correlation"
                    }
                ],
                "trend_analysis": [
                    {
                        "metric_name": "Revenue",
                        "result": {
                            "trend_direction": "increasing",
                            "trend_strength": 0.78
                        },
                        "confidence_level": 0.90,
                        "p_value": 0.02
                    }
                ]
            },
            "ml_insights": [
                {
                    "title": "Revenue Growth Acceleration",
                    "description": "Revenue growth is accelerating beyond historical trends",
                    "confidence_score": 0.88,
                    "impact_score": 0.92,
                    "action_items": ["Scale marketing efforts", "Increase inventory"]
                }
            ],
            "roi_analysis": {
                "total_roi": 0.25,
                "total_investment": 100000,
                "total_returns": 125000
            },
            "dashboard_metrics": {
                "kpis": {
                    "Monthly Revenue": {
                        "current_value": 125000,
                        "target_value": 120000,
                        "previous_value": 115000
                    }
                }
            }
        }
        
        # Generate executive summary
        summary = await summary_generator.generate_executive_summary(
            data=analytics_data,
            summary_type=SummaryType.PERFORMANCE_OVERVIEW,
            target_audience="executive",
            custom_focus=["revenue", "growth"]
        )
        
        # Validate summary structure
        assert summary is not None
        assert summary.title is not None
        assert summary.executive_overview is not None
        assert isinstance(summary.key_highlights, list)
        assert isinstance(summary.critical_insights, list)
        assert isinstance(summary.performance_metrics, dict)
        assert isinstance(summary.trends_and_patterns, list)
        assert isinstance(summary.risks_and_opportunities, dict)
        assert isinstance(summary.strategic_recommendations, list)
        assert isinstance(summary.next_steps, list)
        assert 0 <= summary.confidence_score <= 1
        
        # Validate insights
        assert len(summary.critical_insights) > 0
        
        for insight in summary.critical_insights:
            assert insight.title is not None
            assert insight.summary is not None
            assert insight.detailed_explanation is not None
            assert isinstance(insight.key_metrics, dict)
            assert insight.urgency is not None
            assert 0 <= insight.confidence_level <= 1
            assert insight.business_impact is not None
            assert isinstance(insight.recommended_actions, list)
        
        # Test different summary types
        financial_summary = await summary_generator.generate_executive_summary(
            data=analytics_data,
            summary_type=SummaryType.FINANCIAL_SUMMARY,
            target_audience="board"
        )
        
        assert financial_summary.title != summary.title
        assert "Financial" in financial_summary.title or "Board" in financial_summary.title
    
    @pytest.mark.asyncio
    async def test_interactive_report_builder_workflow(self, report_builder):
        """Test complete interactive report builder workflow"""
        
        # Create interactive report
        report_id = await report_builder.create_report(
            name="Sales Performance Dashboard",
            description="Interactive dashboard for sales performance metrics",
            created_by="test_user",
            layout_type=LayoutType.TWO_COLUMN
        )
        
        assert report_id is not None
        
        # Add text component
        text_component_id = await report_builder.add_component(
            report_id=report_id,
            component_type=ComponentType.TEXT,
            title="Executive Summary",
            position={"x": 0, "y": 0, "width": 600, "height": 200},
            properties={
                "content": "This dashboard provides key insights into sales performance.",
                "font_size": 16,
                "text_align": "left"
            }
        )
        
        assert text_component_id is not None
        
        # Add chart component
        chart_config = ChartConfig(
            chart_type=ChartType.LINE,
            data_columns={"x": "date", "y": "revenue"},
            aggregation="sum",
            color_scheme="blue",
            show_legend=True,
            show_grid=True,
            animation=True
        )
        
        chart_component_id = await report_builder.create_chart_component(
            report_id=report_id,
            title="Revenue Trend",
            position={"x": 0, "y": 220, "width": 600, "height": 400},
            chart_config=chart_config,
            data_source="sales_data"
        )
        
        assert chart_component_id is not None
        
        # Add metric component
        metric_component_id = await report_builder.add_component(
            report_id=report_id,
            component_type=ComponentType.METRIC,
            title="Total Revenue",
            position={"x": 620, "y": 0, "width": 300, "height": 150},
            properties={
                "value_column": "revenue",
                "format": "currency",
                "show_trend": True,
                "trend_period": "previous_month"
            }
        )
        
        assert metric_component_id is not None
        
        # Add table component
        table_component_id = await report_builder.add_component(
            report_id=report_id,
            component_type=ComponentType.TABLE,
            title="Top Products",
            position={"x": 620, "y": 170, "width": 300, "height": 300},
            properties={
                "columns": ["Product", "Revenue", "Units Sold"],
                "sortable": True,
                "filterable": True,
                "paginated": True,
                "page_size": 10
            }
        )
        
        assert table_component_id is not None
        
        # Add data source
        data_source_result = await report_builder.add_data_source(
            report_id=report_id,
            source_name="sales_data",
            source_config={
                "type": "database",
                "connection": "sales_db",
                "query": "SELECT * FROM sales_metrics WHERE date >= '2024-01-01'"
            }
        )
        
        assert data_source_result is True
        
        # Add filter
        filter_result = await report_builder.add_filter(
            report_id=report_id,
            filter_config={
                "name": "Date Range",
                "type": "date_range",
                "column": "date",
                "default_value": "last_30_days"
            }
        )
        
        assert filter_result is True
        
        # Update component
        update_result = await report_builder.update_component(
            report_id=report_id,
            component_id=text_component_id,
            updates={
                "properties": {
                    "content": "Updated executive summary with latest insights.",
                    "font_size": 18,
                    "font_weight": "bold"
                }
            }
        )
        
        assert update_result is True
        
        # Update layout
        layout_update_result = await report_builder.update_layout(
            report_id=report_id,
            layout_updates={
                "global_styling": {
                    "background_color": "#f8f9fa",
                    "font_family": "Helvetica, Arial, sans-serif"
                }
            }
        )
        
        assert layout_update_result is True
        
        # Generate HTML
        html_content = await report_builder.generate_report_html(report_id)
        
        assert html_content is not None
        assert "<!DOCTYPE html>" in html_content
        assert "Sales Performance Dashboard" in html_content
        
        # Generate JSON
        json_content = await report_builder.generate_report_json(report_id)
        
        assert json_content is not None
        assert "report_id" in json_content
        assert "components" in json_content
        assert len(json_content["components"]) == 4  # 4 components added
        
        # Clone report
        cloned_report_id = await report_builder.clone_report(
            report_id=report_id,
            new_name="Cloned Sales Dashboard",
            created_by="test_user"
        )
        
        assert cloned_report_id is not None
        assert cloned_report_id != report_id
        
        # Save as template
        template_id = await report_builder.save_as_template(
            report_id=report_id,
            template_name="Sales Dashboard Template",
            template_description="Template for sales performance dashboards"
        )
        
        assert template_id is not None
        
        # List reports
        reports = await report_builder.list_reports(created_by="test_user")
        
        assert len(reports) >= 2  # Original and cloned report
        
        # List templates
        templates = await report_builder.list_templates()
        
        assert len(templates) >= 1
        
        # Remove component
        remove_result = await report_builder.remove_component(
            report_id=report_id,
            component_id=table_component_id
        )
        
        assert remove_result is True
        
        # Verify component removal
        updated_report = await report_builder.get_report(report_id)
        assert len(updated_report.components) == 3  # One component removed
        
        # Delete cloned report
        delete_result = await report_builder.delete_report(cloned_report_id)
        assert delete_result is True
    
    @pytest.mark.asyncio
    async def test_end_to_end_workflow_integration(
        self, 
        reporting_engine, 
        scheduler, 
        analytics_engine, 
        summary_generator, 
        report_builder,
        sample_data
    ):
        """Test complete end-to-end workflow integration"""
        
        # Step 1: Perform statistical analysis
        analysis_config = AnalysisConfig(
            analysis_types=[
                AnalysisType.DESCRIPTIVE,
                AnalysisType.CORRELATION,
                AnalysisType.TREND_ANALYSIS
            ],
            confidence_level=0.95
        )
        
        statistical_results = await analytics_engine.perform_comprehensive_analysis(
            sample_data, analysis_config
        )
        
        # Step 2: Generate ML insights
        ml_insights = await analytics_engine.generate_ml_insights(
            sample_data, statistical_results
        )
        
        # Step 3: Create comprehensive analytics data
        analytics_data = {
            "statistical_analysis": statistical_results,
            "ml_insights": [
                {
                    "title": insight.title,
                    "description": insight.description,
                    "confidence_score": insight.confidence_score,
                    "impact_score": insight.impact_score,
                    "action_items": insight.action_items or []
                }
                for insight in ml_insights
            ],
            "roi_analysis": {
                "total_roi": 0.18,
                "total_investment": 150000,
                "total_returns": 177000
            },
            "dashboard_metrics": {
                "kpis": {
                    "Revenue": {"current_value": 125000, "target_value": 120000},
                    "Users": {"current_value": 850, "target_value": 800}
                }
            }
        }
        
        # Step 4: Generate executive summary
        executive_summary = await summary_generator.generate_executive_summary(
            data=analytics_data,
            summary_type=SummaryType.PERFORMANCE_OVERVIEW,
            target_audience="executive"
        )
        
        # Step 5: Create comprehensive report with summary
        report_config = ReportConfig(
            report_type=ReportType.EXECUTIVE_SUMMARY,
            format=ReportFormat.PDF,
            title="Comprehensive Analytics Report",
            description="Complete analytics report with statistical analysis and executive summary",
            data_sources=["analytics_data"],
            filters={},
            date_range={
                "start": datetime.utcnow() - timedelta(days=30),
                "end": datetime.utcnow()
            }
        )
        
        comprehensive_report = await reporting_engine.generate_report(report_config)
        
        # Step 6: Create interactive dashboard
        dashboard_id = await report_builder.create_report(
            name="Executive Analytics Dashboard",
            description="Interactive dashboard with comprehensive analytics",
            created_by="system",
            layout_type=LayoutType.GRID
        )
        
        # Add summary component
        await report_builder.add_component(
            report_id=dashboard_id,
            component_type=ComponentType.TEXT,
            title="Executive Summary",
            position={"x": 0, "y": 0, "width": 800, "height": 200},
            properties={"content": executive_summary.executive_overview}
        )
        
        # Add insights components
        y_position = 220
        for i, insight in enumerate(executive_summary.critical_insights[:3]):
            await report_builder.add_component(
                report_id=dashboard_id,
                component_type=ComponentType.TEXT,
                title=insight.title,
                position={"x": 0, "y": y_position, "width": 800, "height": 150},
                properties={
                    "content": f"{insight.summary}\n\nRecommended Actions: {', '.join(insight.recommended_actions)}"
                }
            )
            y_position += 170
        
        # Step 7: Schedule automated report generation
        schedule_config = ScheduleConfig(
            frequency=ScheduleFrequency.WEEKLY,
            start_date=datetime.utcnow(),
            day_of_week=1,  # Monday
            time_of_day="09:00"
        )
        
        delivery_config = DeliveryConfig(
            method=DeliveryMethod.EMAIL,
            recipients=["executive@company.com"],
            settings={"subject": "Weekly Analytics Report"}
        )
        
        schedule = ReportSchedule(
            schedule_id="weekly_analytics",
            name="Weekly Analytics Report",
            description="Automated weekly comprehensive analytics report",
            report_config={
                "report_type": "executive_summary",
                "format": "pdf",
                "title": "Weekly Analytics Summary",
                "description": "Automated weekly analytics report",
                "data_sources": ["weekly_data"],
                "filters": {},
                "date_range": {
                    "start": datetime.utcnow() - timedelta(days=7),
                    "end": datetime.utcnow()
                }
            },
            schedule_config=schedule_config,
            delivery_config=delivery_config,
            status=ScheduleStatus.ACTIVE,
            created_at=datetime.utcnow()
        )
        
        schedule_id = await scheduler.create_schedule(schedule)
        
        # Validate end-to-end workflow
        assert statistical_results is not None
        assert len(ml_insights) > 0
        assert executive_summary is not None
        assert comprehensive_report is not None
        assert dashboard_id is not None
        assert schedule_id is not None
        
        # Verify data flow integrity
        assert len(executive_summary.critical_insights) > 0
        assert executive_summary.confidence_score > 0
        assert comprehensive_report.file_size > 0
        
        # Verify dashboard creation
        dashboard = await report_builder.get_report(dashboard_id)
        assert dashboard is not None
        assert len(dashboard.components) >= 4  # Summary + 3 insights
        
        # Verify schedule creation
        created_schedule = await scheduler.get_schedule(schedule_id)
        assert created_schedule is not None
        assert created_schedule.status == ScheduleStatus.ACTIVE
        
        # Clean up
        await report_builder.delete_report(dashboard_id)
        await scheduler.delete_schedule(schedule_id)
    
    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self, reporting_engine, analytics_engine):
        """Test error handling and recovery mechanisms"""
        
        # Test invalid report configuration
        with pytest.raises(Exception):
            invalid_config = ReportConfig(
                report_type=ReportType.EXECUTIVE_SUMMARY,
                format=ReportFormat.PDF,
                title="",  # Invalid empty title
                description="Test report",
                data_sources=[],
                filters={},
                date_range={}
            )
            await reporting_engine.generate_report(invalid_config)
        
        # Test analysis with insufficient data
        small_data = pd.DataFrame({
            'value': [1, 2, 3]  # Only 3 data points
        })
        
        config = AnalysisConfig(
            analysis_types=[AnalysisType.DESCRIPTIVE],
            min_data_points=30
        )
        
        with pytest.raises(Exception):
            await analytics_engine.perform_comprehensive_analysis(small_data, config)
        
        # Test recovery with valid data
        valid_data = pd.DataFrame({
            'value': list(range(50))  # 50 data points
        })
        
        results = await analytics_engine.perform_comprehensive_analysis(valid_data, config)
        assert results is not None
        assert "descriptive" in results
    
    @pytest.mark.asyncio
    async def test_performance_and_scalability(self, reporting_engine, sample_data):
        """Test performance and scalability of the system"""
        
        # Test concurrent report generation
        tasks = []
        for i in range(5):
            config = ReportConfig(
                report_type=ReportType.PERFORMANCE_METRICS,
                format=ReportFormat.JSON,
                title=f"Performance Report {i}",
                description=f"Test report {i}",
                data_sources=["test_data"],
                filters={},
                date_range={
                    "start": datetime.utcnow() - timedelta(days=1),
                    "end": datetime.utcnow()
                }
            )
            tasks.append(reporting_engine.generate_report(config))
        
        # Execute concurrent report generation
        start_time = datetime.utcnow()
        reports = await asyncio.gather(*tasks)
        end_time = datetime.utcnow()
        
        # Validate all reports were generated
        assert len(reports) == 5
        for report in reports:
            assert report is not None
            assert report.report_id is not None
        
        # Check performance (should complete within reasonable time)
        execution_time = (end_time - start_time).total_seconds()
        assert execution_time < 30  # Should complete within 30 seconds
        
        # Test large dataset analysis
        large_data = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=1000, freq='D'),
            'metric1': [i + (i % 10) * 5 for i in range(1000)],
            'metric2': [i * 2 + (i % 7) * 3 for i in range(1000)],
            'metric3': [i * 0.5 + (i % 5) * 2 for i in range(1000)]
        })
        
        config = AnalysisConfig(
            analysis_types=[AnalysisType.DESCRIPTIVE, AnalysisType.CORRELATION],
            confidence_level=0.95
        )
        
        start_time = datetime.utcnow()
        results = await analytics_engine.perform_comprehensive_analysis(large_data, config)
        end_time = datetime.utcnow()
        
        # Validate results
        assert results is not None
        assert "descriptive" in results
        assert "correlation" in results
        
        # Check performance for large dataset
        analysis_time = (end_time - start_time).total_seconds()
        assert analysis_time < 60  # Should complete within 60 seconds


if __name__ == "__main__":
    pytest.main([__file__, "-v"])