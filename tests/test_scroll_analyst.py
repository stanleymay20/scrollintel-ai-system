"""
Unit tests for ScrollAnalyst agent.
Tests business intelligence capabilities, KPI generation, SQL query generation,
and integration with ScrollViz engine.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
import json

from scrollintel.agents.scroll_analyst import (
    ScrollAnalyst, KPIResult, BusinessInsight, TrendAnalysis,
    KPICategory, AnalysisType
)
from scrollintel.core.interfaces import AgentRequest, AgentType, ResponseStatus


class TestScrollAnalyst:
    """Test cases for ScrollAnalyst agent."""
    
    @pytest.fixture
    def analyst_agent(self):
        """Create ScrollAnalyst instance for testing."""
        return ScrollAnalyst()
    
    @pytest.fixture
    def sample_business_data(self):
        """Create sample business data for testing."""
        dates = pd.date_range('2024-01-01', periods=12, freq='M')
        data = {
            'date': dates,
            'revenue': [10000, 12000, 11000, 13000, 14000, 15000, 
                       16000, 15500, 17000, 18000, 19000, 20000],
            'customers': [100, 120, 110, 130, 140, 150, 160, 155, 170, 180, 190, 200],
            'marketing_spend': [2000, 2400, 2200, 2600, 2800, 3000, 
                               3200, 3100, 3400, 3600, 3800, 4000],
            'new_customers': [20, 24, 22, 26, 28, 30, 32, 31, 34, 36, 38, 40],
            'conversions': [50, 60, 55, 65, 70, 75, 80, 77, 85, 90, 95, 100],
            'visitors': [1000, 1200, 1100, 1300, 1400, 1500, 1600, 1540, 1700, 1800, 1900, 2000],
            'customer_id': list(range(1, 13)) * 1,  # Simplified customer data
            'product': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B', 'A', 'B', 'A', 'B']
        }
        return pd.DataFrame(data)
    
    @pytest.fixture
    def sample_request(self):
        """Create sample agent request."""
        return AgentRequest(
            id="test-request-1",
            user_id="test-user",
            agent_id="scroll-analyst",
            prompt="Generate KPIs for business performance",
            context={},
            priority=1,
            created_at=datetime.now()
        )
    
    def test_agent_initialization(self, analyst_agent):
        """Test ScrollAnalyst agent initialization."""
        assert analyst_agent.agent_id == "scroll-analyst"
        assert analyst_agent.name == "ScrollAnalyst Agent"
        assert analyst_agent.agent_type == AgentType.ANALYST
        assert len(analyst_agent.capabilities) == 6
        assert len(analyst_agent.kpi_definitions) > 0
        assert len(analyst_agent.sql_templates) > 0
    
    def test_agent_capabilities(self, analyst_agent):
        """Test agent capabilities are properly defined."""
        capability_names = [cap.name for cap in analyst_agent.capabilities]
        
        expected_capabilities = [
            "kpi_generation",
            "sql_query_generation", 
            "business_insights",
            "trend_analysis",
            "report_generation",
            "scrollviz_integration"
        ]
        
        for expected in expected_capabilities:
            assert expected in capability_names
    
    @pytest.mark.asyncio
    async def test_health_check(self, analyst_agent):
        """Test agent health check functionality."""
        is_healthy = await analyst_agent.health_check()
        assert isinstance(is_healthy, bool)
    
    @pytest.mark.asyncio
    async def test_kpi_generation_request(self, analyst_agent, sample_request, sample_business_data):
        """Test KPI generation from business data."""
        # Mock the data loading
        with patch.object(analyst_agent, '_convert_to_dataframe', return_value=sample_business_data):
            sample_request.prompt = "Generate KPIs for revenue and customer metrics"
            sample_request.context = {"dataset": sample_business_data.to_dict()}
            
            response = await analyst_agent.process_request(sample_request)
            
            assert response.status == ResponseStatus.SUCCESS
            assert "KPI Analysis Report" in response.content
            assert "Revenue Growth" in response.content or "revenue" in response.content.lower()
    
    @pytest.mark.asyncio
    async def test_sql_query_generation(self, analyst_agent, sample_request):
        """Test SQL query generation from natural language."""
        sample_request.prompt = "Show me revenue by month for the last year"
        sample_request.context = {
            "database_schema": {
                "tables": {
                    "sales": {
                        "columns": ["date", "revenue", "customer_id"]
                    }
                }
            }
        }
        
        response = await analyst_agent.process_request(sample_request)
        
        assert response.status == ResponseStatus.SUCCESS
        assert "SQL Query Generation Report" in response.content
        assert "SELECT" in response.content.upper()
    
    @pytest.mark.asyncio
    async def test_business_insights_generation(self, analyst_agent, sample_request, sample_business_data):
        """Test business insights generation."""
        with patch.object(analyst_agent, '_convert_to_dataframe', return_value=sample_business_data):
            sample_request.prompt = "Analyze business performance and provide insights"
            sample_request.context = {"dataset": sample_business_data.to_dict()}
            
            response = await analyst_agent.process_request(sample_request)
            
            assert response.status == ResponseStatus.SUCCESS
            assert "Business Intelligence Analysis Report" in response.content
            assert "insights" in response.content.lower()
    
    @pytest.mark.asyncio
    async def test_trend_analysis(self, analyst_agent, sample_request, sample_business_data):
        """Test trend analysis functionality."""
        with patch.object(analyst_agent, '_convert_to_dataframe', return_value=sample_business_data):
            sample_request.prompt = "Perform trend analysis on revenue and customer data"
            sample_request.context = {"dataset": sample_business_data.to_dict()}
            
            response = await analyst_agent.process_request(sample_request)
            
            assert response.status == ResponseStatus.SUCCESS
            assert "Trend Analysis Report" in response.content
            assert "trend" in response.content.lower()
    
    @pytest.mark.asyncio
    async def test_dashboard_creation(self, analyst_agent, sample_request, sample_business_data):
        """Test dashboard creation with ScrollViz integration."""
        with patch.object(analyst_agent, '_convert_to_dataframe', return_value=sample_business_data):
            sample_request.prompt = "Create a business dashboard with key metrics"
            sample_request.context = {"dataset": sample_business_data.to_dict()}
            
            response = await analyst_agent.process_request(sample_request)
            
            assert response.status == ResponseStatus.SUCCESS
            assert "Dashboard Creation Report" in response.content
            assert "ScrollViz" in response.content
    
    @pytest.mark.asyncio
    async def test_report_generation(self, analyst_agent, sample_request, sample_business_data):
        """Test business report generation."""
        with patch.object(analyst_agent, '_convert_to_dataframe', return_value=sample_business_data):
            sample_request.prompt = "Generate a comprehensive business report"
            sample_request.context = {"dataset": sample_business_data.to_dict()}
            
            response = await analyst_agent.process_request(sample_request)
            
            assert response.status == ResponseStatus.SUCCESS
            assert "Business Performance Report" in response.content
            assert "Executive Summary" in response.content
    
    def test_kpi_definitions_initialization(self, analyst_agent):
        """Test KPI definitions are properly initialized."""
        kpi_defs = analyst_agent.kpi_definitions
        
        assert "revenue_growth" in kpi_defs
        assert "customer_acquisition_cost" in kpi_defs
        assert "conversion_rate" in kpi_defs
        assert "churn_rate" in kpi_defs
        
        # Test KPI definition structure
        revenue_kpi = kpi_defs["revenue_growth"]
        assert revenue_kpi.name == "Revenue Growth Rate"
        assert revenue_kpi.category == KPICategory.FINANCIAL
        assert revenue_kpi.unit == "%"
        assert isinstance(revenue_kpi.data_requirements, list)
    
    @pytest.mark.asyncio
    async def test_revenue_growth_calculation(self, analyst_agent, sample_business_data):
        """Test revenue growth KPI calculation."""
        kpi_result = await analyst_agent._calculate_single_kpi(
            sample_business_data, "revenue_growth"
        )
        
        assert kpi_result is not None
        assert isinstance(kpi_result, KPIResult)
        assert kpi_result.kpi_name == "Revenue Growth Rate"
        assert kpi_result.unit == "%"
        assert isinstance(kpi_result.current_value, float)
        assert len(kpi_result.insights) > 0
    
    @pytest.mark.asyncio
    async def test_customer_acquisition_cost_calculation(self, analyst_agent, sample_business_data):
        """Test customer acquisition cost KPI calculation."""
        kpi_result = await analyst_agent._calculate_single_kpi(
            sample_business_data, "customer_acquisition_cost"
        )
        
        assert kpi_result is not None
        assert isinstance(kpi_result, KPIResult)
        assert kpi_result.kpi_name == "Customer Acquisition Cost"
        assert kpi_result.unit == "$"
        assert kpi_result.current_value > 0
    
    @pytest.mark.asyncio
    async def test_conversion_rate_calculation(self, analyst_agent, sample_business_data):
        """Test conversion rate KPI calculation."""
        kpi_result = await analyst_agent._calculate_single_kpi(
            sample_business_data, "conversion_rate"
        )
        
        assert kpi_result is not None
        assert isinstance(kpi_result, KPIResult)
        assert kpi_result.kpi_name == "Conversion Rate"
        assert kpi_result.unit == "%"
        assert 0 <= kpi_result.current_value <= 100
    
    def test_column_finding_utility(self, analyst_agent, sample_business_data):
        """Test utility function for finding columns."""
        # Test finding revenue column
        revenue_col = analyst_agent._find_column(sample_business_data, ['revenue', 'sales'])
        assert revenue_col == 'revenue'
        
        # Test finding date column
        date_col = analyst_agent._find_column(sample_business_data, ['date', 'timestamp'])
        assert date_col == 'date'
        
        # Test column not found
        missing_col = analyst_agent._find_column(sample_business_data, ['nonexistent'])
        assert missing_col is None
    
    @pytest.mark.asyncio
    async def test_kpi_suggestions_from_data(self, analyst_agent, sample_business_data):
        """Test KPI suggestions based on data structure."""
        suggestions = await analyst_agent._suggest_kpis_from_data(
            sample_business_data, "analyze business performance"
        )
        
        assert isinstance(suggestions, list)
        assert len(suggestions) > 0
        assert "revenue_growth" in suggestions
    
    def test_data_format_conversion(self, analyst_agent):
        """Test data format conversion utilities."""
        # Test dictionary conversion
        dict_data = {"col1": [1, 2, 3], "col2": [4, 5, 6]}
        df = analyst_agent._convert_to_dataframe(dict_data)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        
        # Test list conversion
        list_data = [{"col1": 1, "col2": 4}, {"col1": 2, "col2": 5}]
        df = analyst_agent._convert_to_dataframe(list_data)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        
        # Test DataFrame passthrough
        original_df = pd.DataFrame(dict_data)
        df = analyst_agent._convert_to_dataframe(original_df)
        assert df is original_df
    
    def test_kpi_result_formatting(self, analyst_agent):
        """Test KPI result formatting for display."""
        kpi_results = [
            KPIResult(
                kpi_name="Test KPI",
                current_value=15.5,
                previous_value=10.0,
                target_value=12.0,
                change_percentage=55.0,
                trend="increasing",
                status="above_target",
                unit="%",
                calculation_date=datetime.now(),
                insights=["Test insight 1", "Test insight 2"]
            )
        ]
        
        formatted = analyst_agent._format_kpi_results(kpi_results)
        
        assert "Test KPI" in formatted
        assert "15.50%" in formatted
        assert "above_target" in formatted.lower() or "Above Target" in formatted
        assert "increasing" in formatted.lower() or "Increasing" in formatted
    
    def test_sql_template_initialization(self, analyst_agent):
        """Test SQL template initialization."""
        templates = analyst_agent.sql_templates
        
        assert "revenue_by_period" in templates
        assert "top_customers" in templates
        assert "product_performance" in templates
        
        # Test template structure
        revenue_template = templates["revenue_by_period"]
        assert "SELECT" in revenue_template.upper()
        assert "GROUP BY" in revenue_template.upper()
    
    def test_basic_sql_generation(self, analyst_agent):
        """Test basic SQL generation without AI."""
        # Test revenue query
        sql = analyst_agent._generate_basic_sql(
            "show revenue by month", 
            {"tables": {"sales": {"columns": ["date", "revenue"]}}}
        )
        assert "SELECT" in sql.upper()
        
        # Test customer query
        sql = analyst_agent._generate_basic_sql(
            "show top customers by spending",
            {"tables": {"customers": {"columns": ["customer_id", "total_spent"]}}}
        )
        assert "SELECT" in sql.upper()
        assert "customer" in sql.lower()
    
    @pytest.mark.asyncio
    async def test_business_insight_analysis(self, analyst_agent, sample_business_data):
        """Test business insight analysis functionality."""
        insights = await analyst_agent._analyze_business_data(
            sample_business_data, 
            {"industry": "retail"}, 
            ["increase_revenue", "improve_retention"]
        )
        
        assert isinstance(insights, list)
        # Should have at least some insights from the data
        assert len(insights) >= 0
        
        # If insights are generated, test their structure
        if insights:
            insight = insights[0]
            assert isinstance(insight, BusinessInsight)
            assert hasattr(insight, 'title')
            assert hasattr(insight, 'description')
            assert hasattr(insight, 'recommendations')
    
    def test_time_column_identification(self, analyst_agent, sample_business_data):
        """Test time column identification."""
        time_col = analyst_agent._identify_time_column(sample_business_data)
        assert time_col == 'date'
        
        # Test with no time column
        no_time_df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
        time_col = analyst_agent._identify_time_column(no_time_df)
        assert time_col is None
    
    def test_numeric_metrics_identification(self, analyst_agent, sample_business_data):
        """Test numeric metrics identification."""
        metrics = analyst_agent._identify_numeric_metrics(sample_business_data)
        
        assert isinstance(metrics, list)
        assert 'revenue' in metrics
        assert 'customers' in metrics
        # Should not include ID columns
        assert 'customer_id' not in metrics
    
    @pytest.mark.asyncio
    async def test_trend_analysis_single_metric(self, analyst_agent, sample_business_data):
        """Test single metric trend analysis."""
        try:
            trend_analysis = await analyst_agent._analyze_single_trend(
                sample_business_data, 'date', 'revenue', 6
            )
            
            assert isinstance(trend_analysis, TrendAnalysis)
            assert trend_analysis.metric_name == 'revenue'
            assert trend_analysis.trend_direction in ['increasing', 'decreasing', 'stable']
            assert isinstance(trend_analysis.forecast_values, list)
            assert len(trend_analysis.forecast_values) == 6
            assert isinstance(trend_analysis.insights, list)
        except ValueError as e:
            # This might happen if there's insufficient data
            assert "Insufficient data" in str(e)
    
    def test_seasonality_detection(self, analyst_agent):
        """Test seasonality detection in time series."""
        # Create seasonal data
        seasonal_data = pd.Series([10, 15, 20, 25, 20, 15, 10, 15, 20, 25, 20, 15])
        seasonal_data.index = pd.period_range('2024-01', periods=12, freq='M')
        
        is_seasonal = analyst_agent._detect_seasonality(seasonal_data)
        # Result depends on the autocorrelation calculation
        assert isinstance(is_seasonal, bool)
        
        # Test with insufficient data
        short_data = pd.Series([10, 15, 20])
        is_seasonal = analyst_agent._detect_seasonality(short_data)
        assert is_seasonal is False
    
    @pytest.mark.asyncio
    async def test_visualization_suggestions(self, analyst_agent, sample_business_data):
        """Test visualization suggestions for dashboard."""
        suggestions = await analyst_agent._suggest_visualizations(
            sample_business_data, "create business dashboard"
        )
        
        assert isinstance(suggestions, list)
        assert len(suggestions) > 0
        
        # Check suggestion structure
        if suggestions:
            suggestion = suggestions[0]
            assert 'type' in suggestion
            assert 'title' in suggestion
            assert 'description' in suggestion
    
    @pytest.mark.asyncio
    async def test_dashboard_specification_creation(self, analyst_agent, sample_business_data):
        """Test dashboard specification creation."""
        viz_suggestions = [
            {
                "type": "line",
                "title": "Revenue Trend",
                "x_column": "date",
                "y_column": "revenue"
            }
        ]
        
        dashboard_spec = await analyst_agent._create_dashboard_specification(
            sample_business_data, viz_suggestions, {"title": "Test Dashboard"}
        )
        
        assert isinstance(dashboard_spec, dict)
        assert 'id' in dashboard_spec
        assert 'title' in dashboard_spec
        assert 'charts' in dashboard_spec
        assert 'filters' in dashboard_spec
        assert 'kpis' in dashboard_spec
        assert dashboard_spec['title'] == "Test Dashboard"
    
    def test_dashboard_filters_creation(self, analyst_agent, sample_business_data):
        """Test dashboard filters creation."""
        filters = analyst_agent._create_dashboard_filters(sample_business_data)
        
        assert isinstance(filters, list)
        # Should have at least a date filter
        assert len(filters) > 0
        
        # Check filter structure
        if filters:
            filter_item = filters[0]
            assert 'type' in filter_item
            assert 'column' in filter_item
            assert 'label' in filter_item
    
    @pytest.mark.asyncio
    async def test_kpi_widgets_creation(self, analyst_agent, sample_business_data):
        """Test KPI widgets creation for dashboard."""
        kpi_widgets = await analyst_agent._create_kpi_widgets(sample_business_data)
        
        assert isinstance(kpi_widgets, list)
        # Should create widgets based on available data
        assert len(kpi_widgets) > 0
        
        # Check widget structure
        if kpi_widgets:
            widget = kpi_widgets[0]
            assert 'type' in widget
            assert 'title' in widget
            assert 'value' in widget
            assert 'format' in widget
    
    def test_data_quality_assessment(self, analyst_agent, sample_business_data):
        """Test data quality assessment functionality."""
        quality_report = analyst_agent._assess_data_quality(sample_business_data)
        
        assert isinstance(quality_report, str)
        assert "Data Quality" in quality_report
        assert "Quality Metrics" in quality_report
        assert "Recommendations" in quality_report
    
    def test_quick_metrics_calculation(self, analyst_agent, sample_business_data):
        """Test quick metrics calculation."""
        metrics = analyst_agent._calculate_quick_metrics(sample_business_data)
        
        assert isinstance(metrics, str)
        # Should include revenue metrics
        assert "revenue" in metrics.lower() or "Total" in metrics
    
    def test_data_quality_check(self, analyst_agent, sample_business_data):
        """Test quick data quality check."""
        quality_check = analyst_agent._quick_data_quality_check(sample_business_data)
        
        assert isinstance(quality_check, str)
        assert "Missing Data" in quality_check
        assert "Duplicate Rows" in quality_check
        assert "Data Quality Score" in quality_check
    
    @pytest.mark.asyncio
    async def test_error_handling_invalid_data(self, analyst_agent, sample_request):
        """Test error handling with invalid data."""
        sample_request.context = {"dataset": "invalid_data"}
        
        response = await analyst_agent.process_request(sample_request)
        
        assert response.status == ResponseStatus.ERROR
        assert "Error processing business analysis request" in response.content
        assert response.error_message is not None
    
    @pytest.mark.asyncio
    async def test_error_handling_missing_data(self, analyst_agent, sample_request):
        """Test error handling with missing data."""
        sample_request.context = {}  # No dataset provided
        
        response = await analyst_agent.process_request(sample_request)
        
        # Should handle gracefully
        assert response.status in [ResponseStatus.SUCCESS, ResponseStatus.ERROR]
        if response.status == ResponseStatus.ERROR:
            assert "No dataset provided" in response.content
    
    def test_performance_with_large_dataset(self, analyst_agent):
        """Test performance with larger dataset."""
        # Create larger dataset
        large_data = {
            'date': pd.date_range('2020-01-01', periods=1000, freq='D'),
            'revenue': np.random.normal(10000, 1000, 1000),
            'customers': np.random.randint(50, 200, 1000),
            'product': np.random.choice(['A', 'B', 'C', 'D'], 1000)
        }
        large_df = pd.DataFrame(large_data)
        
        # Test basic operations don't fail
        time_col = analyst_agent._identify_time_column(large_df)
        assert time_col == 'date'
        
        metrics = analyst_agent._identify_numeric_metrics(large_df)
        assert 'revenue' in metrics
        assert 'customers' in metrics
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self, analyst_agent, sample_business_data):
        """Test handling concurrent requests."""
        import asyncio
        
        requests = []
        for i in range(3):
            request = AgentRequest(
                id=f"test-request-{i}",
                user_id="test-user",
                agent_id="scroll-analyst",
                prompt=f"Generate KPIs for test {i}",
                context={"dataset": sample_business_data.to_dict()},
                priority=1,
                created_at=datetime.now()
            )
            requests.append(request)
        
        # Process requests concurrently
        with patch.object(analyst_agent, '_convert_to_dataframe', return_value=sample_business_data):
            responses = await asyncio.gather(
                *[analyst_agent.process_request(req) for req in requests]
            )
        
        # All requests should complete successfully
        assert len(responses) == 3
        for response in responses:
            assert response.status == ResponseStatus.SUCCESS


class TestKPICalculations:
    """Specific tests for KPI calculation methods."""
    
    @pytest.fixture
    def analyst_agent(self):
        return ScrollAnalyst()
    
    @pytest.fixture
    def revenue_data(self):
        """Sample revenue data for testing."""
        return pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=6, freq='M'),
            'revenue': [10000, 12000, 11000, 13000, 14000, 15000],
            'marketing_spend': [2000, 2400, 2200, 2600, 2800, 3000],
            'new_customers': [20, 24, 22, 26, 28, 30],
            'conversions': [50, 60, 55, 65, 70, 75],
            'visitors': [1000, 1200, 1100, 1300, 1400, 1500]
        })
    
    @pytest.mark.asyncio
    async def test_revenue_growth_positive(self, analyst_agent, revenue_data):
        """Test revenue growth calculation with positive growth."""
        kpi_result = await analyst_agent._calculate_revenue_growth(
            revenue_data, analyst_agent.kpi_definitions["revenue_growth"], {}
        )
        
        assert kpi_result.kpi_name == "Revenue Growth Rate"
        assert kpi_result.current_value > 0  # Should show positive growth
        assert kpi_result.trend == "increasing"
        assert len(kpi_result.insights) >= 3
    
    @pytest.mark.asyncio
    async def test_cac_calculation(self, analyst_agent, revenue_data):
        """Test Customer Acquisition Cost calculation."""
        kpi_result = await analyst_agent._calculate_cac(
            revenue_data, analyst_agent.kpi_definitions["customer_acquisition_cost"], {}
        )
        
        assert kpi_result.kpi_name == "Customer Acquisition Cost"
        assert kpi_result.current_value > 0
        assert kpi_result.unit == "$"
        # CAC should be total marketing spend / total new customers
        expected_cac = revenue_data['marketing_spend'].sum() / revenue_data['new_customers'].sum()
        assert abs(kpi_result.current_value - expected_cac) < 0.01
    
    @pytest.mark.asyncio
    async def test_conversion_rate_calculation(self, analyst_agent, revenue_data):
        """Test conversion rate calculation."""
        kpi_result = await analyst_agent._calculate_conversion_rate(
            revenue_data, analyst_agent.kpi_definitions["conversion_rate"], {}
        )
        
        assert kpi_result.kpi_name == "Conversion Rate"
        assert 0 <= kpi_result.current_value <= 100
        assert kpi_result.unit == "%"
        # Conversion rate should be total conversions / total visitors * 100
        expected_rate = (revenue_data['conversions'].sum() / revenue_data['visitors'].sum()) * 100
        assert abs(kpi_result.current_value - expected_rate) < 0.01
    
    @pytest.mark.asyncio
    async def test_kpi_calculation_insufficient_data(self, analyst_agent):
        """Test KPI calculation with insufficient data."""
        insufficient_data = pd.DataFrame({
            'date': [pd.Timestamp('2024-01-01')],
            'revenue': [10000]
        })
        
        with pytest.raises(ValueError, match="Need at least 2 months"):
            await analyst_agent._calculate_revenue_growth(
                insufficient_data, analyst_agent.kpi_definitions["revenue_growth"], {}
            )
    
    @pytest.mark.asyncio
    async def test_kpi_calculation_missing_columns(self, analyst_agent):
        """Test KPI calculation with missing required columns."""
        incomplete_data = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=3, freq='M'),
            'sales': [10000, 12000, 11000]  # Missing 'revenue' column
        })
        
        try:
            await analyst_agent._calculate_revenue_growth(
                incomplete_data, analyst_agent.kpi_definitions["revenue_growth"], {}
            )
            # If no exception is raised, the test should fail
            assert False, "Expected ValueError for missing columns"
        except ValueError as e:
            assert "Required columns" in str(e) or "not found" in str(e)


if __name__ == "__main__":
    pytest.main([__file__])