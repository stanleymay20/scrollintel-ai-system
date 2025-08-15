# ScrollAnalyst Agent implementation
from scrollintel.core.interfaces import BaseAgent, AgentType, AgentRequest, AgentResponse, AgentCapability, ResponseStatus
import asyncio
import json
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple, Union
from uuid import uuid4
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

# SQL and database libraries
import sqlalchemy
from sqlalchemy import create_engine, text
import sqlite3

# Statistical libraries for KPI calculations
from scipy import stats
import math

# OpenAI integration for intelligent analysis
try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    openai = None

class AnalysisType(Enum):
    KPI_GENERATION = "kpi_generation"
    SQL_QUERY = "sql_query"
    BUSINESS_INSIGHTS = "business_insights"
    TREND_ANALYSIS = "trend_analysis"
    REPORT_GENERATION = "report_generation"
    DASHBOARD_CREATION = "dashboard_creation"

class KPICategory(Enum):
    FINANCIAL = "financial"
    OPERATIONAL = "operational"
    CUSTOMER = "customer"
    MARKETING = "marketing"
    SALES = "sales"
    GROWTH = "growth"
    EFFICIENCY = "efficiency"

@dataclass
class KPIDefinition:
    name: str
    category: KPICategory
    description: str
    formula: str
    target_value: Optional[float]
    unit: str
    frequency: str  # daily, weekly, monthly, quarterly, yearly
    data_requirements: List[str]

@dataclass
class KPIResult:
    kpi_name: str
    current_value: float
    previous_value: Optional[float]
    target_value: Optional[float]
    change_percentage: Optional[float]
    trend: str  # increasing, decreasing, stable
    status: str  # above_target, below_target, on_target
    unit: str
    calculation_date: datetime
    insights: List[str]

@dataclass
class BusinessInsight:
    title: str
    description: str
    category: str
    confidence_score: float
    supporting_data: Dict[str, Any]
    recommendations: List[str]
    priority: str  # high, medium, low

@dataclass
class TrendAnalysis:
    metric_name: str
    trend_direction: str
    trend_strength: float
    seasonal_pattern: bool
    forecast_values: List[float]
    confidence_interval: Tuple[float, float]
    analysis_period: str
    insights: List[str]

class ScrollAnalyst(BaseAgent):
    def __init__(self):
        super().__init__(
            agent_id="scroll-analyst",
            name="ScrollAnalyst Agent",
            agent_type=AgentType.ANALYST
        )
        self.capabilities = [
            AgentCapability(
                name="kpi_generation",
                description="Automated KPI calculation and monitoring with business intelligence",
                input_types=["dataset", "kpi_config", "business_metrics"],
                output_types=["kpi_dashboard", "kpi_report", "performance_metrics"]
            ),
            AgentCapability(
                name="sql_query_generation",
                description="Natural language to SQL conversion for business insights",
                input_types=["natural_language", "database_schema", "business_question"],
                output_types=["sql_query", "query_results", "data_insights"]
            ),
            AgentCapability(
                name="business_insights",
                description="Advanced business intelligence analysis and recommendations",
                input_types=["business_data", "context", "objectives"],
                output_types=["insights_report", "recommendations", "action_items"]
            ),
            AgentCapability(
                name="trend_analysis",
                description="Time series analysis and forecasting for business metrics",
                input_types=["time_series_data", "analysis_config"],
                output_types=["trend_report", "forecasts", "seasonal_analysis"]
            ),
            AgentCapability(
                name="report_generation",
                description="Automated business report generation with data summarization",
                input_types=["data_sources", "report_template", "business_context"],
                output_types=["business_report", "executive_summary", "data_visualizations"]
            ),
            AgentCapability(
                name="scrollviz_integration",
                description="Integration with ScrollViz for automatic chart and dashboard creation",
                input_types=["analysis_results", "visualization_preferences"],
                output_types=["interactive_charts", "dashboards", "visual_reports"]
            )
        ]
        
        # Initialize OpenAI client for AI-enhanced analysis
        if HAS_OPENAI:
            openai.api_key = os.getenv("OPENAI_API_KEY")
        
        # Initialize ScrollViz engine reference
        self.scrollviz_engine = None
        
        # Predefined KPI definitions for common business metrics
        self.kpi_definitions = self._initialize_kpi_definitions()
        
        # SQL query templates for common business questions
        self.sql_templates = self._initialize_sql_templates()
        
    async def process_request(self, request: AgentRequest) -> AgentResponse:
        start_time = asyncio.get_event_loop().time()
        try:
            # Parse the request to determine analysis type
            prompt = request.prompt.lower()
            context = request.context or {}
            
            if "sql" in prompt or "query" in prompt:
                content = await self._generate_sql_query(request.prompt, context)
            elif "trend" in prompt or "forecast" in prompt or "prediction" in prompt:
                content = await self._perform_trend_analysis(request.prompt, context)
            elif "report" in prompt or "summary" in prompt:
                content = await self._generate_business_report(request.prompt, context)
            elif "dashboard" in prompt or "visualization" in prompt or "chart" in prompt:
                content = await self._create_dashboard_with_scrollviz(request.prompt, context)
            elif "insight" in prompt or "intelligence" in prompt:
                content = await self._generate_business_insights(request.prompt, context)
            elif "kpi" in prompt or "metric" in prompt or "performance" in prompt:
                content = await self._generate_kpis(request.prompt, context)
            else:
                content = await self._general_business_analysis(request.prompt, context)
            
            execution_time = asyncio.get_event_loop().time() - start_time
            return AgentResponse(
                id=f"analyst-{uuid4()}",
                request_id=request.id,
                content=content,
                artifacts=[],
                execution_time=execution_time,
                status=ResponseStatus.SUCCESS
            )
        except Exception as e:
            execution_time = asyncio.get_event_loop().time() - start_time
            return AgentResponse(
                id=f"analyst-{uuid4()}",
                request_id=request.id,
                content=f"Error processing business analysis request: {str(e)}",
                artifacts=[],
                execution_time=execution_time,
                status=ResponseStatus.ERROR,
                error_message=str(e)
            )
    
    def get_capabilities(self) -> List[AgentCapability]:
        """Return the capabilities of this agent."""
        return self.capabilities
    
    async def health_check(self) -> bool:
        """Check if the agent is healthy and ready to process requests."""
        try:
            # Test basic functionality
            test_data = pd.DataFrame({
                'revenue': [100, 120, 110, 130],
                'date': pd.date_range('2024-01-01', periods=4, freq='M')
            })
            
            # Test KPI calculation
            kpi_result = await self._calculate_single_kpi(test_data, "revenue_growth")
            
            return kpi_result is not None
        except Exception:
            return False
    
    async def _generate_kpis(self, prompt: str, context: Dict[str, Any]) -> str:
        """Generate KPIs and performance metrics from business data"""
        dataset_path = context.get("dataset_path")
        dataset = context.get("dataset")
        kpi_config = context.get("kpi_config", {})
        
        if dataset_path:
            df = await self._load_dataset(dataset_path)
        elif dataset is not None:
            df = self._convert_to_dataframe(dataset)
        else:
            return "Error: No dataset provided. Please provide dataset_path or dataset in context."
        
        # Determine which KPIs to calculate
        requested_kpis = kpi_config.get("kpis", [])
        if not requested_kpis:
            requested_kpis = await self._suggest_kpis_from_data(df, prompt)
        
        # Calculate KPIs
        kpi_results = []
        for kpi_name in requested_kpis:
            try:
                kpi_result = await self._calculate_single_kpi(df, kpi_name, kpi_config)
                if kpi_result:
                    kpi_results.append(kpi_result)
            except Exception as e:
                print(f"Error calculating KPI {kpi_name}: {str(e)}")
        
        # Generate AI-enhanced insights
        ai_insights = await self._get_ai_kpi_insights(df, kpi_results)
        
        # Format comprehensive KPI report
        report = f"""
# Business KPI Analysis Report

## Executive Summary
Generated {len(kpi_results)} key performance indicators from your business data.

## KPI Dashboard

{self._format_kpi_results(kpi_results)}

## Performance Analysis

### Top Performing Metrics
{self._identify_top_performing_kpis(kpi_results)}

### Areas of Concern
{self._identify_concerning_kpis(kpi_results)}

## AI-Enhanced Business Insights
{ai_insights}

## Recommendations
{self._generate_kpi_recommendations(kpi_results)}

## Next Steps
1. Set up automated KPI monitoring
2. Create alerts for metrics below target
3. Integrate with ScrollViz for real-time dashboards
4. Schedule regular KPI reviews with stakeholders

---
*KPI analysis completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        return report
    
    async def _generate_sql_query(self, prompt: str, context: Dict[str, Any]) -> str:
        """Generate SQL queries from natural language business questions"""
        database_schema = context.get("database_schema", {})
        connection_string = context.get("connection_string")
        business_question = prompt
        
        try:
            # Generate SQL query using AI
            sql_query = await self._natural_language_to_sql(business_question, database_schema)
            
            # Execute query if connection provided
            query_results = None
            if connection_string and sql_query:
                try:
                    query_results = await self._execute_sql_query(sql_query, connection_string)
                except Exception as e:
                    query_results = f"Error executing query: {str(e)}"
            
            # Generate insights from results
            insights = await self._analyze_sql_results(sql_query, query_results, business_question)
        except Exception as e:
            sql_query = "-- Error generating SQL query"
            query_results = None
            insights = f"Error: {str(e)}"
        
        report = f"""
# SQL Query Generation Report

## Business Question
{business_question}

## Generated SQL Query
```sql
{sql_query}
```

## Query Explanation
{await self._explain_sql_query(sql_query)}

## Query Results
{self._format_query_results(query_results)}

## Business Insights
{insights}

## Query Optimization Suggestions
{await self._suggest_query_optimizations(sql_query)}

## Related Queries
{await self._suggest_related_queries(business_question, database_schema)}

---
*SQL analysis completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        return report
    
    async def _generate_business_insights(self, prompt: str, context: Dict[str, Any]) -> str:
        """Generate comprehensive business intelligence insights"""
        dataset_path = context.get("dataset_path")
        dataset = context.get("dataset")
        business_context = context.get("business_context", {})
        objectives = context.get("objectives", [])
        
        try:
            if dataset_path:
                df = await self._load_dataset(dataset_path)
            elif dataset is not None:
                df = self._convert_to_dataframe(dataset)
            else:
                return "Error: No dataset provided for business analysis."
            
            # Perform comprehensive business analysis
            insights = await self._analyze_business_data(df, business_context, objectives)
            
            # Generate AI-powered recommendations
            ai_recommendations = await self._get_ai_business_recommendations(df, insights, prompt)
        except Exception as e:
            return f"Error generating business insights: {str(e)}"
        
        report = f"""
# Business Intelligence Analysis Report

## Data Overview
- **Dataset Size**: {df.shape[0]:,} records, {df.shape[1]} variables
- **Analysis Period**: {self._determine_analysis_period(df)}
- **Business Context**: {business_context.get('industry', 'General')}

## Key Business Insights

{self._format_business_insights(insights)}

## Performance Metrics Summary
{self._calculate_business_performance_summary(df)}

## Market Analysis
{await self._perform_market_analysis(df, business_context)}

## AI-Powered Strategic Recommendations
{ai_recommendations}

## Risk Assessment
{await self._assess_business_risks(df, insights)}

## Action Plan
{self._generate_action_plan(insights)}

## ROI Projections
{await self._calculate_roi_projections(df, insights)}

---
*Business intelligence analysis completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        return report
    
    async def _perform_trend_analysis(self, prompt: str, context: Dict[str, Any]) -> str:
        """Perform time series trend analysis and forecasting"""
        dataset_path = context.get("dataset_path")
        dataset = context.get("dataset")
        time_column = context.get("time_column")
        metrics = context.get("metrics", [])
        forecast_periods = context.get("forecast_periods", 12)
        
        if dataset_path:
            df = await self._load_dataset(dataset_path)
        elif dataset is not None:
            df = self._convert_to_dataframe(dataset)
        else:
            return "Error: No dataset provided for trend analysis."
        
        # Identify time column if not specified
        if not time_column:
            time_column = self._identify_time_column(df)
        
        # Identify metrics to analyze if not specified
        if not metrics:
            metrics = self._identify_numeric_metrics(df)
        
        # Perform trend analysis for each metric
        trend_analyses = []
        for metric in metrics:
            try:
                trend_analysis = await self._analyze_single_trend(df, time_column, metric, forecast_periods)
                trend_analyses.append(trend_analysis)
            except Exception as e:
                print(f"Error analyzing trend for {metric}: {str(e)}")
        
        # Generate AI insights on trends
        ai_trend_insights = await self._get_ai_trend_insights(trend_analyses)
        
        report = f"""
# Business Trend Analysis Report

## Analysis Overview
- **Time Period**: {self._get_analysis_time_range(df, time_column)}
- **Metrics Analyzed**: {len(trend_analyses)}
- **Forecast Horizon**: {forecast_periods} periods

## Trend Analysis Results

{self._format_trend_analyses(trend_analyses)}

## Seasonal Patterns
{self._analyze_seasonal_patterns(trend_analyses)}

## Forecasting Results
{self._format_forecasting_results(trend_analyses)}

## AI-Enhanced Trend Insights
{ai_trend_insights}

## Business Implications
{await self._interpret_trend_business_impact(trend_analyses)}

## Recommendations
{self._generate_trend_recommendations(trend_analyses)}

## Risk Factors
{self._identify_trend_risks(trend_analyses)}

---
*Trend analysis completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        return report
    
    async def _generate_business_report(self, prompt: str, context: Dict[str, Any]) -> str:
        """Generate comprehensive business reports with data summarization"""
        dataset_path = context.get("dataset_path")
        dataset = context.get("dataset")
        report_template = context.get("report_template", "standard")
        business_context = context.get("business_context", {})
        
        if dataset_path:
            df = await self._load_dataset(dataset_path)
        elif dataset is not None:
            df = self._convert_to_dataframe(dataset)
        else:
            return "Error: No dataset provided for report generation."
        
        # Generate different sections of the business report
        executive_summary = await self._generate_executive_summary(df, business_context)
        financial_analysis = await self._generate_financial_analysis(df)
        operational_metrics = await self._generate_operational_metrics(df)
        recommendations = await self._generate_strategic_recommendations(df, business_context)
        
        report = f"""
# Business Performance Report

## Executive Summary
{executive_summary}

## Financial Analysis
{financial_analysis}

## Operational Metrics
{operational_metrics}

## Data Quality Assessment
{self._assess_data_quality(df)}

## Key Performance Indicators
{await self._generate_report_kpis(df)}

## Market Position Analysis
{await self._analyze_market_position(df, business_context)}

## Strategic Recommendations
{recommendations}

## Risk Assessment
{await self._generate_risk_assessment(df)}

## Implementation Roadmap
{self._create_implementation_roadmap(recommendations)}

## Appendix: Data Sources and Methodology
{self._document_methodology(df)}

---
*Business report generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        return report
    
    async def _create_dashboard_with_scrollviz(self, prompt: str, context: Dict[str, Any]) -> str:
        """Create interactive dashboards using ScrollViz integration"""
        dataset_path = context.get("dataset_path")
        dataset = context.get("dataset")
        dashboard_config = context.get("dashboard_config", {})
        
        try:
            if dataset_path:
                df = await self._load_dataset(dataset_path)
            elif dataset is not None:
                df = self._convert_to_dataframe(dataset)
            else:
                return "Error: No dataset provided for dashboard creation."
            
            # Analyze data to suggest appropriate visualizations
            viz_suggestions = await self._suggest_visualizations(df, prompt)
            
            # Create dashboard configuration for ScrollViz
            dashboard_spec = await self._create_dashboard_specification(df, viz_suggestions, dashboard_config)
            
            # Generate ScrollViz integration code
            scrollviz_config = await self._generate_scrollviz_config(dashboard_spec)
        except Exception as e:
            return f"Error creating dashboard: {str(e)}"
        
        report = f"""
# ScrollViz Dashboard Creation Report

## Dashboard Overview
- **Data Source**: {len(df)} records across {len(df.columns)} dimensions
- **Visualizations**: {len(viz_suggestions)} recommended charts
- **Dashboard Type**: {dashboard_config.get('type', 'Business Intelligence')}

## Recommended Visualizations
{self._format_visualization_suggestions(viz_suggestions)}

## Dashboard Configuration
```json
{json.dumps(dashboard_spec, indent=2)}
```

## ScrollViz Integration
{scrollviz_config}

## Interactive Features
{self._describe_interactive_features(dashboard_spec)}

## Real-time Data Binding
{self._setup_realtime_binding(dashboard_spec)}

## Dashboard Deployment
{self._generate_deployment_instructions(dashboard_spec)}

## Performance Optimization
{self._suggest_performance_optimizations(df, dashboard_spec)}

## Next Steps
1. Review and customize dashboard configuration
2. Deploy dashboard using ScrollViz engine
3. Set up data refresh schedules
4. Configure user access and permissions
5. Monitor dashboard performance and usage

---
*Dashboard specification completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        return report
    
    async def _general_business_analysis(self, prompt: str, context: Dict[str, Any]) -> str:
        """General business analysis using AI"""
        dataset_path = context.get("dataset_path")
        dataset = context.get("dataset")
        
        if dataset_path:
            df = await self._load_dataset(dataset_path)
        elif dataset is not None:
            df = self._convert_to_dataframe(dataset)
        else:
            return "Error: No dataset provided for analysis."
        
        # Get AI-powered analysis
        ai_analysis = await self._get_ai_business_analysis(df, prompt)
        
        # Basic business overview
        overview = f"""
# Business Analysis Report

## Dataset Overview
- **Records**: {df.shape[0]:,}
- **Variables**: {df.shape[1]}
- **Data Types**: {df.dtypes.value_counts().to_dict()}

## User Request
{prompt}

## AI-Powered Business Analysis
{ai_analysis}

## Quick Business Metrics
{self._calculate_quick_metrics(df)}

## Data Quality Summary
{self._quick_data_quality_check(df)}

---
*Analysis completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        return overview
    
    # Helper methods for data processing and analysis
    
    async def _load_dataset(self, dataset_path: str) -> pd.DataFrame:
        """Load dataset from various file formats"""
        try:
            if dataset_path.endswith('.csv'):
                return pd.read_csv(dataset_path)
            elif dataset_path.endswith('.xlsx') or dataset_path.endswith('.xls'):
                return pd.read_excel(dataset_path)
            elif dataset_path.endswith('.json'):
                return pd.read_json(dataset_path)
            elif dataset_path.endswith('.parquet'):
                return pd.read_parquet(dataset_path)
            else:
                raise ValueError(f"Unsupported file format: {dataset_path}")
        except Exception as e:
            raise Exception(f"Error loading dataset: {str(e)}")
    
    def _convert_to_dataframe(self, dataset: Any) -> pd.DataFrame:
        """Convert various data formats to DataFrame"""
        if isinstance(dataset, dict):
            return pd.DataFrame(dataset)
        elif isinstance(dataset, list):
            return pd.DataFrame(dataset)
        elif isinstance(dataset, pd.DataFrame):
            return dataset
        else:
            raise ValueError("Unsupported dataset format. Please provide a dictionary, list, or DataFrame.")
    
    def _initialize_kpi_definitions(self) -> Dict[str, KPIDefinition]:
        """Initialize predefined KPI definitions"""
        return {
            "revenue_growth": KPIDefinition(
                name="Revenue Growth Rate",
                category=KPICategory.FINANCIAL,
                description="Percentage increase in revenue over time",
                formula="((Current Revenue - Previous Revenue) / Previous Revenue) * 100",
                target_value=10.0,
                unit="%",
                frequency="monthly",
                data_requirements=["revenue", "date"]
            ),
            "customer_acquisition_cost": KPIDefinition(
                name="Customer Acquisition Cost",
                category=KPICategory.MARKETING,
                description="Cost to acquire a new customer",
                formula="Total Marketing Spend / Number of New Customers",
                target_value=100.0,
                unit="$",
                frequency="monthly",
                data_requirements=["marketing_spend", "new_customers"]
            ),
            "customer_lifetime_value": KPIDefinition(
                name="Customer Lifetime Value",
                category=KPICategory.CUSTOMER,
                description="Total value a customer brings over their lifetime",
                formula="Average Order Value * Purchase Frequency * Customer Lifespan",
                target_value=500.0,
                unit="$",
                frequency="quarterly",
                data_requirements=["order_value", "purchase_frequency", "customer_lifespan"]
            ),
            "conversion_rate": KPIDefinition(
                name="Conversion Rate",
                category=KPICategory.MARKETING,
                description="Percentage of visitors who complete desired action",
                formula="(Conversions / Total Visitors) * 100",
                target_value=5.0,
                unit="%",
                frequency="weekly",
                data_requirements=["conversions", "visitors"]
            ),
            "churn_rate": KPIDefinition(
                name="Customer Churn Rate",
                category=KPICategory.CUSTOMER,
                description="Percentage of customers who stop using service",
                formula="(Customers Lost / Total Customers at Start) * 100",
                target_value=5.0,
                unit="%",
                frequency="monthly",
                data_requirements=["customers_lost", "total_customers"]
            )
        }
    
    def _initialize_sql_templates(self) -> Dict[str, str]:
        """Initialize SQL query templates for common business questions"""
        return {
            "revenue_by_period": """
                SELECT DATE_TRUNC('{period}', date_column) as period,
                       SUM(revenue_column) as total_revenue
                FROM {table_name}
                WHERE date_column >= '{start_date}'
                GROUP BY DATE_TRUNC('{period}', date_column)
                ORDER BY period;
            """,
            "top_customers": """
                SELECT customer_id, customer_name,
                       SUM(order_value) as total_spent,
                       COUNT(*) as order_count
                FROM {table_name}
                GROUP BY customer_id, customer_name
                ORDER BY total_spent DESC
                LIMIT {limit};
            """,
            "product_performance": """
                SELECT product_id, product_name,
                       SUM(quantity) as units_sold,
                       SUM(revenue) as total_revenue,
                       AVG(rating) as avg_rating
                FROM {table_name}
                GROUP BY product_id, product_name
                ORDER BY total_revenue DESC;
            """
        }   
 
    async def _suggest_kpis_from_data(self, df: pd.DataFrame, prompt: str) -> List[str]:
        """Suggest relevant KPIs based on data structure and user prompt"""
        suggested_kpis = []
        columns = [col.lower() for col in df.columns]
        
        # Check for revenue-related KPIs
        if any(col in columns for col in ['revenue', 'sales', 'income']):
            suggested_kpis.append("revenue_growth")
        
        # Check for customer-related KPIs
        if any(col in columns for col in ['customer', 'user', 'client']):
            if any(col in columns for col in ['acquisition', 'new', 'signup']):
                suggested_kpis.append("customer_acquisition_cost")
            if any(col in columns for col in ['churn', 'cancel', 'unsubscribe']):
                suggested_kpis.append("churn_rate")
        
        # Check for conversion-related KPIs
        if any(col in columns for col in ['conversion', 'click', 'visit']):
            suggested_kpis.append("conversion_rate")
        
        # Default to revenue growth if no specific patterns found
        if not suggested_kpis and any(col in columns for col in ['value', 'amount', 'total']):
            suggested_kpis.append("revenue_growth")
        
        return suggested_kpis[:5]  # Limit to 5 KPIs
    
    async def _calculate_single_kpi(self, df: pd.DataFrame, kpi_name: str, config: Dict[str, Any] = None) -> Optional[KPIResult]:
        """Calculate a single KPI from the dataset"""
        if kpi_name not in self.kpi_definitions:
            return None
        
        kpi_def = self.kpi_definitions[kpi_name]
        config = config or {}
        
        try:
            if kpi_name == "revenue_growth":
                return await self._calculate_revenue_growth(df, kpi_def, config)
            elif kpi_name == "customer_acquisition_cost":
                return await self._calculate_cac(df, kpi_def, config)
            elif kpi_name == "customer_lifetime_value":
                return await self._calculate_clv(df, kpi_def, config)
            elif kpi_name == "conversion_rate":
                return await self._calculate_conversion_rate(df, kpi_def, config)
            elif kpi_name == "churn_rate":
                return await self._calculate_churn_rate(df, kpi_def, config)
            else:
                return None
        except Exception as e:
            print(f"Error calculating {kpi_name}: {str(e)}")
            return None
    
    async def _calculate_revenue_growth(self, df: pd.DataFrame, kpi_def: KPIDefinition, config: Dict[str, Any]) -> KPIResult:
        """Calculate revenue growth rate"""
        # Find revenue and date columns
        revenue_col = self._find_column(df, ['revenue', 'sales', 'income', 'total'])
        date_col = self._find_column(df, ['date', 'time', 'created_at', 'timestamp'])
        
        if not revenue_col:
            raise ValueError("Required columns for revenue growth not found: revenue column missing")
        if not date_col:
            raise ValueError("Required columns for revenue growth not found: date column missing")
        
        # Convert date column to datetime
        df_copy = df.copy()
        df_copy[date_col] = pd.to_datetime(df_copy[date_col])
        
        # Group by month and sum revenue
        monthly_revenue = df_copy.groupby(df_copy[date_col].dt.to_period('M'))[revenue_col].sum()
        
        if len(monthly_revenue) < 2:
            raise ValueError("Need at least 2 months of data for growth calculation")
        
        current_value = monthly_revenue.iloc[-1]
        previous_value = monthly_revenue.iloc[-2]
        
        change_percentage = ((current_value - previous_value) / previous_value) * 100
        
        # Determine trend and status
        trend = "increasing" if change_percentage > 0 else "decreasing" if change_percentage < 0 else "stable"
        status = "above_target" if change_percentage >= kpi_def.target_value else "below_target"
        
        return KPIResult(
            kpi_name=kpi_def.name,
            current_value=change_percentage,
            previous_value=0.0,  # Growth rate comparison
            target_value=kpi_def.target_value,
            change_percentage=change_percentage,
            trend=trend,
            status=status,
            unit=kpi_def.unit,
            calculation_date=datetime.now(),
            insights=[
                f"Revenue {'increased' if change_percentage > 0 else 'decreased'} by {abs(change_percentage):.1f}% this month",
                f"Current monthly revenue: ${current_value:,.2f}",
                f"Previous monthly revenue: ${previous_value:,.2f}"
            ]
        )
    
    async def _calculate_cac(self, df: pd.DataFrame, kpi_def: KPIDefinition, config: Dict[str, Any]) -> KPIResult:
        """Calculate Customer Acquisition Cost"""
        marketing_col = self._find_column(df, ['marketing_spend', 'ad_spend', 'marketing_cost'])
        customers_col = self._find_column(df, ['new_customers', 'acquisitions', 'signups'])
        
        if not marketing_col or not customers_col:
            raise ValueError("Required columns for CAC calculation not found")
        
        total_marketing_spend = df[marketing_col].sum()
        total_new_customers = df[customers_col].sum()
        
        if total_new_customers == 0:
            raise ValueError("No new customers found for CAC calculation")
        
        current_value = total_marketing_spend / total_new_customers
        
        status = "below_target" if current_value <= kpi_def.target_value else "above_target"
        
        return KPIResult(
            kpi_name=kpi_def.name,
            current_value=current_value,
            previous_value=None,
            target_value=kpi_def.target_value,
            change_percentage=None,
            trend="stable",
            status=status,
            unit=kpi_def.unit,
            calculation_date=datetime.now(),
            insights=[
                f"Total marketing spend: ${total_marketing_spend:,.2f}",
                f"New customers acquired: {total_new_customers:,}",
                f"Cost per acquisition: ${current_value:.2f}"
            ]
        )
    
    async def _calculate_conversion_rate(self, df: pd.DataFrame, kpi_def: KPIDefinition, config: Dict[str, Any]) -> KPIResult:
        """Calculate conversion rate"""
        conversions_col = self._find_column(df, ['conversions', 'purchases', 'sales'])
        visitors_col = self._find_column(df, ['visitors', 'traffic', 'sessions', 'users'])
        
        if not conversions_col or not visitors_col:
            raise ValueError("Required columns for conversion rate calculation not found")
        
        total_conversions = df[conversions_col].sum()
        total_visitors = df[visitors_col].sum()
        
        if total_visitors == 0:
            raise ValueError("No visitors found for conversion rate calculation")
        
        current_value = (total_conversions / total_visitors) * 100
        
        status = "above_target" if current_value >= kpi_def.target_value else "below_target"
        
        return KPIResult(
            kpi_name=kpi_def.name,
            current_value=current_value,
            previous_value=None,
            target_value=kpi_def.target_value,
            change_percentage=None,
            trend="stable",
            status=status,
            unit=kpi_def.unit,
            calculation_date=datetime.now(),
            insights=[
                f"Total conversions: {total_conversions:,}",
                f"Total visitors: {total_visitors:,}",
                f"Conversion rate: {current_value:.2f}%"
            ]
        )
    
    async def _calculate_clv(self, df: pd.DataFrame, kpi_def: KPIDefinition, config: Dict[str, Any]) -> KPIResult:
        """Calculate Customer Lifetime Value"""
        # This is a simplified CLV calculation
        order_value_col = self._find_column(df, ['order_value', 'purchase_amount', 'revenue'])
        customer_col = self._find_column(df, ['customer_id', 'user_id', 'client_id'])
        
        if not order_value_col or not customer_col:
            raise ValueError("Required columns for CLV calculation not found")
        
        # Calculate average order value and purchase frequency
        avg_order_value = df[order_value_col].mean()
        customer_orders = df.groupby(customer_col).size()
        avg_purchase_frequency = customer_orders.mean()
        
        # Assume average customer lifespan of 2 years (24 months)
        avg_customer_lifespan = 24
        
        current_value = avg_order_value * avg_purchase_frequency * avg_customer_lifespan
        
        status = "above_target" if current_value >= kpi_def.target_value else "below_target"
        
        return KPIResult(
            kpi_name=kpi_def.name,
            current_value=current_value,
            previous_value=None,
            target_value=kpi_def.target_value,
            change_percentage=None,
            trend="stable",
            status=status,
            unit=kpi_def.unit,
            calculation_date=datetime.now(),
            insights=[
                f"Average order value: ${avg_order_value:.2f}",
                f"Average purchase frequency: {avg_purchase_frequency:.1f} orders per customer",
                f"Estimated customer lifetime value: ${current_value:.2f}"
            ]
        )
    
    async def _calculate_churn_rate(self, df: pd.DataFrame, kpi_def: KPIDefinition, config: Dict[str, Any]) -> KPIResult:
        """Calculate customer churn rate"""
        # This requires time-series customer data
        customer_col = self._find_column(df, ['customer_id', 'user_id', 'client_id'])
        date_col = self._find_column(df, ['date', 'time', 'created_at', 'timestamp'])
        
        if not customer_col or not date_col:
            raise ValueError("Required columns for churn rate calculation not found")
        
        # Convert date column to datetime
        df[date_col] = pd.to_datetime(df[date_col])
        
        # Get customers from last two months
        df_sorted = df.sort_values(date_col)
        latest_date = df_sorted[date_col].max()
        one_month_ago = latest_date - timedelta(days=30)
        two_months_ago = latest_date - timedelta(days=60)
        
        customers_two_months_ago = set(df[df[date_col] >= two_months_ago][customer_col].unique())
        customers_one_month_ago = set(df[df[date_col] >= one_month_ago][customer_col].unique())
        
        churned_customers = customers_two_months_ago - customers_one_month_ago
        
        if len(customers_two_months_ago) == 0:
            raise ValueError("No customers found for churn calculation")
        
        current_value = (len(churned_customers) / len(customers_two_months_ago)) * 100
        
        status = "below_target" if current_value <= kpi_def.target_value else "above_target"
        
        return KPIResult(
            kpi_name=kpi_def.name,
            current_value=current_value,
            previous_value=None,
            target_value=kpi_def.target_value,
            change_percentage=None,
            trend="stable",
            status=status,
            unit=kpi_def.unit,
            calculation_date=datetime.now(),
            insights=[
                f"Customers two months ago: {len(customers_two_months_ago):,}",
                f"Churned customers: {len(churned_customers):,}",
                f"Churn rate: {current_value:.2f}%"
            ]
        )
    
    def _find_column(self, df: pd.DataFrame, possible_names: List[str]) -> Optional[str]:
        """Find a column that matches one of the possible names"""
        df_columns_lower = [col.lower() for col in df.columns]
        for name in possible_names:
            if name.lower() in df_columns_lower:
                return df.columns[df_columns_lower.index(name.lower())]
        return None
    
    def _format_kpi_results(self, kpi_results: List[KPIResult]) -> str:
        """Format KPI results for display"""
        if not kpi_results:
            return "No KPIs calculated."
        
        formatted = []
        for kpi in kpi_results:
            status_emoji = "✅" if kpi.status == "above_target" else "⚠️" if kpi.status == "on_target" else "❌"
            trend_emoji = "📈" if kpi.trend == "increasing" else "📉" if kpi.trend == "decreasing" else "➡️"
            
            formatted.append(f"""
### {kpi.kpi_name} {status_emoji}
- **Current Value**: {kpi.current_value:.2f}{kpi.unit}
- **Target**: {kpi.target_value}{kpi.unit} 
- **Trend**: {kpi.trend.title()} {trend_emoji}
- **Status**: {kpi.status.replace('_', ' ').title()}
- **Insights**: {'; '.join(kpi.insights)}
""")
        
        return '\n'.join(formatted)
    
    def _identify_top_performing_kpis(self, kpi_results: List[KPIResult]) -> str:
        """Identify top performing KPIs"""
        above_target = [kpi for kpi in kpi_results if kpi.status == "above_target"]
        
        if not above_target:
            return "No KPIs are currently above target."
        
        formatted = []
        for kpi in above_target:
            formatted.append(f"- **{kpi.kpi_name}**: {kpi.current_value:.2f}{kpi.unit} (Target: {kpi.target_value}{kpi.unit})")
        
        return '\n'.join(formatted)
    
    def _identify_concerning_kpis(self, kpi_results: List[KPIResult]) -> str:
        """Identify KPIs that need attention"""
        below_target = [kpi for kpi in kpi_results if kpi.status == "below_target"]
        
        if not below_target:
            return "All KPIs are meeting or exceeding targets."
        
        formatted = []
        for kpi in below_target:
            gap = abs(kpi.current_value - kpi.target_value) if kpi.target_value else 0
            formatted.append(f"- **{kpi.kpi_name}**: {kpi.current_value:.2f}{kpi.unit} (Gap: {gap:.2f}{kpi.unit})")
        
        return '\n'.join(formatted)
    
    def _generate_kpi_recommendations(self, kpi_results: List[KPIResult]) -> str:
        """Generate recommendations based on KPI performance"""
        recommendations = []
        
        for kpi in kpi_results:
            if kpi.status == "below_target":
                if "revenue" in kpi.kpi_name.lower():
                    recommendations.append("- Focus on increasing sales through marketing campaigns or product improvements")
                elif "acquisition" in kpi.kpi_name.lower():
                    recommendations.append("- Optimize marketing channels and reduce acquisition costs")
                elif "conversion" in kpi.kpi_name.lower():
                    recommendations.append("- Improve website UX and optimize conversion funnel")
                elif "churn" in kpi.kpi_name.lower():
                    recommendations.append("- Implement customer retention programs and improve customer satisfaction")
        
        if not recommendations:
            recommendations.append("- Continue monitoring KPIs and maintain current performance levels")
            recommendations.append("- Consider setting more ambitious targets for continued growth")
        
        return '\n'.join(recommendations)
    
    async def _natural_language_to_sql(self, question: str, schema: Dict[str, Any]) -> str:
        """Convert natural language question to SQL query using AI"""
        if not HAS_OPENAI or not openai.api_key:
            return self._generate_basic_sql(question, schema)
        
        try:
            # Create a prompt for SQL generation
            schema_info = json.dumps(schema, indent=2) if schema else "No schema provided"
            
            prompt = f"""
Convert the following business question to a SQL query:

Question: {question}

Database Schema:
{schema_info}

Generate a SQL query that answers this question. Return only the SQL query without explanation.
"""
            
            response = await openai.ChatCompletion.acreate(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.1
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Error generating SQL with AI: {str(e)}")
            return self._generate_basic_sql(question, schema)
    
    def _generate_basic_sql(self, question: str, schema: Dict[str, Any]) -> str:
        """Generate basic SQL query without AI"""
        question_lower = question.lower()
        
        # Simple pattern matching for common queries
        if "revenue" in question_lower and "month" in question_lower:
            return """
SELECT DATE_TRUNC('month', date_column) as month,
       SUM(revenue_column) as total_revenue
FROM your_table
GROUP BY DATE_TRUNC('month', date_column)
ORDER BY month;
"""
        elif "top" in question_lower and "customer" in question_lower:
            return """
SELECT customer_id, customer_name,
       SUM(order_value) as total_spent
FROM your_table
GROUP BY customer_id, customer_name
ORDER BY total_spent DESC
LIMIT 10;
"""
        else:
            return f"-- SQL query for: {question}\nSELECT * FROM your_table LIMIT 100;"
    
    async def _execute_sql_query(self, query: str, connection_string: str) -> Any:
        """Execute SQL query against database"""
        try:
            engine = create_engine(connection_string)
            result = pd.read_sql(query, engine)
            return result
        except Exception as e:
            raise Exception(f"Error executing SQL query: {str(e)}")
    
    async def _explain_sql_query(self, query: str) -> str:
        """Explain what the SQL query does"""
        explanations = []
        
        query_lower = query.lower()
        
        if "select" in query_lower:
            explanations.append("This query retrieves data from the database")
        if "group by" in query_lower:
            explanations.append("Results are grouped by specific columns for aggregation")
        if "order by" in query_lower:
            explanations.append("Results are sorted in a specific order")
        if "sum(" in query_lower:
            explanations.append("Calculates sum totals for numeric columns")
        if "count(" in query_lower:
            explanations.append("Counts the number of records")
        if "where" in query_lower:
            explanations.append("Filters data based on specific conditions")
        
        return ". ".join(explanations) + "." if explanations else "This query performs data retrieval operations."
    
    def _format_query_results(self, results: Any) -> str:
        """Format SQL query results for display"""
        if results is None:
            return "No query executed."
        
        if isinstance(results, str):
            return results  # Error message
        
        if isinstance(results, pd.DataFrame):
            if len(results) == 0:
                return "Query returned no results."
            
            # Show first 10 rows
            display_df = results.head(10)
            return f"""
**Query Results** ({len(results)} total rows):

{display_df.to_string(index=False)}

{f"... and {len(results) - 10} more rows" if len(results) > 10 else ""}
"""
        
        return str(results)
    
    async def _analyze_sql_results(self, query: str, results: Any, question: str) -> str:
        """Analyze SQL query results and provide insights"""
        if not isinstance(results, pd.DataFrame) or len(results) == 0:
            return "No data available for analysis."
        
        insights = []
        
        # Basic statistics
        if len(results) > 0:
            insights.append(f"Query returned {len(results)} records")
        
        # Analyze numeric columns
        numeric_cols = results.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if not results[col].empty:
                total = results[col].sum()
                avg = results[col].mean()
                insights.append(f"{col}: Total = {total:,.2f}, Average = {avg:.2f}")
        
        # Look for trends in date-based data
        date_cols = results.select_dtypes(include=['datetime64']).columns
        if len(date_cols) > 0 and len(numeric_cols) > 0:
            insights.append("Time-based data detected - consider trend analysis")
        
        return '\n'.join([f"- {insight}" for insight in insights])
    
    async def _suggest_query_optimizations(self, query: str) -> str:
        """Suggest optimizations for SQL query"""
        suggestions = []
        
        query_lower = query.lower()
        
        if "select *" in query_lower:
            suggestions.append("Consider selecting only needed columns instead of SELECT *")
        
        if "where" not in query_lower and "limit" not in query_lower:
            suggestions.append("Add WHERE clause or LIMIT to reduce data volume")
        
        if "group by" in query_lower and "having" not in query_lower:
            suggestions.append("Consider using HAVING clause to filter grouped results")
        
        if not suggestions:
            suggestions.append("Query appears to be well-structured")
        
        return '\n'.join([f"- {suggestion}" for suggestion in suggestions])
    
    async def _suggest_related_queries(self, question: str, schema: Dict[str, Any]) -> str:
        """Suggest related queries that might be useful"""
        related = []
        
        question_lower = question.lower()
        
        if "revenue" in question_lower:
            related.append("Revenue by product category")
            related.append("Revenue growth rate over time")
            related.append("Top revenue-generating customers")
        
        if "customer" in question_lower:
            related.append("Customer acquisition trends")
            related.append("Customer lifetime value analysis")
            related.append("Customer churn analysis")
        
        if not related:
            related.append("Data quality assessment")
            related.append("Summary statistics by category")
            related.append("Time-based trend analysis")
        
        return '\n'.join([f"- {query}" for query in related])
    
    async def _get_ai_kpi_insights(self, df: pd.DataFrame, kpi_results: List[KPIResult]) -> str:
        """Get AI-enhanced insights on KPI performance"""
        if not HAS_OPENAI or not openai.api_key:
            return self._generate_basic_kpi_insights(kpi_results)
        
        try:
            kpi_summary = []
            for kpi in kpi_results:
                kpi_summary.append(f"{kpi.kpi_name}: {kpi.current_value:.2f}{kpi.unit} ({kpi.status})")
            
            prompt = f"""
Analyze these business KPIs and provide strategic insights:

{chr(10).join(kpi_summary)}

Provide 3-4 key insights about business performance and strategic recommendations.
"""
            
            response = await openai.ChatCompletion.acreate(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Error getting AI insights: {str(e)}")
            return self._generate_basic_kpi_insights(kpi_results)
    
    def _generate_basic_kpi_insights(self, kpi_results: List[KPIResult]) -> str:
        """Generate basic insights without AI"""
        insights = []
        
        above_target_count = len([kpi for kpi in kpi_results if kpi.status == "above_target"])
        below_target_count = len([kpi for kpi in kpi_results if kpi.status == "below_target"])
        
        if above_target_count > below_target_count:
            insights.append("Overall business performance is strong with most KPIs exceeding targets")
        elif below_target_count > above_target_count:
            insights.append("Several KPIs are below target, indicating areas for improvement")
        else:
            insights.append("Business performance is mixed with balanced results across KPIs")
        
        # Add specific insights based on KPI types
        for kpi in kpi_results:
            if "revenue" in kpi.kpi_name.lower() and kpi.status == "above_target":
                insights.append("Strong revenue performance indicates healthy business growth")
            elif "churn" in kpi.kpi_name.lower() and kpi.status == "below_target":
                insights.append("High churn rate requires immediate attention to customer retention")
        
        return '\n'.join([f"- {insight}" for insight in insights])
    
    # Additional helper methods for business analysis, trend analysis, etc.
    # (Implementation continues with remaining methods...)
    
    def _calculate_quick_metrics(self, df: pd.DataFrame) -> str:
        """Calculate quick business metrics from any dataset"""
        metrics = []
        
        # Find numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols[:3]:  # Limit to first 3 numeric columns
            total = df[col].sum()
            avg = df[col].mean()
            metrics.append(f"**{col}**: Total = {total:,.2f}, Average = {avg:.2f}")
        
        return '\n'.join(metrics) if metrics else "No numeric columns found for quick metrics."
    
    def _quick_data_quality_check(self, df: pd.DataFrame) -> str:
        """Quick data quality assessment"""
        total_cells = df.shape[0] * df.shape[1]
        missing_cells = df.isnull().sum().sum()
        missing_percentage = (missing_cells / total_cells) * 100
        
        duplicates = df.duplicated().sum()
        
        quality_score = max(0, 100 - missing_percentage - (duplicates / len(df) * 10))
        
        return f"""
- **Missing Data**: {missing_percentage:.1f}% of cells
- **Duplicate Rows**: {duplicates} ({duplicates/len(df)*100:.1f}%)
- **Data Quality Score**: {quality_score:.1f}/100
"""
    
    async def _get_ai_business_analysis(self, df: pd.DataFrame, prompt: str) -> str:
        """Get AI-powered business analysis"""
        if not HAS_OPENAI or not openai.api_key:
            return "AI analysis not available. Please configure OpenAI API key for enhanced insights."
        
        try:
            # Create data summary for AI
            data_summary = f"""
Dataset: {df.shape[0]} rows, {df.shape[1]} columns
Columns: {', '.join(df.columns.tolist())}
Data types: {df.dtypes.value_counts().to_dict()}
"""
            
            ai_prompt = f"""
Analyze this business dataset and provide insights:

{data_summary}

User request: {prompt}

Provide 3-4 key business insights and recommendations.
"""
            
            response = await openai.ChatCompletion.acreate(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": ai_prompt}],
                max_tokens=400,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return f"Error generating AI analysis: {str(e)}"    

    # Business Intelligence Analysis Methods
    
    async def _analyze_business_data(self, df: pd.DataFrame, business_context: Dict[str, Any], objectives: List[str]) -> List[BusinessInsight]:
        """Perform comprehensive business data analysis"""
        insights = []
        
        # Revenue analysis
        revenue_insight = await self._analyze_revenue_patterns(df)
        if revenue_insight:
            insights.append(revenue_insight)
        
        # Customer analysis
        customer_insight = await self._analyze_customer_patterns(df)
        if customer_insight:
            insights.append(customer_insight)
        
        # Product/service analysis
        product_insight = await self._analyze_product_performance(df)
        if product_insight:
            insights.append(product_insight)
        
        # Seasonal analysis
        seasonal_insight = await self._analyze_seasonal_patterns_business(df)
        if seasonal_insight:
            insights.append(seasonal_insight)
        
        return insights
    
    async def _analyze_revenue_patterns(self, df: pd.DataFrame) -> Optional[BusinessInsight]:
        """Analyze revenue patterns and trends"""
        revenue_col = self._find_column(df, ['revenue', 'sales', 'income', 'total'])
        date_col = self._find_column(df, ['date', 'time', 'created_at', 'timestamp'])
        
        if not revenue_col or not date_col:
            return None
        
        df[date_col] = pd.to_datetime(df[date_col])
        monthly_revenue = df.groupby(df[date_col].dt.to_period('M'))[revenue_col].sum()
        
        if len(monthly_revenue) < 3:
            return None
        
        # Calculate growth trend
        growth_rates = monthly_revenue.pct_change().dropna()
        avg_growth = growth_rates.mean() * 100
        
        recommendations = []
        if avg_growth > 5:
            recommendations.append("Maintain current growth strategies")
            recommendations.append("Consider scaling successful initiatives")
        elif avg_growth < 0:
            recommendations.append("Investigate causes of revenue decline")
            recommendations.append("Implement revenue recovery strategies")
        
        return BusinessInsight(
            title="Revenue Growth Analysis",
            description=f"Average monthly revenue growth rate is {avg_growth:.1f}%",
            category="financial",
            confidence_score=0.8,
            supporting_data={
                "avg_growth_rate": avg_growth,
                "total_revenue": monthly_revenue.sum(),
                "months_analyzed": len(monthly_revenue)
            },
            recommendations=recommendations,
            priority="high" if abs(avg_growth) > 10 else "medium"
        )
    
    async def _analyze_customer_patterns(self, df: pd.DataFrame) -> Optional[BusinessInsight]:
        """Analyze customer behavior patterns"""
        customer_col = self._find_column(df, ['customer_id', 'user_id', 'client_id'])
        
        if not customer_col:
            return None
        
        # Customer frequency analysis
        customer_frequency = df[customer_col].value_counts()
        repeat_customers = (customer_frequency > 1).sum()
        total_customers = len(customer_frequency)
        repeat_rate = (repeat_customers / total_customers) * 100
        
        recommendations = []
        if repeat_rate < 30:
            recommendations.append("Focus on customer retention strategies")
            recommendations.append("Implement loyalty programs")
        else:
            recommendations.append("Leverage high repeat rate for referral programs")
        
        return BusinessInsight(
            title="Customer Behavior Analysis",
            description=f"Customer repeat rate is {repeat_rate:.1f}%",
            category="customer",
            confidence_score=0.7,
            supporting_data={
                "repeat_rate": repeat_rate,
                "total_customers": total_customers,
                "repeat_customers": repeat_customers
            },
            recommendations=recommendations,
            priority="high" if repeat_rate < 20 else "medium"
        )
    
    async def _analyze_product_performance(self, df: pd.DataFrame) -> Optional[BusinessInsight]:
        """Analyze product or service performance"""
        product_col = self._find_column(df, ['product', 'item', 'service', 'category'])
        value_col = self._find_column(df, ['revenue', 'sales', 'amount', 'value'])
        
        if not product_col or not value_col:
            return None
        
        product_performance = df.groupby(product_col)[value_col].agg(['sum', 'count', 'mean']).round(2)
        top_products = product_performance.sort_values('sum', ascending=False).head(5)
        
        recommendations = [
            "Focus marketing efforts on top-performing products",
            "Analyze success factors of high-performing items",
            "Consider discontinuing or improving low-performing products"
        ]
        
        return BusinessInsight(
            title="Product Performance Analysis",
            description=f"Top 5 products generate {top_products['sum'].sum():.0f} in total value",
            category="operational",
            confidence_score=0.8,
            supporting_data={
                "top_products": top_products.to_dict(),
                "total_products": len(product_performance)
            },
            recommendations=recommendations,
            priority="medium"
        )
    
    async def _analyze_seasonal_patterns_business(self, df: pd.DataFrame) -> Optional[BusinessInsight]:
        """Analyze seasonal business patterns"""
        date_col = self._find_column(df, ['date', 'time', 'created_at', 'timestamp'])
        value_col = self._find_column(df, ['revenue', 'sales', 'amount', 'value'])
        
        if not date_col or not value_col:
            return None
        
        df[date_col] = pd.to_datetime(df[date_col])
        
        # Check if we have enough data for seasonal analysis (at least 12 months)
        date_range = (df[date_col].max() - df[date_col].min()).days
        if date_range < 300:  # Less than ~10 months
            return None
        
        monthly_data = df.groupby(df[date_col].dt.month)[value_col].mean()
        peak_month = monthly_data.idxmax()
        low_month = monthly_data.idxmin()
        
        month_names = {1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June',
                      7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December'}
        
        recommendations = [
            f"Prepare for peak season in {month_names[peak_month]}",
            f"Develop strategies to boost performance in {month_names[low_month]}",
            "Consider seasonal marketing campaigns"
        ]
        
        return BusinessInsight(
            title="Seasonal Pattern Analysis",
            description=f"Peak performance in {month_names[peak_month]}, lowest in {month_names[low_month]}",
            category="operational",
            confidence_score=0.6,
            supporting_data={
                "peak_month": peak_month,
                "low_month": low_month,
                "seasonal_variation": (monthly_data.max() - monthly_data.min()) / monthly_data.mean() * 100
            },
            recommendations=recommendations,
            priority="medium"
        )
    
    def _format_business_insights(self, insights: List[BusinessInsight]) -> str:
        """Format business insights for display"""
        if not insights:
            return "No significant business insights identified."
        
        formatted = []
        for insight in insights:
            priority_emoji = "🔴" if insight.priority == "high" else "🟡" if insight.priority == "medium" else "🟢"
            confidence_bar = "█" * int(insight.confidence_score * 10) + "░" * (10 - int(insight.confidence_score * 10))
            
            formatted.append(f"""
### {insight.title} {priority_emoji}
**Description**: {insight.description}
**Category**: {insight.category.title()}
**Confidence**: {confidence_bar} ({insight.confidence_score:.1%})

**Recommendations**:
{chr(10).join(f"- {rec}" for rec in insight.recommendations)}
""")
        
        return '\n'.join(formatted)
    
    # Trend Analysis Methods
    
    def _identify_time_column(self, df: pd.DataFrame) -> Optional[str]:
        """Identify the time/date column in the dataset"""
        for col in df.columns:
            if df[col].dtype == 'datetime64[ns]':
                return col
            # Try to convert to datetime, but be more selective
            try:
                # Only consider columns with date-like names or string types
                if col.lower() in ['date', 'time', 'timestamp', 'created_at', 'updated_at'] or df[col].dtype == 'object':
                    pd.to_datetime(df[col].head())
                    return col
            except:
                continue
        return None
    
    def _identify_numeric_metrics(self, df: pd.DataFrame) -> List[str]:
        """Identify numeric columns suitable for trend analysis"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # Filter out ID columns and other non-metric columns
        metrics = [col for col in numeric_cols if not any(keyword in col.lower() 
                  for keyword in ['id', 'index', 'key', 'count', 'number'])]
        return metrics[:5]  # Limit to 5 metrics
    
    async def _analyze_single_trend(self, df: pd.DataFrame, time_col: str, metric: str, forecast_periods: int) -> TrendAnalysis:
        """Analyze trend for a single metric"""
        # Prepare time series data
        df[time_col] = pd.to_datetime(df[time_col])
        ts_data = df.groupby(df[time_col].dt.to_period('M'))[metric].sum()
        
        if len(ts_data) < 3:
            raise ValueError(f"Insufficient data for trend analysis of {metric}")
        
        # Calculate trend direction and strength
        x = np.arange(len(ts_data))
        y = ts_data.values
        
        # Linear regression for trend
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        trend_direction = "increasing" if slope > 0 else "decreasing" if slope < 0 else "stable"
        trend_strength = abs(r_value)  # Correlation coefficient as strength measure
        
        # Simple forecasting (linear extrapolation)
        forecast_x = np.arange(len(ts_data), len(ts_data) + forecast_periods)
        forecast_values = [slope * x + intercept for x in forecast_x]
        
        # Confidence interval (simplified)
        std_dev = np.std(y)
        confidence_interval = (
            min(forecast_values) - 1.96 * std_dev,
            max(forecast_values) + 1.96 * std_dev
        )
        
        # Check for seasonal patterns (simplified)
        seasonal_pattern = self._detect_seasonality(ts_data)
        
        # Generate insights
        insights = [
            f"{metric} shows {trend_direction} trend with {trend_strength:.2f} strength",
            f"Average change per period: {slope:.2f}",
            f"Forecast suggests {trend_direction} pattern will continue"
        ]
        
        if seasonal_pattern:
            insights.append("Seasonal patterns detected in the data")
        
        return TrendAnalysis(
            metric_name=metric,
            trend_direction=trend_direction,
            trend_strength=trend_strength,
            seasonal_pattern=seasonal_pattern,
            forecast_values=forecast_values,
            confidence_interval=confidence_interval,
            analysis_period=f"{ts_data.index[0]} to {ts_data.index[-1]}",
            insights=insights
        )
    
    def _detect_seasonality(self, ts_data: pd.Series) -> bool:
        """Simple seasonality detection"""
        if len(ts_data) < 12:
            return False
        
        # Check for repeating patterns (simplified approach)
        # Calculate autocorrelation at lag 12 (monthly seasonality)
        try:
            autocorr = ts_data.autocorr(lag=min(12, len(ts_data)//2))
            return bool(abs(autocorr) > 0.3)  # Threshold for seasonality, ensure bool return
        except:
            return False
    
    def _format_trend_analyses(self, trend_analyses: List[TrendAnalysis]) -> str:
        """Format trend analysis results"""
        if not trend_analyses:
            return "No trend analyses completed."
        
        formatted = []
        for trend in trend_analyses:
            trend_emoji = "📈" if trend.trend_direction == "increasing" else "📉" if trend.trend_direction == "decreasing" else "➡️"
            seasonal_emoji = "🔄" if trend.seasonal_pattern else ""
            
            formatted.append(f"""
### {trend.metric_name} {trend_emoji} {seasonal_emoji}
- **Trend**: {trend.trend_direction.title()} (Strength: {trend.trend_strength:.2f})
- **Period**: {trend.analysis_period}
- **Seasonal**: {'Yes' if trend.seasonal_pattern else 'No'}
- **Insights**: {'; '.join(trend.insights)}
""")
        
        return '\n'.join(formatted)
    
    def _format_forecasting_results(self, trend_analyses: List[TrendAnalysis]) -> str:
        """Format forecasting results"""
        if not trend_analyses:
            return "No forecasting results available."
        
        formatted = []
        for trend in trend_analyses:
            avg_forecast = np.mean(trend.forecast_values)
            formatted.append(f"""
**{trend.metric_name}**:
- Average forecasted value: {avg_forecast:.2f}
- Confidence range: {trend.confidence_interval[0]:.2f} to {trend.confidence_interval[1]:.2f}
- Next 3 periods: {', '.join(f'{val:.2f}' for val in trend.forecast_values[:3])}
""")
        
        return '\n'.join(formatted)
    
    # Dashboard and Visualization Methods
    
    async def _suggest_visualizations(self, df: pd.DataFrame, prompt: str) -> List[Dict[str, Any]]:
        """Suggest appropriate visualizations based on data and context"""
        suggestions = []
        
        # Time series visualizations
        date_col = self._identify_time_column(df)
        numeric_cols = self._identify_numeric_metrics(df)
        
        if date_col and numeric_cols:
            suggestions.append({
                "type": "line",
                "title": "Trend Analysis",
                "description": "Time series line chart showing trends over time",
                "x_column": date_col,
                "y_column": numeric_cols[0],
                "chart_config": {"show_trend": True, "smooth_line": True}
            })
        
        # KPI dashboard
        if len(numeric_cols) >= 2:
            suggestions.append({
                "type": "bar",
                "title": "Performance Metrics",
                "description": "Bar chart comparing key performance metrics",
                "x_column": "metric_name",
                "y_column": "metric_value",
                "chart_config": {"color_scheme": "business", "show_values": True}
            })
        
        # Category analysis
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        if categorical_cols and numeric_cols:
            suggestions.append({
                "type": "pie",
                "title": "Category Distribution",
                "description": "Pie chart showing distribution by category",
                "labels_column": categorical_cols[0],
                "values_column": numeric_cols[0],
                "chart_config": {"show_percentages": True}
            })
        
        # Correlation heatmap
        if len(numeric_cols) > 2:
            suggestions.append({
                "type": "heatmap",
                "title": "Correlation Analysis",
                "description": "Heatmap showing correlations between metrics",
                "data_type": "correlation_matrix",
                "chart_config": {"color_scale": "RdBu", "show_values": True}
            })
        
        return suggestions
    
    async def _create_dashboard_specification(self, df: pd.DataFrame, viz_suggestions: List[Dict[str, Any]], config: Dict[str, Any]) -> Dict[str, Any]:
        """Create comprehensive dashboard specification"""
        dashboard_spec = {
            "id": f"business_dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "title": config.get("title", "Business Intelligence Dashboard"),
            "description": config.get("description", "Comprehensive business analytics dashboard"),
            "layout": {
                "type": "grid",
                "columns": 2,
                "responsive": True
            },
            "data_source": {
                "type": "dataframe",
                "refresh_interval": config.get("refresh_interval", 300),  # 5 minutes
                "auto_refresh": config.get("auto_refresh", True)
            },
            "charts": viz_suggestions,
            "filters": self._create_dashboard_filters(df),
            "kpis": await self._create_kpi_widgets(df),
            "real_time": {
                "enabled": config.get("real_time", False),
                "update_frequency": "5min"
            },
            "export_options": ["png", "pdf", "html"],
            "sharing": {
                "public": config.get("public", False),
                "permissions": config.get("permissions", ["view"])
            }
        }
        
        return dashboard_spec
    
    def _create_dashboard_filters(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Create interactive filters for dashboard"""
        filters = []
        
        # Date range filter
        date_col = self._identify_time_column(df)
        if date_col:
            filters.append({
                "type": "date_range",
                "column": date_col,
                "label": "Date Range",
                "default": "last_30_days"
            })
        
        # Category filters
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        for col in categorical_cols[:3]:  # Limit to 3 category filters
            unique_values = df[col].unique().tolist()
            if len(unique_values) <= 20:  # Only create filter if reasonable number of options
                filters.append({
                    "type": "multi_select",
                    "column": col,
                    "label": col.replace('_', ' ').title(),
                    "options": unique_values,
                    "default": "all"
                })
        
        return filters
    
    async def _create_kpi_widgets(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Create KPI widgets for dashboard"""
        kpi_widgets = []
        
        # Revenue KPI
        revenue_col = self._find_column(df, ['revenue', 'sales', 'income'])
        if revenue_col:
            total_revenue = df[revenue_col].sum()
            kpi_widgets.append({
                "type": "kpi_card",
                "title": "Total Revenue",
                "value": total_revenue,
                "format": "currency",
                "trend": "up",  # Would calculate actual trend
                "target": total_revenue * 1.1,  # 10% increase target
                "color": "green"
            })
        
        # Customer count KPI
        customer_col = self._find_column(df, ['customer_id', 'user_id', 'client_id'])
        if customer_col:
            unique_customers = df[customer_col].nunique()
            kpi_widgets.append({
                "type": "kpi_card",
                "title": "Total Customers",
                "value": unique_customers,
                "format": "number",
                "trend": "stable",
                "color": "blue"
            })
        
        # Average order value KPI
        if revenue_col and customer_col:
            avg_order_value = df[revenue_col].mean()
            kpi_widgets.append({
                "type": "kpi_card",
                "title": "Avg Order Value",
                "value": avg_order_value,
                "format": "currency",
                "trend": "up",
                "color": "purple"
            })
        
        return kpi_widgets
    
    async def _generate_scrollviz_config(self, dashboard_spec: Dict[str, Any]) -> str:
        """Generate ScrollViz integration configuration"""
        config_template = f"""
## ScrollViz Integration Configuration

### Dashboard Setup
```python
from scrollintel.engines.scroll_viz_engine import ScrollVizEngine

# Initialize ScrollViz engine
viz_engine = ScrollVizEngine()
await viz_engine.initialize()

# Create dashboard
dashboard_config = {json.dumps(dashboard_spec, indent=2)}

# Generate dashboard
dashboard = await viz_engine.process(
    input_data=your_dataframe,
    parameters={{
        "action": "create_dashboard",
        "dashboard_config": dashboard_config
    }}
)
```

### Real-time Data Binding
```python
# Set up real-time updates
async def update_dashboard_data():
    new_data = fetch_latest_data()  # Your data fetching logic
    
    updated_dashboard = await viz_engine.process(
        input_data=new_data,
        parameters={{
            "action": "update_dashboard",
            "dashboard_id": "{dashboard_spec['id']}"
        }}
    )
    
    return updated_dashboard

# Schedule updates every 5 minutes
import asyncio
asyncio.create_task(schedule_dashboard_updates(update_dashboard_data, interval=300))
```

### Chart Customization
```python
# Customize individual charts
for chart in dashboard_config["charts"]:
    chart_result = await viz_engine.process(
        input_data=your_dataframe,
        parameters={{
            "action": "generate_chart",
            "chart_type": chart["type"],
            "x_column": chart["x_column"],
            "y_column": chart["y_column"],
            "title": chart["title"],
            "theme": "business_professional"
        }}
    )
```
"""
        return config_template
    
    def _describe_interactive_features(self, dashboard_spec: Dict[str, Any]) -> str:
        """Describe interactive features of the dashboard"""
        features = []
        
        if dashboard_spec.get("filters"):
            features.append("- **Interactive Filters**: Date range, category selection, and custom filters")
        
        if dashboard_spec.get("kpis"):
            features.append("- **KPI Cards**: Real-time key performance indicators with trend indicators")
        
        features.append("- **Drill-down Capability**: Click on charts to explore detailed data")
        features.append("- **Export Options**: Download charts as PNG, PDF, or HTML")
        features.append("- **Responsive Design**: Optimized for desktop, tablet, and mobile viewing")
        
        if dashboard_spec.get("real_time", {}).get("enabled"):
            features.append("- **Real-time Updates**: Automatic data refresh every 5 minutes")
        
        return '\n'.join(features)
    
    def _setup_realtime_binding(self, dashboard_spec: Dict[str, Any]) -> str:
        """Setup instructions for real-time data binding"""
        if not dashboard_spec.get("real_time", {}).get("enabled"):
            return "Real-time updates are disabled for this dashboard."
        
        return """
### Real-time Data Binding Setup

1. **Data Source Configuration**:
   - Configure your data source for automatic updates
   - Set up database triggers or API webhooks
   - Implement data validation and error handling

2. **Update Frequency**: Every 5 minutes (configurable)

3. **Performance Optimization**:
   - Use incremental data loading
   - Implement caching for frequently accessed data
   - Monitor dashboard performance metrics

4. **Error Handling**:
   - Graceful degradation when data source is unavailable
   - User notifications for data update failures
   - Automatic retry mechanisms
"""
    
    def _generate_deployment_instructions(self, dashboard_spec: Dict[str, Any]) -> str:
        """Generate deployment instructions for the dashboard"""
        return f"""
### Dashboard Deployment Instructions

1. **Save Dashboard Configuration**:
   ```bash
   # Save the dashboard specification
   echo '{json.dumps(dashboard_spec, indent=2)}' > business_dashboard.json
   ```

2. **Deploy with ScrollViz**:
   ```python
   # Load and deploy dashboard
   import json
   from scrollintel.engines.scroll_viz_engine import ScrollVizEngine
   
   with open('business_dashboard.json', 'r') as f:
       dashboard_config = json.load(f)
   
   viz_engine = ScrollVizEngine()
   await viz_engine.initialize()
   
   # Deploy dashboard
   deployed_dashboard = await viz_engine.process(
       input_data=your_data,
       parameters={{"action": "create_dashboard", "dashboard_config": dashboard_config}}
   )
   ```

3. **Access Dashboard**:
   - URL: `/dashboards/{dashboard_spec['id']}`
   - Sharing: {'Public' if dashboard_spec.get('sharing', {}).get('public') else 'Private'}
   - Permissions: {dashboard_spec.get('sharing', {}).get('permissions', ['view'])}

4. **Monitoring and Maintenance**:
   - Monitor dashboard performance
   - Regular data quality checks
   - User feedback collection
   - Periodic dashboard optimization
"""
    
    def _suggest_performance_optimizations(self, df: pd.DataFrame, dashboard_spec: Dict[str, Any]) -> str:
        """Suggest performance optimizations for the dashboard"""
        optimizations = []
        
        # Data size optimization
        if len(df) > 10000:
            optimizations.append("- Consider data aggregation or sampling for large datasets")
            optimizations.append("- Implement pagination for detailed views")
        
        # Chart optimization
        chart_count = len(dashboard_spec.get("charts", []))
        if chart_count > 6:
            optimizations.append("- Consider splitting into multiple dashboard pages")
            optimizations.append("- Use lazy loading for charts below the fold")
        
        # Real-time optimization
        if dashboard_spec.get("real_time", {}).get("enabled"):
            optimizations.append("- Use WebSocket connections for real-time updates")
            optimizations.append("- Implement client-side caching")
        
        # General optimizations
        optimizations.extend([
            "- Use CDN for static assets",
            "- Implement server-side caching",
            "- Optimize database queries with proper indexing",
            "- Monitor and optimize chart rendering performance"
        ])
        
        return '\n'.join(optimizations)
    
    # Additional utility methods
    
    def _calculate_business_performance_summary(self, df: pd.DataFrame) -> str:
        """Calculate overall business performance summary"""
        summary_items = []
        
        # Data coverage
        date_col = self._identify_time_column(df)
        if date_col:
            df[date_col] = pd.to_datetime(df[date_col])
            date_range = (df[date_col].max() - df[date_col].min()).days
            summary_items.append(f"**Data Coverage**: {date_range} days")
        
        # Record count
        summary_items.append(f"**Total Records**: {len(df):,}")
        
        # Key metrics
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            total_value = df[numeric_cols].sum().sum()
            summary_items.append(f"**Total Value Across All Metrics**: {total_value:,.2f}")
        
        # Data quality
        missing_percentage = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
        quality_status = "Excellent" if missing_percentage < 5 else "Good" if missing_percentage < 15 else "Needs Attention"
        summary_items.append(f"**Data Quality**: {quality_status} ({missing_percentage:.1f}% missing)")
        
        return '\n'.join(summary_items)
    
    def _determine_analysis_period(self, df: pd.DataFrame) -> str:
        """Determine the analysis period from the dataset"""
        date_col = self._identify_time_column(df)
        if date_col:
            df[date_col] = pd.to_datetime(df[date_col])
            start_date = df[date_col].min().strftime('%Y-%m-%d')
            end_date = df[date_col].max().strftime('%Y-%m-%d')
            return f"{start_date} to {end_date}"
        return "Date range not available"
    
    def _get_analysis_time_range(self, df: pd.DataFrame, time_col: str) -> str:
        """Get the time range for analysis"""
        df[time_col] = pd.to_datetime(df[time_col])
        start_date = df[time_col].min().strftime('%Y-%m-%d')
        end_date = df[time_col].max().strftime('%Y-%m-%d')
        return f"{start_date} to {end_date}"
    
    def _assess_data_quality(self, df: pd.DataFrame) -> str:
        """Assess overall data quality"""
        total_cells = df.shape[0] * df.shape[1]
        missing_cells = df.isnull().sum().sum()
        missing_percentage = (missing_cells / total_cells) * 100
        
        duplicates = df.duplicated().sum()
        duplicate_percentage = (duplicates / len(df)) * 100
        
        # Data type consistency
        inconsistent_types = 0
        for col in df.columns:
            if df[col].dtype == 'object':
                # Check if numeric data is stored as string
                try:
                    pd.to_numeric(df[col].dropna().head(100))
                    inconsistent_types += 1
                except:
                    pass
        
        quality_score = max(0, 100 - missing_percentage - duplicate_percentage - (inconsistent_types * 5))
        
        quality_level = "Excellent" if quality_score >= 90 else "Good" if quality_score >= 70 else "Fair" if quality_score >= 50 else "Poor"
        
        return f"""
**Overall Data Quality**: {quality_level} ({quality_score:.1f}/100)

**Quality Metrics**:
- Missing Data: {missing_percentage:.1f}% of cells
- Duplicate Records: {duplicate_percentage:.1f}% of rows
- Data Type Issues: {inconsistent_types} columns
- Completeness Score: {100 - missing_percentage:.1f}%

**Recommendations**:
{self._generate_data_quality_recommendations(missing_percentage, duplicate_percentage, inconsistent_types)}
"""
    
    def _generate_data_quality_recommendations(self, missing_pct: float, duplicate_pct: float, type_issues: int) -> str:
        """Generate data quality improvement recommendations"""
        recommendations = []
        
        if missing_pct > 10:
            recommendations.append("- Implement data validation at source to reduce missing values")
            recommendations.append("- Consider imputation strategies for critical missing data")
        
        if duplicate_pct > 5:
            recommendations.append("- Implement deduplication processes")
            recommendations.append("- Review data collection processes to prevent duplicates")
        
        if type_issues > 0:
            recommendations.append("- Standardize data types across all columns")
            recommendations.append("- Implement data type validation in ETL processes")
        
        if not recommendations:
            recommendations.append("- Data quality is good, maintain current standards")
            recommendations.append("- Consider implementing automated quality monitoring")
        
        return '\n'.join(recommendations) 
   
    # Missing methods that need to be implemented
    
    async def _generate_executive_summary(self, df: pd.DataFrame, business_context: Dict[str, Any]) -> str:
        """Generate executive summary for business report"""
        summary_points = []
        
        # Data overview
        summary_points.append(f"Analysis covers {len(df):,} records across {len(df.columns)} business dimensions")
        
        # Key metrics
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            total_value = df[numeric_cols].sum().sum()
            summary_points.append(f"Total business value across all metrics: {total_value:,.2f}")
        
        # Time period
        date_col = self._identify_time_column(df)
        if date_col:
            df[date_col] = pd.to_datetime(df[date_col])
            period = f"{df[date_col].min().strftime('%Y-%m-%d')} to {df[date_col].max().strftime('%Y-%m-%d')}"
            summary_points.append(f"Analysis period: {period}")
        
        # Industry context
        industry = business_context.get('industry', 'General Business')
        summary_points.append(f"Industry context: {industry}")
        
        return '\n'.join([f"- {point}" for point in summary_points])
    
    async def _generate_financial_analysis(self, df: pd.DataFrame) -> str:
        """Generate financial analysis section"""
        financial_metrics = []
        
        # Look for revenue/financial columns
        revenue_col = self._find_column(df, ['revenue', 'sales', 'income', 'total'])
        if revenue_col:
            total_revenue = df[revenue_col].sum()
            avg_revenue = df[revenue_col].mean()
            financial_metrics.append(f"**Total Revenue**: ${total_revenue:,.2f}")
            financial_metrics.append(f"**Average Revenue**: ${avg_revenue:.2f}")
            
            # Growth analysis if time data available
            date_col = self._identify_time_column(df)
            if date_col:
                df[date_col] = pd.to_datetime(df[date_col])
                monthly_revenue = df.groupby(df[date_col].dt.to_period('M'))[revenue_col].sum()
                if len(monthly_revenue) >= 2:
                    growth_rate = ((monthly_revenue.iloc[-1] - monthly_revenue.iloc[0]) / monthly_revenue.iloc[0]) * 100
                    financial_metrics.append(f"**Revenue Growth**: {growth_rate:.1f}%")
        
        # Cost analysis
        cost_col = self._find_column(df, ['cost', 'expense', 'spend'])
        if cost_col:
            total_cost = df[cost_col].sum()
            financial_metrics.append(f"**Total Costs**: ${total_cost:,.2f}")
            
            if revenue_col:
                profit_margin = ((df[revenue_col].sum() - total_cost) / df[revenue_col].sum()) * 100
                financial_metrics.append(f"**Profit Margin**: {profit_margin:.1f}%")
        
        return '\n'.join(financial_metrics) if financial_metrics else "No financial data available for analysis."
    
    async def _generate_operational_metrics(self, df: pd.DataFrame) -> str:
        """Generate operational metrics section"""
        operational_metrics = []
        
        # Customer metrics
        customer_col = self._find_column(df, ['customer', 'user', 'client'])
        if customer_col:
            unique_customers = df[customer_col].nunique()
            operational_metrics.append(f"**Total Customers**: {unique_customers:,}")
            
            # Customer frequency
            customer_frequency = df[customer_col].value_counts()
            avg_frequency = customer_frequency.mean()
            operational_metrics.append(f"**Average Customer Frequency**: {avg_frequency:.1f}")
        
        # Product/service metrics
        product_col = self._find_column(df, ['product', 'service', 'item'])
        if product_col:
            unique_products = df[product_col].nunique()
            operational_metrics.append(f"**Total Products/Services**: {unique_products}")
        
        # Volume metrics
        volume_col = self._find_column(df, ['quantity', 'volume', 'count'])
        if volume_col:
            total_volume = df[volume_col].sum()
            operational_metrics.append(f"**Total Volume**: {total_volume:,.0f}")
        
        return '\n'.join(operational_metrics) if operational_metrics else "No operational data available for analysis."
    
    async def _generate_strategic_recommendations(self, df: pd.DataFrame, business_context: Dict[str, Any]) -> str:
        """Generate strategic recommendations"""
        recommendations = []
        
        # Revenue optimization
        revenue_col = self._find_column(df, ['revenue', 'sales', 'income'])
        if revenue_col:
            revenue_trend = self._calculate_simple_trend(df, revenue_col)
            if revenue_trend < 0:
                recommendations.append("**Revenue Recovery**: Implement strategies to reverse declining revenue trend")
            elif revenue_trend > 0:
                recommendations.append("**Growth Acceleration**: Scale successful revenue-generating activities")
        
        # Customer focus
        customer_col = self._find_column(df, ['customer', 'user', 'client'])
        if customer_col:
            customer_frequency = df[customer_col].value_counts()
            repeat_rate = (customer_frequency > 1).sum() / len(customer_frequency)
            if repeat_rate < 0.3:
                recommendations.append("**Customer Retention**: Focus on improving customer retention and loyalty programs")
            else:
                recommendations.append("**Customer Expansion**: Leverage high retention rate for referral and upselling programs")
        
        # Operational efficiency
        recommendations.append("**Data-Driven Decisions**: Continue leveraging analytics for strategic decision making")
        recommendations.append("**Performance Monitoring**: Implement regular KPI monitoring and reporting")
        
        # Industry-specific recommendations
        industry = business_context.get('industry', '').lower()
        if 'retail' in industry:
            recommendations.append("**Inventory Optimization**: Analyze product performance for inventory management")
        elif 'saas' in industry or 'software' in industry:
            recommendations.append("**User Engagement**: Focus on feature adoption and user engagement metrics")
        
        return '\n'.join(recommendations)
    
    def _calculate_simple_trend(self, df: pd.DataFrame, column: str) -> float:
        """Calculate simple trend direction for a column"""
        try:
            values = df[column].dropna()
            if len(values) < 2:
                return 0.0
            
            # Simple linear trend
            x = np.arange(len(values))
            slope, _ = np.polyfit(x, values, 1)
            return slope
        except:
            return 0.0
    
    async def _perform_market_analysis(self, df: pd.DataFrame, business_context: Dict[str, Any]) -> str:
        """Perform market analysis"""
        market_insights = []
        
        # Market share analysis (if competitor data available)
        market_insights.append("**Market Position**: Analysis based on available business data")
        
        # Customer segmentation
        customer_col = self._find_column(df, ['customer', 'user', 'client'])
        if customer_col:
            customer_segments = df[customer_col].value_counts()
            top_segment_pct = (customer_segments.iloc[0] / customer_segments.sum()) * 100
            market_insights.append(f"**Customer Concentration**: Top customer represents {top_segment_pct:.1f}% of business")
        
        # Product/service analysis
        product_col = self._find_column(df, ['product', 'service', 'category'])
        if product_col:
            product_performance = df.groupby(product_col).size()
            top_product = product_performance.idxmax()
            market_insights.append(f"**Leading Product/Service**: {top_product}")
        
        # Geographic analysis (if location data available)
        location_col = self._find_column(df, ['location', 'region', 'city', 'state'])
        if location_col:
            locations = df[location_col].nunique()
            market_insights.append(f"**Geographic Reach**: Operating in {locations} locations")
        
        return '\n'.join(market_insights)
    
    async def _assess_business_risks(self, df: pd.DataFrame, insights: List[BusinessInsight]) -> str:
        """Assess business risks"""
        risks = []
        
        # Data quality risks
        missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
        if missing_pct > 10:
            risks.append(f"**Data Quality Risk**: {missing_pct:.1f}% missing data may impact decision accuracy")
        
        # Customer concentration risk
        customer_col = self._find_column(df, ['customer', 'user', 'client'])
        if customer_col:
            customer_concentration = df[customer_col].value_counts()
            if len(customer_concentration) > 0:
                top_customer_pct = (customer_concentration.iloc[0] / customer_concentration.sum()) * 100
                if top_customer_pct > 20:
                    risks.append(f"**Customer Concentration Risk**: {top_customer_pct:.1f}% revenue from single customer")
        
        # Revenue volatility
        revenue_col = self._find_column(df, ['revenue', 'sales', 'income'])
        if revenue_col:
            revenue_cv = df[revenue_col].std() / df[revenue_col].mean()
            if revenue_cv > 0.3:
                risks.append("**Revenue Volatility Risk**: High revenue variability detected")
        
        # Insight-based risks
        for insight in insights:
            if insight.priority == "high" and "decline" in insight.description.lower():
                risks.append(f"**Performance Risk**: {insight.title} - {insight.description}")
        
        return '\n'.join(risks) if risks else "No significant business risks identified from current data."
    
    def _generate_action_plan(self, insights: List[BusinessInsight]) -> str:
        """Generate action plan from insights"""
        actions = []
        
        # High priority actions
        high_priority_insights = [i for i in insights if i.priority == "high"]
        if high_priority_insights:
            actions.append("**Immediate Actions (High Priority):**")
            for insight in high_priority_insights:
                for rec in insight.recommendations[:2]:  # Top 2 recommendations
                    actions.append(f"- {rec}")
        
        # Medium priority actions
        medium_priority_insights = [i for i in insights if i.priority == "medium"]
        if medium_priority_insights:
            actions.append("\n**Short-term Actions (Medium Priority):**")
            for insight in medium_priority_insights:
                if insight.recommendations:
                    actions.append(f"- {insight.recommendations[0]}")
        
        # General actions
        actions.extend([
            "\n**Ongoing Actions:**",
            "- Monitor KPIs and business metrics regularly",
            "- Review and update business strategies quarterly",
            "- Invest in data quality and analytics capabilities"
        ])
        
        return '\n'.join(actions)
    
    async def _calculate_roi_projections(self, df: pd.DataFrame, insights: List[BusinessInsight]) -> str:
        """Calculate ROI projections"""
        projections = []
        
        # Revenue projection
        revenue_col = self._find_column(df, ['revenue', 'sales', 'income'])
        if revenue_col:
            current_revenue = df[revenue_col].sum()
            
            # Simple growth projection based on trend
            trend = self._calculate_simple_trend(df, revenue_col)
            if trend > 0:
                projected_growth = min(trend * 12, current_revenue * 0.2)  # Cap at 20% growth
                projected_revenue = current_revenue + projected_growth
                roi_percentage = (projected_growth / current_revenue) * 100
                projections.append(f"**Revenue Projection**: ${projected_revenue:,.2f} ({roi_percentage:.1f}% growth)")
        
        # Cost optimization potential
        cost_col = self._find_column(df, ['cost', 'expense', 'spend'])
        if cost_col:
            current_costs = df[cost_col].sum()
            potential_savings = current_costs * 0.05  # Assume 5% optimization potential
            projections.append(f"**Cost Optimization Potential**: ${potential_savings:,.2f} (5% reduction)")
        
        # Customer value projection
        customer_col = self._find_column(df, ['customer', 'user', 'client'])
        if customer_col and revenue_col:
            avg_customer_value = df[revenue_col].sum() / df[customer_col].nunique()
            customer_growth_potential = avg_customer_value * df[customer_col].nunique() * 0.1  # 10% customer growth
            projections.append(f"**Customer Growth Value**: ${customer_growth_potential:,.2f} (10% customer increase)")
        
        return '\n'.join(projections) if projections else "ROI projections require more historical data for accurate modeling."
    
    async def _generate_report_kpis(self, df: pd.DataFrame) -> str:
        """Generate KPIs for business report"""
        try:
            # Use existing KPI generation logic
            suggested_kpis = await self._suggest_kpis_from_data(df, "business report")
            kpi_results = []
            
            for kpi_name in suggested_kpis[:3]:  # Limit to 3 KPIs for report
                try:
                    kpi_result = await self._calculate_single_kpi(df, kpi_name)
                    if kpi_result:
                        kpi_results.append(kpi_result)
                except Exception:
                    continue
            
            if kpi_results:
                return self._format_kpi_results(kpi_results)
            else:
                return "KPI calculation requires more structured business data."
        except Exception:
            return "Unable to generate KPIs from current dataset structure."
    
    async def _analyze_market_position(self, df: pd.DataFrame, business_context: Dict[str, Any]) -> str:
        """Analyze market position"""
        position_analysis = []
        
        # Market share indicators
        revenue_col = self._find_column(df, ['revenue', 'sales', 'income'])
        if revenue_col:
            total_revenue = df[revenue_col].sum()
            position_analysis.append(f"**Revenue Scale**: ${total_revenue:,.2f} total business value")
        
        # Customer base analysis
        customer_col = self._find_column(df, ['customer', 'user', 'client'])
        if customer_col:
            customer_base = df[customer_col].nunique()
            position_analysis.append(f"**Customer Base**: {customer_base:,} unique customers")
        
        # Product/service portfolio
        product_col = self._find_column(df, ['product', 'service', 'category'])
        if product_col:
            portfolio_size = df[product_col].nunique()
            position_analysis.append(f"**Portfolio Diversity**: {portfolio_size} products/services")
        
        # Competitive positioning (based on available data)
        industry = business_context.get('industry', 'General')
        position_analysis.append(f"**Industry**: {industry}")
        position_analysis.append("**Competitive Analysis**: Requires external market data for comprehensive assessment")
        
        return '\n'.join(position_analysis)
    
    async def _generate_risk_assessment(self, df: pd.DataFrame) -> str:
        """Generate risk assessment for business report"""
        return await self._assess_business_risks(df, [])  # Use existing method
    
    def _create_implementation_roadmap(self, recommendations: str) -> str:
        """Create implementation roadmap"""
        roadmap = [
            "## Implementation Timeline",
            "",
            "**Phase 1 (0-30 days): Immediate Actions**",
            "- Implement high-priority recommendations",
            "- Set up monitoring and tracking systems",
            "- Establish baseline metrics",
            "",
            "**Phase 2 (30-90 days): Strategic Initiatives**",
            "- Execute medium-priority improvements",
            "- Develop detailed action plans",
            "- Begin performance optimization",
            "",
            "**Phase 3 (90+ days): Long-term Growth**",
            "- Scale successful initiatives",
            "- Continuous improvement processes",
            "- Strategic planning and expansion",
            "",
            "**Success Metrics:**",
            "- Monthly KPI reviews",
            "- Quarterly strategy assessments",
            "- Annual comprehensive analysis"
        ]
        
        return '\n'.join(roadmap)
    
    def _document_methodology(self, df: pd.DataFrame) -> str:
        """Document analysis methodology"""
        methodology = [
            "## Data Sources",
            f"- Primary dataset: {len(df)} records, {len(df.columns)} variables",
            f"- Data types: {df.dtypes.value_counts().to_dict()}",
            f"- Analysis period: {self._determine_analysis_period(df)}",
            "",
            "## Analytical Methods",
            "- Descriptive statistics and trend analysis",
            "- KPI calculation and performance benchmarking",
            "- Business intelligence pattern recognition",
            "- Risk assessment and opportunity identification",
            "",
            "## Quality Assurance",
            f"- Data completeness: {((df.shape[0] * df.shape[1] - df.isnull().sum().sum()) / (df.shape[0] * df.shape[1]) * 100):.1f}%",
            f"- Duplicate records: {df.duplicated().sum()} ({df.duplicated().sum()/len(df)*100:.1f}%)",
            "- Statistical validation applied to all calculations"
        ]
        
        return '\n'.join(methodology)
    
    async def _get_ai_trend_insights(self, trend_analyses: List[TrendAnalysis]) -> str:
        """Get AI insights on trend analysis"""
        if not HAS_OPENAI or not openai.api_key:
            return self._generate_basic_trend_insights(trend_analyses)
        
        try:
            trend_summary = []
            for trend in trend_analyses:
                trend_summary.append(f"{trend.metric_name}: {trend.trend_direction} trend (strength: {trend.trend_strength:.2f})")
            
            prompt = f"""
Analyze these business trends and provide strategic insights:

{chr(10).join(trend_summary)}

Provide 3-4 key insights about trend patterns and business implications.
"""
            
            response = await openai.ChatCompletion.acreate(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Error getting AI trend insights: {str(e)}")
            return self._generate_basic_trend_insights(trend_analyses)
    
    def _generate_basic_trend_insights(self, trend_analyses: List[TrendAnalysis]) -> str:
        """Generate basic trend insights without AI"""
        if not trend_analyses:
            return "No trend data available for analysis."
        
        insights = []
        
        increasing_trends = [t for t in trend_analyses if t.trend_direction == "increasing"]
        decreasing_trends = [t for t in trend_analyses if t.trend_direction == "decreasing"]
        
        if len(increasing_trends) > len(decreasing_trends):
            insights.append("Overall business trends are positive with most metrics showing growth")
        elif len(decreasing_trends) > len(increasing_trends):
            insights.append("Several metrics show declining trends requiring attention")
        
        # Seasonal insights
        seasonal_metrics = [t for t in trend_analyses if t.seasonal_pattern]
        if seasonal_metrics:
            insights.append(f"{len(seasonal_metrics)} metrics show seasonal patterns - plan accordingly")
        
        # Strength insights
        strong_trends = [t for t in trend_analyses if t.trend_strength > 0.7]
        if strong_trends:
            insights.append(f"{len(strong_trends)} metrics show strong trend patterns with high predictability")
        
        return '\n'.join([f"- {insight}" for insight in insights])
    
    async def _interpret_trend_business_impact(self, trend_analyses: List[TrendAnalysis]) -> str:
        """Interpret business impact of trends"""
        impacts = []
        
        for trend in trend_analyses:
            if "revenue" in trend.metric_name.lower():
                if trend.trend_direction == "increasing":
                    impacts.append("**Revenue Growth**: Positive revenue trends support business expansion")
                else:
                    impacts.append("**Revenue Concern**: Declining revenue trends require immediate attention")
            
            elif "customer" in trend.metric_name.lower():
                if trend.trend_direction == "increasing":
                    impacts.append("**Customer Growth**: Expanding customer base indicates market success")
                else:
                    impacts.append("**Customer Retention**: Declining customer metrics suggest retention issues")
        
        # General business implications
        if not impacts:
            impacts.append("**Business Performance**: Trend analysis provides insights for strategic planning")
        
        impacts.append("**Forecasting**: Use trend patterns for budget planning and resource allocation")
        
        return '\n'.join(impacts)
    
    def _generate_trend_recommendations(self, trend_analyses: List[TrendAnalysis]) -> str:
        """Generate recommendations based on trend analysis"""
        recommendations = []
        
        for trend in trend_analyses:
            if trend.trend_direction == "decreasing" and trend.trend_strength > 0.5:
                recommendations.append(f"**{trend.metric_name}**: Implement corrective measures to reverse declining trend")
            elif trend.trend_direction == "increasing" and trend.trend_strength > 0.5:
                recommendations.append(f"**{trend.metric_name}**: Scale activities driving positive trend")
            
            if trend.seasonal_pattern:
                recommendations.append(f"**{trend.metric_name}**: Plan for seasonal variations in business operations")
        
        # General recommendations
        recommendations.extend([
            "**Monitoring**: Set up automated trend monitoring and alerts",
            "**Planning**: Use forecasts for strategic planning and budgeting",
            "**Optimization**: Focus resources on metrics with strongest positive trends"
        ])
        
        return '\n'.join(recommendations)
    
    def _identify_trend_risks(self, trend_analyses: List[TrendAnalysis]) -> str:
        """Identify risks from trend analysis"""
        risks = []
        
        # Declining trends
        declining_trends = [t for t in trend_analyses if t.trend_direction == "decreasing"]
        if declining_trends:
            risks.append(f"**Declining Performance**: {len(declining_trends)} metrics showing negative trends")
        
        # High volatility
        volatile_trends = [t for t in trend_analyses if t.trend_strength < 0.3]
        if volatile_trends:
            risks.append(f"**Volatility Risk**: {len(volatile_trends)} metrics show unpredictable patterns")
        
        # Seasonal dependency
        seasonal_trends = [t for t in trend_analyses if t.seasonal_pattern]
        if len(seasonal_trends) > len(trend_analyses) * 0.5:
            risks.append("**Seasonal Dependency**: High reliance on seasonal patterns")
        
        return '\n'.join(risks) if risks else "No significant trend-based risks identified."
    
    def _analyze_seasonal_patterns(self, trend_analyses: List[TrendAnalysis]) -> str:
        """Analyze seasonal patterns across metrics"""
        seasonal_metrics = [t for t in trend_analyses if t.seasonal_pattern]
        
        if not seasonal_metrics:
            return "No significant seasonal patterns detected in the analyzed metrics."
        
        analysis = [
            f"**Seasonal Metrics**: {len(seasonal_metrics)} out of {len(trend_analyses)} metrics show seasonal patterns",
            "",
            "**Seasonal Insights**:"
        ]
        
        for metric in seasonal_metrics:
            analysis.append(f"- {metric.metric_name}: Shows recurring seasonal patterns")
        
        analysis.extend([
            "",
            "**Business Implications**:",
            "- Plan inventory and resources for seasonal variations",
            "- Adjust marketing strategies for seasonal peaks",
            "- Prepare cash flow for seasonal fluctuations"
        ])
        
        return '\n'.join(analysis)
    
    async def _get_ai_business_recommendations(self, df: pd.DataFrame, insights: List[BusinessInsight], prompt: str) -> str:
        """Get AI-powered business recommendations"""
        if not HAS_OPENAI or not openai.api_key:
            return self._generate_basic_recommendations(insights)
        
        try:
            insight_summary = []
            for insight in insights:
                insight_summary.append(f"{insight.title}: {insight.description}")
            
            prompt_text = f"""
Based on this business data analysis, provide strategic recommendations:

Data Insights:
{chr(10).join(insight_summary)}

User Request: {prompt}

Provide 3-5 actionable business recommendations.
"""
            
            response = await openai.ChatCompletion.acreate(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt_text}],
                max_tokens=400,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Error getting AI recommendations: {str(e)}")
            return self._generate_basic_recommendations(insights)
    
    def _generate_basic_recommendations(self, insights: List[BusinessInsight]) -> str:
        """Generate basic recommendations without AI"""
        recommendations = []
        
        for insight in insights:
            if insight.priority == "high":
                recommendations.extend(insight.recommendations)
        
        if not recommendations:
            recommendations = [
                "Continue monitoring key business metrics",
                "Focus on data-driven decision making",
                "Implement regular performance reviews"
            ]
        
        return '\n'.join([f"- {rec}" for rec in recommendations])

    async def set_scrollviz_engine(self, scrollviz_engine):
        """Set reference to ScrollViz engine for integration"""
        self.scrollviz_engine = scrollviz_engine