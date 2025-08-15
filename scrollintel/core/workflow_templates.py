"""
Workflow Templates for ScrollIntel Multi-Agent Scenarios.
Defines common workflow patterns and templates.
"""

from typing import Dict, List, Any
from dataclasses import dataclass, field

from .orchestrator import WorkflowTemplate


class WorkflowTemplateLibrary:
    """Library of predefined workflow templates for common scenarios."""
    
    @staticmethod
    def get_data_science_pipeline() -> WorkflowTemplate:
        """Complete data science pipeline from raw data to insights."""
        return WorkflowTemplate(
            id="data_science_pipeline",
            name="Complete Data Science Pipeline",
            description="End-to-end data science workflow with EDA, modeling, and visualization",
            tasks=[
                {
                    "name": "Data Validation and Profiling",
                    "agent_type": "data_scientist",
                    "prompt": "Validate the uploaded dataset, check data quality, and generate data profile",
                    "context": {
                        "task_type": "data_validation",
                        "required_capabilities": ["data_analysis", "data_quality"]
                    },
                    "timeout": 300.0,
                    "max_retries": 2
                },
                {
                    "name": "Exploratory Data Analysis",
                    "agent_type": "data_scientist",
                    "prompt": "Perform comprehensive EDA including statistical analysis, correlation analysis, and outlier detection",
                    "context": {
                        "task_type": "eda",
                        "required_capabilities": ["statistical_analysis", "visualization"]
                    },
                    "dependencies": [{"task_id": "0", "dependency_type": "completion"}],
                    "timeout": 600.0,
                    "max_retries": 2
                },
                {
                    "name": "Feature Engineering",
                    "agent_type": "data_scientist",
                    "prompt": "Create and select features for machine learning based on EDA insights",
                    "context": {
                        "task_type": "feature_engineering",
                        "required_capabilities": ["feature_engineering", "data_preprocessing"]
                    },
                    "dependencies": [{"task_id": "1", "dependency_type": "completion"}],
                    "timeout": 450.0,
                    "max_retries": 2
                },
                {
                    "name": "Model Training and Evaluation",
                    "agent_type": "ml_engineer",
                    "prompt": "Train multiple ML models, perform hyperparameter tuning, and evaluate performance",
                    "context": {
                        "task_type": "model_training",
                        "required_capabilities": ["model_training", "hyperparameter_tuning"]
                    },
                    "dependencies": [{"task_id": "2", "dependency_type": "completion"}],
                    "timeout": 1200.0,
                    "max_retries": 3
                },
                {
                    "name": "Model Interpretation and Insights",
                    "agent_type": "data_scientist",
                    "prompt": "Generate model interpretations, feature importance, and business insights",
                    "context": {
                        "task_type": "model_interpretation",
                        "required_capabilities": ["model_interpretation", "business_insights"]
                    },
                    "dependencies": [{"task_id": "3", "dependency_type": "completion"}],
                    "timeout": 300.0,
                    "max_retries": 2
                },
                {
                    "name": "Visualization and Dashboard Creation",
                    "agent_type": "bi_developer",
                    "prompt": "Create comprehensive visualizations and interactive dashboard for the analysis",
                    "context": {
                        "task_type": "visualization",
                        "required_capabilities": ["dashboard_creation", "data_visualization"]
                    },
                    "dependencies": [{"task_id": "4", "dependency_type": "completion"}],
                    "timeout": 600.0,
                    "max_retries": 2
                }
            ],
            estimated_duration=3600.0,  # 1 hour
            tags=["data_science", "ml_pipeline", "end_to_end", "visualization"]
        )
    
    @staticmethod
    def get_business_intelligence_report() -> WorkflowTemplate:
        """Business intelligence report generation workflow."""
        return WorkflowTemplate(
            id="bi_report_generation",
            name="Business Intelligence Report Generation",
            description="Generate comprehensive BI report with KPIs, trends, and recommendations",
            tasks=[
                {
                    "name": "Business Data Analysis",
                    "agent_type": "analyst",
                    "prompt": "Analyze business data to identify key metrics, trends, and patterns",
                    "context": {
                        "task_type": "business_analysis",
                        "required_capabilities": ["business_analysis", "trend_analysis"]
                    },
                    "timeout": 450.0,
                    "max_retries": 2
                },
                {
                    "name": "KPI Calculation and Benchmarking",
                    "agent_type": "analyst",
                    "prompt": "Calculate key performance indicators and benchmark against industry standards",
                    "context": {
                        "task_type": "kpi_calculation",
                        "required_capabilities": ["kpi_generation", "benchmarking"]
                    },
                    "dependencies": [{"task_id": "0", "dependency_type": "completion"}],
                    "timeout": 300.0,
                    "max_retries": 2
                },
                {
                    "name": "Forecasting and Predictions",
                    "agent_type": "data_scientist",
                    "prompt": "Generate forecasts and predictions for key business metrics",
                    "context": {
                        "task_type": "forecasting",
                        "required_capabilities": ["time_series_analysis", "forecasting"]
                    },
                    "dependencies": [{"task_id": "1", "dependency_type": "completion"}],
                    "timeout": 600.0,
                    "max_retries": 2
                },
                {
                    "name": "Interactive Dashboard Creation",
                    "agent_type": "bi_developer",
                    "prompt": "Create interactive dashboard with KPIs, trends, and drill-down capabilities",
                    "context": {
                        "task_type": "dashboard_creation",
                        "required_capabilities": ["dashboard_creation", "interactive_visualization"]
                    },
                    "dependencies": [{"task_id": "2", "dependency_type": "completion"}],
                    "timeout": 600.0,
                    "max_retries": 2
                },
                {
                    "name": "Executive Summary Generation",
                    "agent_type": "analyst",
                    "prompt": "Generate executive summary with key insights, recommendations, and action items",
                    "context": {
                        "task_type": "executive_summary",
                        "required_capabilities": ["report_generation", "business_insights"]
                    },
                    "dependencies": [{"task_id": "3", "dependency_type": "completion"}],
                    "timeout": 300.0,
                    "max_retries": 2
                }
            ],
            estimated_duration=2250.0,  # 37.5 minutes
            tags=["business_intelligence", "reporting", "kpi", "dashboard"]
        )
    
    @staticmethod
    def get_ai_model_deployment() -> WorkflowTemplate:
        """AI model deployment pipeline workflow."""
        return WorkflowTemplate(
            id="ai_model_deployment",
            name="AI Model Deployment Pipeline",
            description="Complete pipeline for deploying AI models to production",
            tasks=[
                {
                    "name": "Model Validation and Testing",
                    "agent_type": "ml_engineer",
                    "prompt": "Validate model performance, run comprehensive tests, and check for bias",
                    "context": {
                        "task_type": "model_validation",
                        "required_capabilities": ["model_validation", "bias_detection"]
                    },
                    "timeout": 600.0,
                    "max_retries": 2
                },
                {
                    "name": "Model Optimization and Compression",
                    "agent_type": "ai_engineer",
                    "prompt": "Optimize model for production deployment, including quantization and compression",
                    "context": {
                        "task_type": "model_optimization",
                        "required_capabilities": ["model_optimization", "quantization"]
                    },
                    "dependencies": [{"task_id": "0", "dependency_type": "completion"}],
                    "timeout": 450.0,
                    "max_retries": 2
                },
                {
                    "name": "API Endpoint Creation",
                    "agent_type": "ai_engineer",
                    "prompt": "Create REST API endpoints for model serving with proper error handling",
                    "context": {
                        "task_type": "api_creation",
                        "required_capabilities": ["api_development", "model_serving"]
                    },
                    "dependencies": [{"task_id": "1", "dependency_type": "completion"}],
                    "timeout": 600.0,
                    "max_retries": 2
                },
                {
                    "name": "Monitoring and Logging Setup",
                    "agent_type": "ai_engineer",
                    "prompt": "Set up monitoring, logging, and alerting for the deployed model",
                    "context": {
                        "task_type": "monitoring_setup",
                        "required_capabilities": ["monitoring", "logging"]
                    },
                    "dependencies": [{"task_id": "2", "dependency_type": "completion"}],
                    "timeout": 300.0,
                    "max_retries": 2
                },
                {
                    "name": "Load Testing and Performance Validation",
                    "agent_type": "ml_engineer",
                    "prompt": "Perform load testing and validate performance under production conditions",
                    "context": {
                        "task_type": "load_testing",
                        "required_capabilities": ["load_testing", "performance_validation"]
                    },
                    "dependencies": [{"task_id": "3", "dependency_type": "completion"}],
                    "timeout": 450.0,
                    "max_retries": 2
                },
                {
                    "name": "Documentation and Deployment Guide",
                    "agent_type": "ai_engineer",
                    "prompt": "Generate comprehensive documentation and deployment guide",
                    "context": {
                        "task_type": "documentation",
                        "required_capabilities": ["documentation", "deployment_guide"]
                    },
                    "dependencies": [{"task_id": "4", "dependency_type": "completion"}],
                    "timeout": 300.0,
                    "max_retries": 2
                }
            ],
            estimated_duration=2700.0,  # 45 minutes
            tags=["ai_deployment", "model_serving", "production", "monitoring"]
        )
    
    @staticmethod
    def get_data_quality_assessment() -> WorkflowTemplate:
        """Data quality assessment and remediation workflow."""
        return WorkflowTemplate(
            id="data_quality_assessment",
            name="Data Quality Assessment and Remediation",
            description="Comprehensive data quality assessment with automated remediation suggestions",
            tasks=[
                {
                    "name": "Data Profiling and Schema Analysis",
                    "agent_type": "data_scientist",
                    "prompt": "Profile the dataset and analyze schema consistency, data types, and structure",
                    "context": {
                        "task_type": "data_profiling",
                        "required_capabilities": ["data_profiling", "schema_analysis"]
                    },
                    "timeout": 300.0,
                    "max_retries": 2
                },
                {
                    "name": "Missing Data Analysis",
                    "agent_type": "data_scientist",
                    "prompt": "Analyze missing data patterns and suggest imputation strategies",
                    "context": {
                        "task_type": "missing_data_analysis",
                        "required_capabilities": ["missing_data_analysis", "imputation"]
                    },
                    "dependencies": [{"task_id": "0", "dependency_type": "completion"}],
                    "timeout": 300.0,
                    "max_retries": 2
                },
                {
                    "name": "Outlier Detection and Analysis",
                    "agent_type": "data_scientist",
                    "prompt": "Detect outliers using multiple methods and analyze their impact",
                    "context": {
                        "task_type": "outlier_detection",
                        "required_capabilities": ["outlier_detection", "statistical_analysis"]
                    },
                    "dependencies": [{"task_id": "0", "dependency_type": "completion"}],
                    "timeout": 300.0,
                    "max_retries": 2
                },
                {
                    "name": "Data Consistency and Integrity Checks",
                    "agent_type": "data_scientist",
                    "prompt": "Check data consistency, referential integrity, and business rule violations",
                    "context": {
                        "task_type": "consistency_checks",
                        "required_capabilities": ["data_validation", "integrity_checks"]
                    },
                    "dependencies": [{"task_id": "1", "dependency_type": "completion"}],
                    "timeout": 300.0,
                    "max_retries": 2
                },
                {
                    "name": "Data Quality Report Generation",
                    "agent_type": "analyst",
                    "prompt": "Generate comprehensive data quality report with scores and recommendations",
                    "context": {
                        "task_type": "quality_report",
                        "required_capabilities": ["report_generation", "data_quality_scoring"]
                    },
                    "dependencies": [
                        {"task_id": "2", "dependency_type": "completion"},
                        {"task_id": "3", "dependency_type": "completion"}
                    ],
                    "timeout": 300.0,
                    "max_retries": 2
                },
                {
                    "name": "Remediation Plan Creation",
                    "agent_type": "data_scientist",
                    "prompt": "Create actionable remediation plan with prioritized recommendations",
                    "context": {
                        "task_type": "remediation_plan",
                        "required_capabilities": ["remediation_planning", "data_cleaning"]
                    },
                    "dependencies": [{"task_id": "4", "dependency_type": "completion"}],
                    "timeout": 300.0,
                    "max_retries": 2
                }
            ],
            estimated_duration=1800.0,  # 30 minutes
            tags=["data_quality", "data_profiling", "data_cleaning", "assessment"]
        )
    
    @staticmethod
    def get_competitive_analysis() -> WorkflowTemplate:
        """Competitive analysis and market intelligence workflow."""
        return WorkflowTemplate(
            id="competitive_analysis",
            name="Competitive Analysis and Market Intelligence",
            description="Comprehensive competitive analysis with market positioning insights",
            tasks=[
                {
                    "name": "Market Data Collection and Analysis",
                    "agent_type": "analyst",
                    "prompt": "Collect and analyze market data, trends, and competitive landscape",
                    "context": {
                        "task_type": "market_analysis",
                        "required_capabilities": ["market_analysis", "competitive_intelligence"]
                    },
                    "timeout": 600.0,
                    "max_retries": 2
                },
                {
                    "name": "Competitor Performance Analysis",
                    "agent_type": "analyst",
                    "prompt": "Analyze competitor performance metrics, strengths, and weaknesses",
                    "context": {
                        "task_type": "competitor_analysis",
                        "required_capabilities": ["competitor_analysis", "performance_metrics"]
                    },
                    "dependencies": [{"task_id": "0", "dependency_type": "completion"}],
                    "timeout": 450.0,
                    "max_retries": 2
                },
                {
                    "name": "SWOT Analysis Generation",
                    "agent_type": "analyst",
                    "prompt": "Generate SWOT analysis comparing company position against competitors",
                    "context": {
                        "task_type": "swot_analysis",
                        "required_capabilities": ["swot_analysis", "strategic_analysis"]
                    },
                    "dependencies": [{"task_id": "1", "dependency_type": "completion"}],
                    "timeout": 300.0,
                    "max_retries": 2
                },
                {
                    "name": "Market Opportunity Identification",
                    "agent_type": "analyst",
                    "prompt": "Identify market opportunities and gaps based on competitive analysis",
                    "context": {
                        "task_type": "opportunity_analysis",
                        "required_capabilities": ["opportunity_analysis", "market_gaps"]
                    },
                    "dependencies": [{"task_id": "2", "dependency_type": "completion"}],
                    "timeout": 300.0,
                    "max_retries": 2
                },
                {
                    "name": "Strategic Recommendations",
                    "agent_type": "cto",
                    "prompt": "Generate strategic recommendations based on competitive analysis insights",
                    "context": {
                        "task_type": "strategic_recommendations",
                        "required_capabilities": ["strategic_planning", "business_strategy"]
                    },
                    "dependencies": [{"task_id": "3", "dependency_type": "completion"}],
                    "timeout": 450.0,
                    "max_retries": 2
                },
                {
                    "name": "Competitive Intelligence Dashboard",
                    "agent_type": "bi_developer",
                    "prompt": "Create interactive dashboard for ongoing competitive intelligence monitoring",
                    "context": {
                        "task_type": "intelligence_dashboard",
                        "required_capabilities": ["dashboard_creation", "competitive_monitoring"]
                    },
                    "dependencies": [{"task_id": "4", "dependency_type": "completion"}],
                    "timeout": 600.0,
                    "max_retries": 2
                }
            ],
            estimated_duration=2700.0,  # 45 minutes
            tags=["competitive_analysis", "market_intelligence", "strategy", "swot"]
        )
    
    @staticmethod
    def get_customer_segmentation() -> WorkflowTemplate:
        """Customer segmentation and analysis workflow."""
        return WorkflowTemplate(
            id="customer_segmentation",
            name="Customer Segmentation and Analysis",
            description="Advanced customer segmentation with behavioral analysis and targeting recommendations",
            tasks=[
                {
                    "name": "Customer Data Preparation",
                    "agent_type": "data_scientist",
                    "prompt": "Clean and prepare customer data for segmentation analysis",
                    "context": {
                        "task_type": "data_preparation",
                        "required_capabilities": ["data_cleaning", "customer_data"]
                    },
                    "timeout": 300.0,
                    "max_retries": 2
                },
                {
                    "name": "Behavioral Analysis",
                    "agent_type": "data_scientist",
                    "prompt": "Analyze customer behavior patterns, purchase history, and engagement metrics",
                    "context": {
                        "task_type": "behavioral_analysis",
                        "required_capabilities": ["behavioral_analysis", "customer_analytics"]
                    },
                    "dependencies": [{"task_id": "0", "dependency_type": "completion"}],
                    "timeout": 450.0,
                    "max_retries": 2
                },
                {
                    "name": "Segmentation Model Development",
                    "agent_type": "ml_engineer",
                    "prompt": "Develop clustering models for customer segmentation using multiple algorithms",
                    "context": {
                        "task_type": "segmentation_modeling",
                        "required_capabilities": ["clustering", "unsupervised_learning"]
                    },
                    "dependencies": [{"task_id": "1", "dependency_type": "completion"}],
                    "timeout": 600.0,
                    "max_retries": 2
                },
                {
                    "name": "Segment Profiling and Characterization",
                    "agent_type": "analyst",
                    "prompt": "Profile each customer segment with detailed characteristics and personas",
                    "context": {
                        "task_type": "segment_profiling",
                        "required_capabilities": ["customer_profiling", "persona_development"]
                    },
                    "dependencies": [{"task_id": "2", "dependency_type": "completion"}],
                    "timeout": 300.0,
                    "max_retries": 2
                },
                {
                    "name": "Targeting Strategy Development",
                    "agent_type": "analyst",
                    "prompt": "Develop targeting strategies and recommendations for each segment",
                    "context": {
                        "task_type": "targeting_strategy",
                        "required_capabilities": ["targeting_strategy", "marketing_strategy"]
                    },
                    "dependencies": [{"task_id": "3", "dependency_type": "completion"}],
                    "timeout": 300.0,
                    "max_retries": 2
                },
                {
                    "name": "Segmentation Dashboard and Monitoring",
                    "agent_type": "bi_developer",
                    "prompt": "Create dashboard for segment monitoring and performance tracking",
                    "context": {
                        "task_type": "segmentation_dashboard",
                        "required_capabilities": ["dashboard_creation", "segment_monitoring"]
                    },
                    "dependencies": [{"task_id": "4", "dependency_type": "completion"}],
                    "timeout": 450.0,
                    "max_retries": 2
                }
            ],
            estimated_duration=2400.0,  # 40 minutes
            tags=["customer_segmentation", "clustering", "behavioral_analysis", "targeting"]
        )
    
    @staticmethod
    def get_all_templates() -> Dict[str, WorkflowTemplate]:
        """Get all available workflow templates."""
        return {
            "data_science_pipeline": WorkflowTemplateLibrary.get_data_science_pipeline(),
            "bi_report_generation": WorkflowTemplateLibrary.get_business_intelligence_report(),
            "ai_model_deployment": WorkflowTemplateLibrary.get_ai_model_deployment(),
            "data_quality_assessment": WorkflowTemplateLibrary.get_data_quality_assessment(),
            "competitive_analysis": WorkflowTemplateLibrary.get_competitive_analysis(),
            "customer_segmentation": WorkflowTemplateLibrary.get_customer_segmentation(),
        }