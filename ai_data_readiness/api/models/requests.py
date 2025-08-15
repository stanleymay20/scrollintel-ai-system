"""
Request Models for AI Data Readiness Platform API

This module defines Pydantic models for API request validation and serialization.
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

from ...engines.ai_readiness_reporting_engine import IndustryStandard, ReportType, AIReadinessReport
from ...models.base_models import AIReadinessScore, QualityReport, BiasReport

# Define ReportFormat enum if not already defined
class ReportFormat(str, Enum):
    JSON = "json"
    HTML = "html"
    PDF = "pdf"
    CSV = "csv"


class ReportGenerationRequest(BaseModel):
    """Request model for generating AI readiness report"""
    
    dataset_id: str = Field(..., description="Unique identifier for the dataset")
    report_type: ReportType = Field(
        default=ReportType.DETAILED_TECHNICAL,
        description="Type of report to generate"
    )
    industry_standard: IndustryStandard = Field(
        default=IndustryStandard.GENERAL,
        description="Industry standard for benchmarking"
    )
    include_bias_analysis: bool = Field(
        default=True,
        description="Include bias analysis in the report"
    )
    include_drift_analysis: bool = Field(
        default=False,
        description="Include drift analysis in the report"
    )
    reference_dataset_id: Optional[str] = Field(
        None,
        description="Reference dataset ID for drift analysis"
    )
    protected_attributes: Optional[List[str]] = Field(
        None,
        description="Protected attributes for bias analysis"
    )
    export_formats: Optional[List[ReportFormat]] = Field(
        None,
        description="Formats to export the report"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "dataset_id": "customer_data_2024",
                "report_type": "detailed_technical",
                "industry_standard": "financial_services",
                "include_bias_analysis": True,
                "include_drift_analysis": False,
                "protected_attributes": ["gender", "age_group"],
                "export_formats": ["json", "html"]
            }
        }


class GenerateReportRequest(BaseModel):
    """Request model for generating comprehensive AI readiness report"""
    
    dataset_id: str = Field(..., description="Unique identifier for the dataset")
    ai_readiness_score: AIReadinessScore = Field(..., description="AI readiness assessment results")
    quality_report: QualityReport = Field(..., description="Data quality assessment results")
    bias_report: Optional[BiasReport] = Field(None, description="Bias analysis results (optional)")
    industry: IndustryStandard = Field(
        default=IndustryStandard.GENERAL,
        description="Industry standard for benchmarking"
    )
    target_score: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Target AI readiness score to achieve"
    )
    auto_export: bool = Field(
        default=False,
        description="Automatically export report in background"
    )
    export_formats: Optional[List[ReportFormat]] = Field(
        None,
        description="Formats to export if auto_export is enabled"
    )
    
    @validator('target_score')
    def validate_target_score(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError('Target score must be between 0.0 and 1.0')
        return v
    
    @validator('export_formats')
    def validate_export_formats(cls, v, values):
        if values.get('auto_export') and not v:
            return [ReportFormat.JSON]  # Default format
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "dataset_id": "customer_data_2024",
                "ai_readiness_score": {
                    "overall_score": 0.75,
                    "data_quality_score": 0.80,
                    "feature_quality_score": 0.70,
                    "bias_score": 0.75,
                    "compliance_score": 0.85,
                    "scalability_score": 0.70
                },
                "quality_report": {
                    "dataset_id": "customer_data_2024",
                    "overall_score": 0.80,
                    "completeness_score": 0.85,
                    "accuracy_score": 0.80,
                    "consistency_score": 0.75,
                    "validity_score": 0.80
                },
                "industry": "financial_services",
                "target_score": 0.90,
                "auto_export": True,
                "export_formats": ["json", "html"]
            }
        }


class BenchmarkRequest(BaseModel):
    """Request model for industry benchmarking"""
    
    ai_readiness_score: AIReadinessScore = Field(..., description="AI readiness assessment results")
    industry: IndustryStandard = Field(..., description="Industry standard for comparison")
    include_detailed_analysis: bool = Field(
        default=True,
        description="Include detailed dimension-wise analysis"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "ai_readiness_score": {
                    "overall_score": 0.75,
                    "data_quality_score": 0.80,
                    "feature_quality_score": 0.70,
                    "bias_score": 0.75,
                    "compliance_score": 0.85,
                    "scalability_score": 0.70
                },
                "industry": "healthcare",
                "include_detailed_analysis": True
            }
        }


class ImprovementRoadmapRequest(BaseModel):
    """Request model for generating improvement roadmap"""
    
    dataset_id: str = Field(..., description="Unique identifier for the dataset")
    current_score: AIReadinessScore = Field(..., description="Current AI readiness assessment")
    quality_report: QualityReport = Field(..., description="Data quality assessment results")
    bias_report: Optional[BiasReport] = Field(None, description="Bias analysis results (optional)")
    target_score: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Target AI readiness score to achieve"
    )
    priority_focus: Optional[List[str]] = Field(
        None,
        description="Priority areas to focus on (data_quality, feature_engineering, bias_mitigation, compliance)"
    )
    timeline_constraint: Optional[str] = Field(
        None,
        description="Timeline constraint (e.g., '3 months', '6 weeks')"
    )
    resource_constraints: Optional[Dict[str, Any]] = Field(
        None,
        description="Resource constraints and availability"
    )
    
    @validator('target_score')
    def validate_target_score(cls, v, values):
        current_score = values.get('current_score')
        if current_score and v <= current_score.overall_score:
            raise ValueError('Target score must be higher than current score')
        return v
    
    @validator('priority_focus')
    def validate_priority_focus(cls, v):
        if v:
            valid_areas = ["data_quality", "feature_engineering", "bias_mitigation", "compliance"]
            for area in v:
                if area not in valid_areas:
                    raise ValueError(f'Invalid priority area: {area}. Must be one of {valid_areas}')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "dataset_id": "customer_data_2024",
                "current_score": {
                    "overall_score": 0.75,
                    "data_quality_score": 0.80,
                    "feature_quality_score": 0.70,
                    "bias_score": 0.75,
                    "compliance_score": 0.85,
                    "scalability_score": 0.70
                },
                "quality_report": {
                    "dataset_id": "customer_data_2024",
                    "overall_score": 0.80
                },
                "target_score": 0.90,
                "priority_focus": ["feature_engineering", "bias_mitigation"],
                "timeline_constraint": "3 months",
                "resource_constraints": {
                    "team_size": 3,
                    "budget": 50000,
                    "available_tools": ["python", "spark", "mlflow"]
                }
            }
        }


class ExportReportRequest(BaseModel):
    """Request model for exporting AI readiness report"""
    
    report: AIReadinessReport = Field(..., description="AI readiness report to export")
    format: ReportFormat = Field(..., description="Export format")
    return_content: bool = Field(
        default=True,
        description="Return content directly or save to file"
    )
    include_metadata: bool = Field(
        default=True,
        description="Include metadata in exported report"
    )
    custom_styling: Optional[Dict[str, Any]] = Field(
        None,
        description="Custom styling options for HTML/PDF exports"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "report": {
                    "dataset_id": "customer_data_2024",
                    "overall_score": 0.75,
                    "competitive_position": "Above Average"
                },
                "format": "html",
                "return_content": False,
                "include_metadata": True,
                "custom_styling": {
                    "theme": "professional",
                    "color_scheme": "blue"
                }
            }
        }


class DatasetAnalysisRequest(BaseModel):
    """Request model for dataset analysis"""
    
    dataset_id: str = Field(..., description="Unique identifier for the dataset")
    dataset_path: Optional[str] = Field(None, description="Path to dataset file")
    dataset_url: Optional[str] = Field(None, description="URL to dataset")
    analysis_type: str = Field(
        default="comprehensive",
        description="Type of analysis (quick, comprehensive, custom)"
    )
    custom_parameters: Optional[Dict[str, Any]] = Field(
        None,
        description="Custom analysis parameters"
    )
    
    @validator('analysis_type')
    def validate_analysis_type(cls, v):
        valid_types = ["quick", "comprehensive", "custom"]
        if v not in valid_types:
            raise ValueError(f'Invalid analysis type: {v}. Must be one of {valid_types}')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "dataset_id": "customer_data_2024",
                "dataset_path": "/data/customer_data.csv",
                "analysis_type": "comprehensive",
                "custom_parameters": {
                    "target_column": "churn",
                    "protected_attributes": ["gender", "age_group"],
                    "quality_thresholds": {
                        "completeness": 0.95,
                        "accuracy": 0.90
                    }
                }
            }
        }


class BatchReportRequest(BaseModel):
    """Request model for batch report generation"""
    
    datasets: List[str] = Field(..., description="List of dataset IDs to process")
    report_type: str = Field(
        default="standard",
        description="Type of report to generate"
    )
    industry: IndustryStandard = Field(
        default=IndustryStandard.GENERAL,
        description="Industry standard for benchmarking"
    )
    export_format: ReportFormat = Field(
        default=ReportFormat.JSON,
        description="Export format for all reports"
    )
    parallel_processing: bool = Field(
        default=True,
        description="Enable parallel processing for multiple datasets"
    )
    notification_email: Optional[str] = Field(
        None,
        description="Email address for completion notification"
    )
    
    @validator('datasets')
    def validate_datasets(cls, v):
        if not v:
            raise ValueError('At least one dataset ID must be provided')
        if len(v) > 100:
            raise ValueError('Maximum 100 datasets allowed per batch request')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "datasets": ["dataset_001", "dataset_002", "dataset_003"],
                "report_type": "standard",
                "industry": "financial_services",
                "export_format": "json",
                "parallel_processing": True,
                "notification_email": "analyst@company.com"
            }
        }


class ReportComparisonRequest(BaseModel):
    """Request model for comparing multiple reports"""
    
    report_ids: List[str] = Field(..., description="List of report IDs to compare")
    comparison_dimensions: List[str] = Field(
        default=["overall_score", "data_quality_score", "bias_score"],
        description="Dimensions to compare across reports"
    )
    include_trends: bool = Field(
        default=True,
        description="Include trend analysis if reports are from different time periods"
    )
    benchmark_against: Optional[IndustryStandard] = Field(
        None,
        description="Industry standard to benchmark all reports against"
    )
    
    @validator('report_ids')
    def validate_report_ids(cls, v):
        if len(v) < 2:
            raise ValueError('At least two report IDs required for comparison')
        if len(v) > 10:
            raise ValueError('Maximum 10 reports allowed per comparison')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "report_ids": ["report_001", "report_002", "report_003"],
                "comparison_dimensions": ["overall_score", "data_quality_score", "bias_score"],
                "include_trends": True,
                "benchmark_against": "financial_services"
            }
        }


class CustomReportRequest(BaseModel):
    """Request model for custom report generation"""
    
    dataset_id: str = Field(..., description="Unique identifier for the dataset")
    template_id: Optional[str] = Field(None, description="Custom template ID")
    sections: List[str] = Field(
        default=["executive_summary", "scores", "recommendations"],
        description="Report sections to include"
    )
    custom_metrics: Optional[Dict[str, Any]] = Field(
        None,
        description="Custom metrics to calculate and include"
    )
    branding: Optional[Dict[str, Any]] = Field(
        None,
        description="Custom branding and styling options"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "dataset_id": "customer_data_2024",
                "template_id": "executive_template",
                "sections": ["executive_summary", "scores", "benchmarking", "roadmap"],
                "custom_metrics": {
                    "business_impact_score": 0.85,
                    "roi_potential": 0.75
                },
                "branding": {
                    "company_logo": "logo_url",
                    "color_scheme": "corporate_blue",
                    "font_family": "Arial"
                }
            }
        }