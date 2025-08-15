"""
Response Models for AI Data Readiness Platform API

This module defines Pydantic models for API response serialization and validation.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Union
from datetime import datetime
from enum import Enum

from ...engines.ai_readiness_reporting_engine import (
    IndustryStandard,
    AIReadinessReport,
    ImprovementAction
)

# Define missing classes
class ReportFormat(str, Enum):
    JSON = "json"
    HTML = "html"
    PDF = "pdf"
    CSV = "csv"

class ImprovementRoadmap(BaseModel):
    dataset_id: str
    current_score: float
    target_score: float
    estimated_timeline: str
    phases: List[Dict[str, Any]]

class BenchmarkMetrics(BaseModel):
    industry: str
    overall_percentile: float
    competitive_position: str
    dimension_comparisons: Dict[str, Any]


class BaseResponse(BaseModel):
    """Base response model with common fields"""
    
    success: bool = Field(..., description="Whether the operation was successful")
    message: str = Field(..., description="Human-readable message about the operation")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")


class ReportResponse(BaseResponse):
    """Response model for comprehensive AI readiness report generation"""
    
    report: AIReadinessReport = Field(..., description="Generated AI readiness report")
    generation_time: datetime = Field(..., description="When the report was generated")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about report generation"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "message": "Report generated successfully",
                "timestamp": "2024-01-15T10:30:00Z",
                "report": {
                    "dataset_id": "customer_data_2024",
                    "overall_score": 0.75,
                    "competitive_position": "Above Average",
                    "percentile_ranking": 78.5
                },
                "generation_time": "2024-01-15T10:30:00Z",
                "metadata": {
                    "industry": "financial_services",
                    "target_score": 0.85,
                    "processing_time_ms": 1250
                }
            }
        }


class BenchmarkComparisonResponse(BaseResponse):
    """Response model for benchmark comparison across industries"""
    
    dataset_id: str = Field(..., description="Dataset identifier")
    comparisons: Dict[str, Dict[str, float]] = Field(..., description="Benchmark comparisons by industry")
    best_fit_industry: str = Field(..., description="Industry with best fit for the dataset")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "message": "Benchmark comparison completed successfully",
                "timestamp": "2024-01-15T10:30:00Z",
                "dataset_id": "customer_data_2024",
                "comparisons": {
                    "financial_services": {
                        "data_quality_gap": -0.15,
                        "feature_quality_gap": -0.20,
                        "bias_score_gap": -0.10,
                        "compliance_gap": -0.13,
                        "overall_gap": -0.17
                    },
                    "retail": {
                        "data_quality_gap": -0.05,
                        "feature_quality_gap": -0.10,
                        "bias_score_gap": 0.00,
                        "compliance_gap": -0.10,
                        "overall_gap": -0.06
                    }
                },
                "best_fit_industry": "retail"
            }
        }


class BenchmarkResponse(BaseResponse):
    """Response model for industry benchmarking"""
    
    benchmark_result: Dict[str, Any] = Field(..., description="Benchmarking results")
    industry: IndustryStandard = Field(..., description="Industry used for benchmarking")
    benchmark_date: datetime = Field(..., description="When benchmarking was performed")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "message": "Benchmarking completed successfully",
                "timestamp": "2024-01-15T10:30:00Z",
                "benchmark_result": {
                    "industry": "financial_services",
                    "overall_percentile": 78.5,
                    "competitive_position": "Above Average",
                    "dimension_comparisons": {
                        "data_quality": {
                            "score": 0.80,
                            "industry_threshold": 0.95,
                            "performance": "below",
                            "gap": 0.15
                        }
                    }
                },
                "industry": "financial_services",
                "benchmark_date": "2024-01-15T10:30:00Z"
            }
        }


class RoadmapResponse(BaseResponse):
    """Response model for improvement roadmap generation"""
    
    roadmap: ImprovementRoadmap = Field(..., description="Generated improvement roadmap")
    generated_at: datetime = Field(..., description="When the roadmap was generated")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about roadmap generation"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "message": "Improvement roadmap generated successfully",
                "timestamp": "2024-01-15T10:30:00Z",
                "roadmap": {
                    "dataset_id": "customer_data_2024",
                    "current_score": 0.75,
                    "target_score": 0.85,
                    "estimated_timeline": "3 months",
                    "phases": [
                        {
                            "phase": 1,
                            "name": "Critical Improvements",
                            "duration": "2-4 weeks",
                            "expected_impact": 0.08
                        }
                    ]
                },
                "generated_at": "2024-01-15T10:30:00Z",
                "metadata": {
                    "improvement_needed": 0.10,
                    "total_actions": 5,
                    "high_priority_actions": 2
                }
            }
        }


class ExportResponse(BaseResponse):
    """Response model for report export"""
    
    format: ReportFormat = Field(..., description="Export format used")
    content: Optional[str] = Field(None, description="Exported content (if return_content=True)")
    file_path: Optional[str] = Field(None, description="File path (if return_content=False)")
    file_size: Optional[int] = Field(None, description="File size in bytes")
    exported_at: datetime = Field(..., description="When the export was completed")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "message": "Report exported successfully",
                "timestamp": "2024-01-15T10:30:00Z",
                "format": "html",
                "file_path": "/tmp/report_customer_data_2024_20240115_103000.html",
                "file_size": 15420,
                "exported_at": "2024-01-15T10:30:00Z"
            }
        }


class AnalysisResponse(BaseResponse):
    """Response model for dataset analysis"""
    
    dataset_id: str = Field(..., description="Dataset identifier")
    analysis_results: Dict[str, Any] = Field(..., description="Analysis results")
    quality_score: float = Field(..., description="Overall quality score")
    ai_readiness_score: float = Field(..., description="AI readiness score")
    recommendations: List[str] = Field(default_factory=list, description="Improvement recommendations")
    analysis_duration: float = Field(..., description="Analysis duration in seconds")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "message": "Dataset analysis completed successfully",
                "timestamp": "2024-01-15T10:30:00Z",
                "dataset_id": "customer_data_2024",
                "analysis_results": {
                    "completeness": 0.85,
                    "accuracy": 0.80,
                    "consistency": 0.75,
                    "bias_detected": False
                },
                "quality_score": 0.80,
                "ai_readiness_score": 0.75,
                "recommendations": [
                    "Address missing values in target column",
                    "Improve feature engineering for categorical variables"
                ],
                "analysis_duration": 45.2
            }
        }


class BatchReportResponse(BaseResponse):
    """Response model for batch report generation"""
    
    batch_id: str = Field(..., description="Unique batch processing identifier")
    total_datasets: int = Field(..., description="Total number of datasets in batch")
    processed_datasets: int = Field(..., description="Number of successfully processed datasets")
    failed_datasets: int = Field(..., description="Number of failed datasets")
    processing_status: str = Field(..., description="Overall processing status")
    results: List[Dict[str, Any]] = Field(default_factory=list, description="Individual dataset results")
    estimated_completion: Optional[datetime] = Field(None, description="Estimated completion time")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "message": "Batch processing initiated successfully",
                "timestamp": "2024-01-15T10:30:00Z",
                "batch_id": "batch_20240115_103000",
                "total_datasets": 5,
                "processed_datasets": 3,
                "failed_datasets": 0,
                "processing_status": "in_progress",
                "results": [
                    {
                        "dataset_id": "dataset_001",
                        "status": "completed",
                        "score": 0.85
                    }
                ],
                "estimated_completion": "2024-01-15T11:00:00Z"
            }
        }


class ComparisonResponse(BaseResponse):
    """Response model for report comparison"""
    
    comparison_id: str = Field(..., description="Unique comparison identifier")
    compared_reports: List[str] = Field(..., description="List of compared report IDs")
    comparison_results: Dict[str, Any] = Field(..., description="Comparison analysis results")
    trends: Optional[Dict[str, Any]] = Field(None, description="Trend analysis if applicable")
    insights: List[str] = Field(default_factory=list, description="Key insights from comparison")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "message": "Report comparison completed successfully",
                "timestamp": "2024-01-15T10:30:00Z",
                "comparison_id": "comp_20240115_103000",
                "compared_reports": ["report_001", "report_002", "report_003"],
                "comparison_results": {
                    "best_performing": "report_002",
                    "average_score": 0.78,
                    "score_variance": 0.12,
                    "dimension_analysis": {
                        "data_quality": {
                            "highest": 0.95,
                            "lowest": 0.65,
                            "average": 0.80
                        }
                    }
                },
                "insights": [
                    "Report 002 shows consistently high performance across all dimensions",
                    "Data quality varies significantly across datasets"
                ]
            }
        }


class HealthResponse(BaseResponse):
    """Response model for health check"""
    
    status: str = Field(..., description="Overall health status")
    components: Dict[str, str] = Field(..., description="Individual component status")
    version: str = Field(..., description="Service version")
    uptime: Optional[float] = Field(None, description="Service uptime in seconds")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "message": "Service is healthy",
                "timestamp": "2024-01-15T10:30:00Z",
                "status": "healthy",
                "components": {
                    "reporting_engine": "healthy",
                    "industry_benchmarks": "loaded",
                    "improvement_templates": "loaded",
                    "database": "connected"
                },
                "version": "1.0.0",
                "uptime": 86400.0
            }
        }


class ErrorResponse(BaseResponse):
    """Response model for error cases"""
    
    error_type: str = Field(..., description="Type of error that occurred")
    error_code: Optional[str] = Field(None, description="Specific error code")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    suggestions: Optional[List[str]] = Field(None, description="Suggestions to resolve the error")
    
    class Config:
        schema_extra = {
            "example": {
                "success": False,
                "message": "Validation error in request data",
                "timestamp": "2024-01-15T10:30:00Z",
                "error_type": "ValidationError",
                "error_code": "INVALID_SCORE_RANGE",
                "details": {
                    "field": "target_score",
                    "value": 1.5,
                    "constraint": "must be between 0.0 and 1.0"
                },
                "suggestions": [
                    "Ensure target_score is between 0.0 and 1.0",
                    "Check the API documentation for valid parameter ranges"
                ]
            }
        }


class MetricsResponse(BaseResponse):
    """Response model for service metrics"""
    
    metrics: Dict[str, Any] = Field(..., description="Service metrics")
    period: str = Field(..., description="Metrics collection period")
    collected_at: datetime = Field(..., description="When metrics were collected")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "message": "Metrics retrieved successfully",
                "timestamp": "2024-01-15T10:30:00Z",
                "metrics": {
                    "reports_generated": 1250,
                    "average_processing_time": 2.3,
                    "success_rate": 0.98,
                    "most_used_industry": "financial_services",
                    "export_formats": {
                        "json": 45,
                        "html": 35,
                        "markdown": 20
                    }
                },
                "period": "last_30_days",
                "collected_at": "2024-01-15T10:30:00Z"
            }
        }


class ValidationResponse(BaseResponse):
    """Response model for data validation"""
    
    validation_results: Dict[str, Any] = Field(..., description="Validation results")
    is_valid: bool = Field(..., description="Whether data passed validation")
    warnings: List[str] = Field(default_factory=list, description="Validation warnings")
    errors: List[str] = Field(default_factory=list, description="Validation errors")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "message": "Data validation completed",
                "timestamp": "2024-01-15T10:30:00Z",
                "validation_results": {
                    "schema_valid": True,
                    "data_types_valid": True,
                    "required_fields_present": True,
                    "value_ranges_valid": False
                },
                "is_valid": False,
                "warnings": [
                    "Some categorical values have low frequency"
                ],
                "errors": [
                    "Target score exceeds maximum allowed value"
                ]
            }
        }


# Union type for all possible response types
APIResponse = Union[
    ReportResponse,
    BenchmarkResponse,
    RoadmapResponse,
    ExportResponse,
    AnalysisResponse,
    BatchReportResponse,
    ComparisonResponse,
    HealthResponse,
    ErrorResponse,
    MetricsResponse,
    ValidationResponse
]