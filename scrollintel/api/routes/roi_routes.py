"""
ROI Calculator API routes for the Advanced Analytics Dashboard System.
"""
from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field

from ...engines.roi_calculator import ROICalculator, ROICalculationResult, CostSummary, BenefitSummary
from ...models.roi_models import ROIAnalysis, CostTracking, BenefitTracking, ROIReport
from ...security.auth import get_current_user
from ...models.database import User

router = APIRouter(prefix="/api/roi", tags=["ROI Calculator"])


# Pydantic models for request/response
class CreateROIAnalysisRequest(BaseModel):
    project_id: str = Field(..., description="Unique project identifier")
    project_name: str = Field(..., description="Project name")
    project_description: Optional[str] = Field(None, description="Project description")
    project_start_date: datetime = Field(..., description="Project start date")
    analysis_period_months: int = Field(12, description="Analysis period in months")


class TrackCostRequest(BaseModel):
    project_id: str = Field(..., description="Project identifier")
    cost_category: str = Field(..., description="Cost category")
    description: str = Field(..., description="Cost description")
    amount: float = Field(..., description="Cost amount")
    cost_date: Optional[datetime] = Field(None, description="Cost date")
    vendor: Optional[str] = Field(None, description="Vendor name")
    is_recurring: bool = Field(False, description="Is recurring cost")
    recurrence_frequency: Optional[str] = Field(None, description="Recurrence frequency")


class TrackBenefitRequest(BaseModel):
    project_id: str = Field(..., description="Project identifier")
    benefit_category: str = Field(..., description="Benefit category")
    description: str = Field(..., description="Benefit description")
    quantified_value: float = Field(..., description="Quantified benefit value")
    measurement_method: str = Field(..., description="Measurement method")
    baseline_value: Optional[float] = Field(None, description="Baseline value")
    current_value: Optional[float] = Field(None, description="Current value")
    is_realized: bool = Field(False, description="Is benefit realized")


class CollectCloudCostsRequest(BaseModel):
    project_id: str = Field(..., description="Project identifier")
    provider: str = Field(..., description="Cloud provider (aws, azure, gcp)")
    account_id: str = Field(..., description="Cloud account ID")
    start_date: datetime = Field(..., description="Start date for cost collection")
    end_date: datetime = Field(..., description="End date for cost collection")
    cost_allocation_tags: Optional[Dict[str, str]] = Field(None, description="Cost allocation tags")


class ProductivityMetricRequest(BaseModel):
    project_id: str = Field(..., description="Project identifier")
    metric_name: str = Field(..., description="Metric name")
    baseline_value: float = Field(..., description="Baseline value")
    current_value: float = Field(..., description="Current value")
    measurement_unit: str = Field(..., description="Measurement unit")
    process_or_task: str = Field(..., description="Process or task being measured")
    measurement_method: str = Field("automated_measurement", description="Measurement method")


class ROIAnalysisResponse(BaseModel):
    id: str
    project_id: str
    project_name: str
    project_description: Optional[str]
    project_status: str
    total_investment: float
    total_benefits: float
    roi_percentage: Optional[float]
    net_present_value: Optional[float]
    internal_rate_of_return: Optional[float]
    payback_period_months: Optional[int]
    break_even_date: Optional[datetime]
    confidence_level: float
    created_at: datetime
    updated_at: datetime


class ROICalculationResponse(BaseModel):
    roi_percentage: float
    net_present_value: float
    internal_rate_of_return: float
    payback_period_months: int
    break_even_date: Optional[datetime]
    total_investment: float
    total_benefits: float
    confidence_level: float
    calculation_date: datetime


class CostSummaryResponse(BaseModel):
    total_costs: float
    direct_costs: float
    indirect_costs: float
    operational_costs: float
    infrastructure_costs: float
    personnel_costs: float
    cost_breakdown: Dict[str, float]
    monthly_recurring_costs: float


class BenefitSummaryResponse(BaseModel):
    total_benefits: float
    realized_benefits: float
    projected_benefits: float
    cost_savings: float
    productivity_gains: float
    revenue_increases: float
    benefit_breakdown: Dict[str, float]
    realization_percentage: float


# Initialize ROI Calculator
roi_calculator = ROICalculator()


@router.post("/analysis", response_model=ROIAnalysisResponse)
async def create_roi_analysis(
    request: CreateROIAnalysisRequest,
    current_user: User = Depends(get_current_user)
):
    """Create a new ROI analysis for a project."""
    try:
        roi_analysis = roi_calculator.create_roi_analysis(
            project_id=request.project_id,
            project_name=request.project_name,
            project_description=request.project_description,
            project_start_date=request.project_start_date,
            analysis_period_months=request.analysis_period_months,
            created_by=current_user.username
        )
        
        return ROIAnalysisResponse(
            id=roi_analysis.id,
            project_id=roi_analysis.project_id,
            project_name=roi_analysis.project_name,
            project_description=roi_analysis.project_description,
            project_status=roi_analysis.project_status,
            total_investment=roi_analysis.total_investment,
            total_benefits=roi_analysis.total_benefits,
            roi_percentage=roi_analysis.roi_percentage,
            net_present_value=roi_analysis.net_present_value,
            internal_rate_of_return=roi_analysis.internal_rate_of_return,
            payback_period_months=roi_analysis.payback_period_months,
            break_even_date=roi_analysis.break_even_date,
            confidence_level=roi_analysis.confidence_level,
            created_at=roi_analysis.created_at,
            updated_at=roi_analysis.updated_at
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/analysis/{project_id}", response_model=ROIAnalysisResponse)
async def get_roi_analysis(
    project_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get ROI analysis for a project."""
    try:
        from ...models.database_utils import get_sync_db
        
        with get_sync_db() as session:
            roi_analysis = session.query(ROIAnalysis).filter_by(project_id=project_id).first()
            if not roi_analysis:
                raise HTTPException(status_code=404, detail="ROI analysis not found")
            
            return ROIAnalysisResponse(
                id=roi_analysis.id,
                project_id=roi_analysis.project_id,
                project_name=roi_analysis.project_name,
                project_description=roi_analysis.project_description,
                project_status=roi_analysis.project_status,
                total_investment=roi_analysis.total_investment,
                total_benefits=roi_analysis.total_benefits,
                roi_percentage=roi_analysis.roi_percentage,
                net_present_value=roi_analysis.net_present_value,
                internal_rate_of_return=roi_analysis.internal_rate_of_return,
                payback_period_months=roi_analysis.payback_period_months,
                break_even_date=roi_analysis.break_even_date,
                confidence_level=roi_analysis.confidence_level,
                created_at=roi_analysis.created_at,
                updated_at=roi_analysis.updated_at
            )
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/costs")
async def track_project_costs(
    request: TrackCostRequest,
    current_user: User = Depends(get_current_user)
):
    """Track costs for a project."""
    try:
        cost_item = roi_calculator.track_project_costs(
            project_id=request.project_id,
            cost_category=request.cost_category,
            description=request.description,
            amount=request.amount,
            cost_date=request.cost_date,
            vendor=request.vendor,
            is_recurring=request.is_recurring,
            recurrence_frequency=request.recurrence_frequency
        )
        
        return {
            "id": cost_item.id,
            "project_id": request.project_id,
            "amount": cost_item.amount,
            "category": cost_item.cost_category,
            "description": cost_item.description,
            "created_at": cost_item.created_at
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/benefits")
async def track_project_benefits(
    request: TrackBenefitRequest,
    current_user: User = Depends(get_current_user)
):
    """Track benefits for a project."""
    try:
        benefit_item = roi_calculator.track_project_benefits(
            project_id=request.project_id,
            benefit_category=request.benefit_category,
            description=request.description,
            quantified_value=request.quantified_value,
            measurement_method=request.measurement_method,
            baseline_value=request.baseline_value,
            current_value=request.current_value,
            is_realized=request.is_realized
        )
        
        return {
            "id": benefit_item.id,
            "project_id": request.project_id,
            "quantified_value": benefit_item.quantified_value,
            "category": benefit_item.benefit_category,
            "description": benefit_item.description,
            "is_realized": benefit_item.is_realized,
            "created_at": benefit_item.created_at
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/cloud-costs")
async def collect_cloud_costs(
    request: CollectCloudCostsRequest,
    current_user: User = Depends(get_current_user)
):
    """Automatically collect cloud costs for a project."""
    try:
        collected_costs = roi_calculator.collect_cloud_costs(
            project_id=request.project_id,
            provider=request.provider,
            account_id=request.account_id,
            start_date=request.start_date,
            end_date=request.end_date,
            cost_allocation_tags=request.cost_allocation_tags
        )
        
        return {
            "project_id": request.project_id,
            "provider": request.provider,
            "collected_items": len(collected_costs),
            "total_cost": sum(cost.cost_amount for cost in collected_costs),
            "collection_date": datetime.utcnow()
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/productivity-metrics")
async def measure_productivity_gains(
    request: ProductivityMetricRequest,
    current_user: User = Depends(get_current_user)
):
    """Measure and track productivity gains."""
    try:
        productivity_metric = roi_calculator.measure_productivity_gains(
            project_id=request.project_id,
            metric_name=request.metric_name,
            baseline_value=request.baseline_value,
            current_value=request.current_value,
            measurement_unit=request.measurement_unit,
            process_or_task=request.process_or_task,
            measurement_method=request.measurement_method
        )
        
        improvement_percentage = ((request.current_value - request.baseline_value) / request.baseline_value) * 100
        
        return {
            "id": productivity_metric.id,
            "project_id": request.project_id,
            "metric_name": request.metric_name,
            "improvement_percentage": improvement_percentage,
            "baseline_value": request.baseline_value,
            "current_value": request.current_value,
            "measurement_unit": request.measurement_unit,
            "created_at": productivity_metric.created_at
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/calculate/{project_id}", response_model=ROICalculationResponse)
async def calculate_roi(
    project_id: str,
    current_user: User = Depends(get_current_user)
):
    """Calculate comprehensive ROI metrics for a project."""
    try:
        roi_result = roi_calculator.calculate_roi(project_id)
        
        return ROICalculationResponse(
            roi_percentage=roi_result.roi_percentage,
            net_present_value=roi_result.net_present_value,
            internal_rate_of_return=roi_result.internal_rate_of_return,
            payback_period_months=roi_result.payback_period_months,
            break_even_date=roi_result.break_even_date,
            total_investment=roi_result.total_investment,
            total_benefits=roi_result.total_benefits,
            confidence_level=roi_result.confidence_level,
            calculation_date=roi_result.calculation_date
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/costs/{project_id}", response_model=CostSummaryResponse)
async def get_cost_summary(
    project_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get detailed cost summary for a project."""
    try:
        cost_summary = roi_calculator.get_cost_summary(project_id)
        
        return CostSummaryResponse(
            total_costs=cost_summary.total_costs,
            direct_costs=cost_summary.direct_costs,
            indirect_costs=cost_summary.indirect_costs,
            operational_costs=cost_summary.operational_costs,
            infrastructure_costs=cost_summary.infrastructure_costs,
            personnel_costs=cost_summary.personnel_costs,
            cost_breakdown=cost_summary.cost_breakdown,
            monthly_recurring_costs=cost_summary.monthly_recurring_costs
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/benefits/{project_id}", response_model=BenefitSummaryResponse)
async def get_benefit_summary(
    project_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get detailed benefit summary for a project."""
    try:
        benefit_summary = roi_calculator.get_benefit_summary(project_id)
        
        return BenefitSummaryResponse(
            total_benefits=benefit_summary.total_benefits,
            realized_benefits=benefit_summary.realized_benefits,
            projected_benefits=benefit_summary.projected_benefits,
            cost_savings=benefit_summary.cost_savings,
            productivity_gains=benefit_summary.productivity_gains,
            revenue_increases=benefit_summary.revenue_increases,
            benefit_breakdown=benefit_summary.benefit_breakdown,
            realization_percentage=benefit_summary.realization_percentage
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/reports/{project_id}")
async def generate_roi_report(
    project_id: str,
    report_type: str = Query("detailed", description="Report type"),
    report_format: str = Query("json", description="Report format"),
    current_user: User = Depends(get_current_user)
):
    """Generate comprehensive ROI report with visualizations."""
    try:
        roi_report = roi_calculator.generate_roi_report(
            project_id=project_id,
            report_type=report_type,
            report_format=report_format
        )
        
        return {
            "id": roi_report.id,
            "project_id": project_id,
            "report_name": roi_report.report_name,
            "report_type": roi_report.report_type,
            "report_format": roi_report.report_format,
            "report_data": roi_report.report_data,
            "visualizations": roi_report.visualizations,
            "executive_summary": roi_report.executive_summary,
            "key_findings": roi_report.key_findings,
            "recommendations": roi_report.recommendations,
            "created_at": roi_report.created_at
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/reports/{project_id}")
async def get_roi_reports(
    project_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get all ROI reports for a project."""
    try:
        from ...models.database_utils import get_sync_db
        
        with get_sync_db() as session:
            roi_analysis = session.query(ROIAnalysis).filter_by(project_id=project_id).first()
            if not roi_analysis:
                raise HTTPException(status_code=404, detail="ROI analysis not found")
            
            reports = session.query(ROIReport).filter_by(roi_analysis_id=roi_analysis.id).all()
            
            return [
                {
                    "id": report.id,
                    "report_name": report.report_name,
                    "report_type": report.report_type,
                    "report_format": report.report_format,
                    "created_at": report.created_at,
                    "generation_status": report.generation_status
                }
                for report in reports
            ]
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@r
outer.get("/breakdown/{project_id}")
async def get_detailed_roi_breakdown(
    project_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get detailed ROI breakdown with comprehensive analysis."""
    try:
        breakdown = roi_calculator.generate_detailed_roi_breakdown(project_id)
        return breakdown
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/dashboard/{project_id}")
async def get_roi_dashboard_data(
    project_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get comprehensive ROI dashboard data for visualization."""
    try:
        dashboard_data = roi_calculator.generate_roi_dashboard_data(project_id)
        return dashboard_data
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/efficiency-gains")
async def measure_efficiency_gains(
    request: Dict[str, Any],
    current_user: User = Depends(get_current_user)
):
    """Measure detailed efficiency gains with comprehensive cost impact analysis."""
    try:
        efficiency_metric = roi_calculator.measure_efficiency_gains(
            project_id=request['project_id'],
            process_name=request['process_name'],
            time_before_hours=request['time_before_hours'],
            time_after_hours=request['time_after_hours'],
            frequency_per_month=request.get('frequency_per_month', 1.0),
            hourly_rate=request.get('hourly_rate', 50.0),
            error_rate_before=request.get('error_rate_before', 0.0),
            error_rate_after=request.get('error_rate_after', 0.0),
            measurement_method=request.get('measurement_method', 'time_study')
        )
        
        return {
            "id": efficiency_metric.id,
            "project_id": request['project_id'],
            "process_name": efficiency_metric.process_name,
            "time_saved_hours": efficiency_metric.time_saved_hours,
            "time_saved_percentage": efficiency_metric.time_saved_percentage,
            "monthly_savings": efficiency_metric.monthly_savings,
            "annual_savings": efficiency_metric.annual_savings,
            "quality_improvement_percentage": efficiency_metric.quality_improvement_percentage,
            "created_at": efficiency_metric.created_at
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/projects")
async def list_roi_projects(
    current_user: User = Depends(get_current_user)
):
    """List all projects with ROI analysis."""
    try:
        from ...models.database_utils import get_sync_db
        
        with get_sync_db() as session:
            projects = session.query(ROIAnalysis).all()
            
            return [
                {
                    "project_id": project.project_id,
                    "project_name": project.project_name,
                    "project_status": project.project_status,
                    "roi_percentage": project.roi_percentage,
                    "total_investment": project.total_investment,
                    "total_benefits": project.total_benefits,
                    "payback_period_months": project.payback_period_months,
                    "created_at": project.created_at,
                    "updated_at": project.updated_at
                }
                for project in projects
            ]
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/analysis/{project_id}/status")
async def update_project_status(
    project_id: str,
    status: str,
    current_user: User = Depends(get_current_user)
):
    """Update project status in ROI analysis."""
    try:
        from ...models.database_utils import get_sync_db
        
        with get_sync_db() as session:
            roi_analysis = session.query(ROIAnalysis).filter_by(project_id=project_id).first()
            if not roi_analysis:
                raise HTTPException(status_code=404, detail="ROI analysis not found")
            
            roi_analysis.project_status = status
            roi_analysis.last_updated_by = current_user.username
            roi_analysis.updated_at = datetime.utcnow()
            
            session.commit()
            
            return {
                "project_id": project_id,
                "status": status,
                "updated_at": roi_analysis.updated_at
            }
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))