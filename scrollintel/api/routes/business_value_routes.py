"""
Business Value Tracking API Routes

This module provides REST API endpoints for business value tracking,
ROI calculations, cost savings analysis, productivity measurement,
and competitive advantage assessment.
"""

from datetime import datetime, timedelta
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, Path
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from sqlalchemy import and_, desc

from ...models.database_utils import get_db
from ...engines.business_value_engine import BusinessValueEngine
from ...models.business_value_models import (
    BusinessValueMetric, ROICalculation, CostSavingsRecord,
    ProductivityRecord, CompetitiveAdvantageAssessment,
    BusinessValueMetricCreate, BusinessValueMetricResponse,
    ROICalculationCreate, ROICalculationResponse,
    CostSavingsCreate, CostSavingsResponse,
    ProductivityCreate, ProductivityResponse,
    CompetitiveAdvantageCreate, CompetitiveAdvantageResponse,
    BusinessValueSummary, BusinessValueDashboard,
    MetricType, BusinessUnit, CompetitiveAdvantageType
)

router = APIRouter(prefix="/api/v1/business-value", tags=["Business Value Tracking"])
business_value_engine = BusinessValueEngine()

@router.post("/metrics", response_model=BusinessValueMetricResponse)
async def create_business_value_metric(
    metric_data: BusinessValueMetricCreate,
    db: Session = Depends(get_database_session)
):
    """
    Create a new business value metric for tracking ROI, cost savings,
    productivity gains, or competitive advantages.
    """
    try:
        metric = await business_value_engine.create_business_value_metric(metric_data, db)
        
        return BusinessValueMetricResponse(
            id=metric.id,
            metric_type=metric.metric_type,
            business_unit=metric.business_unit,
            metric_name=metric.metric_name,
            baseline_value=metric.baseline_value,
            current_value=metric.current_value,
            target_value=metric.target_value,
            measurement_period_start=metric.measurement_period_start,
            measurement_period_end=metric.measurement_period_end,
            currency=metric.currency,
            created_at=metric.created_at,
            updated_at=metric.updated_at,
            extra_data=metric.extra_data
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/metrics", response_model=List[BusinessValueMetricResponse])
async def get_business_value_metrics(
    metric_type: Optional[MetricType] = Query(None, description="Filter by metric type"),
    business_unit: Optional[BusinessUnit] = Query(None, description="Filter by business unit"),
    start_date: Optional[datetime] = Query(None, description="Filter by start date"),
    end_date: Optional[datetime] = Query(None, description="Filter by end date"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of results"),
    offset: int = Query(0, ge=0, description="Number of results to skip"),
    db: Session = Depends(get_database_session)
):
    """
    Retrieve business value metrics with optional filtering.
    """
    try:
        query = db.query(BusinessValueMetric)
        
        # Apply filters
        if metric_type:
            query = query.filter(BusinessValueMetric.metric_type == metric_type.value)
        
        if business_unit:
            query = query.filter(BusinessValueMetric.business_unit == business_unit.value)
        
        if start_date:
            query = query.filter(BusinessValueMetric.measurement_period_start >= start_date)
        
        if end_date:
            query = query.filter(BusinessValueMetric.measurement_period_end <= end_date)
        
        # Apply pagination and ordering
        metrics = query.order_by(desc(BusinessValueMetric.created_at)).offset(offset).limit(limit).all()
        
        return [
            BusinessValueMetricResponse(
                id=metric.id,
                metric_type=metric.metric_type,
                business_unit=metric.business_unit,
                metric_name=metric.metric_name,
                baseline_value=metric.baseline_value,
                current_value=metric.current_value,
                target_value=metric.target_value,
                measurement_period_start=metric.measurement_period_start,
                measurement_period_end=metric.measurement_period_end,
                currency=metric.currency,
                created_at=metric.created_at,
                updated_at=metric.updated_at,
                extra_data=metric.extra_data
            )
            for metric in metrics
        ]
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/metrics/{metric_id}", response_model=BusinessValueMetricResponse)
async def get_business_value_metric(
    metric_id: int = Path(..., description="Business value metric ID"),
    db: Session = Depends(get_database_session)
):
    """
    Retrieve a specific business value metric by ID.
    """
    try:
        metric = db.query(BusinessValueMetric).filter(BusinessValueMetric.id == metric_id).first()
        
        if not metric:
            raise HTTPException(status_code=404, detail="Business value metric not found")
        
        return BusinessValueMetricResponse(
            id=metric.id,
            metric_type=metric.metric_type,
            business_unit=metric.business_unit,
            metric_name=metric.metric_name,
            baseline_value=metric.baseline_value,
            current_value=metric.current_value,
            target_value=metric.target_value,
            measurement_period_start=metric.measurement_period_start,
            measurement_period_end=metric.measurement_period_end,
            currency=metric.currency,
            created_at=metric.created_at,
            updated_at=metric.updated_at,
            extra_data=metric.extra_data
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ROI Calculation endpoints
@router.post("/roi-calculations", response_model=ROICalculationResponse)
async def create_roi_calculation(
    roi_data: ROICalculationCreate,
    db: Session = Depends(get_database_session)
):
    """Create a new ROI calculation with comprehensive financial metrics"""
    try:
        roi_calc = await business_value_engine.create_roi_calculation(roi_data, db)
        
        return ROICalculationResponse(
            id=roi_calc.id,
            metric_id=roi_calc.metric_id,
            investment_amount=roi_calc.investment_amount,
            return_amount=roi_calc.return_amount,
            roi_percentage=roi_calc.roi_percentage,
            payback_period_months=roi_calc.payback_period_months,
            npv=roi_calc.npv,
            irr=roi_calc.irr,
            calculation_date=roi_calc.calculation_date,
            calculation_method=roi_calc.calculation_method,
            confidence_level=roi_calc.confidence_level,
            assumptions=roi_calc.assumptions
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Cost Savings endpoints
@router.post("/cost-savings", response_model=CostSavingsResponse)
async def create_cost_savings_record(
    savings_data: CostSavingsCreate,
    db: Session = Depends(get_database_session)
):
    """Create a new cost savings record with automated financial impact analysis"""
    try:
        savings_record = await business_value_engine.create_cost_savings_record(savings_data, db)
        
        return CostSavingsResponse(
            id=savings_record.id,
            metric_id=savings_record.metric_id,
            savings_category=savings_record.savings_category,
            annual_savings=savings_record.annual_savings,
            monthly_savings=savings_record.monthly_savings,
            cost_before=savings_record.cost_before,
            cost_after=savings_record.cost_after,
            savings_source=savings_record.savings_source,
            verification_method=savings_record.verification_method,
            verified=savings_record.verified,
            verification_date=savings_record.verification_date,
            record_date=savings_record.record_date
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Productivity endpoints
@router.post("/productivity", response_model=ProductivityResponse)
async def create_productivity_record(
    productivity_data: ProductivityCreate,
    db: Session = Depends(get_database_session)
):
    """Create a new productivity measurement record for quantifying efficiency gains"""
    try:
        productivity_record = await business_value_engine.create_productivity_record(productivity_data, db)
        
        return ProductivityResponse(
            id=productivity_record.id,
            metric_id=productivity_record.metric_id,
            task_category=productivity_record.task_category,
            baseline_time_hours=productivity_record.baseline_time_hours,
            current_time_hours=productivity_record.current_time_hours,
            efficiency_gain_percentage=productivity_record.efficiency_gain_percentage,
            tasks_completed_baseline=productivity_record.tasks_completed_baseline,
            tasks_completed_current=productivity_record.tasks_completed_current,
            quality_score_baseline=productivity_record.quality_score_baseline,
            quality_score_current=productivity_record.quality_score_current,
            measurement_date=productivity_record.measurement_date
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Competitive Advantage endpoints
@router.post("/competitive-advantage", response_model=CompetitiveAdvantageResponse)
async def create_competitive_advantage_assessment(
    advantage_data: CompetitiveAdvantageCreate,
    db: Session = Depends(get_database_session)
):
    """Create a new competitive advantage assessment for tracking market position"""
    try:
        assessment = await business_value_engine.create_competitive_advantage_assessment(advantage_data, db)
        
        return CompetitiveAdvantageResponse(
            id=assessment.id,
            advantage_type=assessment.advantage_type,
            competitor_name=assessment.competitor_name,
            our_score=assessment.our_score,
            competitor_score=assessment.competitor_score,
            advantage_gap=assessment.advantage_gap,
            market_impact=assessment.market_impact,
            sustainability_months=assessment.sustainability_months,
            assessment_date=assessment.assessment_date,
            assessor=assessment.assessor,
            evidence=assessment.evidence,
            action_items=assessment.action_items
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Summary and Dashboard endpoints
@router.get("/summary", response_model=BusinessValueSummary)
async def get_business_value_summary(
    start_date: Optional[datetime] = Query(None, description="Summary start date"),
    end_date: Optional[datetime] = Query(None, description="Summary end date"),
    business_unit: Optional[BusinessUnit] = Query(None, description="Filter by business unit"),
    db: Session = Depends(get_database_session)
):
    """Generate comprehensive business value summary report"""
    try:
        # Default to last 90 days if no dates provided
        if not end_date:
            end_date = datetime.utcnow()
        if not start_date:
            start_date = end_date - timedelta(days=90)
        
        business_unit_str = business_unit.value if business_unit else None
        
        summary = await business_value_engine.generate_business_value_summary(
            start_date, end_date, business_unit_str, db
        )
        
        return summary
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/dashboard", response_model=BusinessValueDashboard)
async def get_business_value_dashboard(
    business_unit: Optional[BusinessUnit] = Query(None, description="Filter by business unit"),
    db: Session = Depends(get_database_session)
):
    """Get real-time business value dashboard with key metrics and insights"""
    try:
        business_unit_str = business_unit.value if business_unit else None
        
        dashboard = await business_value_engine.get_business_value_dashboard(
            business_unit_str, db
        )
        
        return dashboard
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Calculation utilities
@router.post("/calculate-roi")
async def calculate_roi_metrics(
    investment: float = Query(..., description="Investment amount"),
    returns: float = Query(..., description="Return amount"),
    time_period_months: int = Query(12, description="Time period in months"),
    discount_rate: Optional[float] = Query(None, description="Discount rate for NPV calculation")
):
    """Calculate ROI metrics including ROI percentage, NPV, IRR, and payback period"""
    try:
        from decimal import Decimal
        
        roi_metrics = await business_value_engine.calculate_roi(
            Decimal(str(investment)),
            Decimal(str(returns)),
            time_period_months,
            Decimal(str(discount_rate)) if discount_rate else None
        )
        
        return JSONResponse(content={
            "roi_percentage": float(roi_metrics['roi_percentage']),
            "npv": float(roi_metrics['npv']) if roi_metrics['npv'] else None,
            "irr": float(roi_metrics['irr']) if roi_metrics['irr'] else None,
            "payback_period_months": roi_metrics['payback_period_months']
        })
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))