"""
ROI Calculator Engine for comprehensive cost and benefit tracking with automated calculations.
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from dataclasses import dataclass
from enum import Enum

from ..models.roi_models import (
    ROIAnalysis, CostTracking, BenefitTracking, ROIReport,
    CloudCostCollection, ProductivityMetric,
    CostType, BenefitType, ProjectStatus
)
from ..models.database_utils import get_sync_db
from ..connectors.cloud_cost_collector import CloudConnectorManager
from ..core.config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class ROICalculationResult:
    """Result of ROI calculation with detailed metrics."""
    roi_percentage: float
    net_present_value: float
    internal_rate_of_return: float
    payback_period_months: int
    break_even_date: Optional[datetime]
    total_investment: float
    total_benefits: float
    confidence_level: float
    calculation_date: datetime


@dataclass
class CostSummary:
    """Summary of project costs by category."""
    total_costs: float
    direct_costs: float
    indirect_costs: float
    operational_costs: float
    infrastructure_costs: float
    personnel_costs: float
    cost_breakdown: Dict[str, float]
    monthly_recurring_costs: float


@dataclass
class BenefitSummary:
    """Summary of project benefits by category."""
    total_benefits: float
    realized_benefits: float
    projected_benefits: float
    cost_savings: float
    productivity_gains: float
    revenue_increases: float
    benefit_breakdown: Dict[str, float]
    realization_percentage: float


class ROICalculator:
    """
    Comprehensive ROI Calculator with automated cost collection and benefit measurement.
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.cloud_connector = CloudConnectorManager()
        self.discount_rate = 0.10  # Default 10% discount rate for NPV
        
    def create_roi_analysis(
        self,
        project_id: str,
        project_name: str,
        project_description: str,
        project_start_date: datetime,
        analysis_period_months: int = 12,
        created_by: str = "system"
    ) -> ROIAnalysis:
        """Create a new ROI analysis for a project."""
        try:
            with get_sync_db() as session:
                # Check if analysis already exists
                existing = session.query(ROIAnalysis).filter_by(project_id=project_id).first()
                if existing:
                    logger.warning(f"ROI analysis already exists for project {project_id}")
                    return existing
                
                analysis_start = datetime.utcnow()
                analysis_end = analysis_start + timedelta(days=analysis_period_months * 30)
                
                roi_analysis = ROIAnalysis(
                    project_id=project_id,
                    project_name=project_name,
                    project_description=project_description,
                    project_status=ProjectStatus.PLANNING.value,
                    project_start_date=project_start_date,
                    analysis_period_start=analysis_start,
                    analysis_period_end=analysis_end,
                    created_by=created_by,
                    last_updated_by=created_by,
                    analysis_methodology={
                        "discount_rate": self.discount_rate,
                        "calculation_method": "net_present_value",
                        "benefit_measurement": "automated_and_manual"
                    },
                    assumptions=[
                        "Discount rate of 10% for NPV calculations",
                        "Benefits realized linearly over analysis period",
                        "Costs tracked monthly with automated collection where possible"
                    ],
                    confidence_level=0.8
                )
                
                session.add(roi_analysis)
                session.commit()
                session.refresh(roi_analysis)
                
                logger.info(f"Created ROI analysis for project {project_id}")
                return roi_analysis
                
        except Exception as e:
            logger.error(f"Error creating ROI analysis: {str(e)}")
            raise
    
    def track_project_costs(
        self,
        project_id: str,
        cost_category: str,
        description: str,
        amount: float,
        cost_date: Optional[datetime] = None,
        vendor: Optional[str] = None,
        is_recurring: bool = False,
        recurrence_frequency: Optional[str] = None,
        **kwargs
    ) -> CostTracking:
        """Track a cost item for a project."""
        try:
            with get_sync_db() as session:
                # Get ROI analysis
                roi_analysis = session.query(ROIAnalysis).filter_by(project_id=project_id).first()
                if not roi_analysis:
                    raise ValueError(f"ROI analysis not found for project {project_id}")
                
                cost_item = CostTracking(
                    roi_analysis_id=roi_analysis.id,
                    cost_category=cost_category,
                    description=description,
                    amount=amount,
                    cost_date=cost_date or datetime.utcnow(),
                    vendor=vendor,
                    is_recurring=is_recurring,
                    recurrence_frequency=recurrence_frequency,
                    **kwargs
                )
                
                session.add(cost_item)
                
                # Update total investment in ROI analysis
                self._update_roi_totals(session, roi_analysis.id)
                
                session.commit()
                session.refresh(cost_item)
                
                logger.info(f"Tracked cost of ${amount} for project {project_id}")
                return cost_item
                
        except Exception as e:
            logger.error(f"Error tracking project costs: {str(e)}")
            raise
    
    def track_project_benefits(
        self,
        project_id: str,
        benefit_category: str,
        description: str,
        quantified_value: float,
        measurement_method: str,
        baseline_value: Optional[float] = None,
        current_value: Optional[float] = None,
        is_realized: bool = False,
        **kwargs
    ) -> BenefitTracking:
        """Track a benefit item for a project."""
        try:
            with get_sync_db() as session:
                # Get ROI analysis
                roi_analysis = session.query(ROIAnalysis).filter_by(project_id=project_id).first()
                if not roi_analysis:
                    raise ValueError(f"ROI analysis not found for project {project_id}")
                
                benefit_item = BenefitTracking(
                    roi_analysis_id=roi_analysis.id,
                    benefit_category=benefit_category,
                    description=description,
                    quantified_value=quantified_value,
                    measurement_method=measurement_method,
                    baseline_value=baseline_value,
                    current_value=current_value,
                    is_realized=is_realized,
                    benefit_date=datetime.utcnow(),
                    **kwargs
                )
                
                session.add(benefit_item)
                
                # Update total benefits in ROI analysis
                self._update_roi_totals(session, roi_analysis.id)
                
                session.commit()
                session.refresh(benefit_item)
                
                logger.info(f"Tracked benefit of ${quantified_value} for project {project_id}")
                return benefit_item
                
        except Exception as e:
            logger.error(f"Error tracking project benefits: {str(e)}")
            raise
    
    def collect_cloud_costs(
        self,
        project_id: str,
        provider: str,
        account_id: str,
        start_date: datetime,
        end_date: datetime,
        cost_allocation_tags: Optional[Dict[str, str]] = None
    ) -> List[CloudCostCollection]:
        """Automatically collect cloud costs for a project."""
        try:
            logger.info(f"Collecting cloud costs for project {project_id} from {provider}")
            
            # Get cloud costs using connector
            cloud_costs = self.cloud_connector.get_costs(
                provider=provider,
                account_id=account_id,
                start_date=start_date,
                end_date=end_date,
                tags=cost_allocation_tags
            )
            
            collected_costs = []
            
            with get_sync_db() as session:
                for cost_data in cloud_costs:
                    # Create cloud cost collection record
                    cloud_cost = CloudCostCollection(
                        provider=provider,
                        account_id=account_id,
                        service_name=cost_data.get('service_name', 'Unknown'),
                        cost_amount=cost_data.get('cost', 0.0),
                        billing_period=cost_data.get('billing_period'),
                        usage_start_date=cost_data.get('usage_start_date', start_date),
                        usage_end_date=cost_data.get('usage_end_date', end_date),
                        collection_method='api',
                        raw_data=cost_data,
                        assigned_to_project=project_id
                    )
                    
                    session.add(cloud_cost)
                    collected_costs.append(cloud_cost)
                    
                    # Also create cost tracking record
                    self.track_project_costs(
                        project_id=project_id,
                        cost_category=CostType.CLOUD_SERVICES.value,
                        description=f"{provider} {cost_data.get('service_name', 'service')} costs",
                        amount=cost_data.get('cost', 0.0),
                        cost_date=cost_data.get('usage_end_date', end_date),
                        vendor=provider,
                        data_source=f"{provider}_api",
                        is_automated_collection=True,
                        collection_method='api'
                    )
                
                session.commit()
                
            logger.info(f"Collected {len(collected_costs)} cloud cost items for project {project_id}")
            return collected_costs
            
        except Exception as e:
            logger.error(f"Error collecting cloud costs: {str(e)}")
            raise
    
    def measure_efficiency_gains(
        self,
        project_id: str,
        process_name: str,
        time_before_hours: float,
        time_after_hours: float,
        frequency_per_month: float = 1.0,
        hourly_rate: float = 50.0,
        error_rate_before: float = 0.0,
        error_rate_after: float = 0.0,
        measurement_method: str = "time_study"
    ) -> 'EfficiencyGainMetric':
        """Measure detailed efficiency gains with comprehensive cost impact analysis."""
        try:
            from ..models.roi_models import EfficiencyGainMetric
            
            # Calculate time savings
            time_saved_hours = time_before_hours - time_after_hours
            time_saved_percentage = (time_saved_hours / time_before_hours) * 100 if time_before_hours > 0 else 0
            
            # Calculate cost savings
            cost_savings_per_period = time_saved_hours * hourly_rate
            monthly_savings = cost_savings_per_period * frequency_per_month
            annual_savings = monthly_savings * 12
            
            # Calculate quality improvement
            quality_improvement_percentage = 0.0
            if error_rate_before > 0:
                quality_improvement_percentage = ((error_rate_before - error_rate_after) / error_rate_before) * 100
            
            with get_sync_db() as session:
                efficiency_metric = EfficiencyGainMetric(
                    project_id=project_id,
                    process_name=process_name,
                    efficiency_category="optimization",
                    time_before_hours=time_before_hours,
                    time_after_hours=time_after_hours,
                    time_saved_hours=time_saved_hours,
                    time_saved_percentage=time_saved_percentage,
                    hourly_rate=hourly_rate,
                    cost_savings_per_period=cost_savings_per_period,
                    frequency_per_month=frequency_per_month,
                    monthly_savings=monthly_savings,
                    annual_savings=annual_savings,
                    error_rate_before=error_rate_before,
                    error_rate_after=error_rate_after,
                    quality_improvement_percentage=quality_improvement_percentage,
                    measurement_method=measurement_method
                )
                
                session.add(efficiency_metric)
                session.commit()
                session.refresh(efficiency_metric)
                
                # Track as benefit in ROI analysis
                self.track_project_benefits(
                    project_id=project_id,
                    benefit_category=BenefitType.EFFICIENCY_IMPROVEMENT.value,
                    description=f"Efficiency gain from {process_name}",
                    quantified_value=annual_savings,
                    measurement_method=measurement_method,
                    baseline_value=time_before_hours,
                    current_value=time_after_hours,
                    is_realized=True
                )
                
                logger.info(f"Measured efficiency gain: {time_saved_percentage:.2f}% time savings, ${annual_savings:.2f} annual savings")
                return efficiency_metric
                
        except Exception as e:
            logger.error(f"Error measuring efficiency gains: {str(e)}")
            raise

    def measure_productivity_gains(
        self,
        project_id: str,
        metric_name: str,
        baseline_value: float,
        current_value: float,
        measurement_unit: str,
        process_or_task: str,
        measurement_method: str = "automated_measurement"
    ) -> ProductivityMetric:
        """Measure and track productivity gains from a project."""
        try:
            improvement_percentage = ((current_value - baseline_value) / baseline_value) * 100
            
            with get_sync_db() as session:
                productivity_metric = ProductivityMetric(
                    metric_name=metric_name,
                    metric_category="productivity",
                    metric_type="efficiency_improvement",
                    baseline_value=baseline_value,
                    current_value=current_value,
                    improvement_percentage=improvement_percentage,
                    measurement_unit=measurement_unit,
                    measurement_date=datetime.utcnow(),
                    measurement_period_start=datetime.utcnow() - timedelta(days=30),
                    measurement_period_end=datetime.utcnow(),
                    process_or_task=process_or_task,
                    project_id=project_id,
                    measurement_method=measurement_method,
                    is_automated_collection=True
                )
                
                session.add(productivity_metric)
                session.commit()
                session.refresh(productivity_metric)
                
                # Convert productivity gain to monetary benefit
                if improvement_percentage > 0:
                    # Estimate monetary value (this would be customized per organization)
                    estimated_hourly_value = 50.0  # Default $50/hour
                    hours_saved_per_month = (improvement_percentage / 100) * 160  # Assuming 160 hours/month
                    monthly_benefit = hours_saved_per_month * estimated_hourly_value
                    
                    # Track as benefit
                    self.track_project_benefits(
                        project_id=project_id,
                        benefit_category=BenefitType.PRODUCTIVITY_GAIN.value,
                        description=f"Productivity gain from {metric_name}",
                        quantified_value=monthly_benefit,
                        measurement_method=measurement_method,
                        baseline_value=baseline_value,
                        current_value=current_value,
                        is_realized=True
                    )
                
                logger.info(f"Measured productivity gain: {improvement_percentage:.2f}% for {metric_name}")
                return productivity_metric
                
        except Exception as e:
            logger.error(f"Error measuring productivity gains: {str(e)}")
            raise
    
    def calculate_roi(self, project_id: str) -> ROICalculationResult:
        """Calculate comprehensive ROI metrics for a project."""
        try:
            with get_sync_db() as session:
                roi_analysis = session.query(ROIAnalysis).filter_by(project_id=project_id).first()
                if not roi_analysis:
                    raise ValueError(f"ROI analysis not found for project {project_id}")
                
                # Get all costs and benefits
                costs = session.query(CostTracking).filter_by(roi_analysis_id=roi_analysis.id).all()
                benefits = session.query(BenefitTracking).filter_by(roi_analysis_id=roi_analysis.id).all()
                
                # Calculate totals
                total_investment = sum(cost.amount for cost in costs)
                total_benefits = sum(benefit.quantified_value for benefit in benefits if benefit.is_realized)
                projected_benefits = sum(benefit.quantified_value for benefit in benefits)
                
                # Calculate ROI percentage
                roi_percentage = ((total_benefits - total_investment) / total_investment * 100) if total_investment > 0 else 0
                
                # Calculate NPV
                npv = self._calculate_npv(costs, benefits, self.discount_rate)
                
                # Calculate IRR
                irr = self._calculate_irr(costs, benefits)
                
                # Calculate payback period
                payback_months = self._calculate_payback_period(costs, benefits)
                
                # Calculate break-even date
                break_even_date = self._calculate_break_even_date(roi_analysis.project_start_date, costs, benefits)
                
                # Update ROI analysis record
                roi_analysis.total_investment = total_investment
                roi_analysis.total_benefits = total_benefits
                roi_analysis.realized_benefits = total_benefits
                roi_analysis.projected_benefits = projected_benefits
                roi_analysis.roi_percentage = roi_percentage
                roi_analysis.net_present_value = npv
                roi_analysis.internal_rate_of_return = irr
                roi_analysis.payback_period_months = payback_months
                roi_analysis.break_even_date = break_even_date
                roi_analysis.updated_at = datetime.utcnow()
                
                session.commit()
                
                result = ROICalculationResult(
                    roi_percentage=roi_percentage,
                    net_present_value=npv,
                    internal_rate_of_return=irr,
                    payback_period_months=payback_months,
                    break_even_date=break_even_date,
                    total_investment=total_investment,
                    total_benefits=total_benefits,
                    confidence_level=roi_analysis.confidence_level,
                    calculation_date=datetime.utcnow()
                )
                
                logger.info(f"Calculated ROI for project {project_id}: {roi_percentage:.2f}%")
                return result
                
        except Exception as e:
            logger.error(f"Error calculating ROI: {str(e)}")
            raise
    
    def get_cost_summary(self, project_id: str) -> CostSummary:
        """Get detailed cost summary for a project."""
        try:
            with get_sync_db() as session:
                roi_analysis = session.query(ROIAnalysis).filter_by(project_id=project_id).first()
                if not roi_analysis:
                    raise ValueError(f"ROI analysis not found for project {project_id}")
                
                costs = session.query(CostTracking).filter_by(roi_analysis_id=roi_analysis.id).all()
                
                total_costs = sum(cost.amount for cost in costs)
                
                # Categorize costs
                cost_breakdown = {}
                for cost_type in CostType:
                    category_costs = sum(cost.amount for cost in costs if cost.cost_category == cost_type.value)
                    cost_breakdown[cost_type.value] = category_costs
                
                # Calculate recurring costs
                monthly_recurring = sum(
                    cost.amount for cost in costs 
                    if cost.is_recurring and cost.recurrence_frequency == "monthly"
                )
                
                return CostSummary(
                    total_costs=total_costs,
                    direct_costs=cost_breakdown.get(CostType.DIRECT.value, 0),
                    indirect_costs=cost_breakdown.get(CostType.INDIRECT.value, 0),
                    operational_costs=cost_breakdown.get(CostType.OPERATIONAL.value, 0),
                    infrastructure_costs=cost_breakdown.get(CostType.INFRASTRUCTURE.value, 0),
                    personnel_costs=cost_breakdown.get(CostType.PERSONNEL.value, 0),
                    cost_breakdown=cost_breakdown,
                    monthly_recurring_costs=monthly_recurring
                )
                
        except Exception as e:
            logger.error(f"Error getting cost summary: {str(e)}")
            raise
    
    def get_benefit_summary(self, project_id: str) -> BenefitSummary:
        """Get detailed benefit summary for a project."""
        try:
            with get_sync_db() as session:
                roi_analysis = session.query(ROIAnalysis).filter_by(project_id=project_id).first()
                if not roi_analysis:
                    raise ValueError(f"ROI analysis not found for project {project_id}")
                
                benefits = session.query(BenefitTracking).filter_by(roi_analysis_id=roi_analysis.id).all()
                
                total_benefits = sum(benefit.quantified_value for benefit in benefits)
                realized_benefits = sum(benefit.quantified_value for benefit in benefits if benefit.is_realized)
                projected_benefits = sum(benefit.quantified_value for benefit in benefits if not benefit.is_realized)
                
                # Categorize benefits
                benefit_breakdown = {}
                for benefit_type in BenefitType:
                    category_benefits = sum(
                        benefit.quantified_value for benefit in benefits 
                        if benefit.benefit_category == benefit_type.value
                    )
                    benefit_breakdown[benefit_type.value] = category_benefits
                
                realization_percentage = (realized_benefits / total_benefits * 100) if total_benefits > 0 else 0
                
                return BenefitSummary(
                    total_benefits=total_benefits,
                    realized_benefits=realized_benefits,
                    projected_benefits=projected_benefits,
                    cost_savings=benefit_breakdown.get(BenefitType.COST_SAVINGS.value, 0),
                    productivity_gains=benefit_breakdown.get(BenefitType.PRODUCTIVITY_GAIN.value, 0),
                    revenue_increases=benefit_breakdown.get(BenefitType.REVENUE_INCREASE.value, 0),
                    benefit_breakdown=benefit_breakdown,
                    realization_percentage=realization_percentage
                )
                
        except Exception as e:
            logger.error(f"Error getting benefit summary: {str(e)}")
            raise
    
    def generate_roi_report(
        self,
        project_id: str,
        report_type: str = "detailed",
        report_format: str = "json"
    ) -> ROIReport:
        """Generate comprehensive ROI report with visualizations."""
        try:
            # Calculate current ROI
            roi_result = self.calculate_roi(project_id)
            cost_summary = self.get_cost_summary(project_id)
            benefit_summary = self.get_benefit_summary(project_id)
            
            # Prepare report data
            report_data = {
                "project_id": project_id,
                "calculation_date": roi_result.calculation_date.isoformat(),
                "roi_metrics": {
                    "roi_percentage": roi_result.roi_percentage,
                    "net_present_value": roi_result.net_present_value,
                    "internal_rate_of_return": roi_result.internal_rate_of_return,
                    "payback_period_months": roi_result.payback_period_months,
                    "break_even_date": roi_result.break_even_date.isoformat() if roi_result.break_even_date else None,
                    "confidence_level": roi_result.confidence_level
                },
                "cost_summary": {
                    "total_costs": cost_summary.total_costs,
                    "cost_breakdown": cost_summary.cost_breakdown,
                    "monthly_recurring_costs": cost_summary.monthly_recurring_costs
                },
                "benefit_summary": {
                    "total_benefits": benefit_summary.total_benefits,
                    "realized_benefits": benefit_summary.realized_benefits,
                    "projected_benefits": benefit_summary.projected_benefits,
                    "benefit_breakdown": benefit_summary.benefit_breakdown,
                    "realization_percentage": benefit_summary.realization_percentage
                }
            }
            
            # Generate visualizations config
            visualizations = [
                {
                    "type": "pie_chart",
                    "title": "Cost Breakdown by Category",
                    "data": cost_summary.cost_breakdown
                },
                {
                    "type": "pie_chart", 
                    "title": "Benefit Breakdown by Category",
                    "data": benefit_summary.benefit_breakdown
                },
                {
                    "type": "bar_chart",
                    "title": "ROI Metrics Overview",
                    "data": {
                        "ROI %": roi_result.roi_percentage,
                        "NPV": roi_result.net_present_value,
                        "Payback (months)": roi_result.payback_period_months
                    }
                }
            ]
            
            # Generate executive summary
            executive_summary = self._generate_executive_summary(roi_result, cost_summary, benefit_summary)
            
            # Generate key findings
            key_findings = self._generate_key_findings(roi_result, cost_summary, benefit_summary)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(roi_result, cost_summary, benefit_summary)
            
            with get_sync_db() as session:
                roi_analysis = session.query(ROIAnalysis).filter_by(project_id=project_id).first()
                
                roi_report = ROIReport(
                    roi_analysis_id=roi_analysis.id,
                    report_name=f"ROI Report - {roi_analysis.project_name}",
                    report_type=report_type,
                    report_format=report_format,
                    report_data=report_data,
                    visualizations=visualizations,
                    executive_summary=executive_summary,
                    key_findings=key_findings,
                    recommendations=recommendations,
                    report_period_start=roi_analysis.analysis_period_start,
                    report_period_end=roi_analysis.analysis_period_end,
                    generated_by="roi_calculator"
                )
                
                session.add(roi_report)
                session.commit()
                session.refresh(roi_report)
                
                logger.info(f"Generated ROI report for project {project_id}")
                return roi_report
                
        except Exception as e:
            logger.error(f"Error generating ROI report: {str(e)}")
            raise
    
    def _update_roi_totals(self, session, roi_analysis_id: str):
        """Update total investment and benefits in ROI analysis."""
        roi_analysis = session.query(ROIAnalysis).filter_by(id=roi_analysis_id).first()
        if roi_analysis:
            # Update total investment
            total_investment = session.query(CostTracking).filter_by(
                roi_analysis_id=roi_analysis_id
            ).with_entities(
                session.query(CostTracking.amount).filter_by(roi_analysis_id=roi_analysis_id).subquery().c.amount.sum()
            ).scalar() or 0.0
            
            # Update total benefits
            total_benefits = session.query(BenefitTracking).filter_by(
                roi_analysis_id=roi_analysis_id
            ).with_entities(
                session.query(BenefitTracking.quantified_value).filter_by(roi_analysis_id=roi_analysis_id).subquery().c.quantified_value.sum()
            ).scalar() or 0.0
            
            roi_analysis.total_investment = total_investment
            roi_analysis.total_benefits = total_benefits
            roi_analysis.updated_at = datetime.utcnow()
    
    def _calculate_npv(self, costs: List[CostTracking], benefits: List[BenefitTracking], discount_rate: float) -> float:
        """Calculate Net Present Value."""
        try:
            # Simplified NPV calculation - in practice this would be more sophisticated
            total_costs = sum(cost.amount for cost in costs)
            total_benefits = sum(benefit.quantified_value for benefit in benefits if benefit.is_realized)
            
            # Assume benefits are realized over 12 months
            monthly_benefit = total_benefits / 12
            npv = 0
            
            for month in range(12):
                discounted_benefit = monthly_benefit / ((1 + discount_rate/12) ** month)
                npv += discounted_benefit
            
            return npv - total_costs
            
        except Exception:
            return 0.0
    
    def _calculate_irr(self, costs: List[CostTracking], benefits: List[BenefitTracking]) -> float:
        """Calculate Internal Rate of Return."""
        try:
            # Simplified IRR calculation
            total_costs = sum(cost.amount for cost in costs)
            total_benefits = sum(benefit.quantified_value for benefit in benefits if benefit.is_realized)
            
            if total_costs == 0:
                return 0.0
            
            # Assume 12-month period
            return ((total_benefits / total_costs) ** (1/1)) - 1
            
        except Exception:
            return 0.0
    
    def _calculate_payback_period(self, costs: List[CostTracking], benefits: List[BenefitTracking]) -> int:
        """Calculate payback period in months."""
        try:
            total_costs = sum(cost.amount for cost in costs)
            monthly_benefits = sum(benefit.quantified_value for benefit in benefits if benefit.is_realized) / 12
            
            if monthly_benefits <= 0:
                return 999  # Never pays back
            
            return int(total_costs / monthly_benefits)
            
        except Exception:
            return 999
    
    def _calculate_break_even_date(
        self, 
        project_start: datetime, 
        costs: List[CostTracking], 
        benefits: List[BenefitTracking]
    ) -> Optional[datetime]:
        """Calculate break-even date for the project."""
        try:
            total_costs = sum(cost.amount for cost in costs)
            monthly_benefits = sum(benefit.quantified_value for benefit in benefits if benefit.is_realized) / 12
            
            if monthly_benefits <= 0:
                return None
            
            months_to_break_even = total_costs / monthly_benefits
            return project_start + timedelta(days=int(months_to_break_even * 30))
            
        except Exception:
            return None
    
    def _generate_executive_summary(
        self, 
        roi_result: ROICalculationResult, 
        cost_summary: CostSummary, 
        benefit_summary: BenefitSummary
    ) -> str:
        """Generate executive summary for ROI report."""
        summary = f"""
        Executive Summary - ROI Analysis
        
        This project demonstrates a {roi_result.roi_percentage:.1f}% return on investment with a total investment of ${cost_summary.total_costs:,.2f} 
        and realized benefits of ${benefit_summary.realized_benefits:,.2f}.
        
        Key Metrics:
        - ROI: {roi_result.roi_percentage:.1f}%
        - Net Present Value: ${roi_result.net_present_value:,.2f}
        - Payback Period: {roi_result.payback_period_months} months
        - Benefit Realization: {benefit_summary.realization_percentage:.1f}%
        
        The project shows {'strong' if roi_result.roi_percentage > 20 else 'positive' if roi_result.roi_percentage > 0 else 'negative'} 
        financial performance with {'quick' if roi_result.payback_period_months <= 12 else 'moderate'} payback period.
        """
        
        return summary.strip()
    
    def _generate_executive_summary(
        self, 
        roi_result: ROICalculationResult, 
        cost_summary: CostSummary, 
        benefit_summary: BenefitSummary
    ) -> str:
        """Generate executive summary for ROI report."""
        return f"""
        Executive Summary:
        
        This project has achieved an ROI of {roi_result.roi_percentage:.1f}% with a total investment of ${cost_summary.total_costs:,.2f} 
        and realized benefits of ${benefit_summary.realized_benefits:,.2f}.
        
        The project will pay back its investment in {roi_result.payback_period_months} months, with a Net Present Value of 
        ${roi_result.net_present_value:,.2f}.
        
        Key benefit categories include productivity gains (${benefit_summary.productivity_gains:,.2f}) and cost savings 
        (${benefit_summary.cost_savings:,.2f}).
        """
    
    def _generate_key_findings(
        self, 
        roi_result: ROICalculationResult, 
        cost_summary: CostSummary, 
        benefit_summary: BenefitSummary
    ) -> List[str]:
        """Generate key findings for ROI report."""
        findings = []
        
        if roi_result.roi_percentage > 20:
            findings.append("Project shows strong positive ROI above 20%")
        elif roi_result.roi_percentage > 0:
            findings.append("Project shows positive ROI")
        else:
            findings.append("Project currently shows negative ROI - review required")
        
        if roi_result.payback_period_months <= 12:
            findings.append("Quick payback period of less than 12 months")
        
        if benefit_summary.realization_percentage < 50:
            findings.append("Low benefit realization - focus on benefit capture")
        
        return findings
    
    def _generate_recommendations(
        self, 
        roi_result: ROICalculationResult, 
        cost_summary: CostSummary, 
        benefit_summary: BenefitSummary
    ) -> List[str]:
        """Generate recommendations for ROI report."""
        recommendations = []
        
        if roi_result.roi_percentage < 10:
            recommendations.append("Consider cost optimization or benefit enhancement strategies")
        
        if cost_summary.monthly_recurring_costs > cost_summary.total_costs * 0.1:
            recommendations.append("Review recurring costs for optimization opportunities")
        
        if benefit_summary.realization_percentage < 70:
            recommendations.append("Implement benefit realization tracking and governance")
        
        return recommendations    

    def generate_detailed_roi_breakdown(self, project_id: str) -> Dict[str, Any]:
        """Generate detailed ROI breakdown with comprehensive analysis."""
        try:
            with get_sync_db() as session:
                roi_analysis = session.query(ROIAnalysis).filter_by(project_id=project_id).first()
                if not roi_analysis:
                    raise ValueError(f"ROI analysis not found for project {project_id}")
                
                # Get detailed cost and benefit data
                costs = session.query(CostTracking).filter_by(roi_analysis_id=roi_analysis.id).all()
                benefits = session.query(BenefitTracking).filter_by(roi_analysis_id=roi_analysis.id).all()
                
                # Calculate monthly trends
                monthly_costs = {}
                monthly_benefits = {}
                
                for cost in costs:
                    month_key = cost.cost_date.strftime('%Y-%m')
                    monthly_costs[month_key] = monthly_costs.get(month_key, 0) + cost.amount
                
                for benefit in benefits:
                    month_key = benefit.benefit_date.strftime('%Y-%m')
                    monthly_benefits[month_key] = monthly_benefits.get(month_key, 0) + benefit.quantified_value
                
                # Calculate cumulative ROI over time
                cumulative_roi = []
                cumulative_cost = 0
                cumulative_benefit = 0
                
                all_months = sorted(set(list(monthly_costs.keys()) + list(monthly_benefits.keys())))
                
                for month in all_months:
                    cumulative_cost += monthly_costs.get(month, 0)
                    cumulative_benefit += monthly_benefits.get(month, 0)
                    
                    roi_value = ((cumulative_benefit - cumulative_cost) / cumulative_cost * 100) if cumulative_cost > 0 else 0
                    
                    cumulative_roi.append({
                        'month': month,
                        'cumulative_cost': cumulative_cost,
                        'cumulative_benefit': cumulative_benefit,
                        'roi_percentage': roi_value
                    })
                
                # Risk analysis
                risk_factors = self._analyze_roi_risks(roi_analysis, costs, benefits)
                
                # Sensitivity analysis
                sensitivity_analysis = self._perform_sensitivity_analysis(roi_analysis, costs, benefits)
                
                return {
                    'project_id': project_id,
                    'monthly_trends': {
                        'costs': monthly_costs,
                        'benefits': monthly_benefits
                    },
                    'cumulative_roi_trend': cumulative_roi,
                    'risk_analysis': risk_factors,
                    'sensitivity_analysis': sensitivity_analysis,
                    'cost_efficiency_metrics': self._calculate_cost_efficiency_metrics(costs, benefits),
                    'benefit_realization_timeline': self._analyze_benefit_realization_timeline(benefits)
                }
                
        except Exception as e:
            logger.error(f"Error generating detailed ROI breakdown: {str(e)}")
            raise
    
    def _analyze_roi_risks(self, roi_analysis: ROIAnalysis, costs: List[CostTracking], benefits: List[BenefitTracking]) -> Dict[str, Any]:
        """Analyze risks that could impact ROI."""
        risks = {
            'high_risk_factors': [],
            'medium_risk_factors': [],
            'low_risk_factors': [],
            'overall_risk_score': 0.0
        }
        
        # Analyze cost concentration risk
        total_costs = sum(cost.amount for cost in costs)
        if total_costs > 0:
            cost_by_vendor = {}
            for cost in costs:
                vendor = cost.vendor or 'Unknown'
                cost_by_vendor[vendor] = cost_by_vendor.get(vendor, 0) + cost.amount
            
            max_vendor_percentage = max(cost_by_vendor.values()) / total_costs * 100
            if max_vendor_percentage > 50:
                risks['high_risk_factors'].append(f"High vendor concentration: {max_vendor_percentage:.1f}% from single vendor")
        
        # Analyze benefit realization risk
        total_benefits = sum(benefit.quantified_value for benefit in benefits)
        realized_benefits = sum(benefit.quantified_value for benefit in benefits if benefit.is_realized)
        
        if total_benefits > 0:
            realization_rate = realized_benefits / total_benefits * 100
            if realization_rate < 30:
                risks['high_risk_factors'].append(f"Low benefit realization: {realization_rate:.1f}%")
            elif realization_rate < 60:
                risks['medium_risk_factors'].append(f"Moderate benefit realization: {realization_rate:.1f}%")
        
        # Calculate overall risk score
        risk_score = len(risks['high_risk_factors']) * 3 + len(risks['medium_risk_factors']) * 2 + len(risks['low_risk_factors']) * 1
        risks['overall_risk_score'] = min(risk_score / 10, 1.0)  # Normalize to 0-1 scale
        
        return risks
    
    def _perform_sensitivity_analysis(self, roi_analysis: ROIAnalysis, costs: List[CostTracking], benefits: List[BenefitTracking]) -> Dict[str, Any]:
        """Perform sensitivity analysis on ROI calculations."""
        base_roi = roi_analysis.roi_percentage or 0
        
        scenarios = {
            'optimistic': {'cost_change': -0.1, 'benefit_change': 0.2},  # 10% cost reduction, 20% benefit increase
            'pessimistic': {'cost_change': 0.2, 'benefit_change': -0.1},  # 20% cost increase, 10% benefit reduction
            'cost_overrun': {'cost_change': 0.3, 'benefit_change': 0.0},  # 30% cost overrun
            'benefit_shortfall': {'cost_change': 0.0, 'benefit_change': -0.3}  # 30% benefit shortfall
        }
        
        sensitivity_results = {}
        
        total_costs = sum(cost.amount for cost in costs)
        total_benefits = sum(benefit.quantified_value for benefit in benefits)
        
        for scenario_name, changes in scenarios.items():
            adjusted_costs = total_costs * (1 + changes['cost_change'])
            adjusted_benefits = total_benefits * (1 + changes['benefit_change'])
            
            if adjusted_costs > 0:
                scenario_roi = ((adjusted_benefits - adjusted_costs) / adjusted_costs) * 100
                roi_impact = scenario_roi - base_roi
                
                sensitivity_results[scenario_name] = {
                    'adjusted_roi': scenario_roi,
                    'roi_impact': roi_impact,
                    'cost_change_percent': changes['cost_change'] * 100,
                    'benefit_change_percent': changes['benefit_change'] * 100
                }
        
        return sensitivity_results
    
    def _calculate_cost_efficiency_metrics(self, costs: List[CostTracking], benefits: List[BenefitTracking]) -> Dict[str, float]:
        """Calculate cost efficiency metrics."""
        total_costs = sum(cost.amount for cost in costs)
        total_benefits = sum(benefit.quantified_value for benefit in benefits if benefit.is_realized)
        
        metrics = {
            'cost_per_benefit_dollar': total_costs / total_benefits if total_benefits > 0 else 0,
            'benefit_cost_ratio': total_benefits / total_costs if total_costs > 0 else 0,
            'cost_efficiency_score': min(total_benefits / total_costs, 5.0) if total_costs > 0 else 0  # Cap at 5.0
        }
        
        return metrics
    
    def _analyze_benefit_realization_timeline(self, benefits: List[BenefitTracking]) -> Dict[str, Any]:
        """Analyze benefit realization timeline and patterns."""
        realized_benefits = [b for b in benefits if b.is_realized]
        projected_benefits = [b for b in benefits if not b.is_realized]
        
        timeline_analysis = {
            'total_benefits': len(benefits),
            'realized_count': len(realized_benefits),
            'projected_count': len(projected_benefits),
            'realization_rate': len(realized_benefits) / len(benefits) * 100 if benefits else 0,
            'average_realization_time_days': 0,
            'benefit_categories': {}
        }
        
        # Calculate average realization time
        if realized_benefits:
            realization_times = []
            for benefit in realized_benefits:
                if benefit.actual_realization_date and benefit.projected_realization_date:
                    days_diff = (benefit.actual_realization_date - benefit.projected_realization_date).days
                    realization_times.append(days_diff)
            
            if realization_times:
                timeline_analysis['average_realization_time_days'] = sum(realization_times) / len(realization_times)
        
        # Analyze by benefit category
        for benefit in benefits:
            category = benefit.benefit_category
            if category not in timeline_analysis['benefit_categories']:
                timeline_analysis['benefit_categories'][category] = {
                    'total': 0,
                    'realized': 0,
                    'total_value': 0,
                    'realized_value': 0
                }
            
            timeline_analysis['benefit_categories'][category]['total'] += 1
            timeline_analysis['benefit_categories'][category]['total_value'] += benefit.quantified_value
            
            if benefit.is_realized:
                timeline_analysis['benefit_categories'][category]['realized'] += 1
                timeline_analysis['benefit_categories'][category]['realized_value'] += benefit.quantified_value
        
        return timeline_analysis
    
    def generate_roi_dashboard_data(self, project_id: str) -> Dict[str, Any]:
        """Generate comprehensive data for ROI dashboard visualization."""
        try:
            # Get all ROI data
            roi_result = self.calculate_roi(project_id)
            cost_summary = self.get_cost_summary(project_id)
            benefit_summary = self.get_benefit_summary(project_id)
            detailed_breakdown = self.generate_detailed_roi_breakdown(project_id)
            
            # Prepare dashboard data
            dashboard_data = {
                'summary_metrics': {
                    'roi_percentage': roi_result.roi_percentage,
                    'total_investment': roi_result.total_investment,
                    'total_benefits': roi_result.total_benefits,
                    'net_present_value': roi_result.net_present_value,
                    'payback_period_months': roi_result.payback_period_months,
                    'break_even_date': roi_result.break_even_date.isoformat() if roi_result.break_even_date else None,
                    'confidence_level': roi_result.confidence_level
                },
                'cost_breakdown': {
                    'total_costs': cost_summary.total_costs,
                    'cost_categories': cost_summary.cost_breakdown,
                    'monthly_recurring': cost_summary.monthly_recurring_costs
                },
                'benefit_breakdown': {
                    'total_benefits': benefit_summary.total_benefits,
                    'realized_benefits': benefit_summary.realized_benefits,
                    'projected_benefits': benefit_summary.projected_benefits,
                    'benefit_categories': benefit_summary.benefit_breakdown,
                    'realization_percentage': benefit_summary.realization_percentage
                },
                'trends': detailed_breakdown['monthly_trends'],
                'cumulative_roi': detailed_breakdown['cumulative_roi_trend'],
                'risk_analysis': detailed_breakdown['risk_analysis'],
                'sensitivity_analysis': detailed_breakdown['sensitivity_analysis'],
                'efficiency_metrics': detailed_breakdown['cost_efficiency_metrics'],
                'visualization_configs': self._generate_visualization_configs(roi_result, cost_summary, benefit_summary)
            }
            
            return dashboard_data
            
        except Exception as e:
            logger.error(f"Error generating ROI dashboard data: {str(e)}")
            raise
    
    def _generate_visualization_configs(self, roi_result: ROICalculationResult, cost_summary: CostSummary, benefit_summary: BenefitSummary) -> List[Dict[str, Any]]:
        """Generate visualization configurations for ROI dashboard."""
        visualizations = [
            {
                'type': 'gauge',
                'title': 'ROI Percentage',
                'data': {
                    'value': roi_result.roi_percentage,
                    'min': -50,
                    'max': 100,
                    'thresholds': [
                        {'value': 0, 'color': 'red'},
                        {'value': 15, 'color': 'yellow'},
                        {'value': 30, 'color': 'green'}
                    ]
                }
            },
            {
                'type': 'donut_chart',
                'title': 'Cost Breakdown by Category',
                'data': cost_summary.cost_breakdown
            },
            {
                'type': 'donut_chart',
                'title': 'Benefit Breakdown by Category',
                'data': benefit_summary.benefit_breakdown
            },
            {
                'type': 'bar_chart',
                'title': 'Key Financial Metrics',
                'data': {
                    'Total Investment': roi_result.total_investment,
                    'Total Benefits': roi_result.total_benefits,
                    'Net Present Value': roi_result.net_present_value
                }
            },
            {
                'type': 'progress_bar',
                'title': 'Benefit Realization Progress',
                'data': {
                    'percentage': benefit_summary.realization_percentage,
                    'realized': benefit_summary.realized_benefits,
                    'total': benefit_summary.total_benefits
                }
            }
        ]
        
        return visualizations
    
    def _calculate_break_even_date(self, project_start_date: datetime, costs: List[CostTracking], benefits: List[BenefitTracking]) -> Optional[datetime]:
        """Calculate the break-even date when cumulative benefits exceed cumulative costs."""
        try:
            # Sort costs and benefits by date
            sorted_costs = sorted(costs, key=lambda x: x.cost_date)
            sorted_benefits = sorted(benefits, key=lambda x: x.benefit_date)
            
            cumulative_cost = 0
            cumulative_benefit = 0
            
            # Create timeline of all financial events
            events = []
            
            for cost in sorted_costs:
                events.append({
                    'date': cost.cost_date,
                    'type': 'cost',
                    'amount': cost.amount
                })
            
            for benefit in sorted_benefits:
                if benefit.is_realized:
                    events.append({
                        'date': benefit.benefit_date,
                        'type': 'benefit',
                        'amount': benefit.quantified_value
                    })
            
            # Sort events by date
            events.sort(key=lambda x: x['date'])
            
            # Find break-even point
            for event in events:
                if event['type'] == 'cost':
                    cumulative_cost += event['amount']
                else:
                    cumulative_benefit += event['amount']
                
                # Check if we've reached break-even
                if cumulative_benefit >= cumulative_cost:
                    return event['date']
            
            # If no break-even found, estimate based on current trajectory
            if cumulative_benefit > 0 and cumulative_cost > cumulative_benefit:
                # Estimate monthly benefit rate
                if events:
                    time_span = (events[-1]['date'] - events[0]['date']).days / 30  # months
                    if time_span > 0:
                        monthly_benefit_rate = cumulative_benefit / time_span
                        remaining_deficit = cumulative_cost - cumulative_benefit
                        months_to_break_even = remaining_deficit / monthly_benefit_rate if monthly_benefit_rate > 0 else 12
                        return events[-1]['date'] + timedelta(days=int(months_to_break_even * 30))
            
            return None
            
        except Exception as e:
            logger.error(f"Error calculating break-even date: {str(e)}")
            return None
    
    def _generate_executive_summary(self, roi_result: ROICalculationResult, cost_summary: CostSummary, benefit_summary: BenefitSummary) -> str:
        """Generate executive summary for ROI report."""
        try:
            roi_status = "positive" if roi_result.roi_percentage > 0 else "negative"
            payback_status = "excellent" if roi_result.payback_period_months <= 12 else "good" if roi_result.payback_period_months <= 24 else "extended"
            
            summary = f"""
Executive Summary:

This ROI analysis demonstrates a {roi_status} return on investment of {roi_result.roi_percentage:.1f}% 
with a total investment of ${roi_result.total_investment:,.2f} generating ${roi_result.total_benefits:,.2f} 
in benefits. The project shows a net present value of ${roi_result.net_present_value:,.2f} and a 
{payback_status} payback period of {roi_result.payback_period_months} months.

Key Financial Metrics:
• Total Investment: ${cost_summary.total_costs:,.2f}
• Total Benefits: ${benefit_summary.total_benefits:,.2f}
• Net Benefit: ${roi_result.total_benefits - roi_result.total_investment:,.2f}
• Benefit Realization: {benefit_summary.realization_percentage:.1f}%

The analysis indicates {"strong" if roi_result.roi_percentage > 25 else "moderate" if roi_result.roi_percentage > 10 else "limited"} 
financial returns with a confidence level of {roi_result.confidence_level:.0%}.
            """.strip()
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating executive summary: {str(e)}")
            return "Executive summary generation failed."
    
    def _generate_key_findings(self, roi_result: ROICalculationResult, cost_summary: CostSummary, benefit_summary: BenefitSummary) -> List[str]:
        """Generate key findings for ROI report."""
        try:
            findings = []
            
            # ROI findings
            if roi_result.roi_percentage > 30:
                findings.append(f"Exceptional ROI of {roi_result.roi_percentage:.1f}% significantly exceeds typical project returns")
            elif roi_result.roi_percentage > 15:
                findings.append(f"Strong ROI of {roi_result.roi_percentage:.1f}% demonstrates solid project value")
            elif roi_result.roi_percentage > 0:
                findings.append(f"Positive ROI of {roi_result.roi_percentage:.1f}% indicates project viability")
            else:
                findings.append(f"Negative ROI of {roi_result.roi_percentage:.1f}% requires attention and optimization")
            
            # Payback findings
            if roi_result.payback_period_months <= 6:
                findings.append("Rapid payback period indicates quick value realization")
            elif roi_result.payback_period_months <= 12:
                findings.append("Reasonable payback period aligns with typical project expectations")
            else:
                findings.append("Extended payback period may impact project attractiveness")
            
            # Cost findings
            if cost_summary.monthly_recurring_costs > cost_summary.total_costs * 0.1:
                findings.append("Significant recurring costs require ongoing budget allocation")
            
            # Benefit findings
            if benefit_summary.realization_percentage > 80:
                findings.append("High benefit realization rate demonstrates effective project execution")
            elif benefit_summary.realization_percentage < 50:
                findings.append("Low benefit realization rate indicates potential execution challenges")
            
            # NPV findings
            if roi_result.net_present_value > roi_result.total_investment:
                findings.append("Strong net present value indicates long-term project value")
            
            return findings
            
        except Exception as e:
            logger.error(f"Error generating key findings: {str(e)}")
            return ["Key findings generation failed."]
    
    def _generate_recommendations(self, roi_result: ROICalculationResult, cost_summary: CostSummary, benefit_summary: BenefitSummary) -> List[str]:
        """Generate recommendations for ROI report."""
        try:
            recommendations = []
            
            # ROI-based recommendations
            if roi_result.roi_percentage < 10:
                recommendations.append("Consider optimizing project scope or implementation approach to improve ROI")
                recommendations.append("Review cost structure and identify opportunities for cost reduction")
            
            if roi_result.payback_period_months > 18:
                recommendations.append("Explore ways to accelerate benefit realization to reduce payback period")
            
            # Cost-based recommendations
            if cost_summary.monthly_recurring_costs > 0:
                recommendations.append("Monitor recurring costs closely and implement cost optimization strategies")
            
            if cost_summary.infrastructure_costs > cost_summary.total_costs * 0.4:
                recommendations.append("Evaluate infrastructure efficiency and consider cloud optimization")
            
            # Benefit-based recommendations
            if benefit_summary.realization_percentage < 70:
                recommendations.append("Implement benefit tracking and realization improvement initiatives")
                recommendations.append("Establish regular benefit measurement and validation processes")
            
            if benefit_summary.projected_benefits > benefit_summary.realized_benefits:
                recommendations.append("Focus on converting projected benefits to realized benefits")
            
            # General recommendations
            if roi_result.confidence_level < 0.7:
                recommendations.append("Improve data quality and measurement accuracy to increase confidence")
            
            recommendations.append("Continue monitoring ROI metrics and adjust strategy as needed")
            recommendations.append("Document lessons learned for future project planning")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            return ["Recommendations generation failed."]
