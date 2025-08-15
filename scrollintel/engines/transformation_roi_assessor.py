"""
Transformation ROI Assessor

This engine assesses and validates the return on investment (ROI) of cultural
transformations, including financial metrics, comprehensive value assessment,
and long-term projections.
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class ROIAssessment:
    """Assessment of transformation ROI"""
    roi_percentage: float
    payback_period: float  # months
    net_present_value: float
    benefit_cost_ratio: float
    total_investment: float
    total_benefits: float
    assessment_confidence: float


@dataclass
class ComprehensiveValueAssessment:
    """Comprehensive value assessment including intangibles"""
    total_value_score: float
    tangible_value: float
    intangible_value: float
    value_drivers: List[Dict[str, Any]]
    value_sustainability: float


@dataclass
class ROIValidationResult:
    """Result of ROI validation"""
    calculation_accuracy: float
    data_completeness: float
    assumption_validity: float
    validation_warnings: List[str]
    base_case_roi: float
    sensitivity_analysis: Dict[str, Any]


@dataclass
class LongTermProjection:
    """Long-term value projection"""
    five_year_roi: float
    cumulative_value: float
    value_sustainability_score: float
    value_milestones: List[Dict[str, Any]]
    projection_confidence: float


@dataclass
class BenchmarkComparison:
    """ROI benchmark comparison"""
    performance_ranking: str
    roi_percentile: float
    competitive_advantage_score: float
    benchmark_gaps: List[str]
    improvement_opportunities: List[str]


@dataclass
class RiskAdjustedROI:
    """Risk-adjusted ROI assessment"""
    adjusted_roi_percentage: float
    risk_discount_factor: float
    confidence_level: float
    risk_factors: List[Dict[str, Any]]
    risk_mitigation_recommendations: List[str]


class TransformationROIAssessor:
    """Assesses ROI and value creation from cultural transformations"""
    
    def __init__(self):
        self.discount_rate = 0.08  # 8% discount rate for NPV calculations
        self.risk_free_rate = 0.03  # 3% risk-free rate
        self.confidence_thresholds = {
            "high": 0.85,
            "medium": 0.7,
            "low": 0.5
        }
    
    def calculate_financial_roi(self, roi_data: Dict[str, Any]) -> ROIAssessment:
        """
        Calculate financial ROI of transformation
        
        Args:
            roi_data: Investment and benefit data
        
        Returns:
            ROIAssessment with financial ROI metrics
        """
        try:
            investment_data = roi_data.get("investment_data", {})
            benefit_data = roi_data.get("benefit_data", {})
            
            # Calculate total investment
            direct_costs = sum(investment_data.get("direct_costs", {}).values())
            indirect_costs = sum(investment_data.get("indirect_costs", {}).values())
            total_investment = direct_costs + indirect_costs
            
            # Calculate total benefits (annualized)
            quantifiable_benefits = sum(benefit_data.get("quantifiable_benefits", {}).values())
            qualitative_benefits = sum(
                benefit.get("value_estimate", 0) 
                for benefit in benefit_data.get("qualitative_benefits", [])
            )
            annual_benefits = quantifiable_benefits + qualitative_benefits
            
            # Calculate ROI metrics
            measurement_period = benefit_data.get("measurement_period", 24)  # months
            total_benefits = annual_benefits * (measurement_period / 12)
            
            # ROI percentage
            roi_percentage = ((total_benefits - total_investment) / total_investment * 100) if total_investment > 0 else 0
            
            # Payback period (months)
            monthly_benefits = annual_benefits / 12
            payback_period = total_investment / monthly_benefits if monthly_benefits > 0 else float('inf')
            
            # Net Present Value
            npv = self._calculate_npv(total_investment, annual_benefits, measurement_period / 12)
            
            # Benefit-Cost Ratio
            benefit_cost_ratio = total_benefits / total_investment if total_investment > 0 else 0
            
            # Assessment confidence
            assessment_confidence = self._calculate_roi_confidence(roi_data)
            
            return ROIAssessment(
                roi_percentage=roi_percentage,
                payback_period=payback_period,
                net_present_value=npv,
                benefit_cost_ratio=benefit_cost_ratio,
                total_investment=total_investment,
                total_benefits=total_benefits,
                assessment_confidence=assessment_confidence
            )
            
        except Exception as e:
            logger.error(f"Error calculating financial ROI: {str(e)}")
            return ROIAssessment(
                roi_percentage=0.0,
                payback_period=float('inf'),
                net_present_value=0.0,
                benefit_cost_ratio=0.0,
                total_investment=0.0,
                total_benefits=0.0,
                assessment_confidence=0.0
            )
    
    def assess_comprehensive_value(self, roi_data: Dict[str, Any]) -> ComprehensiveValueAssessment:
        """
        Assess comprehensive value including intangible benefits
        
        Args:
            roi_data: Complete ROI data including intangibles
        
        Returns:
            ComprehensiveValueAssessment with comprehensive value analysis
        """
        try:
            benefit_data = roi_data.get("benefit_data", {})
            
            # Calculate tangible value
            tangible_benefits = benefit_data.get("quantifiable_benefits", {})
            tangible_value = sum(tangible_benefits.values())
            
            # Calculate intangible value
            qualitative_benefits = benefit_data.get("qualitative_benefits", [])
            intangible_value = sum(benefit.get("value_estimate", 0) for benefit in qualitative_benefits)
            
            # Identify value drivers
            value_drivers = []
            for benefit_name, annual_value in tangible_benefits.items():
                value_drivers.append({
                    "name": benefit_name,
                    "type": "tangible",
                    "annual_value": annual_value,
                    "contribution_percentage": annual_value / tangible_value * 100 if tangible_value > 0 else 0
                })
            
            for benefit in qualitative_benefits:
                value_drivers.append({
                    "name": benefit.get("benefit", "unknown"),
                    "type": "intangible",
                    "annual_value": benefit.get("value_estimate", 0),
                    "contribution_percentage": benefit.get("value_estimate", 0) / intangible_value * 100 if intangible_value > 0 else 0
                })
            
            # Calculate total value score (normalized)
            total_annual_value = tangible_value + intangible_value
            investment_data = roi_data.get("investment_data", {})
            total_investment = sum(investment_data.get("direct_costs", {}).values()) + sum(investment_data.get("indirect_costs", {}).values())
            
            total_value_score = min(1.0, total_annual_value / total_investment) if total_investment > 0 else 0
            
            # Assess value sustainability
            value_sustainability = self._assess_value_sustainability(roi_data, value_drivers)
            
            return ComprehensiveValueAssessment(
                total_value_score=total_value_score,
                tangible_value=tangible_value,
                intangible_value=intangible_value,
                value_drivers=value_drivers,
                value_sustainability=value_sustainability
            )
            
        except Exception as e:
            logger.error(f"Error assessing comprehensive value: {str(e)}")
            return ComprehensiveValueAssessment(
                total_value_score=0.0,
                tangible_value=0.0,
                intangible_value=0.0,
                value_drivers=[],
                value_sustainability=0.0
            )
    
    def validate_roi_calculation(self, roi_data: Dict[str, Any]) -> ROIValidationResult:
        """
        Validate ROI calculation accuracy and assumptions
        
        Args:
            roi_data: ROI calculation data
        
        Returns:
            ROIValidationResult with validation assessment
        """
        try:
            # Calculate base case ROI
            base_roi_assessment = self.calculate_financial_roi(roi_data)
            base_case_roi = base_roi_assessment.roi_percentage
            
            # Assess calculation accuracy
            calculation_accuracy = self._assess_calculation_accuracy(roi_data)
            
            # Assess data completeness
            data_completeness = self._assess_data_completeness(roi_data)
            
            # Assess assumption validity
            assumption_validity = self._assess_assumption_validity(roi_data)
            
            # Generate validation warnings
            validation_warnings = self._generate_validation_warnings(
                calculation_accuracy, data_completeness, assumption_validity
            )
            
            # Perform sensitivity analysis
            sensitivity_analysis = self._perform_sensitivity_analysis(roi_data, base_case_roi)
            
            return ROIValidationResult(
                calculation_accuracy=calculation_accuracy,
                data_completeness=data_completeness,
                assumption_validity=assumption_validity,
                validation_warnings=validation_warnings,
                base_case_roi=base_case_roi,
                sensitivity_analysis=sensitivity_analysis
            )
            
        except Exception as e:
            logger.error(f"Error validating ROI calculation: {str(e)}")
            return ROIValidationResult(
                calculation_accuracy=0.0,
                data_completeness=0.0,
                assumption_validity=0.0,
                validation_warnings=["Validation error occurred"],
                base_case_roi=0.0,
                sensitivity_analysis={}
            )
    
    def project_long_term_value(self, projection_data: Dict[str, Any]) -> LongTermProjection:
        """
        Project long-term value creation
        
        Args:
            projection_data: Data for long-term projections
        
        Returns:
            LongTermProjection with long-term value analysis
        """
        try:
            projection_period = projection_data.get("projection_period", 60)  # months
            growth_assumptions = projection_data.get("growth_assumptions", {})
            
            # Calculate base annual benefits
            benefit_data = projection_data.get("benefit_data", {})
            base_annual_benefits = sum(benefit_data.get("quantifiable_benefits", {}).values())
            
            # Project benefits over time with growth
            productivity_growth = growth_assumptions.get("productivity_growth", 0.03)
            benefit_sustainability = growth_assumptions.get("benefit_sustainability", 0.9)
            
            projected_benefits = []
            cumulative_value = 0
            
            for year in range(1, int(projection_period / 12) + 1):
                # Apply growth and sustainability factors
                year_benefits = base_annual_benefits * (1 + productivity_growth) ** year * benefit_sustainability ** year
                projected_benefits.append(year_benefits)
                cumulative_value += year_benefits
            
            # Calculate 5-year ROI
            investment_data = projection_data.get("investment_data", {})
            total_investment = sum(investment_data.get("direct_costs", {}).values()) + sum(investment_data.get("indirect_costs", {}).values())
            
            five_year_benefits = sum(projected_benefits[:5]) if len(projected_benefits) >= 5 else sum(projected_benefits)
            five_year_roi = ((five_year_benefits - total_investment) / total_investment * 100) if total_investment > 0 else 0
            
            # Assess value sustainability
            value_sustainability_score = self._assess_long_term_sustainability(growth_assumptions, projected_benefits)
            
            # Generate value milestones
            value_milestones = self._generate_value_milestones(projected_benefits, total_investment)
            
            # Calculate projection confidence
            projection_confidence = self._calculate_projection_confidence(growth_assumptions, projection_period)
            
            return LongTermProjection(
                five_year_roi=five_year_roi,
                cumulative_value=cumulative_value,
                value_sustainability_score=value_sustainability_score,
                value_milestones=value_milestones,
                projection_confidence=projection_confidence
            )
            
        except Exception as e:
            logger.error(f"Error projecting long-term value: {str(e)}")
            return LongTermProjection(
                five_year_roi=0.0,
                cumulative_value=0.0,
                value_sustainability_score=0.0,
                value_milestones=[],
                projection_confidence=0.0
            )
    
    def benchmark_roi_performance(self,
                                 roi_assessment: ROIAssessment,
                                 benchmark_data: Dict[str, Any]) -> BenchmarkComparison:
        """
        Benchmark ROI performance against industry standards
        
        Args:
            roi_assessment: ROI assessment results
            benchmark_data: Industry benchmark data
        
        Returns:
            BenchmarkComparison with benchmark analysis
        """
        try:
            industry_averages = benchmark_data.get("industry_averages", {})
            best_practices = benchmark_data.get("best_practices", {})
            
            # Compare ROI percentage
            industry_roi = industry_averages.get("cultural_transformation_roi", 50)
            top_quartile_roi = best_practices.get("top_quartile_roi", 100)
            
            # Determine performance ranking
            if roi_assessment.roi_percentage >= top_quartile_roi:
                performance_ranking = "top_quartile"
                roi_percentile = 90
            elif roi_assessment.roi_percentage >= industry_roi * 1.2:
                performance_ranking = "above_average"
                roi_percentile = 75
            elif roi_assessment.roi_percentage >= industry_roi * 0.8:
                performance_ranking = "average"
                roi_percentile = 50
            else:
                performance_ranking = "below_average"
                roi_percentile = 25
            
            # Calculate competitive advantage score
            competitive_advantage_score = min(1.0, roi_assessment.roi_percentage / top_quartile_roi)
            
            # Identify benchmark gaps
            benchmark_gaps = []
            improvement_opportunities = []
            
            if roi_assessment.payback_period > best_practices.get("optimal_payback", 18):
                benchmark_gaps.append("payback_period_longer_than_optimal")
                improvement_opportunities.append("Accelerate benefit realization")
            
            if roi_assessment.benefit_cost_ratio < 2.0:
                benchmark_gaps.append("low_benefit_cost_ratio")
                improvement_opportunities.append("Enhance benefit capture and measurement")
            
            return BenchmarkComparison(
                performance_ranking=performance_ranking,
                roi_percentile=roi_percentile,
                competitive_advantage_score=competitive_advantage_score,
                benchmark_gaps=benchmark_gaps,
                improvement_opportunities=improvement_opportunities
            )
            
        except Exception as e:
            logger.error(f"Error benchmarking ROI performance: {str(e)}")
            return BenchmarkComparison(
                performance_ranking="unknown",
                roi_percentile=50.0,
                competitive_advantage_score=0.0,
                benchmark_gaps=[],
                improvement_opportunities=[]
            )
    
    def calculate_risk_adjusted_roi(self,
                                  roi_data: Dict[str, Any],
                                  risk_factors: Dict[str, Any]) -> RiskAdjustedROI:
        """
        Calculate risk-adjusted ROI
        
        Args:
            roi_data: Base ROI data
            risk_factors: Risk factors with probabilities and impacts
        
        Returns:
            RiskAdjustedROI with risk-adjusted analysis
        """
        try:
            # Calculate base ROI
            base_roi_assessment = self.calculate_financial_roi(roi_data)
            base_roi = base_roi_assessment.roi_percentage
            
            # Calculate risk discount factor
            risk_scores = []
            risk_factor_details = []
            
            for risk_name, risk_info in risk_factors.items():
                probability = risk_info.get("probability", 0)
                impact = risk_info.get("impact", 0)
                risk_score = probability * impact
                
                risk_scores.append(risk_score)
                risk_factor_details.append({
                    "name": risk_name,
                    "probability": probability,
                    "impact": impact,
                    "risk_score": risk_score
                })
            
            overall_risk_score = np.mean(risk_scores) if risk_scores else 0
            risk_discount_factor = 1 - (overall_risk_score * 0.5)  # Max 50% discount
            
            # Calculate risk-adjusted ROI
            adjusted_roi_percentage = base_roi * risk_discount_factor
            
            # Calculate confidence level
            confidence_level = self._calculate_risk_confidence(risk_factors, overall_risk_score)
            
            # Generate risk mitigation recommendations
            risk_mitigation_recommendations = self._generate_risk_mitigation_recommendations(risk_factor_details)
            
            return RiskAdjustedROI(
                adjusted_roi_percentage=adjusted_roi_percentage,
                risk_discount_factor=risk_discount_factor,
                confidence_level=confidence_level,
                risk_factors=risk_factor_details,
                risk_mitigation_recommendations=risk_mitigation_recommendations
            )
            
        except Exception as e:
            logger.error(f"Error calculating risk-adjusted ROI: {str(e)}")
            return RiskAdjustedROI(
                adjusted_roi_percentage=0.0,
                risk_discount_factor=1.0,
                confidence_level=0.0,
                risk_factors=[],
                risk_mitigation_recommendations=[]
            )
    
    # Helper methods
    def _calculate_npv(self, investment: float, annual_benefits: float, years: float) -> float:
        """Calculate Net Present Value"""
        if years <= 0:
            return -investment
        
        # Calculate NPV using discount rate
        npv = -investment
        for year in range(1, int(years) + 1):
            npv += annual_benefits / ((1 + self.discount_rate) ** year)
        
        # Add partial year if applicable
        partial_year = years - int(years)
        if partial_year > 0:
            npv += (annual_benefits * partial_year) / ((1 + self.discount_rate) ** (int(years) + 1))
        
        return npv
    
    def _calculate_roi_confidence(self, roi_data: Dict[str, Any]) -> float:
        """Calculate confidence in ROI assessment"""
        confidence_factors = []
        
        # Data quality factor
        investment_data = roi_data.get("investment_data", {})
        benefit_data = roi_data.get("benefit_data", {})
        
        if investment_data.get("direct_costs") and investment_data.get("indirect_costs"):
            confidence_factors.append(0.9)  # High confidence in investment data
        else:
            confidence_factors.append(0.6)  # Medium confidence
        
        if benefit_data.get("quantifiable_benefits") and len(benefit_data.get("quantifiable_benefits", {})) >= 3:
            confidence_factors.append(0.8)  # Good benefit quantification
        else:
            confidence_factors.append(0.5)  # Limited benefit quantification
        
        # Measurement period factor
        measurement_period = benefit_data.get("measurement_period", 12)
        if measurement_period >= 24:
            confidence_factors.append(0.9)  # Long measurement period
        elif measurement_period >= 12:
            confidence_factors.append(0.7)  # Adequate measurement period
        else:
            confidence_factors.append(0.5)  # Short measurement period
        
        return np.mean(confidence_factors) if confidence_factors else 0.5
    
    def _assess_value_sustainability(self, roi_data: Dict[str, Any], value_drivers: List[Dict[str, Any]]) -> float:
        """Assess sustainability of value creation"""
        sustainability_factors = []
        
        # Check for sustainable value drivers
        sustainable_drivers = ["productivity_improvement", "retention_savings", "innovation_revenue"]
        sustainable_count = sum(1 for driver in value_drivers if driver.get("name") in sustainable_drivers)
        sustainability_factors.append(min(1.0, sustainable_count / 3))
        
        # Check measurement period
        measurement_period = roi_data.get("benefit_data", {}).get("measurement_period", 12)
        if measurement_period >= 24:
            sustainability_factors.append(0.9)
        else:
            sustainability_factors.append(0.6)
        
        return np.mean(sustainability_factors) if sustainability_factors else 0.5
    
    def _assess_calculation_accuracy(self, roi_data: Dict[str, Any]) -> float:
        """Assess accuracy of ROI calculations"""
        accuracy_factors = []
        
        # Check for complete cost data
        investment_data = roi_data.get("investment_data", {})
        if investment_data.get("direct_costs") and investment_data.get("indirect_costs"):
            accuracy_factors.append(0.9)
        else:
            accuracy_factors.append(0.6)
        
        # Check for quantified benefits
        benefit_data = roi_data.get("benefit_data", {})
        quantifiable_benefits = benefit_data.get("quantifiable_benefits", {})
        if len(quantifiable_benefits) >= 3:
            accuracy_factors.append(0.8)
        else:
            accuracy_factors.append(0.5)
        
        return np.mean(accuracy_factors) if accuracy_factors else 0.5
    
    def _assess_data_completeness(self, roi_data: Dict[str, Any]) -> float:
        """Assess completeness of ROI data"""
        required_fields = [
            "investment_data.direct_costs",
            "investment_data.indirect_costs",
            "benefit_data.quantifiable_benefits",
            "benefit_data.measurement_period"
        ]
        
        completeness_score = 0
        for field in required_fields:
            keys = field.split(".")
            data = roi_data
            try:
                for key in keys:
                    data = data[key]
                if data:
                    completeness_score += 1
            except (KeyError, TypeError):
                pass
        
        return completeness_score / len(required_fields)
    
    def _assess_assumption_validity(self, roi_data: Dict[str, Any]) -> float:
        """Assess validity of assumptions"""
        validity_factors = []
        
        # Check benefit estimates reasonableness
        benefit_data = roi_data.get("benefit_data", {})
        quantifiable_benefits = benefit_data.get("quantifiable_benefits", {})
        
        # Productivity improvement should be reasonable (5-25%)
        productivity_improvement = quantifiable_benefits.get("productivity_improvement", 0)
        if 50000 <= productivity_improvement <= 500000:  # Reasonable range
            validity_factors.append(0.9)
        else:
            validity_factors.append(0.6)
        
        # Retention savings should be reasonable
        retention_savings = quantifiable_benefits.get("retention_savings", 0)
        if 50000 <= retention_savings <= 300000:  # Reasonable range
            validity_factors.append(0.8)
        else:
            validity_factors.append(0.6)
        
        return np.mean(validity_factors) if validity_factors else 0.7
    
    def _generate_validation_warnings(self,
                                    calculation_accuracy: float,
                                    data_completeness: float,
                                    assumption_validity: float) -> List[str]:
        """Generate validation warnings"""
        warnings = []
        
        if calculation_accuracy < 0.7:
            warnings.append("ROI calculation accuracy is below acceptable threshold")
        
        if data_completeness < 0.8:
            warnings.append("ROI data is incomplete - some estimates may be unreliable")
        
        if assumption_validity < 0.7:
            warnings.append("Some benefit assumptions may be unrealistic")
        
        return warnings
    
    def _perform_sensitivity_analysis(self, roi_data: Dict[str, Any], base_case_roi: float) -> Dict[str, Any]:
        """Perform sensitivity analysis on ROI"""
        # Simulate different scenarios
        scenarios = {
            "optimistic": 1.2,  # 20% better than expected
            "pessimistic": 0.8,  # 20% worse than expected
        }
        
        scenario_results = {}
        for scenario, factor in scenarios.items():
            # Adjust benefits by factor
            adjusted_roi_data = roi_data.copy()
            benefit_data = adjusted_roi_data.get("benefit_data", {})
            quantifiable_benefits = benefit_data.get("quantifiable_benefits", {})
            
            adjusted_benefits = {k: v * factor for k, v in quantifiable_benefits.items()}
            adjusted_roi_data["benefit_data"]["quantifiable_benefits"] = adjusted_benefits
            
            adjusted_assessment = self.calculate_financial_roi(adjusted_roi_data)
            scenario_results[scenario] = adjusted_assessment.roi_percentage
        
        return {
            "best_case_roi": scenario_results.get("optimistic", base_case_roi),
            "worst_case_roi": scenario_results.get("pessimistic", base_case_roi),
            "confidence_interval": (scenario_results.get("pessimistic", base_case_roi), 
                                  scenario_results.get("optimistic", base_case_roi))
        }
    
    def _assess_long_term_sustainability(self,
                                       growth_assumptions: Dict[str, Any],
                                       projected_benefits: List[float]) -> float:
        """Assess long-term value sustainability"""
        sustainability_factors = []
        
        # Check benefit sustainability factor
        benefit_sustainability = growth_assumptions.get("benefit_sustainability", 0.9)
        sustainability_factors.append(benefit_sustainability)
        
        # Check if benefits decline over time
        if len(projected_benefits) >= 3:
            recent_trend = projected_benefits[-1] / projected_benefits[-3] if projected_benefits[-3] > 0 else 1
            if recent_trend >= 0.9:  # Benefits maintained
                sustainability_factors.append(0.8)
            else:
                sustainability_factors.append(0.6)
        
        return np.mean(sustainability_factors) if sustainability_factors else 0.7
    
    def _generate_value_milestones(self, projected_benefits: List[float], total_investment: float) -> List[Dict[str, Any]]:
        """Generate value creation milestones"""
        milestones = []
        cumulative_benefits = 0
        
        for year, annual_benefits in enumerate(projected_benefits[:5], 1):  # First 5 years
            cumulative_benefits += annual_benefits
            roi_at_milestone = ((cumulative_benefits - total_investment) / total_investment * 100) if total_investment > 0 else 0
            
            milestones.append({
                "year": year,
                "cumulative_benefits": cumulative_benefits,
                "roi_percentage": roi_at_milestone,
                "milestone_type": "annual_review"
            })
        
        return milestones
    
    def _calculate_projection_confidence(self, growth_assumptions: Dict[str, Any], projection_period: int) -> float:
        """Calculate confidence in long-term projections"""
        confidence_factors = []
        
        # Shorter projections are more reliable
        if projection_period <= 36:  # 3 years
            confidence_factors.append(0.8)
        elif projection_period <= 60:  # 5 years
            confidence_factors.append(0.7)
        else:
            confidence_factors.append(0.6)
        
        # Conservative growth assumptions are more reliable
        productivity_growth = growth_assumptions.get("productivity_growth", 0.05)
        if productivity_growth <= 0.05:  # Conservative growth
            confidence_factors.append(0.8)
        else:
            confidence_factors.append(0.6)
        
        return np.mean(confidence_factors) if confidence_factors else 0.7
    
    def _calculate_risk_confidence(self, risk_factors: Dict[str, Any], overall_risk_score: float) -> float:
        """Calculate confidence in risk assessment"""
        confidence_factors = []
        
        # More risk factors assessed = higher confidence
        if len(risk_factors) >= 4:
            confidence_factors.append(0.8)
        else:
            confidence_factors.append(0.6)
        
        # Lower overall risk = higher confidence in projections
        if overall_risk_score <= 0.3:
            confidence_factors.append(0.8)
        elif overall_risk_score <= 0.5:
            confidence_factors.append(0.7)
        else:
            confidence_factors.append(0.6)
        
        return np.mean(confidence_factors) if confidence_factors else 0.7
    
    def _generate_risk_mitigation_recommendations(self, risk_factor_details: List[Dict[str, Any]]) -> List[str]:
        """Generate risk mitigation recommendations"""
        recommendations = []
        
        high_risk_factors = [rf for rf in risk_factor_details if rf.get("risk_score", 0) >= 0.6]
        
        for risk_factor in high_risk_factors:
            risk_name = risk_factor.get("name", "")
            
            if "implementation" in risk_name.lower():
                recommendations.append("Develop detailed implementation plan with contingencies")
            elif "adoption" in risk_name.lower():
                recommendations.append("Enhance change management and communication strategies")
            elif "sustainability" in risk_name.lower():
                recommendations.append("Strengthen reinforcement mechanisms and monitoring")
            elif "external" in risk_name.lower():
                recommendations.append("Create external risk monitoring and response protocols")
        
        return recommendations