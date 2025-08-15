"""
Risk-Benefit Analysis Engine for Crisis Leadership Excellence

This engine provides rapid evaluation of response options under uncertainty,
risk assessment, mitigation strategy generation, and benefit optimization.
"""

import logging
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
import numpy as np
from dataclasses import asdict

from ..models.risk_benefit_models import (
    RiskFactor, BenefitFactor, ResponseOption, MitigationStrategy,
    TradeOffAnalysis, RiskBenefitEvaluation, OptimizationResult,
    RiskLevel, BenefitType, UncertaintyLevel
)

logger = logging.getLogger(__name__)


class RiskBenefitAnalyzer:
    """
    Advanced risk-benefit analysis engine for crisis decision making.
    Provides rapid evaluation under uncertainty with optimization capabilities.
    """
    
    def __init__(self):
        self.risk_weights = {
            RiskLevel.CRITICAL: 1.0,
            RiskLevel.HIGH: 0.8,
            RiskLevel.MEDIUM: 0.6,
            RiskLevel.LOW: 0.4,
            RiskLevel.MINIMAL: 0.2
        }
        
        self.uncertainty_adjustments = {
            UncertaintyLevel.VERY_HIGH: 0.5,
            UncertaintyLevel.HIGH: 0.7,
            UncertaintyLevel.MODERATE: 0.85,
            UncertaintyLevel.LOW: 0.95,
            UncertaintyLevel.VERY_LOW: 1.0
        }
        
        self.benefit_weights = {
            BenefitType.FINANCIAL: 0.25,
            BenefitType.OPERATIONAL: 0.20,
            BenefitType.STRATEGIC: 0.20,
            BenefitType.REPUTATIONAL: 0.15,
            BenefitType.STAKEHOLDER: 0.15,
            BenefitType.COMPETITIVE: 0.05
        }
    
    def evaluate_response_options(
        self,
        crisis_id: str,
        response_options: List[ResponseOption],
        evaluation_criteria: Dict[str, float],
        risk_tolerance: RiskLevel = RiskLevel.MEDIUM,
        time_pressure: str = "moderate"
    ) -> RiskBenefitEvaluation:
        """
        Rapidly evaluate multiple response options under uncertainty.
        
        Args:
            crisis_id: ID of the crisis being addressed
            response_options: List of potential response options
            evaluation_criteria: Weighted criteria for evaluation
            risk_tolerance: Organization's risk tolerance level
            time_pressure: Level of time pressure for decision
            
        Returns:
            Complete risk-benefit evaluation with recommendations
        """
        try:
            logger.info(f"Starting risk-benefit evaluation for crisis {crisis_id}")
            
            # Score each response option
            scored_options = []
            for option in response_options:
                risk_score = self._calculate_risk_score(option.risks, risk_tolerance)
                benefit_score = self._calculate_benefit_score(option.benefits)
                overall_score = self._calculate_overall_score(
                    risk_score, benefit_score, evaluation_criteria, time_pressure
                )
                
                scored_options.append({
                    'option': option,
                    'risk_score': risk_score,
                    'benefit_score': benefit_score,
                    'overall_score': overall_score
                })
            
            # Sort by overall score (higher is better)
            scored_options.sort(key=lambda x: x['overall_score'], reverse=True)
            
            # Generate trade-off analyses
            trade_off_analyses = self._generate_trade_off_analyses(scored_options[:3])
            
            # Create mitigation plan for top option
            mitigation_plan = self._generate_mitigation_strategies(
                scored_options[0]['option'].risks
            )
            
            # Calculate confidence and uncertainty factors
            confidence_score = self._calculate_confidence_score(scored_options[0]['option'])
            uncertainty_factors = self._identify_uncertainty_factors(response_options)
            
            # Create evaluation result
            evaluation = RiskBenefitEvaluation(
                crisis_id=crisis_id,
                response_options=response_options,
                evaluation_criteria=evaluation_criteria,
                risk_tolerance=risk_tolerance,
                time_pressure=time_pressure,
                recommended_option=scored_options[0]['option'].id,
                confidence_score=confidence_score,
                uncertainty_factors=uncertainty_factors,
                trade_off_analyses=trade_off_analyses,
                mitigation_plan=mitigation_plan,
                monitoring_requirements=self._generate_monitoring_requirements(
                    scored_options[0]['option']
                )
            )
            
            logger.info(f"Risk-benefit evaluation completed for crisis {crisis_id}")
            return evaluation
            
        except Exception as e:
            logger.error(f"Error in risk-benefit evaluation: {str(e)}")
            raise
    
    def _calculate_risk_score(
        self, 
        risks: List[RiskFactor], 
        risk_tolerance: RiskLevel
    ) -> float:
        """Calculate weighted risk score for response option."""
        if not risks:
            return 0.0
        
        total_risk = 0.0
        for risk in risks:
            # Base risk calculation
            risk_value = (
                risk.probability * 
                self.risk_weights[risk.impact_severity] *
                self.uncertainty_adjustments[risk.uncertainty_level]
            )
            
            # Adjust for risk tolerance
            tolerance_adjustment = 1.0
            if risk_tolerance == RiskLevel.LOW:
                tolerance_adjustment = 1.5
            elif risk_tolerance == RiskLevel.HIGH:
                tolerance_adjustment = 0.7
            
            total_risk += risk_value * tolerance_adjustment
        
        return min(total_risk / len(risks), 1.0)
    
    def _calculate_benefit_score(self, benefits: List[BenefitFactor]) -> float:
        """Calculate weighted benefit score for response option."""
        if not benefits:
            return 0.0
        
        total_benefit = 0.0
        for benefit in benefits:
            benefit_value = (
                benefit.expected_value *
                benefit.probability_of_realization *
                self.benefit_weights[benefit.benefit_type] *
                self.uncertainty_adjustments[benefit.uncertainty_level]
            )
            total_benefit += benefit_value
        
        return min(total_benefit / len(benefits), 1.0)
    
    def _calculate_overall_score(
        self,
        risk_score: float,
        benefit_score: float,
        criteria: Dict[str, float],
        time_pressure: str
    ) -> float:
        """Calculate overall score considering all factors."""
        # Base calculation: benefit minus risk
        base_score = benefit_score - risk_score
        
        # Apply evaluation criteria weights
        risk_weight = criteria.get('risk_aversion', 0.4)
        benefit_weight = criteria.get('benefit_focus', 0.6)
        
        weighted_score = (benefit_score * benefit_weight) - (risk_score * risk_weight)
        
        # Adjust for time pressure
        time_adjustments = {
            'immediate': 0.9,  # Slight penalty for hasty decisions
            'urgent': 0.95,
            'moderate': 1.0,
            'low': 1.05  # Slight bonus for thoughtful decisions
        }
        
        time_adjustment = time_adjustments.get(time_pressure, 1.0)
        
        return max(0.0, min(1.0, weighted_score * time_adjustment))
    
    def _generate_trade_off_analyses(
        self, 
        top_options: List[Dict]
    ) -> List[TradeOffAnalysis]:
        """Generate trade-off analyses between top options."""
        analyses = []
        
        for i in range(len(top_options) - 1):
            option_a = top_options[i]['option']
            option_b = top_options[i + 1]['option']
            
            # Compare key factors
            comparison_criteria = [
                'risk_level', 'benefit_potential', 'implementation_speed',
                'resource_requirements', 'success_probability'
            ]
            
            trade_off_factors = {
                'risk_difference': top_options[i]['risk_score'] - top_options[i + 1]['risk_score'],
                'benefit_difference': top_options[i]['benefit_score'] - top_options[i + 1]['benefit_score'],
                'score_difference': top_options[i]['overall_score'] - top_options[i + 1]['overall_score']
            }
            
            # Generate recommendation
            if trade_off_factors['score_difference'] > 0.1:
                recommendation = f"Strong preference for {option_a.name}"
                confidence = 0.8
            elif trade_off_factors['score_difference'] > 0.05:
                recommendation = f"Moderate preference for {option_a.name}"
                confidence = 0.6
            else:
                recommendation = "Options are closely matched - consider additional factors"
                confidence = 0.4
            
            analysis = TradeOffAnalysis(
                option_a_id=option_a.id,
                option_b_id=option_b.id,
                comparison_criteria=comparison_criteria,
                trade_off_factors=trade_off_factors,
                recommendation=recommendation,
                confidence_level=confidence,
                decision_rationale=self._generate_trade_off_rationale(
                    option_a, option_b, trade_off_factors
                )
            )
            
            analyses.append(analysis)
        
        return analyses
    
    def _generate_trade_off_rationale(
        self,
        option_a: ResponseOption,
        option_b: ResponseOption,
        factors: Dict[str, float]
    ) -> str:
        """Generate rationale for trade-off analysis."""
        rationale_parts = []
        
        if factors['risk_difference'] > 0.1:
            rationale_parts.append(f"{option_a.name} has significantly higher risk")
        elif factors['risk_difference'] < -0.1:
            rationale_parts.append(f"{option_a.name} has significantly lower risk")
        
        if factors['benefit_difference'] > 0.1:
            rationale_parts.append(f"{option_a.name} offers significantly higher benefits")
        elif factors['benefit_difference'] < -0.1:
            rationale_parts.append(f"{option_a.name} offers significantly lower benefits")
        
        if not rationale_parts:
            rationale_parts.append("Options have similar risk-benefit profiles")
        
        return ". ".join(rationale_parts) + "."
    
    def _generate_mitigation_strategies(
        self, 
        risks: List[RiskFactor]
    ) -> List[MitigationStrategy]:
        """Generate mitigation strategies for identified risks."""
        strategies = []
        
        for risk in risks:
            if risk.impact_severity in [RiskLevel.CRITICAL, RiskLevel.HIGH]:
                # Generate specific mitigation strategies
                strategy_templates = {
                    'financial': [
                        'Establish financial reserves',
                        'Diversify funding sources',
                        'Implement cost controls'
                    ],
                    'operational': [
                        'Create backup procedures',
                        'Cross-train personnel',
                        'Establish redundant systems'
                    ],
                    'reputational': [
                        'Prepare communication plan',
                        'Engage stakeholders proactively',
                        'Monitor public sentiment'
                    ],
                    'regulatory': [
                        'Ensure compliance documentation',
                        'Engage legal counsel',
                        'Maintain regulatory relationships'
                    ]
                }
                
                category_strategies = strategy_templates.get(
                    risk.category.lower(), 
                    ['Monitor situation closely', 'Prepare contingency plans']
                )
                
                for strategy_name in category_strategies:
                    strategy = MitigationStrategy(
                        name=f"{strategy_name} for {risk.name}",
                        description=f"Mitigate {risk.name} through {strategy_name.lower()}",
                        target_risks=[risk.id],
                        effectiveness_score=0.7,  # Default effectiveness
                        implementation_time="short_term",
                        success_probability=0.8
                    )
                    strategies.append(strategy)
        
        return strategies
    
    def _calculate_confidence_score(self, option: ResponseOption) -> float:
        """Calculate confidence score for recommended option."""
        # Base confidence on uncertainty levels
        risk_uncertainties = [r.uncertainty_level for r in option.risks]
        benefit_uncertainties = [b.uncertainty_level for b in option.benefits]
        
        all_uncertainties = risk_uncertainties + benefit_uncertainties
        
        if not all_uncertainties:
            return 0.5
        
        # Calculate average uncertainty adjustment
        avg_uncertainty = np.mean([
            self.uncertainty_adjustments[u] for u in all_uncertainties
        ])
        
        # Higher uncertainty adjustment means lower uncertainty, higher confidence
        return avg_uncertainty
    
    def _identify_uncertainty_factors(
        self, 
        options: List[ResponseOption]
    ) -> List[str]:
        """Identify key uncertainty factors across all options."""
        uncertainty_factors = set()
        
        for option in options:
            for risk in option.risks:
                if risk.uncertainty_level in [UncertaintyLevel.HIGH, UncertaintyLevel.VERY_HIGH]:
                    uncertainty_factors.add(f"High uncertainty in risk: {risk.name}")
            
            for benefit in option.benefits:
                if benefit.uncertainty_level in [UncertaintyLevel.HIGH, UncertaintyLevel.VERY_HIGH]:
                    uncertainty_factors.add(f"High uncertainty in benefit: {benefit.name}")
        
        return list(uncertainty_factors)
    
    def _generate_monitoring_requirements(
        self, 
        option: ResponseOption
    ) -> List[str]:
        """Generate monitoring requirements for selected option."""
        requirements = []
        
        # Monitor high-risk factors
        for risk in option.risks:
            if risk.impact_severity in [RiskLevel.CRITICAL, RiskLevel.HIGH]:
                requirements.append(f"Monitor {risk.name} indicators closely")
        
        # Monitor key benefits
        for benefit in option.benefits:
            if benefit.expected_value > 0.7:
                requirements.append(f"Track {benefit.name} realization progress")
        
        # General monitoring
        requirements.extend([
            "Monitor implementation progress against timeline",
            "Track resource utilization and availability",
            "Assess stakeholder reactions and feedback",
            "Monitor external environment changes"
        ])
        
        return requirements
    
    def optimize_benefits(
        self,
        evaluation: RiskBenefitEvaluation,
        optimization_objective: str = "maximize_total_value"
    ) -> OptimizationResult:
        """
        Optimize benefits for the recommended response option.
        
        Args:
            evaluation: Risk-benefit evaluation to optimize
            optimization_objective: Objective for optimization
            
        Returns:
            Optimization result with enhanced benefits
        """
        try:
            logger.info(f"Starting benefit optimization for evaluation {evaluation.id}")
            
            # Find recommended option
            recommended_option = None
            for option in evaluation.response_options:
                if option.id == evaluation.recommended_option:
                    recommended_option = option
                    break
            
            if not recommended_option:
                raise ValueError("Recommended option not found")
            
            # Optimize benefits
            optimized_benefits = []
            optimization_strategies = []
            
            for benefit in recommended_option.benefits:
                # Create optimized version
                optimized_benefit = BenefitFactor(
                    name=benefit.name,
                    description=benefit.description,
                    benefit_type=benefit.benefit_type,
                    expected_value=min(1.0, benefit.expected_value * 1.2),  # 20% improvement
                    probability_of_realization=min(1.0, benefit.probability_of_realization * 1.1),
                    time_to_realization=benefit.time_to_realization,
                    sustainability=benefit.sustainability,
                    uncertainty_level=benefit.uncertainty_level,
                    optimization_strategies=benefit.optimization_strategies + [
                        "Enhanced resource allocation",
                        "Accelerated implementation timeline",
                        "Stakeholder engagement optimization"
                    ]
                )
                
                optimized_benefits.append(optimized_benefit)
                
                # Add optimization strategies
                if benefit.benefit_type == BenefitType.FINANCIAL:
                    optimization_strategies.append("Focus on high-ROI activities")
                elif benefit.benefit_type == BenefitType.OPERATIONAL:
                    optimization_strategies.append("Streamline operational processes")
                elif benefit.benefit_type == BenefitType.STRATEGIC:
                    optimization_strategies.append("Align with long-term strategic goals")
            
            # Calculate expected improvement
            original_benefit_score = self._calculate_benefit_score(recommended_option.benefits)
            optimized_benefit_score = self._calculate_benefit_score(optimized_benefits)
            expected_improvement = optimized_benefit_score - original_benefit_score
            
            result = OptimizationResult(
                evaluation_id=evaluation.id,
                optimization_objective=optimization_objective,
                optimized_benefits=optimized_benefits,
                optimization_strategies=list(set(optimization_strategies)),
                expected_improvement=expected_improvement,
                implementation_requirements={
                    "additional_resources": "10-20% increase",
                    "timeline_acceleration": "Possible with focused effort",
                    "stakeholder_buy_in": "Critical for success"
                },
                success_probability=0.75
            )
            
            logger.info(f"Benefit optimization completed with {expected_improvement:.2f} improvement")
            return result
            
        except Exception as e:
            logger.error(f"Error in benefit optimization: {str(e)}")
            raise