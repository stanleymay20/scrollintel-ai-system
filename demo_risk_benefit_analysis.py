"""
Risk-Benefit Analysis System Demonstration

This script demonstrates the crisis leadership risk-benefit analysis capabilities
including response option evaluation, mitigation strategy generation, and benefit optimization.
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, List

from scrollintel.engines.risk_benefit_analyzer import RiskBenefitAnalyzer
from scrollintel.models.risk_benefit_models import (
    RiskFactor, BenefitFactor, ResponseOption, RiskLevel, BenefitType, UncertaintyLevel
)


def create_sample_crisis_scenario():
    """Create a sample crisis scenario for demonstration"""
    
    print("🚨 CRISIS SCENARIO: Major System Outage")
    print("=" * 60)
    print("A critical system failure has occurred affecting 80% of customers.")
    print("Multiple response options are being evaluated under time pressure.")
    print()
    
    # Define response options
    response_options = []
    
    # Option 1: Emergency Rollback
    rollback_risks = [
        RiskFactor(
            name="Data Loss Risk",
            description="Potential loss of recent transactions",
            category="operational",
            probability=0.2,
            impact_severity=RiskLevel.HIGH,
            potential_impact="Loss of customer data and transactions",
            time_horizon="immediate",
            uncertainty_level=UncertaintyLevel.MODERATE
        ),
        RiskFactor(
            name="Extended Downtime",
            description="Rollback process may take longer than expected",
            category="operational",
            probability=0.3,
            impact_severity=RiskLevel.MEDIUM,
            potential_impact="Prolonged service disruption",
            time_horizon="immediate",
            uncertainty_level=UncertaintyLevel.HIGH
        )
    ]
    
    rollback_benefits = [
        BenefitFactor(
            name="Quick Recovery",
            description="Fastest path to service restoration",
            benefit_type=BenefitType.OPERATIONAL,
            expected_value=0.9,
            probability_of_realization=0.8,
            time_to_realization="immediate",
            sustainability="temporary",
            uncertainty_level=UncertaintyLevel.LOW
        ),
        BenefitFactor(
            name="Customer Confidence",
            description="Demonstrates quick response capability",
            benefit_type=BenefitType.REPUTATIONAL,
            expected_value=0.6,
            probability_of_realization=0.7,
            time_to_realization="short_term",
            sustainability="medium_term",
            uncertainty_level=UncertaintyLevel.MODERATE
        )
    ]
    
    rollback_option = ResponseOption(
        name="Emergency Rollback",
        description="Immediately rollback to previous stable version",
        category="emergency_response",
        implementation_complexity="low",
        resource_requirements={"engineers": 3, "time_hours": 2, "budget": 5000},
        time_to_implement="immediate",
        risks=rollback_risks,
        benefits=rollback_benefits,
        dependencies=["backup_availability", "rollback_scripts_ready"],
        success_criteria=["service_restored_within_2_hours", "zero_additional_data_loss"]
    )
    
    # Option 2: Gradual Fix
    gradual_risks = [
        RiskFactor(
            name="Prolonged Outage",
            description="Extended service disruption during fix",
            category="operational",
            probability=0.4,
            impact_severity=RiskLevel.HIGH,
            potential_impact="Continued customer impact and revenue loss",
            time_horizon="short_term",
            uncertainty_level=UncertaintyLevel.MODERATE
        ),
        RiskFactor(
            name="Fix Complexity",
            description="Root cause fix may be more complex than anticipated",
            category="technical",
            probability=0.5,
            impact_severity=RiskLevel.MEDIUM,
            potential_impact="Delayed resolution and resource strain",
            time_horizon="short_term",
            uncertainty_level=UncertaintyLevel.HIGH
        )
    ]
    
    gradual_benefits = [
        BenefitFactor(
            name="Permanent Solution",
            description="Addresses root cause preventing future occurrences",
            benefit_type=BenefitType.STRATEGIC,
            expected_value=0.8,
            probability_of_realization=0.9,
            time_to_realization="medium_term",
            sustainability="permanent",
            uncertainty_level=UncertaintyLevel.LOW
        ),
        BenefitFactor(
            name="System Improvement",
            description="Opportunity to enhance system reliability",
            benefit_type=BenefitType.OPERATIONAL,
            expected_value=0.7,
            probability_of_realization=0.8,
            time_to_realization="medium_term",
            sustainability="permanent",
            uncertainty_level=UncertaintyLevel.MODERATE
        ),
        BenefitFactor(
            name="Cost Avoidance",
            description="Prevents future outage costs",
            benefit_type=BenefitType.FINANCIAL,
            expected_value=0.9,
            probability_of_realization=0.7,
            time_to_realization="long_term",
            sustainability="permanent",
            uncertainty_level=UncertaintyLevel.MODERATE
        )
    ]
    
    gradual_option = ResponseOption(
        name="Gradual Fix",
        description="Systematically diagnose and fix the root cause",
        category="systematic_response",
        implementation_complexity="high",
        resource_requirements={"engineers": 8, "time_hours": 8, "budget": 20000},
        time_to_implement="short_term",
        risks=gradual_risks,
        benefits=gradual_benefits,
        dependencies=["expert_team_availability", "diagnostic_tools", "stakeholder_patience"],
        success_criteria=["root_cause_identified", "permanent_fix_implemented", "system_reliability_improved"]
    )
    
    # Option 3: Hybrid Approach
    hybrid_risks = [
        RiskFactor(
            name="Coordination Complexity",
            description="Managing parallel rollback and fix efforts",
            category="operational",
            probability=0.3,
            impact_severity=RiskLevel.MEDIUM,
            potential_impact="Resource conflicts and communication issues",
            time_horizon="immediate",
            uncertainty_level=UncertaintyLevel.MODERATE
        ),
        RiskFactor(
            name="Resource Strain",
            description="High resource requirements for dual approach",
            category="resource",
            probability=0.4,
            impact_severity=RiskLevel.MEDIUM,
            potential_impact="Team burnout and quality issues",
            time_horizon="short_term",
            uncertainty_level=UncertaintyLevel.LOW
        )
    ]
    
    hybrid_benefits = [
        BenefitFactor(
            name="Best of Both",
            description="Quick recovery plus permanent solution",
            benefit_type=BenefitType.STRATEGIC,
            expected_value=0.9,
            probability_of_realization=0.8,
            time_to_realization="immediate",
            sustainability="permanent",
            uncertainty_level=UncertaintyLevel.MODERATE
        ),
        BenefitFactor(
            name="Stakeholder Satisfaction",
            description="Addresses both immediate and long-term concerns",
            benefit_type=BenefitType.STAKEHOLDER,
            expected_value=0.8,
            probability_of_realization=0.9,
            time_to_realization="short_term",
            sustainability="permanent",
            uncertainty_level=UncertaintyLevel.LOW
        )
    ]
    
    hybrid_option = ResponseOption(
        name="Hybrid Approach",
        description="Rollback immediately while fixing root cause in parallel",
        category="comprehensive_response",
        implementation_complexity="medium",
        resource_requirements={"engineers": 10, "time_hours": 6, "budget": 25000},
        time_to_implement="immediate",
        risks=hybrid_risks,
        benefits=hybrid_benefits,
        dependencies=["sufficient_team_size", "parallel_work_capability"],
        success_criteria=["immediate_service_restoration", "root_cause_fixed_within_24_hours"]
    )
    
    response_options = [rollback_option, gradual_option, hybrid_option]
    
    return {
        "crisis_id": "system_outage_2024_001",
        "response_options": response_options,
        "evaluation_criteria": {
            "risk_aversion": 0.3,  # Moderate risk tolerance due to crisis
            "benefit_focus": 0.7   # High focus on benefits/outcomes
        },
        "risk_tolerance": RiskLevel.MEDIUM,
        "time_pressure": "urgent"
    }


def print_response_option_summary(option: ResponseOption):
    """Print a summary of a response option"""
    print(f"📋 {option.name}")
    print(f"   Description: {option.description}")
    print(f"   Complexity: {option.implementation_complexity}")
    print(f"   Time to implement: {option.time_to_implement}")
    print(f"   Resources: {option.resource_requirements}")
    print(f"   Risks: {len(option.risks)} identified")
    print(f"   Benefits: {len(option.benefits)} identified")
    print()


def print_evaluation_results(evaluation):
    """Print detailed evaluation results"""
    print("📊 RISK-BENEFIT EVALUATION RESULTS")
    print("=" * 60)
    
    # Find recommended option
    recommended_option = None
    for option in evaluation.response_options:
        if option.id == evaluation.recommended_option:
            recommended_option = option
            break
    
    print(f"🎯 RECOMMENDED OPTION: {recommended_option.name if recommended_option else 'Unknown'}")
    print(f"📈 Confidence Score: {evaluation.confidence_score:.2f}")
    print()
    
    # Print trade-off analyses
    if evaluation.trade_off_analyses:
        print("⚖️  TRADE-OFF ANALYSES:")
        for i, analysis in enumerate(evaluation.trade_off_analyses, 1):
            print(f"   {i}. {analysis.recommendation}")
            print(f"      Confidence: {analysis.confidence_level:.2f}")
            print(f"      Rationale: {analysis.decision_rationale}")
            print()
    
    # Print mitigation strategies
    if evaluation.mitigation_plan:
        print("🛡️  MITIGATION STRATEGIES:")
        for i, strategy in enumerate(evaluation.mitigation_plan, 1):
            print(f"   {i}. {strategy.name}")
            print(f"      Effectiveness: {strategy.effectiveness_score:.2f}")
            print(f"      Success Probability: {strategy.success_probability:.2f}")
            print()
    
    # Print uncertainty factors
    if evaluation.uncertainty_factors:
        print("❓ UNCERTAINTY FACTORS:")
        for factor in evaluation.uncertainty_factors:
            print(f"   • {factor}")
        print()
    
    # Print monitoring requirements
    if evaluation.monitoring_requirements:
        print("👁️  MONITORING REQUIREMENTS:")
        for requirement in evaluation.monitoring_requirements:
            print(f"   • {requirement}")
        print()


def print_optimization_results(optimization_result):
    """Print benefit optimization results"""
    print("🚀 BENEFIT OPTIMIZATION RESULTS")
    print("=" * 60)
    
    print(f"🎯 Optimization Objective: {optimization_result.optimization_objective}")
    print(f"📈 Expected Improvement: {optimization_result.expected_improvement:.2f}")
    print(f"✅ Success Probability: {optimization_result.success_probability:.2f}")
    print()
    
    if optimization_result.optimization_strategies:
        print("📋 OPTIMIZATION STRATEGIES:")
        for strategy in optimization_result.optimization_strategies:
            print(f"   • {strategy}")
        print()
    
    if optimization_result.optimized_benefits:
        print("💎 OPTIMIZED BENEFITS:")
        for benefit in optimization_result.optimized_benefits:
            print(f"   • {benefit.name}: {benefit.expected_value:.2f} value, {benefit.probability_of_realization:.2f} probability")
        print()
    
    print("🔧 IMPLEMENTATION REQUIREMENTS:")
    for key, value in optimization_result.implementation_requirements.items():
        print(f"   • {key}: {value}")
    print()


async def demonstrate_risk_benefit_analysis():
    """Demonstrate the complete risk-benefit analysis system"""
    
    print("🎯 CRISIS LEADERSHIP EXCELLENCE - RISK-BENEFIT ANALYSIS DEMO")
    print("=" * 80)
    print()
    
    # Create sample scenario
    scenario = create_sample_crisis_scenario()
    
    # Print scenario overview
    print("📋 RESPONSE OPTIONS OVERVIEW:")
    print("-" * 40)
    for option in scenario["response_options"]:
        print_response_option_summary(option)
    
    # Initialize analyzer
    print("🔧 Initializing Risk-Benefit Analyzer...")
    analyzer = RiskBenefitAnalyzer()
    print("✅ Analyzer ready")
    print()
    
    # Perform evaluation
    print("⚡ Performing Risk-Benefit Evaluation...")
    start_time = datetime.now()
    
    evaluation = analyzer.evaluate_response_options(
        crisis_id=scenario["crisis_id"],
        response_options=scenario["response_options"],
        evaluation_criteria=scenario["evaluation_criteria"],
        risk_tolerance=scenario["risk_tolerance"],
        time_pressure=scenario["time_pressure"]
    )
    
    end_time = datetime.now()
    evaluation_time = (end_time - start_time).total_seconds()
    
    print(f"✅ Evaluation completed in {evaluation_time:.2f} seconds")
    print()
    
    # Print results
    print_evaluation_results(evaluation)
    
    # Demonstrate benefit optimization
    print("🚀 Performing Benefit Optimization...")
    optimization_result = analyzer.optimize_benefits(
        evaluation=evaluation,
        optimization_objective="maximize_total_value"
    )
    
    print_optimization_results(optimization_result)
    
    # Demonstrate additional analysis
    print("🔍 ADDITIONAL ANALYSIS CAPABILITIES")
    print("=" * 60)
    
    # Test mitigation strategy generation
    print("🛡️  Testing Mitigation Strategy Generation...")
    sample_risks = scenario["response_options"][0].risks
    mitigation_strategies = analyzer._generate_mitigation_strategies(sample_risks)
    
    print(f"Generated {len(mitigation_strategies)} mitigation strategies:")
    for strategy in mitigation_strategies[:3]:  # Show first 3
        print(f"   • {strategy.name} (Effectiveness: {strategy.effectiveness_score:.2f})")
    print()
    
    # Test confidence calculation
    print("📊 Testing Confidence Calculation...")
    for option in scenario["response_options"]:
        confidence = analyzer._calculate_confidence_score(option)
        print(f"   • {option.name}: {confidence:.2f} confidence")
    print()
    
    # Test uncertainty identification
    print("❓ Testing Uncertainty Factor Identification...")
    uncertainty_factors = analyzer._identify_uncertainty_factors(scenario["response_options"])
    print(f"Identified {len(uncertainty_factors)} uncertainty factors:")
    for factor in uncertainty_factors[:3]:  # Show first 3
        print(f"   • {factor}")
    print()
    
    print("🎉 DEMONSTRATION COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print()
    print("Key Capabilities Demonstrated:")
    print("✅ Rapid response option evaluation under uncertainty")
    print("✅ Risk assessment with multiple severity levels")
    print("✅ Benefit analysis with optimization strategies")
    print("✅ Trade-off analysis between competing options")
    print("✅ Mitigation strategy generation")
    print("✅ Confidence scoring and uncertainty management")
    print("✅ Monitoring requirement generation")
    print("✅ Benefit optimization with implementation guidance")
    print()
    print("This system enables ScrollIntel to make optimal crisis decisions")
    print("with superhuman speed and analytical capability!")


def demonstrate_api_integration():
    """Demonstrate API integration capabilities"""
    
    print("\n🌐 API INTEGRATION DEMONSTRATION")
    print("=" * 60)
    
    # Sample API request data
    api_request = {
        "crisis_id": "api_demo_crisis",
        "response_options": [
            {
                "name": "Quick Fix",
                "description": "Rapid response option",
                "category": "emergency",
                "implementation_complexity": "low",
                "resource_requirements": {"budget": 10000, "personnel": 5},
                "time_to_implement": "immediate",
                "risks": [
                    {
                        "name": "Implementation Risk",
                        "description": "Risk of incomplete fix",
                        "category": "operational",
                        "probability": 0.3,
                        "impact_severity": "medium",
                        "potential_impact": "Temporary solution may fail",
                        "time_horizon": "immediate",
                        "uncertainty_level": "moderate"
                    }
                ],
                "benefits": [
                    {
                        "name": "Quick Resolution",
                        "description": "Fast problem resolution",
                        "benefit_type": "operational",
                        "expected_value": 0.8,
                        "probability_of_realization": 0.7,
                        "time_to_realization": "immediate",
                        "sustainability": "temporary",
                        "uncertainty_level": "low"
                    }
                ],
                "dependencies": ["team_availability"],
                "success_criteria": ["issue_resolved_within_1_hour"]
            }
        ],
        "evaluation_criteria": {
            "risk_aversion": 0.4,
            "benefit_focus": 0.6
        },
        "risk_tolerance": "medium",
        "time_pressure": "urgent"
    }
    
    print("📤 Sample API Request Structure:")
    print(json.dumps(api_request, indent=2)[:500] + "...")
    print()
    
    print("📥 Expected API Response Structure:")
    sample_response = {
        "evaluation_id": "eval_12345",
        "crisis_id": "api_demo_crisis",
        "recommended_option": "option_id_1",
        "confidence_score": 0.85,
        "uncertainty_factors": ["High uncertainty in implementation timeline"],
        "trade_off_analyses": [
            {
                "option_a_id": "option_1",
                "option_b_id": "option_2",
                "recommendation": "Strong preference for Option 1",
                "confidence_level": 0.8,
                "decision_rationale": "Option 1 offers better risk-benefit balance"
            }
        ],
        "mitigation_plan": [
            {
                "name": "Risk Mitigation Strategy 1",
                "description": "Specific mitigation approach",
                "effectiveness_score": 0.7,
                "implementation_time": "short_term",
                "success_probability": 0.8
            }
        ],
        "monitoring_requirements": [
            "Monitor implementation progress",
            "Track risk indicators"
        ],
        "created_at": "2024-01-01T12:00:00Z"
    }
    
    print(json.dumps(sample_response, indent=2))
    print()
    
    print("🔗 Available API Endpoints:")
    print("   POST /api/v1/risk-benefit/evaluate - Evaluate response options")
    print("   POST /api/v1/risk-benefit/optimize-benefits - Optimize benefits")
    print("   POST /api/v1/risk-benefit/generate-mitigation - Generate mitigation strategies")
    print("   POST /api/v1/risk-benefit/trade-off-analysis - Perform trade-off analysis")
    print("   GET  /api/v1/risk-benefit/risk-levels - Get available risk levels")
    print("   GET  /api/v1/risk-benefit/benefit-types - Get available benefit types")
    print("   GET  /api/v1/risk-benefit/uncertainty-levels - Get uncertainty levels")


if __name__ == "__main__":
    # Run the demonstration
    asyncio.run(demonstrate_risk_benefit_analysis())
    
    # Show API integration
    demonstrate_api_integration()