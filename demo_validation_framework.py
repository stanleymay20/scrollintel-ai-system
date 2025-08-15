"""
Demo script for Innovation Validation Framework.

This script demonstrates the capabilities of the validation framework
including innovation validation, methodology selection, and report generation.
"""

import asyncio
import json
from datetime import datetime
from typing import List

from scrollintel.engines.validation_framework import ValidationFramework
from scrollintel.engines.impact_assessment_framework import ImpactAssessmentFramework
from scrollintel.engines.success_prediction_system import SuccessPredictionSystem
from scrollintel.models.validation_models import (
    Innovation, ValidationRequest, ValidationType, ValidationStatus, ImpactLevel, SuccessProbability
)


async def create_sample_innovations() -> List[Innovation]:
    """Create sample innovations for demonstration."""
    innovations = [
        Innovation(
            title="AI-Powered Healthcare Assistant",
            description="An AI system that assists healthcare professionals with diagnosis and treatment recommendations",
            category="Healthcare Technology",
            domain="Healthcare",
            technology_stack=["Python", "TensorFlow", "React", "PostgreSQL", "AWS"],
            target_market="Healthcare providers",
            problem_statement="Healthcare professionals need better tools for accurate diagnosis",
            proposed_solution="AI-powered assistant that analyzes patient data and provides recommendations",
            unique_value_proposition="Reduces diagnostic errors by 40% and improves treatment outcomes",
            competitive_advantages=["Advanced AI algorithms", "Integration with existing systems"],
            estimated_timeline="18 months",
            estimated_cost=2000000.0,
            potential_revenue=10000000.0,
            risk_factors=["Regulatory approval", "Data privacy concerns"],
            success_metrics=["Diagnostic accuracy", "User adoption rate", "Patient outcomes"]
        ),
        Innovation(
            title="Quantum Computing Cloud Platform",
            description="Cloud-based quantum computing platform for enterprise applications",
            category="Quantum Technology",
            domain="Technology",
            technology_stack=["Quantum circuits", "Python", "Kubernetes", "AWS"],
            target_market="Enterprise customers",
            problem_statement="Limited access to quantum computing resources",
            proposed_solution="Cloud platform providing quantum computing as a service",
            unique_value_proposition="First commercially viable quantum cloud platform",
            competitive_advantages=["Proprietary quantum algorithms", "Scalable architecture"],
            estimated_timeline="36 months",
            estimated_cost=50000000.0,
            potential_revenue=500000000.0,
            risk_factors=["Technology maturity", "High development costs", "Competition"],
            success_metrics=["Platform adoption", "Quantum advantage demonstrations", "Revenue growth"]
        ),
        Innovation(
            title="Sustainable Energy Storage System",
            description="Advanced battery technology for renewable energy storage",
            category="Clean Technology",
            domain="Energy",
            technology_stack=["Advanced materials", "IoT sensors", "Machine learning"],
            target_market="Renewable energy providers",
            problem_statement="Need for efficient and sustainable energy storage solutions",
            proposed_solution="Next-generation battery technology with 10x capacity",
            unique_value_proposition="Highest energy density with minimal environmental impact",
            competitive_advantages=["Proprietary materials", "Long lifespan", "Recyclable"],
            estimated_timeline="24 months",
            estimated_cost=15000000.0,
            potential_revenue=100000000.0,
            risk_factors=["Material sourcing", "Manufacturing scale-up"],
            success_metrics=["Energy density", "Cycle life", "Cost per kWh"]
        )
    ]
    
    return innovations


async def demonstrate_validation_framework():
    """Demonstrate the validation framework capabilities."""
    print("🚀 Innovation Validation Framework Demo")
    print("=" * 50)
    
    # Initialize validation framework
    print("\n1. Initializing Validation Framework...")
    framework = ValidationFramework()
    await framework.start()
    
    status = framework.get_status()
    print(f"   ✅ Framework initialized successfully")
    print(f"   📊 Methodologies loaded: {status['methodologies_loaded']}")
    print(f"   📋 Criteria loaded: {status['criteria_loaded']}")
    
    # Create sample innovations
    print("\n2. Creating Sample Innovations...")
    innovations = await create_sample_innovations()
    print(f"   ✅ Created {len(innovations)} sample innovations")
    
    for i, innovation in enumerate(innovations, 1):
        print(f"   {i}. {innovation.title}")
        print(f"      Domain: {innovation.domain}")
        print(f"      Estimated Cost: ${innovation.estimated_cost:,.0f}")
        print(f"      Potential Revenue: ${innovation.potential_revenue:,.0f}")
    
    # Demonstrate validation methodologies
    print("\n3. Available Validation Methodologies...")
    for methodology_id, methodology in framework.validation_methodologies.items():
        print(f"   📋 {methodology.name}")
        print(f"      Description: {methodology.description}")
        print(f"      Validation Types: {len(methodology.validation_types)}")
        print(f"      Accuracy Rate: {methodology.accuracy_rate:.1%}")
        print(f"      Estimated Duration: {methodology.estimated_duration} hours")
        print()
    
    # Validate each innovation
    print("\n4. Validating Innovations...")
    validation_reports = []
    
    for i, innovation in enumerate(innovations, 1):
        print(f"\n   Validating Innovation {i}: {innovation.title}")
        print(f"   {'─' * 60}")
        
        # Perform validation
        report = await framework.validate_innovation(innovation)
        validation_reports.append(report)
        
        # Display results
        print(f"   📊 Overall Score: {report.overall_score:.3f}")
        print(f"   🎯 Result: {report.overall_result.value.replace('_', ' ').title()}")
        print(f"   🔍 Confidence: {report.confidence_level:.1%}")
        
        # Show top validation scores
        print(f"   📈 Top Validation Scores:")
        sorted_scores = sorted(report.validation_scores, key=lambda x: x.score, reverse=True)
        for score in sorted_scores[:3]:
            print(f"      • {score.criteria_id}: {score.score:.3f} (confidence: {score.confidence:.1%})")
        
        # Show strengths and weaknesses
        if report.strengths:
            print(f"   💪 Key Strengths:")
            for strength in report.strengths[:3]:
                print(f"      • {strength}")
        
        if report.weaknesses:
            print(f"   ⚠️  Areas for Improvement:")
            for weakness in report.weaknesses[:3]:
                print(f"      • {weakness}")
        
        # Show recommendations
        if report.recommendations:
            print(f"   💡 Top Recommendations:")
            for recommendation in report.recommendations[:3]:
                print(f"      • {recommendation}")
    
    # Demonstrate validation request processing
    print("\n5. Processing Validation Requests...")
    
    # Create validation request
    validation_request = ValidationRequest(
        innovation_id=innovations[0].id,
        validation_types=[
            ValidationType.TECHNICAL_FEASIBILITY,
            ValidationType.MARKET_VIABILITY,
            ValidationType.RISK_ASSESSMENT
        ],
        priority="high",
        requester="demo_user"
    )
    
    print(f"   📝 Created validation request: {validation_request.id}")
    print(f"   🎯 Validation types: {len(validation_request.validation_types)}")
    
    # Process request
    request_report = await framework.process(
        validation_request,
        {"innovation": innovations[0]}
    )
    
    print(f"   ✅ Request processed successfully")
    print(f"   📊 Report ID: {request_report.id}")
    print(f"   🎯 Overall Score: {request_report.overall_score:.3f}")
    
    # Demonstrate methodology selection
    print("\n6. Methodology Selection Demo...")
    
    for innovation in innovations[:2]:
        print(f"\n   Innovation: {innovation.title}")
        
        # Select methodology
        methodology = await framework.select_validation_methodology(
            innovation,
            [ValidationType.TECHNICAL_FEASIBILITY, ValidationType.MARKET_VIABILITY]
        )
        
        print(f"   📋 Selected Methodology: {methodology.name}")
        print(f"   🎯 Accuracy Rate: {methodology.accuracy_rate:.1%}")
        print(f"   ⏱️  Estimated Duration: {methodology.estimated_duration} hours")
    
    # Show validation analytics
    print("\n7. Validation Analytics...")
    
    total_validations = len(framework.historical_validations)
    if total_validations > 0:
        avg_score = sum(r.overall_score for r in framework.historical_validations) / total_validations
        
        # Result distribution
        result_counts = {}
        for report in framework.historical_validations:
            result = report.overall_result.value
            result_counts[result] = result_counts.get(result, 0) + 1
        
        print(f"   📊 Total Validations: {total_validations}")
        print(f"   📈 Average Score: {avg_score:.3f}")
        print(f"   📋 Result Distribution:")
        for result, count in result_counts.items():
            percentage = (count / total_validations) * 100
            print(f"      • {result.replace('_', ' ').title()}: {count} ({percentage:.1f}%)")
    
    # Demonstrate validation criteria
    print("\n8. Validation Criteria Overview...")
    
    for validation_type, criteria_list in framework.validation_criteria.items():
        if criteria_list:  # Only show types with criteria
            print(f"\n   📋 {validation_type.value.replace('_', ' ').title()}:")
            for criteria in criteria_list:
                print(f"      • {criteria.name} (weight: {criteria.weight:.2f})")
                print(f"        {criteria.description}")
    
    # Performance metrics
    print("\n9. Framework Performance Metrics...")
    metrics = framework.get_metrics()
    
    print(f"   🚀 Engine Status: {metrics['status']}")
    print(f"   📊 Usage Count: {metrics['usage_count']}")
    print(f"   ❌ Error Count: {metrics['error_count']}")
    print(f"   📈 Error Rate: {metrics['error_rate']:.1%}")
    print(f"   🕒 Last Used: {metrics['last_used'] or 'Never'}")
    
    # Summary
    print("\n10. Demo Summary")
    print("=" * 50)
    print(f"✅ Successfully validated {len(innovations)} innovations")
    print(f"📊 Generated {len(validation_reports)} comprehensive reports")
    print(f"🎯 Average validation score: {sum(r.overall_score for r in validation_reports) / len(validation_reports):.3f}")
    print(f"💡 Total recommendations generated: {sum(len(r.recommendations) for r in validation_reports)}")
    print(f"📈 Framework ready for production use")
    
    # Demonstrate Impact Assessment
    print("\n11. Impact Assessment Demo...")
    impact_framework = ImpactAssessmentFramework()
    await impact_framework.start()
    
    for i, innovation in enumerate(innovations[:2], 1):
        print(f"\n   Assessing Impact for Innovation {i}: {innovation.title}")
        print(f"   {'─' * 60}")
        
        # Perform impact assessment
        assessment = await impact_framework.assess_innovation_impact(innovation)
        
        # Display impact results
        print(f"   🎯 Market Impact: {assessment.market_impact.value.replace('_', ' ').title()}")
        print(f"   🔧 Technical Impact: {assessment.technical_impact.value.replace('_', ' ').title()}")
        print(f"   💼 Business Impact: {assessment.business_impact.value.replace('_', ' ').title()}")
        print(f"   👥 Social Impact: {assessment.social_impact.value.replace('_', ' ').title()}")
        print(f"   🌱 Environmental Impact: {assessment.environmental_impact.value.replace('_', ' ').title()}")
        
        # Show key metrics
        print(f"   📊 Market Size: ${assessment.market_size:,.0f}")
        print(f"   💰 Revenue Potential: ${assessment.revenue_potential:,.0f}")
        print(f"   👷 Job Creation: {assessment.job_creation_potential} jobs")
        print(f"   🚀 Disruption Potential: {assessment.disruption_potential:.1%}")
        print(f"   📈 Scalability Factor: {assessment.scalability_factor:.1%}")
        print(f"   ⏰ Time to Market: {assessment.time_to_market} months")
        
        # Show stakeholder impact
        print(f"   🤝 Stakeholder Impact:")
        for stakeholder, impact in assessment.stakeholder_impact.items():
            print(f"      • {stakeholder.title()}: {impact}")
    
    await impact_framework.stop()
    
    # Demonstrate Success Prediction
    print("\n12. Success Prediction Demo...")
    prediction_system = SuccessPredictionSystem()
    await prediction_system.start()
    
    for i, innovation in enumerate(innovations[:2], 1):
        print(f"\n   Predicting Success for Innovation {i}: {innovation.title}")
        print(f"   {'─' * 60}")
        
        # Perform success prediction
        prediction = await prediction_system.predict_innovation_success(innovation)
        
        # Display prediction results
        print(f"   🎯 Overall Success Probability: {prediction.overall_probability:.1%}")
        print(f"   📊 Probability Category: {prediction.probability_category.value.replace('_', ' ').title()}")
        print(f"   🔧 Technical Success: {prediction.technical_success_probability:.1%}")
        print(f"   📈 Market Success: {prediction.market_success_probability:.1%}")
        print(f"   💰 Financial Success: {prediction.financial_success_probability:.1%}")
        print(f"   ⏰ Timeline Success: {prediction.timeline_success_probability:.1%}")
        
        # Show key success factors
        print(f"   💪 Key Success Factors:")
        for factor in prediction.key_success_factors[:3]:
            print(f"      • {factor}")
        
        # Show critical risks
        print(f"   ⚠️  Critical Risks:")
        for risk in prediction.critical_risks[:3]:
            print(f"      • {risk}")
        
        # Show success scenarios
        print(f"   🚀 Success Scenarios:")
        for scenario in prediction.success_scenarios:
            print(f"      • {scenario['scenario']}: {scenario['probability']:.1%} - {scenario['description']}")
        
        # Show optimization opportunities
        print(f"   💡 Top Optimization Opportunities:")
        for opportunity in prediction.optimization_opportunities[:3]:
            print(f"      • {opportunity}")
        
        # Show model performance
        print(f"   📊 Model Accuracy: {prediction.model_accuracy:.1%}")
        print(f"   📋 Data Quality Score: {prediction.data_quality_score:.1%}")
    
    await prediction_system.stop()
    
    print("\n🎉 Innovation Validation Framework Demo Complete!")
    
    # Cleanup
    await framework.stop()


async def demonstrate_advanced_features():
    """Demonstrate advanced validation framework features."""
    print("\n🔬 Advanced Features Demo")
    print("=" * 30)
    
    framework = ValidationFramework()
    await framework.start()
    
    # Create a complex innovation
    complex_innovation = Innovation(
        title="Autonomous Vehicle AI System",
        description="Advanced AI system for fully autonomous vehicles",
        category="Automotive AI",
        domain="Transportation",
        technology_stack=["Deep Learning", "Computer Vision", "LIDAR", "5G", "Edge Computing"],
        target_market="Automotive manufacturers",
        problem_statement="Need for safe and reliable autonomous driving technology",
        proposed_solution="Multi-modal AI system with advanced perception and decision-making",
        unique_value_proposition="Achieves Level 5 autonomy with 99.99% safety record",
        competitive_advantages=["Proprietary algorithms", "Real-time processing", "Multi-sensor fusion"],
        estimated_timeline="48 months",
        estimated_cost=100000000.0,
        potential_revenue=1000000000.0,
        risk_factors=["Regulatory approval", "Safety concerns", "Technology complexity", "Competition"],
        success_metrics=["Safety performance", "Regulatory approval", "Market adoption"]
    )
    
    print(f"🚗 Complex Innovation: {complex_innovation.title}")
    
    # Test different validation type combinations
    validation_combinations = [
        [ValidationType.TECHNICAL_FEASIBILITY],
        [ValidationType.MARKET_VIABILITY],
        [ValidationType.RISK_ASSESSMENT],
        [ValidationType.TECHNICAL_FEASIBILITY, ValidationType.MARKET_VIABILITY],
        [ValidationType.TECHNICAL_FEASIBILITY, ValidationType.RISK_ASSESSMENT],
        list(ValidationType)  # All types
    ]
    
    print("\n📊 Testing Different Validation Combinations...")
    
    for i, validation_types in enumerate(validation_combinations, 1):
        print(f"\n   Combination {i}: {len(validation_types)} validation type(s)")
        
        report = await framework.validate_innovation(complex_innovation, validation_types)
        
        print(f"   📈 Score: {report.overall_score:.3f}")
        print(f"   🎯 Result: {report.overall_result.value}")
        print(f"   🔍 Confidence: {report.confidence_level:.1%}")
        print(f"   📋 Criteria Evaluated: {len(report.validation_scores)}")
    
    await framework.stop()


if __name__ == "__main__":
    print("🚀 Starting Innovation Validation Framework Demo...")
    
    # Run main demo
    asyncio.run(demonstrate_validation_framework())
    
    # Run advanced features demo
    asyncio.run(demonstrate_advanced_features())
    
    print("\n✨ All demos completed successfully!")