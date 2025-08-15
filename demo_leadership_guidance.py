"""
Demo script for Crisis Leadership Guidance System
"""
import asyncio
from scrollintel.engines.leadership_guidance_engine import LeadershipGuidanceEngine
from scrollintel.models.leadership_guidance_models import (
    CrisisType, DecisionUrgency, DecisionContext
)

async def demo_leadership_guidance():
    """Demonstrate crisis leadership guidance capabilities"""
    print("🎯 Crisis Leadership Guidance System Demo")
    print("=" * 50)
    
    # Initialize engine
    engine = LeadershipGuidanceEngine()
    
    # Demo 1: Technical Outage Crisis
    print("\n📊 Demo 1: Technical Outage Crisis Leadership")
    print("-" * 40)
    
    tech_crisis_context = DecisionContext(
        crisis_id="outage_2024_001",
        crisis_type=CrisisType.TECHNICAL_OUTAGE,
        severity_level=8,
        stakeholders_affected=["customers", "technical_team", "executives", "partners"],
        time_pressure=DecisionUrgency.IMMEDIATE,
        available_information={
            "affected_systems": ["payment_gateway", "user_authentication"],
            "estimated_users_impacted": 50000,
            "initial_error_reports": 1200,
            "system_health_status": "critical"
        },
        resource_constraints=["weekend_staffing", "limited_senior_engineers"],
        regulatory_considerations=["payment_compliance", "data_protection"]
    )
    
    try:
        recommendation = engine.get_leadership_guidance(tech_crisis_context)
        
        print(f"Crisis Type: {tech_crisis_context.crisis_type.value}")
        print(f"Severity Level: {tech_crisis_context.severity_level}/10")
        print(f"Time Pressure: {tech_crisis_context.time_pressure.value}")
        print(f"\n🎯 Recommended Leadership Style: {recommendation.recommended_style.value.title()}")
        print(f"Confidence Score: {recommendation.confidence_score:.2f}")
        
        print(f"\n📋 Key Actions:")
        for i, action in enumerate(recommendation.key_actions, 1):
            print(f"  {i}. {action}")
        
        print(f"\n💬 Communication Strategy:")
        print(f"  {recommendation.communication_strategy}")
        
        print(f"\n👥 Stakeholder Priorities:")
        for i, stakeholder in enumerate(recommendation.stakeholder_priorities, 1):
            print(f"  {i}. {stakeholder.replace('_', ' ').title()}")
        
        print(f"\n⚠️ Risk Mitigation Steps:")
        for i, step in enumerate(recommendation.risk_mitigation_steps, 1):
            print(f"  {i}. {step}")
        
        print(f"\n📈 Success Metrics:")
        for i, metric in enumerate(recommendation.success_metrics, 1):
            print(f"  {i}. {metric}")
        
        print(f"\n💡 Rationale:")
        print(f"  {recommendation.rationale}")
        
    except Exception as e:
        print(f"❌ Error getting leadership guidance: {e}")
    
    # Demo 2: Security Breach Crisis
    print("\n\n🔒 Demo 2: Security Breach Crisis Leadership")
    print("-" * 40)
    
    security_crisis_context = DecisionContext(
        crisis_id="breach_2024_001",
        crisis_type=CrisisType.SECURITY_BREACH,
        severity_level=9,
        stakeholders_affected=["customers", "regulators", "legal_team", "security_team", "executives"],
        time_pressure=DecisionUrgency.IMMEDIATE,
        available_information={
            "breach_type": "unauthorized_access",
            "data_potentially_compromised": "customer_personal_info",
            "attack_vector": "phishing_email",
            "systems_affected": ["customer_database", "email_server"]
        },
        resource_constraints=["external_security_experts_needed", "legal_team_availability"],
        regulatory_considerations=["gdpr_compliance", "breach_notification_requirements", "regulatory_reporting"]
    )
    
    try:
        security_recommendation = engine.get_leadership_guidance(security_crisis_context)
        
        print(f"Crisis Type: {security_crisis_context.crisis_type.value}")
        print(f"Severity Level: {security_crisis_context.severity_level}/10")
        print(f"\n🎯 Recommended Leadership Style: {security_recommendation.recommended_style.value.title()}")
        print(f"Confidence Score: {security_recommendation.confidence_score:.2f}")
        
        print(f"\n📋 Key Actions:")
        for i, action in enumerate(security_recommendation.key_actions, 1):
            print(f"  {i}. {action}")
        
        print(f"\n💬 Communication Strategy:")
        print(f"  {security_recommendation.communication_strategy}")
        
        print(f"\n👥 Stakeholder Priorities:")
        for i, stakeholder in enumerate(security_recommendation.stakeholder_priorities, 1):
            print(f"  {i}. {stakeholder.replace('_', ' ').title()}")
        
    except Exception as e:
        print(f"❌ Error getting security crisis guidance: {e}")
    
    # Demo 3: Leadership Effectiveness Assessment
    print("\n\n📊 Demo 3: Leadership Effectiveness Assessment")
    print("-" * 40)
    
    # Simulate performance data from the technical outage response
    performance_data = {
        "decision_quality": 0.85,  # Good decision making
        "communication_effectiveness": 0.75,  # Decent communication
        "stakeholder_confidence": 0.70,  # Moderate confidence
        "team_morale": 0.80,  # Good team morale
        "resolution_speed": 0.90,  # Excellent resolution speed
        "additional_metrics": {
            "customer_satisfaction": 0.65,
            "media_sentiment": 0.60,
            "regulatory_compliance": 0.95
        }
    }
    
    try:
        assessment = engine.assess_leadership_effectiveness(
            "cto_001", "outage_2024_001", performance_data
        )
        
        print(f"Leader ID: {assessment.leader_id}")
        print(f"Crisis ID: {assessment.crisis_id}")
        print(f"Assessment Time: {assessment.assessment_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        print(f"\n📊 Performance Scores:")
        print(f"  Decision Quality: {assessment.decision_quality_score:.2f}")
        print(f"  Communication Effectiveness: {assessment.communication_effectiveness:.2f}")
        print(f"  Stakeholder Confidence: {assessment.stakeholder_confidence:.2f}")
        print(f"  Team Morale Impact: {assessment.team_morale_impact:.2f}")
        print(f"  Crisis Resolution Speed: {assessment.crisis_resolution_speed:.2f}")
        print(f"  Overall Effectiveness: {assessment.overall_effectiveness:.2f}")
        
        print(f"\n💪 Strengths:")
        for strength in assessment.strengths:
            print(f"  • {strength}")
        
        print(f"\n🎯 Improvement Areas:")
        for area in assessment.improvement_areas:
            print(f"  • {area}")
        
        print(f"\n🎓 Coaching Recommendations:")
        for recommendation in assessment.coaching_recommendations:
            print(f"  • {recommendation}")
        
        # Demo 4: Detailed Coaching Guidance
        print("\n\n🎓 Demo 4: Detailed Coaching Guidance")
        print("-" * 40)
        
        coaching_guidance = engine.provide_coaching_guidance(assessment)
        
        for i, guidance in enumerate(coaching_guidance, 1):
            print(f"\nCoaching Area {i}: {guidance.focus_area}")
            print(f"Current Performance: {guidance.current_performance:.2f}")
            print(f"Target Performance: {guidance.target_performance:.2f}")
            print(f"Timeline: {guidance.timeline}")
            
            print(f"\nImprovement Strategies:")
            for strategy in guidance.improvement_strategies:
                print(f"  • {strategy}")
            
            print(f"\nPractice Exercises:")
            for exercise in guidance.practice_exercises:
                print(f"  • {exercise}")
            
            print(f"\nSuccess Indicators:")
            for indicator in guidance.success_indicators:
                print(f"  • {indicator}")
            
            print(f"\nResources:")
            for resource in guidance.resources:
                print(f"  • {resource}")
        
    except Exception as e:
        print(f"❌ Error in leadership assessment: {e}")
    
    # Demo 5: Best Practices Lookup
    print("\n\n📚 Demo 5: Crisis Leadership Best Practices")
    print("-" * 40)
    
    try:
        tech_practices = engine._get_relevant_practices(CrisisType.TECHNICAL_OUTAGE)
        
        print(f"Best Practices for {CrisisType.TECHNICAL_OUTAGE.value}:")
        
        for practice in tech_practices:
            print(f"\n📋 {practice.practice_name}")
            print(f"Description: {practice.description}")
            print(f"Effectiveness Score: {practice.effectiveness_score:.2f}")
            
            print(f"\nImplementation Steps:")
            for step in practice.implementation_steps:
                print(f"  • {step}")
            
            print(f"\nSuccess Indicators:")
            for indicator in practice.success_indicators:
                print(f"  • {indicator}")
            
            print(f"\nCommon Pitfalls:")
            for pitfall in practice.common_pitfalls:
                print(f"  ⚠️ {pitfall}")
        
    except Exception as e:
        print(f"❌ Error getting best practices: {e}")
    
    print("\n" + "=" * 50)
    print("✅ Crisis Leadership Guidance Demo Complete!")
    print("\nKey Capabilities Demonstrated:")
    print("• Crisis-specific leadership style recommendations")
    print("• Actionable guidance based on context and urgency")
    print("• Comprehensive leadership effectiveness assessment")
    print("• Personalized coaching and improvement strategies")
    print("• Best practices database for different crisis types")
    print("• Risk mitigation and stakeholder management guidance")

if __name__ == "__main__":
    asyncio.run(demo_leadership_guidance())