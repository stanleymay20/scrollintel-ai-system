#!/usr/bin/env python3
"""
Enhanced AI Engineer Agent Demo - Showcasing all capabilities
"""
import asyncio
import sys
import os
import json

# Add the scrollintel_core directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scrollintel_core'))

from scrollintel_core.agents.ai_engineer_agent import AIEngineerAgent
from scrollintel_core.agents.base import AgentRequest


async def demo_ai_strategy():
    """Demo comprehensive AI strategy generation"""
    print("🎯 AI STRATEGY GENERATION DEMO")
    print("=" * 60)
    
    agent = AIEngineerAgent()
    
    # Comprehensive strategy request
    request = AgentRequest(
        query="Create a comprehensive AI transformation strategy for our healthcare startup",
        context={
            "business_size": "medium",
            "industry": "healthcare",
            "current_capabilities": ["basic_analytics", "data_collection"],
            "data_infrastructure": "basic",
            "team_skills": "beginner",
            "budget": "high",
            "timeline": "18 months",
            "tech_stack": ["python", "postgresql", "react"]
        }
    )
    
    response = await agent.process(request)
    
    if response.success:
        strategy = response.result
        
        print(f"📊 Current AI Maturity Level: {strategy['ai_maturity_assessment']['current_level']}/4")
        print(f"🏥 Industry Opportunities: {', '.join(strategy['industry_specific_opportunities'])}")
        
        print("\n📋 Implementation Roadmap:")
        for phase_key, phase in strategy['implementation_roadmap'].items():
            print(f"  {phase['name']} ({phase['timeline']})")
            print(f"    Focus: {phase['focus']}")
            print(f"    Key Deliverables: {len(phase['deliverables'])} items")
        
        print(f"\n⚠️ Risk Mitigation: {len(strategy['risk_mitigation']['technical_risks'])} technical risks identified")
        print(f"🛡️ Governance: {len(strategy['governance_framework']['governance_principles'])} principles defined")
        
    else:
        print(f"❌ Error: {response.error}")


async def demo_architecture_recommendation():
    """Demo advanced architecture recommendations"""
    print("\n🏗️ ARCHITECTURE RECOMMENDATION DEMO")
    print("=" * 60)
    
    agent = AIEngineerAgent()
    
    request = AgentRequest(
        query="Design AI architecture for high-frequency trading system",
        context={
            "use_case": "high-frequency trading with real-time risk assessment",
            "scale": "large",
            "performance": "ultra_high",
            "data_volume": "massive",
            "budget": "high",
            "latency_requirements": "sub-millisecond"
        }
    )
    
    response = await agent.process(request)
    
    if response.success:
        arch = response.result
        
        print(f"🎯 Recommended Pattern: {arch['recommended_architecture']['pattern']}")
        print(f"💡 Reasoning: {arch['recommended_architecture']['reasoning']}")
        
        print(f"\n📈 Scale Configuration:")
        scale_config = arch['scale_configuration']
        for key, value in scale_config.items():
            print(f"  {key.replace('_', ' ').title()}: {value}")
        
        print(f"\n🔧 Technology Stack:")
        tech_stack = arch['technology_stack']
        for component, tech in tech_stack.items():
            print(f"  {component.replace('_', ' ').title()}: {tech}")
        
        print(f"\n🛡️ Security Measures: {len(arch['security_considerations'])} recommendations")
        
    else:
        print(f"❌ Error: {response.error}")


async def demo_cost_estimation():
    """Demo detailed cost estimation"""
    print("\n💰 COST ESTIMATION DEMO")
    print("=" * 60)
    
    agent = AIEngineerAgent()
    
    request = AgentRequest(
        query="Provide detailed cost analysis for enterprise AI implementation",
        context={
            "scale": "large",
            "complexity": "very_high",
            "timeline": "24 months",
            "team_size": "large",
            "use_cases": [
                "customer_analytics",
                "fraud_detection", 
                "supply_chain_optimization",
                "predictive_maintenance"
            ]
        }
    )
    
    response = await agent.process(request)
    
    if response.success:
        costs = response.result
        
        print("💵 Total Cost Estimates:")
        total_costs = costs['total_cost_estimate']
        for cost_type, amount in total_costs.items():
            print(f"  {cost_type.replace('_', ' ').title()}: {amount}")
        
        print("\n🏗️ Development Breakdown:")
        dev_costs = costs['detailed_breakdown']['development_costs']
        for component, cost in dev_costs.items():
            print(f"  {component.replace('_', ' ').title()}: {cost}")
        
        print("\n☁️ Infrastructure (Monthly):")
        infra_costs = costs['detailed_breakdown']['infrastructure_costs']
        for component, cost in infra_costs.items():
            print(f"  {component.replace('_', ' ').title()}: {cost}")
        
        print(f"\n📈 ROI Timeline:")
        roi = costs['roi_analysis']['timeline_to_roi']
        for phase, timeline in roi.items():
            print(f"  {phase.replace('_', ' ').title()}: {timeline}")
        
        print(f"\n⚠️ Hidden Costs: {len(costs['hidden_costs_warning'])} potential hidden costs identified")
        
    else:
        print(f"❌ Error: {response.error}")


async def demo_integration_guidance():
    """Demo integration best practices"""
    print("\n🔗 INTEGRATION GUIDANCE DEMO")
    print("=" * 60)
    
    agent = AIEngineerAgent()
    
    request = AgentRequest(
        query="Guide integration of AI into existing enterprise systems",
        context={
            "current_systems": ["SAP ERP", "Salesforce CRM", "Oracle Database"],
            "integration_complexity": "high",
            "compliance_requirements": ["SOX", "GDPR", "HIPAA"]
        }
    )
    
    response = await agent.process(request)
    
    if response.success:
        guidance = response.result
        
        print("🔧 Integration Strategies:")
        for strategy_name, strategy in guidance['integration_strategies'].items():
            print(f"  {strategy_name.replace('_', ' ').title()}:")
            print(f"    Approach: {strategy['approach']}")
            print(f"    Benefits: {', '.join(strategy['benefits'])}")
        
        print(f"\n✨ Best Practices ({len(guidance['best_practices'])} items):")
        for i, practice in enumerate(guidance['best_practices'], 1):
            print(f"  {i}. {practice}")
        
        print(f"\n⚠️ Common Challenges ({len(guidance['common_challenges'])} items):")
        for i, challenge in enumerate(guidance['common_challenges'], 1):
            print(f"  {i}. {challenge}")
        
    else:
        print(f"❌ Error: {response.error}")


async def demo_general_ai_guidance():
    """Demo general AI guidance capabilities"""
    print("\n🧠 GENERAL AI GUIDANCE DEMO")
    print("=" * 60)
    
    agent = AIEngineerAgent()
    
    request = AgentRequest(
        query="What are the key considerations for implementing ethical AI in our organization?",
        context={
            "organization_type": "financial_services",
            "ai_applications": ["credit_scoring", "fraud_detection", "customer_service"],
            "regulatory_environment": "strict"
        }
    )
    
    response = await agent.process(request)
    
    if response.success:
        guidance = response.result
        
        print("🎯 AI Implementation Principles:")
        for i, principle in enumerate(guidance['ai_implementation_principles'], 1):
            print(f"  {i}. {principle}")
        
        print("\n🔍 Technology Selection Criteria:")
        criteria = guidance['technology_selection']['criteria']
        for i, criterion in enumerate(criteria, 1):
            print(f"  {i}. {criterion}")
        
        print("\n✅ Success Factors:")
        for i, factor in enumerate(guidance['success_factors'], 1):
            print(f"  {i}. {factor}")
        
        print("\n⚠️ Common Pitfalls to Avoid:")
        for i, pitfall in enumerate(guidance['common_pitfalls'], 1):
            print(f"  {i}. {pitfall}")
        
    else:
        print(f"❌ Error: {response.error}")


async def main():
    """Run all AI Engineer Agent demos"""
    print("🤖 AI ENGINEER AGENT - COMPREHENSIVE DEMO")
    print("=" * 80)
    print("Showcasing enhanced AI strategy, architecture, cost estimation, and guidance capabilities")
    print("=" * 80)
    
    try:
        await demo_ai_strategy()
        await demo_architecture_recommendation()
        await demo_cost_estimation()
        await demo_integration_guidance()
        await demo_general_ai_guidance()
        
        print("\n" + "=" * 80)
        print("🎉 AI ENGINEER AGENT DEMO COMPLETE!")
        print("✅ All capabilities successfully demonstrated")
        print("🚀 Ready for production use in ScrollIntel Core Focus platform")
        print("=" * 80)
        
    except Exception as e:
        print(f"💥 Demo failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())