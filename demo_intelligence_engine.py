"""
Intelligence and Decision Engine Demo

This script demonstrates the capabilities of the Intelligence and Decision Engine,
showcasing real-time business decision making, risk assessment, ML predictions,
and knowledge graph queries.
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, Any, List

from scrollintel.engines.intelligence_engine import (
    IntelligenceEngine, BusinessContext, DecisionOption, Decision,
    DecisionConfidence, RiskLevel
)


async def demo_business_decision_making():
    """Demonstrate comprehensive business decision making"""
    print("🧠 Intelligence and Decision Engine Demo")
    print("=" * 60)
    
    # Initialize the intelligence engine
    print("\n1. Initializing Intelligence Engine...")
    engine = IntelligenceEngine()
    await engine.start()
    print("✅ Intelligence Engine initialized successfully")
    
    try:
        # Demo 1: Strategic Business Decision
        await demo_strategic_decision(engine)
        
        # Demo 2: Risk Assessment
        await demo_risk_assessment(engine)
        
        # Demo 3: Knowledge Graph Query
        await demo_knowledge_query(engine)
        
        # Demo 4: Learning from Outcomes
        await demo_outcome_learning(engine)
        
        # Demo 5: Performance Metrics
        await demo_performance_metrics(engine)
        
    finally:
        # Clean up
        await engine.stop()
        print("\n✅ Intelligence Engine demo completed successfully")


async def demo_strategic_decision(engine: IntelligenceEngine):
    """Demonstrate strategic business decision making"""
    print("\n" + "="*60)
    print("📊 DEMO 1: Strategic Business Decision Making")
    print("="*60)
    
    # Create a realistic business context
    context = BusinessContext(
        industry="technology",
        business_unit="cloud_services",
        stakeholders=["ceo", "cto", "vp_engineering", "vp_sales", "board"],
        constraints=[
            {"type": "budget", "value": 5000000, "timeline": "12_months"},
            {"type": "team_capacity", "value": 50, "unit": "engineers"},
            {"type": "market_window", "value": 18, "unit": "months"},
            {"type": "competitive_pressure", "level": "high"}
        ],
        objectives=[
            {
                "type": "revenue_growth",
                "target": 25000000,
                "timeline": "24_months",
                "priority": "critical"
            },
            {
                "type": "market_share",
                "target": 0.15,
                "timeline": "18_months",
                "priority": "high"
            },
            {
                "type": "customer_acquisition",
                "target": 1000,
                "timeline": "12_months",
                "priority": "medium"
            }
        ],
        current_state={
            "annual_revenue": 15000000,
            "market_share": 0.08,
            "customer_count": 350,
            "team_size": 45,
            "product_maturity": "growth_stage",
            "competitive_position": "challenger"
        },
        historical_data={
            "revenue_growth_rate": 0.35,
            "customer_churn_rate": 0.08,
            "product_launches": 3,
            "success_rate": 0.67,
            "average_deal_size": 42000
        },
        time_horizon="medium_term",
        budget_constraints={
            "total": 5000000,
            "development": 2500000,
            "marketing": 1500000,
            "operations": 1000000
        },
        regulatory_requirements=["gdpr", "sox", "iso_27001", "pci_dss"]
    )
    
    print(f"📋 Business Context:")
    print(f"   Industry: {context.industry}")
    print(f"   Business Unit: {context.business_unit}")
    print(f"   Current Revenue: ${context.current_state['annual_revenue']:,}")
    print(f"   Market Share: {context.current_state['market_share']:.1%}")
    print(f"   Budget Available: ${context.budget_constraints['total']:,}")
    
    # Create strategic decision options
    options = [
        DecisionOption(
            id="ai_platform_expansion",
            name="AI Platform Expansion",
            description="Develop comprehensive AI/ML platform to compete with major cloud providers",
            expected_outcomes={
                "revenue_potential": 40000000,
                "market_differentiation": 0.9,
                "customer_retention_improvement": 0.25,
                "new_market_segments": 3
            },
            costs={
                "ai_research_development": 1800000,
                "infrastructure_scaling": 1200000,
                "talent_acquisition": 800000,
                "go_to_market": 600000,
                "partnerships": 400000
            },
            benefits={
                "premium_pricing": 8000000,
                "customer_lifetime_value": 12000000,
                "competitive_moat": 15000000,
                "data_monetization": 5000000
            },
            risks=[
                {
                    "type": "technical_complexity",
                    "probability": 0.4,
                    "impact": 2000000,
                    "description": "AI development complexity may exceed estimates"
                },
                {
                    "type": "competitive_response",
                    "probability": 0.7,
                    "impact": 5000000,
                    "description": "Major competitors may accelerate AI offerings"
                },
                {
                    "type": "talent_shortage",
                    "probability": 0.5,
                    "impact": 1500000,
                    "description": "Difficulty hiring specialized AI talent"
                }
            ],
            implementation_complexity=8.5,
            time_to_implement=450,
            resource_requirements={
                "ai_engineers": 20,
                "data_scientists": 12,
                "infrastructure_engineers": 8,
                "product_managers": 4,
                "budget": 4800000,
                "compute_resources": "high",
                "data_partnerships": True
            }
        ),
        
        DecisionOption(
            id="vertical_specialization",
            name="Vertical Market Specialization",
            description="Focus on specific industry verticals with tailored solutions",
            expected_outcomes={
                "revenue_potential": 22000000,
                "market_penetration": 0.3,
                "customer_stickiness": 0.85,
                "pricing_power": 0.4
            },
            costs={
                "vertical_solution_development": 1000000,
                "industry_expertise": 600000,
                "sales_specialization": 500000,
                "compliance_certification": 400000,
                "marketing_repositioning": 300000
            },
            benefits={
                "higher_margins": 4000000,
                "reduced_competition": 3000000,
                "customer_intimacy": 2500000,
                "referral_network": 1500000
            },
            risks=[
                {
                    "type": "market_concentration",
                    "probability": 0.3,
                    "impact": 3000000,
                    "description": "Over-dependence on specific verticals"
                },
                {
                    "type": "industry_downturn",
                    "probability": 0.25,
                    "impact": 4000000,
                    "description": "Economic downturn in target industries"
                }
            ],
            implementation_complexity=6.0,
            time_to_implement=270,
            resource_requirements={
                "industry_specialists": 8,
                "solution_architects": 6,
                "compliance_experts": 4,
                "vertical_sales": 10,
                "budget": 2800000
            }
        ),
        
        DecisionOption(
            id="global_expansion",
            name="International Market Expansion",
            description="Expand to European and Asian markets with localized offerings",
            expected_outcomes={
                "revenue_potential": 35000000,
                "geographic_diversification": 0.6,
                "brand_recognition": 0.7,
                "market_learning": 0.8
            },
            costs={
                "international_setup": 800000,
                "localization": 600000,
                "regulatory_compliance": 700000,
                "local_partnerships": 500000,
                "marketing_launch": 900000
            },
            benefits={
                "market_diversification": 6000000,
                "revenue_growth": 20000000,
                "risk_distribution": 3000000,
                "learning_opportunities": 2000000
            },
            risks=[
                {
                    "type": "regulatory_complexity",
                    "probability": 0.6,
                    "impact": 2000000,
                    "description": "Complex international regulations"
                },
                {
                    "type": "cultural_barriers",
                    "probability": 0.4,
                    "impact": 1500000,
                    "description": "Cultural and business practice differences"
                },
                {
                    "type": "currency_fluctuation",
                    "probability": 0.3,
                    "impact": 1000000,
                    "description": "Foreign exchange rate volatility"
                }
            ],
            implementation_complexity=7.5,
            time_to_implement=360,
            resource_requirements={
                "international_managers": 6,
                "legal_compliance": 4,
                "local_engineers": 15,
                "sales_teams": 12,
                "budget": 3500000,
                "local_infrastructure": True
            }
        )
    ]
    
    print(f"\n🎯 Decision Options:")
    for i, option in enumerate(options, 1):
        total_cost = sum(option.costs.values())
        total_benefit = sum(option.benefits.values())
        roi_estimate = (total_benefit - total_cost) / total_cost if total_cost > 0 else 0
        
        print(f"   {i}. {option.name}")
        print(f"      Cost: ${total_cost:,} | Benefit: ${total_benefit:,} | ROI: {roi_estimate:.1f}x")
        print(f"      Complexity: {option.implementation_complexity}/10 | Timeline: {option.time_to_implement} days")
    
    # Make the decision
    print(f"\n🤖 Making strategic decision...")
    decision = await engine.make_decision(context, options)
    
    # Display results
    print(f"\n🎉 DECISION RESULTS:")
    print(f"   Selected Option: {decision.selected_option.name}")
    print(f"   Confidence Level: {decision.confidence.value.upper()}")
    print(f"   Decision ID: {decision.id}")
    
    print(f"\n💡 Reasoning:")
    for i, reason in enumerate(decision.reasoning, 1):
        print(f"   {i}. {reason}")
    
    print(f"\n⚠️  Risk Assessment Summary:")
    risk_summary = decision.risk_assessment
    if isinstance(risk_summary, dict):
        for category, details in risk_summary.items():
            if isinstance(details, dict) and 'score' in details:
                print(f"   {category.title()}: {details['score']:.2f} ({details.get('impact', 'unknown')})")
    
    print(f"\n📈 Expected Outcomes:")
    expected = decision.expected_outcome
    if isinstance(expected, dict):
        for metric, value in expected.items():
            if isinstance(value, (int, float)):
                if 'revenue' in metric.lower() or 'cost' in metric.lower():
                    print(f"   {metric.replace('_', ' ').title()}: ${value:,.0f}")
                elif 'probability' in metric.lower() or 'rate' in metric.lower():
                    print(f"   {metric.replace('_', ' ').title()}: {value:.1%}")
                else:
                    print(f"   {metric.replace('_', ' ').title()}: {value}")
    
    print(f"\n🌳 Decision Tree Path: {' → '.join(decision.decision_tree_path)}")
    
    return decision


async def demo_risk_assessment(engine: IntelligenceEngine):
    """Demonstrate risk assessment capabilities"""
    print("\n" + "="*60)
    print("⚠️  DEMO 2: Advanced Risk Assessment")
    print("="*60)
    
    # Create a high-risk business scenario
    scenario = {
        "id": "market_disruption_scenario",
        "type": "strategic_initiative",
        "title": "Blockchain-Based Supply Chain Platform",
        "description": "Launch revolutionary blockchain platform for supply chain transparency",
        "impact_areas": ["financial", "operational", "strategic", "regulatory", "technological"],
        "timeline": "long_term",
        "investment_required": 8000000,
        "market_uncertainty": 0.8,
        "technology_maturity": 0.4,
        "regulatory_clarity": 0.3,
        "competitive_intensity": 0.9,
        "customer_readiness": 0.5
    }
    
    context = BusinessContext(
        industry="logistics",
        business_unit="supply_chain_technology",
        stakeholders=["ceo", "cto", "head_of_supply_chain", "legal", "investors"],
        constraints=[
            {"type": "regulatory_uncertainty", "level": "high"},
            {"type": "technology_risk", "level": "very_high"},
            {"type": "market_adoption", "level": "uncertain"}
        ],
        objectives=[
            {"type": "market_leadership", "target": 0.25, "timeline": "36_months"},
            {"type": "revenue_target", "target": 50000000, "timeline": "60_months"}
        ],
        current_state={
            "blockchain_expertise": "limited",
            "supply_chain_knowledge": "expert",
            "market_position": "established",
            "financial_strength": "strong"
        },
        historical_data={
            "innovation_success_rate": 0.6,
            "technology_adoption_speed": "medium",
            "regulatory_navigation": "experienced"
        },
        time_horizon="long_term",
        budget_constraints={"total": 10000000, "acceptable_loss": 3000000},
        regulatory_requirements=["blockchain_compliance", "data_privacy", "international_trade"]
    )
    
    print(f"🎯 Risk Scenario: {scenario['title']}")
    print(f"   Investment Required: ${scenario['investment_required']:,}")
    print(f"   Market Uncertainty: {scenario['market_uncertainty']:.0%}")
    print(f"   Technology Maturity: {scenario['technology_maturity']:.0%}")
    print(f"   Regulatory Clarity: {scenario['regulatory_clarity']:.0%}")
    
    print(f"\n🔍 Conducting comprehensive risk assessment...")
    risk_result = await engine.assess_risk(scenario, context)
    
    print(f"\n📊 RISK ASSESSMENT RESULTS:")
    print(f"   Overall Risk Score: {risk_result.get('overall_risk_score', 0):.2f}/1.0")
    print(f"   Risk Level: {risk_result.get('risk_level', 'unknown').upper()}")
    print(f"   Assessment Confidence: {risk_result.get('confidence', 0):.0%}")
    
    # Display detailed risk categories
    risk_categories = risk_result.get('risk_categories', {})
    if risk_categories:
        print(f"\n🏷️  Risk Categories:")
        for category, details in risk_categories.items():
            if isinstance(details, dict):
                score = details.get('score', 0)
                impact = details.get('impact', 'unknown')
                print(f"   {category.replace('_', ' ').title()}: {score:.2f} ({impact})")
    
    # Display mitigation strategies
    mitigation_strategies = risk_result.get('mitigation_strategies', [])
    if mitigation_strategies:
        print(f"\n🛡️  Recommended Mitigation Strategies:")
        for i, strategy in enumerate(mitigation_strategies, 1):
            if isinstance(strategy, dict):
                print(f"   {i}. {strategy.get('strategy', 'Unknown strategy')}")
                print(f"      Category: {strategy.get('category', 'general').title()}")
                print(f"      Effectiveness: {strategy.get('effectiveness', 0):.0%}")
                print(f"      Implementation Cost: {strategy.get('cost', 'unknown').title()}")
    
    return risk_result


async def demo_knowledge_query(engine: IntelligenceEngine):
    """Demonstrate knowledge graph query capabilities"""
    print("\n" + "="*60)
    print("🧠 DEMO 3: Knowledge Graph Intelligence")
    print("="*60)
    
    # Test various knowledge queries
    queries = [
        {
            "query": "market analysis best practices",
            "context": {"domain": "strategic_planning", "urgency": "high"}
        },
        {
            "query": "financial risk management strategies",
            "context": {"domain": "finance", "industry": "technology"}
        },
        {
            "query": "digital transformation success factors",
            "context": {"domain": "organizational_change", "size": "enterprise"}
        },
        {
            "query": "regulatory compliance frameworks",
            "context": {"domain": "legal", "geography": "global"}
        }
    ]
    
    for i, query_info in enumerate(queries, 1):
        query = query_info["query"]
        context = query_info["context"]
        
        print(f"\n🔍 Query {i}: '{query}'")
        print(f"   Context: {context}")
        
        results = await engine.query_knowledge(query, context)
        
        print(f"   📚 Found {len(results)} relevant knowledge items:")
        
        for j, result in enumerate(results[:3], 1):  # Show top 3 results
            name = result.get("name", "Unknown")
            description = result.get("description", "No description")
            relevance = result.get("relevance_score", result.get("final_relevance", 0))
            knowledge_type = result.get("knowledge_type", result.get("type", "unknown"))
            
            print(f"      {j}. {name}")
            print(f"         Type: {knowledge_type}")
            print(f"         Relevance: {relevance:.2f}")
            print(f"         Description: {description[:100]}...")
    
    return results


async def demo_outcome_learning(engine: IntelligenceEngine):
    """Demonstrate learning from business outcomes"""
    print("\n" + "="*60)
    print("📚 DEMO 4: Continuous Learning from Outcomes")
    print("="*60)
    
    # Simulate learning from previous decision outcomes
    learning_scenarios = [
        {
            "decision_id": "strategic_decision_001",
            "outcome": {
                "success_score": 0.85,
                "financial_impact": 3200000,
                "roi_achieved": 2.4,
                "timeline_variance": -5,  # 5% ahead of schedule
                "stakeholder_satisfaction": 0.9,
                "unexpected_benefits": [
                    "Improved team morale and retention",
                    "Unexpected partnership opportunities",
                    "Enhanced brand recognition"
                ],
                "challenges_faced": [
                    "Initial integration complexity",
                    "Customer training requirements"
                ],
                "lessons_learned": [
                    "Early stakeholder engagement is crucial",
                    "Phased rollout reduces implementation risk",
                    "Customer success team involvement improves adoption"
                ],
                "kpi_results": {
                    "customer_acquisition": 1.15,  # 15% above target
                    "revenue_growth": 1.08,  # 8% above target
                    "market_share": 1.12,  # 12% above target
                    "cost_efficiency": 0.95  # 5% under budget
                }
            }
        },
        {
            "decision_id": "risk_mitigation_002",
            "outcome": {
                "success_score": 0.65,
                "financial_impact": 1800000,
                "roi_achieved": 1.6,
                "timeline_variance": 15,  # 15% over schedule
                "stakeholder_satisfaction": 0.7,
                "unexpected_challenges": [
                    "Regulatory changes during implementation",
                    "Key talent departure",
                    "Technology integration issues"
                ],
                "mitigation_effectiveness": {
                    "risk_1": 0.8,  # 80% effective
                    "risk_2": 0.6,  # 60% effective
                    "risk_3": 0.9   # 90% effective
                },
                "lessons_learned": [
                    "Regulatory monitoring needs improvement",
                    "Talent retention strategies are critical",
                    "Technology due diligence should be more thorough"
                ],
                "improvement_opportunities": [
                    "Implement continuous regulatory scanning",
                    "Develop talent retention programs",
                    "Establish technology advisory board"
                ]
            }
        },
        {
            "decision_id": "innovation_initiative_003",
            "outcome": {
                "success_score": 0.92,
                "financial_impact": 5600000,
                "roi_achieved": 3.8,
                "timeline_variance": -10,  # 10% ahead of schedule
                "stakeholder_satisfaction": 0.95,
                "breakthrough_achievements": [
                    "Patent applications filed: 8",
                    "Industry recognition awards: 3",
                    "Customer testimonials: 25+"
                ],
                "success_factors": [
                    "Cross-functional collaboration",
                    "Customer co-creation approach",
                    "Agile development methodology",
                    "Strong executive sponsorship"
                ],
                "lessons_learned": [
                    "Customer involvement accelerates innovation",
                    "Cross-functional teams drive breakthrough results",
                    "Executive support enables resource flexibility"
                ],
                "best_practices_identified": [
                    "Weekly customer feedback sessions",
                    "Monthly cross-functional reviews",
                    "Quarterly executive check-ins"
                ]
            }
        }
    ]
    
    print(f"📊 Learning from {len(learning_scenarios)} business outcomes...")
    
    for i, scenario in enumerate(learning_scenarios, 1):
        decision_id = scenario["decision_id"]
        outcome = scenario["outcome"]
        success_score = outcome["success_score"]
        
        print(f"\n📈 Outcome {i}: {decision_id}")
        print(f"   Success Score: {success_score:.0%}")
        print(f"   Financial Impact: ${outcome['financial_impact']:,}")
        print(f"   ROI Achieved: {outcome['roi_achieved']:.1f}x")
        print(f"   Timeline Variance: {outcome['timeline_variance']:+.0f}%")
        
        # Learn from the outcome
        await engine.learn_from_outcome(decision_id, outcome)
        
        # Display key lessons
        lessons = outcome.get("lessons_learned", [])
        if lessons:
            print(f"   🎓 Key Lessons:")
            for lesson in lessons[:2]:  # Show top 2 lessons
                print(f"      • {lesson}")
    
    print(f"\n✅ Successfully processed learning from all outcomes")
    print(f"   The system will use these insights to improve future decisions")


async def demo_performance_metrics(engine: IntelligenceEngine):
    """Demonstrate performance metrics and system status"""
    print("\n" + "="*60)
    print("📊 DEMO 5: Performance Metrics & System Status")
    print("="*60)
    
    # Get system status
    print(f"🔍 Checking system status...")
    status = engine.get_status()
    
    print(f"\n🏥 System Health:")
    print(f"   Overall Status: {'✅ HEALTHY' if status.get('healthy') else '❌ UNHEALTHY'}")
    print(f"   Decision Tree: {status.get('decision_tree_status', 'unknown').upper()}")
    print(f"   ML Pipeline: {status.get('ml_pipeline_status', 'unknown').upper()}")
    print(f"   Risk Engine: {status.get('risk_engine_status', 'unknown').upper()}")
    print(f"   Knowledge Graph: {status.get('knowledge_graph_status', 'unknown').upper()}")
    
    # Get performance metrics
    print(f"\n📈 Performance Metrics:")
    metrics = engine.get_metrics()
    
    print(f"   Engine ID: {metrics.get('engine_id')}")
    print(f"   Usage Count: {metrics.get('usage_count', 0)}")
    print(f"   Error Count: {metrics.get('error_count', 0)}")
    print(f"   Error Rate: {metrics.get('error_rate', 0):.2%}")
    print(f"   Uptime: {metrics.get('created_at', 'unknown')}")
    
    # Simulate business impact metrics
    business_metrics = {
        "decisions_made_today": 47,
        "average_confidence": 0.78,
        "decision_accuracy": 0.85,
        "risk_predictions_accuracy": 0.82,
        "knowledge_queries_success_rate": 0.91,
        "average_response_time_ms": 245,
        "cost_savings_generated": 2400000,
        "revenue_opportunities_identified": 8900000,
        "risks_mitigated": 15,
        "stakeholder_satisfaction": 0.87
    }
    
    print(f"\n💼 Business Impact Metrics:")
    print(f"   Decisions Made Today: {business_metrics['decisions_made_today']}")
    print(f"   Average Confidence: {business_metrics['average_confidence']:.0%}")
    print(f"   Decision Accuracy: {business_metrics['decision_accuracy']:.0%}")
    print(f"   Response Time: {business_metrics['average_response_time_ms']}ms")
    print(f"   Cost Savings: ${business_metrics['cost_savings_generated']:,}")
    print(f"   Revenue Opportunities: ${business_metrics['revenue_opportunities_identified']:,}")
    print(f"   Stakeholder Satisfaction: {business_metrics['stakeholder_satisfaction']:.0%}")
    
    # Component capabilities
    capabilities = metrics.get('capabilities', [])
    if capabilities:
        print(f"\n🛠️  Engine Capabilities:")
        for capability in capabilities:
            print(f"   • {capability.replace('_', ' ').title()}")
    
    return metrics


async def main():
    """Run the complete intelligence engine demo"""
    try:
        await demo_business_decision_making()
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("🚀 Starting Intelligence and Decision Engine Demo...")
    asyncio.run(main())