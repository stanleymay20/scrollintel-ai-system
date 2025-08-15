"""
Demo script for Decision Tree System
"""

import asyncio
import json
from datetime import datetime

from scrollintel.engines.decision_tree_engine import DecisionTreeEngine
from scrollintel.models.decision_tree_models import CrisisType, SeverityLevel


async def demo_decision_tree_system():
    """Demonstrate the decision tree system capabilities"""
    print("🌳 Crisis Leadership Excellence - Decision Tree System Demo")
    print("=" * 60)
    
    # Initialize the engine
    engine = DecisionTreeEngine()
    
    # Demo 1: Show available decision trees
    print("\n1. Available Decision Trees:")
    print("-" * 30)
    for tree_id, tree in engine.decision_trees.items():
        print(f"📋 {tree.name} ({tree_id})")
        print(f"   Description: {tree.description}")
        print(f"   Crisis Types: {[ct.value for ct in tree.crisis_types]}")
        print(f"   Severity Levels: {[sl.value for sl in tree.severity_levels]}")
        print(f"   Success Rate: {tree.success_rate:.1%}")
        print(f"   Usage Count: {tree.usage_count}")
        print()
    
    # Demo 2: System Outage Scenario
    print("\n2. System Outage Crisis Scenario:")
    print("-" * 35)
    
    # Find applicable trees
    applicable_trees = engine.find_applicable_trees(
        CrisisType.SYSTEM_OUTAGE,
        SeverityLevel.HIGH
    )
    
    print(f"🔍 Found {len(applicable_trees)} applicable decision trees")
    if applicable_trees:
        selected_tree = applicable_trees[0]
        print(f"📊 Selected: {selected_tree.name}")
        
        # Start decision path
        crisis_id = "outage_2024_001"
        path = engine.start_decision_path(crisis_id, selected_tree.tree_id)
        print(f"🚀 Started decision path: {path.path_id}")
        
        # Navigate with crisis context
        outage_context = {
            "affected_systems": ["web_frontend", "api_gateway", "user_database"],
            "user_count": 15000,
            "business_impact": "high",
            "estimated_revenue_loss": 50000,
            "customer_complaints": 247
        }
        
        print(f"📊 Crisis Context: {json.dumps(outage_context, indent=2)}")
        
        recommendation = engine.navigate_tree(path.path_id, outage_context)
        
        print(f"\n💡 Decision Recommendation:")
        print(f"   Action: {recommendation.recommended_action.title}")
        print(f"   Description: {recommendation.recommended_action.description}")
        print(f"   Confidence: {recommendation.confidence_level.value}")
        print(f"   Success Probability: {recommendation.recommended_action.success_probability:.1%}")
        print(f"   Estimated Duration: {recommendation.recommended_action.estimated_duration} minutes")
        print(f"   Required Resources: {recommendation.recommended_action.required_resources}")
        print(f"   Reasoning: {recommendation.reasoning}")
        
        # Complete the path
        engine.complete_path(path.path_id, "crisis_resolved_successfully", True)
        print(f"✅ Decision path completed successfully")
    
    # Demo 3: Security Breach Scenario
    print("\n3. Security Breach Crisis Scenario:")
    print("-" * 36)
    
    applicable_trees = engine.find_applicable_trees(
        CrisisType.SECURITY_BREACH,
        SeverityLevel.CRITICAL
    )
    
    if applicable_trees:
        selected_tree = applicable_trees[0]
        print(f"📊 Selected: {selected_tree.name}")
        
        # Start decision path
        crisis_id = "breach_2024_001"
        path = engine.start_decision_path(crisis_id, selected_tree.tree_id)
        print(f"🚀 Started decision path: {path.path_id}")
        
        # Navigate with breach context
        breach_context = {
            "compromised_systems": ["user_database", "payment_system"],
            "data_exposure": "pii",
            "attack_vector": "sql_injection",
            "affected_records": 50000,
            "attack_duration": "2_hours",
            "containment_status": "partial"
        }
        
        print(f"📊 Crisis Context: {json.dumps(breach_context, indent=2)}")
        
        recommendation = engine.navigate_tree(path.path_id, breach_context)
        
        print(f"\n💡 Decision Recommendation:")
        print(f"   Action: {recommendation.recommended_action.title}")
        print(f"   Description: {recommendation.recommended_action.description}")
        print(f"   Confidence: {recommendation.confidence_level.value}")
        print(f"   Success Probability: {recommendation.recommended_action.success_probability:.1%}")
        print(f"   Estimated Duration: {recommendation.recommended_action.estimated_duration} minutes")
        print(f"   Required Resources: {recommendation.recommended_action.required_resources}")
        print(f"   Risk Level: {recommendation.recommended_action.risk_level}")
        
        # Complete the path
        engine.complete_path(path.path_id, "breach_contained_investigation_ongoing", True)
        print(f"✅ Decision path completed successfully")
    
    # Demo 4: Financial Crisis Scenario
    print("\n4. Financial Crisis Scenario:")
    print("-" * 28)
    
    applicable_trees = engine.find_applicable_trees(
        CrisisType.FINANCIAL_CRISIS,
        SeverityLevel.HIGH
    )
    
    if applicable_trees:
        selected_tree = applicable_trees[0]
        print(f"📊 Selected: {selected_tree.name}")
        
        # Start decision path
        crisis_id = "financial_2024_001"
        path = engine.start_decision_path(crisis_id, selected_tree.tree_id)
        print(f"🚀 Started decision path: {path.path_id}")
        
        # Navigate with financial context
        financial_context = {
            "cash_reserves": 750000,  # $750K
            "revenue_impact": 0.35,   # 35% revenue drop
            "funding_runway": 4,      # 4 months
            "burn_rate": 200000,      # $200K/month
            "investor_confidence": "low",
            "market_conditions": "recession"
        }
        
        print(f"📊 Crisis Context: {json.dumps(financial_context, indent=2)}")
        
        recommendation = engine.navigate_tree(path.path_id, financial_context)
        
        print(f"\n💡 Decision Recommendation:")
        print(f"   Action: {recommendation.recommended_action.title}")
        print(f"   Description: {recommendation.recommended_action.description}")
        print(f"   Confidence: {recommendation.confidence_level.value}")
        print(f"   Success Probability: {recommendation.recommended_action.success_probability:.1%}")
        print(f"   Estimated Duration: {recommendation.recommended_action.estimated_duration} minutes")
        print(f"   Required Resources: {recommendation.recommended_action.required_resources}")
        print(f"   Risk Level: {recommendation.recommended_action.risk_level}")
        
        # Complete the path with learning
        engine.complete_path(path.path_id, "cash_preservation_activated", True)
        
        # Add learning data
        lessons_learned = [
            "Earlier financial monitoring could have prevented crisis escalation",
            "Diversified revenue streams needed for resilience",
            "Emergency funding sources should be pre-negotiated"
        ]
        engine.learn_from_outcome(path.path_id, 0.8, lessons_learned)
        print(f"✅ Decision path completed with learning data recorded")
    
    # Demo 5: Tree Performance Metrics
    print("\n5. Decision Tree Performance Metrics:")
    print("-" * 38)
    
    for tree_id in engine.decision_trees.keys():
        metrics = engine.get_tree_metrics(tree_id)
        print(f"📈 {tree_id}:")
        print(f"   Total Usage: {metrics.total_usage}")
        print(f"   Success Rate: {metrics.success_rate:.1%}")
        print(f"   Average Decision Time: {metrics.average_decision_time:.1f}s")
        print()
    
    # Demo 6: Tree Optimization
    print("\n6. Decision Tree Optimization:")
    print("-" * 33)
    
    for tree_id in engine.decision_trees.keys():
        suggestions = engine.optimize_tree(tree_id)
        print(f"🔧 Optimization suggestions for {tree_id}:")
        for i, suggestion in enumerate(suggestions, 1):
            print(f"   {i}. {suggestion}")
        print()
    
    # Demo 7: Active Paths Status
    print("\n7. Active Decision Paths:")
    print("-" * 25)
    
    active_count = len(engine.active_paths)
    print(f"📊 Currently active paths: {active_count}")
    
    if active_count > 0:
        for path_id, path in engine.active_paths.items():
            print(f"   🔄 {path_id}:")
            print(f"      Crisis: {path.crisis_id}")
            print(f"      Tree: {path.tree_id}")
            print(f"      Started: {path.start_time}")
            print(f"      Nodes Traversed: {len(path.nodes_traversed)}")
    else:
        print("   ✅ No active decision paths")
    
    print(f"\n🎉 Decision Tree System Demo Complete!")
    print(f"📊 Total Trees Available: {len(engine.decision_trees)}")
    print(f"🧠 Learning Data Points: {len(engine.learning_data)}")
    print(f"⚡ System Ready for Crisis Leadership Excellence!")


if __name__ == "__main__":
    asyncio.run(demo_decision_tree_system())