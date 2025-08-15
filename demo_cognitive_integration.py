"""
Demo script for the Cognitive Integration System

This script demonstrates the unified AGI-level cognitive architecture
that integrates consciousness, intuitive reasoning, memory, meta-learning,
emotion, and personality systems.
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, Any

from scrollintel.engines.cognitive_integrator import (
    CognitiveIntegrator, AttentionFocus, CognitiveLoadLevel
)


class CognitiveIntegrationDemo:
    """Demo class for cognitive integration system"""
    
    def __init__(self):
        self.integrator = CognitiveIntegrator()
        self.demo_scenarios = self._create_demo_scenarios()
    
    def _create_demo_scenarios(self) -> Dict[str, Dict[str, Any]]:
        """Create diverse demo scenarios to showcase cognitive integration"""
        return {
            "strategic_planning": {
                "situation": "Our company needs to develop a comprehensive AI strategy for the next 5 years. We must consider technological trends, competitive landscape, regulatory changes, talent acquisition, and ethical implications while balancing innovation with risk management.",
                "context": {
                    "complexity": 0.9,
                    "time_pressure": 0.6,
                    "stakeholders": ["board_of_directors", "engineering_teams", "product_managers", "customers", "regulators"],
                    "constraints": ["budget_limitations", "regulatory_compliance", "talent_shortage", "market_competition"],
                    "resources": ["research_data", "market_analysis", "technical_expertise", "financial_resources"],
                    "environment": "highly_competitive_tech_industry"
                }
            },
            
            "crisis_management": {
                "situation": "A critical security vulnerability has been discovered in our main product. We need to coordinate an immediate response that includes technical fixes, customer communication, regulatory reporting, and damage control while maintaining business continuity.",
                "context": {
                    "complexity": 0.8,
                    "time_pressure": 0.95,
                    "stakeholders": ["customers", "security_team", "legal_team", "pr_team", "executives"],
                    "constraints": ["time_critical", "public_scrutiny", "legal_requirements", "customer_trust"],
                    "resources": ["security_experts", "communication_channels", "technical_infrastructure"],
                    "environment": "crisis_situation"
                }
            },
            
            "innovation_challenge": {
                "situation": "We want to create a breakthrough AI product that doesn't exist in the market yet. This requires combining cutting-edge research, user experience design, technical feasibility analysis, and business model innovation while ensuring ethical AI principles.",
                "context": {
                    "complexity": 0.85,
                    "time_pressure": 0.4,
                    "stakeholders": ["researchers", "designers", "engineers", "business_strategists", "ethicists"],
                    "constraints": ["technical_limitations", "market_readiness", "ethical_guidelines", "resource_allocation"],
                    "resources": ["research_papers", "prototype_tools", "user_feedback", "technical_talent"],
                    "environment": "innovation_lab"
                }
            },
            
            "team_conflict_resolution": {
                "situation": "Two high-performing teams have a fundamental disagreement about the technical architecture for a critical project. Both approaches have merit, but the conflict is affecting morale and progress. We need to find a solution that preserves relationships while making the best technical decision.",
                "context": {
                    "complexity": 0.6,
                    "time_pressure": 0.7,
                    "stakeholders": ["team_a", "team_b", "project_manager", "technical_lead", "hr_representative"],
                    "constraints": ["interpersonal_dynamics", "technical_trade_offs", "project_timeline", "team_morale"],
                    "resources": ["technical_documentation", "performance_data", "team_feedback", "mediation_expertise"],
                    "environment": "collaborative_workspace"
                }
            },
            
            "ethical_dilemma": {
                "situation": "Our AI system has the potential to significantly improve healthcare outcomes, but it requires access to sensitive patient data. We must balance the potential benefits against privacy concerns, regulatory requirements, and ethical implications while ensuring the technology is accessible and fair.",
                "context": {
                    "complexity": 0.9,
                    "time_pressure": 0.5,
                    "stakeholders": ["patients", "healthcare_providers", "regulators", "privacy_advocates", "researchers"],
                    "constraints": ["privacy_regulations", "ethical_guidelines", "technical_limitations", "social_responsibility"],
                    "resources": ["medical_expertise", "privacy_technologies", "regulatory_guidance", "ethical_frameworks"],
                    "environment": "healthcare_innovation"
                }
            }
        }
    
    async def run_comprehensive_demo(self):
        """Run comprehensive demo of cognitive integration capabilities"""
        print("🧠 Cognitive Integration System Demo")
        print("=" * 50)
        print("Demonstrating unified AGI-level cognitive architecture")
        print("Integrating: Consciousness, Intuitive Reasoning, Memory, Meta-Learning, Emotion, Personality")
        print()
        
        # Start the cognitive integration system
        print("🚀 Starting Cognitive Integration System...")
        await self.integrator.start()
        print("✅ System started successfully")
        print()
        
        try:
            # Demo 1: System Status and Health
            await self._demo_system_status()
            
            # Demo 2: Attention Management
            await self._demo_attention_management()
            
            # Demo 3: Complex Decision Making
            await self._demo_complex_decision_making()
            
            # Demo 4: Cognitive Load Balancing
            await self._demo_cognitive_load_balancing()
            
            # Demo 5: Multi-Scenario Processing
            await self._demo_multi_scenario_processing()
            
            # Demo 6: Self-Regulation and Health Monitoring
            await self._demo_self_regulation()
            
            # Demo 7: Integration Quality Analysis
            await self._demo_integration_quality()
            
        finally:
            # Stop the system
            print("🛑 Stopping Cognitive Integration System...")
            await self.integrator.stop()
            print("✅ System stopped successfully")
    
    async def _demo_system_status(self):
        """Demo system status and initial state"""
        print("📊 DEMO 1: System Status and Health")
        print("-" * 40)
        
        status = self.integrator.get_system_status()
        
        print(f"System Running: {status['running']}")
        print(f"Consciousness Level: {status['cognitive_state'].consciousness_level:.3f}")
        print(f"Reasoning Capacity: {status['cognitive_state'].reasoning_capacity:.3f}")
        print(f"Memory Utilization: {status['cognitive_state'].memory_utilization:.3f}")
        print(f"Learning Efficiency: {status['cognitive_state'].learning_efficiency:.3f}")
        print(f"Emotional Stability: {status['cognitive_state'].emotional_stability:.3f}")
        print(f"Integration Coherence: {status['cognitive_state'].integration_coherence:.3f}")
        print(f"Current Attention Focus: {status['cognitive_state'].attention_focus.value}")
        print(f"Cognitive Load Level: {status['cognitive_state'].cognitive_load.value}")
        print()
    
    async def _demo_attention_management(self):
        """Demo attention management capabilities"""
        print("🎯 DEMO 2: Attention Management")
        print("-" * 40)
        
        attention_scenarios = [
            (AttentionFocus.CONSCIOUSNESS, 0.8, "Deep self-reflection and awareness"),
            (AttentionFocus.REASONING, 0.9, "Complex analytical problem solving"),
            (AttentionFocus.MEMORY, 0.7, "Information retrieval and synthesis"),
            (AttentionFocus.EMOTION, 0.6, "Social and emotional processing"),
            (AttentionFocus.INTEGRATION, 0.8, "System-wide coordination")
        ]
        
        for focus, intensity, description in attention_scenarios:
            print(f"Setting attention focus to {focus.value} (intensity: {intensity})")
            print(f"Scenario: {description}")
            
            allocation = await self.integrator.manage_attention(focus, intensity)
            
            print(f"  Consciousness: {allocation.consciousness_attention:.3f}")
            print(f"  Reasoning: {allocation.reasoning_attention:.3f}")
            print(f"  Memory: {allocation.memory_attention:.3f}")
            print(f"  Learning: {allocation.learning_attention:.3f}")
            print(f"  Emotion: {allocation.emotion_attention:.3f}")
            print(f"  Integration: {allocation.integration_attention:.3f}")
            print()
    
    async def _demo_complex_decision_making(self):
        """Demo complex decision making with full cognitive integration"""
        print("🧩 DEMO 3: Complex Decision Making")
        print("-" * 40)
        
        # Use the strategic planning scenario
        scenario = self.demo_scenarios["strategic_planning"]
        
        print("Scenario: Strategic AI Planning")
        print(f"Situation: {scenario['situation'][:100]}...")
        print(f"Complexity: {scenario['context']['complexity']}")
        print(f"Stakeholders: {len(scenario['context']['stakeholders'])}")
        print(f"Constraints: {len(scenario['context']['constraints'])}")
        print()
        
        print("🔄 Processing through integrated cognitive systems...")
        decision = await self.integrator.process_complex_situation(
            scenario["situation"], 
            scenario["context"]
        )
        
        print("✅ Decision Complete!")
        print(f"Decision ID: {decision.decision_id}")
        print(f"Confidence: {decision.confidence:.3f}")
        print(f"Integration Quality: {decision.integration_quality:.3f}")
        print(f"Supporting Systems: {', '.join(decision.supporting_systems)}")
        print()
        
        print("Reasoning Path:")
        for i, step in enumerate(decision.reasoning_path, 1):
            print(f"  {i}. {step}")
        print()
        
        print("System Contributions:")
        if decision.consciousness_input:
            print(f"  🧠 Consciousness: Awareness level {decision.consciousness_input.get('consciousness_level', 'N/A')}")
        if decision.intuitive_input:
            print(f"  💡 Intuition: Confidence {decision.intuitive_input.get('confidence', 'N/A')}")
        if decision.memory_input:
            print(f"  🗄️ Memory: {len(decision.memory_input.get('supporting_memories', []))} supporting memories")
        if decision.emotional_input:
            emotion_state = decision.emotional_input.get('emotional_state')
            if emotion_state:
                print(f"  ❤️ Emotion: {emotion_state.primary_emotion.value} state")
        if decision.personality_input:
            print(f"  👤 Personality: Influence score {decision.personality_input.get('influence_score', 'N/A')}")
        print()
    
    async def _demo_cognitive_load_balancing(self):
        """Demo cognitive load balancing"""
        print("⚖️ DEMO 4: Cognitive Load Balancing")
        print("-" * 40)
        
        print("Assessing current cognitive loads across all systems...")
        system_loads = await self.integrator.balance_cognitive_load()
        
        print("System Load Analysis:")
        for system, load in system_loads.items():
            load_status = "🔴 HIGH" if load > 0.8 else "🟡 MEDIUM" if load > 0.5 else "🟢 LOW"
            print(f"  {system.capitalize()}: {load:.3f} {load_status}")
        
        max_load = max(system_loads.values())
        print(f"\nOverall System Load: {max_load:.3f}")
        print(f"Load Level: {self.integrator.cognitive_state.cognitive_load.value}")
        print()
    
    async def _demo_multi_scenario_processing(self):
        """Demo processing multiple scenarios to show system versatility"""
        print("🎭 DEMO 5: Multi-Scenario Processing")
        print("-" * 40)
        
        scenarios_to_process = ["crisis_management", "innovation_challenge", "team_conflict_resolution"]
        
        for scenario_name in scenarios_to_process:
            scenario = self.demo_scenarios[scenario_name]
            
            print(f"Processing: {scenario_name.replace('_', ' ').title()}")
            print(f"Complexity: {scenario['context']['complexity']:.1f} | "
                  f"Time Pressure: {scenario['context']['time_pressure']:.1f}")
            
            decision = await self.integrator.process_complex_situation(
                scenario["situation"], 
                scenario["context"]
            )
            
            print(f"✅ Decision: Confidence {decision.confidence:.3f}, "
                  f"Quality {decision.integration_quality:.3f}")
            print(f"   Systems Used: {len(decision.supporting_systems)}")
            print()
        
        print(f"Total Decisions Made: {len(self.integrator.decision_history)}")
        print()
    
    async def _demo_self_regulation(self):
        """Demo self-regulation and health monitoring"""
        print("🏥 DEMO 6: Self-Regulation and Health Monitoring")
        print("-" * 40)
        
        # Monitor cognitive health
        print("Performing comprehensive health assessment...")
        health_report = await self.integrator.monitor_cognitive_health()
        
        print("System Health Report:")
        print(f"  Overall Health Score: {health_report['overall_score']:.3f}")
        print(f"  Integration Health: {health_report['integration_health']:.3f}")
        
        print("\nIndividual System Health:")
        for system, health in health_report["system_health"].items():
            health_status = "🟢 EXCELLENT" if health > 0.8 else "🟡 GOOD" if health > 0.6 else "🔴 NEEDS ATTENTION"
            print(f"  {system.capitalize()}: {health:.3f} {health_status}")
        
        if health_report["issues_identified"]:
            print(f"\nIssues Identified: {len(health_report['issues_identified'])}")
            for issue in health_report["issues_identified"]:
                print(f"  ⚠️ {issue}")
        
        if health_report["recommendations"]:
            print(f"\nRecommendations: {len(health_report['recommendations'])}")
            for rec in health_report["recommendations"]:
                print(f"  💡 {rec}")
        
        # Perform self-regulation
        print("\nPerforming self-regulation...")
        regulation_result = await self.integrator.self_regulate()
        
        print(f"Regulation Effectiveness: {regulation_result['effectiveness']:.3f}")
        if regulation_result["regulation_applied"]:
            print("Regulations Applied:")
            for regulation, details in regulation_result["regulation_applied"].items():
                print(f"  🔧 {regulation}: {details.get('strategy', 'Applied')}")
        print()
    
    async def _demo_integration_quality(self):
        """Demo integration quality analysis"""
        print("🔬 DEMO 7: Integration Quality Analysis")
        print("-" * 40)
        
        if not self.integrator.decision_history:
            print("No decisions in history to analyze.")
            return
        
        # Analyze decision history
        decisions = self.integrator.decision_history
        
        print(f"Analyzing {len(decisions)} decisions...")
        
        # Calculate statistics
        confidences = [d.confidence for d in decisions]
        qualities = [d.integration_quality for d in decisions]
        system_usage = {}
        
        for decision in decisions:
            for system in decision.supporting_systems:
                system_usage[system] = system_usage.get(system, 0) + 1
        
        avg_confidence = sum(confidences) / len(confidences)
        avg_quality = sum(qualities) / len(qualities)
        
        print(f"Average Decision Confidence: {avg_confidence:.3f}")
        print(f"Average Integration Quality: {avg_quality:.3f}")
        
        print("\nSystem Usage Frequency:")
        for system, count in sorted(system_usage.items(), key=lambda x: x[1], reverse=True):
            usage_rate = count / len(decisions)
            print(f"  {system}: {count}/{len(decisions)} ({usage_rate:.1%})")
        
        # Show recent decision details
        if decisions:
            recent_decision = decisions[-1]
            print(f"\nMost Recent Decision Analysis:")
            print(f"  Decision ID: {recent_decision.decision_id}")
            print(f"  Confidence: {recent_decision.confidence:.3f}")
            print(f"  Integration Quality: {recent_decision.integration_quality:.3f}")
            print(f"  Reasoning Steps: {len(recent_decision.reasoning_path)}")
            print(f"  Systems Integrated: {len(recent_decision.supporting_systems)}")
        
        print()
    
    async def run_interactive_demo(self):
        """Run interactive demo where user can input scenarios"""
        print("🎮 Interactive Cognitive Integration Demo")
        print("=" * 50)
        print("Enter your own scenarios to see the cognitive integration system in action!")
        print("Type 'quit' to exit, 'help' for commands, or 'demo' for predefined scenarios.")
        print()
        
        await self.integrator.start()
        
        try:
            while True:
                user_input = input("\n🧠 Enter a situation to analyze: ").strip()
                
                if user_input.lower() == 'quit':
                    break
                elif user_input.lower() == 'help':
                    self._show_help()
                    continue
                elif user_input.lower() == 'demo':
                    await self._run_predefined_scenario()
                    continue
                elif not user_input:
                    continue
                
                # Process user scenario
                print(f"\n🔄 Processing: {user_input}")
                
                # Create context based on input complexity
                context = {
                    "complexity": min(1.0, len(user_input) / 200.0 + 0.3),
                    "time_pressure": 0.5,
                    "stakeholders": ["user", "system"],
                    "constraints": ["user_context"],
                    "resources": ["cognitive_systems"]
                }
                
                decision = await self.integrator.process_complex_situation(user_input, context)
                
                print(f"✅ Analysis Complete!")
                print(f"Confidence: {decision.confidence:.3f}")
                print(f"Integration Quality: {decision.integration_quality:.3f}")
                print(f"Systems Used: {', '.join(decision.supporting_systems)}")
                
                print("\nKey Insights:")
                for i, step in enumerate(decision.reasoning_path[:3], 1):
                    print(f"  {i}. {step}")
                
                if len(decision.reasoning_path) > 3:
                    print(f"  ... and {len(decision.reasoning_path) - 3} more insights")
        
        finally:
            await self.integrator.stop()
    
    def _show_help(self):
        """Show help information"""
        print("\n📖 Help - Available Commands:")
        print("  • Enter any situation or question for analysis")
        print("  • 'demo' - Run a predefined scenario")
        print("  • 'help' - Show this help message")
        print("  • 'quit' - Exit the demo")
        print("\nExample inputs:")
        print("  • 'How should we handle a difficult client situation?'")
        print("  • 'What's the best approach for team collaboration?'")
        print("  • 'How do we balance innovation with risk management?'")
    
    async def _run_predefined_scenario(self):
        """Run a predefined scenario chosen by user"""
        print("\n📋 Available Predefined Scenarios:")
        scenarios = list(self.demo_scenarios.keys())
        
        for i, scenario in enumerate(scenarios, 1):
            title = scenario.replace('_', ' ').title()
            print(f"  {i}. {title}")
        
        try:
            choice = input(f"\nSelect scenario (1-{len(scenarios)}): ").strip()
            scenario_index = int(choice) - 1
            
            if 0 <= scenario_index < len(scenarios):
                scenario_name = scenarios[scenario_index]
                scenario = self.demo_scenarios[scenario_name]
                
                print(f"\n🎭 Running: {scenario_name.replace('_', ' ').title()}")
                print(f"Situation: {scenario['situation'][:150]}...")
                
                decision = await self.integrator.process_complex_situation(
                    scenario["situation"], 
                    scenario["context"]
                )
                
                print(f"\n✅ Analysis Complete!")
                print(f"Confidence: {decision.confidence:.3f}")
                print(f"Integration Quality: {decision.integration_quality:.3f}")
                print(f"Systems Used: {', '.join(decision.supporting_systems)}")
                
                print("\nDetailed Analysis:")
                for step in decision.reasoning_path:
                    print(f"  • {step}")
            else:
                print("Invalid selection.")
        
        except (ValueError, IndexError):
            print("Invalid input. Please enter a number.")


async def main():
    """Main demo function"""
    demo = CognitiveIntegrationDemo()
    
    print("🧠 Cognitive Integration System Demo")
    print("Choose demo mode:")
    print("1. Comprehensive Demo (automated)")
    print("2. Interactive Demo (user input)")
    
    try:
        choice = input("Enter choice (1 or 2): ").strip()
        
        if choice == "1":
            await demo.run_comprehensive_demo()
        elif choice == "2":
            await demo.run_interactive_demo()
        else:
            print("Invalid choice. Running comprehensive demo...")
            await demo.run_comprehensive_demo()
    
    except KeyboardInterrupt:
        print("\n\n👋 Demo interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n❌ Demo error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())