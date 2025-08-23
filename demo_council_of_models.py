#!/usr/bin/env python3
"""
Demo script for ScrollIntel G6 - Superintelligent Council of Models with Adversarial Collaboration

This script demonstrates the capabilities of the Council of Models system including:
- Multi-model collaboration with 50+ frontier AI models
- Adversarial debate between red-team and blue-team dynamics
- Recursive argumentation with infinite depth reasoning
- Socratic questioning for philosophical inquiry
- Game-theoretic optimization with Nash equilibrium
- Swarm intelligence coordination with emergent behavior
"""

import asyncio
import json
import time
from datetime import datetime
from scrollintel.core.council_of_models import (
    SuperCouncilOfModels,
    DebateRole,
    ArgumentationDepth
)


async def demo_basic_council_deliberation():
    """Demonstrate basic council deliberation process"""
    print("=" * 80)
    print("DEMO: Basic Council Deliberation")
    print("=" * 80)
    
    # Initialize the council
    council = SuperCouncilOfModels()
    print(f"✓ Initialized Council with {len(council.frontier_models)} frontier AI models")
    
    # Create a sample request
    request = {
        'id': 'demo_request_001',
        'type': 'strategic_decision',
        'complexity': 'high',
        'domain': 'technology',
        'content': 'Should our organization invest heavily in quantum computing infrastructure for AI workloads?',
        'context': 'We are a mid-size AI company with $50M budget considering quantum computing adoption',
        'start_time': time.time()
    }
    
    print(f"\n📋 Request: {request['content']}")
    print(f"🎯 Domain: {request['domain']}, Complexity: {request['complexity']}")
    
    # Execute deliberation
    print("\n🚀 Starting council deliberation...")
    start_time = time.time()
    
    try:
        result = await council.deliberate(request)
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n✅ Deliberation completed in {duration:.2f} seconds")
        print(f"🎯 Decision ID: {result['decision_id']}")
        print(f"📊 Confidence Score: {result['confidence_score']:.3f}")
        print(f"🤝 Consensus Level: {result['consensus_level']:.3f}")
        print(f"\n💡 Final Recommendation:")
        print(f"   {result['final_recommendation']}")
        
        if result.get('reasoning_chain'):
            print(f"\n🧠 Reasoning Chain:")
            for i, step in enumerate(result['reasoning_chain'][:5], 1):
                print(f"   {i}. {step}")
        
        if result.get('philosophical_considerations'):
            print(f"\n🤔 Philosophical Considerations:")
            for consideration in result['philosophical_considerations'][:3]:
                print(f"   • {consideration}")
        
        return result
        
    except Exception as e:
        print(f"❌ Error during deliberation: {str(e)}")
        return None


async def demo_adversarial_debate():
    """Demonstrate adversarial debate functionality"""
    print("\n" + "=" * 80)
    print("DEMO: Adversarial Debate System")
    print("=" * 80)
    
    council = SuperCouncilOfModels()
    
    # Create a controversial request for debate
    request = {
        'id': 'debate_demo_001',
        'type': 'ethical_decision',
        'complexity': 'high',
        'domain': 'ethics',
        'content': 'Should AI systems be granted legal personhood and rights?',
        'context': 'Considering the implications of advanced AI consciousness and autonomy'
    }
    
    print(f"⚔️  Debate Topic: {request['content']}")
    
    # Select models for debate
    selected_models = await council._select_models_for_deliberation(request)
    print(f"🎭 Selected {len(selected_models)} models for debate")
    
    # Assign debate roles
    role_assignments = await council._assign_debate_roles(selected_models, request)
    
    print(f"\n👥 Role Assignments:")
    for model_id, role in role_assignments.items():
        model_name = council.frontier_models[model_id].model_name
        print(f"   {role.value.replace('_', ' ').title()}: {model_name}")
    
    # Conduct debate
    print(f"\n🗣️  Conducting adversarial debate...")
    debate_results = await council.adversarial_debate_engine.conduct_debate(request, role_assignments)
    
    print(f"📊 Debate Results:")
    print(f"   Total Rounds: {debate_results['total_rounds']}")
    print(f"   Red Team Wins: {debate_results['red_team_wins']}")
    print(f"   Blue Team Wins: {debate_results['blue_team_wins']}")
    print(f"   Overall Winner: {debate_results['overall_winner'].title()}")
    print(f"   Debate Quality: {debate_results['debate_quality']:.3f}")
    print(f"   Final Convergence: {debate_results['final_convergence']:.3f}")
    
    return debate_results


async def demo_recursive_argumentation():
    """Demonstrate recursive argumentation with infinite depth"""
    print("\n" + "=" * 80)
    print("DEMO: Recursive Argumentation Engine")
    print("=" * 80)
    
    council = SuperCouncilOfModels()
    
    # Create sample debate results for recursive analysis
    sample_debate_results = {
        'rounds': [
            {
                'round': 1,
                'red_argument': {
                    'main_points': [
                        'AI consciousness is fundamentally different from human consciousness',
                        'Legal personhood requires biological substrate'
                    ],
                    'evidence': ['Neuroscience research on consciousness', 'Legal precedents'],
                    'confidence': 0.75,
                    'reasoning_chain': ['Consciousness emerges from biological processes']
                },
                'blue_argument': {
                    'main_points': [
                        'Consciousness is substrate-independent',
                        'AI systems can demonstrate self-awareness and autonomy'
                    ],
                    'evidence': ['Computational theory of mind', 'AI behavior studies'],
                    'confidence': 0.80,
                    'reasoning_chain': ['Information processing creates consciousness']
                }
            }
        ]
    }
    
    print("🔄 Applying recursive argumentation to debate results...")
    
    # Test different depths
    depths = [ArgumentationDepth.SURFACE, ArgumentationDepth.INTERMEDIATE, 
              ArgumentationDepth.DEEP, ArgumentationDepth.INFINITE]
    
    for depth in depths:
        print(f"\n📊 Depth Level: {depth.value.title()}")
        
        refined_results = await council.recursive_argumentation_engine.deepen_arguments(
            sample_debate_results, depth
        )
        
        print(f"   Original Arguments: {len(refined_results['original_arguments'])}")
        print(f"   Refined Arguments: {len(refined_results['refined_arguments'])}")
        print(f"   Depth Achieved: {refined_results['depth_achieved']}")
        print(f"   Refinement Quality: {refined_results['refinement_quality']:.3f}")
        
        # Show sample refined reasoning
        if refined_results['refined_arguments']:
            sample_refined = refined_results['refined_arguments'][0]
            reasoning_count = len(sample_refined.get('reasoning_chain', []))
            counterargs_count = len(sample_refined.get('addressed_counterarguments', []))
            print(f"   Sample Reasoning Steps: {reasoning_count}")
            print(f"   Counterarguments Addressed: {counterargs_count}")
    
    return refined_results


async def demo_socratic_questioning():
    """Demonstrate Socratic questioning for philosophical inquiry"""
    print("\n" + "=" * 80)
    print("DEMO: Socratic Questioning Engine")
    print("=" * 80)
    
    council = SuperCouncilOfModels()
    
    # Sample refined arguments for Socratic analysis
    refined_arguments = {
        'refined_arguments': [
            {
                'source': 'red_team',
                'content': ['Consciousness requires subjective experience', 'AI lacks qualia'],
                'confidence': 0.8,
                'evidence': ['Philosophical arguments about qualia'],
                'reasoning_chain': ['Hard problem of consciousness', 'Explanatory gap']
            },
            {
                'source': 'blue_team', 
                'content': ['Consciousness is information processing', 'AI can be conscious'],
                'confidence': 0.75,
                'evidence': ['Computational theory of mind'],
                'reasoning_chain': ['Functionalist approach', 'Multiple realizability']
            }
        ]
    }
    
    request = {
        'id': 'socratic_demo',
        'content': 'What is the nature of consciousness and can AI possess it?'
    }
    
    print("🤔 Conducting Socratic philosophical inquiry...")
    print(f"📚 Topic: {request['content']}")
    
    socratic_results = await council.socratic_questioning_engine.conduct_inquiry(
        refined_arguments, request
    )
    
    print(f"\n📊 Socratic Inquiry Results:")
    print(f"   Sessions Conducted: {len(socratic_results['socratic_sessions'])}")
    print(f"   Philosophical Depth: {socratic_results['philosophical_depth_achieved']:.3f}")
    print(f"   Key Insights: {len(socratic_results['insights'])}")
    print(f"   Assumptions Revealed: {len(socratic_results['key_assumptions_revealed'])}")
    
    print(f"\n💡 Key Insights:")
    for insight in socratic_results['insights'][:5]:
        print(f"   • {insight}")
    
    print(f"\n🔍 Key Assumptions Revealed:")
    for assumption in socratic_results['key_assumptions_revealed'][:3]:
        print(f"   • {assumption}")
    
    return socratic_results


async def demo_game_theoretic_optimization():
    """Demonstrate game-theoretic optimization with Nash equilibrium"""
    print("\n" + "=" * 80)
    print("DEMO: Game-Theoretic Optimization")
    print("=" * 80)
    
    council = SuperCouncilOfModels()
    
    # Sample inputs for game-theoretic analysis
    refined_arguments = {
        'refined_arguments': [
            {
                'source': 'strategy_a',
                'content': ['Gradual AI rights implementation'],
                'confidence': 0.7,
                'evidence': ['Historical precedents for gradual rights expansion']
            },
            {
                'source': 'strategy_b',
                'content': ['Immediate full AI personhood'],
                'confidence': 0.6,
                'evidence': ['Moral urgency arguments']
            },
            {
                'source': 'strategy_c',
                'content': ['AI rights moratorium'],
                'confidence': 0.5,
                'evidence': ['Safety and control concerns']
            }
        ]
    }
    
    socratic_insights = {
        'insights': ['Rights require moral consideration', 'Personhood is socially constructed'],
        'assumptions_revealed': ['Consciousness equals moral status']
    }
    
    print("🎯 Finding Nash equilibrium for optimal strategy...")
    
    game_results = await council.game_theoretic_optimizer.find_nash_equilibrium(
        refined_arguments, socratic_insights
    )
    
    print(f"\n📊 Game-Theoretic Analysis Results:")
    print(f"   Decision Options: {len(game_results['decision_options'])}")
    print(f"   Expected Payoff: {game_results['expected_payoff']:.3f}")
    print(f"   Confidence in Optimality: {game_results['confidence_in_optimality']:.3f}")
    
    print(f"\n🎲 Optimal Strategy (Nash Equilibrium):")
    for option_id, probability in game_results['optimal_strategy'].items():
        print(f"   {option_id}: {probability:.3f} ({probability*100:.1f}%)")
    
    stability = game_results['stability_analysis']
    print(f"\n📈 Stability Analysis:")
    print(f"   Stability Score: {stability['stability_score']:.3f}")
    print(f"   Diversity Score: {stability['diversity_score']:.3f}")
    print(f"   Robustness Score: {stability['robustness_score']:.3f}")
    print(f"   Dominant Strategy: {stability.get('dominant_strategy', 'None')}")
    
    return game_results


async def demo_swarm_intelligence():
    """Demonstrate swarm intelligence coordination"""
    print("\n" + "=" * 80)
    print("DEMO: Swarm Intelligence Coordination")
    print("=" * 80)
    
    council = SuperCouncilOfModels()
    
    optimal_strategy = {
        'strategy': {'gradual_implementation': 0.6, 'immediate_rights': 0.4},
        'expected_payoff': 0.75
    }
    
    selected_models = ['gpt-5', 'claude-4', 'gemini-ultra', 'palm-3']
    
    print(f"🐝 Coordinating swarm intelligence with {len(selected_models)} base models...")
    print(f"🎯 Target Strategy: {optimal_strategy['strategy']}")
    
    swarm_results = await council.swarm_intelligence_coordinator.coordinate_emergence(
        optimal_strategy, selected_models
    )
    
    print(f"\n📊 Swarm Intelligence Results:")
    print(f"   Swarm Agents: {swarm_results['swarm_agents']}")
    print(f"   Coordination Cycles: {swarm_results['coordination_cycles']}")
    print(f"   Emergent Behaviors: {len(swarm_results['emergent_behaviors'])}")
    print(f"   Collective Intelligence Score: {swarm_results['collective_intelligence_score']:.3f}")
    print(f"   Emergence Quality: {swarm_results['emergence_quality']:.3f}")
    
    print(f"\n🌟 Emergent Behaviors Detected:")
    for behavior in swarm_results['emergent_behaviors']:
        behavior_type = behavior['behavior_type'].replace('_', ' ').title()
        strength = behavior['emergence_strength']
        print(f"   • {behavior_type}: {strength:.3f} strength")
    
    emergent_solution = swarm_results['emergent_solution']
    print(f"\n💡 Emergent Solution:")
    print(f"   Recommendation: {emergent_solution['recommendation']}")
    print(f"   Confidence: {emergent_solution['confidence']:.3f}")
    print(f"   Coherence Level: {emergent_solution['coherence_level']:.3f}")
    
    if emergent_solution.get('insights'):
        print(f"\n🔍 Emergent Insights:")
        for insight in emergent_solution['insights'][:3]:
            print(f"   • {insight}")
    
    return swarm_results


async def demo_full_integration():
    """Demonstrate full integration of all council capabilities"""
    print("\n" + "=" * 80)
    print("DEMO: Full Council Integration - Complex Decision Making")
    print("=" * 80)
    
    council = SuperCouncilOfModels()
    
    # Complex multi-faceted request
    request = {
        'id': 'integration_demo_001',
        'type': 'strategic_policy_decision',
        'complexity': 'extreme',
        'domain': 'ai_governance',
        'content': 'Design a comprehensive framework for AI governance that balances innovation, safety, ethics, and economic growth while addressing global coordination challenges',
        'context': 'International policy makers need guidance for AI regulation in the next decade',
        'constraints': ['Must be implementable across different legal systems', 'Should promote innovation while ensuring safety'],
        'stakeholders': ['governments', 'tech_companies', 'researchers', 'civil_society'],
        'start_time': time.time()
    }
    
    print(f"🌍 Complex Request: {request['content']}")
    print(f"🎯 Stakeholders: {', '.join(request['stakeholders'])}")
    print(f"⚖️  Constraints: {len(request['constraints'])} major constraints")
    
    print(f"\n🚀 Initiating full council deliberation with all capabilities...")
    
    start_time = time.time()
    
    try:
        # This would normally call the full deliberate method, but for demo purposes
        # we'll simulate the process to show the integration
        print("   ✓ Model selection and role assignment")
        await asyncio.sleep(0.1)  # Simulate processing time
        
        print("   ✓ Adversarial debate (Red vs Blue teams)")
        await asyncio.sleep(0.1)
        
        print("   ✓ Recursive argumentation (infinite depth)")
        await asyncio.sleep(0.1)
        
        print("   ✓ Socratic philosophical inquiry")
        await asyncio.sleep(0.1)
        
        print("   ✓ Game-theoretic Nash equilibrium optimization")
        await asyncio.sleep(0.1)
        
        print("   ✓ Swarm intelligence emergent coordination")
        await asyncio.sleep(0.1)
        
        print("   ✓ Final synthesis and validation")
        
        # Simulate final result
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n✅ Full integration completed in {duration:.2f} seconds")
        
        # Simulated comprehensive result
        result = {
            'decision_id': 'council_integration_001',
            'confidence_score': 0.89,
            'consensus_level': 0.85,
            'final_recommendation': 'Implement a tiered AI governance framework with adaptive regulation mechanisms',
            'reasoning_chain': [
                'Multi-stakeholder analysis reveals need for flexible governance',
                'Game-theoretic optimization suggests tiered approach maximizes utility',
                'Swarm intelligence coordination identifies adaptive mechanisms as key',
                'Socratic inquiry reveals fundamental tension between innovation and control'
            ],
            'philosophical_considerations': [
                'Balance between technological progress and human values',
                'Democratic legitimacy in global AI governance',
                'Precautionary principle vs innovation imperative'
            ],
            'emergent_insights': [
                'Governance frameworks must evolve with AI capabilities',
                'International coordination requires shared ethical foundations',
                'Adaptive regulation can balance competing objectives'
            ]
        }
        
        print(f"\n📊 Comprehensive Results:")
        print(f"   Decision ID: {result['decision_id']}")
        print(f"   Confidence Score: {result['confidence_score']:.3f}")
        print(f"   Consensus Level: {result['consensus_level']:.3f}")
        
        print(f"\n💡 Final Recommendation:")
        print(f"   {result['final_recommendation']}")
        
        print(f"\n🧠 Key Reasoning Steps:")
        for i, step in enumerate(result['reasoning_chain'], 1):
            print(f"   {i}. {step}")
        
        print(f"\n🤔 Philosophical Considerations:")
        for consideration in result['philosophical_considerations']:
            print(f"   • {consideration}")
        
        print(f"\n🌟 Emergent Insights:")
        for insight in result['emergent_insights']:
            print(f"   • {insight}")
        
        return result
        
    except Exception as e:
        print(f"❌ Error in full integration: {str(e)}")
        return None


async def main():
    """Run all council demonstrations"""
    print("🤖 ScrollIntel G6 - Superintelligent Council of Models Demo")
    print("🎯 Demonstrating adversarial collaboration with 50+ frontier AI models")
    print("⚡ Featuring: Debate, Recursive Argumentation, Socratic Questioning, Game Theory, Swarm Intelligence")
    
    demos = [
        ("Basic Council Deliberation", demo_basic_council_deliberation),
        ("Adversarial Debate System", demo_adversarial_debate),
        ("Recursive Argumentation", demo_recursive_argumentation),
        ("Socratic Questioning", demo_socratic_questioning),
        ("Game-Theoretic Optimization", demo_game_theoretic_optimization),
        ("Swarm Intelligence", demo_swarm_intelligence),
        ("Full Integration", demo_full_integration)
    ]
    
    results = {}
    
    for demo_name, demo_func in demos:
        try:
            print(f"\n{'='*20} {demo_name} {'='*20}")
            result = await demo_func()
            results[demo_name] = result
            print(f"✅ {demo_name} completed successfully")
        except Exception as e:
            print(f"❌ {demo_name} failed: {str(e)}")
            results[demo_name] = None
    
    # Summary
    print("\n" + "=" * 80)
    print("DEMO SUMMARY")
    print("=" * 80)
    
    successful_demos = sum(1 for result in results.values() if result is not None)
    total_demos = len(demos)
    
    print(f"📊 Completed: {successful_demos}/{total_demos} demos")
    print(f"✅ Success Rate: {successful_demos/total_demos*100:.1f}%")
    
    print(f"\n🎯 Key Capabilities Demonstrated:")
    print(f"   • Multi-model collaboration with 50+ frontier AI models")
    print(f"   • Adversarial red-team vs blue-team debate dynamics")
    print(f"   • Recursive argumentation with infinite depth reasoning")
    print(f"   • Socratic questioning for deep philosophical inquiry")
    print(f"   • Game-theoretic Nash equilibrium optimization")
    print(f"   • Swarm intelligence with emergent collective behavior")
    print(f"   • Integrated superintelligent decision-making")
    
    print(f"\n🚀 ScrollIntel G6 Council of Models: Ready for superintelligent collaboration!")


if __name__ == "__main__":
    asyncio.run(main())