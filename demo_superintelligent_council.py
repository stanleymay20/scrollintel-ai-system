#!/usr/bin/env python3
"""
Demo script for the Superintelligent Council of Models with Adversarial Collaboration

This script demonstrates the key capabilities of the ScrollIntel G6 Superintelligent Council:
- 50+ Frontier AI Models orchestration
- Adversarial Debate System with Red-Team vs Blue-Team dynamics
- Recursive Argumentation with infinite depth reasoning chains
- Socratic Questioning Engine for deep philosophical inquiry
- Game-Theoretic Decision Making with Nash equilibrium optimization
- Swarm Intelligence Coordination with emergent collective behavior
"""

import asyncio
import json
import time
from datetime import datetime
from typing import Dict, Any

from scrollintel.core.super_council_of_models import SuperCouncilOfModels, ArgumentationDepth


class SuperCouncilDemo:
    """Demo class for the Superintelligent Council of Models"""
    
    def __init__(self):
        self.council = SuperCouncilOfModels()
        
    def print_header(self, title: str):
        """Print a formatted header"""
        print("\n" + "="*80)
        print(f"  {title}")
        print("="*80)
    
    def print_section(self, title: str):
        """Print a formatted section header"""
        print(f"\n--- {title} ---")
    
    async def demo_initialization(self):
        """Demonstrate council initialization and model inventory"""
        self.print_header("SUPERINTELLIGENT COUNCIL INITIALIZATION")
        
        print(f"Council Engine ID: {self.council.engine_id}")
        print(f"Council Name: {self.council.name}")
        print(f"Total Frontier Models: {len(self.council.frontier_models)}")
        
        # Show status
        status = self.council.get_status()
        print(f"\nCouncil Status: {status['status']}")
        print(f"Engines Status:")
        for engine, engine_status in status['engines_status'].items():
            print(f"  - {engine}: {engine_status}")
        
        # Show some frontier models
        self.print_section("Sample Frontier Models")
        sample_models = list(self.council.frontier_models.items())[:10]
        
        for model_id, capability in sample_models:
            print(f"\n{model_id}:")
            print(f"  Name: {capability.model_name}")
            print(f"  Reasoning: {capability.reasoning_strength:.3f}")
            print(f"  Creativity: {capability.creativity_score:.3f}")
            print(f"  Accuracy: {capability.factual_accuracy:.3f}")
            print(f"  Philosophy: {capability.philosophical_depth:.3f}")
            print(f"  Robustness: {capability.adversarial_robustness:.3f}")
            print(f"  Specializations: {', '.join(capability.specializations[:3])}")
        
        # Show specialized models
        self.print_section("Specialized Domain Models")
        specialized_models = [model for model in self.council.frontier_models.keys() 
                            if model.startswith('specialist_')]
        
        print(f"Total Specialized Models: {len(specialized_models)}")
        
        # Group by domain
        domains = {}
        for model_id in specialized_models[:20]:  # Show first 20
            capability = self.council.frontier_models[model_id]
            for spec in capability.specializations:
                if spec != 'superintelligence':
                    if spec not in domains:
                        domains[spec] = []
                    domains[spec].append(capability.model_name)
        
        for domain, models in list(domains.items())[:10]:  # Show first 10 domains
            print(f"  {domain}: {len(models)} models")
    
    async def demo_model_selection(self):
        """Demonstrate intelligent model selection for different request types"""
        self.print_header("INTELLIGENT MODEL SELECTION")
        
        test_requests = [
            {
                'type': 'ethical_dilemma',
                'complexity': 'high',
                'domain': 'ethical_analysis',
                'content': 'Should AI systems be granted legal rights?'
            },
            {
                'type': 'scientific_reasoning',
                'complexity': 'high',
                'domain': 'scientific_reasoning',
                'content': 'What are the implications of quantum consciousness theories?'
            },
            {
                'type': 'creative_problem',
                'complexity': 'medium',
                'domain': 'creative_synthesis',
                'content': 'Design a novel approach to climate change mitigation'
            },
            {
                'type': 'strategic_decision',
                'complexity': 'high',
                'domain': 'strategic_planning',
                'content': 'Should humanity pursue Mars colonization as a priority?'
            }
        ]
        
        for i, request in enumerate(test_requests):
            self.print_section(f"Request {i+1}: {request['type']}")
            print(f"Content: {request['content']}")
            print(f"Domain: {request['domain']}, Complexity: {request['complexity']}")
            
            selected_models = await self.council._select_models_for_deliberation(request)
            print(f"\nSelected Models ({len(selected_models)}):")
            
            for model_id in selected_models:
                capability = self.council.frontier_models[model_id]
                print(f"  - {capability.model_name}")
                print(f"    Reasoning: {capability.reasoning_strength:.3f}, "
                      f"Domain Match: {'✓' if request['domain'] in capability.specializations else '✗'}")
            
            # Show role assignments
            role_assignments = await self.council._assign_debate_roles(selected_models, request)
            print(f"\nRole Assignments:")
            for model_id, role in role_assignments.items():
                model_name = self.council.frontier_models[model_id].model_name
                print(f"  - {model_name}: {role.value}")
    
    async def demo_adversarial_debate(self):
        """Demonstrate the adversarial debate system"""
        self.print_header("ADVERSARIAL DEBATE SYSTEM")
        
        debate_request = {
            'id': 'demo_debate_1',
            'type': 'philosophical_inquiry',
            'complexity': 'high',
            'domain': 'philosophical_inquiry',
            'content': 'Is consciousness an emergent property or a fundamental aspect of reality?',
            'start_time': time.time()
        }
        
        print(f"Debate Question: {debate_request['content']}")
        
        # Select models and assign roles
        selected_models = await self.council._select_models_for_deliberation(debate_request)
        role_assignments = await self.council._assign_debate_roles(selected_models, debate_request)
        
        print(f"\nDebate Participants ({len(selected_models)} models):")
        red_team = [model for model, role in role_assignments.items() if role.value == 'red_team']
        blue_team = [model for model, role in role_assignments.items() if role.value == 'blue_team']
        moderators = [model for model, role in role_assignments.items() 
                     if role.value in ['moderator', 'socratic_questioner', 'juror']]
        
        print(f"Red Team (Challengers): {len(red_team)} models")
        for model in red_team:
            print(f"  - {self.council.frontier_models[model].model_name}")
        
        print(f"Blue Team (Defenders): {len(blue_team)} models")
        for model in blue_team:
            print(f"  - {self.council.frontier_models[model].model_name}")
        
        print(f"Moderators & Jurors: {len(moderators)} models")
        for model in moderators:
            role = role_assignments[model].value
            print(f"  - {self.council.frontier_models[model].model_name} ({role})")
        
        # Conduct debate
        print(f"\nConducting Adversarial Debate...")
        start_time = time.time()
        
        debate_results = await self.council.adversarial_debate_engine.conduct_debate(
            debate_request, role_assignments
        )
        
        end_time = time.time()
        
        print(f"Debate completed in {end_time - start_time:.2f} seconds")
        print(f"Total Rounds: {debate_results['total_rounds']}")
        print(f"Red Team Average Score: {debate_results['red_team_average_score']:.3f}")
        print(f"Blue Team Average Score: {debate_results['blue_team_average_score']:.3f}")
        print(f"Final Convergence: {debate_results['final_convergence']:.3f}")
        print(f"Agreement Level: {debate_results['agreement_level']:.3f}")
        
        # Show some reasoning
        if debate_results['debate_reasoning']:
            print(f"\nSample Debate Reasoning:")
            for i, reason in enumerate(debate_results['debate_reasoning'][:5]):
                print(f"  {i+1}. {reason}")
    
    async def demo_recursive_argumentation(self):
        """Demonstrate recursive argumentation with infinite depth"""
        self.print_header("RECURSIVE ARGUMENTATION ENGINE")
        
        # Create mock debate results for demonstration
        mock_debate_results = {
            'rounds': [
                {
                    'red_argument': {
                        'main_points': ['Consciousness is emergent from complex neural networks'],
                        'confidence': 0.8
                    },
                    'blue_argument': {
                        'main_points': ['Consciousness is a fundamental property of matter'],
                        'confidence': 0.75
                    }
                }
            ]
        }
        
        print("Testing Recursive Argumentation with different depth levels...")
        
        depth_levels = [
            ArgumentationDepth.SURFACE,
            ArgumentationDepth.INTERMEDIATE,
            ArgumentationDepth.DEEP,
            ArgumentationDepth.INFINITE
        ]
        
        for depth in depth_levels:
            print(f"\n--- {depth.value.upper()} DEPTH ---")
            start_time = time.time()
            
            refined_arguments = await self.council.recursive_argumentation_engine.deepen_arguments(
                mock_debate_results, depth
            )
            
            end_time = time.time()
            
            print(f"Processing time: {end_time - start_time:.3f} seconds")
            print(f"Recursive depth achieved: {refined_arguments['recursive_depth_achieved']}")
            print(f"New reasoning chains: {len(refined_arguments['reasoning_chains'])}")
            
            if refined_arguments['reasoning_chains']:
                print("Sample deeper reasoning:")
                for i, reasoning in enumerate(refined_arguments['reasoning_chains'][:3]):
                    print(f"  {i+1}. {reasoning}")
    
    async def demo_socratic_questioning(self):
        """Demonstrate Socratic questioning for deep philosophical inquiry"""
        self.print_header("SOCRATIC QUESTIONING ENGINE")
        
        refined_arguments = {
            'reasoning_chains': [
                'Consciousness emerges from neural complexity',
                'Information integration creates subjective experience',
                'Quantum effects may play a role in consciousness'
            ],
            'philosophical_insights': [
                'The hard problem of consciousness remains unsolved',
                'Subjective experience cannot be reduced to objective description'
            ]
        }
        
        inquiry_request = {
            'content': 'What is the nature of consciousness and subjective experience?',
            'domain': 'philosophy_of_mind'
        }
        
        print(f"Inquiry Topic: {inquiry_request['content']}")
        print(f"Philosophical Domain: {inquiry_request['domain']}")
        
        start_time = time.time()
        
        socratic_results = await self.council.socratic_questioning_engine.conduct_inquiry(
            refined_arguments, inquiry_request
        )
        
        end_time = time.time()
        
        print(f"\nSocratic Inquiry completed in {end_time - start_time:.3f} seconds")
        print(f"Questions asked: {len(socratic_results['questions_asked'])}")
        print(f"Insights generated: {len(socratic_results['insights'])}")
        print(f"Assumptions challenged: {len(socratic_results['assumptions_challenged'])}")
        print(f"Clarity level: {socratic_results['clarity_level']:.3f}")
        print(f"Philosophical depth: {socratic_results['philosophical_depth']:.3f}")
        
        print(f"\nSample Socratic Questions:")
        for i, question in enumerate(socratic_results['questions_asked'][:3]):
            print(f"  {i+1}. {question.question}")
            print(f"     Domain: {question.philosophical_domain}")
            print(f"     Depth Level: {question.depth_level}")
        
        print(f"\nSample Insights:")
        for i, insight in enumerate(socratic_results['insights'][:3]):
            print(f"  {i+1}. {insight}")
    
    async def demo_game_theoretic_optimization(self):
        """Demonstrate game-theoretic optimization with Nash equilibrium"""
        self.print_header("GAME-THEORETIC OPTIMIZATION")
        
        refined_arguments = {
            'reasoning_chains': ['Argument chain 1', 'Argument chain 2']
        }
        
        socratic_insights = {
            'insights': ['Insight 1', 'Insight 2'],
            'clarity_level': 0.8
        }
        
        print("Finding Nash Equilibrium for optimal decision strategy...")
        
        start_time = time.time()
        
        equilibrium_result = await self.council.game_theoretic_optimizer.find_nash_equilibrium(
            refined_arguments, socratic_insights
        )
        
        end_time = time.time()
        
        print(f"Optimization completed in {end_time - start_time:.3f} seconds")
        print(f"Players: {equilibrium_result['players']}")
        print(f"Optimal Strategy: {equilibrium_result['optimal_decision']}")
        print(f"Expected Utility: {equilibrium_result['expected_utility']:.3f}")
        print(f"Convergence Achieved: {equilibrium_result['convergence_achieved']}")
        
        print(f"\nStrategy Options by Player:")
        for player, strategies in equilibrium_result['strategies'].items():
            print(f"  {player}: {', '.join(strategies)}")
        
        print(f"\nMixed Strategy Equilibrium:")
        for player, strategy_probs in equilibrium_result['equilibrium_strategy']['mixed_strategies'].items():
            print(f"  {player}:")
            for strategy, prob in strategy_probs.items():
                print(f"    {strategy}: {prob:.3f}")
    
    async def demo_swarm_intelligence(self):
        """Demonstrate swarm intelligence coordination"""
        self.print_header("SWARM INTELLIGENCE COORDINATION")
        
        optimal_strategy = {
            'optimal_strategy': 'collaborative_emergence',
            'expected_utility': 0.85
        }
        
        selected_models = ['gpt-5', 'claude-4', 'gemini-ultra', 'palm-3', 'llama-3-400b']
        
        print(f"Coordinating swarm intelligence with {len(selected_models)} base models")
        print(f"Optimal Strategy: {optimal_strategy['optimal_strategy']}")
        print(f"Expected Utility: {optimal_strategy['expected_utility']}")
        
        start_time = time.time()
        
        swarm_results = await self.council.swarm_intelligence_coordinator.coordinate_emergence(
            optimal_strategy, selected_models
        )
        
        end_time = time.time()
        
        print(f"\nSwarm coordination completed in {end_time - start_time:.3f} seconds")
        print(f"Swarm Agents: {swarm_results['swarm_agents']}")
        print(f"Coordination Cycles: {swarm_results['coordination_cycles']}")
        print(f"Collective Intelligence Score: {swarm_results['collective_intelligence_score']:.3f}")
        print(f"Emergent Confidence: {swarm_results['emergent_confidence']:.3f}")
        print(f"Coherence Level: {swarm_results['coherence_level']:.3f}")
        
        print(f"\nEmergent Behaviors Detected:")
        for behavior, detected in swarm_results['emergent_behaviors'].items():
            if behavior != 'insights':
                status = "✓" if detected else "✗"
                print(f"  {status} {behavior.replace('_', ' ').title()}")
        
        print(f"\nEmergent Insights:")
        for i, insight in enumerate(swarm_results['emergent_behaviors']['insights'][:3]):
            print(f"  {i+1}. {insight}")
        
        print(f"\nFinal Recommendation: {swarm_results['recommendation']}")
    
    async def demo_full_deliberation(self):
        """Demonstrate a complete council deliberation"""
        self.print_header("COMPLETE COUNCIL DELIBERATION")
        
        complex_request = {
            'id': 'demo_full_deliberation',
            'type': 'complex_ethical_strategic_decision',
            'complexity': 'high',
            'domain': 'ethical_analysis',
            'content': 'Should humanity develop artificial general intelligence (AGI) given the potential risks and benefits?',
            'context': 'This decision involves ethical considerations, existential risks, economic implications, and the future of human civilization.',
            'start_time': time.time()
        }
        
        print(f"Complex Decision: {complex_request['content']}")
        print(f"Context: {complex_request['context']}")
        print(f"Type: {complex_request['type']}")
        print(f"Complexity: {complex_request['complexity']}")
        print(f"Domain: {complex_request['domain']}")
        
        print(f"\nInitiating full superintelligent council deliberation...")
        print("This will orchestrate all council capabilities:")
        print("  1. Model Selection & Role Assignment")
        print("  2. Adversarial Debate (Red vs Blue Teams)")
        print("  3. Recursive Argumentation (Infinite Depth)")
        print("  4. Socratic Questioning (Philosophical Inquiry)")
        print("  5. Game-Theoretic Optimization (Nash Equilibrium)")
        print("  6. Swarm Intelligence Coordination (Emergent Behavior)")
        print("  7. Final Synthesis & Decision")
        
        start_time = time.time()
        
        # Execute full deliberation
        final_decision = await self.council.deliberate(complex_request)
        
        end_time = time.time()
        
        print(f"\n🎯 DELIBERATION COMPLETE 🎯")
        print(f"Total Processing Time: {end_time - start_time:.2f} seconds")
        print(f"Decision ID: {final_decision['decision_id']}")
        print(f"Confidence Score: {final_decision['confidence_score']:.3f}")
        print(f"Consensus Level: {final_decision['consensus_level']:.3f}")
        
        print(f"\n📋 FINAL RECOMMENDATION:")
        print(f"{final_decision['final_recommendation']}")
        
        print(f"\n🧠 REASONING CHAIN ({len(final_decision['reasoning_chain'])} steps):")
        for i, reason in enumerate(final_decision['reasoning_chain'][:5]):
            print(f"  {i+1}. {reason}")
        if len(final_decision['reasoning_chain']) > 5:
            print(f"  ... and {len(final_decision['reasoning_chain']) - 5} more steps")
        
        print(f"\n🔍 PHILOSOPHICAL CONSIDERATIONS:")
        for i, consideration in enumerate(final_decision['philosophical_considerations'][:3]):
            print(f"  {i+1}. {consideration}")
        
        print(f"\n✨ EMERGENT INSIGHTS:")
        for i, insight in enumerate(final_decision['emergent_insights'][:3]):
            print(f"  {i+1}. {insight}")
        
        # Show deliberation history
        print(f"\n📊 DELIBERATION METRICS:")
        if self.council.debate_history:
            latest_record = self.council.debate_history[-1]
            print(f"  Models Used: {len(latest_record['models_used'])}")
            print(f"  Debate Rounds: {latest_record['debate_rounds']}")
            print(f"  Socratic Questions: {latest_record['socratic_questions']}")
            print(f"  Process Duration: {latest_record['process_duration']:.2f}s")
    
    async def demo_performance_analysis(self):
        """Demonstrate performance analysis and metrics"""
        self.print_header("PERFORMANCE ANALYSIS")
        
        print("Council Performance Metrics:")
        print(f"Total Deliberations: {len(self.council.debate_history)}")
        
        if self.council.model_performance_metrics:
            print(f"Models with Performance Data: {len(self.council.model_performance_metrics)}")
            
            # Show top performing models
            top_models = sorted(
                self.council.model_performance_metrics.items(),
                key=lambda x: x[1]['deliberations_participated'],
                reverse=True
            )[:5]
            
            print(f"\nTop 5 Most Active Models:")
            for model_id, metrics in top_models:
                model_name = self.council.frontier_models[model_id].model_name
                print(f"  {model_name}: {metrics['deliberations_participated']} deliberations")
        
        # Show engine status
        status = self.council.get_status()
        print(f"\nEngine Status:")
        for engine, engine_status in status['engines_status'].items():
            print(f"  {engine.replace('_', ' ').title()}: {engine_status}")
        
        print(f"\nTotal Frontier Models Available: {status['total_models']}")
        
        # Show capability distribution
        print(f"\nCapability Distribution Analysis:")
        reasoning_scores = [cap.reasoning_strength for cap in self.council.frontier_models.values()]
        creativity_scores = [cap.creativity_score for cap in self.council.frontier_models.values()]
        accuracy_scores = [cap.factual_accuracy for cap in self.council.frontier_models.values()]
        
        print(f"  Average Reasoning Strength: {sum(reasoning_scores)/len(reasoning_scores):.3f}")
        print(f"  Average Creativity Score: {sum(creativity_scores)/len(creativity_scores):.3f}")
        print(f"  Average Factual Accuracy: {sum(accuracy_scores)/len(accuracy_scores):.3f}")
        
        # Count specializations
        all_specializations = []
        for cap in self.council.frontier_models.values():
            all_specializations.extend(cap.specializations)
        
        from collections import Counter
        spec_counts = Counter(all_specializations)
        
        print(f"\nTop 10 Specialization Areas:")
        for spec, count in spec_counts.most_common(10):
            print(f"  {spec.replace('_', ' ').title()}: {count} models")


async def main():
    """Main demo function"""
    print("🚀 ScrollIntel G6 - Superintelligent Council of Models Demo")
    print("=" * 80)
    print("Demonstrating 50+ Frontier AI Models with Adversarial Collaboration")
    print("Features: Debate, Recursive Reasoning, Socratic Inquiry, Game Theory, Swarm Intelligence")
    
    demo = SuperCouncilDemo()
    
    try:
        # Run all demonstrations
        await demo.demo_initialization()
        await demo.demo_model_selection()
        await demo.demo_adversarial_debate()
        await demo.demo_recursive_argumentation()
        await demo.demo_socratic_questioning()
        await demo.demo_game_theoretic_optimization()
        await demo.demo_swarm_intelligence()
        await demo.demo_full_deliberation()
        await demo.demo_performance_analysis()
        
        print("\n" + "="*80)
        print("🎉 SUPERINTELLIGENT COUNCIL DEMO COMPLETED SUCCESSFULLY! 🎉")
        print("="*80)
        print("\nThe ScrollIntel G6 Superintelligent Council demonstrates:")
        print("✅ 50+ Frontier AI Models orchestration")
        print("✅ Adversarial Debate with Red vs Blue team dynamics")
        print("✅ Recursive Argumentation with infinite depth reasoning")
        print("✅ Socratic Questioning for deep philosophical inquiry")
        print("✅ Game-Theoretic optimization with Nash equilibrium")
        print("✅ Swarm Intelligence with emergent collective behavior")
        print("✅ Complete integration and superintelligent decision-making")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())