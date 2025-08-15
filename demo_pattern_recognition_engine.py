"""
Demo script for Pattern Recognition Engine

This script demonstrates the capabilities of the pattern recognition engine
including pattern recognition across innovations, pattern analysis and interpretation,
and pattern-based innovation optimization and enhancement.
"""

import asyncio
from datetime import datetime, timedelta
from scrollintel.engines.pattern_recognition_engine import PatternRecognitionEngine
from scrollintel.models.knowledge_integration_models import (
    KnowledgeItem, KnowledgeType, ConfidenceLevel, PatternType
)


async def demo_pattern_recognition():
    """Demonstrate pattern recognition engine capabilities"""
    
    print("🔍 Pattern Recognition Engine Demo")
    print("=" * 50)
    
    # Initialize engine
    engine = PatternRecognitionEngine()
    
    # Create sample knowledge items representing various innovations
    knowledge_items = [
        # AI/ML Research Cluster
        KnowledgeItem(
            id="ai_research_1",
            knowledge_type=KnowledgeType.RESEARCH_FINDING,
            content={
                "title": "Transformer Architecture Optimization",
                "description": "Research on optimizing transformer architectures for better performance and efficiency",
                "methodology": "experimental_comparison",
                "results": {"accuracy_improvement": 0.15, "efficiency_gain": 0.25, "parameter_reduction": 0.30},
                "domain": "artificial_intelligence"
            },
            source="Stanford AI Lab",
            timestamp=datetime.now() - timedelta(days=30),
            confidence=ConfidenceLevel.VERY_HIGH,
            tags=["transformer", "optimization", "artificial intelligence", "deep learning", "efficiency"]
        ),
        
        KnowledgeItem(
            id="ai_research_2",
            knowledge_type=KnowledgeType.RESEARCH_FINDING,
            content={
                "title": "Neural Architecture Search for Vision Tasks",
                "description": "Automated neural architecture search techniques for computer vision applications",
                "methodology": "automated_search",
                "results": {"best_architecture": "EfficientNet-V2", "accuracy": 0.94, "search_time_hours": 48},
                "domain": "computer_vision"
            },
            source="Google Research",
            timestamp=datetime.now() - timedelta(days=25),
            confidence=ConfidenceLevel.HIGH,
            tags=["neural architecture search", "computer vision", "automation", "efficiency", "deep learning"]
        ),
        
        KnowledgeItem(
            id="ai_experiment_1",
            knowledge_type=KnowledgeType.EXPERIMENTAL_RESULT,
            content={
                "title": "Large Language Model Fine-tuning Experiment",
                "description": "Experimental results from fine-tuning large language models on domain-specific tasks",
                "methodology": "controlled_experiment",
                "results": {"baseline_accuracy": 0.78, "fine_tuned_accuracy": 0.91, "training_time_hours": 72},
                "domain": "natural_language_processing"
            },
            source="OpenAI Research",
            timestamp=datetime.now() - timedelta(days=20),
            confidence=ConfidenceLevel.HIGH,
            tags=["large language models", "fine-tuning", "natural language processing", "transfer learning"]
        ),
        
        # Quantum Computing Cluster
        KnowledgeItem(
            id="quantum_research_1",
            knowledge_type=KnowledgeType.RESEARCH_FINDING,
            content={
                "title": "Quantum Error Correction Breakthrough",
                "description": "Novel quantum error correction codes with improved efficiency",
                "methodology": "theoretical_analysis",
                "results": {"error_rate_reduction": 0.85, "qubit_overhead": 0.40, "logical_qubit_fidelity": 0.999},
                "domain": "quantum_computing"
            },
            source="IBM Quantum Research",
            timestamp=datetime.now() - timedelta(days=15),
            confidence=ConfidenceLevel.VERY_HIGH,
            tags=["quantum computing", "error correction", "quantum algorithms", "fault tolerance"]
        ),
        
        KnowledgeItem(
            id="quantum_experiment_1",
            knowledge_type=KnowledgeType.EXPERIMENTAL_RESULT,
            content={
                "title": "Quantum Advantage Demonstration",
                "description": "Experimental demonstration of quantum advantage in optimization problems",
                "methodology": "comparative_experiment",
                "results": {"quantum_speedup": 1000, "problem_size": 100, "success_probability": 0.95},
                "domain": "quantum_computing"
            },
            source="Google Quantum AI",
            timestamp=datetime.now() - timedelta(days=10),
            confidence=ConfidenceLevel.HIGH,
            tags=["quantum computing", "quantum advantage", "optimization", "quantum algorithms"]
        ),
        
        # Biotechnology Cluster
        KnowledgeItem(
            id="biotech_research_1",
            knowledge_type=KnowledgeType.RESEARCH_FINDING,
            content={
                "title": "CRISPR Gene Editing Precision Enhancement",
                "description": "Research on improving CRISPR gene editing precision and reducing off-target effects",
                "methodology": "experimental_validation",
                "results": {"precision_improvement": 0.40, "off_target_reduction": 0.90, "success_rate": 0.95},
                "domain": "biotechnology"
            },
            source="Broad Institute",
            timestamp=datetime.now() - timedelta(days=8),
            confidence=ConfidenceLevel.VERY_HIGH,
            tags=["CRISPR", "gene editing", "biotechnology", "precision medicine", "genetic engineering"]
        ),
        
        KnowledgeItem(
            id="biotech_prototype_1",
            knowledge_type=KnowledgeType.PROTOTYPE_INSIGHT,
            content={
                "title": "AI-Driven Drug Discovery Platform",
                "description": "Prototype of AI system for accelerated drug discovery and development",
                "methodology": "prototype_development",
                "results": {"discovery_time_reduction": 0.60, "success_rate_improvement": 0.35, "cost_reduction": 0.45},
                "domain": "pharmaceutical"
            },
            source="DeepMind Health",
            timestamp=datetime.now() - timedelta(days=5),
            confidence=ConfidenceLevel.MEDIUM,
            tags=["artificial intelligence", "drug discovery", "pharmaceutical", "biotechnology", "automation"]
        ),
        
        # Cross-domain Innovation
        KnowledgeItem(
            id="cross_domain_1",
            knowledge_type=KnowledgeType.INNOVATION_CONCEPT,
            content={
                "title": "Quantum-Enhanced Machine Learning",
                "description": "Concept for combining quantum computing with machine learning for exponential speedups",
                "methodology": "theoretical_framework",
                "results": {"theoretical_speedup": "exponential", "feasibility_score": 0.70, "implementation_complexity": "high"},
                "domain": "quantum_ai"
            },
            source="MIT Quantum Computing Lab",
            timestamp=datetime.now() - timedelta(days=3),
            confidence=ConfidenceLevel.MEDIUM,
            tags=["quantum computing", "machine learning", "artificial intelligence", "quantum algorithms", "hybrid systems"]
        )
    ]
    
    print(f"\n📊 Dataset: {len(knowledge_items)} Knowledge Items")
    print("-" * 40)
    
    # Display knowledge items summary
    type_counts = {}
    for item in knowledge_items:
        type_name = item.knowledge_type.value
        type_counts[type_name] = type_counts.get(type_name, 0) + 1
    
    for type_name, count in type_counts.items():
        print(f"   {type_name.replace('_', ' ').title()}: {count}")
    
    print("\n🔍 Step 1: Recognizing All Pattern Types")
    print("-" * 40)
    
    # Recognize all patterns
    all_patterns_result = await engine.recognize_patterns(knowledge_items)
    
    print(f"🎯 Pattern Recognition Complete")
    print(f"   Patterns Found: {len(all_patterns_result.patterns_found)}")
    print(f"   Processing Time: {all_patterns_result.processing_time:.3f} seconds")
    print(f"   Overall Confidence: {all_patterns_result.confidence.value}")
    print(f"   Analysis Method: {all_patterns_result.analysis_method}")
    print()
    
    # Display patterns by type
    patterns_by_type = {}
    for pattern in all_patterns_result.patterns_found:
        pattern_type = pattern.pattern_type.value
        if pattern_type not in patterns_by_type:
            patterns_by_type[pattern_type] = []
        patterns_by_type[pattern_type].append(pattern)
    
    print("📋 Patterns by Type:")
    for pattern_type, patterns in patterns_by_type.items():
        print(f"   {pattern_type.replace('_', ' ').title()}: {len(patterns)}")
        for pattern in patterns[:2]:  # Show first 2 patterns of each type
            print(f"     • {pattern.description[:60]}...")
            print(f"       Strength: {pattern.strength:.3f}, Confidence: {pattern.confidence.value}")
    print()
    
    print("💡 Generated Recommendations:")
    for i, recommendation in enumerate(all_patterns_result.recommendations, 1):
        print(f"   {i}. {recommendation}")
    print()
    
    print("\n🔬 Step 2: Analyzing Pattern Significance")
    print("-" * 40)
    
    # Analyze significance of strongest patterns
    strong_patterns = [p for p in all_patterns_result.patterns_found if p.strength > 0.5]
    
    for pattern in strong_patterns[:3]:  # Analyze top 3 strong patterns
        significance = await engine.analyze_pattern_significance(pattern, knowledge_items)
        
        print(f"🔍 Pattern: {pattern.id}")
        print(f"   Type: {pattern.pattern_type.value}")
        print(f"   Overall Significance: {significance['significance']:.3f}")
        print(f"   Evidence Count: {significance['evidence_count']}")
        
        print("   📊 Metrics:")
        for metric, value in significance['metrics'].items():
            print(f"     {metric.replace('_', ' ').title()}: {value:.3f}")
        
        if significance['analysis']:
            print("   💡 Analysis Insights:")
            for insight in significance['analysis']:
                print(f"     • {insight}")
        print()
    
    print("\n🧠 Step 3: Interpreting Patterns")
    print("-" * 40)
    
    # Interpret all patterns
    interpretation = await engine.interpret_patterns(all_patterns_result.patterns_found)
    
    print(f"🧠 Pattern Interpretation Complete")
    print(f"   Total Patterns: {interpretation['pattern_count']}")
    print(f"   Pattern Types: {', '.join([pt.value for pt in interpretation['pattern_types']])}")
    print()
    
    print("🔍 Pattern Type Interpretations:")
    for interp in interpretation['interpretations']:
        print(f"   📊 {interp['pattern_type'].replace('_', ' ').title()}:")
        print(f"     Count: {interp['pattern_count']}")
        for insight in interp['insights']:
            print(f"     • {insight}")
    print()
    
    print("💡 Cross-Pattern Insights:")
    for insight in interpretation['insights']:
        if any(word in insight.lower() for word in ['cross', 'together', 'combination']):
            print(f"   • {insight}")
    print()
    
    print("🎯 Interpretation Recommendations:")
    for i, rec in enumerate(interpretation['recommendations'][:5], 1):
        print(f"   {i}. {rec}")
    print()
    
    print("\n🚀 Step 4: Innovation Optimization")
    print("-" * 40)
    
    # Define innovation context
    innovation_context = {
        "innovation_type": "ai_quantum_hybrid_system",
        "current_capabilities": {
            "ai_performance": 0.85,
            "quantum_coherence_time": 100,  # microseconds
            "integration_efficiency": 0.60
        },
        "target_capabilities": {
            "ai_performance": 0.95,
            "quantum_coherence_time": 1000,  # microseconds
            "integration_efficiency": 0.90
        },
        "constraints": {
            "budget": 5000000,  # $5M
            "timeline_months": 18,
            "team_size": 25,
            "risk_tolerance": "medium"
        },
        "domain_focus": ["artificial_intelligence", "quantum_computing"]
    }
    
    # Optimize innovation based on patterns
    optimization = await engine.optimize_innovation_based_on_patterns(
        all_patterns_result.patterns_found, innovation_context
    )
    
    print(f"🚀 Innovation Optimization Complete")
    print(f"   Patterns Used: {optimization['patterns_used']}")
    print()
    
    print("🎯 Prioritized Optimization Recommendations:")
    for i, rec in enumerate(optimization['optimization_recommendations'][:5], 1):
        priority_icon = "🔥" if rec['priority_level'] == 'high' else "⚡" if rec['priority_level'] == 'medium' else "💡"
        print(f"   {i}. {priority_icon} {rec['recommendation']}")
        print(f"      Priority: {rec['priority_level'].title()} (Score: {rec['priority_score']:.3f})")
    print()
    
    print("🛡️ Risk Mitigations:")
    for i, mitigation in enumerate(optimization['risk_mitigations'][:3], 1):
        print(f"   {i}. {mitigation}")
    print()
    
    print("📈 Expected Impact:")
    impact = optimization['expected_impact']
    print(f"   Innovation Speed Improvement: {impact['innovation_speed_improvement']:.1%}")
    print(f"   Quality Improvement: {impact['quality_improvement']:.1%}")
    print(f"   Risk Reduction: {impact['risk_reduction']:.1%}")
    print(f"   Overall Impact Score: {impact['overall_impact_score']:.3f}")
    print()
    
    print("📋 Implementation Plan:")
    plan = optimization['implementation_plan']
    print(f"   Timeline: {plan['timeline']}")
    for phase in plan['phases']:
        print(f"   📅 {phase['phase']} ({phase['duration']})")
        print(f"      Recommendations: {len(phase['recommendations'])}")
    print()
    
    print("\n⚙️ Step 5: Pipeline Enhancement")
    print("-" * 40)
    
    # Define pipeline context
    pipeline_context = {
        "pipeline_type": "research_to_product_innovation",
        "current_stages": [
            "research_ideation",
            "proof_of_concept",
            "prototype_development",
            "validation_testing",
            "product_development",
            "market_launch"
        ],
        "current_bottlenecks": [
            "proof_of_concept_validation",
            "prototype_to_product_transition",
            "cross_domain_integration"
        ],
        "team_structure": {
            "researchers": 8,
            "engineers": 12,
            "product_managers": 3,
            "domain_experts": 5
        },
        "resources": {
            "compute_infrastructure": "high_performance_cluster",
            "lab_equipment": "quantum_and_ai_labs",
            "budget_allocation": "research_heavy"
        }
    }
    
    # Enhance pipeline based on patterns
    enhancement = await engine.enhance_innovation_pipeline(
        all_patterns_result.patterns_found, pipeline_context
    )
    
    print(f"⚙️ Pipeline Enhancement Complete")
    print()
    
    print("🔧 Pipeline Enhancements:")
    for i, enhancement_item in enumerate(enhancement['pipeline_enhancements'][:3], 1):
        print(f"   {i}. {enhancement_item}")
    print()
    
    print("📈 Process Improvements:")
    for i, improvement in enumerate(enhancement['process_improvements'][:3], 1):
        print(f"   {i}. {improvement}")
    print()
    
    print("🚧 Bottleneck Solutions:")
    for i, solution in enumerate(enhancement['bottleneck_solutions'][:3], 1):
        print(f"   {i}. {solution}")
    print()
    
    print("🎯 Optimization Strategy:")
    strategy = enhancement['optimization_strategy']
    print(f"   Overview: {strategy['strategy_overview']}")
    print(f"   Approach: {strategy['implementation_approach']}")
    print()
    
    print("\n📊 Step 6: Pattern Recognition Statistics")
    print("-" * 40)
    
    total_patterns = len(engine.patterns)
    cached_results = len(engine.pattern_cache)
    
    print(f"📊 Engine Statistics:")
    print(f"   Total Patterns Recognized: {total_patterns}")
    print(f"   Cached Results: {cached_results}")
    print()
    
    if total_patterns > 0:
        # Pattern strength distribution
        strengths = [p.strength for p in engine.patterns.values()]
        avg_strength = sum(strengths) / len(strengths)
        high_strength = len([s for s in strengths if s > 0.7])
        medium_strength = len([s for s in strengths if 0.4 <= s <= 0.7])
        low_strength = len([s for s in strengths if s < 0.4])
        
        print("💪 Pattern Strength Distribution:")
        print(f"   High (>0.7): {high_strength} ({high_strength/total_patterns:.1%})")
        print(f"   Medium (0.4-0.7): {medium_strength} ({medium_strength/total_patterns:.1%})")
        print(f"   Low (<0.4): {low_strength} ({low_strength/total_patterns:.1%})")
        print(f"   Average Strength: {avg_strength:.3f}")
        print()
        
        # Confidence distribution
        confidence_counts = {}
        for pattern in engine.patterns.values():
            conf = pattern.confidence.value
            confidence_counts[conf] = confidence_counts.get(conf, 0) + 1
        
        print("🎯 Confidence Distribution:")
        for conf, count in confidence_counts.items():
            print(f"   {conf.replace('_', ' ').title()}: {count} ({count/total_patterns:.1%})")
        print()
        
        # Predictive power analysis
        predictive_powers = [p.predictive_power for p in engine.patterns.values() if p.predictive_power > 0]
        if predictive_powers:
            avg_predictive_power = sum(predictive_powers) / len(predictive_powers)
            print(f"🔮 Average Predictive Power: {avg_predictive_power:.3f}")
    
    print("\n🎉 Demo Complete!")
    print("=" * 50)
    print("The Pattern Recognition Engine successfully:")
    print("✅ Recognized patterns across multiple innovation domains")
    print("✅ Analyzed pattern significance and reliability")
    print("✅ Interpreted patterns to extract meaningful insights")
    print("✅ Optimized innovation strategies based on patterns")
    print("✅ Enhanced innovation pipeline processes")
    print("✅ Provided comprehensive analytics and recommendations")
    print()
    print("🚀 Key Capabilities Demonstrated:")
    print("• Multi-domain pattern recognition (AI, Quantum, Biotech)")
    print("• Cross-pattern relationship identification")
    print("• Innovation optimization with prioritized recommendations")
    print("• Pipeline bottleneck identification and solutions")
    print("• Predictive pattern analysis for strategic planning")


if __name__ == "__main__":
    asyncio.run(demo_pattern_recognition())