"""
Demo script for Knowledge Synthesis Framework

This script demonstrates the capabilities of the knowledge synthesis framework
including research integration, correlation identification, knowledge synthesis,
and validation processes.
"""

import asyncio
import json
from datetime import datetime
from scrollintel.engines.knowledge_synthesis_framework import KnowledgeSynthesisFramework
from scrollintel.models.knowledge_integration_models import SynthesisRequest


async def demo_knowledge_synthesis():
    """Demonstrate knowledge synthesis framework capabilities"""
    
    print("🧠 Knowledge Synthesis Framework Demo")
    print("=" * 50)
    
    # Initialize framework
    framework = KnowledgeSynthesisFramework()
    
    # Sample research findings
    research_findings = [
        {
            "title": "Deep Learning Performance Optimization",
            "description": "Comprehensive study on optimizing deep learning model performance through architectural improvements",
            "methodology": "experimental_comparison",
            "results": {
                "accuracy_improvement": 0.15,
                "training_time_reduction": 0.30,
                "memory_usage_reduction": 0.25
            },
            "keywords": ["deep learning", "optimization", "performance", "architecture"],
            "domain": "artificial intelligence",
            "peer_reviewed": True,
            "replicated": True,
            "sample_size": 1000,
            "statistical_significance": 0.99,
            "source": "AI Research Lab Stanford"
        },
        {
            "title": "Neural Architecture Search for Computer Vision",
            "description": "Automated neural architecture search techniques for computer vision tasks",
            "methodology": "automated_search",
            "results": {
                "best_architecture": "EfficientNet-B7",
                "accuracy": 0.94,
                "parameter_efficiency": 0.85
            },
            "keywords": ["neural architecture search", "computer vision", "automation", "efficiency"],
            "domain": "artificial intelligence",
            "peer_reviewed": True,
            "replicated": False,
            "sample_size": 500,
            "statistical_significance": 0.95,
            "source": "Google Research"
        },
        {
            "title": "Transfer Learning in Natural Language Processing",
            "description": "Analysis of transfer learning effectiveness across different NLP tasks",
            "methodology": "comparative_analysis",
            "results": {
                "average_improvement": 0.22,
                "best_pretrained_model": "BERT-Large",
                "task_generalization": 0.78
            },
            "keywords": ["transfer learning", "NLP", "BERT", "generalization"],
            "domain": "natural language processing",
            "peer_reviewed": True,
            "replicated": True,
            "sample_size": 800,
            "statistical_significance": 0.97,
            "source": "OpenAI Research"
        }
    ]
    
    # Sample experimental results
    experimental_results = [
        {
            "experiment_id": "exp_dl_opt_001",
            "title": "Deep Learning Optimization Experiment",
            "description": "Controlled experiment testing various optimization techniques",
            "methodology": "controlled_experiment",
            "results": {
                "baseline_accuracy": 0.82,
                "optimized_accuracy": 0.91,
                "training_time_hours": 24,
                "convergence_epochs": 150
            },
            "parameters": {
                "learning_rate": 0.001,
                "batch_size": 64,
                "optimizer": "AdamW",
                "regularization": "dropout_0.3"
            },
            "replicated": True,
            "sample_size": 300,
            "metadata": {
                "duration": "2 weeks",
                "resources": "8x V100 GPUs",
                "dataset": "ImageNet-1K"
            }
        },
        {
            "experiment_id": "exp_nas_cv_002",
            "title": "Neural Architecture Search Computer Vision Experiment",
            "description": "Experiment evaluating NAS-discovered architectures on vision tasks",
            "methodology": "architecture_evaluation",
            "results": {
                "top1_accuracy": 0.93,
                "top5_accuracy": 0.98,
                "inference_time_ms": 12,
                "model_size_mb": 45
            },
            "parameters": {
                "search_space": "MobileNet-based",
                "search_strategy": "evolutionary",
                "evaluation_metric": "accuracy_latency_tradeoff"
            },
            "replicated": False,
            "sample_size": 200,
            "metadata": {
                "duration": "1 week",
                "resources": "TPU v3 Pod",
                "dataset": "CIFAR-100"
            }
        }
    ]
    
    print("\n📚 Step 1: Integrating Research Findings")
    print("-" * 40)
    
    # Integrate research findings
    research_items = await framework.integrate_research_findings(research_findings)
    
    for item in research_items:
        print(f"✅ Integrated: {item.content['title']}")
        print(f"   Confidence: {item.confidence.value}")
        print(f"   Tags: {', '.join(item.tags[:3])}...")
        print(f"   ID: {item.id}")
        print()
    
    print(f"📊 Total research findings integrated: {len(research_items)}")
    
    print("\n🧪 Step 2: Integrating Experimental Results")
    print("-" * 40)
    
    # Integrate experimental results
    experiment_items = await framework.integrate_experimental_results(experimental_results)
    
    for item in experiment_items:
        print(f"✅ Integrated: {item.content['title']}")
        print(f"   Experiment ID: {item.content['experiment_id']}")
        print(f"   Confidence: {item.confidence.value}")
        print(f"   ID: {item.id}")
        print()
    
    print(f"📊 Total experimental results integrated: {len(experiment_items)}")
    
    print("\n🔗 Step 3: Identifying Knowledge Correlations")
    print("-" * 40)
    
    # Identify correlations
    correlations = await framework.identify_knowledge_correlations()
    
    print(f"📊 Found {len(correlations)} correlations")
    for correlation in correlations:
        print(f"🔗 Correlation: {correlation.id}")
        print(f"   Items: {correlation.item_ids}")
        print(f"   Type: {correlation.correlation_type}")
        print(f"   Strength: {correlation.strength:.3f}")
        print(f"   Confidence: {correlation.confidence.value}")
        print()
    
    print("\n🧬 Step 4: Synthesizing Knowledge")
    print("-" * 40)
    
    # Create synthesis request for all research findings
    research_synthesis_request = SynthesisRequest(
        id="research_synthesis_demo",
        source_knowledge_ids=[item.id for item in research_items],
        synthesis_goal="Synthesize AI research findings to identify common patterns and insights",
        method_preferences=["integration", "aggregation"],
        priority="high"
    )
    
    # Perform synthesis
    research_synthesis = await framework.synthesize_knowledge(research_synthesis_request)
    
    print(f"🧬 Research Synthesis Complete")
    print(f"   ID: {research_synthesis.id}")
    print(f"   Method: {research_synthesis.synthesis_method}")
    print(f"   Sources: {len(research_synthesis.source_items)}")
    print(f"   Confidence: {research_synthesis.confidence.value}")
    print(f"   Quality Score: {research_synthesis.quality_score:.3f}")
    print(f"   Insights Generated: {len(research_synthesis.insights)}")
    print()
    
    print("💡 Key Insights:")
    for i, insight in enumerate(research_synthesis.insights, 1):
        print(f"   {i}. {insight}")
    print()
    
    # Create synthesis request combining research and experiments
    combined_synthesis_request = SynthesisRequest(
        id="combined_synthesis_demo",
        source_knowledge_ids=[item.id for item in research_items + experiment_items],
        synthesis_goal="Combine research findings with experimental validation",
        method_preferences=["integration"],
        priority="high"
    )
    
    combined_synthesis = await framework.synthesize_knowledge(combined_synthesis_request)
    
    print(f"🧬 Combined Synthesis Complete")
    print(f"   ID: {combined_synthesis.id}")
    print(f"   Method: {combined_synthesis.synthesis_method}")
    print(f"   Sources: {len(combined_synthesis.source_items)}")
    print(f"   Quality Score: {combined_synthesis.quality_score:.3f}")
    print()
    
    print("\n✅ Step 5: Validating Knowledge Quality")
    print("-" * 40)
    
    # Validate a few knowledge items
    validation_methods = ["consistency", "completeness", "reliability"]
    
    for item in research_items[:2]:  # Validate first 2 research items
        validation_result = await framework.validate_knowledge(item.id, validation_methods)
        
        print(f"🔍 Validation: {item.content['title'][:50]}...")
        print(f"   Valid: {'✅' if validation_result.is_valid else '❌'}")
        print(f"   Score: {validation_result.validation_score:.3f}")
        print(f"   Confidence: {validation_result.confidence.value}")
        
        if validation_result.issues_found:
            print(f"   Issues: {len(validation_result.issues_found)}")
            for issue in validation_result.issues_found[:2]:  # Show first 2 issues
                print(f"     - {issue}")
        
        if validation_result.recommendations:
            print(f"   Recommendations: {len(validation_result.recommendations)}")
            for rec in validation_result.recommendations[:2]:  # Show first 2 recommendations
                print(f"     - {rec}")
        print()
    
    print("\n🕸️ Step 6: Creating Knowledge Graph")
    print("-" * 40)
    
    # Create knowledge graph
    knowledge_graph = await framework.create_knowledge_graph()
    
    print(f"🕸️ Knowledge Graph Created")
    print(f"   Nodes: {len(knowledge_graph.nodes)}")
    print(f"   Edges: {len(knowledge_graph.edges)}")
    print(f"   Created: {knowledge_graph.created_at}")
    print()
    
    print("📊 Node Distribution:")
    node_types = {}
    for node in knowledge_graph.nodes:
        node_type = node.knowledge_type.value
        node_types[node_type] = node_types.get(node_type, 0) + 1
    
    for node_type, count in node_types.items():
        print(f"   {node_type}: {count}")
    print()
    
    if knowledge_graph.edges:
        print("🔗 Edge Information:")
        for edge in knowledge_graph.edges[:3]:  # Show first 3 edges
            print(f"   {edge.correlation_type}: strength {edge.strength:.3f}")
    
    print("\n📈 Step 7: Framework Statistics")
    print("-" * 40)
    
    total_knowledge = len(framework.knowledge_store)
    total_correlations = len(framework.correlations)
    total_synthesized = len(framework.synthesized_knowledge)
    total_validated = len(framework.validation_cache)
    
    print(f"📊 Knowledge Base Statistics:")
    print(f"   Total Knowledge Items: {total_knowledge}")
    print(f"   Total Correlations: {total_correlations}")
    print(f"   Total Synthesized Items: {total_synthesized}")
    print(f"   Total Validated Items: {total_validated}")
    print()
    
    # Calculate average confidence
    if framework.knowledge_store:
        confidence_scores = {
            "low": 0, "medium": 0, "high": 0, "very_high": 0
        }
        
        for item in framework.knowledge_store.values():
            confidence_scores[item.confidence.value] += 1
        
        print("🎯 Confidence Distribution:")
        for level, count in confidence_scores.items():
            percentage = (count / total_knowledge) * 100
            print(f"   {level.replace('_', ' ').title()}: {count} ({percentage:.1f}%)")
    
    print("\n🎉 Demo Complete!")
    print("=" * 50)
    print("The Knowledge Synthesis Framework successfully:")
    print("✅ Integrated research findings and experimental results")
    print("✅ Identified correlations between knowledge items")
    print("✅ Synthesized knowledge with insights generation")
    print("✅ Validated knowledge quality and reliability")
    print("✅ Created comprehensive knowledge graph")
    print("✅ Provided detailed analytics and statistics")


if __name__ == "__main__":
    asyncio.run(demo_knowledge_synthesis())