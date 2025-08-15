#!/usr/bin/env python3
"""
Demo script for the PromptEnhancer system.
Showcases ML-based prompt improvement, context-aware suggestions, and feedback learning.
"""

import asyncio
import json
from datetime import datetime
from scrollintel.engines.visual_generation.utils.prompt_enhancer import (
    PromptEnhancer, EnhancementStrategy, SuggestionType
)


async def demo_basic_enhancement():
    """Demonstrate basic prompt enhancement functionality."""
    print("=" * 80)
    print("DEMO: Basic Prompt Enhancement")
    print("=" * 80)
    
    enhancer = PromptEnhancer()
    
    # Test prompts of varying complexity
    test_prompts = [
        "landscape",
        "a beautiful portrait",
        "futuristic city at sunset with dramatic lighting",
        "abstract art composition with geometric shapes and vibrant colors"
    ]
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n{i}. Testing prompt: '{prompt}'")
        print("-" * 60)
        
        try:
            result = await enhancer.enhance_prompt(
                prompt,
                strategy=EnhancementStrategy.MODERATE
            )
            
            print(f"Original:  {result.original_prompt}")
            print(f"Enhanced:  {result.enhanced_prompt}")
            print(f"Strategy:  {result.strategy_used.value}")
            print(f"Confidence: {result.overall_confidence:.2f}")
            print(f"Improvement Score: {result.improvement_score:.2f}")
            print(f"Suggestions Applied: {len(result.suggestions)}")
            
            if result.suggestions:
                print("\nTop Suggestions:")
                for j, suggestion in enumerate(result.suggestions[:3], 1):
                    print(f"  {j}. {suggestion.reasoning} (confidence: {suggestion.confidence:.2f})")
            
        except Exception as e:
            print(f"Error enhancing prompt: {e}")


async def demo_strategy_comparison():
    """Demonstrate different enhancement strategies."""
    print("\n" + "=" * 80)
    print("DEMO: Enhancement Strategy Comparison")
    print("=" * 80)
    
    enhancer = PromptEnhancer()
    test_prompt = "portrait of a person"
    
    strategies = [
        EnhancementStrategy.CONSERVATIVE,
        EnhancementStrategy.MODERATE,
        EnhancementStrategy.AGGRESSIVE
    ]
    
    print(f"Original prompt: '{test_prompt}'")
    print("-" * 60)
    
    for strategy in strategies:
        try:
            result = await enhancer.enhance_prompt(test_prompt, strategy=strategy)
            
            print(f"\n{strategy.value.upper()} Strategy:")
            print(f"Enhanced: {result.enhanced_prompt}")
            print(f"Suggestions: {len(result.suggestions)}")
            print(f"Confidence: {result.overall_confidence:.2f}")
            
        except Exception as e:
            print(f"Error with {strategy.value} strategy: {e}")


async def demo_context_aware_enhancement():
    """Demonstrate context-aware enhancement."""
    print("\n" + "=" * 80)
    print("DEMO: Context-Aware Enhancement")
    print("=" * 80)
    
    enhancer = PromptEnhancer()
    base_prompt = "landscape photography"
    
    contexts = [
        {
            'time_of_day': 'morning',
            'mood': 'peaceful',
            'purpose': 'artistic'
        },
        {
            'time_of_day': 'evening',
            'mood': 'dramatic',
            'purpose': 'commercial'
        },
        {
            'time_of_day': 'night',
            'mood': 'mysterious',
            'purpose': 'documentary'
        }
    ]
    
    print(f"Base prompt: '{base_prompt}'")
    print("-" * 60)
    
    for i, context in enumerate(contexts, 1):
        try:
            result = await enhancer.enhance_prompt(
                base_prompt,
                context=context,
                strategy=EnhancementStrategy.MODERATE
            )
            
            print(f"\n{i}. Context: {context}")
            print(f"Enhanced: {result.enhanced_prompt}")
            print(f"Context suggestions: {len([s for s in result.suggestions if s.suggestion_type == SuggestionType.CONTEXT_ENRICHMENT])}")
            
        except Exception as e:
            print(f"Error with context {i}: {e}")


async def demo_feedback_learning():
    """Demonstrate feedback learning system."""
    print("\n" + "=" * 80)
    print("DEMO: Feedback Learning System")
    print("=" * 80)
    
    enhancer = PromptEnhancer()
    
    # Show initial feedback weights
    print("Initial feedback weights:")
    for key, weight in enhancer.feedback_weights.items():
        print(f"  {key}: {weight:.3f}")
    
    # Simulate positive feedback
    print("\nSimulating positive feedback...")
    await enhancer.learn_from_feedback(
        enhancement_id=1,
        feedback="accepted",
        quality_score=0.9,
        user_rating=5
    )
    
    print("Updated feedback weights after positive feedback:")
    for key, weight in enhancer.feedback_weights.items():
        print(f"  {key}: {weight:.3f}")
    
    # Simulate negative feedback
    print("\nSimulating negative feedback...")
    await enhancer.learn_from_feedback(
        enhancement_id=2,
        feedback="rejected",
        quality_score=0.3,
        user_rating=2
    )
    
    print("Updated feedback weights after negative feedback:")
    for key, weight in enhancer.feedback_weights.items():
        print(f"  {key}: {weight:.3f}")


async def demo_batch_enhancement():
    """Demonstrate batch enhancement functionality."""
    print("\n" + "=" * 80)
    print("DEMO: Batch Enhancement")
    print("=" * 80)
    
    enhancer = PromptEnhancer()
    
    batch_prompts = [
        "portrait photography",
        "landscape at sunset",
        "abstract digital art",
        "architectural photography",
        "street photography"
    ]
    
    print(f"Enhancing {len(batch_prompts)} prompts in batch...")
    print("-" * 60)
    
    try:
        results = await enhancer.batch_enhance_prompts(
            batch_prompts,
            strategy=EnhancementStrategy.MODERATE
        )
        
        for i, (original, result) in enumerate(zip(batch_prompts, results), 1):
            print(f"\n{i}. {original}")
            print(f"   → {result.enhanced_prompt}")
            print(f"   Improvement: {result.improvement_score:.2f}, Confidence: {result.overall_confidence:.2f}")
            
    except Exception as e:
        print(f"Error in batch enhancement: {e}")


async def demo_enhancement_analytics():
    """Demonstrate enhancement analytics."""
    print("\n" + "=" * 80)
    print("DEMO: Enhancement Analytics")
    print("=" * 80)
    
    enhancer = PromptEnhancer()
    
    # Add some mock enhancement history
    enhancer.enhancement_history = [
        {
            'timestamp': datetime.now().isoformat(),
            'improvement_score': 0.8,
            'confidence': 0.9,
            'suggestions_count': 3,
            'strategy': 'moderate'
        },
        {
            'timestamp': datetime.now().isoformat(),
            'improvement_score': 0.75,
            'confidence': 0.85,
            'suggestions_count': 2,
            'strategy': 'conservative'
        },
        {
            'timestamp': datetime.now().isoformat(),
            'improvement_score': 0.9,
            'confidence': 0.95,
            'suggestions_count': 5,
            'strategy': 'aggressive'
        }
    ]
    
    try:
        analytics = await enhancer.get_enhancement_analytics(days=7)
        
        print("Enhancement Analytics (Last 7 days):")
        print("-" * 40)
        print(f"Total enhancements: {analytics['total_enhancements']}")
        print(f"Average improvement score: {analytics['average_improvement_score']:.3f}")
        print(f"Average confidence: {analytics['average_confidence']:.3f}")
        print(f"Average suggestions per enhancement: {analytics['average_suggestions_per_enhancement']:.1f}")
        
        if 'strategy_distribution' in analytics:
            print("\nStrategy distribution:")
            for strategy, count in analytics['strategy_distribution'].items():
                print(f"  {strategy}: {count}")
        
        if 'current_feedback_weights' in analytics:
            print("\nCurrent feedback weights:")
            for key, weight in analytics['current_feedback_weights'].items():
                print(f"  {key}: {weight:.3f}")
                
    except Exception as e:
        print(f"Error getting analytics: {e}")


async def demo_suggestion_types():
    """Demonstrate different types of suggestions."""
    print("\n" + "=" * 80)
    print("DEMO: Suggestion Types")
    print("=" * 80)
    
    enhancer = PromptEnhancer()
    
    # Test prompts designed to trigger different suggestion types
    test_cases = [
        ("low quality prompt", "Should trigger quality improvement suggestions"),
        ("portrait", "Should trigger style and technical suggestions"),
        ("landscape photo", "Should trigger composition and technical suggestions"),
        ("abstract", "Should trigger specificity and style suggestions")
    ]
    
    for prompt, expected in test_cases:
        print(f"\nTesting: '{prompt}' ({expected})")
        print("-" * 60)
        
        try:
            result = await enhancer.enhance_prompt(
                prompt,
                strategy=EnhancementStrategy.MODERATE
            )
            
            # Group suggestions by type
            suggestion_types = {}
            for suggestion in result.suggestions:
                stype = suggestion.suggestion_type.value
                if stype not in suggestion_types:
                    suggestion_types[stype] = []
                suggestion_types[stype].append(suggestion)
            
            print(f"Enhanced: {result.enhanced_prompt}")
            print("Suggestion types found:")
            for stype, suggestions in suggestion_types.items():
                print(f"  {stype}: {len(suggestions)} suggestions")
                for suggestion in suggestions[:2]:  # Show top 2
                    print(f"    - {suggestion.reasoning}")
                    
        except Exception as e:
            print(f"Error: {e}")


async def main():
    """Run all demos."""
    print("ScrollIntel Prompt Enhancement System Demo")
    print("=" * 80)
    print("This demo showcases the intelligent prompt enhancement capabilities")
    print("including ML-based improvements, context-aware suggestions, and feedback learning.")
    print()
    
    try:
        await demo_basic_enhancement()
        await demo_strategy_comparison()
        await demo_context_aware_enhancement()
        await demo_feedback_learning()
        await demo_batch_enhancement()
        await demo_enhancement_analytics()
        await demo_suggestion_types()
        
        print("\n" + "=" * 80)
        print("DEMO COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print("The PromptEnhancer system demonstrates:")
        print("✓ ML-based prompt improvement logic")
        print("✓ Context-aware suggestion generation")
        print("✓ Multiple enhancement strategies")
        print("✓ Feedback learning and adaptation")
        print("✓ Batch processing capabilities")
        print("✓ Comprehensive analytics")
        print("✓ Multiple suggestion types")
        
    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())