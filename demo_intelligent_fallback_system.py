"""
Demo script for the Intelligent Fallback Content Generation System.
Demonstrates all components working together to provide seamless user experiences.
"""

import asyncio
import logging
import time
import random
from datetime import datetime
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import all components
from scrollintel.core.intelligent_fallback_manager import (
    intelligent_fallback_manager, ContentContext, ContentType
)
from scrollintel.core.progressive_content_loader import (
    progressive_content_loader, ContentChunk, ContentPriority, create_content_chunk
)
from scrollintel.core.smart_cache_manager import (
    smart_cache_manager, cache_get, cache_set, StalenessLevel
)
from scrollintel.core.workflow_alternative_engine import (
    workflow_alternative_engine, WorkflowContext, DifficultyLevel
)
from scrollintel.core.intelligent_fallback_integration import (
    intelligent_fallback_integration, IntegratedFallbackRequest, FallbackStrategy,
    with_intelligent_fallback, get_intelligent_fallback
)


class DemoDataService:
    """Mock data service for demonstration."""
    
    def __init__(self):
        self.failure_rate = 0.3  # 30% failure rate for demo
        self.slow_response_rate = 0.2  # 20% slow responses
    
    async def get_sales_data(self, user_id: str = None) -> Dict[str, Any]:
        """Simulate getting sales data with potential failures."""
        await asyncio.sleep(0.1)  # Simulate network delay
        
        if random.random() < self.failure_rate:
            raise Exception("Database connection timeout")
        
        if random.random() < self.slow_response_rate:
            await asyncio.sleep(2.0)  # Simulate slow response
        
        return {
            "total_sales": 125000,
            "monthly_data": [
                {"month": "Jan", "sales": 10000},
                {"month": "Feb", "sales": 12000},
                {"month": "Mar", "sales": 15000},
                {"month": "Apr", "sales": 13000}
            ],
            "top_products": [
                {"name": "Product A", "sales": 5000},
                {"name": "Product B", "sales": 4500},
                {"name": "Product C", "sales": 3200}
            ]
        }
    
    async def generate_chart(self, data: Dict[str, Any], chart_type: str = "bar") -> Dict[str, Any]:
        """Simulate chart generation with potential failures."""
        await asyncio.sleep(0.5)  # Simulate processing time
        
        if random.random() < self.failure_rate:
            raise Exception("Chart rendering service unavailable")
        
        return {
            "type": chart_type,
            "title": "Sales Chart",
            "data": data.get("monthly_data", []),
            "config": {"responsive": True, "animated": True}
        }
    
    async def create_analysis_report(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate analysis report creation with potential failures."""
        await asyncio.sleep(1.0)  # Simulate analysis time
        
        if random.random() < self.failure_rate:
            raise Exception("Analysis engine overloaded")
        
        return {
            "summary": "Sales performance shows positive growth trend",
            "insights": [
                "March showed the highest sales increase",
                "Product A continues to be the top performer",
                "Overall growth rate is 15% compared to last quarter"
            ],
            "recommendations": [
                "Focus marketing efforts on Product A",
                "Investigate factors behind March's success",
                "Consider expanding Product A line"
            ],
            "confidence": 0.85
        }


async def demo_basic_fallback_generation():
    """Demonstrate basic fallback content generation."""
    print("\n" + "="*60)
    print("DEMO 1: Basic Fallback Content Generation")
    print("="*60)
    
    # Test different content types
    content_types = [
        (ContentType.CHART, {"chart_type": "bar", "title": "Sales Chart"}),
        (ContentType.TABLE, {"columns": ["Product", "Sales", "Growth"]}),
        (ContentType.ANALYSIS, {"analysis_type": "trend"}),
        (ContentType.REPORT, {"report_type": "monthly_summary"})
    ]
    
    for content_type, request_data in content_types:
        print(f"\nGenerating fallback for {content_type.value}...")
        
        context = ContentContext(
            user_id="demo_user",
            content_type=content_type,
            original_request=request_data,
            error_context=Exception("Service temporarily unavailable")
        )
        
        start_time = time.time()
        fallback = await intelligent_fallback_manager.generate_fallback_content(context)
        generation_time = time.time() - start_time
        
        print(f"  ✓ Generated {fallback.quality.value} quality fallback in {generation_time:.3f}s")
        print(f"  ✓ Confidence: {fallback.confidence:.2f}")
        print(f"  ✓ User message: {fallback.user_message}")
        print(f"  ✓ Suggested actions: {len(fallback.suggested_actions)} available")
        
        if fallback.alternatives:
            print(f"  ✓ Workflow alternatives: {len(fallback.alternatives)} provided")


async def demo_progressive_content_loading():
    """Demonstrate progressive content loading."""
    print("\n" + "="*60)
    print("DEMO 2: Progressive Content Loading")
    print("="*60)
    
    data_service = DemoDataService()
    
    # Create content chunks with different priorities and load times
    chunks = [
        create_content_chunk(
            chunk_id="sales_summary",
            content_type=ContentType.TEXT,
            loader_function=lambda: asyncio.sleep(0.2) or {"summary": "Q1 Sales: $125K"},
            priority=ContentPriority.CRITICAL
        ),
        create_content_chunk(
            chunk_id="sales_chart",
            content_type=ContentType.CHART,
            loader_function=lambda: data_service.generate_chart({"monthly_data": []}),
            priority=ContentPriority.HIGH
        ),
        create_content_chunk(
            chunk_id="detailed_analysis",
            content_type=ContentType.ANALYSIS,
            loader_function=lambda: data_service.create_analysis_report({}),
            priority=ContentPriority.MEDIUM
        )
    ]
    
    print("Loading content progressively...")
    print("Progress updates:")
    
    async for progress in progressive_content_loader.load_content_progressively(
        progressive_content_loader.create_loading_request(
            user_id="demo_user",
            content_chunks=chunks,
            timeout_seconds=10.0
        )
    ):
        print(f"  {progress.stage.value}: {progress.progress_percentage:.1f}% - {progress.current_operation}")
        
        if progress.partial_results:
            loaded_count = sum(1 for result in progress.partial_results.values() 
                             if result.get("content") is not None)
            print(f"    └─ {loaded_count}/{len(chunks)} chunks loaded")
        
        if progress.stage.value in ["complete", "failed"]:
            break
    
    print(f"✓ Progressive loading completed")


async def demo_smart_caching():
    """Demonstrate smart caching with staleness detection."""
    print("\n" + "="*60)
    print("DEMO 3: Smart Caching with Staleness Detection")
    print("="*60)
    
    # Cache some test data
    test_data = {
        "timestamp": datetime.utcnow().isoformat(),
        "data": "Fresh content",
        "version": 1
    }
    
    print("Setting cache entry...")
    await cache_set("demo_key", test_data, ttl_seconds=5, tags=["demo", "test"])
    
    # Immediate retrieval
    print("Retrieving immediately...")
    cached_value, staleness = await cache_get("demo_key")
    print(f"  ✓ Retrieved: {cached_value['data']}")
    print(f"  ✓ Staleness: {staleness.value}")
    
    # Wait for staleness
    print("Waiting for content to become stale...")
    await asyncio.sleep(2)
    
    cached_value, staleness = await cache_get("demo_key")
    print(f"  ✓ Retrieved: {cached_value['data']}")
    print(f"  ✓ Staleness: {staleness.value}")
    
    # Test staleness tolerance
    print("Testing staleness tolerance...")
    cached_value, staleness = await cache_get("demo_key", staleness_tolerance=StalenessLevel.VERY_STALE)
    if cached_value:
        print(f"  ✓ Accepted stale content: {staleness.value}")
    else:
        print(f"  ✗ Rejected stale content: {staleness.value}")
    
    # Cache statistics
    stats = smart_cache_manager.get_stats()
    print(f"  ✓ Cache stats: {stats.total_entries} entries, {stats.hit_rate:.2f} hit rate")


async def demo_workflow_alternatives():
    """Demonstrate workflow alternative suggestions."""
    print("\n" + "="*60)
    print("DEMO 4: Workflow Alternative Suggestions")
    print("="*60)
    
    # Test different failure scenarios
    scenarios = [
        {
            "workflow": "analyze_sales_data",
            "failure": "timeout",
            "skill": DifficultyLevel.BEGINNER,
            "time": 30
        },
        {
            "workflow": "create_visualization",
            "failure": "chart_generation_failed",
            "skill": DifficultyLevel.INTERMEDIATE,
            "time": 15
        },
        {
            "workflow": "generate_report",
            "failure": "system_overload",
            "skill": DifficultyLevel.ADVANCED,
            "time": 60
        }
    ]
    
    for scenario in scenarios:
        print(f"\nScenario: {scenario['workflow']} failed due to {scenario['failure']}")
        
        context = WorkflowContext(
            user_id="demo_user",
            original_workflow=scenario["workflow"],
            failure_reason=scenario["failure"],
            user_skill_level=scenario["skill"],
            time_constraints=scenario["time"]
        )
        
        result = await workflow_alternative_engine.suggest_alternatives(context)
        
        print(f"  ✓ Found {len(result.alternatives)} alternatives")
        print(f"  ✓ Confidence: {result.confidence_score:.2f}")
        print(f"  ✓ Reasoning: {result.reasoning}")
        
        for i, alt in enumerate(result.alternatives[:2], 1):  # Show top 2
            print(f"    {i}. {alt.name} ({alt.difficulty.value})")
            print(f"       Time: {alt.estimated_total_time_minutes}min, Success: {alt.success_probability:.0%}")
            print(f"       Steps: {len(alt.steps)} steps")


async def demo_integrated_fallback_system():
    """Demonstrate the integrated fallback system."""
    print("\n" + "="*60)
    print("DEMO 5: Integrated Fallback System")
    print("="*60)
    
    data_service = DemoDataService()
    
    # Test different strategies
    strategies = [
        FallbackStrategy.IMMEDIATE_FALLBACK,
        FallbackStrategy.CACHED_CONTENT,
        FallbackStrategy.WORKFLOW_ALTERNATIVE,
        FallbackStrategy.HYBRID
    ]
    
    for strategy in strategies:
        print(f"\nTesting {strategy.value} strategy...")
        
        request = IntegratedFallbackRequest(
            request_id=f"demo_{strategy.value}",
            user_id="demo_user",
            content_type=ContentType.CHART,
            original_function=data_service.generate_chart,
            original_args=({"data": "test"}, "bar"),
            original_kwargs={},
            failure_context=Exception("Chart service unavailable"),
            preferred_strategy=strategy,
            max_wait_time_seconds=10.0
        )
        
        start_time = time.time()
        result = await intelligent_fallback_integration.handle_content_failure(request)
        execution_time = time.time() - start_time
        
        print(f"  ✓ Strategy: {result.strategy_used.value}")
        print(f"  ✓ Success: {result.success}")
        print(f"  ✓ Quality: {result.fallback_quality}")
        print(f"  ✓ Time: {execution_time:.3f}s")
        print(f"  ✓ Cache hit: {result.cache_hit}")
        
        if result.workflow_alternatives:
            print(f"  ✓ Alternatives: {len(result.workflow_alternatives)} provided")
        
        if result.user_message:
            print(f"  ✓ Message: {result.user_message}")


async def demo_decorator_usage():
    """Demonstrate decorator usage for automatic fallbacks."""
    print("\n" + "="*60)
    print("DEMO 6: Decorator Usage for Automatic Fallbacks")
    print("="*60)
    
    data_service = DemoDataService()
    
    # Apply decorator to make functions bulletproof
    @with_intelligent_fallback(ContentType.CHART, max_wait_time=5.0)
    async def get_sales_chart(user_id: str = None):
        """Get sales chart with automatic fallback."""
        data = await data_service.get_sales_data(user_id)
        return await data_service.generate_chart(data, "bar")
    
    @with_intelligent_fallback(ContentType.ANALYSIS, strategy=FallbackStrategy.WORKFLOW_ALTERNATIVE)
    async def get_sales_analysis(user_id: str = None):
        """Get sales analysis with workflow alternatives."""
        data = await data_service.get_sales_data(user_id)
        return await data_service.create_analysis_report(data)
    
    # Test the decorated functions
    print("Testing decorated chart function...")
    try:
        chart_result = await get_sales_chart("demo_user")
        print(f"  ✓ Chart function succeeded: {type(chart_result)}")
        if isinstance(chart_result, dict) and "type" in chart_result:
            print(f"    Chart type: {chart_result['type']}")
    except Exception as e:
        print(f"  ✗ Chart function failed: {e}")
    
    print("\nTesting decorated analysis function...")
    try:
        analysis_result = await get_sales_analysis("demo_user")
        print(f"  ✓ Analysis function succeeded: {type(analysis_result)}")
        if isinstance(analysis_result, dict) and "summary" in analysis_result:
            print(f"    Summary: {analysis_result['summary'][:50]}...")
    except Exception as e:
        print(f"  ✗ Analysis function failed: {e}")


async def demo_performance_metrics():
    """Demonstrate performance metrics and monitoring."""
    print("\n" + "="*60)
    print("DEMO 7: Performance Metrics and Monitoring")
    print("="*60)
    
    # Get cache statistics
    cache_stats = smart_cache_manager.get_stats()
    print("Cache Statistics:")
    print(f"  Total entries: {cache_stats.total_entries}")
    print(f"  Total size: {cache_stats.total_size_bytes / 1024:.1f} KB")
    print(f"  Hit rate: {cache_stats.hit_rate:.2%}")
    print(f"  Average entry size: {cache_stats.average_entry_size:.0f} bytes")
    
    # Get strategy performance
    strategy_stats = intelligent_fallback_integration.get_strategy_performance_stats()
    print("\nStrategy Performance:")
    for strategy, stats in strategy_stats.items():
        if stats["total_attempts"] > 0:
            print(f"  {strategy}:")
            print(f"    Success rate: {stats['success_rate']:.2%}")
            print(f"    Avg response time: {stats['avg_response_time']:.3f}s")
            print(f"    Total attempts: {stats['total_attempts']}")
    
    # Get progressive loader performance
    loader_stats = progressive_content_loader.get_performance_stats()
    if loader_stats:
        print("\nProgressive Loader Performance:")
        for metric, stats in loader_stats.items():
            print(f"  {metric}:")
            print(f"    Avg load time: {stats['avg_load_time']:.3f}s")
            print(f"    Min/Max: {stats['min_load_time']:.3f}s / {stats['max_load_time']:.3f}s")
    
    # Get user statistics
    user_stats = workflow_alternative_engine.get_user_statistics("demo_user")
    print("\nUser Workflow Statistics:")
    print(f"  Total attempts: {user_stats['total_attempts']}")
    print(f"  Success rate: {user_stats['success_rate']:.2%}")
    print(f"  Skill level: {user_stats['skill_level']}")


async def demo_real_world_scenario():
    """Demonstrate a real-world scenario with multiple failures and recoveries."""
    print("\n" + "="*60)
    print("DEMO 8: Real-World Scenario - Dashboard Loading")
    print("="*60)
    
    data_service = DemoDataService()
    data_service.failure_rate = 0.6  # Increase failure rate for demo
    
    print("Simulating dashboard loading with high failure rate...")
    
    # Simulate loading a complex dashboard
    dashboard_components = [
        ("sales_summary", ContentType.TEXT, lambda: data_service.get_sales_data()),
        ("sales_chart", ContentType.CHART, lambda: data_service.generate_chart({})),
        ("analysis_report", ContentType.ANALYSIS, lambda: data_service.create_analysis_report({})),
        ("data_table", ContentType.TABLE, lambda: {"columns": ["A", "B"], "rows": [["1", "2"]]})
    ]
    
    results = {}
    total_start_time = time.time()
    
    for component_name, content_type, loader_func in dashboard_components:
        print(f"\nLoading {component_name}...")
        
        try:
            # Try original function first
            start_time = time.time()
            result = await loader_func()
            load_time = time.time() - start_time
            results[component_name] = {
                "success": True,
                "method": "original",
                "time": load_time,
                "content": result
            }
            print(f"  ✓ Original function succeeded in {load_time:.3f}s")
            
        except Exception as e:
            print(f"  ✗ Original function failed: {e}")
            
            # Use intelligent fallback
            fallback_result = await get_intelligent_fallback(
                content_type=content_type,
                original_function=loader_func,
                args=(),
                kwargs={},
                error=e,
                user_id="demo_user",
                strategy=FallbackStrategy.HYBRID
            )
            
            results[component_name] = {
                "success": fallback_result.success,
                "method": f"fallback_{fallback_result.strategy_used.value}",
                "time": fallback_result.loading_time_seconds,
                "quality": fallback_result.fallback_quality,
                "content": fallback_result.content
            }
            
            if fallback_result.success:
                print(f"  ✓ Fallback succeeded ({fallback_result.strategy_used.value}) in {fallback_result.loading_time_seconds:.3f}s")
                print(f"    Quality: {fallback_result.fallback_quality}")
            else:
                print(f"  ✗ Fallback failed")
    
    total_time = time.time() - total_start_time
    
    # Summary
    print(f"\nDashboard Loading Summary:")
    print(f"  Total time: {total_time:.3f}s")
    successful_components = sum(1 for r in results.values() if r["success"])
    print(f"  Success rate: {successful_components}/{len(results)} ({successful_components/len(results):.1%})")
    
    fallback_used = sum(1 for r in results.values() if "fallback" in r["method"])
    print(f"  Fallbacks used: {fallback_used}/{len(results)}")
    
    print("\nComponent Details:")
    for name, result in results.items():
        status = "✓" if result["success"] else "✗"
        print(f"  {status} {name}: {result['method']} ({result['time']:.3f}s)")


async def main():
    """Run all demos."""
    print("ScrollIntel Intelligent Fallback Content Generation System")
    print("Demo Script - Showcasing Bulletproof User Experience")
    print("=" * 80)
    
    try:
        # Run all demos
        await demo_basic_fallback_generation()
        await demo_progressive_content_loading()
        await demo_smart_caching()
        await demo_workflow_alternatives()
        await demo_integrated_fallback_system()
        await demo_decorator_usage()
        await demo_performance_metrics()
        await demo_real_world_scenario()
        
        print("\n" + "="*80)
        print("✓ All demos completed successfully!")
        print("The Intelligent Fallback System ensures users never see failures.")
        print("="*80)
        
    except Exception as e:
        logger.error(f"Demo failed: {e}", exc_info=True)
        print(f"\n✗ Demo failed: {e}")
    
    finally:
        # Cleanup
        smart_cache_manager.shutdown()


if __name__ == "__main__":
    asyncio.run(main())