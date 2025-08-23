"""
Advanced Analytics Demo

This demo showcases the comprehensive advanced analytics capabilities including
graph analytics, semantic search, pattern recognition, and predictive analytics.
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, List, Any

from scrollintel.models.advanced_analytics_models import (
    AdvancedAnalyticsRequest, AnalyticsType,
    GraphAnalysisRequest, SemanticQuery, PatternRecognitionRequest,
    PredictiveAnalyticsRequest, PatternType, PredictionType
)
from scrollintel.engines.advanced_analytics_engine import AdvancedAnalyticsEngine


class AdvancedAnalyticsDemo:
    """Comprehensive demo of advanced analytics capabilities."""
    
    def __init__(self):
        self.analytics_engine = AdvancedAnalyticsEngine()
        self.demo_results = {}
    
    async def run_complete_demo(self):
        """Run complete advanced analytics demonstration."""
        print("🚀 Starting Advanced Analytics Demo")
        print("=" * 60)
        
        try:
            # 1. Graph Analytics Demo
            await self.demo_graph_analytics()
            
            # 2. Semantic Search Demo
            await self.demo_semantic_search()
            
            # 3. Pattern Recognition Demo
            await self.demo_pattern_recognition()
            
            # 4. Predictive Analytics Demo
            await self.demo_predictive_analytics()
            
            # 5. Comprehensive Analysis Demo
            await self.demo_comprehensive_analysis()
            
            # 6. Business Intelligence Summary
            await self.demo_business_intelligence()
            
            # 7. Performance Summary
            await self.demo_performance_metrics()
            
            print("\n✅ Advanced Analytics Demo Completed Successfully!")
            print("=" * 60)
            
            # Save demo results
            await self.save_demo_results()
            
        except Exception as e:
            print(f"❌ Demo failed: {str(e)}")
            raise
    
    async def demo_graph_analytics(self):
        """Demonstrate graph analytics capabilities."""
        print("\n📊 Graph Analytics Demo")
        print("-" * 40)
        
        # Build enterprise graph
        print("Building enterprise graph from multiple data sources...")
        data_sources = ["crm_data", "erp_data", "financial_data", "operational_data"]
        
        graph_stats = await self.analytics_engine.graph_engine.build_enterprise_graph(data_sources)
        
        print(f"✓ Graph built: {graph_stats['nodes_count']} nodes, {graph_stats['edges_count']} edges")
        print(f"  Execution time: {graph_stats['execution_time_ms']:.2f}ms")
        
        # Perform different types of graph analysis
        analysis_types = [
            ("centrality_analysis", "Identifying key nodes and influencers"),
            ("community_detection", "Detecting business clusters and segments"),
            ("influence_analysis", "Analyzing influence propagation patterns"),
            ("anomaly_detection", "Finding structural anomalies")
        ]
        
        graph_results = {}
        
        for analysis_type, description in analysis_types:
            print(f"\n{description}...")
            
            request = GraphAnalysisRequest(
                analysis_type=analysis_type,
                max_depth=3,
                min_confidence=0.6
            )
            
            result = await self.analytics_engine.graph_engine.analyze_complex_relationships(request)
            
            print(f"✓ {analysis_type}: {len(result.insights)} insights generated")
            print(f"  Confidence: {result.confidence_score:.2f}")
            print(f"  Key insight: {result.insights[0] if result.insights else 'No insights'}")
            
            graph_results[analysis_type] = {
                "insights_count": len(result.insights),
                "confidence": result.confidence_score,
                "execution_time": result.execution_time_ms
            }
        
        # Detect business opportunities
        print("\nDetecting business opportunities through graph analysis...")
        opportunities = await self.analytics_engine.graph_engine.detect_business_opportunities({
            "focus_areas": ["growth", "efficiency", "innovation"],
            "industry": "technology"
        })
        
        print(f"✓ Found {len(opportunities)} business opportunities")
        if opportunities:
            top_opportunity = opportunities[0]
            print(f"  Top opportunity: {top_opportunity.title}")
            print(f"  Confidence: {top_opportunity.confidence:.2f}")
        
        self.demo_results["graph_analytics"] = {
            "graph_statistics": graph_stats,
            "analysis_results": graph_results,
            "opportunities_found": len(opportunities)
        }
    
    async def demo_semantic_search(self):
        """Demonstrate semantic search capabilities."""
        print("\n🔍 Semantic Search Demo")
        print("-" * 40)
        
        # Index enterprise data
        print("Indexing enterprise data for semantic search...")
        data_sources = ["knowledge_base", "customer_communications", "financial_reports", "operational_data"]
        
        index_stats = await self.analytics_engine.search_engine.index_enterprise_data(data_sources)
        
        print(f"✓ Indexed {index_stats['documents_indexed']} documents")
        print(f"  Total size: {index_stats['total_size_mb']:.2f} MB")
        print(f"  Execution time: {index_stats['execution_time_ms']:.2f}ms")
        
        # Perform semantic searches
        search_queries = [
            "customer satisfaction and product quality issues",
            "revenue growth opportunities and market trends",
            "operational efficiency and cost optimization",
            "risk management and compliance requirements"
        ]
        
        search_results = {}
        
        for query_text in search_queries:
            print(f"\nSearching: '{query_text}'...")
            
            query = SemanticQuery(
                query_text=query_text,
                max_results=10,
                similarity_threshold=0.6
            )
            
            result = await self.analytics_engine.search_engine.semantic_search(query)
            
            print(f"✓ Found {result.total_results} results")
            print(f"  Execution time: {result.execution_time_ms:.2f}ms")
            if result.results:
                top_result = result.results[0]
                print(f"  Top result: {top_result.source} (relevance: {top_result.relevance_score:.2f})")
            
            search_results[query_text] = {
                "results_count": result.total_results,
                "execution_time": result.execution_time_ms,
                "insights": result.search_insights
            }
        
        # Test related content discovery
        if self.analytics_engine.search_engine.document_metadata:
            content_id = list(self.analytics_engine.search_engine.document_metadata.keys())[0]
            print(f"\nDiscovering content related to document {content_id[:8]}...")
            
            related_items = await self.analytics_engine.search_engine.discover_related_content(content_id, 5)
            print(f"✓ Found {len(related_items)} related items")
        
        self.demo_results["semantic_search"] = {
            "index_statistics": index_stats,
            "search_results": search_results
        }
    
    async def demo_pattern_recognition(self):
        """Demonstrate pattern recognition capabilities."""
        print("\n📈 Pattern Recognition Demo")
        print("-" * 40)
        
        # Test different data sources and pattern types
        data_sources = ["sales_data", "customer_behavior", "financial_metrics", "operational_metrics"]
        pattern_types = [PatternType.TREND, PatternType.ANOMALY, PatternType.CORRELATION, PatternType.CYCLE]
        
        pattern_results = {}
        
        for data_source in data_sources:
            print(f"\nAnalyzing patterns in {data_source}...")
            
            request = PatternRecognitionRequest(
                data_source=data_source,
                pattern_types=pattern_types,
                sensitivity=0.7,
                min_pattern_strength=0.6
            )
            
            result = await self.analytics_engine.pattern_engine.recognize_patterns(request)
            
            print(f"✓ Found {len(result.patterns)} patterns")
            print(f"  Execution time: {result.execution_time_ms:.2f}ms")
            
            # Show pattern breakdown by type
            pattern_breakdown = {}
            for pattern in result.patterns:
                pattern_type = pattern.pattern_type.value
                if pattern_type not in pattern_breakdown:
                    pattern_breakdown[pattern_type] = 0
                pattern_breakdown[pattern_type] += 1
            
            for pattern_type, count in pattern_breakdown.items():
                print(f"  {pattern_type}: {count} patterns")
            
            # Show top insight
            if result.summary_insights:
                print(f"  Key insight: {result.summary_insights[0]}")
            
            pattern_results[data_source] = {
                "patterns_found": len(result.patterns),
                "pattern_breakdown": pattern_breakdown,
                "business_opportunities": len(result.business_opportunities),
                "execution_time": result.execution_time_ms
            }
        
        # Detect emerging opportunities
        print("\nDetecting emerging opportunities across all data sources...")
        opportunities = await self.analytics_engine.pattern_engine.detect_emerging_opportunities(
            data_sources, lookback_days=90
        )
        
        print(f"✓ Identified {len(opportunities)} emerging opportunities")
        if opportunities:
            top_opportunity = opportunities[0]
            print(f"  Top opportunity: {top_opportunity.title}")
            print(f"  Priority: {top_opportunity.priority}/10")
        
        self.demo_results["pattern_recognition"] = {
            "data_source_results": pattern_results,
            "emerging_opportunities": len(opportunities)
        }
    
    async def demo_predictive_analytics(self):
        """Demonstrate predictive analytics capabilities."""
        print("\n🔮 Predictive Analytics Demo")
        print("-" * 40)
        
        # Revenue forecasting
        print("Generating revenue forecasts...")
        data_sources = ["sales_data", "financial_data", "market_data"]
        
        revenue_prediction = await self.analytics_engine.predictive_engine.forecast_revenue(
            data_sources, forecast_horizon=90
        )
        
        print(f"✓ 90-day revenue forecast: ${revenue_prediction.base_prediction:,.2f}")
        print(f"  Confidence interval: ${revenue_prediction.confidence_interval['lower']:,.2f} - ${revenue_prediction.confidence_interval['upper']:,.2f}")
        print(f"  Model accuracy: {revenue_prediction.model_accuracy:.2%}")
        print(f"  Key drivers: {', '.join(revenue_prediction.key_drivers[:3])}")
        
        # Customer churn prediction
        print("\nPredicting customer churn...")
        churn_predictions = await self.analytics_engine.predictive_engine.predict_customer_churn("customer_data")
        
        if churn_predictions:
            high_risk_customers = [p for p in churn_predictions if p.get("churn_probability", 0) > 0.7]
            print(f"✓ Analyzed customer churn risk")
            print(f"  High-risk customers: {len(high_risk_customers)}")
            print(f"  Total predictions: {len(churn_predictions)}")
        
        # Growth opportunity identification
        print("\nIdentifying growth opportunities...")
        growth_opportunities = await self.analytics_engine.predictive_engine.identify_growth_opportunities(
            data_sources
        )
        
        print(f"✓ Identified {len(growth_opportunities)} growth opportunities")
        if growth_opportunities:
            top_growth_opp = growth_opportunities[0]
            print(f"  Top opportunity: {top_growth_opp.title}")
            print(f"  Confidence: {top_growth_opp.confidence:.2f}")
        
        # Comprehensive prediction request
        print("\nRunning comprehensive predictive analysis...")
        
        request = PredictiveAnalyticsRequest(
            prediction_type=PredictionType.REVENUE_FORECAST,
            data_sources=data_sources,
            target_variable="revenue",
            prediction_horizon=60,
            confidence_level=0.95,
            include_scenarios=True
        )
        
        result = await self.analytics_engine.predictive_engine.predict_business_outcomes(request)
        
        print(f"✓ Generated {len(result.predictions)} predictions")
        print(f"  Business insights: {len(result.business_insights)}")
        print(f"  Recommendations: {len(result.recommended_actions)}")
        print(f"  Execution time: {result.execution_time_ms:.2f}ms")
        
        self.demo_results["predictive_analytics"] = {
            "revenue_forecast": {
                "prediction": revenue_prediction.base_prediction,
                "accuracy": revenue_prediction.model_accuracy
            },
            "churn_analysis": {
                "total_customers": len(churn_predictions),
                "high_risk_count": len([p for p in churn_predictions if p.get("churn_probability", 0) > 0.7])
            },
            "growth_opportunities": len(growth_opportunities),
            "comprehensive_analysis": {
                "predictions": len(result.predictions),
                "insights": len(result.business_insights)
            }
        }
    
    async def demo_comprehensive_analysis(self):
        """Demonstrate comprehensive analysis across all engines."""
        print("\n🎯 Comprehensive Analysis Demo")
        print("-" * 40)
        
        print("Running comprehensive analysis across all analytics engines...")
        
        data_sources = ["sales_data", "customer_data", "financial_data", "operational_data", "market_data"]
        business_context = {
            "industry": "technology",
            "company_size": "enterprise",
            "focus_areas": ["revenue_growth", "operational_efficiency", "customer_retention"],
            "time_horizon": "quarterly"
        }
        
        result = await self.analytics_engine.execute_comprehensive_analysis(
            data_sources, business_context
        )
        
        execution_summary = result["execution_summary"]
        
        print(f"✓ Comprehensive analysis completed")
        print(f"  Engines executed: {execution_summary['engines_executed']}")
        print(f"  Total insights: {execution_summary['total_insights']}")
        print(f"  Business opportunities: {execution_summary['total_opportunities']}")
        print(f"  Execution time: {execution_summary['total_execution_time_ms']:.2f}ms")
        
        # Show insights from each engine
        if result.get("graph_analytics"):
            print(f"  Graph analytics: ✓ Completed")
        
        if result.get("pattern_recognition"):
            print(f"  Pattern recognition: ✓ Completed")
        
        if result.get("predictive_analytics"):
            print(f"  Predictive analytics: ✓ Completed")
        
        # Show top combined insights
        combined_insights = result["combined_insights"]
        if combined_insights:
            print(f"\nTop Combined Insights:")
            for i, insight in enumerate(combined_insights[:3], 1):
                print(f"  {i}. {insight.title} (confidence: {insight.confidence:.2f})")
        
        # Show business opportunities
        opportunities = result["business_opportunities"]
        if opportunities:
            print(f"\nTop Business Opportunities:")
            for i, opportunity in enumerate(opportunities[:3], 1):
                print(f"  {i}. {opportunity}")
        
        self.demo_results["comprehensive_analysis"] = execution_summary
    
    async def demo_business_intelligence(self):
        """Demonstrate business intelligence summary generation."""
        print("\n📋 Business Intelligence Summary Demo")
        print("-" * 40)
        
        print("Generating business intelligence summary...")
        
        summary = await self.analytics_engine.get_business_intelligence_summary(30)
        
        print("✓ Business Intelligence Summary Generated")
        
        # Executive Summary
        exec_summary = summary["executive_summary"]
        print(f"\nExecutive Summary:")
        print(f"  Key insights: {len(exec_summary['key_insights'])}")
        print(f"  Critical actions: {len(exec_summary['critical_actions'])}")
        print(f"  Risk indicators: {len(exec_summary['risk_indicators'])}")
        
        # Performance Metrics
        perf_metrics = summary["performance_metrics"]
        print(f"\nPerformance Metrics:")
        print(f"  Analytics executed: {perf_metrics['analytics_executed']}")
        print(f"  Insights generated: {perf_metrics['insights_generated']}")
        print(f"  Opportunities identified: {perf_metrics['opportunities_identified']}")
        print(f"  Business impact score: {perf_metrics['business_impact_score']}/10")
        
        # Trend Analysis
        trends = summary["trend_analysis"]
        print(f"\nTrend Analysis:")
        for metric, trend in trends.items():
            print(f"  {metric.replace('_', ' ').title()}: {trend}")
        
        # Top Recommendations
        recommendations = summary["recommendations"]
        print(f"\nTop Recommendations:")
        for i, rec in enumerate(recommendations[:3], 1):
            print(f"  {i}. {rec}")
        
        self.demo_results["business_intelligence"] = {
            "insights_count": len(exec_summary['key_insights']),
            "performance_score": perf_metrics['business_impact_score'],
            "recommendations_count": len(recommendations)
        }
    
    async def demo_performance_metrics(self):
        """Demonstrate performance metrics and monitoring."""
        print("\n⚡ Performance Metrics Demo")
        print("-" * 40)
        
        print("Generating performance summary...")
        
        performance_summary = await self.analytics_engine.get_performance_summary()
        
        print(f"✓ Performance Summary Generated")
        print(f"  Total operations: {performance_summary['total_operations']}")
        
        if performance_summary['total_operations'] > 0:
            print(f"  Average execution time: {performance_summary['avg_execution_time_ms']:.2f}ms")
            print(f"  Average throughput: {performance_summary['avg_throughput']:.2f} ops/sec")
        
        # Engine-specific performance
        by_engine = performance_summary.get("by_engine", {})
        if by_engine:
            print(f"\nPerformance by Engine:")
            for engine, metrics in by_engine.items():
                print(f"  {engine}:")
                print(f"    Operations: {metrics['operations']}")
                print(f"    Avg execution time: {metrics['avg_execution_time_ms']:.2f}ms")
                print(f"    Avg throughput: {metrics['avg_throughput']:.2f} ops/sec")
        
        self.demo_results["performance_metrics"] = performance_summary
    
    async def save_demo_results(self):
        """Save demo results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"advanced_analytics_demo_results_{timestamp}.json"
        
        # Add metadata
        demo_metadata = {
            "demo_timestamp": datetime.now().isoformat(),
            "demo_version": "1.0.0",
            "engines_tested": ["graph_analytics", "semantic_search", "pattern_recognition", "predictive_analytics"],
            "total_execution_time": sum(
                result.get("execution_time", 0) 
                for result in self.demo_results.values() 
                if isinstance(result, dict)
            )
        }
        
        final_results = {
            "metadata": demo_metadata,
            "results": self.demo_results
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(final_results, f, indent=2, default=str)
            
            print(f"\n💾 Demo results saved to: {filename}")
            
        except Exception as e:
            print(f"⚠️  Could not save demo results: {str(e)}")
    
    def print_demo_summary(self):
        """Print a summary of the demo results."""
        print("\n📊 Demo Summary")
        print("=" * 60)
        
        if "graph_analytics" in self.demo_results:
            graph_stats = self.demo_results["graph_analytics"]["graph_statistics"]
            print(f"Graph Analytics: {graph_stats['nodes_count']} nodes, {graph_stats['edges_count']} edges")
        
        if "semantic_search" in self.demo_results:
            search_stats = self.demo_results["semantic_search"]["index_statistics"]
            print(f"Semantic Search: {search_stats['documents_indexed']} documents indexed")
        
        if "pattern_recognition" in self.demo_results:
            pattern_stats = self.demo_results["pattern_recognition"]
            total_patterns = sum(
                result["patterns_found"] 
                for result in pattern_stats["data_source_results"].values()
            )
            print(f"Pattern Recognition: {total_patterns} patterns detected")
        
        if "predictive_analytics" in self.demo_results:
            pred_stats = self.demo_results["predictive_analytics"]
            print(f"Predictive Analytics: Revenue forecast ${pred_stats['revenue_forecast']['prediction']:,.2f}")
        
        if "comprehensive_analysis" in self.demo_results:
            comp_stats = self.demo_results["comprehensive_analysis"]
            print(f"Comprehensive Analysis: {comp_stats['total_insights']} total insights")
        
        print("=" * 60)


async def main():
    """Run the advanced analytics demo."""
    demo = AdvancedAnalyticsDemo()
    
    try:
        await demo.run_complete_demo()
        demo.print_demo_summary()
        
    except KeyboardInterrupt:
        print("\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {str(e)}")
        raise


if __name__ == "__main__":
    asyncio.run(main())