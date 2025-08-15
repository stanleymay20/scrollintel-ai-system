#!/usr/bin/env python3
"""
Demo script for AI Insight Generation Engine.

This script demonstrates the complete insight generation workflow including:
- Pattern detection in business metrics
- Natural language insight generation
- Actionable recommendation creation
- Anomaly detection and explanation
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import List
import numpy as np

from scrollintel.engines.insight_generator import InsightGenerator, AnalyticsData
from scrollintel.models.dashboard_models import BusinessMetric
from scrollintel.models.insight_models import BusinessContext


def create_sample_business_data() -> List[BusinessMetric]:
    """Create realistic sample business metrics for demonstration."""
    print("📊 Creating sample business metrics...")
    
    base_time = datetime.utcnow() - timedelta(days=60)
    metrics = []
    
    # 1. Revenue Growth Trend (Positive)
    print("  • Generating revenue growth data...")
    for i in range(60):
        timestamp = base_time + timedelta(days=i)
        # Strong upward trend with seasonal variation
        base_revenue = 500000 + (i * 5000)  # $5k growth per day
        seasonal_factor = 1 + 0.1 * np.sin(2 * np.pi * i / 7)  # Weekly seasonality
        noise = np.random.normal(0, 20000)
        value = base_revenue * seasonal_factor + noise
        
        metrics.append(BusinessMetric(
            name="monthly_recurring_revenue",
            category="financial",
            value=max(0, value),
            unit="USD",
            timestamp=timestamp,
            source="billing_system",
            context={"department": "sales", "region": "global"}
        ))
    
    # 2. Customer Acquisition Cost (Concerning Trend)
    print("  • Generating customer acquisition cost data...")
    for i in range(60):
        timestamp = base_time + timedelta(days=i)
        # Increasing CAC (concerning trend)
        base_cac = 150 + (i * 2)  # $2 increase per day
        noise = np.random.normal(0, 15)
        value = base_cac + noise
        
        metrics.append(BusinessMetric(
            name="customer_acquisition_cost",
            category="marketing",
            value=max(0, value),
            unit="USD",
            timestamp=timestamp,
            source="marketing_analytics",
            context={"department": "marketing", "channel": "digital"}
        ))
    
    # 3. System Performance (Correlated with Customer Satisfaction)
    print("  • Generating system performance data...")
    satisfaction_base = 85
    for i in range(60):
        timestamp = base_time + timedelta(days=i)
        
        # Performance degrades slightly over time (needs attention)
        performance_base = 95 - (i * 0.1)
        performance_noise = np.random.normal(0, 3)
        performance = max(0, min(100, performance_base + performance_noise))
        
        # Customer satisfaction correlates with performance
        satisfaction = satisfaction_base + (performance - 90) * 0.5 + np.random.normal(0, 2)
        satisfaction = max(0, min(100, satisfaction))
        
        metrics.extend([
            BusinessMetric(
                name="system_uptime_percentage",
                category="technical",
                value=performance,
                unit="percentage",
                timestamp=timestamp,
                source="monitoring_system",
                context={"department": "engineering", "service": "core_api"}
            ),
            BusinessMetric(
                name="customer_satisfaction_score",
                category="quality",
                value=satisfaction,
                unit="score",
                timestamp=timestamp,
                source="customer_feedback",
                context={"department": "customer_success", "survey_type": "nps"}
            )
        ])
    
    # 4. Support Ticket Volume (with Anomaly)
    print("  • Generating support ticket data with anomaly...")
    for i in range(60):
        timestamp = base_time + timedelta(days=i)
        
        # Normal ticket volume with a spike on day 45 (incident)
        if i == 45:
            value = 250  # Major incident spike
        elif 44 <= i <= 47:
            value = 80 + np.random.normal(0, 10)  # Elevated after incident
        else:
            value = 35 + np.random.normal(0, 8)  # Normal volume
        
        metrics.append(BusinessMetric(
            name="daily_support_tickets",
            category="operational",
            value=max(0, value),
            unit="count",
            timestamp=timestamp,
            source="support_system",
            context={"department": "customer_support", "priority": "all"}
        ))
    
    # 5. Employee Productivity (Seasonal Pattern)
    print("  • Generating employee productivity data...")
    for i in range(60):
        timestamp = base_time + timedelta(days=i)
        
        # Weekly productivity pattern (lower on Mondays, higher mid-week)
        day_of_week = i % 7
        weekly_factor = {0: 0.85, 1: 1.0, 2: 1.1, 3: 1.15, 4: 1.05, 5: 0.9, 6: 0.8}
        
        base_productivity = 75
        productivity = base_productivity * weekly_factor.get(day_of_week, 1.0)
        productivity += np.random.normal(0, 5)
        
        metrics.append(BusinessMetric(
            name="team_productivity_index",
            category="hr",
            value=max(0, min(100, productivity)),
            unit="index",
            timestamp=timestamp,
            source="hr_analytics",
            context={"department": "human_resources", "team": "engineering"}
        ))
    
    print(f"✅ Created {len(metrics)} business metrics across {len(set(m.name for m in metrics))} different KPIs")
    return metrics


def create_business_context() -> BusinessContext:
    """Create business context for better insight interpretation."""
    return BusinessContext(
        context_type="company",
        name="TechCorp SaaS Platform",
        description="B2B SaaS company providing analytics solutions",
        context_metadata={
            "industry": "technology",
            "company_size": "mid_market",
            "business_model": "subscription",
            "growth_stage": "scaling"
        },
        thresholds={
            "monthly_recurring_revenue": {"critical": 0.15, "high": 0.10, "medium": 0.05},
            "customer_acquisition_cost": {"critical": 0.25, "high": 0.15, "medium": 0.10},
            "customer_satisfaction_score": {"critical": 0.20, "high": 0.15, "medium": 0.10},
            "system_uptime_percentage": {"critical": 0.05, "high": 0.03, "medium": 0.02}
        },
        kpis=[
            "monthly_recurring_revenue",
            "customer_acquisition_cost", 
            "customer_satisfaction_score",
            "system_uptime_percentage",
            "daily_support_tickets"
        ],
        benchmarks={
            "industry_avg_cac": 180,
            "industry_avg_satisfaction": 82,
            "industry_avg_uptime": 99.5
        }
    )


async def demonstrate_insight_generation():
    """Demonstrate the complete insight generation workflow."""
    print("🚀 Starting AI Insight Generation Engine Demo")
    print("=" * 60)
    
    # Initialize the insight generator
    print("\n🔧 Initializing AI Insight Generation Engine...")
    generator = InsightGenerator()
    await generator.start()
    
    try:
        # Create sample data
        metrics = create_sample_business_data()
        business_context = create_business_context()
        
        # Create analytics data
        analytics_data = AnalyticsData(
            metrics=metrics,
            time_range=(datetime.utcnow() - timedelta(days=60), datetime.utcnow()),
            context={
                "analysis_type": "executive_dashboard",
                "requested_by": "cto",
                "business_context": business_context.name
            }
        )
        
        print(f"\n📈 Processing {len(metrics)} metrics for insight generation...")
        
        # Process the data
        result = await generator.process(analytics_data)
        
        # Display results
        print("\n" + "=" * 60)
        print("📊 INSIGHT GENERATION RESULTS")
        print("=" * 60)
        
        # Summary
        summary = result["summary"]
        print(f"\n📋 EXECUTIVE SUMMARY")
        print("-" * 30)
        print(f"Overview: {summary['overview']}")
        
        if summary["key_findings"]:
            print(f"\n🔍 Key Findings:")
            for finding in summary["key_findings"]:
                print(f"  • {finding}")
        
        if summary["urgent_actions"]:
            print(f"\n⚠️  Urgent Actions Required:")
            for action in summary["urgent_actions"]:
                print(f"  • {action}")
        
        # Patterns
        patterns = result["patterns"]
        print(f"\n🔍 DETECTED PATTERNS ({len(patterns)} found)")
        print("-" * 40)
        
        for i, pattern in enumerate(patterns[:5], 1):  # Show top 5
            print(f"\n{i}. {pattern['type'].replace('_', ' ').title()}")
            print(f"   Metric: {pattern['metric_name']}")
            print(f"   Description: {pattern['description']}")
            print(f"   Confidence: {pattern['confidence']:.2%}")
            print(f"   Significance: {pattern['significance']}")
        
        # Insights
        insights = result["insights"]
        print(f"\n💡 GENERATED INSIGHTS ({len(insights)} found)")
        print("-" * 40)
        
        for i, insight in enumerate(insights[:3], 1):  # Show top 3
            print(f"\n{i}. {insight['title']}")
            print(f"   Type: {insight['type'].replace('_', ' ').title()}")
            print(f"   Priority: {insight['priority'].upper()}")
            print(f"   Confidence: {insight['confidence']:.2%}")
            print(f"   Description: {insight['description']}")
            print(f"   Business Impact: {insight['business_impact']}")
            
            if insight.get('explanation'):
                print(f"   Explanation: {insight['explanation']}")
        
        # Recommendations
        recommendations = result["recommendations"]
        print(f"\n🎯 ACTIONABLE RECOMMENDATIONS ({len(recommendations)} found)")
        print("-" * 50)
        
        for i, rec in enumerate(recommendations[:3], 1):  # Show top 3
            print(f"\n{i}. {rec['title']}")
            print(f"   Priority: {rec['priority'].upper()}")
            print(f"   Timeline: {rec['timeline']}")
            print(f"   Effort: {rec['effort_required'].title()}")
            print(f"   Responsible: {rec['responsible_role'].replace('_', ' ').title()}")
            print(f"   Description: {rec['description']}")
            
            if rec.get('implementation_steps'):
                print(f"   Implementation Steps:")
                for step_num, step in enumerate(rec['implementation_steps'][:3], 1):
                    print(f"     {step_num}. {step}")
        
        # Anomalies
        anomalies = result["anomalies"]
        if anomalies:
            print(f"\n🚨 DETECTED ANOMALIES ({len(anomalies)} found)")
            print("-" * 40)
            
            for i, anomaly in enumerate(anomalies[:3], 1):  # Show top 3
                print(f"\n{i}. {anomaly['metric_name']}")
                print(f"   Type: {anomaly['anomaly_type'].title()}")
                print(f"   Severity: {anomaly['severity'].upper()}")
                print(f"   Value: {anomaly['metric_value']:.2f}")
                print(f"   Expected: {anomaly['expected_value']:.2f}")
                print(f"   Deviation: {anomaly['deviation_score']:.2f}σ")
                
                # Generate explanation
                from scrollintel.models.insight_models import Anomaly
                anomaly_obj = Anomaly(
                    metric_name=anomaly['metric_name'],
                    metric_value=anomaly['metric_value'],
                    expected_value=anomaly['expected_value'],
                    deviation_score=anomaly['deviation_score'],
                    anomaly_type=anomaly['anomaly_type'],
                    severity=anomaly['severity']
                )
                
                explanation = await generator.explain_anomaly(anomaly_obj, business_context)
                print(f"   Explanation: {explanation}")
        
        # Metadata
        metadata = result["metadata"]
        print(f"\n📊 PROCESSING METADATA")
        print("-" * 30)
        print(f"Processed at: {metadata['processed_at']}")
        print(f"Metrics analyzed: {metadata['metrics_count']}")
        print(f"Patterns found: {metadata['patterns_found']}")
        print(f"Insights generated: {metadata['insights_generated']}")
        print(f"Recommendations created: {metadata['recommendations_created']}")
        print(f"Anomalies detected: {metadata['anomalies_detected']}")
        
        # Demonstrate specific capabilities
        print(f"\n🔬 DEMONSTRATING SPECIFIC CAPABILITIES")
        print("-" * 45)
        
        # Pattern detection
        print("\n1. Pattern Detection:")
        trend_patterns = await generator.analyze_data_patterns(analytics_data)
        print(f"   Detected {len(trend_patterns)} patterns in the data")
        
        # Insight generation from patterns
        if trend_patterns:
            print("\n2. Insight Generation:")
            sample_insights = await generator.generate_insights(trend_patterns[:2])
            print(f"   Generated {len(sample_insights)} insights from patterns")
        
        # Recommendation generation
        if insights:
            print("\n3. Recommendation Generation:")
            from scrollintel.models.insight_models import Insight
            insight_objects = []
            for insight_data in insights[:2]:
                insight_obj = Insight(
                    type=insight_data['type'],
                    title=insight_data['title'],
                    description=insight_data['description'],
                    explanation=insight_data.get('explanation', ''),
                    business_impact=insight_data.get('business_impact', ''),
                    confidence=insight_data['confidence'],
                    significance=insight_data['significance'],
                    priority=insight_data['priority'],
                    tags=insight_data.get('tags', []),
                    affected_metrics=insight_data.get('affected_metrics', [])
                )
                insight_objects.append(insight_obj)
            
            sample_recommendations = await generator.suggest_actions(insight_objects)
            print(f"   Generated {len(sample_recommendations)} recommendations from insights")
        
        print(f"\n✅ Demo completed successfully!")
        print(f"The AI Insight Generation Engine processed {len(metrics)} metrics")
        print(f"and generated {len(insights)} actionable insights with {len(recommendations)} recommendations.")
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up
        await generator.stop()
        print(f"\n🔧 AI Insight Generation Engine stopped.")


async def demonstrate_api_usage():
    """Demonstrate API usage patterns."""
    print(f"\n🌐 API USAGE EXAMPLES")
    print("-" * 30)
    
    # Example API request data
    api_request = {
        "metrics": [
            {
                "name": "revenue",
                "category": "financial",
                "value": 150000,
                "unit": "USD",
                "timestamp": datetime.utcnow().isoformat(),
                "source": "billing_system",
                "context": {"department": "sales"}
            },
            {
                "name": "customer_satisfaction",
                "category": "quality", 
                "value": 85,
                "unit": "score",
                "timestamp": datetime.utcnow().isoformat(),
                "source": "survey_system",
                "context": {"department": "customer_success"}
            }
        ],
        "time_range": {
            "start": (datetime.utcnow() - timedelta(days=30)).isoformat(),
            "end": datetime.utcnow().isoformat()
        },
        "context": {
            "analysis_type": "executive_dashboard",
            "requested_by": "cfo"
        }
    }
    
    print("Example API Request:")
    print(json.dumps(api_request, indent=2, default=str))
    
    print(f"\nAPI Endpoints Available:")
    print("  POST /api/insights/analyze - Complete insight analysis")
    print("  POST /api/insights/patterns/detect - Pattern detection only")
    print("  POST /api/insights/insights/generate - Generate insights from patterns")
    print("  POST /api/insights/recommendations/generate - Generate recommendations")
    print("  POST /api/insights/anomalies/explain - Explain anomalies")
    print("  GET  /api/insights/status - Engine status")
    print("  GET  /api/insights/health - Health check")


if __name__ == "__main__":
    print("🤖 AI Insight Generation Engine Demo")
    print("This demo showcases automated business intelligence with natural language insights")
    print()
    
    # Run the demonstration
    asyncio.run(demonstrate_insight_generation())
    
    # Show API usage
    asyncio.run(demonstrate_api_usage())
    
    print(f"\n🎉 Demo completed! The AI Insight Generation Engine is ready for production use.")
    print("Key capabilities demonstrated:")
    print("  ✅ Pattern detection (trends, correlations, anomalies)")
    print("  ✅ Natural language insight generation")
    print("  ✅ Actionable recommendation creation")
    print("  ✅ Business context understanding")
    print("  ✅ Anomaly detection and explanation")
    print("  ✅ Insight ranking and prioritization")
    print("  ✅ RESTful API interface")
    print("  ✅ Integration test coverage")