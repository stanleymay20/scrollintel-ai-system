#!/usr/bin/env python3
"""
Comprehensive BI Agent Demo
Demonstrates all BI Agent capabilities including dashboard creation, 
real-time updates, KPI tracking, and alerting system.
"""
import asyncio
import json
import time
from scrollintel_core.agents.bi_agent import BIAgent
from scrollintel_core.agents.base import AgentRequest


async def demo_bi_agent_comprehensive():
    """Comprehensive demonstration of BI Agent capabilities"""
    print("🚀 ScrollIntel BI Agent - Comprehensive Demo")
    print("=" * 60)
    
    # Initialize BI Agent
    bi_agent = BIAgent()
    print(f"✓ Initialized {bi_agent.name}")
    print(f"  Description: {bi_agent.description}")
    
    # Demo 1: Show all capabilities
    print("\n📋 BI Agent Capabilities:")
    capabilities = bi_agent.get_capabilities()
    for i, capability in enumerate(capabilities, 1):
        print(f"  {i}. {capability}")
    
    # Demo 2: Create Executive Dashboard
    print("\n📊 Creating Executive Dashboard...")
    dashboard_request = AgentRequest(
        query="Create an executive dashboard for my e-commerce business",
        context={
            "dashboard_type": "executive",
            "business_type": "ecommerce",
            "user_id": "demo_user_001",
            "data_source": "ecommerce_data"
        }
    )
    
    dashboard_response = await bi_agent.process(dashboard_request)
    if dashboard_response.success:
        dashboard_data = dashboard_response.result
        dashboard_id = dashboard_data.get('dashboard_id')
        print(f"✓ Executive Dashboard Created: {dashboard_id}")
        print(f"  - Widgets: {len(dashboard_data.get('dashboard_config', {}).get('widgets', []))}")
        print(f"  - Refresh Interval: {dashboard_data.get('dashboard_config', {}).get('refresh_interval')}s")
        print(f"  - WebSocket Endpoint: {dashboard_data.get('websocket_endpoint')}")
        
        # Show widget details
        widgets = dashboard_data.get('dashboard_config', {}).get('widgets', [])
        print("  - Widget Configuration:")
        for widget in widgets:
            print(f"    • {widget.get('type')}: {widget.get('metric', 'N/A')}")
    
    # Demo 3: Create Sales Dashboard
    print("\n📈 Creating Sales Dashboard...")
    sales_dashboard_request = AgentRequest(
        query="Create a sales dashboard with real-time updates",
        context={
            "dashboard_type": "sales",
            "user_id": "demo_user_001",
            "data_source": "sales_data"
        }
    )
    
    sales_response = await bi_agent.process(sales_dashboard_request)
    if sales_response.success:
        sales_data = sales_response.result
        sales_dashboard_id = sales_data.get('dashboard_id')
        print(f"✓ Sales Dashboard Created: {sales_dashboard_id}")
        print(f"  - Real-time Updates: {sales_data.get('real_time_enabled')}")
    
    # Demo 4: Calculate Business Metrics
    print("\n📊 Calculating Business Metrics...")
    metrics_request = AgentRequest(
        query="Calculate comprehensive business metrics from our data",
        context={
            "data_source": "business_data",
            "metric_types": ["revenue", "customer", "operational"]
        }
    )
    
    metrics_response = await bi_agent.process(metrics_request)
    if metrics_response.success:
        metrics_data = metrics_response.result
        calculated_metrics = metrics_data.get("calculated_metrics", {})
        print(f"✓ Metrics Calculated from {metrics_data.get('data_points_analyzed')} data points")
        print("  Key Metrics:")
        
        # Format and display key metrics
        key_metrics = {
            "total_revenue": "Total Revenue",
            "revenue_growth_rate": "Revenue Growth Rate",
            "total_customers": "Total Customers",
            "churn_rate": "Customer Churn Rate",
            "avg_quality_score": "Average Quality Score"
        }
        
        for metric_key, metric_name in key_metrics.items():
            if metric_key in calculated_metrics:
                value = calculated_metrics[metric_key]
                if "rate" in metric_key or "churn" in metric_key:
                    print(f"    • {metric_name}: {value:.2f}%")
                elif "revenue" in metric_key and "rate" not in metric_key:
                    print(f"    • {metric_name}: ${value:,.2f}")
                elif "score" in metric_key:
                    print(f"    • {metric_name}: {value:.3f}")
                else:
                    print(f"    • {metric_name}: {value}")
    
    # Demo 5: Setup Alert System
    print("\n🚨 Setting up Alert System...")
    alert_request = AgentRequest(
        query="Setup comprehensive alert system for business metrics",
        context={
            "alert_config": {
                "email_recipients": ["ceo@company.com", "cto@company.com"],
                "webhook_url": "https://api.company.com/alerts",
                "rules": [
                    {
                        "id": "revenue_critical",
                        "metric": "revenue_total",
                        "condition": "less_than",
                        "threshold": 75000,
                        "severity": "critical",
                        "message": "CRITICAL: Revenue dropped below $75,000"
                    },
                    {
                        "id": "churn_warning",
                        "metric": "churn_rate",
                        "condition": "greater_than",
                        "threshold": 8.0,
                        "severity": "high",
                        "message": "HIGH: Customer churn rate exceeds 8%"
                    },
                    {
                        "id": "conversion_low",
                        "metric": "conversion_rate",
                        "condition": "less_than",
                        "threshold": 3.0,
                        "severity": "medium",
                        "message": "MEDIUM: Conversion rate below 3%"
                    }
                ]
            }
        }
    )
    
    alert_response = await bi_agent.process(alert_request)
    if alert_response.success:
        alert_data = alert_response.result
        print(f"✓ Alert System Configured")
        print(f"  - Status: {alert_data.get('alert_system_status', 'Active')}")
        print(f"  - Total Rules: {alert_data.get('total_rules', 0)}")
        
        # Show alert rules
        rules = alert_data.get('alert_rules', [])
        if rules:
            print("  - Alert Rules:")
            for rule in rules[:3]:  # Show first 3 rules
                print(f"    • {rule.get('id')}: {rule.get('message')}")
    
    # Demo 6: Test Alert System
    print("\n🔍 Testing Alert System...")
    # Simulate metrics that would trigger alerts
    test_metrics = {
        "revenue_total": 65000,    # Below 75000 threshold - should trigger critical alert
        "churn_rate": 9.5,         # Above 8.0 threshold - should trigger high alert
        "conversion_rate": 4.2     # Above 3.0 threshold - should not trigger
    }
    
    alert_check_request = AgentRequest(
        query="Check current metrics against all alert rules",
        context={"metrics": test_metrics}
    )
    
    alert_check_response = await bi_agent.process(alert_check_request)
    if alert_check_response.success:
        alert_check_data = alert_check_response.result
        triggered_alerts = alert_check_data.get('triggered_alerts', [])
        
        print(f"✓ Alert Check Completed")
        print(f"  - Metrics Evaluated: {len(test_metrics)}")
        print(f"  - Alerts Triggered: {len(triggered_alerts)}")
        
        if triggered_alerts:
            print("  - Triggered Alerts:")
            for alert in triggered_alerts:
                severity_emoji = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}
                emoji = severity_emoji.get(alert.get('severity', 'medium'), "⚪")
                print(f"    {emoji} {alert.get('message')} (Value: {alert.get('value')})")
    
    # Demo 7: Setup Real-time Updates
    print("\n⚡ Setting up Real-time Updates...")
    if dashboard_response.success:
        realtime_request = AgentRequest(
            query="Setup real-time updates for executive dashboard",
            context={
                "dashboard_id": dashboard_id,
                "update_interval": 15  # 15 seconds
            }
        )
        
        realtime_response = await bi_agent.process(realtime_request)
        if realtime_response.success:
            realtime_data = realtime_response.result
            print(f"✓ Real-time Updates Configured")
            print(f"  - Dashboard: {realtime_data.get('dashboard_id', dashboard_id)}")
            print(f"  - Update Interval: {realtime_data.get('update_interval', 15)} seconds")
            print(f"  - WebSocket Endpoint: {realtime_data.get('websocket_endpoint', 'N/A')}")
            
            features = realtime_data.get('features_enabled', [])
            if features:
                print("  - Real-time Features:")
                for feature in features:
                    print(f"    • {feature}")
    
    # Demo 8: Generate Executive Report
    print("\n📄 Generating Executive Report...")
    report_request = AgentRequest(
        query="Generate comprehensive executive summary report",
        context={
            "report_type": "executive_summary",
            "date_range": "last_30_days",
            "data_source": "business_data"
        }
    )
    
    report_response = await bi_agent.process(report_request)
    if report_response.success:
        report_data = report_response.result
        print(f"✓ Executive Report Generated")
        print(f"  - Report ID: {report_data.get('report_id')}")
        print(f"  - Generated: {report_data.get('generated_at')}")
        
        insights_summary = report_data.get('insights_summary', {})
        print(f"  - Insights: {insights_summary.get('total_insights', 0)}")
        print(f"  - Recommendations: {insights_summary.get('recommendations', 0)}")
        print(f"  - Confidence Score: {insights_summary.get('confidence_score', 0):.2f}")
        
        # Show export options
        export_options = report_data.get('export_options', {})
        if export_options:
            print("  - Export Options:")
            for format_type, url in export_options.items():
                print(f"    • {format_type.upper()}: {url}")
    
    # Demo 9: Dashboard Management
    print("\n🎛️  Dashboard Management...")
    dashboards = bi_agent.get_dashboard_list()
    print(f"✓ Total Dashboards: {len(dashboards)}")
    
    for dashboard in dashboards:
        print(f"  - {dashboard['name']} (ID: {dashboard['id']})")
        print(f"    • Widgets: {dashboard['widget_count']}")
        print(f"    • Refresh: {dashboard['refresh_interval']}s")
    
    # Demo 10: KPI Management
    print("\n📊 KPI Management...")
    kpis = bi_agent.get_kpi_definitions()
    print(f"✓ Available KPIs: {len(kpis)}")
    
    print("  - Standard KPIs:")
    for kpi in kpis[:4]:  # Show first 4 KPIs
        print(f"    • {kpi['name']} ({kpi['unit']})")
        print(f"      Target: {kpi.get('target_value', 'Not set')}")
    
    # Add custom KPI
    custom_kpi_id = bi_agent.add_custom_kpi({
        "name": "Net Promoter Score",
        "calculation": "average",
        "target_value": 50.0,
        "threshold_warning": 30.0,
        "threshold_critical": 10.0,
        "unit": "points",
        "format_type": "number",
        "trend_direction": "higher_better"
    })
    print(f"✓ Custom KPI Added: {custom_kpi_id}")
    
    # Demo 11: Health Check
    print("\n🏥 System Health Check...")
    health_result = await bi_agent.health_check()
    health_status = health_result.get("healthy", False)
    print(f"✓ BI Agent Health: {'Healthy' if health_status else 'Unhealthy'}")
    
    if health_status:
        components = health_result.get("components", {})
        print("  - Component Status:")
        for component, status in components.items():
            status_icon = "✓" if status else "✗"
            print(f"    {status_icon} {component.replace('_', ' ').title()}")
        
        stats = health_result.get("statistics", {})
        print("  - System Statistics:")
        for stat, value in stats.items():
            print(f"    • {stat.replace('_', ' ').title()}: {value}")
    
    # Demo Summary
    print("\n" + "=" * 60)
    print("🎉 BI Agent Demo Completed Successfully!")
    print("\n✅ Features Demonstrated:")
    
    features_demo = [
        "✓ Automatic dashboard generation (Executive & Sales)",
        "✓ Real-time dashboard updates via WebSocket",
        "✓ Comprehensive business metric calculation",
        "✓ KPI monitoring with custom thresholds",
        "✓ Advanced alert system with multiple severity levels",
        "✓ Executive report generation with insights",
        "✓ Dashboard and KPI management",
        "✓ System health monitoring",
        "✓ Custom KPI creation and management",
        "✓ Multi-channel alert notifications"
    ]
    
    for feature in features_demo:
        print(f"  {feature}")
    
    print(f"\n📊 Final Statistics:")
    print(f"  • Active Dashboards: {len(bi_agent.dashboards)}")
    print(f"  • KPI Definitions: {len(bi_agent.kpi_metrics)}")
    print(f"  • Alert Rules: {len(bi_agent.alert_system.alert_rules)}")
    print(f"  • Processing Time: {time.time() - start_time:.2f} seconds")
    
    print("\n🚀 BI Agent is ready for production use!")


if __name__ == "__main__":
    start_time = time.time()
    asyncio.run(demo_bi_agent_comprehensive())