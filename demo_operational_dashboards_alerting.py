"""Demo script for operational dashboards and alerting system."""

import asyncio
import time
import json
from datetime import datetime, timedelta
from pathlib import Path

# Import the operational systems
from ai_data_readiness.core.operational_dashboard import get_dashboard_generator
from ai_data_readiness.core.alerting_system import (
    get_alerting_system, AlertRule, AlertContact, 
    AlertChannel, AlertSeverity, EscalationLevel
)
from ai_data_readiness.core.capacity_planner import get_capacity_planner
from ai_data_readiness.core.platform_monitor import get_platform_monitor


def print_section(title):
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")


def print_subsection(title):
    """Print a formatted subsection header."""
    print(f"\n{'-'*40}")
    print(f" {title}")
    print(f"{'-'*40}")


def demo_dashboard_generator():
    """Demonstrate dashboard generation capabilities."""
    print_section("OPERATIONAL DASHBOARD GENERATOR DEMO")
    
    generator = get_dashboard_generator()
    
    # Show available templates
    print_subsection("Available Dashboard Templates")
    templates = generator.dashboard_templates
    for name, template in templates.items():
        print(f"• {template.name}")
        print(f"  Description: {template.description}")
        print(f"  Widgets: {len(template.widgets)}")
        print(f"  Refresh: {template.refresh_interval}s")
        print()
    
    # Create dashboards from templates
    print_subsection("Creating Dashboards")
    
    dashboard_types = ['system_overview', 'platform_health', 'performance_monitoring']
    
    for dashboard_type in dashboard_types:
        print(f"\nCreating {dashboard_type} dashboard...")
        dashboard = generator.create_dashboard(dashboard_type)
        
        print(f"✓ Dashboard created: {dashboard.name}")
        print(f"  ID: {dashboard.id}")
        print(f"  Metrics: {len(dashboard.metrics)} tracked")
        print(f"  Widgets: {len(dashboard.layout.get('widgets', []))}")
        
        # Get dashboard data
        try:
            data = generator.get_dashboard_data(dashboard, time_range_hours=1)
            print(f"  Data sources: {list(data.keys())}")
        except Exception as e:
            print(f"  Data collection: {e}")
    
    # Generate HTML dashboard
    print_subsection("Generating HTML Dashboard")
    
    try:
        dashboard = generator.create_dashboard('executive_summary')
        html_content = generator.generate_dashboard_html(dashboard, 24)
        
        # Save to file
        output_dir = Path("generated_dashboards")
        output_dir.mkdir(exist_ok=True)
        
        html_file = output_dir / "executive_dashboard.html"
        with open(html_file, 'w') as f:
            f.write(html_content)
        
        print(f"✓ Executive dashboard HTML generated")
        print(f"  File: {html_file}")
        print(f"  Size: {len(html_content):,} characters")
        
    except Exception as e:
        print(f"✗ HTML generation failed: {e}")
    
    # Export dashboard configuration
    print_subsection("Exporting Dashboard Configuration")
    
    try:
        dashboard = generator.create_dashboard('system_overview')
        config_file = output_dir / "system_overview_config.json"
        
        generator.export_dashboard_config(dashboard, str(config_file))
        
        print(f"✓ Dashboard configuration exported")
        print(f"  File: {config_file}")
        
        # Show config preview
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        print(f"  Configuration preview:")
        print(f"    Name: {config['name']}")
        print(f"    Metrics: {len(config['metrics'])}")
        print(f"    Refresh: {config['refresh_interval_seconds']}s")
        
    except Exception as e:
        print(f"✗ Configuration export failed: {e}")


def demo_alerting_system():
    """Demonstrate alerting system capabilities."""
    print_section("ALERTING SYSTEM DEMO")
    
    alerting = get_alerting_system()
    
    # Show default configuration
    print_subsection("Default Alert Configuration")
    
    rules = alerting.get_alert_rules()
    contacts = alerting.get_contacts()
    
    print(f"Default alert rules: {len(rules)}")
    for rule in rules[:3]:  # Show first 3 rules
        print(f"• {rule['name']}")
        print(f"  Metric: {rule['metric_name']} {rule['condition']} {rule['threshold']}")
        print(f"  Severity: {rule['severity']}")
        print(f"  Channels: {', '.join(rule['channels'])}")
        print()
    
    print(f"Default contacts: {len(contacts)}")
    for contact in contacts[:2]:  # Show first 2 contacts
        print(f"• {contact['name']} ({contact['escalation_level']})")
        print(f"  Email: {contact['email']}")
        print(f"  Channels: {', '.join(contact['channels'])}")
        print()
    
    # Create custom alert rule
    print_subsection("Creating Custom Alert Rule")
    
    custom_rule = AlertRule(
        id='demo_custom_rule',
        name='Demo High Processing Time',
        description='Alert when processing time exceeds 5 minutes',
        metric_name='avg_processing_time_seconds',
        condition='>',
        threshold=300.0,
        severity=AlertSeverity.WARNING,
        duration_minutes=2,
        cooldown_minutes=15,
        channels=[AlertChannel.EMAIL, AlertChannel.DASHBOARD],
        escalation_rules={
            EscalationLevel.L2: 10,
            EscalationLevel.L3: 30
        }
    )
    
    alerting.add_alert_rule(custom_rule)
    print(f"✓ Custom alert rule created: {custom_rule.name}")
    print(f"  Threshold: {custom_rule.threshold} seconds")
    print(f"  Escalation: L2 after {custom_rule.escalation_rules.get(EscalationLevel.L2, 0)} min")
    
    # Create custom contact
    print_subsection("Creating Custom Contact")
    
    custom_contact = AlertContact(
        id='demo_contact',
        name='Demo Operations Team',
        email='demo-ops@company.com',
        escalation_level=EscalationLevel.L1,
        channels=[AlertChannel.EMAIL, AlertChannel.SLACK]
    )
    
    alerting.add_contact(custom_contact)
    print(f"✓ Custom contact created: {custom_contact.name}")
    print(f"  Level: {custom_contact.escalation_level.value}")
    print(f"  Channels: {[c.value for c in custom_contact.channels]}")
    
    # Show alerting statistics
    print_subsection("Alerting System Statistics")
    
    stats = alerting.get_alerting_statistics()
    print(f"System status: {'Active' if stats['alerting_active'] else 'Inactive'}")
    print(f"Total incidents: {stats['total_incidents']}")
    print(f"Active incidents: {stats['active_incidents']}")
    print(f"Alert rules: {stats['alert_rules_count']}")
    print(f"Contacts: {stats['contacts_count']}")
    print(f"MTTR: {stats['mttr_minutes']:.1f} minutes")
    print(f"Alert rate: {stats['alerts_per_hour']:.1f} per hour")
    
    # Start alerting system
    print_subsection("Starting Alerting System")
    
    try:
        alerting.start_alerting(check_interval_seconds=10)
        print("✓ Alerting system started")
        print("  Check interval: 10 seconds")
        print("  Monitoring for alert conditions...")
        
        # Let it run for a short time
        time.sleep(5)
        
        # Check for any incidents
        active_incidents = alerting.get_active_incidents()
        print(f"  Active incidents after 5s: {len(active_incidents)}")
        
        # Stop the system
        alerting.stop_alerting()
        print("✓ Alerting system stopped")
        
    except Exception as e:
        print(f"✗ Alerting system error: {e}")


def demo_capacity_planner():
    """Demonstrate capacity planning capabilities."""
    print_section("CAPACITY PLANNING DEMO")
    
    planner = get_capacity_planner()
    
    # Show current capacity status
    print_subsection("Current Capacity Status")
    
    try:
        status = planner.get_current_capacity_status()
        
        if 'overall' in status:
            overall_status = status['overall']['status']
            print(f"Overall capacity status: {overall_status.upper()}")
            print()
            
            # Show component status
            for component, info in status.items():
                if component == 'overall':
                    continue
                
                if isinstance(info, dict):
                    utilization = info.get('utilization', 0)
                    comp_status = info.get('status', 'unknown')
                    headroom = info.get('headroom_percent', 0)
                    
                    print(f"{component.upper()}:")
                    print(f"  Utilization: {utilization:.1%}")
                    print(f"  Status: {comp_status}")
                    print(f"  Headroom: {headroom:.1f}%")
                    print()
        else:
            print("Capacity status not available - insufficient metrics data")
            
    except Exception as e:
        print(f"✗ Error getting capacity status: {e}")
    
    # Generate capacity planning report
    print_subsection("Generating Capacity Planning Report")
    
    try:
        print("Generating 30-day capacity forecast...")
        report = planner.generate_capacity_plan(time_horizon_days=30)
        
        print(f"✓ Capacity planning report generated")
        print(f"  Time horizon: {report.time_horizon_days} days")
        print(f"  Forecasts: {len(report.forecasts)} components")
        print(f"  Recommendations: {len(report.recommendations)} actions")
        print()
        
        # Show executive summary
        print("Executive Summary:")
        print(f"  {report.executive_summary}")
        print()
        
        # Show forecasts
        if report.forecasts:
            print("Resource Forecasts:")
            for forecast in report.forecasts[:3]:  # Show first 3
                print(f"• {forecast.component.value.title()}:")
                print(f"  Current: {forecast.current_utilization:.1%}")
                print(f"  Trend: {forecast.trend_direction}")
                print(f"  Method: {forecast.method_used.value}")
                print(f"  Accuracy: {forecast.accuracy_score:.2f}")
                print()
        
        # Show recommendations
        if report.recommendations:
            print("Capacity Recommendations:")
            for rec in report.recommendations:
                print(f"• {rec.component.value.title()} ({rec.priority.upper()} priority):")
                print(f"  Current: {rec.current_capacity:.0f} units")
                print(f"  Recommended: {rec.recommended_capacity:.0f} units")
                print(f"  Increase: {rec.capacity_increase_percent:.1f}%")
                print(f"  Cost: {rec.estimated_cost}")
                print(f"  Timeline: {rec.implementation_timeline}")
                print(f"  Justification: {rec.justification}")
                print()
        
        # Show risk assessment
        if report.risk_assessment:
            risk = report.risk_assessment
            print(f"Risk Assessment:")
            print(f"  Overall risk: {risk.get('overall_risk_level', 'unknown').upper()}")
            
            risk_factors = risk.get('risk_factors', [])
            if risk_factors:
                print(f"  Risk factors:")
                for factor in risk_factors[:3]:
                    print(f"    - {factor}")
            
            mitigation = risk.get('mitigation_strategies', [])
            if mitigation:
                print(f"  Mitigation strategies:")
                for strategy in mitigation[:2]:
                    print(f"    - {strategy}")
            print()
        
        # Show cost analysis
        if report.cost_analysis:
            cost = report.cost_analysis
            print(f"Cost Analysis:")
            print(f"  Monthly cost: ${cost.get('total_monthly_cost', 0):,.0f}")
            print(f"  Annual cost: ${cost.get('total_annual_cost', 0):,.0f}")
            print(f"  ROI break-even: {cost.get('roi_break_even_months', 0)} months")
            print(f"  Cost-benefit ratio: {cost.get('cost_benefit_ratio', 0):.1f}x")
            print()
        
        # Export the report
        output_dir = Path("generated_reports")
        output_dir.mkdir(exist_ok=True)
        
        report_file = output_dir / f"capacity_plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        planner.export_capacity_plan(report, str(report_file))
        
        print(f"✓ Capacity plan exported to: {report_file}")
        
    except Exception as e:
        print(f"✗ Error generating capacity plan: {e}")


def demo_integration():
    """Demonstrate integration between systems."""
    print_section("SYSTEM INTEGRATION DEMO")
    
    # Get all systems
    generator = get_dashboard_generator()
    alerting = get_alerting_system()
    planner = get_capacity_planner()
    monitor = get_platform_monitor()
    
    print_subsection("Cross-System Integration")
    
    # Create alert management dashboard
    print("Creating integrated alert management dashboard...")
    dashboard = generator.create_dashboard('alert_management')
    
    # Start monitoring and alerting
    print("Starting platform monitoring...")
    monitor.start_monitoring(interval_seconds=30)
    
    print("Starting alerting system...")
    alerting.start_alerting(check_interval_seconds=15)
    
    # Let systems run briefly
    print("Systems running for 10 seconds...")
    time.sleep(10)
    
    # Get integrated status
    print_subsection("Integrated System Status")
    
    # Platform health
    health = monitor.get_health_status()
    print(f"Platform health: {health['status']}")
    
    # Alerting status
    alert_stats = alerting.get_alerting_statistics()
    print(f"Alerting active: {alert_stats['alerting_active']}")
    print(f"Active incidents: {alert_stats['active_incidents']}")
    
    # Capacity status
    try:
        capacity = planner.get_current_capacity_status()
        overall_capacity = capacity.get('overall', {}).get('status', 'unknown')
        print(f"Capacity status: {overall_capacity}")
    except:
        print("Capacity status: unavailable")
    
    # Dashboard data integration
    print_subsection("Dashboard Data Integration")
    
    try:
        dashboard_data = generator.get_dashboard_data(dashboard, 1)
        data_sources = list(dashboard_data.keys())
        print(f"Dashboard data sources: {', '.join(data_sources)}")
        
        # Check for alerts in dashboard data
        alerts = dashboard_data.get('alerts', [])
        print(f"Alerts in dashboard: {len(alerts)}")
        
    except Exception as e:
        print(f"Dashboard data integration: {e}")
    
    # Stop systems
    print_subsection("Stopping Systems")
    
    monitor.stop_monitoring()
    alerting.stop_alerting()
    
    print("✓ All systems stopped")
    
    # Show final statistics
    print_subsection("Final Statistics")
    
    final_stats = alerting.get_alerting_statistics()
    print(f"Total incidents processed: {final_stats['total_incidents']}")
    print(f"Alert rules configured: {final_stats['alert_rules_count']}")
    print(f"Contacts configured: {final_stats['contacts_count']}")


def main():
    """Run the complete operational dashboards and alerting demo."""
    print("AI Data Readiness Platform - Operational Dashboards & Alerting Demo")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Run individual demos
        demo_dashboard_generator()
        demo_alerting_system()
        demo_capacity_planner()
        demo_integration()
        
        print_section("DEMO COMPLETED SUCCESSFULLY")
        print("✓ Dashboard generation demonstrated")
        print("✓ Alerting system demonstrated")
        print("✓ Capacity planning demonstrated")
        print("✓ System integration demonstrated")
        
        print(f"\nGenerated files:")
        print("• generated_dashboards/executive_dashboard.html")
        print("• generated_dashboards/system_overview_config.json")
        print("• generated_reports/capacity_plan_*.json")
        
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
    except Exception as e:
        print(f"\n\nDemo failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\nDemo completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()