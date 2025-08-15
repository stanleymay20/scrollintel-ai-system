"""
Demo script for BI and Analytics Tool Integration functionality

This script demonstrates the capabilities of the BI and Analytics Integration engine
including connecting to BI tools, creating data sources, exporting data in multiple formats,
and distributing reports to stakeholders.
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List
import tempfile
import os

from ai_data_readiness.engines.bi_analytics_integrator import (
    BIAnalyticsIntegrator,
    BIToolConfig,
    DataExportConfig,
    ReportDistributionInfo,
    DataSourceInfo
)


def print_section(title: str):
    """Print a formatted section header"""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")


def print_subsection(title: str):
    """Print a formatted subsection header"""
    print(f"\n{'-'*40}")
    print(f" {title}")
    print(f"{'-'*40}")


def create_sample_dataset() -> pd.DataFrame:
    """Create a sample dataset for demonstration"""
    # Generate sample AI readiness data
    data = {
        'dataset_id': [f'DS_{i:03d}' for i in range(1, 101)],
        'dataset_name': [f'Dataset {i}' for i in range(1, 101)],
        'quality_score': [round(0.6 + (i % 40) * 0.01, 2) for i in range(1, 101)],
        'completeness': [round(0.7 + (i % 30) * 0.01, 2) for i in range(1, 101)],
        'accuracy': [round(0.65 + (i % 35) * 0.01, 2) for i in range(1, 101)],
        'consistency': [round(0.75 + (i % 25) * 0.01, 2) for i in range(1, 101)],
        'ai_readiness_score': [round(0.6 + (i % 40) * 0.01, 2) for i in range(1, 101)],
        'bias_score': [round(0.8 + (i % 20) * 0.01, 2) for i in range(1, 101)],
        'compliance_status': ['Compliant' if i % 3 == 0 else 'Needs Review' if i % 3 == 1 else 'Non-Compliant' for i in range(1, 101)],
        'data_size_gb': [round(0.1 + (i % 50) * 0.5, 1) for i in range(1, 101)],
        'last_updated': [datetime.now() - timedelta(days=i % 30) for i in range(1, 101)],
        'industry': ['Finance' if i % 4 == 0 else 'Healthcare' if i % 4 == 1 else 'Retail' if i % 4 == 2 else 'Technology' for i in range(1, 101)]
    }
    
    return pd.DataFrame(data)


def demo_bi_tool_registration():
    """Demonstrate BI tool registration"""
    print_section("BI Tool Registration Demo")
    
    integrator = BIAnalyticsIntegrator()
    
    # Register different types of BI tools
    bi_tools = [
        {
            "name": "tableau_production",
            "config": BIToolConfig(
                tool_type="tableau",
                server_url="https://tableau.company.com",
                credentials={
                    "username": "ai_readiness_user",
                    "password": "secure_password_123",
                    "site_id": "ai_analytics"
                },
                workspace_id="ai_readiness_project",
                metadata={"environment": "production", "region": "us-east-1"}
            )
        },
        {
            "name": "powerbi_analytics",
            "config": BIToolConfig(
                tool_type="powerbi",
                server_url="https://api.powerbi.com",
                credentials={
                    "client_id": "12345678-1234-1234-1234-123456789012",
                    "client_secret": "secure_client_secret_456",
                    "tenant_id": "87654321-4321-4321-4321-210987654321"
                },
                workspace_id="ai-readiness-workspace",
                metadata={"environment": "production", "subscription": "premium"}
            )
        },
        {
            "name": "looker_dashboard",
            "config": BIToolConfig(
                tool_type="generic",
                server_url="https://looker.company.com",
                credentials={
                    "api_key": "looker_api_key_789",
                    "secret": "looker_secret_012"
                },
                workspace_id="ai_analytics_folder",
                metadata={"tool_version": "22.4", "instance_type": "enterprise"}
            )
        }
    ]
    
    print("Registering BI and Analytics tools...")
    registered_tools = []
    
    for tool in bi_tools:
        print(f"\nRegistering tool: {tool['name']}")
        print(f"  Type: {tool['config'].tool_type}")
        print(f"  Server: {tool['config'].server_url}")
        print(f"  Workspace: {tool['config'].workspace_id}")
        
        # In a real scenario, this would attempt actual connections
        # For demo purposes, we'll simulate the registration
        try:
            # Simulate successful registration for demo
            success = True  # integrator.register_bi_tool(tool['name'], tool['config'])
            if success:
                print(f"  ✓ Successfully registered {tool['name']}")
                registered_tools.append(tool['name'])
            else:
                print(f"  ✗ Failed to register {tool['name']} (connection failed)")
        except Exception as e:
            print(f"  ✗ Registration failed: {str(e)}")
    
    print(f"\nSuccessfully registered {len(registered_tools)} BI tools:")
    for tool_name in registered_tools:
        print(f"  • {tool_name}")
    
    return integrator, registered_tools


def demo_data_source_creation(integrator: BIAnalyticsIntegrator, sample_data: pd.DataFrame):
    """Demonstrate data source creation across BI tools"""
    print_section("Data Source Creation Demo")
    
    # Define datasets to create in BI tools
    datasets = [
        {
            "id": "ai_readiness_summary",
            "name": "AI Readiness Summary Dashboard",
            "description": "Comprehensive view of AI readiness metrics across all datasets",
            "table_name": "ai_readiness_metrics",
            "row_count": len(sample_data),
            "columns": [
                {"name": "dataset_id", "type": "string"},
                {"name": "quality_score", "type": "decimal"},
                {"name": "ai_readiness_score", "type": "decimal"},
                {"name": "compliance_status", "type": "string"},
                {"name": "industry", "type": "string"}
            ]
        },
        {
            "id": "quality_trends",
            "name": "Data Quality Trends",
            "description": "Historical trends in data quality metrics",
            "table_name": "quality_trends",
            "row_count": len(sample_data),
            "columns": [
                {"name": "dataset_id", "type": "string"},
                {"name": "completeness", "type": "decimal"},
                {"name": "accuracy", "type": "decimal"},
                {"name": "consistency", "type": "decimal"},
                {"name": "last_updated", "type": "datetime"}
            ]
        }
    ]
    
    print("Creating data sources across BI tools...")
    
    for dataset in datasets:
        print_subsection(f"Creating: {dataset['name']}")
        print(f"Dataset ID: {dataset['id']}")
        print(f"Description: {dataset['description']}")
        print(f"Rows: {dataset['row_count']:,}")
        print(f"Columns: {len(dataset['columns'])}")
        
        # Simulate data source creation results
        creation_results = {
            "tableau_production": DataSourceInfo(
                source_id=f"tableau_{dataset['id']}",
                source_name=dataset['name'],
                connection_type="extract",
                dataset_id=dataset['id'],
                last_refresh=datetime.now(),
                refresh_status="success",
                row_count=dataset['row_count'],
                metadata={"project_id": "ai_readiness_project", "extract_type": "full"}
            ),
            "powerbi_analytics": DataSourceInfo(
                source_id=f"powerbi_{dataset['id']}",
                source_name=dataset['name'],
                connection_type="dataset",
                dataset_id=dataset['id'],
                last_refresh=datetime.now(),
                refresh_status="success",
                row_count=dataset['row_count'],
                metadata={"workspace_id": "ai-readiness-workspace", "refresh_type": "scheduled"}
            ),
            "looker_dashboard": DataSourceInfo(
                source_id=f"looker_{dataset['id']}",
                source_name=dataset['name'],
                connection_type="view",
                dataset_id=dataset['id'],
                last_refresh=datetime.now(),
                refresh_status="success",
                row_count=dataset['row_count'],
                metadata={"folder_id": "ai_analytics_folder", "model": "ai_readiness"}
            )
        }
        
        print("\nCreation Results:")
        for tool_name, source_info in creation_results.items():
            if source_info:
                print(f"  ✓ {tool_name}: {source_info.source_id}")
                print(f"    Status: {source_info.refresh_status}")
                print(f"    Rows: {source_info.row_count:,}")
            else:
                print(f"  ✗ {tool_name}: Failed to create")
    
    return datasets


def demo_data_export(integrator: BIAnalyticsIntegrator, sample_data: pd.DataFrame):
    """Demonstrate data export in multiple formats"""
    print_section("Data Export Demo")
    
    print(f"Exporting dataset with {len(sample_data)} rows and {len(sample_data.columns)} columns")
    print(f"Sample data preview:")
    print(sample_data.head(3).to_string(index=False))
    
    # Define export configurations
    export_configs = [
        DataExportConfig(
            export_format="csv",
            include_metadata=True,
            compression=None
        ),
        DataExportConfig(
            export_format="json",
            include_metadata=True,
            compression=None
        ),
        DataExportConfig(
            export_format="excel",
            include_metadata=True
        ),
        DataExportConfig(
            export_format="parquet",
            include_metadata=False,
            compression="snappy"
        ),
        DataExportConfig(
            export_format="csv",
            include_metadata=False,
            filters={
                "compliance_status": {"equals": "Compliant"},
                "quality_score": {"range": [0.8, 1.0]}
            }
        )
    ]
    
    print_subsection("Export Configurations")
    for i, config in enumerate(export_configs, 1):
        print(f"{i}. Format: {config.export_format.upper()}")
        print(f"   Metadata: {'Yes' if config.include_metadata else 'No'}")
        if config.compression:
            print(f"   Compression: {config.compression}")
        if config.filters:
            print(f"   Filters: {config.filters}")
        print()
    
    # Perform exports
    print_subsection("Export Results")
    export_results = {}
    
    for config in export_configs:
        try:
            # Create temporary file for demonstration
            suffix = f".{config.export_format}"
            if config.filters:
                suffix = f"_filtered{suffix}"
            
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_file:
                result = integrator.data_exporter.export_dataset(
                    sample_data, 
                    config, 
                    tmp_file.name
                )
                
                # Get file size
                file_size = os.path.getsize(tmp_file.name)
                
                # Count rows in filtered data if filters applied
                if config.filters:
                    filtered_data = sample_data.copy()
                    for column, filter_config in config.filters.items():
                        if 'equals' in filter_config:
                            filtered_data = filtered_data[filtered_data[column] == filter_config['equals']]
                        elif 'range' in filter_config:
                            min_val, max_val = filter_config['range']
                            filtered_data = filtered_data[
                                (filtered_data[column] >= min_val) & 
                                (filtered_data[column] <= max_val)
                            ]
                    row_count = len(filtered_data)
                else:
                    row_count = len(sample_data)
                
                export_key = config.export_format
                if config.filters:
                    export_key += "_filtered"
                
                export_results[export_key] = {
                    'file_path': tmp_file.name,
                    'file_size_kb': round(file_size / 1024, 2),
                    'row_count': row_count,
                    'format': config.export_format,
                    'compressed': config.compression is not None
                }
                
                print(f"✓ {config.export_format.upper()}")
                if config.filters:
                    print(f"  (Filtered)")
                print(f"  File: {os.path.basename(tmp_file.name)}")
                print(f"  Size: {export_results[export_key]['file_size_kb']} KB")
                print(f"  Rows: {row_count:,}")
                print()
                
        except Exception as e:
            print(f"✗ Failed to export {config.export_format}: {str(e)}")
    
    # Summary
    print_subsection("Export Summary")
    total_exports = len([r for r in export_results.values() if r])
    total_size_kb = sum(r['file_size_kb'] for r in export_results.values() if r)
    
    print(f"Successful exports: {total_exports}/{len(export_configs)}")
    print(f"Total size: {total_size_kb:.2f} KB")
    print(f"Formats: {', '.join(set(r['format'] for r in export_results.values()))}")
    
    # Clean up temporary files
    for result in export_results.values():
        try:
            if os.path.exists(result['file_path']):
                os.unlink(result['file_path'])
        except:
            pass
    
    return export_results


def demo_report_distribution():
    """Demonstrate automated report distribution"""
    print_section("Report Distribution Demo")
    
    # Define report distribution configurations
    distributions = [
        ReportDistributionInfo(
            report_id="ai_readiness_executive_summary",
            report_name="AI Readiness Executive Summary",
            recipients=[
                "ceo@company.com",
                "cto@company.com",
                "head.of.data@company.com"
            ],
            distribution_schedule="weekly",
            format="pdf",
            status="active",
            metadata={
                "priority": "high",
                "template": "executive",
                "include_recommendations": True
            }
        ),
        ReportDistributionInfo(
            report_id="quality_metrics_dashboard",
            report_name="Data Quality Metrics Dashboard",
            recipients=[
                "data.team@company.com",
                "quality.analysts@company.com",
                "ml.engineers@company.com"
            ],
            distribution_schedule="daily",
            format="excel",
            status="active",
            metadata={
                "priority": "medium",
                "template": "detailed",
                "include_raw_data": True
            }
        ),
        ReportDistributionInfo(
            report_id="compliance_status_report",
            report_name="AI Compliance Status Report",
            recipients=[
                "compliance@company.com",
                "legal@company.com",
                "risk.management@company.com"
            ],
            distribution_schedule="monthly",
            format="pdf",
            status="active",
            metadata={
                "priority": "high",
                "template": "compliance",
                "include_audit_trail": True
            }
        )
    ]
    
    print("Setting up automated report distributions...")
    
    for dist in distributions:
        print_subsection(f"Report: {dist.report_name}")
        print(f"Report ID: {dist.report_id}")
        print(f"Schedule: {dist.distribution_schedule.capitalize()}")
        print(f"Format: {dist.format.upper()}")
        print(f"Recipients: {len(dist.recipients)}")
        for recipient in dist.recipients:
            print(f"  • {recipient}")
        print(f"Priority: {dist.metadata.get('priority', 'normal').capitalize()}")
        
        # Simulate distribution setup
        print(f"✓ Distribution configured successfully")
        
        # Simulate next scheduled time
        if dist.distribution_schedule == "daily":
            next_send = datetime.now().replace(hour=8, minute=0, second=0) + timedelta(days=1)
        elif dist.distribution_schedule == "weekly":
            days_until_monday = (7 - datetime.now().weekday()) % 7
            if days_until_monday == 0:
                days_until_monday = 7
            next_send = datetime.now().replace(hour=8, minute=0, second=0) + timedelta(days=days_until_monday)
        else:  # monthly
            next_month = datetime.now().replace(day=1, hour=8, minute=0, second=0) + timedelta(days=32)
            next_send = next_month.replace(day=1)
        
        print(f"Next scheduled: {next_send.strftime('%Y-%m-%d %H:%M')}")
        print()
    
    # Distribution summary
    print_subsection("Distribution Summary")
    total_recipients = len(set(recipient for dist in distributions for recipient in dist.recipients))
    daily_reports = len([d for d in distributions if d.distribution_schedule == "daily"])
    weekly_reports = len([d for d in distributions if d.distribution_schedule == "weekly"])
    monthly_reports = len([d for d in distributions if d.distribution_schedule == "monthly"])
    
    print(f"Total reports configured: {len(distributions)}")
    print(f"Unique recipients: {total_recipients}")
    print(f"Distribution frequency:")
    print(f"  • Daily: {daily_reports}")
    print(f"  • Weekly: {weekly_reports}")
    print(f"  • Monthly: {monthly_reports}")
    
    return distributions


def demo_integration_monitoring():
    """Demonstrate BI integration monitoring and health checks"""
    print_section("Integration Monitoring Demo")
    
    # Simulate integration status
    integration_status = {
        "tableau_production": {
            "connected": True,
            "tool_type": "tableau",
            "server_url": "https://tableau.company.com",
            "workspace_id": "ai_readiness_project",
            "data_source_count": 5,
            "last_sync": datetime.now() - timedelta(minutes=15),
            "response_time_ms": 180,
            "status": "healthy",
            "version": "2023.3"
        },
        "powerbi_analytics": {
            "connected": True,
            "tool_type": "powerbi",
            "server_url": "https://api.powerbi.com",
            "workspace_id": "ai-readiness-workspace",
            "data_source_count": 3,
            "last_sync": datetime.now() - timedelta(minutes=5),
            "response_time_ms": 95,
            "status": "healthy",
            "premium_capacity": True
        },
        "looker_dashboard": {
            "connected": False,
            "tool_type": "generic",
            "server_url": "https://looker.company.com",
            "workspace_id": "ai_analytics_folder",
            "data_source_count": 0,
            "last_sync": datetime.now() - timedelta(hours=2),
            "response_time_ms": None,
            "status": "disconnected",
            "error": "Authentication token expired"
        }
    }
    
    print("Checking health status of all BI integrations...")
    
    for tool_name, status in integration_status.items():
        print_subsection(f"Tool: {tool_name}")
        
        # Connection status
        if status['connected']:
            print(f"Status: 🟢 Connected")
            print(f"Response Time: {status['response_time_ms']}ms")
            print(f"Data Sources: {status['data_source_count']}")
            print(f"Last Sync: {status['last_sync'].strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            print(f"Status: 🔴 Disconnected")
            print(f"Error: {status.get('error', 'Unknown error')}")
            print(f"Last Successful Sync: {status['last_sync'].strftime('%Y-%m-%d %H:%M:%S')}")
        
        print(f"Type: {status['tool_type'].title()}")
        print(f"Server: {status['server_url']}")
        print(f"Workspace: {status['workspace_id']}")
        
        # Additional tool-specific information
        if 'version' in status:
            print(f"Version: {status['version']}")
        if 'premium_capacity' in status:
            print(f"Premium Capacity: {'Yes' if status['premium_capacity'] else 'No'}")
    
    # Overall health summary
    print_subsection("Overall Health Summary")
    connected_count = sum(1 for s in integration_status.values() if s['connected'])
    total_count = len(integration_status)
    total_data_sources = sum(s['data_source_count'] for s in integration_status.values())
    avg_response_time = sum(s['response_time_ms'] for s in integration_status.values() if s['response_time_ms']) / connected_count if connected_count > 0 else 0
    
    print(f"Connected Tools: {connected_count}/{total_count}")
    print(f"Total Data Sources: {total_data_sources}")
    print(f"Average Response Time: {avg_response_time:.0f}ms")
    
    if connected_count == total_count:
        print(f"System Health: 🟢 Excellent")
    elif connected_count > total_count / 2:
        print(f"System Health: 🟡 Good (some issues)")
    else:
        print(f"System Health: 🔴 Critical (multiple failures)")
    
    # Recommendations
    print_subsection("Recommendations")
    recommendations = []
    
    for tool_name, status in integration_status.items():
        if not status['connected']:
            recommendations.append(f"• Reconnect {tool_name} - {status.get('error', 'Connection failed')}")
        elif status.get('response_time_ms', 0) > 200:
            recommendations.append(f"• Optimize {tool_name} performance - high response time ({status['response_time_ms']}ms)")
        elif status['data_source_count'] == 0:
            recommendations.append(f"• Add data sources to {tool_name}")
    
    if not recommendations:
        print("✓ All systems operating optimally")
    else:
        for rec in recommendations:
            print(rec)


def demo_best_practices():
    """Demonstrate BI integration best practices and tips"""
    print_section("BI Integration Best Practices")
    
    best_practices = [
        {
            "category": "Data Source Management",
            "practices": [
                "Use consistent naming conventions across all BI tools",
                "Implement automated data refresh schedules",
                "Monitor data source performance and optimize queries",
                "Set up alerts for data refresh failures",
                "Document data lineage and transformations"
            ]
        },
        {
            "category": "Export and Distribution",
            "practices": [
                "Choose appropriate export formats based on use case",
                "Implement data filtering to reduce file sizes",
                "Use compression for large datasets",
                "Set up automated report distribution schedules",
                "Include metadata for better data understanding"
            ]
        },
        {
            "category": "Security and Compliance",
            "practices": [
                "Use secure authentication methods (OAuth, tokens)",
                "Implement role-based access controls",
                "Encrypt sensitive data in exports",
                "Audit data access and distribution",
                "Comply with data privacy regulations (GDPR, CCPA)"
            ]
        },
        {
            "category": "Performance Optimization",
            "practices": [
                "Cache frequently accessed data",
                "Use incremental data updates when possible",
                "Optimize data models for BI tool performance",
                "Monitor and tune query performance",
                "Implement data archiving strategies"
            ]
        },
        {
            "category": "Monitoring and Maintenance",
            "practices": [
                "Set up comprehensive health monitoring",
                "Implement automated error detection and alerting",
                "Regularly review and update integration configurations",
                "Maintain backup and recovery procedures",
                "Document troubleshooting procedures"
            ]
        }
    ]
    
    for practice_group in best_practices:
        print_subsection(practice_group['category'])
        for practice in practice_group['practices']:
            print(f"  • {practice}")


def main():
    """Run the complete BI and Analytics Integration demo"""
    print("🚀 AI Data Readiness Platform - BI & Analytics Integration Demo")
    print("This demo showcases BI and analytics tool integration capabilities")
    
    try:
        # Create sample dataset
        sample_data = create_sample_dataset()
        print(f"\nGenerated sample dataset: {len(sample_data)} rows, {len(sample_data.columns)} columns")
        
        # Demo BI tool registration
        integrator, registered_tools = demo_bi_tool_registration()
        
        # Demo data source creation
        datasets = demo_data_source_creation(integrator, sample_data)
        
        # Demo data export
        export_results = demo_data_export(integrator, sample_data)
        
        # Demo report distribution
        distributions = demo_report_distribution()
        
        # Demo integration monitoring
        demo_integration_monitoring()
        
        # Demo best practices
        demo_best_practices()
        
        print_section("Demo Completed Successfully")
        print("✅ All BI and analytics integration features demonstrated")
        print("\nKey capabilities showcased:")
        print("  • Multi-platform BI tool integration (Tableau, Power BI, Looker)")
        print("  • Automated data source creation and synchronization")
        print("  • Multi-format data export (CSV, JSON, Excel, Parquet)")
        print("  • Intelligent data filtering and compression")
        print("  • Automated report distribution to stakeholders")
        print("  • Comprehensive integration monitoring and health checks")
        print("  • Best practices for enterprise BI integration")
        
        print(f"\nIntegration Summary:")
        print(f"  • BI Tools Registered: {len(registered_tools)}")
        print(f"  • Data Sources Created: {len(datasets) * len(registered_tools)}")
        print(f"  • Export Formats: {len(export_results)}")
        print(f"  • Report Distributions: {len(distributions)}")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()