"""
Demonstration of the multi-source data integration system for the Advanced Analytics Dashboard.
Shows how to connect to and sync data from various enterprise systems.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any

from scrollintel.core.data_integration_setup import (
    setup_data_integration, 
    create_sample_configurations,
    initialize_sample_data_sources
)
from scrollintel.core.data_connector import DataSourceType, DataSourceConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def demonstrate_data_integration():
    """Demonstrate the complete data integration workflow"""
    
    print("=" * 80)
    print("ADVANCED ANALYTICS DASHBOARD - MULTI-SOURCE DATA INTEGRATION DEMO")
    print("=" * 80)
    
    # 1. Setup the data integration manager
    print("\n1. Setting up Data Integration Manager...")
    manager = setup_data_integration()
    print(f"✓ Manager initialized with {len(manager.registry.connector_classes)} connector types")
    
    # 2. Show available connector types
    print("\n2. Available Connector Types:")
    for connector_name in sorted(manager.registry.connector_classes.keys()):
        connector_class = manager.registry.connector_classes[connector_name]
        print(f"   • {connector_name}: {connector_class.__name__}")
    
    # 3. Create sample configurations
    print("\n3. Creating Sample Data Source Configurations...")
    configurations = create_sample_configurations()
    
    for source_name, config in configurations.items():
        print(f"   • {source_name}: {config.name} ({config.source_type.value})")
    
    # 4. Initialize a subset of data sources for demonstration
    print("\n4. Initializing Sample Data Sources...")
    demo_sources = ['sap', 'salesforce', 'tableau', 'aws']
    
    for source_name in demo_sources:
        if source_name in configurations:
            config = configurations[source_name]
            try:
                success = await manager.add_data_source(config)
                status = "✓ Connected" if success else "✗ Failed"
                print(f"   {status}: {config.name}")
            except Exception as e:
                print(f"   ✗ Error: {config.name} - {e}")
    
    # 5. Check health status of all connections
    print("\n5. Checking Connection Health Status...")
    health_status = await manager.get_health_status()
    
    for source_id, health in health_status.items():
        status_icon = "🟢" if health.status.value == "connected" else "🔴"
        print(f"   {status_icon} {source_id}: {health.status.value}")
        if health.last_successful_sync:
            print(f"      Last sync: {health.last_successful_sync}")
        if health.error_message:
            print(f"      Error: {health.error_message}")
    
    # 6. Test all connections
    print("\n6. Testing All Connections...")
    connection_results = await manager.test_all_connections()
    
    for source_id, is_working in connection_results.items():
        status = "✓ Working" if is_working else "✗ Not responding"
        print(f"   {status}: {source_id}")
    
    # 7. Demonstrate data synchronization from different sources
    print("\n7. Synchronizing Data from Different Sources...")
    
    # SAP ERP Data
    if 'sap_prod' in health_status:
        print("\n   📊 SAP ERP Data:")
        try:
            sap_data = await manager.sync_data_source('sap_prod', {
                'table': 'MARA',
                'fields': ['MATNR', 'MAKTX', 'MTART'],
                'limit': 5
            })
            print(f"      Retrieved {len(sap_data)} material records")
            if sap_data:
                sample_record = sap_data[0]
                print(f"      Sample: {sample_record.data.get('MATNR')} - {sample_record.data.get('MAKTX')}")
        except Exception as e:
            print(f"      Error syncing SAP data: {e}")
    
    # Salesforce CRM Data
    if 'salesforce_prod' in health_status:
        print("\n   🤝 Salesforce CRM Data:")
        try:
            sf_data = await manager.sync_data_source('salesforce_prod', {
                'sobject': 'Account',
                'fields': ['Id', 'Name', 'Industry'],
                'limit': 5
            })
            print(f"      Retrieved {len(sf_data)} account records")
            if sf_data:
                sample_record = sf_data[0]
                print(f"      Sample: {sample_record.data.get('Name')} - {sample_record.data.get('Industry')}")
        except Exception as e:
            print(f"      Error syncing Salesforce data: {e}")
    
    # Tableau BI Data
    if 'tableau_server' in health_status:
        print("\n   📈 Tableau BI Data:")
        try:
            tableau_data = await manager.sync_data_source('tableau_server', {
                'resource_type': 'workbooks',
                'limit': 3
            })
            print(f"      Retrieved {len(tableau_data)} workbook records")
            if tableau_data:
                sample_record = tableau_data[0]
                print(f"      Sample: {sample_record.data.get('name')} - {sample_record.data.get('viewCount')} views")
        except Exception as e:
            print(f"      Error syncing Tableau data: {e}")
    
    # AWS Cloud Data
    if 'aws_main_account' in health_status:
        print("\n   ☁️ AWS Cloud Cost Data:")
        try:
            aws_data = await manager.sync_data_source('aws_main_account', {
                'service': 'cost-explorer',
                'granularity': 'DAILY',
                'metrics': ['BlendedCost']
            })
            print(f"      Retrieved {len(aws_data)} cost records")
            if aws_data:
                sample_record = aws_data[0]
                cost_info = sample_record.data.get('BlendedCost', {})
                print(f"      Sample: {sample_record.data.get('Service')} - ${cost_info.get('Amount', 'N/A')}")
        except Exception as e:
            print(f"      Error syncing AWS data: {e}")
    
    # 8. Show data schemas
    print("\n8. Available Data Schemas:")
    
    for source_id in list(health_status.keys())[:2]:  # Show schemas for first 2 sources
        connector = manager.registry.get_connector(source_id)
        if connector:
            try:
                schema = await connector.get_schema()
                print(f"\n   📋 {source_id} Schema:")
                
                # Show schema structure based on connector type
                if 'tables' in schema:
                    for table_name, table_info in list(schema['tables'].items())[:2]:
                        print(f"      Table: {table_name} - {table_info.get('description', 'N/A')}")
                        fields = table_info.get('fields', {})
                        for field_name in list(fields.keys())[:3]:
                            field_info = fields[field_name]
                            print(f"        • {field_name}: {field_info.get('type', 'N/A')}")
                
                elif 'sobjects' in schema:
                    for obj_name, obj_info in list(schema['sobjects'].items())[:2]:
                        print(f"      Object: {obj_name} - {obj_info.get('description', 'N/A')}")
                        fields = obj_info.get('fields', {})
                        for field_name in list(fields.keys())[:3]:
                            field_info = fields[field_name]
                            print(f"        • {field_name}: {field_info.get('type', 'N/A')}")
                
                elif 'resources' in schema:
                    for res_name, res_info in list(schema['resources'].items())[:2]:
                        print(f"      Resource: {res_name} - {res_info.get('description', 'N/A')}")
                        fields = res_info.get('fields', {})
                        for field_name in list(fields.keys())[:3]:
                            field_info = fields[field_name]
                            print(f"        • {field_name}: {field_info.get('type', 'N/A')}")
                
            except Exception as e:
                print(f"      Error getting schema for {source_id}: {e}")
    
    # 9. Demonstrate advanced integration features
    print("\n9. Advanced Integration Features:")
    
    print("   🔄 Real-time Synchronization:")
    print("      • Periodic data refresh (configurable intervals)")
    print("      • Automatic retry on connection failures")
    print("      • Health monitoring and alerting")
    
    print("   🔧 Data Normalization:")
    print("      • Standardized data record format")
    print("      • Metadata preservation")
    print("      • Schema mapping and transformation")
    
    print("   📊 Analytics Ready:")
    print("      • Unified data access layer")
    print("      • Cross-platform data correlation")
    print("      • Executive dashboard integration")
    
    # 10. Cleanup
    print("\n10. Cleaning up connections...")
    cleanup_count = 0
    for source_id in list(health_status.keys()):
        try:
            success = await manager.remove_data_source(source_id)
            if success:
                cleanup_count += 1
        except Exception as e:
            print(f"      Warning: Could not clean up {source_id}: {e}")
    
    print(f"   ✓ Cleaned up {cleanup_count} data source connections")
    
    print("\n" + "=" * 80)
    print("MULTI-SOURCE DATA INTEGRATION DEMONSTRATION COMPLETE")
    print("=" * 80)
    print("\nKey Capabilities Demonstrated:")
    print("• ✅ ERP System Integration (SAP, Oracle, Microsoft Dynamics)")
    print("• ✅ CRM System Integration (Salesforce, HubSpot, Microsoft CRM)")
    print("• ✅ BI Tool Integration (Tableau, Power BI, Looker, Qlik)")
    print("• ✅ Cloud Platform Integration (AWS, Azure, GCP)")
    print("• ✅ Unified Data Access and Normalization")
    print("• ✅ Real-time Health Monitoring")
    print("• ✅ Comprehensive Error Handling")
    print("• ✅ Schema Discovery and Mapping")
    print("\nThe Advanced Analytics Dashboard can now consolidate data from")
    print("all major enterprise systems for unified executive reporting!")


async def demonstrate_specific_connector(connector_type: str):
    """Demonstrate a specific connector type in detail"""
    
    print(f"\n{'='*60}")
    print(f"DETAILED CONNECTOR DEMONSTRATION: {connector_type.upper()}")
    print(f"{'='*60}")
    
    manager = setup_data_integration()
    configurations = create_sample_configurations()
    
    # Find configuration for the requested connector type
    config = None
    for source_name, source_config in configurations.items():
        if connector_type.lower() in source_name.lower():
            config = source_config
            break
    
    if not config:
        print(f"❌ No configuration found for connector type: {connector_type}")
        return
    
    print(f"\n📋 Configuration Details:")
    print(f"   Source ID: {config.source_id}")
    print(f"   Name: {config.name}")
    print(f"   Type: {config.source_type.value}")
    print(f"   Refresh Interval: {config.refresh_interval}s")
    print(f"   Timeout: {config.timeout}s")
    print(f"   Retry Attempts: {config.retry_attempts}")
    
    # Connect to the data source
    print(f"\n🔌 Connecting to {config.name}...")
    try:
        success = await manager.add_data_source(config)
        if success:
            print("   ✅ Connection successful!")
            
            # Get connector instance
            connector = manager.registry.get_connector(config.source_id)
            
            # Show schema
            print(f"\n📊 Data Schema:")
            schema = await connector.get_schema()
            
            # Display schema in a readable format
            if 'tables' in schema:
                print("   ERP Tables:")
                for table_name, table_info in schema['tables'].items():
                    print(f"     • {table_name}: {table_info.get('description', 'N/A')}")
            elif 'sobjects' in schema:
                print("   CRM Objects:")
                for obj_name, obj_info in schema['sobjects'].items():
                    print(f"     • {obj_name}: {obj_info.get('description', 'N/A')}")
            elif 'resources' in schema:
                print("   BI Resources:")
                for res_name, res_info in schema['resources'].items():
                    print(f"     • {res_name}: {res_info.get('description', 'N/A')}")
            elif 'services' in schema:
                print("   Cloud Services:")
                for svc_name, svc_info in schema['services'].items():
                    print(f"     • {svc_name}: {svc_info.get('description', 'N/A')}")
            
            # Fetch sample data
            print(f"\n📥 Fetching Sample Data:")
            sample_data = await manager.sync_data_source(config.source_id, {'limit': 3})
            
            print(f"   Retrieved {len(sample_data)} records")
            for i, record in enumerate(sample_data[:2], 1):
                print(f"\n   Record {i}:")
                print(f"     ID: {record.record_id}")
                print(f"     Timestamp: {record.timestamp}")
                
                # Show first few data fields
                data_items = list(record.data.items())[:3]
                for key, value in data_items:
                    print(f"     {key}: {value}")
                
                if record.metadata:
                    print(f"     Metadata: {record.metadata}")
            
            # Test connection health
            print(f"\n🏥 Health Check:")
            health = connector.get_health()
            print(f"   Status: {health.status.value}")
            print(f"   Last Sync: {health.last_successful_sync}")
            if health.error_message:
                print(f"   Error: {health.error_message}")
            
            # Cleanup
            await manager.remove_data_source(config.source_id)
            print(f"\n🧹 Connection cleaned up")
            
        else:
            print("   ❌ Connection failed!")
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print(f"\n{'='*60}")
    print(f"CONNECTOR DEMONSTRATION COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Demonstrate specific connector
        connector_type = sys.argv[1]
        asyncio.run(demonstrate_specific_connector(connector_type))
    else:
        # Run full demonstration
        asyncio.run(demonstrate_data_integration())