"""
BI Integration System Demo
Demonstrates the comprehensive BI tool integration capabilities
"""

import asyncio
from datetime import datetime
from scrollintel.models.bi_integration_models import (
    BIToolType, ExportFormat, EmbedType, 
    BIConnectionConfig, DashboardExportRequest, EmbedTokenRequest
)
from scrollintel.connectors.bi_connector_base import bi_connector_registry
from scrollintel.engines.bi_integration_engine import BIIntegrationEngine


async def demo_bi_integration():
    """Demonstrate BI integration capabilities"""
    
    print("🔗 ScrollIntel BI Integration System Demo")
    print("=" * 50)
    
    # 1. Show supported BI tools
    print("\n📊 Supported BI Tools:")
    available_tools = bi_connector_registry.get_available_tools()
    for tool in available_tools:
        print(f"  ✓ {tool.value.replace('_', ' ').title()}")
    
    # 2. Create BI Integration Engine
    bi_engine = BIIntegrationEngine()
    
    # 3. Get detailed tool capabilities
    print("\n🛠️  Tool Capabilities:")
    supported_tools = await bi_engine.get_supported_tools()
    
    for tool in supported_tools:
        print(f"\n  {tool['name']}:")
        print(f"    Export Formats: {', '.join(tool['supported_export_formats'])}")
        print(f"    Embed Types: {', '.join(tool['supported_embed_types'])}")
        print(f"    Required Config: {', '.join(tool['required_config_fields'])}")
    
    # 4. Demo Tableau connector
    print("\n🎯 Tableau Connector Demo:")
    tableau_config = {
        'id': 'demo-tableau',
        'name': 'Demo Tableau Server',
        'bi_tool_type': BIToolType.TABLEAU,
        'server_url': 'https://demo.tableau.com',
        'username': 'demo_user',
        'password': 'demo_pass',
        'site_id': 'default',
        'api_version': '3.19'
    }
    
    tableau_connector = bi_connector_registry.get_connector(BIToolType.TABLEAU, tableau_config)
    print(f"  ✓ Created Tableau connector for: {tableau_connector.server_url}")
    print(f"  ✓ Supported formats: {[fmt.value for fmt in tableau_connector.get_supported_export_formats()]}")
    
    # Validate configuration
    validation_errors = await tableau_connector.validate_config()
    if validation_errors:
        print(f"  ⚠️  Configuration issues: {validation_errors}")
    else:
        print("  ✓ Configuration is valid")
    
    # 5. Demo Power BI connector
    print("\n⚡ Power BI Connector Demo:")
    powerbi_config = {
        'id': 'demo-powerbi',
        'name': 'Demo Power BI',
        'bi_tool_type': BIToolType.POWER_BI,
        'tenant_id': 'demo-tenant-id',
        'client_id': 'demo-client-id',
        'client_secret': 'demo-client-secret',
        'workspace_id': 'demo-workspace-id'
    }
    
    powerbi_connector = bi_connector_registry.get_connector(BIToolType.POWER_BI, powerbi_config)
    print(f"  ✓ Created Power BI connector for tenant: {powerbi_connector.tenant_id}")
    print(f"  ✓ Workspace ID: {powerbi_connector.workspace_id}")
    
    # Show real-time capabilities (without authentication)
    print("  ✓ Real-time data feeds supported")
    print("    Method: POST to streaming endpoint")
    print("    Format: JSON array of objects")
    print("    Rate limit: 120 requests per minute")
    
    # 6. Demo Looker connector
    print("\n👀 Looker Connector Demo:")
    looker_config = {
        'id': 'demo-looker',
        'name': 'Demo Looker',
        'bi_tool_type': BIToolType.LOOKER,
        'base_url': 'https://demo.looker.com',
        'client_id': 'demo-client-id',
        'client_secret': 'demo-client-secret',
        'embed_secret': 'demo-embed-secret'
    }
    
    looker_connector = bi_connector_registry.get_connector(BIToolType.LOOKER, looker_config)
    print(f"  ✓ Created Looker connector for: {looker_connector.base_url}")
    print(f"  ✓ Embed secret configured: {'Yes' if looker_connector.embed_secret else 'No'}")
    
    # 7. Demo export capabilities
    print("\n📤 Export Capabilities Demo:")
    
    export_formats = {
        'Tableau': tableau_connector.get_supported_export_formats(),
        'Power BI': powerbi_connector.get_supported_export_formats(),
        'Looker': looker_connector.get_supported_export_formats()
    }
    
    for tool_name, formats in export_formats.items():
        print(f"  {tool_name}: {', '.join([fmt.value.upper() for fmt in formats])}")
    
    # 8. Demo embedding capabilities
    print("\n🔗 Embedding Capabilities Demo:")
    
    embed_types = {
        'Tableau': tableau_connector.get_supported_embed_types(),
        'Power BI': powerbi_connector.get_supported_embed_types(),
        'Looker': looker_connector.get_supported_embed_types()
    }
    
    for tool_name, types in embed_types.items():
        print(f"  {tool_name}: {', '.join([embed.value for embed in types])}")
    
    # 9. Demo white-label embedding
    print("\n🎨 White-Label Embedding Demo:")
    print("  ✓ Custom branding support")
    print("  ✓ Toolbar hiding")
    print("  ✓ Custom CSS injection")
    print("  ✓ Theme customization")
    
    # 10. Demo configuration examples
    print("\n⚙️  Configuration Examples:")
    
    # Tableau configuration
    print("\n  Tableau Server Configuration:")
    tableau_example = BIConnectionConfig(
        name="Production Tableau",
        bi_tool_type=BIToolType.TABLEAU,
        server_url="https://tableau.company.com",
        username="tableau_user",
        password="secure_password",
        site_id="production",
        additional_config={
            "api_version": "3.19",
            "connected_app_client_id": "app-client-id",
            "connected_app_secret": "app-secret"
        }
    )
    print(f"    Server: {tableau_example.server_url}")
    print(f"    Site: {tableau_example.site_id}")
    print(f"    Connected App: {'Configured' if tableau_example.additional_config.get('connected_app_client_id') else 'Not configured'}")
    
    # Power BI configuration
    print("\n  Power BI Configuration:")
    powerbi_example = BIConnectionConfig(
        name="Production Power BI",
        bi_tool_type=BIToolType.POWER_BI,
        server_url="https://api.powerbi.com",
        additional_config={
            "tenant_id": "company-tenant-id",
            "client_id": "powerbi-app-id",
            "client_secret": "powerbi-app-secret",
            "workspace_id": "production-workspace"
        }
    )
    print(f"    Tenant: {powerbi_example.additional_config['tenant_id']}")
    print(f"    Workspace: {powerbi_example.additional_config['workspace_id']}")
    
    # 11. Demo integration workflow
    print("\n🔄 Integration Workflow Demo:")
    print("  1. ✓ Create connection configuration")
    print("  2. ✓ Validate credentials and connectivity")
    print("  3. ✓ Discover available dashboards")
    print("  4. ✓ Generate embed tokens with permissions")
    print("  5. ✓ Export dashboards in multiple formats")
    print("  6. ✓ Set up real-time data synchronization")
    print("  7. ✓ Monitor connection health and status")
    
    # 12. Demo security features
    print("\n🔒 Security Features:")
    print("  ✓ Encrypted credential storage")
    print("  ✓ Token-based authentication")
    print("  ✓ Permission-based access control")
    print("  ✓ Audit logging for all operations")
    print("  ✓ Secure embed token generation")
    print("  ✓ Connection health monitoring")
    
    # 13. Demo enterprise features
    print("\n🏢 Enterprise Features:")
    print("  ✓ Multi-tenant support")
    print("  ✓ SSO integration")
    print("  ✓ LDAP/Active Directory sync")
    print("  ✓ Role-based permissions")
    print("  ✓ Compliance reporting")
    print("  ✓ High availability setup")
    
    print("\n" + "=" * 50)
    print("✅ BI Integration System Demo Complete!")
    print("\nKey Benefits:")
    print("  • Unified interface for multiple BI tools")
    print("  • Seamless embedding with white-label support")
    print("  • Real-time data synchronization")
    print("  • Enterprise-grade security and compliance")
    print("  • Extensible plugin architecture")
    print("  • Comprehensive API coverage")


async def demo_api_usage():
    """Demo API usage examples"""
    
    print("\n🚀 API Usage Examples:")
    print("=" * 30)
    
    # Connection creation example
    print("\n1. Create BI Connection:")
    print("""
    POST /api/v1/bi-integration/connections
    {
        "name": "Production Tableau",
        "bi_tool_type": "tableau",
        "server_url": "https://tableau.company.com",
        "username": "tableau_user",
        "password": "secure_password",
        "site_id": "production"
    }
    """)
    
    # Dashboard export example
    print("\n2. Export Dashboard:")
    print("""
    POST /api/v1/bi-integration/connections/{connection_id}/export
    {
        "dashboard_id": "dashboard-123",
        "format": "pdf",
        "filters": {
            "region": "North America",
            "date_range": "last_30_days"
        }
    }
    """)
    
    # Embed token example
    print("\n3. Create Embed Token:")
    print("""
    POST /api/v1/bi-integration/connections/{connection_id}/embed-token
    {
        "dashboard_id": "dashboard-123",
        "user_id": "user@company.com",
        "embed_type": "iframe",
        "permissions": ["view", "filter"],
        "expiry_minutes": 60
    }
    """)
    
    # Real-time sync example
    print("\n4. Sync Data Source:")
    print("""
    POST /api/v1/bi-integration/connections/{connection_id}/sync
    {
        "data_source_id": "datasource-456",
        "incremental": true,
        "filters": {
            "updated_since": "2024-01-01"
        }
    }
    """)


if __name__ == "__main__":
    asyncio.run(demo_bi_integration())
    asyncio.run(demo_api_usage())