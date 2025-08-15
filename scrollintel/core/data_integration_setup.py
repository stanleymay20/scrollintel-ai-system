"""
Setup and registration of all data connectors for the Advanced Analytics Dashboard.
"""

import logging
from typing import Dict, Any

from .data_connector import DataIntegrationManager, DataSourceType, DataSourceConfig
from ..connectors.erp_connectors import SAPConnector, OracleERPConnector, MicrosoftDynamicsConnector
from ..connectors.crm_connectors import SalesforceConnector, HubSpotConnector, MicrosoftCRMConnector
from ..connectors.bi_connectors import TableauConnector, PowerBIConnector, LookerConnector, QlikConnector
from ..connectors.cloud_connectors import AWSConnector, AzureConnector, GCPConnector

logger = logging.getLogger(__name__)


def setup_data_integration() -> DataIntegrationManager:
    """
    Setup and configure the data integration manager with all available connectors.
    
    Returns:
        DataIntegrationManager: Configured integration manager
    """
    manager = DataIntegrationManager()
    
    # Register ERP connectors
    manager.register_connector_class('sap', SAPConnector)
    manager.register_connector_class('oracle_erp', OracleERPConnector)
    manager.register_connector_class('microsoft_dynamics', MicrosoftDynamicsConnector)
    
    # Register CRM connectors
    manager.register_connector_class('salesforce', SalesforceConnector)
    manager.register_connector_class('hubspot', HubSpotConnector)
    manager.register_connector_class('microsoft_crm', MicrosoftCRMConnector)
    
    # Register BI tool connectors
    manager.register_connector_class('tableau', TableauConnector)
    manager.register_connector_class('powerbi', PowerBIConnector)
    manager.register_connector_class('looker', LookerConnector)
    manager.register_connector_class('qlik', QlikConnector)
    
    # Register cloud platform connectors
    manager.register_connector_class('aws', AWSConnector)
    manager.register_connector_class('azure', AzureConnector)
    manager.register_connector_class('gcp', GCPConnector)
    
    logger.info("Data integration manager setup complete with all connectors registered")
    return manager


def create_sample_configurations() -> Dict[str, DataSourceConfig]:
    """
    Create sample configurations for different data sources.
    
    Returns:
        Dict[str, DataSourceConfig]: Sample configurations by source type
    """
    configurations = {}
    
    # SAP Configuration
    configurations['sap'] = DataSourceConfig(
        source_id='sap_prod',
        source_type=DataSourceType.ERP,
        name='SAP Production System',
        connection_params={
            'host': 'sap.company.com',
            'client': '100',
            'username': 'sap_user',
            'password': 'sap_password'
        },
        refresh_interval=1800,  # 30 minutes
        timeout=60,
        retry_attempts=3,
        enabled=True
    )
    
    # Oracle ERP Configuration
    configurations['oracle_erp'] = DataSourceConfig(
        source_id='oracle_erp_prod',
        source_type=DataSourceType.ERP,
        name='Oracle ERP Cloud',
        connection_params={
            'base_url': 'https://company.oraclecloud.com',
            'username': 'oracle_user',
            'password': 'oracle_password'
        },
        refresh_interval=1200,  # 20 minutes
        timeout=45,
        retry_attempts=3,
        enabled=True
    )
    
    # Salesforce Configuration
    configurations['salesforce'] = DataSourceConfig(
        source_id='salesforce_prod',
        source_type=DataSourceType.CRM,
        name='Salesforce Production',
        connection_params={
            'instance_url': 'https://company.salesforce.com',
            'client_id': 'salesforce_client_id',
            'client_secret': 'salesforce_client_secret',
            'username': 'sf_user@company.com',
            'password': 'sf_password',
            'security_token': 'sf_security_token'
        },
        refresh_interval=900,  # 15 minutes
        timeout=30,
        retry_attempts=3,
        enabled=True
    )
    
    # HubSpot Configuration
    configurations['hubspot'] = DataSourceConfig(
        source_id='hubspot_prod',
        source_type=DataSourceType.CRM,
        name='HubSpot CRM',
        connection_params={
            'access_token': 'hubspot_access_token'
        },
        refresh_interval=600,  # 10 minutes
        timeout=30,
        retry_attempts=3,
        enabled=True
    )
    
    # Tableau Configuration
    configurations['tableau'] = DataSourceConfig(
        source_id='tableau_server',
        source_type=DataSourceType.BI_TOOL,
        name='Tableau Server',
        connection_params={
            'server_url': 'https://tableau.company.com',
            'username': 'tableau_user',
            'password': 'tableau_password',
            'site_id': 'default'
        },
        refresh_interval=3600,  # 1 hour
        timeout=60,
        retry_attempts=2,
        enabled=True
    )
    
    # Power BI Configuration
    configurations['powerbi'] = DataSourceConfig(
        source_id='powerbi_tenant',
        source_type=DataSourceType.BI_TOOL,
        name='Power BI Tenant',
        connection_params={
            'client_id': 'powerbi_client_id',
            'client_secret': 'powerbi_client_secret',
            'tenant_id': 'powerbi_tenant_id'
        },
        refresh_interval=2400,  # 40 minutes
        timeout=45,
        retry_attempts=3,
        enabled=True
    )
    
    # AWS Configuration
    configurations['aws'] = DataSourceConfig(
        source_id='aws_main_account',
        source_type=DataSourceType.CLOUD_PLATFORM,
        name='AWS Main Account',
        connection_params={
            'access_key_id': 'aws_access_key_id',
            'secret_access_key': 'aws_secret_access_key',
            'region': 'us-east-1'
        },
        refresh_interval=7200,  # 2 hours
        timeout=120,
        retry_attempts=3,
        enabled=True
    )
    
    # Azure Configuration
    configurations['azure'] = DataSourceConfig(
        source_id='azure_subscription',
        source_type=DataSourceType.CLOUD_PLATFORM,
        name='Azure Subscription',
        connection_params={
            'client_id': 'azure_client_id',
            'client_secret': 'azure_client_secret',
            'tenant_id': 'azure_tenant_id',
            'subscription_id': 'azure_subscription_id'
        },
        refresh_interval=7200,  # 2 hours
        timeout=120,
        retry_attempts=3,
        enabled=True
    )
    
    # GCP Configuration
    configurations['gcp'] = DataSourceConfig(
        source_id='gcp_project',
        source_type=DataSourceType.CLOUD_PLATFORM,
        name='GCP Project',
        connection_params={
            'service_account_key': 'gcp_service_account_key_json',
            'project_id': 'gcp_project_id',
            'billing_account_id': 'gcp_billing_account_id'
        },
        refresh_interval=7200,  # 2 hours
        timeout=120,
        retry_attempts=3,
        enabled=True
    )
    
    return configurations


async def initialize_sample_data_sources(manager: DataIntegrationManager) -> Dict[str, bool]:
    """
    Initialize sample data sources for testing and demonstration.
    
    Args:
        manager: Data integration manager instance
        
    Returns:
        Dict[str, bool]: Results of initialization attempts
    """
    configurations = create_sample_configurations()
    results = {}
    
    for source_name, config in configurations.items():
        try:
            success = await manager.add_data_source(config)
            results[source_name] = success
            
            if success:
                logger.info(f"Successfully initialized {source_name} data source")
            else:
                logger.warning(f"Failed to initialize {source_name} data source")
                
        except Exception as e:
            logger.error(f"Error initializing {source_name}: {e}")
            results[source_name] = False
    
    return results


# Global data integration manager instance
_global_manager = None


def get_data_integration_manager() -> DataIntegrationManager:
    """
    Get the global data integration manager instance.
    
    Returns:
        DataIntegrationManager: Global manager instance
    """
    global _global_manager
    
    if _global_manager is None:
        _global_manager = setup_data_integration()
    
    return _global_manager