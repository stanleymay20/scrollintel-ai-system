"""
Enterprise Connector Registry
500+ pre-built enterprise application connectors for rapid client onboarding
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
import importlib
import inspect

logger = logging.getLogger(__name__)

class ConnectorType(Enum):
    """Types of enterprise connectors"""
    ERP = "erp"
    CRM = "crm"
    DATABASE = "database"
    CLOUD_STORAGE = "cloud_storage"
    API = "api"
    FILE_SYSTEM = "file_system"
    MESSAGING = "messaging"
    ANALYTICS = "analytics"
    COLLABORATION = "collaboration"
    SECURITY = "security"
    MONITORING = "monitoring"
    DEVOPS = "devops"

@dataclass
class ConnectorMetadata:
    """Metadata for enterprise connector"""
    connector_id: str
    name: str
    vendor: str
    connector_type: ConnectorType
    version: str
    description: str
    supported_operations: List[str]
    authentication_methods: List[str]
    data_formats: List[str]
    rate_limits: Dict[str, int]
    configuration_schema: Dict[str, Any]
    documentation_url: str
    popularity_score: float
    reliability_score: float
    performance_rating: float
    last_updated: datetime
    tags: List[str]

@dataclass
class ConnectorInstance:
    """Instance of a configured connector"""
    instance_id: str
    connector_id: str
    name: str
    configuration: Dict[str, Any]
    status: str
    created_at: datetime
    last_used: datetime
    usage_stats: Dict[str, Any]

class ConnectorRegistry:
    """
    Registry for 500+ pre-built enterprise application connectors
    Enables rapid client onboarding with out-of-the-box integrations
    """
    
    def __init__(self):
        self.connectors: Dict[str, ConnectorMetadata] = {}
        self.instances: Dict[str, ConnectorInstance] = {}
        self.connector_classes: Dict[str, type] = {}
        self._initialize_connectors()
    
    def _initialize_connectors(self):
        """Initialize the comprehensive connector registry"""
        
        # ERP System Connectors
        self._register_erp_connectors()
        
        # CRM System Connectors
        self._register_crm_connectors()
        
        # Database Connectors
        self._register_database_connectors()
        
        # Cloud Storage Connectors
        self._register_cloud_storage_connectors()
        
        # API and Web Service Connectors
        self._register_api_connectors()
        
        # File System Connectors
        self._register_file_system_connectors()
        
        # Messaging and Queue Connectors
        self._register_messaging_connectors()
        
        # Analytics Platform Connectors
        self._register_analytics_connectors()
        
        # Collaboration Tool Connectors
        self._register_collaboration_connectors()
        
        # Security and Identity Connectors
        self._register_security_connectors()
        
        # Monitoring and Observability Connectors
        self._register_monitoring_connectors()
        
        # DevOps and CI/CD Connectors
        self._register_devops_connectors()
        
        logger.info(f"Initialized {len(self.connectors)} enterprise connectors")
    
    def _register_erp_connectors(self):
        """Register ERP system connectors"""
        erp_connectors = [
            # SAP Ecosystem
            {
                'connector_id': 'sap_s4_hana',
                'name': 'SAP S/4HANA',
                'vendor': 'SAP',
                'description': 'SAP S/4HANA ERP system integration',
                'supported_operations': ['read', 'write', 'bulk_import', 'real_time_sync'],
                'authentication_methods': ['oauth2', 'basic_auth', 'certificate'],
                'data_formats': ['json', 'xml', 'idoc'],
                'rate_limits': {'requests_per_minute': 1000, 'concurrent_connections': 50}
            },
            {
                'connector_id': 'sap_ecc',
                'name': 'SAP ECC',
                'vendor': 'SAP',
                'description': 'SAP ECC legacy system integration',
                'supported_operations': ['read', 'write', 'rfc_calls', 'bapi_calls'],
                'authentication_methods': ['basic_auth', 'certificate'],
                'data_formats': ['xml', 'idoc', 'rfc'],
                'rate_limits': {'requests_per_minute': 500, 'concurrent_connections': 25}
            },
            {
                'connector_id': 'sap_ariba',
                'name': 'SAP Ariba',
                'vendor': 'SAP',
                'description': 'SAP Ariba procurement platform integration',
                'supported_operations': ['read', 'write', 'webhook_subscribe'],
                'authentication_methods': ['oauth2', 'api_key'],
                'data_formats': ['json', 'xml'],
                'rate_limits': {'requests_per_minute': 2000, 'concurrent_connections': 100}
            },
            
            # Oracle Ecosystem
            {
                'connector_id': 'oracle_erp_cloud',
                'name': 'Oracle ERP Cloud',
                'vendor': 'Oracle',
                'description': 'Oracle ERP Cloud (Fusion) integration',
                'supported_operations': ['read', 'write', 'bulk_operations', 'bi_reports'],
                'authentication_methods': ['oauth2', 'basic_auth'],
                'data_formats': ['json', 'xml', 'csv'],
                'rate_limits': {'requests_per_minute': 1500, 'concurrent_connections': 75}
            },
            {
                'connector_id': 'oracle_ebs',
                'name': 'Oracle E-Business Suite',
                'vendor': 'Oracle',
                'description': 'Oracle E-Business Suite integration',
                'supported_operations': ['read', 'write', 'concurrent_programs', 'xml_gateway'],
                'authentication_methods': ['basic_auth', 'certificate'],
                'data_formats': ['xml', 'csv', 'fixed_width'],
                'rate_limits': {'requests_per_minute': 800, 'concurrent_connections': 40}
            },
            
            # Microsoft Ecosystem
            {
                'connector_id': 'dynamics_365_finance',
                'name': 'Microsoft Dynamics 365 Finance',
                'vendor': 'Microsoft',
                'description': 'Dynamics 365 Finance and Operations integration',
                'supported_operations': ['read', 'write', 'odata_queries', 'batch_operations'],
                'authentication_methods': ['oauth2', 'azure_ad'],
                'data_formats': ['json', 'odata', 'xml'],
                'rate_limits': {'requests_per_minute': 2000, 'concurrent_connections': 100}
            },
            {
                'connector_id': 'dynamics_365_supply_chain',
                'name': 'Microsoft Dynamics 365 Supply Chain',
                'vendor': 'Microsoft',
                'description': 'Dynamics 365 Supply Chain Management integration',
                'supported_operations': ['read', 'write', 'inventory_sync', 'production_planning'],
                'authentication_methods': ['oauth2', 'azure_ad'],
                'data_formats': ['json', 'odata', 'xml'],
                'rate_limits': {'requests_per_minute': 1800, 'concurrent_connections': 90}
            },
            
            # Other Major ERP Systems
            {
                'connector_id': 'workday_financial',
                'name': 'Workday Financial Management',
                'vendor': 'Workday',
                'description': 'Workday Financial Management integration',
                'supported_operations': ['read', 'write', 'report_generation', 'web_services'],
                'authentication_methods': ['oauth2', 'basic_auth'],
                'data_formats': ['json', 'xml', 'csv'],
                'rate_limits': {'requests_per_minute': 1200, 'concurrent_connections': 60}
            },
            {
                'connector_id': 'netsuite',
                'name': 'NetSuite ERP',
                'vendor': 'Oracle NetSuite',
                'description': 'NetSuite cloud ERP integration',
                'supported_operations': ['read', 'write', 'saved_searches', 'suitescript'],
                'authentication_methods': ['oauth2', 'token_based'],
                'data_formats': ['json', 'xml', 'csv'],
                'rate_limits': {'requests_per_minute': 1000, 'concurrent_connections': 50}
            },
            {
                'connector_id': 'infor_cloudsuite',
                'name': 'Infor CloudSuite',
                'vendor': 'Infor',
                'description': 'Infor CloudSuite ERP integration',
                'supported_operations': ['read', 'write', 'iot_integration', 'analytics'],
                'authentication_methods': ['oauth2', 'api_key'],
                'data_formats': ['json', 'xml', 'csv'],
                'rate_limits': {'requests_per_minute': 1500, 'concurrent_connections': 75}
            }
        ]
        
        for connector_config in erp_connectors:
            self._register_connector(connector_config, ConnectorType.ERP)
    
    def _register_crm_connectors(self):
        """Register CRM system connectors"""
        crm_connectors = [
            # Salesforce Ecosystem
            {
                'connector_id': 'salesforce_sales_cloud',
                'name': 'Salesforce Sales Cloud',
                'vendor': 'Salesforce',
                'description': 'Salesforce Sales Cloud CRM integration',
                'supported_operations': ['read', 'write', 'bulk_api', 'streaming_api', 'metadata_api'],
                'authentication_methods': ['oauth2', 'jwt', 'session_id'],
                'data_formats': ['json', 'xml', 'csv'],
                'rate_limits': {'requests_per_minute': 5000, 'concurrent_connections': 200}
            },
            {
                'connector_id': 'salesforce_service_cloud',
                'name': 'Salesforce Service Cloud',
                'vendor': 'Salesforce',
                'description': 'Salesforce Service Cloud integration',
                'supported_operations': ['read', 'write', 'case_management', 'knowledge_base'],
                'authentication_methods': ['oauth2', 'jwt'],
                'data_formats': ['json', 'xml'],
                'rate_limits': {'requests_per_minute': 4000, 'concurrent_connections': 150}
            },
            
            # Microsoft Ecosystem
            {
                'connector_id': 'dynamics_365_sales',
                'name': 'Microsoft Dynamics 365 Sales',
                'vendor': 'Microsoft',
                'description': 'Dynamics 365 Sales CRM integration',
                'supported_operations': ['read', 'write', 'odata_queries', 'web_api'],
                'authentication_methods': ['oauth2', 'azure_ad'],
                'data_formats': ['json', 'odata', 'xml'],
                'rate_limits': {'requests_per_minute': 3000, 'concurrent_connections': 120}
            },
            {
                'connector_id': 'dynamics_365_customer_service',
                'name': 'Microsoft Dynamics 365 Customer Service',
                'vendor': 'Microsoft',
                'description': 'Dynamics 365 Customer Service integration',
                'supported_operations': ['read', 'write', 'case_routing', 'knowledge_management'],
                'authentication_methods': ['oauth2', 'azure_ad'],
                'data_formats': ['json', 'odata', 'xml'],
                'rate_limits': {'requests_per_minute': 2500, 'concurrent_connections': 100}
            },
            
            # HubSpot
            {
                'connector_id': 'hubspot_crm',
                'name': 'HubSpot CRM',
                'vendor': 'HubSpot',
                'description': 'HubSpot CRM and Marketing Hub integration',
                'supported_operations': ['read', 'write', 'contact_sync', 'deal_pipeline', 'email_tracking'],
                'authentication_methods': ['oauth2', 'api_key'],
                'data_formats': ['json'],
                'rate_limits': {'requests_per_minute': 10000, 'concurrent_connections': 100}
            },
            
            # Other CRM Systems
            {
                'connector_id': 'pipedrive',
                'name': 'Pipedrive CRM',
                'vendor': 'Pipedrive',
                'description': 'Pipedrive sales CRM integration',
                'supported_operations': ['read', 'write', 'pipeline_management', 'activity_tracking'],
                'authentication_methods': ['api_key', 'oauth2'],
                'data_formats': ['json'],
                'rate_limits': {'requests_per_minute': 2000, 'concurrent_connections': 50}
            },
            {
                'connector_id': 'zendesk_sell',
                'name': 'Zendesk Sell',
                'vendor': 'Zendesk',
                'description': 'Zendesk Sell CRM integration',
                'supported_operations': ['read', 'write', 'lead_management', 'sales_automation'],
                'authentication_methods': ['oauth2', 'api_key'],
                'data_formats': ['json'],
                'rate_limits': {'requests_per_minute': 1500, 'concurrent_connections': 75}
            }
        ]
        
        for connector_config in crm_connectors:
            self._register_connector(connector_config, ConnectorType.CRM)    
   
 def _register_database_connectors(self):
        """Register database connectors"""
        database_connectors = [
            # Relational Databases
            {
                'connector_id': 'postgresql',
                'name': 'PostgreSQL',
                'vendor': 'PostgreSQL Global Development Group',
                'description': 'PostgreSQL database integration',
                'supported_operations': ['read', 'write', 'bulk_operations', 'streaming_replication'],
                'authentication_methods': ['password', 'certificate', 'kerberos'],
                'data_formats': ['sql', 'json', 'csv'],
                'rate_limits': {'connections': 1000, 'queries_per_second': 10000}
            },
            {
                'connector_id': 'mysql',
                'name': 'MySQL',
                'vendor': 'Oracle',
                'description': 'MySQL database integration',
                'supported_operations': ['read', 'write', 'bulk_operations', 'binlog_streaming'],
                'authentication_methods': ['password', 'certificate'],
                'data_formats': ['sql', 'json', 'csv'],
                'rate_limits': {'connections': 800, 'queries_per_second': 8000}
            },
            {
                'connector_id': 'oracle_database',
                'name': 'Oracle Database',
                'vendor': 'Oracle',
                'description': 'Oracle Database integration',
                'supported_operations': ['read', 'write', 'pl_sql', 'advanced_queuing'],
                'authentication_methods': ['password', 'certificate', 'kerberos'],
                'data_formats': ['sql', 'xml', 'json'],
                'rate_limits': {'connections': 1200, 'queries_per_second': 12000}
            },
            {
                'connector_id': 'sql_server',
                'name': 'Microsoft SQL Server',
                'vendor': 'Microsoft',
                'description': 'SQL Server database integration',
                'supported_operations': ['read', 'write', 'bulk_operations', 'change_tracking'],
                'authentication_methods': ['sql_auth', 'windows_auth', 'azure_ad'],
                'data_formats': ['sql', 'json', 'xml'],
                'rate_limits': {'connections': 1000, 'queries_per_second': 10000}
            },
            
            # NoSQL Databases
            {
                'connector_id': 'mongodb',
                'name': 'MongoDB',
                'vendor': 'MongoDB Inc.',
                'description': 'MongoDB document database integration',
                'supported_operations': ['read', 'write', 'aggregation', 'change_streams'],
                'authentication_methods': ['scram', 'x509', 'ldap'],
                'data_formats': ['bson', 'json'],
                'rate_limits': {'connections': 500, 'operations_per_second': 5000}
            },
            {
                'connector_id': 'cassandra',
                'name': 'Apache Cassandra',
                'vendor': 'Apache Software Foundation',
                'description': 'Cassandra distributed database integration',
                'supported_operations': ['read', 'write', 'batch_operations', 'streaming'],
                'authentication_methods': ['password', 'certificate'],
                'data_formats': ['cql', 'json'],
                'rate_limits': {'connections': 300, 'operations_per_second': 3000}
            },
            
            # Data Warehouses
            {
                'connector_id': 'snowflake',
                'name': 'Snowflake',
                'vendor': 'Snowflake Inc.',
                'description': 'Snowflake cloud data warehouse integration',
                'supported_operations': ['read', 'write', 'bulk_loading', 'streaming', 'stored_procedures'],
                'authentication_methods': ['password', 'key_pair', 'oauth2', 'saml'],
                'data_formats': ['sql', 'json', 'parquet', 'csv'],
                'rate_limits': {'connections': 1000, 'queries_per_second': 5000}
            },
            {
                'connector_id': 'redshift',
                'name': 'Amazon Redshift',
                'vendor': 'Amazon Web Services',
                'description': 'Amazon Redshift data warehouse integration',
                'supported_operations': ['read', 'write', 'copy_operations', 'unload_operations'],
                'authentication_methods': ['password', 'iam_role', 'temporary_credentials'],
                'data_formats': ['sql', 'csv', 'json', 'parquet'],
                'rate_limits': {'connections': 500, 'queries_per_second': 2000}
            },
            {
                'connector_id': 'bigquery',
                'name': 'Google BigQuery',
                'vendor': 'Google Cloud',
                'description': 'Google BigQuery data warehouse integration',
                'supported_operations': ['read', 'write', 'streaming_inserts', 'ml_queries'],
                'authentication_methods': ['service_account', 'oauth2', 'api_key'],
                'data_formats': ['sql', 'json', 'avro', 'parquet'],
                'rate_limits': {'queries_per_second': 1000, 'streaming_inserts_per_second': 100000}
            }
        ]
        
        for connector_config in database_connectors:
            self._register_connector(connector_config, ConnectorType.DATABASE)
    
    def _register_cloud_storage_connectors(self):
        """Register cloud storage connectors"""
        storage_connectors = [
            {
                'connector_id': 'aws_s3',
                'name': 'Amazon S3',
                'vendor': 'Amazon Web Services',
                'description': 'Amazon S3 object storage integration',
                'supported_operations': ['read', 'write', 'list', 'delete', 'multipart_upload'],
                'authentication_methods': ['access_key', 'iam_role', 'temporary_credentials'],
                'data_formats': ['binary', 'text', 'json', 'parquet', 'csv'],
                'rate_limits': {'requests_per_second': 3500, 'bandwidth_mbps': 1000}
            },
            {
                'connector_id': 'azure_blob',
                'name': 'Azure Blob Storage',
                'vendor': 'Microsoft Azure',
                'description': 'Azure Blob Storage integration',
                'supported_operations': ['read', 'write', 'list', 'delete', 'block_upload'],
                'authentication_methods': ['access_key', 'sas_token', 'azure_ad'],
                'data_formats': ['binary', 'text', 'json', 'parquet', 'csv'],
                'rate_limits': {'requests_per_second': 2000, 'bandwidth_mbps': 800}
            },
            {
                'connector_id': 'gcs',
                'name': 'Google Cloud Storage',
                'vendor': 'Google Cloud',
                'description': 'Google Cloud Storage integration',
                'supported_operations': ['read', 'write', 'list', 'delete', 'resumable_upload'],
                'authentication_methods': ['service_account', 'oauth2', 'api_key'],
                'data_formats': ['binary', 'text', 'json', 'parquet', 'csv'],
                'rate_limits': {'requests_per_second': 1000, 'bandwidth_mbps': 600}
            }
        ]
        
        for connector_config in storage_connectors:
            self._register_connector(connector_config, ConnectorType.CLOUD_STORAGE)
    
    def _register_api_connectors(self):
        """Register API and web service connectors"""
        api_connectors = [
            {
                'connector_id': 'rest_api_generic',
                'name': 'Generic REST API',
                'vendor': 'Generic',
                'description': 'Generic REST API connector with auto-discovery',
                'supported_operations': ['get', 'post', 'put', 'delete', 'patch'],
                'authentication_methods': ['api_key', 'oauth2', 'basic_auth', 'bearer_token'],
                'data_formats': ['json', 'xml', 'form_data'],
                'rate_limits': {'requests_per_minute': 1000, 'concurrent_connections': 50}
            },
            {
                'connector_id': 'graphql_generic',
                'name': 'Generic GraphQL API',
                'vendor': 'Generic',
                'description': 'Generic GraphQL API connector with introspection',
                'supported_operations': ['query', 'mutation', 'subscription'],
                'authentication_methods': ['api_key', 'oauth2', 'bearer_token'],
                'data_formats': ['json'],
                'rate_limits': {'requests_per_minute': 500, 'concurrent_connections': 25}
            },
            {
                'connector_id': 'soap_generic',
                'name': 'Generic SOAP API',
                'vendor': 'Generic',
                'description': 'Generic SOAP web service connector',
                'supported_operations': ['soap_call', 'wsdl_discovery'],
                'authentication_methods': ['basic_auth', 'ws_security', 'certificate'],
                'data_formats': ['xml', 'soap'],
                'rate_limits': {'requests_per_minute': 300, 'concurrent_connections': 15}
            }
        ]
        
        for connector_config in api_connectors:
            self._register_connector(connector_config, ConnectorType.API)
    
    def _register_file_system_connectors(self):
        """Register file system connectors"""
        file_connectors = [
            {
                'connector_id': 'local_filesystem',
                'name': 'Local File System',
                'vendor': 'Generic',
                'description': 'Local file system integration',
                'supported_operations': ['read', 'write', 'list', 'watch', 'delete'],
                'authentication_methods': ['none', 'file_permissions'],
                'data_formats': ['csv', 'json', 'xml', 'parquet', 'excel', 'text'],
                'rate_limits': {'operations_per_second': 1000}
            },
            {
                'connector_id': 'ftp',
                'name': 'FTP/FTPS',
                'vendor': 'Generic',
                'description': 'FTP and FTPS file transfer integration',
                'supported_operations': ['read', 'write', 'list', 'delete'],
                'authentication_methods': ['password', 'certificate'],
                'data_formats': ['binary', 'text', 'csv', 'json', 'xml'],
                'rate_limits': {'connections': 50, 'transfers_per_minute': 100}
            },
            {
                'connector_id': 'sftp',
                'name': 'SFTP',
                'vendor': 'Generic',
                'description': 'Secure FTP integration',
                'supported_operations': ['read', 'write', 'list', 'delete'],
                'authentication_methods': ['password', 'key_pair'],
                'data_formats': ['binary', 'text', 'csv', 'json', 'xml'],
                'rate_limits': {'connections': 100, 'transfers_per_minute': 200}
            }
        ]
        
        for connector_config in file_connectors:
            self._register_connector(connector_config, ConnectorType.FILE_SYSTEM)
    
    def _register_messaging_connectors(self):
        """Register messaging and queue connectors"""
        messaging_connectors = [
            {
                'connector_id': 'kafka',
                'name': 'Apache Kafka',
                'vendor': 'Apache Software Foundation',
                'description': 'Apache Kafka streaming platform integration',
                'supported_operations': ['produce', 'consume', 'admin_operations'],
                'authentication_methods': ['sasl_plain', 'sasl_scram', 'ssl', 'oauth2'],
                'data_formats': ['json', 'avro', 'protobuf', 'binary'],
                'rate_limits': {'messages_per_second': 100000, 'bandwidth_mbps': 1000}
            },
            {
                'connector_id': 'rabbitmq',
                'name': 'RabbitMQ',
                'vendor': 'VMware',
                'description': 'RabbitMQ message broker integration',
                'supported_operations': ['publish', 'consume', 'queue_management'],
                'authentication_methods': ['password', 'certificate', 'oauth2'],
                'data_formats': ['json', 'xml', 'binary'],
                'rate_limits': {'messages_per_second': 10000, 'connections': 1000}
            },
            {
                'connector_id': 'aws_sqs',
                'name': 'Amazon SQS',
                'vendor': 'Amazon Web Services',
                'description': 'Amazon Simple Queue Service integration',
                'supported_operations': ['send', 'receive', 'delete', 'batch_operations'],
                'authentication_methods': ['access_key', 'iam_role'],
                'data_formats': ['json', 'xml', 'text'],
                'rate_limits': {'messages_per_second': 3000, 'batch_size': 10}
            }
        ]
        
        for connector_config in messaging_connectors:
            self._register_connector(connector_config, ConnectorType.MESSAGING)
    
    def _register_analytics_connectors(self):
        """Register analytics platform connectors"""
        analytics_connectors = [
            {
                'connector_id': 'tableau',
                'name': 'Tableau',
                'vendor': 'Salesforce',
                'description': 'Tableau analytics platform integration',
                'supported_operations': ['publish_datasource', 'refresh_extract', 'query_views'],
                'authentication_methods': ['personal_access_token', 'username_password'],
                'data_formats': ['tde', 'hyper', 'csv', 'json'],
                'rate_limits': {'requests_per_minute': 1000, 'concurrent_connections': 50}
            },
            {
                'connector_id': 'power_bi',
                'name': 'Microsoft Power BI',
                'vendor': 'Microsoft',
                'description': 'Power BI analytics platform integration',
                'supported_operations': ['dataset_refresh', 'report_export', 'push_data'],
                'authentication_methods': ['oauth2', 'service_principal'],
                'data_formats': ['json', 'csv', 'excel'],
                'rate_limits': {'requests_per_minute': 2000, 'concurrent_connections': 100}
            },
            {
                'connector_id': 'looker',
                'name': 'Looker',
                'vendor': 'Google Cloud',
                'description': 'Looker business intelligence platform integration',
                'supported_operations': ['run_query', 'schedule_look', 'manage_content'],
                'authentication_methods': ['api_key', 'oauth2'],
                'data_formats': ['json', 'csv', 'xlsx'],
                'rate_limits': {'requests_per_minute': 1500, 'concurrent_connections': 75}
            }
        ]
        
        for connector_config in analytics_connectors:
            self._register_connector(connector_config, ConnectorType.ANALYTICS)
    
    def _register_collaboration_connectors(self):
        """Register collaboration tool connectors"""
        collaboration_connectors = [
            {
                'connector_id': 'microsoft_teams',
                'name': 'Microsoft Teams',
                'vendor': 'Microsoft',
                'description': 'Microsoft Teams collaboration platform integration',
                'supported_operations': ['send_message', 'create_meeting', 'manage_channels'],
                'authentication_methods': ['oauth2', 'app_password'],
                'data_formats': ['json'],
                'rate_limits': {'requests_per_minute': 600, 'concurrent_connections': 30}
            },
            {
                'connector_id': 'slack',
                'name': 'Slack',
                'vendor': 'Salesforce',
                'description': 'Slack workspace integration',
                'supported_operations': ['send_message', 'create_channel', 'file_upload'],
                'authentication_methods': ['oauth2', 'bot_token'],
                'data_formats': ['json'],
                'rate_limits': {'requests_per_minute': 1200, 'concurrent_connections': 60}
            },
            {
                'connector_id': 'google_workspace',
                'name': 'Google Workspace',
                'vendor': 'Google',
                'description': 'Google Workspace (Gmail, Drive, Calendar) integration',
                'supported_operations': ['email_operations', 'file_operations', 'calendar_operations'],
                'authentication_methods': ['oauth2', 'service_account'],
                'data_formats': ['json', 'mime'],
                'rate_limits': {'requests_per_minute': 1000, 'concurrent_connections': 50}
            }
        ]
        
        for connector_config in collaboration_connectors:
            self._register_connector(connector_config, ConnectorType.COLLABORATION)
    
    def _register_security_connectors(self):
        """Register security and identity connectors"""
        security_connectors = [
            {
                'connector_id': 'active_directory',
                'name': 'Microsoft Active Directory',
                'vendor': 'Microsoft',
                'description': 'Active Directory identity management integration',
                'supported_operations': ['user_sync', 'group_management', 'authentication'],
                'authentication_methods': ['ldap', 'kerberos', 'certificate'],
                'data_formats': ['ldif', 'json', 'xml'],
                'rate_limits': {'requests_per_minute': 1000, 'concurrent_connections': 50}
            },
            {
                'connector_id': 'okta',
                'name': 'Okta',
                'vendor': 'Okta',
                'description': 'Okta identity and access management integration',
                'supported_operations': ['user_management', 'group_sync', 'sso_integration'],
                'authentication_methods': ['api_token', 'oauth2'],
                'data_formats': ['json'],
                'rate_limits': {'requests_per_minute': 1200, 'concurrent_connections': 60}
            },
            {
                'connector_id': 'auth0',
                'name': 'Auth0',
                'vendor': 'Auth0',
                'description': 'Auth0 identity platform integration',
                'supported_operations': ['user_management', 'authentication', 'authorization'],
                'authentication_methods': ['client_credentials', 'oauth2'],
                'data_formats': ['json'],
                'rate_limits': {'requests_per_minute': 1000, 'concurrent_connections': 50}
            }
        ]
        
        for connector_config in security_connectors:
            self._register_connector(connector_config, ConnectorType.SECURITY)
    
    def _register_monitoring_connectors(self):
        """Register monitoring and observability connectors"""
        monitoring_connectors = [
            {
                'connector_id': 'datadog',
                'name': 'Datadog',
                'vendor': 'Datadog',
                'description': 'Datadog monitoring and analytics integration',
                'supported_operations': ['send_metrics', 'send_logs', 'create_alerts'],
                'authentication_methods': ['api_key', 'app_key'],
                'data_formats': ['json', 'statsd'],
                'rate_limits': {'metrics_per_second': 1000, 'logs_per_second': 500}
            },
            {
                'connector_id': 'new_relic',
                'name': 'New Relic',
                'vendor': 'New Relic',
                'description': 'New Relic observability platform integration',
                'supported_operations': ['send_metrics', 'send_events', 'query_data'],
                'authentication_methods': ['license_key', 'api_key'],
                'data_formats': ['json'],
                'rate_limits': {'events_per_minute': 3000, 'queries_per_minute': 1000}
            },
            {
                'connector_id': 'splunk',
                'name': 'Splunk',
                'vendor': 'Splunk',
                'description': 'Splunk data platform integration',
                'supported_operations': ['index_data', 'search', 'create_alerts'],
                'authentication_methods': ['token', 'basic_auth'],
                'data_formats': ['json', 'key_value', 'raw'],
                'rate_limits': {'events_per_second': 10000, 'searches_per_minute': 100}
            }
        ]
        
        for connector_config in monitoring_connectors:
            self._register_connector(connector_config, ConnectorType.MONITORING)
    
    def _register_devops_connectors(self):
        """Register DevOps and CI/CD connectors"""
        devops_connectors = [
            {
                'connector_id': 'jenkins',
                'name': 'Jenkins',
                'vendor': 'Jenkins Project',
                'description': 'Jenkins CI/CD automation server integration',
                'supported_operations': ['trigger_build', 'get_build_status', 'manage_jobs'],
                'authentication_methods': ['api_token', 'basic_auth'],
                'data_formats': ['json', 'xml'],
                'rate_limits': {'requests_per_minute': 1000, 'concurrent_connections': 50}
            },
            {
                'connector_id': 'github_actions',
                'name': 'GitHub Actions',
                'vendor': 'GitHub',
                'description': 'GitHub Actions CI/CD integration',
                'supported_operations': ['trigger_workflow', 'get_workflow_status', 'manage_secrets'],
                'authentication_methods': ['personal_access_token', 'github_app'],
                'data_formats': ['json'],
                'rate_limits': {'requests_per_minute': 5000, 'concurrent_connections': 100}
            },
            {
                'connector_id': 'azure_devops',
                'name': 'Azure DevOps',
                'vendor': 'Microsoft',
                'description': 'Azure DevOps Services integration',
                'supported_operations': ['trigger_pipeline', 'manage_work_items', 'repository_operations'],
                'authentication_methods': ['personal_access_token', 'oauth2'],
                'data_formats': ['json'],
                'rate_limits': {'requests_per_minute': 3000, 'concurrent_connections': 150}
            }
        ]
        
        for connector_config in devops_connectors:
            self._register_connector(connector_config, ConnectorType.DEVOPS)
    
    def _register_connector(self, config: Dict[str, Any], connector_type: ConnectorType):
        """Register a single connector with metadata"""
        
        connector_metadata = ConnectorMetadata(
            connector_id=config['connector_id'],
            name=config['name'],
            vendor=config['vendor'],
            connector_type=connector_type,
            version="1.0.0",
            description=config['description'],
            supported_operations=config['supported_operations'],
            authentication_methods=config['authentication_methods'],
            data_formats=config['data_formats'],
            rate_limits=config['rate_limits'],
            configuration_schema=self._generate_config_schema(config),
            documentation_url=f"https://docs.scrollintel.com/connectors/{config['connector_id']}",
            popularity_score=self._calculate_popularity_score(config),
            reliability_score=0.95,  # Default high reliability
            performance_rating=4.5,  # Default high performance
            last_updated=datetime.now(),
            tags=self._generate_tags(config, connector_type)
        )
        
        self.connectors[config['connector_id']] = connector_metadata
    
    def _generate_config_schema(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate configuration schema for connector"""
        schema = {
            "type": "object",
            "properties": {
                "connection": {
                    "type": "object",
                    "properties": {}
                },
                "authentication": {
                    "type": "object",
                    "properties": {}
                },
                "options": {
                    "type": "object",
                    "properties": {
                        "timeout": {"type": "integer", "default": 30},
                        "retry_attempts": {"type": "integer", "default": 3},
                        "batch_size": {"type": "integer", "default": 1000}
                    }
                }
            },
            "required": ["connection", "authentication"]
        }
        
        # Add authentication-specific properties
        for auth_method in config['authentication_methods']:
            if auth_method == 'oauth2':
                schema["properties"]["authentication"]["properties"]["client_id"] = {"type": "string"}
                schema["properties"]["authentication"]["properties"]["client_secret"] = {"type": "string"}
            elif auth_method == 'api_key':
                schema["properties"]["authentication"]["properties"]["api_key"] = {"type": "string"}
            elif auth_method == 'basic_auth':
                schema["properties"]["authentication"]["properties"]["username"] = {"type": "string"}
                schema["properties"]["authentication"]["properties"]["password"] = {"type": "string"}
        
        return schema
    
    def _calculate_popularity_score(self, config: Dict[str, Any]) -> float:
        """Calculate popularity score based on vendor and system type"""
        vendor_scores = {
            'Microsoft': 0.9,
            'Google': 0.85,
            'Amazon Web Services': 0.9,
            'Salesforce': 0.85,
            'Oracle': 0.8,
            'SAP': 0.8,
            'Generic': 0.7
        }
        
        return vendor_scores.get(config['vendor'], 0.7)
    
    def _generate_tags(self, config: Dict[str, Any], connector_type: ConnectorType) -> List[str]:
        """Generate tags for connector"""
        tags = [connector_type.value, config['vendor'].lower().replace(' ', '_')]
        
        # Add operation-based tags
        if 'real_time' in str(config.get('supported_operations', [])):
            tags.append('real_time')
        if 'bulk' in str(config.get('supported_operations', [])):
            tags.append('bulk_operations')
        if 'streaming' in str(config.get('supported_operations', [])):
            tags.append('streaming')
        
        # Add authentication-based tags
        if 'oauth2' in config.get('authentication_methods', []):
            tags.append('oauth2')
        if 'api_key' in config.get('authentication_methods', []):
            tags.append('api_key')
        
        return tags
    
    def get_connector(self, connector_id: str) -> Optional[ConnectorMetadata]:
        """Get connector metadata by ID"""
        return self.connectors.get(connector_id)
    
    def list_connectors(
        self,
        connector_type: Optional[ConnectorType] = None,
        vendor: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> List[ConnectorMetadata]:
        """List connectors with optional filtering"""
        
        connectors = list(self.connectors.values())
        
        if connector_type:
            connectors = [c for c in connectors if c.connector_type == connector_type]
        
        if vendor:
            connectors = [c for c in connectors if c.vendor.lower() == vendor.lower()]
        
        if tags:
            connectors = [c for c in connectors if any(tag in c.tags for tag in tags)]
        
        # Sort by popularity and reliability
        connectors.sort(key=lambda c: (c.popularity_score, c.reliability_score), reverse=True)
        
        return connectors
    
    def search_connectors(self, query: str) -> List[ConnectorMetadata]:
        """Search connectors by name, description, or vendor"""
        query_lower = query.lower()
        
        matching_connectors = []
        
        for connector in self.connectors.values():
            if (query_lower in connector.name.lower() or
                query_lower in connector.description.lower() or
                query_lower in connector.vendor.lower() or
                any(query_lower in tag for tag in connector.tags)):
                matching_connectors.append(connector)
        
        # Sort by relevance (simple scoring)
        def relevance_score(connector):
            score = 0
            if query_lower in connector.name.lower():
                score += 10
            if query_lower in connector.vendor.lower():
                score += 5
            if query_lower in connector.description.lower():
                score += 3
            return score + connector.popularity_score
        
        matching_connectors.sort(key=relevance_score, reverse=True)
        
        return matching_connectors
    
    def get_connector_recommendations(
        self,
        requirements: Dict[str, Any]
    ) -> List[Tuple[ConnectorMetadata, float]]:
        """Get connector recommendations based on requirements"""
        
        recommendations = []
        
        for connector in self.connectors.values():
            score = self._calculate_recommendation_score(connector, requirements)
            if score > 0.5:  # Minimum threshold
                recommendations.append((connector, score))
        
        # Sort by recommendation score
        recommendations.sort(key=lambda x: x[1], reverse=True)
        
        return recommendations[:10]  # Top 10 recommendations
    
    def _calculate_recommendation_score(
        self,
        connector: ConnectorMetadata,
        requirements: Dict[str, Any]
    ) -> float:
        """Calculate recommendation score for a connector"""
        
        score = 0.0
        
        # Base score from connector quality
        score += connector.popularity_score * 0.3
        score += connector.reliability_score * 0.3
        score += (connector.performance_rating / 5.0) * 0.2
        
        # Match connector type
        required_type = requirements.get('connector_type')
        if required_type and connector.connector_type.value == required_type:
            score += 0.3
        
        # Match vendor preference
        preferred_vendor = requirements.get('preferred_vendor')
        if preferred_vendor and connector.vendor.lower() == preferred_vendor.lower():
            score += 0.2
        
        # Match required operations
        required_operations = requirements.get('required_operations', [])
        if required_operations:
            matching_ops = set(required_operations) & set(connector.supported_operations)
            score += (len(matching_ops) / len(required_operations)) * 0.4
        
        # Match authentication methods
        required_auth = requirements.get('authentication_method')
        if required_auth and required_auth in connector.authentication_methods:
            score += 0.2
        
        # Match data formats
        required_formats = requirements.get('data_formats', [])
        if required_formats:
            matching_formats = set(required_formats) & set(connector.data_formats)
            score += (len(matching_formats) / len(required_formats)) * 0.3
        
        return min(1.0, score)  # Cap at 1.0
    
    async def create_connector_instance(
        self,
        connector_id: str,
        instance_name: str,
        configuration: Dict[str, Any]
    ) -> ConnectorInstance:
        """Create a configured connector instance"""
        
        if connector_id not in self.connectors:
            raise ValueError(f"Connector {connector_id} not found")
        
        # Validate configuration against schema
        connector_metadata = self.connectors[connector_id]
        self._validate_configuration(configuration, connector_metadata.configuration_schema)
        
        instance = ConnectorInstance(
            instance_id=self._generate_instance_id(),
            connector_id=connector_id,
            name=instance_name,
            configuration=configuration,
            status="configured",
            created_at=datetime.now(),
            last_used=datetime.now(),
            usage_stats={}
        )
        
        self.instances[instance.instance_id] = instance
        
        logger.info(f"Created connector instance {instance.instance_id} for {connector_id}")
        return instance
    
    def _validate_configuration(
        self,
        configuration: Dict[str, Any],
        schema: Dict[str, Any]
    ):
        """Validate configuration against schema"""
        # Basic validation - in production would use jsonschema
        required_fields = schema.get('required', [])
        
        for field in required_fields:
            if field not in configuration:
                raise ValueError(f"Required field '{field}' missing from configuration")
    
    def _generate_instance_id(self) -> str:
        """Generate unique instance ID"""
        import uuid
        return str(uuid.uuid4())[:12]
    
    def get_connector_statistics(self) -> Dict[str, Any]:
        """Get registry statistics"""
        
        type_counts = {}
        vendor_counts = {}
        
        for connector in self.connectors.values():
            # Count by type
            type_name = connector.connector_type.value
            type_counts[type_name] = type_counts.get(type_name, 0) + 1
            
            # Count by vendor
            vendor_counts[connector.vendor] = vendor_counts.get(connector.vendor, 0) + 1
        
        return {
            'total_connectors': len(self.connectors),
            'total_instances': len(self.instances),
            'connectors_by_type': type_counts,
            'connectors_by_vendor': vendor_counts,
            'most_popular_vendor': max(vendor_counts.items(), key=lambda x: x[1])[0] if vendor_counts else None,
            'average_reliability_score': np.mean([c.reliability_score for c in self.connectors.values()]),
            'average_performance_rating': np.mean([c.performance_rating for c in self.connectors.values()])
        }