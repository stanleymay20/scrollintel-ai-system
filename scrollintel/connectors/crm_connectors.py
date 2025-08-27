"""
CRM system connectors for Salesforce, HubSpot, and Microsoft CRM.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import aiohttp
import json

from ..core.data_connector import (
    BaseDataConnector, DataRecord, ConnectionStatus,
    ConnectorError, ConnectionError, AuthenticationError, 
    DataFetchError, TimeoutError, retry_on_failure
)

logger = logging.getLogger(__name__)


class SalesforceConnector(BaseDataConnector):
    """Connector for Salesforce CRM"""
    
    def get_required_params(self) -> List[str]:
        """Get required Salesforce connection parameters"""
        return ['instance_url', 'client_id', 'client_secret', 'username', 'password']
    
    async def connect(self) -> bool:
        """Connect to Salesforce"""
        try:
            self.status = ConnectionStatus.CONNECTING
            
            # Salesforce connection parameters
            instance_url = self.config.connection_params.get('instance_url')
            client_id = self.config.connection_params.get('client_id')
            client_secret = self.config.connection_params.get('client_secret')
            username = self.config.connection_params.get('username')
            password = self.config.connection_params.get('password')
            security_token = self.config.connection_params.get('security_token')
            
            if not all([instance_url, client_id, client_secret, username, password]):
                raise ValueError("Missing required Salesforce connection parameters")
            
            # Simulate OAuth authentication
            await asyncio.sleep(1)
            
            self.status = ConnectionStatus.CONNECTED
            logger.info(f"Connected to Salesforce: {instance_url}")
            return True
            
        except Exception as e:
            self.status = ConnectionStatus.ERROR
            self.error_message = str(e)
            logger.error(f"Salesforce connection failed: {e}")
            return False
    
    async def disconnect(self) -> bool:
        """Disconnect from Salesforce"""
        try:
            self.status = ConnectionStatus.DISCONNECTED
            logger.info("Disconnected from Salesforce")
            return True
        except Exception as e:
            logger.error(f"Salesforce disconnect failed: {e}")
            return False
    
    async def test_connection(self) -> bool:
        """Test Salesforce connection"""
        try:
            await asyncio.sleep(0.5)
            return self.status == ConnectionStatus.CONNECTED
        except Exception:
            return False
    
    async def fetch_data(self, query: Dict[str, Any]) -> List[DataRecord]:
        """Fetch data from Salesforce using SOQL"""
        if self.status != ConnectionStatus.CONNECTED:
            raise ConnectionError("Not connected to Salesforce")
        
        try:
            sobject = query.get('sobject', 'Account')
            fields = query.get('fields', ['Id', 'Name'])
            where_clause = query.get('where', '')
            limit = query.get('limit', 1000)
            
            # Simulate SOQL query
            await asyncio.sleep(1.5)
            
            # Mock Salesforce data
            mock_data = []
            for i in range(min(limit, 100)):
                if sobject == 'Account':
                    data = {
                        'Id': f'001{i:015d}',
                        'Name': f'Salesforce Account {i}',
                        'Type': 'Customer',
                        'Industry': 'Technology',
                        'AnnualRevenue': 1000000 + i * 50000,
                        'NumberOfEmployees': 100 + i * 10,
                        'CreatedDate': datetime.utcnow().isoformat(),
                        'LastModifiedDate': datetime.utcnow().isoformat()
                    }
                elif sobject == 'Opportunity':
                    data = {
                        'Id': f'006{i:015d}',
                        'Name': f'Opportunity {i}',
                        'StageName': 'Prospecting',
                        'Amount': 50000 + i * 5000,
                        'CloseDate': datetime.utcnow().date().isoformat(),
                        'Probability': 25 + (i % 4) * 25,
                        'CreatedDate': datetime.utcnow().isoformat()
                    }
                else:
                    data = {'Id': f'{i:018d}', 'Name': f'{sobject} {i}'}
                
                record = DataRecord(
                    source_id=self.config.source_id,
                    record_id=f"SF_{sobject}_{data['Id']}",
                    data=data,
                    timestamp=datetime.utcnow(),
                    metadata={
                        'sobject': sobject,
                        'salesforce_instance': self.config.connection_params.get('instance_url'),
                        'fields_requested': fields,
                        'where_clause': where_clause
                    }
                )
                mock_data.append(record)
            
            logger.info(f"Fetched {len(mock_data)} records from Salesforce {sobject}")
            return mock_data
            
        except Exception as e:
            logger.error(f"Salesforce data fetch failed: {e}")
            raise
    
    async def get_schema(self) -> Dict[str, Any]:
        """Get Salesforce object schemas"""
        return {
            'sobjects': {
                'Account': {
                    'description': 'Account Object',
                    'fields': {
                        'Id': {'type': 'id', 'description': 'Account ID'},
                        'Name': {'type': 'string', 'length': 255, 'description': 'Account Name'},
                        'Type': {'type': 'picklist', 'description': 'Account Type'},
                        'Industry': {'type': 'picklist', 'description': 'Industry'},
                        'AnnualRevenue': {'type': 'currency', 'description': 'Annual Revenue'}
                    }
                },
                'Opportunity': {
                    'description': 'Opportunity Object',
                    'fields': {
                        'Id': {'type': 'id', 'description': 'Opportunity ID'},
                        'Name': {'type': 'string', 'length': 120, 'description': 'Opportunity Name'},
                        'StageName': {'type': 'picklist', 'description': 'Stage'},
                        'Amount': {'type': 'currency', 'description': 'Amount'},
                        'CloseDate': {'type': 'date', 'description': 'Close Date'}
                    }
                },
                'Contact': {
                    'description': 'Contact Object',
                    'fields': {
                        'Id': {'type': 'id', 'description': 'Contact ID'},
                        'FirstName': {'type': 'string', 'length': 40, 'description': 'First Name'},
                        'LastName': {'type': 'string', 'length': 80, 'description': 'Last Name'},
                        'Email': {'type': 'email', 'description': 'Email Address'}
                    }
                }
            }
        }


class HubSpotConnector(BaseDataConnector):
    """Connector for HubSpot CRM"""
    
    def get_required_params(self) -> List[str]:
        """Get required HubSpot connection parameters"""
        # Either api_key or access_token is required
        return []  # Custom validation in connect method
    
    async def validate_connection_params(self) -> bool:
        """Custom validation for HubSpot parameters"""
        api_key = self.config.connection_params.get('api_key')
        access_token = self.config.connection_params.get('access_token')
        
        if not (api_key or access_token):
            raise ValueError("Either api_key or access_token is required for HubSpot")
        
        return True
    
    async def connect(self) -> bool:
        """Connect to HubSpot"""
        try:
            self.status = ConnectionStatus.CONNECTING
            
            # HubSpot connection parameters
            api_key = self.config.connection_params.get('api_key')
            access_token = self.config.connection_params.get('access_token')
            
            if not (api_key or access_token):
                raise ValueError("Missing HubSpot API key or access token")
            
            # Simulate API authentication
            await asyncio.sleep(1)
            
            self.status = ConnectionStatus.CONNECTED
            logger.info("Connected to HubSpot CRM")
            return True
            
        except Exception as e:
            self.status = ConnectionStatus.ERROR
            self.error_message = str(e)
            logger.error(f"HubSpot connection failed: {e}")
            return False
    
    async def disconnect(self) -> bool:
        """Disconnect from HubSpot"""
        try:
            self.status = ConnectionStatus.DISCONNECTED
            logger.info("Disconnected from HubSpot")
            return True
        except Exception as e:
            logger.error(f"HubSpot disconnect failed: {e}")
            return False
    
    async def test_connection(self) -> bool:
        """Test HubSpot connection"""
        try:
            await asyncio.sleep(0.5)
            return self.status == ConnectionStatus.CONNECTED
        except Exception:
            return False
    
    async def fetch_data(self, query: Dict[str, Any]) -> List[DataRecord]:
        """Fetch data from HubSpot"""
        if self.status != ConnectionStatus.CONNECTED:
            raise ConnectionError("Not connected to HubSpot")
        
        try:
            object_type = query.get('object_type', 'companies')
            properties = query.get('properties', [])
            filters = query.get('filters', [])
            limit = query.get('limit', 1000)
            
            # Simulate HubSpot API call
            await asyncio.sleep(1.2)
            
            # Mock HubSpot data
            mock_data = []
            for i in range(min(limit, 80)):
                if object_type == 'companies':
                    data = {
                        'id': str(1000 + i),
                        'properties': {
                            'name': f'HubSpot Company {i}',
                            'domain': f'company{i}.com',
                            'industry': 'Technology',
                            'annualrevenue': str(500000 + i * 25000),
                            'numberofemployees': str(50 + i * 5),
                            'createdate': datetime.utcnow().isoformat(),
                            'hs_lastmodifieddate': datetime.utcnow().isoformat()
                        }
                    }
                elif object_type == 'deals':
                    data = {
                        'id': str(2000 + i),
                        'properties': {
                            'dealname': f'HubSpot Deal {i}',
                            'amount': str(25000 + i * 2500),
                            'dealstage': 'appointmentscheduled',
                            'pipeline': 'default',
                            'closedate': datetime.utcnow().date().isoformat(),
                            'createdate': datetime.utcnow().isoformat()
                        }
                    }
                elif object_type == 'contacts':
                    data = {
                        'id': str(3000 + i),
                        'properties': {
                            'firstname': f'Contact{i}',
                            'lastname': f'LastName{i}',
                            'email': f'contact{i}@example.com',
                            'jobtitle': 'Manager',
                            'createdate': datetime.utcnow().isoformat()
                        }
                    }
                else:
                    data = {'id': str(i), 'properties': {}}
                
                record = DataRecord(
                    source_id=self.config.source_id,
                    record_id=f"HS_{object_type.upper()}_{data['id']}",
                    data=data,
                    timestamp=datetime.utcnow(),
                    metadata={
                        'object_type': object_type,
                        'properties_requested': properties,
                        'filters_applied': filters
                    }
                )
                mock_data.append(record)
            
            logger.info(f"Fetched {len(mock_data)} records from HubSpot {object_type}")
            return mock_data
            
        except Exception as e:
            logger.error(f"HubSpot data fetch failed: {e}")
            raise
    
    async def get_schema(self) -> Dict[str, Any]:
        """Get HubSpot object schemas"""
        return {
            'objects': {
                'companies': {
                    'description': 'Company Object',
                    'properties': {
                        'name': {'type': 'string', 'description': 'Company Name'},
                        'domain': {'type': 'string', 'description': 'Company Domain'},
                        'industry': {'type': 'enumeration', 'description': 'Industry'},
                        'annualrevenue': {'type': 'number', 'description': 'Annual Revenue'},
                        'numberofemployees': {'type': 'number', 'description': 'Number of Employees'}
                    }
                },
                'deals': {
                    'description': 'Deal Object',
                    'properties': {
                        'dealname': {'type': 'string', 'description': 'Deal Name'},
                        'amount': {'type': 'number', 'description': 'Deal Amount'},
                        'dealstage': {'type': 'enumeration', 'description': 'Deal Stage'},
                        'closedate': {'type': 'date', 'description': 'Close Date'}
                    }
                },
                'contacts': {
                    'description': 'Contact Object',
                    'properties': {
                        'firstname': {'type': 'string', 'description': 'First Name'},
                        'lastname': {'type': 'string', 'description': 'Last Name'},
                        'email': {'type': 'string', 'description': 'Email Address'},
                        'jobtitle': {'type': 'string', 'description': 'Job Title'}
                    }
                }
            }
        }


class MicrosoftCRMConnector(BaseDataConnector):
    """Connector for Microsoft Dynamics 365 CRM"""
    
    def get_required_params(self) -> List[str]:
        """Get required Microsoft CRM connection parameters"""
        return ['org_url', 'client_id', 'client_secret', 'tenant_id']
    
    async def connect(self) -> bool:
        """Connect to Microsoft CRM"""
        try:
            self.status = ConnectionStatus.CONNECTING
            
            # Microsoft CRM connection parameters
            org_url = self.config.connection_params.get('org_url')
            client_id = self.config.connection_params.get('client_id')
            client_secret = self.config.connection_params.get('client_secret')
            tenant_id = self.config.connection_params.get('tenant_id')
            
            if not all([org_url, client_id, client_secret, tenant_id]):
                raise ValueError("Missing required Microsoft CRM connection parameters")
            
            # Simulate OAuth authentication
            await asyncio.sleep(1)
            
            self.status = ConnectionStatus.CONNECTED
            logger.info(f"Connected to Microsoft CRM: {org_url}")
            return True
            
        except Exception as e:
            self.status = ConnectionStatus.ERROR
            self.error_message = str(e)
            logger.error(f"Microsoft CRM connection failed: {e}")
            return False
    
    async def disconnect(self) -> bool:
        """Disconnect from Microsoft CRM"""
        try:
            self.status = ConnectionStatus.DISCONNECTED
            logger.info("Disconnected from Microsoft CRM")
            return True
        except Exception as e:
            logger.error(f"Microsoft CRM disconnect failed: {e}")
            return False
    
    async def test_connection(self) -> bool:
        """Test Microsoft CRM connection"""
        try:
            await asyncio.sleep(0.5)
            return self.status == ConnectionStatus.CONNECTED
        except Exception:
            return False
    
    async def fetch_data(self, query: Dict[str, Any]) -> List[DataRecord]:
        """Fetch data from Microsoft CRM"""
        if self.status != ConnectionStatus.CONNECTED:
            raise ConnectionError("Not connected to Microsoft CRM")
        
        try:
            entity = query.get('entity', 'accounts')
            select_fields = query.get('select', [])
            filter_query = query.get('filter', '')
            limit = query.get('limit', 1000)
            
            # Simulate CRM Web API call
            await asyncio.sleep(1.3)
            
            # Mock Microsoft CRM data
            mock_data = []
            for i in range(min(limit, 90)):
                if entity == 'accounts':
                    data = {
                        'accountid': f'{i + 4000:08d}-{i:04d}-{i:04d}-{i:04d}-{i:012d}',
                        'name': f'Microsoft CRM Account {i}',
                        'accountnumber': f'CRM-{i:06d}',
                        'revenue': 750000 + i * 30000,
                        'numberofemployees': 75 + i * 8,
                        'industrycode': 1,  # Technology
                        'createdon': datetime.utcnow().isoformat(),
                        'modifiedon': datetime.utcnow().isoformat()
                    }
                elif entity == 'opportunities':
                    data = {
                        'opportunityid': f'{i + 5000:08d}-{i:04d}-{i:04d}-{i:04d}-{i:012d}',
                        'name': f'CRM Opportunity {i}',
                        'estimatedvalue': 35000 + i * 3500,
                        'salesstage': 1,  # Qualify
                        'estimatedclosedate': datetime.utcnow().date().isoformat(),
                        'closeprobability': 25 + (i % 4) * 25,
                        'createdon': datetime.utcnow().isoformat()
                    }
                elif entity == 'contacts':
                    data = {
                        'contactid': f'{i + 6000:08d}-{i:04d}-{i:04d}-{i:04d}-{i:012d}',
                        'firstname': f'Contact{i}',
                        'lastname': f'LastName{i}',
                        'emailaddress1': f'contact{i}@crm.com',
                        'jobtitle': 'Senior Manager',
                        'createdon': datetime.utcnow().isoformat()
                    }
                else:
                    data = {'id': f'{i:08d}-{i:04d}-{i:04d}-{i:04d}-{i:012d}'}
                
                record = DataRecord(
                    source_id=self.config.source_id,
                    record_id=f"MSCRM_{entity.upper()}_{i}",
                    data=data,
                    timestamp=datetime.utcnow(),
                    metadata={
                        'entity': entity,
                        'crm_org': self.config.connection_params.get('org_url'),
                        'select_fields': select_fields,
                        'filter_applied': filter_query
                    }
                )
                mock_data.append(record)
            
            logger.info(f"Fetched {len(mock_data)} records from Microsoft CRM entity {entity}")
            return mock_data
            
        except Exception as e:
            logger.error(f"Microsoft CRM data fetch failed: {e}")
            raise
    
    async def get_schema(self) -> Dict[str, Any]:
        """Get Microsoft CRM schema"""
        return {
            'entities': {
                'accounts': {
                    'description': 'Account Entity',
                    'fields': {
                        'accountid': {'type': 'Edm.Guid', 'description': 'Account ID'},
                        'name': {'type': 'Edm.String', 'description': 'Account Name'},
                        'accountnumber': {'type': 'Edm.String', 'description': 'Account Number'},
                        'revenue': {'type': 'Edm.Money', 'description': 'Annual Revenue'},
                        'numberofemployees': {'type': 'Edm.Int32', 'description': 'Number of Employees'}
                    }
                },
                'opportunities': {
                    'description': 'Opportunity Entity',
                    'fields': {
                        'opportunityid': {'type': 'Edm.Guid', 'description': 'Opportunity ID'},
                        'name': {'type': 'Edm.String', 'description': 'Topic'},
                        'estimatedvalue': {'type': 'Edm.Money', 'description': 'Est. Revenue'},
                        'salesstage': {'type': 'Edm.Int32', 'description': 'Sales Stage'},
                        'estimatedclosedate': {'type': 'Edm.DateTimeOffset', 'description': 'Est. Close Date'}
                    }
                },
                'contacts': {
                    'description': 'Contact Entity',
                    'fields': {
                        'contactid': {'type': 'Edm.Guid', 'description': 'Contact ID'},
                        'firstname': {'type': 'Edm.String', 'description': 'First Name'},
                        'lastname': {'type': 'Edm.String', 'description': 'Last Name'},
                        'emailaddress1': {'type': 'Edm.String', 'description': 'Email'}
                    }
                }
            }
        }