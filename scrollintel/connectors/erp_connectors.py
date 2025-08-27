"""
ERP system connectors for SAP, Oracle, and Microsoft Dynamics.
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


class SAPConnector(BaseDataConnector):
    """Connector for SAP ERP systems"""
    
    def get_required_params(self) -> List[str]:
        """Get required SAP connection parameters"""
        return ['host', 'client', 'username', 'password']
    
    async def connect(self) -> bool:
        """Connect to SAP system"""
        try:
            await self.validate_connection_params()
            self.status = ConnectionStatus.CONNECTING
            
            # SAP connection parameters
            host = self.config.connection_params.get('host')
            client = self.config.connection_params.get('client')
            username = self.config.connection_params.get('username')
            password = self.config.connection_params.get('password')
            
            # Simulate SAP RFC connection with potential failures
            await asyncio.sleep(1)  # Simulate connection time
            
            # Simulate occasional connection failures for testing
            import random
            if random.random() < 0.1:  # 10% failure rate for testing
                raise ConnectionError("SAP system temporarily unavailable")
            
            self.status = ConnectionStatus.CONNECTED
            logger.info(f"Connected to SAP system: {host}")
            return True
            
        except (ValueError, ConnectionError) as e:
            self.status = ConnectionStatus.ERROR
            self.error_message = str(e)
            logger.error(f"SAP connection failed: {e}")
            raise
        except Exception as e:
            self.status = ConnectionStatus.ERROR
            self.error_message = str(e)
            logger.error(f"SAP connection failed: {e}")
            raise ConnectionError(f"SAP connection failed: {e}")
    
    async def disconnect(self) -> bool:
        """Disconnect from SAP system"""
        try:
            self.status = ConnectionStatus.DISCONNECTED
            logger.info("Disconnected from SAP system")
            return True
        except Exception as e:
            logger.error(f"SAP disconnect failed: {e}")
            return False
    
    async def test_connection(self) -> bool:
        """Test SAP connection"""
        try:
            # Simulate connection test
            await asyncio.sleep(0.5)
            return self.status == ConnectionStatus.CONNECTED
        except Exception:
            return False
    
    async def fetch_data(self, query: Dict[str, Any]) -> List[DataRecord]:
        """Fetch data from SAP tables"""
        if self.status != ConnectionStatus.CONNECTED:
            raise ConnectionError("Not connected to SAP system")
        
        try:
            table_name = query.get('table', 'MARA')  # Material master
            fields = query.get('fields', ['*'])
            where_clause = query.get('where', {})
            limit = query.get('limit', 1000)
            
            # Validate query parameters
            if limit > 10000:
                raise ValueError("Query limit exceeds maximum allowed (10000)")
            
            # Simulate SAP data fetch with potential failures
            await asyncio.sleep(2)  # Simulate query time
            
            # Simulate occasional data fetch failures for testing
            import random
            if random.random() < 0.05:  # 5% failure rate for testing
                raise DataFetchError("SAP table temporarily locked")
            
            # Mock SAP data
            mock_data = []
            for i in range(min(limit, 100)):  # Simulate up to 100 records
                record = DataRecord(
                    source_id=self.config.source_id,
                    record_id=f"SAP_{table_name}_{i}",
                    data={
                        'MATNR': f'MAT{i:06d}',  # Material number
                        'MAKTX': f'Material Description {i}',
                        'MTART': 'FERT',  # Material type
                        'MEINS': 'EA',    # Base unit
                        'CREATED_DATE': datetime.utcnow().isoformat(),
                        'PRICE': 100.0 + i * 10
                    },
                    timestamp=datetime.utcnow(),
                    metadata={
                        'table': table_name,
                        'sap_client': self.config.connection_params.get('client'),
                        'query_fields': fields
                    }
                )
                mock_data.append(record)
            
            logger.info(f"Fetched {len(mock_data)} records from SAP table {table_name}")
            return mock_data
            
        except (ValueError, DataFetchError) as e:
            logger.error(f"SAP data fetch failed: {e}")
            raise
        except Exception as e:
            logger.error(f"SAP data fetch failed: {e}")
            raise DataFetchError(f"SAP data fetch failed: {e}")
    
    async def get_schema(self) -> Dict[str, Any]:
        """Get SAP table schemas"""
        return {
            'tables': {
                'MARA': {
                    'description': 'General Material Data',
                    'fields': {
                        'MATNR': {'type': 'CHAR', 'length': 18, 'description': 'Material Number'},
                        'MAKTX': {'type': 'CHAR', 'length': 40, 'description': 'Material Description'},
                        'MTART': {'type': 'CHAR', 'length': 4, 'description': 'Material Type'},
                        'MEINS': {'type': 'UNIT', 'length': 3, 'description': 'Base Unit of Measure'}
                    }
                },
                'VBAK': {
                    'description': 'Sales Document Header Data',
                    'fields': {
                        'VBELN': {'type': 'CHAR', 'length': 10, 'description': 'Sales Document'},
                        'KUNNR': {'type': 'CHAR', 'length': 10, 'description': 'Customer Number'},
                        'NETWR': {'type': 'CURR', 'length': 15, 'description': 'Net Value'}
                    }
                }
            }
        }


class OracleERPConnector(BaseDataConnector):
    """Connector for Oracle ERP Cloud"""
    
    def get_required_params(self) -> List[str]:
        """Get required Oracle ERP connection parameters"""
        return ['base_url', 'username', 'password']
    
    async def connect(self) -> bool:
        """Connect to Oracle ERP"""
        try:
            await self.validate_connection_params()
            self.status = ConnectionStatus.CONNECTING
            
            # Oracle ERP connection parameters
            base_url = self.config.connection_params.get('base_url')
            username = self.config.connection_params.get('username')
            password = self.config.connection_params.get('password')
            
            # Simulate Oracle authentication with potential failures
            await asyncio.sleep(1)
            
            # Simulate occasional authentication failures for testing
            import random
            if random.random() < 0.08:  # 8% failure rate for testing
                raise AuthenticationError("Oracle ERP authentication failed")
            
            self.status = ConnectionStatus.CONNECTED
            logger.info(f"Connected to Oracle ERP: {base_url}")
            return True
            
        except (ValueError, AuthenticationError) as e:
            self.status = ConnectionStatus.ERROR
            self.error_message = str(e)
            logger.error(f"Oracle ERP connection failed: {e}")
            raise
        except Exception as e:
            self.status = ConnectionStatus.ERROR
            self.error_message = str(e)
            logger.error(f"Oracle ERP connection failed: {e}")
            raise ConnectionError(f"Oracle ERP connection failed: {e}")
    
    async def disconnect(self) -> bool:
        """Disconnect from Oracle ERP"""
        try:
            self.status = ConnectionStatus.DISCONNECTED
            logger.info("Disconnected from Oracle ERP")
            return True
        except Exception as e:
            logger.error(f"Oracle ERP disconnect failed: {e}")
            return False
    
    async def test_connection(self) -> bool:
        """Test Oracle ERP connection"""
        try:
            await asyncio.sleep(0.5)
            return self.status == ConnectionStatus.CONNECTED
        except Exception:
            return False
    
    async def fetch_data(self, query: Dict[str, Any]) -> List[DataRecord]:
        """Fetch data from Oracle ERP"""
        if self.status != ConnectionStatus.CONNECTED:
            raise ConnectionError("Not connected to Oracle ERP")
        
        try:
            resource = query.get('resource', 'items')
            filters = query.get('filters', {})
            limit = query.get('limit', 1000)
            
            # Simulate Oracle REST API call
            await asyncio.sleep(1.5)
            
            # Mock Oracle ERP data
            mock_data = []
            for i in range(min(limit, 50)):
                record = DataRecord(
                    source_id=self.config.source_id,
                    record_id=f"ORACLE_{resource.upper()}_{i}",
                    data={
                        'ItemId': i + 1000,
                        'ItemNumber': f'ITEM-{i:05d}',
                        'Description': f'Oracle Item Description {i}',
                        'UnitOfMeasure': 'Each',
                        'ListPrice': 250.0 + i * 15,
                        'CreationDate': datetime.utcnow().isoformat(),
                        'LastUpdateDate': datetime.utcnow().isoformat()
                    },
                    timestamp=datetime.utcnow(),
                    metadata={
                        'resource': resource,
                        'oracle_instance': self.config.connection_params.get('base_url'),
                        'filters_applied': filters
                    }
                )
                mock_data.append(record)
            
            logger.info(f"Fetched {len(mock_data)} records from Oracle ERP resource {resource}")
            return mock_data
            
        except Exception as e:
            logger.error(f"Oracle ERP data fetch failed: {e}")
            raise
    
    async def get_schema(self) -> Dict[str, Any]:
        """Get Oracle ERP schema"""
        return {
            'resources': {
                'items': {
                    'description': 'Item Master Data',
                    'fields': {
                        'ItemId': {'type': 'NUMBER', 'description': 'Item Identifier'},
                        'ItemNumber': {'type': 'VARCHAR2', 'length': 300, 'description': 'Item Number'},
                        'Description': {'type': 'VARCHAR2', 'length': 240, 'description': 'Item Description'},
                        'UnitOfMeasure': {'type': 'VARCHAR2', 'length': 25, 'description': 'Primary Unit of Measure'}
                    }
                },
                'purchaseOrders': {
                    'description': 'Purchase Order Data',
                    'fields': {
                        'PoHeaderId': {'type': 'NUMBER', 'description': 'PO Header ID'},
                        'PoNumber': {'type': 'VARCHAR2', 'length': 30, 'description': 'PO Number'},
                        'SupplierId': {'type': 'NUMBER', 'description': 'Supplier ID'},
                        'TotalAmount': {'type': 'NUMBER', 'description': 'Total PO Amount'}
                    }
                }
            }
        }


class MicrosoftDynamicsConnector(BaseDataConnector):
    """Connector for Microsoft Dynamics 365"""
    
    def get_required_params(self) -> List[str]:
        """Get required Dynamics 365 connection parameters"""
        return ['org_url', 'client_id', 'client_secret', 'tenant_id']
    
    async def connect(self) -> bool:
        """Connect to Microsoft Dynamics 365"""
        try:
            self.status = ConnectionStatus.CONNECTING
            
            # Dynamics 365 connection parameters
            org_url = self.config.connection_params.get('org_url')
            client_id = self.config.connection_params.get('client_id')
            client_secret = self.config.connection_params.get('client_secret')
            tenant_id = self.config.connection_params.get('tenant_id')
            
            if not all([org_url, client_id, client_secret, tenant_id]):
                raise ValueError("Missing required Dynamics 365 connection parameters")
            
            # Simulate OAuth authentication
            await asyncio.sleep(1)
            
            self.status = ConnectionStatus.CONNECTED
            logger.info(f"Connected to Dynamics 365: {org_url}")
            return True
            
        except Exception as e:
            self.status = ConnectionStatus.ERROR
            self.error_message = str(e)
            logger.error(f"Dynamics 365 connection failed: {e}")
            return False
    
    async def disconnect(self) -> bool:
        """Disconnect from Dynamics 365"""
        try:
            self.status = ConnectionStatus.DISCONNECTED
            logger.info("Disconnected from Dynamics 365")
            return True
        except Exception as e:
            logger.error(f"Dynamics 365 disconnect failed: {e}")
            return False
    
    async def test_connection(self) -> bool:
        """Test Dynamics 365 connection"""
        try:
            await asyncio.sleep(0.5)
            return self.status == ConnectionStatus.CONNECTED
        except Exception:
            return False
    
    async def fetch_data(self, query: Dict[str, Any]) -> List[DataRecord]:
        """Fetch data from Dynamics 365"""
        if self.status != ConnectionStatus.CONNECTED:
            raise ConnectionError("Not connected to Dynamics 365")
        
        try:
            entity = query.get('entity', 'products')
            select_fields = query.get('select', [])
            filter_query = query.get('filter', '')
            limit = query.get('limit', 1000)
            
            # Simulate Dynamics Web API call
            await asyncio.sleep(1.2)
            
            # Mock Dynamics 365 data
            mock_data = []
            for i in range(min(limit, 75)):
                record = DataRecord(
                    source_id=self.config.source_id,
                    record_id=f"D365_{entity.upper()}_{i}",
                    data={
                        'productid': f'{i + 2000:08d}-{i:04d}-{i:04d}-{i:04d}-{i:012d}',
                        'name': f'Dynamics Product {i}',
                        'productnumber': f'DYN-{i:06d}',
                        'price': 150.0 + i * 8,
                        'quantityonhand': 100 + i * 5,
                        'createdon': datetime.utcnow().isoformat(),
                        'modifiedon': datetime.utcnow().isoformat()
                    },
                    timestamp=datetime.utcnow(),
                    metadata={
                        'entity': entity,
                        'dynamics_org': self.config.connection_params.get('org_url'),
                        'select_fields': select_fields,
                        'filter_applied': filter_query
                    }
                )
                mock_data.append(record)
            
            logger.info(f"Fetched {len(mock_data)} records from Dynamics 365 entity {entity}")
            return mock_data
            
        except Exception as e:
            logger.error(f"Dynamics 365 data fetch failed: {e}")
            raise
    
    async def get_schema(self) -> Dict[str, Any]:
        """Get Dynamics 365 schema"""
        return {
            'entities': {
                'products': {
                    'description': 'Product Entity',
                    'fields': {
                        'productid': {'type': 'Edm.Guid', 'description': 'Product ID'},
                        'name': {'type': 'Edm.String', 'description': 'Product Name'},
                        'productnumber': {'type': 'Edm.String', 'description': 'Product Number'},
                        'price': {'type': 'Edm.Decimal', 'description': 'List Price'},
                        'quantityonhand': {'type': 'Edm.Decimal', 'description': 'Quantity On Hand'}
                    }
                },
                'accounts': {
                    'description': 'Account Entity',
                    'fields': {
                        'accountid': {'type': 'Edm.Guid', 'description': 'Account ID'},
                        'name': {'type': 'Edm.String', 'description': 'Account Name'},
                        'accountnumber': {'type': 'Edm.String', 'description': 'Account Number'},
                        'revenue': {'type': 'Edm.Money', 'description': 'Annual Revenue'}
                    }
                }
            }
        }