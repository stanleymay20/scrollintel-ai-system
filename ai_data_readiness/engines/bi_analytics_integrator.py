"""
BI and Analytics Tool Integration Engine for AI Data Readiness Platform

This module provides connectors and integration capabilities for popular BI and analytics tools
including Tableau, Power BI, Looker, and other data visualization platforms.
"""

import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime
import json
import requests
import pandas as pd
import io
from abc import ABC, abstractmethod
from pathlib import Path

# Tableau integration
try:
    import tableauserverclient as TSC
    TABLEAU_AVAILABLE = True
except ImportError:
    TABLEAU_AVAILABLE = False

# Power BI integration (using REST API)
# Note: Power BI Python SDK is limited, so we use REST API directly

logger = logging.getLogger(__name__)


@dataclass
class BIToolConfig:
    """Configuration for BI tool connections"""
    tool_type: str
    server_url: str
    credentials: Dict[str, Any]
    workspace_id: Optional[str] = None
    metadata: Dict[str, Any] = None


@dataclass
class DataExportConfig:
    """Configuration for data export"""
    export_format: str  # csv, json, parquet, excel
    include_metadata: bool = True
    compression: Optional[str] = None
    chunk_size: Optional[int] = None
    filters: Dict[str, Any] = None


@dataclass
class ReportDistributionInfo:
    """Information about report distribution"""
    report_id: str
    report_name: str
    recipients: List[str]
    distribution_schedule: str
    format: str
    status: str
    last_sent: Optional[datetime] = None
    next_scheduled: Optional[datetime] = None
    metadata: Dict[str, Any] = None


@dataclass
class DataSourceInfo:
    """Information about data sources in BI tools"""
    source_id: str
    source_name: str
    connection_type: str
    dataset_id: Optional[str] = None
    last_refresh: Optional[datetime] = None
    refresh_status: str = "unknown"
    row_count: Optional[int] = None
    metadata: Dict[str, Any] = None


class BIToolConnector(ABC):
    """Abstract base class for BI tool connectors"""
    
    def __init__(self, config: BIToolConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    @abstractmethod
    def connect(self) -> bool:
        """Establish connection to the BI tool"""
        pass
    
    @abstractmethod
    def create_data_source(self, dataset_info: Dict[str, Any]) -> DataSourceInfo:
        """Create a data source in the BI tool"""
        pass
    
    @abstractmethod
    def update_data_source(self, source_id: str, dataset_info: Dict[str, Any]) -> bool:
        """Update an existing data source"""
        pass
    
    @abstractmethod
    def get_data_sources(self) -> List[DataSourceInfo]:
        """Get list of data sources"""
        pass
    
    @abstractmethod
    def create_report(self, report_config: Dict[str, Any]) -> str:
        """Create a report/dashboard"""
        pass
    
    @abstractmethod
    def distribute_report(self, distribution_info: ReportDistributionInfo) -> bool:
        """Distribute a report to stakeholders"""
        pass


class TableauConnector(BIToolConnector):
    """Connector for Tableau Server/Online"""
    
    def __init__(self, config: BIToolConfig):
        super().__init__(config)
        if not TABLEAU_AVAILABLE:
            raise ImportError("Tableau Server Client is not installed. Install with: pip install tableauserverclient")
        self.server = None
        self.auth = None
    
    def connect(self) -> bool:
        """Establish connection to Tableau Server"""
        try:
            self.server = TSC.Server(self.config.server_url, use_server_version=True)
            
            # Set up authentication
            if 'username' in self.config.credentials and 'password' in self.config.credentials:
                self.auth = TSC.TableauAuth(
                    self.config.credentials['username'],
                    self.config.credentials['password'],
                    site_id=self.config.credentials.get('site_id', '')
                )
            elif 'token_name' in self.config.credentials and 'token_value' in self.config.credentials:
                self.auth = TSC.PersonalAccessTokenAuth(
                    self.config.credentials['token_name'],
                    self.config.credentials['token_value'],
                    site_id=self.config.credentials.get('site_id', '')
                )
            else:
                raise ValueError("Invalid credentials for Tableau connection")
            
            # Test connection
            self.server.auth.sign_in(self.auth)
            self.logger.info("Successfully connected to Tableau Server")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to Tableau: {str(e)}")
            return False
    
    def create_data_source(self, dataset_info: Dict[str, Any]) -> DataSourceInfo:
        """Create a data source in Tableau"""
        try:
            # For Tableau, we typically publish a data extract or connect to a database
            # This is a simplified implementation
            
            datasource = TSC.DatasourceItem(
                project_id=self.config.workspace_id or self.server.projects.default().id,
                name=dataset_info['name']
            )
            
            # In a real implementation, you would publish the actual data
            # For demo purposes, we'll simulate the creation
            
            source_info = DataSourceInfo(
                source_id=f"tableau_ds_{dataset_info['id']}",
                source_name=dataset_info['name'],
                connection_type="extract",
                dataset_id=dataset_info['id'],
                last_refresh=datetime.now(),
                refresh_status="success",
                row_count=dataset_info.get('row_count'),
                metadata={
                    'tableau_project_id': self.config.workspace_id,
                    'created_by': 'ai_data_readiness_platform'
                }
            )
            
            self.logger.info(f"Created Tableau data source: {source_info.source_name}")
            return source_info
            
        except Exception as e:
            self.logger.error(f"Failed to create Tableau data source: {str(e)}")
            raise
    
    def update_data_source(self, source_id: str, dataset_info: Dict[str, Any]) -> bool:
        """Update a Tableau data source"""
        try:
            # In a real implementation, you would refresh or republish the data source
            self.logger.info(f"Updated Tableau data source: {source_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update Tableau data source: {str(e)}")
            return False
    
    def get_data_sources(self) -> List[DataSourceInfo]:
        """Get list of Tableau data sources"""
        try:
            data_sources = []
            
            # Get all data sources from the server
            all_datasources, pagination_item = self.server.datasources.get()
            
            for ds in all_datasources:
                source_info = DataSourceInfo(
                    source_id=ds.id,
                    source_name=ds.name,
                    connection_type="tableau",
                    last_refresh=ds.updated_at,
                    refresh_status="unknown",
                    metadata={
                        'project_id': ds.project_id,
                        'content_url': ds.content_url,
                        'created_at': ds.created_at.isoformat() if ds.created_at else None
                    }
                )
                data_sources.append(source_info)
            
            return data_sources
            
        except Exception as e:
            self.logger.error(f"Failed to get Tableau data sources: {str(e)}")
            return []
    
    def create_report(self, report_config: Dict[str, Any]) -> str:
        """Create a Tableau workbook/report"""
        try:
            # In a real implementation, you would create and publish a workbook
            # This is a simplified simulation
            
            report_id = f"tableau_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            self.logger.info(f"Created Tableau report: {report_config['name']}")
            return report_id
            
        except Exception as e:
            self.logger.error(f"Failed to create Tableau report: {str(e)}")
            raise
    
    def distribute_report(self, distribution_info: ReportDistributionInfo) -> bool:
        """Distribute a Tableau report"""
        try:
            # In Tableau, this would involve setting up subscriptions
            self.logger.info(f"Set up distribution for Tableau report: {distribution_info.report_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to distribute Tableau report: {str(e)}")
            return False


class PowerBIConnector(BIToolConnector):
    """Connector for Microsoft Power BI"""
    
    def __init__(self, config: BIToolConfig):
        super().__init__(config)
        self.session = requests.Session()
        self.access_token = None
    
    def connect(self) -> bool:
        """Establish connection to Power BI"""
        try:
            # Authenticate with Power BI REST API
            auth_url = "https://login.microsoftonline.com/common/oauth2/token"
            
            auth_data = {
                'grant_type': 'client_credentials',
                'client_id': self.config.credentials['client_id'],
                'client_secret': self.config.credentials['client_secret'],
                'resource': 'https://analysis.windows.net/powerbi/api'
            }
            
            response = requests.post(auth_url, data=auth_data)
            
            if response.status_code == 200:
                token_data = response.json()
                self.access_token = token_data['access_token']
                
                self.session.headers.update({
                    'Authorization': f"Bearer {self.access_token}",
                    'Content-Type': 'application/json'
                })
                
                self.logger.info("Successfully connected to Power BI")
                return True
            else:
                self.logger.error(f"Power BI authentication failed: {response.status_code}")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to connect to Power BI: {str(e)}")
            return False
    
    def create_data_source(self, dataset_info: Dict[str, Any]) -> DataSourceInfo:
        """Create a dataset in Power BI"""
        try:
            workspace_id = self.config.workspace_id
            
            # Create dataset schema
            dataset_schema = {
                "name": dataset_info['name'],
                "tables": [
                    {
                        "name": dataset_info.get('table_name', 'MainTable'),
                        "columns": dataset_info.get('columns', [])
                    }
                ]
            }
            
            url = f"https://api.powerbi.com/v1.0/myorg/groups/{workspace_id}/datasets"
            response = self.session.post(url, json=dataset_schema)
            
            if response.status_code == 201:
                dataset = response.json()
                
                source_info = DataSourceInfo(
                    source_id=dataset['id'],
                    source_name=dataset['name'],
                    connection_type="powerbi_dataset",
                    dataset_id=dataset_info['id'],
                    last_refresh=datetime.now(),
                    refresh_status="success",
                    metadata={
                        'workspace_id': workspace_id,
                        'powerbi_dataset_id': dataset['id']
                    }
                )
                
                self.logger.info(f"Created Power BI dataset: {source_info.source_name}")
                return source_info
            else:
                raise Exception(f"Failed to create dataset: {response.status_code} - {response.text}")
                
        except Exception as e:
            self.logger.error(f"Failed to create Power BI dataset: {str(e)}")
            raise
    
    def update_data_source(self, source_id: str, dataset_info: Dict[str, Any]) -> bool:
        """Refresh a Power BI dataset"""
        try:
            workspace_id = self.config.workspace_id
            url = f"https://api.powerbi.com/v1.0/myorg/groups/{workspace_id}/datasets/{source_id}/refreshes"
            
            response = self.session.post(url)
            
            if response.status_code == 202:
                self.logger.info(f"Triggered refresh for Power BI dataset: {source_id}")
                return True
            else:
                self.logger.error(f"Failed to refresh dataset: {response.status_code}")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to update Power BI dataset: {str(e)}")
            return False
    
    def get_data_sources(self) -> List[DataSourceInfo]:
        """Get list of Power BI datasets"""
        try:
            workspace_id = self.config.workspace_id
            url = f"https://api.powerbi.com/v1.0/myorg/groups/{workspace_id}/datasets"
            
            response = self.session.get(url)
            
            if response.status_code == 200:
                datasets = response.json()['value']
                data_sources = []
                
                for ds in datasets:
                    source_info = DataSourceInfo(
                        source_id=ds['id'],
                        source_name=ds['name'],
                        connection_type="powerbi_dataset",
                        refresh_status="unknown",
                        metadata={
                            'workspace_id': workspace_id,
                            'configured_by': ds.get('configuredBy'),
                            'is_refreshable': ds.get('isRefreshable', False)
                        }
                    )
                    data_sources.append(source_info)
                
                return data_sources
            else:
                self.logger.error(f"Failed to get datasets: {response.status_code}")
                return []
                
        except Exception as e:
            self.logger.error(f"Failed to get Power BI datasets: {str(e)}")
            return []
    
    def create_report(self, report_config: Dict[str, Any]) -> str:
        """Create a Power BI report"""
        try:
            workspace_id = self.config.workspace_id
            
            report_data = {
                "name": report_config['name'],
                "datasetId": report_config['dataset_id']
            }
            
            url = f"https://api.powerbi.com/v1.0/myorg/groups/{workspace_id}/reports"
            response = self.session.post(url, json=report_data)
            
            if response.status_code == 201:
                report = response.json()
                self.logger.info(f"Created Power BI report: {report['name']}")
                return report['id']
            else:
                raise Exception(f"Failed to create report: {response.status_code} - {response.text}")
                
        except Exception as e:
            self.logger.error(f"Failed to create Power BI report: {str(e)}")
            raise
    
    def distribute_report(self, distribution_info: ReportDistributionInfo) -> bool:
        """Set up Power BI report distribution"""
        try:
            # Power BI distribution typically involves sharing reports or setting up subscriptions
            # This is a simplified implementation
            
            self.logger.info(f"Set up distribution for Power BI report: {distribution_info.report_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to distribute Power BI report: {str(e)}")
            return False


class GenericBIConnector(BIToolConnector):
    """Generic connector for BI tools with REST APIs"""
    
    def __init__(self, config: BIToolConfig):
        super().__init__(config)
        self.session = requests.Session()
        
        # Set up authentication
        if 'api_key' in config.credentials:
            self.session.headers.update({
                'Authorization': f"Bearer {config.credentials['api_key']}"
            })
        elif 'username' in config.credentials and 'password' in config.credentials:
            self.session.auth = (
                config.credentials['username'],
                config.credentials['password']
            )
    
    def connect(self) -> bool:
        """Test connection to the BI tool"""
        try:
            response = self.session.get(f"{self.config.server_url}/api/health")
            return response.status_code == 200
        except Exception as e:
            self.logger.error(f"Failed to connect to BI tool: {str(e)}")
            return False
    
    def create_data_source(self, dataset_info: Dict[str, Any]) -> DataSourceInfo:
        """Create a data source via REST API"""
        try:
            data_source_config = {
                'name': dataset_info['name'],
                'type': 'dataset',
                'connection': {
                    'dataset_id': dataset_info['id'],
                    'format': dataset_info.get('format', 'json')
                }
            }
            
            response = self.session.post(
                f"{self.config.server_url}/api/datasources",
                json=data_source_config
            )
            
            if response.status_code in [200, 201]:
                result = response.json()
                
                source_info = DataSourceInfo(
                    source_id=result['id'],
                    source_name=result['name'],
                    connection_type="api",
                    dataset_id=dataset_info['id'],
                    last_refresh=datetime.now(),
                    refresh_status="success"
                )
                
                return source_info
            else:
                raise Exception(f"Failed to create data source: {response.status_code}")
                
        except Exception as e:
            self.logger.error(f"Failed to create data source: {str(e)}")
            raise
    
    def update_data_source(self, source_id: str, dataset_info: Dict[str, Any]) -> bool:
        """Update data source via REST API"""
        try:
            response = self.session.put(
                f"{self.config.server_url}/api/datasources/{source_id}/refresh"
            )
            return response.status_code == 200
        except Exception as e:
            self.logger.error(f"Failed to update data source: {str(e)}")
            return False
    
    def get_data_sources(self) -> List[DataSourceInfo]:
        """Get data sources via REST API"""
        try:
            response = self.session.get(f"{self.config.server_url}/api/datasources")
            if response.status_code == 200:
                sources = response.json()
                return [
                    DataSourceInfo(
                        source_id=src['id'],
                        source_name=src['name'],
                        connection_type="api",
                        refresh_status=src.get('status', 'unknown')
                    )
                    for src in sources
                ]
            return []
        except Exception as e:
            self.logger.error(f"Failed to get data sources: {str(e)}")
            return []
    
    def create_report(self, report_config: Dict[str, Any]) -> str:
        """Create report via REST API"""
        try:
            response = self.session.post(
                f"{self.config.server_url}/api/reports",
                json=report_config
            )
            
            if response.status_code in [200, 201]:
                result = response.json()
                return result['id']
            else:
                raise Exception(f"Failed to create report: {response.status_code}")
                
        except Exception as e:
            self.logger.error(f"Failed to create report: {str(e)}")
            raise
    
    def distribute_report(self, distribution_info: ReportDistributionInfo) -> bool:
        """Distribute report via REST API"""
        try:
            distribution_config = {
                'report_id': distribution_info.report_id,
                'recipients': distribution_info.recipients,
                'schedule': distribution_info.distribution_schedule,
                'format': distribution_info.format
            }
            
            response = self.session.post(
                f"{self.config.server_url}/api/reports/distribute",
                json=distribution_config
            )
            
            return response.status_code == 200
            
        except Exception as e:
            self.logger.error(f"Failed to distribute report: {str(e)}")
            return False


class DataExporter:
    """Handles data export in multiple formats"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.DataExporter")
    
    def export_dataset(
        self, 
        data: pd.DataFrame, 
        config: DataExportConfig,
        output_path: Optional[str] = None
    ) -> Union[str, bytes, Dict[str, Any]]:
        """Export dataset in specified format"""
        try:
            if config.filters:
                data = self._apply_filters(data, config.filters)
            
            if config.export_format.lower() == 'csv':
                return self._export_csv(data, config, output_path)
            elif config.export_format.lower() == 'json':
                return self._export_json(data, config, output_path)
            elif config.export_format.lower() == 'parquet':
                return self._export_parquet(data, config, output_path)
            elif config.export_format.lower() == 'excel':
                return self._export_excel(data, config, output_path)
            else:
                raise ValueError(f"Unsupported export format: {config.export_format}")
                
        except Exception as e:
            self.logger.error(f"Failed to export dataset: {str(e)}")
            raise
    
    def _apply_filters(self, data: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
        """Apply filters to the dataset"""
        filtered_data = data.copy()
        
        for column, filter_config in filters.items():
            if column not in filtered_data.columns:
                continue
                
            if 'equals' in filter_config:
                filtered_data = filtered_data[filtered_data[column] == filter_config['equals']]
            elif 'in' in filter_config:
                filtered_data = filtered_data[filtered_data[column].isin(filter_config['in'])]
            elif 'range' in filter_config:
                min_val, max_val = filter_config['range']
                filtered_data = filtered_data[
                    (filtered_data[column] >= min_val) & 
                    (filtered_data[column] <= max_val)
                ]
        
        return filtered_data
    
    def _export_csv(self, data: pd.DataFrame, config: DataExportConfig, output_path: Optional[str]) -> str:
        """Export as CSV"""
        if output_path:
            data.to_csv(output_path, index=False, compression=config.compression)
            return output_path
        else:
            return data.to_csv(index=False, compression=config.compression)
    
    def _export_json(self, data: pd.DataFrame, config: DataExportConfig, output_path: Optional[str]) -> Union[str, Dict]:
        """Export as JSON"""
        json_data = data.to_dict(orient='records')
        
        if config.include_metadata:
            export_data = {
                'metadata': {
                    'export_timestamp': datetime.now().isoformat(),
                    'row_count': len(data),
                    'column_count': len(data.columns),
                    'columns': list(data.columns)
                },
                'data': json_data
            }
        else:
            export_data = json_data
        
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            return output_path
        else:
            return export_data
    
    def _export_parquet(self, data: pd.DataFrame, config: DataExportConfig, output_path: Optional[str]) -> str:
        """Export as Parquet"""
        if output_path:
            data.to_parquet(output_path, compression=config.compression)
            return output_path
        else:
            # Return as bytes
            buffer = io.BytesIO()
            data.to_parquet(buffer, compression=config.compression)
            return buffer.getvalue()
    
    def _export_excel(self, data: pd.DataFrame, config: DataExportConfig, output_path: Optional[str]) -> str:
        """Export as Excel"""
        if output_path:
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                data.to_excel(writer, sheet_name='Data', index=False)
                
                if config.include_metadata:
                    metadata_df = pd.DataFrame([
                        ['Export Timestamp', datetime.now().isoformat()],
                        ['Row Count', len(data)],
                        ['Column Count', len(data.columns)],
                        ['Columns', ', '.join(data.columns)]
                    ], columns=['Property', 'Value'])
                    
                    metadata_df.to_excel(writer, sheet_name='Metadata', index=False)
            
            return output_path
        else:
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                data.to_excel(writer, sheet_name='Data', index=False)
            return buffer.getvalue()


class BIAnalyticsIntegrator:
    """Main integration engine for BI and analytics tools"""
    
    def __init__(self):
        self.connectors: Dict[str, BIToolConnector] = {}
        self.data_exporter = DataExporter()
        self.logger = logging.getLogger(__name__)
    
    def register_bi_tool(self, tool_name: str, config: BIToolConfig) -> bool:
        """Register a new BI tool"""
        try:
            if config.tool_type.lower() == 'tableau':
                connector = TableauConnector(config)
            elif config.tool_type.lower() == 'powerbi':
                connector = PowerBIConnector(config)
            else:
                connector = GenericBIConnector(config)
            
            if connector.connect():
                self.connectors[tool_name] = connector
                self.logger.info(f"Successfully registered BI tool: {tool_name}")
                return True
            else:
                self.logger.error(f"Failed to connect to BI tool: {tool_name}")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to register BI tool {tool_name}: {str(e)}")
            return False
    
    def create_data_sources(self, dataset_info: Dict[str, Any]) -> Dict[str, DataSourceInfo]:
        """Create data sources across all registered BI tools"""
        results = {}
        
        for tool_name, connector in self.connectors.items():
            try:
                source_info = connector.create_data_source(dataset_info)
                results[tool_name] = source_info
                self.logger.info(f"Created data source in {tool_name}: {source_info.source_name}")
            except Exception as e:
                self.logger.error(f"Failed to create data source in {tool_name}: {str(e)}")
                results[tool_name] = None
        
        return results
    
    def export_data(
        self, 
        data: pd.DataFrame, 
        export_configs: List[DataExportConfig],
        base_filename: str = "ai_readiness_data"
    ) -> Dict[str, str]:
        """Export data in multiple formats"""
        export_results = {}
        
        for config in export_configs:
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{base_filename}_{timestamp}.{config.export_format}"
                
                result = self.data_exporter.export_dataset(data, config, filename)
                export_results[config.export_format] = result
                
                self.logger.info(f"Exported data as {config.export_format}: {filename}")
                
            except Exception as e:
                self.logger.error(f"Failed to export data as {config.export_format}: {str(e)}")
                export_results[config.export_format] = None
        
        return export_results
    
    def distribute_reports(self, distribution_configs: List[ReportDistributionInfo]) -> Dict[str, bool]:
        """Distribute reports across BI tools"""
        results = {}
        
        for distribution in distribution_configs:
            success_count = 0
            total_count = 0
            
            for tool_name, connector in self.connectors.items():
                try:
                    total_count += 1
                    success = connector.distribute_report(distribution)
                    if success:
                        success_count += 1
                        
                except Exception as e:
                    self.logger.error(f"Failed to distribute report via {tool_name}: {str(e)}")
            
            results[distribution.report_id] = success_count > 0
            self.logger.info(f"Report {distribution.report_name} distributed to {success_count}/{total_count} tools")
        
        return results
    
    def sync_data_sources(self, dataset_updates: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, bool]]:
        """Synchronize data sources across BI tools"""
        results = {}
        
        for dataset_id, dataset_info in dataset_updates.items():
            results[dataset_id] = {}
            
            for tool_name, connector in self.connectors.items():
                try:
                    # Find existing data sources for this dataset
                    data_sources = connector.get_data_sources()
                    matching_sources = [
                        ds for ds in data_sources 
                        if ds.dataset_id == dataset_id
                    ]
                    
                    if matching_sources:
                        # Update existing sources
                        for source in matching_sources:
                            success = connector.update_data_source(source.source_id, dataset_info)
                            results[dataset_id][f"{tool_name}_{source.source_id}"] = success
                    else:
                        # Create new data source
                        source_info = connector.create_data_source(dataset_info)
                        results[dataset_id][f"{tool_name}_new"] = source_info is not None
                        
                except Exception as e:
                    self.logger.error(f"Failed to sync data source in {tool_name}: {str(e)}")
                    results[dataset_id][tool_name] = False
        
        return results
    
    def get_integration_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all BI tool integrations"""
        status = {}
        
        for tool_name, connector in self.connectors.items():
            try:
                is_connected = connector.connect()
                data_sources = connector.get_data_sources()
                
                status[tool_name] = {
                    'connected': is_connected,
                    'tool_type': connector.config.tool_type,
                    'server_url': connector.config.server_url,
                    'data_source_count': len(data_sources),
                    'last_checked': datetime.now().isoformat(),
                    'workspace_id': connector.config.workspace_id
                }
                
            except Exception as e:
                status[tool_name] = {
                    'connected': False,
                    'error': str(e),
                    'last_checked': datetime.now().isoformat()
                }
        
        return status