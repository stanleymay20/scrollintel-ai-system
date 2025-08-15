"""
API routes for BI and Analytics tool integration
"""

from flask import Blueprint, request, jsonify, send_file
from flask_restx import Api, Resource, fields, Namespace
from typing import Dict, Any, List
import logging
from datetime import datetime
import pandas as pd
import tempfile
import os

from ...engines.bi_analytics_integrator import (
    BIAnalyticsIntegrator,
    BIToolConfig,
    DataExportConfig,
    ReportDistributionInfo,
    DataSourceInfo
)
from ...models.bi_integration_models import (
    BITool,
    BIDataSource,
    BIReport,
    DataExportJob
)

logger = logging.getLogger(__name__)

# Create blueprint
bi_integration_bp = Blueprint('bi_integration', __name__, url_prefix='/api/v1/bi-integration')

# Create namespace for Swagger documentation
ns = Namespace('bi-integration', description='BI and Analytics Tool Integration operations')

# Initialize the integrator
integrator = BIAnalyticsIntegrator()

# Request/Response models for documentation
bi_tool_config_model = ns.model('BIToolConfig', {
    'name': fields.String(required=True, description='BI tool name'),
    'tool_type': fields.String(required=True, description='Tool type (tableau, powerbi, looker, generic)'),
    'server_url': fields.String(required=True, description='BI tool server URL'),
    'workspace_id': fields.String(description='Workspace/Project ID'),
    'credentials': fields.Raw(description='Authentication credentials'),
    'metadata': fields.Raw(description='Additional metadata')
})

data_export_config_model = ns.model('DataExportConfig', {
    'export_format': fields.String(required=True, description='Export format (csv, json, parquet, excel)'),
    'include_metadata': fields.Boolean(description='Include metadata in export'),
    'compression': fields.String(description='Compression type'),
    'filters': fields.Raw(description='Data filters to apply')
})

report_distribution_model = ns.model('ReportDistribution', {
    'report_id': fields.String(required=True, description='Report ID'),
    'report_name': fields.String(required=True, description='Report name'),
    'recipients': fields.List(fields.String, required=True, description='Email recipients'),
    'distribution_schedule': fields.String(required=True, description='Distribution schedule'),
    'format': fields.String(required=True, description='Distribution format')
})

data_source_create_model = ns.model('DataSourceCreate', {
    'dataset_id': fields.String(required=True, description='Dataset ID'),
    'name': fields.String(required=True, description='Data source name'),
    'description': fields.String(description='Data source description'),
    'metadata': fields.Raw(description='Additional metadata')
})


@ns.route('/tools')
class BIToolList(Resource):
    """BI Tool management"""
    
    @ns.doc('list_bi_tools')
    def get(self):
        """Get all registered BI tools"""
        try:
            status = integrator.get_integration_status()
            return {
                'success': True,
                'data': status,
                'message': f'Retrieved status for {len(status)} BI tools'
            }
        except Exception as e:
            logger.error(f"Failed to get BI tool status: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'message': 'Failed to retrieve BI tool status'
            }, 500
    
    @ns.doc('register_bi_tool')
    @ns.expect(bi_tool_config_model)
    def post(self):
        """Register a new BI tool"""
        try:
            data = request.get_json()
            
            # Create BI tool configuration
            config = BIToolConfig(
                tool_type=data['tool_type'],
                server_url=data['server_url'],
                credentials=data.get('credentials', {}),
                workspace_id=data.get('workspace_id'),
                metadata=data.get('metadata', {})
            )
            
            # Register the BI tool
            success = integrator.register_bi_tool(data['name'], config)
            
            if success:
                return {
                    'success': True,
                    'message': f'Successfully registered BI tool: {data["name"]}'
                }
            else:
                return {
                    'success': False,
                    'message': f'Failed to register BI tool: {data["name"]}'
                }, 400
                
        except Exception as e:
            logger.error(f"Failed to register BI tool: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'message': 'Failed to register BI tool'
            }, 500


@ns.route('/tools/<string:tool_name>')
class BIToolDetail(Resource):
    """Individual BI Tool operations"""
    
    @ns.doc('get_bi_tool_status')
    def get(self, tool_name):
        """Get status of a specific BI tool"""
        try:
            status = integrator.get_integration_status()
            
            if tool_name in status:
                return {
                    'success': True,
                    'data': status[tool_name],
                    'message': f'Retrieved status for BI tool: {tool_name}'
                }
            else:
                return {
                    'success': False,
                    'message': f'BI tool not found: {tool_name}'
                }, 404
                
        except Exception as e:
            logger.error(f"Failed to get BI tool status: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'message': 'Failed to retrieve BI tool status'
            }, 500


@ns.route('/data-sources')
class DataSourceList(Resource):
    """Data source management across BI tools"""
    
    @ns.doc('create_data_sources')
    @ns.expect(data_source_create_model)
    def post(self):
        """Create data sources across all registered BI tools"""
        try:
            data = request.get_json()
            
            dataset_info = {
                'id': data['dataset_id'],
                'name': data['name'],
                'description': data.get('description', ''),
                'metadata': data.get('metadata', {})
            }
            
            # Create data sources across all BI tools
            results = integrator.create_data_sources(dataset_info)
            
            # Count successful creations
            successful_count = sum(1 for result in results.values() if result is not None)
            
            return {
                'success': True,
                'data': {
                    'dataset_id': data['dataset_id'],
                    'results': {
                        tool_name: {
                            'success': result is not None,
                            'source_id': result.source_id if result else None,
                            'source_name': result.source_name if result else None
                        }
                        for tool_name, result in results.items()
                    },
                    'successful_count': successful_count,
                    'total_tools': len(results)
                },
                'message': f'Created data sources in {successful_count}/{len(results)} BI tools'
            }
            
        except Exception as e:
            logger.error(f"Failed to create data sources: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'message': 'Failed to create data sources'
            }, 500


@ns.route('/data-sources/sync')
class DataSourceSync(Resource):
    """Data source synchronization"""
    
    @ns.doc('sync_data_sources')
    def post(self):
        """Synchronize data sources across BI tools"""
        try:
            data = request.get_json()
            dataset_updates = data.get('dataset_updates', {})
            
            if not dataset_updates:
                return {
                    'success': False,
                    'message': 'No dataset updates provided'
                }, 400
            
            # Synchronize data sources
            results = integrator.sync_data_sources(dataset_updates)
            
            # Calculate summary statistics
            total_syncs = sum(len(tool_results) for tool_results in results.values())
            successful_syncs = sum(
                sum(1 for success in tool_results.values() if success)
                for tool_results in results.values()
            )
            
            return {
                'success': True,
                'data': {
                    'sync_results': results,
                    'summary': {
                        'total_syncs': total_syncs,
                        'successful_syncs': successful_syncs,
                        'failed_syncs': total_syncs - successful_syncs,
                        'datasets_processed': len(results)
                    }
                },
                'message': f'Synchronized {successful_syncs}/{total_syncs} data sources'
            }
            
        except Exception as e:
            logger.error(f"Failed to sync data sources: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'message': 'Failed to sync data sources'
            }, 500


@ns.route('/export')
class DataExport(Resource):
    """Data export operations"""
    
    @ns.doc('export_data')
    @ns.expect([data_export_config_model])
    def post(self):
        """Export data in multiple formats"""
        try:
            data = request.get_json()
            
            # Get dataset information
            dataset_id = data.get('dataset_id')
            if not dataset_id:
                return {
                    'success': False,
                    'message': 'dataset_id is required'
                }, 400
            
            # Create sample data for demo (in real implementation, fetch from database)
            sample_data = pd.DataFrame({
                'id': range(1, 101),
                'name': [f'Record {i}' for i in range(1, 101)],
                'value': [i * 1.5 for i in range(1, 101)],
                'category': ['A' if i % 2 == 0 else 'B' for i in range(1, 101)],
                'timestamp': pd.date_range('2024-01-01', periods=100, freq='D')
            })
            
            # Parse export configurations
            export_configs = []
            for config_data in data.get('export_configs', []):
                config = DataExportConfig(
                    export_format=config_data['export_format'],
                    include_metadata=config_data.get('include_metadata', True),
                    compression=config_data.get('compression'),
                    filters=config_data.get('filters')
                )
                export_configs.append(config)
            
            # Export data
            export_results = integrator.export_data(
                sample_data,
                export_configs,
                f"dataset_{dataset_id}"
            )
            
            # Count successful exports
            successful_exports = sum(1 for result in export_results.values() if result is not None)
            
            return {
                'success': True,
                'data': {
                    'dataset_id': dataset_id,
                    'export_results': export_results,
                    'summary': {
                        'successful_exports': successful_exports,
                        'total_formats': len(export_configs),
                        'row_count': len(sample_data),
                        'column_count': len(sample_data.columns)
                    }
                },
                'message': f'Exported data in {successful_exports}/{len(export_configs)} formats'
            }
            
        except Exception as e:
            logger.error(f"Failed to export data: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'message': 'Failed to export data'
            }, 500


@ns.route('/export/<string:export_format>/<string:dataset_id>')
class DataExportDownload(Resource):
    """Download exported data files"""
    
    @ns.doc('download_export')
    def get(self, export_format, dataset_id):
        """Download exported data file"""
        try:
            # Create sample data for demo
            sample_data = pd.DataFrame({
                'id': range(1, 101),
                'name': [f'Record {i}' for i in range(1, 101)],
                'value': [i * 1.5 for i in range(1, 101)],
                'category': ['A' if i % 2 == 0 else 'B' for i in range(1, 101)],
                'timestamp': pd.date_range('2024-01-01', periods=100, freq='D')
            })
            
            # Create export configuration
            config = DataExportConfig(
                export_format=export_format,
                include_metadata=True
            )
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{export_format}') as tmp_file:
                result = integrator.data_exporter.export_dataset(
                    sample_data,
                    config,
                    tmp_file.name
                )
                
                # Return file for download
                return send_file(
                    tmp_file.name,
                    as_attachment=True,
                    download_name=f'dataset_{dataset_id}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.{export_format}',
                    mimetype='application/octet-stream'
                )
                
        except Exception as e:
            logger.error(f"Failed to download export: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'message': 'Failed to download export'
            }, 500
        finally:
            # Clean up temporary file
            try:
                if 'tmp_file' in locals():
                    os.unlink(tmp_file.name)
            except:
                pass


@ns.route('/reports/distribute')
class ReportDistribution(Resource):
    """Report distribution operations"""
    
    @ns.doc('distribute_reports')
    @ns.expect([report_distribution_model])
    def post(self):
        """Distribute reports to stakeholders"""
        try:
            data = request.get_json()
            
            # Parse distribution configurations
            distributions = []
            for dist_data in data.get('distributions', []):
                distribution = ReportDistributionInfo(
                    report_id=dist_data['report_id'],
                    report_name=dist_data['report_name'],
                    recipients=dist_data['recipients'],
                    distribution_schedule=dist_data['distribution_schedule'],
                    format=dist_data['format'],
                    status='pending',
                    metadata=dist_data.get('metadata', {})
                )
                distributions.append(distribution)
            
            # Distribute reports
            results = integrator.distribute_reports(distributions)
            
            # Count successful distributions
            successful_count = sum(1 for success in results.values() if success)
            
            return {
                'success': True,
                'data': {
                    'distribution_results': results,
                    'summary': {
                        'successful_distributions': successful_count,
                        'total_reports': len(distributions),
                        'failed_distributions': len(distributions) - successful_count
                    }
                },
                'message': f'Distributed {successful_count}/{len(distributions)} reports'
            }
            
        except Exception as e:
            logger.error(f"Failed to distribute reports: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'message': 'Failed to distribute reports'
            }, 500


@ns.route('/health')
class BIIntegrationHealth(Resource):
    """Health check for BI integration service"""
    
    @ns.doc('health_check')
    def get(self):
        """Check health of BI integration service"""
        try:
            status = integrator.get_integration_status()
            
            connected_tools = sum(1 for s in status.values() if s.get('connected', False))
            total_tools = len(status)
            total_data_sources = sum(s.get('data_source_count', 0) for s in status.values())
            
            health_status = {
                'service': 'bi_integration',
                'status': 'healthy' if connected_tools > 0 else 'degraded',
                'tools': {
                    'total': total_tools,
                    'connected': connected_tools,
                    'disconnected': total_tools - connected_tools
                },
                'data_sources': {
                    'total': total_data_sources
                },
                'timestamp': datetime.now().isoformat()
            }
            
            return {
                'success': True,
                'data': health_status,
                'message': 'BI integration service health check completed'
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'message': 'Health check failed'
            }, 500


# Register routes with the blueprint
bi_integration_bp.add_url_rule('/tools', view_func=BIToolList.as_view('bi_tool_list'))
bi_integration_bp.add_url_rule('/tools/<string:tool_name>', view_func=BIToolDetail.as_view('bi_tool_detail'))
bi_integration_bp.add_url_rule('/data-sources', view_func=DataSourceList.as_view('data_source_list'))
bi_integration_bp.add_url_rule('/data-sources/sync', view_func=DataSourceSync.as_view('data_source_sync'))
bi_integration_bp.add_url_rule('/export', view_func=DataExport.as_view('data_export'))
bi_integration_bp.add_url_rule('/export/<string:export_format>/<string:dataset_id>', view_func=DataExportDownload.as_view('data_export_download'))
bi_integration_bp.add_url_rule('/reports/distribute', view_func=ReportDistribution.as_view('report_distribution'))
bi_integration_bp.add_url_rule('/health', view_func=BIIntegrationHealth.as_view('health_check'))