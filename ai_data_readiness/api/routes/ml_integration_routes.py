"""
API routes for ML platform integration
"""

from flask import Blueprint, request, jsonify
from flask_restx import Api, Resource, fields, Namespace
from typing import Dict, Any, List
import logging
from datetime import datetime

from ...engines.ml_platform_integrator import (
    MLPlatformIntegrator, 
    MLPlatformConfig,
    ModelDeploymentInfo,
    DataQualityCorrelation
)
from ...models.ml_integration_models import (
    MLPlatform,
    ModelDeployment,
    DataQualityCorrelation as DBDataQualityCorrelation
)

logger = logging.getLogger(__name__)

# Create blueprint
ml_integration_bp = Blueprint('ml_integration', __name__, url_prefix='/api/v1/ml-integration')

# Create namespace for Swagger documentation
ns = Namespace('ml-integration', description='ML Platform Integration operations')

# Initialize the integrator
integrator = MLPlatformIntegrator()

# Request/Response models for documentation
platform_config_model = ns.model('MLPlatformConfig', {
    'name': fields.String(required=True, description='Platform name'),
    'platform_type': fields.String(required=True, description='Platform type (mlflow, kubeflow, generic)'),
    'endpoint_url': fields.String(required=True, description='Platform endpoint URL'),
    'credentials': fields.Raw(description='Platform credentials'),
    'metadata': fields.Raw(description='Additional metadata')
})

model_deployment_model = ns.model('ModelDeployment', {
    'model_id': fields.String(required=True, description='Model ID'),
    'model_name': fields.String(required=True, description='Model name'),
    'version': fields.String(description='Model version'),
    'deployment_config': fields.Raw(description='Deployment configuration')
})

correlation_request_model = ns.model('CorrelationRequest', {
    'dataset_id': fields.String(required=True, description='Dataset ID'),
    'quality_score': fields.Float(required=True, description='Overall quality score'),
    'quality_dimensions': fields.Raw(required=True, description='Quality dimensions breakdown')
})


@ns.route('/platforms')
class MLPlatformList(Resource):
    """ML Platform management"""
    
    @ns.doc('list_platforms')
    def get(self):
        """Get all registered ML platforms"""
        try:
            status = integrator.get_platform_status()
            return {
                'success': True,
                'data': status,
                'message': f'Retrieved status for {len(status)} platforms'
            }
        except Exception as e:
            logger.error(f"Failed to get platform status: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'message': 'Failed to retrieve platform status'
            }, 500
    
    @ns.doc('register_platform')
    @ns.expect(platform_config_model)
    def post(self):
        """Register a new ML platform"""
        try:
            data = request.get_json()
            
            # Create platform configuration
            config = MLPlatformConfig(
                platform_type=data['platform_type'],
                endpoint_url=data['endpoint_url'],
                credentials=data.get('credentials', {}),
                metadata=data.get('metadata', {})
            )
            
            # Register the platform
            success = integrator.register_platform(data['name'], config)
            
            if success:
                return {
                    'success': True,
                    'message': f'Successfully registered platform: {data["name"]}'
                }
            else:
                return {
                    'success': False,
                    'message': f'Failed to register platform: {data["name"]}'
                }, 400
                
        except Exception as e:
            logger.error(f"Failed to register platform: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'message': 'Failed to register platform'
            }, 500


@ns.route('/platforms/<string:platform_name>')
class MLPlatformDetail(Resource):
    """Individual ML Platform operations"""
    
    @ns.doc('get_platform_status')
    def get(self, platform_name):
        """Get status of a specific ML platform"""
        try:
            status = integrator.get_platform_status()
            
            if platform_name in status:
                return {
                    'success': True,
                    'data': status[platform_name],
                    'message': f'Retrieved status for platform: {platform_name}'
                }
            else:
                return {
                    'success': False,
                    'message': f'Platform not found: {platform_name}'
                }, 404
                
        except Exception as e:
            logger.error(f"Failed to get platform status: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'message': 'Failed to retrieve platform status'
            }, 500


@ns.route('/models')
class ModelList(Resource):
    """Model management across platforms"""
    
    @ns.doc('list_all_models')
    def get(self):
        """Get models from all registered ML platforms"""
        try:
            all_models = integrator.get_all_models()
            
            # Flatten the results for easier consumption
            flattened_models = []
            for platform_name, models in all_models.items():
                for model in models:
                    model['platform'] = platform_name
                    flattened_models.append(model)
            
            return {
                'success': True,
                'data': {
                    'models': flattened_models,
                    'by_platform': all_models,
                    'total_count': len(flattened_models)
                },
                'message': f'Retrieved {len(flattened_models)} models from {len(all_models)} platforms'
            }
            
        except Exception as e:
            logger.error(f"Failed to retrieve models: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'message': 'Failed to retrieve models'
            }, 500


@ns.route('/models/deploy')
class ModelDeployment(Resource):
    """Model deployment operations"""
    
    @ns.doc('deploy_model')
    @ns.expect(model_deployment_model)
    def post(self):
        """Deploy a model to a specific ML platform"""
        try:
            data = request.get_json()
            platform_name = data.get('platform_name')
            
            if not platform_name:
                return {
                    'success': False,
                    'message': 'platform_name is required'
                }, 400
            
            # Deploy the model
            deployment_info = integrator.deploy_model(platform_name, data)
            
            if deployment_info:
                return {
                    'success': True,
                    'data': {
                        'model_id': deployment_info.model_id,
                        'model_name': deployment_info.model_name,
                        'version': deployment_info.version,
                        'deployment_status': deployment_info.deployment_status,
                        'endpoint_url': deployment_info.endpoint_url,
                        'created_at': deployment_info.created_at.isoformat() if deployment_info.created_at else None
                    },
                    'message': f'Successfully deployed model {deployment_info.model_name}'
                }
            else:
                return {
                    'success': False,
                    'message': 'Failed to deploy model'
                }, 500
                
        except Exception as e:
            logger.error(f"Failed to deploy model: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'message': 'Failed to deploy model'
            }, 500


@ns.route('/correlation/analyze')
class DataQualityCorrelationAnalysis(Resource):
    """Data quality and model performance correlation analysis"""
    
    @ns.doc('analyze_correlation')
    @ns.expect(correlation_request_model)
    def post(self):
        """Analyze correlation between data quality and model performance"""
        try:
            data = request.get_json()
            
            dataset_id = data['dataset_id']
            quality_score = data['quality_score']
            quality_dimensions = data['quality_dimensions']
            
            # Perform correlation analysis
            correlations = integrator.correlate_data_quality_with_performance(
                dataset_id=dataset_id,
                quality_score=quality_score,
                quality_dimensions=quality_dimensions
            )
            
            # Convert to serializable format
            correlation_data = []
            for corr in correlations:
                correlation_data.append({
                    'dataset_id': corr.dataset_id,
                    'model_id': corr.model_id,
                    'quality_score': corr.quality_score,
                    'performance_score': corr.performance_score,
                    'correlation_coefficient': corr.correlation_coefficient,
                    'quality_dimensions': corr.quality_dimensions,
                    'performance_metrics': corr.performance_metrics,
                    'timestamp': corr.timestamp.isoformat()
                })
            
            return {
                'success': True,
                'data': {
                    'correlations': correlation_data,
                    'summary': {
                        'total_models_analyzed': len(correlation_data),
                        'average_correlation': sum(c['correlation_coefficient'] for c in correlation_data) / len(correlation_data) if correlation_data else 0,
                        'high_correlation_count': len([c for c in correlation_data if c['correlation_coefficient'] > 0.7]),
                        'dataset_id': dataset_id,
                        'analysis_timestamp': datetime.now().isoformat()
                    }
                },
                'message': f'Analyzed correlation for {len(correlation_data)} models'
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze correlation: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'message': 'Failed to analyze correlation'
            }, 500


@ns.route('/platforms/<string:platform_name>/models')
class PlatformModelList(Resource):
    """Platform-specific model operations"""
    
    @ns.doc('list_platform_models')
    def get(self, platform_name):
        """Get models from a specific ML platform"""
        try:
            all_models = integrator.get_all_models()
            
            if platform_name in all_models:
                models = all_models[platform_name]
                return {
                    'success': True,
                    'data': {
                        'platform': platform_name,
                        'models': models,
                        'count': len(models)
                    },
                    'message': f'Retrieved {len(models)} models from {platform_name}'
                }
            else:
                return {
                    'success': False,
                    'message': f'Platform not found: {platform_name}'
                }, 404
                
        except Exception as e:
            logger.error(f"Failed to retrieve models from {platform_name}: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'message': f'Failed to retrieve models from {platform_name}'
            }, 500


@ns.route('/health')
class MLIntegrationHealth(Resource):
    """Health check for ML integration service"""
    
    @ns.doc('health_check')
    def get(self):
        """Check health of ML integration service"""
        try:
            status = integrator.get_platform_status()
            
            connected_platforms = sum(1 for s in status.values() if s.get('connected', False))
            total_platforms = len(status)
            
            health_status = {
                'service': 'ml_integration',
                'status': 'healthy' if connected_platforms > 0 else 'degraded',
                'platforms': {
                    'total': total_platforms,
                    'connected': connected_platforms,
                    'disconnected': total_platforms - connected_platforms
                },
                'timestamp': datetime.now().isoformat()
            }
            
            return {
                'success': True,
                'data': health_status,
                'message': 'ML integration service health check completed'
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'message': 'Health check failed'
            }, 500


# Register routes with the blueprint
ml_integration_bp.add_url_rule('/platforms', view_func=MLPlatformList.as_view('platform_list'))
ml_integration_bp.add_url_rule('/platforms/<string:platform_name>', view_func=MLPlatformDetail.as_view('platform_detail'))
ml_integration_bp.add_url_rule('/models', view_func=ModelList.as_view('model_list'))
ml_integration_bp.add_url_rule('/models/deploy', view_func=ModelDeployment.as_view('model_deployment'))
ml_integration_bp.add_url_rule('/correlation/analyze', view_func=DataQualityCorrelationAnalysis.as_view('correlation_analysis'))
ml_integration_bp.add_url_rule('/platforms/<string:platform_name>/models', view_func=PlatformModelList.as_view('platform_model_list'))
ml_integration_bp.add_url_rule('/health', view_func=MLIntegrationHealth.as_view('health_check'))