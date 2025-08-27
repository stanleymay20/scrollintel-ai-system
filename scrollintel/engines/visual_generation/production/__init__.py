"""
Production Visual Generation System
Complete production-ready implementation with deployment, storage, monitoring, and security
"""

import asyncio
import logging
from typing import Dict, Any, Optional

from .model_deployment import production_deployment, initialize_production_models
from .gpu_infrastructure import gpu_infrastructure, initialize_gpu_infrastructure
from .production_config import production_config, validate_environment
from ..storage.cloud_storage import storage_manager, cdn_manager, archival_manager
from ..storage.content_versioning import create_version_manager, create_backup_manager, create_disaster_recovery_manager
from ..monitoring.production_monitoring import monitoring_manager, initialize_monitoring
from ..monitoring.business_metrics import business_metrics, initialize_business_metrics
from ..security.production_security import security_manager, initialize_security

logger = logging.getLogger(__name__)

class ProductionVisualGenerationSystem:
    """Complete production visual generation system"""
    
    def __init__(self):
        self.initialized = False
        self.components = {
            'model_deployment': production_deployment,
            'gpu_infrastructure': gpu_infrastructure,
            'storage_manager': storage_manager,
            'cdn_manager': cdn_manager,
            'archival_manager': archival_manager,
            'monitoring_manager': monitoring_manager,
            'business_metrics': business_metrics,
            'security_manager': security_manager,
            'config': production_config
        }
        
        # Create additional managers
        self.version_manager = create_version_manager(storage_manager)
        self.backup_manager = create_backup_manager(storage_manager)
        self.disaster_recovery_manager = create_disaster_recovery_manager(
            storage_manager, self.backup_manager, self.version_manager
        )
    
    async def initialize_system(self) -> bool:
        """Initialize complete production system"""
        try:
            logger.info("Initializing Production Visual Generation System")
            
            # Initialize components in order
            initialization_steps = [
                ("Environment Validation", validate_environment),
                ("Security System", initialize_security),
                ("GPU Infrastructure", initialize_gpu_infrastructure),
                ("Production Models", initialize_production_models),
                ("Business Metrics", initialize_business_metrics),
                ("Production Monitoring", initialize_monitoring)
            ]
            
            for step_name, init_function in initialization_steps:
                logger.info(f"Initializing {step_name}...")
                
                try:
                    result = await init_function()
                    if result:
                        logger.info(f"✅ {step_name} initialized successfully")
                    else:
                        logger.error(f"❌ {step_name} initialization failed")
                        return False
                except Exception as e:
                    logger.error(f"❌ {step_name} initialization error: {str(e)}")
                    return False
            
            self.initialized = True
            logger.info("🎉 Production Visual Generation System initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"System initialization failed: {str(e)}")
            return False
    
    async def shutdown_system(self):
        """Shutdown production system gracefully"""
        try:
            logger.info("Shutting down Production Visual Generation System")
            
            # Shutdown monitoring
            if hasattr(monitoring_manager, 'stop_monitoring'):
                await monitoring_manager.stop_monitoring()
            
            self.initialized = False
            logger.info("System shutdown complete")
            
        except Exception as e:
            logger.error(f"System shutdown error: {str(e)}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        try:
            status = {
                'initialized': self.initialized,
                'timestamp': asyncio.get_event_loop().time(),
                'components': {}
            }
            
            # Get component statuses
            if hasattr(production_deployment, 'get_model_status'):
                status['components']['models'] = asyncio.create_task(
                    production_deployment.get_model_status()
                )
            
            if hasattr(gpu_infrastructure, 'get_cluster_status'):
                status['components']['gpu_clusters'] = asyncio.create_task(
                    gpu_infrastructure.get_cluster_status()
                )
            
            if hasattr(monitoring_manager, 'get_monitoring_status'):
                status['components']['monitoring'] = monitoring_manager.get_monitoring_status()
            
            # Get configuration
            status['configuration'] = production_config.export_config()
            
            return status
            
        except Exception as e:
            logger.error(f"Failed to get system status: {str(e)}")
            return {'error': str(e)}
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check"""
        try:
            health_status = {
                'overall_health': 'healthy',
                'timestamp': asyncio.get_event_loop().time(),
                'checks': {}
            }
            
            # Check model deployment
            try:
                model_status = await production_deployment.get_model_status()
                ready_models = sum(
                    1 for status in model_status.values()
                    if isinstance(status, dict) and status.get('status') == 'ready'
                )
                health_status['checks']['models'] = {
                    'status': 'healthy' if ready_models > 0 else 'unhealthy',
                    'ready_models': ready_models
                }
            except Exception as e:
                health_status['checks']['models'] = {
                    'status': 'error',
                    'error': str(e)
                }
            
            # Check GPU infrastructure
            try:
                gpu_status = await gpu_infrastructure.get_cluster_status()
                health_status['checks']['gpu'] = {
                    'status': 'healthy' if gpu_status else 'unhealthy',
                    'clusters': len(gpu_status)
                }
            except Exception as e:
                health_status['checks']['gpu'] = {
                    'status': 'error',
                    'error': str(e)
                }
            
            # Check storage
            try:
                storage_metrics = await storage_manager.get_storage_metrics()
                health_status['checks']['storage'] = {
                    'status': 'healthy',
                    'total_objects': storage_metrics.get('total_objects', 0)
                }
            except Exception as e:
                health_status['checks']['storage'] = {
                    'status': 'error',
                    'error': str(e)
                }
            
            # Determine overall health
            unhealthy_checks = [
                check for check in health_status['checks'].values()
                if check['status'] != 'healthy'
            ]
            
            if unhealthy_checks:
                health_status['overall_health'] = 'degraded' if len(unhealthy_checks) < len(health_status['checks']) else 'unhealthy'
            
            return health_status
            
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return {
                'overall_health': 'error',
                'error': str(e)
            }

# Global production system instance
production_system = ProductionVisualGenerationSystem()

async def initialize_production_system():
    """Initialize the complete production system"""
    return await production_system.initialize_system()

async def shutdown_production_system():
    """Shutdown the production system"""
    await production_system.shutdown_system()

def get_production_status():
    """Get production system status"""
    return production_system.get_system_status()

async def production_health_check():
    """Perform production health check"""
    return await production_system.health_check()

# Export key components
__all__ = [
    'production_system',
    'initialize_production_system',
    'shutdown_production_system',
    'get_production_status',
    'production_health_check',
    'production_deployment',
    'gpu_infrastructure',
    'storage_manager',
    'monitoring_manager',
    'security_manager',
    'production_config'
]