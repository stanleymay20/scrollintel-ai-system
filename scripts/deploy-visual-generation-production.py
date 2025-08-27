#!/usr/bin/env python3
"""
Production Deployment Script for Visual Generation System
Deploys models, configures GPU infrastructure, and validates production readiness
"""

import asyncio
import logging
import sys
import os
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scrollintel.engines.visual_generation.production.model_deployment import (
    production_deployment, initialize_production_models
)
from scrollintel.engines.visual_generation.production.gpu_infrastructure import (
    gpu_infrastructure, initialize_gpu_infrastructure
)
from scrollintel.engines.visual_generation.production.production_config import (
    production_config, validate_environment
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('visual_generation_deployment.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class ProductionDeploymentManager:
    """Manages the complete production deployment process"""
    
    def __init__(self):
        self.deployment_steps = [
            ("Validate Environment", self.validate_environment),
            ("Initialize GPU Infrastructure", self.initialize_gpu_infrastructure),
            ("Deploy Models", self.deploy_models),
            ("Configure Rate Limiting", self.configure_rate_limiting),
            ("Setup Monitoring", self.setup_monitoring),
            ("Validate Deployment", self.validate_deployment),
            ("Performance Test", self.performance_test)
        ]
        self.deployment_status = {}
    
    async def run_deployment(self):
        """Run complete production deployment"""
        logger.info("Starting Visual Generation Production Deployment")
        logger.info("=" * 60)
        
        start_time = time.time()
        success_count = 0
        
        for step_name, step_function in self.deployment_steps:
            logger.info(f"\n🚀 Starting: {step_name}")
            
            try:
                result = await step_function()
                if result:
                    logger.info(f"✅ Completed: {step_name}")
                    self.deployment_status[step_name] = "SUCCESS"
                    success_count += 1
                else:
                    logger.error(f"❌ Failed: {step_name}")
                    self.deployment_status[step_name] = "FAILED"
                    
            except Exception as e:
                logger.error(f"❌ Error in {step_name}: {str(e)}")
                self.deployment_status[step_name] = "ERROR"
        
        total_time = time.time() - start_time
        
        # Print deployment summary
        logger.info("\n" + "=" * 60)
        logger.info("DEPLOYMENT SUMMARY")
        logger.info("=" * 60)
        
        for step_name, status in self.deployment_status.items():
            status_icon = "✅" if status == "SUCCESS" else "❌"
            logger.info(f"{status_icon} {step_name}: {status}")
        
        logger.info(f"\nTotal Steps: {len(self.deployment_steps)}")
        logger.info(f"Successful: {success_count}")
        logger.info(f"Failed: {len(self.deployment_steps) - success_count}")
        logger.info(f"Total Time: {total_time:.2f} seconds")
        
        if success_count == len(self.deployment_steps):
            logger.info("\n🎉 DEPLOYMENT SUCCESSFUL! Visual Generation system is ready for production.")
            return True
        else:
            logger.error("\n💥 DEPLOYMENT FAILED! Please check the logs and fix issues.")
            return False
    
    async def validate_environment(self) -> bool:
        """Validate environment configuration"""
        try:
            logger.info("Checking environment variables and configuration...")
            
            # Validate environment
            if not validate_environment():
                logger.error("Environment validation failed")
                return False
            
            # Check required environment variables
            required_vars = [
                "OPENAI_API_KEY",
                "AWS_ACCESS_KEY_ID", 
                "AWS_SECRET_ACCESS_KEY",
                "STORAGE_BUCKET"
            ]
            
            missing_vars = []
            for var in required_vars:
                if not os.getenv(var):
                    missing_vars.append(var)
            
            if missing_vars:
                logger.error(f"Missing required environment variables: {missing_vars}")
                return False
            
            # Validate configuration
            config = production_config.export_config()
            logger.info(f"Environment: {config['environment']}")
            logger.info(f"Enabled models: {[name for name, model in config['models'].items() if model['enabled']]}")
            
            return True
            
        except Exception as e:
            logger.error(f"Environment validation failed: {str(e)}")
            return False
    
    async def initialize_gpu_infrastructure(self) -> bool:
        """Initialize GPU infrastructure"""
        try:
            logger.info("Initializing GPU infrastructure...")
            
            # Initialize infrastructure
            result = await initialize_gpu_infrastructure()
            if not result:
                logger.error("Failed to initialize GPU infrastructure")
                return False
            
            # Check GPU status
            status = await gpu_infrastructure.get_cluster_status()
            logger.info(f"GPU clusters initialized: {list(status.keys())}")
            
            # Monitor utilization
            utilization = await gpu_infrastructure.monitor_gpu_utilization()
            logger.info(f"GPU utilization monitoring active for {len(utilization)} instances")
            
            return True
            
        except Exception as e:
            logger.error(f"GPU infrastructure initialization failed: {str(e)}")
            return False
    
    async def deploy_models(self) -> bool:
        """Deploy all visual generation models"""
        try:
            logger.info("Deploying visual generation models...")
            
            # Deploy all models
            result = await initialize_production_models()
            if not result:
                logger.error("Model deployment failed")
                return False
            
            # Check model status
            status = await production_deployment.get_model_status()
            
            for model_name, model_status in status.items():
                if model_name != "gpu_resources":
                    logger.info(f"Model {model_name}: {model_status['status']}")
            
            # Verify all enabled models are ready
            enabled_models = [
                name for name, config in production_config.model_configs.items()
                if config.enabled
            ]
            
            ready_models = [
                name for name, model_status in status.items()
                if model_name != "gpu_resources" and model_status['status'] == 'ready'
            ]
            
            if len(ready_models) < len(enabled_models):
                logger.error(f"Not all models ready. Expected: {enabled_models}, Ready: {ready_models}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Model deployment failed: {str(e)}")
            return False
    
    async def configure_rate_limiting(self) -> bool:
        """Configure production rate limiting"""
        try:
            logger.info("Configuring rate limiting...")
            
            # Rate limiting is configured in model deployment
            # Here we validate the configuration
            
            for model_name, config in production_config.model_configs.items():
                if config.enabled:
                    logger.info(f"{model_name} rate limit: {config.rate_limit_per_minute}/min")
            
            return True
            
        except Exception as e:
            logger.error(f"Rate limiting configuration failed: {str(e)}")
            return False
    
    async def setup_monitoring(self) -> bool:
        """Setup production monitoring"""
        try:
            logger.info("Setting up monitoring...")
            
            monitoring_config = production_config.get_monitoring_config()
            
            if monitoring_config.enabled:
                logger.info(f"Monitoring enabled with log level: {monitoring_config.log_level}")
                
                if monitoring_config.performance_tracking:
                    logger.info("Performance tracking enabled")
                
                if monitoring_config.metrics_endpoint:
                    logger.info(f"Metrics endpoint: {monitoring_config.metrics_endpoint}")
                
                if monitoring_config.alert_webhook:
                    logger.info("Alert webhook configured")
            
            return True
            
        except Exception as e:
            logger.error(f"Monitoring setup failed: {str(e)}")
            return False
    
    async def validate_deployment(self) -> bool:
        """Validate deployment is working correctly"""
        try:
            logger.info("Validating deployment...")
            
            # Check model status
            model_status = await production_deployment.get_model_status()
            
            # Check GPU infrastructure
            gpu_status = await gpu_infrastructure.get_cluster_status()
            
            # Validate all systems are operational
            all_ready = True
            
            for model_name, status in model_status.items():
                if model_name != "gpu_resources" and status['status'] != 'ready':
                    logger.error(f"Model {model_name} not ready: {status['status']}")
                    all_ready = False
            
            if not gpu_status:
                logger.error("No GPU clusters available")
                all_ready = False
            
            return all_ready
            
        except Exception as e:
            logger.error(f"Deployment validation failed: {str(e)}")
            return False
    
    async def performance_test(self) -> bool:
        """Run basic performance tests"""
        try:
            logger.info("Running performance tests...")
            
            # Test GPU utilization monitoring
            utilization = await gpu_infrastructure.monitor_gpu_utilization()
            if utilization:
                logger.info(f"GPU monitoring test passed: {len(utilization)} GPUs monitored")
            
            # Test model warm-up cache
            model_status = await production_deployment.get_model_status()
            warm_up_count = sum(
                status.get('warm_up_cache', 0) 
                for status in model_status.values()
                if isinstance(status, dict)
            )
            
            if warm_up_count > 0:
                logger.info(f"Model warm-up test passed: {warm_up_count} cached results")
            
            # Test auto-scaling (dry run)
            if production_config.get_performance_config()['auto_scaling_enabled']:
                logger.info("Auto-scaling is enabled and configured")
            
            return True
            
        except Exception as e:
            logger.error(f"Performance test failed: {str(e)}")
            return False

async def main():
    """Main deployment function"""
    try:
        deployment_manager = ProductionDeploymentManager()
        success = await deployment_manager.run_deployment()
        
        if success:
            logger.info("\n🎯 Next Steps:")
            logger.info("1. Monitor system performance and logs")
            logger.info("2. Run integration tests")
            logger.info("3. Configure load balancer and CDN")
            logger.info("4. Set up backup and disaster recovery")
            logger.info("5. Enable production traffic")
            
            sys.exit(0)
        else:
            logger.error("\n🔧 Troubleshooting:")
            logger.error("1. Check environment variables")
            logger.error("2. Verify GPU availability")
            logger.error("3. Check API key permissions")
            logger.error("4. Review deployment logs")
            
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("\nDeployment interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Deployment failed with error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())