#!/usr/bin/env python3
"""
ScrollIntel Agent Steering System - Production Deployment and Launch
Enterprise-grade production deployment with monitoring, UAT, and gradual rollout
"""

import os
import sys
import json
import time
import logging
import subprocess
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import yaml
import psycopg2
import redis
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/production-deployment.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class DeploymentConfig:
    """Production deployment configuration"""
    environment: str = "production"
    deployment_type: str = "gradual"  # gradual, blue_green, canary
    enable_monitoring: bool = True
    enable_uat: bool = True
    enable_feature_flags: bool = True
    rollback_on_failure: bool = True
    health_check_timeout: int = 300
    uat_timeout: int = 1800  # 30 minutes
    canary_percentage: int = 10
    gradual_rollout_stages: List[int] = None
    
    def __post_init__(self):
        if self.gradual_rollout_stages is None:
            self.gradual_rollout_stages = [10, 25, 50, 75, 100]

@dataclass
class DeploymentStatus:
    """Track deployment status"""
    stage: str
    status: str  # pending, in_progress, completed, failed
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error_message: Optional[str] = None
    metrics: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = {}

class ProductionDeploymentManager:
    """Manages enterprise-grade production deployment and launch"""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.deployment_id = f"deploy_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.status_tracker = {}
        self.backup_path = f"backups/{self.deployment_id}"
        self.monitoring_enabled = False
        
        # Create necessary directories
        os.makedirs("logs", exist_ok=True)
        os.makedirs(self.backup_path, exist_ok=True)
        
        logger.info(f"Initialized production deployment manager: {self.deployment_id}")
    
    def deploy_to_production(self) -> bool:
        """Execute complete production deployment with monitoring and UAT"""
        try:
            logger.info("🚀 Starting enterprise production deployment...")
            
            # Stage 1: Pre-deployment validation
            if not self._pre_deployment_validation():
                return False
            
            # Stage 2: Create system backup
            if not self._create_system_backup():
                return False
            
            # Stage 3: Deploy monitoring infrastructure
            if self.config.enable_monitoring:
                if not self._deploy_monitoring_infrastructure():
                    return False
            
            # Stage 4: Deploy application with chosen strategy
            if not self._deploy_application():
                return False
            
            # Stage 5: Run user acceptance testing
            if self.config.enable_uat:
                if not self._run_user_acceptance_testing():
                    return False
            
            # Stage 6: Execute gradual rollout
            if self.config.deployment_type == "gradual":
                if not self._execute_gradual_rollout():
                    return False
            
            # Stage 7: Final validation and go-live
            if not self._final_validation_and_golive():
                return False
            
            # Stage 8: Post-deployment monitoring
            self._setup_post_deployment_monitoring()
            
            logger.info("✅ Production deployment completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Production deployment failed: {str(e)}")
            if self.config.rollback_on_failure:
                self._rollback_deployment()
            return False
    
    def _pre_deployment_validation(self) -> bool:
        """Comprehensive pre-deployment validation"""
        logger.info("🔍 Running pre-deployment validation...")
        
        self._update_status("pre_validation", "in_progress")
        
        try:
            # Validate environment variables
            required_vars = [
                "DATABASE_URL", "REDIS_URL", "JWT_SECRET_KEY",
                "OPENAI_API_KEY", "MONITORING_WEBHOOK_URL"
            ]
            
            missing_vars = []
            for var in required_vars:
                if not os.getenv(var):
                    missing_vars.append(var)
            
            if missing_vars:
                raise ValueError(f"Missing required environment variables: {missing_vars}")
            
            # Test database connectivity
            self._test_database_connection()
            
            # Test Redis connectivity
            self._test_redis_connection()
            
            # Validate Docker images
            self._validate_docker_images()
            
            # Run security checks
            self._run_security_checks()
            
            # Validate Kubernetes cluster (if applicable)
            if self._is_kubernetes_deployment():
                self._validate_kubernetes_cluster()
            
            self._update_status("pre_validation", "completed")
            logger.info("✅ Pre-deployment validation passed")
            return True
            
        except Exception as e:
            self._update_status("pre_validation", "failed", str(e))
            logger.error(f"❌ Pre-deployment validation failed: {str(e)}")
            return False
    
    def _create_system_backup(self) -> bool:
        """Create comprehensive system backup"""
        logger.info("💾 Creating system backup...")
        
        self._update_status("backup", "in_progress")
        
        try:
            # Backup database
            db_backup_path = os.path.join(self.backup_path, "database_backup.sql")
            self._backup_database(db_backup_path)
            
            # Backup uploaded files
            uploads_backup_path = os.path.join(self.backup_path, "uploads_backup.tar.gz")
            self._backup_uploads(uploads_backup_path)
            
            # Backup configuration
            config_backup_path = os.path.join(self.backup_path, "config_backup.json")
            self._backup_configuration(config_backup_path)
            
            # Backup models and data
            models_backup_path = os.path.join(self.backup_path, "models_backup.tar.gz")
            self._backup_models(models_backup_path)
            
            self._update_status("backup", "completed")
            logger.info(f"✅ System backup created at {self.backup_path}")
            return True
            
        except Exception as e:
            self._update_status("backup", "failed", str(e))
            logger.error(f"❌ System backup failed: {str(e)}")
            return False
    
    def _deploy_monitoring_infrastructure(self) -> bool:
        """Deploy comprehensive monitoring infrastructure"""
        logger.info("📊 Deploying monitoring infrastructure...")
        
        self._update_status("monitoring_deployment", "in_progress")
        
        try:
            # Deploy Prometheus
            self._deploy_prometheus()
            
            # Deploy Grafana with dashboards
            self._deploy_grafana()
            
            # Deploy AlertManager
            self._deploy_alertmanager()
            
            # Deploy custom monitoring agents
            self._deploy_custom_monitoring()
            
            # Verify monitoring stack
            self._verify_monitoring_stack()
            
            self.monitoring_enabled = True
            self._update_status("monitoring_deployment", "completed")
            logger.info("✅ Monitoring infrastructure deployed successfully")
            return True
            
        except Exception as e:
            self._update_status("monitoring_deployment", "failed", str(e))
            logger.error(f"❌ Monitoring deployment failed: {str(e)}")
            return False
    
    def _deploy_application(self) -> bool:
        """Deploy application using specified strategy"""
        logger.info(f"🚀 Deploying application using {self.config.deployment_type} strategy...")
        
        self._update_status("app_deployment", "in_progress")
        
        try:
            if self.config.deployment_type == "blue_green":
                return self._deploy_blue_green()
            elif self.config.deployment_type == "canary":
                return self._deploy_canary()
            else:
                return self._deploy_standard()
                
        except Exception as e:
            self._update_status("app_deployment", "failed", str(e))
            logger.error(f"❌ Application deployment failed: {str(e)}")
            return False
    
    def _deploy_blue_green(self) -> bool:
        """Deploy using blue-green strategy"""
        logger.info("🔵🟢 Executing blue-green deployment...")
        
        try:
            # Deploy to green environment
            self._deploy_to_environment("green")
            
            # Run health checks on green
            if not self._health_check_environment("green"):
                raise Exception("Green environment health checks failed")
            
            # Switch traffic to green
            self._switch_traffic("green")
            
            # Verify traffic switch
            if not self._verify_traffic_switch("green"):
                raise Exception("Traffic switch verification failed")
            
            # Cleanup blue environment
            self._cleanup_environment("blue")
            
            self._update_status("app_deployment", "completed")
            return True
            
        except Exception as e:
            logger.error(f"Blue-green deployment failed: {str(e)}")
            # Rollback to blue if green fails
            self._switch_traffic("blue")
            return False
    
    def _deploy_canary(self) -> bool:
        """Deploy using canary strategy"""
        logger.info("🐤 Executing canary deployment...")
        
        try:
            # Deploy canary version
            self._deploy_canary_version()
            
            # Route small percentage of traffic to canary
            self._route_canary_traffic(self.config.canary_percentage)
            
            # Monitor canary metrics
            if not self._monitor_canary_metrics():
                raise Exception("Canary metrics indicate issues")
            
            # Gradually increase canary traffic
            for percentage in [25, 50, 75, 100]:
                self._route_canary_traffic(percentage)
                time.sleep(300)  # Wait 5 minutes between increases
                
                if not self._monitor_canary_metrics():
                    raise Exception(f"Canary failed at {percentage}% traffic")
            
            # Promote canary to production
            self._promote_canary_to_production()
            
            self._update_status("app_deployment", "completed")
            return True
            
        except Exception as e:
            logger.error(f"Canary deployment failed: {str(e)}")
            # Rollback canary
            self._rollback_canary()
            return False
    
    def _deploy_standard(self) -> bool:
        """Deploy using standard rolling update strategy"""
        logger.info("📦 Executing standard deployment...")
        
        try:
            # Build and push Docker images
            self._build_and_push_images()
            
            # Update Kubernetes deployments
            if self._is_kubernetes_deployment():
                self._update_kubernetes_deployments()
            else:
                self._update_docker_compose_deployment()
            
            # Wait for rollout to complete
            self._wait_for_rollout_completion()
            
            # Run post-deployment health checks
            if not self._comprehensive_health_check():
                raise Exception("Post-deployment health checks failed")
            
            self._update_status("app_deployment", "completed")
            return True
            
        except Exception as e:
            logger.error(f"Standard deployment failed: {str(e)}")
            return False
    
    def _run_user_acceptance_testing(self) -> bool:
        """Run comprehensive user acceptance testing"""
        logger.info("👥 Running user acceptance testing...")
        
        self._update_status("uat", "in_progress")
        
        try:
            # Initialize UAT framework
            uat_results = {}
            
            # Test critical user journeys
            uat_results["critical_journeys"] = self._test_critical_user_journeys()
            
            # Test agent interactions
            uat_results["agent_interactions"] = self._test_agent_interactions()
            
            # Test data processing workflows
            uat_results["data_workflows"] = self._test_data_processing_workflows()
            
            # Test security and compliance
            uat_results["security_compliance"] = self._test_security_compliance()
            
            # Test performance under load
            uat_results["performance"] = self._test_performance_load()
            
            # Test integration points
            uat_results["integrations"] = self._test_integration_points()
            
            # Evaluate UAT results
            if not self._evaluate_uat_results(uat_results):
                raise Exception("UAT failed - critical issues detected")
            
            # Generate UAT report
            self._generate_uat_report(uat_results)
            
            self._update_status("uat", "completed", metrics=uat_results)
            logger.info("✅ User acceptance testing completed successfully")
            return True
            
        except Exception as e:
            self._update_status("uat", "failed", str(e))
            logger.error(f"❌ User acceptance testing failed: {str(e)}")
            return False
    
    def _execute_gradual_rollout(self) -> bool:
        """Execute gradual rollout with feature flags"""
        logger.info("📈 Executing gradual rollout...")
        
        self._update_status("gradual_rollout", "in_progress")
        
        try:
            for stage, percentage in enumerate(self.config.gradual_rollout_stages):
                logger.info(f"Rolling out to {percentage}% of users (Stage {stage + 1})")
                
                # Update feature flags
                self._update_feature_flags(percentage)
                
                # Monitor metrics for this stage
                stage_metrics = self._monitor_rollout_stage(percentage)
                
                # Evaluate stage success
                if not self._evaluate_rollout_stage(stage_metrics):
                    raise Exception(f"Rollout stage {percentage}% failed")
                
                # Wait before next stage (except for 100%)
                if percentage < 100:
                    logger.info(f"Waiting 10 minutes before next stage...")
                    time.sleep(600)
            
            self._update_status("gradual_rollout", "completed")
            logger.info("✅ Gradual rollout completed successfully")
            return True
            
        except Exception as e:
            self._update_status("gradual_rollout", "failed", str(e))
            logger.error(f"❌ Gradual rollout failed: {str(e)}")
            return False
    
    def _final_validation_and_golive(self) -> bool:
        """Final validation and go-live procedures"""
        logger.info("🎯 Running final validation and go-live procedures...")
        
        self._update_status("go_live", "in_progress")
        
        try:
            # Final comprehensive health check
            if not self._final_comprehensive_health_check():
                raise Exception("Final health check failed")
            
            # Validate all integrations
            if not self._validate_all_integrations():
                raise Exception("Integration validation failed")
            
            # Performance validation
            if not self._validate_production_performance():
                raise Exception("Performance validation failed")
            
            # Security validation
            if not self._validate_production_security():
                raise Exception("Security validation failed")
            
            # Generate go-live documentation
            self._generate_golive_documentation()
            
            # Send go-live notifications
            self._send_golive_notifications()
            
            # Update DNS and load balancer (if needed)
            self._update_production_routing()
            
            self._update_status("go_live", "completed")
            logger.info("🎉 System is now live in production!")
            return True
            
        except Exception as e:
            self._update_status("go_live", "failed", str(e))
            logger.error(f"❌ Go-live procedures failed: {str(e)}")
            return False
    
    def _setup_post_deployment_monitoring(self):
        """Setup comprehensive post-deployment monitoring"""
        logger.info("📊 Setting up post-deployment monitoring...")
        
        try:
            # Configure alerting rules
            self._configure_alerting_rules()
            
            # Setup automated health checks
            self._setup_automated_health_checks()
            
            # Configure performance monitoring
            self._configure_performance_monitoring()
            
            # Setup business metrics tracking
            self._setup_business_metrics_tracking()
            
            # Configure log aggregation
            self._configure_log_aggregation()
            
            # Setup automated reporting
            self._setup_automated_reporting()
            
            logger.info("✅ Post-deployment monitoring configured")
            
        except Exception as e:
            logger.error(f"❌ Post-deployment monitoring setup failed: {str(e)}")
    
    def _test_critical_user_journeys(self) -> Dict[str, bool]:
        """Test critical user journeys"""
        logger.info("Testing critical user journeys...")
        
        journeys = {
            "user_registration": False,
            "user_login": False,
            "agent_interaction": False,
            "data_upload": False,
            "report_generation": False,
            "dashboard_access": False
        }
        
        try:
            # Test user registration
            journeys["user_registration"] = self._test_user_registration()
            
            # Test user login
            journeys["user_login"] = self._test_user_login()
            
            # Test agent interaction
            journeys["agent_interaction"] = self._test_agent_interaction()
            
            # Test data upload
            journeys["data_upload"] = self._test_data_upload()
            
            # Test report generation
            journeys["report_generation"] = self._test_report_generation()
            
            # Test dashboard access
            journeys["dashboard_access"] = self._test_dashboard_access()
            
        except Exception as e:
            logger.error(f"Critical user journey testing failed: {str(e)}")
        
        return journeys
    
    def _comprehensive_health_check(self) -> bool:
        """Run comprehensive health check"""
        logger.info("Running comprehensive health check...")
        
        try:
            # Check API endpoints
            api_health = self._check_api_health()
            
            # Check database health
            db_health = self._check_database_health()
            
            # Check Redis health
            redis_health = self._check_redis_health()
            
            # Check agent health
            agent_health = self._check_agent_health()
            
            # Check integration health
            integration_health = self._check_integration_health()
            
            all_healthy = all([
                api_health, db_health, redis_health, 
                agent_health, integration_health
            ])
            
            if all_healthy:
                logger.info("✅ All health checks passed")
            else:
                logger.error("❌ Some health checks failed")
            
            return all_healthy
            
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return False
    
    def _rollback_deployment(self):
        """Rollback deployment to previous state"""
        logger.info("🔄 Rolling back deployment...")
        
        try:
            # Restore database from backup
            self._restore_database_backup()
            
            # Restore uploaded files
            self._restore_uploads_backup()
            
            # Restore configuration
            self._restore_configuration_backup()
            
            # Revert application deployment
            self._revert_application_deployment()
            
            # Verify rollback
            if self._comprehensive_health_check():
                logger.info("✅ Rollback completed successfully")
            else:
                logger.error("❌ Rollback verification failed")
                
        except Exception as e:
            logger.error(f"❌ Rollback failed: {str(e)}")
    
    def _update_status(self, stage: str, status: str, error_message: str = None, metrics: Dict = None):
        """Update deployment status"""
        if stage not in self.status_tracker:
            self.status_tracker[stage] = DeploymentStatus(stage=stage, status=status)
        
        self.status_tracker[stage].status = status
        if status == "in_progress" and not self.status_tracker[stage].start_time:
            self.status_tracker[stage].start_time = datetime.now()
        elif status in ["completed", "failed"]:
            self.status_tracker[stage].end_time = datetime.now()
        
        if error_message:
            self.status_tracker[stage].error_message = error_message
        if metrics:
            self.status_tracker[stage].metrics = metrics
        
        # Save status to file
        self._save_deployment_status()
    
    def _save_deployment_status(self):
        """Save deployment status to file"""
        status_file = f"logs/deployment_status_{self.deployment_id}.json"
        
        status_data = {
            "deployment_id": self.deployment_id,
            "config": asdict(self.config),
            "stages": {
                stage: {
                    "stage": status.stage,
                    "status": status.status,
                    "start_time": status.start_time.isoformat() if status.start_time else None,
                    "end_time": status.end_time.isoformat() if status.end_time else None,
                    "error_message": status.error_message,
                    "metrics": status.metrics
                }
                for stage, status in self.status_tracker.items()
            }
        }
        
        with open(status_file, 'w') as f:
            json.dump(status_data, f, indent=2)
    
    # Helper methods for specific operations
    def _test_database_connection(self):
        """Test database connectivity"""
        try:
            conn = psycopg2.connect(os.getenv("DATABASE_URL"))
            conn.close()
            logger.info("✅ Database connection test passed")
        except Exception as e:
            raise Exception(f"Database connection failed: {str(e)}")
    
    def _test_redis_connection(self):
        """Test Redis connectivity"""
        try:
            r = redis.from_url(os.getenv("REDIS_URL"))
            r.ping()
            logger.info("✅ Redis connection test passed")
        except Exception as e:
            raise Exception(f"Redis connection failed: {str(e)}")
    
    def _validate_docker_images(self):
        """Validate Docker images exist and are accessible"""
        images = [
            "scrollintel-backend:latest",
            "scrollintel-frontend:latest"
        ]
        
        for image in images:
            result = subprocess.run(
                ["docker", "image", "inspect", image],
                capture_output=True, text=True
            )
            if result.returncode != 0:
                raise Exception(f"Docker image {image} not found")
        
        logger.info("✅ Docker image validation passed")
    
    def _is_kubernetes_deployment(self) -> bool:
        """Check if this is a Kubernetes deployment"""
        return os.path.exists("k8s/") or os.getenv("KUBERNETES_DEPLOYMENT") == "true"
    
    def _backup_database(self, backup_path: str):
        """Backup database"""
        cmd = f"pg_dump {os.getenv('DATABASE_URL')} > {backup_path}"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception(f"Database backup failed: {result.stderr}")
    
    def _backup_uploads(self, backup_path: str):
        """Backup uploaded files"""
        if os.path.exists("uploads"):
            cmd = f"tar -czf {backup_path} uploads/"
            subprocess.run(cmd, shell=True, check=True)
    
    def _backup_configuration(self, backup_path: str):
        """Backup configuration"""
        config_data = {
            "environment_vars": dict(os.environ),
            "deployment_config": asdict(self.config)
        }
        with open(backup_path, 'w') as f:
            json.dump(config_data, f, indent=2)
    
    def _backup_models(self, backup_path: str):
        """Backup ML models"""
        if os.path.exists("models"):
            cmd = f"tar -czf {backup_path} models/"
            subprocess.run(cmd, shell=True, check=True)

def main():
    """Main deployment function"""
    # Check if running as part of orchestrator
    deployment_phase = os.getenv("DEPLOYMENT_PHASE")
    
    if deployment_phase == "pre_deployment":
        # Run only pre-deployment validation when called by orchestrator
        config = DeploymentConfig(
            deployment_type=os.getenv("DEPLOYMENT_TYPE", "gradual"),
            enable_monitoring=False,  # Monitoring setup is separate phase
            enable_uat=False,  # UAT is separate phase
            rollback_on_failure=os.getenv("ROLLBACK_ON_FAILURE", "true").lower() == "true"
        )
        
        deployment_manager = ProductionDeploymentManager(config)
        
        # Run only pre-deployment steps
        success = (
            deployment_manager._pre_deployment_validation() and
            deployment_manager._create_system_backup()
        )
        
        if success:
            print("✅ Pre-deployment validation and backup completed successfully!")
            sys.exit(0)
        else:
            print("❌ Pre-deployment validation failed!")
            sys.exit(1)
    
    else:
        # Run full deployment (legacy mode)
        config = DeploymentConfig(
            deployment_type=os.getenv("DEPLOYMENT_TYPE", "gradual"),
            enable_monitoring=os.getenv("ENABLE_MONITORING", "true").lower() == "true",
            enable_uat=os.getenv("ENABLE_UAT", "true").lower() == "true",
            rollback_on_failure=os.getenv("ROLLBACK_ON_FAILURE", "true").lower() == "true"
        )
        
        # Initialize deployment manager
        deployment_manager = ProductionDeploymentManager(config)
        
        # Execute deployment
        success = deployment_manager.deploy_to_production()
        
        if success:
            print("🎉 Production deployment completed successfully!")
            sys.exit(0)
        else:
            print("❌ Production deployment failed!")
            sys.exit(1)

if __name__ == "__main__":
    main()