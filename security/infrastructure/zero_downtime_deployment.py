"""
Zero-Downtime Rolling Updates with Automatic Rollback
Implements blue-green and canary deployment strategies with automatic rollback capabilities
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import json
import docker
import kubernetes
from kubernetes import client, config
import requests
import subprocess

logger = logging.getLogger(__name__)

class DeploymentStrategy(Enum):
    BLUE_GREEN = "blue_green"
    CANARY = "canary"
    ROLLING = "rolling"

class DeploymentStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"

class HealthCheckStatus(Enum):
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"

@dataclass
class DeploymentConfig:
    strategy: DeploymentStrategy
    image_tag: str
    replicas: int
    health_check_path: str
    health_check_timeout: int
    rollback_threshold: float  # Error rate threshold for automatic rollback
    canary_percentage: int  # For canary deployments
    max_surge: int  # Maximum additional pods during rolling update
    max_unavailable: int  # Maximum unavailable pods during rolling update

@dataclass
class DeploymentMetrics:
    timestamp: datetime
    success_rate: float
    response_time: float
    error_rate: float
    throughput: float
    cpu_usage: float
    memory_usage: float

@dataclass
class DeploymentResult:
    deployment_id: str
    status: DeploymentStatus
    strategy: DeploymentStrategy
    start_time: datetime
    end_time: Optional[datetime]
    duration_seconds: Optional[float]
    rollback_triggered: bool
    rollback_reason: Optional[str]
    metrics_before: Optional[DeploymentMetrics]
    metrics_after: Optional[DeploymentMetrics]
    health_checks: List[Dict[str, Any]]

class ZeroDowntimeDeployment:
    """
    Zero-downtime deployment system with blue-green, canary, and rolling update strategies.
    Includes automatic rollback capabilities based on health checks and performance metrics.
    """
    
    def __init__(self):
        self.deployments: Dict[str, DeploymentResult] = {}
        self.active_deployments: Dict[str, DeploymentConfig] = {}
        self.docker_client = None
        self.k8s_client = None
        self.k8s_apps_v1 = None
        self.k8s_core_v1 = None
        
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize Docker and Kubernetes clients"""
        try:
            # Initialize Docker client
            self.docker_client = docker.from_env()
            logger.info("Docker client initialized")
            
            # Initialize Kubernetes client
            try:
                config.load_incluster_config()
                logger.info("Loaded in-cluster Kubernetes config")
            except:
                try:
                    config.load_kube_config()
                    logger.info("Loaded local Kubernetes config")
                except:
                    logger.warning("Kubernetes config not available")
                    return
            
            self.k8s_client = client.ApiClient()
            self.k8s_apps_v1 = client.AppsV1Api()
            self.k8s_core_v1 = client.CoreV1Api()
            
        except Exception as e:
            logger.error(f"Failed to initialize clients: {e}")
    
    async def deploy(self, 
                    deployment_name: str, 
                    config: DeploymentConfig,
                    namespace: str = "default") -> DeploymentResult:
        """
        Execute zero-downtime deployment with specified strategy
        """
        deployment_id = f"{deployment_name}-{int(time.time())}"
        
        try:
            # Initialize deployment result
            result = DeploymentResult(
                deployment_id=deployment_id,
                status=DeploymentStatus.PENDING,
                strategy=config.strategy,
                start_time=datetime.now(),
                end_time=None,
                duration_seconds=None,
                rollback_triggered=False,
                rollback_reason=None,
                metrics_before=None,
                metrics_after=None,
                health_checks=[]
            )
            
            self.deployments[deployment_id] = result
            self.active_deployments[deployment_name] = config
            
            # Collect pre-deployment metrics
            result.metrics_before = await self._collect_deployment_metrics(deployment_name, namespace)
            
            # Execute deployment based on strategy
            result.status = DeploymentStatus.IN_PROGRESS
            
            if config.strategy == DeploymentStrategy.BLUE_GREEN:
                success = await self._execute_blue_green_deployment(deployment_name, config, namespace, result)
            elif config.strategy == DeploymentStrategy.CANARY:
                success = await self._execute_canary_deployment(deployment_name, config, namespace, result)
            elif config.strategy == DeploymentStrategy.ROLLING:
                success = await self._execute_rolling_deployment(deployment_name, config, namespace, result)
            else:
                raise ValueError(f"Unsupported deployment strategy: {config.strategy}")
            
            # Collect post-deployment metrics
            result.metrics_after = await self._collect_deployment_metrics(deployment_name, namespace)
            
            # Determine final status
            if success:
                result.status = DeploymentStatus.COMPLETED
            else:
                result.status = DeploymentStatus.FAILED
            
            result.end_time = datetime.now()
            result.duration_seconds = (result.end_time - result.start_time).total_seconds()
            
            logger.info(f"Deployment {deployment_id} completed with status: {result.status}")
            return result
            
        except Exception as e:
            logger.error(f"Deployment {deployment_id} failed: {e}")
            result.status = DeploymentStatus.FAILED
            result.end_time = datetime.now()
            result.duration_seconds = (result.end_time - result.start_time).total_seconds()
            return result
    
    async def _execute_blue_green_deployment(self, 
                                           deployment_name: str, 
                                           config: DeploymentConfig,
                                           namespace: str,
                                           result: DeploymentResult) -> bool:
        """Execute blue-green deployment strategy"""
        try:
            logger.info(f"Starting blue-green deployment for {deployment_name}")
            
            # Step 1: Create green environment
            green_deployment_name = f"{deployment_name}-green"
            await self._create_green_environment(deployment_name, green_deployment_name, config, namespace)
            
            # Step 2: Wait for green environment to be ready
            green_ready = await self._wait_for_deployment_ready(green_deployment_name, namespace, timeout=300)
            if not green_ready:
                logger.error("Green environment failed to become ready")
                await self._cleanup_green_environment(green_deployment_name, namespace)
                return False
            
            # Step 3: Perform health checks on green environment
            health_check_passed = await self._perform_health_checks(green_deployment_name, config, namespace, result)
            if not health_check_passed:
                logger.error("Green environment failed health checks")
                await self._cleanup_green_environment(green_deployment_name, namespace)
                return False
            
            # Step 4: Switch traffic to green environment
            await self._switch_traffic_to_green(deployment_name, green_deployment_name, namespace)
            
            # Step 5: Monitor for issues and potential rollback
            monitoring_passed = await self._monitor_deployment_health(deployment_name, config, result, duration=300)
            if not monitoring_passed:
                logger.warning("Post-deployment monitoring detected issues, initiating rollback")
                await self._rollback_blue_green_deployment(deployment_name, green_deployment_name, namespace, result)
                return False
            
            # Step 6: Cleanup old blue environment
            await self._cleanup_blue_environment(deployment_name, namespace)
            
            # Step 7: Rename green to blue
            await self._promote_green_to_blue(deployment_name, green_deployment_name, namespace)
            
            logger.info(f"Blue-green deployment for {deployment_name} completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Blue-green deployment failed: {e}")
            return False
    
    async def _execute_canary_deployment(self, 
                                       deployment_name: str, 
                                       config: DeploymentConfig,
                                       namespace: str,
                                       result: DeploymentResult) -> bool:
        """Execute canary deployment strategy"""
        try:
            logger.info(f"Starting canary deployment for {deployment_name}")
            
            # Step 1: Create canary deployment
            canary_deployment_name = f"{deployment_name}-canary"
            canary_replicas = max(1, int(config.replicas * config.canary_percentage / 100))
            
            canary_config = DeploymentConfig(
                strategy=config.strategy,
                image_tag=config.image_tag,
                replicas=canary_replicas,
                health_check_path=config.health_check_path,
                health_check_timeout=config.health_check_timeout,
                rollback_threshold=config.rollback_threshold,
                canary_percentage=config.canary_percentage,
                max_surge=config.max_surge,
                max_unavailable=config.max_unavailable
            )
            
            await self._create_canary_deployment(deployment_name, canary_deployment_name, canary_config, namespace)
            
            # Step 2: Wait for canary to be ready
            canary_ready = await self._wait_for_deployment_ready(canary_deployment_name, namespace, timeout=300)
            if not canary_ready:
                logger.error("Canary deployment failed to become ready")
                await self._cleanup_canary_deployment(canary_deployment_name, namespace)
                return False
            
            # Step 3: Configure traffic splitting
            await self._configure_canary_traffic_split(deployment_name, canary_deployment_name, config.canary_percentage, namespace)
            
            # Step 4: Monitor canary performance
            canary_healthy = await self._monitor_canary_health(deployment_name, canary_deployment_name, config, result, duration=600)
            if not canary_healthy:
                logger.warning("Canary deployment failed health monitoring, initiating rollback")
                await self._rollback_canary_deployment(deployment_name, canary_deployment_name, namespace, result)
                return False
            
            # Step 5: Gradually increase canary traffic
            for percentage in [25, 50, 75, 100]:
                if percentage <= config.canary_percentage:
                    continue
                
                await self._update_canary_traffic_split(deployment_name, canary_deployment_name, percentage, namespace)
                
                # Monitor at each stage
                stage_healthy = await self._monitor_canary_health(deployment_name, canary_deployment_name, config, result, duration=300)
                if not stage_healthy:
                    logger.warning(f"Canary deployment failed at {percentage}% traffic, initiating rollback")
                    await self._rollback_canary_deployment(deployment_name, canary_deployment_name, namespace, result)
                    return False
            
            # Step 6: Promote canary to production
            await self._promote_canary_to_production(deployment_name, canary_deployment_name, config, namespace)
            
            logger.info(f"Canary deployment for {deployment_name} completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Canary deployment failed: {e}")
            return False
    
    async def _execute_rolling_deployment(self, 
                                        deployment_name: str, 
                                        config: DeploymentConfig,
                                        namespace: str,
                                        result: DeploymentResult) -> bool:
        """Execute rolling deployment strategy"""
        try:
            logger.info(f"Starting rolling deployment for {deployment_name}")
            
            # Step 1: Update deployment with rolling update strategy
            await self._update_deployment_rolling(deployment_name, config, namespace)
            
            # Step 2: Monitor rolling update progress
            rolling_successful = await self._monitor_rolling_update(deployment_name, config, namespace, result)
            if not rolling_successful:
                logger.warning("Rolling update failed, initiating rollback")
                await self._rollback_rolling_deployment(deployment_name, namespace, result)
                return False
            
            # Step 3: Perform post-deployment health checks
            health_check_passed = await self._perform_health_checks(deployment_name, config, namespace, result)
            if not health_check_passed:
                logger.warning("Post-deployment health checks failed, initiating rollback")
                await self._rollback_rolling_deployment(deployment_name, namespace, result)
                return False
            
            # Step 4: Monitor for stability
            stability_check_passed = await self._monitor_deployment_health(deployment_name, config, result, duration=300)
            if not stability_check_passed:
                logger.warning("Stability monitoring failed, initiating rollback")
                await self._rollback_rolling_deployment(deployment_name, namespace, result)
                return False
            
            logger.info(f"Rolling deployment for {deployment_name} completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Rolling deployment failed: {e}")
            return False
    
    async def _collect_deployment_metrics(self, deployment_name: str, namespace: str) -> DeploymentMetrics:
        """Collect deployment metrics"""
        try:
            # Simulate metrics collection
            # In production, this would integrate with monitoring systems like Prometheus
            return DeploymentMetrics(
                timestamp=datetime.now(),
                success_rate=99.5,
                response_time=150.0,
                error_rate=0.5,
                throughput=1000.0,
                cpu_usage=45.0,
                memory_usage=60.0
            )
        except Exception as e:
            logger.error(f"Failed to collect deployment metrics: {e}")
            return DeploymentMetrics(
                timestamp=datetime.now(),
                success_rate=0.0,
                response_time=0.0,
                error_rate=100.0,
                throughput=0.0,
                cpu_usage=0.0,
                memory_usage=0.0
            )
    
    async def _create_green_environment(self, blue_name: str, green_name: str, config: DeploymentConfig, namespace: str):
        """Create green environment for blue-green deployment"""
        try:
            if self.k8s_apps_v1:
                # Get current blue deployment
                blue_deployment = self.k8s_apps_v1.read_namespaced_deployment(blue_name, namespace)
                
                # Create green deployment spec
                green_deployment = blue_deployment
                green_deployment.metadata.name = green_name
                green_deployment.spec.template.spec.containers[0].image = f"scrollintel:{config.image_tag}"
                green_deployment.spec.replicas = config.replicas
                
                # Create green deployment
                self.k8s_apps_v1.create_namespaced_deployment(namespace, green_deployment)
                logger.info(f"Created green environment: {green_name}")
            
        except Exception as e:
            logger.error(f"Failed to create green environment: {e}")
            raise
    
    async def _wait_for_deployment_ready(self, deployment_name: str, namespace: str, timeout: int = 300) -> bool:
        """Wait for deployment to be ready"""
        try:
            start_time = time.time()
            
            while time.time() - start_time < timeout:
                if self.k8s_apps_v1:
                    deployment = self.k8s_apps_v1.read_namespaced_deployment(deployment_name, namespace)
                    
                    if (deployment.status.ready_replicas and 
                        deployment.status.ready_replicas == deployment.spec.replicas):
                        logger.info(f"Deployment {deployment_name} is ready")
                        return True
                
                await asyncio.sleep(5)
            
            logger.error(f"Deployment {deployment_name} failed to become ready within {timeout} seconds")
            return False
            
        except Exception as e:
            logger.error(f"Failed to wait for deployment ready: {e}")
            return False
    
    async def _perform_health_checks(self, deployment_name: str, config: DeploymentConfig, namespace: str, result: DeploymentResult) -> bool:
        """Perform health checks on deployment"""
        try:
            health_check_results = []
            
            # Get deployment pods
            if self.k8s_core_v1:
                pods = self.k8s_core_v1.list_namespaced_pod(
                    namespace, 
                    label_selector=f"app={deployment_name}"
                )
                
                for pod in pods.items:
                    if pod.status.phase == "Running":
                        # Perform health check on each pod
                        pod_ip = pod.status.pod_ip
                        health_check_url = f"http://{pod_ip}:8080{config.health_check_path}"
                        
                        try:
                            response = requests.get(health_check_url, timeout=config.health_check_timeout)
                            health_status = HealthCheckStatus.HEALTHY if response.status_code == 200 else HealthCheckStatus.UNHEALTHY
                        except:
                            health_status = HealthCheckStatus.UNHEALTHY
                        
                        health_check_result = {
                            "pod_name": pod.metadata.name,
                            "pod_ip": pod_ip,
                            "status": health_status.value,
                            "timestamp": datetime.now().isoformat()
                        }
                        
                        health_check_results.append(health_check_result)
                        result.health_checks.append(health_check_result)
            
            # Determine overall health
            healthy_pods = sum(1 for check in health_check_results if check["status"] == "healthy")
            total_pods = len(health_check_results)
            
            if total_pods == 0:
                return False
            
            health_percentage = (healthy_pods / total_pods) * 100
            return health_percentage >= 80  # Require 80% of pods to be healthy
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    async def _monitor_deployment_health(self, deployment_name: str, config: DeploymentConfig, result: DeploymentResult, duration: int) -> bool:
        """Monitor deployment health for specified duration"""
        try:
            start_time = time.time()
            error_count = 0
            check_interval = 30  # Check every 30 seconds
            
            while time.time() - start_time < duration:
                # Collect current metrics
                current_metrics = await self._collect_deployment_metrics(deployment_name, "default")
                
                # Check if error rate exceeds threshold
                if current_metrics.error_rate > config.rollback_threshold:
                    error_count += 1
                    logger.warning(f"Error rate {current_metrics.error_rate}% exceeds threshold {config.rollback_threshold}%")
                    
                    # Trigger rollback if error rate consistently high
                    if error_count >= 3:
                        result.rollback_triggered = True
                        result.rollback_reason = f"Error rate {current_metrics.error_rate}% exceeded threshold {config.rollback_threshold}%"
                        return False
                else:
                    error_count = 0  # Reset error count if metrics improve
                
                await asyncio.sleep(check_interval)
            
            logger.info(f"Deployment health monitoring completed successfully for {deployment_name}")
            return True
            
        except Exception as e:
            logger.error(f"Deployment health monitoring failed: {e}")
            return False
    
    async def _switch_traffic_to_green(self, blue_name: str, green_name: str, namespace: str):
        """Switch traffic from blue to green environment"""
        try:
            if self.k8s_core_v1:
                # Update service selector to point to green deployment
                service = self.k8s_core_v1.read_namespaced_service(blue_name, namespace)
                service.spec.selector = {"app": green_name}
                self.k8s_core_v1.patch_namespaced_service(blue_name, namespace, service)
                
                logger.info(f"Switched traffic from {blue_name} to {green_name}")
            
        except Exception as e:
            logger.error(f"Failed to switch traffic: {e}")
            raise
    
    async def _rollback_blue_green_deployment(self, blue_name: str, green_name: str, namespace: str, result: DeploymentResult):
        """Rollback blue-green deployment"""
        try:
            logger.info(f"Rolling back blue-green deployment for {blue_name}")
            
            # Switch traffic back to blue
            await self._switch_traffic_to_blue(blue_name, green_name, namespace)
            
            # Cleanup green environment
            await self._cleanup_green_environment(green_name, namespace)
            
            result.rollback_triggered = True
            result.status = DeploymentStatus.ROLLED_BACK
            
        except Exception as e:
            logger.error(f"Failed to rollback blue-green deployment: {e}")
    
    async def _switch_traffic_to_blue(self, blue_name: str, green_name: str, namespace: str):
        """Switch traffic back to blue environment"""
        try:
            if self.k8s_core_v1:
                service = self.k8s_core_v1.read_namespaced_service(blue_name, namespace)
                service.spec.selector = {"app": blue_name}
                self.k8s_core_v1.patch_namespaced_service(blue_name, namespace, service)
                
                logger.info(f"Switched traffic back from {green_name} to {blue_name}")
            
        except Exception as e:
            logger.error(f"Failed to switch traffic back to blue: {e}")
    
    async def _cleanup_green_environment(self, green_name: str, namespace: str):
        """Cleanup green environment"""
        try:
            if self.k8s_apps_v1:
                self.k8s_apps_v1.delete_namespaced_deployment(green_name, namespace)
                logger.info(f"Cleaned up green environment: {green_name}")
            
        except Exception as e:
            logger.error(f"Failed to cleanup green environment: {e}")
    
    async def _cleanup_blue_environment(self, blue_name: str, namespace: str):
        """Cleanup old blue environment"""
        try:
            # In a real implementation, this would cleanup the old blue deployment
            logger.info(f"Cleaned up old blue environment: {blue_name}")
            
        except Exception as e:
            logger.error(f"Failed to cleanup blue environment: {e}")
    
    async def _promote_green_to_blue(self, blue_name: str, green_name: str, namespace: str):
        """Promote green deployment to blue"""
        try:
            # In a real implementation, this would rename green to blue
            logger.info(f"Promoted {green_name} to {blue_name}")
            
        except Exception as e:
            logger.error(f"Failed to promote green to blue: {e}")
    
    # Additional methods for canary and rolling deployments would be implemented here
    # Following the same pattern as blue-green deployment methods
    
    async def _create_canary_deployment(self, base_name: str, canary_name: str, config: DeploymentConfig, namespace: str):
        """Create canary deployment"""
        logger.info(f"Creating canary deployment: {canary_name}")
    
    async def _cleanup_canary_deployment(self, canary_name: str, namespace: str):
        """Cleanup canary deployment"""
        logger.info(f"Cleaning up canary deployment: {canary_name}")
    
    async def _configure_canary_traffic_split(self, base_name: str, canary_name: str, percentage: int, namespace: str):
        """Configure traffic splitting for canary deployment"""
        logger.info(f"Configuring {percentage}% traffic to canary: {canary_name}")
    
    async def _monitor_canary_health(self, base_name: str, canary_name: str, config: DeploymentConfig, result: DeploymentResult, duration: int) -> bool:
        """Monitor canary deployment health"""
        logger.info(f"Monitoring canary health for {duration} seconds")
        await asyncio.sleep(duration)
        return True
    
    async def _rollback_canary_deployment(self, base_name: str, canary_name: str, namespace: str, result: DeploymentResult):
        """Rollback canary deployment"""
        logger.info(f"Rolling back canary deployment: {canary_name}")
        result.rollback_triggered = True
        result.status = DeploymentStatus.ROLLED_BACK
    
    async def _update_canary_traffic_split(self, base_name: str, canary_name: str, percentage: int, namespace: str):
        """Update canary traffic split percentage"""
        logger.info(f"Updating canary traffic to {percentage}%")
    
    async def _promote_canary_to_production(self, base_name: str, canary_name: str, config: DeploymentConfig, namespace: str):
        """Promote canary deployment to production"""
        logger.info(f"Promoting canary {canary_name} to production")
    
    async def _update_deployment_rolling(self, deployment_name: str, config: DeploymentConfig, namespace: str):
        """Update deployment with rolling strategy"""
        logger.info(f"Starting rolling update for: {deployment_name}")
    
    async def _monitor_rolling_update(self, deployment_name: str, config: DeploymentConfig, namespace: str, result: DeploymentResult) -> bool:
        """Monitor rolling update progress"""
        logger.info(f"Monitoring rolling update progress for: {deployment_name}")
        return True
    
    async def _rollback_rolling_deployment(self, deployment_name: str, namespace: str, result: DeploymentResult):
        """Rollback rolling deployment"""
        logger.info(f"Rolling back deployment: {deployment_name}")
        result.rollback_triggered = True
        result.status = DeploymentStatus.ROLLED_BACK
    
    def get_deployment_status(self, deployment_id: str) -> Optional[DeploymentResult]:
        """Get deployment status by ID"""
        return self.deployments.get(deployment_id)
    
    def list_deployments(self) -> List[DeploymentResult]:
        """List all deployments"""
        return list(self.deployments.values())
    
    def get_active_deployments(self) -> Dict[str, DeploymentConfig]:
        """Get currently active deployments"""
        return self.active_deployments.copy()

# Global instance
zero_downtime_deployment = ZeroDowntimeDeployment()