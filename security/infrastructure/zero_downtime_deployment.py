"""
Zero-Downtime Rolling Updates with Automatic Rollback
Implements blue-green, canary, and rolling deployment strategies
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import time

logger = logging.getLogger(__name__)

class DeploymentStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLING_BACK = "rolling_back"
    ROLLED_BACK = "rolled_back"

class DeploymentStrategy(Enum):
    ROLLING = "rolling"
    BLUE_GREEN = "blue_green"
    CANARY = "canary"

class HealthCheckStatus(Enum):
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"

@dataclass
class DeploymentConfig:
    """Configuration for zero-downtime deployment"""
    strategy: DeploymentStrategy
    application_name: str
    version: str
    replicas: int
    health_check_url: str
    health_check_timeout: int = 30
    rollback_threshold: float = 0.1  # 10% error rate triggers rollback
    canary_percentage: int = 10  # For canary deployments
    max_unavailable: int = 1  # For rolling deployments
    deployment_timeout: int = 600  # 10 minutes
    
@dataclass
class DeploymentMetrics:
    """Metrics collected during deployment"""
    timestamp: datetime
    success_rate: float
    error_rate: float
    response_time: float
    throughput: float
    cpu_utilization: float
    memory_utilization: float
    active_connections: int

@dataclass
class DeploymentResult:
    """Result of a deployment operation"""
    deployment_id: str
    status: DeploymentStatus
    strategy: DeploymentStrategy
    start_time: datetime
    end_time: Optional[datetime]
    success: bool
    error_message: Optional[str]
    metrics: List[DeploymentMetrics]
    rollback_triggered: bool
    rollback_reason: Optional[str]

class ZeroDowntimeDeployment:
    """
    Zero-Downtime Deployment System
    Provides rolling updates, blue-green deployments, canary releases
    with automatic rollback capabilities
    """
    
    def __init__(self):
        self.active_deployments: Dict[str, DeploymentResult] = {}
        self.deployment_history: List[DeploymentResult] = []
        self.health_check_interval = 10  # seconds
        self.metrics_collection_interval = 5  # seconds
        
        # Rollback thresholds
        self.error_rate_threshold = 0.05  # 5% error rate
        self.response_time_threshold = 1000  # 1 second
        self.success_rate_threshold = 0.95  # 95% success rate
        
    async def deploy_application(self, config: DeploymentConfig) -> DeploymentResult:
        """Deploy application using specified strategy with zero downtime"""
        deployment_id = f"{config.application_name}-{config.version}-{int(time.time())}"
        
        deployment_result = DeploymentResult(
            deployment_id=deployment_id,
            status=DeploymentStatus.PENDING,
            strategy=config.strategy,
            start_time=datetime.now(),
            end_time=None,
            success=False,
            error_message=None,
            metrics=[],
            rollback_triggered=False,
            rollback_reason=None
        )
        
        self.active_deployments[deployment_id] = deployment_result
        
        try:
            logger.info(f"Starting {config.strategy.value} deployment: {deployment_id}")
            deployment_result.status = DeploymentStatus.IN_PROGRESS
            
            # Execute deployment based on strategy
            if config.strategy == DeploymentStrategy.ROLLING:
                success = await self._rolling_deployment(config, deployment_result)
            elif config.strategy == DeploymentStrategy.BLUE_GREEN:
                success = await self._blue_green_deployment(config, deployment_result)
            elif config.strategy == DeploymentStrategy.CANARY:
                success = await self._canary_deployment(config, deployment_result)
            else:
                raise ValueError(f"Unsupported deployment strategy: {config.strategy}")
            
            deployment_result.success = success
            deployment_result.status = DeploymentStatus.COMPLETED if success else DeploymentStatus.FAILED
            deployment_result.end_time = datetime.now()
            
            logger.info(f"Deployment {deployment_id} {'completed successfully' if success else 'failed'}")
            
        except Exception as e:
            logger.error(f"Deployment {deployment_id} failed: {str(e)}")
            deployment_result.status = DeploymentStatus.FAILED
            deployment_result.success = False
            deployment_result.error_message = str(e)
            deployment_result.end_time = datetime.now()
        
        finally:
            # Move to history and clean up
            self.deployment_history.append(deployment_result)
            if deployment_id in self.active_deployments:
                del self.active_deployments[deployment_id]
        
        return deployment_result
    
    async def _rolling_deployment(self, config: DeploymentConfig, result: DeploymentResult) -> bool:
        """Execute rolling deployment strategy"""
        try:
            logger.info(f"Starting rolling deployment for {config.application_name}")
            
            # Calculate deployment batches
            batch_size = max(1, config.replicas // 4)  # Deploy in 4 batches
            batches = self._calculate_deployment_batches(config.replicas, batch_size)
            
            # Deploy each batch
            for batch_num, batch_replicas in enumerate(batches, 1):
                logger.info(f"Deploying batch {batch_num}/{len(batches)} ({batch_replicas} replicas)")
                
                # Deploy batch
                await self._deploy_batch(config, batch_replicas, result)
                
                # Wait for batch to be healthy
                if not await self._wait_for_batch_health(config, batch_replicas, result):
                    logger.error(f"Batch {batch_num} failed health checks")
                    await self._trigger_rollback(config, result, "Batch health check failed")
                    return False
                
                # Check deployment metrics
                if not await self._validate_deployment_metrics(config, result):
                    logger.error(f"Batch {batch_num} failed metrics validation")
                    await self._trigger_rollback(config, result, "Metrics validation failed")
                    return False
                
                # Brief pause between batches
                await asyncio.sleep(5)
            
            logger.info(f"Rolling deployment completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Rolling deployment failed: {str(e)}")
            await self._trigger_rollback(config, result, f"Rolling deployment error: {str(e)}")
            return False
    
    async def _blue_green_deployment(self, config: DeploymentConfig, result: DeploymentResult) -> bool:
        """Execute blue-green deployment strategy"""
        try:
            logger.info(f"Starting blue-green deployment for {config.application_name}")
            
            # Deploy to green environment
            logger.info("Deploying to green environment")
            await self._deploy_green_environment(config, result)
            
            # Health check green environment
            if not await self._health_check_green_environment(config, result):
                logger.error("Green environment failed health checks")
                await self._cleanup_green_environment(config)
                return False
            
            # Validate green environment metrics
            if not await self._validate_green_environment_metrics(config, result):
                logger.error("Green environment failed metrics validation")
                await self._cleanup_green_environment(config)
                return False
            
            # Switch traffic to green
            logger.info("Switching traffic to green environment")
            await self._switch_traffic_to_green(config, result)
            
            # Monitor post-switch metrics
            if not await self._monitor_post_switch_metrics(config, result):
                logger.error("Post-switch monitoring failed")
                await self._switch_traffic_to_blue(config, result)
                return False
            
            # Cleanup blue environment
            logger.info("Cleaning up blue environment")
            await self._cleanup_blue_environment(config)
            
            logger.info("Blue-green deployment completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Blue-green deployment failed: {str(e)}")
            await self._switch_traffic_to_blue(config, result)
            await self._cleanup_green_environment(config)
            return False
    
    async def _canary_deployment(self, config: DeploymentConfig, result: DeploymentResult) -> bool:
        """Execute canary deployment strategy"""
        try:
            logger.info(f"Starting canary deployment for {config.application_name}")
            
            # Calculate canary replicas
            canary_replicas = max(1, (config.replicas * config.canary_percentage) // 100)
            
            # Deploy canary version
            logger.info(f"Deploying canary version ({canary_replicas} replicas)")
            await self._deploy_canary_version(config, canary_replicas, result)
            
            # Health check canary
            if not await self._health_check_canary(config, result):
                logger.error("Canary failed health checks")
                await self._cleanup_canary(config)
                return False
            
            # Route small percentage of traffic to canary
            logger.info(f"Routing {config.canary_percentage}% traffic to canary")
            await self._route_traffic_to_canary(config, config.canary_percentage, result)
            
            # Monitor canary metrics
            canary_success = await self._monitor_canary_metrics(config, result)
            
            if canary_success:
                # Gradually increase canary traffic
                for percentage in [25, 50, 75, 100]:
                    logger.info(f"Increasing canary traffic to {percentage}%")
                    await self._route_traffic_to_canary(config, percentage, result)
                    
                    if not await self._monitor_canary_metrics(config, result):
                        logger.error(f"Canary failed at {percentage}% traffic")
                        await self._rollback_canary_traffic(config, result)
                        return False
                    
                    await asyncio.sleep(30)  # Wait between traffic increases
                
                # Complete canary deployment
                logger.info("Completing canary deployment")
                await self._complete_canary_deployment(config, result)
                return True
            else:
                logger.error("Canary monitoring failed")
                await self._rollback_canary_traffic(config, result)
                return False
            
        except Exception as e:
            logger.error(f"Canary deployment failed: {str(e)}")
            await self._rollback_canary_traffic(config, result)
            return False
    
    def _calculate_deployment_batches(self, total_replicas: int, batch_size: int) -> List[int]:
        """Calculate deployment batches for rolling deployment"""
        batches = []
        remaining = total_replicas
        
        while remaining > 0:
            current_batch = min(batch_size, remaining)
            batches.append(current_batch)
            remaining -= current_batch
        
        return batches
    
    async def _deploy_batch(self, config: DeploymentConfig, batch_size: int, result: DeploymentResult):
        """Deploy a batch of replicas"""
        logger.info(f"Deploying batch of {batch_size} replicas")
        # Simulate batch deployment
        await asyncio.sleep(2)
    
    async def _wait_for_batch_health(self, config: DeploymentConfig, batch_size: int, result: DeploymentResult) -> bool:
        """Wait for batch to become healthy"""
        max_wait_time = config.health_check_timeout
        wait_time = 0
        
        while wait_time < max_wait_time:
            health_status = await self._check_batch_health(config, batch_size)
            if health_status == HealthCheckStatus.HEALTHY:
                return True
            elif health_status == HealthCheckStatus.UNHEALTHY:
                return False
            
            await asyncio.sleep(5)
            wait_time += 5
        
        return False
    
    async def _check_batch_health(self, config: DeploymentConfig, batch_size: int) -> HealthCheckStatus:
        """Check health of deployed batch"""
        # Simulate health check
        await asyncio.sleep(1)
        import random
        
        # 90% chance of healthy, 5% degraded, 5% unhealthy
        rand = random.random()
        if rand < 0.9:
            return HealthCheckStatus.HEALTHY
        elif rand < 0.95:
            return HealthCheckStatus.DEGRADED
        else:
            return HealthCheckStatus.UNHEALTHY
    
    async def _validate_deployment_metrics(self, config: DeploymentConfig, result: DeploymentResult) -> bool:
        """Validate deployment metrics against thresholds"""
        # Collect current metrics
        metrics = await self._collect_deployment_metrics(config)
        result.metrics.append(metrics)
        
        # Check against thresholds
        if metrics.error_rate > self.error_rate_threshold:
            logger.warning(f"Error rate {metrics.error_rate} exceeds threshold {self.error_rate_threshold}")
            return False
        
        if metrics.response_time > self.response_time_threshold:
            logger.warning(f"Response time {metrics.response_time}ms exceeds threshold {self.response_time_threshold}ms")
            return False
        
        if metrics.success_rate < self.success_rate_threshold:
            logger.warning(f"Success rate {metrics.success_rate} below threshold {self.success_rate_threshold}")
            return False
        
        return True
    
    async def _collect_deployment_metrics(self, config: DeploymentConfig) -> DeploymentMetrics:
        """Collect current deployment metrics"""
        # Simulate metrics collection
        import random
        
        return DeploymentMetrics(
            timestamp=datetime.now(),
            success_rate=random.uniform(0.95, 1.0),
            error_rate=random.uniform(0.0, 0.02),
            response_time=random.uniform(50, 200),
            throughput=random.uniform(1000, 5000),
            cpu_utilization=random.uniform(30, 70),
            memory_utilization=random.uniform(40, 80),
            active_connections=random.randint(100, 1000)
        )
    
    async def _trigger_rollback(self, config: DeploymentConfig, result: DeploymentResult, reason: str):
        """Trigger automatic rollback"""
        logger.warning(f"Triggering rollback: {reason}")
        result.rollback_triggered = True
        result.rollback_reason = reason
        result.status = DeploymentStatus.ROLLING_BACK
        
        # Execute rollback based on strategy
        if config.strategy == DeploymentStrategy.ROLLING:
            await self._rollback_rolling_deployment(config, result)
        elif config.strategy == DeploymentStrategy.BLUE_GREEN:
            await self._rollback_blue_green_deployment(config, result)
        elif config.strategy == DeploymentStrategy.CANARY:
            await self._rollback_canary_deployment(config, result)
        
        result.status = DeploymentStatus.ROLLED_BACK
        logger.info("Rollback completed")
    
    async def _rollback_rolling_deployment(self, config: DeploymentConfig, result: DeploymentResult):
        """Rollback rolling deployment"""
        logger.info("Rolling back rolling deployment")
        # Simulate rollback
        await asyncio.sleep(3)
    
    async def _rollback_blue_green_deployment(self, config: DeploymentConfig, result: DeploymentResult):
        """Rollback blue-green deployment"""
        logger.info("Rolling back blue-green deployment")
        await self._switch_traffic_to_blue(config, result)
        await self._cleanup_green_environment(config)
    
    async def _rollback_canary_deployment(self, config: DeploymentConfig, result: DeploymentResult):
        """Rollback canary deployment"""
        logger.info("Rolling back canary deployment")
        await self._rollback_canary_traffic(config, result)
        await self._cleanup_canary(config)
    
    # Blue-Green deployment methods
    async def _deploy_green_environment(self, config: DeploymentConfig, result: DeploymentResult):
        """Deploy to green environment"""
        logger.info("Deploying to green environment")
        await asyncio.sleep(3)  # Simulate deployment
    
    async def _health_check_green_environment(self, config: DeploymentConfig, result: DeploymentResult) -> bool:
        """Health check green environment"""
        logger.info("Health checking green environment")
        await asyncio.sleep(2)
        return True  # Simulate successful health check
    
    async def _validate_green_environment_metrics(self, config: DeploymentConfig, result: DeploymentResult) -> bool:
        """Validate green environment metrics"""
        return await self._validate_deployment_metrics(config, result)
    
    async def _switch_traffic_to_green(self, config: DeploymentConfig, result: DeploymentResult):
        """Switch traffic to green environment"""
        logger.info("Switching traffic to green environment")
        await asyncio.sleep(1)
    
    async def _switch_traffic_to_blue(self, config: DeploymentConfig, result: DeploymentResult):
        """Switch traffic back to blue environment"""
        logger.info("Switching traffic back to blue environment")
        await asyncio.sleep(1)
    
    async def _monitor_post_switch_metrics(self, config: DeploymentConfig, result: DeploymentResult) -> bool:
        """Monitor metrics after traffic switch"""
        logger.info("Monitoring post-switch metrics")
        await asyncio.sleep(30)  # Monitor for 30 seconds
        return await self._validate_deployment_metrics(config, result)
    
    async def _cleanup_green_environment(self, config: DeploymentConfig):
        """Cleanup green environment"""
        logger.info("Cleaning up green environment")
        await asyncio.sleep(1)
    
    async def _cleanup_blue_environment(self, config: DeploymentConfig):
        """Cleanup blue environment"""
        logger.info("Cleaning up blue environment")
        await asyncio.sleep(1)
    
    # Canary deployment methods
    async def _deploy_canary_version(self, config: DeploymentConfig, replicas: int, result: DeploymentResult):
        """Deploy canary version"""
        logger.info(f"Deploying canary version with {replicas} replicas")
        await asyncio.sleep(2)
    
    async def _health_check_canary(self, config: DeploymentConfig, result: DeploymentResult) -> bool:
        """Health check canary deployment"""
        logger.info("Health checking canary deployment")
        await asyncio.sleep(1)
        return True
    
    async def _route_traffic_to_canary(self, config: DeploymentConfig, percentage: int, result: DeploymentResult):
        """Route traffic percentage to canary"""
        logger.info(f"Routing {percentage}% traffic to canary")
        await asyncio.sleep(1)
    
    async def _monitor_canary_metrics(self, config: DeploymentConfig, result: DeploymentResult) -> bool:
        """Monitor canary metrics"""
        logger.info("Monitoring canary metrics")
        await asyncio.sleep(10)  # Monitor for 10 seconds
        return await self._validate_deployment_metrics(config, result)
    
    async def _rollback_canary_traffic(self, config: DeploymentConfig, result: DeploymentResult):
        """Rollback canary traffic"""
        logger.info("Rolling back canary traffic")
        await self._route_traffic_to_canary(config, 0, result)
    
    async def _complete_canary_deployment(self, config: DeploymentConfig, result: DeploymentResult):
        """Complete canary deployment"""
        logger.info("Completing canary deployment")
        await asyncio.sleep(2)
    
    async def _cleanup_canary(self, config: DeploymentConfig):
        """Cleanup canary deployment"""
        logger.info("Cleaning up canary deployment")
        await asyncio.sleep(1)
    
    def get_deployment_status(self, deployment_id: str) -> Optional[DeploymentResult]:
        """Get status of a deployment"""
        if deployment_id in self.active_deployments:
            return self.active_deployments[deployment_id]
        
        for deployment in self.deployment_history:
            if deployment.deployment_id == deployment_id:
                return deployment
        
        return None
    
    def get_deployment_history(self, limit: int = 50) -> List[DeploymentResult]:
        """Get deployment history"""
        return self.deployment_history[-limit:]