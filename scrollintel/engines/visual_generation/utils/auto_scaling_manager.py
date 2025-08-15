"""
Auto-scaling infrastructure for visual generation workloads.
"""

import asyncio
import time
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import json
import boto3
import kubernetes
from google.cloud import compute_v1
from azure.identity import DefaultAzureCredential
from azure.mgmt.compute import ComputeManagementClient

from ..config import InfrastructureConfig
from ..exceptions import InfrastructureError


class CloudProvider(Enum):
    """Supported cloud providers."""
    AWS = "aws"
    GCP = "gcp"
    AZURE = "azure"
    KUBERNETES = "kubernetes"
    LOCAL = "local"


class WorkerType(Enum):
    """Types of workers for different workloads."""
    GPU_WORKER = "gpu_worker"
    CPU_WORKER = "cpu_worker"
    MEMORY_OPTIMIZED = "memory_optimized"
    COMPUTE_OPTIMIZED = "compute_optimized"


@dataclass
class WorkerSpec:
    """Specification for a worker instance."""
    worker_type: WorkerType
    instance_type: str
    gpu_count: int = 0
    cpu_count: int = 4
    memory_gb: int = 16
    storage_gb: int = 100
    cost_per_hour: float = 0.5
    max_concurrent_jobs: int = 1
    startup_time_seconds: int = 300
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class ScalingMetrics:
    """Metrics used for scaling decisions."""
    queue_length: int = 0
    active_workers: int = 0
    idle_workers: int = 0
    cpu_utilization: float = 0.0
    gpu_utilization: float = 0.0
    memory_utilization: float = 0.0
    average_job_duration: float = 0.0
    pending_jobs: int = 0
    failed_jobs: int = 0
    cost_per_hour: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class ScalingPolicy:
    """Auto-scaling policy configuration."""
    min_workers: int = 1
    max_workers: int = 10
    target_queue_length: int = 5
    scale_up_threshold: float = 0.8
    scale_down_threshold: float = 0.3
    scale_up_cooldown: int = 300  # seconds
    scale_down_cooldown: int = 600  # seconds
    cost_optimization_enabled: bool = True
    preemptible_instances: bool = True
    multi_cloud_enabled: bool = False


class AutoScalingManager:
    """Manages auto-scaling of visual generation infrastructure."""
    
    def __init__(self, config: InfrastructureConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Cloud provider clients
        self.cloud_clients: Dict[CloudProvider, Any] = {}
        self.worker_specs: Dict[WorkerType, WorkerSpec] = {}
        self.scaling_policies: Dict[WorkerType, ScalingPolicy] = {}
        
        # State tracking
        self.active_workers: Dict[str, Dict[str, Any]] = {}
        self.scaling_history: List[Dict[str, Any]] = []
        self.last_scale_action: Dict[WorkerType, float] = {}
        
        # Monitoring
        self.metrics_history: List[ScalingMetrics] = []
        self.monitoring_task: Optional[asyncio.Task] = None
        
        self._initialize_cloud_clients()
        self._initialize_worker_specs()
        self._initialize_scaling_policies()
    
    def _initialize_cloud_clients(self):
        """Initialize cloud provider clients."""
        try:
            # AWS
            if hasattr(self.config, 'aws_enabled') and self.config.aws_enabled:
                self.cloud_clients[CloudProvider.AWS] = {
                    'ec2': boto3.client('ec2'),
                    'autoscaling': boto3.client('autoscaling'),
                    'ecs': boto3.client('ecs')
                }
            
            # GCP
            if hasattr(self.config, 'gcp_enabled') and self.config.gcp_enabled:
                self.cloud_clients[CloudProvider.GCP] = {
                    'compute': compute_v1.InstancesClient(),
                    'instance_groups': compute_v1.InstanceGroupManagersClient()
                }
            
            # Azure
            if hasattr(self.config, 'azure_enabled') and self.config.azure_enabled:
                credential = DefaultAzureCredential()
                self.cloud_clients[CloudProvider.AZURE] = {
                    'compute': ComputeManagementClient(
                        credential, 
                        getattr(self.config, 'azure_subscription_id', '')
                    )
                }
            
            # Kubernetes
            if hasattr(self.config, 'kubernetes_enabled') and self.config.kubernetes_enabled:
                kubernetes.config.load_incluster_config()
                self.cloud_clients[CloudProvider.KUBERNETES] = {
                    'apps_v1': kubernetes.client.AppsV1Api(),
                    'core_v1': kubernetes.client.CoreV1Api(),
                    'custom_objects': kubernetes.client.CustomObjectsApi()
                }
                
        except Exception as e:
            self.logger.warning(f"Failed to initialize some cloud clients: {e}")
    
    def _initialize_worker_specs(self):
        """Initialize worker specifications for different workload types."""
        self.worker_specs = {
            WorkerType.GPU_WORKER: WorkerSpec(
                worker_type=WorkerType.GPU_WORKER,
                instance_type="g4dn.xlarge",  # AWS example
                gpu_count=1,
                cpu_count=4,
                memory_gb=16,
                storage_gb=125,
                cost_per_hour=0.526,
                max_concurrent_jobs=2,
                startup_time_seconds=180,
                tags={"workload": "gpu_generation", "auto_scaling": "true"}
            ),
            WorkerType.CPU_WORKER: WorkerSpec(
                worker_type=WorkerType.CPU_WORKER,
                instance_type="c5.2xlarge",  # AWS example
                gpu_count=0,
                cpu_count=8,
                memory_gb=16,
                storage_gb=100,
                cost_per_hour=0.34,
                max_concurrent_jobs=4,
                startup_time_seconds=120,
                tags={"workload": "cpu_processing", "auto_scaling": "true"}
            ),
            WorkerType.MEMORY_OPTIMIZED: WorkerSpec(
                worker_type=WorkerType.MEMORY_OPTIMIZED,
                instance_type="r5.2xlarge",  # AWS example
                gpu_count=0,
                cpu_count=8,
                memory_gb=64,
                storage_gb=100,
                cost_per_hour=0.504,
                max_concurrent_jobs=2,
                startup_time_seconds=150,
                tags={"workload": "memory_intensive", "auto_scaling": "true"}
            )
        }
    
    def _initialize_scaling_policies(self):
        """Initialize scaling policies for different worker types."""
        self.scaling_policies = {
            WorkerType.GPU_WORKER: ScalingPolicy(
                min_workers=0,
                max_workers=20,
                target_queue_length=3,
                scale_up_threshold=0.7,
                scale_down_threshold=0.2,
                scale_up_cooldown=180,
                scale_down_cooldown=300,
                cost_optimization_enabled=True,
                preemptible_instances=True
            ),
            WorkerType.CPU_WORKER: ScalingPolicy(
                min_workers=1,
                max_workers=50,
                target_queue_length=10,
                scale_up_threshold=0.8,
                scale_down_threshold=0.3,
                scale_up_cooldown=120,
                scale_down_cooldown=240,
                cost_optimization_enabled=True,
                preemptible_instances=True
            ),
            WorkerType.MEMORY_OPTIMIZED: ScalingPolicy(
                min_workers=0,
                max_workers=10,
                target_queue_length=2,
                scale_up_threshold=0.6,
                scale_down_threshold=0.2,
                scale_up_cooldown=240,
                scale_down_cooldown=360,
                cost_optimization_enabled=True,
                preemptible_instances=False  # More stable for memory-intensive tasks
            )
        }
    
    async def start_monitoring(self):
        """Start the auto-scaling monitoring loop."""
        if self.monitoring_task is None:
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())
            self.logger.info("Auto-scaling monitoring started")
    
    async def stop_monitoring(self):
        """Stop the auto-scaling monitoring loop."""
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
            self.monitoring_task = None
            self.logger.info("Auto-scaling monitoring stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring and scaling loop."""
        while True:
            try:
                # Collect metrics
                metrics = await self._collect_metrics()
                self.metrics_history.append(metrics)
                
                # Keep only recent metrics (last 24 hours)
                cutoff_time = time.time() - 86400
                self.metrics_history = [
                    m for m in self.metrics_history 
                    if m.timestamp > cutoff_time
                ]
                
                # Make scaling decisions
                await self._make_scaling_decisions(metrics)
                
                # Optimize costs
                if any(policy.cost_optimization_enabled for policy in self.scaling_policies.values()):
                    await self._optimize_costs()
                
                # Wait before next iteration
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(60)  # Wait longer on error
    
    async def _collect_metrics(self) -> ScalingMetrics:
        """Collect current system metrics."""
        # This would integrate with your job queue and worker monitoring
        # For now, return mock metrics
        return ScalingMetrics(
            queue_length=len(self._get_pending_jobs()),
            active_workers=len([w for w in self.active_workers.values() if w['status'] == 'running']),
            idle_workers=len([w for w in self.active_workers.values() if w['status'] == 'idle']),
            cpu_utilization=await self._get_average_cpu_utilization(),
            gpu_utilization=await self._get_average_gpu_utilization(),
            memory_utilization=await self._get_average_memory_utilization(),
            average_job_duration=await self._get_average_job_duration(),
            pending_jobs=len(self._get_pending_jobs()),
            failed_jobs=len(self._get_failed_jobs()),
            cost_per_hour=await self._calculate_current_cost()
        )
    
    def _get_pending_jobs(self) -> List[Dict[str, Any]]:
        """Get list of pending jobs from queue."""
        # This would integrate with your actual job queue
        return []
    
    def _get_failed_jobs(self) -> List[Dict[str, Any]]:
        """Get list of recently failed jobs."""
        # This would integrate with your job tracking system
        return []
    
    async def _get_average_cpu_utilization(self) -> float:
        """Get average CPU utilization across all workers."""
        if not self.active_workers:
            return 0.0
        
        total_utilization = 0.0
        count = 0
        
        for worker_id, worker_info in self.active_workers.items():
            try:
                utilization = await self._get_worker_cpu_utilization(worker_id)
                total_utilization += utilization
                count += 1
            except Exception as e:
                self.logger.warning(f"Failed to get CPU utilization for worker {worker_id}: {e}")
        
        return total_utilization / count if count > 0 else 0.0
    
    async def _get_average_gpu_utilization(self) -> float:
        """Get average GPU utilization across all GPU workers."""
        gpu_workers = [
            w for w in self.active_workers.values() 
            if w.get('worker_type') == WorkerType.GPU_WORKER
        ]
        
        if not gpu_workers:
            return 0.0
        
        total_utilization = 0.0
        count = 0
        
        for worker_info in gpu_workers:
            try:
                utilization = await self._get_worker_gpu_utilization(worker_info['id'])
                total_utilization += utilization
                count += 1
            except Exception as e:
                self.logger.warning(f"Failed to get GPU utilization for worker {worker_info['id']}: {e}")
        
        return total_utilization / count if count > 0 else 0.0
    
    async def _get_average_memory_utilization(self) -> float:
        """Get average memory utilization across all workers."""
        if not self.active_workers:
            return 0.0
        
        total_utilization = 0.0
        count = 0
        
        for worker_id, worker_info in self.active_workers.items():
            try:
                utilization = await self._get_worker_memory_utilization(worker_id)
                total_utilization += utilization
                count += 1
            except Exception as e:
                self.logger.warning(f"Failed to get memory utilization for worker {worker_id}: {e}")
        
        return total_utilization / count if count > 0 else 0.0
    
    async def _get_worker_cpu_utilization(self, worker_id: str) -> float:
        """Get CPU utilization for a specific worker."""
        # This would integrate with your monitoring system (CloudWatch, Prometheus, etc.)
        return 0.5  # Mock value
    
    async def _get_worker_gpu_utilization(self, worker_id: str) -> float:
        """Get GPU utilization for a specific worker."""
        # This would integrate with nvidia-smi or similar GPU monitoring
        return 0.7  # Mock value
    
    async def _get_worker_memory_utilization(self, worker_id: str) -> float:
        """Get memory utilization for a specific worker."""
        # This would integrate with your monitoring system
        return 0.6  # Mock value
    
    async def _get_average_job_duration(self) -> float:
        """Get average job duration from recent history."""
        # This would analyze job completion times
        return 300.0  # Mock value (5 minutes)
    
    async def _calculate_current_cost(self) -> float:
        """Calculate current hourly cost of all active workers."""
        total_cost = 0.0
        
        for worker_info in self.active_workers.values():
            worker_type = worker_info.get('worker_type')
            if worker_type and worker_type in self.worker_specs:
                spec = self.worker_specs[worker_type]
                total_cost += spec.cost_per_hour
        
        return total_cost
    
    async def _make_scaling_decisions(self, metrics: ScalingMetrics):
        """Make scaling decisions based on current metrics."""
        for worker_type, policy in self.scaling_policies.items():
            await self._evaluate_scaling_for_worker_type(worker_type, policy, metrics)
    
    async def _evaluate_scaling_for_worker_type(
        self, 
        worker_type: WorkerType, 
        policy: ScalingPolicy, 
        metrics: ScalingMetrics
    ):
        """Evaluate scaling decisions for a specific worker type."""
        current_workers = len([
            w for w in self.active_workers.values() 
            if w.get('worker_type') == worker_type
        ])
        
        # Check cooldown periods
        last_action_time = self.last_scale_action.get(worker_type, 0)
        current_time = time.time()
        
        # Determine if scaling is needed
        should_scale_up = (
            metrics.queue_length > policy.target_queue_length and
            (metrics.cpu_utilization > policy.scale_up_threshold or 
             metrics.gpu_utilization > policy.scale_up_threshold) and
            current_workers < policy.max_workers and
            current_time - last_action_time > policy.scale_up_cooldown
        )
        
        should_scale_down = (
            metrics.queue_length < policy.target_queue_length // 2 and
            metrics.cpu_utilization < policy.scale_down_threshold and
            metrics.gpu_utilization < policy.scale_down_threshold and
            current_workers > policy.min_workers and
            current_time - last_action_time > policy.scale_down_cooldown
        )
        
        if should_scale_up:
            await self._scale_up(worker_type, policy, metrics)
        elif should_scale_down:
            await self._scale_down(worker_type, policy, metrics)
    
    async def _scale_up(self, worker_type: WorkerType, policy: ScalingPolicy, metrics: ScalingMetrics):
        """Scale up workers of the specified type."""
        try:
            # Calculate number of workers to add
            workers_to_add = min(
                max(1, metrics.queue_length // policy.target_queue_length),
                policy.max_workers - len([
                    w for w in self.active_workers.values() 
                    if w.get('worker_type') == worker_type
                ])
            )
            
            if workers_to_add <= 0:
                return
            
            self.logger.info(f"Scaling up {workers_to_add} {worker_type.value} workers")
            
            # Choose optimal cloud provider and instance type
            provider, instance_config = await self._choose_optimal_provider(worker_type, policy)
            
            # Launch workers
            for _ in range(workers_to_add):
                worker_id = await self._launch_worker(provider, worker_type, instance_config)
                if worker_id:
                    self.active_workers[worker_id] = {
                        'id': worker_id,
                        'worker_type': worker_type,
                        'provider': provider,
                        'status': 'launching',
                        'launch_time': time.time(),
                        'instance_config': instance_config
                    }
            
            # Record scaling action
            self.last_scale_action[worker_type] = time.time()
            self.scaling_history.append({
                'timestamp': time.time(),
                'action': 'scale_up',
                'worker_type': worker_type.value,
                'workers_added': workers_to_add,
                'provider': provider.value,
                'metrics': metrics.__dict__
            })
            
        except Exception as e:
            self.logger.error(f"Failed to scale up {worker_type.value} workers: {e}")
    
    async def _scale_down(self, worker_type: WorkerType, policy: ScalingPolicy, metrics: ScalingMetrics):
        """Scale down workers of the specified type."""
        try:
            current_workers = [
                w for w in self.active_workers.values() 
                if w.get('worker_type') == worker_type and w.get('status') in ['running', 'idle']
            ]
            
            min_workers = policy.min_workers
            workers_to_remove = min(
                len(current_workers) - min_workers,
                max(1, len(current_workers) // 3)  # Remove at most 1/3 at a time
            )
            
            if workers_to_remove <= 0:
                return
            
            self.logger.info(f"Scaling down {workers_to_remove} {worker_type.value} workers")
            
            # Choose workers to remove (prefer idle workers, then by cost)
            workers_to_terminate = await self._choose_workers_to_terminate(
                current_workers, workers_to_remove
            )
            
            # Terminate workers
            for worker_info in workers_to_terminate:
                await self._terminate_worker(worker_info)
                del self.active_workers[worker_info['id']]
            
            # Record scaling action
            self.last_scale_action[worker_type] = time.time()
            self.scaling_history.append({
                'timestamp': time.time(),
                'action': 'scale_down',
                'worker_type': worker_type.value,
                'workers_removed': workers_to_remove,
                'metrics': metrics.__dict__
            })
            
        except Exception as e:
            self.logger.error(f"Failed to scale down {worker_type.value} workers: {e}")
    
    async def _choose_optimal_provider(
        self, 
        worker_type: WorkerType, 
        policy: ScalingPolicy
    ) -> tuple[CloudProvider, Dict[str, Any]]:
        """Choose the optimal cloud provider and configuration."""
        # For now, return AWS as default
        # In a real implementation, this would compare costs, availability, etc.
        provider = CloudProvider.AWS
        
        spec = self.worker_specs[worker_type]
        instance_config = {
            'instance_type': spec.instance_type,
            'image_id': 'ami-12345678',  # Your custom AMI with GPU drivers, etc.
            'security_groups': ['sg-12345678'],
            'subnet_id': 'subnet-12345678',
            'user_data': self._generate_user_data(worker_type),
            'tags': spec.tags,
            'spot_instance': policy.preemptible_instances
        }
        
        return provider, instance_config
    
    def _generate_user_data(self, worker_type: WorkerType) -> str:
        """Generate user data script for worker initialization."""
        return f"""#!/bin/bash
# Initialize worker for {worker_type.value}
cd /opt/scrollintel
./scripts/setup-worker.sh {worker_type.value}
systemctl start scrollintel-worker
"""
    
    async def _launch_worker(
        self, 
        provider: CloudProvider, 
        worker_type: WorkerType, 
        instance_config: Dict[str, Any]
    ) -> Optional[str]:
        """Launch a new worker instance."""
        try:
            if provider == CloudProvider.AWS:
                return await self._launch_aws_worker(instance_config)
            elif provider == CloudProvider.GCP:
                return await self._launch_gcp_worker(instance_config)
            elif provider == CloudProvider.AZURE:
                return await self._launch_azure_worker(instance_config)
            elif provider == CloudProvider.KUBERNETES:
                return await self._launch_k8s_worker(worker_type, instance_config)
            else:
                raise InfrastructureError(f"Unsupported provider: {provider}")
                
        except Exception as e:
            self.logger.error(f"Failed to launch worker on {provider.value}: {e}")
            return None
    
    async def _launch_aws_worker(self, instance_config: Dict[str, Any]) -> str:
        """Launch worker on AWS EC2."""
        ec2 = self.cloud_clients[CloudProvider.AWS]['ec2']
        
        if instance_config.get('spot_instance', False):
            # Launch spot instance
            response = ec2.request_spot_instances(
                SpotPrice='0.50',  # Max price
                LaunchSpecification={
                    'ImageId': instance_config['image_id'],
                    'InstanceType': instance_config['instance_type'],
                    'SecurityGroups': instance_config['security_groups'],
                    'SubnetId': instance_config['subnet_id'],
                    'UserData': instance_config['user_data']
                }
            )
            return response['SpotInstanceRequests'][0]['SpotInstanceRequestId']
        else:
            # Launch on-demand instance
            response = ec2.run_instances(
                ImageId=instance_config['image_id'],
                MinCount=1,
                MaxCount=1,
                InstanceType=instance_config['instance_type'],
                SecurityGroups=instance_config['security_groups'],
                SubnetId=instance_config['subnet_id'],
                UserData=instance_config['user_data'],
                TagSpecifications=[{
                    'ResourceType': 'instance',
                    'Tags': [{'Key': k, 'Value': v} for k, v in instance_config['tags'].items()]
                }]
            )
            return response['Instances'][0]['InstanceId']
    
    async def _launch_gcp_worker(self, instance_config: Dict[str, Any]) -> str:
        """Launch worker on Google Cloud Platform."""
        # Implementation for GCP instance launch
        pass
    
    async def _launch_azure_worker(self, instance_config: Dict[str, Any]) -> str:
        """Launch worker on Microsoft Azure."""
        # Implementation for Azure VM launch
        pass
    
    async def _launch_k8s_worker(self, worker_type: WorkerType, instance_config: Dict[str, Any]) -> str:
        """Launch worker on Kubernetes."""
        apps_v1 = self.cloud_clients[CloudProvider.KUBERNETES]['apps_v1']
        
        # Create deployment for the worker
        deployment_manifest = {
            'apiVersion': 'apps/v1',
            'kind': 'Deployment',
            'metadata': {
                'name': f'scrollintel-{worker_type.value}-{int(time.time())}',
                'labels': {'app': 'scrollintel-worker', 'worker-type': worker_type.value}
            },
            'spec': {
                'replicas': 1,
                'selector': {'matchLabels': {'app': 'scrollintel-worker'}},
                'template': {
                    'metadata': {'labels': {'app': 'scrollintel-worker', 'worker-type': worker_type.value}},
                    'spec': {
                        'containers': [{
                            'name': 'scrollintel-worker',
                            'image': 'scrollintel/worker:latest',
                            'env': [
                                {'name': 'WORKER_TYPE', 'value': worker_type.value},
                                {'name': 'REDIS_URL', 'value': self.config.redis_url}
                            ],
                            'resources': self._get_k8s_resources(worker_type)
                        }]
                    }
                }
            }
        }
        
        response = apps_v1.create_namespaced_deployment(
            namespace='default',
            body=deployment_manifest
        )
        
        return response.metadata.name
    
    def _get_k8s_resources(self, worker_type: WorkerType) -> Dict[str, Any]:
        """Get Kubernetes resource requirements for worker type."""
        spec = self.worker_specs[worker_type]
        
        resources = {
            'requests': {
                'cpu': f"{spec.cpu_count}",
                'memory': f"{spec.memory_gb}Gi"
            },
            'limits': {
                'cpu': f"{spec.cpu_count}",
                'memory': f"{spec.memory_gb}Gi"
            }
        }
        
        if spec.gpu_count > 0:
            resources['requests']['nvidia.com/gpu'] = str(spec.gpu_count)
            resources['limits']['nvidia.com/gpu'] = str(spec.gpu_count)
        
        return resources
    
    async def _choose_workers_to_terminate(
        self, 
        workers: List[Dict[str, Any]], 
        count: int
    ) -> List[Dict[str, Any]]:
        """Choose which workers to terminate (prefer idle, then by cost)."""
        # Sort by preference: idle first, then by cost (highest first for cost savings)
        def sort_key(worker):
            is_idle = worker.get('status') == 'idle'
            worker_type = worker.get('worker_type')
            cost = self.worker_specs[worker_type].cost_per_hour if worker_type else 0
            return (not is_idle, -cost)  # Idle first, then highest cost first
        
        sorted_workers = sorted(workers, key=sort_key)
        return sorted_workers[:count]
    
    async def _terminate_worker(self, worker_info: Dict[str, Any]):
        """Terminate a specific worker."""
        try:
            provider = worker_info.get('provider')
            worker_id = worker_info['id']
            
            if provider == CloudProvider.AWS:
                await self._terminate_aws_worker(worker_id)
            elif provider == CloudProvider.GCP:
                await self._terminate_gcp_worker(worker_id)
            elif provider == CloudProvider.AZURE:
                await self._terminate_azure_worker(worker_id)
            elif provider == CloudProvider.KUBERNETES:
                await self._terminate_k8s_worker(worker_id)
            
            self.logger.info(f"Terminated worker {worker_id} on {provider.value}")
            
        except Exception as e:
            self.logger.error(f"Failed to terminate worker {worker_info['id']}: {e}")
    
    async def _terminate_aws_worker(self, instance_id: str):
        """Terminate AWS EC2 instance."""
        ec2 = self.cloud_clients[CloudProvider.AWS]['ec2']
        ec2.terminate_instances(InstanceIds=[instance_id])
    
    async def _terminate_gcp_worker(self, instance_id: str):
        """Terminate GCP instance."""
        # Implementation for GCP instance termination
        pass
    
    async def _terminate_azure_worker(self, vm_id: str):
        """Terminate Azure VM."""
        # Implementation for Azure VM termination
        pass
    
    async def _terminate_k8s_worker(self, deployment_name: str):
        """Terminate Kubernetes deployment."""
        apps_v1 = self.cloud_clients[CloudProvider.KUBERNETES]['apps_v1']
        apps_v1.delete_namespaced_deployment(
            name=deployment_name,
            namespace='default'
        )
    
    async def _optimize_costs(self):
        """Optimize costs by switching to cheaper instances when possible."""
        try:
            # Analyze current workload patterns
            recent_metrics = self.metrics_history[-10:] if len(self.metrics_history) >= 10 else self.metrics_history
            
            if not recent_metrics:
                return
            
            avg_utilization = sum(m.cpu_utilization for m in recent_metrics) / len(recent_metrics)
            avg_gpu_utilization = sum(m.gpu_utilization for m in recent_metrics) / len(recent_metrics)
            
            # If utilization is consistently low, consider switching to smaller instances
            if avg_utilization < 0.3 and avg_gpu_utilization < 0.3:
                await self._suggest_cost_optimizations()
            
        except Exception as e:
            self.logger.error(f"Cost optimization error: {e}")
    
    async def _suggest_cost_optimizations(self):
        """Suggest cost optimization opportunities."""
        suggestions = []
        
        for worker_type, workers in self._group_workers_by_type().items():
            if len(workers) > 1:
                # Suggest consolidation
                suggestions.append({
                    'type': 'consolidation',
                    'worker_type': worker_type.value,
                    'current_workers': len(workers),
                    'suggested_workers': max(1, len(workers) // 2),
                    'estimated_savings': self._calculate_consolidation_savings(workers)
                })
        
        if suggestions:
            self.logger.info(f"Cost optimization suggestions: {suggestions}")
    
    def _group_workers_by_type(self) -> Dict[WorkerType, List[Dict[str, Any]]]:
        """Group active workers by type."""
        groups = {}
        for worker_info in self.active_workers.values():
            worker_type = worker_info.get('worker_type')
            if worker_type:
                if worker_type not in groups:
                    groups[worker_type] = []
                groups[worker_type].append(worker_info)
        return groups
    
    def _calculate_consolidation_savings(self, workers: List[Dict[str, Any]]) -> float:
        """Calculate potential savings from worker consolidation."""
        if not workers:
            return 0.0
        
        worker_type = workers[0].get('worker_type')
        if not worker_type or worker_type not in self.worker_specs:
            return 0.0
        
        cost_per_hour = self.worker_specs[worker_type].cost_per_hour
        workers_to_remove = len(workers) // 2
        
        return workers_to_remove * cost_per_hour * 24 * 30  # Monthly savings
    
    async def get_scaling_status(self) -> Dict[str, Any]:
        """Get current auto-scaling status and metrics."""
        current_metrics = await self._collect_metrics()
        
        return {
            'monitoring_active': self.monitoring_task is not None,
            'current_metrics': current_metrics.__dict__,
            'active_workers': {
                worker_type.value: len([
                    w for w in self.active_workers.values() 
                    if w.get('worker_type') == worker_type
                ])
                for worker_type in WorkerType
            },
            'scaling_policies': {
                worker_type.value: {
                    'min_workers': policy.min_workers,
                    'max_workers': policy.max_workers,
                    'target_queue_length': policy.target_queue_length
                }
                for worker_type, policy in self.scaling_policies.items()
            },
            'recent_actions': self.scaling_history[-10:],
            'total_cost_per_hour': await self._calculate_current_cost()
        }
    
    async def update_scaling_policy(self, worker_type: WorkerType, policy_updates: Dict[str, Any]):
        """Update scaling policy for a worker type."""
        if worker_type not in self.scaling_policies:
            raise ValueError(f"Unknown worker type: {worker_type}")
        
        policy = self.scaling_policies[worker_type]
        for key, value in policy_updates.items():
            if hasattr(policy, key):
                setattr(policy, key, value)
        
        self.logger.info(f"Updated scaling policy for {worker_type.value}: {policy_updates}")
    
    async def manual_scale(self, worker_type: WorkerType, target_count: int):
        """Manually scale workers to a specific count."""
        current_count = len([
            w for w in self.active_workers.values() 
            if w.get('worker_type') == worker_type
        ])
        
        if target_count > current_count:
            # Scale up
            workers_to_add = target_count - current_count
            policy = self.scaling_policies[worker_type]
            metrics = await self._collect_metrics()
            
            for _ in range(workers_to_add):
                provider, instance_config = await self._choose_optimal_provider(worker_type, policy)
                worker_id = await self._launch_worker(provider, worker_type, instance_config)
                if worker_id:
                    self.active_workers[worker_id] = {
                        'id': worker_id,
                        'worker_type': worker_type,
                        'provider': provider,
                        'status': 'launching',
                        'launch_time': time.time(),
                        'instance_config': instance_config
                    }
        
        elif target_count < current_count:
            # Scale down
            workers_to_remove = current_count - target_count
            current_workers = [
                w for w in self.active_workers.values() 
                if w.get('worker_type') == worker_type
            ]
            
            workers_to_terminate = await self._choose_workers_to_terminate(
                current_workers, workers_to_remove
            )
            
            for worker_info in workers_to_terminate:
                await self._terminate_worker(worker_info)
                del self.active_workers[worker_info['id']]
        
        self.logger.info(f"Manually scaled {worker_type.value} to {target_count} workers")
    
    async def cleanup(self):
        """Clean up auto-scaling manager resources."""
        await self.stop_monitoring()
        
        # Terminate all managed workers
        for worker_info in list(self.active_workers.values()):
            await self._terminate_worker(worker_info)
        
        self.active_workers.clear()
        self.logger.info("Auto-scaling manager cleaned up")