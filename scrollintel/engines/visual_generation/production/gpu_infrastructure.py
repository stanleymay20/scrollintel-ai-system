"""
GPU Infrastructure Management for Visual Generation
Handles GPU cluster provisioning, monitoring, and optimization
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import psutil
import torch
import boto3
from google.cloud import compute_v1
import kubernetes

logger = logging.getLogger(__name__)

class CloudProvider(Enum):
    AWS = "aws"
    GCP = "gcp"
    AZURE = "azure"
    LOCAL = "local"

class GPUType(Enum):
    A100_80GB = "a100-80gb"
    A100_40GB = "a100-40gb"
    V100_32GB = "v100-32gb"
    T4_16GB = "t4-16gb"
    RTX_4090 = "rtx-4090"

@dataclass
class GPUInstance:
    instance_id: str
    gpu_type: GPUType
    cloud_provider: CloudProvider
    status: str
    memory_total_gb: float
    memory_used_gb: float
    utilization_percent: float
    cost_per_hour: float
    region: str
    availability_zone: str

@dataclass
class ClusterMetrics:
    total_instances: int
    active_instances: int
    total_gpu_memory_gb: float
    used_gpu_memory_gb: float
    average_utilization: float
    total_cost_per_hour: float
    queue_length: int
    average_response_time: float

class GPUInfrastructureManager:
    """Manages GPU infrastructure across multiple cloud providers"""
    
    def __init__(self):
        self.gpu_instances: Dict[str, GPUInstance] = {}
        self.cluster_metrics: Dict[str, ClusterMetrics] = {}
        self.auto_scaling_enabled = True
        self.cost_optimization_enabled = True
        
        # Cloud provider clients
        self.aws_client = None
        self.gcp_client = None
        self.azure_client = None
        
        # Initialize cloud clients
        self._initialize_cloud_clients()
        
        # Kubernetes client for container orchestration
        self.k8s_client = None
        self._initialize_kubernetes()
        
    def _initialize_cloud_clients(self):
        """Initialize cloud provider clients"""
        try:
            # AWS EC2 client
            self.aws_client = boto3.client('ec2')
            logger.info("AWS client initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize AWS client: {str(e)}")
        
        try:
            # GCP Compute client
            self.gcp_client = compute_v1.InstancesClient()
            logger.info("GCP client initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize GCP client: {str(e)}")
    
    def _initialize_kubernetes(self):
        """Initialize Kubernetes client for container orchestration"""
        try:
            kubernetes.config.load_incluster_config()
            self.k8s_client = kubernetes.client.CoreV1Api()
            logger.info("Kubernetes client initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize Kubernetes client: {str(e)}")
    
    async def provision_gpu_cluster(self, cluster_name: str, gpu_type: GPUType, 
                                  instance_count: int, cloud_provider: CloudProvider) -> bool:
        """Provision GPU cluster on specified cloud provider"""
        try:
            logger.info(f"Provisioning {instance_count} {gpu_type.value} instances on {cloud_provider.value}")
            
            if cloud_provider == CloudProvider.AWS:
                return await self._provision_aws_cluster(cluster_name, gpu_type, instance_count)
            elif cloud_provider == CloudProvider.GCP:
                return await self._provision_gcp_cluster(cluster_name, gpu_type, instance_count)
            elif cloud_provider == CloudProvider.LOCAL:
                return await self._setup_local_cluster(cluster_name, gpu_type, instance_count)
            else:
                logger.error(f"Unsupported cloud provider: {cloud_provider}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to provision GPU cluster: {str(e)}")
            return False
    
    async def _provision_aws_cluster(self, cluster_name: str, gpu_type: GPUType, 
                                   instance_count: int) -> bool:
        """Provision GPU instances on AWS"""
        try:
            # Map GPU types to AWS instance types
            instance_type_mapping = {
                GPUType.A100_80GB: "p4d.24xlarge",
                GPUType.A100_40GB: "p4d.12xlarge", 
                GPUType.V100_32GB: "p3.8xlarge",
                GPUType.T4_16GB: "g4dn.xlarge"
            }
            
            instance_type = instance_type_mapping.get(gpu_type)
            if not instance_type:
                logger.error(f"Unsupported GPU type for AWS: {gpu_type}")
                return False
            
            # Launch EC2 instances
            response = await asyncio.to_thread(
                self.aws_client.run_instances,
                ImageId="ami-0c02fb55956c7d316",  # Deep Learning AMI
                MinCount=instance_count,
                MaxCount=instance_count,
                InstanceType=instance_type,
                KeyName="visual-generation-key",
                SecurityGroupIds=["sg-visual-generation"],
                SubnetId="subnet-visual-generation",
                TagSpecifications=[{
                    'ResourceType': 'instance',
                    'Tags': [
                        {'Key': 'Name', 'Value': f'{cluster_name}-gpu-instance'},
                        {'Key': 'Cluster', 'Value': cluster_name},
                        {'Key': 'Purpose', 'Value': 'visual-generation'}
                    ]
                }]
            )
            
            # Track provisioned instances
            for instance in response['Instances']:
                gpu_instance = GPUInstance(
                    instance_id=instance['InstanceId'],
                    gpu_type=gpu_type,
                    cloud_provider=CloudProvider.AWS,
                    status="launching",
                    memory_total_gb=self._get_gpu_memory(gpu_type),
                    memory_used_gb=0.0,
                    utilization_percent=0.0,
                    cost_per_hour=self._get_instance_cost(instance_type),
                    region=instance['Placement']['AvailabilityZone'][:-1],
                    availability_zone=instance['Placement']['AvailabilityZone']
                )
                self.gpu_instances[instance['InstanceId']] = gpu_instance
            
            logger.info(f"Successfully launched {instance_count} AWS instances")
            return True
            
        except Exception as e:
            logger.error(f"Failed to provision AWS cluster: {str(e)}")
            return False
    
    async def _provision_gcp_cluster(self, cluster_name: str, gpu_type: GPUType, 
                                   instance_count: int) -> bool:
        """Provision GPU instances on Google Cloud Platform"""
        try:
            # Map GPU types to GCP machine types
            machine_type_mapping = {
                GPUType.A100_80GB: "a2-highgpu-1g",
                GPUType.A100_40GB: "a2-highgpu-1g",
                GPUType.V100_32GB: "n1-standard-8",
                GPUType.T4_16GB: "n1-standard-4"
            }
            
            machine_type = machine_type_mapping.get(gpu_type)
            if not machine_type:
                logger.error(f"Unsupported GPU type for GCP: {gpu_type}")
                return False
            
            # Create instance configuration
            instance_config = {
                "name": f"{cluster_name}-gpu-instance",
                "machine_type": f"zones/us-central1-a/machineTypes/{machine_type}",
                "disks": [{
                    "boot": True,
                    "auto_delete": True,
                    "initialize_params": {
                        "source_image": "projects/deeplearning-platform-release/global/images/family/pytorch-latest-gpu"
                    }
                }],
                "network_interfaces": [{
                    "network": "global/networks/default",
                    "access_configs": [{"type": "ONE_TO_ONE_NAT", "name": "External NAT"}]
                }],
                "guest_accelerators": [{
                    "accelerator_type": f"zones/us-central1-a/acceleratorTypes/{gpu_type.value}",
                    "accelerator_count": 1
                }],
                "scheduling": {
                    "on_host_maintenance": "TERMINATE"
                },
                "tags": {
                    "items": ["visual-generation", cluster_name]
                }
            }
            
            # Launch instances
            for i in range(instance_count):
                instance_name = f"{cluster_name}-gpu-{i}"
                instance_config["name"] = instance_name
                
                operation = await asyncio.to_thread(
                    self.gcp_client.insert,
                    project="your-project-id",
                    zone="us-central1-a",
                    instance_resource=instance_config
                )
                
                # Track provisioned instance
                gpu_instance = GPUInstance(
                    instance_id=instance_name,
                    gpu_type=gpu_type,
                    cloud_provider=CloudProvider.GCP,
                    status="launching",
                    memory_total_gb=self._get_gpu_memory(gpu_type),
                    memory_used_gb=0.0,
                    utilization_percent=0.0,
                    cost_per_hour=self._get_gcp_instance_cost(machine_type),
                    region="us-central1",
                    availability_zone="us-central1-a"
                )
                self.gpu_instances[instance_name] = gpu_instance
            
            logger.info(f"Successfully launched {instance_count} GCP instances")
            return True
            
        except Exception as e:
            logger.error(f"Failed to provision GCP cluster: {str(e)}")
            return False
    
    async def _setup_local_cluster(self, cluster_name: str, gpu_type: GPUType, 
                                 instance_count: int) -> bool:
        """Setup local GPU cluster"""
        try:
            if not torch.cuda.is_available():
                logger.error("CUDA not available for local GPU cluster")
                return False
            
            gpu_count = torch.cuda.device_count()
            if gpu_count < instance_count:
                logger.warning(f"Requested {instance_count} GPUs but only {gpu_count} available")
                instance_count = gpu_count
            
            # Create local GPU instances
            for i in range(instance_count):
                device = torch.device(f"cuda:{i}")
                gpu_props = torch.cuda.get_device_properties(device)
                
                gpu_instance = GPUInstance(
                    instance_id=f"local-gpu-{i}",
                    gpu_type=gpu_type,
                    cloud_provider=CloudProvider.LOCAL,
                    status="ready",
                    memory_total_gb=gpu_props.total_memory / (1024**3),
                    memory_used_gb=torch.cuda.memory_allocated(device) / (1024**3),
                    utilization_percent=0.0,
                    cost_per_hour=0.0,  # No cost for local
                    region="local",
                    availability_zone="local"
                )
                self.gpu_instances[f"local-gpu-{i}"] = gpu_instance
            
            logger.info(f"Successfully setup local cluster with {instance_count} GPUs")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup local cluster: {str(e)}")
            return False
    
    def _get_gpu_memory(self, gpu_type: GPUType) -> float:
        """Get GPU memory in GB for given GPU type"""
        memory_mapping = {
            GPUType.A100_80GB: 80.0,
            GPUType.A100_40GB: 40.0,
            GPUType.V100_32GB: 32.0,
            GPUType.T4_16GB: 16.0,
            GPUType.RTX_4090: 24.0
        }
        return memory_mapping.get(gpu_type, 16.0)
    
    def _get_instance_cost(self, instance_type: str) -> float:
        """Get AWS instance cost per hour"""
        cost_mapping = {
            "p4d.24xlarge": 32.77,
            "p4d.12xlarge": 16.39,
            "p3.8xlarge": 12.24,
            "g4dn.xlarge": 0.526
        }
        return cost_mapping.get(instance_type, 1.0)
    
    def _get_gcp_instance_cost(self, machine_type: str) -> float:
        """Get GCP instance cost per hour"""
        cost_mapping = {
            "a2-highgpu-1g": 3.67,
            "n1-standard-8": 0.38,
            "n1-standard-4": 0.19
        }
        return cost_mapping.get(machine_type, 1.0)
    
    async def monitor_gpu_utilization(self) -> Dict[str, Any]:
        """Monitor GPU utilization across all instances"""
        try:
            utilization_data = {}
            
            for instance_id, gpu_instance in self.gpu_instances.items():
                if gpu_instance.cloud_provider == CloudProvider.LOCAL:
                    # Monitor local GPU
                    device_idx = int(instance_id.split('-')[-1])
                    device = torch.device(f"cuda:{device_idx}")
                    
                    memory_allocated = torch.cuda.memory_allocated(device)
                    memory_reserved = torch.cuda.memory_reserved(device)
                    memory_total = torch.cuda.get_device_properties(device).total_memory
                    
                    utilization = (memory_reserved / memory_total) * 100
                    
                    utilization_data[instance_id] = {
                        "memory_used_gb": memory_allocated / (1024**3),
                        "memory_reserved_gb": memory_reserved / (1024**3),
                        "memory_total_gb": memory_total / (1024**3),
                        "utilization_percent": utilization,
                        "temperature": self._get_gpu_temperature(device_idx),
                        "power_usage": self._get_gpu_power_usage(device_idx)
                    }
                    
                    # Update instance data
                    gpu_instance.memory_used_gb = memory_allocated / (1024**3)
                    gpu_instance.utilization_percent = utilization
                
                else:
                    # For cloud instances, we would query cloud provider APIs
                    # This is a simplified version
                    utilization_data[instance_id] = {
                        "memory_used_gb": gpu_instance.memory_used_gb,
                        "utilization_percent": gpu_instance.utilization_percent,
                        "status": gpu_instance.status
                    }
            
            return utilization_data
            
        except Exception as e:
            logger.error(f"Failed to monitor GPU utilization: {str(e)}")
            return {}
    
    def _get_gpu_temperature(self, device_idx: int) -> float:
        """Get GPU temperature (simplified implementation)"""
        try:
            # In production, this would use nvidia-ml-py or similar
            return 65.0  # Placeholder temperature
        except:
            return 0.0
    
    def _get_gpu_power_usage(self, device_idx: int) -> float:
        """Get GPU power usage in watts (simplified implementation)"""
        try:
            # In production, this would use nvidia-ml-py or similar
            return 250.0  # Placeholder power usage
        except:
            return 0.0
    
    async def auto_scale_cluster(self, cluster_name: str, target_utilization: float = 80.0) -> bool:
        """Automatically scale cluster based on utilization"""
        try:
            if not self.auto_scaling_enabled:
                return False
            
            # Calculate current cluster utilization
            cluster_instances = [
                instance for instance in self.gpu_instances.values()
                if cluster_name in instance.instance_id
            ]
            
            if not cluster_instances:
                logger.warning(f"No instances found for cluster {cluster_name}")
                return False
            
            total_utilization = sum(instance.utilization_percent for instance in cluster_instances)
            average_utilization = total_utilization / len(cluster_instances)
            
            logger.info(f"Cluster {cluster_name} average utilization: {average_utilization:.2f}%")
            
            # Scale up if utilization is high
            if average_utilization > target_utilization:
                scale_up_count = max(1, int(len(cluster_instances) * 0.5))  # Scale up by 50%
                logger.info(f"Scaling up cluster {cluster_name} by {scale_up_count} instances")
                
                # Get cluster configuration
                sample_instance = cluster_instances[0]
                return await self.provision_gpu_cluster(
                    cluster_name,
                    sample_instance.gpu_type,
                    scale_up_count,
                    sample_instance.cloud_provider
                )
            
            # Scale down if utilization is low
            elif average_utilization < target_utilization * 0.3:  # Scale down threshold
                scale_down_count = max(1, int(len(cluster_instances) * 0.3))  # Scale down by 30%
                logger.info(f"Scaling down cluster {cluster_name} by {scale_down_count} instances")
                
                # Terminate least utilized instances
                sorted_instances = sorted(cluster_instances, key=lambda x: x.utilization_percent)
                instances_to_terminate = sorted_instances[:scale_down_count]
                
                for instance in instances_to_terminate:
                    await self.terminate_instance(instance.instance_id)
                
                return True
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to auto-scale cluster {cluster_name}: {str(e)}")
            return False
    
    async def terminate_instance(self, instance_id: str) -> bool:
        """Terminate GPU instance"""
        try:
            if instance_id not in self.gpu_instances:
                logger.error(f"Instance {instance_id} not found")
                return False
            
            gpu_instance = self.gpu_instances[instance_id]
            
            if gpu_instance.cloud_provider == CloudProvider.AWS:
                await asyncio.to_thread(
                    self.aws_client.terminate_instances,
                    InstanceIds=[instance_id]
                )
            elif gpu_instance.cloud_provider == CloudProvider.GCP:
                await asyncio.to_thread(
                    self.gcp_client.delete,
                    project="your-project-id",
                    zone=gpu_instance.availability_zone,
                    instance=instance_id
                )
            
            # Remove from tracking
            del self.gpu_instances[instance_id]
            
            logger.info(f"Successfully terminated instance {instance_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to terminate instance {instance_id}: {str(e)}")
            return False
    
    async def optimize_costs(self) -> Dict[str, Any]:
        """Optimize GPU infrastructure costs"""
        try:
            optimization_report = {
                "current_cost_per_hour": 0.0,
                "optimized_cost_per_hour": 0.0,
                "potential_savings": 0.0,
                "recommendations": []
            }
            
            # Calculate current costs
            current_cost = sum(instance.cost_per_hour for instance in self.gpu_instances.values())
            optimization_report["current_cost_per_hour"] = current_cost
            
            # Analyze utilization patterns
            low_utilization_instances = [
                instance for instance in self.gpu_instances.values()
                if instance.utilization_percent < 20.0
            ]
            
            # Generate recommendations
            if low_utilization_instances:
                potential_savings = sum(instance.cost_per_hour for instance in low_utilization_instances)
                optimization_report["potential_savings"] = potential_savings
                optimization_report["recommendations"].append({
                    "type": "terminate_underutilized",
                    "description": f"Terminate {len(low_utilization_instances)} underutilized instances",
                    "savings_per_hour": potential_savings
                })
            
            # Spot instance recommendations
            spot_savings = current_cost * 0.7  # Assume 70% savings with spot instances
            optimization_report["recommendations"].append({
                "type": "use_spot_instances",
                "description": "Switch to spot instances for non-critical workloads",
                "savings_per_hour": spot_savings
            })
            
            optimization_report["optimized_cost_per_hour"] = current_cost - optimization_report["potential_savings"]
            
            return optimization_report
            
        except Exception as e:
            logger.error(f"Failed to optimize costs: {str(e)}")
            return {}
    
    async def get_cluster_status(self) -> Dict[str, ClusterMetrics]:
        """Get comprehensive cluster status"""
        try:
            cluster_status = {}
            
            # Group instances by cluster
            clusters = {}
            for instance in self.gpu_instances.values():
                cluster_name = instance.instance_id.split('-')[0]  # Extract cluster name
                if cluster_name not in clusters:
                    clusters[cluster_name] = []
                clusters[cluster_name].append(instance)
            
            # Calculate metrics for each cluster
            for cluster_name, instances in clusters.items():
                total_memory = sum(instance.memory_total_gb for instance in instances)
                used_memory = sum(instance.memory_used_gb for instance in instances)
                avg_utilization = sum(instance.utilization_percent for instance in instances) / len(instances)
                total_cost = sum(instance.cost_per_hour for instance in instances)
                
                cluster_metrics = ClusterMetrics(
                    total_instances=len(instances),
                    active_instances=len([i for i in instances if i.status == "ready"]),
                    total_gpu_memory_gb=total_memory,
                    used_gpu_memory_gb=used_memory,
                    average_utilization=avg_utilization,
                    total_cost_per_hour=total_cost,
                    queue_length=0,  # Would be populated from job queue
                    average_response_time=0.0  # Would be calculated from metrics
                )
                
                cluster_status[cluster_name] = cluster_metrics
            
            return cluster_status
            
        except Exception as e:
            logger.error(f"Failed to get cluster status: {str(e)}")
            return {}

# Global infrastructure manager instance
gpu_infrastructure = GPUInfrastructureManager()

async def initialize_gpu_infrastructure():
    """Initialize GPU infrastructure"""
    try:
        # Setup local cluster for development/testing
        await gpu_infrastructure.provision_gpu_cluster(
            "local-dev", GPUType.RTX_4090, 1, CloudProvider.LOCAL
        )
        
        logger.info("GPU infrastructure initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize GPU infrastructure: {str(e)}")
        return False

async def get_infrastructure_status():
    """Get infrastructure status"""
    return await gpu_infrastructure.get_cluster_status()