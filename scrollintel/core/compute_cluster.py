"""
Scalable Compute Cluster Management with Kubernetes Orchestration
"""
from kubernetes import client, config
from kubernetes.client.rest import ApiException
import yaml
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum
import time

logger = logging.getLogger(__name__)

class DeploymentStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    FAILED = "failed"
    SUCCEEDED = "succeeded"

@dataclass
class ComputeResource:
    """Compute resource specification"""
    cpu_request: str = "100m"
    cpu_limit: str = "500m"
    memory_request: str = "128Mi"
    memory_limit: str = "512Mi"
    gpu_request: int = 0

@dataclass
class ScalingConfig:
    """Auto-scaling configuration"""
    min_replicas: int = 1
    max_replicas: int = 10
    target_cpu_utilization: int = 70
    target_memory_utilization: int = 80

class KubernetesClusterManager:
    """Manages Kubernetes compute cluster for scalable workloads"""
    
    def __init__(self, namespace: str = "scrollintel-g6"):
        self.namespace = namespace
        self.v1 = None
        self.apps_v1 = None
        self.autoscaling_v1 = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Kubernetes client"""
        try:
            # Try to load in-cluster config first
            config.load_incluster_config()
            logger.info("Loaded in-cluster Kubernetes config")
        except config.ConfigException:
            try:
                # Fallback to local kubeconfig
                config.load_kube_config()
                logger.info("Loaded local Kubernetes config")
            except config.ConfigException as e:
                logger.error(f"Failed to load Kubernetes config: {e}")
                raise
        
        self.v1 = client.CoreV1Api()
        self.apps_v1 = client.AppsV1Api()
        self.autoscaling_v1 = client.AutoscalingV1Api()
        
        # Ensure namespace exists
        self._ensure_namespace()
    
    def _ensure_namespace(self):
        """Ensure the namespace exists"""
        try:
            self.v1.read_namespace(name=self.namespace)
            logger.info(f"Namespace {self.namespace} exists")
        except ApiException as e:
            if e.status == 404:
                # Create namespace
                namespace_body = client.V1Namespace(
                    metadata=client.V1ObjectMeta(name=self.namespace)
                )
                self.v1.create_namespace(body=namespace_body)
                logger.info(f"Created namespace {self.namespace}")
            else:
                logger.error(f"Failed to check namespace: {e}")
                raise
    
    def create_deployment(self, 
                         name: str, 
                         image: str, 
                         replicas: int = 1,
                         resources: Optional[ComputeResource] = None,
                         env_vars: Optional[Dict[str, str]] = None,
                         ports: Optional[List[int]] = None) -> bool:
        """Create a Kubernetes deployment"""
        try:
            if resources is None:
                resources = ComputeResource()
            
            # Container specification
            container = client.V1Container(
                name=name,
                image=image,
                resources=client.V1ResourceRequirements(
                    requests={
                        "cpu": resources.cpu_request,
                        "memory": resources.memory_request
                    },
                    limits={
                        "cpu": resources.cpu_limit,
                        "memory": resources.memory_limit
                    }
                )
            )
            
            # Add GPU resources if requested
            if resources.gpu_request > 0:
                container.resources.requests["nvidia.com/gpu"] = str(resources.gpu_request)
                container.resources.limits["nvidia.com/gpu"] = str(resources.gpu_request)
            
            # Add environment variables
            if env_vars:
                container.env = [
                    client.V1EnvVar(name=k, value=v) 
                    for k, v in env_vars.items()
                ]
            
            # Add ports
            if ports:
                container.ports = [
                    client.V1ContainerPort(container_port=port) 
                    for port in ports
                ]
            
            # Pod template
            template = client.V1PodTemplateSpec(
                metadata=client.V1ObjectMeta(labels={"app": name}),
                spec=client.V1PodSpec(containers=[container])
            )
            
            # Deployment specification
            spec = client.V1DeploymentSpec(
                replicas=replicas,
                selector=client.V1LabelSelector(
                    match_labels={"app": name}
                ),
                template=template
            )
            
            # Deployment object
            deployment = client.V1Deployment(
                api_version="apps/v1",
                kind="Deployment",
                metadata=client.V1ObjectMeta(name=name),
                spec=spec
            )
            
            # Create deployment
            self.apps_v1.create_namespaced_deployment(
                body=deployment,
                namespace=self.namespace
            )
            
            logger.info(f"Created deployment {name}")
            return True
            
        except ApiException as e:
            logger.error(f"Failed to create deployment {name}: {e}")
            return False
    
    def create_service(self, 
                      name: str, 
                      selector: Dict[str, str],
                      ports: List[Dict[str, Any]],
                      service_type: str = "ClusterIP") -> bool:
        """Create a Kubernetes service"""
        try:
            service_ports = [
                client.V1ServicePort(
                    name=port.get("name", f"port-{port['port']}"),
                    port=port["port"],
                    target_port=port.get("target_port", port["port"]),
                    protocol=port.get("protocol", "TCP")
                )
                for port in ports
            ]
            
            service = client.V1Service(
                api_version="v1",
                kind="Service",
                metadata=client.V1ObjectMeta(name=name),
                spec=client.V1ServiceSpec(
                    selector=selector,
                    ports=service_ports,
                    type=service_type
                )
            )
            
            self.v1.create_namespaced_service(
                body=service,
                namespace=self.namespace
            )
            
            logger.info(f"Created service {name}")
            return True
            
        except ApiException as e:
            logger.error(f"Failed to create service {name}: {e}")
            return False
    
    def create_hpa(self, 
                   name: str, 
                   deployment_name: str,
                   scaling_config: ScalingConfig) -> bool:
        """Create Horizontal Pod Autoscaler"""
        try:
            hpa = client.V1HorizontalPodAutoscaler(
                api_version="autoscaling/v1",
                kind="HorizontalPodAutoscaler",
                metadata=client.V1ObjectMeta(name=name),
                spec=client.V1HorizontalPodAutoscalerSpec(
                    scale_target_ref=client.V1CrossVersionObjectReference(
                        api_version="apps/v1",
                        kind="Deployment",
                        name=deployment_name
                    ),
                    min_replicas=scaling_config.min_replicas,
                    max_replicas=scaling_config.max_replicas,
                    target_cpu_utilization_percentage=scaling_config.target_cpu_utilization
                )
            )
            
            self.autoscaling_v1.create_namespaced_horizontal_pod_autoscaler(
                body=hpa,
                namespace=self.namespace
            )
            
            logger.info(f"Created HPA {name}")
            return True
            
        except ApiException as e:
            logger.error(f"Failed to create HPA {name}: {e}")
            return False
    
    def get_deployment_status(self, name: str) -> Dict[str, Any]:
        """Get deployment status"""
        try:
            deployment = self.apps_v1.read_namespaced_deployment(
                name=name,
                namespace=self.namespace
            )
            
            status = deployment.status
            return {
                "name": name,
                "replicas": status.replicas or 0,
                "ready_replicas": status.ready_replicas or 0,
                "available_replicas": status.available_replicas or 0,
                "unavailable_replicas": status.unavailable_replicas or 0,
                "conditions": [
                    {
                        "type": condition.type,
                        "status": condition.status,
                        "reason": condition.reason,
                        "message": condition.message
                    }
                    for condition in (status.conditions or [])
                ]
            }
            
        except ApiException as e:
            logger.error(f"Failed to get deployment status {name}: {e}")
            return {"name": name, "error": str(e)}
    
    def scale_deployment(self, name: str, replicas: int) -> bool:
        """Scale deployment to specified number of replicas"""
        try:
            # Get current deployment
            deployment = self.apps_v1.read_namespaced_deployment(
                name=name,
                namespace=self.namespace
            )
            
            # Update replica count
            deployment.spec.replicas = replicas
            
            # Apply update
            self.apps_v1.patch_namespaced_deployment(
                name=name,
                namespace=self.namespace,
                body=deployment
            )
            
            logger.info(f"Scaled deployment {name} to {replicas} replicas")
            return True
            
        except ApiException as e:
            logger.error(f"Failed to scale deployment {name}: {e}")
            return False
    
    def delete_deployment(self, name: str) -> bool:
        """Delete deployment"""
        try:
            self.apps_v1.delete_namespaced_deployment(
                name=name,
                namespace=self.namespace
            )
            logger.info(f"Deleted deployment {name}")
            return True
            
        except ApiException as e:
            logger.error(f"Failed to delete deployment {name}: {e}")
            return False
    
    def get_cluster_resources(self) -> Dict[str, Any]:
        """Get cluster resource information"""
        try:
            nodes = self.v1.list_node()
            
            total_cpu = 0
            total_memory = 0
            allocatable_cpu = 0
            allocatable_memory = 0
            
            for node in nodes.items:
                if node.status.capacity:
                    cpu = node.status.capacity.get("cpu", "0")
                    memory = node.status.capacity.get("memory", "0Ki")
                    
                    # Parse CPU (can be in millicores or cores)
                    if cpu.endswith("m"):
                        total_cpu += int(cpu[:-1])
                    else:
                        total_cpu += int(cpu) * 1000
                    
                    # Parse memory (convert to bytes)
                    if memory.endswith("Ki"):
                        total_memory += int(memory[:-2]) * 1024
                    elif memory.endswith("Mi"):
                        total_memory += int(memory[:-2]) * 1024 * 1024
                    elif memory.endswith("Gi"):
                        total_memory += int(memory[:-2]) * 1024 * 1024 * 1024
                
                if node.status.allocatable:
                    cpu = node.status.allocatable.get("cpu", "0")
                    memory = node.status.allocatable.get("memory", "0Ki")
                    
                    if cpu.endswith("m"):
                        allocatable_cpu += int(cpu[:-1])
                    else:
                        allocatable_cpu += int(cpu) * 1000
                    
                    if memory.endswith("Ki"):
                        allocatable_memory += int(memory[:-2]) * 1024
                    elif memory.endswith("Mi"):
                        allocatable_memory += int(memory[:-2]) * 1024 * 1024
                    elif memory.endswith("Gi"):
                        allocatable_memory += int(memory[:-2]) * 1024 * 1024 * 1024
            
            return {
                "nodes": len(nodes.items),
                "total_cpu_millicores": total_cpu,
                "total_memory_bytes": total_memory,
                "allocatable_cpu_millicores": allocatable_cpu,
                "allocatable_memory_bytes": allocatable_memory
            }
            
        except ApiException as e:
            logger.error(f"Failed to get cluster resources: {e}")
            return {"error": str(e)}

# Global cluster manager instance
_cluster_manager: Optional[KubernetesClusterManager] = None

def get_cluster_manager() -> KubernetesClusterManager:
    """Get global cluster manager instance"""
    global _cluster_manager
    
    if _cluster_manager is None:
        _cluster_manager = KubernetesClusterManager()
    
    return _cluster_manager