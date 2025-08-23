"""
Blue-Green and Canary Deployment Strategies with Security Validation
Implements secure deployment strategies for DevSecOps pipeline
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
from dataclasses import dataclass
import json
import time
import kubernetes
from kubernetes import client, config

logger = logging.getLogger(__name__)

class DeploymentStrategy(Enum):
    BLUE_GREEN = "blue_green"
    CANARY = "canary"
    ROLLING = "rolling"
    RECREATE = "recreate"

class DeploymentStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    VALIDATING = "validating"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"

class SecurityValidationStatus(Enum):
    PENDING = "pending"
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"

@dataclass
class SecurityValidation:
    validation_id: str
    name: str
    status: SecurityValidationStatus
    score: float
    findings: List[Dict[str, Any]]
    execution_time: float
    timestamp: datetime

@dataclass
class DeploymentConfig:
    strategy: DeploymentStrategy
    namespace: str
    service_name: str
    image: str
    replicas: int
    security_validations: List[str]
    rollback_threshold: float
    validation_timeout: int
    traffic_split_config: Optional[Dict[str, Any]] = None
    canary_config: Optional[Dict[str, Any]] = None

@dataclass
class DeploymentResult:
    deployment_id: str
    strategy: DeploymentStrategy
    status: DeploymentStatus
    start_time: datetime
    end_time: Optional[datetime]
    security_validations: List[SecurityValidation]
    metrics: Dict[str, Any]
    rollback_reason: Optional[str] = None

class SecureDeploymentStrategies:
    """
    Implements blue-green and canary deployment strategies with security validation
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.k8s_client = self._initialize_k8s_client()
        self.security_validators = {
            "runtime_security_scan": self._validate_runtime_security,
            "network_policy_check": self._validate_network_policies,
            "rbac_validation": self._validate_rbac,
            "secret_scanning": self._validate_secrets,
            "compliance_check": self._validate_compliance,
            "performance_test": self._validate_performance
        }
        self.active_deployments = {}
        
    def _initialize_k8s_client(self):
        """Initialize Kubernetes client"""
        try:
            config.load_incluster_config()
        except config.ConfigException:
            config.load_kube_config()
        
        return client.ApiClient()
    
    async def deploy_blue_green(
        self, 
        deployment_config: DeploymentConfig
    ) -> DeploymentResult:
        """Execute blue-green deployment with security validation"""
        deployment_id = f"bg_{deployment_config.service_name}_{int(time.time())}"
        start_time = datetime.now()
        
        logger.info(f"Starting blue-green deployment {deployment_id}")
        
        try:
            # Create deployment result
            result = DeploymentResult(
                deployment_id=deployment_id,
                strategy=DeploymentStrategy.BLUE_GREEN,
                status=DeploymentStatus.PENDING,
                start_time=start_time,
                end_time=None,
                security_validations=[],
                metrics={}
            )
            
            self.active_deployments[deployment_id] = result
            
            # Step 1: Deploy green environment
            result.status = DeploymentStatus.IN_PROGRESS
            green_deployment = await self._create_green_deployment(deployment_config)
            
            # Step 2: Wait for green deployment to be ready
            await self._wait_for_deployment_ready(green_deployment, deployment_config.namespace)
            
            # Step 3: Run security validations on green environment
            result.status = DeploymentStatus.VALIDATING
            security_validations = await self._run_security_validations(
                deployment_config, green_deployment
            )
            result.security_validations = security_validations
            
            # Step 4: Check if security validations passed
            security_passed = self._evaluate_security_validations(security_validations)
            
            if not security_passed:
                logger.error(f"Security validations failed for deployment {deployment_id}")
                await self._cleanup_green_deployment(green_deployment, deployment_config.namespace)
                result.status = DeploymentStatus.FAILED
                result.rollback_reason = "Security validation failures"
                result.end_time = datetime.now()
                return result
            
            # Step 5: Switch traffic to green (blue-green cutover)
            await self._switch_traffic_to_green(deployment_config, green_deployment)
            
            # Step 6: Monitor post-deployment metrics
            post_deployment_metrics = await self._monitor_post_deployment(
                deployment_config, green_deployment, duration_minutes=5
            )
            result.metrics = post_deployment_metrics
            
            # Step 7: Check if rollback is needed based on metrics
            if self._should_rollback(post_deployment_metrics, deployment_config.rollback_threshold):
                logger.warning(f"Rolling back deployment {deployment_id} due to poor metrics")
                await self._rollback_blue_green(deployment_config, green_deployment)
                result.status = DeploymentStatus.ROLLED_BACK
                result.rollback_reason = "Poor post-deployment metrics"
            else:
                # Step 8: Cleanup old blue deployment
                await self._cleanup_blue_deployment(deployment_config)
                result.status = DeploymentStatus.COMPLETED
            
            result.end_time = datetime.now()
            logger.info(f"Blue-green deployment {deployment_id} completed with status: {result.status}")
            
            return result
            
        except Exception as e:
            logger.error(f"Blue-green deployment {deployment_id} failed: {str(e)}")
            result.status = DeploymentStatus.FAILED
            result.rollback_reason = f"Deployment error: {str(e)}"
            result.end_time = datetime.now()
            return result
    
    async def deploy_canary(
        self, 
        deployment_config: DeploymentConfig
    ) -> DeploymentResult:
        """Execute canary deployment with security validation"""
        deployment_id = f"canary_{deployment_config.service_name}_{int(time.time())}"
        start_time = datetime.now()
        
        logger.info(f"Starting canary deployment {deployment_id}")
        
        try:
            result = DeploymentResult(
                deployment_id=deployment_id,
                strategy=DeploymentStrategy.CANARY,
                status=DeploymentStatus.PENDING,
                start_time=start_time,
                end_time=None,
                security_validations=[],
                metrics={}
            )
            
            self.active_deployments[deployment_id] = result
            
            # Get canary configuration
            canary_config = deployment_config.canary_config or {
                "initial_traffic": 10,
                "increment_traffic": 20,
                "max_traffic": 100,
                "validation_interval": 300  # 5 minutes
            }
            
            # Step 1: Deploy canary version
            result.status = DeploymentStatus.IN_PROGRESS
            canary_deployment = await self._create_canary_deployment(deployment_config)
            
            # Step 2: Wait for canary deployment to be ready
            await self._wait_for_deployment_ready(canary_deployment, deployment_config.namespace)
            
            # Step 3: Run security validations on canary
            result.status = DeploymentStatus.VALIDATING
            security_validations = await self._run_security_validations(
                deployment_config, canary_deployment
            )
            result.security_validations = security_validations
            
            # Step 4: Check if security validations passed
            security_passed = self._evaluate_security_validations(security_validations)
            
            if not security_passed:
                logger.error(f"Security validations failed for canary deployment {deployment_id}")
                await self._cleanup_canary_deployment(canary_deployment, deployment_config.namespace)
                result.status = DeploymentStatus.FAILED
                result.rollback_reason = "Security validation failures"
                result.end_time = datetime.now()
                return result
            
            # Step 5: Gradual traffic shifting with monitoring
            current_traffic = canary_config["initial_traffic"]
            all_metrics = {}
            
            while current_traffic <= canary_config["max_traffic"]:
                # Update traffic split
                await self._update_traffic_split(
                    deployment_config, canary_deployment, current_traffic
                )
                
                # Monitor for validation interval
                logger.info(f"Monitoring canary with {current_traffic}% traffic for {canary_config['validation_interval']}s")
                await asyncio.sleep(canary_config["validation_interval"])
                
                # Collect metrics
                metrics = await self._collect_canary_metrics(
                    deployment_config, canary_deployment, current_traffic
                )
                all_metrics[f"traffic_{current_traffic}"] = metrics
                
                # Check if rollback is needed
                if self._should_rollback(metrics, deployment_config.rollback_threshold):
                    logger.warning(f"Rolling back canary deployment {deployment_id} at {current_traffic}% traffic")
                    await self._rollback_canary(deployment_config, canary_deployment)
                    result.status = DeploymentStatus.ROLLED_BACK
                    result.rollback_reason = f"Poor metrics at {current_traffic}% traffic"
                    result.metrics = all_metrics
                    result.end_time = datetime.now()
                    return result
                
                # Increment traffic
                if current_traffic < canary_config["max_traffic"]:
                    current_traffic = min(
                        current_traffic + canary_config["increment_traffic"],
                        canary_config["max_traffic"]
                    )
                else:
                    break
            
            # Step 6: Complete canary deployment
            await self._complete_canary_deployment(deployment_config, canary_deployment)
            result.status = DeploymentStatus.COMPLETED
            result.metrics = all_metrics
            result.end_time = datetime.now()
            
            logger.info(f"Canary deployment {deployment_id} completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Canary deployment {deployment_id} failed: {str(e)}")
            result.status = DeploymentStatus.FAILED
            result.rollback_reason = f"Deployment error: {str(e)}"
            result.end_time = datetime.now()
            return result
    
    async def _create_green_deployment(self, config: DeploymentConfig) -> str:
        """Create green deployment for blue-green strategy"""
        green_name = f"{config.service_name}-green"
        
        # Create deployment manifest
        deployment_manifest = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": green_name,
                "namespace": config.namespace,
                "labels": {
                    "app": config.service_name,
                    "version": "green",
                    "deployment-strategy": "blue-green"
                }
            },
            "spec": {
                "replicas": config.replicas,
                "selector": {
                    "matchLabels": {
                        "app": config.service_name,
                        "version": "green"
                    }
                },
                "template": {
                    "metadata": {
                        "labels": {
                            "app": config.service_name,
                            "version": "green"
                        }
                    },
                    "spec": {
                        "containers": [{
                            "name": config.service_name,
                            "image": config.image,
                            "ports": [{"containerPort": 8080}],
                            "securityContext": {
                                "runAsNonRoot": True,
                                "runAsUser": 1000,
                                "readOnlyRootFilesystem": True,
                                "allowPrivilegeEscalation": False
                            },
                            "resources": {
                                "requests": {"memory": "256Mi", "cpu": "250m"},
                                "limits": {"memory": "512Mi", "cpu": "500m"}
                            }
                        }]
                    }
                }
            }
        }
        
        # Create deployment
        apps_v1 = client.AppsV1Api(self.k8s_client)
        apps_v1.create_namespaced_deployment(
            namespace=config.namespace,
            body=deployment_manifest
        )
        
        return green_name
    
    async def _create_canary_deployment(self, config: DeploymentConfig) -> str:
        """Create canary deployment"""
        canary_name = f"{config.service_name}-canary"
        
        # Calculate canary replicas (start with 1 replica)
        canary_replicas = 1
        
        deployment_manifest = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": canary_name,
                "namespace": config.namespace,
                "labels": {
                    "app": config.service_name,
                    "version": "canary",
                    "deployment-strategy": "canary"
                }
            },
            "spec": {
                "replicas": canary_replicas,
                "selector": {
                    "matchLabels": {
                        "app": config.service_name,
                        "version": "canary"
                    }
                },
                "template": {
                    "metadata": {
                        "labels": {
                            "app": config.service_name,
                            "version": "canary"
                        }
                    },
                    "spec": {
                        "containers": [{
                            "name": config.service_name,
                            "image": config.image,
                            "ports": [{"containerPort": 8080}],
                            "securityContext": {
                                "runAsNonRoot": True,
                                "runAsUser": 1000,
                                "readOnlyRootFilesystem": True,
                                "allowPrivilegeEscalation": False
                            },
                            "resources": {
                                "requests": {"memory": "256Mi", "cpu": "250m"},
                                "limits": {"memory": "512Mi", "cpu": "500m"}
                            }
                        }]
                    }
                }
            }
        }
        
        apps_v1 = client.AppsV1Api(self.k8s_client)
        apps_v1.create_namespaced_deployment(
            namespace=config.namespace,
            body=deployment_manifest
        )
        
        return canary_name
    
    async def _wait_for_deployment_ready(self, deployment_name: str, namespace: str):
        """Wait for deployment to be ready"""
        apps_v1 = client.AppsV1Api(self.k8s_client)
        
        for _ in range(60):  # Wait up to 10 minutes
            try:
                deployment = apps_v1.read_namespaced_deployment(
                    name=deployment_name,
                    namespace=namespace
                )
                
                if (deployment.status.ready_replicas and 
                    deployment.status.ready_replicas == deployment.spec.replicas):
                    logger.info(f"Deployment {deployment_name} is ready")
                    return
                
                await asyncio.sleep(10)
                
            except Exception as e:
                logger.error(f"Error checking deployment status: {str(e)}")
                await asyncio.sleep(10)
        
        raise Exception(f"Deployment {deployment_name} did not become ready within timeout")
    
    async def _run_security_validations(
        self, 
        config: DeploymentConfig,
        deployment_name: str
    ) -> List[SecurityValidation]:
        """Run security validations on deployment"""
        validations = []
        
        for validation_name in config.security_validations:
            if validation_name in self.security_validators:
                logger.info(f"Running security validation: {validation_name}")
                
                start_time = datetime.now()
                try:
                    validator = self.security_validators[validation_name]
                    result = await validator(config, deployment_name)
                    
                    execution_time = (datetime.now() - start_time).total_seconds()
                    
                    validation = SecurityValidation(
                        validation_id=f"{deployment_name}_{validation_name}",
                        name=validation_name,
                        status=result["status"],
                        score=result["score"],
                        findings=result["findings"],
                        execution_time=execution_time,
                        timestamp=start_time
                    )
                    
                    validations.append(validation)
                    
                except Exception as e:
                    logger.error(f"Security validation {validation_name} failed: {str(e)}")
                    
                    validation = SecurityValidation(
                        validation_id=f"{deployment_name}_{validation_name}",
                        name=validation_name,
                        status=SecurityValidationStatus.FAILED,
                        score=0.0,
                        findings=[{"error": str(e)}],
                        execution_time=(datetime.now() - start_time).total_seconds(),
                        timestamp=start_time
                    )
                    
                    validations.append(validation)
        
        return validations
    
    async def _validate_runtime_security(
        self, 
        config: DeploymentConfig,
        deployment_name: str
    ) -> Dict[str, Any]:
        """Validate runtime security of deployment"""
        # Simulate runtime security scan
        await asyncio.sleep(2)
        
        findings = []
        score = 85.0
        
        # Check for common runtime security issues
        # This would integrate with actual runtime security tools
        
        return {
            "status": SecurityValidationStatus.PASSED,
            "score": score,
            "findings": findings
        }
    
    async def _validate_network_policies(
        self, 
        config: DeploymentConfig,
        deployment_name: str
    ) -> Dict[str, Any]:
        """Validate network policies"""
        await asyncio.sleep(1)
        
        findings = []
        score = 90.0
        
        # Check if network policies are properly configured
        networking_v1 = client.NetworkingV1Api(self.k8s_client)
        
        try:
            policies = networking_v1.list_namespaced_network_policy(namespace=config.namespace)
            
            if not policies.items:
                findings.append({
                    "severity": "high",
                    "message": "No network policies found in namespace",
                    "recommendation": "Implement network policies for micro-segmentation"
                })
                score = 60.0
            
        except Exception as e:
            findings.append({
                "severity": "medium",
                "message": f"Could not validate network policies: {str(e)}"
            })
            score = 70.0
        
        status = SecurityValidationStatus.PASSED if score >= 80 else SecurityValidationStatus.WARNING
        
        return {
            "status": status,
            "score": score,
            "findings": findings
        }
    
    async def _validate_rbac(
        self, 
        config: DeploymentConfig,
        deployment_name: str
    ) -> Dict[str, Any]:
        """Validate RBAC configuration"""
        await asyncio.sleep(1)
        
        findings = []
        score = 95.0
        
        # Check RBAC configuration
        # This would validate service accounts, roles, and role bindings
        
        return {
            "status": SecurityValidationStatus.PASSED,
            "score": score,
            "findings": findings
        }
    
    async def _validate_secrets(
        self, 
        config: DeploymentConfig,
        deployment_name: str
    ) -> Dict[str, Any]:
        """Validate secrets management"""
        await asyncio.sleep(1)
        
        findings = []
        score = 88.0
        
        # Check for proper secrets management
        # This would scan for hardcoded secrets, proper secret mounting, etc.
        
        return {
            "status": SecurityValidationStatus.PASSED,
            "score": score,
            "findings": findings
        }
    
    async def _validate_compliance(
        self, 
        config: DeploymentConfig,
        deployment_name: str
    ) -> Dict[str, Any]:
        """Validate compliance requirements"""
        await asyncio.sleep(2)
        
        findings = []
        score = 92.0
        
        # Check compliance with various frameworks
        # This would validate against SOC2, GDPR, etc.
        
        return {
            "status": SecurityValidationStatus.PASSED,
            "score": score,
            "findings": findings
        }
    
    async def _validate_performance(
        self, 
        config: DeploymentConfig,
        deployment_name: str
    ) -> Dict[str, Any]:
        """Validate performance requirements"""
        await asyncio.sleep(3)
        
        findings = []
        score = 87.0
        
        # Run performance tests
        # This would execute load tests, check response times, etc.
        
        return {
            "status": SecurityValidationStatus.PASSED,
            "score": score,
            "findings": findings
        }
    
    def _evaluate_security_validations(self, validations: List[SecurityValidation]) -> bool:
        """Evaluate if security validations passed"""
        if not validations:
            return False
        
        # Check if any validation failed
        failed_validations = [v for v in validations if v.status == SecurityValidationStatus.FAILED]
        if failed_validations:
            return False
        
        # Check average score
        average_score = sum(v.score for v in validations) / len(validations)
        if average_score < 80.0:
            return False
        
        return True
    
    async def _switch_traffic_to_green(self, config: DeploymentConfig, green_deployment: str):
        """Switch traffic from blue to green"""
        # Update service selector to point to green deployment
        core_v1 = client.CoreV1Api(self.k8s_client)
        
        service = core_v1.read_namespaced_service(
            name=config.service_name,
            namespace=config.namespace
        )
        
        # Update selector to green version
        service.spec.selector["version"] = "green"
        
        core_v1.patch_namespaced_service(
            name=config.service_name,
            namespace=config.namespace,
            body=service
        )
        
        logger.info(f"Traffic switched to green deployment: {green_deployment}")
    
    async def _update_traffic_split(
        self, 
        config: DeploymentConfig,
        canary_deployment: str,
        traffic_percentage: int
    ):
        """Update traffic split for canary deployment"""
        # This would typically use Istio, Linkerd, or similar service mesh
        # For now, simulate traffic splitting
        
        logger.info(f"Updated traffic split: {traffic_percentage}% to canary {canary_deployment}")
        
        # In a real implementation, this would update VirtualService or similar resources
        # to control traffic distribution between stable and canary versions
    
    async def _monitor_post_deployment(
        self, 
        config: DeploymentConfig,
        deployment_name: str,
        duration_minutes: int
    ) -> Dict[str, Any]:
        """Monitor post-deployment metrics"""
        logger.info(f"Monitoring {deployment_name} for {duration_minutes} minutes")
        
        # Simulate monitoring
        await asyncio.sleep(duration_minutes * 60)
        
        # Return simulated metrics
        return {
            "error_rate": 0.02,  # 2% error rate
            "response_time_p95": 150,  # 150ms
            "cpu_usage": 45.0,  # 45% CPU
            "memory_usage": 60.0,  # 60% memory
            "request_rate": 1000  # 1000 requests/minute
        }
    
    async def _collect_canary_metrics(
        self, 
        config: DeploymentConfig,
        canary_deployment: str,
        traffic_percentage: int
    ) -> Dict[str, Any]:
        """Collect metrics for canary deployment"""
        # Simulate metric collection
        await asyncio.sleep(30)
        
        return {
            "traffic_percentage": traffic_percentage,
            "error_rate": 0.01,  # 1% error rate
            "response_time_p95": 140,  # 140ms
            "cpu_usage": 40.0,  # 40% CPU
            "memory_usage": 55.0,  # 55% memory
            "request_rate": traffic_percentage * 10  # Proportional to traffic
        }
    
    def _should_rollback(self, metrics: Dict[str, Any], threshold: float) -> bool:
        """Determine if rollback is needed based on metrics"""
        error_rate = metrics.get("error_rate", 0.0)
        response_time = metrics.get("response_time_p95", 0)
        
        # Rollback conditions
        if error_rate > 0.05:  # 5% error rate
            return True
        
        if response_time > 500:  # 500ms response time
            return True
        
        # Custom threshold check
        overall_score = 100 - (error_rate * 1000) - (response_time / 10)
        if overall_score < threshold:
            return True
        
        return False
    
    async def _rollback_blue_green(self, config: DeploymentConfig, green_deployment: str):
        """Rollback blue-green deployment"""
        logger.info(f"Rolling back blue-green deployment: {green_deployment}")
        
        # Switch traffic back to blue
        core_v1 = client.CoreV1Api(self.k8s_client)
        service = core_v1.read_namespaced_service(
            name=config.service_name,
            namespace=config.namespace
        )
        
        service.spec.selector["version"] = "blue"
        core_v1.patch_namespaced_service(
            name=config.service_name,
            namespace=config.namespace,
            body=service
        )
        
        # Cleanup green deployment
        await self._cleanup_green_deployment(green_deployment, config.namespace)
    
    async def _rollback_canary(self, config: DeploymentConfig, canary_deployment: str):
        """Rollback canary deployment"""
        logger.info(f"Rolling back canary deployment: {canary_deployment}")
        
        # Remove canary traffic
        await self._update_traffic_split(config, canary_deployment, 0)
        
        # Cleanup canary deployment
        await self._cleanup_canary_deployment(canary_deployment, config.namespace)
    
    async def _cleanup_green_deployment(self, green_deployment: str, namespace: str):
        """Cleanup green deployment"""
        apps_v1 = client.AppsV1Api(self.k8s_client)
        
        try:
            apps_v1.delete_namespaced_deployment(
                name=green_deployment,
                namespace=namespace
            )
            logger.info(f"Cleaned up green deployment: {green_deployment}")
        except Exception as e:
            logger.error(f"Failed to cleanup green deployment: {str(e)}")
    
    async def _cleanup_blue_deployment(self, config: DeploymentConfig):
        """Cleanup old blue deployment"""
        blue_deployment = f"{config.service_name}-blue"
        apps_v1 = client.AppsV1Api(self.k8s_client)
        
        try:
            apps_v1.delete_namespaced_deployment(
                name=blue_deployment,
                namespace=config.namespace
            )
            logger.info(f"Cleaned up blue deployment: {blue_deployment}")
        except Exception as e:
            logger.error(f"Failed to cleanup blue deployment: {str(e)}")
    
    async def _cleanup_canary_deployment(self, canary_deployment: str, namespace: str):
        """Cleanup canary deployment"""
        apps_v1 = client.AppsV1Api(self.k8s_client)
        
        try:
            apps_v1.delete_namespaced_deployment(
                name=canary_deployment,
                namespace=namespace
            )
            logger.info(f"Cleaned up canary deployment: {canary_deployment}")
        except Exception as e:
            logger.error(f"Failed to cleanup canary deployment: {str(e)}")
    
    async def _complete_canary_deployment(self, config: DeploymentConfig, canary_deployment: str):
        """Complete canary deployment by promoting to stable"""
        logger.info(f"Promoting canary deployment to stable: {canary_deployment}")
        
        # Update stable deployment with canary image
        apps_v1 = client.AppsV1Api(self.k8s_client)
        
        stable_deployment = apps_v1.read_namespaced_deployment(
            name=config.service_name,
            namespace=config.namespace
        )
        
        # Update image
        stable_deployment.spec.template.spec.containers[0].image = config.image
        
        apps_v1.patch_namespaced_deployment(
            name=config.service_name,
            namespace=config.namespace,
            body=stable_deployment
        )
        
        # Cleanup canary deployment
        await self._cleanup_canary_deployment(canary_deployment, config.namespace)
    
    async def get_deployment_status(self, deployment_id: str) -> Optional[DeploymentResult]:
        """Get status of active deployment"""
        return self.active_deployments.get(deployment_id)
    
    async def list_active_deployments(self) -> List[DeploymentResult]:
        """List all active deployments"""
        return list(self.active_deployments.values())