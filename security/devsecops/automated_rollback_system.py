"""
Automated Rollback System Triggered by Security Incidents
Implements intelligent rollback capabilities for DevSecOps pipeline
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
from dataclasses import dataclass, field
import json
import time
import kubernetes
from kubernetes import client, config

logger = logging.getLogger(__name__)

class IncidentSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class RollbackTrigger(Enum):
    SECURITY_INCIDENT = "security_incident"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    ERROR_RATE_SPIKE = "error_rate_spike"
    COMPLIANCE_VIOLATION = "compliance_violation"
    MANUAL_TRIGGER = "manual_trigger"
    HEALTH_CHECK_FAILURE = "health_check_failure"

class RollbackStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class RollbackStrategy(Enum):
    IMMEDIATE = "immediate"
    GRADUAL = "gradual"
    BLUE_GREEN_SWITCH = "blue_green_switch"
    CANARY_ROLLBACK = "canary_rollback"

@dataclass
class SecurityIncident:
    incident_id: str
    title: str
    description: str
    severity: IncidentSeverity
    affected_services: List[str]
    affected_environments: List[str]
    detection_time: datetime
    indicators: List[Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RollbackConfiguration:
    service_name: str
    environment: str
    strategy: RollbackStrategy
    target_version: Optional[str]
    rollback_triggers: List[RollbackTrigger]
    thresholds: Dict[str, Any]
    approval_required: bool
    auto_rollback_enabled: bool
    rollback_timeout: int  # seconds
    health_check_config: Dict[str, Any]

@dataclass
class RollbackExecution:
    rollback_id: str
    service_name: str
    environment: str
    trigger: RollbackTrigger
    trigger_details: Dict[str, Any]
    strategy: RollbackStrategy
    status: RollbackStatus
    start_time: datetime
    end_time: Optional[datetime]
    current_version: str
    target_version: str
    steps_completed: List[str]
    error_message: Optional[str] = None
    approval_status: Optional[str] = None

class AutomatedRollbackSystem:
    """
    Automated rollback system triggered by security incidents and other conditions
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.k8s_client = self._initialize_k8s_client()
        self.rollback_configs = {}
        self.active_rollbacks = {}
        self.incident_handlers = self._initialize_incident_handlers()
        self.monitoring_tasks = []
        
        # Start monitoring tasks
        asyncio.create_task(self._start_monitoring())
    
    def _initialize_k8s_client(self):
        """Initialize Kubernetes client"""
        try:
            config.load_incluster_config()
        except config.ConfigException:
            config.load_kube_config()
        
        return client.ApiClient()
    
    def _initialize_incident_handlers(self) -> Dict[str, Callable]:
        """Initialize incident detection handlers"""
        return {
            "security_scanner": self._handle_security_incident,
            "performance_monitor": self._handle_performance_incident,
            "error_rate_monitor": self._handle_error_rate_incident,
            "compliance_monitor": self._handle_compliance_incident,
            "health_check_monitor": self._handle_health_check_incident
        }
    
    async def register_service(
        self, 
        service_name: str,
        environment: str,
        rollback_config: Dict[str, Any]
    ) -> RollbackConfiguration:
        """Register service for automated rollback monitoring"""
        config_key = f"{service_name}:{environment}"
        
        rollback_configuration = RollbackConfiguration(
            service_name=service_name,
            environment=environment,
            strategy=RollbackStrategy(rollback_config.get("strategy", "immediate")),
            target_version=rollback_config.get("target_version"),
            rollback_triggers=[
                RollbackTrigger(trigger) for trigger in rollback_config.get("triggers", [])
            ],
            thresholds=rollback_config.get("thresholds", {}),
            approval_required=rollback_config.get("approval_required", False),
            auto_rollback_enabled=rollback_config.get("auto_rollback_enabled", True),
            rollback_timeout=rollback_config.get("rollback_timeout", 300),
            health_check_config=rollback_config.get("health_check_config", {})
        )
        
        self.rollback_configs[config_key] = rollback_configuration
        
        logger.info(f"Registered service for rollback monitoring: {config_key}")
        return rollback_configuration
    
    async def _start_monitoring(self):
        """Start monitoring tasks for all registered services"""
        while True:
            try:
                # Monitor for security incidents
                await self._monitor_security_incidents()
                
                # Monitor performance metrics
                await self._monitor_performance_metrics()
                
                # Monitor error rates
                await self._monitor_error_rates()
                
                # Monitor compliance violations
                await self._monitor_compliance_violations()
                
                # Monitor health checks
                await self._monitor_health_checks()
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {str(e)}")
                await asyncio.sleep(60)  # Wait longer on error
    
    async def _monitor_security_incidents(self):
        """Monitor for security incidents that should trigger rollbacks"""
        # Simulate security incident detection
        # In a real implementation, this would integrate with SIEM, security scanners, etc.
        
        for config_key, rollback_config in self.rollback_configs.items():
            if RollbackTrigger.SECURITY_INCIDENT not in rollback_config.rollback_triggers:
                continue
            
            # Check for security incidents
            incidents = await self._detect_security_incidents(rollback_config)
            
            for incident in incidents:
                if self._should_trigger_rollback(incident, rollback_config):
                    await self._trigger_rollback(
                        rollback_config,
                        RollbackTrigger.SECURITY_INCIDENT,
                        {"incident": incident}
                    )
    
    async def _detect_security_incidents(
        self, 
        rollback_config: RollbackConfiguration
    ) -> List[SecurityIncident]:
        """Detect security incidents for a service"""
        incidents = []
        
        # Simulate security incident detection
        # This would integrate with actual security monitoring tools
        
        # Example: Check for suspicious activity patterns
        suspicious_activity = await self._check_suspicious_activity(rollback_config)
        if suspicious_activity:
            incident = SecurityIncident(
                incident_id=f"sec-{int(time.time())}",
                title="Suspicious Activity Detected",
                description="Unusual access patterns detected",
                severity=IncidentSeverity.HIGH,
                affected_services=[rollback_config.service_name],
                affected_environments=[rollback_config.environment],
                detection_time=datetime.now(),
                indicators=suspicious_activity
            )
            incidents.append(incident)
        
        return incidents
    
    async def _check_suspicious_activity(
        self, 
        rollback_config: RollbackConfiguration
    ) -> List[Dict[str, Any]]:
        """Check for suspicious activity indicators"""
        # Simulate suspicious activity detection
        # This would check logs, metrics, security events, etc.
        
        indicators = []
        
        # Example indicators
        if rollback_config.service_name == "payment-service":
            # Simulate detection of unusual payment patterns
            indicators.append({
                "type": "unusual_access_pattern",
                "description": "High number of failed authentication attempts",
                "count": 150,
                "threshold": 50,
                "time_window": "5 minutes"
            })
        
        return indicators
    
    async def _monitor_performance_metrics(self):
        """Monitor performance metrics for rollback triggers"""
        for config_key, rollback_config in self.rollback_configs.items():
            if RollbackTrigger.PERFORMANCE_DEGRADATION not in rollback_config.rollback_triggers:
                continue
            
            # Check performance metrics
            performance_issue = await self._check_performance_degradation(rollback_config)
            
            if performance_issue:
                await self._trigger_rollback(
                    rollback_config,
                    RollbackTrigger.PERFORMANCE_DEGRADATION,
                    {"performance_issue": performance_issue}
                )
    
    async def _check_performance_degradation(
        self, 
        rollback_config: RollbackConfiguration
    ) -> Optional[Dict[str, Any]]:
        """Check for performance degradation"""
        thresholds = rollback_config.thresholds
        
        # Simulate performance metrics collection
        current_metrics = {
            "response_time_p95": 250,  # ms
            "cpu_usage": 85,  # %
            "memory_usage": 90,  # %
            "request_rate": 500  # requests/minute
        }
        
        # Check thresholds
        if current_metrics["response_time_p95"] > thresholds.get("max_response_time", 200):
            return {
                "type": "high_response_time",
                "current_value": current_metrics["response_time_p95"],
                "threshold": thresholds.get("max_response_time", 200),
                "metrics": current_metrics
            }
        
        if current_metrics["cpu_usage"] > thresholds.get("max_cpu_usage", 80):
            return {
                "type": "high_cpu_usage",
                "current_value": current_metrics["cpu_usage"],
                "threshold": thresholds.get("max_cpu_usage", 80),
                "metrics": current_metrics
            }
        
        return None
    
    async def _monitor_error_rates(self):
        """Monitor error rates for rollback triggers"""
        for config_key, rollback_config in self.rollback_configs.items():
            if RollbackTrigger.ERROR_RATE_SPIKE not in rollback_config.rollback_triggers:
                continue
            
            # Check error rates
            error_spike = await self._check_error_rate_spike(rollback_config)
            
            if error_spike:
                await self._trigger_rollback(
                    rollback_config,
                    RollbackTrigger.ERROR_RATE_SPIKE,
                    {"error_spike": error_spike}
                )
    
    async def _check_error_rate_spike(
        self, 
        rollback_config: RollbackConfiguration
    ) -> Optional[Dict[str, Any]]:
        """Check for error rate spikes"""
        thresholds = rollback_config.thresholds
        
        # Simulate error rate collection
        current_error_rate = 8.5  # %
        baseline_error_rate = 2.0  # %
        
        max_error_rate = thresholds.get("max_error_rate", 5.0)
        
        if current_error_rate > max_error_rate:
            return {
                "type": "error_rate_spike",
                "current_error_rate": current_error_rate,
                "baseline_error_rate": baseline_error_rate,
                "threshold": max_error_rate,
                "spike_factor": current_error_rate / baseline_error_rate
            }
        
        return None
    
    async def _monitor_compliance_violations(self):
        """Monitor for compliance violations"""
        for config_key, rollback_config in self.rollback_configs.items():
            if RollbackTrigger.COMPLIANCE_VIOLATION not in rollback_config.rollback_triggers:
                continue
            
            # Check compliance violations
            violations = await self._check_compliance_violations(rollback_config)
            
            if violations:
                await self._trigger_rollback(
                    rollback_config,
                    RollbackTrigger.COMPLIANCE_VIOLATION,
                    {"violations": violations}
                )
    
    async def _check_compliance_violations(
        self, 
        rollback_config: RollbackConfiguration
    ) -> List[Dict[str, Any]]:
        """Check for compliance violations"""
        violations = []
        
        # Simulate compliance checking
        # This would integrate with compliance monitoring tools
        
        # Example: Check for data exposure
        if rollback_config.service_name == "user-service":
            violations.append({
                "type": "data_exposure",
                "description": "Potential PII exposure detected in logs",
                "severity": "high",
                "compliance_framework": "GDPR"
            })
        
        return violations
    
    async def _monitor_health_checks(self):
        """Monitor health check failures"""
        for config_key, rollback_config in self.rollback_configs.items():
            if RollbackTrigger.HEALTH_CHECK_FAILURE not in rollback_config.rollback_triggers:
                continue
            
            # Check health status
            health_failure = await self._check_health_failures(rollback_config)
            
            if health_failure:
                await self._trigger_rollback(
                    rollback_config,
                    RollbackTrigger.HEALTH_CHECK_FAILURE,
                    {"health_failure": health_failure}
                )
    
    async def _check_health_failures(
        self, 
        rollback_config: RollbackConfiguration
    ) -> Optional[Dict[str, Any]]:
        """Check for health check failures"""
        health_config = rollback_config.health_check_config
        
        # Simulate health check
        health_status = await self._perform_health_check(rollback_config)
        
        failure_threshold = health_config.get("failure_threshold", 3)
        
        if health_status["consecutive_failures"] >= failure_threshold:
            return {
                "type": "health_check_failure",
                "consecutive_failures": health_status["consecutive_failures"],
                "threshold": failure_threshold,
                "last_success": health_status["last_success"],
                "failure_details": health_status["failure_details"]
            }
        
        return None
    
    async def _perform_health_check(
        self, 
        rollback_config: RollbackConfiguration
    ) -> Dict[str, Any]:
        """Perform health check for service"""
        # Simulate health check
        # This would make actual HTTP requests to health endpoints
        
        return {
            "consecutive_failures": 0,
            "last_success": datetime.now(),
            "failure_details": []
        }
    
    def _should_trigger_rollback(
        self, 
        incident: SecurityIncident,
        rollback_config: RollbackConfiguration
    ) -> bool:
        """Determine if incident should trigger rollback"""
        # Check severity threshold
        severity_threshold = rollback_config.thresholds.get("min_incident_severity", "medium")
        severity_levels = {"low": 1, "medium": 2, "high": 3, "critical": 4}
        
        if severity_levels[incident.severity.value] < severity_levels[severity_threshold]:
            return False
        
        # Check if service is affected
        if rollback_config.service_name not in incident.affected_services:
            return False
        
        # Check if environment is affected
        if rollback_config.environment not in incident.affected_environments:
            return False
        
        return True
    
    async def _trigger_rollback(
        self, 
        rollback_config: RollbackConfiguration,
        trigger: RollbackTrigger,
        trigger_details: Dict[str, Any]
    ) -> RollbackExecution:
        """Trigger automated rollback"""
        rollback_id = f"rollback-{int(time.time())}-{rollback_config.service_name}"
        
        # Check if rollback is already in progress
        existing_rollback = self._get_active_rollback(
            rollback_config.service_name, rollback_config.environment
        )
        
        if existing_rollback:
            logger.warning(f"Rollback already in progress for {rollback_config.service_name}")
            return existing_rollback
        
        # Get current version
        current_version = await self._get_current_version(rollback_config)
        
        # Determine target version
        target_version = await self._determine_target_version(rollback_config, current_version)
        
        rollback_execution = RollbackExecution(
            rollback_id=rollback_id,
            service_name=rollback_config.service_name,
            environment=rollback_config.environment,
            trigger=trigger,
            trigger_details=trigger_details,
            strategy=rollback_config.strategy,
            status=RollbackStatus.PENDING,
            start_time=datetime.now(),
            end_time=None,
            current_version=current_version,
            target_version=target_version,
            steps_completed=[]
        )
        
        self.active_rollbacks[rollback_id] = rollback_execution
        
        logger.critical(f"Triggering rollback {rollback_id} for {rollback_config.service_name} due to {trigger.value}")
        
        # Check if approval is required
        if rollback_config.approval_required:
            rollback_execution.status = RollbackStatus.PENDING
            await self._request_rollback_approval(rollback_execution)
        else:
            # Execute rollback immediately
            asyncio.create_task(self._execute_rollback(rollback_execution, rollback_config))
        
        return rollback_execution
    
    def _get_active_rollback(self, service_name: str, environment: str) -> Optional[RollbackExecution]:
        """Get active rollback for service in environment"""
        for rollback in self.active_rollbacks.values():
            if (rollback.service_name == service_name and 
                rollback.environment == environment and
                rollback.status in [RollbackStatus.PENDING, RollbackStatus.IN_PROGRESS]):
                return rollback
        return None
    
    async def _get_current_version(self, rollback_config: RollbackConfiguration) -> str:
        """Get current deployed version"""
        # Get current deployment version from Kubernetes
        apps_v1 = client.AppsV1Api(self.k8s_client)
        
        try:
            deployment = apps_v1.read_namespaced_deployment(
                name=rollback_config.service_name,
                namespace=rollback_config.environment
            )
            
            # Extract version from image tag
            image = deployment.spec.template.spec.containers[0].image
            if ":" in image:
                return image.split(":")[-1]
            else:
                return "latest"
                
        except Exception as e:
            logger.error(f"Error getting current version: {str(e)}")
            return "unknown"
    
    async def _determine_target_version(
        self, 
        rollback_config: RollbackConfiguration,
        current_version: str
    ) -> str:
        """Determine target version for rollback"""
        if rollback_config.target_version:
            return rollback_config.target_version
        
        # Get previous stable version
        # This would typically query a deployment history or artifact registry
        return await self._get_previous_stable_version(rollback_config, current_version)
    
    async def _get_previous_stable_version(
        self, 
        rollback_config: RollbackConfiguration,
        current_version: str
    ) -> str:
        """Get previous stable version"""
        # Simulate getting previous version
        # This would query deployment history, Git tags, or artifact registry
        
        # For demo purposes, assume version format like v1.2.3
        if current_version.startswith("v"):
            try:
                parts = current_version[1:].split(".")
                if len(parts) >= 3:
                    patch = int(parts[2])
                    if patch > 0:
                        return f"v{parts[0]}.{parts[1]}.{patch - 1}"
            except ValueError:
                pass
        
        return "previous-stable"
    
    async def _request_rollback_approval(self, rollback_execution: RollbackExecution):
        """Request approval for rollback"""
        logger.info(f"Requesting approval for rollback {rollback_execution.rollback_id}")
        
        # In a real implementation, this would send notifications to approvers
        # For now, simulate approval after a delay
        await asyncio.sleep(5)
        
        # Auto-approve for demo (in real implementation, this would wait for human approval)
        rollback_execution.approval_status = "auto-approved"
        
        # Start rollback execution
        rollback_config = self.rollback_configs[f"{rollback_execution.service_name}:{rollback_execution.environment}"]
        asyncio.create_task(self._execute_rollback(rollback_execution, rollback_config))
    
    async def _execute_rollback(
        self, 
        rollback_execution: RollbackExecution,
        rollback_config: RollbackConfiguration
    ):
        """Execute the rollback process"""
        rollback_execution.status = RollbackStatus.IN_PROGRESS
        
        logger.info(f"Executing rollback {rollback_execution.rollback_id} using {rollback_execution.strategy.value} strategy")
        
        try:
            if rollback_execution.strategy == RollbackStrategy.IMMEDIATE:
                await self._execute_immediate_rollback(rollback_execution, rollback_config)
            elif rollback_execution.strategy == RollbackStrategy.GRADUAL:
                await self._execute_gradual_rollback(rollback_execution, rollback_config)
            elif rollback_execution.strategy == RollbackStrategy.BLUE_GREEN_SWITCH:
                await self._execute_blue_green_rollback(rollback_execution, rollback_config)
            elif rollback_execution.strategy == RollbackStrategy.CANARY_ROLLBACK:
                await self._execute_canary_rollback(rollback_execution, rollback_config)
            
            rollback_execution.status = RollbackStatus.COMPLETED
            rollback_execution.end_time = datetime.now()
            
            logger.info(f"Rollback {rollback_execution.rollback_id} completed successfully")
            
        except Exception as e:
            rollback_execution.status = RollbackStatus.FAILED
            rollback_execution.error_message = str(e)
            rollback_execution.end_time = datetime.now()
            
            logger.error(f"Rollback {rollback_execution.rollback_id} failed: {str(e)}")
    
    async def _execute_immediate_rollback(
        self, 
        rollback_execution: RollbackExecution,
        rollback_config: RollbackConfiguration
    ):
        """Execute immediate rollback strategy"""
        steps = [
            "Updating deployment image",
            "Waiting for rollout",
            "Verifying health checks",
            "Updating service routing"
        ]
        
        for step in steps:
            logger.info(f"Rollback {rollback_execution.rollback_id}: {step}")
            
            # Simulate step execution
            await asyncio.sleep(2)
            
            rollback_execution.steps_completed.append(step)
        
        # Update Kubernetes deployment
        await self._update_deployment_image(
            rollback_config.service_name,
            rollback_config.environment,
            rollback_execution.target_version
        )
    
    async def _execute_gradual_rollback(
        self, 
        rollback_execution: RollbackExecution,
        rollback_config: RollbackConfiguration
    ):
        """Execute gradual rollback strategy"""
        steps = [
            "Creating rollback deployment",
            "Gradually shifting traffic (25%)",
            "Gradually shifting traffic (50%)",
            "Gradually shifting traffic (75%)",
            "Completing traffic shift (100%)",
            "Cleaning up old deployment"
        ]
        
        for step in steps:
            logger.info(f"Rollback {rollback_execution.rollback_id}: {step}")
            
            # Simulate step execution with longer delays for gradual rollback
            await asyncio.sleep(5)
            
            rollback_execution.steps_completed.append(step)
    
    async def _execute_blue_green_rollback(
        self, 
        rollback_execution: RollbackExecution,
        rollback_config: RollbackConfiguration
    ):
        """Execute blue-green rollback strategy"""
        steps = [
            "Identifying green environment",
            "Switching traffic to blue environment",
            "Verifying blue environment health",
            "Cleaning up green environment"
        ]
        
        for step in steps:
            logger.info(f"Rollback {rollback_execution.rollback_id}: {step}")
            
            await asyncio.sleep(3)
            rollback_execution.steps_completed.append(step)
    
    async def _execute_canary_rollback(
        self, 
        rollback_execution: RollbackExecution,
        rollback_config: RollbackConfiguration
    ):
        """Execute canary rollback strategy"""
        steps = [
            "Reducing canary traffic to 0%",
            "Verifying stable version health",
            "Removing canary deployment",
            "Updating routing configuration"
        ]
        
        for step in steps:
            logger.info(f"Rollback {rollback_execution.rollback_id}: {step}")
            
            await asyncio.sleep(2)
            rollback_execution.steps_completed.append(step)
    
    async def _update_deployment_image(
        self, 
        service_name: str,
        environment: str,
        target_version: str
    ):
        """Update Kubernetes deployment image"""
        apps_v1 = client.AppsV1Api(self.k8s_client)
        
        try:
            # Get current deployment
            deployment = apps_v1.read_namespaced_deployment(
                name=service_name,
                namespace=environment
            )
            
            # Update image tag
            current_image = deployment.spec.template.spec.containers[0].image
            if ":" in current_image:
                base_image = current_image.split(":")[0]
                new_image = f"{base_image}:{target_version}"
            else:
                new_image = f"{current_image}:{target_version}"
            
            deployment.spec.template.spec.containers[0].image = new_image
            
            # Apply update
            apps_v1.patch_namespaced_deployment(
                name=service_name,
                namespace=environment,
                body=deployment
            )
            
            logger.info(f"Updated deployment {service_name} to version {target_version}")
            
        except Exception as e:
            logger.error(f"Failed to update deployment image: {str(e)}")
            raise
    
    async def manual_rollback(
        self, 
        service_name: str,
        environment: str,
        target_version: str,
        reason: str,
        initiated_by: str
    ) -> RollbackExecution:
        """Manually trigger rollback"""
        config_key = f"{service_name}:{environment}"
        
        if config_key not in self.rollback_configs:
            raise ValueError(f"Service {service_name} not registered for rollback")
        
        rollback_config = self.rollback_configs[config_key]
        
        # Override target version
        rollback_config.target_version = target_version
        
        trigger_details = {
            "reason": reason,
            "initiated_by": initiated_by,
            "manual_trigger": True
        }
        
        return await self._trigger_rollback(
            rollback_config,
            RollbackTrigger.MANUAL_TRIGGER,
            trigger_details
        )
    
    async def cancel_rollback(self, rollback_id: str, cancelled_by: str) -> bool:
        """Cancel ongoing rollback"""
        if rollback_id not in self.active_rollbacks:
            return False
        
        rollback_execution = self.active_rollbacks[rollback_id]
        
        if rollback_execution.status not in [RollbackStatus.PENDING, RollbackStatus.IN_PROGRESS]:
            return False
        
        rollback_execution.status = RollbackStatus.CANCELLED
        rollback_execution.end_time = datetime.now()
        rollback_execution.error_message = f"Cancelled by {cancelled_by}"
        
        logger.info(f"Rollback {rollback_id} cancelled by {cancelled_by}")
        return True
    
    def get_rollback_status(self, rollback_id: str) -> Optional[RollbackExecution]:
        """Get rollback execution status"""
        return self.active_rollbacks.get(rollback_id)
    
    def list_active_rollbacks(self) -> List[RollbackExecution]:
        """List all active rollbacks"""
        return [
            rollback for rollback in self.active_rollbacks.values()
            if rollback.status in [RollbackStatus.PENDING, RollbackStatus.IN_PROGRESS]
        ]
    
    def get_rollback_history(
        self, 
        service_name: Optional[str] = None,
        environment: Optional[str] = None,
        limit: int = 50
    ) -> List[RollbackExecution]:
        """Get rollback history with optional filters"""
        rollbacks = list(self.active_rollbacks.values())
        
        if service_name:
            rollbacks = [r for r in rollbacks if r.service_name == service_name]
        
        if environment:
            rollbacks = [r for r in rollbacks if r.environment == environment]
        
        # Sort by start time (most recent first)
        rollbacks.sort(key=lambda r: r.start_time, reverse=True)
        
        return rollbacks[:limit]
    
    async def generate_rollback_report(self, rollback_id: str) -> Dict[str, Any]:
        """Generate comprehensive rollback report"""
        if rollback_id not in self.active_rollbacks:
            raise ValueError(f"Rollback {rollback_id} not found")
        
        rollback = self.active_rollbacks[rollback_id]
        
        return {
            "rollback_summary": {
                "rollback_id": rollback.rollback_id,
                "service_name": rollback.service_name,
                "environment": rollback.environment,
                "trigger": rollback.trigger.value,
                "strategy": rollback.strategy.value,
                "status": rollback.status.value,
                "start_time": rollback.start_time.isoformat(),
                "end_time": rollback.end_time.isoformat() if rollback.end_time else None,
                "duration": (rollback.end_time - rollback.start_time).total_seconds() if rollback.end_time else None
            },
            "version_info": {
                "current_version": rollback.current_version,
                "target_version": rollback.target_version
            },
            "trigger_details": rollback.trigger_details,
            "execution_details": {
                "steps_completed": rollback.steps_completed,
                "error_message": rollback.error_message,
                "approval_status": rollback.approval_status
            },
            "impact_assessment": await self._assess_rollback_impact(rollback)
        }
    
    async def _assess_rollback_impact(self, rollback: RollbackExecution) -> Dict[str, Any]:
        """Assess the impact of rollback"""
        # Simulate impact assessment
        return {
            "service_availability": "maintained",
            "data_consistency": "preserved",
            "user_impact": "minimal",
            "downstream_services": "not affected",
            "estimated_recovery_time": "5 minutes"
        }