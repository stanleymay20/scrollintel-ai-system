"""
Configuration Drift Detection with 60-Second Auto-Remediation
Monitors infrastructure configuration changes and automatically remediates drift
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import json
import hashlib
import difflib

logger = logging.getLogger(__name__)

class DriftSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class RemediationStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    MANUAL_REQUIRED = "manual_required"

class ResourceType(Enum):
    KUBERNETES_DEPLOYMENT = "kubernetes_deployment"
    KUBERNETES_SERVICE = "kubernetes_service"
    KUBERNETES_CONFIGMAP = "kubernetes_configmap"
    KUBERNETES_SECRET = "kubernetes_secret"
    DOCKER_CONTAINER = "docker_container"
    NGINX_CONFIG = "nginx_config"
    DATABASE_CONFIG = "database_config"
    SECURITY_POLICY = "security_policy"
    NETWORK_POLICY = "network_policy"
    FIREWALL_RULE = "firewall_rule"

@dataclass
class ConfigurationBaseline:
    """Baseline configuration for a resource"""
    resource_id: str
    resource_type: ResourceType
    configuration: Dict[str, Any]
    checksum: str
    created_at: datetime
    last_verified: datetime
    tags: Dict[str, str] = field(default_factory=dict)

@dataclass
class DriftDetectionResult:
    """Result of configuration drift detection"""
    resource_id: str
    resource_type: ResourceType
    drift_detected: bool
    drift_severity: DriftSeverity
    expected_config: Dict[str, Any]
    actual_config: Dict[str, Any]
    configuration_diff: List[str]
    detection_time: datetime
    auto_remediation_available: bool
    remediation_actions: List[str]

@dataclass
class RemediationAction:
    """Configuration remediation action"""
    action_id: str
    resource_id: str
    resource_type: ResourceType
    action_type: str
    parameters: Dict[str, Any]
    status: RemediationStatus
    created_at: datetime
    completed_at: Optional[datetime]
    error_message: Optional[str]
    rollback_available: bool

class ConfigurationDriftDetection:
    """
    Configuration Drift Detection System
    Monitors infrastructure configuration changes and provides
    60-second auto-remediation capabilities
    """
    
    def __init__(self):
        self.baselines: Dict[str, ConfigurationBaseline] = {}
        self.drift_history: List[DriftDetectionResult] = []
        self.remediation_queue: List[RemediationAction] = []
        self.remediation_history: List[RemediationAction] = []
        
        # Detection settings
        self.detection_interval = 60  # 60-second detection interval
        self.auto_remediation_enabled = True
        self.remediation_timeout = 60  # 60-second remediation timeout
        
        # Severity thresholds
        self.severity_thresholds = {
            'critical_keys': ['security', 'authentication', 'authorization', 'encryption'],
            'high_keys': ['network', 'firewall', 'access_control', 'backup'],
            'medium_keys': ['monitoring', 'logging', 'alerting', 'performance'],
            'low_keys': ['ui', 'display', 'formatting', 'comments']
        }
        
        # Auto-remediation rules
        self.auto_remediation_rules = {
            ResourceType.KUBERNETES_DEPLOYMENT: ['restart_pods', 'update_config', 'rollback_version'],
            ResourceType.KUBERNETES_SERVICE: ['update_service', 'recreate_service'],
            ResourceType.KUBERNETES_CONFIGMAP: ['update_configmap', 'recreate_configmap'],
            ResourceType.NGINX_CONFIG: ['reload_nginx', 'restore_config'],
            ResourceType.DATABASE_CONFIG: ['restart_service', 'restore_config'],
            ResourceType.SECURITY_POLICY: ['restore_policy', 'alert_security_team'],
            ResourceType.NETWORK_POLICY: ['restore_policy', 'isolate_resource'],
            ResourceType.FIREWALL_RULE: ['restore_rules', 'emergency_lockdown']
        }
        
        self._monitoring_active = False
    
    async def start_drift_monitoring(self) -> Dict[str, Any]:
        """Start configuration drift monitoring"""
        try:
            self._monitoring_active = True
            
            # Start monitoring tasks
            monitoring_tasks = [
                self._continuous_drift_detection(),
                self._auto_remediation_processor(),
                self._baseline_maintenance()
            ]
            
            await asyncio.gather(*monitoring_tasks)
            
            return {
                'status': 'success',
                'message': 'Configuration drift monitoring started',
                'detection_interval': self.detection_interval,
                'auto_remediation_enabled': self.auto_remediation_enabled
            }
            
        except Exception as e:
            logger.error(f"Failed to start drift monitoring: {str(e)}")
            return {
                'status': 'error',
                'message': f'Failed to start monitoring: {str(e)}'
            }
    
    async def register_configuration_baseline(self, resource_id: str, resource_type: ResourceType, 
                                            configuration: Dict[str, Any], tags: Dict[str, str] = None) -> bool:
        """Register a configuration baseline for monitoring"""
        try:
            # Calculate configuration checksum
            config_json = json.dumps(configuration, sort_keys=True)
            checksum = hashlib.sha256(config_json.encode()).hexdigest()
            
            baseline = ConfigurationBaseline(
                resource_id=resource_id,
                resource_type=resource_type,
                configuration=configuration,
                checksum=checksum,
                created_at=datetime.now(),
                last_verified=datetime.now(),
                tags=tags or {}
            )
            
            self.baselines[resource_id] = baseline
            
            logger.info(f"Registered configuration baseline for {resource_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register baseline for {resource_id}: {str(e)}")
            return False
    
    async def _continuous_drift_detection(self):
        """Continuous configuration drift detection"""
        while self._monitoring_active:
            try:
                # Check all registered baselines
                for resource_id, baseline in self.baselines.items():
                    drift_result = await self._detect_configuration_drift(resource_id, baseline)
                    
                    if drift_result.drift_detected:
                        logger.warning(f"Configuration drift detected for {resource_id}: {drift_result.drift_severity.value}")
                        
                        # Store drift result
                        self.drift_history.append(drift_result)
                        
                        # Trigger auto-remediation if enabled and available
                        if (self.auto_remediation_enabled and 
                            drift_result.auto_remediation_available and
                            drift_result.drift_severity in [DriftSeverity.HIGH, DriftSeverity.CRITICAL]):
                            
                            await self._queue_auto_remediation(drift_result)
                
                # Clean up old drift history
                cutoff_time = datetime.now() - timedelta(days=7)
                self.drift_history = [d for d in self.drift_history if d.detection_time > cutoff_time]
                
                await asyncio.sleep(self.detection_interval)
                
            except Exception as e:
                logger.error(f"Drift detection error: {str(e)}")
                await asyncio.sleep(self.detection_interval)
    
    async def _detect_configuration_drift(self, resource_id: str, baseline: ConfigurationBaseline) -> DriftDetectionResult:
        """Detect configuration drift for a specific resource"""
        try:
            # Get current configuration
            current_config = await self._get_current_configuration(resource_id, baseline.resource_type)
            
            # Calculate current checksum
            current_config_json = json.dumps(current_config, sort_keys=True)
            current_checksum = hashlib.sha256(current_config_json.encode()).hexdigest()
            
            # Check for drift
            drift_detected = current_checksum != baseline.checksum
            
            if drift_detected:
                # Calculate configuration diff
                config_diff = self._calculate_configuration_diff(baseline.configuration, current_config)
                
                # Determine drift severity
                drift_severity = self._calculate_drift_severity(config_diff, baseline.resource_type)
                
                # Check if auto-remediation is available
                auto_remediation_available = self._is_auto_remediation_available(baseline.resource_type, drift_severity)
                
                # Generate remediation actions
                remediation_actions = self._generate_remediation_actions(baseline.resource_type, config_diff)
                
                return DriftDetectionResult(
                    resource_id=resource_id,
                    resource_type=baseline.resource_type,
                    drift_detected=True,
                    drift_severity=drift_severity,
                    expected_config=baseline.configuration,
                    actual_config=current_config,
                    configuration_diff=config_diff,
                    detection_time=datetime.now(),
                    auto_remediation_available=auto_remediation_available,
                    remediation_actions=remediation_actions
                )
            else:
                # Update last verified time
                baseline.last_verified = datetime.now()
                
                return DriftDetectionResult(
                    resource_id=resource_id,
                    resource_type=baseline.resource_type,
                    drift_detected=False,
                    drift_severity=DriftSeverity.LOW,
                    expected_config=baseline.configuration,
                    actual_config=current_config,
                    configuration_diff=[],
                    detection_time=datetime.now(),
                    auto_remediation_available=False,
                    remediation_actions=[]
                )
                
        except Exception as e:
            logger.error(f"Failed to detect drift for {resource_id}: {str(e)}")
            raise
    
    async def _get_current_configuration(self, resource_id: str, resource_type: ResourceType) -> Dict[str, Any]:
        """Get current configuration for a resource"""
        # This would integrate with actual infrastructure APIs
        # For now, simulate configuration retrieval
        
        if resource_type == ResourceType.KUBERNETES_DEPLOYMENT:
            return await self._get_kubernetes_deployment_config(resource_id)
        elif resource_type == ResourceType.KUBERNETES_SERVICE:
            return await self._get_kubernetes_service_config(resource_id)
        elif resource_type == ResourceType.NGINX_CONFIG:
            return await self._get_nginx_config(resource_id)
        elif resource_type == ResourceType.DATABASE_CONFIG:
            return await self._get_database_config(resource_id)
        elif resource_type == ResourceType.SECURITY_POLICY:
            return await self._get_security_policy_config(resource_id)
        else:
            return {}
    
    async def _get_kubernetes_deployment_config(self, resource_id: str) -> Dict[str, Any]:
        """Get Kubernetes deployment configuration"""
        # Simulate getting K8s deployment config
        import random
        
        # Simulate occasional configuration drift
        if random.random() < 0.1:  # 10% chance of drift
            return {
                'replicas': random.choice([2, 3, 5]),  # Different from baseline
                'image': 'app:v1.2.0',  # Different version
                'resources': {
                    'requests': {'cpu': '100m', 'memory': '128Mi'},
                    'limits': {'cpu': '500m', 'memory': '512Mi'}
                },
                'env': [
                    {'name': 'ENV', 'value': 'production'},
                    {'name': 'DEBUG', 'value': 'false'}
                ]
            }
        else:
            # Return baseline configuration
            return {
                'replicas': 3,
                'image': 'app:v1.1.0',
                'resources': {
                    'requests': {'cpu': '100m', 'memory': '128Mi'},
                    'limits': {'cpu': '500m', 'memory': '512Mi'}
                },
                'env': [
                    {'name': 'ENV', 'value': 'production'},
                    {'name': 'DEBUG', 'value': 'false'}
                ]
            }
    
    async def _get_kubernetes_service_config(self, resource_id: str) -> Dict[str, Any]:
        """Get Kubernetes service configuration"""
        return {
            'type': 'ClusterIP',
            'ports': [{'port': 80, 'targetPort': 8080}],
            'selector': {'app': 'myapp'}
        }
    
    async def _get_nginx_config(self, resource_id: str) -> Dict[str, Any]:
        """Get NGINX configuration"""
        return {
            'server_name': 'example.com',
            'listen': 80,
            'root': '/var/www/html',
            'index': 'index.html'
        }
    
    async def _get_database_config(self, resource_id: str) -> Dict[str, Any]:
        """Get database configuration"""
        return {
            'max_connections': 100,
            'shared_buffers': '256MB',
            'effective_cache_size': '1GB'
        }
    
    async def _get_security_policy_config(self, resource_id: str) -> Dict[str, Any]:
        """Get security policy configuration"""
        return {
            'authentication_required': True,
            'encryption_enabled': True,
            'access_control': 'rbac'
        }
    
    def _calculate_configuration_diff(self, expected: Dict[str, Any], actual: Dict[str, Any]) -> List[str]:
        """Calculate configuration differences"""
        expected_json = json.dumps(expected, sort_keys=True, indent=2)
        actual_json = json.dumps(actual, sort_keys=True, indent=2)
        
        diff = list(difflib.unified_diff(
            expected_json.splitlines(keepends=True),
            actual_json.splitlines(keepends=True),
            fromfile='expected',
            tofile='actual'
        ))
        
        return diff
    
    def _calculate_drift_severity(self, config_diff: List[str], resource_type: ResourceType) -> DriftSeverity:
        """Calculate drift severity based on configuration changes"""
        if not config_diff:
            return DriftSeverity.LOW
        
        diff_text = '\n'.join(config_diff).lower()
        
        # Check for critical changes
        for critical_key in self.severity_thresholds['critical_keys']:
            if critical_key in diff_text:
                return DriftSeverity.CRITICAL
        
        # Check for high severity changes
        for high_key in self.severity_thresholds['high_keys']:
            if high_key in diff_text:
                return DriftSeverity.HIGH
        
        # Check for medium severity changes
        for medium_key in self.severity_thresholds['medium_keys']:
            if medium_key in diff_text:
                return DriftSeverity.MEDIUM
        
        return DriftSeverity.LOW
    
    def _is_auto_remediation_available(self, resource_type: ResourceType, severity: DriftSeverity) -> bool:
        """Check if auto-remediation is available for resource type and severity"""
        if not self.auto_remediation_enabled:
            return False
        
        # Auto-remediation available for known resource types
        if resource_type not in self.auto_remediation_rules:
            return False
        
        # Only auto-remediate high and critical severity drifts
        return severity in [DriftSeverity.HIGH, DriftSeverity.CRITICAL]
    
    def _generate_remediation_actions(self, resource_type: ResourceType, config_diff: List[str]) -> List[str]:
        """Generate remediation actions based on resource type and drift"""
        if resource_type not in self.auto_remediation_rules:
            return ['manual_review_required']
        
        return self.auto_remediation_rules[resource_type]
    
    async def _queue_auto_remediation(self, drift_result: DriftDetectionResult):
        """Queue auto-remediation action"""
        try:
            for action_type in drift_result.remediation_actions:
                action = RemediationAction(
                    action_id=f"{drift_result.resource_id}-{action_type}-{int(datetime.now().timestamp())}",
                    resource_id=drift_result.resource_id,
                    resource_type=drift_result.resource_type,
                    action_type=action_type,
                    parameters={
                        'expected_config': drift_result.expected_config,
                        'actual_config': drift_result.actual_config,
                        'severity': drift_result.drift_severity.value
                    },
                    status=RemediationStatus.PENDING,
                    created_at=datetime.now(),
                    completed_at=None,
                    error_message=None,
                    rollback_available=True
                )
                
                self.remediation_queue.append(action)
                logger.info(f"Queued auto-remediation action: {action.action_id}")
        
        except Exception as e:
            logger.error(f"Failed to queue auto-remediation: {str(e)}")
    
    async def _auto_remediation_processor(self):
        """Process auto-remediation queue"""
        while self._monitoring_active:
            try:
                if self.remediation_queue:
                    # Process pending remediation actions
                    pending_actions = [a for a in self.remediation_queue if a.status == RemediationStatus.PENDING]
                    
                    for action in pending_actions[:5]:  # Process up to 5 actions at once
                        await self._execute_remediation_action(action)
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Auto-remediation processor error: {str(e)}")
                await asyncio.sleep(30)
    
    async def _execute_remediation_action(self, action: RemediationAction):
        """Execute a remediation action"""
        try:
            logger.info(f"Executing remediation action: {action.action_id}")
            action.status = RemediationStatus.IN_PROGRESS
            
            # Set timeout for remediation
            try:
                await asyncio.wait_for(
                    self._perform_remediation(action),
                    timeout=self.remediation_timeout
                )
                
                action.status = RemediationStatus.COMPLETED
                action.completed_at = datetime.now()
                logger.info(f"Remediation action completed: {action.action_id}")
                
            except asyncio.TimeoutError:
                action.status = RemediationStatus.FAILED
                action.error_message = "Remediation timeout"
                logger.error(f"Remediation action timed out: {action.action_id}")
            
            # Move to history and remove from queue
            self.remediation_history.append(action)
            self.remediation_queue.remove(action)
            
        except Exception as e:
            action.status = RemediationStatus.FAILED
            action.error_message = str(e)
            logger.error(f"Remediation action failed: {action.action_id} - {str(e)}")
            
            # Move to history
            self.remediation_history.append(action)
            if action in self.remediation_queue:
                self.remediation_queue.remove(action)
    
    async def _perform_remediation(self, action: RemediationAction):
        """Perform the actual remediation"""
        action_type = action.action_type
        resource_type = action.resource_type
        
        if resource_type == ResourceType.KUBERNETES_DEPLOYMENT:
            await self._remediate_kubernetes_deployment(action)
        elif resource_type == ResourceType.KUBERNETES_SERVICE:
            await self._remediate_kubernetes_service(action)
        elif resource_type == ResourceType.NGINX_CONFIG:
            await self._remediate_nginx_config(action)
        elif resource_type == ResourceType.DATABASE_CONFIG:
            await self._remediate_database_config(action)
        elif resource_type == ResourceType.SECURITY_POLICY:
            await self._remediate_security_policy(action)
        else:
            raise ValueError(f"Unsupported resource type for remediation: {resource_type}")
    
    async def _remediate_kubernetes_deployment(self, action: RemediationAction):
        """Remediate Kubernetes deployment configuration"""
        if action.action_type == 'restart_pods':
            logger.info(f"Restarting pods for deployment: {action.resource_id}")
            await asyncio.sleep(2)  # Simulate pod restart
        elif action.action_type == 'update_config':
            logger.info(f"Updating configuration for deployment: {action.resource_id}")
            await asyncio.sleep(3)  # Simulate config update
        elif action.action_type == 'rollback_version':
            logger.info(f"Rolling back deployment: {action.resource_id}")
            await asyncio.sleep(4)  # Simulate rollback
    
    async def _remediate_kubernetes_service(self, action: RemediationAction):
        """Remediate Kubernetes service configuration"""
        if action.action_type == 'update_service':
            logger.info(f"Updating service: {action.resource_id}")
            await asyncio.sleep(1)
        elif action.action_type == 'recreate_service':
            logger.info(f"Recreating service: {action.resource_id}")
            await asyncio.sleep(2)
    
    async def _remediate_nginx_config(self, action: RemediationAction):
        """Remediate NGINX configuration"""
        if action.action_type == 'reload_nginx':
            logger.info(f"Reloading NGINX configuration: {action.resource_id}")
            await asyncio.sleep(1)
        elif action.action_type == 'restore_config':
            logger.info(f"Restoring NGINX configuration: {action.resource_id}")
            await asyncio.sleep(2)
    
    async def _remediate_database_config(self, action: RemediationAction):
        """Remediate database configuration"""
        if action.action_type == 'restart_service':
            logger.info(f"Restarting database service: {action.resource_id}")
            await asyncio.sleep(5)
        elif action.action_type == 'restore_config':
            logger.info(f"Restoring database configuration: {action.resource_id}")
            await asyncio.sleep(3)
    
    async def _remediate_security_policy(self, action: RemediationAction):
        """Remediate security policy configuration"""
        if action.action_type == 'restore_policy':
            logger.info(f"Restoring security policy: {action.resource_id}")
            await asyncio.sleep(1)
        elif action.action_type == 'alert_security_team':
            logger.info(f"Alerting security team about: {action.resource_id}")
            await asyncio.sleep(1)
    
    async def _baseline_maintenance(self):
        """Maintain configuration baselines"""
        while self._monitoring_active:
            try:
                # Update baselines that haven't been verified recently
                cutoff_time = datetime.now() - timedelta(hours=24)
                
                for resource_id, baseline in self.baselines.items():
                    if baseline.last_verified < cutoff_time:
                        # Re-verify baseline
                        current_config = await self._get_current_configuration(resource_id, baseline.resource_type)
                        
                        # If configuration is stable, update baseline
                        if self._is_configuration_stable(baseline, current_config):
                            await self._update_baseline(resource_id, current_config)
                
                await asyncio.sleep(3600)  # Run every hour
                
            except Exception as e:
                logger.error(f"Baseline maintenance error: {str(e)}")
                await asyncio.sleep(3600)
    
    def _is_configuration_stable(self, baseline: ConfigurationBaseline, current_config: Dict[str, Any]) -> bool:
        """Check if configuration has been stable"""
        # Simple stability check - in production, this would be more sophisticated
        return baseline.configuration == current_config
    
    async def _update_baseline(self, resource_id: str, new_config: Dict[str, Any]):
        """Update configuration baseline"""
        if resource_id in self.baselines:
            baseline = self.baselines[resource_id]
            
            # Update configuration and checksum
            baseline.configuration = new_config
            config_json = json.dumps(new_config, sort_keys=True)
            baseline.checksum = hashlib.sha256(config_json.encode()).hexdigest()
            baseline.last_verified = datetime.now()
            
            logger.info(f"Updated baseline for {resource_id}")
    
    def get_drift_summary(self) -> Dict[str, Any]:
        """Get drift detection summary"""
        recent_drifts = [d for d in self.drift_history if d.detection_time > datetime.now() - timedelta(hours=24)]
        
        severity_counts = {
            'critical': len([d for d in recent_drifts if d.drift_severity == DriftSeverity.CRITICAL]),
            'high': len([d for d in recent_drifts if d.drift_severity == DriftSeverity.HIGH]),
            'medium': len([d for d in recent_drifts if d.drift_severity == DriftSeverity.MEDIUM]),
            'low': len([d for d in recent_drifts if d.drift_severity == DriftSeverity.LOW])
        }
        
        remediation_stats = {
            'pending': len([a for a in self.remediation_queue if a.status == RemediationStatus.PENDING]),
            'in_progress': len([a for a in self.remediation_queue if a.status == RemediationStatus.IN_PROGRESS]),
            'completed': len([a for a in self.remediation_history if a.status == RemediationStatus.COMPLETED]),
            'failed': len([a for a in self.remediation_history if a.status == RemediationStatus.FAILED])
        }
        
        return {
            'total_baselines': len(self.baselines),
            'recent_drifts': len(recent_drifts),
            'severity_breakdown': severity_counts,
            'remediation_stats': remediation_stats,
            'auto_remediation_enabled': self.auto_remediation_enabled,
            'detection_interval': self.detection_interval
        }