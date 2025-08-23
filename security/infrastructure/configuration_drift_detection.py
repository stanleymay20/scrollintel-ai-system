"""
Configuration Drift Detection System
Deploys configuration drift detection with 60-second auto-remediation
"""

import asyncio
import logging
import time
import hashlib
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import yaml
import os
import subprocess
import difflib

logger = logging.getLogger(__name__)

class DriftSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class DriftStatus(Enum):
    DETECTED = "detected"
    ANALYZING = "analyzing"
    REMEDIATING = "remediating"
    REMEDIATED = "remediated"
    FAILED = "failed"
    IGNORED = "ignored"

class ConfigurationType(Enum):
    SYSTEM = "system"
    APPLICATION = "application"
    NETWORK = "network"
    SECURITY = "security"
    DATABASE = "database"
    KUBERNETES = "kubernetes"
    DOCKER = "docker"

@dataclass
class ConfigurationBaseline:
    config_id: str
    config_type: ConfigurationType
    system_name: str
    config_path: str
    baseline_hash: str
    baseline_content: str
    created_at: datetime
    last_validated: datetime
    validation_frequency_minutes: int

@dataclass
class DriftDetection:
    drift_id: str
    config_baseline: ConfigurationBaseline
    detected_at: datetime
    current_hash: str
    current_content: str
    drift_severity: DriftSeverity
    drift_status: DriftStatus
    differences: List[str]
    auto_remediation_enabled: bool
    remediation_attempts: int
    last_remediation_attempt: Optional[datetime]

@dataclass
class RemediationAction:
    action_id: str
    drift_detection: DriftDetection
    action_type: str  # restore, update_baseline, ignore
    executed_at: datetime
    execution_time_seconds: float
    success: bool
    error_message: Optional[str]
    changes_applied: List[str]

class ConfigurationDriftDetection:
    """
    Advanced configuration drift detection system with 60-second auto-remediation.
    Monitors system, application, and infrastructure configurations for unauthorized changes.
    """
    
    def __init__(self):
        self.configuration_baselines: Dict[str, ConfigurationBaseline] = {}
        self.drift_detections: Dict[str, DriftDetection] = {}
        self.remediation_actions: Dict[str, RemediationAction] = {}
        
        # Configuration settings
        self.monitoring_enabled = True
        self.auto_remediation_enabled = True
        self.remediation_timeout_seconds = 60
        self.max_remediation_attempts = 3
        self.scan_interval_seconds = 30
        
        # Monitored configuration paths
        self.monitored_configs = {
            ConfigurationType.SYSTEM: [
                "/etc/passwd",
                "/etc/group",
                "/etc/sudoers",
                "/etc/ssh/sshd_config",
                "/etc/hosts",
                "/etc/resolv.conf"
            ],
            ConfigurationType.APPLICATION: [
                "/etc/nginx/nginx.conf",
                "/etc/apache2/apache2.conf",
                "/opt/app/config.yaml",
                "/opt/app/settings.json"
            ],
            ConfigurationType.SECURITY: [
                "/etc/security/limits.conf",
                "/etc/pam.d/common-auth",
                "/etc/fail2ban/jail.conf"
            ],
            ConfigurationType.NETWORK: [
                "/etc/network/interfaces",
                "/etc/iptables/rules.v4",
                "/etc/firewall.conf"
            ],
            ConfigurationType.DATABASE: [
                "/etc/postgresql/postgresql.conf",
                "/etc/mysql/my.cnf",
                "/etc/redis/redis.conf"
            ]
        }
        
        self._initialize_system()
        logger.info("Configuration drift detection system initialized")
    
    def _initialize_system(self):
        """Initialize the drift detection system"""
        try:
            # Create baseline configurations for monitored files
            asyncio.create_task(self._create_initial_baselines())
            
            # Start continuous monitoring
            if self.monitoring_enabled:
                asyncio.create_task(self._start_continuous_monitoring())
            
        except Exception as e:
            logger.error(f"Failed to initialize drift detection system: {e}")
    
    async def _create_initial_baselines(self):
        """Create initial configuration baselines"""
        try:
            logger.info("Creating initial configuration baselines")
            
            for config_type, config_paths in self.monitored_configs.items():
                for config_path in config_paths:
                    try:
                        await self._create_configuration_baseline(config_type, config_path)
                    except Exception as e:
                        logger.warning(f"Failed to create baseline for {config_path}: {e}")
            
            logger.info(f"Created {len(self.configuration_baselines)} configuration baselines")
            
        except Exception as e:
            logger.error(f"Failed to create initial baselines: {e}")
    
    async def _create_configuration_baseline(self, config_type: ConfigurationType, config_path: str) -> ConfigurationBaseline:
        """Create a configuration baseline for a specific file"""
        try:
            # Read configuration content
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
            else:
                # Create placeholder for non-existent files
                content = f"# Configuration file {config_path} does not exist\n"
            
            # Calculate hash
            content_hash = hashlib.sha256(content.encode()).hexdigest()
            
            # Extract system name from path
            system_name = os.path.basename(config_path).split('.')[0]
            
            baseline = ConfigurationBaseline(
                config_id=f"{config_type.value}_{system_name}_{int(time.time())}",
                config_type=config_type,
                system_name=system_name,
                config_path=config_path,
                baseline_hash=content_hash,
                baseline_content=content,
                created_at=datetime.now(),
                last_validated=datetime.now(),
                validation_frequency_minutes=5  # Check every 5 minutes
            )
            
            self.configuration_baselines[baseline.config_id] = baseline
            logger.info(f"Created baseline for {config_path}")
            
            return baseline
            
        except Exception as e:
            logger.error(f"Failed to create baseline for {config_path}: {e}")
            raise
    
    async def _start_continuous_monitoring(self):
        """Start continuous configuration monitoring"""
        try:
            logger.info("Starting continuous configuration monitoring")
            
            while self.monitoring_enabled:
                try:
                    # Scan all configuration baselines
                    await self._scan_configuration_drift()
                    
                    # Process detected drifts
                    await self._process_drift_detections()
                    
                    # Wait before next scan
                    await asyncio.sleep(self.scan_interval_seconds)
                    
                except Exception as e:
                    logger.error(f"Error in continuous monitoring: {e}")
                    await asyncio.sleep(self.scan_interval_seconds)
            
        except Exception as e:
            logger.error(f"Failed to start continuous monitoring: {e}")
    
    async def _scan_configuration_drift(self):
        """Scan for configuration drift across all baselines"""
        try:
            drift_count = 0
            
            for baseline_id, baseline in self.configuration_baselines.items():
                try:
                    # Check if it's time to validate this baseline
                    time_since_validation = datetime.now() - baseline.last_validated
                    if time_since_validation.total_seconds() >= (baseline.validation_frequency_minutes * 60):
                        
                        drift_detected = await self._check_configuration_drift(baseline)
                        if drift_detected:
                            drift_count += 1
                        
                        # Update last validated time
                        baseline.last_validated = datetime.now()
                
                except Exception as e:
                    logger.error(f"Failed to scan drift for baseline {baseline_id}: {e}")
            
            if drift_count > 0:
                logger.info(f"Detected configuration drift in {drift_count} configurations")
            
        except Exception as e:
            logger.error(f"Failed to scan configuration drift: {e}")
    
    async def _check_configuration_drift(self, baseline: ConfigurationBaseline) -> bool:
        """Check for drift in a specific configuration"""
        try:
            # Read current configuration content
            if os.path.exists(baseline.config_path):
                with open(baseline.config_path, 'r', encoding='utf-8', errors='ignore') as f:
                    current_content = f.read()
            else:
                current_content = f"# Configuration file {baseline.config_path} does not exist\n"
            
            # Calculate current hash
            current_hash = hashlib.sha256(current_content.encode()).hexdigest()
            
            # Check for drift
            if current_hash != baseline.baseline_hash:
                # Drift detected
                await self._create_drift_detection(baseline, current_hash, current_content)
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to check drift for {baseline.config_path}: {e}")
            return False
    
    async def _create_drift_detection(self, baseline: ConfigurationBaseline, current_hash: str, current_content: str):
        """Create a drift detection record"""
        try:
            # Calculate differences
            differences = self._calculate_differences(baseline.baseline_content, current_content)
            
            # Assess drift severity
            severity = self._assess_drift_severity(differences, baseline.config_type)
            
            drift_detection = DriftDetection(
                drift_id=f"drift_{baseline.config_id}_{int(time.time())}",
                config_baseline=baseline,
                detected_at=datetime.now(),
                current_hash=current_hash,
                current_content=current_content,
                drift_severity=severity,
                drift_status=DriftStatus.DETECTED,
                differences=differences,
                auto_remediation_enabled=self.auto_remediation_enabled and severity != DriftSeverity.LOW,
                remediation_attempts=0,
                last_remediation_attempt=None
            )
            
            self.drift_detections[drift_detection.drift_id] = drift_detection
            
            logger.warning(f"Configuration drift detected: {baseline.config_path} ({severity.value} severity)")
            
            # Send alert
            await self._send_drift_alert(drift_detection)
            
        except Exception as e:
            logger.error(f"Failed to create drift detection: {e}")
    
    def _calculate_differences(self, baseline_content: str, current_content: str) -> List[str]:
        """Calculate differences between baseline and current configuration"""
        try:
            baseline_lines = baseline_content.splitlines(keepends=True)
            current_lines = current_content.splitlines(keepends=True)
            
            diff = list(difflib.unified_diff(
                baseline_lines,
                current_lines,
                fromfile='baseline',
                tofile='current',
                lineterm=''
            ))
            
            # Filter out diff headers and return meaningful changes
            meaningful_changes = []
            for line in diff:
                if line.startswith('+') and not line.startswith('+++'):
                    meaningful_changes.append(f"Added: {line[1:].strip()}")
                elif line.startswith('-') and not line.startswith('---'):
                    meaningful_changes.append(f"Removed: {line[1:].strip()}")
            
            return meaningful_changes
            
        except Exception as e:
            logger.error(f"Failed to calculate differences: {e}")
            return ["Error calculating differences"]
    
    def _assess_drift_severity(self, differences: List[str], config_type: ConfigurationType) -> DriftSeverity:
        """Assess the severity of configuration drift"""
        try:
            change_count = len(differences)
            
            # Security configurations are always high priority
            if config_type == ConfigurationType.SECURITY:
                if change_count > 0:
                    return DriftSeverity.CRITICAL
            
            # System configurations
            elif config_type == ConfigurationType.SYSTEM:
                if change_count >= 5:
                    return DriftSeverity.HIGH
                elif change_count >= 2:
                    return DriftSeverity.MEDIUM
                else:
                    return DriftSeverity.LOW
            
            # Application configurations
            elif config_type == ConfigurationType.APPLICATION:
                if change_count >= 10:
                    return DriftSeverity.HIGH
                elif change_count >= 3:
                    return DriftSeverity.MEDIUM
                else:
                    return DriftSeverity.LOW
            
            # Network configurations
            elif config_type == ConfigurationType.NETWORK:
                if change_count >= 3:
                    return DriftSeverity.HIGH
                elif change_count >= 1:
                    return DriftSeverity.MEDIUM
                else:
                    return DriftSeverity.LOW
            
            # Database configurations
            elif config_type == ConfigurationType.DATABASE:
                if change_count >= 5:
                    return DriftSeverity.HIGH
                elif change_count >= 2:
                    return DriftSeverity.MEDIUM
                else:
                    return DriftSeverity.LOW
            
            # Default assessment
            if change_count >= 5:
                return DriftSeverity.HIGH
            elif change_count >= 2:
                return DriftSeverity.MEDIUM
            else:
                return DriftSeverity.LOW
                
        except Exception as e:
            logger.error(f"Failed to assess drift severity: {e}")
            return DriftSeverity.MEDIUM
    
    async def _send_drift_alert(self, drift_detection: DriftDetection):
        """Send alert for detected configuration drift"""
        try:
            alert_message = f"""
            CONFIGURATION DRIFT DETECTED - {drift_detection.drift_severity.value.upper()}
            
            System: {drift_detection.config_baseline.system_name}
            Configuration: {drift_detection.config_baseline.config_path}
            Type: {drift_detection.config_baseline.config_type.value}
            Detected: {drift_detection.detected_at.isoformat()}
            
            Changes Detected:
            {chr(10).join(drift_detection.differences[:10])}  # Show first 10 changes
            
            Auto-remediation: {'Enabled' if drift_detection.auto_remediation_enabled else 'Disabled'}
            """
            
            # Log alert (in production, this would send to monitoring systems)
            logger.warning(f"DRIFT ALERT: {alert_message}")
            
        except Exception as e:
            logger.error(f"Failed to send drift alert: {e}")
    
    async def _process_drift_detections(self):
        """Process all detected configuration drifts"""
        try:
            for drift_id, drift_detection in self.drift_detections.items():
                if (drift_detection.drift_status == DriftStatus.DETECTED and 
                    drift_detection.auto_remediation_enabled):
                    
                    # Check if we should attempt remediation
                    if self._should_attempt_remediation(drift_detection):
                        await self._attempt_auto_remediation(drift_detection)
            
        except Exception as e:
            logger.error(f"Failed to process drift detections: {e}")
    
    def _should_attempt_remediation(self, drift_detection: DriftDetection) -> bool:
        """Determine if auto-remediation should be attempted"""
        try:
            # Check maximum attempts
            if drift_detection.remediation_attempts >= self.max_remediation_attempts:
                return False
            
            # Check time since last attempt
            if drift_detection.last_remediation_attempt:
                time_since_last = datetime.now() - drift_detection.last_remediation_attempt
                if time_since_last.total_seconds() < self.remediation_timeout_seconds:
                    return False
            
            # Check severity - don't auto-remediate critical changes without human approval
            if drift_detection.drift_severity == DriftSeverity.CRITICAL:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to determine remediation eligibility: {e}")
            return False
    
    async def _attempt_auto_remediation(self, drift_detection: DriftDetection):
        """Attempt automatic remediation of configuration drift"""
        try:
            drift_detection.drift_status = DriftStatus.REMEDIATING
            drift_detection.remediation_attempts += 1
            drift_detection.last_remediation_attempt = datetime.now()
            
            start_time = time.time()
            
            logger.info(f"Attempting auto-remediation for {drift_detection.config_baseline.config_path}")
            
            # Determine remediation strategy
            remediation_success = await self._execute_remediation(drift_detection)
            
            execution_time = time.time() - start_time
            
            # Create remediation action record
            action = RemediationAction(
                action_id=f"remediation_{drift_detection.drift_id}_{int(time.time())}",
                drift_detection=drift_detection,
                action_type="restore" if remediation_success else "failed",
                executed_at=datetime.now(),
                execution_time_seconds=execution_time,
                success=remediation_success,
                error_message=None if remediation_success else "Remediation failed",
                changes_applied=["Configuration restored to baseline"] if remediation_success else []
            )
            
            self.remediation_actions[action.action_id] = action
            
            if remediation_success:
                drift_detection.drift_status = DriftStatus.REMEDIATED
                logger.info(f"Auto-remediation successful for {drift_detection.config_baseline.config_path}")
            else:
                drift_detection.drift_status = DriftStatus.FAILED
                logger.error(f"Auto-remediation failed for {drift_detection.config_baseline.config_path}")
            
        except Exception as e:
            logger.error(f"Failed to attempt auto-remediation: {e}")
            drift_detection.drift_status = DriftStatus.FAILED
    
    async def _execute_remediation(self, drift_detection: DriftDetection) -> bool:
        """Execute the actual remediation"""
        try:
            baseline = drift_detection.config_baseline
            
            # Create backup of current configuration
            backup_path = f"{baseline.config_path}.backup.{int(time.time())}"
            
            try:
                # Backup current configuration
                if os.path.exists(baseline.config_path):
                    with open(baseline.config_path, 'r') as src, open(backup_path, 'w') as dst:
                        dst.write(src.read())
                
                # Restore baseline configuration
                with open(baseline.config_path, 'w') as f:
                    f.write(baseline.baseline_content)
                
                # Verify restoration
                with open(baseline.config_path, 'r') as f:
                    restored_content = f.read()
                
                restored_hash = hashlib.sha256(restored_content.encode()).hexdigest()
                
                if restored_hash == baseline.baseline_hash:
                    logger.info(f"Successfully restored {baseline.config_path}")
                    
                    # Restart services if needed
                    await self._restart_affected_services(baseline)
                    
                    return True
                else:
                    logger.error(f"Restoration verification failed for {baseline.config_path}")
                    return False
                
            except Exception as e:
                # Restore from backup if remediation failed
                if os.path.exists(backup_path):
                    try:
                        with open(backup_path, 'r') as src, open(baseline.config_path, 'w') as dst:
                            dst.write(src.read())
                    except:
                        pass
                
                logger.error(f"Remediation execution failed: {e}")
                return False
            
        except Exception as e:
            logger.error(f"Failed to execute remediation: {e}")
            return False
    
    async def _restart_affected_services(self, baseline: ConfigurationBaseline):
        """Restart services affected by configuration changes"""
        try:
            # Map configuration files to services
            service_map = {
                "/etc/nginx/nginx.conf": ["nginx"],
                "/etc/apache2/apache2.conf": ["apache2"],
                "/etc/ssh/sshd_config": ["ssh", "sshd"],
                "/etc/postgresql/postgresql.conf": ["postgresql"],
                "/etc/mysql/my.cnf": ["mysql"],
                "/etc/redis/redis.conf": ["redis"]
            }
            
            services = service_map.get(baseline.config_path, [])
            
            for service in services:
                try:
                    # Simulate service restart (in production, use actual service management)
                    logger.info(f"Restarting service: {service}")
                    await asyncio.sleep(1)  # Simulate restart time
                    
                except Exception as e:
                    logger.error(f"Failed to restart service {service}: {e}")
            
        except Exception as e:
            logger.error(f"Failed to restart affected services: {e}")
    
    async def create_configuration_baseline(self, config_type: ConfigurationType, config_path: str) -> str:
        """Create a new configuration baseline"""
        try:
            baseline = await self._create_configuration_baseline(config_type, config_path)
            return baseline.config_id
            
        except Exception as e:
            logger.error(f"Failed to create configuration baseline: {e}")
            raise
    
    async def update_configuration_baseline(self, config_id: str) -> bool:
        """Update an existing configuration baseline"""
        try:
            baseline = self.configuration_baselines.get(config_id)
            if not baseline:
                raise ValueError(f"Configuration baseline {config_id} not found")
            
            # Read current configuration
            if os.path.exists(baseline.config_path):
                with open(baseline.config_path, 'r', encoding='utf-8', errors='ignore') as f:
                    current_content = f.read()
            else:
                current_content = f"# Configuration file {baseline.config_path} does not exist\n"
            
            # Update baseline
            baseline.baseline_content = current_content
            baseline.baseline_hash = hashlib.sha256(current_content.encode()).hexdigest()
            baseline.last_validated = datetime.now()
            
            logger.info(f"Updated baseline for {baseline.config_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update configuration baseline: {e}")
            return False
    
    async def ignore_drift_detection(self, drift_id: str) -> bool:
        """Ignore a specific drift detection"""
        try:
            drift_detection = self.drift_detections.get(drift_id)
            if not drift_detection:
                raise ValueError(f"Drift detection {drift_id} not found")
            
            drift_detection.drift_status = DriftStatus.IGNORED
            logger.info(f"Ignored drift detection {drift_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to ignore drift detection: {e}")
            return False
    
    async def manual_remediation(self, drift_id: str) -> bool:
        """Manually trigger remediation for a drift detection"""
        try:
            drift_detection = self.drift_detections.get(drift_id)
            if not drift_detection:
                raise ValueError(f"Drift detection {drift_id} not found")
            
            # Force remediation regardless of auto-remediation settings
            original_auto_setting = drift_detection.auto_remediation_enabled
            drift_detection.auto_remediation_enabled = True
            
            await self._attempt_auto_remediation(drift_detection)
            
            # Restore original setting
            drift_detection.auto_remediation_enabled = original_auto_setting
            
            return drift_detection.drift_status == DriftStatus.REMEDIATED
            
        except Exception as e:
            logger.error(f"Failed to perform manual remediation: {e}")
            return False
    
    def get_configuration_baseline(self, config_id: str) -> Optional[ConfigurationBaseline]:
        """Get configuration baseline by ID"""
        return self.configuration_baselines.get(config_id)
    
    def list_configuration_baselines(self) -> List[ConfigurationBaseline]:
        """List all configuration baselines"""
        return list(self.configuration_baselines.values())
    
    def get_drift_detection(self, drift_id: str) -> Optional[DriftDetection]:
        """Get drift detection by ID"""
        return self.drift_detections.get(drift_id)
    
    def list_drift_detections(self, status: Optional[DriftStatus] = None) -> List[DriftDetection]:
        """List drift detections, optionally filtered by status"""
        detections = list(self.drift_detections.values())
        
        if status:
            detections = [d for d in detections if d.drift_status == status]
        
        return detections
    
    def get_remediation_action(self, action_id: str) -> Optional[RemediationAction]:
        """Get remediation action by ID"""
        return self.remediation_actions.get(action_id)
    
    def list_remediation_actions(self) -> List[RemediationAction]:
        """List all remediation actions"""
        return list(self.remediation_actions.values())
    
    async def get_drift_statistics(self) -> Dict[str, Any]:
        """Get configuration drift statistics"""
        try:
            total_baselines = len(self.configuration_baselines)
            total_detections = len(self.drift_detections)
            
            # Count by status
            status_counts = {}
            for status in DriftStatus:
                status_counts[status.value] = len([d for d in self.drift_detections.values() if d.drift_status == status])
            
            # Count by severity
            severity_counts = {}
            for severity in DriftSeverity:
                severity_counts[severity.value] = len([d for d in self.drift_detections.values() if d.drift_severity == severity])
            
            # Count by configuration type
            type_counts = {}
            for config_type in ConfigurationType:
                type_counts[config_type.value] = len([b for b in self.configuration_baselines.values() if b.config_type == config_type])
            
            # Remediation statistics
            total_remediations = len(self.remediation_actions)
            successful_remediations = len([a for a in self.remediation_actions.values() if a.success])
            
            return {
                "total_baselines": total_baselines,
                "total_detections": total_detections,
                "status_breakdown": status_counts,
                "severity_breakdown": severity_counts,
                "type_breakdown": type_counts,
                "total_remediations": total_remediations,
                "successful_remediations": successful_remediations,
                "remediation_success_rate": (successful_remediations / total_remediations * 100) if total_remediations > 0 else 0,
                "auto_remediation_enabled": self.auto_remediation_enabled,
                "monitoring_enabled": self.monitoring_enabled
            }
            
        except Exception as e:
            logger.error(f"Failed to get drift statistics: {e}")
            return {}
    
    def enable_monitoring(self):
        """Enable configuration monitoring"""
        self.monitoring_enabled = True
        if not hasattr(self, '_monitoring_task') or self._monitoring_task.done():
            self._monitoring_task = asyncio.create_task(self._start_continuous_monitoring())
        logger.info("Configuration monitoring enabled")
    
    def disable_monitoring(self):
        """Disable configuration monitoring"""
        self.monitoring_enabled = False
        logger.info("Configuration monitoring disabled")
    
    def enable_auto_remediation(self):
        """Enable automatic remediation"""
        self.auto_remediation_enabled = True
        logger.info("Auto-remediation enabled")
    
    def disable_auto_remediation(self):
        """Disable automatic remediation"""
        self.auto_remediation_enabled = False
        logger.info("Auto-remediation disabled")

# Global instance
configuration_drift_detection = ConfigurationDriftDetection()