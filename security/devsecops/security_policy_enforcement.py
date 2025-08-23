"""
Automated Security Policy Enforcement Across All Environments
Implements comprehensive security policy enforcement for DevSecOps pipeline
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
from dataclasses import dataclass, field
import json
import yaml
import re
from pathlib import Path

logger = logging.getLogger(__name__)

class PolicyType(Enum):
    SECURITY = "security"
    COMPLIANCE = "compliance"
    OPERATIONAL = "operational"
    GOVERNANCE = "governance"

class PolicySeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class PolicyAction(Enum):
    ALLOW = "allow"
    WARN = "warn"
    BLOCK = "block"
    QUARANTINE = "quarantine"

class PolicyScope(Enum):
    GLOBAL = "global"
    ENVIRONMENT = "environment"
    APPLICATION = "application"
    RESOURCE = "resource"

class ViolationStatus(Enum):
    ACTIVE = "active"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"
    INVESTIGATING = "investigating"

@dataclass
class SecurityPolicy:
    policy_id: str
    name: str
    description: str
    policy_type: PolicyType
    severity: PolicySeverity
    action: PolicyAction
    scope: PolicyScope
    rules: List[Dict[str, Any]]
    environments: List[str]
    enabled: bool
    created_by: str
    created_at: datetime
    updated_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PolicyViolation:
    violation_id: str
    policy_id: str
    resource_id: str
    resource_type: str
    environment: str
    severity: PolicySeverity
    message: str
    details: Dict[str, Any]
    status: ViolationStatus
    detected_at: datetime
    resolved_at: Optional[datetime] = None
    suppressed_by: Optional[str] = None
    suppression_reason: Optional[str] = None

@dataclass
class PolicyEnforcementResult:
    enforcement_id: str
    timestamp: datetime
    environment: str
    policies_evaluated: int
    violations_found: int
    violations: List[PolicyViolation]
    actions_taken: List[Dict[str, Any]]
    execution_time: float

class SecurityPolicyEnforcement:
    """
    Automated security policy enforcement across all environments
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.policies = {}
        self.violations = {}
        self.policy_engines = self._initialize_policy_engines()
        self.enforcement_history = []
        
        # Load default policies
        asyncio.create_task(self._load_default_policies())
    
    def _initialize_policy_engines(self) -> Dict[str, Callable]:
        """Initialize policy enforcement engines"""
        return {
            "opa": self._enforce_opa_policies,
            "kubernetes": self._enforce_kubernetes_policies,
            "terraform": self._enforce_terraform_policies,
            "docker": self._enforce_docker_policies,
            "aws": self._enforce_aws_policies,
            "network": self._enforce_network_policies,
            "data": self._enforce_data_policies
        }
    
    async def _load_default_policies(self):
        """Load default security policies"""
        default_policies = [
            # Container Security Policies
            {
                "policy_id": "container-root-user",
                "name": "Container Root User Policy",
                "description": "Containers must not run as root user",
                "policy_type": PolicyType.SECURITY,
                "severity": PolicySeverity.ERROR,
                "action": PolicyAction.BLOCK,
                "scope": PolicyScope.GLOBAL,
                "rules": [
                    {
                        "type": "container_security_context",
                        "condition": "runAsUser != 0 AND runAsNonRoot == true"
                    }
                ],
                "environments": ["development", "staging", "production"]
            },
            {
                "policy_id": "container-privileged-mode",
                "name": "Container Privileged Mode Policy",
                "description": "Containers must not run in privileged mode",
                "policy_type": PolicyType.SECURITY,
                "severity": PolicySeverity.CRITICAL,
                "action": PolicyAction.BLOCK,
                "scope": PolicyScope.GLOBAL,
                "rules": [
                    {
                        "type": "container_security_context",
                        "condition": "privileged != true"
                    }
                ],
                "environments": ["development", "staging", "production"]
            },
            # Network Security Policies
            {
                "policy_id": "network-ingress-tls",
                "name": "Ingress TLS Policy",
                "description": "All ingress resources must use TLS",
                "policy_type": PolicyType.SECURITY,
                "severity": PolicySeverity.ERROR,
                "action": PolicyAction.BLOCK,
                "scope": PolicyScope.GLOBAL,
                "rules": [
                    {
                        "type": "kubernetes_ingress",
                        "condition": "spec.tls is defined"
                    }
                ],
                "environments": ["staging", "production"]
            },
            # Resource Limits Policies
            {
                "policy_id": "resource-limits-required",
                "name": "Resource Limits Required",
                "description": "All containers must have resource limits defined",
                "policy_type": PolicyType.OPERATIONAL,
                "severity": PolicySeverity.WARNING,
                "action": PolicyAction.WARN,
                "scope": PolicyScope.GLOBAL,
                "rules": [
                    {
                        "type": "container_resources",
                        "condition": "limits.memory is defined AND limits.cpu is defined"
                    }
                ],
                "environments": ["development", "staging", "production"]
            },
            # Compliance Policies
            {
                "policy_id": "data-encryption-at-rest",
                "name": "Data Encryption at Rest",
                "description": "All data stores must have encryption at rest enabled",
                "policy_type": PolicyType.COMPLIANCE,
                "severity": PolicySeverity.CRITICAL,
                "action": PolicyAction.BLOCK,
                "scope": PolicyScope.GLOBAL,
                "rules": [
                    {
                        "type": "aws_rds",
                        "condition": "storage_encrypted == true"
                    },
                    {
                        "type": "aws_s3",
                        "condition": "server_side_encryption_configuration is defined"
                    }
                ],
                "environments": ["production"]
            }
        ]
        
        for policy_data in default_policies:
            await self.create_policy(policy_data, "system")
    
    async def create_policy(self, policy_data: Dict[str, Any], created_by: str) -> SecurityPolicy:
        """Create a new security policy"""
        policy_id = policy_data.get("policy_id") or f"policy-{int(datetime.now().timestamp())}"
        
        policy = SecurityPolicy(
            policy_id=policy_id,
            name=policy_data["name"],
            description=policy_data["description"],
            policy_type=PolicyType(policy_data["policy_type"]),
            severity=PolicySeverity(policy_data["severity"]),
            action=PolicyAction(policy_data["action"]),
            scope=PolicyScope(policy_data["scope"]),
            rules=policy_data["rules"],
            environments=policy_data.get("environments", ["all"]),
            enabled=policy_data.get("enabled", True),
            created_by=created_by,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            metadata=policy_data.get("metadata", {})
        )
        
        self.policies[policy_id] = policy
        logger.info(f"Created security policy: {policy_id}")
        
        return policy
    
    async def update_policy(
        self, 
        policy_id: str,
        updates: Dict[str, Any],
        updated_by: str
    ) -> SecurityPolicy:
        """Update existing security policy"""
        if policy_id not in self.policies:
            raise ValueError(f"Policy {policy_id} not found")
        
        policy = self.policies[policy_id]
        
        # Update policy fields
        for field, value in updates.items():
            if hasattr(policy, field):
                if field in ["policy_type", "severity", "action", "scope"]:
                    # Handle enum fields
                    enum_class = {
                        "policy_type": PolicyType,
                        "severity": PolicySeverity,
                        "action": PolicyAction,
                        "scope": PolicyScope
                    }[field]
                    setattr(policy, field, enum_class(value))
                else:
                    setattr(policy, field, value)
        
        policy.updated_at = datetime.now()
        
        logger.info(f"Updated security policy: {policy_id} by {updated_by}")
        return policy
    
    async def delete_policy(self, policy_id: str) -> bool:
        """Delete security policy"""
        if policy_id not in self.policies:
            return False
        
        del self.policies[policy_id]
        logger.info(f"Deleted security policy: {policy_id}")
        return True
    
    async def enforce_policies(
        self, 
        environment: str,
        resource_context: Dict[str, Any]
    ) -> PolicyEnforcementResult:
        """Enforce security policies for given environment and context"""
        enforcement_id = f"enforcement-{int(datetime.now().timestamp())}"
        start_time = datetime.now()
        
        logger.info(f"Starting policy enforcement for environment: {environment}")
        
        # Get applicable policies for environment
        applicable_policies = self._get_applicable_policies(environment)
        
        violations = []
        actions_taken = []
        
        for policy in applicable_policies:
            if not policy.enabled:
                continue
            
            try:
                # Evaluate policy rules
                policy_violations = await self._evaluate_policy(policy, resource_context, environment)
                violations.extend(policy_violations)
                
                # Take enforcement actions
                for violation in policy_violations:
                    action_result = await self._take_enforcement_action(policy, violation)
                    if action_result:
                        actions_taken.append(action_result)
                
            except Exception as e:
                logger.error(f"Error enforcing policy {policy.policy_id}: {str(e)}")
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        result = PolicyEnforcementResult(
            enforcement_id=enforcement_id,
            timestamp=start_time,
            environment=environment,
            policies_evaluated=len(applicable_policies),
            violations_found=len(violations),
            violations=violations,
            actions_taken=actions_taken,
            execution_time=execution_time
        )
        
        self.enforcement_history.append(result)
        
        logger.info(f"Policy enforcement completed: {len(violations)} violations found")
        return result
    
    def _get_applicable_policies(self, environment: str) -> List[SecurityPolicy]:
        """Get policies applicable to the given environment"""
        applicable_policies = []
        
        for policy in self.policies.values():
            if ("all" in policy.environments or 
                environment in policy.environments):
                applicable_policies.append(policy)
        
        return applicable_policies
    
    async def _evaluate_policy(
        self, 
        policy: SecurityPolicy,
        resource_context: Dict[str, Any],
        environment: str
    ) -> List[PolicyViolation]:
        """Evaluate a single policy against resource context"""
        violations = []
        
        for rule in policy.rules:
            rule_type = rule.get("type")
            condition = rule.get("condition")
            
            # Get appropriate policy engine
            engine = self._get_policy_engine(rule_type)
            if not engine:
                logger.warning(f"No policy engine found for rule type: {rule_type}")
                continue
            
            try:
                # Evaluate rule using appropriate engine
                rule_violations = await engine(policy, rule, resource_context, environment)
                violations.extend(rule_violations)
                
            except Exception as e:
                logger.error(f"Error evaluating rule {rule_type}: {str(e)}")
        
        return violations
    
    def _get_policy_engine(self, rule_type: str) -> Optional[Callable]:
        """Get appropriate policy engine for rule type"""
        engine_mapping = {
            "container_security_context": self.policy_engines["kubernetes"],
            "container_resources": self.policy_engines["kubernetes"],
            "kubernetes_ingress": self.policy_engines["kubernetes"],
            "aws_rds": self.policy_engines["aws"],
            "aws_s3": self.policy_engines["aws"],
            "terraform_resource": self.policy_engines["terraform"],
            "docker_image": self.policy_engines["docker"],
            "network_policy": self.policy_engines["network"],
            "data_classification": self.policy_engines["data"]
        }
        
        return engine_mapping.get(rule_type)
    
    async def _enforce_kubernetes_policies(
        self, 
        policy: SecurityPolicy,
        rule: Dict[str, Any],
        resource_context: Dict[str, Any],
        environment: str
    ) -> List[PolicyViolation]:
        """Enforce Kubernetes-specific policies"""
        violations = []
        
        # Get Kubernetes resources from context
        k8s_resources = resource_context.get("kubernetes_resources", [])
        
        for resource in k8s_resources:
            resource_type = resource.get("kind", "Unknown")
            resource_name = resource.get("metadata", {}).get("name", "Unknown")
            
            # Evaluate rule condition
            violation = await self._evaluate_kubernetes_rule(
                policy, rule, resource, environment
            )
            
            if violation:
                violation_id = f"violation-{int(datetime.now().timestamp())}-{len(self.violations)}"
                
                policy_violation = PolicyViolation(
                    violation_id=violation_id,
                    policy_id=policy.policy_id,
                    resource_id=f"{resource_type}/{resource_name}",
                    resource_type=resource_type,
                    environment=environment,
                    severity=policy.severity,
                    message=violation["message"],
                    details=violation["details"],
                    status=ViolationStatus.ACTIVE,
                    detected_at=datetime.now()
                )
                
                violations.append(policy_violation)
                self.violations[violation_id] = policy_violation
        
        return violations
    
    async def _evaluate_kubernetes_rule(
        self, 
        policy: SecurityPolicy,
        rule: Dict[str, Any],
        resource: Dict[str, Any],
        environment: str
    ) -> Optional[Dict[str, Any]]:
        """Evaluate Kubernetes rule against resource"""
        rule_type = rule.get("type")
        condition = rule.get("condition")
        
        if rule_type == "container_security_context":
            return await self._check_container_security_context(resource, condition)
        elif rule_type == "container_resources":
            return await self._check_container_resources(resource, condition)
        elif rule_type == "kubernetes_ingress":
            return await self._check_ingress_configuration(resource, condition)
        
        return None
    
    async def _check_container_security_context(
        self, 
        resource: Dict[str, Any],
        condition: str
    ) -> Optional[Dict[str, Any]]:
        """Check container security context"""
        if resource.get("kind") != "Deployment":
            return None
        
        containers = (resource.get("spec", {})
                     .get("template", {})
                     .get("spec", {})
                     .get("containers", []))
        
        for container in containers:
            security_context = container.get("securityContext", {})
            
            # Check runAsUser
            if "runAsUser != 0" in condition:
                run_as_user = security_context.get("runAsUser")
                if run_as_user is None or run_as_user == 0:
                    return {
                        "message": f"Container {container.get('name')} runs as root user",
                        "details": {
                            "container": container.get("name"),
                            "runAsUser": run_as_user,
                            "condition": condition
                        }
                    }
            
            # Check runAsNonRoot
            if "runAsNonRoot == true" in condition:
                run_as_non_root = security_context.get("runAsNonRoot")
                if not run_as_non_root:
                    return {
                        "message": f"Container {container.get('name')} not configured to run as non-root",
                        "details": {
                            "container": container.get("name"),
                            "runAsNonRoot": run_as_non_root,
                            "condition": condition
                        }
                    }
            
            # Check privileged
            if "privileged != true" in condition:
                privileged = security_context.get("privileged")
                if privileged:
                    return {
                        "message": f"Container {container.get('name')} runs in privileged mode",
                        "details": {
                            "container": container.get("name"),
                            "privileged": privileged,
                            "condition": condition
                        }
                    }
        
        return None
    
    async def _check_container_resources(
        self, 
        resource: Dict[str, Any],
        condition: str
    ) -> Optional[Dict[str, Any]]:
        """Check container resource limits"""
        if resource.get("kind") != "Deployment":
            return None
        
        containers = (resource.get("spec", {})
                     .get("template", {})
                     .get("spec", {})
                     .get("containers", []))
        
        for container in containers:
            resources = container.get("resources", {})
            limits = resources.get("limits", {})
            
            # Check memory limits
            if "limits.memory is defined" in condition:
                if "memory" not in limits:
                    return {
                        "message": f"Container {container.get('name')} missing memory limits",
                        "details": {
                            "container": container.get("name"),
                            "limits": limits,
                            "condition": condition
                        }
                    }
            
            # Check CPU limits
            if "limits.cpu is defined" in condition:
                if "cpu" not in limits:
                    return {
                        "message": f"Container {container.get('name')} missing CPU limits",
                        "details": {
                            "container": container.get("name"),
                            "limits": limits,
                            "condition": condition
                        }
                    }
        
        return None
    
    async def _check_ingress_configuration(
        self, 
        resource: Dict[str, Any],
        condition: str
    ) -> Optional[Dict[str, Any]]:
        """Check ingress configuration"""
        if resource.get("kind") != "Ingress":
            return None
        
        spec = resource.get("spec", {})
        
        # Check TLS configuration
        if "spec.tls is defined" in condition:
            tls = spec.get("tls")
            if not tls:
                return {
                    "message": f"Ingress {resource.get('metadata', {}).get('name')} missing TLS configuration",
                    "details": {
                        "ingress": resource.get("metadata", {}).get("name"),
                        "tls": tls,
                        "condition": condition
                    }
                }
        
        return None
    
    async def _enforce_aws_policies(
        self, 
        policy: SecurityPolicy,
        rule: Dict[str, Any],
        resource_context: Dict[str, Any],
        environment: str
    ) -> List[PolicyViolation]:
        """Enforce AWS-specific policies"""
        violations = []
        
        # Get AWS resources from context
        aws_resources = resource_context.get("aws_resources", [])
        
        for resource in aws_resources:
            violation = await self._evaluate_aws_rule(policy, rule, resource, environment)
            
            if violation:
                violation_id = f"violation-{int(datetime.now().timestamp())}-{len(self.violations)}"
                
                policy_violation = PolicyViolation(
                    violation_id=violation_id,
                    policy_id=policy.policy_id,
                    resource_id=resource.get("resource_id", "Unknown"),
                    resource_type=resource.get("resource_type", "Unknown"),
                    environment=environment,
                    severity=policy.severity,
                    message=violation["message"],
                    details=violation["details"],
                    status=ViolationStatus.ACTIVE,
                    detected_at=datetime.now()
                )
                
                violations.append(policy_violation)
                self.violations[violation_id] = policy_violation
        
        return violations
    
    async def _evaluate_aws_rule(
        self, 
        policy: SecurityPolicy,
        rule: Dict[str, Any],
        resource: Dict[str, Any],
        environment: str
    ) -> Optional[Dict[str, Any]]:
        """Evaluate AWS rule against resource"""
        rule_type = rule.get("type")
        condition = rule.get("condition")
        
        if rule_type == "aws_rds":
            return await self._check_rds_encryption(resource, condition)
        elif rule_type == "aws_s3":
            return await self._check_s3_encryption(resource, condition)
        
        return None
    
    async def _check_rds_encryption(
        self, 
        resource: Dict[str, Any],
        condition: str
    ) -> Optional[Dict[str, Any]]:
        """Check RDS encryption configuration"""
        if resource.get("resource_type") != "aws_rds_instance":
            return None
        
        config = resource.get("configuration", {})
        
        if "storage_encrypted == true" in condition:
            storage_encrypted = config.get("storage_encrypted", False)
            if not storage_encrypted:
                return {
                    "message": f"RDS instance {resource.get('resource_id')} does not have encryption at rest enabled",
                    "details": {
                        "resource_id": resource.get("resource_id"),
                        "storage_encrypted": storage_encrypted,
                        "condition": condition
                    }
                }
        
        return None
    
    async def _check_s3_encryption(
        self, 
        resource: Dict[str, Any],
        condition: str
    ) -> Optional[Dict[str, Any]]:
        """Check S3 encryption configuration"""
        if resource.get("resource_type") != "aws_s3_bucket":
            return None
        
        config = resource.get("configuration", {})
        
        if "server_side_encryption_configuration is defined" in condition:
            encryption_config = config.get("server_side_encryption_configuration")
            if not encryption_config:
                return {
                    "message": f"S3 bucket {resource.get('resource_id')} does not have server-side encryption configured",
                    "details": {
                        "resource_id": resource.get("resource_id"),
                        "encryption_config": encryption_config,
                        "condition": condition
                    }
                }
        
        return None
    
    async def _enforce_terraform_policies(
        self, 
        policy: SecurityPolicy,
        rule: Dict[str, Any],
        resource_context: Dict[str, Any],
        environment: str
    ) -> List[PolicyViolation]:
        """Enforce Terraform-specific policies"""
        # Placeholder for Terraform policy enforcement
        return []
    
    async def _enforce_docker_policies(
        self, 
        policy: SecurityPolicy,
        rule: Dict[str, Any],
        resource_context: Dict[str, Any],
        environment: str
    ) -> List[PolicyViolation]:
        """Enforce Docker-specific policies"""
        # Placeholder for Docker policy enforcement
        return []
    
    async def _enforce_network_policies(
        self, 
        policy: SecurityPolicy,
        rule: Dict[str, Any],
        resource_context: Dict[str, Any],
        environment: str
    ) -> List[PolicyViolation]:
        """Enforce network-specific policies"""
        # Placeholder for network policy enforcement
        return []
    
    async def _enforce_data_policies(
        self, 
        policy: SecurityPolicy,
        rule: Dict[str, Any],
        resource_context: Dict[str, Any],
        environment: str
    ) -> List[PolicyViolation]:
        """Enforce data-specific policies"""
        # Placeholder for data policy enforcement
        return []
    
    async def _enforce_opa_policies(
        self, 
        policy: SecurityPolicy,
        rule: Dict[str, Any],
        resource_context: Dict[str, Any],
        environment: str
    ) -> List[PolicyViolation]:
        """Enforce policies using Open Policy Agent"""
        # Placeholder for OPA policy enforcement
        return []
    
    async def _take_enforcement_action(
        self, 
        policy: SecurityPolicy,
        violation: PolicyViolation
    ) -> Optional[Dict[str, Any]]:
        """Take enforcement action based on policy"""
        action = policy.action
        
        if action == PolicyAction.ALLOW:
            return None
        elif action == PolicyAction.WARN:
            return await self._send_warning(policy, violation)
        elif action == PolicyAction.BLOCK:
            return await self._block_resource(policy, violation)
        elif action == PolicyAction.QUARANTINE:
            return await self._quarantine_resource(policy, violation)
        
        return None
    
    async def _send_warning(
        self, 
        policy: SecurityPolicy,
        violation: PolicyViolation
    ) -> Dict[str, Any]:
        """Send warning for policy violation"""
        logger.warning(f"Policy violation warning: {violation.message}")
        
        # In a real implementation, this would send notifications
        return {
            "action": "warning_sent",
            "policy_id": policy.policy_id,
            "violation_id": violation.violation_id,
            "timestamp": datetime.now().isoformat()
        }
    
    async def _block_resource(
        self, 
        policy: SecurityPolicy,
        violation: PolicyViolation
    ) -> Dict[str, Any]:
        """Block resource due to policy violation"""
        logger.error(f"Blocking resource due to policy violation: {violation.message}")
        
        # In a real implementation, this would prevent resource deployment
        return {
            "action": "resource_blocked",
            "policy_id": policy.policy_id,
            "violation_id": violation.violation_id,
            "resource_id": violation.resource_id,
            "timestamp": datetime.now().isoformat()
        }
    
    async def _quarantine_resource(
        self, 
        policy: SecurityPolicy,
        violation: PolicyViolation
    ) -> Dict[str, Any]:
        """Quarantine resource due to policy violation"""
        logger.error(f"Quarantining resource due to policy violation: {violation.message}")
        
        # In a real implementation, this would isolate the resource
        return {
            "action": "resource_quarantined",
            "policy_id": policy.policy_id,
            "violation_id": violation.violation_id,
            "resource_id": violation.resource_id,
            "timestamp": datetime.now().isoformat()
        }
    
    async def resolve_violation(
        self, 
        violation_id: str,
        resolved_by: str,
        resolution_notes: str = ""
    ) -> bool:
        """Mark violation as resolved"""
        if violation_id not in self.violations:
            return False
        
        violation = self.violations[violation_id]
        violation.status = ViolationStatus.RESOLVED
        violation.resolved_at = datetime.now()
        
        logger.info(f"Violation {violation_id} resolved by {resolved_by}")
        return True
    
    async def suppress_violation(
        self, 
        violation_id: str,
        suppressed_by: str,
        suppression_reason: str
    ) -> bool:
        """Suppress violation with reason"""
        if violation_id not in self.violations:
            return False
        
        violation = self.violations[violation_id]
        violation.status = ViolationStatus.SUPPRESSED
        violation.suppressed_by = suppressed_by
        violation.suppression_reason = suppression_reason
        
        logger.info(f"Violation {violation_id} suppressed by {suppressed_by}: {suppression_reason}")
        return True
    
    def get_policy_violations(
        self, 
        environment: Optional[str] = None,
        status: Optional[ViolationStatus] = None
    ) -> List[PolicyViolation]:
        """Get policy violations with optional filters"""
        violations = list(self.violations.values())
        
        if environment:
            violations = [v for v in violations if v.environment == environment]
        
        if status:
            violations = [v for v in violations if v.status == status]
        
        return violations
    
    def get_policy_compliance_report(self, environment: str) -> Dict[str, Any]:
        """Generate policy compliance report for environment"""
        applicable_policies = self._get_applicable_policies(environment)
        environment_violations = self.get_policy_violations(environment=environment)
        
        # Group violations by policy
        violations_by_policy = {}
        for violation in environment_violations:
            policy_id = violation.policy_id
            if policy_id not in violations_by_policy:
                violations_by_policy[policy_id] = []
            violations_by_policy[policy_id].append(violation)
        
        # Calculate compliance scores
        policy_compliance = {}
        for policy in applicable_policies:
            policy_violations = violations_by_policy.get(policy.policy_id, [])
            active_violations = [v for v in policy_violations if v.status == ViolationStatus.ACTIVE]
            
            policy_compliance[policy.policy_id] = {
                "policy_name": policy.name,
                "total_violations": len(policy_violations),
                "active_violations": len(active_violations),
                "compliance_score": max(0, 100 - len(active_violations) * 10)
            }
        
        # Overall compliance score
        total_active_violations = len([v for v in environment_violations if v.status == ViolationStatus.ACTIVE])
        overall_compliance_score = max(0, 100 - total_active_violations * 5)
        
        return {
            "environment": environment,
            "timestamp": datetime.now().isoformat(),
            "overall_compliance_score": overall_compliance_score,
            "total_policies": len(applicable_policies),
            "total_violations": len(environment_violations),
            "active_violations": total_active_violations,
            "policy_compliance": policy_compliance,
            "violation_summary": {
                "critical": len([v for v in environment_violations if v.severity == PolicySeverity.CRITICAL]),
                "error": len([v for v in environment_violations if v.severity == PolicySeverity.ERROR]),
                "warning": len([v for v in environment_violations if v.severity == PolicySeverity.WARNING]),
                "info": len([v for v in environment_violations if v.severity == PolicySeverity.INFO])
            }
        }