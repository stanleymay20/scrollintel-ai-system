"""
Container Security Policies Implementation
Implements restricted pod security standards and security contexts
"""

import yaml
import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class SecurityLevel(Enum):
    PRIVILEGED = "privileged"
    BASELINE = "baseline"
    RESTRICTED = "restricted"

@dataclass
class SecurityContext:
    run_as_user: Optional[int]
    run_as_group: Optional[int]
    run_as_non_root: bool
    read_only_root_filesystem: bool
    allow_privilege_escalation: bool
    capabilities_add: List[str]
    capabilities_drop: List[str]
    seccomp_profile: Optional[str]
    selinux_options: Optional[Dict[str, str]]

@dataclass
class PodSecurityPolicy:
    name: str
    security_level: SecurityLevel
    security_context: SecurityContext
    allowed_volumes: List[str]
    forbidden_syscalls: List[str]
    required_labels: Dict[str, str]
    resource_limits: Dict[str, str]

class ContainerSecurityPolicyManager:
    """Manages container security policies and enforcement"""
    
    def __init__(self):
        self.policies = self._load_default_policies()
        self.security_contexts = self._load_security_contexts()
        
    def _load_default_policies(self) -> Dict[str, PodSecurityPolicy]:
        """Load default security policies"""
        policies = {}
        
        # Restricted policy (most secure)
        restricted_context = SecurityContext(
            run_as_user=1000,
            run_as_group=1000,
            run_as_non_root=True,
            read_only_root_filesystem=True,
            allow_privilege_escalation=False,
            capabilities_add=[],
            capabilities_drop=["ALL"],
            seccomp_profile="RuntimeDefault",
            selinux_options=None
        )
        
        policies["restricted"] = PodSecurityPolicy(
            name="restricted",
            security_level=SecurityLevel.RESTRICTED,
            security_context=restricted_context,
            allowed_volumes=[
                "configMap", "downwardAPI", "emptyDir", 
                "persistentVolumeClaim", "projected", "secret"
            ],
            forbidden_syscalls=[
                "mount", "umount", "reboot", "swapon", "swapoff"
            ],
            required_labels={
                "security.scrollintel.com/policy": "restricted",
                "security.scrollintel.com/scan-required": "true"
            },
            resource_limits={
                "memory": "512Mi",
                "cpu": "500m",
                "ephemeral-storage": "1Gi"
            }
        )
        
        # Baseline policy (moderate security)
        baseline_context = SecurityContext(
            run_as_user=1000,
            run_as_group=1000,
            run_as_non_root=True,
            read_only_root_filesystem=False,
            allow_privilege_escalation=False,
            capabilities_add=[],
            capabilities_drop=["ALL"],
            seccomp_profile="RuntimeDefault",
            selinux_options=None
        )
        
        policies["baseline"] = PodSecurityPolicy(
            name="baseline",
            security_level=SecurityLevel.BASELINE,
            security_context=baseline_context,
            allowed_volumes=[
                "configMap", "downwardAPI", "emptyDir", 
                "persistentVolumeClaim", "projected", "secret", "hostPath"
            ],
            forbidden_syscalls=[
                "mount", "umount", "reboot"
            ],
            required_labels={
                "security.scrollintel.com/policy": "baseline"
            },
            resource_limits={
                "memory": "1Gi",
                "cpu": "1000m",
                "ephemeral-storage": "2Gi"
            }
        )
        
        return policies
    
    def _load_security_contexts(self) -> Dict[str, Dict[str, Any]]:
        """Load predefined security contexts"""
        return {
            "api-service": {
                "securityContext": {
                    "runAsNonRoot": True,
                    "runAsUser": 1000,
                    "runAsGroup": 1000,
                    "fsGroup": 1000,
                    "seccompProfile": {
                        "type": "RuntimeDefault"
                    }
                },
                "containerSecurityContext": {
                    "allowPrivilegeEscalation": False,
                    "readOnlyRootFilesystem": True,
                    "runAsNonRoot": True,
                    "runAsUser": 1000,
                    "runAsGroup": 1000,
                    "capabilities": {
                        "drop": ["ALL"]
                    },
                    "seccompProfile": {
                        "type": "RuntimeDefault"
                    }
                }
            },
            "database": {
                "securityContext": {
                    "runAsNonRoot": True,
                    "runAsUser": 999,
                    "runAsGroup": 999,
                    "fsGroup": 999,
                    "seccompProfile": {
                        "type": "RuntimeDefault"
                    }
                },
                "containerSecurityContext": {
                    "allowPrivilegeEscalation": False,
                    "readOnlyRootFilesystem": False,  # Database needs write access
                    "runAsNonRoot": True,
                    "runAsUser": 999,
                    "runAsGroup": 999,
                    "capabilities": {
                        "drop": ["ALL"]
                    },
                    "seccompProfile": {
                        "type": "RuntimeDefault"
                    }
                }
            },
            "frontend": {
                "securityContext": {
                    "runAsNonRoot": True,
                    "runAsUser": 101,  # nginx user
                    "runAsGroup": 101,
                    "fsGroup": 101,
                    "seccompProfile": {
                        "type": "RuntimeDefault"
                    }
                },
                "containerSecurityContext": {
                    "allowPrivilegeEscalation": False,
                    "readOnlyRootFilesystem": True,
                    "runAsNonRoot": True,
                    "runAsUser": 101,
                    "runAsGroup": 101,
                    "capabilities": {
                        "drop": ["ALL"]
                    },
                    "seccompProfile": {
                        "type": "RuntimeDefault"
                    }
                }
            }
        }
    
    def generate_pod_security_policy(self, policy_name: str) -> Dict[str, Any]:
        """Generate Kubernetes PodSecurityPolicy manifest"""
        if policy_name not in self.policies:
            raise ValueError(f"Unknown policy: {policy_name}")
        
        policy = self.policies[policy_name]
        
        psp = {
            "apiVersion": "policy/v1beta1",
            "kind": "PodSecurityPolicy",
            "metadata": {
                "name": f"scrollintel-{policy.name}",
                "labels": {
                    "security.scrollintel.com/policy": policy.name
                }
            },
            "spec": {
                "privileged": False,
                "allowPrivilegeEscalation": policy.security_context.allow_privilege_escalation,
                "requiredDropCapabilities": policy.security_context.capabilities_drop,
                "allowedCapabilities": policy.security_context.capabilities_add,
                "volumes": policy.allowed_volumes,
                "hostNetwork": False,
                "hostIPC": False,
                "hostPID": False,
                "runAsUser": {
                    "rule": "MustRunAsNonRoot"
                },
                "seLinux": {
                    "rule": "RunAsAny"
                },
                "fsGroup": {
                    "rule": "RunAsAny"
                },
                "readOnlyRootFilesystem": policy.security_context.read_only_root_filesystem
            }
        }
        
        return psp
    
    def generate_security_context_constraints(self, policy_name: str) -> Dict[str, Any]:
        """Generate OpenShift SecurityContextConstraints"""
        if policy_name not in self.policies:
            raise ValueError(f"Unknown policy: {policy_name}")
        
        policy = self.policies[policy_name]
        
        scc = {
            "apiVersion": "security.openshift.io/v1",
            "kind": "SecurityContextConstraints",
            "metadata": {
                "name": f"scrollintel-{policy.name}"
            },
            "allowHostDirVolumePlugin": False,
            "allowHostIPC": False,
            "allowHostNetwork": False,
            "allowHostPID": False,
            "allowHostPorts": False,
            "allowPrivilegedContainer": False,
            "allowedCapabilities": policy.security_context.capabilities_add,
            "defaultAddCapabilities": [],
            "requiredDropCapabilities": policy.security_context.capabilities_drop,
            "allowedFlexVolumes": [],
            "fsGroup": {
                "type": "MustRunAs",
                "ranges": [{"min": 1000, "max": 65535}]
            },
            "readOnlyRootFilesystem": policy.security_context.read_only_root_filesystem,
            "runAsUser": {
                "type": "MustRunAsNonRoot"
            },
            "seLinuxContext": {
                "type": "MustRunAs"
            },
            "supplementalGroups": {
                "type": "MustRunAs",
                "ranges": [{"min": 1000, "max": 65535}]
            },
            "volumes": policy.allowed_volumes
        }
        
        return scc
    
    def generate_deployment_with_security(self, app_name: str, image: str, 
                                        security_profile: str = "api-service") -> Dict[str, Any]:
        """Generate secure deployment manifest"""
        if security_profile not in self.security_contexts:
            raise ValueError(f"Unknown security profile: {security_profile}")
        
        security_config = self.security_contexts[security_profile]
        
        deployment = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": f"{app_name}-secure",
                "namespace": "scrollintel",
                "labels": {
                    "app": app_name,
                    "security.scrollintel.com/profile": security_profile,
                    "security.scrollintel.com/hardened": "true"
                }
            },
            "spec": {
                "replicas": 3,
                "selector": {
                    "matchLabels": {
                        "app": app_name
                    }
                },
                "template": {
                    "metadata": {
                        "labels": {
                            "app": app_name,
                            "security.scrollintel.com/profile": security_profile
                        },
                        "annotations": {
                            "sidecar.istio.io/inject": "true",
                            "security.scrollintel.com/scan-date": "2024-01-01",
                            "security.scrollintel.com/policy-version": "v1.0"
                        }
                    },
                    "spec": {
                        "serviceAccountName": f"{app_name}-sa",
                        "securityContext": security_config["securityContext"],
                        "containers": [
                            {
                                "name": app_name,
                                "image": image,
                                "securityContext": security_config["containerSecurityContext"],
                                "ports": [
                                    {
                                        "containerPort": 8000,
                                        "protocol": "TCP"
                                    }
                                ],
                                "env": [
                                    {
                                        "name": "SECURITY_PROFILE",
                                        "value": security_profile
                                    }
                                ],
                                "volumeMounts": [
                                    {
                                        "name": "tmp",
                                        "mountPath": "/tmp"
                                    },
                                    {
                                        "name": "cache",
                                        "mountPath": "/app/cache"
                                    }
                                ],
                                "resources": {
                                    "requests": {
                                        "memory": "256Mi",
                                        "cpu": "250m"
                                    },
                                    "limits": {
                                        "memory": "512Mi",
                                        "cpu": "500m"
                                    }
                                },
                                "livenessProbe": {
                                    "httpGet": {
                                        "path": "/health",
                                        "port": 8000
                                    },
                                    "initialDelaySeconds": 30,
                                    "periodSeconds": 10
                                },
                                "readinessProbe": {
                                    "httpGet": {
                                        "path": "/ready",
                                        "port": 8000
                                    },
                                    "initialDelaySeconds": 5,
                                    "periodSeconds": 5
                                }
                            }
                        ],
                        "volumes": [
                            {
                                "name": "tmp",
                                "emptyDir": {}
                            },
                            {
                                "name": "cache",
                                "emptyDir": {}
                            }
                        ]
                    }
                }
            }
        }
        
        return deployment
    
    def validate_pod_security(self, pod_spec: Dict[str, Any]) -> List[str]:
        """Validate pod specification against security policies"""
        violations = []
        
        # Check security context
        security_context = pod_spec.get("securityContext", {})
        
        if not security_context.get("runAsNonRoot", False):
            violations.append("Pod must run as non-root user")
        
        if security_context.get("runAsUser") == 0:
            violations.append("Pod cannot run as root user (UID 0)")
        
        # Check containers
        containers = pod_spec.get("containers", [])
        for i, container in enumerate(containers):
            container_security = container.get("securityContext", {})
            
            if container_security.get("privileged", False):
                violations.append(f"Container {i} cannot run in privileged mode")
            
            if container_security.get("allowPrivilegeEscalation", True):
                violations.append(f"Container {i} must disable privilege escalation")
            
            if not container_security.get("readOnlyRootFilesystem", False):
                violations.append(f"Container {i} should use read-only root filesystem")
            
            capabilities = container_security.get("capabilities", {})
            if "ALL" not in capabilities.get("drop", []):
                violations.append(f"Container {i} must drop ALL capabilities")
        
        # Check volumes
        volumes = pod_spec.get("volumes", [])
        allowed_volume_types = ["configMap", "downwardAPI", "emptyDir", 
                               "persistentVolumeClaim", "projected", "secret"]
        
        for volume in volumes:
            volume_type = next((k for k in volume.keys() if k != "name"), None)
            if volume_type not in allowed_volume_types:
                violations.append(f"Volume type '{volume_type}' is not allowed")
        
        return violations
    
    def generate_admission_controller_webhook(self) -> Dict[str, Any]:
        """Generate admission controller webhook for policy enforcement"""
        webhook = {
            "apiVersion": "admissionregistration.k8s.io/v1",
            "kind": "ValidatingAdmissionWebhook",
            "metadata": {
                "name": "scrollintel-security-policy-webhook"
            },
            "webhooks": [
                {
                    "name": "pod-security-policy.scrollintel.com",
                    "clientConfig": {
                        "service": {
                            "name": "scrollintel-admission-webhook",
                            "namespace": "scrollintel-security",
                            "path": "/validate-pod-security"
                        }
                    },
                    "rules": [
                        {
                            "operations": ["CREATE", "UPDATE"],
                            "apiGroups": [""],
                            "apiVersions": ["v1"],
                            "resources": ["pods"]
                        }
                    ],
                    "admissionReviewVersions": ["v1", "v1beta1"],
                    "sideEffects": "None",
                    "failurePolicy": "Fail"
                }
            ]
        }
        
        return webhook
    
    def export_policies_to_yaml(self, output_dir: str):
        """Export all security policies to YAML files"""
        import os
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Export Pod Security Policies
        for policy_name in self.policies:
            psp = self.generate_pod_security_policy(policy_name)
            with open(f"{output_dir}/psp-{policy_name}.yaml", "w") as f:
                yaml.dump(psp, f, default_flow_style=False)
        
        # Export Security Context Constraints
        for policy_name in self.policies:
            scc = self.generate_security_context_constraints(policy_name)
            with open(f"{output_dir}/scc-{policy_name}.yaml", "w") as f:
                yaml.dump(scc, f, default_flow_style=False)
        
        # Export sample secure deployments
        for profile in self.security_contexts:
            deployment = self.generate_deployment_with_security(
                f"sample-{profile}", f"scrollintel/{profile}:latest", profile
            )
            with open(f"{output_dir}/deployment-{profile}.yaml", "w") as f:
                yaml.dump(deployment, f, default_flow_style=False)
        
        # Export admission controller webhook
        webhook = self.generate_admission_controller_webhook()
        with open(f"{output_dir}/admission-webhook.yaml", "w") as f:
            yaml.dump(webhook, f, default_flow_style=False)
        
        logger.info(f"Security policies exported to {output_dir}")

# Example usage
if __name__ == "__main__":
    manager = ContainerSecurityPolicyManager()
    
    # Generate and export all policies
    manager.export_policies_to_yaml("./security/kubernetes/policies")
    
    # Validate a sample pod spec
    sample_pod = {
        "securityContext": {
            "runAsNonRoot": True,
            "runAsUser": 1000
        },
        "containers": [
            {
                "name": "app",
                "securityContext": {
                    "allowPrivilegeEscalation": False,
                    "readOnlyRootFilesystem": True,
                    "capabilities": {
                        "drop": ["ALL"]
                    }
                }
            }
        ]
    }
    
    violations = manager.validate_pod_security(sample_pod)
    if violations:
        print("Security violations found:")
        for violation in violations:
            print(f"- {violation}")
    else:
        print("Pod specification is secure")