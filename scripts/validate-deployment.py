#!/usr/bin/env python3
"""
Deployment Validation Script for Agent Steering System
Comprehensive validation of deployment status and configuration
"""

import asyncio
import subprocess
import json
import logging
import sys
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ValidationStatus(Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    WARNING = "WARNING"

@dataclass
class ValidationResult:
    check_name: str
    status: ValidationStatus
    message: str
    details: Dict[str, Any]

class DeploymentValidator:
    def __init__(self, namespace: str = "agent-steering-system"):
        self.namespace = namespace
        self.results: List[ValidationResult] = []
    
    def run_kubectl_command(self, args: List[str]) -> Tuple[str, str, int]:
        """Run kubectl command and return stdout, stderr, and return code"""
        try:
            cmd = ["kubectl"] + args
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
            return result.stdout, result.stderr, result.returncode
        except subprocess.TimeoutExpired:
            return "", "Command timed out", 1
        except Exception as e:
            return "", str(e), 1
    
    def add_result(self, check_name: str, status: ValidationStatus, message: str, details: Dict[str, Any] = None):
        """Add a validation result"""
        self.results.append(ValidationResult(
            check_name=check_name,
            status=status,
            message=message,
            details=details or {}
        ))
    
    def validate_namespace(self) -> bool:
        """Validate that the namespace exists and is active"""
        stdout, stderr, returncode = self.run_kubectl_command([
            "get", "namespace", self.namespace, "-o", "json"
        ])
        
        if returncode != 0:
            self.add_result(
                "Namespace Existence",
                ValidationStatus.FAIL,
                f"Namespace {self.namespace} does not exist",
                {"error": stderr}
            )
            return False
        
        try:
            namespace_data = json.loads(stdout)
            phase = namespace_data.get("status", {}).get("phase", "Unknown")
            
            if phase == "Active":
                self.add_result(
                    "Namespace Existence",
                    ValidationStatus.PASS,
                    f"Namespace {self.namespace} is active",
                    {"phase": phase}
                )
                return True
            else:
                self.add_result(
                    "Namespace Existence",
                    ValidationStatus.FAIL,
                    f"Namespace {self.namespace} is not active (phase: {phase})",
                    {"phase": phase}
                )
                return False
        
        except json.JSONDecodeError:
            self.add_result(
                "Namespace Existence",
                ValidationStatus.FAIL,
                "Failed to parse namespace information",
                {"stdout": stdout}
            )
            return False
    
    def validate_deployments(self) -> bool:
        """Validate that all required deployments are running"""
        required_deployments = [
            "orchestration-engine",
            "intelligence-engine",
            "prometheus",
            "grafana"
        ]
        
        stdout, stderr, returncode = self.run_kubectl_command([
            "get", "deployments", "-n", self.namespace, "-o", "json"
        ])
        
        if returncode != 0:
            self.add_result(
                "Deployments Status",
                ValidationStatus.FAIL,
                "Failed to get deployment information",
                {"error": stderr}
            )
            return False
        
        try:
            deployments_data = json.loads(stdout)
            deployments = deployments_data.get("items", [])
            
            deployment_status = {}
            for deployment in deployments:
                name = deployment.get("metadata", {}).get("name")
                status = deployment.get("status", {})
                
                ready_replicas = status.get("readyReplicas", 0)
                desired_replicas = status.get("replicas", 0)
                
                deployment_status[name] = {
                    "ready": ready_replicas,
                    "desired": desired_replicas,
                    "healthy": ready_replicas == desired_replicas and desired_replicas > 0
                }
            
            all_healthy = True
            missing_deployments = []
            
            for required_deployment in required_deployments:
                if required_deployment not in deployment_status:
                    missing_deployments.append(required_deployment)
                    all_healthy = False
                elif not deployment_status[required_deployment]["healthy"]:
                    all_healthy = False
            
            if missing_deployments:
                self.add_result(
                    "Deployments Status",
                    ValidationStatus.FAIL,
                    f"Missing deployments: {', '.join(missing_deployments)}",
                    {"missing": missing_deployments, "status": deployment_status}
                )
                return False
            
            if all_healthy:
                self.add_result(
                    "Deployments Status",
                    ValidationStatus.PASS,
                    "All required deployments are healthy",
                    {"status": deployment_status}
                )
                return True
            else:
                unhealthy = [name for name, status in deployment_status.items() 
                           if not status["healthy"]]
                self.add_result(
                    "Deployments Status",
                    ValidationStatus.FAIL,
                    f"Unhealthy deployments: {', '.join(unhealthy)}",
                    {"status": deployment_status}
                )
                return False
        
        except json.JSONDecodeError:
            self.add_result(
                "Deployments Status",
                ValidationStatus.FAIL,
                "Failed to parse deployment information",
                {"stdout": stdout}
            )
            return False
    
    def validate_services(self) -> bool:
        """Validate that all required services are available"""
        required_services = [
            "orchestration-service",
            "intelligence-service",
            "prometheus-service",
            "grafana-service",
            "postgresql-service",
            "redis-service"
        ]
        
        stdout, stderr, returncode = self.run_kubectl_command([
            "get", "services", "-n", self.namespace, "-o", "json"
        ])
        
        if returncode != 0:
            self.add_result(
                "Services Status",
                ValidationStatus.FAIL,
                "Failed to get service information",
                {"error": stderr}
            )
            return False
        
        try:
            services_data = json.loads(stdout)
            services = services_data.get("items", [])
            
            service_names = [service.get("metadata", {}).get("name") for service in services]
            missing_services = [svc for svc in required_services if svc not in service_names]
            
            if missing_services:
                self.add_result(
                    "Services Status",
                    ValidationStatus.FAIL,
                    f"Missing services: {', '.join(missing_services)}",
                    {"missing": missing_services, "available": service_names}
                )
                return False
            else:
                self.add_result(
                    "Services Status",
                    ValidationStatus.PASS,
                    "All required services are available",
                    {"services": service_names}
                )
                return True
        
        except json.JSONDecodeError:
            self.add_result(
                "Services Status",
                ValidationStatus.FAIL,
                "Failed to parse service information",
                {"stdout": stdout}
            )
            return False
    
    def validate_pods(self) -> bool:
        """Validate that all pods are running and ready"""
        stdout, stderr, returncode = self.run_kubectl_command([
            "get", "pods", "-n", self.namespace, "-o", "json"
        ])
        
        if returncode != 0:
            self.add_result(
                "Pods Status",
                ValidationStatus.FAIL,
                "Failed to get pod information",
                {"error": stderr}
            )
            return False
        
        try:
            pods_data = json.loads(stdout)
            pods = pods_data.get("items", [])
            
            pod_status = {}
            unhealthy_pods = []
            
            for pod in pods:
                name = pod.get("metadata", {}).get("name")
                status = pod.get("status", {})
                phase = status.get("phase", "Unknown")
                
                conditions = status.get("conditions", [])
                ready_condition = next((c for c in conditions if c.get("type") == "Ready"), {})
                is_ready = ready_condition.get("status") == "True"
                
                pod_status[name] = {
                    "phase": phase,
                    "ready": is_ready
                }
                
                if phase != "Running" or not is_ready:
                    unhealthy_pods.append(name)
            
            if unhealthy_pods:
                self.add_result(
                    "Pods Status",
                    ValidationStatus.WARNING if len(unhealthy_pods) < len(pods) / 2 else ValidationStatus.FAIL,
                    f"Unhealthy pods: {', '.join(unhealthy_pods)}",
                    {"unhealthy": unhealthy_pods, "status": pod_status}
                )
                return len(unhealthy_pods) < len(pods) / 2
            else:
                self.add_result(
                    "Pods Status",
                    ValidationStatus.PASS,
                    f"All {len(pods)} pods are running and ready",
                    {"status": pod_status}
                )
                return True
        
        except json.JSONDecodeError:
            self.add_result(
                "Pods Status",
                ValidationStatus.FAIL,
                "Failed to parse pod information",
                {"stdout": stdout}
            )
            return False
    
    def validate_persistent_volumes(self) -> bool:
        """Validate that persistent volumes are bound"""
        stdout, stderr, returncode = self.run_kubectl_command([
            "get", "pvc", "-n", self.namespace, "-o", "json"
        ])
        
        if returncode != 0:
            self.add_result(
                "Persistent Volumes",
                ValidationStatus.WARNING,
                "No persistent volume claims found or failed to retrieve",
                {"error": stderr}
            )
            return True  # Not critical for basic functionality
        
        try:
            pvc_data = json.loads(stdout)
            pvcs = pvc_data.get("items", [])
            
            if not pvcs:
                self.add_result(
                    "Persistent Volumes",
                    ValidationStatus.WARNING,
                    "No persistent volume claims found",
                    {}
                )
                return True
            
            unbound_pvcs = []
            for pvc in pvcs:
                name = pvc.get("metadata", {}).get("name")
                status = pvc.get("status", {}).get("phase", "Unknown")
                
                if status != "Bound":
                    unbound_pvcs.append(f"{name} ({status})")
            
            if unbound_pvcs:
                self.add_result(
                    "Persistent Volumes",
                    ValidationStatus.WARNING,
                    f"Unbound PVCs: {', '.join(unbound_pvcs)}",
                    {"unbound": unbound_pvcs}
                )
                return True  # Warning, not failure
            else:
                self.add_result(
                    "Persistent Volumes",
                    ValidationStatus.PASS,
                    f"All {len(pvcs)} persistent volume claims are bound",
                    {"count": len(pvcs)}
                )
                return True
        
        except json.JSONDecodeError:
            self.add_result(
                "Persistent Volumes",
                ValidationStatus.WARNING,
                "Failed to parse PVC information",
                {"stdout": stdout}
            )
            return True
    
    def validate_ingress(self) -> bool:
        """Validate that ingress is configured"""
        stdout, stderr, returncode = self.run_kubectl_command([
            "get", "ingress", "-n", self.namespace, "-o", "json"
        ])
        
        if returncode != 0:
            self.add_result(
                "Ingress Configuration",
                ValidationStatus.WARNING,
                "No ingress found or failed to retrieve",
                {"error": stderr}
            )
            return True  # Not critical for internal testing
        
        try:
            ingress_data = json.loads(stdout)
            ingresses = ingress_data.get("items", [])
            
            if not ingresses:
                self.add_result(
                    "Ingress Configuration",
                    ValidationStatus.WARNING,
                    "No ingress resources found",
                    {}
                )
                return True
            
            ingress_info = []
            for ingress in ingresses:
                name = ingress.get("metadata", {}).get("name")
                rules = ingress.get("spec", {}).get("rules", [])
                hosts = [rule.get("host") for rule in rules if rule.get("host")]
                
                ingress_info.append({
                    "name": name,
                    "hosts": hosts
                })
            
            self.add_result(
                "Ingress Configuration",
                ValidationStatus.PASS,
                f"Found {len(ingresses)} ingress resource(s)",
                {"ingresses": ingress_info}
            )
            return True
        
        except json.JSONDecodeError:
            self.add_result(
                "Ingress Configuration",
                ValidationStatus.WARNING,
                "Failed to parse ingress information",
                {"stdout": stdout}
            )
            return True
    
    def validate_resource_quotas(self) -> bool:
        """Validate resource usage and quotas"""
        # Check node resources
        stdout, stderr, returncode = self.run_kubectl_command([
            "top", "nodes"
        ])
        
        if returncode != 0:
            self.add_result(
                "Resource Usage",
                ValidationStatus.WARNING,
                "Failed to get node resource usage (metrics-server may not be available)",
                {"error": stderr}
            )
            return True  # Not critical
        
        # Check pod resources
        stdout, stderr, returncode = self.run_kubectl_command([
            "top", "pods", "-n", self.namespace
        ])
        
        if returncode != 0:
            self.add_result(
                "Resource Usage",
                ValidationStatus.WARNING,
                "Failed to get pod resource usage",
                {"error": stderr}
            )
            return True
        
        self.add_result(
            "Resource Usage",
            ValidationStatus.PASS,
            "Resource usage information available",
            {"metrics_available": True}
        )
        return True
    
    def run_all_validations(self) -> bool:
        """Run all validation checks"""
        logger.info(f"Starting deployment validation for namespace: {self.namespace}")
        
        validations = [
            ("Namespace", self.validate_namespace),
            ("Deployments", self.validate_deployments),
            ("Services", self.validate_services),
            ("Pods", self.validate_pods),
            ("Persistent Volumes", self.validate_persistent_volumes),
            ("Ingress", self.validate_ingress),
            ("Resources", self.validate_resource_quotas)
        ]
        
        overall_success = True
        
        for validation_name, validation_func in validations:
            logger.info(f"Running {validation_name} validation...")
            try:
                success = validation_func()
                if not success:
                    overall_success = False
            except Exception as e:
                logger.error(f"Validation {validation_name} failed with exception: {e}")
                self.add_result(
                    validation_name,
                    ValidationStatus.FAIL,
                    f"Validation failed with exception: {str(e)}",
                    {"exception": str(e)}
                )
                overall_success = False
        
        return overall_success
    
    def print_results(self) -> bool:
        """Print validation results and return overall success"""
        print(f"\n{'='*80}")
        print(f"Agent Steering System Deployment Validation Results")
        print(f"Namespace: {self.namespace}")
        print(f"{'='*80}")
        
        passed = sum(1 for r in self.results if r.status == ValidationStatus.PASS)
        failed = sum(1 for r in self.results if r.status == ValidationStatus.FAIL)
        warnings = sum(1 for r in self.results if r.status == ValidationStatus.WARNING)
        
        for result in self.results:
            status_color = {
                ValidationStatus.PASS: "\033[92m",     # Green
                ValidationStatus.FAIL: "\033[91m",     # Red
                ValidationStatus.WARNING: "\033[93m"   # Yellow
            }.get(result.status, "\033[0m")
            
            reset_color = "\033[0m"
            
            print(f"{result.check_name:<25} {status_color}{result.status.value:<8}{reset_color} {result.message}")
        
        print(f"\n{'='*80}")
        
        overall_success = failed == 0
        
        if overall_success:
            if warnings == 0:
                print(f"\033[92m✓ DEPLOYMENT VALIDATION PASSED\033[0m")
            else:
                print(f"\033[93m⚠ DEPLOYMENT VALIDATION PASSED WITH WARNINGS\033[0m")
        else:
            print(f"\033[91m✗ DEPLOYMENT VALIDATION FAILED\033[0m")
        
        print(f"Results: {passed} passed, {failed} failed, {warnings} warnings")
        print(f"{'='*80}\n")
        
        return overall_success

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Agent Steering System Deployment Validator")
    parser.add_argument(
        "--namespace", "-n",
        default="agent-steering-system",
        help="Kubernetes namespace to validate"
    )
    parser.add_argument(
        "--json", "-j",
        action="store_true",
        help="Output results in JSON format"
    )
    
    args = parser.parse_args()
    
    validator = DeploymentValidator(args.namespace)
    overall_success = validator.run_all_validations()
    
    if args.json:
        output = {
            "timestamp": time.time(),
            "namespace": args.namespace,
            "overall_success": overall_success,
            "summary": {
                "passed": sum(1 for r in validator.results if r.status == ValidationStatus.PASS),
                "failed": sum(1 for r in validator.results if r.status == ValidationStatus.FAIL),
                "warnings": sum(1 for r in validator.results if r.status == ValidationStatus.WARNING)
            },
            "results": [
                {
                    "check_name": r.check_name,
                    "status": r.status.value,
                    "message": r.message,
                    "details": r.details
                }
                for r in validator.results
            ]
        }
        print(json.dumps(output, indent=2))
    else:
        overall_success = validator.print_results()
    
    sys.exit(0 if overall_success else 1)

if __name__ == "__main__":
    main()