#!/usr/bin/env python3
"""
Infrastructure Security Foundation Deployment Script
Deploys zero-trust network architecture, container security policies, mTLS, and security scanning
"""

import os
import sys
import yaml
import subprocess
import logging
from pathlib import Path
from typing import Dict, List, Any

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SecurityInfrastructureDeployer:
    """Deploys enterprise security infrastructure"""
    
    def __init__(self, config_path: str = "security/config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self.security_dir = Path("security")
        
    def _load_config(self) -> Dict[str, Any]:
        """Load deployment configuration"""
        default_config = {
            "cluster": {
                "name": "scrollintel-cluster",
                "namespace": "scrollintel"
            },
            "zero_trust": {
                "enabled": True,
                "jwt_secret": "your-jwt-secret-here"
            },
            "mtls": {
                "enabled": True,
                "ca_cert_path": "./certs/ca.crt",
                "ca_key_path": "./certs/ca.key"
            },
            "container_security": {
                "policy_level": "restricted",
                "enforce_pod_security": True
            },
            "security_scanning": {
                "terraform_enabled": True,
                "helm_enabled": True,
                "external_scanners": ["checkov", "trivy"]
            }
        }
        
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                user_config = yaml.safe_load(f)
                default_config.update(user_config)
        
        return default_config
    
    def deploy_network_policies(self) -> bool:
        """Deploy Kubernetes NetworkPolicies for zero-trust"""
        logger.info("Deploying Kubernetes NetworkPolicies...")
        
        try:
            # Apply network policies
            network_policies_path = self.security_dir / "kubernetes" / "network_policies.yaml"
            result = subprocess.run([
                "kubectl", "apply", "-f", str(network_policies_path)
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("NetworkPolicies deployed successfully")
                return True
            else:
                logger.error(f"Failed to deploy NetworkPolicies: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error deploying NetworkPolicies: {e}")
            return False
    
    def deploy_pod_security_policies(self) -> bool:
        """Deploy Pod Security Policies and Security Context Constraints"""
        logger.info("Deploying Pod Security Policies...")
        
        try:
            # Generate and apply security policies
            from container.security_policies import ContainerSecurityPolicyManager
            
            policy_manager = ContainerSecurityPolicyManager()
            
            # Export policies to temporary directory
            temp_dir = "/tmp/security-policies"
            policy_manager.export_policies_to_yaml(temp_dir)
            
            # Apply all policy files
            for policy_file in os.listdir(temp_dir):
                if policy_file.endswith('.yaml'):
                    file_path = os.path.join(temp_dir, policy_file)
                    result = subprocess.run([
                        "kubectl", "apply", "-f", file_path
                    ], capture_output=True, text=True)
                    
                    if result.returncode != 0:
                        logger.warning(f"Failed to apply {policy_file}: {result.stderr}")
            
            logger.info("Pod Security Policies deployed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error deploying Pod Security Policies: {e}")
            return False
    
    def deploy_istio_service_mesh(self) -> bool:
        """Deploy Istio service mesh with mTLS"""
        logger.info("Deploying Istio service mesh...")
        
        try:
            # Check if Istio is installed
            result = subprocess.run([
                "kubectl", "get", "namespace", "istio-system"
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.info("Installing Istio...")
                # Install Istio (simplified - in production use proper Istio installation)
                subprocess.run(["istioctl", "install", "--set", "values.defaultRevision=default", "-y"])
            
            # Apply Istio configurations
            istio_config_path = self.security_dir / "istio" / "service_mesh_config.yaml"
            result = subprocess.run([
                "kubectl", "apply", "-f", str(istio_config_path)
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("Istio service mesh deployed successfully")
                return True
            else:
                logger.error(f"Failed to deploy Istio configuration: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error deploying Istio service mesh: {e}")
            return False
    
    def setup_mtls_certificates(self) -> bool:
        """Setup mTLS certificates for services"""
        logger.info("Setting up mTLS certificates...")
        
        try:
            from mtls.certificate_manager import CertificateManager, CertificateAuthority, ServiceIdentity
            
            # Initialize CA and certificate manager
            ca = CertificateAuthority(
                self.config["mtls"]["ca_cert_path"],
                self.config["mtls"]["ca_key_path"]
            )
            cert_manager = CertificateManager(ca, "./certs/services")
            
            # Create certificates for core services
            services = [
                ServiceIdentity("scrollintel-api", "scrollintel", "production", "prod"),
                ServiceIdentity("scrollintel-frontend", "scrollintel", "production", "prod"),
                ServiceIdentity("scrollintel-database", "scrollintel", "production", "prod")
            ]
            
            for service in services:
                cert_path, key_path = cert_manager.get_or_create_certificate(service)
                logger.info(f"Certificate created for {service.service_name}: {cert_path}")
            
            logger.info("mTLS certificates setup completed")
            return True
            
        except Exception as e:
            logger.error(f"Error setting up mTLS certificates: {e}")
            return False
    
    def run_security_scans(self) -> bool:
        """Run infrastructure security scans"""
        logger.info("Running infrastructure security scans...")
        
        try:
            from infrastructure.terraform_security_scanner import InfrastructureSecurityScanner
            
            scanner = InfrastructureSecurityScanner()
            
            # Scan infrastructure configurations
            results = scanner.scan_infrastructure("./infrastructure")
            
            # Generate report
            report = scanner.generate_report(results)
            
            # Save report
            report_path = f"security_scan_report_{int(time.time())}.md"
            with open(report_path, 'w') as f:
                f.write(report)
            
            logger.info(f"Security scan completed. Report saved to {report_path}")
            
            # Check for critical findings
            critical_findings = 0
            for result in results.values():
                critical_findings += len([f for f in result.findings 
                                        if f.severity.value == "critical"])
            
            if critical_findings > 0:
                logger.warning(f"Found {critical_findings} critical security findings!")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error running security scans: {e}")
            return False
    
    def deploy_zero_trust_gateway(self) -> bool:
        """Deploy zero-trust gateway service"""
        logger.info("Deploying zero-trust gateway...")
        
        try:
            # Create zero-trust gateway deployment
            gateway_deployment = {
                "apiVersion": "apps/v1",
                "kind": "Deployment",
                "metadata": {
                    "name": "zero-trust-gateway",
                    "namespace": self.config["cluster"]["namespace"]
                },
                "spec": {
                    "replicas": 3,
                    "selector": {
                        "matchLabels": {
                            "app": "zero-trust-gateway"
                        }
                    },
                    "template": {
                        "metadata": {
                            "labels": {
                                "app": "zero-trust-gateway"
                            }
                        },
                        "spec": {
                            "containers": [
                                {
                                    "name": "gateway",
                                    "image": "scrollintel/zero-trust-gateway:latest",
                                    "ports": [
                                        {
                                            "containerPort": 8080
                                        }
                                    ],
                                    "env": [
                                        {
                                            "name": "JWT_SECRET",
                                            "valueFrom": {
                                                "secretKeyRef": {
                                                    "name": "zero-trust-secrets",
                                                    "key": "jwt-secret"
                                                }
                                            }
                                        }
                                    ]
                                }
                            ]
                        }
                    }
                }
            }
            
            # Save and apply deployment
            with open("/tmp/zero-trust-gateway.yaml", "w") as f:
                yaml.dump(gateway_deployment, f)
            
            result = subprocess.run([
                "kubectl", "apply", "-f", "/tmp/zero-trust-gateway.yaml"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("Zero-trust gateway deployed successfully")
                return True
            else:
                logger.error(f"Failed to deploy zero-trust gateway: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error deploying zero-trust gateway: {e}")
            return False
    
    def validate_deployment(self) -> bool:
        """Validate security infrastructure deployment"""
        logger.info("Validating security infrastructure deployment...")
        
        validations = []
        
        # Check NetworkPolicies
        result = subprocess.run([
            "kubectl", "get", "networkpolicies", "-n", self.config["cluster"]["namespace"]
        ], capture_output=True, text=True)
        validations.append(("NetworkPolicies", result.returncode == 0))
        
        # Check PodSecurityPolicies
        result = subprocess.run([
            "kubectl", "get", "podsecuritypolicies"
        ], capture_output=True, text=True)
        validations.append(("PodSecurityPolicies", result.returncode == 0))
        
        # Check Istio installation
        result = subprocess.run([
            "kubectl", "get", "pods", "-n", "istio-system"
        ], capture_output=True, text=True)
        validations.append(("Istio", result.returncode == 0))
        
        # Check certificates
        cert_dir = Path("./certs/services")
        cert_files = list(cert_dir.glob("*.crt")) if cert_dir.exists() else []
        validations.append(("mTLS Certificates", len(cert_files) > 0))
        
        # Report validation results
        all_passed = True
        for component, passed in validations:
            status = "✓ PASS" if passed else "✗ FAIL"
            logger.info(f"{component}: {status}")
            if not passed:
                all_passed = False
        
        return all_passed
    
    def deploy_all(self) -> bool:
        """Deploy complete security infrastructure"""
        logger.info("Starting complete security infrastructure deployment...")
        
        deployment_steps = [
            ("Network Policies", self.deploy_network_policies),
            ("Pod Security Policies", self.deploy_pod_security_policies),
            ("Istio Service Mesh", self.deploy_istio_service_mesh),
            ("mTLS Certificates", self.setup_mtls_certificates),
            ("Zero Trust Gateway", self.deploy_zero_trust_gateway),
            ("Security Scans", self.run_security_scans)
        ]
        
        failed_steps = []
        
        for step_name, step_func in deployment_steps:
            logger.info(f"Executing: {step_name}")
            try:
                if not step_func():
                    failed_steps.append(step_name)
                    logger.error(f"Failed: {step_name}")
                else:
                    logger.info(f"Completed: {step_name}")
            except Exception as e:
                logger.error(f"Error in {step_name}: {e}")
                failed_steps.append(step_name)
        
        # Validate deployment
        if not failed_steps:
            logger.info("Running deployment validation...")
            if self.validate_deployment():
                logger.info("🎉 Security infrastructure deployment completed successfully!")
                return True
            else:
                logger.error("Deployment validation failed")
                return False
        else:
            logger.error(f"Deployment failed. Failed steps: {', '.join(failed_steps)}")
            return False

def main():
    """Main deployment function"""
    import argparse
    import time
    
    parser = argparse.ArgumentParser(description="Deploy ScrollIntel Security Infrastructure")
    parser.add_argument("--config", default="security/config.yaml", 
                       help="Configuration file path")
    parser.add_argument("--validate-only", action="store_true",
                       help="Only validate existing deployment")
    parser.add_argument("--component", choices=[
        "network-policies", "pod-security", "istio", "mtls", "gateway", "scans"
    ], help="Deploy specific component only")
    
    args = parser.parse_args()
    
    deployer = SecurityInfrastructureDeployer(args.config)
    
    if args.validate_only:
        success = deployer.validate_deployment()
    elif args.component:
        component_map = {
            "network-policies": deployer.deploy_network_policies,
            "pod-security": deployer.deploy_pod_security_policies,
            "istio": deployer.deploy_istio_service_mesh,
            "mtls": deployer.setup_mtls_certificates,
            "gateway": deployer.deploy_zero_trust_gateway,
            "scans": deployer.run_security_scans
        }
        success = component_map[args.component]()
    else:
        success = deployer.deploy_all()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()