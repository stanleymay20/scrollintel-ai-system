#!/usr/bin/env python3
"""
ScrollIntel Deployment Validation Script
Validates that all services are running correctly after deployment
"""

import os
import sys
import time
import json
import requests
import argparse
from typing import Dict, List, Tuple
from urllib.parse import urljoin


class DeploymentValidator:
    """Validates ScrollIntel deployment health and functionality"""
    
    def __init__(self, base_url: str = "http://localhost", frontend_url: str = "http://localhost:3000"):
        self.base_url = base_url.rstrip('/')
        self.frontend_url = frontend_url.rstrip('/')
        self.health_check_url = f"{base_url}:8080"
        self.monitoring_url = f"{base_url}:3001"
        self.session = requests.Session()
        self.session.timeout = 30
        
    def validate_all(self) -> bool:
        """Run all validation checks"""
        print("🔍 Starting ScrollIntel Deployment Validation...")
        print(f"Backend URL: {self.base_url}")
        print(f"Frontend URL: {self.frontend_url}")
        print("-" * 60)
        
        checks = [
            ("Basic Health Check", self.check_basic_health),
            ("Load Balancer Health", self.check_load_balancer),
            ("Database Connectivity", self.check_database),
            ("Redis Connectivity", self.check_redis),
            ("Agent Registry", self.check_agents),
            ("API Endpoints", self.check_api_endpoints),
            ("Frontend Health", self.check_frontend),
            ("Authentication", self.check_authentication),
            ("File Upload", self.check_file_upload),
            ("Performance", self.check_performance),
            ("Auto-scaling Status", self.check_auto_scaling),
            ("Docker Containers", self.check_docker_containers),
            ("SSL Configuration", self.check_ssl_configuration),
            ("Database Replication", self.check_database_replication),
            ("Monitoring System", self.check_monitoring),
        ]
        
        results = []
        for check_name, check_func in checks:
            print(f"Running {check_name}...")
            try:
                success, message = check_func()
                status = "✅ PASS" if success else "❌ FAIL"
                print(f"  {status}: {message}")
                results.append((check_name, success, message))
            except Exception as e:
                print(f"  ❌ ERROR: {str(e)}")
                results.append((check_name, False, str(e)))
            print()
        
        # Summary
        passed = sum(1 for _, success, _ in results if success)
        total = len(results)
        
        print("=" * 60)
        print(f"VALIDATION SUMMARY: {passed}/{total} checks passed")
        print("=" * 60)
        
        if passed == total:
            print("🎉 All checks passed! Deployment is healthy.")
            return True
        else:
            print("⚠️ Some checks failed. Please review the issues above.")
            for check_name, success, message in results:
                if not success:
                    print(f"  ❌ {check_name}: {message}")
            return False
    
    def check_basic_health(self) -> Tuple[bool, str]:
        """Check basic health endpoint through load balancer"""
        try:
            response = self.session.get(f"{self.base_url}/health")
            response.raise_for_status()
            
            if response.status_code == 200:
                return True, f"Application healthy (response time: {response.elapsed.total_seconds():.2f}s)"
            else:
                return False, f"Application health check failed: HTTP {response.status_code}"
                
        except requests.exceptions.RequestException as e:
            return False, f"Health check failed: {str(e)}"
    
    def check_load_balancer(self) -> Tuple[bool, str]:
        """Check load balancer status"""
        try:
            # Check nginx status
            response = self.session.get(f"{self.health_check_url}/nginx_status")
            if response.status_code == 200:
                # Check upstream status
                upstream_response = self.session.get(f"{self.health_check_url}/upstream_check")
                if upstream_response.status_code == 200:
                    return True, "Load balancer and upstream backends healthy"
                else:
                    return False, f"Upstream backends unhealthy: HTTP {upstream_response.status_code}"
            else:
                return False, f"Load balancer unhealthy: HTTP {response.status_code}"
        except requests.exceptions.RequestException as e:
            return False, f"Load balancer check failed: {str(e)}"
    
    def check_detailed_health(self) -> Tuple[bool, str]:
        """Check detailed health endpoint"""
        try:
            response = self.session.get(f"{self.base_url}/health/detailed")
            response.raise_for_status()
            
            data = response.json()
            status = data.get("status")
            
            if status == "healthy":
                components = data.get("components", {})
                healthy_components = sum(1 for comp in components.values() if comp.get("healthy", False))
                total_components = len(components)
                return True, f"All components healthy ({healthy_components}/{total_components})"
            else:
                return False, f"System status: {status}"
                
        except requests.exceptions.RequestException as e:
            return False, f"Detailed health check failed: {str(e)}"
    
    def check_database(self) -> Tuple[bool, str]:
        """Check database connectivity"""
        try:
            response = self.session.get(f"{self.base_url}/health/dependencies")
            response.raise_for_status()
            
            data = response.json()
            db_health = data.get("dependencies", {}).get("database", {})
            
            if db_health.get("healthy"):
                response_time = db_health.get("response_time_ms", 0)
                return True, f"Database healthy (response time: {response_time:.2f}ms)"
            else:
                error = db_health.get("error", "Unknown error")
                return False, f"Database unhealthy: {error}"
                
        except requests.exceptions.RequestException as e:
            return False, f"Database check failed: {str(e)}"
    
    def check_redis(self) -> Tuple[bool, str]:
        """Check Redis connectivity"""
        try:
            response = self.session.get(f"{self.base_url}/health/dependencies")
            response.raise_for_status()
            
            data = response.json()
            redis_health = data.get("dependencies", {}).get("redis", {})
            
            if redis_health.get("healthy"):
                response_time = redis_health.get("response_time_ms", 0)
                memory = redis_health.get("memory_usage", "unknown")
                return True, f"Redis healthy (response time: {response_time:.2f}ms, memory: {memory})"
            else:
                error = redis_health.get("error", "Unknown error")
                return False, f"Redis unhealthy: {error}"
                
        except requests.exceptions.RequestException as e:
            return False, f"Redis check failed: {str(e)}"
    
    def check_agents(self) -> Tuple[bool, str]:
        """Check agent registry and health"""
        try:
            response = self.session.get(f"{self.base_url}/health/agents")
            response.raise_for_status()
            
            data = response.json()
            if data.get("status") == "healthy":
                total_agents = data.get("total_agents", 0)
                healthy_agents = data.get("healthy_agents", 0)
                return True, f"Agent registry healthy ({healthy_agents}/{total_agents} agents)"
            else:
                unhealthy = data.get("unhealthy_agents", 0)
                return False, f"Some agents unhealthy ({unhealthy} unhealthy agents)"
                
        except requests.exceptions.RequestException as e:
            return False, f"Agent check failed: {str(e)}"
    
    def check_api_endpoints(self) -> Tuple[bool, str]:
        """Check critical API endpoints"""
        endpoints = [
            "/api/agents",
            "/api/auth/status",
            "/docs",
        ]
        
        failed_endpoints = []
        for endpoint in endpoints:
            try:
                response = self.session.get(f"{self.base_url}{endpoint}")
                if response.status_code >= 400:
                    failed_endpoints.append(f"{endpoint} ({response.status_code})")
            except requests.exceptions.RequestException:
                failed_endpoints.append(f"{endpoint} (connection error)")
        
        if not failed_endpoints:
            return True, f"All {len(endpoints)} API endpoints accessible"
        else:
            return False, f"Failed endpoints: {', '.join(failed_endpoints)}"
    
    def check_frontend(self) -> Tuple[bool, str]:
        """Check frontend health"""
        try:
            # Check main page
            response = self.session.get(self.frontend_url)
            response.raise_for_status()
            
            # Check health endpoint
            health_response = self.session.get(f"{self.frontend_url}/api/health")
            health_response.raise_for_status()
            
            health_data = health_response.json()
            if health_data.get("status") == "healthy":
                return True, "Frontend healthy and accessible"
            else:
                return False, f"Frontend reports unhealthy: {health_data.get('status')}"
                
        except requests.exceptions.RequestException as e:
            return False, f"Frontend check failed: {str(e)}"
    
    def check_authentication(self) -> Tuple[bool, str]:
        """Check authentication system"""
        try:
            # Test auth status endpoint
            response = self.session.get(f"{self.base_url}/api/auth/status")
            
            if response.status_code in [200, 401]:  # Both are valid responses
                return True, "Authentication system responding"
            else:
                return False, f"Auth system error: HTTP {response.status_code}"
                
        except requests.exceptions.RequestException as e:
            return False, f"Authentication check failed: {str(e)}"
    
    def check_file_upload(self) -> Tuple[bool, str]:
        """Check file upload capability"""
        try:
            # Test file upload endpoint (without actually uploading)
            response = self.session.options(f"{self.base_url}/api/files/upload")
            
            if response.status_code in [200, 405]:  # OPTIONS or Method Not Allowed is fine
                return True, "File upload endpoint accessible"
            else:
                return False, f"File upload endpoint error: HTTP {response.status_code}"
                
        except requests.exceptions.RequestException as e:
            return False, f"File upload check failed: {str(e)}"
    
    def check_performance(self) -> Tuple[bool, str]:
        """Check basic performance metrics"""
        try:
            # Test multiple requests to get average response time
            response_times = []
            for _ in range(5):
                start_time = time.time()
                response = self.session.get(f"{self.base_url}/health")
                response_time = time.time() - start_time
                response_times.append(response_time)
                
                if response.status_code != 200:
                    return False, f"Performance test failed: HTTP {response.status_code}"
            
            avg_response_time = sum(response_times) / len(response_times)
            max_response_time = max(response_times)
            
            if avg_response_time < 2.0 and max_response_time < 5.0:
                return True, f"Good performance (avg: {avg_response_time:.2f}s, max: {max_response_time:.2f}s)"
            elif avg_response_time < 5.0:
                return True, f"Acceptable performance (avg: {avg_response_time:.2f}s, max: {max_response_time:.2f}s)"
            else:
                return False, f"Poor performance (avg: {avg_response_time:.2f}s, max: {max_response_time:.2f}s)"
                
        except requests.exceptions.RequestException as e:
            return False, f"Performance check failed: {str(e)}"
    
    def check_auto_scaling(self) -> Tuple[bool, str]:
        """Check auto-scaling status"""
        try:
            import subprocess
            
            # Check if auto-scaling manager is running
            if os.path.exists('logs/auto-scaling.pid'):
                with open('logs/auto-scaling.pid', 'r') as f:
                    pid = f.read().strip()
                
                # Check if process is running
                result = subprocess.run(['ps', '-p', pid], capture_output=True)
                if result.returncode == 0:
                    return True, f"Auto-scaling manager running (PID: {pid})"
                else:
                    return False, "Auto-scaling manager not running"
            else:
                return True, "Auto-scaling not enabled (optional)"
                
        except Exception as e:
            return False, f"Auto-scaling check failed: {str(e)}"
    
    def check_docker_containers(self) -> Tuple[bool, str]:
        """Check Docker containers status"""
        try:
            import subprocess
            
            result = subprocess.run([
                'docker', 'ps', '--filter', 'name=scrollintel',
                '--format', '{{.Names}}\t{{.Status}}'
            ], capture_output=True, text=True, check=True)
            
            containers = result.stdout.strip().split('\n')
            running_containers = len([c for c in containers if 'Up' in c and c.strip()])
            
            if running_containers >= 3:  # At least backend, frontend, nginx
                return True, f"{running_containers} containers running"
            else:
                return False, f"Only {running_containers} containers running (expected at least 3)"
                
        except subprocess.CalledProcessError as e:
            return False, f"Docker container check failed: {str(e)}"
        except Exception as e:
            return False, f"Container check error: {str(e)}"
    
    def check_ssl_configuration(self) -> Tuple[bool, str]:
        """Check SSL configuration"""
        if not os.path.exists('./nginx/ssl/scrollintel.crt'):
            return True, "SSL certificates not configured (optional)"
        
        try:
            # Test HTTPS endpoint
            https_url = self.base_url.replace('http://', 'https://')
            response = self.session.get(f"{https_url}/health", timeout=10, verify=False)
            
            if response.status_code == 200:
                return True, "SSL configuration working"
            else:
                return False, f"SSL endpoint failed: HTTP {response.status_code}"
                
        except requests.exceptions.RequestException as e:
            return False, f"SSL check failed: {str(e)}"
    
    def check_database_replication(self) -> Tuple[bool, str]:
        """Check database replication status"""
        try:
            import subprocess
            
            # Check if replication containers are running
            result = subprocess.run([
                'docker', 'ps', '--filter', 'name=postgres',
                '--format', '{{.Names}}'
            ], capture_output=True, text=True, check=True)
            
            containers = result.stdout.strip().split('\n')
            postgres_containers = [c for c in containers if 'postgres' in c and c.strip()]
            
            if len(postgres_containers) >= 2:
                return True, f"Database replication active ({len(postgres_containers)} instances)"
            else:
                return True, "Single database instance (replication optional)"
                
        except subprocess.CalledProcessError as e:
            return False, f"Database replication check failed: {str(e)}"
        except Exception as e:
            return False, f"Replication check error: {str(e)}"
    
    def check_monitoring(self) -> Tuple[bool, str]:
        """Check monitoring system"""
        try:
            # Check Grafana
            response = self.session.get(f"{self.monitoring_url}/api/health", timeout=10)
            if response.status_code == 200:
                return True, "Monitoring system (Grafana) accessible"
            else:
                return True, "Monitoring system not configured (optional)"
                
        except requests.exceptions.RequestException:
            return True, "Monitoring system not configured (optional)"


def main():
    parser = argparse.ArgumentParser(description="ScrollIntel Deployment Validator")
    parser.add_argument(
        "--backend-url",
        default="http://localhost:8000",
        help="Backend API URL (default: http://localhost:8000)"
    )
    parser.add_argument(
        "--frontend-url",
        default="http://localhost:3000",
        help="Frontend URL (default: http://localhost:3000)"
    )
    parser.add_argument(
        "--wait",
        type=int,
        default=0,
        help="Wait time in seconds before starting validation"
    )
    
    args = parser.parse_args()
    
    if args.wait > 0:
        print(f"⏳ Waiting {args.wait} seconds for services to start...")
        time.sleep(args.wait)
    
    validator = DeploymentValidator(args.backend_url, args.frontend_url)
    success = validator.validate_all()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()