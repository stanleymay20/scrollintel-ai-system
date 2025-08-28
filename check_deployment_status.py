#!/usr/bin/env python3
"""
ScrollIntel Comprehensive Deployment Status Checker
Advanced monitoring and health checking for ScrollIntel AI Platform
"""

import requests
import time
import sys
import json
import os
import psutil
import subprocess
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import concurrent.futures
from pathlib import Path
import socket
import platform
import logging
from urllib.parse import urlparse

@dataclass
class ServiceStatus:
    name: str
    url: str
    status: str
    response_time: float
    details: Dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    category: str = "service"

class ScrollIntelDeploymentChecker:
    def __init__(self, timeout: int = 10, verbose: bool = False):
        self.timeout = timeout
        self.verbose = verbose
        self.results = []
        self.start_time = datetime.now()
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging configuration"""
        level = logging.DEBUG if self.verbose else logging.INFO
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler()]
        )
        self.logger = logging.getLogger(__name__)
        
    def check_port_connectivity(self, host: str, port: int, service_name: str) -> ServiceStatus:
        """Check if a port is accessible"""
        start_time = time.time()
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self.timeout)
            result = sock.connect_ex((host, port))
            sock.close()
            
            response_time = (time.time() - start_time) * 1000
            
            if result == 0:
                return ServiceStatus(
                    service_name, f"{host}:{port}", "✅ PORT OPEN", 
                    response_time, {"host": host, "port": port}, category="network"
                )
            else:
                return ServiceStatus(
                    service_name, f"{host}:{port}", "❌ PORT CLOSED", 
                    response_time, {"host": host, "port": port, "error_code": result}, category="network"
                )
        except Exception as e:
            return ServiceStatus(
                service_name, f"{host}:{port}", "❌ CONNECTION ERROR", 
                (time.time() - start_time) * 1000, {"error": str(e)}, category="network"
            )
    
    def check_process_running(self, process_name: str) -> ServiceStatus:
        """Check if a process is running"""
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                if process_name.lower() in proc.info['name'].lower():
                    return ServiceStatus(
                        f"Process: {process_name}", "local", "✅ RUNNING", 0,
                        {"pid": proc.info['pid'], "cmdline": ' '.join(proc.info['cmdline'][:3])}, 
                        category="process"
                    )
            return ServiceStatus(
                f"Process: {process_name}", "local", "❌ NOT RUNNING", 0,
                {"error": "Process not found"}, category="process"
            )
        except Exception as e:
            return ServiceStatus(
                f"Process: {process_name}", "local", "❌ ERROR", 0,
                {"error": str(e)}, category="process"
            )

    def check_service(self, url: str, service_name: str, expected_status: int = 200) -> ServiceStatus:
        """Enhanced service health check with detailed metrics"""
        start_time = time.time()
        try:
            response = requests.get(url, timeout=self.timeout)
            response_time = (time.time() - start_time) * 1000
            
            if response.status_code == expected_status:
                status = "✅ HEALTHY"
                details = {
                    "status_code": response.status_code,
                    "response_time_ms": round(response_time, 2),
                    "content_length": len(response.content) if response.content else 0
                }
                
                # Try to parse JSON response for additional details
                try:
                    if response.headers.get('content-type', '').startswith('application/json'):
                        json_data = response.json()
                        details["json_response"] = json_data
                except:
                    pass
                    
            else:
                status = f"⚠️  DEGRADED ({response.status_code})"
                details = {"status_code": response.status_code, "response_time_ms": round(response_time, 2)}
                
        except requests.exceptions.ConnectionError:
            status = "❌ DOWN (Connection Refused)"
            details = {"error": "Connection refused", "response_time_ms": round((time.time() - start_time) * 1000, 2)}
        except requests.exceptions.Timeout:
            status = f"⏰ TIMEOUT (>{self.timeout}s)"
            details = {"error": "Timeout", "response_time_ms": round((time.time() - start_time) * 1000, 2)}
        except Exception as e:
            status = f"❌ ERROR"
            details = {"error": str(e), "response_time_ms": round((time.time() - start_time) * 1000, 2)}
            
        return ServiceStatus(service_name, url, status, round((time.time() - start_time) * 1000, 2), details, category="api")

    def check_database_connection(self) -> ServiceStatus:
        """Check database connectivity"""
        try:
            # Try to import and test database connection
            import sqlite3
            
            # Check if database file exists
            db_paths = ["scrollintel.db", "database.db", "app.db"]
            db_found = False
            
            for db_path in db_paths:
                if os.path.exists(db_path):
                    try:
                        conn = sqlite3.connect(db_path)
                        cursor = conn.cursor()
                        cursor.execute("SELECT 1")
                        conn.close()
                        db_found = True
                        return ServiceStatus("Database", db_path, "✅ CONNECTED", 0, {"type": "SQLite", "path": db_path}, category="database")
                    except Exception as e:
                        return ServiceStatus("Database", db_path, "❌ ERROR", 0, {"error": str(e)}, category="database")
            
            if not db_found:
                return ServiceStatus("Database", "N/A", "⚠️  NO DB FILE", 0, {"error": "No database file found"}, category="database")
                
        except ImportError:
            return ServiceStatus("Database", "N/A", "❌ NO DRIVER", 0, {"error": "Database driver not available"}, category="database")

    def check_system_resources(self) -> Dict:
        """Check system resource usage"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return {
                "cpu_usage": f"{cpu_percent}%",
                "memory_usage": f"{memory.percent}%",
                "memory_available": f"{memory.available / (1024**3):.1f}GB",
                "disk_usage": f"{disk.percent}%",
                "disk_free": f"{disk.free / (1024**3):.1f}GB"
            }
        except Exception as e:
            return {"error": str(e)}

    def check_docker_services(self) -> List[ServiceStatus]:
        """Check Docker container status"""
        services = []
        try:
            result = subprocess.run(['docker', 'ps', '--format', 'json'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                containers = []
                for line in result.stdout.strip().split('\n'):
                    if line:
                        try:
                            containers.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
                
                for container in containers:
                    name = container.get('Names', 'Unknown')
                    status = container.get('Status', 'Unknown')
                    if 'Up' in status:
                        services.append(ServiceStatus(f"Docker: {name}", "docker", "✅ RUNNING", 0, {"status": status}, category="docker"))
                    else:
                        services.append(ServiceStatus(f"Docker: {name}", "docker", "❌ STOPPED", 0, {"status": status}, category="docker"))
            else:
                services.append(ServiceStatus("Docker", "docker", "❌ NOT AVAILABLE", 0, {"error": "Docker not running"}, category="docker"))
        except subprocess.TimeoutExpired:
            services.append(ServiceStatus("Docker", "docker", "⏰ TIMEOUT", 0, {"error": "Docker command timeout"}, category="docker"))
        except FileNotFoundError:
            services.append(ServiceStatus("Docker", "docker", "❌ NOT INSTALLED", 0, {"error": "Docker not installed"}, category="docker"))
        except Exception as e:
            services.append(ServiceStatus("Docker", "docker", "❌ ERROR", 0, {"error": str(e)}, category="docker"))
            
        return services

    def check_file_system(self) -> List[ServiceStatus]:
        """Check critical files and directories"""
        critical_paths = [
            ("Frontend Build", "frontend/dist"),
            ("Frontend Source", "frontend/src"),
            ("Backend Source", "scrollintel"),
            ("Requirements", "requirements.txt"),
            ("Docker Compose", "docker-compose.yml"),
            ("Environment", ".env"),
        ]
        
        results = []
        for name, path in critical_paths:
            if os.path.exists(path):
                if os.path.isdir(path):
                    file_count = len(list(Path(path).rglob('*')))
                    results.append(ServiceStatus(name, path, "✅ EXISTS", 0, {"type": "directory", "files": file_count}, category="filesystem"))
                else:
                    size = os.path.getsize(path)
                    results.append(ServiceStatus(name, path, "✅ EXISTS", 0, {"type": "file", "size_bytes": size}, category="filesystem"))
            else:
                results.append(ServiceStatus(name, path, "❌ MISSING", 0, {"error": "Path not found"}, category="filesystem"))
                
        return results

    def check_environment_variables(self) -> List[ServiceStatus]:
        """Check critical environment variables"""
        critical_env_vars = [
            "DATABASE_URL", "SECRET_KEY", "API_KEY", "OPENAI_API_KEY", 
            "ENVIRONMENT", "DEBUG", "PORT", "HOST"
        ]
        
        results = []
        for var in critical_env_vars:
            value = os.getenv(var)
            if value:
                # Mask sensitive values
                display_value = "***" if any(sensitive in var.lower() for sensitive in ["key", "secret", "password", "token"]) else value[:20] + "..." if len(value) > 20 else value
                results.append(ServiceStatus(f"ENV: {var}", "environment", "✅ SET", 0, {"value": display_value}, category="environment"))
            else:
                results.append(ServiceStatus(f"ENV: {var}", "environment", "⚠️  NOT SET", 0, {"warning": "Variable not set"}, category="environment"))
        
        return results

    def check_network_connectivity(self) -> List[ServiceStatus]:
        """Check network connectivity to external services"""
        external_services = [
            ("8.8.8.8", 53, "Google DNS"),
            ("1.1.1.1", 53, "Cloudflare DNS"),
            ("github.com", 443, "GitHub"),
            ("api.openai.com", 443, "OpenAI API"),
        ]
        
        results = []
        for host, port, name in external_services:
            result = self.check_port_connectivity(host, port, f"External: {name}")
            results.append(result)
        
        return results

    def generate_health_report(self, results: List[ServiceStatus]) -> str:
        """Generate a detailed health report"""
        categories = {}
        for result in results:
            category = result.category
            if category not in categories:
                categories[category] = {"healthy": 0, "total": 0, "issues": []}
            
            categories[category]["total"] += 1
            if "✅" in result.status:
                categories[category]["healthy"] += 1
            else:
                categories[category]["issues"].append(result)
        
        report = []
        report.append("📋 DETAILED HEALTH REPORT")
        report.append("=" * 50)
        
        for category, data in categories.items():
            health_rate = (data["healthy"] / data["total"]) * 100
            status_icon = "✅" if health_rate == 100 else "⚠️" if health_rate >= 50 else "❌"
            report.append(f"\n{status_icon} {category.upper()}: {data['healthy']}/{data['total']} ({health_rate:.1f}%)")
            
            if data["issues"]:
                for issue in data["issues"]:
                    report.append(f"  • {issue.name}: {issue.status}")
        
        return "\n".join(report)

    def save_results_to_file(self, results: List[ServiceStatus], filename: str = None):
        """Save results to JSON file for historical tracking"""
        if filename is None:
            filename = f"deployment_status_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        data = {
            "timestamp": self.start_time.isoformat(),
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "results": [
                {
                    "name": r.name,
                    "url": r.url,
                    "status": r.status,
                    "response_time": r.response_time,
                    "details": r.details,
                    "category": r.category,
                    "timestamp": r.timestamp.isoformat()
                }
                for r in results
            ]
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"📄 Results saved to: {filename}")
        except Exception as e:
            print(f"❌ Failed to save results: {e}")

    def run_comprehensive_check(self, save_report: bool = False) -> Dict:
        """Run all health checks concurrently"""
        print(f"🚀 ScrollIntel Comprehensive Deployment Status Check")
        print(f"⏰ Started: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        
        # Core services to check
        core_services = [
            ("http://localhost:8000/health", "Backend Health API"),
            ("http://localhost:8000/docs", "API Documentation"),
            ("http://localhost:8000/api/v1/agents", "Agents API"),
            ("http://localhost:3000", "Frontend Application"),
            ("http://localhost:3000/api/health", "Frontend Health"),
        ]
        
        # Additional API endpoints
        api_endpoints = [
            ("http://localhost:8000/api/v1/chat", "Chat API"),
            ("http://localhost:8000/api/v1/upload", "File Upload API"),
            ("http://localhost:8000/api/v1/monitoring", "Monitoring API"),
            ("http://localhost:8000/api/v1/analytics", "Analytics API"),
        ]
        
        all_results = []
        
        # Check core services
        print("\n🔧 Core Services:")
        print("-" * 40)
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            future_to_service = {executor.submit(self.check_service, url, name): (url, name) 
                               for url, name in core_services}
            
            for future in concurrent.futures.as_completed(future_to_service):
                result = future.result()
                all_results.append(result)
                print(f"{result.status:<25} {result.name} ({result.response_time:.0f}ms)")
        
        # Check API endpoints
        print("\n🌐 API Endpoints:")
        print("-" * 40)
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            future_to_service = {executor.submit(self.check_service, url, name): (url, name) 
                               for url, name in api_endpoints}
            
            for future in concurrent.futures.as_completed(future_to_service):
                result = future.result()
                all_results.append(result)
                print(f"{result.status:<25} {result.name} ({result.response_time:.0f}ms)")
        
        # Check database
        print("\n💾 Database:")
        print("-" * 40)
        db_result = self.check_database_connection()
        all_results.append(db_result)
        print(f"{db_result.status:<25} {db_result.name}")
        
        # Check Docker services
        print("\n🐳 Docker Services:")
        print("-" * 40)
        docker_results = self.check_docker_services()
        all_results.extend(docker_results)
        for result in docker_results:
            print(f"{result.status:<25} {result.name}")
        
        # Check file system
        print("\n📁 File System:")
        print("-" * 40)
        fs_results = self.check_file_system()
        all_results.extend(fs_results)
        for result in fs_results:
            print(f"{result.status:<25} {result.name}")
        
        # Check environment variables
        print("\n🌍 Environment Variables:")
        print("-" * 40)
        env_results = self.check_environment_variables()
        all_results.extend(env_results)
        for result in env_results:
            print(f"{result.status:<25} {result.name}")
        
        # Check network connectivity
        print("\n🌐 Network Connectivity:")
        print("-" * 40)
        network_results = self.check_network_connectivity()
        all_results.extend(network_results)
        for result in network_results:
            print(f"{result.status:<25} {result.name} ({result.response_time:.0f}ms)")
        
        # Check critical processes
        print("\n⚙️  Critical Processes:")
        print("-" * 40)
        critical_processes = ["python", "node", "nginx", "postgres", "redis"]
        for process in critical_processes:
            result = self.check_process_running(process)
            all_results.append(result)
            print(f"{result.status:<25} {result.name}")
        
        # System resources
        print("\n💻 System Resources:")
        print("-" * 40)
        resources = self.check_system_resources()
        for key, value in resources.items():
            if key != "error":
                print(f"{'✅ ' + key.replace('_', ' ').title():<25} {value}")
            else:
                print(f"{'❌ System Resources':<25} Error: {value}")
        
        # Summary
        healthy_count = sum(1 for r in all_results if "✅" in r.status)
        total_count = len(all_results)
        
        print("\n" + "=" * 80)
        print(f"📊 DEPLOYMENT STATUS SUMMARY")
        print(f"⏱️  Check Duration: {(datetime.now() - self.start_time).total_seconds():.1f}s")
        print(f"✅ Healthy Services: {healthy_count}/{total_count}")
        print(f"📈 Success Rate: {(healthy_count/total_count*100):.1f}%")
        
        # Generate detailed health report
        if self.verbose:
            print("\n" + self.generate_health_report(all_results))
        
        # Save results if requested
        if save_report:
            self.save_results_to_file(all_results)
        
        if healthy_count == total_count:
            print("\n🎉 ALL SYSTEMS OPERATIONAL!")
            print("\n🔗 Quick Access Links:")
            print("🌐 Application: http://localhost:3000")
            print("🔧 API Documentation: http://localhost:8000/docs")
            print("📊 Health Dashboard: http://localhost:8000/health")
            print("🤖 Chat Interface: http://localhost:3000/chat")
            print("📈 Monitoring: http://localhost:3000/monitoring")
            return {"status": "healthy", "results": all_results, "summary": {"healthy": healthy_count, "total": total_count}}
        else:
            print(f"\n⚠️  {total_count - healthy_count} SERVICES NEED ATTENTION")
            print("\n🔧 Troubleshooting Commands:")
            print("📋 Check logs: docker-compose logs -f")
            print("🔄 Restart services: docker-compose restart")
            print("🏗️  Rebuild: docker-compose up --build")
            print("🧪 Run tests: python -m pytest tests/")
            print("🔍 Debug mode: python check_deployment_status.py --verbose")
            return {"status": "degraded", "results": all_results, "summary": {"healthy": healthy_count, "total": total_count}}

def main():
    """Main entry point with command line argument support"""
    import argparse
    
    parser = argparse.ArgumentParser(description="ScrollIntel Deployment Status Checker")
    parser.add_argument("--timeout", type=int, default=10, help="Request timeout in seconds")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    parser.add_argument("--save-report", "-s", action="store_true", help="Save results to JSON file")
    parser.add_argument("--continuous", "-c", type=int, help="Run continuously every N seconds")
    parser.add_argument("--json", action="store_true", help="Output results in JSON format")
    
    args = parser.parse_args()
    
    def run_check():
        checker = ScrollIntelDeploymentChecker(timeout=args.timeout, verbose=args.verbose)
        result = checker.run_comprehensive_check(save_report=args.save_report)
        
        if args.json:
            print(json.dumps(result, indent=2, default=str))
        
        return result
    
    if args.continuous:
        print(f"🔄 Running continuous monitoring every {args.continuous} seconds...")
        print("Press Ctrl+C to stop")
        try:
            while True:
                result = run_check()
                if result["status"] != "healthy":
                    print(f"⚠️  Issues detected at {datetime.now().strftime('%H:%M:%S')}")
                time.sleep(args.continuous)
        except KeyboardInterrupt:
            print("\n👋 Monitoring stopped")
            sys.exit(0)
    else:
        result = run_check()
        # Exit with appropriate code
        if result["status"] != "healthy":
            sys.exit(1)

if __name__ == "__main__":
    main()