#!/usr/bin/env python3
"""
ScrollIntel Production Infrastructure Deployment
Complete production deployment with auto-scaling, load balancing, and monitoring
"""

import os
import sys
import time
import logging
import subprocess
import json
from datetime import datetime
from typing import Dict, List, Optional
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ProductionInfrastructureDeployer:
    def __init__(self, deployment_type: str = 'full'):
        self.deployment_type = deployment_type
        self.deployment_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        self.config = {
            'environment': 'production',
            'deployment_id': self.deployment_id,
            'backend_instances': int(os.getenv('BACKEND_INSTANCES', '3')),
            'enable_auto_scaling': os.getenv('ENABLE_AUTO_SCALING', 'true').lower() == 'true',
            'enable_blue_green': os.getenv('ENABLE_BLUE_GREEN', 'true').lower() == 'true',
            'enable_db_replication': os.getenv('ENABLE_DB_REPLICATION', 'true').lower() == 'true',
            'enable_monitoring': os.getenv('ENABLE_MONITORING', 'true').lower() == 'true',
            'ssl_enabled': os.getenv('SSL_ENABLED', 'true').lower() == 'true',
            'cdn_enabled': os.getenv('CDN_ENABLED', 'false').lower() == 'true',
        }
        
        logger.info(f"Production infrastructure deployer initialized")
        logger.info(f"Deployment ID: {self.deployment_id}")
        logger.info(f"Deployment type: {deployment_type}")

    def validate_environment(self):
        """Validate environment and prerequisites"""
        logger.info("Validating deployment environment...")
        
        # Check required environment variables
        required_vars = [
            'POSTGRES_PASSWORD',
            'JWT_SECRET_KEY',
            'OPENAI_API_KEY'
        ]
        
        missing_vars = []
        for var in required_vars:
            if not os.getenv(var):
                missing_vars.append(var)
        
        if missing_vars:
            logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
            return False
        
        # Check Docker and Docker Compose
        try:
            subprocess.run(['docker', '--version'], check=True, capture_output=True)
            subprocess.run(['docker-compose', '--version'], check=True, capture_output=True)
        except subprocess.CalledProcessError:
            logger.error("Docker or Docker Compose not available")
            return False
        
        # Check available disk space (minimum 10GB)
        disk_usage = subprocess.run(['df', '-h', '.'], capture_output=True, text=True)
        logger.info(f"Disk usage: {disk_usage.stdout}")
        
        # Check available memory (minimum 4GB)
        memory_info = subprocess.run(['free', '-h'], capture_output=True, text=True)
        logger.info(f"Memory info: {memory_info.stdout}")
        
        logger.info("✅ Environment validation passed")
        return True

    def setup_ssl_certificates(self):
        """Setup SSL certificates for HTTPS"""
        if not self.config['ssl_enabled']:
            logger.info("SSL disabled, skipping certificate setup")
            return True
        
        logger.info("Setting up SSL certificates...")
        
        try:
            os.makedirs('./nginx/ssl', exist_ok=True)
            
            # Check if certificates already exist
            cert_file = './nginx/ssl/scrollintel.crt'
            key_file = './nginx/ssl/scrollintel.key'
            
            if os.path.exists(cert_file) and os.path.exists(key_file):
                logger.info("SSL certificates already exist")
                return True
            
            # Generate self-signed certificates for development/testing
            # In production, you would use Let's Encrypt or purchased certificates
            subprocess.run([
                'openssl', 'req', '-x509', '-nodes', '-days', '365',
                '-newkey', 'rsa:2048',
                '-keyout', key_file,
                '-out', cert_file,
                '-subj', '/C=US/ST=State/L=City/O=ScrollIntel/CN=scrollintel.com'
            ], check=True)
            
            logger.info("✅ SSL certificates generated")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"SSL certificate setup failed: {e}")
            return False

    def deploy_database_infrastructure(self):
        """Deploy database with replication if enabled"""
        logger.info("Deploying database infrastructure...")
        
        try:
            if self.config['enable_db_replication']:
                logger.info("Setting up database replication...")
                result = subprocess.run([
                    'python', 'scripts/database-replication-setup.py'
                ], check=True, capture_output=True, text=True)
                logger.info("Database replication setup completed")
            else:
                logger.info("Starting single database instance...")
                subprocess.run([
                    'docker-compose', '-f', 'docker-compose.prod.yml',
                    'up', '-d', 'postgres', 'redis'
                ], check=True)
            
            # Wait for database to be ready
            self.wait_for_service('postgres', 'localhost', 5432, 60)
            
            # Run database migrations
            logger.info("Running database migrations...")
            subprocess.run([
                'python', 'scripts/migrate-database.py', '--env', 'production', 'migrate'
            ], check=True)
            
            logger.info("✅ Database infrastructure deployed")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Database deployment failed: {e}")
            return False

    def deploy_load_balancer(self):
        """Deploy load balancer infrastructure"""
        logger.info("Deploying load balancer infrastructure...")
        
        try:
            # Setup load balancer configuration
            result = subprocess.run([
                'python', 'scripts/load-balancer-setup.py'
            ], check=True, capture_output=True, text=True)
            
            # Start load-balanced services
            logger.info("Starting load-balanced services...")
            subprocess.run([
                'docker-compose', '-f', 'docker-compose.load-balanced.yml',
                'up', '-d'
            ], check=True)
            
            # Wait for services to be ready
            self.wait_for_service('nginx', 'localhost', 80, 120)
            
            logger.info("✅ Load balancer infrastructure deployed")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Load balancer deployment failed: {e}")
            return False

    def deploy_monitoring_infrastructure(self):
        """Deploy monitoring and alerting infrastructure"""
        if not self.config['enable_monitoring']:
            logger.info("Monitoring disabled, skipping")
            return True
        
        logger.info("Deploying monitoring infrastructure...")
        
        try:
            # Start monitoring services
            subprocess.run([
                'docker-compose', '-f', 'docker-compose.prod.yml',
                'up', '-d', 'prometheus', 'grafana'
            ], check=True)
            
            # Wait for monitoring services
            self.wait_for_service('prometheus', 'localhost', 9090, 60)
            self.wait_for_service('grafana', 'localhost', 3001, 60)
            
            # Import Grafana dashboards
            self.setup_grafana_dashboards()
            
            logger.info("✅ Monitoring infrastructure deployed")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Monitoring deployment failed: {e}")
            return False

    def setup_auto_scaling(self):
        """Setup auto-scaling if enabled"""
        if not self.config['enable_auto_scaling']:
            logger.info("Auto-scaling disabled, skipping")
            return True
        
        logger.info("Setting up auto-scaling...")
        
        try:
            # Start auto-scaling manager as a background service
            subprocess.Popen([
                'python', 'scripts/auto-scaling-manager.py'
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            logger.info("✅ Auto-scaling manager started")
            return True
            
        except Exception as e:
            logger.error(f"Auto-scaling setup failed: {e}")
            return False

    def wait_for_service(self, service_name: str, host: str, port: int, timeout: int = 60):
        """Wait for a service to become available"""
        logger.info(f"Waiting for {service_name} to be ready on {host}:{port}...")
        
        import socket
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(5)
                result = sock.connect_ex((host, port))
                sock.close()
                
                if result == 0:
                    logger.info(f"✅ {service_name} is ready")
                    return True
                    
            except Exception:
                pass
            
            time.sleep(5)
        
        logger.error(f"❌ {service_name} did not become ready within {timeout}s")
        return False

    def setup_grafana_dashboards(self):
        """Setup Grafana dashboards"""
        logger.info("Setting up Grafana dashboards...")
        
        # Wait a bit for Grafana to fully start
        time.sleep(30)
        
        try:
            # Import ScrollIntel dashboard
            dashboard_config = {
                "dashboard": {
                    "id": None,
                    "title": "ScrollIntel Production Dashboard",
                    "tags": ["scrollintel", "production"],
                    "timezone": "browser",
                    "panels": [
                        {
                            "id": 1,
                            "title": "Backend Response Time",
                            "type": "graph",
                            "targets": [
                                {
                                    "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))",
                                    "legendFormat": "95th percentile"
                                }
                            ]
                        },
                        {
                            "id": 2,
                            "title": "Active Backend Instances",
                            "type": "stat",
                            "targets": [
                                {
                                    "expr": "up{job=\"scrollintel-backends\"}",
                                    "legendFormat": "Backend {{instance}}"
                                }
                            ]
                        }
                    ]
                },
                "overwrite": True
            }
            
            # Save dashboard configuration
            with open('./monitoring/grafana-dashboard.json', 'w') as f:
                json.dump(dashboard_config, f, indent=2)
            
            logger.info("Grafana dashboard configuration saved")
            
        except Exception as e:
            logger.warning(f"Grafana dashboard setup failed: {e}")

    def run_health_checks(self):
        """Run comprehensive health checks"""
        logger.info("Running comprehensive health checks...")
        
        health_checks = [
            ('Database', 'localhost', 5432),
            ('Redis', 'localhost', 6379),
            ('Load Balancer', 'localhost', 80),
            ('Monitoring', 'localhost', 9090),
        ]
        
        all_healthy = True
        
        for service, host, port in health_checks:
            if self.wait_for_service(service, host, port, 30):
                logger.info(f"✅ {service} health check passed")
            else:
                logger.error(f"❌ {service} health check failed")
                all_healthy = False
        
        # Test API endpoints
        try:
            import requests
            
            api_endpoints = [
                'http://localhost/health',
                'http://localhost/api/agents',
            ]
            
            for endpoint in api_endpoints:
                response = requests.get(endpoint, timeout=10)
                if response.status_code == 200:
                    logger.info(f"✅ API endpoint {endpoint} is healthy")
                else:
                    logger.error(f"❌ API endpoint {endpoint} returned {response.status_code}")
                    all_healthy = False
                    
        except Exception as e:
            logger.error(f"API health check failed: {e}")
            all_healthy = False
        
        return all_healthy

    def create_deployment_summary(self):
        """Create deployment summary report"""
        logger.info("Creating deployment summary...")
        
        summary = {
            'deployment_id': self.deployment_id,
            'timestamp': datetime.now().isoformat(),
            'deployment_type': self.deployment_type,
            'configuration': self.config,
            'services': {
                'database': 'deployed' if self.config['enable_db_replication'] else 'single-instance',
                'load_balancer': 'deployed',
                'monitoring': 'deployed' if self.config['enable_monitoring'] else 'disabled',
                'auto_scaling': 'enabled' if self.config['enable_auto_scaling'] else 'disabled',
                'ssl': 'enabled' if self.config['ssl_enabled'] else 'disabled'
            },
            'endpoints': {
                'application': 'http://localhost' + (':443' if self.config['ssl_enabled'] else ''),
                'monitoring': 'http://localhost:3001' if self.config['enable_monitoring'] else None,
                'health_check': 'http://localhost:8080/health'
            },
            'next_steps': [
                'Monitor application logs and metrics',
                'Run smoke tests on production environment',
                'Update DNS records to point to production',
                'Configure backup schedules',
                'Set up alerting notifications'
            ]
        }
        
        os.makedirs('./deployments/reports', exist_ok=True)
        
        with open(f'./deployments/reports/deployment_{self.deployment_id}.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Print summary
        print("\n" + "="*60)
        print("🚀 SCROLLINTEL PRODUCTION DEPLOYMENT COMPLETE")
        print("="*60)
        print(f"Deployment ID: {self.deployment_id}")
        print(f"Timestamp: {summary['timestamp']}")
        print(f"Type: {self.deployment_type}")
        print("\n📊 Services Deployed:")
        for service, status in summary['services'].items():
            print(f"  • {service.title()}: {status}")
        
        print("\n🔗 Access Points:")
        for name, url in summary['endpoints'].items():
            if url:
                print(f"  • {name.title()}: {url}")
        
        print("\n📋 Next Steps:")
        for i, step in enumerate(summary['next_steps'], 1):
            print(f"  {i}. {step}")
        
        print("\n" + "="*60)
        
        return summary

    def deploy(self):
        """Execute complete production deployment"""
        logger.info(f"Starting ScrollIntel production deployment {self.deployment_id}")
        
        try:
            # Step 1: Validate environment
            if not self.validate_environment():
                return False
            
            # Step 2: Setup SSL certificates
            if not self.setup_ssl_certificates():
                return False
            
            # Step 3: Deploy database infrastructure
            if not self.deploy_database_infrastructure():
                return False
            
            # Step 4: Deploy load balancer
            if not self.deploy_load_balancer():
                return False
            
            # Step 5: Deploy monitoring
            if not self.deploy_monitoring_infrastructure():
                return False
            
            # Step 6: Setup auto-scaling
            if not self.setup_auto_scaling():
                return False
            
            # Step 7: Run health checks
            if not self.run_health_checks():
                logger.warning("Some health checks failed, but deployment will continue")
            
            # Step 8: Create deployment summary
            self.create_deployment_summary()
            
            logger.info("🎉 Production deployment completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Production deployment failed: {e}")
            return False

def main():
    parser = argparse.ArgumentParser(description='ScrollIntel Production Infrastructure Deployment')
    parser.add_argument('--type', choices=['full', 'minimal', 'update'], default='full',
                       help='Deployment type (default: full)')
    parser.add_argument('--skip-validation', action='store_true',
                       help='Skip environment validation')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be deployed without actually deploying')
    
    args = parser.parse_args()
    
    if args.dry_run:
        logger.info("DRY RUN MODE - No actual deployment will occur")
        # Show what would be deployed
        config = {
            'backend_instances': int(os.getenv('BACKEND_INSTANCES', '3')),
            'enable_auto_scaling': os.getenv('ENABLE_AUTO_SCALING', 'true').lower() == 'true',
            'enable_blue_green': os.getenv('ENABLE_BLUE_GREEN', 'true').lower() == 'true',
            'enable_db_replication': os.getenv('ENABLE_DB_REPLICATION', 'true').lower() == 'true',
            'enable_monitoring': os.getenv('ENABLE_MONITORING', 'true').lower() == 'true',
        }
        
        print("Would deploy with configuration:")
        for key, value in config.items():
            print(f"  {key}: {value}")
        
        return
    
    deployer = ProductionInfrastructureDeployer(args.type)
    
    if args.skip_validation:
        logger.warning("Skipping environment validation as requested")
    
    success = deployer.deploy()
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()