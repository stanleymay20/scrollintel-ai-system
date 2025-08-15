#!/usr/bin/env python3
"""
ScrollIntel Blue-Green Deployment Manager
Implements zero-downtime deployments using blue-green strategy
"""

import os
import sys
import time
import json
import logging
import subprocess
import requests
from datetime import datetime
from typing import Dict, Optional, List
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BlueGreenDeployment:
    def __init__(self):
        self.config = {
            'health_check_timeout': int(os.getenv('HEALTH_CHECK_TIMEOUT', '300')),
            'health_check_interval': int(os.getenv('HEALTH_CHECK_INTERVAL', '10')),
            'warmup_time': int(os.getenv('WARMUP_TIME', '60')),
            'rollback_on_failure': os.getenv('ROLLBACK_ON_FAILURE', 'true').lower() == 'true',
            'backup_retention': int(os.getenv('BACKUP_RETENTION', '5')),
        }
        
        self.deployment_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.backup_dir = f"./backups/deployment_{self.deployment_id}"
        
        # Environment colors
        self.environments = {
            'blue': {
                'compose_file': 'docker-compose.blue.yml',
                'port': 8000,
                'name': 'blue'
            },
            'green': {
                'compose_file': 'docker-compose.green.yml', 
                'port': 8001,
                'name': 'green'
            }
        }
        
        self.current_env = self.get_current_environment()
        self.target_env = 'green' if self.current_env == 'blue' else 'blue'
        
        logger.info(f"Blue-Green Deployment initialized")
        logger.info(f"Current environment: {self.current_env}")
        logger.info(f"Target environment: {self.target_env}")

    def get_current_environment(self) -> str:
        """Determine which environment is currently active"""
        try:
            # Check which containers are running
            result = subprocess.run([
                'docker', 'ps', '--filter', 'name=scrollintel-backend',
                '--format', '{{.Names}}'
            ], capture_output=True, text=True, check=True)
            
            containers = result.stdout.strip().split('\n')
            
            if any('blue' in container for container in containers):
                return 'blue'
            elif any('green' in container for container in containers):
                return 'green'
            else:
                # Default to blue if no containers are running
                return 'blue'
                
        except subprocess.CalledProcessError:
            logger.warning("Could not determine current environment, defaulting to blue")
            return 'blue'

    def create_compose_files(self):
        """Create blue and green docker-compose files"""
        base_compose = self.load_base_compose()
        
        for env_name, env_config in self.environments.items():
            compose_data = self.customize_compose_for_environment(base_compose, env_name, env_config)
            
            with open(env_config['compose_file'], 'w') as f:
                yaml.dump(compose_data, f, default_flow_style=False)
            
            logger.info(f"Created {env_config['compose_file']} for {env_name} environment")

    def load_base_compose(self) -> Dict:
        """Load the base docker-compose configuration"""
        try:
            with open('docker-compose.prod.yml', 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.error("Base docker-compose.prod.yml not found")
            sys.exit(1)

    def customize_compose_for_environment(self, base_compose: Dict, env_name: str, env_config: Dict) -> Dict:
        """Customize compose configuration for specific environment"""
        compose_data = base_compose.copy()
        
        # Update service names and ports
        if 'services' in compose_data:
            # Update backend service
            if 'backend' in compose_data['services']:
                backend_service = compose_data['services']['backend']
                backend_service['container_name'] = f"scrollintel-backend-{env_name}"
                backend_service['ports'] = [f"{env_config['port']}:8000"]
                
                # Add environment-specific labels
                if 'labels' not in backend_service:
                    backend_service['labels'] = {}
                backend_service['labels']['environment'] = env_name
                backend_service['labels']['deployment_id'] = self.deployment_id
            
            # Update other services
            for service_name, service_config in compose_data['services'].items():
                if service_name != 'backend':
                    if 'container_name' in service_config:
                        service_config['container_name'] = f"{service_config['container_name']}-{env_name}"
        
        return compose_data

    def backup_current_state(self):
        """Create backup of current deployment state"""
        logger.info("Creating backup of current deployment state...")
        
        os.makedirs(self.backup_dir, exist_ok=True)
        
        try:
            # Backup database
            self.backup_database()
            
            # Backup configuration files
            config_files = [
                'docker-compose.prod.yml',
                'nginx/nginx.conf',
                '.env.production'
            ]
            
            for config_file in config_files:
                if os.path.exists(config_file):
                    subprocess.run([
                        'cp', config_file, f"{self.backup_dir}/{os.path.basename(config_file)}"
                    ], check=True)
            
            # Save deployment metadata
            metadata = {
                'deployment_id': self.deployment_id,
                'timestamp': datetime.now().isoformat(),
                'current_environment': self.current_env,
                'target_environment': self.target_env,
                'git_commit': self.get_git_commit(),
                'docker_images': self.get_current_images()
            }
            
            with open(f"{self.backup_dir}/metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Backup created at {self.backup_dir}")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Backup failed: {e}")
            raise

    def backup_database(self):
        """Backup the database"""
        try:
            database_url = os.getenv('DATABASE_URL')
            if not database_url:
                logger.warning("DATABASE_URL not set, skipping database backup")
                return
            
            backup_file = f"{self.backup_dir}/database_backup.sql"
            
            subprocess.run([
                'pg_dump', database_url, '-f', backup_file
            ], check=True)
            
            logger.info("Database backup completed")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Database backup failed: {e}")
            raise

    def get_git_commit(self) -> Optional[str]:
        """Get current git commit hash"""
        try:
            result = subprocess.run([
                'git', 'rev-parse', 'HEAD'
            ], capture_output=True, text=True, check=True)
            return result.stdout.strip()
        except subprocess.CalledProcessError:
            return None

    def get_current_images(self) -> List[str]:
        """Get list of currently running Docker images"""
        try:
            result = subprocess.run([
                'docker', 'ps', '--filter', 'name=scrollintel',
                '--format', '{{.Image}}'
            ], capture_output=True, text=True, check=True)
            return result.stdout.strip().split('\n')
        except subprocess.CalledProcessError:
            return []

    def deploy_to_target_environment(self):
        """Deploy new version to target environment"""
        logger.info(f"Deploying to {self.target_env} environment...")
        
        target_config = self.environments[self.target_env]
        
        try:
            # Build new images
            logger.info("Building new Docker images...")
            subprocess.run([
                'docker', 'build', '-t', f'scrollintel-backend:{self.target_env}',
                '--target', 'production', '.'
            ], check=True)
            
            # Deploy to target environment
            logger.info(f"Starting {self.target_env} environment...")
            subprocess.run([
                'docker-compose', '-f', target_config['compose_file'],
                'up', '-d', '--build'
            ], check=True)
            
            # Wait for warmup
            logger.info(f"Warming up {self.target_env} environment...")
            time.sleep(self.config['warmup_time'])
            
            logger.info(f"Deployment to {self.target_env} environment completed")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Deployment to {self.target_env} failed: {e}")
            raise

    def health_check(self, environment: str) -> bool:
        """Perform comprehensive health check on environment"""
        env_config = self.environments[environment]
        base_url = f"http://localhost:{env_config['port']}"
        
        logger.info(f"Performing health check on {environment} environment...")
        
        # Health check endpoints
        endpoints = [
            '/health',
            '/api/health',
            '/api/agents',
        ]
        
        start_time = time.time()
        timeout = self.config['health_check_timeout']
        
        while time.time() - start_time < timeout:
            try:
                all_healthy = True
                
                for endpoint in endpoints:
                    response = requests.get(
                        f"{base_url}{endpoint}",
                        timeout=10,
                        headers={'User-Agent': 'BlueGreenHealthCheck/1.0'}
                    )
                    
                    if response.status_code != 200:
                        logger.warning(f"Health check failed for {endpoint}: {response.status_code}")
                        all_healthy = False
                        break
                
                if all_healthy:
                    # Additional performance check
                    response_time = self.measure_response_time(base_url)
                    if response_time < 2000:  # 2 seconds
                        logger.info(f"Health check passed for {environment} environment")
                        return True
                    else:
                        logger.warning(f"Response time too slow: {response_time}ms")
                
            except requests.RequestException as e:
                logger.warning(f"Health check request failed: {e}")
            
            time.sleep(self.config['health_check_interval'])
        
        logger.error(f"Health check failed for {environment} environment after {timeout}s")
        return False

    def measure_response_time(self, base_url: str) -> float:
        """Measure average response time"""
        total_time = 0
        num_requests = 5
        
        for _ in range(num_requests):
            start_time = time.time()
            try:
                requests.get(f"{base_url}/health", timeout=10)
                total_time += (time.time() - start_time) * 1000
            except requests.RequestException:
                return float('inf')
        
        return total_time / num_requests

    def switch_traffic(self):
        """Switch traffic from current to target environment"""
        logger.info(f"Switching traffic from {self.current_env} to {self.target_env}...")
        
        try:
            # Update nginx configuration
            self.update_nginx_config()
            
            # Reload nginx
            subprocess.run(['docker', 'exec', 'scrollintel-nginx-prod', 'nginx', '-s', 'reload'], check=True)
            
            # Wait for traffic to switch
            time.sleep(10)
            
            # Verify traffic switch
            if self.verify_traffic_switch():
                logger.info("Traffic switch completed successfully")
                return True
            else:
                logger.error("Traffic switch verification failed")
                return False
                
        except subprocess.CalledProcessError as e:
            logger.error(f"Traffic switch failed: {e}")
            return False

    def update_nginx_config(self):
        """Update nginx configuration to point to target environment"""
        target_port = self.environments[self.target_env]['port']
        
        # Read current nginx config
        with open('nginx/nginx.conf', 'r') as f:
            config = f.read()
        
        # Update upstream backend server
        config = config.replace(
            'server backend:8000',
            f'server backend:{target_port}'
        )
        
        # Write updated config
        with open('nginx/nginx.conf', 'w') as f:
            f.write(config)
        
        logger.info(f"Updated nginx config to point to port {target_port}")

    def verify_traffic_switch(self) -> bool:
        """Verify that traffic is being routed to target environment"""
        try:
            # Make requests through nginx and check which backend responds
            response = requests.get('http://localhost/health', timeout=10)
            
            if response.status_code == 200:
                # Check response headers or content to identify which environment served the request
                # This would depend on your specific implementation
                return True
            
        except requests.RequestException as e:
            logger.error(f"Traffic verification failed: {e}")
        
        return False

    def cleanup_old_environment(self):
        """Stop and remove containers from old environment"""
        logger.info(f"Cleaning up {self.current_env} environment...")
        
        current_config = self.environments[self.current_env]
        
        try:
            # Stop old environment
            subprocess.run([
                'docker-compose', '-f', current_config['compose_file'],
                'down'
            ], check=True)
            
            # Remove old images (optional)
            subprocess.run([
                'docker', 'image', 'prune', '-f'
            ], check=True)
            
            logger.info(f"Cleanup of {self.current_env} environment completed")
            
        except subprocess.CalledProcessError as e:
            logger.warning(f"Cleanup failed: {e}")

    def rollback(self):
        """Rollback to previous environment"""
        logger.error("Initiating rollback...")
        
        try:
            # Switch traffic back to current environment
            current_port = self.environments[self.current_env]['port']
            
            # Restore nginx config
            with open('nginx/nginx.conf', 'r') as f:
                config = f.read()
            
            config = config.replace(
                f'server backend:{self.environments[self.target_env]["port"]}',
                f'server backend:{current_port}'
            )
            
            with open('nginx/nginx.conf', 'w') as f:
                f.write(config)
            
            # Reload nginx
            subprocess.run(['docker', 'exec', 'scrollintel-nginx-prod', 'nginx', '-s', 'reload'], check=True)
            
            # Stop target environment
            target_config = self.environments[self.target_env]
            subprocess.run([
                'docker-compose', '-f', target_config['compose_file'],
                'down'
            ], check=True)
            
            logger.info("Rollback completed successfully")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Rollback failed: {e}")

    def send_notification(self, message: str, success: bool = True):
        """Send deployment notification"""
        webhook_url = os.getenv('SLACK_WEBHOOK_URL')
        if webhook_url:
            try:
                emoji = '✅' if success else '❌'
                payload = {
                    'text': f"{emoji} ScrollIntel Blue-Green Deployment: {message}",
                    'username': 'ScrollIntel Deployer',
                    'icon_emoji': ':rocket:' if success else ':warning:'
                }
                requests.post(webhook_url, json=payload, timeout=10)
            except requests.RequestException as e:
                logger.warning(f"Failed to send notification: {e}")

    def deploy(self):
        """Execute complete blue-green deployment"""
        logger.info(f"Starting blue-green deployment {self.deployment_id}")
        
        try:
            # Step 1: Create compose files
            self.create_compose_files()
            
            # Step 2: Backup current state
            self.backup_current_state()
            
            # Step 3: Deploy to target environment
            self.deploy_to_target_environment()
            
            # Step 4: Health check target environment
            if not self.health_check(self.target_env):
                if self.config['rollback_on_failure']:
                    self.rollback()
                    self.send_notification(f"Deployment {self.deployment_id} failed and was rolled back", False)
                    return False
                else:
                    logger.error("Health check failed but rollback is disabled")
                    return False
            
            # Step 5: Switch traffic
            if not self.switch_traffic():
                if self.config['rollback_on_failure']:
                    self.rollback()
                    self.send_notification(f"Traffic switch failed, deployment {self.deployment_id} rolled back", False)
                    return False
                else:
                    logger.error("Traffic switch failed but rollback is disabled")
                    return False
            
            # Step 6: Final verification
            time.sleep(30)  # Allow traffic to stabilize
            if not self.health_check(self.target_env):
                logger.error("Final health check failed")
                if self.config['rollback_on_failure']:
                    self.rollback()
                    self.send_notification(f"Final verification failed, deployment {self.deployment_id} rolled back", False)
                    return False
            
            # Step 7: Cleanup old environment
            self.cleanup_old_environment()
            
            # Step 8: Success notification
            self.send_notification(f"Deployment {self.deployment_id} completed successfully", True)
            
            logger.info(f"Blue-green deployment {self.deployment_id} completed successfully!")
            logger.info(f"Active environment: {self.target_env}")
            
            return True
            
        except Exception as e:
            logger.error(f"Deployment failed with error: {e}")
            if self.config['rollback_on_failure']:
                self.rollback()
                self.send_notification(f"Deployment {self.deployment_id} failed with error: {str(e)}", False)
            return False

if __name__ == '__main__':
    deployment = BlueGreenDeployment()
    success = deployment.deploy()
    sys.exit(0 if success else 1)