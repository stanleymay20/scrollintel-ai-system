#!/usr/bin/env python3
"""
ScrollIntel Auto-Scaling Manager
Monitors system metrics and automatically scales containers based on CPU and memory usage
"""

import os
import time
import json
import logging
import subprocess
import psutil
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AutoScalingManager:
    def __init__(self):
        self.config = {
            'cpu_threshold_up': float(os.getenv('CPU_SCALE_UP_THRESHOLD', '70')),
            'cpu_threshold_down': float(os.getenv('CPU_SCALE_DOWN_THRESHOLD', '30')),
            'memory_threshold_up': float(os.getenv('MEMORY_SCALE_UP_THRESHOLD', '80')),
            'memory_threshold_down': float(os.getenv('MEMORY_SCALE_DOWN_THRESHOLD', '40')),
            'min_replicas': int(os.getenv('MIN_REPLICAS', '2')),
            'max_replicas': int(os.getenv('MAX_REPLICAS', '10')),
            'scale_up_cooldown': int(os.getenv('SCALE_UP_COOLDOWN', '300')),  # 5 minutes
            'scale_down_cooldown': int(os.getenv('SCALE_DOWN_COOLDOWN', '600')),  # 10 minutes
            'check_interval': int(os.getenv('CHECK_INTERVAL', '60')),  # 1 minute
        }
        
        self.last_scale_action = None
        self.metrics_history = []
        self.current_replicas = self.get_current_replicas()
        
        logger.info(f"Auto-scaling manager initialized with config: {self.config}")

    def get_current_replicas(self) -> int:
        """Get current number of backend replicas"""
        try:
            result = subprocess.run(
                ['docker', 'ps', '--filter', 'name=scrollintel-backend', '--format', '{{.Names}}'],
                capture_output=True, text=True, check=True
            )
            replicas = len([line for line in result.stdout.strip().split('\n') if line])
            logger.info(f"Current replicas: {replicas}")
            return replicas
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to get current replicas: {e}")
            return self.config['min_replicas']

    def get_system_metrics(self) -> Dict:
        """Collect system metrics from all backend containers"""
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'cpu_percent': 0,
            'memory_percent': 0,
            'response_time': 0,
            'error_rate': 0,
            'active_connections': 0
        }
        
        try:
            # Get container metrics
            result = subprocess.run([
                'docker', 'stats', '--no-stream', '--format',
                'table {{.Container}}\t{{.CPUPerc}}\t{{.MemPerc}}\t{{.NetIO}}'
            ], capture_output=True, text=True, check=True)
            
            cpu_values = []
            memory_values = []
            
            for line in result.stdout.strip().split('\n')[1:]:  # Skip header
                if 'scrollintel-backend' in line:
                    parts = line.split('\t')
                    if len(parts) >= 3:
                        cpu_str = parts[1].replace('%', '')
                        memory_str = parts[2].replace('%', '')
                        
                        try:
                            cpu_values.append(float(cpu_str))
                            memory_values.append(float(memory_str))
                        except ValueError:
                            continue
            
            if cpu_values:
                metrics['cpu_percent'] = sum(cpu_values) / len(cpu_values)
            if memory_values:
                metrics['memory_percent'] = sum(memory_values) / len(memory_values)
            
            # Get application metrics
            app_metrics = self.get_application_metrics()
            metrics.update(app_metrics)
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to get system metrics: {e}")
        
        return metrics

    def get_application_metrics(self) -> Dict:
        """Get application-specific metrics from health endpoint"""
        metrics = {
            'response_time': 0,
            'error_rate': 0,
            'active_connections': 0
        }
        
        try:
            start_time = time.time()
            response = requests.get('http://localhost:8000/health', timeout=10)
            response_time = (time.time() - start_time) * 1000  # Convert to ms
            
            metrics['response_time'] = response_time
            
            if response.status_code == 200:
                health_data = response.json()
                metrics['active_connections'] = health_data.get('active_connections', 0)
                metrics['error_rate'] = health_data.get('error_rate', 0)
            
        except requests.RequestException as e:
            logger.warning(f"Failed to get application metrics: {e}")
            metrics['error_rate'] = 100  # Treat as error if health check fails
        
        return metrics

    def should_scale_up(self, metrics: Dict) -> bool:
        """Determine if we should scale up based on metrics"""
        if self.current_replicas >= self.config['max_replicas']:
            return False
        
        # Check cooldown period
        if (self.last_scale_action and 
            datetime.now() - self.last_scale_action < timedelta(seconds=self.config['scale_up_cooldown'])):
            return False
        
        # Check thresholds
        cpu_high = metrics['cpu_percent'] > self.config['cpu_threshold_up']
        memory_high = metrics['memory_percent'] > self.config['memory_threshold_up']
        response_slow = metrics['response_time'] > 2000  # 2 seconds
        error_rate_high = metrics['error_rate'] > 5  # 5%
        
        # Scale up if any critical threshold is exceeded
        return cpu_high or memory_high or response_slow or error_rate_high

    def should_scale_down(self, metrics: Dict) -> bool:
        """Determine if we should scale down based on metrics"""
        if self.current_replicas <= self.config['min_replicas']:
            return False
        
        # Check cooldown period
        if (self.last_scale_action and 
            datetime.now() - self.last_scale_action < timedelta(seconds=self.config['scale_down_cooldown'])):
            return False
        
        # Check if metrics have been consistently low
        if len(self.metrics_history) < 5:  # Need at least 5 data points
            return False
        
        recent_metrics = self.metrics_history[-5:]
        
        # All recent metrics should be below thresholds
        all_cpu_low = all(m['cpu_percent'] < self.config['cpu_threshold_down'] for m in recent_metrics)
        all_memory_low = all(m['memory_percent'] < self.config['memory_threshold_down'] for m in recent_metrics)
        all_response_fast = all(m['response_time'] < 1000 for m in recent_metrics)  # 1 second
        all_error_low = all(m['error_rate'] < 1 for m in recent_metrics)  # 1%
        
        return all_cpu_low and all_memory_low and all_response_fast and all_error_low

    def scale_up(self) -> bool:
        """Scale up the backend service"""
        try:
            new_replicas = min(self.current_replicas + 1, self.config['max_replicas'])
            
            logger.info(f"Scaling up from {self.current_replicas} to {new_replicas} replicas")
            
            # Use docker-compose to scale
            result = subprocess.run([
                'docker-compose', '-f', 'docker-compose.prod.yml',
                'up', '-d', '--scale', f'backend={new_replicas}'
            ], capture_output=True, text=True, check=True)
            
            self.current_replicas = new_replicas
            self.last_scale_action = datetime.now()
            
            # Wait for new container to be healthy
            time.sleep(30)
            
            # Verify scaling was successful
            actual_replicas = self.get_current_replicas()
            if actual_replicas == new_replicas:
                logger.info(f"Successfully scaled up to {new_replicas} replicas")
                self.send_notification(f"Scaled up to {new_replicas} replicas")
                return True
            else:
                logger.error(f"Scaling verification failed. Expected {new_replicas}, got {actual_replicas}")
                return False
                
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to scale up: {e}")
            return False

    def scale_down(self) -> bool:
        """Scale down the backend service"""
        try:
            new_replicas = max(self.current_replicas - 1, self.config['min_replicas'])
            
            logger.info(f"Scaling down from {self.current_replicas} to {new_replicas} replicas")
            
            # Use docker-compose to scale
            result = subprocess.run([
                'docker-compose', '-f', 'docker-compose.prod.yml',
                'up', '-d', '--scale', f'backend={new_replicas}'
            ], capture_output=True, text=True, check=True)
            
            self.current_replicas = new_replicas
            self.last_scale_action = datetime.now()
            
            # Wait for containers to stop gracefully
            time.sleep(15)
            
            # Verify scaling was successful
            actual_replicas = self.get_current_replicas()
            if actual_replicas == new_replicas:
                logger.info(f"Successfully scaled down to {new_replicas} replicas")
                self.send_notification(f"Scaled down to {new_replicas} replicas")
                return True
            else:
                logger.error(f"Scaling verification failed. Expected {new_replicas}, got {actual_replicas}")
                return False
                
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to scale down: {e}")
            return False

    def send_notification(self, message: str):
        """Send notification about scaling action"""
        webhook_url = os.getenv('SLACK_WEBHOOK_URL')
        if webhook_url:
            try:
                payload = {
                    'text': f"🔄 ScrollIntel Auto-Scaling: {message}",
                    'username': 'ScrollIntel Auto-Scaler',
                    'icon_emoji': ':chart_with_upwards_trend:'
                }
                requests.post(webhook_url, json=payload, timeout=10)
            except requests.RequestException as e:
                logger.warning(f"Failed to send notification: {e}")

    def save_metrics(self, metrics: Dict):
        """Save metrics to file for analysis"""
        metrics_file = 'logs/scaling_metrics.jsonl'
        os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
        
        try:
            with open(metrics_file, 'a') as f:
                f.write(json.dumps(metrics) + '\n')
        except IOError as e:
            logger.warning(f"Failed to save metrics: {e}")

    def run(self):
        """Main monitoring and scaling loop"""
        logger.info("Starting auto-scaling manager...")
        
        while True:
            try:
                # Collect metrics
                metrics = self.get_system_metrics()
                
                # Add to history (keep last 20 entries)
                self.metrics_history.append(metrics)
                if len(self.metrics_history) > 20:
                    self.metrics_history.pop(0)
                
                # Save metrics
                self.save_metrics(metrics)
                
                # Log current status
                logger.info(
                    f"Metrics - CPU: {metrics['cpu_percent']:.1f}%, "
                    f"Memory: {metrics['memory_percent']:.1f}%, "
                    f"Response: {metrics['response_time']:.0f}ms, "
                    f"Errors: {metrics['error_rate']:.1f}%, "
                    f"Replicas: {self.current_replicas}"
                )
                
                # Make scaling decisions
                if self.should_scale_up(metrics):
                    logger.info("Scaling up due to high resource usage")
                    self.scale_up()
                elif self.should_scale_down(metrics):
                    logger.info("Scaling down due to low resource usage")
                    self.scale_down()
                
                # Wait before next check
                time.sleep(self.config['check_interval'])
                
            except KeyboardInterrupt:
                logger.info("Shutting down auto-scaling manager...")
                break
            except Exception as e:
                logger.error(f"Unexpected error in scaling loop: {e}")
                time.sleep(self.config['check_interval'])

if __name__ == '__main__':
    manager = AutoScalingManager()
    manager.run()