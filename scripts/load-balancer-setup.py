#!/usr/bin/env python3
"""
ScrollIntel Load Balancer Setup
Configures advanced load balancing with health checks and failover
"""

import os
import sys
import time
import logging
import subprocess
import requests
from typing import Dict, List, Optional
import yaml
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LoadBalancerManager:
    def __init__(self):
        self.config = {
            'backend_instances': int(os.getenv('BACKEND_INSTANCES', '3')),
            'health_check_interval': int(os.getenv('HEALTH_CHECK_INTERVAL', '30')),
            'health_check_timeout': int(os.getenv('HEALTH_CHECK_TIMEOUT', '10')),
            'max_fails': int(os.getenv('MAX_FAILS', '3')),
            'fail_timeout': int(os.getenv('FAIL_TIMEOUT', '30')),
            'load_balancing_method': os.getenv('LOAD_BALANCING_METHOD', 'least_conn'),
        }
        
        self.backend_ports = list(range(8000, 8000 + self.config['backend_instances']))
        
        logger.info(f"Load balancer manager initialized with {self.config['backend_instances']} backend instances")

    def create_nginx_config(self):
        """Create advanced nginx configuration with load balancing"""
        logger.info("Creating advanced nginx load balancer configuration...")
        
        # Generate upstream backend servers
        upstream_servers = []
        for i, port in enumerate(self.backend_ports):
            upstream_servers.append(f"        server backend-{i+1}:{port} max_fails={self.config['max_fails']} fail_timeout={self.config['fail_timeout']}s;")
        
        upstream_config = '\n'.join(upstream_servers)
        
        nginx_config = f"""
# ================================
# ScrollIntel Advanced Load Balancer Configuration
# High-availability nginx with health checks and failover
# ================================

user nginx;
worker_processes auto;
error_log /var/log/nginx/error.log warn;
pid /var/run/nginx.pid;

# Load dynamic modules
load_module modules/ngx_http_upstream_check_module.so;

events {{
    worker_connections 2048;
    use epoll;
    multi_accept on;
    accept_mutex off;
}}

http {{
    include /etc/nginx/mime.types;
    default_type application/octet-stream;
    
    # Logging format with load balancing info
    log_format main '$remote_addr - $remote_user [$time_local] "$request" '
                    '$status $body_bytes_sent "$http_referer" '
                    '"$http_user_agent" "$http_x_forwarded_for" '
                    'rt=$request_time uct="$upstream_connect_time" '
                    'uht="$upstream_header_time" urt="$upstream_response_time" '
                    'upstream="$upstream_addr" cache="$upstream_cache_status"';
    
    access_log /var/log/nginx/access.log main;
    
    # Basic settings optimized for load balancing
    sendfile on;
    tcp_nopush on;
    tcp_nodelay on;
    keepalive_timeout 65;
    keepalive_requests 1000;
    types_hash_max_size 2048;
    client_max_body_size 100M;
    client_body_timeout 60s;
    client_header_timeout 60s;
    
    # Buffer settings
    proxy_buffering on;
    proxy_buffer_size 128k;
    proxy_buffers 4 256k;
    proxy_busy_buffers_size 256k;
    
    # Gzip compression
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_proxied any;
    gzip_comp_level 6;
    gzip_types
        text/plain
        text/css
        text/xml
        text/javascript
        application/json
        application/javascript
        application/xml+rss
        application/atom+xml
        image/svg+xml;
    
    # Rate limiting zones
    limit_req_zone $binary_remote_addr zone=api:10m rate=100r/s;
    limit_req_zone $binary_remote_addr zone=auth:10m rate=10r/s;
    limit_req_zone $binary_remote_addr zone=upload:10m rate=5r/s;
    
    # Connection limiting
    limit_conn_zone $binary_remote_addr zone=perip:10m;
    limit_conn_zone $server_name zone=perserver:10m;
    
    # Upstream backend servers with health checks
    upstream scrollintel_backend {{
        {self.config['load_balancing_method']};
{upstream_config}
        
        # Connection pooling
        keepalive 32;
        keepalive_requests 100;
        keepalive_timeout 60s;
        
        # Health check configuration
        check interval={self.config['health_check_interval']}s 
              rise=2 
              fall={self.config['max_fails']} 
              timeout={self.config['health_check_timeout']}s 
              default_down=true 
              type=http;
        check_http_send "GET /health HTTP/1.0\\r\\n\\r\\n";
        check_http_expect_alive http_2xx http_3xx;
    }}
    
    # Upstream for frontend servers
    upstream scrollintel_frontend {{
        least_conn;
        server frontend-1:3000 max_fails=2 fail_timeout=30s;
        server frontend-2:3000 max_fails=2 fail_timeout=30s backup;
        keepalive 16;
    }}
    
    # Cache configuration
    proxy_cache_path /var/cache/nginx/scrollintel 
                     levels=1:2 
                     keys_zone=scrollintel_cache:10m 
                     max_size=1g 
                     inactive=60m 
                     use_temp_path=off;
    
    # Health check status page
    server {{
        listen 8080;
        server_name health.scrollintel.local;
        
        location /nginx_status {{
            stub_status on;
            access_log off;
            allow 127.0.0.1;
            allow 10.0.0.0/8;
            allow 172.16.0.0/12;
            allow 192.168.0.0/16;
            deny all;
        }}
        
        location /upstream_check {{
            check_status;
            access_log off;
            allow 127.0.0.1;
            allow 10.0.0.0/8;
            allow 172.16.0.0/12;
            allow 192.168.0.0/16;
            deny all;
        }}
        
        location /health {{
            access_log off;
            return 200 "healthy\\n";
            add_header Content-Type text/plain;
        }}
    }}
    
    # Main application server
    server {{
        listen 80;
        listen [::]:80;
        server_name scrollintel.com *.scrollintel.com;
        
        # Security headers
        add_header X-Frame-Options DENY always;
        add_header X-Content-Type-Options nosniff always;
        add_header X-XSS-Protection "1; mode=block" always;
        add_header Referrer-Policy "strict-origin-when-cross-origin" always;
        add_header Content-Security-Policy "default-src 'self'; script-src 'self' 'unsafe-inline' 'unsafe-eval'; style-src 'self' 'unsafe-inline';" always;
        
        # Connection limits
        limit_conn perip 20;
        limit_conn perserver 1000;
        
        # API routes with load balancing
        location /api/ {{
            limit_req zone=api burst=50 nodelay;
            
            # Proxy settings
            proxy_pass http://scrollintel_backend;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection 'upgrade';
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            proxy_set_header X-Forwarded-Host $server_name;
            proxy_cache_bypass $http_upgrade;
            
            # Timeouts
            proxy_connect_timeout 30s;
            proxy_send_timeout 60s;
            proxy_read_timeout 60s;
            
            # Retry logic
            proxy_next_upstream error timeout invalid_header http_500 http_502 http_503 http_504;
            proxy_next_upstream_tries 3;
            proxy_next_upstream_timeout 30s;
            
            # Caching for GET requests
            proxy_cache scrollintel_cache;
            proxy_cache_valid 200 302 10m;
            proxy_cache_valid 404 1m;
            proxy_cache_use_stale error timeout updating http_500 http_502 http_503 http_504;
            proxy_cache_lock on;
            proxy_cache_lock_timeout 5s;
            
            # Cache bypass for non-GET requests
            proxy_cache_methods GET HEAD;
            proxy_cache_bypass $request_method;
        }}
        
        # Authentication routes (stricter rate limiting)
        location /api/auth/ {{
            limit_req zone=auth burst=20 nodelay;
            
            proxy_pass http://scrollintel_backend;
            proxy_http_version 1.1;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # No caching for auth endpoints
            proxy_cache off;
            add_header Cache-Control "no-cache, no-store, must-revalidate";
        }}
        
        # File upload routes
        location /api/upload/ {{
            limit_req zone=upload burst=10 nodelay;
            
            # Increase timeouts for file uploads
            client_max_body_size 100M;
            proxy_connect_timeout 60s;
            proxy_send_timeout 300s;
            proxy_read_timeout 300s;
            
            proxy_pass http://scrollintel_backend;
            proxy_http_version 1.1;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # No caching for uploads
            proxy_cache off;
        }}
        
        # WebSocket support
        location /ws {{
            proxy_pass http://scrollintel_backend;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # WebSocket specific timeouts
            proxy_read_timeout 86400s;
            proxy_send_timeout 86400s;
            
            # Disable caching
            proxy_cache off;
        }}
        
        # Health check endpoint
        location /health {{
            proxy_pass http://scrollintel_backend/health;
            access_log off;
            
            # Quick health check timeouts
            proxy_connect_timeout 5s;
            proxy_send_timeout 5s;
            proxy_read_timeout 5s;
        }}
        
        # Static files and frontend with caching
        location / {{
            proxy_pass http://scrollintel_frontend;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection 'upgrade';
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            proxy_cache_bypass $http_upgrade;
            
            # Cache static assets aggressively
            location ~* \\.(js|css|png|jpg|jpeg|gif|ico|svg|woff|woff2|ttf|eot)$ {{
                expires 1y;
                add_header Cache-Control "public, immutable";
                proxy_pass http://scrollintel_frontend;
                
                # Cache in nginx
                proxy_cache scrollintel_cache;
                proxy_cache_valid 200 1y;
                proxy_cache_use_stale error timeout updating;
            }}
        }}
        
        # Error pages
        error_page 404 /404.html;
        error_page 500 502 503 504 /50x.html;
        
        location = /50x.html {{
            root /usr/share/nginx/html;
        }}
        
        # Maintenance mode
        location @maintenance {{
            root /usr/share/nginx/html;
            try_files /maintenance.html =503;
        }}
    }}
    
    # SSL/HTTPS configuration
    server {{
        listen 443 ssl http2;
        listen [::]:443 ssl http2;
        server_name scrollintel.com *.scrollintel.com;
        
        # SSL certificates
        ssl_certificate /etc/nginx/ssl/scrollintel.crt;
        ssl_certificate_key /etc/nginx/ssl/scrollintel.key;
        
        # SSL configuration
        ssl_protocols TLSv1.2 TLSv1.3;
        ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384;
        ssl_prefer_server_ciphers off;
        ssl_session_cache shared:SSL:10m;
        ssl_session_timeout 10m;
        ssl_session_tickets off;
        
        # OCSP stapling
        ssl_stapling on;
        ssl_stapling_verify on;
        
        # Security headers
        add_header Strict-Transport-Security "max-age=31536000; includeSubDomains; preload" always;
        
        # Include the same location blocks as HTTP server
        include /etc/nginx/conf.d/scrollintel-locations.conf;
    }}
}}
"""
        
        # Write nginx configuration
        os.makedirs('./nginx/conf.d', exist_ok=True)
        
        with open('./nginx/nginx-lb.conf', 'w') as f:
            f.write(nginx_config)
        
        logger.info("Advanced nginx load balancer configuration created")

    def create_backend_compose_config(self):
        """Create Docker Compose configuration for multiple backend instances"""
        logger.info("Creating Docker Compose configuration for load-balanced backends...")
        
        services = {}
        
        # Create multiple backend instances
        for i in range(self.config['backend_instances']):
            instance_name = f"backend-{i+1}"
            port = self.backend_ports[i]
            
            services[instance_name] = {
                'build': {
                    'context': '.',
                    'dockerfile': 'Dockerfile',
                    'target': 'production'
                },
                'container_name': f'scrollintel-{instance_name}',
                'environment': [
                    'ENVIRONMENT=production',
                    'DEBUG=false',
                    'LOG_LEVEL=WARNING',
                    'POSTGRES_HOST=postgres-master',
                    'POSTGRES_PORT=5432',
                    f'POSTGRES_DB={os.getenv("POSTGRES_DB", "scrollintel")}',
                    f'POSTGRES_USER={os.getenv("POSTGRES_USER", "postgres")}',
                    f'POSTGRES_PASSWORD={os.getenv("POSTGRES_PASSWORD")}',
                    'REDIS_HOST=redis',
                    'REDIS_PORT=6379',
                    f'JWT_SECRET_KEY={os.getenv("JWT_SECRET_KEY")}',
                    f'OPENAI_API_KEY={os.getenv("OPENAI_API_KEY")}',
                    f'INSTANCE_ID={i+1}',
                    f'INSTANCE_NAME={instance_name}'
                ],
                'volumes': [
                    './models:/app/models:ro',
                    './uploads:/app/uploads',
                    f'./logs/{instance_name}:/app/logs'
                ],
                'networks': ['scrollintel-lb-network'],
                'depends_on': {
                    'postgres-master': {'condition': 'service_healthy'},
                    'redis': {'condition': 'service_healthy'}
                },
                'healthcheck': {
                    'test': ['CMD', 'curl', '-f', 'http://localhost:8000/health'],
                    'interval': '30s',
                    'timeout': '10s',
                    'retries': 3,
                    'start_period': '60s'
                },
                'restart': 'unless-stopped',
                'deploy': {
                    'resources': {
                        'limits': {
                            'memory': '1G',
                            'cpus': '0.5'
                        }
                    }
                },
                'ports': [f'{port}:8000']
            }
        
        # Create frontend instances
        for i in range(2):  # 2 frontend instances
            instance_name = f"frontend-{i+1}"
            
            services[instance_name] = {
                'build': {
                    'context': './frontend',
                    'dockerfile': 'Dockerfile'
                },
                'container_name': f'scrollintel-{instance_name}',
                'environment': [
                    'NODE_ENV=production',
                    'NEXT_PUBLIC_API_URL=http://nginx:80'
                ],
                'networks': ['scrollintel-lb-network'],
                'healthcheck': {
                    'test': ['CMD', 'curl', '-f', 'http://localhost:3000'],
                    'interval': '30s',
                    'timeout': '10s',
                    'retries': 3
                },
                'restart': 'unless-stopped',
                'deploy': {
                    'resources': {
                        'limits': {
                            'memory': '512M',
                            'cpus': '0.25'
                        }
                    }
                }
            }
        
        # Add nginx load balancer
        services['nginx'] = {
            'image': 'nginx:alpine',
            'container_name': 'scrollintel-nginx-lb',
            'ports': [
                '80:80',
                '443:443',
                '8080:8080'  # Health check port
            ],
            'volumes': [
                './nginx/nginx-lb.conf:/etc/nginx/nginx.conf:ro',
                './nginx/ssl:/etc/nginx/ssl:ro',
                './logs/nginx:/var/log/nginx',
                'nginx_cache:/var/cache/nginx'
            ],
            'networks': ['scrollintel-lb-network'],
            'depends_on': [f'backend-{i+1}' for i in range(self.config['backend_instances'])] + 
                          ['frontend-1', 'frontend-2'],
            'healthcheck': {
                'test': ['CMD', 'curl', '-f', 'http://localhost:8080/health'],
                'interval': '30s',
                'timeout': '10s',
                'retries': 3
            },
            'restart': 'unless-stopped'
        }
        
        # Complete compose configuration
        compose_config = {
            'version': '3.8',
            'services': services,
            'volumes': {
                'nginx_cache': None
            },
            'networks': {
                'scrollintel-lb-network': {
                    'driver': 'bridge'
                }
            }
        }
        
        with open('docker-compose.load-balanced.yml', 'w') as f:
            yaml.dump(compose_config, f, default_flow_style=False)
        
        logger.info("Load-balanced Docker Compose configuration created")

    def create_health_check_script(self):
        """Create health check monitoring script"""
        logger.info("Creating health check monitoring script...")
        
        health_check_script = f"""#!/usr/bin/env python3
\"\"\"
ScrollIntel Load Balancer Health Check Monitor
Monitors backend health and provides detailed status information
\"\"\"

import requests
import json
import time
from datetime import datetime
from typing import Dict, List

class HealthCheckMonitor:
    def __init__(self):
        self.backend_ports = {list(self.backend_ports)}
        self.frontend_ports = [3000, 3001]
        self.nginx_status_url = "http://localhost:8080"
        
    def check_backend_health(self, port: int) -> Dict:
        \"\"\"Check health of a backend instance\"\"\"
        try:
            start_time = time.time()
            response = requests.get(f"http://localhost:{{port}}/health", timeout=5)
            response_time = (time.time() - start_time) * 1000
            
            return {{
                'port': port,
                'status': 'healthy' if response.status_code == 200 else 'unhealthy',
                'response_time': round(response_time, 2),
                'status_code': response.status_code,
                'timestamp': datetime.now().isoformat()
            }}
        except requests.RequestException as e:
            return {{
                'port': port,
                'status': 'unhealthy',
                'response_time': None,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }}
    
    def check_nginx_status(self) -> Dict:
        \"\"\"Check nginx load balancer status\"\"\"
        try:
            # Get nginx status
            response = requests.get(f"{{self.nginx_status_url}}/nginx_status", timeout=5)
            status_text = response.text
            
            # Parse nginx status
            lines = status_text.strip().split('\\n')
            active_connections = int(lines[0].split(':')[1].strip())
            
            # Get upstream status
            upstream_response = requests.get(f"{{self.nginx_status_url}}/upstream_check", timeout=5)
            
            return {{
                'status': 'healthy',
                'active_connections': active_connections,
                'upstream_status': upstream_response.text,
                'timestamp': datetime.now().isoformat()
            }}
        except requests.RequestException as e:
            return {{
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }}
    
    def run_health_checks(self) -> Dict:
        \"\"\"Run comprehensive health checks\"\"\"
        results = {{
            'timestamp': datetime.now().isoformat(),
            'backends': [],
            'nginx': None,
            'summary': {{
                'healthy_backends': 0,
                'total_backends': len(self.backend_ports),
                'nginx_healthy': False
            }}
        }}
        
        # Check all backend instances
        for port in self.backend_ports:
            backend_status = self.check_backend_health(port)
            results['backends'].append(backend_status)
            
            if backend_status['status'] == 'healthy':
                results['summary']['healthy_backends'] += 1
        
        # Check nginx
        nginx_status = self.check_nginx_status()
        results['nginx'] = nginx_status
        results['summary']['nginx_healthy'] = nginx_status['status'] == 'healthy'
        
        return results
    
    def print_status(self, results: Dict):
        \"\"\"Print formatted status report\"\"\"
        print(f"\\n=== ScrollIntel Load Balancer Health Check ===")
        print(f"Timestamp: {{results['timestamp']}}")
        print(f"\\nBackend Status ({{results['summary']['healthy_backends']}}/{{results['summary']['total_backends']}} healthy):")
        
        for backend in results['backends']:
            status_icon = "✅" if backend['status'] == 'healthy' else "❌"
            response_time = f"{{backend['response_time']}}ms" if backend.get('response_time') else "N/A"
            print(f"  {{status_icon}} Port {{backend['port']}}: {{backend['status']}} ({{response_time}})")
        
        nginx_icon = "✅" if results['summary']['nginx_healthy'] else "❌"
        print(f"\\nNginx Load Balancer: {{nginx_icon}} {{results['nginx']['status']}}")
        
        if results['nginx']['status'] == 'healthy':
            print(f"  Active connections: {{results['nginx'].get('active_connections', 'N/A')}}")

if __name__ == '__main__':
    monitor = HealthCheckMonitor()
    
    while True:
        try:
            results = monitor.run_health_checks()
            monitor.print_status(results)
            
            # Save results to file
            with open('./logs/health_check_results.json', 'w') as f:
                json.dump(results, f, indent=2)
            
            time.sleep(30)  # Check every 30 seconds
            
        except KeyboardInterrupt:
            print("\\nHealth check monitor stopped")
            break
        except Exception as e:
            print(f"Health check error: {{e}}")
            time.sleep(30)
"""
        
        with open('./scripts/health-check-monitor.py', 'w') as f:
            f.write(health_check_script)
        
        # Make script executable
        os.chmod('./scripts/health-check-monitor.py', 0o755)
        
        logger.info("Health check monitoring script created")

    def create_load_balancer_metrics(self):
        """Create Prometheus metrics configuration for load balancer"""
        logger.info("Creating load balancer metrics configuration...")
        
        prometheus_config = {
            'global': {
                'scrape_interval': '15s',
                'evaluation_interval': '15s'
            },
            'rule_files': [
                'alert_rules_lb.yml'
            ],
            'scrape_configs': [
                {
                    'job_name': 'nginx-lb',
                    'static_configs': [
                        {
                            'targets': ['nginx:8080']
                        }
                    ],
                    'metrics_path': '/nginx_status',
                    'scrape_interval': '30s'
                }
            ]
        }
        
        # Add backend instances to monitoring
        backend_targets = [f'backend-{i+1}:8000' for i in range(self.config['backend_instances'])]
        
        prometheus_config['scrape_configs'].append({
            'job_name': 'scrollintel-backends',
            'static_configs': [
                {
                    'targets': backend_targets
                }
            ],
            'metrics_path': '/metrics',
            'scrape_interval': '30s'
        })
        
        # Alert rules for load balancer
        alert_rules = {
            'groups': [
                {
                    'name': 'scrollintel-lb-alerts',
                    'rules': [
                        {
                            'alert': 'BackendDown',
                            'expr': 'up{job="scrollintel-backends"} == 0',
                            'for': '1m',
                            'labels': {
                                'severity': 'critical'
                            },
                            'annotations': {
                                'summary': 'ScrollIntel backend instance is down',
                                'description': 'Backend instance {{ $labels.instance }} has been down for more than 1 minute'
                            }
                        },
                        {
                            'alert': 'HighResponseTime',
                            'expr': 'nginx_http_request_duration_seconds{quantile="0.95"} > 2',
                            'for': '5m',
                            'labels': {
                                'severity': 'warning'
                            },
                            'annotations': {
                                'summary': 'High response time detected',
                                'description': '95th percentile response time is {{ $value }}s'
                            }
                        },
                        {
                            'alert': 'LoadBalancerDown',
                            'expr': 'up{job="nginx-lb"} == 0',
                            'for': '30s',
                            'labels': {
                                'severity': 'critical'
                            },
                            'annotations': {
                                'summary': 'Load balancer is down',
                                'description': 'Nginx load balancer has been down for more than 30 seconds'
                            }
                        }
                    ]
                }
            ]
        }
        
        os.makedirs('./monitoring', exist_ok=True)
        
        with open('./monitoring/prometheus-lb.yml', 'w') as f:
            yaml.dump(prometheus_config, f, default_flow_style=False)
        
        with open('./monitoring/alert_rules_lb.yml', 'w') as f:
            yaml.dump(alert_rules, f, default_flow_style=False)
        
        logger.info("Load balancer metrics configuration created")

    def setup_load_balancer(self):
        """Complete load balancer setup"""
        logger.info("Setting up advanced load balancer configuration...")
        
        try:
            # Step 1: Create nginx configuration
            self.create_nginx_config()
            
            # Step 2: Create Docker Compose configuration
            self.create_backend_compose_config()
            
            # Step 3: Create health check monitoring
            self.create_health_check_script()
            
            # Step 4: Create metrics configuration
            self.create_load_balancer_metrics()
            
            # Step 5: Create necessary directories
            os.makedirs('./logs/nginx', exist_ok=True)
            for i in range(self.config['backend_instances']):
                os.makedirs(f'./logs/backend-{i+1}', exist_ok=True)
            
            logger.info("✅ Load balancer setup completed successfully!")
            logger.info(f"Configuration created for {self.config['backend_instances']} backend instances")
            logger.info("Next steps:")
            logger.info("1. Start the load-balanced environment: docker-compose -f docker-compose.load-balanced.yml up -d")
            logger.info("2. Monitor health: python scripts/health-check-monitor.py")
            logger.info("3. Check nginx status: http://localhost:8080/nginx_status")
            logger.info("4. Check upstream status: http://localhost:8080/upstream_check")
            
            return True
            
        except Exception as e:
            logger.error(f"Load balancer setup failed: {e}")
            return False

if __name__ == '__main__':
    manager = LoadBalancerManager()
    success = manager.setup_load_balancer()
    sys.exit(0 if success else 1)