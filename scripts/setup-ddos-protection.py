#!/usr/bin/env python3

"""
DDoS Protection and Rate Limiting Setup
Configures advanced DDoS protection for ScrollIntel
"""

import os
import sys
import json
import subprocess
from typing import Dict, List
from pathlib import Path

class DDoSProtectionManager:
    """Manages DDoS protection configuration"""
    
    def __init__(self):
        self.nginx_conf_dir = Path("/etc/nginx/conf.d")
        self.fail2ban_dir = Path("/etc/fail2ban")
        
    def create_rate_limiting_config(self) -> None:
        """Create advanced rate limiting configuration"""
        print("🛡️  Setting up rate limiting...")
        
        rate_limit_config = """
# ================================
# Advanced Rate Limiting Configuration
# ================================

# Define rate limiting zones
limit_req_zone $binary_remote_addr zone=general:10m rate=10r/s;
limit_req_zone $binary_remote_addr zone=api:10m rate=5r/s;
limit_req_zone $binary_remote_addr zone=auth:10m rate=2r/s;
limit_req_zone $binary_remote_addr zone=upload:10m rate=1r/s;
limit_req_zone $binary_remote_addr zone=search:10m rate=3r/s;

# Connection limiting
limit_conn_zone $binary_remote_addr zone=conn_limit_per_ip:10m;
limit_conn_zone $server_name zone=conn_limit_per_server:10m;

# Request body size limits
client_max_body_size 100M;
client_body_buffer_size 128k;
client_header_buffer_size 3m;
large_client_header_buffers 4 256k;

# Timeout configurations
client_body_timeout 60s;
client_header_timeout 60s;
keepalive_timeout 65s;
send_timeout 60s;

# Slow loris protection
client_body_timeout 10s;
client_header_timeout 10s;
keepalive_timeout 5s 5s;
send_timeout 10s;

# Buffer overflow protection
client_body_buffer_size 1k;
client_header_buffer_size 1k;
client_max_body_size 100M;
large_client_header_buffers 2 1k;
"""
        
        config_file = self.nginx_conf_dir / "rate-limiting.conf"
        with open(config_file, 'w') as f:
            f.write(rate_limit_config)
        
        print(f"✅ Rate limiting config created: {config_file}")
    
    def create_geo_blocking_config(self) -> None:
        """Create geo-blocking configuration"""
        print("🌍 Setting up geo-blocking...")
        
        geo_config = """
# ================================
# Geo-blocking Configuration
# ================================

# Define allowed countries (ISO 3166-1 alpha-2)
geo $allowed_country {
    default 1;
    
    # Block known problematic countries (adjust as needed)
    CN 0;  # China
    RU 0;  # Russia
    KP 0;  # North Korea
    IR 0;  # Iran
    
    # Allow specific countries
    US 1;  # United States
    CA 1;  # Canada
    GB 1;  # United Kingdom
    DE 1;  # Germany
    FR 1;  # France
    JP 1;  # Japan
    AU 1;  # Australia
    NL 1;  # Netherlands
    SE 1;  # Sweden
    NO 1;  # Norway
    DK 1;  # Denmark
    FI 1;  # Finland
    CH 1;  # Switzerland
    AT 1;  # Austria
    BE 1;  # Belgium
    IT 1;  # Italy
    ES 1;  # Spain
    PT 1;  # Portugal
    IE 1;  # Ireland
    NZ 1;  # New Zealand
    SG 1;  # Singapore
    HK 1;  # Hong Kong
    KR 1;  # South Korea
    TW 1;  # Taiwan
    IN 1;  # India
    BR 1;  # Brazil
    MX 1;  # Mexico
    AR 1;  # Argentina
    CL 1;  # Chile
    CO 1;  # Colombia
    PE 1;  # Peru
    ZA 1;  # South Africa
    EG 1;  # Egypt
    IL 1;  # Israel
    AE 1;  # UAE
    SA 1;  # Saudi Arabia
    TR 1;  # Turkey
    GR 1;  # Greece
    PL 1;  # Poland
    CZ 1;  # Czech Republic
    HU 1;  # Hungary
    RO 1;  # Romania
    BG 1;  # Bulgaria
    HR 1;  # Croatia
    SI 1;  # Slovenia
    SK 1;  # Slovakia
    LT 1;  # Lithuania
    LV 1;  # Latvia
    EE 1;  # Estonia
    IS 1;  # Iceland
    LU 1;  # Luxembourg
    MT 1;  # Malta
    CY 1;  # Cyprus
}

# Block requests from disallowed countries
map $allowed_country $blocked_country {
    0 1;
    1 0;
}
"""
        
        config_file = self.nginx_conf_dir / "geo-blocking.conf"
        with open(config_file, 'w') as f:
            f.write(geo_config)
        
        print(f"✅ Geo-blocking config created: {config_file}")
    
    def create_bot_protection_config(self) -> None:
        """Create bot protection configuration"""
        print("🤖 Setting up bot protection...")
        
        bot_config = """
# ================================
# Bot Protection Configuration
# ================================

# Define bad bots
map $http_user_agent $bad_bot {
    default 0;
    ~*bot 1;
    ~*crawler 1;
    ~*spider 1;
    ~*scraper 1;
    ~*scanner 1;
    ~*grabber 1;
    ~*extractor 1;
    ~*harvester 1;
    ~*collector 1;
    ~*siphon 1;
    ~*sucker 1;
    ~*leech 1;
    ~*vampire 1;
    ~*libwww 1;
    ~*curl 1;
    ~*wget 1;
    ~*python 1;
    ~*perl 1;
    ~*ruby 1;
    ~*java 1;
    ~*go-http 1;
    ~*okhttp 1;
    ~*httpclient 1;
    ~*requests 1;
    ~*urllib 1;
    
    # Allow legitimate bots
    ~*googlebot 0;
    ~*bingbot 0;
    ~*slurp 0;
    ~*duckduckbot 0;
    ~*baiduspider 0;
    ~*yandexbot 0;
    ~*facebookexternalhit 0;
    ~*twitterbot 0;
    ~*linkedinbot 0;
    ~*whatsapp 0;
    ~*telegrambot 0;
    ~*applebot 0;
}

# Define suspicious patterns
map $request_uri $suspicious_request {
    default 0;
    ~*\.(php|asp|aspx|jsp|cgi)$ 1;
    ~*/wp- 1;
    ~*/admin 1;
    ~*/phpmyadmin 1;
    ~*/xmlrpc 1;
    ~*\.env 1;
    ~*\.git 1;
    ~*\.svn 1;
    ~*\.htaccess 1;
    ~*\.htpasswd 1;
    ~*/config 1;
    ~*/backup 1;
    ~*/sql 1;
    ~*/database 1;
    ~*eval\( 1;
    ~*base64_ 1;
    ~*script 1;
    ~*union.*select 1;
    ~*concat.*\( 1;
    ~*\.\./\.\. 1;
    ~*proc/self/environ 1;
}

# Rate limiting for bots
limit_req_zone $binary_remote_addr zone=bots:10m rate=1r/m;

# Apply bot protection
if ($bad_bot) {
    return 444;
}

if ($suspicious_request) {
    return 444;
}
"""
        
        config_file = self.nginx_conf_dir / "bot-protection.conf"
        with open(config_file, 'w') as f:
            f.write(bot_config)
        
        print(f"✅ Bot protection config created: {config_file}")
    
    def setup_fail2ban(self) -> None:
        """Setup Fail2Ban for intrusion prevention"""
        print("🚫 Setting up Fail2Ban...")
        
        # Install fail2ban if not present
        try:
            subprocess.run(["which", "fail2ban-server"], check=True, capture_output=True)
        except subprocess.CalledProcessError:
            print("Installing Fail2Ban...")
            if os.path.exists("/usr/bin/apt-get"):
                subprocess.run(["apt-get", "update"], check=True)
                subprocess.run(["apt-get", "install", "-y", "fail2ban"], check=True)
            elif os.path.exists("/usr/bin/yum"):
                subprocess.run(["yum", "install", "-y", "fail2ban"], check=True)
        
        # Create custom jail configuration
        jail_config = """
[DEFAULT]
bantime = 3600
findtime = 600
maxretry = 5
backend = auto
usedns = warn
logencoding = auto
enabled = false
mode = normal
filter = %(__name__)s[mode=%(mode)s]

[sshd]
enabled = true
port = ssh
logpath = /var/log/auth.log
maxretry = 3
bantime = 86400

[nginx-http-auth]
enabled = true
port = http,https
logpath = /var/log/nginx/error.log
maxretry = 3

[nginx-limit-req]
enabled = true
port = http,https
logpath = /var/log/nginx/error.log
maxretry = 10
findtime = 600
bantime = 7200

[nginx-botsearch]
enabled = true
port = http,https
logpath = /var/log/nginx/access.log
maxretry = 2
bantime = 86400

[scrollintel-api]
enabled = true
port = http,https
logpath = /var/log/nginx/access.log
filter = scrollintel-api
maxretry = 10
findtime = 300
bantime = 3600
"""
        
        jail_file = self.fail2ban_dir / "jail.d" / "scrollintel.conf"
        jail_file.parent.mkdir(parents=True, exist_ok=True)
        with open(jail_file, 'w') as f:
            f.write(jail_config)
        
        # Create custom filter for API abuse
        filter_config = """
[Definition]
failregex = ^<HOST> -.*"(GET|POST|PUT|DELETE) /api/.*" (4[0-9][0-9]|5[0-9][0-9]) .*$
            ^<HOST> -.*"(GET|POST|PUT|DELETE) /api/auth/.*" (401|403) .*$
ignoreregex =
"""
        
        filter_file = self.fail2ban_dir / "filter.d" / "scrollintel-api.conf"
        filter_file.parent.mkdir(parents=True, exist_ok=True)
        with open(filter_file, 'w') as f:
            f.write(filter_config)
        
        print("✅ Fail2Ban configuration created")
    
    def create_monitoring_script(self) -> None:
        """Create DDoS monitoring script"""
        print("📊 Creating DDoS monitoring script...")
        
        monitoring_script = """#!/bin/bash

# ================================
# DDoS Monitoring Script
# ================================

LOG_FILE="/var/log/nginx/access.log"
ALERT_THRESHOLD=1000
TIME_WINDOW=60

# Function to send alert
send_alert() {
    local message="$1"
    echo "$(date): $message" >> /var/log/ddos-monitor.log
    
    # Send email alert (configure SMTP)
    # echo "$message" | mail -s "DDoS Alert - ScrollIntel" admin@scrollintel.com
    
    # Send to monitoring system
    curl -X POST "https://hooks.slack.com/YOUR_WEBHOOK" \
        -H 'Content-type: application/json' \
        --data "{\"text\":\"🚨 DDoS Alert: $message\"}" 2>/dev/null || true
}

# Monitor request rate
monitor_requests() {
    local current_time=$(date +%s)
    local start_time=$((current_time - TIME_WINDOW))
    
    # Count requests in the last minute
    local request_count=$(awk -v start="$start_time" '
        {
            # Parse timestamp from log
            gsub(/\[|\]/, "", $4)
            cmd = "date -d \"" $4 "\" +%s"
            cmd | getline timestamp
            close(cmd)
            
            if (timestamp >= start) {
                count++
            }
        }
        END { print count+0 }
    ' "$LOG_FILE")
    
    if [ "$request_count" -gt "$ALERT_THRESHOLD" ]; then
        send_alert "High request rate detected: $request_count requests in $TIME_WINDOW seconds"
    fi
}

# Monitor for suspicious patterns
monitor_patterns() {
    local suspicious_count=$(tail -n 1000 "$LOG_FILE" | grep -E "(bot|crawler|scanner|hack|exploit|injection)" | wc -l)
    
    if [ "$suspicious_count" -gt 50 ]; then
        send_alert "Suspicious activity detected: $suspicious_count suspicious requests in recent logs"
    fi
}

# Monitor error rates
monitor_errors() {
    local error_count=$(tail -n 1000 "$LOG_FILE" | grep -E " (4[0-9][0-9]|5[0-9][0-9]) " | wc -l)
    
    if [ "$error_count" -gt 100 ]; then
        send_alert "High error rate detected: $error_count errors in recent logs"
    fi
}

# Main monitoring loop
main() {
    while true; do
        monitor_requests
        monitor_patterns
        monitor_errors
        sleep 30
    done
}

# Run monitoring
main &
echo $! > /var/run/ddos-monitor.pid
"""
        
        script_file = Path("/usr/local/bin/ddos-monitor.sh")
        with open(script_file, 'w') as f:
            f.write(monitoring_script)
        
        # Make executable
        os.chmod(script_file, 0o755)
        
        print(f"✅ DDoS monitoring script created: {script_file}")
    
    def create_systemd_service(self) -> None:
        """Create systemd service for DDoS monitoring"""
        print("⚙️  Creating systemd service...")
        
        service_config = """[Unit]
Description=ScrollIntel DDoS Monitoring Service
After=network.target nginx.service
Requires=nginx.service

[Service]
Type=forking
ExecStart=/usr/local/bin/ddos-monitor.sh
PIDFile=/var/run/ddos-monitor.pid
Restart=always
RestartSec=10
User=root
Group=root

[Install]
WantedBy=multi-user.target
"""
        
        service_file = Path("/etc/systemd/system/ddos-monitor.service")
        with open(service_file, 'w') as f:
            f.write(service_config)
        
        # Reload systemd and enable service
        subprocess.run(["systemctl", "daemon-reload"], check=True)
        subprocess.run(["systemctl", "enable", "ddos-monitor"], check=True)
        
        print("✅ Systemd service created and enabled")

def main():
    """Main execution function"""
    print("🛡️  ScrollIntel DDoS Protection Setup")
    print("=" * 40)
    
    manager = DDoSProtectionManager()
    
    try:
        # Create rate limiting configuration
        manager.create_rate_limiting_config()
        
        # Create geo-blocking configuration
        manager.create_geo_blocking_config()
        
        # Create bot protection configuration
        manager.create_bot_protection_config()
        
        # Setup Fail2Ban
        manager.setup_fail2ban()
        
        # Create monitoring script
        manager.create_monitoring_script()
        
        # Create systemd service
        manager.create_systemd_service()
        
        print("\n✅ DDoS protection setup completed successfully!")
        print("🔄 Restart nginx and fail2ban services to apply changes:")
        print("   sudo systemctl restart nginx")
        print("   sudo systemctl restart fail2ban")
        print("   sudo systemctl start ddos-monitor")
        
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()