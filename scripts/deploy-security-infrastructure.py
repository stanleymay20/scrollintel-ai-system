#!/usr/bin/env python3

"""
Security Infrastructure Deployment Script
Deploys SSL, CDN, and security infrastructure for ScrollIntel production
"""

import os
import sys
import subprocess
import json
from pathlib import Path
from typing import Dict, List, Optional
import argparse

class SecurityInfrastructureDeployer:
    """Deploys security infrastructure components"""
    
    def __init__(self, environment: str = "production"):
        self.environment = environment
        self.project_root = Path(__file__).parent.parent
        self.scripts_dir = self.project_root / "scripts"
        self.nginx_dir = self.project_root / "nginx"
        
    def log(self, message: str) -> None:
        """Log deployment message"""
        print(f"🚀 {message}")
    
    def run_command(self, command: List[str], check: bool = True) -> subprocess.CompletedProcess:
        """Run shell command"""
        self.log(f"Running: {' '.join(command)}")
        return subprocess.run(command, check=check, capture_output=True, text=True)
    
    def setup_ssl_certificates(self) -> None:
        """Setup SSL certificates"""
        self.log("Setting up SSL certificates...")
        
        ssl_script = self.scripts_dir / "setup-ssl-certificates.sh"
        
        if self.environment == "production":
            self.run_command(["bash", str(ssl_script), "production"])
        else:
            self.run_command(["bash", str(ssl_script), "development"])
        
        self.log("✅ SSL certificates configured")
    
    def configure_nginx_security(self) -> None:
        """Configure Nginx security settings"""
        self.log("Configuring Nginx security...")
        
        # Copy security configuration files
        nginx_conf_dir = Path("/etc/nginx")
        nginx_conf_d = nginx_conf_dir / "conf.d"
        
        # Create directories if they don't exist
        nginx_conf_d.mkdir(parents=True, exist_ok=True)
        
        # Copy configuration files
        config_files = [
            "ssl-config.conf",
            "security-headers.conf",
            "cdn-config.conf"
        ]
        
        for config_file in config_files:
            source = self.nginx_dir / config_file
            dest = nginx_conf_dir / config_file
            
            if source.exists():
                self.run_command(["cp", str(source), str(dest)])
                self.log(f"Copied {config_file}")
        
        # Copy main nginx configuration
        main_config = self.nginx_dir / "nginx.conf"
        if main_config.exists():
            self.run_command(["cp", str(main_config), "/etc/nginx/nginx.conf"])
            self.log("Updated main nginx configuration")
        
        # Test nginx configuration
        try:
            self.run_command(["nginx", "-t"])
            self.log("✅ Nginx configuration is valid")
        except subprocess.CalledProcessError as e:
            self.log(f"❌ Nginx configuration error: {e.stderr}")
            raise
    
    def setup_ddos_protection(self) -> None:
        """Setup DDoS protection"""
        self.log("Setting up DDoS protection...")
        
        ddos_script = self.scripts_dir / "setup-ddos-protection.py"
        self.run_command(["python3", str(ddos_script)])
        
        self.log("✅ DDoS protection configured")
    
    def setup_backup_system(self) -> None:
        """Setup backup and recovery system"""
        self.log("Setting up backup system...")
        
        backup_script = self.scripts_dir / "setup-backup-recovery.py"
        self.run_command(["python3", str(backup_script), "--setup"])
        
        self.log("✅ Backup system configured")
    
    def setup_cloudflare_cdn(self) -> None:
        """Setup Cloudflare CDN"""
        self.log("Setting up Cloudflare CDN...")
        
        # Check if Cloudflare credentials are available
        required_env_vars = [
            "CLOUDFLARE_API_TOKEN",
            "CLOUDFLARE_ZONE_ID",
            "CLOUDFLARE_EMAIL"
        ]
        
        missing_vars = [var for var in required_env_vars if not os.getenv(var)]
        
        if missing_vars:
            self.log(f"⚠️  Skipping Cloudflare setup - missing environment variables: {missing_vars}")
            return
        
        cloudflare_script = self.scripts_dir / "setup-cloudflare-cdn.py"
        try:
            self.run_command(["python3", str(cloudflare_script)])
            self.log("✅ Cloudflare CDN configured")
        except subprocess.CalledProcessError as e:
            self.log(f"⚠️  Cloudflare setup failed: {e.stderr}")
    
    def create_error_pages(self) -> None:
        """Create custom error pages"""
        self.log("Creating custom error pages...")
        
        error_pages_dir = Path("/usr/share/nginx/html")
        error_pages_dir.mkdir(parents=True, exist_ok=True)
        
        # 50x error page
        error_50x = """<!DOCTYPE html>
<html>
<head>
    <title>ScrollIntel - Service Temporarily Unavailable</title>
    <style>
        body { font-family: Arial, sans-serif; text-align: center; padding: 50px; }
        .error-container { max-width: 600px; margin: 0 auto; }
        .error-code { font-size: 72px; color: #e74c3c; margin-bottom: 20px; }
        .error-message { font-size: 24px; color: #333; margin-bottom: 30px; }
        .error-description { font-size: 16px; color: #666; line-height: 1.6; }
        .back-link { margin-top: 30px; }
        .back-link a { color: #3498db; text-decoration: none; }
    </style>
</head>
<body>
    <div class="error-container">
        <div class="error-code">503</div>
        <div class="error-message">Service Temporarily Unavailable</div>
        <div class="error-description">
            We're currently performing maintenance on our servers. 
            Please try again in a few minutes.
        </div>
        <div class="back-link">
            <a href="/">← Back to ScrollIntel</a>
        </div>
    </div>
</body>
</html>"""
        
        # Rate limit error page
        rate_limit_page = """<!DOCTYPE html>
<html>
<head>
    <title>ScrollIntel - Rate Limit Exceeded</title>
    <style>
        body { font-family: Arial, sans-serif; text-align: center; padding: 50px; }
        .error-container { max-width: 600px; margin: 0 auto; }
        .error-code { font-size: 72px; color: #f39c12; margin-bottom: 20px; }
        .error-message { font-size: 24px; color: #333; margin-bottom: 30px; }
        .error-description { font-size: 16px; color: #666; line-height: 1.6; }
        .back-link { margin-top: 30px; }
        .back-link a { color: #3498db; text-decoration: none; }
    </style>
</head>
<body>
    <div class="error-container">
        <div class="error-code">429</div>
        <div class="error-message">Too Many Requests</div>
        <div class="error-description">
            You've exceeded the rate limit for this service. 
            Please wait a moment before trying again.
        </div>
        <div class="back-link">
            <a href="/">← Back to ScrollIntel</a>
        </div>
    </div>
</body>
</html>"""
        
        with open(error_pages_dir / "50x.html", 'w') as f:
            f.write(error_50x)
        
        with open(error_pages_dir / "rate-limit.html", 'w') as f:
            f.write(rate_limit_page)
        
        self.log("✅ Custom error pages created")
    
    def setup_monitoring(self) -> None:
        """Setup security monitoring"""
        self.log("Setting up security monitoring...")
        
        # Create monitoring script
        monitoring_script = """#!/bin/bash

# Security monitoring script
LOG_FILE="/var/log/security-monitor.log"

# Function to log with timestamp
log_message() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" >> $LOG_FILE
}

# Monitor SSL certificate expiry
check_ssl_expiry() {
    local domain="scrollintel.com"
    local expiry_date=$(echo | openssl s_client -servername $domain -connect $domain:443 2>/dev/null | openssl x509 -noout -dates | grep notAfter | cut -d= -f2)
    local expiry_epoch=$(date -d "$expiry_date" +%s)
    local current_epoch=$(date +%s)
    local days_until_expiry=$(( (expiry_epoch - current_epoch) / 86400 ))
    
    if [ $days_until_expiry -lt 30 ]; then
        log_message "WARNING: SSL certificate expires in $days_until_expiry days"
        # Send alert (implement notification system)
    fi
}

# Monitor failed login attempts
check_failed_logins() {
    local failed_count=$(grep "authentication failure" /var/log/auth.log | grep "$(date '+%Y-%m-%d')" | wc -l)
    
    if [ $failed_count -gt 50 ]; then
        log_message "WARNING: High number of failed login attempts: $failed_count"
    fi
}

# Monitor disk space for backups
check_backup_space() {
    local backup_usage=$(df /var/backups | tail -1 | awk '{print $5}' | sed 's/%//')
    
    if [ $backup_usage -gt 80 ]; then
        log_message "WARNING: Backup disk usage is at ${backup_usage}%"
    fi
}

# Run checks
check_ssl_expiry
check_failed_logins
check_backup_space

log_message "Security monitoring check completed"
"""
        
        monitoring_script_path = Path("/usr/local/bin/security-monitor.sh")
        with open(monitoring_script_path, 'w') as f:
            f.write(monitoring_script)
        
        os.chmod(monitoring_script_path, 0o755)
        
        # Add to cron
        cron_entry = "0 */6 * * * /usr/local/bin/security-monitor.sh"
        
        try:
            # Add cron job
            current_cron = subprocess.run(
                ["crontab", "-l"], 
                capture_output=True, 
                text=True
            ).stdout
            
            if cron_entry not in current_cron:
                new_cron = current_cron + f"\n{cron_entry}\n"
                subprocess.run(
                    ["crontab", "-"],
                    input=new_cron,
                    text=True,
                    check=True
                )
        except subprocess.CalledProcessError:
            # No existing crontab
            subprocess.run(
                ["crontab", "-"],
                input=f"{cron_entry}\n",
                text=True,
                check=True
            )
        
        self.log("✅ Security monitoring configured")
    
    def restart_services(self) -> None:
        """Restart necessary services"""
        self.log("Restarting services...")
        
        services = ["nginx", "fail2ban"]
        
        for service in services:
            try:
                self.run_command(["systemctl", "restart", service])
                self.run_command(["systemctl", "enable", service])
                self.log(f"✅ {service} restarted and enabled")
            except subprocess.CalledProcessError as e:
                self.log(f"⚠️  Failed to restart {service}: {e.stderr}")
        
        # Start DDoS monitoring if available
        try:
            self.run_command(["systemctl", "start", "ddos-monitor"])
            self.run_command(["systemctl", "enable", "ddos-monitor"])
            self.log("✅ DDoS monitoring started")
        except subprocess.CalledProcessError:
            self.log("⚠️  DDoS monitoring service not available")
    
    def validate_deployment(self) -> None:
        """Validate the security infrastructure deployment"""
        self.log("Validating deployment...")
        
        validation_script = self.scripts_dir / "validate-security-infrastructure.py"
        
        try:
            result = self.run_command(["python3", str(validation_script)], check=False)
            
            if result.returncode == 0:
                self.log("✅ Security infrastructure validation PASSED")
            else:
                self.log("⚠️  Security infrastructure validation had issues")
                self.log("Check the validation report for details")
        
        except Exception as e:
            self.log(f"⚠️  Validation failed: {e}")
    
    def deploy(self) -> None:
        """Deploy complete security infrastructure"""
        self.log("Starting security infrastructure deployment...")
        self.log(f"Environment: {self.environment}")
        
        try:
            # Setup SSL certificates
            self.setup_ssl_certificates()
            
            # Configure Nginx security
            self.configure_nginx_security()
            
            # Setup DDoS protection
            self.setup_ddos_protection()
            
            # Setup backup system
            self.setup_backup_system()
            
            # Setup Cloudflare CDN (if configured)
            self.setup_cloudflare_cdn()
            
            # Create error pages
            self.create_error_pages()
            
            # Setup monitoring
            self.setup_monitoring()
            
            # Restart services
            self.restart_services()
            
            # Validate deployment
            self.validate_deployment()
            
            self.log("🎉 Security infrastructure deployment completed successfully!")
            
            # Print next steps
            print("\n" + "="*60)
            print("🔒 SECURITY INFRASTRUCTURE DEPLOYED")
            print("="*60)
            print("Next steps:")
            print("1. Update DNS records to point to your server")
            print("2. Configure Cloudflare settings (if using)")
            print("3. Test SSL certificates and security headers")
            print("4. Monitor logs for any issues")
            print("5. Schedule regular security audits")
            print("\nImportant files:")
            print("- SSL certificates: /etc/nginx/ssl/")
            print("- Nginx config: /etc/nginx/nginx.conf")
            print("- Backup config: /etc/scrollintel/backup-config.json")
            print("- Security logs: /var/log/security-monitor.log")
            print("="*60)
        
        except Exception as e:
            self.log(f"❌ Deployment failed: {e}")
            sys.exit(1)

def main():
    """Main deployment function"""
    parser = argparse.ArgumentParser(description="Deploy ScrollIntel security infrastructure")
    parser.add_argument(
        "--environment",
        choices=["development", "production"],
        default="production",
        help="Deployment environment"
    )
    parser.add_argument(
        "--skip-ssl",
        action="store_true",
        help="Skip SSL certificate setup"
    )
    parser.add_argument(
        "--skip-cloudflare",
        action="store_true",
        help="Skip Cloudflare CDN setup"
    )
    
    args = parser.parse_args()
    
    # Check if running as root
    if os.geteuid() != 0:
        print("❌ This script must be run as root")
        sys.exit(1)
    
    deployer = SecurityInfrastructureDeployer(args.environment)
    deployer.deploy()

if __name__ == "__main__":
    main()