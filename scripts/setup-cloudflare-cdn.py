#!/usr/bin/env python3

"""
Cloudflare CDN Setup and Configuration Script
Automates Cloudflare setup for ScrollIntel production deployment
"""

import os
import sys
import json
import requests
import time
from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class CloudflareConfig:
    """Cloudflare configuration settings"""
    api_token: str
    zone_id: str
    domain: str
    email: str

class CloudflareManager:
    """Manages Cloudflare CDN configuration"""
    
    def __init__(self, config: CloudflareConfig):
        self.config = config
        self.base_url = "https://api.cloudflare.com/client/v4"
        self.headers = {
            "Authorization": f"Bearer {config.api_token}",
            "Content-Type": "application/json"
        }
    
    def _make_request(self, method: str, endpoint: str, data: Optional[Dict] = None) -> Dict:
        """Make API request to Cloudflare"""
        url = f"{self.base_url}{endpoint}"
        
        try:
            if method.upper() == "GET":
                response = requests.get(url, headers=self.headers)
            elif method.upper() == "POST":
                response = requests.post(url, headers=self.headers, json=data)
            elif method.upper() == "PUT":
                response = requests.put(url, headers=self.headers, json=data)
            elif method.upper() == "PATCH":
                response = requests.patch(url, headers=self.headers, json=data)
            elif method.upper() == "DELETE":
                response = requests.delete(url, headers=self.headers)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            response.raise_for_status()
            return response.json()
        
        except requests.exceptions.RequestException as e:
            print(f"❌ API request failed: {e}")
            sys.exit(1)
    
    def setup_dns_records(self) -> None:
        """Setup DNS records for the domain"""
        print("🌐 Setting up DNS records...")
        
        # DNS records to create
        dns_records = [
            {
                "type": "A",
                "name": "@",
                "content": "YOUR_SERVER_IP",  # Replace with actual server IP
                "ttl": 300,
                "proxied": True
            },
            {
                "type": "A",
                "name": "www",
                "content": "YOUR_SERVER_IP",  # Replace with actual server IP
                "ttl": 300,
                "proxied": True
            },
            {
                "type": "A",
                "name": "api",
                "content": "YOUR_SERVER_IP",  # Replace with actual server IP
                "ttl": 300,
                "proxied": True
            },
            {
                "type": "CNAME",
                "name": "cdn",
                "content": self.config.domain,
                "ttl": 300,
                "proxied": True
            }
        ]
        
        for record in dns_records:
            try:
                result = self._make_request(
                    "POST",
                    f"/zones/{self.config.zone_id}/dns_records",
                    record
                )
                print(f"✅ Created DNS record: {record['name']} -> {record['content']}")
            except Exception as e:
                print(f"⚠️  DNS record creation failed for {record['name']}: {e}")
    
    def configure_security_settings(self) -> None:
        """Configure Cloudflare security settings"""
        print("🔒 Configuring security settings...")
        
        security_settings = [
            # SSL/TLS settings
            {
                "endpoint": f"/zones/{self.config.zone_id}/settings/ssl",
                "value": "strict"
            },
            {
                "endpoint": f"/zones/{self.config.zone_id}/settings/always_use_https",
                "value": "on"
            },
            {
                "endpoint": f"/zones/{self.config.zone_id}/settings/min_tls_version",
                "value": "1.2"
            },
            {
                "endpoint": f"/zones/{self.config.zone_id}/settings/tls_1_3",
                "value": "on"
            },
            # Security settings
            {
                "endpoint": f"/zones/{self.config.zone_id}/settings/security_level",
                "value": "medium"
            },
            {
                "endpoint": f"/zones/{self.config.zone_id}/settings/browser_check",
                "value": "on"
            },
            {
                "endpoint": f"/zones/{self.config.zone_id}/settings/challenge_ttl",
                "value": 1800
            },
            # Performance settings
            {
                "endpoint": f"/zones/{self.config.zone_id}/settings/brotli",
                "value": "on"
            },
            {
                "endpoint": f"/zones/{self.config.zone_id}/settings/minify",
                "value": {
                    "css": "on",
                    "html": "on",
                    "js": "on"
                }
            }
        ]
        
        for setting in security_settings:
            try:
                data = {"value": setting["value"]}
                result = self._make_request("PATCH", setting["endpoint"], data)
                print(f"✅ Updated setting: {setting['endpoint'].split('/')[-1]}")
            except Exception as e:
                print(f"⚠️  Setting update failed: {e}")
    
    def setup_page_rules(self) -> None:
        """Setup Cloudflare page rules for optimization"""
        print("📄 Setting up page rules...")
        
        page_rules = [
            {
                "targets": [
                    {
                        "target": "url",
                        "constraint": {
                            "operator": "matches",
                            "value": f"{self.config.domain}/api/*"
                        }
                    }
                ],
                "actions": [
                    {"id": "cache_level", "value": "bypass"},
                    {"id": "security_level", "value": "high"}
                ],
                "priority": 1,
                "status": "active"
            },
            {
                "targets": [
                    {
                        "target": "url",
                        "constraint": {
                            "operator": "matches",
                            "value": f"{self.config.domain}/*.js"
                        }
                    }
                ],
                "actions": [
                    {"id": "cache_level", "value": "cache_everything"},
                    {"id": "edge_cache_ttl", "value": 31536000}  # 1 year
                ],
                "priority": 2,
                "status": "active"
            },
            {
                "targets": [
                    {
                        "target": "url",
                        "constraint": {
                            "operator": "matches",
                            "value": f"{self.config.domain}/*.css"
                        }
                    }
                ],
                "actions": [
                    {"id": "cache_level", "value": "cache_everything"},
                    {"id": "edge_cache_ttl", "value": 31536000}  # 1 year
                ],
                "priority": 3,
                "status": "active"
            }
        ]
        
        for rule in page_rules:
            try:
                result = self._make_request(
                    "POST",
                    f"/zones/{self.config.zone_id}/pagerules",
                    rule
                )
                print(f"✅ Created page rule with priority {rule['priority']}")
            except Exception as e:
                print(f"⚠️  Page rule creation failed: {e}")
    
    def setup_firewall_rules(self) -> None:
        """Setup Cloudflare firewall rules"""
        print("🛡️  Setting up firewall rules...")
        
        firewall_rules = [
            {
                "filter": {
                    "expression": "(http.request.uri.path contains \"/admin\" and not ip.src in {YOUR_ADMIN_IPS})",
                    "paused": False
                },
                "action": "block",
                "priority": 1,
                "description": "Block admin access from unauthorized IPs"
            },
            {
                "filter": {
                    "expression": "(http.request.method eq \"POST\" and http.request.uri.path contains \"/api/auth\" and cf.threat_score gt 10)",
                    "paused": False
                },
                "action": "challenge",
                "priority": 2,
                "description": "Challenge suspicious auth requests"
            },
            {
                "filter": {
                    "expression": "(rate(5m) > 100)",
                    "paused": False
                },
                "action": "challenge",
                "priority": 3,
                "description": "Rate limiting - challenge high request rates"
            }
        ]
        
        for rule in firewall_rules:
            try:
                # Create filter first
                filter_result = self._make_request(
                    "POST",
                    f"/zones/{self.config.zone_id}/filters",
                    rule["filter"]
                )
                
                # Create firewall rule
                firewall_rule = {
                    "filter": {"id": filter_result["result"]["id"]},
                    "action": rule["action"],
                    "priority": rule["priority"],
                    "description": rule["description"]
                }
                
                result = self._make_request(
                    "POST",
                    f"/zones/{self.config.zone_id}/firewall/rules",
                    firewall_rule
                )
                print(f"✅ Created firewall rule: {rule['description']}")
            except Exception as e:
                print(f"⚠️  Firewall rule creation failed: {e}")
    
    def purge_cache(self) -> None:
        """Purge Cloudflare cache"""
        print("🧹 Purging Cloudflare cache...")
        
        try:
            result = self._make_request(
                "POST",
                f"/zones/{self.config.zone_id}/purge_cache",
                {"purge_everything": True}
            )
            print("✅ Cache purged successfully")
        except Exception as e:
            print(f"⚠️  Cache purge failed: {e}")

def load_config() -> CloudflareConfig:
    """Load configuration from environment variables"""
    api_token = os.getenv("CLOUDFLARE_API_TOKEN")
    zone_id = os.getenv("CLOUDFLARE_ZONE_ID")
    domain = os.getenv("DOMAIN", "scrollintel.com")
    email = os.getenv("CLOUDFLARE_EMAIL")
    
    if not all([api_token, zone_id, email]):
        print("❌ Missing required environment variables:")
        print("   - CLOUDFLARE_API_TOKEN")
        print("   - CLOUDFLARE_ZONE_ID")
        print("   - CLOUDFLARE_EMAIL")
        sys.exit(1)
    
    return CloudflareConfig(
        api_token=api_token,
        zone_id=zone_id,
        domain=domain,
        email=email
    )

def main():
    """Main execution function"""
    print("🚀 ScrollIntel Cloudflare CDN Setup")
    print("=" * 40)
    
    # Load configuration
    config = load_config()
    manager = CloudflareManager(config)
    
    try:
        # Setup DNS records
        manager.setup_dns_records()
        
        # Configure security settings
        manager.configure_security_settings()
        
        # Setup page rules
        manager.setup_page_rules()
        
        # Setup firewall rules
        manager.setup_firewall_rules()
        
        # Purge cache
        manager.purge_cache()
        
        print("\n✅ Cloudflare CDN setup completed successfully!")
        print(f"🌐 Domain: {config.domain}")
        print("📊 Check your Cloudflare dashboard for verification")
        
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()