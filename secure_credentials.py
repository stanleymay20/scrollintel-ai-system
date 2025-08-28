#!/usr/bin/env python3
"""
ScrollIntel Secure Credentials Manager
Helps manage and secure API keys and credentials
"""

import os
import json
import hashlib
import base64
from datetime import datetime
from pathlib import Path
import subprocess

class CredentialsManager:
    def __init__(self):
        self.env_files = ['.env.production', '.env.secure.backup']
        self.sensitive_keys = [
            'ANTHROPIC_API_KEY', 'OPENAI_API_KEY', 'GOOGLE_AI_API_KEY',
            'JWT_SECRET_KEY', 'SESSION_SECRET_KEY', 'AWS_SECRET_ACCESS_KEY',
            'GOOGLE_CLIENT_SECRET', 'GITHUB_CLIENT_SECRET', 'STRIPE_SECRET_KEY',
            'STRIPE_WEBHOOK_SECRET', 'PAYPAL_CLIENT_SECRET', 'TWILIO_AUTH_TOKEN'
        ]
    
    def check_file_permissions(self):
        """Check and fix file permissions for security"""
        print("🔒 Checking file permissions...")
        
        for env_file in self.env_files:
            if os.path.exists(env_file):
                # Get current permissions
                stat = os.stat(env_file)
                permissions = oct(stat.st_mode)[-3:]
                
                print(f"📄 {env_file}: {permissions}")
                
                # Set secure permissions (600 = read/write for owner only)
                if permissions != '600':
                    try:
                        os.chmod(env_file, 0o600)
                        print(f"✅ Fixed permissions for {env_file}")
                    except Exception as e:
                        print(f"❌ Could not fix permissions for {env_file}: {e}")
                else:
                    print(f"✅ {env_file} has secure permissions")
    
    def validate_keys(self):
        """Validate that all required keys are present and properly formatted"""
        print("\\n🔍 Validating API keys...")
        
        missing_keys = []
        invalid_keys = []
        
        for env_file in self.env_files:
            if not os.path.exists(env_file):
                continue
                
            print(f"\\n📄 Checking {env_file}:")
            
            with open(env_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            for key in self.sensitive_keys:
                if f"{key}=" in content:
                    # Extract the value
                    for line in content.split('\\n'):
                        if line.startswith(f"{key}="):
                            value = line.split('=', 1)[1].strip()
                            
                            # Basic validation
                            if not value or value in ['your-key-here', 'replace-with-your-key']:
                                invalid_keys.append(f"{key} (placeholder value)")
                            elif len(value) < 10:
                                invalid_keys.append(f"{key} (too short)")
                            else:
                                print(f"  ✅ {key}: Valid ({len(value)} chars)")
                            break
                else:
                    missing_keys.append(key)
        
        if missing_keys:
            print(f"\\n⚠️  Missing keys: {', '.join(missing_keys)}")
        
        if invalid_keys:
            print(f"\\n❌ Invalid keys: {', '.join(invalid_keys)}")
        
        if not missing_keys and not invalid_keys:
            print("\\n🎉 All keys are present and valid!")
    
    def create_key_inventory(self):
        """Create an inventory of all keys (without exposing values)"""
        print("\\n📋 Creating key inventory...")
        
        inventory = {
            'timestamp': datetime.now().isoformat(),
            'files_checked': [],
            'keys_found': {},
            'security_status': {}
        }
        
        for env_file in self.env_files:
            if not os.path.exists(env_file):
                continue
            
            inventory['files_checked'].append(env_file)
            
            with open(env_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            file_keys = {}
            for line in content.split('\\n'):
                if '=' in line and not line.startswith('#'):
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    if key and value:
                        # Create hash of value for verification (not the actual value)
                        value_hash = hashlib.sha256(value.encode()).hexdigest()[:8]
                        file_keys[key] = {
                            'length': len(value),
                            'hash': value_hash,
                            'is_sensitive': key in self.sensitive_keys
                        }
            
            inventory['keys_found'][env_file] = file_keys
        
        # Save inventory
        with open('credentials_inventory.json', 'w') as f:
            json.dump(inventory, f, indent=2)
        
        print("✅ Inventory saved to credentials_inventory.json")
    
    def backup_credentials(self):
        """Create encrypted backup of credentials"""
        print("\\n💾 Creating encrypted backup...")
        
        try:
            # Check if gpg is available
            subprocess.run(['gpg', '--version'], capture_output=True, check=True)
            
            for env_file in self.env_files:
                if os.path.exists(env_file):
                    backup_name = f"{env_file}.{datetime.now().strftime('%Y%m%d_%H%M%S')}.gpg"
                    
                    # Encrypt the file
                    cmd = ['gpg', '--symmetric', '--cipher-algo', 'AES256', '--output', backup_name, env_file]
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    
                    if result.returncode == 0:
                        print(f"✅ Encrypted backup created: {backup_name}")
                    else:
                        print(f"❌ Failed to create backup for {env_file}")
        
        except subprocess.CalledProcessError:
            print("⚠️  GPG not available. Install GPG for encrypted backups.")
        except Exception as e:
            print(f"❌ Backup failed: {e}")
    
    def security_recommendations(self):
        """Display security recommendations"""
        print("\\n🛡️  SECURITY RECOMMENDATIONS:")
        print("=" * 50)
        
        recommendations = [
            "1. 🔐 Enable 2FA on all provider accounts (OpenAI, Anthropic, etc.)",
            "2. 🔄 Rotate API keys quarterly",
            "3. 📊 Set up billing alerts for all services",
            "4. 🚨 Monitor API usage for anomalies",
            "5. 🔒 Use environment-specific keys (dev/staging/prod)",
            "6. 📝 Document key rotation procedures",
            "7. 🏢 Use corporate accounts instead of personal ones",
            "8. 🔍 Regular security audits",
            "9. 💾 Backup keys in multiple secure locations",
            "10. 🚫 Never commit .env files to version control"
        ]
        
        for rec in recommendations:
            print(f"  {rec}")
        
        print("\\n🔗 Additional Resources:")
        print("  • OWASP API Security: https://owasp.org/www-project-api-security/")
        print("  • AWS Security Best Practices: https://aws.amazon.com/security/")
        print("  • OpenAI Security Guidelines: https://platform.openai.com/docs/guides/safety-best-practices")
    
    def run_security_check(self):
        """Run complete security check"""
        print("🔒 ScrollIntel Credentials Security Check")
        print("=" * 50)
        
        self.check_file_permissions()
        self.validate_keys()
        self.create_key_inventory()
        self.security_recommendations()
        
        print("\\n✅ Security check complete!")
        print("💡 Run this script regularly to maintain security")

def main():
    """Main entry point"""
    manager = CredentialsManager()
    
    import argparse
    parser = argparse.ArgumentParser(description="ScrollIntel Credentials Manager")
    parser.add_argument("--check", action="store_true", help="Run security check")
    parser.add_argument("--backup", action="store_true", help="Create encrypted backup")
    parser.add_argument("--inventory", action="store_true", help="Create key inventory")
    parser.add_argument("--permissions", action="store_true", help="Fix file permissions")
    
    args = parser.parse_args()
    
    if args.check or not any(vars(args).values()):
        manager.run_security_check()
    elif args.backup:
        manager.backup_credentials()
    elif args.inventory:
        manager.create_key_inventory()
    elif args.permissions:
        manager.check_file_permissions()

if __name__ == "__main__":
    main()