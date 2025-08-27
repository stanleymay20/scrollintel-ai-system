#!/usr/bin/env python3
"""
ScrollIntel.com Setup Script
Handles prerequisites, domain configuration, and initial setup.
"""

import os
import subprocess
import sys
import json
from pathlib import Path

def run_command(command, description):
    """Run a command with error handling."""
    print(f"[WRENCH] {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"[CHECK] {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[X] {description} failed: {e}")
        if e.stderr:
            print(f"Error: {e.stderr}")
        return False

def check_prerequisites():
    """Check if all prerequisites are installed."""
    print("[SEARCH] Checking prerequisites...")
    
    prerequisites = [
        ("docker", "Docker"),
        ("docker-compose", "Docker Compose"),
        ("git", "Git"),
        ("curl", "cURL"),
        ("python3", "Python 3")
    ]
    
    missing = []
    for cmd, name in prerequisites:
        if not run_command(f"which {cmd}", f"Checking {name}"):
            missing.append(name)
    
    if missing:
        print(f"[X] Missing prerequisites: {', '.join(missing)}")
        print("\nPlease install the missing prerequisites and run this script again.")
        return False
    
    print("[CHECK] All prerequisites are installed")
    return True

def setup_domain_configuration():
    """Setup domain configuration."""
    print("\n[CLIPBOARD] Domain Configuration")
    print("=" * 30)
    
    domain = input("Enter your domain (default: scrollintel.com): ").strip()
    if not domain:
        domain = "scrollintel.com"
    
    print(f"\n[GLOBE] Domain Configuration:")
    print(f"   Main site: https://{domain}")
    print(f"   API: https://api.{domain}")
    print(f"   App: https://app.{domain}")
    print(f"   Monitoring: https://grafana.{domain}")
    
    confirm = input("\nIs this configuration correct? (Y/n): ").strip().lower()
    if confirm == 'n':
        return setup_domain_configuration()
    
    return domain

def setup_environment_variables():
    """Setup environment variables."""
    print("\n🔐 Environment Configuration")
    print("=" * 30)
    
    env_vars = {}
    
    # Database configuration
    print("\n[CHART] Database Configuration:")
    env_vars['DB_PASSWORD'] = input("Enter database password (or press Enter for auto-generated): ").strip()
    if not env_vars['DB_PASSWORD']:
        import secrets
        env_vars['DB_PASSWORD'] = secrets.token_urlsafe(32)
        print(f"Generated database password: {env_vars['DB_PASSWORD']}")
    
    # Security keys
    print("\n[LOCK] Security Configuration:")
    env_vars['SECRET_KEY'] = input("Enter secret key (or press Enter for auto-generated): ").strip()
    if not env_vars['SECRET_KEY']:
        import secrets
        env_vars['SECRET_KEY'] = secrets.token_urlsafe(64)
        print(f"Generated secret key: {env_vars['SECRET_KEY'][:20]}...")
    
    # API Keys
    print("\n🤖 AI Service Configuration:")
    env_vars['OPENAI_API_KEY'] = input("Enter OpenAI API key (optional): ").strip()
    env_vars['ANTHROPIC_API_KEY'] = input("Enter Anthropic API key (optional): ").strip()
    
    # Email configuration
    print("\n📧 Email Configuration:")
    env_vars['SMTP_HOST'] = input("Enter SMTP host (default: smtp.gmail.com): ").strip() or "smtp.gmail.com"
    env_vars['SMTP_USER'] = input("Enter SMTP username: ").strip()
    env_vars['SMTP_PASSWORD'] = input("Enter SMTP password: ").strip()
    
    return env_vars

def create_initial_structure():
    """Create initial project structure."""
    print("\n📁 Creating project structure...")
    
    directories = [
        'frontend/src/app/api',
        'scrollintel/api/routes',
        'monitoring',
        'nginx',
        'scripts',
        'backups',
        'logs',
        'uploads',
        'letsencrypt'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"[CHECK] Created directory: {directory}")
    
    return True

def setup_git_repository():
    """Setup Git repository if not already initialized."""
    print("\n[MEMO] Setting up Git repository...")
    
    if not os.path.exists('.git'):
        if run_command("git init", "Initializing Git repository"):
            run_command("git add .", "Adding files to Git")
            run_command('git commit -m "Initial ScrollIntel.com setup"', "Creating initial commit")
    else:
        print("[CHECK] Git repository already exists")
    
    return True

def create_quick_start_guide():
    """Create a quick start guide."""
    guide_content = """# ScrollIntel.com Quick Start Guide

## [ROCKET] Getting Started

Your ScrollIntel.com platform has been set up! Follow these steps to complete the deployment:

### 1. Domain Setup
- Point your domain DNS A record to this server's IP address
- Ensure subdomains (api, app, grafana) also point to this server

### 2. Complete Deployment
```bash
python3 deploy_scrollintel_com_complete.py
```

### 3. Verify Deployment
After deployment, check these URLs:
- Main site: https://scrollintel.com
- API: https://api.scrollintel.com/docs
- Monitoring: https://grafana.scrollintel.com

### 4. Management Commands
```bash
# Deploy updates
./deploy.sh

# Check status
./status.sh

# View logs
docker-compose -f docker-compose.production.yml logs -f

# Backup database
docker-compose -f docker-compose.production.yml exec backup /backup.sh
```

### 5. Security Checklist
- [ ] Update all default passwords
- [ ] Configure firewall rules
- [ ] Set up monitoring alerts
- [ ] Configure backup storage
- [ ] Review SSL certificate setup

### 6. Customization
- Update branding in `frontend/src/app/`
- Configure AI models in `scrollintel/core/config.py`
- Add custom agents in `scrollintel/agents/`
- Set up integrations in `scrollintel/connectors/`

## 📞 Support
For support and documentation, visit: https://docs.scrollintel.com
"""
    
    with open('QUICK_START.md', 'w') as f:
        f.write(guide_content)
    
    print("[CHECK] Quick start guide created: QUICK_START.md")
    return True

def main():
    """Main setup function."""
    print("[DIRECT_HIT] ScrollIntel.com Setup")
    print("=" * 30)
    print("This script will prepare your system for ScrollIntel.com deployment.")
    print()
    
    # Check prerequisites
    if not check_prerequisites():
        sys.exit(1)
    
    # Setup domain configuration
    domain = setup_domain_configuration()
    
    # Setup environment variables
    env_vars = setup_environment_variables()
    
    # Create project structure
    if not create_initial_structure():
        sys.exit(1)
    
    # Setup Git repository
    setup_git_repository()
    
    # Create quick start guide
    create_quick_start_guide()
    
    print("\n[PARTY] ScrollIntel.com setup completed!")
    print("\n[CLIPBOARD] Next Steps:")
    print("1. Review the configuration in QUICK_START.md")
    print("2. Point your domain DNS to this server")
    print("3. Run: python3 deploy_scrollintel_com_complete.py")
    print("\n[ROCKET] Your ScrollIntel.com platform will be ready in minutes!")

if __name__ == "__main__":
    main()