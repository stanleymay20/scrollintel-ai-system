#!/usr/bin/env python3
"""
ScrollIntel™ Environment Setup Script
Automatically configures environment variables and secrets
"""

import os
import secrets
import shutil
import sys
from pathlib import Path

def print_status(message, status="INFO"):
    colors = {
        "INFO": "\033[0;34m",
        "SUCCESS": "\033[0;32m", 
        "WARNING": "\033[1;33m",
        "ERROR": "\033[0;31m"
    }
    reset = "\033[0m"
    print(f"{colors.get(status, '')}[{status}]{reset} {message}")

def generate_secret(length=32):
    """Generate a secure random secret"""
    return secrets.token_hex(length)

def setup_env_file():
    """Setup .env file from .env.example"""
    env_example = Path(".env.example")
    env_file = Path(".env")
    
    if not env_example.exists():
        print_status(".env.example file not found!", "ERROR")
        return False
    
    if env_file.exists():
        print_status(".env file already exists, backing up...", "WARNING")
        shutil.copy(env_file, f".env.backup.{secrets.token_hex(4)}")
    
    # Copy example to .env
    shutil.copy(env_example, env_file)
    print_status("Created .env from .env.example", "SUCCESS")
    
    # Read current content
    with open(env_file, 'r') as f:
        content = f.read()
    
    # Generate and set JWT secret
    if "JWT_SECRET_KEY=" in content and not any(line.startswith("JWT_SECRET_KEY=") and len(line.split("=", 1)[1].strip()) > 10 for line in content.split('\n')):
        jwt_secret = generate_secret(32)
        content = content.replace("JWT_SECRET_KEY=", f"JWT_SECRET_KEY={jwt_secret}")
        print_status("Generated secure JWT secret", "SUCCESS")
    
    # Generate and set database password
    if "POSTGRES_PASSWORD=" in content and not any(line.startswith("POSTGRES_PASSWORD=") and len(line.split("=", 1)[1].strip()) > 5 for line in content.split('\n')):
        db_password = generate_secret(16)
        content = content.replace("POSTGRES_PASSWORD=", f"POSTGRES_PASSWORD={db_password}")
        print_status("Generated database password", "SUCCESS")
    
    # Write updated content
    with open(env_file, 'w') as f:
        f.write(content)
    
    return True

def check_api_keys():
    """Check if API keys are configured"""
    env_file = Path(".env")
    if not env_file.exists():
        return False
    
    with open(env_file, 'r') as f:
        content = f.read()
    
    # Check for OpenAI API key
    if "OPENAI_API_KEY=sk-" not in content:
        print_status("OpenAI API key not configured", "WARNING")
        print_status("Add your OpenAI API key to .env: OPENAI_API_KEY=sk-your-key-here", "INFO")
        return False
    else:
        print_status("OpenAI API key configured", "SUCCESS")
        return True

def main():
    print("🔧 ScrollIntel™ Environment Setup")
    print("=" * 35)
    
    # Change to script directory
    script_dir = Path(__file__).parent.parent
    os.chdir(script_dir)
    
    # Setup environment file
    if not setup_env_file():
        sys.exit(1)
    
    # Check API keys
    check_api_keys()
    
    print_status("Environment setup complete!", "SUCCESS")
    print("\n📋 Next steps:")
    print("1. Add your OpenAI API key to .env file")
    print("2. Run: docker-compose up -d")
    print("3. Access: http://localhost:3000")

if __name__ == "__main__":
    main()