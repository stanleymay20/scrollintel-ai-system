#!/usr/bin/env python3
"""
Generate secure secrets for ScrollIntel production
"""

import secrets
import string
import base64
import os

def generate_jwt_secret(length=64):
    """Generate a secure JWT secret"""
    return secrets.token_urlsafe(length)

def generate_session_secret(length=32):
    """Generate a secure session secret"""
    alphabet = string.ascii_letters + string.digits + "!@#$%^&*"
    return ''.join(secrets.choice(alphabet) for _ in range(length))

def generate_api_key(length=32):
    """Generate a secure API key"""
    return secrets.token_hex(length)

def main():
    print("🔐 ScrollIntel Production Secrets Generator")
    print("=" * 50)
    
    # Generate secrets
    jwt_secret = generate_jwt_secret()
    session_secret = generate_session_secret()
    api_key = generate_api_key()
    
    print(f"JWT_SECRET_KEY={jwt_secret}")
    print(f"SESSION_SECRET_KEY={session_secret}")
    print(f"INTERNAL_API_KEY={api_key}")
    
    print("\n🔒 Security Notes:")
    print("1. Copy these secrets to your .env.production file")
    print("2. Store them securely in your password manager")
    print("3. Never commit them to version control")
    print("4. Rotate them regularly (every 90 days)")
    
    # Save to secure file
    with open('.env.secrets', 'w') as f:
        f.write(f"JWT_SECRET_KEY={jwt_secret}\n")
        f.write(f"SESSION_SECRET_KEY={session_secret}\n")
        f.write(f"INTERNAL_API_KEY={api_key}\n")
    
    # Set secure permissions
    os.chmod('.env.secrets', 0o600)
    print("\n✅ Secrets saved to .env.secrets (600 permissions)")

if __name__ == "__main__":
    main()