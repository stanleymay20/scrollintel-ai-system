#!/usr/bin/env python3
"""
ScrollIntel.com Quick Setup Guide
Simple setup for your new scrollintel.com domain
"""

import os
import json

def print_banner():
    print("""
╔══════════════════════════════════════════════════════════════╗
║                    ScrollIntel.com Setup                     ║
║              Production-Ready AI Platform Launch             ║
╚══════════════════════════════════════════════════════════════╝
    """)

def main():
    print_banner()
    
    domain = "scrollintel.com"
    api_subdomain = "api.scrollintel.com"
    app_subdomain = "app.scrollintel.com"
    
    print(f"🎯 Setting up ScrollIntel for {domain}")
    print()
    
    print("📋 DEPLOYMENT CHECKLIST:")
    print("=" * 50)
    print()
    
    print("1. 🌐 DNS CONFIGURATION")
    print(f"   Configure these DNS records for {domain}:")
    print(f"   A     {domain}           -> YOUR_SERVER_IP")
    print(f"   A     {app_subdomain}    -> YOUR_SERVER_IP")
    print(f"   A     {api_subdomain}    -> YOUR_SERVER_IP")
    print(f"   CNAME www.{domain}       -> {domain}")
    print()
    
    print("2. 🔑 ENVIRONMENT VARIABLES")
    print("   Set these required environment variables:")
    print("   OPENAI_API_KEY=your_openai_api_key")
    print("   JWT_SECRET_KEY=your_jwt_secret_key")
    print("   DATABASE_URL=postgresql://user:pass@host:port/db")
    print()
    
    print("3. 🚀 DEPLOYMENT OPTIONS")
    print("   Choose your deployment method:")
    print()
    print("   Option A: Quick Local Test")
    print("   python run_simple.py")
    print("   → Ready in 30 seconds, perfect for testing")
    print()
    print("   Option B: Production Docker")
    print("   docker-compose -f docker-compose.prod.yml up -d")
    print("   → Full production setup with monitoring")
    print()
    print("   Option C: Heavy Volume (Enterprise)")
    print("   ./start_heavy_volume.sh")
    print("   → Handles 50GB files, 1000+ users")
    print()
    
    print("4. 🔒 SSL CERTIFICATES")
    print("   Get free SSL certificates with Let's Encrypt:")
    print("   sudo certbot --nginx -d scrollintel.com -d app.scrollintel.com -d api.scrollintel.com")
    print()
    
    print("5. 📊 MONITORING SETUP")
    print("   Access your monitoring dashboards:")
    print(f"   Grafana: https://{app_subdomain}/monitoring")
    print(f"   Health Check: https://{api_subdomain}/health")
    print(f"   API Docs: https://{api_subdomain}/docs")
    print()
    
    print("6. 🎉 LAUNCH VERIFICATION")
    print("   After deployment, verify these URLs work:")
    print(f"   ✅ Main Site: https://{domain}")
    print(f"   ✅ Application: https://{app_subdomain}")
    print(f"   ✅ API: https://{api_subdomain}")
    print()
    
    print("🚀 IMMEDIATE NEXT STEPS:")
    print("=" * 50)
    print("1. Set up your DNS records (point to your server IP)")
    print("2. Choose a deployment option above")
    print("3. Get SSL certificates")
    print("4. Test the platform")
    print("5. Start using your AI agents!")
    print()
    
    print("💡 QUICK START FOR TESTING:")
    print("If you want to test locally first:")
    print("1. python run_simple.py")
    print("2. Open http://localhost:8000")
    print("3. Upload a CSV file and chat with AI agents")
    print()
    
    print("📞 SUPPORT:")
    print("- Documentation: Check the docs/ directory")
    print("- Health Check: python health_check.py")
    print("- Status: python scrollintel_deployment_status.py")
    print()
    
    print(f"🌟 Your ScrollIntel platform will be live at https://{domain}")
    print("Ready to replace your CTO with AI! 🤖")

if __name__ == "__main__":
    main()