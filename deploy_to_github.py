#!/usr/bin/env python3
"""
Deploy ScrollIntel to GitHub for cloud deployment
"""

import os
import subprocess

def run_command(command):
    """Run command and return result"""
    try:
        print(f"🔧 {command}")
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ Success")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed: {e.stderr}")
        return False

def deploy_to_github():
    """Deploy ScrollIntel to GitHub"""
    print("🚀 ScrollIntel GitHub Deployment")
    print("=" * 50)
    
    # Check if git is initialized
    if not os.path.exists(".git"):
        print("📁 Initializing Git repository...")
        run_command("git init")
    
    # Create .gitignore if it doesn't exist
    if not os.path.exists(".gitignore"):
        gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Logs
*.log
logs/

# Database
*.db
*.sqlite3

# Environment variables
.env.local
.env.development.local
.env.test.local
.env.production.local

# Node modules
node_modules/

# Next.js
.next/
out/

# Uploads
uploads/
temp/

# Models
models/
*.pkl
*.joblib
"""
        with open(".gitignore", "w") as f:
            f.write(gitignore_content)
        print("✅ Created .gitignore")
    
    # Add all files
    print("📦 Adding files to Git...")
    run_command("git add .")
    
    # Commit
    print("💾 Committing changes...")
    run_command('git commit -m "ScrollIntel production ready - AI CTO replacement platform"')
    
    # Instructions for GitHub
    print("\n🎯 NEXT STEPS:")
    print("=" * 50)
    print("1. Create a new repository on GitHub:")
    print("   🔗 https://github.com/new")
    print("   📝 Name: scrollintel")
    print("   📝 Description: AI-powered CTO replacement platform")
    print("")
    print("2. Connect your local repo to GitHub:")
    print("   git remote add origin https://github.com/YOUR_USERNAME/scrollintel.git")
    print("   git branch -M main")
    print("   git push -u origin main")
    print("")
    print("3. Deploy to cloud platforms:")
    print("   🚂 Railway: https://railway.app")
    print("   ☁️  Render: https://render.com")
    print("   ▲  Vercel: https://vercel.com")
    print("")
    print("✅ Your code is ready for cloud deployment!")
    
    return True

if __name__ == "__main__":
    deploy_to_github()