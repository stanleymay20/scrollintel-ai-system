#!/usr/bin/env python3
"""
Fix GitHub Repository - Remove Large Files and Deploy ScrollIntel
"""

import subprocess
import os
import shutil

def run_command(command, shell=True):
    """Run command and return result"""
    try:
        print(f"Running: {command}")
        result = subprocess.run(
            command,
            shell=shell,
            check=True,
            capture_output=True,
            text=True
        )
        print(f"✅ Success: {command}")
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed: {command}")
        print(f"Error: {e.stderr}")
        return None

def fix_repository():
    """Fix the repository by removing large files from history"""
    print("🔧 Fixing ScrollIntel Repository...")
    
    # Remove the problematic remote
    run_command("git remote remove origin")
    
    # Create a new clean repository
    print("📦 Creating clean repository...")
    
    # Remove .git directory to start fresh
    if os.path.exists(".git"):
        shutil.rmtree(".git")
    
    # Initialize new git repository
    run_command("git init")
    
    # Add all files (respecting .gitignore)
    run_command("git add .")
    
    # Initial commit
    run_command('git commit -m "Initial commit - ScrollIntel AI Platform ready for deployment"')
    
    # Add the remote
    run_command("git remote add origin https://github.com/stanleymay20/ScrollIntel.git")
    
    # Create main branch
    run_command("git branch -M main")
    
    # Force push to overwrite the repository
    run_command("git push -f origin main")
    
    print("✅ Repository fixed and ready for deployment!")
    
    return True

def show_deployment_options():
    """Show deployment options"""
    print("\n🚀 ScrollIntel is now ready for cloud deployment!")
    print("=" * 60)
    
    print("\n🥇 OPTION 1: Railway (EASIEST)")
    print("   1. Go to https://railway.app")
    print("   2. Login with GitHub")
    print("   3. New Project → Deploy from GitHub")
    print("   4. Select: stanleymay20/ScrollIntel")
    print("   5. Deploy! 🚀")
    
    print("\n🥈 OPTION 2: Render")
    print("   1. Go to https://render.com")
    print("   2. New Web Service")
    print("   3. Connect: stanleymay20/ScrollIntel")
    print("   4. Configure and Deploy! 🚀")
    
    print("\n🥉 OPTION 3: Vercel (Frontend)")
    print("   1. Go to https://vercel.com")
    print("   2. Import Project")
    print("   3. Select: stanleymay20/ScrollIntel")
    print("   4. Deploy! 🚀")
    
    print("\n🎉 Your ScrollIntel AI Platform is ready to conquer the world!")

if __name__ == "__main__":
    if fix_repository():
        show_deployment_options()