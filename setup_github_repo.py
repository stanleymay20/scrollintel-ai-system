#!/usr/bin/env python3
"""
GitHub Repository Setup Script for ScrollIntel
"""

import subprocess
import sys
import os

def run_command(command, check=True):
    """Run a command and return the result"""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, check=check)
        return result.stdout.strip(), result.stderr.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {command}")
        print(f"Error: {e.stderr}")
        return None, e.stderr

def setup_github_repo():
    """Set up GitHub repository for ScrollIntel"""
    
    print("🚀 Setting up ScrollIntel GitHub Repository")
    print("=" * 50)
    
    # Check if GitHub CLI is installed
    stdout, stderr = run_command("gh --version", check=False)
    if stderr:
        print("❌ GitHub CLI not found. Please install it first:")
        print("   Visit: https://cli.github.com/")
        print("   Or run: winget install GitHub.cli")
        return False
    
    print(f"✅ GitHub CLI found: {stdout}")
    
    # Check if user is logged in
    stdout, stderr = run_command("gh auth status", check=False)
    if "not logged in" in stderr.lower() or stderr:
        print("🔐 Please log in to GitHub:")
        run_command("gh auth login")
    
    # Create repository
    repo_name = "scrollintel-ai-system"
    description = "ScrollIntel™ - Advanced AI System with Multi-Agent Architecture, Visual Generation, and Enterprise Features"
    
    print(f"\n📦 Creating GitHub repository: {repo_name}")
    
    # Create the repository
    create_cmd = f'gh repo create {repo_name} --public --description "{description}" --clone=false'
    stdout, stderr = run_command(create_cmd, check=False)
    
    if "already exists" in stderr:
        print("⚠️  Repository already exists. Using existing repository.")
    elif stderr and "successfully created" not in stdout:
        print(f"❌ Error creating repository: {stderr}")
        return False
    else:
        print("✅ Repository created successfully!")
    
    # Get the repository URL
    username_cmd = "gh api user --jq .login"
    username, _ = run_command(username_cmd)
    
    if not username:
        print("❌ Could not get GitHub username")
        return False
    
    repo_url = f"https://github.com/{username}/{repo_name}.git"
    
    # Add remote origin
    print(f"\n🔗 Adding remote origin: {repo_url}")
    run_command("git remote remove origin", check=False)  # Remove if exists
    stdout, stderr = run_command(f"git remote add origin {repo_url}")
    
    if stderr:
        print(f"❌ Error adding remote: {stderr}")
        return False
    
    print("✅ Remote origin added successfully!")
    
    # Push to GitHub
    print("\n⬆️  Pushing code to GitHub...")
    stdout, stderr = run_command("git push -u origin main")
    
    if stderr and "error" in stderr.lower():
        print(f"❌ Error pushing to GitHub: {stderr}")
        return False
    
    print("✅ Code pushed successfully!")
    
    # Open repository in browser
    print(f"\n🌐 Opening repository in browser...")
    run_command(f"gh repo view {username}/{repo_name} --web", check=False)
    
    print("\n🎉 SUCCESS! Your ScrollIntel repository is now on GitHub!")
    print(f"📍 Repository URL: https://github.com/{username}/{repo_name}")
    print("\n📋 Next steps:")
    print("   1. Add a detailed README.md")
    print("   2. Set up GitHub Actions for CI/CD")
    print("   3. Configure branch protection rules")
    print("   4. Add collaborators if needed")
    
    return True

if __name__ == "__main__":
    success = setup_github_repo()
    if not success:
        sys.exit(1)