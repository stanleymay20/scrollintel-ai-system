#!/usr/bin/env python3
"""
Manual GitHub Repository Setup Instructions for ScrollIntel
"""

import subprocess
import webbrowser

def get_git_info():
    """Get current git information"""
    try:
        # Get current branch
        branch = subprocess.check_output(['git', 'branch', '--show-current'], text=True).strip()
        
        # Get commit count
        commit_count = subprocess.check_output(['git', 'rev-list', '--count', 'HEAD'], text=True).strip()
        
        # Get latest commit
        latest_commit = subprocess.check_output(['git', 'log', '-1', '--oneline'], text=True).strip()
        
        return branch, commit_count, latest_commit
    except Exception as e:
        return "main", "unknown", f"Error: {e}"

def main():
    print("🚀 ScrollIntel GitHub Repository Setup")
    print("=" * 50)
    
    branch, commit_count, latest_commit = get_git_info()
    
    print(f"📊 Current Git Status:")
    print(f"   Branch: {branch}")
    print(f"   Commits: {commit_count}")
    print(f"   Latest: {latest_commit}")
    
    print("\n📋 Manual Setup Instructions:")
    print("=" * 30)
    
    print("\n1️⃣  Create GitHub Repository:")
    print("   • Go to: https://github.com/new")
    print("   • Repository name: scrollintel-ai-system")
    print("   • Description: ScrollIntel™ - Advanced AI System with Multi-Agent Architecture")
    print("   • Make it Public")
    print("   • Don't initialize with README (we have files already)")
    print("   • Click 'Create repository'")
    
    print("\n2️⃣  After creating the repository, run these commands:")
    print("   git remote add origin https://github.com/YOUR_USERNAME/scrollintel-ai-system.git")
    print("   git branch -M main")
    print("   git push -u origin main")
    
    print("\n3️⃣  Alternative: Use these pre-made commands:")
    print("   (Replace YOUR_USERNAME with your actual GitHub username)")
    
    # Open GitHub in browser
    try:
        webbrowser.open("https://github.com/new")
        print("\n🌐 Opening GitHub in your browser...")
    except:
        print("\n🌐 Please manually go to: https://github.com/new")
    
    print("\n📝 Repository Details to Use:")
    print("   Name: scrollintel-ai-system")
    print("   Description: ScrollIntel™ - Advanced AI System with Multi-Agent Architecture, Visual Generation, and Enterprise Features")
    print("   Type: Public")
    print("   Initialize: No (uncheck all options)")
    
    print("\n🎯 What you'll get:")
    print("   • Complete ScrollIntel codebase on GitHub")
    print("   • Professional repository structure")
    print("   • Ready for collaboration and deployment")
    print("   • CI/CD pipeline ready")
    
    # Wait for user input
    input("\n⏸️  Press Enter after you've created the GitHub repository...")
    
    # Ask for username
    username = input("🔤 Enter your GitHub username: ").strip()
    
    if username:
        repo_url = f"https://github.com/{username}/scrollintel-ai-system.git"
        
        print(f"\n🔗 Setting up remote origin...")
        
        # Remove existing origin if any
        subprocess.run(['git', 'remote', 'remove', 'origin'], capture_output=True)
        
        # Add new origin
        result = subprocess.run(['git', 'remote', 'add', 'origin', repo_url], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Remote origin added successfully!")
            
            # Push to GitHub
            print("⬆️  Pushing to GitHub...")
            push_result = subprocess.run(['git', 'push', '-u', 'origin', 'main'], capture_output=True, text=True)
            
            if push_result.returncode == 0:
                print("🎉 SUCCESS! Your code is now on GitHub!")
                print(f"📍 Repository URL: {repo_url}")
                
                # Open the repository
                try:
                    webbrowser.open(repo_url.replace('.git', ''))
                    print("🌐 Opening your repository...")
                except:
                    print(f"🌐 Visit your repository: {repo_url.replace('.git', '')}")
                    
            else:
                print(f"❌ Error pushing to GitHub: {push_result.stderr}")
                print("\n🔧 Try these commands manually:")
                print(f"   git remote add origin {repo_url}")
                print("   git push -u origin main")
        else:
            print(f"❌ Error adding remote: {result.stderr}")
    
    print("\n📚 Next Steps:")
    print("   1. Add a comprehensive README.md")
    print("   2. Set up GitHub Actions for CI/CD")
    print("   3. Configure deployment to your chosen platform")
    print("   4. Add collaborators if needed")

if __name__ == "__main__":
    main()