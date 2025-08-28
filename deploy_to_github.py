#!/usr/bin/env python3
"""
Deploy ScrollIntel to GitHub for Railway deployment
"""

import os
import subprocess
import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_command(cmd, check=True):
    """Run a shell command"""
    logger.info(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.stdout:
        logger.info(f"Output: {result.stdout}")
    if result.stderr:
        logger.warning(f"Error: {result.stderr}")
    
    if check and result.returncode != 0:
        logger.error(f"Command failed with return code {result.returncode}")
        sys.exit(1)
    
    return result

def main():
    """Main deployment function"""
    logger.info("Deploying ScrollIntel to GitHub for Railway...")
    
    # Check if we're in a git repository
    result = run_command("git status", check=False)
    if result.returncode != 0:
        logger.info("Initializing git repository...")
        run_command("git init")
        run_command("git branch -M main")
    
    # Add all files
    logger.info("Adding files to git...")
    run_command("git add .")
    
    # Commit changes
    logger.info("Committing changes...")
    commit_message = "Railway deployment fixes - network health optimization"
    run_command(f'git commit -m "{commit_message}"')
    
    # Check if remote exists
    result = run_command("git remote -v", check=False)
    if "origin" not in result.stdout:
        # You'll need to set this to your actual GitHub repository
        repo_url = input("Enter your GitHub repository URL (e.g., https://github.com/username/scrollintel.git): ")
        if repo_url:
            run_command(f"git remote add origin {repo_url}")
        else:
            logger.error("No repository URL provided")
            sys.exit(1)
    
    # Push to GitHub
    logger.info("Pushing to GitHub...")
    run_command("git push -u origin main")
    
    logger.info("✅ Successfully deployed to GitHub!")
    logger.info("🚀 Now you can redeploy on Railway")
    logger.info("📋 Railway deployment checklist:")
    logger.info("   1. Connect your GitHub repository to Railway")
    logger.info("   2. Set environment variables (DATABASE_URL, etc.)")
    logger.info("   3. Deploy using the railway.json configuration")
    logger.info("   4. Monitor the /health endpoint for successful deployment")

if __name__ == "__main__":
    main()