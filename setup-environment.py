#!/usr/bin/env python3
"""
ScrollIntel Environment Setup
Sets up the development environment for ScrollIntel launch
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def print_header():
    """Print setup header"""
    print("🔧 ScrollIntel Environment Setup")
    print("=" * 50)
    print("Setting up your development environment...")
    print("=" * 50)
    print()

def create_env_file():
    """Create .env file from example"""
    print("📝 Setting up environment variables...")
    
    if os.path.exists(".env"):
        print("✅ .env file already exists")
        return True
    
    if os.path.exists(".env.example"):
        try:
            shutil.copy(".env.example", ".env")
            print("✅ Created .env from .env.example")
            
            # Update with development defaults
            with open(".env", "r") as f:
                content = f.read()
            
            # Replace with development-friendly defaults
            content = content.replace(
                "DATABASE_URL=postgresql://user:password@localhost/dbname",
                "DATABASE_URL=postgresql://scrollintel:password@localhost/scrollintel_dev"
            )
            content = content.replace(
                "REDIS_HOST=localhost",
                "REDIS_HOST=localhost"
            )
            content = content.replace(
                "JWT_SECRET=your-secret-key-change-in-production",
                "JWT_SECRET=dev-secret-key-not-for-production"
            )
            
            with open(".env", "w") as f:
                f.write(content)
            
            print("✅ Updated .env with development defaults")
            return True
            
        except Exception as e:
            print(f"❌ Failed to create .env file: {e}")
            return False
    else:
        # Create basic .env file
        env_content = """# ScrollIntel Development Environment
ENVIRONMENT=development
DEBUG=true

# Database
DATABASE_URL=postgresql://scrollintel:password@localhost/scrollintel_dev

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379

# Security
JWT_SECRET=dev-secret-key-not-for-production

# API Configuration
RATE_LIMIT_PER_SECOND=10
RATE_LIMIT_PER_MINUTE=100
RATE_LIMIT_PER_HOUR=1000
RATE_LIMIT_PER_DAY=10000

# Email (for development - configure for production)
SMTP_SERVER=localhost
SMTP_PORT=587
EMAIL_USERNAME=
EMAIL_PASSWORD=
FROM_EMAIL=noreply@scrollintel.com

# Application
BASE_URL=http://localhost:3000
API_URL=http://localhost:8000
"""
        
        try:
            with open(".env", "w") as f:
                f.write(env_content)
            print("✅ Created basic .env file")
            return True
        except Exception as e:
            print(f"❌ Failed to create .env file: {e}")
            return False

def check_system_requirements():
    """Check system requirements"""
    print("🔍 Checking System Requirements...")
    
    checks = []
    
    # Check Python version
    if sys.version_info >= (3, 8):
        print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
        checks.append(True)
    else:
        print(f"❌ Python {sys.version_info.major}.{sys.version_info.minor} (3.8+ required)")
        checks.append(False)
    
    # Check pip
    try:
        subprocess.run([sys.executable, "-m", "pip", "--version"], 
                      check=True, capture_output=True)
        print("✅ pip available")
        checks.append(True)
    except:
        print("❌ pip not available")
        checks.append(False)
    
    # Check git (optional but recommended)
    try:
        subprocess.run(["git", "--version"], check=True, capture_output=True)
        print("✅ git available")
        checks.append(True)
    except:
        print("⚠️  git not available (optional)")
        checks.append(True)  # Not critical
    
    return all(checks)

def install_basic_dependencies():
    """Install basic dependencies"""
    print("\n📦 Installing Basic Dependencies...")
    
    # Create a minimal requirements file if it doesn't exist
    if not os.path.exists("requirements.txt"):
        basic_requirements = """fastapi==0.104.1
uvicorn==0.24.0
pydantic==2.5.0
sqlalchemy==2.0.23
alembic==1.13.0
redis==5.0.1
passlib==1.7.4
bcrypt==4.1.2
python-jose[cryptography]==3.3.0
python-multipart==0.0.6
aiohttp==3.9.1
psutil==5.9.6
requests==2.31.0
pytest==7.4.3
pytest-asyncio==0.21.1
"""
        
        with open("requirements.txt", "w") as f:
            f.write(basic_requirements)
        print("✅ Created basic requirements.txt")
    
    try:
        print("⏳ Installing dependencies (this may take a few minutes)...")
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ], capture_output=True, text=True, timeout=600)
        
        if result.returncode == 0:
            print("✅ Dependencies installed successfully")
            return True
        else:
            print(f"❌ Failed to install dependencies: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Dependency installation timed out")
        return False
    except Exception as e:
        print(f"❌ Error installing dependencies: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    print("\n📁 Creating Directories...")
    
    directories = [
        "logs",
        "data",
        "config",
        "backups",
        "uploads"
    ]
    
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            print(f"✅ {directory}/")
        except Exception as e:
            print(f"❌ Failed to create {directory}/: {e}")
            return False
    
    return True

def run_quick_validation():
    """Run quick validation to ensure setup worked"""
    print("\n🧪 Running Quick Validation...")
    
    try:
        # Test imports
        sys.path.insert(0, os.path.join(os.path.abspath('.'), 'scrollintel', 'core'))
        
        import config
        import production_infrastructure
        import user_onboarding
        import api_stability
        
        print("✅ Core modules can be imported")
        
        # Test configuration
        test_config = config.get_default_config()
        if config.validate_config(test_config):
            print("✅ Configuration system works")
        else:
            print("❌ Configuration validation failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Quick validation failed: {e}")
        return False

def show_next_steps():
    """Show next steps"""
    print("\n🎯 Next Steps:")
    print("=" * 30)
    print()
    print("Your environment is now set up! You can:")
    print()
    print("1. 🚀 Start the step-by-step launch:")
    print("   python start-launch.py")
    print()
    print("2. 🎮 Use the interactive launch coordinator:")
    print("   python scripts/launch-coordinator.py")
    print()
    print("3. 🧪 Run tests manually:")
    print("   python test_immediate_priority_direct.py")
    print()
    print("4. 🖥️  Start development server directly:")
    print("   python scrollintel/api/production_main.py")
    print()
    print("📚 For more information, see:")
    print("   - SCROLLINTEL_STEP_BY_STEP_LAUNCH_PLAN.md")
    print("   - IMMEDIATE_PRIORITY_IMPLEMENTATION_SUMMARY.md")

def main():
    """Main setup function"""
    print_header()
    
    setup_steps = []
    
    # Step 1: System requirements
    setup_steps.append(("System Requirements", check_system_requirements()))
    
    # Step 2: Environment file
    setup_steps.append(("Environment Configuration", create_env_file()))
    
    # Step 3: Dependencies
    setup_steps.append(("Dependency Installation", install_basic_dependencies()))
    
    # Step 4: Directories
    setup_steps.append(("Directory Creation", create_directories()))
    
    # Step 5: Quick validation
    setup_steps.append(("Quick Validation", run_quick_validation()))
    
    # Calculate results
    passed = sum(result for _, result in setup_steps)
    total = len(setup_steps)
    success_rate = (passed / total) * 100
    
    # Print summary
    print("\n" + "=" * 50)
    print("📊 SETUP SUMMARY")
    print("=" * 50)
    
    for step_name, result in setup_steps:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {step_name}")
    
    print(f"\nOverall: {passed}/{total} ({success_rate:.1f}%)")
    
    if success_rate == 100:
        print("\n🎉 ENVIRONMENT SETUP COMPLETE!")
        show_next_steps()
        return True
    else:
        print("\n❌ SETUP INCOMPLETE")
        print("Please address the failed checks and run this script again.")
        return False

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nSetup interrupted. Run again when ready: python setup-environment.py")
        sys.exit(1)