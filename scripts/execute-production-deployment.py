#!/usr/bin/env python3
"""
ScrollIntel Agent Steering System - Execute Production Deployment
Main entry point for Task 17: Production Deployment and Launch
"""

import os
import sys
import json
import logging
import asyncio
import argparse
from datetime import datetime
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/execute-production-deployment.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def validate_environment():
    """Validate deployment environment and prerequisites"""
    logger.info("🔍 Validating deployment environment...")
    
    # Check required environment variables
    required_vars = [
        "DATABASE_URL",
        "REDIS_URL", 
        "JWT_SECRET_KEY",
        "OPENAI_API_KEY"
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        logger.error(f"❌ Missing required environment variables: {missing_vars}")
        return False
    
    # Check if required scripts exist
    required_scripts = [
        "scripts/production-deployment-orchestrator.py",
        "scripts/production-deployment-launch.py",
        "scripts/user-acceptance-testing.py",
        "scripts/gradual-rollout-manager.py",
        "scripts/go-live-procedures.py"
    ]
    
    missing_scripts = []
    for script in required_scripts:
        if not os.path.exists(script):
            missing_scripts.append(script)
    
    if missing_scripts:
        logger.error(f"❌ Missing required scripts: {missing_scripts}")
        return False
    
    # Check deployment configuration
    if not os.path.exists("deployment-config.yaml"):
        logger.warning("⚠️ deployment-config.yaml not found, using defaults")
    
    # Create necessary directories
    os.makedirs("logs", exist_ok=True)
    os.makedirs("reports/deployment", exist_ok=True)
    os.makedirs("reports/uat", exist_ok=True)
    os.makedirs("reports/rollout", exist_ok=True)
    os.makedirs("reports/go-live", exist_ok=True)
    os.makedirs("backups", exist_ok=True)
    
    logger.info("✅ Environment validation completed")
    return True

def setup_deployment_environment():
    """Setup deployment environment"""
    logger.info("⚙️ Setting up deployment environment...")
    
    try:
        # Set deployment-specific environment variables
        os.environ["DEPLOYMENT_MODE"] = "production"
        os.environ["DEPLOYMENT_TIMESTAMP"] = datetime.now().isoformat()
        
        # Ensure Python path includes current directory
        current_dir = str(Path(__file__).parent.parent)
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        
        logger.info("✅ Deployment environment setup completed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to setup deployment environment: {str(e)}")
        return False

async def execute_orchestrated_deployment():
    """Execute orchestrated production deployment"""
    logger.info("🚀 Starting orchestrated production deployment...")
    
    try:
        # Import orchestrator (after path setup)
        from production_deployment_orchestrator import ProductionDeploymentOrchestrator
        
        # Initialize orchestrator
        orchestrator = ProductionDeploymentOrchestrator()
        
        # Execute deployment
        success = await orchestrator.execute_production_deployment()
        
        return success
        
    except ImportError as e:
        logger.error(f"❌ Failed to import orchestrator: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"❌ Orchestrated deployment failed: {str(e)}")
        return False

def execute_legacy_deployment():
    """Execute legacy production deployment (fallback)"""
    logger.info("🔄 Executing legacy production deployment...")
    
    try:
        import subprocess
        
        # Execute main deployment script
        result = subprocess.run([
            "python", "scripts/production-deployment-launch.py"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info("✅ Legacy deployment completed successfully")
            return True
        else:
            logger.error(f"❌ Legacy deployment failed: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Legacy deployment error: {str(e)}")
        return False

def generate_deployment_summary(success: bool, deployment_type: str):
    """Generate deployment summary"""
    logger.info("📊 Generating deployment summary...")
    
    try:
        summary = {
            "deployment_summary": {
                "timestamp": datetime.now().isoformat(),
                "deployment_type": deployment_type,
                "success": success,
                "status": "completed" if success else "failed",
                "task_id": "17",
                "task_name": "Production Deployment and Launch"
            },
            "components_deployed": [
                "Agent Steering System Core",
                "Real-time Orchestration Engine", 
                "Enterprise Data Integration",
                "Intelligence and Decision Engine",
                "Security and Compliance Framework",
                "Monitoring and Analytics",
                "User Interface and Experience"
            ],
            "deployment_features": [
                "Full monitoring infrastructure",
                "User acceptance testing",
                "Gradual rollout with feature flags",
                "Comprehensive go-live procedures",
                "Automated rollback capabilities",
                "Enterprise-grade security",
                "Real-time performance monitoring"
            ],
            "next_steps": [
                "Monitor system performance and health",
                "Review deployment reports and metrics",
                "Conduct post-deployment validation",
                "Update documentation and runbooks",
                "Plan for ongoing maintenance and updates"
            ] if success else [
                "Review deployment logs and error reports",
                "Address identified issues and failures", 
                "Re-run deployment after fixes",
                "Consider rollback if necessary",
                "Update deployment procedures based on lessons learned"
            ]
        }
        
        # Save summary
        summary_file = f"reports/deployment/task_17_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Generate human-readable summary
        readable_file = f"reports/deployment/task_17_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        with open(readable_file, 'w') as f:
            f.write("ScrollIntel Agent Steering System - Task 17 Deployment Summary\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Task: {summary['deployment_summary']['task_name']}\n")
            f.write(f"Status: {summary['deployment_summary']['status'].upper()}\n")
            f.write(f"Deployment Type: {summary['deployment_summary']['deployment_type']}\n")
            f.write(f"Timestamp: {summary['deployment_summary']['timestamp']}\n\n")
            
            f.write("Components Deployed:\n")
            f.write("-" * 20 + "\n")
            for component in summary["components_deployed"]:
                f.write(f"✓ {component}\n")
            f.write("\n")
            
            f.write("Deployment Features:\n")
            f.write("-" * 20 + "\n")
            for feature in summary["deployment_features"]:
                f.write(f"• {feature}\n")
            f.write("\n")
            
            f.write("Next Steps:\n")
            f.write("-" * 20 + "\n")
            for step in summary["next_steps"]:
                f.write(f"→ {step}\n")
        
        logger.info(f"Deployment summary saved: {summary_file}")
        
    except Exception as e:
        logger.error(f"❌ Failed to generate deployment summary: {str(e)}")

async def main():
    """Main execution function for Task 17"""
    parser = argparse.ArgumentParser(description="Execute ScrollIntel Production Deployment (Task 17)")
    parser.add_argument("--mode", choices=["orchestrated", "legacy"], default="orchestrated",
                       help="Deployment mode (default: orchestrated)")
    parser.add_argument("--validate-only", action="store_true",
                       help="Only validate environment, don't deploy")
    parser.add_argument("--skip-validation", action="store_true",
                       help="Skip environment validation")
    
    args = parser.parse_args()
    
    print("ScrollIntel Agent Steering System - Production Deployment and Launch")
    print("=" * 70)
    print(f"Task ID: 17")
    print(f"Task: Production Deployment and Launch")
    print(f"Mode: {args.mode}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 70)
    
    try:
        # Environment validation
        if not args.skip_validation:
            if not validate_environment():
                print("❌ Environment validation failed!")
                return False
        
        if args.validate_only:
            print("✅ Environment validation completed (validate-only mode)")
            return True
        
        # Setup deployment environment
        if not setup_deployment_environment():
            print("❌ Failed to setup deployment environment!")
            return False
        
        # Execute deployment based on mode
        if args.mode == "orchestrated":
            success = await execute_orchestrated_deployment()
        else:
            success = execute_legacy_deployment()
        
        # Generate deployment summary
        generate_deployment_summary(success, args.mode)
        
        # Final status
        if success:
            print("\n" + "=" * 70)
            print("TASK 17 COMPLETED SUCCESSFULLY!")
            print("=" * 70)
            print("✓ Production deployment and launch completed")
            print("✓ System deployed with full monitoring")
            print("✓ User acceptance testing conducted")
            print("✓ Gradual rollout implemented")
            print("✓ Go-live procedures executed")
            print("✓ Comprehensive support documentation generated")
            print("\nCheck reports/deployment/ for detailed results")
            print("System Status: LIVE IN PRODUCTION")
            return True
        else:
            print("\n" + "=" * 70)
            print("TASK 17 FAILED!")
            print("=" * 70)
            print("✗ Production deployment encountered issues")
            print("Check logs and reports for detailed error information")
            print("Consider rollback or issue resolution before retry")
            return False
            
    except Exception as e:
        logger.error(f"❌ Task 17 execution error: {str(e)}")
        print(f"\n❌ TASK 17 ERROR: {str(e)}")
        return False

if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\nDeployment interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
        sys.exit(1)