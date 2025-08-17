#!/usr/bin/env python3
"""
Production Launch Checklist
Final verification before going live.
"""

import os
import sys
import subprocess
import json
from datetime import datetime
from pathlib import Path

class ProductionLaunchChecklist:
    """Final production launch checklist."""
    
    def __init__(self):
        self.checks = []
        self.critical_issues = []
    
    def check(self, name: str, status: bool, details: str = ""):
        """Record a check result."""
        self.checks.append({
            "name": name,
            "status": "✅ PASS" if status else "❌ FAIL",
            "details": details
        })
        
        if not status:
            self.critical_issues.append(name)
        
        print(f"{'✅' if status else '❌'} {name}")
        if details:
            print(f"   {details}")
    
    def run_checklist(self):
        """Run the complete production launch checklist."""
        print("🚀 SCROLLINTEL PRODUCTION LAUNCH CHECKLIST")
        print("=" * 60)
        print(f"Launch verification: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # 1. Environment Configuration
        print("🔧 ENVIRONMENT CONFIGURATION")
        print("-" * 30)
        
        # Check environment files
        env_files = ['.env', '.env.production', '.env.example']
        for env_file in env_files:
            exists = os.path.exists(env_file)
            self.check(f"Environment file: {env_file}", exists)
        
        # Check critical environment variables
        critical_vars = ['JWT_SECRET_KEY', 'DATABASE_URL']
        for var in critical_vars:
            has_var = var in os.environ or self._check_env_file(var)
            self.check(f"Environment variable: {var}", has_var)
        
        print()
        
        # 2. Docker Configuration
        print("🐳 DOCKER CONFIGURATION")
        print("-" * 30)
        
        docker_files = ['Dockerfile', 'docker-compose.yml', 'docker-compose.prod.yml']
        for docker_file in docker_files:
            exists = os.path.exists(docker_file)
            self.check(f"Docker file: {docker_file}", exists)
        
        print()
        
        # 3. Frontend Build
        print("🎨 FRONTEND BUILD")
        print("-" * 30)
        
        frontend_files = [
            'frontend/package.json',
            'frontend/next.config.js',
            'frontend/tailwind.config.js'
        ]
        
        for file in frontend_files:
            exists = os.path.exists(file)
            self.check(f"Frontend file: {file}", exists)
        
        # Check if node_modules exists
        node_modules = os.path.exists('frontend/node_modules')
        self.check("Frontend dependencies installed", node_modules, 
                  "Run 'npm install' in frontend directory if missing")
        
        print()
        
        # 4. Database Setup
        print("🗄️ DATABASE SETUP")
        print("-" * 30)
        
        db_files = [
            'scrollintel/models/database.py',
            'alembic.ini',
            'init_database.py'
        ]
        
        for file in db_files:
            exists = os.path.exists(file)
            self.check(f"Database file: {file}", exists)
        
        print()
        
        # 5. Security Configuration
        print("🔒 SECURITY CONFIGURATION")
        print("-" * 30)
        
        security_files = [
            'scrollintel/security/auth.py',
            'scrollintel/security/permissions.py',
            'scrollintel/security/middleware.py'
        ]
        
        for file in security_files:
            exists = os.path.exists(file)
            self.check(f"Security file: {file}", exists)
        
        print()
        
        # 6. Monitoring Setup
        print("📊 MONITORING SETUP")
        print("-" * 30)
        
        monitoring_files = [
            'monitoring/prometheus.yml',
            'monitoring/alert_rules.yml',
            'nginx/nginx.conf'
        ]
        
        for file in monitoring_files:
            exists = os.path.exists(file)
            self.check(f"Monitoring file: {file}", exists)
        
        print()
        
        # 7. Documentation
        print("📚 DOCUMENTATION")
        print("-" * 30)
        
        doc_files = [
            'README.md',
            'QUICK_START_GUIDE.md',
            'docs/DEPLOYMENT.md',
            'docs/TROUBLESHOOTING.md'
        ]
        
        for file in doc_files:
            exists = os.path.exists(file)
            self.check(f"Documentation: {file}", exists)
        
        print()
        
        # 8. Test Coverage
        print("🧪 TEST COVERAGE")
        print("-" * 30)
        
        test_dirs = ['tests', 'frontend/src/__tests__']
        for test_dir in test_dirs:
            exists = os.path.exists(test_dir)
            if exists:
                test_count = len(list(Path(test_dir).rglob('*.py'))) + len(list(Path(test_dir).rglob('*.test.*')))
                self.check(f"Test directory: {test_dir}", True, f"{test_count} test files")
            else:
                self.check(f"Test directory: {test_dir}", False)
        
        print()
        
        # Generate final report
        self.generate_launch_report()
    
    def _check_env_file(self, var_name: str) -> bool:
        """Check if variable exists in .env files."""
        env_files = ['.env', '.env.production']
        for env_file in env_files:
            if os.path.exists(env_file):
                with open(env_file, 'r') as f:
                    content = f.read()
                    if f"{var_name}=" in content:
                        return True
        return False
    
    def generate_launch_report(self):
        """Generate final launch report."""
        total_checks = len(self.checks)
        passed_checks = len([c for c in self.checks if "✅" in c["status"]])
        
        print("📋 LAUNCH READINESS REPORT")
        print("=" * 60)
        print(f"Total Checks: {total_checks}")
        print(f"Passed: {passed_checks}")
        print(f"Failed: {len(self.critical_issues)}")
        print(f"Success Rate: {(passed_checks/total_checks)*100:.1f}%")
        print()
        
        if self.critical_issues:
            print("❌ CRITICAL ISSUES TO RESOLVE:")
            for issue in self.critical_issues:
                print(f"  - {issue}")
            print()
            print("🚨 LAUNCH STATUS: NOT READY")
            print("Resolve critical issues before production deployment.")
        else:
            print("✅ LAUNCH STATUS: READY FOR PRODUCTION!")
            print()
            print("🚀 NEXT STEPS:")
            print("1. Run: docker-compose -f docker-compose.prod.yml up -d")
            print("2. Verify health endpoints are responding")
            print("3. Run smoke tests against production")
            print("4. Monitor logs and metrics")
            print("5. Announce launch! 🎉")
        
        print()
        print("Launch checklist completed at:", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))


def main():
    """Run the production launch checklist."""
    checklist = ProductionLaunchChecklist()
    checklist.run_checklist()
    
    return 0 if not checklist.critical_issues else 1


if __name__ == "__main__":
    exit(main())