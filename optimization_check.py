#!/usr/bin/env python3
"""
ScrollIntel Optimization Assessment Tool
Comprehensive check of application optimization status
"""

import os
import sys
import json
import time
import asyncio
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

class OptimizationChecker:
    def __init__(self):
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "overall_score": 0,
            "categories": {},
            "recommendations": [],
            "critical_issues": [],
            "warnings": []
        }
    
    def check_environment_config(self) -> Dict[str, Any]:
        """Check environment configuration optimization"""
        score = 0
        issues = []
        
        # Check if .env file exists and has required variables
        env_file = Path('.env')
        if env_file.exists():
            score += 20
            with open(env_file, 'r') as f:
                env_content = f.read()
                
            # Check for production optimizations
            if 'DEBUG=false' in env_content:
                score += 10
            else:
                issues.append("DEBUG should be false for production")
                
            if 'ENVIRONMENT=production' in env_content:
                score += 10
            else:
                issues.append("ENVIRONMENT should be set to production")
                
            # Check for required API keys
            required_keys = ['OPENAI_API_KEY', 'DATABASE_URL', 'JWT_SECRET_KEY']
            for key in required_keys:
                if key in env_content and not f'{key}=' in env_content:
                    score += 5
                else:
                    issues.append(f"Missing or empty {key}")
        else:
            issues.append("No .env file found")
            
        return {
            "score": min(score, 100),
            "issues": issues,
            "status": "optimized" if score >= 80 else "needs_improvement"
        }
    
    def check_database_optimization(self) -> Dict[str, Any]:
        """Check database configuration and optimization"""
        score = 0
        issues = []
        
        # Check for database files and configuration
        db_files = [
            'scrollintel.db',
            'scrollintel_dev.db',
            'alembic.ini'
        ]
        
        for db_file in db_files:
            if Path(db_file).exists():
                score += 15
                
        # Check for migration files
        alembic_dir = Path('alembic')
        if alembic_dir.exists() and (alembic_dir / 'versions').exists():
            score += 20
        else:
            issues.append("Database migrations not properly configured")
            
        # Check for database optimization scripts
        if Path('scripts/optimize-database.py').exists():
            score += 15
        else:
            issues.append("Database optimization scripts missing")
            
        return {
            "score": min(score, 100),
            "issues": issues,
            "status": "optimized" if score >= 70 else "needs_improvement"
        }
    
    def check_performance_monitoring(self) -> Dict[str, Any]:
        """Check performance monitoring setup"""
        score = 0
        issues = []
        
        # Check for monitoring components
        monitoring_files = [
            'scrollintel/core/performance_monitor.py',
            'scrollintel/core/monitoring.py',
            'scrollintel/api/middleware/performance_middleware.py',
            'monitoring/prometheus.yml'
        ]
        
        for file_path in monitoring_files:
            if Path(file_path).exists():
                score += 20
            else:
                issues.append(f"Missing monitoring component: {file_path}")
                
        # Check for bulletproof middleware
        if Path('scrollintel/core/bulletproof_middleware.py').exists():
            score += 20
        else:
            issues.append("Bulletproof middleware not implemented")
            
        return {
            "score": min(score, 100),
            "issues": issues,
            "status": "optimized" if score >= 80 else "needs_improvement"
        }
    
    def check_security_optimization(self) -> Dict[str, Any]:
        """Check security optimization"""
        score = 0
        issues = []
        
        # Check for security components
        security_files = [
            'scrollintel/security/auth.py',
            'scrollintel/api/middleware/security_middleware.py',
            'scrollintel/core/audit_system.py',
            'security/enterprise_security_framework.py'
        ]
        
        for file_path in security_files:
            if Path(file_path).exists():
                score += 20
            else:
                issues.append(f"Missing security component: {file_path}")
                
        # Check for SSL/TLS configuration
        if Path('nginx/nginx.conf').exists():
            score += 20
        else:
            issues.append("Nginx configuration missing")
            
        return {
            "score": min(score, 100),
            "issues": issues,
            "status": "optimized" if score >= 70 else "needs_improvement"
        }
    
    def check_frontend_optimization(self) -> Dict[str, Any]:
        """Check frontend optimization"""
        score = 0
        issues = []
        
        # Check for frontend build configuration
        frontend_files = [
            'frontend/package.json',
            'frontend/next.config.js',
            'frontend/tailwind.config.js',
            'frontend/Dockerfile'
        ]
        
        for file_path in frontend_files:
            if Path(file_path).exists():
                score += 20
            else:
                issues.append(f"Missing frontend file: {file_path}")
                
        # Check for production build optimization
        if Path('frontend/package.json').exists():
            try:
                with open('frontend/package.json', 'r') as f:
                    package_json = json.load(f)
                    if 'build' in package_json.get('scripts', {}):
                        score += 20
                    else:
                        issues.append("No build script in package.json")
            except:
                issues.append("Invalid package.json")
                
        return {
            "score": min(score, 100),
            "issues": issues,
            "status": "optimized" if score >= 80 else "needs_improvement"
        }
    
    def check_deployment_readiness(self) -> Dict[str, Any]:
        """Check deployment readiness"""
        score = 0
        issues = []
        
        # Check for deployment files
        deployment_files = [
            'Dockerfile',
            'docker-compose.yml',
            'docker-compose.prod.yml',
            'requirements.txt',
            'Procfile'
        ]
        
        for file_path in deployment_files:
            if Path(file_path).exists():
                score += 15
            else:
                issues.append(f"Missing deployment file: {file_path}")
                
        # Check for deployment scripts
        deployment_scripts = [
            'scripts/production-deploy.sh',
            'scripts/health-check.py',
            'deploy_now.py'
        ]
        
        for script_path in deployment_scripts:
            if Path(script_path).exists():
                score += 10
                
        return {
            "score": min(score, 100),
            "issues": issues,
            "status": "ready" if score >= 80 else "needs_work"
        }
    
    def check_api_optimization(self) -> Dict[str, Any]:
        """Check API optimization"""
        score = 0
        issues = []
        
        # Check for API components
        api_components = [
            'scrollintel/api/main.py',
            'scrollintel/api/gateway.py',
            'scrollintel/api/routes/',
            'scrollintel/core/rate_limiter.py'
        ]
        
        for component in api_components:
            if Path(component).exists():
                score += 20
            else:
                issues.append(f"Missing API component: {component}")
                
        # Check for caching
        if Path('scrollintel/core/smart_cache_manager.py').exists():
            score += 20
        else:
            issues.append("Caching system not implemented")
            
        return {
            "score": min(score, 100),
            "issues": issues,
            "status": "optimized" if score >= 80 else "needs_improvement"
        }
    
    def run_comprehensive_check(self) -> Dict[str, Any]:
        """Run comprehensive optimization check"""
        print("🔍 Running ScrollIntel Optimization Assessment...")
        
        # Run all checks
        checks = {
            "environment": self.check_environment_config(),
            "database": self.check_database_optimization(),
            "performance": self.check_performance_monitoring(),
            "security": self.check_security_optimization(),
            "frontend": self.check_frontend_optimization(),
            "deployment": self.check_deployment_readiness(),
            "api": self.check_api_optimization()
        }
        
        # Calculate overall score
        total_score = sum(check["score"] for check in checks.values())
        overall_score = total_score / len(checks)
        
        # Collect all issues
        all_issues = []
        critical_issues = []
        warnings = []
        
        for category, check in checks.items():
            for issue in check["issues"]:
                if "Missing" in issue or "not implemented" in issue:
                    critical_issues.append(f"{category}: {issue}")
                else:
                    warnings.append(f"{category}: {issue}")
        
        # Generate recommendations
        recommendations = []
        if overall_score < 70:
            recommendations.append("🚨 CRITICAL: Application needs significant optimization before production use")
        elif overall_score < 85:
            recommendations.append("⚠️ WARNING: Application needs optimization improvements")
        else:
            recommendations.append("✅ GOOD: Application is well optimized")
            
        # Add specific recommendations
        if checks["performance"]["score"] < 80:
            recommendations.append("Implement comprehensive performance monitoring")
        if checks["security"]["score"] < 70:
            recommendations.append("Strengthen security implementation")
        if checks["deployment"]["score"] < 80:
            recommendations.append("Complete deployment configuration")
            
        self.results.update({
            "overall_score": round(overall_score, 2),
            "categories": checks,
            "recommendations": recommendations,
            "critical_issues": critical_issues,
            "warnings": warnings
        })
        
        return self.results
    
    def print_results(self):
        """Print formatted results"""
        results = self.results
        
        print("\n" + "="*60)
        print("📊 SCROLLINTEL OPTIMIZATION ASSESSMENT REPORT")
        print("="*60)
        
        # Overall score
        score = results["overall_score"]
        if score >= 85:
            status_emoji = "🟢"
            status_text = "EXCELLENT"
        elif score >= 70:
            status_emoji = "🟡"
            status_text = "GOOD"
        else:
            status_emoji = "🔴"
            status_text = "NEEDS WORK"
            
        print(f"\n{status_emoji} OVERALL OPTIMIZATION SCORE: {score}/100 ({status_text})")
        
        # Category breakdown
        print("\n📋 CATEGORY BREAKDOWN:")
        for category, data in results["categories"].items():
            score = data["score"]
            status = data["status"]
            emoji = "✅" if score >= 80 else "⚠️" if score >= 60 else "❌"
            print(f"  {emoji} {category.upper()}: {score}/100 ({status})")
        
        # Recommendations
        if results["recommendations"]:
            print("\n💡 RECOMMENDATIONS:")
            for rec in results["recommendations"]:
                print(f"  • {rec}")
        
        # Critical issues
        if results["critical_issues"]:
            print("\n🚨 CRITICAL ISSUES:")
            for issue in results["critical_issues"]:
                print(f"  • {issue}")
        
        # Warnings
        if results["warnings"]:
            print("\n⚠️ WARNINGS:")
            for warning in results["warnings"]:
                print(f"  • {warning}")
        
        print("\n" + "="*60)
        print(f"📅 Assessment completed at: {results['timestamp']}")
        print("="*60)

def main():
    """Main function"""
    checker = OptimizationChecker()
    results = checker.run_comprehensive_check()
    checker.print_results()
    
    # Save results to file
    with open('optimization_assessment_report.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📄 Detailed report saved to: optimization_assessment_report.json")
    
    # Return exit code based on score
    if results["overall_score"] >= 70:
        return 0
    else:
        return 1

if __name__ == "__main__":
    sys.exit(main())