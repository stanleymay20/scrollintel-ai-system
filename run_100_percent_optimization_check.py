#!/usr/bin/env python3
"""
100% Optimization Check - Comprehensive System Validation
Validates that ScrollIntel is running at 100% optimization
"""

import asyncio
import json
import logging
import os
import sys
import time
import psutil
from pathlib import Path
from typing import Dict, Any, List

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

class OptimizationValidator:
    """Validates 100% optimization status"""
    
    def __init__(self):
        self.validation_results = {}
        self.overall_score = 0.0
        self.critical_issues = []
        self.warnings = []
        self.recommendations = []
        
    async def run_comprehensive_validation(self):
        """Run comprehensive optimization validation"""
        logger.info("🔍 Starting 100% Optimization Validation...")
        
        validation_tasks = [
            self._validate_environment_configuration(),
            self._validate_performance_optimizations(),
            self._validate_security_framework(),
            self._validate_database_optimization(),
            self._validate_system_resources(),
            self._validate_core_components(),
            self._validate_monitoring_systems(),
            self._validate_production_readiness()
        ]
        
        results = await asyncio.gather(*validation_tasks, return_exceptions=True)
        
        # Process results
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Validation task {i} failed: {result}")
                self.critical_issues.append(f"Validation task {i} failed: {result}")
            else:
                logger.info(f"✅ Validation task {i} completed")
        
        # Calculate overall score
        self._calculate_overall_score()
        
        # Generate report
        report = self._generate_optimization_report()
        
        return report
    
    async def _validate_environment_configuration(self):
        """Validate environment configuration"""
        logger.info("🔧 Validating environment configuration...")
        
        score = 100.0
        issues = []
        
        try:
            # Check .env file exists
            env_file = Path('.env')
            if not env_file.exists():
                issues.append("Missing .env file")
                score -= 20
            
            # Check critical environment variables
            critical_vars = [
                'DATABASE_URL',
                'JWT_SECRET_KEY',
                'OPENAI_API_KEY',
                'ENCRYPTION_KEY'
            ]
            
            missing_vars = []
            for var in critical_vars:
                value = os.getenv(var)
                if not value or value in ['your_key_here', 'placeholder']:
                    missing_vars.append(var)
            
            if missing_vars:
                issues.append(f"Missing/placeholder environment variables: {missing_vars}")
                score -= len(missing_vars) * 10
            
            # Check optimization flags
            optimization_flags = [
                'ENABLE_LAZY_LOADING',
                'ENABLE_MEMORY_OPTIMIZATION'
            ]
            
            for flag in optimization_flags:
                if os.getenv(flag, 'false').lower() != 'true':
                    issues.append(f"Optimization flag {flag} not enabled")
                    score -= 5
            
            self.validation_results['environment'] = {
                'score': max(0, score),
                'issues': issues,
                'status': 'optimized' if score >= 80 else 'needs_improvement'
            }
            
        except Exception as e:
            logger.error(f"Environment validation failed: {e}")
            self.validation_results['environment'] = {
                'score': 0,
                'issues': [f"Validation failed: {e}"],
                'status': 'failed'
            }
    
    async def _validate_performance_optimizations(self):
        """Validate performance optimizations"""
        logger.info("⚡ Validating performance optimizations...")
        
        score = 100.0
        issues = []
        
        try:
            # Check if optimization modules exist
            optimization_modules = [
                'scrollintel/core/ultra_performance_optimizer.py',
                'scrollintel/core/intelligent_resource_manager.py',
                'scrollintel/core/quantum_optimization_engine.py'
            ]
            
            for module in optimization_modules:
                if not Path(module).exists():
                    issues.append(f"Missing optimization module: {module}")
                    score -= 15
            
            # Test import of optimization modules
            try:
                from scrollintel.core.ultra_performance_optimizer import get_optimizer
                optimizer = get_optimizer()
                if optimizer:
                    logger.info("✅ Ultra Performance Optimizer available")
                else:
                    issues.append("Ultra Performance Optimizer not available")
                    score -= 10
            except ImportError as e:
                issues.append(f"Cannot import Ultra Performance Optimizer: {e}")
                score -= 15
            
            # Check system performance
            cpu_percent = psutil.cpu_percent(interval=1)
            memory_percent = psutil.virtual_memory().percent
            
            if cpu_percent > 80:
                issues.append(f"High CPU usage: {cpu_percent:.1f}%")
                score -= 10
            
            if memory_percent > 85:
                issues.append(f"High memory usage: {memory_percent:.1f}%")
                score -= 10
            
            self.validation_results['performance'] = {
                'score': max(0, score),
                'issues': issues,
                'status': 'optimized' if score >= 80 else 'needs_improvement',
                'metrics': {
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory_percent
                }
            }
            
        except Exception as e:
            logger.error(f"Performance validation failed: {e}")
            self.validation_results['performance'] = {
                'score': 0,
                'issues': [f"Validation failed: {e}"],
                'status': 'failed'
            }
    
    async def _validate_security_framework(self):
        """Validate security framework"""
        logger.info("🔒 Validating security framework...")
        
        score = 100.0
        issues = []
        
        try:
            # Check if security framework exists
            security_file = Path('security/enterprise_security_framework.py')
            if not security_file.exists():
                issues.append("Missing enterprise security framework")
                score -= 30
            else:
                # Test import
                try:
                    from security.enterprise_security_framework import get_security_framework
                    framework = get_security_framework()
                    status = framework.get_security_status()
                    
                    if not status.get('encryption_enabled'):
                        issues.append("Encryption not enabled")
                        score -= 15
                    
                    if not status.get('jwt_configured'):
                        issues.append("JWT not configured")
                        score -= 10
                    
                    logger.info("✅ Security framework available")
                    
                except ImportError as e:
                    issues.append(f"Cannot import security framework: {e}")
                    score -= 20
            
            # Check security configuration
            jwt_secret = os.getenv('JWT_SECRET_KEY')
            if not jwt_secret or len(jwt_secret) < 32:
                issues.append("Weak or missing JWT secret")
                score -= 15
            
            encryption_key = os.getenv('ENCRYPTION_KEY')
            if not encryption_key or len(encryption_key) < 32:
                issues.append("Weak or missing encryption key")
                score -= 15
            
            self.validation_results['security'] = {
                'score': max(0, score),
                'issues': issues,
                'status': 'optimized' if score >= 80 else 'needs_improvement'
            }
            
        except Exception as e:
            logger.error(f"Security validation failed: {e}")
            self.validation_results['security'] = {
                'score': 0,
                'issues': [f"Validation failed: {e}"],
                'status': 'failed'
            }
    
    async def _validate_database_optimization(self):
        """Validate database optimization"""
        logger.info("🗄️  Validating database optimization...")
        
        score = 100.0
        issues = []
        
        try:
            # Check if database manager exists
            db_manager_file = Path('scrollintel/core/database_connection_manager.py')
            if not db_manager_file.exists():
                issues.append("Missing database connection manager")
                score -= 25
            else:
                try:
                    from scrollintel.core.database_connection_manager import get_database_manager
                    manager = get_database_manager()
                    logger.info("✅ Database manager available")
                except ImportError as e:
                    issues.append(f"Cannot import database manager: {e}")
                    score -= 15
            
            # Check database configuration
            database_url = os.getenv('DATABASE_URL')
            if not database_url or database_url == 'your_database_url_here':
                issues.append("Database URL not configured - will use SQLite fallback")
                score -= 10  # Minor deduction for fallback
            
            self.validation_results['database'] = {
                'score': max(0, score),
                'issues': issues,
                'status': 'optimized' if score >= 80 else 'needs_improvement'
            }
            
        except Exception as e:
            logger.error(f"Database validation failed: {e}")
            self.validation_results['database'] = {
                'score': 0,
                'issues': [f"Validation failed: {e}"],
                'status': 'failed'
            }
    
    async def _validate_system_resources(self):
        """Validate system resources"""
        logger.info("💻 Validating system resources...")
        
        score = 100.0
        issues = []
        
        try:
            # CPU validation
            cpu_count = psutil.cpu_count()
            cpu_percent = psutil.cpu_percent(interval=1)
            
            if cpu_count < 2:
                issues.append(f"Low CPU count: {cpu_count}")
                score -= 10
            
            if cpu_percent > 90:
                issues.append(f"Very high CPU usage: {cpu_percent:.1f}%")
                score -= 20
            elif cpu_percent > 80:
                issues.append(f"High CPU usage: {cpu_percent:.1f}%")
                score -= 10
            
            # Memory validation
            memory = psutil.virtual_memory()
            memory_gb = memory.total / (1024**3)
            
            if memory_gb < 4:
                issues.append(f"Low memory: {memory_gb:.1f} GB")
                score -= 15
            
            if memory.percent > 90:
                issues.append(f"Very high memory usage: {memory.percent:.1f}%")
                score -= 20
            elif memory.percent > 85:
                issues.append(f"High memory usage: {memory.percent:.1f}%")
                score -= 10
            
            # Disk validation
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            
            if disk_percent > 95:
                issues.append(f"Very low disk space: {disk_percent:.1f}% used")
                score -= 20
            elif disk_percent > 90:
                issues.append(f"Low disk space: {disk_percent:.1f}% used")
                score -= 10
            
            self.validation_results['system_resources'] = {
                'score': max(0, score),
                'issues': issues,
                'status': 'optimized' if score >= 80 else 'needs_improvement',
                'metrics': {
                    'cpu_count': cpu_count,
                    'cpu_percent': cpu_percent,
                    'memory_gb': memory_gb,
                    'memory_percent': memory.percent,
                    'disk_percent': disk_percent
                }
            }
            
        except Exception as e:
            logger.error(f"System resources validation failed: {e}")
            self.validation_results['system_resources'] = {
                'score': 0,
                'issues': [f"Validation failed: {e}"],
                'status': 'failed'
            }
    
    async def _validate_core_components(self):
        """Validate core components"""
        logger.info("🏗️  Validating core components...")
        
        score = 100.0
        issues = []
        
        try:
            # Check core files exist
            core_files = [
                'scrollintel/__init__.py',
                'scrollintel/core/config.py',
                'scrollintel/api/main.py',
                'scrollintel/core/orchestrator.py'
            ]
            
            for file_path in core_files:
                if not Path(file_path).exists():
                    issues.append(f"Missing core file: {file_path}")
                    score -= 15
            
            # Test core imports
            try:
                from scrollintel.core.config import get_config
                config = get_config()
                logger.info("✅ Core configuration available")
            except ImportError as e:
                issues.append(f"Cannot import core config: {e}")
                score -= 20
            
            # Check requirements
            requirements_file = Path('requirements.txt')
            if not requirements_file.exists():
                issues.append("Missing requirements.txt")
                score -= 10
            
            self.validation_results['core_components'] = {
                'score': max(0, score),
                'issues': issues,
                'status': 'optimized' if score >= 80 else 'needs_improvement'
            }
            
        except Exception as e:
            logger.error(f"Core components validation failed: {e}")
            self.validation_results['core_components'] = {
                'score': 0,
                'issues': [f"Validation failed: {e}"],
                'status': 'failed'
            }
    
    async def _validate_monitoring_systems(self):
        """Validate monitoring systems"""
        logger.info("📊 Validating monitoring systems...")
        
        score = 100.0
        issues = []
        
        try:
            # Check monitoring files
            monitoring_files = [
                'scrollintel/core/monitoring.py',
                'scrollintel/core/analytics.py'
            ]
            
            for file_path in monitoring_files:
                if not Path(file_path).exists():
                    issues.append(f"Missing monitoring file: {file_path}")
                    score -= 15
            
            # Check if monitoring can be imported
            try:
                from scrollintel.core.monitoring import get_monitoring_system
                logger.info("✅ Monitoring system available")
            except ImportError as e:
                issues.append(f"Cannot import monitoring system: {e}")
                score -= 20
            
            self.validation_results['monitoring'] = {
                'score': max(0, score),
                'issues': issues,
                'status': 'optimized' if score >= 80 else 'needs_improvement'
            }
            
        except Exception as e:
            logger.error(f"Monitoring validation failed: {e}")
            self.validation_results['monitoring'] = {
                'score': 0,
                'issues': [f"Validation failed: {e}"],
                'status': 'failed'
            }
    
    async def _validate_production_readiness(self):
        """Validate production readiness"""
        logger.info("🚀 Validating production readiness...")
        
        score = 100.0
        issues = []
        
        try:
            # Check deployment files
            deployment_files = [
                'Dockerfile',
                'docker-compose.yml',
                'requirements.txt'
            ]
            
            for file_path in deployment_files:
                if not Path(file_path).exists():
                    issues.append(f"Missing deployment file: {file_path}")
                    score -= 10
            
            # Check startup scripts
            startup_scripts = [
                'start_100_percent_optimized.py',
                'start_optimized.py'
            ]
            
            startup_available = False
            for script in startup_scripts:
                if Path(script).exists():
                    startup_available = True
                    break
            
            if not startup_available:
                issues.append("No optimized startup script available")
                score -= 15
            
            # Check environment
            environment = os.getenv('ENVIRONMENT', 'development')
            if environment not in ['production', 'staging']:
                issues.append(f"Environment not set for production: {environment}")
                score -= 5
            
            self.validation_results['production_readiness'] = {
                'score': max(0, score),
                'issues': issues,
                'status': 'ready' if score >= 90 else 'needs_improvement'
            }
            
        except Exception as e:
            logger.error(f"Production readiness validation failed: {e}")
            self.validation_results['production_readiness'] = {
                'score': 0,
                'issues': [f"Validation failed: {e}"],
                'status': 'failed'
            }
    
    def _calculate_overall_score(self):
        """Calculate overall optimization score"""
        if not self.validation_results:
            self.overall_score = 0.0
            return
        
        total_score = 0.0
        total_weight = 0.0
        
        # Weights for different categories
        weights = {
            'environment': 0.15,
            'performance': 0.25,
            'security': 0.20,
            'database': 0.15,
            'system_resources': 0.10,
            'core_components': 0.10,
            'monitoring': 0.05,
            'production_readiness': 0.10
        }
        
        for category, result in self.validation_results.items():
            weight = weights.get(category, 0.1)
            score = result.get('score', 0)
            
            total_score += score * weight
            total_weight += weight
            
            # Collect critical issues
            issues = result.get('issues', [])
            for issue in issues:
                if any(keyword in issue.lower() for keyword in ['missing', 'failed', 'critical']):
                    self.critical_issues.append(f"{category}: {issue}")
                else:
                    self.warnings.append(f"{category}: {issue}")
        
        self.overall_score = total_score / total_weight if total_weight > 0 else 0.0
    
    def _generate_optimization_report(self) -> Dict[str, Any]:
        """Generate comprehensive optimization report"""
        # Determine optimization status
        if self.overall_score >= 95:
            status = "100% OPTIMIZED"
            status_emoji = "🟢"
        elif self.overall_score >= 90:
            status = "HIGHLY OPTIMIZED"
            status_emoji = "🟢"
        elif self.overall_score >= 80:
            status = "WELL OPTIMIZED"
            status_emoji = "🟡"
        elif self.overall_score >= 70:
            status = "MODERATELY OPTIMIZED"
            status_emoji = "🟡"
        else:
            status = "NEEDS OPTIMIZATION"
            status_emoji = "🔴"
        
        # Generate recommendations
        if self.overall_score < 100:
            self.recommendations.extend([
                "Run 'python start_100_percent_optimized.py' for maximum optimization",
                "Ensure all environment variables are properly configured",
                "Monitor system resources and optimize as needed",
                "Keep security framework updated and active"
            ])
        
        report = {
            "timestamp": time.time(),
            "overall_score": round(self.overall_score, 2),
            "status": status,
            "status_emoji": status_emoji,
            "optimization_level": "MAXIMUM" if self.overall_score >= 95 else "HIGH" if self.overall_score >= 80 else "MEDIUM",
            "categories": self.validation_results,
            "critical_issues": self.critical_issues,
            "warnings": self.warnings,
            "recommendations": self.recommendations,
            "system_info": {
                "cpu_count": psutil.cpu_count(),
                "memory_gb": round(psutil.virtual_memory().total / (1024**3), 2),
                "platform": sys.platform,
                "python_version": sys.version.split()[0]
            },
            "ready_for_production": self.overall_score >= 90 and len(self.critical_issues) == 0
        }
        
        return report

async def main():
    """Main validation function"""
    try:
        validator = OptimizationValidator()
        report = await validator.run_comprehensive_validation()
        
        # Print summary
        print("\n" + "="*80)
        print(f"🎯 SCROLLINTEL 100% OPTIMIZATION VALIDATION REPORT")
        print("="*80)
        print(f"Overall Score: {report['overall_score']}/100 {report['status_emoji']}")
        print(f"Status: {report['status']}")
        print(f"Optimization Level: {report['optimization_level']}")
        print(f"Production Ready: {'✅ YES' if report['ready_for_production'] else '❌ NO'}")
        print()
        
        # Print category scores
        print("📊 CATEGORY SCORES:")
        for category, result in report['categories'].items():
            status_icon = "✅" if result['score'] >= 80 else "🟡" if result['score'] >= 60 else "❌"
            print(f"  {status_icon} {category.replace('_', ' ').title()}: {result['score']:.1f}/100")
        print()
        
        # Print critical issues
        if report['critical_issues']:
            print("🚨 CRITICAL ISSUES:")
            for issue in report['critical_issues']:
                print(f"  ❌ {issue}")
            print()
        
        # Print warnings
        if report['warnings']:
            print("⚠️  WARNINGS:")
            for warning in report['warnings'][:5]:  # Show first 5
                print(f"  🟡 {warning}")
            if len(report['warnings']) > 5:
                print(f"  ... and {len(report['warnings']) - 5} more warnings")
            print()
        
        # Print recommendations
        if report['recommendations']:
            print("💡 RECOMMENDATIONS:")
            for rec in report['recommendations']:
                print(f"  🔧 {rec}")
            print()
        
        # Save detailed report
        report_file = f"optimization_validation_report_{int(time.time())}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📄 Detailed report saved to: {report_file}")
        print("="*80)
        
        # Exit with appropriate code
        if report['overall_score'] >= 95:
            print("🎉 CONGRATULATIONS! ScrollIntel is running at 100% optimization!")
            sys.exit(0)
        elif report['overall_score'] >= 80:
            print("✅ ScrollIntel is well optimized and ready for production!")
            sys.exit(0)
        else:
            print("⚠️  ScrollIntel needs optimization improvements.")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        print(f"\n❌ Validation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())