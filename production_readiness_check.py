#!/usr/bin/env python3
"""
ScrollIntel Production Readiness Assessment
Comprehensive evaluation of system readiness for production deployment.
"""

import os
import sys
from datetime import datetime

def main():
    print('🔍 SCROLLINTEL PRODUCTION READINESS ASSESSMENT')
    print('=' * 60)
    print(f'Assessment Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print()

    # 1. Basic Import Test
    print('📦 1. BASIC IMPORT TEST')
    print('-' * 30)
    try:
        import scrollintel
        from scrollintel.core import config
        from scrollintel.agents import scroll_cto_agent
        from scrollintel.api import gateway
        print('✅ Core modules import successfully')
        import_score = 100
    except Exception as e:
        print(f'❌ Import failed: {e}')
        import_score = 0

    print(f'Import Score: {import_score}%')
    print()

    # 2. File Structure Analysis
    print('📁 2. FILE STRUCTURE ANALYSIS')
    print('-' * 30)

    # Count key components
    agents_count = len([f for f in os.listdir('scrollintel/agents') if f.endswith('.py') and not f.startswith('__')])
    engines_count = len([f for f in os.listdir('scrollintel/engines') if f.endswith('.py') and not f.startswith('__')])
    api_routes_count = len([f for f in os.listdir('scrollintel/api/routes') if f.endswith('.py') and not f.startswith('__')])

    print(f'Agents: {agents_count} files')
    print(f'Engines: {engines_count} files')
    print(f'API Routes: {api_routes_count} files')

    # Check for key files
    key_files = [
        'scrollintel/core/config.py',
        'scrollintel/models/database.py',
        'scrollintel/api/gateway.py',
        'requirements.txt',
        'Dockerfile',
        'docker-compose.yml'
    ]

    missing_files = []
    for file in key_files:
        if not os.path.exists(file):
            missing_files.append(file)

    if missing_files:
        print(f'❌ Missing key files: {missing_files}')
        structure_score = 70
    else:
        print('✅ All key files present')
        structure_score = 100

    print(f'Structure Score: {structure_score}%')
    print()

    # 3. Configuration Check
    print('⚙️ 3. CONFIGURATION CHECK')
    print('-' * 30)
    try:
        from scrollintel.core.config import get_config
        config = get_config()
        print('✅ Configuration loads successfully')
        
        # Check environment files
        env_files = ['.env', '.env.example', '.env.production', '.env.test']
        existing_env = [f for f in env_files if os.path.exists(f)]
        print(f'Environment files: {existing_env}')
        
        config_score = 90 if len(existing_env) >= 2 else 70
        
    except Exception as e:
        print(f'❌ Configuration failed: {e}')
        config_score = 0

    print(f'Configuration Score: {config_score}%')
    print()

    # 4. Test Coverage Analysis
    print('🧪 4. TEST COVERAGE ANALYSIS')
    print('-' * 30)

    test_files = []
    for root, dirs, files in os.walk('tests'):
        for file in files:
            if file.startswith('test_') and file.endswith('.py'):
                test_files.append(file)

    integration_tests = [f for f in test_files if 'integration' in f]
    unit_tests = [f for f in test_files if 'integration' not in f]

    print(f'Total test files: {len(test_files)}')
    print(f'Unit tests: {len(unit_tests)}')
    print(f'Integration tests: {len(integration_tests)}')

    if len(test_files) > 50:
        test_score = 100
    elif len(test_files) > 30:
        test_score = 80
    elif len(test_files) > 10:
        test_score = 60
    else:
        test_score = 40

    print(f'Test Coverage Score: {test_score}%')
    print()

    # 5. Security Features
    print('🔒 5. SECURITY FEATURES')
    print('-' * 30)

    security_features = []
    if os.path.exists('scrollintel/security'):
        security_files = os.listdir('scrollintel/security')
        if 'auth.py' in security_files:
            security_features.append('Authentication')
        if 'permissions.py' in security_files:
            security_features.append('Authorization')
        if 'audit.py' in security_files:
            security_features.append('Audit Logging')
        if 'middleware.py' in security_files:
            security_features.append('Security Middleware')

    print(f'Security features: {security_features}')
    security_score = min(100, len(security_features) * 25)
    print(f'Security Score: {security_score}%')
    print()

    # 6. Deployment Readiness
    print('🚀 6. DEPLOYMENT READINESS')
    print('-' * 30)

    deployment_files = [
        'Dockerfile',
        'docker-compose.yml',
        'docker-compose.prod.yml',
        'requirements.txt',
        'nginx/nginx.conf'
    ]

    deployment_ready = []
    for file in deployment_files:
        if os.path.exists(file):
            deployment_ready.append(file)

    print(f'Deployment files: {deployment_ready}')
    deployment_score = min(100, len(deployment_ready) * 20)
    print(f'Deployment Score: {deployment_score}%')
    print()

    # 7. Frontend Assessment
    print('🎨 7. FRONTEND ASSESSMENT')
    print('-' * 30)
    
    frontend_files = []
    if os.path.exists('frontend'):
        if os.path.exists('frontend/package.json'):
            frontend_files.append('package.json')
        if os.path.exists('frontend/next.config.js'):
            frontend_files.append('next.config.js')
        if os.path.exists('frontend/tailwind.config.js'):
            frontend_files.append('tailwind.config.js')
        if os.path.exists('frontend/src/app'):
            frontend_files.append('Next.js App Router')
        if os.path.exists('frontend/src/components'):
            frontend_files.append('React Components')
    
    print(f'Frontend files: {frontend_files}')
    frontend_score = min(100, len(frontend_files) * 20)
    print(f'Frontend Score: {frontend_score}%')
    print()

    # 8. Database Assessment
    print('🗄️ 8. DATABASE ASSESSMENT')
    print('-' * 30)
    
    db_files = []
    if os.path.exists('scrollintel/models/database.py'):
        db_files.append('Database Models')
    if os.path.exists('scrollintel/models/database_utils.py'):
        db_files.append('Database Utils')
    if os.path.exists('alembic.ini'):
        db_files.append('Alembic Migrations')
    if os.path.exists('scrollintel/models/seed_data.py'):
        db_files.append('Seed Data')
    
    print(f'Database files: {db_files}')
    db_score = min(100, len(db_files) * 25)
    print(f'Database Score: {db_score}%')
    print()

    # Calculate Overall Score
    overall_score = (
        import_score * 0.15 +
        structure_score * 0.15 +
        config_score * 0.10 +
        test_score * 0.20 +
        security_score * 0.15 +
        deployment_score * 0.10 +
        frontend_score * 0.10 +
        db_score * 0.05
    )

    print('📊 OVERALL ASSESSMENT')
    print('=' * 60)
    print(f'Import Test: {import_score}%')
    print(f'Structure: {structure_score}%')
    print(f'Configuration: {config_score}%')
    print(f'Test Coverage: {test_score}%')
    print(f'Security: {security_score}%')
    print(f'Deployment: {deployment_score}%')
    print(f'Frontend: {frontend_score}%')
    print(f'Database: {db_score}%')
    print('-' * 60)
    print(f'OVERALL PRODUCTION READINESS: {overall_score:.1f}%')
    print()

    if overall_score >= 90:
        status = '🟢 PRODUCTION READY'
        recommendations = [
            "System is production ready!",
            "Consider performance testing under load",
            "Set up monitoring and alerting"
        ]
    elif overall_score >= 75:
        status = '🟡 MOSTLY READY - Minor fixes needed'
        recommendations = [
            "Fix import issues if any",
            "Complete missing security features",
            "Add more comprehensive tests"
        ]
    elif overall_score >= 60:
        status = '🟠 NEEDS WORK - Major improvements required'
        recommendations = [
            "Fix critical import and configuration issues",
            "Implement missing security features",
            "Add comprehensive test coverage",
            "Complete deployment configuration"
        ]
    else:
        status = '🔴 NOT READY - Significant development needed'
        recommendations = [
            "Fix fundamental import and structure issues",
            "Implement core security features",
            "Add basic test coverage",
            "Complete basic deployment setup"
        ]

    print(f'STATUS: {status}')
    print()
    print('📋 RECOMMENDATIONS:')
    for i, rec in enumerate(recommendations, 1):
        print(f'{i}. {rec}')
    
    return overall_score

if __name__ == "__main__":
    score = main()
    sys.exit(0 if score >= 75 else 1)