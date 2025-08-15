#!/usr/bin/env python3
"""
ScrollIntel Monitoring System Validation Script
Validates that all monitoring components are properly implemented and functional
"""

import sys
import asyncio
import json
from pathlib import Path
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def validate_file_exists(file_path: Path, description: str) -> bool:
    """Validate that a file exists"""
    if file_path.exists():
        logger.info(f"✓ {description}: {file_path}")
        return True
    else:
        logger.error(f"✗ {description}: {file_path} (NOT FOUND)")
        return False

def validate_monitoring_files() -> bool:
    """Validate all monitoring files exist"""
    logger.info("Validating monitoring files...")
    
    files_to_check = [
        # Core monitoring modules
        (project_root / "scrollintel/core/monitoring.py", "Core monitoring module"),
        (project_root / "scrollintel/core/uptime_monitor.py", "Uptime monitoring module"),
        (project_root / "scrollintel/core/log_aggregation.py", "Log aggregation module"),
        
        # API routes
        (project_root / "scrollintel/api/routes/status_routes.py", "Status API routes"),
        
        # Configuration files
        (project_root / "monitoring/prometheus.yml", "Prometheus configuration"),
        (project_root / "monitoring/alert_rules.yml", "Alert rules configuration"),
        (project_root / "monitoring/alertmanager.yml", "Alertmanager configuration"),
        (project_root / "monitoring/grafana-dashboard-comprehensive.json", "Grafana dashboard"),
        
        # Documentation
        (project_root / "docs/incident-response-runbooks.md", "Incident response runbooks"),
        
        # Scripts
        (project_root / "scripts/start-monitoring.py", "Monitoring startup script"),
        
        # Frontend components
        (project_root / "frontend/src/pages/status.tsx", "Status page component"),
        (project_root / "frontend/src/components/monitoring/comprehensive-monitoring-dashboard.tsx", "Monitoring dashboard"),
        
        # Tests
        (project_root / "tests/test_monitoring_comprehensive.py", "Comprehensive monitoring tests"),
    ]
    
    all_exist = True
    for file_path, description in files_to_check:
        if not validate_file_exists(file_path, description):
            all_exist = False
    
    return all_exist

def validate_monitoring_imports() -> bool:
    """Validate that monitoring modules can be imported"""
    logger.info("Validating monitoring module imports...")
    
    try:
        from scrollintel.core.monitoring import MetricsCollector, PerformanceMonitor, metrics_collector
        logger.info("✓ Core monitoring imports successful")
        
        from scrollintel.core.uptime_monitor import UptimeMonitor, HealthCheck, ServiceStatus
        logger.info("✓ Uptime monitoring imports successful")
        
        from scrollintel.core.log_aggregation import LogAggregator, LogEntry, log_aggregator
        logger.info("✓ Log aggregation imports successful")
        
        return True
        
    except ImportError as e:
        logger.error(f"✗ Import error: {e}")
        return False

def validate_monitoring_functionality() -> bool:
    """Validate basic monitoring functionality"""
    logger.info("Validating monitoring functionality...")
    
    try:
        # Test metrics collector
        from scrollintel.core.monitoring import MetricsCollector
        collector = MetricsCollector()
        
        # Test recording metrics
        collector.record_request("GET", "/test", 200, 0.5)
        collector.record_agent_request("test_agent", "success", 1.0)
        collector.record_error("TestError", "test")
        
        # Test system metrics collection
        metrics = collector.collect_system_metrics()
        if metrics is None:
            logger.error("✗ System metrics collection failed")
            return False
        
        logger.info("✓ Metrics collector functionality validated")
        
        # Test uptime monitor
        from scrollintel.core.uptime_monitor import UptimeMonitor
        uptime_monitor = UptimeMonitor()
        
        # Test status page data generation
        status_data = uptime_monitor.get_status_page_data()
        if not isinstance(status_data, dict):
            logger.error("✗ Status page data generation failed")
            return False
        
        logger.info("✓ Uptime monitor functionality validated")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Functionality validation error: {e}")
        return False

async def validate_async_functionality() -> bool:
    """Validate asynchronous monitoring functionality"""
    logger.info("Validating async monitoring functionality...")
    
    try:
        # Test log aggregator
        from scrollintel.core.log_aggregation import LogAggregator, LogEntry
        from datetime import datetime
        
        aggregator = LogAggregator()
        
        # Test structured logging
        entry = LogEntry(
            timestamp=datetime.utcnow(),
            level="INFO",
            component="test",
            message="Test validation message"
        )
        
        await aggregator.log_structured(entry)
        if len(aggregator.log_buffer) == 0:
            logger.error("✗ Log aggregation failed")
            return False
        
        # Test buffer flush
        await aggregator.flush_buffer()
        logger.info("✓ Log aggregation functionality validated")
        
        # Test health check
        from scrollintel.core.uptime_monitor import UptimeMonitor, HealthCheck
        monitor = UptimeMonitor()
        
        # Create a simple health check
        check = HealthCheck(
            name="validation_test",
            url="https://httpbin.org/status/200",
            timeout=10
        )
        
        try:
            success, response_time, error = await monitor.perform_health_check(check)
            logger.info(f"✓ Health check test: success={success}, time={response_time:.3f}s")
        except Exception as e:
            logger.warning(f"Health check test failed (network issue?): {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Async functionality validation error: {e}")
        return False

def validate_configuration_files() -> bool:
    """Validate monitoring configuration files"""
    logger.info("Validating configuration files...")
    
    try:
        # Validate Prometheus config
        prometheus_config = project_root / "monitoring/prometheus.yml"
        if prometheus_config.exists():
            with open(prometheus_config, 'r') as f:
                content = f.read()
                if 'scrollintel-backend' in content and 'scrape_configs' in content:
                    logger.info("✓ Prometheus configuration validated")
                else:
                    logger.error("✗ Prometheus configuration incomplete")
                    return False
        
        # Validate alert rules
        alert_rules = project_root / "monitoring/alert_rules.yml"
        if alert_rules.exists():
            with open(alert_rules, 'r') as f:
                content = f.read()
                if 'HighCPUUsage' in content and 'groups' in content:
                    logger.info("✓ Alert rules configuration validated")
                else:
                    logger.error("✗ Alert rules configuration incomplete")
                    return False
        
        # Validate Grafana dashboard
        grafana_dashboard = project_root / "monitoring/grafana-dashboard-comprehensive.json"
        if grafana_dashboard.exists():
            with open(grafana_dashboard, 'r') as f:
                dashboard_data = json.load(f)
                if 'dashboard' in dashboard_data and 'panels' in dashboard_data['dashboard']:
                    logger.info("✓ Grafana dashboard configuration validated")
                else:
                    logger.error("✗ Grafana dashboard configuration incomplete")
                    return False
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Configuration validation error: {e}")
        return False

def validate_documentation() -> bool:
    """Validate monitoring documentation"""
    logger.info("Validating documentation...")
    
    try:
        # Check incident response runbooks
        runbooks = project_root / "docs/incident-response-runbooks.md"
        if runbooks.exists():
            with open(runbooks, 'r') as f:
                content = f.read()
                required_sections = [
                    "High CPU Usage",
                    "Database Connection Issues",
                    "High Error Rate",
                    "Emergency Contacts"
                ]
                
                missing_sections = []
                for section in required_sections:
                    if section not in content:
                        missing_sections.append(section)
                
                if missing_sections:
                    logger.error(f"✗ Runbooks missing sections: {missing_sections}")
                    return False
                else:
                    logger.info("✓ Incident response runbooks validated")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Documentation validation error: {e}")
        return False

def generate_validation_report(results: dict) -> None:
    """Generate validation report"""
    logger.info("\n" + "="*60)
    logger.info("SCROLLINTEL MONITORING VALIDATION REPORT")
    logger.info("="*60)
    
    total_checks = len(results)
    passed_checks = sum(1 for result in results.values() if result)
    
    for check_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{status:<8} {check_name}")
    
    logger.info("-"*60)
    logger.info(f"SUMMARY: {passed_checks}/{total_checks} checks passed")
    
    if passed_checks == total_checks:
        logger.info("🎉 ALL MONITORING COMPONENTS VALIDATED SUCCESSFULLY!")
        logger.info("\nNext steps:")
        logger.info("1. Start monitoring system: python scripts/start-monitoring.py")
        logger.info("2. Access Grafana dashboard: http://localhost:3000")
        logger.info("3. View Prometheus metrics: http://localhost:9090")
        logger.info("4. Check status page: http://localhost:8000/status")
    else:
        logger.error("❌ SOME MONITORING COMPONENTS FAILED VALIDATION")
        logger.error("Please fix the failed components before deploying to production")
    
    logger.info("="*60)

async def main():
    """Main validation function"""
    logger.info("Starting ScrollIntel Monitoring System Validation...")
    
    validation_results = {
        "File Structure": validate_monitoring_files(),
        "Module Imports": validate_monitoring_imports(),
        "Basic Functionality": validate_monitoring_functionality(),
        "Async Functionality": await validate_async_functionality(),
        "Configuration Files": validate_configuration_files(),
        "Documentation": validate_documentation()
    }
    
    generate_validation_report(validation_results)
    
    # Return success status
    return all(validation_results.values())

if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("Validation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Validation failed with error: {e}")
        sys.exit(1)