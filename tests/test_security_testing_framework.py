"""
Tests for Security Testing and Validation Framework
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, patch

from security.testing import (
    SecurityTestFramework,
    SecurityTestResult,
    SecurityTestType,
    SecuritySeverity,
    VulnerabilityScanner,
    SecurityChaosEngineer,
    SecurityPerformanceTester,
    SecurityRegressionTester,
    SecurityMetricsCollector
)

class TestSecurityTestFramework:
    """Test the main security test framework"""
    
    @pytest.fixture
    def framework(self):
        """Create test framework instance"""
        config = {
            'penetration': {'enabled': True},
            'vulnerability': {'enabled': True},
            'chaos': {'enabled': True},
            'performance': {'enabled': True},
            'regression': {'enabled': True},
            'metrics': {'enabled': True}
        }
        return SecurityTestFramework(config)
    
    @pytest.fixture
    def target_config(self):
        """Create test target configuration"""
        return {
            'base_url': 'http://localhost:8000',
            'host': 'localhost',
            'version': '1.0.0'
        }
    
    @pytest.mark.asyncio
    async def test_framework_initialization(self, framework):
        """Test framework initialization"""
        assert framework is not None
        assert hasattr(framework, 'penetration_tester')
        assert hasattr(framework, 'vulnerability_scanner')
        assert hasattr(framework, 'chaos_engineer')
        assert hasattr(framework, 'performance_tester')
        assert hasattr(framework, 'regression_tester')
        assert hasattr(framework, 'metrics_collector')
    
    @pytest.mark.asyncio
    async def test_comprehensive_security_tests(self, framework, target_config):
        """Test comprehensive security testing"""
        results = await framework.run_comprehensive_security_tests(target_config)
        
        assert isinstance(results, dict)
        assert 'summary' in results
        assert 'test_results' in results
        assert 'recommendations' in results
        
        # Check that all test types are represented
        test_types = set()
        for result_data in results['test_results']:
            test_types.add(result_data['test_type'])
        
        expected_types = {'penetration', 'vulnerability', 'chaos', 'performance', 'regression'}
        assert len(test_types.intersection(expected_types)) > 0
    
    @pytest.mark.asyncio
    async def test_penetration_testing(self, framework, target_config):
        """Test penetration testing component"""
        results = await framework._run_penetration_tests(target_config)
        
        assert isinstance(results, list)
        assert len(results) > 0
        
        for result in results:
            assert isinstance(result, SecurityTestResult)
            assert result.test_type == SecurityTestType.PENETRATION
            assert result.test_id is not None
            assert result.test_name is not None
            assert result.status in ['passed', 'failed', 'error']
    
    @pytest.mark.asyncio
    async def test_vulnerability_scanning(self, framework, target_config):
        """Test vulnerability scanning component"""
        results = await framework._run_vulnerability_scans(target_config)
        
        assert isinstance(results, list)
        assert len(results) > 0
        
        for result in results:
            assert isinstance(result, SecurityTestResult)
            assert result.test_type == SecurityTestType.VULNERABILITY
            assert result.findings is not None
    
    @pytest.mark.asyncio
    async def test_chaos_engineering(self, framework, target_config):
        """Test chaos engineering component"""
        results = await framework._run_chaos_tests(target_config)
        
        assert isinstance(results, list)
        assert len(results) > 0
        
        for result in results:
            assert isinstance(result, SecurityTestResult)
            assert result.test_type == SecurityTestType.CHAOS
            assert 'resilience_score' in result.metadata or len(result.findings) > 0
    
    @pytest.mark.asyncio
    async def test_performance_testing(self, framework, target_config):
        """Test security performance testing component"""
        results = await framework._run_performance_tests(target_config)
        
        assert isinstance(results, list)
        assert len(results) > 0
        
        for result in results:
            assert isinstance(result, SecurityTestResult)
            assert result.test_type == SecurityTestType.PERFORMANCE
    
    @pytest.mark.asyncio
    async def test_regression_testing(self, framework, target_config):
        """Test regression testing component"""
        results = await framework._run_regression_tests(target_config)
        
        assert isinstance(results, list)
        assert len(results) > 0
        
        for result in results:
            assert isinstance(result, SecurityTestResult)
            assert result.test_type == SecurityTestType.REGRESSION

class TestVulnerabilityScanner:
    """Test vulnerability scanner component"""
    
    @pytest.fixture
    def scanner(self):
        """Create vulnerability scanner instance"""
        config = {'enabled': True}
        return VulnerabilityScanner(config)
    
    @pytest.fixture
    def target_config(self):
        """Create test target configuration"""
        return {
            'base_url': 'http://localhost:8000',
            'host': 'localhost'
        }
    
    @pytest.mark.asyncio
    async def test_scanner_initialization(self, scanner):
        """Test scanner initialization"""
        assert scanner is not None
        assert hasattr(scanner, 'scan_modules')
        assert len(scanner.scan_modules) > 0
    
    @pytest.mark.asyncio
    async def test_vulnerability_scan(self, scanner, target_config):
        """Test vulnerability scanning"""
        results = await scanner.scan(target_config)
        
        assert isinstance(results, list)
        assert len(results) > 0
        
        for result in results:
            assert isinstance(result, SecurityTestResult)
            assert result.test_type == SecurityTestType.VULNERABILITY
    
    @pytest.mark.asyncio
    async def test_network_vulnerability_scan(self, scanner, target_config):
        """Test network vulnerability scanning"""
        findings = await scanner._network_vulnerability_scan(target_config)
        
        assert isinstance(findings, list)
        # Findings may be empty if no vulnerabilities found
    
    @pytest.mark.asyncio
    async def test_web_application_scan(self, scanner, target_config):
        """Test web application scanning"""
        findings = await scanner._web_application_scan(target_config)
        
        assert isinstance(findings, list)

class TestSecurityChaosEngineer:
    """Test security chaos engineering component"""
    
    @pytest.fixture
    def chaos_engineer(self):
        """Create chaos engineer instance"""
        config = {'enabled': True}
        return SecurityChaosEngineer(config)
    
    @pytest.fixture
    def target_config(self):
        """Create test target configuration"""
        return {
            'base_url': 'http://localhost:8000',
            'host': 'localhost'
        }
    
    @pytest.mark.asyncio
    async def test_chaos_engineer_initialization(self, chaos_engineer):
        """Test chaos engineer initialization"""
        assert chaos_engineer is not None
        assert hasattr(chaos_engineer, 'attack_simulators')
        assert len(chaos_engineer.attack_simulators) > 0
    
    @pytest.mark.asyncio
    async def test_chaos_tests(self, chaos_engineer, target_config):
        """Test chaos engineering tests"""
        results = await chaos_engineer.run_chaos_tests(target_config)
        
        assert isinstance(results, list)
        assert len(results) > 0
        
        for result in results:
            assert isinstance(result, SecurityTestResult)
            assert result.test_type == SecurityTestType.CHAOS
    
    @pytest.mark.asyncio
    async def test_ddos_simulation(self, chaos_engineer, target_config):
        """Test DDoS attack simulation"""
        from security.testing.chaos_engineering import ChaosExperiment, AttackType
        
        experiment = ChaosExperiment(
            experiment_id="test_ddos",
            name="Test DDoS",
            attack_type=AttackType.DDOS,
            description="Test DDoS simulation",
            duration=5,  # Short duration for testing
            intensity=0.1,  # Low intensity for testing
            target_components=["web_server"],
            success_criteria={"max_response_time": 5000},
            rollback_strategy="Stop attack"
        )
        
        result = await chaos_engineer._simulate_ddos_attack(experiment, target_config)
        
        assert isinstance(result, dict)
        assert 'attack_type' in result
        assert result['attack_type'] == 'ddos'

class TestSecurityPerformanceTester:
    """Test security performance testing component"""
    
    @pytest.fixture
    def performance_tester(self):
        """Create performance tester instance"""
        config = {'enabled': True}
        return SecurityPerformanceTester(config)
    
    @pytest.fixture
    def target_config(self):
        """Create test target configuration"""
        return {
            'base_url': 'http://localhost:8000',
            'host': 'localhost'
        }
    
    @pytest.mark.asyncio
    async def test_performance_tester_initialization(self, performance_tester):
        """Test performance tester initialization"""
        assert performance_tester is not None
        assert hasattr(performance_tester, 'load_profiles')
        assert len(performance_tester.load_profiles) > 0
        assert hasattr(performance_tester, 'security_scenarios')
        assert len(performance_tester.security_scenarios) > 0
    
    @pytest.mark.asyncio
    async def test_security_performance_tests(self, performance_tester, target_config):
        """Test security performance testing"""
        results = await performance_tester.test_security_performance(target_config)
        
        assert isinstance(results, list)
        assert len(results) > 0
        
        for result in results:
            assert isinstance(result, SecurityTestResult)
            assert result.test_type == SecurityTestType.PERFORMANCE

class TestSecurityRegressionTester:
    """Test security regression testing component"""
    
    @pytest.fixture
    def regression_tester(self):
        """Create regression tester instance"""
        config = {'enabled': True}
        return SecurityRegressionTester(config)
    
    @pytest.fixture
    def target_config(self):
        """Create test target configuration"""
        return {
            'base_url': 'http://localhost:8000',
            'host': 'localhost'
        }
    
    @pytest.mark.asyncio
    async def test_regression_tester_initialization(self, regression_tester):
        """Test regression tester initialization"""
        assert regression_tester is not None
        assert hasattr(regression_tester, 'test_cases')
    
    @pytest.mark.asyncio
    async def test_regression_tests(self, regression_tester, target_config):
        """Test regression testing"""
        results = await regression_tester.run_regression_tests(target_config)
        
        assert isinstance(results, list)
        assert len(results) > 0
        
        for result in results:
            assert isinstance(result, SecurityTestResult)
            assert result.test_type == SecurityTestType.REGRESSION

class TestSecurityMetricsCollector:
    """Test security metrics collector component"""
    
    @pytest.fixture
    def metrics_collector(self):
        """Create metrics collector instance"""
        config = {'metrics_db_path': ':memory:'}  # Use in-memory database for testing
        return SecurityMetricsCollector(config)
    
    @pytest.fixture
    def sample_test_results(self):
        """Create sample test results"""
        return [
            SecurityTestResult(
                test_id="test_001",
                test_type=SecurityTestType.PENETRATION,
                test_name="Test Penetration",
                status="passed",
                severity=SecuritySeverity.INFO,
                findings=[],
                execution_time=1.0,
                timestamp=datetime.now(),
                recommendations=["Maintain security"]
            ),
            SecurityTestResult(
                test_id="test_002",
                test_type=SecurityTestType.VULNERABILITY,
                test_name="Test Vulnerability",
                status="failed",
                severity=SecuritySeverity.HIGH,
                findings=[{"severity": "high", "type": "vulnerability"}],
                execution_time=2.0,
                timestamp=datetime.now(),
                recommendations=["Fix vulnerability"]
            )
        ]
    
    def test_metrics_collector_initialization(self, metrics_collector):
        """Test metrics collector initialization"""
        assert metrics_collector is not None
        assert hasattr(metrics_collector, 'db_path')
    
    @pytest.mark.asyncio
    async def test_collect_security_metrics(self, metrics_collector, sample_test_results):
        """Test security metrics collection"""
        target_config = {'base_url': 'http://localhost:8000'}
        
        report = await metrics_collector.collect_security_metrics(sample_test_results, target_config)
        
        assert isinstance(report, dict)
        assert 'summary' in report
        assert 'findings' in report
        assert 'test_results' in report
    
    def test_dashboard_data(self, metrics_collector):
        """Test dashboard data retrieval"""
        dashboard_data = metrics_collector.get_metrics_dashboard_data()
        
        assert isinstance(dashboard_data, dict)
        assert 'recent_metrics' in dashboard_data
        assert 'recent_trends' in dashboard_data

class TestSecurityTestResult:
    """Test SecurityTestResult data class"""
    
    def test_security_test_result_creation(self):
        """Test creating SecurityTestResult"""
        result = SecurityTestResult(
            test_id="test_001",
            test_type=SecurityTestType.PENETRATION,
            test_name="Test Name",
            status="passed",
            severity=SecuritySeverity.INFO,
            findings=[],
            execution_time=1.0,
            timestamp=datetime.now(),
            recommendations=["Test recommendation"]
        )
        
        assert result.test_id == "test_001"
        assert result.test_type == SecurityTestType.PENETRATION
        assert result.test_name == "Test Name"
        assert result.status == "passed"
        assert result.severity == SecuritySeverity.INFO
        assert isinstance(result.findings, list)
        assert result.execution_time == 1.0
        assert isinstance(result.timestamp, datetime)
        assert len(result.recommendations) == 1

# Integration tests
class TestSecurityTestingIntegration:
    """Integration tests for the security testing framework"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_security_testing(self):
        """Test end-to-end security testing workflow"""
        # Initialize framework
        config = {
            'penetration': {'enabled': True},
            'vulnerability': {'enabled': True},
            'chaos': {'enabled': True},
            'performance': {'enabled': True},
            'regression': {'enabled': True},
            'metrics': {'enabled': True, 'metrics_db_path': ':memory:'}
        }
        
        framework = SecurityTestFramework(config)
        
        target_config = {
            'base_url': 'http://localhost:8000',
            'host': 'localhost',
            'version': '1.0.0'
        }
        
        # Run comprehensive tests
        results = await framework.run_comprehensive_security_tests(target_config)
        
        # Validate results structure
        assert isinstance(results, dict)
        assert 'summary' in results
        assert 'test_results' in results
        assert 'recommendations' in results
        assert 'trend_analysis' in results
        
        # Validate summary
        summary = results['summary']
        assert 'total_tests' in summary
        assert 'passed_tests' in summary
        assert 'failed_tests' in summary
        assert 'test_coverage' in summary
        
        # Validate test results
        test_results = results['test_results']
        assert isinstance(test_results, list)
        assert len(test_results) > 0
        
        # Validate that all test types are covered
        test_types = set(result['test_type'] for result in test_results)
        expected_types = {'penetration', 'vulnerability', 'chaos', 'performance', 'regression'}
        assert len(test_types.intersection(expected_types)) > 0
    
    @pytest.mark.asyncio
    async def test_security_metrics_workflow(self):
        """Test security metrics collection workflow"""
        config = {'metrics_db_path': ':memory:'}
        metrics_collector = SecurityMetricsCollector(config)
        
        # Create sample test results
        test_results = [
            SecurityTestResult(
                test_id="test_001",
                test_type=SecurityTestType.PENETRATION,
                test_name="Penetration Test",
                status="passed",
                severity=SecuritySeverity.INFO,
                findings=[],
                execution_time=1.0,
                timestamp=datetime.now(),
                recommendations=[]
            ),
            SecurityTestResult(
                test_id="test_002",
                test_type=SecurityTestType.VULNERABILITY,
                test_name="Vulnerability Scan",
                status="failed",
                severity=SecuritySeverity.HIGH,
                findings=[{"severity": "high", "type": "vulnerability"}],
                execution_time=2.0,
                timestamp=datetime.now(),
                recommendations=["Fix vulnerability"]
            )
        ]
        
        target_config = {'base_url': 'http://localhost:8000'}
        
        # Collect metrics
        report = await metrics_collector.collect_security_metrics(test_results, target_config)
        
        # Validate report
        assert isinstance(report, dict)
        assert 'summary' in report
        assert 'findings' in report
        
        # Test dashboard data
        dashboard_data = metrics_collector.get_metrics_dashboard_data()
        assert isinstance(dashboard_data, dict)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])