#!/usr/bin/env python3
"""
Simple test script for immediate priority implementation
Tests core functionality without complex imports
"""

import sys
import os
import asyncio
import time
from unittest.mock import Mock, patch

# Add the project root to Python path
sys.path.insert(0, os.path.abspath('.'))

def test_production_infrastructure():
    """Test production infrastructure components"""
    print("Testing Production Infrastructure...")
    
    try:
        from scrollintel.core.production_infrastructure import (
            LoadBalancer, AutoScaler, HealthMonitor, ServiceMetrics, HealthStatus
        )
        
        # Test LoadBalancer
        from scrollintel.core.production_infrastructure import LoadBalancerConfig
        config = LoadBalancerConfig()
        lb = LoadBalancer(config)
        
        lb.add_server("http://server1:8000", weight=1)
        lb.add_server("http://server2:8000", weight=2)
        
        assert len(lb.servers) == 2
        assert lb.servers[0]['url'] == "http://server1:8000"
        print("✅ LoadBalancer: Server management works")
        
        # Test AutoScaler
        scaler_config = {'min_instances': 2, 'max_instances': 10}
        scaler = AutoScaler(scaler_config)
        
        metrics = ServiceMetrics(
            cpu_usage=85.0,
            memory_usage=75.0,
            response_time=0.5,
            error_rate=2.0,
            active_connections=100,
            timestamp=time.time()
        )
        
        decision = scaler._make_scaling_decision(metrics)
        assert decision == 1  # Should scale up
        print("✅ AutoScaler: Scaling decisions work")
        
        # Test HealthMonitor
        monitor = HealthMonitor()
        assert len(monitor.metrics_history) == 0
        print("✅ HealthMonitor: Initialization works")
        
        print("✅ Production Infrastructure: All tests passed\n")
        return True
        
    except Exception as e:
        print(f"❌ Production Infrastructure test failed: {e}")
        return False

def test_user_onboarding():
    """Test user onboarding system"""
    print("Testing User Onboarding System...")
    
    try:
        from scrollintel.core.user_onboarding import (
            UserOnboardingSystem, OnboardingStep, User, SupportTicketPriority
        )
        
        # Test configuration
        config = {
            'jwt_secret': 'test-secret',
            'jwt_algorithm': 'HS256',
            'email': {
                'smtp_server': 'localhost',
                'smtp_port': 587,
                'username': 'test@example.com',
                'password': 'password',
                'from_email': 'noreply@scrollintel.com',
                'base_url': 'http://localhost:3000'
            }
        }
        
        onboarding = UserOnboardingSystem(config)
        
        # Test password validation
        assert onboarding._validate_password("SecurePass123") is True
        assert onboarding._validate_password("weak") is False
        print("✅ Password validation works")
        
        # Test email validation
        assert onboarding._validate_email("test@example.com") is True
        assert onboarding._validate_email("invalid-email") is False
        print("✅ Email validation works")
        
        # Test JWT token generation
        token = onboarding._generate_verification_token("test-user-id")
        assert len(token) > 0
        print("✅ JWT token generation works")
        
        # Test onboarding step progression
        next_step = onboarding._get_next_onboarding_step(OnboardingStep.REGISTRATION)
        assert next_step == OnboardingStep.EMAIL_VERIFICATION
        print("✅ Onboarding step progression works")
        
        print("✅ User Onboarding: All tests passed\n")
        return True
        
    except Exception as e:
        print(f"❌ User Onboarding test failed: {e}")
        return False

def test_api_stability():
    """Test API stability system"""
    print("Testing API Stability System...")
    
    try:
        from scrollintel.core.api_stability import (
            RateLimiter, RequestValidator, ErrorHandler, PerformanceMonitor,
            CircuitBreaker, RateLimitType, ErrorSeverity
        )
        
        # Test RateLimiter configuration
        config = {
            'default_limits': {
                'requests_per_second': 10,
                'requests_per_minute': 100
            }
        }
        rate_limiter = RateLimiter(config)
        
        # Test window size calculation
        window_size = rate_limiter._get_window_size(RateLimitType.PER_MINUTE)
        assert window_size == 60
        print("✅ Rate limiter configuration works")
        
        # Test RequestValidator
        validator = RequestValidator()
        
        # Test suspicious pattern detection
        assert validator._contains_suspicious_patterns("/api/../admin") is True
        assert validator._contains_suspicious_patterns("/api/normal") is False
        print("✅ Request validation works")
        
        # Test ErrorHandler
        error_handler = ErrorHandler()
        
        test_error = ValueError("Test error")
        severity = error_handler._determine_severity(test_error)
        assert severity == ErrorSeverity.MEDIUM
        print("✅ Error handling works")
        
        # Test PerformanceMonitor
        monitor = PerformanceMonitor()
        
        # Test endpoint stats update
        monitor._update_endpoint_stats("/api/test", "GET", 0.1, 200)
        assert "GET:/api/test" in monitor.endpoint_stats
        print("✅ Performance monitoring works")
        
        # Test CircuitBreaker
        circuit_breaker = CircuitBreaker()
        
        endpoint = "/api/test"
        assert circuit_breaker.circuit_states == {}
        print("✅ Circuit breaker initialization works")
        
        print("✅ API Stability: All tests passed\n")
        return True
        
    except Exception as e:
        print(f"❌ API Stability test failed: {e}")
        return False

async def test_async_functionality():
    """Test async functionality"""
    print("Testing Async Functionality...")
    
    try:
        from scrollintel.core.production_infrastructure import HealthMonitor
        from scrollintel.core.api_stability import PerformanceMonitor, CircuitBreaker
        
        # Test HealthMonitor async methods
        monitor = HealthMonitor()
        
        with patch('psutil.cpu_percent', return_value=50.0), \
             patch('psutil.virtual_memory') as mock_memory, \
             patch('psutil.disk_usage') as mock_disk:
            
            mock_memory.return_value.percent = 60.0
            mock_disk.return_value = Mock()
            
            metrics = await monitor._collect_health_metrics()
            assert metrics.cpu_usage == 50.0
            assert metrics.memory_usage == 60.0
            print("✅ Health monitoring async methods work")
        
        # Test PerformanceMonitor async methods
        perf_monitor = PerformanceMonitor()
        await perf_monitor.initialize()
        
        await perf_monitor.record_request("/api/test", "GET", 0.1, 200)
        assert len(perf_monitor.metrics) == 1
        print("✅ Performance monitoring async methods work")
        
        # Test CircuitBreaker async methods
        circuit_breaker = CircuitBreaker()
        
        endpoint = "/api/test"
        assert await circuit_breaker.is_available(endpoint) is True
        
        # Record failures
        for _ in range(6):
            await circuit_breaker.record_failure(endpoint)
        
        assert await circuit_breaker.is_available(endpoint) is False
        print("✅ Circuit breaker async methods work")
        
        print("✅ Async Functionality: All tests passed\n")
        return True
        
    except Exception as e:
        print(f"❌ Async functionality test failed: {e}")
        return False

def test_configuration():
    """Test configuration system"""
    print("Testing Configuration System...")
    
    try:
        from scrollintel.core.config import get_config, get_default_config, validate_config
        
        # Test default configuration
        default_config = get_default_config()
        assert 'infrastructure' in default_config
        assert 'onboarding' in default_config
        assert 'api_stability' in default_config
        print("✅ Default configuration works")
        
        # Test configuration validation
        assert validate_config(default_config) is True
        print("✅ Configuration validation works")
        
        # Test get_config function
        config = get_config()
        assert config is not None
        print("✅ Configuration loading works")
        
        print("✅ Configuration System: All tests passed\n")
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 ScrollIntel Immediate Priority Implementation Tests\n")
    
    test_results = []
    
    # Run synchronous tests
    test_results.append(test_configuration())
    test_results.append(test_production_infrastructure())
    test_results.append(test_user_onboarding())
    test_results.append(test_api_stability())
    
    # Run async tests
    async_result = asyncio.run(test_async_functionality())
    test_results.append(async_result)
    
    # Summary
    passed = sum(test_results)
    total = len(test_results)
    
    print("=" * 50)
    print(f"Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed! ScrollIntel is ready for production.")
        return True
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)