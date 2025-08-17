"""
Test Suite for Immediate Priority Implementation
Tests production infrastructure, user onboarding, and API stability
"""

import pytest
import asyncio
import json
import time
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta

from scrollintel.core.production_infrastructure import (
    ProductionInfrastructure, LoadBalancer, AutoScaler, HealthMonitor, 
    CacheManager, HealthStatus, ServiceMetrics
)
from scrollintel.core.user_onboarding import (
    UserOnboardingSystem, OnboardingStep, SupportTicketPriority,
    User, OnboardingProgress, SupportTicket
)
from scrollintel.core.api_stability import (
    APIStabilitySystem, RateLimiter, RequestValidator, ErrorHandler,
    PerformanceMonitor, CircuitBreaker, RateLimitType
)

class TestProductionInfrastructure:
    """Test production infrastructure components"""
    
    @pytest.fixture
    def infrastructure_config(self):
        return {
            'redis_host': 'localhost',
            'redis_port': 6379,
            'database_url': 'postgresql://test:test@localhost/test',
            'scaling': {
                'min_instances': 2,
                'max_instances': 10,
                'target_cpu': 70.0
            }
        }
    
    @pytest.fixture
    def infrastructure(self, infrastructure_config):
        return ProductionInfrastructure(infrastructure_config)
    
    @pytest.mark.asyncio
    async def test_infrastructure_initialization(self, infrastructure):
        """Test infrastructure initialization"""
        with patch('redis.Redis'), patch('sqlalchemy.create_engine'):
            await infrastructure.initialize()
            assert infrastructure.redis_client is not None
            assert infrastructure.db_pool is not None
    
    @pytest.mark.asyncio
    async def test_system_health_check(self, infrastructure):
        """Test system health monitoring"""
        with patch('redis.Redis'), patch('sqlalchemy.create_engine'):
            await infrastructure.initialize()
            
            health = await infrastructure.get_system_health()
            
            assert 'status' in health
            assert 'metrics' in health
            assert 'load_balancer' in health
            assert 'timestamp' in health
    
    def test_load_balancer_server_management(self):
        """Test load balancer server management"""
        from scrollintel.core.production_infrastructure import LoadBalancerConfig
        
        config = LoadBalancerConfig()
        lb = LoadBalancer(config)
        
        # Add servers
        lb.add_server("http://server1:8000", weight=1)
        lb.add_server("http://server2:8000", weight=2)
        
        assert len(lb.servers) == 2
        assert lb.servers[0]['url'] == "http://server1:8000"
        assert lb.servers[1]['weight'] == 2
    
    @pytest.mark.asyncio
    async def test_load_balancer_health_checks(self):
        """Test load balancer health checking"""
        from scrollintel.core.production_infrastructure import LoadBalancerConfig
        
        config = LoadBalancerConfig()
        lb = LoadBalancer(config)
        lb.add_server("http://server1:8000")
        
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_get.return_value.__aenter__.return_value = mock_response
            
            status = await lb.health_check("http://server1:8000")
            assert status == HealthStatus.HEALTHY
    
    def test_auto_scaler_metrics_collection(self):
        """Test auto-scaler metrics collection"""
        config = {'min_instances': 2, 'max_instances': 10}
        scaler = AutoScaler(config)
        
        # Test scaling decision logic
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
    
    @pytest.mark.asyncio
    async def test_health_monitor_metrics(self):
        """Test health monitoring metrics collection"""
        monitor = HealthMonitor()
        
        with patch('psutil.cpu_percent', return_value=50.0), \
             patch('psutil.virtual_memory') as mock_memory:
            
            mock_memory.return_value.percent = 60.0
            
            metrics = await monitor._collect_health_metrics()
            
            assert metrics.cpu_usage == 50.0
            assert metrics.memory_usage == 60.0
            assert isinstance(metrics.timestamp, float)
    
    def test_cache_manager_status(self):
        """Test cache manager status reporting"""
        cache = CacheManager()
        cache.cache_stats = {
            'hits': 100,
            'misses': 20,
            'evictions': 5,
            'memory_usage': 1024
        }
        
        status = cache.get_status()
        
        assert status['hit_rate'] == 100 / 120  # hits / (hits + misses)
        assert status['total_requests'] == 120
        assert status['memory_usage'] == 1024

class TestUserOnboarding:
    """Test user onboarding system"""
    
    @pytest.fixture
    def onboarding_config(self):
        return {
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
    
    @pytest.fixture
    def onboarding_system(self, onboarding_config):
        return UserOnboardingSystem(onboarding_config)
    
    @pytest.mark.asyncio
    async def test_user_registration(self, onboarding_system):
        """Test user registration process"""
        with patch.object(onboarding_system, '_user_exists', return_value=False), \
             patch.object(onboarding_system, '_save_user'), \
             patch.object(onboarding_system, '_save_onboarding_progress'), \
             patch.object(onboarding_system.email_service, 'send_verification_email'):
            
            result = await onboarding_system.register_user(
                email="test@example.com",
                username="testuser",
                password="SecurePass123"
            )
            
            assert result['success'] is True
            assert 'user_id' in result
            assert 'Registration successful' in result['message']
    
    @pytest.mark.asyncio
    async def test_user_registration_validation(self, onboarding_system):
        """Test user registration validation"""
        # Test invalid email
        result = await onboarding_system.register_user(
            email="invalid-email",
            username="testuser",
            password="SecurePass123"
        )
        assert result['success'] is False
        assert 'Invalid email format' in result['error']
        
        # Test weak password
        result = await onboarding_system.register_user(
            email="test@example.com",
            username="testuser",
            password="weak"
        )
        assert result['success'] is False
        assert 'Password does not meet requirements' in result['error']
    
    @pytest.mark.asyncio
    async def test_email_verification(self, onboarding_system):
        """Test email verification process"""
        user_id = "test-user-id"
        token = onboarding_system._generate_verification_token(user_id)
        
        with patch.object(onboarding_system, '_update_user_verification'), \
             patch.object(onboarding_system, '_advance_onboarding_step'):
            
            result = await onboarding_system.verify_email(token)
            
            assert result['success'] is True
            assert 'Email verified successfully' in result['message']
    
    @pytest.mark.asyncio
    async def test_user_authentication(self, onboarding_system):
        """Test user authentication"""
        user = User(
            id="test-user-id",
            email="test@example.com",
            username="testuser",
            password_hash=onboarding_system.pwd_context.hash("SecurePass123"),
            is_verified=True
        )
        
        with patch.object(onboarding_system, '_get_user_by_email', return_value=user), \
             patch.object(onboarding_system, '_update_last_login'):
            
            result = await onboarding_system.authenticate_user(
                email="test@example.com",
                password="SecurePass123"
            )
            
            assert result['success'] is True
            assert 'access_token' in result
            assert result['user']['email'] == "test@example.com"
    
    @pytest.mark.asyncio
    async def test_onboarding_progress(self, onboarding_system):
        """Test onboarding progress tracking"""
        progress = OnboardingProgress(
            user_id="test-user-id",
            current_step=OnboardingStep.PROFILE_SETUP,
            completed_steps=[OnboardingStep.REGISTRATION, OnboardingStep.EMAIL_VERIFICATION],
            step_data={},
            started_at=datetime.utcnow()
        )
        
        with patch.object(onboarding_system, '_get_onboarding_progress', return_value=progress):
            result = await onboarding_system.get_onboarding_status("test-user-id")
            
            assert result['success'] is True
            assert result['current_step'] == OnboardingStep.PROFILE_SETUP.value
            assert len(result['completed_steps']) == 2
            assert result['progress_percentage'] > 0
    
    @pytest.mark.asyncio
    async def test_support_ticket_creation(self, onboarding_system):
        """Test support ticket creation"""
        with patch.object(onboarding_system.support_system, 'save_ticket'), \
             patch.object(onboarding_system.support_system, 'notify_support_team'):
            
            result = await onboarding_system.create_support_ticket(
                user_id="test-user-id",
                subject="Test Issue",
                description="This is a test support ticket",
                category="technical",
                priority=SupportTicketPriority.MEDIUM
            )
            
            assert result['success'] is True
            assert 'ticket_id' in result
            assert 'Support ticket created successfully' in result['message']
    
    def test_password_validation(self, onboarding_system):
        """Test password validation"""
        # Valid password
        assert onboarding_system._validate_password("SecurePass123") is True
        
        # Invalid passwords
        assert onboarding_system._validate_password("short") is False
        assert onboarding_system._validate_password("nouppercase123") is False
        assert onboarding_system._validate_password("NOLOWERCASE123") is False
        assert onboarding_system._validate_password("NoNumbers") is False
    
    def test_email_validation(self, onboarding_system):
        """Test email validation"""
        # Valid emails
        assert onboarding_system._validate_email("test@example.com") is True
        assert onboarding_system._validate_email("user.name+tag@domain.co.uk") is True
        
        # Invalid emails
        assert onboarding_system._validate_email("invalid-email") is False
        assert onboarding_system._validate_email("@domain.com") is False
        assert onboarding_system._validate_email("user@") is False

class TestAPIStability:
    """Test API stability system"""
    
    @pytest.fixture
    def api_config(self):
        return {
            'redis_host': 'localhost',
            'redis_port': 6379,
            'rate_limiting': {
                'default_limits': {
                    'requests_per_second': 10,
                    'requests_per_minute': 100,
                    'requests_per_hour': 1000,
                    'requests_per_day': 10000
                }
            }
        }
    
    @pytest.fixture
    def api_system(self, api_config):
        return APIStabilitySystem(api_config)
    
    @pytest.mark.asyncio
    async def test_api_system_initialization(self, api_system):
        """Test API stability system initialization"""
        with patch('redis.Redis'):
            await api_system.initialize()
            assert api_system.redis_client is not None
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self):
        """Test rate limiting functionality"""
        config = {
            'default_limits': {
                'requests_per_second': 2,
                'requests_per_minute': 10
            }
        }
        
        rate_limiter = RateLimiter(config)
        
        # Mock Redis client
        mock_redis = Mock()
        mock_redis.pipeline.return_value.execute.return_value = [None, 0, None, None]
        
        await rate_limiter.initialize(mock_redis)
        
        # Test rate limit check
        result = await rate_limiter.check_rate_limit("user123", "/api/test", "GET")
        assert result['allowed'] is True
    
    @pytest.mark.asyncio
    async def test_request_validation(self):
        """Test request validation"""
        validator = RequestValidator()
        
        # Mock request
        mock_request = Mock()
        mock_request.method = "POST"
        mock_request.headers = {'content-type': 'application/json', 'content-length': '1000'}
        mock_request.url.path = "/api/test"
        
        result = await validator.validate_request(mock_request)
        assert result['valid'] is True
        
        # Test suspicious pattern detection
        mock_request.url.path = "/api/../admin"
        result = await validator.validate_request(mock_request)
        assert result['valid'] is False
    
    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling and reporting"""
        error_handler = ErrorHandler()
        
        # Mock request
        mock_request = Mock()
        mock_request.url.path = "/api/test"
        mock_request.method = "GET"
        
        # Test error handling
        test_error = ValueError("Test error")
        response = await error_handler.handle_error(test_error, mock_request, 0.5)
        
        assert response.status_code == 500
        assert len(error_handler.recent_errors) == 1
    
    @pytest.mark.asyncio
    async def test_performance_monitoring(self):
        """Test performance monitoring"""
        monitor = PerformanceMonitor()
        await monitor.initialize()
        
        # Record some metrics
        await monitor.record_request("/api/test", "GET", 0.1, 200)
        await monitor.record_request("/api/test", "GET", 0.2, 200)
        await monitor.record_request("/api/slow", "POST", 1.5, 500)
        
        status = await monitor.get_status()
        
        assert status['total_requests'] == 3
        assert status['error_rate'] > 0
        assert len(status['slowest_endpoints']) > 0
    
    @pytest.mark.asyncio
    async def test_circuit_breaker(self):
        """Test circuit breaker functionality"""
        circuit_breaker = CircuitBreaker()
        
        endpoint = "/api/test"
        
        # Initially should be available
        assert await circuit_breaker.is_available(endpoint) is True
        
        # Record failures to trigger circuit breaker
        for _ in range(6):  # Exceed failure threshold
            await circuit_breaker.record_failure(endpoint)
        
        # Should now be unavailable
        assert await circuit_breaker.is_available(endpoint) is False
        
        # Check status
        status = await circuit_breaker.get_status()
        assert status['open_circuits'] == 1

class TestIntegration:
    """Integration tests for all systems"""
    
    @pytest.mark.asyncio
    async def test_full_system_integration(self):
        """Test integration of all immediate priority systems"""
        # Configuration
        config = {
            'infrastructure': {
                'redis_host': 'localhost',
                'redis_port': 6379,
                'database_url': 'postgresql://test:test@localhost/test'
            },
            'onboarding': {
                'jwt_secret': 'test-secret',
                'email': {
                    'smtp_server': 'localhost',
                    'from_email': 'test@scrollintel.com'
                }
            },
            'api_stability': {
                'redis_host': 'localhost',
                'redis_port': 6379,
                'rate_limiting': {
                    'default_limits': {
                        'requests_per_second': 10
                    }
                }
            }
        }
        
        # Initialize systems
        with patch('redis.Redis'), \
             patch('sqlalchemy.create_engine'), \
             patch('smtplib.SMTP'):
            
            # Infrastructure
            infrastructure = ProductionInfrastructure(config['infrastructure'])
            await infrastructure.initialize()
            
            # Onboarding
            onboarding = UserOnboardingSystem(config['onboarding'])
            
            # API Stability
            api_stability = APIStabilitySystem(config['api_stability'])
            await api_stability.initialize()
            
            # Test system health
            infra_health = await infrastructure.get_system_health()
            api_status = await api_stability.get_system_status()
            
            assert infra_health['status'] is not None
            assert api_status['rate_limiter'] is not None
    
    @pytest.mark.asyncio
    async def test_production_readiness_metrics(self):
        """Test production readiness metrics"""
        # Test uptime target (99.9%)
        uptime_target = 0.999
        
        # Test response time target (<200ms)
        response_time_target = 0.2
        
        # Test error rate target (<1%)
        error_rate_target = 0.01
        
        # Simulate metrics
        monitor = PerformanceMonitor()
        await monitor.initialize()
        
        # Record successful requests
        for i in range(1000):
            response_time = 0.1 + (i % 10) * 0.01  # Vary response times
            status_code = 200 if i < 990 else 500  # 1% error rate
            
            await monitor.record_request(
                f"/api/endpoint{i%5}", "GET", response_time, status_code
            )
        
        status = await monitor.get_status()
        
        # Verify metrics meet targets
        assert status['average_response_time'] < response_time_target
        assert status['error_rate'] <= error_rate_target * 100  # Convert to percentage
        assert status['total_requests'] == 1000

if __name__ == "__main__":
    pytest.main([__file__, "-v"])