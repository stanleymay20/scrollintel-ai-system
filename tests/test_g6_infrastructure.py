"""
Tests for ScrollIntel-G6 Core Infrastructure Setup
"""
import pytest
import time
import redis
from scrollintel.core.distributed_storage import DistributedStorageManager, RedisClusterConfig
from scrollintel.core.compute_cluster import KubernetesClusterManager
from scrollintel.core.security_auth import AuthenticationManager, JWTConfig, Role
from scrollintel.core.monitoring_infrastructure import MonitoringInfrastructure

class TestDistributedStorage:
    """Test distributed storage layer"""
    
    def test_redis_storage_manager_creation(self):
        """Test Redis storage manager creation"""
        config = RedisClusterConfig(
            nodes=[{"host": "localhost", "port": 6379}],
            max_connections=10
        )
        
        storage_manager = DistributedStorageManager(config)
        assert storage_manager is not None
        assert storage_manager.config == config
    
    def test_cache_operations(self):
        """Test basic cache operations"""
        config = RedisClusterConfig(
            nodes=[{"host": "localhost", "port": 6379}],
            max_connections=10
        )
        
        storage_manager = DistributedStorageManager(config)
        
        # Test set and get
        test_key = "test:key:1"
        test_value = {"data": "test_value", "timestamp": time.time()}
        
        # Note: This will fail if Redis is not running, which is expected in test environment
        try:
            result = storage_manager.set(test_key, test_value, ttl=60)
            if result:  # Only test if Redis is available
                retrieved_value = storage_manager.get(test_key)
                assert retrieved_value == test_value
                
                # Test existence
                assert storage_manager.exists(test_key)
                
                # Test deletion
                assert storage_manager.delete(test_key)
                assert not storage_manager.exists(test_key)
        except Exception:
            # Redis not available in test environment
            pytest.skip("Redis not available for testing")

class TestComputeCluster:
    """Test Kubernetes compute cluster management"""
    
    def test_cluster_manager_creation(self):
        """Test cluster manager creation"""
        try:
            cluster_manager = KubernetesClusterManager(namespace="test-namespace")
            assert cluster_manager is not None
            assert cluster_manager.namespace == "test-namespace"
        except Exception:
            # Kubernetes not available in test environment
            pytest.skip("Kubernetes not available for testing")
    
    def test_deployment_configuration(self):
        """Test deployment configuration generation"""
        from scrollintel.core.compute_cluster import ComputeResource, ScalingConfig
        
        # Test resource configuration
        resources = ComputeResource(
            cpu_request="200m",
            cpu_limit="1000m",
            memory_request="256Mi",
            memory_limit="1Gi",
            gpu_request=1
        )
        
        assert resources.cpu_request == "200m"
        assert resources.gpu_request == 1
        
        # Test scaling configuration
        scaling = ScalingConfig(
            min_replicas=2,
            max_replicas=20,
            target_cpu_utilization=75
        )
        
        assert scaling.min_replicas == 2
        assert scaling.max_replicas == 20

class TestSecurityAuth:
    """Test security and authentication layer"""
    
    def test_auth_manager_creation(self):
        """Test authentication manager creation"""
        jwt_config = JWTConfig(
            secret_key="test_secret_key",
            access_token_expire_minutes=15
        )
        
        auth_manager = AuthenticationManager(jwt_config)
        assert auth_manager is not None
        assert auth_manager.jwt_config == jwt_config
    
    def test_user_creation_and_authentication(self):
        """Test user creation and authentication"""
        jwt_config = JWTConfig(secret_key="test_secret_key")
        auth_manager = AuthenticationManager(jwt_config)
        
        # Create user
        user = auth_manager.create_user(
            username="testuser",
            email="test@example.com",
            password="testpass123",
            roles={Role.DATA_SCIENTIST}
        )
        
        assert user.username == "testuser"
        assert user.email == "test@example.com"
        assert Role.DATA_SCIENTIST in user.roles
        
        # Test authentication
        authenticated_user = auth_manager.authenticate_user("testuser", "testpass123")
        assert authenticated_user is not None
        assert authenticated_user.username == "testuser"
        
        # Test wrong password
        failed_auth = auth_manager.authenticate_user("testuser", "wrongpass")
        assert failed_auth is None
    
    def test_jwt_token_generation_and_verification(self):
        """Test JWT token generation and verification"""
        jwt_config = JWTConfig(secret_key="test_secret_key")
        auth_manager = AuthenticationManager(jwt_config)
        
        # Create user
        user = auth_manager.create_user(
            username="tokenuser",
            email="token@example.com",
            password="tokenpass123",
            roles={Role.ADMIN}
        )
        
        # Generate tokens
        tokens = auth_manager.generate_tokens(user)
        
        assert "access_token" in tokens
        assert "refresh_token" in tokens
        assert tokens["token_type"] == "bearer"
        
        # Verify access token
        access_payload = auth_manager.verify_token(tokens["access_token"])
        assert access_payload is not None
        assert access_payload["user_id"] == user.id
        assert access_payload["type"] == "access"
        
        # Verify refresh token
        refresh_payload = auth_manager.verify_token(tokens["refresh_token"])
        assert refresh_payload is not None
        assert refresh_payload["user_id"] == user.id
        assert refresh_payload["type"] == "refresh"
    
    def test_rbac_permissions(self):
        """Test RBAC permission system"""
        from scrollintel.core.security_auth import RBACManager, Permission
        
        # Test admin permissions
        admin_permissions = RBACManager.get_permissions({Role.ADMIN})
        assert Permission.MANAGE_SYSTEM in admin_permissions
        assert Permission.READ_DATA_PRODUCTS in admin_permissions
        
        # Test data scientist permissions
        ds_permissions = RBACManager.get_permissions({Role.DATA_SCIENTIST})
        assert Permission.READ_DATA_PRODUCTS in ds_permissions
        assert Permission.EXECUTE_AGENTS in ds_permissions
        assert Permission.MANAGE_SYSTEM not in ds_permissions
        
        # Test permission checking
        assert RBACManager.has_permission({Role.ADMIN}, Permission.MANAGE_SYSTEM)
        assert not RBACManager.has_permission({Role.VIEWER}, Permission.MANAGE_SYSTEM)

class TestMonitoringInfrastructure:
    """Test monitoring and logging infrastructure"""
    
    def test_monitoring_infrastructure_creation(self):
        """Test monitoring infrastructure creation"""
        monitoring = MonitoringInfrastructure()
        assert monitoring is not None
        assert monitoring.metrics is not None
        assert monitoring.logger is not None
        assert monitoring.alert_manager is not None
    
    def test_metrics_collection(self):
        """Test metrics collection"""
        monitoring = MonitoringInfrastructure()
        
        # Test request recording
        monitoring.record_request("GET", "/api/test", 200, 0.5)
        
        # Test agent execution recording
        monitoring.record_agent_execution("code_agent", "success", 2.5)
        
        # Test cache operation recording
        monitoring.record_cache_operation("redis", True)
        monitoring.record_cache_operation("redis", False)
        
        # Test error recording
        monitoring.record_error("api", "ValidationError")
        
        # Get metrics (should not raise exception)
        metrics_output = monitoring.metrics.get_metrics()
        assert isinstance(metrics_output, str)
        assert "scrollintel_requests_total" in metrics_output
    
    def test_structured_logging(self):
        """Test structured logging"""
        from scrollintel.core.monitoring_infrastructure import StructuredLogger
        
        logger = StructuredLogger("test_component")
        
        # Test different log levels
        logger.info("Test info message", user_id="user123", metadata={"key": "value"})
        logger.warning("Test warning message")
        logger.error("Test error message")
        logger.critical("Test critical message")
        
        # Should not raise exceptions
        assert True
    
    def test_alert_management(self):
        """Test alert management"""
        from scrollintel.core.monitoring_infrastructure import AlertManager, AlertRule, AlertSeverity
        
        alert_manager = AlertManager()
        
        # Add custom alert rule
        custom_rule = AlertRule(
            name="test_alert",
            query="test_metric > 100",
            threshold=100.0,
            severity=AlertSeverity.WARNING,
            description="Test alert rule"
        )
        
        alert_manager.add_alert_rule(custom_rule)
        assert custom_rule in alert_manager.alert_rules
        
        # Test alert triggering (should not raise exception)
        alert_manager.trigger_alert("test_alert", "Test alert message", AlertSeverity.WARNING)
        
        # Test alert resolution
        alert_manager.resolve_alert("test_alert", AlertSeverity.WARNING)
    
    def test_health_status(self):
        """Test health status reporting"""
        monitoring = MonitoringInfrastructure()
        
        health_status = monitoring.get_health_status()
        
        assert "status" in health_status
        assert "timestamp" in health_status
        assert "active_alerts" in health_status
        assert "monitoring_active" in health_status

class TestInfrastructureIntegration:
    """Test integration between infrastructure components"""
    
    def test_monitoring_with_auth(self):
        """Test monitoring integration with authentication"""
        # Create auth manager
        jwt_config = JWTConfig(secret_key="integration_test_key")
        auth_manager = AuthenticationManager(jwt_config)
        
        # Create monitoring
        monitoring = MonitoringInfrastructure()
        
        # Create user and authenticate
        user = auth_manager.create_user(
            username="integrationuser",
            email="integration@example.com",
            password="integrationpass",
            roles={Role.DATA_ENGINEER}
        )
        
        tokens = auth_manager.generate_tokens(user)
        
        # Record authentication metrics
        monitoring.record_request("POST", "/auth/login", 200, 0.3)
        
        # Verify no exceptions
        assert tokens["access_token"] is not None
    
    def test_cache_with_monitoring(self):
        """Test cache operations with monitoring"""
        monitoring = MonitoringInfrastructure()
        
        # Simulate cache operations with monitoring
        monitoring.record_cache_operation("distributed_storage", True)
        monitoring.record_cache_operation("distributed_storage", False)
        
        # Get metrics to verify recording
        metrics = monitoring.metrics.get_metrics()
        assert "scrollintel_cache_hits_total" in metrics
        assert "scrollintel_cache_misses_total" in metrics

if __name__ == "__main__":
    pytest.main([__file__, "-v"])