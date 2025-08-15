"""
Integration tests for the complete Autonomous Innovation Lab system
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

from scrollintel.core.autonomous_innovation_lab_deployment import AutonomousInnovationLabDeployment
from scrollintel.core.autonomous_innovation_lab import LabConfiguration

class TestAutonomousInnovationLabIntegration:
    """Integration tests for the complete autonomous innovation lab system"""
    
    @pytest.fixture
    async def mock_dependencies(self):
        """Mock all external dependencies"""
        with patch.multiple(
            'scrollintel.core.autonomous_innovation_lab_deployment',
            BreakthroughInnovationIntegration=Mock(),
            QuantumAIIntegration=Mock(),
            Orchestrator=Mock(),
            SystemMonitor=Mock()
        ) as mocks:
            
            # Configure mocks
            mocks['BreakthroughInnovationIntegration'].return_value.start = AsyncMock()
            mocks['BreakthroughInnovationIntegration'].return_value.validate = AsyncMock(return_value={"validated": True})
            
            mocks['QuantumAIIntegration'].return_value.start = AsyncMock()
            mocks['QuantumAIIntegration'].return_value.validate = AsyncMock(return_value={"validated": True})
            
            orchestrator_mock = mocks['Orchestrator'].return_value
            orchestrator_mock.initialize = AsyncMock()
            orchestrator_mock.get_status = AsyncMock(return_value={"status": "active"})
            orchestrator_mock.shutdown = AsyncMock()
            
            monitor_mock = mocks['SystemMonitor'].return_value
            monitor_mock.initialize = AsyncMock()
            monitor_mock.start_monitoring = AsyncMock()
            monitor_mock.get_status = AsyncMock(return_value={"status": "active"})
            monitor_mock.stop_monitoring = AsyncMock()
            
            yield mocks
    
    @pytest.fixture
    async def deployment_system(self, mock_dependencies):
        """Create deployment system with mocked dependencies"""
        with patch('scrollintel.core.autonomous_innovation_lab_deployment.AutonomousInnovationLab') as lab_mock:
            # Configure lab mock
            lab_instance = Mock()
            lab_instance._validate_configuration = Mock(return_value=True)
            lab_instance.start_lab = AsyncMock(return_value=True)
            lab_instance.stop_lab = AsyncMock()
            lab_instance.get_lab_status = AsyncMock(return_value={
                "status": "active",
                "is_running": True,
                "active_projects": 5,
                "metrics": {"success_rate": 0.85, "total_innovations": 10},
                "research_domains": ["ai", "quantum", "biotech"]
            })
            lab_instance.validate_lab_capability = AsyncMock(return_value={
                "overall_success": True,
                "domain_results": {
                    "ai": {"success": True, "capabilities": {
                        "topic_generation": True,
                        "experiment_planning": True,
                        "prototype_generation": True,
                        "innovation_validation": True,
                        "knowledge_synthesis": True
                    }},
                    "quantum": {"success": True, "capabilities": {
                        "topic_generation": True,
                        "experiment_planning": True,
                        "prototype_generation": True,
                        "innovation_validation": True,
                        "knowledge_synthesis": True
                    }}
                }
            })
            
            lab_mock.return_value = lab_instance
            
            deployment = AutonomousInnovationLabDeployment()
            yield deployment
    
    @pytest.mark.asyncio
    async def test_complete_system_deployment(self, deployment_system):
        """Test complete system deployment process"""
        # Deploy the system
        result = await deployment_system.deploy_complete_system()
        
        assert result["success"] is True
        assert "deployment_id" in result
        assert "lab_status" in result
        assert "integrations" in result
        assert "validation_results" in result
        assert deployment_system.deployment_status == "deployed"
    
    @pytest.mark.asyncio
    async def test_deployment_with_custom_config(self, deployment_system):
        """Test deployment with custom configuration"""
        custom_config = {
            "max_concurrent_projects": 20,
            "research_domains": ["ai", "quantum", "biotech", "nanotech"],
            "quality_threshold": 0.9
        }
        
        result = await deployment_system.deploy_complete_system(custom_config)
        
        assert result["success"] is True
        assert deployment_system.lab is not None
    
    @pytest.mark.asyncio
    async def test_deployment_readiness_validation(self, deployment_system):
        """Test deployment readiness validation"""
        # Initialize components first
        deployment_system.lab = Mock()
        deployment_system.lab._validate_configuration = Mock(return_value=True)
        deployment_system.integrations = {"test": Mock()}
        deployment_system.orchestrator = Mock()
        deployment_system.monitor = Mock()
        
        # Mock resource check
        with patch.object(deployment_system, '_check_system_resources', 
                         return_value={"sufficient": True, "issues": []}):
            
            validation = await deployment_system._validate_deployment_readiness()
            
            assert validation["ready"] is True
            assert len(validation["issues"]) == 0
    
    @pytest.mark.asyncio
    async def test_deployment_readiness_validation_failure(self, deployment_system):
        """Test deployment readiness validation with failures"""
        # Don't initialize components to trigger validation failure
        validation = await deployment_system._validate_deployment_readiness()
        
        assert validation["ready"] is False
        assert len(validation["issues"]) > 0
    
    @pytest.mark.asyncio
    async def test_system_integrations_initialization(self, deployment_system):
        """Test system integrations initialization"""
        await deployment_system._initialize_integrations()
        
        assert "breakthrough_innovation" in deployment_system.integrations
        assert "quantum_ai" in deployment_system.integrations
        assert len(deployment_system.integrations) >= 2
    
    @pytest.mark.asyncio
    async def test_orchestration_initialization(self, deployment_system):
        """Test orchestration and monitoring initialization"""
        await deployment_system._initialize_orchestration()
        
        assert deployment_system.orchestrator is not None
        assert deployment_system.monitor is not None
    
    @pytest.mark.asyncio
    async def test_post_deployment_validation(self, deployment_system):
        """Test post-deployment validation"""
        # Set up deployment system
        await deployment_system.deploy_complete_system()
        
        # Run post-deployment validation
        validation = await deployment_system._post_deployment_validation()
        
        assert "lab_capability" in validation
        assert "integrations" in validation
        assert "orchestration" in validation
        assert "monitoring" in validation
        assert validation["overall_success"] is True
    
    @pytest.mark.asyncio
    async def test_complete_system_validation(self, deployment_system):
        """Test complete system validation"""
        # Deploy system first
        await deployment_system.deploy_complete_system()
        
        # Validate complete system
        validation = await deployment_system.validate_complete_system()
        
        assert validation["success"] is True
        assert validation["overall_system_valid"] is True
        assert "validation_results" in validation
        
        # Check validation components
        results = validation["validation_results"]
        assert "lab_capabilities" in results
        assert "integrations" in results
        assert "performance" in results
        assert "resilience" in results
        assert "innovation_pipeline" in results
    
    @pytest.mark.asyncio
    async def test_system_validation_not_deployed(self, deployment_system):
        """Test system validation when not deployed"""
        validation = await deployment_system.validate_complete_system()
        
        assert validation["success"] is False
        assert "System not deployed" in validation["error"]
    
    @pytest.mark.asyncio
    async def test_integration_validation(self, deployment_system):
        """Test integration validation"""
        # Set up integrations
        await deployment_system._initialize_integrations()
        
        validation = await deployment_system._validate_all_integrations()
        
        assert validation["all_integrations_valid"] is True
        assert "integration_results" in validation
        assert validation["total_integrations"] >= 2
    
    @pytest.mark.asyncio
    async def test_performance_validation(self, deployment_system):
        """Test system performance validation"""
        # Deploy system
        await deployment_system.deploy_complete_system()
        
        validation = await deployment_system._validate_system_performance()
        
        assert "performance_acceptable" in validation
        assert "metrics" in validation
        assert "thresholds" in validation
    
    @pytest.mark.asyncio
    async def test_resilience_validation(self, deployment_system):
        """Test system resilience validation"""
        validation = await deployment_system._validate_system_resilience()
        
        assert validation["resilience_validated"] is True
        assert "test_results" in validation
        assert validation["total_tests"] > 0
    
    @pytest.mark.asyncio
    async def test_innovation_pipeline_validation(self, deployment_system):
        """Test innovation pipeline validation"""
        validation = await deployment_system._validate_innovation_pipeline()
        
        assert validation["pipeline_operational"] is True
        assert "stage_results" in validation
        assert validation["total_stages"] > 0
    
    @pytest.mark.asyncio
    async def test_deployment_status_reporting(self, deployment_system):
        """Test deployment status reporting"""
        # Before deployment
        status = await deployment_system.get_deployment_status()
        assert status["deployment_status"] == "not_deployed"
        
        # After deployment
        await deployment_system.deploy_complete_system()
        status = await deployment_system.get_deployment_status()
        
        assert status["deployment_status"] == "deployed"
        assert "lab_status" in status
        assert "integrations" in status
    
    @pytest.mark.asyncio
    async def test_system_shutdown(self, deployment_system):
        """Test graceful system shutdown"""
        # Deploy system first
        await deployment_system.deploy_complete_system()
        
        # Shutdown system
        result = await deployment_system.shutdown_system()
        
        assert result["success"] is True
        assert deployment_system.deployment_status == "shutdown"
    
    @pytest.mark.asyncio
    async def test_lab_configuration_creation(self, deployment_system):
        """Test lab configuration creation"""
        # Test default configuration
        config = await deployment_system._create_lab_configuration()
        
        assert config.max_concurrent_projects == 15
        assert len(config.research_domains) >= 10
        assert config.quality_threshold == 0.8
        assert config.continuous_learning is True
        
        # Test custom configuration
        custom_config = {
            "max_concurrent_projects": 25,
            "quality_threshold": 0.9
        }
        
        config = await deployment_system._create_lab_configuration(custom_config)
        
        assert config.max_concurrent_projects == 25
        assert config.quality_threshold == 0.9
    
    @pytest.mark.asyncio
    async def test_system_resource_check(self, deployment_system):
        """Test system resource checking"""
        resource_check = await deployment_system._check_system_resources()
        
        assert "sufficient" in resource_check
        assert "issues" in resource_check
        # In this mock implementation, resources should be sufficient
        assert resource_check["sufficient"] is True
    
    @pytest.mark.asyncio
    async def test_integration_startup(self, deployment_system):
        """Test integration startup process"""
        # Initialize integrations first
        await deployment_system._initialize_integrations()
        
        # Start integrations
        await deployment_system._start_integrations()
        
        # Verify integrations were started
        for integration in deployment_system.integrations.values():
            if hasattr(integration, 'start'):
                integration.start.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_monitoring_startup(self, deployment_system):
        """Test monitoring system startup"""
        # Initialize monitoring
        await deployment_system._initialize_orchestration()
        
        # Start monitoring
        await deployment_system._start_monitoring()
        
        # Verify monitoring was started
        if deployment_system.monitor:
            deployment_system.monitor.start_monitoring.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_end_to_end_deployment_lifecycle(self, deployment_system):
        """Test complete deployment lifecycle"""
        # 1. Deploy system
        deploy_result = await deployment_system.deploy_complete_system()
        assert deploy_result["success"] is True
        
        # 2. Validate system
        validation_result = await deployment_system.validate_complete_system()
        assert validation_result["success"] is True
        assert validation_result["overall_system_valid"] is True
        
        # 3. Get status
        status = await deployment_system.get_deployment_status()
        assert status["deployment_status"] == "deployed"
        
        # 4. Shutdown system
        shutdown_result = await deployment_system.shutdown_system()
        assert shutdown_result["success"] is True
        assert deployment_system.deployment_status == "shutdown"
    
    @pytest.mark.asyncio
    async def test_deployment_failure_handling(self, deployment_system):
        """Test deployment failure handling"""
        # Mock lab startup failure
        with patch.object(deployment_system, '_initialize_integrations', 
                         side_effect=Exception("Integration failure")):
            
            result = await deployment_system.deploy_complete_system()
            
            assert result["success"] is False
            assert "Integration failure" in result["error"]
            assert deployment_system.deployment_status == "failed"
    
    @pytest.mark.asyncio
    async def test_validation_error_handling(self, deployment_system):
        """Test validation error handling"""
        # Test validation when system is not properly initialized
        deployment_system.lab = None
        
        validation = await deployment_system.validate_complete_system()
        
        assert validation["success"] is False
        assert "System not deployed" in validation["error"]

class TestLabConfigurationIntegration:
    """Test lab configuration in integration context"""
    
    def test_production_configuration(self):
        """Test production-ready lab configuration"""
        config = LabConfiguration(
            max_concurrent_projects=50,
            research_domains=[
                "artificial_intelligence", "quantum_computing", "biotechnology",
                "nanotechnology", "renewable_energy", "space_technology",
                "materials_science", "robotics", "cybersecurity", "blockchain",
                "augmented_reality", "autonomous_systems"
            ],
            quality_threshold=0.85,
            innovation_targets={
                "breakthrough_innovations": 100,
                "validated_prototypes": 500,
                "research_publications": 1000,
                "patent_applications": 200
            },
            continuous_learning=True
        )
        
        assert config.max_concurrent_projects == 50
        assert len(config.research_domains) == 12
        assert config.quality_threshold == 0.85
        assert config.innovation_targets["breakthrough_innovations"] == 100
        assert config.continuous_learning is True

class TestSystemIntegrationScenarios:
    """Test various system integration scenarios"""
    
    @pytest.mark.asyncio
    async def test_high_load_deployment(self, deployment_system):
        """Test deployment under high load conditions"""
        high_load_config = {
            "max_concurrent_projects": 100,
            "research_domains": [f"domain_{i}" for i in range(20)],
            "quality_threshold": 0.95
        }
        
        result = await deployment_system.deploy_complete_system(high_load_config)
        
        assert result["success"] is True
        # System should handle high load configuration
    
    @pytest.mark.asyncio
    async def test_minimal_deployment(self, deployment_system):
        """Test minimal system deployment"""
        minimal_config = {
            "max_concurrent_projects": 1,
            "research_domains": ["ai"],
            "quality_threshold": 0.5
        }
        
        result = await deployment_system.deploy_complete_system(minimal_config)
        
        assert result["success"] is True
        # System should work with minimal configuration
    
    @pytest.mark.asyncio
    async def test_multi_domain_validation(self, deployment_system):
        """Test validation across multiple research domains"""
        await deployment_system.deploy_complete_system()
        
        validation = await deployment_system.validate_complete_system()
        
        # Should validate capabilities across all configured domains
        lab_capabilities = validation["validation_results"]["lab_capabilities"]
        assert lab_capabilities["overall_success"] is True
        assert len(lab_capabilities["domain_results"]) >= 2

if __name__ == "__main__":
    pytest.main([__file__])