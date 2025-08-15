"""
Tests for Cultural Transformation Leadership System Deployment
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock

from scrollintel.core.cultural_transformation_deployment import (
    CulturalTransformationDeployment,
    OrganizationType,
    TransformationValidationResult
)

class TestCulturalTransformationDeployment:
    """Test cultural transformation deployment system"""
    
    @pytest.fixture
    def deployment_system(self):
        """Create deployment system instance"""
        return CulturalTransformationDeployment()
    
    @pytest.fixture
    def sample_organization_type(self):
        """Create sample organization type"""
        return OrganizationType(
            name="Tech Startup",
            size="startup",
            industry="technology",
            culture_maturity="emerging",
            complexity="medium"
        )
    
    @pytest.mark.asyncio
    async def test_deploy_complete_system(self, deployment_system):
        """Test complete system deployment"""
        # Mock component initialization
        with patch.object(deployment_system, '_initialize_components', new_callable=AsyncMock) as mock_init, \
             patch.object(deployment_system, '_integrate_components', new_callable=AsyncMock) as mock_integrate, \
             patch.object(deployment_system, '_validate_system_health', new_callable=AsyncMock) as mock_health, \
             patch.object(deployment_system, '_deploy_production_system', new_callable=AsyncMock) as mock_deploy:
            
            mock_health.return_value = {"overall_health": "healthy"}
            mock_deploy.return_value = {"deployment_status": "success"}
            
            result = await deployment_system.deploy_complete_system()
            
            assert result["deployment_status"] == "success"
            assert "system_health" in result
            assert "deployment_result" in result
            assert result["components_integrated"] > 0
            assert deployment_system.system_status == "deployed"
            
            mock_init.assert_called_once()
            mock_integrate.assert_called_once()
            mock_health.assert_called_once()
            mock_deploy.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_validate_across_organization_types(self, deployment_system):
        """Test validation across organization types"""
        # Mock validation for individual organization types
        with patch.object(deployment_system, '_validate_organization_type', new_callable=AsyncMock) as mock_validate:
            mock_validate.return_value = TransformationValidationResult(
                organization_type=OrganizationType("Test Org", "medium", "technology", "mature", "medium"),
                assessment_accuracy=0.9,
                transformation_effectiveness=0.85,
                behavioral_change_success=0.8,
                engagement_improvement=0.82,
                sustainability_score=0.78,
                overall_success=0.83,
                validation_timestamp=datetime.now(),
                detailed_metrics={"test": "metrics"}
            )
            
            results = await deployment_system.validate_across_organization_types()
            
            assert len(results) > 0
            assert all(isinstance(result, TransformationValidationResult) for result in results)
            assert all(result.overall_success > 0 for result in results)
            assert deployment_system.validation_results == results
    
    @pytest.mark.asyncio
    async def test_create_continuous_learning_system(self, deployment_system):
        """Test continuous learning system creation"""
        # Mock learning system setup
        with patch.object(deployment_system, '_setup_feedback_collection', new_callable=AsyncMock) as mock_feedback, \
             patch.object(deployment_system, '_setup_performance_monitoring', new_callable=AsyncMock) as mock_monitoring, \
             patch.object(deployment_system, '_setup_adaptation_engine', new_callable=AsyncMock) as mock_adaptation, \
             patch.object(deployment_system, '_setup_knowledge_base', new_callable=AsyncMock) as mock_knowledge, \
             patch.object(deployment_system, '_setup_improvement_pipeline', new_callable=AsyncMock) as mock_improvement, \
             patch.object(deployment_system, '_integrate_learning_system', new_callable=AsyncMock) as mock_integrate:
            
            mock_feedback.return_value = {"feedback_channels": ["surveys", "metrics"]}
            mock_monitoring.return_value = {"monitoring_metrics": ["success", "engagement"]}
            mock_adaptation.return_value = {"adaptation_algorithms": ["rl", "genetic"]}
            mock_knowledge.return_value = {"knowledge_sources": ["outcomes", "practices"]}
            mock_improvement.return_value = {"improvement_identification": "automated"}
            
            result = await deployment_system.create_continuous_learning_system()
            
            assert result["learning_system_status"] == "active"
            assert len(result["components"]) == 5
            assert "learning_capabilities" in result
            assert len(result["learning_capabilities"]) > 0
            
            mock_feedback.assert_called_once()
            mock_monitoring.assert_called_once()
            mock_adaptation.assert_called_once()
            mock_knowledge.assert_called_once()
            mock_improvement.assert_called_once()
            mock_integrate.assert_called_once()
    
    def test_get_all_components(self, deployment_system):
        """Test getting all components"""
        components = deployment_system._get_all_components()
        
        assert isinstance(components, dict)
        assert len(components) > 15  # Should have all major components
        assert "assessment_engine" in components
        assert "vision_engine" in components
        assert "behavioral_analysis" in components
        assert "messaging_engine" in components
        assert "progress_tracking" in components
    
    def test_get_organization_types(self, deployment_system):
        """Test getting organization types for validation"""
        org_types = deployment_system._get_organization_types()
        
        assert isinstance(org_types, list)
        assert len(org_types) >= 10  # Should have comprehensive coverage
        assert all(isinstance(org_type, OrganizationType) for org_type in org_types)
        
        # Check for variety in organization characteristics
        sizes = {org_type.size for org_type in org_types}
        industries = {org_type.industry for org_type in org_types}
        maturities = {org_type.culture_maturity for org_type in org_types}
        
        assert len(sizes) >= 4  # Multiple sizes
        assert len(industries) >= 5  # Multiple industries
        assert len(maturities) >= 3  # Multiple maturity levels
    
    @pytest.mark.asyncio
    async def test_validate_organization_type(self, deployment_system, sample_organization_type):
        """Test validation for specific organization type"""
        # Mock simulation and metrics calculation
        with patch.object(deployment_system, '_create_test_organization') as mock_create, \
             patch.object(deployment_system, '_run_transformation_simulation', new_callable=AsyncMock) as mock_simulate, \
             patch.object(deployment_system, '_calculate_validation_metrics') as mock_calculate:
            
            mock_create.return_value = {"type": sample_organization_type}
            mock_simulate.return_value = {"simulation": "results"}
            mock_calculate.return_value = {
                "assessment_accuracy": 0.9,
                "transformation_effectiveness": 0.85,
                "behavioral_change_success": 0.8,
                "engagement_improvement": 0.82,
                "sustainability_score": 0.78,
                "overall_success": 0.83
            }
            
            result = await deployment_system._validate_organization_type(sample_organization_type)
            
            assert isinstance(result, TransformationValidationResult)
            assert result.organization_type == sample_organization_type
            assert result.overall_success > 0.8
            assert result.assessment_accuracy > 0.8
            assert result.transformation_effectiveness > 0.8
            
            mock_create.assert_called_once_with(sample_organization_type)
            mock_simulate.assert_called_once()
            mock_calculate.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_run_transformation_simulation(self, deployment_system):
        """Test transformation simulation"""
        test_organization = {
            "type": OrganizationType("Test", "medium", "tech", "mature", "medium"),
            "employee_count": 500,
            "departments": ["Engineering", "Sales"],
            "current_culture": {"values": ["innovation"]},
            "challenges": ["scaling"]
        }
        
        with patch.object(deployment_system, '_simulate_transformation_step', new_callable=AsyncMock) as mock_step:
            mock_step.return_value = {
                "step": "test_step",
                "success": True,
                "effectiveness": 0.85,
                "duration": 30,
                "resources_used": 0.3,
                "stakeholder_satisfaction": 0.8
            }
            
            result = await deployment_system._run_transformation_simulation(test_organization)
            
            assert isinstance(result, dict)
            assert len(result) == 10  # Should have all transformation steps
            assert all("effectiveness" in step_result for step_result in result.values())
            assert all(step_result["success"] for step_result in result.values())
    
    def test_calculate_validation_metrics(self, deployment_system, sample_organization_type):
        """Test validation metrics calculation"""
        simulation_results = {
            "step1": {"effectiveness": 0.9, "stakeholder_satisfaction": 0.85},
            "step2": {"effectiveness": 0.8, "stakeholder_satisfaction": 0.8},
            "step3": {"effectiveness": 0.85, "stakeholder_satisfaction": 0.82}
        }
        
        metrics = deployment_system._calculate_validation_metrics(simulation_results, sample_organization_type)
        
        assert isinstance(metrics, dict)
        assert "assessment_accuracy" in metrics
        assert "transformation_effectiveness" in metrics
        assert "behavioral_change_success" in metrics
        assert "engagement_improvement" in metrics
        assert "sustainability_score" in metrics
        assert "overall_success" in metrics
        assert "complexity_handling" in metrics
        assert "scalability_score" in metrics
        assert "adaptability_score" in metrics
        
        # All metrics should be between 0 and 1
        for metric_value in metrics.values():
            assert 0 <= metric_value <= 1
    
    def test_generate_validation_report(self, deployment_system):
        """Test validation report generation"""
        validation_results = [
            TransformationValidationResult(
                organization_type=OrganizationType("Org1", "small", "tech", "emerging", "low"),
                assessment_accuracy=0.9,
                transformation_effectiveness=0.85,
                behavioral_change_success=0.8,
                engagement_improvement=0.82,
                sustainability_score=0.78,
                overall_success=0.83,
                validation_timestamp=datetime.now(),
                detailed_metrics={}
            ),
            TransformationValidationResult(
                organization_type=OrganizationType("Org2", "large", "finance", "mature", "high"),
                assessment_accuracy=0.88,
                transformation_effectiveness=0.82,
                behavioral_change_success=0.79,
                engagement_improvement=0.81,
                sustainability_score=0.76,
                overall_success=0.81,
                validation_timestamp=datetime.now(),
                detailed_metrics={}
            )
        ]
        
        with patch.object(deployment_system, '_group_results_by_size') as mock_size, \
             patch.object(deployment_system, '_group_results_by_industry') as mock_industry, \
             patch.object(deployment_system, '_group_results_by_complexity') as mock_complexity, \
             patch.object(deployment_system, '_generate_recommendations') as mock_recommendations:
            
            mock_size.return_value = {"small": 0.83, "large": 0.81}
            mock_industry.return_value = {"tech": 0.83, "finance": 0.81}
            mock_complexity.return_value = {"low": 0.83, "high": 0.81}
            mock_recommendations.return_value = ["recommendation1", "recommendation2"]
            
            report = deployment_system._generate_validation_report(validation_results)
            
            assert isinstance(report, dict)
            assert "overall_validation_success" in report
            assert "total_organizations_tested" in report
            assert "success_by_size" in report
            assert "success_by_industry" in report
            assert "success_by_complexity" in report
            assert "recommendations" in report
            assert report["total_organizations_tested"] == 2
            assert 0.8 < report["overall_validation_success"] < 0.85
    
    def test_get_system_status(self, deployment_system):
        """Test system status retrieval"""
        # Add some validation results
        deployment_system.validation_results = [
            TransformationValidationResult(
                organization_type=OrganizationType("Test", "medium", "tech", "mature", "medium"),
                assessment_accuracy=0.9,
                transformation_effectiveness=0.85,
                behavioral_change_success=0.8,
                engagement_improvement=0.82,
                sustainability_score=0.78,
                overall_success=0.83,
                validation_timestamp=datetime.now(),
                detailed_metrics={}
            )
        ]
        
        status = deployment_system.get_system_status()
        
        assert isinstance(status, dict)
        assert "status" in status
        assert "components_count" in status
        assert "validation_results_count" in status
        assert "last_validation" in status
        assert status["components_count"] > 0
        assert status["validation_results_count"] == 1
        assert status["last_validation"] is not None
    
    @pytest.mark.asyncio
    async def test_system_health_validation(self, deployment_system):
        """Test system health validation"""
        with patch.object(deployment_system, '_check_component_status', new_callable=AsyncMock) as mock_components, \
             patch.object(deployment_system, '_check_integration_status', new_callable=AsyncMock) as mock_integrations, \
             patch.object(deployment_system, '_check_performance_metrics', new_callable=AsyncMock) as mock_performance, \
             patch.object(deployment_system, '_check_resource_availability', new_callable=AsyncMock) as mock_resources, \
             patch.object(deployment_system, '_check_security_validation', new_callable=AsyncMock) as mock_security:
            
            mock_components.return_value = {"status": "healthy"}
            mock_integrations.return_value = {"status": "healthy"}
            mock_performance.return_value = {"status": "healthy"}
            mock_resources.return_value = {"status": "healthy"}
            mock_security.return_value = {"status": "healthy"}
            
            health = await deployment_system._validate_system_health()
            
            assert health["overall_health"] == "healthy"
            assert "health_checks" in health
            assert len(health["health_checks"]) == 5
            
            mock_components.assert_called_once()
            mock_integrations.assert_called_once()
            mock_performance.assert_called_once()
            mock_resources.assert_called_once()
            mock_security.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_production_deployment(self, deployment_system):
        """Test production deployment steps"""
        with patch.object(deployment_system, '_validate_production_config', new_callable=AsyncMock) as mock_config, \
             patch.object(deployment_system, '_apply_security_hardening', new_callable=AsyncMock) as mock_security, \
             patch.object(deployment_system, '_optimize_performance', new_callable=AsyncMock) as mock_performance, \
             patch.object(deployment_system, '_setup_monitoring', new_callable=AsyncMock) as mock_monitoring, \
             patch.object(deployment_system, '_configure_backups', new_callable=AsyncMock) as mock_backups, \
             patch.object(deployment_system, '_configure_load_balancing', new_callable=AsyncMock) as mock_balancing:
            
            mock_config.return_value = {"config_validation": "passed"}
            mock_security.return_value = {"security_hardening": "applied"}
            mock_performance.return_value = {"performance_optimization": "applied"}
            mock_monitoring.return_value = {"monitoring": "configured"}
            mock_backups.return_value = {"backups": "configured"}
            mock_balancing.return_value = {"load_balancing": "configured"}
            
            result = await deployment_system._deploy_production_system()
            
            assert isinstance(result, dict)
            assert len(result) == 6  # All deployment steps
            assert all(step["status"] == "success" for step in result.values())
            
            mock_config.assert_called_once()
            mock_security.assert_called_once()
            mock_performance.assert_called_once()
            mock_monitoring.assert_called_once()
            mock_backups.assert_called_once()
            mock_balancing.assert_called_once()
    
    def test_organization_type_creation(self, deployment_system):
        """Test organization type creation and characteristics"""
        org_types = deployment_system._get_organization_types()
        
        for org_type in org_types:
            assert isinstance(org_type.name, str)
            assert org_type.size in ["startup", "small", "medium", "large", "enterprise"]
            assert isinstance(org_type.industry, str)
            assert org_type.culture_maturity in ["emerging", "developing", "mature", "advanced"]
            assert org_type.complexity in ["low", "medium", "high", "very_high"]
    
    def test_employee_count_mapping(self, deployment_system):
        """Test employee count mapping for different organization sizes"""
        assert deployment_system._get_employee_count("startup") == 25
        assert deployment_system._get_employee_count("small") == 100
        assert deployment_system._get_employee_count("medium") == 500
        assert deployment_system._get_employee_count("large") == 2000
        assert deployment_system._get_employee_count("enterprise") == 10000
        assert deployment_system._get_employee_count("unknown") == 500  # default
    
    def test_department_mapping(self, deployment_system):
        """Test department mapping for different industries"""
        tech_departments = deployment_system._get_departments("technology")
        assert "Engineering" in tech_departments
        assert "Product" in tech_departments
        
        healthcare_departments = deployment_system._get_departments("healthcare")
        assert "Clinical" in healthcare_departments
        
        finance_departments = deployment_system._get_departments("financial")
        assert "Trading" in finance_departments
        assert "Risk" in finance_departments
    
    def test_complexity_score_calculation(self, deployment_system):
        """Test complexity score calculation"""
        assert deployment_system._calculate_complexity_score(
            OrganizationType("Test", "medium", "tech", "mature", "low")
        ) == 0.95
        
        assert deployment_system._calculate_complexity_score(
            OrganizationType("Test", "medium", "tech", "mature", "very_high")
        ) == 0.65
    
    def test_scalability_score_calculation(self, deployment_system):
        """Test scalability score calculation"""
        assert deployment_system._calculate_scalability_score(
            OrganizationType("Test", "startup", "tech", "mature", "medium")
        ) == 0.9
        
        assert deployment_system._calculate_scalability_score(
            OrganizationType("Test", "enterprise", "tech", "mature", "medium")
        ) == 0.7
    
    def test_adaptability_score_calculation(self, deployment_system):
        """Test adaptability score calculation"""
        assert deployment_system._calculate_adaptability_score(
            OrganizationType("Test", "medium", "tech", "emerging", "medium")
        ) == 0.9
        
        assert deployment_system._calculate_adaptability_score(
            OrganizationType("Test", "medium", "tech", "advanced", "medium")
        ) == 0.75

@pytest.mark.integration
class TestCulturalTransformationDeploymentIntegration:
    """Integration tests for cultural transformation deployment"""
    
    @pytest.mark.asyncio
    async def test_full_deployment_workflow(self):
        """Test complete deployment workflow"""
        deployment_system = CulturalTransformationDeployment()
        
        # Test deployment
        with patch.object(deployment_system, '_initialize_components', new_callable=AsyncMock), \
             patch.object(deployment_system, '_integrate_components', new_callable=AsyncMock), \
             patch.object(deployment_system, '_validate_system_health', new_callable=AsyncMock) as mock_health, \
             patch.object(deployment_system, '_deploy_production_system', new_callable=AsyncMock) as mock_deploy:
            
            mock_health.return_value = {"overall_health": "healthy"}
            mock_deploy.return_value = {"deployment_status": "success"}
            
            deployment_result = await deployment_system.deploy_complete_system()
            assert deployment_result["deployment_status"] == "success"
        
        # Test validation
        with patch.object(deployment_system, '_validate_organization_type', new_callable=AsyncMock) as mock_validate:
            mock_validate.return_value = TransformationValidationResult(
                organization_type=OrganizationType("Test", "medium", "tech", "mature", "medium"),
                assessment_accuracy=0.9,
                transformation_effectiveness=0.85,
                behavioral_change_success=0.8,
                engagement_improvement=0.82,
                sustainability_score=0.78,
                overall_success=0.83,
                validation_timestamp=datetime.now(),
                detailed_metrics={}
            )
            
            validation_results = await deployment_system.validate_across_organization_types()
            assert len(validation_results) > 0
        
        # Test learning system
        with patch.object(deployment_system, '_setup_feedback_collection', new_callable=AsyncMock), \
             patch.object(deployment_system, '_setup_performance_monitoring', new_callable=AsyncMock), \
             patch.object(deployment_system, '_setup_adaptation_engine', new_callable=AsyncMock), \
             patch.object(deployment_system, '_setup_knowledge_base', new_callable=AsyncMock), \
             patch.object(deployment_system, '_setup_improvement_pipeline', new_callable=AsyncMock), \
             patch.object(deployment_system, '_integrate_learning_system', new_callable=AsyncMock):
            
            learning_result = await deployment_system.create_continuous_learning_system()
            assert learning_result["learning_system_status"] == "active"
        
        # Verify final system status
        final_status = deployment_system.get_system_status()
        assert final_status["status"] == "deployed"
        assert final_status["validation_results_count"] > 0