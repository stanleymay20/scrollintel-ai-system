"""
Tests for Crisis Leadership Excellence Deployment System

Comprehensive test suite for deployment, validation, and continuous learning
of the crisis leadership excellence system.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock

from scrollintel.core.crisis_leadership_excellence_deployment import (
    CrisisLeadershipExcellenceDeployment,
    ValidationLevel,
    DeploymentStatus,
    CrisisScenario,
    DeploymentMetrics
)
from scrollintel.core.crisis_leadership_excellence import CrisisType, CrisisSeverity


class TestCrisisLeadershipExcellenceDeployment:
    """Test suite for crisis leadership excellence deployment"""
    
    @pytest.fixture
    def deployment_system(self):
        """Create deployment system instance for testing"""
        return CrisisLeadershipExcellenceDeployment()
    
    @pytest.fixture
    def sample_crisis_scenario(self):
        """Create sample crisis scenario for testing"""
        return CrisisScenario(
            scenario_id="test_scenario_001",
            crisis_type=CrisisType.SYSTEM_OUTAGE,
            severity=CrisisSeverity.HIGH,
            description="Test system outage scenario",
            signals=[
                {"type": "system_alert", "severity": "high", "affected_services": ["api", "database"]},
                {"type": "customer_complaints", "volume": "medium", "sentiment": "negative"}
            ],
            expected_outcomes={
                "response_time": 300,
                "stakeholder_notification": True,
                "team_formation": True
            },
            success_criteria={
                "response_speed": 0.8,
                "communication_effectiveness": 0.75,
                "team_coordination": 0.8
            }
        )
    
    @pytest.mark.asyncio
    async def test_deployment_system_initialization(self, deployment_system):
        """Test deployment system initialization"""
        assert deployment_system.deployment_status == DeploymentStatus.INITIALIZING
        assert deployment_system.crisis_system is not None
        assert deployment_system.effectiveness_tester is not None
        assert len(deployment_system.test_scenarios) > 0
        assert deployment_system.learning_data is not None
    
    @pytest.mark.asyncio
    async def test_basic_deployment(self, deployment_system):
        """Test basic deployment with minimal validation"""
        with patch.object(deployment_system, '_validate_system_components', new_callable=AsyncMock) as mock_validate:
            with patch.object(deployment_system, '_test_system_integration', new_callable=AsyncMock) as mock_integration:
                with patch.object(deployment_system, '_test_crisis_response_capabilities', new_callable=AsyncMock) as mock_capabilities:
                    with patch.object(deployment_system, '_benchmark_system_performance', new_callable=AsyncMock) as mock_benchmark:
                        with patch.object(deployment_system, '_setup_continuous_learning', new_callable=AsyncMock) as mock_learning:
                            with patch.object(deployment_system, '_finalize_deployment', new_callable=AsyncMock) as mock_finalize:
                                
                                # Mock successful validation
                                deployment_system.deployment_metrics = DeploymentMetrics(
                                    deployment_timestamp=datetime.now(),
                                    validation_level=ValidationLevel.BASIC,
                                    component_health={'test_component': 0.9},
                                    integration_scores={'test_integration': 0.85},
                                    crisis_response_capabilities={'test_capability': 0.8},
                                    performance_benchmarks={'test_performance': 0.82},
                                    continuous_learning_metrics={'learning_system_active': True},
                                    overall_readiness_score=0.85,
                                    deployment_success=True
                                )
                                
                                result = await deployment_system.deploy_complete_system(ValidationLevel.BASIC)
                                
                                assert result.deployment_success is True
                                assert result.overall_readiness_score > 0.8
                                assert deployment_system.deployment_status == DeploymentStatus.DEPLOYED
                                
                                # Verify all phases were called
                                mock_validate.assert_called_once()
                                mock_integration.assert_called_once()
                                mock_capabilities.assert_called_once()
                                mock_benchmark.assert_called_once()
                                mock_learning.assert_called_once()
                                mock_finalize.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_comprehensive_deployment(self, deployment_system):
        """Test comprehensive deployment with full validation"""
        with patch.object(deployment_system.crisis_system, 'handle_crisis', new_callable=AsyncMock) as mock_handle_crisis:
            with patch.object(deployment_system.effectiveness_tester, 'test_crisis_response_effectiveness', new_callable=AsyncMock) as mock_effectiveness:
                
                # Mock crisis response
                mock_response = Mock()
                mock_response.__dict__ = {
                    'crisis_id': 'test_crisis_001',
                    'response_plan': {'actions': ['notify_stakeholders', 'form_team']},
                    'team_formation': {'team_members': ['leader', 'technical', 'communication']},
                    'resource_allocation': {'resources': ['servers', 'personnel']},
                    'communication_strategy': {'channels': ['email', 'slack', 'phone']},
                    'success_metrics': {'response_time': 180, 'effectiveness': 0.85}
                }
                mock_handle_crisis.return_value = mock_response
                
                # Mock effectiveness testing
                mock_effectiveness.return_value = {
                    'overall_score': 0.85,
                    'leadership_score': 0.8,
                    'stakeholder_satisfaction': 0.82,
                    'communication_effectiveness': 0.88
                }
                
                result = await deployment_system.deploy_complete_system(ValidationLevel.COMPREHENSIVE)
                
                assert result.deployment_success is True
                assert result.validation_level == ValidationLevel.COMPREHENSIVE
                assert len(deployment_system.validation_history) > 0
    
    @pytest.mark.asyncio
    async def test_component_validation(self, deployment_system):
        """Test individual component validation"""
        await deployment_system._validate_system_components()
        
        assert deployment_system.deployment_metrics is not None
        assert len(deployment_system.deployment_metrics.component_health) > 0
        
        # Check that all expected components are validated
        expected_components = [
            'crisis_detector', 'decision_engine', 'info_synthesizer', 'risk_analyzer',
            'stakeholder_notifier', 'message_coordinator', 'media_manager',
            'resource_assessor', 'resource_allocator', 'external_coordinator',
            'team_former', 'role_assigner', 'performance_monitor'
        ]
        
        for component in expected_components:
            assert component in deployment_system.deployment_metrics.component_health
            assert 0.0 <= deployment_system.deployment_metrics.component_health[component] <= 1.0
    
    @pytest.mark.asyncio
    async def test_component_health_assessment(self, deployment_system):
        """Test component health assessment"""
        # Test with mock component
        mock_component = Mock()
        mock_component.__dict__ = {'initialized': True}
        
        health_score = await deployment_system._test_component_health('test_component', mock_component)
        
        assert 0.0 <= health_score <= 1.0
        assert health_score > 0.5  # Should have reasonable health score
    
    @pytest.mark.asyncio
    async def test_system_integration_testing(self, deployment_system):
        """Test system integration testing"""
        deployment_system.deployment_metrics = DeploymentMetrics(
            deployment_timestamp=datetime.now(),
            validation_level=ValidationLevel.BASIC
        )
        
        with patch.object(deployment_system.crisis_system.crisis_detector, 'detect_crisis', new_callable=AsyncMock) as mock_detect:
            mock_detect.return_value = {'crisis_detected': True, 'type': 'system_outage'}
            
            await deployment_system._test_system_integration(ValidationLevel.BASIC)
            
            assert len(deployment_system.deployment_metrics.integration_scores) > 0
            
            # Check integration test results
            for score in deployment_system.deployment_metrics.integration_scores.values():
                assert 0.0 <= score <= 1.0
    
    @pytest.mark.asyncio
    async def test_crisis_response_capabilities_testing(self, deployment_system):
        """Test crisis response capabilities testing"""
        deployment_system.deployment_metrics = DeploymentMetrics(
            deployment_timestamp=datetime.now(),
            validation_level=ValidationLevel.BASIC
        )
        
        with patch.object(deployment_system.crisis_system, 'handle_crisis', new_callable=AsyncMock) as mock_handle:
            with patch.object(deployment_system.effectiveness_tester, 'test_crisis_response_effectiveness', new_callable=AsyncMock) as mock_effectiveness:
                
                # Mock crisis response
                mock_response = Mock()
                mock_response.__dict__ = {'crisis_id': 'test', 'response_plan': {}}
                mock_handle.return_value = mock_response
                
                # Mock effectiveness results
                mock_effectiveness.return_value = {
                    'overall_score': 0.85,
                    'stakeholder_satisfaction': 0.8,
                    'communication_effectiveness': 0.82,
                    'leadership_score': 0.88
                }
                
                await deployment_system._test_crisis_response_capabilities(ValidationLevel.BASIC)
                
                assert len(deployment_system.deployment_metrics.crisis_response_capabilities) > 0
                
                # Check capability scores
                for score in deployment_system.deployment_metrics.crisis_response_capabilities.values():
                    assert 0.0 <= score <= 1.0
    
    @pytest.mark.asyncio
    async def test_performance_benchmarking(self, deployment_system):
        """Test system performance benchmarking"""
        deployment_system.deployment_metrics = DeploymentMetrics(
            deployment_timestamp=datetime.now(),
            validation_level=ValidationLevel.BASIC
        )
        
        with patch.object(deployment_system.crisis_system.crisis_detector, 'detect_crisis', new_callable=AsyncMock) as mock_detect:
            with patch.object(deployment_system.crisis_system.decision_engine, 'get_decision_options', new_callable=AsyncMock) as mock_decision:
                with patch.object(deployment_system, '_test_system_throughput', new_callable=AsyncMock) as mock_throughput:
                    
                    mock_detect.return_value = {'crisis_detected': True}
                    mock_decision.return_value = {'options': ['option1', 'option2']}
                    mock_throughput.return_value = 0.8
                    
                    await deployment_system._benchmark_system_performance()
                    
                    assert len(deployment_system.deployment_metrics.performance_benchmarks) > 0
                    
                    # Check performance metrics
                    expected_metrics = [
                        'crisis_detection_speed', 'decision_making_speed', 'overall_response_time',
                        'system_throughput', 'memory_efficiency', 'scalability_score'
                    ]
                    
                    for metric in expected_metrics:
                        assert metric in deployment_system.deployment_metrics.performance_benchmarks
                        assert 0.0 <= deployment_system.deployment_metrics.performance_benchmarks[metric] <= 1.0
    
    @pytest.mark.asyncio
    async def test_system_throughput_testing(self, deployment_system):
        """Test system throughput testing"""
        with patch.object(deployment_system.crisis_system, 'handle_crisis', new_callable=AsyncMock) as mock_handle:
            # Mock successful responses
            mock_response = Mock()
            mock_response.__dict__ = {'crisis_id': 'test'}
            mock_handle.return_value = mock_response
            
            throughput_score = await deployment_system._test_system_throughput()
            
            assert 0.0 <= throughput_score <= 1.0
            assert mock_handle.call_count == 5  # Should test 5 concurrent scenarios
    
    @pytest.mark.asyncio
    async def test_continuous_learning_setup(self, deployment_system):
        """Test continuous learning system setup"""
        deployment_system.deployment_metrics = DeploymentMetrics(
            deployment_timestamp=datetime.now(),
            validation_level=ValidationLevel.BASIC
        )
        
        await deployment_system._setup_continuous_learning()
        
        assert len(deployment_system.deployment_metrics.continuous_learning_metrics) > 0
        assert deployment_system.deployment_metrics.continuous_learning_metrics['learning_system_active'] is True
        
        # Check learning system components
        expected_metrics = [
            'learning_system_active', 'data_collection_rate', 'pattern_recognition_accuracy',
            'improvement_identification', 'adaptation_speed', 'knowledge_retention'
        ]
        
        for metric in expected_metrics:
            assert metric in deployment_system.deployment_metrics.continuous_learning_metrics
    
    @pytest.mark.asyncio
    async def test_multi_crisis_handling(self, deployment_system):
        """Test multi-crisis handling capability"""
        with patch.object(deployment_system.crisis_system, 'handle_crisis', new_callable=AsyncMock) as mock_handle:
            # Mock successful responses
            mock_response = Mock()
            mock_response.__dict__ = {'crisis_id': 'test'}
            mock_handle.return_value = mock_response
            
            multi_crisis_score = await deployment_system._test_multi_crisis_handling()
            
            assert 0.0 <= multi_crisis_score <= 1.0
            assert mock_handle.call_count == 3  # Should handle 3 concurrent crises
    
    @pytest.mark.asyncio
    async def test_crisis_leadership_validation(self, deployment_system):
        """Test crisis leadership excellence validation"""
        with patch.object(deployment_system.crisis_system, 'handle_crisis', new_callable=AsyncMock) as mock_handle:
            with patch.object(deployment_system.effectiveness_tester, 'test_crisis_response_effectiveness', new_callable=AsyncMock) as mock_effectiveness:
                
                # Mock crisis response
                mock_response = Mock()
                mock_response.__dict__ = {'crisis_id': 'test', 'response_plan': {}}
                mock_handle.return_value = mock_response
                
                # Mock effectiveness results
                mock_effectiveness.return_value = {
                    'overall_score': 0.85,
                    'leadership_score': 0.8,
                    'stakeholder_satisfaction': 0.82
                }
                
                # Test validation with custom scenarios
                test_scenarios = [
                    {
                        'scenario_id': 'test_001',
                        'crisis_type': 'system_outage',
                        'signals': [{'type': 'system_alert', 'severity': 'high'}]
                    }
                ]
                
                result = await deployment_system.validate_crisis_leadership_excellence(test_scenarios)
                
                assert result['scenarios_tested'] == 1
                assert result['overall_success_rate'] >= 0.0
                assert result['average_response_time'] >= 0.0
                assert result['leadership_effectiveness'] >= 0.0
                assert result['stakeholder_satisfaction'] >= 0.0
                assert len(result['detailed_results']) == 1
    
    @pytest.mark.asyncio
    async def test_deployment_status_tracking(self, deployment_system):
        """Test deployment status tracking"""
        # Test initial status
        status = await deployment_system.get_deployment_status()
        assert status['deployment_status'] == DeploymentStatus.INITIALIZING.value
        
        # Test status after setting metrics
        deployment_system.deployment_metrics = DeploymentMetrics(
            deployment_timestamp=datetime.now(),
            validation_level=ValidationLevel.BASIC,
            overall_readiness_score=0.85,
            deployment_success=True
        )
        
        status = await deployment_system.get_deployment_status()
        assert status['deployment_metrics'] is not None
        assert status['system_health'] is not None
    
    @pytest.mark.asyncio
    async def test_continuous_learning_insights(self, deployment_system):
        """Test continuous learning insights generation"""
        # Add some mock learning data
        deployment_system.learning_data['crisis_responses'] = [
            {
                'scenario_id': 'test_001',
                'response_time': 120.0,
                'effectiveness': {'overall_score': 0.85},
                'timestamp': datetime.now()
            },
            {
                'scenario_id': 'test_002',
                'response_time': 150.0,
                'effectiveness': {'overall_score': 0.9},
                'timestamp': datetime.now()
            }
        ]
        
        insights = deployment_system.get_continuous_learning_insights()
        
        assert insights['total_crises_handled'] == 2
        assert insights['average_response_time'] == 135.0
        assert insights['average_effectiveness'] == 0.875
        assert insights['improvement_trend'] in ['improving', 'declining', 'stable', 'insufficient_data']
        assert len(insights['best_performing_scenarios']) <= 3
    
    @pytest.mark.asyncio
    async def test_stress_test_scenarios(self, deployment_system):
        """Test stress test scenario generation"""
        stress_scenarios = deployment_system._generate_stress_scenarios()
        
        assert len(stress_scenarios) > 0
        
        for scenario in stress_scenarios:
            assert scenario.complexity_level == "extreme"
            assert len(scenario.signals) > 3  # Stress scenarios should have multiple signals
            assert scenario.crisis_type in [CrisisType.SYSTEM_OUTAGE, CrisisType.OPERATIONAL_FAILURE]
    
    @pytest.mark.asyncio
    async def test_scenario_selection_by_validation_level(self, deployment_system):
        """Test scenario selection based on validation level"""
        # Test basic level
        basic_scenarios = deployment_system._select_test_scenarios(ValidationLevel.BASIC)
        assert len(basic_scenarios) <= 2
        
        # Test comprehensive level
        comprehensive_scenarios = deployment_system._select_test_scenarios(ValidationLevel.COMPREHENSIVE)
        assert len(comprehensive_scenarios) >= len(basic_scenarios)
        
        # Test stress test level
        stress_scenarios = deployment_system._select_test_scenarios(ValidationLevel.STRESS_TEST)
        assert len(stress_scenarios) >= len(comprehensive_scenarios)
        
        # Test production ready level
        production_scenarios = deployment_system._select_test_scenarios(ValidationLevel.PRODUCTION_READY)
        assert len(production_scenarios) >= len(stress_scenarios)
    
    @pytest.mark.asyncio
    async def test_readiness_score_calculation(self, deployment_system):
        """Test overall readiness score calculation"""
        deployment_system.deployment_metrics = DeploymentMetrics(
            deployment_timestamp=datetime.now(),
            validation_level=ValidationLevel.BASIC,
            component_health={'comp1': 0.9, 'comp2': 0.8},
            integration_scores={'int1': 0.85, 'int2': 0.9},
            crisis_response_capabilities={'cap1': 0.8, 'cap2': 0.85},
            performance_benchmarks={'perf1': 0.9, 'perf2': 0.8},
            continuous_learning_metrics={'learning_system_active': True, 'metric1': 0.85}
        )
        
        deployment_system._calculate_readiness_score()
        
        assert 0.0 <= deployment_system.deployment_metrics.overall_readiness_score <= 1.0
        assert deployment_system.deployment_metrics.overall_readiness_score > 0.5  # Should be reasonable
    
    @pytest.mark.asyncio
    async def test_improvement_opportunity_identification(self, deployment_system):
        """Test improvement opportunity identification"""
        deployment_system.deployment_metrics = DeploymentMetrics(
            deployment_timestamp=datetime.now(),
            validation_level=ValidationLevel.BASIC,
            component_health={'weak_component': 0.6, 'strong_component': 0.9},
            crisis_response_capabilities={'weak_capability': 0.65, 'strong_capability': 0.95},
            performance_benchmarks={'slow_metric': 0.7, 'fast_metric': 0.9}
        )
        
        opportunities = deployment_system._identify_improvement_opportunities()
        
        assert len(opportunities) > 0
        assert any('weak_component' in opp for opp in opportunities)
        assert any('weak_capability' in opp for opp in opportunities)
        assert any('slow_metric' in opp for opp in opportunities)
    
    @pytest.mark.asyncio
    async def test_deployment_failure_handling(self, deployment_system):
        """Test deployment failure handling"""
        with patch.object(deployment_system, '_validate_system_components', side_effect=Exception("Validation failed")):
            
            with pytest.raises(Exception, match="Validation failed"):
                await deployment_system.deploy_complete_system(ValidationLevel.BASIC)
            
            assert deployment_system.deployment_status == DeploymentStatus.FAILED
    
    @pytest.mark.asyncio
    async def test_emergency_deployment_scenario(self, deployment_system):
        """Test emergency deployment with minimal validation"""
        with patch.object(deployment_system, '_finalize_deployment', new_callable=AsyncMock) as mock_finalize:
            # Mock minimal validation success
            deployment_system.deployment_metrics = DeploymentMetrics(
                deployment_timestamp=datetime.now(),
                validation_level=ValidationLevel.BASIC,
                component_health={'essential_component': 0.75},
                integration_scores={'critical_integration': 0.7},
                crisis_response_capabilities={'emergency_response': 0.7},
                performance_benchmarks={'basic_performance': 0.7},
                continuous_learning_metrics={'learning_system_active': True},
                overall_readiness_score=0.7,
                deployment_success=True
            )
            
            result = await deployment_system.deploy_complete_system(ValidationLevel.BASIC)
            
            assert result.deployment_success is True
            assert result.overall_readiness_score >= 0.7  # Minimum acceptable for emergency
            assert deployment_system.deployment_status == DeploymentStatus.DEPLOYED
    
    def test_crisis_scenario_creation(self, sample_crisis_scenario):
        """Test crisis scenario creation and validation"""
        assert sample_crisis_scenario.scenario_id == "test_scenario_001"
        assert sample_crisis_scenario.crisis_type == CrisisType.SYSTEM_OUTAGE
        assert sample_crisis_scenario.severity == CrisisSeverity.HIGH
        assert len(sample_crisis_scenario.signals) == 2
        assert len(sample_crisis_scenario.expected_outcomes) == 3
        assert len(sample_crisis_scenario.success_criteria) == 3
        
        # Validate success criteria are within valid range
        for criterion, value in sample_crisis_scenario.success_criteria.items():
            assert 0.0 <= value <= 1.0
    
    @pytest.mark.asyncio
    async def test_deployment_metrics_persistence(self, deployment_system):
        """Test deployment metrics persistence in validation history"""
        # Create mock deployment metrics
        metrics1 = DeploymentMetrics(
            deployment_timestamp=datetime.now(),
            validation_level=ValidationLevel.BASIC,
            overall_readiness_score=0.8,
            deployment_success=True
        )
        
        metrics2 = DeploymentMetrics(
            deployment_timestamp=datetime.now() + timedelta(hours=1),
            validation_level=ValidationLevel.COMPREHENSIVE,
            overall_readiness_score=0.9,
            deployment_success=True
        )
        
        # Add to validation history
        deployment_system.validation_history.extend([metrics1, metrics2])
        
        assert len(deployment_system.validation_history) == 2
        assert deployment_system.validation_history[0].overall_readiness_score == 0.8
        assert deployment_system.validation_history[1].overall_readiness_score == 0.9
        assert deployment_system.validation_history[1].validation_level == ValidationLevel.COMPREHENSIVE


@pytest.mark.integration
class TestCrisisLeadershipDeploymentIntegration:
    """Integration tests for crisis leadership deployment"""
    
    @pytest.fixture
    def deployment_system(self):
        """Create deployment system for integration testing"""
        return CrisisLeadershipExcellenceDeployment()
    
    @pytest.mark.asyncio
    async def test_end_to_end_deployment_flow(self, deployment_system):
        """Test complete end-to-end deployment flow"""
        # This test would require actual system components to be available
        # For now, we'll test the flow with mocked components
        
        with patch.object(deployment_system.crisis_system, 'handle_crisis', new_callable=AsyncMock) as mock_handle:
            with patch.object(deployment_system.effectiveness_tester, 'test_crisis_response_effectiveness', new_callable=AsyncMock) as mock_effectiveness:
                
                # Mock successful crisis handling
                mock_response = Mock()
                mock_response.__dict__ = {
                    'crisis_id': 'integration_test_001',
                    'response_plan': {'actions': ['immediate_response']},
                    'team_formation': {'team_size': 5},
                    'resource_allocation': {'allocated_resources': 10},
                    'communication_strategy': {'channels': 3},
                    'success_metrics': {'response_time': 120, 'effectiveness': 0.9}
                }
                mock_handle.return_value = mock_response
                
                # Mock effectiveness testing
                mock_effectiveness.return_value = {
                    'overall_score': 0.9,
                    'leadership_score': 0.85,
                    'stakeholder_satisfaction': 0.88,
                    'communication_effectiveness': 0.92
                }
                
                # Execute full deployment
                result = await deployment_system.deploy_complete_system(ValidationLevel.COMPREHENSIVE)
                
                # Verify deployment success
                assert result.deployment_success is True
                assert result.overall_readiness_score > 0.8
                assert deployment_system.deployment_status == DeploymentStatus.DEPLOYED
                
                # Verify learning data was collected
                assert len(deployment_system.learning_data['crisis_responses']) > 0
                
                # Test validation after deployment
                validation_result = await deployment_system.validate_crisis_leadership_excellence()
                assert validation_result['overall_success_rate'] > 0.8
    
    @pytest.mark.asyncio
    async def test_concurrent_crisis_handling_during_deployment(self, deployment_system):
        """Test system's ability to handle crises during deployment validation"""
        
        with patch.object(deployment_system.crisis_system, 'handle_crisis', new_callable=AsyncMock) as mock_handle:
            # Mock concurrent crisis responses
            mock_responses = []
            for i in range(3):
                mock_response = Mock()
                mock_response.__dict__ = {
                    'crisis_id': f'concurrent_crisis_{i}',
                    'response_plan': {'priority': i},
                    'team_formation': {'team_id': i},
                    'resource_allocation': {'resources': i * 2},
                    'communication_strategy': {'urgency': 'high'},
                    'success_metrics': {'effectiveness': 0.8 + (i * 0.05)}
                }
                mock_responses.append(mock_response)
            
            mock_handle.side_effect = mock_responses
            
            # Test concurrent crisis handling
            throughput_score = await deployment_system._test_system_throughput()
            
            assert throughput_score > 0.5  # Should handle concurrent crises reasonably well
            assert mock_handle.call_count == 5  # Should attempt 5 concurrent scenarios