"""
End-to-End Workflow Integration Tests
Tests complete business scenarios from data ingestion to decision making
"""
import pytest
import asyncio
import json
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
import pandas as pd
import numpy as np

from scrollintel.core.realtime_orchestration_engine import RealtimeOrchestrationEngine
from scrollintel.engines.intelligence_engine import IntelligenceEngine
from scrollintel.core.agent_registry import AgentRegistry
from scrollintel.agents.scroll_cto_agent import ScrollCTOAgent
from scrollintel.agents.scroll_bi_agent import ScrollBIAgent
from scrollintel.agents.scroll_ml_engineer import ScrollMLEngineer
from scrollintel.core.data_pipeline import DataPipeline
from scrollintel.models.agent_steering_models import BusinessTask, TaskPriority, BusinessContext


class TestEndToEndWorkflows:
    """Test complete business workflows from start to finish"""
    
    @pytest.fixture
    def orchestration_engine(self):
        """Create orchestration engine with mock dependencies"""
        engine = RealtimeOrchestrationEngine()
        engine.agent_registry = Mock()
        engine.intelligence_engine = Mock()
        engine.data_pipeline = Mock()
        return engine
    
    @pytest.fixture
    def business_scenarios(self):
        """Real-world business scenarios for testing"""
        return {
            'customer_churn_prediction': {
                'description': 'Predict and prevent customer churn using ML analysis',
                'data_sources': ['crm', 'support_tickets', 'usage_analytics'],
                'expected_agents': ['data_scientist', 'ml_engineer', 'bi_analyst'],
                'success_criteria': {
                    'prediction_accuracy': 0.85,
                    'response_time': 300,  # 5 minutes
                    'actionable_insights': 5
                }
            },
            'financial_fraud_detection': {
                'description': 'Real-time fraud detection and response',
                'data_sources': ['transactions', 'user_behavior', 'external_feeds'],
                'expected_agents': ['security_analyst', 'ml_engineer', 'compliance_officer'],
                'success_criteria': {
                    'detection_accuracy': 0.95,
                    'response_time': 30,  # 30 seconds
                    'false_positive_rate': 0.02
                }
            },
            'supply_chain_optimization': {
                'description': 'Optimize supply chain operations and inventory',
                'data_sources': ['erp', 'suppliers', 'logistics', 'market_data'],
                'expected_agents': ['operations_analyst', 'forecast_agent', 'cto_agent'],
                'success_criteria': {
                    'cost_reduction': 0.15,
                    'inventory_optimization': 0.20,
                    'delivery_improvement': 0.10
                }
            },
            'market_intelligence_analysis': {
                'description': 'Comprehensive market analysis and competitive intelligence',
                'data_sources': ['market_data', 'competitor_analysis', 'social_media', 'news'],
                'expected_agents': ['market_analyst', 'competitive_intelligence', 'strategy_advisor'],
                'success_criteria': {
                    'insight_quality': 0.90,
                    'market_coverage': 0.95,
                    'strategic_recommendations': 10
                }
            }
        }
    
    @pytest.fixture
    def mock_enterprise_data(self):
        """Mock enterprise data for workflows"""
        return {
            'customers': pd.DataFrame({
                'customer_id': range(1, 10001),
                'signup_date': pd.date_range('2020-01-01', periods=10000),
                'last_activity': pd.date_range('2024-01-01', periods=10000),
                'total_spent': np.random.uniform(100, 50000, 10000),
                'support_tickets': np.random.poisson(2, 10000),
                'churn_risk': np.random.uniform(0, 1, 10000)
            }),
            'transactions': pd.DataFrame({
                'transaction_id': range(1, 100001),
                'customer_id': np.random.randint(1, 10001, 100000),
                'amount': np.random.uniform(10, 5000, 100000),
                'timestamp': pd.date_range('2024-01-01', periods=100000, freq='5min'),
                'merchant': np.random.choice(['Amazon', 'Walmart', 'Target'], 100000),
                'fraud_score': np.random.uniform(0, 1, 100000)
            }),
            'inventory': pd.DataFrame({
                'product_id': range(1, 5001),
                'current_stock': np.random.randint(0, 1000, 5000),
                'reorder_point': np.random.randint(50, 200, 5000),
                'lead_time_days': np.random.randint(1, 30, 5000),
                'demand_forecast': np.random.randint(10, 500, 5000),
                'supplier_reliability': np.random.uniform(0.7, 1.0, 5000)
            })
        }
    
    @pytest.mark.asyncio
    async def test_customer_churn_prediction_workflow(self, orchestration_engine, business_scenarios, mock_enterprise_data):
        """Test complete customer churn prediction workflow"""
        scenario = business_scenarios['customer_churn_prediction']
        
        # Mock agent responses
        with patch.multiple(
            'scrollintel.agents',
            ScrollDataScientist=Mock(),
            ScrollMLEngineer=Mock(),
            ScrollBIAgent=Mock()
        ) as mock_agents:
            
            # Configure mock agents
            for agent_name, agent_mock in mock_agents.items():
                agent_instance = Mock()
                agent_instance.process_task = AsyncMock()
                agent_instance.get_capabilities.return_value = ['data_analysis', 'ml_modeling']
                agent_mock.return_value = agent_instance
            
            # Create business task
            task = BusinessTask(
                id='churn_prediction_001',
                title='Customer Churn Prediction Analysis',
                description=scenario['description'],
                priority=TaskPriority.HIGH,
                business_context=BusinessContext(
                    objective='Reduce customer churn by 25%',
                    timeline='2 weeks',
                    budget=50000,
                    stakeholders=['CMO', 'Head of Customer Success']
                )
            )
            
            # Mock data pipeline
            orchestration_engine.data_pipeline.extract_data = AsyncMock(
                return_value=mock_enterprise_data['customers']
            )
            
            # Mock intelligence engine
            orchestration_engine.intelligence_engine.analyze_churn_patterns = AsyncMock(
                return_value={
                    'high_risk_customers': 1250,
                    'churn_probability_model': {'accuracy': 0.87, 'precision': 0.82},
                    'key_factors': ['support_tickets', 'last_activity', 'total_spent'],
                    'recommended_actions': [
                        'Proactive customer outreach for high-risk segments',
                        'Improve support response times',
                        'Implement loyalty program for high-value customers'
                    ]
                }
            )
            
            # Execute workflow
            start_time = time.time()
            result = await orchestration_engine.execute_business_workflow(task)
            execution_time = time.time() - start_time
            
            # Validate results
            assert result['status'] == 'completed'
            assert result['churn_analysis']['accuracy'] >= scenario['success_criteria']['prediction_accuracy']
            assert execution_time <= scenario['success_criteria']['response_time']
            assert len(result['actionable_insights']) >= scenario['success_criteria']['actionable_insights']
            
            # Validate agent coordination
            assert result['agents_involved'] >= len(scenario['expected_agents'])
            assert 'data_processing_time' in result['performance_metrics']
            assert 'model_training_time' in result['performance_metrics']
    
    @pytest.mark.asyncio
    async def test_fraud_detection_workflow(self, orchestration_engine, business_scenarios, mock_enterprise_data):
        """Test real-time fraud detection workflow"""
        scenario = business_scenarios['financial_fraud_detection']
        
        # Mock real-time transaction stream
        async def mock_transaction_stream():
            for i in range(100):
                yield {
                    'transaction_id': f'txn_{i}',
                    'amount': np.random.uniform(10, 5000),
                    'timestamp': datetime.now(),
                    'merchant': np.random.choice(['Amazon', 'Walmart', 'Target']),
                    'customer_id': np.random.randint(1, 10000)
                }
                await asyncio.sleep(0.01)  # Simulate real-time stream
        
        with patch('scrollintel.core.data_pipeline.get_transaction_stream', mock_transaction_stream):
            # Mock fraud detection models
            orchestration_engine.intelligence_engine.detect_fraud = AsyncMock(
                return_value={
                    'fraud_probability': 0.95,
                    'risk_factors': ['unusual_amount', 'new_merchant', 'time_pattern'],
                    'recommended_action': 'block_transaction',
                    'confidence_score': 0.92
                }
            )
            
            # Create real-time fraud detection task
            task = BusinessTask(
                id='fraud_detection_001',
                title='Real-time Fraud Detection',
                description=scenario['description'],
                priority=TaskPriority.CRITICAL,
                business_context=BusinessContext(
                    objective='Prevent fraudulent transactions in real-time',
                    timeline='immediate',
                    compliance_requirements=['PCI-DSS', 'SOX']
                )
            )
            
            # Execute real-time workflow
            fraud_alerts = []
            start_time = time.time()
            
            async for transaction in mock_transaction_stream():
                result = await orchestration_engine.process_realtime_transaction(transaction)
                
                if result['fraud_detected']:
                    fraud_alerts.append(result)
                    response_time = time.time() - start_time
                    
                    # Validate real-time requirements
                    assert response_time <= scenario['success_criteria']['response_time']
                    assert result['fraud_probability'] >= scenario['success_criteria']['detection_accuracy']
                
                if len(fraud_alerts) >= 5:  # Test with 5 fraud cases
                    break
            
            # Validate fraud detection performance
            assert len(fraud_alerts) > 0
            false_positive_rate = sum(1 for alert in fraud_alerts if alert['false_positive']) / len(fraud_alerts)
            assert false_positive_rate <= scenario['success_criteria']['false_positive_rate']
    
    @pytest.mark.asyncio
    async def test_supply_chain_optimization_workflow(self, orchestration_engine, business_scenarios, mock_enterprise_data):
        """Test supply chain optimization workflow"""
        scenario = business_scenarios['supply_chain_optimization']
        
        with patch.multiple(
            'scrollintel.agents',
            OperationsAnalyst=Mock(),
            ForecastAgent=Mock(),
            ScrollCTOAgent=Mock()
        ) as mock_agents:
            
            # Configure supply chain analysis
            orchestration_engine.intelligence_engine.optimize_supply_chain = AsyncMock(
                return_value={
                    'inventory_optimization': {
                        'cost_reduction': 0.18,
                        'stock_optimization': 0.22,
                        'reorder_recommendations': 150
                    },
                    'supplier_analysis': {
                        'reliability_scores': {'supplier_a': 0.95, 'supplier_b': 0.87},
                        'cost_analysis': {'potential_savings': 125000},
                        'risk_assessment': {'high_risk_suppliers': 3}
                    },
                    'logistics_optimization': {
                        'delivery_improvement': 0.12,
                        'route_optimization': {'fuel_savings': 0.15},
                        'warehouse_efficiency': 0.20
                    }
                }
            )
            
            # Create supply chain task
            task = BusinessTask(
                id='supply_chain_opt_001',
                title='Supply Chain Optimization Analysis',
                description=scenario['description'],
                priority=TaskPriority.HIGH,
                business_context=BusinessContext(
                    objective='Reduce supply chain costs by 15% while improving delivery times',
                    timeline='1 month',
                    budget=200000
                )
            )
            
            # Mock data from multiple sources
            orchestration_engine.data_pipeline.extract_multi_source_data = AsyncMock(
                return_value={
                    'inventory': mock_enterprise_data['inventory'],
                    'suppliers': pd.DataFrame({
                        'supplier_id': range(1, 51),
                        'reliability_score': np.random.uniform(0.7, 1.0, 50),
                        'cost_per_unit': np.random.uniform(10, 100, 50),
                        'lead_time': np.random.randint(1, 30, 50)
                    }),
                    'logistics': pd.DataFrame({
                        'route_id': range(1, 101),
                        'distance': np.random.uniform(50, 500, 100),
                        'cost': np.random.uniform(100, 1000, 100),
                        'delivery_time': np.random.uniform(1, 10, 100)
                    })
                }
            )
            
            # Execute workflow
            result = await orchestration_engine.execute_business_workflow(task)
            
            # Validate optimization results
            assert result['status'] == 'completed'
            assert result['cost_reduction'] >= scenario['success_criteria']['cost_reduction']
            assert result['inventory_optimization'] >= scenario['success_criteria']['inventory_optimization']
            assert result['delivery_improvement'] >= scenario['success_criteria']['delivery_improvement']
            
            # Validate business impact
            assert 'roi_projection' in result
            assert 'implementation_plan' in result
            assert len(result['optimization_recommendations']) > 0
    
    @pytest.mark.asyncio
    async def test_market_intelligence_workflow(self, orchestration_engine, business_scenarios):
        """Test comprehensive market intelligence analysis workflow"""
        scenario = business_scenarios['market_intelligence_analysis']
        
        # Mock market data sources
        market_data = {
            'competitor_analysis': pd.DataFrame({
                'competitor': ['CompanyA', 'CompanyB', 'CompanyC'],
                'market_share': [0.25, 0.20, 0.15],
                'revenue_growth': [0.12, 0.08, 0.15],
                'product_launches': [3, 2, 4],
                'customer_satisfaction': [4.2, 3.8, 4.5]
            }),
            'market_trends': pd.DataFrame({
                'trend': ['AI_adoption', 'Cloud_migration', 'Sustainability'],
                'growth_rate': [0.35, 0.28, 0.22],
                'market_size': [50000000, 75000000, 30000000],
                'adoption_stage': ['growth', 'maturity', 'early']
            }),
            'social_sentiment': pd.DataFrame({
                'platform': ['Twitter', 'LinkedIn', 'Reddit'],
                'mention_volume': [15000, 8000, 5000],
                'sentiment_score': [0.65, 0.72, 0.58],
                'engagement_rate': [0.08, 0.12, 0.15]
            })
        }
        
        orchestration_engine.data_pipeline.extract_market_data = AsyncMock(
            return_value=market_data
        )
        
        # Mock intelligence analysis
        orchestration_engine.intelligence_engine.analyze_market_intelligence = AsyncMock(
            return_value={
                'competitive_positioning': {
                    'market_position': 'strong',
                    'competitive_advantages': ['technology', 'customer_service'],
                    'threats': ['new_entrants', 'price_competition'],
                    'opportunities': ['emerging_markets', 'product_expansion']
                },
                'market_insights': {
                    'growth_opportunities': [
                        {'market': 'AI_services', 'potential': 0.40, 'timeline': '6_months'},
                        {'market': 'Enterprise_cloud', 'potential': 0.25, 'timeline': '12_months'}
                    ],
                    'risk_factors': [
                        {'risk': 'Economic_downturn', 'probability': 0.30, 'impact': 'high'},
                        {'risk': 'Regulatory_changes', 'probability': 0.20, 'impact': 'medium'}
                    ]
                },
                'strategic_recommendations': [
                    'Invest in AI capabilities to capture 40% growth opportunity',
                    'Strengthen customer retention programs',
                    'Expand into emerging markets within 12 months',
                    'Develop strategic partnerships to mitigate competitive threats'
                ]
            }
        )
        
        # Create market intelligence task
        task = BusinessTask(
            id='market_intel_001',
            title='Comprehensive Market Intelligence Analysis',
            description=scenario['description'],
            priority=TaskPriority.HIGH,
            business_context=BusinessContext(
                objective='Identify growth opportunities and competitive threats',
                timeline='2 weeks',
                stakeholders=['CEO', 'CMO', 'Head of Strategy']
            )
        )
        
        # Execute workflow
        result = await orchestration_engine.execute_business_workflow(task)
        
        # Validate market intelligence results
        assert result['status'] == 'completed'
        assert result['insight_quality'] >= scenario['success_criteria']['insight_quality']
        assert result['market_coverage'] >= scenario['success_criteria']['market_coverage']
        assert len(result['strategic_recommendations']) >= scenario['success_criteria']['strategic_recommendations']
        
        # Validate strategic insights
        assert 'competitive_analysis' in result
        assert 'growth_opportunities' in result
        assert 'risk_assessment' in result
        assert 'actionable_recommendations' in result
        
        # Validate business value
        assert 'revenue_impact_projection' in result
        assert 'implementation_roadmap' in result
    
    @pytest.mark.asyncio
    async def test_multi_agent_coordination(self, orchestration_engine):
        """Test coordination between multiple agents in complex workflows"""
        
        # Create complex multi-step task
        task = BusinessTask(
            id='complex_analysis_001',
            title='Multi-Domain Business Analysis',
            description='Comprehensive analysis requiring multiple specialized agents',
            priority=TaskPriority.HIGH,
            business_context=BusinessContext(
                objective='Complete business transformation analysis',
                timeline='1 week',
                budget=100000
            )
        )
        
        # Mock multiple agents
        agents = {
            'cto_agent': Mock(),
            'bi_agent': Mock(),
            'ml_engineer': Mock(),
            'data_scientist': Mock(),
            'compliance_officer': Mock()
        }
        
        for agent_name, agent_mock in agents.items():
            agent_mock.process_task = AsyncMock(return_value={
                'status': 'completed',
                'results': f'{agent_name}_analysis_results',
                'confidence': 0.9,
                'execution_time': 30
            })
            agent_mock.get_capabilities.return_value = [f'{agent_name}_capability']
        
        orchestration_engine.agent_registry.get_available_agents.return_value = list(agents.values())
        
        # Test agent selection and coordination
        selected_agents = await orchestration_engine.select_agents_for_task(task)
        assert len(selected_agents) >= 3  # Should select multiple agents
        
        # Test parallel execution
        start_time = time.time()
        results = await orchestration_engine.coordinate_parallel_execution(task, selected_agents)
        execution_time = time.time() - start_time
        
        # Validate coordination
        assert len(results) == len(selected_agents)
        assert all(result['status'] == 'completed' for result in results)
        assert execution_time < 60  # Should complete in parallel efficiently
        
        # Test result synthesis
        synthesized_result = await orchestration_engine.synthesize_agent_results(results)
        assert 'combined_insights' in synthesized_result
        assert 'confidence_score' in synthesized_result
        assert 'business_recommendations' in synthesized_result
    
    @pytest.mark.asyncio
    async def test_workflow_error_recovery(self, orchestration_engine):
        """Test error handling and recovery in complex workflows"""
        
        # Create task that will encounter errors
        task = BusinessTask(
            id='error_recovery_001',
            title='Error Recovery Test',
            description='Test workflow error handling',
            priority=TaskPriority.MEDIUM
        )
        
        # Mock agents with different failure scenarios
        failing_agent = Mock()
        failing_agent.process_task = AsyncMock(side_effect=Exception("Agent processing failed"))
        
        recovering_agent = Mock()
        call_count = 0
        
        async def mock_process_with_recovery(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise Exception("Temporary failure")
            return {'status': 'completed', 'results': 'recovered_results'}
        
        recovering_agent.process_task = mock_process_with_recovery
        
        successful_agent = Mock()
        successful_agent.process_task = AsyncMock(return_value={
            'status': 'completed',
            'results': 'successful_results'
        })
        
        agents = [failing_agent, recovering_agent, successful_agent]
        orchestration_engine.agent_registry.get_available_agents.return_value = agents
        
        # Test error recovery
        result = await orchestration_engine.execute_with_error_recovery(task)
        
        # Validate recovery behavior
        assert result['status'] == 'completed'
        assert result['failed_agents'] == 1  # One agent failed permanently
        assert result['recovered_agents'] == 1  # One agent recovered after retries
        assert result['successful_agents'] == 1  # One agent succeeded immediately
        
        # Validate that workflow continued despite failures
        assert 'partial_results' in result
        assert len(result['completed_tasks']) >= 1
    
    @pytest.mark.asyncio
    async def test_performance_under_load(self, orchestration_engine):
        """Test workflow performance under high load"""
        
        # Create multiple concurrent tasks
        tasks = []
        for i in range(20):
            task = BusinessTask(
                id=f'load_test_{i}',
                title=f'Load Test Task {i}',
                description='Performance testing task',
                priority=TaskPriority.MEDIUM
            )
            tasks.append(task)
        
        # Mock fast-responding agents
        mock_agent = Mock()
        mock_agent.process_task = AsyncMock(return_value={
            'status': 'completed',
            'results': 'load_test_results',
            'execution_time': 1
        })
        
        orchestration_engine.agent_registry.get_available_agents.return_value = [mock_agent] * 10
        
        # Execute concurrent workflows
        start_time = time.time()
        results = await asyncio.gather(*[
            orchestration_engine.execute_business_workflow(task) for task in tasks
        ])
        total_time = time.time() - start_time
        
        # Validate performance
        assert len(results) == 20
        assert all(result['status'] == 'completed' for result in results)
        assert total_time < 30  # Should complete within 30 seconds with parallelization
        
        # Validate resource utilization
        performance_metrics = await orchestration_engine.get_performance_metrics()
        assert performance_metrics['concurrent_tasks'] == 20
        assert performance_metrics['average_response_time'] < 5
        assert performance_metrics['throughput'] > 0.5  # Tasks per second


class TestWorkflowValidation:
    """Test workflow validation and quality assurance"""
    
    @pytest.mark.asyncio
    async def test_business_rule_validation(self, orchestration_engine):
        """Test validation of business rules in workflows"""
        
        # Define business rules
        business_rules = {
            'budget_limits': {'max_budget': 1000000, 'approval_required_above': 500000},
            'compliance_requirements': ['SOX', 'GDPR', 'PCI-DSS'],
            'data_retention': {'max_days': 2555, 'archive_after_days': 365},
            'security_levels': {'min_encryption': 'AES-256', 'access_control': 'RBAC'}
        }
        
        # Test rule validation
        task = BusinessTask(
            id='rule_validation_001',
            title='Business Rule Validation Test',
            description='Test business rule compliance',
            priority=TaskPriority.HIGH,
            business_context=BusinessContext(
                budget=750000,  # Above approval threshold
                compliance_requirements=['SOX', 'GDPR'],
                security_level='high'
            )
        )
        
        validation_result = await orchestration_engine.validate_business_rules(task, business_rules)
        
        # Validate rule checking
        assert validation_result['budget_approval_required'] is True
        assert validation_result['compliance_satisfied'] is True
        assert validation_result['security_adequate'] is True
        assert validation_result['overall_valid'] is True
    
    @pytest.mark.asyncio
    async def test_data_quality_validation(self, orchestration_engine, mock_enterprise_data):
        """Test data quality validation in workflows"""
        
        # Introduce data quality issues
        poor_quality_data = mock_enterprise_data['customers'].copy()
        poor_quality_data.loc[0:100, 'total_spent'] = None  # Missing values
        poor_quality_data.loc[200:300, 'customer_id'] = -1  # Invalid IDs
        
        # Test data quality validation
        quality_report = await orchestration_engine.validate_data_quality(
            poor_quality_data,
            quality_rules={
                'completeness': {'threshold': 0.95},
                'validity': {'customer_id': {'min': 1}},
                'consistency': {'total_spent': {'type': 'numeric', 'min': 0}}
            }
        )
        
        # Validate quality assessment
        assert quality_report['completeness_score'] < 0.95
        assert quality_report['validity_issues'] > 0
        assert quality_report['overall_quality'] == 'poor'
        assert len(quality_report['recommendations']) > 0
    
    @pytest.mark.asyncio
    async def test_result_accuracy_validation(self, orchestration_engine):
        """Test validation of workflow result accuracy"""
        
        # Mock workflow results
        workflow_results = {
            'predictions': [0.8, 0.6, 0.9, 0.3, 0.7],
            'actual_outcomes': [1, 0, 1, 0, 1],
            'confidence_scores': [0.9, 0.8, 0.95, 0.7, 0.85],
            'business_metrics': {
                'revenue_impact': 150000,
                'cost_savings': 75000,
                'efficiency_gain': 0.20
            }
        }
        
        # Test accuracy validation
        accuracy_report = await orchestration_engine.validate_result_accuracy(workflow_results)
        
        # Validate accuracy metrics
        assert 'prediction_accuracy' in accuracy_report
        assert 'confidence_calibration' in accuracy_report
        assert 'business_impact_validation' in accuracy_report
        assert accuracy_report['overall_accuracy'] >= 0.7


if __name__ == "__main__":
    pytest.main([__file__, "-v"])