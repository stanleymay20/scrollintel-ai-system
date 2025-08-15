"""
Integration Tests for Agent Interactions
Tests multi-agent workflows and communication patterns
"""
import pytest
import asyncio
from typing import Dict, Any
from unittest.mock import patch, Mock

from scrollintel.core.registry import AgentRegistry
from scrollintel.agents.scroll_cto_agent import ScrollCTOAgent
from scrollintel.agents.scroll_data_scientist import ScrollDataScientist
from scrollintel.agents.scroll_ml_engineer import ScrollMLEngineer
from scrollintel.agents.scroll_ai_engineer import ScrollAIEngineer
from scrollintel.agents.scroll_analyst import ScrollAnalyst
from scrollintel.agents.scroll_bi_agent import ScrollBIAgent
from scrollintel.core.orchestrator import TaskOrchestrator


class TestAgentInteractions:
    """Test agent-to-agent communication and workflows"""
    
    @pytest.mark.asyncio
    async def test_cto_to_data_scientist_workflow(self, agent_registry, mock_ai_services, test_helper):
        """Test CTO agent requesting data analysis from Data Scientist"""
        # Setup agents
        cto_agent = ScrollCTOAgent()
        ds_agent = ScrollDataScientist()
        
        agent_registry.register_agent(cto_agent)
        agent_registry.register_agent(ds_agent)
        
        # Mock AI responses
        with patch('scrollintel.agents.scroll_cto_agent.openai') as mock_openai:
            mock_openai.chat.completions.create.return_value = Mock(
                choices=[Mock(message=Mock(content="Request data analysis for user behavior patterns"))]
            )
            
            # CTO requests analysis
            cto_request = {
                "prompt": "Analyze user behavior patterns for scaling decisions",
                "context": {"data_source": "user_analytics"}
            }
            
            cto_response = await cto_agent.process_request(cto_request)
            test_helper.assert_agent_response(cto_response)
            
            # Verify CTO can delegate to Data Scientist
            assert "data analysis" in cto_response['content'].lower()
    
    @pytest.mark.asyncio
    async def test_data_scientist_to_ml_engineer_pipeline(self, agent_registry, sample_datasets, mock_ai_services):
        """Test Data Scientist to ML Engineer model training pipeline"""
        ds_agent = ScrollDataScientist()
        ml_agent = ScrollMLEngineer()
        
        agent_registry.register_agent(ds_agent)
        agent_registry.register_agent(ml_agent)
        
        # Data Scientist analyzes data
        ds_request = {
            "prompt": "Analyze ML dataset and recommend model approach",
            "context": {"dataset": sample_datasets['ml_data'].to_dict()}
        }
        
        with patch('scrollintel.agents.scroll_data_scientist.anthropic') as mock_claude:
            mock_claude.messages.create.return_value = Mock(
                content=[Mock(text="Recommend classification model with feature engineering")]
            )
            
            ds_response = await ds_agent.process_request(ds_request)
            assert ds_response['status'] == 'success'
            
            # ML Engineer receives recommendation
            ml_request = {
                "prompt": "Train classification model based on data scientist recommendation",
                "context": {
                    "dataset": sample_datasets['ml_data'].to_dict(),
                    "recommendation": ds_response['content']
                }
            }
            
            ml_response = await ml_agent.process_request(ml_request)
            assert ml_response['status'] == 'success'
    
    @pytest.mark.asyncio
    async def test_ai_engineer_rag_integration(self, agent_registry, mock_ai_services):
        """Test AI Engineer RAG capabilities with vector operations"""
        ai_agent = ScrollAIEngineer()
        agent_registry.register_agent(ai_agent)
        
        # Mock vector database operations
        with patch('scrollintel.agents.scroll_ai_engineer.pinecone') as mock_pinecone:
            mock_pinecone.query.return_value = {
                'matches': [
                    {'id': 'doc1', 'score': 0.9, 'metadata': {'text': 'Sample document content'}}
                ]
            }
            
            rag_request = {
                "prompt": "Find information about machine learning best practices",
                "context": {"use_rag": True}
            }
            
            response = await ai_agent.process_request(rag_request)
            assert response['status'] == 'success'
            assert 'machine learning' in response['content'].lower()
    
    @pytest.mark.asyncio
    async def test_analyst_to_bi_dashboard_workflow(self, agent_registry, sample_datasets, mock_ai_services):
        """Test Analyst to BI Agent dashboard creation workflow"""
        analyst = ScrollAnalyst()
        bi_agent = ScrollBIAgent()
        
        agent_registry.register_agent(analyst)
        agent_registry.register_agent(bi_agent)
        
        # Analyst generates insights
        analyst_request = {
            "prompt": "Generate KPIs from sales data",
            "context": {"dataset": sample_datasets['csv_data'].to_dict()}
        }
        
        with patch('scrollintel.agents.scroll_analyst.openai') as mock_openai:
            mock_openai.chat.completions.create.return_value = Mock(
                choices=[Mock(message=Mock(content="KPIs: Revenue Growth 15%, Customer Acquisition 200"))]
            )
            
            analyst_response = await analyst.process_request(analyst_request)
            assert analyst_response['status'] == 'success'
            
            # BI Agent creates dashboard
            bi_request = {
                "prompt": "Create dashboard from analyst insights",
                "context": {
                    "insights": analyst_response['content'],
                    "dataset": sample_datasets['csv_data'].to_dict()
                }
            }
            
            bi_response = await bi_agent.process_request(bi_request)
            assert bi_response['status'] == 'success'
    
    @pytest.mark.asyncio
    async def test_orchestrated_multi_agent_workflow(self, agent_registry, sample_datasets, mock_ai_services):
        """Test orchestrated workflow involving multiple agents"""
        orchestrator = TaskOrchestrator(agent_registry)
        
        # Register all agents
        agents = [
            ScrollCTOAgent(),
            ScrollDataScientist(),
            ScrollMLEngineer(),
            ScrollAnalyst(),
            ScrollBIAgent()
        ]
        
        for agent in agents:
            agent_registry.register_agent(agent)
        
        # Define complex workflow
        workflow_request = {
            "workflow_type": "complete_analysis",
            "prompt": "Perform complete analysis from data to dashboard",
            "context": {
                "dataset": sample_datasets['ml_data'].to_dict(),
                "requirements": ["analysis", "modeling", "visualization"]
            }
        }
        
        # Mock all AI services
        with patch('scrollintel.agents.scroll_cto_agent.openai') as mock_openai, \
             patch('scrollintel.agents.scroll_data_scientist.anthropic') as mock_claude, \
             patch('scrollintel.agents.scroll_ml_engineer.openai') as mock_ml_openai:
            
            # Configure mocks
            mock_openai.chat.completions.create.return_value = Mock(
                choices=[Mock(message=Mock(content="Orchestrate data analysis workflow"))]
            )
            mock_claude.messages.create.return_value = Mock(
                content=[Mock(text="Data analysis complete, recommend ML model")]
            )
            mock_ml_openai.chat.completions.create.return_value = Mock(
                choices=[Mock(message=Mock(content="ML model trained successfully"))]
            )
            
            # Execute workflow
            workflow_result = await orchestrator.execute_workflow(workflow_request)
            
            assert workflow_result['status'] == 'success'
            assert len(workflow_result['steps']) > 1
            assert all(step['status'] == 'success' for step in workflow_result['steps'])
    
    @pytest.mark.asyncio
    async def test_agent_error_handling_and_recovery(self, agent_registry, mock_ai_services):
        """Test agent error handling and recovery mechanisms"""
        cto_agent = ScrollCTOAgent()
        agent_registry.register_agent(cto_agent)
        
        # Test with failing AI service
        with patch('scrollintel.agents.scroll_cto_agent.openai') as mock_openai:
            # First call fails
            mock_openai.chat.completions.create.side_effect = Exception("API Error")
            
            request = {
                "prompt": "Provide technical architecture recommendation",
                "context": {}
            }
            
            response = await cto_agent.process_request(request)
            
            # Should handle error gracefully
            assert response['status'] == 'error'
            assert 'error' in response['content'].lower()
    
    @pytest.mark.asyncio
    async def test_concurrent_agent_requests(self, agent_registry, sample_datasets, mock_ai_services):
        """Test concurrent requests to multiple agents"""
        agents = [
            ScrollCTOAgent(),
            ScrollDataScientist(),
            ScrollMLEngineer(),
            ScrollAnalyst()
        ]
        
        for agent in agents:
            agent_registry.register_agent(agent)
        
        # Create concurrent requests
        requests = [
            {
                "agent": agents[0],
                "request": {"prompt": "Architecture decision", "context": {}}
            },
            {
                "agent": agents[1],
                "request": {"prompt": "Data analysis", "context": {"data": sample_datasets['csv_data'].to_dict()}}
            },
            {
                "agent": agents[2],
                "request": {"prompt": "Model training", "context": {"data": sample_datasets['ml_data'].to_dict()}}
            },
            {
                "agent": agents[3],
                "request": {"prompt": "Generate insights", "context": {"data": sample_datasets['csv_data'].to_dict()}}
            }
        ]
        
        # Mock all AI services
        with patch('scrollintel.agents.scroll_cto_agent.openai') as mock_openai, \
             patch('scrollintel.agents.scroll_data_scientist.anthropic') as mock_claude, \
             patch('scrollintel.agents.scroll_ml_engineer.openai') as mock_ml_openai, \
             patch('scrollintel.agents.scroll_analyst.openai') as mock_analyst_openai:
            
            # Configure mocks
            mock_openai.chat.completions.create.return_value = Mock(
                choices=[Mock(message=Mock(content="Architecture recommendation"))]
            )
            mock_claude.messages.create.return_value = Mock(
                content=[Mock(text="Data analysis results")]
            )
            mock_ml_openai.chat.completions.create.return_value = Mock(
                choices=[Mock(message=Mock(content="Model training complete"))]
            )
            mock_analyst_openai.chat.completions.create.return_value = Mock(
                choices=[Mock(message=Mock(content="Business insights generated"))]
            )
            
            # Execute concurrent requests
            tasks = [
                req["agent"].process_request(req["request"]) 
                for req in requests
            ]
            
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Verify all requests completed
            assert len(responses) == 4
            for response in responses:
                if isinstance(response, Exception):
                    pytest.fail(f"Request failed with exception: {response}")
                assert response['status'] in ['success', 'error']
    
    @pytest.mark.asyncio
    async def test_agent_capability_discovery(self, agent_registry):
        """Test agent capability discovery and routing"""
        # Register agents with different capabilities
        cto_agent = ScrollCTOAgent()
        ds_agent = ScrollDataScientist()
        ml_agent = ScrollMLEngineer()
        
        agent_registry.register_agent(cto_agent)
        agent_registry.register_agent(ds_agent)
        agent_registry.register_agent(ml_agent)
        
        # Test capability discovery
        capabilities = agent_registry.get_capabilities()
        
        assert len(capabilities) >= 3
        
        # Test routing based on capabilities
        architecture_request = {
            "prompt": "Design system architecture",
            "required_capabilities": ["architecture", "technical_decisions"]
        }
        
        routed_agent = agent_registry.route_request(architecture_request)
        assert routed_agent is not None
        assert isinstance(routed_agent, ScrollCTOAgent)
        
        # Test data analysis routing
        analysis_request = {
            "prompt": "Perform statistical analysis",
            "required_capabilities": ["data_analysis", "statistics"]
        }
        
        routed_agent = agent_registry.route_request(analysis_request)
        assert routed_agent is not None
        assert isinstance(routed_agent, ScrollDataScientist)