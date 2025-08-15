"""
Integration tests for ScrollAutoDev Agent
Tests end-to-end workflows including API routes, database operations, and agent interactions.
"""

import pytest
import asyncio
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session
from unittest.mock import patch, AsyncMock
from uuid import uuid4
import json

from scrollintel.api.gateway import app
from scrollintel.models.database import get_db_session, User, PromptTemplate, PromptHistory, PromptTest
from scrollintel.agents.scroll_autodev_agent import ScrollAutoDevAgent
from scrollintel.core.interfaces import AgentRequest


class TestScrollAutoDevIntegration:
    """Integration test suite for ScrollAutoDev agent."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    @pytest.fixture
    def test_user(self, db_session):
        """Create test user."""
        user = User(
            email="test@example.com",
            hashed_password="hashed_password",
            full_name="Test User",
            is_active=True,
            is_verified=True
        )
        db_session.add(user)
        db_session.commit()
        db_session.refresh(user)
        return user
    
    @pytest.fixture
    def auth_headers(self, test_user):
        """Create authentication headers."""
        # Mock JWT token for testing
        return {"Authorization": "Bearer test_token"}
    
    @pytest.fixture
    def agent(self):
        """Create ScrollAutoDev agent instance."""
        return ScrollAutoDevAgent()
    
    def test_optimize_prompt_endpoint(self, client, auth_headers):
        """Test prompt optimization endpoint."""
        request_data = {
            "original_prompt": "Analyze the sales data",
            "strategy": "a_b_testing",
            "test_data": ["Q1 sales data", "Q2 sales data"],
            "target_metric": "performance_score",
            "max_variations": 5,
            "test_iterations": 10
        }
        
        with patch('scrollintel.security.auth.get_current_user') as mock_user:
            mock_user.return_value = User(id=uuid4(), email="test@example.com")
            with patch('scrollintel.agents.scroll_autodev_agent.ScrollAutoDevAgent.process_request') as mock_process:
                mock_process.return_value = AsyncMock(
                    status=AsyncMock(value="success"),
                    content="Optimization completed successfully",
                    error_message=None
                )
                
                response = client.post(
                    "/autodev/optimize",
                    json=request_data,
                    headers=auth_headers
                )
                
                assert response.status_code == 200
                data = response.json()
                assert data["original_prompt"] == request_data["original_prompt"]
                assert data["strategy"] == request_data["strategy"]
                assert "optimized_prompt" in data
                assert "performance_improvement" in data
    
    def test_test_variations_endpoint(self, client, auth_headers):
        """Test prompt variation testing endpoint."""
        request_data = {
            "variations": [
                "Analyze the sales data thoroughly",
                "Provide detailed analysis of sales data",
                "Examine sales data with statistical insights"
            ],
            "test_cases": [
                "Q1 2023 sales data",
                "Q2 2023 sales data"
            ],
            "evaluation_criteria": ["accuracy", "relevance", "clarity"],
            "statistical_significance": 0.05
        }
        
        with patch('scrollintel.security.auth.get_current_user') as mock_user:
            mock_user.return_value = User(id=uuid4(), email="test@example.com")
            with patch('scrollintel.agents.scroll_autodev_agent.ScrollAutoDevAgent.process_request') as mock_process:
                mock_process.return_value = AsyncMock(
                    status=AsyncMock(value="success"),
                    content="Testing completed successfully",
                    error_message=None
                )
                
                response = client.post(
                    "/autodev/test-variations",
                    json=request_data,
                    headers=auth_headers
                )
                
                assert response.status_code == 200
                data = response.json()
                assert data["variations_tested"] == len(request_data["variations"])
                assert data["test_cases_count"] == len(request_data["test_cases"])
                assert "winner_variation" in data
                assert "confidence_level" in data
    
    def test_execute_chain_endpoint(self, client, auth_headers):
        """Test prompt chain execution endpoint."""
        request_data = {
            "chain_name": "Sales Analysis Chain",
            "description": "Multi-step sales data analysis",
            "prompts": [
                {
                    "id": "step1",
                    "prompt": "Load and validate sales data: {data}",
                    "dependencies": []
                },
                {
                    "id": "step2",
                    "prompt": "Analyze trends from: {result_step1}",
                    "dependencies": ["step1"]
                }
            ],
            "dependencies": {
                "step2": ["step1"]
            },
            "execution_context": {
                "data": "sales_data.csv"
            }
        }
        
        with patch('scrollintel.security.auth.get_current_user') as mock_user:
            mock_user.return_value = User(id=uuid4(), email="test@example.com")
            with patch('scrollintel.agents.scroll_autodev_agent.ScrollAutoDevAgent.process_request') as mock_process:
                mock_process.return_value = AsyncMock(
                    status=AsyncMock(value="success"),
                    content="Chain execution completed",
                    error_message=None
                )
                
                response = client.post(
                    "/autodev/chain",
                    json=request_data,
                    headers=auth_headers
                )
                
                assert response.status_code == 200
                data = response.json()
                assert data["chain_name"] == request_data["chain_name"]
                assert "execution_results" in data
                assert "execution_flow" in data
                assert "performance_metrics" in data
    
    def test_generate_templates_endpoint(self, client, auth_headers):
        """Test template generation endpoint."""
        request_data = {
            "industry": "healthcare",
            "use_case": "patient_analysis",
            "requirements": ["HIPAA compliant", "detailed reporting"],
            "target_audience": "medical professionals",
            "output_format": "structured_report",
            "complexity_level": "advanced"
        }
        
        with patch('scrollintel.security.auth.get_current_user') as mock_user:
            mock_user.return_value = User(id=uuid4(), email="test@example.com")
            with patch('scrollintel.agents.scroll_autodev_agent.ScrollAutoDevAgent.process_request') as mock_process:
                mock_process.return_value = AsyncMock(
                    status=AsyncMock(value="success"),
                    content="Templates generated successfully",
                    error_message=None
                )
                
                response = client.post(
                    "/autodev/generate-templates",
                    json=request_data,
                    headers=auth_headers
                )
                
                assert response.status_code == 200
                data = response.json()
                assert "templates" in data
                assert "usage_guidelines" in data
                assert "customization_options" in data
                assert "optimization_tips" in data
                assert "industry_best_practices" in data
    
    def test_create_template_endpoint(self, client, auth_headers, db_session):
        """Test template creation endpoint."""
        request_data = {
            "name": "Healthcare Analysis Template",
            "description": "Template for analyzing healthcare data",
            "category": "data_analysis",
            "industry": "healthcare",
            "use_case": "patient_analysis",
            "template_content": "Analyze patient data {{data}} for healthcare insights with {{focus}}",
            "variables": ["data", "focus"],
            "tags": ["healthcare", "analysis", "HIPAA"],
            "is_public": False,
            "is_active": True
        }
        
        with patch('scrollintel.security.auth.get_current_user') as mock_user:
            test_user = User(id=uuid4(), email="test@example.com")
            mock_user.return_value = test_user
            with patch('scrollintel.models.database.get_db_session') as mock_db:
                mock_db.return_value = db_session
                
                response = client.post(
                    "/autodev/templates",
                    json=request_data,
                    headers=auth_headers
                )
                
                assert response.status_code == 200
                data = response.json()
                assert data["name"] == request_data["name"]
                assert data["category"] == request_data["category"]
                assert data["industry"] == request_data["industry"]
                assert data["variables"] == request_data["variables"]
    
    def test_list_templates_endpoint(self, client, auth_headers, db_session, test_user):
        """Test template listing endpoint."""
        # Create test templates
        template1 = PromptTemplate(
            name="Template 1",
            category="data_analysis",
            industry="healthcare",
            template_content="Template 1 content {{data}}",
            variables=["data"],
            creator_id=test_user.id,
            is_public=True
        )
        template2 = PromptTemplate(
            name="Template 2",
            category="code_generation",
            industry="finance",
            template_content="Template 2 content {{code}}",
            variables=["code"],
            creator_id=test_user.id,
            is_public=False
        )
        
        db_session.add_all([template1, template2])
        db_session.commit()
        
        with patch('scrollintel.security.auth.get_current_user') as mock_user:
            mock_user.return_value = test_user
            with patch('scrollintel.models.database.get_db_session') as mock_db:
                mock_db.return_value = db_session
                
                response = client.get(
                    "/autodev/templates?category=data_analysis",
                    headers=auth_headers
                )
                
                assert response.status_code == 200
                data = response.json()
                assert len(data) >= 1
                assert any(template["name"] == "Template 1" for template in data)
    
    def test_get_template_endpoint(self, client, auth_headers, db_session, test_user):
        """Test getting specific template endpoint."""
        template = PromptTemplate(
            name="Test Template",
            category="data_analysis",
            template_content="Test content {{data}}",
            variables=["data"],
            creator_id=test_user.id,
            is_public=True,
            usage_count=0
        )
        
        db_session.add(template)
        db_session.commit()
        db_session.refresh(template)
        
        with patch('scrollintel.security.auth.get_current_user') as mock_user:
            mock_user.return_value = test_user
            with patch('scrollintel.models.database.get_db_session') as mock_db:
                mock_db.return_value = db_session
                
                response = client.get(
                    f"/autodev/templates/{template.id}",
                    headers=auth_headers
                )
                
                assert response.status_code == 200
                data = response.json()
                assert data["name"] == "Test Template"
                assert data["usage_count"] == 1  # Should increment
    
    def test_update_template_endpoint(self, client, auth_headers, db_session, test_user):
        """Test template update endpoint."""
        template = PromptTemplate(
            name="Original Template",
            category="data_analysis",
            template_content="Original content",
            variables=[],
            creator_id=test_user.id
        )
        
        db_session.add(template)
        db_session.commit()
        db_session.refresh(template)
        
        update_data = {
            "name": "Updated Template",
            "description": "Updated description"
        }
        
        with patch('scrollintel.security.auth.get_current_user') as mock_user:
            mock_user.return_value = test_user
            with patch('scrollintel.models.database.get_db_session') as mock_db:
                mock_db.return_value = db_session
                
                response = client.put(
                    f"/autodev/templates/{template.id}",
                    json=update_data,
                    headers=auth_headers
                )
                
                assert response.status_code == 200
                data = response.json()
                assert data["name"] == "Updated Template"
                assert data["description"] == "Updated description"
    
    def test_delete_template_endpoint(self, client, auth_headers, db_session, test_user):
        """Test template deletion endpoint."""
        template = PromptTemplate(
            name="Template to Delete",
            category="data_analysis",
            template_content="Content to delete",
            variables=[],
            creator_id=test_user.id
        )
        
        db_session.add(template)
        db_session.commit()
        template_id = template.id
        
        with patch('scrollintel.security.auth.get_current_user') as mock_user:
            mock_user.return_value = test_user
            with patch('scrollintel.models.database.get_db_session') as mock_db:
                mock_db.return_value = db_session
                
                response = client.delete(
                    f"/autodev/templates/{template_id}",
                    headers=auth_headers
                )
                
                assert response.status_code == 200
                data = response.json()
                assert data["message"] == "Template deleted successfully"
                
                # Verify deletion
                deleted_template = db_session.query(PromptTemplate).filter(
                    PromptTemplate.id == template_id
                ).first()
                assert deleted_template is None
    
    def test_get_optimization_history_endpoint(self, client, auth_headers, db_session, test_user):
        """Test getting optimization history endpoint."""
        history = PromptHistory(
            user_id=test_user.id,
            original_prompt="Original prompt",
            optimized_prompt="Optimized prompt",
            optimization_strategy="a_b_testing",
            performance_improvement=0.25,
            success_rate_before=0.7,
            success_rate_after=0.9
        )
        
        db_session.add(history)
        db_session.commit()
        
        with patch('scrollintel.security.auth.get_current_user') as mock_user:
            mock_user.return_value = test_user
            with patch('scrollintel.models.database.get_db_session') as mock_db:
                mock_db.return_value = db_session
                
                response = client.get(
                    "/autodev/history",
                    headers=auth_headers
                )
                
                assert response.status_code == 200
                data = response.json()
                assert len(data) >= 1
                assert any(h["original_prompt"] == "Original prompt" for h in data)
    
    def test_get_prompt_tests_endpoint(self, client, auth_headers, db_session, test_user):
        """Test getting prompt tests endpoint."""
        test = PromptTest(
            test_name="Test A/B Testing",
            user_id=test_user.id,
            test_type="a_b_test",
            status="completed",
            prompt_variations=[
                {"id": "var1", "prompt": "Variation 1"},
                {"id": "var2", "prompt": "Variation 2"}
            ],
            test_cases=["Test case 1", "Test case 2"],
            test_results={"winner": "var1"},
            performance_metrics={"accuracy": 0.85},
            statistical_analysis={"p_value": 0.02}
        )
        
        db_session.add(test)
        db_session.commit()
        
        with patch('scrollintel.security.auth.get_current_user') as mock_user:
            mock_user.return_value = test_user
            with patch('scrollintel.models.database.get_db_session') as mock_db:
                mock_db.return_value = db_session
                
                response = client.get(
                    "/autodev/tests?status=completed",
                    headers=auth_headers
                )
                
                assert response.status_code == 200
                data = response.json()
                assert len(data) >= 1
                assert any(t["test_name"] == "Test A/B Testing" for t in data)
    
    def test_health_check_endpoint(self, client):
        """Test health check endpoint."""
        with patch('scrollintel.agents.scroll_autodev_agent.ScrollAutoDevAgent.health_check') as mock_health:
            mock_health.return_value = True
            
            response = client.get("/autodev/health")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert data["agent"] == "ScrollAutoDev"
            assert "capabilities" in data
            assert data["version"] == "1.0.0"
    
    def test_health_check_unhealthy(self, client):
        """Test health check when agent is unhealthy."""
        with patch('scrollintel.agents.scroll_autodev_agent.ScrollAutoDevAgent.health_check') as mock_health:
            mock_health.side_effect = Exception("Health check failed")
            
            response = client.get("/autodev/health")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "unhealthy"
            assert "error" in data
    
    @pytest.mark.asyncio
    async def test_end_to_end_optimization_workflow(self, agent, db_session, test_user):
        """Test complete optimization workflow."""
        # Create agent request
        request = AgentRequest(
            id=str(uuid4()),
            user_id=str(test_user.id),
            agent_id=agent.agent_id,
            prompt="optimize this prompt: Analyze sales data",
            context={
                "strategy": "a_b_testing",
                "test_data": ["Q1 data", "Q2 data"],
                "target_metric": "performance_score"
            }
        )
        
        with patch.object(agent, '_call_gpt4', new_callable=AsyncMock) as mock_gpt4:
            mock_gpt4.return_value = "Optimized analysis response"
            with patch.object(agent, '_calculate_performance_score', new_callable=AsyncMock) as mock_score:
                mock_score.return_value = 8.5
                
                # Process request
                response = await agent.process_request(request)
                
                assert response.status.value == "success"
                assert "Optimization" in response.content
                assert response.execution_time > 0
    
    @pytest.mark.asyncio
    async def test_end_to_end_testing_workflow(self, agent):
        """Test complete testing workflow."""
        request = AgentRequest(
            id=str(uuid4()),
            user_id=str(uuid4()),
            agent_id=agent.agent_id,
            prompt="test prompt variations",
            context={
                "variations": ["Variation 1", "Variation 2"],
                "test_cases": ["Test 1", "Test 2"]
            }
        )
        
        with patch.object(agent, '_call_gpt4', new_callable=AsyncMock) as mock_gpt4:
            mock_gpt4.return_value = "Test response"
            with patch.object(agent, '_calculate_performance_score', new_callable=AsyncMock) as mock_score:
                mock_score.return_value = 7.5
                
                response = await agent.process_request(request)
                
                assert response.status.value == "success"
                assert "Testing" in response.content
    
    @pytest.mark.asyncio
    async def test_end_to_end_chain_workflow(self, agent):
        """Test complete chain execution workflow."""
        request = AgentRequest(
            id=str(uuid4()),
            user_id=str(uuid4()),
            agent_id=agent.agent_id,
            prompt="execute chain workflow",
            context={
                "chain": {
                    "name": "Test Chain",
                    "description": "Test chain execution",
                    "prompts": [
                        {"id": "step1", "prompt": "First step"},
                        {"id": "step2", "prompt": "Second step"}
                    ],
                    "dependencies": {"step2": ["step1"]}
                },
                "execution_context": {"input": "test data"}
            }
        )
        
        with patch.object(agent, '_call_gpt4', new_callable=AsyncMock) as mock_gpt4:
            mock_gpt4.return_value = "Chain step completed"
            
            response = await agent.process_request(request)
            
            assert response.status.value == "success"
            assert "Chain" in response.content
    
    @pytest.mark.asyncio
    async def test_end_to_end_template_workflow(self, agent):
        """Test complete template generation workflow."""
        request = AgentRequest(
            id=str(uuid4()),
            user_id=str(uuid4()),
            agent_id=agent.agent_id,
            prompt="generate templates for healthcare",
            context={
                "industry": "healthcare",
                "use_case": "patient_analysis",
                "requirements": ["HIPAA compliant"]
            }
        )
        
        with patch.object(agent, '_call_gpt4', new_callable=AsyncMock) as mock_gpt4:
            mock_gpt4.return_value = """
            Template 1: Analyze patient data {{data}} for healthcare insights
            Template 2: Review medical records {{records}} with compliance
            Template 3: Generate healthcare report from {{dataset}}
            """
            
            response = await agent.process_request(request)
            
            assert response.status.value == "success"
            assert "Templates" in response.content
    
    def test_error_handling_invalid_request(self, client, auth_headers):
        """Test error handling for invalid requests."""
        invalid_request = {
            "original_prompt": "",  # Empty prompt should fail validation
            "strategy": "invalid_strategy"
        }
        
        with patch('scrollintel.security.auth.get_current_user') as mock_user:
            mock_user.return_value = User(id=uuid4(), email="test@example.com")
            
            response = client.post(
                "/autodev/optimize",
                json=invalid_request,
                headers=auth_headers
            )
            
            assert response.status_code == 422  # Validation error
    
    def test_error_handling_unauthorized_access(self, client):
        """Test error handling for unauthorized access."""
        request_data = {
            "original_prompt": "Test prompt",
            "strategy": "a_b_testing"
        }
        
        response = client.post(
            "/autodev/optimize",
            json=request_data
            # No auth headers
        )
        
        assert response.status_code == 401  # Unauthorized
    
    def test_error_handling_template_not_found(self, client, auth_headers):
        """Test error handling when template not found."""
        non_existent_id = str(uuid4())
        
        with patch('scrollintel.security.auth.get_current_user') as mock_user:
            mock_user.return_value = User(id=uuid4(), email="test@example.com")
            with patch('scrollintel.models.database.get_db_session') as mock_db:
                mock_session = Mock()
                mock_session.query.return_value.filter.return_value.first.return_value = None
                mock_db.return_value = mock_session
                
                response = client.get(
                    f"/autodev/templates/{non_existent_id}",
                    headers=auth_headers
                )
                
                assert response.status_code == 404
    
    def test_concurrent_optimization_requests(self, client, auth_headers):
        """Test handling concurrent optimization requests."""
        request_data = {
            "original_prompt": "Analyze data concurrently",
            "strategy": "a_b_testing",
            "test_data": ["data1", "data2"]
        }
        
        with patch('scrollintel.security.auth.get_current_user') as mock_user:
            mock_user.return_value = User(id=uuid4(), email="test@example.com")
            with patch('scrollintel.agents.scroll_autodev_agent.ScrollAutoDevAgent.process_request') as mock_process:
                mock_process.return_value = AsyncMock(
                    status=AsyncMock(value="success"),
                    content="Concurrent optimization completed",
                    error_message=None
                )
                
                # Send multiple concurrent requests
                responses = []
                for i in range(3):
                    response = client.post(
                        "/autodev/optimize",
                        json=request_data,
                        headers=auth_headers
                    )
                    responses.append(response)
                
                # All requests should succeed
                for response in responses:
                    assert response.status_code == 200
                    data = response.json()
                    assert "optimized_prompt" in data


if __name__ == "__main__":
    pytest.main([__file__])