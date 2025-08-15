"""
Unit tests for ScrollAutoDev Agent - Advanced Prompt Engineering
Tests prompt optimization, A/B testing, template generation, and chain management.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from uuid import uuid4
from datetime import datetime

from scrollintel.agents.scroll_autodev_agent import (
    ScrollAutoDevAgent, PromptOptimizationStrategy, PromptCategory,
    PromptVariation, PromptTestResult, PromptChain
)
from scrollintel.core.interfaces import AgentRequest, AgentType, ResponseStatus


class TestScrollAutoDevAgent:
    """Test suite for ScrollAutoDev agent."""
    
    @pytest.fixture
    def agent(self):
        """Create ScrollAutoDev agent instance."""
        return ScrollAutoDevAgent()
    
    @pytest.fixture
    def sample_request(self):
        """Create sample agent request."""
        return AgentRequest(
            id=str(uuid4()),
            user_id=str(uuid4()),
            agent_id="scroll-autodev-agent",
            prompt="optimize this prompt: Analyze the data",
            context={"strategy": "a_b_testing", "test_data": ["sample1", "sample2"]}
        )
    
    def test_agent_initialization(self, agent):
        """Test agent initialization."""
        assert agent.agent_id == "scroll-autodev-agent"
        assert agent.name == "ScrollAutoDev Agent"
        assert agent.agent_type == AgentType.AI_ENGINEER
        assert len(agent.capabilities) == 4
        
        # Check capabilities
        capability_names = [cap.name for cap in agent.capabilities]
        assert "prompt_optimization" in capability_names
        assert "prompt_testing" in capability_names
        assert "prompt_chain_management" in capability_names
        assert "template_generation" in capability_names
    
    def test_prompt_templates_initialization(self, agent):
        """Test prompt templates are properly initialized."""
        assert len(agent.prompt_templates) > 0
        assert PromptCategory.DATA_ANALYSIS in agent.prompt_templates
        assert PromptCategory.CODE_GENERATION in agent.prompt_templates
        assert PromptCategory.BUSINESS_INTELLIGENCE in agent.prompt_templates
        assert PromptCategory.STRATEGIC_PLANNING in agent.prompt_templates
        
        # Check template content
        data_templates = agent.prompt_templates[PromptCategory.DATA_ANALYSIS]
        assert len(data_templates) > 0
        assert "{data}" in data_templates[0]
    
    @pytest.mark.asyncio
    async def test_process_request_optimization(self, agent, sample_request):
        """Test prompt optimization request processing."""
        with patch.object(agent, '_optimize_prompt', new_callable=AsyncMock) as mock_optimize:
            mock_optimize.return_value = "Optimization completed successfully"
            
            response = await agent.process_request(sample_request)
            
            assert response.status == ResponseStatus.SUCCESS
            assert "Optimization completed" in response.content
            assert response.execution_time > 0
            mock_optimize.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_process_request_testing(self, agent):
        """Test prompt variation testing request processing."""
        request = AgentRequest(
            id=str(uuid4()),
            user_id=str(uuid4()),
            agent_id="scroll-autodev-agent",
            prompt="test these prompt variations",
            context={
                "variations": ["Variation 1", "Variation 2"],
                "test_cases": ["Test case 1", "Test case 2"]
            }
        )
        
        with patch.object(agent, '_test_prompt_variations', new_callable=AsyncMock) as mock_test:
            mock_test.return_value = "Testing completed successfully"
            
            response = await agent.process_request(request)
            
            assert response.status == ResponseStatus.SUCCESS
            assert "Testing completed" in response.content
            mock_test.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_process_request_chain_management(self, agent):
        """Test prompt chain management request processing."""
        request = AgentRequest(
            id=str(uuid4()),
            user_id=str(uuid4()),
            agent_id="scroll-autodev-agent",
            prompt="manage prompt chain workflow",
            context={
                "chain": {
                    "name": "Test Chain",
                    "prompts": [{"id": "1", "prompt": "Step 1"}]
                }
            }
        )
        
        with patch.object(agent, '_manage_prompt_chain', new_callable=AsyncMock) as mock_chain:
            mock_chain.return_value = "Chain management completed"
            
            response = await agent.process_request(request)
            
            assert response.status == ResponseStatus.SUCCESS
            assert "Chain management" in response.content
            mock_chain.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_process_request_template_generation(self, agent):
        """Test template generation request processing."""
        request = AgentRequest(
            id=str(uuid4()),
            user_id=str(uuid4()),
            agent_id="scroll-autodev-agent",
            prompt="generate templates for healthcare analysis",
            context={
                "industry": "healthcare",
                "use_case": "patient_analysis",
                "requirements": ["HIPAA compliant", "detailed reporting"]
            }
        )
        
        with patch.object(agent, '_generate_templates', new_callable=AsyncMock) as mock_templates:
            mock_templates.return_value = "Templates generated successfully"
            
            response = await agent.process_request(request)
            
            assert response.status == ResponseStatus.SUCCESS
            assert "Templates generated" in response.content
            mock_templates.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_process_request_error_handling(self, agent, sample_request):
        """Test error handling in request processing."""
        with patch.object(agent, '_optimize_prompt', side_effect=Exception("Test error")):
            response = await agent.process_request(sample_request)
            
            assert response.status == ResponseStatus.ERROR
            assert "Error in prompt engineering" in response.content
            assert response.error_message == "Test error"
    
    @pytest.mark.asyncio
    async def test_generate_ab_variations(self, agent):
        """Test A/B variation generation."""
        original_prompt = "Analyze the sales data"
        
        with patch.object(agent, '_call_gpt4', new_callable=AsyncMock) as mock_gpt4:
            mock_gpt4.return_value = """
            Analyze the sales data with detailed statistical analysis and insights.
            Please analyze the sales data and provide comprehensive business recommendations.
            Perform thorough sales data analysis with trend identification and forecasting.
            Conduct detailed sales data analysis including seasonal patterns and anomalies.
            Analyze sales data systematically with focus on actionable business insights.
            """
            
            variations = await agent._generate_ab_variations(original_prompt)
            
            assert len(variations) == 5
            assert all(isinstance(var, str) for var in variations)
            assert all(len(var.strip()) > 0 for var in variations)
            mock_gpt4.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_generate_ab_variations_fallback(self, agent):
        """Test A/B variation generation with fallback."""
        original_prompt = "Analyze the data"
        
        with patch.object(agent, '_call_gpt4', side_effect=Exception("API error")):
            variations = await agent._generate_ab_variations(original_prompt)
            
            assert len(variations) == 5
            assert all("analyze" in var.lower() for var in variations)
    
    @pytest.mark.asyncio
    async def test_comprehensive_prompt_analysis(self, agent):
        """Test comprehensive prompt analysis."""
        prompt = "Analyze the quarterly sales data"
        
        with patch.object(agent, '_call_gpt4', new_callable=AsyncMock) as mock_gpt4:
            mock_gpt4.return_value = """
            {
                "quality_score": 7.5,
                "strengths": ["Clear objective", "Specific timeframe"],
                "improvements": ["Add output format", "Include comparison baseline"],
                "detailed_analysis": "The prompt is well-structured but could benefit from more specificity.",
                "recommendations": "Consider adding expected output format and comparison criteria.",
                "variations": [
                    "Analyze quarterly sales data and provide insights in table format",
                    "Compare Q3 sales data with previous quarters and identify trends",
                    "Perform comprehensive quarterly sales analysis with recommendations"
                ]
            }
            """
            
            analysis = await agent._comprehensive_prompt_analysis(prompt)
            
            assert analysis["quality_score"] == 7.5
            assert len(analysis["strengths"]) == 2
            assert len(analysis["improvements"]) == 2
            assert len(analysis["variations"]) == 3
            mock_gpt4.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_comprehensive_prompt_analysis_fallback(self, agent):
        """Test prompt analysis with fallback when JSON parsing fails."""
        prompt = "Test prompt"
        
        with patch.object(agent, '_call_gpt4', new_callable=AsyncMock) as mock_gpt4:
            mock_gpt4.return_value = "Non-JSON response"
            
            analysis = await agent._comprehensive_prompt_analysis(prompt)
            
            assert analysis["quality_score"] == 5.0
            assert "Basic structure present" in analysis["strengths"]
            assert "detailed analysis" in analysis
    
    @pytest.mark.asyncio
    async def test_calculate_performance_score(self, agent):
        """Test performance score calculation."""
        prompt = "Analyze data"
        input_data = "Sample input"
        output = "Detailed analysis with insights"
        
        with patch.object(agent, '_call_gpt4', new_callable=AsyncMock) as mock_gpt4:
            mock_gpt4.return_value = "8.5"
            
            score = await agent._calculate_performance_score(prompt, input_data, output)
            
            assert score == 8.5
            mock_gpt4.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_calculate_performance_score_fallback(self, agent):
        """Test performance score calculation with fallback."""
        prompt = "Test"
        input_data = "Input"
        output = "Output"
        
        with patch.object(agent, '_call_gpt4', side_effect=Exception("API error")):
            score = await agent._calculate_performance_score(prompt, input_data, output)
            
            assert score == 5.0  # Default fallback score
    
    @pytest.mark.asyncio
    async def test_run_prompt_tests(self, agent):
        """Test running prompt tests on variations."""
        variations = [
            PromptVariation(
                id="var1",
                original_prompt="Original",
                variation="Variation 1",
                strategy=PromptOptimizationStrategy.A_B_TESTING
            ),
            PromptVariation(
                id="var2",
                original_prompt="Original",
                variation="Variation 2",
                strategy=PromptOptimizationStrategy.A_B_TESTING
            )
        ]
        test_cases = ["Test case 1", "Test case 2"]
        
        with patch.object(agent, '_call_gpt4', new_callable=AsyncMock) as mock_gpt4:
            mock_gpt4.return_value = "Test response"
            with patch.object(agent, '_calculate_performance_score', new_callable=AsyncMock) as mock_score:
                mock_score.return_value = 8.0
                
                results = await agent._run_prompt_tests(variations, test_cases)
                
                assert len(results) == 4  # 2 variations × 2 test cases
                assert all(isinstance(result, PromptTestResult) for result in results)
                assert all(result.success for result in results)
                assert mock_gpt4.call_count == 4
                assert mock_score.call_count == 4
    
    def test_select_best_variation(self, agent):
        """Test selecting the best variation from test results."""
        test_results = [
            PromptTestResult(
                prompt_id="var1",
                variation_id="var1",
                input_data="Test",
                output_data="Output",
                performance_score=8.5,
                response_time=1.0,
                success=True
            ),
            PromptTestResult(
                prompt_id="var2",
                variation_id="var2",
                input_data="Test",
                output_data="Output",
                performance_score=7.2,
                response_time=1.5,
                success=True
            )
        ]
        
        best = agent._select_best_variation(test_results, "performance_score")
        
        assert best.id == "var1"
        assert best.performance_score == 8.5
    
    def test_select_best_variation_empty_results(self, agent):
        """Test selecting best variation with empty results."""
        best = agent._select_best_variation([], "performance_score")
        
        assert best.id == "default"
        assert best.variation == "No variations tested"
    
    @pytest.mark.asyncio
    async def test_ai_evaluate_variations(self, agent):
        """Test AI evaluation of variations."""
        variations = [
            PromptVariation(
                id="var1",
                original_prompt="Original",
                variation="Variation 1",
                strategy=PromptOptimizationStrategy.A_B_TESTING
            ),
            PromptVariation(
                id="var2",
                original_prompt="Original",
                variation="Variation 2",
                strategy=PromptOptimizationStrategy.A_B_TESTING
            )
        ]
        
        with patch.object(agent, '_call_gpt4', new_callable=AsyncMock) as mock_gpt4:
            mock_gpt4.return_value = "2"  # Select second variation
            
            best = await agent._ai_evaluate_variations(variations, "Original prompt")
            
            assert best.id == "var2"
            assert best.performance_score == 8.0
            mock_gpt4.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_ai_evaluate_variations_fallback(self, agent):
        """Test AI evaluation with fallback."""
        variations = [
            PromptVariation(
                id="var1",
                original_prompt="Original",
                variation="Variation 1",
                strategy=PromptOptimizationStrategy.A_B_TESTING
            )
        ]
        
        with patch.object(agent, '_call_gpt4', side_effect=Exception("API error")):
            best = await agent._ai_evaluate_variations(variations, "Original")
            
            assert best.id == "var1"
            assert best.performance_score == 6.0  # Fallback score
    
    def test_categorize_use_case(self, agent):
        """Test use case categorization."""
        assert agent._categorize_use_case("data analysis") == PromptCategory.DATA_ANALYSIS
        assert agent._categorize_use_case("code generation") == PromptCategory.CODE_GENERATION
        assert agent._categorize_use_case("business intelligence") == PromptCategory.BUSINESS_INTELLIGENCE
        assert agent._categorize_use_case("strategic planning") == PromptCategory.STRATEGIC_PLANNING
        assert agent._categorize_use_case("creative writing") == PromptCategory.CREATIVE_WRITING
        assert agent._categorize_use_case("technical documentation") == PromptCategory.TECHNICAL_DOCUMENTATION
        assert agent._categorize_use_case("customer service") == PromptCategory.CUSTOMER_SERVICE
        assert agent._categorize_use_case("research analysis") == PromptCategory.RESEARCH_ANALYSIS
        assert agent._categorize_use_case("unknown") == PromptCategory.DATA_ANALYSIS  # Default
    
    @pytest.mark.asyncio
    async def test_generate_semantic_variations(self, agent):
        """Test semantic variation generation."""
        original_prompt = "Analyze the data"
        
        with patch.object(agent, '_call_gpt4', new_callable=AsyncMock) as mock_gpt4:
            mock_gpt4.return_value = """
            Examine the information
            Study the dataset
            Review the data
            Investigate the information
            Assess the data
            """
            
            variations = await agent._generate_semantic_variations(original_prompt)
            
            assert len(variations) == 5
            assert all(isinstance(var, str) for var in variations)
            mock_gpt4.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_generate_performance_variations(self, agent):
        """Test performance-optimized variation generation."""
        original_prompt = "Analyze the data"
        
        with patch.object(agent, '_call_gpt4', new_callable=AsyncMock) as mock_gpt4:
            mock_gpt4.return_value = """
            Please provide a detailed analysis of the data with specific examples
            Acting as an expert, analyze the data and include methodology
            Analyze the data with clear headings and bullet points
            Thoroughly analyze the data with actionable recommendations
            Analyze the data including context and limitations
            """
            
            variations = await agent._generate_performance_variations(original_prompt)
            
            assert len(variations) == 5
            assert all("analyze" in var.lower() for var in variations)
            mock_gpt4.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_generate_prompt_chain(self, agent):
        """Test prompt chain generation."""
        prompt = "Create analysis workflow"
        context = {"complexity": "high", "steps": 5}
        
        with patch.object(agent, '_call_gpt4', new_callable=AsyncMock) as mock_gpt4:
            mock_gpt4.return_value = '{"chain_name": "Analysis Chain", "prompts": []}'
            
            result = await agent._generate_prompt_chain(prompt, context)
            
            assert "Generated prompt chain" in result
            mock_gpt4.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_execute_prompt_chain(self, agent):
        """Test prompt chain execution."""
        chain = PromptChain(
            id="chain1",
            name="Test Chain",
            description="Test chain execution",
            prompts=[
                {"id": "step1", "prompt": "First step: {input}"},
                {"id": "step2", "prompt": "Second step: {result_step1}"}
            ],
            dependencies={"step2": ["step1"]},
            category=PromptCategory.DATA_ANALYSIS
        )
        
        execution_context = {"input": "test data"}
        
        with patch.object(agent, '_call_gpt4', new_callable=AsyncMock) as mock_gpt4:
            mock_gpt4.return_value = "Step completed successfully"
            
            results = await agent._execute_prompt_chain(chain, execution_context)
            
            assert len(results) == 2
            assert all(result["success"] for result in results)
            assert mock_gpt4.call_count == 2
    
    @pytest.mark.asyncio
    async def test_create_custom_templates(self, agent):
        """Test custom template creation."""
        industry = "healthcare"
        use_case = "patient_analysis"
        requirements = ["HIPAA compliant", "detailed reporting"]
        category = PromptCategory.DATA_ANALYSIS
        
        with patch.object(agent, '_call_gpt4', new_callable=AsyncMock) as mock_gpt4:
            mock_gpt4.return_value = """
            Template 1: Analyze patient data {{data}} for healthcare insights
            Template 2: Review medical records {{records}} with compliance focus
            Template 3: Generate healthcare report from {{dataset}} with recommendations
            """
            
            templates = await agent._create_custom_templates(industry, use_case, requirements, category)
            
            assert len(templates) <= 3
            assert all("template" in template for template in templates)
            assert all("variables" in template for template in templates)
            mock_gpt4.assert_called_once()
    
    def test_extract_variables(self, agent):
        """Test variable extraction from templates."""
        template = "Analyze {{data}} and generate {{report}} with {{format}}"
        variables = agent._extract_variables(template)
        
        assert set(variables) == {"data", "report", "format"}
    
    def test_extract_variables_no_variables(self, agent):
        """Test variable extraction with no variables."""
        template = "Simple template without variables"
        variables = agent._extract_variables(template)
        
        assert variables == []
    
    def test_extract_variables_duplicates(self, agent):
        """Test variable extraction with duplicates."""
        template = "Use {{data}} to analyze {{data}} and create {{report}}"
        variables = agent._extract_variables(template)
        
        assert set(variables) == {"data", "report"}
        assert len(variables) == 2  # No duplicates
    
    @pytest.mark.asyncio
    async def test_health_check_success(self, agent):
        """Test successful health check."""
        with patch.object(agent, '_call_gpt4', new_callable=AsyncMock) as mock_gpt4:
            mock_gpt4.return_value = "OK"
            
            is_healthy = await agent.health_check()
            
            assert is_healthy is True
            mock_gpt4.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_health_check_failure(self, agent):
        """Test health check failure."""
        with patch.object(agent, '_call_gpt4', side_effect=Exception("API error")):
            is_healthy = await agent.health_check()
            
            assert is_healthy is False
    
    def test_get_capabilities(self, agent):
        """Test getting agent capabilities."""
        capabilities = agent.get_capabilities()
        
        assert len(capabilities) == 4
        capability_names = [cap.name for cap in capabilities]
        assert "prompt_optimization" in capability_names
        assert "prompt_testing" in capability_names
        assert "prompt_chain_management" in capability_names
        assert "template_generation" in capability_names
    
    @pytest.mark.asyncio
    async def test_store_optimization_results(self, agent):
        """Test storing optimization results."""
        original = "Original prompt"
        variations = [
            PromptVariation(
                id="var1",
                original_prompt=original,
                variation="Optimized prompt",
                strategy=PromptOptimizationStrategy.A_B_TESTING,
                performance_score=8.5,
                success_rate=0.9,
                avg_response_time=1.2
            )
        ]
        best = variations[0]
        
        with patch('scrollintel.models.database.get_db_session') as mock_session:
            mock_db = Mock()
            mock_session.return_value = mock_db
            
            # Should not raise exception
            await agent._store_optimization_results(original, variations, best)
            
            mock_db.add.assert_called_once()
            mock_db.commit.assert_called_once()
            mock_db.close.assert_called_once()


class TestPromptVariation:
    """Test PromptVariation dataclass."""
    
    def test_prompt_variation_creation(self):
        """Test PromptVariation creation."""
        variation = PromptVariation(
            id="test-id",
            original_prompt="Original",
            variation="Variation",
            strategy=PromptOptimizationStrategy.A_B_TESTING
        )
        
        assert variation.id == "test-id"
        assert variation.original_prompt == "Original"
        assert variation.variation == "Variation"
        assert variation.strategy == PromptOptimizationStrategy.A_B_TESTING
        assert variation.performance_score == 0.0
        assert variation.test_count == 0
        assert variation.success_rate == 0.0
        assert variation.avg_response_time == 0.0
        assert isinstance(variation.created_at, datetime)
    
    def test_prompt_variation_with_custom_values(self):
        """Test PromptVariation with custom values."""
        custom_time = datetime(2023, 1, 1)
        variation = PromptVariation(
            id="test-id",
            original_prompt="Original",
            variation="Variation",
            strategy=PromptOptimizationStrategy.GENETIC_ALGORITHM,
            performance_score=8.5,
            test_count=10,
            success_rate=0.9,
            avg_response_time=1.5,
            created_at=custom_time
        )
        
        assert variation.performance_score == 8.5
        assert variation.test_count == 10
        assert variation.success_rate == 0.9
        assert variation.avg_response_time == 1.5
        assert variation.created_at == custom_time


class TestPromptTestResult:
    """Test PromptTestResult dataclass."""
    
    def test_prompt_test_result_creation(self):
        """Test PromptTestResult creation."""
        result = PromptTestResult(
            prompt_id="prompt-1",
            variation_id="var-1",
            input_data="Test input",
            output_data="Test output",
            performance_score=8.0,
            response_time=1.2,
            success=True
        )
        
        assert result.prompt_id == "prompt-1"
        assert result.variation_id == "var-1"
        assert result.input_data == "Test input"
        assert result.output_data == "Test output"
        assert result.performance_score == 8.0
        assert result.response_time == 1.2
        assert result.success is True
        assert result.error_message is None
        assert isinstance(result.timestamp, datetime)
    
    def test_prompt_test_result_with_error(self):
        """Test PromptTestResult with error."""
        result = PromptTestResult(
            prompt_id="prompt-1",
            variation_id="var-1",
            input_data="Test input",
            output_data="",
            performance_score=0.0,
            response_time=0.0,
            success=False,
            error_message="Test error"
        )
        
        assert result.success is False
        assert result.error_message == "Test error"


class TestPromptChain:
    """Test PromptChain dataclass."""
    
    def test_prompt_chain_creation(self):
        """Test PromptChain creation."""
        chain = PromptChain(
            id="chain-1",
            name="Test Chain",
            description="Test chain description",
            prompts=[{"id": "step1", "prompt": "First step"}],
            dependencies={"step2": ["step1"]},
            category=PromptCategory.DATA_ANALYSIS
        )
        
        assert chain.id == "chain-1"
        assert chain.name == "Test Chain"
        assert chain.description == "Test chain description"
        assert len(chain.prompts) == 1
        assert chain.dependencies == {"step2": ["step1"]}
        assert chain.category == PromptCategory.DATA_ANALYSIS
        assert isinstance(chain.created_at, datetime)


class TestPromptOptimizationStrategy:
    """Test PromptOptimizationStrategy enum."""
    
    def test_strategy_values(self):
        """Test strategy enum values."""
        assert PromptOptimizationStrategy.A_B_TESTING.value == "a_b_testing"
        assert PromptOptimizationStrategy.GENETIC_ALGORITHM.value == "genetic_algorithm"
        assert PromptOptimizationStrategy.REINFORCEMENT_LEARNING.value == "reinforcement_learning"
        assert PromptOptimizationStrategy.SEMANTIC_SIMILARITY.value == "semantic_similarity"
        assert PromptOptimizationStrategy.PERFORMANCE_BASED.value == "performance_based"


class TestPromptCategory:
    """Test PromptCategory enum."""
    
    def test_category_values(self):
        """Test category enum values."""
        assert PromptCategory.DATA_ANALYSIS.value == "data_analysis"
        assert PromptCategory.CODE_GENERATION.value == "code_generation"
        assert PromptCategory.CREATIVE_WRITING.value == "creative_writing"
        assert PromptCategory.BUSINESS_INTELLIGENCE.value == "business_intelligence"
        assert PromptCategory.TECHNICAL_DOCUMENTATION.value == "technical_documentation"
        assert PromptCategory.CUSTOMER_SERVICE.value == "customer_service"
        assert PromptCategory.RESEARCH_ANALYSIS.value == "research_analysis"
        assert PromptCategory.STRATEGIC_PLANNING.value == "strategic_planning"


if __name__ == "__main__":
    pytest.main([__file__])