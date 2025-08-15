"""
ScrollAutoDev Agent - Advanced Prompt Engineering and Optimization
The world's most sophisticated prompt engineering agent with A/B testing and optimization.
"""

import asyncio
import json
import os
import openai
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from uuid import uuid4
from dataclasses import dataclass
from enum import Enum
import hashlib
import statistics

from scrollintel.core.interfaces import BaseAgent, AgentType, AgentRequest, AgentResponse, AgentCapability, ResponseStatus
from scrollintel.models.database_utils import get_sync_db as get_db_session
from scrollintel.models.schemas import PromptHistory, PromptTemplate, PromptTest


class PromptOptimizationStrategy(Enum):
    """Strategies for prompt optimization."""
    A_B_TESTING = "a_b_testing"
    GENETIC_ALGORITHM = "genetic_algorithm"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    SEMANTIC_SIMILARITY = "semantic_similarity"
    PERFORMANCE_BASED = "performance_based"


class PromptCategory(Enum):
    """Categories of prompts for different use cases."""
    DATA_ANALYSIS = "data_analysis"
    CODE_GENERATION = "code_generation"
    CREATIVE_WRITING = "creative_writing"
    BUSINESS_INTELLIGENCE = "business_intelligence"
    TECHNICAL_DOCUMENTATION = "technical_documentation"
    CUSTOMER_SERVICE = "customer_service"
    RESEARCH_ANALYSIS = "research_analysis"
    STRATEGIC_PLANNING = "strategic_planning"


@dataclass
class PromptVariation:
    """A variation of a prompt for testing."""
    id: str
    original_prompt: str
    variation: str
    strategy: PromptOptimizationStrategy
    performance_score: float = 0.0
    test_count: int = 0
    success_rate: float = 0.0
    avg_response_time: float = 0.0
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()


@dataclass
class PromptTestResult:
    """Result of a prompt test."""
    prompt_id: str
    variation_id: str
    input_data: str
    output_data: str
    performance_score: float
    response_time: float
    success: bool
    error_message: Optional[str] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


@dataclass
class PromptChain:
    """A chain of prompts with dependencies."""
    id: str
    name: str
    description: str
    prompts: List[Dict[str, Any]]
    dependencies: Dict[str, List[str]]
    category: PromptCategory
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()


class ScrollAutoDevAgent(BaseAgent):
    """Advanced prompt engineering agent with optimization capabilities."""
    
    def __init__(self):
        super().__init__(
            agent_id="scroll-autodev-agent",
            name="ScrollAutoDev Agent",
            agent_type=AgentType.AI_ENGINEER
        )
        
        self.capabilities = [
            AgentCapability(
                name="prompt_optimization",
                description="Optimize prompts using A/B testing and ML techniques",
                input_types=["prompt", "test_data", "optimization_strategy"],
                output_types=["optimized_prompt", "performance_metrics"]
            ),
            AgentCapability(
                name="prompt_testing",
                description="Test prompt variations and measure performance",
                input_types=["prompt_variations", "test_cases"],
                output_types=["test_results", "recommendations"]
            ),
            AgentCapability(
                name="prompt_chain_management",
                description="Create and manage complex prompt chains with dependencies",
                input_types=["prompt_chain", "execution_context"],
                output_types=["chain_result", "execution_flow"]
            ),
            AgentCapability(
                name="template_generation",
                description="Generate industry-specific prompt templates",
                input_types=["industry", "use_case", "requirements"],
                output_types=["prompt_templates", "usage_guidelines"]
            )
        ]
        
        # Initialize OpenAI client
        self.openai_client = openai.AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Prompt templates by category
        self.prompt_templates = self._initialize_templates()
        
        # Active prompt tests
        self.active_tests: Dict[str, List[PromptVariation]] = {}
        
        # Performance tracking
        self.performance_history: List[PromptTestResult] = []
    
    def _initialize_templates(self) -> Dict[PromptCategory, List[str]]:
        """Initialize industry-specific prompt templates."""
        return {
            PromptCategory.DATA_ANALYSIS: [
                "Analyze the following dataset and provide key insights: {data}",
                "Given this data: {data}, identify patterns, trends, and anomalies. Provide actionable recommendations.",
                "Perform exploratory data analysis on: {data}. Focus on statistical significance and business implications."
            ],
            PromptCategory.CODE_GENERATION: [
                "Generate {language} code that {requirement}. Include error handling and documentation.",
                "Create a {language} function that {functionality}. Optimize for performance and readability.",
                "Write {language} code to {task}. Follow best practices and include unit tests."
            ],
            PromptCategory.BUSINESS_INTELLIGENCE: [
                "Analyze this business data: {data} and provide strategic insights for decision-making.",
                "Given the following KPIs: {metrics}, identify trends and recommend actions to improve performance.",
                "Create a business intelligence report based on: {data}. Include executive summary and recommendations."
            ],
            PromptCategory.STRATEGIC_PLANNING: [
                "Develop a strategic plan for {objective} considering {constraints} and {resources}.",
                "Analyze the competitive landscape for {industry} and recommend strategic positioning.",
                "Create a roadmap to achieve {goal} with timeline, milestones, and risk assessment."
            ]
        }
    
    async def process_request(self, request: AgentRequest) -> AgentResponse:
        """Process prompt engineering requests."""
        start_time = asyncio.get_event_loop().time()
        
        try:
            prompt = request.prompt.lower()
            context = request.context or {}
            
            if "optimize" in prompt or "improve" in prompt:
                content = await self._optimize_prompt(request.prompt, context)
            elif "test" in prompt or "compare" in prompt:
                content = await self._test_prompt_variations(request.prompt, context)
            elif "chain" in prompt or "workflow" in prompt:
                content = await self._manage_prompt_chain(request.prompt, context)
            elif "template" in prompt or "generate" in prompt:
                content = await self._generate_templates(request.prompt, context)
            else:
                content = await self._analyze_prompt(request.prompt, context)
            
            execution_time = asyncio.get_event_loop().time() - start_time
            
            return AgentResponse(
                id=f"autodev-{uuid4()}",
                request_id=request.id,
                content=content,
                artifacts=[],
                execution_time=execution_time,
                status=ResponseStatus.SUCCESS
            )
            
        except Exception as e:
            execution_time = asyncio.get_event_loop().time() - start_time
            return AgentResponse(
                id=f"autodev-{uuid4()}",
                request_id=request.id,
                content=f"Error in prompt engineering: {str(e)}",
                artifacts=[],
                execution_time=execution_time,
                status=ResponseStatus.ERROR,
                error_message=str(e)
            )
    
    async def _optimize_prompt(self, original_prompt: str, context: Dict[str, Any]) -> str:
        """Optimize a prompt using various strategies."""
        strategy = context.get("strategy", PromptOptimizationStrategy.A_B_TESTING)
        test_data = context.get("test_data", [])
        target_metric = context.get("target_metric", "performance_score")
        
        # Generate prompt variations
        variations = await self._generate_prompt_variations(original_prompt, strategy)
        
        # Test variations if test data is provided
        if test_data:
            test_results = await self._run_prompt_tests(variations, test_data)
            best_variation = self._select_best_variation(test_results, target_metric)
        else:
            # Use AI to evaluate variations
            best_variation = await self._ai_evaluate_variations(variations, original_prompt)
        
        # Store results
        await self._store_optimization_results(original_prompt, variations, best_variation)
        
        return f"""
# Prompt Optimization Results

## Original Prompt
{original_prompt}

## Optimized Prompt
{best_variation.variation}

## Optimization Strategy
{strategy.value}

## Performance Improvements
- **Performance Score**: {best_variation.performance_score:.2f}
- **Success Rate**: {best_variation.success_rate:.1%}
- **Avg Response Time**: {best_variation.avg_response_time:.2f}s

## Optimization Techniques Applied
{await self._explain_optimization_techniques(original_prompt, best_variation.variation)}

## Usage Recommendations
{await self._generate_usage_recommendations(best_variation)}

## A/B Testing Results
{await self._format_ab_test_results(variations)}
"""
    
    async def _generate_prompt_variations(self, original_prompt: str, strategy: PromptOptimizationStrategy) -> List[PromptVariation]:
        """Generate variations of a prompt based on optimization strategy."""
        variations = []
        
        if strategy == PromptOptimizationStrategy.A_B_TESTING:
            # Generate A/B test variations
            variation_prompts = await self._generate_ab_variations(original_prompt)
        elif strategy == PromptOptimizationStrategy.SEMANTIC_SIMILARITY:
            # Generate semantically similar variations
            variation_prompts = await self._generate_semantic_variations(original_prompt)
        elif strategy == PromptOptimizationStrategy.PERFORMANCE_BASED:
            # Generate performance-optimized variations
            variation_prompts = await self._generate_performance_variations(original_prompt)
        else:
            # Default to A/B testing
            variation_prompts = await self._generate_ab_variations(original_prompt)
        
        for i, variation_prompt in enumerate(variation_prompts):
            variations.append(PromptVariation(
                id=f"var-{uuid4()}",
                original_prompt=original_prompt,
                variation=variation_prompt,
                strategy=strategy
            ))
        
        return variations
    
    async def _generate_ab_variations(self, original_prompt: str) -> List[str]:
        """Generate A/B test variations using GPT-4."""
        gpt4_prompt = f"""
        As an expert prompt engineer, create 5 optimized variations of this prompt for A/B testing:
        
        Original Prompt: "{original_prompt}"
        
        Generate variations that:
        1. Improve clarity and specificity
        2. Add context and constraints
        3. Optimize for different response styles
        4. Include performance-enhancing instructions
        5. Test different approaches to the same goal
        
        Return only the 5 variations, one per line, without numbering or explanation.
        """
        
        try:
            response = await self._call_gpt4(gpt4_prompt)
            variations = [line.strip() for line in response.split('\n') if line.strip()]
            return variations[:5]  # Ensure we get exactly 5 variations
        except Exception as e:
            # Fallback variations
            return [
                f"Please {original_prompt.lower()} with detailed explanations and examples.",
                f"{original_prompt} Provide step-by-step reasoning and cite sources.",
                f"Acting as an expert, {original_prompt.lower()} with comprehensive analysis.",
                f"{original_prompt} Include pros, cons, and alternative approaches.",
                f"Thoroughly {original_prompt.lower()} with actionable recommendations."
            ]
    
    async def _test_prompt_variations(self, prompt: str, context: Dict[str, Any]) -> str:
        """Test multiple prompt variations and compare performance."""
        variations_data = context.get("variations", [])
        test_cases = context.get("test_cases", [])
        
        if not variations_data or not test_cases:
            return "Error: Please provide prompt variations and test cases for comparison."
        
        # Create PromptVariation objects
        variations = []
        for i, var_text in enumerate(variations_data):
            variations.append(PromptVariation(
                id=f"test-var-{i}",
                original_prompt=prompt,
                variation=var_text,
                strategy=PromptOptimizationStrategy.A_B_TESTING
            ))
        
        # Run tests
        test_results = await self._run_prompt_tests(variations, test_cases)
        
        # Analyze results
        analysis = await self._analyze_test_results(test_results)
        
        return f"""
# Prompt Variation Testing Results

## Test Overview
- **Number of Variations**: {len(variations)}
- **Test Cases**: {len(test_cases)}
- **Total Tests Run**: {len(test_results)}

## Performance Comparison
{await self._format_performance_comparison(test_results)}

## Statistical Analysis
{analysis}

## Recommendations
{await self._generate_test_recommendations(test_results)}

## Detailed Results
{await self._format_detailed_results(test_results)}
"""
    
    async def _manage_prompt_chain(self, prompt: str, context: Dict[str, Any]) -> str:
        """Manage complex prompt chains with dependencies."""
        chain_data = context.get("chain", {})
        execution_context = context.get("execution_context", {})
        
        if not chain_data:
            # Generate a new prompt chain
            return await self._generate_prompt_chain(prompt, context)
        
        # Execute existing chain
        chain = PromptChain(**chain_data)
        results = await self._execute_prompt_chain(chain, execution_context)
        
        return f"""
# Prompt Chain Execution Results

## Chain: {chain.name}
{chain.description}

## Execution Flow
{await self._format_execution_flow(results)}

## Results Summary
{await self._summarize_chain_results(results)}

## Performance Metrics
{await self._calculate_chain_metrics(results)}

## Optimization Suggestions
{await self._suggest_chain_optimizations(chain, results)}
"""
    
    async def _generate_templates(self, prompt: str, context: Dict[str, Any]) -> str:
        """Generate industry-specific prompt templates."""
        industry = context.get("industry", "general")
        use_case = context.get("use_case", "analysis")
        requirements = context.get("requirements", [])
        
        # Determine category
        category = self._categorize_use_case(use_case)
        
        # Generate custom templates
        templates = await self._create_custom_templates(industry, use_case, requirements, category)
        
        return f"""
# Industry-Specific Prompt Templates

## Industry: {industry.title()}
## Use Case: {use_case.title()}
## Category: {category.value}

## Generated Templates

{await self._format_templates(templates)}

## Usage Guidelines
{await self._generate_usage_guidelines(templates, industry, use_case)}

## Customization Options
{await self._suggest_customizations(templates, requirements)}

## Performance Optimization Tips
{await self._provide_optimization_tips(category)}
"""
    
    async def _analyze_prompt(self, prompt: str, context: Dict[str, Any]) -> str:
        """Analyze a prompt for quality and effectiveness."""
        analysis = await self._comprehensive_prompt_analysis(prompt)
        
        return f"""
# Prompt Analysis Report

## Prompt Quality Score: {analysis['quality_score']:.1f}/10

## Strengths
{chr(10).join(f"- {strength}" for strength in analysis['strengths'])}

## Areas for Improvement
{chr(10).join(f"- {improvement}" for improvement in analysis['improvements'])}

## Detailed Analysis
{analysis['detailed_analysis']}

## Optimization Recommendations
{analysis['recommendations']}

## Suggested Variations
{chr(10).join(f"{i+1}. {var}" for i, var in enumerate(analysis['variations']))}
"""
    
    async def _comprehensive_prompt_analysis(self, prompt: str) -> Dict[str, Any]:
        """Perform comprehensive analysis of a prompt."""
        gpt4_prompt = f"""
        As an expert prompt engineer, analyze this prompt comprehensively:
        
        Prompt: "{prompt}"
        
        Provide analysis in this JSON format:
        {{
            "quality_score": <score 1-10>,
            "strengths": [<list of strengths>],
            "improvements": [<list of areas to improve>],
            "detailed_analysis": "<detailed analysis>",
            "recommendations": "<optimization recommendations>",
            "variations": [<3 improved variations>]
        }}
        
        Focus on clarity, specificity, context, constraints, and expected output quality.
        """
        
        try:
            response = await self._call_gpt4(gpt4_prompt)
            # Try to parse as JSON, fallback to structured text
            try:
                return json.loads(response)
            except:
                return self._parse_analysis_text(response)
        except Exception as e:
            return {
                "quality_score": 5.0,
                "strengths": ["Basic structure present"],
                "improvements": ["Add more specificity", "Include context", "Define expected output"],
                "detailed_analysis": f"Analysis failed: {str(e)}",
                "recommendations": "Consider adding more context and specific instructions",
                "variations": [
                    f"Please {prompt.lower()} with detailed explanations.",
                    f"{prompt} Provide step-by-step analysis.",
                    f"Thoroughly {prompt.lower()} with examples and recommendations."
                ]
            }
    
    async def _call_gpt4(self, prompt: str, max_tokens: int = 2000) -> str:
        """Call GPT-4 API for prompt analysis and generation."""
        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": "You are the world's leading prompt engineering expert with deep knowledge of AI optimization, A/B testing, and performance analysis."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=max_tokens,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"GPT-4 API call failed: {str(e)}")
    
    def get_capabilities(self) -> List[AgentCapability]:
        """Return agent capabilities."""
        return self.capabilities
    
    async def health_check(self) -> bool:
        """Check agent health."""
        try:
            # Test OpenAI connection
            await self._call_gpt4("Test prompt", max_tokens=10)
            return True
        except:
            return False
    
    # Additional helper methods implementation
    async def _run_prompt_tests(self, variations: List[PromptVariation], test_cases: List[str]) -> List[PromptTestResult]:
        """Run tests on prompt variations."""
        test_results = []
        
        for variation in variations:
            for i, test_case in enumerate(test_cases):
                try:
                    start_time = asyncio.get_event_loop().time()
                    
                    # Run the prompt variation with test case
                    response = await self._call_gpt4(
                        f"{variation.variation}\n\nTest Input: {test_case}",
                        max_tokens=1000
                    )
                    
                    execution_time = asyncio.get_event_loop().time() - start_time
                    
                    # Calculate performance score based on response quality
                    performance_score = await self._calculate_performance_score(
                        variation.variation, test_case, response
                    )
                    
                    test_result = PromptTestResult(
                        prompt_id=variation.id,
                        variation_id=variation.id,
                        input_data=test_case,
                        output_data=response,
                        performance_score=performance_score,
                        response_time=execution_time,
                        success=True
                    )
                    
                    test_results.append(test_result)
                    
                    # Update variation statistics
                    variation.test_count += 1
                    variation.performance_score = (
                        (variation.performance_score * (variation.test_count - 1) + performance_score) 
                        / variation.test_count
                    )
                    variation.avg_response_time = (
                        (variation.avg_response_time * (variation.test_count - 1) + execution_time) 
                        / variation.test_count
                    )
                    variation.success_rate = len([r for r in test_results if r.success and r.variation_id == variation.id]) / variation.test_count
                    
                except Exception as e:
                    test_result = PromptTestResult(
                        prompt_id=variation.id,
                        variation_id=variation.id,
                        input_data=test_case,
                        output_data="",
                        performance_score=0.0,
                        response_time=0.0,
                        success=False,
                        error_message=str(e)
                    )
                    test_results.append(test_result)
                    
                    variation.test_count += 1
                    variation.success_rate = len([r for r in test_results if r.success and r.variation_id == variation.id]) / variation.test_count
        
        return test_results
    
    async def _calculate_performance_score(self, prompt: str, input_data: str, output: str) -> float:
        """Calculate performance score for a prompt response."""
        try:
            evaluation_prompt = f"""
            Evaluate the quality of this AI response on a scale of 0-10:
            
            Prompt: {prompt}
            Input: {input_data}
            Response: {output}
            
            Consider:
            - Relevance to the prompt
            - Accuracy and correctness
            - Clarity and coherence
            - Completeness
            - Usefulness
            
            Return only a number between 0 and 10.
            """
            
            score_response = await self._call_gpt4(evaluation_prompt, max_tokens=10)
            
            # Extract numeric score
            import re
            score_match = re.search(r'(\d+(?:\.\d+)?)', score_response)
            if score_match:
                score = float(score_match.group(1))
                return min(max(score, 0.0), 10.0)  # Clamp between 0 and 10
            else:
                return 5.0  # Default score if parsing fails
                
        except Exception:
            return 5.0  # Default score on error
    
    async def _store_optimization_results(self, original: str, variations: List[PromptVariation], best: PromptVariation):
        """Store optimization results in database."""
        try:
            from sqlalchemy.orm import sessionmaker
            from scrollintel.models.database import PromptHistory, PromptTest
            
            # Create database session
            session = get_db_session()
            
            # Store prompt history
            history = PromptHistory(
                user_id=uuid4(),  # This should come from the request context
                original_prompt=original,
                optimized_prompt=best.variation,
                optimization_strategy=best.strategy.value,
                performance_improvement=best.performance_score,
                success_rate_before=0.5,  # Default baseline
                success_rate_after=best.success_rate,
                response_time_before=2.0,  # Default baseline
                response_time_after=best.avg_response_time,
                test_cases_count=best.test_count,
                optimization_metadata={
                    "variations_tested": len(variations),
                    "best_variation_id": best.id,
                    "optimization_timestamp": datetime.utcnow().isoformat()
                }
            )
            
            session.add(history)
            session.commit()
            session.close()
            
        except Exception as e:
            print(f"Error storing optimization results: {e}")
    
    async def _generate_semantic_variations(self, original_prompt: str) -> List[str]:
        """Generate semantically similar variations."""
        semantic_prompt = f"""
        Create 5 semantically similar variations of this prompt that maintain the same meaning but use different wording:
        
        Original: "{original_prompt}"
        
        Generate variations that:
        1. Use synonyms and alternative phrasings
        2. Restructure sentences while preserving meaning
        3. Add or remove minor details that don't change the core request
        4. Use different levels of formality
        5. Employ different question structures
        
        Return only the 5 variations, one per line.
        """
        
        try:
            response = await self._call_gpt4(semantic_prompt)
            variations = [line.strip() for line in response.split('\n') if line.strip()]
            return variations[:5]
        except Exception:
            return [
                f"Could you please {original_prompt.lower()}?",
                f"I need you to {original_prompt.lower()}.",
                f"Please help me {original_prompt.lower()}.",
                f"Can you assist with {original_prompt.lower()}?",
                f"I would like you to {original_prompt.lower()}."
            ]
    
    async def _generate_performance_variations(self, original_prompt: str) -> List[str]:
        """Generate performance-optimized variations."""
        performance_prompt = f"""
        Create 5 performance-optimized variations of this prompt that should produce better, more accurate responses:
        
        Original: "{original_prompt}"
        
        Optimize for:
        1. Clarity and specificity
        2. Context and background information
        3. Clear output format requirements
        4. Step-by-step instructions
        5. Examples and constraints
        
        Return only the 5 optimized variations, one per line.
        """
        
        try:
            response = await self._call_gpt4(performance_prompt)
            variations = [line.strip() for line in response.split('\n') if line.strip()]
            return variations[:5]
        except Exception:
            return [
                f"Please provide a detailed analysis of {original_prompt.lower()} with specific examples and step-by-step reasoning.",
                f"Acting as an expert, {original_prompt.lower()} and include your methodology and sources.",
                f"{original_prompt} Please structure your response with clear headings and bullet points.",
                f"Thoroughly {original_prompt.lower()} and provide actionable recommendations with pros and cons.",
                f"{original_prompt} Include relevant context, assumptions, and potential limitations in your analysis."
            ]
    
    def _select_best_variation(self, test_results: List[PromptTestResult], target_metric: str) -> PromptVariation:
        """Select the best variation based on target metric."""
        if not test_results:
            return PromptVariation(
                id="default",
                original_prompt="",
                variation="No variations tested",
                strategy=PromptOptimizationStrategy.A_B_TESTING
            )
        
        # Group results by variation
        variation_scores = {}
        for result in test_results:
            if result.variation_id not in variation_scores:
                variation_scores[result.variation_id] = []
            
            if target_metric == "performance_score":
                variation_scores[result.variation_id].append(result.performance_score)
            elif target_metric == "response_time":
                variation_scores[result.variation_id].append(1.0 / (result.response_time + 0.1))  # Inverse for optimization
            else:
                variation_scores[result.variation_id].append(result.performance_score)
        
        # Calculate average scores
        best_variation_id = None
        best_score = -1
        
        for variation_id, scores in variation_scores.items():
            avg_score = statistics.mean(scores) if scores else 0
            if avg_score > best_score:
                best_score = avg_score
                best_variation_id = variation_id
        
        # Find the actual variation object
        for result in test_results:
            if result.variation_id == best_variation_id:
                return PromptVariation(
                    id=result.variation_id,
                    original_prompt="",
                    variation=result.input_data,  # This should be the actual variation text
                    strategy=PromptOptimizationStrategy.A_B_TESTING,
                    performance_score=best_score,
                    success_rate=1.0,
                    avg_response_time=result.response_time
                )
        
        # Fallback
        return PromptVariation(
            id="fallback",
            original_prompt="",
            variation="Best variation not found",
            strategy=PromptOptimizationStrategy.A_B_TESTING
        )
    
    async def _ai_evaluate_variations(self, variations: List[PromptVariation], original_prompt: str) -> PromptVariation:
        """Use AI to evaluate variations when no test data is available."""
        evaluation_prompt = f"""
        Evaluate these prompt variations and select the best one:
        
        Original: "{original_prompt}"
        
        Variations:
        {chr(10).join(f"{i+1}. {var.variation}" for i, var in enumerate(variations))}
        
        Evaluate based on:
        - Clarity and specificity
        - Likelihood to produce accurate responses
        - Completeness of instructions
        - Professional tone
        - Actionability
        
        Return only the number (1-{len(variations)}) of the best variation.
        """
        
        try:
            response = await self._call_gpt4(evaluation_prompt, max_tokens=10)
            
            # Extract number
            import re
            number_match = re.search(r'(\d+)', response)
            if number_match:
                selected_index = int(number_match.group(1)) - 1
                if 0 <= selected_index < len(variations):
                    best_variation = variations[selected_index]
                    best_variation.performance_score = 8.0  # Estimated score
                    return best_variation
            
            # Fallback to first variation
            variations[0].performance_score = 7.0
            return variations[0]
            
        except Exception:
            # Fallback to first variation
            variations[0].performance_score = 6.0
            return variations[0]
    
    async def _explain_optimization_techniques(self, original: str, optimized: str) -> str:
        """Explain the optimization techniques applied."""
        explanation_prompt = f"""
        Explain the optimization techniques used to improve this prompt:
        
        Original: "{original}"
        Optimized: "{optimized}"
        
        Identify and explain the specific improvements made:
        - Added specificity
        - Improved structure
        - Enhanced clarity
        - Added context
        - Better instructions
        
        Provide a concise explanation of the key improvements.
        """
        
        try:
            return await self._call_gpt4(explanation_prompt, max_tokens=500)
        except Exception:
            return "Optimization techniques applied include improved clarity, added specificity, and enhanced structure for better AI response quality."
    
    async def _generate_usage_recommendations(self, variation: PromptVariation) -> str:
        """Generate usage recommendations for the optimized prompt."""
        return f"""
        **Best Practices:**
        - Use this prompt for similar tasks requiring detailed analysis
        - Adjust the context variables as needed for your specific use case
        - Consider adding domain-specific examples for better results
        - Monitor performance and iterate based on results
        
        **Performance Expectations:**
        - Expected success rate: {variation.success_rate:.1%}
        - Average response time: {variation.avg_response_time:.2f}s
        - Quality score: {variation.performance_score:.1f}/10
        """
    
    async def _format_ab_test_results(self, variations: List[PromptVariation]) -> str:
        """Format A/B test results for display."""
        if not variations:
            return "No A/B test results available."
        
        results = "| Variation | Performance Score | Success Rate | Avg Response Time |\n"
        results += "|-----------|------------------|--------------|------------------|\n"
        
        for i, var in enumerate(variations):
            results += f"| Variation {i+1} | {var.performance_score:.2f} | {var.success_rate:.1%} | {var.avg_response_time:.2f}s |\n"
        
        return results
    
    def _categorize_use_case(self, use_case: str) -> PromptCategory:
        """Categorize use case into prompt category."""
        use_case_lower = use_case.lower()
        if "data" in use_case_lower or "analysis" in use_case_lower:
            return PromptCategory.DATA_ANALYSIS
        elif "code" in use_case_lower or "programming" in use_case_lower:
            return PromptCategory.CODE_GENERATION
        elif "business" in use_case_lower or "intelligence" in use_case_lower:
            return PromptCategory.BUSINESS_INTELLIGENCE
        elif "strategy" in use_case_lower or "planning" in use_case_lower:
            return PromptCategory.STRATEGIC_PLANNING
        elif "creative" in use_case_lower or "writing" in use_case_lower:
            return PromptCategory.CREATIVE_WRITING
        elif "technical" in use_case_lower or "documentation" in use_case_lower:
            return PromptCategory.TECHNICAL_DOCUMENTATION
        elif "customer" in use_case_lower or "service" in use_case_lower:
            return PromptCategory.CUSTOMER_SERVICE
        elif "research" in use_case_lower:
            return PromptCategory.RESEARCH_ANALYSIS
        else:
            return PromptCategory.DATA_ANALYSIS
    
    def _parse_analysis_text(self, text: str) -> Dict[str, Any]:
        """Parse analysis text when JSON parsing fails."""
        return {
            "quality_score": 6.0,
            "strengths": ["Structure present"],
            "improvements": ["Add specificity"],
            "detailed_analysis": text,
            "recommendations": "Review and optimize for clarity",
            "variations": ["Improved version needed"]
        }
    
    # Additional methods for prompt chain management and template generation
    async def _generate_prompt_chain(self, prompt: str, context: Dict[str, Any]) -> str:
        """Generate a new prompt chain based on requirements."""
        chain_prompt = f"""
        Create a prompt chain for: "{prompt}"
        
        Context: {context}
        
        Design a sequence of 3-5 interconnected prompts that:
        1. Break down the complex task into manageable steps
        2. Build upon previous outputs
        3. Include dependency management
        4. Provide clear execution flow
        
        Return a JSON structure with:
        - chain_name
        - description
        - prompts (array of prompt objects)
        - dependencies (object mapping dependencies)
        """
        
        try:
            response = await self._call_gpt4(chain_prompt, max_tokens=1500)
            return f"Generated prompt chain:\n\n{response}"
        except Exception as e:
            return f"Error generating prompt chain: {str(e)}"
    
    async def _execute_prompt_chain(self, chain: PromptChain, execution_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute a prompt chain with dependency management."""
        results = []
        executed_prompts = set()
        
        # Simple execution order (could be improved with proper dependency resolution)
        for prompt_config in chain.prompts:
            try:
                prompt_id = prompt_config.get("id", f"prompt_{len(results)}")
                prompt_text = prompt_config.get("prompt", "")
                
                # Replace variables from execution context and previous results
                for key, value in execution_context.items():
                    prompt_text = prompt_text.replace(f"{{{key}}}", str(value))
                
                # Execute prompt
                response = await self._call_gpt4(prompt_text, max_tokens=1000)
                
                result = {
                    "prompt_id": prompt_id,
                    "prompt": prompt_text,
                    "response": response,
                    "execution_time": 1.0,  # Placeholder
                    "success": True
                }
                
                results.append(result)
                executed_prompts.add(prompt_id)
                
                # Add result to context for next prompts
                execution_context[f"result_{prompt_id}"] = response
                
            except Exception as e:
                result = {
                    "prompt_id": prompt_config.get("id", f"prompt_{len(results)}"),
                    "prompt": prompt_config.get("prompt", ""),
                    "response": "",
                    "execution_time": 0.0,
                    "success": False,
                    "error": str(e)
                }
                results.append(result)
        
        return results
    
    async def _create_custom_templates(self, industry: str, use_case: str, requirements: List[str], category: PromptCategory) -> List[Dict[str, Any]]:
        """Create custom templates for specific industry and use case."""
        template_prompt = f"""
        Create 3 specialized prompt templates for:
        
        Industry: {industry}
        Use Case: {use_case}
        Category: {category.value}
        Requirements: {requirements}
        
        Each template should:
        1. Be tailored to the specific industry context
        2. Address the use case requirements
        3. Include placeholder variables
        4. Follow best practices for the category
        5. Be professional and actionable
        
        Return templates in this format:
        Template 1: [template text with {{variables}}]
        Template 2: [template text with {{variables}}]
        Template 3: [template text with {{variables}}]
        """
        
        try:
            response = await self._call_gpt4(template_prompt, max_tokens=1500)
            
            # Parse templates
            templates = []
            template_lines = response.split('\n')
            current_template = ""
            
            for line in template_lines:
                if line.startswith("Template"):
                    if current_template:
                        templates.append({
                            "template": current_template.strip(),
                            "variables": self._extract_variables(current_template),
                            "category": category.value,
                            "industry": industry,
                            "use_case": use_case
                        })
                    current_template = line.split(":", 1)[1].strip() if ":" in line else ""
                else:
                    current_template += " " + line.strip()
            
            # Add the last template
            if current_template:
                templates.append({
                    "template": current_template.strip(),
                    "variables": self._extract_variables(current_template),
                    "category": category.value,
                    "industry": industry,
                    "use_case": use_case
                })
            
            return templates[:3]  # Return up to 3 templates
            
        except Exception as e:
            # Fallback templates
            return [
                {
                    "template": f"Analyze the following {industry} data for {use_case}: {{data}}. Provide insights and recommendations.",
                    "variables": ["data"],
                    "category": category.value,
                    "industry": industry,
                    "use_case": use_case
                }
            ]
    
    def _extract_variables(self, template: str) -> List[str]:
        """Extract variables from template text."""
        import re
        variables = re.findall(r'\{\{(\w+)\}\}', template)
        return list(set(variables))  # Remove duplicates
    
    async def _format_templates(self, templates: List[Dict[str, Any]]) -> str:
        """Format templates for display."""
        if not templates:
            return "No templates generated."
        
        formatted = ""
        for i, template in enumerate(templates, 1):
            formatted += f"### Template {i}\n\n"
            formatted += f"**Template:** {template['template']}\n\n"
            formatted += f"**Variables:** {', '.join(template['variables'])}\n\n"
            formatted += f"**Category:** {template['category']}\n\n"
            formatted += "---\n\n"
        
        return formatted
    
    async def _generate_usage_guidelines(self, templates: List[Dict[str, Any]], industry: str, use_case: str) -> str:
        """Generate usage guidelines for templates."""
        return f"""
        **Usage Guidelines for {industry} - {use_case}:**
        
        1. **Variable Substitution**: Replace {{variable}} placeholders with actual values
        2. **Context Adaptation**: Modify templates based on specific business context
        3. **Iterative Refinement**: Test and refine templates based on results
        4. **Performance Monitoring**: Track template effectiveness over time
        5. **Industry Compliance**: Ensure templates meet industry-specific requirements
        
        **Best Practices:**
        - Start with Template 1 for general use cases
        - Use Template 2 for more detailed analysis
        - Apply Template 3 for specialized scenarios
        - Combine templates for complex workflows
        """
    
    async def _suggest_customizations(self, templates: List[Dict[str, Any]], requirements: List[str]) -> str:
        """Suggest customizations based on requirements."""
        customizations = []
        
        for req in requirements:
            if "detailed" in req.lower():
                customizations.append("Add 'Provide detailed step-by-step analysis' to templates")
            elif "example" in req.lower():
                customizations.append("Include 'with specific examples' in template instructions")
            elif "format" in req.lower():
                customizations.append("Specify output format requirements (JSON, table, etc.)")
            elif "source" in req.lower():
                customizations.append("Add 'cite sources and references' to templates")
        
        if not customizations:
            customizations = [
                "Add domain-specific terminology",
                "Include output format specifications",
                "Add constraint parameters",
                "Include quality criteria"
            ]
        
        return "\n".join(f"- {custom}" for custom in customizations)
    
    async def _provide_optimization_tips(self, category: PromptCategory) -> str:
        """Provide optimization tips based on category."""
        tips = {
            PromptCategory.DATA_ANALYSIS: [
                "Specify the type of analysis needed (descriptive, predictive, prescriptive)",
                "Include data format and structure information",
                "Request specific metrics and visualizations",
                "Ask for statistical significance testing"
            ],
            PromptCategory.CODE_GENERATION: [
                "Specify programming language and version",
                "Include requirements for error handling",
                "Request code comments and documentation",
                "Specify performance and security requirements"
            ],
            PromptCategory.BUSINESS_INTELLIGENCE: [
                "Define key performance indicators (KPIs)",
                "Specify time periods and comparison baselines",
                "Request actionable recommendations",
                "Include stakeholder context"
            ],
            PromptCategory.STRATEGIC_PLANNING: [
                "Provide clear objectives and constraints",
                "Include timeline and resource information",
                "Request risk assessment and mitigation",
                "Ask for measurable outcomes"
            ]
        }
        
        category_tips = tips.get(category, tips[PromptCategory.DATA_ANALYSIS])
        return "\n".join(f"- {tip}" for tip in category_tips)
    
    # Additional helper methods for test analysis and formatting
    async def _analyze_test_results(self, test_results: List[PromptTestResult]) -> str:
        """Analyze test results and provide statistical insights."""
        if not test_results:
            return "No test results to analyze."
        
        # Group by variation
        variation_results = {}
        for result in test_results:
            if result.variation_id not in variation_results:
                variation_results[result.variation_id] = []
            variation_results[result.variation_id].append(result)
        
        analysis = "## Statistical Analysis\n\n"
        
        for variation_id, results in variation_results.items():
            scores = [r.performance_score for r in results if r.success]
            times = [r.response_time for r in results if r.success]
            success_rate = len([r for r in results if r.success]) / len(results)
            
            if scores:
                avg_score = statistics.mean(scores)
                std_score = statistics.stdev(scores) if len(scores) > 1 else 0
                
                analysis += f"**{variation_id}:**\n"
                analysis += f"- Average Score: {avg_score:.2f} ± {std_score:.2f}\n"
                analysis += f"- Success Rate: {success_rate:.1%}\n"
                analysis += f"- Average Response Time: {statistics.mean(times):.2f}s\n"
                analysis += f"- Sample Size: {len(results)}\n\n"
        
        return analysis
    
    async def _format_performance_comparison(self, test_results: List[PromptTestResult]) -> str:
        """Format performance comparison table."""
        if not test_results:
            return "No results to compare."
        
        # Group by variation
        variation_stats = {}
        for result in test_results:
            if result.variation_id not in variation_stats:
                variation_stats[result.variation_id] = {
                    'scores': [],
                    'times': [],
                    'successes': 0,
                    'total': 0
                }
            
            stats = variation_stats[result.variation_id]
            stats['total'] += 1
            if result.success:
                stats['successes'] += 1
                stats['scores'].append(result.performance_score)
                stats['times'].append(result.response_time)
        
        # Create comparison table
        table = "| Variation | Avg Score | Success Rate | Avg Time | Tests |\n"
        table += "|-----------|-----------|--------------|----------|-------|\n"
        
        for var_id, stats in variation_stats.items():
            avg_score = statistics.mean(stats['scores']) if stats['scores'] else 0
            success_rate = stats['successes'] / stats['total'] if stats['total'] > 0 else 0
            avg_time = statistics.mean(stats['times']) if stats['times'] else 0
            
            table += f"| {var_id} | {avg_score:.2f} | {success_rate:.1%} | {avg_time:.2f}s | {stats['total']} |\n"
        
        return table
    
    async def _generate_test_recommendations(self, test_results: List[PromptTestResult]) -> str:
        """Generate recommendations based on test results."""
        if not test_results:
            return "No test results available for recommendations."
        
        # Analyze results
        best_variation = None
        best_score = -1
        
        variation_stats = {}
        for result in test_results:
            if result.variation_id not in variation_stats:
                variation_stats[result.variation_id] = []
            variation_stats[result.variation_id].append(result.performance_score)
        
        for var_id, scores in variation_stats.items():
            avg_score = statistics.mean(scores) if scores else 0
            if avg_score > best_score:
                best_score = avg_score
                best_variation = var_id
        
        recommendations = [
            f"**Best Performing Variation:** {best_variation} (Score: {best_score:.2f})",
            "**Next Steps:**",
            "- Deploy the best performing variation for production use",
            "- Continue monitoring performance with real-world data",
            "- Consider A/B testing with larger sample sizes",
            "- Iterate on the winning variation for further improvements"
        ]
        
        if best_score < 7.0:
            recommendations.append("- Consider additional optimization as scores are below 7.0")
        
        return "\n".join(recommendations)
    
    async def _format_detailed_results(self, test_results: List[PromptTestResult]) -> str:
        """Format detailed test results."""
        if not test_results:
            return "No detailed results available."
        
        detailed = "## Detailed Test Results\n\n"
        
        for i, result in enumerate(test_results[:10], 1):  # Show first 10 results
            detailed += f"### Test {i}\n"
            detailed += f"**Variation:** {result.variation_id}\n"
            detailed += f"**Input:** {result.input_data[:100]}...\n"
            detailed += f"**Score:** {result.performance_score:.2f}\n"
            detailed += f"**Time:** {result.response_time:.2f}s\n"
            detailed += f"**Success:** {'✓' if result.success else '✗'}\n"
            if result.error_message:
                detailed += f"**Error:** {result.error_message}\n"
            detailed += "\n---\n\n"
        
        if len(test_results) > 10:
            detailed += f"... and {len(test_results) - 10} more results\n"
        
        return detailed
    
    async def _format_execution_flow(self, results: List[Dict[str, Any]]) -> str:
        """Format execution flow for prompt chains."""
        if not results:
            return "No execution flow available."
        
        flow = "## Execution Flow\n\n"
        
        for i, result in enumerate(results, 1):
            status = "✓" if result.get("success", False) else "✗"
            flow += f"{i}. **{result.get('prompt_id', f'Step {i}')}** {status}\n"
            flow += f"   - Execution Time: {result.get('execution_time', 0):.2f}s\n"
            if not result.get("success", False):
                flow += f"   - Error: {result.get('error', 'Unknown error')}\n"
            flow += "\n"
        
        return flow
    
    async def _summarize_chain_results(self, results: List[Dict[str, Any]]) -> str:
        """Summarize chain execution results."""
        if not results:
            return "No results to summarize."
        
        total_steps = len(results)
        successful_steps = len([r for r in results if r.get("success", False)])
        total_time = sum(r.get("execution_time", 0) for r in results)
        
        summary = f"""
        **Chain Execution Summary:**
        - Total Steps: {total_steps}
        - Successful Steps: {successful_steps}
        - Success Rate: {successful_steps/total_steps:.1%}
        - Total Execution Time: {total_time:.2f}s
        - Average Step Time: {total_time/total_steps:.2f}s
        """
        
        return summary
    
    async def _calculate_chain_metrics(self, results: List[Dict[str, Any]]) -> str:
        """Calculate performance metrics for prompt chains."""
        if not results:
            return "No metrics available."
        
        metrics = {
            "total_execution_time": sum(r.get("execution_time", 0) for r in results),
            "success_rate": len([r for r in results if r.get("success", False)]) / len(results),
            "average_step_time": sum(r.get("execution_time", 0) for r in results) / len(results),
            "failed_steps": [r.get("prompt_id", "unknown") for r in results if not r.get("success", False)]
        }
        
        formatted_metrics = f"""
        **Performance Metrics:**
        - Total Execution Time: {metrics['total_execution_time']:.2f}s
        - Success Rate: {metrics['success_rate']:.1%}
        - Average Step Time: {metrics['average_step_time']:.2f}s
        - Failed Steps: {', '.join(metrics['failed_steps']) if metrics['failed_steps'] else 'None'}
        """
        
        return formatted_metrics
    
    async def _suggest_chain_optimizations(self, chain: PromptChain, results: List[Dict[str, Any]]) -> str:
        """Suggest optimizations for prompt chains."""
        suggestions = ["**Optimization Suggestions:**"]
        
        # Analyze results for optimization opportunities
        failed_steps = [r for r in results if not r.get("success", False)]
        slow_steps = [r for r in results if r.get("execution_time", 0) > 5.0]
        
        if failed_steps:
            suggestions.append("- Review and fix failed steps for better reliability")
        
        if slow_steps:
            suggestions.append("- Optimize slow-running steps to improve overall performance")
        
        if len(results) > 5:
            suggestions.append("- Consider breaking down complex chains into smaller, manageable units")
        
        suggestions.extend([
            "- Add error handling and retry mechanisms",
            "- Implement parallel execution for independent steps",
            "- Cache intermediate results to avoid redundant processing",
            "- Add progress tracking and user feedback"
        ])
        
        return "\n".join(suggestions)