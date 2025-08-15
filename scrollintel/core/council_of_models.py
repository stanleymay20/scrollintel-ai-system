"""
Council of Models orchestration for ScrollIntel-G6.
Implements debate → critique → revise cycles with juror protocols.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import openai
from anthropic import Anthropic
import google.generativeai as genai

from .proof_of_workflow import create_workflow_attestation
from ..core.config import get_config

logger = logging.getLogger(__name__)


class ModelType(Enum):
    SCROLL_CORE_M = "scroll_core_m"
    GPT_5 = "gpt-5"
    CLAUDE_3_5 = "claude-3-5-sonnet"
    GEMINI_PRO = "gemini-pro"
    DEEPSEEK_V3 = "deepseek-v3"
    LLAMA_3_1 = "llama-3.1-405b"


@dataclass
class ModelResponse:
    """Response from a model in the council."""
    model_type: ModelType
    content: str
    confidence: float
    reasoning: str
    metadata: Dict[str, Any]
    timestamp: datetime


@dataclass
class CouncilDecision:
    """Final decision from the council of models."""
    winning_response: ModelResponse
    all_responses: List[ModelResponse]
    debate_rounds: int
    consensus_score: float
    verifier_scores: Dict[str, float]
    final_content: str
    attestation_id: str


class ModelClient:
    """Base class for model clients."""
    
    async def generate_response(self, prompt: str, context: Dict[str, Any]) -> ModelResponse:
        raise NotImplementedError


class ScrollCoreClient(ModelClient):
    """Client for ScrollCore-M model."""
    
    def __init__(self):
        self.model_version = "scroll-core-m-1.0"
    
    async def generate_response(self, prompt: str, context: Dict[str, Any]) -> ModelResponse:
        # This would integrate with the actual ScrollCore model
        # For now, simulate with a high-quality response
        content = f"ScrollCore-M Response: {prompt[:100]}..."
        
        return ModelResponse(
            model_type=ModelType.SCROLL_CORE_M,
            content=content,
            confidence=0.95,
            reasoning="ScrollCore-M analysis with scroll alignment verification",
            metadata={"model_version": self.model_version, "scroll_aligned": True},
            timestamp=datetime.utcnow()
        )


class GPT5Client(ModelClient):
    """Client for GPT-5 model."""
    
    def __init__(self):
        self.client = openai.AsyncOpenAI()
        self.model = "gpt-4"  # Will be updated to gpt-5 when available
    
    async def generate_response(self, prompt: str, context: Dict[str, Any]) -> ModelResponse:
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant participating in a council of models."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=2000
            )
            
            content = response.choices[0].message.content
            
            return ModelResponse(
                model_type=ModelType.GPT_5,
                content=content,
                confidence=0.85,
                reasoning="GPT-5 analysis with general knowledge",
                metadata={"model": self.model, "tokens_used": response.usage.total_tokens},
                timestamp=datetime.utcnow()
            )
        except Exception as e:
            logger.error(f"GPT-5 client error: {e}")
            return ModelResponse(
                model_type=ModelType.GPT_5,
                content=f"Error: {str(e)}",
                confidence=0.0,
                reasoning="Model unavailable",
                metadata={"error": str(e)},
                timestamp=datetime.utcnow()
            )


class ClaudeClient(ModelClient):
    """Client for Claude model."""
    
    def __init__(self):
        config = get_config()
        self.client = Anthropic(api_key=config.anthropic_api_key)
        self.model = "claude-3-5-sonnet-20241022"
    
    async def generate_response(self, prompt: str, context: Dict[str, Any]) -> ModelResponse:
        try:
            response = await asyncio.to_thread(
                self.client.messages.create,
                model=self.model,
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}]
            )
            
            content = response.content[0].text
            
            return ModelResponse(
                model_type=ModelType.CLAUDE_3_5,
                content=content,
                confidence=0.88,
                reasoning="Claude analysis with constitutional AI principles",
                metadata={"model": self.model, "tokens_used": response.usage.input_tokens + response.usage.output_tokens},
                timestamp=datetime.utcnow()
            )
        except Exception as e:
            logger.error(f"Claude client error: {e}")
            return ModelResponse(
                model_type=ModelType.CLAUDE_3_5,
                content=f"Error: {str(e)}",
                confidence=0.0,
                reasoning="Model unavailable",
                metadata={"error": str(e)},
                timestamp=datetime.utcnow()
            )


class JurorVerifier:
    """Lightweight model for scoring candidate responses."""
    
    def __init__(self):
        self.weights = {
            "policy_fit": 0.3,
            "factuality": 0.3,
            "spec_coverage": 0.2,
            "scroll_alignment": 0.2
        }
    
    async def score_response(self, response: ModelResponse, task_context: Dict[str, Any]) -> Dict[str, float]:
        """Score a response on multiple criteria."""
        scores = {}
        
        # Policy fit score
        scores["policy_fit"] = await self._score_policy_fit(response, task_context)
        
        # Factuality score
        scores["factuality"] = await self._score_factuality(response, task_context)
        
        # Spec coverage score
        scores["spec_coverage"] = await self._score_spec_coverage(response, task_context)
        
        # Scroll alignment score
        scores["scroll_alignment"] = await self._score_scroll_alignment(response, task_context)
        
        # Calculate weighted total
        scores["total"] = sum(scores[key] * self.weights[key] for key in self.weights)
        
        return scores
    
    async def _score_policy_fit(self, response: ModelResponse, context: Dict[str, Any]) -> float:
        """Score how well the response fits organizational policies."""
        # Simplified scoring - in production this would use a specialized model
        if "scroll" in response.content.lower():
            return 0.9
        return 0.7
    
    async def _score_factuality(self, response: ModelResponse, context: Dict[str, Any]) -> float:
        """Score the factual accuracy of the response."""
        # Simplified scoring - in production this would use fact-checking
        if response.confidence > 0.8:
            return 0.85
        return 0.6
    
    async def _score_spec_coverage(self, response: ModelResponse, context: Dict[str, Any]) -> float:
        """Score how well the response covers the specification requirements."""
        # Simplified scoring - in production this would check against specs
        if len(response.content) > 100:
            return 0.8
        return 0.5
    
    async def _score_scroll_alignment(self, response: ModelResponse, context: Dict[str, Any]) -> float:
        """Score alignment with scroll doctrine."""
        if response.model_type == ModelType.SCROLL_CORE_M:
            return 1.0
        if response.metadata.get("scroll_aligned", False):
            return 0.9
        return 0.6


class CouncilOfModels:
    """Orchestrates debate between multiple AI models."""
    
    def __init__(self):
        self.clients = {
            ModelType.SCROLL_CORE_M: ScrollCoreClient(),
            ModelType.GPT_5: GPT5Client(),
            ModelType.CLAUDE_3_5: ClaudeClient(),
        }
        self.juror = JurorVerifier()
        self.max_debate_rounds = 3
        self.consensus_threshold = 0.8
    
    async def deliberate(
        self,
        task: str,
        context: Dict[str, Any],
        user_id: str,
        high_risk: bool = False
    ) -> CouncilDecision:
        """Conduct a council deliberation on a high-risk task."""
        
        logger.info(f"Starting council deliberation for task: {task[:50]}...")
        
        # Initial responses from all models
        initial_responses = await self._gather_initial_responses(task, context)
        
        # Debate rounds
        current_responses = initial_responses
        debate_rounds = 0
        
        while debate_rounds < self.max_debate_rounds:
            debate_rounds += 1
            logger.info(f"Council debate round {debate_rounds}")
            
            # Score current responses
            scored_responses = await self._score_responses(current_responses, context)
            
            # Check for consensus
            consensus_score = self._calculate_consensus(scored_responses)
            if consensus_score >= self.consensus_threshold:
                logger.info(f"Consensus reached after {debate_rounds} rounds")
                break
            
            # Generate critique and revision prompts
            critique_prompt = self._generate_critique_prompt(scored_responses, task)
            
            # Get revised responses
            current_responses = await self._gather_revised_responses(critique_prompt, context)
        
        # Final scoring and selection
        final_scored_responses = await self._score_responses(current_responses, context)
        winning_response = max(final_scored_responses, key=lambda x: x[1]["total"])
        
        # Create final decision
        decision = CouncilDecision(
            winning_response=winning_response[0],
            all_responses=current_responses,
            debate_rounds=debate_rounds,
            consensus_score=self._calculate_consensus(final_scored_responses),
            verifier_scores={r[0].model_type.value: r[1]["total"] for r in final_scored_responses},
            final_content=winning_response[0].content,
            attestation_id=""
        )
        
        # Create workflow attestation
        attestation = create_workflow_attestation(
            action_type="council_deliberation",
            agent_id="council_of_models",
            user_id=user_id,
            prompt=task,
            tools_used=["council_of_models", "juror_verifier"],
            datasets_used=[],
            model_version="council-v1.0",
            verifier_evidence={
                "consensus_score": decision.consensus_score,
                "debate_rounds": decision.debate_rounds,
                "verifier_scores": decision.verifier_scores
            },
            content=decision.final_content
        )
        
        decision.attestation_id = attestation.id
        
        logger.info(f"Council deliberation completed. Winner: {winning_response[0].model_type.value}")
        return decision
    
    async def _gather_initial_responses(self, task: str, context: Dict[str, Any]) -> List[ModelResponse]:
        """Gather initial responses from all available models."""
        responses = []
        
        tasks = []
        for model_type, client in self.clients.items():
            tasks.append(client.generate_response(task, context))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, ModelResponse):
                responses.append(result)
            else:
                logger.error(f"Model response error: {result}")
        
        return responses
    
    async def _gather_revised_responses(self, critique_prompt: str, context: Dict[str, Any]) -> List[ModelResponse]:
        """Gather revised responses after critique."""
        return await self._gather_initial_responses(critique_prompt, context)
    
    async def _score_responses(self, responses: List[ModelResponse], context: Dict[str, Any]) -> List[Tuple[ModelResponse, Dict[str, float]]]:
        """Score all responses using the juror verifier."""
        scored_responses = []
        
        for response in responses:
            scores = await self.juror.score_response(response, context)
            scored_responses.append((response, scores))
        
        return scored_responses
    
    def _calculate_consensus(self, scored_responses: List[Tuple[ModelResponse, Dict[str, float]]]) -> float:
        """Calculate consensus score based on response similarity."""
        if len(scored_responses) < 2:
            return 1.0
        
        scores = [score["total"] for _, score in scored_responses]
        mean_score = sum(scores) / len(scores)
        variance = sum((s - mean_score) ** 2 for s in scores) / len(scores)
        
        # Higher consensus when variance is lower
        consensus = max(0.0, 1.0 - variance)
        return consensus
    
    def _generate_critique_prompt(self, scored_responses: List[Tuple[ModelResponse, Dict[str, float]]], original_task: str) -> str:
        """Generate a critique prompt for the next debate round."""
        
        # Find the best and worst responses
        best_response = max(scored_responses, key=lambda x: x[1]["total"])
        worst_response = min(scored_responses, key=lambda x: x[1]["total"])
        
        critique_prompt = f"""
Original task: {original_task}

Previous responses have been evaluated. Here's the analysis:

Best response (score: {best_response[1]['total']:.2f}) from {best_response[0].model_type.value}:
{best_response[0].content[:200]}...

Worst response (score: {worst_response[1]['total']:.2f}) from {worst_response[0].model_type.value}:
{worst_response[0].content[:200]}...

Please provide an improved response that addresses the weaknesses identified and builds on the strengths. Focus on:
1. Better policy alignment
2. Improved factual accuracy
3. More comprehensive spec coverage
4. Stronger scroll doctrine alignment

Improved response:
"""
        
        return critique_prompt


# Global council instance
council = CouncilOfModels()


async def council_deliberation(
    task: str,
    context: Dict[str, Any],
    user_id: str,
    high_risk: bool = False
) -> CouncilDecision:
    """Conduct a council deliberation (convenience function)."""
    return await council.deliberate(task, context, user_id, high_risk)


async def quick_council_vote(
    task: str,
    context: Dict[str, Any],
    user_id: str
) -> str:
    """Get a quick council vote without full deliberation."""
    decision = await council_deliberation(task, context, user_id, high_risk=False)
    return decision.final_content