"""
Memory Integration System for AGI Cognitive Architecture

Coordinates working memory, long-term memory, and episodic memory systems
to provide unified memory-guided reasoning and decision-making.
"""

from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
import asyncio
import logging
from dataclasses import dataclass
from enum import Enum

from .working_memory import WorkingMemorySystem
from .long_term_memory import LongTermMemorySystem
from .episodic_memory import EpisodicMemorySystem
from ..models.memory_models import (
    MemoryTrace, MemoryType, EpisodicMemory, MemoryRetrievalQuery,
    MemoryRetrievalResult, MemorySystemMetrics
)

logger = logging.getLogger(__name__)


class MemoryGuidedDecisionType(Enum):
    """Types of memory-guided decisions"""
    PATTERN_RECOGNITION = "pattern_recognition"
    ANALOGICAL_REASONING = "analogical_reasoning"
    CASE_BASED_REASONING = "case_based_reasoning"
    EXPERIENTIAL_LEARNING = "experiential_learning"


@dataclass
class MemoryGuidedInsight:
    """Insight generated from memory integration"""
    insight_type: MemoryGuidedDecisionType
    content: str
    confidence: float
    supporting_memories: List[str]
    episodic_context: Optional[str] = None
    reasoning_path: List[str] = None


@dataclass
class CreativeConnection:
    """Creative connection between disparate memories"""
    memory_ids: List[str]
    connection_type: str
    novelty_score: float
    potential_applications: List[str]
    synthesis_description: str


class MemoryIntegrationSystem:
    """
    Unified memory system integrating working, long-term, and episodic memory.
    
    Provides memory-guided reasoning, creative connections, and decision support
    through coordinated memory operations.
    """
    
    def __init__(self, 
                 working_memory_capacity: int = 7,
                 lt_memory_db: str = "long_term_memory.db",
                 episodic_db: str = "episodic_memory.db"):
        
        # Initialize memory subsystems
        self.working_memory = WorkingMemorySystem(capacity=working_memory_capacity)
        self.long_term_memory = LongTermMemorySystem(db_path=lt_memory_db)
        self.episodic_memory = EpisodicMemorySystem(db_path=episodic_db)
        
        # Integration state
        self._running = False
        self._integration_task: Optional[asyncio.Task] = None
        self._creative_task: Optional[asyncio.Task] = None
        
        # Memory interaction tracking
        self.memory_interactions: List[Tuple[datetime, str, str]] = []
        self.creative_connections: List[CreativeConnection] = []
        
        # Performance metrics
        self.integration_metrics = MemorySystemMetrics()
    
    async def start(self):
        """Start the integrated memory system"""
        if not self._running:
            self._running = True
            
            # Start subsystems
            await self.working_memory.start()
            await self.long_term_memory.start()
            await self.episodic_memory.start()
            
            # Start integration processes
            self._integration_task = asyncio.create_task(self._integration_loop())
            self._creative_task = asyncio.create_task(self._creative_connections_loop())
            
            logger.info("Memory integration system started")
    
    async def stop(self):
        """Stop the integrated memory system"""
        self._running = False
        
        # Cancel integration tasks
        for task in [self._integration_task, self._creative_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Stop subsystems
        await self.working_memory.stop()
        await self.long_term_memory.stop()
        await self.episodic_memory.stop()
        
        logger.info("Memory integration system stopped")
    
    async def store_experience(self, content: Any, 
                             experience_type: str = "general",
                             context: Optional[Dict[str, Any]] = None,
                             importance: float = 0.5,
                             emotional_valence: float = 0.0) -> Dict[str, str]:
        """
        Store experience across all memory systems.
        
        Args:
            content: Experience content
            experience_type: Type of experience
            context: Contextual information
            importance: Importance level
            emotional_valence: Emotional charge
            
        Returns:
            Dictionary with memory IDs from each system
        """
        memory_ids = {}
        
        # Store in working memory for immediate access
        wm_slot = await self.working_memory.store(
            content=content,
            importance=importance
        )
        if wm_slot is not None:
            memory_ids["working_memory"] = f"wm_slot_{wm_slot}"
        
        # Store in long-term memory for persistent access
        lt_memory_id = await self.long_term_memory.store_memory(
            content=content,
            memory_type=MemoryType.SEMANTIC,
            context=context,
            importance=importance
        )
        memory_ids["long_term_memory"] = lt_memory_id
        
        # Store as episodic memory if it's an experience
        if experience_type in ["experience", "learning", "decision", "problem_solving"]:
            episode_id = await self.episodic_memory.store_episode(
                event_description=str(content),
                location=context.get("location") if context else None,
                participants=context.get("participants") if context else None,
                emotions={"valence": emotional_valence} if emotional_valence != 0.0 else None,
                outcome=context.get("outcome") if context else None,
                lessons_learned=context.get("lessons") if context else None
            )
            memory_ids["episodic_memory"] = episode_id
        
        # Track interaction
        self.memory_interactions.append((
            datetime.now(), "store", f"type:{experience_type},importance:{importance}"
        ))
        
        logger.debug(f"Stored experience across memory systems: {memory_ids}")
        return memory_ids
    
    async def retrieve_relevant_memories(self, query: str,
                                       context: Optional[Dict[str, Any]] = None,
                                       max_results: int = 10) -> Dict[str, List[Any]]:
        """
        Retrieve relevant memories from all systems.
        
        Args:
            query: Search query
            context: Additional context for retrieval
            max_results: Maximum results per system
            
        Returns:
            Dictionary with results from each memory system
        """
        results = {}
        
        # Search working memory
        wm_results = self.working_memory.search(query, similarity_threshold=0.3)
        results["working_memory"] = [
            {"slot": slot, "content": content, "similarity": sim}
            for slot, content, sim in wm_results
        ]
        
        # Search long-term memory
        lt_query = MemoryRetrievalQuery()
        lt_query.content_keywords = query.split()
        lt_query.max_results = max_results
        lt_query.include_associations = True
        if context:
            lt_query.context_filters = context
        
        lt_results = await self.long_term_memory.search_memories(lt_query)
        results["long_term_memory"] = [
            {
                "memory": result.memory,
                "relevance": result.relevance_score,
                "associations": result.associated_memories
            }
            for result in lt_results
        ]
        
        # Search episodic memory
        ep_results = await self.episodic_memory.search_episodes(
            query=query,
            location=context.get("location") if context else None,
            max_results=max_results
        )
        results["episodic_memory"] = [
            {
                "episode": episode,
                "relevance": self._calculate_episodic_relevance(episode, query)
            }
            for episode in ep_results
        ]
        
        # Track interaction
        self.memory_interactions.append((
            datetime.now(), "retrieve", f"query:{query[:50]}"
        ))
        
        return results
    
    async def generate_memory_guided_insight(self, problem: str,
                                           context: Optional[Dict[str, Any]] = None) -> MemoryGuidedInsight:
        """
        Generate insight using integrated memory systems.
        
        Args:
            problem: Problem or question to address
            context: Additional context
            
        Returns:
            Memory-guided insight
        """
        # Retrieve relevant memories
        memories = await self.retrieve_relevant_memories(problem, context, max_results=5)
        
        # Analyze patterns across memory systems
        patterns = await self._analyze_memory_patterns(memories, problem)
        
        # Generate insight based on strongest pattern
        if patterns:
            best_pattern = max(patterns, key=lambda p: p["confidence"])
            
            insight = MemoryGuidedInsight(
                insight_type=MemoryGuidedDecisionType(best_pattern["type"]),
                content=best_pattern["insight"],
                confidence=best_pattern["confidence"],
                supporting_memories=best_pattern["memory_ids"],
                reasoning_path=best_pattern["reasoning_steps"]
            )
            
            # Add episodic context if available
            if memories["episodic_memory"]:
                most_relevant_episode = max(
                    memories["episodic_memory"],
                    key=lambda x: x["relevance"]
                )
                insight.episodic_context = most_relevant_episode["episode"].id
            
            return insight
        
        # Fallback insight
        return MemoryGuidedInsight(
            insight_type=MemoryGuidedDecisionType.PATTERN_RECOGNITION,
            content=f"No clear patterns found for: {problem}",
            confidence=0.1,
            supporting_memories=[]
        )
    
    async def create_creative_connections(self, domain: str = "general",
                                        novelty_threshold: float = 0.6) -> List[CreativeConnection]:
        """
        Create creative connections between disparate memories.
        
        Args:
            domain: Domain to focus on
            novelty_threshold: Minimum novelty score
            
        Returns:
            List of creative connections
        """
        connections = []
        
        # Get diverse memories from long-term storage
        diverse_memories = await self._get_diverse_memories(domain, count=20)
        
        # Find unexpected connections
        for i, memory1 in enumerate(diverse_memories):
            for memory2 in diverse_memories[i+1:]:
                connection = await self._evaluate_creative_connection(memory1, memory2)
                
                if connection and connection.novelty_score >= novelty_threshold:
                    connections.append(connection)
        
        # Sort by novelty and potential
        connections.sort(key=lambda c: c.novelty_score, reverse=True)
        
        # Store top connections
        self.creative_connections.extend(connections[:5])
        
        return connections
    
    async def consolidate_working_to_longterm(self, importance_threshold: float = 0.6):
        """Consolidate important working memory contents to long-term storage"""
        active_contents = self.working_memory.get_active_contents()
        
        for slot_index, content in active_contents:
            slot = self.working_memory.slots[slot_index]
            
            # Consolidate if important enough
            if slot.activation_level >= importance_threshold:
                # Store in long-term memory
                lt_memory_id = await self.long_term_memory.store_memory(
                    content=content,
                    memory_type=MemoryType.SEMANTIC,
                    importance=slot.activation_level
                )
                
                # Update working memory slot with reference
                slot.source_memory_id = lt_memory_id
                
                logger.debug(f"Consolidated working memory slot {slot_index} to long-term memory {lt_memory_id}")
    
    async def memory_guided_decision(self, decision_context: Dict[str, Any],
                                   options: List[str]) -> Dict[str, Any]:
        """
        Make decision guided by memory systems.
        
        Args:
            decision_context: Context for the decision
            options: Available options
            
        Returns:
            Decision recommendation with reasoning
        """
        option_scores = {}
        
        for option in options:
            # Get memories relevant to this option
            option_memories = await self.retrieve_relevant_memories(
                query=option,
                context=decision_context,
                max_results=5
            )
            
            # Score option based on memory evidence
            score = await self._score_option_from_memories(option, option_memories)
            option_scores[option] = score
        
        # Select best option
        best_option = max(option_scores.items(), key=lambda x: x[1]["total_score"])
        
        return {
            "recommended_option": best_option[0],
            "confidence": best_option[1]["confidence"],
            "reasoning": best_option[1]["reasoning"],
            "supporting_evidence": best_option[1]["evidence"],
            "all_scores": option_scores
        }
    
    async def _analyze_memory_patterns(self, memories: Dict[str, List[Any]], 
                                     problem: str) -> List[Dict[str, Any]]:
        """Analyze patterns across memory systems"""
        patterns = []
        
        # Pattern 1: Analogical reasoning from episodic memory
        if memories["episodic_memory"]:
            similar_episodes = [ep for ep in memories["episodic_memory"] if ep["relevance"] > 0.5]
            if similar_episodes:
                pattern = {
                    "type": "analogical_reasoning",
                    "confidence": 0.7,
                    "insight": f"Similar situation encountered before: {similar_episodes[0]['episode'].event_description}",
                    "memory_ids": [ep["episode"].id for ep in similar_episodes],
                    "reasoning_steps": [
                        "Retrieved similar past experiences",
                        "Identified analogous patterns",
                        "Applied lessons learned"
                    ]
                }
                patterns.append(pattern)
        
        # Pattern 2: Semantic associations from long-term memory
        if memories["long_term_memory"]:
            associated_memories = []
            for result in memories["long_term_memory"]:
                associated_memories.extend(result["associations"])
            
            if associated_memories:
                pattern = {
                    "type": "pattern_recognition",
                    "confidence": 0.6,
                    "insight": f"Found {len(associated_memories)} related concepts that might apply",
                    "memory_ids": [mem.memory.id for mem in associated_memories],
                    "reasoning_steps": [
                        "Identified semantic associations",
                        "Found related concepts",
                        "Synthesized connections"
                    ]
                }
                patterns.append(pattern)
        
        # Pattern 3: Working memory synthesis
        if memories["working_memory"]:
            active_concepts = [mem["content"] for mem in memories["working_memory"]]
            if len(active_concepts) >= 2:
                pattern = {
                    "type": "case_based_reasoning",
                    "confidence": 0.5,
                    "insight": f"Current working memory contains {len(active_concepts)} relevant concepts",
                    "memory_ids": [f"wm_{i}" for i in range(len(active_concepts))],
                    "reasoning_steps": [
                        "Analyzed active working memory",
                        "Found relevant concepts",
                        "Applied immediate context"
                    ]
                }
                patterns.append(pattern)
        
        return patterns
    
    async def _get_diverse_memories(self, domain: str, count: int) -> List[MemoryTrace]:
        """Get diverse memories for creative connection generation"""
        query = MemoryRetrievalQuery()
        query.max_results = count * 2  # Get more to filter for diversity
        query.content_keywords = [domain] if domain != "general" else []
        
        all_memories = await self.long_term_memory.search_memories(query)
        
        # Select diverse memories (simple diversity based on content differences)
        diverse_memories = []
        for result in all_memories:
            memory = result.memory
            
            # Check if sufficiently different from already selected
            is_diverse = True
            for selected in diverse_memories:
                similarity = self._calculate_memory_similarity(memory, selected)
                if similarity > 0.7:  # Too similar
                    is_diverse = False
                    break
            
            if is_diverse:
                diverse_memories.append(memory)
                
            if len(diverse_memories) >= count:
                break
        
        return diverse_memories
    
    async def _evaluate_creative_connection(self, memory1: MemoryTrace, 
                                          memory2: MemoryTrace) -> Optional[CreativeConnection]:
        """Evaluate potential creative connection between two memories"""
        # Calculate semantic distance (novelty comes from connecting distant concepts)
        similarity = self._calculate_memory_similarity(memory1, memory2)
        novelty_score = 1.0 - similarity  # More distant = more novel
        
        if novelty_score < 0.3:  # Too similar to be creative
            return None
        
        # Generate connection description
        connection_type = self._determine_connection_type(memory1, memory2)
        synthesis = f"Connection between {str(memory1.content)[:50]} and {str(memory2.content)[:50]}"
        
        # Generate potential applications
        applications = [
            f"Apply insights from {memory1.memory_type.value} to {memory2.memory_type.value}",
            f"Combine approaches from both domains",
            f"Use {connection_type} relationship for innovation"
        ]
        
        return CreativeConnection(
            memory_ids=[memory1.id, memory2.id],
            connection_type=connection_type,
            novelty_score=novelty_score,
            potential_applications=applications,
            synthesis_description=synthesis
        )
    
    def _calculate_memory_similarity(self, memory1: MemoryTrace, memory2: MemoryTrace) -> float:
        """Calculate similarity between two memories"""
        # Simple text-based similarity
        content1 = str(memory1.content).lower().split()
        content2 = str(memory2.content).lower().split()
        
        if not content1 or not content2:
            return 0.0
        
        set1, set2 = set(content1), set(content2)
        intersection = set1.intersection(set2)
        union = set1.union(set2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _determine_connection_type(self, memory1: MemoryTrace, memory2: MemoryTrace) -> str:
        """Determine type of connection between memories"""
        if memory1.memory_type == memory2.memory_type:
            return "intra_domain"
        elif memory1.memory_type == MemoryType.PROCEDURAL or memory2.memory_type == MemoryType.PROCEDURAL:
            return "method_transfer"
        else:
            return "cross_domain"
    
    def _calculate_episodic_relevance(self, episode: EpisodicMemory, query: str) -> float:
        """Calculate relevance of episode to query"""
        query_words = set(query.lower().split())
        episode_words = set(episode.event_description.lower().split())
        
        if not query_words or not episode_words:
            return 0.0
        
        intersection = query_words.intersection(episode_words)
        return len(intersection) / len(query_words) if query_words else 0.0
    
    async def _score_option_from_memories(self, option: str, 
                                        memories: Dict[str, List[Any]]) -> Dict[str, Any]:
        """Score decision option based on memory evidence"""
        total_score = 0.0
        confidence = 0.0
        reasoning = []
        evidence = []
        
        # Score from episodic memory (past experiences)
        if memories["episodic_memory"]:
            ep_score = 0.0
            for ep_result in memories["episodic_memory"]:
                episode = ep_result["episode"]
                if episode.outcome:
                    # Positive outcome increases score
                    outcome_sentiment = self._analyze_outcome_sentiment(episode.outcome)
                    ep_score += outcome_sentiment * ep_result["relevance"]
                    evidence.append(f"Past experience: {episode.event_description[:50]}")
            
            total_score += ep_score * 0.4  # 40% weight for episodic
            reasoning.append(f"Episodic memory suggests score: {ep_score:.2f}")
        
        # Score from long-term memory (semantic knowledge)
        if memories["long_term_memory"]:
            lt_score = 0.0
            for lt_result in memories["long_term_memory"]:
                memory = lt_result["memory"]
                lt_score += memory.importance * lt_result["relevance"]
                evidence.append(f"Knowledge: {str(memory.content)[:50]}")
            
            total_score += lt_score * 0.4  # 40% weight for long-term
            reasoning.append(f"Long-term memory suggests score: {lt_score:.2f}")
        
        # Score from working memory (current context)
        if memories["working_memory"]:
            wm_score = 0.0
            for wm_result in memories["working_memory"]:
                wm_score += wm_result["similarity"]
                evidence.append(f"Current context: {str(wm_result['content'])[:50]}")
            
            total_score += wm_score * 0.2  # 20% weight for working memory
            reasoning.append(f"Working memory suggests score: {wm_score:.2f}")
        
        # Calculate confidence based on amount of evidence
        confidence = min(1.0, len(evidence) / 5.0)  # Max confidence with 5+ pieces of evidence
        
        return {
            "total_score": total_score,
            "confidence": confidence,
            "reasoning": reasoning,
            "evidence": evidence
        }
    
    def _analyze_outcome_sentiment(self, outcome: str) -> float:
        """Simple sentiment analysis of outcome (positive/negative)"""
        positive_words = {"success", "good", "effective", "worked", "solved", "achieved"}
        negative_words = {"failed", "bad", "ineffective", "problem", "error", "mistake"}
        
        outcome_lower = outcome.lower()
        positive_count = sum(1 for word in positive_words if word in outcome_lower)
        negative_count = sum(1 for word in negative_words if word in outcome_lower)
        
        if positive_count > negative_count:
            return 0.8
        elif negative_count > positive_count:
            return -0.8
        else:
            return 0.0
    
    async def _integration_loop(self):
        """Background integration and consolidation"""
        while self._running:
            try:
                # Consolidate working memory to long-term
                await self.consolidate_working_to_longterm()
                
                # Update metrics
                self._update_integration_metrics()
                
                await asyncio.sleep(30.0)  # Run every 30 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in memory integration loop: {e}")
                await asyncio.sleep(30.0)
    
    async def _creative_connections_loop(self):
        """Background creative connection generation"""
        while self._running:
            try:
                # Generate creative connections periodically
                new_connections = await self.create_creative_connections()
                
                if new_connections:
                    logger.info(f"Generated {len(new_connections)} new creative connections")
                
                await asyncio.sleep(300.0)  # Run every 5 minutes
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in creative connections loop: {e}")
                await asyncio.sleep(300.0)
    
    def _update_integration_metrics(self):
        """Update integration system metrics"""
        # Combine metrics from all subsystems
        wm_metrics = self.working_memory.get_metrics()
        lt_metrics = self.long_term_memory.get_metrics()
        ep_metrics = self.episodic_memory.get_metrics()
        
        self.integration_metrics.total_memories = (
            lt_metrics.total_memories + ep_metrics.total_memories
        )
        self.integration_metrics.working_memory_utilization = wm_metrics.working_memory_utilization
        self.integration_metrics.memory_efficiency = (
            wm_metrics.memory_efficiency + lt_metrics.memory_efficiency + ep_metrics.memory_efficiency
        ) / 3.0
        
        self.integration_metrics.last_updated = datetime.now()
    
    def get_system_metrics(self) -> Dict[str, MemorySystemMetrics]:
        """Get metrics from all memory subsystems"""
        return {
            "integration": self.integration_metrics,
            "working_memory": self.working_memory.get_metrics(),
            "long_term_memory": self.long_term_memory.get_metrics(),
            "episodic_memory": self.episodic_memory.get_metrics()
        }
    
    def get_debug_info(self) -> Dict[str, Any]:
        """Get comprehensive debug information"""
        return {
            "system_status": "running" if self._running else "stopped",
            "memory_interactions": len(self.memory_interactions),
            "creative_connections": len(self.creative_connections),
            "working_memory": self.working_memory.get_debug_info(),
            "long_term_memory": self.long_term_memory.get_debug_info(),
            "episodic_memory": self.episodic_memory.get_debug_info(),
            "recent_interactions": self.memory_interactions[-5:] if self.memory_interactions else []
        }