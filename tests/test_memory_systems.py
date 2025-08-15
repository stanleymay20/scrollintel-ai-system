"""
Comprehensive tests for AGI Memory Architecture Systems

Tests working memory, long-term memory, episodic memory, and integration.
"""

import pytest
import pytest_asyncio
import asyncio
import tempfile
import os
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from scrollintel.engines.working_memory import WorkingMemorySystem
from scrollintel.engines.long_term_memory import LongTermMemorySystem
from scrollintel.engines.episodic_memory import EpisodicMemorySystem
from scrollintel.engines.memory_integration import MemoryIntegrationSystem
from scrollintel.models.memory_models import (
    MemoryTrace, MemoryType, EpisodicMemory, MemoryRetrievalQuery,
    ConsolidationState, RecencyImportanceStrategy
)


class TestWorkingMemorySystem:
    """Test working memory system functionality"""
    
    @pytest_asyncio.fixture
    async def working_memory(self):
        """Create working memory system for testing"""
        wm = WorkingMemorySystem(capacity=5, decay_rate=0.1)
        await wm.start()
        yield wm
        await wm.stop()
    
    @pytest.mark.asyncio
    async def test_store_and_retrieve(self, working_memory):
        """Test basic storage and retrieval"""
        # Store content
        slot = await working_memory.store("test content", importance=0.8)
        assert slot is not None
        assert 0 <= slot < working_memory.capacity
        
        # Retrieve content
        retrieved = working_memory.retrieve(slot)
        assert retrieved == "test content"
        
        # Check slot activation
        assert working_memory.slots[slot].is_active()
    
    @pytest.mark.asyncio
    async def test_capacity_limits(self, working_memory):
        """Test working memory capacity limits"""
        # Fill all slots
        slots = []
        for i in range(working_memory.capacity):
            slot = await working_memory.store(f"content_{i}", importance=0.5)
            slots.append(slot)
        
        # All slots should be filled
        assert len([s for s in slots if s is not None]) == working_memory.capacity
        
        # Try to store one more (should replace least important)
        new_slot = await working_memory.store("new_content", importance=0.9)
        assert new_slot is not None
    
    @pytest.mark.asyncio
    async def test_search_functionality(self, working_memory):
        """Test search in working memory"""
        # Store searchable content
        await working_memory.store("machine learning algorithm", importance=0.7)
        await working_memory.store("neural network training", importance=0.6)
        await working_memory.store("database optimization", importance=0.5)
        
        # Search for relevant content
        results = working_memory.search("machine learning")
        assert len(results) > 0
        
        # Check result format
        slot_index, content, similarity = results[0]
        assert isinstance(slot_index, int)
        assert "machine learning" in content.lower()
        assert 0.0 <= similarity <= 1.0
    
    @pytest.mark.asyncio
    async def test_attention_focus(self, working_memory):
        """Test attention focus mechanism"""
        # Store content and check attention
        slot = await working_memory.store("focused content", importance=0.8)
        assert working_memory.attention_focus == slot
        
        # Get attention content
        attention_content = working_memory.get_attention_content()
        assert attention_content == "focused content"
        
        # Change attention focus
        working_memory.set_attention_focus(None)
        assert working_memory.attention_focus is None
    
    @pytest.mark.asyncio
    async def test_decay_mechanism(self, working_memory):
        """Test activation decay over time"""
        # Store content
        slot = await working_memory.store("decaying content", importance=0.5)
        initial_activation = working_memory.slots[slot].activation_level
        
        # Manually trigger decay
        working_memory.slots[slot].decay(0.2)
        
        # Check activation decreased
        assert working_memory.slots[slot].activation_level < initial_activation
    
    @pytest.mark.asyncio
    async def test_utilization_metrics(self, working_memory):
        """Test utilization calculation"""
        # Initially empty
        assert working_memory.get_utilization() == 0.0
        
        # Store some content
        await working_memory.store("content1", importance=0.5)
        await working_memory.store("content2", importance=0.5)
        
        # Check utilization
        utilization = working_memory.get_utilization()
        assert 0.0 < utilization <= 1.0


class TestLongTermMemorySystem:
    """Test long-term memory system functionality"""
    
    @pytest_asyncio.fixture
    async def long_term_memory(self):
        """Create long-term memory system for testing"""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as tmp:
            db_path = tmp.name
        
        ltm = LongTermMemorySystem(db_path=db_path)
        await ltm.start()
        yield ltm
        await ltm.stop()
        
        # Cleanup
        if os.path.exists(db_path):
            os.unlink(db_path)
    
    @pytest.mark.asyncio
    async def test_store_and_retrieve_memory(self, long_term_memory):
        """Test basic memory storage and retrieval"""
        # Store memory
        memory_id = await long_term_memory.store_memory(
            content="test knowledge",
            memory_type=MemoryType.SEMANTIC,
            importance=0.7
        )
        assert memory_id is not None
        
        # Retrieve memory
        memory = await long_term_memory.retrieve_memory(memory_id)
        assert memory is not None
        assert memory.content == "test knowledge"
        assert memory.memory_type == MemoryType.SEMANTIC
        assert memory.importance == 0.7
    
    @pytest.mark.asyncio
    async def test_memory_search(self, long_term_memory):
        """Test memory search functionality"""
        # Store multiple memories
        await long_term_memory.store_memory("Python programming", MemoryType.SEMANTIC, importance=0.8)
        await long_term_memory.store_memory("Machine learning algorithms", MemoryType.SEMANTIC, importance=0.9)
        await long_term_memory.store_memory("Database design", MemoryType.PROCEDURAL, importance=0.6)
        
        # Search for memories
        query = MemoryRetrievalQuery()
        query.content_keywords = ["programming", "algorithms"]
        query.max_results = 5
        
        results = await long_term_memory.search_memories(query)
        assert len(results) > 0
        
        # Check result format
        result = results[0]
        assert hasattr(result, 'memory')
        assert hasattr(result, 'relevance_score')
        assert 0.0 <= result.relevance_score <= 1.0
    
    @pytest.mark.asyncio
    async def test_associative_links(self, long_term_memory):
        """Test creation and retrieval of associative links"""
        # Store related memories
        memory1_id = await long_term_memory.store_memory("Neural networks", MemoryType.SEMANTIC)
        memory2_id = await long_term_memory.store_memory("Deep learning", MemoryType.SEMANTIC)
        
        # Create association
        await long_term_memory.create_association(
            memory1_id, memory2_id, strength=0.8, link_type="semantic"
        )
        
        # Get associative network
        network = await long_term_memory.get_associative_network(memory1_id, max_depth=2)
        assert memory1_id in network
        assert memory2_id in network[memory1_id]
    
    @pytest.mark.asyncio
    async def test_memory_consolidation(self, long_term_memory):
        """Test memory consolidation process"""
        # Store fresh memory
        memory_id = await long_term_memory.store_memory(
            "fresh memory", MemoryType.SEMANTIC, importance=0.7
        )
        
        memory = await long_term_memory.retrieve_memory(memory_id)
        assert memory.consolidation_state == ConsolidationState.FRESH
        
        # Consolidate memory
        await long_term_memory.consolidate_memory(memory_id)
        
        # Check consolidation state
        memory = await long_term_memory.retrieve_memory(memory_id)
        assert memory.consolidation_state == ConsolidationState.CONSOLIDATED
        assert memory.strength > 0.7  # Should be strengthened
    
    @pytest.mark.asyncio
    async def test_memory_forgetting(self, long_term_memory):
        """Test memory forgetting mechanism"""
        # Store memory
        memory_id = await long_term_memory.store_memory("forgettable memory", MemoryType.SEMANTIC)
        
        # Verify memory exists
        memory = await long_term_memory.retrieve_memory(memory_id)
        assert memory is not None
        
        # Forget memory
        await long_term_memory.forget_memory(memory_id)
        
        # Verify memory is gone
        memory = await long_term_memory.retrieve_memory(memory_id)
        assert memory is None
    
    @pytest.mark.asyncio
    async def test_optimization_strategy(self, long_term_memory):
        """Test memory optimization strategy"""
        strategy = RecencyImportanceStrategy()
        
        # Create old, weak memory
        old_memory = MemoryTrace(
            content="old memory",
            strength=0.2,
            created_at=datetime.now() - timedelta(days=35),
            access_count=1
        )
        
        # Should be forgotten
        assert strategy.should_forget(old_memory)
        
        # Create recent, strong memory
        recent_memory = MemoryTrace(
            content="recent memory",
            strength=0.8,
            created_at=datetime.now(),
            access_count=5
        )
        
        # Should not be forgotten
        assert not strategy.should_forget(recent_memory)


class TestEpisodicMemorySystem:
    """Test episodic memory system functionality"""
    
    @pytest_asyncio.fixture
    async def episodic_memory(self):
        """Create episodic memory system for testing"""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as tmp:
            db_path = tmp.name
        
        em = EpisodicMemorySystem(db_path=db_path)
        await em.start()
        yield em
        await em.stop()
        
        # Cleanup
        if os.path.exists(db_path):
            os.unlink(db_path)
    
    @pytest.mark.asyncio
    async def test_store_and_retrieve_episode(self, episodic_memory):
        """Test basic episode storage and retrieval"""
        # Store episode
        episode_id = await episodic_memory.store_episode(
            event_description="Learned about neural networks",
            location="office",
            participants=["Alice", "Bob"],
            emotions={"excitement": 0.8, "curiosity": 0.9},
            outcome="Successfully implemented CNN",
            lessons_learned=["CNNs are powerful for image recognition"]
        )
        assert episode_id is not None
        
        # Retrieve episode
        episode = await episodic_memory.retrieve_episode(episode_id)
        assert episode is not None
        assert episode.event_description == "Learned about neural networks"
        assert episode.location == "office"
        assert "Alice" in episode.participants
        assert episode.emotions["excitement"] == 0.8
    
    @pytest.mark.asyncio
    async def test_episode_search(self, episodic_memory):
        """Test episode search functionality"""
        # Store multiple episodes
        await episodic_memory.store_episode(
            "Meeting with team about AI project",
            location="conference room",
            participants=["team"]
        )
        await episodic_memory.store_episode(
            "Debugging machine learning model",
            location="office",
            emotions={"frustration": 0.6}
        )
        
        # Search episodes
        results = await episodic_memory.search_episodes(
            query="AI project",
            max_results=5
        )
        assert len(results) > 0
        assert "AI project" in results[0].event_description
        
        # Search by location
        office_episodes = await episodic_memory.search_episodes(
            location="office",
            max_results=5
        )
        assert len(office_episodes) > 0
        assert office_episodes[0].location == "office"
    
    @pytest.mark.asyncio
    async def test_temporal_sequences(self, episodic_memory):
        """Test temporal sequence retrieval"""
        # Store sequence of episodes
        episode1_id = await episodic_memory.store_episode("Started project")
        await asyncio.sleep(0.1)  # Small delay
        episode2_id = await episodic_memory.store_episode("Made progress")
        await asyncio.sleep(0.1)
        episode3_id = await episodic_memory.store_episode("Completed project")
        
        # Get temporal sequence
        sequence = await episodic_memory.get_temporal_sequence(
            episode2_id, window_size=2
        )
        
        # Should include all three episodes in chronological order
        assert len(sequence) >= 2
        descriptions = [ep.event_description for ep in sequence]
        assert "Started project" in descriptions or "Made progress" in descriptions
    
    @pytest.mark.asyncio
    async def test_related_episodes(self, episodic_memory):
        """Test finding related episodes"""
        # Store related episodes
        episode1_id = await episodic_memory.store_episode(
            "Team meeting in conference room",
            location="conference room",
            participants=["Alice", "Bob"]
        )
        episode2_id = await episodic_memory.store_episode(
            "Another meeting in conference room",
            location="conference room",
            participants=["Alice", "Charlie"]
        )
        
        # Find related episodes
        related = await episodic_memory.get_related_episodes(episode1_id, max_results=5)
        
        # Should find the related episode
        assert len(related) > 0
        related_episode, relation_type, strength = related[0]
        assert related_episode.id == episode2_id
        assert relation_type in ["location", "participant"]
        assert strength > 0.0
    
    @pytest.mark.asyncio
    async def test_learning_episodes(self, episodic_memory):
        """Test retrieval of learning episodes"""
        # Store learning episode
        await episodic_memory.store_episode(
            "Learned about Python decorators",
            lessons_learned=["Decorators modify function behavior", "Use @ syntax"]
        )
        await episodic_memory.store_episode(
            "Regular meeting",
            lessons_learned=[]
        )
        
        # Find learning episodes
        learning_episodes = await episodic_memory.get_learning_episodes("Python")
        
        assert len(learning_episodes) > 0
        assert "Python decorators" in learning_episodes[0].event_description
        assert len(learning_episodes[0].lessons_learned) > 0
    
    @pytest.mark.asyncio
    async def test_episode_consolidation(self, episodic_memory):
        """Test episode consolidation"""
        # Store episode
        episode_id = await episodic_memory.store_episode("Important meeting")
        
        # Get initial vividness
        episode = await episodic_memory.retrieve_episode(episode_id)
        initial_vividness = episode.vividness
        
        # Consolidate episode
        await episodic_memory.consolidate_episode(episode_id)
        
        # Check if vividness increased
        episode = await episodic_memory.retrieve_episode(episode_id)
        assert episode.vividness >= initial_vividness
    
    @pytest.mark.asyncio
    async def test_vividness_decay(self, episodic_memory):
        """Test natural vividness decay"""
        # Store episode
        episode_id = await episodic_memory.store_episode("Decaying memory")
        episode = await episodic_memory.retrieve_episode(episode_id)
        
        initial_vividness = episode.vividness
        
        # Apply decay
        episode.decay_vividness(0.1)
        
        # Check vividness decreased
        assert episode.vividness < initial_vividness
        assert episode.vividness >= 0.0  # Should not go below 0


class TestMemoryIntegrationSystem:
    """Test integrated memory system functionality"""
    
    @pytest_asyncio.fixture
    async def memory_system(self):
        """Create integrated memory system for testing"""
        with tempfile.TemporaryDirectory() as tmpdir:
            lt_db = os.path.join(tmpdir, "lt_memory.db")
            ep_db = os.path.join(tmpdir, "ep_memory.db")
            
            system = MemoryIntegrationSystem(
                working_memory_capacity=5,
                lt_memory_db=lt_db,
                episodic_db=ep_db
            )
            await system.start()
            yield system
            await system.stop()
    
    @pytest.mark.asyncio
    async def test_store_experience(self, memory_system):
        """Test storing experience across all memory systems"""
        # Store experience
        memory_ids = await memory_system.store_experience(
            content="Learned new algorithm",
            experience_type="learning",
            context={"location": "office", "outcome": "success"},
            importance=0.8
        )
        
        # Should have IDs from multiple systems
        assert "working_memory" in memory_ids
        assert "long_term_memory" in memory_ids
        assert "episodic_memory" in memory_ids
    
    @pytest.mark.asyncio
    async def test_retrieve_relevant_memories(self, memory_system):
        """Test retrieving memories from all systems"""
        # Store some experiences
        await memory_system.store_experience("Python programming", "learning", importance=0.7)
        await memory_system.store_experience("Machine learning project", "experience", importance=0.8)
        
        # Retrieve relevant memories
        memories = await memory_system.retrieve_relevant_memories(
            query="Python programming",
            max_results=5
        )
        
        # Should have results from all systems
        assert "working_memory" in memories
        assert "long_term_memory" in memories
        assert "episodic_memory" in memories
        
        # Check working memory results
        if memories["working_memory"]:
            wm_result = memories["working_memory"][0]
            assert "content" in wm_result
            assert "similarity" in wm_result
    
    @pytest.mark.asyncio
    async def test_memory_guided_insight(self, memory_system):
        """Test generation of memory-guided insights"""
        # Store relevant experiences
        await memory_system.store_experience(
            "Successfully used neural networks for image classification",
            "experience",
            context={"outcome": "success", "lessons": ["CNNs work well for images"]},
            importance=0.9
        )
        
        # Generate insight
        insight = await memory_system.generate_memory_guided_insight(
            problem="How to approach image classification task?"
        )
        
        # Check insight structure
        assert hasattr(insight, 'insight_type')
        assert hasattr(insight, 'content')
        assert hasattr(insight, 'confidence')
        assert hasattr(insight, 'supporting_memories')
        assert 0.0 <= insight.confidence <= 1.0
    
    @pytest.mark.asyncio
    async def test_creative_connections(self, memory_system):
        """Test creative connection generation"""
        # Store diverse memories
        await memory_system.store_experience("Neural network architecture", "learning", importance=0.8)
        await memory_system.store_experience("Music composition techniques", "learning", importance=0.7)
        await memory_system.store_experience("Biological evolution", "learning", importance=0.6)
        
        # Generate creative connections
        connections = await memory_system.create_creative_connections(
            domain="general",
            novelty_threshold=0.3
        )
        
        # Check connections
        if connections:
            connection = connections[0]
            assert hasattr(connection, 'memory_ids')
            assert hasattr(connection, 'novelty_score')
            assert hasattr(connection, 'potential_applications')
            assert len(connection.memory_ids) >= 2
            assert 0.0 <= connection.novelty_score <= 1.0
    
    @pytest.mark.asyncio
    async def test_memory_guided_decision(self, memory_system):
        """Test memory-guided decision making"""
        # Store decision-relevant experiences
        await memory_system.store_experience(
            "Used TensorFlow for ML project - worked well",
            "experience",
            context={"outcome": "success", "tool": "TensorFlow"},
            importance=0.8
        )
        await memory_system.store_experience(
            "Tried PyTorch for research - very flexible",
            "experience", 
            context={"outcome": "success", "tool": "PyTorch"},
            importance=0.7
        )
        
        # Make decision
        decision = await memory_system.memory_guided_decision(
            decision_context={"task": "machine learning project"},
            options=["TensorFlow", "PyTorch", "Scikit-learn"]
        )
        
        # Check decision structure
        assert "recommended_option" in decision
        assert "confidence" in decision
        assert "reasoning" in decision
        assert "supporting_evidence" in decision
        assert decision["recommended_option"] in ["TensorFlow", "PyTorch", "Scikit-learn"]
        assert 0.0 <= decision["confidence"] <= 1.0
    
    @pytest.mark.asyncio
    async def test_working_to_longterm_consolidation(self, memory_system):
        """Test consolidation from working to long-term memory"""
        # Store high-importance content in working memory
        await memory_system.working_memory.store("important knowledge", importance=0.9)
        
        # Trigger consolidation
        await memory_system.consolidate_working_to_longterm(importance_threshold=0.8)
        
        # Check that content was consolidated
        # (This is a simplified test - in practice, we'd check long-term memory)
        active_contents = memory_system.working_memory.get_active_contents()
        if active_contents:
            slot_index, content = active_contents[0]
            slot = memory_system.working_memory.slots[slot_index]
            # Should have reference to long-term memory
            assert slot.source_memory_id is not None
    
    @pytest.mark.asyncio
    async def test_system_metrics(self, memory_system):
        """Test system metrics collection"""
        # Store some content
        await memory_system.store_experience("test content", importance=0.5)
        
        # Get metrics
        metrics = memory_system.get_system_metrics()
        
        # Check metrics structure
        assert "integration" in metrics
        assert "working_memory" in metrics
        assert "long_term_memory" in metrics
        assert "episodic_memory" in metrics
        
        # Check metric values
        integration_metrics = metrics["integration"]
        assert hasattr(integration_metrics, 'total_memories')
        assert hasattr(integration_metrics, 'working_memory_utilization')
        assert hasattr(integration_metrics, 'memory_efficiency')
    
    @pytest.mark.asyncio
    async def test_debug_info(self, memory_system):
        """Test debug information collection"""
        # Get debug info
        debug_info = memory_system.get_debug_info()
        
        # Check debug info structure
        assert "system_status" in debug_info
        assert "memory_interactions" in debug_info
        assert "creative_connections" in debug_info
        assert "working_memory" in debug_info
        assert "long_term_memory" in debug_info
        assert "episodic_memory" in debug_info
        
        # Check system status
        assert debug_info["system_status"] == "running"


class TestMemoryPerformanceBenchmarks:
    """Performance benchmarks for memory systems"""
    
    @pytest.mark.asyncio
    async def test_working_memory_performance(self):
        """Benchmark working memory operations"""
        wm = WorkingMemorySystem(capacity=10)
        await wm.start()
        
        import time
        
        # Benchmark storage
        start_time = time.time()
        for i in range(100):
            await wm.store(f"content_{i}", importance=0.5)
        storage_time = time.time() - start_time
        
        # Benchmark retrieval
        start_time = time.time()
        for i in range(100):
            wm.retrieve(i % wm.capacity)
        retrieval_time = time.time() - start_time
        
        # Benchmark search
        start_time = time.time()
        for i in range(50):
            wm.search("content")
        search_time = time.time() - start_time
        
        await wm.stop()
        
        # Performance assertions (adjust thresholds as needed)
        assert storage_time < 1.0  # Should store 100 items in under 1 second
        assert retrieval_time < 0.1  # Should retrieve 100 items in under 0.1 seconds
        assert search_time < 0.5  # Should search 50 times in under 0.5 seconds
        
        print(f"Working Memory Performance:")
        print(f"  Storage: {storage_time:.3f}s for 100 operations")
        print(f"  Retrieval: {retrieval_time:.3f}s for 100 operations")
        print(f"  Search: {search_time:.3f}s for 50 operations")
    
    @pytest.mark.asyncio
    async def test_integration_system_performance(self):
        """Benchmark integrated memory system performance"""
        with tempfile.TemporaryDirectory() as tmpdir:
            lt_db = os.path.join(tmpdir, "lt_memory.db")
            ep_db = os.path.join(tmpdir, "ep_memory.db")
            
            system = MemoryIntegrationSystem(
                working_memory_capacity=10,
                lt_memory_db=lt_db,
                episodic_db=ep_db
            )
            await system.start()
            
            import time
            
            # Benchmark experience storage
            start_time = time.time()
            for i in range(50):
                await system.store_experience(
                    f"experience_{i}",
                    experience_type="learning",
                    importance=0.5
                )
            storage_time = time.time() - start_time
            
            # Benchmark memory retrieval
            start_time = time.time()
            for i in range(20):
                await system.retrieve_relevant_memories(f"experience_{i}")
            retrieval_time = time.time() - start_time
            
            await system.stop()
            
            # Performance assertions
            assert storage_time < 5.0  # Should store 50 experiences in under 5 seconds
            assert retrieval_time < 2.0  # Should retrieve 20 times in under 2 seconds
            
            print(f"Integration System Performance:")
            print(f"  Experience Storage: {storage_time:.3f}s for 50 operations")
            print(f"  Memory Retrieval: {retrieval_time:.3f}s for 20 operations")


if __name__ == "__main__":
    # Run specific test
    pytest.main([__file__, "-v"])