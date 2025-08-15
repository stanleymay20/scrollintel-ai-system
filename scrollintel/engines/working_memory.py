"""
Working Memory System for AGI Cognitive Architecture

Implements limited-capacity working memory with attention-based maintenance
and interference management, similar to human working memory.
"""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import asyncio
import logging
from collections import deque
import numpy as np

from ..models.memory_models import (
    WorkingMemorySlot, MemoryTrace, MemoryType, 
    MemorySystemMetrics, MemoryRetrievalQuery
)

logger = logging.getLogger(__name__)


class WorkingMemorySystem:
    """
    Working memory system with limited capacity and active maintenance.
    
    Implements the central executive, phonological loop, and visuospatial
    sketchpad components of Baddeley's working memory model.
    """
    
    def __init__(self, capacity: int = 7, decay_rate: float = 0.05):
        self.capacity = capacity
        self.decay_rate = decay_rate
        self.slots: List[WorkingMemorySlot] = [WorkingMemorySlot() for _ in range(capacity)]
        self.attention_focus: Optional[int] = None
        self.rehearsal_queue: deque = deque()
        self.interference_threshold = 0.8
        self.maintenance_interval = 1.0  # seconds
        self._running = False
        self._maintenance_task: Optional[asyncio.Task] = None
        
        # Performance metrics
        self.metrics = MemorySystemMetrics()
        self.access_history: List[Tuple[datetime, str]] = []
    
    async def start(self):
        """Start the working memory maintenance process"""
        if not self._running:
            self._running = True
            self._maintenance_task = asyncio.create_task(self._maintenance_loop())
            logger.info("Working memory system started")
    
    async def stop(self):
        """Stop the working memory maintenance process"""
        self._running = False
        if self._maintenance_task:
            self._maintenance_task.cancel()
            try:
                await self._maintenance_task
            except asyncio.CancelledError:
                pass
        logger.info("Working memory system stopped")
    
    async def store(self, content: Any, importance: float = 0.5, 
                   source_memory_id: Optional[str] = None) -> Optional[int]:
        """
        Store content in working memory.
        
        Args:
            content: Content to store
            importance: Importance level (0.0 to 1.0)
            source_memory_id: ID of source long-term memory if applicable
            
        Returns:
            Slot index if successful, None if failed
        """
        # Find available slot or least important occupied slot
        target_slot = self._find_storage_slot(importance)
        
        if target_slot is None:
            logger.warning("Working memory full, cannot store new content")
            return None
        
        # Store content in slot
        self.slots[target_slot].content = content
        self.slots[target_slot].activation_level = min(1.0, importance + 0.2)
        self.slots[target_slot].last_updated = datetime.now()
        self.slots[target_slot].source_memory_id = source_memory_id
        
        # Update attention focus to new content
        self.attention_focus = target_slot
        
        # Add to rehearsal queue if important
        if importance > 0.6:
            self.rehearsal_queue.append(target_slot)
        
        self._update_metrics()
        self.access_history.append((datetime.now(), f"store_slot_{target_slot}"))
        
        logger.debug(f"Stored content in working memory slot {target_slot}")
        return target_slot
    
    def retrieve(self, slot_index: int) -> Optional[Any]:
        """
        Retrieve content from specific working memory slot.
        
        Args:
            slot_index: Index of slot to retrieve from
            
        Returns:
            Content if available, None otherwise
        """
        if not (0 <= slot_index < self.capacity):
            return None
        
        slot = self.slots[slot_index]
        if not slot.is_active():
            return None
        
        # Boost activation through retrieval
        slot.activation_level = min(1.0, slot.activation_level + 0.1)
        slot.last_updated = datetime.now()
        
        # Update attention focus
        self.attention_focus = slot_index
        
        self.access_history.append((datetime.now(), f"retrieve_slot_{slot_index}"))
        return slot.content
    
    def search(self, query: str, similarity_threshold: float = 0.3) -> List[Tuple[int, Any, float]]:
        """
        Search working memory for content matching query.
        
        Args:
            query: Search query
            similarity_threshold: Minimum similarity for results
            
        Returns:
            List of (slot_index, content, similarity_score) tuples
        """
        results = []
        
        for i, slot in enumerate(self.slots):
            if not slot.is_active() or slot.content is None:
                continue
            
            # Simple text similarity (could be enhanced with embeddings)
            similarity = self._calculate_similarity(query, str(slot.content))
            
            if similarity >= similarity_threshold:
                results.append((i, slot.content, similarity))
        
        # Sort by similarity score
        results.sort(key=lambda x: x[2], reverse=True)
        
        self.access_history.append((datetime.now(), f"search_{query}"))
        return results
    
    def get_active_contents(self) -> List[Tuple[int, Any]]:
        """Get all active contents in working memory"""
        active_contents = []
        
        for i, slot in enumerate(self.slots):
            if slot.is_active() and slot.content is not None:
                active_contents.append((i, slot.content))
        
        return active_contents
    
    def clear_slot(self, slot_index: int):
        """Clear specific working memory slot"""
        if 0 <= slot_index < self.capacity:
            self.slots[slot_index] = WorkingMemorySlot()
            if self.attention_focus == slot_index:
                self.attention_focus = None
            
            self.access_history.append((datetime.now(), f"clear_slot_{slot_index}"))
    
    def clear_all(self):
        """Clear all working memory slots"""
        self.slots = [WorkingMemorySlot() for _ in range(self.capacity)]
        self.attention_focus = None
        self.rehearsal_queue.clear()
        
        self.access_history.append((datetime.now(), "clear_all"))
    
    def get_utilization(self) -> float:
        """Get current working memory utilization (0.0 to 1.0)"""
        active_slots = sum(1 for slot in self.slots if slot.is_active())
        return active_slots / self.capacity
    
    def get_attention_content(self) -> Optional[Any]:
        """Get content currently in attention focus"""
        if self.attention_focus is not None:
            return self.slots[self.attention_focus].content
        return None
    
    def set_attention_focus(self, slot_index: Optional[int]):
        """Set attention focus to specific slot"""
        if slot_index is None or (0 <= slot_index < self.capacity):
            self.attention_focus = slot_index
            if slot_index is not None:
                # Boost activation of focused slot
                self.slots[slot_index].activation_level = min(
                    1.0, self.slots[slot_index].activation_level + 0.2
                )
    
    def _find_storage_slot(self, importance: float) -> Optional[int]:
        """Find best slot for storing new content"""
        # First, try to find empty slot
        for i, slot in enumerate(self.slots):
            if not slot.is_active():
                return i
        
        # If no empty slots, find least important active slot
        if importance > 0.5:  # Only replace if new content is reasonably important
            min_activation = float('inf')
            min_slot = None
            
            for i, slot in enumerate(self.slots):
                if slot.activation_level < min_activation:
                    min_activation = slot.activation_level
                    min_slot = i
            
            # Only replace if new content is more important
            if importance > min_activation + 0.1:
                return min_slot
        
        return None
    
    def _calculate_similarity(self, query: str, content: str) -> float:
        """Calculate similarity between query and content (simplified)"""
        query_words = set(query.lower().split())
        content_words = set(content.lower().split())
        
        if not query_words or not content_words:
            return 0.0
        
        intersection = query_words.intersection(content_words)
        union = query_words.union(content_words)
        
        return len(intersection) / len(union) if union else 0.0
    
    async def _maintenance_loop(self):
        """Background maintenance of working memory"""
        while self._running:
            try:
                await self._perform_maintenance()
                await asyncio.sleep(self.maintenance_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in working memory maintenance: {e}")
                await asyncio.sleep(self.maintenance_interval)
    
    async def _perform_maintenance(self):
        """Perform maintenance operations"""
        # Decay activation levels
        for slot in self.slots:
            if slot.is_active():
                slot.decay(self.decay_rate)
        
        # Rehearse items in rehearsal queue
        if self.rehearsal_queue:
            slot_index = self.rehearsal_queue.popleft()
            if 0 <= slot_index < self.capacity and self.slots[slot_index].is_active():
                # Boost activation through rehearsal
                self.slots[slot_index].activation_level = min(
                    1.0, self.slots[slot_index].activation_level + 0.05
                )
                # Re-add to queue if still important
                if self.slots[slot_index].activation_level > 0.6:
                    self.rehearsal_queue.append(slot_index)
        
        # Update metrics
        self._update_metrics()
    
    def _update_metrics(self):
        """Update performance metrics"""
        active_slots = sum(1 for slot in self.slots if slot.is_active())
        self.metrics.working_memory_utilization = active_slots / self.capacity
        self.metrics.last_updated = datetime.now()
        
        # Calculate average activation
        total_activation = sum(slot.activation_level for slot in self.slots)
        self.metrics.memory_efficiency = total_activation / self.capacity
    
    def get_metrics(self) -> MemorySystemMetrics:
        """Get current system metrics"""
        return self.metrics
    
    def get_debug_info(self) -> Dict[str, Any]:
        """Get debug information about working memory state"""
        return {
            "capacity": self.capacity,
            "active_slots": sum(1 for slot in self.slots if slot.is_active()),
            "attention_focus": self.attention_focus,
            "rehearsal_queue_size": len(self.rehearsal_queue),
            "utilization": self.get_utilization(),
            "slot_activations": [slot.activation_level for slot in self.slots],
            "recent_access": self.access_history[-10:] if self.access_history else []
        }