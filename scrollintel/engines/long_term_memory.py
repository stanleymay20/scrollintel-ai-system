"""
Long-Term Memory System for AGI Cognitive Architecture

Implements persistent memory storage with consolidation, retrieval,
and associative networks for creative connections.
"""

from typing import Dict, List, Optional, Any, Set, Tuple
from datetime import datetime, timedelta
import asyncio
import logging
import json
import sqlite3
import numpy as np
from collections import defaultdict
import heapq

from ..models.memory_models import (
    MemoryTrace, MemoryType, ConsolidationState, AssociativeLink,
    MemoryRetrievalQuery, MemoryRetrievalResult, MemoryConsolidationJob,
    MemoryOptimizationStrategy, RecencyImportanceStrategy, MemorySystemMetrics
)

logger = logging.getLogger(__name__)


class LongTermMemorySystem:
    """
    Long-term memory system with persistent storage and associative networks.
    
    Provides semantic memory, procedural memory, and associative connections
    for creative reasoning and knowledge integration.
    """
    
    def __init__(self, db_path: str = "memory.db", 
                 optimization_strategy: Optional[MemoryOptimizationStrategy] = None):
        self.db_path = db_path
        self.optimization_strategy = optimization_strategy or RecencyImportanceStrategy()
        
        # In-memory caches for performance
        self.memory_cache: Dict[str, MemoryTrace] = {}
        self.association_cache: Dict[str, List[AssociativeLink]] = defaultdict(list)
        self.consolidation_queue: List[MemoryConsolidationJob] = []
        
        # System state
        self._running = False
        self._consolidation_task: Optional[asyncio.Task] = None
        self._optimization_task: Optional[asyncio.Task] = None
        
        # Performance metrics
        self.metrics = MemorySystemMetrics()
        
        # Initialize database
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database for persistent storage"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Memory traces table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS memory_traces (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                memory_type TEXT NOT NULL,
                strength REAL NOT NULL,
                created_at TEXT NOT NULL,
                last_accessed TEXT NOT NULL,
                access_count INTEGER NOT NULL,
                consolidation_state TEXT NOT NULL,
                context TEXT,
                importance REAL NOT NULL,
                emotional_valence REAL NOT NULL
            )
        ''')
        
        # Associations table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS associations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                strength REAL NOT NULL,
                link_type TEXT NOT NULL,
                created_at TEXT NOT NULL,
                activation_count INTEGER NOT NULL,
                FOREIGN KEY (source_id) REFERENCES memory_traces (id),
                FOREIGN KEY (target_id) REFERENCES memory_traces (id)
            )
        ''')
        
        # Create indexes for performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_memory_type ON memory_traces (memory_type)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_created_at ON memory_traces (created_at)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_importance ON memory_traces (importance)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_associations_source ON associations (source_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_associations_target ON associations (target_id)')
        
        conn.commit()
        conn.close()
        
        logger.info("Long-term memory database initialized")
    
    async def start(self):
        """Start the long-term memory system"""
        if not self._running:
            self._running = True
            self._consolidation_task = asyncio.create_task(self._consolidation_loop())
            self._optimization_task = asyncio.create_task(self._optimization_loop())
            await self._load_cache()
            logger.info("Long-term memory system started")
    
    async def stop(self):
        """Stop the long-term memory system"""
        self._running = False
        
        # Cancel background tasks
        for task in [self._consolidation_task, self._optimization_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Save cache to database
        await self._save_cache()
        logger.info("Long-term memory system stopped")
    
    async def store_memory(self, content: Any, memory_type: MemoryType = MemoryType.SEMANTIC,
                          context: Optional[Dict[str, Any]] = None,
                          importance: float = 0.5) -> str:
        """
        Store new memory in long-term storage.
        
        Args:
            content: Memory content
            memory_type: Type of memory
            context: Additional context information
            importance: Importance level (0.0 to 1.0)
            
        Returns:
            Memory ID
        """
        memory = MemoryTrace(
            content=content,
            memory_type=memory_type,
            context=context or {},
            importance=importance
        )
        
        # Add to cache
        self.memory_cache[memory.id] = memory
        
        # Schedule for consolidation
        consolidation_job = MemoryConsolidationJob(
            memory_id=memory.id,
            priority=importance
        )
        heapq.heappush(self.consolidation_queue, consolidation_job)
        
        # Create automatic associations
        await self._create_automatic_associations(memory)
        
        self._update_metrics()
        logger.debug(f"Stored memory {memory.id} in long-term storage")
        
        return memory.id
    
    async def retrieve_memory(self, memory_id: str) -> Optional[MemoryTrace]:
        """Retrieve specific memory by ID"""
        # Check cache first
        if memory_id in self.memory_cache:
            memory = self.memory_cache[memory_id]
            memory.access()
            return memory
        
        # Load from database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM memory_traces WHERE id = ?', (memory_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            memory = self._row_to_memory_trace(row)
            memory.access()
            self.memory_cache[memory_id] = memory
            return memory
        
        return None
    
    async def search_memories(self, query: MemoryRetrievalQuery) -> List[MemoryRetrievalResult]:
        """
        Search memories based on query criteria.
        
        Args:
            query: Search query with filters and criteria
            
        Returns:
            List of matching memories with relevance scores
        """
        results = []
        
        # Search in cache first
        for memory in self.memory_cache.values():
            if self._matches_query(memory, query):
                relevance = self._calculate_relevance(memory, query)
                if relevance >= query.similarity_threshold:
                    result = MemoryRetrievalResult(
                        memory=memory,
                        relevance_score=relevance
                    )
                    
                    # Add associated memories if requested
                    if query.include_associations:
                        result.associated_memories = await self._get_associated_memories(
                            memory.id, max_results=3
                        )
                    
                    results.append(result)
        
        # Search database if needed
        if len(results) < query.max_results:
            db_results = await self._search_database(query)
            results.extend(db_results)
        
        # Sort by relevance and limit results
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        results = results[:query.max_results]
        
        # Update access counts
        for result in results:
            result.memory.access()
        
        return results
    
    async def create_association(self, source_id: str, target_id: str,
                               strength: float = 0.5, link_type: str = "semantic"):
        """Create associative link between memories"""
        link = AssociativeLink(
            source_id=source_id,
            target_id=target_id,
            strength=strength,
            link_type=link_type
        )
        
        self.association_cache[source_id].append(link)
        self.association_cache[target_id].append(link)
        
        # Update memory associations
        if source_id in self.memory_cache:
            self.memory_cache[source_id].associations.add(target_id)
        if target_id in self.memory_cache:
            self.memory_cache[target_id].associations.add(source_id)
        
        logger.debug(f"Created association between {source_id} and {target_id}")
    
    async def get_associative_network(self, memory_id: str, 
                                    max_depth: int = 2) -> Dict[str, List[str]]:
        """
        Get associative network around a memory.
        
        Args:
            memory_id: Central memory ID
            max_depth: Maximum traversal depth
            
        Returns:
            Network as adjacency list
        """
        network = defaultdict(list)
        visited = set()
        queue = [(memory_id, 0)]
        
        while queue:
            current_id, depth = queue.pop(0)
            
            if current_id in visited or depth > max_depth:
                continue
            
            visited.add(current_id)
            
            # Get associations for current memory
            associations = self.association_cache.get(current_id, [])
            
            for link in associations:
                target_id = link.target_id if link.source_id == current_id else link.source_id
                
                if link.strength > 0.3:  # Only include strong associations
                    network[current_id].append(target_id)
                    
                    if depth < max_depth:
                        queue.append((target_id, depth + 1))
        
        return dict(network)
    
    async def consolidate_memory(self, memory_id: str):
        """Consolidate specific memory from working to long-term storage"""
        if memory_id not in self.memory_cache:
            return
        
        memory = self.memory_cache[memory_id]
        
        # Update consolidation state
        if memory.consolidation_state == ConsolidationState.FRESH:
            memory.consolidation_state = ConsolidationState.CONSOLIDATING
            
            # Strengthen memory through consolidation
            memory.strength = min(1.0, memory.strength + 0.2)
            
            # Create additional associations during consolidation
            await self._strengthen_associations(memory_id)
            
            memory.consolidation_state = ConsolidationState.CONSOLIDATED
            
            logger.debug(f"Consolidated memory {memory_id}")
    
    async def forget_memory(self, memory_id: str):
        """Remove memory from system (forgetting)"""
        # Remove from cache
        if memory_id in self.memory_cache:
            del self.memory_cache[memory_id]
        
        # Remove associations
        if memory_id in self.association_cache:
            del self.association_cache[memory_id]
        
        # Remove from database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM memory_traces WHERE id = ?', (memory_id,))
        cursor.execute('DELETE FROM associations WHERE source_id = ? OR target_id = ?', 
                      (memory_id, memory_id))
        
        conn.commit()
        conn.close()
        
        logger.debug(f"Forgot memory {memory_id}")
    
    def _matches_query(self, memory: MemoryTrace, query: MemoryRetrievalQuery) -> bool:
        """Check if memory matches query criteria"""
        # Check memory type filter
        if query.memory_types and memory.memory_type not in query.memory_types:
            return False
        
        # Check time range filter
        if query.time_range:
            start_time, end_time = query.time_range
            if not (start_time <= memory.created_at <= end_time):
                return False
        
        # Check minimum strength
        if memory.strength < query.min_strength:
            return False
        
        # Check context filters
        for key, value in query.context_filters.items():
            if key not in memory.context or memory.context[key] != value:
                return False
        
        return True
    
    def _calculate_relevance(self, memory: MemoryTrace, query: MemoryRetrievalQuery) -> float:
        """Calculate relevance score for memory against query"""
        relevance = 0.0
        
        # Content keyword matching
        if query.content_keywords:
            content_str = str(memory.content).lower()
            keyword_matches = sum(1 for keyword in query.content_keywords 
                                if keyword.lower() in content_str)
            relevance += (keyword_matches / len(query.content_keywords)) * 0.6
        
        # Recency bonus
        days_old = (datetime.now() - memory.created_at).days
        recency_score = max(0, 1.0 - (days_old / 365.0))  # Decay over a year
        relevance += recency_score * 0.2
        
        # Importance and strength
        relevance += memory.importance * 0.1
        relevance += memory.strength * 0.1
        
        return min(1.0, relevance)
    
    async def _create_automatic_associations(self, memory: MemoryTrace):
        """Create automatic associations for new memory"""
        # Find similar memories for association
        similar_memories = []
        
        for existing_id, existing_memory in self.memory_cache.items():
            if existing_id == memory.id:
                continue
            
            # Simple similarity based on content overlap
            similarity = self._calculate_content_similarity(memory, existing_memory)
            
            if similarity > 0.4:
                similar_memories.append((existing_id, similarity))
        
        # Create associations with most similar memories
        similar_memories.sort(key=lambda x: x[1], reverse=True)
        
        for existing_id, similarity in similar_memories[:5]:  # Top 5 similar
            await self.create_association(
                memory.id, existing_id, 
                strength=similarity, 
                link_type="semantic"
            )
    
    def _calculate_content_similarity(self, memory1: MemoryTrace, memory2: MemoryTrace) -> float:
        """Calculate similarity between two memories"""
        # Simple text-based similarity (could be enhanced with embeddings)
        content1 = str(memory1.content).lower().split()
        content2 = str(memory2.content).lower().split()
        
        if not content1 or not content2:
            return 0.0
        
        set1, set2 = set(content1), set(content2)
        intersection = set1.intersection(set2)
        union = set1.union(set2)
        
        return len(intersection) / len(union) if union else 0.0
    
    async def _get_associated_memories(self, memory_id: str, 
                                     max_results: int = 5) -> List[MemoryRetrievalResult]:
        """Get memories associated with given memory"""
        associated = []
        
        for link in self.association_cache.get(memory_id, []):
            target_id = link.target_id if link.source_id == memory_id else link.source_id
            
            if target_id in self.memory_cache:
                memory = self.memory_cache[target_id]
                result = MemoryRetrievalResult(
                    memory=memory,
                    relevance_score=link.strength,
                    retrieval_path=[memory_id, target_id]
                )
                associated.append(result)
        
        # Sort by association strength
        associated.sort(key=lambda x: x.relevance_score, reverse=True)
        return associated[:max_results]
    
    async def _search_database(self, query: MemoryRetrievalQuery) -> List[MemoryRetrievalResult]:
        """Search database for memories matching query"""
        results = []
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Build SQL query based on filters
        sql_conditions = []
        params = []
        
        if query.memory_types:
            placeholders = ','.join('?' * len(query.memory_types))
            sql_conditions.append(f'memory_type IN ({placeholders})')
            params.extend([mt.value for mt in query.memory_types])
        
        if query.min_strength > 0:
            sql_conditions.append('strength >= ?')
            params.append(query.min_strength)
        
        if query.time_range:
            sql_conditions.append('created_at BETWEEN ? AND ?')
            params.extend([query.time_range[0].isoformat(), query.time_range[1].isoformat()])
        
        where_clause = ' AND '.join(sql_conditions) if sql_conditions else '1=1'
        
        cursor.execute(f'''
            SELECT * FROM memory_traces 
            WHERE {where_clause}
            ORDER BY importance DESC, strength DESC
            LIMIT ?
        ''', params + [query.max_results])
        
        rows = cursor.fetchall()
        conn.close()
        
        for row in rows:
            memory = self._row_to_memory_trace(row)
            relevance = self._calculate_relevance(memory, query)
            
            if relevance >= query.similarity_threshold:
                result = MemoryRetrievalResult(
                    memory=memory,
                    relevance_score=relevance
                )
                results.append(result)
                
                # Add to cache
                self.memory_cache[memory.id] = memory
        
        return results
    
    def _row_to_memory_trace(self, row) -> MemoryTrace:
        """Convert database row to MemoryTrace object"""
        return MemoryTrace(
            id=row[0],
            content=json.loads(row[1]) if row[1].startswith('{') else row[1],
            memory_type=MemoryType(row[2]),
            strength=row[3],
            created_at=datetime.fromisoformat(row[4]),
            last_accessed=datetime.fromisoformat(row[5]),
            access_count=row[6],
            consolidation_state=ConsolidationState(row[7]),
            context=json.loads(row[8]) if row[8] else {},
            importance=row[9],
            emotional_valence=row[10]
        )
    
    async def _consolidation_loop(self):
        """Background consolidation of memories"""
        while self._running:
            try:
                if self.consolidation_queue:
                    job = heapq.heappop(self.consolidation_queue)
                    await self.consolidate_memory(job.memory_id)
                else:
                    await asyncio.sleep(1.0)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in consolidation loop: {e}")
                await asyncio.sleep(1.0)
    
    async def _optimization_loop(self):
        """Background optimization and forgetting"""
        while self._running:
            try:
                await asyncio.sleep(60.0)  # Run every minute
                await self._perform_optimization()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in optimization loop: {e}")
    
    async def _perform_optimization(self):
        """Perform memory optimization and forgetting"""
        memories_to_forget = []
        
        for memory_id, memory in self.memory_cache.items():
            if self.optimization_strategy.should_forget(memory):
                memories_to_forget.append(memory_id)
        
        # Forget selected memories
        for memory_id in memories_to_forget:
            await self.forget_memory(memory_id)
        
        if memories_to_forget:
            logger.info(f"Forgot {len(memories_to_forget)} memories during optimization")
        
        self._update_metrics()
    
    async def _strengthen_associations(self, memory_id: str):
        """Strengthen associations during consolidation"""
        for link in self.association_cache.get(memory_id, []):
            link.strengthen(0.05)
    
    async def _load_cache(self):
        """Load frequently accessed memories into cache"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Load most important and recently accessed memories
        cursor.execute('''
            SELECT * FROM memory_traces 
            ORDER BY importance DESC, last_accessed DESC 
            LIMIT 1000
        ''')
        
        rows = cursor.fetchall()
        conn.close()
        
        for row in rows:
            memory = self._row_to_memory_trace(row)
            self.memory_cache[memory.id] = memory
        
        logger.info(f"Loaded {len(rows)} memories into cache")
    
    async def _save_cache(self):
        """Save cache to database"""
        if not self.memory_cache:
            return
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for memory in self.memory_cache.values():
            cursor.execute('''
                INSERT OR REPLACE INTO memory_traces 
                (id, content, memory_type, strength, created_at, last_accessed, 
                 access_count, consolidation_state, context, importance, emotional_valence)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                memory.id,
                json.dumps(memory.content) if isinstance(memory.content, (dict, list)) else str(memory.content),
                memory.memory_type.value,
                memory.strength,
                memory.created_at.isoformat(),
                memory.last_accessed.isoformat(),
                memory.access_count,
                memory.consolidation_state.value,
                json.dumps(memory.context),
                memory.importance,
                memory.emotional_valence
            ))
        
        # Save associations
        for source_id, links in self.association_cache.items():
            for link in links:
                cursor.execute('''
                    INSERT OR REPLACE INTO associations 
                    (source_id, target_id, strength, link_type, created_at, activation_count)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    link.source_id,
                    link.target_id,
                    link.strength,
                    link.link_type,
                    link.created_at.isoformat(),
                    link.activation_count
                ))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Saved {len(self.memory_cache)} memories to database")
    
    def _update_metrics(self):
        """Update system metrics"""
        self.metrics.total_memories = len(self.memory_cache)
        self.metrics.consolidation_queue_size = len(self.consolidation_queue)
        
        # Calculate association density
        total_associations = sum(len(links) for links in self.association_cache.values())
        if self.metrics.total_memories > 0:
            self.metrics.association_density = total_associations / self.metrics.total_memories
        
        self.metrics.last_updated = datetime.now()
    
    def get_metrics(self) -> MemorySystemMetrics:
        """Get current system metrics"""
        return self.metrics
    
    def get_debug_info(self) -> Dict[str, Any]:
        """Get debug information about long-term memory state"""
        return {
            "total_memories": len(self.memory_cache),
            "consolidation_queue_size": len(self.consolidation_queue),
            "association_count": sum(len(links) for links in self.association_cache.values()),
            "memory_types": {mt.value: sum(1 for m in self.memory_cache.values() if m.memory_type == mt) 
                           for mt in MemoryType},
            "consolidation_states": {cs.value: sum(1 for m in self.memory_cache.values() if m.consolidation_state == cs) 
                                   for cs in ConsolidationState}
        }