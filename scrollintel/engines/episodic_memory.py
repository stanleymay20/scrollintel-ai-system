"""
Episodic Memory System for AGI Cognitive Architecture

Implements autobiographical memory for experiences, events, and temporal sequences.
Provides context-rich memory storage with emotional and sensory details.
"""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import asyncio
import logging
import json
import sqlite3
from collections import defaultdict, deque
import numpy as np

from ..models.memory_models import (
    EpisodicMemory, MemoryTrace, MemoryType, MemorySystemMetrics,
    MemoryRetrievalQuery, MemoryRetrievalResult
)

logger = logging.getLogger(__name__)


class EpisodicMemorySystem:
    """
    Episodic memory system for storing and retrieving experiential memories.
    
    Maintains temporal sequences, contextual details, and emotional associations
    for autobiographical experiences and learning episodes.
    """
    
    def __init__(self, db_path: str = "episodic_memory.db", max_cache_size: int = 500):
        self.db_path = db_path
        self.max_cache_size = max_cache_size
        
        # In-memory storage for recent episodes
        self.episode_cache: Dict[str, EpisodicMemory] = {}
        self.temporal_index: deque = deque(maxlen=1000)  # Recent episodes by time
        self.location_index: Dict[str, List[str]] = defaultdict(list)
        self.participant_index: Dict[str, List[str]] = defaultdict(list)
        
        # System state
        self._running = False
        self._consolidation_task: Optional[asyncio.Task] = None
        self._decay_task: Optional[asyncio.Task] = None
        
        # Performance metrics
        self.metrics = MemorySystemMetrics()
        
        # Initialize database
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database for episodic memory storage"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Episodes table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS episodes (
                id TEXT PRIMARY KEY,
                event_description TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                location TEXT,
                participants TEXT,
                emotions TEXT,
                sensory_details TEXT,
                outcome TEXT,
                lessons_learned TEXT,
                related_memories TEXT,
                vividness REAL NOT NULL,
                created_at TEXT NOT NULL
            )
        ''')
        
        # Temporal sequences table for episode relationships
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS temporal_sequences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                episode_id TEXT NOT NULL,
                sequence_type TEXT NOT NULL,
                position INTEGER NOT NULL,
                sequence_data TEXT,
                FOREIGN KEY (episode_id) REFERENCES episodes (id)
            )
        ''')
        
        # Create indexes
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON episodes (timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_location ON episodes (location)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_vividness ON episodes (vividness)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_sequence_type ON temporal_sequences (sequence_type)')
        
        conn.commit()
        conn.close()
        
        logger.info("Episodic memory database initialized")
    
    async def start(self):
        """Start the episodic memory system"""
        if not self._running:
            self._running = True
            self._consolidation_task = asyncio.create_task(self._consolidation_loop())
            self._decay_task = asyncio.create_task(self._decay_loop())
            await self._load_recent_episodes()
            logger.info("Episodic memory system started")
    
    async def stop(self):
        """Stop the episodic memory system"""
        self._running = False
        
        # Cancel background tasks
        for task in [self._consolidation_task, self._decay_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Save cache to database
        await self._save_cache()
        logger.info("Episodic memory system stopped")
    
    async def store_episode(self, event_description: str,
                           location: Optional[str] = None,
                           participants: Optional[List[str]] = None,
                           emotions: Optional[Dict[str, float]] = None,
                           sensory_details: Optional[Dict[str, Any]] = None,
                           outcome: Optional[str] = None,
                           lessons_learned: Optional[List[str]] = None) -> str:
        """
        Store new episodic memory.
        
        Args:
            event_description: Description of the event
            location: Where the event occurred
            participants: Who was involved
            emotions: Emotional state during event
            sensory_details: Sensory information
            outcome: Result or outcome of event
            lessons_learned: What was learned from the experience
            
        Returns:
            Episode ID
        """
        episode = EpisodicMemory(
            event_description=event_description,
            location=location,
            participants=participants or [],
            emotions=emotions or {},
            sensory_details=sensory_details or {},
            outcome=outcome,
            lessons_learned=lessons_learned or []
        )
        
        # Add to cache
        self.episode_cache[episode.id] = episode
        
        # Update indexes
        self.temporal_index.append((episode.timestamp, episode.id))
        if location:
            self.location_index[location].append(episode.id)
        for participant in episode.participants:
            self.participant_index[participant].append(episode.id)
        
        # Create temporal sequences
        await self._create_temporal_sequences(episode)
        
        # Manage cache size
        if len(self.episode_cache) > self.max_cache_size:
            await self._evict_oldest_episodes()
        
        self._update_metrics()
        logger.debug(f"Stored episode {episode.id}: {event_description[:50]}...")
        
        return episode.id
    
    async def retrieve_episode(self, episode_id: str) -> Optional[EpisodicMemory]:
        """Retrieve specific episode by ID"""
        # Check cache first
        if episode_id in self.episode_cache:
            return self.episode_cache[episode_id]
        
        # Load from database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM episodes WHERE id = ?', (episode_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            episode = self._row_to_episode(row)
            self.episode_cache[episode_id] = episode
            return episode
        
        return None
    
    async def search_episodes(self, query: str = "",
                            location: Optional[str] = None,
                            participant: Optional[str] = None,
                            time_range: Optional[Tuple[datetime, datetime]] = None,
                            emotion_filter: Optional[str] = None,
                            min_vividness: float = 0.0,
                            max_results: int = 10) -> List[EpisodicMemory]:
        """
        Search episodes based on various criteria.
        
        Args:
            query: Text query for event description
            location: Filter by location
            participant: Filter by participant
            time_range: Filter by time range
            emotion_filter: Filter by emotion type
            min_vividness: Minimum vividness threshold
            max_results: Maximum number of results
            
        Returns:
            List of matching episodes
        """
        results = []
        
        # Search in cache first
        for episode in self.episode_cache.values():
            if self._matches_episode_criteria(episode, query, location, participant,
                                            time_range, emotion_filter, min_vividness):
                results.append(episode)
        
        # Search database if needed
        if len(results) < max_results:
            db_results = await self._search_database(
                query, location, participant, time_range, 
                emotion_filter, min_vividness, max_results - len(results)
            )
            results.extend(db_results)
        
        # Sort by relevance (recency and vividness)
        results.sort(key=lambda e: (e.timestamp, e.vividness), reverse=True)
        
        return results[:max_results]
    
    async def get_temporal_sequence(self, episode_id: str,
                                  sequence_type: str = "chronological",
                                  window_size: int = 5) -> List[EpisodicMemory]:
        """
        Get temporal sequence of episodes around a specific episode.
        
        Args:
            episode_id: Central episode ID
            sequence_type: Type of sequence (chronological, causal, etc.)
            window_size: Number of episodes before and after
            
        Returns:
            List of episodes in temporal sequence
        """
        episode = await self.retrieve_episode(episode_id)
        if not episode:
            return []
        
        # Get episodes in time window
        target_time = episode.timestamp
        start_time = target_time - timedelta(hours=window_size)
        end_time = target_time + timedelta(hours=window_size)
        
        sequence_episodes = await self.search_episodes(
            time_range=(start_time, end_time),
            max_results=window_size * 2 + 1
        )
        
        # Sort chronologically
        sequence_episodes.sort(key=lambda e: e.timestamp)
        
        return sequence_episodes
    
    async def get_related_episodes(self, episode_id: str,
                                 relation_types: Optional[List[str]] = None,
                                 max_results: int = 5) -> List[Tuple[EpisodicMemory, str, float]]:
        """
        Get episodes related to a specific episode.
        
        Args:
            episode_id: Source episode ID
            relation_types: Types of relations to consider
            max_results: Maximum number of related episodes
            
        Returns:
            List of (episode, relation_type, strength) tuples
        """
        episode = await self.retrieve_episode(episode_id)
        if not episode:
            return []
        
        related = []
        relation_types = relation_types or ["location", "participant", "emotion", "outcome"]
        
        # Find episodes with shared characteristics
        for other_episode in self.episode_cache.values():
            if other_episode.id == episode_id:
                continue
            
            relations = self._calculate_episode_relations(episode, other_episode)
            
            for relation_type, strength in relations.items():
                if relation_type in relation_types and strength > 0.3:
                    related.append((other_episode, relation_type, strength))
        
        # Sort by relation strength
        related.sort(key=lambda x: x[2], reverse=True)
        
        return related[:max_results]
    
    async def get_learning_episodes(self, topic: str,
                                  max_results: int = 10) -> List[EpisodicMemory]:
        """Get episodes where learning occurred on a specific topic"""
        learning_episodes = []
        
        for episode in self.episode_cache.values():
            if episode.lessons_learned:
                for lesson in episode.lessons_learned:
                    if topic.lower() in lesson.lower():
                        learning_episodes.append(episode)
                        break
        
        # Sort by recency and vividness
        learning_episodes.sort(key=lambda e: (e.timestamp, e.vividness), reverse=True)
        
        return learning_episodes[:max_results]
    
    async def consolidate_episode(self, episode_id: str):
        """Consolidate episode memory (strengthen and create associations)"""
        episode = await self.retrieve_episode(episode_id)
        if not episode:
            return
        
        # Strengthen vividness if recently accessed
        if episode.vividness < 1.0:
            episode.vividness = min(1.0, episode.vividness + 0.1)
        
        # Create associations with related episodes
        related_episodes = await self.get_related_episodes(episode_id, max_results=3)
        
        for related_episode, relation_type, strength in related_episodes:
            episode.related_memories.add(related_episode.id)
            related_episode.related_memories.add(episode_id)
        
        logger.debug(f"Consolidated episode {episode_id}")
    
    def _matches_episode_criteria(self, episode: EpisodicMemory,
                                query: str, location: Optional[str],
                                participant: Optional[str],
                                time_range: Optional[Tuple[datetime, datetime]],
                                emotion_filter: Optional[str],
                                min_vividness: float) -> bool:
        """Check if episode matches search criteria"""
        # Text query
        if query and query.lower() not in episode.event_description.lower():
            return False
        
        # Location filter
        if location and episode.location != location:
            return False
        
        # Participant filter
        if participant and participant not in episode.participants:
            return False
        
        # Time range filter
        if time_range:
            start_time, end_time = time_range
            if not (start_time <= episode.timestamp <= end_time):
                return False
        
        # Emotion filter
        if emotion_filter and emotion_filter not in episode.emotions:
            return False
        
        # Vividness filter
        if episode.vividness < min_vividness:
            return False
        
        return True
    
    def _calculate_episode_relations(self, episode1: EpisodicMemory,
                                   episode2: EpisodicMemory) -> Dict[str, float]:
        """Calculate relation strengths between two episodes"""
        relations = {}
        
        # Location similarity
        if episode1.location and episode2.location:
            relations["location"] = 1.0 if episode1.location == episode2.location else 0.0
        
        # Participant overlap
        if episode1.participants and episode2.participants:
            common_participants = set(episode1.participants) & set(episode2.participants)
            total_participants = set(episode1.participants) | set(episode2.participants)
            relations["participant"] = len(common_participants) / len(total_participants) if total_participants else 0.0
        
        # Emotional similarity
        if episode1.emotions and episode2.emotions:
            common_emotions = set(episode1.emotions.keys()) & set(episode2.emotions.keys())
            if common_emotions:
                emotion_similarity = sum(
                    1.0 - abs(episode1.emotions[emotion] - episode2.emotions[emotion])
                    for emotion in common_emotions
                ) / len(common_emotions)
                relations["emotion"] = emotion_similarity
        
        # Outcome similarity
        if episode1.outcome and episode2.outcome:
            # Simple text similarity for outcomes
            outcome1_words = set(episode1.outcome.lower().split())
            outcome2_words = set(episode2.outcome.lower().split())
            if outcome1_words and outcome2_words:
                intersection = outcome1_words & outcome2_words
                union = outcome1_words | outcome2_words
                relations["outcome"] = len(intersection) / len(union)
        
        # Temporal proximity
        time_diff = abs((episode1.timestamp - episode2.timestamp).total_seconds())
        temporal_strength = max(0.0, 1.0 - (time_diff / (24 * 3600)))  # Decay over 24 hours
        relations["temporal"] = temporal_strength
        
        return relations
    
    async def _create_temporal_sequences(self, episode: EpisodicMemory):
        """Create temporal sequence relationships for episode"""
        # Find recent episodes for sequence creation
        recent_episodes = []
        cutoff_time = episode.timestamp - timedelta(hours=2)
        
        for timestamp, ep_id in reversed(self.temporal_index):
            if timestamp < cutoff_time:
                break
            if ep_id != episode.id:
                recent_episodes.append(ep_id)
        
        # Store sequence information in database
        if recent_episodes:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for i, prev_episode_id in enumerate(recent_episodes[:3]):  # Last 3 episodes
                cursor.execute('''
                    INSERT INTO temporal_sequences 
                    (episode_id, sequence_type, position, sequence_data)
                    VALUES (?, ?, ?, ?)
                ''', (episode.id, "chronological", i, prev_episode_id))
            
            conn.commit()
            conn.close()
    
    async def _search_database(self, query: str, location: Optional[str],
                             participant: Optional[str],
                             time_range: Optional[Tuple[datetime, datetime]],
                             emotion_filter: Optional[str],
                             min_vividness: float,
                             max_results: int) -> List[EpisodicMemory]:
        """Search database for episodes matching criteria"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Build SQL query
        conditions = []
        params = []
        
        if query:
            conditions.append("event_description LIKE ?")
            params.append(f"%{query}%")
        
        if location:
            conditions.append("location = ?")
            params.append(location)
        
        if time_range:
            conditions.append("timestamp BETWEEN ? AND ?")
            params.extend([time_range[0].isoformat(), time_range[1].isoformat()])
        
        if min_vividness > 0:
            conditions.append("vividness >= ?")
            params.append(min_vividness)
        
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        
        cursor.execute(f'''
            SELECT * FROM episodes 
            WHERE {where_clause}
            ORDER BY timestamp DESC, vividness DESC
            LIMIT ?
        ''', params + [max_results])
        
        rows = cursor.fetchall()
        conn.close()
        
        episodes = []
        for row in rows:
            episode = self._row_to_episode(row)
            
            # Additional filtering for participant and emotion
            if participant and participant not in episode.participants:
                continue
            if emotion_filter and emotion_filter not in episode.emotions:
                continue
            
            episodes.append(episode)
            # Add to cache
            self.episode_cache[episode.id] = episode
        
        return episodes
    
    def _row_to_episode(self, row) -> EpisodicMemory:
        """Convert database row to EpisodicMemory object"""
        return EpisodicMemory(
            id=row[0],
            event_description=row[1],
            timestamp=datetime.fromisoformat(row[2]),
            location=row[3],
            participants=json.loads(row[4]) if row[4] else [],
            emotions=json.loads(row[5]) if row[5] else {},
            sensory_details=json.loads(row[6]) if row[6] else {},
            outcome=row[7],
            lessons_learned=json.loads(row[8]) if row[8] else [],
            related_memories=set(json.loads(row[9])) if row[9] else set(),
            vividness=row[10]
        )
    
    async def _consolidation_loop(self):
        """Background consolidation of episodic memories"""
        while self._running:
            try:
                # Consolidate recent important episodes
                recent_episodes = list(self.episode_cache.values())[-10:]  # Last 10 episodes
                
                for episode in recent_episodes:
                    if episode.vividness > 0.7:  # Only consolidate vivid episodes
                        await self.consolidate_episode(episode.id)
                
                await asyncio.sleep(30.0)  # Run every 30 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in episodic consolidation loop: {e}")
                await asyncio.sleep(30.0)
    
    async def _decay_loop(self):
        """Background decay of episode vividness"""
        while self._running:
            try:
                # Apply natural forgetting to episodes
                for episode in self.episode_cache.values():
                    episode.decay_vividness(0.001)  # Very slow decay
                
                await asyncio.sleep(3600.0)  # Run every hour
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in episodic decay loop: {e}")
                await asyncio.sleep(3600.0)
    
    async def _evict_oldest_episodes(self):
        """Remove oldest episodes from cache when it's full"""
        # Sort episodes by timestamp
        sorted_episodes = sorted(
            self.episode_cache.items(),
            key=lambda x: x[1].timestamp
        )
        
        # Remove oldest 10% of episodes
        num_to_remove = max(1, len(sorted_episodes) // 10)
        
        for i in range(num_to_remove):
            episode_id, episode = sorted_episodes[i]
            
            # Save to database before removing from cache
            await self._save_episode_to_db(episode)
            del self.episode_cache[episode_id]
        
        logger.debug(f"Evicted {num_to_remove} episodes from cache")
    
    async def _save_episode_to_db(self, episode: EpisodicMemory):
        """Save single episode to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO episodes 
            (id, event_description, timestamp, location, participants, emotions,
             sensory_details, outcome, lessons_learned, related_memories, vividness, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            episode.id,
            episode.event_description,
            episode.timestamp.isoformat(),
            episode.location,
            json.dumps(episode.participants),
            json.dumps(episode.emotions),
            json.dumps(episode.sensory_details),
            episode.outcome,
            json.dumps(episode.lessons_learned),
            json.dumps(list(episode.related_memories)),
            episode.vividness,
            datetime.now().isoformat()
        ))
        
        conn.commit()
        conn.close()
    
    async def _load_recent_episodes(self):
        """Load recent episodes into cache on startup"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Load episodes from last 7 days
        cutoff_time = datetime.now() - timedelta(days=7)
        
        cursor.execute('''
            SELECT * FROM episodes 
            WHERE timestamp >= ?
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (cutoff_time.isoformat(), self.max_cache_size))
        
        rows = cursor.fetchall()
        conn.close()
        
        for row in rows:
            episode = self._row_to_episode(row)
            self.episode_cache[episode.id] = episode
            
            # Update indexes
            self.temporal_index.append((episode.timestamp, episode.id))
            if episode.location:
                self.location_index[episode.location].append(episode.id)
            for participant in episode.participants:
                self.participant_index[participant].append(episode.id)
        
        logger.info(f"Loaded {len(rows)} recent episodes into cache")
    
    async def _save_cache(self):
        """Save all cached episodes to database"""
        for episode in self.episode_cache.values():
            await self._save_episode_to_db(episode)
        
        logger.info(f"Saved {len(self.episode_cache)} episodes to database")
    
    def _update_metrics(self):
        """Update system metrics"""
        self.metrics.total_memories = len(self.episode_cache)
        
        # Calculate average vividness
        if self.episode_cache:
            avg_vividness = sum(ep.vividness for ep in self.episode_cache.values()) / len(self.episode_cache)
            self.metrics.memory_efficiency = avg_vividness
        
        self.metrics.last_updated = datetime.now()
    
    def get_metrics(self) -> MemorySystemMetrics:
        """Get current system metrics"""
        return self.metrics
    
    def get_debug_info(self) -> Dict[str, Any]:
        """Get debug information about episodic memory state"""
        return {
            "cached_episodes": len(self.episode_cache),
            "temporal_index_size": len(self.temporal_index),
            "location_index_size": len(self.location_index),
            "participant_index_size": len(self.participant_index),
            "average_vividness": sum(ep.vividness for ep in self.episode_cache.values()) / len(self.episode_cache) if self.episode_cache else 0.0,
            "recent_episodes": [(ep.timestamp.isoformat(), ep.event_description[:50]) 
                              for ep in sorted(self.episode_cache.values(), 
                                             key=lambda x: x.timestamp, reverse=True)[:5]]
        }