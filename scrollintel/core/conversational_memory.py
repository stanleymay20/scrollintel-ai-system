"""
Conversational Memory System - Maintains context and memory across agent interactions
Implements requirements 2.3, 6.1, 6.2 for conversational context and memory.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from uuid import uuid4
import sqlite3
import aiosqlite
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ConversationTurn:
    """Single turn in a conversation."""
    turn_id: str
    conversation_id: str
    user_message: str
    agent_response: str
    agent_id: str
    timestamp: datetime
    user_sentiment: Optional[float] = None
    response_quality_score: Optional[float] = None
    topics_discussed: List[str] = field(default_factory=list)
    entities_mentioned: List[str] = field(default_factory=list)
    user_satisfaction: Optional[int] = None  # 1-5 scale


@dataclass
class ConversationContext:
    """Context for maintaining conversation state and memory."""
    conversation_id: str
    user_id: str
    agent_id: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_activity: datetime = field(default_factory=datetime.utcnow)
    turn_count: int = 0
    context_summary: str = ""
    active_topics: List[str] = field(default_factory=list)
    user_mood: Optional[str] = None
    conversation_goal: Optional[str] = None


@dataclass
class ResponseContext:
    """Context for generating responses."""
    conversation_context: ConversationContext
    recent_turns: List[ConversationTurn] = field(default_factory=list)
    user_profile: Optional['UserProfile'] = None
    agent_state: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UserProfile:
    """User profile for personalized interactions."""
    user_id: str
    preferred_communication_style: str = "professional"
    expertise_level: str = "intermediate"  # beginner, intermediate, expert
    preferred_agents: List[str] = field(default_factory=list)
    interaction_history: Dict[str, int] = field(default_factory=dict)  # agent_id -> count
    topics_of_interest: List[str] = field(default_factory=list)
    feedback_patterns: Dict[str, float] = field(default_factory=dict)
    last_active: Optional[datetime] = None
    total_interactions: int = 0


@dataclass
class ConversationSummary:
    """Summary of a conversation for quick reference."""
    conversation_id: str
    user_id: str
    agent_id: str
    start_time: datetime
    end_time: Optional[datetime]
    total_turns: int
    main_topics: List[str]
    user_satisfaction_avg: Optional[float]
    key_outcomes: List[str] = field(default_factory=list)
    follow_up_needed: bool = False


class ConversationalMemoryEngine:
    """Engine for managing conversational memory and context."""
    
    def __init__(self, db_path: str = "conversational_memory.db"):
        self.db_path = db_path
        self.active_conversations: Dict[str, Dict[str, Any]] = {}
        self.user_profiles: Dict[str, UserProfile] = {}
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize SQLite database for persistent memory storage."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Conversations table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    conversation_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    agent_id TEXT NOT NULL,
                    start_time TIMESTAMP NOT NULL,
                    end_time TIMESTAMP,
                    total_turns INTEGER DEFAULT 0,
                    user_satisfaction_avg REAL,
                    main_topics TEXT,  -- JSON array
                    key_outcomes TEXT,  -- JSON array
                    follow_up_needed BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Conversation turns table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversation_turns (
                    turn_id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    user_message TEXT NOT NULL,
                    agent_response TEXT NOT NULL,
                    agent_id TEXT NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    user_sentiment REAL,
                    response_quality_score REAL,
                    topics_discussed TEXT,  -- JSON array
                    entities_mentioned TEXT,  -- JSON array
                    user_satisfaction INTEGER,
                    FOREIGN KEY (conversation_id) REFERENCES conversations (conversation_id)
                )
            """)
            
            # User profiles table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_profiles (
                    user_id TEXT PRIMARY KEY,
                    preferred_communication_style TEXT DEFAULT 'professional',
                    expertise_level TEXT DEFAULT 'intermediate',
                    preferred_agents TEXT,  -- JSON array
                    interaction_history TEXT,  -- JSON object
                    topics_of_interest TEXT,  -- JSON array
                    feedback_patterns TEXT,  -- JSON object
                    last_active TIMESTAMP,
                    total_interactions INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Agent memory table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS agent_memory (
                    memory_id TEXT PRIMARY KEY,
                    agent_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    memory_type TEXT NOT NULL,  -- preference, fact, pattern
                    memory_content TEXT NOT NULL,
                    confidence_score REAL DEFAULT 1.0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    access_count INTEGER DEFAULT 1
                )
            """)
            
            conn.commit()
            conn.close()
            logger.info("Conversational memory database initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize conversational memory database: {e}")
    
    async def start_conversation(self, user_id: str, agent_id: str) -> str:
        """Start a new conversation and return conversation ID."""
        conversation_id = f"conv_{uuid4()}"
        
        # Initialize active conversation
        self.active_conversations[conversation_id] = {
            "conversation_id": conversation_id,
            "user_id": user_id,
            "agent_id": agent_id,
            "start_time": datetime.now(),
            "turns": [],
            "context": {},
            "user_preferences": await self._get_user_preferences(user_id),
            "agent_memory": await self._get_agent_memory(agent_id, user_id)
        }
        
        # Store in database
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT INTO conversations (conversation_id, user_id, agent_id, start_time)
                VALUES (?, ?, ?, ?)
            """, (conversation_id, user_id, agent_id, datetime.now()))
            await db.commit()
        
        logger.info(f"Started conversation {conversation_id} between user {user_id} and agent {agent_id}")
        return conversation_id
    
    async def add_conversation_turn(
        self, 
        conversation_id: str, 
        user_message: str, 
        agent_response: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Add a turn to the conversation."""
        if conversation_id not in self.active_conversations:
            raise ValueError(f"Conversation {conversation_id} not found")
        
        conversation = self.active_conversations[conversation_id]
        turn_id = f"turn_{uuid4()}"
        
        # Extract topics and entities (simplified - could use NLP)
        topics = self._extract_topics(user_message + " " + agent_response)
        entities = self._extract_entities(user_message + " " + agent_response)
        
        turn = ConversationTurn(
            turn_id=turn_id,
            conversation_id=conversation_id,
            user_message=user_message,
            agent_response=agent_response,
            agent_id=conversation["agent_id"],
            timestamp=datetime.now(),
            topics_discussed=topics,
            entities_mentioned=entities,
            user_sentiment=metadata.get("user_sentiment") if metadata else None,
            response_quality_score=metadata.get("response_quality") if metadata else None,
            user_satisfaction=metadata.get("user_satisfaction") if metadata else None
        )
        
        # Add to active conversation
        conversation["turns"].append(turn)
        
        # Update context
        self._update_conversation_context(conversation_id, turn)
        
        # Store in database
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT INTO conversation_turns (
                    turn_id, conversation_id, user_message, agent_response, 
                    agent_id, timestamp, user_sentiment, response_quality_score,
                    topics_discussed, entities_mentioned, user_satisfaction
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                turn_id, conversation_id, user_message, agent_response,
                conversation["agent_id"], datetime.now(), turn.user_sentiment,
                turn.response_quality_score, json.dumps(topics), 
                json.dumps(entities), turn.user_satisfaction
            ))
            await db.commit()
        
        # Update agent memory
        await self._update_agent_memory(conversation["agent_id"], conversation["user_id"], turn)
        
        logger.debug(f"Added turn {turn_id} to conversation {conversation_id}")
        return turn_id
    
    async def get_conversation_context(self, conversation_id: str) -> Dict[str, Any]:
        """Get current conversation context."""
        if conversation_id not in self.active_conversations:
            # Try to load from database
            await self._load_conversation(conversation_id)
        
        if conversation_id in self.active_conversations:
            conversation = self.active_conversations[conversation_id]
            return {
                "conversation_id": conversation_id,
                "user_id": conversation["user_id"],
                "agent_id": conversation["agent_id"],
                "turn_count": len(conversation["turns"]),
                "recent_topics": self._get_recent_topics(conversation_id),
                "user_preferences": conversation.get("user_preferences", {}),
                "agent_memory": conversation.get("agent_memory", {}),
                "context": conversation.get("context", {})
            }
        
        return {}
    
    async def get_user_conversation_history(self, user_id: str, limit: int = 10) -> List[ConversationSummary]:
        """Get user's conversation history."""
        summaries = []
        
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute("""
                SELECT conversation_id, agent_id, start_time, end_time, total_turns,
                       user_satisfaction_avg, main_topics, key_outcomes, follow_up_needed
                FROM conversations 
                WHERE user_id = ? 
                ORDER BY start_time DESC 
                LIMIT ?
            """, (user_id, limit)) as cursor:
                async for row in cursor:
                    summary = ConversationSummary(
                        conversation_id=row[0],
                        user_id=user_id,
                        agent_id=row[1],
                        start_time=datetime.fromisoformat(row[2]),
                        end_time=datetime.fromisoformat(row[3]) if row[3] else None,
                        total_turns=row[4],
                        user_satisfaction_avg=row[5],
                        main_topics=json.loads(row[6]) if row[6] else [],
                        key_outcomes=json.loads(row[7]) if row[7] else [],
                        follow_up_needed=bool(row[8])
                    )
                    summaries.append(summary)
        
        return summaries
    
    async def end_conversation(self, conversation_id: str) -> ConversationSummary:
        """End a conversation and create summary."""
        if conversation_id not in self.active_conversations:
            raise ValueError(f"Conversation {conversation_id} not found")
        
        conversation = self.active_conversations[conversation_id]
        turns = conversation["turns"]
        
        # Create summary
        main_topics = self._get_main_topics(turns)
        avg_satisfaction = self._calculate_average_satisfaction(turns)
        key_outcomes = self._extract_key_outcomes(turns)
        
        summary = ConversationSummary(
            conversation_id=conversation_id,
            user_id=conversation["user_id"],
            agent_id=conversation["agent_id"],
            start_time=conversation["start_time"],
            end_time=datetime.now(),
            total_turns=len(turns),
            main_topics=main_topics,
            user_satisfaction_avg=avg_satisfaction,
            key_outcomes=key_outcomes,
            follow_up_needed=self._needs_follow_up(turns)
        )
        
        # Update database
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                UPDATE conversations 
                SET end_time = ?, total_turns = ?, user_satisfaction_avg = ?,
                    main_topics = ?, key_outcomes = ?, follow_up_needed = ?
                WHERE conversation_id = ?
            """, (
                datetime.now(), len(turns), avg_satisfaction,
                json.dumps(main_topics), json.dumps(key_outcomes),
                summary.follow_up_needed, conversation_id
            ))
            await db.commit()
        
        # Update user profile
        await self._update_user_profile(conversation["user_id"], summary)
        
        # Remove from active conversations
        del self.active_conversations[conversation_id]
        
        logger.info(f"Ended conversation {conversation_id} with {len(turns)} turns")
        return summary
    
    async def get_agent_memory_for_user(self, agent_id: str, user_id: str) -> Dict[str, Any]:
        """Get agent's memory about a specific user."""
        memory = {}
        
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute("""
                SELECT memory_type, memory_content, confidence_score, created_at
                FROM agent_memory 
                WHERE agent_id = ? AND user_id = ?
                ORDER BY last_accessed DESC
            """, (agent_id, user_id)) as cursor:
                async for row in cursor:
                    memory_type = row[0]
                    if memory_type not in memory:
                        memory[memory_type] = []
                    memory[memory_type].append({
                        "content": row[1],
                        "confidence": row[2],
                        "created_at": row[3]
                    })
        
        return memory
    
    def _extract_topics(self, text: str) -> List[str]:
        """Extract topics from text (simplified implementation)."""
        topics = []
        text_lower = text.lower()
        
        # Technical topics
        if any(word in text_lower for word in ["data", "analysis", "dataset", "csv"]):
            topics.append("data_analysis")
        if any(word in text_lower for word in ["model", "ml", "machine learning", "ai"]):
            topics.append("machine_learning")
        if any(word in text_lower for word in ["architecture", "system", "design", "scalability"]):
            topics.append("system_architecture")
        if any(word in text_lower for word in ["dashboard", "visualization", "chart", "graph"]):
            topics.append("data_visualization")
        if any(word in text_lower for word in ["api", "endpoint", "integration", "service"]):
            topics.append("api_development")
        if any(word in text_lower for word in ["database", "sql", "query", "table"]):
            topics.append("database")
        if any(word in text_lower for word in ["performance", "optimization", "speed", "latency"]):
            topics.append("performance")
        if any(word in text_lower for word in ["security", "authentication", "authorization", "encryption"]):
            topics.append("security")
        
        return topics
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extract entities from text (simplified implementation)."""
        entities = []
        text_lower = text.lower()
        
        # Technology entities
        technologies = [
            "python", "javascript", "java", "react", "fastapi", "django",
            "postgresql", "mysql", "mongodb", "redis", "aws", "azure", "gcp",
            "docker", "kubernetes", "tensorflow", "pytorch", "pandas", "numpy"
        ]
        
        for tech in technologies:
            if tech in text_lower:
                entities.append(tech)
        
        return entities
    
    def _update_conversation_context(self, conversation_id: str, turn: ConversationTurn):
        """Update conversation context based on new turn."""
        if conversation_id not in self.active_conversations:
            return
        
        conversation = self.active_conversations[conversation_id]
        context = conversation.setdefault("context", {})
        
        # Update topic tracking
        topics = context.setdefault("topics", {})
        for topic in turn.topics_discussed:
            topics[topic] = topics.get(topic, 0) + 1
        
        # Update entity tracking
        entities = context.setdefault("entities", {})
        for entity in turn.entities_mentioned:
            entities[entity] = entities.get(entity, 0) + 1
        
        # Update sentiment tracking
        if turn.user_sentiment is not None:
            sentiments = context.setdefault("sentiments", [])
            sentiments.append(turn.user_sentiment)
            if len(sentiments) > 10:  # Keep only recent sentiments
                sentiments.pop(0)
    
    def _get_recent_topics(self, conversation_id: str, limit: int = 5) -> List[str]:
        """Get recent topics from conversation."""
        if conversation_id not in self.active_conversations:
            return []
        
        conversation = self.active_conversations[conversation_id]
        recent_turns = conversation["turns"][-limit:]
        
        topics = []
        for turn in recent_turns:
            topics.extend(turn.topics_discussed)
        
        # Return unique topics in order of appearance
        seen = set()
        unique_topics = []
        for topic in reversed(topics):
            if topic not in seen:
                unique_topics.append(topic)
                seen.add(topic)
        
        return list(reversed(unique_topics))
    
    def _get_main_topics(self, turns: List[ConversationTurn]) -> List[str]:
        """Get main topics from conversation turns."""
        topic_counts = {}
        for turn in turns:
            for topic in turn.topics_discussed:
                topic_counts[topic] = topic_counts.get(topic, 0) + 1
        
        # Return top 5 topics
        sorted_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)
        return [topic for topic, count in sorted_topics[:5]]
    
    def _calculate_average_satisfaction(self, turns: List[ConversationTurn]) -> Optional[float]:
        """Calculate average user satisfaction from turns."""
        satisfactions = [turn.user_satisfaction for turn in turns if turn.user_satisfaction is not None]
        if satisfactions:
            return sum(satisfactions) / len(satisfactions)
        return None
    
    def _extract_key_outcomes(self, turns: List[ConversationTurn]) -> List[str]:
        """Extract key outcomes from conversation turns."""
        outcomes = []
        
        # Look for outcome indicators in agent responses
        for turn in turns:
            response_lower = turn.agent_response.lower()
            if any(word in response_lower for word in ["completed", "solved", "resolved", "implemented"]):
                # Extract the sentence containing the outcome
                sentences = turn.agent_response.split('.')
                for sentence in sentences:
                    if any(word in sentence.lower() for word in ["completed", "solved", "resolved", "implemented"]):
                        outcomes.append(sentence.strip())
                        break
        
        return outcomes[:3]  # Return top 3 outcomes
    
    def _needs_follow_up(self, turns: List[ConversationTurn]) -> bool:
        """Determine if conversation needs follow-up."""
        if not turns:
            return False
        
        last_turn = turns[-1]
        
        # Check if last response indicates follow-up needed
        response_lower = last_turn.agent_response.lower()
        follow_up_indicators = [
            "follow up", "check back", "monitor", "review", "update",
            "let me know", "keep me posted", "schedule", "next steps"
        ]
        
        return any(indicator in response_lower for indicator in follow_up_indicators)
    
    async def _get_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """Get user preferences from database."""
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute("""
                SELECT preferred_communication_style, expertise_level, preferred_agents,
                       topics_of_interest, feedback_patterns
                FROM user_profiles WHERE user_id = ?
            """, (user_id,)) as cursor:
                row = await cursor.fetchone()
                if row:
                    return {
                        "communication_style": row[0],
                        "expertise_level": row[1],
                        "preferred_agents": json.loads(row[2]) if row[2] else [],
                        "topics_of_interest": json.loads(row[3]) if row[3] else [],
                        "feedback_patterns": json.loads(row[4]) if row[4] else {}
                    }
        
        return {}
    
    async def _get_agent_memory(self, agent_id: str, user_id: str) -> Dict[str, Any]:
        """Get agent memory for specific user."""
        return await self.get_agent_memory_for_user(agent_id, user_id)
    
    async def _update_agent_memory(self, agent_id: str, user_id: str, turn: ConversationTurn):
        """Update agent memory based on conversation turn."""
        memories_to_add = []
        
        # Extract preferences from user message
        user_message_lower = turn.user_message.lower()
        if "prefer" in user_message_lower or "like" in user_message_lower:
            memories_to_add.append({
                "type": "preference",
                "content": f"User preference: {turn.user_message[:100]}...",
                "confidence": 0.8
            })
        
        # Extract facts from conversation
        if turn.topics_discussed:
            memories_to_add.append({
                "type": "fact",
                "content": f"Discussed topics: {', '.join(turn.topics_discussed)}",
                "confidence": 1.0
            })
        
        # Extract patterns from user satisfaction
        if turn.user_satisfaction is not None:
            memories_to_add.append({
                "type": "pattern",
                "content": f"User satisfaction: {turn.user_satisfaction}/5 for {turn.topics_discussed}",
                "confidence": 0.9
            })
        
        # Store memories in database
        async with aiosqlite.connect(self.db_path) as db:
            for memory in memories_to_add:
                memory_id = f"mem_{uuid4()}"
                await db.execute("""
                    INSERT INTO agent_memory (
                        memory_id, agent_id, user_id, memory_type, 
                        memory_content, confidence_score
                    ) VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    memory_id, agent_id, user_id, memory["type"],
                    memory["content"], memory["confidence"]
                ))
            await db.commit()
    
    async def _update_user_profile(self, user_id: str, summary: ConversationSummary):
        """Update user profile based on conversation summary."""
        async with aiosqlite.connect(self.db_path) as db:
            # Check if profile exists
            async with db.execute("SELECT user_id FROM user_profiles WHERE user_id = ?", (user_id,)) as cursor:
                exists = await cursor.fetchone()
            
            if exists:
                # Update existing profile
                await db.execute("""
                    UPDATE user_profiles 
                    SET total_interactions = total_interactions + 1,
                        last_active = ?,
                        updated_at = ?
                    WHERE user_id = ?
                """, (datetime.now(), datetime.now(), user_id))
            else:
                # Create new profile
                await db.execute("""
                    INSERT INTO user_profiles (user_id, total_interactions, last_active)
                    VALUES (?, 1, ?)
                """, (user_id, datetime.now()))
            
            await db.commit()
    
    async def _load_conversation(self, conversation_id: str):
        """Load conversation from database into active conversations."""
        async with aiosqlite.connect(self.db_path) as db:
            # Load conversation metadata
            async with db.execute("""
                SELECT user_id, agent_id, start_time
                FROM conversations WHERE conversation_id = ?
            """, (conversation_id,)) as cursor:
                row = await cursor.fetchone()
                if not row:
                    return
                
                user_id, agent_id, start_time = row
                
                # Load conversation turns
                turns = []
                async with db.execute("""
                    SELECT turn_id, user_message, agent_response, timestamp,
                           user_sentiment, response_quality_score, topics_discussed,
                           entities_mentioned, user_satisfaction
                    FROM conversation_turns 
                    WHERE conversation_id = ?
                    ORDER BY timestamp
                """, (conversation_id,)) as turn_cursor:
                    async for turn_row in turn_cursor:
                        turn = ConversationTurn(
                            turn_id=turn_row[0],
                            conversation_id=conversation_id,
                            user_message=turn_row[1],
                            agent_response=turn_row[2],
                            agent_id=agent_id,
                            timestamp=datetime.fromisoformat(turn_row[3]),
                            user_sentiment=turn_row[4],
                            response_quality_score=turn_row[5],
                            topics_discussed=json.loads(turn_row[6]) if turn_row[6] else [],
                            entities_mentioned=json.loads(turn_row[7]) if turn_row[7] else [],
                            user_satisfaction=turn_row[8]
                        )
                        turns.append(turn)
                
                # Recreate active conversation
                self.active_conversations[conversation_id] = {
                    "conversation_id": conversation_id,
                    "user_id": user_id,
                    "agent_id": agent_id,
                    "start_time": datetime.fromisoformat(start_time),
                    "turns": turns,
                    "context": {},
                    "user_preferences": await self._get_user_preferences(user_id),
                    "agent_memory": await self._get_agent_memory(agent_id, user_id)
                }
                
                # Rebuild context from turns
                for turn in turns:
                    self._update_conversation_context(conversation_id, turn)