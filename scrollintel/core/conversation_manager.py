"""
Conversation Manager for ScrollIntel.
Handles chat conversations, message history, and context management.
"""

import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
from sqlalchemy.orm import Session

from .database import get_db
from ..models.database import Base
from ..core.interfaces import AgentError


class ConversationManager:
    """Manages chat conversations and message history."""
    
    def __init__(self):
        self.active_conversations: Dict[str, Dict] = {}
    
    def create_conversation(self, user_id: str, agent_type: str = "cto") -> str:
        """Create a new conversation."""
        conversation_id = str(uuid.uuid4())
        
        self.active_conversations[conversation_id] = {
            "id": conversation_id,
            "user_id": user_id,
            "agent_type": agent_type,
            "messages": [],
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        
        return conversation_id
    
    def add_message(self, conversation_id: str, role: str, content: str, metadata: Optional[Dict] = None) -> Dict:
        """Add a message to a conversation."""
        if conversation_id not in self.active_conversations:
            raise AgentError(f"Conversation {conversation_id} not found")
        
        message = {
            "id": str(uuid.uuid4()),
            "role": role,  # "user" or "assistant"
            "content": content,
            "metadata": metadata or {},
            "timestamp": datetime.utcnow()
        }
        
        self.active_conversations[conversation_id]["messages"].append(message)
        self.active_conversations[conversation_id]["updated_at"] = datetime.utcnow()
        
        return message
    
    def get_conversation(self, conversation_id: str) -> Optional[Dict]:
        """Get a conversation by ID."""
        return self.active_conversations.get(conversation_id)
    
    def get_conversation_history(self, conversation_id: str, limit: int = 50) -> List[Dict]:
        """Get conversation message history."""
        conversation = self.get_conversation(conversation_id)
        if not conversation:
            return []
        
        messages = conversation["messages"]
        return messages[-limit:] if limit else messages
    
    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation."""
        if conversation_id in self.active_conversations:
            del self.active_conversations[conversation_id]
            return True
        return False
    
    def list_user_conversations(self, user_id: str) -> List[Dict]:
        """List all conversations for a user."""
        user_conversations = []
        for conv in self.active_conversations.values():
            if conv["user_id"] == user_id:
                # Return summary without full message history
                user_conversations.append({
                    "id": conv["id"],
                    "agent_type": conv["agent_type"],
                    "created_at": conv["created_at"],
                    "updated_at": conv["updated_at"],
                    "message_count": len(conv["messages"])
                })
        
        return sorted(user_conversations, key=lambda x: x["updated_at"], reverse=True)


# Global conversation manager instance
conversation_manager = ConversationManager()


def get_conversation_manager() -> ConversationManager:
    """Get the global conversation manager instance."""
    return conversation_manager