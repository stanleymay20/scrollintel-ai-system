"""
Session management system with Redis integration for EXOUSIA security.
Handles user sessions, session validation, and session cleanup.
"""

import json
import redis.asyncio as redis
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, List
from uuid import uuid4, UUID

from ..core.interfaces import SecurityContext, UserRole, SecurityError
from ..core.config import get_config
from .audit import audit_logger, AuditAction


class SessionData:
    """Session data structure."""
    
    def __init__(self, user_id: str, role: UserRole, permissions: List[str],
                 ip_address: str, user_agent: str, created_at: datetime,
                 last_activity: datetime, expires_at: datetime):
        self.user_id = user_id
        self.role = role
        self.permissions = permissions
        self.ip_address = ip_address
        self.user_agent = user_agent
        self.created_at = created_at
        self.last_activity = last_activity
        self.expires_at = expires_at
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Redis storage."""
        return {
            "user_id": self.user_id,
            "role": self.role.value,
            "permissions": self.permissions,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "expires_at": self.expires_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SessionData':
        """Create from dictionary loaded from Redis."""
        return cls(
            user_id=data["user_id"],
            role=UserRole(data["role"]),
            permissions=data["permissions"],
            ip_address=data["ip_address"],
            user_agent=data["user_agent"],
            created_at=datetime.fromisoformat(data["created_at"]),
            last_activity=datetime.fromisoformat(data["last_activity"]),
            expires_at=datetime.fromisoformat(data["expires_at"])
        )
    
    def is_expired(self) -> bool:
        """Check if session is expired."""
        return datetime.now(timezone.utc) > self.expires_at
    
    def update_activity(self) -> None:
        """Update last activity timestamp."""
        self.last_activity = datetime.now(timezone.utc)


class SessionManager:
    """Manages user sessions with Redis backend."""
    
    def __init__(self):
        self.config = get_config()
        self.redis_client: Optional[redis.Redis] = None
        self.session_timeout = timedelta(minutes=self.config.session_timeout_minutes)
        self.max_sessions_per_user = self.config.max_sessions_per_user
    
    async def initialize(self) -> None:
        """Initialize Redis connection."""
        try:
            self.redis_client = redis.Redis(
                host=self.config.redis_host,
                port=self.config.redis_port,
                password=self.config.redis_password,
                db=self.config.redis_db,
                decode_responses=True
            )
            
            # Test connection
            await self.redis_client.ping()
            
        except Exception as e:
            raise SecurityError(f"Failed to initialize session manager: {str(e)}")
    
    async def close(self) -> None:
        """Close Redis connection."""
        if self.redis_client:
            await self.redis_client.close()
    
    def _get_session_key(self, session_id: str) -> str:
        """Get Redis key for session."""
        return f"session:{session_id}"
    
    def _get_user_sessions_key(self, user_id: str) -> str:
        """Get Redis key for user sessions."""
        return f"user_sessions:{user_id}"
    
    async def create_session(self, user_id: str, role: UserRole, permissions: List[str],
                           ip_address: str, user_agent: str) -> str:
        """Create a new session."""
        if not self.redis_client:
            raise SecurityError("Session manager not initialized")
        
        session_id = str(uuid4())
        now = datetime.now(timezone.utc)
        expires_at = now + self.session_timeout
        
        session_data = SessionData(
            user_id=user_id,
            role=role,
            permissions=permissions,
            ip_address=ip_address,
            user_agent=user_agent,
            created_at=now,
            last_activity=now,
            expires_at=expires_at
        )
        
        # Check and enforce max sessions per user
        await self._enforce_max_sessions(user_id)
        
        # Store session data
        session_key = self._get_session_key(session_id)
        await self.redis_client.setex(
            session_key,
            int(self.session_timeout.total_seconds()),
            json.dumps(session_data.to_dict())
        )
        
        # Add to user sessions set
        user_sessions_key = self._get_user_sessions_key(user_id)
        await self.redis_client.sadd(user_sessions_key, session_id)
        await self.redis_client.expire(user_sessions_key, int(self.session_timeout.total_seconds()))
        
        # Log session creation
        await audit_logger.log(
            action=AuditAction.LOGIN_SUCCESS,
            resource_type="session",
            resource_id=session_id,
            user_id=UUID(user_id),
            ip_address=ip_address,
            user_agent=user_agent,
            details={"role": role.value}
        )
        
        return session_id
    
    async def get_session(self, session_id: str) -> Optional[SessionData]:
        """Get session data."""
        if not self.redis_client:
            raise SecurityError("Session manager not initialized")
        
        session_key = self._get_session_key(session_id)
        session_json = await self.redis_client.get(session_key)
        
        if not session_json:
            return None
        
        try:
            session_dict = json.loads(session_json)
            session_data = SessionData.from_dict(session_dict)
            
            # Check if session is expired
            if session_data.is_expired():
                await self.delete_session(session_id)
                return None
            
            return session_data
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            # Invalid session data, delete it
            await self.redis_client.delete(session_key)
            return None
    
    async def validate_session(self, session_id: str, ip_address: str) -> Optional[SecurityContext]:
        """Validate session and return security context."""
        session_data = await self.get_session(session_id)
        
        if not session_data:
            return None
        
        # Optional: Check IP address consistency
        # if session_data.ip_address != ip_address:
        #     await self.delete_session(session_id)
        #     await audit_logger.log_suspicious_activity(
        #         context=SecurityContext(
        #             user_id=session_data.user_id,
        #             role=session_data.role,
        #             permissions=session_data.permissions,
        #             session_id=session_id,
        #             ip_address=ip_address
        #         ),
        #         activity_type="ip_address_change",
        #         details={
        #             "original_ip": session_data.ip_address,
        #             "new_ip": ip_address
        #         }
        #     )
        #     return None
        
        # Update last activity
        await self.update_session_activity(session_id)
        
        return SecurityContext(
            user_id=session_data.user_id,
            role=session_data.role,
            permissions=session_data.permissions,
            session_id=session_id,
            ip_address=ip_address
        )
    
    async def update_session_activity(self, session_id: str) -> None:
        """Update session last activity timestamp."""
        if not self.redis_client:
            return
        
        session_data = await self.get_session(session_id)
        if not session_data:
            return
        
        session_data.update_activity()
        
        # Update in Redis
        session_key = self._get_session_key(session_id)
        await self.redis_client.setex(
            session_key,
            int(self.session_timeout.total_seconds()),
            json.dumps(session_data.to_dict())
        )
    
    async def delete_session(self, session_id: str) -> None:
        """Delete a session."""
        if not self.redis_client:
            return
        
        # Get session data for logging
        session_data = await self.get_session(session_id)
        
        # Delete session
        session_key = self._get_session_key(session_id)
        await self.redis_client.delete(session_key)
        
        # Remove from user sessions set
        if session_data:
            user_sessions_key = self._get_user_sessions_key(session_data.user_id)
            await self.redis_client.srem(user_sessions_key, session_id)
            
            # Log session deletion
            await audit_logger.log(
                action=AuditAction.LOGOUT,
                resource_type="session",
                resource_id=session_id,
                user_id=UUID(session_data.user_id),
                ip_address=session_data.ip_address,
                details={"reason": "manual_logout"}
            )
    
    async def delete_user_sessions(self, user_id: str) -> None:
        """Delete all sessions for a user."""
        if not self.redis_client:
            return
        
        user_sessions_key = self._get_user_sessions_key(user_id)
        session_ids = await self.redis_client.smembers(user_sessions_key)
        
        # Delete each session
        for session_id in session_ids:
            await self.delete_session(session_id)
        
        # Delete user sessions set
        await self.redis_client.delete(user_sessions_key)
    
    async def get_user_sessions(self, user_id: str) -> List[SessionData]:
        """Get all active sessions for a user."""
        if not self.redis_client:
            return []
        
        user_sessions_key = self._get_user_sessions_key(user_id)
        session_ids = await self.redis_client.smembers(user_sessions_key)
        
        sessions = []
        for session_id in session_ids:
            session_data = await self.get_session(session_id)
            if session_data:
                sessions.append(session_data)
        
        return sessions
    
    async def _enforce_max_sessions(self, user_id: str) -> None:
        """Enforce maximum sessions per user."""
        sessions = await self.get_user_sessions(user_id)
        
        if len(sessions) >= self.max_sessions_per_user:
            # Sort by last activity and remove oldest sessions
            sessions.sort(key=lambda s: s.last_activity)
            sessions_to_remove = sessions[:len(sessions) - self.max_sessions_per_user + 1]
            
            for session in sessions_to_remove:
                # Find session ID by comparing data
                user_sessions_key = self._get_user_sessions_key(user_id)
                session_ids = await self.redis_client.smembers(user_sessions_key)
                
                for session_id in session_ids:
                    session_data = await self.get_session(session_id)
                    if (session_data and 
                        session_data.created_at == session.created_at and
                        session_data.ip_address == session.ip_address):
                        await self.delete_session(session_id)
                        break
    
    async def cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions (background task)."""
        if not self.redis_client:
            return 0
        
        cleaned_count = 0
        
        # Get all session keys
        session_keys = await self.redis_client.keys("session:*")
        
        for session_key in session_keys:
            session_json = await self.redis_client.get(session_key)
            if not session_json:
                continue
            
            try:
                session_dict = json.loads(session_json)
                session_data = SessionData.from_dict(session_dict)
                
                if session_data.is_expired():
                    session_id = session_key.split(":", 1)[1]
                    await self.delete_session(session_id)
                    cleaned_count += 1
                    
            except (json.JSONDecodeError, KeyError, ValueError):
                # Invalid session data, delete it
                await self.redis_client.delete(session_key)
                cleaned_count += 1
        
        return cleaned_count
    
    async def get_session_stats(self) -> Dict[str, Any]:
        """Get session statistics."""
        if not self.redis_client:
            return {}
        
        session_keys = await self.redis_client.keys("session:*")
        user_session_keys = await self.redis_client.keys("user_sessions:*")
        
        return {
            "total_sessions": len(session_keys),
            "total_users_with_sessions": len(user_session_keys),
            "session_timeout_minutes": self.session_timeout.total_seconds() / 60,
            "max_sessions_per_user": self.max_sessions_per_user
        }


# Global session manager instance
session_manager = SessionManager()