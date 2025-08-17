"""
Session Management System
Implements timeout controls and concurrent session limits
"""

import logging
import secrets
import hashlib
import hmac
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)

class SessionStatus(Enum):
    ACTIVE = "active"
    EXPIRED = "expired"
    TERMINATED = "terminated"
    SUSPENDED = "suspended"

class SessionType(Enum):
    WEB = "web"
    API = "api"
    MOBILE = "mobile"
    DESKTOP = "desktop"

@dataclass
class SessionInfo:
    session_id: str
    user_id: str
    session_type: SessionType
    created_at: datetime
    last_activity: datetime
    expires_at: datetime
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    device_fingerprint: Optional[str] = None
    status: SessionStatus = SessionStatus.ACTIVE
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SessionConfig:
    max_concurrent_sessions: int = 5
    session_timeout_minutes: int = 30
    absolute_timeout_hours: int = 8
    remember_me_days: int = 30
    require_device_verification: bool = True
    allow_concurrent_same_device: bool = True

class SessionManager:
    def __init__(self, config: SessionConfig):
        self.config = config
        self.active_sessions: Dict[str, SessionInfo] = {}
        self.user_sessions: Dict[str, Set[str]] = {}  # user_id -> session_ids
        self.session_tokens: Dict[str, str] = {}  # session_id -> token_hash
        self.device_fingerprints: Dict[str, Set[str]] = {}  # user_id -> device_fingerprints
        
    def create_session(self, user_id: str, session_type: SessionType,
                      ip_address: Optional[str] = None,
                      user_agent: Optional[str] = None,
                      device_fingerprint: Optional[str] = None,
                      remember_me: bool = False) -> tuple[str, str]:
        """Create a new session and return session_id and token"""
        try:
            # Check concurrent session limits
            if not self._check_concurrent_session_limit(user_id, device_fingerprint):
                raise ValueError("Maximum concurrent sessions exceeded")
            
            # Generate session ID and token
            session_id = secrets.token_urlsafe(32)
            session_token = secrets.token_urlsafe(64)
            
            # Calculate expiration times
            now = datetime.utcnow()
            if remember_me:
                expires_at = now + timedelta(days=self.config.remember_me_days)
            else:
                expires_at = now + timedelta(minutes=self.config.session_timeout_minutes)
            
            # Create session info
            session_info = SessionInfo(
                session_id=session_id,
                user_id=user_id,
                session_type=session_type,
                created_at=now,
                last_activity=now,
                expires_at=expires_at,
                ip_address=ip_address,
                user_agent=user_agent,
                device_fingerprint=device_fingerprint,
                metadata={
                    "remember_me": remember_me,
                    "absolute_timeout": (now + timedelta(hours=self.config.absolute_timeout_hours)).isoformat()
                }
            )
            
            # Store session
            self.active_sessions[session_id] = session_info
            
            # Update user sessions mapping
            if user_id not in self.user_sessions:
                self.user_sessions[user_id] = set()
            self.user_sessions[user_id].add(session_id)
            
            # Store token hash
            token_hash = hashlib.sha256(session_token.encode()).hexdigest()
            self.session_tokens[session_id] = token_hash
            
            # Store device fingerprint
            if device_fingerprint:
                if user_id not in self.device_fingerprints:
                    self.device_fingerprints[user_id] = set()
                self.device_fingerprints[user_id].add(device_fingerprint)
            
            logger.info(f"Session created for user {user_id}: {session_id}")
            return session_id, session_token
            
        except Exception as e:
            logger.error(f"Session creation failed: {str(e)}")
            raise
    
    def validate_session(self, session_id: str, session_token: str,
                        ip_address: Optional[str] = None) -> Optional[SessionInfo]:
        """Validate session and return session info if valid"""
        try:
            if session_id not in self.active_sessions:
                logger.warning(f"Session not found: {session_id}")
                return None
            
            session_info = self.active_sessions[session_id]
            
            # Check session status
            if session_info.status != SessionStatus.ACTIVE:
                logger.warning(f"Session not active: {session_id}")
                return None
            
            # Validate token
            if not self._validate_session_token(session_id, session_token):
                logger.warning(f"Invalid session token: {session_id}")
                return None
            
            # Check expiration
            now = datetime.utcnow()
            if now > session_info.expires_at:
                logger.info(f"Session expired: {session_id}")
                self._expire_session(session_id)
                return None
            
            # Check absolute timeout
            absolute_timeout_str = session_info.metadata.get("absolute_timeout")
            if absolute_timeout_str:
                absolute_timeout = datetime.fromisoformat(absolute_timeout_str)
                if now > absolute_timeout:
                    logger.info(f"Session absolute timeout: {session_id}")
                    self._expire_session(session_id)
                    return None
            
            # Check IP address consistency (optional security check)
            if (ip_address and session_info.ip_address and 
                ip_address != session_info.ip_address):
                logger.warning(f"IP address mismatch for session {session_id}")
                # Could terminate session or require re-authentication
            
            # Update last activity
            session_info.last_activity = now
            
            # Extend session if not remember_me
            if not session_info.metadata.get("remember_me", False):
                session_info.expires_at = now + timedelta(minutes=self.config.session_timeout_minutes)
            
            logger.debug(f"Session validated: {session_id}")
            return session_info
            
        except Exception as e:
            logger.error(f"Session validation failed: {str(e)}")
            return None
    
    def terminate_session(self, session_id: str, reason: str = "user_logout") -> bool:
        """Terminate a specific session"""
        try:
            if session_id not in self.active_sessions:
                return False
            
            session_info = self.active_sessions[session_id]
            session_info.status = SessionStatus.TERMINATED
            session_info.metadata["termination_reason"] = reason
            session_info.metadata["terminated_at"] = datetime.utcnow().isoformat()
            
            # Remove from user sessions
            if session_info.user_id in self.user_sessions:
                self.user_sessions[session_info.user_id].discard(session_id)
            
            # Remove token
            if session_id in self.session_tokens:
                del self.session_tokens[session_id]
            
            logger.info(f"Session terminated: {session_id} - {reason}")
            return True
            
        except Exception as e:
            logger.error(f"Session termination failed: {str(e)}")
            return False
    
    def terminate_all_user_sessions(self, user_id: str, except_session: Optional[str] = None,
                                   reason: str = "admin_action") -> int:
        """Terminate all sessions for a user"""
        try:
            if user_id not in self.user_sessions:
                return 0
            
            session_ids = list(self.user_sessions[user_id])
            terminated_count = 0
            
            for session_id in session_ids:
                if session_id != except_session:
                    if self.terminate_session(session_id, reason):
                        terminated_count += 1
            
            logger.info(f"Terminated {terminated_count} sessions for user {user_id}")
            return terminated_count
            
        except Exception as e:
            logger.error(f"Bulk session termination failed: {str(e)}")
            return 0
    
    def suspend_session(self, session_id: str, reason: str) -> bool:
        """Suspend a session (can be resumed)"""
        try:
            if session_id not in self.active_sessions:
                return False
            
            session_info = self.active_sessions[session_id]
            session_info.status = SessionStatus.SUSPENDED
            session_info.metadata["suspension_reason"] = reason
            session_info.metadata["suspended_at"] = datetime.utcnow().isoformat()
            
            logger.info(f"Session suspended: {session_id} - {reason}")
            return True
            
        except Exception as e:
            logger.error(f"Session suspension failed: {str(e)}")
            return False
    
    def resume_session(self, session_id: str) -> bool:
        """Resume a suspended session"""
        try:
            if session_id not in self.active_sessions:
                return False
            
            session_info = self.active_sessions[session_id]
            
            if session_info.status != SessionStatus.SUSPENDED:
                return False
            
            # Check if session hasn't expired while suspended
            if datetime.utcnow() > session_info.expires_at:
                self._expire_session(session_id)
                return False
            
            session_info.status = SessionStatus.ACTIVE
            session_info.metadata["resumed_at"] = datetime.utcnow().isoformat()
            
            logger.info(f"Session resumed: {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Session resume failed: {str(e)}")
            return False
    
    def get_user_sessions(self, user_id: str, active_only: bool = True) -> List[SessionInfo]:
        """Get all sessions for a user"""
        try:
            if user_id not in self.user_sessions:
                return []
            
            session_ids = self.user_sessions[user_id]
            sessions = []
            
            for session_id in session_ids:
                if session_id in self.active_sessions:
                    session_info = self.active_sessions[session_id]
                    
                    if active_only and session_info.status != SessionStatus.ACTIVE:
                        continue
                    
                    sessions.append(session_info)
            
            # Sort by last activity (most recent first)
            sessions.sort(key=lambda x: x.last_activity, reverse=True)
            
            return sessions
            
        except Exception as e:
            logger.error(f"Get user sessions failed: {str(e)}")
            return []
    
    def cleanup_expired_sessions(self) -> int:
        """Remove expired sessions"""
        try:
            now = datetime.utcnow()
            expired_sessions = []
            
            for session_id, session_info in self.active_sessions.items():
                if (session_info.status == SessionStatus.ACTIVE and 
                    now > session_info.expires_at):
                    expired_sessions.append(session_id)
                
                # Also check absolute timeout
                absolute_timeout_str = session_info.metadata.get("absolute_timeout")
                if absolute_timeout_str:
                    absolute_timeout = datetime.fromisoformat(absolute_timeout_str)
                    if now > absolute_timeout:
                        expired_sessions.append(session_id)
            
            # Remove duplicates
            expired_sessions = list(set(expired_sessions))
            
            for session_id in expired_sessions:
                self._expire_session(session_id)
            
            if expired_sessions:
                logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
            
            return len(expired_sessions)
            
        except Exception as e:
            logger.error(f"Session cleanup failed: {str(e)}")
            return 0
    
    def get_session_statistics(self) -> Dict[str, Any]:
        """Get session statistics"""
        try:
            now = datetime.utcnow()
            
            total_sessions = len(self.active_sessions)
            active_sessions = sum(1 for s in self.active_sessions.values() 
                                if s.status == SessionStatus.ACTIVE)
            
            # Sessions by type
            sessions_by_type = {}
            for session_type in SessionType:
                sessions_by_type[session_type.value] = sum(
                    1 for s in self.active_sessions.values() 
                    if s.session_type == session_type and s.status == SessionStatus.ACTIVE
                )
            
            # Recent activity (last hour)
            recent_activity = sum(
                1 for s in self.active_sessions.values()
                if s.last_activity > now - timedelta(hours=1)
            )
            
            return {
                "total_sessions": total_sessions,
                "active_sessions": active_sessions,
                "sessions_by_type": sessions_by_type,
                "recent_activity": recent_activity,
                "unique_users": len(self.user_sessions),
                "timestamp": now.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Session statistics failed: {str(e)}")
            return {}
    
    def _check_concurrent_session_limit(self, user_id: str, 
                                      device_fingerprint: Optional[str] = None) -> bool:
        """Check if user can create a new session"""
        try:
            if user_id not in self.user_sessions:
                return True
            
            active_sessions = [
                session_id for session_id in self.user_sessions[user_id]
                if (session_id in self.active_sessions and 
                    self.active_sessions[session_id].status == SessionStatus.ACTIVE)
            ]
            
            # If allowing concurrent sessions from same device
            if (self.config.allow_concurrent_same_device and device_fingerprint):
                # Count sessions from different devices
                different_device_sessions = [
                    session_id for session_id in active_sessions
                    if (self.active_sessions[session_id].device_fingerprint != device_fingerprint)
                ]
                
                if len(different_device_sessions) >= self.config.max_concurrent_sessions:
                    # Terminate oldest session from different device
                    oldest_session = min(
                        different_device_sessions,
                        key=lambda sid: self.active_sessions[sid].created_at
                    )
                    self.terminate_session(oldest_session, "concurrent_limit_exceeded")
            
            elif len(active_sessions) >= self.config.max_concurrent_sessions:
                # Terminate oldest session
                oldest_session = min(
                    active_sessions,
                    key=lambda sid: self.active_sessions[sid].created_at
                )
                self.terminate_session(oldest_session, "concurrent_limit_exceeded")
            
            return True
            
        except Exception as e:
            logger.error(f"Concurrent session check failed: {str(e)}")
            return False
    
    def _validate_session_token(self, session_id: str, session_token: str) -> bool:
        """Validate session token using constant-time comparison"""
        try:
            if session_id not in self.session_tokens:
                return False
            
            stored_hash = self.session_tokens[session_id]
            provided_hash = hashlib.sha256(session_token.encode()).hexdigest()
            
            return hmac.compare_digest(stored_hash, provided_hash)
            
        except Exception as e:
            logger.error(f"Token validation failed: {str(e)}")
            return False
    
    def _expire_session(self, session_id: str):
        """Mark session as expired"""
        if session_id in self.active_sessions:
            session_info = self.active_sessions[session_id]
            session_info.status = SessionStatus.EXPIRED
            session_info.metadata["expired_at"] = datetime.utcnow().isoformat()
            
            # Remove from user sessions
            if session_info.user_id in self.user_sessions:
                self.user_sessions[session_info.user_id].discard(session_id)
            
            # Remove token
            if session_id in self.session_tokens:
                del self.session_tokens[session_id]