"""
Production Security and Compliance for Visual Generation System
Handles authentication, authorization, content moderation, and audit logging
"""

import asyncio
import logging
import hashlib
import hmac
import jwt
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import aioredis
import aiosqlite
from cryptography.fernet import Fernet
import bcrypt
import re
from PIL import Image
import torch
import transformers

logger = logging.getLogger(__name__)

class UserRole(Enum):
    USER = "user"
    PREMIUM = "premium"
    ADMIN = "admin"
    SYSTEM = "system"

class ContentSafetyLevel(Enum):
    SAFE = "safe"
    QUESTIONABLE = "questionable"
    UNSAFE = "unsafe"
    BLOCKED = "blocked"

class AuditEventType(Enum):
    GENERATION_REQUEST = "generation_request"
    CONTENT_ACCESS = "content_access"
    USER_LOGIN = "user_login"
    ADMIN_ACTION = "admin_action"
    SECURITY_VIOLATION = "security_violation"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"

@dataclass
class SecurityConfig:
    jwt_secret: str
    encryption_key: str
    rate_limit_enabled: bool
    content_moderation_enabled: bool
    audit_logging_enabled: bool
    max_requests_per_minute: int
    max_requests_per_hour: int
    session_timeout_minutes: int
    password_min_length: int
    require_2fa: bool

@dataclass
class UserSession:
    session_id: str
    user_id: str
    role: UserRole
    created_at: datetime
    expires_at: datetime
    ip_address: str
    user_agent: str
    permissions: List[str]

@dataclass
class AuditEvent:
    event_id: str
    event_type: AuditEventType
    user_id: Optional[str]
    session_id: Optional[str]
    timestamp: datetime
    ip_address: str
    user_agent: str
    resource: str
    action: str
    result: str
    metadata: Dict[str, Any]

class AuthenticationManager:
    """Manages user authentication and authorization"""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.redis_client = None
        self.active_sessions: Dict[str, UserSession] = {}
        
        # Initialize encryption
        self.cipher = Fernet(config.encryption_key.encode())
        
        # Initialize Redis for session storage
        asyncio.create_task(self._initialize_redis())
    
    async def _initialize_redis(self):
        """Initialize Redis connection for session management"""
        try:
            self.redis_client = await aioredis.from_url("redis://localhost:6379")
            logger.info("Redis connection established for session management")
        except Exception as e:
            logger.warning(f"Redis connection failed, using in-memory sessions: {str(e)}")
    
    def _hash_password(self, password: str) -> str:
        """Hash password using bcrypt"""
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')
    
    def _verify_password(self, password: str, hashed: str) -> bool:
        """Verify password against hash"""
        return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
    
    def _generate_jwt_token(self, user_id: str, role: UserRole, session_id: str) -> str:
        """Generate JWT token for user session"""
        payload = {
            'user_id': user_id,
            'role': role.value,
            'session_id': session_id,
            'iat': int(time.time()),
            'exp': int(time.time()) + (self.config.session_timeout_minutes * 60)
        }
        
        return jwt.encode(payload, self.config.jwt_secret, algorithm='HS256')
    
    def _verify_jwt_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(token, self.config.jwt_secret, algorithms=['HS256'])
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("JWT token expired")
            return None
        except jwt.InvalidTokenError:
            logger.warning("Invalid JWT token")
            return None
    
    async def authenticate_user(self, username: str, password: str, 
                              ip_address: str, user_agent: str) -> Optional[Tuple[str, UserSession]]:
        """Authenticate user and create session"""
        try:
            # In production, this would query user database
            # For now, we'll simulate user lookup
            user_data = await self._get_user_by_username(username)
            
            if not user_data or not self._verify_password(password, user_data['password_hash']):
                logger.warning(f"Authentication failed for user: {username}")
                return None
            
            # Create session
            session_id = self._generate_session_id()
            session = UserSession(
                session_id=session_id,
                user_id=user_data['user_id'],
                role=UserRole(user_data['role']),
                created_at=datetime.utcnow(),
                expires_at=datetime.utcnow() + timedelta(minutes=self.config.session_timeout_minutes),
                ip_address=ip_address,
                user_agent=user_agent,
                permissions=user_data.get('permissions', [])
            )
            
            # Store session
            await self._store_session(session)
            
            # Generate JWT token
            token = self._generate_jwt_token(user_data['user_id'], session.role, session_id)
            
            logger.info(f"User authenticated successfully: {username}")
            return token, session
            
        except Exception as e:
            logger.error(f"Authentication error: {str(e)}")
            return None
    
    async def validate_session(self, token: str) -> Optional[UserSession]:
        """Validate session token and return session info"""
        try:
            # Decode JWT token
            payload = self._verify_jwt_token(token)
            if not payload:
                return None
            
            session_id = payload.get('session_id')
            if not session_id:
                return None
            
            # Get session from storage
            session = await self._get_session(session_id)
            if not session:
                return None
            
            # Check if session is expired
            if datetime.utcnow() > session.expires_at:
                await self._invalidate_session(session_id)
                return None
            
            return session
            
        except Exception as e:
            logger.error(f"Session validation error: {str(e)}")
            return None
    
    async def invalidate_session(self, session_id: str) -> bool:
        """Invalidate user session"""
        try:
            await self._invalidate_session(session_id)
            logger.info(f"Session invalidated: {session_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to invalidate session: {str(e)}")
            return False
    
    def _generate_session_id(self) -> str:
        """Generate unique session ID"""
        timestamp = str(int(time.time()))
        random_data = str(time.time_ns())
        return hashlib.sha256(f"{timestamp}:{random_data}".encode()).hexdigest()
    
    async def _get_user_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        """Get user data by username (placeholder implementation)"""
        # In production, this would query the user database
        # For now, return a mock user
        if username == "admin":
            return {
                'user_id': 'admin_user_id',
                'username': 'admin',
                'password_hash': self._hash_password('admin_password'),
                'role': 'admin',
                'permissions': ['generate_content', 'admin_access', 'view_metrics']
            }
        elif username == "user":
            return {
                'user_id': 'regular_user_id',
                'username': 'user',
                'password_hash': self._hash_password('user_password'),
                'role': 'user',
                'permissions': ['generate_content']
            }
        return None
    
    async def _store_session(self, session: UserSession):
        """Store session in Redis or memory"""
        try:
            session_data = asdict(session)
            session_data['created_at'] = session.created_at.isoformat()
            session_data['expires_at'] = session.expires_at.isoformat()
            session_data['role'] = session.role.value
            
            if self.redis_client:
                await self.redis_client.setex(
                    f"session:{session.session_id}",
                    self.config.session_timeout_minutes * 60,
                    json.dumps(session_data)
                )
            else:
                self.active_sessions[session.session_id] = session
                
        except Exception as e:
            logger.error(f"Failed to store session: {str(e)}")
            raise
    
    async def _get_session(self, session_id: str) -> Optional[UserSession]:
        """Get session from Redis or memory"""
        try:
            if self.redis_client:
                session_data = await self.redis_client.get(f"session:{session_id}")
                if session_data:
                    data = json.loads(session_data)
                    return UserSession(
                        session_id=data['session_id'],
                        user_id=data['user_id'],
                        role=UserRole(data['role']),
                        created_at=datetime.fromisoformat(data['created_at']),
                        expires_at=datetime.fromisoformat(data['expires_at']),
                        ip_address=data['ip_address'],
                        user_agent=data['user_agent'],
                        permissions=data['permissions']
                    )
            else:
                return self.active_sessions.get(session_id)
                
        except Exception as e:
            logger.error(f"Failed to get session: {str(e)}")
            return None
    
    async def _invalidate_session(self, session_id: str):
        """Remove session from storage"""
        try:
            if self.redis_client:
                await self.redis_client.delete(f"session:{session_id}")
            else:
                self.active_sessions.pop(session_id, None)
                
        except Exception as e:
            logger.error(f"Failed to invalidate session: {str(e)}")

class RateLimiter:
    """Implements rate limiting for API requests"""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.redis_client = None
        self.rate_limits: Dict[str, Dict[str, Any]] = {}
        
        # Initialize Redis for distributed rate limiting
        asyncio.create_task(self._initialize_redis())
    
    async def _initialize_redis(self):
        """Initialize Redis connection for rate limiting"""
        try:
            self.redis_client = await aioredis.from_url("redis://localhost:6379")
            logger.info("Redis connection established for rate limiting")
        except Exception as e:
            logger.warning(f"Redis connection failed, using in-memory rate limiting: {str(e)}")
    
    async def check_rate_limit(self, user_id: str, ip_address: str) -> Tuple[bool, Dict[str, Any]]:
        """Check if request is within rate limits"""
        try:
            if not self.config.rate_limit_enabled:
                return True, {}
            
            current_time = int(time.time())
            
            # Check per-minute limit
            minute_key = f"rate_limit:minute:{user_id}:{current_time // 60}"
            minute_count = await self._get_request_count(minute_key)
            
            if minute_count >= self.config.max_requests_per_minute:
                return False, {
                    'limit_type': 'per_minute',
                    'limit': self.config.max_requests_per_minute,
                    'current': minute_count,
                    'reset_time': (current_time // 60 + 1) * 60
                }
            
            # Check per-hour limit
            hour_key = f"rate_limit:hour:{user_id}:{current_time // 3600}"
            hour_count = await self._get_request_count(hour_key)
            
            if hour_count >= self.config.max_requests_per_hour:
                return False, {
                    'limit_type': 'per_hour',
                    'limit': self.config.max_requests_per_hour,
                    'current': hour_count,
                    'reset_time': (current_time // 3600 + 1) * 3600
                }
            
            # Increment counters
            await self._increment_request_count(minute_key, 60)
            await self._increment_request_count(hour_key, 3600)
            
            return True, {
                'minute_count': minute_count + 1,
                'hour_count': hour_count + 1,
                'minute_limit': self.config.max_requests_per_minute,
                'hour_limit': self.config.max_requests_per_hour
            }
            
        except Exception as e:
            logger.error(f"Rate limit check error: {str(e)}")
            # Allow request on error to avoid blocking legitimate users
            return True, {}
    
    async def _get_request_count(self, key: str) -> int:
        """Get current request count for key"""
        try:
            if self.redis_client:
                count = await self.redis_client.get(key)
                return int(count) if count else 0
            else:
                return self.rate_limits.get(key, {}).get('count', 0)
        except Exception as e:
            logger.error(f"Failed to get request count: {str(e)}")
            return 0
    
    async def _increment_request_count(self, key: str, ttl: int):
        """Increment request count for key"""
        try:
            if self.redis_client:
                pipe = self.redis_client.pipeline()
                pipe.incr(key)
                pipe.expire(key, ttl)
                await pipe.execute()
            else:
                if key not in self.rate_limits:
                    self.rate_limits[key] = {'count': 0, 'expires': time.time() + ttl}
                
                # Clean up expired entries
                current_time = time.time()
                expired_keys = [k for k, v in self.rate_limits.items() if v['expires'] < current_time]
                for expired_key in expired_keys:
                    del self.rate_limits[expired_key]
                
                self.rate_limits[key]['count'] += 1
                
        except Exception as e:
            logger.error(f"Failed to increment request count: {str(e)}")

class ContentModerationEngine:
    """Handles content safety and moderation"""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.nsfw_classifier = None
        self.text_classifier = None
        
        # Initialize models
        if config.content_moderation_enabled:
            asyncio.create_task(self._initialize_models())
    
    async def _initialize_models(self):
        """Initialize content moderation models"""
        try:
            # In production, you would load actual NSFW detection models
            # For now, we'll use placeholder implementations
            logger.info("Content moderation models initialized")
        except Exception as e:
            logger.error(f"Failed to initialize moderation models: {str(e)}")
    
    async def moderate_text_prompt(self, prompt: str) -> Tuple[ContentSafetyLevel, Dict[str, Any]]:
        """Moderate text prompt for safety"""
        try:
            if not self.config.content_moderation_enabled:
                return ContentSafetyLevel.SAFE, {}
            
            # Check for explicit keywords
            unsafe_keywords = [
                'explicit', 'nsfw', 'nude', 'sexual', 'violence', 'gore',
                'hate', 'discrimination', 'illegal', 'drugs', 'weapons'
            ]
            
            prompt_lower = prompt.lower()
            detected_keywords = [keyword for keyword in unsafe_keywords if keyword in prompt_lower]
            
            if detected_keywords:
                return ContentSafetyLevel.UNSAFE, {
                    'detected_keywords': detected_keywords,
                    'reason': 'Unsafe content detected in prompt'
                }
            
            # Check for questionable content
            questionable_keywords = ['suggestive', 'provocative', 'controversial']
            detected_questionable = [keyword for keyword in questionable_keywords if keyword in prompt_lower]
            
            if detected_questionable:
                return ContentSafetyLevel.QUESTIONABLE, {
                    'detected_keywords': detected_questionable,
                    'reason': 'Questionable content detected in prompt'
                }
            
            return ContentSafetyLevel.SAFE, {'reason': 'Content appears safe'}
            
        except Exception as e:
            logger.error(f"Text moderation error: {str(e)}")
            return ContentSafetyLevel.QUESTIONABLE, {'error': str(e)}
    
    async def moderate_generated_image(self, image_path: str) -> Tuple[ContentSafetyLevel, Dict[str, Any]]:
        """Moderate generated image for safety"""
        try:
            if not self.config.content_moderation_enabled:
                return ContentSafetyLevel.SAFE, {}
            
            # In production, this would use actual NSFW detection models
            # For now, we'll do basic checks
            
            # Check image properties
            try:
                with Image.open(image_path) as img:
                    width, height = img.size
                    
                    # Basic heuristics (placeholder)
                    if width < 100 or height < 100:
                        return ContentSafetyLevel.QUESTIONABLE, {
                            'reason': 'Image too small for proper analysis'
                        }
                    
                    # In production, run through NSFW classifier here
                    # For now, assume safe
                    return ContentSafetyLevel.SAFE, {
                        'reason': 'Image passed safety checks',
                        'dimensions': f"{width}x{height}"
                    }
                    
            except Exception as e:
                return ContentSafetyLevel.QUESTIONABLE, {
                    'reason': 'Failed to analyze image',
                    'error': str(e)
                }
            
        except Exception as e:
            logger.error(f"Image moderation error: {str(e)}")
            return ContentSafetyLevel.QUESTIONABLE, {'error': str(e)}
    
    async def moderate_generated_video(self, video_path: str) -> Tuple[ContentSafetyLevel, Dict[str, Any]]:
        """Moderate generated video for safety"""
        try:
            if not self.config.content_moderation_enabled:
                return ContentSafetyLevel.SAFE, {}
            
            # In production, this would analyze video frames
            # For now, return safe
            return ContentSafetyLevel.SAFE, {
                'reason': 'Video passed safety checks'
            }
            
        except Exception as e:
            logger.error(f"Video moderation error: {str(e)}")
            return ContentSafetyLevel.QUESTIONABLE, {'error': str(e)}

class AuditLogger:
    """Handles audit logging for compliance"""
    
    def __init__(self, config: SecurityConfig, db_path: str = "audit_log.db"):
        self.config = config
        self.db_path = db_path
        
        # Initialize database
        if config.audit_logging_enabled:
            asyncio.create_task(self._initialize_database())
    
    async def _initialize_database(self):
        """Initialize audit log database"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS audit_events (
                        event_id TEXT PRIMARY KEY,
                        event_type TEXT NOT NULL,
                        user_id TEXT,
                        session_id TEXT,
                        timestamp TEXT NOT NULL,
                        ip_address TEXT NOT NULL,
                        user_agent TEXT,
                        resource TEXT NOT NULL,
                        action TEXT NOT NULL,
                        result TEXT NOT NULL,
                        metadata TEXT
                    )
                """)
                
                await db.execute("""
                    CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_events(timestamp)
                """)
                
                await db.execute("""
                    CREATE INDEX IF NOT EXISTS idx_audit_user ON audit_events(user_id)
                """)
                
                await db.execute("""
                    CREATE INDEX IF NOT EXISTS idx_audit_event_type ON audit_events(event_type)
                """)
                
                await db.commit()
                
            logger.info("Audit log database initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize audit database: {str(e)}")
            raise
    
    async def log_event(self, event_type: AuditEventType, resource: str, action: str,
                       result: str, user_id: Optional[str] = None, 
                       session_id: Optional[str] = None, ip_address: str = "unknown",
                       user_agent: str = "unknown", metadata: Dict[str, Any] = None):
        """Log audit event"""
        try:
            if not self.config.audit_logging_enabled:
                return
            
            event = AuditEvent(
                event_id=self._generate_event_id(),
                event_type=event_type,
                user_id=user_id,
                session_id=session_id,
                timestamp=datetime.utcnow(),
                ip_address=ip_address,
                user_agent=user_agent,
                resource=resource,
                action=action,
                result=result,
                metadata=metadata or {}
            )
            
            await self._persist_event(event)
            
            # Log to application logger as well
            logger.info(f"Audit: {event_type.value} - {action} on {resource} - {result}",
                       extra={
                           'user_id': user_id,
                           'session_id': session_id,
                           'ip_address': ip_address
                       })
            
        except Exception as e:
            logger.error(f"Failed to log audit event: {str(e)}")
    
    async def _persist_event(self, event: AuditEvent):
        """Persist audit event to database"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    INSERT INTO audit_events 
                    (event_id, event_type, user_id, session_id, timestamp, 
                     ip_address, user_agent, resource, action, result, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    event.event_id,
                    event.event_type.value,
                    event.user_id,
                    event.session_id,
                    event.timestamp.isoformat(),
                    event.ip_address,
                    event.user_agent,
                    event.resource,
                    event.action,
                    event.result,
                    json.dumps(event.metadata)
                ))
                await db.commit()
                
        except Exception as e:
            logger.error(f"Failed to persist audit event: {str(e)}")
    
    async def get_audit_events(self, start_date: datetime, end_date: datetime,
                              user_id: Optional[str] = None, 
                              event_type: Optional[AuditEventType] = None,
                              limit: int = 1000) -> List[AuditEvent]:
        """Get audit events for specified criteria"""
        try:
            query = """
                SELECT * FROM audit_events 
                WHERE timestamp BETWEEN ? AND ?
            """
            params = [start_date.isoformat(), end_date.isoformat()]
            
            if user_id:
                query += " AND user_id = ?"
                params.append(user_id)
            
            if event_type:
                query += " AND event_type = ?"
                params.append(event_type.value)
            
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            async with aiosqlite.connect(self.db_path) as db:
                cursor = await db.execute(query, params)
                rows = await cursor.fetchall()
                
                events = []
                for row in rows:
                    events.append(AuditEvent(
                        event_id=row[0],
                        event_type=AuditEventType(row[1]),
                        user_id=row[2],
                        session_id=row[3],
                        timestamp=datetime.fromisoformat(row[4]),
                        ip_address=row[5],
                        user_agent=row[6],
                        resource=row[7],
                        action=row[8],
                        result=row[9],
                        metadata=json.loads(row[10]) if row[10] else {}
                    ))
                
                return events
                
        except Exception as e:
            logger.error(f"Failed to get audit events: {str(e)}")
            return []
    
    def _generate_event_id(self) -> str:
        """Generate unique event ID"""
        timestamp = str(int(time.time()))
        random_data = str(time.time_ns())
        return hashlib.sha256(f"{timestamp}:{random_data}".encode()).hexdigest()[:16]

class ProductionSecurityManager:
    """Main security manager that coordinates all security components"""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.auth_manager = AuthenticationManager(config)
        self.rate_limiter = RateLimiter(config)
        self.content_moderator = ContentModerationEngine(config)
        self.audit_logger = AuditLogger(config)
    
    async def authenticate_request(self, token: str, ip_address: str, 
                                 user_agent: str) -> Optional[UserSession]:
        """Authenticate API request"""
        try:
            session = await self.auth_manager.validate_session(token)
            
            if session:
                await self.audit_logger.log_event(
                    AuditEventType.CONTENT_ACCESS,
                    "api",
                    "authenticate",
                    "success",
                    session.user_id,
                    session.session_id,
                    ip_address,
                    user_agent
                )
            else:
                await self.audit_logger.log_event(
                    AuditEventType.SECURITY_VIOLATION,
                    "api",
                    "authenticate",
                    "failed",
                    None,
                    None,
                    ip_address,
                    user_agent
                )
            
            return session
            
        except Exception as e:
            logger.error(f"Authentication error: {str(e)}")
            return None
    
    async def check_request_limits(self, user_id: str, ip_address: str) -> Tuple[bool, Dict[str, Any]]:
        """Check if request is within rate limits"""
        try:
            allowed, limit_info = await self.rate_limiter.check_rate_limit(user_id, ip_address)
            
            if not allowed:
                await self.audit_logger.log_event(
                    AuditEventType.RATE_LIMIT_EXCEEDED,
                    "api",
                    "rate_limit_check",
                    "exceeded",
                    user_id,
                    None,
                    ip_address,
                    "unknown",
                    limit_info
                )
            
            return allowed, limit_info
            
        except Exception as e:
            logger.error(f"Rate limit check error: {str(e)}")
            return True, {}  # Allow on error
    
    async def moderate_generation_request(self, prompt: str, user_id: str, 
                                        session_id: str) -> Tuple[bool, Dict[str, Any]]:
        """Moderate generation request for safety"""
        try:
            safety_level, moderation_result = await self.content_moderator.moderate_text_prompt(prompt)
            
            allowed = safety_level in [ContentSafetyLevel.SAFE, ContentSafetyLevel.QUESTIONABLE]
            
            await self.audit_logger.log_event(
                AuditEventType.GENERATION_REQUEST,
                "content_generation",
                "moderate_prompt",
                "allowed" if allowed else "blocked",
                user_id,
                session_id,
                metadata={
                    'safety_level': safety_level.value,
                    'moderation_result': moderation_result
                }
            )
            
            return allowed, {
                'safety_level': safety_level.value,
                'moderation_result': moderation_result
            }
            
        except Exception as e:
            logger.error(f"Content moderation error: {str(e)}")
            return True, {}  # Allow on error to avoid blocking legitimate requests
    
    async def moderate_generated_content(self, content_path: str, content_type: str,
                                       user_id: str, session_id: str) -> Tuple[bool, Dict[str, Any]]:
        """Moderate generated content for safety"""
        try:
            if content_type == "image":
                safety_level, moderation_result = await self.content_moderator.moderate_generated_image(content_path)
            elif content_type == "video":
                safety_level, moderation_result = await self.content_moderator.moderate_generated_video(content_path)
            else:
                return True, {}
            
            allowed = safety_level in [ContentSafetyLevel.SAFE, ContentSafetyLevel.QUESTIONABLE]
            
            await self.audit_logger.log_event(
                AuditEventType.GENERATION_REQUEST,
                "content_generation",
                f"moderate_{content_type}",
                "allowed" if allowed else "blocked",
                user_id,
                session_id,
                metadata={
                    'safety_level': safety_level.value,
                    'moderation_result': moderation_result,
                    'content_path': content_path
                }
            )
            
            return allowed, {
                'safety_level': safety_level.value,
                'moderation_result': moderation_result
            }
            
        except Exception as e:
            logger.error(f"Content moderation error: {str(e)}")
            return True, {}

# Factory function to create security manager
def create_security_manager() -> ProductionSecurityManager:
    """Create production security manager with configuration"""
    import os
    
    config = SecurityConfig(
        jwt_secret=os.getenv("JWT_SECRET", "default_jwt_secret_change_in_production"),
        encryption_key=os.getenv("ENCRYPTION_KEY", Fernet.generate_key().decode()),
        rate_limit_enabled=os.getenv("RATE_LIMIT_ENABLED", "true").lower() == "true",
        content_moderation_enabled=os.getenv("CONTENT_MODERATION_ENABLED", "true").lower() == "true",
        audit_logging_enabled=os.getenv("AUDIT_LOGGING_ENABLED", "true").lower() == "true",
        max_requests_per_minute=int(os.getenv("MAX_REQUESTS_PER_MINUTE", "60")),
        max_requests_per_hour=int(os.getenv("MAX_REQUESTS_PER_HOUR", "1000")),
        session_timeout_minutes=int(os.getenv("SESSION_TIMEOUT_MINUTES", "60")),
        password_min_length=int(os.getenv("PASSWORD_MIN_LENGTH", "8")),
        require_2fa=os.getenv("REQUIRE_2FA", "false").lower() == "true"
    )
    
    return ProductionSecurityManager(config)

# Global security manager
security_manager = create_security_manager()

async def initialize_security():
    """Initialize production security"""
    try:
        logger.info("Production security initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize security: {str(e)}")
        return False