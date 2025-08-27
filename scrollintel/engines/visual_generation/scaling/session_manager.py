"""
Distributed session management for visual generation.
Handles user sessions, request tracking, and state management across multiple workers.
"""

import asyncio
import logging
import json
import uuid
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
import redis.asyncio as redis
from redis.asyncio import RedisCluster

logger = logging.getLogger(__name__)


class SessionStatus(Enum):
    ACTIVE = "active"
    IDLE = "idle"
    EXPIRED = "expired"
    TERMINATED = "terminated"


class RequestStatus(Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class GenerationRequest:
    """Represents a visual generation request."""
    id: str
    user_id: str
    session_id: str
    request_type: str  # 'image' or 'video'
    prompt: str
    parameters: Dict[str, Any]
    status: RequestStatus = RequestStatus.QUEUED
    worker_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: float = 0.0
    result_urls: List[str] = field(default_factory=list)
    error_message: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    priority: int = 0  # Higher number = higher priority
    
    @property
    def processing_time(self) -> Optional[float]:
        """Get processing time in seconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None
    
    @property
    def total_time(self) -> float:
        """Get total time since creation in seconds."""
        end_time = self.completed_at or datetime.now()
        return (end_time - self.created_at).total_seconds()


@dataclass
class UserSession:
    """Represents a user session for visual generation."""
    id: str
    user_id: str
    created_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    status: SessionStatus = SessionStatus.ACTIVE
    active_requests: Set[str] = field(default_factory=set)
    completed_requests: Set[str] = field(default_factory=set)
    failed_requests: Set[str] = field(default_factory=set)
    total_requests: int = 0
    concurrent_limit: int = 5
    rate_limit_per_minute: int = 10
    rate_limit_window: List[datetime] = field(default_factory=list)
    preferences: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_expired(self) -> bool:
        """Check if session has expired (24 hours of inactivity)."""
        return datetime.now() > self.last_activity + timedelta(hours=24)
    
    @property
    def can_make_request(self) -> bool:
        """Check if user can make a new request based on limits."""
        # Check concurrent limit
        if len(self.active_requests) >= self.concurrent_limit:
            return False
        
        # Check rate limit
        now = datetime.now()
        minute_ago = now - timedelta(minutes=1)
        
        # Clean old entries
        self.rate_limit_window = [
            timestamp for timestamp in self.rate_limit_window
            if timestamp > minute_ago
        ]
        
        return len(self.rate_limit_window) < self.rate_limit_per_minute
    
    def record_request(self):
        """Record a new request for rate limiting."""
        self.rate_limit_window.append(datetime.now())
        self.total_requests += 1
        self.last_activity = datetime.now()


class DistributedSessionManager:
    """
    Manages user sessions and requests across distributed visual generation workers.
    """
    
    def __init__(self, redis_nodes: List[Dict[str, Any]] = None):
        self.redis_cluster: Optional[RedisCluster] = None
        self.local_sessions: Dict[str, UserSession] = {}
        self.local_requests: Dict[str, GenerationRequest] = {}
        
        # Default Redis cluster configuration
        if redis_nodes is None:
            redis_nodes = [
                {"host": "localhost", "port": 7000},
                {"host": "localhost", "port": 7001},
                {"host": "localhost", "port": 7002}
            ]
        self.redis_nodes = redis_nodes
        
        # Configuration
        self.session_ttl = 24 * 3600  # 24 hours
        self.request_ttl = 7 * 24 * 3600  # 7 days
        self.cleanup_interval = 300  # 5 minutes
        
        # Background tasks
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False
    
    async def initialize(self):
        """Initialize the session manager."""
        try:
            # Initialize Redis cluster
            self.redis_cluster = RedisCluster(
                startup_nodes=self.redis_nodes,
                decode_responses=True,
                skip_full_coverage_check=True,
                health_check_interval=30
            )
            
            # Test connection
            await self.redis_cluster.ping()
            logger.info("Connected to Redis cluster for session management")
            
        except Exception as e:
            logger.warning(f"Failed to connect to Redis cluster: {e}. Using local storage only.")
            self.redis_cluster = None
        
        # Start background cleanup
        self._running = True
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
    
    async def close(self):
        """Close session manager."""
        self._running = False
        
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        if self.redis_cluster:
            await self.redis_cluster.close()
    
    async def create_session(self, user_id: str, preferences: Dict[str, Any] = None) -> UserSession:
        """Create a new user session."""
        session_id = str(uuid.uuid4())
        
        session = UserSession(
            id=session_id,
            user_id=user_id,
            preferences=preferences or {}
        )
        
        # Store locally
        self.local_sessions[session_id] = session
        
        # Store in Redis
        if self.redis_cluster:
            await self._store_session_redis(session)
        
        logger.info(f"Created session {session_id} for user {user_id}")
        return session
    
    async def get_session(self, session_id: str) -> Optional[UserSession]:
        """Get session by ID."""
        # Try local cache first
        if session_id in self.local_sessions:
            session = self.local_sessions[session_id]
            if not session.is_expired:
                return session
            else:
                # Remove expired session
                del self.local_sessions[session_id]
        
        # Try Redis
        if self.redis_cluster:
            session = await self._get_session_redis(session_id)
            if session and not session.is_expired:
                # Cache locally
                self.local_sessions[session_id] = session
                return session
        
        return None
    
    async def update_session(self, session: UserSession):
        """Update session data."""
        session.last_activity = datetime.now()
        
        # Update locally
        self.local_sessions[session.id] = session
        
        # Update in Redis
        if self.redis_cluster:
            await self._store_session_redis(session)
    
    async def terminate_session(self, session_id: str):
        """Terminate a session."""
        session = await self.get_session(session_id)
        if session:
            session.status = SessionStatus.TERMINATED
            
            # Cancel active requests
            for request_id in list(session.active_requests):
                await self.cancel_request(request_id)
            
            await self.update_session(session)
            logger.info(f"Terminated session {session_id}")
    
    async def create_request(self, session_id: str, request_type: str, prompt: str,
                           parameters: Dict[str, Any], priority: int = 0) -> Optional[GenerationRequest]:
        """Create a new generation request."""
        session = await self.get_session(session_id)
        if not session:
            logger.error(f"Session {session_id} not found")
            return None
        
        # Check if user can make request
        if not session.can_make_request():
            logger.warning(f"Request denied for session {session_id} - limits exceeded")
            return None
        
        request_id = str(uuid.uuid4())
        
        request = GenerationRequest(
            id=request_id,
            user_id=session.user_id,
            session_id=session_id,
            request_type=request_type,
            prompt=prompt,
            parameters=parameters,
            priority=priority
        )
        
        # Update session
        session.record_request()
        session.active_requests.add(request_id)
        await self.update_session(session)
        
        # Store request
        self.local_requests[request_id] = request
        
        if self.redis_cluster:
            await self._store_request_redis(request)
        
        logger.info(f"Created request {request_id} for session {session_id}")
        return request
    
    async def get_request(self, request_id: str) -> Optional[GenerationRequest]:
        """Get request by ID."""
        # Try local cache first
        if request_id in self.local_requests:
            return self.local_requests[request_id]
        
        # Try Redis
        if self.redis_cluster:
            request = await self._get_request_redis(request_id)
            if request:
                # Cache locally
                self.local_requests[request_id] = request
                return request
        
        return None
    
    async def update_request(self, request: GenerationRequest):
        """Update request data."""
        # Update locally
        self.local_requests[request.id] = request
        
        # Update in Redis
        if self.redis_cluster:
            await self._store_request_redis(request)
        
        # Update session if status changed
        if request.status in [RequestStatus.COMPLETED, RequestStatus.FAILED, RequestStatus.CANCELLED]:
            session = await self.get_session(request.session_id)
            if session:
                session.active_requests.discard(request.id)
                
                if request.status == RequestStatus.COMPLETED:
                    session.completed_requests.add(request.id)
                elif request.status == RequestStatus.FAILED:
                    session.failed_requests.add(request.id)
                
                await self.update_session(session)
    
    async def start_request_processing(self, request_id: str, worker_id: str):
        """Mark request as started processing."""
        request = await self.get_request(request_id)
        if request:
            request.status = RequestStatus.PROCESSING
            request.worker_id = worker_id
            request.started_at = datetime.now()
            await self.update_request(request)
    
    async def update_request_progress(self, request_id: str, progress: float):
        """Update request progress."""
        request = await self.get_request(request_id)
        if request:
            request.progress = max(0.0, min(1.0, progress))
            await self.update_request(request)
    
    async def complete_request(self, request_id: str, result_urls: List[str]):
        """Mark request as completed."""
        request = await self.get_request(request_id)
        if request:
            request.status = RequestStatus.COMPLETED
            request.completed_at = datetime.now()
            request.result_urls = result_urls
            request.progress = 1.0
            await self.update_request(request)
    
    async def fail_request(self, request_id: str, error_message: str, retry: bool = True):
        """Mark request as failed."""
        request = await self.get_request(request_id)
        if request:
            request.error_message = error_message
            
            if retry and request.retry_count < request.max_retries:
                # Retry the request
                request.retry_count += 1
                request.status = RequestStatus.QUEUED
                request.worker_id = None
                request.started_at = None
                request.progress = 0.0
                logger.info(f"Retrying request {request_id} (attempt {request.retry_count})")
            else:
                # Mark as failed
                request.status = RequestStatus.FAILED
                request.completed_at = datetime.now()
            
            await self.update_request(request)
    
    async def cancel_request(self, request_id: str):
        """Cancel a request."""
        request = await self.get_request(request_id)
        if request:
            request.status = RequestStatus.CANCELLED
            request.completed_at = datetime.now()
            await self.update_request(request)
    
    async def get_user_requests(self, user_id: str, status: Optional[RequestStatus] = None,
                               limit: int = 50) -> List[GenerationRequest]:
        """Get requests for a user."""
        requests = []
        
        # Search local requests
        for request in self.local_requests.values():
            if request.user_id == user_id:
                if status is None or request.status == status:
                    requests.append(request)
        
        # Search Redis (simplified - in production, maintain user request index)
        if self.redis_cluster:
            try:
                # This is a simplified approach - in production, maintain proper indexes
                pattern = f"visual_gen:request:*"
                keys = await self.redis_cluster.keys(pattern)
                
                for key in keys[:200]:  # Limit search
                    try:
                        request_data = await self.redis_cluster.get(key)
                        if request_data:
                            request = GenerationRequest(**json.loads(request_data))
                            if request.user_id == user_id:
                                if status is None or request.status == status:
                                    requests.append(request)
                    except Exception:
                        continue
            except Exception as e:
                logger.error(f"Error searching Redis for user requests: {e}")
        
        # Sort by creation time (newest first) and limit
        requests.sort(key=lambda r: r.created_at, reverse=True)
        return requests[:limit]
    
    async def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status."""
        queued_count = 0
        processing_count = 0
        
        for request in self.local_requests.values():
            if request.status == RequestStatus.QUEUED:
                queued_count += 1
            elif request.status == RequestStatus.PROCESSING:
                processing_count += 1
        
        return {
            'queued_requests': queued_count,
            'processing_requests': processing_count,
            'active_sessions': len([s for s in self.local_sessions.values() 
                                  if s.status == SessionStatus.ACTIVE]),
            'total_sessions': len(self.local_sessions)
        }
    
    async def _store_session_redis(self, session: UserSession):
        """Store session in Redis."""
        try:
            key = f"visual_gen:session:{session.id}"
            # Convert sets to lists for JSON serialization
            session_dict = asdict(session)
            session_dict['active_requests'] = list(session.active_requests)
            session_dict['completed_requests'] = list(session.completed_requests)
            session_dict['failed_requests'] = list(session.failed_requests)
            session_dict['rate_limit_window'] = [
                ts.isoformat() for ts in session.rate_limit_window
            ]
            session_dict['created_at'] = session.created_at.isoformat()
            session_dict['last_activity'] = session.last_activity.isoformat()
            
            await self.redis_cluster.setex(
                key,
                self.session_ttl,
                json.dumps(session_dict)
            )
        except Exception as e:
            logger.error(f"Error storing session in Redis: {e}")
    
    async def _get_session_redis(self, session_id: str) -> Optional[UserSession]:
        """Get session from Redis."""
        try:
            key = f"visual_gen:session:{session_id}"
            session_data = await self.redis_cluster.get(key)
            
            if session_data:
                session_dict = json.loads(session_data)
                
                # Convert back from JSON format
                session_dict['active_requests'] = set(session_dict['active_requests'])
                session_dict['completed_requests'] = set(session_dict['completed_requests'])
                session_dict['failed_requests'] = set(session_dict['failed_requests'])
                session_dict['rate_limit_window'] = [
                    datetime.fromisoformat(ts) for ts in session_dict['rate_limit_window']
                ]
                session_dict['created_at'] = datetime.fromisoformat(session_dict['created_at'])
                session_dict['last_activity'] = datetime.fromisoformat(session_dict['last_activity'])
                session_dict['status'] = SessionStatus(session_dict['status'])
                
                return UserSession(**session_dict)
        except Exception as e:
            logger.error(f"Error getting session from Redis: {e}")
        
        return None
    
    async def _store_request_redis(self, request: GenerationRequest):
        """Store request in Redis."""
        try:
            key = f"visual_gen:request:{request.id}"
            request_dict = asdict(request)
            request_dict['created_at'] = request.created_at.isoformat()
            if request.started_at:
                request_dict['started_at'] = request.started_at.isoformat()
            if request.completed_at:
                request_dict['completed_at'] = request.completed_at.isoformat()
            
            await self.redis_cluster.setex(
                key,
                self.request_ttl,
                json.dumps(request_dict)
            )
        except Exception as e:
            logger.error(f"Error storing request in Redis: {e}")
    
    async def _get_request_redis(self, request_id: str) -> Optional[GenerationRequest]:
        """Get request from Redis."""
        try:
            key = f"visual_gen:request:{request_id}"
            request_data = await self.redis_cluster.get(key)
            
            if request_data:
                request_dict = json.loads(request_data)
                
                # Convert back from JSON format
                request_dict['created_at'] = datetime.fromisoformat(request_dict['created_at'])
                if request_dict.get('started_at'):
                    request_dict['started_at'] = datetime.fromisoformat(request_dict['started_at'])
                if request_dict.get('completed_at'):
                    request_dict['completed_at'] = datetime.fromisoformat(request_dict['completed_at'])
                request_dict['status'] = RequestStatus(request_dict['status'])
                
                return GenerationRequest(**request_dict)
        except Exception as e:
            logger.error(f"Error getting request from Redis: {e}")
        
        return None
    
    async def _cleanup_loop(self):
        """Background cleanup of expired sessions and old requests."""
        while self._running:
            try:
                await self._cleanup_expired_sessions()
                await self._cleanup_old_requests()
                await asyncio.sleep(self.cleanup_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(60)  # Wait before retry
    
    async def _cleanup_expired_sessions(self):
        """Clean up expired sessions."""
        expired_sessions = []
        
        for session_id, session in self.local_sessions.items():
            if session.is_expired:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            del self.local_sessions[session_id]
            logger.info(f"Cleaned up expired session {session_id}")
    
    async def _cleanup_old_requests(self):
        """Clean up old completed/failed requests."""
        cutoff_time = datetime.now() - timedelta(days=7)
        old_requests = []
        
        for request_id, request in self.local_requests.items():
            if (request.status in [RequestStatus.COMPLETED, RequestStatus.FAILED, RequestStatus.CANCELLED] and
                request.completed_at and request.completed_at < cutoff_time):
                old_requests.append(request_id)
        
        for request_id in old_requests:
            del self.local_requests[request_id]
            logger.info(f"Cleaned up old request {request_id}")


# Global session manager instance
session_manager = DistributedSessionManager()