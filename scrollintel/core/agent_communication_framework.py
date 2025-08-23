"""
Agent Communication Framework

Secure, encrypted messaging system for agent-to-agent communication with
collaboration session management, distributed state synchronization, and
resource locking capabilities.
"""

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Set
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import hashlib
import logging
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_

from ..models.agent_communication_models import (
    AgentMessage, CollaborationSession, SessionParticipant, SessionMessage,
    ResourceLock, DistributedState, MessageType, MessagePriority,
    CollaborationStatus, ResourceLockStatus, SecurityLevel,
    AgentMessageCreate, AgentMessageResponse, CollaborationSessionCreate,
    CollaborationSessionResponse, ResourceLockRequest, ResourceLockResponse,
    StateUpdateRequest, StateResponse, MessageDeliveryStatus
)


logger = logging.getLogger(__name__)


class EncryptionManager:
    """Manages encryption and decryption of agent communications"""
    
    def __init__(self, master_key: str):
        self.master_key = master_key.encode()
        self._key_cache: Dict[str, Fernet] = {}
    
    def _derive_key(self, agent_id: str, salt: bytes = None) -> Fernet:
        """Derive encryption key for specific agent"""
        if salt is None:
            salt = hashlib.sha256(agent_id.encode()).digest()[:16]
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(self.master_key))
        return Fernet(key)
    
    def get_agent_cipher(self, agent_id: str) -> Fernet:
        """Get or create cipher for agent"""
        if agent_id not in self._key_cache:
            self._key_cache[agent_id] = self._derive_key(agent_id)
        return self._key_cache[agent_id]
    
    def encrypt_message(self, content: Dict[str, Any], from_agent: str, to_agent: str) -> tuple[str, str]:
        """Encrypt message content and return encrypted data with hash"""
        # Use sender's cipher for encryption
        cipher = self.get_agent_cipher(from_agent)
        
        # Serialize content
        content_json = json.dumps(content, sort_keys=True)
        content_bytes = content_json.encode('utf-8')
        
        # Encrypt
        encrypted_data = cipher.encrypt(content_bytes)
        encrypted_str = base64.urlsafe_b64encode(encrypted_data).decode('utf-8')
        
        # Generate hash
        content_hash = hashlib.sha256(content_bytes).hexdigest()
        
        return encrypted_str, content_hash
    
    def decrypt_message(self, encrypted_content: str, from_agent: str) -> Dict[str, Any]:
        """Decrypt message content"""
        cipher = self.get_agent_cipher(from_agent)
        
        # Decode and decrypt
        encrypted_data = base64.urlsafe_b64decode(encrypted_content.encode('utf-8'))
        decrypted_bytes = cipher.decrypt(encrypted_data)
        
        # Deserialize
        content_json = decrypted_bytes.decode('utf-8')
        return json.loads(content_json)
    
    def verify_message_integrity(self, content: Dict[str, Any], content_hash: str) -> bool:
        """Verify message integrity using hash"""
        content_json = json.dumps(content, sort_keys=True)
        calculated_hash = hashlib.sha256(content_json.encode('utf-8')).hexdigest()
        return calculated_hash == content_hash


class MessageQueue:
    """Manages message queuing and delivery"""
    
    def __init__(self):
        self._queues: Dict[str, asyncio.Queue] = {}
        self._subscribers: Dict[str, Set[Callable]] = {}
        self._delivery_callbacks: Dict[str, Callable] = {}
    
    def get_agent_queue(self, agent_id: str) -> asyncio.Queue:
        """Get or create message queue for agent"""
        if agent_id not in self._queues:
            self._queues[agent_id] = asyncio.Queue()
        return self._queues[agent_id]
    
    async def enqueue_message(self, agent_id: str, message: AgentMessage):
        """Add message to agent's queue"""
        queue = self.get_agent_queue(agent_id)
        await queue.put(message)
        
        # Notify subscribers
        if agent_id in self._subscribers:
            for callback in self._subscribers[agent_id]:
                try:
                    await callback(message)
                except Exception as e:
                    logger.error(f"Error in message subscriber callback: {e}")
    
    async def dequeue_message(self, agent_id: str, timeout: float = None) -> Optional[AgentMessage]:
        """Get next message from agent's queue"""
        queue = self.get_agent_queue(agent_id)
        try:
            if timeout:
                return await asyncio.wait_for(queue.get(), timeout=timeout)
            else:
                return await queue.get()
        except asyncio.TimeoutError:
            return None
    
    def subscribe_to_messages(self, agent_id: str, callback: Callable):
        """Subscribe to messages for an agent"""
        if agent_id not in self._subscribers:
            self._subscribers[agent_id] = set()
        self._subscribers[agent_id].add(callback)
    
    def unsubscribe_from_messages(self, agent_id: str, callback: Callable):
        """Unsubscribe from messages for an agent"""
        if agent_id in self._subscribers:
            self._subscribers[agent_id].discard(callback)


class AgentCommunicationFramework:
    """Main agent communication framework"""
    
    def __init__(self, db_session: Session, encryption_key: str):
        self.db = db_session
        self.encryption_manager = EncryptionManager(encryption_key)
        self.message_queue = MessageQueue()
        self._active_sessions: Dict[str, CollaborationSession] = {}
        self._resource_locks: Dict[str, ResourceLock] = {}
        self._state_cache: Dict[str, Dict[str, Any]] = {}
        
    # Message Management
    async def send_message(
        self, 
        from_agent_id: str, 
        message_request: AgentMessageCreate
    ) -> AgentMessageResponse:
        """Send encrypted message between agents"""
        try:
            # Generate message ID
            message_id = str(uuid.uuid4())
            
            # Encrypt content
            encrypted_content, content_hash = self.encryption_manager.encrypt_message(
                message_request.content, from_agent_id, message_request.to_agent_id
            )
            
            # Calculate expiration
            expires_at = None
            if message_request.expires_in_seconds:
                expires_at = datetime.utcnow() + timedelta(seconds=message_request.expires_in_seconds)
            
            # Create message record
            message = AgentMessage(
                id=message_id,
                from_agent_id=from_agent_id,
                to_agent_id=message_request.to_agent_id,
                message_type=message_request.message_type.value,
                priority=message_request.priority.value,
                security_level=message_request.security_level.value,
                encrypted_content=encrypted_content,
                content_hash=content_hash,
                encryption_key_id=from_agent_id,  # Use sender's key
                correlation_id=message_request.correlation_id,
                session_id=message_request.session_id,
                reply_to=message_request.reply_to,
                expires_at=expires_at,
                sent_at=datetime.utcnow()
            )
            
            # Save to database
            self.db.add(message)
            self.db.commit()
            
            # Queue for delivery
            await self.message_queue.enqueue_message(message_request.to_agent_id, message)
            
            logger.info(f"Message {message_id} sent from {from_agent_id} to {message_request.to_agent_id}")
            
            return AgentMessageResponse.from_orm(message)
            
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            raise
    
    async def receive_messages(
        self, 
        agent_id: str, 
        message_types: Optional[List[MessageType]] = None,
        timeout: float = None
    ) -> List[Dict[str, Any]]:
        """Receive and decrypt messages for an agent"""
        try:
            messages = []
            
            # Get messages from queue
            while True:
                message = await self.message_queue.dequeue_message(agent_id, timeout=0.1)
                if not message:
                    break
                
                # Filter by message type if specified
                if message_types and MessageType(message.message_type) not in message_types:
                    continue
                
                # Check expiration
                if message.expires_at and datetime.utcnow() > message.expires_at:
                    continue
                
                # Decrypt content
                try:
                    decrypted_content = self.encryption_manager.decrypt_message(
                        message.encrypted_content, message.from_agent_id
                    )
                    
                    # Verify integrity
                    if not self.encryption_manager.verify_message_integrity(
                        decrypted_content, message.content_hash
                    ):
                        logger.warning(f"Message integrity check failed for message {message.id}")
                        continue
                    
                    # Mark as delivered
                    message.is_delivered = True
                    message.delivered_at = datetime.utcnow()
                    self.db.commit()
                    
                    messages.append({
                        'id': message.id,
                        'from_agent_id': message.from_agent_id,
                        'message_type': message.message_type,
                        'priority': message.priority,
                        'content': decrypted_content,
                        'correlation_id': message.correlation_id,
                        'session_id': message.session_id,
                        'reply_to': message.reply_to,
                        'created_at': message.created_at,
                        'sent_at': message.sent_at
                    })
                    
                except Exception as e:
                    logger.error(f"Error decrypting message {message.id}: {e}")
                    continue
            
            return messages
            
        except Exception as e:
            logger.error(f"Error receiving messages for agent {agent_id}: {e}")
            raise
    
    async def acknowledge_message(self, agent_id: str, message_id: str) -> bool:
        """Acknowledge receipt of a message"""
        try:
            message = self.db.query(AgentMessage).filter(
                and_(
                    AgentMessage.id == message_id,
                    AgentMessage.to_agent_id == agent_id
                )
            ).first()
            
            if message:
                message.is_acknowledged = True
                message.acknowledged_at = datetime.utcnow()
                self.db.commit()
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error acknowledging message {message_id}: {e}")
            return False
    
    # Collaboration Session Management
    async def create_collaboration_session(
        self, 
        initiator_agent_id: str, 
        session_request: CollaborationSessionCreate
    ) -> CollaborationSessionResponse:
        """Create a new collaboration session"""
        try:
            session_id = str(uuid.uuid4())
            
            # Create session
            session = CollaborationSession(
                id=session_id,
                initiator_agent_id=initiator_agent_id,
                session_name=session_request.session_name,
                description=session_request.description,
                security_level=session_request.security_level.value,
                max_participants=session_request.max_participants,
                session_timeout=session_request.session_timeout,
                auto_cleanup=session_request.auto_cleanup,
                objective=session_request.objective,
                context={},
                shared_state={}
            )
            
            self.db.add(session)
            
            # Add initiator as participant
            initiator_participant = SessionParticipant(
                id=str(uuid.uuid4()),
                session_id=session_id,
                agent_id=initiator_agent_id,
                role="initiator",
                status="joined",
                can_invite_others=True,
                can_modify_state=True,
                can_end_session=True,
                joined_at=datetime.utcnow()
            )
            
            self.db.add(initiator_participant)
            
            # Invite other participants
            for agent_id in session_request.participants:
                if agent_id != initiator_agent_id:
                    participant = SessionParticipant(
                        id=str(uuid.uuid4()),
                        session_id=session_id,
                        agent_id=agent_id,
                        role="participant",
                        status="invited"
                    )
                    self.db.add(participant)
                    
                    # Send invitation message
                    invite_message = AgentMessageCreate(
                        to_agent_id=agent_id,
                        message_type=MessageType.COLLABORATION_INVITE,
                        content={
                            'session_id': session_id,
                            'session_name': session_request.session_name,
                            'initiator': initiator_agent_id,
                            'objective': session_request.objective
                        },
                        priority=MessagePriority.HIGH,
                        session_id=session_id
                    )
                    
                    await self.send_message(initiator_agent_id, invite_message)
            
            self.db.commit()
            
            # Cache active session
            self._active_sessions[session_id] = session
            
            logger.info(f"Collaboration session {session_id} created by {initiator_agent_id}")
            
            return CollaborationSessionResponse(
                id=session.id,
                initiator_agent_id=session.initiator_agent_id,
                session_name=session.session_name,
                description=session.description,
                status=CollaborationStatus(session.status),
                security_level=SecurityLevel(session.security_level),
                max_participants=session.max_participants,
                session_timeout=session.session_timeout,
                auto_cleanup=session.auto_cleanup,
                objective=session.objective,
                context=session.context,
                shared_state=session.shared_state,
                created_at=session.created_at,
                started_at=session.started_at,
                ended_at=session.ended_at,
                last_activity=session.last_activity,
                participant_count=len(session_request.participants) + 1
            )
            
        except Exception as e:
            logger.error(f"Error creating collaboration session: {e}")
            raise
    
    async def join_collaboration_session(self, agent_id: str, session_id: str) -> bool:
        """Join a collaboration session"""
        try:
            # Find participant record
            participant = self.db.query(SessionParticipant).filter(
                and_(
                    SessionParticipant.session_id == session_id,
                    SessionParticipant.agent_id == agent_id,
                    SessionParticipant.status == "invited"
                )
            ).first()
            
            if not participant:
                return False
            
            # Update participant status
            participant.status = "joined"
            participant.joined_at = datetime.utcnow()
            
            # Update session activity
            session = self.db.query(CollaborationSession).filter(
                CollaborationSession.id == session_id
            ).first()
            
            if session:
                session.last_activity = datetime.utcnow()
                if session.status == CollaborationStatus.PENDING.value:
                    session.status = CollaborationStatus.ACTIVE.value
                    session.started_at = datetime.utcnow()
            
            self.db.commit()
            
            # Send acceptance message to initiator
            accept_message = AgentMessageCreate(
                to_agent_id=session.initiator_agent_id,
                message_type=MessageType.COLLABORATION_ACCEPT,
                content={
                    'session_id': session_id,
                    'agent_id': agent_id
                },
                session_id=session_id
            )
            
            await self.send_message(agent_id, accept_message)
            
            logger.info(f"Agent {agent_id} joined collaboration session {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error joining collaboration session: {e}")
            return False
    
    async def leave_collaboration_session(self, agent_id: str, session_id: str) -> bool:
        """Leave a collaboration session"""
        try:
            participant = self.db.query(SessionParticipant).filter(
                and_(
                    SessionParticipant.session_id == session_id,
                    SessionParticipant.agent_id == agent_id,
                    SessionParticipant.status == "joined"
                )
            ).first()
            
            if not participant:
                return False
            
            participant.status = "left"
            participant.left_at = datetime.utcnow()
            
            # Update session activity
            session = self.db.query(CollaborationSession).filter(
                CollaborationSession.id == session_id
            ).first()
            
            if session:
                session.last_activity = datetime.utcnow()
                
                # Check if session should end
                active_participants = self.db.query(SessionParticipant).filter(
                    and_(
                        SessionParticipant.session_id == session_id,
                        SessionParticipant.status == "joined"
                    )
                ).count()
                
                if active_participants == 0:
                    session.status = CollaborationStatus.COMPLETED.value
                    session.ended_at = datetime.utcnow()
            
            self.db.commit()
            
            logger.info(f"Agent {agent_id} left collaboration session {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error leaving collaboration session: {e}")
            return False
    
    async def send_session_message(
        self, 
        agent_id: str, 
        session_id: str, 
        message_content: Dict[str, Any],
        message_type: MessageType = MessageType.NOTIFICATION
    ) -> bool:
        """Send message to all participants in a collaboration session"""
        try:
            # Verify agent is in session
            participant = self.db.query(SessionParticipant).filter(
                and_(
                    SessionParticipant.session_id == session_id,
                    SessionParticipant.agent_id == agent_id,
                    SessionParticipant.status == "joined"
                )
            ).first()
            
            if not participant:
                return False
            
            # Get all active participants
            participants = self.db.query(SessionParticipant).filter(
                and_(
                    SessionParticipant.session_id == session_id,
                    SessionParticipant.status == "joined",
                    SessionParticipant.agent_id != agent_id
                )
            ).all()
            
            # Send message to each participant
            for p in participants:
                message_request = AgentMessageCreate(
                    to_agent_id=p.agent_id,
                    message_type=message_type,
                    content=message_content,
                    session_id=session_id
                )
                
                await self.send_message(agent_id, message_request)
            
            # Store session message
            session_message = SessionMessage(
                id=str(uuid.uuid4()),
                session_id=session_id,
                sender_agent_id=agent_id,
                message_type=message_type.value,
                encrypted_content="",  # Will be encrypted
                content_hash="",
                sequence_number=self._get_next_sequence_number(session_id)
            )
            
            # Encrypt session message
            encrypted_content, content_hash = self.encryption_manager.encrypt_message(
                message_content, agent_id, session_id
            )
            session_message.encrypted_content = encrypted_content
            session_message.content_hash = content_hash
            
            self.db.add(session_message)
            self.db.commit()
            
            return True
            
        except Exception as e:
            logger.error(f"Error sending session message: {e}")
            return False
    
    def _get_next_sequence_number(self, session_id: str) -> int:
        """Get next sequence number for session message"""
        last_message = self.db.query(SessionMessage).filter(
            SessionMessage.session_id == session_id
        ).order_by(SessionMessage.sequence_number.desc()).first()
        
        return (last_message.sequence_number + 1) if last_message else 1
    
    # Distributed State Synchronization
    async def update_shared_state(
        self, 
        agent_id: str, 
        state_request: StateUpdateRequest,
        session_id: Optional[str] = None
    ) -> StateResponse:
        """Update distributed shared state"""
        try:
            state_id = f"{state_request.state_namespace}:{state_request.state_key}"
            
            # Check if state exists
            existing_state = self.db.query(DistributedState).filter(
                and_(
                    DistributedState.state_key == state_request.state_key,
                    DistributedState.state_namespace == state_request.state_namespace
                )
            ).first()
            
            if existing_state:
                # Handle version conflict
                if (state_request.expected_version and 
                    existing_state.state_version != state_request.expected_version):
                    
                    if state_request.conflict_resolution_strategy == "fail_on_conflict":
                        raise ValueError(f"Version conflict: expected {state_request.expected_version}, got {existing_state.state_version}")
                    elif state_request.conflict_resolution_strategy == "merge":
                        # Implement merge logic
                        merged_value = self._merge_state_values(
                            existing_state.state_value, 
                            state_request.state_value
                        )
                        state_request.state_value = merged_value
                
                # Update existing state
                existing_state.state_value = state_request.state_value
                existing_state.state_version += 1
                existing_state.last_modified_by = agent_id
                existing_state.updated_at = datetime.utcnow()
                existing_state.state_hash = self._calculate_state_hash(state_request.state_value)
                
                if session_id:
                    existing_state.owner_session_id = session_id
                
                state = existing_state
            else:
                # Create new state
                state = DistributedState(
                    id=str(uuid.uuid4()),
                    state_key=state_request.state_key,
                    state_namespace=state_request.state_namespace,
                    state_value=state_request.state_value,
                    state_version=1,
                    state_hash=self._calculate_state_hash(state_request.state_value),
                    owner_agent_id=agent_id,
                    owner_session_id=session_id,
                    last_modified_by=agent_id,
                    conflict_resolution_strategy=state_request.conflict_resolution_strategy
                )
                
                self.db.add(state)
            
            self.db.commit()
            
            # Update cache
            cache_key = f"{state_request.state_namespace}:{state_request.state_key}"
            self._state_cache[cache_key] = state_request.state_value
            
            # Notify other agents in session if applicable
            if session_id:
                await self._notify_state_change(agent_id, session_id, state_request)
            
            logger.info(f"State updated: {state_id} by {agent_id}")
            
            return StateResponse.from_orm(state)
            
        except Exception as e:
            logger.error(f"Error updating shared state: {e}")
            raise
    
    async def get_shared_state(
        self, 
        agent_id: str, 
        state_key: str, 
        state_namespace: str
    ) -> Optional[StateResponse]:
        """Get distributed shared state"""
        try:
            # Check cache first
            cache_key = f"{state_namespace}:{state_key}"
            if cache_key in self._state_cache:
                # Get from database to ensure consistency
                pass
            
            state = self.db.query(DistributedState).filter(
                and_(
                    DistributedState.state_key == state_key,
                    DistributedState.state_namespace == state_namespace
                )
            ).first()
            
            if state:
                # Update cache
                self._state_cache[cache_key] = state.state_value
                return StateResponse.from_orm(state)
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting shared state: {e}")
            raise
    
    async def synchronize_state(self, agent_id: str, session_id: str) -> Dict[str, Any]:
        """Synchronize all state for a session"""
        try:
            # Get all state for session
            states = self.db.query(DistributedState).filter(
                DistributedState.owner_session_id == session_id
            ).all()
            
            synchronized_state = {}
            for state in states:
                key = f"{state.state_namespace}:{state.state_key}"
                synchronized_state[key] = {
                    'value': state.state_value,
                    'version': state.state_version,
                    'last_modified_by': state.last_modified_by,
                    'updated_at': state.updated_at.isoformat()
                }
                
                # Update synchronization timestamp
                state.synchronized_at = datetime.utcnow()
            
            self.db.commit()
            
            return synchronized_state
            
        except Exception as e:
            logger.error(f"Error synchronizing state: {e}")
            raise
    
    def _merge_state_values(self, existing: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
        """Merge state values using deep merge strategy"""
        merged = existing.copy()
        
        for key, value in new.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = self._merge_state_values(merged[key], value)
            else:
                merged[key] = value
        
        return merged
    
    def _calculate_state_hash(self, state_value: Dict[str, Any]) -> str:
        """Calculate hash of state value for integrity checking"""
        state_json = json.dumps(state_value, sort_keys=True)
        return hashlib.sha256(state_json.encode('utf-8')).hexdigest()
    
    async def _notify_state_change(
        self, 
        agent_id: str, 
        session_id: str, 
        state_request: StateUpdateRequest
    ):
        """Notify other session participants of state change"""
        try:
            await self.send_session_message(
                agent_id,
                session_id,
                {
                    'type': 'state_change',
                    'state_key': state_request.state_key,
                    'state_namespace': state_request.state_namespace,
                    'state_value': state_request.state_value,
                    'modified_by': agent_id
                },
                MessageType.STATE_SYNC
            )
        except Exception as e:
            logger.error(f"Error notifying state change: {e}")
    
    # Resource Locking and Conflict Resolution
    async def request_resource_lock(
        self, 
        agent_id: str, 
        lock_request: ResourceLockRequest,
        session_id: Optional[str] = None
    ) -> ResourceLockResponse:
        """Request a lock on a resource"""
        try:
            lock_id = str(uuid.uuid4())
            
            # Check for existing locks
            existing_locks = self.db.query(ResourceLock).filter(
                and_(
                    ResourceLock.resource_id == lock_request.resource_id,
                    ResourceLock.status.in_([
                        ResourceLockStatus.GRANTED.value,
                        ResourceLockStatus.REQUESTED.value
                    ])
                )
            ).all()
            
            # Determine if lock can be granted
            can_grant = True
            if existing_locks:
                for existing_lock in existing_locks:
                    if existing_lock.lock_type == "exclusive" or lock_request.lock_type == "exclusive":
                        can_grant = False
                        break
            
            # Calculate expiration
            expires_at = None
            if lock_request.lock_duration_seconds:
                expires_at = datetime.utcnow() + timedelta(seconds=lock_request.lock_duration_seconds)
            
            # Create lock record
            lock = ResourceLock(
                id=lock_id,
                resource_id=lock_request.resource_id,
                resource_type=lock_request.resource_type,
                lock_type=lock_request.lock_type,
                holder_agent_id=agent_id,
                holder_session_id=session_id,
                status=ResourceLockStatus.GRANTED.value if can_grant else ResourceLockStatus.REQUESTED.value,
                priority=lock_request.priority,
                lock_reason=lock_request.lock_reason,
                lock_context=lock_request.context,
                expires_at=expires_at
            )
            
            if can_grant:
                lock.granted_at = datetime.utcnow()
            
            self.db.add(lock)
            self.db.commit()
            
            # Cache lock
            self._resource_locks[lock_id] = lock
            
            # Send notification if lock was granted
            if can_grant:
                await self._notify_lock_granted(agent_id, lock_request, session_id)
            else:
                await self._notify_lock_queued(agent_id, lock_request, session_id)
            
            logger.info(f"Resource lock {lock_id} {'granted' if can_grant else 'queued'} for {agent_id}")
            
            return ResourceLockResponse.from_orm(lock)
            
        except Exception as e:
            logger.error(f"Error requesting resource lock: {e}")
            raise
    
    async def release_resource_lock(self, agent_id: str, lock_id: str) -> bool:
        """Release a resource lock"""
        try:
            lock = self.db.query(ResourceLock).filter(
                and_(
                    ResourceLock.id == lock_id,
                    ResourceLock.holder_agent_id == agent_id,
                    ResourceLock.status == ResourceLockStatus.GRANTED.value
                )
            ).first()
            
            if not lock:
                return False
            
            # Release lock
            lock.status = ResourceLockStatus.RELEASED.value
            lock.released_at = datetime.utcnow()
            
            # Remove from cache
            if lock_id in self._resource_locks:
                del self._resource_locks[lock_id]
            
            # Check for queued locks on same resource
            queued_locks = self.db.query(ResourceLock).filter(
                and_(
                    ResourceLock.resource_id == lock.resource_id,
                    ResourceLock.status == ResourceLockStatus.REQUESTED.value
                )
            ).order_by(ResourceLock.priority.desc(), ResourceLock.requested_at.asc()).all()
            
            # Grant next lock if possible
            if queued_locks:
                await self._process_queued_locks(lock.resource_id)
            
            self.db.commit()
            
            logger.info(f"Resource lock {lock_id} released by {agent_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error releasing resource lock: {e}")
            return False
    
    async def check_resource_lock(self, agent_id: str, resource_id: str) -> Optional[ResourceLockResponse]:
        """Check if agent has a lock on a resource"""
        try:
            lock = self.db.query(ResourceLock).filter(
                and_(
                    ResourceLock.resource_id == resource_id,
                    ResourceLock.holder_agent_id == agent_id,
                    ResourceLock.status == ResourceLockStatus.GRANTED.value
                )
            ).first()
            
            if lock:
                return ResourceLockResponse.from_orm(lock)
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking resource lock: {e}")
            return None
    
    async def _process_queued_locks(self, resource_id: str):
        """Process queued locks for a resource"""
        try:
            queued_locks = self.db.query(ResourceLock).filter(
                and_(
                    ResourceLock.resource_id == resource_id,
                    ResourceLock.status == ResourceLockStatus.REQUESTED.value
                )
            ).order_by(ResourceLock.priority.desc(), ResourceLock.requested_at.asc()).all()
            
            for lock in queued_locks:
                # Check if lock can be granted
                conflicting_locks = self.db.query(ResourceLock).filter(
                    and_(
                        ResourceLock.resource_id == resource_id,
                        ResourceLock.status == ResourceLockStatus.GRANTED.value,
                        ResourceLock.id != lock.id
                    )
                ).all()
                
                can_grant = True
                if conflicting_locks:
                    for existing_lock in conflicting_locks:
                        if existing_lock.lock_type == "exclusive" or lock.lock_type == "exclusive":
                            can_grant = False
                            break
                
                if can_grant:
                    lock.status = ResourceLockStatus.GRANTED.value
                    lock.granted_at = datetime.utcnow()
                    
                    # Notify agent
                    await self._notify_lock_granted(
                        lock.holder_agent_id, 
                        ResourceLockRequest(
                            resource_id=lock.resource_id,
                            resource_type=lock.resource_type,
                            lock_type=lock.lock_type
                        ),
                        lock.holder_session_id
                    )
                    
                    # If exclusive lock, stop processing
                    if lock.lock_type == "exclusive":
                        break
            
            self.db.commit()
            
        except Exception as e:
            logger.error(f"Error processing queued locks: {e}")
    
    async def _notify_lock_granted(
        self, 
        agent_id: str, 
        lock_request: ResourceLockRequest, 
        session_id: Optional[str]
    ):
        """Notify agent that lock was granted"""
        try:
            message_request = AgentMessageCreate(
                to_agent_id=agent_id,
                message_type=MessageType.RESOURCE_GRANT,
                content={
                    'resource_id': lock_request.resource_id,
                    'resource_type': lock_request.resource_type,
                    'lock_type': lock_request.lock_type
                },
                session_id=session_id
            )
            
            await self.send_message("system", message_request)
            
        except Exception as e:
            logger.error(f"Error notifying lock granted: {e}")
    
    async def _notify_lock_queued(
        self, 
        agent_id: str, 
        lock_request: ResourceLockRequest, 
        session_id: Optional[str]
    ):
        """Notify agent that lock was queued"""
        try:
            message_request = AgentMessageCreate(
                to_agent_id=agent_id,
                message_type=MessageType.NOTIFICATION,
                content={
                    'type': 'lock_queued',
                    'resource_id': lock_request.resource_id,
                    'resource_type': lock_request.resource_type,
                    'message': 'Lock request queued due to conflicting locks'
                },
                session_id=session_id
            )
            
            await self.send_message("system", message_request)
            
        except Exception as e:
            logger.error(f"Error notifying lock queued: {e}")
    
    # Cleanup and Maintenance
    async def cleanup_expired_resources(self):
        """Clean up expired messages, locks, and sessions"""
        try:
            now = datetime.utcnow()
            
            # Clean up expired messages
            expired_messages = self.db.query(AgentMessage).filter(
                and_(
                    AgentMessage.expires_at.isnot(None),
                    AgentMessage.expires_at < now
                )
            ).all()
            
            for message in expired_messages:
                self.db.delete(message)
            
            # Clean up expired locks
            expired_locks = self.db.query(ResourceLock).filter(
                and_(
                    ResourceLock.expires_at.isnot(None),
                    ResourceLock.expires_at < now,
                    ResourceLock.status == ResourceLockStatus.GRANTED.value
                )
            ).all()
            
            for lock in expired_locks:
                lock.status = ResourceLockStatus.EXPIRED.value
                lock.released_at = now
                
                # Process queued locks
                await self._process_queued_locks(lock.resource_id)
            
            # Clean up inactive sessions
            timeout_threshold = now - timedelta(hours=1)
            inactive_sessions = self.db.query(CollaborationSession).filter(
                and_(
                    CollaborationSession.status == CollaborationStatus.ACTIVE.value,
                    CollaborationSession.last_activity < timeout_threshold,
                    CollaborationSession.auto_cleanup == True
                )
            ).all()
            
            for session in inactive_sessions:
                session.status = CollaborationStatus.COMPLETED.value
                session.ended_at = now
            
            self.db.commit()
            
            logger.info(f"Cleaned up {len(expired_messages)} expired messages, "
                       f"{len(expired_locks)} expired locks, "
                       f"{len(inactive_sessions)} inactive sessions")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    async def get_agent_status(self, agent_id: str) -> Dict[str, Any]:
        """Get comprehensive status for an agent"""
        try:
            # Get message counts
            pending_messages = self.db.query(AgentMessage).filter(
                and_(
                    AgentMessage.to_agent_id == agent_id,
                    AgentMessage.is_delivered == False
                )
            ).count()
            
            # Get active sessions
            active_sessions = self.db.query(SessionParticipant).filter(
                and_(
                    SessionParticipant.agent_id == agent_id,
                    SessionParticipant.status == "joined"
                )
            ).count()
            
            # Get held locks
            held_locks = self.db.query(ResourceLock).filter(
                and_(
                    ResourceLock.holder_agent_id == agent_id,
                    ResourceLock.status == ResourceLockStatus.GRANTED.value
                )
            ).count()
            
            return {
                'agent_id': agent_id,
                'pending_messages': pending_messages,
                'active_sessions': active_sessions,
                'held_locks': held_locks,
                'last_activity': datetime.utcnow().isoformat(),
                'status': 'active'
            }
            
        except Exception as e:
            logger.error(f"Error getting agent status: {e}")
            return {
                'agent_id': agent_id,
                'status': 'error',
                'error': str(e)
            }