"""
Tests for Agent Communication Framework

Comprehensive tests for secure messaging, collaboration sessions,
distributed state synchronization, and resource locking.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from scrollintel.core.agent_communication_framework import (
    AgentCommunicationFramework, EncryptionManager, MessageQueue
)
from scrollintel.models.agent_communication_models import (
    Base, AgentMessageCreate, CollaborationSessionCreate, ResourceLockRequest,
    StateUpdateRequest, MessageType, MessagePriority, SecurityLevel,
    CollaborationStatus, ResourceLockStatus
)


# Test Database Setup
@pytest.fixture
def test_db():
    """Create test database"""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    return SessionLocal()


@pytest.fixture
def encryption_manager():
    """Create encryption manager for testing"""
    return EncryptionManager("test_master_key_12345")


@pytest.fixture
def message_queue():
    """Create message queue for testing"""
    return MessageQueue()


@pytest.fixture
def communication_framework(test_db):
    """Create communication framework for testing"""
    return AgentCommunicationFramework(test_db, "test_encryption_key_12345")


class TestEncryptionManager:
    """Test encryption and decryption functionality"""
    
    def test_encrypt_decrypt_message(self, encryption_manager):
        """Test message encryption and decryption"""
        content = {"message": "Hello, Agent!", "data": {"value": 42}}
        from_agent = "agent_1"
        to_agent = "agent_2"
        
        # Encrypt message
        encrypted_content, content_hash = encryption_manager.encrypt_message(
            content, from_agent, to_agent
        )
        
        assert encrypted_content is not None
        assert content_hash is not None
        assert len(content_hash) == 64  # SHA256 hash length
        
        # Decrypt message
        decrypted_content = encryption_manager.decrypt_message(
            encrypted_content, from_agent
        )
        
        assert decrypted_content == content
    
    def test_message_integrity_verification(self, encryption_manager):
        """Test message integrity verification"""
        content = {"test": "data"}
        from_agent = "agent_1"
        to_agent = "agent_2"
        
        encrypted_content, content_hash = encryption_manager.encrypt_message(
            content, from_agent, to_agent
        )
        
        # Verify correct hash
        assert encryption_manager.verify_message_integrity(content, content_hash)
        
        # Verify incorrect hash fails
        tampered_content = {"test": "tampered_data"}
        assert not encryption_manager.verify_message_integrity(tampered_content, content_hash)
    
    def test_agent_specific_keys(self, encryption_manager):
        """Test that different agents get different encryption keys"""
        content = {"message": "test"}
        
        # Encrypt with different agents
        encrypted_1, _ = encryption_manager.encrypt_message(content, "agent_1", "agent_2")
        encrypted_2, _ = encryption_manager.encrypt_message(content, "agent_2", "agent_1")
        
        # Should produce different encrypted content
        assert encrypted_1 != encrypted_2
        
        # But should decrypt to same content
        decrypted_1 = encryption_manager.decrypt_message(encrypted_1, "agent_1")
        decrypted_2 = encryption_manager.decrypt_message(encrypted_2, "agent_2")
        
        assert decrypted_1 == content
        assert decrypted_2 == content


class TestMessageQueue:
    """Test message queuing functionality"""
    
    @pytest.mark.asyncio
    async def test_enqueue_dequeue_message(self, message_queue, test_db):
        """Test basic message queuing"""
        from scrollintel.models.agent_communication_models import AgentMessage
        
        agent_id = "test_agent"
        message = AgentMessage(
            id="msg_1",
            from_agent_id="sender",
            to_agent_id=agent_id,
            message_type=MessageType.NOTIFICATION.value,
            encrypted_content="encrypted_test",
            content_hash="hash_test",
            encryption_key_id="key_1"
        )
        
        # Enqueue message
        await message_queue.enqueue_message(agent_id, message)
        
        # Dequeue message
        dequeued = await message_queue.dequeue_message(agent_id, timeout=1.0)
        
        assert dequeued is not None
        assert dequeued.id == "msg_1"
        assert dequeued.from_agent_id == "sender"
    
    @pytest.mark.asyncio
    async def test_message_subscription(self, message_queue, test_db):
        """Test message subscription callbacks"""
        from scrollintel.models.agent_communication_models import AgentMessage
        
        agent_id = "test_agent"
        received_messages = []
        
        async def message_callback(message):
            received_messages.append(message)
        
        # Subscribe to messages
        message_queue.subscribe_to_messages(agent_id, message_callback)
        
        # Send message
        message = AgentMessage(
            id="msg_1",
            from_agent_id="sender",
            to_agent_id=agent_id,
            message_type=MessageType.NOTIFICATION.value,
            encrypted_content="encrypted_test",
            content_hash="hash_test",
            encryption_key_id="key_1"
        )
        
        await message_queue.enqueue_message(agent_id, message)
        
        # Give callback time to execute
        await asyncio.sleep(0.1)
        
        assert len(received_messages) == 1
        assert received_messages[0].id == "msg_1"
    
    @pytest.mark.asyncio
    async def test_timeout_handling(self, message_queue):
        """Test timeout when no messages available"""
        agent_id = "empty_agent"
        
        # Try to dequeue with timeout
        message = await message_queue.dequeue_message(agent_id, timeout=0.1)
        
        assert message is None


class TestAgentCommunicationFramework:
    """Test main communication framework functionality"""
    
    @pytest.mark.asyncio
    async def test_send_receive_message(self, communication_framework):
        """Test sending and receiving messages"""
        from_agent = "agent_1"
        to_agent = "agent_2"
        
        message_request = AgentMessageCreate(
            to_agent_id=to_agent,
            message_type=MessageType.TASK_REQUEST,
            content={"task": "process_data", "priority": "high"},
            priority=MessagePriority.HIGH
        )
        
        # Send message
        response = await communication_framework.send_message(from_agent, message_request)
        
        assert response.from_agent_id == from_agent
        assert response.to_agent_id == to_agent
        assert response.message_type == MessageType.TASK_REQUEST
        assert response.priority == MessagePriority.HIGH
        
        # Receive message
        messages = await communication_framework.receive_messages(to_agent)
        
        assert len(messages) == 1
        assert messages[0]['from_agent_id'] == from_agent
        assert messages[0]['content']['task'] == "process_data"
    
    @pytest.mark.asyncio
    async def test_message_acknowledgment(self, communication_framework):
        """Test message acknowledgment"""
        from_agent = "agent_1"
        to_agent = "agent_2"
        
        message_request = AgentMessageCreate(
            to_agent_id=to_agent,
            message_type=MessageType.NOTIFICATION,
            content={"message": "test"}
        )
        
        # Send message
        response = await communication_framework.send_message(from_agent, message_request)
        message_id = response.id
        
        # Acknowledge message
        success = await communication_framework.acknowledge_message(to_agent, message_id)
        assert success
        
        # Try to acknowledge non-existent message
        success = await communication_framework.acknowledge_message(to_agent, "fake_id")
        assert not success
    
    @pytest.mark.asyncio
    async def test_collaboration_session_lifecycle(self, communication_framework):
        """Test complete collaboration session lifecycle"""
        initiator = "agent_1"
        participants = ["agent_2", "agent_3"]
        
        session_request = CollaborationSessionCreate(
            session_name="Test Collaboration",
            description="Testing session functionality",
            objective={"goal": "test_collaboration"},
            participants=participants,
            max_participants=5
        )
        
        # Create session
        session = await communication_framework.create_collaboration_session(
            initiator, session_request
        )
        
        assert session.initiator_agent_id == initiator
        assert session.session_name == "Test Collaboration"
        assert session.status == CollaborationStatus.PENDING
        
        session_id = session.id
        
        # Join session as participant
        success = await communication_framework.join_collaboration_session("agent_2", session_id)
        assert success
        
        # Send session message
        success = await communication_framework.send_session_message(
            "agent_2",
            session_id,
            {"message": "Hello from agent_2"},
            MessageType.NOTIFICATION
        )
        assert success
        
        # Leave session
        success = await communication_framework.leave_collaboration_session("agent_2", session_id)
        assert success
    
    @pytest.mark.asyncio
    async def test_distributed_state_management(self, communication_framework):
        """Test distributed state synchronization"""
        agent_id = "agent_1"
        session_id = "session_1"
        
        # Update state
        state_request = StateUpdateRequest(
            state_key="shared_counter",
            state_namespace="test_session",
            state_value={"count": 42, "last_updated": "agent_1"}
        )
        
        state_response = await communication_framework.update_shared_state(
            agent_id, state_request, session_id
        )
        
        assert state_response.state_key == "shared_counter"
        assert state_response.state_namespace == "test_session"
        assert state_response.state_value["count"] == 42
        assert state_response.state_version == 1
        
        # Get state
        retrieved_state = await communication_framework.get_shared_state(
            agent_id, "shared_counter", "test_session"
        )
        
        assert retrieved_state is not None
        assert retrieved_state.state_value["count"] == 42
        
        # Update state again (version increment)
        state_request.state_value = {"count": 43, "last_updated": "agent_1"}
        state_request.expected_version = 1
        
        updated_state = await communication_framework.update_shared_state(
            agent_id, state_request, session_id
        )
        
        assert updated_state.state_version == 2
        assert updated_state.state_value["count"] == 43
    
    @pytest.mark.asyncio
    async def test_state_conflict_resolution(self, communication_framework):
        """Test state conflict resolution"""
        agent_id = "agent_1"
        
        # Create initial state
        state_request = StateUpdateRequest(
            state_key="conflict_test",
            state_namespace="test",
            state_value={"value": 1}
        )
        
        initial_state = await communication_framework.update_shared_state(
            agent_id, state_request
        )
        
        # Try to update with wrong expected version (should fail with fail_on_conflict)
        state_request.state_value = {"value": 2}
        state_request.expected_version = 999
        state_request.conflict_resolution_strategy = "fail_on_conflict"
        
        with pytest.raises(ValueError, match="Version conflict"):
            await communication_framework.update_shared_state(agent_id, state_request)
        
        # Update with merge strategy
        state_request.conflict_resolution_strategy = "merge"
        state_request.expected_version = 999
        state_request.state_value = {"new_field": "merged"}
        
        merged_state = await communication_framework.update_shared_state(
            agent_id, state_request
        )
        
        # Should contain both original and new fields
        assert "value" in merged_state.state_value
        assert "new_field" in merged_state.state_value
    
    @pytest.mark.asyncio
    async def test_resource_locking(self, communication_framework):
        """Test resource locking and conflict resolution"""
        agent_1 = "agent_1"
        agent_2 = "agent_2"
        resource_id = "shared_resource"
        
        # Agent 1 requests exclusive lock
        lock_request_1 = ResourceLockRequest(
            resource_id=resource_id,
            resource_type="database",
            lock_type="exclusive",
            lock_reason="Processing data",
            lock_duration_seconds=60
        )
        
        lock_1 = await communication_framework.request_resource_lock(
            agent_1, lock_request_1
        )
        
        assert lock_1.holder_agent_id == agent_1
        assert lock_1.status == ResourceLockStatus.GRANTED
        assert lock_1.resource_id == resource_id
        
        # Agent 2 requests lock on same resource (should be queued)
        lock_request_2 = ResourceLockRequest(
            resource_id=resource_id,
            resource_type="database",
            lock_type="exclusive",
            lock_reason="Also processing data"
        )
        
        lock_2 = await communication_framework.request_resource_lock(
            agent_2, lock_request_2
        )
        
        assert lock_2.holder_agent_id == agent_2
        assert lock_2.status == ResourceLockStatus.REQUESTED  # Should be queued
        
        # Check lock status
        active_lock = await communication_framework.check_resource_lock(agent_1, resource_id)
        assert active_lock is not None
        assert active_lock.holder_agent_id == agent_1
        
        # Release lock
        success = await communication_framework.release_resource_lock(agent_1, lock_1.id)
        assert success
        
        # Check that lock is released
        released_lock = await communication_framework.check_resource_lock(agent_1, resource_id)
        assert released_lock is None
    
    @pytest.mark.asyncio
    async def test_shared_resource_locks(self, communication_framework):
        """Test shared resource locks"""
        agent_1 = "agent_1"
        agent_2 = "agent_2"
        resource_id = "shared_read_resource"
        
        # Both agents request shared locks
        lock_request_1 = ResourceLockRequest(
            resource_id=resource_id,
            resource_type="file",
            lock_type="shared",
            lock_reason="Reading data"
        )
        
        lock_request_2 = ResourceLockRequest(
            resource_id=resource_id,
            resource_type="file",
            lock_type="shared",
            lock_reason="Also reading data"
        )
        
        # Both should be granted
        lock_1 = await communication_framework.request_resource_lock(agent_1, lock_request_1)
        lock_2 = await communication_framework.request_resource_lock(agent_2, lock_request_2)
        
        assert lock_1.status == ResourceLockStatus.GRANTED
        assert lock_2.status == ResourceLockStatus.GRANTED
        
        # Both should have active locks
        active_lock_1 = await communication_framework.check_resource_lock(agent_1, resource_id)
        active_lock_2 = await communication_framework.check_resource_lock(agent_2, resource_id)
        
        assert active_lock_1 is not None
        assert active_lock_2 is not None
    
    @pytest.mark.asyncio
    async def test_agent_status(self, communication_framework):
        """Test agent status reporting"""
        agent_id = "test_agent"
        
        # Get initial status
        status = await communication_framework.get_agent_status(agent_id)
        
        assert status['agent_id'] == agent_id
        assert 'pending_messages' in status
        assert 'active_sessions' in status
        assert 'held_locks' in status
        assert status['status'] == 'active'
    
    @pytest.mark.asyncio
    async def test_cleanup_expired_resources(self, communication_framework):
        """Test cleanup of expired resources"""
        # This test would need to manipulate timestamps to simulate expiration
        # For now, just test that cleanup runs without error
        await communication_framework.cleanup_expired_resources()
        
        # Should complete without raising exceptions
        assert True
    
    @pytest.mark.asyncio
    async def test_message_expiration(self, communication_framework):
        """Test message expiration handling"""
        from_agent = "agent_1"
        to_agent = "agent_2"
        
        # Send message with short expiration
        message_request = AgentMessageCreate(
            to_agent_id=to_agent,
            message_type=MessageType.NOTIFICATION,
            content={"message": "expires_soon"},
            expires_in_seconds=1  # 1 second expiration
        )
        
        response = await communication_framework.send_message(from_agent, message_request)
        assert response.expires_at is not None
        
        # Wait for expiration
        await asyncio.sleep(1.1)
        
        # Message should be filtered out when receiving
        messages = await communication_framework.receive_messages(to_agent)
        
        # Should not receive expired message
        expired_messages = [m for m in messages if m['content']['message'] == 'expires_soon']
        assert len(expired_messages) == 0
    
    @pytest.mark.asyncio
    async def test_session_state_synchronization(self, communication_framework):
        """Test session state synchronization"""
        agent_id = "agent_1"
        session_id = "sync_session"
        
        # Create multiple state entries for session
        states = [
            StateUpdateRequest(
                state_key=f"key_{i}",
                state_namespace="sync_test",
                state_value={"value": i}
            )
            for i in range(3)
        ]
        
        # Update all states
        for state_request in states:
            await communication_framework.update_shared_state(
                agent_id, state_request, session_id
            )
        
        # Synchronize session state
        synchronized_state = await communication_framework.synchronize_state(
            agent_id, session_id
        )
        
        assert len(synchronized_state) == 3
        
        # Check that all states are present
        for i in range(3):
            key = f"sync_test:key_{i}"
            assert key in synchronized_state
            assert synchronized_state[key]['value']['value'] == i


if __name__ == "__main__":
    pytest.main([__file__])