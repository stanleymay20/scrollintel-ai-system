"""
Integration Tests for Agent Communication API Routes

Tests for REST API endpoints for agent communication, collaboration sessions,
distributed state synchronization, and resource management.
"""

import pytest
import json
from datetime import datetime
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from unittest.mock import patch, Mock

from scrollintel.api.main import app
from scrollintel.models.agent_communication_models import Base
from scrollintel.models.database import get_db


# Test Database Setup
@pytest.fixture
def test_db():
    """Create test database"""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    return SessionLocal()


@pytest.fixture
def client(test_db):
    """Create test client with test database"""
    def override_get_db():
        try:
            yield test_db
        finally:
            test_db.close()
    
    app.dependency_overrides[get_db] = override_get_db
    
    with TestClient(app) as test_client:
        yield test_client
    
    app.dependency_overrides.clear()


class TestMessageEndpoints:
    """Test message-related API endpoints"""
    
    def test_send_message(self, client):
        """Test sending a message between agents"""
        message_data = {
            "to_agent_id": "agent_2",
            "message_type": "task_request",
            "content": {"task": "process_data", "priority": "high"},
            "priority": "high",
            "security_level": "internal"
        }
        
        response = client.post(
            "/api/v1/agent-communication/messages/send?from_agent_id=agent_1",
            json=message_data
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["from_agent_id"] == "agent_1"
        assert data["to_agent_id"] == "agent_2"
        assert data["message_type"] == "task_request"
        assert data["priority"] == "high"
        assert "id" in data
    
    def test_receive_messages(self, client):
        """Test receiving messages for an agent"""
        # First send a message
        message_data = {
            "to_agent_id": "agent_2",
            "message_type": "notification",
            "content": {"message": "Hello Agent 2"},
            "priority": "normal"
        }
        
        send_response = client.post(
            "/api/v1/agent-communication/messages/send?from_agent_id=agent_1",
            json=message_data
        )
        assert send_response.status_code == 200
        
        # Then receive messages
        response = client.get(
            "/api/v1/agent-communication/messages/agent_2?timeout=1.0"
        )
        
        assert response.status_code == 200
        messages = response.json()
        
        assert len(messages) >= 1
        assert messages[0]["from_agent_id"] == "agent_1"
        assert messages[0]["content"]["message"] == "Hello Agent 2"
    
    def test_acknowledge_message(self, client):
        """Test acknowledging a message"""
        # Send message first
        message_data = {
            "to_agent_id": "agent_2",
            "message_type": "notification",
            "content": {"message": "Test acknowledgment"}
        }
        
        send_response = client.post(
            "/api/v1/agent-communication/messages/send?from_agent_id=agent_1",
            json=message_data
        )
        message_id = send_response.json()["id"]
        
        # Acknowledge message
        response = client.post(
            f"/api/v1/agent-communication/messages/{message_id}/acknowledge?agent_id=agent_2"
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "acknowledged"
        assert data["message_id"] == message_id
    
    def test_broadcast_message(self, client):
        """Test broadcasting message to multiple agents"""
        broadcast_data = {
            "message_content": {"announcement": "System maintenance scheduled"},
            "recipient_agents": ["agent_1", "agent_2", "agent_3"],
            "message_type": "notification",
            "priority": "high"
        }
        
        response = client.post(
            "/api/v1/agent-communication/messages/broadcast?from_agent_id=system",
            json=broadcast_data
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["recipients"] == 3
        assert len(data["results"]) == 3
        assert all(result["status"] == "sent" for result in data["results"])
    
    def test_receive_messages_with_filter(self, client):
        """Test receiving messages with type filter"""
        # Send different types of messages
        messages = [
            {
                "to_agent_id": "agent_2",
                "message_type": "task_request",
                "content": {"task": "process"}
            },
            {
                "to_agent_id": "agent_2", 
                "message_type": "notification",
                "content": {"message": "info"}
            }
        ]
        
        for msg in messages:
            client.post(
                "/api/v1/agent-communication/messages/send?from_agent_id=agent_1",
                json=msg
            )
        
        # Receive only task_request messages
        response = client.get(
            "/api/v1/agent-communication/messages/agent_2?message_types=task_request&timeout=1.0"
        )
        
        assert response.status_code == 200
        received_messages = response.json()
        
        # Should only get task_request messages
        assert all(msg["message_type"] == "task_request" for msg in received_messages)


class TestCollaborationEndpoints:
    """Test collaboration session API endpoints"""
    
    def test_create_collaboration_session(self, client):
        """Test creating a collaboration session"""
        session_data = {
            "session_name": "Test Collaboration",
            "description": "Testing collaboration features",
            "objective": {"goal": "test_collaboration", "deadline": "2024-12-31"},
            "participants": ["agent_2", "agent_3"],
            "security_level": "internal",
            "max_participants": 5,
            "session_timeout": 3600
        }
        
        response = client.post(
            "/api/v1/agent-communication/sessions?initiator_agent_id=agent_1",
            json=session_data
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["initiator_agent_id"] == "agent_1"
        assert data["session_name"] == "Test Collaboration"
        assert data["status"] == "pending"
        assert data["max_participants"] == 5
        assert "id" in data
    
    def test_join_collaboration_session(self, client):
        """Test joining a collaboration session"""
        # Create session first
        session_data = {
            "session_name": "Join Test",
            "objective": {"goal": "test_join"},
            "participants": ["agent_2"]
        }
        
        create_response = client.post(
            "/api/v1/agent-communication/sessions?initiator_agent_id=agent_1",
            json=session_data
        )
        session_id = create_response.json()["id"]
        
        # Join session
        response = client.post(
            f"/api/v1/agent-communication/sessions/{session_id}/join?agent_id=agent_2"
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "joined"
        assert data["session_id"] == session_id
        assert data["agent_id"] == "agent_2"
    
    def test_leave_collaboration_session(self, client):
        """Test leaving a collaboration session"""
        # Create and join session first
        session_data = {
            "session_name": "Leave Test",
            "objective": {"goal": "test_leave"},
            "participants": ["agent_2"]
        }
        
        create_response = client.post(
            "/api/v1/agent-communication/sessions?initiator_agent_id=agent_1",
            json=session_data
        )
        session_id = create_response.json()["id"]
        
        # Join session
        client.post(
            f"/api/v1/agent-communication/sessions/{session_id}/join?agent_id=agent_2"
        )
        
        # Leave session
        response = client.post(
            f"/api/v1/agent-communication/sessions/{session_id}/leave?agent_id=agent_2"
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "left"
        assert data["session_id"] == session_id
    
    def test_send_session_message(self, client):
        """Test sending message to session participants"""
        # Create session
        session_data = {
            "session_name": "Message Test",
            "objective": {"goal": "test_messaging"},
            "participants": ["agent_2"]
        }
        
        create_response = client.post(
            "/api/v1/agent-communication/sessions?initiator_agent_id=agent_1",
            json=session_data
        )
        session_id = create_response.json()["id"]
        
        # Join session
        client.post(
            f"/api/v1/agent-communication/sessions/{session_id}/join?agent_id=agent_2"
        )
        
        # Send session message
        message_data = {
            "message_content": {"update": "Progress report", "percentage": 50},
            "message_type": "notification"
        }
        
        response = client.post(
            f"/api/v1/agent-communication/sessions/{session_id}/message?agent_id=agent_2",
            json=message_data
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "sent"
        assert data["session_id"] == session_id
        assert data["sender"] == "agent_2"


class TestStateEndpoints:
    """Test distributed state API endpoints"""
    
    def test_update_shared_state(self, client):
        """Test updating distributed shared state"""
        state_data = {
            "state_key": "shared_counter",
            "state_namespace": "test_session",
            "state_value": {"count": 42, "last_updated": "agent_1"},
            "conflict_resolution_strategy": "last_write_wins"
        }
        
        response = client.put(
            "/api/v1/agent-communication/state?agent_id=agent_1",
            json=state_data
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["state_key"] == "shared_counter"
        assert data["state_namespace"] == "test_session"
        assert data["state_value"]["count"] == 42
        assert data["state_version"] == 1
        assert data["owner_agent_id"] == "agent_1"
    
    def test_get_shared_state(self, client):
        """Test retrieving distributed shared state"""
        # Create state first
        state_data = {
            "state_key": "test_data",
            "state_namespace": "test_ns",
            "state_value": {"data": "test_value"}
        }
        
        client.put(
            "/api/v1/agent-communication/state?agent_id=agent_1",
            json=state_data
        )
        
        # Get state
        response = client.get(
            "/api/v1/agent-communication/state/test_ns/test_data?agent_id=agent_1"
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["state_key"] == "test_data"
        assert data["state_namespace"] == "test_ns"
        assert data["state_value"]["data"] == "test_value"
    
    def test_synchronize_session_state(self, client):
        """Test synchronizing session state"""
        session_id = "sync_session"
        
        # Create multiple state entries
        states = [
            {
                "state_key": f"key_{i}",
                "state_namespace": "sync_test",
                "state_value": {"value": i}
            }
            for i in range(3)
        ]
        
        for state_data in states:
            client.put(
                f"/api/v1/agent-communication/state?agent_id=agent_1&session_id={session_id}",
                json=state_data
            )
        
        # Synchronize state
        response = client.post(
            f"/api/v1/agent-communication/state/synchronize/{session_id}?agent_id=agent_1"
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["session_id"] == session_id
        assert data["state_count"] == 3
        assert "synchronized_at" in data
    
    def test_state_version_conflict(self, client):
        """Test state version conflict handling"""
        # Create initial state
        state_data = {
            "state_key": "conflict_test",
            "state_namespace": "test",
            "state_value": {"value": 1}
        }
        
        response = client.put(
            "/api/v1/agent-communication/state?agent_id=agent_1",
            json=state_data
        )
        initial_version = response.json()["state_version"]
        
        # Try to update with wrong expected version
        conflict_data = {
            "state_key": "conflict_test",
            "state_namespace": "test",
            "state_value": {"value": 2},
            "expected_version": 999,
            "conflict_resolution_strategy": "fail_on_conflict"
        }
        
        response = client.put(
            "/api/v1/agent-communication/state?agent_id=agent_1",
            json=conflict_data
        )
        
        assert response.status_code == 409  # Conflict


class TestResourceLockEndpoints:
    """Test resource locking API endpoints"""
    
    def test_request_resource_lock(self, client):
        """Test requesting a resource lock"""
        lock_data = {
            "resource_id": "shared_database",
            "resource_type": "database",
            "lock_type": "exclusive",
            "lock_reason": "Data processing",
            "lock_duration_seconds": 300,
            "priority": 1
        }
        
        response = client.post(
            "/api/v1/agent-communication/locks?agent_id=agent_1",
            json=lock_data
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["resource_id"] == "shared_database"
        assert data["resource_type"] == "database"
        assert data["lock_type"] == "exclusive"
        assert data["holder_agent_id"] == "agent_1"
        assert data["status"] == "granted"
    
    def test_release_resource_lock(self, client):
        """Test releasing a resource lock"""
        # Request lock first
        lock_data = {
            "resource_id": "test_resource",
            "resource_type": "file",
            "lock_type": "exclusive"
        }
        
        lock_response = client.post(
            "/api/v1/agent-communication/locks?agent_id=agent_1",
            json=lock_data
        )
        lock_id = lock_response.json()["id"]
        
        # Release lock
        response = client.delete(
            f"/api/v1/agent-communication/locks/{lock_id}?agent_id=agent_1"
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "released"
        assert data["lock_id"] == lock_id
    
    def test_check_resource_lock(self, client):
        """Test checking resource lock status"""
        # Request lock first
        lock_data = {
            "resource_id": "check_resource",
            "resource_type": "service",
            "lock_type": "exclusive"
        }
        
        client.post(
            "/api/v1/agent-communication/locks?agent_id=agent_1",
            json=lock_data
        )
        
        # Check lock
        response = client.get(
            "/api/v1/agent-communication/locks/agent_1/check_resource"
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["resource_id"] == "check_resource"
        assert data["holder_agent_id"] == "agent_1"
        assert data["status"] == "granted"
    
    def test_lock_conflict_handling(self, client):
        """Test handling of conflicting lock requests"""
        resource_id = "conflict_resource"
        
        # Agent 1 requests exclusive lock
        lock_data_1 = {
            "resource_id": resource_id,
            "resource_type": "database",
            "lock_type": "exclusive"
        }
        
        response_1 = client.post(
            "/api/v1/agent-communication/locks?agent_id=agent_1",
            json=lock_data_1
        )
        assert response_1.json()["status"] == "granted"
        
        # Agent 2 requests lock on same resource
        lock_data_2 = {
            "resource_id": resource_id,
            "resource_type": "database",
            "lock_type": "exclusive"
        }
        
        response_2 = client.post(
            "/api/v1/agent-communication/locks?agent_id=agent_2",
            json=lock_data_2
        )
        
        # Should be queued due to conflict
        assert response_2.json()["status"] == "requested"


class TestStatusEndpoints:
    """Test status and monitoring API endpoints"""
    
    def test_get_agent_status(self, client):
        """Test getting agent communication status"""
        response = client.get(
            "/api/v1/agent-communication/status/agent_1"
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["agent_id"] == "agent_1"
        assert "pending_messages" in data
        assert "active_sessions" in data
        assert "held_locks" in data
        assert data["status"] == "active"
    
    def test_cleanup_expired_resources(self, client):
        """Test cleanup endpoint"""
        response = client.post(
            "/api/v1/agent-communication/cleanup"
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "cleanup_scheduled"
        assert "scheduled_at" in data
    
    def test_health_check(self, client):
        """Test health check endpoint"""
        response = client.get(
            "/api/v1/agent-communication/health"
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "healthy"
        assert data["service"] == "agent_communication"
        assert "timestamp" in data
        assert data["version"] == "1.0.0"


class TestErrorHandling:
    """Test error handling in API endpoints"""
    
    def test_invalid_message_data(self, client):
        """Test handling of invalid message data"""
        invalid_data = {
            "to_agent_id": "",  # Invalid empty agent ID
            "message_type": "invalid_type",  # Invalid message type
            "content": "not_a_dict"  # Invalid content type
        }
        
        response = client.post(
            "/api/v1/agent-communication/messages/send?from_agent_id=agent_1",
            json=invalid_data
        )
        
        assert response.status_code == 422  # Validation error
    
    def test_nonexistent_session_operations(self, client):
        """Test operations on non-existent sessions"""
        fake_session_id = "fake_session_123"
        
        # Try to join non-existent session
        response = client.post(
            f"/api/v1/agent-communication/sessions/{fake_session_id}/join?agent_id=agent_1"
        )
        
        assert response.status_code == 404
    
    def test_unauthorized_lock_release(self, client):
        """Test releasing lock by non-owner"""
        # Agent 1 creates lock
        lock_data = {
            "resource_id": "protected_resource",
            "resource_type": "file",
            "lock_type": "exclusive"
        }
        
        lock_response = client.post(
            "/api/v1/agent-communication/locks?agent_id=agent_1",
            json=lock_data
        )
        lock_id = lock_response.json()["id"]
        
        # Agent 2 tries to release Agent 1's lock
        response = client.delete(
            f"/api/v1/agent-communication/locks/{lock_id}?agent_id=agent_2"
        )
        
        assert response.status_code == 404  # Not found (not owned by agent_2)
    
    def test_nonexistent_state_retrieval(self, client):
        """Test retrieving non-existent state"""
        response = client.get(
            "/api/v1/agent-communication/state/nonexistent/fake_key?agent_id=agent_1"
        )
        
        assert response.status_code == 404


if __name__ == "__main__":
    pytest.main([__file__])