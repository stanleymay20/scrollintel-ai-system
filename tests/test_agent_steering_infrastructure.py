#!/usr/bin/env python3
"""
Test Agent Steering System Infrastructure

Tests the core infrastructure foundation components:
- Database schemas and operations
- Secure communication protocols  
- Real-time message queuing and event streaming
- Agent registry and management
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import asyncio
import uuid
from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from scrollintel.models.agent_steering_models import (
    Base, Agent, Task, AgentStatus, TaskStatus, TaskPriority,
    AgentPerformanceMetric, SystemEvent
)
from scrollintel.core.realtime_messaging import (
    RealTimeMessagingSystem, EventType, MessagePriority
)
from scrollintel.core.secure_communication import (
    SecureCommunicationManager, MessageType as SecureMessageType, SecurityLevel
)


class TestAgentSteeringInfrastructure:
    """Test suite for Agent Steering System infrastructure"""
    
    @pytest.fixture
    def db_session(self):
        """Create test database session"""
        # Use in-memory SQLite for testing
        engine = create_engine("sqlite:///:memory:", echo=False)
        Base.metadata.create_all(engine)
        
        SessionLocal = sessionmaker(bind=engine)
        session = SessionLocal()
        
        yield session
        
        session.close()
    
    @pytest.fixture
    async def messaging_system(self):
        """Create test messaging system"""
        system = RealTimeMessagingSystem()
        system.start()
        
        yield system
        
        system.stop()
    
    @pytest.fixture
    def secure_comm_manager(self):
        """Create test secure communication manager"""
        return SecureCommunicationManager()
    
    def test_database_schema_creation(self, db_session):
        """Test that database schemas are created correctly"""
        # Test agent creation
        agent = Agent(
            name="TestAgent",
            type="test",
            version="1.0.0",
            capabilities=[{"name": "test_capability", "type": "test"}],
            endpoint_url="http://localhost:8000/test",
            health_check_url="http://localhost:8000/health",
            status=AgentStatus.ACTIVE
        )
        
        db_session.add(agent)
        db_session.commit()
        
        # Verify agent was created
        retrieved_agent = db_session.query(Agent).filter(Agent.name == "TestAgent").first()
        assert retrieved_agent is not None
        assert retrieved_agent.name == "TestAgent"
        assert retrieved_agent.status == AgentStatus.ACTIVE
        
        # Test task creation
        task = Task(
            title="Test Task",
            description="A test task",
            task_type="test",
            priority=TaskPriority.HIGH,
            requirements={"test": True},
            status=TaskStatus.PENDING
        )
        
        db_session.add(task)
        db_session.commit()
        
        # Verify task was created
        retrieved_task = db_session.query(Task).filter(Task.title == "Test Task").first()
        assert retrieved_task is not None
        assert retrieved_task.priority == TaskPriority.HIGH
        assert retrieved_task.status == TaskStatus.PENDING
        
        # Test performance metric creation
        metric = AgentPerformanceMetric(
            agent_id=agent.id,
            response_time=1.5,
            throughput=100.0,
            accuracy=95.0,
            reliability=99.0,
            cpu_usage=50.0,
            memory_usage=60.0,
            network_usage=10.0
        )
        
        db_session.add(metric)
        db_session.commit()
        
        # Verify metric was created
        retrieved_metric = db_session.query(AgentPerformanceMetric).filter(
            AgentPerformanceMetric.agent_id == agent.id
        ).first()
        assert retrieved_metric is not None
        assert retrieved_metric.response_time == 1.5
        assert retrieved_metric.accuracy == 95.0
        
        print("✅ Database schema tests passed")
    
    async def test_realtime_messaging(self, messaging_system):
        """Test real-time messaging and event streaming"""
        received_events = []
        
        def event_handler(event):
            received_events.append(event)
        
        # Subscribe to events
        success = messaging_system.subscribe(
            "test_subscriber",
            [EventType.AGENT_REGISTERED, EventType.TASK_ASSIGNED],
            event_handler
        )
        assert success
        
        # Publish events
        event1 = messaging_system.create_event(
            EventType.AGENT_REGISTERED,
            "test_source",
            {"agent_id": "test-123", "agent_name": "TestAgent"},
            MessagePriority.NORMAL
        )
        
        event2 = messaging_system.create_event(
            EventType.TASK_ASSIGNED,
            "test_orchestrator",
            {"task_id": "task-456", "agent_id": "test-123"},
            MessagePriority.HIGH
        )
        
        success1 = messaging_system.publish(event1)
        success2 = messaging_system.publish(event2)
        
        assert success1
        assert success2
        
        # Wait for event processing
        await asyncio.sleep(1)
        
        # Verify events were received
        assert len(received_events) == 2
        assert received_events[0].event_type == EventType.AGENT_REGISTERED
        assert received_events[1].event_type == EventType.TASK_ASSIGNED
        
        # Test system status
        status = messaging_system.get_system_status()
        assert status["processing_stats"]["events_processed"] >= 2
        
        print("✅ Real-time messaging tests passed")
    
    async def test_secure_communication(self, secure_comm_manager):
        """Test secure communication protocols"""
        # Create protocols for two test agents
        agent1_id = "agent-1"
        agent2_id = "agent-2"
        
        protocol1 = secure_comm_manager.create_protocol(agent1_id)
        protocol2 = secure_comm_manager.create_protocol(agent2_id)
        
        assert protocol1 is not None
        assert protocol2 is not None
        
        # Set security policies
        secure_comm_manager.set_security_policy(agent1_id, {
            "min_security_level": SecurityLevel.HIGH,
            "require_encryption": True
        })
        
        # Establish secure channels
        success1 = await protocol1.establish_secure_channel(agent2_id)
        success2 = await protocol2.establish_secure_channel(agent1_id)
        
        assert success1
        assert success2
        
        # Send secure messages
        success = await protocol1.send_secure_message(
            agent2_id,
            SecureMessageType.COORDINATION,
            {"message": "Test coordination", "data": {"test": True}},
            SecurityLevel.HIGH
        )
        assert success
        
        # Test system status
        system_status = secure_comm_manager.get_system_status()
        assert system_status["total_protocols"] == 2
        assert system_status["active_connections"] >= 2
        
        print("✅ Secure communication tests passed")
    
    def test_agent_task_relationships(self, db_session):
        """Test relationships between agents and tasks"""
        # Create agent
        agent = Agent(
            name="RelationshipTestAgent",
            type="test",
            version="1.0.0",
            capabilities=[{"name": "test", "type": "test"}],
            endpoint_url="http://localhost:8000/test",
            health_check_url="http://localhost:8000/health",
            status=AgentStatus.ACTIVE
        )
        
        db_session.add(agent)
        db_session.commit()
        
        # Create task assigned to agent
        task = Task(
            title="Relationship Test Task",
            description="Test task relationships",
            task_type="test",
            priority=TaskPriority.MEDIUM,
            requirements={"test": True},
            status=TaskStatus.RUNNING,
            assigned_agent_id=agent.id
        )
        
        db_session.add(task)
        db_session.commit()
        
        # Test relationship queries
        agent_with_tasks = db_session.query(Agent).filter(Agent.id == agent.id).first()
        assert len(agent_with_tasks.tasks) == 1
        assert agent_with_tasks.tasks[0].title == "Relationship Test Task"
        
        task_with_agent = db_session.query(Task).filter(Task.id == task.id).first()
        assert task_with_agent.assigned_agent is not None
        assert task_with_agent.assigned_agent.name == "RelationshipTestAgent"
        
        print("✅ Agent-task relationship tests passed")
    
    def test_system_events_logging(self, db_session):
        """Test system events logging"""
        # Create system event
        event = SystemEvent(
            event_type="test_event",
            severity="INFO",
            source="test_component",
            message="Test system event",
            details={"test_data": "test_value"}
        )
        
        db_session.add(event)
        db_session.commit()
        
        # Verify event was logged
        retrieved_event = db_session.query(SystemEvent).filter(
            SystemEvent.event_type == "test_event"
        ).first()
        
        assert retrieved_event is not None
        assert retrieved_event.severity == "INFO"
        assert retrieved_event.source == "test_component"
        assert retrieved_event.details["test_data"] == "test_value"
        
        print("✅ System events logging tests passed")


def run_infrastructure_tests():
    """Run all infrastructure tests"""
    print("🧪 Running Agent Steering System Infrastructure Tests")
    print("=" * 60)
    
    test_instance = TestAgentSteeringInfrastructure()
    
    # Create test database session
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    
    try:
        # Run database tests
        print("💾 Testing database schemas...")
        test_instance.test_database_schema_creation(session)
        test_instance.test_agent_task_relationships(session)
        test_instance.test_system_events_logging(session)
        
        # Run async tests
        print("\n📡 Testing real-time messaging...")
        async def run_async_tests():
            messaging_system = RealTimeMessagingSystem()
            messaging_system.start()
            
            try:
                await test_instance.test_realtime_messaging(messaging_system)
                
                secure_comm_manager = SecureCommunicationManager()
                await test_instance.test_secure_communication(secure_comm_manager)
                
            finally:
                messaging_system.stop()
        
        asyncio.run(run_async_tests())
        
        print("\n✅ All infrastructure tests passed!")
        print("\n🚀 Core Infrastructure Foundation is working correctly:")
        print("   • Database schemas for agent registry, task management, and performance tracking")
        print("   • Secure communication protocols with encryption and authentication")
        print("   • Real-time message queuing and event streaming infrastructure")
        print("   • Agent-task relationships and system event logging")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Infrastructure tests failed: {e}")
        return False
        
    finally:
        session.close()


if __name__ == "__main__":
    success = run_infrastructure_tests()
    exit(0 if success else 1)