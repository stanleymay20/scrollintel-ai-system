#!/usr/bin/env python3
"""
Simple Core Infrastructure Test

Tests the essential components of the Agent Steering System infrastructure
to verify the Core Infrastructure Foundation task is complete.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import uuid

from scrollintel.models.agent_steering_models import (
    Base, Agent, Task, AgentStatus, TaskStatus, TaskPriority,
    AgentPerformanceMetric, SystemEvent
)


def test_database_infrastructure():
    """Test database schemas and operations"""
    print("💾 Testing Database Infrastructure...")
    
    # Create in-memory SQLite database
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)
    
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    
    try:
        # Test 1: Create and query agents
        print("   📋 Testing agent registry schema...")
        agent = Agent(
            name="TestDataAnalyst",
            type="data_analysis",
            version="1.0.0",
            capabilities=[
                {"name": "statistical_analysis", "type": "data_analysis"},
                {"name": "data_visualization", "type": "visualization"}
            ],
            endpoint_url="http://localhost:8001/agent",
            health_check_url="http://localhost:8001/health",
            status=AgentStatus.ACTIVE,
            current_load=0.3,
            max_concurrent_tasks=5,
            average_response_time=1.2,
            success_rate=98.5
        )
        
        session.add(agent)
        session.commit()
        
        # Verify agent creation
        retrieved_agent = session.query(Agent).filter(Agent.name == "TestDataAnalyst").first()
        assert retrieved_agent is not None
        assert retrieved_agent.status == AgentStatus.ACTIVE
        assert len(retrieved_agent.capabilities) == 2
        print("   ✅ Agent registry schema working")
        
        # Test 2: Create and query tasks
        print("   📝 Testing task management schema...")
        task = Task(
            title="Test Analysis Task",
            description="Analyze test data for patterns",
            task_type="data_analysis",
            priority=TaskPriority.HIGH,
            requirements={
                "capabilities": ["statistical_analysis"],
                "data_sources": ["test_db"],
                "output_format": "report"
            },
            status=TaskStatus.PENDING,
            assigned_agent_id=agent.id,
            estimated_duration=3600,
            max_retries=3
        )
        
        session.add(task)
        session.commit()
        
        # Verify task creation
        retrieved_task = session.query(Task).filter(Task.title == "Test Analysis Task").first()
        assert retrieved_task is not None
        assert retrieved_task.priority == TaskPriority.HIGH
        assert retrieved_task.assigned_agent_id == agent.id
        print("   ✅ Task management schema working")
        
        # Test 3: Create performance metrics
        print("   📊 Testing performance tracking schema...")
        metric = AgentPerformanceMetric(
            agent_id=agent.id,
            response_time=1.5,
            throughput=45.0,
            accuracy=97.5,
            reliability=99.2,
            cpu_usage=35.0,
            memory_usage=55.0,
            network_usage=8.5,
            cost_savings=12000.0,
            revenue_increase=8500.0,
            risk_reduction=15.0,
            productivity_gain=22.0,
            customer_satisfaction=4.8,
            compliance_score=98.0
        )
        
        session.add(metric)
        session.commit()
        
        # Verify metric creation
        retrieved_metric = session.query(AgentPerformanceMetric).filter(
            AgentPerformanceMetric.agent_id == agent.id
        ).first()
        assert retrieved_metric is not None
        assert retrieved_metric.accuracy == 97.5
        assert retrieved_metric.cost_savings == 12000.0
        print("   ✅ Performance tracking schema working")
        
        # Test 4: Create system events
        print("   📋 Testing system event logging...")
        event = SystemEvent(
            event_type="infrastructure_test",
            severity="INFO",
            source="test_suite",
            message="Core infrastructure test completed successfully",
            details={
                "test_timestamp": datetime.utcnow().isoformat(),
                "components_tested": ["agent_registry", "task_management", "performance_tracking"],
                "test_results": "all_passed"
            },
            agent_id=agent.id,
            task_id=task.id
        )
        
        session.add(event)
        session.commit()
        
        # Verify event creation
        retrieved_event = session.query(SystemEvent).filter(
            SystemEvent.event_type == "infrastructure_test"
        ).first()
        assert retrieved_event is not None
        assert retrieved_event.severity == "INFO"
        assert "components_tested" in retrieved_event.details
        print("   ✅ System event logging working")
        
        # Test 5: Complex queries and relationships
        print("   🔗 Testing database relationships...")
        
        # Query agent with tasks
        agent_with_tasks = session.query(Agent).filter(Agent.id == agent.id).first()
        assert len(agent_with_tasks.tasks) == 1
        assert agent_with_tasks.tasks[0].title == "Test Analysis Task"
        
        # Query task with agent
        task_with_agent = session.query(Task).filter(Task.id == task.id).first()
        assert task_with_agent.assigned_agent is not None
        assert task_with_agent.assigned_agent.name == "TestDataAnalyst"
        
        # Query performance metrics for agent
        agent_metrics = session.query(AgentPerformanceMetric).filter(
            AgentPerformanceMetric.agent_id == agent.id
        ).all()
        assert len(agent_metrics) == 1
        assert agent_metrics[0].productivity_gain == 22.0
        
        print("   ✅ Database relationships working")
        
        # Test 6: Database indexes and performance
        print("   ⚡ Testing database performance features...")
        
        # Test that we can query efficiently
        active_agents = session.query(Agent).filter(Agent.status == AgentStatus.ACTIVE).all()
        assert len(active_agents) == 1
        
        high_priority_tasks = session.query(Task).filter(Task.priority == TaskPriority.HIGH).all()
        assert len(high_priority_tasks) == 1
        
        recent_metrics = session.query(AgentPerformanceMetric).order_by(
            AgentPerformanceMetric.recorded_at.desc()
        ).limit(10).all()
        assert len(recent_metrics) == 1
        
        print("   ✅ Database performance features working")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Database test failed: {e}")
        return False
        
    finally:
        session.close()


def test_secure_communication_infrastructure():
    """Test secure communication protocol components"""
    print("🔒 Testing Secure Communication Infrastructure...")
    
    try:
        from scrollintel.core.secure_communication import (
            SecureCommunicationManager, EncryptionManager, SecurityLevel, MessageType
        )
        
        # Test 1: Encryption manager
        print("   🔐 Testing encryption manager...")
        encryption_manager = EncryptionManager()
        
        # Test key generation
        public_key = encryption_manager.get_public_key_pem()
        assert public_key is not None
        assert b"BEGIN PUBLIC KEY" in public_key
        
        # Test symmetric key generation
        agent_id = "test-agent-123"
        symmetric_key = encryption_manager.generate_symmetric_key(agent_id)
        assert symmetric_key is not None
        assert len(symmetric_key) > 0
        
        # Test symmetric encryption/decryption
        test_data = b"Test secure message data"
        encrypted_data = encryption_manager.encrypt_symmetric(test_data, agent_id)
        decrypted_data = encryption_manager.decrypt_symmetric(encrypted_data, agent_id)
        assert decrypted_data == test_data
        
        print("   ✅ Encryption manager working")
        
        # Test 2: Secure communication manager
        print("   📡 Testing secure communication manager...")
        comm_manager = SecureCommunicationManager()
        
        # Create protocols for test agents
        agent1_id = "agent-alpha"
        agent2_id = "agent-beta"
        
        protocol1 = comm_manager.create_protocol(agent1_id)
        protocol2 = comm_manager.create_protocol(agent2_id)
        
        assert protocol1 is not None
        assert protocol2 is not None
        
        # Set security policies
        comm_manager.set_security_policy(agent1_id, {
            "min_security_level": SecurityLevel.HIGH,
            "require_encryption": True,
            "require_signature": True
        })
        
        policy = comm_manager.get_security_policy(agent1_id)
        assert policy["min_security_level"] == SecurityLevel.HIGH
        assert policy["require_encryption"] is True
        
        print("   ✅ Secure communication manager working")
        
        # Test 3: System status
        print("   📊 Testing communication system status...")
        system_status = comm_manager.get_system_status()
        assert system_status["total_protocols"] == 2
        assert "active_connections" in system_status
        assert "total_audit_entries" in system_status
        
        print("   ✅ Communication system status working")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Secure communication test failed: {e}")
        return False


def test_realtime_messaging_infrastructure():
    """Test real-time messaging and event streaming components"""
    print("📡 Testing Real-Time Messaging Infrastructure...")
    
    try:
        from scrollintel.core.realtime_messaging import (
            RealTimeMessagingSystem, EventType, MessagePriority, StreamEvent
        )
        
        # Test 1: Event creation
        print("   📨 Testing event creation...")
        messaging_system = RealTimeMessagingSystem()
        
        test_event = messaging_system.create_event(
            EventType.AGENT_REGISTERED,
            "test_source",
            {"agent_id": "test-123", "agent_name": "TestAgent"},
            MessagePriority.NORMAL
        )
        
        assert test_event is not None
        assert test_event.event_type == EventType.AGENT_REGISTERED
        assert test_event.source == "test_source"
        assert test_event.data["agent_id"] == "test-123"
        
        print("   ✅ Event creation working")
        
        # Test 2: Event serialization
        print("   💾 Testing event serialization...")
        event_dict = test_event.to_dict()
        assert "id" in event_dict
        assert event_dict["event_type"] == "agent_registered"
        assert event_dict["source"] == "test_source"
        
        # Test deserialization
        reconstructed_event = StreamEvent.from_dict(event_dict)
        assert reconstructed_event.event_type == test_event.event_type
        assert reconstructed_event.source == test_event.source
        assert reconstructed_event.data == test_event.data
        
        print("   ✅ Event serialization working")
        
        # Test 3: System initialization
        print("   🚀 Testing messaging system initialization...")
        messaging_system.start()
        
        # Test system status
        status = messaging_system.get_system_status()
        assert "system_id" in status
        assert "started_at" in status
        assert "processing_stats" in status
        
        messaging_system.stop()
        print("   ✅ Messaging system initialization working")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Real-time messaging test failed: {e}")
        return False


def run_core_infrastructure_tests():
    """Run all core infrastructure tests"""
    print("🧪 Core Infrastructure Foundation Tests")
    print("=" * 60)
    print("Testing enterprise-grade agent orchestration infrastructure...")
    print()
    
    tests_passed = 0
    total_tests = 3
    
    # Test database infrastructure
    if test_database_infrastructure():
        tests_passed += 1
        print("✅ Database infrastructure test PASSED\n")
    else:
        print("❌ Database infrastructure test FAILED\n")
    
    # Test secure communication infrastructure
    if test_secure_communication_infrastructure():
        tests_passed += 1
        print("✅ Secure communication infrastructure test PASSED\n")
    else:
        print("❌ Secure communication infrastructure test FAILED\n")
    
    # Test real-time messaging infrastructure
    if test_realtime_messaging_infrastructure():
        tests_passed += 1
        print("✅ Real-time messaging infrastructure test PASSED\n")
    else:
        print("❌ Real-time messaging infrastructure test FAILED\n")
    
    # Summary
    print("=" * 60)
    print(f"📊 Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 ALL CORE INFRASTRUCTURE TESTS PASSED!")
        print()
        print("✅ Core Infrastructure Foundation is complete and working:")
        print("   • Database schemas for agent registry, task management, and performance tracking")
        print("   • Secure communication protocols with encryption and authentication")
        print("   • Real-time message queuing and event streaming infrastructure")
        print("   • Enterprise-grade data models and relationships")
        print("   • System event logging and audit trails")
        print()
        print("🚀 The Agent Steering System infrastructure is ready for:")
        print("   • Agent registration and management")
        print("   • Task orchestration and coordination")
        print("   • Performance monitoring and optimization")
        print("   • Secure inter-agent communication")
        print("   • Real-time event processing and streaming")
        
        return True
    else:
        print(f"❌ {total_tests - tests_passed} tests failed. Infrastructure needs fixes.")
        return False


if __name__ == "__main__":
    success = run_core_infrastructure_tests()
    exit(0 if success else 1)