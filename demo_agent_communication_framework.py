"""
Agent Communication Framework Demo

Demonstrates the secure agent-to-agent communication system with:
- Encrypted messaging between agents
- Collaboration session management
- Distributed state synchronization
- Resource locking and conflict resolution
"""

import asyncio
import json
from datetime import datetime, timedelta
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from scrollintel.core.agent_communication_framework import AgentCommunicationFramework
from scrollintel.models.agent_communication_models import (
    Base, AgentMessageCreate, CollaborationSessionCreate, ResourceLockRequest,
    StateUpdateRequest, MessageType, MessagePriority, SecurityLevel
)


async def demo_basic_messaging():
    """Demonstrate basic encrypted messaging between agents"""
    print("\n" + "="*60)
    print("DEMO: Basic Encrypted Messaging")
    print("="*60)
    
    # Setup
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    db = SessionLocal()
    
    framework = AgentCommunicationFramework(db, "demo_encryption_key_12345")
    
    # Agent 1 sends task request to Agent 2
    print("\n1. Agent 1 sends encrypted task request to Agent 2...")
    
    task_request = AgentMessageCreate(
        to_agent_id="data_scientist_agent",
        message_type=MessageType.TASK_REQUEST,
        content={
            "task_type": "data_analysis",
            "dataset": "customer_transactions_2024",
            "analysis_type": "correlation_analysis",
            "deadline": "2024-12-31T23:59:59Z",
            "priority": "high",
            "parameters": {
                "columns": ["amount", "category", "customer_age"],
                "method": "pearson"
            }
        },
        priority=MessagePriority.HIGH,
        security_level=SecurityLevel.CONFIDENTIAL,
        correlation_id="task_001"
    )
    
    response = await framework.send_message("ai_engineer_agent", task_request)
    print(f"   ✓ Message sent: {response.id}")
    print(f"   ✓ Encrypted content length: {len(response.encrypted_content) if hasattr(response, 'encrypted_content') else 'N/A'}")
    print(f"   ✓ Security level: {response.security_level}")
    
    # Agent 2 receives and processes message
    print("\n2. Agent 2 receives and decrypts message...")
    
    messages = await framework.receive_messages("data_scientist_agent", timeout=1.0)
    
    if messages:
        message = messages[0]
        print(f"   ✓ Received message from: {message['from_agent_id']}")
        print(f"   ✓ Task type: {message['content']['task_type']}")
        print(f"   ✓ Dataset: {message['content']['dataset']}")
        print(f"   ✓ Correlation ID: {message['correlation_id']}")
        
        # Acknowledge message
        await framework.acknowledge_message("data_scientist_agent", message['id'])
        print(f"   ✓ Message acknowledged")
        
        # Send response back
        print("\n3. Agent 2 sends analysis results back...")
        
        response_message = AgentMessageCreate(
            to_agent_id="ai_engineer_agent",
            message_type=MessageType.TASK_RESPONSE,
            content={
                "task_id": message['correlation_id'],
                "status": "completed",
                "results": {
                    "correlation_matrix": {
                        "amount_category": 0.73,
                        "amount_customer_age": -0.12,
                        "category_customer_age": 0.05
                    },
                    "insights": [
                        "Strong positive correlation between amount and category",
                        "Weak negative correlation between amount and customer age",
                        "No significant correlation between category and customer age"
                    ],
                    "confidence_score": 0.94
                },
                "execution_time": "2.3 seconds",
                "completed_at": datetime.utcnow().isoformat()
            },
            reply_to=message['id'],
            correlation_id=message['correlation_id']
        )
        
        await framework.send_message("data_scientist_agent", response_message)
        print(f"   ✓ Analysis results sent back")
    
    db.close()


async def demo_collaboration_session():
    """Demonstrate multi-agent collaboration session"""
    print("\n" + "="*60)
    print("DEMO: Multi-Agent Collaboration Session")
    print("="*60)
    
    # Setup
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    db = SessionLocal()
    
    framework = AgentCommunicationFramework(db, "demo_encryption_key_12345")
    
    # Create collaboration session
    print("\n1. CTO Agent creates collaboration session for system architecture review...")
    
    session_request = CollaborationSessionCreate(
        session_name="System Architecture Review Q4 2024",
        description="Quarterly review of system architecture and performance optimization",
        objective={
            "goal": "architecture_review",
            "scope": "entire_platform",
            "deliverables": [
                "performance_analysis",
                "scalability_recommendations", 
                "security_assessment",
                "cost_optimization_plan"
            ],
            "deadline": "2024-12-15T17:00:00Z"
        },
        participants=[
            "ai_engineer_agent",
            "data_scientist_agent", 
            "ml_engineer_agent",
            "qa_agent"
        ],
        security_level=SecurityLevel.CONFIDENTIAL,
        max_participants=6,
        session_timeout=7200  # 2 hours
    )
    
    session = await framework.create_collaboration_session("cto_agent", session_request)
    print(f"   ✓ Session created: {session.id}")
    print(f"   ✓ Session name: {session.session_name}")
    print(f"   ✓ Participants invited: {len(session_request.participants)}")
    
    session_id = session.id
    
    # Agents join the session
    print("\n2. Agents join the collaboration session...")
    
    agents = ["ai_engineer_agent", "data_scientist_agent", "ml_engineer_agent"]
    
    for agent in agents:
        success = await framework.join_collaboration_session(agent, session_id)
        if success:
            print(f"   ✓ {agent} joined session")
    
    # Collaborative discussion
    print("\n3. Agents collaborate on architecture analysis...")
    
    # AI Engineer shares performance metrics
    await framework.send_session_message(
        "ai_engineer_agent",
        session_id,
        {
            "type": "performance_report",
            "metrics": {
                "avg_response_time": "245ms",
                "throughput": "1,250 requests/second",
                "error_rate": "0.02%",
                "cpu_utilization": "68%",
                "memory_usage": "4.2GB"
            },
            "bottlenecks": [
                "Database query optimization needed",
                "Cache hit ratio could be improved",
                "Some API endpoints need refactoring"
            ]
        },
        MessageType.NOTIFICATION
    )
    print(f"   ✓ AI Engineer shared performance metrics")
    
    # Data Scientist provides analysis insights
    await framework.send_session_message(
        "data_scientist_agent",
        session_id,
        {
            "type": "data_analysis",
            "findings": {
                "user_behavior_patterns": {
                    "peak_hours": "9-11 AM, 2-4 PM",
                    "geographic_distribution": "60% US, 25% EU, 15% APAC",
                    "feature_usage": {
                        "dashboard": "85%",
                        "reports": "72%", 
                        "analytics": "45%"
                    }
                },
                "recommendations": [
                    "Implement regional caching for EU and APAC",
                    "Optimize dashboard queries for peak hours",
                    "Consider feature usage in architecture decisions"
                ]
            }
        },
        MessageType.NOTIFICATION
    )
    print(f"   ✓ Data Scientist shared analysis insights")
    
    # ML Engineer discusses model performance
    await framework.send_session_message(
        "ml_engineer_agent",
        session_id,
        {
            "type": "ml_performance_report",
            "model_metrics": {
                "prediction_accuracy": "94.2%",
                "inference_latency": "12ms",
                "model_size": "245MB",
                "training_time": "3.5 hours"
            },
            "scaling_considerations": [
                "Model serving infrastructure needs horizontal scaling",
                "Consider model quantization for edge deployment",
                "Implement A/B testing framework for model updates"
            ]
        },
        MessageType.NOTIFICATION
    )
    print(f"   ✓ ML Engineer shared model performance data")
    
    db.close()


async def demo_distributed_state_sync():
    """Demonstrate distributed state synchronization"""
    print("\n" + "="*60)
    print("DEMO: Distributed State Synchronization")
    print("="*60)
    
    # Setup
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    db = SessionLocal()
    
    framework = AgentCommunicationFramework(db, "demo_encryption_key_12345")
    
    session_id = "distributed_processing_session"
    
    print("\n1. Multiple agents update shared processing state...")
    
    # Agent 1 initializes processing state
    state_request_1 = StateUpdateRequest(
        state_key="processing_progress",
        state_namespace="data_pipeline",
        state_value={
            "total_records": 1000000,
            "processed_records": 0,
            "processing_rate": 0,
            "estimated_completion": None,
            "agents": {
                "ai_engineer_agent": {"status": "initializing", "records_processed": 0}
            }
        }
    )
    
    state_1 = await framework.update_shared_state("ai_engineer_agent", state_request_1, session_id)
    print(f"   ✓ AI Engineer initialized processing state (version {state_1.state_version})")
    
    # Agent 2 updates with its progress
    await asyncio.sleep(0.1)  # Simulate processing time
    
    current_state = await framework.get_shared_state("data_scientist_agent", "processing_progress", "data_pipeline")
    
    updated_state = current_state.state_value.copy()
    updated_state["processed_records"] = 25000
    updated_state["processing_rate"] = 5000  # records per minute
    updated_state["agents"]["data_scientist_agent"] = {
        "status": "processing",
        "records_processed": 25000,
        "start_time": datetime.utcnow().isoformat()
    }
    
    state_request_2 = StateUpdateRequest(
        state_key="processing_progress",
        state_namespace="data_pipeline",
        state_value=updated_state,
        expected_version=current_state.state_version
    )
    
    state_2 = await framework.update_shared_state("data_scientist_agent", state_request_2, session_id)
    print(f"   ✓ Data Scientist updated progress (version {state_2.state_version})")
    
    # Agent 3 adds its contribution
    await asyncio.sleep(0.1)
    
    current_state = await framework.get_shared_state("ml_engineer_agent", "processing_progress", "data_pipeline")
    
    updated_state = current_state.state_value.copy()
    updated_state["processed_records"] = 50000
    updated_state["processing_rate"] = 7500
    updated_state["agents"]["ml_engineer_agent"] = {
        "status": "processing",
        "records_processed": 25000,
        "start_time": datetime.utcnow().isoformat()
    }
    
    # Calculate estimated completion
    remaining_records = updated_state["total_records"] - updated_state["processed_records"]
    completion_time = datetime.utcnow() + timedelta(minutes=remaining_records / updated_state["processing_rate"])
    updated_state["estimated_completion"] = completion_time.isoformat()
    
    state_request_3 = StateUpdateRequest(
        state_key="processing_progress",
        state_namespace="data_pipeline",
        state_value=updated_state,
        expected_version=current_state.state_version
    )
    
    state_3 = await framework.update_shared_state("ml_engineer_agent", state_request_3, session_id)
    print(f"   ✓ ML Engineer updated progress with completion estimate (version {state_3.state_version})")
    
    # Synchronize state across all agents
    print("\n2. Synchronizing state across all agents...")
    
    synchronized_state = await framework.synchronize_state("ai_engineer_agent", session_id)
    
    print(f"   ✓ State synchronized across {len(synchronized_state)} state entries")
    
    for key, state_info in synchronized_state.items():
        print(f"   ✓ {key}: version {state_info['version']}, last modified by {state_info['last_modified_by']}")
    
    db.close()


async def demo_resource_locking():
    """Demonstrate resource locking and conflict resolution"""
    print("\n" + "="*60)
    print("DEMO: Resource Locking and Conflict Resolution")
    print("="*60)
    
    # Setup
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    db = SessionLocal()
    
    framework = AgentCommunicationFramework(db, "demo_encryption_key_12345")
    
    print("\n1. Multiple agents request locks on shared resources...")
    
    # Agent 1 requests exclusive lock on database
    print("\n   AI Engineer requests exclusive lock on customer database...")
    
    lock_request_1 = ResourceLockRequest(
        resource_id="customer_database",
        resource_type="database",
        lock_type="exclusive",
        lock_reason="Performing data migration and schema updates",
        lock_duration_seconds=300,  # 5 minutes
        priority=2
    )
    
    lock_1 = await framework.request_resource_lock("ai_engineer_agent", lock_request_1)
    print(f"   ✓ Lock {lock_1.id} - Status: {lock_1.status}")
    
    # Agent 2 requests lock on same database (should be queued)
    print("\n   Data Scientist requests lock on same database...")
    
    lock_request_2 = ResourceLockRequest(
        resource_id="customer_database",
        resource_type="database", 
        lock_type="exclusive",
        lock_reason="Running analytics queries",
        priority=1
    )
    
    lock_2 = await framework.request_resource_lock("data_scientist_agent", lock_request_2)
    print(f"   ✓ Lock {lock_2.id} - Status: {lock_2.status} (queued due to conflict)")
    
    # Agent 3 requests shared lock on different resource
    print("\n   ML Engineer requests shared lock on model repository...")
    
    lock_request_3 = ResourceLockRequest(
        resource_id="model_repository",
        resource_type="file_system",
        lock_type="shared",
        lock_reason="Reading model artifacts for inference"
    )
    
    lock_3 = await framework.request_resource_lock("ml_engineer_agent", lock_request_3)
    print(f"   ✓ Lock {lock_3.id} - Status: {lock_3.status}")
    
    # Agent 4 requests another shared lock on same resource (should be granted)
    print("\n   QA Agent requests shared lock on model repository...")
    
    lock_request_4 = ResourceLockRequest(
        resource_id="model_repository",
        resource_type="file_system",
        lock_type="shared",
        lock_reason="Model validation and testing"
    )
    
    lock_4 = await framework.request_resource_lock("qa_agent", lock_request_4)
    print(f"   ✓ Lock {lock_4.id} - Status: {lock_4.status} (shared locks compatible)")
    
    # Check lock status
    print("\n2. Checking lock status...")
    
    active_lock = await framework.check_resource_lock("ai_engineer_agent", "customer_database")
    if active_lock:
        print(f"   ✓ AI Engineer holds lock on customer_database: {active_lock.id}")
    
    # Release lock and observe queue processing
    print("\n3. AI Engineer releases database lock...")
    
    success = await framework.release_resource_lock("ai_engineer_agent", lock_1.id)
    if success:
        print(f"   ✓ Lock {lock_1.id} released")
        print(f"   ✓ Queued lock for Data Scientist should now be processed")
    
    # Verify Data Scientist now has the lock
    await asyncio.sleep(0.1)  # Give time for queue processing
    
    new_active_lock = await framework.check_resource_lock("data_scientist_agent", "customer_database")
    if new_active_lock:
        print(f"   ✓ Data Scientist now holds lock: {new_active_lock.id}")
    
    db.close()


async def demo_agent_status_monitoring():
    """Demonstrate agent status monitoring"""
    print("\n" + "="*60)
    print("DEMO: Agent Status Monitoring")
    print("="*60)
    
    # Setup
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    db = SessionLocal()
    
    framework = AgentCommunicationFramework(db, "demo_encryption_key_12345")
    
    # Create some activity for monitoring
    print("\n1. Creating agent activity for monitoring...")
    
    # Send some messages
    for i in range(3):
        message = AgentMessageCreate(
            to_agent_id="monitoring_agent",
            message_type=MessageType.NOTIFICATION,
            content={"message": f"Test message {i+1}"}
        )
        await framework.send_message("system", message)
    
    # Create a session
    session_request = CollaborationSessionCreate(
        session_name="Monitoring Test Session",
        objective={"goal": "testing"},
        participants=["monitoring_agent"]
    )
    
    session = await framework.create_collaboration_session("system", session_request)
    await framework.join_collaboration_session("monitoring_agent", session.id)
    
    # Request a resource lock
    lock_request = ResourceLockRequest(
        resource_id="test_resource",
        resource_type="service",
        lock_type="exclusive"
    )
    
    await framework.request_resource_lock("monitoring_agent", lock_request)
    
    # Get agent status
    print("\n2. Checking agent status...")
    
    status = await framework.get_agent_status("monitoring_agent")
    
    print(f"   ✓ Agent ID: {status['agent_id']}")
    print(f"   ✓ Pending messages: {status['pending_messages']}")
    print(f"   ✓ Active sessions: {status['active_sessions']}")
    print(f"   ✓ Held locks: {status['held_locks']}")
    print(f"   ✓ Status: {status['status']}")
    
    # Cleanup expired resources
    print("\n3. Running cleanup process...")
    
    await framework.cleanup_expired_resources()
    print(f"   ✓ Cleanup completed")
    
    db.close()


async def main():
    """Run all demonstrations"""
    print("Agent Communication Framework - Comprehensive Demo")
    print("=" * 80)
    print("Demonstrating enterprise-grade agent-to-agent communication with:")
    print("• Secure, encrypted messaging")
    print("• Multi-agent collaboration sessions")
    print("• Distributed state synchronization")
    print("• Resource locking and conflict resolution")
    print("• Real-time monitoring and status tracking")
    
    try:
        await demo_basic_messaging()
        await demo_collaboration_session()
        await demo_distributed_state_sync()
        await demo_resource_locking()
        await demo_agent_status_monitoring()
        
        print("\n" + "="*80)
        print("✅ ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY")
        print("="*80)
        print("\nThe Agent Communication Framework provides:")
        print("• End-to-end encryption for all agent communications")
        print("• Scalable collaboration session management")
        print("• Conflict-free distributed state synchronization")
        print("• Intelligent resource locking with automatic queue processing")
        print("• Comprehensive monitoring and status tracking")
        print("• Enterprise-grade security and reliability")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())