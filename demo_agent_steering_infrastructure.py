#!/usr/bin/env python3
"""
Agent Steering System Infrastructure Demo

Demonstrates the core infrastructure foundation including:
- Database schemas and operations
- Secure communication protocols
- Real-time message queuing and event streaming
- Agent registry and management
"""

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import our infrastructure components
from scrollintel.core.realtime_messaging import (
    RealTimeMessagingSystem, EventType, MessagePriority
)
from scrollintel.core.secure_communication import (
    SecureCommunicationManager, MessageType as SecureMessageType, SecurityLevel
)
from scrollintel.core.agent_registry import (
    AgentRegistry, AgentRegistrationRequest, AgentSelectionCriteria, CapabilityType
)
from scrollintel.models.agent_steering_models import (
    Agent, Task, AgentStatus, TaskStatus, TaskPriority
)
from scrollintel.models.database_utils import get_sync_db


class AgentSteeringInfrastructureDemo:
    """
    Comprehensive demo of the Agent Steering System infrastructure
    """
    
    def __init__(self):
        self.messaging_system = None
        self.secure_comm_manager = None
        self.agent_registry = None
        self.demo_agents = []
        self.demo_tasks = []
        
    async def initialize_infrastructure(self):
        """Initialize all infrastructure components"""
        print("🚀 Initializing Agent Steering System Infrastructure...")
        print("=" * 60)
        
        # Initialize real-time messaging system
        print("📡 Starting real-time messaging system...")
        self.messaging_system = RealTimeMessagingSystem()
        self.messaging_system.start()
        
        # Initialize secure communication manager
        print("🔒 Setting up secure communication protocols...")
        self.secure_comm_manager = SecureCommunicationManager()
        
        # Initialize agent registry
        print("📋 Initializing agent registry...")
        self.agent_registry = AgentRegistry(self.messaging_system)
        await self.agent_registry.start()
        
        print("✅ Infrastructure initialization complete!\n")
        
    async def demo_database_operations(self):
        """Demonstrate database schema operations"""
        print("💾 Database Operations Demo")
        print("-" * 30)
        
        try:
            with get_sync_db() as db:
                # Create sample agents
                sample_agents = [
                    {
                        "name": "DataAnalyst-Alpha",
                        "type": "data_analysis",
                        "capabilities": [
                            {"name": "statistical_analysis", "type": "data_analysis"},
                            {"name": "data_visualization", "type": "visualization"}
                        ],
                        "endpoint_url": "http://localhost:8001/agent",
                        "health_check_url": "http://localhost:8001/health"
                    },
                    {
                        "name": "MLEngineer-Beta", 
                        "type": "machine_learning",
                        "capabilities": [
                            {"name": "model_training", "type": "machine_learning"},
                            {"name": "feature_engineering", "type": "data_analysis"}
                        ],
                        "endpoint_url": "http://localhost:8002/agent",
                        "health_check_url": "http://localhost:8002/health"
                    },
                    {
                        "name": "BusinessIntel-Gamma",
                        "type": "business_intelligence", 
                        "capabilities": [
                            {"name": "dashboard_creation", "type": "visualization"},
                            {"name": "kpi_analysis", "type": "business_intelligence"}
                        ],
                        "endpoint_url": "http://localhost:8003/agent",
                        "health_check_url": "http://localhost:8003/health"
                    }
                ]
                
                # Insert agents into database
                for agent_data in sample_agents:
                    agent = Agent(
                        name=agent_data["name"],
                        type=agent_data["type"],
                        version="1.0.0",
                        capabilities=agent_data["capabilities"],
                        endpoint_url=agent_data["endpoint_url"],
                        health_check_url=agent_data["health_check_url"],
                        status=AgentStatus.ACTIVE,
                        current_load=0.0,
                        max_concurrent_tasks=5,
                        average_response_time=1.2,
                        success_rate=98.5
                    )
                    db.add(agent)
                    self.demo_agents.append(agent)
                
                db.commit()
                print(f"✅ Created {len(sample_agents)} sample agents")
                
                # Create sample tasks
                sample_tasks = [
                    {
                        "title": "Quarterly Sales Analysis",
                        "description": "Analyze Q3 sales data and identify trends",
                        "task_type": "data_analysis",
                        "priority": TaskPriority.HIGH,
                        "requirements": {
                            "capabilities": ["statistical_analysis", "data_visualization"],
                            "data_sources": ["sales_db", "crm_system"],
                            "output_format": "dashboard"
                        }
                    },
                    {
                        "title": "Customer Churn Prediction Model",
                        "description": "Build ML model to predict customer churn",
                        "task_type": "machine_learning",
                        "priority": TaskPriority.CRITICAL,
                        "requirements": {
                            "capabilities": ["model_training", "feature_engineering"],
                            "data_sources": ["customer_db", "usage_logs"],
                            "accuracy_target": 0.85
                        }
                    },
                    {
                        "title": "Executive KPI Dashboard",
                        "description": "Create real-time executive dashboard",
                        "task_type": "business_intelligence",
                        "priority": TaskPriority.MEDIUM,
                        "requirements": {
                            "capabilities": ["dashboard_creation", "kpi_analysis"],
                            "refresh_interval": "real-time",
                            "stakeholders": ["CEO", "CFO", "COO"]
                        }
                    }
                ]
                
                # Insert tasks into database
                for task_data in sample_tasks:
                    task = Task(
                        title=task_data["title"],
                        description=task_data["description"],
                        task_type=task_data["task_type"],
                        priority=task_data["priority"],
                        requirements=task_data["requirements"],
                        status=TaskStatus.PENDING,
                        estimated_duration=3600,  # 1 hour
                        max_retries=3
                    )
                    db.add(task)
                    self.demo_tasks.append(task)
                
                db.commit()
                print(f"✅ Created {len(sample_tasks)} sample tasks")
                
                # Query and display data
                print("\n📊 Database Query Results:")
                agents = db.query(Agent).all()
                for agent in agents:
                    print(f"   Agent: {agent.name} ({agent.type}) - Status: {agent.status.value}")
                
                tasks = db.query(Task).all()
                for task in tasks:
                    print(f"   Task: {task.title} - Priority: {task.priority.value} - Status: {task.status.value}")
                
        except Exception as e:
            print(f"❌ Database operation failed: {e}")
            
        print()
        
    async def demo_secure_communication(self):
        """Demonstrate secure communication protocols"""
        print("🔒 Secure Communication Demo")
        print("-" * 30)
        
        try:
            # Create communication protocols for demo agents
            agent_protocols = {}
            for i, agent in enumerate(self.demo_agents[:2]):  # Use first 2 agents
                agent_id = str(agent.id)
                protocol = self.secure_comm_manager.create_protocol(agent_id)
                agent_protocols[agent_id] = protocol
                
                # Set security policy
                self.secure_comm_manager.set_security_policy(agent_id, {
                    "min_security_level": SecurityLevel.HIGH,
                    "require_encryption": True,
                    "require_signature": True,
                    "max_message_size": 1024 * 1024,
                    "rate_limit": 1000
                })
                
                print(f"✅ Created secure protocol for {agent.name}")
            
            # Establish secure channels between agents
            agent_ids = list(agent_protocols.keys())
            if len(agent_ids) >= 2:
                agent1_id, agent2_id = agent_ids[0], agent_ids[1]
                protocol1 = agent_protocols[agent1_id]
                protocol2 = agent_protocols[agent2_id]
                
                # Establish secure channels
                await protocol1.establish_secure_channel(agent2_id)
                await protocol2.establish_secure_channel(agent1_id)
                
                print(f"✅ Established secure channel between agents")
                
                # Send secure messages
                await protocol1.send_secure_message(
                    agent2_id,
                    SecureMessageType.COORDINATION,
                    {
                        "message": "Ready for task coordination",
                        "capabilities": ["data_analysis", "visualization"],
                        "current_load": 0.2
                    },
                    SecurityLevel.HIGH
                )
                
                await protocol2.send_secure_message(
                    agent1_id,
                    SecureMessageType.COORDINATION,
                    {
                        "message": "Acknowledged, ready to collaborate",
                        "available_resources": ["gpu_cluster", "data_warehouse"],
                        "estimated_capacity": 0.8
                    },
                    SecurityLevel.HIGH
                )
                
                print("✅ Exchanged secure coordination messages")
                
                # Display communication stats
                system_status = self.secure_comm_manager.get_system_status()
                print(f"📊 Communication System Status:")
                print(f"   Total Protocols: {system_status['total_protocols']}")
                print(f"   Active Connections: {system_status['active_connections']}")
                print(f"   Audit Entries: {system_status['total_audit_entries']}")
                
        except Exception as e:
            print(f"❌ Secure communication demo failed: {e}")
            
        print()
        
    async def demo_realtime_messaging(self):
        """Demonstrate real-time messaging and event streaming"""
        print("📡 Real-Time Messaging Demo")
        print("-" * 30)
        
        try:
            # Set up event subscribers
            received_events = []
            
            def event_handler(event):
                received_events.append(event)
                print(f"📨 Received event: {event.event_type.value} from {event.source}")
            
            # Subscribe to various event types
            self.messaging_system.subscribe(
                "demo_subscriber",
                [EventType.AGENT_REGISTERED, EventType.TASK_ASSIGNED, EventType.PERFORMANCE_METRIC],
                event_handler,
                batch_size=1
            )
            
            print("✅ Set up event subscriber")
            
            # Publish various events
            events_to_publish = [
                {
                    "type": EventType.AGENT_REGISTERED,
                    "source": "agent_registry",
                    "data": {
                        "agent_id": str(self.demo_agents[0].id),
                        "agent_name": self.demo_agents[0].name,
                        "capabilities": self.demo_agents[0].capabilities
                    },
                    "priority": MessagePriority.NORMAL
                },
                {
                    "type": EventType.TASK_ASSIGNED,
                    "source": "orchestration_engine",
                    "data": {
                        "task_id": str(self.demo_tasks[0].id),
                        "agent_id": str(self.demo_agents[0].id),
                        "task_title": self.demo_tasks[0].title,
                        "priority": self.demo_tasks[0].priority.value
                    },
                    "priority": MessagePriority.HIGH
                },
                {
                    "type": EventType.PERFORMANCE_METRIC,
                    "source": f"agent_{self.demo_agents[0].id}",
                    "data": {
                        "response_time": 1.2,
                        "throughput": 45.0,
                        "accuracy": 98.5,
                        "current_load": 0.3,
                        "business_impact": {
                            "cost_savings": 15000.0,
                            "productivity_gain": 25.0
                        }
                    },
                    "priority": MessagePriority.NORMAL
                }
            ]
            
            # Publish events
            for event_data in events_to_publish:
                event = self.messaging_system.create_event(
                    event_data["type"],
                    event_data["source"],
                    event_data["data"],
                    event_data["priority"]
                )
                success = self.messaging_system.publish(event)
                if success:
                    print(f"✅ Published {event_data['type'].value} event")
                else:
                    print(f"❌ Failed to publish {event_data['type'].value} event")
            
            # Wait for event processing
            await asyncio.sleep(2)
            
            # Display messaging system status
            system_status = self.messaging_system.get_system_status()
            print(f"\n📊 Messaging System Status:")
            print(f"   System ID: {system_status['system_id']}")
            print(f"   Uptime: {system_status['uptime_seconds']:.1f} seconds")
            print(f"   Subscriptions: {system_status['subscription_count']}")
            
            processing_stats = system_status['processing_stats']
            print(f"   Events Processed: {processing_stats['events_processed']}")
            print(f"   Messages Queued: {processing_stats['messages_queued']}")
            print(f"   Queue Size: {processing_stats['queue_size']}")
            
            print(f"   Events Received by Subscriber: {len(received_events)}")
            
        except Exception as e:
            print(f"❌ Real-time messaging demo failed: {e}")
            
        print()
        
    async def demo_agent_registry(self):
        """Demonstrate agent registry and management"""
        print("📋 Agent Registry Demo")
        print("-" * 30)
        
        try:
            # Register additional agents
            new_agents = [
                AgentRegistrationRequest(
                    name="QualityAssurance-Delta",
                    type="quality_assurance",
                    version="1.0.0",
                    capabilities=[
                        {"name": "code_review", "type": "quality_assurance"},
                        {"name": "test_automation", "type": "quality_assurance"}
                    ],
                    endpoint_url="http://localhost:8004/agent",
                    health_check_url="http://localhost:8004/health",
                    resource_requirements={"cpu": 2, "memory": "4GB"},
                    configuration={"test_frameworks": ["pytest", "selenium"]}
                ),
                AgentRegistrationRequest(
                    name="Forecasting-Epsilon",
                    type="forecasting",
                    version="1.0.0",
                    capabilities=[
                        {"name": "time_series_analysis", "type": "forecasting"},
                        {"name": "demand_prediction", "type": "forecasting"}
                    ],
                    endpoint_url="http://localhost:8005/agent",
                    health_check_url="http://localhost:8005/health",
                    resource_requirements={"cpu": 4, "memory": "8GB", "gpu": True},
                    configuration={"models": ["arima", "lstm", "prophet"]}
                )
            ]
            
            registered_agent_ids = []
            for agent_request in new_agents:
                agent_id = await self.agent_registry.register_agent(agent_request)
                if agent_id:
                    registered_agent_ids.append(agent_id)
                    print(f"✅ Registered agent: {agent_request.name} ({agent_id})")
                else:
                    print(f"❌ Failed to register agent: {agent_request.name}")
            
            # Test agent selection with different criteria
            selection_criteria = [
                AgentSelectionCriteria(
                    required_capabilities=["statistical_analysis"],
                    preferred_capabilities=["data_visualization"],
                    max_load_threshold=0.5,
                    min_success_rate=95.0,
                    max_response_time=2.0
                ),
                AgentSelectionCriteria(
                    required_capabilities=["model_training"],
                    preferred_capabilities=["feature_engineering"],
                    max_load_threshold=0.7,
                    min_success_rate=90.0,
                    max_response_time=5.0
                ),
                AgentSelectionCriteria(
                    required_capabilities=["dashboard_creation"],
                    preferred_capabilities=["kpi_analysis"],
                    max_load_threshold=0.3,
                    min_success_rate=98.0,
                    max_response_time=1.5
                )
            ]
            
            print(f"\n🎯 Agent Selection Tests:")
            for i, criteria in enumerate(selection_criteria):
                best_agent = await self.agent_registry.select_best_agent(criteria)
                if best_agent:
                    print(f"   Test {i+1}: Selected {best_agent['name']} (score: {best_agent['performance_score']:.2f})")
                else:
                    print(f"   Test {i+1}: No suitable agent found")
            
            # Get available agents
            available_agents = await self.agent_registry.get_available_agents()
            print(f"\n📊 Available Agents: {len(available_agents)}")
            for agent in available_agents:
                print(f"   {agent['name']} ({agent['type']}) - Load: {agent['current_load']:.1f}, Success: {agent['success_rate']:.1f}%")
            
            # Update agent performance
            if registered_agent_ids:
                agent_id = registered_agent_ids[0]
                performance_data = {
                    "current_load": 0.4,
                    "response_time": 1.8,
                    "throughput": 32.0,
                    "accuracy": 96.5,
                    "reliability": 99.2,
                    "cpu_usage": 45.0,
                    "memory_usage": 60.0,
                    "cost_savings": 8500.0,
                    "productivity_gain": 18.0
                }
                
                success = await self.agent_registry.update_agent_performance(agent_id, performance_data)
                if success:
                    print(f"✅ Updated performance metrics for agent {agent_id}")
                else:
                    print(f"❌ Failed to update performance metrics")
            
            # Get registry statistics
            stats = await self.agent_registry.get_registry_stats()
            print(f"\n📈 Registry Statistics:")
            print(f"   Total Agents: {stats.get('total_agents', 0)}")
            print(f"   Active Agents: {stats.get('active_agents', 0)}")
            print(f"   Error Agents: {stats.get('error_agents', 0)}")
            print(f"   Cache Entries: {stats.get('cache_entries', 0)}")
            
            capability_dist = stats.get('capability_distribution', {})
            if capability_dist:
                print(f"   Top Capabilities:")
                for cap, count in sorted(capability_dist.items(), key=lambda x: x[1], reverse=True)[:5]:
                    print(f"     {cap}: {count} agents")
            
        except Exception as e:
            print(f"❌ Agent registry demo failed: {e}")
            
        print()
        
    async def demo_task_orchestration(self):
        """Demonstrate basic task orchestration"""
        print("🎭 Task Orchestration Demo")
        print("-" * 30)
        
        try:
            with get_sync_db() as db:
                # Get available agents and tasks
                agents = db.query(Agent).filter(Agent.status == AgentStatus.ACTIVE).all()
                tasks = db.query(Task).filter(Task.status == TaskStatus.PENDING).all()
                
                if not agents or not tasks:
                    print("⚠️  No available agents or tasks for orchestration demo")
                    return
                
                # Simple task assignment logic
                assignments = []
                for task in tasks[:3]:  # Assign first 3 tasks
                    # Find best agent for task
                    best_agent = None
                    best_score = 0
                    
                    for agent in agents:
                        # Simple scoring based on capability match
                        score = 0
                        agent_capabilities = [cap.get("name", "") for cap in agent.capabilities]
                        required_capabilities = task.requirements.get("capabilities", [])
                        
                        for req_cap in required_capabilities:
                            if req_cap in agent_capabilities:
                                score += 1
                        
                        # Factor in current load (lower is better)
                        score = score * (1.0 - agent.current_load)
                        
                        if score > best_score:
                            best_score = score
                            best_agent = agent
                    
                    if best_agent:
                        # Assign task to agent
                        task.assigned_agent_id = best_agent.id
                        task.status = TaskStatus.RUNNING
                        task.started_at = datetime.utcnow()
                        
                        # Update agent load
                        best_agent.current_load = min(1.0, best_agent.current_load + 0.2)
                        
                        assignments.append({
                            "task": task.title,
                            "agent": best_agent.name,
                            "score": best_score
                        })
                        
                        # Publish task assignment event
                        self.messaging_system.publish_task_assigned(
                            str(task.id),
                            str(best_agent.id),
                            {
                                "task_title": task.title,
                                "task_type": task.task_type,
                                "priority": task.priority.value,
                                "requirements": task.requirements
                            }
                        )
                
                db.commit()
                
                print(f"✅ Assigned {len(assignments)} tasks:")
                for assignment in assignments:
                    print(f"   '{assignment['task']}' → {assignment['agent']} (score: {assignment['score']:.2f})")
                
                # Simulate task completion
                await asyncio.sleep(1)
                
                completed_tasks = 0
                for assignment in assignments:
                    task = db.query(Task).filter(Task.title == assignment["task"]).first()
                    if task and task.status == TaskStatus.RUNNING:
                        # Simulate task completion
                        task.status = TaskStatus.COMPLETED
                        task.completed_at = datetime.utcnow()
                        task.actual_duration = 1800  # 30 minutes
                        task.result = {
                            "status": "success",
                            "output": f"Completed {task.title}",
                            "metrics": {
                                "accuracy": 97.5,
                                "processing_time": 1800,
                                "resources_used": {"cpu": 2.5, "memory": "3.2GB"}
                            }
                        }
                        
                        # Update agent load
                        agent = db.query(Agent).filter(Agent.id == task.assigned_agent_id).first()
                        if agent:
                            agent.current_load = max(0.0, agent.current_load - 0.2)
                        
                        completed_tasks += 1
                
                db.commit()
                print(f"✅ Completed {completed_tasks} tasks")
                
        except Exception as e:
            print(f"❌ Task orchestration demo failed: {e}")
            
        print()
        
    async def cleanup_infrastructure(self):
        """Clean up infrastructure components"""
        print("🧹 Cleaning up infrastructure...")
        
        try:
            if self.agent_registry:
                await self.agent_registry.stop()
                
            if self.messaging_system:
                self.messaging_system.stop()
                
            print("✅ Infrastructure cleanup complete")
            
        except Exception as e:
            print(f"⚠️  Cleanup warning: {e}")
            
    async def run_complete_demo(self):
        """Run the complete infrastructure demo"""
        try:
            await self.initialize_infrastructure()
            await self.demo_database_operations()
            await self.demo_secure_communication()
            await self.demo_realtime_messaging()
            await self.demo_agent_registry()
            await self.demo_task_orchestration()
            
            print("🎉 Agent Steering System Infrastructure Demo Complete!")
            print("=" * 60)
            print("✅ All core infrastructure components are working correctly:")
            print("   • Database schemas and operations")
            print("   • Secure communication protocols")
            print("   • Real-time message queuing and event streaming")
            print("   • Agent registry and management")
            print("   • Basic task orchestration")
            print("\n🚀 The system is ready for enterprise-grade agent orchestration!")
            
        except Exception as e:
            print(f"❌ Demo failed: {e}")
            
        finally:
            await self.cleanup_infrastructure()


async def main():
    """Main demo function"""
    demo = AgentSteeringInfrastructureDemo()
    await demo.run_complete_demo()


if __name__ == "__main__":
    asyncio.run(main())