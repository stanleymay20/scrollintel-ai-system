"""
Demo script for ScrollIntel orchestration and task coordination system.
Shows multi-agent workflows, dependency management, and message passing.
"""

import asyncio
import logging
from datetime import datetime
from typing import List

from scrollintel.core.orchestrator import TaskOrchestrator, WorkflowStatus
from scrollintel.core.workflow_templates import WorkflowTemplateLibrary
from scrollintel.core.registry import AgentRegistry
from scrollintel.core.message_bus import MessageBus, MessageType, MessagePriority, initialize_message_bus
from scrollintel.core.interfaces import (
    BaseAgent,
    AgentType,
    AgentStatus,
    AgentRequest,
    AgentResponse,
    ResponseStatus,
    AgentCapability,
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DemoAgent(BaseAgent):
    """Demo agent that simulates real agent behavior."""
    
    def __init__(self, agent_id: str, name: str, agent_type: AgentType, processing_time: float = 1.0):
        super().__init__(agent_id, name, agent_type)
        self.processing_time = processing_time
        self.processed_requests = []
    
    async def process_request(self, request: AgentRequest) -> AgentResponse:
        """Process a request with simulated work."""
        logger.info(f"🤖 {self.name} processing: {request.prompt}")
        self.processed_requests.append(request)
        
        # Simulate processing time
        await asyncio.sleep(self.processing_time)
        
        # Generate response based on agent type and context
        content = self._generate_response(request)
        artifacts = [f"{self.agent_type.value}_output_{len(self.processed_requests)}.json"]
        
        logger.info(f"✅ {self.name} completed: {request.prompt[:50]}...")
        
        return AgentResponse(
            id=f"response_{request.id}",
            request_id=request.id,
            content=content,
            artifacts=artifacts,
            execution_time=self.processing_time,
            status=ResponseStatus.SUCCESS,
        )
    
    def _generate_response(self, request: AgentRequest) -> str:
        """Generate a response based on agent type and request context."""
        task_type = request.context.get("task_type", "general")
        
        responses = {
            AgentType.DATA_SCIENTIST: {
                "data_validation": f"✓ Dataset validated: {request.context.get('dataset_rows', 1000)} rows, {request.context.get('dataset_cols', 10)} columns. Quality score: 85%",
                "eda": "📊 EDA completed: Found 3 key patterns, 2 outlier clusters, strong correlation between features A-C",
                "feature_engineering": "🔧 Created 15 new features, selected top 8 using mutual information. Feature importance calculated.",
                "general": f"📈 Data analysis completed for: {request.prompt}",
            },
            AgentType.ML_ENGINEER: {
                "model_training": "🎯 Trained 5 models: RandomForest (0.89), XGBoost (0.91), Neural Net (0.87). Best: XGBoost",
                "model_evaluation": "📋 Model evaluation: Precision=0.91, Recall=0.88, F1=0.89. Cross-validation score: 0.90±0.02",
                "general": f"🤖 ML engineering completed for: {request.prompt}",
            },
            AgentType.AI_ENGINEER: {
                "model_deployment": "🚀 Model deployed to production. API endpoint: /predict. Health check: ✓",
                "general": f"⚡ AI engineering completed for: {request.prompt}",
            },
            AgentType.ANALYST: {
                "business_analysis": "💼 Business metrics analyzed: Revenue +15%, Customer satisfaction 4.2/5, Churn rate -8%",
                "kpi_generation": "📊 KPIs generated: 12 key metrics, 5 alerts configured, dashboard updated",
                "general": f"📈 Business analysis completed for: {request.prompt}",
            },
            AgentType.BI_DEVELOPER: {
                "dashboard_creation": "📱 Interactive dashboard created: 8 charts, 3 filters, real-time updates enabled",
                "visualization": "🎨 Visualizations created: 5 charts, 2 maps, 1 interactive timeline",
                "general": f"📊 BI development completed for: {request.prompt}",
            },
        }
        
        agent_responses = responses.get(self.agent_type, {})
        return agent_responses.get(task_type, agent_responses.get("general", f"Task completed: {request.prompt}"))
    
    def get_capabilities(self) -> List[AgentCapability]:
        """Return agent capabilities."""
        capabilities_map = {
            AgentType.DATA_SCIENTIST: ["data_analysis", "statistical_modeling", "feature_engineering"],
            AgentType.ML_ENGINEER: ["model_training", "hyperparameter_tuning", "model_evaluation"],
            AgentType.AI_ENGINEER: ["model_deployment", "api_development", "system_integration"],
            AgentType.ANALYST: ["business_analysis", "kpi_generation", "reporting"],
            AgentType.BI_DEVELOPER: ["dashboard_creation", "data_visualization", "reporting"],
        }
        
        caps = capabilities_map.get(self.agent_type, ["general"])
        return [
            AgentCapability(
                name=cap,
                description=f"{cap.replace('_', ' ').title()} capability",
                input_types=["text", "json"],
                output_types=["text", "json", "visualization"],
            )
            for cap in caps
        ]
    
    async def health_check(self) -> bool:
        """Health check."""
        return self.status == AgentStatus.ACTIVE


async def demo_workflow_templates():
    """Demo workflow templates functionality."""
    print("\n" + "="*60)
    print("🎯 DEMO: Workflow Templates")
    print("="*60)
    
    # Get all available templates
    templates = WorkflowTemplateLibrary.get_all_templates()
    
    print(f"📋 Available workflow templates: {len(templates)}")
    for template_id, template in templates.items():
        print(f"  • {template.name} ({template_id})")
        print(f"    └─ {template.description}")
        print(f"    └─ {len(template.tasks)} tasks, ~{template.estimated_duration/60:.1f} min")
    
    return templates


async def demo_custom_workflow():
    """Demo custom workflow creation and execution."""
    print("\n" + "="*60)
    print("🚀 DEMO: Custom Workflow Execution")
    print("="*60)
    
    # Create registry and orchestrator
    registry = AgentRegistry()
    orchestrator = TaskOrchestrator(registry)
    
    # Create demo agents
    agents = [
        DemoAgent("data_scientist", "ScrollDataScientist", AgentType.DATA_SCIENTIST, 0.5),
        DemoAgent("ml_engineer", "ScrollMLEngineer", AgentType.ML_ENGINEER, 0.8),
        DemoAgent("analyst", "ScrollAnalyst", AgentType.ANALYST, 0.3),
        DemoAgent("bi_developer", "ScrollBI", AgentType.BI_DEVELOPER, 0.6),
    ]
    
    # Register agents
    print("🔧 Registering agents...")
    for agent in agents:
        await registry.register_agent(agent)
        print(f"  ✓ {agent.name} ({agent.agent_type.value})")
    
    # Create custom workflow
    print("\n📝 Creating custom workflow...")
    tasks = [
        {
            "name": "Data Analysis & Validation",
            "agent_type": "data_scientist",
            "prompt": "Analyze the customer dataset and validate data quality",
            "context": {"task_type": "data_validation", "dataset_rows": 50000, "dataset_cols": 25},
            "timeout": 10.0,
        },
        {
            "name": "Business Metrics Analysis",
            "agent_type": "analyst",
            "prompt": "Analyze business metrics and identify key trends",
            "context": {"task_type": "business_analysis"},
            "dependencies": [{"task_id": "0", "dependency_type": "completion"}],
            "timeout": 10.0,
        },
        {
            "name": "ML Model Training",
            "agent_type": "ml_engineer",
            "prompt": "Train predictive models based on the analyzed data",
            "context": {"task_type": "model_training"},
            "dependencies": [{"task_id": "0", "dependency_type": "completion"}],
            "timeout": 10.0,
        },
        {
            "name": "Dashboard Creation",
            "agent_type": "bi_developer",
            "prompt": "Create interactive dashboard with insights and model results",
            "context": {"task_type": "dashboard_creation"},
            "dependencies": [
                {"task_id": "1", "dependency_type": "completion"},
                {"task_id": "2", "dependency_type": "completion"}
            ],
            "timeout": 10.0,
        },
    ]
    
    workflow_id = await orchestrator.create_custom_workflow(
        name="Customer Analytics Pipeline",
        tasks=tasks,
        context={"project": "customer_analytics", "version": "1.0"},
        description="End-to-end customer analytics with ML and visualization",
    )
    
    print(f"✓ Workflow created: {workflow_id}")
    
    # Show workflow status before execution
    status = orchestrator.get_workflow_status(workflow_id)
    print(f"📊 Initial status: {status['status']}")
    print(f"📋 Tasks: {len(status['tasks'])}")
    
    # Execute workflow
    print("\n🎬 Executing workflow...")
    print("⏳ This will take a few seconds as tasks run with dependencies...")
    
    start_time = datetime.now()
    result = await orchestrator.execute_workflow(workflow_id)
    end_time = datetime.now()
    
    execution_time = (end_time - start_time).total_seconds()
    
    # Show results
    print(f"\n🎉 Workflow completed!")
    print(f"⏱️  Total execution time: {execution_time:.2f} seconds")
    print(f"📊 Status: {result['status']}")
    print(f"📈 Progress: {result['progress']}%")
    
    # Show task details
    final_status = orchestrator.get_workflow_status(workflow_id)
    print(f"\n📋 Task Results:")
    for i, task in enumerate(final_status['tasks']):
        print(f"  {i+1}. {task['name']}")
        print(f"     Status: {task['status']}")
        print(f"     Time: {task['execution_time']:.2f}s")
    
    # Show agent activity
    print(f"\n🤖 Agent Activity:")
    for agent in agents:
        print(f"  • {agent.name}: {len(agent.processed_requests)} requests processed")
    
    return orchestrator, workflow_id


async def demo_template_workflow():
    """Demo workflow from template."""
    print("\n" + "="*60)
    print("📋 DEMO: Template-Based Workflow")
    print("="*60)
    
    # Create registry and orchestrator
    registry = AgentRegistry()
    orchestrator = TaskOrchestrator(registry)
    
    # Create demo agents
    agents = [
        DemoAgent("data_scientist", "ScrollDataScientist", AgentType.DATA_SCIENTIST, 0.3),
        DemoAgent("ml_engineer", "ScrollMLEngineer", AgentType.ML_ENGINEER, 0.4),
        DemoAgent("bi_developer", "ScrollBI", AgentType.BI_DEVELOPER, 0.3),
    ]
    
    # Register agents
    print("🔧 Registering agents...")
    for agent in agents:
        await registry.register_agent(agent)
    
    # Create workflow from template
    print("\n📋 Creating workflow from 'data_science_pipeline' template...")
    workflow_id = await orchestrator.create_workflow_from_template(
        template_id="data_science_pipeline",
        name="Sales Data Science Pipeline",
        context={"dataset": "sales_data.csv", "target": "revenue"},
    )
    
    print(f"✓ Workflow created from template: {workflow_id}")
    
    # Execute workflow
    print("\n🎬 Executing template workflow...")
    start_time = datetime.now()
    result = await orchestrator.execute_workflow(workflow_id)
    end_time = datetime.now()
    
    execution_time = (end_time - start_time).total_seconds()
    
    print(f"\n🎉 Template workflow completed!")
    print(f"⏱️  Execution time: {execution_time:.2f} seconds")
    print(f"📊 Status: {result['status']}")
    
    return orchestrator, workflow_id


async def demo_message_bus():
    """Demo message bus functionality."""
    print("\n" + "="*60)
    print("📨 DEMO: Message Bus System")
    print("="*60)
    
    # Initialize message bus
    message_bus = await initialize_message_bus()
    
    print("🚀 Message bus started")
    
    # Send some messages
    print("\n📤 Sending messages...")
    
    # Send normal priority message
    msg_id1 = await message_bus.send_message(
        sender_id="demo_user",
        recipient_id="data_scientist",
        message_type=MessageType.REQUEST,
        payload={"task": "analyze_data", "dataset": "customer_data.csv"},
        priority=MessagePriority.NORMAL,
    )
    print(f"  ✓ Sent normal priority message: {msg_id1[:8]}...")
    
    # Send high priority message
    msg_id2 = await message_bus.send_message(
        sender_id="demo_user",
        recipient_id="ml_engineer",
        message_type=MessageType.REQUEST,
        payload={"task": "urgent_model_fix", "issue": "production_error"},
        priority=MessagePriority.HIGH,
    )
    print(f"  ✓ Sent high priority message: {msg_id2[:8]}...")
    
    # Broadcast message
    broadcast_ids = await message_bus.broadcast_message(
        sender_id="system",
        message_type=MessageType.EVENT,
        payload={"event": "system_maintenance", "scheduled": "2024-01-15T02:00:00Z"},
        priority=MessagePriority.NORMAL,
    )
    print(f"  ✓ Broadcast message to {len(broadcast_ids)} recipients")
    
    # Show message bus stats
    await asyncio.sleep(0.1)  # Let messages process
    stats = message_bus.get_stats()
    print(f"\n📊 Message Bus Stats:")
    print(f"  • Messages sent: {stats['messages_sent']}")
    print(f"  • Messages delivered: {stats['messages_delivered']}")
    print(f"  • Pending messages: {stats['pending_messages']}")
    print(f"  • Subscribers: {stats['subscribers']}")
    
    return message_bus


async def demo_workflow_monitoring():
    """Demo workflow monitoring and progress tracking."""
    print("\n" + "="*60)
    print("📊 DEMO: Workflow Monitoring")
    print("="*60)
    
    # Create registry and orchestrator
    registry = AgentRegistry()
    orchestrator = TaskOrchestrator(registry)
    
    # Create agents
    agents = [
        DemoAgent("data_scientist", "ScrollDataScientist", AgentType.DATA_SCIENTIST, 1.0),
        DemoAgent("ml_engineer", "ScrollMLEngineer", AgentType.ML_ENGINEER, 1.5),
    ]
    
    for agent in agents:
        await registry.register_agent(agent)
    
    # Progress tracking
    progress_updates = []
    
    def progress_callback(workflow_id: str, progress: float):
        progress_updates.append((workflow_id, progress))
        print(f"📈 Progress update: {progress:.1f}%")
    
    # Create workflow
    tasks = [
        {
            "name": "Data Processing",
            "agent_type": "data_scientist",
            "prompt": "Process the input data",
            "timeout": 5.0,
        },
        {
            "name": "Model Training",
            "agent_type": "ml_engineer",
            "prompt": "Train the ML model",
            "dependencies": [{"task_id": "0", "dependency_type": "completion"}],
            "timeout": 5.0,
        },
    ]
    
    workflow_id = await orchestrator.create_custom_workflow(
        name="Monitored Workflow",
        tasks=tasks,
    )
    
    # Register progress callback
    orchestrator.register_progress_callback(workflow_id, progress_callback)
    
    print("🎬 Executing workflow with progress monitoring...")
    await orchestrator.execute_workflow(workflow_id)
    
    print(f"\n📊 Progress tracking completed:")
    print(f"  • Total progress updates: {len(progress_updates)}")
    print(f"  • Final progress: {progress_updates[-1][1] if progress_updates else 0}%")
    
    return orchestrator


async def main():
    """Run all orchestration demos."""
    print("🎯 ScrollIntel Orchestration System Demo")
    print("=" * 60)
    
    try:
        # Demo 1: Workflow Templates
        await demo_workflow_templates()
        
        # Demo 2: Custom Workflow
        await demo_custom_workflow()
        
        # Demo 3: Template Workflow
        await demo_template_workflow()
        
        # Demo 4: Message Bus
        message_bus = await demo_message_bus()
        
        # Demo 5: Workflow Monitoring
        await demo_workflow_monitoring()
        
        print("\n" + "="*60)
        print("🎉 All demos completed successfully!")
        print("="*60)
        
        # Cleanup
        await message_bus.stop()
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())