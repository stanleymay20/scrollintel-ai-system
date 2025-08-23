"""
Real-Time Orchestration Engine for Agent Steering System
Requirements: 1.1, 1.2, 1.3
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from uuid import uuid4
from collections import deque

from .agent_registry import AgentRegistry, AgentSelectionCriteria
from .message_bus import MessageBus
from .interfaces import AgentRequest, AgentResponse, ResponseStatus

logger = logging.getLogger(__name__)


class TaskPriority(int, Enum):
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4
    EMERGENCY = 5


class TaskStatus(str, Enum):
    PENDING = "pending"
    QUEUED = "queued"
    ASSIGNED = "assigned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class CollaborationMode(str, Enum):
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    PIPELINE = "pipeline"
    CONSENSUS = "consensus"
    COMPETITIVE = "competitive"


@dataclass
class OrchestrationTask:
    id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    description: str = ""
    priority: TaskPriority = TaskPriority.NORMAL
    status: TaskStatus = TaskStatus.PENDING
    required_capabilities: List[str] = field(default_factory=list)
    payload: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    timeout_seconds: float = 300.0
    collaboration_mode: CollaborationMode = CollaborationMode.SEQUENTIAL
    collaboration_agents: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    assigned_agent_id: Optional[str] = None
    result: Optional[AgentResponse] = None
    error_message: Optional[str] = None

class 
RealTimeOrchestrationEngine:
    """Main Real-Time Orchestration Engine"""
    
    def __init__(self, agent_registry: AgentRegistry, message_bus: MessageBus):
        self.agent_registry = agent_registry
        self.message_bus = message_bus
        self.active_tasks: Dict[str, OrchestrationTask] = {}
        self.task_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self.completed_tasks: deque = deque(maxlen=1000)
        self._running = False
        self._processor_task: Optional[asyncio.Task] = None
        self.engine_stats = {
            "tasks_processed": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "uptime_start": None
        }
    
    async def start(self):
        if self._running:
            return
        self._running = True
        self.engine_stats["uptime_start"] = datetime.utcnow()
        await self.message_bus.start()
        self._processor_task = asyncio.create_task(self._process_tasks())
        logger.info("Real-Time Orchestration Engine started")
    
    async def stop(self):
        if not self._running:
            return
        self._running = False
        if self._processor_task:
            self._processor_task.cancel()
        logger.info("Real-Time Orchestration Engine stopped")
    
    async def submit_task(
        self,
        name: str,
        description: str,
        required_capabilities: List[str],
        payload: Dict[str, Any],
        priority: TaskPriority = TaskPriority.NORMAL,
        collaboration_mode: CollaborationMode = CollaborationMode.SEQUENTIAL,
        collaboration_agents: Optional[List[str]] = None,
        timeout_seconds: float = 300.0,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        task = OrchestrationTask(
            name=name,
            description=description,
            priority=priority,
            required_capabilities=required_capabilities,
            payload=payload,
            context=context or {},
            timeout_seconds=timeout_seconds,
            collaboration_mode=collaboration_mode,
            collaboration_agents=collaboration_agents or []
        )
        
        self.active_tasks[task.id] = task
        priority_value = -task.priority.value
        await self.task_queue.put((priority_value, task.created_at, task))
        task.status = TaskStatus.QUEUED
        self.engine_stats["tasks_processed"] += 1
        
        return task.id    

    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        task = self.active_tasks.get(task_id)
        if not task:
            for completed_task in self.completed_tasks:
                if completed_task.id == task_id:
                    task = completed_task
                    break
        
        if not task:
            return None
        
        return {
            "id": task.id,
            "name": task.name,
            "description": task.description,
            "status": task.status.value,
            "priority": task.priority.name,
            "assigned_agent_id": task.assigned_agent_id,
            "collaboration_mode": task.collaboration_mode.value,
            "collaboration_agents": task.collaboration_agents,
            "created_at": task.created_at.isoformat(),
            "error_message": task.error_message,
            "execution_metrics": {}
        }
    
    async def cancel_task(self, task_id: str) -> bool:
        task = self.active_tasks.get(task_id)
        if not task or task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
            return False
        
        task.status = TaskStatus.CANCELLED
        self.completed_tasks.append(task)
        del self.active_tasks[task_id]
        return True
    
    async def _process_tasks(self):
        while self._running:
            try:
                try:
                    _, _, task = await asyncio.wait_for(self.task_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                
                if task.id not in self.active_tasks:
                    continue
                
                await self._execute_task(task)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in task processor: {str(e)}")
    
    async def _execute_task(self, task: OrchestrationTask):
        try:
            task.status = TaskStatus.RUNNING
            await asyncio.sleep(0.1)  # Simulate processing
            
            mock_response = AgentResponse(
                id=str(uuid4()),
                request_id=str(uuid4()),
                agent_id="mock-agent",
                status=ResponseStatus.SUCCESS,
                content={"result": f"Completed: {task.name}"}
            )
            
            task.status = TaskStatus.COMPLETED
            task.result = mock_response
            self.engine_stats["tasks_completed"] += 1
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error_message = str(e)
            self.engine_stats["tasks_failed"] += 1
        
        finally:
            self.completed_tasks.append(task)
            if task.id in self.active_tasks:
                del self.active_tasks[task.id]
    
    def get_engine_stats(self) -> Dict[str, Any]:
        uptime = None
        if self.engine_stats["uptime_start"]:
            uptime = (datetime.utcnow() - self.engine_stats["uptime_start"]).total_seconds()
        
        return {
            "engine_status": {"running": self._running, "uptime_seconds": uptime},
            "task_statistics": {**self.engine_stats, "active_tasks": len(self.active_tasks)},
            "distribution_stats": {},
            "load_balancing_stats": {},
            "coordination_stats": {},
            "component_status": {"message_bus": "active" if self.message_bus._running else "inactive"}
        }
    
    def get_active_tasks(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": task.id,
                "name": task.name,
                "status": task.status.value,
                "priority": task.priority.name,
                "created_at": task.created_at.isoformat()
            }
            for task in self.active_tasks.values()
        ]