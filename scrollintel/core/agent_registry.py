"""
Agent Registry and Management System

This module implements the central agent registry for dynamic agent discovery,
capability matching, performance-based selection, health monitoring, and
lifecycle management for the Agent Steering System.

Requirements: 1.1, 1.2, 6.1
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
from concurrent.futures import ThreadPoolExecutor
import threading

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete, and_, or_
from sqlalchemy.orm import selectinload

from ..models.agent_steering_models import (
    Agent, AgentCapability, AgentStatus, AgentPerformanceMetrics,
    AgentHealthCheck, AgentConfiguration, TaskRequirement
)
from ..core.config import get_settings
from ..models.database import get_async_session

logger = logging.getLogger(__name__)
settings = get_settings()


class AgentSelectionCriteria(Enum):
    """Criteria for agent selection algorithms"""
    PERFORMANCE_BASED = "performance_based"
    LOAD_BALANCED = "load_balanced"
    CAPABILITY_MATCH = "capability_match"
    AVAILABILITY_FIRST = "availability_first"
    COST_OPTIMIZED = "cost_optimized"


@dataclass
class AgentMatchScore:
    """Score for agent capability matching"""
    agent_id: str
    capability_score: float
    performance_score: float
    availability_score: float
    load_score: float
    overall_score: float
    reasoning: Dict[str, Any]


@dataclass
class HealthCheckResult:
    """Result of agent health check"""
    agent_id: str
    is_healthy: bool
    response_time: float
    last_heartbeat: datetime
    error_count: int
    warnings: List[str]
    metrics: Dict[str, Any]


class EnterpriseAgentRegistry:
    """
    Enterprise-grade agent registry with real-time discovery, capability matching,
    performance-based selection, and automatic failover capabilities.
    """
    
    def __init__(self):
        self.agents: Dict[str, Agent] = {}
        self.agent_capabilities: Dict[str, List[AgentCapability]] = {}
        self.agent_performance: Dict[str, AgentPerformanceMetrics] = {}
        self.agent_health: Dict[str, HealthCheckResult] = {}
        self.agent_locks: Dict[str, threading.Lock] = {}
        self.selection_algorithms: Dict[str, callable] = {}
        self.health_monitor_task: Optional[asyncio.Task] = None
        self.performance_tracker_task: Optional[asyncio.Task] = None
        self.executor = ThreadPoolExecutor(max_workers=10)
        
        # Initialize selection algorithms
        self._initialize_selection_algorithms()
        
        # Start background monitoring tasks
        asyncio.create_task(self._start_monitoring_tasks())
    
    async def register_agent(
        self, 
        agent: Agent, 
        capabilities: List[AgentCapability],
        configuration: AgentConfiguration
    ) -> bool:
        """
        Register a new agent with the registry.
        
        Args:
            agent: Agent instance to register
            capabilities: List of agent capabilities
            configuration: Agent configuration
            
        Returns:
            bool: True if registration successful
        """
        try:
            async with get_async_session() as session:
                # Validate agent doesn't already exist
                existing = await session.execute(
                    select(Agent).where(Agent.id == agent.id)
                )
                if existing.scalar_one_or_none():
                    logger.warning(f"Agent {agent.id} already registered")
                    return False
                
                # Set initial status and timestamps
                agent.status = AgentStatus.INITIALIZING
                agent.registered_at = datetime.utcnow()
                agent.last_heartbeat = datetime.utcnow()
                agent.configuration = configuration
                
                # Store agent in database
                session.add(agent)
                
                # Store capabilities
                for capability in capabilities:
                    capability.agent_id = agent.id
                    session.add(capability)
                
                await session.commit()
                
                # Update in-memory cache
                self.agents[agent.id] = agent
                self.agent_capabilities[agent.id] = capabilities
                self.agent_locks[agent.id] = threading.Lock()
                
                # Initialize performance metrics
                initial_metrics = AgentPerformanceMetrics(
                    agent_id=agent.id,
                    response_time=0.0,
                    throughput=0.0,
                    accuracy=1.0,
                    reliability=1.0,
                    success_rate=1.0,
                    error_rate=0.0,
                    resource_utilization=0.0,
                    business_impact_score=0.0,
                    last_updated=datetime.utcnow()
                )
                
                session.add(initial_metrics)
                await session.commit()
                
                self.agent_performance[agent.id] = initial_metrics
                
                # Perform initial health check
                await self._perform_health_check(agent.id)
                
                # Update agent status to available
                agent.status = AgentStatus.AVAILABLE
                await session.merge(agent)
                await session.commit()
                
                logger.info(f"Successfully registered agent {agent.id} with {len(capabilities)} capabilities")
                return True
                
        except Exception as e:
            logger.error(f"Failed to register agent {agent.id}: {str(e)}")
            return False
    
    async def deregister_agent(self, agent_id: str) -> bool:
        """
        Deregister an agent from the registry.
        
        Args:
            agent_id: ID of agent to deregister
            
        Returns:
            bool: True if deregistration successful
        """
        try:
            async with get_async_session() as session:
                # Update agent status to deregistering
                if agent_id in self.agents:
                    self.agents[agent_id].status = AgentStatus.DEREGISTERING
                
                # Remove from database
                await session.execute(
                    delete(Agent).where(Agent.id == agent_id)
                )
                await session.execute(
                    delete(AgentCapability).where(AgentCapability.agent_id == agent_id)
                )
                await session.execute(
                    delete(AgentPerformanceMetrics).where(AgentPerformanceMetrics.agent_id == agent_id)
                )
                await session.execute(
                    delete(AgentHealthCheck).where(AgentHealthCheck.agent_id == agent_id)
                )
                
                await session.commit()
                
                # Remove from in-memory cache
                self.agents.pop(agent_id, None)
                self.agent_capabilities.pop(agent_id, None)
                self.agent_performance.pop(agent_id, None)
                self.agent_health.pop(agent_id, None)
                self.agent_locks.pop(agent_id, None)
                
                logger.info(f"Successfully deregistered agent {agent_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to deregister agent {agent_id}: {str(e)}")
            return False
    
    async def get_available_agents(
        self, 
        requirements: Optional[List[TaskRequirement]] = None
    ) -> List[Agent]:
        """
        Get list of available agents, optionally filtered by requirements.
        
        Args:
            requirements: Optional task requirements for filtering
            
        Returns:
            List of available agents
        """
        try:
            available_agents = []
            
            for agent_id, agent in self.agents.items():
                # Check if agent is available and healthy
                if (agent.status == AgentStatus.AVAILABLE and 
                    self._is_agent_healthy(agent_id)):
                    
                    # If requirements specified, check capability match
                    if requirements:
                        if await self._matches_requirements(agent_id, requirements):
                            available_agents.append(agent)
                    else:
                        available_agents.append(agent)
            
            logger.debug(f"Found {len(available_agents)} available agents")
            return available_agents
            
        except Exception as e:
            logger.error(f"Failed to get available agents: {str(e)}")
            return []
    
    async def select_optimal_agents(
        self,
        requirements: List[TaskRequirement],
        selection_criteria: AgentSelectionCriteria = AgentSelectionCriteria.PERFORMANCE_BASED,
        max_agents: int = 5
    ) -> List[AgentMatchScore]:
        """
        Select optimal agents based on requirements and selection criteria.
        
        Args:
            requirements: Task requirements
            selection_criteria: Selection algorithm to use
            max_agents: Maximum number of agents to return
            
        Returns:
            List of agent match scores, sorted by overall score
        """
        try:
            # Get available agents
            available_agents = await self.get_available_agents(requirements)
            
            if not available_agents:
                logger.warning("No available agents found for requirements")
                return []
            
            # Score each agent
            agent_scores = []
            for agent in available_agents:
                score = await self._calculate_agent_score(agent.id, requirements, selection_criteria)
                if score:
                    agent_scores.append(score)
            
            # Sort by overall score (descending)
            agent_scores.sort(key=lambda x: x.overall_score, reverse=True)
            
            # Return top agents
            selected_agents = agent_scores[:max_agents]
            
            logger.info(f"Selected {len(selected_agents)} optimal agents using {selection_criteria.value}")
            return selected_agents
            
        except Exception as e:
            logger.error(f"Failed to select optimal agents: {str(e)}")
            return []
    
    async def update_agent_performance(
        self, 
        agent_id: str, 
        metrics: Dict[str, Any]
    ) -> bool:
        """
        Update agent performance metrics.
        
        Args:
            agent_id: ID of agent to update
            metrics: Performance metrics dictionary
            
        Returns:
            bool: True if update successful
        """
        try:
            if agent_id not in self.agents:
                logger.warning(f"Agent {agent_id} not found in registry")
                return False
            
            async with get_async_session() as session:
                # Get current metrics
                result = await session.execute(
                    select(AgentPerformanceMetrics).where(
                        AgentPerformanceMetrics.agent_id == agent_id
                    )
                )
                current_metrics = result.scalar_one_or_none()
                
                if not current_metrics:
                    # Create new metrics record
                    current_metrics = AgentPerformanceMetrics(agent_id=agent_id)
                    session.add(current_metrics)
                
                # Update metrics with exponential moving average
                alpha = 0.3  # Smoothing factor
                
                if 'response_time' in metrics:
                    current_metrics.response_time = (
                        alpha * metrics['response_time'] + 
                        (1 - alpha) * current_metrics.response_time
                    )
                
                if 'throughput' in metrics:
                    current_metrics.throughput = (
                        alpha * metrics['throughput'] + 
                        (1 - alpha) * current_metrics.throughput
                    )
                
                if 'accuracy' in metrics:
                    current_metrics.accuracy = (
                        alpha * metrics['accuracy'] + 
                        (1 - alpha) * current_metrics.accuracy
                    )
                
                if 'success_rate' in metrics:
                    current_metrics.success_rate = (
                        alpha * metrics['success_rate'] + 
                        (1 - alpha) * current_metrics.success_rate
                    )
                
                if 'error_rate' in metrics:
                    current_metrics.error_rate = (
                        alpha * metrics['error_rate'] + 
                        (1 - alpha) * current_metrics.error_rate
                    )
                
                if 'resource_utilization' in metrics:
                    current_metrics.resource_utilization = metrics['resource_utilization']
                
                if 'business_impact_score' in metrics:
                    current_metrics.business_impact_score = (
                        alpha * metrics['business_impact_score'] + 
                        (1 - alpha) * current_metrics.business_impact_score
                    )
                
                current_metrics.last_updated = datetime.utcnow()
                
                await session.commit()
                
                # Update in-memory cache
                self.agent_performance[agent_id] = current_metrics
                
                logger.debug(f"Updated performance metrics for agent {agent_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to update agent performance {agent_id}: {str(e)}")
            return False
    
    async def handle_agent_failure(self, agent_id: str, failure_context: Dict[str, Any]) -> bool:
        """
        Handle agent failure with automatic failover.
        
        Args:
            agent_id: ID of failed agent
            failure_context: Context information about the failure
            
        Returns:
            bool: True if failover successful
        """
        try:
            if agent_id not in self.agents:
                logger.warning(f"Failed agent {agent_id} not found in registry")
                return False
            
            # Update agent status
            self.agents[agent_id].status = AgentStatus.FAILED
            self.agents[agent_id].last_failure = datetime.utcnow()
            
            # Log failure details
            logger.error(f"Agent {agent_id} failed: {failure_context}")
            
            # Update failure metrics
            await self._update_failure_metrics(agent_id, failure_context)
            
            # Trigger automatic failover if configured
            if self.agents[agent_id].configuration.auto_failover_enabled:
                await self._trigger_failover(agent_id, failure_context)
            
            # Notify monitoring system
            await self._notify_failure(agent_id, failure_context)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to handle agent failure {agent_id}: {str(e)}")
            return False
    
    async def get_agent_health_status(self, agent_id: str) -> Optional[HealthCheckResult]:
        """
        Get current health status of an agent.
        
        Args:
            agent_id: ID of agent to check
            
        Returns:
            Health check result or None if agent not found
        """
        return self.agent_health.get(agent_id)
    
    async def get_registry_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive registry statistics.
        
        Returns:
            Dictionary containing registry statistics
        """
        try:
            stats = {
                'total_agents': len(self.agents),
                'available_agents': len([a for a in self.agents.values() if a.status == AgentStatus.AVAILABLE]),
                'busy_agents': len([a for a in self.agents.values() if a.status == AgentStatus.BUSY]),
                'failed_agents': len([a for a in self.agents.values() if a.status == AgentStatus.FAILED]),
                'maintenance_agents': len([a for a in self.agents.values() if a.status == AgentStatus.MAINTENANCE]),
                'healthy_agents': len([aid for aid in self.agents.keys() if self._is_agent_healthy(aid)]),
                'average_response_time': 0.0,
                'average_success_rate': 0.0,
                'total_capabilities': sum(len(caps) for caps in self.agent_capabilities.values()),
                'last_updated': datetime.utcnow().isoformat()
            }
            
            # Calculate averages
            if self.agent_performance:
                response_times = [m.response_time for m in self.agent_performance.values()]
                success_rates = [m.success_rate for m in self.agent_performance.values()]
                
                stats['average_response_time'] = sum(response_times) / len(response_times)
                stats['average_success_rate'] = sum(success_rates) / len(success_rates)
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get registry statistics: {str(e)}")
            return {}
    
    # Private methods
    
    def _initialize_selection_algorithms(self):
        """Initialize agent selection algorithms"""
        self.selection_algorithms = {
            AgentSelectionCriteria.PERFORMANCE_BASED: self._performance_based_selection,
            AgentSelectionCriteria.LOAD_BALANCED: self._load_balanced_selection,
            AgentSelectionCriteria.CAPABILITY_MATCH: self._capability_match_selection,
            AgentSelectionCriteria.AVAILABILITY_FIRST: self._availability_first_selection,
            AgentSelectionCriteria.COST_OPTIMIZED: self._cost_optimized_selection
        }
    
    async def _start_monitoring_tasks(self):
        """Start background monitoring tasks"""
        try:
            # Start health monitoring
            self.health_monitor_task = asyncio.create_task(self._health_monitor_loop())
            
            # Start performance tracking
            self.performance_tracker_task = asyncio.create_task(self._performance_tracker_loop())
            
            logger.info("Started agent registry monitoring tasks")
            
        except Exception as e:
            logger.error(f"Failed to start monitoring tasks: {str(e)}")
    
    async def _health_monitor_loop(self):
        """Background task for continuous health monitoring"""
        while True:
            try:
                # Check health of all registered agents
                for agent_id in list(self.agents.keys()):
                    await self._perform_health_check(agent_id)
                
                # Sleep for health check interval
                await asyncio.sleep(settings.AGENT_HEALTH_CHECK_INTERVAL)
                
            except Exception as e:
                logger.error(f"Error in health monitor loop: {str(e)}")
                await asyncio.sleep(5)  # Brief pause before retrying
    
    async def _performance_tracker_loop(self):
        """Background task for performance tracking"""
        while True:
            try:
                # Update performance metrics for all agents
                for agent_id in list(self.agents.keys()):
                    await self._track_agent_performance(agent_id)
                
                # Sleep for performance tracking interval
                await asyncio.sleep(settings.AGENT_PERFORMANCE_TRACK_INTERVAL)
                
            except Exception as e:
                logger.error(f"Error in performance tracker loop: {str(e)}")
                await asyncio.sleep(5)  # Brief pause before retrying
    
    async def _perform_health_check(self, agent_id: str) -> HealthCheckResult:
        """
        Perform health check on specific agent.
        
        Args:
            agent_id: ID of agent to check
            
        Returns:
            Health check result
        """
        try:
            agent = self.agents.get(agent_id)
            if not agent:
                return HealthCheckResult(
                    agent_id=agent_id,
                    is_healthy=False,
                    response_time=0.0,
                    last_heartbeat=datetime.utcnow(),
                    error_count=1,
                    warnings=["Agent not found in registry"],
                    metrics={}
                )
            
            start_time = time.time()
            
            # Simulate health check (in real implementation, this would ping the agent)
            # For now, check if agent has recent heartbeat
            time_since_heartbeat = (datetime.utcnow() - agent.last_heartbeat).total_seconds()
            is_healthy = time_since_heartbeat < settings.AGENT_HEARTBEAT_TIMEOUT
            
            response_time = time.time() - start_time
            
            # Get current performance metrics
            performance = self.agent_performance.get(agent_id)
            error_count = int(performance.error_rate * 100) if performance else 0
            
            warnings = []
            if time_since_heartbeat > settings.AGENT_HEARTBEAT_WARNING_THRESHOLD:
                warnings.append(f"No heartbeat for {time_since_heartbeat:.1f} seconds")
            
            if performance and performance.success_rate < 0.8:
                warnings.append(f"Low success rate: {performance.success_rate:.2f}")
            
            health_result = HealthCheckResult(
                agent_id=agent_id,
                is_healthy=is_healthy,
                response_time=response_time,
                last_heartbeat=agent.last_heartbeat,
                error_count=error_count,
                warnings=warnings,
                metrics={
                    'cpu_usage': 0.5,  # Placeholder
                    'memory_usage': 0.3,  # Placeholder
                    'active_tasks': agent.active_tasks
                }
            )
            
            # Store health result
            self.agent_health[agent_id] = health_result
            
            # Update agent status based on health
            if not is_healthy and agent.status == AgentStatus.AVAILABLE:
                agent.status = AgentStatus.UNHEALTHY
                logger.warning(f"Agent {agent_id} marked as unhealthy")
            elif is_healthy and agent.status == AgentStatus.UNHEALTHY:
                agent.status = AgentStatus.AVAILABLE
                logger.info(f"Agent {agent_id} recovered and marked as available")
            
            return health_result
            
        except Exception as e:
            logger.error(f"Health check failed for agent {agent_id}: {str(e)}")
            return HealthCheckResult(
                agent_id=agent_id,
                is_healthy=False,
                response_time=0.0,
                last_heartbeat=datetime.utcnow(),
                error_count=1,
                warnings=[f"Health check error: {str(e)}"],
                metrics={}
            )
    
    def _is_agent_healthy(self, agent_id: str) -> bool:
        """Check if agent is healthy based on latest health check"""
        health_result = self.agent_health.get(agent_id)
        return health_result.is_healthy if health_result else False
    
    async def _matches_requirements(self, agent_id: str, requirements: List[TaskRequirement]) -> bool:
        """Check if agent matches task requirements"""
        try:
            agent_capabilities = self.agent_capabilities.get(agent_id, [])
            
            for requirement in requirements:
                # Check if agent has required capability
                has_capability = any(
                    cap.name == requirement.capability_name and
                    cap.version >= requirement.min_version
                    for cap in agent_capabilities
                )
                
                if not has_capability:
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking requirements for agent {agent_id}: {str(e)}")
            return False
    
    async def _calculate_agent_score(
        self, 
        agent_id: str, 
        requirements: List[TaskRequirement],
        selection_criteria: AgentSelectionCriteria
    ) -> Optional[AgentMatchScore]:
        """Calculate comprehensive score for agent selection"""
        try:
            # Get selection algorithm
            algorithm = self.selection_algorithms.get(selection_criteria)
            if not algorithm:
                logger.error(f"Unknown selection criteria: {selection_criteria}")
                return None
            
            return await algorithm(agent_id, requirements)
            
        except Exception as e:
            logger.error(f"Error calculating agent score for {agent_id}: {str(e)}")
            return None
    
    async def _performance_based_selection(
        self, 
        agent_id: str, 
        requirements: List[TaskRequirement]
    ) -> AgentMatchScore:
        """Performance-based agent selection algorithm"""
        agent = self.agents[agent_id]
        capabilities = self.agent_capabilities.get(agent_id, [])
        performance = self.agent_performance.get(agent_id)
        health = self.agent_health.get(agent_id)
        
        # Calculate capability score
        capability_score = self._calculate_capability_score(capabilities, requirements)
        
        # Calculate performance score
        performance_score = 0.0
        if performance:
            performance_score = (
                performance.success_rate * 0.3 +
                (1.0 - performance.error_rate) * 0.2 +
                min(performance.accuracy, 1.0) * 0.2 +
                min(performance.reliability, 1.0) * 0.2 +
                min(performance.business_impact_score / 100.0, 1.0) * 0.1
            )
        
        # Calculate availability score
        availability_score = 1.0 if agent.status == AgentStatus.AVAILABLE else 0.0
        if health and not health.is_healthy:
            availability_score *= 0.5
        
        # Calculate load score (inverse of current load)
        load_score = max(0.0, 1.0 - (agent.active_tasks / agent.max_concurrent_tasks))
        
        # Calculate overall score
        overall_score = (
            capability_score * 0.4 +
            performance_score * 0.3 +
            availability_score * 0.2 +
            load_score * 0.1
        )
        
        return AgentMatchScore(
            agent_id=agent_id,
            capability_score=capability_score,
            performance_score=performance_score,
            availability_score=availability_score,
            load_score=load_score,
            overall_score=overall_score,
            reasoning={
                'selection_criteria': 'performance_based',
                'capability_match': capability_score > 0.7,
                'high_performance': performance_score > 0.8,
                'available': availability_score > 0.0,
                'low_load': load_score > 0.5
            }
        )
    
    async def _load_balanced_selection(
        self, 
        agent_id: str, 
        requirements: List[TaskRequirement]
    ) -> AgentMatchScore:
        """Load-balanced agent selection algorithm"""
        # Similar to performance-based but prioritizes load balancing
        score = await self._performance_based_selection(agent_id, requirements)
        
        # Adjust overall score to prioritize load balancing
        score.overall_score = (
            score.capability_score * 0.3 +
            score.performance_score * 0.2 +
            score.availability_score * 0.2 +
            score.load_score * 0.3  # Higher weight for load
        )
        
        score.reasoning['selection_criteria'] = 'load_balanced'
        return score
    
    async def _capability_match_selection(
        self, 
        agent_id: str, 
        requirements: List[TaskRequirement]
    ) -> AgentMatchScore:
        """Capability-focused agent selection algorithm"""
        score = await self._performance_based_selection(agent_id, requirements)
        
        # Adjust overall score to prioritize capability matching
        score.overall_score = (
            score.capability_score * 0.6 +  # Higher weight for capabilities
            score.performance_score * 0.2 +
            score.availability_score * 0.1 +
            score.load_score * 0.1
        )
        
        score.reasoning['selection_criteria'] = 'capability_match'
        return score
    
    async def _availability_first_selection(
        self, 
        agent_id: str, 
        requirements: List[TaskRequirement]
    ) -> AgentMatchScore:
        """Availability-first agent selection algorithm"""
        score = await self._performance_based_selection(agent_id, requirements)
        
        # Adjust overall score to prioritize availability
        score.overall_score = (
            score.capability_score * 0.2 +
            score.performance_score * 0.2 +
            score.availability_score * 0.4 +  # Higher weight for availability
            score.load_score * 0.2
        )
        
        score.reasoning['selection_criteria'] = 'availability_first'
        return score
    
    async def _cost_optimized_selection(
        self, 
        agent_id: str, 
        requirements: List[TaskRequirement]
    ) -> AgentMatchScore:
        """Cost-optimized agent selection algorithm"""
        score = await self._performance_based_selection(agent_id, requirements)
        
        # Factor in cost considerations (placeholder implementation)
        agent = self.agents[agent_id]
        cost_score = 1.0 - (agent.cost_per_hour / 100.0)  # Normalize cost
        
        # Adjust overall score to consider cost
        score.overall_score = (
            score.capability_score * 0.3 +
            score.performance_score * 0.2 +
            score.availability_score * 0.2 +
            score.load_score * 0.1 +
            cost_score * 0.2  # Cost consideration
        )
        
        score.reasoning['selection_criteria'] = 'cost_optimized'
        score.reasoning['cost_efficient'] = cost_score > 0.7
        return score
    
    def _calculate_capability_score(
        self, 
        capabilities: List[AgentCapability], 
        requirements: List[TaskRequirement]
    ) -> float:
        """Calculate how well agent capabilities match requirements"""
        if not requirements:
            return 1.0
        
        matched_requirements = 0
        total_score = 0.0
        
        for requirement in requirements:
            best_match_score = 0.0
            
            for capability in capabilities:
                if capability.name == requirement.capability_name:
                    # Base score for having the capability
                    score = 0.7
                    
                    # Bonus for version compatibility
                    if capability.version >= requirement.min_version:
                        score += 0.2
                    
                    # Bonus for performance level
                    if hasattr(capability, 'performance_level'):
                        score += min(capability.performance_level / 10.0, 0.1)
                    
                    best_match_score = max(best_match_score, score)
            
            if best_match_score > 0:
                matched_requirements += 1
                total_score += best_match_score
        
        if not requirements:
            return 1.0
        
        # Calculate final score
        match_ratio = matched_requirements / len(requirements)
        average_score = total_score / len(requirements) if requirements else 0.0
        
        return match_ratio * average_score
    
    async def _track_agent_performance(self, agent_id: str):
        """Track and update agent performance metrics"""
        try:
            # This would collect real performance data in production
            # For now, simulate some performance tracking
            pass
            
        except Exception as e:
            logger.error(f"Error tracking performance for agent {agent_id}: {str(e)}")
    
    async def _update_failure_metrics(self, agent_id: str, failure_context: Dict[str, Any]):
        """Update failure-related metrics"""
        try:
            performance = self.agent_performance.get(agent_id)
            if performance:
                # Increase error rate
                performance.error_rate = min(1.0, performance.error_rate + 0.1)
                performance.success_rate = max(0.0, performance.success_rate - 0.1)
                performance.reliability = max(0.0, performance.reliability - 0.05)
                performance.last_updated = datetime.utcnow()
                
                # Update in database
                async with get_async_session() as session:
                    await session.merge(performance)
                    await session.commit()
            
        except Exception as e:
            logger.error(f"Error updating failure metrics for agent {agent_id}: {str(e)}")
    
    async def _trigger_failover(self, failed_agent_id: str, failure_context: Dict[str, Any]):
        """Trigger automatic failover for failed agent"""
        try:
            logger.info(f"Triggering failover for agent {failed_agent_id}")
            
            # Find suitable replacement agents
            failed_agent = self.agents[failed_agent_id]
            failed_capabilities = self.agent_capabilities.get(failed_agent_id, [])
            
            # Create requirements from failed agent's capabilities
            requirements = [
                TaskRequirement(
                    capability_name=cap.name,
                    min_version=cap.version,
                    required=True
                )
                for cap in failed_capabilities
            ]
            
            # Select replacement agents
            replacement_agents = await self.select_optimal_agents(
                requirements=requirements,
                selection_criteria=AgentSelectionCriteria.AVAILABILITY_FIRST,
                max_agents=2
            )
            
            if replacement_agents:
                logger.info(f"Found {len(replacement_agents)} replacement agents for {failed_agent_id}")
                # In production, this would redistribute tasks to replacement agents
            else:
                logger.warning(f"No suitable replacement agents found for {failed_agent_id}")
            
        except Exception as e:
            logger.error(f"Error triggering failover for agent {failed_agent_id}: {str(e)}")
    
    async def _notify_failure(self, agent_id: str, failure_context: Dict[str, Any]):
        """Notify monitoring system of agent failure"""
        try:
            # In production, this would send alerts to monitoring systems
            logger.critical(f"AGENT FAILURE ALERT: Agent {agent_id} has failed - {failure_context}")
            
        except Exception as e:
            logger.error(f"Error notifying failure for agent {agent_id}: {str(e)}")


# Global registry instance
_registry_instance: Optional[EnterpriseAgentRegistry] = None


def get_agent_registry() -> EnterpriseAgentRegistry:
    """Get the global agent registry instance"""
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = EnterpriseAgentRegistry()
    return _registry_instance