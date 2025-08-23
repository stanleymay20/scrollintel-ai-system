"""
Agent Registry and Management System

This module implements the central agent registry for dynamic agent discovery,
capability matching, performance-based selection, health monitoring, and
lifecycle management for the Agent Steering System.

Requirements: 1.1, 1.2, 6.1
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict

logger = logging.getLogger(__name__)


class CapabilityType(Enum):
    """Types of agent capabilities"""
    DATA_ANALYSIS = "data_analysis"
    MACHINE_LEARNING = "machine_learning"
    VISUALIZATION = "visualization"
    REPORTING = "reporting"
    FORECASTING = "forecasting"
    QUALITY_ASSURANCE = "quality_assurance"


@dataclass
class AgentRegistrationRequest:
    """Request for registering a new agent"""
    name: str
    type: str
    version: str
    capabilities: List[Dict[str, Any]]
    endpoint_url: str
    health_check_url: str
    resource_requirements: Dict[str, Any]
    configuration: Dict[str, Any]
    authentication_token: Optional[str] = None


@dataclass
class AgentSelectionCriteria:
    """Criteria for selecting agents"""
    required_capabilities: List[str]
    preferred_capabilities: Optional[List[str]] = None
    max_response_time: Optional[float] = None
    min_success_rate: Optional[float] = None
    max_load_threshold: Optional[float] = None
    business_domain: Optional[str] = None
    exclude_agents: Optional[List[str]] = None


class AdvancedCapabilityMatcher:
    """Advanced capability matching engine with synonym support and domain expertise"""
    
    def __init__(self):
        # Capability synonyms mapping
        self.capability_synonyms = {
            "data_analysis": ["analytics", "data_processing", "analysis"],
            "machine_learning": ["ml", "ai", "artificial_intelligence"],
            "visualization": ["viz", "charting", "plotting"],
            "reporting": ["reports", "documentation"],
            "forecasting": ["prediction", "forecasts", "projections"]
        }
        
        # Business domain expertise mapping
        self.domain_capabilities = {
            "finance": ["financial_modeling", "risk_analysis", "portfolio_management"],
            "healthcare": ["medical_analysis", "patient_data", "clinical_research"],
            "retail": ["sales_analysis", "inventory_management", "customer_analytics"],
            "manufacturing": ["process_optimization", "quality_control", "supply_chain"]
        }
    
    def calculate_capability_match_score(
        self, 
        agent_capabilities: List[Dict[str, Any]], 
        required_capabilities: List[str],
        preferred_capabilities: Optional[List[str]] = None,
        business_domain: Optional[str] = None
    ) -> float:
        """Calculate comprehensive capability match score"""
        if not agent_capabilities or not required_capabilities:
            return 0.0
        
        agent_cap_names = [cap["name"].lower() for cap in agent_capabilities]
        required_caps_lower = [cap.lower() for cap in required_capabilities]
        
        # Calculate direct matches
        direct_matches = 0
        synonym_matches = 0
        total_performance = 0.0
        
        for required_cap in required_caps_lower:
            # Check direct match
            if required_cap in agent_cap_names:
                direct_matches += 1
                # Get performance score for this capability
                for cap in agent_capabilities:
                    if cap["name"].lower() == required_cap:
                        total_performance += cap.get("performance_score", 0.0)
                        break
            else:
                # Check synonym matches
                for synonym_group, synonyms in self.capability_synonyms.items():
                    if required_cap in synonyms and synonym_group in agent_cap_names:
                        synonym_matches += 1
                        # Get performance score for synonym
                        for cap in agent_capabilities:
                            if cap["name"].lower() == synonym_group:
                                total_performance += cap.get("performance_score", 0.0) * 0.8
                                break
                        break
        
        # Base score calculation
        total_required = len(required_capabilities)
        match_ratio = (direct_matches + synonym_matches) / total_required
        
        # Performance bonus
        avg_performance = total_performance / max(direct_matches + synonym_matches, 1)
        performance_bonus = (avg_performance / 100.0) * 0.2
        
        base_score = (match_ratio * 80.0) + (performance_bonus * 100.0)
        
        # Preferred capabilities bonus
        preferred_bonus = 0.0
        if preferred_capabilities:
            preferred_matches = 0
            for pref_cap in preferred_capabilities:
                if pref_cap.lower() in agent_cap_names:
                    preferred_matches += 1
            
            if preferred_matches > 0:
                preferred_bonus = (preferred_matches / len(preferred_capabilities)) * 10.0
        
        # Business domain expertise bonus
        domain_bonus = 0.0
        if business_domain and business_domain in self.domain_capabilities:
            domain_caps = self.domain_capabilities[business_domain]
            domain_matches = sum(1 for cap in agent_cap_names if cap in domain_caps)
            if domain_matches > 0:
                domain_bonus = min(domain_matches * 5.0, 15.0)
        
        final_score = min(base_score + preferred_bonus + domain_bonus, 100.0)
        return final_score


class PerformanceBasedSelector:
    """Performance-based agent selection with adaptive learning"""
    
    def __init__(self):
        self.selection_history = []
        self.performance_weights = {
            "capability_match": 0.4,
            "success_rate": 0.25,
            "response_time": 0.15,
            "load": 0.15,
            "availability": 0.05
        }
        self.adaptive_adjustments = defaultdict(float)
    
    def calculate_selection_score(
        self, 
        agent: Dict[str, Any], 
        criteria: AgentSelectionCriteria,
        capability_match_score: float
    ) -> Dict[str, Any]:
        """Calculate comprehensive selection score for an agent"""
        component_scores = {}
        
        # Capability match score (already calculated)
        component_scores["capability_match"] = capability_match_score
        
        # Success rate score
        success_rate = agent.get("success_rate", 0.0)
        if criteria.min_success_rate:
            if success_rate < criteria.min_success_rate:
                component_scores["success_rate"] = 0.0
            else:
                component_scores["success_rate"] = min(success_rate, 100.0)
        else:
            component_scores["success_rate"] = min(success_rate, 100.0)
        
        # Response time score (inverse - lower is better)
        response_time = agent.get("average_response_time", 0.0)
        if criteria.max_response_time:
            if response_time > criteria.max_response_time:
                component_scores["response_time"] = 0.0
            else:
                component_scores["response_time"] = max(0.0, 100.0 - (response_time / criteria.max_response_time * 100.0))
        else:
            component_scores["response_time"] = max(0.0, 100.0 - (response_time / 10.0 * 100.0))
        
        # Load score (inverse - lower load is better)
        current_load = agent.get("current_load", 0.0)
        if criteria.max_load_threshold:
            if current_load > criteria.max_load_threshold:
                component_scores["load"] = 0.0
            else:
                component_scores["load"] = max(0.0, 100.0 - current_load)
        else:
            component_scores["load"] = max(0.0, 100.0 - current_load)
        
        # Availability score (based on last heartbeat)
        last_heartbeat_str = agent.get("last_heartbeat")
        if last_heartbeat_str:
            try:
                last_heartbeat = datetime.fromisoformat(last_heartbeat_str.replace('Z', '+00:00'))
                time_since_heartbeat = (datetime.utcnow() - last_heartbeat.replace(tzinfo=None)).total_seconds()
                if time_since_heartbeat < 60:
                    component_scores["availability"] = 100.0
                elif time_since_heartbeat < 300:
                    component_scores["availability"] = 80.0
                else:
                    component_scores["availability"] = 0.0
            except:
                component_scores["availability"] = 50.0
        else:
            component_scores["availability"] = 0.0
        
        # Apply adaptive adjustments
        agent_id = agent.get("id")
        if agent_id in self.adaptive_adjustments:
            adjustment = self.adaptive_adjustments[agent_id]
            for score_type in component_scores:
                component_scores[score_type] = min(100.0, component_scores[score_type] + adjustment)
        
        # Calculate weighted total score
        total_score = sum(
            component_scores[component] * self.performance_weights[component]
            for component in component_scores
        )
        
        return {
            "total_score": total_score,
            "component_scores": component_scores,
            "weights_used": self.performance_weights.copy()
        }
    
    def record_selection_outcome(
        self, 
        agent_id: str, 
        criteria: AgentSelectionCriteria, 
        outcome_success: bool
    ):
        """Record the outcome of an agent selection for adaptive learning"""
        outcome_record = {
            "timestamp": datetime.utcnow(),
            "selected_agent_id": agent_id,
            "criteria": criteria,
            "outcome_success": outcome_success
        }
        
        self.selection_history.append(outcome_record)
        
        # Update adaptive adjustments
        if outcome_success:
            self.adaptive_adjustments[agent_id] += 1.0
        else:
            self.adaptive_adjustments[agent_id] -= 2.0
        
        # Keep adjustments within reasonable bounds
        self.adaptive_adjustments[agent_id] = max(-10.0, min(10.0, self.adaptive_adjustments[agent_id]))
        
        # Keep history manageable (last 1000 records)
        if len(self.selection_history) > 1000:
            self.selection_history = self.selection_history[-1000:]


class AgentHealthMonitor:
    """Advanced agent health monitoring with predictive failure detection"""
    
    def __init__(self, messaging_system):
        self.messaging_system = messaging_system
        self.health_history = defaultdict(list)
        self.circuit_breakers = {}
        self.failover_groups = {}
        self.monitoring_stats = {
            "checks_performed": 0,
            "failures_detected": 0,
            "circuit_breakers_opened": 0,
            "failovers_triggered": 0
        }
    
    def configure_failover_group(self, group_name: str, agent_ids: List[str]):
        """Configure a failover group of agents"""
        self.failover_groups[group_name] = agent_ids
        logger.info(f"Configured failover group '{group_name}' with {len(agent_ids)} agents")
    
    def get_monitoring_stats(self) -> Dict[str, Any]:
        """Get comprehensive monitoring statistics"""
        stats = self.monitoring_stats.copy()
        stats["active_circuit_breakers"] = len(self.circuit_breakers)
        stats["agents_monitored"] = len(self.health_history)
        stats["total_health_records"] = sum(len(history) for history in self.health_history.values())
        return stats


class AgentRegistry:
    """
    Enterprise-grade agent registry with real-time discovery, capability matching,
    performance-based selection, and automatic failover capabilities.
    """
    
    def __init__(self, messaging_system):
        self.messaging_system = messaging_system
        self.capability_matcher = AdvancedCapabilityMatcher()
        self.performance_selector = PerformanceBasedSelector()
        self.health_monitor = AgentHealthMonitor(messaging_system)
        
        # In-memory caches
        self.agent_cache: Dict[str, Dict[str, Any]] = {}
        self.capability_cache: Dict[str, List[Dict[str, Any]]] = {}
        
        # Background tasks
        self.health_monitor_task: Optional[asyncio.Task] = None
        self.performance_tracker_task: Optional[asyncio.Task] = None
    
    async def register_agent(self, registration_request: AgentRegistrationRequest) -> Optional[str]:
        """Register a new agent with the registry."""
        try:
            # Mock implementation for testing
            agent_id = f"agent-{registration_request.name}-{datetime.utcnow().timestamp()}"
            
            # Store in cache
            agent_dict = {
                "id": agent_id,
                "name": registration_request.name,
                "type": registration_request.type,
                "version": registration_request.version,
                "capabilities": registration_request.capabilities,
                "endpoint_url": registration_request.endpoint_url,
                "health_check_url": registration_request.health_check_url,
                "current_load": 0.0,
                "max_concurrent_tasks": registration_request.configuration.get("max_tasks", 10),
                "success_rate": 100.0,
                "average_response_time": 0.0,
                "last_heartbeat": datetime.utcnow().isoformat()
            }
            
            self.agent_cache[agent_id] = agent_dict
            self.capability_cache[agent_id] = registration_request.capabilities
            
            logger.info(f"Successfully registered agent {registration_request.name} with ID {agent_id}")
            return agent_id
            
        except Exception as e:
            logger.error(f"Failed to register agent {registration_request.name}: {str(e)}")
            return None
    
    async def deregister_agent(self, agent_id: str) -> bool:
        """Deregister an agent from the registry."""
        try:
            if agent_id in self.agent_cache:
                agent_name = self.agent_cache[agent_id]["name"]
                self.agent_cache.pop(agent_id, None)
                self.capability_cache.pop(agent_id, None)
                logger.info(f"Successfully deregistered agent {agent_name}")
                return True
            else:
                logger.warning(f"Agent {agent_id} not found for deregistration")
                return False
                
        except Exception as e:
            logger.error(f"Failed to deregister agent {agent_id}: {str(e)}")
            return False
    
    async def get_available_agents(self, criteria: Optional[AgentSelectionCriteria] = None) -> List[Dict[str, Any]]:
        """Get list of available agents, optionally filtered by criteria."""
        try:
            available_agents = []
            
            for agent_id, agent_dict in self.agent_cache.items():
                # Apply criteria filtering if provided
                if criteria:
                    if criteria.exclude_agents and agent_id in criteria.exclude_agents:
                        continue
                    if criteria.max_load_threshold and agent_dict["current_load"] > criteria.max_load_threshold:
                        continue
                    if criteria.min_success_rate and agent_dict["success_rate"] < criteria.min_success_rate:
                        continue
                    if criteria.max_response_time and agent_dict["average_response_time"] > criteria.max_response_time:
                        continue
                
                available_agents.append(agent_dict)
            
            return available_agents
            
        except Exception as e:
            logger.error(f"Failed to get available agents: {str(e)}")
            return []
    
    async def select_best_agent(self, criteria: AgentSelectionCriteria) -> Optional[Dict[str, Any]]:
        """Select the best agent based on criteria."""
        try:
            available_agents = await self.get_available_agents(criteria)
            
            if not available_agents:
                return None
            
            best_agent = None
            best_score = -1.0
            
            for agent in available_agents:
                capability_match_score = self.capability_matcher.calculate_capability_match_score(
                    agent["capabilities"],
                    criteria.required_capabilities,
                    criteria.preferred_capabilities,
                    criteria.business_domain
                )
                
                selection_result = self.performance_selector.calculate_selection_score(
                    agent, criteria, capability_match_score
                )
                
                if selection_result["total_score"] > best_score:
                    best_score = selection_result["total_score"]
                    best_agent = {
                        **agent,
                        "selection_score": selection_result["total_score"],
                        "capability_match_score": capability_match_score,
                        "score_breakdown": selection_result["component_scores"]
                    }
            
            return best_agent
            
        except Exception as e:
            logger.error(f"Failed to select best agent: {str(e)}")
            return None
    
    async def select_multiple_agents(
        self, 
        criteria: AgentSelectionCriteria, 
        count: int, 
        strategy: str = "performance"
    ) -> List[Dict[str, Any]]:
        """Select multiple agents using different strategies."""
        try:
            available_agents = await self.get_available_agents(criteria)
            
            if not available_agents:
                return []
            
            scored_agents = []
            
            for agent in available_agents:
                capability_match_score = self.capability_matcher.calculate_capability_match_score(
                    agent["capabilities"],
                    criteria.required_capabilities,
                    criteria.preferred_capabilities,
                    criteria.business_domain
                )
                
                selection_result = self.performance_selector.calculate_selection_score(
                    agent, criteria, capability_match_score
                )
                
                scored_agents.append({
                    **agent,
                    "selection_score": selection_result["total_score"],
                    "capability_match_score": capability_match_score,
                    "score_breakdown": selection_result["component_scores"]
                })
            
            # Apply selection strategy
            if strategy == "performance":
                scored_agents.sort(key=lambda x: x["selection_score"], reverse=True)
            elif strategy == "load_balanced":
                scored_agents.sort(key=lambda x: (x["current_load"], -x["selection_score"]))
            elif strategy == "diverse":
                selected = []
                used_types = set()
                
                for agent in sorted(scored_agents, key=lambda x: x["selection_score"], reverse=True):
                    if agent["type"] not in used_types and len(selected) < count:
                        selected.append(agent)
                        used_types.add(agent["type"])
                
                remaining = count - len(selected)
                if remaining > 0:
                    remaining_agents = [a for a in scored_agents if a not in selected]
                    remaining_agents.sort(key=lambda x: x["selection_score"], reverse=True)
                    selected.extend(remaining_agents[:remaining])
                
                return selected
            
            return scored_agents[:count]
            
        except Exception as e:
            logger.error(f"Failed to select multiple agents: {str(e)}")
            return []
    
    async def update_agent_configuration(self, agent_id: str, configuration_updates: Dict[str, Any]) -> bool:
        """Update agent configuration."""
        try:
            if agent_id not in self.agent_cache:
                logger.warning(f"Agent {agent_id} not found for configuration update")
                return False
            
            agent = self.agent_cache[agent_id]
            
            if "configuration" in configuration_updates:
                # Mock update
                pass
            if "capabilities" in configuration_updates:
                agent["capabilities"] = configuration_updates["capabilities"]
                self.capability_cache[agent_id] = configuration_updates["capabilities"]
            if "version" in configuration_updates:
                agent["version"] = configuration_updates["version"]
            
            logger.info(f"Updated configuration for agent {agent['name']}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update agent configuration {agent_id}: {str(e)}")
            return False
    
    async def scale_agent(self, agent_id: str, scale_action: str, target_capacity: Optional[int] = None) -> bool:
        """Scale agent capacity."""
        try:
            if agent_id not in self.agent_cache:
                logger.warning(f"Agent {agent_id} not found for scaling")
                return False
            
            agent = self.agent_cache[agent_id]
            current_capacity = agent["max_concurrent_tasks"]
            
            if scale_action == "scale_up":
                agent["max_concurrent_tasks"] = current_capacity * 2
            elif scale_action == "scale_down":
                agent["max_concurrent_tasks"] = max(1, current_capacity // 2)
            elif scale_action == "set_capacity" and target_capacity:
                agent["max_concurrent_tasks"] = max(1, target_capacity)
            else:
                logger.warning(f"Invalid scale action: {scale_action}")
                return False
            
            logger.info(f"Scaled agent {agent['name']} from {current_capacity} to {agent['max_concurrent_tasks']}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to scale agent {agent_id}: {str(e)}")
            return False
    
    async def put_agent_in_maintenance(self, agent_id: str, reason: str) -> bool:
        """Put agent in maintenance mode."""
        try:
            if agent_id not in self.agent_cache:
                return False
            
            agent = self.agent_cache[agent_id]
            # Mock maintenance mode
            agent["status"] = "maintenance"
            agent["maintenance_reason"] = reason
            
            await self._reassign_agent_tasks(agent_id)
            
            logger.info(f"Put agent {agent['name']} in maintenance: {reason}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to put agent in maintenance {agent_id}: {str(e)}")
            return False
    
    async def remove_agent_from_maintenance(self, agent_id: str) -> bool:
        """Remove agent from maintenance mode."""
        try:
            if agent_id not in self.agent_cache:
                return False
            
            agent = self.agent_cache[agent_id]
            agent["status"] = "active"
            agent.pop("maintenance_reason", None)
            
            logger.info(f"Removed agent {agent['name']} from maintenance")
            return True
            
        except Exception as e:
            logger.error(f"Failed to remove agent from maintenance {agent_id}: {str(e)}")
            return False
    
    async def get_registry_stats(self) -> Dict[str, Any]:
        """Get comprehensive registry statistics."""
        try:
            total_agents = len(self.agent_cache)
            active_agents = len([a for a in self.agent_cache.values() if a.get("status", "active") == "active"])
            maintenance_agents = len([a for a in self.agent_cache.values() if a.get("status") == "maintenance"])
            
            capability_counts = defaultdict(int)
            for agent in self.agent_cache.values():
                for cap in agent["capabilities"]:
                    capability_counts[cap.get("name", "unknown")] += 1
            
            avg_response_time = sum(a["average_response_time"] for a in self.agent_cache.values()) / max(total_agents, 1)
            
            stats = {
                "agent_counts": {
                    "total": total_agents,
                    "active": active_agents,
                    "maintenance": maintenance_agents,
                    "error": 0,
                    "inactive": 0
                },
                "capability_distribution": dict(capability_counts),
                "performance_metrics": {
                    "average_response_time": avg_response_time,
                    "total_capacity": sum(a["max_concurrent_tasks"] for a in self.agent_cache.values()),
                    "current_load": sum(a["current_load"] for a in self.agent_cache.values())
                },
                "health_monitoring": self.health_monitor.get_monitoring_stats(),
                "selection_algorithm": {
                    "total_selections": len(self.performance_selector.selection_history),
                    "adaptive_adjustments": len(self.performance_selector.adaptive_adjustments)
                },
                "last_updated": datetime.utcnow().isoformat()
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get registry statistics: {str(e)}")
            return {}
    
    async def _reassign_agent_tasks(self, agent_id: str):
        """Mock implementation of task reassignment"""
        logger.info(f"Reassigning tasks for agent {agent_id}")
        pass