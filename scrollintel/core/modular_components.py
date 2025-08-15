"""
Modular Component Architecture for ScrollIntel

This module provides a framework for creating loosely coupled, replaceable
system modules with well-defined interfaces and dependency management.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Type, Protocol, runtime_checkable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import json
import uuid
from abc import ABC, abstractmethod
import threading
from collections import defaultdict
import inspect

logger = logging.getLogger(__name__)

class ComponentType(Enum):
    """Types of system components"""
    AGENT = "agent"
    ENGINE = "engine"
    SERVICE = "service"
    PROCESSOR = "processor"
    VALIDATOR = "validator"
    TRANSFORMER = "transformer"
    ANALYZER = "analyzer"
    GENERATOR = "generator"

class ComponentStatus(Enum):
    """Component lifecycle status"""
    CREATED = "created"
    INITIALIZING = "initializing"
    READY = "ready"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"

@runtime_checkable
class ComponentInterface(Protocol):
    """Base interface that all components must implement"""
    
    @property
    def component_id(self) -> str:
        """Unique component identifier"""
        ...
    
    @property
    def component_type(self) -> ComponentType:
        """Component type classification"""
        ...
    
    @property
    def dependencies(self) -> List[str]:
        """List of component IDs this component depends on"""
        ...
    
    async def initialize(self) -> bool:
        """Initialize the component"""
        ...
    
    async def shutdown(self) -> bool:
        """Shutdown the component gracefully"""
        ...
    
    def get_interface_version(self) -> str:
        """Get the interface version for compatibility checking"""
        ...

@dataclass
class ComponentMetadata:
    """Metadata for component registration"""
    component_id: str
    component_type: ComponentType
    name: str
    description: str
    version: str
    interface_version: str
    dependencies: List[str] = field(default_factory=list)
    provides: List[str] = field(default_factory=list)  # What interfaces this component provides
    requires: List[str] = field(default_factory=list)  # What interfaces this component requires
    tags: List[str] = field(default_factory=list)
    configuration: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)@d
ataclass
class ComponentInstance:
    """Runtime instance of a component"""
    metadata: ComponentMetadata
    component: ComponentInterface
    status: ComponentStatus = ComponentStatus.CREATED
    start_time: Optional[datetime] = None
    stop_time: Optional[datetime] = None
    error: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)

class BaseComponent(ABC):
    """Base implementation for modular components"""
    
    def __init__(self, component_id: str, component_type: ComponentType, 
                 name: str, version: str = "1.0.0"):
        self._component_id = component_id
        self._component_type = component_type
        self._name = name
        self._version = version
        self._interface_version = "1.0.0"
        self._dependencies: List[str] = []
        self._status = ComponentStatus.CREATED
        self._initialized = False
        self._lock = threading.RLock()
    
    @property
    def component_id(self) -> str:
        return self._component_id
    
    @property
    def component_type(self) -> ComponentType:
        return self._component_type
    
    @property
    def dependencies(self) -> List[str]:
        return self._dependencies.copy()
    
    def get_interface_version(self) -> str:
        return self._interface_version
    
    async def initialize(self) -> bool:
        """Initialize the component"""
        with self._lock:
            if self._initialized:
                return True
            
            try:
                self._status = ComponentStatus.INITIALIZING
                success = await self._initialize_impl()
                
                if success:
                    self._status = ComponentStatus.READY
                    self._initialized = True
                else:
                    self._status = ComponentStatus.ERROR
                
                return success
                
            except Exception as e:
                logger.error(f"Component {self._component_id} initialization failed: {e}")
                self._status = ComponentStatus.ERROR
                return False
    
    async def shutdown(self) -> bool:
        """Shutdown the component"""
        with self._lock:
            if not self._initialized:
                return True
            
            try:
                self._status = ComponentStatus.STOPPING
                success = await self._shutdown_impl()
                
                self._status = ComponentStatus.STOPPED
                self._initialized = False
                
                return success
                
            except Exception as e:
                logger.error(f"Component {self._component_id} shutdown failed: {e}")
                self._status = ComponentStatus.ERROR
                return False
    
    @abstractmethod
    async def _initialize_impl(self) -> bool:
        """Component-specific initialization logic"""
        pass
    
    @abstractmethod
    async def _shutdown_impl(self) -> bool:
        """Component-specific shutdown logic"""
        pass
    
    def add_dependency(self, component_id: str):
        """Add a dependency to this component"""
        if component_id not in self._dependencies:
            self._dependencies.append(component_id)

class ComponentRegistry:
    """Registry for managing modular components"""
    
    def __init__(self):
        self._components: Dict[str, ComponentInstance] = {}
        self._interfaces: Dict[str, List[str]] = defaultdict(list)  # interface -> component_ids
        self._dependency_graph: Dict[str, List[str]] = defaultdict(list)
        self._reverse_dependencies: Dict[str, List[str]] = defaultdict(list)
        self._lock = threading.RLock()
    
    def register_component(self, component: ComponentInterface, 
                          metadata: ComponentMetadata) -> bool:
        """Register a component with the registry"""
        try:
            with self._lock:
                if metadata.component_id in self._components:
                    logger.warning(f"Component {metadata.component_id} already registered")
                    return False
                
                # Validate component implements required interface
                if not isinstance(component, ComponentInterface):
                    logger.error(f"Component {metadata.component_id} does not implement ComponentInterface")
                    return False
                
                # Create component instance
                instance = ComponentInstance(
                    metadata=metadata,
                    component=component,
                    status=ComponentStatus.CREATED
                )
                
                self._components[metadata.component_id] = instance
                
                # Update interface mappings
                for interface in metadata.provides:
                    self._interfaces[interface].append(metadata.component_id)
                
                # Update dependency graph
                for dep in metadata.dependencies:
                    self._dependency_graph[metadata.component_id].append(dep)
                    self._reverse_dependencies[dep].append(metadata.component_id)
                
                logger.info(f"Registered component {metadata.component_id} ({metadata.component_type.value})")
                return True
                
        except Exception as e:
            logger.error(f"Failed to register component {metadata.component_id}: {e}")
            return False
    
    def unregister_component(self, component_id: str) -> bool:
        """Unregister a component"""
        try:
            with self._lock:
                if component_id not in self._components:
                    return False
                
                instance = self._components[component_id]
                
                # Check if other components depend on this one
                dependents = self._reverse_dependencies.get(component_id, [])
                if dependents:
                    logger.warning(f"Cannot unregister {component_id}, components depend on it: {dependents}")
                    return False
                
                # Shutdown component if running
                if instance.status in [ComponentStatus.READY, ComponentStatus.RUNNING]:
                    asyncio.create_task(instance.component.shutdown())
                
                # Remove from registry
                del self._components[component_id]
                
                # Clean up interface mappings
                for interface in instance.metadata.provides:
                    if component_id in self._interfaces[interface]:
                        self._interfaces[interface].remove(component_id)
                        if not self._interfaces[interface]:
                            del self._interfaces[interface]
                
                # Clean up dependency graph
                for dep in instance.metadata.dependencies:
                    if component_id in self._reverse_dependencies[dep]:
                        self._reverse_dependencies[dep].remove(component_id)
                
                del self._dependency_graph[component_id]
                
                logger.info(f"Unregistered component {component_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to unregister component {component_id}: {e}")
            return False
    
    def get_component(self, component_id: str) -> Optional[ComponentInstance]:
        """Get component instance by ID"""
        with self._lock:
            return self._components.get(component_id)
    
    def find_components_by_interface(self, interface: str) -> List[ComponentInstance]:
        """Find components that provide a specific interface"""
        with self._lock:
            component_ids = self._interfaces.get(interface, [])
            return [self._components[cid] for cid in component_ids if cid in self._components]
    
    def find_components_by_type(self, component_type: ComponentType) -> List[ComponentInstance]:
        """Find components by type"""
        with self._lock:
            return [
                instance for instance in self._components.values()
                if instance.metadata.component_type == component_type
            ]
    
    def get_dependency_order(self) -> List[str]:
        """Get components in dependency order (topological sort)"""
        with self._lock:
            # Kahn's algorithm for topological sorting
            in_degree = {cid: 0 for cid in self._components.keys()}
            
            # Calculate in-degrees
            for cid, deps in self._dependency_graph.items():
                for dep in deps:
                    if dep in in_degree:
                        in_degree[cid] += 1
            
            # Find nodes with no incoming edges
            queue = [cid for cid, degree in in_degree.items() if degree == 0]
            result = []
            
            while queue:
                current = queue.pop(0)
                result.append(current)
                
                # Update in-degrees of dependent components
                for dependent in self._reverse_dependencies.get(current, []):
                    in_degree[dependent] -= 1
                    if in_degree[dependent] == 0:
                        queue.append(dependent)
            
            # Check for circular dependencies
            if len(result) != len(self._components):
                remaining = set(self._components.keys()) - set(result)
                logger.error(f"Circular dependencies detected in components: {remaining}")
                return []
            
            return result
    
    async def initialize_all(self) -> bool:
        """Initialize all components in dependency order"""
        dependency_order = self.get_dependency_order()
        if not dependency_order:
            return False
        
        success = True
        for component_id in dependency_order:
            instance = self._components.get(component_id)
            if instance:
                logger.info(f"Initializing component {component_id}")
                instance.start_time = datetime.now()
                
                if not await instance.component.initialize():
                    logger.error(f"Failed to initialize component {component_id}")
                    instance.status = ComponentStatus.ERROR
                    success = False
                    break
                else:
                    instance.status = ComponentStatus.READY
        
        return success
    
    async def shutdown_all(self) -> bool:
        """Shutdown all components in reverse dependency order"""
        dependency_order = self.get_dependency_order()
        if not dependency_order:
            return False
        
        success = True
        for component_id in reversed(dependency_order):
            instance = self._components.get(component_id)
            if instance and instance.status in [ComponentStatus.READY, ComponentStatus.RUNNING]:
                logger.info(f"Shutting down component {component_id}")
                
                if not await instance.component.shutdown():
                    logger.error(f"Failed to shutdown component {component_id}")
                    success = False
                else:
                    instance.status = ComponentStatus.STOPPED
                    instance.stop_time = datetime.now()
        
        return success
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status"""
        with self._lock:
            status_counts = defaultdict(int)
            for instance in self._components.values():
                status_counts[instance.status.value] += 1
            
            return {
                "total_components": len(self._components),
                "status_breakdown": dict(status_counts),
                "interfaces_provided": len(self._interfaces),
                "dependency_graph_valid": len(self.get_dependency_order()) == len(self._components)
            }

# Global component registry
component_registry = ComponentRegistry()