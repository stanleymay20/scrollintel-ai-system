"""
Research Project Management Engine for Autonomous Innovation Lab

This engine provides autonomous research project planning, execution,
milestone tracking, and resource coordination capabilities.
"""

import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Tuple
import logging
from dataclasses import asdict

from ..models.research_coordination_models import (
    ResearchProject, ResearchMilestone, ResearchResource, ResearchCoordinationMetrics,
    ProjectStatus, MilestoneStatus, ResourceType
)
from ..models.research_coordination_models import ResearchTopic, Hypothesis
from .base_engine import BaseEngine


class ResearchProjectManager(BaseEngine):
    """
    Autonomous research project management engine that handles:
    - Research project planning and execution
    - Milestone tracking and management
    - Resource coordination and optimization
    """
    
    def __init__(self):
        from ..engines.base_engine import EngineCapability
        super().__init__(
            engine_id="research_project_manager",
            name="Research Project Manager",
            capabilities=[EngineCapability.DATA_ANALYSIS]
        )
        self.logger = logging.getLogger(__name__)
        self.active_projects: Dict[str, ResearchProject] = {}
        self.resource_pool: Dict[str, ResearchResource] = {}
        self.project_templates: Dict[str, Dict[str, Any]] = {}
        
        # Initialize default project templates
        self._initialize_project_templates()
    
    async def initialize(self) -> None:
        """Initialize the research project manager"""
        self.logger.info("Initializing Research Project Manager")
        # Initialize any required resources
        pass
    
    async def process(self, input_data: Any, parameters: Dict[str, Any] = None) -> Any:
        """Process research project management requests"""
        # This could be used for batch processing of projects
        return {"status": "processed", "data": input_data}
    
    async def cleanup(self) -> None:
        """Clean up research project manager resources"""
        self.logger.info("Cleaning up Research Project Manager")
        # Clean up any resources
        pass
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of the research project manager"""
        return {
            "healthy": True,
            "active_projects": len(self.active_projects),
            "resource_pool_size": len(self.resource_pool),
            "status": self.status.value
        }
    
    def _initialize_project_templates(self):
        """Initialize default research project templates"""
        self.project_templates = {
            "basic_research": {
                "milestones": [
                    {"name": "Literature Review", "duration_days": 14, "dependencies": []},
                    {"name": "Hypothesis Formation", "duration_days": 7, "dependencies": ["Literature Review"]},
                    {"name": "Experimental Design", "duration_days": 10, "dependencies": ["Hypothesis Formation"]},
                    {"name": "Data Collection", "duration_days": 30, "dependencies": ["Experimental Design"]},
                    {"name": "Analysis", "duration_days": 21, "dependencies": ["Data Collection"]},
                    {"name": "Results Validation", "duration_days": 14, "dependencies": ["Analysis"]},
                    {"name": "Publication", "duration_days": 30, "dependencies": ["Results Validation"]}
                ],
                "resource_requirements": {
                    ResourceType.COMPUTATIONAL: 100.0,
                    ResourceType.DATA: 50.0,
                    ResourceType.BUDGET: 10000.0
                }
            },
            "applied_research": {
                "milestones": [
                    {"name": "Problem Definition", "duration_days": 7, "dependencies": []},
                    {"name": "Solution Design", "duration_days": 14, "dependencies": ["Problem Definition"]},
                    {"name": "Prototype Development", "duration_days": 21, "dependencies": ["Solution Design"]},
                    {"name": "Testing", "duration_days": 14, "dependencies": ["Prototype Development"]},
                    {"name": "Optimization", "duration_days": 14, "dependencies": ["Testing"]},
                    {"name": "Validation", "duration_days": 10, "dependencies": ["Optimization"]},
                    {"name": "Deployment", "duration_days": 7, "dependencies": ["Validation"]}
                ],
                "resource_requirements": {
                    ResourceType.COMPUTATIONAL: 200.0,
                    ResourceType.EQUIPMENT: 50.0,
                    ResourceType.BUDGET: 25000.0
                }
            }
        }
    
    async def create_research_project(
        self,
        research_topic: ResearchTopic,
        hypotheses: List[Hypothesis],
        project_type: str = "basic_research",
        priority: int = 5
    ) -> ResearchProject:
        """
        Create a new autonomous research project
        
        Args:
            research_topic: Research topic to investigate
            hypotheses: Research hypotheses to test
            project_type: Type of research project template to use
            priority: Project priority (1-10)
            
        Returns:
            Created research project
        """
        try:
            # Create project
            project = ResearchProject(
                name=f"Research: {research_topic.title}",
                description=research_topic.description,
                research_domain=research_topic.domain,
                objectives=research_topic.research_questions,
                hypotheses=[h.statement for h in hypotheses],
                methodology=research_topic.methodology,
                priority=priority
            )
            
            # Apply project template
            await self._apply_project_template(project, project_type)
            
            # Allocate initial resources
            await self._allocate_project_resources(project, project_type)
            
            # Generate project timeline
            await self._generate_project_timeline(project)
            
            # Store project
            self.active_projects[project.id] = project
            
            self.logger.info(f"Created research project: {project.name}")
            return project
            
        except Exception as e:
            self.logger.error(f"Error creating research project: {str(e)}")
            raise
    
    async def _apply_project_template(self, project: ResearchProject, template_name: str):
        """Apply project template to create milestones"""
        if template_name not in self.project_templates:
            template_name = "basic_research"
        
        template = self.project_templates[template_name]
        
        # Create milestones from template
        for milestone_config in template["milestones"]:
            milestone = ResearchMilestone(
                project_id=project.id,
                name=milestone_config["name"],
                description=f"Complete {milestone_config['name']} for {project.name}",
                dependencies=milestone_config.get("dependencies", [])
            )
            
            # Set success criteria based on milestone type
            milestone.success_criteria = self._generate_milestone_criteria(milestone.name)
            
            project.milestones.append(milestone)
    
    def _generate_milestone_criteria(self, milestone_name: str) -> List[str]:
        """Generate success criteria for milestone"""
        criteria_map = {
            "Literature Review": [
                "Comprehensive review of relevant literature completed",
                "Knowledge gaps identified",
                "Research landscape mapped"
            ],
            "Hypothesis Formation": [
                "Testable hypotheses formulated",
                "Hypotheses aligned with research objectives",
                "Success metrics defined"
            ],
            "Experimental Design": [
                "Experimental methodology defined",
                "Control variables identified",
                "Data collection plan created"
            ],
            "Data Collection": [
                "Required data collected",
                "Data quality validated",
                "Data preprocessing completed"
            ],
            "Analysis": [
                "Statistical analysis completed",
                "Results interpreted",
                "Findings documented"
            ]
        }
        
        return criteria_map.get(milestone_name, ["Milestone objectives achieved"])
    
    async def _allocate_project_resources(self, project: ResearchProject, template_name: str):
        """Allocate resources to project based on template"""
        if template_name not in self.project_templates:
            return
        
        template = self.project_templates[template_name]
        resource_requirements = template.get("resource_requirements", {})
        
        for resource_type, amount in resource_requirements.items():
            # Find available resources
            available_resource = await self._find_available_resource(resource_type, amount)
            
            if available_resource:
                # Allocate resource
                available_resource.allocated += amount
                available_resource.available = available_resource.capacity - available_resource.allocated
                project.allocated_resources.append(available_resource)
            else:
                # Create new resource if none available
                new_resource = ResearchResource(
                    resource_type=resource_type,
                    capacity=amount * 2,  # Create with extra capacity
                    allocated=amount,
                    cost_per_unit=self._get_resource_cost(resource_type)
                )
                self.resource_pool[new_resource.id] = new_resource
                project.allocated_resources.append(new_resource)
    
    async def _find_available_resource(
        self, 
        resource_type: ResourceType, 
        required_amount: float
    ) -> Optional[ResearchResource]:
        """Find available resource of specified type and amount"""
        for resource in self.resource_pool.values():
            if (resource.resource_type == resource_type and 
                resource.available >= required_amount):
                return resource
        return None
    
    def _get_resource_cost(self, resource_type: ResourceType) -> float:
        """Get cost per unit for resource type"""
        cost_map = {
            ResourceType.COMPUTATIONAL: 0.1,
            ResourceType.DATA: 0.05,
            ResourceType.HUMAN: 100.0,
            ResourceType.EQUIPMENT: 10.0,
            ResourceType.BUDGET: 1.0
        }
        return cost_map.get(resource_type, 1.0)
    
    async def _generate_project_timeline(self, project: ResearchProject):
        """Generate project timeline based on milestones"""
        if not project.milestones:
            return
        
        # Sort milestones by dependencies
        sorted_milestones = self._topological_sort_milestones(project.milestones)
        
        # Set timeline
        current_date = datetime.now()
        project.planned_start = current_date
        
        for milestone in sorted_milestones:
            # Calculate start date based on dependencies
            milestone_start = current_date
            
            for dep_name in milestone.dependencies:
                dep_milestone = next((m for m in project.milestones if m.name == dep_name), None)
                if dep_milestone and dep_milestone.planned_end:
                    milestone_start = max(milestone_start, dep_milestone.planned_end)
            
            milestone.planned_start = milestone_start
            
            # Estimate duration (default 7 days if not specified)
            duration_days = milestone.metadata.get("duration_days", 7)
            milestone.planned_end = milestone_start + timedelta(days=duration_days)
            
            current_date = milestone.planned_end
        
        # Set project end date
        project.planned_end = max(m.planned_end for m in project.milestones if m.planned_end)
    
    def _topological_sort_milestones(self, milestones: List[ResearchMilestone]) -> List[ResearchMilestone]:
        """Sort milestones based on dependencies"""
        # Simple topological sort implementation
        sorted_milestones = []
        remaining = milestones.copy()
        
        while remaining:
            # Find milestones with no unresolved dependencies
            ready = []
            for milestone in remaining:
                deps_satisfied = all(
                    any(m.name == dep for m in sorted_milestones)
                    for dep in milestone.dependencies
                )
                if deps_satisfied:
                    ready.append(milestone)
            
            if not ready:
                # Handle circular dependencies by taking first remaining
                ready = [remaining[0]]
            
            # Add ready milestones to sorted list
            for milestone in ready:
                sorted_milestones.append(milestone)
                remaining.remove(milestone)
        
        return sorted_milestones
    
    async def update_milestone_progress(
        self, 
        project_id: str, 
        milestone_id: str, 
        progress: float,
        status: Optional[MilestoneStatus] = None
    ) -> bool:
        """
        Update milestone progress and status
        
        Args:
            project_id: Project identifier
            milestone_id: Milestone identifier
            progress: Progress percentage (0-100)
            status: Optional milestone status
            
        Returns:
            Success status
        """
        try:
            if project_id not in self.active_projects:
                return False
            
            project = self.active_projects[project_id]
            milestone = next((m for m in project.milestones if m.id == milestone_id), None)
            
            if not milestone:
                return False
            
            # Update progress
            milestone.progress_percentage = min(100.0, max(0.0, progress))
            
            # Update status
            if status:
                milestone.status = status
            elif progress >= 100.0:
                milestone.status = MilestoneStatus.COMPLETED
                milestone.actual_end = datetime.now()
            elif progress > 0:
                milestone.status = MilestoneStatus.IN_PROGRESS
                if not milestone.actual_start:
                    milestone.actual_start = datetime.now()
            
            # Update project progress
            project.progress_percentage = project.calculate_progress()
            project.updated_at = datetime.now()
            
            # Check if project is completed
            if all(m.status == MilestoneStatus.COMPLETED for m in project.milestones):
                project.status = ProjectStatus.COMPLETED
                project.actual_end = datetime.now()
            
            self.logger.info(f"Updated milestone {milestone.name} progress to {progress}%")
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating milestone progress: {str(e)}")
            return False
    
    async def optimize_resource_allocation(self, project_id: str) -> Dict[str, Any]:
        """
        Optimize resource allocation for project
        
        Args:
            project_id: Project identifier
            
        Returns:
            Optimization results
        """
        try:
            if project_id not in self.active_projects:
                return {"error": "Project not found"}
            
            project = self.active_projects[project_id]
            
            # Analyze current resource utilization
            utilization = project.get_resource_utilization()
            
            # Identify optimization opportunities
            optimizations = []
            
            for resource_type, util_rate in utilization.items():
                if util_rate < 50:  # Under-utilized
                    optimizations.append({
                        "type": "reduce_allocation",
                        "resource_type": resource_type.value,
                        "current_utilization": util_rate,
                        "recommended_reduction": (50 - util_rate) / 100
                    })
                elif util_rate > 90:  # Over-utilized
                    optimizations.append({
                        "type": "increase_allocation",
                        "resource_type": resource_type.value,
                        "current_utilization": util_rate,
                        "recommended_increase": (util_rate - 90) / 100
                    })
            
            # Apply optimizations
            for optimization in optimizations:
                await self._apply_resource_optimization(project, optimization)
            
            return {
                "project_id": project_id,
                "optimizations_applied": len(optimizations),
                "optimizations": optimizations,
                "new_utilization": project.get_resource_utilization()
            }
            
        except Exception as e:
            self.logger.error(f"Error optimizing resource allocation: {str(e)}")
            return {"error": str(e)}
    
    async def _apply_resource_optimization(
        self, 
        project: ResearchProject, 
        optimization: Dict[str, Any]
    ):
        """Apply resource optimization to project"""
        resource_type_str = optimization["resource_type"]
        resource_type = ResourceType(resource_type_str)
        
        # Find project resource of this type
        project_resource = next(
            (r for r in project.allocated_resources if r.resource_type == resource_type),
            None
        )
        
        if not project_resource:
            return
        
        if optimization["type"] == "reduce_allocation":
            reduction = optimization["recommended_reduction"]
            amount_to_reduce = project_resource.allocated * reduction
            project_resource.allocated -= amount_to_reduce
            project_resource.available += amount_to_reduce
            
        elif optimization["type"] == "increase_allocation":
            increase = optimization["recommended_increase"]
            amount_to_increase = project_resource.allocated * increase
            
            # Check if resource has capacity
            if project_resource.available >= amount_to_increase:
                project_resource.allocated += amount_to_increase
                project_resource.available -= amount_to_increase
    
    async def get_project_status(self, project_id: str) -> Optional[Dict[str, Any]]:
        """
        Get comprehensive project status
        
        Args:
            project_id: Project identifier
            
        Returns:
            Project status information
        """
        try:
            if project_id not in self.active_projects:
                return None
            
            project = self.active_projects[project_id]
            
            # Calculate metrics
            active_milestones = project.get_active_milestones()
            overdue_milestones = project.get_overdue_milestones()
            resource_utilization = project.get_resource_utilization()
            
            return {
                "project": asdict(project),
                "metrics": {
                    "progress_percentage": project.progress_percentage,
                    "active_milestones": len(active_milestones),
                    "overdue_milestones": len(overdue_milestones),
                    "resource_utilization": resource_utilization,
                    "budget_utilization": (project.budget_used / project.budget_allocated * 100) if project.budget_allocated > 0 else 0
                },
                "next_milestones": [asdict(m) for m in active_milestones[:3]],
                "risks": self._identify_project_risks(project)
            }
            
        except Exception as e:
            self.logger.error(f"Error getting project status: {str(e)}")
            return None
    
    def _identify_project_risks(self, project: ResearchProject) -> List[Dict[str, Any]]:
        """Identify project risks"""
        risks = []
        
        # Check for overdue milestones
        overdue_milestones = project.get_overdue_milestones()
        if overdue_milestones:
            risks.append({
                "type": "schedule_risk",
                "severity": "high" if len(overdue_milestones) > 2 else "medium",
                "description": f"{len(overdue_milestones)} milestones are overdue",
                "mitigation": "Review milestone dependencies and resource allocation"
            })
        
        # Check resource utilization
        utilization = project.get_resource_utilization()
        for resource_type, util_rate in utilization.items():
            if util_rate > 95:
                risks.append({
                    "type": "resource_risk",
                    "severity": "high",
                    "description": f"{resource_type.value} resources over-utilized ({util_rate:.1f}%)",
                    "mitigation": "Increase resource allocation or adjust timeline"
                })
        
        # Check budget utilization
        if project.budget_allocated > 0:
            budget_util = project.budget_used / project.budget_allocated * 100
            if budget_util > 90:
                risks.append({
                    "type": "budget_risk",
                    "severity": "high" if budget_util > 100 else "medium",
                    "description": f"Budget utilization at {budget_util:.1f}%",
                    "mitigation": "Review spending and adjust budget allocation"
                })
        
        return risks
    
    async def get_coordination_metrics(self) -> ResearchCoordinationMetrics:
        """
        Get overall research coordination metrics
        
        Returns:
            Coordination performance metrics
        """
        try:
            metrics = ResearchCoordinationMetrics()
            
            # Project metrics
            metrics.total_projects = len(self.active_projects)
            metrics.active_projects = len([p for p in self.active_projects.values() 
                                         if p.status == ProjectStatus.ACTIVE])
            metrics.completed_projects = len([p for p in self.active_projects.values() 
                                            if p.status == ProjectStatus.COMPLETED])
            
            # Resource metrics
            metrics.total_resources = len(self.resource_pool)
            if self.resource_pool:
                total_capacity = sum(r.capacity for r in self.resource_pool.values())
                total_allocated = sum(r.allocated for r in self.resource_pool.values())
                metrics.resource_utilization_rate = (total_allocated / total_capacity * 100) if total_capacity > 0 else 0
            
            # Milestone metrics
            all_milestones = []
            for project in self.active_projects.values():
                all_milestones.extend(project.milestones)
            
            metrics.total_milestones = len(all_milestones)
            metrics.completed_milestones = len([m for m in all_milestones 
                                              if m.status == MilestoneStatus.COMPLETED])
            metrics.overdue_milestones = len([m for m in all_milestones if m.is_overdue()])
            
            if metrics.total_milestones > 0:
                metrics.milestone_completion_rate = (metrics.completed_milestones / metrics.total_milestones * 100)
            
            # Performance metrics
            completed_projects = [p for p in self.active_projects.values() 
                                if p.status == ProjectStatus.COMPLETED and p.actual_start and p.actual_end]
            
            if completed_projects:
                total_duration = sum((p.actual_end - p.actual_start).days for p in completed_projects)
                metrics.average_project_duration = total_duration / len(completed_projects)
                metrics.success_rate = len(completed_projects) / metrics.total_projects * 100
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating coordination metrics: {str(e)}")
            return ResearchCoordinationMetrics()