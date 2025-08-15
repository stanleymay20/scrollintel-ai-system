"""
Tests for Research Project Management Engine

This module tests the autonomous research project planning, execution,
milestone tracking, and resource coordination capabilities.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from scrollintel.engines.research_project_manager import ResearchProjectManager
from scrollintel.models.research_coordination_models import (
    ResearchProject, ResearchMilestone, ResearchResource, 
    ProjectStatus, MilestoneStatus, ResourceType
)
from scrollintel.models.research_coordination_models import ResearchTopic, Hypothesis


@pytest.fixture
def project_manager():
    """Create research project manager instance"""
    return ResearchProjectManager()


@pytest.fixture
def sample_research_topic():
    """Create sample research topic"""
    return ResearchTopic(
        title="Machine Learning Optimization",
        description="Research on optimizing ML algorithms",
        domain="artificial_intelligence",
        research_questions=["How to improve ML efficiency?", "What are optimal parameters?"],
        methodology="experimental_analysis"
    )


@pytest.fixture
def sample_hypotheses():
    """Create sample hypotheses"""
    return [
        Hypothesis(
            statement="Optimized algorithms perform 20% better",
            confidence=0.8,
            testable=True
        ),
        Hypothesis(
            statement="Parameter tuning reduces training time",
            confidence=0.7,
            testable=True
        )
    ]


class TestResearchProjectManager:
    """Test research project management functionality"""
    
    @pytest.mark.asyncio
    async def test_create_research_project(self, project_manager, sample_research_topic, sample_hypotheses):
        """Test creating a new research project"""
        # Create project
        project = await project_manager.create_research_project(
            research_topic=sample_research_topic,
            hypotheses=sample_hypotheses,
            project_type="basic_research",
            priority=7
        )
        
        # Verify project creation
        assert project is not None
        assert project.name == f"Research: {sample_research_topic.title}"
        assert project.description == sample_research_topic.description
        assert project.research_domain == sample_research_topic.domain
        assert project.priority == 7
        assert project.status == ProjectStatus.PLANNING
        assert len(project.objectives) == len(sample_research_topic.research_questions)
        assert len(project.hypotheses) == len(sample_hypotheses)
        
        # Verify milestones created
        assert len(project.milestones) > 0
        milestone_names = [m.name for m in project.milestones]
        assert "Literature Review" in milestone_names
        assert "Hypothesis Formation" in milestone_names
        
        # Verify resources allocated
        assert len(project.allocated_resources) > 0
        
        # Verify timeline generated
        assert project.planned_start is not None
        assert project.planned_end is not None
        assert project.planned_end > project.planned_start
        
        # Verify project stored
        assert project.id in project_manager.active_projects
    
    @pytest.mark.asyncio
    async def test_create_applied_research_project(self, project_manager, sample_research_topic, sample_hypotheses):
        """Test creating applied research project"""
        project = await project_manager.create_research_project(
            research_topic=sample_research_topic,
            hypotheses=sample_hypotheses,
            project_type="applied_research",
            priority=5
        )
        
        # Verify applied research milestones
        milestone_names = [m.name for m in project.milestones]
        assert "Problem Definition" in milestone_names
        assert "Solution Design" in milestone_names
        assert "Prototype Development" in milestone_names
        assert "Deployment" in milestone_names
    
    @pytest.mark.asyncio
    async def test_update_milestone_progress(self, project_manager, sample_research_topic, sample_hypotheses):
        """Test updating milestone progress"""
        # Create project
        project = await project_manager.create_research_project(
            research_topic=sample_research_topic,
            hypotheses=sample_hypotheses
        )
        
        # Get first milestone
        milestone = project.milestones[0]
        
        # Update progress
        success = await project_manager.update_milestone_progress(
            project_id=project.id,
            milestone_id=milestone.id,
            progress=50.0
        )
        
        assert success is True
        
        # Verify milestone updated
        updated_milestone = next(m for m in project.milestones if m.id == milestone.id)
        assert updated_milestone.progress_percentage == 50.0
        assert updated_milestone.status == MilestoneStatus.IN_PROGRESS
        assert updated_milestone.actual_start is not None
        
        # Complete milestone
        success = await project_manager.update_milestone_progress(
            project_id=project.id,
            milestone_id=milestone.id,
            progress=100.0
        )
        
        assert success is True
        updated_milestone = next(m for m in project.milestones if m.id == milestone.id)
        assert updated_milestone.status == MilestoneStatus.COMPLETED
        assert updated_milestone.actual_end is not None
    
    @pytest.mark.asyncio
    async def test_update_milestone_progress_invalid_project(self, project_manager):
        """Test updating milestone progress with invalid project"""
        success = await project_manager.update_milestone_progress(
            project_id="invalid_id",
            milestone_id="milestone_id",
            progress=50.0
        )
        
        assert success is False
    
    @pytest.mark.asyncio
    async def test_optimize_resource_allocation(self, project_manager, sample_research_topic, sample_hypotheses):
        """Test resource allocation optimization"""
        # Create project
        project = await project_manager.create_research_project(
            research_topic=sample_research_topic,
            hypotheses=sample_hypotheses
        )
        
        # Simulate resource over-utilization
        if project.allocated_resources:
            resource = project.allocated_resources[0]
            resource.allocated = resource.capacity * 0.95  # 95% utilization
        
        # Optimize resources
        result = await project_manager.optimize_resource_allocation(project.id)
        
        assert "error" not in result
        assert "project_id" in result
        assert "optimizations_applied" in result
        assert result["project_id"] == project.id
    
    @pytest.mark.asyncio
    async def test_optimize_resource_allocation_invalid_project(self, project_manager):
        """Test resource optimization with invalid project"""
        result = await project_manager.optimize_resource_allocation("invalid_id")
        
        assert "error" in result
        assert result["error"] == "Project not found"
    
    @pytest.mark.asyncio
    async def test_get_project_status(self, project_manager, sample_research_topic, sample_hypotheses):
        """Test getting project status"""
        # Create project
        project = await project_manager.create_research_project(
            research_topic=sample_research_topic,
            hypotheses=sample_hypotheses
        )
        
        # Update some milestone progress
        if project.milestones:
            await project_manager.update_milestone_progress(
                project_id=project.id,
                milestone_id=project.milestones[0].id,
                progress=75.0
            )
        
        # Get status
        status = await project_manager.get_project_status(project.id)
        
        assert status is not None
        assert "project" in status
        assert "metrics" in status
        assert "next_milestones" in status
        assert "risks" in status
        
        # Verify metrics
        metrics = status["metrics"]
        assert "progress_percentage" in metrics
        assert "active_milestones" in metrics
        assert "resource_utilization" in metrics
        assert "budget_utilization" in metrics
        
        # Verify project data
        project_data = status["project"]
        assert project_data["id"] == project.id
        assert project_data["name"] == project.name
    
    @pytest.mark.asyncio
    async def test_get_project_status_invalid_project(self, project_manager):
        """Test getting status for invalid project"""
        status = await project_manager.get_project_status("invalid_id")
        assert status is None
    
    @pytest.mark.asyncio
    async def test_project_completion(self, project_manager, sample_research_topic, sample_hypotheses):
        """Test project completion when all milestones are done"""
        # Create project
        project = await project_manager.create_research_project(
            research_topic=sample_research_topic,
            hypotheses=sample_hypotheses
        )
        
        # Complete all milestones
        for milestone in project.milestones:
            await project_manager.update_milestone_progress(
                project_id=project.id,
                milestone_id=milestone.id,
                progress=100.0
            )
        
        # Verify project completed
        updated_project = project_manager.active_projects[project.id]
        assert updated_project.status == ProjectStatus.COMPLETED
        assert updated_project.actual_end is not None
    
    @pytest.mark.asyncio
    async def test_get_coordination_metrics(self, project_manager, sample_research_topic, sample_hypotheses):
        """Test getting coordination metrics"""
        # Create multiple projects
        for i in range(3):
            topic = ResearchTopic(
                title=f"Research Topic {i}",
                description=f"Description {i}",
                domain="test_domain",
                research_questions=[f"Question {i}"],
                methodology="test_method"
            )
            
            await project_manager.create_research_project(
                research_topic=topic,
                hypotheses=sample_hypotheses,
                priority=i + 1
            )
        
        # Get metrics
        metrics = await project_manager.get_coordination_metrics()
        
        assert metrics is not None
        assert metrics.total_projects == 3
        assert metrics.active_projects >= 0
        assert metrics.total_resources > 0
        assert metrics.total_milestones > 0
        assert metrics.calculated_at is not None
    
    def test_milestone_dependencies_sorting(self, project_manager):
        """Test milestone dependency sorting"""
        # Create milestones with dependencies
        milestones = [
            ResearchMilestone(name="C", dependencies=["A", "B"]),
            ResearchMilestone(name="A", dependencies=[]),
            ResearchMilestone(name="B", dependencies=["A"]),
            ResearchMilestone(name="D", dependencies=["C"])
        ]
        
        # Sort milestones
        sorted_milestones = project_manager._topological_sort_milestones(milestones)
        
        # Verify order
        names = [m.name for m in sorted_milestones]
        assert names.index("A") < names.index("B")
        assert names.index("A") < names.index("C")
        assert names.index("B") < names.index("C")
        assert names.index("C") < names.index("D")
    
    def test_milestone_criteria_generation(self, project_manager):
        """Test milestone success criteria generation"""
        criteria = project_manager._generate_milestone_criteria("Literature Review")
        
        assert len(criteria) > 0
        assert any("literature" in c.lower() for c in criteria)
        assert any("review" in c.lower() for c in criteria)
        
        # Test unknown milestone
        criteria = project_manager._generate_milestone_criteria("Unknown Milestone")
        assert len(criteria) == 1
        assert "objectives achieved" in criteria[0].lower()
    
    def test_resource_cost_calculation(self, project_manager):
        """Test resource cost calculation"""
        computational_cost = project_manager._get_resource_cost(ResourceType.COMPUTATIONAL)
        data_cost = project_manager._get_resource_cost(ResourceType.DATA)
        human_cost = project_manager._get_resource_cost(ResourceType.HUMAN)
        
        assert computational_cost > 0
        assert data_cost > 0
        assert human_cost > 0
        assert human_cost > computational_cost  # Human resources should be more expensive
    
    def test_project_risk_identification(self, project_manager):
        """Test project risk identification"""
        # Create project with overdue milestones
        project = ResearchProject(
            name="Test Project",
            planned_start=datetime.now() - timedelta(days=30),
            planned_end=datetime.now() + timedelta(days=30)
        )
        
        # Add overdue milestone
        overdue_milestone = ResearchMilestone(
            project_id=project.id,
            name="Overdue Milestone",
            planned_end=datetime.now() - timedelta(days=5),
            status=MilestoneStatus.IN_PROGRESS
        )
        project.milestones.append(overdue_milestone)
        
        # Add over-utilized resource
        over_resource = ResearchResource(
            resource_type=ResourceType.COMPUTATIONAL,
            capacity=100.0,
            allocated=98.0
        )
        project.allocated_resources.append(over_resource)
        
        # Set high budget utilization
        project.budget_allocated = 1000.0
        project.budget_used = 950.0
        
        # Identify risks
        risks = project_manager._identify_project_risks(project)
        
        assert len(risks) > 0
        
        # Check for schedule risk
        schedule_risks = [r for r in risks if r["type"] == "schedule_risk"]
        assert len(schedule_risks) > 0
        
        # Check for resource risk
        resource_risks = [r for r in risks if r["type"] == "resource_risk"]
        assert len(resource_risks) > 0
        
        # Check for budget risk
        budget_risks = [r for r in risks if r["type"] == "budget_risk"]
        assert len(budget_risks) > 0
    
    @pytest.mark.asyncio
    async def test_find_available_resource(self, project_manager):
        """Test finding available resources"""
        # Add resource to pool
        resource = ResearchResource(
            resource_type=ResourceType.COMPUTATIONAL,
            capacity=100.0,
            allocated=30.0
        )
        project_manager.resource_pool[resource.id] = resource
        
        # Find available resource
        found_resource = await project_manager._find_available_resource(
            ResourceType.COMPUTATIONAL, 50.0
        )
        
        assert found_resource is not None
        assert found_resource.id == resource.id
        
        # Try to find resource with insufficient capacity
        found_resource = await project_manager._find_available_resource(
            ResourceType.COMPUTATIONAL, 80.0
        )
        
        assert found_resource is None
    
    def test_project_progress_calculation(self):
        """Test project progress calculation"""
        project = ResearchProject()
        
        # Add milestones with different progress
        milestone1 = ResearchMilestone(progress_percentage=50.0)
        milestone2 = ResearchMilestone(progress_percentage=75.0)
        milestone3 = ResearchMilestone(progress_percentage=25.0)
        
        project.milestones = [milestone1, milestone2, milestone3]
        
        # Calculate progress
        progress = project.calculate_progress()
        
        expected_progress = (50.0 + 75.0 + 25.0) / 3
        assert progress == expected_progress
        
        # Test with no milestones
        project.milestones = []
        progress = project.calculate_progress()
        assert progress == 0.0
    
    def test_resource_utilization_calculation(self):
        """Test resource utilization calculation"""
        project = ResearchProject()
        
        # Add resources with different utilization
        resource1 = ResearchResource(
            resource_type=ResourceType.COMPUTATIONAL,
            capacity=100.0,
            allocated=80.0
        )
        resource2 = ResearchResource(
            resource_type=ResourceType.COMPUTATIONAL,
            capacity=50.0,
            allocated=30.0
        )
        resource3 = ResearchResource(
            resource_type=ResourceType.DATA,
            capacity=200.0,
            allocated=100.0
        )
        
        project.allocated_resources = [resource1, resource2, resource3]
        
        # Calculate utilization
        utilization = project.get_resource_utilization()
        
        # Verify computational utilization
        expected_comp_util = ((80.0/100.0) + (30.0/50.0)) * 100
        assert utilization[ResourceType.COMPUTATIONAL] == expected_comp_util
        
        # Verify data utilization
        expected_data_util = (100.0/200.0) * 100
        assert utilization[ResourceType.DATA] == expected_data_util
    
    def test_milestone_overdue_check(self):
        """Test milestone overdue checking"""
        # Create overdue milestone
        overdue_milestone = ResearchMilestone(
            name="Overdue",
            planned_end=datetime.now() - timedelta(days=1),
            status=MilestoneStatus.IN_PROGRESS
        )
        
        assert overdue_milestone.is_overdue() is True
        
        # Create future milestone
        future_milestone = ResearchMilestone(
            name="Future",
            planned_end=datetime.now() + timedelta(days=1),
            status=MilestoneStatus.IN_PROGRESS
        )
        
        assert future_milestone.is_overdue() is False
        
        # Create completed milestone (should not be overdue)
        completed_milestone = ResearchMilestone(
            name="Completed",
            planned_end=datetime.now() - timedelta(days=1),
            status=MilestoneStatus.COMPLETED
        )
        
        assert completed_milestone.is_overdue() is False
    
    def test_milestone_duration_calculation(self):
        """Test milestone duration calculation"""
        start_date = datetime.now()
        end_date = start_date + timedelta(days=7)
        
        milestone = ResearchMilestone(
            planned_start=start_date,
            planned_end=end_date
        )
        
        duration = milestone.get_duration_days()
        assert duration == 7
        
        # Test with missing dates
        milestone_no_dates = ResearchMilestone()
        duration = milestone_no_dates.get_duration_days()
        assert duration is None


if __name__ == "__main__":
    pytest.main([__file__])