"""
Comprehensive tests for the complete Autonomous Innovation Lab system
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

from scrollintel.core.autonomous_innovation_lab import (
    AutonomousInnovationLab, LabConfiguration
)
from scrollintel.models.innovation_lab_integration_models import (
    InnovationProject, LabStatus
)

class TestAutonomousInnovationLabComplete:
    """Test the complete autonomous innovation lab system"""
    
    @pytest.fixture
    def lab_config(self):
        """Create test lab configuration"""
        return LabConfiguration(
            max_concurrent_projects=5,
            research_domains=["ai", "quantum", "biotech"],
            quality_threshold=0.7,
            innovation_targets={
                "breakthrough_innovations": 3,
                "validated_prototypes": 10,
                "research_publications": 20,
                "patent_applications": 5
            },
            continuous_learning=True
        )
    
    @pytest.fixture
    def mock_engines(self):
        """Create mock engines for testing"""
        engines = {}
        
        # Mock research engine
        engines['research_engine'] = Mock()
        engines['research_engine'].is_ready = AsyncMock(return_value=True)
        engines['research_engine'].start = AsyncMock()
        engines['research_engine'].generate_research_topics = AsyncMock(return_value=[
            Mock(title="AI Innovation", id="topic_1"),
            Mock(title="Quantum Computing", id="topic_2")
        ])
        engines['research_engine'].analyze_literature = AsyncMock(return_value=Mock())
        engines['research_engine'].form_hypotheses = AsyncMock(return_value=[
            Mock(id="hyp_1"), Mock(id="hyp_2")
        ])
        
        # Mock experiment planner
        engines['experiment_planner'] = Mock()
        engines['experiment_planner'].is_ready = AsyncMock(return_value=True)
        engines['experiment_planner'].start = AsyncMock()
        engines['experiment_planner'].plan_experiment = AsyncMock(return_value=Mock(
            id="exp_1", status="pending"
        ))
        engines['experiment_planner'].generate_protocol = AsyncMock(return_value=Mock(
            id="protocol_1", quality_score=0.8
        ))
        engines['experiment_planner'].allocate_resources = AsyncMock(return_value=Mock())
        
        # Mock rapid prototyper
        engines['rapid_prototyper'] = Mock()
        engines['rapid_prototyper'].is_ready = AsyncMock(return_value=True)
        engines['rapid_prototyper'].start = AsyncMock()
        engines['rapid_prototyper'].create_rapid_prototype = AsyncMock(return_value=Mock(
            id="proto_1", validated=False
        ))
        
        # Mock validation framework
        engines['validation_framework'] = Mock()
        engines['validation_framework'].is_ready = AsyncMock(return_value=True)
        engines['validation_framework'].start = AsyncMock()
        engines['validation_framework'].validate_innovation = AsyncMock(return_value=Mock(
            is_valid=True, confidence=0.85
        ))
        
        # Mock knowledge synthesizer
        engines['knowledge_synthesizer'] = Mock()
        engines['knowledge_synthesizer'].is_ready = AsyncMock(return_value=True)
        engines['knowledge_synthesizer'].start = AsyncMock()
        engines['knowledge_synthesizer'].synthesize_knowledge = AsyncMock(return_value=Mock())
        
        # Mock innovation accelerator
        engines['innovation_accelerator'] = Mock()
        engines['innovation_accelerator'].is_ready = AsyncMock(return_value=True)
        engines['innovation_accelerator'].start = AsyncMock()
        engines['innovation_accelerator'].create_acceleration_plan = AsyncMock(return_value=Mock())
        engines['innovation_accelerator'].execute_acceleration = AsyncMock()
        
        # Mock quality controller
        engines['quality_controller'] = Mock()
        engines['quality_controller'].is_ready = AsyncMock(return_value=True)
        engines['quality_controller'].start = AsyncMock()
        engines['quality_controller'].assess_project_quality = AsyncMock(return_value=Mock(
            issues=[]
        ))
        
        # Mock error detector
        engines['error_detector'] = Mock()
        engines['error_detector'].is_ready = AsyncMock(return_value=True)
        engines['error_detector'].start = AsyncMock()
        engines['error_detector'].handle_system_error = AsyncMock()
        engines['error_detector'].correct_issue = AsyncMock()
        
        return engines
    
    @pytest.fixture
    def innovation_lab(self, lab_config, mock_engines):
        """Create innovation lab with mocked engines"""
        with patch.multiple(
            'scrollintel.core.autonomous_innovation_lab',
            AutomatedResearchEngine=Mock(return_value=mock_engines['research_engine']),
            ExperimentPlanner=Mock(return_value=mock_engines['experiment_planner']),
            RapidPrototyper=Mock(return_value=mock_engines['rapid_prototyper']),
            ValidationFramework=Mock(return_value=mock_engines['validation_framework']),
            KnowledgeSynthesisFramework=Mock(return_value=mock_engines['knowledge_synthesizer']),
            InnovationAccelerationSystem=Mock(return_value=mock_engines['innovation_accelerator']),
            QualityControlAutomation=Mock(return_value=mock_engines['quality_controller']),
            ErrorDetectionCorrection=Mock(return_value=mock_engines['error_detector'])
        ):
            lab = AutonomousInnovationLab(lab_config)
            yield lab
    
    @pytest.mark.asyncio
    async def test_lab_initialization(self, innovation_lab):
        """Test autonomous innovation lab initialization"""
        assert innovation_lab.config is not None
        assert innovation_lab.status == LabStatus.INITIALIZING
        assert innovation_lab.active_projects == {}
        assert innovation_lab.metrics is not None
        assert not innovation_lab.is_running
    
    @pytest.mark.asyncio
    async def test_lab_startup(self, innovation_lab):
        """Test autonomous innovation lab startup process"""
        # Start the lab
        success = await innovation_lab.start_lab()
        
        assert success is True
        assert innovation_lab.is_running is True
        assert innovation_lab.status == LabStatus.ACTIVE
        
        # Verify engines were started
        assert innovation_lab.research_engine.start.called
        assert innovation_lab.experiment_planner.start.called
        assert innovation_lab.rapid_prototyper.start.called
    
    @pytest.mark.asyncio
    async def test_system_readiness_validation(self, innovation_lab):
        """Test system readiness validation"""
        # Test successful validation
        ready = await innovation_lab._validate_system_readiness()
        assert ready is True
        
        # Test failed validation
        innovation_lab.research_engine.is_ready.return_value = False
        ready = await innovation_lab._validate_system_readiness()
        assert ready is False
    
    @pytest.mark.asyncio
    async def test_configuration_validation(self, innovation_lab):
        """Test lab configuration validation"""
        # Test valid configuration
        assert innovation_lab._validate_configuration() is True
        
        # Test invalid configuration
        innovation_lab.config.max_concurrent_projects = 0
        assert innovation_lab._validate_configuration() is False
        
        innovation_lab.config.max_concurrent_projects = 5
        innovation_lab.config.research_domains = []
        assert innovation_lab._validate_configuration() is False
    
    @pytest.mark.asyncio
    async def test_research_opportunity_generation(self, innovation_lab):
        """Test autonomous research opportunity generation"""
        await innovation_lab.start_lab()
        
        # Generate research opportunities
        await innovation_lab._generate_research_opportunities()
        
        # Verify research topics were generated
        assert innovation_lab.research_engine.generate_research_topics.called
        
        # Check if projects were created
        assert len(innovation_lab.active_projects) > 0
    
    @pytest.mark.asyncio
    async def test_innovation_project_creation(self, innovation_lab):
        """Test innovation project creation from research topics"""
        topic = Mock(title="Test Innovation Topic")
        domain = "ai"
        
        project = await innovation_lab._create_innovation_project(topic, domain)
        
        assert project is not None
        assert project.domain == domain
        assert project.status == "active"
        assert len(project.hypotheses) > 0
        assert len(project.experiment_plans) > 0
    
    @pytest.mark.asyncio
    async def test_experiment_execution(self, innovation_lab):
        """Test autonomous experiment execution"""
        # Create test project
        project = InnovationProject(
            id="test_project",
            title="Test Project",
            domain="ai",
            research_topic=Mock(),
            hypotheses=[Mock()],
            experiment_plans=[Mock(id="exp_1", status="pending")],
            status="active",
            created_at=datetime.now()
        )
        
        # Execute experiment
        await innovation_lab._execute_experiment(project, project.experiment_plans[0])
        
        # Verify experiment was processed
        assert project.experiment_plans[0].status == "completed"
        assert hasattr(project.experiment_plans[0], 'results')
    
    @pytest.mark.asyncio
    async def test_prototype_generation(self, innovation_lab):
        """Test autonomous prototype generation"""
        # Create project with successful experiments
        project = InnovationProject(
            id="test_project",
            title="Test Project",
            domain="ai",
            research_topic=Mock(),
            hypotheses=[Mock()],
            experiment_plans=[],
            status="active",
            created_at=datetime.now()
        )
        
        # Add successful experiment
        successful_exp = Mock(
            id="exp_1",
            analysis=Mock(success_probability=0.8, insights="Test insights"),
            prototype_generated=False
        )
        project.successful_experiments = [successful_exp]
        project.prototypes = []
        
        # Generate prototypes
        await innovation_lab._generate_prototypes(project)
        
        # Verify prototype was created
        assert len(project.prototypes) > 0
        assert successful_exp.prototype_generated is True
    
    @pytest.mark.asyncio
    async def test_innovation_validation(self, innovation_lab):
        """Test autonomous innovation validation"""
        # Create project with prototypes
        project = InnovationProject(
            id="test_project",
            title="Test Project",
            domain="ai",
            research_topic=Mock(),
            hypotheses=[Mock()],
            experiment_plans=[],
            status="active",
            created_at=datetime.now()
        )
        
        # Add prototype
        prototype = Mock(id="proto_1", validated=False)
        project.prototypes = [prototype]
        project.validated_innovations = []
        
        # Validate innovations
        await innovation_lab._validate_innovations(project)
        
        # Verify validation occurred
        assert prototype.validated is True
        assert hasattr(prototype, 'validation_result')
        assert len(project.validated_innovations) > 0
    
    @pytest.mark.asyncio
    async def test_knowledge_synthesis(self, innovation_lab):
        """Test autonomous knowledge synthesis and learning"""
        # Create projects with results
        project = InnovationProject(
            id="test_project",
            title="Test Project",
            domain="ai",
            research_topic=Mock(),
            hypotheses=[Mock()],
            experiment_plans=[],
            status="active",
            created_at=datetime.now()
        )
        
        # Add successful experiments with results
        successful_exp = Mock(results={"data": "test_data"})
        project.successful_experiments = [successful_exp]
        innovation_lab.active_projects["test_project"] = project
        
        # Synthesize knowledge
        await innovation_lab._synthesize_and_learn()
        
        # Verify knowledge synthesis was called
        assert innovation_lab.knowledge_synthesizer.synthesize_knowledge.called
    
    @pytest.mark.asyncio
    async def test_quality_control_cycle(self, innovation_lab):
        """Test autonomous quality control and error correction"""
        # Create project
        project = InnovationProject(
            id="test_project",
            title="Test Project",
            domain="ai",
            research_topic=Mock(),
            hypotheses=[Mock()],
            experiment_plans=[],
            status="active",
            created_at=datetime.now()
        )
        innovation_lab.active_projects["test_project"] = project
        
        # Run quality control cycle
        await innovation_lab._quality_control_cycle()
        
        # Verify quality assessment was performed
        assert innovation_lab.quality_controller.assess_project_quality.called
    
    @pytest.mark.asyncio
    async def test_lab_capability_validation(self, innovation_lab):
        """Test comprehensive lab capability validation"""
        await innovation_lab.start_lab()
        
        # Validate all domains
        validation_result = await innovation_lab.validate_lab_capability()
        
        assert validation_result["overall_success"] is True
        assert "domain_results" in validation_result
        assert len(validation_result["domain_results"]) == len(innovation_lab.config.research_domains)
        
        # Validate specific domain
        domain_result = await innovation_lab.validate_lab_capability("ai")
        assert "ai" in domain_result["domain_results"]
        assert domain_result["domain_results"]["ai"]["success"] is True
    
    @pytest.mark.asyncio
    async def test_domain_capability_validation(self, innovation_lab):
        """Test validation of capability in specific research domain"""
        domain = "ai"
        
        result = await innovation_lab._validate_domain_capability(domain)
        
        assert result["success"] is True
        assert result["domain"] == domain
        assert "capabilities" in result
        
        capabilities = result["capabilities"]
        assert capabilities["topic_generation"] is True
        assert capabilities["experiment_planning"] is True
        assert capabilities["prototype_generation"] is True
        assert capabilities["innovation_validation"] is True
        assert capabilities["knowledge_synthesis"] is True
    
    @pytest.mark.asyncio
    async def test_lab_status_reporting(self, innovation_lab):
        """Test lab status and metrics reporting"""
        await innovation_lab.start_lab()
        
        status = await innovation_lab.get_lab_status()
        
        assert "status" in status
        assert "is_running" in status
        assert "active_projects" in status
        assert "metrics" in status
        assert "research_domains" in status
    
    @pytest.mark.asyncio
    async def test_metrics_updating(self, innovation_lab):
        """Test autonomous metrics updating"""
        # Add test projects
        project1 = InnovationProject(
            id="proj1", title="Project 1", domain="ai",
            research_topic=Mock(), hypotheses=[Mock()],
            experiment_plans=[], status="active",
            created_at=datetime.now()
        )
        project1.validated_innovations = [Mock(), Mock()]
        
        project2 = InnovationProject(
            id="proj2", title="Project 2", domain="quantum",
            research_topic=Mock(), hypotheses=[Mock()],
            experiment_plans=[], status="completed",
            created_at=datetime.now()
        )
        project2.validated_innovations = [Mock()]
        
        innovation_lab.active_projects = {"proj1": project1, "proj2": project2}
        
        # Update metrics
        await innovation_lab._update_metrics()
        
        # Verify metrics were updated
        assert innovation_lab.metrics.active_projects == 1
        assert innovation_lab.metrics.total_innovations == 3
        assert innovation_lab.last_validation is not None
    
    @pytest.mark.asyncio
    async def test_continuous_innovation_loop(self, innovation_lab):
        """Test the continuous autonomous innovation loop"""
        await innovation_lab.start_lab()
        
        # Let the innovation loop run briefly
        await asyncio.sleep(0.1)
        
        # Stop the lab
        await innovation_lab.stop_lab()
        
        assert innovation_lab.is_running is False
        assert innovation_lab.status == LabStatus.STOPPED
    
    @pytest.mark.asyncio
    async def test_error_handling_in_innovation_loop(self, innovation_lab):
        """Test error handling in the innovation loop"""
        # Mock an error in research opportunity generation
        innovation_lab.research_engine.generate_research_topics.side_effect = Exception("Test error")
        
        await innovation_lab.start_lab()
        
        # Let the loop run and handle the error
        await asyncio.sleep(0.1)
        
        # Verify error was handled
        assert innovation_lab.error_detector.handle_system_error.called
        
        await innovation_lab.stop_lab()
    
    @pytest.mark.asyncio
    async def test_project_lifecycle_management(self, innovation_lab):
        """Test complete project lifecycle management"""
        await innovation_lab.start_lab()
        
        # Create and add a test project
        project = InnovationProject(
            id="lifecycle_test",
            title="Lifecycle Test Project",
            domain="ai",
            research_topic=Mock(),
            hypotheses=[Mock()],
            experiment_plans=[Mock(id="exp_1", status="pending")],
            status="active",
            created_at=datetime.now()
        )
        
        innovation_lab.active_projects["lifecycle_test"] = project
        
        # Process the project
        await innovation_lab._process_project(project)
        
        # Verify project was processed
        assert project.experiment_plans[0].status in ["completed", "failed"]
    
    @pytest.mark.asyncio
    async def test_innovation_acceleration(self, innovation_lab):
        """Test innovation acceleration for promising innovations"""
        # Create project with validated innovations
        project = InnovationProject(
            id="accel_test",
            title="Acceleration Test",
            domain="ai",
            research_topic=Mock(),
            hypotheses=[Mock()],
            experiment_plans=[],
            status="active",
            created_at=datetime.now()
        )
        
        project.validated_innovations = [Mock(id="innovation_1")]
        innovation_lab.active_projects["accel_test"] = project
        
        # Run acceleration
        await innovation_lab._validate_and_accelerate()
        
        # Verify acceleration was attempted
        assert innovation_lab.innovation_accelerator.create_acceleration_plan.called
        assert innovation_lab.innovation_accelerator.execute_acceleration.called
    
    @pytest.mark.asyncio
    async def test_lab_shutdown(self, innovation_lab):
        """Test proper lab shutdown"""
        await innovation_lab.start_lab()
        assert innovation_lab.is_running is True
        
        await innovation_lab.stop_lab()
        
        assert innovation_lab.is_running is False
        assert innovation_lab.status == LabStatus.STOPPED
        
        # Verify all engines were stopped
        assert innovation_lab.research_engine.stop.called
        assert innovation_lab.experiment_planner.stop.called
        assert innovation_lab.rapid_prototyper.stop.called

class TestLabConfiguration:
    """Test lab configuration functionality"""
    
    def test_default_configuration(self):
        """Test default lab configuration"""
        config = LabConfiguration()
        
        assert config.max_concurrent_projects == 10
        assert len(config.research_domains) > 0
        assert config.quality_threshold == 0.8
        assert config.continuous_learning is True
        assert "breakthrough_innovations" in config.innovation_targets
    
    def test_custom_configuration(self):
        """Test custom lab configuration"""
        config = LabConfiguration(
            max_concurrent_projects=5,
            research_domains=["ai", "quantum"],
            quality_threshold=0.9,
            innovation_targets={"innovations": 10},
            continuous_learning=False
        )
        
        assert config.max_concurrent_projects == 5
        assert config.research_domains == ["ai", "quantum"]
        assert config.quality_threshold == 0.9
        assert config.innovation_targets == {"innovations": 10}
        assert config.continuous_learning is False

class TestInnovationLabIntegration:
    """Test integration between lab components"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_innovation_pipeline(self, innovation_lab):
        """Test complete end-to-end innovation pipeline"""
        await innovation_lab.start_lab()
        
        # Simulate complete innovation pipeline
        # 1. Generate research opportunities
        await innovation_lab._generate_research_opportunities()
        assert len(innovation_lab.active_projects) > 0
        
        # 2. Process projects (experiments, prototypes, validation)
        for project in innovation_lab.active_projects.values():
            await innovation_lab._process_project(project)
        
        # 3. Knowledge synthesis
        await innovation_lab._synthesize_and_learn()
        
        # 4. Quality control
        await innovation_lab._quality_control_cycle()
        
        # 5. Update metrics
        await innovation_lab._update_metrics()
        
        # Verify pipeline completion
        assert innovation_lab.metrics.total_innovations >= 0
        assert innovation_lab.last_validation is not None
        
        await innovation_lab.stop_lab()
    
    @pytest.mark.asyncio
    async def test_multi_domain_research_coordination(self, innovation_lab):
        """Test coordination of research across multiple domains"""
        await innovation_lab.start_lab()
        
        # Generate opportunities in multiple domains
        for domain in innovation_lab.config.research_domains:
            topics = await innovation_lab.research_engine.generate_research_topics(domain)
            assert len(topics) > 0
        
        # Verify projects were created across domains
        await innovation_lab._generate_research_opportunities()
        
        domains_with_projects = set()
        for project in innovation_lab.active_projects.values():
            domains_with_projects.add(project.domain)
        
        # Should have projects in multiple domains
        assert len(domains_with_projects) > 1
        
        await innovation_lab.stop_lab()
    
    @pytest.mark.asyncio
    async def test_system_resilience_and_recovery(self, innovation_lab):
        """Test system resilience and error recovery"""
        await innovation_lab.start_lab()
        
        # Simulate various error conditions
        errors = [
            Exception("Research engine error"),
            Exception("Experiment planning error"),
            Exception("Prototype generation error")
        ]
        
        for error in errors:
            await innovation_lab.error_detector.handle_system_error(error)
        
        # Verify system continues operating
        assert innovation_lab.is_running is True
        assert innovation_lab.status == LabStatus.ACTIVE
        
        await innovation_lab.stop_lab()

if __name__ == "__main__":
    pytest.main([__file__])