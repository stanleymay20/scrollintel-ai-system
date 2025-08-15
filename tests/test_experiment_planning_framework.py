"""
Tests for experiment planning framework.
"""
import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from scrollintel.engines.experiment_planner import ExperimentPlanner
from scrollintel.engines.methodology_selector import MethodologySelector
from scrollintel.engines.timeline_planner import TimelinePlanner, TimelineConstraint
from scrollintel.models.experimental_design_models import (
    ExperimentPlan, ExperimentType, MethodologyType, ExperimentStatus,
    ValidationStudy, MethodologyRecommendation, ExperimentMilestone,
    ExperimentOptimization
)


class TestExperimentPlanner:
    """Test cases for ExperimentPlanner."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.planner = ExperimentPlanner()
    
    def test_plan_experiment_basic(self):
        """Test basic experiment planning."""
        research_question = "Does variable X affect outcome Y?"
        domain = "engineering"
        
        plan = self.planner.plan_experiment(
            research_question=research_question,
            domain=domain
        )
        
        assert isinstance(plan, ExperimentPlan)
        assert plan.research_question == research_question
        assert plan.experiment_type in ExperimentType
        assert plan.methodology in MethodologyType
        assert len(plan.hypotheses) > 0
        assert len(plan.variables) > 0
        assert len(plan.conditions) > 0
        assert plan.protocol is not None
        assert len(plan.resource_requirements) > 0
        assert len(plan.timeline) > 0
        assert len(plan.success_criteria) > 0
        assert plan.status == ExperimentStatus.PLANNED
    
    def test_plan_experiment_with_constraints(self):
        """Test experiment planning with constraints."""
        research_question = "How does temperature affect reaction rate?"
        domain = "chemistry"
        constraints = {
            "budget": 5000.0,
            "timeline": timedelta(weeks=4),
            "sample_size": 50
        }
        
        plan = self.planner.plan_experiment(
            research_question=research_question,
            domain=domain,
            constraints=constraints
        )
        
        assert isinstance(plan, ExperimentPlan)
        assert plan.estimated_completion is not None
        # Check that constraints influenced the plan
        total_cost = sum(r.cost_estimate or 0 for r in plan.resource_requirements)
        assert total_cost <= constraints["budget"] * 1.2  # Allow some flexibility
    
    def test_create_validation_study(self):
        """Test validation study creation."""
        # First create an experiment plan
        plan = self.planner.plan_experiment(
            research_question="Test question",
            domain="engineering"
        )
        
        validation_study = self.planner.create_validation_study(
            experiment_plan=plan
        )
        
        assert isinstance(validation_study, ValidationStudy)
        assert validation_study.experiment_plan_id == plan.plan_id
        assert len(validation_study.validation_methods) > 0
        assert len(validation_study.validation_criteria) > 0
        assert validation_study.status == ExperimentStatus.PLANNED
    
    def test_recommend_methodology(self):
        """Test methodology recommendation."""
        research_question = "What is the optimal algorithm performance?"
        domain = "computer_science"
        
        recommendations = self.planner.recommend_methodology(
            research_question=research_question,
            domain=domain
        )
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        
        for rec in recommendations:
            assert isinstance(rec, MethodologyRecommendation)
            assert rec.suitability_score > 0
            assert len(rec.advantages) > 0
            assert len(rec.disadvantages) > 0
            assert rec.estimated_duration > timedelta(0)
    
    def test_optimize_experiment_design(self):
        """Test experiment design optimization."""
        # Create a basic experiment plan
        plan = self.planner.plan_experiment(
            research_question="Test optimization",
            domain="engineering"
        )
        
        optimization_goals = ["cost", "time"]
        
        optimization = self.planner.optimize_experiment_design(
            experiment_plan=plan,
            optimization_goals=optimization_goals
        )
        
        assert isinstance(optimization, ExperimentOptimization)
        assert optimization.experiment_plan_id == plan.plan_id
        assert len(optimization.optimization_strategies) > 0
        assert len(optimization.current_metrics) > 0
        assert len(optimization.optimized_metrics) > 0
        assert 0 <= optimization.confidence_score <= 1


class TestMethodologySelector:
    """Test cases for MethodologySelector."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.selector = MethodologySelector()
    
    def test_select_optimal_methodology(self):
        """Test optimal methodology selection."""
        research_context = {
            "domain": "engineering",
            "complexity": "medium",
            "sample_size": 100,
            "objectives": ["hypothesis_testing"],
            "timeline": timedelta(weeks=8),
            "budget": 10000.0
        }
        
        recommendation = self.selector.select_optimal_methodology(
            research_context=research_context
        )
        
        assert isinstance(recommendation, MethodologyRecommendation)
        assert recommendation.methodology in MethodologyType
        assert recommendation.experiment_type in ExperimentType
        assert 0 <= recommendation.suitability_score <= 1
        assert len(recommendation.advantages) > 0
        assert len(recommendation.disadvantages) > 0
        assert recommendation.estimated_duration > timedelta(0)
    
    def test_compare_methodologies(self):
        """Test methodology comparison."""
        research_context = {
            "domain": "social_sciences",
            "complexity": "high",
            "sample_size": 50
        }
        
        methodologies = [
            MethodologyType.QUANTITATIVE,
            MethodologyType.QUALITATIVE,
            MethodologyType.MIXED_METHODS
        ]
        
        recommendations = self.selector.compare_methodologies(
            research_context=research_context,
            methodologies=methodologies
        )
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        
        # Should be sorted by suitability score
        scores = [r.suitability_score for r in recommendations]
        assert scores == sorted(scores, reverse=True)
    
    def test_optimize_methodology_selection(self):
        """Test methodology selection optimization."""
        research_context = {
            "domain": "computer_science",
            "complexity": "medium",
            "budget": 5000.0
        }
        
        optimization_criteria = ["cost", "time"]
        
        recommendation = self.selector.optimize_methodology_selection(
            research_context=research_context,
            optimization_criteria=optimization_criteria
        )
        
        assert isinstance(recommendation, MethodologyRecommendation)
        assert recommendation.suitability_score > 0
        # Should favor cost and time efficient methodologies
        assert recommendation.methodology in [
            MethodologyType.COMPUTATIONAL,
            MethodologyType.SIMULATION,
            MethodologyType.QUANTITATIVE
        ]
    
    def test_methodology_profiles_initialization(self):
        """Test that methodology profiles are properly initialized."""
        profiles = self.selector.methodology_profiles
        
        assert len(profiles) == len(MethodologyType)
        
        for methodology_type in MethodologyType:
            assert methodology_type in profiles
            profile = profiles[methodology_type]
            assert len(profile.strengths) > 0
            assert len(profile.weaknesses) > 0
            assert len(profile.suitable_domains) > 0
            assert profile.typical_duration > timedelta(0)
            assert 0 <= profile.accuracy_potential <= 1
            assert 0 <= profile.reliability_potential <= 1


class TestTimelinePlanner:
    """Test cases for TimelinePlanner."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.planner = TimelinePlanner()
    
    def test_create_experiment_timeline(self):
        """Test experiment timeline creation."""
        from scrollintel.models.experimental_design_models import (
            ExperimentalProtocol, ResourceRequirement
        )
        
        protocol = ExperimentalProtocol(
            protocol_id="test_protocol",
            title="Test Protocol",
            objective="Test objective",
            methodology=MethodologyType.QUANTITATIVE,
            procedures=["Step 1", "Step 2"],
            materials_required=["Material 1"],
            safety_considerations=["Safety 1"],
            quality_controls=["QC 1"],
            data_collection_methods=["Method 1"],
            analysis_plan="Test analysis",
            estimated_duration=timedelta(weeks=8)
        )
        
        resources = [
            ResourceRequirement(
                resource_type="personnel",
                resource_name="Researcher",
                quantity_needed=2,
                duration_needed=timedelta(weeks=6),
                cost_estimate=5000.0
            )
        ]
        
        milestones = self.planner.create_experiment_timeline(
            protocol=protocol,
            resources=resources,
            experiment_type=ExperimentType.CONTROLLED,
            methodology=MethodologyType.QUANTITATIVE
        )
        
        assert isinstance(milestones, list)
        assert len(milestones) > 0
        
        for milestone in milestones:
            assert isinstance(milestone, ExperimentMilestone)
            assert milestone.milestone_id
            assert milestone.name
            assert milestone.target_date > datetime.now()
    
    def test_optimize_timeline(self):
        """Test timeline optimization."""
        # Create sample milestones
        milestones = [
            ExperimentMilestone(
                milestone_id="m1",
                name="Planning Complete",
                description="Planning phase complete",
                target_date=datetime.now() + timedelta(weeks=2),
                deliverables=["Plan"],
                completion_criteria=["Approved"]
            ),
            ExperimentMilestone(
                milestone_id="m2",
                name="Data Collection Complete",
                description="Data collection complete",
                target_date=datetime.now() + timedelta(weeks=8),
                dependencies=["m1"],
                deliverables=["Dataset"],
                completion_criteria=["Quality validated"]
            )
        ]
        
        optimization_goals = ["minimize_duration"]
        
        optimized_milestones, optimization_result = self.planner.optimize_timeline(
            milestones=milestones,
            optimization_goals=optimization_goals
        )
        
        assert isinstance(optimized_milestones, list)
        assert len(optimized_milestones) == len(milestones)
        
        assert optimization_result.original_duration >= timedelta(0)
        assert optimization_result.optimized_duration >= timedelta(0)
        assert len(optimization_result.optimization_strategies) > 0
        assert 0 <= optimization_result.confidence_score <= 1
    
    def test_create_milestone_dependencies(self):
        """Test milestone dependency creation."""
        milestones = [
            ExperimentMilestone(
                milestone_id="m1",
                name="Planning Complete",
                description="Planning phase",
                target_date=datetime.now() + timedelta(weeks=1)
            ),
            ExperimentMilestone(
                milestone_id="m2",
                name="Data Collection Complete",
                description="Data collection phase",
                target_date=datetime.now() + timedelta(weeks=4)
            ),
            ExperimentMilestone(
                milestone_id="m3",
                name="Analysis Complete",
                description="Analysis phase",
                target_date=datetime.now() + timedelta(weeks=6)
            )
        ]
        
        updated_milestones = self.planner.create_milestone_dependencies(milestones)
        
        assert len(updated_milestones) == len(milestones)
        
        # Check that later milestones have dependencies
        analysis_milestone = next(m for m in updated_milestones if "Analysis" in m.name)
        assert len(analysis_milestone.dependencies) > 0
    
    def test_calculate_critical_path(self):
        """Test critical path calculation."""
        milestones = [
            ExperimentMilestone(
                milestone_id="m1",
                name="Planning Complete",
                description="Planning",
                target_date=datetime.now() + timedelta(weeks=1),
                dependencies=[]
            ),
            ExperimentMilestone(
                milestone_id="m2",
                name="Data Collection Complete",
                description="Data collection",
                target_date=datetime.now() + timedelta(weeks=4),
                dependencies=["m1"]
            ),
            ExperimentMilestone(
                milestone_id="m3",
                name="Analysis Complete",
                description="Analysis",
                target_date=datetime.now() + timedelta(weeks=6),
                dependencies=["m2"]
            )
        ]
        
        critical_path = self.planner.calculate_critical_path(milestones)
        
        assert isinstance(critical_path, list)
        assert len(critical_path) > 0
        
        # All milestone IDs should be valid
        milestone_ids = {m.milestone_id for m in milestones}
        for path_id in critical_path:
            assert path_id in milestone_ids
    
    def test_timeline_constraints(self):
        """Test timeline with constraints."""
        from scrollintel.models.experimental_design_models import (
            ExperimentalProtocol, ResourceRequirement
        )
        
        protocol = ExperimentalProtocol(
            protocol_id="test_protocol",
            title="Test Protocol",
            objective="Test objective",
            methodology=MethodologyType.QUANTITATIVE,
            procedures=["Step 1"],
            materials_required=["Material 1"],
            safety_considerations=["Safety 1"],
            quality_controls=["QC 1"],
            data_collection_methods=["Method 1"],
            analysis_plan="Test analysis",
            estimated_duration=timedelta(weeks=12)
        )
        
        resources = [
            ResourceRequirement(
                resource_type="personnel",
                resource_name="Researcher",
                quantity_needed=1,
                duration_needed=timedelta(weeks=10),
                cost_estimate=3000.0
            )
        ]
        
        constraints = [
            TimelineConstraint(
                constraint_type="deadline",
                description="Project deadline",
                end_date=datetime.now() + timedelta(weeks=8),
                priority="high"
            )
        ]
        
        milestones = self.planner.create_experiment_timeline(
            protocol=protocol,
            resources=resources,
            experiment_type=ExperimentType.CONTROLLED,
            methodology=MethodologyType.QUANTITATIVE,
            constraints=constraints
        )
        
        assert len(milestones) > 0
        
        # Check that timeline respects deadline constraint (with reasonable buffer)
        latest_milestone = max(milestones, key=lambda m: m.target_date)
        deadline = constraints[0].end_date
        
        # Allow reasonable buffer for compressed timeline
        buffer_days = max(14, int((deadline - datetime.now()).days * 0.2))
        assert latest_milestone.target_date <= deadline + timedelta(days=buffer_days)


class TestIntegration:
    """Integration tests for experiment planning framework."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.experiment_planner = ExperimentPlanner()
        self.methodology_selector = MethodologySelector()
        self.timeline_planner = TimelinePlanner()
    
    def test_full_experiment_planning_workflow(self):
        """Test complete experiment planning workflow."""
        # Step 1: Plan experiment
        research_question = "How does algorithm X compare to algorithm Y in terms of performance?"
        domain = "computer_science"
        constraints = {
            "budget": 8000.0,
            "timeline": timedelta(weeks=10)
        }
        
        experiment_plan = self.experiment_planner.plan_experiment(
            research_question=research_question,
            domain=domain,
            constraints=constraints
        )
        
        assert isinstance(experiment_plan, ExperimentPlan)
        
        # Step 2: Create validation study
        validation_study = self.experiment_planner.create_validation_study(
            experiment_plan=experiment_plan
        )
        
        assert isinstance(validation_study, ValidationStudy)
        assert validation_study.experiment_plan_id == experiment_plan.plan_id
        
        # Step 3: Optimize design
        optimization = self.experiment_planner.optimize_experiment_design(
            experiment_plan=experiment_plan,
            optimization_goals=["cost", "time"]
        )
        
        assert isinstance(optimization, ExperimentOptimization)
        assert optimization.confidence_score > 0
        
        # Step 4: Optimize timeline
        optimized_timeline, timeline_optimization = self.timeline_planner.optimize_timeline(
            milestones=experiment_plan.timeline,
            optimization_goals=["minimize_duration"]
        )
        
        assert len(optimized_timeline) == len(experiment_plan.timeline)
        assert timeline_optimization.confidence_score > 0
    
    def test_methodology_selection_integration(self):
        """Test methodology selection with different domains."""
        test_domains = ["engineering", "social_sciences", "computer_science"]
        
        for domain in test_domains:
            research_context = {
                "domain": domain,
                "complexity": "medium",
                "sample_size": 100
            }
            
            recommendation = self.methodology_selector.select_optimal_methodology(
                research_context=research_context
            )
            
            # Check that we get a valid methodology recommendation
            assert isinstance(recommendation, MethodologyRecommendation)
            assert recommendation.methodology in MethodologyType
            assert recommendation.suitability_score > 0.3  # Should be reasonably suitable
            assert len(recommendation.advantages) > 0
            assert len(recommendation.disadvantages) > 0
    
    def test_error_handling(self):
        """Test error handling in experiment planning."""
        # Test with empty research question
        with pytest.raises(Exception):
            self.experiment_planner.plan_experiment(
                research_question="",
                domain="engineering"
            )
        
        # Test with impossible constraints - should handle gracefully
        impossible_constraints = {
            "budget": 0.01,  # Impossibly low budget
            "timeline": timedelta(hours=1)  # Impossibly short timeline
        }
        
        # Should handle gracefully
        plan = self.experiment_planner.plan_experiment(
            research_question="Test question",
            domain="engineering",
            constraints=impossible_constraints
        )
        # Should create a minimal viable plan
        assert isinstance(plan, ExperimentPlan)
        assert plan.research_question == "Test question"


if __name__ == "__main__":
    pytest.main([__file__])