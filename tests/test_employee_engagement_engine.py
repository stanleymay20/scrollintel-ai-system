"""
Tests for Employee Engagement Engine
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from scrollintel.engines.employee_engagement_engine import EmployeeEngagementEngine
from scrollintel.models.employee_engagement_models import (
    Employee, EngagementActivity, EngagementStrategy, EngagementMetric,
    EngagementAssessment, EngagementPlan, EngagementFeedback, EngagementReport,
    EngagementImprovementPlan, EngagementLevel, EngagementActivityType,
    EngagementMetricType
)


class TestEmployeeEngagementEngine:
    
    def setup_method(self):
        """Set up test fixtures"""
        self.engine = EmployeeEngagementEngine()
        self.organization_id = "test_org_123"
        self.target_groups = ["engineering", "sales", "marketing"]
        self.cultural_objectives = [
            "Increase collaboration",
            "Improve communication",
            "Enhance innovation"
        ]
        self.current_engagement_data = {
            "engagement_levels": {
                "engineering": 0.75,
                "sales": 0.55,
                "marketing": 0.68
            },
            "communication_scores": {
                "internal": 0.65,
                "management": 0.58
            }
        }
        
        # Sample employee profiles
        self.employee_profiles = [
            Employee(
                id="emp_001",
                name="John Doe",
                department="engineering",
                role="senior_developer",
                manager_id="mgr_001",
                hire_date=datetime.now() - timedelta(days=365),
                engagement_level=EngagementLevel.ENGAGED,
                cultural_alignment_score=0.8
            ),
            Employee(
                id="emp_002",
                name="Jane Smith",
                department="sales",
                role="account_manager",
                manager_id="mgr_002",
                hire_date=datetime.now() - timedelta(days=180),
                engagement_level=EngagementLevel.SOMEWHAT_ENGAGED,
                cultural_alignment_score=0.6
            ),
            Employee(
                id="emp_003",
                name="Bob Johnson",
                department="marketing",
                role="marketing_specialist",
                manager_id="mgr_003",
                hire_date=datetime.now() - timedelta(days=90),
                engagement_level=EngagementLevel.DISENGAGED,
                cultural_alignment_score=0.4
            )
        ]
    
    def test_develop_engagement_strategy(self):
        """Test engagement strategy development"""
        strategy = self.engine.develop_engagement_strategy(
            organization_id=self.organization_id,
            target_groups=self.target_groups,
            cultural_objectives=self.cultural_objectives,
            current_engagement_data=self.current_engagement_data
        )
        
        assert isinstance(strategy, EngagementStrategy)
        assert strategy.organization_id == self.organization_id
        assert strategy.target_groups == self.target_groups
        assert len(strategy.objectives) >= len(self.cultural_objectives)
        assert strategy.budget_allocated > 0
        assert strategy.status == "draft"
        assert strategy.id in self.engine.engagement_strategies
    
    def test_design_engagement_activities(self):
        """Test engagement activity design"""
        # First create a strategy
        strategy = self.engine.develop_engagement_strategy(
            organization_id=self.organization_id,
            target_groups=self.target_groups,
            cultural_objectives=self.cultural_objectives,
            current_engagement_data=self.current_engagement_data
        )
        
        activities = self.engine.design_engagement_activities(
            strategy=strategy,
            employee_profiles=self.employee_profiles
        )
        
        assert isinstance(activities, list)
        assert len(activities) > 0
        
        for activity in activities:
            assert isinstance(activity, EngagementActivity)
            assert activity.id in self.engine.engagement_activities
            assert len(activity.objectives) > 0
            assert activity.duration_minutes > 0
            assert len(activity.cultural_values_addressed) > 0
    
    def test_execute_engagement_activity(self):
        """Test engagement activity execution"""
        # Create strategy and activities
        strategy = self.engine.develop_engagement_strategy(
            organization_id=self.organization_id,
            target_groups=self.target_groups,
            cultural_objectives=self.cultural_objectives,
            current_engagement_data=self.current_engagement_data
        )
        
        activities = self.engine.design_engagement_activities(
            strategy=strategy,
            employee_profiles=self.employee_profiles
        )
        
        # Execute first activity
        activity = activities[0]
        participants = ["emp_001", "emp_002"]
        execution_context = {"send_pre_survey": True}
        
        execution_report = self.engine.execute_engagement_activity(
            activity_id=activity.id,
            participants=participants,
            execution_context=execution_context
        )
        
        assert isinstance(execution_report, dict)
        assert execution_report["activity_id"] == activity.id
        assert execution_report["participants"] == participants
        assert "execution_date" in execution_report
        assert "preparation_results" in execution_report
        assert "execution_results" in execution_report
        assert "followup_results" in execution_report
        assert "success_indicators" in execution_report
        
        # Check that activity status was updated
        updated_activity = self.engine.engagement_activities[activity.id]
        assert updated_activity.status == "completed"
        assert updated_activity.completion_date is not None
    
    def test_measure_engagement_effectiveness(self):
        """Test engagement effectiveness measurement"""
        measurement_period = {
            "start": datetime.now() - timedelta(days=30),
            "end": datetime.now()
        }
        
        report = self.engine.measure_engagement_effectiveness(
            organization_id=self.organization_id,
            measurement_period=measurement_period
        )
        
        assert isinstance(report, EngagementReport)
        assert report.organization_id == self.organization_id
        assert report.report_period == measurement_period
        assert isinstance(report.overall_engagement_score, float)
        assert isinstance(report.engagement_by_department, dict)
        assert isinstance(report.engagement_trends, dict)
        assert isinstance(report.activity_effectiveness, dict)
        assert isinstance(report.key_insights, list)
        assert isinstance(report.recommendations, list)
        assert len(report.metrics) > 0
        assert report.generated_date is not None
    
    def test_create_improvement_plan(self):
        """Test improvement plan creation"""
        # Create a mock engagement report
        report = EngagementReport(
            id="test_report",
            organization_id=self.organization_id,
            report_period={
                "start": datetime.now() - timedelta(days=30),
                "end": datetime.now()
            },
            overall_engagement_score=3.2,
            engagement_by_department={"engineering": 3.5, "sales": 2.8},
            engagement_trends={"overall": [3.0, 3.1, 3.1, 3.2]},
            activity_effectiveness={"workshops": 0.75},
            key_insights=["Engagement below target", "Sales needs attention"],
            recommendations=["Increase feedback", "Improve recognition"],
            metrics=[],
            generated_date=datetime.now(),
            generated_by="test"
        )
        
        improvement_goals = [
            "Increase overall engagement to 4.0",
            "Improve sales team engagement",
            "Enhance communication effectiveness"
        ]
        
        improvement_plan = self.engine.create_improvement_plan(
            engagement_report=report,
            improvement_goals=improvement_goals
        )
        
        assert isinstance(improvement_plan, EngagementImprovementPlan)
        assert improvement_plan.organization_id == self.organization_id
        assert isinstance(improvement_plan.current_state, dict)
        assert isinstance(improvement_plan.target_state, dict)
        assert len(improvement_plan.improvement_strategies) > 0
        assert isinstance(improvement_plan.implementation_timeline, dict)
        assert isinstance(improvement_plan.resource_requirements, dict)
        assert len(improvement_plan.success_criteria) > 0
        assert len(improvement_plan.risk_mitigation) > 0
        assert isinstance(improvement_plan.monitoring_plan, dict)
    
    def test_process_engagement_feedback(self):
        """Test engagement feedback processing"""
        feedback = EngagementFeedback(
            id="test_feedback",
            employee_id="emp_001",
            activity_id="activity_001",
            feedback_type="activity",
            rating=4.0,
            comments="Great workshop! I learned a lot about communication skills.",
            suggestions=["More hands-on exercises", "Longer duration"],
            sentiment="neutral",
            themes=[],
            submitted_date=datetime.now(),
            processed=False
        )
        
        processing_results = self.engine.process_engagement_feedback(feedback)
        
        assert isinstance(processing_results, dict)
        assert processing_results["feedback_id"] == feedback.id
        assert "sentiment_analysis" in processing_results
        assert "themes" in processing_results
        assert "insights" in processing_results
        assert "response_recommendations" in processing_results
        assert "processed_date" in processing_results
        
        # Check that feedback was updated
        assert feedback.processed is True
        assert feedback.sentiment != "neutral"  # Should be analyzed
        assert len(feedback.themes) > 0
        assert feedback.id in self.engine.feedback_data
    
    def test_analyze_engagement_gaps(self):
        """Test engagement gap analysis"""
        gaps = self.engine._analyze_engagement_gaps(self.current_engagement_data)
        
        assert isinstance(gaps, dict)
        assert "low_engagement_areas" in gaps
        assert "communication_gaps" in gaps
        assert "participation_gaps" in gaps
        assert "satisfaction_gaps" in gaps
        
        # Sales should be identified as low engagement (0.55 < 0.6)
        assert "sales" in gaps["low_engagement_areas"]
        
        # Management communication should be identified as gap (0.58 < 0.7)
        assert "management" in gaps["communication_gaps"]
    
    def test_identify_audience_needs(self):
        """Test audience needs identification"""
        engagement_gaps = {
            "low_engagement_areas": ["sales"],
            "communication_gaps": ["management"],
            "participation_gaps": [],
            "satisfaction_gaps": []
        }
        
        audience_needs = self.engine._identify_audience_needs(
            self.target_groups, engagement_gaps
        )
        
        assert isinstance(audience_needs, dict)
        assert len(audience_needs) == len(self.target_groups)
        
        for group in self.target_groups:
            assert group in audience_needs
            assert isinstance(audience_needs[group], list)
            assert len(audience_needs[group]) > 0
        
        # Sales should have additional needs due to low engagement
        assert len(audience_needs["sales"]) >= 3
    
    def test_design_activities_for_different_levels(self):
        """Test activity design for different engagement levels"""
        strategy = EngagementStrategy(
            id="test_strategy",
            organization_id=self.organization_id,
            name="Test Strategy",
            description="Test",
            target_groups=self.target_groups,
            objectives=self.cultural_objectives,
            activities=[],
            timeline={},
            success_metrics=[],
            budget_allocated=10000.0,
            owner="test",
            created_date=datetime.now()
        )
        
        # Test for each engagement level
        for level in EngagementLevel:
            level_employees = [
                emp for emp in self.employee_profiles 
                if emp.engagement_level == level
            ]
            
            if level_employees:
                activities = self.engine._design_activities_for_level(
                    strategy, level, level_employees
                )
                
                assert isinstance(activities, list)
                if level == EngagementLevel.DISENGAGED:
                    # Should have re-engagement activities
                    assert any("feedback" in act.name.lower() for act in activities)
                elif level == EngagementLevel.HIGHLY_ENGAGED:
                    # Should have leadership activities
                    assert any("leadership" in act.name.lower() or "mentoring" in act.name.lower() for act in activities)
    
    def test_create_personalized_activities(self):
        """Test personalized activity creation"""
        strategy = EngagementStrategy(
            id="test_strategy",
            organization_id=self.organization_id,
            name="Test Strategy",
            description="Test",
            target_groups=self.target_groups,
            objectives=self.cultural_objectives,
            activities=[],
            timeline={},
            success_metrics=[],
            budget_allocated=10000.0,
            owner="test",
            created_date=datetime.now()
        )
        
        activities = self.engine._create_personalized_activities(
            strategy, self.employee_profiles
        )
        
        assert isinstance(activities, list)
        # Should create activities based on employee needs
        for activity in activities:
            assert isinstance(activity, EngagementActivity)
            assert len(activity.objectives) > 0
            assert len(activity.cultural_values_addressed) > 0
    
    def test_optimize_activity_sequence(self):
        """Test activity sequence optimization"""
        # Create sample activities with different types
        activities = [
            EngagementActivity(
                id="act_1",
                name="Team Building",
                activity_type=EngagementActivityType.TEAM_BUILDING,
                description="Test",
                target_audience=["all"],
                objectives=["test"],
                duration_minutes=120,
                facilitator="test",
                materials_needed=[],
                success_criteria=[],
                cultural_values_addressed=[],
                created_date=datetime.now()
            ),
            EngagementActivity(
                id="act_2",
                name="Feedback Session",
                activity_type=EngagementActivityType.FEEDBACK_SESSION,
                description="Test",
                target_audience=["all"],
                objectives=["test"],
                duration_minutes=60,
                facilitator="test",
                materials_needed=[],
                success_criteria=[],
                cultural_values_addressed=[],
                created_date=datetime.now()
            ),
            EngagementActivity(
                id="act_3",
                name="Workshop",
                activity_type=EngagementActivityType.WORKSHOP,
                description="Test",
                target_audience=["all"],
                objectives=["test"],
                duration_minutes=180,
                facilitator="test",
                materials_needed=[],
                success_criteria=[],
                cultural_values_addressed=[],
                created_date=datetime.now()
            )
        ]
        
        optimized = self.engine._optimize_activity_sequence(activities)
        
        assert len(optimized) == len(activities)
        # Feedback sessions should come first (priority 1)
        assert optimized[0].activity_type == EngagementActivityType.FEEDBACK_SESSION
        # Workshop should come before team building (priority 3 vs 5)
        feedback_idx = next(i for i, act in enumerate(optimized) if act.activity_type == EngagementActivityType.FEEDBACK_SESSION)
        workshop_idx = next(i for i, act in enumerate(optimized) if act.activity_type == EngagementActivityType.WORKSHOP)
        team_idx = next(i for i, act in enumerate(optimized) if act.activity_type == EngagementActivityType.TEAM_BUILDING)
        
        assert feedback_idx < workshop_idx < team_idx
    
    def test_calculate_engagement_metrics(self):
        """Test engagement metrics calculation"""
        data = {
            "engagement_scores": {
                "overall": 3.8,
                "by_department": {
                    "engineering": 4.1,
                    "sales": 3.5
                }
            },
            "participation_rates": {
                "activities": 0.75,
                "surveys": 0.80
            },
            "satisfaction_scores": {
                "job_satisfaction": 3.7,
                "culture_fit": 3.9
            },
            "retention_rate": 0.92
        }
        
        metrics = self.engine._calculate_engagement_metrics(data)
        
        assert isinstance(metrics, dict)
        assert metrics["overall_score"] == 3.8
        assert metrics["by_department"]["engineering"] == 4.1
        assert metrics["by_department"]["sales"] == 3.5
        assert 0.7 <= metrics["avg_participation"] <= 0.8
        assert 3.7 <= metrics["avg_satisfaction"] <= 3.9
        assert metrics["retention_rate"] == 0.92
    
    def test_analyze_feedback_sentiment(self):
        """Test feedback sentiment analysis"""
        # Positive feedback
        positive_feedback = EngagementFeedback(
            id="pos_feedback",
            employee_id="emp_001",
            feedback_type="general",
            comments="I love working here! The team is great and I enjoy my projects.",
            suggestions=[],
            sentiment="neutral",
            themes=[],
            submitted_date=datetime.now()
        )
        
        sentiment = self.engine._analyze_feedback_sentiment(positive_feedback)
        
        assert sentiment["sentiment"] == "positive"
        assert sentiment["positive_indicators"] > 0
        assert sentiment["confidence"] > 0
        
        # Negative feedback
        negative_feedback = EngagementFeedback(
            id="neg_feedback",
            employee_id="emp_002",
            feedback_type="general",
            comments="I hate the poor communication and bad management decisions.",
            suggestions=[],
            sentiment="neutral",
            themes=[],
            submitted_date=datetime.now()
        )
        
        sentiment = self.engine._analyze_feedback_sentiment(negative_feedback)
        
        assert sentiment["sentiment"] == "negative"
        assert sentiment["negative_indicators"] > 0
    
    def test_extract_feedback_themes(self):
        """Test feedback theme extraction"""
        feedback = EngagementFeedback(
            id="theme_feedback",
            employee_id="emp_001",
            feedback_type="general",
            comments="Need better communication from management and more recognition for good work. Also want development opportunities.",
            suggestions=[],
            sentiment="neutral",
            themes=[],
            submitted_date=datetime.now()
        )
        
        themes = self.engine._extract_feedback_themes(feedback)
        
        assert isinstance(themes, list)
        assert "communication" in themes
        assert "recognition" in themes
        assert "development" in themes
        assert "management" in themes
    
    def test_generate_improvement_recommendations(self):
        """Test improvement recommendation generation"""
        insights = [
            "Engagement is below target and needs attention",
            "Sales department shows declining engagement",
            "Recognition events are most effective activities",
            "Communication needs improvement"
        ]
        
        recommendations = self.engine._generate_improvement_recommendations(insights)
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        
        # Should include specific recommendations based on insights
        recommendation_text = " ".join(recommendations).lower()
        assert "engagement" in recommendation_text
        assert any("recognition" in rec.lower() for rec in recommendations)
    
    def test_error_handling(self):
        """Test error handling in various scenarios"""
        # Test with invalid organization ID
        with pytest.raises(Exception):
            self.engine.develop_engagement_strategy(
                organization_id="",
                target_groups=[],
                cultural_objectives=[],
                current_engagement_data={}
            )
        
        # Test executing non-existent activity
        with pytest.raises(ValueError, match="Activity not found"):
            self.engine.execute_engagement_activity(
                activity_id="non_existent",
                participants=["emp_001"],
                execution_context={}
            )
    
    def test_integration_workflow(self):
        """Test complete integration workflow"""
        # 1. Develop strategy
        strategy = self.engine.develop_engagement_strategy(
            organization_id=self.organization_id,
            target_groups=self.target_groups,
            cultural_objectives=self.cultural_objectives,
            current_engagement_data=self.current_engagement_data
        )
        
        # 2. Design activities
        activities = self.engine.design_engagement_activities(
            strategy=strategy,
            employee_profiles=self.employee_profiles
        )
        
        # 3. Execute an activity
        if activities:
            execution_report = self.engine.execute_engagement_activity(
                activity_id=activities[0].id,
                participants=["emp_001", "emp_002"],
                execution_context={}
            )
            assert execution_report["activity_id"] == activities[0].id
        
        # 4. Process feedback
        feedback = EngagementFeedback(
            id="integration_feedback",
            employee_id="emp_001",
            feedback_type="activity",
            comments="Great session, learned a lot!",
            suggestions=["More interactive elements"],
            sentiment="neutral",
            themes=[],
            submitted_date=datetime.now()
        )
        
        processing_results = self.engine.process_engagement_feedback(feedback)
        assert processing_results["feedback_id"] == feedback.id
        
        # 5. Measure effectiveness
        measurement_period = {
            "start": datetime.now() - timedelta(days=30),
            "end": datetime.now()
        }
        
        report = self.engine.measure_engagement_effectiveness(
            organization_id=self.organization_id,
            measurement_period=measurement_period
        )
        
        # 6. Create improvement plan
        improvement_plan = self.engine.create_improvement_plan(
            engagement_report=report,
            improvement_goals=["Increase engagement", "Improve communication"]
        )
        
        assert improvement_plan.organization_id == self.organization_id
        assert len(improvement_plan.improvement_strategies) > 0
        
        # Verify all components are working together
        assert len(self.engine.engagement_strategies) >= 1
        assert len(self.engine.engagement_activities) >= 1
        assert len(self.engine.feedback_data) >= 1