"""
Tests for Habit Formation Engine

Tests organizational habit design, formation strategies,
and sustainability mechanisms.
"""

import pytest
from datetime import datetime
from uuid import uuid4

from scrollintel.engines.habit_formation_engine import HabitFormationEngine
from scrollintel.models.habit_formation_models import (
    HabitType, HabitFrequency, HabitStage, SustainabilityLevel
)


class TestHabitFormationEngine:
    """Test suite for HabitFormationEngine"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.engine = HabitFormationEngine()
        self.habit_name = "Daily Team Standup"
        self.habit_description = "Daily 15-minute team synchronization meeting"
        self.habit_type = HabitType.COMMUNICATION
        self.target_behavior = "improve team communication and coordination"
        self.participants = ["alice", "bob", "charlie", "david", "eve"]
        self.cultural_values = ["transparency", "collaboration", "accountability"]
        self.business_objectives = ["improve team productivity", "reduce miscommunication"]
    
    def test_design_organizational_habit(self):
        """Test organizational habit design and identification"""
        habit = self.engine.design_organizational_habit(
            name=self.habit_name,
            description=self.habit_description,
            habit_type=self.habit_type,
            target_behavior=self.target_behavior,
            participants=self.participants,
            cultural_values=self.cultural_values,
            business_objectives=self.business_objectives
        )
        
        # Verify habit structure
        assert habit.id is not None
        assert habit.name == self.habit_name
        assert habit.description == self.habit_description
        assert habit.habit_type == self.habit_type
        assert habit.target_behavior == self.target_behavior
        assert habit.participants == self.participants
        assert habit.stage == HabitStage.DESIGN
        assert habit.created_by == "ScrollIntel"
        
        # Verify generated components
        assert isinstance(habit.trigger_conditions, list)
        assert len(habit.trigger_conditions) > 0
        assert isinstance(habit.execution_steps, list)
        assert len(habit.execution_steps) > 0
        assert isinstance(habit.success_indicators, list)
        assert len(habit.success_indicators) > 0
        assert isinstance(habit.frequency, HabitFrequency)
        assert habit.duration_minutes > 0
        assert isinstance(habit.facilitators, list)
        assert isinstance(habit.resources_required, list)
        
        # Verify scores
        assert 0.0 <= habit.cultural_alignment <= 1.0
        assert isinstance(habit.business_impact, str)
        assert len(habit.business_impact) > 0
    
    def test_generate_trigger_conditions(self):
        """Test trigger condition generation for different habit types"""
        # Test communication habit
        comm_triggers = self.engine._generate_trigger_conditions(
            HabitType.COMMUNICATION, "improve team communication"
        )
        assert len(comm_triggers) > 0
        assert any("meeting" in trigger.lower() for trigger in comm_triggers)
        
        # Test collaboration habit
        collab_triggers = self.engine._generate_trigger_conditions(
            HabitType.COLLABORATION, "enhance team collaboration"
        )
        assert len(collab_triggers) > 0
        assert any("project" in trigger.lower() or "team" in trigger.lower() for trigger in collab_triggers)
        
        # Test learning habit
        learning_triggers = self.engine._generate_trigger_conditions(
            HabitType.LEARNING, "continuous learning"
        )
        assert len(learning_triggers) > 0
        assert any("learning" in trigger.lower() or "skill" in trigger.lower() for trigger in learning_triggers)
    
    def test_generate_execution_steps(self):
        """Test execution step generation for different habit types"""
        # Test communication habit
        comm_steps = self.engine._generate_execution_steps(
            HabitType.COMMUNICATION, "improve communication"
        )
        assert len(comm_steps) > 0
        assert any("communication" in step.lower() for step in comm_steps)
        
        # Test innovation habit
        innovation_steps = self.engine._generate_execution_steps(
            HabitType.INNOVATION, "foster innovation"
        )
        assert len(innovation_steps) > 0
        assert any("idea" in step.lower() or "innovation" in step.lower() for step in innovation_steps)
    
    def test_determine_optimal_frequency(self):
        """Test optimal frequency determination"""
        # Test daily habits
        daily_frequency = self.engine._determine_optimal_frequency(
            HabitType.COMMUNICATION, "daily standup"
        )
        assert daily_frequency == HabitFrequency.DAILY
        
        # Test weekly habits
        weekly_frequency = self.engine._determine_optimal_frequency(
            HabitType.COLLABORATION, "weekly collaboration session"
        )
        assert weekly_frequency == HabitFrequency.WEEKLY
        
        # Test behavior-specific frequency
        behavior_daily = self.engine._determine_optimal_frequency(
            HabitType.LEARNING, "daily learning habit"
        )
        assert behavior_daily == HabitFrequency.DAILY
    
    def test_estimate_habit_duration(self):
        """Test habit duration estimation"""
        # Test small team
        small_duration = self.engine._estimate_habit_duration(
            HabitType.COMMUNICATION, 5
        )
        assert 5 <= small_duration <= 120
        
        # Test large team
        large_duration = self.engine._estimate_habit_duration(
            HabitType.COMMUNICATION, 25
        )
        assert large_duration >= small_duration  # Larger teams need more time
        assert 5 <= large_duration <= 120
    
    def test_calculate_cultural_alignment(self):
        """Test cultural alignment calculation"""
        # Test high alignment
        high_alignment = self.engine._calculate_cultural_alignment(
            HabitType.COMMUNICATION, "improve transparency", ["transparency", "openness"]
        )
        assert high_alignment > 0.7
        
        # Test moderate alignment
        moderate_alignment = self.engine._calculate_cultural_alignment(
            HabitType.QUALITY, "improve quality", ["excellence", "precision"]
        )
        assert 0.5 <= moderate_alignment <= 1.0
        
        # Test no cultural values
        default_alignment = self.engine._calculate_cultural_alignment(
            HabitType.COMMUNICATION, "improve communication", []
        )
        assert default_alignment == 0.7  # Default moderate alignment
    
    def test_create_habit_formation_strategy(self):
        """Test habit formation strategy creation"""
        habit = self.engine.design_organizational_habit(
            name=self.habit_name,
            description=self.habit_description,
            habit_type=self.habit_type,
            target_behavior=self.target_behavior,
            participants=self.participants
        )
        
        organizational_context = {
            'change_readiness': 'high',
            'resource_availability': 'medium',
            'leadership_support': 'strong'
        }
        
        strategy = self.engine.create_habit_formation_strategy(
            habit=habit,
            organizational_context=organizational_context
        )
        
        # Verify strategy structure
        assert strategy.id is not None
        assert strategy.habit_id == habit.id
        assert isinstance(strategy.formation_phases, list)
        assert len(strategy.formation_phases) > 0
        assert strategy.timeline_weeks > 0
        assert isinstance(strategy.key_milestones, list)
        assert len(strategy.key_milestones) > 0
        assert isinstance(strategy.success_metrics, list)
        assert len(strategy.success_metrics) > 0
        assert isinstance(strategy.reinforcement_mechanisms, list)
        assert isinstance(strategy.barrier_mitigation, list)
        assert isinstance(strategy.stakeholder_engagement, dict)
        assert isinstance(strategy.resource_allocation, dict)
        assert isinstance(strategy.risk_assessment, dict)
    
    def test_estimate_formation_timeline(self):
        """Test formation timeline estimation"""
        habit = self.engine.design_organizational_habit(
            name=self.habit_name,
            description=self.habit_description,
            habit_type=self.habit_type,
            target_behavior=self.target_behavior,
            participants=self.participants
        )
        
        # Test high readiness
        high_readiness_context = {'change_readiness': 'high'}
        high_timeline = self.engine._estimate_formation_timeline(habit, high_readiness_context)
        
        # Test low readiness
        low_readiness_context = {'change_readiness': 'low'}
        low_timeline = self.engine._estimate_formation_timeline(habit, low_readiness_context)
        
        # Low readiness should take longer
        assert low_timeline >= high_timeline
        assert 6 <= high_timeline <= 52
        assert 6 <= low_timeline <= 52
    
    def test_implement_habit_sustainability_mechanisms(self):
        """Test habit sustainability mechanism implementation"""
        habit = self.engine.design_organizational_habit(
            name=self.habit_name,
            description=self.habit_description,
            habit_type=self.habit_type,
            target_behavior=self.target_behavior,
            participants=self.participants
        )
        
        formation_progress = {
            'participation_rate': 0.8,
            'consistency_rate': 0.75,
            'weeks_since_formation': 12
        }
        
        sustainability = self.engine.implement_habit_sustainability_mechanisms(
            habit=habit,
            formation_progress=formation_progress
        )
        
        # Verify sustainability structure
        assert sustainability.id is not None
        assert sustainability.habit_id == habit.id
        assert isinstance(sustainability.sustainability_level, SustainabilityLevel)
        assert isinstance(sustainability.reinforcement_systems, list)
        assert len(sustainability.reinforcement_systems) > 0
        assert isinstance(sustainability.monitoring_mechanisms, list)
        assert len(sustainability.monitoring_mechanisms) > 0
        assert isinstance(sustainability.feedback_loops, list)
        assert len(sustainability.feedback_loops) > 0
        assert isinstance(sustainability.adaptation_triggers, list)
        assert isinstance(sustainability.renewal_strategies, list)
        assert isinstance(sustainability.institutional_support, list)
        assert 0.0 <= sustainability.cultural_integration <= 1.0
        assert isinstance(sustainability.resilience_factors, list)
        assert isinstance(sustainability.vulnerability_points, list)
        assert isinstance(sustainability.mitigation_plans, list)
        assert 0.0 <= sustainability.sustainability_score <= 1.0
        assert sustainability.next_review > sustainability.last_assessment
    
    def test_assess_sustainability_level(self):
        """Test sustainability level assessment"""
        habit = self.engine.design_organizational_habit(
            name=self.habit_name,
            description=self.habit_description,
            habit_type=self.habit_type,
            target_behavior=self.target_behavior,
            participants=self.participants
        )
        
        # Test fragile sustainability (early stage, low performance)
        fragile_progress = {
            'participation_rate': 0.5,
            'consistency_rate': 0.4,
            'weeks_since_formation': 2
        }
        fragile_level = self.engine._assess_sustainability_level(habit, fragile_progress)
        assert fragile_level == SustainabilityLevel.FRAGILE
        
        # Test stable sustainability (mature, good performance)
        stable_progress = {
            'participation_rate': 0.85,
            'consistency_rate': 0.8,
            'weeks_since_formation': 20
        }
        stable_level = self.engine._assess_sustainability_level(habit, stable_progress)
        assert stable_level in [SustainabilityLevel.STABLE, SustainabilityLevel.ROBUST]
    
    def test_track_habit_progress(self):
        """Test habit progress tracking"""
        habit = self.engine.design_organizational_habit(
            name=self.habit_name,
            description=self.habit_description,
            habit_type=self.habit_type,
            target_behavior=self.target_behavior,
            participants=self.participants
        )
        
        participant_id = "alice"
        tracking_period = "2024-W01"
        execution_count = 4
        target_count = 5
        quality_score = 0.8
        engagement_level = 0.9
        
        progress = self.engine.track_habit_progress(
            habit_id=habit.id,
            participant_id=participant_id,
            tracking_period=tracking_period,
            execution_count=execution_count,
            target_count=target_count,
            quality_score=quality_score,
            engagement_level=engagement_level
        )
        
        # Verify progress structure
        assert progress.id is not None
        assert progress.habit_id == habit.id
        assert progress.participant_id == participant_id
        assert progress.tracking_period == tracking_period
        assert progress.execution_count == execution_count
        assert progress.target_count == target_count
        assert progress.consistency_rate == execution_count / target_count
        assert progress.quality_score == quality_score
        assert progress.engagement_level == engagement_level
        assert isinstance(progress.barriers_encountered, list)
        assert isinstance(progress.support_received, list)
        assert isinstance(progress.improvements_noted, list)
        assert isinstance(progress.feedback_provided, str)
        assert len(progress.feedback_provided) > 0
        assert isinstance(progress.next_period_goals, list)
        assert len(progress.next_period_goals) > 0
    
    def test_progress_feedback_generation(self):
        """Test progress feedback generation"""
        # Test excellent progress
        excellent_feedback = self.engine._generate_progress_feedback(0.9, 0.9, 0.9)
        assert "excellent" in excellent_feedback.lower()
        
        # Test poor consistency
        poor_consistency_feedback = self.engine._generate_progress_feedback(0.3, 0.8, 0.7)
        assert "consistency" in poor_consistency_feedback.lower()
        
        # Test poor quality
        poor_quality_feedback = self.engine._generate_progress_feedback(0.8, 0.4, 0.7)
        assert "quality" in poor_quality_feedback.lower()
    
    def test_calculate_habit_formation_metrics(self):
        """Test comprehensive habit formation metrics calculation"""
        organization_id = "test_org_123"
        
        # Create habits for the organization
        habit1 = self.engine.design_organizational_habit(
            name="Daily Standup",
            description="Daily team sync",
            habit_type=HabitType.COMMUNICATION,
            target_behavior="improve communication",
            participants=[organization_id, "alice", "bob"]
        )
        
        habit2 = self.engine.design_organizational_habit(
            name="Weekly Retrospective",
            description="Weekly team reflection",
            habit_type=HabitType.REFLECTION,
            target_behavior="continuous improvement",
            participants=[organization_id, "charlie", "david"]
        )
        
        # Track some progress
        self.engine.track_habit_progress(
            habit_id=habit1.id,
            participant_id="alice",
            tracking_period="2024-W01",
            execution_count=5,
            target_count=5,
            quality_score=0.8,
            engagement_level=0.9
        )
        
        metrics = self.engine.calculate_habit_formation_metrics(organization_id)
        
        # Verify metrics structure
        assert metrics.organization_id == organization_id
        assert metrics.total_habits_designed >= 2
        assert metrics.habits_in_formation >= 0
        assert metrics.habits_established >= 0
        assert metrics.habits_institutionalized >= 0
        assert metrics.average_formation_time_weeks > 0
        assert 0.0 <= metrics.overall_success_rate <= 1.0
        assert 0.0 <= metrics.participant_engagement_average <= 1.0
        assert 0.0 <= metrics.sustainability_index <= 1.0
        assert 0.0 <= metrics.cultural_integration_score <= 1.0
        assert 0.0 <= metrics.business_impact_score <= 1.0
        assert metrics.roi_achieved >= 0.0
        assert metrics.calculated_date is not None
    
    def test_get_habit(self):
        """Test habit retrieval"""
        habit = self.engine.design_organizational_habit(
            name=self.habit_name,
            description=self.habit_description,
            habit_type=self.habit_type,
            target_behavior=self.target_behavior,
            participants=self.participants
        )
        
        # Should retrieve the same habit
        retrieved = self.engine.get_habit(habit.id)
        assert retrieved is not None
        assert retrieved.id == habit.id
        assert retrieved.name == habit.name
        
        # Should return None for non-existent ID
        non_existent = self.engine.get_habit("non-existent-id")
        assert non_existent is None
    
    def test_get_organization_habits(self):
        """Test organization habits retrieval"""
        organization_id = "test_org_456"
        
        # Create habits for the organization
        habit1 = self.engine.design_organizational_habit(
            name="Habit 1",
            description="First habit",
            habit_type=HabitType.COMMUNICATION,
            target_behavior="behavior 1",
            participants=[organization_id, "alice"]
        )
        
        habit2 = self.engine.design_organizational_habit(
            name="Habit 2",
            description="Second habit",
            habit_type=HabitType.COLLABORATION,
            target_behavior="behavior 2",
            participants=[organization_id, "bob"]
        )
        
        # Create habit for different organization
        other_habit = self.engine.design_organizational_habit(
            name="Other Habit",
            description="Other organization habit",
            habit_type=HabitType.LEARNING,
            target_behavior="other behavior",
            participants=["other_org", "charlie"]
        )
        
        # Should retrieve only organization's habits
        org_habits = self.engine.get_organization_habits(organization_id)
        assert len(org_habits) == 2
        habit_ids = [h.id for h in org_habits]
        assert habit1.id in habit_ids
        assert habit2.id in habit_ids
        assert other_habit.id not in habit_ids
        
        # Should return empty list for non-existent organization
        empty_habits = self.engine.get_organization_habits("non-existent-org")
        assert len(empty_habits) == 0
    
    def test_get_formation_strategy(self):
        """Test formation strategy retrieval"""
        habit = self.engine.design_organizational_habit(
            name=self.habit_name,
            description=self.habit_description,
            habit_type=self.habit_type,
            target_behavior=self.target_behavior,
            participants=self.participants
        )
        
        strategy = self.engine.create_habit_formation_strategy(habit)
        
        # Should retrieve the same strategy
        retrieved = self.engine.get_formation_strategy(strategy.id)
        assert retrieved is not None
        assert retrieved.id == strategy.id
        assert retrieved.habit_id == strategy.habit_id
        
        # Should return None for non-existent ID
        non_existent = self.engine.get_formation_strategy("non-existent-id")
        assert non_existent is None
    
    def test_get_sustainability_mechanism(self):
        """Test sustainability mechanism retrieval"""
        habit = self.engine.design_organizational_habit(
            name=self.habit_name,
            description=self.habit_description,
            habit_type=self.habit_type,
            target_behavior=self.target_behavior,
            participants=self.participants
        )
        
        sustainability = self.engine.implement_habit_sustainability_mechanisms(habit)
        
        # Should retrieve the same sustainability mechanism
        retrieved = self.engine.get_sustainability_mechanism(sustainability.id)
        assert retrieved is not None
        assert retrieved.id == sustainability.id
        assert retrieved.habit_id == sustainability.habit_id
        
        # Should return None for non-existent ID
        non_existent = self.engine.get_sustainability_mechanism("non-existent-id")
        assert non_existent is None
    
    def test_error_handling(self):
        """Test error handling in habit formation engine"""
        # Test with invalid cultural alignment
        with pytest.raises(ValueError):
            habit = self.engine.design_organizational_habit(
                name=self.habit_name,
                description=self.habit_description,
                habit_type=self.habit_type,
                target_behavior=self.target_behavior,
                participants=self.participants
            )
            # Manually set invalid cultural alignment to test validation
            habit.cultural_alignment = 1.5  # Invalid: > 1.0
            habit.__post_init__()
        
        # Test with invalid duration
        with pytest.raises(ValueError):
            habit = self.engine.design_organizational_habit(
                name=self.habit_name,
                description=self.habit_description,
                habit_type=self.habit_type,
                target_behavior=self.target_behavior,
                participants=self.participants
            )
            # Manually set invalid duration to test validation
            habit.duration_minutes = -5  # Invalid: negative
            habit.__post_init__()
        
        # Test progress tracking with invalid scores
        habit = self.engine.design_organizational_habit(
            name=self.habit_name,
            description=self.habit_description,
            habit_type=self.habit_type,
            target_behavior=self.target_behavior,
            participants=self.participants
        )
        
        with pytest.raises(ValueError):
            self.engine.track_habit_progress(
                habit_id=habit.id,
                participant_id="test_participant",
                tracking_period="2024-W01",
                execution_count=5,
                target_count=5,
                quality_score=1.5,  # Invalid: > 1.0
                engagement_level=0.8
            )
    
    def test_habit_type_specific_behavior(self):
        """Test habit type specific behavior generation"""
        # Test different habit types generate appropriate content
        habit_types_to_test = [
            HabitType.COMMUNICATION,
            HabitType.COLLABORATION,
            HabitType.LEARNING,
            HabitType.INNOVATION,
            HabitType.QUALITY,
            HabitType.FEEDBACK,
            HabitType.RECOGNITION,
            HabitType.PLANNING,
            HabitType.REFLECTION,
            HabitType.WELLNESS
        ]
        
        for habit_type in habit_types_to_test:
            habit = self.engine.design_organizational_habit(
                name=f"Test {habit_type.value} Habit",
                description=f"Test habit for {habit_type.value}",
                habit_type=habit_type,
                target_behavior=f"improve {habit_type.value}",
                participants=self.participants
            )
            
            # Verify habit was created successfully
            assert habit.habit_type == habit_type
            assert len(habit.trigger_conditions) > 0
            assert len(habit.execution_steps) > 0
            assert len(habit.success_indicators) > 0
            assert habit.duration_minutes > 0
            
            # Verify habit-specific content
            habit_content = (
                ' '.join(habit.trigger_conditions) + ' ' +
                ' '.join(habit.execution_steps) + ' ' +
                ' '.join(habit.success_indicators)
            ).lower()
            
            # Should contain habit type related keywords
            assert habit_type.value in habit_content or any(
                keyword in habit_content 
                for keyword in habit_type.value.split('_')
            )


if __name__ == "__main__":
    pytest.main([__file__])