"""
Tests for Behavior Modification Engine

Tests behavior modification strategy development, intervention creation,
and progress tracking with optimization.
"""

import pytest
from datetime import datetime
from uuid import uuid4

from scrollintel.engines.behavior_modification_engine import BehaviorModificationEngine
from scrollintel.models.behavior_modification_models import (
    ModificationTechnique, ModificationStatus, ProgressLevel
)


class TestBehaviorModificationEngine:
    """Test suite for BehaviorModificationEngine"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.engine = BehaviorModificationEngine()
        self.target_behavior = "improve team communication"
        self.current_state = {
            'complexity': 'medium',
            'participant_count': 15,
            'strength': 0.4,
            'frequency': 0.3,
            'quality': 0.5
        }
        self.desired_outcome = "achieve clear and effective team communication"
        self.constraints = {
            'urgency': 'medium',
            'budget': 'medium',
            'timeline_weeks': 12
        }
        self.stakeholders = ["team_lead", "hr_manager", "team_members"]
    
    def test_develop_modification_strategy(self):
        """Test systematic behavior change strategy development"""
        strategy = self.engine.develop_modification_strategy(
            target_behavior=self.target_behavior,
            current_state=self.current_state,
            desired_outcome=self.desired_outcome,
            constraints=self.constraints,
            stakeholders=self.stakeholders
        )
        
        # Verify strategy structure
        assert strategy.id is not None
        assert strategy.target_behavior == self.target_behavior
        assert strategy.desired_outcome == self.desired_outcome
        assert isinstance(strategy.techniques, list)
        assert len(strategy.techniques) > 0
        assert strategy.timeline_weeks > 0
        assert isinstance(strategy.success_criteria, list)
        assert len(strategy.success_criteria) > 0
        assert isinstance(strategy.resources_required, list)
        assert strategy.stakeholders == self.stakeholders
        assert isinstance(strategy.risk_factors, list)
        assert isinstance(strategy.mitigation_strategies, list)
        assert strategy.created_by == "ScrollIntel"
        
        # Verify techniques are appropriate
        assert all(isinstance(t, ModificationTechnique) for t in strategy.techniques)
        assert ModificationTechnique.FEEDBACK in strategy.techniques  # Always included
        
        # Verify timeline is reasonable
        assert 4 <= strategy.timeline_weeks <= 52
    
    def test_select_modification_techniques(self):
        """Test modification technique selection"""
        techniques = self.engine._select_modification_techniques(
            target_behavior="improve collaboration",
            current_state=self.current_state,
            desired_outcome="better teamwork",
            constraints=self.constraints
        )
        
        # Should return appropriate techniques
        assert len(techniques) > 0
        assert len(techniques) <= 5  # Limited to 5 techniques
        assert ModificationTechnique.FEEDBACK in techniques  # Always included
        
        # Should include collaboration-specific techniques
        collaboration_techniques = [
            ModificationTechnique.PEER_INFLUENCE,
            ModificationTechnique.ENVIRONMENTAL_DESIGN,
            ModificationTechnique.GOAL_SETTING
        ]
        assert any(t in techniques for t in collaboration_techniques)
    
    def test_estimate_modification_timeline(self):
        """Test timeline estimation for behavior modification"""
        techniques = [
            ModificationTechnique.FEEDBACK,
            ModificationTechnique.COACHING,
            ModificationTechnique.TRAINING
        ]
        
        timeline = self.engine._estimate_modification_timeline(
            target_behavior=self.target_behavior,
            current_state=self.current_state,
            techniques=techniques
        )
        
        # Should return reasonable timeline
        assert 4 <= timeline <= 52
        assert isinstance(timeline, int)
        
        # Test with high complexity
        complex_state = self.current_state.copy()
        complex_state['complexity'] = 'high'
        complex_timeline = self.engine._estimate_modification_timeline(
            target_behavior=self.target_behavior,
            current_state=complex_state,
            techniques=techniques
        )
        
        # Complex behaviors should take longer
        assert complex_timeline >= timeline
    
    def test_define_success_criteria(self):
        """Test success criteria definition"""
        criteria = self.engine._define_success_criteria(
            target_behavior=self.target_behavior,
            desired_outcome=self.desired_outcome,
            current_state=self.current_state
        )
        
        # Should define meaningful criteria
        assert len(criteria) > 0
        assert all(isinstance(c, str) for c in criteria)
        assert all(len(c) > 0 for c in criteria)
        
        # Should include quantitative targets
        frequency_criteria = [c for c in criteria if "frequency" in c.lower()]
        quality_criteria = [c for c in criteria if "quality" in c.lower()]
        assert len(frequency_criteria) > 0 or len(quality_criteria) > 0
        
        # Should include sustainability criteria
        sustainability_criteria = [c for c in criteria if "maintain" in c.lower() or "sustain" in c.lower()]
        assert len(sustainability_criteria) > 0
    
    def test_create_modification_interventions(self):
        """Test intervention creation for modification strategy"""
        strategy = self.engine.develop_modification_strategy(
            target_behavior=self.target_behavior,
            current_state=self.current_state,
            desired_outcome=self.desired_outcome,
            constraints=self.constraints,
            stakeholders=self.stakeholders
        )
        
        participants = ["alice", "bob", "charlie", "david"]
        facilitators = ["coach1", "trainer1"]
        
        interventions = self.engine.create_modification_interventions(
            strategy=strategy,
            participants=participants,
            facilitators=facilitators
        )
        
        # Should create interventions for each technique
        assert len(interventions) == len(strategy.techniques)
        
        # Verify intervention structure
        for intervention in interventions:
            assert intervention.id is not None
            assert intervention.strategy_id == strategy.id
            assert intervention.technique in strategy.techniques
            assert intervention.target_participants == participants
            assert intervention.duration_days > 0
            assert intervention.status == ModificationStatus.PLANNED
            assert isinstance(intervention.implementation_steps, list)
            assert len(intervention.implementation_steps) > 0
            assert isinstance(intervention.resources_needed, list)
            assert isinstance(intervention.success_metrics, list)
            assert intervention.assigned_facilitator in facilitators
    
    def test_track_behavior_change_progress(self):
        """Test behavior change progress tracking"""
        strategy = self.engine.develop_modification_strategy(
            target_behavior=self.target_behavior,
            current_state=self.current_state,
            desired_outcome=self.desired_outcome
        )
        
        participant_id = "participant_123"
        baseline_measurement = 0.3
        current_measurement = 0.6
        target_measurement = 0.8
        
        progress = self.engine.track_behavior_change_progress(
            strategy_id=strategy.id,
            participant_id=participant_id,
            current_measurement=current_measurement,
            baseline_measurement=baseline_measurement,
            target_measurement=target_measurement
        )
        
        # Verify progress structure
        assert progress.id is not None
        assert progress.strategy_id == strategy.id
        assert progress.participant_id == participant_id
        assert progress.baseline_measurement == baseline_measurement
        assert progress.current_measurement == current_measurement
        assert progress.target_measurement == target_measurement
        assert progress.progress_level in ProgressLevel
        assert progress.improvement_rate >= 0.0
        assert isinstance(progress.milestones_achieved, list)
        assert isinstance(progress.challenges_encountered, list)
        assert isinstance(progress.adjustments_made, list)
        assert progress.next_review_date > progress.last_updated
        
        # Verify progress level calculation
        expected_progress_ratio = (current_measurement - baseline_measurement) / (target_measurement - baseline_measurement)
        if expected_progress_ratio >= 0.5:
            assert progress.progress_level in [ProgressLevel.MODERATE_PROGRESS, ProgressLevel.SIGNIFICANT_PROGRESS]
    
    def test_determine_progress_level(self):
        """Test progress level determination"""
        baseline = 0.2
        target = 0.8
        
        # Test different progress levels
        test_cases = [
            (0.2, ProgressLevel.NO_CHANGE),
            (0.3, ProgressLevel.MINIMAL_PROGRESS),
            (0.5, ProgressLevel.MODERATE_PROGRESS),
            (0.7, ProgressLevel.SIGNIFICANT_PROGRESS),
            (0.8, ProgressLevel.COMPLETE_CHANGE)
        ]
        
        for current, expected_level in test_cases:
            level = self.engine._determine_progress_level(baseline, current, target)
            assert level == expected_level
    
    def test_optimize_modification_strategy(self):
        """Test behavior modification strategy optimization"""
        strategy = self.engine.develop_modification_strategy(
            target_behavior=self.target_behavior,
            current_state=self.current_state,
            desired_outcome=self.desired_outcome
        )
        
        # Create sample progress data
        progress_data = []
        for i in range(3):
            progress = self.engine.track_behavior_change_progress(
                strategy_id=strategy.id,
                participant_id=f"participant_{i}",
                current_measurement=0.4 + (i * 0.1),
                baseline_measurement=0.3,
                target_measurement=0.8
            )
            progress_data.append(progress)
        
        effectiveness_data = {"participant_feedback": 0.7, "facilitator_assessment": 0.6}
        
        optimization = self.engine.optimize_modification_strategy(
            strategy_id=strategy.id,
            progress_data=progress_data,
            effectiveness_data=effectiveness_data
        )
        
        # Verify optimization structure
        assert optimization.strategy_id == strategy.id
        assert 0.0 <= optimization.current_effectiveness <= 1.0
        assert isinstance(optimization.optimization_opportunities, list)
        assert isinstance(optimization.recommended_adjustments, list)
        assert optimization.expected_improvement >= 0.0
        assert optimization.implementation_effort in ["low", "medium", "high"]
        assert optimization.risk_level in ["low", "medium", "high"]
        assert 0.0 <= optimization.priority_score <= 1.0
        assert optimization.analysis_date is not None
    
    def test_calculate_strategy_effectiveness(self):
        """Test strategy effectiveness calculation"""
        # Create progress data with different effectiveness levels
        progress_data = []
        
        # High effectiveness progress
        high_progress = self.engine.track_behavior_change_progress(
            strategy_id="test_strategy",
            participant_id="high_performer",
            current_measurement=0.8,
            baseline_measurement=0.3,
            target_measurement=0.9
        )
        progress_data.append(high_progress)
        
        # Medium effectiveness progress
        medium_progress = self.engine.track_behavior_change_progress(
            strategy_id="test_strategy",
            participant_id="medium_performer",
            current_measurement=0.6,
            baseline_measurement=0.3,
            target_measurement=0.9
        )
        progress_data.append(medium_progress)
        
        effectiveness = self.engine._calculate_strategy_effectiveness(progress_data)
        
        # Should return reasonable effectiveness score
        assert 0.0 <= effectiveness <= 1.0
        
        # Test with empty progress data
        empty_effectiveness = self.engine._calculate_strategy_effectiveness([])
        assert empty_effectiveness == 0.5  # Neutral effectiveness
    
    def test_calculate_modification_metrics(self):
        """Test comprehensive behavior modification metrics calculation"""
        organization_id = "test_org_123"
        
        # Create some strategies with the organization in stakeholders
        strategy1 = self.engine.develop_modification_strategy(
            target_behavior="improve communication",
            current_state=self.current_state,
            desired_outcome="better communication",
            stakeholders=[organization_id, "manager1"]
        )
        
        strategy2 = self.engine.develop_modification_strategy(
            target_behavior="enhance collaboration",
            current_state=self.current_state,
            desired_outcome="better teamwork",
            stakeholders=[organization_id, "manager2"]
        )
        
        # Create interventions
        participants = ["alice", "bob", "charlie"]
        self.engine.create_modification_interventions(strategy1, participants)
        self.engine.create_modification_interventions(strategy2, participants)
        
        # Track some progress
        for i, participant in enumerate(participants):
            self.engine.track_behavior_change_progress(
                strategy_id=strategy1.id,
                participant_id=participant,
                current_measurement=0.5 + (i * 0.1),
                baseline_measurement=0.3,
                target_measurement=0.8
            )
        
        metrics = self.engine.calculate_modification_metrics(organization_id)
        
        # Verify metrics structure
        assert metrics.organization_id == organization_id
        assert metrics.total_strategies >= 2
        assert metrics.participants_engaged >= len(participants)
        assert 0.0 <= metrics.overall_success_rate <= 1.0
        assert metrics.average_improvement_rate >= 0.0
        assert metrics.time_to_change_average > 0.0
        assert 0.0 <= metrics.participant_satisfaction_average <= 1.0
        assert metrics.cost_per_participant > 0.0
        assert metrics.roi_achieved >= 0.0
        assert 0.0 <= metrics.sustainability_index <= 1.0
        assert metrics.calculated_date is not None
    
    def test_get_strategy(self):
        """Test strategy retrieval"""
        strategy = self.engine.develop_modification_strategy(
            target_behavior=self.target_behavior,
            current_state=self.current_state,
            desired_outcome=self.desired_outcome
        )
        
        # Should retrieve the same strategy
        retrieved = self.engine.get_strategy(strategy.id)
        assert retrieved is not None
        assert retrieved.id == strategy.id
        assert retrieved.target_behavior == strategy.target_behavior
        
        # Should return None for non-existent ID
        non_existent = self.engine.get_strategy("non-existent-id")
        assert non_existent is None
    
    def test_get_organization_strategies(self):
        """Test organization strategies retrieval"""
        organization_id = "test_org_456"
        
        # Create strategies for the organization
        strategy1 = self.engine.develop_modification_strategy(
            target_behavior="behavior1",
            current_state=self.current_state,
            desired_outcome="outcome1",
            stakeholders=[organization_id, "manager1"]
        )
        
        strategy2 = self.engine.develop_modification_strategy(
            target_behavior="behavior2",
            current_state=self.current_state,
            desired_outcome="outcome2",
            stakeholders=[organization_id, "manager2"]
        )
        
        # Create strategy for different organization
        other_strategy = self.engine.develop_modification_strategy(
            target_behavior="other_behavior",
            current_state=self.current_state,
            desired_outcome="other_outcome",
            stakeholders=["other_org", "other_manager"]
        )
        
        # Should retrieve only organization's strategies
        org_strategies = self.engine.get_organization_strategies(organization_id)
        assert len(org_strategies) == 2
        strategy_ids = [s.id for s in org_strategies]
        assert strategy1.id in strategy_ids
        assert strategy2.id in strategy_ids
        assert other_strategy.id not in strategy_ids
        
        # Should return empty list for non-existent organization
        empty_strategies = self.engine.get_organization_strategies("non-existent-org")
        assert len(empty_strategies) == 0
    
    def test_intervention_details_generation(self):
        """Test intervention details generation for different techniques"""
        target_behavior = "test_behavior"
        
        # Test specific techniques
        techniques_to_test = [
            ModificationTechnique.POSITIVE_REINFORCEMENT,
            ModificationTechnique.COACHING,
            ModificationTechnique.TRAINING,
            ModificationTechnique.MODELING,
            ModificationTechnique.ENVIRONMENTAL_DESIGN
        ]
        
        for technique in techniques_to_test:
            details = self.engine._get_intervention_details(technique, target_behavior)
            
            # Verify details structure
            assert 'name' in details
            assert 'description' in details
            assert 'duration_days' in details
            assert 'frequency' in details
            assert 'steps' in details
            assert 'resources' in details
            assert 'metrics' in details
            
            # Verify content
            assert target_behavior in details['name']
            assert details['duration_days'] > 0
            assert isinstance(details['steps'], list)
            assert len(details['steps']) > 0
            assert isinstance(details['resources'], list)
            assert isinstance(details['metrics'], list)
    
    def test_error_handling(self):
        """Test error handling in behavior modification engine"""
        # Test with invalid timeline
        with pytest.raises(ValueError):
            self.engine.develop_modification_strategy(
                target_behavior=self.target_behavior,
                current_state=self.current_state,
                desired_outcome=self.desired_outcome,
                constraints={'timeline_weeks': -5}  # Invalid negative timeline
            )
        
        # Test progress tracking with invalid measurements
        strategy = self.engine.develop_modification_strategy(
            target_behavior=self.target_behavior,
            current_state=self.current_state,
            desired_outcome=self.desired_outcome
        )
        
        with pytest.raises(ValueError):
            self.engine.track_behavior_change_progress(
                strategy_id=strategy.id,
                participant_id="test_participant",
                current_measurement=1.5,  # Invalid: > 1.0
                baseline_measurement=0.3,
                target_measurement=0.8
            )


if __name__ == "__main__":
    pytest.main([__file__])