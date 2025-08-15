"""
Tests for Progress Tracking Engine

This module contains comprehensive tests for the progress tracking functionality
in the cultural transformation leadership system.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from scrollintel.engines.progress_tracking_engine import ProgressTrackingEngine
from scrollintel.models.progress_tracking_models import (
    ProgressStatus, MilestoneType, ProgressMetric, TransformationMilestone
)
from scrollintel.models.cultural_assessment_models import CulturalTransformation


class TestProgressTrackingEngine:
    """Test cases for ProgressTrackingEngine"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.engine = ProgressTrackingEngine()
        self.sample_transformation = CulturalTransformation(
            id="test_transformation_001",
            organization_id="org_001",
            current_culture={"values": ["efficiency"], "behaviors": ["task_focused"]},
            target_culture={"values": ["innovation", "collaboration"], "behaviors": ["creative", "team_oriented"]},
            vision={"statement": "Transform to innovation culture", "values": ["innovation", "collaboration"]},
            roadmap={"phases": ["assessment", "planning", "implementation"]},
            interventions=[],
            progress=0.0,
            start_date=datetime.now(),
            target_completion=datetime.now() + timedelta(days=180)
        )
    
    def test_initialize_progress_tracking(self):
        """Test progress tracking initialization"""
        result = self.engine.initialize_progress_tracking(self.sample_transformation)
        
        assert result is not None
        assert result['transformation_id'] == self.sample_transformation.id
        assert 'milestones' in result
        assert 'metrics' in result
        assert result['overall_progress'] == 0.0
        assert result['status'] == ProgressStatus.IN_PROGRESS
        
        # Verify milestones were created
        milestones = result['milestones']
        assert len(milestones) >= 3  # Assessment, planning, implementation
        
        # Verify metrics were created
        metrics = result['metrics']
        assert len(metrics) >= 3  # Engagement, alignment, behavior
        
        # Verify data is stored in engine
        assert self.sample_transformation.id in self.engine.progress_data
    
    def test_track_milestone_progress(self):
        """Test milestone progress tracking"""
        # Initialize tracking first
        self.engine.initialize_progress_tracking(self.sample_transformation)
        
        # Get a milestone to update
        tracking_data = self.engine.progress_data[self.sample_transformation.id]
        milestone_id = list(tracking_data['milestones'].keys())[0]
        
        # Update milestone progress
        progress_update = {
            'progress_percentage': 75.0,
            'status': 'in_progress',
            'validation_results': {'criterion_1': {'met': True}}
        }
        
        updated_milestone = self.engine.track_milestone_progress(
            self.sample_transformation.id, milestone_id, progress_update
        )
        
        assert updated_milestone.progress_percentage == 75.0
        assert updated_milestone.status == ProgressStatus.IN_PROGRESS
        assert 'criterion_1' in updated_milestone.validation_results
        
        # Verify overall progress was updated
        updated_tracking = self.engine.progress_data[self.sample_transformation.id]
        assert updated_tracking['overall_progress'] > 0
    
    def test_track_milestone_completion(self):
        """Test milestone completion tracking"""
        # Initialize tracking
        self.engine.initialize_progress_tracking(self.sample_transformation)
        
        # Get a milestone to complete
        tracking_data = self.engine.progress_data[self.sample_transformation.id]
        milestone_id = list(tracking_data['milestones'].keys())[0]
        
        # Complete milestone
        progress_update = {
            'progress_percentage': 100.0,
            'status': 'completed'
        }
        
        updated_milestone = self.engine.track_milestone_progress(
            self.sample_transformation.id, milestone_id, progress_update
        )
        
        assert updated_milestone.progress_percentage == 100.0
        assert updated_milestone.status == ProgressStatus.COMPLETED
        assert updated_milestone.completion_date is not None
    
    def test_update_progress_metrics(self):
        """Test progress metrics updating"""
        # Initialize tracking
        self.engine.initialize_progress_tracking(self.sample_transformation)
        
        # Update metrics
        metric_updates = {
            f"engagement_{self.sample_transformation.id}": 65.0,
            f"alignment_{self.sample_transformation.id}": 70.0
        }
        
        updated_metrics = self.engine.update_progress_metrics(
            self.sample_transformation.id, metric_updates
        )
        
        assert len(updated_metrics) == 2
        
        for metric in updated_metrics:
            assert metric.current_value in [65.0, 70.0]
            assert metric.last_updated is not None
            assert metric.trend in ["improving", "declining", "stable"]
    
    def test_generate_progress_report(self):
        """Test progress report generation"""
        # Initialize and add some progress
        self.engine.initialize_progress_tracking(self.sample_transformation)
        
        # Update some metrics
        metric_updates = {
            f"engagement_{self.sample_transformation.id}": 60.0
        }
        self.engine.update_progress_metrics(self.sample_transformation.id, metric_updates)
        
        # Generate report
        report = self.engine.generate_progress_report(self.sample_transformation.id)
        
        assert report is not None
        assert report.transformation_id == self.sample_transformation.id
        assert report.report_date is not None
        assert isinstance(report.overall_progress, float)
        assert isinstance(report.milestones, list)
        assert isinstance(report.metrics, list)
        assert isinstance(report.achievements, list)
        assert isinstance(report.challenges, list)
        assert isinstance(report.next_steps, list)
        assert isinstance(report.risk_indicators, dict)
        assert isinstance(report.recommendations, list)
    
    def test_create_progress_dashboard(self):
        """Test progress dashboard creation"""
        # Initialize tracking
        self.engine.initialize_progress_tracking(self.sample_transformation)
        
        # Create dashboard
        dashboard = self.engine.create_progress_dashboard(self.sample_transformation.id)
        
        assert dashboard is not None
        assert dashboard.transformation_id == self.sample_transformation.id
        assert dashboard.dashboard_date is not None
        assert isinstance(dashboard.overall_health_score, float)
        assert 0 <= dashboard.overall_health_score <= 100
        assert isinstance(dashboard.progress_charts, dict)
        assert isinstance(dashboard.milestone_timeline, list)
        assert isinstance(dashboard.metric_trends, dict)
        assert isinstance(dashboard.alert_indicators, list)
        assert isinstance(dashboard.executive_summary, str)
    
    def test_validate_milestone_completion(self):
        """Test milestone completion validation"""
        # Initialize tracking
        self.engine.initialize_progress_tracking(self.sample_transformation)
        
        # Get a milestone to validate
        tracking_data = self.engine.progress_data[self.sample_transformation.id]
        milestone_id = list(tracking_data['milestones'].keys())[0]
        
        # Validate milestone
        validation_result = self.engine.validate_milestone_completion(
            self.sample_transformation.id, milestone_id
        )
        
        assert validation_result is not None
        assert validation_result['milestone_id'] == milestone_id
        assert 'validation_results' in validation_result
        assert 'all_criteria_met' in validation_result
        assert 'status' in validation_result
        assert isinstance(validation_result['all_criteria_met'], bool)
    
    def test_milestone_delay_detection(self):
        """Test milestone delay detection and alerting"""
        # Initialize tracking
        self.engine.initialize_progress_tracking(self.sample_transformation)
        
        # Get a milestone and make it overdue
        tracking_data = self.engine.progress_data[self.sample_transformation.id]
        milestone_id = list(tracking_data['milestones'].keys())[0]
        milestone = tracking_data['milestones'][milestone_id]
        
        # Set target date in the past
        milestone.target_date = datetime.now() - timedelta(days=1)
        
        # Update milestone progress (should trigger delay check)
        progress_update = {'progress_percentage': 50.0}
        self.engine.track_milestone_progress(
            self.sample_transformation.id, milestone_id, progress_update
        )
        
        # Check if alert was created
        alerts = self.engine.alerts.get(self.sample_transformation.id, [])
        delay_alerts = [a for a in alerts if a.alert_type == "milestone_delay"]
        assert len(delay_alerts) > 0
    
    def test_metric_decline_detection(self):
        """Test metric decline detection and alerting"""
        # Initialize tracking
        self.engine.initialize_progress_tracking(self.sample_transformation)
        
        # Update metric with declining value
        metric_id = f"engagement_{self.sample_transformation.id}"
        
        # First update - set baseline
        self.engine.update_progress_metrics(self.sample_transformation.id, {metric_id: 60.0})
        
        # Second update - decline below threshold
        self.engine.update_progress_metrics(self.sample_transformation.id, {metric_id: 30.0})
        
        # Check if alert was created
        alerts = self.engine.alerts.get(self.sample_transformation.id, [])
        metric_alerts = [a for a in alerts if a.alert_type == "metric_decline"]
        assert len(metric_alerts) > 0
    
    def test_overall_progress_calculation(self):
        """Test overall progress calculation"""
        # Initialize tracking
        self.engine.initialize_progress_tracking(self.sample_transformation)
        
        # Update multiple milestones
        tracking_data = self.engine.progress_data[self.sample_transformation.id]
        milestone_ids = list(tracking_data['milestones'].keys())
        
        # Update first milestone to 50%
        self.engine.track_milestone_progress(
            self.sample_transformation.id, milestone_ids[0], {'progress_percentage': 50.0}
        )
        
        # Update second milestone to 75%
        if len(milestone_ids) > 1:
            self.engine.track_milestone_progress(
                self.sample_transformation.id, milestone_ids[1], {'progress_percentage': 75.0}
            )
        
        # Check overall progress calculation
        updated_tracking = self.engine.progress_data[self.sample_transformation.id]
        expected_progress = (50.0 + 75.0 + 0.0) / 3  # Third milestone still at 0%
        assert abs(updated_tracking['overall_progress'] - expected_progress) < 0.1
    
    def test_health_score_calculation(self):
        """Test health score calculation"""
        # Initialize tracking
        self.engine.initialize_progress_tracking(self.sample_transformation)
        
        # Add some progress
        tracking_data = self.engine.progress_data[self.sample_transformation.id]
        milestone_id = list(tracking_data['milestones'].keys())[0]
        
        self.engine.track_milestone_progress(
            self.sample_transformation.id, milestone_id, {'progress_percentage': 80.0}
        )
        
        # Calculate health score
        health_score = self.engine._calculate_health_score(self.sample_transformation.id)
        
        assert isinstance(health_score, float)
        assert 0 <= health_score <= 100
    
    def test_error_handling_invalid_transformation(self):
        """Test error handling for invalid transformation ID"""
        with pytest.raises(ValueError, match="No progress tracking found"):
            self.engine.track_milestone_progress("invalid_id", "milestone_id", {})
        
        with pytest.raises(ValueError, match="No progress tracking found"):
            self.engine.update_progress_metrics("invalid_id", {})
        
        with pytest.raises(ValueError, match="No progress tracking found"):
            self.engine.generate_progress_report("invalid_id")
    
    def test_error_handling_invalid_milestone(self):
        """Test error handling for invalid milestone ID"""
        # Initialize tracking
        self.engine.initialize_progress_tracking(self.sample_transformation)
        
        with pytest.raises(ValueError, match="Milestone .* not found"):
            self.engine.track_milestone_progress(
                self.sample_transformation.id, "invalid_milestone", {}
            )
    
    def test_milestone_dependencies(self):
        """Test milestone dependency handling"""
        # Initialize tracking
        self.engine.initialize_progress_tracking(self.sample_transformation)
        
        # Check that milestones have proper dependencies
        tracking_data = self.engine.progress_data[self.sample_transformation.id]
        milestones = list(tracking_data['milestones'].values())
        
        # Find planning milestone (should depend on assessment)
        planning_milestones = [m for m in milestones if m.milestone_type == MilestoneType.PLANNING]
        if planning_milestones:
            planning_milestone = planning_milestones[0]
            assert len(planning_milestone.dependencies) > 0
    
    def test_progress_metric_completion_percentage(self):
        """Test progress metric completion percentage calculation"""
        metric = ProgressMetric(
            id="test_metric",
            name="Test Metric",
            description="Test metric",
            current_value=75.0,
            target_value=100.0,
            unit="percentage",
            category="test",
            last_updated=datetime.now(),
            trend="improving"
        )
        
        assert metric.completion_percentage == 75.0
        
        # Test edge case - target is 0
        metric.target_value = 0.0
        metric.current_value = 10.0
        assert metric.completion_percentage == 100.0
        
        # Test edge case - exceeds target
        metric.target_value = 50.0
        metric.current_value = 75.0
        assert metric.completion_percentage == 100.0
    
    def test_milestone_overdue_property(self):
        """Test milestone overdue property"""
        milestone = TransformationMilestone(
            id="test_milestone",
            transformation_id="test_transformation",
            name="Test Milestone",
            description="Test milestone",
            milestone_type=MilestoneType.ASSESSMENT,
            target_date=datetime.now() - timedelta(days=1),  # Yesterday
            status=ProgressStatus.IN_PROGRESS
        )
        
        assert milestone.is_overdue is True
        
        # Test completed milestone (not overdue even if past target date)
        milestone.status = ProgressStatus.COMPLETED
        assert milestone.is_overdue is False
        
        # Test future target date
        milestone.target_date = datetime.now() + timedelta(days=1)
        milestone.status = ProgressStatus.IN_PROGRESS
        assert milestone.is_overdue is False
    
    def test_progress_report_properties(self):
        """Test progress report computed properties"""
        # Initialize tracking and generate report
        self.engine.initialize_progress_tracking(self.sample_transformation)
        
        # Complete one milestone
        tracking_data = self.engine.progress_data[self.sample_transformation.id]
        milestone_id = list(tracking_data['milestones'].keys())[0]
        self.engine.track_milestone_progress(
            self.sample_transformation.id, milestone_id, 
            {'progress_percentage': 100.0, 'status': 'completed'}
        )
        
        # Make another milestone overdue
        milestone_id_2 = list(tracking_data['milestones'].keys())[1]
        milestone_2 = tracking_data['milestones'][milestone_id_2]
        milestone_2.target_date = datetime.now() - timedelta(days=1)
        
        report = self.engine.generate_progress_report(self.sample_transformation.id)
        
        # Test completed milestones property
        completed = report.completed_milestones
        assert len(completed) == 1
        assert completed[0].status == ProgressStatus.COMPLETED
        
        # Test overdue milestones property
        overdue = report.overdue_milestones
        assert len(overdue) >= 1
        assert all(m.is_overdue for m in overdue)


if __name__ == "__main__":
    pytest.main([__file__])