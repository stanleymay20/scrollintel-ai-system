"""
Progress Tracking Engine for Cultural Transformation Leadership

This engine provides continuous monitoring of transformation progress,
milestone tracking, and progress reporting capabilities.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import asdict

from ..models.progress_tracking_models import (
    ProgressMetric, TransformationMilestone, ProgressReport,
    ProgressDashboard, ProgressAlert, ProgressStatus, MilestoneType
)
from ..models.cultural_assessment_models import CulturalTransformation


class ProgressTrackingEngine:
    """Engine for tracking cultural transformation progress"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.progress_data = {}
        self.milestones = {}
        self.metrics = {}
        self.alerts = {}
    
    def initialize_progress_tracking(self, transformation: CulturalTransformation) -> Dict[str, Any]:
        """Initialize progress tracking for a transformation"""
        try:
            # Create initial milestones based on transformation plan
            milestones = self._create_transformation_milestones(transformation)
            
            # Initialize progress metrics
            metrics = self._initialize_progress_metrics(transformation)
            
            # Set up tracking structure
            tracking_data = {
                'transformation_id': transformation.id,
                'start_date': datetime.now(),
                'milestones': milestones,
                'metrics': metrics,
                'overall_progress': 0.0,
                'status': ProgressStatus.IN_PROGRESS
            }
            
            self.progress_data[transformation.id] = tracking_data
            
            self.logger.info(f"Initialized progress tracking for transformation {transformation.id}")
            return tracking_data
            
        except Exception as e:
            self.logger.error(f"Error initializing progress tracking: {str(e)}")
            raise
    
    def track_milestone_progress(self, transformation_id: str, milestone_id: str, 
                               progress_update: Dict[str, Any]) -> TransformationMilestone:
        """Track progress on a specific milestone"""
        try:
            if transformation_id not in self.progress_data:
                raise ValueError(f"No progress tracking found for transformation {transformation_id}")
            
            milestone = self._get_milestone(transformation_id, milestone_id)
            if not milestone:
                raise ValueError(f"Milestone {milestone_id} not found")
            
            # Update milestone progress
            milestone.progress_percentage = progress_update.get('progress_percentage', milestone.progress_percentage)
            milestone.status = ProgressStatus(progress_update.get('status', milestone.status.value))
            
            if milestone.status == ProgressStatus.COMPLETED:
                milestone.completion_date = datetime.now()
            
            # Update validation results if provided
            if 'validation_results' in progress_update:
                milestone.validation_results.update(progress_update['validation_results'])
            
            # Check for delays and create alerts
            self._check_milestone_delays(transformation_id, milestone)
            
            # Update overall transformation progress
            self._update_overall_progress(transformation_id)
            
            self.logger.info(f"Updated milestone {milestone_id} progress to {milestone.progress_percentage}%")
            return milestone
            
        except Exception as e:
            self.logger.error(f"Error tracking milestone progress: {str(e)}")
            raise
    
    def update_progress_metrics(self, transformation_id: str, 
                              metric_updates: Dict[str, float]) -> List[ProgressMetric]:
        """Update progress metrics"""
        try:
            if transformation_id not in self.progress_data:
                raise ValueError(f"No progress tracking found for transformation {transformation_id}")
            
            updated_metrics = []
            tracking_data = self.progress_data[transformation_id]
            
            for metric_id, new_value in metric_updates.items():
                metric = self._get_metric(transformation_id, metric_id)
                if metric:
                    old_value = metric.current_value
                    metric.current_value = new_value
                    metric.last_updated = datetime.now()
                    
                    # Determine trend
                    if new_value > old_value:
                        metric.trend = "improving"
                    elif new_value < old_value:
                        metric.trend = "declining"
                    else:
                        metric.trend = "stable"
                    
                    updated_metrics.append(metric)
                    
                    # Check for metric alerts
                    self._check_metric_alerts(transformation_id, metric)
            
            self.logger.info(f"Updated {len(updated_metrics)} metrics for transformation {transformation_id}")
            return updated_metrics
            
        except Exception as e:
            self.logger.error(f"Error updating progress metrics: {str(e)}")
            raise
    
    def generate_progress_report(self, transformation_id: str) -> ProgressReport:
        """Generate comprehensive progress report"""
        try:
            if transformation_id not in self.progress_data:
                raise ValueError(f"No progress tracking found for transformation {transformation_id}")
            
            tracking_data = self.progress_data[transformation_id]
            
            # Gather current milestones and metrics
            milestones = list(tracking_data['milestones'].values())
            metrics = list(tracking_data['metrics'].values())
            
            # Analyze achievements and challenges
            achievements = self._identify_achievements(milestones, metrics)
            challenges = self._identify_challenges(milestones, metrics)
            next_steps = self._generate_next_steps(milestones)
            
            # Calculate risk indicators
            risk_indicators = self._calculate_risk_indicators(milestones, metrics)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(milestones, metrics, risk_indicators)
            
            report = ProgressReport(
                id=f"report_{transformation_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                transformation_id=transformation_id,
                report_date=datetime.now(),
                overall_progress=tracking_data['overall_progress'],
                milestones=milestones,
                metrics=metrics,
                achievements=achievements,
                challenges=challenges,
                next_steps=next_steps,
                risk_indicators=risk_indicators,
                recommendations=recommendations
            )
            
            self.logger.info(f"Generated progress report for transformation {transformation_id}")
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating progress report: {str(e)}")
            raise
    
    def create_progress_dashboard(self, transformation_id: str) -> ProgressDashboard:
        """Create progress visualization dashboard"""
        try:
            if transformation_id not in self.progress_data:
                raise ValueError(f"No progress tracking found for transformation {transformation_id}")
            
            tracking_data = self.progress_data[transformation_id]
            
            # Calculate overall health score
            health_score = self._calculate_health_score(transformation_id)
            
            # Create progress charts data
            progress_charts = self._create_progress_charts(transformation_id)
            
            # Create milestone timeline
            milestone_timeline = self._create_milestone_timeline(transformation_id)
            
            # Create metric trends
            metric_trends = self._create_metric_trends(transformation_id)
            
            # Create alert indicators
            alert_indicators = self._create_alert_indicators(transformation_id)
            
            # Generate executive summary
            executive_summary = self._generate_executive_summary(transformation_id)
            
            dashboard = ProgressDashboard(
                transformation_id=transformation_id,
                dashboard_date=datetime.now(),
                overall_health_score=health_score,
                progress_charts=progress_charts,
                milestone_timeline=milestone_timeline,
                metric_trends=metric_trends,
                alert_indicators=alert_indicators,
                executive_summary=executive_summary
            )
            
            self.logger.info(f"Created progress dashboard for transformation {transformation_id}")
            return dashboard
            
        except Exception as e:
            self.logger.error(f"Error creating progress dashboard: {str(e)}")
            raise
    
    def validate_milestone_completion(self, transformation_id: str, 
                                    milestone_id: str) -> Dict[str, Any]:
        """Validate milestone completion against success criteria"""
        try:
            milestone = self._get_milestone(transformation_id, milestone_id)
            if not milestone:
                raise ValueError(f"Milestone {milestone_id} not found")
            
            validation_results = {}
            
            for criterion in milestone.success_criteria:
                # Validate each success criterion
                validation_result = self._validate_success_criterion(
                    transformation_id, criterion
                )
                validation_results[criterion] = validation_result
            
            # Update milestone validation results
            milestone.validation_results = validation_results
            
            # Determine if milestone is truly complete
            all_criteria_met = all(result.get('met', False) for result in validation_results.values())
            
            if all_criteria_met and milestone.status != ProgressStatus.COMPLETED:
                milestone.status = ProgressStatus.COMPLETED
                milestone.completion_date = datetime.now()
            elif not all_criteria_met and milestone.status == ProgressStatus.COMPLETED:
                milestone.status = ProgressStatus.IN_PROGRESS
                milestone.completion_date = None
            
            self.logger.info(f"Validated milestone {milestone_id} completion: {all_criteria_met}")
            return {
                'milestone_id': milestone_id,
                'validation_results': validation_results,
                'all_criteria_met': all_criteria_met,
                'status': milestone.status.value
            }
            
        except Exception as e:
            self.logger.error(f"Error validating milestone completion: {str(e)}")
            raise
    
    def _create_transformation_milestones(self, transformation: CulturalTransformation) -> Dict[str, TransformationMilestone]:
        """Create initial milestones for transformation"""
        milestones = {}
        
        # Assessment milestone
        assessment_milestone = TransformationMilestone(
            id=f"assessment_{transformation.id}",
            transformation_id=transformation.id,
            name="Cultural Assessment Complete",
            description="Complete comprehensive cultural assessment",
            milestone_type=MilestoneType.ASSESSMENT,
            target_date=datetime.now() + timedelta(days=30),
            success_criteria=[
                "Culture mapping completed",
                "Subcultures identified",
                "Health metrics established"
            ]
        )
        milestones[assessment_milestone.id] = assessment_milestone
        
        # Planning milestone
        planning_milestone = TransformationMilestone(
            id=f"planning_{transformation.id}",
            transformation_id=transformation.id,
            name="Transformation Plan Finalized",
            description="Complete transformation strategy and roadmap",
            milestone_type=MilestoneType.PLANNING,
            target_date=datetime.now() + timedelta(days=60),
            success_criteria=[
                "Vision and values defined",
                "Roadmap created",
                "Interventions designed"
            ],
            dependencies=[assessment_milestone.id]
        )
        milestones[planning_milestone.id] = planning_milestone
        
        # Implementation milestones
        impl_milestone = TransformationMilestone(
            id=f"implementation_{transformation.id}",
            transformation_id=transformation.id,
            name="Core Interventions Deployed",
            description="Deploy key transformation interventions",
            milestone_type=MilestoneType.IMPLEMENTATION,
            target_date=datetime.now() + timedelta(days=120),
            success_criteria=[
                "Key interventions launched",
                "Employee engagement initiated",
                "Communication plan executed"
            ],
            dependencies=[planning_milestone.id]
        )
        milestones[impl_milestone.id] = impl_milestone
        
        return milestones
    
    def _initialize_progress_metrics(self, transformation: CulturalTransformation) -> Dict[str, ProgressMetric]:
        """Initialize progress metrics for transformation"""
        metrics = {}
        
        # Employee engagement metric
        engagement_metric = ProgressMetric(
            id=f"engagement_{transformation.id}",
            name="Employee Engagement Score",
            description="Overall employee engagement with transformation",
            current_value=0.0,
            target_value=85.0,
            unit="percentage",
            category="engagement",
            last_updated=datetime.now(),
            trend="stable"
        )
        metrics[engagement_metric.id] = engagement_metric
        
        # Cultural alignment metric
        alignment_metric = ProgressMetric(
            id=f"alignment_{transformation.id}",
            name="Cultural Alignment Score",
            description="Alignment between current and target culture",
            current_value=0.0,
            target_value=90.0,
            unit="percentage",
            category="alignment",
            last_updated=datetime.now(),
            trend="stable"
        )
        metrics[alignment_metric.id] = alignment_metric
        
        # Behavior change metric
        behavior_metric = ProgressMetric(
            id=f"behavior_{transformation.id}",
            name="Behavior Change Adoption",
            description="Adoption rate of new behaviors",
            current_value=0.0,
            target_value=80.0,
            unit="percentage",
            category="behavior",
            last_updated=datetime.now(),
            trend="stable"
        )
        metrics[behavior_metric.id] = behavior_metric
        
        return metrics
    
    def _get_milestone(self, transformation_id: str, milestone_id: str) -> Optional[TransformationMilestone]:
        """Get milestone by ID"""
        tracking_data = self.progress_data.get(transformation_id, {})
        return tracking_data.get('milestones', {}).get(milestone_id)
    
    def _get_metric(self, transformation_id: str, metric_id: str) -> Optional[ProgressMetric]:
        """Get metric by ID"""
        tracking_data = self.progress_data.get(transformation_id, {})
        return tracking_data.get('metrics', {}).get(metric_id)
    
    def _update_overall_progress(self, transformation_id: str):
        """Update overall transformation progress"""
        tracking_data = self.progress_data[transformation_id]
        milestones = tracking_data['milestones'].values()
        
        if not milestones:
            tracking_data['overall_progress'] = 0.0
            return
        
        total_progress = sum(milestone.progress_percentage for milestone in milestones)
        tracking_data['overall_progress'] = total_progress / len(milestones)
    
    def _check_milestone_delays(self, transformation_id: str, milestone: TransformationMilestone):
        """Check for milestone delays and create alerts"""
        if milestone.is_overdue and milestone.status != ProgressStatus.COMPLETED:
            alert = ProgressAlert(
                id=f"delay_{milestone.id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                transformation_id=transformation_id,
                alert_type="milestone_delay",
                severity="high",
                message=f"Milestone '{milestone.name}' is overdue",
                created_date=datetime.now()
            )
            
            if transformation_id not in self.alerts:
                self.alerts[transformation_id] = []
            self.alerts[transformation_id].append(alert)
    
    def _check_metric_alerts(self, transformation_id: str, metric: ProgressMetric):
        """Check for metric-based alerts"""
        if metric.trend == "declining" and metric.completion_percentage < 50:
            alert = ProgressAlert(
                id=f"metric_{metric.id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                transformation_id=transformation_id,
                alert_type="metric_decline",
                severity="medium",
                message=f"Metric '{metric.name}' is declining and below target",
                created_date=datetime.now()
            )
            
            if transformation_id not in self.alerts:
                self.alerts[transformation_id] = []
            self.alerts[transformation_id].append(alert)
    
    def _identify_achievements(self, milestones: List[TransformationMilestone], 
                             metrics: List[ProgressMetric]) -> List[str]:
        """Identify key achievements"""
        achievements = []
        
        completed_milestones = [m for m in milestones if m.status == ProgressStatus.COMPLETED]
        if completed_milestones:
            achievements.append(f"Completed {len(completed_milestones)} major milestones")
        
        improving_metrics = [m for m in metrics if m.trend == "improving"]
        if improving_metrics:
            achievements.append(f"{len(improving_metrics)} key metrics showing improvement")
        
        return achievements
    
    def _identify_challenges(self, milestones: List[TransformationMilestone], 
                           metrics: List[ProgressMetric]) -> List[str]:
        """Identify current challenges"""
        challenges = []
        
        overdue_milestones = [m for m in milestones if m.is_overdue]
        if overdue_milestones:
            challenges.append(f"{len(overdue_milestones)} milestones are overdue")
        
        declining_metrics = [m for m in metrics if m.trend == "declining"]
        if declining_metrics:
            challenges.append(f"{len(declining_metrics)} metrics showing decline")
        
        return challenges
    
    def _generate_next_steps(self, milestones: List[TransformationMilestone]) -> List[str]:
        """Generate recommended next steps"""
        next_steps = []
        
        in_progress = [m for m in milestones if m.status == ProgressStatus.IN_PROGRESS]
        if in_progress:
            next_steps.append(f"Focus on completing {len(in_progress)} active milestones")
        
        not_started = [m for m in milestones if m.status == ProgressStatus.NOT_STARTED]
        if not_started:
            next_steps.append(f"Initiate {len(not_started)} pending milestones")
        
        return next_steps
    
    def _calculate_risk_indicators(self, milestones: List[TransformationMilestone], 
                                 metrics: List[ProgressMetric]) -> Dict[str, float]:
        """Calculate risk indicators"""
        risk_indicators = {}
        
        # Schedule risk
        overdue_count = len([m for m in milestones if m.is_overdue])
        total_milestones = len(milestones)
        risk_indicators['schedule_risk'] = (overdue_count / total_milestones * 100) if total_milestones > 0 else 0
        
        # Performance risk
        declining_count = len([m for m in metrics if m.trend == "declining"])
        total_metrics = len(metrics)
        risk_indicators['performance_risk'] = (declining_count / total_metrics * 100) if total_metrics > 0 else 0
        
        return risk_indicators
    
    def _generate_recommendations(self, milestones: List[TransformationMilestone], 
                                metrics: List[ProgressMetric], 
                                risk_indicators: Dict[str, float]) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        if risk_indicators.get('schedule_risk', 0) > 20:
            recommendations.append("Review and adjust milestone timelines")
        
        if risk_indicators.get('performance_risk', 0) > 30:
            recommendations.append("Investigate declining metrics and implement corrective actions")
        
        return recommendations
    
    def _calculate_health_score(self, transformation_id: str) -> float:
        """Calculate overall transformation health score"""
        tracking_data = self.progress_data[transformation_id]
        
        # Combine progress, milestone completion, and metric performance
        progress_score = tracking_data['overall_progress']
        
        milestones = list(tracking_data['milestones'].values())
        completed_ratio = len([m for m in milestones if m.status == ProgressStatus.COMPLETED]) / len(milestones) if milestones else 0
        milestone_score = completed_ratio * 100
        
        metrics = list(tracking_data['metrics'].values())
        avg_metric_completion = sum(m.completion_percentage for m in metrics) / len(metrics) if metrics else 0
        
        # Weighted average
        health_score = (progress_score * 0.4 + milestone_score * 0.3 + avg_metric_completion * 0.3)
        return min(100.0, max(0.0, health_score))
    
    def _create_progress_charts(self, transformation_id: str) -> Dict[str, Any]:
        """Create progress chart data"""
        return {
            'overall_progress': self.progress_data[transformation_id]['overall_progress'],
            'milestone_completion': self._get_milestone_completion_data(transformation_id),
            'metric_trends': self._get_metric_trend_data(transformation_id)
        }
    
    def _create_milestone_timeline(self, transformation_id: str) -> List[Dict[str, Any]]:
        """Create milestone timeline data"""
        milestones = list(self.progress_data[transformation_id]['milestones'].values())
        timeline = []
        
        for milestone in sorted(milestones, key=lambda m: m.target_date):
            timeline.append({
                'id': milestone.id,
                'name': milestone.name,
                'target_date': milestone.target_date.isoformat(),
                'completion_date': milestone.completion_date.isoformat() if milestone.completion_date else None,
                'status': milestone.status.value,
                'progress': milestone.progress_percentage
            })
        
        return timeline
    
    def _create_metric_trends(self, transformation_id: str) -> Dict[str, List[float]]:
        """Create metric trend data"""
        # Simplified - in real implementation, would track historical values
        metrics = self.progress_data[transformation_id]['metrics']
        trends = {}
        
        for metric_id, metric in metrics.items():
            trends[metric.name] = [metric.current_value]  # Would be historical data
        
        return trends
    
    def _create_alert_indicators(self, transformation_id: str) -> List[Dict[str, Any]]:
        """Create alert indicator data"""
        alerts = self.alerts.get(transformation_id, [])
        indicators = []
        
        for alert in alerts:
            if not alert.resolved_date:  # Only active alerts
                indicators.append({
                    'type': alert.alert_type,
                    'severity': alert.severity,
                    'message': alert.message,
                    'created': alert.created_date.isoformat()
                })
        
        return indicators
    
    def _generate_executive_summary(self, transformation_id: str) -> str:
        """Generate executive summary"""
        tracking_data = self.progress_data[transformation_id]
        progress = tracking_data['overall_progress']
        
        milestones = list(tracking_data['milestones'].values())
        completed = len([m for m in milestones if m.status == ProgressStatus.COMPLETED])
        total = len(milestones)
        
        return f"Transformation is {progress:.1f}% complete with {completed}/{total} milestones achieved."
    
    def _get_milestone_completion_data(self, transformation_id: str) -> Dict[str, int]:
        """Get milestone completion statistics"""
        milestones = list(self.progress_data[transformation_id]['milestones'].values())
        
        return {
            'completed': len([m for m in milestones if m.status == ProgressStatus.COMPLETED]),
            'in_progress': len([m for m in milestones if m.status == ProgressStatus.IN_PROGRESS]),
            'not_started': len([m for m in milestones if m.status == ProgressStatus.NOT_STARTED]),
            'blocked': len([m for m in milestones if m.status == ProgressStatus.BLOCKED])
        }
    
    def _get_metric_trend_data(self, transformation_id: str) -> Dict[str, float]:
        """Get metric trend data"""
        metrics = list(self.progress_data[transformation_id]['metrics'].values())
        
        return {
            'improving': len([m for m in metrics if m.trend == "improving"]),
            'declining': len([m for m in metrics if m.trend == "declining"]),
            'stable': len([m for m in metrics if m.trend == "stable"])
        }
    
    def _validate_success_criterion(self, transformation_id: str, criterion: str) -> Dict[str, Any]:
        """Validate a success criterion"""
        # Simplified validation - in real implementation would have specific validation logic
        return {
            'criterion': criterion,
            'met': True,  # Would be actual validation result
            'evidence': f"Validation evidence for {criterion}",
            'validated_date': datetime.now().isoformat()
        }