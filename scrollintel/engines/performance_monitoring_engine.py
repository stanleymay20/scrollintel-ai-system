"""
Performance Monitoring Engine for Crisis Leadership Excellence

This engine provides real-time team performance tracking, issue identification,
and optimization support during crisis situations.
"""

import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
import logging
from dataclasses import asdict

from ..models.performance_monitoring_models import (
    PerformanceStatus, InterventionType, SupportType,
    PerformanceMetric, TeamMemberPerformance, PerformanceIssue,
    PerformanceIntervention, SupportProvision, TeamPerformanceOverview,
    PerformanceOptimization, PerformanceAlert, PerformanceReport
)

logger = logging.getLogger(__name__)


class PerformanceMonitoringEngine:
    """Engine for real-time team performance monitoring during crisis"""
    
    def __init__(self):
        self.performance_data = {}
        self.active_alerts = {}
        self.intervention_history = {}
        self.support_provisions = {}
        
    async def track_team_performance(self, crisis_id: str, team_members: List[str]) -> TeamPerformanceOverview:
        """Track real-time performance of crisis team members"""
        try:
            member_performances = []
            
            for member_id in team_members:
                performance = await self._assess_member_performance(crisis_id, member_id)
                member_performances.append(performance)
            
            # Calculate team-level metrics
            overall_score = sum(p.overall_score for p in member_performances) / len(member_performances)
            team_efficiency = await self._calculate_team_efficiency(member_performances)
            collaboration_index = await self._calculate_collaboration_index(member_performances)
            stress_level_avg = sum(p.stress_level for p in member_performances) / len(member_performances)
            task_completion_rate = sum(p.task_completion_rate for p in member_performances) / len(member_performances)
            response_time_avg = sum(p.response_time_avg for p in member_performances) / len(member_performances)
            
            # Count critical issues and active interventions
            critical_issues = sum(1 for p in member_performances if p.performance_status == PerformanceStatus.CRITICAL)
            active_interventions = sum(len(p.interventions_needed) for p in member_performances)
            active_support = len([s for s in self.support_provisions.values() if s.crisis_id == crisis_id])
            
            team_overview = TeamPerformanceOverview(
                crisis_id=crisis_id,
                team_id=f"crisis_team_{crisis_id}",
                overall_performance_score=overall_score,
                team_efficiency=team_efficiency,
                collaboration_index=collaboration_index,
                stress_level_avg=stress_level_avg,
                task_completion_rate=task_completion_rate,
                response_time_avg=response_time_avg,
                member_performances=member_performances,
                critical_issues_count=critical_issues,
                interventions_active=active_interventions,
                support_provisions_active=active_support,
                last_updated=datetime.now()
            )
            
            # Store performance data
            self.performance_data[crisis_id] = team_overview
            
            logger.info(f"Team performance tracked for crisis {crisis_id}: {overall_score:.2f}")
            return team_overview
            
        except Exception as e:
            logger.error(f"Error tracking team performance: {str(e)}")
            raise
    
    async def _assess_member_performance(self, crisis_id: str, member_id: str) -> TeamMemberPerformance:
        """Assess individual team member performance"""
        # Simulate performance metrics collection
        metrics = [
            PerformanceMetric("task_completion", 85.0, "percentage", datetime.now(), 70.0, 100.0),
            PerformanceMetric("response_time", 2.5, "minutes", datetime.now(), 0.0, 5.0),
            PerformanceMetric("quality_score", 88.0, "percentage", datetime.now(), 75.0, 100.0),
            PerformanceMetric("stress_level", 6.5, "scale_1_10", datetime.now(), 1.0, 10.0, True),
            PerformanceMetric("collaboration", 92.0, "percentage", datetime.now(), 80.0, 100.0)
        ]
        
        # Calculate overall performance score
        overall_score = (metrics[0].value + metrics[2].value + metrics[4].value) / 3
        
        # Determine performance status
        if overall_score >= 90:
            status = PerformanceStatus.EXCELLENT
        elif overall_score >= 80:
            status = PerformanceStatus.GOOD
        elif overall_score >= 70:
            status = PerformanceStatus.AVERAGE
        elif overall_score >= 60:
            status = PerformanceStatus.BELOW_AVERAGE
        else:
            status = PerformanceStatus.CRITICAL
        
        # Identify issues and needed interventions
        issues = []
        interventions = []
        
        if metrics[1].value > 4.0:  # Response time too high
            issues.append("Slow response time")
            interventions.append(InterventionType.ADDITIONAL_SUPPORT)
        
        if metrics[3].value > 8.0:  # High stress level
            issues.append("High stress level")
            interventions.append(InterventionType.COACHING)
        
        if metrics[2].value < 75.0:  # Low quality score
            issues.append("Quality concerns")
            interventions.append(InterventionType.TRAINING)
        
        return TeamMemberPerformance(
            member_id=member_id,
            member_name=f"Member_{member_id}",
            role="Crisis Response Specialist",
            crisis_id=crisis_id,
            performance_status=status,
            overall_score=overall_score,
            metrics=metrics,
            task_completion_rate=metrics[0].value,
            response_time_avg=metrics[1].value,
            quality_score=metrics[2].value,
            stress_level=metrics[3].value,
            collaboration_score=metrics[4].value,
            last_updated=datetime.now(),
            issues_identified=issues,
            interventions_needed=interventions
        )
    
    async def identify_performance_issues(self, crisis_id: str) -> List[PerformanceIssue]:
        """Identify performance issues requiring intervention"""
        try:
            issues = []
            
            if crisis_id not in self.performance_data:
                return issues
            
            team_overview = self.performance_data[crisis_id]
            
            for member_performance in team_overview.member_performances:
                for issue_desc in member_performance.issues_identified:
                    issue = PerformanceIssue(
                        issue_id=f"issue_{crisis_id}_{member_performance.member_id}_{len(issues)}",
                        member_id=member_performance.member_id,
                        crisis_id=crisis_id,
                        issue_type=self._categorize_issue(issue_desc),
                        severity=self._assess_issue_severity(member_performance.performance_status),
                        description=issue_desc,
                        impact_assessment=await self._assess_issue_impact(issue_desc, member_performance),
                        identified_at=datetime.now()
                    )
                    issues.append(issue)
            
            logger.info(f"Identified {len(issues)} performance issues for crisis {crisis_id}")
            return issues
            
        except Exception as e:
            logger.error(f"Error identifying performance issues: {str(e)}")
            raise
    
    async def implement_intervention(self, crisis_id: str, member_id: str, 
                                   intervention_type: InterventionType) -> PerformanceIntervention:
        """Implement performance intervention"""
        try:
            intervention_id = f"intervention_{crisis_id}_{member_id}_{datetime.now().timestamp()}"
            
            intervention = PerformanceIntervention(
                intervention_id=intervention_id,
                member_id=member_id,
                crisis_id=crisis_id,
                intervention_type=intervention_type,
                description=await self._generate_intervention_description(intervention_type),
                expected_outcome=await self._define_expected_outcome(intervention_type),
                implemented_at=datetime.now()
            )
            
            # Store intervention
            if crisis_id not in self.intervention_history:
                self.intervention_history[crisis_id] = []
            self.intervention_history[crisis_id].append(intervention)
            
            # Execute intervention logic
            await self._execute_intervention(intervention)
            
            logger.info(f"Implemented {intervention_type.value} intervention for member {member_id}")
            return intervention
            
        except Exception as e:
            logger.error(f"Error implementing intervention: {str(e)}")
            raise
    
    async def provide_support(self, crisis_id: str, member_id: str, 
                            support_type: SupportType, provider: str) -> SupportProvision:
        """Provide support to team member"""
        try:
            support_id = f"support_{crisis_id}_{member_id}_{datetime.now().timestamp()}"
            
            support = SupportProvision(
                support_id=support_id,
                member_id=member_id,
                crisis_id=crisis_id,
                support_type=support_type,
                description=await self._generate_support_description(support_type),
                provider=provider,
                provided_at=datetime.now()
            )
            
            # Store support provision
            self.support_provisions[support_id] = support
            
            # Execute support provision
            await self._execute_support_provision(support)
            
            logger.info(f"Provided {support_type.value} support to member {member_id}")
            return support
            
        except Exception as e:
            logger.error(f"Error providing support: {str(e)}")
            raise
    
    async def optimize_team_performance(self, crisis_id: str) -> List[PerformanceOptimization]:
        """Generate performance optimization recommendations"""
        try:
            optimizations = []
            
            if crisis_id not in self.performance_data:
                return optimizations
            
            team_overview = self.performance_data[crisis_id]
            
            # Analyze team performance areas for optimization
            if team_overview.overall_performance_score < 80:
                optimizations.append(await self._create_overall_performance_optimization(team_overview))
            
            if team_overview.team_efficiency < 75:
                optimizations.append(await self._create_efficiency_optimization(team_overview))
            
            if team_overview.collaboration_index < 85:
                optimizations.append(await self._create_collaboration_optimization(team_overview))
            
            if team_overview.stress_level_avg > 7:
                optimizations.append(await self._create_stress_reduction_optimization(team_overview))
            
            logger.info(f"Generated {len(optimizations)} optimization recommendations for crisis {crisis_id}")
            return optimizations
            
        except Exception as e:
            logger.error(f"Error optimizing team performance: {str(e)}")
            raise
    
    async def generate_performance_alerts(self, crisis_id: str) -> List[PerformanceAlert]:
        """Generate performance monitoring alerts"""
        try:
            alerts = []
            
            if crisis_id not in self.performance_data:
                return alerts
            
            team_overview = self.performance_data[crisis_id]
            
            # Check for critical performance issues
            for member in team_overview.member_performances:
                if member.performance_status == PerformanceStatus.CRITICAL:
                    alert = PerformanceAlert(
                        alert_id=f"alert_{crisis_id}_{member.member_id}_{datetime.now().timestamp()}",
                        crisis_id=crisis_id,
                        member_id=member.member_id,
                        alert_type="CRITICAL_PERFORMANCE",
                        severity="HIGH",
                        message=f"Critical performance detected for {member.member_name}",
                        triggered_at=datetime.now()
                    )
                    alerts.append(alert)
                
                if member.stress_level > 8.5:
                    alert = PerformanceAlert(
                        alert_id=f"alert_{crisis_id}_{member.member_id}_stress_{datetime.now().timestamp()}",
                        crisis_id=crisis_id,
                        member_id=member.member_id,
                        alert_type="HIGH_STRESS",
                        severity="MEDIUM",
                        message=f"High stress level detected for {member.member_name}",
                        triggered_at=datetime.now()
                    )
                    alerts.append(alert)
            
            # Store alerts
            if crisis_id not in self.active_alerts:
                self.active_alerts[crisis_id] = []
            self.active_alerts[crisis_id].extend(alerts)
            
            logger.info(f"Generated {len(alerts)} performance alerts for crisis {crisis_id}")
            return alerts
            
        except Exception as e:
            logger.error(f"Error generating performance alerts: {str(e)}")
            raise
    
    async def generate_performance_report(self, crisis_id: str, 
                                        time_period_hours: int = 24) -> PerformanceReport:
        """Generate comprehensive performance report"""
        try:
            if crisis_id not in self.performance_data:
                raise ValueError(f"No performance data found for crisis {crisis_id}")
            
            team_overview = self.performance_data[crisis_id]
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=time_period_hours)
            
            # Generate key insights
            insights = await self._generate_performance_insights(team_overview)
            
            # Generate performance trends
            trends = await self._analyze_performance_trends(crisis_id, start_time, end_time)
            
            # Generate optimization recommendations
            recommendations = await self.optimize_team_performance(crisis_id)
            
            # Calculate success metrics
            success_metrics = {
                "overall_performance": team_overview.overall_performance_score,
                "team_efficiency": team_overview.team_efficiency,
                "collaboration_index": team_overview.collaboration_index,
                "task_completion_rate": team_overview.task_completion_rate,
                "average_response_time": team_overview.response_time_avg,
                "stress_level": team_overview.stress_level_avg
            }
            
            report = PerformanceReport(
                report_id=f"report_{crisis_id}_{datetime.now().timestamp()}",
                crisis_id=crisis_id,
                report_type="COMPREHENSIVE_PERFORMANCE",
                generated_at=datetime.now(),
                time_period_start=start_time,
                time_period_end=end_time,
                team_overview=team_overview,
                key_insights=insights,
                performance_trends=trends,
                recommendations=recommendations,
                success_metrics=success_metrics
            )
            
            logger.info(f"Generated performance report for crisis {crisis_id}")
            return report
            
        except Exception as e:
            logger.error(f"Error generating performance report: {str(e)}")
            raise
    
    # Helper methods
    async def _calculate_team_efficiency(self, member_performances: List[TeamMemberPerformance]) -> float:
        """Calculate team efficiency score"""
        if not member_performances:
            return 0.0
        
        efficiency_scores = []
        for member in member_performances:
            # Efficiency based on task completion rate and response time
            time_efficiency = max(0, 100 - (member.response_time_avg * 10))
            efficiency = (member.task_completion_rate + time_efficiency) / 2
            efficiency_scores.append(efficiency)
        
        return sum(efficiency_scores) / len(efficiency_scores)
    
    async def _calculate_collaboration_index(self, member_performances: List[TeamMemberPerformance]) -> float:
        """Calculate team collaboration index"""
        if not member_performances:
            return 0.0
        
        collaboration_scores = [member.collaboration_score for member in member_performances]
        return sum(collaboration_scores) / len(collaboration_scores)
    
    def _categorize_issue(self, issue_description: str) -> str:
        """Categorize performance issue type"""
        if "response time" in issue_description.lower():
            return "RESPONSE_TIME"
        elif "stress" in issue_description.lower():
            return "STRESS_MANAGEMENT"
        elif "quality" in issue_description.lower():
            return "QUALITY_CONTROL"
        else:
            return "GENERAL_PERFORMANCE"
    
    def _assess_issue_severity(self, performance_status: PerformanceStatus) -> str:
        """Assess issue severity based on performance status"""
        if performance_status == PerformanceStatus.CRITICAL:
            return "HIGH"
        elif performance_status == PerformanceStatus.BELOW_AVERAGE:
            return "MEDIUM"
        else:
            return "LOW"
    
    async def _assess_issue_impact(self, issue_desc: str, member_performance: TeamMemberPerformance) -> str:
        """Assess the impact of a performance issue"""
        if member_performance.performance_status == PerformanceStatus.CRITICAL:
            return f"Critical impact on team performance. Member {member_performance.member_name} requires immediate intervention."
        elif "stress" in issue_desc.lower():
            return "High stress levels may lead to burnout and reduced team morale."
        elif "response time" in issue_desc.lower():
            return "Slow response times may delay crisis resolution and impact team coordination."
        else:
            return "Performance issue may affect overall team effectiveness."
    
    async def _generate_intervention_description(self, intervention_type: InterventionType) -> str:
        """Generate intervention description"""
        descriptions = {
            InterventionType.COACHING: "Provide one-on-one coaching to improve performance and address specific challenges",
            InterventionType.RESOURCE_ALLOCATION: "Reallocate resources to better support team member's responsibilities",
            InterventionType.ROLE_REASSIGNMENT: "Reassign roles to better match team member's strengths and capabilities",
            InterventionType.ADDITIONAL_SUPPORT: "Provide additional support staff or resources to reduce workload",
            InterventionType.TRAINING: "Provide targeted training to address skill gaps and improve performance",
            InterventionType.WORKLOAD_ADJUSTMENT: "Adjust workload to prevent burnout and maintain performance quality"
        }
        return descriptions.get(intervention_type, "Performance intervention")
    
    async def _define_expected_outcome(self, intervention_type: InterventionType) -> str:
        """Define expected outcome for intervention"""
        outcomes = {
            InterventionType.COACHING: "Improved performance metrics and increased confidence",
            InterventionType.RESOURCE_ALLOCATION: "Better resource utilization and reduced bottlenecks",
            InterventionType.ROLE_REASSIGNMENT: "Better role-skill alignment and improved performance",
            InterventionType.ADDITIONAL_SUPPORT: "Reduced workload stress and maintained quality",
            InterventionType.TRAINING: "Enhanced skills and improved task execution",
            InterventionType.WORKLOAD_ADJUSTMENT: "Sustainable performance levels and reduced burnout risk"
        }
        return outcomes.get(intervention_type, "Improved team member performance")
    
    async def _execute_intervention(self, intervention: PerformanceIntervention):
        """Execute intervention logic"""
        # Simulate intervention execution
        await asyncio.sleep(0.1)
        intervention.completion_status = "in_progress"
    
    async def _generate_support_description(self, support_type: SupportType) -> str:
        """Generate support description"""
        descriptions = {
            SupportType.TECHNICAL_SUPPORT: "Provide technical assistance and troubleshooting support",
            SupportType.EMOTIONAL_SUPPORT: "Provide emotional support and stress management assistance",
            SupportType.RESOURCE_SUPPORT: "Provide additional resources and tools needed for task completion",
            SupportType.MENTORING: "Provide mentoring and guidance from experienced team members",
            SupportType.SKILL_DEVELOPMENT: "Provide skill development opportunities and training resources"
        }
        return descriptions.get(support_type, "Team member support")
    
    async def _execute_support_provision(self, support: SupportProvision):
        """Execute support provision logic"""
        # Simulate support provision execution
        await asyncio.sleep(0.1)
        support.effectiveness_rating = 8.5
    
    async def _create_overall_performance_optimization(self, team_overview: TeamPerformanceOverview) -> PerformanceOptimization:
        """Create overall performance optimization"""
        return PerformanceOptimization(
            optimization_id=f"opt_overall_{team_overview.crisis_id}_{datetime.now().timestamp()}",
            crisis_id=team_overview.crisis_id,
            target_area="Overall Performance",
            current_performance=team_overview.overall_performance_score,
            target_performance=85.0,
            optimization_strategy="Implement targeted interventions for underperforming team members",
            implementation_steps=[
                "Identify specific performance gaps",
                "Implement targeted coaching and training",
                "Provide additional resources where needed",
                "Monitor progress and adjust interventions"
            ],
            expected_impact="15-20% improvement in overall team performance",
            priority_level="HIGH",
            estimated_completion_time=120,
            resources_required=["Performance coaches", "Training materials", "Additional support staff"]
        )
    
    async def _create_efficiency_optimization(self, team_overview: TeamPerformanceOverview) -> PerformanceOptimization:
        """Create efficiency optimization"""
        return PerformanceOptimization(
            optimization_id=f"opt_efficiency_{team_overview.crisis_id}_{datetime.now().timestamp()}",
            crisis_id=team_overview.crisis_id,
            target_area="Team Efficiency",
            current_performance=team_overview.team_efficiency,
            target_performance=85.0,
            optimization_strategy="Streamline processes and improve resource allocation",
            implementation_steps=[
                "Analyze current workflow bottlenecks",
                "Implement process improvements",
                "Optimize resource allocation",
                "Establish efficiency monitoring"
            ],
            expected_impact="10-15% improvement in team efficiency",
            priority_level="MEDIUM",
            estimated_completion_time=90,
            resources_required=["Process analysts", "Workflow tools", "Resource optimization software"]
        )
    
    async def _create_collaboration_optimization(self, team_overview: TeamPerformanceOverview) -> PerformanceOptimization:
        """Create collaboration optimization"""
        return PerformanceOptimization(
            optimization_id=f"opt_collaboration_{team_overview.crisis_id}_{datetime.now().timestamp()}",
            crisis_id=team_overview.crisis_id,
            target_area="Team Collaboration",
            current_performance=team_overview.collaboration_index,
            target_performance=90.0,
            optimization_strategy="Enhance communication and teamwork practices",
            implementation_steps=[
                "Implement regular team check-ins",
                "Establish clear communication protocols",
                "Provide collaboration training",
                "Use collaboration tools and platforms"
            ],
            expected_impact="Improved team coordination and communication",
            priority_level="MEDIUM",
            estimated_completion_time=60,
            resources_required=["Communication platforms", "Collaboration tools", "Team building facilitators"]
        )
    
    async def _create_stress_reduction_optimization(self, team_overview: TeamPerformanceOverview) -> PerformanceOptimization:
        """Create stress reduction optimization"""
        return PerformanceOptimization(
            optimization_id=f"opt_stress_{team_overview.crisis_id}_{datetime.now().timestamp()}",
            crisis_id=team_overview.crisis_id,
            target_area="Stress Management",
            current_performance=10.0 - team_overview.stress_level_avg,  # Invert stress level for optimization
            target_performance=7.0,
            optimization_strategy="Implement stress reduction and wellness programs",
            implementation_steps=[
                "Provide stress management training",
                "Implement regular breaks and rotation",
                "Offer wellness support services",
                "Monitor and adjust workload distribution"
            ],
            expected_impact="Reduced stress levels and improved team resilience",
            priority_level="HIGH",
            estimated_completion_time=45,
            resources_required=["Wellness counselors", "Stress management tools", "Workload balancing systems"]
        )
    
    async def _generate_performance_insights(self, team_overview: TeamPerformanceOverview) -> List[str]:
        """Generate key performance insights"""
        insights = []
        
        if team_overview.overall_performance_score >= 85:
            insights.append("Team is performing at excellent levels with strong coordination")
        elif team_overview.overall_performance_score < 70:
            insights.append("Team performance is below optimal levels and requires immediate attention")
        
        if team_overview.stress_level_avg > 7:
            insights.append("High stress levels detected across the team - wellness interventions recommended")
        
        if team_overview.collaboration_index >= 90:
            insights.append("Excellent team collaboration and communication observed")
        elif team_overview.collaboration_index < 80:
            insights.append("Team collaboration could be improved through better communication protocols")
        
        if team_overview.critical_issues_count > 0:
            insights.append(f"{team_overview.critical_issues_count} critical performance issues require immediate intervention")
        
        return insights
    
    async def _analyze_performance_trends(self, crisis_id: str, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """Analyze performance trends over time"""
        # Simulate trend analysis
        return {
            "performance_trend": "stable",
            "efficiency_trend": "improving",
            "stress_trend": "increasing",
            "collaboration_trend": "stable",
            "response_time_trend": "improving"
        }