#!/usr/bin/env python3
"""
Progress Tracking System Demo

This script demonstrates the progress tracking capabilities for cultural
transformation leadership, including milestone tracking, metrics monitoring,
and progress reporting.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, Any

from scrollintel.engines.progress_tracking_engine import ProgressTrackingEngine
from scrollintel.models.cultural_assessment_models import CulturalTransformation


class ProgressTrackingDemo:
    """Demo class for progress tracking system"""
    
    def __init__(self):
        self.engine = ProgressTrackingEngine()
        self.transformation_id = "demo_transformation_001"
    
    def create_sample_transformation(self) -> CulturalTransformation:
        """Create a sample cultural transformation for demo"""
        return CulturalTransformation(
            id=self.transformation_id,
            organization_id="demo_org_001",
            current_culture={
                "values": ["efficiency", "hierarchy", "stability"],
                "behaviors": ["task_focused", "individual_work", "risk_averse"],
                "norms": ["formal_communication", "top_down_decisions"]
            },
            target_culture={
                "values": ["innovation", "collaboration", "agility"],
                "behaviors": ["creative_thinking", "team_oriented", "experimental"],
                "norms": ["open_communication", "shared_decision_making"]
            },
            vision={
                "statement": "Transform to an innovation-driven, collaborative culture",
                "values": ["innovation", "collaboration", "agility", "transparency"],
                "behaviors": ["creative_problem_solving", "cross_functional_teamwork"]
            },
            roadmap={
                "phases": ["assessment", "planning", "implementation", "validation"],
                "duration_months": 12,
                "key_milestones": ["culture_mapped", "vision_aligned", "interventions_deployed"]
            },
            interventions=[
                {"type": "leadership_training", "target": "managers"},
                {"type": "team_building", "target": "all_employees"},
                {"type": "communication_workshops", "target": "cross_functional_teams"}
            ],
            progress=0.0,
            start_date=datetime.now(),
            target_completion=datetime.now() + timedelta(days=365)
        )
    
    async def demo_initialization(self):
        """Demonstrate progress tracking initialization"""
        print("🚀 Cultural Transformation Progress Tracking Demo")
        print("=" * 60)
        
        # Create sample transformation
        transformation = self.create_sample_transformation()
        
        print(f"\n📋 Initializing Progress Tracking")
        print(f"Transformation ID: {transformation.id}")
        print(f"Organization: {transformation.organization_id}")
        print(f"Target Completion: {transformation.target_completion.strftime('%Y-%m-%d')}")
        
        # Initialize progress tracking
        tracking_data = self.engine.initialize_progress_tracking(transformation)
        
        print(f"\n✅ Progress Tracking Initialized")
        print(f"Initial Progress: {tracking_data['overall_progress']:.1f}%")
        print(f"Milestones Created: {len(tracking_data['milestones'])}")
        print(f"Metrics Initialized: {len(tracking_data['metrics'])}")
        
        return tracking_data
    
    async def demo_milestone_tracking(self):
        """Demonstrate milestone progress tracking"""
        print(f"\n📊 Milestone Progress Tracking")
        print("-" * 40)
        
        # Get milestones
        tracking_data = self.engine.progress_data[self.transformation_id]
        milestones = list(tracking_data['milestones'].values())
        
        print(f"\nAvailable Milestones:")
        for i, milestone in enumerate(milestones, 1):
            print(f"{i}. {milestone.name} ({milestone.milestone_type.value})")
            print(f"   Target: {milestone.target_date.strftime('%Y-%m-%d')}")
            print(f"   Status: {milestone.status.value}")
        
        # Update first milestone (Assessment)
        assessment_milestone = milestones[0]
        print(f"\n🎯 Updating Assessment Milestone Progress")
        
        progress_updates = [
            {"progress_percentage": 25.0, "status": "in_progress"},
            {"progress_percentage": 60.0, "status": "in_progress"},
            {"progress_percentage": 100.0, "status": "completed", 
             "validation_results": {
                 "Culture mapping completed": {"met": True, "evidence": "Comprehensive culture map created"},
                 "Subcultures identified": {"met": True, "evidence": "3 distinct subcultures mapped"},
                 "Health metrics established": {"met": True, "evidence": "Baseline metrics defined"}
             }}
        ]
        
        for i, update in enumerate(progress_updates, 1):
            print(f"\n   Step {i}: Updating to {update['progress_percentage']}%")
            updated_milestone = self.engine.track_milestone_progress(
                self.transformation_id, assessment_milestone.id, update
            )
            print(f"   Status: {updated_milestone.status.value}")
            if updated_milestone.completion_date:
                print(f"   Completed: {updated_milestone.completion_date.strftime('%Y-%m-%d %H:%M')}")
            
            # Show overall progress update
            updated_tracking = self.engine.progress_data[self.transformation_id]
            print(f"   Overall Progress: {updated_tracking['overall_progress']:.1f}%")
        
        # Start planning milestone
        planning_milestone = milestones[1]
        print(f"\n🎯 Starting Planning Milestone")
        self.engine.track_milestone_progress(
            self.transformation_id, planning_milestone.id,
            {"progress_percentage": 40.0, "status": "in_progress"}
        )
        print(f"   Planning milestone now at 40% progress")
    
    async def demo_metrics_tracking(self):
        """Demonstrate metrics tracking"""
        print(f"\n📈 Progress Metrics Tracking")
        print("-" * 40)
        
        # Get current metrics
        tracking_data = self.engine.progress_data[self.transformation_id]
        metrics = list(tracking_data['metrics'].values())
        
        print(f"\nCurrent Metrics:")
        for metric in metrics:
            print(f"• {metric.name}: {metric.current_value:.1f}/{metric.target_value:.1f} {metric.unit}")
            print(f"  Completion: {metric.completion_percentage:.1f}% | Trend: {metric.trend}")
        
        # Simulate metric updates over time
        print(f"\n📊 Simulating Metric Updates Over Time")
        
        metric_scenarios = [
            {
                "week": 1,
                "updates": {
                    f"engagement_{self.transformation_id}": 35.0,
                    f"alignment_{self.transformation_id}": 25.0,
                    f"behavior_{self.transformation_id}": 15.0
                }
            },
            {
                "week": 4,
                "updates": {
                    f"engagement_{self.transformation_id}": 55.0,
                    f"alignment_{self.transformation_id}": 45.0,
                    f"behavior_{self.transformation_id}": 30.0
                }
            },
            {
                "week": 8,
                "updates": {
                    f"engagement_{self.transformation_id}": 70.0,
                    f"alignment_{self.transformation_id}": 65.0,
                    f"behavior_{self.transformation_id}": 50.0
                }
            },
            {
                "week": 12,
                "updates": {
                    f"engagement_{self.transformation_id}": 80.0,
                    f"alignment_{self.transformation_id}": 75.0,
                    f"behavior_{self.transformation_id}": 65.0
                }
            }
        ]
        
        for scenario in metric_scenarios:
            print(f"\n   Week {scenario['week']} Updates:")
            updated_metrics = self.engine.update_progress_metrics(
                self.transformation_id, scenario['updates']
            )
            
            for metric in updated_metrics:
                print(f"   • {metric.name}: {metric.current_value:.1f} ({metric.trend})")
                print(f"     Completion: {metric.completion_percentage:.1f}%")
    
    async def demo_progress_reporting(self):
        """Demonstrate progress reporting"""
        print(f"\n📋 Progress Report Generation")
        print("-" * 40)
        
        # Generate comprehensive progress report
        report = self.engine.generate_progress_report(self.transformation_id)
        
        print(f"\n📊 Progress Report - {report.report_date.strftime('%Y-%m-%d %H:%M')}")
        print(f"Overall Progress: {report.overall_progress:.1f}%")
        
        print(f"\n🎯 Milestone Status:")
        for milestone in report.milestones:
            status_icon = "✅" if milestone.status.value == "completed" else "🔄" if milestone.status.value == "in_progress" else "⏳"
            overdue_flag = " ⚠️ OVERDUE" if milestone.is_overdue else ""
            print(f"   {status_icon} {milestone.name}: {milestone.progress_percentage:.1f}%{overdue_flag}")
        
        print(f"\n📈 Key Metrics:")
        for metric in report.metrics:
            trend_icon = "📈" if metric.trend == "improving" else "📉" if metric.trend == "declining" else "➡️"
            print(f"   {trend_icon} {metric.name}: {metric.current_value:.1f}/{metric.target_value:.1f}")
        
        print(f"\n🏆 Achievements:")
        for achievement in report.achievements:
            print(f"   • {achievement}")
        
        print(f"\n⚠️ Challenges:")
        for challenge in report.challenges:
            print(f"   • {challenge}")
        
        print(f"\n🎯 Next Steps:")
        for step in report.next_steps:
            print(f"   • {step}")
        
        print(f"\n🚨 Risk Indicators:")
        for risk_type, risk_level in report.risk_indicators.items():
            risk_icon = "🔴" if risk_level > 50 else "🟡" if risk_level > 20 else "🟢"
            print(f"   {risk_icon} {risk_type.replace('_', ' ').title()}: {risk_level:.1f}%")
        
        print(f"\n💡 Recommendations:")
        for recommendation in report.recommendations:
            print(f"   • {recommendation}")
    
    async def demo_dashboard_creation(self):
        """Demonstrate dashboard creation"""
        print(f"\n📊 Progress Dashboard Creation")
        print("-" * 40)
        
        # Create progress dashboard
        dashboard = self.engine.create_progress_dashboard(self.transformation_id)
        
        print(f"\n🎛️ Dashboard Overview")
        print(f"Health Score: {dashboard.overall_health_score:.1f}/100")
        print(f"Generated: {dashboard.dashboard_date.strftime('%Y-%m-%d %H:%M')}")
        
        print(f"\n📈 Progress Charts:")
        charts = dashboard.progress_charts
        print(f"   Overall Progress: {charts['overall_progress']:.1f}%")
        
        milestone_data = charts['milestone_completion']
        print(f"   Milestones - Completed: {milestone_data['completed']}, In Progress: {milestone_data['in_progress']}")
        
        print(f"\n📅 Milestone Timeline:")
        for milestone_info in dashboard.milestone_timeline[:3]:  # Show first 3
            status_icon = "✅" if milestone_info['status'] == "completed" else "🔄"
            print(f"   {status_icon} {milestone_info['name']}")
            print(f"      Target: {milestone_info['target_date'][:10]}")
            print(f"      Progress: {milestone_info['progress']:.1f}%")
        
        print(f"\n🚨 Active Alerts:")
        if dashboard.alert_indicators:
            for alert in dashboard.alert_indicators:
                severity_icon = "🔴" if alert['severity'] == "high" else "🟡" if alert['severity'] == "medium" else "🟢"
                print(f"   {severity_icon} {alert['message']}")
        else:
            print("   ✅ No active alerts")
        
        print(f"\n📝 Executive Summary:")
        print(f"   {dashboard.executive_summary}")
    
    async def demo_milestone_validation(self):
        """Demonstrate milestone validation"""
        print(f"\n✅ Milestone Validation")
        print("-" * 40)
        
        # Get completed milestone for validation
        tracking_data = self.engine.progress_data[self.transformation_id]
        completed_milestones = [
            m for m in tracking_data['milestones'].values() 
            if m.status.value == "completed"
        ]
        
        if completed_milestones:
            milestone = completed_milestones[0]
            print(f"\n🔍 Validating: {milestone.name}")
            
            validation_result = self.engine.validate_milestone_completion(
                self.transformation_id, milestone.id
            )
            
            print(f"All Criteria Met: {'✅ Yes' if validation_result['all_criteria_met'] else '❌ No'}")
            print(f"Final Status: {validation_result['status']}")
            
            print(f"\n📋 Validation Results:")
            for criterion, result in validation_result['validation_results'].items():
                status_icon = "✅" if result['met'] else "❌"
                print(f"   {status_icon} {criterion}")
                if 'evidence' in result:
                    print(f"      Evidence: {result['evidence']}")
    
    async def demo_alerts_system(self):
        """Demonstrate alerts system"""
        print(f"\n🚨 Alerts System Demo")
        print("-" * 40)
        
        # Create an overdue milestone to trigger alert
        tracking_data = self.engine.progress_data[self.transformation_id]
        milestones = list(tracking_data['milestones'].values())
        
        # Find an incomplete milestone and make it overdue
        incomplete_milestone = None
        for milestone in milestones:
            if milestone.status.value != "completed":
                incomplete_milestone = milestone
                break
        
        if incomplete_milestone:
            print(f"\n⏰ Creating Overdue Scenario")
            print(f"Making '{incomplete_milestone.name}' overdue...")
            
            # Set target date in the past
            incomplete_milestone.target_date = datetime.now() - timedelta(days=2)
            
            # Update milestone to trigger alert check
            self.engine.track_milestone_progress(
                self.transformation_id, incomplete_milestone.id,
                {"progress_percentage": 30.0}
            )
            
            print(f"   Milestone is now overdue by 2 days")
        
        # Create a declining metric scenario
        print(f"\n📉 Creating Metric Decline Scenario")
        declining_metric_id = f"engagement_{self.transformation_id}"
        
        # Update metric with declining value
        self.engine.update_progress_metrics(
            self.transformation_id, {declining_metric_id: 25.0}
        )
        print(f"   Employee engagement dropped to 25%")
        
        # Show all alerts
        alerts = self.engine.alerts.get(self.transformation_id, [])
        active_alerts = [alert for alert in alerts if not alert.resolved_date]
        
        print(f"\n🚨 Active Alerts ({len(active_alerts)}):")
        for alert in active_alerts:
            severity_icon = "🔴" if alert.severity == "high" else "🟡" if alert.severity == "medium" else "🟢"
            print(f"   {severity_icon} [{alert.alert_type.upper()}] {alert.message}")
            print(f"      Created: {alert.created_date.strftime('%Y-%m-%d %H:%M')}")
            print(f"      Severity: {alert.severity}")
    
    async def run_complete_demo(self):
        """Run the complete progress tracking demo"""
        try:
            # Initialize system
            await self.demo_initialization()
            
            # Demonstrate milestone tracking
            await self.demo_milestone_tracking()
            
            # Demonstrate metrics tracking
            await self.demo_metrics_tracking()
            
            # Generate progress report
            await self.demo_progress_reporting()
            
            # Create dashboard
            await self.demo_dashboard_creation()
            
            # Validate milestones
            await self.demo_milestone_validation()
            
            # Demonstrate alerts
            await self.demo_alerts_system()
            
            print(f"\n🎉 Progress Tracking Demo Complete!")
            print("=" * 60)
            print("The system successfully demonstrated:")
            print("✅ Progress tracking initialization")
            print("✅ Milestone progress monitoring")
            print("✅ Metrics tracking and trending")
            print("✅ Comprehensive progress reporting")
            print("✅ Interactive dashboard creation")
            print("✅ Milestone validation")
            print("✅ Automated alerting system")
            
        except Exception as e:
            print(f"\n❌ Demo Error: {str(e)}")
            raise


async def main():
    """Main demo function"""
    demo = ProgressTrackingDemo()
    await demo.run_complete_demo()


if __name__ == "__main__":
    asyncio.run(main())