"""
Demo: Credibility and Trust Management for Board Executive Mastery

This demo showcases the credibility building and trust management capabilities
for board and executive relationships in the ScrollIntel system.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, Any

from scrollintel.engines.credibility_building_engine import CredibilityBuildingEngine
from scrollintel.engines.trust_management_engine import TrustManagementEngine
from scrollintel.models.credibility_models import (
    CredibilityLevel, TrustLevel, StakeholderProfile, RelationshipEvent
)


class CredibilityTrustManagementDemo:
    """Demo class for credibility and trust management functionality"""
    
    def __init__(self):
        self.credibility_engine = CredibilityBuildingEngine()
        self.trust_engine = TrustManagementEngine()
        
    def create_sample_stakeholder_profiles(self) -> Dict[str, StakeholderProfile]:
        """Create sample stakeholder profiles for demonstration"""
        profiles = {
            "board_chair": StakeholderProfile(
                id="board_chair_001",
                name="Margaret Chen",
                role="Board Chair",
                background="Former Fortune 500 CEO with 25 years experience",
                values=["strategic vision", "operational excellence", "stakeholder value"],
                communication_preferences={"format": "executive summary", "frequency": "weekly"},
                decision_making_style="strategic",
                influence_level=0.95,
                credibility_assessment=None,
                trust_assessment=None,
                relationship_events=[]
            ),
            "tech_board_member": StakeholderProfile(
                id="tech_member_001",
                name="David Rodriguez",
                role="Technology Board Member",
                background="Former CTO of major tech company",
                values=["innovation", "technical excellence", "scalability"],
                communication_preferences={"format": "technical deep-dive", "frequency": "bi-weekly"},
                decision_making_style="analytical",
                influence_level=0.85,
                credibility_assessment=None,
                trust_assessment=None,
                relationship_events=[]
            ),
            "finance_board_member": StakeholderProfile(
                id="finance_member_001",
                name="Sarah Johnson",
                role="Finance Board Member",
                background="Former CFO with strong financial background",
                values=["fiscal responsibility", "transparency", "risk management"],
                communication_preferences={"format": "financial metrics", "frequency": "monthly"},
                decision_making_style="data-driven",
                influence_level=0.80,
                credibility_assessment=None,
                trust_assessment=None,
                relationship_events=[]
            )
        }
        return profiles
    
    def create_sample_evidence_data(self, stakeholder_type: str) -> Dict[str, Any]:
        """Create sample evidence data for credibility assessment"""
        base_data = {
            "expertise": {
                "years_experience": 12,
                "certifications": 8,
                "domain_knowledge_score": 0.85,
                "evidence": [
                    "Led successful digital transformation initiatives",
                    "Published thought leadership articles",
                    "Recognized industry expert"
                ],
                "historical_scores": [0.75, 0.78, 0.82, 0.85]
            },
            "track_record": {
                "success_rate": 0.88,
                "project_count": 45,
                "evidence": [
                    "Delivered 88% of projects on time and budget",
                    "Achieved 15% cost savings in infrastructure",
                    "Improved system reliability by 40%"
                ],
                "historical_scores": [0.80, 0.83, 0.86, 0.88]
            },
            "transparency": {
                "transparency_score": 0.82,
                "evidence": [
                    "Regular board reporting with full disclosure",
                    "Proactive communication of challenges",
                    "Open about technical limitations"
                ],
                "historical_scores": [0.75, 0.78, 0.80, 0.82]
            },
            "consistency": {
                "consistency_score": 0.90,
                "evidence": [
                    "Consistent messaging across all stakeholders",
                    "Reliable performance over time",
                    "Predictable communication style"
                ],
                "historical_scores": [0.85, 0.87, 0.88, 0.90]
            },
            "communication": {
                "communication_effectiveness": 0.85,
                "evidence": [
                    "Clear and concise board presentations",
                    "Effective technical explanations",
                    "Strong listening skills"
                ],
                "historical_scores": [0.80, 0.82, 0.84, 0.85]
            },
            "results_delivery": {
                "delivery_score": 0.92,
                "evidence": [
                    "Exceeded performance targets",
                    "Delivered major initiatives ahead of schedule",
                    "Achieved cost optimization goals"
                ],
                "historical_scores": [0.88, 0.89, 0.91, 0.92]
            },
            "strategic_insight": {
                "strategic_thinking_score": 0.87,
                "evidence": [
                    "Provided strategic technology roadmap",
                    "Identified market opportunities",
                    "Aligned technology with business strategy"
                ],
                "historical_scores": [0.82, 0.84, 0.86, 0.87]
            },
            "problem_solving": {
                "problem_solving_score": 0.89,
                "evidence": [
                    "Resolved complex technical challenges",
                    "Innovative solutions to business problems",
                    "Effective crisis management"
                ],
                "historical_scores": [0.85, 0.86, 0.88, 0.89]
            }
        }
        
        # Adjust data based on stakeholder type
        if stakeholder_type == "tech_board_member":
            base_data["expertise"]["domain_knowledge_score"] = 0.95
            base_data["strategic_insight"]["strategic_thinking_score"] = 0.92
        elif stakeholder_type == "finance_board_member":
            base_data["transparency"]["transparency_score"] = 0.95
            base_data["results_delivery"]["delivery_score"] = 0.95
        
        return base_data
    
    def create_sample_relationship_data(self, stakeholder_type: str) -> Dict[str, Any]:
        """Create sample relationship data for trust assessment"""
        base_data = {
            "reliability": {
                "commitment_fulfillment_rate": 0.90,
                "consistency_score": 0.88,
                "punctuality_score": 0.95,
                "evidence": [
                    "Met all board meeting commitments",
                    "Delivered on promised timelines",
                    "Consistent follow-through on action items"
                ],
                "historical_scores": [0.85, 0.87, 0.89, 0.91],
                "last_interaction": datetime.now().isoformat()
            },
            "competence": {
                "technical_competence": 0.92,
                "problem_solving_ability": 0.88,
                "decision_quality": 0.85,
                "evidence": [
                    "Demonstrated deep technical expertise",
                    "Made sound strategic decisions",
                    "Effectively solved complex problems"
                ],
                "historical_scores": [0.82, 0.85, 0.87, 0.88],
                "last_interaction": datetime.now().isoformat()
            },
            "benevolence": {
                "stakeholder_focus": 0.85,
                "support_provided": 0.82,
                "consideration_shown": 0.88,
                "evidence": [
                    "Actively supported board initiatives",
                    "Considered stakeholder interests in decisions",
                    "Provided helpful guidance to team"
                ],
                "historical_scores": [0.80, 0.82, 0.84, 0.85],
                "last_interaction": datetime.now().isoformat()
            },
            "integrity": {
                "honesty_score": 0.95,
                "ethical_behavior": 0.92,
                "transparency_level": 0.88,
                "evidence": [
                    "Always honest in communications",
                    "Demonstrated ethical decision-making",
                    "Transparent about challenges and risks"
                ],
                "historical_scores": [0.90, 0.91, 0.92, 0.92],
                "last_interaction": datetime.now().isoformat()
            },
            "relationship_history": [
                {
                    "date": "2024-01-15",
                    "event": "Initial board presentation",
                    "outcome": "very positive",
                    "notes": "Impressed with technical depth and strategic thinking"
                },
                {
                    "date": "2024-02-20",
                    "event": "Quarterly review meeting",
                    "outcome": "positive",
                    "notes": "Good progress on key initiatives"
                },
                {
                    "date": "2024-03-10",
                    "event": "Crisis management discussion",
                    "outcome": "excellent",
                    "notes": "Handled crisis with transparency and competence"
                }
            ]
        }
        
        # Adjust data based on stakeholder type
        if stakeholder_type == "board_chair":
            base_data["benevolence"]["stakeholder_focus"] = 0.92
            base_data["integrity"]["transparency_level"] = 0.95
        elif stakeholder_type == "tech_board_member":
            base_data["competence"]["technical_competence"] = 0.98
            base_data["competence"]["problem_solving_ability"] = 0.95
        
        return base_data
    
    def create_sample_relationship_events(self, stakeholder_id: str) -> list:
        """Create sample relationship events"""
        events = [
            RelationshipEvent(
                id=f"event_{stakeholder_id}_001",
                stakeholder_id=stakeholder_id,
                event_type="board_presentation",
                description="Quarterly technology update presentation",
                date=datetime.now() - timedelta(days=30),
                credibility_impact=0.15,
                trust_impact=0.12,
                lessons_learned=[
                    "Technical depth was well-received",
                    "Strategic alignment resonated with board"
                ],
                follow_up_actions=[
                    "Schedule deep-dive session on AI strategy",
                    "Provide detailed roadmap document"
                ]
            ),
            RelationshipEvent(
                id=f"event_{stakeholder_id}_002",
                stakeholder_id=stakeholder_id,
                event_type="crisis_management",
                description="Handled security incident with transparency",
                date=datetime.now() - timedelta(days=15),
                credibility_impact=0.20,
                trust_impact=0.25,
                lessons_learned=[
                    "Transparency during crisis builds trust",
                    "Quick response and clear communication are crucial"
                ],
                follow_up_actions=[
                    "Implement enhanced security measures",
                    "Regular security briefings to board"
                ]
            ),
            RelationshipEvent(
                id=f"event_{stakeholder_id}_003",
                stakeholder_id=stakeholder_id,
                event_type="strategic_planning",
                description="Contributed to 3-year strategic plan",
                date=datetime.now() - timedelta(days=7),
                credibility_impact=0.10,
                trust_impact=0.15,
                lessons_learned=[
                    "Strategic thinking is highly valued",
                    "Technology-business alignment is key"
                ],
                follow_up_actions=[
                    "Develop detailed implementation timeline",
                    "Identify key technology investments"
                ]
            )
        ]
        return events
    
    async def demonstrate_credibility_assessment(self):
        """Demonstrate credibility assessment functionality"""
        print("\n" + "="*80)
        print("CREDIBILITY ASSESSMENT DEMONSTRATION")
        print("="*80)
        
        stakeholder_profiles = self.create_sample_stakeholder_profiles()
        
        for stakeholder_type, profile in stakeholder_profiles.items():
            print(f"\n--- Assessing Credibility with {profile.name} ({profile.role}) ---")
            
            evidence_data = self.create_sample_evidence_data(stakeholder_type)
            assessment = self.credibility_engine.assess_credibility(profile.id, evidence_data)
            
            print(f"Overall Credibility Score: {assessment.overall_score:.2f}")
            print(f"Credibility Level: {assessment.level.value}")
            print(f"Key Strengths: {', '.join(assessment.strengths)}")
            print(f"Improvement Areas: {', '.join(assessment.improvement_areas)}")
            
            print("\nCredibility Metrics by Factor:")
            for metric in assessment.metrics:
                print(f"  {metric.factor.value}: {metric.score:.2f} ({metric.trend})")
            
            # Develop credibility plan
            target_level = CredibilityLevel.EXCEPTIONAL
            plan = self.credibility_engine.develop_credibility_plan(assessment, target_level)
            
            print(f"\nCredibility Building Plan:")
            print(f"  Target Level: {plan.target_level.value}")
            print(f"  Timeline: {plan.timeline}")
            print(f"  Number of Actions: {len(plan.actions)}")
            print(f"  Key Actions:")
            for action in plan.actions[:3]:  # Show first 3 actions
                print(f"    - {action.title}: {action.description}")
    
    async def demonstrate_trust_assessment(self):
        """Demonstrate trust assessment functionality"""
        print("\n" + "="*80)
        print("TRUST ASSESSMENT DEMONSTRATION")
        print("="*80)
        
        stakeholder_profiles = self.create_sample_stakeholder_profiles()
        
        for stakeholder_type, profile in stakeholder_profiles.items():
            print(f"\n--- Assessing Trust with {profile.name} ({profile.role}) ---")
            
            relationship_data = self.create_sample_relationship_data(stakeholder_type)
            assessment = self.trust_engine.assess_trust(profile.id, relationship_data)
            
            print(f"Overall Trust Score: {assessment.overall_score:.2f}")
            print(f"Trust Level: {assessment.level.value}")
            print(f"Trust Drivers: {', '.join(assessment.trust_drivers)}")
            print(f"Trust Barriers: {', '.join(assessment.trust_barriers)}")
            
            print("\nTrust Metrics by Dimension:")
            for metric in assessment.metrics:
                print(f"  {metric.dimension}: {metric.score:.2f} ({metric.trend})")
            
            # Develop trust building strategy
            target_level = TrustLevel.COMPLETE_TRUST
            strategy = self.trust_engine.develop_trust_building_strategy(assessment, target_level)
            
            print(f"\nTrust Building Strategy:")
            print(f"  Target Level: {strategy.target_trust_level.value}")
            print(f"  Timeline: {strategy.timeline}")
            print(f"  Key Actions:")
            for action in strategy.key_actions[:3]:  # Show first 3 actions
                print(f"    - {action}")
    
    async def demonstrate_progress_tracking(self):
        """Demonstrate progress tracking functionality"""
        print("\n" + "="*80)
        print("PROGRESS TRACKING DEMONSTRATION")
        print("="*80)
        
        # Use board chair as example
        profile = self.create_sample_stakeholder_profiles()["board_chair"]
        
        # Create credibility plan
        evidence_data = self.create_sample_evidence_data("board_chair")
        credibility_assessment = self.credibility_engine.assess_credibility(profile.id, evidence_data)
        credibility_plan = self.credibility_engine.develop_credibility_plan(
            credibility_assessment, CredibilityLevel.EXCEPTIONAL
        )
        
        # Create trust strategy
        relationship_data = self.create_sample_relationship_data("board_chair")
        trust_assessment = self.trust_engine.assess_trust(profile.id, relationship_data)
        trust_strategy = self.trust_engine.develop_trust_building_strategy(
            trust_assessment, TrustLevel.COMPLETE_TRUST
        )
        
        # Create recent events
        recent_events = self.create_sample_relationship_events(profile.id)
        
        print(f"--- Tracking Progress with {profile.name} ---")
        
        # Track credibility progress
        credibility_progress = self.credibility_engine.track_credibility_progress(
            credibility_plan, recent_events
        )
        
        print(f"\nCredibility Progress:")
        print(f"  Actions Completed: {credibility_progress['actions_completed']}")
        print(f"  Actions In Progress: {credibility_progress['actions_in_progress']}")
        print(f"  Recent Impacts: {len(credibility_progress['recent_impacts'])}")
        print(f"  Recommendations: {len(credibility_progress['recommendations'])}")
        
        # Track trust progress
        trust_progress = self.trust_engine.track_trust_progress(trust_strategy, recent_events)
        
        print(f"\nTrust Progress:")
        print(f"  Trust Trend: {trust_progress['trust_trend']}")
        print(f"  Recent Trust Impacts: {len(trust_progress['recent_trust_impacts'])}")
        print(f"  Relationship Quality:")
        quality = trust_progress['relationship_quality_indicators']
        print(f"    Interaction Frequency: {quality['interaction_frequency']}")
        print(f"    Positive Interactions: {quality['positive_interactions']}")
        print(f"    Communication Quality: {quality['communication_quality']}")
    
    async def demonstrate_trust_recovery(self):
        """Demonstrate trust recovery functionality"""
        print("\n" + "="*80)
        print("TRUST RECOVERY DEMONSTRATION")
        print("="*80)
        
        # Simulate a trust breach scenario
        profile = self.create_sample_stakeholder_profiles()["tech_board_member"]
        
        # Create a damaged trust scenario
        damaged_relationship_data = self.create_sample_relationship_data("tech_board_member")
        # Simulate trust damage
        damaged_relationship_data["reliability"]["commitment_fulfillment_rate"] = 0.4
        damaged_relationship_data["integrity"]["transparency_level"] = 0.3
        
        trust_assessment = self.trust_engine.assess_trust(profile.id, damaged_relationship_data)
        
        print(f"--- Trust Recovery for {profile.name} ---")
        print(f"Current Trust Level: {trust_assessment.level.value}")
        print(f"Trust Score: {trust_assessment.overall_score:.2f}")
        
        # Create recovery plan
        breach_description = "Failed to deliver critical system upgrade on promised timeline and was not transparent about delays"
        target_level = TrustLevel.TRUSTING
        
        recovery_plan = self.trust_engine.create_trust_recovery_plan(
            profile.id, breach_description, trust_assessment, target_level
        )
        
        print(f"\nTrust Recovery Plan:")
        print(f"  Recovery Strategy: {recovery_plan.recovery_strategy}")
        print(f"  Timeline: {recovery_plan.timeline}")
        print(f"  Target Level: {recovery_plan.target_trust_level.value}")
        
        print(f"\nImmediate Actions:")
        for action in recovery_plan.immediate_actions:
            print(f"    - {action}")
        
        print(f"\nLong-term Actions:")
        for action in recovery_plan.long_term_actions[:3]:  # Show first 3
            print(f"    - {action}")
        
        print(f"\nSuccess Metrics:")
        for metric in recovery_plan.success_metrics[:3]:  # Show first 3
            print(f"    - {metric}")
    
    async def demonstrate_effectiveness_measurement(self):
        """Demonstrate trust effectiveness measurement"""
        print("\n" + "="*80)
        print("TRUST EFFECTIVENESS MEASUREMENT")
        print("="*80)
        
        stakeholder_profiles = list(self.create_sample_stakeholder_profiles().values())
        
        # Simulate trust assessments for all stakeholders
        for profile in stakeholder_profiles:
            relationship_data = self.create_sample_relationship_data(profile.id.split('_')[0])
            trust_assessment = self.trust_engine.assess_trust(profile.id, relationship_data)
            profile.trust_assessment = trust_assessment
        
        effectiveness = self.trust_engine.measure_trust_effectiveness(stakeholder_profiles)
        
        print(f"Overall Trust Management Effectiveness:")
        print(f"  Total Stakeholders: {effectiveness['total_stakeholders']}")
        print(f"  Average Trust Score: {effectiveness['average_trust_score']:.2f}")
        print(f"  High Trust Relationships: {effectiveness['high_trust_relationships']}")
        print(f"  At-Risk Relationships: {effectiveness['at_risk_relationships']}")
        
        print(f"\nTrust Distribution:")
        for level, count in effectiveness['trust_distribution'].items():
            print(f"  {level}: {count}")
        
        print(f"\nImprovement Opportunities:")
        for opportunity in effectiveness['improvement_opportunities']:
            print(f"  - {opportunity}")
        
        print(f"\nSuccess Stories:")
        for story in effectiveness['success_stories']:
            print(f"  - {story}")
    
    async def demonstrate_credibility_report(self):
        """Demonstrate credibility report generation"""
        print("\n" + "="*80)
        print("CREDIBILITY REPORT DEMONSTRATION")
        print("="*80)
        
        stakeholder_profiles = self.create_sample_stakeholder_profiles()
        assessments = []
        
        # Create assessments for all stakeholders
        for stakeholder_type, profile in stakeholder_profiles.items():
            evidence_data = self.create_sample_evidence_data(stakeholder_type)
            assessment = self.credibility_engine.assess_credibility(profile.id, evidence_data)
            assessments.append(assessment)
        
        report = self.credibility_engine.generate_credibility_report(assessments)
        
        print(f"Credibility Status Report:")
        print(f"  Report Date: {report.report_date.strftime('%Y-%m-%d %H:%M')}")
        print(f"  Overall Credibility Score: {report.overall_credibility_score:.2f}")
        print(f"  Stakeholders Assessed: {len(report.stakeholder_assessments)}")
        
        print(f"\nKey Achievements:")
        for achievement in report.key_achievements:
            print(f"  - {achievement}")
        
        print(f"\nAreas for Improvement:")
        for area in report.areas_for_improvement:
            print(f"  - {area}")
        
        print(f"\nRecommended Actions:")
        for action in report.recommended_actions:
            print(f"  - {action}")
        
        print(f"\nTrend Analysis:")
        trend = report.trend_analysis
        print(f"  Overall Trend: {trend.get('overall_trend', 'N/A')}")
        print(f"  Strongest Factors: {', '.join(trend.get('strongest_factors', []))}")
        print(f"  Weakest Factors: {', '.join(trend.get('weakest_factors', []))}")
    
    async def run_complete_demo(self):
        """Run the complete credibility and trust management demo"""
        print("SCROLLINTEL BOARD EXECUTIVE MASTERY")
        print("Credibility and Trust Management System Demo")
        print("="*80)
        
        try:
            await self.demonstrate_credibility_assessment()
            await self.demonstrate_trust_assessment()
            await self.demonstrate_progress_tracking()
            await self.demonstrate_trust_recovery()
            await self.demonstrate_effectiveness_measurement()
            await self.demonstrate_credibility_report()
            
            print("\n" + "="*80)
            print("DEMO COMPLETED SUCCESSFULLY")
            print("="*80)
            print("\nKey Capabilities Demonstrated:")
            print("✓ Credibility assessment and building")
            print("✓ Trust measurement and management")
            print("✓ Progress tracking and monitoring")
            print("✓ Trust recovery planning")
            print("✓ Effectiveness measurement")
            print("✓ Comprehensive reporting")
            print("\nThe system is ready for board and executive relationship management!")
            
        except Exception as e:
            print(f"\nDemo encountered an error: {str(e)}")
            import traceback
            traceback.print_exc()


async def main():
    """Main function to run the demo"""
    demo = CredibilityTrustManagementDemo()
    await demo.run_complete_demo()


if __name__ == "__main__":
    asyncio.run(main())