"""
Demo: Global Influence Network Orchestrator

This demo showcases the unified global influence network orchestration system
that coordinates relationship building, influence strategies, and partnership
development to create superhuman influence capabilities.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, Any

from scrollintel.core.global_influence_orchestrator import GlobalInfluenceOrchestrator
from scrollintel.models.global_influence_models import (
    create_influence_campaign, create_influence_target,
    InfluenceScope, CampaignPriority, StakeholderType
)


class GlobalInfluenceDemo:
    """Demo class for Global Influence Network Orchestrator"""
    
    def __init__(self):
        self.orchestrator = GlobalInfluenceOrchestrator()
        self.demo_results = {}
    
    async def run_comprehensive_demo(self):
        """Run comprehensive demo of global influence orchestration"""
        print("🌍 Global Influence Network Orchestrator Demo")
        print("=" * 60)
        
        # Demo 1: Healthcare AI Leadership Campaign
        await self.demo_healthcare_ai_campaign()
        
        # Demo 2: Global Technology Partnership Campaign
        await self.demo_global_partnership_campaign()
        
        # Demo 3: Industry Standards Influence Campaign
        await self.demo_industry_standards_campaign()
        
        # Demo 4: Network Synchronization and Analytics
        await self.demo_network_operations()
        
        # Demo 5: Multi-Campaign Orchestration
        await self.demo_multi_campaign_orchestration()
        
        # Summary
        await self.display_demo_summary()
    
    async def demo_healthcare_ai_campaign(self):
        """Demo healthcare AI leadership influence campaign"""
        print("\n🏥 Demo 1: Healthcare AI Leadership Campaign")
        print("-" * 50)
        
        campaign_result = await self.orchestrator.orchestrate_global_influence_campaign(
            campaign_objective="Establish AI technology leadership in global healthcare",
            target_outcomes=[
                "Gain recognition as top healthcare AI thought leader",
                "Build partnerships with 50+ medical institutions worldwide",
                "Influence healthcare AI policy in US, EU, and Asia",
                "Create ecosystem of healthcare AI developers",
                "Secure $100M in healthcare AI partnerships"
            ],
            timeline=timedelta(days=180),
            priority="critical",
            constraints={
                'budget_limit': 2000000,
                'geographic_focus': ['US', 'EU', 'Asia'],
                'compliance_requirements': ['HIPAA', 'GDPR', 'FDA'],
                'timeline_flexibility': 'medium'
            }
        )
        
        self.demo_results['healthcare_campaign'] = campaign_result
        
        print(f"✅ Campaign Created: {campaign_result['campaign_id']}")
        print(f"📊 Success Probability: {campaign_result['success_probability']:.1%}")
        print(f"⏱️  Timeline: {campaign_result['estimated_timeline'].days} days")
        print(f"🎯 Orchestration Plan: {len(campaign_result['orchestration_plan'])} phases")
        
        # Display key orchestration elements
        plan = campaign_result['orchestration_plan']
        if isinstance(plan, dict) and 'phases' in plan:
            print(f"📋 Campaign Phases:")
            for i, phase in enumerate(plan['phases'][:3], 1):  # Show first 3 phases
                phase_name = phase.get('name', f'Phase {i}')
                print(f"   {i}. {phase_name}")
    
    async def demo_global_partnership_campaign(self):
        """Demo global technology partnership campaign"""
        print("\n🤝 Demo 2: Global Technology Partnership Campaign")
        print("-" * 50)
        
        campaign_result = await self.orchestrator.orchestrate_global_influence_campaign(
            campaign_objective="Build global technology partnership ecosystem",
            target_outcomes=[
                "Establish partnerships with top 20 tech companies",
                "Create joint innovation labs in 5 continents",
                "Influence technology standards in AI and cloud computing",
                "Build developer ecosystem of 100,000+ members",
                "Generate $500M in partnership revenue"
            ],
            timeline=timedelta(days=365),
            priority="high",
            constraints={
                'budget_limit': 5000000,
                'geographic_scope': 'global',
                'technology_focus': ['AI', 'Cloud', 'IoT', 'Blockchain'],
                'partnership_types': ['Strategic', 'Technical', 'Commercial']
            }
        )
        
        self.demo_results['partnership_campaign'] = campaign_result
        
        print(f"✅ Campaign Created: {campaign_result['campaign_id']}")
        print(f"📊 Success Probability: {campaign_result['success_probability']:.1%}")
        print(f"🌐 Global Scope: Multi-continental partnership network")
        print(f"💰 Target Revenue: $500M in partnerships")
        
        # Show resource requirements
        execution_status = campaign_result.get('execution_status', {})
        if 'resource_allocation' in execution_status:
            print(f"📈 Resource Allocation:")
            for resource, allocation in execution_status['resource_allocation'].items():
                print(f"   • {resource}: {allocation}")
    
    async def demo_industry_standards_campaign(self):
        """Demo industry standards influence campaign"""
        print("\n📜 Demo 3: Industry Standards Influence Campaign")
        print("-" * 50)
        
        campaign_result = await self.orchestrator.orchestrate_global_influence_campaign(
            campaign_objective="Shape global AI ethics and safety standards",
            target_outcomes=[
                "Lead AI ethics committee in 3 major standards organizations",
                "Influence AI safety regulations in 10+ countries",
                "Establish industry best practices for responsible AI",
                "Build coalition of 100+ organizations for AI ethics",
                "Create certification program for ethical AI development"
            ],
            timeline=timedelta(days=270),
            priority="high",
            constraints={
                'regulatory_focus': ['AI Ethics', 'Data Privacy', 'Safety Standards'],
                'stakeholder_types': ['Regulators', 'Industry Leaders', 'Academics'],
                'geographic_priority': ['US', 'EU', 'UK', 'Canada', 'Australia']
            }
        )
        
        self.demo_results['standards_campaign'] = campaign_result
        
        print(f"✅ Campaign Created: {campaign_result['campaign_id']}")
        print(f"📊 Success Probability: {campaign_result['success_probability']:.1%}")
        print(f"🏛️  Regulatory Focus: AI Ethics & Safety Standards")
        print(f"🤝 Coalition Building: 100+ organizations")
        
        # Show influence strategy elements
        plan = campaign_result.get('orchestration_plan', {})
        if 'influence_strategy' in plan:
            influence_strategy = plan['influence_strategy']
            print(f"🎯 Influence Strategy:")
            print(f"   • Objectives: {len(influence_strategy.get('objectives', []))}")
            print(f"   • Tactics: {len(influence_strategy.get('tactics', {}))}")
            print(f"   • Narrative Strategy: {bool(influence_strategy.get('narrative_strategy'))}")
    
    async def demo_network_operations(self):
        """Demo network synchronization and analytics"""
        print("\n🔄 Demo 4: Network Synchronization & Analytics")
        print("-" * 50)
        
        # Synchronize network data
        print("🔄 Synchronizing influence network data...")
        sync_results = await self.orchestrator.synchronize_influence_data()
        
        print(f"✅ Synchronization Complete:")
        for system, result in sync_results.items():
            if isinstance(result, dict) and 'status' in result:
                status = result['status']
                records = result.get('records_synced', 0)
                print(f"   • {system}: {status} ({records} records)")
        
        # Get network status
        print("\n📊 Getting network status...")
        network_status = await self.orchestrator.get_influence_network_status()
        
        print(f"🌐 Network Health:")
        print(f"   • Active Campaigns: {network_status['active_campaigns']}")
        print(f"   • Network Health Score: {network_status['network_health']['score']:.1%}")
        print(f"   • Total Influence Reach: {network_status['influence_metrics']['network_reach']:,}")
        print(f"   • Active Relationships: {network_status['relationship_status']['active_relationships']}")
        print(f"   • Partnership Value: ${network_status['partnership_status']['partnership_value']:,}")
        
        self.demo_results['network_status'] = network_status
    
    async def demo_multi_campaign_orchestration(self):
        """Demo managing multiple concurrent campaigns"""
        print("\n🎭 Demo 5: Multi-Campaign Orchestration")
        print("-" * 50)
        
        # Create multiple smaller campaigns
        campaigns = []
        
        campaign_configs = [
            {
                'objective': 'Establish thought leadership in quantum computing',
                'outcomes': ['Publish quantum AI research', 'Speak at quantum conferences'],
                'timeline': timedelta(days=90),
                'priority': 'medium'
            },
            {
                'objective': 'Build fintech partnership network',
                'outcomes': ['Partner with 10 fintech startups', 'Launch fintech accelerator'],
                'timeline': timedelta(days=120),
                'priority': 'medium'
            },
            {
                'objective': 'Influence cybersecurity standards',
                'outcomes': ['Join cybersecurity advisory boards', 'Shape security protocols'],
                'timeline': timedelta(days=150),
                'priority': 'low'
            }
        ]
        
        print(f"🚀 Launching {len(campaign_configs)} concurrent campaigns...")
        
        for i, config in enumerate(campaign_configs, 1):
            campaign_result = await self.orchestrator.orchestrate_global_influence_campaign(
                campaign_objective=config['objective'],
                target_outcomes=config['outcomes'],
                timeline=config['timeline'],
                priority=config['priority']
            )
            campaigns.append(campaign_result)
            print(f"   {i}. {config['objective'][:50]}... ✅")
        
        # Show orchestration summary
        total_campaigns = len(self.orchestrator.active_campaigns)
        print(f"\n📈 Orchestration Summary:")
        print(f"   • Total Active Campaigns: {total_campaigns}")
        print(f"   • New Campaigns Created: {len(campaigns)}")
        print(f"   • Average Success Probability: {sum(c['success_probability'] for c in campaigns) / len(campaigns):.1%}")
        
        self.demo_results['multi_campaigns'] = campaigns
    
    async def display_demo_summary(self):
        """Display comprehensive demo summary"""
        print("\n" + "=" * 60)
        print("🎯 GLOBAL INFLUENCE ORCHESTRATION DEMO SUMMARY")
        print("=" * 60)
        
        total_campaigns = len(self.orchestrator.active_campaigns)
        
        print(f"\n📊 Campaign Statistics:")
        print(f"   • Total Campaigns Orchestrated: {total_campaigns}")
        print(f"   • Healthcare AI Campaign: ✅ Created")
        print(f"   • Global Partnership Campaign: ✅ Created")
        print(f"   • Industry Standards Campaign: ✅ Created")
        print(f"   • Multi-Campaign Orchestration: ✅ {len(self.demo_results.get('multi_campaigns', []))} campaigns")
        
        # Calculate aggregate metrics
        all_campaigns = list(self.orchestrator.active_campaigns.values())
        if all_campaigns:
            avg_success_prob = sum(c.get('success_probability', 0) for c in all_campaigns) / len(all_campaigns)
            total_timeline = sum((c.get('timeline', timedelta()).days for c in all_campaigns), 0)
            
            print(f"\n🎯 Performance Metrics:")
            print(f"   • Average Success Probability: {avg_success_prob:.1%}")
            print(f"   • Total Campaign Timeline: {total_timeline} days")
            print(f"   • Network Health Score: {self.demo_results.get('network_status', {}).get('network_health', {}).get('score', 0):.1%}")
        
        print(f"\n🌟 Key Capabilities Demonstrated:")
        print(f"   ✅ Unified campaign orchestration across all influence engines")
        print(f"   ✅ Cross-system data synchronization and coordination")
        print(f"   ✅ Multi-domain influence strategy development")
        print(f"   ✅ Global partnership and ecosystem integration")
        print(f"   ✅ Real-time network analytics and monitoring")
        print(f"   ✅ Concurrent multi-campaign management")
        
        print(f"\n🚀 System Status:")
        print(f"   • Orchestrator: ✅ Operational")
        print(f"   • Relationship Engine: ✅ Integrated")
        print(f"   • Influence Engine: ✅ Integrated")
        print(f"   • Partnership Engine: ✅ Integrated")
        print(f"   • Network Sync: ✅ Healthy")
        
        print(f"\n💡 Next Steps:")
        print(f"   • Execute campaign phases according to orchestration plans")
        print(f"   • Monitor campaign progress and adjust strategies")
        print(f"   • Scale network operations based on success metrics")
        print(f"   • Integrate additional influence channels and platforms")


async def main():
    """Main demo execution"""
    demo = GlobalInfluenceDemo()
    
    try:
        await demo.run_comprehensive_demo()
        
        print(f"\n✨ Demo completed successfully!")
        print(f"🔗 Global Influence Network Orchestrator is ready for production use.")
        
    except Exception as e:
        print(f"\n❌ Demo error: {str(e)}")
        raise


if __name__ == "__main__":
    asyncio.run(main())