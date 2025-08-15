#!/usr/bin/env python3
"""
Complete Analytics and Marketing Infrastructure Demo
Demonstrates Google Analytics integration, user behavior tracking, conversion funnels,
A/B testing, marketing attribution, and user segmentation
"""

import asyncio
import json
import random
from datetime import datetime, timedelta
from typing import Dict, List, Any
import logging

from scrollintel.core.analytics_tracker import analytics_tracker, EventType
from scrollintel.core.conversion_funnel import funnel_analyzer
from scrollintel.core.ab_testing import ab_testing_framework, ExperimentStatus, VariantType
from scrollintel.core.marketing_attribution import marketing_attribution, AttributionModel, CampaignStatus
from scrollintel.core.user_segmentation import user_segmentation, SegmentType

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AnalyticsMarketingDemo:
    """Complete analytics and marketing infrastructure demonstration"""
    
    def __init__(self):
        self.demo_users = []
        self.demo_campaigns = []
        self.demo_experiments = []
        
    async def run_complete_demo(self):
        """Run complete analytics and marketing demo"""
        logger.info("🚀 Starting Complete Analytics & Marketing Infrastructure Demo")
        
        try:
            # 1. Setup demo data
            await self.setup_demo_data()
            
            # 2. Demonstrate analytics tracking
            await self.demo_analytics_tracking()
            
            # 3. Demonstrate conversion funnels
            await self.demo_conversion_funnels()
            
            # 4. Demonstrate A/B testing
            await self.demo_ab_testing()
            
            # 5. Demonstrate marketing attribution
            await self.demo_marketing_attribution()
            
            # 6. Demonstrate user segmentation
            await self.demo_user_segmentation()
            
            # 7. Generate comprehensive reports
            await self.generate_comprehensive_reports()
            
            logger.info("✅ Complete Analytics & Marketing Demo completed successfully!")
            
        except Exception as e:
            logger.error(f"❌ Demo failed: {str(e)}")
            raise
    
    async def setup_demo_data(self):
        """Setup demo users, campaigns, and experiments"""
        logger.info("📊 Setting up demo data...")
        
        # Create demo users
        self.demo_users = [
            {
                "user_id": f"user_{i:03d}",
                "user_type": random.choice(["free", "trial", "paid"]),
                "subscription_tier": random.choice(["basic", "premium", "enterprise"]),
                "signup_date": datetime.utcnow() - timedelta(days=random.randint(1, 90)),
                "properties": {
                    "company_size": random.choice(["startup", "small", "medium", "enterprise"]),
                    "industry": random.choice(["tech", "finance", "healthcare", "retail", "other"]),
                    "use_case": random.choice(["analytics", "reporting", "automation", "insights"])
                }
            }
            for i in range(100)
        ]
        
        # Create demo marketing campaigns
        campaign_configs = [
            {
                "name": "Google Search Campaign",
                "type": "search",
                "source": "google",
                "medium": "cpc",
                "budget": 5000.0,
                "content": "search_ads",
                "term": "data analytics platform"
            },
            {
                "name": "Facebook Social Campaign",
                "type": "social",
                "source": "facebook",
                "medium": "social",
                "budget": 3000.0,
                "content": "video_ad",
                "term": None
            },
            {
                "name": "Email Newsletter Campaign",
                "type": "email",
                "source": "newsletter",
                "medium": "email",
                "budget": 500.0,
                "content": "weekly_newsletter",
                "term": None
            },
            {
                "name": "LinkedIn Professional Campaign",
                "type": "social",
                "source": "linkedin",
                "medium": "social",
                "budget": 2000.0,
                "content": "professional_ad",
                "term": "business intelligence"
            }
        ]
        
        for config in campaign_configs:
            campaign_id = await marketing_attribution.create_campaign(
                name=config["name"],
                description=f"Demo {config['type']} campaign",
                campaign_type=config["type"],
                source=config["source"],
                medium=config["medium"],
                budget=config["budget"],
                start_date=datetime.utcnow() - timedelta(days=30),
                content=config["content"],
                term=config["term"]
            )
            
            # Activate campaign
            campaign = marketing_attribution.campaigns[campaign_id]
            campaign.status = CampaignStatus.ACTIVE
            
            self.demo_campaigns.append({
                "campaign_id": campaign_id,
                "config": config
            })
        
        # Create demo A/B test experiments
        experiment_configs = [
            {
                "name": "Landing Page CTA Button Test",
                "description": "Test different CTA button colors and text",
                "hypothesis": "Red button with 'Start Free Trial' will increase conversions",
                "success_metrics": ["conversion_rate", "click_through_rate"],
                "variants": [
                    {"name": "Control - Blue Button", "type": "control", "traffic_allocation": 50.0,
                     "configuration": {"button_color": "blue", "button_text": "Get Started"}},
                    {"name": "Treatment - Red Button", "type": "treatment", "traffic_allocation": 50.0,
                     "configuration": {"button_color": "red", "button_text": "Start Free Trial"}}
                ]
            },
            {
                "name": "Onboarding Flow Test",
                "description": "Test simplified vs detailed onboarding",
                "hypothesis": "Simplified onboarding will improve completion rates",
                "success_metrics": ["onboarding_completion_rate", "time_to_activation"],
                "variants": [
                    {"name": "Control - Detailed", "type": "control", "traffic_allocation": 50.0,
                     "configuration": {"flow_type": "detailed", "steps": 5}},
                    {"name": "Treatment - Simplified", "type": "treatment", "traffic_allocation": 50.0,
                     "configuration": {"flow_type": "simplified", "steps": 3}}
                ]
            }
        ]
        
        for config in experiment_configs:
            experiment_id = await ab_testing_framework.create_experiment(
                name=config["name"],
                description=config["description"],
                hypothesis=config["hypothesis"],
                success_metrics=config["success_metrics"],
                variants=config["variants"]
            )
            
            # Start experiment
            await ab_testing_framework.start_experiment(experiment_id)
            
            self.demo_experiments.append({
                "experiment_id": experiment_id,
                "config": config
            })
        
        logger.info(f"✅ Created {len(self.demo_users)} users, {len(self.demo_campaigns)} campaigns, {len(self.demo_experiments)} experiments")
    
    async def demo_analytics_tracking(self):
        """Demonstrate analytics tracking functionality"""
        logger.info("📈 Demonstrating analytics tracking...")
        
        # Simulate user activity over the past 30 days
        for day in range(30):
            date = datetime.utcnow() - timedelta(days=day)
            daily_users = random.sample(self.demo_users, random.randint(20, 60))
            
            for user in daily_users:
                user_id = user["user_id"]
                session_id = f"session_{user_id}_{day}"
                
                # Track page views
                pages = ["/", "/dashboard", "/analytics", "/agents", "/upload", "/settings", "/pricing"]
                for _ in range(random.randint(2, 8)):
                    page = random.choice(pages)
                    await analytics_tracker.track_page_view(
                        user_id=user_id,
                        session_id=session_id,
                        page_url=f"https://scrollintel.com{page}",
                        page_title=f"ScrollIntel - {page.strip('/').title() or 'Home'}",
                        user_agent="Mozilla/5.0 (Demo Browser)",
                        ip_address="192.168.1.1",
                        referrer="https://google.com" if page == "/" else f"https://scrollintel.com{random.choice(pages)}"
                    )
                
                # Track user events
                events = [
                    ("button_click", {"button": "signup", "page": "/"}),
                    ("feature_used", {"feature": "data_upload", "file_type": "csv"}),
                    ("agent_interaction", {"agent": "data_scientist", "query_type": "analysis"}),
                    ("dashboard_viewed", {"dashboard_type": "analytics", "widgets": 5}),
                    ("report_generated", {"report_type": "insights", "data_points": 1000}),
                    ("export_data", {"format": "pdf", "report_type": "summary"})
                ]
                
                for _ in range(random.randint(1, 5)):
                    event_name, properties = random.choice(events)
                    properties.update({
                        "user_type": user["user_type"],
                        "subscription_tier": user["subscription_tier"]
                    })
                    
                    await analytics_tracker.track_event(
                        user_id=user_id,
                        session_id=session_id,
                        event_name=event_name,
                        properties=properties,
                        page_url="https://scrollintel.com/dashboard",
                        user_agent="Mozilla/5.0 (Demo Browser)",
                        ip_address="192.168.1.1"
                    )
                
                # Some users convert
                if random.random() < 0.15:  # 15% conversion rate
                    conversion_types = ["trial_started", "subscription_created", "upgrade_completed"]
                    conversion_type = random.choice(conversion_types)
                    conversion_value = {"trial_started": 0, "subscription_created": 99, "upgrade_completed": 199}[conversion_type]
                    
                    await analytics_tracker.track_conversion(
                        user_id=user_id,
                        conversion_type=conversion_type,
                        value=conversion_value,
                        properties={
                            "plan": random.choice(["basic", "premium", "enterprise"]),
                            "billing_cycle": random.choice(["monthly", "annual"])
                        }
                    )
        
        # Get analytics summary
        summary = await analytics_tracker.get_analytics_summary(30)
        logger.info(f"📊 Analytics Summary: {summary['unique_users']} users, {summary['total_events']} events, {summary['conversions']} conversions ({summary['conversion_rate']:.1f}% rate)")
    
    async def demo_conversion_funnels(self):
        """Demonstrate conversion funnel analysis"""
        logger.info("🔄 Demonstrating conversion funnels...")
        
        # Simulate user journeys through funnels
        for user in random.sample(self.demo_users, 80):
            user_id = user["user_id"]
            session_id = f"funnel_session_{user_id}"
            
            # User onboarding funnel
            funnel_steps = [
                ("page_view", {}, "/"),
                ("user_signup", {"method": "email"}, "/signup"),
                ("onboarding_completed", {"steps_completed": 5}, "/onboarding"),
                ("first_analysis", {"data_type": "csv"}, "/dashboard"),
                ("user_activated", {"features_used": 3}, "/dashboard")
            ]
            
            # Users drop off at different stages
            completion_probability = 0.8
            
            for event_name, properties, page_url in funnel_steps:
                if random.random() < completion_probability:
                    await funnel_analyzer.track_user_journey(
                        user_id=user_id,
                        session_id=session_id,
                        event_name=event_name,
                        event_properties=properties,
                        page_url=page_url
                    )
                    completion_probability *= 0.85  # Decreasing probability
                else:
                    break  # User drops off
        
        # Analyze funnel performance
        funnel_analysis = await funnel_analyzer.analyze_funnel_performance("user_onboarding", 30)
        logger.info(f"🔄 Onboarding Funnel: {funnel_analysis.total_users} users, {len(funnel_analysis.optimization_suggestions)} suggestions")
        
        # Get funnel summary
        funnel_summary = await funnel_analyzer.get_funnel_summary(30)
        logger.info(f"📈 Funnel Summary: {len(funnel_summary)} funnels analyzed")
    
    async def demo_ab_testing(self):
        """Demonstrate A/B testing functionality"""
        logger.info("🧪 Demonstrating A/B testing...")
        
        # Assign users to experiments and simulate results
        for experiment in self.demo_experiments:
            experiment_id = experiment["experiment_id"]
            experiment_config = experiment["config"]
            
            # Assign users to experiment
            test_users = random.sample(self.demo_users, 50)
            
            for user in test_users:
                user_id = user["user_id"]
                session_id = f"ab_session_{user_id}"
                
                # Assign to variant
                variant_id = await ab_testing_framework.assign_user_to_experiment(
                    user_id=user_id,
                    experiment_id=experiment_id,
                    session_id=session_id,
                    user_properties=user["properties"]
                )
                
                if variant_id:
                    # Simulate different conversion rates for variants
                    is_treatment = "treatment" in variant_id.lower()
                    base_conversion_rate = 0.12
                    treatment_lift = 0.03  # 3% lift for treatment
                    
                    conversion_rate = base_conversion_rate + (treatment_lift if is_treatment else 0)
                    
                    # Record results for success metrics
                    for metric in experiment_config["success_metrics"]:
                        if metric == "conversion_rate":
                            metric_value = 1.0 if random.random() < conversion_rate else 0.0
                        elif metric == "click_through_rate":
                            metric_value = 1.0 if random.random() < (0.25 + (0.05 if is_treatment else 0)) else 0.0
                        elif metric == "onboarding_completion_rate":
                            metric_value = 1.0 if random.random() < (0.70 + (0.10 if is_treatment else 0)) else 0.0
                        elif metric == "time_to_activation":
                            metric_value = random.uniform(120, 600) - (60 if is_treatment else 0)  # seconds
                        else:
                            metric_value = random.uniform(0, 1)
                        
                        await ab_testing_framework.record_experiment_result(
                            user_id=user_id,
                            experiment_id=experiment_id,
                            metric_name=metric,
                            metric_value=metric_value
                        )
        
        # Analyze experiment results
        for experiment in self.demo_experiments:
            experiment_id = experiment["experiment_id"]
            analyses = await ab_testing_framework.analyze_experiment(experiment_id)
            
            logger.info(f"🧪 Experiment '{experiment['config']['name']}':")
            for metric_analysis_key, analysis in analyses.items():
                logger.info(f"  - {analysis.metric_name}: {analysis.recommendation}")
        
        # Get experiment dashboard
        dashboard = await ab_testing_framework.get_experiment_dashboard()
        logger.info(f"📊 A/B Testing Dashboard: {dashboard['running_experiments']} running, {dashboard['total_users_in_experiments']} users in experiments")
    
    async def demo_marketing_attribution(self):
        """Demonstrate marketing attribution"""
        logger.info("🎯 Demonstrating marketing attribution...")
        
        # Simulate marketing touchpoints and conversions
        for user in random.sample(self.demo_users, 70):
            user_id = user["user_id"]
            
            # Create user journey with multiple touchpoints
            touchpoint_count = random.randint(1, 4)
            
            for i in range(touchpoint_count):
                session_id = f"attribution_session_{user_id}_{i}"
                campaign = random.choice(self.demo_campaigns)
                
                # Create UTM parameters
                utm_parameters = {
                    "utm_source": campaign["config"]["source"],
                    "utm_medium": campaign["config"]["medium"],
                    "utm_campaign": campaign["config"]["name"].lower().replace(" ", "_"),
                    "utm_content": campaign["config"]["content"],
                    "utm_term": campaign["config"]["term"]
                }
                
                # Track touchpoint
                await marketing_attribution.track_touchpoint(
                    user_id=user_id,
                    session_id=session_id,
                    page_url=f"https://scrollintel.com/?utm_source={utm_parameters['utm_source']}&utm_medium={utm_parameters['utm_medium']}",
                    referrer=f"https://{campaign['config']['source']}.com",
                    user_agent="Mozilla/5.0 (Demo Browser)",
                    ip_address="192.168.1.1",
                    utm_parameters=utm_parameters
                )
            
            # Some users convert
            if random.random() < 0.20:  # 20% conversion rate
                conversion_types = ["subscription_created", "trial_started", "demo_requested"]
                conversion_type = random.choice(conversion_types)
                conversion_values = {"subscription_created": 99, "trial_started": 0, "demo_requested": 0}
                
                await marketing_attribution.track_conversion(
                    user_id=user_id,
                    session_id=f"conversion_session_{user_id}",
                    conversion_type=conversion_type,
                    conversion_value=conversion_values[conversion_type],
                    attribution_model=AttributionModel.LAST_TOUCH
                )
        
        # Generate attribution reports
        for model in [AttributionModel.FIRST_TOUCH, AttributionModel.LAST_TOUCH, AttributionModel.LINEAR]:
            report = await marketing_attribution.generate_attribution_report(model, 30)
            logger.info(f"🎯 {model.value.title()} Attribution: ${report.roi_analysis['total_revenue']:.2f} revenue, {report.roi_analysis['overall_roi']:.1f}% ROI")
        
        # Get marketing dashboard
        dashboard = await marketing_attribution.get_marketing_dashboard(30)
        logger.info(f"📊 Marketing Dashboard: {dashboard['active_campaigns']} campaigns, {dashboard['total_touchpoints']} touchpoints")
    
    async def demo_user_segmentation(self):
        """Demonstrate user segmentation and cohort analysis"""
        logger.info("👥 Demonstrating user segmentation...")
        
        # Update user profiles with activity data
        for user in self.demo_users:
            user_id = user["user_id"]
            
            # Create activity events
            events = []
            activity_level = random.choice(["low", "medium", "high"])
            event_count = {"low": 5, "medium": 15, "high": 30}[activity_level]
            
            for i in range(event_count):
                event_date = user["signup_date"] + timedelta(days=random.randint(0, 60))
                events.append({
                    "event_name": random.choice(["page_view", "feature_used", "data_uploaded", "report_generated"]),
                    "timestamp": event_date,
                    "session_id": f"segment_session_{user_id}_{i}",
                    "properties": {"activity_level": activity_level}
                })
            
            # Update user profile
            profile = await user_segmentation.update_user_profile(
                user_id=user_id,
                events=events,
                properties=user["properties"]
            )
            
            # Manually set some calculated values for demo
            profile.engagement_level = activity_level
            profile.lifecycle_stage = random.choice(["new", "active", "engaged", "at_risk", "churned"])
        
        # Create custom segments
        custom_segments = [
            {
                "name": "High-Value Enterprise Users",
                "description": "Enterprise users with high engagement",
                "segment_type": SegmentType.BEHAVIORAL,
                "criteria": {
                    "engagement_level": "high",
                    "properties": {"company_size": "enterprise"}
                }
            },
            {
                "name": "Trial Users at Risk",
                "description": "Trial users with low engagement",
                "segment_type": SegmentType.BEHAVIORAL,
                "criteria": {
                    "engagement_level": "low",
                    "properties": {"user_type": "trial"}
                }
            }
        ]
        
        for segment_config in custom_segments:
            segment_id = await user_segmentation.create_custom_segment(
                name=segment_config["name"],
                description=segment_config["description"],
                segment_type=segment_config["segment_type"],
                criteria=segment_config["criteria"]
            )
            logger.info(f"👥 Created segment '{segment_config['name']}' with ID: {segment_id}")
        
        # Perform cohort analysis
        acquisition_cohorts = [c for c in user_segmentation.cohorts.values() 
                             if c.name == "Weekly Acquisition Cohorts"]
        
        if acquisition_cohorts:
            cohort_id = acquisition_cohorts[0].cohort_id
            analysis = await user_segmentation.perform_cohort_analysis(cohort_id, 8)
            logger.info(f"📊 Cohort Analysis: {len(analysis.user_counts)} cohorts, {len(analysis.insights)} insights")
        
        # Get segmentation dashboard
        dashboard = await user_segmentation.get_segmentation_dashboard()
        logger.info(f"👥 Segmentation Dashboard: {dashboard['total_users']} users, {dashboard['total_segments']} segments")
    
    async def generate_comprehensive_reports(self):
        """Generate comprehensive analytics and marketing reports"""
        logger.info("📋 Generating comprehensive reports...")
        
        # Analytics summary report
        analytics_summary = await analytics_tracker.get_analytics_summary(30)
        
        # Funnel summary report
        funnel_summary = await funnel_analyzer.get_funnel_summary(30)
        
        # A/B testing dashboard
        ab_dashboard = await ab_testing_framework.get_experiment_dashboard()
        
        # Marketing attribution report
        attribution_report = await marketing_attribution.generate_attribution_report(AttributionModel.LAST_TOUCH, 30)
        
        # User segmentation dashboard
        segmentation_dashboard = await user_segmentation.get_segmentation_dashboard()
        
        # Compile comprehensive report
        comprehensive_report = {
            "report_generated_at": datetime.utcnow().isoformat(),
            "period": "Last 30 days",
            "analytics_summary": {
                "total_users": analytics_summary["unique_users"],
                "total_events": analytics_summary["total_events"],
                "total_sessions": analytics_summary["sessions"],
                "conversion_rate": analytics_summary["conversion_rate"],
                "top_pages": analytics_summary["top_pages"][:5],
                "top_events": analytics_summary["top_events"][:5]
            },
            "conversion_funnels": {
                "total_funnels": len(funnel_summary),
                "funnel_performance": {
                    funnel_id: {
                        "total_users": data["total_users"],
                        "overall_conversion_rate": data["overall_conversion_rate"],
                        "optimization_suggestions": data["optimization_suggestions_count"]
                    }
                    for funnel_id, data in funnel_summary.items()
                }
            },
            "ab_testing": {
                "total_experiments": ab_dashboard["total_experiments"],
                "running_experiments": ab_dashboard["running_experiments"],
                "users_in_experiments": ab_dashboard["total_users_in_experiments"]
            },
            "marketing_attribution": {
                "total_revenue": attribution_report.roi_analysis["total_revenue"],
                "total_spend": attribution_report.roi_analysis["total_spend"],
                "roi": attribution_report.roi_analysis["overall_roi"],
                "roas": attribution_report.roi_analysis["overall_roas"],
                "top_campaigns": len(attribution_report.campaign_performance),
                "top_channels": len(attribution_report.channel_performance)
            },
            "user_segmentation": {
                "total_users": segmentation_dashboard["total_users"],
                "total_segments": segmentation_dashboard["total_segments"],
                "engagement_distribution": segmentation_dashboard["engagement_distribution"],
                "lifecycle_distribution": segmentation_dashboard["lifecycle_distribution"],
                "avg_behavioral_score": segmentation_dashboard["avg_behavioral_score"]
            }
        }
        
        # Save report to file
        report_filename = f"analytics_marketing_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_filename, 'w') as f:
            json.dump(comprehensive_report, f, indent=2, default=str)
        
        logger.info(f"📋 Comprehensive report saved to: {report_filename}")
        
        # Print summary
        logger.info("📊 COMPREHENSIVE ANALYTICS & MARKETING REPORT SUMMARY")
        logger.info("=" * 60)
        logger.info(f"📈 Analytics: {comprehensive_report['analytics_summary']['total_users']} users, {comprehensive_report['analytics_summary']['conversion_rate']:.1f}% conversion rate")
        logger.info(f"🔄 Funnels: {comprehensive_report['conversion_funnels']['total_funnels']} funnels analyzed")
        logger.info(f"🧪 A/B Tests: {comprehensive_report['ab_testing']['running_experiments']} running experiments")
        logger.info(f"🎯 Attribution: ${comprehensive_report['marketing_attribution']['total_revenue']:.2f} revenue, {comprehensive_report['marketing_attribution']['roi']:.1f}% ROI")
        logger.info(f"👥 Segmentation: {comprehensive_report['user_segmentation']['total_segments']} segments, {comprehensive_report['user_segmentation']['avg_behavioral_score']:.1f} avg score")
        logger.info("=" * 60)


async def main():
    """Run the complete analytics and marketing demo"""
    demo = AnalyticsMarketingDemo()
    await demo.run_complete_demo()


if __name__ == "__main__":
    asyncio.run(main())