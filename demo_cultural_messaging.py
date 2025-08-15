"""
Cultural Messaging Framework Demo

Demonstrates the cultural messaging framework capabilities including
strategy creation, message customization, and effectiveness tracking.
"""

import asyncio
import json
from datetime import datetime, timedelta
from scrollintel.engines.cultural_messaging_engine import CulturalMessagingEngine
from scrollintel.models.cultural_messaging_models import MessageType, AudienceType, MessageChannel


async def demo_cultural_messaging():
    """Demonstrate cultural messaging framework capabilities"""
    print("🎯 Cultural Messaging Framework Demo")
    print("=" * 50)
    
    # Initialize engine
    engine = CulturalMessagingEngine()
    
    # 1. Create Messaging Strategy
    print("\n1. Creating Messaging Strategy...")
    
    cultural_vision = """
    To be the world's most innovative technology company that empowers 
    people and organizations to achieve extraordinary results through 
    cutting-edge solutions and exceptional service.
    """
    
    core_values = [
        "Innovation",
        "Customer Focus", 
        "Integrity",
        "Excellence",
        "Collaboration"
    ]
    
    audience_data = [
        {
            "name": "Executive Leadership",
            "type": "leadership_team",
            "size": 15,
            "characteristics": {
                "seniority": "C-level",
                "decision_making": "strategic",
                "communication_style": "formal"
            },
            "preferences": {
                "channel": "email",
                "timing": "morning",
                "format": "detailed"
            },
            "cultural_context": {
                "location": "headquarters",
                "influence": "high"
            }
        },
        {
            "name": "Engineering Teams",
            "type": "department_specific",
            "size": 250,
            "characteristics": {
                "technical": True,
                "collaborative": True,
                "innovation_focused": True
            },
            "preferences": {
                "channel": "slack",
                "timing": "afternoon",
                "format": "concise"
            },
            "cultural_context": {
                "department": "engineering",
                "work_style": "agile"
            }
        },
        {
            "name": "Sales Organization",
            "type": "department_specific",
            "size": 120,
            "characteristics": {
                "customer_focused": True,
                "results_driven": True,
                "relationship_oriented": True
            },
            "preferences": {
                "channel": "email",
                "timing": "morning",
                "format": "actionable"
            },
            "cultural_context": {
                "department": "sales",
                "performance_driven": True
            }
        },
        {
            "name": "Remote Workforce",
            "type": "remote_workers",
            "size": 180,
            "characteristics": {
                "distributed": True,
                "self_directed": True,
                "technology_savvy": True
            },
            "preferences": {
                "channel": "slack",
                "timing": "flexible",
                "format": "engaging"
            },
            "cultural_context": {
                "location": "distributed",
                "connection_needs": "high"
            }
        },
        {
            "name": "New Hires",
            "type": "new_hires",
            "size": 45,
            "characteristics": {
                "learning_oriented": True,
                "culture_absorbing": True,
                "enthusiastic": True
            },
            "preferences": {
                "channel": "email",
                "timing": "morning",
                "format": "educational"
            },
            "cultural_context": {
                "tenure": "0-6 months",
                "support_needs": "high"
            }
        }
    ]
    
    strategy = engine.create_messaging_strategy(
        organization_id="tech_innovators_inc",
        cultural_vision=cultural_vision.strip(),
        core_values=core_values,
        audience_data=audience_data
    )
    
    print(f"✅ Strategy created: {strategy.id}")
    print(f"   📊 Key themes identified: {', '.join(strategy.key_themes)}")
    print(f"   👥 Audience segments: {len(strategy.audience_segments)}")
    print(f"   📝 Message templates: {len(strategy.message_templates)}")
    
    # 2. Create Cultural Messages
    print("\n2. Creating Cultural Messages...")
    
    # Vision communication message
    vision_message = engine.create_cultural_message(
        title="Our Vision for the Future",
        content="""
        Team,
        
        I want to share our exciting vision for the future. We are committed to becoming 
        the world's most innovative technology company that empowers people and organizations 
        to achieve extraordinary results.
        
        This vision is built on our core values of Innovation, Customer Focus, Integrity, 
        Excellence, and Collaboration. These values guide every decision we make and every 
        solution we create.
        
        Our innovation drives breakthrough solutions. Our customer focus ensures we deliver 
        exceptional value. Our integrity builds trust with all stakeholders. Our pursuit 
        of excellence sets us apart. Our collaboration multiplies our impact.
        
        Together, we will transform industries and create lasting positive change in the world.
        
        Best regards,
        Leadership Team
        """,
        message_type=MessageType.VISION_COMMUNICATION,
        cultural_themes=["innovation", "customer_focus", "excellence", "collaboration"],
        key_values=["Innovation", "Customer Focus", "Integrity", "Excellence", "Collaboration"]
    )
    
    print(f"✅ Vision message created: {vision_message.title}")
    print(f"   🎯 Alignment score: {vision_message.metadata.get('alignment_score', 0):.2f}")
    
    # Values reinforcement message
    values_message = engine.create_cultural_message(
        title="Celebrating Innovation Excellence",
        content="""
        Team,
        
        I'm excited to share a fantastic example of our Innovation value in action!
        
        Our engineering team recently developed a breakthrough solution that reduces 
        customer onboarding time by 75%. This innovation directly demonstrates how 
        our commitment to excellence and customer focus creates real value.
        
        This achievement shows what's possible when we combine technical innovation 
        with deep customer understanding. The team collaborated across departments, 
        maintained the highest standards of integrity, and delivered exceptional results.
        
        I encourage everyone to think about how you can bring innovation to your work. 
        What processes can you improve? What new approaches can you try? How can you 
        better serve our customers?
        
        Let's continue building on this momentum and creating extraordinary results together.
        
        Congratulations to the entire team!
        """,
        message_type=MessageType.VALUES_REINFORCEMENT,
        cultural_themes=["innovation", "excellence", "customer_focus", "collaboration"],
        key_values=["Innovation", "Excellence", "Customer Focus", "Collaboration", "Integrity"]
    )
    
    print(f"✅ Values message created: {values_message.title}")
    print(f"   🎯 Alignment score: {values_message.metadata.get('alignment_score', 0):.2f}")
    
    # 3. Customize Messages for Different Audiences
    print("\n3. Customizing Messages for Audiences...")
    
    # Get audience IDs
    leadership_audience = None
    engineering_audience = None
    sales_audience = None
    remote_audience = None
    
    for audience_id, audience in engine.audience_profiles.items():
        if audience.name == "Executive Leadership":
            leadership_audience = audience_id
        elif audience.name == "Engineering Teams":
            engineering_audience = audience_id
        elif audience.name == "Sales Organization":
            sales_audience = audience_id
        elif audience.name == "Remote Workforce":
            remote_audience = audience_id
    
    # Customize vision message for leadership
    leadership_customization = engine.customize_message_for_audience(
        message_id=vision_message.id,
        audience_id=leadership_audience,
        channel=MessageChannel.EMAIL,
        delivery_timing=datetime.now() + timedelta(hours=1)
    )
    
    print(f"✅ Leadership customization created")
    print(f"   📧 Channel: {leadership_customization.channel.value}")
    print(f"   🎯 Personalization score: {leadership_customization.personalization_data.get('personalization_score', 0):.2f}")
    
    # Customize values message for engineering
    engineering_customization = engine.customize_message_for_audience(
        message_id=values_message.id,
        audience_id=engineering_audience,
        channel=MessageChannel.SLACK,
        delivery_timing=datetime.now() + timedelta(hours=2)
    )
    
    print(f"✅ Engineering customization created")
    print(f"   💬 Channel: {engineering_customization.channel.value}")
    print(f"   📝 Content preview: {engineering_customization.customized_content[:100]}...")
    
    # 4. Track Message Effectiveness
    print("\n4. Tracking Message Effectiveness...")
    
    # Simulate engagement data for leadership audience
    leadership_engagement = {
        "channel": "email",
        "views": 15,
        "clicks": 12,
        "shares": 3,
        "responses": 8,
        "sentiment_score": 0.85,
        "time_spent": 180,
        "actions_taken": 10,
        "discussions_started": 2,
        "follow_up_actions": 6,
        "feedback": {
            "alignment_indicators": {
                "cultural_resonance": 0.90,
                "message_clarity": 0.88,
                "relevance": 0.92
            },
            "comments": [
                "Clear vision alignment with strategic goals",
                "Inspiring and actionable message",
                "Well-articulated value proposition"
            ]
        }
    }
    
    leadership_effectiveness = engine.track_message_effectiveness(
        message_id=vision_message.id,
        audience_id=leadership_audience,
        engagement_data=leadership_engagement
    )
    
    print(f"✅ Leadership effectiveness tracked")
    print(f"   📊 Effectiveness score: {leadership_effectiveness.effectiveness_score:.2f}")
    print(f"   🎯 Cultural alignment: {leadership_effectiveness.cultural_alignment_score:.2f}")
    print(f"   📈 Behavior indicators: {len(leadership_effectiveness.behavior_change_indicators)} metrics")
    
    # Simulate engagement data for engineering audience
    engineering_engagement = {
        "channel": "slack",
        "views": 220,
        "clicks": 85,
        "shares": 15,
        "responses": 45,
        "sentiment_score": 0.78,
        "time_spent": 95,
        "actions_taken": 60,
        "discussions_started": 12,
        "follow_up_actions": 25,
        "feedback": {
            "alignment_indicators": {
                "cultural_resonance": 0.82,
                "message_clarity": 0.85,
                "relevance": 0.88
            },
            "comments": [
                "Great example of technical innovation",
                "Motivating to see our work recognized",
                "Clear connection to customer value"
            ]
        }
    }
    
    engineering_effectiveness = engine.track_message_effectiveness(
        message_id=values_message.id,
        audience_id=engineering_audience,
        engagement_data=engineering_engagement
    )
    
    print(f"✅ Engineering effectiveness tracked")
    print(f"   📊 Effectiveness score: {engineering_effectiveness.effectiveness_score:.2f}")
    print(f"   🎯 Cultural alignment: {engineering_effectiveness.cultural_alignment_score:.2f}")
    print(f"   💡 Recommendations: {len(engineering_effectiveness.recommendations)} suggestions")
    
    # 5. Create and Launch Messaging Campaign
    print("\n5. Creating Messaging Campaign...")
    
    campaign = engine.create_messaging_campaign(
        name="Q1 Cultural Transformation Initiative",
        description="Reinforce our cultural values and vision alignment across all teams",
        cultural_objectives=[
            "Increase innovation mindset adoption",
            "Strengthen customer focus behaviors",
            "Enhance cross-team collaboration",
            "Build cultural pride and engagement"
        ],
        target_audiences=[leadership_audience, engineering_audience, sales_audience, remote_audience],
        messages=[vision_message.id, values_message.id],
        duration_days=45
    )
    
    print(f"✅ Campaign created: {campaign.name}")
    print(f"   🎯 Objectives: {len(campaign.cultural_objectives)} goals")
    print(f"   👥 Target audiences: {len(campaign.target_audiences)} segments")
    print(f"   📅 Duration: {campaign.start_date.strftime('%Y-%m-%d')} to {campaign.end_date.strftime('%Y-%m-%d')}")
    
    # 6. Optimize Messaging Strategy
    print("\n6. Optimizing Messaging Strategy...")
    
    performance_data = {
        "avg_effectiveness": 0.72,
        "audience_breakdown": {
            "executive_leadership": 0.88,
            "engineering_teams": 0.75,
            "sales_organization": 0.68,
            "remote_workforce": 0.58,
            "new_hires": 0.82
        },
        "channel_breakdown": {
            "email": 0.78,
            "slack": 0.68,
            "intranet": 0.65,
            "town_hall": 0.85
        },
        "theme_breakdown": {
            "innovation": 0.80,
            "customer_focus": 0.75,
            "excellence": 0.70,
            "collaboration": 0.68,
            "integrity": 0.72
        },
        "improvement_areas": ["engagement", "personalization", "timing"]
    }
    
    optimizations = engine.optimize_messaging_strategy(
        strategy_id=strategy.id,
        performance_data=performance_data
    )
    
    print(f"✅ Strategy optimized")
    print(f"   📈 Content improvements: {len(optimizations['content_improvements'])} suggestions")
    print(f"   🎯 Audience targeting: Enhanced segmentation recommended")
    print(f"   📱 Channel optimization: Multi-channel approach suggested")
    print(f"   ⏰ Timing recommendations: Optimal scheduling identified")
    
    # 7. Display Analytics Summary
    print("\n7. Analytics Summary")
    print("=" * 30)
    
    print(f"📊 Overall Performance:")
    print(f"   • Total messages created: {len(engine.message_history)}")
    print(f"   • Active campaigns: {len([c for c in engine.active_campaigns.values() if c.status in ['planned', 'active']])}")
    print(f"   • Audience segments: {len(engine.audience_profiles)}")
    print(f"   • Average effectiveness: {performance_data['avg_effectiveness']:.1%}")
    
    print(f"\n🎯 Top Performing Segments:")
    for audience, score in sorted(performance_data['audience_breakdown'].items(), 
                                key=lambda x: x[1], reverse=True)[:3]:
        print(f"   • {audience.replace('_', ' ').title()}: {score:.1%}")
    
    print(f"\n📱 Channel Performance:")
    for channel, score in sorted(performance_data['channel_breakdown'].items(), 
                               key=lambda x: x[1], reverse=True):
        print(f"   • {channel.title()}: {score:.1%}")
    
    print(f"\n💡 Key Recommendations:")
    for i, improvement in enumerate(optimizations['content_improvements'][:3], 1):
        print(f"   {i}. {improvement}")
    
    print(f"\n🚀 Next Steps:")
    print(f"   • Implement personalization improvements for remote workforce")
    print(f"   • Leverage town hall format for high-impact messages")
    print(f"   • Create micro-segments for better targeting")
    print(f"   • Develop interactive content for higher engagement")
    
    print(f"\n✨ Cultural Messaging Framework Demo Complete!")
    print(f"   The system successfully demonstrated comprehensive cultural")
    print(f"   communication capabilities with measurable effectiveness tracking.")


if __name__ == "__main__":
    asyncio.run(demo_cultural_messaging())