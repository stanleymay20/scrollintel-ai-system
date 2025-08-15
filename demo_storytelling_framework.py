"""
Storytelling Framework Demo

Demonstrates the storytelling framework capabilities including narrative strategy,
story creation, personalization, and impact measurement.
"""

import asyncio
import json
from datetime import datetime, timedelta
from scrollintel.engines.storytelling_engine import StorytellingEngine
from scrollintel.models.storytelling_models import (
    StoryType, NarrativeStructure, DeliveryFormat
)


async def demo_storytelling_framework():
    """Demonstrate storytelling framework capabilities"""
    print("📚 Storytelling Framework Demo")
    print("=" * 50)
    
    # Initialize engine
    engine = StorytellingEngine()
    
    # 1. Create Narrative Strategy
    print("\n1. Creating Narrative Strategy...")
    
    transformation_vision = """
    We envision a future where our organization becomes a beacon of innovation,
    where every team member feels empowered to contribute their unique talents,
    and where we consistently deliver exceptional value to our customers through
    collaborative excellence and continuous learning.
    """
    
    core_narratives = [
        "Innovation is not just what we do, it's who we are",
        "Every challenge is an opportunity for growth and breakthrough",
        "Our diverse perspectives create our competitive advantage",
        "Customer success is the measure of our success",
        "Continuous learning fuels continuous improvement",
        "Together we achieve what seems impossible alone"
    ]
    
    audience_preferences = {
        "leadership": {
            "complexity": "high",
            "format": "formal",
            "focus": "strategic_impact",
            "preferred_length": "detailed"
        },
        "engineering": {
            "complexity": "high",
            "format": "technical",
            "focus": "innovation_process",
            "preferred_length": "moderate"
        },
        "sales": {
            "complexity": "moderate",
            "format": "energetic",
            "focus": "customer_impact",
            "preferred_length": "concise"
        },
        "marketing": {
            "complexity": "moderate",
            "format": "creative",
            "focus": "brand_story",
            "preferred_length": "engaging"
        },
        "new_hires": {
            "complexity": "simple",
            "format": "welcoming",
            "focus": "culture_integration",
            "preferred_length": "comprehensive"
        }
    }
    
    strategy = engine.create_narrative_strategy(
        organization_id="innovative_tech_corp",
        transformation_vision=transformation_vision.strip(),
        core_narratives=core_narratives,
        audience_preferences=audience_preferences
    )
    
    print(f"✅ Narrative strategy created: {strategy.id}")
    print(f"   🎭 Story themes identified: {', '.join(strategy.story_themes)}")
    print(f"   👥 Character archetypes: {len(strategy.character_archetypes)}")
    print(f"   📋 Narrative guidelines: {len(strategy.narrative_guidelines)} categories")
    
    # Display character archetypes
    print(f"\n   Character Archetypes:")
    for archetype in strategy.character_archetypes:
        print(f"   • {archetype.name} ({archetype.role})")
        print(f"     Characteristics: {', '.join(archetype.characteristics.keys())}")
    
    # 2. Create Transformation Stories
    print("\n2. Creating Transformation Stories...")
    
    # Hero's Journey Story
    hero_story = engine.create_transformation_story(
        title="From Skeptic to Champion: Maria's Innovation Journey",
        story_type=StoryType.TRANSFORMATION_JOURNEY,
        narrative_structure=NarrativeStructure.HERO_JOURNEY,
        content="""
        Maria had been with the company for eight years, comfortable in her role as a 
        senior analyst. She was known for her meticulous attention to detail and 
        reliable delivery, but when the CEO announced the digital transformation 
        initiative, Maria felt a familiar knot in her stomach.
        
        "Another corporate buzzword initiative," she thought, remembering past failed 
        attempts at change. But this time felt different. The company was investing 
        in real training, bringing in mentors, and creating innovation labs.
        
        The turning point came during a workshop on design thinking. Maria found herself 
        sketching solutions to customer problems she'd never considered before. For the 
        first time in years, she felt genuinely excited about her work.
        
        The journey wasn't smooth. Maria struggled with imposter syndrome when presenting 
        to senior leadership. She made mistakes with new technologies. Some colleagues 
        questioned why she was "trying to be something she wasn't."
        
        But Maria had discovered something powerful: her analytical skills, combined with 
        creative thinking, could solve problems in ways neither approach could alone. 
        She began mentoring other skeptics, showing them that transformation wasn't about 
        becoming someone else—it was about becoming the fullest version of yourself.
        
        Today, Maria leads our customer experience innovation team. She's helped launch 
        three breakthrough products and has become one of our most effective change 
        champions. Her story reminds us that the greatest transformations often begin 
        with the greatest skeptics.
        
        The lesson: Innovation isn't about having all the answers—it's about having 
        the courage to ask better questions.
        """,
        cultural_themes=["transformation", "innovation", "courage", "growth"],
        key_messages=[
            "Transformation is possible for everyone",
            "Skepticism can become strength",
            "Combining different skills creates innovation",
            "Mentoring others amplifies your impact"
        ],
        target_outcomes=[
            "Inspire reluctant employees to embrace change",
            "Show that transformation builds on existing strengths",
            "Encourage peer mentoring and support",
            "Demonstrate the value of diverse thinking approaches"
        ]
    )
    
    print(f"✅ Hero's journey story created: {hero_story.title}")
    print(f"   📊 Narrative strength: {hero_story.metadata.get('narrative_strength', 0):.2f}")
    print(f"   💫 Emotional resonance: {hero_story.metadata.get('emotional_resonance', 0):.2f}")
    print(f"   🎯 Cultural alignment: {hero_story.metadata.get('cultural_alignment', 0):.2f}")
    
    # Success Story
    success_story = engine.create_transformation_story(
        title="The Impossible Deadline: How Team Phoenix Soared",
        story_type=StoryType.SUCCESS_STORY,
        narrative_structure=NarrativeStructure.CHALLENGE_ACTION_RESULT,
        content="""
        When our biggest client called with an "impossible" request—deliver a complete 
        platform redesign in six weeks instead of six months—most people thought we'd 
        have to say no. The timeline seemed unrealistic, the scope was massive, and 
        our team was already stretched thin.
        
        But Team Phoenix saw an opportunity. Led by Sarah from engineering, Marcus from 
        design, and Priya from product management, they proposed something radical: 
        what if we could use this challenge to prove our new collaborative approach?
        
        Instead of the traditional waterfall process, they created daily cross-functional 
        sprints. Designers and engineers worked side-by-side. Product managers became 
        real-time customer advocates. Every decision was made with the customer in the room.
        
        The breakthrough came in week three. While testing a prototype with actual users, 
        they discovered that 60% of the requested features weren't actually needed. The 
        customer wanted simplicity, not complexity. This insight allowed them to focus 
        on what truly mattered.
        
        The result? They delivered not just on time, but with a solution that exceeded 
        expectations. Customer satisfaction scores hit an all-time high. The client 
        signed a three-year extension. And most importantly, Team Phoenix had proven 
        that our values of collaboration, customer focus, and innovation weren't just 
        words on a wall—they were competitive advantages.
        
        Today, the "Phoenix Method" has been adopted across the organization. What started 
        as an impossible deadline became a blueprint for how we tackle every challenge.
        
        The lesson: When we truly collaborate and focus on customer value, we can 
        achieve what seems impossible.
        """,
        cultural_themes=["collaboration", "customer_focus", "innovation", "excellence"],
        key_messages=[
            "Collaboration creates breakthrough results",
            "Customer focus simplifies complex challenges",
            "Constraints can spark innovation",
            "Success methods can be scaled across the organization"
        ],
        target_outcomes=[
            "Celebrate collaborative achievement",
            "Reinforce customer-centric values",
            "Encourage teams to embrace challenging goals",
            "Promote the adoption of proven methodologies"
        ]
    )
    
    print(f"✅ Success story created: {success_story.title}")
    print(f"   📊 Narrative strength: {success_story.metadata.get('narrative_strength', 0):.2f}")
    print(f"   💫 Emotional resonance: {success_story.metadata.get('emotional_resonance', 0):.2f}")
    
    # Vision Narrative
    vision_story = engine.create_transformation_story(
        title="The Future We're Building Together",
        story_type=StoryType.VISION_NARRATIVE,
        narrative_structure=NarrativeStructure.THREE_ACT,
        content="""
        Imagine walking into our office five years from now. The first thing you notice 
        isn't the technology—though it's seamlessly integrated into everything we do. 
        It's the energy. Teams are collaborating across disciplines in ways that seemed 
        impossible just a few years ago.
        
        In the innovation lab, a diverse group is prototyping solutions to challenges 
        that don't even exist yet. They're not just thinking about what customers want 
        today, but what they'll need tomorrow. Every prototype is tested with real users, 
        refined based on feedback, and improved through collective intelligence.
        
        Our customer success stories aren't just metrics on a dashboard—they're personal 
        relationships built on trust and mutual growth. Customers don't just buy our 
        products; they co-create them with us. They know that when they succeed, we succeed.
        
        Every team member, from new graduates to seasoned veterans, feels empowered to 
        contribute their unique perspective. Learning isn't something that happens in 
        training sessions—it's woven into the fabric of how we work. Mistakes are 
        learning opportunities, and breakthrough ideas can come from anyone, anywhere.
        
        Our impact extends beyond our walls. We're known not just for what we build, 
        but for how we build it. Other organizations study our culture. Universities 
        send students to learn from our approach. We've become a beacon for what's 
        possible when innovation, collaboration, and human potential align.
        
        This isn't a distant dream—it's the future we're building together, one story, 
        one breakthrough, one transformed life at a time.
        
        The question isn't whether this future is possible. The question is: what role 
        will you play in creating it?
        """,
        cultural_themes=["vision", "innovation", "collaboration", "empowerment", "growth"],
        key_messages=[
            "The future is something we create together",
            "Innovation happens when diverse perspectives unite",
            "Customer relationships are partnerships",
            "Every person has unique value to contribute",
            "Our impact extends beyond our organization"
        ],
        target_outcomes=[
            "Inspire long-term commitment to transformation",
            "Create emotional connection to organizational vision",
            "Encourage individual ownership of collective success",
            "Position the organization as an industry leader"
        ]
    )
    
    print(f"✅ Vision narrative created: {vision_story.title}")
    print(f"   📊 Narrative strength: {vision_story.metadata.get('narrative_strength', 0):.2f}")
    print(f"   💫 Emotional resonance: {vision_story.metadata.get('emotional_resonance', 0):.2f}")
    
    # 3. Personalize Stories for Different Audiences
    print("\n3. Personalizing Stories for Audiences...")
    
    # Personalize hero story for engineering team
    engineering_profile = {
        "department": "engineering",
        "role_level": "mixed",
        "language_complexity": "advanced",
        "interests": ["technical_innovation", "problem_solving", "continuous_learning"],
        "communication_style": "direct",
        "cultural_background": "collaborative",
        "value_priorities": ["innovation", "excellence", "growth"],
        "preferred_examples": "technical_challenges"
    }
    
    eng_personalization = engine.personalize_story_for_audience(
        story_id=hero_story.id,
        audience_id="engineering_team",
        audience_profile=engineering_profile,
        delivery_format=DeliveryFormat.PRESENTATION
    )
    
    print(f"✅ Engineering personalization created")
    print(f"   🎯 Personalization score: {eng_personalization.personalization_score:.2f}")
    print(f"   🗣️ Language style: {eng_personalization.language_style}")
    print(f"   📱 Delivery format: {eng_personalization.delivery_format.value}")
    
    # Personalize success story for sales team
    sales_profile = {
        "department": "sales",
        "role_level": "mixed",
        "language_complexity": "moderate",
        "interests": ["customer_success", "results", "team_achievement"],
        "communication_style": "energetic",
        "cultural_background": "competitive",
        "value_priorities": ["customer_focus", "excellence", "collaboration"],
        "preferred_examples": "customer_wins"
    }
    
    sales_personalization = engine.personalize_story_for_audience(
        story_id=success_story.id,
        audience_id="sales_team",
        audience_profile=sales_profile,
        delivery_format=DeliveryFormat.VIDEO_STORY
    )
    
    print(f"✅ Sales personalization created")
    print(f"   🎯 Personalization score: {sales_personalization.personalization_score:.2f}")
    print(f"   🗣️ Language style: {sales_personalization.language_style}")
    print(f"   📱 Delivery format: {sales_personalization.delivery_format.value}")
    
    # 4. Measure Story Impact
    print("\n4. Measuring Story Impact...")
    
    # Simulate engagement data for hero story with engineering team
    hero_engagement_data = {
        "format": "presentation",
        "views": 85,
        "completion_rate": 0.92,
        "shares": 18,
        "comments": 24,
        "emotional_responses": {
            "inspiration": 32,
            "hope": 28,
            "determination": 25,
            "empathy": 20
        },
        "time_spent": 420,  # 7 minutes average
        "interaction_points": [
            {"timestamp": 45, "type": "question", "content": "How long did Maria's transformation take?"},
            {"timestamp": 180, "type": "highlight", "content": "combining different skills creates innovation"},
            {"timestamp": 320, "type": "discussion", "content": "Similar experience with design thinking workshop"}
        ]
    }
    
    hero_feedback_data = {
        "overall_rating": 0.88,
        "emotional_ratings": {
            "inspiration": 0.91,
            "hope": 0.85,
            "determination": 0.89,
            "empathy": 0.82
        },
        "comprehension_score": 0.94,
        "relevance_score": 0.87,
        "inspiration_score": 0.93,
        "action_intent": 0.79,
        "cultural_alignment_rating": 0.86,
        "theme_resonance": 0.90,
        "message_recall_score": 0.83,
        "memorability_score": 0.88,
        "motivation_to_change": 0.85,
        "belief_in_transformation": 0.91,
        "commitment_to_action": 0.78,
        "feedback_comments": [
            "Maria's story really resonated with my own experience",
            "Love how it shows that skepticism can be a strength",
            "The technical + creative combination is exactly what we need",
            "Would like to see more specific examples of the tools she used"
        ]
    }
    
    hero_impact = engine.measure_story_impact(
        story_id=hero_story.id,
        audience_id="engineering_team",
        engagement_data=hero_engagement_data,
        feedback_data=hero_feedback_data
    )
    
    print(f"✅ Hero story impact measured")
    print(f"   📊 Overall impact score: {hero_impact.impact_score:.2f}")
    print(f"   💫 Emotional impact (top 3):")
    top_emotions = sorted(hero_impact.emotional_impact.items(), key=lambda x: x[1], reverse=True)[:3]
    for emotion, score in top_emotions:
        print(f"     • {emotion.title()}: {score:.2f}")
    print(f"   🎯 Cultural alignment: {hero_impact.cultural_alignment:.2f}")
    print(f"   🧠 Message retention: {hero_impact.message_retention:.2f}")
    print(f"   🚀 Transformation influence: {hero_impact.transformation_influence:.2f}")
    
    # Simulate engagement data for success story with sales team
    success_engagement_data = {
        "format": "video_story",
        "views": 120,
        "completion_rate": 0.87,
        "shares": 35,
        "comments": 42,
        "emotional_responses": {
            "excitement": 45,
            "pride": 38,
            "inspiration": 33,
            "confidence": 29
        },
        "time_spent": 380,
        "interaction_points": [
            {"timestamp": 60, "type": "reaction", "content": "🔥"},
            {"timestamp": 150, "type": "share", "content": "This is exactly what we do!"},
            {"timestamp": 280, "type": "comment", "content": "Phoenix Method sounds like our Q3 approach"}
        ]
    }
    
    success_feedback_data = {
        "overall_rating": 0.84,
        "emotional_ratings": {
            "excitement": 0.89,
            "pride": 0.86,
            "inspiration": 0.81,
            "confidence": 0.83
        },
        "comprehension_score": 0.88,
        "relevance_score": 0.92,
        "inspiration_score": 0.85,
        "action_intent": 0.87,
        "cultural_alignment_rating": 0.89,
        "theme_resonance": 0.91,
        "message_recall_score": 0.80,
        "memorability_score": 0.85,
        "motivation_to_change": 0.82,
        "belief_in_transformation": 0.88,
        "commitment_to_action": 0.86
    }
    
    success_impact = engine.measure_story_impact(
        story_id=success_story.id,
        audience_id="sales_team",
        engagement_data=success_engagement_data,
        feedback_data=success_feedback_data
    )
    
    print(f"✅ Success story impact measured")
    print(f"   📊 Overall impact score: {success_impact.impact_score:.2f}")
    print(f"   💫 Emotional impact (top 3):")
    top_emotions = sorted(success_impact.emotional_impact.items(), key=lambda x: x[1], reverse=True)[:3]
    for emotion, score in top_emotions:
        print(f"     • {emotion.title()}: {score:.2f}")
    
    # 5. Create Storytelling Campaign
    print("\n5. Creating Storytelling Campaign...")
    
    campaign = engine.create_storytelling_campaign(
        name="Transformation Champions Campaign",
        description="A series of inspiring stories to accelerate cultural transformation",
        transformation_objectives=[
            "Increase employee engagement with transformation initiatives",
            "Build confidence in change capabilities",
            "Foster collaborative innovation mindset",
            "Strengthen customer-centric culture",
            "Encourage continuous learning and growth"
        ],
        target_audiences=["engineering_team", "sales_team", "marketing_team", "leadership_team"],
        stories=[hero_story.id, success_story.id, vision_story.id],
        duration_days=90
    )
    
    print(f"✅ Campaign created: {campaign.name}")
    print(f"   🎯 Transformation objectives: {len(campaign.transformation_objectives)}")
    print(f"   👥 Target audiences: {len(campaign.target_audiences)}")
    print(f"   📚 Stories included: {len(campaign.stories)}")
    print(f"   📅 Campaign duration: {campaign.start_date.strftime('%Y-%m-%d')} to {campaign.end_date.strftime('%Y-%m-%d')}")
    print(f"   🎭 Narrative arc theme: {campaign.narrative_arc['theme']}")
    
    # 6. Optimize Story Performance
    print("\n6. Optimizing Story Performance...")
    
    performance_data = {
        "avg_impact_score": 0.78,
        "engagement_breakdown": {
            "engineering_team": 0.88,
            "sales_team": 0.84,
            "marketing_team": 0.72,
            "leadership_team": 0.91
        },
        "emotional_impact": {
            "inspiration": 0.87,
            "hope": 0.82,
            "determination": 0.85,
            "excitement": 0.79,
            "pride": 0.83
        },
        "audience_feedback": {
            "positive_sentiment": 0.86,
            "constructive_feedback": 0.14,
            "engagement_requests": 0.23
        },
        "completion_rates": {
            "average": 0.89,
            "presentation": 0.92,
            "video_story": 0.87,
            "written_narrative": 0.88
        },
        "sharing_data": {
            "internal_shares": 0.28,
            "external_mentions": 0.12,
            "discussion_generation": 0.35
        },
        "improvement_areas": ["personalization", "interactivity", "follow_up_actions"]
    }
    
    hero_optimizations = engine.optimize_story_performance(
        story_id=hero_story.id,
        performance_data=performance_data
    )
    
    print(f"✅ Hero story optimized")
    print(f"   📈 Narrative improvements: {len(hero_optimizations['narrative_improvements'])} suggestions")
    print(f"   👥 Character enhancements: {len(hero_optimizations['character_enhancements'])} suggestions")
    print(f"   💫 Emotional optimization: {len(hero_optimizations['emotional_optimization'])} suggestions")
    
    # 7. Generate Story Analytics
    print("\n7. Generating Story Analytics...")
    
    time_period = {
        "start": datetime.now() - timedelta(days=30),
        "end": datetime.now()
    }
    
    hero_analytics = engine.generate_story_analytics(
        story_id=hero_story.id,
        time_period=time_period
    )
    
    print(f"✅ Analytics generated for hero story")
    print(f"   📊 Engagement metrics:")
    for metric, value in hero_analytics.engagement_metrics.items():
        if isinstance(value, (int, float)):
            print(f"     • {metric.replace('_', ' ').title()}: {value:,.0f}" if isinstance(value, int) else f"     • {metric.replace('_', ' ').title()}: {value:.2f}")
    
    print(f"   🎯 Impact metrics:")
    for metric, value in hero_analytics.impact_metrics.items():
        print(f"     • {metric.replace('_', ' ').title()}: {value:.2f}")
    
    print(f"   💡 Optimization recommendations:")
    for i, rec in enumerate(hero_analytics.optimization_recommendations[:3], 1):
        print(f"     {i}. {rec}")
    
    # 8. Display Comprehensive Analytics Dashboard
    print("\n8. Analytics Dashboard Summary")
    print("=" * 40)
    
    print(f"📚 Story Portfolio:")
    print(f"   • Total stories created: {len(engine.transformation_stories)}")
    print(f"   • Active campaigns: {len([c for c in engine.story_campaigns.values() if c.status in ['planned', 'active']])}")
    print(f"   • Story personalizations: {len(engine.story_personalizations)}")
    print(f"   • Impact measurements: {len(engine.impact_data)}")
    
    print(f"\n🎭 Story Types Performance:")
    story_types = {}
    for story in engine.transformation_stories.values():
        story_type = story.story_type.value
        if story_type not in story_types:
            story_types[story_type] = []
        story_types[story_type].append(story.metadata.get('narrative_strength', 0))
    
    for story_type, strengths in story_types.items():
        avg_strength = sum(strengths) / len(strengths)
        print(f"   • {story_type.replace('_', ' ').title()}: {avg_strength:.2f} avg strength")
    
    print(f"\n💫 Emotional Impact Leaders:")
    all_impacts = list(engine.impact_data.values())
    if all_impacts:
        avg_emotional_impact = {}
        for impact in all_impacts:
            for emotion, score in impact.emotional_impact.items():
                if emotion not in avg_emotional_impact:
                    avg_emotional_impact[emotion] = []
                avg_emotional_impact[emotion].append(score)
        
        for emotion, scores in avg_emotional_impact.items():
            avg_score = sum(scores) / len(scores)
            print(f"   • {emotion.title()}: {avg_score:.2f}")
    
    print(f"\n🎯 Audience Engagement:")
    audience_performance = {
        "engineering_team": 0.88,
        "sales_team": 0.84,
        "marketing_team": 0.72,
        "leadership_team": 0.91
    }
    
    for audience, score in sorted(audience_performance.items(), key=lambda x: x[1], reverse=True):
        print(f"   • {audience.replace('_', ' ').title()}: {score:.1%}")
    
    print(f"\n📈 Key Success Metrics:")
    print(f"   • Average story impact score: {performance_data['avg_impact_score']:.1%}")
    print(f"   • Average completion rate: {performance_data['completion_rates']['average']:.1%}")
    print(f"   • Internal sharing rate: {performance_data['sharing_data']['internal_shares']:.1%}")
    print(f"   • Discussion generation rate: {performance_data['sharing_data']['discussion_generation']:.1%}")
    
    print(f"\n🚀 Strategic Recommendations:")
    strategic_recommendations = [
        "Develop more technical transformation stories for engineering audiences",
        "Create interactive story formats to boost engagement",
        "Implement story-based mentoring programs",
        "Build customer success story library",
        "Establish story impact measurement standards",
        "Train leaders in storytelling techniques"
    ]
    
    for i, rec in enumerate(strategic_recommendations, 1):
        print(f"   {i}. {rec}")
    
    print(f"\n✨ Storytelling Framework Demo Complete!")
    print(f"   The system successfully demonstrated comprehensive storytelling")
    print(f"   capabilities with measurable impact tracking and optimization.")
    print(f"   Stories are now ready to drive cultural transformation!")


if __name__ == "__main__":
    asyncio.run(demo_storytelling_framework())