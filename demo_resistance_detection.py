#!/usr/bin/env python3
"""
Demo script for Cultural Change Resistance Detection System
Demonstrates early identification of cultural resistance patterns with source analysis and impact prediction.
"""
import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, Any

from scrollintel.engines.resistance_detection_engine import ResistanceDetectionEngine
from scrollintel.models.resistance_detection_models import (
    ResistanceType, ResistanceSeverity, ResistanceSource as ResistanceSourceEnum
)
from scrollintel.models.cultural_assessment_models import Organization
from scrollintel.models.transformation_roadmap_models import Transformation


def create_sample_organization() -> Organization:
    """Create a sample organization for demonstration"""
    return Organization(
        id="demo_org_001",
        name="TechCorp Innovation Labs",
        cultural_dimensions={
            "collaboration": 0.72,
            "innovation": 0.68,
            "adaptability": 0.65,
            "transparency": 0.70,
            "accountability": 0.75
        },
        values=[],
        behaviors=[],
        norms=[],
        subcultures=[],
        health_score=0.70,
        assessment_date=datetime.now()
    )


def create_sample_transformation() -> Transformation:
    """Create a sample transformation for demonstration"""
    return Transformation(
        id="demo_trans_001",
        organization_id="demo_org_001",
        current_culture=None,
        target_culture=None,
        vision=None,
        roadmap=None,
        interventions=[],
        progress=0.35,
        start_date=datetime.now() - timedelta(days=45),
        target_completion=datetime.now() + timedelta(days=120)
    )


def create_monitoring_data_scenario_1() -> Dict[str, Any]:
    """Create monitoring data showing early resistance signs"""
    return {
        "behavioral_data": {
            "attendance": {
                "meeting_attendance": 0.68,
                "training_attendance": 0.62,
                "voluntary_session_attendance": 0.45
            },
            "participation": {
                "active_participation": 0.55,
                "voluntary_feedback": 0.40,
                "initiative_volunteering": 0.35
            },
            "compliance": {
                "policy_compliance": 0.85,
                "process_adherence": 0.78,
                "deadline_adherence": 0.82
            }
        },
        "communication_data": {
            "sentiment": {
                "overall_sentiment": -0.12,
                "change_sentiment": -0.28,
                "leadership_sentiment": -0.15,
                "future_outlook_sentiment": -0.22
            },
            "feedback": {
                "negative_feedback_rate": 0.35,
                "concern_reports": 18,
                "complaint_frequency": 0.25,
                "suggestion_rate": 0.15
            },
            "concerns": {
                "resistance_indicators": 12,
                "rumor_reports": 5,
                "skepticism_expressions": 8,
                "fear_expressions": 6
            }
        },
        "engagement_data": {
            "scores": {
                "engagement_score": 0.58,
                "satisfaction_score": 0.55,
                "commitment_score": 0.62,
                "trust_score": 0.60
            },
            "voluntary_participation": {
                "initiative_participation": 0.42,
                "feedback_provision": 0.38,
                "peer_support": 0.65
            }
        },
        "performance_data": {
            "productivity": {
                "team_productivity": 0.82,
                "individual_productivity": 0.78,
                "collaboration_effectiveness": 0.70
            },
            "quality": {
                "work_quality": 0.88,
                "error_rate": 0.06,
                "rework_rate": 0.08
            }
        }
    }


def create_monitoring_data_scenario_2() -> Dict[str, Any]:
    """Create monitoring data showing active resistance"""
    return {
        "behavioral_data": {
            "attendance": {
                "meeting_attendance": 0.55,
                "training_attendance": 0.48,
                "voluntary_session_attendance": 0.25
            },
            "participation": {
                "active_participation": 0.35,
                "voluntary_feedback": 0.20,
                "initiative_volunteering": 0.15
            },
            "compliance": {
                "policy_compliance": 0.75,
                "process_adherence": 0.65,
                "deadline_adherence": 0.70
            }
        },
        "communication_data": {
            "sentiment": {
                "overall_sentiment": -0.35,
                "change_sentiment": -0.55,
                "leadership_sentiment": -0.40,
                "future_outlook_sentiment": -0.45
            },
            "feedback": {
                "negative_feedback_rate": 0.65,
                "concern_reports": 35,
                "complaint_frequency": 0.50,
                "suggestion_rate": 0.05
            },
            "concerns": {
                "resistance_indicators": 25,
                "rumor_reports": 12,
                "skepticism_expressions": 18,
                "fear_expressions": 15
            }
        },
        "engagement_data": {
            "scores": {
                "engagement_score": 0.35,
                "satisfaction_score": 0.30,
                "commitment_score": 0.40,
                "trust_score": 0.35
            },
            "voluntary_participation": {
                "initiative_participation": 0.20,
                "feedback_provision": 0.15,
                "peer_support": 0.45
            }
        },
        "performance_data": {
            "productivity": {
                "team_productivity": 0.70,
                "individual_productivity": 0.68,
                "collaboration_effectiveness": 0.50
            },
            "quality": {
                "work_quality": 0.80,
                "error_rate": 0.12,
                "rework_rate": 0.15
            }
        }
    }


def create_historical_data() -> Dict[str, Any]:
    """Create historical data for resistance prediction"""
    return {
        "past_detections": [
            {
                "type": "passive_resistance",
                "phase": "implementation",
                "probability": 0.75,
                "duration_days": 21,
                "resolution_method": "increased_communication"
            },
            {
                "type": "skepticism",
                "phase": "planning",
                "probability": 0.60,
                "duration_days": 14,
                "resolution_method": "stakeholder_engagement"
            },
            {
                "type": "fear_based",
                "phase": "rollout",
                "probability": 0.45,
                "duration_days": 10,
                "resolution_method": "training_support"
            }
        ],
        "stakeholder_patterns": {
            "middle_management": {
                "resistance_tendency": 0.65,
                "influence_level": 0.80,
                "typical_concerns": ["authority_loss", "workload_increase"]
            },
            "front_line_employees": {
                "resistance_tendency": 0.45,
                "influence_level": 0.60,
                "typical_concerns": ["job_security", "skill_requirements"]
            },
            "senior_leadership": {
                "resistance_tendency": 0.25,
                "influence_level": 0.95,
                "typical_concerns": ["roi_uncertainty", "timeline_pressure"]
            }
        },
        "intervention_history": {
            "training_programs": {
                "resistance_rate": 0.30,
                "effectiveness": 0.75,
                "typical_duration": 15
            },
            "policy_changes": {
                "resistance_rate": 0.55,
                "effectiveness": 0.60,
                "typical_duration": 25
            },
            "communication_campaigns": {
                "resistance_rate": 0.20,
                "effectiveness": 0.80,
                "typical_duration": 10
            }
        },
        "organizational_factors": {
            "change_history": "moderate",
            "leadership_credibility": 0.75,
            "communication_effectiveness": 0.70,
            "resource_availability": 0.65
        }
    }


async def demonstrate_resistance_detection():
    """Demonstrate resistance detection capabilities"""
    print("🔍 Cultural Change Resistance Detection System Demo")
    print("=" * 60)
    
    # Initialize components
    engine = ResistanceDetectionEngine()
    organization = create_sample_organization()
    transformation = create_sample_transformation()
    
    print(f"\n📊 Organization: {organization.name}")
    print(f"🔄 Transformation Progress: {transformation.progress:.1%}")
    print(f"📅 Days Since Start: {(datetime.now() - transformation.start_date).days}")
    
    # Scenario 1: Early resistance signs
    print("\n" + "="*60)
    print("📈 SCENARIO 1: Early Resistance Detection")
    print("="*60)
    
    monitoring_data_1 = create_monitoring_data_scenario_1()
    print("\n🔍 Analyzing monitoring data for early resistance signs...")
    
    detections_1 = engine.detect_resistance_patterns(
        organization=organization,
        transformation=transformation,
        monitoring_data=monitoring_data_1
    )
    
    print(f"✅ Detected {len(detections_1)} resistance patterns")
    
    # Display monitoring data summary
    print("\n📊 Monitoring Data Summary:")
    behavioral = monitoring_data_1['behavioral_data']
    communication = monitoring_data_1['communication_data']
    engagement = monitoring_data_1['engagement_data']
    
    print(f"  • Meeting Attendance: {behavioral['attendance']['meeting_attendance']:.1%}")
    print(f"  • Active Participation: {behavioral['participation']['active_participation']:.1%}")
    print(f"  • Change Sentiment: {communication['sentiment']['change_sentiment']:.2f}")
    print(f"  • Engagement Score: {engagement['scores']['engagement_score']:.1%}")
    print(f"  • Concern Reports: {communication['feedback']['concern_reports']}")
    
    # Scenario 2: Active resistance
    print("\n" + "="*60)
    print("🚨 SCENARIO 2: Active Resistance Detection")
    print("="*60)
    
    monitoring_data_2 = create_monitoring_data_scenario_2()
    print("\n🔍 Analyzing monitoring data for active resistance...")
    
    detections_2 = engine.detect_resistance_patterns(
        organization=organization,
        transformation=transformation,
        monitoring_data=monitoring_data_2
    )
    
    print(f"⚠️  Detected {len(detections_2)} resistance patterns")
    
    # Display monitoring data summary
    print("\n📊 Monitoring Data Summary:")
    behavioral = monitoring_data_2['behavioral_data']
    communication = monitoring_data_2['communication_data']
    engagement = monitoring_data_2['engagement_data']
    
    print(f"  • Meeting Attendance: {behavioral['attendance']['meeting_attendance']:.1%}")
    print(f"  • Active Participation: {behavioral['participation']['active_participation']:.1%}")
    print(f"  • Change Sentiment: {communication['sentiment']['change_sentiment']:.2f}")
    print(f"  • Engagement Score: {engagement['scores']['engagement_score']:.1%}")
    print(f"  • Concern Reports: {communication['feedback']['concern_reports']}")
    
    # Demonstrate source analysis (using mock detection)
    print("\n" + "="*60)
    print("🎯 RESISTANCE SOURCE ANALYSIS")
    print("="*60)
    
    from scrollintel.models.resistance_detection_models import ResistanceDetection
    
    mock_detection = ResistanceDetection(
        id="demo_detection_001",
        organization_id=organization.id,
        transformation_id=transformation.id,
        resistance_type=ResistanceType.PASSIVE_RESISTANCE,
        source=ResistanceSourceEnum.TEAM,
        severity=ResistanceSeverity.MODERATE,
        confidence_score=0.78,
        detected_at=datetime.now(),
        indicators_triggered=["low_participation", "delayed_compliance", "negative_sentiment"],
        affected_areas=["engineering_team", "product_team"],
        potential_impact={"timeline_delay": 0.15, "resource_increase": 0.10},
        detection_method="behavioral_analysis",
        raw_data=monitoring_data_1
    )
    
    print(f"\n🔍 Analyzing sources for detection: {mock_detection.id}")
    print(f"  • Type: {mock_detection.resistance_type.value}")
    print(f"  • Severity: {mock_detection.severity.value}")
    print(f"  • Confidence: {mock_detection.confidence_score:.1%}")
    
    sources = engine.analyze_resistance_sources(
        detection=mock_detection,
        organization=organization
    )
    
    print(f"✅ Identified {len(sources)} resistance sources")
    
    # Demonstrate impact assessment
    print("\n" + "="*60)
    print("📊 RESISTANCE IMPACT ASSESSMENT")
    print("="*60)
    
    print(f"\n🔍 Assessing impact for detection: {mock_detection.id}")
    
    impact_assessment = engine.assess_resistance_impact(
        detection=mock_detection,
        transformation=transformation
    )
    
    print(f"✅ Impact assessment completed with {impact_assessment.assessment_confidence:.1%} confidence")
    print(f"\n📈 Impact Summary:")
    print(f"  • Timeline Delay: {impact_assessment.transformation_impact.get('timeline_delay', 0):.1%}")
    print(f"  • Resource Increase: {impact_assessment.transformation_impact.get('resource_increase', 0):.1%}")
    print(f"  • Success Probability Reduction: {impact_assessment.success_probability_reduction:.1%}")
    print(f"  • Critical Path Disruption: {'Yes' if impact_assessment.critical_path_disruption else 'No'}")
    print(f"  • Delay Days: {impact_assessment.timeline_impact.get('delay_days', 0)}")
    print(f"  • Additional Effort: {impact_assessment.resource_impact.get('additional_effort', 0):.1%}")
    
    print(f"\n🔗 Cascading Effects:")
    for effect in impact_assessment.cascading_effects:
        print(f"  • {effect.replace('_', ' ').title()}")
    
    # Demonstrate resistance prediction
    print("\n" + "="*60)
    print("🔮 FUTURE RESISTANCE PREDICTION")
    print("="*60)
    
    historical_data = create_historical_data()
    print(f"\n🔍 Predicting future resistance using historical data...")
    print(f"  • Past Detections: {len(historical_data['past_detections'])}")
    print(f"  • Stakeholder Groups: {len(historical_data['stakeholder_patterns'])}")
    print(f"  • Intervention History: {len(historical_data['intervention_history'])}")
    
    predictions = engine.predict_future_resistance(
        organization=organization,
        transformation=transformation,
        historical_data=historical_data
    )
    
    print(f"✅ Generated {len(predictions)} resistance predictions")
    
    # Display historical patterns
    print(f"\n📊 Historical Resistance Patterns:")
    for detection in historical_data['past_detections']:
        print(f"  • {detection['type'].replace('_', ' ').title()}: {detection['probability']:.1%} probability")
    
    print(f"\n👥 Stakeholder Resistance Tendencies:")
    for group, data in historical_data['stakeholder_patterns'].items():
        print(f"  • {group.replace('_', ' ').title()}: {data['resistance_tendency']:.1%} tendency")
    
    # Demonstrate monitoring configuration
    print("\n" + "="*60)
    print("⚙️ MONITORING CONFIGURATION")
    print("="*60)
    
    print(f"\n🔧 Resistance monitoring configuration:")
    print(f"  • Detection Sensitivity: High (0.7)")
    print(f"  • Monitoring Frequency: Daily")
    print(f"  • Alert Thresholds: Moderate (0.6), High (0.8)")
    print(f"  • Monitoring Channels: Behavioral, Communication, Engagement")
    print(f"  • Stakeholder Groups: All levels")
    print(f"  • Reporting Schedule: Weekly")
    
    # Summary
    print("\n" + "="*60)
    print("📋 RESISTANCE DETECTION SUMMARY")
    print("="*60)
    
    print(f"\n✅ System Capabilities Demonstrated:")
    print(f"  • Early resistance pattern identification")
    print(f"  • Multi-channel monitoring data analysis")
    print(f"  • Resistance source analysis and categorization")
    print(f"  • Impact assessment with timeline and resource predictions")
    print(f"  • Future resistance prediction using historical patterns")
    print(f"  • Configurable monitoring and alerting")
    
    print(f"\n🎯 Key Benefits:")
    print(f"  • Proactive resistance management")
    print(f"  • Data-driven intervention planning")
    print(f"  • Reduced transformation risk")
    print(f"  • Improved success probability")
    print(f"  • Stakeholder-specific strategies")
    
    print(f"\n🚀 Next Steps:")
    print(f"  • Implement continuous monitoring")
    print(f"  • Develop mitigation strategies")
    print(f"  • Train change management team")
    print(f"  • Establish escalation procedures")
    
    print(f"\n✨ Resistance Detection System Demo Complete!")


if __name__ == "__main__":
    asyncio.run(demonstrate_resistance_detection())