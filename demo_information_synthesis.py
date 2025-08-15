"""
Demo: Information Synthesis Engine for Crisis Leadership Excellence

This demo showcases the rapid processing of incomplete and conflicting information,
information prioritization and filtering, and confidence scoring with uncertainty management.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List

from scrollintel.engines.information_synthesis_engine import InformationSynthesisEngine
from scrollintel.models.information_synthesis_models import (
    InformationItem, SynthesisRequest, FilterCriteria,
    InformationSource, InformationPriority
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def create_crisis_scenario_data() -> List[InformationItem]:
    """Create realistic crisis scenario information items"""
    
    # Simulate a major system outage crisis with conflicting information
    crisis_items = [
        # Critical system alerts
        InformationItem(
            content="ALERT: Primary database cluster has failed. All write operations are blocked. Estimated impact: 100% of users affected.",
            source=InformationSource.INTERNAL_SYSTEMS,
            confidence_score=0.95,
            reliability_score=0.98,
            priority=InformationPriority.CRITICAL,
            tags=["database", "failure", "critical", "outage"],
            verification_status="verified",
            timestamp=datetime.now() - timedelta(minutes=2)
        ),
        
        # Customer service reports
        InformationItem(
            content="Customer support receiving 500+ calls per minute. Users reporting inability to access accounts and complete transactions.",
            source=InformationSource.STAKEHOLDER_INPUT,
            confidence_score=0.9,
            reliability_score=0.85,
            priority=InformationPriority.CRITICAL,
            tags=["customers", "support", "access", "transactions"],
            verification_status="verified",
            timestamp=datetime.now() - timedelta(minutes=1)
        ),
        
        # Media monitoring
        InformationItem(
            content="Social media reports indicate widespread service disruption. #ServiceDown trending with 10K+ mentions in last hour.",
            source=InformationSource.SOCIAL_MEDIA,
            confidence_score=0.8,
            reliability_score=0.7,
            priority=InformationPriority.HIGH,
            tags=["social", "media", "trending", "disruption"],
            verification_status="verified",
            timestamp=datetime.now() - timedelta(minutes=3)
        ),
        
        # Conflicting information from monitoring
        InformationItem(
            content="External monitoring services show 99.9% uptime. All health checks passing normally.",
            source=InformationSource.EXTERNAL_REPORTS,
            confidence_score=0.7,
            reliability_score=0.8,
            priority=InformationPriority.MEDIUM,
            tags=["monitoring", "uptime", "health"],
            verification_status="verified",
            timestamp=datetime.now() - timedelta(minutes=5)
        ),
        
        # Sensor data
        InformationItem(
            content="Data center temperature and power levels normal. No hardware failures detected in primary facility.",
            source=InformationSource.SENSOR_DATA,
            confidence_score=0.85,
            reliability_score=0.9,
            priority=InformationPriority.MEDIUM,
            tags=["datacenter", "hardware", "normal"],
            verification_status="verified",
            timestamp=datetime.now() - timedelta(minutes=4)
        ),
        
        # Expert analysis
        InformationItem(
            content="Database team analysis: Issue appears to be software-related, not hardware. Potential data corruption in primary node.",
            source=InformationSource.EXPERT_ANALYSIS,
            confidence_score=0.8,
            reliability_score=0.9,
            priority=InformationPriority.HIGH,
            tags=["database", "software", "corruption", "analysis"],
            verification_status="verified",
            timestamp=datetime.now() - timedelta(minutes=1)
        ),
        
        # Regulatory concerns
        InformationItem(
            content="Compliance team alert: Extended outage may trigger regulatory reporting requirements within 4 hours.",
            source=InformationSource.STAKEHOLDER_INPUT,
            confidence_score=0.9,
            reliability_score=0.95,
            priority=InformationPriority.HIGH,
            tags=["compliance", "regulatory", "reporting"],
            verification_status="verified",
            timestamp=datetime.now() - timedelta(minutes=3)
        ),
        
        # Incomplete information
        InformationItem(
            content="Partial report from backup systems... connection lost during transmission...",
            source=InformationSource.INTERNAL_SYSTEMS,
            confidence_score=0.3,
            reliability_score=0.6,
            priority=InformationPriority.MEDIUM,
            tags=["backup", "incomplete"],
            verification_status="unverified",
            timestamp=datetime.now() - timedelta(minutes=6)
        ),
        
        # Contradictory information
        InformationItem(
            content="Network operations center reports no significant issues. All systems operating within normal parameters.",
            source=InformationSource.INTERNAL_SYSTEMS,
            confidence_score=0.6,
            reliability_score=0.8,
            priority=InformationPriority.MEDIUM,
            tags=["network", "normal", "operations"],
            verification_status="disputed",
            timestamp=datetime.now() - timedelta(minutes=7)
        ),
        
        # Media reports
        InformationItem(
            content="Tech news outlets reporting major outage affecting millions of users. Competitor services seeing increased traffic.",
            source=InformationSource.MEDIA_MONITORING,
            confidence_score=0.75,
            reliability_score=0.65,
            priority=InformationPriority.HIGH,
            tags=["media", "outage", "competitors"],
            verification_status="unverified",
            timestamp=datetime.now() - timedelta(minutes=8)
        )
    ]
    
    return crisis_items


async def demonstrate_basic_synthesis():
    """Demonstrate basic information synthesis capabilities"""
    print("\n" + "="*80)
    print("DEMO: Basic Information Synthesis")
    print("="*80)
    
    # Initialize engine
    engine = InformationSynthesisEngine()
    
    # Create crisis scenario data
    crisis_items = await create_crisis_scenario_data()
    
    # Add items to engine
    print(f"\n📥 Adding {len(crisis_items)} information items to synthesis engine...")
    item_ids = []
    for item in crisis_items:
        item_id = await engine.add_information_item(item)
        item_ids.append(item_id)
        print(f"   ✓ Added: {item.source.value} - {item.content[:50]}...")
    
    # Create synthesis request
    request = SynthesisRequest(
        crisis_id="demo_crisis_001",
        requester="crisis_manager",
        information_items=item_ids,
        urgency_level=InformationPriority.CRITICAL,
        synthesis_focus=["system_status", "customer_impact", "resolution_strategy"]
    )
    
    print(f"\n🔄 Performing information synthesis for crisis {request.crisis_id}...")
    
    # Perform synthesis
    synthesis = await engine.synthesize_information(request)
    
    # Display results
    print(f"\n📊 SYNTHESIS RESULTS")
    print(f"   Crisis ID: {synthesis.crisis_id}")
    print(f"   Confidence Level: {synthesis.confidence_level:.2f}")
    print(f"   Priority Score: {synthesis.priority_score:.2f}")
    print(f"   Processing Time: {synthesis.synthesis_timestamp}")
    
    print(f"\n🔍 KEY FINDINGS ({len(synthesis.key_findings)}):")
    for i, finding in enumerate(synthesis.key_findings, 1):
        print(f"   {i}. {finding}")
    
    print(f"\n⚠️  INFORMATION GAPS ({len(synthesis.information_gaps)}):")
    for i, gap in enumerate(synthesis.information_gaps, 1):
        print(f"   {i}. {gap}")
    
    print(f"\n💡 RECOMMENDATIONS ({len(synthesis.recommendations)}):")
    for i, rec in enumerate(synthesis.recommendations, 1):
        print(f"   {i}. {rec}")
    
    print(f"\n🔗 CONFLICTS IDENTIFIED: {len(synthesis.conflicts_identified)}")
    print(f"📋 SOURCE ITEMS PROCESSED: {len(synthesis.source_items)}")
    
    return synthesis


async def demonstrate_filtering_capabilities():
    """Demonstrate information filtering capabilities"""
    print("\n" + "="*80)
    print("DEMO: Information Filtering and Prioritization")
    print("="*80)
    
    engine = InformationSynthesisEngine()
    crisis_items = await create_crisis_scenario_data()
    
    # Add items to engine
    for item in crisis_items:
        await engine.add_information_item(item)
    
    print(f"\n📊 Original dataset: {len(crisis_items)} items")
    
    # Test different filter criteria
    filter_scenarios = [
        {
            "name": "High Confidence Only",
            "criteria": FilterCriteria(min_confidence=0.8),
            "description": "Filter for high-confidence information (≥0.8)"
        },
        {
            "name": "Critical Priority Only",
            "criteria": FilterCriteria(priority_threshold=InformationPriority.CRITICAL),
            "description": "Filter for critical priority information only"
        },
        {
            "name": "Internal Sources Only",
            "criteria": FilterCriteria(required_sources=[InformationSource.INTERNAL_SYSTEMS, InformationSource.EXPERT_ANALYSIS]),
            "description": "Filter for internal and expert sources only"
        },
        {
            "name": "Recent Information",
            "criteria": FilterCriteria(time_window_hours=1),
            "description": "Filter for information from last hour"
        },
        {
            "name": "Verified Information",
            "criteria": FilterCriteria(min_confidence=0.7, min_reliability=0.8),
            "description": "Filter for verified, reliable information"
        }
    ]
    
    for scenario in filter_scenarios:
        print(f"\n🔍 {scenario['name']}")
        print(f"   Description: {scenario['description']}")
        
        # Apply filter
        filtered_items = await engine._filter_information(crisis_items, scenario['criteria'])
        
        print(f"   Results: {len(filtered_items)}/{len(crisis_items)} items passed filter")
        
        if filtered_items:
            print("   Filtered items:")
            for item in filtered_items[:3]:  # Show first 3
                print(f"     • {item.source.value}: {item.content[:60]}...")
                print(f"       Confidence: {item.confidence_score:.2f}, Priority: {item.priority.value}")


async def demonstrate_conflict_resolution():
    """Demonstrate conflict detection and resolution"""
    print("\n" + "="*80)
    print("DEMO: Conflict Detection and Resolution")
    print("="*80)
    
    engine = InformationSynthesisEngine()
    
    # Create specifically conflicting information
    conflicting_items = [
        InformationItem(
            content="System is fully operational with no issues detected",
            source=InformationSource.INTERNAL_SYSTEMS,
            confidence_score=0.9,
            reliability_score=0.95,
            priority=InformationPriority.HIGH,
            tags=["system", "operational"],
            timestamp=datetime.now()
        ),
        InformationItem(
            content="System has completely failed and is not operational",
            source=InformationSource.EXTERNAL_REPORTS,
            confidence_score=0.8,
            reliability_score=0.7,
            priority=InformationPriority.HIGH,
            tags=["system", "failed"],
            timestamp=datetime.now() - timedelta(minutes=2)
        ),
        InformationItem(
            content="Partial system failure detected in secondary components",
            source=InformationSource.SENSOR_DATA,
            confidence_score=0.85,
            reliability_score=0.9,
            priority=InformationPriority.MEDIUM,
            tags=["system", "partial", "failure"],
            timestamp=datetime.now() - timedelta(minutes=1)
        )
    ]
    
    # Add items to engine
    item_ids = []
    for item in conflicting_items:
        item_id = await engine.add_information_item(item)
        item_ids.append(item_id)
    
    print(f"\n📥 Added {len(conflicting_items)} potentially conflicting items:")
    for item in conflicting_items:
        print(f"   • {item.source.value}: {item.content}")
        print(f"     Confidence: {item.confidence_score:.2f}, Reliability: {item.reliability_score:.2f}")
    
    # Detect conflicts
    print(f"\n🔍 Detecting conflicts...")
    conflicts = await engine._detect_conflicts(conflicting_items)
    
    print(f"   Found {len(conflicts)} conflicts:")
    for i, conflict in enumerate(conflicts, 1):
        print(f"   {i}. Type: {conflict.conflict_type.value}")
        print(f"      Severity: {conflict.severity:.2f}")
        print(f"      Description: {conflict.description}")
        print(f"      Items involved: {len(conflict.conflicting_items)}")
    
    # Resolve conflicts
    print(f"\n🔧 Resolving conflicts...")
    resolved_items = await engine._resolve_conflicts(conflicting_items, conflicts)
    
    print(f"   Resolution results:")
    print(f"   Original items: {len(conflicting_items)}")
    print(f"   Resolved items: {len(resolved_items)}")
    
    print(f"\n   Remaining items after resolution:")
    for item in resolved_items:
        print(f"   • {item.source.value}: {item.content[:60]}...")
        print(f"     Confidence: {item.confidence_score:.2f}, Reliability: {item.reliability_score:.2f}")


async def demonstrate_uncertainty_assessment():
    """Demonstrate uncertainty assessment capabilities"""
    print("\n" + "="*80)
    print("DEMO: Uncertainty Assessment and Management")
    print("="*80)
    
    engine = InformationSynthesisEngine()
    
    # Create items with varying uncertainty levels
    uncertain_items = [
        InformationItem(
            content="Confirmed: Database primary node has failed",
            source=InformationSource.INTERNAL_SYSTEMS,
            confidence_score=0.95,
            reliability_score=0.98,
            priority=InformationPriority.CRITICAL,
            verification_status="verified"
        ),
        InformationItem(
            content="Unconfirmed reports suggest possible network issues",
            source=InformationSource.SOCIAL_MEDIA,
            confidence_score=0.4,
            reliability_score=0.3,
            priority=InformationPriority.LOW,
            verification_status="unverified"
        ),
        InformationItem(
            content="Conflicting information from multiple monitoring sources",
            source=InformationSource.EXTERNAL_REPORTS,
            confidence_score=0.5,
            reliability_score=0.6,
            priority=InformationPriority.MEDIUM,
            verification_status="disputed"
        ),
        InformationItem(
            content="Partial data suggests...",  # Incomplete
            source=InformationSource.SENSOR_DATA,
            confidence_score=0.3,
            reliability_score=0.7,
            priority=InformationPriority.MEDIUM,
            verification_status="unverified"
        )
    ]
    
    # Add items to engine
    for item in uncertain_items:
        await engine.add_information_item(item)
    
    # Create mock synthesis and conflicts for assessment
    from scrollintel.models.information_synthesis_models import SynthesizedInformation, InformationConflict, ConflictType
    
    synthesis = SynthesizedInformation(
        crisis_id="uncertainty_demo",
        key_findings=["Mixed information quality detected"],
        confidence_level=0.6
    )
    
    conflicts = [
        InformationConflict(
            conflict_type=ConflictType.CONTRADICTORY_FACTS,
            conflicting_items=["item1", "item2"],
            description="Contradictory system status reports",
            severity=0.8,
            resolved=False
        )
    ]
    
    print(f"\n📊 Assessing uncertainty for {len(uncertain_items)} information items...")
    
    # Assess uncertainty
    uncertainty = await engine._assess_uncertainty(uncertain_items, conflicts, synthesis)
    
    print(f"\n📈 UNCERTAINTY ASSESSMENT RESULTS:")
    print(f"   Overall Uncertainty: {uncertainty.overall_uncertainty:.2f} (0=certain, 1=very uncertain)")
    print(f"   Information Completeness: {uncertainty.information_completeness:.2f}")
    print(f"   Source Diversity: {uncertainty.source_diversity:.2f}")
    print(f"   Temporal Consistency: {uncertainty.temporal_consistency:.2f}")
    print(f"   Conflict Resolution Confidence: {uncertainty.conflict_resolution_confidence:.2f}")
    
    print(f"\n⚠️  KEY UNCERTAINTIES ({len(uncertainty.key_uncertainties)}):")
    for i, uncertainty_factor in enumerate(uncertainty.key_uncertainties, 1):
        print(f"   {i}. {uncertainty_factor}")
    
    print(f"\n🛠️  MITIGATION STRATEGIES ({len(uncertainty.mitigation_strategies)}):")
    for i, strategy in enumerate(uncertainty.mitigation_strategies, 1):
        print(f"   {i}. {strategy}")


async def demonstrate_rapid_processing():
    """Demonstrate rapid processing under time pressure"""
    print("\n" + "="*80)
    print("DEMO: Rapid Processing Under Time Pressure")
    print("="*80)
    
    engine = InformationSynthesisEngine()
    
    # Create a large dataset to simulate high-volume crisis information
    print(f"\n⚡ Generating high-volume crisis information dataset...")
    
    large_dataset = []
    sources = list(InformationSource)
    priorities = list(InformationPriority)
    
    # Generate 100 information items
    for i in range(100):
        item = InformationItem(
            content=f"Crisis information item {i}: Status update from monitoring system {i%10}",
            source=sources[i % len(sources)],
            confidence_score=0.5 + (i % 5) * 0.1,  # Vary confidence
            reliability_score=0.6 + (i % 4) * 0.1,  # Vary reliability
            priority=priorities[i % len(priorities)],
            tags=[f"system_{i%10}", f"status_{i%5}"],
            timestamp=datetime.now() - timedelta(minutes=i%60)
        )
        large_dataset.append(item)
    
    # Add items to engine
    print(f"📥 Adding {len(large_dataset)} items to engine...")
    item_ids = []
    for item in large_dataset:
        item_id = await engine.add_information_item(item)
        item_ids.append(item_id)
    
    # Create urgent synthesis request
    request = SynthesisRequest(
        crisis_id="rapid_processing_demo",
        requester="emergency_coordinator",
        information_items=item_ids,
        urgency_level=InformationPriority.CRITICAL,
        synthesis_focus=["immediate_actions", "critical_status"]
    )
    
    print(f"\n⏱️  Performing rapid synthesis under time pressure...")
    start_time = datetime.now()
    
    # Perform synthesis
    synthesis = await engine.synthesize_information(request)
    
    processing_time = (datetime.now() - start_time).total_seconds()
    
    print(f"\n📊 RAPID PROCESSING RESULTS:")
    print(f"   Processing Time: {processing_time:.2f} seconds")
    print(f"   Items Processed: {len(synthesis.source_items)}")
    print(f"   Synthesis Confidence: {synthesis.confidence_level:.2f}")
    print(f"   Key Findings Generated: {len(synthesis.key_findings)}")
    print(f"   Conflicts Detected: {len(synthesis.conflicts_identified)}")
    print(f"   Recommendations: {len(synthesis.recommendations)}")
    
    # Performance assessment
    items_per_second = len(synthesis.source_items) / processing_time if processing_time > 0 else 0
    print(f"   Processing Rate: {items_per_second:.1f} items/second")
    
    if processing_time < engine.max_processing_time:
        print(f"   ✅ Processing completed within time limit ({engine.max_processing_time}s)")
    else:
        print(f"   ⚠️  Processing exceeded time limit ({engine.max_processing_time}s)")
    
    # Show top findings for rapid decision-making
    print(f"\n🎯 TOP PRIORITY FINDINGS FOR IMMEDIATE ACTION:")
    critical_findings = [f for f in synthesis.key_findings if f.startswith("CRITICAL:")]
    for i, finding in enumerate(critical_findings[:3], 1):
        print(f"   {i}. {finding}")


async def demonstrate_synthesis_metrics():
    """Demonstrate synthesis performance metrics"""
    print("\n" + "="*80)
    print("DEMO: Synthesis Performance Metrics")
    print("="*80)
    
    engine = InformationSynthesisEngine()
    crisis_items = await create_crisis_scenario_data()
    
    # Add items and perform synthesis
    item_ids = []
    for item in crisis_items:
        item_id = await engine.add_information_item(item)
        item_ids.append(item_id)
    
    request = SynthesisRequest(
        crisis_id="metrics_demo",
        requester="metrics_analyst",
        information_items=item_ids,
        urgency_level=InformationPriority.HIGH
    )
    
    synthesis = await engine.synthesize_information(request)
    
    # Get metrics
    metrics = await engine.get_synthesis_metrics(synthesis.id)
    
    print(f"\n📊 SYNTHESIS PERFORMANCE METRICS:")
    print(f"   Processing Time: {metrics.processing_time_seconds:.2f} seconds")
    print(f"   Items Processed: {metrics.items_processed}")
    print(f"   Items Filtered Out: {metrics.items_filtered_out}")
    print(f"   Conflicts Detected: {metrics.conflicts_detected}")
    print(f"   Conflicts Resolved: {metrics.conflicts_resolved}")
    print(f"   Confidence Improvement: {metrics.confidence_improvement:.2f}")
    print(f"   Uncertainty Reduction: {metrics.uncertainty_reduction:.2f}")
    print(f"   Synthesis Quality Score: {metrics.synthesis_quality_score:.2f}")
    
    # Calculate additional metrics
    if metrics.processing_time_seconds > 0:
        throughput = metrics.items_processed / metrics.processing_time_seconds
        print(f"   Processing Throughput: {throughput:.1f} items/second")
    
    if metrics.conflicts_detected > 0:
        resolution_rate = metrics.conflicts_resolved / metrics.conflicts_detected
        print(f"   Conflict Resolution Rate: {resolution_rate:.1%}")


async def main():
    """Run all information synthesis demonstrations"""
    print("🚨 CRISIS LEADERSHIP EXCELLENCE - INFORMATION SYNTHESIS ENGINE DEMO")
    print("This demo showcases rapid processing of incomplete and conflicting information")
    print("with advanced prioritization, filtering, and uncertainty management capabilities.")
    
    try:
        # Run all demonstrations
        await demonstrate_basic_synthesis()
        await demonstrate_filtering_capabilities()
        await demonstrate_conflict_resolution()
        await demonstrate_uncertainty_assessment()
        await demonstrate_rapid_processing()
        await demonstrate_synthesis_metrics()
        
        print("\n" + "="*80)
        print("✅ DEMO COMPLETED SUCCESSFULLY")
        print("="*80)
        print("\nThe Information Synthesis Engine demonstrates:")
        print("• Rapid processing of large volumes of crisis information")
        print("• Intelligent filtering and prioritization of information")
        print("• Advanced conflict detection and resolution")
        print("• Comprehensive uncertainty assessment and management")
        print("• Real-time performance metrics and quality assessment")
        print("• Scalable processing under time pressure")
        print("\nThis enables ScrollIntel to synthesize complex, conflicting information")
        print("during crisis situations and provide clear, actionable insights for")
        print("rapid decision-making under extreme pressure.")
        
    except Exception as e:
        logger.error(f"Demo failed: {str(e)}")
        print(f"\n❌ Demo failed: {str(e)}")


if __name__ == "__main__":
    asyncio.run(main())