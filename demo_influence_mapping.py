"""
Demo script for Influence Mapping Engine - Global Influence Network System
Demonstrates comprehensive influence network mapping, power structure analysis,
and competitive positioning capabilities.
"""
import asyncio
import json
from datetime import datetime
from scrollintel.engines.influence_mapping_engine import InfluenceMappingEngine
from scrollintel.models.relationship_models import Relationship, RelationshipType
from scrollintel.models.influence_strategy_models import InfluenceTarget, InfluenceType


async def create_sample_data():
    """Create comprehensive sample data for demonstration"""
    
    # Create influence targets representing key industry players
    influence_targets = [
        # Technology Leaders
        InfluenceTarget(
            id="tech_ceo_1",
            name="Sarah Chen",
            title="CEO",
            organization="TechVanguard Inc",
            industry="Technology",
            influence_score=0.95,
            influence_type=InfluenceType.INDUSTRY_LEADER,
            expertise_areas=["AI", "Cloud Computing", "Digital Transformation"],
            geographic_reach=["North America", "Europe", "Asia"]
        ),
        InfluenceTarget(
            id="tech_cto_1",
            name="Marcus Rodriguez",
            title="CTO",
            organization="InnovateTech Corp",
            industry="Technology",
            influence_score=0.88,
            influence_type=InfluenceType.TECHNICAL_EXPERT,
            expertise_areas=["Machine Learning", "Cybersecurity", "DevOps"],
            geographic_reach=["North America", "Europe"]
        ),
        
        # Financial Sector Leaders
        InfluenceTarget(
            id="finance_ceo_1",
            name="David Thompson",
            title="CEO",
            organization="GlobalFinance Group",
            industry="Financial Services",
            influence_score=0.92,
            influence_type=InfluenceType.INDUSTRY_LEADER,
            expertise_areas=["Investment Banking", "Risk Management", "Fintech"],
            geographic_reach=["Global"]
        ),
        InfluenceTarget(
            id="finance_cfo_1",
            name="Lisa Wang",
            title="CFO",
            organization="CapitalVentures Ltd",
            industry="Financial Services",
            influence_score=0.85,
            influence_type=InfluenceType.DECISION_MAKER,
            expertise_areas=["Corporate Finance", "M&A", "Strategic Planning"],
            geographic_reach=["North America", "Asia"]
        ),
        
        # Media and Communications
        InfluenceTarget(
            id="media_exec_1",
            name="Jennifer Adams",
            title="Editor-in-Chief",
            organization="TechToday Magazine",
            industry="Media",
            influence_score=0.78,
            influence_type=InfluenceType.THOUGHT_LEADER,
            expertise_areas=["Technology Journalism", "Industry Analysis", "Content Strategy"],
            geographic_reach=["North America", "Europe"]
        ),
        
        # Government and Policy
        InfluenceTarget(
            id="policy_maker_1",
            name="Robert Kim",
            title="Director of Technology Policy",
            organization="Department of Commerce",
            industry="Government",
            influence_score=0.82,
            influence_type=InfluenceType.POLICY_MAKER,
            expertise_areas=["Technology Policy", "Regulation", "International Trade"],
            geographic_reach=["North America"]
        ),
        
        # Academic and Research
        InfluenceTarget(
            id="academic_1",
            name="Dr. Elena Petrov",
            title="Professor of Computer Science",
            organization="Stanford University",
            industry="Academia",
            influence_score=0.75,
            influence_type=InfluenceType.THOUGHT_LEADER,
            expertise_areas=["Artificial Intelligence", "Ethics in AI", "Research"],
            geographic_reach=["Global"]
        ),
        
        # Venture Capital
        InfluenceTarget(
            id="vc_partner_1",
            name="Michael Chang",
            title="Managing Partner",
            organization="Innovation Ventures",
            industry="Venture Capital",
            influence_score=0.87,
            influence_type=InfluenceType.DECISION_MAKER,
            expertise_areas=["Early Stage Investing", "Technology Startups", "Market Analysis"],
            geographic_reach=["North America", "Asia"]
        )
    ]
    
    # Create relationships between these targets
    relationships = [
        # Professional relationships
        Relationship(
            id="rel_1",
            source_id="tech_ceo_1",
            target_id="tech_cto_1",
            relationship_type=RelationshipType.PROFESSIONAL,
            strength=0.9,
            interaction_frequency=0.8,
            trust_level=0.85,
            collaboration_history=[
                {"event": "Joint AI Initiative", "date": "2023-06-15", "outcome": "successful"},
                {"event": "Industry Conference Panel", "date": "2023-09-20", "outcome": "positive"}
            ]
        ),
        
        # Business partnerships
        Relationship(
            id="rel_2",
            source_id="tech_ceo_1",
            target_id="finance_ceo_1",
            relationship_type=RelationshipType.BUSINESS,
            strength=0.75,
            interaction_frequency=0.6,
            trust_level=0.8,
            collaboration_history=[
                {"event": "Strategic Partnership Agreement", "date": "2023-03-10", "outcome": "ongoing"}
            ]
        ),
        
        # Media relationships
        Relationship(
            id="rel_3",
            source_id="tech_ceo_1",
            target_id="media_exec_1",
            relationship_type=RelationshipType.PROFESSIONAL,
            strength=0.7,
            interaction_frequency=0.5,
            trust_level=0.75,
            collaboration_history=[
                {"event": "Exclusive Interview", "date": "2023-08-05", "outcome": "positive coverage"}
            ]
        ),
        
        # Investment relationships
        Relationship(
            id="rel_4",
            source_id="vc_partner_1",
            target_id="tech_cto_1",
            relationship_type=RelationshipType.BUSINESS,
            strength=0.85,
            interaction_frequency=0.7,
            trust_level=0.9,
            collaboration_history=[
                {"event": "Series B Investment", "date": "2023-01-15", "outcome": "successful"},
                {"event": "Board Advisory Role", "date": "2023-02-01", "outcome": "ongoing"}
            ]
        ),
        
        # Academic collaborations
        Relationship(
            id="rel_5",
            source_id="academic_1",
            target_id="tech_ceo_1",
            relationship_type=RelationshipType.PROFESSIONAL,
            strength=0.65,
            interaction_frequency=0.4,
            trust_level=0.8,
            collaboration_history=[
                {"event": "Research Collaboration", "date": "2023-04-20", "outcome": "published paper"}
            ]
        ),
        
        # Policy connections
        Relationship(
            id="rel_6",
            source_id="policy_maker_1",
            target_id="finance_ceo_1",
            relationship_type=RelationshipType.PROFESSIONAL,
            strength=0.7,
            interaction_frequency=0.3,
            trust_level=0.75,
            collaboration_history=[
                {"event": "Policy Advisory Committee", "date": "2023-05-10", "outcome": "ongoing"}
            ]
        ),
        
        # Cross-industry connections
        Relationship(
            id="rel_7",
            source_id="finance_cfo_1",
            target_id="vc_partner_1",
            relationship_type=RelationshipType.PROFESSIONAL,
            strength=0.8,
            interaction_frequency=0.6,
            trust_level=0.85,
            collaboration_history=[
                {"event": "Investment Committee", "date": "2023-07-12", "outcome": "successful deals"}
            ]
        )
    ]
    
    return influence_targets, relationships


async def demonstrate_influence_mapping():
    """Demonstrate the influence mapping engine capabilities"""
    
    print("🌐 Global Influence Network System - Influence Mapping Demo")
    print("=" * 60)
    
    # Initialize the engine
    engine = InfluenceMappingEngine()
    
    # Create sample data
    print("\n📊 Creating sample influence network data...")
    targets, relationships = await create_sample_data()
    
    print(f"   • Created {len(targets)} influence targets")
    print(f"   • Created {len(relationships)} relationships")
    
    # Build the influence network
    print("\n🔗 Building influence network...")
    network_result = await engine.build_influence_network(
        relationships=relationships,
        influence_targets=targets
    )
    
    print(f"   • Network ID: {network_result['network_id']}")
    print(f"   • Network Size: {network_result['network_size']} nodes")
    print(f"   • Connections: {network_result['connection_count']} edges")
    print(f"   • Network Density: {network_result['metrics']['basic_metrics']['density']:.3f}")
    
    # Display network metrics
    print("\n📈 Network Metrics:")
    metrics = network_result['metrics']
    basic_metrics = metrics['basic_metrics']
    
    print(f"   • Average Clustering: {basic_metrics['avg_clustering']:.3f}")
    print(f"   • Connected Components: {metrics['connectivity']['weakly_connected_components']}")
    print(f"   • Largest Component: {metrics['connectivity']['largest_component_size']} nodes")
    
    # Show top influencers by centrality
    print("\n🎯 Top Influencers by Network Position:")
    centrality_measures = metrics['centrality_measures']
    degree_centrality = centrality_measures['degree_centrality']
    
    # Sort by degree centrality
    sorted_influencers = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)
    
    for i, (node_id, centrality) in enumerate(sorted_influencers[:5]):
        # Find the target info
        target_info = next((t for t in targets if t.id == node_id), None)
        if target_info:
            print(f"   {i+1}. {target_info.name} ({target_info.organization})")
            print(f"      • Centrality Score: {centrality:.3f}")
            print(f"      • Influence Score: {target_info.influence_score:.3f}")
            print(f"      • Industry: {target_info.industry}")
    
    # Analyze power structures
    print("\n🏛️ Analyzing power structures...")
    power_structure = await engine.analyze_power_structures()
    
    print("   Power Hierarchy:")
    for level, members in power_structure.hierarchy_levels.items():
        if members:
            print(f"   • {level.title()}: {len(members)} members")
            for member_id in members[:3]:  # Show first 3
                target_info = next((t for t in targets if t.id == member_id), None)
                if target_info:
                    print(f"     - {target_info.name} ({target_info.organization})")
    
    print(f"\n   Key Decision Makers: {len(power_structure.decision_makers)}")
    for decision_maker_id in power_structure.decision_makers[:3]:
        target_info = next((t for t in targets if t.id == decision_maker_id), None)
        if target_info:
            print(f"   • {target_info.name} - {target_info.title} at {target_info.organization}")
    
    print(f"\n   Power Brokers: {len(power_structure.power_brokers)}")
    for broker_id in power_structure.power_brokers[:3]:
        target_info = next((t for t in targets if t.id == broker_id), None)
        if target_info:
            print(f"   • {target_info.name} - High betweenness centrality")
    
    print(f"\n   Influence Clusters: {len(power_structure.influence_clusters)}")
    for i, cluster in enumerate(power_structure.influence_clusters[:3]):
        print(f"   • Cluster {i+1}: {len(cluster)} members")
    
    # Assess competitive position
    print("\n🏆 Assessing competitive position...")
    competitive_position = await engine.assess_competitive_position(
        our_organization="TechVanguard Inc",
        competitors=["InnovateTech Corp", "GlobalFinance Group"]
    )
    
    print("   Our Position:")
    our_pos = competitive_position.our_position
    print(f"   • Organization: {our_pos['organization']}")
    print(f"   • Influence Score: {our_pos['influence_score']:.3f}")
    print(f"   • Network Reach: {our_pos['network_reach']}")
    print(f"   • Strategic Position: {our_pos['strategic_position']}")
    
    print(f"\n   Competitor Analysis: {len(competitive_position.competitor_positions)} competitors")
    print(f"   • Market Gaps Identified: {len(competitive_position.market_gaps)}")
    print(f"   • Strategic Opportunities: {len(competitive_position.strategic_opportunities)}")
    print(f"   • Threat Assessment: {len(competitive_position.threat_assessment)} threats")
    
    # Identify network gaps
    print("\n🔍 Identifying network gaps...")
    network_gaps = await engine.identify_network_gaps()
    
    print(f"   Total Gaps Identified: {len(network_gaps)}")
    for i, gap in enumerate(network_gaps[:3]):
        print(f"   {i+1}. {gap.gap_type.title()} Gap")
        print(f"      • Impact Score: {gap.impact_score:.3f}")
        print(f"      • Priority: {gap.priority_level}")
        print(f"      • Estimated Effort: {gap.estimated_effort}")
    
    # Monitor influence shifts
    print("\n📊 Monitoring influence shifts...")
    influence_shifts = await engine.monitor_influence_shifts(time_window_days=30)
    
    print(f"   Analysis Period: {influence_shifts['analysis_period']}")
    print(f"   Influence Changes: {len(influence_shifts['influence_changes'])} detected")
    print(f"   Emerging Influencers: {len(influence_shifts['emerging_influencers'])}")
    print(f"   Declining Influence: {len(influence_shifts['declining_influence'])}")
    print(f"   Relationship Changes: {len(influence_shifts['relationship_changes'])} tracked")
    print(f"   Alerts Generated: {len(influence_shifts['alerts'])}")
    
    # Display sample network visualization data
    print("\n🎨 Network Visualization Data:")
    print("   Node Information:")
    for node in network_result['nodes'][:3]:
        print(f"   • {node['name']} ({node['organization']})")
        print(f"     - Position: {node['network_position']}")
        print(f"     - Influence Level: {node['influence_level']}")
        print(f"     - Centrality: {node['centrality_score']:.3f}")
    
    print("\n   Edge Information:")
    for edge in network_result['edges'][:3]:
        source_name = next((t.name for t in targets if t.id == edge['source_id']), edge['source_id'])
        target_name = next((t.name for t in targets if t.id == edge['target_id']), edge['target_id'])
        print(f"   • {source_name} → {target_name}")
        print(f"     - Type: {edge['relationship_type']}")
        print(f"     - Strength: {edge['strength']:.3f}")
        print(f"     - Trust Level: {edge['trust_level']:.3f}")
    
    print("\n✅ Influence mapping demonstration completed!")
    print("\nKey Capabilities Demonstrated:")
    print("• ✅ Network construction from relationships and targets")
    print("• ✅ Comprehensive network metrics calculation")
    print("• ✅ Power structure analysis and hierarchy identification")
    print("• ✅ Competitive positioning assessment")
    print("• ✅ Network gap identification and prioritization")
    print("• ✅ Influence shift monitoring and alerting")
    print("• ✅ Real-time network visualization data")


async def demonstrate_advanced_features():
    """Demonstrate advanced influence mapping features"""
    
    print("\n🚀 Advanced Features Demonstration")
    print("=" * 40)
    
    engine = InfluenceMappingEngine()
    targets, relationships = await create_sample_data()
    
    # Build network
    await engine.build_influence_network(relationships=relationships, influence_targets=targets)
    
    # Test different influence levels
    print("\n📊 Influence Level Analysis:")
    for target in targets:
        level = engine._determine_influence_level(target.influence_score)
        position_score = engine.influence_graph.nodes.get(target.id, {}).get('centrality_score', 0)
        network_position = engine._determine_network_position(position_score, 0.5, 0.5)
        
        print(f"   • {target.name}:")
        print(f"     - Influence Level: {level.value}")
        print(f"     - Network Position: {network_position.value}")
        print(f"     - Industry: {target.industry}")
    
    # Demonstrate influence pathways
    print("\n🛤️ Influence Pathways:")
    print("   Analyzing potential influence paths between key players...")
    
    # Show structural holes (opportunities for bridge-building)
    power_structure = await engine.analyze_power_structures()
    if power_structure.structural_holes:
        print(f"\n🔗 Structural Holes (Bridge Opportunities): {len(power_structure.structural_holes)}")
        for i, (node1, node2) in enumerate(power_structure.structural_holes[:3]):
            name1 = next((t.name for t in targets if t.id == node1), node1)
            name2 = next((t.name for t in targets if t.id == node2), node2)
            print(f"   {i+1}. Bridge opportunity between {name1} and {name2}")
    
    print("\n🎯 Strategic Recommendations:")
    print("   Based on network analysis, consider:")
    print("   • Strengthening relationships with high-centrality nodes")
    print("   • Building bridges across structural holes")
    print("   • Engaging with emerging influencers early")
    print("   • Monitoring competitive positioning changes")


if __name__ == "__main__":
    async def main():
        await demonstrate_influence_mapping()
        await demonstrate_advanced_features()
    
    asyncio.run(main())