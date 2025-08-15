"""
Demo script for Research Coordination System

This script demonstrates the autonomous research project management,
milestone tracking, resource coordination, and research collaboration capabilities.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any

from scrollintel.engines.research_project_manager import ResearchProjectManager
from scrollintel.engines.research_collaboration_system import ResearchCollaborationSystem
from scrollintel.models.research_coordination_models import (
    ResearchProject, KnowledgeAsset, CollaborationType
)
from scrollintel.models.research_coordination_models import ResearchTopic, Hypothesis


def print_section(title: str):
    """Print a formatted section header"""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")


def print_subsection(title: str):
    """Print a formatted subsection header"""
    print(f"\n{'-'*40}")
    print(f" {title}")
    print(f"{'-'*40}")


async def demo_research_project_management():
    """Demonstrate research project management capabilities"""
    print_section("RESEARCH PROJECT MANAGEMENT DEMO")
    
    # Initialize project manager
    project_manager = ResearchProjectManager()
    
    # Create sample research topics and hypotheses
    research_topics = [
        ResearchTopic(
            title="AI Algorithm Optimization",
            description="Research on optimizing artificial intelligence algorithms for better performance",
            domain="artificial_intelligence",
            research_questions=[
                "How can we reduce AI algorithm computational complexity?",
                "What optimization techniques work best for neural networks?",
                "How to balance accuracy and efficiency in AI models?"
            ],
            methodology="experimental_analysis"
        ),
        ResearchTopic(
            title="Quantum Machine Learning",
            description="Exploring quantum computing applications in machine learning",
            domain="quantum_computing",
            research_questions=[
                "Can quantum algorithms accelerate ML training?",
                "What are the limitations of quantum ML?",
                "How to implement quantum neural networks?"
            ],
            methodology="theoretical_analysis"
        ),
        ResearchTopic(
            title="Distributed Computing Optimization",
            description="Research on optimizing distributed computing systems",
            domain="distributed_systems",
            research_questions=[
                "How to minimize communication overhead in distributed systems?",
                "What are optimal load balancing strategies?",
                "How to handle fault tolerance efficiently?"
            ],
            methodology="experimental_analysis"
        )
    ]
    
    hypotheses = [
        Hypothesis(
            statement="Optimized algorithms can achieve 30% better performance",
            confidence=0.8,
            testable=True
        ),
        Hypothesis(
            statement="Quantum algorithms provide exponential speedup for specific ML tasks",
            confidence=0.7,
            testable=True
        ),
        Hypothesis(
            statement="Adaptive load balancing reduces system latency by 25%",
            confidence=0.75,
            testable=True
        )
    ]
    
    print_subsection("Creating Research Projects")
    
    projects = []
    for i, topic in enumerate(research_topics):
        print(f"\nCreating project: {topic.title}")
        
        project = await project_manager.create_research_project(
            research_topic=topic,
            hypotheses=[hypotheses[i]],
            project_type="basic_research" if i % 2 == 0 else "applied_research",
            priority=8 - i
        )
        
        projects.append(project)
        
        print(f"✓ Project created: {project.name}")
        print(f"  - ID: {project.id}")
        print(f"  - Status: {project.status.value}")
        print(f"  - Priority: {project.priority}")
        print(f"  - Milestones: {len(project.milestones)}")
        print(f"  - Resources: {len(project.allocated_resources)}")
        print(f"  - Timeline: {project.planned_start.strftime('%Y-%m-%d')} to {project.planned_end.strftime('%Y-%m-%d')}")
    
    print_subsection("Project Status and Milestone Tracking")
    
    # Simulate project progress
    for project in projects:
        print(f"\nProject: {project.name}")
        
        # Update some milestones
        for i, milestone in enumerate(project.milestones[:3]):  # Update first 3 milestones
            progress = min(100.0, (i + 1) * 30 + (i * 10))  # Varying progress
            
            success = await project_manager.update_milestone_progress(
                project_id=project.id,
                milestone_id=milestone.id,
                progress=progress
            )
            
            if success:
                print(f"  ✓ {milestone.name}: {progress}% complete")
        
        # Get project status
        status = await project_manager.get_project_status(project.id)
        if status:
            metrics = status["metrics"]
            print(f"  - Overall Progress: {metrics['progress_percentage']:.1f}%")
            print(f"  - Active Milestones: {metrics['active_milestones']}")
            print(f"  - Resource Utilization: {len(metrics['resource_utilization'])} types")
            
            if status["risks"]:
                print(f"  - Risks Identified: {len(status['risks'])}")
                for risk in status["risks"][:2]:  # Show first 2 risks
                    print(f"    • {risk['type']}: {risk['description']}")
    
    print_subsection("Resource Optimization")
    
    # Demonstrate resource optimization
    for project in projects[:2]:  # Optimize first 2 projects
        print(f"\nOptimizing resources for: {project.name}")
        
        result = await project_manager.optimize_resource_allocation(project.id)
        
        if "error" not in result:
            print(f"  ✓ Optimizations applied: {result['optimizations_applied']}")
            if result['optimizations']:
                for opt in result['optimizations']:
                    print(f"    • {opt['type']} for {opt['resource_type']}")
        else:
            print(f"  ⚠ Optimization failed: {result['error']}")
    
    print_subsection("Coordination Metrics")
    
    # Get overall coordination metrics
    metrics = await project_manager.get_coordination_metrics()
    
    print(f"\nOverall Research Coordination Metrics:")
    print(f"  - Total Projects: {metrics.total_projects}")
    print(f"  - Active Projects: {metrics.active_projects}")
    print(f"  - Completed Projects: {metrics.completed_projects}")
    print(f"  - Total Resources: {metrics.total_resources}")
    print(f"  - Resource Utilization: {metrics.resource_utilization_rate:.1f}%")
    print(f"  - Total Milestones: {metrics.total_milestones}")
    print(f"  - Milestone Completion Rate: {metrics.milestone_completion_rate:.1f}%")
    print(f"  - Success Rate: {metrics.success_rate:.1f}%")
    
    return projects


async def demo_research_collaboration():
    """Demonstrate research collaboration capabilities"""
    print_section("RESEARCH COLLABORATION DEMO")
    
    # Initialize collaboration system
    collaboration_system = ResearchCollaborationSystem()
    
    # Create sample projects for collaboration
    projects = [
        ResearchProject(
            name="Neural Network Optimization",
            description="Optimizing neural network architectures",
            research_domain="artificial_intelligence",
            objectives=["Improve network efficiency", "Reduce training time"],
            hypotheses=["Pruned networks maintain accuracy"],
            methodology="experimental_analysis",
            priority=8,
            planned_start=datetime.now(),
            planned_end=datetime.now() + timedelta(days=90)
        ),
        ResearchProject(
            name="Deep Learning Acceleration",
            description="Accelerating deep learning computations",
            research_domain="machine_learning",
            objectives=["Speed up training", "Optimize GPU usage"],
            hypotheses=["Parallel processing improves speed"],
            methodology="experimental_analysis",
            priority=7,
            planned_start=datetime.now() + timedelta(days=5),
            planned_end=datetime.now() + timedelta(days=85)
        ),
        ResearchProject(
            name="Quantum Neural Networks",
            description="Implementing neural networks on quantum computers",
            research_domain="quantum_computing",
            objectives=["Develop quantum NN algorithms", "Test quantum hardware"],
            hypotheses=["Quantum NNs solve problems faster"],
            methodology="theoretical_analysis",
            priority=9,
            planned_start=datetime.now() + timedelta(days=15),
            planned_end=datetime.now() + timedelta(days=120)
        )
    ]
    
    # Add resources to projects
    from scrollintel.models.research_coordination_models import ResearchResource, ResourceType
    
    for i, project in enumerate(projects):
        project.allocated_resources = [
            ResearchResource(
                resource_type=ResourceType.COMPUTATIONAL,
                capacity=100.0,
                allocated=50.0 + (i * 15)  # Varying utilization
            ),
            ResearchResource(
                resource_type=ResourceType.DATA,
                capacity=50.0,
                allocated=25.0 + (i * 5)
            )
        ]
    
    print_subsection("Identifying Collaboration Opportunities")
    
    # Identify collaboration opportunities
    synergies = await collaboration_system.identify_collaboration_opportunities(
        projects=projects,
        min_synergy_score=0.3
    )
    
    print(f"\nFound {len(synergies)} collaboration opportunities:")
    
    for i, synergy in enumerate(synergies[:3]):  # Show top 3
        project_names = [next(p.name for p in projects if p.id == pid) for pid in synergy.project_ids]
        
        print(f"\n{i+1}. Collaboration between:")
        print(f"   Projects: {' & '.join(project_names)}")
        print(f"   Overall Score: {synergy.overall_score:.3f}")
        print(f"   Potential Score: {synergy.potential_score:.3f}")
        print(f"   Feasibility Score: {synergy.feasibility_score:.3f}")
        print(f"   Implementation Complexity: {synergy.implementation_complexity}")
        
        if synergy.complementary_strengths:
            print(f"   Complementary Strengths:")
            for strength in synergy.complementary_strengths[:2]:
                print(f"     • {strength}")
        
        if synergy.collaboration_opportunities:
            print(f"   Opportunities:")
            for opportunity in synergy.collaboration_opportunities[:2]:
                print(f"     • {opportunity}")
        
        if synergy.recommended_actions:
            print(f"   Recommendations:")
            for action in synergy.recommended_actions[:2]:
                print(f"     • {action}")
    
    print_subsection("Creating Research Collaborations")
    
    # Create collaborations from top synergies
    collaborations = []
    collaboration_types = [CollaborationType.KNOWLEDGE_SHARING, CollaborationType.RESOURCE_SHARING, CollaborationType.JOINT_RESEARCH]
    
    for i, synergy in enumerate(synergies[:3]):
        collab_type = collaboration_types[i % len(collaboration_types)]
        
        print(f"\nCreating {collab_type.value} collaboration...")
        
        collaboration = await collaboration_system.create_collaboration(
            synergy=synergy,
            collaboration_type=collab_type
        )
        
        collaborations.append(collaboration)
        
        project_names = [next(p.name for p in projects if p.id == pid) for pid in synergy.project_ids]
        
        print(f"✓ Collaboration created: {collaboration.id}")
        print(f"  - Type: {collaboration.collaboration_type.value}")
        print(f"  - Projects: {' & '.join(project_names)}")
        print(f"  - Synergy Score: {collaboration.synergy_score:.3f}")
        print(f"  - Coordination Frequency: {collaboration.coordination_frequency}")
    
    print_subsection("Knowledge Sharing")
    
    # Demonstrate knowledge sharing
    knowledge_assets = [
        KnowledgeAsset(
            title="Neural Network Pruning Techniques",
            description="Comprehensive guide to neural network pruning methods",
            content="Detailed analysis of various pruning techniques and their effectiveness",
            asset_type="research_finding",
            domain="artificial_intelligence",
            keywords=["pruning", "neural networks", "optimization"],
            confidence_score=0.9,
            validation_status="validated"
        ),
        KnowledgeAsset(
            title="GPU Optimization Strategies",
            description="Best practices for GPU utilization in deep learning",
            content="Performance optimization techniques for GPU-accelerated training",
            asset_type="methodology",
            domain="machine_learning",
            keywords=["GPU", "optimization", "performance"],
            confidence_score=0.85,
            validation_status="validated"
        ),
        KnowledgeAsset(
            title="Quantum Computing Fundamentals",
            description="Introduction to quantum computing principles",
            content="Basic concepts and principles of quantum computation",
            asset_type="research_finding",
            domain="quantum_computing",
            keywords=["quantum", "computing", "fundamentals"],
            confidence_score=0.8,
            validation_status="pending"
        )
    ]
    
    for i, asset in enumerate(knowledge_assets):
        source_project = projects[i]
        target_projects = [p.id for p in projects if p.id != source_project.id]
        
        print(f"\nSharing knowledge asset: {asset.title}")
        print(f"  From: {source_project.name}")
        print(f"  To: {len(target_projects)} other projects")
        
        success = await collaboration_system.share_knowledge_asset(
            source_project_id=source_project.id,
            asset=asset,
            target_project_ids=target_projects
        )
        
        if success:
            print(f"  ✓ Knowledge asset shared successfully")
        else:
            print(f"  ⚠ Failed to share knowledge asset")
    
    print_subsection("Collaboration Metrics")
    
    # Get collaboration metrics
    metrics = await collaboration_system.get_collaboration_metrics()
    
    print(f"\nCollaboration System Metrics:")
    print(f"  - Total Collaborations: {metrics['total_collaborations']}")
    print(f"  - Active Collaborations: {metrics['active_collaborations']}")
    print(f"  - Knowledge Assets: {metrics['knowledge_assets']}")
    print(f"  - Identified Synergies: {metrics['identified_synergies']}")
    print(f"  - Exploited Synergies: {metrics['exploited_synergies']}")
    
    if "average_synergy_score" in metrics:
        print(f"  - Average Synergy Score: {metrics['average_synergy_score']:.3f}")
    
    if "knowledge_sharing_rate" in metrics:
        print(f"  - Knowledge Sharing Rate: {metrics['knowledge_sharing_rate']:.2f}")
    
    if "collaboration_effectiveness" in metrics:
        print(f"  - Collaboration Effectiveness: {metrics['collaboration_effectiveness']:.3f}")
    
    return collaborations


async def demo_integrated_research_coordination():
    """Demonstrate integrated research coordination"""
    print_section("INTEGRATED RESEARCH COORDINATION DEMO")
    
    print_subsection("End-to-End Research Coordination Workflow")
    
    # Create integrated workflow
    project_manager = ResearchProjectManager()
    collaboration_system = ResearchCollaborationSystem()
    
    # Step 1: Create research projects
    print("\n1. Creating research projects...")
    
    research_topic = ResearchTopic(
        title="Advanced AI Systems Integration",
        description="Research on integrating multiple AI systems for enhanced performance",
        domain="artificial_intelligence",
        research_questions=[
            "How to integrate different AI models effectively?",
            "What are the challenges in multi-AI coordination?",
            "How to optimize resource sharing between AI systems?"
        ],
        methodology="experimental_analysis"
    )
    
    hypotheses = [
        Hypothesis(
            statement="Integrated AI systems outperform individual systems by 40%",
            confidence=0.85,
            testable=True
        )
    ]
    
    project = await project_manager.create_research_project(
        research_topic=research_topic,
        hypotheses=hypotheses,
        project_type="applied_research",
        priority=9
    )
    
    print(f"✓ Created project: {project.name}")
    
    # Step 2: Simulate project progress
    print("\n2. Simulating project progress...")
    
    for i, milestone in enumerate(project.milestones[:4]):
        progress = min(100.0, (i + 1) * 25)
        await project_manager.update_milestone_progress(
            project_id=project.id,
            milestone_id=milestone.id,
            progress=progress
        )
        print(f"   ✓ {milestone.name}: {progress}% complete")
    
    # Step 3: Optimize resources
    print("\n3. Optimizing project resources...")
    
    optimization_result = await project_manager.optimize_resource_allocation(project.id)
    if "error" not in optimization_result:
        print(f"   ✓ Applied {optimization_result['optimizations_applied']} optimizations")
    
    # Step 4: Generate knowledge assets
    print("\n4. Generating knowledge assets...")
    
    knowledge_asset = KnowledgeAsset(
        title="AI Systems Integration Framework",
        description="Comprehensive framework for integrating multiple AI systems",
        content="Detailed methodology and best practices for AI integration",
        asset_type="methodology",
        domain="artificial_intelligence",
        keywords=["AI", "integration", "framework", "systems"],
        confidence_score=0.92,
        validation_status="validated"
    )
    
    await collaboration_system.share_knowledge_asset(
        source_project_id=project.id,
        asset=knowledge_asset,
        target_project_ids=[]  # No targets for this demo
    )
    
    print(f"   ✓ Created knowledge asset: {knowledge_asset.title}")
    
    # Step 5: Final metrics
    print("\n5. Final coordination metrics...")
    
    project_metrics = await project_manager.get_coordination_metrics()
    collaboration_metrics = await collaboration_system.get_collaboration_metrics()
    
    print(f"\nFinal Results:")
    print(f"  - Projects Managed: {project_metrics.total_projects}")
    print(f"  - Overall Progress: {project.calculate_progress():.1f}%")
    print(f"  - Resource Efficiency: {project_metrics.resource_utilization_rate:.1f}%")
    print(f"  - Knowledge Assets Created: {collaboration_metrics['knowledge_assets']}")
    print(f"  - System Success Rate: {project_metrics.success_rate:.1f}%")


async def main():
    """Main demo function"""
    print("🚀 AUTONOMOUS INNOVATION LAB - RESEARCH COORDINATION DEMO")
    print("=" * 80)
    
    try:
        # Run project management demo
        projects = await demo_research_project_management()
        
        # Run collaboration demo
        collaborations = await demo_research_collaboration()
        
        # Run integrated demo
        await demo_integrated_research_coordination()
        
        print_section("DEMO COMPLETED SUCCESSFULLY")
        print("\n✅ All research coordination capabilities demonstrated successfully!")
        print("\nKey Features Demonstrated:")
        print("  • Autonomous research project creation and management")
        print("  • Intelligent milestone tracking and progress monitoring")
        print("  • Dynamic resource allocation and optimization")
        print("  • Automated collaboration opportunity identification")
        print("  • Intelligent knowledge sharing and asset management")
        print("  • Comprehensive performance metrics and analytics")
        print("  • End-to-end research coordination workflow")
        
        print(f"\nDemo Statistics:")
        print(f"  • Research Projects Created: {len(projects) if 'projects' in locals() else 0}")
        print(f"  • Collaborations Established: {len(collaborations) if 'collaborations' in locals() else 0}")
        print(f"  • Knowledge Assets Shared: 3+")
        print(f"  • Optimization Operations: Multiple")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())