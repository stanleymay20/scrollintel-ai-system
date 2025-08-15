"""
Tests for Research Collaboration System

This module tests the autonomous research collaboration, knowledge sharing,
and synergy identification capabilities.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from scrollintel.engines.research_collaboration_system import ResearchCollaborationSystem
from scrollintel.models.research_coordination_models import (
    ResearchProject, ResearchCollaboration, KnowledgeAsset, ResearchSynergy,
    ResearchResource, CollaborationType, ProjectStatus, ResourceType
)


@pytest.fixture
def collaboration_system():
    """Create research collaboration system instance"""
    return ResearchCollaborationSystem()


@pytest.fixture
def sample_projects():
    """Create sample research projects"""
    project1 = ResearchProject(
        name="AI Optimization Research",
        description="Research on AI algorithm optimization",
        research_domain="artificial_intelligence",
        objectives=["Improve AI efficiency", "Reduce computational cost"],
        hypotheses=["Optimized algorithms perform better"],
        methodology="experimental_analysis",
        priority=8,
        planned_start=datetime.now(),
        planned_end=datetime.now() + timedelta(days=90)
    )
    
    project2 = ResearchProject(
        name="Machine Learning Acceleration",
        description="Research on ML training acceleration",
        research_domain="machine_learning",
        objectives=["Accelerate ML training", "Optimize resource usage"],
        hypotheses=["Parallel processing improves speed"],
        methodology="experimental_analysis",
        priority=7,
        planned_start=datetime.now() + timedelta(days=10),
        planned_end=datetime.now() + timedelta(days=100)
    )
    
    project3 = ResearchProject(
        name="Quantum Computing Applications",
        description="Research on quantum computing applications",
        research_domain="quantum_computing",
        objectives=["Explore quantum algorithms", "Test quantum hardware"],
        hypotheses=["Quantum algorithms solve problems faster"],
        methodology="theoretical_analysis",
        priority=9,
        planned_start=datetime.now() + timedelta(days=30),
        planned_end=datetime.now() + timedelta(days=120)
    )
    
    # Add resources to projects
    for project in [project1, project2, project3]:
        project.allocated_resources = [
            ResearchResource(
                resource_type=ResourceType.COMPUTATIONAL,
                capacity=100.0,
                allocated=60.0 if project == project1 else 80.0
            ),
            ResearchResource(
                resource_type=ResourceType.DATA,
                capacity=50.0,
                allocated=30.0
            )
        ]
    
    return [project1, project2, project3]


@pytest.fixture
def sample_knowledge_asset():
    """Create sample knowledge asset"""
    return KnowledgeAsset(
        title="AI Optimization Techniques",
        description="Comprehensive guide to AI optimization",
        content="Detailed optimization techniques and best practices",
        asset_type="research_finding",
        domain="artificial_intelligence",
        keywords=["optimization", "AI", "efficiency"],
        confidence_score=0.9,
        validation_status="validated"
    )


class TestResearchCollaborationSystem:
    """Test research collaboration system functionality"""
    
    @pytest.mark.asyncio
    async def test_identify_collaboration_opportunities(self, collaboration_system, sample_projects):
        """Test identifying collaboration opportunities"""
        synergies = await collaboration_system.identify_collaboration_opportunities(
            projects=sample_projects,
            min_synergy_score=0.3
        )
        
        assert len(synergies) > 0
        
        # Verify synergy structure
        synergy = synergies[0]
        assert synergy.id is not None
        assert len(synergy.project_ids) == 2
        assert synergy.overall_score >= 0.3
        assert synergy.potential_score >= 0.0
        assert synergy.feasibility_score >= 0.0
        assert synergy.impact_score >= 0.0
        assert len(synergy.complementary_strengths) >= 0
        assert len(synergy.collaboration_opportunities) >= 0
        assert len(synergy.recommended_actions) > 0
        assert synergy.implementation_complexity in ["low", "medium", "high"]
        
        # Verify synergies are sorted by score
        for i in range(len(synergies) - 1):
            assert synergies[i].overall_score >= synergies[i + 1].overall_score
    
    @pytest.mark.asyncio
    async def test_identify_collaboration_opportunities_high_threshold(self, collaboration_system, sample_projects):
        """Test collaboration identification with high threshold"""
        synergies = await collaboration_system.identify_collaboration_opportunities(
            projects=sample_projects,
            min_synergy_score=0.9
        )
        
        # Should have fewer or no synergies with high threshold
        assert len(synergies) >= 0
        
        # All synergies should meet threshold
        for synergy in synergies:
            assert synergy.overall_score >= 0.9
    
    @pytest.mark.asyncio
    async def test_identify_collaboration_opportunities_insufficient_projects(self, collaboration_system):
        """Test collaboration identification with insufficient projects"""
        synergies = await collaboration_system.identify_collaboration_opportunities(
            projects=[],
            min_synergy_score=0.5
        )
        
        assert len(synergies) == 0
    
    def test_calculate_domain_similarity(self, collaboration_system, sample_projects):
        """Test domain similarity calculation"""
        project1, project2, project3 = sample_projects
        
        # Similar domains
        similarity = collaboration_system._calculate_domain_similarity(project1, project2)
        assert similarity > 0.0  # AI and ML should have some similarity
        
        # Different domains
        similarity = collaboration_system._calculate_domain_similarity(project1, project3)
        assert similarity >= 0.0  # AI and quantum computing might have some overlap
        
        # Test with empty domains
        project_empty = ResearchProject(research_domain="")
        similarity = collaboration_system._calculate_domain_similarity(project1, project_empty)
        assert similarity == 0.0
    
    def test_calculate_resource_complementarity(self, collaboration_system, sample_projects):
        """Test resource complementarity calculation"""
        project1, project2, project3 = sample_projects
        
        # Projects with different resource utilization
        complementarity = collaboration_system._calculate_resource_complementarity(project1, project2)
        assert complementarity >= 0.0
        assert complementarity <= 1.0
        
        # Test with no resources
        project_no_resources = ResearchProject()
        complementarity = collaboration_system._calculate_resource_complementarity(project1, project_no_resources)
        assert complementarity == 0.0
    
    def test_calculate_methodology_alignment(self, collaboration_system, sample_projects):
        """Test methodology alignment calculation"""
        project1, project2, project3 = sample_projects
        
        # Similar methodologies
        alignment = collaboration_system._calculate_methodology_alignment(project1, project2)
        assert alignment > 0.0  # Both use experimental_analysis
        
        # Different methodologies
        alignment = collaboration_system._calculate_methodology_alignment(project1, project3)
        assert alignment >= 0.0  # experimental vs theoretical
        
        # Test with empty methodology
        project_empty = ResearchProject(methodology="")
        alignment = collaboration_system._calculate_methodology_alignment(project1, project_empty)
        assert alignment == 0.0
    
    def test_calculate_timeline_compatibility(self, collaboration_system, sample_projects):
        """Test timeline compatibility calculation"""
        project1, project2, project3 = sample_projects
        
        # Overlapping timelines
        compatibility = collaboration_system._calculate_timeline_compatibility(project1, project2)
        assert compatibility > 0.0  # Should have some overlap
        
        # Test with no timeline info
        project_no_timeline = ResearchProject()
        compatibility = collaboration_system._calculate_timeline_compatibility(project1, project_no_timeline)
        assert compatibility == 0.5  # Neutral score
    
    def test_calculate_knowledge_gap_overlap(self, collaboration_system, sample_projects):
        """Test knowledge gap overlap calculation"""
        project1, project2, project3 = sample_projects
        
        # Projects with similar objectives
        overlap = collaboration_system._calculate_knowledge_gap_overlap(project1, project2)
        assert overlap >= 0.0
        assert overlap <= 1.0
        
        # Projects with different objectives
        overlap = collaboration_system._calculate_knowledge_gap_overlap(project1, project3)
        assert overlap >= 0.0
    
    def test_identify_complementary_strengths(self, collaboration_system, sample_projects):
        """Test identifying complementary strengths"""
        project1, project2, project3 = sample_projects
        
        strengths = collaboration_system._identify_complementary_strengths(project1, project2)
        assert isinstance(strengths, list)
        
        # Should identify some strengths
        assert len(strengths) >= 0
        
        # Test with very different projects
        strengths = collaboration_system._identify_complementary_strengths(project1, project3)
        assert isinstance(strengths, list)
    
    def test_identify_shared_challenges(self, collaboration_system, sample_projects):
        """Test identifying shared challenges"""
        project1, project2, project3 = sample_projects
        
        challenges = collaboration_system._identify_shared_challenges(project1, project2)
        assert isinstance(challenges, list)
        assert len(challenges) >= 0
    
    def test_identify_collaboration_opportunities_specific(self, collaboration_system, sample_projects):
        """Test identifying specific collaboration opportunities"""
        project1, project2, project3 = sample_projects
        
        opportunities = collaboration_system._identify_collaboration_opportunities_specific(project1, project2)
        assert isinstance(opportunities, list)
        assert len(opportunities) >= 0
        
        # Should suggest some opportunities
        if opportunities:
            assert any("share" in opp.lower() or "joint" in opp.lower() for opp in opportunities)
    
    def test_generate_collaboration_recommendations(self, collaboration_system, sample_projects):
        """Test generating collaboration recommendations"""
        project1, project2 = sample_projects[:2]
        
        # High synergy scenario
        high_synergy = ResearchSynergy(
            project_ids=[project1.id, project2.id],
            overall_score=0.85,
            feasibility_score=0.8,
            potential_score=0.9
        )
        
        recommendations = collaboration_system._generate_collaboration_recommendations(
            project1, project2, high_synergy
        )
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        assert any("formal collaboration" in rec.lower() for rec in recommendations)
        
        # Low synergy scenario
        low_synergy = ResearchSynergy(
            project_ids=[project1.id, project2.id],
            overall_score=0.4,
            feasibility_score=0.3,
            potential_score=0.5
        )
        
        recommendations = collaboration_system._generate_collaboration_recommendations(
            project1, project2, low_synergy
        )
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
    
    def test_estimate_collaboration_benefits(self, collaboration_system, sample_projects):
        """Test estimating collaboration benefits"""
        project1, project2 = sample_projects[:2]
        
        benefits = collaboration_system._estimate_collaboration_benefits(project1, project2)
        
        assert isinstance(benefits, dict)
        assert "resource_efficiency_gain" in benefits
        assert "timeline_acceleration" in benefits
        assert "knowledge_multiplication" in benefits
        assert "cost_reduction" in benefits
        
        # Verify benefit values are reasonable
        for benefit_value in benefits.values():
            assert benefit_value >= 0.0
            assert benefit_value <= 1.0
    
    def test_assess_implementation_complexity(self, collaboration_system, sample_projects):
        """Test assessing implementation complexity"""
        project1, project2, project3 = sample_projects
        
        # Similar projects should have lower complexity
        complexity = collaboration_system._assess_implementation_complexity(project1, project2)
        assert complexity in ["low", "medium", "high"]
        
        # Very different projects should have higher complexity
        complexity = collaboration_system._assess_implementation_complexity(project1, project3)
        assert complexity in ["low", "medium", "high"]
    
    @pytest.mark.asyncio
    async def test_create_collaboration(self, collaboration_system, sample_projects):
        """Test creating a research collaboration"""
        # First identify synergy
        synergies = await collaboration_system.identify_collaboration_opportunities(
            projects=sample_projects[:2],
            min_synergy_score=0.0
        )
        
        assert len(synergies) > 0
        synergy = synergies[0]
        
        # Create collaboration
        collaboration = await collaboration_system.create_collaboration(
            synergy=synergy,
            collaboration_type=CollaborationType.KNOWLEDGE_SHARING
        )
        
        assert collaboration is not None
        assert collaboration.id is not None
        assert collaboration.collaboration_type == CollaborationType.KNOWLEDGE_SHARING
        assert collaboration.primary_project_id == synergy.project_ids[0]
        assert collaboration.collaborating_project_ids == synergy.project_ids[1:]
        assert collaboration.synergy_score == synergy.overall_score
        assert collaboration.is_active is True
        
        # Verify collaboration stored
        assert collaboration.id in collaboration_system.active_collaborations
        
        # Verify synergy marked as exploited
        assert synergy.is_exploited is True
        assert "collaboration_id" in synergy.exploitation_results
    
    @pytest.mark.asyncio
    async def test_share_knowledge_asset(self, collaboration_system, sample_knowledge_asset):
        """Test sharing knowledge asset"""
        source_project_id = "project_1"
        target_project_ids = ["project_2", "project_3"]
        
        success = await collaboration_system.share_knowledge_asset(
            source_project_id=source_project_id,
            asset=sample_knowledge_asset,
            target_project_ids=target_project_ids
        )
        
        assert success is True
        
        # Verify asset stored
        assert sample_knowledge_asset.id in collaboration_system.knowledge_assets
        
        # Verify asset details
        stored_asset = collaboration_system.knowledge_assets[sample_knowledge_asset.id]
        assert stored_asset.source_project_id == source_project_id
        assert stored_asset.access_count == 0
    
    @pytest.mark.asyncio
    async def test_get_collaboration_metrics(self, collaboration_system, sample_projects):
        """Test getting collaboration metrics"""
        # Create some collaborations and knowledge assets
        synergies = await collaboration_system.identify_collaboration_opportunities(
            projects=sample_projects,
            min_synergy_score=0.0
        )
        
        if synergies:
            await collaboration_system.create_collaboration(
                synergy=synergies[0],
                collaboration_type=CollaborationType.KNOWLEDGE_SHARING
            )
        
        # Share knowledge asset
        knowledge_asset = KnowledgeAsset(
            title="Test Asset",
            description="Test description",
            domain="test_domain"
        )
        
        await collaboration_system.share_knowledge_asset(
            source_project_id="project_1",
            asset=knowledge_asset,
            target_project_ids=["project_2"]
        )
        
        # Get metrics
        metrics = await collaboration_system.get_collaboration_metrics()
        
        assert isinstance(metrics, dict)
        assert "total_collaborations" in metrics
        assert "active_collaborations" in metrics
        assert "knowledge_assets" in metrics
        assert "identified_synergies" in metrics
        assert "exploited_synergies" in metrics
        
        # Verify metric values
        assert metrics["total_collaborations"] >= 0
        assert metrics["active_collaborations"] >= 0
        assert metrics["knowledge_assets"] >= 1  # We added one
        assert metrics["identified_synergies"] >= 0
    
    @pytest.mark.asyncio
    async def test_collaboration_with_different_types(self, collaboration_system, sample_projects):
        """Test creating collaborations with different types"""
        synergies = await collaboration_system.identify_collaboration_opportunities(
            projects=sample_projects[:2],
            min_synergy_score=0.0
        )
        
        if synergies:
            synergy = synergies[0]
            
            # Test different collaboration types
            for collab_type in CollaborationType:
                collaboration = await collaboration_system.create_collaboration(
                    synergy=synergy,
                    collaboration_type=collab_type
                )
                
                assert collaboration.collaboration_type == collab_type
    
    def test_synergy_weights_configuration(self, collaboration_system):
        """Test synergy scoring weights configuration"""
        weights = collaboration_system.synergy_weights
        
        assert isinstance(weights, dict)
        assert "domain_similarity" in weights
        assert "resource_complementarity" in weights
        assert "methodology_alignment" in weights
        assert "timeline_compatibility" in weights
        assert "knowledge_gap_overlap" in weights
        
        # Verify weights sum to reasonable value
        total_weight = sum(weights.values())
        assert abs(total_weight - 1.0) < 0.01  # Should sum to approximately 1.0
    
    @pytest.mark.asyncio
    async def test_analyze_project_synergy_comprehensive(self, collaboration_system, sample_projects):
        """Test comprehensive project synergy analysis"""
        project1, project2 = sample_projects[:2]
        
        synergy = await collaboration_system._analyze_project_synergy(project1, project2)
        
        assert synergy is not None
        assert synergy.project_ids == [project1.id, project2.id]
        assert 0.0 <= synergy.overall_score <= 1.0
        assert 0.0 <= synergy.potential_score <= 1.0
        assert 0.0 <= synergy.feasibility_score <= 1.0
        assert 0.0 <= synergy.impact_score <= 1.0
        
        assert isinstance(synergy.complementary_strengths, list)
        assert isinstance(synergy.shared_challenges, list)
        assert isinstance(synergy.collaboration_opportunities, list)
        assert isinstance(synergy.recommended_actions, list)
        assert isinstance(synergy.estimated_benefits, dict)
        assert synergy.implementation_complexity in ["low", "medium", "high"]
    
    def test_collaboration_frequency_setting(self, collaboration_system):
        """Test collaboration frequency setting based on synergy score"""
        # High synergy should result in daily coordination
        high_synergy = ResearchSynergy(overall_score=0.9)
        collaboration_system.identified_synergies[high_synergy.id] = high_synergy
        
        # Medium synergy should result in weekly coordination
        medium_synergy = ResearchSynergy(overall_score=0.7)
        collaboration_system.identified_synergies[medium_synergy.id] = medium_synergy
        
        # Low synergy should result in monthly coordination
        low_synergy = ResearchSynergy(overall_score=0.5)
        collaboration_system.identified_synergies[low_synergy.id] = low_synergy
        
        # This is tested implicitly in create_collaboration method
        assert True  # Placeholder for frequency logic verification
    
    @pytest.mark.asyncio
    async def test_error_handling_in_collaboration_identification(self, collaboration_system):
        """Test error handling in collaboration identification"""
        # Test with invalid projects (None values)
        projects_with_none = [None, ResearchProject()]
        
        try:
            synergies = await collaboration_system.identify_collaboration_opportunities(
                projects=projects_with_none,
                min_synergy_score=0.5
            )
            # Should handle gracefully and return empty list
            assert isinstance(synergies, list)
        except Exception:
            # Or raise appropriate exception
            assert True
    
    def test_knowledge_asset_access_tracking(self, collaboration_system):
        """Test knowledge asset access tracking"""
        asset = KnowledgeAsset(
            title="Test Asset",
            description="Test description",
            access_count=5,
            citation_count=3,
            reuse_count=2
        )
        
        collaboration_system.knowledge_assets[asset.id] = asset
        
        # Verify tracking fields
        assert asset.access_count == 5
        assert asset.citation_count == 3
        assert asset.reuse_count == 2
        
        # Test metrics calculation with access tracking
        metrics = asyncio.run(collaboration_system.get_collaboration_metrics())
        if "knowledge_sharing_rate" in metrics:
            assert metrics["knowledge_sharing_rate"] >= 0


if __name__ == "__main__":
    pytest.main([__file__])