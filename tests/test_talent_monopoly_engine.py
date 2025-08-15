"""
Tests for Global Talent Monopoly Engine

Comprehensive test suite for the talent identification, recruitment,
and retention system designed to monopolize global technical talent.
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, patch

from scrollintel.engines.talent_monopoly_engine import (
    TalentMonopolyEngine,
    TalentCategory,
    TalentTier,
    TalentProfile,
    RecruitmentCampaign,
    RecruitmentStatus
)


@pytest.fixture
def talent_engine():
    """Create talent monopoly engine for testing"""
    return TalentMonopolyEngine()


@pytest.fixture
def sample_talent():
    """Create sample talent profile for testing"""
    return TalentProfile(
        id="test_talent_001",
        name="Test AI Researcher",
        category=TalentCategory.AI_RESEARCHER,
        tier=TalentTier.EXCEPTIONAL,
        current_company="Google DeepMind",
        location="San Francisco, CA",
        skills=["Deep Learning", "Neural Networks", "Transformers"],
        achievements=["Breakthrough Research", "Industry Recognition"],
        publications=["ICML Paper", "NIPS Paper"],
        patents=["AI Algorithm Patent"],
        github_profile="https://github.com/test_talent",
        linkedin_profile="https://linkedin.com/in/test_talent",
        compensation_estimate=2000000.0,
        acquisition_priority=9,
        recruitment_status=RecruitmentStatus.IDENTIFIED,
        contact_history=[],
        retention_score=0.0,
        created_at=datetime.now(),
        updated_at=datetime.now()
    )


class TestTalentMonopolyEngine:
    """Test cases for TalentMonopolyEngine"""
    
    def test_initialization(self, talent_engine):
        """Test engine initialization"""
        assert isinstance(talent_engine.talent_database, dict)
        assert isinstance(talent_engine.compensation_tiers, dict)
        assert isinstance(talent_engine.recruitment_campaigns, dict)
        assert isinstance(talent_engine.talent_pipeline, list)
        assert isinstance(talent_engine.retention_programs, dict)
        
        # Check compensation tiers are initialized
        assert len(talent_engine.compensation_tiers) == 4
        assert TalentTier.LEGENDARY in talent_engine.compensation_tiers
        assert TalentTier.EXCEPTIONAL in talent_engine.compensation_tiers
        
        # Check retention programs are initialized
        assert len(talent_engine.retention_programs) > 0
        assert "research_freedom" in talent_engine.retention_programs
        assert "equity_acceleration" in talent_engine.retention_programs
    
    def test_compensation_tiers(self, talent_engine):
        """Test compensation tier structure"""
        legendary_comp = talent_engine.compensation_tiers[TalentTier.LEGENDARY]
        exceptional_comp = talent_engine.compensation_tiers[TalentTier.EXCEPTIONAL]
        
        # Legendary tier should have higher compensation
        assert legendary_comp.base_salary > exceptional_comp.base_salary
        assert legendary_comp.equity_percentage > exceptional_comp.equity_percentage
        assert legendary_comp.signing_bonus > exceptional_comp.signing_bonus
        assert legendary_comp.total_package_value > exceptional_comp.total_package_value
        
        # Check minimum values
        assert legendary_comp.base_salary >= 2000000.0  # $2M minimum
        assert legendary_comp.equity_percentage >= 5.0   # 5% minimum
        assert legendary_comp.signing_bonus >= 5000000.0 # $5M minimum
    
    @pytest.mark.asyncio
    async def test_identify_global_talent(self, talent_engine):
        """Test global talent identification"""
        # Test AI researcher identification
        talents = await talent_engine.identify_global_talent(
            category=TalentCategory.AI_RESEARCHER,
            target_count=50
        )
        
        assert len(talents) == 50
        assert all(t.category == TalentCategory.AI_RESEARCHER for t in talents)
        assert all(isinstance(t.tier, TalentTier) for t in talents)
        assert all(t.id in talent_engine.talent_database for t in talents)
        
        # Check tier distribution (should have some legendary talents)
        tier_counts = {}
        for talent in talents:
            tier_counts[talent.tier] = tier_counts.get(talent.tier, 0) + 1
        
        assert TalentTier.LEGENDARY in tier_counts
        assert tier_counts[TalentTier.LEGENDARY] > 0
    
    @pytest.mark.asyncio
    async def test_create_recruitment_campaign(self, talent_engine):
        """Test recruitment campaign creation"""
        campaign = await talent_engine.create_recruitment_campaign(
            category=TalentCategory.AI_RESEARCHER,
            target_tier=TalentTier.LEGENDARY,
            target_count=10,
            budget=100000000.0
        )
        
        assert isinstance(campaign, RecruitmentCampaign)
        assert campaign.target_category == TalentCategory.AI_RESEARCHER
        assert campaign.target_tier == TalentTier.LEGENDARY
        assert campaign.target_count == 10
        assert campaign.budget == 100000000.0
        assert campaign.id in talent_engine.recruitment_campaigns
        
        # Check strategies are appropriate for legendary tier
        assert "executive_recruitment" in campaign.strategies
        assert "research_collaboration_offer" in campaign.strategies
        assert len(campaign.strategies) > 4  # Should have enhanced strategies
    
    @pytest.mark.asyncio
    async def test_execute_acquisition(self, talent_engine, sample_talent):
        """Test talent acquisition process"""
        # Add sample talent to database
        talent_engine.talent_database[sample_talent.id] = sample_talent
        
        # Execute acquisition
        success = await talent_engine.execute_acquisition(sample_talent.id)
        
        assert success is True
        assert sample_talent.recruitment_status == RecruitmentStatus.ACQUIRED
        assert sample_talent.id in talent_engine.talent_pipeline
        assert len(sample_talent.contact_history) > 0
        
        # Check contact history has all acquisition steps
        steps = [entry["step"] for entry in sample_talent.contact_history]
        expected_steps = [
            "initial_contact", "interest_assessment", "detailed_discussion",
            "offer_presentation", "negotiation", "contract_signing", "onboarding"
        ]
        assert all(step in steps for step in expected_steps)
    
    @pytest.mark.asyncio
    async def test_execute_acquisition_nonexistent_talent(self, talent_engine):
        """Test acquisition of non-existent talent"""
        success = await talent_engine.execute_acquisition("nonexistent_talent")
        assert success is False
    
    @pytest.mark.asyncio
    async def test_implement_retention_program(self, talent_engine, sample_talent):
        """Test retention program implementation"""
        # Add sample talent to database
        talent_engine.talent_database[sample_talent.id] = sample_talent
        
        # Implement retention program
        metrics = await talent_engine.implement_retention_program(sample_talent.id)
        
        assert isinstance(metrics, dict)
        assert "retention_score" in metrics
        assert "retention_cost" in metrics
        assert "programs_applied" in metrics
        
        # Check retention score is improved
        assert metrics["retention_score"] > 0.7  # Should be high
        assert sample_talent.retention_score == metrics["retention_score"]
        assert sample_talent.recruitment_status == RecruitmentStatus.RETAINED
        
        # Check cost is reasonable
        assert metrics["retention_cost"] > 0
        assert metrics["programs_applied"] == len(talent_engine.retention_programs)
    
    @pytest.mark.asyncio
    async def test_implement_retention_program_nonexistent_talent(self, talent_engine):
        """Test retention program for non-existent talent"""
        metrics = await talent_engine.implement_retention_program("nonexistent_talent")
        assert metrics == {}
    
    @pytest.mark.asyncio
    async def test_monitor_talent_pipeline(self, talent_engine):
        """Test talent pipeline monitoring"""
        # Add some test talents
        await talent_engine.identify_global_talent(
            category=TalentCategory.AI_RESEARCHER,
            target_count=20
        )
        
        # Acquire some talents
        talent_ids = list(talent_engine.talent_database.keys())[:5]
        for talent_id in talent_ids:
            await talent_engine.execute_acquisition(talent_id)
        
        # Monitor pipeline
        metrics = await talent_engine.monitor_talent_pipeline()
        
        assert isinstance(metrics, dict)
        assert "total_talents" in metrics
        assert "tier_distribution" in metrics
        assert "category_distribution" in metrics
        assert "status_distribution" in metrics
        assert "acquisition_rate" in metrics
        assert "retention_rate" in metrics
        assert "total_compensation_cost" in metrics
        
        # Check values make sense
        assert metrics["total_talents"] >= 5
        assert metrics["acquisition_rate"] >= 0.0
        assert metrics["total_compensation_cost"] > 0
    
    @pytest.mark.asyncio
    async def test_analyze_competitive_landscape(self, talent_engine):
        """Test competitive landscape analysis"""
        analysis = await talent_engine.analyze_competitive_landscape()
        
        assert isinstance(analysis, dict)
        assert "competitors" in analysis
        assert "our_position" in analysis
        assert "competitive_advantages" in analysis
        assert "recommendations" in analysis
        
        # Check competitors data
        competitors = analysis["competitors"]
        assert "Google" in competitors
        assert "Microsoft" in competitors
        assert "OpenAI" in competitors
        
        # Check competitive advantages
        advantages = analysis["competitive_advantages"]
        assert len(advantages) > 0
        assert any("compensation" in adv.lower() for adv in advantages)
        assert any("research" in adv.lower() for adv in advantages)
    
    def test_get_talent_statistics_empty(self, talent_engine):
        """Test talent statistics with empty database"""
        stats = talent_engine.get_talent_statistics()
        
        assert stats["total_talents"] == 0
        assert "message" in stats
    
    def test_get_talent_statistics_with_data(self, talent_engine, sample_talent):
        """Test talent statistics with data"""
        # Add sample talent
        talent_engine.talent_database[sample_talent.id] = sample_talent
        
        stats = talent_engine.get_talent_statistics()
        
        assert stats["total_talents"] == 1
        assert "tier_distribution" in stats
        assert "category_distribution" in stats
        assert "status_distribution" in stats
        
        # Check distributions
        assert stats["tier_distribution"]["exceptional"] == 1
        assert stats["category_distribution"]["ai_researcher"] == 1
        assert stats["status_distribution"]["identified"] == 1
    
    def test_determine_talent_tier(self, talent_engine):
        """Test talent tier determination logic"""
        # Test legendary tier (top 0.1%)
        tier = talent_engine._determine_talent_tier(0, 1000)
        assert tier == TalentTier.LEGENDARY
        
        # Test exceptional tier (top 1%)
        tier = talent_engine._determine_talent_tier(5, 1000)
        assert tier == TalentTier.EXCEPTIONAL
        
        # Test elite tier (top 5%)
        tier = talent_engine._determine_talent_tier(25, 1000)
        assert tier == TalentTier.ELITE
        
        # Test premium tier
        tier = talent_engine._determine_talent_tier(100, 1000)
        assert tier == TalentTier.PREMIUM
    
    def test_calculate_acquisition_probability(self, talent_engine, sample_talent):
        """Test acquisition probability calculation"""
        # Test with competitive offer
        prob = talent_engine._calculate_acquisition_probability(
            sample_talent, 
            sample_talent.compensation_estimate * 3  # 3x current compensation
        )
        
        assert 0.0 <= prob <= 1.0
        assert prob > 0.5  # Should be reasonably high with good offer
        
        # Test with low offer
        prob_low = talent_engine._calculate_acquisition_probability(
            sample_talent,
            sample_talent.compensation_estimate * 0.5  # 0.5x current compensation
        )
        
        assert prob_low < prob  # Lower offer should have lower probability
    
    def test_get_recruitment_strategies(self, talent_engine):
        """Test recruitment strategy selection"""
        # Test legendary tier strategies
        legendary_strategies = talent_engine._get_recruitment_strategies(TalentTier.LEGENDARY)
        assert "executive_recruitment" in legendary_strategies
        assert "research_collaboration_offer" in legendary_strategies
        assert len(legendary_strategies) > 4
        
        # Test premium tier strategies
        premium_strategies = talent_engine._get_recruitment_strategies(TalentTier.PREMIUM)
        assert "direct_outreach" in premium_strategies
        assert len(premium_strategies) == 4  # Base strategies only
    
    def test_get_category_skills(self, talent_engine):
        """Test skill mapping for categories"""
        ai_skills = talent_engine._get_category_skills(TalentCategory.AI_RESEARCHER)
        assert "Deep Learning" in ai_skills
        assert "Neural Networks" in ai_skills
        
        ml_skills = talent_engine._get_category_skills(TalentCategory.ML_ENGINEER)
        assert "MLOps" in ml_skills
        assert "TensorFlow" in ml_skills
    
    def test_generate_achievements(self, talent_engine):
        """Test achievement generation by tier"""
        legendary_achievements = talent_engine._generate_achievements(TalentTier.LEGENDARY)
        assert "Breakthrough Research Publication" in legendary_achievements
        assert len(legendary_achievements) > 4
        
        premium_achievements = talent_engine._generate_achievements(TalentTier.PREMIUM)
        assert len(premium_achievements) == 2  # Base achievements only
    
    def test_estimate_current_compensation(self, talent_engine):
        """Test compensation estimation by tier"""
        legendary_comp = talent_engine._estimate_current_compensation(TalentTier.LEGENDARY)
        exceptional_comp = talent_engine._estimate_current_compensation(TalentTier.EXCEPTIONAL)
        
        assert legendary_comp > exceptional_comp
        assert legendary_comp >= 5000000.0  # $5M for legendary
        assert exceptional_comp >= 2000000.0  # $2M for exceptional
    
    def test_calculate_priority(self, talent_engine):
        """Test priority calculation"""
        # AI researcher should get priority boost
        ai_priority = talent_engine._calculate_priority(
            TalentTier.LEGENDARY, 
            TalentCategory.AI_RESEARCHER
        )
        
        other_priority = talent_engine._calculate_priority(
            TalentTier.LEGENDARY,
            TalentCategory.SOFTWARE_ARCHITECT
        )
        
        assert ai_priority > other_priority
        assert ai_priority == 10  # Max priority
        assert 1 <= other_priority <= 10


@pytest.mark.asyncio
async def test_full_talent_acquisition_workflow():
    """Test complete talent acquisition workflow"""
    engine = TalentMonopolyEngine()
    
    # 1. Identify talents
    talents = await engine.identify_global_talent(
        category=TalentCategory.AI_RESEARCHER,
        target_count=10
    )
    assert len(talents) == 10
    
    # 2. Create recruitment campaign
    campaign = await engine.create_recruitment_campaign(
        category=TalentCategory.AI_RESEARCHER,
        target_tier=TalentTier.EXCEPTIONAL,
        target_count=5,
        budget=50000000.0
    )
    assert campaign.target_count == 5
    
    # 3. Acquire talents
    talent_ids = [t.id for t in talents[:3]]
    for talent_id in talent_ids:
        success = await engine.execute_acquisition(talent_id)
        assert success is True
    
    # 4. Implement retention programs
    for talent_id in talent_ids:
        metrics = await engine.implement_retention_program(talent_id)
        assert metrics["retention_score"] > 0.7
    
    # 5. Monitor pipeline
    pipeline_metrics = await engine.monitor_talent_pipeline()
    assert pipeline_metrics["total_talents"] >= 3
    assert pipeline_metrics["acquisition_rate"] > 0
    
    # 6. Analyze competitive landscape
    competitive_analysis = await engine.analyze_competitive_landscape()
    assert len(competitive_analysis["competitors"]) > 0
    assert len(competitive_analysis["competitive_advantages"]) > 0


if __name__ == "__main__":
    pytest.main([__file__])