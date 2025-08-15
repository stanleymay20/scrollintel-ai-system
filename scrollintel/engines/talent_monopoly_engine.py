"""
Global Talent Monopoly Engine

This engine implements a comprehensive talent acquisition and retention system
designed to monopolize the world's top technical talent through superior
compensation, opportunities, and strategic recruitment.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json

logger = logging.getLogger(__name__)


class TalentCategory(Enum):
    """Categories of talent for strategic acquisition"""
    AI_RESEARCHER = "ai_researcher"
    ML_ENGINEER = "ml_engineer"
    SOFTWARE_ARCHITECT = "software_architect"
    DATA_SCIENTIST = "data_scientist"
    SYSTEMS_ENGINEER = "systems_engineer"
    PRODUCT_MANAGER = "product_manager"
    RESEARCH_SCIENTIST = "research_scientist"
    SECURITY_EXPERT = "security_expert"
    INFRASTRUCTURE_EXPERT = "infrastructure_expert"
    BUSINESS_STRATEGIST = "business_strategist"


class TalentTier(Enum):
    """Talent tiers for compensation and priority"""
    LEGENDARY = "legendary"  # Top 0.1% globally
    EXCEPTIONAL = "exceptional"  # Top 1% globally
    ELITE = "elite"  # Top 5% globally
    PREMIUM = "premium"  # Top 10% globally


class RecruitmentStatus(Enum):
    """Status of recruitment efforts"""
    IDENTIFIED = "identified"
    CONTACTED = "contacted"
    ENGAGED = "engaged"
    NEGOTIATING = "negotiating"
    ACQUIRED = "acquired"
    RETAINED = "retained"


@dataclass
class TalentProfile:
    """Comprehensive talent profile for strategic acquisition"""
    id: str
    name: str
    category: TalentCategory
    tier: TalentTier
    current_company: str
    location: str
    skills: List[str]
    achievements: List[str]
    publications: List[str]
    patents: List[str]
    github_profile: Optional[str]
    linkedin_profile: Optional[str]
    compensation_estimate: float
    acquisition_priority: int  # 1-10 scale
    recruitment_status: RecruitmentStatus
    contact_history: List[Dict]
    retention_score: float
    created_at: datetime
    updated_at: datetime


@dataclass
class CompensationPackage:
    """Comprehensive compensation package for talent acquisition"""
    base_salary: float
    equity_percentage: float
    signing_bonus: float
    annual_bonus_target: float
    research_budget: float
    conference_budget: float
    relocation_package: float
    benefits_value: float
    total_package_value: float
    tier_multiplier: float


@dataclass
class RecruitmentCampaign:
    """Strategic recruitment campaign for specific talent categories"""
    id: str
    name: str
    target_category: TalentCategory
    target_tier: TalentTier
    target_count: int
    budget: float
    timeline_months: int
    strategies: List[str]
    success_metrics: Dict[str, float]
    status: str
    created_at: datetime


class TalentMonopolyEngine:
    """
    Global Talent Monopoly Engine
    
    Implements comprehensive talent acquisition and retention strategies
    to monopolize the world's top technical talent.
    """
    
    def __init__(self):
        self.talent_database: Dict[str, TalentProfile] = {}
        self.compensation_tiers: Dict[TalentTier, CompensationPackage] = {}
        self.recruitment_campaigns: Dict[str, RecruitmentCampaign] = {}
        self.talent_pipeline: List[str] = []
        self.retention_programs: Dict[str, Dict] = {}
        self.global_talent_map: Dict[str, List[str]] = {}
        self.competitor_intelligence: Dict[str, Dict] = {}
        
        self._initialize_compensation_tiers()
        self._initialize_retention_programs()
    
    def _initialize_compensation_tiers(self):
        """Initialize compensation tiers for talent acquisition"""
        self.compensation_tiers = {
            TalentTier.LEGENDARY: CompensationPackage(
                base_salary=2000000.0,  # $2M base
                equity_percentage=5.0,   # 5% equity
                signing_bonus=5000000.0, # $5M signing
                annual_bonus_target=2000000.0,
                research_budget=1000000.0,
                conference_budget=100000.0,
                relocation_package=500000.0,
                benefits_value=200000.0,
                total_package_value=10800000.0,
                tier_multiplier=10.0
            ),
            TalentTier.EXCEPTIONAL: CompensationPackage(
                base_salary=1000000.0,  # $1M base
                equity_percentage=2.0,   # 2% equity
                signing_bonus=2000000.0, # $2M signing
                annual_bonus_target=1000000.0,
                research_budget=500000.0,
                conference_budget=50000.0,
                relocation_package=250000.0,
                benefits_value=150000.0,
                total_package_value=4950000.0,
                tier_multiplier=5.0
            ),
            TalentTier.ELITE: CompensationPackage(
                base_salary=500000.0,   # $500K base
                equity_percentage=1.0,  # 1% equity
                signing_bonus=1000000.0, # $1M signing
                annual_bonus_target=500000.0,
                research_budget=250000.0,
                conference_budget=25000.0,
                relocation_package=100000.0,
                benefits_value=100000.0,
                total_package_value=2475000.0,
                tier_multiplier=2.5
            ),
            TalentTier.PREMIUM: CompensationPackage(
                base_salary=300000.0,   # $300K base
                equity_percentage=0.5,  # 0.5% equity
                signing_bonus=500000.0, # $500K signing
                annual_bonus_target=300000.0,
                research_budget=100000.0,
                conference_budget=15000.0,
                relocation_package=50000.0,
                benefits_value=75000.0,
                total_package_value=1340000.0,
                tier_multiplier=1.5
            )
        }
    
    def _initialize_retention_programs(self):
        """Initialize retention programs for acquired talent"""
        self.retention_programs = {
            "research_freedom": {
                "description": "Complete research autonomy and funding",
                "budget_per_person": 500000.0,
                "retention_impact": 0.3
            },
            "equity_acceleration": {
                "description": "Accelerated equity vesting for high performers",
                "retention_impact": 0.25
            },
            "sabbatical_program": {
                "description": "Paid sabbaticals for research and learning",
                "budget_per_person": 200000.0,
                "retention_impact": 0.2
            },
            "conference_speaking": {
                "description": "Global conference speaking opportunities",
                "budget_per_person": 50000.0,
                "retention_impact": 0.15
            },
            "publication_support": {
                "description": "Support for research publications and patents",
                "budget_per_person": 100000.0,
                "retention_impact": 0.2
            }
        }
    
    async def identify_global_talent(self, category: TalentCategory, 
                                   target_count: int = 1000) -> List[TalentProfile]:
        """
        Identify top global talent in specified category
        
        Args:
            category: Target talent category
            target_count: Number of talents to identify
            
        Returns:
            List of identified talent profiles
        """
        logger.info(f"Identifying top {target_count} talents in {category.value}")
        
        # Simulate comprehensive talent identification
        identified_talents = []
        
        # Search strategies
        search_strategies = [
            "github_analysis",
            "research_paper_analysis", 
            "conference_speaker_tracking",
            "patent_holder_identification",
            "startup_founder_tracking",
            "big_tech_employee_mapping",
            "academic_researcher_identification",
            "open_source_contributor_analysis"
        ]
        
        for i in range(target_count):
            talent_id = f"{category.value}_{i:04d}"
            
            # Determine talent tier based on achievements
            tier = self._determine_talent_tier(i, target_count)
            
            talent = TalentProfile(
                id=talent_id,
                name=f"Talent_{talent_id}",
                category=category,
                tier=tier,
                current_company=self._get_likely_company(tier),
                location=self._get_talent_location(),
                skills=self._get_category_skills(category),
                achievements=self._generate_achievements(tier),
                publications=self._generate_publications(tier),
                patents=self._generate_patents(tier),
                github_profile=f"https://github.com/talent_{talent_id}",
                linkedin_profile=f"https://linkedin.com/in/talent_{talent_id}",
                compensation_estimate=self._estimate_current_compensation(tier),
                acquisition_priority=self._calculate_priority(tier, category),
                recruitment_status=RecruitmentStatus.IDENTIFIED,
                contact_history=[],
                retention_score=0.0,
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            
            identified_talents.append(talent)
            self.talent_database[talent_id] = talent
        
        logger.info(f"Identified {len(identified_talents)} talents in {category.value}")
        return identified_talents
    
    def _determine_talent_tier(self, index: int, total: int) -> TalentTier:
        """Determine talent tier based on ranking"""
        percentile = index / total  # Use index directly for better distribution
        
        if percentile <= 0.001:  # Top 0.1%
            return TalentTier.LEGENDARY
        elif percentile <= 0.01:  # Top 1%
            return TalentTier.EXCEPTIONAL
        elif percentile <= 0.05:  # Top 5%
            return TalentTier.ELITE
        else:
            return TalentTier.PREMIUM
    
    def _get_likely_company(self, tier: TalentTier) -> str:
        """Get likely current company based on talent tier"""
        legendary_companies = ["Google DeepMind", "OpenAI", "Anthropic", "Meta AI"]
        exceptional_companies = ["Google", "Microsoft", "Apple", "Amazon", "Tesla"]
        elite_companies = ["Netflix", "Uber", "Airbnb", "Stripe", "Databricks"]
        premium_companies = ["Various Tech Companies"]
        
        if tier == TalentTier.LEGENDARY:
            return legendary_companies[0]  # Simplified
        elif tier == TalentTier.EXCEPTIONAL:
            return exceptional_companies[0]
        elif tier == TalentTier.ELITE:
            return elite_companies[0]
        else:
            return premium_companies[0]
    
    def _get_talent_location(self) -> str:
        """Get talent location from global distribution"""
        locations = [
            "San Francisco, CA", "Seattle, WA", "New York, NY",
            "London, UK", "Toronto, CA", "Berlin, DE",
            "Singapore", "Tokyo, JP", "Sydney, AU"
        ]
        return locations[0]  # Simplified
    
    def _get_category_skills(self, category: TalentCategory) -> List[str]:
        """Get relevant skills for talent category"""
        skill_map = {
            TalentCategory.AI_RESEARCHER: [
                "Deep Learning", "Neural Networks", "Transformers",
                "Reinforcement Learning", "Computer Vision", "NLP"
            ],
            TalentCategory.ML_ENGINEER: [
                "MLOps", "TensorFlow", "PyTorch", "Kubernetes",
                "Model Deployment", "Data Pipelines"
            ],
            TalentCategory.SOFTWARE_ARCHITECT: [
                "System Design", "Microservices", "Cloud Architecture",
                "Distributed Systems", "API Design", "Performance Optimization"
            ]
        }
        return skill_map.get(category, ["General Technical Skills"])
    
    def _generate_achievements(self, tier: TalentTier) -> List[str]:
        """Generate achievements based on talent tier"""
        base_achievements = ["Technical Excellence", "Team Leadership"]
        
        if tier == TalentTier.LEGENDARY:
            return base_achievements + [
                "Breakthrough Research Publication",
                "Industry-Changing Innovation",
                "Multiple Patent Portfolio",
                "Global Recognition Award"
            ]
        elif tier == TalentTier.EXCEPTIONAL:
            return base_achievements + [
                "Significant Research Contribution",
                "Product Innovation",
                "Patent Portfolio"
            ]
        else:
            return base_achievements
    
    def _generate_publications(self, tier: TalentTier) -> List[str]:
        """Generate publications based on talent tier"""
        if tier == TalentTier.LEGENDARY:
            return ["Nature AI Paper", "Science ML Paper", "NIPS Best Paper"]
        elif tier == TalentTier.EXCEPTIONAL:
            return ["ICML Paper", "ICLR Paper"]
        elif tier == TalentTier.ELITE:
            return ["Conference Paper"]
        else:
            return []
    
    def _generate_patents(self, tier: TalentTier) -> List[str]:
        """Generate patents based on talent tier"""
        if tier == TalentTier.LEGENDARY:
            return ["AI Algorithm Patent", "ML System Patent", "Data Processing Patent"]
        elif tier == TalentTier.EXCEPTIONAL:
            return ["ML Algorithm Patent"]
        else:
            return []
    
    def _estimate_current_compensation(self, tier: TalentTier) -> float:
        """Estimate current compensation for talent tier"""
        estimates = {
            TalentTier.LEGENDARY: 5000000.0,    # $5M
            TalentTier.EXCEPTIONAL: 2000000.0,  # $2M
            TalentTier.ELITE: 800000.0,         # $800K
            TalentTier.PREMIUM: 400000.0        # $400K
        }
        return estimates.get(tier, 300000.0)
    
    def _calculate_priority(self, tier: TalentTier, category: TalentCategory) -> int:
        """Calculate acquisition priority (1-10 scale)"""
        tier_priority = {
            TalentTier.LEGENDARY: 8,  # Reduced to allow for category boost
            TalentTier.EXCEPTIONAL: 6,
            TalentTier.ELITE: 4,
            TalentTier.PREMIUM: 2
        }
        
        # AI researchers get priority boost
        category_boost = 2 if category == TalentCategory.AI_RESEARCHER else 0
        
        return min(10, tier_priority.get(tier, 1) + category_boost)
    
    async def create_recruitment_campaign(self, category: TalentCategory,
                                        target_tier: TalentTier,
                                        target_count: int,
                                        budget: float) -> RecruitmentCampaign:
        """
        Create strategic recruitment campaign
        
        Args:
            category: Target talent category
            target_tier: Target talent tier
            target_count: Number of talents to acquire
            budget: Campaign budget
            
        Returns:
            Created recruitment campaign
        """
        campaign_id = f"campaign_{category.value}_{target_tier.value}_{datetime.now().strftime('%Y%m%d')}"
        
        # Define recruitment strategies based on tier
        strategies = self._get_recruitment_strategies(target_tier)
        
        campaign = RecruitmentCampaign(
            id=campaign_id,
            name=f"{category.value.title()} {target_tier.value.title()} Acquisition",
            target_category=category,
            target_tier=target_tier,
            target_count=target_count,
            budget=budget,
            timeline_months=12,
            strategies=strategies,
            success_metrics={
                "acquisition_rate": 0.8,  # 80% success rate
                "retention_rate": 0.95,   # 95% retention
                "time_to_hire": 90        # 90 days average
            },
            status="active",
            created_at=datetime.now()
        )
        
        self.recruitment_campaigns[campaign_id] = campaign
        logger.info(f"Created recruitment campaign: {campaign.name}")
        
        return campaign
    
    def _get_recruitment_strategies(self, tier: TalentTier) -> List[str]:
        """Get recruitment strategies based on talent tier"""
        base_strategies = [
            "direct_outreach",
            "referral_program",
            "conference_networking",
            "social_media_engagement"
        ]
        
        if tier in [TalentTier.LEGENDARY, TalentTier.EXCEPTIONAL]:
            return base_strategies + [
                "executive_recruitment",
                "research_collaboration_offer",
                "equity_partnership_proposal",
                "sabbatical_opportunity",
                "conference_keynote_invitation"
            ]
        else:
            return base_strategies
    
    async def execute_acquisition(self, talent_id: str) -> bool:
        """
        Execute talent acquisition process
        
        Args:
            talent_id: ID of talent to acquire
            
        Returns:
            Success status of acquisition
        """
        if talent_id not in self.talent_database:
            logger.error(f"Talent {talent_id} not found in database")
            return False
        
        talent = self.talent_database[talent_id]
        logger.info(f"Executing acquisition for {talent.name} ({talent.tier.value})")
        
        # Get compensation package for tier
        compensation = self.compensation_tiers[talent.tier]
        
        # Calculate competitive offer (2x current compensation minimum)
        competitive_multiplier = max(2.0, compensation.tier_multiplier)
        offer_value = talent.compensation_estimate * competitive_multiplier
        
        # Ensure offer meets tier minimum
        if offer_value < compensation.total_package_value:
            offer_value = compensation.total_package_value
        
        # Execute acquisition steps
        acquisition_steps = [
            "initial_contact",
            "interest_assessment", 
            "detailed_discussion",
            "offer_presentation",
            "negotiation",
            "contract_signing",
            "onboarding"
        ]
        
        for step in acquisition_steps:
            success = await self._execute_acquisition_step(talent, step, offer_value)
            if not success:
                logger.warning(f"Acquisition step {step} failed for {talent.name}")
                return False
        
        # Update talent status
        talent.recruitment_status = RecruitmentStatus.ACQUIRED
        talent.updated_at = datetime.now()
        
        # Add to talent pipeline
        self.talent_pipeline.append(talent_id)
        
        logger.info(f"Successfully acquired {talent.name} with ${offer_value:,.0f} package")
        return True
    
    async def _execute_acquisition_step(self, talent: TalentProfile, 
                                      step: str, offer_value: float) -> bool:
        """Execute individual acquisition step"""
        # Simulate acquisition step execution
        logger.debug(f"Executing {step} for {talent.name}")
        
        # Update contact history
        contact_entry = {
            "timestamp": datetime.now().isoformat(),
            "step": step,
            "status": "completed",
            "offer_value": offer_value if step == "offer_presentation" else None
        }
        talent.contact_history.append(contact_entry)
        
        # Simulate success probability based on tier and offer
        success_probability = self._calculate_acquisition_probability(talent, offer_value)
        
        # Simulate step execution (always succeed for demo)
        return True
    
    def _calculate_acquisition_probability(self, talent: TalentProfile, 
                                         offer_value: float) -> float:
        """Calculate probability of successful acquisition"""
        # Base probability by tier
        base_prob = {
            TalentTier.LEGENDARY: 0.6,     # Harder to acquire
            TalentTier.EXCEPTIONAL: 0.7,
            TalentTier.ELITE: 0.8,
            TalentTier.PREMIUM: 0.9
        }
        
        # Offer multiplier effect
        offer_multiplier = offer_value / talent.compensation_estimate
        multiplier_bonus = min(0.3, (offer_multiplier - 1.0) * 0.1)
        
        return min(0.95, base_prob.get(talent.tier, 0.5) + multiplier_bonus)
    
    async def implement_retention_program(self, talent_id: str) -> Dict[str, float]:
        """
        Implement comprehensive retention program for acquired talent
        
        Args:
            talent_id: ID of talent to retain
            
        Returns:
            Retention program metrics
        """
        if talent_id not in self.talent_database:
            logger.error(f"Talent {talent_id} not found")
            return {}
        
        talent = self.talent_database[talent_id]
        logger.info(f"Implementing retention program for {talent.name}")
        
        # Apply all retention programs
        total_retention_impact = 0.0
        total_retention_cost = 0.0
        
        for program_name, program_details in self.retention_programs.items():
            # Apply program
            retention_impact = program_details["retention_impact"]
            program_cost = program_details.get("budget_per_person", 0.0)
            
            total_retention_impact += retention_impact
            total_retention_cost += program_cost
            
            logger.debug(f"Applied {program_name} to {talent.name}")
        
        # Calculate final retention score
        base_retention = 0.7  # 70% base retention
        final_retention_score = min(0.99, base_retention + total_retention_impact)
        
        # Update talent retention score
        talent.retention_score = final_retention_score
        talent.recruitment_status = RecruitmentStatus.RETAINED
        talent.updated_at = datetime.now()
        
        metrics = {
            "retention_score": final_retention_score,
            "retention_cost": total_retention_cost,
            "programs_applied": len(self.retention_programs),
            "retention_probability": final_retention_score
        }
        
        logger.info(f"Retention program implemented for {talent.name}: {final_retention_score:.2%} retention probability")
        return metrics
    
    async def monitor_talent_pipeline(self) -> Dict[str, any]:
        """
        Monitor and analyze talent pipeline performance
        
        Returns:
            Pipeline analytics and metrics
        """
        logger.info("Monitoring talent pipeline performance")
        
        # Analyze talent distribution
        tier_distribution = {}
        category_distribution = {}
        status_distribution = {}
        
        total_compensation_cost = 0.0
        total_retention_cost = 0.0
        
        for talent_id in self.talent_pipeline:
            if talent_id in self.talent_database:
                talent = self.talent_database[talent_id]
                
                # Count by tier
                tier_key = talent.tier.value
                tier_distribution[tier_key] = tier_distribution.get(tier_key, 0) + 1
                
                # Count by category
                cat_key = talent.category.value
                category_distribution[cat_key] = category_distribution.get(cat_key, 0) + 1
                
                # Count by status
                status_key = talent.recruitment_status.value
                status_distribution[status_key] = status_distribution.get(status_key, 0) + 1
                
                # Calculate costs
                if talent.tier in self.compensation_tiers:
                    comp_package = self.compensation_tiers[talent.tier]
                    total_compensation_cost += comp_package.total_package_value
                
                # Retention costs
                for program_details in self.retention_programs.values():
                    total_retention_cost += program_details.get("budget_per_person", 0.0)
        
        # Calculate success metrics
        total_talents = len(self.talent_database)  # Use total database, not just pipeline
        pipeline_size = len(self.talent_pipeline)
        acquired_count = status_distribution.get("acquired", 0)
        retained_count = status_distribution.get("retained", 0)
        
        acquisition_rate = pipeline_size / max(1, total_talents)  # Pipeline vs total identified
        retention_rate = retained_count / max(1, acquired_count)
        
        pipeline_metrics = {
            "total_talents": total_talents,
            "pipeline_size": pipeline_size,
            "tier_distribution": tier_distribution,
            "category_distribution": category_distribution,
            "status_distribution": status_distribution,
            "acquisition_rate": acquisition_rate,
            "retention_rate": retention_rate,
            "total_compensation_cost": total_compensation_cost,
            "total_retention_cost": total_retention_cost,
            "average_compensation": total_compensation_cost / max(1, pipeline_size),
            "pipeline_value": total_compensation_cost + total_retention_cost
        }
        
        logger.info(f"Pipeline monitoring complete: {total_talents} talents, "
                   f"{acquisition_rate:.1%} acquisition rate, "
                   f"{retention_rate:.1%} retention rate")
        
        return pipeline_metrics
    
    async def analyze_competitive_landscape(self) -> Dict[str, any]:
        """
        Analyze competitive talent acquisition landscape
        
        Returns:
            Competitive intelligence and recommendations
        """
        logger.info("Analyzing competitive talent landscape")
        
        # Major competitors in talent acquisition
        competitors = {
            "Google": {
                "talent_count": 50000,
                "ai_researchers": 2000,
                "average_compensation": 400000,
                "retention_rate": 0.85,
                "acquisition_budget": 2000000000  # $2B
            },
            "Microsoft": {
                "talent_count": 45000,
                "ai_researchers": 1500,
                "average_compensation": 380000,
                "retention_rate": 0.82,
                "acquisition_budget": 1800000000  # $1.8B
            },
            "Meta": {
                "talent_count": 35000,
                "ai_researchers": 1200,
                "average_compensation": 420000,
                "retention_rate": 0.78,
                "acquisition_budget": 1500000000  # $1.5B
            },
            "OpenAI": {
                "talent_count": 800,
                "ai_researchers": 400,
                "average_compensation": 800000,
                "retention_rate": 0.90,
                "acquisition_budget": 500000000  # $500M
            }
        }
        
        # Calculate our competitive position
        our_metrics = await self.monitor_talent_pipeline()
        
        competitive_analysis = {
            "competitors": competitors,
            "our_position": {
                "talent_count": our_metrics["total_talents"],
                "average_compensation": our_metrics["average_compensation"],
                "retention_rate": our_metrics["retention_rate"],
                "total_budget": our_metrics["pipeline_value"]
            },
            "competitive_advantages": [
                "Superior compensation packages (2-10x multiplier)",
                "Complete research freedom and funding",
                "Equity participation in breakthrough technology",
                "Global talent monopoly strategy",
                "Unlimited resource access"
            ],
            "recommendations": [
                "Target legendary tier talents from competitors",
                "Offer 3x current compensation as minimum",
                "Provide research sabbaticals and publication support",
                "Create exclusive AI research opportunities",
                "Implement immediate equity acceleration programs"
            ]
        }
        
        logger.info("Competitive landscape analysis complete")
        return competitive_analysis
    
    def get_talent_statistics(self) -> Dict[str, any]:
        """Get comprehensive talent statistics"""
        total_talents = len(self.talent_database)
        
        if total_talents == 0:
            return {"total_talents": 0, "message": "No talents in database"}
        
        # Calculate statistics
        tier_counts = {}
        category_counts = {}
        status_counts = {}
        
        for talent in self.talent_database.values():
            tier_counts[talent.tier.value] = tier_counts.get(talent.tier.value, 0) + 1
            category_counts[talent.category.value] = category_counts.get(talent.category.value, 0) + 1
            status_counts[talent.recruitment_status.value] = status_counts.get(talent.recruitment_status.value, 0) + 1
        
        return {
            "total_talents": total_talents,
            "tier_distribution": tier_counts,
            "category_distribution": category_counts,
            "status_distribution": status_counts,
            "campaigns_active": len(self.recruitment_campaigns),
            "pipeline_size": len(self.talent_pipeline)
        }


# Global talent monopoly engine instance
talent_monopoly_engine = TalentMonopolyEngine()