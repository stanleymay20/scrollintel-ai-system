"""
Demand Creation Engine - Strategic demand generation through proof-of-concept deployments
and thought leadership to create inevitable market demand
"""

from typing import Dict, List, Optional, Any, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import logging

logger = logging.getLogger(__name__)

class ProofOfConceptType(Enum):
    PILOT_DEPLOYMENT = "pilot_deployment"
    TECHNICAL_DEMO = "technical_demo"
    BENCHMARK_STUDY = "benchmark_study"
    CASE_STUDY = "case_study"
    INDUSTRY_SHOWCASE = "industry_showcase"

class ThoughtLeadershipType(Enum):
    RESEARCH_PAPER = "research_paper"
    INDUSTRY_REPORT = "industry_report"
    CONFERENCE_KEYNOTE = "conference_keynote"
    EXPERT_PANEL = "expert_panel"
    MEDIA_INTERVIEW = "media_interview"
    BLOG_SERIES = "blog_series"
    PODCAST_SERIES = "podcast_series"

class IndustryVertical(Enum):
    TECHNOLOGY = "technology"
    FINANCIAL_SERVICES = "financial_services"
    HEALTHCARE = "healthcare"
    MANUFACTURING = "manufacturing"
    RETAIL = "retail"
    ENERGY = "energy"
    TELECOMMUNICATIONS = "telecommunications"
    GOVERNMENT = "government"

class DemandStage(Enum):
    AWARENESS = "awareness"
    INTEREST = "interest"
    CONSIDERATION = "consideration"
    INTENT = "intent"
    EVALUATION = "evaluation"
    PURCHASE = "purchase"

@dataclass
class ProofOfConcept:
    id: str
    name: str
    poc_type: ProofOfConceptType
    target_enterprise: str
    industry_vertical: IndustryVertical
    objectives: List[str]
    success_metrics: Dict[str, float]
    timeline_weeks: int
    investment_required: float
    expected_roi: float
    status: str = "planned"
    results: Dict[str, Any] = field(default_factory=dict)
    lessons_learned: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None

@dataclass
class ThoughtLeadershipPiece:
    id: str
    title: str
    content_type: ThoughtLeadershipType
    target_audience: List[str]
    key_messages: List[str]
    distribution_channels: List[str]
    author_credentials: str
    publication_date: datetime
    engagement_metrics: Dict[str, int] = field(default_factory=dict)
    influence_score: float = 0.0
    citations: int = 0
    media_mentions: int = 0

@dataclass
class IndustryStandard:
    id: str
    name: str
    description: str
    industry_vertical: IndustryVertical
    standard_type: str  # "technical", "operational", "ethical", "performance"
    development_stage: str  # "proposal", "draft", "review", "approved", "adopted"
    stakeholders: List[str]
    adoption_rate: float = 0.0
    competitive_advantage: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class DemandMetrics:
    stage: DemandStage
    volume: int
    quality_score: float  # 0-1
    conversion_rate: float  # to next stage
    velocity: float  # days to progress
    sources: Dict[str, int]  # source -> count
    last_updated: datetime = field(default_factory=datetime.now)

class DemandCreationEngine:
    """
    Creates inevitable market demand through strategic proof-of-concept deployments,
    thought leadership, and industry standard creation.
    """
    
    def __init__(self):
        self.proof_of_concepts: Dict[str, ProofOfConcept] = {}
        self.thought_leadership: Dict[str, ThoughtLeadershipPiece] = {}
        self.industry_standards: Dict[str, IndustryStandard] = {}
        self.demand_pipeline: Dict[DemandStage, DemandMetrics] = {}
        self.enterprise_targets: Dict[str, Dict] = {}
        self.influence_network: Dict[str, List[str]] = {}
        
    async def deploy_proof_of_concept(
        self, 
        enterprise_name: str,
        industry: IndustryVertical,
        poc_type: ProofOfConceptType,
        objectives: List[str],
        timeline_weeks: int = 12
    ) -> ProofOfConcept:
        """Deploy proof-of-concept at target enterprise"""
        try:
            poc_id = f"poc_{len(self.proof_of_concepts) + 1}_{enterprise_name.lower().replace(' ', '_')}"
            
            # Calculate investment and expected ROI based on POC type and industry
            investment_multipliers = {
                ProofOfConceptType.PILOT_DEPLOYMENT: 1.0,
                ProofOfConceptType.TECHNICAL_DEMO: 0.3,
                ProofOfConceptType.BENCHMARK_STUDY: 0.5,
                ProofOfConceptType.CASE_STUDY: 0.2,
                ProofOfConceptType.INDUSTRY_SHOWCASE: 0.7
            }
            
            industry_multipliers = {
                IndustryVertical.TECHNOLOGY: 1.2,
                IndustryVertical.FINANCIAL_SERVICES: 1.5,
                IndustryVertical.HEALTHCARE: 1.3,
                IndustryVertical.MANUFACTURING: 1.0,
                IndustryVertical.RETAIL: 0.8,
                IndustryVertical.ENERGY: 1.1,
                IndustryVertical.TELECOMMUNICATIONS: 1.0,
                IndustryVertical.GOVERNMENT: 0.9
            }
            
            base_investment = 500000  # $500K base
            investment = base_investment * investment_multipliers[poc_type] * industry_multipliers[industry]
            expected_roi = investment * (2.5 + (timeline_weeks / 52))  # ROI increases with timeline
            
            poc = ProofOfConcept(
                id=poc_id,
                name=f"{enterprise_name} {poc_type.value.title()} POC",
                poc_type=poc_type,
                target_enterprise=enterprise_name,
                industry_vertical=industry,
                objectives=objectives,
                success_metrics=self._generate_success_metrics(poc_type, objectives),
                timeline_weeks=timeline_weeks,
                investment_required=investment,
                expected_roi=expected_roi,
                status="initiated"
            )
            
            self.proof_of_concepts[poc_id] = poc
            
            # Add to enterprise targets
            if enterprise_name not in self.enterprise_targets:
                self.enterprise_targets[enterprise_name] = {
                    "industry": industry.value,
                    "pocs": [],
                    "engagement_level": "initial",
                    "decision_makers": [],
                    "influence_score": 0.0
                }
            
            self.enterprise_targets[enterprise_name]["pocs"].append(poc_id)
            
            logger.info(f"Deployed POC: {poc.name} (Investment: ${investment:,.0f})")
            return poc
            
        except Exception as e:
            logger.error(f"Error deploying POC: {str(e)}")
            raise
    
    def _generate_success_metrics(self, poc_type: ProofOfConceptType, objectives: List[str]) -> Dict[str, float]:
        """Generate success metrics based on POC type and objectives"""
        base_metrics = {
            "technical_performance": 0.0,
            "user_satisfaction": 0.0,
            "business_impact": 0.0,
            "adoption_rate": 0.0,
            "roi_achievement": 0.0
        }
        
        # Add specific metrics based on POC type
        type_specific_metrics = {
            ProofOfConceptType.PILOT_DEPLOYMENT: {
                "system_uptime": 0.0,
                "processing_efficiency": 0.0,
                "cost_reduction": 0.0
            },
            ProofOfConceptType.TECHNICAL_DEMO: {
                "feature_completeness": 0.0,
                "performance_benchmarks": 0.0,
                "integration_success": 0.0
            },
            ProofOfConceptType.BENCHMARK_STUDY: {
                "competitive_advantage": 0.0,
                "performance_improvement": 0.0,
                "accuracy_metrics": 0.0
            },
            ProofOfConceptType.CASE_STUDY: {
                "stakeholder_satisfaction": 0.0,
                "process_improvement": 0.0,
                "knowledge_transfer": 0.0
            },
            ProofOfConceptType.INDUSTRY_SHOWCASE: {
                "industry_recognition": 0.0,
                "media_coverage": 0.0,
                "peer_validation": 0.0
            }
        }
        
        base_metrics.update(type_specific_metrics.get(poc_type, {}))
        
        # Add objective-specific metrics
        for objective in objectives:
            if "cost" in objective.lower():
                base_metrics["cost_optimization"] = 0.0
            if "efficiency" in objective.lower():
                base_metrics["efficiency_gain"] = 0.0
            if "quality" in objective.lower():
                base_metrics["quality_improvement"] = 0.0
            if "speed" in objective.lower():
                base_metrics["speed_improvement"] = 0.0
        
        return base_metrics
    
    async def execute_poc_phase(self, poc_id: str, phase: str) -> Dict[str, Any]:
        """Execute specific phase of proof-of-concept"""
        try:
            if poc_id not in self.proof_of_concepts:
                raise ValueError(f"POC not found: {poc_id}")
            
            poc = self.proof_of_concepts[poc_id]
            
            # Simulate phase execution based on POC type
            phase_results = await self._simulate_poc_execution(poc, phase)
            
            # Update POC results
            if "phases" not in poc.results:
                poc.results["phases"] = {}
            
            poc.results["phases"][phase] = {
                "completed_at": datetime.now().isoformat(),
                "results": phase_results,
                "success_rate": phase_results.get("success_rate", 0.8)
            }
            
            # Update success metrics
            for metric, value in phase_results.get("metrics", {}).items():
                if metric in poc.success_metrics:
                    poc.success_metrics[metric] = value
            
            # Check if POC is complete
            total_phases = phase_results.get("total_phases", 4)
            completed_phases = len(poc.results.get("phases", {}))
            
            if completed_phases >= total_phases:
                poc.status = "completed"
                poc.completed_at = datetime.now()
                await self._finalize_poc_results(poc)
            
            logger.info(f"Completed phase '{phase}' for POC: {poc.name}")
            return phase_results
            
        except Exception as e:
            logger.error(f"Error executing POC phase: {str(e)}")
            raise
    
    async def _simulate_poc_execution(self, poc: ProofOfConcept, phase: str) -> Dict[str, Any]:
        """Simulate POC phase execution with realistic results"""
        import random
        
        # Base success rates by POC type
        success_rates = {
            ProofOfConceptType.PILOT_DEPLOYMENT: 0.85,
            ProofOfConceptType.TECHNICAL_DEMO: 0.90,
            ProofOfConceptType.BENCHMARK_STUDY: 0.88,
            ProofOfConceptType.CASE_STUDY: 0.92,
            ProofOfConceptType.INDUSTRY_SHOWCASE: 0.87
        }
        
        base_success_rate = success_rates.get(poc.poc_type, 0.85)
        
        # Industry factors
        industry_factors = {
            IndustryVertical.TECHNOLOGY: 1.1,
            IndustryVertical.FINANCIAL_SERVICES: 0.9,
            IndustryVertical.HEALTHCARE: 0.95,
            IndustryVertical.MANUFACTURING: 1.0,
            IndustryVertical.RETAIL: 1.05,
            IndustryVertical.ENERGY: 0.98,
            IndustryVertical.TELECOMMUNICATIONS: 1.02,
            IndustryVertical.GOVERNMENT: 0.92
        }
        
        industry_factor = industry_factors.get(poc.industry_vertical, 1.0)
        final_success_rate = min(0.98, base_success_rate * industry_factor)
        
        # Generate phase-specific results
        phase_results = {
            "phase": phase,
            "success_rate": final_success_rate,
            "total_phases": 4,
            "metrics": {},
            "achievements": [],
            "challenges": [],
            "stakeholder_feedback": []
        }
        
        # Generate realistic metrics
        for metric in poc.success_metrics.keys():
            if random.random() < final_success_rate:
                # Successful metric
                base_value = random.uniform(0.7, 0.95)
                phase_results["metrics"][metric] = base_value
                
                if base_value > 0.85:
                    phase_results["achievements"].append(f"Exceeded expectations in {metric}")
            else:
                # Challenging metric
                base_value = random.uniform(0.4, 0.7)
                phase_results["metrics"][metric] = base_value
                phase_results["challenges"].append(f"Below target performance in {metric}")
        
        # Generate stakeholder feedback
        feedback_templates = [
            "Impressed with technical capabilities and potential",
            "Significant improvement over current solution",
            "Concerns about integration complexity",
            "Strong business case demonstrated",
            "Positive user experience feedback",
            "Questions about scalability and maintenance"
        ]
        
        num_feedback = random.randint(2, 4)
        phase_results["stakeholder_feedback"] = random.sample(feedback_templates, num_feedback)
        
        return phase_results
    
    async def _finalize_poc_results(self, poc: ProofOfConcept):
        """Finalize POC results and calculate overall impact"""
        try:
            # Calculate overall success score
            metric_scores = list(poc.success_metrics.values())
            overall_score = sum(metric_scores) / len(metric_scores) if metric_scores else 0
            
            # Generate lessons learned
            if overall_score > 0.8:
                poc.lessons_learned.extend([
                    "Strong technical performance validates AI CTO capabilities",
                    "Positive stakeholder reception indicates market readiness",
                    "Successful integration demonstrates practical viability"
                ])
            elif overall_score > 0.6:
                poc.lessons_learned.extend([
                    "Good performance with areas for improvement identified",
                    "Mixed stakeholder feedback provides valuable insights",
                    "Technical challenges highlight development priorities"
                ])
            else:
                poc.lessons_learned.extend([
                    "Significant challenges identified requiring attention",
                    "Stakeholder concerns need to be addressed",
                    "Technical improvements necessary before broader deployment"
                ])
            
            # Update enterprise engagement
            enterprise = self.enterprise_targets[poc.target_enterprise]
            if overall_score > 0.8:
                enterprise["engagement_level"] = "high_interest"
                enterprise["influence_score"] = min(1.0, enterprise["influence_score"] + 0.3)
            elif overall_score > 0.6:
                enterprise["engagement_level"] = "moderate_interest"
                enterprise["influence_score"] = min(1.0, enterprise["influence_score"] + 0.2)
            else:
                enterprise["engagement_level"] = "cautious"
                enterprise["influence_score"] = min(1.0, enterprise["influence_score"] + 0.1)
            
            # Calculate actual ROI
            actual_roi = poc.expected_roi * overall_score
            poc.results["final_roi"] = actual_roi
            poc.results["overall_score"] = overall_score
            
            logger.info(f"Finalized POC results: {poc.name} (Score: {overall_score:.2f})")
            
        except Exception as e:
            logger.error(f"Error finalizing POC results: {str(e)}")
            raise
    
    async def create_thought_leadership_content(
        self,
        title: str,
        content_type: ThoughtLeadershipType,
        target_audience: List[str],
        key_messages: List[str],
        author_credentials: str
    ) -> ThoughtLeadershipPiece:
        """Create thought leadership content to establish industry authority"""
        try:
            content_id = f"tl_{len(self.thought_leadership) + 1}_{title.lower().replace(' ', '_')[:20]}"
            
            # Determine distribution channels based on content type
            distribution_channels = self._get_distribution_channels(content_type)
            
            content = ThoughtLeadershipPiece(
                id=content_id,
                title=title,
                content_type=content_type,
                target_audience=target_audience,
                key_messages=key_messages,
                distribution_channels=distribution_channels,
                author_credentials=author_credentials,
                publication_date=datetime.now() + timedelta(days=7)  # 1 week lead time
            )
            
            self.thought_leadership[content_id] = content
            
            # Schedule content promotion
            await self._schedule_content_promotion(content)
            
            logger.info(f"Created thought leadership content: {title}")
            return content
            
        except Exception as e:
            logger.error(f"Error creating thought leadership content: {str(e)}")
            raise
    
    def _get_distribution_channels(self, content_type: ThoughtLeadershipType) -> List[str]:
        """Get appropriate distribution channels for content type"""
        channel_mapping = {
            ThoughtLeadershipType.RESEARCH_PAPER: [
                "academic_journals", "industry_publications", "company_website", "linkedin"
            ],
            ThoughtLeadershipType.INDUSTRY_REPORT: [
                "industry_media", "analyst_firms", "company_website", "email_newsletter"
            ],
            ThoughtLeadershipType.CONFERENCE_KEYNOTE: [
                "conference_proceedings", "video_platforms", "social_media", "press_release"
            ],
            ThoughtLeadershipType.EXPERT_PANEL: [
                "industry_events", "webinar_platforms", "podcast_networks", "social_media"
            ],
            ThoughtLeadershipType.MEDIA_INTERVIEW: [
                "traditional_media", "online_publications", "podcast_platforms", "social_media"
            ],
            ThoughtLeadershipType.BLOG_SERIES: [
                "company_blog", "medium", "linkedin_articles", "industry_blogs"
            ],
            ThoughtLeadershipType.PODCAST_SERIES: [
                "podcast_platforms", "company_website", "social_media", "email_newsletter"
            ]
        }
        
        return channel_mapping.get(content_type, ["company_website", "social_media"])
    
    async def _schedule_content_promotion(self, content: ThoughtLeadershipPiece):
        """Schedule promotion activities for thought leadership content"""
        try:
            promotion_schedule = {
                "pre_launch": datetime.now() + timedelta(days=3),
                "launch": content.publication_date,
                "post_launch_1": content.publication_date + timedelta(days=7),
                "post_launch_2": content.publication_date + timedelta(days=30)
            }
            
            # Simulate scheduling promotion activities
            logger.info(f"Scheduled promotion for: {content.title}")
            
        except Exception as e:
            logger.error(f"Error scheduling content promotion: {str(e)}")
            raise
    
    async def track_content_engagement(self, content_id: str, engagement_data: Dict[str, int]):
        """Track engagement metrics for thought leadership content"""
        try:
            if content_id not in self.thought_leadership:
                raise ValueError(f"Content not found: {content_id}")
            
            content = self.thought_leadership[content_id]
            content.engagement_metrics.update(engagement_data)
            
            # Calculate influence score
            influence_score = (
                engagement_data.get("views", 0) * 0.1 +
                engagement_data.get("shares", 0) * 0.3 +
                engagement_data.get("comments", 0) * 0.5 +
                engagement_data.get("citations", 0) * 1.0 +
                engagement_data.get("media_mentions", 0) * 2.0
            ) / 1000  # Normalize to 0-1 scale
            
            content.influence_score = min(1.0, influence_score)
            content.citations = engagement_data.get("citations", 0)
            content.media_mentions = engagement_data.get("media_mentions", 0)
            
            logger.info(f"Updated engagement for: {content.title} (Influence: {content.influence_score:.3f})")
            
        except Exception as e:
            logger.error(f"Error tracking content engagement: {str(e)}")
            raise
    
    async def create_industry_standard(
        self,
        name: str,
        description: str,
        industry: IndustryVertical,
        standard_type: str,
        stakeholders: List[str]
    ) -> IndustryStandard:
        """Create and promote new industry standard that favors ScrollIntel"""
        try:
            standard_id = f"std_{len(self.industry_standards) + 1}_{name.lower().replace(' ', '_')[:20]}"
            
            standard = IndustryStandard(
                id=standard_id,
                name=name,
                description=description,
                industry_vertical=industry,
                standard_type=standard_type,
                development_stage="proposal",
                stakeholders=stakeholders
            )
            
            self.industry_standards[standard_id] = standard
            
            # Begin standard development process
            await self._initiate_standard_development(standard)
            
            logger.info(f"Created industry standard: {name}")
            return standard
            
        except Exception as e:
            logger.error(f"Error creating industry standard: {str(e)}")
            raise
    
    async def _initiate_standard_development(self, standard: IndustryStandard):
        """Initiate the industry standard development process"""
        try:
            # Simulate standard development phases
            development_phases = [
                "stakeholder_engagement",
                "draft_development", 
                "industry_review",
                "revision_cycle",
                "approval_process",
                "adoption_campaign"
            ]
            
            # Schedule development activities
            logger.info(f"Initiated standard development: {standard.name}")
            
        except Exception as e:
            logger.error(f"Error initiating standard development: {str(e)}")
            raise
    
    async def advance_standard_development(self, standard_id: str, new_stage: str) -> Dict[str, Any]:
        """Advance industry standard through development stages"""
        try:
            if standard_id not in self.industry_standards:
                raise ValueError(f"Standard not found: {standard_id}")
            
            standard = self.industry_standards[standard_id]
            previous_stage = standard.development_stage
            standard.development_stage = new_stage
            
            # Calculate adoption rate and competitive advantage based on stage
            stage_multipliers = {
                "proposal": {"adoption": 0.0, "advantage": 0.1},
                "draft": {"adoption": 0.05, "advantage": 0.2},
                "review": {"adoption": 0.15, "advantage": 0.4},
                "approved": {"adoption": 0.40, "advantage": 0.7},
                "adopted": {"adoption": 0.75, "advantage": 0.9}
            }
            
            multipliers = stage_multipliers.get(new_stage, {"adoption": 0.0, "advantage": 0.0})
            standard.adoption_rate = multipliers["adoption"]
            standard.competitive_advantage = multipliers["advantage"]
            
            advancement_result = {
                "standard_id": standard_id,
                "previous_stage": previous_stage,
                "new_stage": new_stage,
                "adoption_rate": standard.adoption_rate,
                "competitive_advantage": standard.competitive_advantage,
                "advancement_date": datetime.now().isoformat()
            }
            
            logger.info(f"Advanced standard '{standard.name}' from {previous_stage} to {new_stage}")
            return advancement_result
            
        except Exception as e:
            logger.error(f"Error advancing standard development: {str(e)}")
            raise
    
    async def measure_demand_pipeline(self) -> Dict[DemandStage, DemandMetrics]:
        """Measure current demand pipeline across all stages"""
        try:
            # Simulate demand measurement based on POCs and thought leadership
            total_pocs = len(self.proof_of_concepts)
            total_content = len(self.thought_leadership)
            total_standards = len(self.industry_standards)
            
            # Calculate demand metrics for each stage
            pipeline_metrics = {}
            
            for stage in DemandStage:
                # Base volume calculation
                base_volume = self._calculate_stage_volume(stage, total_pocs, total_content, total_standards)
                
                # Quality score based on POC success and content influence
                quality_score = self._calculate_quality_score(stage)
                
                # Conversion rate to next stage
                conversion_rate = self._calculate_conversion_rate(stage)
                
                # Velocity (days to progress to next stage)
                velocity = self._calculate_stage_velocity(stage)
                
                # Sources breakdown
                sources = self._calculate_demand_sources(stage, total_pocs, total_content, total_standards)
                
                metrics = DemandMetrics(
                    stage=stage,
                    volume=base_volume,
                    quality_score=quality_score,
                    conversion_rate=conversion_rate,
                    velocity=velocity,
                    sources=sources
                )
                
                pipeline_metrics[stage] = metrics
            
            self.demand_pipeline = pipeline_metrics
            logger.info("Updated demand pipeline measurements")
            
            return pipeline_metrics
            
        except Exception as e:
            logger.error(f"Error measuring demand pipeline: {str(e)}")
            raise
    
    def _calculate_stage_volume(self, stage: DemandStage, pocs: int, content: int, standards: int) -> int:
        """Calculate volume for specific demand stage"""
        stage_multipliers = {
            DemandStage.AWARENESS: {"poc": 50, "content": 100, "standard": 25},
            DemandStage.INTEREST: {"poc": 30, "content": 60, "standard": 15},
            DemandStage.CONSIDERATION: {"poc": 20, "content": 30, "standard": 10},
            DemandStage.INTENT: {"poc": 15, "content": 15, "standard": 8},
            DemandStage.EVALUATION: {"poc": 10, "content": 8, "standard": 5},
            DemandStage.PURCHASE: {"poc": 5, "content": 3, "standard": 2}
        }
        
        multipliers = stage_multipliers.get(stage, {"poc": 10, "content": 10, "standard": 5})
        
        volume = (
            pocs * multipliers["poc"] +
            content * multipliers["content"] +
            standards * multipliers["standard"]
        )
        
        return max(1, volume)
    
    def _calculate_quality_score(self, stage: DemandStage) -> float:
        """Calculate quality score for demand stage"""
        # Base quality scores by stage
        base_scores = {
            DemandStage.AWARENESS: 0.6,
            DemandStage.INTEREST: 0.65,
            DemandStage.CONSIDERATION: 0.7,
            DemandStage.INTENT: 0.75,
            DemandStage.EVALUATION: 0.8,
            DemandStage.PURCHASE: 0.85
        }
        
        base_score = base_scores.get(stage, 0.6)
        
        # Adjust based on POC success rates
        if self.proof_of_concepts:
            avg_poc_score = sum(
                poc.results.get("overall_score", 0.7) 
                for poc in self.proof_of_concepts.values()
            ) / len(self.proof_of_concepts)
            
            quality_adjustment = (avg_poc_score - 0.7) * 0.2  # Max ±0.2 adjustment
            base_score = max(0.1, min(1.0, base_score + quality_adjustment))
        
        return base_score
    
    def _calculate_conversion_rate(self, stage: DemandStage) -> float:
        """Calculate conversion rate to next stage"""
        conversion_rates = {
            DemandStage.AWARENESS: 0.25,
            DemandStage.INTEREST: 0.35,
            DemandStage.CONSIDERATION: 0.45,
            DemandStage.INTENT: 0.60,
            DemandStage.EVALUATION: 0.75,
            DemandStage.PURCHASE: 1.0  # Final stage
        }
        
        return conversion_rates.get(stage, 0.3)
    
    def _calculate_stage_velocity(self, stage: DemandStage) -> float:
        """Calculate average days to progress through stage"""
        velocities = {
            DemandStage.AWARENESS: 30.0,
            DemandStage.INTEREST: 45.0,
            DemandStage.CONSIDERATION: 60.0,
            DemandStage.INTENT: 90.0,
            DemandStage.EVALUATION: 120.0,
            DemandStage.PURCHASE: 180.0
        }
        
        return velocities.get(stage, 60.0)
    
    def _calculate_demand_sources(self, stage: DemandStage, pocs: int, content: int, standards: int) -> Dict[str, int]:
        """Calculate demand sources breakdown"""
        # Source contribution varies by stage
        if stage in [DemandStage.AWARENESS, DemandStage.INTEREST]:
            return {
                "thought_leadership": int(content * 0.6),
                "industry_standards": int(standards * 0.3),
                "proof_of_concepts": int(pocs * 0.1)
            }
        elif stage in [DemandStage.CONSIDERATION, DemandStage.INTENT]:
            return {
                "proof_of_concepts": int(pocs * 0.5),
                "thought_leadership": int(content * 0.3),
                "industry_standards": int(standards * 0.2)
            }
        else:  # EVALUATION, PURCHASE
            return {
                "proof_of_concepts": int(pocs * 0.7),
                "thought_leadership": int(content * 0.2),
                "industry_standards": int(standards * 0.1)
            }
    
    async def generate_demand_forecast(self, months_ahead: int = 12) -> Dict[str, Any]:
        """Generate demand forecast based on current activities"""
        try:
            current_pipeline = await self.measure_demand_pipeline()
            
            # Project demand growth based on planned activities
            growth_factors = {
                "poc_completion_rate": 0.8,  # 80% of POCs complete successfully
                "content_influence_growth": 1.2,  # 20% monthly growth in influence
                "standard_adoption_rate": 0.6,  # 60% of standards get adopted
                "market_maturity_factor": 1.1  # 10% market maturity boost
            }
            
            forecast = {
                "forecast_period_months": months_ahead,
                "current_pipeline": {
                    stage.value: {
                        "volume": metrics.volume,
                        "quality_score": metrics.quality_score,
                        "conversion_rate": metrics.conversion_rate
                    }
                    for stage, metrics in current_pipeline.items()
                },
                "projected_pipeline": {},
                "key_assumptions": growth_factors,
                "confidence_level": 0.75,
                "generated_at": datetime.now().isoformat()
            }
            
            # Calculate projections for each stage
            for stage, current_metrics in current_pipeline.items():
                monthly_growth_rate = 0.15  # 15% monthly growth
                projected_volume = int(current_metrics.volume * (1 + monthly_growth_rate) ** months_ahead)
                
                # Quality improves with experience
                quality_improvement = min(0.2, months_ahead * 0.01)
                projected_quality = min(1.0, current_metrics.quality_score + quality_improvement)
                
                # Conversion rates improve with optimization
                conversion_improvement = min(0.1, months_ahead * 0.005)
                projected_conversion = min(1.0, current_metrics.conversion_rate + conversion_improvement)
                
                forecast["projected_pipeline"][stage.value] = {
                    "volume": projected_volume,
                    "quality_score": projected_quality,
                    "conversion_rate": projected_conversion,
                    "growth_rate": monthly_growth_rate
                }
            
            # Calculate total addressable demand
            total_current_volume = sum(m.volume for m in current_pipeline.values())
            total_projected_volume = sum(
                p["volume"] for p in forecast["projected_pipeline"].values()
            )
            
            forecast["summary"] = {
                "total_current_demand": total_current_volume,
                "total_projected_demand": total_projected_volume,
                "growth_multiple": total_projected_volume / total_current_volume if total_current_volume > 0 else 1.0,
                "revenue_potential": total_projected_volume * 2500000,  # $2.5M average deal size
                "market_penetration": min(0.25, total_projected_volume / 10000)  # Max 25% penetration
            }
            
            logger.info(f"Generated demand forecast: {total_projected_volume:,} total projected demand")
            return forecast
            
        except Exception as e:
            logger.error(f"Error generating demand forecast: {str(e)}")
            raise
    
    async def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive demand creation report"""
        try:
            # Measure current pipeline
            pipeline = await self.measure_demand_pipeline()
            
            # Generate forecast
            forecast = await self.generate_demand_forecast()
            
            # Calculate POC performance
            poc_performance = self._calculate_poc_performance()
            
            # Calculate thought leadership impact
            tl_impact = self._calculate_thought_leadership_impact()
            
            # Calculate industry standards progress
            standards_progress = self._calculate_standards_progress()
            
            report = {
                "executive_summary": {
                    "total_pocs_deployed": len(self.proof_of_concepts),
                    "successful_pocs": len([p for p in self.proof_of_concepts.values() if p.results.get("overall_score", 0) > 0.7]),
                    "thought_leadership_pieces": len(self.thought_leadership),
                    "industry_standards_created": len(self.industry_standards),
                    "total_demand_volume": sum(m.volume for m in pipeline.values()),
                    "average_demand_quality": sum(m.quality_score for m in pipeline.values()) / len(pipeline) if pipeline else 0,
                    "projected_revenue_potential": forecast["summary"]["revenue_potential"]
                },
                "proof_of_concept_performance": poc_performance,
                "thought_leadership_impact": tl_impact,
                "industry_standards_progress": standards_progress,
                "demand_pipeline": {
                    stage.value: {
                        "volume": metrics.volume,
                        "quality_score": metrics.quality_score,
                        "conversion_rate": metrics.conversion_rate,
                        "velocity_days": metrics.velocity,
                        "sources": metrics.sources
                    }
                    for stage, metrics in pipeline.items()
                },
                "demand_forecast": forecast,
                "enterprise_engagement": {
                    enterprise: {
                        "industry": data["industry"],
                        "engagement_level": data["engagement_level"],
                        "influence_score": data["influence_score"],
                        "active_pocs": len(data["pocs"])
                    }
                    for enterprise, data in self.enterprise_targets.items()
                },
                "recommendations": self._generate_strategic_recommendations(),
                "generated_at": datetime.now().isoformat()
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating comprehensive report: {str(e)}")
            raise
    
    def _calculate_poc_performance(self) -> Dict[str, Any]:
        """Calculate overall POC performance metrics"""
        if not self.proof_of_concepts:
            return {"total": 0, "performance": {}}
        
        completed_pocs = [p for p in self.proof_of_concepts.values() if p.status == "completed"]
        
        if not completed_pocs:
            return {
                "total": len(self.proof_of_concepts),
                "completed": 0,
                "in_progress": len(self.proof_of_concepts),
                "performance": {}
            }
        
        avg_score = sum(p.results.get("overall_score", 0) for p in completed_pocs) / len(completed_pocs)
        total_investment = sum(p.investment_required for p in self.proof_of_concepts.values())
        total_roi = sum(p.results.get("final_roi", 0) for p in completed_pocs)
        
        return {
            "total": len(self.proof_of_concepts),
            "completed": len(completed_pocs),
            "in_progress": len(self.proof_of_concepts) - len(completed_pocs),
            "average_success_score": avg_score,
            "total_investment": total_investment,
            "total_roi": total_roi,
            "roi_multiple": total_roi / total_investment if total_investment > 0 else 0,
            "success_rate": len([p for p in completed_pocs if p.results.get("overall_score", 0) > 0.7]) / len(completed_pocs)
        }
    
    def _calculate_thought_leadership_impact(self) -> Dict[str, Any]:
        """Calculate thought leadership impact metrics"""
        if not self.thought_leadership:
            return {"total": 0, "impact": {}}
        
        total_influence = sum(tl.influence_score for tl in self.thought_leadership.values())
        total_citations = sum(tl.citations for tl in self.thought_leadership.values())
        total_mentions = sum(tl.media_mentions for tl in self.thought_leadership.values())
        
        return {
            "total_pieces": len(self.thought_leadership),
            "total_influence_score": total_influence,
            "average_influence": total_influence / len(self.thought_leadership),
            "total_citations": total_citations,
            "total_media_mentions": total_mentions,
            "high_impact_pieces": len([tl for tl in self.thought_leadership.values() if tl.influence_score > 0.7])
        }
    
    def _calculate_standards_progress(self) -> Dict[str, Any]:
        """Calculate industry standards progress"""
        if not self.industry_standards:
            return {"total": 0, "progress": {}}
        
        stage_counts = {}
        total_adoption = sum(std.adoption_rate for std in self.industry_standards.values())
        total_advantage = sum(std.competitive_advantage for std in self.industry_standards.values())
        
        for std in self.industry_standards.values():
            stage = std.development_stage
            stage_counts[stage] = stage_counts.get(stage, 0) + 1
        
        return {
            "total_standards": len(self.industry_standards),
            "stage_breakdown": stage_counts,
            "average_adoption_rate": total_adoption / len(self.industry_standards),
            "average_competitive_advantage": total_advantage / len(self.industry_standards),
            "approved_standards": len([s for s in self.industry_standards.values() if s.development_stage in ["approved", "adopted"]])
        }
    
    def _generate_strategic_recommendations(self) -> List[str]:
        """Generate strategic recommendations based on current performance"""
        recommendations = []
        
        # POC recommendations
        poc_success_rate = len([p for p in self.proof_of_concepts.values() if p.results.get("overall_score", 0) > 0.7]) / len(self.proof_of_concepts) if self.proof_of_concepts else 0
        
        if poc_success_rate < 0.8:
            recommendations.append("Improve POC execution methodology to increase success rate")
        
        if len(self.proof_of_concepts) < 50:
            recommendations.append("Accelerate POC deployment to reach 100+ enterprise target")
        
        # Thought leadership recommendations
        avg_influence = sum(tl.influence_score for tl in self.thought_leadership.values()) / len(self.thought_leadership) if self.thought_leadership else 0
        
        if avg_influence < 0.6:
            recommendations.append("Enhance thought leadership content quality and distribution")
        
        if len(self.thought_leadership) < 20:
            recommendations.append("Increase thought leadership content production")
        
        # Industry standards recommendations
        approved_standards = len([s for s in self.industry_standards.values() if s.development_stage in ["approved", "adopted"]])
        
        if approved_standards < 3:
            recommendations.append("Accelerate industry standard development and approval process")
        
        # Demand pipeline recommendations
        if hasattr(self, 'demand_pipeline') and self.demand_pipeline:
            purchase_volume = self.demand_pipeline.get(DemandStage.PURCHASE, DemandMetrics(DemandStage.PURCHASE, 0, 0, 0, 0, {})).volume
            
            if purchase_volume < 100:
                recommendations.append("Focus on converting evaluation stage prospects to purchase")
        
        return recommendations