"""
Automated Research Engine for Autonomous Innovation Lab

This module provides autonomous research capabilities including:
- Research topic generation for promising research directions
- Comprehensive literature analysis and knowledge gap identification
- Automated hypothesis formation and testable research question generation
- Systematic research planning and methodology development
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import uuid

logger = logging.getLogger(__name__)


class ResearchDomain(Enum):
    """Research domains for topic generation"""
    ARTIFICIAL_INTELLIGENCE = "artificial_intelligence"
    MACHINE_LEARNING = "machine_learning"
    QUANTUM_COMPUTING = "quantum_computing"
    BIOTECHNOLOGY = "biotechnology"
    NANOTECHNOLOGY = "nanotechnology"
    RENEWABLE_ENERGY = "renewable_energy"
    SPACE_TECHNOLOGY = "space_technology"
    ROBOTICS = "robotics"
    BLOCKCHAIN = "blockchain"
    CYBERSECURITY = "cybersecurity"


class ResearchStatus(Enum):
    """Status of research projects"""
    PLANNING = "planning"
    ACTIVE = "active"
    ANALYSIS = "analysis"
    COMPLETED = "completed"
    PAUSED = "paused"


@dataclass
class ResearchTopic:
    """Research topic with metadata"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    domain: ResearchDomain = ResearchDomain.ARTIFICIAL_INTELLIGENCE
    description: str = ""
    keywords: List[str] = field(default_factory=list)
    novelty_score: float = 0.0
    feasibility_score: float = 0.0
    impact_potential: float = 0.0
    research_gaps: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class LiteratureSource:
    """Literature source information"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    authors: List[str] = field(default_factory=list)
    publication_year: int = 0
    journal: str = ""
    doi: str = ""
    abstract: str = ""
    keywords: List[str] = field(default_factory=list)
    citation_count: int = 0
    relevance_score: float = 0.0


@dataclass
class LiteratureAnalysis:
    """Comprehensive literature analysis results"""
    topic_id: str = ""
    sources: List[LiteratureSource] = field(default_factory=list)
    knowledge_gaps: List[str] = field(default_factory=list)
    research_trends: List[str] = field(default_factory=list)
    key_findings: List[str] = field(default_factory=list)
    methodological_gaps: List[str] = field(default_factory=list)
    theoretical_gaps: List[str] = field(default_factory=list)
    empirical_gaps: List[str] = field(default_factory=list)
    analysis_confidence: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class Hypothesis:
    """Research hypothesis with testability metrics"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    topic_id: str = ""
    statement: str = ""
    null_hypothesis: str = ""
    alternative_hypothesis: str = ""
    variables: Dict[str, str] = field(default_factory=dict)
    testability_score: float = 0.0
    novelty_score: float = 0.0
    significance_potential: float = 0.0
    required_resources: List[str] = field(default_factory=list)
    expected_timeline: timedelta = field(default_factory=lambda: timedelta(days=30))
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class ResearchMethodology:
    """Research methodology specification"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    methodology_type: str = ""  # experimental, observational, computational, theoretical
    data_collection_methods: List[str] = field(default_factory=list)
    analysis_methods: List[str] = field(default_factory=list)
    validation_approaches: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    ethical_considerations: List[str] = field(default_factory=list)


@dataclass
class ResearchPlan:
    """Systematic research plan"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    topic_id: str = ""
    hypothesis_id: str = ""
    title: str = ""
    objectives: List[str] = field(default_factory=list)
    methodology: ResearchMethodology = field(default_factory=ResearchMethodology)
    timeline: Dict[str, datetime] = field(default_factory=dict)
    milestones: List[str] = field(default_factory=list)
    resource_requirements: Dict[str, Any] = field(default_factory=dict)
    success_criteria: List[str] = field(default_factory=list)
    risk_assessment: Dict[str, float] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


class TopicGenerator:
    """Generates promising research topics autonomously"""
    
    def __init__(self):
        self.domain_expertise = {
            ResearchDomain.ARTIFICIAL_INTELLIGENCE: [
                "neural architecture search", "few-shot learning", "causal inference",
                "multimodal learning", "federated learning", "explainable AI",
                "adversarial robustness", "continual learning", "meta-learning"
            ],
            ResearchDomain.MACHINE_LEARNING: [
                "self-supervised learning", "graph neural networks", "transformer architectures",
                "reinforcement learning", "generative models", "optimization algorithms",
                "uncertainty quantification", "domain adaptation", "active learning"
            ],
            ResearchDomain.QUANTUM_COMPUTING: [
                "quantum algorithms", "quantum error correction", "quantum machine learning",
                "quantum cryptography", "quantum simulation", "quantum sensing",
                "quantum communication", "quantum supremacy", "quantum annealing"
            ]
        }
    
    async def generate_topics(self, domain: ResearchDomain, count: int = 5) -> List[ResearchTopic]:
        """Generate promising research topics for a domain"""
        try:
            topics = []
            base_keywords = self.domain_expertise.get(domain, [])
            
            for i in range(count):
                topic = await self._create_research_topic(domain, base_keywords, i)
                topics.append(topic)
            
            # Sort by combined novelty and impact potential
            topics.sort(key=lambda t: (t.novelty_score + t.impact_potential) / 2, reverse=True)
            
            logger.info(f"Generated {len(topics)} research topics for domain {domain.value}")
            return topics
            
        except Exception as e:
            logger.error(f"Error generating research topics: {str(e)}")
            return []
    
    async def _create_research_topic(self, domain: ResearchDomain, keywords: List[str], index: int) -> ResearchTopic:
        """Create a single research topic"""
        # Simulate advanced topic generation logic
        topic_templates = [
            "Novel approaches to {keyword1} using {keyword2} for enhanced {keyword3}",
            "Autonomous {keyword1} optimization through {keyword2} integration",
            "Cross-domain {keyword1} applications in {keyword2} systems",
            "Scalable {keyword1} frameworks for {keyword2} environments",
            "Adaptive {keyword1} methodologies for {keyword2} challenges"
        ]
        
        # Ensure we have at least 3 keywords
        if len(keywords) < 3:
            keywords = keywords + ["innovation", "optimization", "research", "analysis", "performance"]
        
        selected_keywords = keywords[:3]
        template = topic_templates[index % len(topic_templates)]
        
        title = template.format(
            keyword1=selected_keywords[0],
            keyword2=selected_keywords[1],
            keyword3=selected_keywords[2]
        )
        
        return ResearchTopic(
            title=title,
            domain=domain,
            description=f"Advanced research in {title.lower()} with focus on breakthrough innovations",
            keywords=selected_keywords,
            novelty_score=0.7 + (index * 0.05),  # Simulate novelty scoring
            feasibility_score=0.8 - (index * 0.03),  # Simulate feasibility scoring
            impact_potential=0.75 + (index * 0.04),  # Simulate impact scoring
            research_gaps=[
                f"Limited understanding of {selected_keywords[0]} scalability",
                f"Lack of standardized {selected_keywords[1]} methodologies",
                f"Insufficient {selected_keywords[2]} validation frameworks"
            ]
        )


class LiteratureAnalyzer:
    """Analyzes literature and identifies knowledge gaps"""
    
    def __init__(self):
        self.analysis_cache = {}
    
    async def analyze_literature(self, topic: ResearchTopic) -> LiteratureAnalysis:
        """Perform comprehensive literature analysis"""
        try:
            # Check cache first
            if topic.id in self.analysis_cache:
                return self.analysis_cache[topic.id]
            
            # Simulate literature search and analysis
            sources = await self._search_literature(topic)
            gaps = await self._identify_knowledge_gaps(topic, sources)
            trends = await self._analyze_research_trends(sources)
            findings = await self._extract_key_findings(sources)
            
            analysis = LiteratureAnalysis(
                topic_id=topic.id,
                sources=sources,
                knowledge_gaps=gaps,
                research_trends=trends,
                key_findings=findings,
                methodological_gaps=await self._identify_methodological_gaps(sources),
                theoretical_gaps=await self._identify_theoretical_gaps(sources),
                empirical_gaps=await self._identify_empirical_gaps(sources),
                analysis_confidence=0.85  # Simulate confidence scoring
            )
            
            # Cache the analysis
            self.analysis_cache[topic.id] = analysis
            
            logger.info(f"Completed literature analysis for topic: {topic.title}")
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing literature: {str(e)}")
            return LiteratureAnalysis(topic_id=topic.id)
    
    async def _search_literature(self, topic: ResearchTopic) -> List[LiteratureSource]:
        """Search for relevant literature sources"""
        # Simulate literature database search
        sources = []
        for i, keyword in enumerate(topic.keywords[:5]):
            source = LiteratureSource(
                title=f"Advanced {keyword} research: A comprehensive review",
                authors=[f"Author {i+1}", f"Author {i+2}"],
                publication_year=2023 - i,
                journal=f"Journal of {topic.domain.value.replace('_', ' ').title()}",
                abstract=f"This paper presents novel approaches to {keyword} with applications in {topic.domain.value}",
                keywords=[keyword] + topic.keywords[:2],
                citation_count=100 - (i * 10),
                relevance_score=0.9 - (i * 0.1)
            )
            sources.append(source)
        
        return sources
    
    async def _identify_knowledge_gaps(self, topic: ResearchTopic, sources: List[LiteratureSource]) -> List[str]:
        """Identify knowledge gaps from literature analysis"""
        gaps = [
            f"Limited scalability studies in {topic.domain.value}",
            f"Insufficient real-world validation of {topic.keywords[0] if topic.keywords else 'methods'}",
            f"Lack of standardized evaluation metrics for {topic.title.lower()}",
            f"Missing cross-domain applications of {topic.keywords[1] if len(topic.keywords) > 1 else 'techniques'}",
            f"Inadequate theoretical foundations for {topic.keywords[2] if len(topic.keywords) > 2 else 'approaches'}"
        ]
        return gaps
    
    async def _analyze_research_trends(self, sources: List[LiteratureSource]) -> List[str]:
        """Analyze research trends from literature"""
        trends = [
            "Increasing focus on autonomous systems",
            "Growing emphasis on explainability and interpretability",
            "Rising interest in cross-domain applications",
            "Shift towards scalable and efficient methodologies",
            "Enhanced integration of theoretical and empirical approaches"
        ]
        return trends
    
    async def _extract_key_findings(self, sources: List[LiteratureSource]) -> List[str]:
        """Extract key findings from literature"""
        findings = [
            "Current methodologies show promise but lack scalability",
            "Theoretical frameworks need empirical validation",
            "Cross-domain applications demonstrate significant potential",
            "Standardization efforts are fragmented across research groups",
            "Integration challenges persist in real-world deployments"
        ]
        return findings
    
    async def _identify_methodological_gaps(self, sources: List[LiteratureSource]) -> List[str]:
        """Identify methodological gaps"""
        return [
            "Lack of standardized experimental protocols",
            "Insufficient statistical power in validation studies",
            "Limited reproducibility frameworks",
            "Inadequate control group methodologies"
        ]
    
    async def _identify_theoretical_gaps(self, sources: List[LiteratureSource]) -> List[str]:
        """Identify theoretical gaps"""
        return [
            "Missing unified theoretical framework",
            "Incomplete mathematical foundations",
            "Lack of formal verification methods",
            "Insufficient theoretical complexity analysis"
        ]
    
    async def _identify_empirical_gaps(self, sources: List[LiteratureSource]) -> List[str]:
        """Identify empirical gaps"""
        return [
            "Limited large-scale empirical studies",
            "Insufficient real-world validation data",
            "Lack of longitudinal performance studies",
            "Missing comparative empirical analysis"
        ]


class HypothesisFormer:
    """Forms testable research hypotheses automatically"""
    
    def __init__(self):
        self.hypothesis_templates = [
            "If {intervention} is applied to {system}, then {outcome} will improve by {metric}",
            "The implementation of {method} will result in {improvement} compared to {baseline}",
            "Systems utilizing {approach} will demonstrate {advantage} over traditional {alternative}",
            "The integration of {technology} will enhance {performance} in {domain} applications",
            "Optimized {algorithm} will achieve {target} performance in {scenario} conditions"
        ]
    
    async def form_hypotheses(self, literature_analysis: LiteratureAnalysis) -> List[Hypothesis]:
        """Generate testable hypotheses from literature analysis"""
        try:
            hypotheses = []
            
            # Generate hypotheses based on knowledge gaps
            for gap in literature_analysis.knowledge_gaps[:3]:
                hypothesis = await self._create_hypothesis_from_gap(gap, literature_analysis)
                hypotheses.append(hypothesis)
            
            # Generate hypotheses based on research trends
            for trend in literature_analysis.research_trends[:2]:
                hypothesis = await self._create_hypothesis_from_trend(trend, literature_analysis)
                hypotheses.append(hypothesis)
            
            # Evaluate and rank hypotheses
            for hypothesis in hypotheses:
                await self._evaluate_hypothesis(hypothesis)
            
            # Sort by testability and significance potential
            hypotheses.sort(key=lambda h: (h.testability_score + h.significance_potential) / 2, reverse=True)
            
            logger.info(f"Generated {len(hypotheses)} testable hypotheses")
            return hypotheses
            
        except Exception as e:
            logger.error(f"Error forming hypotheses: {str(e)}")
            return []
    
    async def _create_hypothesis_from_gap(self, gap: str, analysis: LiteratureAnalysis) -> Hypothesis:
        """Create hypothesis to address a knowledge gap"""
        # Extract key terms from gap
        gap_terms = gap.lower().split()
        key_terms = [term for term in gap_terms if len(term) > 4][:3]
        
        template = self.hypothesis_templates[0]
        statement = template.format(
            intervention=f"novel {key_terms[0] if key_terms else 'method'}",
            system=f"{key_terms[1] if len(key_terms) > 1 else 'system'} framework",
            outcome=f"{key_terms[2] if len(key_terms) > 2 else 'performance'}",
            metric="at least 20%"
        )
        
        return Hypothesis(
            topic_id=analysis.topic_id,
            statement=statement,
            null_hypothesis=f"No significant improvement will be observed",
            alternative_hypothesis=f"Significant improvement will be observed",
            variables={
                "independent": f"novel {key_terms[0] if key_terms else 'method'}",
                "dependent": f"{key_terms[2] if len(key_terms) > 2 else 'performance'} metrics",
                "control": "baseline system performance"
            },
            required_resources=["computational resources", "validation datasets", "evaluation metrics"]
        )
    
    async def _create_hypothesis_from_trend(self, trend: str, analysis: LiteratureAnalysis) -> Hypothesis:
        """Create hypothesis based on research trend"""
        trend_terms = trend.lower().split()
        key_terms = [term for term in trend_terms if len(term) > 4][:3]
        
        template = self.hypothesis_templates[1]
        statement = template.format(
            method=f"{key_terms[0] if key_terms else 'advanced'} methodology",
            improvement=f"enhanced {key_terms[1] if len(key_terms) > 1 else 'performance'}",
            baseline="current state-of-the-art approaches"
        )
        
        return Hypothesis(
            topic_id=analysis.topic_id,
            statement=statement,
            null_hypothesis="No significant difference from baseline",
            alternative_hypothesis="Significant improvement over baseline",
            variables={
                "independent": f"{key_terms[0] if key_terms else 'advanced'} methodology",
                "dependent": f"{key_terms[1] if len(key_terms) > 1 else 'performance'} measures",
                "control": "baseline methodology"
            },
            required_resources=["experimental setup", "comparison baselines", "statistical analysis tools"]
        )
    
    async def _evaluate_hypothesis(self, hypothesis: Hypothesis) -> None:
        """Evaluate hypothesis testability and significance"""
        # Simulate hypothesis evaluation
        hypothesis.testability_score = 0.8  # Based on variable clarity and measurability
        hypothesis.novelty_score = 0.75     # Based on uniqueness of approach
        hypothesis.significance_potential = 0.85  # Based on potential impact


class ResearchPlanner:
    """Develops systematic research plans and methodologies"""
    
    def __init__(self):
        self.methodology_templates = {
            "experimental": ResearchMethodology(
                name="Experimental Research",
                methodology_type="experimental",
                data_collection_methods=["controlled experiments", "randomized trials", "A/B testing"],
                analysis_methods=["statistical analysis", "hypothesis testing", "effect size calculation"],
                validation_approaches=["cross-validation", "replication studies", "peer review"]
            ),
            "computational": ResearchMethodology(
                name="Computational Research",
                methodology_type="computational",
                data_collection_methods=["simulation", "modeling", "algorithm implementation"],
                analysis_methods=["performance analysis", "complexity analysis", "benchmarking"],
                validation_approaches=["theoretical proof", "empirical validation", "comparative analysis"]
            )
        }
    
    async def create_research_plan(self, hypothesis: Hypothesis, literature_analysis: LiteratureAnalysis) -> ResearchPlan:
        """Create systematic research plan"""
        try:
            # Select appropriate methodology
            methodology = await self._select_methodology(hypothesis, literature_analysis)
            
            # Create timeline and milestones
            timeline = await self._create_timeline(hypothesis)
            milestones = await self._define_milestones(hypothesis)
            
            # Assess resource requirements
            resources = await self._assess_resource_requirements(hypothesis, methodology)
            
            # Define success criteria
            success_criteria = await self._define_success_criteria(hypothesis)
            
            # Perform risk assessment
            risks = await self._assess_risks(hypothesis, methodology)
            
            plan = ResearchPlan(
                topic_id=hypothesis.topic_id,
                hypothesis_id=hypothesis.id,
                title=f"Research Plan: {hypothesis.statement[:50]}...",
                objectives=await self._define_objectives(hypothesis),
                methodology=methodology,
                timeline=timeline,
                milestones=milestones,
                resource_requirements=resources,
                success_criteria=success_criteria,
                risk_assessment=risks
            )
            
            logger.info(f"Created research plan for hypothesis: {hypothesis.id}")
            return plan
            
        except Exception as e:
            logger.error(f"Error creating research plan: {str(e)}")
            return ResearchPlan(topic_id=hypothesis.topic_id, hypothesis_id=hypothesis.id)
    
    async def _select_methodology(self, hypothesis: Hypothesis, analysis: LiteratureAnalysis) -> ResearchMethodology:
        """Select appropriate research methodology"""
        # Simple methodology selection logic
        if "algorithm" in hypothesis.statement.lower() or "computational" in hypothesis.statement.lower():
            return self.methodology_templates["computational"]
        else:
            return self.methodology_templates["experimental"]
    
    async def _create_timeline(self, hypothesis: Hypothesis) -> Dict[str, datetime]:
        """Create research timeline"""
        start_date = datetime.now()
        return {
            "project_start": start_date,
            "literature_review": start_date + timedelta(weeks=2),
            "methodology_design": start_date + timedelta(weeks=4),
            "data_collection": start_date + timedelta(weeks=8),
            "analysis": start_date + timedelta(weeks=12),
            "validation": start_date + timedelta(weeks=14),
            "reporting": start_date + timedelta(weeks=16),
            "project_end": start_date + timedelta(weeks=18)
        }
    
    async def _define_milestones(self, hypothesis: Hypothesis) -> List[str]:
        """Define research milestones"""
        return [
            "Complete comprehensive literature review",
            "Finalize research methodology and experimental design",
            "Complete data collection and preprocessing",
            "Conduct primary analysis and hypothesis testing",
            "Perform validation and sensitivity analysis",
            "Prepare research findings and documentation",
            "Submit for peer review and publication"
        ]
    
    async def _assess_resource_requirements(self, hypothesis: Hypothesis, methodology: ResearchMethodology) -> Dict[str, Any]:
        """Assess resource requirements"""
        return {
            "computational": {
                "cpu_hours": 1000,
                "memory_gb": 64,
                "storage_tb": 1,
                "gpu_hours": 500 if "learning" in hypothesis.statement.lower() else 0
            },
            "human": {
                "researcher_hours": 400,
                "domain_expert_hours": 40,
                "statistical_consultant_hours": 20
            },
            "financial": {
                "equipment_cost": 5000,
                "software_licenses": 2000,
                "publication_fees": 1500,
                "total_budget": 15000
            },
            "data": {
                "datasets_required": hypothesis.required_resources,
                "data_size_gb": 100,
                "annotation_hours": 50
            }
        }
    
    async def _define_success_criteria(self, hypothesis: Hypothesis) -> List[str]:
        """Define success criteria"""
        return [
            f"Achieve statistical significance (p < 0.05) in hypothesis testing",
            f"Demonstrate practical significance with effect size > 0.5",
            f"Validate results through independent replication",
            f"Achieve performance improvement of at least 15%",
            f"Complete all milestones within planned timeline",
            f"Publish findings in peer-reviewed venue"
        ]
    
    async def _assess_risks(self, hypothesis: Hypothesis, methodology: ResearchMethodology) -> Dict[str, float]:
        """Assess research risks"""
        return {
            "technical_feasibility": 0.2,  # Low risk
            "resource_availability": 0.3,  # Medium risk
            "timeline_adherence": 0.4,     # Medium risk
            "result_significance": 0.3,    # Medium risk
            "reproducibility": 0.2,        # Low risk
            "ethical_approval": 0.1        # Very low risk
        }
    
    async def _define_objectives(self, hypothesis: Hypothesis) -> List[str]:
        """Define research objectives"""
        return [
            f"Test the validity of the proposed hypothesis: {hypothesis.statement}",
            f"Develop and validate novel methodological approaches",
            f"Contribute to theoretical understanding of the research domain",
            f"Provide empirical evidence for practical applications",
            f"Establish benchmarks for future research comparisons"
        ]


class AutomatedResearchEngine:
    """Main automated research engine coordinating all components"""
    
    def __init__(self):
        self.topic_generator = TopicGenerator()
        self.literature_analyzer = LiteratureAnalyzer()
        self.hypothesis_former = HypothesisFormer()
        self.research_planner = ResearchPlanner()
        self.active_projects = {}
    
    async def generate_research_topics(self, domain: ResearchDomain, count: int = 5) -> List[ResearchTopic]:
        """Generate promising research topics for a domain"""
        return await self.topic_generator.generate_topics(domain, count)
    
    async def analyze_literature(self, topic: ResearchTopic) -> LiteratureAnalysis:
        """Perform comprehensive literature analysis"""
        return await self.literature_analyzer.analyze_literature(topic)
    
    async def form_hypotheses(self, literature_analysis: LiteratureAnalysis) -> List[Hypothesis]:
        """Generate testable research hypotheses"""
        return await self.hypothesis_former.form_hypotheses(literature_analysis)
    
    async def create_research_plan(self, hypothesis: Hypothesis, literature_analysis: LiteratureAnalysis) -> ResearchPlan:
        """Create systematic research plan"""
        return await self.research_planner.create_research_plan(hypothesis, literature_analysis)
    
    async def conduct_autonomous_research(self, domain: ResearchDomain, topic_count: int = 3) -> Dict[str, Any]:
        """Conduct complete autonomous research process"""
        try:
            logger.info(f"Starting autonomous research for domain: {domain.value}")
            
            # Step 1: Generate research topics
            topics = await self.generate_research_topics(domain, topic_count)
            if not topics:
                raise Exception("Failed to generate research topics")
            
            research_results = {
                "domain": domain.value,
                "topics": [],
                "total_hypotheses": 0,
                "total_plans": 0,
                "started_at": datetime.now().isoformat()
            }
            
            # Step 2: Process each topic
            for topic in topics:
                topic_result = {
                    "topic": topic,
                    "literature_analysis": None,
                    "hypotheses": [],
                    "research_plans": []
                }
                
                # Step 3: Analyze literature
                literature_analysis = await self.analyze_literature(topic)
                topic_result["literature_analysis"] = literature_analysis
                
                # Step 4: Form hypotheses
                hypotheses = await self.form_hypotheses(literature_analysis)
                topic_result["hypotheses"] = hypotheses
                research_results["total_hypotheses"] += len(hypotheses)
                
                # Step 5: Create research plans
                for hypothesis in hypotheses[:2]:  # Limit to top 2 hypotheses per topic
                    research_plan = await self.create_research_plan(hypothesis, literature_analysis)
                    topic_result["research_plans"].append(research_plan)
                    research_results["total_plans"] += 1
                
                research_results["topics"].append(topic_result)
            
            research_results["completed_at"] = datetime.now().isoformat()
            
            logger.info(f"Completed autonomous research: {research_results['total_hypotheses']} hypotheses, {research_results['total_plans']} plans")
            return research_results
            
        except Exception as e:
            logger.error(f"Error in autonomous research: {str(e)}")
            return {"error": str(e), "domain": domain.value}
    
    async def get_research_status(self, project_id: str) -> Dict[str, Any]:
        """Get status of active research project"""
        if project_id in self.active_projects:
            return self.active_projects[project_id]
        return {"error": "Project not found"}
    
    async def list_active_projects(self) -> List[Dict[str, Any]]:
        """List all active research projects"""
        return list(self.active_projects.values())