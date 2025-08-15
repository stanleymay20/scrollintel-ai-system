"""
API routes for Automated Research Engine

Provides endpoints for:
- Research topic generation
- Literature analysis
- Hypothesis formation
- Research planning
- Autonomous research execution
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import logging

from scrollintel.engines.automated_research_engine import (
    AutomatedResearchEngine,
    ResearchDomain,
    ResearchTopic,
    LiteratureAnalysis,
    Hypothesis,
    ResearchPlan
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/research", tags=["automated-research"])

# Initialize the research engine
research_engine = AutomatedResearchEngine()


# Request/Response Models
class TopicGenerationRequest(BaseModel):
    domain: str = Field(..., description="Research domain")
    count: int = Field(default=5, ge=1, le=20, description="Number of topics to generate")


class LiteratureAnalysisRequest(BaseModel):
    topic_id: str = Field(..., description="Research topic ID")


class HypothesisFormationRequest(BaseModel):
    topic_id: str = Field(..., description="Research topic ID")


class ResearchPlanRequest(BaseModel):
    hypothesis_id: str = Field(..., description="Hypothesis ID")
    topic_id: str = Field(..., description="Research topic ID")


class AutonomousResearchRequest(BaseModel):
    domain: str = Field(..., description="Research domain")
    topic_count: int = Field(default=3, ge=1, le=10, description="Number of topics to research")


class ResearchTopicResponse(BaseModel):
    id: str
    title: str
    domain: str
    description: str
    keywords: List[str]
    novelty_score: float
    feasibility_score: float
    impact_potential: float
    research_gaps: List[str]
    created_at: str


class LiteratureAnalysisResponse(BaseModel):
    topic_id: str
    knowledge_gaps: List[str]
    research_trends: List[str]
    key_findings: List[str]
    methodological_gaps: List[str]
    theoretical_gaps: List[str]
    empirical_gaps: List[str]
    analysis_confidence: float
    source_count: int
    created_at: str


class HypothesisResponse(BaseModel):
    id: str
    topic_id: str
    statement: str
    null_hypothesis: str
    alternative_hypothesis: str
    variables: Dict[str, str]
    testability_score: float
    novelty_score: float
    significance_potential: float
    required_resources: List[str]
    expected_timeline_days: int
    created_at: str


class ResearchPlanResponse(BaseModel):
    id: str
    topic_id: str
    hypothesis_id: str
    title: str
    objectives: List[str]
    methodology_type: str
    timeline: Dict[str, str]
    milestones: List[str]
    resource_requirements: Dict[str, Any]
    success_criteria: List[str]
    risk_assessment: Dict[str, float]
    created_at: str


@router.post("/topics/generate", response_model=List[ResearchTopicResponse])
async def generate_research_topics(request: TopicGenerationRequest):
    """Generate promising research topics for a domain"""
    try:
        # Validate domain
        try:
            domain = ResearchDomain(request.domain.lower())
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid domain. Valid domains: {[d.value for d in ResearchDomain]}"
            )
        
        # Generate topics
        topics = await research_engine.generate_research_topics(domain, request.count)
        
        # Convert to response format
        response_topics = []
        for topic in topics:
            response_topics.append(ResearchTopicResponse(
                id=topic.id,
                title=topic.title,
                domain=topic.domain.value,
                description=topic.description,
                keywords=topic.keywords,
                novelty_score=topic.novelty_score,
                feasibility_score=topic.feasibility_score,
                impact_potential=topic.impact_potential,
                research_gaps=topic.research_gaps,
                created_at=topic.created_at.isoformat()
            ))
        
        logger.info(f"Generated {len(response_topics)} research topics for domain {request.domain}")
        return response_topics
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating research topics: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to generate research topics")


@router.post("/literature/analyze", response_model=LiteratureAnalysisResponse)
async def analyze_literature(request: LiteratureAnalysisRequest):
    """Perform comprehensive literature analysis for a research topic"""
    try:
        # Find the topic (in a real implementation, this would query a database)
        # For now, we'll create a mock topic
        topic = ResearchTopic(
            id=request.topic_id,
            title="Mock Research Topic",
            domain=ResearchDomain.ARTIFICIAL_INTELLIGENCE,
            description="Mock topic for literature analysis",
            keywords=["artificial intelligence", "machine learning", "research"]
        )
        
        # Perform literature analysis
        analysis = await research_engine.analyze_literature(topic)
        
        # Convert to response format
        response = LiteratureAnalysisResponse(
            topic_id=analysis.topic_id,
            knowledge_gaps=analysis.knowledge_gaps,
            research_trends=analysis.research_trends,
            key_findings=analysis.key_findings,
            methodological_gaps=analysis.methodological_gaps,
            theoretical_gaps=analysis.theoretical_gaps,
            empirical_gaps=analysis.empirical_gaps,
            analysis_confidence=analysis.analysis_confidence,
            source_count=len(analysis.sources),
            created_at=analysis.created_at.isoformat()
        )
        
        logger.info(f"Completed literature analysis for topic {request.topic_id}")
        return response
        
    except Exception as e:
        logger.error(f"Error analyzing literature: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to analyze literature")


@router.post("/hypotheses/generate", response_model=List[HypothesisResponse])
async def generate_hypotheses(request: HypothesisFormationRequest):
    """Generate testable research hypotheses from literature analysis"""
    try:
        # Create mock literature analysis (in real implementation, retrieve from database)
        analysis = LiteratureAnalysis(
            topic_id=request.topic_id,
            knowledge_gaps=[
                "Limited scalability studies in artificial intelligence",
                "Insufficient real-world validation of machine learning methods",
                "Lack of standardized evaluation metrics"
            ],
            research_trends=[
                "Increasing focus on autonomous systems",
                "Growing emphasis on explainability"
            ],
            key_findings=[
                "Current methodologies show promise but lack scalability",
                "Theoretical frameworks need empirical validation"
            ]
        )
        
        # Generate hypotheses
        hypotheses = await research_engine.form_hypotheses(analysis)
        
        # Convert to response format
        response_hypotheses = []
        for hypothesis in hypotheses:
            response_hypotheses.append(HypothesisResponse(
                id=hypothesis.id,
                topic_id=hypothesis.topic_id,
                statement=hypothesis.statement,
                null_hypothesis=hypothesis.null_hypothesis,
                alternative_hypothesis=hypothesis.alternative_hypothesis,
                variables=hypothesis.variables,
                testability_score=hypothesis.testability_score,
                novelty_score=hypothesis.novelty_score,
                significance_potential=hypothesis.significance_potential,
                required_resources=hypothesis.required_resources,
                expected_timeline_days=hypothesis.expected_timeline.days,
                created_at=hypothesis.created_at.isoformat()
            ))
        
        logger.info(f"Generated {len(response_hypotheses)} hypotheses for topic {request.topic_id}")
        return response_hypotheses
        
    except Exception as e:
        logger.error(f"Error generating hypotheses: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to generate hypotheses")


@router.post("/plans/create", response_model=ResearchPlanResponse)
async def create_research_plan(request: ResearchPlanRequest):
    """Create systematic research plan for a hypothesis"""
    try:
        # Create mock hypothesis and analysis (in real implementation, retrieve from database)
        hypothesis = Hypothesis(
            id=request.hypothesis_id,
            topic_id=request.topic_id,
            statement="Mock hypothesis for research planning",
            null_hypothesis="No significant effect",
            alternative_hypothesis="Significant effect exists"
        )
        
        analysis = LiteratureAnalysis(topic_id=request.topic_id)
        
        # Create research plan
        plan = await research_engine.create_research_plan(hypothesis, analysis)
        
        # Convert timeline to string format
        timeline_str = {k: v.isoformat() for k, v in plan.timeline.items()}
        
        # Convert to response format
        response = ResearchPlanResponse(
            id=plan.id,
            topic_id=plan.topic_id,
            hypothesis_id=plan.hypothesis_id,
            title=plan.title,
            objectives=plan.objectives,
            methodology_type=plan.methodology.methodology_type,
            timeline=timeline_str,
            milestones=plan.milestones,
            resource_requirements=plan.resource_requirements,
            success_criteria=plan.success_criteria,
            risk_assessment=plan.risk_assessment,
            created_at=plan.created_at.isoformat()
        )
        
        logger.info(f"Created research plan for hypothesis {request.hypothesis_id}")
        return response
        
    except Exception as e:
        logger.error(f"Error creating research plan: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to create research plan")


@router.post("/autonomous/start")
async def start_autonomous_research(request: AutonomousResearchRequest, background_tasks: BackgroundTasks):
    """Start autonomous research process for a domain"""
    try:
        # Validate domain
        try:
            domain = ResearchDomain(request.domain.lower())
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid domain. Valid domains: {[d.value for d in ResearchDomain]}"
            )
        
        # Start autonomous research in background
        background_tasks.add_task(
            research_engine.conduct_autonomous_research,
            domain,
            request.topic_count
        )
        
        return {
            "message": "Autonomous research started",
            "domain": request.domain,
            "topic_count": request.topic_count,
            "status": "initiated"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting autonomous research: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to start autonomous research")


@router.get("/autonomous/status/{project_id}")
async def get_research_status(project_id: str):
    """Get status of autonomous research project"""
    try:
        status = await research_engine.get_research_status(project_id)
        return status
        
    except Exception as e:
        logger.error(f"Error getting research status: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get research status")


@router.get("/autonomous/projects")
async def list_active_projects():
    """List all active research projects"""
    try:
        projects = await research_engine.list_active_projects()
        return {"projects": projects, "count": len(projects)}
        
    except Exception as e:
        logger.error(f"Error listing projects: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to list projects")


@router.get("/domains")
async def get_research_domains():
    """Get available research domains"""
    try:
        domains = [{"value": domain.value, "name": domain.value.replace("_", " ").title()} 
                  for domain in ResearchDomain]
        return {"domains": domains}
        
    except Exception as e:
        logger.error(f"Error getting domains: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get domains")


@router.get("/health")
async def health_check():
    """Health check endpoint for automated research engine"""
    try:
        # Test basic functionality
        test_domain = ResearchDomain.ARTIFICIAL_INTELLIGENCE
        topics = await research_engine.generate_research_topics(test_domain, 1)
        
        return {
            "status": "healthy",
            "engine": "automated_research",
            "components": {
                "topic_generator": "operational",
                "literature_analyzer": "operational", 
                "hypothesis_former": "operational",
                "research_planner": "operational"
            },
            "test_result": f"Generated {len(topics)} test topics"
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }