"""
API Routes for Fundamental Research Engine

This module provides REST API endpoints for the fundamental research capabilities
including hypothesis generation, experiment design, breakthrough detection, and
research paper generation.
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import List, Dict, Any, Optional
import logging

from ...engines.fundamental_research_engine import FundamentalResearchEngine, ResearchContext
from ...models.fundamental_research_models import (
    ResearchDomain, ResearchMethodology, HypothesisStatus,
    HypothesisCreate, HypothesisResponse, ExperimentDesign, ExperimentResults,
    ResearchBreakthroughCreate, ResearchBreakthroughResponse, ResearchPaper,
    ResearchInsight
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/fundamental-research", tags=["fundamental-research"])

# Global engine instance
research_engine = FundamentalResearchEngine()

@router.post("/hypotheses/generate", response_model=List[HypothesisResponse])
async def generate_research_hypotheses(
    domain: ResearchDomain,
    existing_knowledge: List[str] = [],
    research_gaps: List[str] = [],
    available_resources: Dict[str, Any] = {},
    constraints: List[str] = [],
    num_hypotheses: int = 5
):
    """
    Generate novel research hypotheses using AI-assisted analysis
    
    Args:
        domain: Research domain for hypothesis generation
        existing_knowledge: List of existing knowledge in the domain
        research_gaps: Identified gaps in current research
        available_resources: Available resources for research
        constraints: Research constraints and limitations
        num_hypotheses: Number of hypotheses to generate
        
    Returns:
        List of generated research hypotheses with quality scores
    """
    try:
        logger.info(f"Generating {num_hypotheses} hypotheses for domain: {domain}")
        
        context = ResearchContext(
            domain=domain,
            existing_knowledge=existing_knowledge,
            research_gaps=research_gaps,
            available_resources=available_resources,
            constraints=constraints
        )
        
        hypotheses = await research_engine.generate_research_hypotheses(
            context=context,
            num_hypotheses=num_hypotheses
        )
        
        logger.info(f"Successfully generated {len(hypotheses)} hypotheses")
        return hypotheses
        
    except Exception as e:
        logger.error(f"Error generating hypotheses: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate hypotheses: {str(e)}")

@router.get("/hypotheses/{hypothesis_id}", response_model=HypothesisResponse)
async def get_hypothesis(hypothesis_id: str):
    """Get a specific research hypothesis by ID"""
    try:
        hypothesis = research_engine.hypothesis_database.get(hypothesis_id)
        if not hypothesis:
            raise HTTPException(status_code=404, detail=f"Hypothesis {hypothesis_id} not found")
        
        return hypothesis
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving hypothesis {hypothesis_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve hypothesis: {str(e)}")

@router.get("/hypotheses", response_model=List[HypothesisResponse])
async def list_hypotheses(
    domain: Optional[ResearchDomain] = None,
    status: Optional[HypothesisStatus] = None,
    min_novelty: Optional[float] = None,
    min_impact: Optional[float] = None,
    limit: int = 50
):
    """List research hypotheses with optional filtering"""
    try:
        hypotheses = list(research_engine.hypothesis_database.values())
        
        # Apply filters
        if domain:
            hypotheses = [h for h in hypotheses if h.domain == domain]
        
        if status:
            hypotheses = [h for h in hypotheses if h.status == status]
        
        if min_novelty is not None:
            hypotheses = [h for h in hypotheses if h.novelty_score >= min_novelty]
        
        if min_impact is not None:
            hypotheses = [h for h in hypotheses if h.impact_potential >= min_impact]
        
        # Sort by combined score and limit results
        hypotheses.sort(
            key=lambda h: h.novelty_score * h.impact_potential * h.feasibility_score,
            reverse=True
        )
        
        return hypotheses[:limit]
        
    except Exception as e:
        logger.error(f"Error listing hypotheses: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list hypotheses: {str(e)}")

@router.post("/experiments/design", response_model=ExperimentDesign)
async def design_experiment(hypothesis_id: str):
    """
    Design breakthrough experiments for a given hypothesis
    
    Args:
        hypothesis_id: ID of the hypothesis to design experiments for
        
    Returns:
        Detailed experimental design with methodology and protocols
    """
    try:
        logger.info(f"Designing experiment for hypothesis: {hypothesis_id}")
        
        experiment_design = await research_engine.design_experiments(hypothesis_id)
        
        logger.info(f"Successfully designed experiment for hypothesis: {hypothesis_id}")
        return experiment_design
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error designing experiment: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to design experiment: {str(e)}")

@router.post("/results/analyze", response_model=Dict[str, Any])
async def analyze_research_results(experiment_results: ExperimentResults):
    """
    Analyze research results and detect potential breakthroughs
    
    Args:
        experiment_results: Results from conducted experiments
        
    Returns:
        Analysis results including insights and breakthrough detection
    """
    try:
        logger.info(f"Analyzing results for experiment: {experiment_results.experiment_id}")
        
        insights, is_breakthrough = await research_engine.analyze_research_results(experiment_results)
        
        response = {
            "experiment_id": experiment_results.experiment_id,
            "insights": [insight.dict() for insight in insights],
            "is_breakthrough": is_breakthrough,
            "confidence_level": experiment_results.confidence_level,
            "analysis_summary": {
                "total_insights": len(insights),
                "breakthrough_detected": is_breakthrough,
                "high_significance_insights": len([i for i in insights if i.significance > 0.8])
            }
        }
        
        logger.info(f"Analysis complete. Breakthrough detected: {is_breakthrough}")
        return response
        
    except Exception as e:
        logger.error(f"Error analyzing results: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to analyze results: {str(e)}")

@router.post("/breakthroughs", response_model=ResearchBreakthroughResponse)
async def create_research_breakthrough(breakthrough_data: ResearchBreakthroughCreate):
    """
    Create a new research breakthrough record
    
    Args:
        breakthrough_data: Data for the research breakthrough
        
    Returns:
        Created research breakthrough with assigned ID
    """
    try:
        logger.info(f"Creating research breakthrough: {breakthrough_data.title}")
        
        breakthrough = await research_engine.create_research_breakthrough(breakthrough_data)
        
        logger.info(f"Successfully created breakthrough: {breakthrough.id}")
        return breakthrough
        
    except Exception as e:
        logger.error(f"Error creating breakthrough: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create breakthrough: {str(e)}")

@router.get("/breakthroughs/{breakthrough_id}", response_model=ResearchBreakthroughResponse)
async def get_research_breakthrough(breakthrough_id: str):
    """Get a specific research breakthrough by ID"""
    try:
        breakthrough = research_engine.breakthrough_database.get(breakthrough_id)
        if not breakthrough:
            raise HTTPException(status_code=404, detail=f"Breakthrough {breakthrough_id} not found")
        
        return breakthrough
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving breakthrough {breakthrough_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve breakthrough: {str(e)}")

@router.get("/breakthroughs", response_model=List[ResearchBreakthroughResponse])
async def list_research_breakthroughs(
    domain: Optional[ResearchDomain] = None,
    min_novelty: Optional[float] = None,
    min_impact: Optional[float] = None,
    limit: int = 50
):
    """List research breakthroughs with optional filtering"""
    try:
        breakthroughs = list(research_engine.breakthrough_database.values())
        
        # Apply filters
        if domain:
            breakthroughs = [b for b in breakthroughs if b.domain == domain]
        
        if min_novelty is not None:
            breakthroughs = [b for b in breakthroughs if b.novelty_assessment >= min_novelty]
        
        if min_impact is not None:
            breakthroughs = [b for b in breakthroughs if b.impact_assessment >= min_impact]
        
        # Sort by combined score and limit results
        breakthroughs.sort(
            key=lambda b: b.novelty_assessment * b.impact_assessment,
            reverse=True
        )
        
        return breakthroughs[:limit]
        
    except Exception as e:
        logger.error(f"Error listing breakthroughs: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list breakthroughs: {str(e)}")

@router.post("/papers/generate", response_model=ResearchPaper)
async def generate_research_paper(breakthrough_id: str, background_tasks: BackgroundTasks):
    """
    Generate publication-quality research paper with novel contributions
    
    Args:
        breakthrough_id: ID of the research breakthrough
        background_tasks: Background task handler for async processing
        
    Returns:
        Generated research paper with all sections
    """
    try:
        logger.info(f"Generating research paper for breakthrough: {breakthrough_id}")
        
        # Verify breakthrough exists
        breakthrough = research_engine.breakthrough_database.get(breakthrough_id)
        if not breakthrough:
            raise HTTPException(status_code=404, detail=f"Breakthrough {breakthrough_id} not found")
        
        paper = await research_engine.generate_research_paper(breakthrough_id)
        
        logger.info(f"Successfully generated paper: {paper.title}")
        return paper
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating paper: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate paper: {str(e)}")

@router.get("/quality-metrics/{breakthrough_id}", response_model=Dict[str, float])
async def get_research_quality_metrics(breakthrough_id: str):
    """
    Get quality metrics for a research breakthrough
    
    Args:
        breakthrough_id: ID of the research breakthrough
        
    Returns:
        Quality metrics including novelty, impact, and reproducibility scores
    """
    try:
        metrics = await research_engine.get_research_quality_metrics(breakthrough_id)
        return metrics
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting quality metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get quality metrics: {str(e)}")

@router.get("/domains", response_model=List[str])
async def get_research_domains():
    """Get list of available research domains"""
    return [domain.value for domain in ResearchDomain]

@router.get("/methodologies", response_model=List[str])
async def get_research_methodologies():
    """Get list of available research methodologies"""
    return [methodology.value for methodology in ResearchMethodology]

@router.get("/stats", response_model=Dict[str, Any])
async def get_research_stats():
    """Get overall research statistics"""
    try:
        hypotheses = list(research_engine.hypothesis_database.values())
        breakthroughs = list(research_engine.breakthrough_database.values())
        
        stats = {
            "total_hypotheses": len(hypotheses),
            "total_breakthroughs": len(breakthroughs),
            "domains_covered": len(set(h.domain for h in hypotheses)),
            "average_novelty": sum(h.novelty_score for h in hypotheses) / len(hypotheses) if hypotheses else 0,
            "average_impact": sum(h.impact_potential for h in hypotheses) / len(hypotheses) if hypotheses else 0,
            "breakthrough_rate": len(breakthroughs) / len(hypotheses) if hypotheses else 0,
            "high_quality_breakthroughs": len([b for b in breakthroughs if b.novelty_assessment > 0.8])
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"Error getting research stats: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get research stats: {str(e)}")

@router.post("/validate-hypothesis", response_model=Dict[str, Any])
async def validate_hypothesis_quality(hypothesis_data: HypothesisCreate):
    """
    Validate the quality of a hypothesis before creation
    
    Args:
        hypothesis_data: Hypothesis data to validate
        
    Returns:
        Validation results with quality scores and recommendations
    """
    try:
        # Create temporary context for validation
        context = ResearchContext(
            domain=hypothesis_data.domain,
            existing_knowledge=[],
            research_gaps=[],
            available_resources={},
            constraints=[]
        )
        
        # Simulate validation (in real implementation, this would use ML models)
        validation_results = {
            "is_valid": True,
            "quality_score": (hypothesis_data.novelty_score + hypothesis_data.feasibility_score + hypothesis_data.impact_potential) / 3,
            "novelty_assessment": hypothesis_data.novelty_score,
            "feasibility_assessment": hypothesis_data.feasibility_score,
            "impact_assessment": hypothesis_data.impact_potential,
            "recommendations": [
                "Consider strengthening theoretical foundation",
                "Add more specific testable predictions",
                "Clarify experimental methodology"
            ],
            "domain_appropriateness": 0.9,
            "testability_score": 0.8
        }
        
        return validation_results
        
    except Exception as e:
        logger.error(f"Error validating hypothesis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to validate hypothesis: {str(e)}")

@router.post("/research-pipeline", response_model=Dict[str, Any])
async def run_research_pipeline(
    domain: ResearchDomain,
    research_goals: List[str],
    available_resources: Dict[str, Any] = {},
    background_tasks: BackgroundTasks = None
):
    """
    Run complete research pipeline from hypothesis generation to paper creation
    
    Args:
        domain: Research domain
        research_goals: List of research goals and objectives
        available_resources: Available resources for research
        background_tasks: Background task handler
        
    Returns:
        Pipeline execution results with generated artifacts
    """
    try:
        logger.info(f"Running research pipeline for domain: {domain}")
        
        # Create research context
        context = ResearchContext(
            domain=domain,
            existing_knowledge=[],
            research_gaps=research_goals,
            available_resources=available_resources,
            constraints=[]
        )
        
        # Generate hypotheses
        hypotheses = await research_engine.generate_research_hypotheses(context, num_hypotheses=3)
        
        # Select best hypothesis
        best_hypothesis = max(hypotheses, key=lambda h: h.novelty_score * h.impact_potential)
        
        # Design experiment
        experiment_design = await research_engine.design_experiments(best_hypothesis.id)
        
        pipeline_results = {
            "pipeline_id": str(uuid.uuid4()),
            "domain": domain.value,
            "generated_hypotheses": len(hypotheses),
            "selected_hypothesis": {
                "id": best_hypothesis.id,
                "title": best_hypothesis.title,
                "quality_score": best_hypothesis.novelty_score * best_hypothesis.impact_potential
            },
            "experiment_design": {
                "methodology": experiment_design.methodology.value,
                "timeline": experiment_design.timeline,
                "success_criteria": experiment_design.success_criteria
            },
            "next_steps": [
                "Execute experimental design",
                "Collect and analyze results",
                "Detect breakthrough patterns",
                "Generate research paper"
            ],
            "estimated_completion": "6-12 months",
            "resource_requirements": experiment_design.resources_required
        }
        
        logger.info(f"Research pipeline completed for domain: {domain}")
        return pipeline_results
        
    except Exception as e:
        logger.error(f"Error running research pipeline: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to run research pipeline: {str(e)}")

# Import uuid for pipeline ID generation
import uuid