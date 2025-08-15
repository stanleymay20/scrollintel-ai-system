"""
API routes for Demand Creation System
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging
from pydantic import BaseModel

from scrollintel.engines.demand_creation_engine import (
    DemandCreationEngine,
    ProofOfConceptType,
    ThoughtLeadershipType,
    IndustryVertical,
    DemandStage,
    ProofOfConcept,
    ThoughtLeadershipPiece,
    IndustryStandard,
    DemandMetrics
)

logger = logging.getLogger(__name__)

# Pydantic models for request/response
class POCDeploymentRequest(BaseModel):
    enterprise_name: str
    industry: IndustryVertical
    poc_type: ProofOfConceptType
    objectives: List[str]
    timeline_weeks: int = 12

class POCPhaseRequest(BaseModel):
    poc_id: str
    phase: str

class ThoughtLeadershipRequest(BaseModel):
    title: str
    content_type: ThoughtLeadershipType
    target_audience: List[str]
    key_messages: List[str]
    author_credentials: str

class EngagementTrackingRequest(BaseModel):
    content_id: str
    engagement_data: Dict[str, int]

class IndustryStandardRequest(BaseModel):
    name: str
    description: str
    industry: IndustryVertical
    standard_type: str
    stakeholders: List[str]

class StandardAdvancementRequest(BaseModel):
    standard_id: str
    new_stage: str

# Initialize router and engine
router = APIRouter(prefix="/api/demand-creation", tags=["demand-creation"])
demand_engine = DemandCreationEngine()

@router.post("/poc/deploy", response_model=Dict[str, Any])
async def deploy_proof_of_concept(request: POCDeploymentRequest):
    """Deploy a proof-of-concept at target enterprise"""
    try:
        poc = await demand_engine.deploy_proof_of_concept(
            enterprise_name=request.enterprise_name,
            industry=request.industry,
            poc_type=request.poc_type,
            objectives=request.objectives,
            timeline_weeks=request.timeline_weeks
        )
        
        return {
            "success": True,
            "poc_id": poc.id,
            "poc_name": poc.name,
            "investment_required": poc.investment_required,
            "expected_roi": poc.expected_roi,
            "status": poc.status,
            "timeline_weeks": poc.timeline_weeks
        }
        
    except Exception as e:
        logger.error(f"Error deploying POC: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/poc/execute-phase", response_model=Dict[str, Any])
async def execute_poc_phase(request: POCPhaseRequest):
    """Execute specific phase of proof-of-concept"""
    try:
        phase_results = await demand_engine.execute_poc_phase(
            poc_id=request.poc_id,
            phase=request.phase
        )
        
        return {
            "success": True,
            "poc_id": request.poc_id,
            "phase": request.phase,
            "results": phase_results
        }
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error executing POC phase: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/poc/{poc_id}", response_model=Dict[str, Any])
async def get_poc_details(poc_id: str):
    """Get detailed information about a specific POC"""
    try:
        if poc_id not in demand_engine.proof_of_concepts:
            raise HTTPException(status_code=404, detail=f"POC not found: {poc_id}")
        
        poc = demand_engine.proof_of_concepts[poc_id]
        
        return {
            "id": poc.id,
            "name": poc.name,
            "poc_type": poc.poc_type.value,
            "target_enterprise": poc.target_enterprise,
            "industry_vertical": poc.industry_vertical.value,
            "objectives": poc.objectives,
            "success_metrics": poc.success_metrics,
            "timeline_weeks": poc.timeline_weeks,
            "investment_required": poc.investment_required,
            "expected_roi": poc.expected_roi,
            "status": poc.status,
            "results": poc.results,
            "lessons_learned": poc.lessons_learned,
            "created_at": poc.created_at.isoformat(),
            "completed_at": poc.completed_at.isoformat() if poc.completed_at else None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting POC details: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/poc", response_model=Dict[str, Any])
async def list_pocs(
    status: Optional[str] = None,
    industry: Optional[IndustryVertical] = None,
    poc_type: Optional[ProofOfConceptType] = None
):
    """List all POCs with optional filtering"""
    try:
        pocs = list(demand_engine.proof_of_concepts.values())
        
        # Apply filters
        if status:
            pocs = [poc for poc in pocs if poc.status == status]
        if industry:
            pocs = [poc for poc in pocs if poc.industry_vertical == industry]
        if poc_type:
            pocs = [poc for poc in pocs if poc.poc_type == poc_type]
        
        poc_summaries = []
        for poc in pocs:
            poc_summaries.append({
                "id": poc.id,
                "name": poc.name,
                "poc_type": poc.poc_type.value,
                "target_enterprise": poc.target_enterprise,
                "industry_vertical": poc.industry_vertical.value,
                "status": poc.status,
                "investment_required": poc.investment_required,
                "expected_roi": poc.expected_roi,
                "created_at": poc.created_at.isoformat()
            })
        
        return {
            "success": True,
            "total_count": len(poc_summaries),
            "pocs": poc_summaries
        }
        
    except Exception as e:
        logger.error(f"Error listing POCs: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/thought-leadership/create", response_model=Dict[str, Any])
async def create_thought_leadership_content(request: ThoughtLeadershipRequest):
    """Create thought leadership content to establish industry authority"""
    try:
        content = await demand_engine.create_thought_leadership_content(
            title=request.title,
            content_type=request.content_type,
            target_audience=request.target_audience,
            key_messages=request.key_messages,
            author_credentials=request.author_credentials
        )
        
        return {
            "success": True,
            "content_id": content.id,
            "title": content.title,
            "content_type": content.content_type.value,
            "distribution_channels": content.distribution_channels,
            "publication_date": content.publication_date.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error creating thought leadership content: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/thought-leadership/track-engagement", response_model=Dict[str, Any])
async def track_content_engagement(request: EngagementTrackingRequest):
    """Track engagement metrics for thought leadership content"""
    try:
        await demand_engine.track_content_engagement(
            content_id=request.content_id,
            engagement_data=request.engagement_data
        )
        
        # Get updated content details
        content = demand_engine.thought_leadership[request.content_id]
        
        return {
            "success": True,
            "content_id": request.content_id,
            "influence_score": content.influence_score,
            "engagement_metrics": content.engagement_metrics,
            "citations": content.citations,
            "media_mentions": content.media_mentions
        }
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error tracking content engagement: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/thought-leadership/{content_id}", response_model=Dict[str, Any])
async def get_thought_leadership_content(content_id: str):
    """Get detailed information about thought leadership content"""
    try:
        if content_id not in demand_engine.thought_leadership:
            raise HTTPException(status_code=404, detail=f"Content not found: {content_id}")
        
        content = demand_engine.thought_leadership[content_id]
        
        return {
            "id": content.id,
            "title": content.title,
            "content_type": content.content_type.value,
            "target_audience": content.target_audience,
            "key_messages": content.key_messages,
            "distribution_channels": content.distribution_channels,
            "author_credentials": content.author_credentials,
            "publication_date": content.publication_date.isoformat(),
            "engagement_metrics": content.engagement_metrics,
            "influence_score": content.influence_score,
            "citations": content.citations,
            "media_mentions": content.media_mentions
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting thought leadership content: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/thought-leadership", response_model=Dict[str, Any])
async def list_thought_leadership_content(
    content_type: Optional[ThoughtLeadershipType] = None,
    min_influence_score: Optional[float] = None
):
    """List all thought leadership content with optional filtering"""
    try:
        content_list = list(demand_engine.thought_leadership.values())
        
        # Apply filters
        if content_type:
            content_list = [content for content in content_list if content.content_type == content_type]
        if min_influence_score is not None:
            content_list = [content for content in content_list if content.influence_score >= min_influence_score]
        
        content_summaries = []
        for content in content_list:
            content_summaries.append({
                "id": content.id,
                "title": content.title,
                "content_type": content.content_type.value,
                "publication_date": content.publication_date.isoformat(),
                "influence_score": content.influence_score,
                "citations": content.citations,
                "media_mentions": content.media_mentions
            })
        
        return {
            "success": True,
            "total_count": len(content_summaries),
            "content": content_summaries
        }
        
    except Exception as e:
        logger.error(f"Error listing thought leadership content: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/industry-standard/create", response_model=Dict[str, Any])
async def create_industry_standard(request: IndustryStandardRequest):
    """Create and promote new industry standard that favors ScrollIntel"""
    try:
        standard = await demand_engine.create_industry_standard(
            name=request.name,
            description=request.description,
            industry=request.industry,
            standard_type=request.standard_type,
            stakeholders=request.stakeholders
        )
        
        return {
            "success": True,
            "standard_id": standard.id,
            "name": standard.name,
            "industry_vertical": standard.industry_vertical.value,
            "standard_type": standard.standard_type,
            "development_stage": standard.development_stage,
            "stakeholders": standard.stakeholders
        }
        
    except Exception as e:
        logger.error(f"Error creating industry standard: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/industry-standard/advance", response_model=Dict[str, Any])
async def advance_standard_development(request: StandardAdvancementRequest):
    """Advance industry standard through development stages"""
    try:
        advancement_result = await demand_engine.advance_standard_development(
            standard_id=request.standard_id,
            new_stage=request.new_stage
        )
        
        return {
            "success": True,
            **advancement_result
        }
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error advancing standard development: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/industry-standard/{standard_id}", response_model=Dict[str, Any])
async def get_industry_standard(standard_id: str):
    """Get detailed information about an industry standard"""
    try:
        if standard_id not in demand_engine.industry_standards:
            raise HTTPException(status_code=404, detail=f"Standard not found: {standard_id}")
        
        standard = demand_engine.industry_standards[standard_id]
        
        return {
            "id": standard.id,
            "name": standard.name,
            "description": standard.description,
            "industry_vertical": standard.industry_vertical.value,
            "standard_type": standard.standard_type,
            "development_stage": standard.development_stage,
            "stakeholders": standard.stakeholders,
            "adoption_rate": standard.adoption_rate,
            "competitive_advantage": standard.competitive_advantage,
            "created_at": standard.created_at.isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting industry standard: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/industry-standard", response_model=Dict[str, Any])
async def list_industry_standards(
    industry: Optional[IndustryVertical] = None,
    standard_type: Optional[str] = None,
    development_stage: Optional[str] = None
):
    """List all industry standards with optional filtering"""
    try:
        standards = list(demand_engine.industry_standards.values())
        
        # Apply filters
        if industry:
            standards = [std for std in standards if std.industry_vertical == industry]
        if standard_type:
            standards = [std for std in standards if std.standard_type == standard_type]
        if development_stage:
            standards = [std for std in standards if std.development_stage == development_stage]
        
        standard_summaries = []
        for standard in standards:
            standard_summaries.append({
                "id": standard.id,
                "name": standard.name,
                "industry_vertical": standard.industry_vertical.value,
                "standard_type": standard.standard_type,
                "development_stage": standard.development_stage,
                "adoption_rate": standard.adoption_rate,
                "competitive_advantage": standard.competitive_advantage,
                "created_at": standard.created_at.isoformat()
            })
        
        return {
            "success": True,
            "total_count": len(standard_summaries),
            "standards": standard_summaries
        }
        
    except Exception as e:
        logger.error(f"Error listing industry standards: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/demand-pipeline", response_model=Dict[str, Any])
async def get_demand_pipeline():
    """Get current demand pipeline metrics across all stages"""
    try:
        pipeline_metrics = await demand_engine.measure_demand_pipeline()
        
        pipeline_data = {}
        for stage, metrics in pipeline_metrics.items():
            pipeline_data[stage.value] = {
                "volume": metrics.volume,
                "quality_score": metrics.quality_score,
                "conversion_rate": metrics.conversion_rate,
                "velocity": metrics.velocity,
                "sources": metrics.sources,
                "last_updated": metrics.last_updated.isoformat()
            }
        
        return {
            "success": True,
            "pipeline_metrics": pipeline_data,
            "total_pocs": len(demand_engine.proof_of_concepts),
            "total_content": len(demand_engine.thought_leadership),
            "total_standards": len(demand_engine.industry_standards)
        }
        
    except Exception as e:
        logger.error(f"Error getting demand pipeline: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/enterprise-targets", response_model=Dict[str, Any])
async def get_enterprise_targets():
    """Get information about enterprise targets and their engagement levels"""
    try:
        return {
            "success": True,
            "enterprise_targets": demand_engine.enterprise_targets,
            "total_enterprises": len(demand_engine.enterprise_targets)
        }
        
    except Exception as e:
        logger.error(f"Error getting enterprise targets: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/analytics/summary", response_model=Dict[str, Any])
async def get_demand_creation_analytics():
    """Get comprehensive analytics summary for demand creation activities"""
    try:
        # Calculate summary statistics
        total_pocs = len(demand_engine.proof_of_concepts)
        completed_pocs = len([poc for poc in demand_engine.proof_of_concepts.values() if poc.status == "completed"])
        
        total_content = len(demand_engine.thought_leadership)
        high_influence_content = len([content for content in demand_engine.thought_leadership.values() if content.influence_score > 0.7])
        
        total_standards = len(demand_engine.industry_standards)
        adopted_standards = len([std for std in demand_engine.industry_standards.values() if std.development_stage == "adopted"])
        
        # Calculate average metrics
        avg_poc_investment = sum(poc.investment_required for poc in demand_engine.proof_of_concepts.values()) / max(total_pocs, 1)
        avg_influence_score = sum(content.influence_score for content in demand_engine.thought_leadership.values()) / max(total_content, 1)
        
        return {
            "success": True,
            "summary": {
                "proof_of_concepts": {
                    "total": total_pocs,
                    "completed": completed_pocs,
                    "completion_rate": completed_pocs / max(total_pocs, 1),
                    "average_investment": avg_poc_investment
                },
                "thought_leadership": {
                    "total": total_content,
                    "high_influence": high_influence_content,
                    "high_influence_rate": high_influence_content / max(total_content, 1),
                    "average_influence_score": avg_influence_score
                },
                "industry_standards": {
                    "total": total_standards,
                    "adopted": adopted_standards,
                    "adoption_rate": adopted_standards / max(total_standards, 1)
                },
                "enterprise_engagement": {
                    "total_enterprises": len(demand_engine.enterprise_targets),
                    "high_engagement": len([ent for ent in demand_engine.enterprise_targets.values() if ent.get("engagement_level") == "high_interest"])
                }
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting demand creation analytics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))