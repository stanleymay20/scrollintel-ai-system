"""
Talent Monopoly API Routes

Provides REST API endpoints for the global talent monopoly system,
enabling talent identification, recruitment, and retention management.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict, List, Optional
from pydantic import BaseModel
import logging

from scrollintel.engines.talent_monopoly_engine import (
    talent_monopoly_engine,
    TalentCategory,
    TalentTier,
    TalentProfile,
    RecruitmentCampaign
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/talent-monopoly", tags=["talent-monopoly"])


# Request/Response Models
class TalentIdentificationRequest(BaseModel):
    category: str
    target_count: int = 1000
    priority_tier: Optional[str] = None


class RecruitmentCampaignRequest(BaseModel):
    category: str
    target_tier: str
    target_count: int
    budget: float
    timeline_months: int = 12


class TalentAcquisitionRequest(BaseModel):
    talent_ids: List[str]
    priority_order: bool = True


class RetentionProgramRequest(BaseModel):
    talent_id: str
    custom_programs: Optional[List[str]] = None


@router.post("/identify-talent")
async def identify_talent(request: TalentIdentificationRequest):
    """
    Identify top global talent in specified category
    
    Executes comprehensive talent identification across multiple sources
    including GitHub, research publications, patents, and industry networks.
    """
    try:
        # Validate category
        try:
            category = TalentCategory(request.category.lower())
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid category. Valid options: {[c.value for c in TalentCategory]}"
            )
        
        logger.info(f"Starting talent identification for {category.value}")
        
        # Execute talent identification
        identified_talents = await talent_monopoly_engine.identify_global_talent(
            category=category,
            target_count=request.target_count
        )
        
        # Convert to response format
        talent_summaries = []
        for talent in identified_talents:
            talent_summaries.append({
                "id": talent.id,
                "name": talent.name,
                "category": talent.category.value,
                "tier": talent.tier.value,
                "current_company": talent.current_company,
                "location": talent.location,
                "skills": talent.skills[:5],  # Top 5 skills
                "compensation_estimate": talent.compensation_estimate,
                "acquisition_priority": talent.acquisition_priority,
                "status": talent.recruitment_status.value
            })
        
        return {
            "success": True,
            "message": f"Identified {len(identified_talents)} talents in {category.value}",
            "category": category.value,
            "total_identified": len(identified_talents),
            "talents": talent_summaries,
            "tier_distribution": _calculate_tier_distribution(identified_talents)
        }
        
    except Exception as e:
        logger.error(f"Error in talent identification: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Talent identification failed: {str(e)}")


@router.post("/create-campaign")
async def create_recruitment_campaign(request: RecruitmentCampaignRequest):
    """
    Create strategic recruitment campaign for specific talent category and tier
    
    Establishes comprehensive recruitment strategy with budget allocation,
    timeline, and success metrics for systematic talent acquisition.
    """
    try:
        # Validate inputs
        try:
            category = TalentCategory(request.category.lower())
            target_tier = TalentTier(request.target_tier.lower())
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Invalid parameter: {str(e)}")
        
        logger.info(f"Creating recruitment campaign for {category.value} {target_tier.value}")
        
        # Create campaign
        campaign = await talent_monopoly_engine.create_recruitment_campaign(
            category=category,
            target_tier=target_tier,
            target_count=request.target_count,
            budget=request.budget
        )
        
        return {
            "success": True,
            "message": f"Created recruitment campaign: {campaign.name}",
            "campaign": {
                "id": campaign.id,
                "name": campaign.name,
                "category": campaign.target_category.value,
                "tier": campaign.target_tier.value,
                "target_count": campaign.target_count,
                "budget": campaign.budget,
                "timeline_months": campaign.timeline_months,
                "strategies": campaign.strategies,
                "success_metrics": campaign.success_metrics,
                "status": campaign.status
            }
        }
        
    except Exception as e:
        logger.error(f"Error creating recruitment campaign: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Campaign creation failed: {str(e)}")


@router.post("/acquire-talent")
async def acquire_talent(request: TalentAcquisitionRequest, background_tasks: BackgroundTasks):
    """
    Execute talent acquisition for specified talents
    
    Implements comprehensive acquisition process including competitive
    compensation packages, negotiation, and onboarding coordination.
    """
    try:
        logger.info(f"Starting acquisition process for {len(request.talent_ids)} talents")
        
        acquisition_results = []
        
        # Process acquisitions (can be done in parallel for production)
        for talent_id in request.talent_ids:
            try:
                success = await talent_monopoly_engine.execute_acquisition(talent_id)
                
                acquisition_results.append({
                    "talent_id": talent_id,
                    "success": success,
                    "status": "acquired" if success else "failed"
                })
                
            except Exception as e:
                logger.error(f"Acquisition failed for {talent_id}: {str(e)}")
                acquisition_results.append({
                    "talent_id": talent_id,
                    "success": False,
                    "status": "error",
                    "error": str(e)
                })
        
        # Calculate success metrics
        successful_acquisitions = sum(1 for r in acquisition_results if r["success"])
        success_rate = successful_acquisitions / len(request.talent_ids)
        
        return {
            "success": True,
            "message": f"Acquisition process completed: {successful_acquisitions}/{len(request.talent_ids)} successful",
            "total_attempts": len(request.talent_ids),
            "successful_acquisitions": successful_acquisitions,
            "success_rate": success_rate,
            "results": acquisition_results
        }
        
    except Exception as e:
        logger.error(f"Error in talent acquisition: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Talent acquisition failed: {str(e)}")


@router.post("/implement-retention")
async def implement_retention_program(request: RetentionProgramRequest):
    """
    Implement comprehensive retention program for acquired talent
    
    Deploys multi-faceted retention strategies including research freedom,
    equity acceleration, sabbaticals, and publication support.
    """
    try:
        logger.info(f"Implementing retention program for talent {request.talent_id}")
        
        # Execute retention program
        retention_metrics = await talent_monopoly_engine.implement_retention_program(
            talent_id=request.talent_id
        )
        
        if not retention_metrics:
            raise HTTPException(status_code=404, detail="Talent not found")
        
        return {
            "success": True,
            "message": f"Retention program implemented for {request.talent_id}",
            "talent_id": request.talent_id,
            "retention_metrics": retention_metrics,
            "programs_applied": list(talent_monopoly_engine.retention_programs.keys())
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error implementing retention program: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Retention program failed: {str(e)}")


@router.get("/pipeline-analytics")
async def get_pipeline_analytics():
    """
    Get comprehensive talent pipeline analytics and performance metrics
    
    Provides detailed analysis of talent acquisition performance,
    retention rates, cost analysis, and competitive positioning.
    """
    try:
        logger.info("Generating pipeline analytics")
        
        # Get pipeline metrics
        pipeline_metrics = await talent_monopoly_engine.monitor_talent_pipeline()
        
        # Get competitive analysis
        competitive_analysis = await talent_monopoly_engine.analyze_competitive_landscape()
        
        # Get basic statistics
        talent_stats = talent_monopoly_engine.get_talent_statistics()
        
        return {
            "success": True,
            "message": "Pipeline analytics generated successfully",
            "pipeline_metrics": pipeline_metrics,
            "competitive_analysis": competitive_analysis,
            "talent_statistics": talent_stats,
            "generated_at": talent_monopoly_engine.talent_database[list(talent_monopoly_engine.talent_database.keys())[0]].created_at.isoformat() if talent_monopoly_engine.talent_database else None
        }
        
    except Exception as e:
        logger.error(f"Error generating pipeline analytics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analytics generation failed: {str(e)}")


@router.get("/talent/{talent_id}")
async def get_talent_details(talent_id: str):
    """
    Get detailed information about specific talent
    
    Returns comprehensive talent profile including skills, achievements,
    recruitment status, and retention metrics.
    """
    try:
        if talent_id not in talent_monopoly_engine.talent_database:
            raise HTTPException(status_code=404, detail="Talent not found")
        
        talent = talent_monopoly_engine.talent_database[talent_id]
        
        # Get compensation package for tier
        compensation_package = talent_monopoly_engine.compensation_tiers.get(talent.tier)
        
        return {
            "success": True,
            "talent": {
                "id": talent.id,
                "name": talent.name,
                "category": talent.category.value,
                "tier": talent.tier.value,
                "current_company": talent.current_company,
                "location": talent.location,
                "skills": talent.skills,
                "achievements": talent.achievements,
                "publications": talent.publications,
                "patents": talent.patents,
                "github_profile": talent.github_profile,
                "linkedin_profile": talent.linkedin_profile,
                "compensation_estimate": talent.compensation_estimate,
                "acquisition_priority": talent.acquisition_priority,
                "recruitment_status": talent.recruitment_status.value,
                "retention_score": talent.retention_score,
                "contact_history": talent.contact_history,
                "created_at": talent.created_at.isoformat(),
                "updated_at": talent.updated_at.isoformat()
            },
            "compensation_package": {
                "base_salary": compensation_package.base_salary,
                "equity_percentage": compensation_package.equity_percentage,
                "signing_bonus": compensation_package.signing_bonus,
                "total_package_value": compensation_package.total_package_value
            } if compensation_package else None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving talent details: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve talent details: {str(e)}")


@router.get("/campaigns")
async def get_recruitment_campaigns():
    """
    Get all active recruitment campaigns
    
    Returns list of all recruitment campaigns with their status,
    progress metrics, and performance indicators.
    """
    try:
        campaigns = []
        
        for campaign_id, campaign in talent_monopoly_engine.recruitment_campaigns.items():
            campaigns.append({
                "id": campaign.id,
                "name": campaign.name,
                "category": campaign.target_category.value,
                "tier": campaign.target_tier.value,
                "target_count": campaign.target_count,
                "budget": campaign.budget,
                "timeline_months": campaign.timeline_months,
                "strategies": campaign.strategies,
                "success_metrics": campaign.success_metrics,
                "status": campaign.status,
                "created_at": campaign.created_at.isoformat()
            })
        
        return {
            "success": True,
            "message": f"Retrieved {len(campaigns)} recruitment campaigns",
            "total_campaigns": len(campaigns),
            "campaigns": campaigns
        }
        
    except Exception as e:
        logger.error(f"Error retrieving campaigns: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve campaigns: {str(e)}")


@router.get("/compensation-tiers")
async def get_compensation_tiers():
    """
    Get compensation tier structure and packages
    
    Returns detailed compensation packages for each talent tier
    including salary, equity, bonuses, and total package values.
    """
    try:
        compensation_data = {}
        
        for tier, package in talent_monopoly_engine.compensation_tiers.items():
            compensation_data[tier.value] = {
                "base_salary": package.base_salary,
                "equity_percentage": package.equity_percentage,
                "signing_bonus": package.signing_bonus,
                "annual_bonus_target": package.annual_bonus_target,
                "research_budget": package.research_budget,
                "conference_budget": package.conference_budget,
                "relocation_package": package.relocation_package,
                "benefits_value": package.benefits_value,
                "total_package_value": package.total_package_value,
                "tier_multiplier": package.tier_multiplier
            }
        
        return {
            "success": True,
            "message": "Compensation tiers retrieved successfully",
            "compensation_tiers": compensation_data,
            "retention_programs": talent_monopoly_engine.retention_programs
        }
        
    except Exception as e:
        logger.error(f"Error retrieving compensation tiers: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve compensation data: {str(e)}")


def _calculate_tier_distribution(talents: List[TalentProfile]) -> Dict[str, int]:
    """Calculate distribution of talents by tier"""
    distribution = {}
    for talent in talents:
        tier = talent.tier.value
        distribution[tier] = distribution.get(tier, 0) + 1
    return distribution