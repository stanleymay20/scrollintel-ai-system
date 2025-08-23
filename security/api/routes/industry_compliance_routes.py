"""
Industry Compliance API Routes
Provides REST API endpoints for industry-tailored compliance modules
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

from ...compliance.industry_compliance_framework import (
    IndustryComplianceOrchestrator, IndustryType, ComplianceStatus
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/compliance", tags=["Industry Compliance"])

# Global orchestrator instance
compliance_orchestrator = IndustryComplianceOrchestrator()

@router.get("/industries")
async def get_supported_industries():
    """Get list of supported industries"""
    try:
        industries = [industry.value for industry in IndustryType]
        return {
            "supported_industries": industries,
            "total_count": len(industries)
        }
    except Exception as e:
        logger.error(f"Error getting supported industries: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve supported industries")

@router.get("/industries/{industry}/rules")
async def get_industry_rules(industry: str):
    """Get compliance rules for specific industry"""
    try:
        industry_type = IndustryType(industry.lower())
        rules = compliance_orchestrator.get_industry_rules(industry_type)
        
        return {
            "industry": industry,
            "rules": [
                {
                    "rule_id": rule.rule_id,
                    "name": rule.name,
                    "description": rule.description,
                    "regulation": rule.regulation,
                    "severity": rule.severity,
                    "automated_check": rule.automated_check,
                    "remediation_steps": rule.remediation_steps,
                    "evidence_requirements": rule.evidence_requirements
                }
                for rule in rules
            ],
            "total_rules": len(rules)
        }
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Unsupported industry: {industry}")
    except Exception as e:
        logger.error(f"Error getting rules for industry {industry}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve industry rules")

@router.post("/industries/{industry}/assess")
async def assess_industry_compliance(
    industry: str,
    system_data: Dict[str, Any],
    background_tasks: BackgroundTasks
):
    """Assess compliance for specific industry"""
    try:
        industry_type = IndustryType(industry.lower())
        
        # Perform compliance assessment
        assessment = compliance_orchestrator.assess_industry_compliance(
            industry_type, system_data
        )
        
        # Schedule background task for detailed reporting
        background_tasks.add_task(
            _generate_detailed_report,
            assessment.assessment_id,
            industry_type,
            assessment
        )
        
        return {
            "assessment_id": assessment.assessment_id,
            "industry": industry,
            "timestamp": assessment.timestamp.isoformat(),
            "overall_status": assessment.overall_status.value,
            "compliance_rate": (len(assessment.compliant_rules) / len(assessment.rules_evaluated)) * 100,
            "risk_score": assessment.risk_score,
            "rules_evaluated": len(assessment.rules_evaluated),
            "compliant_rules": len(assessment.compliant_rules),
            "non_compliant_rules": len(assessment.non_compliant_rules),
            "recommendations": assessment.recommendations[:5],  # Top 5 recommendations
            "detailed_report_generating": True
        }
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Unsupported industry: {industry}")
    except Exception as e:
        logger.error(f"Error assessing compliance for industry {industry}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to assess compliance")

@router.get("/assessments/{assessment_id}")
async def get_assessment_details(assessment_id: str):
    """Get detailed assessment results"""
    try:
        # In a real implementation, this would retrieve from database
        # For now, return a placeholder response
        return {
            "assessment_id": assessment_id,
            "status": "completed",
            "message": "Assessment details would be retrieved from database"
        }
    except Exception as e:
        logger.error(f"Error getting assessment {assessment_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve assessment")

@router.post("/multi-industry-assessment")
async def assess_multiple_industries(
    assessment_request: Dict[str, Any],
    background_tasks: BackgroundTasks
):
    """Assess compliance across multiple industries"""
    try:
        industries = assessment_request.get("industries", [])
        system_data = assessment_request.get("system_data", {})
        
        if not industries:
            raise HTTPException(status_code=400, detail="No industries specified")
        
        assessments = []
        for industry in industries:
            try:
                industry_type = IndustryType(industry.lower())
                assessment = compliance_orchestrator.assess_industry_compliance(
                    industry_type, system_data
                )
                assessments.append(assessment)
            except ValueError:
                logger.warning(f"Skipping unsupported industry: {industry}")
                continue
        
        if not assessments:
            raise HTTPException(status_code=400, detail="No valid industries provided")
        
        # Generate cross-industry report
        cross_industry_report = compliance_orchestrator.generate_cross_industry_report(assessments)
        
        # Schedule background task for detailed reporting
        background_tasks.add_task(
            _generate_cross_industry_detailed_report,
            assessments,
            cross_industry_report
        )
        
        return {
            "assessment_timestamp": datetime.utcnow().isoformat(),
            "industries_assessed": len(assessments),
            "overall_compliance_rate": cross_industry_report["overall_compliance_rate"],
            "industry_breakdown": cross_industry_report["industry_breakdown"],
            "critical_findings_count": len(cross_industry_report["critical_findings"]),
            "recommendations_count": len(cross_industry_report["recommendations"]),
            "detailed_report_generating": True
        }
    except Exception as e:
        logger.error(f"Error in multi-industry assessment: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to perform multi-industry assessment")

@router.get("/compliance-dashboard")
async def get_compliance_dashboard():
    """Get compliance dashboard data"""
    try:
        # This would typically aggregate data from recent assessments
        dashboard_data = {
            "summary": {
                "total_industries_supported": len(IndustryType),
                "recent_assessments": 0,  # Would be from database
                "average_compliance_rate": 0.0,  # Would be calculated from recent assessments
                "critical_findings": 0  # Would be from recent assessments
            },
            "industry_status": {
                industry.value: {
                    "last_assessment": None,
                    "compliance_rate": 0.0,
                    "status": "not_assessed"
                }
                for industry in IndustryType
            },
            "trending_issues": [],  # Would be from analysis of recent assessments
            "upcoming_assessments": []  # Would be from scheduled assessments
        }
        
        return dashboard_data
    except Exception as e:
        logger.error(f"Error generating compliance dashboard: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to generate compliance dashboard")

@router.get("/regulations/{regulation_name}")
async def get_regulation_details(regulation_name: str):
    """Get details about specific regulation"""
    try:
        # This would typically query a regulations database
        regulation_info = {
            "regulation_name": regulation_name,
            "description": f"Details about {regulation_name}",
            "applicable_industries": [],
            "key_requirements": [],
            "compliance_controls": [],
            "last_updated": datetime.utcnow().isoformat()
        }
        
        return regulation_info
    except Exception as e:
        logger.error(f"Error getting regulation details for {regulation_name}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve regulation details")

@router.post("/compliance-gap-analysis")
async def perform_gap_analysis(
    gap_analysis_request: Dict[str, Any]
):
    """Perform compliance gap analysis"""
    try:
        industry = gap_analysis_request.get("industry")
        current_controls = gap_analysis_request.get("current_controls", [])
        target_framework = gap_analysis_request.get("target_framework", "")
        
        if not industry:
            raise HTTPException(status_code=400, detail="Industry is required")
        
        industry_type = IndustryType(industry.lower())
        rules = compliance_orchestrator.get_industry_rules(industry_type)
        
        # Perform gap analysis
        gaps = []
        for rule in rules:
            if rule.rule_id not in current_controls:
                gaps.append({
                    "rule_id": rule.rule_id,
                    "name": rule.name,
                    "regulation": rule.regulation,
                    "severity": rule.severity,
                    "remediation_steps": rule.remediation_steps,
                    "priority": _calculate_gap_priority(rule)
                })
        
        # Sort gaps by priority
        gaps.sort(key=lambda x: x["priority"], reverse=True)
        
        return {
            "industry": industry,
            "target_framework": target_framework,
            "total_requirements": len(rules),
            "current_controls": len(current_controls),
            "identified_gaps": len(gaps),
            "compliance_percentage": (len(current_controls) / len(rules)) * 100,
            "gaps": gaps[:20],  # Top 20 gaps
            "recommendations": _generate_gap_recommendations(gaps[:10])
        }
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Unsupported industry: {industry}")
    except Exception as e:
        logger.error(f"Error performing gap analysis: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to perform gap analysis")

async def _generate_detailed_report(
    assessment_id: str,
    industry_type: IndustryType,
    assessment
):
    """Background task to generate detailed compliance report"""
    try:
        module = compliance_orchestrator.modules.get(industry_type)
        if module:
            detailed_report = module.generate_report(assessment)
            # In a real implementation, this would be saved to database
            logger.info(f"Generated detailed report for assessment {assessment_id}")
    except Exception as e:
        logger.error(f"Error generating detailed report for {assessment_id}: {str(e)}")

async def _generate_cross_industry_detailed_report(
    assessments: List,
    cross_industry_report: Dict[str, Any]
):
    """Background task to generate detailed cross-industry report"""
    try:
        # Generate comprehensive cross-industry analysis
        # In a real implementation, this would be saved to database
        logger.info("Generated detailed cross-industry compliance report")
    except Exception as e:
        logger.error(f"Error generating cross-industry detailed report: {str(e)}")

def _calculate_gap_priority(rule) -> int:
    """Calculate priority score for compliance gap"""
    severity_weights = {
        "critical": 10,
        "high": 7,
        "medium": 4,
        "low": 1
    }
    
    base_score = severity_weights.get(rule.severity.lower(), 1)
    
    # Increase priority for automated checks (easier to implement)
    if rule.automated_check:
        base_score += 2
    
    return base_score

def _generate_gap_recommendations(gaps: List[Dict[str, Any]]) -> List[str]:
    """Generate recommendations based on identified gaps"""
    recommendations = []
    
    critical_gaps = [gap for gap in gaps if gap["severity"] == "critical"]
    if critical_gaps:
        recommendations.append(f"Address {len(critical_gaps)} critical compliance gaps immediately")
    
    automated_gaps = [gap for gap in gaps if gap.get("automated_check", False)]
    if automated_gaps:
        recommendations.append(f"Implement {len(automated_gaps)} automated compliance controls for quick wins")
    
    recommendations.append("Prioritize gaps based on regulatory deadlines and business impact")
    recommendations.append("Establish a compliance remediation timeline with clear milestones")
    
    return recommendations