"""
ScrollIntel Competitive Intelligence API Routes
RESTful API endpoints for competitive analysis and market positioning
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from typing import Dict, List, Optional, Any
import asyncio
from datetime import datetime

from scrollintel.core.competitive_intelligence_system import (
    competitive_intelligence,
    CompetitorTier,
    ThreatLevel,
    MarketPosition
)

router = APIRouter(prefix="/api/v1/competitive-intelligence", tags=["Competitive Intelligence"])

@router.get("/dashboard")
async def get_competitive_dashboard():
    """Get real-time competitive intelligence dashboard"""
    try:
        dashboard = competitive_intelligence.get_competitive_dashboard()
        return {
            "status": "success",
            "data": dashboard,
            "timestamp": datetime.now().isoformat(),
            "message": "Competitive intelligence dashboard retrieved successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve dashboard: {str(e)}")

@router.get("/threats/analysis")
async def analyze_competitive_threats():
    """Analyze current competitive threats and market dynamics"""
    try:
        threat_analysis = await competitive_intelligence.analyze_competitive_threats()
        return {
            "status": "success",
            "data": threat_analysis,
            "timestamp": datetime.now().isoformat(),
            "message": "Competitive threat analysis completed successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to analyze threats: {str(e)}")

@router.get("/positioning/report")
async def generate_positioning_report():
    """Generate comprehensive market positioning and competitive analysis report"""
    try:
        report = await competitive_intelligence.generate_market_positioning_report()
        return {
            "status": "success",
            "data": report,
            "timestamp": datetime.now().isoformat(),
            "message": "Market positioning report generated successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate report: {str(e)}")

@router.get("/competitors")
async def get_competitors(
    tier: Optional[str] = Query(None, description="Filter by competitor tier"),
    threat_level: Optional[str] = Query(None, description="Filter by threat level"),
    market_position: Optional[str] = Query(None, description="Filter by market position")
):
    """Get competitor profiles with optional filtering"""
    try:
        competitors = competitive_intelligence.competitors
        
        # Apply filters
        filtered_competitors = {}
        for comp_id, competitor in competitors.items():
            include = True
            
            if tier and competitor.tier.value != tier:
                include = False
            if threat_level and competitor.threat_level.value != threat_level:
                include = False
            if market_position and competitor.market_position.value != market_position:
                include = False
            
            if include:
                # Convert competitor profile to dict for JSON serialization
                filtered_competitors[comp_id] = {
                    "competitor_id": competitor.competitor_id,
                    "company_name": competitor.company_name,
                    "tier": competitor.tier.value,
                    "market_position": competitor.market_position.value,
                    "threat_level": competitor.threat_level.value,
                    "annual_revenue": competitor.annual_revenue,
                    "employee_count": competitor.employee_count,
                    "funding_raised": competitor.funding_raised,
                    "key_products": competitor.key_products,
                    "target_market": competitor.target_market,
                    "pricing_model": competitor.pricing_model,
                    "key_differentiators": competitor.key_differentiators,
                    "weaknesses": competitor.weaknesses,
                    "recent_developments": competitor.recent_developments,
                    "market_share": competitor.market_share,
                    "customer_count": competitor.customer_count,
                    "geographic_presence": competitor.geographic_presence,
                    "technology_stack": competitor.technology_stack,
                    "leadership_team": competitor.leadership_team,
                    "financial_health": competitor.financial_health,
                    "growth_trajectory": competitor.growth_trajectory,
                    "competitive_advantages": competitor.competitive_advantages,
                    "strategic_partnerships": competitor.strategic_partnerships
                }
        
        return {
            "status": "success",
            "data": {
                "competitors": filtered_competitors,
                "total_count": len(filtered_competitors),
                "filters_applied": {
                    "tier": tier,
                    "threat_level": threat_level,
                    "market_position": market_position
                }
            },
            "timestamp": datetime.now().isoformat(),
            "message": f"Retrieved {len(filtered_competitors)} competitor profiles"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve competitors: {str(e)}")

@router.get("/competitors/{competitor_id}")
async def get_competitor_profile(competitor_id: str):
    """Get detailed profile for specific competitor"""
    try:
        if competitor_id not in competitive_intelligence.competitors:
            raise HTTPException(status_code=404, detail=f"Competitor {competitor_id} not found")
        
        competitor = competitive_intelligence.competitors[competitor_id]
        
        # Convert to dict for JSON serialization
        competitor_profile = {
            "competitor_id": competitor.competitor_id,
            "company_name": competitor.company_name,
            "tier": competitor.tier.value,
            "market_position": competitor.market_position.value,
            "threat_level": competitor.threat_level.value,
            "annual_revenue": competitor.annual_revenue,
            "employee_count": competitor.employee_count,
            "funding_raised": competitor.funding_raised,
            "key_products": competitor.key_products,
            "target_market": competitor.target_market,
            "pricing_model": competitor.pricing_model,
            "key_differentiators": competitor.key_differentiators,
            "weaknesses": competitor.weaknesses,
            "recent_developments": competitor.recent_developments,
            "market_share": competitor.market_share,
            "customer_count": competitor.customer_count,
            "geographic_presence": competitor.geographic_presence,
            "technology_stack": competitor.technology_stack,
            "leadership_team": competitor.leadership_team,
            "financial_health": competitor.financial_health,
            "growth_trajectory": competitor.growth_trajectory,
            "competitive_advantages": competitor.competitive_advantages,
            "strategic_partnerships": competitor.strategic_partnerships
        }
        
        return {
            "status": "success",
            "data": competitor_profile,
            "timestamp": datetime.now().isoformat(),
            "message": f"Competitor profile for {competitor.company_name} retrieved successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve competitor profile: {str(e)}")

@router.get("/market/intelligence")
async def get_market_intelligence():
    """Get comprehensive market intelligence analysis"""
    try:
        market_intel = competitive_intelligence.market_intelligence
        
        # Convert to dict for JSON serialization
        intelligence_data = {
            "market_size": market_intel.market_size,
            "growth_rate": market_intel.growth_rate,
            "key_trends": market_intel.key_trends,
            "adoption_barriers": market_intel.adoption_barriers,
            "buyer_personas": market_intel.buyer_personas,
            "decision_criteria": market_intel.decision_criteria,
            "budget_allocation_patterns": market_intel.budget_allocation_patterns,
            "technology_readiness": market_intel.technology_readiness,
            "regulatory_environment": market_intel.regulatory_environment,
            "competitive_landscape_summary": market_intel.competitive_landscape_summary
        }
        
        return {
            "status": "success",
            "data": intelligence_data,
            "timestamp": datetime.now().isoformat(),
            "message": "Market intelligence data retrieved successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve market intelligence: {str(e)}")

@router.get("/positioning/strategy")
async def get_positioning_strategy():
    """Get ScrollIntel's market positioning strategy"""
    try:
        positioning = competitive_intelligence.positioning_strategy
        
        # Convert to dict for JSON serialization
        strategy_data = {
            "unique_value_proposition": positioning.unique_value_proposition,
            "key_differentiators": positioning.key_differentiators,
            "target_segments": positioning.target_segments,
            "messaging_framework": positioning.messaging_framework,
            "competitive_advantages": positioning.competitive_advantages,
            "proof_points": positioning.proof_points,
            "objection_handling": positioning.objection_handling,
            "pricing_strategy": positioning.pricing_strategy,
            "go_to_market_approach": positioning.go_to_market_approach
        }
        
        return {
            "status": "success",
            "data": strategy_data,
            "timestamp": datetime.now().isoformat(),
            "message": "Positioning strategy retrieved successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve positioning strategy: {str(e)}")

@router.get("/threats/monitoring")
async def get_threat_monitoring():
    """Get real-time threat monitoring and alerts"""
    try:
        # Get current threat levels
        critical_threats = [
            comp for comp in competitive_intelligence.competitors.values() 
            if comp.threat_level == ThreatLevel.CRITICAL
        ]
        high_threats = [
            comp for comp in competitive_intelligence.competitors.values() 
            if comp.threat_level == ThreatLevel.HIGH
        ]
        
        # Generate threat monitoring data
        monitoring_data = {
            "threat_summary": {
                "critical_threats": len(critical_threats),
                "high_threats": len(high_threats),
                "total_active_threats": len(critical_threats) + len(high_threats),
                "threat_trend": "STABLE",
                "last_updated": datetime.now().isoformat()
            },
            "critical_threats": [
                {
                    "company_name": comp.company_name,
                    "threat_level": comp.threat_level.value,
                    "market_share": comp.market_share,
                    "growth_trajectory": comp.growth_trajectory,
                    "key_concerns": comp.competitive_advantages[:3],  # Top 3 concerns
                    "recent_activity": comp.recent_developments[-1] if comp.recent_developments else None
                }
                for comp in critical_threats
            ],
            "high_threats": [
                {
                    "company_name": comp.company_name,
                    "threat_level": comp.threat_level.value,
                    "market_share": comp.market_share,
                    "growth_trajectory": comp.growth_trajectory,
                    "key_concerns": comp.competitive_advantages[:2],  # Top 2 concerns
                    "recent_activity": comp.recent_developments[-1] if comp.recent_developments else None
                }
                for comp in high_threats
            ],
            "monitoring_alerts": [
                {
                    "alert_type": "MARKET_OPPORTUNITY",
                    "message": "AI CTO market showing 45% growth with no category leader",
                    "priority": "HIGH",
                    "action_required": "Accelerate market category creation"
                },
                {
                    "alert_type": "COMPETITIVE_GAP",
                    "message": "No direct competitors offering complete CTO replacement",
                    "priority": "CRITICAL",
                    "action_required": "Establish first-mover advantage immediately"
                }
            ]
        }
        
        return {
            "status": "success",
            "data": monitoring_data,
            "timestamp": datetime.now().isoformat(),
            "message": "Threat monitoring data retrieved successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve threat monitoring: {str(e)}")

@router.get("/opportunities/analysis")
async def analyze_market_opportunities():
    """Analyze current market opportunities and strategic recommendations"""
    try:
        # Generate market opportunity analysis
        opportunities = [
            {
                "opportunity_id": "ai_cto_category_creation",
                "title": "Complete AI CTO Category Creation",
                "market_size": 50000000000,  # $50B
                "competitive_gap": "No direct competitors offering complete CTO replacement",
                "time_to_market": "Immediate - First-mover advantage available",
                "success_probability": 0.95,
                "investment_required": 25000000,  # $25M
                "expected_roi": 12.0,  # 1200%
                "strategic_importance": "CRITICAL",
                "key_success_factors": [
                    "Aggressive thought leadership and market education",
                    "Fortune 500 enterprise pilot program execution",
                    "Technology leadership and patent protection",
                    "Strategic partnership development"
                ]
            },
            {
                "opportunity_id": "enterprise_ai_leadership_vacuum",
                "title": "Enterprise AI Leadership Vacuum",
                "market_size": 25000000000,  # $25B
                "competitive_gap": "Existing solutions provide partial assistance, not complete replacement",
                "time_to_market": "30-60 days for market entry",
                "success_probability": 0.90,
                "investment_required": 15000000,  # $15M
                "expected_roi": 8.0,  # 800%
                "strategic_importance": "HIGH",
                "key_success_factors": [
                    "Enterprise demonstration campaign",
                    "C-level executive engagement",
                    "Measurable ROI validation",
                    "Change management support"
                ]
            },
            {
                "opportunity_id": "global_scalability_advantage",
                "title": "Global Scalability Advantage",
                "market_size": 100000000000,  # $100B
                "competitive_gap": "Human-dependent competitors cannot scale globally",
                "time_to_market": "12-18 months for international expansion",
                "success_probability": 0.85,
                "investment_required": 50000000,  # $50M
                "expected_roi": 15.0,  # 1500%
                "strategic_importance": "HIGH",
                "key_success_factors": [
                    "Proven domestic market success",
                    "Localization and compliance framework",
                    "International partnership network",
                    "Global enterprise customer base"
                ]
            }
        ]
        
        return {
            "status": "success",
            "data": {
                "opportunities": opportunities,
                "total_market_opportunity": sum(opp["market_size"] for opp in opportunities),
                "weighted_success_probability": sum(opp["success_probability"] * opp["market_size"] for opp in opportunities) / sum(opp["market_size"] for opp in opportunities),
                "total_investment_required": sum(opp["investment_required"] for opp in opportunities),
                "expected_combined_roi": sum(opp["expected_roi"] * opp["investment_required"] for opp in opportunities) / sum(opp["investment_required"] for opp in opportunities)
            },
            "timestamp": datetime.now().isoformat(),
            "message": "Market opportunity analysis completed successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to analyze opportunities: {str(e)}")

@router.get("/health")
async def health_check():
    """Health check endpoint for competitive intelligence system"""
    try:
        # Verify system components
        competitors_loaded = len(competitive_intelligence.competitors) > 0
        market_intel_loaded = competitive_intelligence.market_intelligence is not None
        positioning_loaded = competitive_intelligence.positioning_strategy is not None
        
        return {
            "status": "healthy",
            "components": {
                "competitors_database": "operational" if competitors_loaded else "error",
                "market_intelligence": "operational" if market_intel_loaded else "error",
                "positioning_strategy": "operational" if positioning_loaded else "error"
            },
            "metrics": {
                "competitors_tracked": len(competitive_intelligence.competitors),
                "market_size": competitive_intelligence.market_intelligence.market_size if market_intel_loaded else 0,
                "positioning_differentiators": len(competitive_intelligence.positioning_strategy.key_differentiators) if positioning_loaded else 0
            },
            "timestamp": datetime.now().isoformat(),
            "message": "Competitive intelligence system is operational"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")