"""
Marketing Attribution and Campaign Tracking
Tracks marketing campaigns, attribution models, and ROI analysis
"""

import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import asyncio
import logging
from enum import Enum
from urllib.parse import urlparse, parse_qs

logger = logging.getLogger(__name__)

class AttributionModel(Enum):
    FIRST_TOUCH = "first_touch"
    LAST_TOUCH = "last_touch"
    LINEAR = "linear"
    TIME_DECAY = "time_decay"
    POSITION_BASED = "position_based"

class CampaignStatus(Enum):
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    DRAFT = "draft"

@dataclass
class MarketingCampaign:
    campaign_id: str
    name: str
    description: str
    campaign_type: str  # email, social, search, display, etc.
    source: str  # google, facebook, email, etc.
    medium: str  # cpc, organic, email, social, etc.
    content: Optional[str]  # ad content identifier
    term: Optional[str]  # search terms
    budget: float
    start_date: datetime
    end_date: Optional[datetime]
    status: CampaignStatus
    target_audience: Dict[str, Any]
    goals: List[str]
    created_at: datetime
    metadata: Dict[str, Any]

@dataclass
class TouchPoint:
    touchpoint_id: str
    user_id: str
    session_id: str
    campaign_id: Optional[str]
    source: str
    medium: str
    content: Optional[str]
    term: Optional[str]
    page_url: str
    referrer: Optional[str]
    timestamp: datetime
    user_agent: str
    ip_address: str
    utm_parameters: Dict[str, str]

@dataclass
class Conversion:
    conversion_id: str
    user_id: str
    session_id: str
    conversion_type: str
    conversion_value: float
    timestamp: datetime
    attributed_touchpoints: List[str]
    attribution_weights: Dict[str, float]
    campaign_attribution: Dict[str, float]

@dataclass
class AttributionReport:
    report_id: str
    attribution_model: AttributionModel
    period_start: datetime
    period_end: datetime
    campaign_performance: Dict[str, Dict[str, Any]]
    channel_performance: Dict[str, Dict[str, Any]]
    conversion_paths: List[Dict[str, Any]]
    roi_analysis: Dict[str, Any]
    generated_at: datetime

class MarketingAttributionTracker:
    """Marketing attribution and campaign tracking system"""
    
    def __init__(self):
        self.campaigns: Dict[str, MarketingCampaign] = {}
        self.touchpoints: List[TouchPoint] = []
        self.conversions: List[Conversion] = []
        self.attribution_reports: List[AttributionReport] = []
        
        # Attribution model weights
        self.attribution_weights = {
            AttributionModel.FIRST_TOUCH: {"first": 1.0},
            AttributionModel.LAST_TOUCH: {"last": 1.0},
            AttributionModel.LINEAR: {"equal": True},
            AttributionModel.TIME_DECAY: {"decay_rate": 0.7},
            AttributionModel.POSITION_BASED: {"first": 0.4, "last": 0.4, "middle": 0.2}
        }
    
    async def create_campaign(self,
                             name: str,
                             description: str,
                             campaign_type: str,
                             source: str,
                             medium: str,
                             budget: float,
                             start_date: datetime,
                             end_date: Optional[datetime] = None,
                             content: Optional[str] = None,
                             term: Optional[str] = None,
                             target_audience: Dict[str, Any] = None,
                             goals: List[str] = None) -> str:
        """Create a new marketing campaign"""
        try:
            campaign_id = str(uuid.uuid4())
            
            campaign = MarketingCampaign(
                campaign_id=campaign_id,
                name=name,
                description=description,
                campaign_type=campaign_type,
                source=source,
                medium=medium,
                content=content,
                term=term,
                budget=budget,
                start_date=start_date,
                end_date=end_date,
                status=CampaignStatus.DRAFT,
                target_audience=target_audience or {},
                goals=goals or [],
                created_at=datetime.utcnow(),
                metadata={}
            )
            
            self.campaigns[campaign_id] = campaign
            logger.info(f"Created campaign: {name} ({campaign_id})")
            return campaign_id
            
        except Exception as e:
            logger.error(f"Error creating campaign: {str(e)}")
            raise
    
    async def track_touchpoint(self,
                              user_id: str,
                              session_id: str,
                              page_url: str,
                              referrer: Optional[str],
                              user_agent: str,
                              ip_address: str,
                              utm_parameters: Dict[str, str] = None) -> str:
        """Track a marketing touchpoint"""
        try:
            # Extract UTM parameters from URL if not provided
            if not utm_parameters:
                utm_parameters = self._extract_utm_parameters(page_url)
            
            # Find matching campaign
            campaign_id = await self._find_matching_campaign(utm_parameters)
            
            touchpoint = TouchPoint(
                touchpoint_id=str(uuid.uuid4()),
                user_id=user_id,
                session_id=session_id,
                campaign_id=campaign_id,
                source=utm_parameters.get("utm_source", self._extract_source_from_referrer(referrer)),
                medium=utm_parameters.get("utm_medium", self._extract_medium_from_referrer(referrer)),
                content=utm_parameters.get("utm_content"),
                term=utm_parameters.get("utm_term"),
                page_url=page_url,
                referrer=referrer,
                timestamp=datetime.utcnow(),
                user_agent=user_agent,
                ip_address=ip_address,
                utm_parameters=utm_parameters
            )
            
            self.touchpoints.append(touchpoint)
            logger.info(f"Tracked touchpoint for user {user_id} from {touchpoint.source}/{touchpoint.medium}")
            return touchpoint.touchpoint_id
            
        except Exception as e:
            logger.error(f"Error tracking touchpoint: {str(e)}")
            raise
    
    def _extract_utm_parameters(self, url: str) -> Dict[str, str]:
        """Extract UTM parameters from URL"""
        try:
            parsed_url = urlparse(url)
            query_params = parse_qs(parsed_url.query)
            
            utm_params = {}
            utm_keys = ["utm_source", "utm_medium", "utm_campaign", "utm_content", "utm_term"]
            
            for key in utm_keys:
                if key in query_params:
                    utm_params[key] = query_params[key][0]
            
            return utm_params
            
        except Exception:
            return {}
    
    def _extract_source_from_referrer(self, referrer: Optional[str]) -> str:
        """Extract source from referrer URL"""
        if not referrer:
            return "direct"
        
        try:
            parsed_referrer = urlparse(referrer)
            domain = parsed_referrer.netloc.lower()
            
            # Common source mappings
            source_mappings = {
                "google.com": "google",
                "facebook.com": "facebook",
                "twitter.com": "twitter",
                "linkedin.com": "linkedin",
                "youtube.com": "youtube",
                "instagram.com": "instagram",
                "bing.com": "bing",
                "yahoo.com": "yahoo"
            }
            
            for domain_part, source in source_mappings.items():
                if domain_part in domain:
                    return source
            
            return domain
            
        except Exception:
            return "unknown"
    
    def _extract_medium_from_referrer(self, referrer: Optional[str]) -> str:
        """Extract medium from referrer URL"""
        if not referrer:
            return "direct"
        
        try:
            parsed_referrer = urlparse(referrer)
            domain = parsed_referrer.netloc.lower()
            
            # Common medium mappings
            if any(search_engine in domain for search_engine in ["google", "bing", "yahoo", "duckduckgo"]):
                return "organic"
            elif any(social in domain for social in ["facebook", "twitter", "linkedin", "instagram"]):
                return "social"
            else:
                return "referral"
                
        except Exception:
            return "unknown"
    
    async def _find_matching_campaign(self, utm_parameters: Dict[str, str]) -> Optional[str]:
        """Find campaign matching UTM parameters"""
        if not utm_parameters:
            return None
        
        utm_campaign = utm_parameters.get("utm_campaign")
        utm_source = utm_parameters.get("utm_source")
        utm_medium = utm_parameters.get("utm_medium")
        
        for campaign in self.campaigns.values():
            if campaign.status != CampaignStatus.ACTIVE:
                continue
            
            # Match by campaign name or source/medium
            if (utm_campaign and utm_campaign.lower() in campaign.name.lower()) or \
               (utm_source == campaign.source and utm_medium == campaign.medium):
                return campaign.campaign_id
        
        return None
    
    async def track_conversion(self,
                              user_id: str,
                              session_id: str,
                              conversion_type: str,
                              conversion_value: float,
                              attribution_model: AttributionModel = AttributionModel.LAST_TOUCH) -> str:
        """Track a conversion and perform attribution"""
        try:
            # Get user's touchpoint journey
            user_touchpoints = [
                tp for tp in self.touchpoints
                if tp.user_id == user_id
            ]
            
            if not user_touchpoints:
                logger.warning(f"No touchpoints found for user {user_id}")
                return ""
            
            # Sort touchpoints by timestamp
            user_touchpoints.sort(key=lambda x: x.timestamp)
            
            # Apply attribution model
            attribution_weights = await self._calculate_attribution_weights(
                user_touchpoints, attribution_model
            )
            
            # Calculate campaign attribution
            campaign_attribution = {}
            for touchpoint_id, weight in attribution_weights.items():
                touchpoint = next(tp for tp in user_touchpoints if tp.touchpoint_id == touchpoint_id)
                if touchpoint.campaign_id:
                    campaign_attribution[touchpoint.campaign_id] = \
                        campaign_attribution.get(touchpoint.campaign_id, 0) + weight
            
            # Create conversion record
            conversion = Conversion(
                conversion_id=str(uuid.uuid4()),
                user_id=user_id,
                session_id=session_id,
                conversion_type=conversion_type,
                conversion_value=conversion_value,
                timestamp=datetime.utcnow(),
                attributed_touchpoints=[tp.touchpoint_id for tp in user_touchpoints],
                attribution_weights=attribution_weights,
                campaign_attribution=campaign_attribution
            )
            
            self.conversions.append(conversion)
            logger.info(f"Tracked conversion: {conversion_type} (${conversion_value}) for user {user_id}")
            return conversion.conversion_id
            
        except Exception as e:
            logger.error(f"Error tracking conversion: {str(e)}")
            raise
    
    async def _calculate_attribution_weights(self,
                                           touchpoints: List[TouchPoint],
                                           model: AttributionModel) -> Dict[str, float]:
        """Calculate attribution weights based on model"""
        if not touchpoints:
            return {}
        
        weights = {}
        
        if model == AttributionModel.FIRST_TOUCH:
            weights[touchpoints[0].touchpoint_id] = 1.0
            
        elif model == AttributionModel.LAST_TOUCH:
            weights[touchpoints[-1].touchpoint_id] = 1.0
            
        elif model == AttributionModel.LINEAR:
            weight_per_touchpoint = 1.0 / len(touchpoints)
            for tp in touchpoints:
                weights[tp.touchpoint_id] = weight_per_touchpoint
                
        elif model == AttributionModel.TIME_DECAY:
            decay_rate = self.attribution_weights[model]["decay_rate"]
            total_weight = 0
            
            # Calculate weights with time decay (more recent = higher weight)
            for i, tp in enumerate(reversed(touchpoints)):
                weight = decay_rate ** i
                weights[tp.touchpoint_id] = weight
                total_weight += weight
            
            # Normalize weights
            for tp_id in weights:
                weights[tp_id] /= total_weight
                
        elif model == AttributionModel.POSITION_BASED:
            if len(touchpoints) == 1:
                weights[touchpoints[0].touchpoint_id] = 1.0
            elif len(touchpoints) == 2:
                weights[touchpoints[0].touchpoint_id] = 0.5
                weights[touchpoints[-1].touchpoint_id] = 0.5
            else:
                # First and last get 40% each, middle touchpoints share 20%
                weights[touchpoints[0].touchpoint_id] = 0.4
                weights[touchpoints[-1].touchpoint_id] = 0.4
                
                middle_weight = 0.2 / (len(touchpoints) - 2)
                for tp in touchpoints[1:-1]:
                    weights[tp.touchpoint_id] = middle_weight
        
        return weights
    
    async def generate_attribution_report(self,
                                        attribution_model: AttributionModel,
                                        days: int = 30) -> AttributionReport:
        """Generate comprehensive attribution report"""
        try:
            period_end = datetime.utcnow()
            period_start = period_end - timedelta(days=days)
            
            # Filter conversions in period
            period_conversions = [
                c for c in self.conversions
                if period_start <= c.timestamp <= period_end
            ]
            
            if not period_conversions:
                return self._create_empty_report(attribution_model, period_start, period_end)
            
            # Calculate campaign performance
            campaign_performance = await self._calculate_campaign_performance(
                period_conversions, attribution_model
            )
            
            # Calculate channel performance
            channel_performance = await self._calculate_channel_performance(
                period_conversions, attribution_model
            )
            
            # Analyze conversion paths
            conversion_paths = await self._analyze_conversion_paths(period_conversions)
            
            # Calculate ROI
            roi_analysis = await self._calculate_roi_analysis(campaign_performance)
            
            report = AttributionReport(
                report_id=str(uuid.uuid4()),
                attribution_model=attribution_model,
                period_start=period_start,
                period_end=period_end,
                campaign_performance=campaign_performance,
                channel_performance=channel_performance,
                conversion_paths=conversion_paths,
                roi_analysis=roi_analysis,
                generated_at=datetime.utcnow()
            )
            
            self.attribution_reports.append(report)
            logger.info(f"Generated attribution report with {len(period_conversions)} conversions")
            return report
            
        except Exception as e:
            logger.error(f"Error generating attribution report: {str(e)}")
            raise
    
    def _create_empty_report(self, model: AttributionModel, start: datetime, end: datetime) -> AttributionReport:
        """Create empty report when no data available"""
        return AttributionReport(
            report_id=str(uuid.uuid4()),
            attribution_model=model,
            period_start=start,
            period_end=end,
            campaign_performance={},
            channel_performance={},
            conversion_paths=[],
            roi_analysis={"message": "No conversions in period"},
            generated_at=datetime.utcnow()
        )
    
    async def _calculate_campaign_performance(self,
                                            conversions: List[Conversion],
                                            model: AttributionModel) -> Dict[str, Dict[str, Any]]:
        """Calculate performance metrics for each campaign"""
        campaign_metrics = {}
        
        for conversion in conversions:
            for campaign_id, attribution_weight in conversion.campaign_attribution.items():
                if campaign_id not in campaign_metrics:
                    campaign_metrics[campaign_id] = {
                        "conversions": 0,
                        "attributed_conversions": 0.0,
                        "revenue": 0.0,
                        "attributed_revenue": 0.0
                    }
                
                campaign_metrics[campaign_id]["conversions"] += 1
                campaign_metrics[campaign_id]["attributed_conversions"] += attribution_weight
                campaign_metrics[campaign_id]["revenue"] += conversion.conversion_value
                campaign_metrics[campaign_id]["attributed_revenue"] += conversion.conversion_value * attribution_weight
        
        # Add campaign details and calculate additional metrics
        for campaign_id, metrics in campaign_metrics.items():
            if campaign_id in self.campaigns:
                campaign = self.campaigns[campaign_id]
                metrics["campaign_name"] = campaign.name
                metrics["campaign_type"] = campaign.campaign_type
                metrics["source"] = campaign.source
                metrics["medium"] = campaign.medium
                metrics["budget"] = campaign.budget
                
                # Calculate ROI
                if campaign.budget > 0:
                    metrics["roi"] = (metrics["attributed_revenue"] - campaign.budget) / campaign.budget * 100
                    metrics["roas"] = metrics["attributed_revenue"] / campaign.budget
                else:
                    metrics["roi"] = 0
                    metrics["roas"] = 0
        
        return campaign_metrics
    
    async def _calculate_channel_performance(self,
                                           conversions: List[Conversion],
                                           model: AttributionModel) -> Dict[str, Dict[str, Any]]:
        """Calculate performance metrics for each channel (source/medium)"""
        channel_metrics = {}
        
        for conversion in conversions:
            # Get touchpoints for this conversion
            conversion_touchpoints = [
                tp for tp in self.touchpoints
                if tp.touchpoint_id in conversion.attributed_touchpoints
            ]
            
            for touchpoint in conversion_touchpoints:
                channel_key = f"{touchpoint.source}/{touchpoint.medium}"
                attribution_weight = conversion.attribution_weights.get(touchpoint.touchpoint_id, 0)
                
                if channel_key not in channel_metrics:
                    channel_metrics[channel_key] = {
                        "touchpoints": 0,
                        "conversions": 0,
                        "attributed_conversions": 0.0,
                        "revenue": 0.0,
                        "attributed_revenue": 0.0
                    }
                
                channel_metrics[channel_key]["touchpoints"] += 1
                channel_metrics[channel_key]["conversions"] += 1
                channel_metrics[channel_key]["attributed_conversions"] += attribution_weight
                channel_metrics[channel_key]["revenue"] += conversion.conversion_value
                channel_metrics[channel_key]["attributed_revenue"] += conversion.conversion_value * attribution_weight
        
        return channel_metrics
    
    async def _analyze_conversion_paths(self, conversions: List[Conversion]) -> List[Dict[str, Any]]:
        """Analyze common conversion paths"""
        path_analysis = {}
        
        for conversion in conversions:
            # Get touchpoints for this conversion
            conversion_touchpoints = [
                tp for tp in self.touchpoints
                if tp.touchpoint_id in conversion.attributed_touchpoints
            ]
            
            # Sort by timestamp
            conversion_touchpoints.sort(key=lambda x: x.timestamp)
            
            # Create path string
            path = " -> ".join([f"{tp.source}/{tp.medium}" for tp in conversion_touchpoints])
            
            if path not in path_analysis:
                path_analysis[path] = {
                    "path": path,
                    "conversions": 0,
                    "total_value": 0.0,
                    "avg_value": 0.0,
                    "touchpoint_count": len(conversion_touchpoints)
                }
            
            path_analysis[path]["conversions"] += 1
            path_analysis[path]["total_value"] += conversion.conversion_value
            path_analysis[path]["avg_value"] = path_analysis[path]["total_value"] / path_analysis[path]["conversions"]
        
        # Sort by conversion count
        return sorted(path_analysis.values(), key=lambda x: x["conversions"], reverse=True)[:20]
    
    async def _calculate_roi_analysis(self, campaign_performance: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate overall ROI analysis"""
        total_spend = sum(metrics.get("budget", 0) for metrics in campaign_performance.values())
        total_revenue = sum(metrics.get("attributed_revenue", 0) for metrics in campaign_performance.values())
        total_conversions = sum(metrics.get("attributed_conversions", 0) for metrics in campaign_performance.values())
        
        roi = ((total_revenue - total_spend) / total_spend * 100) if total_spend > 0 else 0
        roas = (total_revenue / total_spend) if total_spend > 0 else 0
        
        return {
            "total_spend": total_spend,
            "total_revenue": total_revenue,
            "total_conversions": total_conversions,
            "overall_roi": roi,
            "overall_roas": roas,
            "cost_per_conversion": (total_spend / total_conversions) if total_conversions > 0 else 0,
            "revenue_per_conversion": (total_revenue / total_conversions) if total_conversions > 0 else 0
        }
    
    async def get_marketing_dashboard(self, days: int = 30) -> Dict[str, Any]:
        """Get marketing attribution dashboard data"""
        try:
            # Generate report for dashboard
            report = await self.generate_attribution_report(AttributionModel.LAST_TOUCH, days)
            
            # Get active campaigns
            active_campaigns = [c for c in self.campaigns.values() if c.status == CampaignStatus.ACTIVE]
            
            # Get recent touchpoints
            recent_touchpoints = [
                tp for tp in self.touchpoints
                if tp.timestamp >= datetime.utcnow() - timedelta(days=days)
            ]
            
            return {
                "active_campaigns": len(active_campaigns),
                "total_touchpoints": len(recent_touchpoints),
                "total_conversions": len([c for c in self.conversions if c.timestamp >= datetime.utcnow() - timedelta(days=days)]),
                "roi_analysis": report.roi_analysis,
                "top_campaigns": sorted(
                    report.campaign_performance.items(),
                    key=lambda x: x[1].get("attributed_revenue", 0),
                    reverse=True
                )[:5],
                "top_channels": sorted(
                    report.channel_performance.items(),
                    key=lambda x: x[1].get("attributed_revenue", 0),
                    reverse=True
                )[:5],
                "top_conversion_paths": report.conversion_paths[:5]
            }
            
        except Exception as e:
            logger.error(f"Error getting marketing dashboard: {str(e)}")
            raise

# Global marketing attribution tracker instance
marketing_attribution = MarketingAttributionTracker()