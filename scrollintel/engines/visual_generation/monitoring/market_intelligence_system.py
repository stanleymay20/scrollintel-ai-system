"""
Market Intelligence System for Visual Generation Dominance Validation

This module implements automated testing against competitor platforms, quality superiority
measurement and reporting, performance benchmdevelopments and maintain
competitive advantage.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

import aiohttp
import feedparser
import numpy as np
from bs4 import BeautifulSoup
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, desc

from scrollintel.core.config import get_settings
from scrollintel.models.competitive_analysis_models import (
    MarketIntelligence, CompetitorUpdate, StrategicRecommendation,
    CompetitiveAlert
)

logger = logging.getLogger(__name__)
settings = get_settings()


class IntelligenceSource(Enum):
    """Sources for market intelligence gathering."""
    TECH_NEWS = "tech_news"
    RESEARCH_PAPERS = "research_papers"
    PATENT_FILINGS = "patent_filings"
    SOCIAL_MEDIA = "social_media"
    COMPETITOR_BLOGS = "competitor_blogs"
    INDUSTRY_REPORTS = "industry_reports"
    CONFERENCE_PROCEEDINGS = "conference_proceedings"
    GITHUB_REPOSITORIES = "github_repositories"


@dataclass
class IntelligenceItem:
    """Individual intelligence item from various sources."""
    source: str
    title: str
    content: str
    url: str
    published_date: datetime
    relevance_score: float
    sentiment: str
    keywords: List[str]
    impact_assessment: str


@dataclass
class TrendAnalysis:
    """Analysis of market trends and patterns."""
    trend_name: str
    trend_strength: float  # 0-1
    growth_trajectory: str  # emerging, growing, mature, declining
    market_impact: str  # low, medium, high
    time_horizon: str  # short, medium, long
    supporting_evidence: List[str]
    confidence_level: float  # 0-1


@dataclass
class CompetitorIntelligence:
    """Intelligence about specific competitors."""
    competitor_name: str
    recent_updates: List[Dict[str, Any]]
    strategic_moves: List[str]
    technology_developments: List[str]
    market_positioning: str
    threat_level: float  # 0-1
    opportunity_areas: List[str]


class MarketIntelligenceSystem:
    """
    Advanced market intelligence system for comprehensive industry monitoring
    and competitive analysis.
    """
    
    def __init__(self, db_session: AsyncSession):
        self.db_session = db_session
        self.intelligence_sources = self._initialize_intelligence_sources()
        self.trend_analyzer = TrendAnalyzer()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.threat_assessor = ThreatAssessor()
        
    def _initialize_intelligence_sources(self) -> Dict[str, Dict[str, Any]]:
        """Initialize intelligence gathering sources."""
        return {
            IntelligenceSource.TECH_NEWS.value: {
                "sources": [
                    "https://techcrunch.com/feed/",
                    "https://venturebeat.com/feed/",
                    "https://www.theverge.com/rss/index.xml",
                    "https://arstechnica.com/feed/",
                    "https://www.wired.com/feed/"
                ],
                "keywords": ["AI", "artificial intelligence", "machine learning", "computer vision", "generative AI"],
                "update_frequency": 3600  # seconds
            },
            IntelligenceSource.RESEARCH_PAPERS.value: {
                "sources": [
                    "https://arxiv.org/rss/cs.CV",  # Computer Vision
                    "https://arxiv.org/rss/cs.AI",  # Artificial Intelligence
                    "https://arxiv.org/rss/cs.LG",  # Machine Learning
                    "https://arxiv.org/rss/cs.GR"   # Graphics
                ],
                "keywords": ["diffusion models", "neural rendering", "image generation", "video synthesis"],
                "update_frequency": 7200
            },
            IntelligenceSource.COMPETITOR_BLOGS.value: {
                "sources": [
                    "https://openai.com/blog/rss.xml",
                    "https://blog.midjourney.com/feed/",
                    "https://stability.ai/blog/feed/",
                    "https://runwayml.com/blog/feed/"
                ],
                "keywords": ["product update", "new feature", "model release", "pricing"],
                "update_frequency": 1800
            },
            IntelligenceSource.GITHUB_REPOSITORIES.value: {
                "sources": [
                    "https://github.com/CompVis/stable-diffusion",
                    "https://github.com/openai/DALL-E",
                    "https://github.com/runwayml/stable-diffusion"
                ],
                "keywords": ["commit", "release", "update", "feature"],
                "update_frequency": 3600
            }
        }
    
    async def gather_comprehensive_intelligence(self) -> Dict[str, Any]:
        """
        Gather comprehensive market intelligence from all sources.
        
        Returns:
            Dict containing comprehensive intelligence report
        """
        try:
            logger.info("Starting comprehensive market intelligence gathering")
            
            # Gather intelligence from all sources
            intelligence_items = await self._gather_from_all_sources()
            
            # Analyze trends and patterns
            trend_analysis = await self._analyze_market_trends(intelligence_items)
            
            # Assess competitor intelligence
            competitor_intelligence = await self._analyze_competitor_intelligence(intelligence_items)
            
            # Identify market opportunities
            market_opportunities = await self._identify_market_opportunities(
                intelligence_items, trend_analysis
            )
            
            # Assess threats and risks
            threat_assessment = await self._assess_threats_and_risks(
                intelligence_items, competitor_intelligence
            )
            
            # Generate strategic recommendations
            strategic_recommendations = await self._generate_strategic_recommendations(
                trend_analysis, competitor_intelligence, market_opportunities, threat_assessment
            )
            
            # Create comprehensive report
            intelligence_report = {
                "report_timestamp": datetime.utcnow().isoformat(),
                "intelligence_summary": {
                    "total_items_analyzed": len(intelligence_items),
                    "sources_monitored": len(self.intelligence_sources),
                    "trends_identified": len(trend_analysis),
                    "competitors_analyzed": len(competitor_intelligence),
                    "opportunities_found": len(market_opportunities),
                    "threats_assessed": len(threat_assessment)
                },
                "trend_analysis": trend_analysis,
                "competitor_intelligence": competitor_intelligence,
                "market_opportunities": market_opportunities,
                "threat_assessment": threat_assessment,
                "strategic_recommendations": strategic_recommendations,
                "intelligence_items": intelligence_items[:50],  # Top 50 most relevant
                "next_update": (datetime.utcnow() + timedelta(hours=6)).isoformat()
            }
            
            # Store intelligence report
            await self._store_intelligence_report(intelligence_report)
            
            # Generate alerts if necessary
            await self._generate_competitive_alerts(intelligence_report)
            
            logger.info("Market intelligence gathering completed successfully")
            return intelligence_report
            
        except Exception as e:
            logger.error(f"Error in market intelligence gathering: {str(e)}")
            raise
    
    async def _gather_from_all_sources(self) -> List[IntelligenceItem]:
        """Gather intelligence from all configured sources."""
        all_items = []
        
        for source_type, config in self.intelligence_sources.items():
            try:
                items = await self._gather_from_source_type(source_type, config)
                all_items.extend(items)
                
                # Add delay between source types
                await asyncio.sleep(1.0)
                
            except Exception as e:
                logger.warning(f"Failed to gather from {source_type}: {str(e)}")
        
        # Sort by relevance and recency
        all_items.sort(key=lambda x: (x.relevance_score, x.published_date), reverse=True)
        
        return all_items
    
    async def _gather_from_source_type(
        self, 
        source_type: str, 
        config: Dict[str, Any]
    ) -> List[IntelligenceItem]:
        """Gather intelligence from a specific source type."""
        items = []
        
        if source_type == IntelligenceSource.TECH_NEWS.value:
            items = await self._gather_from_rss_feeds(config)
        elif source_type == IntelligenceSource.RESEARCH_PAPERS.value:
            items = await self._gather_from_arxiv(config)
        elif source_type == IntelligenceSource.COMPETITOR_BLOGS.value:
            items = await self._gather_from_competitor_blogs(config)
        elif source_type == IntelligenceSource.GITHUB_REPOSITORIES.value:
            items = await self._gather_from_github(config)
        
        return items
    
    async def _gather_from_rss_feeds(self, config: Dict[str, Any]) -> List[IntelligenceItem]:
        """Gather intelligence from RSS feeds."""
        items = []
        keywords = config["keywords"]
        
        async with aiohttp.ClientSession() as session:
            for feed_url in config["sources"]:
                try:
                    async with session.get(feed_url) as response:
                        if response.status == 200:
                            feed_content = await response.text()
                            feed = feedparser.parse(feed_content)
                            
                            for entry in feed.entries[:10]:  # Latest 10 entries
                                # Check relevance based on keywords
                                relevance = self._calculate_relevance(
                                    entry.title + " " + entry.get('summary', ''), 
                                    keywords
                                )
                                
                                if relevance > 0.3:  # Minimum relevance threshold
                                    item = IntelligenceItem(
                                        source=feed_url,
                                        title=entry.title,
                                        content=entry.get('summary', ''),
                                        url=entry.link,
                                        published_date=self._parse_date(entry.get('published')),
                                        relevance_score=relevance,
                                        sentiment=await self.sentiment_analyzer.analyze(entry.title),
                                        keywords=self._extract_keywords(entry.title + " " + entry.get('summary', '')),
                                        impact_assessment=self._assess_impact(relevance)
                                    )
                                    items.append(item)
                    
                    await asyncio.sleep(0.5)  # Rate limiting
                    
                except Exception as e:
                    logger.warning(f"Failed to gather from RSS feed {feed_url}: {str(e)}")
        
        return items
    
    async def _gather_from_arxiv(self, config: Dict[str, Any]) -> List[IntelligenceItem]:
        """Gather intelligence from arXiv research papers."""
        items = []
        keywords = config["keywords"]
        
        async with aiohttp.ClientSession() as session:
            for feed_url in config["sources"]:
                try:
                    async with session.get(feed_url) as response:
                        if response.status == 200:
                            feed_content = await response.text()
                            feed = feedparser.parse(feed_content)
                            
                            for entry in feed.entries[:5]:  # Latest 5 papers
                                relevance = self._calculate_relevance(
                                    entry.title + " " + entry.get('summary', ''), 
                                    keywords
                                )
                                
                                if relevance > 0.4:  # Higher threshold for research papers
                                    item = IntelligenceItem(
                                        source=feed_url,
                                        title=entry.title,
                                        content=entry.get('summary', ''),
                                        url=entry.link,
                                        published_date=self._parse_date(entry.get('published')),
                                        relevance_score=relevance,
                                        sentiment="neutral",  # Research papers are typically neutral
                                        keywords=self._extract_keywords(entry.title),
                                        impact_assessment="high"  # Research has high potential impact
                                    )
                                    items.append(item)
                    
                    await asyncio.sleep(1.0)  # Longer delay for arXiv
                    
                except Exception as e:
                    logger.warning(f"Failed to gather from arXiv {feed_url}: {str(e)}")
        
        return items
    
    async def _gather_from_competitor_blogs(self, config: Dict[str, Any]) -> List[IntelligenceItem]:
        """Gather intelligence from competitor blogs."""
        items = []
        keywords = config["keywords"]
        
        async with aiohttp.ClientSession() as session:
            for blog_url in config["sources"]:
                try:
                    async with session.get(blog_url) as response:
                        if response.status == 200:
                            feed_content = await response.text()
                            feed = feedparser.parse(feed_content)
                            
                            for entry in feed.entries[:5]:  # Latest 5 posts
                                relevance = self._calculate_relevance(
                                    entry.title + " " + entry.get('summary', ''), 
                                    keywords
                                )
                                
                                if relevance > 0.2:  # Lower threshold for competitor updates
                                    item = IntelligenceItem(
                                        source=blog_url,
                                        title=entry.title,
                                        content=entry.get('summary', ''),
                                        url=entry.link,
                                        published_date=self._parse_date(entry.get('published')),
                                        relevance_score=relevance,
                                        sentiment=await self.sentiment_analyzer.analyze(entry.title),
                                        keywords=self._extract_keywords(entry.title),
                                        impact_assessment="high"  # Competitor updates are important
                                    )
                                    items.append(item)
                    
                    await asyncio.sleep(0.5)
                    
                except Exception as e:
                    logger.warning(f"Failed to gather from competitor blog {blog_url}: {str(e)}")
        
        return items
    
    async def _gather_from_github(self, config: Dict[str, Any]) -> List[IntelligenceItem]:
        """Gather intelligence from GitHub repositories."""
        items = []
        
        # This would require GitHub API integration
        # For now, return simulated data
        simulated_items = [
            IntelligenceItem(
                source="github",
                title="Stable Diffusion XL 1.0 Release",
                content="Major update with improved quality and performance",
                url="https://github.com/Stability-AI/generative-models",
                published_date=datetime.utcnow() - timedelta(days=1),
                relevance_score=0.9,
                sentiment="positive",
                keywords=["stable diffusion", "xl", "release", "improvement"],
                impact_assessment="high"
            )
        ]
        
        return simulated_items
    
    def _calculate_relevance(self, text: str, keywords: List[str]) -> float:
        """Calculate relevance score based on keyword matching."""
        text_lower = text.lower()
        matches = sum(1 for keyword in keywords if keyword.lower() in text_lower)
        return min(matches / len(keywords), 1.0)
    
    def _parse_date(self, date_str: Optional[str]) -> datetime:
        """Parse date string to datetime object."""
        if not date_str:
            return datetime.utcnow()
        
        try:
            # Try common date formats
            from dateutil import parser
            return parser.parse(date_str)
        except:
            return datetime.utcnow()
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text."""
        # Simple keyword extraction - in production would use NLP
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        words = text.lower().split()
        keywords = [word for word in words if len(word) > 3 and word not in common_words]
        return keywords[:10]  # Top 10 keywords
    
    def _assess_impact(self, relevance_score: float) -> str:
        """Assess impact level based on relevance score."""
        if relevance_score >= 0.8:
            return "high"
        elif relevance_score >= 0.5:
            return "medium"
        else:
            return "low"
    
    async def _analyze_market_trends(self, intelligence_items: List[IntelligenceItem]) -> List[TrendAnalysis]:
        """Analyze market trends from intelligence items."""
        return await self.trend_analyzer.analyze_trends(intelligence_items)
    
    async def _analyze_competitor_intelligence(
        self, 
        intelligence_items: List[IntelligenceItem]
    ) -> List[CompetitorIntelligence]:
        """Analyze competitor-specific intelligence."""
        competitors = ["midjourney", "dalle3", "runway_ml", "pika_labs", "stable_diffusion"]
        competitor_intel = []
        
        for competitor in competitors:
            # Filter items related to this competitor
            competitor_items = [
                item for item in intelligence_items 
                if competitor.lower() in item.title.lower() or 
                   competitor.lower() in item.content.lower()
            ]
            
            if competitor_items:
                intel = CompetitorIntelligence(
                    competitor_name=competitor,
                    recent_updates=[
                        {
                            "title": item.title,
                            "date": item.published_date.isoformat(),
                            "impact": item.impact_assessment
                        }
                        for item in competitor_items[:5]
                    ],
                    strategic_moves=self._extract_strategic_moves(competitor_items),
                    technology_developments=self._extract_tech_developments(competitor_items),
                    market_positioning=self._assess_market_positioning(competitor),
                    threat_level=await self.threat_assessor.assess_competitor_threat(competitor, competitor_items),
                    opportunity_areas=self._identify_opportunity_areas(competitor_items)
                )
                competitor_intel.append(intel)
        
        return competitor_intel
    
    def _extract_strategic_moves(self, items: List[IntelligenceItem]) -> List[str]:
        """Extract strategic moves from intelligence items."""
        strategic_keywords = ["partnership", "acquisition", "funding", "expansion", "launch"]
        moves = []
        
        for item in items:
            for keyword in strategic_keywords:
                if keyword in item.title.lower() or keyword in item.content.lower():
                    moves.append(f"{keyword.title()}: {item.title}")
                    break
        
        return moves[:5]  # Top 5 strategic moves
    
    def _extract_tech_developments(self, items: List[IntelligenceItem]) -> List[str]:
        """Extract technology developments from intelligence items."""
        tech_keywords = ["model", "algorithm", "breakthrough", "innovation", "improvement"]
        developments = []
        
        for item in items:
            for keyword in tech_keywords:
                if keyword in item.title.lower() or keyword in item.content.lower():
                    developments.append(f"{keyword.title()}: {item.title}")
                    break
        
        return developments[:5]  # Top 5 tech developments
    
    def _assess_market_positioning(self, competitor: str) -> str:
        """Assess competitor's market positioning."""
        positioning_map = {
            "midjourney": "Creative Community Leader",
            "dalle3": "Enterprise Integration Focus",
            "runway_ml": "Video Generation Specialist",
            "pika_labs": "Emerging Video Player",
            "stable_diffusion": "Open Source Champion"
        }
        return positioning_map.get(competitor, "Niche Player")
    
    def _identify_opportunity_areas(self, items: List[IntelligenceItem]) -> List[str]:
        """Identify opportunity areas from competitor intelligence."""
        opportunity_keywords = ["limitation", "challenge", "gap", "missing", "need"]
        opportunities = []
        
        for item in items:
            for keyword in opportunity_keywords:
                if keyword in item.content.lower():
                    opportunities.append(f"Opportunity in: {item.title}")
                    break
        
        return opportunities[:3]  # Top 3 opportunities
    
    async def _identify_market_opportunities(
        self, 
        intelligence_items: List[IntelligenceItem],
        trend_analysis: List[TrendAnalysis]
    ) -> List[Dict[str, Any]]:
        """Identify market opportunities from intelligence and trends."""
        opportunities = []
        
        # Opportunities from trends
        for trend in trend_analysis:
            if trend.growth_trajectory in ["emerging", "growing"] and trend.market_impact == "high":
                opportunities.append({
                    "type": "trend_opportunity",
                    "title": f"Capitalize on {trend.trend_name}",
                    "description": f"Growing trend with {trend.trend_strength:.1%} strength",
                    "priority": "high" if trend.confidence_level > 0.8 else "medium",
                    "time_horizon": trend.time_horizon,
                    "supporting_evidence": trend.supporting_evidence[:3]
                })
        
        # Opportunities from competitor gaps
        gap_keywords = ["expensive", "slow", "limited", "complex", "difficult"]
        for item in intelligence_items:
            for keyword in gap_keywords:
                if keyword in item.content.lower() and item.relevance_score > 0.6:
                    opportunities.append({
                        "type": "competitor_gap",
                        "title": f"Address {keyword} issue in market",
                        "description": item.title,
                        "priority": "medium",
                        "time_horizon": "short",
                        "supporting_evidence": [item.url]
                    })
                    break
        
        return opportunities[:10]  # Top 10 opportunities
    
    async def _assess_threats_and_risks(
        self,
        intelligence_items: List[IntelligenceItem],
        competitor_intelligence: List[CompetitorIntelligence]
    ) -> Dict[str, float]:
        """Assess threats and risks from market intelligence."""
        threats = {
            "new_entrants": 0.0,
            "price_competition": 0.0,
            "technology_disruption": 0.0,
            "regulatory_changes": 0.0,
            "market_saturation": 0.0
        }
        
        # Analyze threats from intelligence items
        threat_keywords = {
            "new_entrants": ["startup", "new player", "enters market", "launches"],
            "price_competition": ["price cut", "free", "cheaper", "cost reduction"],
            "technology_disruption": ["breakthrough", "revolutionary", "game changer"],
            "regulatory_changes": ["regulation", "policy", "compliance", "legal"],
            "market_saturation": ["crowded", "saturated", "competition", "many players"]
        }
        
        for threat_type, keywords in threat_keywords.items():
            threat_score = 0.0
            relevant_items = 0
            
            for item in intelligence_items:
                for keyword in keywords:
                    if keyword in item.content.lower():
                        threat_score += item.relevance_score
                        relevant_items += 1
                        break
            
            if relevant_items > 0:
                threats[threat_type] = min(threat_score / relevant_items, 1.0)
        
        # Factor in competitor threat levels
        avg_competitor_threat = np.mean([comp.threat_level for comp in competitor_intelligence])
        threats["competitive_pressure"] = avg_competitor_threat
        
        return threats
    
    async def _generate_strategic_recommendations(
        self,
        trend_analysis: List[TrendAnalysis],
        competitor_intelligence: List[CompetitorIntelligence],
        market_opportunities: List[Dict[str, Any]],
        threat_assessment: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """Generate strategic recommendations based on intelligence analysis."""
        recommendations = []
        
        # Recommendations based on trends
        for trend in trend_analysis:
            if trend.growth_trajectory == "emerging" and trend.confidence_level > 0.7:
                recommendations.append({
                    "type": "trend_investment",
                    "priority": "high",
                    "title": f"Invest in {trend.trend_name}",
                    "description": f"Emerging trend with high confidence ({trend.confidence_level:.1%})",
                    "expected_impact": "Market leadership in emerging area",
                    "timeline": trend.time_horizon,
                    "implementation_effort": "medium"
                })
        
        # Recommendations based on competitor gaps
        for comp in competitor_intelligence:
            if comp.threat_level < 0.3:  # Low threat competitors
                recommendations.append({
                    "type": "competitive_advantage",
                    "priority": "medium",
                    "title": f"Maintain advantage over {comp.competitor_name}",
                    "description": f"Low threat level ({comp.threat_level:.1%}) indicates strong position",
                    "expected_impact": "Sustained competitive advantage",
                    "timeline": "ongoing",
                    "implementation_effort": "low"
                })
        
        # Recommendations based on opportunities
        high_priority_opportunities = [opp for opp in market_opportunities if opp["priority"] == "high"]
        for opp in high_priority_opportunities[:3]:  # Top 3 high-priority opportunities
            recommendations.append({
                "type": "market_opportunity",
                "priority": "high",
                "title": opp["title"],
                "description": opp["description"],
                "expected_impact": "Market expansion and revenue growth",
                "timeline": opp["time_horizon"],
                "implementation_effort": "medium"
            })
        
        # Recommendations based on threats
        high_threats = {k: v for k, v in threat_assessment.items() if v > 0.6}
        for threat_type, threat_level in high_threats.items():
            recommendations.append({
                "type": "threat_mitigation",
                "priority": "high",
                "title": f"Mitigate {threat_type.replace('_', ' ')} risk",
                "description": f"High threat level ({threat_level:.1%}) requires immediate attention",
                "expected_impact": "Risk reduction and market stability",
                "timeline": "short",
                "implementation_effort": "high"
            })
        
        return recommendations
    
    async def _store_intelligence_report(self, intelligence_report: Dict[str, Any]) -> None:
        """Store intelligence report in database."""
        try:
            # Store main intelligence report
            intelligence_record = MarketIntelligence(
                industry_trends=json.dumps([trend.__dict__ for trend in intelligence_report["trend_analysis"]]),
                emerging_technologies=json.dumps([trend.trend_name for trend in intelligence_report["trend_analysis"] if trend.growth_trajectory == "emerging"]),
                competitor_updates=json.dumps([comp.__dict__ for comp in intelligence_report["competitor_intelligence"]]),
                market_opportunities=json.dumps(intelligence_report["market_opportunities"]),
                threat_assessment=json.dumps(intelligence_report["threat_assessment"]),
                recommendation_priority="high",
                report_timestamp=datetime.utcnow()
            )
            self.db_session.add(intelligence_record)
            
            # Store strategic recommendations
            for rec in intelligence_report["strategic_recommendations"]:
                rec_record = StrategicRecommendation(
                    recommendation_type=rec["type"],
                    priority_level=rec["priority"],
                    recommendation_title=rec["title"],
                    recommendation_description=rec["description"],
                    supporting_analysis=json.dumps(rec),
                    expected_impact=rec.get("expected_impact"),
                    implementation_effort=rec.get("implementation_effort"),
                    timeline_estimate=rec.get("timeline"),
                    status="pending"
                )
                self.db_session.add(rec_record)
            
            await self.db_session.commit()
            logger.info("Intelligence report stored successfully")
            
        except Exception as e:
            logger.error(f"Failed to store intelligence report: {str(e)}")
            await self.db_session.rollback()
            raise
    
    async def _generate_competitive_alerts(self, intelligence_report: Dict[str, Any]) -> None:
        """Generate competitive alerts based on intelligence report."""
        try:
            # High threat alerts
            for threat_type, threat_level in intelligence_report["threat_assessment"].items():
                if threat_level > 0.7:  # High threat threshold
                    alert = CompetitiveAlert(
                        alert_type="high_threat",
                        severity_level="critical",
                        alert_title=f"High {threat_type.replace('_', ' ')} threat detected",
                        alert_description=f"Threat level: {threat_level:.1%}. Immediate attention required.",
                        impact_assessment="High impact on competitive position",
                        recommended_actions=json.dumps([
                            "Assess immediate response options",
                            "Develop mitigation strategy",
                            "Monitor threat evolution"
                        ])
                    )
                    self.db_session.add(alert)
            
            # High-priority opportunity alerts
            high_priority_opportunities = [
                opp for opp in intelligence_report["market_opportunities"] 
                if opp["priority"] == "high"
            ]
            
            if high_priority_opportunities:
                alert = CompetitiveAlert(
                    alert_type="market_opportunity",
                    severity_level="warning",
                    alert_title=f"{len(high_priority_opportunities)} high-priority opportunities identified",
                    alert_description="Multiple high-value market opportunities detected",
                    impact_assessment="Potential for significant market expansion",
                    recommended_actions=json.dumps([
                        "Evaluate opportunity feasibility",
                        "Develop capture strategy",
                        "Allocate resources for implementation"
                    ])
                )
                self.db_session.add(alert)
            
            await self.db_session.commit()
            logger.info("Competitive alerts generated successfully")
            
        except Exception as e:
            logger.error(f"Failed to generate competitive alerts: {str(e)}")
            await self.db_session.rollback()


class TrendAnalyzer:
    """Analyzer for market trends and patterns."""
    
    async def analyze_trends(self, intelligence_items: List[IntelligenceItem]) -> List[TrendAnalysis]:
        """Analyze trends from intelligence items."""
        # Group items by keywords to identify trends
        keyword_groups = {}
        for item in intelligence_items:
            for keyword in item.keywords:
                if keyword not in keyword_groups:
                    keyword_groups[keyword] = []
                keyword_groups[keyword].append(item)
        
        trends = []
        for keyword, items in keyword_groups.items():
            if len(items) >= 3:  # Minimum items to constitute a trend
                trend_strength = len(items) / len(intelligence_items)
                
                # Analyze growth trajectory
                recent_items = [item for item in items if item.published_date > datetime.utcnow() - timedelta(days=30)]
                growth_trajectory = "growing" if len(recent_items) > len(items) / 2 else "stable"
                
                # Assess market impact
                avg_relevance = np.mean([item.relevance_score for item in items])
                market_impact = "high" if avg_relevance > 0.7 else "medium" if avg_relevance > 0.4 else "low"
                
                trend = TrendAnalysis(
                    trend_name=keyword,
                    trend_strength=trend_strength,
                    growth_trajectory=growth_trajectory,
                    market_impact=market_impact,
                    time_horizon="short" if len(recent_items) > 5 else "medium",
                    supporting_evidence=[item.title for item in items[:3]],
                    confidence_level=min(trend_strength * 2, 1.0)
                )
                trends.append(trend)
        
        # Sort by trend strength
        trends.sort(key=lambda x: x.trend_strength, reverse=True)
        return trends[:10]  # Top 10 trends


class SentimentAnalyzer:
    """Simple sentiment analyzer for intelligence items."""
    
    async def analyze(self, text: str) -> str:
        """Analyze sentiment of text."""
        # Simple keyword-based sentiment analysis
        positive_words = ["breakthrough", "success", "improvement", "innovation", "growth"]
        negative_words = ["failure", "decline", "problem", "issue", "challenge"]
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            return "positive"
        elif negative_count > positive_count:
            return "negative"
        else:
            return "neutral"


class ThreatAssessor:
    """Assessor for competitive threats."""
    
    async def assess_competitor_threat(
        self, 
        competitor: str, 
        intelligence_items: List[IntelligenceItem]
    ) -> float:
        """Assess threat level from a specific competitor."""
        if not intelligence_items:
            return 0.3  # Default moderate threat
        
        # Factors that increase threat level
        threat_keywords = ["funding", "breakthrough", "partnership", "expansion", "improvement"]
        threat_score = 0.0
        
        for item in intelligence_items:
            item_threat = 0.0
            for keyword in threat_keywords:
                if keyword in item.content.lower():
                    item_threat += 0.2
            
            # Factor in recency (more recent = higher threat)
            days_old = (datetime.utcnow() - item.published_date).days
            recency_factor = max(0.1, 1.0 - (days_old / 30))  # Decay over 30 days
            
            threat_score += item_threat * recency_factor * item.relevance_score
        
        # Normalize threat score
        normalized_threat = min(threat_score / len(intelligence_items), 1.0)
        
        return normalized_threat