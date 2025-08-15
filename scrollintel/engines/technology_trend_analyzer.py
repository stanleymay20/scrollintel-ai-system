"""
Technology Trend Analysis Engine using patent databases and research papers
"""
import asyncio
import json
import re
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Tuple
import logging
from dataclasses import asdict
import aiohttp
from collections import defaultdict, Counter

from ..models.breakthrough_models import TechnologyTrend, TechnologyDomain


class TechnologyTrendAnalyzer:
    """
    Advanced technology trend analysis using patent databases and research papers
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.patent_apis = self._initialize_patent_apis()
        self.research_apis = self._initialize_research_apis()
        self.trend_cache = {}
        self.analysis_cache = {}
        
    def _initialize_patent_apis(self) -> Dict[str, Dict[str, str]]:
        """Initialize patent database API configurations"""
        return {
            'uspto': {
                'base_url': 'https://api.uspto.gov/ds-api',
                'search_endpoint': '/search/patents',
                'api_key': 'uspto_api_key',
                'rate_limit': 100  # requests per minute
            },
            'epo': {
                'base_url': 'https://ops.epo.org/3.2',
                'search_endpoint': '/rest-services/published-data/search',
                'api_key': 'epo_api_key',
                'rate_limit': 50
            },
            'google_patents': {
                'base_url': 'https://patents.google.com/api',
                'search_endpoint': '/search',
                'api_key': 'google_patents_key',
                'rate_limit': 200
            },
            'wipo': {
                'base_url': 'https://patentscope.wipo.int/search/rest',
                'search_endpoint': '/patents',
                'api_key': 'wipo_api_key',
                'rate_limit': 30
            }
        }
    
    def _initialize_research_apis(self) -> Dict[str, Dict[str, str]]:
        """Initialize research paper API configurations"""
        return {
            'arxiv': {
                'base_url': 'http://export.arxiv.org/api',
                'search_endpoint': '/query',
                'api_key': None,
                'rate_limit': 300
            },
            'pubmed': {
                'base_url': 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils',
                'search_endpoint': '/esearch.fcgi',
                'api_key': 'pubmed_api_key',
                'rate_limit': 100
            },
            'ieee': {
                'base_url': 'https://ieeexploreapi.ieee.org/api/v1',
                'search_endpoint': '/search/articles',
                'api_key': 'ieee_api_key',
                'rate_limit': 200
            },
            'semantic_scholar': {
                'base_url': 'https://api.semanticscholar.org/graph/v1',
                'search_endpoint': '/paper/search',
                'api_key': 'semantic_scholar_key',
                'rate_limit': 100
            },
            'crossref': {
                'base_url': 'https://api.crossref.org',
                'search_endpoint': '/works',
                'api_key': None,
                'rate_limit': 50
            }
        }

    async def analyze_technology_trends(
        self, 
        domain: TechnologyDomain,
        timeframe_years: int = 5,
        depth: str = 'comprehensive'
    ) -> List[TechnologyTrend]:
        """
        Analyze technology trends in a specific domain using multiple data sources
        """
        self.logger.info(f"Analyzing technology trends for {domain.value}")
        
        # Check cache first
        cache_key = f"{domain.value}_{timeframe_years}_{depth}"
        if cache_key in self.trend_cache:
            cache_time, cached_trends = self.trend_cache[cache_key]
            if datetime.now() - cache_time < timedelta(hours=6):  # 6-hour cache
                return cached_trends
        
        # Parallel analysis from multiple sources
        patent_analysis_task = self._analyze_patent_trends(domain, timeframe_years)
        research_analysis_task = self._analyze_research_trends(domain, timeframe_years)
        investment_analysis_task = self._analyze_investment_trends(domain, timeframe_years)
        
        patent_trends, research_trends, investment_data = await asyncio.gather(
            patent_analysis_task,
            research_analysis_task,
            investment_analysis_task
        )
        
        # Synthesize trends from all sources
        synthesized_trends = await self._synthesize_trend_data(
            patent_trends, research_trends, investment_data, domain
        )
        
        # Calculate momentum and predictions
        final_trends = []
        for trend_data in synthesized_trends:
            trend = await self._calculate_trend_metrics(trend_data, domain)
            final_trends.append(trend)
        
        # Cache results
        self.trend_cache[cache_key] = (datetime.now(), final_trends)
        
        return final_trends

    async def _analyze_patent_trends(
        self, 
        domain: TechnologyDomain, 
        timeframe_years: int
    ) -> Dict[str, Any]:
        """
        Analyze patent trends across multiple patent databases
        """
        self.logger.info(f"Analyzing patent trends for {domain.value}")
        
        # Define domain-specific search terms
        search_terms = self._get_domain_search_terms(domain)
        
        # Query multiple patent databases
        patent_data = {}
        for db_name, api_config in self.patent_apis.items():
            try:
                db_results = await self._query_patent_database(
                    db_name, api_config, search_terms, timeframe_years
                )
                patent_data[db_name] = db_results
            except Exception as e:
                self.logger.error(f"Error querying {db_name}: {e}")
                patent_data[db_name] = {'patents': [], 'total': 0}
        
        # Analyze patent trends
        trend_analysis = await self._analyze_patent_data(patent_data, domain)
        
        return trend_analysis

    async def _analyze_research_trends(
        self, 
        domain: TechnologyDomain, 
        timeframe_years: int
    ) -> Dict[str, Any]:
        """
        Analyze research paper trends across multiple academic databases
        """
        self.logger.info(f"Analyzing research trends for {domain.value}")
        
        search_terms = self._get_domain_search_terms(domain)
        
        # Query multiple research databases
        research_data = {}
        for db_name, api_config in self.research_apis.items():
            try:
                db_results = await self._query_research_database(
                    db_name, api_config, search_terms, timeframe_years
                )
                research_data[db_name] = db_results
            except Exception as e:
                self.logger.error(f"Error querying {db_name}: {e}")
                research_data[db_name] = {'papers': [], 'total': 0}
        
        # Analyze research trends
        trend_analysis = await self._analyze_research_data(research_data, domain)
        
        return trend_analysis

    async def _analyze_investment_trends(
        self, 
        domain: TechnologyDomain, 
        timeframe_years: int
    ) -> Dict[str, Any]:
        """
        Analyze investment trends in the technology domain
        """
        self.logger.info(f"Analyzing investment trends for {domain.value}")
        
        # Simulate investment data analysis
        # In production, this would query investment databases like Crunchbase, PitchBook, etc.
        investment_data = {
            'total_investment_millions': 5000.0,
            'deal_count': 150,
            'average_deal_size': 33.3,
            'growth_rate': 0.25,
            'top_investors': ['Andreessen Horowitz', 'Sequoia Capital', 'Google Ventures'],
            'funding_stages': {
                'seed': 0.2,
                'series_a': 0.3,
                'series_b': 0.25,
                'series_c+': 0.25
            },
            'geographic_distribution': {
                'north_america': 0.6,
                'europe': 0.2,
                'asia': 0.15,
                'other': 0.05
            }
        }
        
        return investment_data

    def _get_domain_search_terms(self, domain: TechnologyDomain) -> List[str]:
        """
        Get domain-specific search terms for patent and research queries
        """
        domain_terms = {
            TechnologyDomain.ARTIFICIAL_INTELLIGENCE: [
                'artificial intelligence', 'machine learning', 'deep learning',
                'neural network', 'natural language processing', 'computer vision',
                'reinforcement learning', 'transformer', 'large language model'
            ],
            TechnologyDomain.QUANTUM_COMPUTING: [
                'quantum computing', 'quantum algorithm', 'quantum gate',
                'quantum entanglement', 'quantum error correction', 'qubit',
                'quantum supremacy', 'quantum annealing', 'quantum cryptography'
            ],
            TechnologyDomain.BIOTECHNOLOGY: [
                'biotechnology', 'genetic engineering', 'CRISPR', 'gene therapy',
                'synthetic biology', 'bioinformatics', 'protein engineering',
                'cell therapy', 'biomarker', 'personalized medicine'
            ],
            TechnologyDomain.ROBOTICS: [
                'robotics', 'autonomous robot', 'robotic arm', 'humanoid robot',
                'robot navigation', 'robot perception', 'swarm robotics',
                'soft robotics', 'robot learning', 'human-robot interaction'
            ],
            TechnologyDomain.BLOCKCHAIN: [
                'blockchain', 'cryptocurrency', 'smart contract', 'distributed ledger',
                'consensus algorithm', 'proof of stake', 'decentralized finance',
                'NFT', 'web3', 'cryptocurrency mining'
            ]
        }
        
        return domain_terms.get(domain, ['technology', 'innovation', 'research'])

    async def _query_patent_database(
        self, 
        db_name: str, 
        api_config: Dict[str, str], 
        search_terms: List[str], 
        timeframe_years: int
    ) -> Dict[str, Any]:
        """
        Query a specific patent database
        """
        # Simulate patent database query
        # In production, this would make actual API calls
        
        patents = []
        for i, term in enumerate(search_terms[:5]):  # Limit to top 5 terms
            patent = {
                'id': f"{db_name}_patent_{i}",
                'title': f"Innovation in {term}",
                'abstract': f"This patent describes advances in {term} technology...",
                'inventors': [f"Inventor {i}", f"Co-inventor {i}"],
                'assignee': f"Tech Company {i}",
                'publication_date': datetime.now() - timedelta(days=i*30),
                'citations': 10 + i*5,
                'classification': f"G06F{i}/00",
                'search_term': term
            }
            patents.append(patent)
        
        return {
            'patents': patents,
            'total': len(patents),
            'database': db_name,
            'query_time': datetime.now()
        }

    async def _query_research_database(
        self, 
        db_name: str, 
        api_config: Dict[str, str], 
        search_terms: List[str], 
        timeframe_years: int
    ) -> Dict[str, Any]:
        """
        Query a specific research paper database
        """
        # Simulate research database query
        # In production, this would make actual API calls
        
        papers = []
        for i, term in enumerate(search_terms[:5]):  # Limit to top 5 terms
            paper = {
                'id': f"{db_name}_paper_{i}",
                'title': f"Advances in {term}: A Comprehensive Study",
                'abstract': f"This paper presents novel approaches to {term}...",
                'authors': [f"Dr. Author {i}", f"Prof. Co-author {i}"],
                'journal': f"Journal of {term} Research",
                'publication_date': datetime.now() - timedelta(days=i*15),
                'citations': 25 + i*10,
                'doi': f"10.1000/{db_name}.{i}",
                'keywords': [term, 'innovation', 'technology'],
                'search_term': term
            }
            papers.append(paper)
        
        return {
            'papers': papers,
            'total': len(papers),
            'database': db_name,
            'query_time': datetime.now()
        }

    async def _analyze_patent_data(
        self, 
        patent_data: Dict[str, Any], 
        domain: TechnologyDomain
    ) -> Dict[str, Any]:
        """
        Analyze aggregated patent data to identify trends
        """
        all_patents = []
        for db_data in patent_data.values():
            all_patents.extend(db_data.get('patents', []))
        
        # Analyze patent trends
        trend_analysis = {
            'total_patents': len(all_patents),
            'growth_trends': self._calculate_growth_trends(all_patents, 'publication_date'),
            'top_assignees': self._get_top_entities(all_patents, 'assignee'),
            'top_inventors': self._get_top_entities(all_patents, 'inventors'),
            'citation_trends': self._analyze_citation_trends(all_patents),
            'technology_clusters': self._identify_technology_clusters(all_patents),
            'emerging_areas': self._identify_emerging_areas(all_patents),
            'geographic_distribution': self._analyze_geographic_distribution(all_patents)
        }
        
        return trend_analysis

    async def _analyze_research_data(
        self, 
        research_data: Dict[str, Any], 
        domain: TechnologyDomain
    ) -> Dict[str, Any]:
        """
        Analyze aggregated research paper data to identify trends
        """
        all_papers = []
        for db_data in research_data.values():
            all_papers.extend(db_data.get('papers', []))
        
        # Analyze research trends
        trend_analysis = {
            'total_papers': len(all_papers),
            'growth_trends': self._calculate_growth_trends(all_papers, 'publication_date'),
            'top_authors': self._get_top_entities(all_papers, 'authors'),
            'top_journals': self._get_top_entities(all_papers, 'journal'),
            'citation_trends': self._analyze_citation_trends(all_papers),
            'research_clusters': self._identify_research_clusters(all_papers),
            'emerging_topics': self._identify_emerging_topics(all_papers),
            'collaboration_networks': self._analyze_collaboration_networks(all_papers)
        }
        
        return trend_analysis

    async def _synthesize_trend_data(
        self, 
        patent_trends: Dict[str, Any], 
        research_trends: Dict[str, Any], 
        investment_data: Dict[str, Any], 
        domain: TechnologyDomain
    ) -> List[Dict[str, Any]]:
        """
        Synthesize trends from patents, research, and investment data
        """
        # Combine and correlate data from all sources
        synthesized_trends = []
        
        # Extract key trends from each source
        patent_areas = patent_trends.get('emerging_areas', [])
        research_topics = research_trends.get('emerging_topics', [])
        
        # Create unified trend objects
        all_trend_names = set(patent_areas + research_topics)
        
        for trend_name in list(all_trend_names)[:10]:  # Top 10 trends
            trend_data = {
                'name': trend_name,
                'domain': domain,
                'patent_activity': self._get_trend_patent_activity(trend_name, patent_trends),
                'research_activity': self._get_trend_research_activity(trend_name, research_trends),
                'investment_activity': self._get_trend_investment_activity(trend_name, investment_data),
                'key_players': self._identify_trend_key_players(trend_name, patent_trends, research_trends),
                'growth_indicators': self._calculate_trend_growth_indicators(trend_name, patent_trends, research_trends)
            }
            synthesized_trends.append(trend_data)
        
        return synthesized_trends

    async def _calculate_trend_metrics(
        self, 
        trend_data: Dict[str, Any], 
        domain: TechnologyDomain
    ) -> TechnologyTrend:
        """
        Calculate comprehensive metrics for a technology trend
        """
        # Calculate momentum score based on multiple factors
        momentum_score = self._calculate_momentum_score(trend_data)
        
        # Predict breakthrough timeline
        breakthrough_timeline = self._predict_breakthrough_timeline(trend_data)
        
        return TechnologyTrend(
            trend_name=trend_data['name'],
            domain=domain,
            momentum_score=momentum_score,
            patent_activity=trend_data['patent_activity'],
            research_papers=trend_data['research_activity'],
            investment_millions=trend_data['investment_activity'],
            key_players=trend_data['key_players'],
            predicted_breakthrough_timeline=breakthrough_timeline
        )

    # Helper methods for data analysis
    
    def _calculate_growth_trends(self, items: List[Dict], date_field: str) -> Dict[str, float]:
        """Calculate growth trends from timestamped items"""
        if not items:
            return {'annual_growth': 0.0, 'monthly_growth': 0.0}
        
        # Group by year and month
        yearly_counts = defaultdict(int)
        monthly_counts = defaultdict(int)
        
        for item in items:
            date = item.get(date_field)
            if isinstance(date, datetime):
                yearly_counts[date.year] += 1
                monthly_counts[f"{date.year}-{date.month:02d}"] += 1
        
        # Calculate growth rates
        years = sorted(yearly_counts.keys())
        if len(years) >= 2:
            recent_year = yearly_counts[years[-1]]
            previous_year = yearly_counts[years[-2]]
            annual_growth = (recent_year - previous_year) / previous_year if previous_year > 0 else 0
        else:
            annual_growth = 0.0
        
        return {
            'annual_growth': annual_growth,
            'monthly_growth': annual_growth / 12  # Approximate
        }

    def _get_top_entities(self, items: List[Dict], field: str, top_n: int = 5) -> List[str]:
        """Get top entities from a field"""
        entity_counts = Counter()
        
        for item in items:
            entities = item.get(field, [])
            if isinstance(entities, str):
                entities = [entities]
            elif isinstance(entities, list):
                pass
            else:
                continue
                
            for entity in entities:
                entity_counts[entity] += 1
        
        return [entity for entity, count in entity_counts.most_common(top_n)]

    def _analyze_citation_trends(self, items: List[Dict]) -> Dict[str, float]:
        """Analyze citation trends"""
        if not items:
            return {'average_citations': 0.0, 'citation_growth': 0.0}
        
        citations = [item.get('citations', 0) for item in items]
        average_citations = sum(citations) / len(citations)
        
        return {
            'average_citations': average_citations,
            'citation_growth': 0.15  # Simulated growth rate
        }

    def _identify_technology_clusters(self, patents: List[Dict]) -> List[str]:
        """Identify technology clusters from patents"""
        # Simulate clustering analysis
        return ['AI/ML Cluster', 'Hardware Cluster', 'Software Cluster']

    def _identify_emerging_areas(self, patents: List[Dict]) -> List[str]:
        """Identify emerging technology areas"""
        # Extract from search terms and titles
        areas = set()
        for patent in patents:
            if 'search_term' in patent:
                areas.add(patent['search_term'])
        return list(areas)[:5]

    def _analyze_geographic_distribution(self, items: List[Dict]) -> Dict[str, float]:
        """Analyze geographic distribution"""
        # Simulate geographic analysis
        return {
            'north_america': 0.5,
            'europe': 0.25,
            'asia': 0.2,
            'other': 0.05
        }

    def _identify_research_clusters(self, papers: List[Dict]) -> List[str]:
        """Identify research clusters"""
        return ['Theoretical Research', 'Applied Research', 'Experimental Research']

    def _identify_emerging_topics(self, papers: List[Dict]) -> List[str]:
        """Identify emerging research topics"""
        topics = set()
        for paper in papers:
            if 'search_term' in paper:
                topics.add(paper['search_term'])
        return list(topics)[:5]

    def _analyze_collaboration_networks(self, papers: List[Dict]) -> Dict[str, Any]:
        """Analyze collaboration networks"""
        return {
            'collaboration_rate': 0.8,
            'average_authors_per_paper': 3.5,
            'international_collaboration': 0.3
        }

    def _get_trend_patent_activity(self, trend_name: str, patent_trends: Dict[str, Any]) -> int:
        """Get patent activity for a specific trend"""
        return 150  # Simulated value

    def _get_trend_research_activity(self, trend_name: str, research_trends: Dict[str, Any]) -> int:
        """Get research activity for a specific trend"""
        return 300  # Simulated value

    def _get_trend_investment_activity(self, trend_name: str, investment_data: Dict[str, Any]) -> float:
        """Get investment activity for a specific trend"""
        return 500.0  # Simulated value in millions

    def _identify_trend_key_players(
        self, 
        trend_name: str, 
        patent_trends: Dict[str, Any], 
        research_trends: Dict[str, Any]
    ) -> List[str]:
        """Identify key players in a trend"""
        patent_players = patent_trends.get('top_assignees', [])
        research_players = research_trends.get('top_authors', [])
        
        # Combine and deduplicate
        all_players = list(set(patent_players + research_players))
        return all_players[:5]

    def _calculate_trend_growth_indicators(
        self, 
        trend_name: str, 
        patent_trends: Dict[str, Any], 
        research_trends: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate growth indicators for a trend"""
        return {
            'patent_growth': 0.25,
            'research_growth': 0.30,
            'overall_momentum': 0.275
        }

    def _calculate_momentum_score(self, trend_data: Dict[str, Any]) -> float:
        """Calculate momentum score for a trend"""
        # Weighted combination of different factors
        patent_weight = 0.3
        research_weight = 0.3
        investment_weight = 0.4
        
        # Normalize values (simplified)
        patent_score = min(trend_data['patent_activity'] / 200.0, 1.0)
        research_score = min(trend_data['research_activity'] / 400.0, 1.0)
        investment_score = min(trend_data['investment_activity'] / 1000.0, 1.0)
        
        momentum = (
            patent_score * patent_weight +
            research_score * research_weight +
            investment_score * investment_weight
        )
        
        return min(momentum, 1.0)

    def _predict_breakthrough_timeline(self, trend_data: Dict[str, Any]) -> int:
        """Predict breakthrough timeline in years"""
        # Simple heuristic based on activity levels
        momentum = self._calculate_momentum_score(trend_data)
        
        if momentum > 0.8:
            return 2
        elif momentum > 0.6:
            return 3
        elif momentum > 0.4:
            return 5
        else:
            return 7