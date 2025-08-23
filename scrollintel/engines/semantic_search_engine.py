"""
Semantic Search Engine for Enterprise Data Sources

This engine provides advanced semantic search capabilities that exceed traditional
keyword-based search by understanding context, intent, and relationships across
all enterprise data sources.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from collections import defaultdict
import json
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy

from ..models.advanced_analytics_models import (
    SemanticQuery, SemanticSearchResult, SemanticSearchResponse,
    AnalyticsInsight
)
from ..core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class SemanticSearchEngine:
    """
    Enterprise-grade semantic search engine for comprehensive data discovery.
    
    Capabilities:
    - Natural language query understanding
    - Multi-modal semantic search across text, documents, and structured data
    - Context-aware result ranking
    - Entity extraction and relationship mapping
    - Real-time search across streaming data
    - Federated search across multiple enterprise systems
    """
    
    def __init__(self):
        self.sentence_transformer = None
        self.tfidf_vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
        self.nlp = None
        self.document_embeddings = {}
        self.document_metadata = {}
        self.search_index = {}
        self.entity_cache = {}
        
        # Initialize models asynchronously
        asyncio.create_task(self._initialize_models())
    
    async def _initialize_models(self):
        """Initialize NLP models for semantic search."""
        try:
            # Load sentence transformer for semantic embeddings
            self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Load spaCy model for NER and linguistic analysis
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                logger.warning("spaCy model not found, using basic NLP processing")
                self.nlp = None
            
            logger.info("Semantic search models initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing semantic search models: {str(e)}")
    
    async def index_enterprise_data(self, data_sources: List[str]) -> Dict[str, Any]:
        """
        Index data from multiple enterprise sources for semantic search.
        
        Args:
            data_sources: List of data source identifiers
            
        Returns:
            Indexing statistics and metrics
        """
        start_time = datetime.utcnow()
        
        try:
            total_documents = 0
            total_size_mb = 0
            
            for source in data_sources:
                documents = await self._extract_documents_from_source(source)
                
                for doc in documents:
                    await self._index_document(doc)
                    total_documents += 1
                    total_size_mb += len(doc.get('content', '')) / (1024 * 1024)
            
            # Build search indices
            await self._build_search_indices()
            
            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            logger.info(f"Indexed {total_documents} documents from {len(data_sources)} sources in {execution_time:.2f}ms")
            
            return {
                "documents_indexed": total_documents,
                "data_sources": len(data_sources),
                "total_size_mb": total_size_mb,
                "execution_time_ms": execution_time,
                "index_size": len(self.document_embeddings)
            }
            
        except Exception as e:
            logger.error(f"Error indexing enterprise data: {str(e)}")
            raise
    
    async def semantic_search(self, query: SemanticQuery) -> SemanticSearchResponse:
        """
        Perform semantic search across indexed enterprise data.
        
        Args:
            query: Semantic search query with parameters
            
        Returns:
            Comprehensive search results with semantic relevance
        """
        start_time = datetime.utcnow()
        
        try:
            # Process and understand the query
            processed_query = await self._process_query(query)
            
            # Perform multi-modal search
            semantic_results = await self._semantic_similarity_search(processed_query)
            keyword_results = await self._keyword_search(processed_query)
            entity_results = await self._entity_based_search(processed_query)
            
            # Combine and rank results
            combined_results = await self._combine_and_rank_results(
                semantic_results, keyword_results, entity_results, query
            )
            
            # Apply filters and limits
            filtered_results = self._apply_filters(combined_results, query)
            final_results = filtered_results[:query.max_results]
            
            # Generate search insights
            insights = await self._generate_search_insights(query, final_results)
            
            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            response = SemanticSearchResponse(
                query=query,
                results=final_results,
                total_results=len(filtered_results),
                execution_time_ms=execution_time,
                search_insights=insights
            )
            
            logger.info(f"Semantic search completed: {len(final_results)} results in {execution_time:.2f}ms")
            
            return response
            
        except Exception as e:
            logger.error(f"Error in semantic search: {str(e)}")
            raise
    
    async def discover_related_content(self, content_id: str, max_results: int = 20) -> List[SemanticSearchResult]:
        """
        Discover content related to a specific document or entity.
        
        Args:
            content_id: ID of the source content
            max_results: Maximum number of related items to return
            
        Returns:
            List of related content items
        """
        try:
            if content_id not in self.document_embeddings:
                return []
            
            source_embedding = self.document_embeddings[content_id]
            related_items = []
            
            # Calculate similarity with all other documents
            for doc_id, embedding in self.document_embeddings.items():
                if doc_id != content_id:
                    similarity = cosine_similarity([source_embedding], [embedding])[0][0]
                    
                    if similarity > 0.3:  # Minimum similarity threshold
                        metadata = self.document_metadata.get(doc_id, {})
                        
                        result = SemanticSearchResult(
                            content=metadata.get('content', '')[:500],  # First 500 chars
                            source=metadata.get('source', 'unknown'),
                            source_type=metadata.get('source_type', 'document'),
                            relevance_score=float(similarity),
                            metadata=metadata,
                            extracted_entities=metadata.get('entities', []),
                            key_concepts=metadata.get('concepts', [])
                        )
                        related_items.append(result)
            
            # Sort by relevance and return top results
            related_items.sort(key=lambda x: x.relevance_score, reverse=True)
            
            logger.info(f"Found {len(related_items)} related items for content {content_id}")
            
            return related_items[:max_results]
            
        except Exception as e:
            logger.error(f"Error discovering related content: {str(e)}")
            return []
    
    async def extract_insights_from_search_patterns(self, search_history: List[SemanticQuery]) -> List[AnalyticsInsight]:
        """
        Extract business insights from search patterns and user behavior.
        
        Args:
            search_history: Historical search queries
            
        Returns:
            List of insights derived from search patterns
        """
        try:
            insights = []
            
            if not search_history:
                return insights
            
            # Analyze query patterns
            query_texts = [query.query_text for query in search_history]
            
            # Extract common themes
            common_themes = await self._extract_common_themes(query_texts)
            if common_themes:
                insight = AnalyticsInsight(
                    title="Common Search Themes",
                    description=f"Analysis of search patterns reveals {len(common_themes)} recurring themes: {', '.join(common_themes[:3])}",
                    insight_type="search_pattern",
                    confidence=0.8,
                    business_impact="Understanding search patterns can guide content strategy and information architecture",
                    supporting_data={"themes": common_themes},
                    recommended_actions=[
                        "Develop content around frequently searched themes",
                        "Improve information architecture for common topics",
                        "Create targeted knowledge bases for popular searches"
                    ],
                    priority=6
                )
                insights.append(insight)
            
            # Analyze search result gaps
            gap_analysis = await self._analyze_search_gaps(search_history)
            if gap_analysis:
                insights.extend(gap_analysis)
            
            # Analyze entity interest patterns
            entity_insights = await self._analyze_entity_search_patterns(search_history)
            if entity_insights:
                insights.extend(entity_insights)
            
            logger.info(f"Generated {len(insights)} insights from search patterns")
            
            return insights
            
        except Exception as e:
            logger.error(f"Error extracting search insights: {str(e)}")
            return []
    
    async def _extract_documents_from_source(self, source: str) -> List[Dict[str, Any]]:
        """Extract documents from a specific data source."""
        documents = []
        
        if source == "knowledge_base":
            # Simulate knowledge base documents
            kb_docs = [
                {
                    "id": f"kb_{i}",
                    "title": f"Knowledge Article {i}",
                    "content": self._generate_sample_content("knowledge", i),
                    "source": "knowledge_base",
                    "source_type": "article",
                    "created_at": datetime.utcnow(),
                    "tags": ["documentation", "process", "guide"]
                }
                for i in range(50)
            ]
            documents.extend(kb_docs)
            
        elif source == "customer_communications":
            # Simulate customer communication documents
            comm_docs = [
                {
                    "id": f"comm_{i}",
                    "title": f"Customer Communication {i}",
                    "content": self._generate_sample_content("communication", i),
                    "source": "customer_communications",
                    "source_type": "email",
                    "created_at": datetime.utcnow(),
                    "tags": ["customer", "support", "inquiry"]
                }
                for i in range(30)
            ]
            documents.extend(comm_docs)
            
        elif source == "financial_reports":
            # Simulate financial report documents
            fin_docs = [
                {
                    "id": f"fin_{i}",
                    "title": f"Financial Report {i}",
                    "content": self._generate_sample_content("financial", i),
                    "source": "financial_reports",
                    "source_type": "report",
                    "created_at": datetime.utcnow(),
                    "tags": ["finance", "analysis", "metrics"]
                }
                for i in range(20)
            ]
            documents.extend(fin_docs)
            
        elif source == "operational_data":
            # Simulate operational documents
            ops_docs = [
                {
                    "id": f"ops_{i}",
                    "title": f"Operational Document {i}",
                    "content": self._generate_sample_content("operational", i),
                    "source": "operational_data",
                    "source_type": "log",
                    "created_at": datetime.utcnow(),
                    "tags": ["operations", "performance", "monitoring"]
                }
                for i in range(40)
            ]
            documents.extend(ops_docs)
        
        return documents
    
    def _generate_sample_content(self, content_type: str, index: int) -> str:
        """Generate sample content for different document types."""
        content_templates = {
            "knowledge": [
                f"This knowledge article explains the process for handling customer inquiries related to product {index}. The standard procedure involves initial assessment, categorization, and routing to appropriate specialists.",
                f"Documentation for system configuration {index} includes setup procedures, troubleshooting steps, and maintenance guidelines for optimal performance.",
                f"Best practices guide {index} covers methodology for data analysis, quality assurance, and reporting standards used across the organization."
            ],
            "communication": [
                f"Customer inquiry {index} regarding product features and pricing options. Customer expressed interest in enterprise solutions and requested detailed proposal.",
                f"Support ticket {index} resolved: Issue with system integration affecting data synchronization. Solution implemented with monitoring in place.",
                f"Sales communication {index}: Follow-up with prospect regarding implementation timeline and resource requirements for deployment."
            ],
            "financial": [
                f"Financial analysis {index} shows revenue growth of 15% quarter-over-quarter with strong performance in enterprise segment. Cost optimization initiatives yielding positive results.",
                f"Budget report {index} indicates spending within allocated limits across all departments. Recommendations for resource reallocation to high-priority projects.",
                f"Investment analysis {index} evaluates ROI for technology infrastructure upgrades. Projected benefits include improved efficiency and reduced operational costs."
            ],
            "operational": [
                f"System performance report {index}: Average response time 150ms, uptime 99.8%, throughput 1000 requests per second. No critical issues detected.",
                f"Operational metrics {index} show improvement in process efficiency by 12%. Automation initiatives reducing manual effort and error rates.",
                f"Infrastructure monitoring {index}: All systems operating within normal parameters. Capacity utilization at 65% with room for growth."
            ]
        }
        
        templates = content_templates.get(content_type, ["Generic content"])
        return templates[index % len(templates)]
    
    async def _index_document(self, document: Dict[str, Any]):
        """Index a single document for semantic search."""
        doc_id = document['id']
        content = document['content']
        
        # Generate semantic embedding
        if self.sentence_transformer:
            embedding = self.sentence_transformer.encode(content)
            self.document_embeddings[doc_id] = embedding
        
        # Extract entities and concepts
        entities = await self._extract_entities(content)
        concepts = await self._extract_key_concepts(content)
        
        # Store metadata
        metadata = {
            **document,
            'entities': entities,
            'concepts': concepts,
            'indexed_at': datetime.utcnow()
        }
        self.document_metadata[doc_id] = metadata
        
        # Update entity cache
        for entity in entities:
            if entity not in self.entity_cache:
                self.entity_cache[entity] = []
            self.entity_cache[entity].append(doc_id)
    
    async def _build_search_indices(self):
        """Build additional search indices for faster retrieval."""
        # Build TF-IDF index for keyword search
        if self.document_metadata:
            documents = [meta['content'] for meta in self.document_metadata.values()]
            try:
                self.tfidf_vectorizer.fit(documents)
                
                # Create TF-IDF vectors for all documents
                tfidf_matrix = self.tfidf_vectorizer.transform(documents)
                
                doc_ids = list(self.document_metadata.keys())
                for i, doc_id in enumerate(doc_ids):
                    self.search_index[doc_id] = tfidf_matrix[i]
                    
            except Exception as e:
                logger.warning(f"Could not build TF-IDF index: {str(e)}")
    
    async def _process_query(self, query: SemanticQuery) -> Dict[str, Any]:
        """Process and understand the search query."""
        processed = {
            'original_text': query.query_text,
            'cleaned_text': self._clean_query_text(query.query_text),
            'entities': [],
            'concepts': [],
            'intent': 'search',
            'embedding': None
        }
        
        # Extract entities from query
        processed['entities'] = await self._extract_entities(query.query_text)
        
        # Extract key concepts
        processed['concepts'] = await self._extract_key_concepts(query.query_text)
        
        # Generate query embedding
        if self.sentence_transformer:
            processed['embedding'] = self.sentence_transformer.encode(query.query_text)
        
        # Determine search intent
        processed['intent'] = self._determine_search_intent(query.query_text)
        
        return processed
    
    def _clean_query_text(self, text: str) -> str:
        """Clean and normalize query text."""
        # Remove special characters and normalize whitespace
        cleaned = re.sub(r'[^\w\s]', ' ', text)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip().lower()
        return cleaned
    
    async def _extract_entities(self, text: str) -> List[str]:
        """Extract named entities from text."""
        entities = []
        
        if self.nlp:
            try:
                doc = self.nlp(text)
                entities = [ent.text for ent in doc.ents if ent.label_ in ['PERSON', 'ORG', 'PRODUCT', 'MONEY', 'DATE']]
            except Exception as e:
                logger.warning(f"Entity extraction failed: {str(e)}")
        
        # Fallback: simple pattern matching
        if not entities:
            # Look for capitalized words (potential entities)
            words = text.split()
            entities = [word for word in words if word[0].isupper() and len(word) > 2]
        
        return list(set(entities))  # Remove duplicates
    
    async def _extract_key_concepts(self, text: str) -> List[str]:
        """Extract key concepts from text."""
        concepts = []
        
        # Business-related keywords
        business_keywords = [
            'revenue', 'profit', 'customer', 'product', 'service', 'market', 'sales',
            'growth', 'strategy', 'performance', 'efficiency', 'cost', 'investment',
            'analysis', 'report', 'data', 'system', 'process', 'quality', 'risk'
        ]
        
        text_lower = text.lower()
        for keyword in business_keywords:
            if keyword in text_lower:
                concepts.append(keyword)
        
        # Extract noun phrases if spaCy is available
        if self.nlp:
            try:
                doc = self.nlp(text)
                noun_phrases = [chunk.text for chunk in doc.noun_chunks if len(chunk.text.split()) <= 3]
                concepts.extend(noun_phrases[:5])  # Limit to top 5
            except Exception as e:
                logger.warning(f"Concept extraction failed: {str(e)}")
        
        return list(set(concepts))  # Remove duplicates
    
    def _determine_search_intent(self, query_text: str) -> str:
        """Determine the intent behind the search query."""
        query_lower = query_text.lower()
        
        if any(word in query_lower for word in ['how', 'what', 'why', 'when', 'where']):
            return 'question'
        elif any(word in query_lower for word in ['find', 'search', 'look for', 'locate']):
            return 'find'
        elif any(word in query_lower for word in ['analyze', 'compare', 'evaluate']):
            return 'analysis'
        elif any(word in query_lower for word in ['report', 'summary', 'overview']):
            return 'report'
        else:
            return 'search'
    
    async def _semantic_similarity_search(self, processed_query: Dict[str, Any]) -> List[Tuple[str, float]]:
        """Perform semantic similarity search using embeddings."""
        results = []
        
        if processed_query['embedding'] is not None and self.document_embeddings:
            query_embedding = processed_query['embedding']
            
            for doc_id, doc_embedding in self.document_embeddings.items():
                similarity = cosine_similarity([query_embedding], [doc_embedding])[0][0]
                if similarity > 0.1:  # Minimum similarity threshold
                    results.append((doc_id, float(similarity)))
        
        # Sort by similarity score
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:100]  # Top 100 semantic matches
    
    async def _keyword_search(self, processed_query: Dict[str, Any]) -> List[Tuple[str, float]]:
        """Perform keyword-based search using TF-IDF."""
        results = []
        
        if hasattr(self.tfidf_vectorizer, 'vocabulary_') and self.search_index:
            try:
                query_vector = self.tfidf_vectorizer.transform([processed_query['cleaned_text']])
                
                for doc_id, doc_vector in self.search_index.items():
                    similarity = cosine_similarity(query_vector, doc_vector)[0][0]
                    if similarity > 0.05:  # Minimum similarity threshold
                        results.append((doc_id, float(similarity)))
                        
            except Exception as e:
                logger.warning(f"Keyword search failed: {str(e)}")
        
        # Sort by similarity score
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:100]  # Top 100 keyword matches
    
    async def _entity_based_search(self, processed_query: Dict[str, Any]) -> List[Tuple[str, float]]:
        """Perform entity-based search."""
        results = []
        
        query_entities = processed_query['entities']
        
        if query_entities and self.entity_cache:
            entity_matches = defaultdict(int)
            
            for entity in query_entities:
                if entity in self.entity_cache:
                    for doc_id in self.entity_cache[entity]:
                        entity_matches[doc_id] += 1
            
            # Convert to results with scores based on entity match count
            for doc_id, match_count in entity_matches.items():
                score = min(1.0, match_count / len(query_entities))  # Normalize score
                results.append((doc_id, score))
        
        # Sort by score
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:50]  # Top 50 entity matches
    
    async def _combine_and_rank_results(self, semantic_results: List[Tuple[str, float]], 
                                      keyword_results: List[Tuple[str, float]], 
                                      entity_results: List[Tuple[str, float]], 
                                      query: SemanticQuery) -> List[SemanticSearchResult]:
        """Combine and rank results from different search methods."""
        # Combine scores with weights
        combined_scores = defaultdict(float)
        
        # Semantic search weight: 0.5
        for doc_id, score in semantic_results:
            combined_scores[doc_id] += 0.5 * score
        
        # Keyword search weight: 0.3
        for doc_id, score in keyword_results:
            combined_scores[doc_id] += 0.3 * score
        
        # Entity search weight: 0.2
        for doc_id, score in entity_results:
            combined_scores[doc_id] += 0.2 * score
        
        # Create search results
        search_results = []
        for doc_id, combined_score in combined_scores.items():
            if doc_id in self.document_metadata:
                metadata = self.document_metadata[doc_id]
                
                result = SemanticSearchResult(
                    content=metadata['content'][:1000],  # First 1000 chars
                    source=metadata['source'],
                    source_type=metadata['source_type'],
                    relevance_score=min(1.0, combined_score),  # Cap at 1.0
                    metadata=metadata,
                    extracted_entities=metadata.get('entities', []),
                    key_concepts=metadata.get('concepts', [])
                )
                search_results.append(result)
        
        # Sort by relevance score
        search_results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        return search_results
    
    def _apply_filters(self, results: List[SemanticSearchResult], query: SemanticQuery) -> List[SemanticSearchResult]:
        """Apply filters to search results."""
        filtered_results = results
        
        # Apply similarity threshold
        filtered_results = [r for r in filtered_results if r.relevance_score >= query.similarity_threshold]
        
        # Apply custom filters
        if query.filters:
            for filter_key, filter_value in query.filters.items():
                if filter_key == 'source_type':
                    filtered_results = [r for r in filtered_results if r.source_type == filter_value]
                elif filter_key == 'source':
                    filtered_results = [r for r in filtered_results if r.source == filter_value]
                elif filter_key == 'tags':
                    if isinstance(filter_value, list):
                        filtered_results = [r for r in filtered_results 
                                          if any(tag in r.metadata.get('tags', []) for tag in filter_value)]
                    else:
                        filtered_results = [r for r in filtered_results 
                                          if filter_value in r.metadata.get('tags', [])]
        
        return filtered_results
    
    async def _generate_search_insights(self, query: SemanticQuery, results: List[SemanticSearchResult]) -> List[str]:
        """Generate insights about the search results."""
        insights = []
        
        if not results:
            insights.append("No results found. Consider broadening your search terms or checking spelling.")
            return insights
        
        # Analyze result distribution by source
        source_counts = defaultdict(int)
        for result in results:
            source_counts[result.source] += 1
        
        if source_counts:
            top_source = max(source_counts.items(), key=lambda x: x[1])
            insights.append(f"Most relevant results ({top_source[1]}) found in {top_source[0]}")
        
        # Analyze relevance scores
        avg_relevance = np.mean([r.relevance_score for r in results])
        if avg_relevance > 0.8:
            insights.append("High-quality matches found with strong semantic relevance")
        elif avg_relevance > 0.5:
            insights.append("Good matches found, consider refining query for better results")
        else:
            insights.append("Moderate matches found, try different keywords or broader terms")
        
        # Analyze entities in results
        all_entities = []
        for result in results[:10]:  # Top 10 results
            all_entities.extend(result.extracted_entities)
        
        if all_entities:
            entity_counts = defaultdict(int)
            for entity in all_entities:
                entity_counts[entity] += 1
            
            common_entities = sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)[:3]
            if common_entities:
                entity_names = [entity for entity, _ in common_entities]
                insights.append(f"Key entities in results: {', '.join(entity_names)}")
        
        return insights
    
    async def _extract_common_themes(self, query_texts: List[str]) -> List[str]:
        """Extract common themes from search queries."""
        if not query_texts:
            return []
        
        # Simple approach: find most common words
        all_words = []
        for query in query_texts:
            words = self._clean_query_text(query).split()
            all_words.extend([word for word in words if len(word) > 3])  # Filter short words
        
        if not all_words:
            return []
        
        # Count word frequencies
        word_counts = defaultdict(int)
        for word in all_words:
            word_counts[word] += 1
        
        # Return most common themes
        common_themes = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        return [theme for theme, count in common_themes[:10] if count > 1]
    
    async def _analyze_search_gaps(self, search_history: List[SemanticQuery]) -> List[AnalyticsInsight]:
        """Analyze gaps in search results to identify content needs."""
        insights = []
        
        # Simulate analysis of queries with poor results
        low_result_queries = [q for q in search_history if len(q.query_text.split()) > 3]  # Complex queries
        
        if len(low_result_queries) > len(search_history) * 0.3:  # More than 30% complex queries
            insight = AnalyticsInsight(
                title="Content Gap Identified",
                description=f"Analysis shows {len(low_result_queries)} complex queries that may indicate content gaps in the knowledge base",
                insight_type="content_gap",
                confidence=0.75,
                business_impact="Addressing content gaps can improve user satisfaction and reduce support burden",
                supporting_data={"complex_queries": len(low_result_queries)},
                recommended_actions=[
                    "Create content addressing complex query topics",
                    "Improve existing documentation depth",
                    "Develop FAQ sections for common complex queries"
                ],
                priority=7
            )
            insights.append(insight)
        
        return insights
    
    async def _analyze_entity_search_patterns(self, search_history: List[SemanticQuery]) -> List[AnalyticsInsight]:
        """Analyze entity search patterns for business insights."""
        insights = []
        
        # Extract entities from all queries
        all_entities = []
        for query in search_history:
            entities = await self._extract_entities(query.query_text)
            all_entities.extend(entities)
        
        if all_entities:
            entity_counts = defaultdict(int)
            for entity in all_entities:
                entity_counts[entity] += 1
            
            # Find frequently searched entities
            frequent_entities = [entity for entity, count in entity_counts.items() if count > 2]
            
            if frequent_entities:
                insight = AnalyticsInsight(
                    title="High-Interest Entities",
                    description=f"Identified {len(frequent_entities)} entities with high search frequency, indicating strong business interest",
                    insight_type="entity_interest",
                    confidence=0.85,
                    business_impact="High-interest entities represent key business focus areas and opportunities",
                    supporting_data={"frequent_entities": frequent_entities[:5]},
                    recommended_actions=[
                        "Develop comprehensive content around high-interest entities",
                        "Monitor entity-related business developments",
                        "Create entity-specific dashboards and reports"
                    ],
                    priority=8
                )
                insights.append(insight)
        
        return insights