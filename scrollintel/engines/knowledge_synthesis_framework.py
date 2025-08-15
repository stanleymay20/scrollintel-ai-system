"""
Knowledge Synthesis Framework for Autonomous Innovation Lab

This module implements the knowledge synthesis framework that integrates
research findings and experimental results, identifies correlations and patterns,
and provides knowledge validation and quality assurance.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Set
from datetime import datetime, timedelta
import numpy as np
from dataclasses import asdict

from ..models.knowledge_integration_models import (
    KnowledgeItem, KnowledgeCorrelation, SynthesizedKnowledge,
    KnowledgeValidationResult, KnowledgeGraph, SynthesisRequest,
    KnowledgeType, ConfidenceLevel
)

logger = logging.getLogger(__name__)


class KnowledgeSynthesisFramework:
    """
    Framework for synthesizing knowledge from research findings and experimental results
    """
    
    def __init__(self):
        self.knowledge_store: Dict[str, KnowledgeItem] = {}
        self.correlations: Dict[str, KnowledgeCorrelation] = {}
        self.synthesized_knowledge: Dict[str, SynthesizedKnowledge] = {}
        self.validation_cache: Dict[str, KnowledgeValidationResult] = {}
        
    async def integrate_research_findings(
        self,
        findings: List[Dict[str, Any]]
    ) -> List[KnowledgeItem]:
        """
        Integrate research findings into the knowledge base
        
        Args:
            findings: List of research findings to integrate
            
        Returns:
            List of created knowledge items
        """
        try:
            knowledge_items = []
            
            for finding in findings:
                # Create knowledge item from research finding
                knowledge_item = KnowledgeItem(
                    id=f"research_{datetime.now().timestamp()}_{len(self.knowledge_store)}",
                    knowledge_type=KnowledgeType.RESEARCH_FINDING,
                    content=finding,
                    source=finding.get('source', 'unknown'),
                    timestamp=datetime.now(),
                    confidence=self._determine_confidence(finding),
                    metadata={
                        'integration_method': 'research_findings',
                        'processed_at': datetime.now().isoformat()
                    },
                    tags=self._extract_tags(finding)
                )
                
                # Store knowledge item
                self.knowledge_store[knowledge_item.id] = knowledge_item
                knowledge_items.append(knowledge_item)
                
                # Identify relationships with existing knowledge
                await self._identify_relationships(knowledge_item)
                
            logger.info(f"Integrated {len(knowledge_items)} research findings")
            return knowledge_items
            
        except Exception as e:
            logger.error(f"Error integrating research findings: {str(e)}")
            raise
    
    async def integrate_experimental_results(
        self,
        results: List[Dict[str, Any]]
    ) -> List[KnowledgeItem]:
        """
        Integrate experimental results into the knowledge base
        
        Args:
            results: List of experimental results to integrate
            
        Returns:
            List of created knowledge items
        """
        try:
            knowledge_items = []
            
            for result in results:
                # Create knowledge item from experimental result
                knowledge_item = KnowledgeItem(
                    id=f"experiment_{datetime.now().timestamp()}_{len(self.knowledge_store)}",
                    knowledge_type=KnowledgeType.EXPERIMENTAL_RESULT,
                    content=result,
                    source=result.get('experiment_id', 'unknown'),
                    timestamp=datetime.now(),
                    confidence=self._determine_confidence(result),
                    metadata={
                        'integration_method': 'experimental_results',
                        'processed_at': datetime.now().isoformat(),
                        'experiment_metadata': result.get('metadata', {})
                    },
                    tags=self._extract_tags(result)
                )
                
                # Store knowledge item
                self.knowledge_store[knowledge_item.id] = knowledge_item
                knowledge_items.append(knowledge_item)
                
                # Identify relationships with existing knowledge
                await self._identify_relationships(knowledge_item)
                
            logger.info(f"Integrated {len(knowledge_items)} experimental results")
            return knowledge_items
            
        except Exception as e:
            logger.error(f"Error integrating experimental results: {str(e)}")
            raise
    
    async def identify_knowledge_correlations(
        self,
        knowledge_ids: Optional[List[str]] = None
    ) -> List[KnowledgeCorrelation]:
        """
        Identify correlations between knowledge items
        
        Args:
            knowledge_ids: Optional list of specific knowledge IDs to analyze
            
        Returns:
            List of identified correlations
        """
        try:
            if knowledge_ids is None:
                knowledge_ids = list(self.knowledge_store.keys())
            
            correlations = []
            
            # Analyze pairs of knowledge items for correlations
            for i, id1 in enumerate(knowledge_ids):
                for id2 in knowledge_ids[i+1:]:
                    correlation = await self._analyze_correlation(id1, id2)
                    if correlation and correlation.strength > 0.3:  # Threshold for meaningful correlation
                        correlations.append(correlation)
                        self.correlations[correlation.id] = correlation
            
            logger.info(f"Identified {len(correlations)} knowledge correlations")
            return correlations
            
        except Exception as e:
            logger.error(f"Error identifying correlations: {str(e)}")
            raise
    
    async def synthesize_knowledge(
        self,
        synthesis_request: SynthesisRequest
    ) -> SynthesizedKnowledge:
        """
        Synthesize knowledge from multiple sources
        
        Args:
            synthesis_request: Request specifying synthesis parameters
            
        Returns:
            Synthesized knowledge result
        """
        try:
            # Get source knowledge items
            source_items = [
                self.knowledge_store[kid] 
                for kid in synthesis_request.source_knowledge_ids
                if kid in self.knowledge_store
            ]
            
            if not source_items:
                raise ValueError("No valid source knowledge items found")
            
            # Perform synthesis based on method
            synthesis_method = synthesis_request.method_preferences[0] if synthesis_request.method_preferences else "default"
            synthesized_content = await self._perform_synthesis(source_items, synthesis_method)
            
            # Generate insights
            insights = await self._generate_insights(source_items, synthesized_content)
            
            # Create synthesized knowledge
            synthesized = SynthesizedKnowledge(
                id=f"synthesis_{datetime.now().timestamp()}",
                source_items=synthesis_request.source_knowledge_ids,
                synthesis_method=synthesis_method,
                synthesized_content=synthesized_content,
                insights=insights,
                confidence=self._calculate_synthesis_confidence(source_items),
                created_at=datetime.now(),
                quality_score=await self._calculate_quality_score(synthesized_content, source_items)
            )
            
            # Store synthesized knowledge
            self.synthesized_knowledge[synthesized.id] = synthesized
            
            logger.info(f"Synthesized knowledge from {len(source_items)} sources")
            return synthesized
            
        except Exception as e:
            logger.error(f"Error synthesizing knowledge: {str(e)}")
            raise
    
    async def validate_knowledge(
        self,
        knowledge_id: str,
        validation_methods: List[str] = None
    ) -> KnowledgeValidationResult:
        """
        Validate knowledge item for quality assurance
        
        Args:
            knowledge_id: ID of knowledge item to validate
            validation_methods: List of validation methods to use
            
        Returns:
            Validation result
        """
        try:
            if knowledge_id not in self.knowledge_store:
                raise ValueError(f"Knowledge item {knowledge_id} not found")
            
            knowledge_item = self.knowledge_store[knowledge_id]
            
            if validation_methods is None:
                validation_methods = ["consistency", "completeness", "reliability"]
            
            # Perform validation checks
            validation_results = {}
            issues_found = []
            recommendations = []
            
            for method in validation_methods:
                result = await self._perform_validation_check(knowledge_item, method)
                validation_results[method] = result
                
                if not result.get('passed', True):
                    issues_found.extend(result.get('issues', []))
                    recommendations.extend(result.get('recommendations', []))
            
            # Calculate overall validation score
            validation_score = np.mean([
                result.get('score', 0.0) for result in validation_results.values()
            ])
            
            # Determine if knowledge is valid
            is_valid = validation_score >= 0.7 and len(issues_found) == 0
            
            # Create validation result
            validation_result = KnowledgeValidationResult(
                knowledge_id=knowledge_id,
                validation_method=", ".join(validation_methods),
                is_valid=is_valid,
                confidence=ConfidenceLevel.HIGH if validation_score > 0.8 else ConfidenceLevel.MEDIUM,
                validation_score=validation_score,
                issues_found=issues_found,
                recommendations=recommendations
            )
            
            # Cache validation result
            self.validation_cache[knowledge_id] = validation_result
            
            logger.info(f"Validated knowledge {knowledge_id} with score {validation_score}")
            return validation_result
            
        except Exception as e:
            logger.error(f"Error validating knowledge: {str(e)}")
            raise
    
    async def create_knowledge_graph(
        self,
        knowledge_ids: Optional[List[str]] = None
    ) -> KnowledgeGraph:
        """
        Create a knowledge graph from stored knowledge and correlations
        
        Args:
            knowledge_ids: Optional list of specific knowledge IDs to include
            
        Returns:
            Knowledge graph representation
        """
        try:
            if knowledge_ids is None:
                knowledge_ids = list(self.knowledge_store.keys())
            
            # Get nodes (knowledge items)
            nodes = [
                self.knowledge_store[kid] 
                for kid in knowledge_ids 
                if kid in self.knowledge_store
            ]
            
            # Get edges (correlations)
            edges = [
                correlation for correlation in self.correlations.values()
                if any(item_id in knowledge_ids for item_id in correlation.item_ids)
            ]
            
            # Create knowledge graph
            knowledge_graph = KnowledgeGraph(
                nodes=nodes,
                edges=edges,
                metadata={
                    'node_count': len(nodes),
                    'edge_count': len(edges),
                    'creation_method': 'knowledge_synthesis_framework'
                }
            )
            
            logger.info(f"Created knowledge graph with {len(nodes)} nodes and {len(edges)} edges")
            return knowledge_graph
            
        except Exception as e:
            logger.error(f"Error creating knowledge graph: {str(e)}")
            raise
    
    # Private helper methods
    
    def _determine_confidence(self, data: Dict[str, Any]) -> ConfidenceLevel:
        """Determine confidence level based on data quality indicators"""
        confidence_score = 0.0
        
        # Check for quality indicators
        if data.get('peer_reviewed', False):
            confidence_score += 0.3
        if data.get('replicated', False):
            confidence_score += 0.2
        if data.get('sample_size', 0) > 100:
            confidence_score += 0.2
        if data.get('statistical_significance', 0) > 0.95:
            confidence_score += 0.3
        
        if confidence_score >= 0.8:
            return ConfidenceLevel.VERY_HIGH
        elif confidence_score >= 0.6:
            return ConfidenceLevel.HIGH
        elif confidence_score >= 0.4:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW
    
    def _extract_tags(self, data: Dict[str, Any]) -> List[str]:
        """Extract relevant tags from data"""
        tags = []
        
        # Extract from keywords
        if 'keywords' in data:
            tags.extend(data['keywords'])
        
        # Extract from domain
        if 'domain' in data:
            tags.append(data['domain'])
        
        # Extract from methodology
        if 'methodology' in data:
            tags.append(f"method_{data['methodology']}")
        
        return list(set(tags))  # Remove duplicates
    
    async def _identify_relationships(self, knowledge_item: KnowledgeItem):
        """Identify relationships between knowledge item and existing knowledge"""
        relationships = []
        
        for existing_id, existing_item in self.knowledge_store.items():
            if existing_id != knowledge_item.id:
                similarity = await self._calculate_similarity(knowledge_item, existing_item)
                if similarity > 0.5:  # Threshold for relationship
                    relationships.append(existing_id)
        
        knowledge_item.relationships = relationships
    
    async def _calculate_similarity(self, item1: KnowledgeItem, item2: KnowledgeItem) -> float:
        """Calculate similarity between two knowledge items"""
        # Simple similarity based on tags and content overlap
        tag_overlap = len(set(item1.tags) & set(item2.tags)) / max(len(set(item1.tags) | set(item2.tags)), 1)
        
        # Content similarity (simplified)
        content_similarity = 0.0
        if 'title' in item1.content and 'title' in item2.content:
            # Simple word overlap
            words1 = set(item1.content['title'].lower().split())
            words2 = set(item2.content['title'].lower().split())
            content_similarity = len(words1 & words2) / max(len(words1 | words2), 1)
        
        return (tag_overlap + content_similarity) / 2
    
    async def _analyze_correlation(self, id1: str, id2: str) -> Optional[KnowledgeCorrelation]:
        """Analyze correlation between two knowledge items"""
        item1 = self.knowledge_store.get(id1)
        item2 = self.knowledge_store.get(id2)
        
        if not item1 or not item2:
            return None
        
        # Calculate correlation strength
        similarity = await self._calculate_similarity(item1, item2)
        
        if similarity > 0.3:
            correlation = KnowledgeCorrelation(
                id=f"corr_{id1}_{id2}",
                item_ids=[id1, id2],
                correlation_type="similarity",
                strength=similarity,
                confidence=ConfidenceLevel.MEDIUM if similarity > 0.6 else ConfidenceLevel.LOW,
                description=f"Correlation between {item1.knowledge_type.value} and {item2.knowledge_type.value}",
                discovered_at=datetime.now()
            )
            return correlation
        
        return None
    
    async def _perform_synthesis(self, source_items: List[KnowledgeItem], method: str) -> Dict[str, Any]:
        """Perform knowledge synthesis using specified method"""
        if method == "aggregation":
            return await self._aggregate_synthesis(source_items)
        elif method == "integration":
            return await self._integration_synthesis(source_items)
        else:  # default
            return await self._default_synthesis(source_items)
    
    async def _default_synthesis(self, source_items: List[KnowledgeItem]) -> Dict[str, Any]:
        """Default synthesis method"""
        return {
            'synthesis_type': 'default',
            'source_count': len(source_items),
            'combined_content': {
                'findings': [item.content for item in source_items],
                'confidence_levels': [item.confidence.value for item in source_items],
                'sources': [item.source for item in source_items]
            },
            'synthesis_timestamp': datetime.now().isoformat()
        }
    
    async def _aggregate_synthesis(self, source_items: List[KnowledgeItem]) -> Dict[str, Any]:
        """Aggregation-based synthesis"""
        aggregated_data = {}
        
        # Aggregate numerical data
        for item in source_items:
            for key, value in item.content.items():
                if isinstance(value, (int, float)):
                    if key not in aggregated_data:
                        aggregated_data[key] = []
                    aggregated_data[key].append(value)
        
        # Calculate statistics
        statistics = {}
        for key, values in aggregated_data.items():
            statistics[key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'count': len(values)
            }
        
        return {
            'synthesis_type': 'aggregation',
            'statistics': statistics,
            'source_count': len(source_items)
        }
    
    async def _integration_synthesis(self, source_items: List[KnowledgeItem]) -> Dict[str, Any]:
        """Integration-based synthesis"""
        integrated_content = {
            'research_findings': [],
            'experimental_results': [],
            'prototype_insights': [],
            'validation_outcomes': []
        }
        
        # Categorize content by type
        for item in source_items:
            if item.knowledge_type == KnowledgeType.RESEARCH_FINDING:
                integrated_content['research_findings'].append(item.content)
            elif item.knowledge_type == KnowledgeType.EXPERIMENTAL_RESULT:
                integrated_content['experimental_results'].append(item.content)
            elif item.knowledge_type == KnowledgeType.PROTOTYPE_INSIGHT:
                integrated_content['prototype_insights'].append(item.content)
            elif item.knowledge_type == KnowledgeType.VALIDATION_OUTCOME:
                integrated_content['validation_outcomes'].append(item.content)
        
        return {
            'synthesis_type': 'integration',
            'integrated_content': integrated_content,
            'cross_references': await self._find_cross_references(source_items)
        }
    
    async def _find_cross_references(self, source_items: List[KnowledgeItem]) -> List[Dict[str, Any]]:
        """Find cross-references between source items"""
        cross_refs = []
        
        for i, item1 in enumerate(source_items):
            for item2 in source_items[i+1:]:
                if item1.id in item2.relationships or item2.id in item1.relationships:
                    cross_refs.append({
                        'item1_id': item1.id,
                        'item2_id': item2.id,
                        'relationship_type': 'direct_reference'
                    })
        
        return cross_refs
    
    async def _generate_insights(self, source_items: List[KnowledgeItem], synthesized_content: Dict[str, Any]) -> List[str]:
        """Generate insights from synthesized knowledge"""
        insights = []
        
        # Generate insights based on synthesis type
        if synthesized_content.get('synthesis_type') == 'aggregation':
            statistics = synthesized_content.get('statistics', {})
            for key, stats in statistics.items():
                if stats['std'] > stats['mean'] * 0.5:  # High variability
                    insights.append(f"High variability observed in {key} across sources")
                if stats['count'] > 5:  # Sufficient data
                    insights.append(f"Consistent pattern identified in {key} across {stats['count']} sources")
        
        elif synthesized_content.get('synthesis_type') == 'integration':
            # Generate insights for integration synthesis
            integrated_content = synthesized_content.get('integrated_content', {})
            
            # Count different types of content
            research_count = len(integrated_content.get('research_findings', []))
            experiment_count = len(integrated_content.get('experimental_results', []))
            
            if research_count > 0:
                insights.append(f"Integrated {research_count} research findings providing theoretical foundation")
            
            if experiment_count > 0:
                insights.append(f"Incorporated {experiment_count} experimental results for empirical validation")
            
            # Check for cross-references
            cross_refs = synthesized_content.get('cross_references', [])
            if cross_refs:
                insights.append(f"Identified {len(cross_refs)} cross-references between knowledge sources")
        
        # Generate insights based on source diversity
        source_types = set(item.knowledge_type for item in source_items)
        if len(source_types) > 2:
            insights.append("Multi-modal knowledge synthesis combining research, experiments, and validation")
        elif len(source_types) == 2:
            types_list = [t.value.replace('_', ' ') for t in source_types]
            insights.append(f"Cross-domain synthesis combining {' and '.join(types_list)}")
        
        # Generate insights based on confidence levels
        high_confidence_count = sum(1 for item in source_items if item.confidence in [ConfidenceLevel.HIGH, ConfidenceLevel.VERY_HIGH])
        if high_confidence_count / len(source_items) > 0.7:
            insights.append("High confidence synthesis based on reliable sources")
        elif high_confidence_count / len(source_items) < 0.3:
            insights.append("Lower confidence synthesis - additional validation recommended")
        
        # Generate insights based on source count
        if len(source_items) >= 5:
            insights.append(f"Comprehensive synthesis from {len(source_items)} diverse knowledge sources")
        elif len(source_items) >= 2:
            insights.append(f"Synthesis combining {len(source_items)} complementary knowledge sources")
        
        # Generate insights based on tags/keywords overlap
        all_tags = []
        for item in source_items:
            all_tags.extend(item.tags)
        
        if all_tags:
            from collections import Counter
            tag_counts = Counter(all_tags)
            common_tags = [tag for tag, count in tag_counts.items() if count > 1]
            
            if common_tags:
                insights.append(f"Common themes identified: {', '.join(common_tags[:3])}")
        
        # Ensure we always have at least one insight
        if not insights:
            insights.append("Knowledge synthesis completed successfully with integrated findings")
        
        return insights
    
    def _calculate_synthesis_confidence(self, source_items: List[KnowledgeItem]) -> ConfidenceLevel:
        """Calculate confidence level for synthesized knowledge"""
        confidence_scores = {
            ConfidenceLevel.LOW: 0.25,
            ConfidenceLevel.MEDIUM: 0.5,
            ConfidenceLevel.HIGH: 0.75,
            ConfidenceLevel.VERY_HIGH: 1.0
        }
        
        avg_confidence = np.mean([confidence_scores[item.confidence] for item in source_items])
        
        if avg_confidence >= 0.8:
            return ConfidenceLevel.VERY_HIGH
        elif avg_confidence >= 0.6:
            return ConfidenceLevel.HIGH
        elif avg_confidence >= 0.4:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW
    
    async def _calculate_quality_score(self, synthesized_content: Dict[str, Any], source_items: List[KnowledgeItem]) -> float:
        """Calculate quality score for synthesized knowledge"""
        quality_factors = []
        
        # Source diversity
        source_types = len(set(item.knowledge_type for item in source_items))
        quality_factors.append(min(source_types / 4, 1.0))  # Normalize to max 4 types
        
        # Source count
        source_count_score = min(len(source_items) / 10, 1.0)  # Normalize to max 10 sources
        quality_factors.append(source_count_score)
        
        # Content completeness
        content_completeness = len(synthesized_content) / 10  # Assume 10 fields is complete
        quality_factors.append(min(content_completeness, 1.0))
        
        # Average confidence
        confidence_scores = {
            ConfidenceLevel.LOW: 0.25,
            ConfidenceLevel.MEDIUM: 0.5,
            ConfidenceLevel.HIGH: 0.75,
            ConfidenceLevel.VERY_HIGH: 1.0
        }
        avg_confidence = np.mean([confidence_scores[item.confidence] for item in source_items])
        quality_factors.append(avg_confidence)
        
        return np.mean(quality_factors)
    
    async def _perform_validation_check(self, knowledge_item: KnowledgeItem, method: str) -> Dict[str, Any]:
        """Perform specific validation check"""
        if method == "consistency":
            return await self._check_consistency(knowledge_item)
        elif method == "completeness":
            return await self._check_completeness(knowledge_item)
        elif method == "reliability":
            return await self._check_reliability(knowledge_item)
        else:
            return {"passed": True, "score": 0.5, "issues": [], "recommendations": []}
    
    async def _check_consistency(self, knowledge_item: KnowledgeItem) -> Dict[str, Any]:
        """Check knowledge consistency"""
        issues = []
        recommendations = []
        score = 1.0
        
        # Check for internal consistency
        content = knowledge_item.content
        
        # Check for contradictory information
        if 'results' in content and 'conclusions' in content:
            # Simplified consistency check
            if len(content.get('results', [])) == 0 and len(content.get('conclusions', [])) > 0:
                issues.append("Conclusions present without supporting results")
                recommendations.append("Verify conclusions are supported by results")
                score -= 0.3
        
        # Check timestamp consistency
        if knowledge_item.timestamp > datetime.now():
            issues.append("Future timestamp detected")
            recommendations.append("Verify timestamp accuracy")
            score -= 0.2
        
        return {
            "passed": len(issues) == 0,
            "score": max(score, 0.0),
            "issues": issues,
            "recommendations": recommendations
        }
    
    async def _check_completeness(self, knowledge_item: KnowledgeItem) -> Dict[str, Any]:
        """Check knowledge completeness"""
        issues = []
        recommendations = []
        score = 1.0
        
        required_fields = ['title', 'description', 'methodology']
        missing_fields = [field for field in required_fields if field not in knowledge_item.content]
        
        if missing_fields:
            issues.extend([f"Missing required field: {field}" for field in missing_fields])
            recommendations.extend([f"Add {field} information" for field in missing_fields])
            score -= 0.2 * len(missing_fields)
        
        # Check content depth
        if isinstance(knowledge_item.content.get('description'), str):
            if len(knowledge_item.content['description']) < 50:
                issues.append("Description too brief")
                recommendations.append("Provide more detailed description")
                score -= 0.1
        
        return {
            "passed": len(issues) == 0,
            "score": max(score, 0.0),
            "issues": issues,
            "recommendations": recommendations
        }
    
    async def _check_reliability(self, knowledge_item: KnowledgeItem) -> Dict[str, Any]:
        """Check knowledge reliability"""
        issues = []
        recommendations = []
        score = 1.0
        
        # Check source reliability
        if not knowledge_item.source or knowledge_item.source == 'unknown':
            issues.append("Unknown or missing source")
            recommendations.append("Verify and document knowledge source")
            score -= 0.3
        
        # Check confidence level
        if knowledge_item.confidence == ConfidenceLevel.LOW:
            issues.append("Low confidence level")
            recommendations.append("Seek additional validation or higher quality sources")
            score -= 0.2
        
        # Check for validation history
        if knowledge_item.id not in self.validation_cache:
            recommendations.append("Consider additional validation methods")
            score -= 0.1
        
        return {
            "passed": len(issues) == 0,
            "score": max(score, 0.0),
            "issues": issues,
            "recommendations": recommendations
        }