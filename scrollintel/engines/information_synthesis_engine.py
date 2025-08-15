"""
Information Synthesis Engine for Crisis Leadership Excellence

This engine provides rapid processing of incomplete and conflicting information,
implements information prioritization and filtering, and builds confidence scoring
with uncertainty management for crisis decision-making.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Set
from collections import defaultdict
import statistics
import re

from ..models.information_synthesis_models import (
    InformationItem, InformationConflict, SynthesizedInformation,
    FilterCriteria, UncertaintyAssessment, SynthesisRequest,
    SynthesisMetrics, InformationSource, InformationPriority,
    ConflictType
)
# Crisis model will be imported when needed


class InformationSynthesisEngine:
    """
    Engine for synthesizing information during crisis situations
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.information_store: Dict[str, InformationItem] = {}
        self.conflict_store: Dict[str, InformationConflict] = {}
        self.synthesis_cache: Dict[str, SynthesizedInformation] = {}
        
        # Configuration
        self.max_processing_time = 30  # seconds
        self.confidence_threshold = 0.7
        self.reliability_threshold = 0.6
        
    async def synthesize_information(
        self,
        request: SynthesisRequest
    ) -> SynthesizedInformation:
        """
        Main synthesis method that processes information items
        """
        start_time = datetime.now()
        
        try:
            # Retrieve information items
            items = await self._retrieve_information_items(request.information_items)
            
            # Apply filtering
            filtered_items = await self._filter_information(items, request.filter_criteria)
            
            # Detect and resolve conflicts
            conflicts = await self._detect_conflicts(filtered_items)
            resolved_items = await self._resolve_conflicts(filtered_items, conflicts)
            
            # Prioritize information
            prioritized_items = await self._prioritize_information(resolved_items)
            
            # Generate synthesis
            synthesis = await self._generate_synthesis(
                prioritized_items, conflicts, request
            )
            
            # Calculate uncertainty
            uncertainty = await self._assess_uncertainty(
                prioritized_items, conflicts, synthesis
            )
            synthesis.uncertainty_factors = uncertainty.key_uncertainties
            
            # Cache result
            self.synthesis_cache[synthesis.id] = synthesis
            
            processing_time = (datetime.now() - start_time).total_seconds()
            self.logger.info(f"Information synthesis completed in {processing_time:.2f}s")
            
            return synthesis
            
        except Exception as e:
            self.logger.error(f"Information synthesis failed: {str(e)}")
            raise
    
    async def _retrieve_information_items(
        self,
        item_ids: List[str]
    ) -> List[InformationItem]:
        """Retrieve information items from storage"""
        items = []
        for item_id in item_ids:
            if item_id in self.information_store:
                items.append(self.information_store[item_id])
            else:
                self.logger.warning(f"Information item {item_id} not found")
        return items
    
    async def _filter_information(
        self,
        items: List[InformationItem],
        criteria: Optional[FilterCriteria]
    ) -> List[InformationItem]:
        """Filter information based on criteria"""
        if not criteria:
            return items
        
        filtered_items = []
        
        for item in items:
            # Check confidence threshold
            if item.confidence_score < criteria.min_confidence:
                continue
                
            # Check reliability threshold
            if item.reliability_score < criteria.min_reliability:
                continue
            
            # Check source requirements
            if criteria.required_sources and item.source not in criteria.required_sources:
                continue
                
            if criteria.excluded_sources and item.source in criteria.excluded_sources:
                continue
            
            # Check priority threshold
            priority_values = {
                InformationPriority.CRITICAL: 4,
                InformationPriority.HIGH: 3,
                InformationPriority.MEDIUM: 2,
                InformationPriority.LOW: 1
            }
            
            if priority_values[item.priority] < priority_values[criteria.priority_threshold]:
                continue
            
            # Check time window
            if criteria.time_window_hours:
                cutoff_time = datetime.now() - timedelta(hours=criteria.time_window_hours)
                if item.timestamp < cutoff_time:
                    continue
            
            # Check tags
            if criteria.required_tags:
                if not all(tag in item.tags for tag in criteria.required_tags):
                    continue
            
            if criteria.excluded_tags:
                if any(tag in item.tags for tag in criteria.excluded_tags):
                    continue
            
            filtered_items.append(item)
        
        self.logger.info(f"Filtered {len(items)} items to {len(filtered_items)}")
        return filtered_items
    
    async def _detect_conflicts(
        self,
        items: List[InformationItem]
    ) -> List[InformationConflict]:
        """Detect conflicts between information items"""
        conflicts = []
        
        # Group items by content similarity for conflict detection
        content_groups = await self._group_similar_content(items)
        
        for group in content_groups:
            if len(group) > 1:
                group_conflicts = await self._analyze_group_conflicts(group)
                conflicts.extend(group_conflicts)
        
        # Detect timing conflicts
        timing_conflicts = await self._detect_timing_conflicts(items)
        conflicts.extend(timing_conflicts)
        
        # Detect source reliability conflicts
        reliability_conflicts = await self._detect_reliability_conflicts(items)
        conflicts.extend(reliability_conflicts)
        
        self.logger.info(f"Detected {len(conflicts)} conflicts")
        return conflicts
    
    async def _group_similar_content(
        self,
        items: List[InformationItem]
    ) -> List[List[InformationItem]]:
        """Group items with similar content for conflict analysis"""
        # Simple keyword-based grouping (can be enhanced with NLP)
        groups = defaultdict(list)
        
        for item in items:
            # Extract key terms from content
            key_terms = self._extract_key_terms(item.content)
            if key_terms:
                group_key = tuple(sorted(key_terms[:3]))  # Use top 3 terms as group key
                groups[group_key].append(item)
            else:
                # If no key terms, group by first few words
                words = item.content.lower().split()[:3]
                group_key = tuple(words)
                groups[group_key].append(item)
        
        # Also create groups based on topic similarity (system-related items)
        system_items = [item for item in items if 'system' in item.content.lower()]
        if len(system_items) > 1:
            groups[('system_topic',)] = system_items
        
        return [group for group in groups.values() if len(group) > 1]
    
    def _extract_key_terms(self, content: str) -> List[str]:
        """Extract key terms from content (simplified implementation)"""
        # Remove common words and extract meaningful terms
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        words = re.findall(r'\b\w+\b', content.lower())
        key_terms = [word for word in words if word not in common_words and len(word) > 3]
        return key_terms[:10]  # Return top 10 terms
    
    async def _analyze_group_conflicts(
        self,
        group: List[InformationItem]
    ) -> List[InformationConflict]:
        """Analyze conflicts within a group of similar items"""
        conflicts = []
        
        for i, item1 in enumerate(group):
            for item2 in group[i+1:]:
                conflict = await self._compare_items_for_conflict(item1, item2)
                if conflict:
                    conflicts.append(conflict)
        
        return conflicts
    
    async def _compare_items_for_conflict(
        self,
        item1: InformationItem,
        item2: InformationItem
    ) -> Optional[InformationConflict]:
        """Compare two items for potential conflicts"""
        # Check for contradictory information
        if await self._are_contradictory(item1.content, item2.content):
            return InformationConflict(
                conflict_type=ConflictType.CONTRADICTORY_FACTS,
                conflicting_items=[item1.id, item2.id],
                description=f"Contradictory information between sources {item1.source.value} and {item2.source.value}",
                severity=0.8
            )
        
        # Check for timing discrepancies
        time_diff = abs((item1.timestamp - item2.timestamp).total_seconds())
        if time_diff > 3600 and self._similar_content(item1.content, item2.content):  # 1 hour
            return InformationConflict(
                conflict_type=ConflictType.TIMING_DISCREPANCY,
                conflicting_items=[item1.id, item2.id],
                description=f"Timing discrepancy of {time_diff/3600:.1f} hours for similar content",
                severity=0.5
            )
        
        return None
    
    async def _are_contradictory(self, content1: str, content2: str) -> bool:
        """Check if two pieces of content are contradictory (simplified)"""
        # Look for negation patterns
        negation_words = ['not', 'no', 'never', 'none', 'nothing', 'false', 'incorrect', 'failed', 'failure']
        positive_words = ['operational', 'working', 'running', 'normal', 'healthy', 'up', 'active']
        negative_words = ['failed', 'down', 'broken', 'error', 'issue', 'problem', 'outage', 'disruption']
        
        content1_lower = content1.lower()
        content2_lower = content2.lower()
        
        # Check for explicit negation patterns
        content1_has_negation = any(word in content1_lower for word in negation_words)
        content2_has_negation = any(word in content2_lower for word in negation_words)
        
        if content1_has_negation != content2_has_negation:
            # Check if they share key terms
            terms1 = set(self._extract_key_terms(content1))
            terms2 = set(self._extract_key_terms(content2))
            shared_terms = terms1.intersection(terms2)
            
            if len(shared_terms) >= 1:
                return True
        
        # Check for positive vs negative sentiment about same topic
        content1_has_positive = any(word in content1_lower for word in positive_words)
        content1_has_negative = any(word in content1_lower for word in negative_words)
        content2_has_positive = any(word in content2_lower for word in positive_words)
        content2_has_negative = any(word in content2_lower for word in negative_words)
        
        # If one is clearly positive and other is clearly negative about similar topics
        if ((content1_has_positive and not content1_has_negative) and 
            (content2_has_negative and not content2_has_positive)) or \
           ((content2_has_positive and not content2_has_negative) and 
            (content1_has_negative and not content1_has_positive)):
            
            # Check if they share key terms (same topic)
            terms1 = set(self._extract_key_terms(content1))
            terms2 = set(self._extract_key_terms(content2))
            shared_terms = terms1.intersection(terms2)
            
            if len(shared_terms) >= 1:
                return True
        
        # Special case: check for "operational" vs "not operational" pattern
        if 'operational' in content1_lower and 'operational' in content2_lower:
            if ('not operational' in content2_lower and 'not operational' not in content1_lower) or \
               ('not operational' in content1_lower and 'not operational' not in content2_lower):
                return True
        
        return False
    
    def _similar_content(self, content1: str, content2: str) -> bool:
        """Check if two pieces of content are similar"""
        terms1 = set(self._extract_key_terms(content1))
        terms2 = set(self._extract_key_terms(content2))
        
        if not terms1 or not terms2:
            return False
        
        intersection = terms1.intersection(terms2)
        union = terms1.union(terms2)
        
        similarity = len(intersection) / len(union) if union else 0
        return similarity > 0.3
    
    async def _detect_timing_conflicts(
        self,
        items: List[InformationItem]
    ) -> List[InformationConflict]:
        """Detect timing-based conflicts"""
        conflicts = []
        
        # Group items by time windows
        time_groups = defaultdict(list)
        for item in items:
            # Group by hour
            hour_key = item.timestamp.replace(minute=0, second=0, microsecond=0)
            time_groups[hour_key].append(item)
        
        # Look for conflicting information in same time window
        for time_window, group_items in time_groups.items():
            if len(group_items) > 1:
                group_conflicts = await self._analyze_group_conflicts(group_items)
                conflicts.extend(group_conflicts)
        
        return conflicts
    
    async def _detect_reliability_conflicts(
        self,
        items: List[InformationItem]
    ) -> List[InformationConflict]:
        """Detect conflicts based on source reliability"""
        conflicts = []
        
        # Group by content similarity
        content_groups = await self._group_similar_content(items)
        
        for group in content_groups:
            if len(group) > 1:
                # Check for significant reliability differences
                reliabilities = [item.reliability_score for item in group]
                if max(reliabilities) - min(reliabilities) > 0.4:
                    high_rel_items = [item for item in group if item.reliability_score > 0.7]
                    low_rel_items = [item for item in group if item.reliability_score < 0.4]
                    
                    if high_rel_items and low_rel_items:
                        conflict = InformationConflict(
                            conflict_type=ConflictType.SOURCE_RELIABILITY,
                            conflicting_items=[item.id for item in group],
                            description="Significant reliability difference between sources for similar content",
                            severity=0.6
                        )
                        conflicts.append(conflict)
        
        return conflicts
    
    async def _resolve_conflicts(
        self,
        items: List[InformationItem],
        conflicts: List[InformationConflict]
    ) -> List[InformationItem]:
        """Resolve conflicts and return cleaned information"""
        resolved_items = items.copy()
        
        for conflict in conflicts:
            resolution_strategy = await self._determine_resolution_strategy(conflict, items)
            
            if resolution_strategy == "prefer_reliable":
                resolved_items = await self._prefer_reliable_sources(conflict, resolved_items)
            elif resolution_strategy == "prefer_recent":
                resolved_items = await self._prefer_recent_information(conflict, resolved_items)
            elif resolution_strategy == "merge_information":
                resolved_items = await self._merge_conflicting_information(conflict, resolved_items)
            
            conflict.resolution_strategy = resolution_strategy
            conflict.resolved = True
            conflict.resolution_timestamp = datetime.now()
        
        return resolved_items
    
    async def _determine_resolution_strategy(
        self,
        conflict: InformationConflict,
        items: List[InformationItem]
    ) -> str:
        """Determine the best strategy to resolve a conflict"""
        conflicting_items = [
            item for item in items if item.id in conflict.conflicting_items
        ]
        
        if conflict.conflict_type == ConflictType.SOURCE_RELIABILITY:
            return "prefer_reliable"
        elif conflict.conflict_type == ConflictType.TIMING_DISCREPANCY:
            return "prefer_recent"
        elif conflict.conflict_type == ConflictType.CONTRADICTORY_FACTS:
            # Check reliability difference
            reliabilities = [item.reliability_score for item in conflicting_items]
            if max(reliabilities) - min(reliabilities) > 0.3:
                return "prefer_reliable"
            else:
                return "merge_information"
        
        return "prefer_reliable"
    
    async def _prefer_reliable_sources(
        self,
        conflict: InformationConflict,
        items: List[InformationItem]
    ) -> List[InformationItem]:
        """Resolve conflict by preferring more reliable sources"""
        conflicting_items = [
            item for item in items if item.id in conflict.conflicting_items
        ]
        
        if not conflicting_items:
            return items
        
        # Find the most reliable item
        most_reliable = max(conflicting_items, key=lambda x: x.reliability_score)
        
        # Remove less reliable items
        items_to_remove = [
            item.id for item in conflicting_items 
            if item.id != most_reliable.id
        ]
        
        return [item for item in items if item.id not in items_to_remove]
    
    async def _prefer_recent_information(
        self,
        conflict: InformationConflict,
        items: List[InformationItem]
    ) -> List[InformationItem]:
        """Resolve conflict by preferring more recent information"""
        conflicting_items = [
            item for item in items if item.id in conflict.conflicting_items
        ]
        
        if not conflicting_items:
            return items
        
        # Find the most recent item
        most_recent = max(conflicting_items, key=lambda x: x.timestamp)
        
        # Remove older items
        items_to_remove = [
            item.id for item in conflicting_items 
            if item.id != most_recent.id
        ]
        
        return [item for item in items if item.id not in items_to_remove]
    
    async def _merge_conflicting_information(
        self,
        conflict: InformationConflict,
        items: List[InformationItem]
    ) -> List[InformationItem]:
        """Merge conflicting information into a single item"""
        conflicting_items = [
            item for item in items if item.id in conflict.conflicting_items
        ]
        
        if len(conflicting_items) < 2:
            return items
        
        # Create merged item
        merged_content = f"MERGED: {' | '.join([item.content for item in conflicting_items])}"
        merged_confidence = statistics.mean([item.confidence_score for item in conflicting_items]) * 0.8  # Reduce confidence due to conflict
        merged_reliability = statistics.mean([item.reliability_score for item in conflicting_items]) * 0.8
        
        merged_item = InformationItem(
            content=merged_content,
            source=conflicting_items[0].source,  # Use first source
            confidence_score=merged_confidence,
            reliability_score=merged_reliability,
            priority=max(conflicting_items, key=lambda x: x.priority.value).priority,
            tags=list(set().union(*[item.tags for item in conflicting_items])),
            verification_status="merged_conflict"
        )
        
        # Remove original conflicting items and add merged item
        filtered_items = [item for item in items if item.id not in conflict.conflicting_items]
        filtered_items.append(merged_item)
        
        return filtered_items
    
    async def _prioritize_information(
        self,
        items: List[InformationItem]
    ) -> List[InformationItem]:
        """Prioritize information based on multiple factors"""
        def priority_score(item: InformationItem) -> float:
            # Base priority score
            priority_values = {
                InformationPriority.CRITICAL: 1.0,
                InformationPriority.HIGH: 0.8,
                InformationPriority.MEDIUM: 0.6,
                InformationPriority.LOW: 0.4
            }
            
            base_score = priority_values[item.priority]
            
            # Adjust for confidence and reliability
            confidence_factor = item.confidence_score
            reliability_factor = item.reliability_score
            
            # Adjust for recency (more recent = higher priority)
            time_diff = (datetime.now() - item.timestamp).total_seconds()
            recency_factor = max(0.1, 1.0 - (time_diff / 86400))  # Decay over 24 hours
            
            return base_score * confidence_factor * reliability_factor * recency_factor
        
        # Sort by priority score (highest first)
        prioritized_items = sorted(items, key=priority_score, reverse=True)
        
        return prioritized_items
    
    async def _generate_synthesis(
        self,
        items: List[InformationItem],
        conflicts: List[InformationConflict],
        request: SynthesisRequest
    ) -> SynthesizedInformation:
        """Generate the final synthesis from processed information"""
        # Extract key findings
        key_findings = await self._extract_key_findings(items)
        
        # Calculate overall confidence
        if items:
            confidence_scores = [item.confidence_score for item in items]
            overall_confidence = statistics.mean(confidence_scores)
            
            # Reduce confidence if there were many conflicts
            conflict_penalty = min(0.3, len(conflicts) * 0.05)
            overall_confidence = max(0.1, overall_confidence - conflict_penalty)
        else:
            overall_confidence = 0.0
        
        # Identify information gaps
        information_gaps = await self._identify_information_gaps(items, request)
        
        # Generate recommendations
        recommendations = await self._generate_recommendations(items, conflicts)
        
        # Calculate priority score
        priority_score = await self._calculate_synthesis_priority(items, request)
        
        synthesis = SynthesizedInformation(
            crisis_id=request.crisis_id,
            key_findings=key_findings,
            confidence_level=overall_confidence,
            information_gaps=information_gaps,
            recommendations=recommendations,
            source_items=[item.id for item in items],
            conflicts_identified=[conflict.id for conflict in conflicts],
            priority_score=priority_score
        )
        
        return synthesis
    
    async def _extract_key_findings(
        self,
        items: List[InformationItem]
    ) -> List[str]:
        """Extract key findings from information items"""
        findings = []
        
        # Group items by priority and extract findings
        critical_items = [item for item in items if item.priority == InformationPriority.CRITICAL]
        high_items = [item for item in items if item.priority == InformationPriority.HIGH]
        
        # Process critical items first
        for item in critical_items[:5]:  # Top 5 critical items
            if item.confidence_score > 0.7:
                findings.append(f"CRITICAL: {item.content[:200]}...")
        
        # Process high priority items
        for item in high_items[:3]:  # Top 3 high priority items
            if item.confidence_score > 0.6:
                findings.append(f"HIGH: {item.content[:150]}...")
        
        # Add summary finding if we have enough items
        if len(items) > 10:
            findings.append(f"Analysis based on {len(items)} information sources with average confidence {statistics.mean([item.confidence_score for item in items]):.2f}")
        
        return findings[:10]  # Limit to 10 key findings
    
    async def _identify_information_gaps(
        self,
        items: List[InformationItem],
        request: SynthesisRequest
    ) -> List[str]:
        """Identify gaps in available information"""
        gaps = []
        
        # Check source diversity
        sources_present = set(item.source for item in items)
        all_sources = set(InformationSource)
        missing_sources = all_sources - sources_present
        
        if missing_sources:
            gaps.append(f"Missing information from sources: {', '.join([s.value for s in missing_sources])}")
        
        # Check temporal coverage
        if items:
            timestamps = [item.timestamp for item in items]
            time_span = max(timestamps) - min(timestamps)
            
            if time_span.total_seconds() < 3600:  # Less than 1 hour of coverage
                gaps.append("Limited temporal coverage - information spans less than 1 hour")
        
        # Check confidence levels
        low_confidence_count = len([item for item in items if item.confidence_score < 0.5])
        if low_confidence_count > len(items) * 0.3:
            gaps.append("High proportion of low-confidence information")
        
        # Check verification status
        unverified_count = len([item for item in items if item.verification_status == "unverified"])
        if unverified_count > len(items) * 0.5:
            gaps.append("Many information items remain unverified")
        
        return gaps
    
    async def _generate_recommendations(
        self,
        items: List[InformationItem],
        conflicts: List[InformationConflict]
    ) -> List[str]:
        """Generate recommendations based on synthesis"""
        recommendations = []
        
        # Recommendations based on conflicts
        if conflicts:
            unresolved_conflicts = [c for c in conflicts if not c.resolved]
            if unresolved_conflicts:
                recommendations.append(f"Resolve {len(unresolved_conflicts)} remaining information conflicts")
        
        # Recommendations based on information quality
        low_confidence_items = [item for item in items if item.confidence_score < 0.5]
        if low_confidence_items:
            recommendations.append(f"Verify {len(low_confidence_items)} low-confidence information items")
        
        # Recommendations based on source diversity
        sources_present = set(item.source for item in items)
        if len(sources_present) < 3:
            recommendations.append("Seek additional information sources for better coverage")
        
        # Recommendations based on critical information
        critical_items = [item for item in items if item.priority == InformationPriority.CRITICAL]
        if critical_items:
            recommendations.append(f"Immediate attention required for {len(critical_items)} critical information items")
        
        return recommendations
    
    async def _calculate_synthesis_priority(
        self,
        items: List[InformationItem],
        request: SynthesisRequest
    ) -> float:
        """Calculate overall priority score for the synthesis"""
        if not items:
            return 0.0
        
        # Base priority from request
        priority_values = {
            InformationPriority.CRITICAL: 1.0,
            InformationPriority.HIGH: 0.8,
            InformationPriority.MEDIUM: 0.6,
            InformationPriority.LOW: 0.4
        }
        
        base_priority = priority_values[request.urgency_level]
        
        # Adjust based on item priorities
        item_priorities = [priority_values[item.priority] for item in items]
        avg_item_priority = statistics.mean(item_priorities)
        
        # Adjust based on confidence
        avg_confidence = statistics.mean([item.confidence_score for item in items])
        
        # Calculate final priority
        final_priority = (base_priority * 0.4 + avg_item_priority * 0.4 + avg_confidence * 0.2)
        
        return min(1.0, final_priority)
    
    async def _assess_uncertainty(
        self,
        items: List[InformationItem],
        conflicts: List[InformationConflict],
        synthesis: SynthesizedInformation
    ) -> UncertaintyAssessment:
        """Assess uncertainty in the synthesized information"""
        if not items:
            return UncertaintyAssessment(overall_uncertainty=1.0)
        
        # Calculate information completeness
        total_possible_sources = len(InformationSource)
        actual_sources = len(set(item.source for item in items))
        information_completeness = actual_sources / total_possible_sources
        
        # Calculate source diversity
        source_counts = defaultdict(int)
        for item in items:
            source_counts[item.source] += 1
        
        # Higher diversity = lower uncertainty
        max_count = max(source_counts.values()) if source_counts else 1
        total_items = len(items)
        source_diversity = 1.0 - (max_count / total_items) if total_items > 0 else 0.0
        
        # Calculate temporal consistency
        if len(items) > 1:
            timestamps = [item.timestamp for item in items]
            time_span = (max(timestamps) - min(timestamps)).total_seconds()
            # Normalize to 0-1 scale (24 hours = full consistency)
            temporal_consistency = min(1.0, time_span / 86400)
        else:
            temporal_consistency = 0.5
        
        # Calculate conflict resolution confidence
        total_conflicts = len(conflicts)
        resolved_conflicts = len([c for c in conflicts if c.resolved])
        conflict_resolution_confidence = resolved_conflicts / total_conflicts if total_conflicts > 0 else 1.0
        
        # Calculate overall uncertainty
        confidence_factor = 1.0 - statistics.mean([item.confidence_score for item in items])
        conflict_factor = min(0.5, total_conflicts * 0.1)
        completeness_factor = 1.0 - information_completeness
        
        overall_uncertainty = (confidence_factor * 0.4 + conflict_factor * 0.3 + completeness_factor * 0.3)
        overall_uncertainty = max(0.0, min(1.0, overall_uncertainty))
        
        # Identify key uncertainties
        key_uncertainties = []
        
        if overall_uncertainty > 0.7:
            key_uncertainties.append("High overall uncertainty due to low confidence information")
        
        if information_completeness < 0.5:
            key_uncertainties.append("Incomplete information coverage across sources")
        
        if total_conflicts > len(items) * 0.2:
            key_uncertainties.append("High number of information conflicts detected")
        
        if source_diversity < 0.3:
            key_uncertainties.append("Low source diversity - information may be biased")
        
        # Generate mitigation strategies
        mitigation_strategies = []
        
        if information_completeness < 0.7:
            mitigation_strategies.append("Seek additional information sources")
        
        if conflict_resolution_confidence < 0.8:
            mitigation_strategies.append("Manually review and resolve remaining conflicts")
        
        if confidence_factor > 0.5:
            mitigation_strategies.append("Verify low-confidence information through additional sources")
        
        return UncertaintyAssessment(
            overall_uncertainty=overall_uncertainty,
            information_completeness=information_completeness,
            source_diversity=source_diversity,
            temporal_consistency=temporal_consistency,
            conflict_resolution_confidence=conflict_resolution_confidence,
            key_uncertainties=key_uncertainties,
            mitigation_strategies=mitigation_strategies
        )
    
    async def add_information_item(self, item: InformationItem) -> str:
        """Add a new information item to the store"""
        self.information_store[item.id] = item
        self.logger.info(f"Added information item {item.id}")
        return item.id
    
    async def get_synthesis_metrics(self, synthesis_id: str) -> Optional[SynthesisMetrics]:
        """Get metrics for a synthesis process"""
        if synthesis_id not in self.synthesis_cache:
            return None
        
        synthesis = self.synthesis_cache[synthesis_id]
        
        # Calculate metrics (simplified implementation)
        return SynthesisMetrics(
            processing_time_seconds=1.0,  # Would be tracked during actual processing
            items_processed=len(synthesis.source_items),
            items_filtered_out=0,  # Would be tracked during filtering
            conflicts_detected=len(synthesis.conflicts_identified),
            conflicts_resolved=len([c for c in self.conflict_store.values() if c.resolved]),
            confidence_improvement=0.1,  # Would be calculated based on before/after
            uncertainty_reduction=0.1,  # Would be calculated based on before/after
            synthesis_quality_score=synthesis.confidence_level
        )