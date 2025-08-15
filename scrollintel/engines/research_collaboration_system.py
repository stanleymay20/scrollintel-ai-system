"""
Research Collaboration System for Autonomous Innovation Lab

This system provides autonomous research collaboration, knowledge sharing,
and synergy identification capabilities.
"""

import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Set, Tuple
import logging
from dataclasses import asdict
import numpy as np
from collections import defaultdict

from ..models.research_coordination_models import (
    ResearchProject, ResearchCollaboration, KnowledgeAsset, ResearchSynergy,
    CollaborationType, ProjectStatus
)
from .base_engine import BaseEngine


class ResearchCollaborationSystem(BaseEngine):
    """
    Autonomous research collaboration system that handles:
    - Research collaboration coordination
    - Knowledge sharing and integration
    - Synergy identification and exploitation
    """
    
    def __init__(self):
        from ..engines.base_engine import EngineCapability
        super().__init__(
            engine_id="research_collaboration_system",
            name="Research Collaboration System",
            capabilities=[EngineCapability.DATA_ANALYSIS]
        )
        self.logger = logging.getLogger(__name__)
        self.active_collaborations: Dict[str, ResearchCollaboration] = {}
        self.knowledge_assets: Dict[str, KnowledgeAsset] = {}
        self.identified_synergies: Dict[str, ResearchSynergy] = {}
        self.collaboration_patterns: Dict[str, Any] = {}
        
        # Initialize collaboration scoring weights
        self.synergy_weights = {
            "domain_similarity": 0.3,
            "resource_complementarity": 0.25,
            "methodology_alignment": 0.2,
            "timeline_compatibility": 0.15,
            "knowledge_gap_overlap": 0.1
        }
    
    async def initialize(self) -> None:
        """Initialize the research collaboration system"""
        self.logger.info("Initializing Research Collaboration System")
        # Initialize any required resources
        pass
    
    async def process(self, input_data: Any, parameters: Dict[str, Any] = None) -> Any:
        """Process research collaboration requests"""
        # This could be used for batch processing of collaborations
        return {"status": "processed", "data": input_data}
    
    async def cleanup(self) -> None:
        """Clean up research collaboration system resources"""
        self.logger.info("Cleaning up Research Collaboration System")
        # Clean up any resources
        pass
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of the research collaboration system"""
        return {
            "healthy": True,
            "active_collaborations": len(self.active_collaborations),
            "knowledge_assets": len(self.knowledge_assets),
            "identified_synergies": len(self.identified_synergies),
            "status": self.status.value
        }
    
    async def identify_collaboration_opportunities(
        self,
        projects: List[ResearchProject],
        min_synergy_score: float = 0.6
    ) -> List[ResearchSynergy]:
        """
        Identify collaboration opportunities between research projects
        
        Args:
            projects: List of research projects to analyze
            min_synergy_score: Minimum synergy score threshold
            
        Returns:
            List of identified synergies
        """
        try:
            synergies = []
            
            # Analyze all project pairs
            for i, project1 in enumerate(projects):
                for j, project2 in enumerate(projects[i+1:], i+1):
                    synergy = await self._analyze_project_synergy(project1, project2)
                    
                    if synergy.overall_score >= min_synergy_score:
                        synergies.append(synergy)
                        self.identified_synergies[synergy.id] = synergy
            
            # Sort by synergy score
            synergies.sort(key=lambda s: s.overall_score, reverse=True)
            
            self.logger.info(f"Identified {len(synergies)} collaboration opportunities")
            return synergies
            
        except Exception as e:
            self.logger.error(f"Error identifying collaboration opportunities: {str(e)}")
            return []
    
    async def _analyze_project_synergy(
        self,
        project1: ResearchProject,
        project2: ResearchProject
    ) -> ResearchSynergy:
        """Analyze synergy potential between two projects"""
        synergy = ResearchSynergy(
            project_ids=[project1.id, project2.id]
        )
        
        # Calculate individual synergy components
        domain_score = self._calculate_domain_similarity(project1, project2)
        resource_score = self._calculate_resource_complementarity(project1, project2)
        methodology_score = self._calculate_methodology_alignment(project1, project2)
        timeline_score = self._calculate_timeline_compatibility(project1, project2)
        knowledge_score = self._calculate_knowledge_gap_overlap(project1, project2)
        
        # Calculate weighted overall score
        synergy.overall_score = (
            domain_score * self.synergy_weights["domain_similarity"] +
            resource_score * self.synergy_weights["resource_complementarity"] +
            methodology_score * self.synergy_weights["methodology_alignment"] +
            timeline_score * self.synergy_weights["timeline_compatibility"] +
            knowledge_score * self.synergy_weights["knowledge_gap_overlap"]
        )
        
        # Set component scores
        synergy.potential_score = (domain_score + knowledge_score) / 2
        synergy.feasibility_score = (resource_score + timeline_score) / 2
        synergy.impact_score = methodology_score
        
        # Identify specific synergy details
        synergy.complementary_strengths = self._identify_complementary_strengths(project1, project2)
        synergy.shared_challenges = self._identify_shared_challenges(project1, project2)
        synergy.collaboration_opportunities = self._identify_collaboration_opportunities_specific(project1, project2)
        
        # Generate recommendations
        synergy.recommended_actions = self._generate_collaboration_recommendations(project1, project2, synergy)
        synergy.estimated_benefits = self._estimate_collaboration_benefits(project1, project2)
        synergy.implementation_complexity = self._assess_implementation_complexity(project1, project2)
        
        return synergy
    
    def _calculate_domain_similarity(self, project1: ResearchProject, project2: ResearchProject) -> float:
        """Calculate domain similarity score"""
        if not project1.research_domain or not project2.research_domain:
            return 0.0
        
        # Simple keyword-based similarity
        domain1_words = set(project1.research_domain.lower().split())
        domain2_words = set(project2.research_domain.lower().split())
        
        if not domain1_words or not domain2_words:
            return 0.0
        
        intersection = domain1_words.intersection(domain2_words)
        union = domain1_words.union(domain2_words)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _calculate_resource_complementarity(self, project1: ResearchProject, project2: ResearchProject) -> float:
        """Calculate resource complementarity score"""
        if not project1.allocated_resources or not project2.allocated_resources:
            return 0.0
        
        # Analyze resource types and utilization
        p1_resources = {r.resource_type: r.allocated / r.capacity for r in project1.allocated_resources if r.capacity > 0}
        p2_resources = {r.resource_type: r.allocated / r.capacity for r in project2.allocated_resources if r.capacity > 0}
        
        complementarity_score = 0.0
        total_types = set(p1_resources.keys()).union(set(p2_resources.keys()))
        
        for resource_type in total_types:
            p1_util = p1_resources.get(resource_type, 0.0)
            p2_util = p2_resources.get(resource_type, 0.0)
            
            # High complementarity when one project has low utilization and other has high
            if p1_util < 0.5 and p2_util > 0.8:
                complementarity_score += 1.0
            elif p2_util < 0.5 and p1_util > 0.8:
                complementarity_score += 1.0
            elif abs(p1_util - p2_util) > 0.3:
                complementarity_score += 0.5
        
        return min(1.0, complementarity_score / len(total_types)) if total_types else 0.0
    
    def _calculate_methodology_alignment(self, project1: ResearchProject, project2: ResearchProject) -> float:
        """Calculate methodology alignment score"""
        if not project1.methodology or not project2.methodology:
            return 0.0
        
        # Simple text similarity for methodology
        method1_words = set(project1.methodology.lower().split())
        method2_words = set(project2.methodology.lower().split())
        
        if not method1_words or not method2_words:
            return 0.0
        
        intersection = method1_words.intersection(method2_words)
        union = method1_words.union(method2_words)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _calculate_timeline_compatibility(self, project1: ResearchProject, project2: ResearchProject) -> float:
        """Calculate timeline compatibility score"""
        if not all([project1.planned_start, project1.planned_end, project2.planned_start, project2.planned_end]):
            return 0.5  # Neutral score if timeline info missing
        
        # Calculate overlap percentage
        start1, end1 = project1.planned_start, project1.planned_end
        start2, end2 = project2.planned_start, project2.planned_end
        
        overlap_start = max(start1, start2)
        overlap_end = min(end1, end2)
        
        if overlap_start >= overlap_end:
            return 0.0  # No overlap
        
        overlap_duration = (overlap_end - overlap_start).days
        total_duration = max((end1 - start1).days, (end2 - start2).days)
        
        return min(1.0, overlap_duration / total_duration) if total_duration > 0 else 0.0
    
    def _calculate_knowledge_gap_overlap(self, project1: ResearchProject, project2: ResearchProject) -> float:
        """Calculate knowledge gap overlap score"""
        # Analyze objectives and hypotheses for overlap
        p1_objectives = set(obj.lower() for obj in project1.objectives)
        p2_objectives = set(obj.lower() for obj in project2.objectives)
        
        p1_hypotheses = set(hyp.lower() for hyp in project1.hypotheses)
        p2_hypotheses = set(hyp.lower() for hyp in project2.hypotheses)
        
        # Calculate overlap in objectives and hypotheses
        obj_overlap = len(p1_objectives.intersection(p2_objectives)) / len(p1_objectives.union(p2_objectives)) if p1_objectives.union(p2_objectives) else 0.0
        hyp_overlap = len(p1_hypotheses.intersection(p2_hypotheses)) / len(p1_hypotheses.union(p2_hypotheses)) if p1_hypotheses.union(p2_hypotheses) else 0.0
        
        return (obj_overlap + hyp_overlap) / 2
    
    def _identify_complementary_strengths(self, project1: ResearchProject, project2: ResearchProject) -> List[str]:
        """Identify complementary strengths between projects"""
        strengths = []
        
        # Resource complementarity
        p1_resource_types = {r.resource_type for r in project1.allocated_resources}
        p2_resource_types = {r.resource_type for r in project2.allocated_resources}
        
        unique_p1 = p1_resource_types - p2_resource_types
        unique_p2 = p2_resource_types - p1_resource_types
        
        if unique_p1:
            strengths.append(f"Project 1 provides unique {', '.join(r.value for r in unique_p1)} resources")
        if unique_p2:
            strengths.append(f"Project 2 provides unique {', '.join(r.value for r in unique_p2)} resources")
        
        # Methodology complementarity
        if project1.methodology and project2.methodology and project1.methodology != project2.methodology:
            strengths.append("Complementary research methodologies")
        
        # Domain expertise
        if project1.research_domain != project2.research_domain:
            strengths.append("Cross-domain expertise combination")
        
        return strengths
    
    def _identify_shared_challenges(self, project1: ResearchProject, project2: ResearchProject) -> List[str]:
        """Identify shared challenges between projects"""
        challenges = []
        
        # Similar objectives might indicate shared challenges
        p1_obj_words = set()
        for obj in project1.objectives:
            p1_obj_words.update(obj.lower().split())
        
        p2_obj_words = set()
        for obj in project2.objectives:
            p2_obj_words.update(obj.lower().split())
        
        common_words = p1_obj_words.intersection(p2_obj_words)
        if common_words:
            challenges.append(f"Shared research focus areas: {', '.join(list(common_words)[:3])}")
        
        # Resource constraints
        p1_high_util = [r.resource_type.value for r in project1.allocated_resources if r.allocated / r.capacity > 0.8]
        p2_high_util = [r.resource_type.value for r in project2.allocated_resources if r.allocated / r.capacity > 0.8]
        
        common_constraints = set(p1_high_util).intersection(set(p2_high_util))
        if common_constraints:
            challenges.append(f"Shared resource constraints: {', '.join(common_constraints)}")
        
        return challenges
    
    def _identify_collaboration_opportunities_specific(self, project1: ResearchProject, project2: ResearchProject) -> List[str]:
        """Identify specific collaboration opportunities"""
        opportunities = []
        
        # Knowledge sharing opportunities
        if project1.research_domain == project2.research_domain:
            opportunities.append("Share domain-specific knowledge and insights")
        
        # Resource sharing opportunities
        p1_underutil = [r for r in project1.allocated_resources if r.allocated / r.capacity < 0.5]
        p2_overutil = [r for r in project2.allocated_resources if r.allocated / r.capacity > 0.8]
        
        for p1_resource in p1_underutil:
            for p2_resource in p2_overutil:
                if p1_resource.resource_type == p2_resource.resource_type:
                    opportunities.append(f"Share {p1_resource.resource_type.value} resources")
        
        # Joint research opportunities
        if len(set(project1.objectives).intersection(set(project2.objectives))) > 0:
            opportunities.append("Conduct joint research on shared objectives")
        
        # Methodology sharing
        if project1.methodology and project2.methodology:
            opportunities.append("Exchange and validate research methodologies")
        
        return opportunities
    
    def _generate_collaboration_recommendations(
        self,
        project1: ResearchProject,
        project2: ResearchProject,
        synergy: ResearchSynergy
    ) -> List[str]:
        """Generate specific collaboration recommendations"""
        recommendations = []
        
        if synergy.overall_score > 0.8:
            recommendations.append("Establish formal collaboration agreement")
            recommendations.append("Create joint research milestones")
            recommendations.append("Set up regular coordination meetings")
        elif synergy.overall_score > 0.6:
            recommendations.append("Start with informal knowledge sharing")
            recommendations.append("Explore resource sharing opportunities")
            recommendations.append("Consider joint publications")
        else:
            recommendations.append("Monitor for future collaboration opportunities")
            recommendations.append("Share relevant findings periodically")
        
        # Specific recommendations based on synergy type
        if synergy.feasibility_score > 0.7:
            recommendations.append("Implement shared resource allocation system")
        
        if synergy.potential_score > 0.7:
            recommendations.append("Develop joint research proposals")
        
        return recommendations
    
    def _estimate_collaboration_benefits(self, project1: ResearchProject, project2: ResearchProject) -> Dict[str, float]:
        """Estimate quantitative benefits of collaboration"""
        benefits = {}
        
        # Resource efficiency gain
        p1_util = sum(r.allocated / r.capacity for r in project1.allocated_resources if r.capacity > 0)
        p2_util = sum(r.allocated / r.capacity for r in project2.allocated_resources if r.capacity > 0)
        
        if p1_util > 0 and p2_util > 0:
            avg_util = (p1_util + p2_util) / 2
            benefits["resource_efficiency_gain"] = min(0.3, (1.0 - avg_util) * 0.5)
        
        # Timeline acceleration
        if project1.planned_end and project2.planned_end:
            max_duration = max((project1.planned_end - project1.planned_start).days,
                             (project2.planned_end - project2.planned_start).days)
            benefits["timeline_acceleration"] = min(0.25, max_duration * 0.001)
        
        # Knowledge multiplication factor
        shared_objectives = len(set(project1.objectives).intersection(set(project2.objectives)))
        total_objectives = len(set(project1.objectives).union(set(project2.objectives)))
        
        if total_objectives > 0:
            benefits["knowledge_multiplication"] = shared_objectives / total_objectives
        
        # Cost reduction potential
        benefits["cost_reduction"] = benefits.get("resource_efficiency_gain", 0.0) * 0.8
        
        return benefits
    
    def _assess_implementation_complexity(self, project1: ResearchProject, project2: ResearchProject) -> str:
        """Assess implementation complexity of collaboration"""
        complexity_score = 0
        
        # Timeline alignment complexity
        if project1.planned_start and project2.planned_start:
            start_diff = abs((project1.planned_start - project2.planned_start).days)
            if start_diff > 30:
                complexity_score += 1
        
        # Resource coordination complexity
        resource_types_overlap = len(
            {r.resource_type for r in project1.allocated_resources}.intersection(
                {r.resource_type for r in project2.allocated_resources}
            )
        )
        if resource_types_overlap > 3:
            complexity_score += 1
        
        # Domain difference complexity
        if project1.research_domain != project2.research_domain:
            complexity_score += 1
        
        # Priority difference complexity
        if abs(project1.priority - project2.priority) > 3:
            complexity_score += 1
        
        if complexity_score <= 1:
            return "low"
        elif complexity_score <= 2:
            return "medium"
        else:
            return "high"
    
    async def create_collaboration(
        self,
        synergy: ResearchSynergy,
        collaboration_type: CollaborationType = CollaborationType.KNOWLEDGE_SHARING
    ) -> ResearchCollaboration:
        """
        Create a research collaboration based on identified synergy
        
        Args:
            synergy: Identified research synergy
            collaboration_type: Type of collaboration to establish
            
        Returns:
            Created collaboration
        """
        try:
            collaboration = ResearchCollaboration(
                collaboration_type=collaboration_type,
                primary_project_id=synergy.project_ids[0],
                collaborating_project_ids=synergy.project_ids[1:],
                joint_objectives=synergy.collaboration_opportunities,
                synergy_score=synergy.overall_score
            )
            
            # Set collaboration details based on synergy
            collaboration.shared_resources = [
                benefit for benefit in synergy.estimated_benefits.keys()
                if "resource" in benefit
            ]
            
            collaboration.shared_knowledge = synergy.complementary_strengths
            
            # Set coordination frequency based on synergy score
            if synergy.overall_score > 0.8:
                collaboration.coordination_frequency = "daily"
            elif synergy.overall_score > 0.6:
                collaboration.coordination_frequency = "weekly"
            else:
                collaboration.coordination_frequency = "monthly"
            
            # Store collaboration
            self.active_collaborations[collaboration.id] = collaboration
            
            # Mark synergy as exploited
            synergy.is_exploited = True
            synergy.exploitation_results = {
                "collaboration_id": collaboration.id,
                "collaboration_type": collaboration_type.value,
                "created_at": datetime.now().isoformat()
            }
            
            self.logger.info(f"Created collaboration {collaboration.id} between projects {synergy.project_ids}")
            return collaboration
            
        except Exception as e:
            self.logger.error(f"Error creating collaboration: {str(e)}")
            raise
    
    async def share_knowledge_asset(
        self,
        source_project_id: str,
        asset: KnowledgeAsset,
        target_project_ids: List[str]
    ) -> bool:
        """
        Share knowledge asset between projects
        
        Args:
            source_project_id: Source project ID
            asset: Knowledge asset to share
            target_project_ids: Target project IDs
            
        Returns:
            Success status
        """
        try:
            # Store knowledge asset
            asset.source_project_id = source_project_id
            asset.access_count = 0
            self.knowledge_assets[asset.id] = asset
            
            # Update collaboration knowledge sharing metrics
            for collaboration in self.active_collaborations.values():
                if (source_project_id == collaboration.primary_project_id or
                    source_project_id in collaboration.collaborating_project_ids):
                    
                    # Update knowledge transfer rate
                    collaboration.knowledge_transfer_rate += 1.0
                    collaboration.updated_at = datetime.now()
            
            self.logger.info(f"Shared knowledge asset {asset.title} from project {source_project_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error sharing knowledge asset: {str(e)}")
            return False
    
    async def get_collaboration_metrics(self) -> Dict[str, Any]:
        """
        Get collaboration system metrics
        
        Returns:
            Collaboration metrics
        """
        try:
            metrics = {
                "total_collaborations": len(self.active_collaborations),
                "active_collaborations": len([c for c in self.active_collaborations.values() if c.is_active]),
                "knowledge_assets": len(self.knowledge_assets),
                "identified_synergies": len(self.identified_synergies),
                "exploited_synergies": len([s for s in self.identified_synergies.values() if s.is_exploited])
            }
            
            # Calculate average synergy score
            if self.identified_synergies:
                avg_synergy = sum(s.overall_score for s in self.identified_synergies.values()) / len(self.identified_synergies)
                metrics["average_synergy_score"] = avg_synergy
            
            # Calculate knowledge sharing rate
            if self.knowledge_assets:
                total_access = sum(asset.access_count for asset in self.knowledge_assets.values())
                metrics["knowledge_sharing_rate"] = total_access / len(self.knowledge_assets)
            
            # Calculate collaboration effectiveness
            active_collabs = [c for c in self.active_collaborations.values() if c.is_active]
            if active_collabs:
                avg_synergy_score = sum(c.synergy_score for c in active_collabs) / len(active_collabs)
                avg_knowledge_transfer = sum(c.knowledge_transfer_rate for c in active_collabs) / len(active_collabs)
                metrics["collaboration_effectiveness"] = (avg_synergy_score + avg_knowledge_transfer / 10) / 2
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating collaboration metrics: {str(e)}")
            return {}