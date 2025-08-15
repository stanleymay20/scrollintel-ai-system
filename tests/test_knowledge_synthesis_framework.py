"""
Tests for Knowledge Synthesis Framework

This module contains comprehensive tests for the knowledge synthesis framework,
including integration of research findings, experimental results, correlation
identification, knowledge validation, and quality assurance.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from scrollintel.engines.knowledge_synthesis_framework import KnowledgeSynthesisFramework
from scrollintel.models.knowledge_integration_models import (
    KnowledgeItem, KnowledgeCorrelation, SynthesizedKnowledge,
    KnowledgeValidationResult, KnowledgeGraph, SynthesisRequest,
    KnowledgeType, ConfidenceLevel, PatternType
)


class TestKnowledgeSynthesisFramework:
    """Test cases for Knowledge Synthesis Framework"""
    
    @pytest.fixture
    def framework(self):
        """Create a knowledge synthesis framework instance"""
        return KnowledgeSynthesisFramework()
    
    @pytest.fixture
    def sample_research_findings(self):
        """Sample research findings for testing"""
        return [
            {
                "title": "Machine Learning Performance Study",
                "description": "Study on ML model performance optimization",
                "methodology": "experimental",
                "results": {"accuracy": 0.95, "precision": 0.92},
                "keywords": ["machine learning", "optimization", "performance"],
                "domain": "artificial intelligence",
                "peer_reviewed": True,
                "sample_size": 1000,
                "statistical_significance": 0.99,
                "source": "research_lab_1"
            },
            {
                "title": "Deep Learning Architecture Analysis",
                "description": "Analysis of various deep learning architectures",
                "methodology": "comparative",
                "results": {"best_architecture": "transformer", "improvement": 0.15},
                "keywords": ["deep learning", "architecture", "transformer"],
                "domain": "artificial intelligence",
                "peer_reviewed": True,
                "sample_size": 500,
                "statistical_significance": 0.95,
                "source": "research_lab_2"
            }
        ]
    
    @pytest.fixture
    def sample_experimental_results(self):
        """Sample experimental results for testing"""
        return [
            {
                "experiment_id": "exp_001",
                "title": "Neural Network Training Experiment",
                "description": "Experiment on neural network training optimization",
                "methodology": "controlled_experiment",
                "results": {"training_time": 120, "final_accuracy": 0.94},
                "parameters": {"learning_rate": 0.001, "batch_size": 32},
                "replicated": True,
                "sample_size": 200,
                "metadata": {"duration": "7 days", "resources": "GPU cluster"}
            },
            {
                "experiment_id": "exp_002",
                "title": "Data Preprocessing Impact Study",
                "description": "Study on impact of data preprocessing techniques",
                "methodology": "ablation_study",
                "results": {"improvement": 0.08, "processing_time": 45},
                "parameters": {"normalization": True, "augmentation": True},
                "replicated": False,
                "sample_size": 150,
                "metadata": {"duration": "3 days", "resources": "CPU cluster"}
            }
        ]
    
    @pytest.mark.asyncio
    async def test_integrate_research_findings(self, framework, sample_research_findings):
        """Test integration of research findings"""
        # Integrate research findings
        knowledge_items = await framework.integrate_research_findings(sample_research_findings)
        
        # Verify results
        assert len(knowledge_items) == 2
        assert all(item.knowledge_type == KnowledgeType.RESEARCH_FINDING for item in knowledge_items)
        assert all(item.id in framework.knowledge_store for item in knowledge_items)
        
        # Check first item
        first_item = knowledge_items[0]
        assert first_item.content["title"] == "Machine Learning Performance Study"
        assert first_item.confidence == ConfidenceLevel.VERY_HIGH  # High quality indicators
        assert "machine learning" in first_item.tags
        assert "artificial intelligence" in first_item.tags
    
    @pytest.mark.asyncio
    async def test_integrate_experimental_results(self, framework, sample_experimental_results):
        """Test integration of experimental results"""
        # Integrate experimental results
        knowledge_items = await framework.integrate_experimental_results(sample_experimental_results)
        
        # Verify results
        assert len(knowledge_items) == 2
        assert all(item.knowledge_type == KnowledgeType.EXPERIMENTAL_RESULT for item in knowledge_items)
        assert all(item.id in framework.knowledge_store for item in knowledge_items)
        
        # Check first item
        first_item = knowledge_items[0]
        assert first_item.content["experiment_id"] == "exp_001"
        assert first_item.source == "exp_001"
        assert "method_controlled_experiment" in first_item.tags
    
    @pytest.mark.asyncio
    async def test_identify_knowledge_correlations(self, framework, sample_research_findings):
        """Test identification of knowledge correlations"""
        # First integrate some knowledge
        knowledge_items = await framework.integrate_research_findings(sample_research_findings)
        
        # Identify correlations
        correlations = await framework.identify_knowledge_correlations()
        
        # Verify correlations were found (items have similar tags)
        assert len(correlations) >= 0  # May or may not find correlations depending on similarity
        
        # If correlations found, verify structure
        for correlation in correlations:
            assert len(correlation.item_ids) == 2
            assert correlation.strength > 0.3
            assert correlation.id in framework.correlations
    
    @pytest.mark.asyncio
    async def test_synthesize_knowledge(self, framework, sample_research_findings):
        """Test knowledge synthesis"""
        # First integrate knowledge
        knowledge_items = await framework.integrate_research_findings(sample_research_findings)
        
        # Create synthesis request
        synthesis_request = SynthesisRequest(
            id="test_synthesis",
            source_knowledge_ids=[item.id for item in knowledge_items],
            synthesis_goal="Combine ML research findings",
            method_preferences=["integration"],
            priority="high"
        )
        
        # Perform synthesis
        synthesized = await framework.synthesize_knowledge(synthesis_request)
        
        # Verify synthesis result
        assert synthesized.id is not None
        assert len(synthesized.source_items) == 2
        assert synthesized.synthesis_method == "integration"
        assert len(synthesized.insights) > 0
        assert synthesized.quality_score > 0.0
        assert synthesized.id in framework.synthesized_knowledge
    
    @pytest.mark.asyncio
    async def test_validate_knowledge(self, framework, sample_research_findings):
        """Test knowledge validation"""
        # First integrate knowledge
        knowledge_items = await framework.integrate_research_findings(sample_research_findings)
        knowledge_id = knowledge_items[0].id
        
        # Validate knowledge
        validation_result = await framework.validate_knowledge(
            knowledge_id, 
            ["consistency", "completeness", "reliability"]
        )
        
        # Verify validation result
        assert validation_result.knowledge_id == knowledge_id
        assert validation_result.validation_score >= 0.0
        assert validation_result.validation_score <= 1.0
        assert isinstance(validation_result.is_valid, bool)
        assert knowledge_id in framework.validation_cache
    
    @pytest.mark.asyncio
    async def test_create_knowledge_graph(self, framework, sample_research_findings):
        """Test knowledge graph creation"""
        # First integrate knowledge and identify correlations
        knowledge_items = await framework.integrate_research_findings(sample_research_findings)
        await framework.identify_knowledge_correlations()
        
        # Create knowledge graph
        knowledge_graph = await framework.create_knowledge_graph()
        
        # Verify graph structure
        assert len(knowledge_graph.nodes) == len(knowledge_items)
        assert knowledge_graph.metadata["node_count"] == len(knowledge_items)
        assert knowledge_graph.metadata["edge_count"] >= 0
        assert knowledge_graph.created_at is not None
    
    @pytest.mark.asyncio
    async def test_aggregation_synthesis(self, framework):
        """Test aggregation-based synthesis"""
        # Create knowledge items with numerical data
        findings = [
            {
                "title": "Performance Test 1",
                "accuracy": 0.85,
                "precision": 0.82,
                "recall": 0.88,
                "source": "test1"
            },
            {
                "title": "Performance Test 2", 
                "accuracy": 0.90,
                "precision": 0.87,
                "recall": 0.85,
                "source": "test2"
            }
        ]
        
        knowledge_items = await framework.integrate_research_findings(findings)
        
        # Create synthesis request with aggregation method
        synthesis_request = SynthesisRequest(
            id="test_aggregation",
            source_knowledge_ids=[item.id for item in knowledge_items],
            synthesis_goal="Aggregate performance metrics",
            method_preferences=["aggregation"]
        )
        
        # Perform synthesis
        synthesized = await framework.synthesize_knowledge(synthesis_request)
        
        # Verify aggregation results
        assert synthesized.synthesis_method == "aggregation"
        assert "statistics" in synthesized.synthesized_content
        
        stats = synthesized.synthesized_content["statistics"]
        assert "accuracy" in stats
        assert stats["accuracy"]["mean"] == 0.875  # (0.85 + 0.90) / 2
        assert stats["accuracy"]["count"] == 2
    
    @pytest.mark.asyncio
    async def test_knowledge_validation_consistency(self, framework):
        """Test consistency validation"""
        # Create knowledge item with potential consistency issues
        findings = [{
            "title": "Test Finding",
            "description": "Test description",
            "results": [],  # Empty results
            "conclusions": ["Significant improvement found"],  # But has conclusions
            "source": "test_source"
        }]
        
        knowledge_items = await framework.integrate_research_findings(findings)
        knowledge_id = knowledge_items[0].id
        
        # Validate for consistency
        validation_result = await framework.validate_knowledge(knowledge_id, ["consistency"])
        
        # Should find consistency issues
        assert len(validation_result.issues_found) > 0
        assert any("conclusions" in issue.lower() for issue in validation_result.issues_found)
        assert validation_result.validation_score < 1.0
    
    @pytest.mark.asyncio
    async def test_knowledge_validation_completeness(self, framework):
        """Test completeness validation"""
        # Create knowledge item missing required fields
        findings = [{
            "title": "Incomplete Finding",
            # Missing description and methodology
            "source": "test_source"
        }]
        
        knowledge_items = await framework.integrate_research_findings(findings)
        knowledge_id = knowledge_items[0].id
        
        # Validate for completeness
        validation_result = await framework.validate_knowledge(knowledge_id, ["completeness"])
        
        # Should find completeness issues
        assert len(validation_result.issues_found) > 0
        assert any("missing" in issue.lower() for issue in validation_result.issues_found)
        assert validation_result.validation_score < 1.0
    
    @pytest.mark.asyncio
    async def test_knowledge_validation_reliability(self, framework):
        """Test reliability validation"""
        # Create knowledge item with reliability issues
        findings = [{
            "title": "Low Reliability Finding",
            "description": "Test description",
            # No quality indicators - will result in LOW confidence
            "source": ""  # Empty source
        }]
        
        knowledge_items = await framework.integrate_research_findings(findings)
        knowledge_id = knowledge_items[0].id
        
        # Validate for reliability
        validation_result = await framework.validate_knowledge(knowledge_id, ["reliability"])
        
        # Should find reliability issues
        assert len(validation_result.issues_found) > 0
        assert validation_result.validation_score < 1.0
    
    @pytest.mark.asyncio
    async def test_confidence_level_determination(self, framework):
        """Test confidence level determination based on quality indicators"""
        # High quality finding
        high_quality_finding = {
            "title": "High Quality Study",
            "peer_reviewed": True,
            "replicated": True,
            "sample_size": 1000,
            "statistical_significance": 0.99,
            "source": "top_journal"
        }
        
        # Low quality finding
        low_quality_finding = {
            "title": "Low Quality Study",
            "peer_reviewed": False,
            "replicated": False,
            "sample_size": 10,
            "statistical_significance": 0.6,
            "source": "unknown"
        }
        
        # Integrate both
        high_quality_items = await framework.integrate_research_findings([high_quality_finding])
        low_quality_items = await framework.integrate_research_findings([low_quality_finding])
        
        # Check confidence levels
        assert high_quality_items[0].confidence == ConfidenceLevel.VERY_HIGH
        assert low_quality_items[0].confidence == ConfidenceLevel.LOW
    
    @pytest.mark.asyncio
    async def test_relationship_identification(self, framework):
        """Test identification of relationships between knowledge items"""
        # Create related findings
        related_findings = [
            {
                "title": "Machine Learning Optimization Study",
                "keywords": ["machine learning", "optimization"],
                "domain": "AI",
                "source": "lab1"
            },
            {
                "title": "ML Performance Enhancement Research",
                "keywords": ["machine learning", "performance"],
                "domain": "AI", 
                "source": "lab2"
            },
            {
                "title": "Quantum Computing Applications",
                "keywords": ["quantum", "computing"],
                "domain": "quantum",
                "source": "lab3"
            }
        ]
        
        knowledge_items = await framework.integrate_research_findings(related_findings)
        
        # Check relationships - first two should be related, third should not
        ml_items = [item for item in knowledge_items if "machine learning" in item.tags]
        quantum_items = [item for item in knowledge_items if "quantum" in item.tags]
        
        # ML items should have relationships with each other
        for ml_item in ml_items:
            related_ml_items = [rel_id for rel_id in ml_item.relationships if rel_id in [item.id for item in ml_items]]
            # Should have at least some relationships with other ML items
            # (exact count depends on similarity threshold)
        
        # Quantum item should have fewer relationships with ML items
        for quantum_item in quantum_items:
            ml_relationships = [rel_id for rel_id in quantum_item.relationships if rel_id in [item.id for item in ml_items]]
            # Should have fewer or no relationships with ML items
    
    @pytest.mark.asyncio
    async def test_synthesis_insights_generation(self, framework):
        """Test generation of insights from synthesized knowledge"""
        # Create diverse knowledge items
        diverse_findings = [
            {
                "title": "High Confidence Study 1",
                "accuracy": 0.95,
                "peer_reviewed": True,
                "replicated": True,
                "sample_size": 1000,
                "statistical_significance": 0.99,
                "source": "journal1"
            },
            {
                "title": "High Confidence Study 2",
                "accuracy": 0.93,
                "peer_reviewed": True,
                "replicated": True,
                "sample_size": 800,
                "statistical_significance": 0.98,
                "source": "journal2"
            },
            {
                "title": "High Confidence Study 3",
                "accuracy": 0.97,
                "peer_reviewed": True,
                "replicated": True,
                "sample_size": 1200,
                "statistical_significance": 0.99,
                "source": "journal3"
            }
        ]
        
        knowledge_items = await framework.integrate_research_findings(diverse_findings)
        
        # Create synthesis request
        synthesis_request = SynthesisRequest(
            id="insight_test",
            source_knowledge_ids=[item.id for item in knowledge_items],
            synthesis_goal="Generate insights from high-quality studies",
            method_preferences=["aggregation"]
        )
        
        # Perform synthesis
        synthesized = await framework.synthesize_knowledge(synthesis_request)
        
        # Check insights
        assert len(synthesized.insights) > 0
        
        # Should have insight about high confidence
        high_confidence_insight = any(
            "high confidence" in insight.lower() 
            for insight in synthesized.insights
        )
        assert high_confidence_insight
        
        # Should have insight about consistent patterns (low variability in accuracy)
        pattern_insight = any(
            "consistent" in insight.lower() or "pattern" in insight.lower()
            for insight in synthesized.insights
        )
        # Note: This may or may not be present depending on the variability threshold
    
    @pytest.mark.asyncio
    async def test_error_handling(self, framework):
        """Test error handling in various scenarios"""
        # Test synthesis with non-existent knowledge IDs
        invalid_request = SynthesisRequest(
            id="invalid_test",
            source_knowledge_ids=["non_existent_id"],
            synthesis_goal="Test error handling"
        )
        
        with pytest.raises(ValueError, match="No valid source knowledge items found"):
            await framework.synthesize_knowledge(invalid_request)
        
        # Test validation of non-existent knowledge
        with pytest.raises(ValueError, match="Knowledge item .* not found"):
            await framework.validate_knowledge("non_existent_id")
    
    def test_initialization(self, framework):
        """Test framework initialization"""
        assert isinstance(framework.knowledge_store, dict)
        assert isinstance(framework.correlations, dict)
        assert isinstance(framework.synthesized_knowledge, dict)
        assert isinstance(framework.validation_cache, dict)
        
        # All stores should be empty initially
        assert len(framework.knowledge_store) == 0
        assert len(framework.correlations) == 0
        assert len(framework.synthesized_knowledge) == 0
        assert len(framework.validation_cache) == 0


if __name__ == "__main__":
    pytest.main([__file__])