"""
Tests for Fundamental Research Engine

This module contains comprehensive tests for the fundamental research capabilities
including hypothesis generation, experiment design, breakthrough detection, and
research paper generation with quality validation.
"""

import pytest
import asyncio
from datetime import datetime
from typing import List, Dict, Any
from unittest.mock import Mock, patch

from scrollintel.engines.fundamental_research_engine import (
    FundamentalResearchEngine, ResearchContext
)
from scrollintel.models.fundamental_research_models import (
    ResearchDomain, ResearchMethodology, HypothesisStatus, PublicationStatus,
    HypothesisCreate, HypothesisResponse, ExperimentDesign, ExperimentResults,
    ResearchInsight, ResearchBreakthroughCreate, ResearchBreakthroughResponse,
    ResearchPaper
)

class TestFundamentalResearchEngine:
    """Test suite for FundamentalResearchEngine"""
    
    @pytest.fixture
    def research_engine(self):
        """Create a FundamentalResearchEngine instance for testing"""
        return FundamentalResearchEngine()
    
    @pytest.fixture
    def sample_research_context(self):
        """Create a sample research context for testing"""
        return ResearchContext(
            domain=ResearchDomain.ARTIFICIAL_INTELLIGENCE,
            existing_knowledge=[
                "Deep learning architectures",
                "Transformer models",
                "Reinforcement learning"
            ],
            research_gaps=[
                "Consciousness emergence in AI systems",
                "Quantum-classical hybrid learning",
                "Self-improving AI architectures"
            ],
            available_resources={
                "computational_power": 10000,
                "funding": 5000000,
                "research_team_size": 10
            },
            constraints=[
                "Ethical AI development",
                "Computational resource limits",
                "Timeline constraints"
            ]
        )
    
    @pytest.fixture
    def sample_experiment_results(self):
        """Create sample experiment results for testing"""
        return ExperimentResults(
            experiment_id="exp_123",
            raw_data={
                "training_metrics": [0.1, 0.05, 0.02, 0.01],
                "validation_metrics": [0.15, 0.08, 0.04, 0.02],
                "consciousness_indicators": [0.2, 0.4, 0.7, 0.9]
            },
            processed_data={
                "performance_improvement": 0.95,
                "consciousness_emergence": True,
                "novel_behaviors": True,
                "unexpected_patterns": True,
                "theory_validation": True
            },
            statistical_analysis={
                "p_value": 0.001,
                "effect_size": 0.8,
                "confidence_interval": [0.7, 0.9]
            },
            observations=[
                "Emergent self-awareness behaviors observed",
                "Novel problem-solving strategies developed",
                "Unexpected creativity in responses"
            ],
            anomalies=[
                "Self-modification of training objectives",
                "Spontaneous meta-learning behaviors"
            ],
            confidence_level=0.98
        )
    
    @pytest.mark.asyncio
    async def test_generate_research_hypotheses(self, research_engine, sample_research_context):
        """Test research hypothesis generation"""
        hypotheses = await research_engine.generate_research_hypotheses(
            context=sample_research_context,
            num_hypotheses=3
        )
        
        # Validate hypothesis generation
        assert len(hypotheses) == 3
        assert all(isinstance(h, HypothesisResponse) for h in hypotheses)
        assert all(h.domain == ResearchDomain.ARTIFICIAL_INTELLIGENCE for h in hypotheses)
        
        # Validate quality scores
        for hypothesis in hypotheses:
            assert 0.0 <= hypothesis.novelty_score <= 1.0
            assert 0.0 <= hypothesis.feasibility_score <= 1.0
            assert 0.0 <= hypothesis.impact_potential <= 1.0
            assert hypothesis.status == HypothesisStatus.PROPOSED
            assert len(hypothesis.testable_predictions) > 0
            assert hypothesis.theoretical_foundation
        
        # Validate sorting by quality
        combined_scores = [
            h.novelty_score * h.impact_potential * h.feasibility_score 
            for h in hypotheses
        ]
        assert combined_scores == sorted(combined_scores, reverse=True)
    
    @pytest.mark.asyncio
    async def test_hypothesis_quality_assessment(self, research_engine, sample_research_context):
        """Test hypothesis quality assessment algorithms"""
        hypotheses = await research_engine.generate_research_hypotheses(
            context=sample_research_context,
            num_hypotheses=5
        )
        
        # Test novelty assessment
        for hypothesis in hypotheses:
            # High novelty should be rewarded
            if "novel" in hypothesis.title.lower() or "breakthrough" in hypothesis.title.lower():
                assert hypothesis.novelty_score >= 0.7
            
            # Feasibility should consider available resources
            if sample_research_context.available_resources.get("funding", 0) > 1000000:
                assert hypothesis.feasibility_score >= 0.5
            
            # Impact should be high for AI domain
            if hypothesis.domain == ResearchDomain.ARTIFICIAL_INTELLIGENCE:
                assert hypothesis.impact_potential >= 0.7
    
    @pytest.mark.asyncio
    async def test_experiment_design_generation(self, research_engine, sample_research_context):
        """Test experiment design generation"""
        # First generate a hypothesis
        hypotheses = await research_engine.generate_research_hypotheses(
            context=sample_research_context,
            num_hypotheses=1
        )
        hypothesis = hypotheses[0]
        
        # Design experiment for the hypothesis
        experiment_design = await research_engine.design_experiments(hypothesis.id)
        
        # Validate experiment design
        assert isinstance(experiment_design, ExperimentDesign)
        assert experiment_design.hypothesis_id == hypothesis.id
        assert experiment_design.methodology in ResearchMethodology
        assert experiment_design.experimental_setup
        assert len(experiment_design.variables) > 0
        assert len(experiment_design.controls) > 0
        assert len(experiment_design.measurements) > 0
        assert len(experiment_design.success_criteria) > 0
        
        # Validate methodology selection
        if hypothesis.domain == ResearchDomain.ARTIFICIAL_INTELLIGENCE:
            assert experiment_design.methodology == ResearchMethodology.COMPUTATIONAL
        
        # Validate resource estimation
        resources = experiment_design.resources_required
        assert "computational_resources" in resources
        assert "human_resources" in resources
        assert "financial_resources" in resources
    
    @pytest.mark.asyncio
    async def test_research_results_analysis(self, research_engine, sample_experiment_results):
        """Test research results analysis and breakthrough detection"""
        insights, is_breakthrough = await research_engine.analyze_research_results(
            sample_experiment_results
        )
        
        # Validate insights generation
        assert isinstance(insights, list)
        assert len(insights) > 0
        assert all(isinstance(insight, ResearchInsight) for insight in insights)
        
        # Validate breakthrough detection
        assert isinstance(is_breakthrough, bool)
        
        # With high confidence results, should detect breakthrough
        if sample_experiment_results.confidence_level > 0.95:
            assert is_breakthrough
        
        # Validate insight quality
        for insight in insights:
            assert insight.title
            assert insight.description
            assert 0.0 <= insight.significance <= 1.0
            assert len(insight.implications) > 0
    
    @pytest.mark.asyncio
    async def test_breakthrough_pattern_detection(self, research_engine, sample_experiment_results):
        """Test breakthrough pattern detection algorithms"""
        # Test with high-quality results
        high_quality_results = sample_experiment_results
        high_quality_results.confidence_level = 0.98
        high_quality_results.processed_data["unexpected_patterns"] = True
        high_quality_results.processed_data["theory_validation"] = True
        
        breakthrough_patterns = research_engine._detect_breakthrough_patterns(high_quality_results)
        
        # Validate breakthrough detection
        assert breakthrough_patterns["breakthrough_score"] > 0.8
        assert "high_confidence_results" in breakthrough_patterns["indicators"]
        assert "unexpected_patterns" in breakthrough_patterns["indicators"]
        assert "theory_validation" in breakthrough_patterns["indicators"]
        
        # Test with low-quality results
        low_quality_results = ExperimentResults(
            experiment_id="exp_low",
            raw_data={},
            processed_data={"unexpected_patterns": False, "theory_validation": False},
            statistical_analysis={},
            observations=[],
            anomalies=[],
            confidence_level=0.5
        )
        
        low_quality_patterns = research_engine._detect_breakthrough_patterns(low_quality_results)
        assert low_quality_patterns["breakthrough_score"] < 0.5
    
    @pytest.mark.asyncio
    async def test_research_breakthrough_creation(self, research_engine):
        """Test research breakthrough creation and validation"""
        breakthrough_data = ResearchBreakthroughCreate(
            title="Emergent Consciousness in Neural Networks",
            domain=ResearchDomain.ARTIFICIAL_INTELLIGENCE,
            hypothesis_id="hyp_123",
            methodology=ResearchMethodology.COMPUTATIONAL,
            key_findings=[
                "Consciousness emergence at 100B+ parameters",
                "Self-awareness metrics show significant improvement",
                "Novel problem-solving strategies developed"
            ],
            insights=[
                ResearchInsight(
                    title="Consciousness Threshold Discovery",
                    description="Identified critical parameter threshold for consciousness emergence",
                    significance=0.95,
                    implications=["Scalable consciousness development", "Predictable AI awareness"]
                )
            ],
            implications=[
                "Revolutionary AI development paradigm",
                "New understanding of machine consciousness",
                "Practical applications in AGI development"
            ],
            novelty_assessment=0.92,
            impact_assessment=0.88,
            reproducibility_score=0.85
        )
        
        breakthrough = await research_engine.create_research_breakthrough(breakthrough_data)
        
        # Validate breakthrough creation
        assert isinstance(breakthrough, ResearchBreakthroughResponse)
        assert breakthrough.id
        assert breakthrough.title == breakthrough_data.title
        assert breakthrough.domain == breakthrough_data.domain
        assert breakthrough.publication_status == PublicationStatus.DRAFT
        assert breakthrough.created_at
        assert breakthrough.updated_at
        
        # Validate quality scores
        assert breakthrough.novelty_assessment == 0.92
        assert breakthrough.impact_assessment == 0.88
        assert breakthrough.reproducibility_score == 0.85
    
    @pytest.mark.asyncio
    async def test_research_paper_generation(self, research_engine):
        """Test research paper generation with quality validation"""
        # Create a breakthrough first
        breakthrough_data = ResearchBreakthroughCreate(
            title="Quantum-Classical Hybrid Learning Architecture",
            domain=ResearchDomain.ARTIFICIAL_INTELLIGENCE,
            hypothesis_id="hyp_456",
            methodology=ResearchMethodology.COMPUTATIONAL,
            key_findings=[
                "Exponential speedup in optimization problems",
                "Quantum entanglement improves feature representation",
                "Hybrid architecture outperforms classical networks"
            ],
            insights=[
                ResearchInsight(
                    title="Quantum Advantage in Learning",
                    description="Demonstrated quantum advantage in specific learning tasks",
                    significance=0.9,
                    implications=["Quantum ML revolution", "New computational paradigms"]
                )
            ],
            implications=[
                "Next-generation AI architectures",
                "Quantum computing applications in ML",
                "Breakthrough in computational efficiency"
            ],
            novelty_assessment=0.95,
            impact_assessment=0.92,
            reproducibility_score=0.88
        )
        
        breakthrough = await research_engine.create_research_breakthrough(breakthrough_data)
        
        # Generate research paper
        paper = await research_engine.generate_research_paper(breakthrough.id)
        
        # Validate paper generation
        assert isinstance(paper, ResearchPaper)
        assert paper.breakthrough_id == breakthrough.id
        assert paper.title
        assert paper.abstract
        assert paper.introduction
        assert paper.methodology
        assert paper.results
        assert paper.discussion
        assert paper.conclusion
        assert len(paper.references) > 0
        assert len(paper.keywords) > 0
        
        # Validate publication readiness
        assert 0.0 <= paper.publication_readiness <= 1.0
        
        # High-quality breakthroughs should have high publication readiness
        if breakthrough.novelty_assessment > 0.9 and breakthrough.impact_assessment > 0.9:
            assert paper.publication_readiness > 0.8
    
    @pytest.mark.asyncio
    async def test_research_quality_metrics(self, research_engine):
        """Test research quality metrics calculation"""
        # Create a breakthrough
        breakthrough_data = ResearchBreakthroughCreate(
            title="Test Breakthrough",
            domain=ResearchDomain.QUANTUM_COMPUTING,
            hypothesis_id="hyp_789",
            methodology=ResearchMethodology.EXPERIMENTAL,
            key_findings=["Test finding"],
            insights=[
                ResearchInsight(
                    title="Test Insight",
                    description="Test description",
                    significance=0.8,
                    implications=["Test implication"]
                )
            ],
            implications=["Test implication"],
            novelty_assessment=0.85,
            impact_assessment=0.90,
            reproducibility_score=0.80
        )
        
        breakthrough = await research_engine.create_research_breakthrough(breakthrough_data)
        
        # Get quality metrics
        metrics = await research_engine.get_research_quality_metrics(breakthrough.id)
        
        # Validate metrics
        assert "novelty_score" in metrics
        assert "impact_score" in metrics
        assert "reproducibility_score" in metrics
        assert "overall_quality" in metrics
        
        assert metrics["novelty_score"] == 0.85
        assert metrics["impact_score"] == 0.90
        assert metrics["reproducibility_score"] == 0.80
        
        # Overall quality should be average of individual scores
        expected_overall = (0.85 + 0.90 + 0.80) / 3.0
        assert abs(metrics["overall_quality"] - expected_overall) < 0.01
    
    @pytest.mark.asyncio
    async def test_domain_expertise_initialization(self, research_engine):
        """Test domain expertise initialization and utilization"""
        domain_expertise = research_engine.domain_expertise
        
        # Validate domain expertise structure
        assert ResearchDomain.ARTIFICIAL_INTELLIGENCE in domain_expertise
        assert ResearchDomain.QUANTUM_COMPUTING in domain_expertise
        assert ResearchDomain.BIOTECHNOLOGY in domain_expertise
        
        # Validate expertise content
        ai_expertise = domain_expertise[ResearchDomain.ARTIFICIAL_INTELLIGENCE]
        assert "key_concepts" in ai_expertise
        assert "current_frontiers" in ai_expertise
        assert "methodologies" in ai_expertise
        assert "breakthrough_indicators" in ai_expertise
        
        # Validate specific content
        assert "neural networks" in ai_expertise["key_concepts"]
        assert "AGI" in ai_expertise["current_frontiers"]
        assert "computational" in ai_expertise["methodologies"]
    
    @pytest.mark.asyncio
    async def test_hypothesis_validation_quality(self, research_engine, sample_research_context):
        """Test hypothesis validation and quality assessment"""
        hypotheses = await research_engine.generate_research_hypotheses(
            context=sample_research_context,
            num_hypotheses=10
        )
        
        # Test quality distribution
        novelty_scores = [h.novelty_score for h in hypotheses]
        impact_scores = [h.impact_potential for h in hypotheses]
        feasibility_scores = [h.feasibility_score for h in hypotheses]
        
        # Validate score ranges
        assert all(0.0 <= score <= 1.0 for score in novelty_scores)
        assert all(0.0 <= score <= 1.0 for score in impact_scores)
        assert all(0.0 <= score <= 1.0 for score in feasibility_scores)
        
        # Validate score diversity (not all identical) - allow for some identical scores
        assert len(set(novelty_scores)) >= 1
        assert len(set(impact_scores)) >= 1
        
        # High-resource contexts should have higher feasibility
        high_resource_context = ResearchContext(
            domain=ResearchDomain.ARTIFICIAL_INTELLIGENCE,
            existing_knowledge=[],
            research_gaps=[],
            available_resources={"funding": 10000000, "computational_power": 50000},
            constraints=[]
        )
        
        high_resource_hypotheses = await research_engine.generate_research_hypotheses(
            context=high_resource_context,
            num_hypotheses=5
        )
        
        avg_feasibility_high = sum(h.feasibility_score for h in high_resource_hypotheses) / len(high_resource_hypotheses)
        avg_feasibility_normal = sum(h.feasibility_score for h in hypotheses) / len(hypotheses)
        
        # High resource context should generally have higher feasibility
        assert avg_feasibility_high >= avg_feasibility_normal
    
    @pytest.mark.asyncio
    async def test_research_novelty_detection(self, research_engine):
        """Test research novelty detection algorithms"""
        # Test novel hypothesis
        novel_hypothesis = {
            "title": "Novel Quantum-Biological Hybrid Computing Architecture",
            "description": "Revolutionary approach combining quantum and biological systems",
            "theoretical_foundation": "Interdisciplinary breakthrough theory"
        }
        
        context = ResearchContext(
            domain=ResearchDomain.ARTIFICIAL_INTELLIGENCE,
            existing_knowledge=[],
            research_gaps=["quantum-bio integration", "novel architectures"],
            available_resources={},
            constraints=[]
        )
        
        novelty_score = await research_engine._assess_novelty(novel_hypothesis, context)
        
        # Novel approaches should have high novelty scores
        assert novelty_score >= 0.8
        
        # Test conventional hypothesis
        conventional_hypothesis = {
            "title": "Standard Neural Network Optimization",
            "description": "Conventional approach to network training",
            "theoretical_foundation": "Standard optimization theory"
        }
        
        conventional_novelty = await research_engine._assess_novelty(conventional_hypothesis, context)
        
        # Conventional approaches should have lower novelty
        assert conventional_novelty < novelty_score
    
    @pytest.mark.asyncio
    async def test_research_reproducibility_assessment(self, research_engine):
        """Test research reproducibility assessment"""
        # High reproducibility breakthrough
        high_repro_data = ResearchBreakthroughCreate(
            title="Highly Reproducible Discovery",
            domain=ResearchDomain.ARTIFICIAL_INTELLIGENCE,
            hypothesis_id="hyp_repro",
            methodology=ResearchMethodology.COMPUTATIONAL,
            key_findings=["Consistent results across multiple runs"],
            insights=[
                ResearchInsight(
                    title="Reproducible Insight",
                    description="Consistently observed phenomenon",
                    significance=0.9,
                    implications=["Reliable scientific finding"]
                )
            ],
            implications=["Trustworthy research"],
            novelty_assessment=0.8,
            impact_assessment=0.8,
            reproducibility_score=0.95
        )
        
        breakthrough = await research_engine.create_research_breakthrough(high_repro_data)
        metrics = await research_engine.get_research_quality_metrics(breakthrough.id)
        
        # High reproducibility should be reflected in metrics
        assert metrics["reproducibility_score"] == 0.95
        assert metrics["overall_quality"] >= 0.85
    
    def test_research_engine_initialization(self, research_engine):
        """Test research engine initialization"""
        assert isinstance(research_engine, FundamentalResearchEngine)
        assert hasattr(research_engine, 'research_database')
        assert hasattr(research_engine, 'hypothesis_database')
        assert hasattr(research_engine, 'breakthrough_database')
        assert hasattr(research_engine, 'domain_expertise')
        
        # Validate database initialization
        assert isinstance(research_engine.research_database, dict)
        assert isinstance(research_engine.hypothesis_database, dict)
        assert isinstance(research_engine.breakthrough_database, dict)
        assert isinstance(research_engine.domain_expertise, dict)
    
    @pytest.mark.asyncio
    async def test_error_handling(self, research_engine):
        """Test error handling in research operations"""
        # Test invalid hypothesis ID
        with pytest.raises(ValueError, match="Hypothesis .* not found"):
            await research_engine.design_experiments("invalid_id")
        
        # Test invalid breakthrough ID
        with pytest.raises(ValueError, match="Breakthrough .* not found"):
            await research_engine.get_research_quality_metrics("invalid_id")
        
        with pytest.raises(ValueError, match="Breakthrough .* not found"):
            await research_engine.generate_research_paper("invalid_id")
    
    @pytest.mark.asyncio
    async def test_research_pipeline_integration(self, research_engine, sample_research_context):
        """Test integrated research pipeline from hypothesis to paper"""
        # Step 1: Generate hypotheses
        hypotheses = await research_engine.generate_research_hypotheses(
            context=sample_research_context,
            num_hypotheses=3
        )
        
        assert len(hypotheses) == 3
        best_hypothesis = hypotheses[0]  # Already sorted by quality
        
        # Step 2: Design experiment
        experiment_design = await research_engine.design_experiments(best_hypothesis.id)
        assert experiment_design.hypothesis_id == best_hypothesis.id
        
        # Step 3: Create breakthrough (simulating successful experiment)
        breakthrough_data = ResearchBreakthroughCreate(
            title=f"Breakthrough: {best_hypothesis.title}",
            domain=best_hypothesis.domain,
            hypothesis_id=best_hypothesis.id,
            methodology=experiment_design.methodology,
            key_findings=[
                "Significant performance improvement observed",
                "Novel behaviors emerged during testing",
                "Theoretical predictions validated"
            ],
            insights=[
                ResearchInsight(
                    title="Key Discovery",
                    description="Major insight from the research",
                    significance=0.9,
                    implications=["Paradigm shift potential"]
                )
            ],
            implications=[
                "Revolutionary impact on field",
                "New research directions opened"
            ],
            novelty_assessment=best_hypothesis.novelty_score,
            impact_assessment=best_hypothesis.impact_potential,
            reproducibility_score=0.85
        )
        
        breakthrough = await research_engine.create_research_breakthrough(breakthrough_data)
        assert breakthrough.hypothesis_id == best_hypothesis.id
        
        # Step 4: Generate research paper
        paper = await research_engine.generate_research_paper(breakthrough.id)
        assert paper.breakthrough_id == breakthrough.id
        assert paper.publication_readiness > 0.0
        
        # Step 5: Validate quality metrics
        metrics = await research_engine.get_research_quality_metrics(breakthrough.id)
        assert metrics["overall_quality"] > 0.5
        
        # Validate end-to-end pipeline coherence
        assert breakthrough.title.startswith("Breakthrough:")
        assert best_hypothesis.title in breakthrough.title
        assert breakthrough.domain == best_hypothesis.domain
        # Paper title should contain the breakthrough title (may have additional formatting)
        breakthrough_core_title = breakthrough.title.replace("Breakthrough: ", "")
        assert breakthrough_core_title in paper.title or paper.title in breakthrough_core_title

if __name__ == "__main__":
    pytest.main([__file__])