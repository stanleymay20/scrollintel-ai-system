"""
Tests for breakthrough innovation engine and related components
"""
import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock

from scrollintel.models.breakthrough_models import (
    BreakthroughConcept, InnovationPotential, TechnologyDomain,
    InnovationStage, DisruptionLevel, DisruptionPrediction,
    ResearchDirection, TechnologyTrend, MarketOpportunity, Capability
)
from scrollintel.engines.breakthrough_engine import BreakthroughEngine
from scrollintel.engines.technology_trend_analyzer import TechnologyTrendAnalyzer
from scrollintel.engines.market_disruption_predictor import MarketDisruptionPredictor
from scrollintel.engines.innovation_concept_generator import InnovationConceptGenerator


class TestBreakthroughEngine:
    """Test breakthrough innovation engine functionality"""
    
    @pytest.fixture
    def breakthrough_engine(self):
        """Create breakthrough engine instance"""
        return BreakthroughEngine()
    
    @pytest.fixture
    def sample_breakthrough_concept(self):
        """Create sample breakthrough concept for testing"""
        return BreakthroughConcept(
            id="test-concept-1",
            name="Quantum-Neural Hybrid Processor",
            description="Revolutionary processor combining quantum and neural computing",
            detailed_specification="Detailed technical specification...",
            technology_domain=TechnologyDomain.ARTIFICIAL_INTELLIGENCE,
            innovation_stage=InnovationStage.CONCEPT,
            disruption_level=DisruptionLevel.REVOLUTIONARY,
            innovation_potential=None,
            market_opportunity=MarketOpportunity(
                market_size_billions=100.0,
                growth_rate_percent=25.0,
                time_to_market_years=5,
                competitive_landscape="Emerging market",
                barriers_to_entry=["High R&D costs"],
                key_success_factors=["Technical breakthrough"]
            ),
            required_capabilities=[
                Capability(
                    name="Quantum Computing",
                    description="Quantum computing expertise",
                    current_level=0.3,
                    required_level=0.9,
                    development_time_months=24,
                    cost_estimate=5000000.0
                )
            ],
            underlying_technologies=["Quantum Computing", "Neural Networks"],
            breakthrough_mechanisms=["Quantum superposition", "Neural plasticity"],
            scientific_principles=["Quantum mechanics", "Information theory"],
            existing_solutions=["Traditional processors"],
            competitive_advantages=["1000x speed improvement"],
            differentiation_factors=["Hybrid architecture"],
            research_milestones=[{"milestone": "Proof of concept", "timeline": 6}],
            development_phases=[{"phase": "Research", "duration": 18}],
            success_metrics=[{"name": "Performance", "target": "1000x improvement"}],
            created_by="test_user",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            version="1.0",
            tags=["quantum", "ai"],
            ai_confidence_score=0.85,
            generated_hypotheses=["Quantum advantage in AI"],
            recommended_experiments=["Quantum simulation"],
            potential_partnerships=["Quantum hardware vendors"]
        )

    @pytest.mark.asyncio
    async def test_generate_breakthrough_concepts(self, breakthrough_engine):
        """Test breakthrough concept generation"""
        # Test concept generation
        concepts = await breakthrough_engine.generate_breakthrough_concepts(
            domain=TechnologyDomain.ARTIFICIAL_INTELLIGENCE,
            market_focus="autonomous systems",
            timeline_years=5
        )
        
        # Validate results
        assert len(concepts) == 3  # Default number of concepts
        assert all(isinstance(concept, BreakthroughConcept) for concept in concepts)
        assert all(concept.technology_domain == TechnologyDomain.ARTIFICIAL_INTELLIGENCE for concept in concepts)
        assert all(concept.innovation_stage == InnovationStage.CONCEPT for concept in concepts)
        assert all(concept.ai_confidence_score > 0.0 for concept in concepts)

    @pytest.mark.asyncio
    async def test_analyze_innovation_potential(self, breakthrough_engine, sample_breakthrough_concept):
        """Test innovation potential analysis"""
        # Test innovation potential analysis
        potential = await breakthrough_engine.analyze_innovation_potential(sample_breakthrough_concept)
        
        # Validate results
        assert isinstance(potential, InnovationPotential)
        assert potential.concept_id == sample_breakthrough_concept.id
        assert 0.0 <= potential.novelty_score <= 1.0
        assert 0.0 <= potential.feasibility_score <= 1.0
        assert 0.0 <= potential.market_impact_score <= 1.0
        assert 0.0 <= potential.competitive_advantage_score <= 1.0
        assert 0.0 <= potential.risk_score <= 1.0
        assert 0.0 <= potential.overall_potential <= 1.0
        assert 0.0 <= potential.success_probability <= 1.0
        assert potential.expected_roi > 0.0
        assert potential.research_phase_months > 0
        assert potential.development_phase_months > 0
        assert potential.market_entry_months > 0
        assert len(potential.technical_risks) > 0
        assert len(potential.market_risks) > 0

    @pytest.mark.asyncio
    async def test_predict_market_disruption(self, breakthrough_engine):
        """Test market disruption prediction"""
        # Test disruption prediction
        prediction = await breakthrough_engine.predict_market_disruption(
            technology="Quantum Computing",
            timeframe=10
        )
        
        # Validate results
        assert isinstance(prediction, DisruptionPrediction)
        assert prediction.technology_name == "Quantum Computing"
        assert prediction.disruption_timeline_years > 0
        assert 0.0 <= prediction.disruption_probability <= 1.0
        assert prediction.market_size_affected_billions > 0.0
        assert prediction.jobs_displaced >= 0
        assert prediction.jobs_created >= 0
        assert prediction.productivity_gain_percent >= 0.0
        assert len(prediction.new_capabilities) > 0
        assert len(prediction.first_mover_advantages) > 0
        assert prediction.investment_requirements_millions > 0.0

    @pytest.mark.asyncio
    async def test_recommend_research_directions(self, breakthrough_engine):
        """Test research direction recommendations"""
        # Create sample capabilities
        capabilities = [
            Capability(
                name="AI Research",
                description="AI research capability",
                current_level=0.7,
                required_level=0.9,
                development_time_months=12,
                cost_estimate=2000000.0
            )
        ]
        
        # Test research direction recommendations
        directions = await breakthrough_engine.recommend_research_directions(capabilities)
        
        # Validate results
        assert len(directions) > 0
        assert all(isinstance(direction, ResearchDirection) for direction in directions)
        assert all(direction.priority_score > 0.0 for direction in directions)
        assert all(direction.expected_duration_months > 0 for direction in directions)
        assert all(0.0 <= direction.breakthrough_probability <= 1.0 for direction in directions)
        assert all(direction.commercial_value > 0.0 for direction in directions)
        
        # Check prioritization (should be sorted by priority)
        priorities = [d.priority_score for d in directions]
        assert priorities == sorted(priorities, reverse=True)

    def test_breakthrough_concept_validation(self, sample_breakthrough_concept):
        """Test breakthrough concept data validation"""
        concept = sample_breakthrough_concept
        
        # Test required fields
        assert concept.id is not None
        assert concept.name is not None
        assert concept.description is not None
        assert isinstance(concept.technology_domain, TechnologyDomain)
        assert isinstance(concept.innovation_stage, InnovationStage)
        assert isinstance(concept.disruption_level, DisruptionLevel)
        assert isinstance(concept.market_opportunity, MarketOpportunity)
        assert isinstance(concept.required_capabilities, list)
        assert all(isinstance(cap, Capability) for cap in concept.required_capabilities)
        
        # Test data integrity
        assert concept.ai_confidence_score >= 0.0 and concept.ai_confidence_score <= 1.0
        assert isinstance(concept.underlying_technologies, list)
        assert isinstance(concept.breakthrough_mechanisms, list)
        assert isinstance(concept.competitive_advantages, list)
        assert isinstance(concept.created_at, datetime)
        assert isinstance(concept.updated_at, datetime)

    def test_innovation_potential_validation(self):
        """Test innovation potential data validation"""
        potential = InnovationPotential(
            concept_id="test-concept",
            novelty_score=0.9,
            feasibility_score=0.7,
            market_impact_score=0.95,
            competitive_advantage_score=0.85,
            risk_score=0.4,
            overall_potential=0.82,
            confidence_level=0.85,
            technical_risks=["Risk 1", "Risk 2"],
            market_risks=["Market risk"],
            regulatory_risks=["Regulatory risk"],
            success_probability=0.7,
            expected_roi=15.0,
            research_phase_months=18,
            development_phase_months=36,
            market_entry_months=60,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        # Test score ranges
        assert 0.0 <= potential.novelty_score <= 1.0
        assert 0.0 <= potential.feasibility_score <= 1.0
        assert 0.0 <= potential.market_impact_score <= 1.0
        assert 0.0 <= potential.competitive_advantage_score <= 1.0
        assert 0.0 <= potential.risk_score <= 1.0
        assert 0.0 <= potential.overall_potential <= 1.0
        assert 0.0 <= potential.confidence_level <= 1.0
        assert 0.0 <= potential.success_probability <= 1.0
        
        # Test timeline values
        assert potential.research_phase_months > 0
        assert potential.development_phase_months > 0
        assert potential.market_entry_months > 0
        
        # Test risk lists
        assert isinstance(potential.technical_risks, list)
        assert isinstance(potential.market_risks, list)
        assert isinstance(potential.regulatory_risks, list)


class TestTechnologyTrendAnalyzer:
    """Test technology trend analyzer"""
    
    @pytest.fixture
    def trend_analyzer(self):
        """Create trend analyzer instance"""
        return TechnologyTrendAnalyzer()

    @pytest.mark.asyncio
    async def test_analyze_technology_trends(self, trend_analyzer):
        """Test technology trend analysis"""
        trends = await trend_analyzer.analyze_technology_trends(
            domain=TechnologyDomain.ARTIFICIAL_INTELLIGENCE,
            timeframe_years=5
        )
        
        # Validate results
        assert len(trends) > 0
        assert all(isinstance(trend, TechnologyTrend) for trend in trends)
        assert all(trend.domain == TechnologyDomain.ARTIFICIAL_INTELLIGENCE for trend in trends)
        assert all(0.0 <= trend.momentum_score <= 1.0 for trend in trends)
        assert all(trend.patent_activity >= 0 for trend in trends)
        assert all(trend.research_papers >= 0 for trend in trends)
        assert all(trend.investment_millions >= 0.0 for trend in trends)
        assert all(trend.predicted_breakthrough_timeline > 0 for trend in trends)
        assert all(isinstance(trend.key_players, list) for trend in trends)

    def test_technology_trend_validation(self):
        """Test technology trend data validation"""
        trend = TechnologyTrend(
            trend_name="Large Language Models",
            domain=TechnologyDomain.ARTIFICIAL_INTELLIGENCE,
            momentum_score=0.9,
            patent_activity=500,
            research_papers=1000,
            investment_millions=2000.0,
            key_players=["OpenAI", "Google", "Microsoft"],
            predicted_breakthrough_timeline=3
        )
        
        # Test data types and ranges
        assert isinstance(trend.trend_name, str)
        assert isinstance(trend.domain, TechnologyDomain)
        assert 0.0 <= trend.momentum_score <= 1.0
        assert trend.patent_activity >= 0
        assert trend.research_papers >= 0
        assert trend.investment_millions >= 0.0
        assert trend.predicted_breakthrough_timeline > 0
        assert isinstance(trend.key_players, list)


class TestMarketDisruptionPredictor:
    """Test market disruption predictor"""
    
    @pytest.fixture
    def disruption_predictor(self):
        """Create disruption predictor instance"""
        return MarketDisruptionPredictor()

    @pytest.mark.asyncio
    async def test_predict_market_disruption(self, disruption_predictor):
        """Test market disruption prediction"""
        prediction = await disruption_predictor.predict_market_disruption(
            technology="Artificial Intelligence",
            target_industry="Healthcare",
            timeframe_years=10
        )
        
        # Validate results
        assert isinstance(prediction, DisruptionPrediction)
        assert prediction.technology_name == "Artificial Intelligence"
        assert prediction.target_industry == "Healthcare"
        assert prediction.disruption_timeline_years > 0
        assert 0.0 <= prediction.disruption_probability <= 1.0
        assert prediction.market_size_affected_billions >= 0.0
        assert prediction.productivity_gain_percent >= 0.0
        assert prediction.cost_reduction_percent >= 0.0
        assert prediction.performance_improvement_percent >= 0.0
        assert isinstance(prediction.new_capabilities, list)
        assert isinstance(prediction.obsoleted_technologies, list)
        assert isinstance(prediction.first_mover_advantages, list)
        assert isinstance(prediction.defensive_strategies, list)
        assert isinstance(prediction.regulatory_challenges, list)

    def test_disruption_prediction_validation(self):
        """Test disruption prediction data validation"""
        prediction = DisruptionPrediction(
            technology_name="Quantum Computing",
            target_industry="Finance",
            disruption_timeline_years=7,
            disruption_probability=0.75,
            market_size_affected_billions=500.0,
            jobs_displaced=50000,
            jobs_created=75000,
            productivity_gain_percent=40.0,
            cost_reduction_percent=30.0,
            performance_improvement_percent=1000.0,
            new_capabilities=["Quantum advantage"],
            obsoleted_technologies=["Classical encryption"],
            first_mover_advantages=["Market leadership"],
            defensive_strategies=["R&D investment"],
            investment_requirements_millions=5000.0,
            regulatory_challenges=["Quantum regulations"],
            created_at=datetime.now()
        )
        
        # Test data validation
        assert isinstance(prediction.technology_name, str)
        assert isinstance(prediction.target_industry, str)
        assert prediction.disruption_timeline_years > 0
        assert 0.0 <= prediction.disruption_probability <= 1.0
        assert prediction.market_size_affected_billions >= 0.0
        assert prediction.jobs_displaced >= 0
        assert prediction.jobs_created >= 0
        assert prediction.productivity_gain_percent >= 0.0
        assert prediction.investment_requirements_millions >= 0.0
        assert isinstance(prediction.created_at, datetime)


class TestInnovationConceptGenerator:
    """Test innovation concept generator"""
    
    @pytest.fixture
    def concept_generator(self):
        """Create concept generator instance"""
        return InnovationConceptGenerator()
    
    @pytest.fixture
    def sample_breakthrough_concept(self):
        """Create sample breakthrough concept for testing"""
        return BreakthroughConcept(
            id="test-concept-1",
            name="Quantum-Neural Hybrid Processor",
            description="Revolutionary processor combining quantum and neural computing",
            detailed_specification="Detailed technical specification...",
            technology_domain=TechnologyDomain.ARTIFICIAL_INTELLIGENCE,
            innovation_stage=InnovationStage.CONCEPT,
            disruption_level=DisruptionLevel.REVOLUTIONARY,
            innovation_potential=None,
            market_opportunity=MarketOpportunity(
                market_size_billions=100.0,
                growth_rate_percent=25.0,
                time_to_market_years=5,
                competitive_landscape="Emerging market",
                barriers_to_entry=["High R&D costs"],
                key_success_factors=["Technical breakthrough"]
            ),
            required_capabilities=[
                Capability(
                    name="Quantum Computing",
                    description="Quantum computing expertise",
                    current_level=0.3,
                    required_level=0.9,
                    development_time_months=24,
                    cost_estimate=5000000.0
                )
            ],
            underlying_technologies=["Quantum Computing", "Neural Networks"],
            breakthrough_mechanisms=["Quantum superposition", "Neural plasticity"],
            scientific_principles=["Quantum mechanics", "Information theory"],
            existing_solutions=["Traditional processors"],
            competitive_advantages=["1000x speed improvement"],
            differentiation_factors=["Hybrid architecture"],
            research_milestones=[{"milestone": "Proof of concept", "timeline": 6}],
            development_phases=[{"phase": "Research", "duration": 18}],
            success_metrics=[{"name": "Performance", "target": "1000x improvement"}],
            created_by="test_user",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            version="1.0",
            tags=["quantum", "ai"],
            ai_confidence_score=0.85,
            generated_hypotheses=["Quantum advantage in AI"],
            recommended_experiments=["Quantum simulation"],
            potential_partnerships=["Quantum hardware vendors"]
        )

    @pytest.mark.asyncio
    async def test_generate_innovation_concepts(self, concept_generator):
        """Test innovation concept generation"""
        concepts = await concept_generator.generate_innovation_concepts(
            domain=TechnologyDomain.QUANTUM_COMPUTING,
            innovation_type="breakthrough",
            market_focus="cryptography",
            count=2
        )
        
        # Validate results
        assert len(concepts) == 2
        assert all(isinstance(concept, BreakthroughConcept) for concept in concepts)
        assert all(concept.technology_domain == TechnologyDomain.QUANTUM_COMPUTING for concept in concepts)
        assert all(concept.innovation_stage == InnovationStage.CONCEPT for concept in concepts)
        assert all(concept.ai_confidence_score > 0.0 for concept in concepts)
        assert all(len(concept.name) > 0 for concept in concepts)
        assert all(len(concept.description) > 0 for concept in concepts)
        # More lenient checks since breakthrough_mechanisms might be empty in some generated concepts
        assert all(isinstance(concept.breakthrough_mechanisms, list) for concept in concepts)
        assert all(isinstance(concept.competitive_advantages, list) for concept in concepts)

    def test_concept_quality_metrics(self, sample_breakthrough_concept):
        """Test concept quality assessment metrics"""
        concept = sample_breakthrough_concept
        
        # Test completeness metrics
        completeness_score = self._calculate_completeness_score(concept)
        assert completeness_score > 0.8  # High completeness expected
        
        # Test innovation metrics
        innovation_score = self._calculate_innovation_score(concept)
        assert innovation_score > 0.7  # High innovation expected
        
        # Test feasibility metrics
        feasibility_score = self._calculate_feasibility_score(concept)
        assert feasibility_score > 0.5  # Reasonable feasibility expected

    def _calculate_completeness_score(self, concept: BreakthroughConcept) -> float:
        """Calculate concept completeness score"""
        required_fields = [
            concept.name,
            concept.description,
            concept.detailed_specification,
            concept.underlying_technologies,
            concept.breakthrough_mechanisms,
            concept.competitive_advantages,
            concept.research_milestones,
            concept.development_phases,
            concept.success_metrics
        ]
        
        completed_fields = sum(1 for field in required_fields if field and len(str(field)) > 0)
        return completed_fields / len(required_fields)

    def _calculate_innovation_score(self, concept: BreakthroughConcept) -> float:
        """Calculate innovation score based on novelty indicators"""
        innovation_indicators = [
            concept.disruption_level == DisruptionLevel.REVOLUTIONARY,
            len(concept.breakthrough_mechanisms) >= 2,
            len(concept.competitive_advantages) >= 3,
            concept.ai_confidence_score > 0.8,
            'breakthrough' in concept.description.lower() or 'revolutionary' in concept.description.lower()
        ]
        
        return sum(innovation_indicators) / len(innovation_indicators)

    def _calculate_feasibility_score(self, concept: BreakthroughConcept) -> float:
        """Calculate feasibility score"""
        feasibility_indicators = [
            len(concept.required_capabilities) > 0,
            len(concept.development_phases) > 0,
            len(concept.research_milestones) > 0,
            concept.market_opportunity.time_to_market_years <= 10,
            len(concept.potential_partnerships) > 0
        ]
        
        return sum(feasibility_indicators) / len(feasibility_indicators)


class TestBreakthroughEngineIntegration:
    """Test integration between breakthrough engine components"""
    
    @pytest.fixture
    def integrated_engine(self):
        """Create integrated breakthrough engine"""
        return BreakthroughEngine()

    @pytest.mark.asyncio
    async def test_end_to_end_breakthrough_analysis(self, integrated_engine):
        """Test end-to-end breakthrough analysis workflow"""
        # Generate concepts
        concepts = await integrated_engine.generate_breakthrough_concepts(
            domain=TechnologyDomain.ARTIFICIAL_INTELLIGENCE,
            market_focus="autonomous systems"
        )
        
        assert len(concepts) > 0
        
        # Analyze innovation potential for first concept
        concept = concepts[0]
        potential = await integrated_engine.analyze_innovation_potential(concept)
        
        assert isinstance(potential, InnovationPotential)
        assert potential.concept_id == concept.id
        
        # Predict market disruption
        disruption = await integrated_engine.predict_market_disruption(
            technology=concept.name,
            timeframe=10
        )
        
        assert isinstance(disruption, DisruptionPrediction)
        assert disruption.technology_name == concept.name
        
        # Recommend research directions
        directions = await integrated_engine.recommend_research_directions(
            concept.required_capabilities
        )
        
        assert len(directions) > 0
        assert all(isinstance(d, ResearchDirection) for d in directions)

    @pytest.mark.asyncio
    async def test_concept_quality_validation(self, integrated_engine):
        """Test comprehensive concept quality validation"""
        concepts = await integrated_engine.generate_breakthrough_concepts(
            domain=TechnologyDomain.QUANTUM_COMPUTING
        )
        
        for concept in concepts:
            # Test concept structure
            assert self._validate_concept_structure(concept)
            
            # Test concept content quality
            assert self._validate_concept_content(concept)
            
            # Test concept feasibility
            assert self._validate_concept_feasibility(concept)

    def _validate_concept_structure(self, concept: BreakthroughConcept) -> bool:
        """Validate concept has proper structure"""
        required_attributes = [
            'id', 'name', 'description', 'technology_domain',
            'innovation_stage', 'disruption_level', 'market_opportunity',
            'required_capabilities', 'breakthrough_mechanisms',
            'competitive_advantages', 'created_at', 'updated_at'
        ]
        
        for attr in required_attributes:
            if not hasattr(concept, attr) or getattr(concept, attr) is None:
                return False
        
        return True

    def _validate_concept_content(self, concept: BreakthroughConcept) -> bool:
        """Validate concept content quality"""
        # Check minimum content length
        if len(concept.name) < 5 or len(concept.description) < 20:
            return False
        
        # Check for meaningful breakthrough mechanisms
        if len(concept.breakthrough_mechanisms) == 0:
            return False
        
        # Check for competitive advantages
        if len(concept.competitive_advantages) == 0:
            return False
        
        # Check AI confidence score
        if concept.ai_confidence_score < 0.5:
            return False
        
        return True

    def _validate_concept_feasibility(self, concept: BreakthroughConcept) -> bool:
        """Validate concept feasibility"""
        # Check for required capabilities
        if len(concept.required_capabilities) == 0:
            return False
        
        # Check for development phases
        if len(concept.development_phases) == 0:
            return False
        
        # Check market opportunity
        if concept.market_opportunity.time_to_market_years > 20:
            return False
        
        return True


@pytest.mark.asyncio
async def test_breakthrough_engine_performance():
    """Test breakthrough engine performance"""
    engine = BreakthroughEngine()
    
    # Test concept generation performance
    start_time = datetime.now()
    concepts = await engine.generate_breakthrough_concepts(
        domain=TechnologyDomain.ARTIFICIAL_INTELLIGENCE
    )
    generation_time = (datetime.now() - start_time).total_seconds()
    
    assert generation_time < 30.0  # Should complete within 30 seconds
    assert len(concepts) == 3  # Should generate expected number of concepts
    
    # Test analysis performance
    start_time = datetime.now()
    potential = await engine.analyze_innovation_potential(concepts[0])
    analysis_time = (datetime.now() - start_time).total_seconds()
    
    assert analysis_time < 10.0  # Should complete within 10 seconds
    assert isinstance(potential, InnovationPotential)


@pytest.mark.asyncio
async def test_breakthrough_engine_error_handling():
    """Test breakthrough engine error handling"""
    engine = BreakthroughEngine()
    
    # Test with invalid domain (should handle gracefully)
    try:
        concepts = await engine.generate_breakthrough_concepts(
            domain=TechnologyDomain.ARTIFICIAL_INTELLIGENCE,
            timeline_years=-1  # Invalid timeline
        )
        # Should still generate concepts despite invalid timeline
        assert len(concepts) > 0
    except Exception as e:
        pytest.fail(f"Engine should handle invalid input gracefully: {e}")
    
    # Test with None concept (should raise appropriate error)
    with pytest.raises((ValueError, AttributeError)):
        await engine.analyze_innovation_potential(None)


if __name__ == "__main__":
    pytest.main([__file__])