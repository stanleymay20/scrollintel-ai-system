"""
Tests for Rapid Prototyping System

This module contains comprehensive tests for the autonomous innovation lab's
rapid prototyping capabilities.
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock

from scrollintel.engines.rapid_prototyper import (
    RapidPrototyper, TechnologySelector, PrototypeGenerator, 
    QualityController, PrototypingTechnology
)
from scrollintel.models.prototype_models import (
    Concept, Prototype, PrototypeType, PrototypeStatus,
    ConceptCategory, TechnologyStack, QualityMetrics,
    ValidationResult, create_concept_from_description
)


class TestTechnologySelector:
    """Test technology selection functionality"""
    
    @pytest.fixture
    def technology_selector(self):
        return TechnologySelector()
    
    @pytest.fixture
    def sample_concept(self):
        return create_concept_from_description(
            name="Web Dashboard",
            description="A web-based analytics dashboard with real-time data visualization",
            category=ConceptCategory.PRODUCT
        )
    
    @pytest.mark.asyncio
    async def test_select_optimal_technology_web(self, technology_selector, sample_concept):
        """Test technology selection for web applications"""
        technology_stack = await technology_selector.select_optimal_technology(sample_concept)
        
        assert technology_stack is not None
        assert technology_stack.primary_technology in [
            PrototypingTechnology.WEB_FRONTEND.value,
            PrototypingTechnology.API_SERVICE.value
        ]
        assert technology_stack.framework is not None
        assert technology_stack.language is not None
    
    @pytest.mark.asyncio
    async def test_select_optimal_technology_ml(self, technology_selector):
        """Test technology selection for ML applications"""
        ml_concept = create_concept_from_description(
            name="ML Model",
            description="A machine learning model for predictive analytics with neural networks",
            category=ConceptCategory.TECHNOLOGY
        )
        
        technology_stack = await technology_selector.select_optimal_technology(ml_concept)
        
        assert technology_stack is not None
        assert technology_stack.primary_technology == PrototypingTechnology.ML_MODEL.value
        assert "Python" in technology_stack.language
    
    @pytest.mark.asyncio
    async def test_select_optimal_technology_mobile(self, technology_selector):
        """Test technology selection for mobile applications"""
        mobile_concept = create_concept_from_description(
            name="Mobile App",
            description="A mobile application for iOS and Android with real-time features",
            category=ConceptCategory.PRODUCT
        )
        
        technology_stack = await technology_selector.select_optimal_technology(mobile_concept)
        
        assert technology_stack is not None
        assert technology_stack.primary_technology == PrototypingTechnology.MOBILE_APP.value
    
    def test_analyze_concept_requirements(self, technology_selector, sample_concept):
        """Test concept requirements analysis"""
        requirements = technology_selector._analyze_concept_requirements(sample_concept)
        
        assert isinstance(requirements, dict)
        assert "complexity" in requirements
        assert "scalability_need" in requirements
        assert "user_interface_need" in requirements
        assert all(0 <= v <= 1 for v in requirements.values())
    
    def test_calculate_technology_score(self, technology_selector):
        """Test technology scoring calculation"""
        requirements = {
            "complexity": 0.8,
            "scalability_need": 0.9,
            "user_interface_need": 0.7,
            "data_processing_need": 0.6,
            "real_time_need": 0.5,
            "mobile_need": 0.3,
            "ai_ml_need": 0.2
        }
        
        profile = {
            "complexity_support": 0.9,
            "scalability": 0.8,
            "deployment_ease": 0.7,
            "development_time": 2.0
        }
        
        score = technology_selector._calculate_technology_score(requirements, profile)
        assert isinstance(score, float)
        assert score >= 0


class TestPrototypeGenerator:
    """Test prototype code generation functionality"""
    
    @pytest.fixture
    def prototype_generator(self):
        return PrototypeGenerator()
    
    @pytest.fixture
    def sample_concept(self):
        return create_concept_from_description(
            name="API Service",
            description="A REST API service for data management",
            category=ConceptCategory.TECHNOLOGY
        )
    
    @pytest.fixture
    def api_technology_stack(self):
        return TechnologyStack(
            primary_technology=PrototypingTechnology.API_SERVICE.value,
            framework="FastAPI",
            language="Python",
            supporting_tools=["Docker", "PostgreSQL"],
            deployment_target="cloud"
        )
    
    @pytest.mark.asyncio
    async def test_generate_prototype_code_api(self, prototype_generator, sample_concept, api_technology_stack):
        """Test API prototype code generation"""
        generated_code = await prototype_generator.generate_prototype_code(
            sample_concept, api_technology_stack
        )
        
        assert isinstance(generated_code, dict)
        assert len(generated_code) > 0
        assert "main" in generated_code
        assert "FastAPI" in generated_code["main"]
        assert "requirements" in generated_code
    
    @pytest.mark.asyncio
    async def test_generate_prototype_code_web(self, prototype_generator):
        """Test web frontend prototype code generation"""
        web_concept = create_concept_from_description(
            name="Dashboard",
            description="A web dashboard for analytics",
            category=ConceptCategory.PRODUCT
        )
        
        web_stack = TechnologyStack(
            primary_technology=PrototypingTechnology.WEB_FRONTEND.value,
            framework="React",
            language="TypeScript",
            supporting_tools=["Webpack", "Jest"],
            deployment_target="cdn"
        )
        
        generated_code = await prototype_generator.generate_prototype_code(web_concept, web_stack)
        
        assert isinstance(generated_code, dict)
        assert len(generated_code) > 0
        if "app" in generated_code:
            assert "React" in generated_code["app"]
    
    def test_extract_template_variables(self, prototype_generator, sample_concept):
        """Test template variable extraction"""
        variables = prototype_generator._extract_template_variables(sample_concept)
        
        assert isinstance(variables, dict)
        assert "title" in variables
        assert "description" in variables
        assert "name" in variables
        assert "model_name" in variables
        assert variables["title"] == sample_concept.name


class TestQualityController:
    """Test prototype quality control functionality"""
    
    @pytest.fixture
    def quality_controller(self):
        return QualityController()
    
    @pytest.fixture
    def sample_prototype(self):
        concept = create_concept_from_description(
            name="Test Prototype",
            description="A test prototype for validation",
            category=ConceptCategory.TECHNOLOGY
        )
        
        return Prototype(
            concept_id=concept.id,
            name="Test Prototype",
            description="A test prototype",
            prototype_type=PrototypeType.API_SERVICE,
            status=PrototypeStatus.FUNCTIONAL,
            quality_metrics=QualityMetrics(
                code_coverage=0.8,
                performance_score=0.7,
                usability_score=0.6,
                reliability_score=0.8
            ),
            error_handling_implemented=True,
            documentation="Test documentation for the prototype"
        )
    
    @pytest.mark.asyncio
    async def test_validate_prototype_quality(self, quality_controller, sample_prototype):
        """Test prototype quality validation"""
        validation_result = await quality_controller.validate_prototype_quality(sample_prototype)
        
        assert isinstance(validation_result, ValidationResult)
        assert validation_result.prototype_id == sample_prototype.id
        assert isinstance(validation_result.overall_score, float)
        assert 0 <= validation_result.overall_score <= 1
        assert isinstance(validation_result.passes_validation, bool)
        assert isinstance(validation_result.recommendations, list)
    
    @pytest.mark.asyncio
    async def test_validate_code_quality(self, quality_controller, sample_prototype):
        """Test code quality validation"""
        score = await quality_controller._validate_code_quality(sample_prototype)
        
        assert isinstance(score, float)
        assert 0 <= score <= 1
    
    @pytest.mark.asyncio
    async def test_validate_functionality(self, quality_controller, sample_prototype):
        """Test functionality validation"""
        score = await quality_controller._validate_functionality(sample_prototype)
        
        assert isinstance(score, float)
        assert 0 <= score <= 1
    
    @pytest.mark.asyncio
    async def test_validate_usability(self, quality_controller, sample_prototype):
        """Test usability validation"""
        score = await quality_controller._validate_usability(sample_prototype)
        
        assert isinstance(score, float)
        assert 0 <= score <= 1
    
    def test_generate_recommendations(self, quality_controller):
        """Test recommendation generation"""
        validation_results = {
            "code_quality": 0.6,
            "functionality": 0.8,
            "usability": 0.5
        }
        
        recommendations = quality_controller._generate_recommendations(validation_results)
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        assert any("code quality" in rec.lower() for rec in recommendations)
        assert any("usability" in rec.lower() for rec in recommendations)


class TestRapidPrototyper:
    """Test main rapid prototyper functionality"""
    
    @pytest.fixture
    def rapid_prototyper(self):
        return RapidPrototyper()
    
    @pytest.fixture
    def sample_concept(self):
        return create_concept_from_description(
            name="Innovation Prototype",
            description="An innovative solution for automated testing",
            category=ConceptCategory.TECHNOLOGY
        )
    
    @pytest.mark.asyncio
    async def test_create_rapid_prototype(self, rapid_prototyper, sample_concept):
        """Test rapid prototype creation"""
        prototype = await rapid_prototyper.create_rapid_prototype(sample_concept)
        
        assert isinstance(prototype, Prototype)
        assert prototype.concept_id == sample_concept.id
        assert prototype.status in [PrototypeStatus.FUNCTIONAL, PrototypeStatus.VALIDATED]
        assert prototype.technology_stack is not None
        assert len(prototype.generated_code) > 0
        assert prototype.development_progress == 1.0
    
    @pytest.mark.asyncio
    async def test_get_prototype_status(self, rapid_prototyper, sample_concept):
        """Test prototype status retrieval"""
        prototype = await rapid_prototyper.create_rapid_prototype(sample_concept)
        
        retrieved_prototype = await rapid_prototyper.get_prototype_status(prototype.id)
        
        assert retrieved_prototype is not None
        assert retrieved_prototype.id == prototype.id
        assert retrieved_prototype.status == prototype.status
    
    @pytest.mark.asyncio
    async def test_list_active_prototypes(self, rapid_prototyper, sample_concept):
        """Test listing active prototypes"""
        prototype = await rapid_prototyper.create_rapid_prototype(sample_concept)
        
        prototypes = await rapid_prototyper.list_active_prototypes()
        
        assert isinstance(prototypes, list)
        assert len(prototypes) > 0
        assert any(p.id == prototype.id for p in prototypes)
    
    @pytest.mark.asyncio
    async def test_optimize_prototype(self, rapid_prototyper, sample_concept):
        """Test prototype optimization"""
        prototype = await rapid_prototyper.create_rapid_prototype(sample_concept)
        
        optimized_prototype = await rapid_prototyper.optimize_prototype(prototype.id)
        
        assert optimized_prototype is not None
        assert optimized_prototype.id == prototype.id
        assert optimized_prototype.validation_result is not None
    
    @pytest.mark.asyncio
    async def test_optimize_nonexistent_prototype(self, rapid_prototyper):
        """Test optimization of non-existent prototype"""
        with pytest.raises(ValueError, match="not found"):
            await rapid_prototyper.optimize_prototype("nonexistent-id")
    
    def test_calculate_complexity_multiplier(self, rapid_prototyper, sample_concept):
        """Test complexity multiplier calculation"""
        multiplier = rapid_prototyper._calculate_complexity_multiplier(sample_concept)
        
        assert isinstance(multiplier, float)
        assert 1.0 <= multiplier <= 3.0
    
    def test_determine_prototype_type(self, rapid_prototyper):
        """Test prototype type determination"""
        web_stack = TechnologyStack(primary_technology=PrototypingTechnology.WEB_FRONTEND.value)
        api_stack = TechnologyStack(primary_technology=PrototypingTechnology.API_SERVICE.value)
        ml_stack = TechnologyStack(primary_technology=PrototypingTechnology.ML_MODEL.value)
        
        assert rapid_prototyper._determine_prototype_type(web_stack) == PrototypeType.WEB_APP
        assert rapid_prototyper._determine_prototype_type(api_stack) == PrototypeType.API_SERVICE
        assert rapid_prototyper._determine_prototype_type(ml_stack) == PrototypeType.ML_MODEL
    
    @pytest.mark.asyncio
    async def test_create_prototyping_plan(self, rapid_prototyper, sample_concept):
        """Test prototyping plan creation"""
        technology_stack = TechnologyStack(
            primary_technology=PrototypingTechnology.API_SERVICE.value,
            framework="FastAPI",
            language="Python"
        )
        
        plan = await rapid_prototyper._create_prototyping_plan(sample_concept, technology_stack)
        
        assert plan.concept_id == sample_concept.id
        assert plan.technology_stack == technology_stack
        assert len(plan.development_phases) > 0
        assert plan.estimated_duration > 0
        assert plan.quality_targets is not None
        assert len(plan.validation_criteria) > 0


class TestPrototypeModels:
    """Test prototype data models"""
    
    def test_concept_creation(self):
        """Test concept model creation"""
        concept = create_concept_from_description(
            name="Test Concept",
            description="A test concept",
            category=ConceptCategory.PRODUCT
        )
        
        assert concept.name == "Test Concept"
        assert concept.description == "A test concept"
        assert concept.category == ConceptCategory.PRODUCT
        assert concept.id is not None
        assert isinstance(concept.creation_timestamp, datetime)
    
    def test_prototype_creation(self):
        """Test prototype model creation"""
        prototype = Prototype(
            name="Test Prototype",
            description="A test prototype",
            prototype_type=PrototypeType.WEB_APP,
            status=PrototypeStatus.PLANNED
        )
        
        assert prototype.name == "Test Prototype"
        assert prototype.prototype_type == PrototypeType.WEB_APP
        assert prototype.status == PrototypeStatus.PLANNED
        assert prototype.id is not None
        assert isinstance(prototype.creation_timestamp, datetime)
    
    def test_technology_stack_creation(self):
        """Test technology stack model creation"""
        stack = TechnologyStack(
            primary_technology="web_frontend",
            framework="React",
            language="TypeScript",
            supporting_tools=["Webpack", "Jest"],
            deployment_target="cdn"
        )
        
        assert stack.primary_technology == "web_frontend"
        assert stack.framework == "React"
        assert stack.language == "TypeScript"
        assert "Webpack" in stack.supporting_tools
    
    def test_quality_metrics_creation(self):
        """Test quality metrics model creation"""
        metrics = QualityMetrics(
            code_coverage=0.8,
            performance_score=0.7,
            usability_score=0.9,
            reliability_score=0.8
        )
        
        assert metrics.code_coverage == 0.8
        assert metrics.performance_score == 0.7
        assert metrics.usability_score == 0.9
        assert metrics.reliability_score == 0.8
    
    def test_validation_result_creation(self):
        """Test validation result model creation"""
        result = ValidationResult(
            prototype_id="test-id",
            overall_score=0.8,
            category_scores={"code_quality": 0.9, "functionality": 0.7},
            passes_validation=True,
            recommendations=["Improve performance"]
        )
        
        assert result.prototype_id == "test-id"
        assert result.overall_score == 0.8
        assert result.passes_validation is True
        assert "Improve performance" in result.recommendations


# Integration tests

class TestRapidPrototypingIntegration:
    """Integration tests for rapid prototyping system"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_prototype_creation(self):
        """Test complete prototype creation workflow"""
        # Create concept
        concept = create_concept_from_description(
            name="E-commerce API",
            description="A REST API for e-commerce platform with user management and product catalog",
            category=ConceptCategory.PRODUCT
        )
        
        # Create rapid prototyper
        rapid_prototyper = RapidPrototyper()
        
        # Create prototype
        prototype = await rapid_prototyper.create_rapid_prototype(concept)
        
        # Verify prototype creation
        assert prototype is not None
        assert prototype.concept_id == concept.id
        assert prototype.status in [PrototypeStatus.FUNCTIONAL, PrototypeStatus.VALIDATED]
        assert prototype.technology_stack is not None
        assert len(prototype.generated_code) > 0
        
        # Verify prototype can be retrieved
        retrieved_prototype = await rapid_prototyper.get_prototype_status(prototype.id)
        assert retrieved_prototype is not None
        assert retrieved_prototype.id == prototype.id
        
        # Verify prototype optimization
        optimized_prototype = await rapid_prototyper.optimize_prototype(prototype.id)
        assert optimized_prototype is not None
        assert optimized_prototype.validation_result is not None
    
    @pytest.mark.asyncio
    async def test_multiple_prototype_creation(self):
        """Test creating multiple prototypes simultaneously"""
        concepts = [
            create_concept_from_description(
                name=f"Concept {i}",
                description=f"Test concept {i} for parallel processing",
                category=ConceptCategory.TECHNOLOGY
            )
            for i in range(3)
        ]
        
        rapid_prototyper = RapidPrototyper()
        
        # Create prototypes in parallel
        tasks = [
            rapid_prototyper.create_rapid_prototype(concept)
            for concept in concepts
        ]
        
        prototypes = await asyncio.gather(*tasks)
        
        # Verify all prototypes were created
        assert len(prototypes) == 3
        for i, prototype in enumerate(prototypes):
            assert prototype.concept_id == concepts[i].id
            assert prototype.status in [PrototypeStatus.FUNCTIONAL, PrototypeStatus.VALIDATED]
        
        # Verify all prototypes are listed
        active_prototypes = await rapid_prototyper.list_active_prototypes()
        assert len(active_prototypes) >= 3
    
    @pytest.mark.asyncio
    async def test_prototype_quality_validation_workflow(self):
        """Test complete quality validation workflow"""
        concept = create_concept_from_description(
            name="Quality Test Prototype",
            description="A prototype for testing quality validation workflow",
            category=ConceptCategory.TECHNOLOGY
        )
        
        rapid_prototyper = RapidPrototyper()
        prototype = await rapid_prototyper.create_rapid_prototype(concept)
        
        # Verify validation was performed
        assert prototype.validation_result is not None
        assert isinstance(prototype.validation_result.overall_score, float)
        assert 0 <= prototype.validation_result.overall_score <= 1
        
        # Test optimization improves quality
        initial_score = prototype.validation_result.overall_score
        optimized_prototype = await rapid_prototyper.optimize_prototype(prototype.id)
        
        # Quality should be maintained or improved
        assert optimized_prototype.validation_result.overall_score >= initial_score


if __name__ == "__main__":
    pytest.main([__file__, "-v"])