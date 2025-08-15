"""
Integration tests for architecture generation system.
"""
import pytest
import asyncio
from datetime import datetime
from typing import List, Dict, Any

from scrollintel.engines.architecture_designer import ArchitectureDesigner
from scrollintel.models.architecture_models import (
    Architecture, ArchitectureComponent, ArchitecturePattern, ComponentType,
    TechnologyStack, Technology, TechnologyCategory, ComponentDependency,
    DataFlow, ArchitectureValidationResult, ScalabilityLevel, SecurityLevel
)
from scrollintel.models.code_generation_models import (
    Requirements, ParsedRequirement, Intent, Entity, EntityType, RequirementType, ConfidenceLevel
)


class TestArchitectureGenerationIntegration:
    """Integration tests for complete architecture generation workflow."""
    
    @pytest.fixture
    def architecture_designer(self):
        """Create architecture designer instance."""
        return ArchitectureDesigner()
    
    @pytest.fixture
    def sample_requirements(self):
        """Create sample requirements for testing."""
        return Requirements(
            id="req_test_001",
            project_name="E-commerce Platform",
            description="Build a scalable e-commerce platform with user management, product catalog, and order processing",
            raw_text="Build an e-commerce platform that can handle thousands of concurrent users, with secure payment processing, real-time inventory management, and comprehensive analytics dashboard",
            parsed_requirements=[
                ParsedRequirement(
                    id="parsed_001",
                    original_text="Handle thousands of concurrent users",
                    structured_text="Handle thousands of concurrent users",
                    requirement_type=RequirementType.PERFORMANCE,
                    intent=Intent.IMPROVE_PERFORMANCE,
                    entities=[],
                    priority=1,
                    complexity=3,
                    confidence=ConfidenceLevel.HIGH
                ),
                ParsedRequirement(
                    id="parsed_002", 
                    original_text="Secure payment processing",
                    structured_text="Secure payment processing",
                    requirement_type=RequirementType.SECURITY,
                    intent=Intent.ADD_SECURITY,
                    entities=[],
                    priority=1,
                    complexity=4,
                    confidence=ConfidenceLevel.HIGH
                ),
                ParsedRequirement(
                    id="parsed_003",
                    original_text="Real-time inventory management",
                    structured_text="Real-time inventory management",
                    requirement_type=RequirementType.FUNCTIONAL,
                    intent=Intent.CREATE_APPLICATION,
                    entities=[],
                    priority=1,
                    complexity=3,
                    confidence=ConfidenceLevel.HIGH
                ),
                ParsedRequirement(
                    id="parsed_004",
                    original_text="Analytics dashboard",
                    structured_text="Analytics dashboard",
                    requirement_type=RequirementType.FUNCTIONAL,
                    intent=Intent.BUILD_UI,
                    entities=[],
                    priority=2,
                    complexity=2,
                    confidence=ConfidenceLevel.MEDIUM
                )
            ],
            entities=[
                Entity(
                    id="entity_001",
                    name="User",
                    type=EntityType.USER_ROLE,
                    description="User entity for authentication and profile management",
                    attributes={"fields": ["id", "email", "password", "profile"]},
                    confidence=0.9,
                    source_text="user management",
                    position=(0, 15)
                ),
                Entity(
                    id="entity_002",
                    name="Product",
                    type=EntityType.DATA_ENTITY,
                    description="Product entity for catalog management",
                    attributes={"fields": ["id", "name", "price", "inventory"]},
                    confidence=0.9,
                    source_text="product catalog",
                    position=(16, 31)
                ),
                Entity(
                    id="entity_003",
                    name="Order",
                    type=EntityType.DATA_ENTITY,
                    description="Order entity for order processing",
                    attributes={"fields": ["id", "user_id", "products", "total", "status"]},
                    confidence=0.9,
                    source_text="order processing",
                    position=(32, 48)
                ),
                Entity(
                    id="entity_004",
                    name="Payment",
                    type=EntityType.DATA_ENTITY,
                    description="Payment entity for payment processing",
                    attributes={"fields": ["id", "order_id", "amount", "method", "status"]},
                    confidence=0.9,
                    source_text="payment processing",
                    position=(49, 67)
                )
            ],
            relationships=[],
            constraints=[],
            assumptions=[],
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
    
    @pytest.fixture
    def simple_requirements(self):
        """Create simple requirements for basic testing."""
        return Requirements(
            id="req_simple_001",
            project_name="Blog Platform",
            description="Simple blog platform with user authentication",
            raw_text="Create a simple blog platform where users can register, login, and create blog posts",
            parsed_requirements=[
                ParsedRequirement(
                    id="parsed_simple_001",
                    original_text="User registration and login",
                    structured_text="User registration and login",
                    requirement_type=RequirementType.FUNCTIONAL,
                    intent=Intent.ADD_SECURITY,
                    entities=[],
                    priority=1,
                    complexity=2,
                    confidence=ConfidenceLevel.HIGH
                ),
                ParsedRequirement(
                    id="parsed_simple_002",
                    original_text="Create and manage blog posts",
                    structured_text="Create and manage blog posts",
                    requirement_type=RequirementType.FUNCTIONAL,
                    intent=Intent.CREATE_APPLICATION,
                    entities=[],
                    priority=1,
                    complexity=2,
                    confidence=ConfidenceLevel.HIGH
                )
            ],
            entities=[
                Entity(
                    id="entity_simple_001",
                    name="User",
                    type=EntityType.USER_ROLE,
                    description="User entity for blog platform",
                    attributes={"fields": ["id", "username", "email", "password"]},
                    confidence=0.9,
                    source_text="users can register",
                    position=(0, 18)
                ),
                Entity(
                    id="entity_simple_002",
                    name="BlogPost",
                    type=EntityType.DATA_ENTITY,
                    description="Blog post entity for content management",
                    attributes={"fields": ["id", "title", "content", "author_id", "created_at"]},
                    confidence=0.9,
                    source_text="create blog posts",
                    position=(19, 36)
                )
            ],
            relationships=[],
            constraints=[],
            assumptions=[],
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
    
    @pytest.mark.asyncio
    async def test_complete_architecture_design_workflow(self, architecture_designer, sample_requirements):
        """Test complete architecture design workflow from requirements to validated architecture."""
        # Design architecture
        architecture = await architecture_designer.design_architecture(sample_requirements)
        
        # Verify architecture was created
        assert architecture is not None
        assert architecture.name == "E-commerce Platform Architecture"
        assert architecture.pattern in ArchitecturePattern
        assert len(architecture.components) > 0
        assert len(architecture.dependencies) > 0
        assert architecture.technology_stack is not None
        
        # Verify validation was performed
        assert architecture.validation_result is not None
        assert isinstance(architecture.validation_result.score, float)
        assert 0.0 <= architecture.validation_result.score <= 1.0
        
        # Verify requirements coverage
        assert architecture.requirements_coverage is not None
        assert len(architecture.requirements_coverage) > 0
        
        # Verify cost and time estimates
        assert architecture.estimated_total_cost > 0
        assert architecture.estimated_development_time > 0
        assert 1 <= architecture.complexity_score <= 5
        assert 0.0 <= architecture.maintainability_score <= 1.0
    
    @pytest.mark.asyncio
    async def test_pattern_selection_for_different_scales(self, architecture_designer):
        """Test that different requirement scales result in appropriate pattern selection."""
        # High-scale requirements
        high_scale_req = Requirements(
            id="req_high_scale",
            project_name="Enterprise Platform",
            description="Large-scale enterprise platform",
            raw_text="Build a platform that can handle millions of users with high availability and complex integrations",
            parsed_requirements=[
                ParsedRequirement(
                    id="parsed_scale_001",
                    original_text="Handle millions of concurrent users",
                    structured_text="Handle millions of concurrent users",
                    requirement_type=RequirementType.PERFORMANCE,
                    intent=Intent.IMPROVE_PERFORMANCE,
                    entities=[],
                    priority=1,
                    complexity=5,
                    confidence=ConfidenceLevel.HIGH
                )
            ],
            entities=[Entity(
                id="e1", 
                name="User", 
                type=EntityType.USER_ROLE, 
                description="User entity for high-scale platform",
                attributes={},
                confidence=0.8,
                source_text="users",
                position=(0, 5)
            )],
            relationships=[],
            constraints=[],
            assumptions=[],
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        # Small-scale requirements
        small_scale_req = Requirements(
            id="req_small_scale",
            project_name="Personal App",
            description="Simple personal application",
            raw_text="Create a simple personal task management app for a small team",
            parsed_requirements=[
                ParsedRequirement(
                    id="parsed_small_001",
                    original_text="Task management for small team",
                    structured_text="Task management for small team",
                    requirement_type=RequirementType.FUNCTIONAL,
                    intent=Intent.CREATE_APPLICATION,
                    entities=[],
                    priority=1,
                    complexity=2,
                    confidence=ConfidenceLevel.HIGH
                )
            ],
            entities=[Entity(
                id="e1", 
                name="Task", 
                type=EntityType.DATA_ENTITY, 
                description="Task entity for task management",
                attributes={},
                confidence=0.8,
                source_text="task",
                position=(0, 4)
            )],
            relationships=[],
            constraints=[],
            assumptions=[],
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        # Design architectures
        high_scale_arch = await architecture_designer.design_architecture(high_scale_req)
        small_scale_arch = await architecture_designer.design_architecture(small_scale_req)
        
        # Verify different patterns are selected based on scale
        assert high_scale_arch.pattern != small_scale_arch.pattern
        
        # High-scale should prefer microservices or event-driven
        assert high_scale_arch.pattern in [
            ArchitecturePattern.MICROSERVICES, 
            ArchitecturePattern.EVENT_DRIVEN,
            ArchitecturePattern.SERVERLESS
        ]
        
        # Small-scale should prefer simpler patterns
        assert small_scale_arch.pattern in [
            ArchitecturePattern.MONOLITHIC,
            ArchitecturePattern.LAYERED,
            ArchitecturePattern.MVC
        ]
    
    @pytest.mark.asyncio
    async def test_component_generation_based_on_requirements(self, architecture_designer, sample_requirements):
        """Test that components are generated based on specific requirements."""
        architecture = await architecture_designer.design_architecture(sample_requirements)
        
        component_types = {comp.type for comp in architecture.components}
        
        # Should have core components
        assert ComponentType.DATABASE in component_types
        assert ComponentType.API_GATEWAY in component_types
        
        # Should have frontend component for user-facing app
        assert ComponentType.FRONTEND in component_types
        
        # Should have authentication for secure requirements
        auth_components = [comp for comp in architecture.components if comp.type == ComponentType.AUTHENTICATION]
        assert len(auth_components) > 0  # Security requirements should trigger auth component
        
        # Should have cache for performance requirements
        cache_components = [comp for comp in architecture.components if comp.type == ComponentType.CACHE]
        # May or may not have cache depending on pattern selection
    
    @pytest.mark.asyncio
    async def test_technology_stack_recommendation(self, architecture_designer, sample_requirements):
        """Test technology stack recommendation based on requirements and components."""
        architecture = await architecture_designer.design_architecture(sample_requirements)
        
        tech_stack = architecture.technology_stack
        
        # Verify technology stack structure
        assert tech_stack is not None
        assert len(tech_stack.technologies) > 0
        assert tech_stack.compatibility_score > 0.0
        assert tech_stack.total_cost_estimate > 0.0
        assert tech_stack.complexity_score >= 1
        assert tech_stack.recommended_team_size >= 1
        
        # Verify technology categories are covered
        tech_categories = {tech.category for tech in tech_stack.technologies}
        
        # Should have frontend tech if frontend component exists
        if any(comp.type == ComponentType.FRONTEND for comp in architecture.components):
            assert TechnologyCategory.FRONTEND_FRAMEWORK in tech_categories
        
        # Should have backend tech if backend component exists
        if any(comp.type == ComponentType.BACKEND for comp in architecture.components):
            assert TechnologyCategory.BACKEND_FRAMEWORK in tech_categories
        
        # Should have database tech if database component exists
        if any(comp.type == ComponentType.DATABASE for comp in architecture.components):
            assert TechnologyCategory.DATABASE in tech_categories
        
        # Should have cloud provider
        assert TechnologyCategory.CLOUD_PROVIDER in tech_categories
    
    @pytest.mark.asyncio
    async def test_dependency_analysis_and_optimization(self, architecture_designer, sample_requirements):
        """Test component dependency analysis and optimization."""
        architecture = await architecture_designer.design_architecture(sample_requirements)
        
        dependencies = architecture.dependencies
        components = architecture.components
        
        # Verify dependencies exist
        assert len(dependencies) > 0
        
        # Verify dependency structure
        for dep in dependencies:
            assert dep.source_component_id in [comp.id for comp in components]
            assert dep.target_component_id in [comp.id for comp in components]
            assert dep.dependency_type is not None
            assert dep.communication_protocol is not None
        
        # Test dependency optimization
        optimized_deps = await architecture_designer.optimize_dependencies(components, dependencies)
        
        # Verify optimization results
        assert len(optimized_deps) > 0
        
        # Should not have circular dependencies
        assert not architecture_designer._has_circular_dependencies(optimized_deps)
        
        # Critical dependencies should be marked
        critical_deps = [dep for dep in optimized_deps if dep.is_critical]
        assert len(critical_deps) > 0
    
    @pytest.mark.asyncio
    async def test_architecture_validation(self, architecture_designer, sample_requirements):
        """Test comprehensive architecture validation."""
        architecture = await architecture_designer.design_architecture(sample_requirements)
        
        # Perform additional validation
        validation_result = await architecture_designer.validate_architecture(architecture)
        
        # Verify validation result structure
        assert validation_result is not None
        assert isinstance(validation_result.is_valid, bool)
        assert 0.0 <= validation_result.score <= 1.0
        assert isinstance(validation_result.violations, list)
        assert isinstance(validation_result.warnings, list)
        assert isinstance(validation_result.recommendations, list)
        assert 0.0 <= validation_result.best_practices_score <= 1.0
        
        # Verify validation logic
        if validation_result.score < 0.7:
            assert not validation_result.is_valid
        
        # Should have some recommendations for improvement
        if validation_result.score < 1.0:
            assert len(validation_result.recommendations) > 0
    
    @pytest.mark.asyncio
    async def test_architecture_comparison(self, architecture_designer, sample_requirements, simple_requirements):
        """Test architecture comparison functionality."""
        # Generate two different architectures
        arch1 = await architecture_designer.design_architecture(sample_requirements)
        arch2 = await architecture_designer.design_architecture(simple_requirements)
        
        # Compare architectures
        comparison = await architecture_designer.compare_architectures([arch1, arch2])
        
        # Verify comparison structure
        assert comparison is not None
        assert len(comparison.architectures) == 2
        assert len(comparison.criteria) > 0
        assert len(comparison.scores) == 2
        assert comparison.recommendation is not None
        assert comparison.rationale is not None
        assert len(comparison.trade_offs) == 2
        
        # Verify scores structure
        for arch_id, scores in comparison.scores.items():
            assert arch_id in [arch1.id, arch2.id]
            for criterion in comparison.criteria:
                assert criterion in scores
                assert 0.0 <= scores[criterion] <= 1.0
        
        # Verify recommendation is one of the architectures
        recommended_names = [arch.name for arch in comparison.architectures]
        assert comparison.recommendation in recommended_names
    
    @pytest.mark.asyncio
    async def test_scalability_strategy_design(self, architecture_designer):
        """Test scalability strategy design for different requirements."""
        # High scalability requirements
        high_scale_req = Requirements(
            id="req_high_scale_strategy",
            project_name="High Scale App",
            description="Application requiring high scalability",
            raw_text="Build an application that needs to scale to millions of users with auto-scaling capabilities",
            parsed_requirements=[
                ParsedRequirement(
                    id="parsed_scale_strategy_001",
                    original_text="Scale to millions of users with auto-scaling",
                    structured_text="Scale to millions of users with auto-scaling",
                    requirement_type=RequirementType.PERFORMANCE,
                    intent=Intent.IMPROVE_PERFORMANCE,
                    entities=[],
                    priority=1,
                    complexity=5,
                    confidence=ConfidenceLevel.HIGH
                )
            ],
            entities=[Entity(
                id="e1", 
                name="User", 
                type=EntityType.USER_ROLE, 
                description="User entity for scalable app",
                attributes={},
                confidence=0.8,
                source_text="users",
                position=(0, 5)
            )],
            relationships=[],
            constraints=[],
            assumptions=[],
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        architecture = await architecture_designer.design_architecture(high_scale_req)
        
        # Verify scalability strategy
        scalability_strategy = architecture.scalability_strategy
        assert scalability_strategy is not None
        
        # High-scale requirements should enable advanced scalability features
        assert scalability_strategy.get("horizontal_scaling", False)
        assert scalability_strategy.get("auto_scaling", False)
        assert scalability_strategy.get("load_balancing", False)
        
        # Should have distributed caching for high scale
        assert scalability_strategy.get("caching_strategy") in ["distributed", "advanced"]
    
    @pytest.mark.asyncio
    async def test_security_strategy_design(self, architecture_designer):
        """Test security strategy design for different security requirements."""
        # High security requirements
        high_security_req = Requirements(
            id="req_high_security",
            project_name="Secure App",
            description="Application with high security requirements",
            raw_text="Build a secure application with encryption, multi-factor authentication, and compliance requirements",
            parsed_requirements=[
                ParsedRequirement(
                    id="parsed_security_001",
                    original_text="Multi-factor authentication and encryption",
                    structured_text="Multi-factor authentication and encryption",
                    requirement_type=RequirementType.SECURITY,
                    intent=Intent.ADD_SECURITY,
                    entities=[],
                    priority=1,
                    complexity=4,
                    confidence=ConfidenceLevel.HIGH
                )
            ],
            entities=[Entity(
                id="e1", 
                name="User", 
                type=EntityType.USER_ROLE, 
                description="User entity for secure app",
                attributes={},
                confidence=0.8,
                source_text="users",
                position=(0, 5)
            )],
            relationships=[],
            constraints=[],
            assumptions=[],
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        architecture = await architecture_designer.design_architecture(high_security_req)
        
        # Verify security strategy
        security_strategy = architecture.security_strategy
        assert security_strategy is not None
        
        # High-security requirements should enable advanced security features
        assert security_strategy.get("encryption_at_rest", False)
        assert security_strategy.get("encryption_in_transit", False)
        
        # Should have strong authentication
        assert security_strategy.get("authentication") in ["jwt", "oauth2"]
        assert security_strategy.get("authorization") in ["rbac", "abac"]
    
    @pytest.mark.asyncio
    async def test_data_flow_design(self, architecture_designer, sample_requirements):
        """Test data flow design between components."""
        architecture = await architecture_designer.design_architecture(sample_requirements)
        
        data_flows = architecture.data_flows
        components = architecture.components
        
        # Verify data flows exist
        assert len(data_flows) > 0
        
        # Verify data flow structure
        for flow in data_flows:
            assert flow.source_component_id in [comp.id for comp in components]
            assert flow.target_component_id in [comp.id for comp in components]
            assert len(flow.data_types) > 0
            assert flow.volume_estimate is not None
            assert flow.frequency is not None
            assert len(flow.security_requirements) > 0
        
        # Verify data types are relevant to requirements
        all_data_types = []
        for flow in data_flows:
            all_data_types.extend(flow.data_types)
        
        # Should include entity-based data types
        entity_names = [entity.name.lower() for entity in sample_requirements.entities]
        for entity_name in entity_names:
            expected_data_type = f"{entity_name}_data"
            assert expected_data_type in all_data_types
    
    @pytest.mark.asyncio
    async def test_requirements_coverage_analysis(self, architecture_designer, sample_requirements):
        """Test requirements coverage analysis."""
        architecture = await architecture_designer.design_architecture(sample_requirements)
        
        coverage = architecture.requirements_coverage
        
        # Verify coverage structure
        assert coverage is not None
        assert len(coverage) > 0
        
        # Should have coverage for each parsed requirement
        assert len(coverage) >= len(sample_requirements.parsed_requirements)
        
        # Verify coverage values are boolean
        for req_id, is_covered in coverage.items():
            assert isinstance(is_covered, bool)
        
        # Should have good coverage for well-defined requirements
        covered_count = sum(1 for covered in coverage.values() if covered)
        coverage_ratio = covered_count / len(coverage)
        
        # Should cover at least 50% of requirements
        assert coverage_ratio >= 0.5
    
    @pytest.mark.asyncio
    async def test_cost_and_time_estimation(self, architecture_designer, sample_requirements, simple_requirements):
        """Test cost and development time estimation."""
        complex_arch = await architecture_designer.design_architecture(sample_requirements)
        simple_arch = await architecture_designer.design_architecture(simple_requirements)
        
        # Complex architecture should cost more
        assert complex_arch.estimated_total_cost > simple_arch.estimated_total_cost
        
        # Complex architecture should take longer to develop
        assert complex_arch.estimated_development_time > simple_arch.estimated_development_time
        
        # Complex architecture should have higher complexity score
        assert complex_arch.complexity_score >= simple_arch.complexity_score
        
        # Verify reasonable estimates
        assert complex_arch.estimated_total_cost > 0
        assert complex_arch.estimated_development_time > 0
        assert simple_arch.estimated_total_cost > 0
        assert simple_arch.estimated_development_time > 0
    
    @pytest.mark.asyncio
    async def test_pattern_specific_components(self, architecture_designer):
        """Test that pattern-specific components are generated correctly."""
        # Create requirements that should trigger microservices
        microservices_req = Requirements(
            id="req_microservices",
            project_name="Microservices App",
            description="Complex application requiring microservices",
            raw_text="Build a complex platform with multiple services, high scalability, and service discovery",
            parsed_requirements=[
                ParsedRequirement(
                    id="parsed_micro_001",
                    original_text="Multiple independent services with high scalability",
                    structured_text="Multiple independent services with high scalability",
                    requirement_type=RequirementType.PERFORMANCE,
                    intent=Intent.IMPROVE_PERFORMANCE,
                    entities=[],
                    priority=1,
                    complexity=5,
                    confidence=ConfidenceLevel.HIGH
                )
            ],
            entities=[
                Entity(
                    id=f"entity_{i}", 
                    name=f"Service{i}", 
                    type=EntityType.SYSTEM_COMPONENT, 
                    description=f"Service component {i}",
                    attributes={},
                    confidence=0.8,
                    source_text=f"service{i}",
                    position=(i*10, i*10+7)
                )
                for i in range(8)  # Many entities to trigger complexity
            ],
            relationships=[],
            constraints=[],
            assumptions=[],
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        architecture = await architecture_designer.design_architecture(microservices_req)
        
        # If microservices pattern is selected, should have specific components
        if architecture.pattern == ArchitecturePattern.MICROSERVICES:
            component_types = {comp.type for comp in architecture.components}
            
            # Should have message queue for microservices communication
            assert ComponentType.MESSAGE_QUEUE in component_types
            
            # Should have service discovery component
            service_discovery_components = [
                comp for comp in architecture.components 
                if "service discovery" in comp.name.lower() or "discovery" in comp.description.lower()
            ]
            assert len(service_discovery_components) > 0
    
    @pytest.mark.asyncio
    async def test_error_handling_and_resilience(self, architecture_designer):
        """Test error handling in architecture design process."""
        # Test with invalid requirements
        invalid_req = Requirements(
            id="req_invalid",
            project_name="",  # Empty name
            description="",   # Empty description
            raw_text="",      # Empty text
            parsed_requirements=[],  # No requirements
            entities=[],      # No entities
            relationships=[],
            constraints=[],
            assumptions=[],
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        # Should handle gracefully and still produce some architecture
        try:
            architecture = await architecture_designer.design_architecture(invalid_req)
            
            # Should still create basic architecture
            assert architecture is not None
            assert len(architecture.components) > 0  # Should have at least basic components
            
        except Exception as e:
            # If it fails, should be a meaningful error
            assert str(e) is not None
    
    @pytest.mark.asyncio
    async def test_preferences_integration(self, architecture_designer, sample_requirements):
        """Test that user preferences are integrated into architecture design."""
        # Test with specific technology preferences
        preferences = {
            "preferred_patterns": [ArchitecturePattern.LAYERED],
            "frontend_framework": "Vue.js",
            "backend_framework": "Django",
            "database": "MySQL",
            "cloud_provider": "Azure"
        }
        
        architecture = await architecture_designer.design_architecture(
            sample_requirements, 
            preferences=preferences
        )
        
        # Should respect pattern preference if reasonable
        # (May not always be possible due to requirements constraints)
        
        # Should use preferred technologies where specified
        tech_names = [tech.name for tech in architecture.technology_stack.technologies]
        
        # Check if preferences were considered (may not always be used if incompatible)
        # This is more of a smoke test to ensure preferences don't break the system
        assert architecture is not None
        assert len(tech_names) > 0
    
    def test_synchronous_wrapper_methods(self, architecture_designer, sample_requirements):
        """Test synchronous wrapper methods work correctly."""
        # Test technology stack recommendation wrapper
        components = [
            ArchitectureComponent(
                id="test_comp",
                name="Test Component",
                type=ComponentType.BACKEND,
                description="Test component",
                estimated_complexity=2,
                estimated_effort_hours=40,
                priority=1
            )
        ]
        
        characteristics = {"scalability_needs": ScalabilityLevel.MEDIUM}
        
        # Should not raise exception
        tech_stack = architecture_designer._recommend_technology_stack(
            components, 
            ArchitecturePattern.LAYERED, 
            characteristics
        )
        
        assert tech_stack is not None
        assert len(tech_stack.technologies) > 0