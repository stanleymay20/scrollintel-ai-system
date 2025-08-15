"""
Architecture Designer Engine for automated architecture design and pattern selection.
"""
import logging
from typing import List, Dict, Any, Optional, Set, Tuple
from datetime import datetime

from ..models.architecture_models import (
    Architecture, ArchitectureComponent, ArchitecturePattern, ComponentType,
    TechnologyStack, Technology, TechnologyCategory, ComponentDependency,
    DataFlow, ArchitectureValidationResult, ScalabilityLevel, SecurityLevel
)
from ..models.code_generation_models import Requirements, RequirementType, Intent
from .base_engine import BaseEngine

logger = logging.getLogger(__name__)


class ArchitectureDesigner(BaseEngine):
    """Engine for designing system architectures from requirements."""
    
    def __init__(self):
        super().__init__(
            engine_id="architecture_designer",
            name="Architecture Designer",
            capabilities=[]
        )
    
    async def initialize(self) -> None:
        """Initialize the architecture designer engine."""
        logger.info("Initializing Architecture Designer engine")
    
    async def process(self, input_data: Any, parameters: Dict[str, Any] = None) -> Any:
        """Process architecture design request."""
        if isinstance(input_data, Requirements):
            return await self.design_architecture(input_data, parameters)
        else:
            raise ValueError("Input data must be Requirements object")
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        logger.info("Cleaning up Architecture Designer engine")
    
    def get_status(self) -> Dict[str, Any]:
        """Get engine status."""
        return {
            "engine_id": self.engine_id,
            "name": self.name,
            "status": self.status.value,
            "healthy": True
        }
    
    async def design_architecture(
        self, 
        requirements: Requirements,
        preferences: Optional[Dict[str, Any]] = None
    ) -> Architecture:
        """Design a complete architecture from requirements."""
        try:
            logger.info(f"Starting architecture design for project: {requirements.project_name}")
            
            # Analyze requirements
            characteristics = self._analyze_requirements(requirements)
            
            # Select architecture pattern
            pattern = self._select_architecture_pattern(characteristics, preferences)
            
            # Design components
            components = self._design_components(requirements, pattern, characteristics)
            
            # Create dependencies
            dependencies = self._analyze_dependencies(components, requirements)
            
            # Create data flows
            data_flows = self._design_data_flows(components, dependencies, requirements)
            
            # Create technology stack
            tech_stack = self._recommend_technology_stack(components, pattern, characteristics, preferences)
            
            # Create architecture
            architecture = Architecture(
                id=f"arch_{requirements.id}_{int(datetime.utcnow().timestamp())}",
                name=f"{requirements.project_name} Architecture",
                description=f"Generated architecture for {requirements.project_name}",
                pattern=pattern,
                components=components,
                dependencies=dependencies,
                data_flows=data_flows,
                technology_stack=tech_stack,
                scalability_strategy=self._design_scalability_strategy(characteristics),
                security_strategy=self._design_security_strategy(characteristics),
                deployment_strategy=self._design_deployment_strategy(characteristics),
                estimated_total_cost=self._calculate_total_cost(components, tech_stack),
                estimated_development_time=self._estimate_development_time(components),
                complexity_score=self._calculate_complexity_score(components, dependencies),
                maintainability_score=self._calculate_maintainability_score(components, pattern),
                requirements_coverage=self._analyze_requirements_coverage(requirements, components)
            )
            
            # Validate architecture
            validation_result = await self.validate_architecture(architecture)
            architecture.validation_result = validation_result
            
            logger.info(f"Architecture design completed for {requirements.project_name}")
            return architecture
            
        except Exception as e:
            logger.error(f"Architecture design failed: {e}")
            raise
    
    def _analyze_requirements(self, requirements: Requirements) -> Dict[str, Any]:
        """Analyze requirements to extract architecture characteristics."""
        characteristics = {
            "scalability_needs": ScalabilityLevel.MEDIUM,
            "security_needs": SecurityLevel.STANDARD,
            "performance_critical": False,
            "data_intensive": False,
            "user_facing": True,
            "integration_complexity": "medium",
            "team_size": 5
        }
        
        # Analyze parsed requirements for keywords and types
        for req in requirements.parsed_requirements:
            req_text = req.structured_text.lower()
            
            # Check requirement type first
            if req.requirement_type == RequirementType.PERFORMANCE:
                characteristics["performance_critical"] = True
                # Check for high scalability indicators
                if any(keyword in req_text for keyword in ["million", "millions", "scale", "concurrent", "high availability"]):
                    characteristics["scalability_needs"] = ScalabilityLevel.HIGH
                elif any(keyword in req_text for keyword in ["thousand", "thousands", "scalable"]):
                    characteristics["scalability_needs"] = ScalabilityLevel.HIGH
            
            if req.requirement_type == RequirementType.SECURITY:
                characteristics["security_needs"] = SecurityLevel.HIGH
            
            # Keyword-based analysis
            if any(keyword in req_text for keyword in ["million", "millions"]):
                characteristics["scalability_needs"] = ScalabilityLevel.ENTERPRISE
            elif any(keyword in req_text for keyword in ["thousand", "thousands", "concurrent", "scale"]):
                characteristics["scalability_needs"] = ScalabilityLevel.HIGH
            
            if any(keyword in req_text for keyword in ["secure", "authentication", "authorization", "encrypt"]):
                characteristics["security_needs"] = SecurityLevel.HIGH
            
            if any(keyword in req_text for keyword in ["fast", "performance", "real-time", "latency"]):
                characteristics["performance_critical"] = True
            
            if any(keyword in req_text for keyword in ["data", "analytics", "reporting", "dashboard"]):
                characteristics["data_intensive"] = True
            
            # Check complexity based on requirement complexity
            if req.complexity >= 4:
                characteristics["integration_complexity"] = "high"
            elif req.complexity <= 2:
                characteristics["integration_complexity"] = "low"
        
        # Analyze entities for complexity
        entity_count = len(requirements.entities)
        if entity_count > 10:
            characteristics["integration_complexity"] = "high"
        elif entity_count < 3:
            characteristics["integration_complexity"] = "low"
        
        # Analyze project description for additional context
        project_text = f"{requirements.project_name} {requirements.raw_text}".lower()
        if any(keyword in project_text for keyword in ["enterprise", "large-scale", "complex"]):
            characteristics["integration_complexity"] = "high"
            if characteristics["scalability_needs"] == ScalabilityLevel.MEDIUM:
                characteristics["scalability_needs"] = ScalabilityLevel.HIGH
        
        if any(keyword in project_text for keyword in ["simple", "personal", "small"]):
            characteristics["integration_complexity"] = "low"
            if characteristics["scalability_needs"] == ScalabilityLevel.MEDIUM:
                characteristics["scalability_needs"] = ScalabilityLevel.LOW
        
        return characteristics
    
    def _select_architecture_pattern(
        self, 
        characteristics: Dict[str, Any], 
        preferences: Optional[Dict[str, Any]] = None
    ) -> ArchitecturePattern:
        """Select the most appropriate architecture pattern."""
        scores = {}
        
        # Score each pattern
        for pattern in ArchitecturePattern:
            score = self._score_pattern(pattern, characteristics, preferences)
            scores[pattern] = score
        
        # Select pattern with highest score
        best_pattern = max(scores, key=scores.get)
        logger.info(f"Selected architecture pattern: {best_pattern.value}")
        
        return best_pattern
    
    def _score_pattern(
        self, 
        pattern: ArchitecturePattern, 
        characteristics: Dict[str, Any],
        preferences: Optional[Dict[str, Any]] = None
    ) -> float:
        """Score an architecture pattern based on requirements."""
        score = 0.1  # Base score for all patterns
        
        if pattern == ArchitecturePattern.MICROSERVICES:
            if characteristics["scalability_needs"] in [ScalabilityLevel.HIGH, ScalabilityLevel.ENTERPRISE]:
                score += 0.4
            if characteristics["integration_complexity"] == "high":
                score += 0.3
            if characteristics["team_size"] > 8:
                score += 0.2
            else:
                score -= 0.2  # Penalty for small teams
            if characteristics["data_intensive"]:
                score += 0.1
        
        elif pattern == ArchitecturePattern.MONOLITHIC:
            if characteristics["team_size"] <= 5:
                score += 0.4
            if characteristics["integration_complexity"] == "low":
                score += 0.3
            if characteristics["scalability_needs"] in [ScalabilityLevel.LOW, ScalabilityLevel.MEDIUM]:
                score += 0.2
            # Penalty for high scalability needs
            if characteristics["scalability_needs"] in [ScalabilityLevel.HIGH, ScalabilityLevel.ENTERPRISE]:
                score -= 0.3
        
        elif pattern == ArchitecturePattern.LAYERED:
            if characteristics["integration_complexity"] == "medium":
                score += 0.3
            if characteristics["team_size"] <= 8:
                score += 0.2
            score += 0.2  # Generally solid choice
        
        elif pattern == ArchitecturePattern.SERVERLESS:
            if characteristics["scalability_needs"] in [ScalabilityLevel.HIGH, ScalabilityLevel.ENTERPRISE]:
                score += 0.3
            if characteristics["performance_critical"]:
                score -= 0.2  # Cold start penalty
            score += 0.1
        
        elif pattern == ArchitecturePattern.EVENT_DRIVEN:
            if characteristics["scalability_needs"] in [ScalabilityLevel.HIGH, ScalabilityLevel.ENTERPRISE]:
                score += 0.3
            if characteristics["data_intensive"]:
                score += 0.2
            if characteristics["integration_complexity"] == "high":
                score += 0.2
        
        # Apply preferences
        if preferences and "preferred_patterns" in preferences:
            if pattern in preferences["preferred_patterns"]:
                score += 0.3
        
        return max(0.0, min(1.0, score))
    
    def _design_components(
        self, 
        requirements: Requirements, 
        pattern: ArchitecturePattern,
        characteristics: Dict[str, Any]
    ) -> List[ArchitectureComponent]:
        """Design architecture components."""
        components = []
        
        # Core components
        components.extend(self._create_core_components(requirements, characteristics))
        
        # Pattern-specific components
        if pattern == ArchitecturePattern.MICROSERVICES:
            components.extend(self._create_microservices_components(requirements, characteristics))
        
        # Feature-specific components
        if characteristics["data_intensive"]:
            components.extend(self._create_data_components(requirements, characteristics))
        
        if characteristics["user_facing"]:
            components.extend(self._create_frontend_components(requirements, characteristics))
        
        return components
    
    def _create_core_components(
        self, 
        requirements: Requirements, 
        characteristics: Dict[str, Any]
    ) -> List[ArchitectureComponent]:
        """Create core components needed by all architectures."""
        components = []
        
        # API Gateway
        components.append(ArchitectureComponent(
            id="api_gateway",
            name="API Gateway",
            type=ComponentType.API_GATEWAY,
            description="Central entry point for all API requests",
            responsibilities=["Request routing", "Authentication", "Rate limiting"],
            scalability_requirements=characteristics["scalability_needs"],
            security_requirements=characteristics["security_needs"],
            estimated_complexity=2,
            estimated_effort_hours=40,
            priority=1
        ))
        
        # Database
        components.append(ArchitectureComponent(
            id="primary_database",
            name="Primary Database",
            type=ComponentType.DATABASE,
            description="Main data storage system",
            responsibilities=["Data persistence", "Data integrity", "Query processing"],
            scalability_requirements=characteristics["scalability_needs"],
            security_requirements=characteristics["security_needs"],
            estimated_complexity=3,
            estimated_effort_hours=60,
            priority=1
        ))
        
        # Authentication Service for high security needs
        if characteristics["security_needs"] in [SecurityLevel.HIGH, SecurityLevel.ENTERPRISE]:
            components.append(ArchitectureComponent(
                id="auth_service",
                name="Authentication Service",
                type=ComponentType.AUTHENTICATION,
                description="User authentication and authorization",
                responsibilities=["User authentication", "Token management", "Session management"],
                scalability_requirements=characteristics["scalability_needs"],
                security_requirements=characteristics["security_needs"],
                estimated_complexity=4,
                estimated_effort_hours=80,
                priority=1
            ))
        
        return components
    
    def _create_microservices_components(
        self, 
        requirements: Requirements, 
        characteristics: Dict[str, Any]
    ) -> List[ArchitectureComponent]:
        """Create components specific to microservices architecture."""
        components = []
        
        # Service Discovery
        components.append(ArchitectureComponent(
            id="service_discovery",
            name="Service Discovery",
            type=ComponentType.BACKEND,
            description="Service registration and discovery mechanism",
            responsibilities=["Service registration", "Service discovery", "Health checking"],
            estimated_complexity=3,
            estimated_effort_hours=50,
            priority=2
        ))
        
        # Message Queue
        components.append(ArchitectureComponent(
            id="message_queue",
            name="Message Queue",
            type=ComponentType.MESSAGE_QUEUE,
            description="Asynchronous communication between services",
            responsibilities=["Message routing", "Message persistence", "Dead letter handling"],
            estimated_complexity=3,
            estimated_effort_hours=45,
            priority=2
        ))
        
        return components
    
    def _create_data_components(
        self, 
        requirements: Requirements, 
        characteristics: Dict[str, Any]
    ) -> List[ArchitectureComponent]:
        """Create components for data-intensive applications."""
        components = []
        
        # Cache Layer
        components.append(ArchitectureComponent(
            id="cache_layer",
            name="Cache Layer",
            type=ComponentType.CACHE,
            description="High-performance data caching",
            responsibilities=["Data caching", "Cache invalidation", "Performance optimization"],
            estimated_complexity=2,
            estimated_effort_hours=35,
            priority=2
        ))
        
        return components
    
    def _create_frontend_components(
        self, 
        requirements: Requirements, 
        characteristics: Dict[str, Any]
    ) -> List[ArchitectureComponent]:
        """Create frontend components."""
        components = []
        
        # Web Application
        components.append(ArchitectureComponent(
            id="web_app",
            name="Web Application",
            type=ComponentType.FRONTEND,
            description="User-facing web application",
            responsibilities=["User interface", "User experience", "Client-side logic"],
            estimated_complexity=3,
            estimated_effort_hours=120,
            priority=1
        ))
        
        return components
    
    def _analyze_dependencies(
        self, 
        components: List[ArchitectureComponent], 
        requirements: Requirements
    ) -> List[ComponentDependency]:
        """Analyze and create component dependencies."""
        dependencies = []
        
        for component in components:
            if component.type == ComponentType.FRONTEND:
                # Frontend depends on API Gateway
                api_gateway = next((c for c in components if c.type == ComponentType.API_GATEWAY), None)
                if api_gateway:
                    dependencies.append(ComponentDependency(
                        id=f"dep_{component.id}_{api_gateway.id}",
                        source_component_id=component.id,
                        target_component_id=api_gateway.id,
                        dependency_type="api_call",
                        is_critical=True,
                        communication_protocol="HTTP/HTTPS",
                        data_flow_direction="bidirectional"
                    ))
            
            elif component.type == ComponentType.API_GATEWAY:
                # API Gateway depends on backend services
                backend_components = [c for c in components if c.type == ComponentType.BACKEND]
                for backend in backend_components:
                    dependencies.append(ComponentDependency(
                        id=f"dep_{component.id}_{backend.id}",
                        source_component_id=component.id,
                        target_component_id=backend.id,
                        dependency_type="service_call",
                        is_critical=True,
                        communication_protocol="HTTP/gRPC",
                        data_flow_direction="bidirectional"
                    ))
            
            elif component.type == ComponentType.BACKEND:
                # Backend depends on database
                databases = [c for c in components if c.type == ComponentType.DATABASE]
                for db in databases:
                    dependencies.append(ComponentDependency(
                        id=f"dep_{component.id}_{db.id}",
                        source_component_id=component.id,
                        target_component_id=db.id,
                        dependency_type="data_access",
                        is_critical=True,
                        communication_protocol="SQL/NoSQL",
                        data_flow_direction="bidirectional"
                    ))
        
        return dependencies
    
    def _design_data_flows(
        self, 
        components: List[ArchitectureComponent], 
        dependencies: List[ComponentDependency],
        requirements: Requirements
    ) -> List[DataFlow]:
        """Design data flows between components."""
        data_flows = []
        
        for dep in dependencies:
            source_comp = next((c for c in components if c.id == dep.source_component_id), None)
            target_comp = next((c for c in components if c.id == dep.target_component_id), None)
            
            if source_comp and target_comp:
                data_flows.append(DataFlow(
                    id=f"flow_{dep.id}",
                    name=f"{source_comp.name} to {target_comp.name}",
                    description=f"Data flow from {source_comp.name} to {target_comp.name}",
                    source_component_id=dep.source_component_id,
                    target_component_id=dep.target_component_id,
                    data_types=self._infer_data_types(source_comp, target_comp, requirements),
                    volume_estimate=self._estimate_data_volume(source_comp, target_comp),
                    frequency="real-time" if dep.is_critical else "batch",
                    security_requirements=["encryption_in_transit"]
                ))
        
        return data_flows
    
    def _infer_data_types(
        self, 
        source: ArchitectureComponent, 
        target: ArchitectureComponent,
        requirements: Requirements
    ) -> List[str]:
        """Infer data types flowing between components."""
        data_types = []
        
        if source.type == ComponentType.FRONTEND and target.type == ComponentType.API_GATEWAY:
            data_types = ["user_requests", "form_data", "authentication_tokens"]
        elif source.type == ComponentType.API_GATEWAY and target.type == ComponentType.BACKEND:
            data_types = ["api_requests", "business_data", "metadata"]
        elif source.type == ComponentType.BACKEND and target.type == ComponentType.DATABASE:
            data_types = ["entity_data", "query_results", "transaction_data"]
        
        # Add entity-specific data types
        for entity in requirements.entities:
            data_types.append(f"{entity.name.lower()}_data")
        
        return data_types
    
    def _estimate_data_volume(
        self, 
        source: ArchitectureComponent, 
        target: ArchitectureComponent
    ) -> str:
        """Estimate data volume between components."""
        if source.type == ComponentType.FRONTEND:
            return "low"
        elif source.type == ComponentType.API_GATEWAY:
            return "medium"
        elif source.type == ComponentType.BACKEND:
            return "medium"
        else:
            return "low"
    
    def _recommend_technology_stack(
        self, 
        components: List[ArchitectureComponent],
        pattern: ArchitecturePattern,
        characteristics: Dict[str, Any],
        preferences: Optional[Dict[str, Any]] = None
    ) -> TechnologyStack:
        """Recommend technology stack based on components, pattern, and characteristics."""
        technologies = []
        
        # Frontend technology selection
        if any(c.type == ComponentType.FRONTEND for c in components):
            frontend_tech = self._select_frontend_technology(characteristics, preferences)
            technologies.append(frontend_tech)
        
        # Backend technology selection
        if any(c.type == ComponentType.BACKEND for c in components):
            backend_tech = self._select_backend_technology(pattern, characteristics, preferences)
            technologies.append(backend_tech)
        
        # Database technology selection
        if any(c.type == ComponentType.DATABASE for c in components):
            database_tech = self._select_database_technology(characteristics, preferences)
            technologies.append(database_tech)
        
        # Caching technology
        if any(c.type == ComponentType.CACHE for c in components):
            cache_tech = self._select_cache_technology(characteristics, preferences)
            technologies.append(cache_tech)
        
        # Message queue technology for microservices
        if any(c.type == ComponentType.MESSAGE_QUEUE for c in components):
            queue_tech = self._select_message_queue_technology(pattern, characteristics, preferences)
            technologies.append(queue_tech)
        
        # Cloud platform selection
        cloud_tech = self._select_cloud_platform(characteristics, preferences)
        technologies.append(cloud_tech)
        
        # Container platform
        container_tech = self._select_container_platform(pattern, characteristics, preferences)
        technologies.append(container_tech)
        
        # CI/CD platform
        cicd_tech = self._select_cicd_platform(characteristics, preferences)
        technologies.append(cicd_tech)
        
        # Calculate compatibility and metrics
        compatibility_score = self._calculate_technology_compatibility(technologies)
        total_cost = sum(tech.cost_factor for tech in technologies)
        complexity_score = self._calculate_stack_complexity(technologies)
        team_size = self._estimate_team_size(technologies, complexity_score)
        
        return TechnologyStack(
            id=f"stack_{int(datetime.utcnow().timestamp())}",
            name=f"{pattern.value.title()} Technology Stack",
            description=f"Optimized technology stack for {pattern.value} architecture",
            technologies=technologies,
            compatibility_score=compatibility_score,
            total_cost_estimate=total_cost,
            complexity_score=complexity_score,
            recommended_team_size=team_size
        )
    
    def _select_frontend_technology(
        self, 
        characteristics: Dict[str, Any], 
        preferences: Optional[Dict[str, Any]] = None
    ) -> Technology:
        """Select optimal frontend technology."""
        if preferences and "frontend_framework" in preferences:
            framework = preferences["frontend_framework"]
        else:
            # Select based on characteristics
            if characteristics.get("performance_critical", False):
                framework = "Svelte"
            elif characteristics.get("team_size", 5) > 10:
                framework = "Angular"
            else:
                framework = "React"
        
        tech_specs = {
            "React": {
                "pros": ["Large ecosystem", "Component-based", "Strong community", "Flexible"],
                "cons": ["Learning curve", "Frequent updates", "Bundle size"],
                "learning_curve": 3,
                "performance_score": 4,
                "cost_factor": 1.0
            },
            "Vue.js": {
                "pros": ["Easy to learn", "Progressive", "Good documentation", "Lightweight"],
                "cons": ["Smaller ecosystem", "Less job market", "Corporate backing"],
                "learning_curve": 2,
                "performance_score": 4,
                "cost_factor": 0.9
            },
            "Angular": {
                "pros": ["Full framework", "TypeScript", "Enterprise ready", "Strong structure"],
                "cons": ["Steep learning curve", "Complex", "Heavy"],
                "learning_curve": 4,
                "performance_score": 3,
                "cost_factor": 1.3
            },
            "Svelte": {
                "pros": ["No runtime", "Small bundles", "Fast", "Simple"],
                "cons": ["Smaller ecosystem", "Newer", "Less tooling"],
                "learning_curve": 2,
                "performance_score": 5,
                "cost_factor": 0.8
            }
        }
        
        specs = tech_specs.get(framework, tech_specs["React"])
        
        return Technology(
            name=framework,
            category=TechnologyCategory.FRONTEND_FRAMEWORK,
            description=f"Modern {framework} framework for building user interfaces",
            pros=specs["pros"],
            cons=specs["cons"],
            use_cases=["SPAs", "PWAs", "Web applications"],
            learning_curve=specs["learning_curve"],
            community_support=5 if framework == "React" else 4,
            maturity=5 if framework in ["React", "Angular"] else 4,
            performance_score=specs["performance_score"],
            cost_factor=specs["cost_factor"]
        )
    
    def _select_backend_technology(
        self, 
        pattern: ArchitecturePattern,
        characteristics: Dict[str, Any], 
        preferences: Optional[Dict[str, Any]] = None
    ) -> Technology:
        """Select optimal backend technology."""
        if preferences and "backend_framework" in preferences:
            framework = preferences["backend_framework"]
        else:
            # Select based on pattern and characteristics
            if pattern == ArchitecturePattern.MICROSERVICES:
                framework = "FastAPI"
            elif characteristics.get("data_intensive", False):
                framework = "Django"
            elif characteristics.get("performance_critical", False):
                framework = "FastAPI"
            else:
                framework = "FastAPI"
        
        tech_specs = {
            "FastAPI": {
                "pros": ["High performance", "Type safety", "Auto documentation", "Modern"],
                "cons": ["Newer ecosystem", "Learning curve", "Less mature"],
                "learning_curve": 2,
                "performance_score": 5,
                "cost_factor": 1.0
            },
            "Django": {
                "pros": ["Batteries included", "ORM", "Admin interface", "Mature"],
                "cons": ["Monolithic", "Heavy", "Less flexible"],
                "learning_curve": 3,
                "performance_score": 3,
                "cost_factor": 1.1
            },
            "Express.js": {
                "pros": ["Lightweight", "Flexible", "Large ecosystem", "JavaScript"],
                "cons": ["Callback hell", "No structure", "Security concerns"],
                "learning_curve": 2,
                "performance_score": 4,
                "cost_factor": 0.9
            },
            "Spring Boot": {
                "pros": ["Enterprise ready", "Strong ecosystem", "Java", "Mature"],
                "cons": ["Heavy", "Complex", "Verbose"],
                "learning_curve": 4,
                "performance_score": 4,
                "cost_factor": 1.4
            }
        }
        
        specs = tech_specs.get(framework, tech_specs["FastAPI"])
        
        return Technology(
            name=framework,
            category=TechnologyCategory.BACKEND_FRAMEWORK,
            description=f"High-performance {framework} backend framework",
            pros=specs["pros"],
            cons=specs["cons"],
            use_cases=["REST APIs", "Microservices", "Web services"],
            learning_curve=specs["learning_curve"],
            community_support=5 if framework in ["Django", "Express.js"] else 4,
            maturity=5 if framework in ["Django", "Spring Boot"] else 4,
            performance_score=specs["performance_score"],
            cost_factor=specs["cost_factor"]
        )
    
    def _select_database_technology(
        self, 
        characteristics: Dict[str, Any], 
        preferences: Optional[Dict[str, Any]] = None
    ) -> Technology:
        """Select optimal database technology."""
        if preferences and "database" in preferences:
            database = preferences["database"]
        else:
            # Select based on characteristics
            if characteristics.get("scalability_needs") == ScalabilityLevel.ENTERPRISE:
                database = "MongoDB"
            elif characteristics.get("data_intensive", False):
                database = "PostgreSQL"
            else:
                database = "PostgreSQL"
        
        tech_specs = {
            "PostgreSQL": {
                "pros": ["ACID compliance", "Strong consistency", "Rich queries", "Mature"],
                "cons": ["Complex clustering", "Resource intensive", "Learning curve"],
                "learning_curve": 3,
                "performance_score": 4,
                "cost_factor": 1.2
            },
            "MongoDB": {
                "pros": ["Flexible schema", "Horizontal scaling", "JSON documents", "Fast"],
                "cons": ["Eventual consistency", "Memory usage", "Complex queries"],
                "learning_curve": 2,
                "performance_score": 4,
                "cost_factor": 1.1
            },
            "MySQL": {
                "pros": ["Popular", "Fast reads", "Simple", "Good tooling"],
                "cons": ["Limited features", "Replication lag", "Storage engines"],
                "learning_curve": 2,
                "performance_score": 4,
                "cost_factor": 1.0
            }
        }
        
        specs = tech_specs.get(database, tech_specs["PostgreSQL"])
        
        return Technology(
            name=database,
            category=TechnologyCategory.DATABASE,
            description=f"Robust {database} database system",
            pros=specs["pros"],
            cons=specs["cons"],
            use_cases=["Transactional apps", "Data storage", "Analytics"],
            learning_curve=specs["learning_curve"],
            community_support=5,
            maturity=5,
            performance_score=specs["performance_score"],
            cost_factor=specs["cost_factor"]
        )
    
    def _select_cache_technology(
        self, 
        characteristics: Dict[str, Any], 
        preferences: Optional[Dict[str, Any]] = None
    ) -> Technology:
        """Select optimal caching technology."""
        cache_tech = "Redis"
        
        return Technology(
            name=cache_tech,
            category=TechnologyCategory.CACHING,
            description="High-performance in-memory data structure store",
            pros=["Fast", "Versatile", "Clustering", "Persistence"],
            cons=["Memory usage", "Complexity", "Single-threaded"],
            use_cases=["Caching", "Session storage", "Real-time analytics"],
            learning_curve=2,
            community_support=5,
            maturity=5,
            performance_score=5,
            cost_factor=0.8
        )
    
    def _select_message_queue_technology(
        self, 
        pattern: ArchitecturePattern,
        characteristics: Dict[str, Any], 
        preferences: Optional[Dict[str, Any]] = None
    ) -> Technology:
        """Select optimal message queue technology."""
        if characteristics.get("scalability_needs") == ScalabilityLevel.ENTERPRISE:
            queue_tech = "Apache Kafka"
        else:
            queue_tech = "RabbitMQ"
        
        tech_specs = {
            "RabbitMQ": {
                "pros": ["Reliable", "Feature rich", "Easy to use", "Good tooling"],
                "cons": ["Single point of failure", "Memory usage", "Complex clustering"],
                "performance_score": 4,
                "cost_factor": 1.0
            },
            "Apache Kafka": {
                "pros": ["High throughput", "Scalable", "Durable", "Stream processing"],
                "cons": ["Complex", "Resource intensive", "Learning curve"],
                "performance_score": 5,
                "cost_factor": 1.3
            }
        }
        
        specs = tech_specs.get(queue_tech, tech_specs["RabbitMQ"])
        
        return Technology(
            name=queue_tech,
            category=TechnologyCategory.MESSAGING,
            description=f"Reliable {queue_tech} message broker",
            pros=specs["pros"],
            cons=specs["cons"],
            use_cases=["Async messaging", "Event streaming", "Service decoupling"],
            learning_curve=3,
            community_support=5,
            maturity=5,
            performance_score=specs["performance_score"],
            cost_factor=specs["cost_factor"]
        )
    
    def _select_cloud_platform(
        self, 
        characteristics: Dict[str, Any], 
        preferences: Optional[Dict[str, Any]] = None
    ) -> Technology:
        """Select optimal cloud platform."""
        if preferences and "cloud_provider" in preferences:
            platform = preferences["cloud_provider"]
        else:
            platform = "AWS"  # Default to AWS for comprehensive services
        
        tech_specs = {
            "AWS": {
                "pros": ["Comprehensive services", "Global infrastructure", "Strong security", "Mature"],
                "cons": ["Complex pricing", "Vendor lock-in", "Learning curve"],
                "cost_factor": 1.5
            },
            "Azure": {
                "pros": ["Microsoft integration", "Hybrid cloud", "Enterprise focus", "Good pricing"],
                "cons": ["Less mature", "Smaller ecosystem", "Regional availability"],
                "cost_factor": 1.4
            },
            "GCP": {
                "pros": ["AI/ML services", "Competitive pricing", "Innovation", "Kubernetes"],
                "cons": ["Smaller market share", "Less enterprise features", "Support"],
                "cost_factor": 1.3
            }
        }
        
        specs = tech_specs.get(platform, tech_specs["AWS"])
        
        return Technology(
            name=platform,
            category=TechnologyCategory.CLOUD_PROVIDER,
            description=f"{platform} cloud platform services",
            pros=specs["pros"],
            cons=specs["cons"],
            use_cases=["Scalable apps", "Global deployment", "Managed services"],
            learning_curve=4,
            community_support=5,
            maturity=5,
            performance_score=5,
            cost_factor=specs["cost_factor"]
        )
    
    def _select_container_platform(
        self, 
        pattern: ArchitecturePattern,
        characteristics: Dict[str, Any], 
        preferences: Optional[Dict[str, Any]] = None
    ) -> Technology:
        """Select optimal container platform."""
        if pattern == ArchitecturePattern.MICROSERVICES:
            container_tech = "Kubernetes"
        else:
            container_tech = "Docker"
        
        tech_specs = {
            "Docker": {
                "pros": ["Simple", "Lightweight", "Fast", "Popular"],
                "cons": ["Single host", "No orchestration", "Limited scaling"],
                "cost_factor": 0.5
            },
            "Kubernetes": {
                "pros": ["Orchestration", "Scaling", "Self-healing", "Industry standard"],
                "cons": ["Complex", "Resource overhead", "Learning curve"],
                "cost_factor": 1.2
            }
        }
        
        specs = tech_specs.get(container_tech, tech_specs["Docker"])
        
        return Technology(
            name=container_tech,
            category=TechnologyCategory.CONTAINER_PLATFORM,
            description=f"{container_tech} containerization platform",
            pros=specs["pros"],
            cons=specs["cons"],
            use_cases=["Containerization", "Deployment", "Scaling"],
            learning_curve=3 if container_tech == "Kubernetes" else 2,
            community_support=5,
            maturity=5,
            performance_score=4,
            cost_factor=specs["cost_factor"]
        )
    
    def _select_cicd_platform(
        self, 
        characteristics: Dict[str, Any], 
        preferences: Optional[Dict[str, Any]] = None
    ) -> Technology:
        """Select optimal CI/CD platform."""
        cicd_tech = "GitHub Actions"
        
        return Technology(
            name=cicd_tech,
            category=TechnologyCategory.CI_CD,
            description="Automated CI/CD pipeline platform",
            pros=["Integrated", "Free tier", "Easy setup", "Good ecosystem"],
            cons=["GitHub dependency", "Limited customization", "Pricing"],
            use_cases=["Automated testing", "Deployment", "Code quality"],
            learning_curve=2,
            community_support=5,
            maturity=4,
            performance_score=4,
            cost_factor=0.7
        )
    
    def _calculate_technology_compatibility(self, technologies: List[Technology]) -> float:
        """Calculate compatibility score between technologies."""
        # Simple compatibility matrix - in real implementation, this would be more sophisticated
        compatibility_matrix = {
            ("React", "FastAPI"): 0.9,
            ("React", "Django"): 0.8,
            ("Vue.js", "FastAPI"): 0.9,
            ("Angular", "Spring Boot"): 0.9,
            ("PostgreSQL", "FastAPI"): 0.9,
            ("MongoDB", "Express.js"): 0.9,
            ("Redis", "FastAPI"): 0.9,
            ("Kubernetes", "AWS"): 0.9,
            ("Docker", "GitHub Actions"): 0.9
        }
        
        total_score = 0.0
        comparisons = 0
        
        for i, tech1 in enumerate(technologies):
            for tech2 in technologies[i+1:]:
                key = (tech1.name, tech2.name)
                reverse_key = (tech2.name, tech1.name)
                
                score = compatibility_matrix.get(key, compatibility_matrix.get(reverse_key, 0.7))
                total_score += score
                comparisons += 1
        
        return total_score / comparisons if comparisons > 0 else 0.8
    
    def _calculate_stack_complexity(self, technologies: List[Technology]) -> int:
        """Calculate overall stack complexity."""
        if not technologies:
            return 1
        
        avg_complexity = sum(tech.learning_curve for tech in technologies) / len(technologies)
        return min(5, max(1, int(avg_complexity)))
    
    def _estimate_team_size(self, technologies: List[Technology], complexity_score: int) -> int:
        """Estimate recommended team size based on technologies and complexity."""
        base_size = 3
        
        # Add team members based on technology diversity
        tech_categories = {tech.category for tech in technologies}
        base_size += len(tech_categories) - 2  # Subtract 2 as baseline includes frontend + backend
        
        # Adjust for complexity
        base_size += complexity_score - 2
        
        return max(3, min(15, base_size))
    
    def _design_scalability_strategy(self, characteristics: Dict[str, Any]) -> Dict[str, Any]:
        """Design scalability strategy."""
        strategy = {
            "horizontal_scaling": True,
            "vertical_scaling": False,
            "auto_scaling": False,
            "load_balancing": False,
            "caching_strategy": "basic",
            "database_scaling": "read_replicas"
        }
        
        if characteristics["scalability_needs"] in [ScalabilityLevel.HIGH, ScalabilityLevel.ENTERPRISE]:
            strategy.update({
                "auto_scaling": True,
                "load_balancing": True,
                "caching_strategy": "distributed",
                "database_scaling": "sharding"
            })
        
        return strategy
    
    def _design_security_strategy(self, characteristics: Dict[str, Any]) -> Dict[str, Any]:
        """Design security strategy."""
        strategy = {
            "authentication": "jwt",
            "authorization": "rbac",
            "encryption_at_rest": True,
            "encryption_in_transit": True,
            "api_security": "oauth2",
            "monitoring": "basic"
        }
        
        if characteristics["security_needs"] in [SecurityLevel.HIGH, SecurityLevel.ENTERPRISE]:
            strategy.update({
                "multi_factor_auth": True,
                "api_security": "oauth2_pkce",
                "monitoring": "advanced",
                "compliance": ["SOC2", "GDPR"],
                "vulnerability_scanning": True
            })
        
        return strategy
    
    def _design_deployment_strategy(self, characteristics: Dict[str, Any]) -> Dict[str, Any]:
        """Design deployment strategy."""
        strategy = {
            "deployment_model": "blue_green",
            "ci_cd": True,
            "infrastructure_as_code": True,
            "monitoring": "basic",
            "backup_strategy": "daily"
        }
        
        if characteristics["scalability_needs"] == ScalabilityLevel.ENTERPRISE:
            strategy.update({
                "deployment_model": "canary",
                "monitoring": "comprehensive",
                "backup_strategy": "continuous",
                "disaster_recovery": True
            })
        
        return strategy
    
    def _calculate_total_cost(
        self, 
        components: List[ArchitectureComponent], 
        tech_stack: TechnologyStack
    ) -> float:
        """Calculate estimated total cost."""
        base_cost = len(components) * 1000
        tech_cost = tech_stack.total_cost_estimate * 500
        return base_cost + tech_cost
    
    def _estimate_development_time(self, components: List[ArchitectureComponent]) -> int:
        """Estimate total development time in hours."""
        return sum(comp.estimated_effort_hours for comp in components)
    
    def _calculate_complexity_score(
        self, 
        components: List[ArchitectureComponent], 
        dependencies: List[ComponentDependency]
    ) -> int:
        """Calculate overall architecture complexity score."""
        component_complexity = sum(comp.estimated_complexity for comp in components) / len(components)
        dependency_complexity = len(dependencies) / len(components) if components else 0
        total_complexity = (component_complexity + dependency_complexity) / 2
        return min(5, max(1, int(total_complexity)))
    
    def _calculate_maintainability_score(
        self, 
        components: List[ArchitectureComponent], 
        pattern: ArchitecturePattern
    ) -> float:
        """Calculate maintainability score."""
        base_score = 0.7
        
        if pattern == ArchitecturePattern.MONOLITHIC:
            base_score += 0.1
        elif pattern == ArchitecturePattern.MICROSERVICES:
            base_score -= 0.1
        elif pattern == ArchitecturePattern.LAYERED:
            base_score += 0.2
        
        if len(components) > 10:
            base_score -= 0.1
        elif len(components) < 5:
            base_score += 0.1
        
        return max(0.0, min(1.0, base_score))
    
    def _analyze_requirements_coverage(
        self, 
        requirements: Requirements, 
        components: List[ArchitectureComponent]
    ) -> Dict[str, bool]:
        """Analyze how well components cover requirements."""
        coverage = {}
        
        for req in requirements.parsed_requirements:
            req_id = f"req_{req.id}" if hasattr(req, 'id') else f"req_{hash(req.structured_text)}"
            
            # Enhanced coverage analysis
            covered = False
            req_text = req.structured_text.lower()
            
            # Check coverage based on requirement type and intent
            if req.requirement_type == RequirementType.PERFORMANCE:
                # Performance requirements covered by scalability features
                if any(comp.scalability_requirements in [ScalabilityLevel.HIGH, ScalabilityLevel.ENTERPRISE] 
                       for comp in components):
                    covered = True
                elif any(comp.type == ComponentType.CACHE for comp in components):
                    covered = True
            
            elif req.requirement_type == RequirementType.SECURITY:
                # Security requirements covered by auth components or security features
                if any(comp.type == ComponentType.AUTHENTICATION for comp in components):
                    covered = True
                elif any(comp.security_requirements in [SecurityLevel.HIGH, SecurityLevel.ENTERPRISE] 
                         for comp in components):
                    covered = True
            
            elif req.requirement_type == RequirementType.FUNCTIONAL:
                # Functional requirements covered by relevant components
                if req.intent == Intent.BUILD_UI:
                    covered = any(comp.type == ComponentType.FRONTEND for comp in components)
                elif req.intent == Intent.CREATE_API:
                    covered = any(comp.type == ComponentType.API_GATEWAY for comp in components)
                elif req.intent == Intent.DESIGN_DATABASE:
                    covered = any(comp.type == ComponentType.DATABASE for comp in components)
                else:
                    # General functional requirements - check for keyword overlap
                    for component in components:
                        comp_text = f"{component.name} {component.description} {' '.join(component.responsibilities)}".lower()
                        
                        # Check for keyword overlap with more lenient matching
                        req_words = set(req_text.split())
                        comp_words = set(comp_text.split())
                        
                        # Remove common words that don't add meaning
                        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
                        req_words = req_words - common_words
                        comp_words = comp_words - common_words
                        
                        if len(req_words.intersection(comp_words)) >= 1:
                            covered = True
                            break
            
            # Fallback: check for any keyword overlap
            if not covered:
                for component in components:
                    comp_text = f"{component.name} {component.description} {' '.join(component.responsibilities)}".lower()
                    
                    # Check for any meaningful keyword overlap
                    req_words = set(req_text.split())
                    comp_words = set(comp_text.split())
                    
                    # Remove common words
                    common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'that', 'can', 'will', 'should'}
                    req_words = req_words - common_words
                    comp_words = comp_words - common_words
                    
                    if len(req_words.intersection(comp_words)) >= 1:
                        covered = True
                        break
            
            coverage[req_id] = covered
        
        return coverage
    
    async def validate_architecture(self, architecture: Architecture) -> ArchitectureValidationResult:
        """Validate architecture against best practices and rules."""
        try:
            logger.info(f"Validating architecture: {architecture.name}")
            
            violations = []
            warnings = []
            recommendations = []
            
            # Get validation rules
            validation_rules = self._get_validation_rules()
            
            # Run all validation rules
            for rule in validation_rules:
                result = self._apply_validation_rule(rule, architecture)
                if result:
                    if result["severity"] == "error":
                        violations.append(result)
                    elif result["severity"] == "warning":
                        warnings.append(result["message"])
            
            # Generate recommendations based on violations and warnings
            recommendations = self._generate_recommendations(architecture, violations, warnings)
            
            # Calculate scores
            overall_score = self._calculate_validation_score(architecture, violations, warnings)
            best_practices_score = self._calculate_best_practices_score(architecture)
            
            result = ArchitectureValidationResult(
                is_valid=overall_score >= 0.7,
                score=overall_score,
                violations=[{"rule": v["rule"], "message": v["message"], "severity": v["severity"]} for v in violations],
                warnings=warnings,
                recommendations=recommendations,
                best_practices_score=best_practices_score
            )
            
            logger.info(f"Architecture validation completed with score: {overall_score:.2f}")
            return result
            
        except Exception as e:
            logger.error(f"Architecture validation failed: {e}")
            raise
    
    def _get_validation_rules(self) -> List[Dict[str, Any]]:
        """Get comprehensive validation rules."""
        from ..models.architecture_models import ArchitectureValidationRule
        
        return [
            {
                "id": "missing_database",
                "name": "Database Component Required",
                "description": "Architecture must include a database component",
                "category": "essential_components",
                "severity": "error",
                "condition": lambda arch: ComponentType.DATABASE not in {comp.type for comp in arch.components},
                "recommendation": "Add a database component to store application data"
            },
            {
                "id": "missing_api_gateway",
                "name": "API Gateway Recommended",
                "description": "Complex architectures should include an API Gateway",
                "category": "scalability",
                "severity": "warning",
                "condition": lambda arch: (ComponentType.API_GATEWAY not in {comp.type for comp in arch.components} 
                                         and len(arch.components) > 3),
                "recommendation": "Consider adding an API Gateway for better request management and security"
            },
            {
                "id": "missing_authentication",
                "name": "Authentication Component Required",
                "description": "User-facing applications should include authentication",
                "category": "security",
                "severity": "warning",
                "condition": lambda arch: (any(comp.type == ComponentType.FRONTEND for comp in arch.components) 
                                         and ComponentType.AUTHENTICATION not in {comp.type for comp in arch.components}),
                "recommendation": "Add authentication component for user security"
            },
            {
                "id": "high_complexity",
                "name": "High Complexity Warning",
                "description": "Architecture complexity is very high",
                "category": "maintainability",
                "severity": "warning",
                "condition": lambda arch: arch.complexity_score > 4,
                "recommendation": "Consider simplifying the architecture to improve maintainability"
            },
            {
                "id": "no_caching",
                "name": "Caching Recommended",
                "description": "Performance-critical applications should include caching",
                "category": "performance",
                "severity": "warning",
                "condition": lambda arch: (arch.estimated_development_time > 200 
                                         and ComponentType.CACHE not in {comp.type for comp in arch.components}),
                "recommendation": "Consider adding a caching layer to improve performance"
            },
            {
                "id": "single_point_failure",
                "name": "Single Point of Failure",
                "description": "Critical components should have redundancy",
                "category": "reliability",
                "severity": "warning",
                "condition": lambda arch: len([comp for comp in arch.components if comp.type == ComponentType.DATABASE]) == 1,
                "recommendation": "Consider database replication or clustering for high availability"
            },
            {
                "id": "missing_monitoring",
                "name": "Monitoring Component Missing",
                "description": "Production architectures should include monitoring",
                "category": "observability",
                "severity": "warning",
                "condition": lambda arch: ComponentType.MONITORING not in {comp.type for comp in arch.components},
                "recommendation": "Add monitoring and logging components for production readiness"
            },
            {
                "id": "insecure_communication",
                "name": "Insecure Communication",
                "description": "All communication should be encrypted",
                "category": "security",
                "severity": "error",
                "condition": lambda arch: any(dep.communication_protocol in ["HTTP", "TCP"] 
                                            for dep in arch.dependencies),
                "recommendation": "Use encrypted protocols (HTTPS, TLS) for all communications"
            },
            {
                "id": "circular_dependencies",
                "name": "Circular Dependencies",
                "description": "Architecture should not have circular dependencies",
                "category": "design",
                "severity": "error",
                "condition": lambda arch: self._has_circular_dependencies(arch.dependencies),
                "recommendation": "Resolve circular dependencies by introducing interfaces or breaking cycles"
            },
            {
                "id": "excessive_dependencies",
                "name": "Excessive Dependencies",
                "description": "Components should not have too many dependencies",
                "category": "maintainability",
                "severity": "warning",
                "condition": lambda arch: any(len([dep for dep in arch.dependencies 
                                                 if dep.source_component_id == comp.id]) > 5 
                                            for comp in arch.components),
                "recommendation": "Reduce component dependencies to improve maintainability"
            }
        ]
    
    def _apply_validation_rule(self, rule: Dict[str, Any], architecture: Architecture) -> Optional[Dict[str, Any]]:
        """Apply a validation rule to an architecture."""
        try:
            if rule["condition"](architecture):
                return {
                    "rule": rule["id"],
                    "message": rule["description"],
                    "severity": rule["severity"],
                    "category": rule["category"],
                    "recommendation": rule["recommendation"]
                }
        except Exception as e:
            logger.warning(f"Failed to apply validation rule {rule['id']}: {e}")
        
        return None
    
    def _generate_recommendations(
        self, 
        architecture: Architecture, 
        violations: List[Dict[str, Any]], 
        warnings: List[str]
    ) -> List[str]:
        """Generate improvement recommendations."""
        recommendations = []
        
        # Add recommendations from violations
        for violation in violations:
            if "recommendation" in violation:
                recommendations.append(violation["recommendation"])
        
        # Add recommendations based on warnings
        for warning in warnings:
            if "single point of failure" in warning.lower():
                recommendations.append("Implement database clustering or replication for high availability")
            elif "monitoring" in warning.lower():
                recommendations.append("Add comprehensive monitoring and logging systems")
            elif "api gateway" in warning.lower():
                recommendations.append("Consider adding an API Gateway for better request management")
            elif "authentication" in warning.lower():
                recommendations.append("Implement proper authentication and authorization mechanisms")
            elif "caching" in warning.lower():
                recommendations.append("Add caching layer to improve performance")
            elif "complexity" in warning.lower():
                recommendations.append("Simplify architecture to reduce complexity and improve maintainability")
        
        # Pattern-specific recommendations
        if architecture.pattern == ArchitecturePattern.MONOLITHIC and len(architecture.components) > 8:
            recommendations.append("Consider migrating to microservices architecture for better scalability")
        
        if architecture.pattern == ArchitecturePattern.MICROSERVICES and len(architecture.components) < 4:
            recommendations.append("Consider monolithic architecture for simpler deployment and maintenance")
        
        # Technology stack recommendations
        if architecture.technology_stack.complexity_score > 4:
            recommendations.append("Consider using simpler technologies to reduce learning curve")
        
        if architecture.technology_stack.compatibility_score < 0.7:
            recommendations.append("Review technology choices for better compatibility")
        
        # Performance recommendations
        if architecture.estimated_development_time > 500:
            recommendations.append("Consider breaking down the project into smaller phases")
        
        # Security recommendations
        security_components = [comp for comp in architecture.components 
                             if comp.type in [ComponentType.AUTHENTICATION, ComponentType.AUTHORIZATION]]
        if not security_components and any(comp.type == ComponentType.FRONTEND for comp in architecture.components):
            recommendations.append("Implement proper authentication and authorization mechanisms")
        
        return list(set(recommendations))  # Remove duplicates
    
    def _calculate_validation_score(
        self, 
        architecture: Architecture, 
        violations: List[Dict[str, Any]], 
        warnings: List[str]
    ) -> float:
        """Calculate overall validation score."""
        base_score = 1.0
        
        # Deduct for violations
        error_violations = [v for v in violations if v["severity"] == "error"]
        base_score -= len(error_violations) * 0.2
        
        # Deduct for warnings
        base_score -= len(warnings) * 0.05
        
        # Adjust for architecture characteristics
        if architecture.complexity_score > 4:
            base_score -= 0.1
        
        if architecture.maintainability_score < 0.5:
            base_score -= 0.1
        
        # Bonus for good practices
        if architecture.technology_stack.compatibility_score > 0.8:
            base_score += 0.05
        
        if len(architecture.components) >= 3 and len(architecture.components) <= 8:
            base_score += 0.05  # Good component count
        
        return max(0.0, min(1.0, base_score))
    
    def _calculate_best_practices_score(self, architecture: Architecture) -> float:
        """Calculate best practices adherence score."""
        score = 0.0
        total_checks = 0
        
        # Check for essential components
        component_types = {comp.type for comp in architecture.components}
        essential_components = [ComponentType.DATABASE, ComponentType.API_GATEWAY]
        
        for comp_type in essential_components:
            total_checks += 1
            if comp_type in component_types:
                score += 1
        
        # Check for security practices
        total_checks += 1
        if any(comp.type == ComponentType.AUTHENTICATION for comp in architecture.components):
            score += 1
        
        # Check for scalability practices
        total_checks += 1
        if architecture.scalability_strategy.get("horizontal_scaling", False):
            score += 1
        
        # Check for monitoring
        total_checks += 1
        if ComponentType.MONITORING in component_types:
            score += 1
        
        # Check for proper dependency management
        total_checks += 1
        if not self._has_circular_dependencies(architecture.dependencies):
            score += 1
        
        # Check technology stack compatibility
        total_checks += 1
        if architecture.technology_stack.compatibility_score > 0.7:
            score += 1
        
        return score / total_checks if total_checks > 0 else 0.0
    
    async def optimize_dependencies(
        self, 
        components: List[ArchitectureComponent], 
        dependencies: List[ComponentDependency]
    ) -> List[ComponentDependency]:
        """Optimize component dependencies to reduce complexity and improve performance."""
        try:
            logger.info("Optimizing component dependencies")
            
            # Remove redundant dependencies
            optimized_deps = self._remove_redundant_dependencies(dependencies)
            
            # Check for circular dependencies and resolve them
            if self._has_circular_dependencies(optimized_deps):
                optimized_deps = self._resolve_circular_dependencies(optimized_deps)
            
            # Optimize dependency types and protocols
            optimized_deps = self._optimize_dependency_protocols(optimized_deps, components)
            
            # Mark critical dependencies
            optimized_deps = self._mark_critical_dependencies(optimized_deps, components)
            
            logger.info(f"Dependency optimization completed. Reduced from {len(dependencies)} to {len(optimized_deps)} dependencies")
            return optimized_deps
            
        except Exception as e:
            logger.error(f"Dependency optimization failed: {e}")
            raise
    
    def _remove_redundant_dependencies(self, dependencies: List[ComponentDependency]) -> List[ComponentDependency]:
        """Remove redundant dependencies."""
        seen = set()
        unique_deps = []
        
        for dep in dependencies:
            # Create a key based on source, target, and type
            key = (dep.source_component_id, dep.target_component_id, dep.dependency_type)
            if key not in seen:
                seen.add(key)
                unique_deps.append(dep)
        
        return unique_deps
    
    def _has_circular_dependencies(self, dependencies: List[ComponentDependency]) -> bool:
        """Check if there are circular dependencies."""
        # Build adjacency list
        graph = {}
        for dep in dependencies:
            if dep.source_component_id not in graph:
                graph[dep.source_component_id] = []
            graph[dep.source_component_id].append(dep.target_component_id)
        
        # DFS to detect cycles
        visited = set()
        rec_stack = set()
        
        def has_cycle(node):
            if node in rec_stack:
                return True
            if node in visited:
                return False
            
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in graph.get(node, []):
                if has_cycle(neighbor):
                    return True
            
            rec_stack.remove(node)
            return False
        
        for node in graph:
            if node not in visited:
                if has_cycle(node):
                    return True
        
        return False
    
    def _resolve_circular_dependencies(self, dependencies: List[ComponentDependency]) -> List[ComponentDependency]:
        """Resolve circular dependencies by introducing intermediary components or changing dependency types."""
        # For now, implement a simple strategy: convert bidirectional dependencies to unidirectional
        resolved_deps = []
        
        for dep in dependencies:
            # Check if there's a reverse dependency
            reverse_dep = next((d for d in dependencies 
                              if d.source_component_id == dep.target_component_id 
                              and d.target_component_id == dep.source_component_id), None)
            
            if reverse_dep and dep.data_flow_direction == "bidirectional":
                # Keep only one direction, prefer the one with higher priority
                if dep.is_critical or not reverse_dep.is_critical:
                    resolved_deps.append(dep)
            else:
                resolved_deps.append(dep)
        
        return resolved_deps
    
    def _optimize_dependency_protocols(
        self, 
        dependencies: List[ComponentDependency], 
        components: List[ArchitectureComponent]
    ) -> List[ComponentDependency]:
        """Optimize communication protocols for dependencies."""
        component_map = {comp.id: comp for comp in components}
        
        for dep in dependencies:
            source_comp = component_map.get(dep.source_component_id)
            target_comp = component_map.get(dep.target_component_id)
            
            if source_comp and target_comp:
                # Optimize protocol based on component types
                if (source_comp.type == ComponentType.FRONTEND and 
                    target_comp.type == ComponentType.API_GATEWAY):
                    dep.communication_protocol = "HTTPS"
                    dep.latency_requirement = 200  # 200ms for user-facing requests
                
                elif (source_comp.type == ComponentType.BACKEND and 
                      target_comp.type == ComponentType.DATABASE):
                    dep.communication_protocol = "TCP/SQL"
                    dep.latency_requirement = 50   # 50ms for database queries
                
                elif (source_comp.type == ComponentType.BACKEND and 
                      target_comp.type == ComponentType.BACKEND):
                    dep.communication_protocol = "gRPC"
                    dep.latency_requirement = 100  # 100ms for service-to-service
        
        return dependencies
    
    def _mark_critical_dependencies(
        self, 
        dependencies: List[ComponentDependency], 
        components: List[ArchitectureComponent]
    ) -> List[ComponentDependency]:
        """Mark critical dependencies based on component importance."""
        component_map = {comp.id: comp for comp in components}
        
        for dep in dependencies:
            source_comp = component_map.get(dep.source_component_id)
            target_comp = component_map.get(dep.target_component_id)
            
            if source_comp and target_comp:
                # Mark as critical if involves high-priority components
                if (source_comp.priority == 1 or target_comp.priority == 1 or
                    target_comp.type in [ComponentType.DATABASE, ComponentType.AUTHENTICATION]):
                    dep.is_critical = True
        
        return dependencies
    
    async def compare_architectures(self, architectures: List[Architecture]) -> 'ArchitectureComparison':
        """Compare multiple architectures and provide recommendations."""
        try:
            logger.info(f"Comparing {len(architectures)} architectures")
            
            from ..models.architecture_models import ArchitectureComparison
            
            # Define comparison criteria
            criteria = [
                "complexity",
                "maintainability", 
                "scalability",
                "security",
                "cost",
                "development_time",
                "performance"
            ]
            
            # Calculate scores for each architecture
            scores = {}
            for arch in architectures:
                arch_scores = {}
                
                # Complexity (lower is better, so invert)
                arch_scores["complexity"] = 1.0 - (arch.complexity_score - 1) / 4
                
                # Maintainability (direct score)
                arch_scores["maintainability"] = arch.maintainability_score
                
                # Scalability (based on scalability strategy)
                scalability_features = sum([
                    arch.scalability_strategy.get("horizontal_scaling", False),
                    arch.scalability_strategy.get("auto_scaling", False),
                    arch.scalability_strategy.get("load_balancing", False),
                    arch.scalability_strategy.get("caching_strategy") == "distributed"
                ])
                arch_scores["scalability"] = scalability_features / 4
                
                # Security (based on security strategy)
                security_features = sum([
                    arch.security_strategy.get("encryption_at_rest", False),
                    arch.security_strategy.get("encryption_in_transit", False),
                    arch.security_strategy.get("multi_factor_auth", False),
                    arch.security_strategy.get("monitoring") == "advanced"
                ])
                arch_scores["security"] = security_features / 4
                
                # Cost (lower is better, normalize and invert)
                max_cost = max(a.estimated_total_cost for a in architectures)
                arch_scores["cost"] = 1.0 - (arch.estimated_total_cost / max_cost) if max_cost > 0 else 1.0
                
                # Development time (lower is better, normalize and invert)
                max_time = max(a.estimated_development_time for a in architectures)
                arch_scores["development_time"] = 1.0 - (arch.estimated_development_time / max_time) if max_time > 0 else 1.0
                
                # Performance (based on validation score)
                arch_scores["performance"] = arch.validation_result.score if arch.validation_result else 0.7
                
                scores[arch.id] = arch_scores
            
            # Calculate overall scores and determine recommendation
            overall_scores = {}
            for arch_id, arch_scores in scores.items():
                overall_scores[arch_id] = sum(arch_scores.values()) / len(arch_scores)
            
            # Find best architecture
            best_arch_id = max(overall_scores, key=overall_scores.get)
            best_arch = next(arch for arch in architectures if arch.id == best_arch_id)
            
            # Generate trade-offs analysis
            trade_offs = {}
            for arch in architectures:
                arch_scores = scores[arch.id]
                strengths = [criterion for criterion, score in arch_scores.items() if score > 0.7]
                weaknesses = [criterion for criterion, score in arch_scores.items() if score < 0.5]
                
                trade_offs[arch.id] = f"Strengths: {', '.join(strengths)}. Weaknesses: {', '.join(weaknesses)}"
            
            comparison = ArchitectureComparison(
                id=f"comparison_{int(datetime.utcnow().timestamp())}",
                name=f"Architecture Comparison - {len(architectures)} Options",
                architectures=architectures,
                criteria=criteria,
                scores=scores,
                recommendation=best_arch.name,
                rationale=f"Selected based on highest overall score ({overall_scores[best_arch_id]:.2f})",
                trade_offs=trade_offs
            )
            
            logger.info(f"Architecture comparison completed. Recommended: {best_arch.name}")
            return comparison
            
        except Exception as e:
            logger.error(f"Architecture comparison failed: {e}")
            raise