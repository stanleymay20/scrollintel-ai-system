# ScrollCTOAgent implementation
from scrollintel.core.interfaces import BaseAgent, AgentType, AgentRequest, AgentResponse, AgentCapability, ResponseStatus
import asyncio
import json
import os
import openai
from datetime import datetime
from typing import List, Dict, Any, Optional
from uuid import uuid4
from dataclasses import dataclass
from enum import Enum

class TechnologyStack(Enum):
    PYTHON_FASTAPI = "python_fastapi"
    NODE_EXPRESS = "node_express"
    JAVA_SPRING = "java_spring"
    DOTNET_CORE = "dotnet_core"
    GOLANG_GIN = "golang_gin"
    RUST_ACTIX = "rust_actix"

class DatabaseType(Enum):
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    MONGODB = "mongodb"
    REDIS = "redis"
    ELASTICSEARCH = "elasticsearch"
    CASSANDRA = "cassandra"

class CloudProvider(Enum):
    AWS = "aws"
    AZURE = "azure"
    GCP = "gcp"
    VERCEL = "vercel"
    RENDER = "render"
    HEROKU = "heroku"

@dataclass
class ArchitectureTemplate:
    name: str
    description: str
    tech_stack: TechnologyStack
    database: DatabaseType
    cloud_provider: CloudProvider
    estimated_cost_monthly: float
    scalability_rating: int  # 1-10
    complexity_rating: int   # 1-10
    use_cases: List[str]
    pros: List[str]
    cons: List[str]

@dataclass
class TechnologyComparison:
    technologies: List[str]
    criteria: Dict[str, Dict[str, float]]  # tech_name -> criterion -> score
    recommendation: str
    reasoning: str

@dataclass
class ScalingStrategy:
    current_load: str
    projected_load: str
    recommended_approach: str
    infrastructure_changes: List[str]
    estimated_cost_impact: float
    timeline: str
    risks: List[str]

class ScrollCTOAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            agent_id="scroll-cto-agent",
            name="ScrollCTO Agent", 
            agent_type=AgentType.CTO
        )
        self.capabilities = [
            AgentCapability(
                name="architecture_design",
                description="Design system architecture and recommend technology stacks",
                input_types=["requirements", "business_context"],
                output_types=["architecture", "recommendations"]
            ),
            AgentCapability(
                name="technology_comparison",
                description="Compare technologies and provide recommendations with cost analysis",
                input_types=["technology_options", "requirements"],
                output_types=["comparison", "recommendation"]
            ),
            AgentCapability(
                name="scaling_strategy",
                description="Generate scaling strategies based on current and projected requirements",
                input_types=["current_metrics", "growth_projections"],
                output_types=["scaling_plan", "cost_analysis"]
            ),
            AgentCapability(
                name="technical_decision",
                description="Make technical decisions with reasoning and trade-off analysis",
                input_types=["problem_statement", "constraints"],
                output_types=["decision", "rationale"]
            )
        ]
        
        # Initialize OpenAI client for GPT-4 integration
        self.openai_client = openai.AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        self._initialize_templates()
    
    def _initialize_templates(self):
        """Initialize architecture templates for different use cases"""
        self.architecture_templates = [
            ArchitectureTemplate(
                name="Startup MVP",
                description="Minimal viable product for rapid development and deployment",
                tech_stack=TechnologyStack.PYTHON_FASTAPI,
                database=DatabaseType.POSTGRESQL,
                cloud_provider=CloudProvider.RENDER,
                estimated_cost_monthly=50.0,
                scalability_rating=6,
                complexity_rating=3,
                use_cases=["MVP", "prototype", "small team"],
                pros=["Fast development", "Low cost", "Simple deployment"],
                cons=["Limited scalability", "Single point of failure"]
            ),
            ArchitectureTemplate(
                name="Enterprise Scale",
                description="High-performance, scalable architecture for enterprise applications",
                tech_stack=TechnologyStack.JAVA_SPRING,
                database=DatabaseType.POSTGRESQL,
                cloud_provider=CloudProvider.AWS,
                estimated_cost_monthly=2000.0,
                scalability_rating=9,
                complexity_rating=8,
                use_cases=["Enterprise", "high traffic", "complex business logic"],
                pros=["High scalability", "Robust ecosystem", "Enterprise support"],
                cons=["High complexity", "Expensive", "Slower development"]
            ),
            ArchitectureTemplate(
                name="AI/ML Platform",
                description="Optimized for AI/ML workloads with GPU support and data processing",
                tech_stack=TechnologyStack.PYTHON_FASTAPI,
                database=DatabaseType.POSTGRESQL,
                cloud_provider=CloudProvider.AWS,
                estimated_cost_monthly=1500.0,
                scalability_rating=8,
                complexity_rating=7,
                use_cases=["AI/ML", "data processing", "analytics"],
                pros=["ML ecosystem", "GPU support", "Data tools"],
                cons=["Resource intensive", "Complex ML ops"]
            )
        ]
    
    async def process_request(self, request: AgentRequest) -> AgentResponse:
        start_time = asyncio.get_event_loop().time()
        try:
            # Parse the request to determine the type of CTO assistance needed
            prompt = request.prompt.lower()
            context = request.context or {}
            
            if "architecture" in prompt or "design" in prompt:
                content = await self._generate_architecture_recommendation(request.prompt, context)
            elif "compare" in prompt or "technology" in prompt:
                content = await self._generate_technology_comparison(request.prompt, context)
            elif "scaling" in prompt or "scale" in prompt:
                content = await self._generate_scaling_strategy(request.prompt, context)
            else:
                content = await self._generate_technical_decision(request.prompt, context)
            
            execution_time = asyncio.get_event_loop().time() - start_time
            return AgentResponse(
                id=f"cto-{uuid4()}",
                request_id=request.id,
                content=content,
                artifacts=[],
                execution_time=execution_time,
                status=ResponseStatus.SUCCESS
            )
        except Exception as e:
            execution_time = asyncio.get_event_loop().time() - start_time
            return AgentResponse(
                id=f"cto-{uuid4()}",
                request_id=request.id,
                content=f"Error processing CTO request: {str(e)}",
                artifacts=[],
                execution_time=execution_time,
                status=ResponseStatus.ERROR,
                error_message=str(e)
            )
    
    async def _generate_architecture_recommendation(self, prompt: str, context: Dict[str, Any]) -> str:
        """Generate architecture recommendations based on requirements using GPT-4"""
        business_type = context.get("business_type", "startup")
        expected_users = context.get("expected_users", 1000)
        budget = context.get("budget", 500)
        
        # Select appropriate template based on context
        if budget < 200:
            template = self.architecture_templates[0]  # Startup MVP
        elif expected_users > 100000:
            template = self.architecture_templates[1]  # Enterprise Scale
        else:
            template = self.architecture_templates[2]  # AI/ML Platform
        
        # Use GPT-4 to enhance the recommendation with specific insights
        gpt4_prompt = f"""
        As a senior CTO, provide detailed architecture recommendations for the following requirements:
        
        User Request: {prompt}
        Business Context:
        - Business Type: {business_type}
        - Expected Users: {expected_users}
        - Budget: ${budget}/month
        
        Base Template Selected: {template.name}
        - Tech Stack: {template.tech_stack.value}
        - Database: {template.database.value}
        - Cloud Provider: {template.cloud_provider.value}
        
        Please provide:
        1. Detailed justification for this architecture choice
        2. Specific implementation considerations
        3. Potential risks and mitigation strategies
        4. Alternative approaches and why they weren't chosen
        5. Step-by-step implementation roadmap
        
        Focus on practical, actionable advice that a development team can implement.
        """
        
        try:
            gpt4_response = await self._call_gpt4(gpt4_prompt)
            enhanced_recommendation = f"""
# Architecture Recommendation (Enhanced by AI)

## Recommended Stack: {template.name}
{template.description}

### Technology Stack
- **Backend**: {template.tech_stack.value}
- **Database**: {template.database.value}
- **Cloud Provider**: {template.cloud_provider.value}

### Cost Analysis
- **Estimated Monthly Cost**: ${template.estimated_cost_monthly}
- **Scalability Rating**: {template.scalability_rating}/10
- **Complexity Rating**: {template.complexity_rating}/10

### Use Cases
{chr(10).join(f"- {use_case}" for use_case in template.use_cases)}

### Pros
{chr(10).join(f"- {pro}" for pro in template.pros)}

### Cons
{chr(10).join(f"- {con}" for con in template.cons)}

## AI-Enhanced Analysis

{gpt4_response}

### Risk Mitigation
- Implement comprehensive monitoring
- Set up automated backups
- Plan for disaster recovery
- Establish CI/CD pipeline
"""
            return enhanced_recommendation
        except Exception as e:
            # Fallback to template-based recommendation if GPT-4 fails
            return self._generate_fallback_architecture_recommendation(template)
    
    async def _generate_technology_comparison(self, prompt: str, context: Dict[str, Any]) -> str:
        """Generate technology comparison with cost analysis using GPT-4"""
        # Extract technologies from context or use defaults
        technologies = context.get("technologies", ["Python/FastAPI", "Node.js/Express", "Java/Spring"])
        project_requirements = context.get("requirements", "general web application")
        team_size = context.get("team_size", 5)
        timeline = context.get("timeline", "6 months")
        
        # Base comparison data
        comparison = TechnologyComparison(
            technologies=technologies,
            criteria={
                "Python/FastAPI": {
                    "development_speed": 9.0,
                    "performance": 7.0,
                    "scalability": 8.0,
                    "cost": 8.0,
                    "ecosystem": 9.0
                },
                "Node.js/Express": {
                    "development_speed": 8.0,
                    "performance": 8.0,
                    "scalability": 7.0,
                    "cost": 9.0,
                    "ecosystem": 8.0
                },
                "Java/Spring": {
                    "development_speed": 6.0,
                    "performance": 9.0,
                    "scalability": 9.0,
                    "cost": 6.0,
                    "ecosystem": 9.0
                }
            },
            recommendation="Python/FastAPI",
            reasoning="Best balance of development speed, cost-effectiveness, and AI/ML ecosystem support"
        )
        
        # Use GPT-4 for enhanced analysis
        gpt4_prompt = f"""
        As a senior CTO, provide a detailed technology comparison for the following scenario:
        
        User Request: {prompt}
        Technologies to Compare: {', '.join(technologies)}
        Project Requirements: {project_requirements}
        Team Size: {team_size} developers
        Timeline: {timeline}
        
        Please provide:
        1. Detailed analysis of each technology's strengths and weaknesses
        2. Specific considerations for the given project requirements
        3. Cost analysis including development time, infrastructure, and maintenance
        4. Team expertise requirements and learning curve
        5. Long-term strategic implications
        6. Final recommendation with clear reasoning
        
        Be specific about trade-offs and provide actionable insights.
        """
        
        try:
            gpt4_analysis = await self._call_gpt4(gpt4_prompt)
        except Exception:
            gpt4_analysis = "Enhanced AI analysis unavailable - using baseline comparison"
        
        result = f"""
# Technology Comparison Analysis

## Technologies Evaluated
{chr(10).join(f"- {tech}" for tech in comparison.technologies)}

## Scoring Matrix (1-10 scale)
| Technology | Dev Speed | Performance | Scalability | Cost | Ecosystem |
|------------|-----------|-------------|-------------|------|-----------|
"""
        
        for tech, scores in comparison.criteria.items():
            result += f"| {tech} | {scores['development_speed']:.1f} | {scores['performance']:.1f} | {scores['scalability']:.1f} | {scores['cost']:.1f} | {scores['ecosystem']:.1f} |\n"
        
        result += f"""
## Recommendation: {comparison.recommendation}

### Reasoning
{comparison.reasoning}

## AI-Enhanced Analysis

{gpt4_analysis}

### Cost Analysis
- **Development Cost**: Lower due to faster development cycles
- **Infrastructure Cost**: Moderate, scales well with usage
- **Maintenance Cost**: Low, strong community support
- **Total Cost of Ownership**: Optimal for AI/ML focused applications

### Next Steps
1. Set up proof of concept with recommended stack
2. Benchmark performance with expected load
3. Evaluate team expertise and training needs
4. Plan migration strategy if switching from current stack
"""
        return result
    
    async def _generate_scaling_strategy(self, prompt: str, context: Dict[str, Any]) -> str:
        """Generate scaling strategy based on current and projected load using GPT-4"""
        current_users = context.get("current_users", 1000)
        projected_users = context.get("projected_users", 10000)
        current_cost = context.get("current_cost", 200)
        current_architecture = context.get("current_architecture", "monolithic")
        performance_issues = context.get("performance_issues", [])
        
        strategy = ScalingStrategy(
            current_load=f"{current_users} users",
            projected_load=f"{projected_users} users",
            recommended_approach="Horizontal scaling with load balancing",
            infrastructure_changes=[
                "Implement load balancer",
                "Add Redis caching layer",
                "Set up database read replicas",
                "Containerize applications with Docker",
                "Implement auto-scaling groups"
            ],
            estimated_cost_impact=current_cost * 3.5,
            timeline="3-6 months",
            risks=[
                "Database bottlenecks during peak load",
                "Session management complexity",
                "Increased monitoring complexity"
            ]
        )
        
        # Use GPT-4 for enhanced scaling analysis
        gpt4_prompt = f"""
        As a senior CTO, provide a comprehensive scaling strategy for the following scenario:
        
        User Request: {prompt}
        Current Load: {current_users} users
        Projected Load: {projected_users} users (Growth Factor: {projected_users / current_users:.1f}x)
        Current Monthly Cost: ${current_cost}
        Current Architecture: {current_architecture}
        Performance Issues: {', '.join(performance_issues) if performance_issues else 'None reported'}
        
        Please provide:
        1. Detailed scaling approach (vertical vs horizontal vs hybrid)
        2. Specific infrastructure changes with priorities
        3. Database scaling strategy
        4. Caching and CDN recommendations
        5. Monitoring and alerting improvements
        6. Cost projections with breakdown
        7. Implementation timeline with milestones
        8. Risk assessment and mitigation strategies
        9. Performance benchmarks to track
        
        Focus on practical, phased implementation that minimizes downtime.
        """
        
        try:
            gpt4_analysis = await self._call_gpt4(gpt4_prompt, max_tokens=2500)
        except Exception:
            gpt4_analysis = "Enhanced AI analysis unavailable - using baseline scaling strategy"
        
        result = f"""
# Scaling Strategy

## Current vs Projected Load
- **Current**: {strategy.current_load}
- **Projected**: {strategy.projected_load}
- **Growth Factor**: {projected_users / current_users:.1f}x

## Recommended Approach
{strategy.recommended_approach}

## Infrastructure Changes Required
{chr(10).join(f"- {change}" for change in strategy.infrastructure_changes)}

## Cost Impact
- **Current Monthly Cost**: ${current_cost}
- **Projected Monthly Cost**: ${strategy.estimated_cost_impact:.0f}
- **Cost Multiplier**: {strategy.estimated_cost_impact / current_cost:.1f}x

## Implementation Timeline
{strategy.timeline}

## Risk Assessment
{chr(10).join(f"- {risk}" for risk in strategy.risks)}

## AI-Enhanced Analysis

{gpt4_analysis}

## Mitigation Strategies
- Implement comprehensive monitoring and alerting
- Set up staging environment that mirrors production
- Plan for gradual rollout with feature flags
- Establish rollback procedures
- Conduct load testing before deployment

## Success Metrics
- Response time < 200ms for 95% of requests
- 99.9% uptime
- Auto-scaling triggers working correctly
- Database performance within acceptable limits
"""
        return result
    
    async def _generate_technical_decision(self, prompt: str, context: Dict[str, Any]) -> str:
        """Generate technical decisions with reasoning using GPT-4"""
        constraints = context.get("constraints", [])
        stakeholders = context.get("stakeholders", [])
        timeline = context.get("timeline", "not specified")
        budget = context.get("budget", "not specified")
        current_tech_stack = context.get("current_tech_stack", "not specified")
        
        # Use GPT-4 for comprehensive technical decision analysis
        gpt4_prompt = f"""
        As a senior CTO, provide a comprehensive technical decision analysis for:
        
        Decision Required: {prompt}
        
        Context:
        - Constraints: {', '.join(constraints) if constraints else 'None specified'}
        - Stakeholders: {', '.join(stakeholders) if stakeholders else 'Not specified'}
        - Timeline: {timeline}
        - Budget: {budget}
        - Current Tech Stack: {current_tech_stack}
        
        Please provide:
        1. Problem statement and context analysis
        2. Technical feasibility assessment
        3. Business impact analysis
        4. Risk assessment with mitigation strategies
        5. Cost-benefit analysis
        6. Alternative approaches with pros/cons
        7. Recommended approach with detailed reasoning
        8. Implementation roadmap with phases
        9. Success criteria and KPIs
        10. Next steps and action items
        
        Use a structured decision framework and provide actionable recommendations.
        """
        
        try:
            gpt4_analysis = await self._call_gpt4(gpt4_prompt, max_tokens=2500)
            return f"""
# Technical Decision Analysis (AI-Enhanced)

## Problem Statement
{prompt}

## AI-Enhanced Analysis

{gpt4_analysis}

## Decision Framework Applied
1. **Technical Feasibility**: Assessed implementation complexity and resource requirements
2. **Business Impact**: Evaluated alignment with business objectives and ROI
3. **Risk Assessment**: Identified potential risks and mitigation strategies
4. **Cost-Benefit Analysis**: Analyzed investment vs expected returns
5. **Timeline Considerations**: Evaluated fit within current roadmap and priorities

## Next Steps
1. Review and validate the analysis with key stakeholders
2. Conduct technical spike if needed to validate assumptions
3. Create detailed implementation plan with milestones
4. Allocate necessary resources and begin execution
"""
        except Exception:
            # Fallback to template-based decision framework
            return f"""
# Technical Decision Analysis

## Problem Statement
{prompt}

## Decision Framework Applied
1. **Technical Feasibility**: Can this be implemented with current resources?
2. **Business Impact**: How does this align with business objectives?
3. **Risk Assessment**: What are the potential risks and mitigation strategies?
4. **Cost-Benefit Analysis**: Is the investment justified by expected returns?
5. **Timeline Considerations**: How does this fit into current roadmap?

## Recommendation
Based on the analysis above, I recommend proceeding with a phased approach:

### Phase 1: Research and Prototyping (2-4 weeks)
- Conduct technical spike to validate feasibility
- Create proof of concept
- Identify potential blockers

### Phase 2: Implementation Planning (1-2 weeks)
- Define detailed requirements
- Create implementation roadmap
- Allocate resources

### Phase 3: Development and Testing (4-8 weeks)
- Implement core functionality
- Comprehensive testing
- Performance optimization

### Phase 4: Deployment and Monitoring (1-2 weeks)
- Staged rollout
- Monitor key metrics
- Gather user feedback

## Success Criteria
- Technical implementation meets performance requirements
- Business objectives are achieved
- No significant negative impact on existing systems
- Team can maintain and extend the solution

*Note: Enhanced AI analysis unavailable - using template-based framework*
"""
    
    def get_capabilities(self) -> List[AgentCapability]:
        return self.capabilities
    
    async def health_check(self) -> bool:
        return True
    
    def get_architecture_templates(self) -> List[ArchitectureTemplate]:
        """Get available architecture templates"""
        return self.architecture_templates
    
    def add_architecture_template(self, template: ArchitectureTemplate):
        """Add a new architecture template"""
        self.architecture_templates.append(template)
    
    async def _call_gpt4(self, prompt: str, max_tokens: int = 2000) -> str:
        """Call GPT-4 API with the given prompt"""
        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a senior CTO with 15+ years of experience in system architecture, technology selection, and scaling strategies. Provide detailed, practical, and actionable technical advice."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                max_tokens=max_tokens,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"GPT-4 API call failed: {str(e)}")
    
    def _generate_fallback_architecture_recommendation(self, template: ArchitectureTemplate) -> str:
        """Generate fallback recommendation when GPT-4 is unavailable"""
        return f"""
# Architecture Recommendation

## Recommended Stack: {template.name}
{template.description}

### Technology Stack
- **Backend**: {template.tech_stack.value}
- **Database**: {template.database.value}
- **Cloud Provider**: {template.cloud_provider.value}

### Cost Analysis
- **Estimated Monthly Cost**: ${template.estimated_cost_monthly}
- **Scalability Rating**: {template.scalability_rating}/10
- **Complexity Rating**: {template.complexity_rating}/10

### Use Cases
{chr(10).join(f"- {use_case}" for use_case in template.use_cases)}

### Pros
{chr(10).join(f"- {pro}" for pro in template.pros)}

### Cons
{chr(10).join(f"- {con}" for con in template.cons)}

### Implementation Roadmap
1. Set up development environment
2. Configure database and basic API structure
3. Implement core business logic
4. Add authentication and security
5. Deploy to staging environment
6. Performance testing and optimization
7. Production deployment

### Risk Mitigation
- Implement comprehensive monitoring
- Set up automated backups
- Plan for disaster recovery
- Establish CI/CD pipeline

*Note: Enhanced AI analysis unavailable - using template-based recommendation*
"""