"""
CTO Agent - Technology recommendations and architecture decisions
"""
import time
import json
import asyncio
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime, timedelta

from .base import Agent, AgentRequest, AgentResponse

logger = logging.getLogger(__name__)


class TechnologyTrendData:
    """Mock technology trend data - in production this would come from external APIs"""
    
    @staticmethod
    def get_trending_technologies() -> Dict[str, Any]:
        return {
            "frontend": {
                "rising": ["Next.js", "Svelte", "Solid.js"],
                "stable": ["React", "Vue.js", "Angular"],
                "declining": ["jQuery", "Backbone.js"]
            },
            "backend": {
                "rising": ["FastAPI", "Deno", "Bun"],
                "stable": ["Node.js", "Django", "Spring Boot"],
                "declining": ["PHP", "Ruby on Rails"]
            },
            "databases": {
                "rising": ["Supabase", "PlanetScale", "Neon"],
                "stable": ["PostgreSQL", "MongoDB", "Redis"],
                "declining": ["MySQL", "Oracle"]
            },
            "cloud": {
                "rising": ["Vercel", "Railway", "Render"],
                "stable": ["AWS", "GCP", "Azure"],
                "declining": ["Heroku", "DigitalOcean Apps"]
            }
        }
    
    @staticmethod
    def get_technology_comparison(tech1: str, tech2: str) -> Dict[str, Any]:
        """Compare two technologies"""
        comparisons = {
            ("react", "vue"): {
                "performance": {"react": 8, "vue": 9},
                "learning_curve": {"react": 7, "vue": 9},
                "ecosystem": {"react": 10, "vue": 8},
                "job_market": {"react": 10, "vue": 7},
                "recommendation": "React for larger teams and complex apps, Vue for rapid development"
            },
            ("fastapi", "django"): {
                "performance": {"fastapi": 10, "django": 7},
                "development_speed": {"fastapi": 8, "django": 9},
                "documentation": {"fastapi": 10, "django": 9},
                "ecosystem": {"fastapi": 7, "django": 10},
                "recommendation": "FastAPI for APIs and modern apps, Django for full-stack web apps"
            }
        }
        
        key = tuple(sorted([tech1.lower(), tech2.lower()]))
        return comparisons.get(key, {
            "message": f"Detailed comparison for {tech1} vs {tech2} not available",
            "recommendation": "Consider factors like team expertise, project requirements, and long-term maintenance"
        })


class CTOAgent(Agent):
    """CTO Agent for technology stack recommendations and architecture decisions"""
    
    def __init__(self):
        super().__init__(
            name="CTO Agent",
            description="Provides expert guidance on technology stack, scaling, and infrastructure decisions with latest technology trends"
        )
        self.trend_data = TechnologyTrendData()
    
    def get_capabilities(self) -> List[str]:
        """Return CTO agent capabilities"""
        return [
            "Technology stack recommendations with latest trends",
            "Architecture decision support with reasoning",
            "Scaling strategy recommendations based on business context",
            "Infrastructure planning and optimization",
            "Technology evaluation and comparison features",
            "Technical risk assessment and mitigation",
            "Integration with latest technology trend data",
            "Cost-benefit analysis for technology decisions",
            "Team skill assessment and technology adoption planning"
        ]
    
    async def process(self, request: AgentRequest) -> AgentResponse:
        """Process CTO-related requests with enhanced capabilities"""
        start_time = time.time()
        
        try:
            query = request.query.lower()
            context = request.context
            
            # Enhanced request analysis with more specific routing
            if "technology stack" in query or "tech stack" in query:
                result = await self._recommend_tech_stack(context)
            elif "compare" in query and ("vs" in query or "versus" in query):
                result = await self._compare_technologies(query, context)
            elif "architecture" in query or "system design" in query:
                result = await self._provide_architecture_guidance(context)
            elif "scaling" in query or "scale" in query:
                result = await self._recommend_scaling_strategy(context)
            elif "infrastructure" in query:
                result = await self._provide_infrastructure_guidance(context)
            elif "trends" in query or "trending" in query:
                result = await self._get_technology_trends(context)
            elif "cost" in query or "budget" in query:
                result = await self._analyze_technology_costs(context)
            elif "team" in query and ("skill" in query or "hire" in query):
                result = await self._provide_team_guidance(context)
            else:
                result = await self._provide_general_cto_guidance(request.query, context)
            
            # Calculate confidence score based on context and query specificity
            confidence_score = self._calculate_confidence_score(query, context)
            
            return AgentResponse(
                agent_name=self.name,
                success=True,
                result=result,
                metadata={
                    "request_type": self._classify_request_type(query),
                    "context_used": bool(context),
                    "trend_data_accessed": True,
                    "reasoning_provided": True
                },
                confidence_score=confidence_score,
                processing_time=time.time() - start_time,
                suggestions=self._generate_follow_up_suggestions(query)
            )
            
        except Exception as e:
            logger.error(f"CTO Agent error: {e}")
            return AgentResponse(
                agent_name=self.name,
                success=False,
                error=str(e),
                processing_time=time.time() - start_time
            )
    
    async def _recommend_tech_stack(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Recommend technology stack based on requirements with latest trends"""
        business_type = context.get("business_type", "general")
        scale = context.get("scale", "small")
        budget = context.get("budget", "medium")
        team_size = context.get("team_size", "small")
        timeline = context.get("timeline", "medium")
        
        # Get latest technology trends
        trends = self.trend_data.get_trending_technologies()
        
        # Customize recommendations based on context
        recommendations = self._generate_contextual_recommendations(
            business_type, scale, budget, team_size, timeline, trends
        )
        
        # Add reasoning for each recommendation
        reasoning = self._generate_tech_stack_reasoning(
            recommendations, business_type, scale, budget
        )
        
        return {
            "recommendations": recommendations,
            "reasoning": reasoning,
            "rationale": f"Based on {business_type} business with {scale} scale, {budget} budget, and {team_size} team",
            "technology_trends": {
                "considered": True,
                "trending_technologies": trends,
                "trend_impact": "Recommendations include both stable and emerging technologies"
            },
            "next_steps": self._generate_implementation_steps(recommendations),
            "risk_assessment": self._assess_technology_risks(recommendations),
            "estimated_timeline": self._estimate_implementation_timeline(recommendations, team_size),
            "cost_breakdown": self._estimate_technology_costs(recommendations, scale)
        }
    
    def _generate_contextual_recommendations(self, business_type: str, scale: str, 
                                           budget: str, team_size: str, timeline: str,
                                           trends: Dict[str, Any]) -> Dict[str, Any]:
        """Generate contextual technology recommendations"""
        
        # Base recommendations
        base_stack = {
            "frontend": {
                "framework": "React with TypeScript",
                "styling": "Tailwind CSS",
                "state_management": "Zustand",
                "build_tool": "Vite"
            },
            "backend": {
                "framework": "FastAPI (Python)",
                "database": "PostgreSQL",
                "cache": "Redis",
                "task_queue": "Celery"
            },
            "infrastructure": {
                "hosting": "Render or Railway",
                "monitoring": "Prometheus + Grafana",
                "ci_cd": "GitHub Actions",
                "cdn": "Cloudflare"
            }
        }
        
        # Adjust based on context
        if scale == "large" or budget == "high":
            base_stack["frontend"]["framework"] = "Next.js with TypeScript"
            base_stack["infrastructure"]["hosting"] = "AWS or GCP with Kubernetes"
            base_stack["infrastructure"]["database"] = "Managed PostgreSQL (RDS/Cloud SQL)"
            
        if team_size == "small" and timeline == "fast":
            base_stack["backend"]["framework"] = "FastAPI (for rapid development)"
            base_stack["frontend"]["framework"] = "Next.js (full-stack capabilities)"
            
        if business_type == "ecommerce":
            base_stack["backend"]["payment"] = "Stripe API"
            base_stack["backend"]["search"] = "Elasticsearch"
            base_stack["infrastructure"]["cdn"] = "CloudFront or Cloudflare"
            
        return base_stack
    
    def _generate_tech_stack_reasoning(self, recommendations: Dict[str, Any],
                                     business_type: str, scale: str, budget: str) -> Dict[str, Any]:
        """Generate detailed reasoning for technology choices"""
        return {
            "frontend_reasoning": {
                "framework_choice": f"Selected {recommendations['frontend']['framework']} for excellent TypeScript support, large ecosystem, and strong community",
                "styling_choice": "Tailwind CSS provides utility-first approach with excellent developer experience",
                "state_management": "Zustand offers simple, scalable state management without boilerplate"
            },
            "backend_reasoning": {
                "framework_choice": f"FastAPI chosen for automatic API documentation, type safety, and high performance",
                "database_choice": "PostgreSQL provides ACID compliance, excellent performance, and rich feature set",
                "cache_choice": "Redis enables fast data access and session management"
            },
            "infrastructure_reasoning": {
                "hosting_choice": f"Recommended hosting balances cost-effectiveness with scalability for {scale} scale",
                "monitoring_choice": "Prometheus + Grafana provides comprehensive observability",
                "ci_cd_choice": "GitHub Actions integrates seamlessly with code repository"
            }
        }
    
    async def _compare_technologies(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Compare two technologies based on query"""
        # Extract technology names from query
        technologies = self._extract_technologies_from_query(query)
        
        if len(technologies) >= 2:
            tech1, tech2 = technologies[0], technologies[1]
            comparison = self.trend_data.get_technology_comparison(tech1, tech2)
            
            return {
                "comparison": {
                    "technology_1": tech1,
                    "technology_2": tech2,
                    "detailed_comparison": comparison,
                    "market_trends": {
                        tech1: self._get_technology_trend_info(tech1),
                        tech2: self._get_technology_trend_info(tech2)
                    }
                },
                "recommendation": comparison.get("recommendation", "Both technologies have their merits"),
                "decision_factors": [
                    "Team expertise and learning curve",
                    "Project requirements and complexity",
                    "Long-term maintenance considerations",
                    "Community support and ecosystem",
                    "Performance requirements"
                ]
            }
        else:
            return {
                "error": "Could not identify two technologies to compare",
                "suggestion": "Please specify two technologies like 'React vs Vue' or 'FastAPI vs Django'"
            }
    
    def _extract_technologies_from_query(self, query: str) -> List[str]:
        """Extract technology names from comparison query"""
        # Simple extraction - in production this would be more sophisticated
        common_techs = [
            "react", "vue", "angular", "svelte", "next.js", "nuxt",
            "fastapi", "django", "flask", "express", "spring",
            "postgresql", "mysql", "mongodb", "redis",
            "aws", "gcp", "azure", "vercel", "netlify"
        ]
        
        found_techs = []
        query_lower = query.lower()
        
        for tech in common_techs:
            if tech in query_lower:
                found_techs.append(tech)
                
        return found_techs[:2]  # Return first two found
    
    def _get_technology_trend_info(self, technology: str) -> Dict[str, Any]:
        """Get trend information for a specific technology"""
        trends = self.trend_data.get_trending_technologies()
        
        for category, tech_lists in trends.items():
            for status, tech_list in tech_lists.items():
                if technology.lower() in [t.lower() for t in tech_list]:
                    return {
                        "category": category,
                        "trend_status": status,
                        "market_position": f"Currently {status} in {category} space"
                    }
        
        return {
            "category": "unknown",
            "trend_status": "stable",
            "market_position": "Established technology with steady adoption"
        }
    
    async def _get_technology_trends(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Get latest technology trends"""
        trends = self.trend_data.get_trending_technologies()
        
        return {
            "current_trends": trends,
            "analysis": {
                "rising_stars": self._analyze_rising_technologies(trends),
                "safe_bets": self._analyze_stable_technologies(trends),
                "technologies_to_avoid": self._analyze_declining_technologies(trends)
            },
            "recommendations": {
                "for_new_projects": "Consider rising technologies for competitive advantage",
                "for_existing_projects": "Stick with stable technologies unless compelling reason to migrate",
                "migration_strategy": "Plan gradual migration from declining technologies"
            },
            "last_updated": datetime.utcnow().isoformat()
        }
    
    def _analyze_rising_technologies(self, trends: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze rising technologies across categories"""
        rising = {}
        for category, tech_data in trends.items():
            rising[category] = {
                "technologies": tech_data.get("rising", []),
                "why_rising": f"These {category} technologies are gaining adoption due to improved developer experience and performance"
            }
        return rising
    
    def _analyze_stable_technologies(self, trends: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze stable technologies across categories"""
        stable = {}
        for category, tech_data in trends.items():
            stable[category] = {
                "technologies": tech_data.get("stable", []),
                "why_stable": f"These {category} technologies have proven track records and large ecosystems"
            }
        return stable
    
    def _analyze_declining_technologies(self, trends: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze declining technologies across categories"""
        declining = {}
        for category, tech_data in trends.items():
            declining[category] = {
                "technologies": tech_data.get("declining", []),
                "why_declining": f"These {category} technologies are being superseded by more modern alternatives"
            }
        return declining
    
    async def _analyze_technology_costs(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze technology costs and provide budget recommendations"""
        scale = context.get("scale", "small")
        team_size = context.get("team_size", "small")
        
        cost_breakdown = {
            "development_costs": {
                "small_team": "$5,000-15,000/month",
                "medium_team": "$15,000-50,000/month", 
                "large_team": "$50,000+/month"
            },
            "infrastructure_costs": {
                "mvp": "$100-500/month",
                "growth": "$500-2,000/month",
                "scale": "$2,000-10,000+/month"
            },
            "tooling_costs": {
                "essential": "$200-500/month",
                "professional": "$500-1,500/month",
                "enterprise": "$1,500+/month"
            }
        }
        
        return {
            "cost_breakdown": cost_breakdown,
            "recommendations_by_scale": {
                "startup": "Focus on free/low-cost tools, managed services to reduce operational overhead",
                "growth": "Invest in monitoring, security, and developer productivity tools",
                "enterprise": "Consider enterprise solutions for compliance, security, and support"
            },
            "cost_optimization_tips": [
                "Use managed services to reduce operational costs",
                "Implement auto-scaling to optimize resource usage",
                "Choose tools with transparent, predictable pricing",
                "Monitor and optimize cloud resource usage regularly"
            ]
        }
    
    async def _provide_team_guidance(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Provide team building and skill development guidance"""
        current_team_size = context.get("current_team_size", 0)
        target_team_size = context.get("target_team_size", 5)
        budget = context.get("budget", "medium")
        
        return {
            "hiring_roadmap": {
                "phase_1": "Full-stack developer + DevOps engineer",
                "phase_2": "Frontend specialist + Backend specialist",
                "phase_3": "Data engineer + QA engineer + Product manager"
            },
            "skill_priorities": [
                "TypeScript/JavaScript proficiency",
                "Modern framework experience (React/Vue/Angular)",
                "API development and database design",
                "Cloud platform experience",
                "Testing and CI/CD practices"
            ],
            "team_structure_recommendations": {
                "small_team": "T-shaped developers with broad skills",
                "medium_team": "Mix of generalists and specialists",
                "large_team": "Specialized roles with clear ownership"
            },
            "skill_development_plan": [
                "Establish coding standards and best practices",
                "Implement code review processes",
                "Provide learning budget for courses and conferences",
                "Encourage open source contributions"
            ]
        }

    async def _provide_architecture_guidance(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Provide system architecture guidance with detailed reasoning"""
        business_type = context.get("business_type", "general")
        scale = context.get("scale", "small")
        complexity = context.get("complexity", "medium")
        
        return {
            "architecture_principles": [
                "Start with monolith, evolve to microservices when needed",
                "Design for failure and resilience from day one",
                "Implement proper separation of concerns",
                "Use event-driven architecture for scalability",
                "Follow SOLID principles and clean architecture"
            ],
            "recommended_patterns": {
                "api_design": {
                    "pattern": "RESTful APIs with OpenAPI documentation",
                    "reasoning": "Provides clear contracts and automatic documentation"
                },
                "data_access": {
                    "pattern": "Repository pattern with ORM",
                    "reasoning": "Abstracts data layer and enables easier testing"
                },
                "error_handling": {
                    "pattern": "Centralized error handling with proper logging",
                    "reasoning": "Consistent error responses and better debugging"
                },
                "security": {
                    "pattern": "JWT authentication with role-based access control",
                    "reasoning": "Stateless authentication with fine-grained permissions"
                }
            },
            "architecture_by_scale": {
                "small": {
                    "approach": "Modular monolith",
                    "reasoning": "Simpler deployment and debugging while maintaining good structure"
                },
                "medium": {
                    "approach": "Service-oriented architecture",
                    "reasoning": "Balance between simplicity and scalability"
                },
                "large": {
                    "approach": "Microservices with API gateway",
                    "reasoning": "Independent scaling and deployment of services"
                }
            },
            "scalability_considerations": [
                "Horizontal scaling with load balancers",
                "Database read replicas for read-heavy workloads", 
                "Caching strategy with Redis for frequently accessed data",
                "CDN for static assets and global distribution",
                "Asynchronous processing for long-running tasks"
            ],
            "decision_framework": {
                "when_to_split_services": [
                    "Different scaling requirements",
                    "Different teams owning different parts",
                    "Different technology requirements",
                    "Clear business domain boundaries"
                ],
                "when_to_stay_monolithic": [
                    "Small team (< 10 developers)",
                    "Unclear domain boundaries",
                    "Rapid prototyping phase",
                    "Limited operational expertise"
                ]
            }
        }
    
    async def _recommend_scaling_strategy(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Recommend scaling strategy based on business context"""
        current_users = context.get("current_users", 1000)
        target_users = context.get("target_users", 10000)
        growth_rate = context.get("growth_rate", "medium")
        budget = context.get("budget", "medium")
        
        scaling_strategy = self._determine_scaling_approach(current_users, target_users, growth_rate)
        
        return {
            "current_assessment": {
                "users": current_users,
                "estimated_load": f"{current_users * 10} requests/hour",
                "current_bottlenecks": self._identify_potential_bottlenecks(current_users)
            },
            "scaling_phases": {
                "phase_1": {
                    "users": "0-1K",
                    "strategy": "Single server with managed database",
                    "cost": "$100-300/month",
                    "reasoning": "Simple setup, easy to debug, cost-effective"
                },
                "phase_2": {
                    "users": "1K-10K", 
                    "strategy": "Load balancer + multiple app servers + database replica",
                    "cost": "$300-1500/month",
                    "reasoning": "Horizontal scaling, improved availability"
                },
                "phase_3": {
                    "users": "10K-100K",
                    "strategy": "Microservices + auto-scaling + CDN + caching",
                    "cost": "$1500-5000/month",
                    "reasoning": "Independent scaling, global distribution"
                },
                "phase_4": {
                    "users": "100K+",
                    "strategy": "Multi-region deployment + advanced caching + edge computing",
                    "cost": "$5000+/month",
                    "reasoning": "Global scale, low latency, high availability"
                }
            },
            "immediate_actions": self._get_immediate_scaling_actions(current_users, target_users),
            "metrics_to_monitor": [
                "Response time (< 200ms target)",
                "Database connection count",
                "Memory usage (< 80%)",
                "CPU utilization (< 70%)",
                "Error rate (< 0.1%)",
                "Throughput (requests/second)"
            ],
            "scaling_triggers": {
                "scale_up_when": [
                    "Response time > 500ms consistently",
                    "CPU usage > 80% for 5+ minutes",
                    "Memory usage > 90%",
                    "Error rate > 1%"
                ],
                "scale_down_when": [
                    "CPU usage < 30% for 30+ minutes",
                    "Memory usage < 50%",
                    "Low traffic periods (configurable)"
                ]
            },
            "cost_optimization": {
                "strategies": [
                    "Use auto-scaling to match demand",
                    "Implement efficient caching",
                    "Optimize database queries",
                    "Use CDN for static assets",
                    "Consider spot instances for non-critical workloads"
                ],
                "estimated_savings": "20-40% with proper optimization"
            }
        }
    
    def _determine_scaling_approach(self, current_users: int, target_users: int, growth_rate: str) -> str:
        """Determine the appropriate scaling approach"""
        if growth_rate == "rapid" and target_users > current_users * 10:
            return "aggressive_scaling"
        elif growth_rate == "slow" and target_users < current_users * 3:
            return "conservative_scaling"
        else:
            return "balanced_scaling"
    
    def _identify_potential_bottlenecks(self, current_users: int) -> List[str]:
        """Identify potential bottlenecks based on user count"""
        bottlenecks = []
        
        if current_users > 1000:
            bottlenecks.extend([
                "Database connection limits",
                "Single server CPU/memory limits"
            ])
        
        if current_users > 5000:
            bottlenecks.extend([
                "Database query performance",
                "Network bandwidth limits",
                "Session storage scalability"
            ])
            
        if current_users > 10000:
            bottlenecks.extend([
                "Database write performance",
                "Cache invalidation complexity",
                "Cross-service communication latency"
            ])
            
        return bottlenecks or ["No significant bottlenecks expected at current scale"]
    
    def _get_immediate_scaling_actions(self, current_users: int, target_users: int) -> List[str]:
        """Get immediate actions based on current and target users"""
        actions = [
            "Implement application performance monitoring (APM)",
            "Set up database connection pooling",
            "Add basic caching layer (Redis)",
            "Optimize critical database queries"
        ]
        
        if target_users > current_users * 5:
            actions.extend([
                "Plan database read replica setup",
                "Implement horizontal scaling architecture",
                "Set up load testing environment"
            ])
            
        if target_users > 10000:
            actions.extend([
                "Design microservices architecture",
                "Plan CDN implementation",
                "Set up auto-scaling infrastructure"
            ])
            
        return actions
    
    def _generate_implementation_steps(self, recommendations: Dict[str, Any]) -> List[str]:
        """Generate implementation steps based on recommendations"""
        return [
            "Set up development environment with recommended tools",
            "Create project structure following best practices",
            "Implement core backend API with authentication",
            "Build frontend components and routing",
            "Set up database schema and migrations",
            "Implement caching and session management",
            "Add monitoring and logging",
            "Set up CI/CD pipeline",
            "Deploy to staging environment",
            "Conduct load testing and optimization",
            "Deploy to production with monitoring"
        ]
    
    def _assess_technology_risks(self, recommendations: Dict[str, Any]) -> Dict[str, Any]:
        """Assess risks associated with technology choices"""
        return {
            "low_risk": [
                "Using established frameworks with large communities",
                "Choosing technologies with good documentation",
                "Selecting tools with proven track records"
            ],
            "medium_risk": [
                "Adopting newer technologies without long-term support guarantees",
                "Using technologies with smaller communities",
                "Implementing complex architectures without sufficient expertise"
            ],
            "high_risk": [
                "Betting on experimental or alpha-stage technologies",
                "Using technologies without clear migration paths",
                "Over-engineering solutions for current needs"
            ],
            "mitigation_strategies": [
                "Start with proven technologies and migrate gradually",
                "Maintain good documentation and knowledge sharing",
                "Have backup plans for critical technology choices",
                "Invest in team training and expertise development"
            ]
        }
    
    def _estimate_implementation_timeline(self, recommendations: Dict[str, Any], team_size: str) -> Dict[str, Any]:
        """Estimate implementation timeline based on recommendations and team size"""
        base_timeline = {
            "mvp": "2-4 months",
            "full_featured": "6-12 months",
            "enterprise_ready": "12-18 months"
        }
        
        team_multipliers = {
            "small": 1.5,
            "medium": 1.0,
            "large": 0.7
        }
        
        multiplier = team_multipliers.get(team_size, 1.0)
        
        return {
            "estimated_timeline": {
                "mvp": f"{int(2 * multiplier)}-{int(4 * multiplier)} months",
                "full_featured": f"{int(6 * multiplier)}-{int(12 * multiplier)} months",
                "enterprise_ready": f"{int(12 * multiplier)}-{int(18 * multiplier)} months"
            },
            "factors_affecting_timeline": [
                "Team experience with chosen technologies",
                "Complexity of business requirements",
                "Integration requirements with existing systems",
                "Quality and testing requirements",
                "Deployment and infrastructure complexity"
            ],
            "timeline_optimization_tips": [
                "Use proven technologies to reduce learning curve",
                "Implement MVP first, then iterate",
                "Leverage existing libraries and frameworks",
                "Automate testing and deployment early"
            ]
        }
    
    def _estimate_technology_costs(self, recommendations: Dict[str, Any], scale: str) -> Dict[str, Any]:
        """Estimate costs for recommended technologies"""
        cost_ranges = {
            "small": {
                "development_tools": "$100-300/month",
                "hosting": "$50-200/month", 
                "database": "$25-100/month",
                "monitoring": "$0-50/month",
                "total": "$175-650/month"
            },
            "medium": {
                "development_tools": "$300-800/month",
                "hosting": "$200-800/month",
                "database": "$100-400/month", 
                "monitoring": "$50-200/month",
                "total": "$650-2200/month"
            },
            "large": {
                "development_tools": "$800-2000/month",
                "hosting": "$800-5000/month",
                "database": "$400-2000/month",
                "monitoring": "$200-800/month",
                "total": "$2200-9800/month"
            }
        }
        
        return {
            "cost_breakdown": cost_ranges.get(scale, cost_ranges["medium"]),
            "cost_optimization_opportunities": [
                "Use free tiers for development and testing",
                "Implement auto-scaling to optimize resource usage",
                "Choose managed services to reduce operational overhead",
                "Monitor and optimize resource usage regularly"
            ],
            "hidden_costs_to_consider": [
                "Developer training and onboarding",
                "Third-party service integrations",
                "Security and compliance tools",
                "Backup and disaster recovery",
                "Support and maintenance"
            ]
        }

    async def _provide_infrastructure_guidance(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Provide comprehensive infrastructure guidance"""
        scale = context.get("scale", "small")
        budget = context.get("budget", "medium")
        compliance_requirements = context.get("compliance", [])
        
        return {
            "infrastructure_stack": {
                "compute": {
                    "recommendation": "Container-based deployment (Docker)",
                    "reasoning": "Consistent environments, easy scaling, platform independence"
                },
                "orchestration": {
                    "small_scale": "Docker Compose",
                    "large_scale": "Kubernetes or managed container services",
                    "reasoning": "Start simple, scale to orchestration when needed"
                },
                "database": {
                    "recommendation": "Managed PostgreSQL (RDS, Cloud SQL, or Supabase)",
                    "reasoning": "Reduces operational overhead, automatic backups, scaling"
                },
                "cache": {
                    "recommendation": "Managed Redis (ElastiCache, Cloud Memorystore)",
                    "reasoning": "High performance, managed maintenance and scaling"
                },
                "storage": {
                    "recommendation": "Object storage (S3, GCS, or compatible)",
                    "reasoning": "Scalable, durable, cost-effective for files and assets"
                }
            },
            "deployment_strategy": {
                "development": {
                    "approach": "Local Docker Compose",
                    "benefits": "Fast iteration, offline development, consistent environment"
                },
                "staging": {
                    "approach": "Cloud-based staging environment",
                    "benefits": "Production-like testing, integration testing, performance validation"
                },
                "production": {
                    "approach": "Blue-green deployment with health checks",
                    "benefits": "Zero-downtime deployments, easy rollbacks, risk mitigation"
                }
            },
            "monitoring_stack": {
                "metrics": {
                    "tool": "Prometheus + Grafana",
                    "reasoning": "Open source, powerful querying, excellent visualization"
                },
                "logging": {
                    "approach": "Structured logging with centralized collection",
                    "tools": "ELK stack or managed solutions like DataDog"
                },
                "alerting": {
                    "tool": "PagerDuty or similar for critical alerts",
                    "strategy": "Alert on symptoms, not causes; avoid alert fatigue"
                },
                "uptime": {
                    "tool": "External monitoring service (Pingdom, UptimeRobot)",
                    "reasoning": "Independent monitoring, customer-facing status pages"
                }
            },
            "security_considerations": {
                "network": "VPC with private subnets, security groups, WAF",
                "data": "Encryption at rest and in transit, key management",
                "access": "IAM roles, least privilege principle, MFA",
                "compliance": self._get_compliance_recommendations(compliance_requirements)
            },
            "cost_optimization": {
                "strategies": [
                    "Use managed services to reduce operational costs",
                    "Implement auto-scaling for variable workloads",
                    "Use spot instances for non-critical workloads",
                    "Regular cost reviews and resource optimization"
                ],
                "monitoring": "Set up billing alerts and cost tracking"
            },
            "disaster_recovery": {
                "backup_strategy": "Automated daily backups with point-in-time recovery",
                "multi_region": "Consider multi-region deployment for critical applications",
                "rto_rpo": "Define Recovery Time Objective (RTO) and Recovery Point Objective (RPO)"
            }
        }
    
    def _get_compliance_recommendations(self, requirements: List[str]) -> Dict[str, Any]:
        """Get compliance-specific recommendations"""
        recommendations = {}
        
        if "gdpr" in [r.lower() for r in requirements]:
            recommendations["gdpr"] = {
                "data_protection": "Implement data encryption and access controls",
                "privacy": "Data minimization, consent management, right to deletion",
                "documentation": "Maintain data processing records and privacy policies"
            }
            
        if "soc2" in [r.lower() for r in requirements]:
            recommendations["soc2"] = {
                "security": "Access controls, encryption, vulnerability management",
                "availability": "Monitoring, incident response, business continuity",
                "processing_integrity": "Data validation, error handling, quality controls"
            }
            
        if not recommendations:
            recommendations["general"] = {
                "security": "Implement basic security best practices",
                "monitoring": "Log access and changes for audit trails",
                "backup": "Regular backups and disaster recovery planning"
            }
            
        return recommendations

    async def _provide_general_cto_guidance(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Provide comprehensive general CTO guidance"""
        business_stage = context.get("business_stage", "startup")
        
        return {
            "guidance": f"As your AI CTO, here's strategic guidance for your {business_stage} stage:",
            "strategic_priorities": self._get_stage_specific_priorities(business_stage),
            "technical_debt_management": {
                "prevention": [
                    "Establish coding standards and style guides",
                    "Implement automated code quality checks",
                    "Regular architecture reviews",
                    "Documentation as part of development process"
                ],
                "remediation": [
                    "Regular refactoring sprints",
                    "Technical debt tracking and prioritization",
                    "Code review processes",
                    "Automated testing to prevent regressions"
                ]
            },
            "team_building_strategy": {
                "hiring_priorities": self._get_hiring_priorities(business_stage),
                "skill_development": [
                    "Invest in continuous learning budget",
                    "Encourage conference attendance and knowledge sharing",
                    "Internal tech talks and code reviews",
                    "Mentorship programs for junior developers"
                ],
                "culture_building": [
                    "Foster innovation and experimentation",
                    "Encourage ownership and accountability",
                    "Promote work-life balance and sustainable practices",
                    "Build inclusive and diverse teams"
                ]
            },
            "technology_decision_framework": {
                "evaluation_criteria": [
                    "Alignment with business goals",
                    "Team expertise and learning curve",
                    "Long-term maintenance and support",
                    "Community and ecosystem strength",
                    "Performance and scalability requirements"
                ],
                "decision_process": [
                    "Define requirements and constraints",
                    "Research and evaluate options",
                    "Prototype and test critical paths",
                    "Make decision with clear rationale",
                    "Document and communicate decision"
                ]
            },
            "risk_management": {
                "technical_risks": [
                    "Single points of failure in architecture",
                    "Vendor lock-in with proprietary technologies",
                    "Security vulnerabilities and data breaches",
                    "Scalability bottlenecks"
                ],
                "mitigation_strategies": [
                    "Implement redundancy and failover mechanisms",
                    "Choose open standards and avoid vendor lock-in",
                    "Regular security audits and penetration testing",
                    "Performance testing and capacity planning"
                ]
            }
        }
    
    def _get_stage_specific_priorities(self, business_stage: str) -> List[str]:
        """Get priorities specific to business stage"""
        priorities = {
            "startup": [
                "Build MVP quickly to validate market fit",
                "Focus on core features, avoid feature creep",
                "Implement basic monitoring and error tracking",
                "Plan for rapid iteration and pivoting"
            ],
            "growth": [
                "Scale infrastructure to handle increased load",
                "Implement proper testing and CI/CD processes",
                "Build team and establish development processes",
                "Focus on performance and user experience"
            ],
            "mature": [
                "Optimize for efficiency and cost reduction",
                "Invest in advanced monitoring and observability",
                "Focus on security and compliance",
                "Plan for long-term technical strategy"
            ]
        }
        return priorities.get(business_stage, priorities["startup"])
    
    def _get_hiring_priorities(self, business_stage: str) -> List[str]:
        """Get hiring priorities based on business stage"""
        priorities = {
            "startup": [
                "Full-stack developers who can wear multiple hats",
                "Senior developer who can make architectural decisions",
                "DevOps engineer for infrastructure and deployment"
            ],
            "growth": [
                "Frontend and backend specialists",
                "QA engineer for testing and quality assurance",
                "Product manager for feature prioritization",
                "Data engineer for analytics and insights"
            ],
            "mature": [
                "Security engineer for compliance and risk management",
                "Site reliability engineer for operational excellence",
                "Technical architect for long-term planning",
                "Engineering manager for team leadership"
            ]
        }
        return priorities.get(business_stage, priorities["startup"])
    
    def _calculate_confidence_score(self, query: str, context: Dict[str, Any]) -> float:
        """Calculate confidence score for the response"""
        base_confidence = 0.8
        
        # Increase confidence if we have relevant context
        if context:
            base_confidence += 0.1
            
        # Increase confidence for specific, well-defined queries
        specific_terms = ["technology stack", "architecture", "scaling", "infrastructure"]
        if any(term in query.lower() for term in specific_terms):
            base_confidence += 0.1
            
        return min(base_confidence, 1.0)
    
    def _generate_follow_up_suggestions(self, query: str) -> List[str]:
        """Generate follow-up suggestions based on the query"""
        suggestions = []
        
        if "technology stack" in query.lower():
            suggestions.extend([
                "Would you like a detailed comparison between specific technologies?",
                "Do you need guidance on implementation timeline and costs?",
                "Would you like recommendations for team skills and hiring?"
            ])
        elif "architecture" in query.lower():
            suggestions.extend([
                "Would you like specific guidance on database design?",
                "Do you need help with API design patterns?",
                "Would you like recommendations for monitoring and observability?"
            ])
        elif "scaling" in query.lower():
            suggestions.extend([
                "Would you like help with performance optimization strategies?",
                "Do you need guidance on infrastructure cost optimization?",
                "Would you like recommendations for monitoring and alerting?"
            ])
        else:
            suggestions.extend([
                "Would you like technology stack recommendations?",
                "Do you need help with architecture decisions?",
                "Would you like guidance on scaling strategies?"
            ])
            
        return suggestions[:3]  # Return top 3 suggestions
    
    def _classify_request_type(self, query: str) -> str:
        """Classify the type of CTO request with enhanced categorization"""
        query_lower = query.lower()
        
        if "technology stack" in query_lower or "tech stack" in query_lower:
            return "tech_stack_recommendation"
        elif "compare" in query_lower and ("vs" in query_lower or "versus" in query_lower):
            return "technology_comparison"
        elif "architecture" in query_lower or "system design" in query_lower:
            return "architecture_guidance"
        elif "scaling" in query_lower or "scale" in query_lower:
            return "scaling_strategy"
        elif "infrastructure" in query_lower:
            return "infrastructure_guidance"
        elif "trends" in query_lower or "trending" in query_lower:
            return "technology_trends"
        elif "cost" in query_lower or "budget" in query_lower:
            return "cost_analysis"
        elif "team" in query_lower and ("skill" in query_lower or "hire" in query_lower):
            return "team_guidance"
        else:
            return "general_cto_guidance"