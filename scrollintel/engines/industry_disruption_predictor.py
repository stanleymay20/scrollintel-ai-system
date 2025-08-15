"""
Industry Disruption Predictor for Big Tech CTO Capabilities

This engine predicts industry disruptions using advanced pattern recognition,
market analysis, and technology trend forecasting.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Tuple
import numpy as np
from dataclasses import asdict

from ..models.strategic_planning_models import (
    DisruptionPrediction, TechnologyDomain, MarketChange, IndustryForecast
)

logger = logging.getLogger(__name__)


class IndustryDisruptionPredictor:
    """
    Advanced engine for predicting industry disruptions and market shifts
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.disruption_patterns = self._initialize_disruption_patterns()
        self.market_indicators = self._initialize_market_indicators()
        self.technology_signals = self._initialize_technology_signals()
        
    def _initialize_disruption_patterns(self) -> Dict[str, Any]:
        """Initialize historical disruption patterns"""
        return {
            "s_curve_adoption": {
                "early_phase_duration": 3,  # years
                "rapid_growth_phase": 5,    # years
                "maturity_phase": 8,        # years
                "adoption_threshold": 0.10  # 10% market penetration
            },
            "technology_convergence": {
                "convergence_indicators": [
                    "cross_domain_patents",
                    "interdisciplinary_research",
                    "hybrid_business_models",
                    "platform_integration"
                ],
                "convergence_acceleration": 0.15  # 15% annual increase
            },
            "market_concentration": {
                "disruption_threshold": 0.70,  # HHI above 0.7
                "incumbent_vulnerability": 0.60,
                "new_entrant_advantage": 0.40
            },
            "regulatory_disruption": {
                "policy_lag_time": 2,  # years behind technology
                "compliance_cost_threshold": 0.15,  # 15% of revenue
                "regulatory_uncertainty_index": 0.45
            }
        }
    
    def _initialize_market_indicators(self) -> Dict[str, Any]:
        """Initialize market disruption indicators"""
        return {
            "venture_capital_flow": {
                "disruption_threshold": 2.0,  # 2x increase in funding
                "sector_concentration": 0.30,  # 30% of VC in one sector
                "unicorn_creation_rate": 0.25   # 25% increase annually
            },
            "patent_activity": {
                "filing_acceleration": 0.20,  # 20% increase in filings
                "cross_industry_citations": 0.15,
                "breakthrough_patent_ratio": 0.05
            },
            "talent_migration": {
                "skill_demand_shift": 0.30,  # 30% change in job postings
                "salary_inflation_rate": 0.25,
                "cross_industry_mobility": 0.20
            },
            "customer_behavior": {
                "adoption_acceleration": 0.40,  # 40% faster adoption
                "expectation_evolution": 0.35,
                "switching_cost_reduction": 0.25
            }
        }
    
    def _initialize_technology_signals(self) -> Dict[str, Any]:
        """Initialize technology disruption signals"""
        return {
            "performance_improvement": {
                "exponential_threshold": 2.0,  # 2x improvement annually
                "cost_reduction_rate": 0.30,   # 30% cost reduction
                "accessibility_increase": 0.50  # 50% more accessible
            },
            "platform_effects": {
                "network_effect_strength": 0.60,
                "ecosystem_growth_rate": 0.40,
                "developer_adoption": 0.35
            },
            "standardization": {
                "standard_emergence_time": 18,  # months
                "industry_consensus_level": 0.70,
                "interoperability_score": 0.80
            }
        }
    
    async def predict_industry_disruption(
        self,
        industry: str,
        time_horizon: int,
        market_data: Optional[Dict[str, Any]] = None
    ) -> List[DisruptionPrediction]:
        """
        Predict potential disruptions for a specific industry
        
        Args:
            industry: Target industry for analysis
            time_horizon: Prediction horizon in years
            market_data: Optional market data for analysis
            
        Returns:
            List of disruption predictions with probabilities and impacts
        """
        try:
            self.logger.info(f"Predicting disruptions for {industry} over {time_horizon} years")
            
            # Analyze current market state
            market_state = await self._analyze_market_state(industry, market_data)
            
            # Identify technology disruption signals
            tech_signals = await self._identify_technology_signals(industry)
            
            # Assess regulatory disruption potential
            regulatory_risks = await self._assess_regulatory_disruption(industry)
            
            # Analyze competitive dynamics
            competitive_dynamics = await self._analyze_competitive_dynamics(industry)
            
            # Generate disruption scenarios
            disruptions = await self._generate_disruption_scenarios(
                industry, time_horizon, market_state, tech_signals, 
                regulatory_risks, competitive_dynamics
            )
            
            # Rank and filter disruptions by probability and impact
            significant_disruptions = [
                d for d in disruptions 
                if d.probability > 0.15 and d.impact_magnitude > 5.0
            ]
            
            # Sort by combined risk score (probability * impact)
            significant_disruptions.sort(
                key=lambda x: x.probability * x.impact_magnitude, 
                reverse=True
            )
            
            self.logger.info(f"Identified {len(significant_disruptions)} significant disruptions")
            return significant_disruptions[:10]  # Return top 10
            
        except Exception as e:
            self.logger.error(f"Error predicting industry disruption: {str(e)}")
            raise
    
    async def _analyze_market_state(
        self,
        industry: str,
        market_data: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze current market state for disruption indicators"""
        
        # Default market state if no data provided
        if not market_data:
            market_data = self._get_default_market_data(industry)
        
        state = {
            "market_maturity": self._assess_market_maturity(industry, market_data),
            "concentration_level": self._calculate_market_concentration(market_data),
            "growth_rate": market_data.get("growth_rate", 0.10),
            "innovation_rate": self._assess_innovation_rate(industry, market_data),
            "customer_satisfaction": market_data.get("customer_satisfaction", 0.70),
            "switching_costs": self._assess_switching_costs(industry),
            "regulatory_burden": self._assess_regulatory_burden(industry),
            "technology_debt": self._assess_technology_debt(industry)
        }
        
        return state
    
    def _get_default_market_data(self, industry: str) -> Dict[str, Any]:
        """Get default market data for industry"""
        defaults = {
            "artificial_intelligence": {
                "growth_rate": 0.25,
                "market_size": 150e9,
                "customer_satisfaction": 0.75,
                "major_players": 8,
                "innovation_cycle": 18  # months
            },
            "cloud_computing": {
                "growth_rate": 0.18,
                "market_size": 400e9,
                "customer_satisfaction": 0.80,
                "major_players": 5,
                "innovation_cycle": 12
            },
            "cybersecurity": {
                "growth_rate": 0.15,
                "market_size": 200e9,
                "customer_satisfaction": 0.65,
                "major_players": 12,
                "innovation_cycle": 24
            }
        }
        
        return defaults.get(
            industry.lower().replace(" ", "_"),
            {
                "growth_rate": 0.10,
                "market_size": 100e9,
                "customer_satisfaction": 0.70,
                "major_players": 10,
                "innovation_cycle": 18
            }
        )
    
    def _assess_market_maturity(
        self,
        industry: str,
        market_data: Dict[str, Any]
    ) -> float:
        """Assess market maturity (0-1 scale)"""
        
        # Factors indicating market maturity
        growth_rate = market_data.get("growth_rate", 0.10)
        innovation_cycle = market_data.get("innovation_cycle", 18)  # months
        major_players = market_data.get("major_players", 10)
        
        # Lower growth rate indicates higher maturity
        growth_maturity = max(0, 1 - (growth_rate / 0.30))
        
        # Longer innovation cycles indicate higher maturity
        cycle_maturity = min(1, innovation_cycle / 36)
        
        # Fewer major players can indicate consolidation (maturity)
        player_maturity = max(0, 1 - (major_players / 20))
        
        # Weighted average
        maturity = (growth_maturity * 0.4 + cycle_maturity * 0.3 + player_maturity * 0.3)
        
        return min(1.0, max(0.0, maturity))
    
    def _calculate_market_concentration(self, market_data: Dict[str, Any]) -> float:
        """Calculate market concentration (HHI-like metric)"""
        major_players = market_data.get("major_players", 10)
        
        # Estimate concentration based on number of major players
        if major_players <= 3:
            return 0.80  # Highly concentrated
        elif major_players <= 6:
            return 0.60  # Moderately concentrated
        elif major_players <= 10:
            return 0.40  # Competitive
        else:
            return 0.25  # Fragmented
    
    def _assess_innovation_rate(
        self,
        industry: str,
        market_data: Dict[str, Any]
    ) -> float:
        """Assess rate of innovation in industry"""
        innovation_cycle = market_data.get("innovation_cycle", 18)  # months
        
        # Shorter cycles indicate higher innovation rate
        innovation_rate = max(0.1, 1.0 - (innovation_cycle / 48))
        
        # Industry-specific adjustments
        industry_multipliers = {
            "artificial_intelligence": 1.3,
            "quantum_computing": 1.2,
            "biotechnology": 0.8,
            "cybersecurity": 1.1,
            "cloud_computing": 1.0
        }
        
        multiplier = industry_multipliers.get(
            industry.lower().replace(" ", "_"), 1.0
        )
        
        return min(1.0, innovation_rate * multiplier)
    
    def _assess_switching_costs(self, industry: str) -> float:
        """Assess customer switching costs (0-1 scale)"""
        switching_costs = {
            "cloud_computing": 0.70,  # High switching costs
            "enterprise_software": 0.80,
            "social_media": 0.60,
            "cybersecurity": 0.65,
            "artificial_intelligence": 0.50,
            "e_commerce": 0.30,
            "mobile_apps": 0.20
        }
        
        return switching_costs.get(
            industry.lower().replace(" ", "_"), 0.50
        )
    
    def _assess_regulatory_burden(self, industry: str) -> float:
        """Assess regulatory burden (0-1 scale)"""
        regulatory_burden = {
            "financial_services": 0.90,
            "healthcare": 0.85,
            "biotechnology": 0.80,
            "telecommunications": 0.75,
            "artificial_intelligence": 0.60,
            "cybersecurity": 0.65,
            "cloud_computing": 0.55,
            "e_commerce": 0.50
        }
        
        return regulatory_burden.get(
            industry.lower().replace(" ", "_"), 0.50
        )
    
    def _assess_technology_debt(self, industry: str) -> float:
        """Assess technology debt in industry (0-1 scale)"""
        tech_debt = {
            "financial_services": 0.80,  # High legacy system burden
            "healthcare": 0.75,
            "telecommunications": 0.70,
            "manufacturing": 0.65,
            "retail": 0.60,
            "artificial_intelligence": 0.30,  # New industry, less debt
            "cloud_computing": 0.35,
            "cybersecurity": 0.40
        }
        
        return tech_debt.get(
            industry.lower().replace(" ", "_"), 0.55
        )
    
    async def _identify_technology_signals(self, industry: str) -> Dict[str, Any]:
        """Identify technology disruption signals"""
        
        signals = {
            "performance_trends": await self._analyze_performance_trends(industry),
            "cost_trends": await self._analyze_cost_trends(industry),
            "accessibility_trends": await self._analyze_accessibility_trends(industry),
            "convergence_signals": await self._detect_convergence_signals(industry),
            "platform_emergence": await self._detect_platform_emergence(industry)
        }
        
        return signals
    
    async def _analyze_performance_trends(self, industry: str) -> Dict[str, float]:
        """Analyze technology performance improvement trends"""
        
        # Industry-specific performance improvement rates
        performance_rates = {
            "artificial_intelligence": {
                "compute_efficiency": 0.35,  # 35% annual improvement
                "model_accuracy": 0.25,
                "inference_speed": 0.40,
                "energy_efficiency": 0.20
            },
            "quantum_computing": {
                "qubit_count": 0.50,  # 50% annual improvement
                "error_rates": -0.30,  # 30% annual reduction
                "coherence_time": 0.25,
                "gate_fidelity": 0.15
            },
            "cloud_computing": {
                "compute_density": 0.20,
                "network_speed": 0.25,
                "storage_capacity": 0.30,
                "cost_efficiency": 0.15
            }
        }
        
        return performance_rates.get(
            industry.lower().replace(" ", "_"),
            {
                "general_performance": 0.15,
                "cost_efficiency": 0.10,
                "user_experience": 0.12,
                "reliability": 0.08
            }
        )
    
    async def _analyze_cost_trends(self, industry: str) -> Dict[str, float]:
        """Analyze cost reduction trends"""
        
        cost_trends = {
            "artificial_intelligence": {
                "compute_costs": -0.25,  # 25% annual reduction
                "data_costs": -0.15,
                "talent_costs": 0.20,    # 20% annual increase
                "infrastructure_costs": -0.18
            },
            "cloud_computing": {
                "storage_costs": -0.20,
                "compute_costs": -0.15,
                "bandwidth_costs": -0.12,
                "management_costs": -0.10
            },
            "renewable_energy": {
                "solar_costs": -0.15,
                "wind_costs": -0.12,
                "battery_costs": -0.20,
                "installation_costs": -0.08
            }
        }
        
        return cost_trends.get(
            industry.lower().replace(" ", "_"),
            {
                "production_costs": -0.08,
                "distribution_costs": -0.05,
                "maintenance_costs": -0.06,
                "labor_costs": 0.03
            }
        )
    
    async def _analyze_accessibility_trends(self, industry: str) -> Dict[str, float]:
        """Analyze technology accessibility improvement trends"""
        
        accessibility_trends = {
            "artificial_intelligence": {
                "api_availability": 0.40,  # 40% increase in accessible APIs
                "no_code_tools": 0.50,
                "educational_resources": 0.35,
                "hardware_requirements": -0.20  # 20% reduction
            },
            "cloud_computing": {
                "service_variety": 0.25,
                "geographic_coverage": 0.20,
                "pricing_flexibility": 0.30,
                "integration_ease": 0.22
            }
        }
        
        return accessibility_trends.get(
            industry.lower().replace(" ", "_"),
            {
                "user_friendliness": 0.15,
                "cost_accessibility": 0.12,
                "technical_barriers": -0.10,
                "market_availability": 0.18
            }
        )
    
    async def _detect_convergence_signals(self, industry: str) -> Dict[str, float]:
        """Detect technology convergence signals"""
        
        convergence_signals = {
            "cross_domain_patents": 0.25,  # 25% of patents cite other domains
            "interdisciplinary_research": 0.30,
            "hybrid_business_models": 0.20,
            "platform_integration": 0.35,
            "talent_mobility": 0.28
        }
        
        # Industry-specific adjustments
        if "artificial_intelligence" in industry.lower():
            convergence_signals["cross_domain_patents"] *= 1.5
            convergence_signals["interdisciplinary_research"] *= 1.4
        
        return convergence_signals
    
    async def _detect_platform_emergence(self, industry: str) -> Dict[str, float]:
        """Detect platform emergence signals"""
        
        platform_signals = {
            "ecosystem_growth": 0.30,  # 30% annual growth in ecosystem
            "developer_adoption": 0.25,
            "third_party_integrations": 0.35,
            "network_effects": 0.40,
            "standardization_progress": 0.20
        }
        
        return platform_signals
    
    async def _assess_regulatory_disruption(self, industry: str) -> Dict[str, Any]:
        """Assess potential regulatory disruption"""
        
        regulatory_risks = {
            "policy_uncertainty": self._assess_policy_uncertainty(industry),
            "compliance_cost_trends": self._assess_compliance_costs(industry),
            "international_coordination": self._assess_international_coordination(industry),
            "enforcement_trends": self._assess_enforcement_trends(industry)
        }
        
        return regulatory_risks
    
    def _assess_policy_uncertainty(self, industry: str) -> float:
        """Assess policy uncertainty level (0-1 scale)"""
        uncertainty_levels = {
            "artificial_intelligence": 0.75,  # High uncertainty
            "cryptocurrency": 0.85,
            "biotechnology": 0.70,
            "social_media": 0.65,
            "cybersecurity": 0.60,
            "cloud_computing": 0.45,
            "e_commerce": 0.40
        }
        
        return uncertainty_levels.get(
            industry.lower().replace(" ", "_"), 0.50
        )
    
    def _assess_compliance_costs(self, industry: str) -> float:
        """Assess compliance cost trends (annual change rate)"""
        cost_trends = {
            "financial_services": 0.15,  # 15% annual increase
            "healthcare": 0.12,
            "artificial_intelligence": 0.25,  # Rapidly increasing
            "social_media": 0.20,
            "cybersecurity": 0.10,
            "cloud_computing": 0.08
        }
        
        return cost_trends.get(
            industry.lower().replace(" ", "_"), 0.05
        )
    
    def _assess_international_coordination(self, industry: str) -> float:
        """Assess international regulatory coordination (0-1 scale)"""
        coordination_levels = {
            "financial_services": 0.70,  # High coordination
            "telecommunications": 0.65,
            "aviation": 0.80,
            "artificial_intelligence": 0.30,  # Low coordination
            "cybersecurity": 0.45,
            "cloud_computing": 0.40
        }
        
        return coordination_levels.get(
            industry.lower().replace(" ", "_"), 0.50
        )
    
    def _assess_enforcement_trends(self, industry: str) -> float:
        """Assess regulatory enforcement trend intensity"""
        enforcement_trends = {
            "social_media": 0.80,  # Increasing enforcement
            "artificial_intelligence": 0.70,
            "financial_services": 0.60,
            "healthcare": 0.55,
            "cybersecurity": 0.65,
            "e_commerce": 0.50
        }
        
        return enforcement_trends.get(
            industry.lower().replace(" ", "_"), 0.45
        )
    
    async def _analyze_competitive_dynamics(self, industry: str) -> Dict[str, Any]:
        """Analyze competitive dynamics for disruption potential"""
        
        dynamics = {
            "incumbent_vulnerability": await self._assess_incumbent_vulnerability(industry),
            "new_entrant_advantages": await self._assess_new_entrant_advantages(industry),
            "competitive_intensity": await self._assess_competitive_intensity(industry),
            "market_consolidation": await self._assess_consolidation_trends(industry)
        }
        
        return dynamics
    
    async def _assess_incumbent_vulnerability(self, industry: str) -> float:
        """Assess incumbent vulnerability to disruption"""
        
        # Factors that make incumbents vulnerable
        vulnerability_factors = {
            "technology_debt": self._assess_technology_debt(industry),
            "customer_satisfaction": 1.0 - 0.70,  # Assume 70% satisfaction
            "innovation_rate": 1.0 - self._assess_innovation_rate(industry, {}),
            "regulatory_burden": self._assess_regulatory_burden(industry),
            "switching_costs": 1.0 - self._assess_switching_costs(industry)
        }
        
        # Weighted vulnerability score
        vulnerability = (
            vulnerability_factors["technology_debt"] * 0.25 +
            vulnerability_factors["customer_satisfaction"] * 0.20 +
            vulnerability_factors["innovation_rate"] * 0.25 +
            vulnerability_factors["regulatory_burden"] * 0.15 +
            vulnerability_factors["switching_costs"] * 0.15
        )
        
        return min(1.0, max(0.0, vulnerability))
    
    async def _assess_new_entrant_advantages(self, industry: str) -> float:
        """Assess advantages available to new entrants"""
        
        advantages = {
            "technology_accessibility": 1.0 - self._assess_switching_costs(industry),
            "capital_efficiency": 0.60,  # Assume modern entrants are more efficient
            "regulatory_flexibility": 1.0 - self._assess_regulatory_burden(industry),
            "talent_availability": 0.70,  # Assume good talent availability
            "market_opportunity": min(1.0, self._get_default_market_data(industry)["growth_rate"] / 0.20)
        }
        
        # Weighted advantage score
        advantage_score = (
            advantages["technology_accessibility"] * 0.25 +
            advantages["capital_efficiency"] * 0.20 +
            advantages["regulatory_flexibility"] * 0.20 +
            advantages["talent_availability"] * 0.15 +
            advantages["market_opportunity"] * 0.20
        )
        
        return min(1.0, max(0.0, advantage_score))
    
    async def _assess_competitive_intensity(self, industry: str) -> float:
        """Assess competitive intensity in industry"""
        
        intensity_factors = {
            "artificial_intelligence": 0.85,
            "cloud_computing": 0.75,
            "cybersecurity": 0.70,
            "social_media": 0.80,
            "e_commerce": 0.75,
            "biotechnology": 0.60,
            "quantum_computing": 0.90
        }
        
        return intensity_factors.get(
            industry.lower().replace(" ", "_"), 0.65
        )
    
    async def _assess_consolidation_trends(self, industry: str) -> float:
        """Assess market consolidation trends"""
        
        consolidation_trends = {
            "cloud_computing": 0.70,  # High consolidation
            "social_media": 0.75,
            "e_commerce": 0.65,
            "artificial_intelligence": 0.55,  # Moderate consolidation
            "cybersecurity": 0.45,  # Lower consolidation
            "biotechnology": 0.40
        }
        
        return consolidation_trends.get(
            industry.lower().replace(" ", "_"), 0.50
        )
    
    async def _generate_disruption_scenarios(
        self,
        industry: str,
        time_horizon: int,
        market_state: Dict[str, Any],
        tech_signals: Dict[str, Any],
        regulatory_risks: Dict[str, Any],
        competitive_dynamics: Dict[str, Any]
    ) -> List[DisruptionPrediction]:
        """Generate comprehensive disruption scenarios"""
        
        disruptions = []
        
        # Technology-driven disruption
        if tech_signals["performance_trends"].get("general_performance", 0.15) > 0.20:
            tech_disruption = DisruptionPrediction(
                industry=industry,
                disruption_type="Technology Performance Breakthrough",
                probability=min(0.80, tech_signals["performance_trends"].get("general_performance", 0.15) * 3),
                time_horizon=max(2, int(time_horizon * 0.4)),
                impact_magnitude=8.0,
                key_drivers=[
                    "Exponential performance improvements",
                    "Cost reduction acceleration",
                    "Accessibility democratization"
                ],
                affected_sectors=self._get_affected_sectors(industry),
                opportunities=[
                    "New business model creation",
                    "Market expansion possibilities",
                    "Competitive advantage establishment"
                ],
                threats=[
                    "Incumbent technology obsolescence",
                    "Skill requirement evolution",
                    "Market structure disruption"
                ],
                recommended_actions=[
                    "Invest in next-generation technology",
                    "Develop new capabilities",
                    "Create strategic partnerships"
                ]
            )
            disruptions.append(tech_disruption)
        
        # Regulatory disruption
        if regulatory_risks["policy_uncertainty"] > 0.60:
            regulatory_disruption = DisruptionPrediction(
                industry=industry,
                disruption_type="Regulatory Paradigm Shift",
                probability=regulatory_risks["policy_uncertainty"],
                time_horizon=max(1, int(time_horizon * 0.3)),
                impact_magnitude=7.0,
                key_drivers=[
                    "Policy uncertainty resolution",
                    "Compliance requirement changes",
                    "International coordination shifts"
                ],
                affected_sectors=self._get_affected_sectors(industry),
                opportunities=[
                    "Regulatory compliance advantage",
                    "Market access expansion",
                    "Competitive moat creation"
                ],
                threats=[
                    "Compliance cost increases",
                    "Market access restrictions",
                    "Business model constraints"
                ],
                recommended_actions=[
                    "Engage with regulators proactively",
                    "Build compliance capabilities",
                    "Develop regulatory expertise"
                ]
            )
            disruptions.append(regulatory_disruption)
        
        # Competitive disruption
        if competitive_dynamics["new_entrant_advantages"] > 0.65:
            competitive_disruption = DisruptionPrediction(
                industry=industry,
                disruption_type="New Entrant Market Disruption",
                probability=competitive_dynamics["new_entrant_advantages"],
                time_horizon=max(3, int(time_horizon * 0.5)),
                impact_magnitude=6.5,
                key_drivers=[
                    "Low barriers to entry",
                    "Technology accessibility",
                    "Capital availability"
                ],
                affected_sectors=self._get_affected_sectors(industry),
                opportunities=[
                    "Partnership opportunities",
                    "Acquisition targets",
                    "Innovation acceleration"
                ],
                threats=[
                    "Market share erosion",
                    "Price pressure",
                    "Talent competition"
                ],
                recommended_actions=[
                    "Strengthen competitive moats",
                    "Accelerate innovation",
                    "Build ecosystem advantages"
                ]
            )
            disruptions.append(competitive_disruption)
        
        # Market maturity disruption
        if market_state["market_maturity"] > 0.70:
            maturity_disruption = DisruptionPrediction(
                industry=industry,
                disruption_type="Market Maturity Transformation",
                probability=market_state["market_maturity"] * 0.8,
                time_horizon=max(5, int(time_horizon * 0.7)),
                impact_magnitude=5.5,
                key_drivers=[
                    "Market saturation",
                    "Customer expectation evolution",
                    "Value chain optimization"
                ],
                affected_sectors=self._get_affected_sectors(industry),
                opportunities=[
                    "Adjacent market expansion",
                    "Value chain integration",
                    "Service transformation"
                ],
                threats=[
                    "Growth rate decline",
                    "Margin pressure",
                    "Commoditization risk"
                ],
                recommended_actions=[
                    "Diversify into adjacent markets",
                    "Transform business model",
                    "Focus on differentiation"
                ]
            )
            disruptions.append(maturity_disruption)
        
        return disruptions
    
    def _get_affected_sectors(self, industry: str) -> List[str]:
        """Get sectors affected by industry disruption"""
        
        sector_map = {
            "artificial_intelligence": [
                "Software development",
                "Professional services",
                "Healthcare",
                "Financial services",
                "Manufacturing",
                "Transportation"
            ],
            "cloud_computing": [
                "Enterprise IT",
                "Software development",
                "Data analytics",
                "Digital media",
                "E-commerce"
            ],
            "cybersecurity": [
                "Financial services",
                "Healthcare",
                "Government",
                "Critical infrastructure",
                "E-commerce"
            ],
            "biotechnology": [
                "Healthcare",
                "Pharmaceuticals",
                "Agriculture",
                "Environmental services",
                "Materials science"
            ]
        }
        
        return sector_map.get(
            industry.lower().replace(" ", "_"),
            ["Technology", "Business services", "Manufacturing"]
        )