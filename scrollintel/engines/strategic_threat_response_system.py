"""
Strategic Threat Assessment and Response System

This module provides comprehensive strategic threat assessment,
response planning, and automated mitigation capabilities.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import asdict
from enum import Enum

from ..models.competitive_intelligence_models import (
    StrategicThreat, ThreatLevel, CompetitiveIntelligence
)


class ResponseStrategy(Enum):
    """Strategic response strategies"""
    DEFENSIVE = "defensive"
    OFFENSIVE = "offensive"
    COLLABORATIVE = "collaborative"
    DISRUPTIVE = "disruptive"
    ADAPTIVE = "adaptive"


class ThreatCategory(Enum):
    """Categories of strategic threats"""
    COMPETITIVE = "competitive"
    TECHNOLOGICAL = "technological"
    REGULATORY = "regulatory"
    MARKET = "market"
    OPERATIONAL = "operational"
    FINANCIAL = "financial"


class StrategicThreatResponseSystem:
    """
    Advanced strategic threat assessment and response system for
    Big Tech CTO operations and competitive intelligence.
    """
    
    def __init__(self):
        self.threat_assessments = {}
        self.response_plans = {}
        self.mitigation_strategies = {}
        self.threat_monitoring = {}
        self.response_history = {}
        self.escalation_rules = {}
        
    async def assess_strategic_threat(
        self, 
        threat_data: Dict[str, Any],
        assessment_depth: str = "comprehensive"
    ) -> StrategicThreat:
        """
        Perform comprehensive strategic threat assessment.
        
        Args:
            threat_data: Raw threat information and indicators
            assessment_depth: Level of assessment ("basic", "detailed", "comprehensive")
            
        Returns:
            Comprehensive strategic threat assessment
        """
        try:
            # Analyze threat characteristics
            threat_analysis = await self._analyze_threat_characteristics(threat_data)
            
            # Assess threat probability and impact
            probability_impact = await self._assess_probability_impact(threat_data, threat_analysis)
            
            # Identify affected business areas
            affected_areas = await self._identify_affected_areas(threat_data, threat_analysis)
            
            # Generate threat indicators and warning signals
            indicators = await self._generate_threat_indicators(threat_data, threat_analysis)
            
            # Develop mitigation strategies
            mitigation_strategies = await self._develop_mitigation_strategies(
                threat_data, threat_analysis, probability_impact
            )
            
            # Create response options
            response_options = await self._create_response_options(
                threat_data, threat_analysis, mitigation_strategies
            )
            
            # Calculate resource requirements
            resource_requirements = await self._calculate_resource_requirements(
                response_options, affected_areas
            )
            
            # Create strategic threat assessment
            threat = StrategicThreat(
                id=f"threat_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                threat_name=threat_data.get('name', 'Unknown Threat'),
                threat_type=threat_analysis['category'],
                source_competitor=threat_data.get('source', 'Unknown'),
                threat_level=self._determine_threat_level(probability_impact),
                probability=probability_impact['probability'],
                potential_impact=probability_impact['impact'],
                time_horizon=threat_analysis['time_horizon'],
                affected_business_areas=affected_areas,
                threat_indicators=indicators['current'],
                early_warning_signals=indicators['early_warning'],
                mitigation_strategies=mitigation_strategies,
                response_options=response_options,
                resource_requirements=resource_requirements,
                success_metrics=await self._define_success_metrics(threat_data, response_options),
                monitoring_frequency=self._determine_monitoring_frequency(probability_impact),
                escalation_triggers=await self._define_escalation_triggers(probability_impact),
                stakeholder_impact=await self._assess_stakeholder_impact(affected_areas),
                identified_date=datetime.now(),
                last_assessment=datetime.now(),
                status="active"
            )
            
            # Store threat assessment
            self.threat_assessments[threat.id] = threat
            
            return threat
            
        except Exception as e:
            raise Exception(f"Threat assessment failed: {str(e)}")
    
    async def _analyze_threat_characteristics(self, threat_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze core characteristics of the threat."""
        # Categorize threat type
        category = self._categorize_threat(threat_data)
        
        # Assess threat sophistication
        sophistication = self._assess_threat_sophistication(threat_data)
        
        # Determine time horizon
        time_horizon = self._determine_time_horizon(threat_data, category)
        
        # Analyze threat vectors
        vectors = self._identify_threat_vectors(threat_data, category)
        
        return {
            'category': category,
            'sophistication': sophistication,
            'time_horizon': time_horizon,
            'vectors': vectors,
            'scope': self._assess_threat_scope(threat_data),
            'persistence': self._assess_threat_persistence(threat_data)
        }
    
    def _categorize_threat(self, threat_data: Dict[str, Any]) -> str:
        """Categorize the type of strategic threat."""
        threat_keywords = {
            'competitive': ['competitor', 'market share', 'pricing', 'product launch'],
            'technological': ['technology', 'innovation', 'disruption', 'breakthrough'],
            'regulatory': ['regulation', 'compliance', 'policy', 'government'],
            'market': ['market', 'demand', 'economic', 'industry'],
            'operational': ['operations', 'supply chain', 'infrastructure', 'talent'],
            'financial': ['funding', 'investment', 'revenue', 'cost']
        }
        
        threat_text = str(threat_data).lower()
        
        for category, keywords in threat_keywords.items():
            if any(keyword in threat_text for keyword in keywords):
                return category
        
        return 'competitive'  # Default category
    
    def _assess_threat_sophistication(self, threat_data: Dict[str, Any]) -> str:
        """Assess the sophistication level of the threat."""
        sophistication_indicators = {
            'basic': ['simple', 'straightforward', 'obvious'],
            'moderate': ['coordinated', 'planned', 'strategic'],
            'advanced': ['sophisticated', 'complex', 'multi-faceted'],
            'expert': ['highly sophisticated', 'expert-level', 'state-of-the-art']
        }
        
        threat_text = str(threat_data).lower()
        
        for level, indicators in sophistication_indicators.items():
            if any(indicator in threat_text for indicator in indicators):
                return level
        
        return 'moderate'  # Default level
    
    def _determine_time_horizon(self, threat_data: Dict[str, Any], category: str) -> str:
        """Determine the time horizon for threat materialization."""
        horizon_map = {
            'competitive': '3-6 months',
            'technological': '6-18 months',
            'regulatory': '12-24 months',
            'market': '6-12 months',
            'operational': '1-3 months',
            'financial': '3-9 months'
        }
        
        return horizon_map.get(category, '6-12 months')
    
    def _identify_threat_vectors(self, threat_data: Dict[str, Any], category: str) -> List[str]:
        """Identify potential threat vectors."""
        vector_map = {
            'competitive': [
                'Product differentiation',
                'Pricing pressure',
                'Market positioning',
                'Customer acquisition',
                'Talent poaching'
            ],
            'technological': [
                'Disruptive innovation',
                'Platform obsolescence',
                'Standards fragmentation',
                'IP infringement',
                'Technology leapfrogging'
            ],
            'regulatory': [
                'Compliance requirements',
                'Market access restrictions',
                'Data governance',
                'Antitrust actions',
                'Industry standards'
            ],
            'market': [
                'Demand shifts',
                'Economic downturn',
                'Industry consolidation',
                'Customer behavior changes',
                'Supply chain disruption'
            ]
        }
        
        return vector_map.get(category, ['Unknown vector'])
    
    def _assess_threat_scope(self, threat_data: Dict[str, Any]) -> str:
        """Assess the scope of threat impact."""
        scope_indicators = {
            'local': ['specific product', 'single market', 'limited'],
            'regional': ['multiple markets', 'business unit', 'regional'],
            'global': ['company-wide', 'all markets', 'enterprise', 'global'],
            'industry': ['industry-wide', 'ecosystem', 'market transformation']
        }
        
        threat_text = str(threat_data).lower()
        
        for scope, indicators in scope_indicators.items():
            if any(indicator in threat_text for indicator in indicators):
                return scope
        
        return 'regional'  # Default scope
    
    def _assess_threat_persistence(self, threat_data: Dict[str, Any]) -> str:
        """Assess how persistent the threat is likely to be."""
        persistence_indicators = {
            'temporary': ['short-term', 'temporary', 'one-time'],
            'recurring': ['periodic', 'cyclical', 'recurring'],
            'persistent': ['ongoing', 'continuous', 'sustained'],
            'permanent': ['permanent', 'structural', 'fundamental']
        }
        
        threat_text = str(threat_data).lower()
        
        for persistence, indicators in persistence_indicators.items():
            if any(indicator in threat_text for indicator in indicators):
                return persistence
        
        return 'persistent'  # Default persistence
    
    async def _assess_probability_impact(
        self, 
        threat_data: Dict[str, Any], 
        threat_analysis: Dict[str, Any]
    ) -> Dict[str, float]:
        """Assess threat probability and potential impact."""
        # Base probability assessment
        base_probability = self._calculate_base_probability(threat_data, threat_analysis)
        
        # Impact assessment
        impact_score = self._calculate_impact_score(threat_data, threat_analysis)
        
        # Adjust for threat characteristics
        probability_adjustments = self._calculate_probability_adjustments(threat_analysis)
        impact_adjustments = self._calculate_impact_adjustments(threat_analysis)
        
        final_probability = min(1.0, max(0.0, base_probability + probability_adjustments))
        final_impact = min(10.0, max(0.0, impact_score + impact_adjustments))
        
        return {
            'probability': final_probability,
            'impact': final_impact,
            'risk_score': final_probability * final_impact,
            'confidence': self._calculate_assessment_confidence(threat_data, threat_analysis)
        }
    
    def _calculate_base_probability(self, threat_data: Dict[str, Any], threat_analysis: Dict[str, Any]) -> float:
        """Calculate base probability of threat materialization."""
        category_probabilities = {
            'competitive': 0.7,
            'technological': 0.5,
            'regulatory': 0.4,
            'market': 0.6,
            'operational': 0.8,
            'financial': 0.6
        }
        
        return category_probabilities.get(threat_analysis['category'], 0.5)
    
    def _calculate_impact_score(self, threat_data: Dict[str, Any], threat_analysis: Dict[str, Any]) -> float:
        """Calculate potential impact score (0-10 scale)."""
        scope_impact = {
            'local': 3.0,
            'regional': 5.0,
            'global': 8.0,
            'industry': 10.0
        }
        
        sophistication_impact = {
            'basic': 1.0,
            'moderate': 2.0,
            'advanced': 3.0,
            'expert': 4.0
        }
        
        base_impact = scope_impact.get(threat_analysis['scope'], 5.0)
        sophistication_bonus = sophistication_impact.get(threat_analysis['sophistication'], 2.0)
        
        return min(10.0, base_impact + sophistication_bonus)
    
    def _calculate_probability_adjustments(self, threat_analysis: Dict[str, Any]) -> float:
        """Calculate probability adjustments based on threat characteristics."""
        adjustments = 0.0
        
        # Sophistication adjustment
        sophistication_adj = {
            'basic': -0.1,
            'moderate': 0.0,
            'advanced': 0.1,
            'expert': 0.2
        }
        adjustments += sophistication_adj.get(threat_analysis['sophistication'], 0.0)
        
        # Persistence adjustment
        persistence_adj = {
            'temporary': -0.2,
            'recurring': -0.1,
            'persistent': 0.1,
            'permanent': 0.2
        }
        adjustments += persistence_adj.get(threat_analysis['persistence'], 0.0)
        
        return adjustments
    
    def _calculate_impact_adjustments(self, threat_analysis: Dict[str, Any]) -> float:
        """Calculate impact adjustments based on threat characteristics."""
        adjustments = 0.0
        
        # Vector count adjustment
        vector_count = len(threat_analysis['vectors'])
        if vector_count > 3:
            adjustments += 1.0
        elif vector_count > 1:
            adjustments += 0.5
        
        return adjustments
    
    def _calculate_assessment_confidence(self, threat_data: Dict[str, Any], threat_analysis: Dict[str, Any]) -> float:
        """Calculate confidence level in the threat assessment."""
        base_confidence = 0.7
        
        # Data quality adjustment
        data_quality = len(threat_data.get('sources', []))
        if data_quality > 5:
            base_confidence += 0.2
        elif data_quality > 2:
            base_confidence += 0.1
        
        # Analysis depth adjustment
        if threat_analysis['sophistication'] in ['advanced', 'expert']:
            base_confidence += 0.1
        
        return min(1.0, base_confidence)
    
    async def _identify_affected_areas(
        self, 
        threat_data: Dict[str, Any], 
        threat_analysis: Dict[str, Any]
    ) -> List[str]:
        """Identify business areas affected by the threat."""
        category_areas = {
            'competitive': [
                'Product Development',
                'Sales and Marketing',
                'Customer Success',
                'Pricing Strategy'
            ],
            'technological': [
                'Engineering',
                'Research and Development',
                'Product Strategy',
                'Infrastructure'
            ],
            'regulatory': [
                'Legal and Compliance',
                'Product Development',
                'Data Management',
                'Operations'
            ],
            'market': [
                'Sales and Marketing',
                'Business Development',
                'Strategic Planning',
                'Finance'
            ],
            'operational': [
                'Operations',
                'Human Resources',
                'Supply Chain',
                'Infrastructure'
            ],
            'financial': [
                'Finance',
                'Strategic Planning',
                'Business Development',
                'Operations'
            ]
        }
        
        base_areas = category_areas.get(threat_analysis['category'], ['Operations'])
        
        # Add scope-based areas
        if threat_analysis['scope'] in ['global', 'industry']:
            base_areas.extend(['Executive Leadership', 'Strategic Planning'])
        
        return list(set(base_areas))  # Remove duplicates
    
    async def _generate_threat_indicators(
        self, 
        threat_data: Dict[str, Any], 
        threat_analysis: Dict[str, Any]
    ) -> Dict[str, List[str]]:
        """Generate current threat indicators and early warning signals."""
        category_indicators = {
            'competitive': {
                'current': [
                    'Competitor product announcements',
                    'Pricing changes in market',
                    'Customer churn increases',
                    'Talent acquisition by competitors'
                ],
                'early_warning': [
                    'Increased competitor R&D spending',
                    'Patent filings in core areas',
                    'Key executive hires',
                    'Partnership announcements'
                ]
            },
            'technological': {
                'current': [
                    'Breakthrough technology announcements',
                    'Industry standard changes',
                    'Platform adoption shifts',
                    'Performance benchmark changes'
                ],
                'early_warning': [
                    'Research paper publications',
                    'Conference presentations',
                    'Open source project activity',
                    'Academic collaboration increases'
                ]
            },
            'regulatory': {
                'current': [
                    'New regulation proposals',
                    'Compliance requirement changes',
                    'Industry investigation announcements',
                    'Policy consultation papers'
                ],
                'early_warning': [
                    'Political rhetoric changes',
                    'Lobbying activity increases',
                    'Industry association statements',
                    'Academic policy research'
                ]
            }
        }
        
        return category_indicators.get(
            threat_analysis['category'], 
            {
                'current': ['General market indicators'],
                'early_warning': ['Industry trend signals']
            }
        )
    
    async def _develop_mitigation_strategies(
        self, 
        threat_data: Dict[str, Any], 
        threat_analysis: Dict[str, Any],
        probability_impact: Dict[str, float]
    ) -> List[str]:
        """Develop comprehensive mitigation strategies."""
        strategies = []
        
        # Category-specific strategies
        category_strategies = {
            'competitive': [
                'Accelerate product development cycles',
                'Strengthen customer relationships',
                'Enhance competitive differentiation',
                'Improve pricing strategy',
                'Expand market presence'
            ],
            'technological': [
                'Invest in R&D capabilities',
                'Build strategic technology partnerships',
                'Acquire complementary technologies',
                'Develop platform extensibility',
                'Create technology standards'
            ],
            'regulatory': [
                'Enhance compliance frameworks',
                'Engage with regulatory bodies',
                'Build policy advocacy capabilities',
                'Implement privacy-by-design',
                'Develop regulatory expertise'
            ],
            'market': [
                'Diversify market presence',
                'Build economic resilience',
                'Strengthen supply chains',
                'Develop alternative revenue streams',
                'Enhance market intelligence'
            ]
        }
        
        strategies.extend(category_strategies.get(threat_analysis['category'], []))
        
        # Risk-level specific strategies
        if probability_impact['risk_score'] > 7.0:
            strategies.extend([
                'Establish crisis response team',
                'Develop contingency plans',
                'Increase monitoring frequency',
                'Prepare stakeholder communications'
            ])
        
        # Scope-specific strategies
        if threat_analysis['scope'] in ['global', 'industry']:
            strategies.extend([
                'Build industry coalitions',
                'Engage with ecosystem partners',
                'Develop thought leadership',
                'Influence industry standards'
            ])
        
        return strategies
    
    async def _create_response_options(
        self, 
        threat_data: Dict[str, Any], 
        threat_analysis: Dict[str, Any],
        mitigation_strategies: List[str]
    ) -> List[Dict[str, Any]]:
        """Create strategic response options."""
        response_options = []
        
        # Defensive response
        defensive_option = {
            'strategy': ResponseStrategy.DEFENSIVE.value,
            'description': 'Protect current market position and capabilities',
            'actions': [
                'Strengthen existing products and services',
                'Improve customer retention programs',
                'Enhance competitive barriers',
                'Optimize operational efficiency'
            ],
            'timeline': '3-6 months',
            'cost_estimate': 5000000,
            'risk_level': 'low',
            'success_probability': 0.75
        }
        response_options.append(defensive_option)
        
        # Offensive response
        offensive_option = {
            'strategy': ResponseStrategy.OFFENSIVE.value,
            'description': 'Proactively counter threat with aggressive market moves',
            'actions': [
                'Launch competing products or services',
                'Aggressive pricing strategies',
                'Market expansion initiatives',
                'Strategic acquisitions'
            ],
            'timeline': '6-12 months',
            'cost_estimate': 15000000,
            'risk_level': 'medium',
            'success_probability': 0.60
        }
        response_options.append(offensive_option)
        
        # Collaborative response
        collaborative_option = {
            'strategy': ResponseStrategy.COLLABORATIVE.value,
            'description': 'Address threat through partnerships and alliances',
            'actions': [
                'Form strategic partnerships',
                'Join industry consortiums',
                'Develop ecosystem alliances',
                'Share threat intelligence'
            ],
            'timeline': '4-8 months',
            'cost_estimate': 8000000,
            'risk_level': 'medium',
            'success_probability': 0.70
        }
        response_options.append(collaborative_option)
        
        # Disruptive response (for high-impact threats)
        if threat_analysis['category'] == 'technological' and threat_analysis['sophistication'] in ['advanced', 'expert']:
            disruptive_option = {
                'strategy': ResponseStrategy.DISRUPTIVE.value,
                'description': 'Counter threat with disruptive innovation',
                'actions': [
                    'Develop breakthrough technologies',
                    'Create new business models',
                    'Disrupt existing value chains',
                    'Redefine market categories'
                ],
                'timeline': '12-24 months',
                'cost_estimate': 25000000,
                'risk_level': 'high',
                'success_probability': 0.45
            }
            response_options.append(disruptive_option)
        
        # Adaptive response
        adaptive_option = {
            'strategy': ResponseStrategy.ADAPTIVE.value,
            'description': 'Continuously adapt strategy based on threat evolution',
            'actions': [
                'Implement agile response framework',
                'Establish continuous monitoring',
                'Build adaptive capabilities',
                'Develop scenario-based planning'
            ],
            'timeline': 'ongoing',
            'cost_estimate': 3000000,
            'risk_level': 'low',
            'success_probability': 0.80
        }
        response_options.append(adaptive_option)
        
        return response_options
    
    async def _calculate_resource_requirements(
        self, 
        response_options: List[Dict[str, Any]], 
        affected_areas: List[str]
    ) -> Dict[str, Any]:
        """Calculate resource requirements for threat response."""
        # Calculate average resource needs across response options
        avg_cost = sum(option['cost_estimate'] for option in response_options) / len(response_options)
        
        # Estimate personnel requirements based on affected areas
        personnel_map = {
            'Engineering': 15,
            'Product Development': 10,
            'Sales and Marketing': 12,
            'Legal and Compliance': 5,
            'Strategic Planning': 8,
            'Operations': 20,
            'Finance': 6,
            'Human Resources': 4
        }
        
        total_personnel = sum(personnel_map.get(area, 5) for area in affected_areas)
        
        return {
            'budget': avg_cost,
            'personnel': total_personnel,
            'timeline': '6-12 months',
            'key_roles': [
                'Threat Response Lead',
                'Strategic Analyst',
                'Technical Lead',
                'Communications Manager'
            ],
            'external_resources': [
                'Consulting services',
                'Legal expertise',
                'Technology partners',
                'Market research'
            ],
            'infrastructure': [
                'Monitoring systems',
                'Communication platforms',
                'Analysis tools',
                'Reporting dashboards'
            ]
        }
    
    async def _define_success_metrics(
        self, 
        threat_data: Dict[str, Any], 
        response_options: List[Dict[str, Any]]
    ) -> List[str]:
        """Define success metrics for threat response."""
        return [
            'Threat impact reduction percentage',
            'Response implementation timeline adherence',
            'Cost efficiency of response measures',
            'Stakeholder satisfaction scores',
            'Market position maintenance',
            'Competitive advantage preservation',
            'Risk mitigation effectiveness',
            'Business continuity metrics'
        ]
    
    def _determine_monitoring_frequency(self, probability_impact: Dict[str, float]) -> str:
        """Determine appropriate monitoring frequency based on risk level."""
        risk_score = probability_impact['risk_score']
        
        if risk_score >= 8.0:
            return 'daily'
        elif risk_score >= 6.0:
            return 'weekly'
        elif risk_score >= 4.0:
            return 'bi-weekly'
        else:
            return 'monthly'
    
    async def _define_escalation_triggers(self, probability_impact: Dict[str, float]) -> List[str]:
        """Define escalation triggers for threat monitoring."""
        base_triggers = [
            'Threat probability increase > 20%',
            'Impact assessment increase > 2 points',
            'New threat vectors identified',
            'Response timeline delays > 30%'
        ]
        
        # Add risk-specific triggers
        if probability_impact['risk_score'] >= 7.0:
            base_triggers.extend([
                'Media coverage increase',
                'Stakeholder concern escalation',
                'Competitive response acceleration'
            ])
        
        return base_triggers
    
    async def _assess_stakeholder_impact(self, affected_areas: List[str]) -> Dict[str, str]:
        """Assess impact on different stakeholder groups."""
        stakeholder_impact = {
            'customers': 'medium',
            'employees': 'medium',
            'investors': 'medium',
            'partners': 'low',
            'regulators': 'low',
            'community': 'low'
        }
        
        # Adjust based on affected areas
        if 'Sales and Marketing' in affected_areas or 'Customer Success' in affected_areas:
            stakeholder_impact['customers'] = 'high'
        
        if 'Human Resources' in affected_areas or 'Operations' in affected_areas:
            stakeholder_impact['employees'] = 'high'
        
        if 'Finance' in affected_areas or 'Strategic Planning' in affected_areas:
            stakeholder_impact['investors'] = 'high'
        
        if 'Legal and Compliance' in affected_areas:
            stakeholder_impact['regulators'] = 'high'
        
        return stakeholder_impact
    
    def _determine_threat_level(self, probability_impact: Dict[str, float]) -> ThreatLevel:
        """Determine threat level based on probability and impact."""
        risk_score = probability_impact['risk_score']
        
        if risk_score >= 8.0:
            return ThreatLevel.CRITICAL
        elif risk_score >= 6.0:
            return ThreatLevel.HIGH
        elif risk_score >= 4.0:
            return ThreatLevel.MEDIUM
        else:
            return ThreatLevel.LOW
    
    async def execute_response_plan(
        self, 
        threat_id: str, 
        selected_strategy: ResponseStrategy
    ) -> Dict[str, Any]:
        """Execute selected response plan for a strategic threat."""
        try:
            threat = self.threat_assessments.get(threat_id)
            if not threat:
                return {'status': 'error', 'message': 'Threat not found'}
            
            # Find selected response option
            selected_option = None
            for option in threat.response_options:
                if option['strategy'] == selected_strategy.value:
                    selected_option = option
                    break
            
            if not selected_option:
                return {'status': 'error', 'message': 'Response strategy not found'}
            
            # Create execution plan
            execution_plan = {
                'threat_id': threat_id,
                'strategy': selected_strategy.value,
                'start_date': datetime.now(),
                'estimated_completion': datetime.now() + timedelta(days=180),
                'actions': selected_option['actions'],
                'milestones': await self._create_execution_milestones(selected_option),
                'resource_allocation': await self._allocate_resources(threat, selected_option),
                'monitoring_plan': await self._create_monitoring_plan(threat),
                'communication_plan': await self._create_communication_plan(threat),
                'risk_mitigation': await self._create_risk_mitigation_plan(selected_option),
                'success_criteria': threat.success_metrics,
                'status': 'initiated'
            }
            
            # Store execution plan
            self.response_plans[threat_id] = execution_plan
            
            # Update threat status
            threat.status = 'response_active'
            
            return {
                'status': 'success',
                'execution_plan': execution_plan,
                'message': f'Response plan initiated for threat {threat_id}'
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Response plan execution failed: {str(e)}'
            }
    
    async def _create_execution_milestones(self, selected_option: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create execution milestones for response plan."""
        milestones = []
        
        # Parse timeline
        timeline_months = 6  # Default
        if 'months' in selected_option['timeline']:
            timeline_parts = selected_option['timeline'].split('-')
            if len(timeline_parts) > 1:
                timeline_months = int(timeline_parts[1].split()[0])
        
        # Create milestones
        milestone_count = min(4, len(selected_option['actions']))
        for i in range(milestone_count):
            milestone_date = datetime.now() + timedelta(days=(timeline_months * 30 // milestone_count) * (i + 1))
            milestones.append({
                'milestone': f"Milestone {i + 1}",
                'description': selected_option['actions'][i] if i < len(selected_option['actions']) else f"Complete phase {i + 1}",
                'due_date': milestone_date,
                'status': 'pending',
                'dependencies': [],
                'success_criteria': f"Successfully implement {selected_option['actions'][i]}" if i < len(selected_option['actions']) else f"Phase {i + 1} objectives met"
            })
        
        return milestones
    
    async def _allocate_resources(self, threat: StrategicThreat, selected_option: Dict[str, Any]) -> Dict[str, Any]:
        """Allocate resources for response execution."""
        return {
            'budget_allocation': {
                'total_budget': selected_option['cost_estimate'],
                'quarterly_allocation': selected_option['cost_estimate'] / 4,
                'contingency_reserve': selected_option['cost_estimate'] * 0.2
            },
            'team_allocation': {
                'core_team_size': threat.resource_requirements.get('personnel', 20) // 2,
                'extended_team_size': threat.resource_requirements.get('personnel', 20),
                'key_roles': threat.resource_requirements.get('key_roles', []),
                'reporting_structure': 'Matrix organization with dedicated threat response lead'
            },
            'timeline_allocation': {
                'planning_phase': '2 weeks',
                'execution_phase': selected_option['timeline'],
                'monitoring_phase': 'ongoing',
                'review_phase': '2 weeks'
            }
        }
    
    async def _create_monitoring_plan(self, threat: StrategicThreat) -> Dict[str, Any]:
        """Create monitoring plan for response execution."""
        return {
            'monitoring_frequency': threat.monitoring_frequency,
            'key_indicators': threat.threat_indicators,
            'escalation_triggers': threat.escalation_triggers,
            'reporting_schedule': {
                'daily_updates': threat.threat_level == ThreatLevel.CRITICAL,
                'weekly_reports': True,
                'monthly_reviews': True,
                'quarterly_assessments': True
            },
            'monitoring_tools': [
                'Threat intelligence platforms',
                'Market monitoring systems',
                'Competitive analysis tools',
                'Performance dashboards'
            ]
        }
    
    async def _create_communication_plan(self, threat: StrategicThreat) -> Dict[str, Any]:
        """Create communication plan for stakeholders."""
        return {
            'stakeholder_groups': list(threat.stakeholder_impact.keys()),
            'communication_frequency': {
                'executive_team': 'weekly',
                'affected_teams': 'bi-weekly',
                'broader_organization': 'monthly',
                'external_stakeholders': 'as_needed'
            },
            'communication_channels': [
                'Executive briefings',
                'Team meetings',
                'Email updates',
                'Dashboard reports',
                'Stakeholder presentations'
            ],
            'key_messages': [
                'Threat identification and assessment',
                'Response strategy and rationale',
                'Progress updates and milestones',
                'Impact mitigation measures',
                'Success metrics and outcomes'
            ]
        }
    
    async def _create_risk_mitigation_plan(self, selected_option: Dict[str, Any]) -> Dict[str, Any]:
        """Create risk mitigation plan for response execution."""
        return {
            'execution_risks': [
                'Timeline delays',
                'Budget overruns',
                'Resource constraints',
                'Technical challenges',
                'Market changes'
            ],
            'mitigation_strategies': [
                'Agile project management',
                'Regular checkpoint reviews',
                'Contingency planning',
                'Risk monitoring',
                'Stakeholder engagement'
            ],
            'contingency_plans': [
                'Alternative response strategies',
                'Resource reallocation options',
                'Timeline adjustment procedures',
                'Escalation protocols'
            ],
            'success_probability': selected_option['success_probability']
        }
    
    async def get_threat_status(self, threat_id: str) -> Dict[str, Any]:
        """Get comprehensive status of a strategic threat."""
        threat = self.threat_assessments.get(threat_id)
        response_plan = self.response_plans.get(threat_id)
        
        if not threat:
            return {'error': 'Threat not found'}
        
        return {
            'threat': asdict(threat),
            'response_plan': response_plan,
            'current_status': threat.status,
            'last_updated': threat.last_assessment,
            'monitoring_active': threat_id in self.threat_monitoring,
            'response_active': threat_id in self.response_plans
        }