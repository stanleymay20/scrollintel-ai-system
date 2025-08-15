"""
Risk Assessment Communication Engine

This engine creates clear risk communication and mitigation strategy presentation,
implements risk visualization and impact communication, and builds risk
communication effectiveness measurement.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging
from enum import Enum
import json

logger = logging.getLogger(__name__)

class RiskCategory(Enum):
    STRATEGIC = "strategic"
    OPERATIONAL = "operational"
    FINANCIAL = "financial"
    TECHNOLOGICAL = "technological"
    REGULATORY = "regulatory"
    MARKET = "market"
    REPUTATIONAL = "reputational"
    CYBERSECURITY = "cybersecurity"

class RiskLevel(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NEGLIGIBLE = "negligible"

class ImpactType(Enum):
    REVENUE_LOSS = "revenue_loss"
    COST_INCREASE = "cost_increase"
    OPERATIONAL_DISRUPTION = "operational_disruption"
    REPUTATION_DAMAGE = "reputation_damage"
    REGULATORY_PENALTY = "regulatory_penalty"
    COMPETITIVE_DISADVANTAGE = "competitive_disadvantage"
    CUSTOMER_LOSS = "customer_loss"
    TALENT_LOSS = "talent_loss"

class CommunicationAudience(Enum):
    BOARD = "board"
    EXECUTIVE_TEAM = "executive_team"
    INVESTORS = "investors"
    REGULATORS = "regulators"
    EMPLOYEES = "employees"
    CUSTOMERS = "customers"
    MEDIA = "media"

@dataclass
class RiskImpact:
    impact_type: ImpactType
    probability: float  # 0-1
    financial_impact: float  # Dollar amount
    timeline: str  # When impact would occur
    description: str
    affected_stakeholders: List[str]

@dataclass
class MitigationStrategy:
    strategy_id: str
    title: str
    description: str
    implementation_cost: float
    implementation_timeline: str
    effectiveness_rating: float  # 0-1
    responsible_party: str
    success_metrics: List[str]
    dependencies: List[str]

@dataclass
class RiskScenario:
    scenario_id: str
    title: str
    description: str
    probability: float
    potential_impacts: List[RiskImpact]
    trigger_events: List[str]
    early_warning_indicators: List[str]

@dataclass
class Risk:
    risk_id: str
    title: str
    description: str
    category: RiskCategory
    risk_level: RiskLevel
    probability: float
    potential_impacts: List[RiskImpact]
    current_controls: List[str]
    mitigation_strategies: List[MitigationStrategy]
    risk_scenarios: List[RiskScenario]
    risk_owner: str
    last_assessed: datetime
    next_review_date: datetime
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class RiskCommunication:
    communication_id: str
    risk_id: str
    audience: CommunicationAudience
    communication_type: str  # presentation, report, dashboard, alert
    key_messages: List[str]
    visual_elements: List[Dict[str, Any]]
    action_items: List[str]
    communication_effectiveness: float
    created_at: datetime = field(default_factory=datetime.now)
    delivered_at: Optional[datetime] = None

@dataclass
class RiskVisualization:
    visualization_id: str
    visualization_type: str  # heatmap, matrix, timeline, dashboard
    risk_data: Dict[str, Any]
    audience: CommunicationAudience
    visual_config: Dict[str, Any]
    effectiveness_score: float
    created_at: datetime = field(default_factory=datetime.now)

class RiskCommunicationEngine:
    """Engine for risk assessment communication and visualization"""
    
    def __init__(self):
        self.risks: List[Risk] = []
        self.communications: List[RiskCommunication] = []
        self.visualizations: List[RiskVisualization] = []
        self.communication_templates = self._initialize_templates()
        self.effectiveness_metrics = {}
    
    def _initialize_templates(self) -> Dict[str, Dict[str, Any]]:
        """Initialize communication templates for different audiences"""
        return {
            CommunicationAudience.BOARD.value: {
                'format': 'executive_summary',
                'key_elements': ['strategic_impact', 'financial_implications', 'mitigation_overview'],
                'tone': 'formal_strategic',
                'detail_level': 'high_level',
                'focus_areas': ['business_impact', 'strategic_implications', 'board_actions_required']
            },
            CommunicationAudience.EXECUTIVE_TEAM.value: {
                'format': 'detailed_briefing',
                'key_elements': ['operational_impact', 'implementation_details', 'resource_requirements'],
                'tone': 'professional_tactical',
                'detail_level': 'detailed',
                'focus_areas': ['operational_implications', 'implementation_plans', 'resource_allocation']
            },
            CommunicationAudience.INVESTORS.value: {
                'format': 'investor_briefing',
                'key_elements': ['financial_impact', 'market_implications', 'competitive_position'],
                'tone': 'confident_transparent',
                'detail_level': 'strategic',
                'focus_areas': ['financial_implications', 'market_position', 'value_protection']
            },
            CommunicationAudience.REGULATORS.value: {
                'format': 'compliance_report',
                'key_elements': ['regulatory_compliance', 'control_measures', 'remediation_plans'],
                'tone': 'formal_compliant',
                'detail_level': 'comprehensive',
                'focus_areas': ['compliance_status', 'control_effectiveness', 'improvement_plans']
            }
        }
    
    def add_risk(self, risk: Risk) -> None:
        """Add a risk to the system"""
        self.risks.append(risk)
        logger.info(f"Added risk: {risk.title}")
    
    def create_risk_communication(
        self,
        risk_id: str,
        audience: CommunicationAudience,
        communication_type: str,
        custom_messages: Optional[List[str]] = None
    ) -> RiskCommunication:
        """Create risk communication for specific audience"""
        
        risk = self._get_risk_by_id(risk_id)
        if not risk:
            raise ValueError(f"Risk {risk_id} not found")
        
        # Generate audience-specific messages
        key_messages = custom_messages or self._generate_key_messages(risk, audience)
        
        # Create visual elements
        visual_elements = self._create_visual_elements(risk, audience)
        
        # Generate action items
        action_items = self._generate_action_items(risk, audience)
        
        communication = RiskCommunication(
            communication_id=f"comm_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            risk_id=risk_id,
            audience=audience,
            communication_type=communication_type,
            key_messages=key_messages,
            visual_elements=visual_elements,
            action_items=action_items,
            communication_effectiveness=0.0  # Will be measured after delivery
        )
        
        self.communications.append(communication)
        logger.info(f"Created risk communication for {audience.value}")
        
        return communication
    
    def _get_risk_by_id(self, risk_id: str) -> Optional[Risk]:
        """Get risk by ID"""
        return next((r for r in self.risks if r.risk_id == risk_id), None)
    
    def _generate_key_messages(self, risk: Risk, audience: CommunicationAudience) -> List[str]:
        """Generate key messages for specific audience"""
        
        template = self.communication_templates[audience.value]
        messages = []
        
        if audience == CommunicationAudience.BOARD:
            messages = self._generate_board_messages(risk)
        elif audience == CommunicationAudience.EXECUTIVE_TEAM:
            messages = self._generate_executive_messages(risk)
        elif audience == CommunicationAudience.INVESTORS:
            messages = self._generate_investor_messages(risk)
        elif audience == CommunicationAudience.REGULATORS:
            messages = self._generate_regulator_messages(risk)
        else:
            messages = self._generate_generic_messages(risk)
        
        return messages
    
    def _generate_board_messages(self, risk: Risk) -> List[str]:
        """Generate board-specific risk messages"""
        messages = []
        
        # Strategic impact message
        total_financial_impact = sum(impact.financial_impact for impact in risk.potential_impacts)
        messages.append(
            f"Strategic Risk Alert: {risk.title} poses a {risk.risk_level.value} risk "
            f"with potential financial impact of ${total_financial_impact:,.0f}."
        )
        
        # Business implications
        key_impacts = [impact.impact_type.value.replace('_', ' ').title() for impact in risk.potential_impacts[:3]]
        messages.append(
            f"Key business implications include: {', '.join(key_impacts)}. "
            f"Probability of occurrence: {risk.probability:.0%}."
        )
        
        # Mitigation overview
        if risk.mitigation_strategies:
            total_mitigation_cost = sum(strategy.implementation_cost for strategy in risk.mitigation_strategies)
            avg_effectiveness = sum(strategy.effectiveness_rating for strategy in risk.mitigation_strategies) / len(risk.mitigation_strategies)
            messages.append(
                f"Proposed mitigation strategies require ${total_mitigation_cost:,.0f} investment "
                f"with {avg_effectiveness:.0%} average effectiveness rating."
            )
        
        # Board action required
        messages.append(
            f"Board action required: Review and approve risk mitigation strategy. "
            f"Risk owner: {risk.risk_owner}. Next review: {risk.next_review_date.strftime('%B %Y')}."
        )
        
        return messages
    
    def _generate_executive_messages(self, risk: Risk) -> List[str]:
        """Generate executive team-specific risk messages"""
        messages = []
        
        # Operational impact
        operational_impacts = [i for i in risk.potential_impacts if i.impact_type == ImpactType.OPERATIONAL_DISRUPTION]
        if operational_impacts:
            messages.append(
                f"Operational Risk: {risk.title} could disrupt operations with "
                f"{operational_impacts[0].probability:.0%} probability."
            )
        
        # Implementation details
        if risk.mitigation_strategies:
            strategy_timelines = [s.implementation_timeline for s in risk.mitigation_strategies]
            messages.append(
                f"Mitigation implementation timelines: {', '.join(strategy_timelines)}. "
                f"Coordination required across {len(set(s.responsible_party for s in risk.mitigation_strategies))} departments."
            )
        
        # Resource requirements
        total_cost = sum(strategy.implementation_cost for strategy in risk.mitigation_strategies)
        messages.append(
            f"Total resource requirement: ${total_cost:,.0f}. "
            f"Implementation dependencies: {len(set().union(*(s.dependencies for s in risk.mitigation_strategies)))} items."
        )
        
        return messages
    
    def _generate_investor_messages(self, risk: Risk) -> List[str]:
        """Generate investor-specific risk messages"""
        messages = []
        
        # Financial impact focus
        revenue_impacts = [i for i in risk.potential_impacts if i.impact_type == ImpactType.REVENUE_LOSS]
        if revenue_impacts:
            total_revenue_risk = sum(i.financial_impact for i in revenue_impacts)
            messages.append(
                f"Financial Risk Disclosure: {risk.title} presents potential revenue impact "
                f"of ${total_revenue_risk:,.0f} with {risk.probability:.0%} probability."
            )
        
        # Market implications
        market_impacts = [i for i in risk.potential_impacts if i.impact_type == ImpactType.COMPETITIVE_DISADVANTAGE]
        if market_impacts:
            messages.append(
                f"Market Position: Risk may affect competitive positioning. "
                f"Mitigation strategies in place to protect market share."
            )
        
        # Value protection
        if risk.mitigation_strategies:
            avg_effectiveness = sum(s.effectiveness_rating for s in risk.mitigation_strategies) / len(risk.mitigation_strategies)
            messages.append(
                f"Value Protection: Comprehensive mitigation plan with {avg_effectiveness:.0%} "
                f"effectiveness rating to protect shareholder value."
            )
        
        return messages
    
    def _generate_regulator_messages(self, risk: Risk) -> List[str]:
        """Generate regulator-specific risk messages"""
        messages = []
        
        # Compliance status
        if risk.category == RiskCategory.REGULATORY:
            messages.append(
                f"Regulatory Compliance: {risk.title} identified and assessed. "
                f"Current control measures: {len(risk.current_controls)} controls in place."
            )
        
        # Control effectiveness
        messages.append(
            f"Risk Management: {risk.risk_level.value.title()} risk under active management. "
            f"Risk owner assigned: {risk.risk_owner}. Regular review cycle established."
        )
        
        # Improvement plans
        if risk.mitigation_strategies:
            messages.append(
                f"Improvement Plan: {len(risk.mitigation_strategies)} mitigation strategies "
                f"identified for implementation. Compliance enhancement in progress."
            )
        
        return messages
    
    def _generate_generic_messages(self, risk: Risk) -> List[str]:
        """Generate generic risk messages"""
        return [
            f"Risk Identified: {risk.title} - {risk.risk_level.value} level risk",
            f"Category: {risk.category.value.title()} risk with {risk.probability:.0%} probability",
            f"Mitigation: {len(risk.mitigation_strategies)} strategies identified for implementation"
        ]
    
    def _create_visual_elements(self, risk: Risk, audience: CommunicationAudience) -> List[Dict[str, Any]]:
        """Create visual elements for risk communication"""
        visual_elements = []
        
        # Risk level indicator
        visual_elements.append({
            'type': 'risk_indicator',
            'data': {
                'risk_level': risk.risk_level.value,
                'probability': risk.probability,
                'category': risk.category.value
            },
            'config': {
                'color_scheme': self._get_risk_color_scheme(risk.risk_level),
                'size': 'large' if audience == CommunicationAudience.BOARD else 'medium'
            }
        })
        
        # Impact visualization
        if risk.potential_impacts:
            visual_elements.append({
                'type': 'impact_chart',
                'data': {
                    'impacts': [
                        {
                            'type': impact.impact_type.value,
                            'probability': impact.probability,
                            'financial_impact': impact.financial_impact,
                            'timeline': impact.timeline
                        }
                        for impact in risk.potential_impacts
                    ]
                },
                'config': {
                    'chart_type': 'bubble_chart',
                    'x_axis': 'probability',
                    'y_axis': 'financial_impact',
                    'size': 'timeline'
                }
            })
        
        # Mitigation timeline
        if risk.mitigation_strategies:
            visual_elements.append({
                'type': 'mitigation_timeline',
                'data': {
                    'strategies': [
                        {
                            'title': strategy.title,
                            'timeline': strategy.implementation_timeline,
                            'cost': strategy.implementation_cost,
                            'effectiveness': strategy.effectiveness_rating
                        }
                        for strategy in risk.mitigation_strategies
                    ]
                },
                'config': {
                    'timeline_type': 'gantt',
                    'show_costs': audience in [CommunicationAudience.BOARD, CommunicationAudience.EXECUTIVE_TEAM],
                    'show_effectiveness': True
                }
            })
        
        # Risk scenarios (for detailed audiences)
        if audience in [CommunicationAudience.EXECUTIVE_TEAM, CommunicationAudience.REGULATORS] and risk.risk_scenarios:
            visual_elements.append({
                'type': 'scenario_analysis',
                'data': {
                    'scenarios': [
                        {
                            'title': scenario.title,
                            'probability': scenario.probability,
                            'impacts': len(scenario.potential_impacts),
                            'triggers': scenario.trigger_events
                        }
                        for scenario in risk.risk_scenarios
                    ]
                },
                'config': {
                    'visualization_type': 'tree_diagram',
                    'show_probabilities': True,
                    'show_triggers': True
                }
            })
        
        return visual_elements
    
    def _get_risk_color_scheme(self, risk_level: RiskLevel) -> Dict[str, str]:
        """Get color scheme for risk level"""
        color_schemes = {
            RiskLevel.CRITICAL: {'primary': '#DC2626', 'secondary': '#FEE2E2'},
            RiskLevel.HIGH: {'primary': '#EA580C', 'secondary': '#FED7AA'},
            RiskLevel.MEDIUM: {'primary': '#D97706', 'secondary': '#FEF3C7'},
            RiskLevel.LOW: {'primary': '#65A30D', 'secondary': '#ECFCCB'},
            RiskLevel.NEGLIGIBLE: {'primary': '#059669', 'secondary': '#D1FAE5'}
        }
        return color_schemes.get(risk_level, color_schemes[RiskLevel.MEDIUM])
    
    def _generate_action_items(self, risk: Risk, audience: CommunicationAudience) -> List[str]:
        """Generate action items for specific audience"""
        action_items = []
        
        if audience == CommunicationAudience.BOARD:
            action_items = [
                "Review and approve risk mitigation strategy",
                "Allocate budget for mitigation implementation",
                "Assign executive sponsor for risk management",
                f"Schedule follow-up review for {risk.next_review_date.strftime('%B %Y')}"
            ]
        elif audience == CommunicationAudience.EXECUTIVE_TEAM:
            action_items = [
                "Implement approved mitigation strategies",
                "Coordinate cross-functional risk response",
                "Monitor risk indicators and early warnings",
                "Report progress to board monthly"
            ]
        elif audience == CommunicationAudience.INVESTORS:
            action_items = [
                "Monitor risk mitigation progress",
                "Assess impact on financial projections",
                "Communicate updates in quarterly reports"
            ]
        elif audience == CommunicationAudience.REGULATORS:
            action_items = [
                "Submit compliance documentation",
                "Implement required control measures",
                "Schedule regulatory review meetings",
                "Provide progress updates quarterly"
            ]
        
        return action_items
    
    def create_risk_visualization(
        self,
        visualization_type: str,
        risk_data: Dict[str, Any],
        audience: CommunicationAudience,
        custom_config: Optional[Dict[str, Any]] = None
    ) -> RiskVisualization:
        """Create risk visualization"""
        
        # Default configurations for different visualization types
        default_configs = {
            'heatmap': {
                'x_axis': 'probability',
                'y_axis': 'impact',
                'color_scale': 'risk_levels',
                'grid_size': '5x5'
            },
            'matrix': {
                'dimensions': ['probability', 'impact'],
                'quadrants': ['low_low', 'low_high', 'high_low', 'high_high'],
                'risk_placement': 'automatic'
            },
            'timeline': {
                'time_axis': 'implementation_timeline',
                'events': 'mitigation_milestones',
                'show_dependencies': True
            },
            'dashboard': {
                'layout': 'grid',
                'widgets': ['risk_summary', 'top_risks', 'mitigation_progress'],
                'refresh_rate': 'real_time'
            }
        }
        
        visual_config = custom_config or default_configs.get(visualization_type, {})
        
        # Calculate effectiveness score based on audience and visualization type
        effectiveness_score = self._calculate_visualization_effectiveness(
            visualization_type, audience, risk_data
        )
        
        visualization = RiskVisualization(
            visualization_id=f"viz_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            visualization_type=visualization_type,
            risk_data=risk_data,
            audience=audience,
            visual_config=visual_config,
            effectiveness_score=effectiveness_score
        )
        
        self.visualizations.append(visualization)
        logger.info(f"Created {visualization_type} visualization for {audience.value}")
        
        return visualization
    
    def _calculate_visualization_effectiveness(
        self,
        viz_type: str,
        audience: CommunicationAudience,
        risk_data: Dict[str, Any]
    ) -> float:
        """Calculate visualization effectiveness score"""
        
        # Base effectiveness by visualization type and audience
        effectiveness_matrix = {
            'heatmap': {
                CommunicationAudience.BOARD: 0.9,
                CommunicationAudience.EXECUTIVE_TEAM: 0.8,
                CommunicationAudience.INVESTORS: 0.85,
                CommunicationAudience.REGULATORS: 0.7
            },
            'matrix': {
                CommunicationAudience.BOARD: 0.85,
                CommunicationAudience.EXECUTIVE_TEAM: 0.9,
                CommunicationAudience.INVESTORS: 0.8,
                CommunicationAudience.REGULATORS: 0.85
            },
            'timeline': {
                CommunicationAudience.BOARD: 0.7,
                CommunicationAudience.EXECUTIVE_TEAM: 0.95,
                CommunicationAudience.INVESTORS: 0.75,
                CommunicationAudience.REGULATORS: 0.9
            },
            'dashboard': {
                CommunicationAudience.BOARD: 0.8,
                CommunicationAudience.EXECUTIVE_TEAM: 0.85,
                CommunicationAudience.INVESTORS: 0.9,
                CommunicationAudience.REGULATORS: 0.8
            }
        }
        
        base_score = effectiveness_matrix.get(viz_type, {}).get(audience, 0.7)
        
        # Adjust based on data quality and completeness
        data_quality_factor = min(len(risk_data) / 10, 1.0)  # More data generally better
        
        # Adjust based on risk complexity
        complexity_factor = 1.0
        if 'risks' in risk_data:
            num_risks = len(risk_data['risks'])
            if num_risks > 20:
                complexity_factor = 0.9  # Too many risks reduce effectiveness
            elif num_risks < 3:
                complexity_factor = 0.8  # Too few risks also reduce effectiveness
        
        final_score = base_score * data_quality_factor * complexity_factor
        return min(final_score, 1.0)
    
    def measure_communication_effectiveness(
        self,
        communication_id: str,
        feedback_data: Dict[str, Any]
    ) -> float:
        """Measure effectiveness of risk communication"""
        
        communication = next(
            (c for c in self.communications if c.communication_id == communication_id),
            None
        )
        
        if not communication:
            raise ValueError(f"Communication {communication_id} not found")
        
        # Calculate effectiveness based on feedback
        effectiveness_factors = {}
        
        # Clarity score (1-5 scale)
        if 'clarity_rating' in feedback_data:
            effectiveness_factors['clarity'] = feedback_data['clarity_rating'] / 5.0
        
        # Usefulness score (1-5 scale)
        if 'usefulness_rating' in feedback_data:
            effectiveness_factors['usefulness'] = feedback_data['usefulness_rating'] / 5.0
        
        # Action taken (binary)
        if 'action_taken' in feedback_data:
            effectiveness_factors['action_response'] = 1.0 if feedback_data['action_taken'] else 0.3
        
        # Understanding level (1-5 scale)
        if 'understanding_rating' in feedback_data:
            effectiveness_factors['understanding'] = feedback_data['understanding_rating'] / 5.0
        
        # Time to decision (faster is better)
        if 'decision_time_hours' in feedback_data:
            decision_time = feedback_data['decision_time_hours']
            # Normalize: 24 hours = 1.0, 1 hour = 0.5, immediate = 0.0
            time_factor = max(0.0, 1.0 - (decision_time / 24.0))
            effectiveness_factors['response_time'] = time_factor
        
        # Calculate weighted average
        weights = {
            'clarity': 0.25,
            'usefulness': 0.25,
            'action_response': 0.2,
            'understanding': 0.2,
            'response_time': 0.1
        }
        
        total_score = 0.0
        total_weight = 0.0
        
        for factor, score in effectiveness_factors.items():
            if factor in weights:
                total_score += score * weights[factor]
                total_weight += weights[factor]
        
        # Default to 0.5 if no feedback available
        effectiveness_score = total_score / total_weight if total_weight > 0 else 0.5
        
        # Update communication record
        communication.communication_effectiveness = effectiveness_score
        
        # Store in metrics for analysis
        self.effectiveness_metrics[communication_id] = {
            'score': effectiveness_score,
            'factors': effectiveness_factors,
            'feedback_date': datetime.now(),
            'audience': communication.audience.value
        }
        
        logger.info(f"Communication effectiveness measured: {effectiveness_score:.2f}")
        
        return effectiveness_score
    
    def get_risk_communication_analytics(self) -> Dict[str, Any]:
        """Get analytics on risk communication effectiveness"""
        
        if not self.communications:
            return {
                'total_communications': 0,
                'average_effectiveness': 0.0,
                'effectiveness_by_audience': {},
                'effectiveness_by_type': {},
                'improvement_recommendations': []
            }
        
        # Overall statistics
        total_communications = len(self.communications)
        communications_with_scores = [c for c in self.communications if c.communication_effectiveness > 0]
        
        if communications_with_scores:
            average_effectiveness = sum(c.communication_effectiveness for c in communications_with_scores) / len(communications_with_scores)
        else:
            average_effectiveness = 0.0
        
        # Effectiveness by audience
        effectiveness_by_audience = {}
        for audience in CommunicationAudience:
            audience_comms = [c for c in communications_with_scores if c.audience == audience]
            if audience_comms:
                avg_score = sum(c.communication_effectiveness for c in audience_comms) / len(audience_comms)
                effectiveness_by_audience[audience.value] = {
                    'average_score': avg_score,
                    'total_communications': len(audience_comms),
                    'score_range': {
                        'min': min(c.communication_effectiveness for c in audience_comms),
                        'max': max(c.communication_effectiveness for c in audience_comms)
                    }
                }
        
        # Effectiveness by communication type
        effectiveness_by_type = {}
        comm_types = set(c.communication_type for c in communications_with_scores)
        for comm_type in comm_types:
            type_comms = [c for c in communications_with_scores if c.communication_type == comm_type]
            avg_score = sum(c.communication_effectiveness for c in type_comms) / len(type_comms)
            effectiveness_by_type[comm_type] = {
                'average_score': avg_score,
                'total_communications': len(type_comms)
            }
        
        # Generate improvement recommendations
        improvement_recommendations = self._generate_improvement_recommendations(
            effectiveness_by_audience, effectiveness_by_type, average_effectiveness
        )
        
        return {
            'total_communications': total_communications,
            'communications_measured': len(communications_with_scores),
            'average_effectiveness': average_effectiveness,
            'effectiveness_by_audience': effectiveness_by_audience,
            'effectiveness_by_type': effectiveness_by_type,
            'improvement_recommendations': improvement_recommendations,
            'top_performing_communications': self._get_top_performing_communications(),
            'communication_trends': self._analyze_communication_trends()
        }
    
    def _generate_improvement_recommendations(
        self,
        by_audience: Dict[str, Any],
        by_type: Dict[str, Any],
        overall_avg: float
    ) -> List[str]:
        """Generate recommendations for improving communication effectiveness"""
        
        recommendations = []
        
        # Overall effectiveness recommendations
        if overall_avg < 0.6:
            recommendations.append("Overall communication effectiveness is below target. Consider comprehensive review of communication strategies.")
        elif overall_avg < 0.7:
            recommendations.append("Communication effectiveness is moderate. Focus on clarity and actionability improvements.")
        
        # Audience-specific recommendations
        for audience, data in by_audience.items():
            if data['average_score'] < 0.6:
                recommendations.append(f"Improve {audience} communications - current effectiveness below 60%")
            elif data['score_range']['max'] - data['score_range']['min'] > 0.4:
                recommendations.append(f"Standardize {audience} communication quality - high variability detected")
        
        # Type-specific recommendations
        lowest_type = min(by_type.items(), key=lambda x: x[1]['average_score'], default=(None, {'average_score': 1.0}))
        if lowest_type[0] and lowest_type[1]['average_score'] < 0.7:
            recommendations.append(f"Improve {lowest_type[0]} communication format - lowest performing type")
        
        # Visual recommendations
        viz_effectiveness = sum(v.effectiveness_score for v in self.visualizations) / len(self.visualizations) if self.visualizations else 0
        if viz_effectiveness < 0.7:
            recommendations.append("Enhance risk visualizations - current visual effectiveness below target")
        
        return recommendations
    
    def _get_top_performing_communications(self) -> List[Dict[str, Any]]:
        """Get top performing communications for benchmarking"""
        
        scored_comms = [c for c in self.communications if c.communication_effectiveness > 0]
        top_comms = sorted(scored_comms, key=lambda x: x.communication_effectiveness, reverse=True)[:5]
        
        return [
            {
                'communication_id': comm.communication_id,
                'audience': comm.audience.value,
                'type': comm.communication_type,
                'effectiveness_score': comm.communication_effectiveness,
                'key_messages_count': len(comm.key_messages),
                'visual_elements_count': len(comm.visual_elements)
            }
            for comm in top_comms
        ]
    
    def _analyze_communication_trends(self) -> Dict[str, Any]:
        """Analyze trends in communication effectiveness over time"""
        
        if len(self.communications) < 2:
            return {'trend': 'insufficient_data', 'analysis': 'Need more communications for trend analysis'}
        
        # Sort communications by creation date
        sorted_comms = sorted(self.communications, key=lambda x: x.created_at)
        scored_comms = [c for c in sorted_comms if c.communication_effectiveness > 0]
        
        if len(scored_comms) < 2:
            return {'trend': 'insufficient_scored_data', 'analysis': 'Need more scored communications for trend analysis'}
        
        # Calculate trend
        recent_half = scored_comms[len(scored_comms)//2:]
        earlier_half = scored_comms[:len(scored_comms)//2]
        
        recent_avg = sum(c.communication_effectiveness for c in recent_half) / len(recent_half)
        earlier_avg = sum(c.communication_effectiveness for c in earlier_half) / len(earlier_half)
        
        trend_direction = 'improving' if recent_avg > earlier_avg else 'declining' if recent_avg < earlier_avg else 'stable'
        trend_magnitude = abs(recent_avg - earlier_avg)
        
        return {
            'trend': trend_direction,
            'magnitude': trend_magnitude,
            'recent_average': recent_avg,
            'earlier_average': earlier_avg,
            'analysis': f"Communication effectiveness is {trend_direction} with {trend_magnitude:.2f} change"
        }
    
    def generate_risk_communication_report(
        self,
        risk_id: str,
        audience: CommunicationAudience
    ) -> Dict[str, Any]:
        """Generate comprehensive risk communication report"""
        
        risk = self._get_risk_by_id(risk_id)
        if not risk:
            raise ValueError(f"Risk {risk_id} not found")
        
        # Create communication if not exists
        existing_comm = next(
            (c for c in self.communications if c.risk_id == risk_id and c.audience == audience),
            None
        )
        
        if not existing_comm:
            existing_comm = self.create_risk_communication(
                risk_id, audience, 'report'
            )
        
        # Generate comprehensive report
        report = {
            'risk_overview': {
                'title': risk.title,
                'category': risk.category.value,
                'level': risk.risk_level.value,
                'probability': f"{risk.probability:.0%}",
                'owner': risk.risk_owner,
                'last_assessed': risk.last_assessed.strftime('%Y-%m-%d')
            },
            'impact_analysis': [
                {
                    'type': impact.impact_type.value,
                    'probability': f"{impact.probability:.0%}",
                    'financial_impact': f"${impact.financial_impact:,.0f}",
                    'timeline': impact.timeline,
                    'affected_stakeholders': impact.affected_stakeholders
                }
                for impact in risk.potential_impacts
            ],
            'mitigation_strategies': [
                {
                    'title': strategy.title,
                    'cost': f"${strategy.implementation_cost:,.0f}",
                    'timeline': strategy.implementation_timeline,
                    'effectiveness': f"{strategy.effectiveness_rating:.0%}",
                    'responsible_party': strategy.responsible_party
                }
                for strategy in risk.mitigation_strategies
            ],
            'communication_details': {
                'audience': audience.value,
                'key_messages': existing_comm.key_messages,
                'action_items': existing_comm.action_items,
                'visual_elements': len(existing_comm.visual_elements),
                'effectiveness_score': existing_comm.communication_effectiveness
            },
            'recommendations': self._generate_report_recommendations(risk, audience),
            'next_steps': self._generate_next_steps(risk, audience)
        }
        
        return report
    
    def _generate_report_recommendations(self, risk: Risk, audience: CommunicationAudience) -> List[str]:
        """Generate recommendations for risk report"""
        recommendations = []
        
        # Risk level recommendations
        if risk.risk_level in [RiskLevel.CRITICAL, RiskLevel.HIGH]:
            recommendations.append("Immediate attention required - escalate to senior leadership")
            recommendations.append("Consider emergency response protocols")
        
        # Mitigation recommendations
        if not risk.mitigation_strategies:
            recommendations.append("Develop comprehensive mitigation strategies")
        elif len(risk.mitigation_strategies) < 2:
            recommendations.append("Consider additional mitigation options for redundancy")
        
        # Communication recommendations
        if audience == CommunicationAudience.BOARD:
            recommendations.append("Schedule board presentation for risk review")
            recommendations.append("Prepare executive summary for board materials")
        
        return recommendations
    
    def _generate_next_steps(self, risk: Risk, audience: CommunicationAudience) -> List[str]:
        """Generate next steps for risk management"""
        next_steps = []
        
        # Standard next steps based on audience
        if audience == CommunicationAudience.BOARD:
            next_steps.extend([
                "Review and approve mitigation budget",
                "Assign executive sponsor",
                "Schedule quarterly risk reviews"
            ])
        elif audience == CommunicationAudience.EXECUTIVE_TEAM:
            next_steps.extend([
                "Implement approved mitigation strategies",
                "Establish monitoring protocols",
                "Coordinate cross-functional response"
            ])
        
        # Risk-specific next steps
        if risk.risk_level == RiskLevel.CRITICAL:
            next_steps.insert(0, "Activate crisis management protocols")
        
        if not risk.current_controls:
            next_steps.append("Implement immediate control measures")
        
        return next_steps