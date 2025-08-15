"""
Crisis Detection and Assessment Engine

This engine provides comprehensive crisis detection, classification, impact assessment,
and escalation management capabilities for ScrollIntel's crisis leadership system.
"""

from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import asyncio
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class CrisisType(Enum):
    """Types of crises that can be detected and managed"""
    SYSTEM_OUTAGE = "system_outage"
    SECURITY_BREACH = "security_breach"
    DATA_LOSS = "data_loss"
    FINANCIAL_CRISIS = "financial_crisis"
    REGULATORY_VIOLATION = "regulatory_violation"
    REPUTATION_DAMAGE = "reputation_damage"
    PERSONNEL_CRISIS = "personnel_crisis"
    SUPPLY_CHAIN_DISRUPTION = "supply_chain_disruption"
    MARKET_VOLATILITY = "market_volatility"
    NATURAL_DISASTER = "natural_disaster"
    CYBER_ATTACK = "cyber_attack"
    LEGAL_ISSUE = "legal_issue"


class SeverityLevel(Enum):
    """Crisis severity levels for prioritization and response"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4
    CATASTROPHIC = 5


class CrisisStatus(Enum):
    """Current status of crisis management"""
    DETECTED = "detected"
    ASSESSED = "assessed"
    ESCALATED = "escalated"
    RESPONDING = "responding"
    CONTAINED = "contained"
    RESOLVED = "resolved"


@dataclass
class Signal:
    """Individual signal that may indicate potential crisis"""
    source: str
    signal_type: str
    value: Any
    timestamp: datetime
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PotentialCrisis:
    """Potential crisis identified by early warning system"""
    id: str
    crisis_type: CrisisType
    probability: float
    signals: List[Signal]
    predicted_impact: str
    time_to_crisis: Optional[timedelta]
    confidence_score: float


@dataclass
class CrisisClassification:
    """Classification result for identified crisis"""
    crisis_type: CrisisType
    severity_level: SeverityLevel
    confidence: float
    sub_categories: List[str]
    related_crises: List[str]
    classification_rationale: str


@dataclass
class ImpactAssessment:
    """Comprehensive impact assessment of crisis"""
    financial_impact: Dict[str, float]
    operational_impact: Dict[str, str]
    reputation_impact: Dict[str, float]
    stakeholder_impact: Dict[str, List[str]]
    timeline_impact: Dict[str, timedelta]
    recovery_estimate: timedelta
    cascading_risks: List[str]
    mitigation_urgency: SeverityLevel


@dataclass
class Crisis:
    """Complete crisis representation"""
    id: str
    crisis_type: CrisisType
    severity_level: SeverityLevel
    start_time: datetime
    affected_areas: List[str]
    stakeholders_impacted: List[str]
    current_status: CrisisStatus
    signals: List[Signal]
    classification: Optional[CrisisClassification] = None
    impact_assessment: Optional[ImpactAssessment] = None
    escalation_history: List[Dict[str, Any]] = field(default_factory=list)
    resolution_time: Optional[datetime] = None


class EarlyWarningSystem:
    """Proactive crisis identification system"""
    
    def __init__(self):
        self.signal_sources = {}
        self.pattern_detectors = {}
        self.threshold_monitors = {}
        self.anomaly_detectors = {}
    
    async def monitor_signals(self) -> List[Signal]:
        """Continuously monitor various signal sources for crisis indicators"""
        signals = []
        
        # System health signals
        system_signals = await self._monitor_system_health()
        signals.extend(system_signals)
        
        # Security signals
        security_signals = await self._monitor_security_indicators()
        signals.extend(security_signals)
        
        # Financial signals
        financial_signals = await self._monitor_financial_indicators()
        signals.extend(financial_signals)
        
        # Market signals
        market_signals = await self._monitor_market_conditions()
        signals.extend(market_signals)
        
        # Operational signals
        operational_signals = await self._monitor_operational_metrics()
        signals.extend(operational_signals)
        
        return signals
    
    async def _monitor_system_health(self) -> List[Signal]:
        """Monitor system health indicators"""
        signals = []
        
        # CPU usage monitoring
        cpu_usage = await self._get_cpu_usage()
        if cpu_usage > 90:
            signals.append(Signal(
                source="system_monitor",
                signal_type="high_cpu_usage",
                value=cpu_usage,
                timestamp=datetime.now(),
                confidence=0.9,
                metadata={"threshold": 90, "duration": "5min"}
            ))
        
        # Memory usage monitoring
        memory_usage = await self._get_memory_usage()
        if memory_usage > 85:
            signals.append(Signal(
                source="system_monitor",
                signal_type="high_memory_usage",
                value=memory_usage,
                timestamp=datetime.now(),
                confidence=0.85,
                metadata={"threshold": 85}
            ))
        
        # Error rate monitoring
        error_rate = await self._get_error_rate()
        if error_rate > 5:  # 5% error rate
            signals.append(Signal(
                source="application_monitor",
                signal_type="high_error_rate",
                value=error_rate,
                timestamp=datetime.now(),
                confidence=0.95,
                metadata={"threshold": 5, "window": "1hour"}
            ))
        
        return signals
    
    async def _monitor_security_indicators(self) -> List[Signal]:
        """Monitor security-related indicators"""
        signals = []
        
        # Failed login attempts
        failed_logins = await self._get_failed_login_count()
        if failed_logins > 100:  # 100 failed logins in last hour
            signals.append(Signal(
                source="security_monitor",
                signal_type="high_failed_logins",
                value=failed_logins,
                timestamp=datetime.now(),
                confidence=0.8,
                metadata={"window": "1hour", "threshold": 100}
            ))
        
        # Unusual network traffic
        network_anomaly = await self._detect_network_anomalies()
        if network_anomaly:
            signals.append(Signal(
                source="network_monitor",
                signal_type="network_anomaly",
                value=network_anomaly,
                timestamp=datetime.now(),
                confidence=0.75,
                metadata={"anomaly_type": "traffic_spike"}
            ))
        
        return signals
    
    async def _monitor_financial_indicators(self) -> List[Signal]:
        """Monitor financial health indicators"""
        signals = []
        
        # Revenue drop
        revenue_change = await self._get_revenue_change()
        if revenue_change < -20:  # 20% revenue drop
            signals.append(Signal(
                source="financial_monitor",
                signal_type="revenue_drop",
                value=revenue_change,
                timestamp=datetime.now(),
                confidence=0.9,
                metadata={"threshold": -20, "period": "week"}
            ))
        
        # Cash flow issues
        cash_flow = await self._get_cash_flow_status()
        if cash_flow < 30:  # Less than 30 days runway
            signals.append(Signal(
                source="financial_monitor",
                signal_type="low_cash_flow",
                value=cash_flow,
                timestamp=datetime.now(),
                confidence=0.95,
                metadata={"runway_days": cash_flow}
            ))
        
        return signals
    
    async def _monitor_market_conditions(self) -> List[Signal]:
        """Monitor external market conditions"""
        signals = []
        
        # Market volatility
        volatility = await self._get_market_volatility()
        if volatility > 0.3:  # High volatility threshold
            signals.append(Signal(
                source="market_monitor",
                signal_type="high_volatility",
                value=volatility,
                timestamp=datetime.now(),
                confidence=0.7,
                metadata={"threshold": 0.3}
            ))
        
        return signals
    
    async def _monitor_operational_metrics(self) -> List[Signal]:
        """Monitor operational performance metrics"""
        signals = []
        
        # Customer satisfaction drop
        satisfaction_score = await self._get_customer_satisfaction()
        if satisfaction_score < 3.0:  # Below 3.0 on 5-point scale
            signals.append(Signal(
                source="customer_monitor",
                signal_type="low_satisfaction",
                value=satisfaction_score,
                timestamp=datetime.now(),
                confidence=0.8,
                metadata={"scale": "5_point", "threshold": 3.0}
            ))
        
        return signals
    
    async def detect_potential_crises(self, signals: List[Signal]) -> List[PotentialCrisis]:
        """Analyze signals to identify potential crises"""
        potential_crises = []
        
        # Group signals by potential crisis type
        crisis_patterns = self._analyze_signal_patterns(signals)
        
        for pattern in crisis_patterns:
            crisis = PotentialCrisis(
                id=f"potential_{pattern['type']}_{datetime.now().timestamp()}",
                crisis_type=pattern['crisis_type'],
                probability=pattern['probability'],
                signals=pattern['signals'],
                predicted_impact=pattern['predicted_impact'],
                time_to_crisis=pattern.get('time_to_crisis'),
                confidence_score=pattern['confidence']
            )
            potential_crises.append(crisis)
        
        return potential_crises
    
    def _analyze_signal_patterns(self, signals: List[Signal]) -> List[Dict[str, Any]]:
        """Analyze signal patterns to identify crisis indicators"""
        patterns = []
        
        # System outage pattern
        system_signals = [s for s in signals if s.source in ['system_monitor', 'application_monitor']]
        if len(system_signals) >= 2:
            patterns.append({
                'type': 'system_outage',
                'crisis_type': CrisisType.SYSTEM_OUTAGE,
                'probability': min(0.9, len(system_signals) * 0.3),
                'signals': system_signals,
                'predicted_impact': 'Service disruption affecting all users',
                'time_to_crisis': timedelta(minutes=15),
                'confidence': 0.85
            })
        
        # Security breach pattern
        security_signals = [s for s in signals if s.source in ['security_monitor', 'network_monitor']]
        if len(security_signals) >= 1:
            patterns.append({
                'type': 'security_breach',
                'crisis_type': CrisisType.SECURITY_BREACH,
                'probability': min(0.8, len(security_signals) * 0.4),
                'signals': security_signals,
                'predicted_impact': 'Potential data compromise and system access',
                'time_to_crisis': timedelta(hours=1),
                'confidence': 0.75
            })
        
        # Financial crisis pattern
        financial_signals = [s for s in signals if s.source == 'financial_monitor']
        if len(financial_signals) >= 1:
            patterns.append({
                'type': 'financial_crisis',
                'crisis_type': CrisisType.FINANCIAL_CRISIS,
                'probability': min(0.7, len(financial_signals) * 0.5),
                'signals': financial_signals,
                'predicted_impact': 'Business continuity and operational funding at risk',
                'time_to_crisis': timedelta(days=7),
                'confidence': 0.8
            })
        
        return patterns
    
    # Mock methods for demonstration - would integrate with real monitoring systems
    async def _get_cpu_usage(self) -> float:
        return 75.0  # Mock CPU usage
    
    async def _get_memory_usage(self) -> float:
        return 80.0  # Mock memory usage
    
    async def _get_error_rate(self) -> float:
        return 2.5  # Mock error rate
    
    async def _get_failed_login_count(self) -> int:
        return 50  # Mock failed login count
    
    async def _detect_network_anomalies(self) -> Optional[Dict[str, Any]]:
        return None  # Mock network anomaly detection
    
    async def _get_revenue_change(self) -> float:
        return -5.0  # Mock revenue change percentage
    
    async def _get_cash_flow_status(self) -> int:
        return 90  # Mock days of runway
    
    async def _get_market_volatility(self) -> float:
        return 0.15  # Mock market volatility
    
    async def _get_customer_satisfaction(self) -> float:
        return 4.2  # Mock customer satisfaction score


class CrisisClassifier:
    """System for categorizing crisis types and severity levels"""
    
    def __init__(self):
        self.classification_rules = self._initialize_classification_rules()
        self.severity_matrix = self._initialize_severity_matrix()
    
    def classify_crisis(self, crisis: Crisis) -> CrisisClassification:
        """Categorize crisis type and determine severity level"""
        
        # Analyze signals to determine crisis type
        crisis_type = self._determine_crisis_type(crisis.signals)
        
        # Calculate severity based on impact and urgency
        severity_level = self._calculate_severity(crisis, crisis_type)
        
        # Determine confidence in classification
        confidence = self._calculate_classification_confidence(crisis.signals, crisis_type)
        
        # Identify sub-categories and related crises
        sub_categories = self._identify_sub_categories(crisis_type, crisis.signals)
        related_crises = self._identify_related_crises(crisis_type)
        
        # Generate classification rationale
        rationale = self._generate_classification_rationale(
            crisis_type, severity_level, crisis.signals
        )
        
        return CrisisClassification(
            crisis_type=crisis_type,
            severity_level=severity_level,
            confidence=confidence,
            sub_categories=sub_categories,
            related_crises=related_crises,
            classification_rationale=rationale
        )
    
    def _determine_crisis_type(self, signals: List[Signal]) -> CrisisType:
        """Determine the primary crisis type based on signals"""
        signal_weights = {}
        
        for signal in signals:
            if signal.source == 'system_monitor' or signal.source == 'application_monitor':
                signal_weights[CrisisType.SYSTEM_OUTAGE] = signal_weights.get(CrisisType.SYSTEM_OUTAGE, 0) + signal.confidence
            elif signal.source == 'security_monitor' or signal.source == 'network_monitor':
                signal_weights[CrisisType.SECURITY_BREACH] = signal_weights.get(CrisisType.SECURITY_BREACH, 0) + signal.confidence
            elif signal.source == 'financial_monitor':
                signal_weights[CrisisType.FINANCIAL_CRISIS] = signal_weights.get(CrisisType.FINANCIAL_CRISIS, 0) + signal.confidence
            elif signal.source == 'customer_monitor':
                signal_weights[CrisisType.REPUTATION_DAMAGE] = signal_weights.get(CrisisType.REPUTATION_DAMAGE, 0) + signal.confidence
        
        # Return crisis type with highest weight
        if signal_weights:
            return max(signal_weights.items(), key=lambda x: x[1])[0]
        
        return CrisisType.SYSTEM_OUTAGE  # Default fallback
    
    def _calculate_severity(self, crisis: Crisis, crisis_type: CrisisType) -> SeverityLevel:
        """Calculate severity level based on crisis characteristics"""
        severity_score = 0
        
        # Base severity by crisis type
        type_severity = {
            CrisisType.SYSTEM_OUTAGE: 3,
            CrisisType.SECURITY_BREACH: 4,
            CrisisType.FINANCIAL_CRISIS: 4,
            CrisisType.REPUTATION_DAMAGE: 3,
            CrisisType.DATA_LOSS: 5,
            CrisisType.REGULATORY_VIOLATION: 4,
            CrisisType.CYBER_ATTACK: 5
        }
        
        severity_score = type_severity.get(crisis_type, 2)
        
        # Adjust based on signal strength
        signal_strength = sum(signal.confidence for signal in crisis.signals) / len(crisis.signals) if crisis.signals else 0
        if signal_strength > 0.8:
            severity_score += 1
        elif signal_strength < 0.5:
            severity_score -= 1
        
        # Adjust based on affected areas
        if len(crisis.affected_areas) > 5:
            severity_score += 1
        
        # Adjust based on stakeholder impact
        if len(crisis.stakeholders_impacted) > 10:
            severity_score += 1
        
        # Clamp to valid range
        severity_score = max(1, min(5, severity_score))
        
        return SeverityLevel(severity_score)
    
    def _calculate_classification_confidence(self, signals: List[Signal], crisis_type: CrisisType) -> float:
        """Calculate confidence in crisis classification"""
        if not signals:
            return 0.5
        
        # Average signal confidence
        avg_confidence = sum(signal.confidence for signal in signals) / len(signals)
        
        # Adjust based on signal consistency
        type_consistent_signals = 0
        for signal in signals:
            if self._signal_matches_crisis_type(signal, crisis_type):
                type_consistent_signals += 1
        
        consistency_factor = type_consistent_signals / len(signals)
        
        return min(0.95, avg_confidence * consistency_factor)
    
    def _signal_matches_crisis_type(self, signal: Signal, crisis_type: CrisisType) -> bool:
        """Check if signal is consistent with crisis type"""
        type_sources = {
            CrisisType.SYSTEM_OUTAGE: ['system_monitor', 'application_monitor'],
            CrisisType.SECURITY_BREACH: ['security_monitor', 'network_monitor'],
            CrisisType.FINANCIAL_CRISIS: ['financial_monitor'],
            CrisisType.REPUTATION_DAMAGE: ['customer_monitor', 'social_monitor']
        }
        
        expected_sources = type_sources.get(crisis_type, [])
        return signal.source in expected_sources
    
    def _identify_sub_categories(self, crisis_type: CrisisType, signals: List[Signal]) -> List[str]:
        """Identify specific sub-categories within crisis type"""
        sub_categories = []
        
        if crisis_type == CrisisType.SYSTEM_OUTAGE:
            for signal in signals:
                if signal.signal_type == 'high_cpu_usage':
                    sub_categories.append('performance_degradation')
                elif signal.signal_type == 'high_error_rate':
                    sub_categories.append('application_failure')
        elif crisis_type == CrisisType.SECURITY_BREACH:
            for signal in signals:
                if signal.signal_type == 'high_failed_logins':
                    sub_categories.append('authentication_attack')
                elif signal.signal_type == 'network_anomaly':
                    sub_categories.append('network_intrusion')
        
        return list(set(sub_categories))  # Remove duplicates
    
    def _identify_related_crises(self, crisis_type: CrisisType) -> List[str]:
        """Identify crises that commonly occur together"""
        related_map = {
            CrisisType.SYSTEM_OUTAGE: ['reputation_damage', 'financial_impact'],
            CrisisType.SECURITY_BREACH: ['data_loss', 'regulatory_violation', 'reputation_damage'],
            CrisisType.FINANCIAL_CRISIS: ['personnel_crisis', 'operational_disruption'],
            CrisisType.DATA_LOSS: ['regulatory_violation', 'reputation_damage']
        }
        
        return related_map.get(crisis_type, [])
    
    def _generate_classification_rationale(self, crisis_type: CrisisType, severity_level: SeverityLevel, signals: List[Signal]) -> str:
        """Generate human-readable rationale for classification"""
        signal_summary = f"{len(signals)} signals detected"
        type_reason = f"Classified as {crisis_type.value} based on signal patterns"
        severity_reason = f"Severity level {severity_level.value} due to impact assessment"
        
        return f"{type_reason}. {severity_reason}. {signal_summary}."
    
    def _initialize_classification_rules(self) -> Dict[str, Any]:
        """Initialize classification rules and patterns"""
        return {
            'signal_patterns': {},
            'escalation_thresholds': {},
            'severity_factors': {}
        }
    
    def _initialize_severity_matrix(self) -> Dict[str, Any]:
        """Initialize severity calculation matrix"""
        return {
            'impact_weights': {},
            'urgency_factors': {},
            'stakeholder_multipliers': {}
        }


class ImpactAssessor:
    """Engine for rapid consequence evaluation"""
    
    def __init__(self):
        self.impact_models = self._initialize_impact_models()
        self.stakeholder_registry = self._initialize_stakeholder_registry()
    
    def assess_impact(self, crisis: Crisis) -> ImpactAssessment:
        """Evaluate potential consequences and scope of crisis"""
        
        # Assess financial impact
        financial_impact = self._assess_financial_impact(crisis)
        
        # Assess operational impact
        operational_impact = self._assess_operational_impact(crisis)
        
        # Assess reputation impact
        reputation_impact = self._assess_reputation_impact(crisis)
        
        # Assess stakeholder impact
        stakeholder_impact = self._assess_stakeholder_impact(crisis)
        
        # Assess timeline impact
        timeline_impact = self._assess_timeline_impact(crisis)
        
        # Estimate recovery time
        recovery_estimate = self._estimate_recovery_time(crisis)
        
        # Identify cascading risks
        cascading_risks = self._identify_cascading_risks(crisis)
        
        # Determine mitigation urgency
        mitigation_urgency = self._determine_mitigation_urgency(crisis)
        
        return ImpactAssessment(
            financial_impact=financial_impact,
            operational_impact=operational_impact,
            reputation_impact=reputation_impact,
            stakeholder_impact=stakeholder_impact,
            timeline_impact=timeline_impact,
            recovery_estimate=recovery_estimate,
            cascading_risks=cascading_risks,
            mitigation_urgency=mitigation_urgency
        )
    
    def _assess_financial_impact(self, crisis: Crisis) -> Dict[str, float]:
        """Assess financial consequences of crisis"""
        impact = {
            'immediate_cost': 0.0,
            'revenue_loss': 0.0,
            'recovery_cost': 0.0,
            'opportunity_cost': 0.0,
            'total_estimated_cost': 0.0
        }
        
        # Base costs by crisis type
        if crisis.crisis_type == CrisisType.SYSTEM_OUTAGE:
            impact['immediate_cost'] = 50000  # Emergency response costs
            impact['revenue_loss'] = 100000 * crisis.severity_level.value  # Hourly revenue loss
            impact['recovery_cost'] = 25000  # System restoration
        elif crisis.crisis_type == CrisisType.SECURITY_BREACH:
            impact['immediate_cost'] = 100000  # Incident response
            impact['revenue_loss'] = 200000 * crisis.severity_level.value  # Customer loss
            impact['recovery_cost'] = 150000  # Security hardening
        elif crisis.crisis_type == CrisisType.FINANCIAL_CRISIS:
            impact['immediate_cost'] = 0  # No immediate costs
            impact['revenue_loss'] = 500000 * crisis.severity_level.value  # Business impact
            impact['recovery_cost'] = 75000  # Restructuring costs
        
        # Calculate total
        impact['total_estimated_cost'] = sum(impact.values()) - impact['total_estimated_cost']
        
        return impact
    
    def _assess_operational_impact(self, crisis: Crisis) -> Dict[str, str]:
        """Assess operational consequences"""
        impact = {
            'service_availability': 'normal',
            'team_productivity': 'normal',
            'customer_experience': 'normal',
            'business_continuity': 'normal'
        }
        
        if crisis.crisis_type == CrisisType.SYSTEM_OUTAGE:
            if crisis.severity_level.value >= 4:
                impact['service_availability'] = 'severely_degraded'
                impact['customer_experience'] = 'severely_impacted'
            elif crisis.severity_level.value >= 3:
                impact['service_availability'] = 'degraded'
                impact['customer_experience'] = 'impacted'
        
        if crisis.crisis_type == CrisisType.SECURITY_BREACH:
            impact['service_availability'] = 'restricted'
            impact['team_productivity'] = 'reduced'
            impact['business_continuity'] = 'at_risk'
        
        return impact
    
    def _assess_reputation_impact(self, crisis: Crisis) -> Dict[str, float]:
        """Assess reputation and brand impact"""
        impact = {
            'brand_damage_score': 0.0,
            'customer_trust_impact': 0.0,
            'media_attention_level': 0.0,
            'recovery_difficulty': 0.0
        }
        
        # Base impact by crisis type
        reputation_multipliers = {
            CrisisType.SYSTEM_OUTAGE: 0.3,
            CrisisType.SECURITY_BREACH: 0.8,
            CrisisType.DATA_LOSS: 0.9,
            CrisisType.FINANCIAL_CRISIS: 0.6,
            CrisisType.REGULATORY_VIOLATION: 0.7
        }
        
        multiplier = reputation_multipliers.get(crisis.crisis_type, 0.4)
        severity_factor = crisis.severity_level.value / 5.0
        
        impact['brand_damage_score'] = multiplier * severity_factor * 10
        impact['customer_trust_impact'] = multiplier * severity_factor * 8
        impact['media_attention_level'] = multiplier * severity_factor * 7
        impact['recovery_difficulty'] = multiplier * severity_factor * 6
        
        return impact
    
    def _assess_stakeholder_impact(self, crisis: Crisis) -> Dict[str, List[str]]:
        """Assess impact on different stakeholder groups"""
        impact = {
            'customers': [],
            'employees': [],
            'investors': [],
            'partners': [],
            'regulators': [],
            'media': []
        }
        
        if crisis.crisis_type == CrisisType.SYSTEM_OUTAGE:
            impact['customers'] = ['service_disruption', 'data_access_issues']
            impact['employees'] = ['increased_workload', 'stress']
            impact['partners'] = ['integration_failures']
        
        if crisis.crisis_type == CrisisType.SECURITY_BREACH:
            impact['customers'] = ['data_privacy_concerns', 'service_restrictions']
            impact['employees'] = ['security_protocols', 'investigation_cooperation']
            impact['investors'] = ['financial_exposure', 'reputation_risk']
            impact['regulators'] = ['compliance_review', 'potential_penalties']
            impact['media'] = ['negative_coverage', 'transparency_demands']
        
        return impact
    
    def _assess_timeline_impact(self, crisis: Crisis) -> Dict[str, timedelta]:
        """Assess timeline and duration impacts"""
        impact = {
            'immediate_response_time': timedelta(minutes=15),
            'containment_time': timedelta(hours=2),
            'resolution_time': timedelta(hours=8),
            'recovery_time': timedelta(days=1),
            'full_restoration_time': timedelta(days=3)
        }
        
        # Adjust based on crisis type and severity
        severity_multiplier = crisis.severity_level.value / 3.0
        
        if crisis.crisis_type == CrisisType.SECURITY_BREACH:
            impact['containment_time'] = timedelta(hours=1 * severity_multiplier)
            impact['resolution_time'] = timedelta(days=1 * severity_multiplier)
            impact['recovery_time'] = timedelta(days=3 * severity_multiplier)
        
        return impact
    
    def _estimate_recovery_time(self, crisis: Crisis) -> timedelta:
        """Estimate total recovery time"""
        base_recovery_times = {
            CrisisType.SYSTEM_OUTAGE: timedelta(hours=4),
            CrisisType.SECURITY_BREACH: timedelta(days=7),
            CrisisType.FINANCIAL_CRISIS: timedelta(days=30),
            CrisisType.DATA_LOSS: timedelta(days=14),
            CrisisType.REPUTATION_DAMAGE: timedelta(days=60)
        }
        
        base_time = base_recovery_times.get(crisis.crisis_type, timedelta(days=7))
        severity_multiplier = crisis.severity_level.value / 3.0
        
        return timedelta(seconds=base_time.total_seconds() * severity_multiplier)
    
    def _identify_cascading_risks(self, crisis: Crisis) -> List[str]:
        """Identify potential cascading risks and secondary crises"""
        risks = []
        
        if crisis.crisis_type == CrisisType.SYSTEM_OUTAGE:
            risks.extend([
                'customer_churn',
                'revenue_loss',
                'reputation_damage',
                'partner_relationship_strain'
            ])
        
        if crisis.crisis_type == CrisisType.SECURITY_BREACH:
            risks.extend([
                'data_loss',
                'regulatory_investigation',
                'legal_liability',
                'customer_lawsuits',
                'competitive_disadvantage'
            ])
        
        if crisis.crisis_type == CrisisType.FINANCIAL_CRISIS:
            risks.extend([
                'layoffs',
                'operational_cuts',
                'investor_confidence_loss',
                'credit_rating_downgrade'
            ])
        
        return risks
    
    def _determine_mitigation_urgency(self, crisis: Crisis) -> SeverityLevel:
        """Determine urgency of mitigation actions"""
        urgency_score = crisis.severity_level.value
        
        # Increase urgency for time-sensitive crises
        if crisis.crisis_type in [CrisisType.SECURITY_BREACH, CrisisType.SYSTEM_OUTAGE]:
            urgency_score = min(5, urgency_score + 1)
        
        # Increase urgency if many stakeholders affected
        if len(crisis.stakeholders_impacted) > 10:
            urgency_score = min(5, urgency_score + 1)
        
        return SeverityLevel(urgency_score)
    
    def _initialize_impact_models(self) -> Dict[str, Any]:
        """Initialize impact assessment models"""
        return {
            'financial_models': {},
            'operational_models': {},
            'reputation_models': {}
        }
    
    def _initialize_stakeholder_registry(self) -> Dict[str, Any]:
        """Initialize stakeholder registry and impact mappings"""
        return {
            'stakeholder_groups': {},
            'impact_mappings': {},
            'communication_preferences': {}
        }


class EscalationManager:
    """System for automated crisis escalation"""
    
    def __init__(self):
        self.escalation_rules = self._initialize_escalation_rules()
        self.notification_channels = self._initialize_notification_channels()
        self.escalation_history = []
    
    def should_escalate(self, crisis: Crisis) -> bool:
        """Determine if crisis should be escalated"""
        
        # Always escalate critical and catastrophic crises
        if crisis.severity_level.value >= 4:
            return True
        
        # Escalate based on crisis type
        auto_escalate_types = [
            CrisisType.SECURITY_BREACH,
            CrisisType.DATA_LOSS,
            CrisisType.REGULATORY_VIOLATION
        ]
        if crisis.crisis_type in auto_escalate_types:
            return True
        
        # Escalate if many stakeholders affected
        if len(crisis.stakeholders_impacted) > 20:
            return True
        
        # Escalate if crisis has been active for too long
        if crisis.start_time and (datetime.now() - crisis.start_time) > timedelta(hours=2):
            return True
        
        return False
    
    async def escalate_crisis(self, crisis: Crisis) -> Dict[str, Any]:
        """Execute crisis escalation process"""
        escalation_result = {
            'escalated': False,
            'escalation_level': None,
            'notifications_sent': [],
            'escalation_time': datetime.now(),
            'escalation_reason': None
        }
        
        if not self.should_escalate(crisis):
            escalation_result['escalation_reason'] = 'No escalation criteria met'
            return escalation_result
        
        # Determine escalation level
        escalation_level = self._determine_escalation_level(crisis)
        escalation_result['escalation_level'] = escalation_level
        
        # Send notifications
        notifications = await self._send_escalation_notifications(crisis, escalation_level)
        escalation_result['notifications_sent'] = notifications
        
        # Update crisis status
        crisis.current_status = CrisisStatus.ESCALATED
        
        # Record escalation
        escalation_record = {
            'timestamp': datetime.now(),
            'level': escalation_level,
            'reason': self._get_escalation_reason(crisis),
            'notifications': notifications
        }
        crisis.escalation_history.append(escalation_record)
        self.escalation_history.append(escalation_record)
        
        escalation_result['escalated'] = True
        escalation_result['escalation_reason'] = escalation_record['reason']
        
        logger.info(f"Crisis {crisis.id} escalated to level {escalation_level}")
        
        return escalation_result
    
    def _determine_escalation_level(self, crisis: Crisis) -> int:
        """Determine appropriate escalation level"""
        
        # Level 1: Team lead notification
        # Level 2: Department head notification  
        # Level 3: Executive team notification
        # Level 4: CEO and board notification
        # Level 5: External stakeholder notification
        
        if crisis.severity_level == SeverityLevel.CATASTROPHIC:
            return 5
        elif crisis.severity_level == SeverityLevel.CRITICAL:
            return 4
        elif crisis.severity_level == SeverityLevel.HIGH:
            return 3
        elif crisis.crisis_type in [CrisisType.SECURITY_BREACH, CrisisType.DATA_LOSS]:
            return 4  # Always escalate security issues high
        else:
            return 2
    
    async def _send_escalation_notifications(self, crisis: Crisis, escalation_level: int) -> List[Dict[str, Any]]:
        """Send notifications for crisis escalation"""
        notifications = []
        
        # Define notification recipients by level
        recipients_by_level = {
            1: ['team_lead'],
            2: ['team_lead', 'department_head'],
            3: ['team_lead', 'department_head', 'executive_team'],
            4: ['team_lead', 'department_head', 'executive_team', 'ceo'],
            5: ['team_lead', 'department_head', 'executive_team', 'ceo', 'board', 'external_stakeholders']
        }
        
        recipients = recipients_by_level.get(escalation_level, ['team_lead'])
        
        for recipient in recipients:
            notification = await self._send_notification(crisis, recipient, escalation_level)
            notifications.append(notification)
        
        return notifications
    
    async def _send_notification(self, crisis: Crisis, recipient: str, escalation_level: int) -> Dict[str, Any]:
        """Send individual notification"""
        notification = {
            'recipient': recipient,
            'channel': self._get_notification_channel(recipient),
            'message': self._generate_escalation_message(crisis, escalation_level),
            'sent_time': datetime.now(),
            'status': 'sent'
        }
        
        # In real implementation, would actually send notification
        logger.info(f"Escalation notification sent to {recipient} via {notification['channel']}")
        
        return notification
    
    def _get_notification_channel(self, recipient: str) -> str:
        """Get appropriate notification channel for recipient"""
        channel_map = {
            'team_lead': 'slack',
            'department_head': 'email',
            'executive_team': 'phone',
            'ceo': 'phone',
            'board': 'email',
            'external_stakeholders': 'email'
        }
        
        return channel_map.get(recipient, 'email')
    
    def _generate_escalation_message(self, crisis: Crisis, escalation_level: int) -> str:
        """Generate escalation notification message"""
        urgency_text = {
            1: 'Low Priority',
            2: 'Medium Priority', 
            3: 'High Priority',
            4: 'Critical Priority',
            5: 'Emergency Priority'
        }
        
        message = f"""
CRISIS ESCALATION - {urgency_text.get(escalation_level, 'Unknown Priority')}

Crisis ID: {crisis.id}
Type: {crisis.crisis_type.value}
Severity: {crisis.severity_level.name}
Status: {crisis.current_status.value}
Start Time: {crisis.start_time}

Affected Areas: {', '.join(crisis.affected_areas)}
Stakeholders Impacted: {len(crisis.stakeholders_impacted)} groups

Immediate action required. Please respond to crisis management team.
        """.strip()
        
        return message
    
    def _get_escalation_reason(self, crisis: Crisis) -> str:
        """Get human-readable escalation reason"""
        reasons = []
        
        if crisis.severity_level.value >= 4:
            reasons.append(f"High severity level ({crisis.severity_level.name})")
        
        if crisis.crisis_type in [CrisisType.SECURITY_BREACH, CrisisType.DATA_LOSS]:
            reasons.append(f"Critical crisis type ({crisis.crisis_type.value})")
        
        if len(crisis.stakeholders_impacted) > 20:
            reasons.append(f"Large stakeholder impact ({len(crisis.stakeholders_impacted)} groups)")
        
        if crisis.start_time and (datetime.now() - crisis.start_time) > timedelta(hours=2):
            duration = datetime.now() - crisis.start_time
            reasons.append(f"Extended duration ({duration})")
        
        return '; '.join(reasons) if reasons else 'Automatic escalation criteria met'
    
    def _initialize_escalation_rules(self) -> Dict[str, Any]:
        """Initialize escalation rules and thresholds"""
        return {
            'severity_thresholds': {},
            'time_thresholds': {},
            'stakeholder_thresholds': {}
        }
    
    def _initialize_notification_channels(self) -> Dict[str, Any]:
        """Initialize notification channels and preferences"""
        return {
            'email': {},
            'sms': {},
            'phone': {},
            'slack': {},
            'teams': {}
        }


class CrisisDetectionEngine:
    """Main crisis detection and assessment engine"""
    
    def __init__(self):
        self.early_warning_system = EarlyWarningSystem()
        self.crisis_classifier = CrisisClassifier()
        self.impact_assessor = ImpactAssessor()
        self.escalation_manager = EscalationManager()
        self.active_crises = {}
        self.crisis_history = []
    
    async def detect_and_assess_crises(self) -> List[Crisis]:
        """Main detection and assessment workflow"""
        try:
            # Monitor signals for potential crises
            signals = await self.early_warning_system.monitor_signals()
            
            # Detect potential crises from signals
            potential_crises = await self.early_warning_system.detect_potential_crises(signals)
            
            # Convert potential crises to actual crises if thresholds met
            new_crises = []
            for potential_crisis in potential_crises:
                if potential_crisis.probability > 0.7:  # Threshold for crisis activation
                    crisis = self._create_crisis_from_potential(potential_crisis)
                    new_crises.append(crisis)
            
            # Process each new crisis
            processed_crises = []
            for crisis in new_crises:
                processed_crisis = await self._process_crisis(crisis)
                processed_crises.append(processed_crisis)
                self.active_crises[crisis.id] = processed_crisis
            
            return processed_crises
            
        except Exception as e:
            logger.error(f"Error in crisis detection and assessment: {str(e)}")
            return []
    
    async def _process_crisis(self, crisis: Crisis) -> Crisis:
        """Process individual crisis through full pipeline"""
        
        # Classify the crisis
        classification = self.crisis_classifier.classify_crisis(crisis)
        crisis.classification = classification
        crisis.crisis_type = classification.crisis_type
        crisis.severity_level = classification.severity_level
        
        # Assess impact
        impact_assessment = self.impact_assessor.assess_impact(crisis)
        crisis.impact_assessment = impact_assessment
        
        # Update crisis status
        crisis.current_status = CrisisStatus.ASSESSED
        
        # Check for escalation
        escalation_result = await self.escalation_manager.escalate_crisis(crisis)
        
        # Log crisis processing
        logger.info(f"Processed crisis {crisis.id}: {crisis.crisis_type.value} - {crisis.severity_level.name}")
        
        return crisis
    
    def _create_crisis_from_potential(self, potential_crisis: PotentialCrisis) -> Crisis:
        """Convert potential crisis to actual crisis"""
        return Crisis(
            id=potential_crisis.id.replace('potential_', 'crisis_'),
            crisis_type=potential_crisis.crisis_type,
            severity_level=SeverityLevel.MEDIUM,  # Will be updated by classifier
            start_time=datetime.now(),
            affected_areas=self._determine_affected_areas(potential_crisis),
            stakeholders_impacted=self._determine_stakeholders(potential_crisis),
            current_status=CrisisStatus.DETECTED,
            signals=potential_crisis.signals
        )
    
    def _determine_affected_areas(self, potential_crisis: PotentialCrisis) -> List[str]:
        """Determine affected areas from potential crisis"""
        areas = []
        
        if potential_crisis.crisis_type == CrisisType.SYSTEM_OUTAGE:
            areas = ['production_systems', 'customer_services', 'api_endpoints']
        elif potential_crisis.crisis_type == CrisisType.SECURITY_BREACH:
            areas = ['user_data', 'authentication_systems', 'network_infrastructure']
        elif potential_crisis.crisis_type == CrisisType.FINANCIAL_CRISIS:
            areas = ['revenue_streams', 'operational_budget', 'investor_relations']
        
        return areas
    
    def _determine_stakeholders(self, potential_crisis: PotentialCrisis) -> List[str]:
        """Determine impacted stakeholders from potential crisis"""
        stakeholders = []
        
        if potential_crisis.crisis_type == CrisisType.SYSTEM_OUTAGE:
            stakeholders = ['customers', 'support_team', 'engineering_team']
        elif potential_crisis.crisis_type == CrisisType.SECURITY_BREACH:
            stakeholders = ['customers', 'security_team', 'legal_team', 'executives']
        elif potential_crisis.crisis_type == CrisisType.FINANCIAL_CRISIS:
            stakeholders = ['investors', 'employees', 'executives', 'board']
        
        return stakeholders
    
    async def get_active_crises(self) -> List[Crisis]:
        """Get all currently active crises"""
        return list(self.active_crises.values())
    
    async def get_crisis_by_id(self, crisis_id: str) -> Optional[Crisis]:
        """Get specific crisis by ID"""
        return self.active_crises.get(crisis_id)
    
    async def resolve_crisis(self, crisis_id: str) -> bool:
        """Mark crisis as resolved"""
        if crisis_id in self.active_crises:
            crisis = self.active_crises[crisis_id]
            crisis.current_status = CrisisStatus.RESOLVED
            crisis.resolution_time = datetime.now()
            
            # Move to history
            self.crisis_history.append(crisis)
            del self.active_crises[crisis_id]
            
            logger.info(f"Crisis {crisis_id} resolved")
            return True
        
        return False
    
    async def get_crisis_metrics(self) -> Dict[str, Any]:
        """Get crisis detection and management metrics"""
        active_count = len(self.active_crises)
        total_resolved = len(self.crisis_history)
        
        # Calculate average resolution time
        resolved_with_times = [c for c in self.crisis_history if c.resolution_time]
        if resolved_with_times:
            avg_resolution_time = sum(
                (c.resolution_time - c.start_time).total_seconds() 
                for c in resolved_with_times
            ) / len(resolved_with_times)
        else:
            avg_resolution_time = 0
        
        # Crisis type distribution
        type_distribution = {}
        all_crises = list(self.active_crises.values()) + self.crisis_history
        for crisis in all_crises:
            crisis_type = crisis.crisis_type.value
            type_distribution[crisis_type] = type_distribution.get(crisis_type, 0) + 1
        
        return {
            'active_crises': active_count,
            'total_resolved': total_resolved,
            'average_resolution_time_seconds': avg_resolution_time,
            'crisis_type_distribution': type_distribution,
            'escalation_rate': len(self.escalation_manager.escalation_history) / max(1, total_resolved + active_count)
        }