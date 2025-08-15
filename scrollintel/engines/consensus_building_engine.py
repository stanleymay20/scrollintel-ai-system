"""
Consensus Building Engine for Board Executive Mastery System

This engine provides comprehensive consensus building capabilities for board-level
decision making, including stakeholder analysis, strategy development, and facilitation.
"""

import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Tuple
import logging
from dataclasses import asdict
import statistics

from ..models.consensus_building_models import (
    ConsensusBuilding, BoardMemberProfile, ConsensusPosition, ConsensusBarrier,
    ConsensusStrategy as ConsensusStrategyModel, ConsensusAction, ConsensusMetrics, ConsensusRecommendation,
    ConsensusOptimization, ConsensusVisualization, ConsensusAchievement,
    ConsensusStatus, StakeholderPosition, InfluenceLevel, ConsensusStrategyType
)

logger = logging.getLogger(__name__)


class ConsensusBuildingEngine:
    """
    Engine for board consensus building strategy development and facilitation
    
    Provides comprehensive consensus building capabilities including:
    - Stakeholder position analysis
    - Consensus strategy development
    - Progress tracking and measurement
    - Optimization recommendations
    - Achievement validation
    """
    
    def __init__(self):
        self.consensus_processes: Dict[str, ConsensusBuilding] = {}
        self.consensus_templates: Dict[str, Dict[str, Any]] = {}
        self._initialize_consensus_templates()
    
    def _initialize_consensus_templates(self):
        """Initialize consensus building templates for common scenarios"""
        self.consensus_templates = {
            "strategic_decision": {
                "typical_barriers": [
                    {"type": "information", "description": "Insufficient market data"},
                    {"type": "interests", "description": "Conflicting departmental priorities"},
                    {"type": "trust", "description": "Concerns about implementation capability"}
                ],
                "recommended_strategies": [
                    ConsensusStrategyType.INFORMATION_SHARING,
                    ConsensusStrategyType.STAKEHOLDER_ENGAGEMENT,
                    ConsensusStrategyType.COALITION_BUILDING
                ]
            },
            "budget_allocation": {
                "typical_barriers": [
                    {"type": "interests", "description": "Competing resource needs"},
                    {"type": "values", "description": "Different investment philosophies"},
                    {"type": "information", "description": "ROI uncertainty"}
                ],
                "recommended_strategies": [
                    ConsensusStrategyType.COMPROMISE_SEEKING,
                    ConsensusStrategyType.INFORMATION_SHARING,
                    ConsensusStrategyType.INCREMENTAL_AGREEMENT
                ]
            },
            "personnel_decision": {
                "typical_barriers": [
                    {"type": "trust", "description": "Concerns about candidate fit"},
                    {"type": "process", "description": "Evaluation process concerns"},
                    {"type": "values", "description": "Cultural alignment questions"}
                ],
                "recommended_strategies": [
                    ConsensusStrategyType.DIRECT_PERSUASION,
                    ConsensusStrategyType.INFORMATION_SHARING,
                    ConsensusStrategyType.STAKEHOLDER_ENGAGEMENT
                ]
            }
        }
    
    def create_consensus_building(
        self,
        title: str,
        description: str,
        decision_topic: str,
        target_consensus_level: ConsensusStatus = ConsensusStatus.STRONG_CONSENSUS,
        deadline: Optional[datetime] = None,
        facilitator_id: str = "system"
    ) -> ConsensusBuilding:
        """Create a new consensus building process"""
        
        process_id = str(uuid.uuid4())
        
        consensus_process = ConsensusBuilding(
            id=process_id,
            title=title,
            description=description,
            decision_topic=decision_topic,
            created_at=datetime.now(),
            target_consensus_level=target_consensus_level,
            current_consensus_level=ConsensusStatus.NOT_STARTED,
            deadline=deadline,
            board_members=[],
            stakeholder_positions=[],
            barriers=[],
            strategies=[],
            actions=[],
            consensus_score=0.0,
            momentum=0.0,
            key_influencers=[],
            coalition_map={},
            facilitator_id=facilitator_id,
            success_probability=0.5,
            last_updated=datetime.now()
        )
        
        self.consensus_processes[process_id] = consensus_process
        logger.info(f"Created consensus building process: {title} (ID: {process_id})")
        
        return consensus_process
    
    def add_board_member(
        self,
        process_id: str,
        name: str,
        role: str,
        influence_level: InfluenceLevel,
        decision_making_style: str = "analytical",
        key_concerns: List[str] = None,
        motivations: List[str] = None,
        communication_preferences: List[str] = None
    ) -> BoardMemberProfile:
        """Add board member profile to consensus process"""
        
        if process_id not in self.consensus_processes:
            raise ValueError(f"Consensus process {process_id} not found")
        
        member_id = str(uuid.uuid4())
        
        member = BoardMemberProfile(
            id=member_id,
            name=name,
            role=role,
            influence_level=influence_level,
            decision_making_style=decision_making_style,
            key_concerns=key_concerns or [],
            motivations=motivations or [],
            communication_preferences=communication_preferences or [],
            historical_positions={},
            relationship_network=[]
        )
        
        self.consensus_processes[process_id].board_members.append(member)
        self._update_process_timestamp(process_id)
        
        logger.info(f"Added board member: {name} to process {process_id}")
        return member
    
    def update_stakeholder_position(
        self,
        process_id: str,
        stakeholder_id: str,
        stakeholder_name: str,
        position: StakeholderPosition,
        confidence_level: float = 0.8,
        key_concerns: List[str] = None,
        requirements_for_support: List[str] = None,
        deal_breakers: List[str] = None
    ) -> ConsensusPosition:
        """Update or add stakeholder position"""
        
        if process_id not in self.consensus_processes:
            raise ValueError(f"Consensus process {process_id} not found")
        
        process = self.consensus_processes[process_id]
        
        # Find existing position or create new one
        existing_position = next(
            (pos for pos in process.stakeholder_positions if pos.stakeholder_id == stakeholder_id),
            None
        )
        
        if existing_position:
            existing_position.current_position = position
            existing_position.confidence_level = confidence_level
            existing_position.key_concerns = key_concerns or existing_position.key_concerns
            existing_position.requirements_for_support = requirements_for_support or existing_position.requirements_for_support
            existing_position.deal_breakers = deal_breakers or existing_position.deal_breakers
            existing_position.last_updated = datetime.now()
            position_obj = existing_position
        else:
            position_obj = ConsensusPosition(
                stakeholder_id=stakeholder_id,
                stakeholder_name=stakeholder_name,
                current_position=position,
                confidence_level=confidence_level,
                key_concerns=key_concerns or [],
                requirements_for_support=requirements_for_support or [],
                deal_breakers=deal_breakers or [],
                influence_on_others=[],
                last_updated=datetime.now()
            )
            process.stakeholder_positions.append(position_obj)
        
        # Update consensus metrics
        self._calculate_consensus_score(process_id)
        self._update_coalition_map(process_id)
        self._update_process_timestamp(process_id)
        
        logger.info(f"Updated position for {stakeholder_name}: {position.value}")
        return position_obj
    
    def identify_consensus_barriers(
        self,
        process_id: str,
        decision_type: str = "strategic_decision"
    ) -> List[ConsensusBarrier]:
        """Identify barriers to consensus based on stakeholder positions"""
        
        if process_id not in self.consensus_processes:
            raise ValueError(f"Consensus process {process_id} not found")
        
        process = self.consensus_processes[process_id]
        barriers = []
        
        # Analyze stakeholder positions for barriers
        opposing_stakeholders = [
            pos for pos in process.stakeholder_positions 
            if pos.current_position in [StakeholderPosition.OPPOSE, StakeholderPosition.STRONGLY_OPPOSE]
        ]
        
        neutral_stakeholders = [
            pos for pos in process.stakeholder_positions 
            if pos.current_position in [StakeholderPosition.NEUTRAL, StakeholderPosition.UNDECIDED]
        ]
        
        # Information barriers
        if any("information" in concern.lower() or "data" in concern.lower() 
               for pos in process.stakeholder_positions for concern in pos.key_concerns):
            barriers.append(ConsensusBarrier(
                id=str(uuid.uuid4()),
                description="Insufficient information or data to make informed decision",
                barrier_type="information",
                affected_stakeholders=[pos.stakeholder_id for pos in neutral_stakeholders + opposing_stakeholders],
                severity=0.7,
                mitigation_strategies=[
                    "Provide comprehensive data analysis",
                    "Conduct expert presentations",
                    "Share case studies and benchmarks"
                ],
                estimated_resolution_time="2-3 weeks"
            ))
        
        # Trust barriers
        trust_concerns = [
            pos for pos in process.stakeholder_positions 
            if any("trust" in concern.lower() or "confidence" in concern.lower() 
                   for concern in pos.key_concerns)
        ]
        if trust_concerns:
            barriers.append(ConsensusBarrier(
                id=str(uuid.uuid4()),
                description="Trust and confidence concerns about proposal or implementation",
                barrier_type="trust",
                affected_stakeholders=[pos.stakeholder_id for pos in trust_concerns],
                severity=0.8,
                mitigation_strategies=[
                    "Increase transparency in decision process",
                    "Provide implementation guarantees",
                    "Establish oversight mechanisms"
                ],
                estimated_resolution_time="3-4 weeks"
            ))
        
        # Interest conflicts
        if len(opposing_stakeholders) > 0:
            barriers.append(ConsensusBarrier(
                id=str(uuid.uuid4()),
                description="Conflicting interests and priorities among stakeholders",
                barrier_type="interests",
                affected_stakeholders=[pos.stakeholder_id for pos in opposing_stakeholders],
                severity=0.6,
                mitigation_strategies=[
                    "Identify win-win solutions",
                    "Negotiate compromises",
                    "Align with broader organizational goals"
                ],
                estimated_resolution_time="2-4 weeks"
            ))
        
        # Add template barriers if applicable
        if decision_type in self.consensus_templates:
            template_barriers = self.consensus_templates[decision_type]["typical_barriers"]
            for template_barrier in template_barriers:
                barriers.append(ConsensusBarrier(
                    id=str(uuid.uuid4()),
                    description=template_barrier["description"],
                    barrier_type=template_barrier["type"],
                    affected_stakeholders=[pos.stakeholder_id for pos in process.stakeholder_positions],
                    severity=0.5,
                    mitigation_strategies=[
                        "Address through targeted communication",
                        "Provide additional context and rationale"
                    ],
                    estimated_resolution_time="1-2 weeks"
                ))
        
        # Update process barriers
        process.barriers.extend(barriers)
        self._update_process_timestamp(process_id)
        
        logger.info(f"Identified {len(barriers)} consensus barriers for process {process_id}")
        return barriers
    
    def develop_consensus_strategy(
        self,
        process_id: str,
        decision_type: str = "strategic_decision"
    ) -> List[ConsensusStrategyModel]:
        """Develop strategies for building consensus"""
        
        if process_id not in self.consensus_processes:
            raise ValueError(f"Consensus process {process_id} not found")
        
        process = self.consensus_processes[process_id]
        strategies = []
        
        # Analyze current situation
        support_count = len([pos for pos in process.stakeholder_positions 
                           if pos.current_position in [StakeholderPosition.SUPPORT, StakeholderPosition.STRONGLY_SUPPORT]])
        oppose_count = len([pos for pos in process.stakeholder_positions 
                          if pos.current_position in [StakeholderPosition.OPPOSE, StakeholderPosition.STRONGLY_OPPOSE]])
        neutral_count = len([pos for pos in process.stakeholder_positions 
                           if pos.current_position in [StakeholderPosition.NEUTRAL, StakeholderPosition.UNDECIDED]])
        
        # Strategy 1: Information sharing (if information barriers exist)
        info_barriers = [b for b in process.barriers if b.barrier_type == "information"]
        if info_barriers or neutral_count > 0:
            strategies.append(ConsensusStrategyModel(
                id=str(uuid.uuid4()),
                strategy_type=ConsensusStrategyType.INFORMATION_SHARING,
                target_stakeholders=[pos.stakeholder_id for pos in process.stakeholder_positions 
                                   if pos.current_position in [StakeholderPosition.NEUTRAL, StakeholderPosition.UNDECIDED]],
                description="Provide comprehensive information to address knowledge gaps",
                tactics=[
                    "Prepare detailed briefing materials",
                    "Conduct expert presentations",
                    "Share relevant case studies and benchmarks",
                    "Provide Q&A sessions"
                ],
                expected_outcomes=[
                    "Increased stakeholder understanding",
                    "Reduced uncertainty",
                    "More informed decision making"
                ],
                success_probability=0.7,
                estimated_timeline="2-3 weeks",
                resource_requirements=["Research team", "Presentation materials", "Expert time"],
                risks=["Information overload", "Conflicting expert opinions"]
            ))
        
        # Strategy 2: Coalition building (if there's existing support)
        if support_count > 0:
            strategies.append(ConsensusStrategyModel(
                id=str(uuid.uuid4()),
                strategy_type=ConsensusStrategyType.COALITION_BUILDING,
                target_stakeholders=[pos.stakeholder_id for pos in process.stakeholder_positions 
                                   if pos.current_position in [StakeholderPosition.SUPPORT, StakeholderPosition.STRONGLY_SUPPORT]],
                description="Build coalition of supporters to influence others",
                tactics=[
                    "Identify key influencers among supporters",
                    "Coordinate messaging and advocacy",
                    "Leverage relationships and networks",
                    "Create peer-to-peer influence opportunities"
                ],
                expected_outcomes=[
                    "Stronger support base",
                    "Peer influence on undecided members",
                    "Momentum building"
                ],
                success_probability=0.6,
                estimated_timeline="3-4 weeks",
                resource_requirements=["Coordination effort", "Communication channels"],
                risks=["Perception of manipulation", "Backfire if poorly executed"]
            ))
        
        # Strategy 3: Direct persuasion (for specific concerns)
        high_influence_opposers = [
            pos for pos in process.stakeholder_positions 
            if pos.current_position in [StakeholderPosition.OPPOSE, StakeholderPosition.STRONGLY_OPPOSE]
            and any(member.id == pos.stakeholder_id and member.influence_level in [InfluenceLevel.HIGH, InfluenceLevel.CRITICAL] 
                   for member in process.board_members)
        ]
        
        if high_influence_opposers:
            strategies.append(ConsensusStrategyModel(
                id=str(uuid.uuid4()),
                strategy_type=ConsensusStrategyType.DIRECT_PERSUASION,
                target_stakeholders=[pos.stakeholder_id for pos in high_influence_opposers],
                description="Direct engagement with key opposing stakeholders",
                tactics=[
                    "One-on-one meetings with key opposers",
                    "Address specific concerns directly",
                    "Provide customized solutions",
                    "Negotiate modifications to proposal"
                ],
                expected_outcomes=[
                    "Reduced opposition",
                    "Specific concerns addressed",
                    "Potential position changes"
                ],
                success_probability=0.5,
                estimated_timeline="2-3 weeks",
                resource_requirements=["Senior leadership time", "Flexibility in proposal"],
                risks=["Entrenched positions", "Appearance of favoritism"]
            ))
        
        # Strategy 4: Compromise seeking (if significant opposition exists)
        if oppose_count > support_count * 0.5:
            strategies.append(ConsensusStrategyModel(
                id=str(uuid.uuid4()),
                strategy_type=ConsensusStrategyType.COMPROMISE_SEEKING,
                target_stakeholders=[pos.stakeholder_id for pos in process.stakeholder_positions],
                description="Seek compromise solutions that address key concerns",
                tactics=[
                    "Identify areas of flexibility in proposal",
                    "Explore win-win modifications",
                    "Negotiate phased implementation",
                    "Create safeguards and oversight mechanisms"
                ],
                expected_outcomes=[
                    "Modified proposal with broader appeal",
                    "Reduced opposition",
                    "Acceptable middle ground"
                ],
                success_probability=0.8,
                estimated_timeline="3-5 weeks",
                resource_requirements=["Negotiation expertise", "Proposal flexibility"],
                risks=["Diluted proposal effectiveness", "New concerns from modifications"]
            ))
        
        # Add template strategies if applicable
        if decision_type in self.consensus_templates:
            template_strategies = self.consensus_templates[decision_type]["recommended_strategies"]
            # Template strategies are already covered above, but we could add specific variations
        
        # Update process strategies
        process.strategies.extend(strategies)
        self._update_process_timestamp(process_id)
        
        logger.info(f"Developed {len(strategies)} consensus strategies for process {process_id}")
        return strategies
    
    def create_consensus_action(
        self,
        process_id: str,
        title: str,
        description: str,
        action_type: str,
        target_stakeholders: List[str],
        responsible_party: str,
        deadline: datetime,
        expected_impact: str
    ) -> ConsensusAction:
        """Create specific consensus building action"""
        
        if process_id not in self.consensus_processes:
            raise ValueError(f"Consensus process {process_id} not found")
        
        action_id = str(uuid.uuid4())
        
        action = ConsensusAction(
            id=action_id,
            title=title,
            description=description,
            action_type=action_type,
            target_stakeholders=target_stakeholders,
            responsible_party=responsible_party,
            deadline=deadline,
            status="planned",
            expected_impact=expected_impact,
            actual_impact=None,
            follow_up_required=False
        )
        
        self.consensus_processes[process_id].actions.append(action)
        self._update_process_timestamp(process_id)
        
        logger.info(f"Created consensus action: {title}")
        return action
    
    def track_consensus_progress(self, process_id: str) -> ConsensusMetrics:
        """Track and measure consensus building progress"""
        
        if process_id not in self.consensus_processes:
            raise ValueError(f"Consensus process {process_id} not found")
        
        process = self.consensus_processes[process_id]
        metrics_id = str(uuid.uuid4())
        
        # Calculate support percentages
        total_stakeholders = len(process.stakeholder_positions)
        if total_stakeholders == 0:
            return ConsensusMetrics(
                id=metrics_id,
                consensus_building_id=process_id,
                measurement_date=datetime.now(),
                support_percentage=0.0,
                opposition_percentage=0.0,
                neutral_percentage=0.0,
                weighted_support_score=0.0,
                momentum_direction="stable",
                key_concerns_addressed=0,
                barriers_resolved=0,
                new_barriers_identified=0,
                stakeholder_engagement_level=0.0,
                communication_effectiveness=0.0,
                trust_level=0.0
            )
        
        support_count = len([pos for pos in process.stakeholder_positions 
                           if pos.current_position in [StakeholderPosition.SUPPORT, StakeholderPosition.STRONGLY_SUPPORT]])
        oppose_count = len([pos for pos in process.stakeholder_positions 
                          if pos.current_position in [StakeholderPosition.OPPOSE, StakeholderPosition.STRONGLY_OPPOSE]])
        neutral_count = len([pos for pos in process.stakeholder_positions 
                           if pos.current_position in [StakeholderPosition.NEUTRAL, StakeholderPosition.UNDECIDED]])
        
        support_percentage = (support_count / total_stakeholders) * 100
        opposition_percentage = (oppose_count / total_stakeholders) * 100
        neutral_percentage = (neutral_count / total_stakeholders) * 100
        
        # Calculate weighted support score (considering influence levels)
        weighted_support = 0.0
        total_weight = 0.0
        
        for pos in process.stakeholder_positions:
            # Find corresponding board member for influence weight
            member = next((m for m in process.board_members if m.id == pos.stakeholder_id), None)
            weight = 1.0  # Default weight
            
            if member:
                weight_map = {
                    InfluenceLevel.LOW: 0.5,
                    InfluenceLevel.MODERATE: 1.0,
                    InfluenceLevel.HIGH: 2.0,
                    InfluenceLevel.CRITICAL: 3.0
                }
                weight = weight_map.get(member.influence_level, 1.0)
            
            position_score = {
                StakeholderPosition.STRONGLY_SUPPORT: 1.0,
                StakeholderPosition.SUPPORT: 0.75,
                StakeholderPosition.NEUTRAL: 0.5,
                StakeholderPosition.UNDECIDED: 0.5,
                StakeholderPosition.OPPOSE: 0.25,
                StakeholderPosition.STRONGLY_OPPOSE: 0.0
            }.get(pos.current_position, 0.5)
            
            weighted_support += position_score * weight
            total_weight += weight
        
        weighted_support_score = (weighted_support / total_weight) if total_weight > 0 else 0.0
        
        # Determine momentum direction
        momentum_direction = "stable"
        if process.momentum > 0.1:
            momentum_direction = "positive"
        elif process.momentum < -0.1:
            momentum_direction = "negative"
        
        # Calculate engagement and effectiveness metrics
        completed_actions = len([a for a in process.actions if a.status == "completed"])
        total_actions = len(process.actions)
        engagement_level = (completed_actions / total_actions) if total_actions > 0 else 0.0
        
        # Estimate communication effectiveness based on position confidence
        avg_confidence = statistics.mean([pos.confidence_level for pos in process.stakeholder_positions]) if process.stakeholder_positions else 0.0
        communication_effectiveness = avg_confidence
        
        # Estimate trust level based on trust-related barriers
        trust_barriers = len([b for b in process.barriers if b.barrier_type == "trust"])
        trust_level = max(0.0, 1.0 - (trust_barriers * 0.2))
        
        metrics = ConsensusMetrics(
            id=metrics_id,
            consensus_building_id=process_id,
            measurement_date=datetime.now(),
            support_percentage=support_percentage,
            opposition_percentage=opposition_percentage,
            neutral_percentage=neutral_percentage,
            weighted_support_score=weighted_support_score,
            momentum_direction=momentum_direction,
            key_concerns_addressed=0,  # Would be calculated based on action outcomes
            barriers_resolved=0,       # Would be tracked over time
            new_barriers_identified=0, # Would be tracked over time
            stakeholder_engagement_level=engagement_level,
            communication_effectiveness=communication_effectiveness,
            trust_level=trust_level
        )
        
        # Update process consensus score and status
        process.consensus_score = weighted_support_score
        
        # Update consensus status based on score
        if weighted_support_score >= 0.95:
            process.current_consensus_level = ConsensusStatus.UNANIMOUS
        elif weighted_support_score >= 0.8:
            process.current_consensus_level = ConsensusStatus.STRONG_CONSENSUS
        elif weighted_support_score >= 0.6:
            process.current_consensus_level = ConsensusStatus.PARTIAL_CONSENSUS
        elif weighted_support_score >= 0.3:
            process.current_consensus_level = ConsensusStatus.IN_PROGRESS
        else:
            process.current_consensus_level = ConsensusStatus.BLOCKED
        
        self._update_process_timestamp(process_id)
        
        logger.info(f"Tracked consensus progress: {support_percentage:.1f}% support, {weighted_support_score:.2f} weighted score")
        return metrics
    
    def generate_consensus_recommendation(self, process_id: str) -> ConsensusRecommendation:
        """Generate recommendations for achieving consensus"""
        
        if process_id not in self.consensus_processes:
            raise ValueError(f"Consensus process {process_id} not found")
        
        process = self.consensus_processes[process_id]
        recommendation_id = str(uuid.uuid4())
        
        # Analyze current situation
        metrics = self.track_consensus_progress(process_id)
        
        # Determine recommended approach based on current state
        if metrics.support_percentage >= 70:
            recommended_approach = ConsensusStrategyType.COALITION_BUILDING
            approach_description = "Leverage existing support to build momentum"
        elif metrics.opposition_percentage >= 50:
            recommended_approach = ConsensusStrategyType.COMPROMISE_SEEKING
            approach_description = "Seek compromise to address opposition concerns"
        elif metrics.neutral_percentage >= 50:
            recommended_approach = ConsensusStrategyType.INFORMATION_SHARING
            approach_description = "Provide information to help undecided stakeholders"
        else:
            recommended_approach = ConsensusStrategyType.STAKEHOLDER_ENGAGEMENT
            approach_description = "Engage directly with key stakeholders"
        
        # Generate priority actions
        priority_actions = []
        
        if process.barriers:
            priority_actions.append(f"Address top barrier: {process.barriers[0].description}")
        
        if metrics.neutral_percentage > 30:
            priority_actions.append("Focus on converting neutral stakeholders")
        
        if metrics.trust_level < 0.7:
            priority_actions.append("Build trust through transparency and engagement")
        
        if not priority_actions:
            priority_actions = ["Continue current consensus building efforts"]
        
        # Identify key stakeholders to focus on
        high_influence_undecided = []
        for pos in process.stakeholder_positions:
            if pos.current_position in [StakeholderPosition.NEUTRAL, StakeholderPosition.UNDECIDED]:
                member = next((m for m in process.board_members if m.id == pos.stakeholder_id), None)
                if member and member.influence_level in [InfluenceLevel.HIGH, InfluenceLevel.CRITICAL]:
                    high_influence_undecided.append(pos.stakeholder_name)
        
        key_stakeholders = high_influence_undecided[:3] if high_influence_undecided else ["All stakeholders"]
        
        # Generate timeline recommendation
        if process.deadline:
            days_remaining = (process.deadline - datetime.now()).days
            if days_remaining < 14:
                timeline_recommendation = "Accelerated timeline required - focus on quick wins"
            elif days_remaining < 30:
                timeline_recommendation = "Standard timeline - maintain current pace"
            else:
                timeline_recommendation = "Extended timeline available - thorough consensus building"
        else:
            timeline_recommendation = "4-6 weeks for comprehensive consensus building"
        
        # Generate communication strategy
        analytical_members = len([m for m in process.board_members if m.decision_making_style == "analytical"])
        total_members = len(process.board_members)
        
        if analytical_members > total_members * 0.6:
            communication_strategy = "Data-driven approach with detailed analysis and evidence"
        else:
            communication_strategy = "Balanced approach combining data, stories, and relationship building"
        
        # Generate meeting recommendations
        meeting_recommendations = [
            "Schedule one-on-one meetings with key undecided stakeholders",
            "Organize group discussion sessions to address common concerns",
            "Provide pre-meeting materials with key information"
        ]
        
        if metrics.opposition_percentage > 30:
            meeting_recommendations.append("Hold separate sessions with opposing stakeholders")
        
        # Generate negotiation points and compromises
        negotiation_points = []
        compromise_options = []
        
        for pos in process.stakeholder_positions:
            if pos.requirements_for_support:
                negotiation_points.extend(pos.requirements_for_support[:2])
            if pos.deal_breakers:
                compromise_options.append(f"Address deal-breaker: {pos.deal_breakers[0]}")
        
        if not negotiation_points:
            negotiation_points = ["Timeline flexibility", "Implementation approach", "Resource allocation"]
        
        if not compromise_options:
            compromise_options = ["Phased implementation", "Pilot program approach", "Enhanced oversight"]
        
        # Calculate success probability
        base_probability = metrics.weighted_support_score
        barrier_penalty = len(process.barriers) * 0.05
        strategy_bonus = len(process.strategies) * 0.02
        
        success_probability = max(0.1, min(0.95, base_probability - barrier_penalty + strategy_bonus))
        
        recommendation = ConsensusRecommendation(
            id=recommendation_id,
            consensus_building_id=process_id,
            title=f"Consensus Building Recommendation: {approach_description}",
            description=f"Strategic approach to achieve {process.target_consensus_level.value} consensus",
            recommended_approach=recommended_approach,
            priority_actions=priority_actions,
            key_stakeholders_to_focus=key_stakeholders,
            timeline_recommendation=timeline_recommendation,
            communication_strategy=communication_strategy,
            meeting_recommendations=meeting_recommendations,
            negotiation_points=list(set(negotiation_points))[:5],  # Remove duplicates, limit to 5
            compromise_options=list(set(compromise_options))[:3],   # Remove duplicates, limit to 3
            potential_risks=[
                "Stakeholder fatigue from extended process",
                "External factors changing stakeholder positions",
                "New concerns emerging during process"
            ],
            mitigation_strategies=[
                "Maintain regular communication and updates",
                "Monitor external environment for changes",
                "Build flexibility into consensus approach"
            ],
            contingency_plans=[
                "Escalate to higher authority if consensus fails",
                "Implement decision with majority support",
                "Modify proposal based on feedback"
            ],
            success_probability=success_probability,
            critical_success_factors=[
                "Strong leadership commitment",
                "Transparent communication",
                "Addressing key stakeholder concerns"
            ],
            early_warning_indicators=[
                "Declining stakeholder engagement",
                "Increasing opposition",
                "New barriers emerging"
            ]
        )
        
        logger.info(f"Generated consensus recommendation with {success_probability:.1%} success probability")
        return recommendation
    
    def optimize_consensus_process(self, process_id: str) -> ConsensusOptimization:
        """Generate optimization recommendations for consensus building"""
        
        if process_id not in self.consensus_processes:
            raise ValueError(f"Consensus process {process_id} not found")
        
        process = self.consensus_processes[process_id]
        optimization_id = str(uuid.uuid4())
        
        # Analyze current process for optimization opportunities
        metrics = self.track_consensus_progress(process_id)
        
        # Process improvements
        process_improvements = []
        if metrics.stakeholder_engagement_level < 0.7:
            process_improvements.append("Increase stakeholder engagement through more interactive sessions")
        
        if len(process.actions) < 3:
            process_improvements.append("Develop more specific action items with clear owners")
        
        if not process.deadline:
            process_improvements.append("Establish clear timeline and milestones")
        
        process_improvements.append("Implement regular progress check-ins")
        
        # Communication enhancements
        communication_enhancements = []
        if metrics.communication_effectiveness < 0.8:
            communication_enhancements.extend([
                "Tailor communication style to individual stakeholder preferences",
                "Use multiple communication channels (meetings, documents, informal discussions)",
                "Provide regular progress updates to all stakeholders"
            ])
        
        # Engagement strategies
        engagement_strategies = [
            "Create opportunities for stakeholder input and feedback",
            "Recognize and acknowledge stakeholder contributions",
            "Provide clear roles and responsibilities for each stakeholder"
        ]
        
        # Stakeholder-specific approaches
        stakeholder_approaches = {}
        for member in process.board_members:
            approaches = []
            
            if member.decision_making_style == "analytical":
                approaches.extend([
                    "Provide detailed data and analysis",
                    "Present logical arguments with evidence",
                    "Allow time for thorough review"
                ])
            elif member.decision_making_style == "intuitive":
                approaches.extend([
                    "Share vision and strategic narrative",
                    "Discuss broader implications and opportunities",
                    "Use storytelling and examples"
                ])
            elif member.decision_making_style == "collaborative":
                approaches.extend([
                    "Involve in solution development",
                    "Seek input and feedback regularly",
                    "Emphasize team decision making"
                ])
            
            if member.influence_level in [InfluenceLevel.HIGH, InfluenceLevel.CRITICAL]:
                approaches.append("Prioritize one-on-one engagement")
            
            stakeholder_approaches[member.id] = approaches
        
        # Coalition building opportunities
        coalition_opportunities = []
        supporters = [pos for pos in process.stakeholder_positions 
                     if pos.current_position in [StakeholderPosition.SUPPORT, StakeholderPosition.STRONGLY_SUPPORT]]
        
        if len(supporters) >= 2:
            coalition_opportunities.append("Form coalition of existing supporters to influence others")
        
        # Find stakeholders with similar concerns for potential alignment
        concern_groups = {}
        for pos in process.stakeholder_positions:
            for concern in pos.key_concerns:
                if concern not in concern_groups:
                    concern_groups[concern] = []
                concern_groups[concern].append(pos.stakeholder_name)
        
        for concern, stakeholders in concern_groups.items():
            if len(stakeholders) > 1:
                coalition_opportunities.append(f"Address shared concern '{concern}' with group: {', '.join(stakeholders)}")
        
        # Influence leverage points
        influence_points = []
        for member in process.board_members:
            if member.influence_level in [InfluenceLevel.HIGH, InfluenceLevel.CRITICAL]:
                pos = next((p for p in process.stakeholder_positions if p.stakeholder_id == member.id), None)
                if pos and pos.current_position in [StakeholderPosition.NEUTRAL, StakeholderPosition.UNDECIDED]:
                    influence_points.append(f"Convert high-influence stakeholder: {member.name}")
        
        # Timeline optimizations
        timeline_optimizations = [
            "Run parallel workstreams for different stakeholder groups",
            "Pre-socialize key concepts before formal meetings",
            "Use asynchronous communication for information sharing"
        ]
        
        if process.deadline and (process.deadline - datetime.now()).days < 30:
            timeline_optimizations.extend([
                "Focus on quick wins and essential consensus points",
                "Streamline decision process while maintaining quality"
            ])
        
        # Quick wins
        quick_wins = []
        easy_supporters = [pos for pos in process.stakeholder_positions 
                          if pos.current_position == StakeholderPosition.NEUTRAL 
                          and pos.confidence_level < 0.6]
        
        if easy_supporters:
            quick_wins.append("Target low-confidence neutral stakeholders for quick conversion")
        
        if process.barriers:
            info_barriers = [b for b in process.barriers if b.barrier_type == "information"]
            if info_barriers:
                quick_wins.append("Address information barriers with targeted briefings")
        
        quick_wins.append("Celebrate early agreements and build momentum")
        
        # Quality improvements
        quality_improvements = [
            "Ensure all stakeholder concerns are documented and addressed",
            "Validate understanding through feedback loops",
            "Maintain decision audit trail for transparency"
        ]
        
        optimization = ConsensusOptimization(
            id=optimization_id,
            consensus_building_id=process_id,
            process_improvements=process_improvements,
            communication_enhancements=communication_enhancements,
            engagement_strategies=engagement_strategies,
            stakeholder_specific_approaches=stakeholder_approaches,
            coalition_building_opportunities=coalition_opportunities,
            influence_leverage_points=influence_points,
            accelerated_timeline_options=timeline_optimizations,
            parallel_workstream_opportunities=[
                "Separate technical and business discussions",
                "Run stakeholder engagement in parallel with information gathering"
            ],
            quick_wins=quick_wins,
            decision_quality_enhancements=quality_improvements,
            information_gaps_to_address=[
                barrier.description for barrier in process.barriers 
                if barrier.barrier_type == "information"
            ][:3],
            expertise_to_bring_in=[
                "Subject matter experts for technical questions",
                "External benchmarking data",
                "Implementation experience from similar decisions"
            ]
        )
        
        logger.info(f"Generated consensus optimization with {len(process_improvements)} process improvements")
        return optimization
    
    def _calculate_consensus_score(self, process_id: str):
        """Calculate and update consensus score"""
        if process_id not in self.consensus_processes:
            return
        
        process = self.consensus_processes[process_id]
        if not process.stakeholder_positions:
            process.consensus_score = 0.0
            return
        
        # Calculate weighted consensus score
        total_weight = 0.0
        weighted_score = 0.0
        
        for pos in process.stakeholder_positions:
            member = next((m for m in process.board_members if m.id == pos.stakeholder_id), None)
            weight = 1.0
            
            if member:
                weight_map = {
                    InfluenceLevel.LOW: 0.5,
                    InfluenceLevel.MODERATE: 1.0,
                    InfluenceLevel.HIGH: 2.0,
                    InfluenceLevel.CRITICAL: 3.0
                }
                weight = weight_map.get(member.influence_level, 1.0)
            
            position_score = {
                StakeholderPosition.STRONGLY_SUPPORT: 1.0,
                StakeholderPosition.SUPPORT: 0.75,
                StakeholderPosition.NEUTRAL: 0.5,
                StakeholderPosition.UNDECIDED: 0.5,
                StakeholderPosition.OPPOSE: 0.25,
                StakeholderPosition.STRONGLY_OPPOSE: 0.0
            }.get(pos.current_position, 0.5)
            
            weighted_score += position_score * weight
            total_weight += weight
        
        process.consensus_score = (weighted_score / total_weight) if total_weight > 0 else 0.0
    
    def _update_coalition_map(self, process_id: str):
        """Update coalition mapping based on current positions"""
        if process_id not in self.consensus_processes:
            return
        
        process = self.consensus_processes[process_id]
        coalition_map = {}
        
        for pos in process.stakeholder_positions:
            position_key = pos.current_position.value
            if position_key not in coalition_map:
                coalition_map[position_key] = []
            coalition_map[position_key].append(pos.stakeholder_id)
        
        process.coalition_map = coalition_map
    
    def _update_process_timestamp(self, process_id: str):
        """Update process last updated timestamp"""
        if process_id in self.consensus_processes:
            self.consensus_processes[process_id].last_updated = datetime.now()
    
    def get_consensus_process(self, process_id: str) -> Optional[ConsensusBuilding]:
        """Get consensus building process by ID"""
        return self.consensus_processes.get(process_id)
    
    def list_consensus_processes(self) -> List[ConsensusBuilding]:
        """List all consensus building processes"""
        return list(self.consensus_processes.values())
    
    def create_consensus_visualization(
        self,
        process_id: str,
        visualization_type: str = "stakeholder_map"
    ) -> ConsensusVisualization:
        """Create visualization for consensus building process"""
        
        if process_id not in self.consensus_processes:
            raise ValueError(f"Consensus process {process_id} not found")
        
        process = self.consensus_processes[process_id]
        viz_id = str(uuid.uuid4())
        
        chart_config = {}
        title = ""
        description = ""
        executive_summary = ""
        
        if visualization_type == "stakeholder_map":
            title = "Stakeholder Position and Influence Map"
            description = "Visual representation of stakeholder positions and influence levels"
            
            stakeholder_data = []
            for pos in process.stakeholder_positions:
                member = next((m for m in process.board_members if m.id == pos.stakeholder_id), None)
                influence_numeric = 1.0
                
                if member:
                    influence_map = {
                        InfluenceLevel.LOW: 0.25,
                        InfluenceLevel.MODERATE: 0.5,
                        InfluenceLevel.HIGH: 0.75,
                        InfluenceLevel.CRITICAL: 1.0
                    }
                    influence_numeric = influence_map.get(member.influence_level, 0.5)
                
                position_numeric = {
                    StakeholderPosition.STRONGLY_OPPOSE: 0.0,
                    StakeholderPosition.OPPOSE: 0.25,
                    StakeholderPosition.NEUTRAL: 0.5,
                    StakeholderPosition.UNDECIDED: 0.5,
                    StakeholderPosition.SUPPORT: 0.75,
                    StakeholderPosition.STRONGLY_SUPPORT: 1.0
                }.get(pos.current_position, 0.5)
                
                stakeholder_data.append({
                    "name": pos.stakeholder_name,
                    "position": position_numeric,
                    "influence": influence_numeric,
                    "confidence": pos.confidence_level,
                    "concerns": len(pos.key_concerns)
                })
            
            chart_config = {
                "type": "scatter",
                "data": stakeholder_data,
                "x_axis": "position",
                "y_axis": "influence",
                "bubble_size": "confidence"
            }
            
            executive_summary = f"Stakeholder analysis of {len(process.stakeholder_positions)} board members"
        
        elif visualization_type == "consensus_timeline":
            title = "Consensus Building Timeline"
            description = "Progress tracking over time"
            
            # This would typically show historical data
            timeline_data = [{
                "date": process.created_at.isoformat(),
                "consensus_score": 0.0,
                "support_percentage": 0.0
            }]
            
            if process.consensus_score > 0:
                timeline_data.append({
                    "date": process.last_updated.isoformat(),
                    "consensus_score": process.consensus_score,
                    "support_percentage": len([pos for pos in process.stakeholder_positions 
                                             if pos.current_position in [StakeholderPosition.SUPPORT, StakeholderPosition.STRONGLY_SUPPORT]]) / len(process.stakeholder_positions) * 100 if process.stakeholder_positions else 0
                })
            
            chart_config = {
                "type": "line",
                "data": timeline_data,
                "x_axis": "date",
                "y_axis": "consensus_score"
            }
            
            executive_summary = f"Consensus progress from {process.created_at.strftime('%Y-%m-%d')} to present"
        
        elif visualization_type == "influence_network":
            title = "Board Member Influence Network"
            description = "Relationship and influence connections between board members"
            
            network_data = []
            for member in process.board_members:
                network_data.append({
                    "id": member.id,
                    "name": member.name,
                    "influence": member.influence_level.value,
                    "connections": member.relationship_network
                })
            
            chart_config = {
                "type": "network",
                "data": network_data,
                "node_size": "influence",
                "connections": "relationship_network"
            }
            
            executive_summary = f"Influence network analysis of {len(process.board_members)} board members"
        
        visualization = ConsensusVisualization(
            id=viz_id,
            consensus_building_id=process_id,
            visualization_type=visualization_type,
            title=title,
            description=description,
            chart_config=chart_config,
            executive_summary=executive_summary
        )
        
        logger.info(f"Created {visualization_type} visualization for consensus process {process_id}")
        return visualization