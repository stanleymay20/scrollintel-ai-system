"""
Habit Formation Engine for Cultural Transformation Leadership

This engine creates new positive organizational habits, builds habit formation
strategies and implementation frameworks, and implements sustainability and
reinforcement mechanisms.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from uuid import uuid4

from ..models.habit_formation_models import (
    OrganizationalHabit, HabitFormationStrategy, HabitImplementation,
    HabitSustainability, HabitProgress, HabitReinforcementMechanism,
    HabitFormationPlan, HabitFormationMetrics, HabitOptimization,
    HabitType, HabitFrequency, HabitStage, SustainabilityLevel
)

logger = logging.getLogger(__name__)


class HabitFormationEngine:
    """Engine for organizational habit formation and sustainability"""
    
    def __init__(self):
        self.habits = {}
        self.formation_strategies = {}
        self.implementations = {}
        self.sustainability_mechanisms = {}
        self.progress_tracking = {}
        self.reinforcement_mechanisms = {}
        self.formation_plans = {}
        
    def design_organizational_habit(
        self,
        name: str,
        description: str,
        habit_type: HabitType,
        target_behavior: str,
        participants: List[str],
        cultural_values: List[str] = None,
        business_objectives: List[str] = None
    ) -> OrganizationalHabit:
        """
        Create new positive organizational habit identification and design
        
        Requirements: 3.3, 3.4 - Create new positive organizational habit identification and design
        """
        try:
            habit_id = str(uuid4())
            cultural_values = cultural_values or []
            business_objectives = business_objectives or []
            
            # Analyze habit type and generate appropriate components
            trigger_conditions = self._generate_trigger_conditions(habit_type, target_behavior)
            execution_steps = self._generate_execution_steps(habit_type, target_behavior)
            success_indicators = self._generate_success_indicators(habit_type, target_behavior)
            
            # Determine optimal frequency
            frequency = self._determine_optimal_frequency(habit_type, target_behavior)
            
            # Estimate duration
            duration_minutes = self._estimate_habit_duration(habit_type, len(participants))
            
            # Identify required resources
            resources_required = self._identify_habit_resources(habit_type, participants)
            
            # Calculate cultural alignment
            cultural_alignment = self._calculate_cultural_alignment(
                habit_type, target_behavior, cultural_values
            )
            
            # Assess business impact
            business_impact = self._assess_business_impact(
                habit_type, target_behavior, business_objectives
            )
            
            # Identify facilitators
            facilitators = self._identify_facilitators(habit_type, participants)
            
            habit = OrganizationalHabit(
                id=habit_id,
                name=name,
                description=description,
                habit_type=habit_type,
                target_behavior=target_behavior,
                trigger_conditions=trigger_conditions,
                execution_steps=execution_steps,
                success_indicators=success_indicators,
                frequency=frequency,
                duration_minutes=duration_minutes,
                participants=participants,
                facilitators=facilitators,
                resources_required=resources_required,
                cultural_alignment=cultural_alignment,
                business_impact=business_impact,
                stage=HabitStage.DESIGN,
                created_date=datetime.now(),
                created_by="ScrollIntel"
            )
            
            self.habits[habit_id] = habit
            logger.info(f"Designed organizational habit: {name}")
            
            return habit
            
        except Exception as e:
            logger.error(f"Error designing organizational habit: {str(e)}")
            raise
    
    def _generate_trigger_conditions(
        self,
        habit_type: HabitType,
        target_behavior: str
    ) -> List[str]:
        """Generate appropriate trigger conditions for habit type"""
        
        base_triggers = {
            HabitType.COMMUNICATION: [
                "start_of_team_meeting",
                "project_status_update",
                "weekly_standup",
                "conflict_detection"
            ],
            HabitType.COLLABORATION: [
                "new_project_initiation",
                "cross_team_interaction",
                "problem_solving_session",
                "knowledge_sharing_opportunity"
            ],
            HabitType.LEARNING: [
                "end_of_project",
                "monthly_review",
                "skill_gap_identification",
                "new_technology_introduction"
            ],
            HabitType.INNOVATION: [
                "brainstorming_session",
                "process_improvement_opportunity",
                "customer_feedback_review",
                "quarterly_innovation_time"
            ],
            HabitType.QUALITY: [
                "deliverable_completion",
                "code_review",
                "quality_checkpoint",
                "customer_delivery"
            ],
            HabitType.FEEDBACK: [
                "task_completion",
                "milestone_achievement",
                "performance_review_period",
                "team_interaction"
            ],
            HabitType.RECOGNITION: [
                "achievement_milestone",
                "exceptional_performance",
                "team_success",
                "individual_contribution"
            ],
            HabitType.PLANNING: [
                "sprint_planning",
                "project_kickoff",
                "weekly_planning_session",
                "goal_setting_period"
            ],
            HabitType.REFLECTION: [
                "end_of_sprint",
                "project_completion",
                "monthly_retrospective",
                "learning_opportunity"
            ],
            HabitType.WELLNESS: [
                "workday_start",
                "lunch_break",
                "high_stress_period",
                "team_building_opportunity"
            ]
        }
        
        triggers = base_triggers.get(habit_type, ["daily_routine", "weekly_check_in"])
        
        # Add behavior-specific triggers
        if "communication" in target_behavior.lower():
            triggers.append("communication_need_identified")
        if "collaboration" in target_behavior.lower():
            triggers.append("collaborative_opportunity")
        if "improvement" in target_behavior.lower():
            triggers.append("improvement_opportunity_identified")
        
        return triggers[:4]  # Limit to 4 triggers for clarity
    
    def _generate_execution_steps(
        self,
        habit_type: HabitType,
        target_behavior: str
    ) -> List[str]:
        """Generate execution steps for the habit"""
        
        base_steps = {
            HabitType.COMMUNICATION: [
                "Identify communication need or opportunity",
                "Choose appropriate communication method",
                "Prepare clear and concise message",
                "Deliver communication effectively",
                "Confirm understanding and gather feedback"
            ],
            HabitType.COLLABORATION: [
                "Identify collaboration opportunity",
                "Reach out to relevant team members",
                "Establish collaboration framework",
                "Execute collaborative activities",
                "Document outcomes and learnings"
            ],
            HabitType.LEARNING: [
                "Identify learning opportunity or need",
                "Allocate dedicated time for learning",
                "Engage with learning materials or activities",
                "Apply new knowledge or skills",
                "Share learnings with team"
            ],
            HabitType.INNOVATION: [
                "Identify innovation opportunity",
                "Generate creative ideas or solutions",
                "Evaluate feasibility and impact",
                "Develop prototype or pilot",
                "Share results and iterate"
            ],
            HabitType.QUALITY: [
                "Review quality standards and requirements",
                "Conduct thorough quality check",
                "Document quality assessment",
                "Address any quality issues",
                "Confirm quality standards met"
            ],
            HabitType.FEEDBACK: [
                "Identify feedback opportunity",
                "Prepare constructive feedback",
                "Deliver feedback in appropriate setting",
                "Ensure understanding and acceptance",
                "Follow up on feedback implementation"
            ],
            HabitType.RECOGNITION: [
                "Identify achievement or contribution",
                "Choose appropriate recognition method",
                "Deliver recognition publicly or privately",
                "Document recognition for future reference",
                "Encourage continued excellence"
            ],
            HabitType.PLANNING: [
                "Review current status and objectives",
                "Identify priorities and dependencies",
                "Create detailed action plan",
                "Allocate resources and timeline",
                "Communicate plan to stakeholders"
            ],
            HabitType.REFLECTION: [
                "Set aside dedicated reflection time",
                "Review recent activities and outcomes",
                "Identify successes and challenges",
                "Extract key learnings and insights",
                "Plan improvements for future"
            ],
            HabitType.WELLNESS: [
                "Assess current wellness state",
                "Engage in wellness activity",
                "Monitor impact on well-being",
                "Adjust activity based on needs",
                "Encourage team wellness participation"
            ]
        }
        
        return base_steps.get(habit_type, [
            "Identify opportunity for habit execution",
            "Prepare for habit activity",
            "Execute habit according to plan",
            "Monitor execution quality",
            "Document results and improvements"
        ])
    
    def _generate_success_indicators(
        self,
        habit_type: HabitType,
        target_behavior: str
    ) -> List[str]:
        """Generate success indicators for the habit"""
        
        base_indicators = {
            HabitType.COMMUNICATION: [
                "Clear message delivery",
                "Recipient understanding confirmed",
                "Reduced communication conflicts",
                "Improved information flow"
            ],
            HabitType.COLLABORATION: [
                "Increased cross-team interactions",
                "Successful collaborative outcomes",
                "Enhanced team cohesion",
                "Knowledge sharing frequency"
            ],
            HabitType.LEARNING: [
                "New skills or knowledge acquired",
                "Learning applied to work",
                "Knowledge shared with others",
                "Continuous improvement demonstrated"
            ],
            HabitType.INNOVATION: [
                "Creative ideas generated",
                "Innovation initiatives launched",
                "Process improvements implemented",
                "Novel solutions developed"
            ],
            HabitType.QUALITY: [
                "Quality standards consistently met",
                "Defect rates reduced",
                "Customer satisfaction improved",
                "Quality processes followed"
            ],
            HabitType.FEEDBACK: [
                "Regular feedback provided",
                "Constructive feedback received",
                "Performance improvements observed",
                "Feedback culture strengthened"
            ],
            HabitType.RECOGNITION: [
                "Achievements acknowledged",
                "Team morale improved",
                "Recognition frequency increased",
                "Positive culture reinforced"
            ],
            HabitType.PLANNING: [
                "Clear plans developed",
                "Goals and priorities defined",
                "Resource allocation optimized",
                "Planning consistency maintained"
            ],
            HabitType.REFLECTION: [
                "Regular reflection sessions held",
                "Key insights identified",
                "Improvements implemented",
                "Learning culture fostered"
            ],
            HabitType.WELLNESS: [
                "Wellness activities completed",
                "Stress levels managed",
                "Team wellness improved",
                "Work-life balance enhanced"
            ]
        }
        
        return base_indicators.get(habit_type, [
            "Habit executed consistently",
            "Target behavior demonstrated",
            "Positive outcomes achieved",
            "Habit sustainability maintained"
        ])
    
    def _determine_optimal_frequency(
        self,
        habit_type: HabitType,
        target_behavior: str
    ) -> HabitFrequency:
        """Determine optimal frequency for habit execution"""
        
        frequency_mapping = {
            HabitType.COMMUNICATION: HabitFrequency.DAILY,
            HabitType.COLLABORATION: HabitFrequency.WEEKLY,
            HabitType.LEARNING: HabitFrequency.WEEKLY,
            HabitType.INNOVATION: HabitFrequency.BI_WEEKLY,
            HabitType.QUALITY: HabitFrequency.DAILY,
            HabitType.FEEDBACK: HabitFrequency.WEEKLY,
            HabitType.RECOGNITION: HabitFrequency.WEEKLY,
            HabitType.PLANNING: HabitFrequency.WEEKLY,
            HabitType.REFLECTION: HabitFrequency.WEEKLY,
            HabitType.WELLNESS: HabitFrequency.DAILY
        }
        
        # Adjust based on target behavior
        base_frequency = frequency_mapping.get(habit_type, HabitFrequency.WEEKLY)
        
        if "daily" in target_behavior.lower():
            return HabitFrequency.DAILY
        elif "weekly" in target_behavior.lower():
            return HabitFrequency.WEEKLY
        elif "monthly" in target_behavior.lower():
            return HabitFrequency.MONTHLY
        
        return base_frequency
    
    def _estimate_habit_duration(
        self,
        habit_type: HabitType,
        participant_count: int
    ) -> int:
        """Estimate duration in minutes for habit execution"""
        
        base_durations = {
            HabitType.COMMUNICATION: 15,
            HabitType.COLLABORATION: 30,
            HabitType.LEARNING: 45,
            HabitType.INNOVATION: 60,
            HabitType.QUALITY: 20,
            HabitType.FEEDBACK: 15,
            HabitType.RECOGNITION: 10,
            HabitType.PLANNING: 30,
            HabitType.REFLECTION: 25,
            HabitType.WELLNESS: 15
        }
        
        base_duration = base_durations.get(habit_type, 30)
        
        # Adjust for participant count
        if participant_count > 10:
            base_duration = int(base_duration * 1.5)
        elif participant_count > 20:
            base_duration = int(base_duration * 2.0)
        
        return min(120, max(5, base_duration))  # Between 5 and 120 minutes
    
    def _identify_habit_resources(
        self,
        habit_type: HabitType,
        participants: List[str]
    ) -> List[str]:
        """Identify resources required for habit execution"""
        
        base_resources = {
            HabitType.COMMUNICATION: [
                "communication_platform",
                "meeting_space",
                "documentation_tools"
            ],
            HabitType.COLLABORATION: [
                "collaboration_tools",
                "shared_workspace",
                "project_management_system"
            ],
            HabitType.LEARNING: [
                "learning_materials",
                "training_resources",
                "knowledge_repository"
            ],
            HabitType.INNOVATION: [
                "brainstorming_tools",
                "prototyping_resources",
                "innovation_time_allocation"
            ],
            HabitType.QUALITY: [
                "quality_checklists",
                "review_templates",
                "quality_metrics_dashboard"
            ],
            HabitType.FEEDBACK: [
                "feedback_forms",
                "private_meeting_space",
                "feedback_tracking_system"
            ],
            HabitType.RECOGNITION: [
                "recognition_platform",
                "awards_budget",
                "communication_channels"
            ],
            HabitType.PLANNING: [
                "planning_templates",
                "goal_tracking_system",
                "resource_allocation_tools"
            ],
            HabitType.REFLECTION: [
                "reflection_templates",
                "quiet_space",
                "documentation_system"
            ],
            HabitType.WELLNESS: [
                "wellness_activities",
                "relaxation_space",
                "wellness_tracking_tools"
            ]
        }
        
        resources = base_resources.get(habit_type, ["basic_tools", "time_allocation"])
        
        # Add common resources
        resources.extend([
            "participant_time",
            "facilitator_support",
            "progress_tracking_system"
        ])
        
        return list(set(resources))  # Remove duplicates
    
    def _calculate_cultural_alignment(
        self,
        habit_type: HabitType,
        target_behavior: str,
        cultural_values: List[str]
    ) -> float:
        """Calculate alignment with cultural values"""
        
        if not cultural_values:
            return 0.7  # Default moderate alignment
        
        # Define alignment scores for habit types with common cultural values
        alignment_matrix = {
            HabitType.COMMUNICATION: {
                'transparency': 0.9,
                'openness': 0.9,
                'collaboration': 0.8,
                'respect': 0.8,
                'trust': 0.7
            },
            HabitType.COLLABORATION: {
                'teamwork': 0.9,
                'collaboration': 0.9,
                'unity': 0.8,
                'cooperation': 0.9,
                'partnership': 0.8
            },
            HabitType.LEARNING: {
                'growth': 0.9,
                'development': 0.9,
                'curiosity': 0.8,
                'innovation': 0.7,
                'excellence': 0.8
            },
            HabitType.INNOVATION: {
                'creativity': 0.9,
                'innovation': 0.9,
                'experimentation': 0.8,
                'risk_taking': 0.7,
                'entrepreneurship': 0.8
            },
            HabitType.QUALITY: {
                'excellence': 0.9,
                'quality': 0.9,
                'precision': 0.8,
                'reliability': 0.8,
                'professionalism': 0.7
            }
        }
        
        habit_alignments = alignment_matrix.get(habit_type, {})
        
        # Calculate alignment score
        total_alignment = 0.0
        matched_values = 0
        
        for value in cultural_values:
            value_lower = value.lower()
            for cultural_key, alignment_score in habit_alignments.items():
                if cultural_key in value_lower or value_lower in cultural_key:
                    total_alignment += alignment_score
                    matched_values += 1
                    break
        
        if matched_values == 0:
            return 0.5  # Neutral alignment if no matches
        
        return min(1.0, total_alignment / matched_values)
    
    def _assess_business_impact(
        self,
        habit_type: HabitType,
        target_behavior: str,
        business_objectives: List[str]
    ) -> str:
        """Assess business impact of the habit"""
        
        impact_descriptions = {
            HabitType.COMMUNICATION: "Improves information flow, reduces misunderstandings, enhances team coordination",
            HabitType.COLLABORATION: "Increases team effectiveness, accelerates problem-solving, improves innovation",
            HabitType.LEARNING: "Builds organizational capability, improves adaptability, drives continuous improvement",
            HabitType.INNOVATION: "Generates new ideas, improves processes, creates competitive advantages",
            HabitType.QUALITY: "Reduces defects, improves customer satisfaction, enhances reputation",
            HabitType.FEEDBACK: "Accelerates performance improvement, builds trust, enhances development",
            HabitType.RECOGNITION: "Improves morale, increases retention, reinforces positive behaviors",
            HabitType.PLANNING: "Improves execution, reduces waste, enhances predictability",
            HabitType.REFLECTION: "Accelerates learning, improves decision-making, builds wisdom",
            HabitType.WELLNESS: "Reduces burnout, improves productivity, enhances sustainability"
        }
        
        base_impact = impact_descriptions.get(habit_type, "Supports organizational effectiveness and culture")
        
        # Enhance with business objective alignment
        if business_objectives:
            objective_keywords = [obj.lower() for obj in business_objectives]
            if any(keyword in base_impact.lower() for keyword in objective_keywords):
                base_impact += " - Directly supports key business objectives"
        
        return base_impact
    
    def _identify_facilitators(
        self,
        habit_type: HabitType,
        participants: List[str]
    ) -> List[str]:
        """Identify appropriate facilitators for the habit"""
        
        facilitator_roles = {
            HabitType.COMMUNICATION: ["communication_lead", "team_lead"],
            HabitType.COLLABORATION: ["project_manager", "team_lead"],
            HabitType.LEARNING: ["learning_coordinator", "subject_matter_expert"],
            HabitType.INNOVATION: ["innovation_champion", "creative_lead"],
            HabitType.QUALITY: ["quality_manager", "process_owner"],
            HabitType.FEEDBACK: ["manager", "hr_representative"],
            HabitType.RECOGNITION: ["team_lead", "hr_representative"],
            HabitType.PLANNING: ["project_manager", "team_lead"],
            HabitType.REFLECTION: ["coach", "team_lead"],
            HabitType.WELLNESS: ["wellness_coordinator", "hr_representative"]
        }
        
        suggested_roles = facilitator_roles.get(habit_type, ["team_lead", "manager"])
        
        # For small teams, team members can facilitate
        if len(participants) <= 5:
            suggested_roles.append("rotating_team_member")
        
        return suggested_roles[:2]  # Limit to 2 facilitator types
    
    def create_habit_formation_strategy(
        self,
        habit: OrganizationalHabit,
        organizational_context: Dict[str, Any] = None
    ) -> HabitFormationStrategy:
        """
        Build habit formation strategy and implementation framework
        
        Requirements: 3.3, 3.4 - Build habit formation strategy and implementation framework
        """
        try:
            strategy_id = str(uuid4())
            organizational_context = organizational_context or {}
            
            # Define formation phases
            formation_phases = self._define_formation_phases(habit)
            
            # Estimate timeline
            timeline_weeks = self._estimate_formation_timeline(habit, organizational_context)
            
            # Define key milestones
            key_milestones = self._define_formation_milestones(habit, timeline_weeks)
            
            # Define success metrics
            success_metrics = self._define_formation_success_metrics(habit)
            
            # Design reinforcement mechanisms
            reinforcement_mechanisms = self._design_reinforcement_mechanisms(habit)
            
            # Identify barriers and mitigation strategies
            barrier_mitigation = self._identify_barrier_mitigation(habit, organizational_context)
            
            # Plan stakeholder engagement
            stakeholder_engagement = self._plan_stakeholder_engagement(habit)
            
            # Allocate resources
            resource_allocation = self._allocate_formation_resources(habit, timeline_weeks)
            
            # Assess risks
            risk_assessment = self._assess_formation_risks(habit, organizational_context)
            
            strategy = HabitFormationStrategy(
                id=strategy_id,
                habit_id=habit.id,
                strategy_name=f"Formation Strategy: {habit.name}",
                description=f"Comprehensive strategy to establish {habit.name} as organizational habit",
                formation_phases=formation_phases,
                timeline_weeks=timeline_weeks,
                key_milestones=key_milestones,
                success_metrics=success_metrics,
                reinforcement_mechanisms=reinforcement_mechanisms,
                barrier_mitigation=barrier_mitigation,
                stakeholder_engagement=stakeholder_engagement,
                resource_allocation=resource_allocation,
                risk_assessment=risk_assessment,
                created_date=datetime.now()
            )
            
            self.formation_strategies[strategy_id] = strategy
            logger.info(f"Created habit formation strategy for {habit.name}")
            
            return strategy
            
        except Exception as e:
            logger.error(f"Error creating habit formation strategy: {str(e)}")
            raise
    
    def _define_formation_phases(self, habit: OrganizationalHabit) -> List[str]:
        """Define phases for habit formation"""
        
        return [
            "Awareness and Buy-in",
            "Initial Implementation",
            "Consistency Building",
            "Reinforcement and Optimization",
            "Institutionalization"
        ]
    
    def _estimate_formation_timeline(
        self,
        habit: OrganizationalHabit,
        organizational_context: Dict[str, Any]
    ) -> int:
        """Estimate timeline for habit formation in weeks"""
        
        # Base timeline based on habit complexity
        base_weeks = {
            HabitFrequency.DAILY: 8,
            HabitFrequency.WEEKLY: 12,
            HabitFrequency.BI_WEEKLY: 16,
            HabitFrequency.MONTHLY: 20,
            HabitFrequency.QUARTERLY: 24,
            HabitFrequency.EVENT_BASED: 16
        }.get(habit.frequency, 12)
        
        # Adjust for participant count
        if len(habit.participants) > 20:
            base_weeks = int(base_weeks * 1.3)
        elif len(habit.participants) > 50:
            base_weeks = int(base_weeks * 1.5)
        
        # Adjust for organizational readiness
        readiness = organizational_context.get('change_readiness', 'medium')
        readiness_multipliers = {
            'high': 0.8,
            'medium': 1.0,
            'low': 1.4
        }
        
        final_weeks = int(base_weeks * readiness_multipliers[readiness])
        
        return max(6, min(52, final_weeks))  # Between 6 and 52 weeks
    
    def _define_formation_milestones(
        self,
        habit: OrganizationalHabit,
        timeline_weeks: int
    ) -> List[str]:
        """Define key milestones for habit formation"""
        
        milestones = []
        
        # Phase-based milestones
        phase_duration = timeline_weeks // 5
        
        milestones.extend([
            f"Week {phase_duration}: Awareness campaign completed, stakeholder buy-in achieved",
            f"Week {phase_duration * 2}: Initial implementation launched, first executions completed",
            f"Week {phase_duration * 3}: Consistency patterns established, 70% participation achieved",
            f"Week {phase_duration * 4}: Reinforcement systems active, optimization improvements implemented",
            f"Week {timeline_weeks}: Habit institutionalized, sustainability mechanisms in place"
        ])
        
        # Habit-specific milestones
        if habit.habit_type == HabitType.COMMUNICATION:
            milestones.append(f"Week {timeline_weeks // 2}: Communication quality metrics improved by 25%")
        elif habit.habit_type == HabitType.COLLABORATION:
            milestones.append(f"Week {timeline_weeks // 2}: Cross-team collaboration instances increased by 40%")
        elif habit.habit_type == HabitType.LEARNING:
            milestones.append(f"Week {timeline_weeks // 2}: Learning activities completed by 80% of participants")
        
        return milestones
    
    def _define_formation_success_metrics(self, habit: OrganizationalHabit) -> List[str]:
        """Define success metrics for habit formation"""
        
        base_metrics = [
            "Participation rate >= 80%",
            "Consistency rate >= 70%",
            "Quality score >= 4.0/5.0",
            "Participant satisfaction >= 75%",
            "Sustainability score >= 0.8"
        ]
        
        # Add habit-specific metrics
        habit_specific_metrics = {
            HabitType.COMMUNICATION: [
                "Communication clarity improved by 30%",
                "Communication conflicts reduced by 50%"
            ],
            HabitType.COLLABORATION: [
                "Cross-team interactions increased by 40%",
                "Collaborative project success rate >= 85%"
            ],
            HabitType.LEARNING: [
                "New skills acquired per participant >= 2",
                "Knowledge sharing instances increased by 60%"
            ],
            HabitType.INNOVATION: [
                "Innovation ideas generated per month >= 5",
                "Innovation implementation rate >= 20%"
            ],
            HabitType.QUALITY: [
                "Quality defects reduced by 40%",
                "Quality process compliance >= 90%"
            ]
        }
        
        specific_metrics = habit_specific_metrics.get(habit.habit_type, [])
        base_metrics.extend(specific_metrics)
        
        return base_metrics
    
    def _design_reinforcement_mechanisms(self, habit: OrganizationalHabit) -> List[str]:
        """Design reinforcement mechanisms for habit sustainability"""
        
        mechanisms = [
            "Regular progress tracking and feedback",
            "Peer recognition and social reinforcement",
            "Manager support and encouragement",
            "Integration with performance reviews"
        ]
        
        # Add habit-specific reinforcement
        if habit.habit_type in [HabitType.RECOGNITION, HabitType.FEEDBACK]:
            mechanisms.append("Public celebration of habit execution")
        
        if habit.habit_type in [HabitType.LEARNING, HabitType.INNOVATION]:
            mechanisms.append("Showcase and sharing of outcomes")
        
        if habit.habit_type in [HabitType.COLLABORATION, HabitType.COMMUNICATION]:
            mechanisms.append("Team-based rewards and recognition")
        
        mechanisms.extend([
            "Habit execution reminders and prompts",
            "Success story documentation and sharing",
            "Continuous improvement and optimization"
        ])
        
        return mechanisms
    
    def implement_habit_sustainability_mechanisms(
        self,
        habit: OrganizationalHabit,
        formation_progress: Dict[str, Any] = None
    ) -> HabitSustainability:
        """
        Implement habit sustainability and reinforcement mechanisms
        
        Requirements: 3.3, 3.4 - Implement habit sustainability and reinforcement mechanisms
        """
        try:
            sustainability_id = str(uuid4())
            formation_progress = formation_progress or {}
            
            # Assess current sustainability level
            sustainability_level = self._assess_sustainability_level(habit, formation_progress)
            
            # Design reinforcement systems
            reinforcement_systems = self._design_sustainability_reinforcement(habit)
            
            # Create monitoring mechanisms
            monitoring_mechanisms = self._create_monitoring_mechanisms(habit)
            
            # Establish feedback loops
            feedback_loops = self._establish_feedback_loops(habit)
            
            # Define adaptation triggers
            adaptation_triggers = self._define_adaptation_triggers(habit)
            
            # Plan renewal strategies
            renewal_strategies = self._plan_renewal_strategies(habit)
            
            # Identify institutional support
            institutional_support = self._identify_institutional_support(habit)
            
            # Calculate cultural integration
            cultural_integration = self._calculate_cultural_integration(habit, formation_progress)
            
            # Identify resilience factors and vulnerabilities
            resilience_factors, vulnerability_points = self._analyze_resilience_vulnerabilities(habit)
            
            # Create mitigation plans
            mitigation_plans = self._create_vulnerability_mitigation_plans(vulnerability_points)
            
            # Calculate overall sustainability score
            sustainability_score = self._calculate_sustainability_score(
                sustainability_level, cultural_integration, resilience_factors, vulnerability_points
            )
            
            sustainability = HabitSustainability(
                id=sustainability_id,
                habit_id=habit.id,
                sustainability_level=sustainability_level,
                reinforcement_systems=reinforcement_systems,
                monitoring_mechanisms=monitoring_mechanisms,
                feedback_loops=feedback_loops,
                adaptation_triggers=adaptation_triggers,
                renewal_strategies=renewal_strategies,
                institutional_support=institutional_support,
                cultural_integration=cultural_integration,
                resilience_factors=resilience_factors,
                vulnerability_points=vulnerability_points,
                mitigation_plans=mitigation_plans,
                sustainability_score=sustainability_score,
                last_assessment=datetime.now(),
                next_review=datetime.now() + timedelta(weeks=4)
            )
            
            self.sustainability_mechanisms[sustainability_id] = sustainability
            logger.info(f"Implemented sustainability mechanisms for habit {habit.name}")
            
            return sustainability
            
        except Exception as e:
            logger.error(f"Error implementing sustainability mechanisms: {str(e)}")
            raise
    
    def _assess_sustainability_level(
        self,
        habit: OrganizationalHabit,
        formation_progress: Dict[str, Any]
    ) -> SustainabilityLevel:
        """Assess current sustainability level of the habit"""
        
        # Get progress indicators
        participation_rate = formation_progress.get('participation_rate', 0.5)
        consistency_rate = formation_progress.get('consistency_rate', 0.5)
        time_since_formation = formation_progress.get('weeks_since_formation', 0)
        
        # Calculate sustainability score
        sustainability_score = (participation_rate + consistency_rate) / 2
        
        # Adjust for time factor
        if time_since_formation < 4:
            return SustainabilityLevel.FRAGILE
        elif time_since_formation < 12:
            if sustainability_score >= 0.8:
                return SustainabilityLevel.DEVELOPING
            else:
                return SustainabilityLevel.FRAGILE
        elif time_since_formation < 24:
            if sustainability_score >= 0.9:
                return SustainabilityLevel.STABLE
            elif sustainability_score >= 0.7:
                return SustainabilityLevel.DEVELOPING
            else:
                return SustainabilityLevel.FRAGILE
        else:  # 24+ weeks
            if sustainability_score >= 0.95:
                return SustainabilityLevel.SELF_REINFORCING
            elif sustainability_score >= 0.85:
                return SustainabilityLevel.ROBUST
            elif sustainability_score >= 0.7:
                return SustainabilityLevel.STABLE
            else:
                return SustainabilityLevel.DEVELOPING
    
    def _design_sustainability_reinforcement(self, habit: OrganizationalHabit) -> List[str]:
        """Design reinforcement systems for sustainability"""
        
        systems = [
            "Automated habit execution reminders",
            "Progress tracking and visualization dashboard",
            "Peer accountability partnerships",
            "Manager check-ins and support",
            "Integration with team rituals and meetings"
        ]
        
        # Add habit-specific reinforcement
        if habit.frequency == HabitFrequency.DAILY:
            systems.append("Daily habit execution notifications")
        elif habit.frequency == HabitFrequency.WEEKLY:
            systems.append("Weekly habit review sessions")
        
        if len(habit.participants) > 10:
            systems.append("Team-based habit challenges and competitions")
        
        systems.extend([
            "Success story collection and sharing",
            "Habit impact measurement and reporting",
            "Continuous improvement feedback integration"
        ])
        
        return systems
    
    def _create_monitoring_mechanisms(self, habit: OrganizationalHabit) -> List[str]:
        """Create monitoring mechanisms for habit sustainability"""
        
        return [
            "Participation rate tracking",
            "Execution quality assessment",
            "Consistency pattern analysis",
            "Impact measurement and evaluation",
            "Participant satisfaction surveys",
            "Facilitator feedback collection",
            "Resource utilization monitoring",
            "Cultural integration assessment"
        ]
    
    def _establish_feedback_loops(self, habit: OrganizationalHabit) -> List[str]:
        """Establish feedback loops for habit optimization"""
        
        return [
            "Weekly participant feedback collection",
            "Monthly habit effectiveness review",
            "Quarterly sustainability assessment",
            "Real-time execution quality feedback",
            "Peer feedback and recognition system",
            "Manager observation and coaching feedback",
            "Impact measurement feedback to participants",
            "Continuous improvement suggestion system"
        ]
    
    def track_habit_progress(
        self,
        habit_id: str,
        participant_id: str,
        tracking_period: str,
        execution_count: int,
        target_count: int,
        quality_score: float = 0.8,
        engagement_level: float = 0.8
    ) -> HabitProgress:
        """Track progress of habit formation and execution"""
        try:
            progress_id = str(uuid4())
            
            # Calculate consistency rate
            consistency_rate = min(1.0, execution_count / target_count) if target_count > 0 else 0.0
            
            # Generate contextual information
            barriers_encountered = self._identify_progress_barriers(consistency_rate, quality_score)
            support_received = self._identify_support_received(engagement_level)
            improvements_noted = self._identify_improvements(consistency_rate, quality_score)
            feedback_provided = self._generate_progress_feedback(consistency_rate, quality_score, engagement_level)
            next_period_goals = self._set_next_period_goals(consistency_rate, quality_score)
            
            progress = HabitProgress(
                id=progress_id,
                habit_id=habit_id,
                participant_id=participant_id,
                tracking_period=tracking_period,
                execution_count=execution_count,
                target_count=target_count,
                consistency_rate=consistency_rate,
                quality_score=quality_score,
                engagement_level=engagement_level,
                barriers_encountered=barriers_encountered,
                support_received=support_received,
                improvements_noted=improvements_noted,
                feedback_provided=feedback_provided,
                next_period_goals=next_period_goals,
                recorded_date=datetime.now()
            )
            
            self.progress_tracking[progress_id] = progress
            logger.info(f"Tracked habit progress for participant {participant_id}")
            
            return progress
            
        except Exception as e:
            logger.error(f"Error tracking habit progress: {str(e)}")
            raise
    
    def calculate_habit_formation_metrics(
        self,
        organization_id: str
    ) -> HabitFormationMetrics:
        """Calculate comprehensive habit formation metrics"""
        
        # Get organization habits (simplified - would query by organization)
        org_habits = [h for h in self.habits.values() if organization_id in h.participants]
        
        # Calculate metrics
        total_habits_designed = len(org_habits)
        habits_in_formation = len([h for h in org_habits if h.stage in [
            HabitStage.INITIATION, HabitStage.FORMATION
        ]])
        habits_established = len([h for h in org_habits if h.stage == HabitStage.MAINTENANCE])
        habits_institutionalized = len([h for h in org_habits if h.stage == HabitStage.INSTITUTIONALIZED])
        
        # Get progress data for organization
        org_progress = [p for p in self.progress_tracking.values() 
                       if p.habit_id in [h.id for h in org_habits]]
        
        # Calculate averages
        if org_progress:
            participant_engagement_average = sum(p.engagement_level for p in org_progress) / len(org_progress)
        else:
            participant_engagement_average = 0.0
        
        # Simplified metrics (would be more complex in real implementation)
        average_formation_time_weeks = 12.0
        overall_success_rate = 0.75
        sustainability_index = 0.8
        cultural_integration_score = sum(h.cultural_alignment for h in org_habits) / len(org_habits) if org_habits else 0.0
        business_impact_score = 0.7
        roi_achieved = 3.2
        
        return HabitFormationMetrics(
            organization_id=organization_id,
            total_habits_designed=total_habits_designed,
            habits_in_formation=habits_in_formation,
            habits_established=habits_established,
            habits_institutionalized=habits_institutionalized,
            average_formation_time_weeks=average_formation_time_weeks,
            overall_success_rate=overall_success_rate,
            participant_engagement_average=participant_engagement_average,
            sustainability_index=sustainability_index,
            cultural_integration_score=cultural_integration_score,
            business_impact_score=business_impact_score,
            roi_achieved=roi_achieved,
            calculated_date=datetime.now()
        )
    
    def get_habit(self, habit_id: str) -> Optional[OrganizationalHabit]:
        """Get organizational habit by ID"""
        return self.habits.get(habit_id)
    
    def get_organization_habits(self, organization_id: str) -> List[OrganizationalHabit]:
        """Get all habits for an organization"""
        return [
            habit for habit in self.habits.values()
            if organization_id in habit.participants
        ]
    
    def get_formation_strategy(self, strategy_id: str) -> Optional[HabitFormationStrategy]:
        """Get habit formation strategy by ID"""
        return self.formation_strategies.get(strategy_id)
    
    def get_sustainability_mechanism(self, sustainability_id: str) -> Optional[HabitSustainability]:
        """Get habit sustainability mechanism by ID"""
        return self.sustainability_mechanisms.get(sustainability_id)
    
    # Helper methods for progress tracking
    def _identify_progress_barriers(self, consistency_rate: float, quality_score: float) -> List[str]:
        """Identify barriers based on progress metrics"""
        barriers = []
        
        if consistency_rate < 0.5:
            barriers.extend(["Time constraints", "Competing priorities", "Lack of reminders"])
        if quality_score < 0.6:
            barriers.extend(["Insufficient training", "Unclear expectations", "Resource limitations"])
        if consistency_rate < 0.3:
            barriers.append("Low motivation or engagement")
        
        return barriers
    
    def _identify_support_received(self, engagement_level: float) -> List[str]:
        """Identify support received based on engagement"""
        support = ["Manager encouragement", "Peer collaboration"]
        
        if engagement_level > 0.8:
            support.extend(["Excellent facilitation", "Strong team support", "Clear guidance"])
        elif engagement_level > 0.6:
            support.extend(["Good facilitation", "Team support", "Regular feedback"])
        else:
            support.extend(["Basic support", "Minimal guidance"])
        
        return support
    
    def _identify_improvements(self, consistency_rate: float, quality_score: float) -> List[str]:
        """Identify improvements based on metrics"""
        improvements = []
        
        if consistency_rate > 0.8:
            improvements.append("Excellent habit consistency achieved")
        if quality_score > 0.8:
            improvements.append("High quality execution maintained")
        if consistency_rate > 0.6 and quality_score > 0.7:
            improvements.append("Strong overall habit performance")
        
        return improvements
    
    def _generate_progress_feedback(
        self, 
        consistency_rate: float, 
        quality_score: float, 
        engagement_level: float
    ) -> str:
        """Generate feedback based on progress metrics"""
        
        if consistency_rate > 0.8 and quality_score > 0.8:
            return "Excellent progress! You're demonstrating strong habit formation."
        elif consistency_rate > 0.6 and quality_score > 0.7:
            return "Good progress! Continue building consistency and quality."
        elif consistency_rate < 0.5:
            return "Focus on improving consistency. Consider setting reminders and reducing barriers."
        elif quality_score < 0.6:
            return "Work on execution quality. Seek additional training or clarification."
        else:
            return "Keep working on building this habit. Progress takes time and persistence."
    
    def _set_next_period_goals(self, consistency_rate: float, quality_score: float) -> List[str]:
        """Set goals for next tracking period"""
        goals = []
        
        if consistency_rate < 0.8:
            target_rate = min(1.0, consistency_rate + 0.2)
            goals.append(f"Increase consistency rate to {target_rate:.1f}")
        
        if quality_score < 0.8:
            target_quality = min(1.0, quality_score + 0.15)
            goals.append(f"Improve execution quality to {target_quality:.1f}")
        
        goals.extend([
            "Maintain regular habit execution",
            "Seek feedback and continuous improvement",
            "Support team members in habit formation"
        ])
        
        return goals[:3]  # Limit to 3 goals
    
    # Additional helper methods for sustainability mechanisms
    def _define_adaptation_triggers(self, habit: OrganizationalHabit) -> List[str]:
        """Define triggers for habit adaptation"""
        return [
            "Participation rate drops below 70%",
            "Quality scores decline for 2+ weeks",
            "Participant feedback indicates issues",
            "Organizational changes affect habit execution",
            "New technology or processes introduced",
            "Quarterly sustainability review identifies gaps"
        ]
    
    def _plan_renewal_strategies(self, habit: OrganizationalHabit) -> List[str]:
        """Plan strategies for habit renewal and revitalization"""
        return [
            "Refresh habit execution methods",
            "Introduce new reinforcement mechanisms",
            "Update success indicators and metrics",
            "Enhance facilitator training and support",
            "Celebrate habit achievements and milestones",
            "Integrate habit with new organizational initiatives"
        ]
    
    def _identify_institutional_support(self, habit: OrganizationalHabit) -> List[str]:
        """Identify institutional support mechanisms"""
        return [
            "Leadership endorsement and participation",
            "Integration with performance management",
            "Resource allocation and budget support",
            "Policy and procedure integration",
            "Training and development programs",
            "Recognition and reward systems",
            "Communication and awareness campaigns"
        ]
    
    def _calculate_cultural_integration(
        self, 
        habit: OrganizationalHabit, 
        formation_progress: Dict[str, Any]
    ) -> float:
        """Calculate level of cultural integration"""
        
        base_integration = habit.cultural_alignment
        
        # Adjust based on formation progress
        participation_rate = formation_progress.get('participation_rate', 0.5)
        time_factor = min(1.0, formation_progress.get('weeks_since_formation', 0) / 24)
        
        integration_score = (base_integration + participation_rate + time_factor) / 3
        
        return min(1.0, max(0.0, integration_score))
    
    def _analyze_resilience_vulnerabilities(
        self, 
        habit: OrganizationalHabit
    ) -> tuple[List[str], List[str]]:
        """Analyze resilience factors and vulnerability points"""
        
        resilience_factors = [
            "Strong participant engagement",
            "Clear value proposition",
            "Effective reinforcement systems",
            "Leadership support",
            "Cultural alignment"
        ]
        
        vulnerability_points = [
            "Dependency on key facilitators",
            "Resource constraints",
            "Competing organizational priorities",
            "Participant turnover",
            "Technology or process changes"
        ]
        
        # Add habit-specific factors
        if habit.frequency == HabitFrequency.DAILY:
            vulnerability_points.append("Daily execution burden")
        if len(habit.participants) > 20:
            vulnerability_points.append("Coordination complexity")
        
        return resilience_factors, vulnerability_points
    
    def _create_vulnerability_mitigation_plans(self, vulnerability_points: List[str]) -> List[str]:
        """Create mitigation plans for vulnerability points"""
        
        mitigation_plans = []
        
        for vulnerability in vulnerability_points:
            if "facilitator" in vulnerability.lower():
                mitigation_plans.append("Train backup facilitators and create rotation system")
            elif "resource" in vulnerability.lower():
                mitigation_plans.append("Secure dedicated budget and resource commitments")
            elif "priority" in vulnerability.lower():
                mitigation_plans.append("Integrate habit with key organizational initiatives")
            elif "turnover" in vulnerability.lower():
                mitigation_plans.append("Create comprehensive onboarding for new participants")
            elif "technology" in vulnerability.lower():
                mitigation_plans.append("Build adaptable habit execution methods")
            else:
                mitigation_plans.append(f"Monitor and address {vulnerability.lower()}")
        
        return mitigation_plans
    
    def _calculate_sustainability_score(
        self,
        sustainability_level: SustainabilityLevel,
        cultural_integration: float,
        resilience_factors: List[str],
        vulnerability_points: List[str]
    ) -> float:
        """Calculate overall sustainability score"""
        
        # Base score from sustainability level
        level_scores = {
            SustainabilityLevel.FRAGILE: 0.3,
            SustainabilityLevel.DEVELOPING: 0.5,
            SustainabilityLevel.STABLE: 0.7,
            SustainabilityLevel.ROBUST: 0.85,
            SustainabilityLevel.SELF_REINFORCING: 0.95
        }
        
        base_score = level_scores[sustainability_level]
        
        # Adjust for cultural integration
        integration_factor = cultural_integration * 0.2
        
        # Adjust for resilience vs vulnerability ratio
        resilience_ratio = len(resilience_factors) / (len(resilience_factors) + len(vulnerability_points))
        resilience_factor = (resilience_ratio - 0.5) * 0.2
        
        final_score = base_score + integration_factor + resilience_factor
        
        return min(1.0, max(0.0, final_score))  
  
    def _identify_barrier_mitigation(
        self, 
        habit: OrganizationalHabit, 
        organizational_context: Dict[str, Any]
    ) -> List[str]:
        """Identify barriers and mitigation strategies for habit formation"""
        
        mitigation_strategies = []
        
        # Common barriers and mitigations
        mitigation_strategies.extend([
            "Conduct stakeholder engagement sessions to build buy-in",
            "Provide clear communication about habit benefits and purpose",
            "Start with pilot group before full rollout",
            "Establish regular check-ins and support mechanisms",
            "Create accountability partnerships among participants"
        ])
        
        # Habit-specific mitigations
        if habit.frequency == HabitFrequency.DAILY:
            mitigation_strategies.append("Address daily execution burden with time management support")
        
        if len(habit.participants) > 20:
            mitigation_strategies.append("Use phased rollout approach for large groups")
        
        # Context-specific mitigations
        change_readiness = organizational_context.get('change_readiness', 'medium')
        if change_readiness == 'low':
            mitigation_strategies.extend([
                "Increase change management support and communication",
                "Provide additional training and coaching resources"
            ])
        
        return mitigation_strategies
    
    def _plan_stakeholder_engagement(self, habit: OrganizationalHabit) -> Dict[str, List[str]]:
        """Plan stakeholder engagement for habit formation"""
        
        return {
            "leadership": [
                "Secure executive sponsorship and visible support",
                "Regular leadership communication about habit importance",
                "Leadership participation in habit activities"
            ],
            "participants": [
                "Involve participants in habit design and refinement",
                "Provide regular feedback and recognition",
                "Create peer support networks"
            ],
            "facilitators": [
                "Provide comprehensive facilitator training",
                "Establish facilitator support network",
                "Regular facilitator feedback and coaching"
            ],
            "stakeholders": [
                "Regular progress updates and success stories",
                "Address concerns and feedback promptly",
                "Celebrate milestones and achievements"
            ]
        }
    
    def _allocate_formation_resources(self, habit: OrganizationalHabit, timeline_weeks: int) -> Dict[str, Any]:
        """Allocate resources for habit formation"""
        
        return {
            "budget": f"${timeline_weeks * 500}",  # Simplified budget calculation
            "time_allocation": f"{timeline_weeks} weeks",
            "personnel": habit.facilitators + ["project_coordinator"],
            "tools": habit.resources_required,
            "training": ["facilitator_training", "participant_onboarding"]
        }
    
    def _assess_formation_risks(
        self, 
        habit: OrganizationalHabit, 
        organizational_context: Dict[str, Any]
    ) -> Dict[str, str]:
        """Assess risks for habit formation"""
        
        risks = {
            "participation": "Medium - May face initial resistance or low engagement",
            "sustainability": "Medium - Requires ongoing reinforcement and support",
            "resource": "Low - Resources are well-defined and allocated",
            "timeline": "Low - Timeline is realistic and achievable"
        }
        
        # Adjust based on context
        change_readiness = organizational_context.get('change_readiness', 'medium')
        if change_readiness == 'low':
            risks["participation"] = "High - Low change readiness increases resistance risk"
        
        if len(habit.participants) > 30:
            risks["coordination"] = "High - Large participant group increases coordination complexity"
        
        return risks