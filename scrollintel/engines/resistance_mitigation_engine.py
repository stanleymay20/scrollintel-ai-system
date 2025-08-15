"""
Cultural Change Resistance Mitigation Engine
Implements targeted resistance addressing strategies with intervention design and resolution tracking.
"""
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Tuple
import uuid
from dataclasses import asdict

from ..models.resistance_mitigation_models import (
    MitigationPlan, MitigationIntervention, ResistanceAddressingStrategy,
    MitigationExecution, ResistanceResolution, MitigationValidation,
    MitigationTemplate, MitigationMetrics, MitigationStrategy,
    InterventionType, MitigationStatus
)
from ..models.resistance_detection_models import (
    ResistanceDetection, ResistanceType, ResistanceSeverity
)
from ..models.cultural_assessment_models import Organization
from ..models.transformation_roadmap_models import Transformation


class ResistanceMitigationEngine:
    """Engine for mitigating cultural change resistance"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.addressing_strategies = self._initialize_addressing_strategies()
        self.mitigation_templates = self._initialize_mitigation_templates()
        
    def create_mitigation_plan(
        self,
        detection: ResistanceDetection,
        organization: Organization,
        transformation: Transformation,
        constraints: Optional[Dict[str, Any]] = None
    ) -> MitigationPlan:
        """
        Create targeted resistance addressing strategies
        
        Args:
            detection: Detected resistance instance
            organization: Organization context
            transformation: Transformation being impacted
            constraints: Resource and timeline constraints
            
        Returns:
            Comprehensive mitigation plan
        """
        try:
            # Select appropriate strategies
            strategies = self._select_mitigation_strategies(
                detection, organization, transformation, constraints
            )
            
            # Design specific interventions
            interventions = self._design_interventions(
                detection, strategies, organization, constraints
            )
            
            # Identify target stakeholders
            target_stakeholders = self._identify_target_stakeholders(
                detection, organization
            )
            
            # Define success criteria
            success_criteria = self._define_success_criteria(
                detection, transformation
            )
            
            # Create timeline
            timeline = self._create_mitigation_timeline(
                interventions, constraints
            )
            
            # Estimate resource requirements
            resource_requirements = self._estimate_resource_requirements(
                interventions, organization
            )
            
            # Identify risk factors
            risk_factors = self._identify_risk_factors(
                detection, interventions, organization
            )
            
            # Create contingency plans
            contingency_plans = self._create_contingency_plans(
                risk_factors, strategies
            )
            
            mitigation_plan = MitigationPlan(
                id=str(uuid.uuid4()),
                detection_id=detection.id,
                organization_id=organization.id,
                transformation_id=detection.transformation_id,
                resistance_type=detection.resistance_type,
                severity=detection.severity,
                strategies=strategies,
                interventions=interventions,
                target_stakeholders=target_stakeholders,
                success_criteria=success_criteria,
                timeline=timeline,
                resource_requirements=resource_requirements,
                risk_factors=risk_factors,
                contingency_plans=contingency_plans,
                created_at=datetime.now(),
                created_by="system"
            )
            
            self.logger.info(f"Created mitigation plan {mitigation_plan.id} for detection {detection.id}")
            return mitigation_plan
            
        except Exception as e:
            self.logger.error(f"Error creating mitigation plan: {str(e)}")
            raise
    
    def execute_mitigation_plan(
        self,
        plan: MitigationPlan,
        organization: Organization
    ) -> MitigationExecution:
        """
        Execute mitigation plan with intervention coordination
        
        Args:
            plan: Mitigation plan to execute
            organization: Organization context
            
        Returns:
            Execution tracking object
        """
        try:
            # Initialize execution tracking
            execution = MitigationExecution(
                id=str(uuid.uuid4()),
                plan_id=plan.id,
                execution_phase="initiation",
                start_date=datetime.now(),
                end_date=None,
                status=MitigationStatus.IN_PROGRESS,
                progress_percentage=0.0,
                completed_interventions=[],
                active_interventions=[],
                pending_interventions=[i.id for i in plan.interventions],
                resource_utilization={},
                stakeholder_engagement={},
                interim_results={},
                challenges_encountered=[],
                adjustments_made=[],
                next_steps=[]
            )
            
            # Execute interventions in sequence
            for intervention in plan.interventions:
                execution_result = self._execute_intervention(
                    intervention, plan, organization, execution
                )
                execution = self._update_execution_progress(
                    execution, intervention, execution_result
                )
            
            # Finalize execution
            execution.status = MitigationStatus.COMPLETED
            execution.end_date = datetime.now()
            execution.progress_percentage = 100.0
            
            self.logger.info(f"Completed mitigation plan execution {execution.id}")
            return execution
            
        except Exception as e:
            self.logger.error(f"Error executing mitigation plan: {str(e)}")
            raise
    
    def track_resistance_resolution(
        self,
        detection: ResistanceDetection,
        plan: MitigationPlan,
        execution: MitigationExecution
    ) -> ResistanceResolution:
        """
        Track resolution of resistance instance
        
        Args:
            detection: Original resistance detection
            plan: Mitigation plan used
            execution: Execution results
            
        Returns:
            Resolution tracking with effectiveness assessment
        """
        try:
            # Assess resolution effectiveness
            effectiveness_rating = self._assess_resolution_effectiveness(
                detection, plan, execution
            )
            
            # Measure stakeholder satisfaction
            stakeholder_satisfaction = self._measure_stakeholder_satisfaction(
                plan, execution
            )
            
            # Identify behavioral changes
            behavioral_changes = self._identify_behavioral_changes(
                detection, execution
            )
            
            # Assess cultural impact
            cultural_impact = self._assess_cultural_impact(
                detection, plan, execution
            )
            
            # Extract lessons learned
            lessons_learned = self._extract_lessons_learned(
                plan, execution
            )
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                detection, plan, execution
            )
            
            # Determine follow-up requirements
            follow_up_required, follow_up_schedule = self._determine_follow_up(
                effectiveness_rating, behavioral_changes
            )
            
            resolution = ResistanceResolution(
                id=str(uuid.uuid4()),
                detection_id=detection.id,
                plan_id=plan.id,
                resolution_date=datetime.now(),
                resolution_method=self._determine_resolution_method(plan),
                final_status="resolved" if effectiveness_rating >= 0.7 else "partially_resolved",
                effectiveness_rating=effectiveness_rating,
                stakeholder_satisfaction=stakeholder_satisfaction,
                behavioral_changes=behavioral_changes,
                cultural_impact=cultural_impact,
                lessons_learned=lessons_learned,
                recommendations=recommendations,
                follow_up_required=follow_up_required,
                follow_up_schedule=follow_up_schedule
            )
            
            self.logger.info(f"Tracked resistance resolution {resolution.id}")
            return resolution
            
        except Exception as e:
            self.logger.error(f"Error tracking resistance resolution: {str(e)}")
            raise
    
    def validate_mitigation_effectiveness(
        self,
        plan: MitigationPlan,
        execution: MitigationExecution,
        post_intervention_data: Dict[str, Any]
    ) -> MitigationValidation:
        """
        Validate effectiveness of mitigation efforts
        
        Args:
            plan: Mitigation plan that was executed
            execution: Execution results
            post_intervention_data: Data collected after interventions
            
        Returns:
            Validation results with effectiveness assessment
        """
        try:
            # Check success criteria
            success_criteria_met = self._check_success_criteria(
                plan, post_intervention_data
            )
            
            # Calculate quantitative results
            quantitative_results = self._calculate_quantitative_results(
                plan, post_intervention_data
            )
            
            # Collect qualitative feedback
            qualitative_feedback = self._collect_qualitative_feedback(
                plan, execution, post_intervention_data
            )
            
            # Assess stakeholder perspectives
            stakeholder_assessments = self._assess_stakeholder_perspectives(
                plan, post_intervention_data
            )
            
            # Measure behavioral indicators
            behavioral_indicators = self._measure_behavioral_indicators(
                plan, post_intervention_data
            )
            
            # Assess cultural metrics
            cultural_metrics = self._assess_cultural_metrics(
                plan, post_intervention_data
            )
            
            # Evaluate sustainability
            sustainability_assessment = self._evaluate_sustainability(
                plan, execution, post_intervention_data
            )
            
            # Generate improvement recommendations
            improvement_recommendations = self._generate_improvement_recommendations(
                plan, execution, post_intervention_data
            )
            
            # Calculate validation confidence
            validation_confidence = self._calculate_validation_confidence(
                success_criteria_met, quantitative_results, qualitative_feedback
            )
            
            validation = MitigationValidation(
                id=str(uuid.uuid4()),
                plan_id=plan.id,
                validation_date=datetime.now(),
                validation_method="comprehensive_assessment",
                success_criteria_met=success_criteria_met,
                quantitative_results=quantitative_results,
                qualitative_feedback=qualitative_feedback,
                stakeholder_assessments=stakeholder_assessments,
                behavioral_indicators=behavioral_indicators,
                cultural_metrics=cultural_metrics,
                sustainability_assessment=sustainability_assessment,
                improvement_recommendations=improvement_recommendations,
                validation_confidence=validation_confidence
            )
            
            self.logger.info(f"Validated mitigation effectiveness {validation.id}")
            return validation
            
        except Exception as e:
            self.logger.error(f"Error validating mitigation effectiveness: {str(e)}")
            raise
    
    def _select_mitigation_strategies(
        self,
        detection: ResistanceDetection,
        organization: Organization,
        transformation: Transformation,
        constraints: Optional[Dict[str, Any]]
    ) -> List[MitigationStrategy]:
        """Select appropriate mitigation strategies based on resistance type and context"""
        strategy_mapping = {
            ResistanceType.ACTIVE_OPPOSITION: [
                MitigationStrategy.STAKEHOLDER_ENGAGEMENT,
                MitigationStrategy.LEADERSHIP_INTERVENTION,
                MitigationStrategy.COMMUNICATION_ENHANCEMENT
            ],
            ResistanceType.PASSIVE_RESISTANCE: [
                MitigationStrategy.INCENTIVE_ALIGNMENT,
                MitigationStrategy.PEER_INFLUENCE,
                MitigationStrategy.TRAINING_SUPPORT
            ],
            ResistanceType.SKEPTICISM: [
                MitigationStrategy.COMMUNICATION_ENHANCEMENT,
                MitigationStrategy.FEEDBACK_INTEGRATION,
                MitigationStrategy.GRADUAL_IMPLEMENTATION
            ],
            ResistanceType.FEAR_BASED: [
                MitigationStrategy.TRAINING_SUPPORT,
                MitigationStrategy.RESOURCE_PROVISION,
                MitigationStrategy.COMMUNICATION_ENHANCEMENT
            ],
            ResistanceType.RESOURCE_BASED: [
                MitigationStrategy.RESOURCE_PROVISION,
                MitigationStrategy.PROCESS_MODIFICATION,
                MitigationStrategy.GRADUAL_IMPLEMENTATION
            ]
        }
        
        base_strategies = strategy_mapping.get(detection.resistance_type, [
            MitigationStrategy.COMMUNICATION_ENHANCEMENT,
            MitigationStrategy.STAKEHOLDER_ENGAGEMENT
        ])
        
        # Adjust based on severity
        if detection.severity in [ResistanceSeverity.HIGH, ResistanceSeverity.CRITICAL]:
            if MitigationStrategy.LEADERSHIP_INTERVENTION not in base_strategies:
                base_strategies.append(MitigationStrategy.LEADERSHIP_INTERVENTION)
        
        return base_strategies[:3]  # Limit to top 3 strategies
    
    def _design_interventions(
        self,
        detection: ResistanceDetection,
        strategies: List[MitigationStrategy],
        organization: Organization,
        constraints: Optional[Dict[str, Any]]
    ) -> List[MitigationIntervention]:
        """Design specific interventions for each strategy"""
        interventions = []
        
        for strategy in strategies:
            intervention_designs = self._get_intervention_designs_for_strategy(
                strategy, detection, organization, constraints
            )
            interventions.extend(intervention_designs)
        
        return interventions
    
    def _get_intervention_designs_for_strategy(
        self,
        strategy: MitigationStrategy,
        detection: ResistanceDetection,
        organization: Organization,
        constraints: Optional[Dict[str, Any]]
    ) -> List[MitigationIntervention]:
        """Get intervention designs for a specific strategy"""
        intervention_templates = {
            MitigationStrategy.COMMUNICATION_ENHANCEMENT: [
                {
                    "type": InterventionType.TOWN_HALL_MEETING,
                    "title": "Transformation Clarity Session",
                    "description": "Address concerns and provide clear communication about changes",
                    "duration": 2.0
                },
                {
                    "type": InterventionType.COMMUNICATION_CAMPAIGN,
                    "title": "Change Benefits Campaign",
                    "description": "Multi-channel campaign highlighting transformation benefits",
                    "duration": 8.0
                }
            ],
            MitigationStrategy.STAKEHOLDER_ENGAGEMENT: [
                {
                    "type": InterventionType.TEAM_WORKSHOP,
                    "title": "Stakeholder Engagement Workshop",
                    "description": "Interactive session to engage resistant stakeholders",
                    "duration": 4.0
                }
            ],
            MitigationStrategy.TRAINING_SUPPORT: [
                {
                    "type": InterventionType.TRAINING_SESSION,
                    "title": "Skills Development Training",
                    "description": "Training to build capabilities for new processes",
                    "duration": 6.0
                }
            ]
        }
        
        templates = intervention_templates.get(strategy, [])
        interventions = []
        
        for template in templates:
            intervention = MitigationIntervention(
                id=str(uuid.uuid4()),
                plan_id="",  # Will be set when plan is created
                intervention_type=template["type"],
                strategy=strategy,
                title=template["title"],
                description=template["description"],
                target_audience=detection.affected_areas,
                facilitators=["change_management_team"],
                duration_hours=template["duration"],
                scheduled_date=datetime.now() + timedelta(days=7),
                completion_date=None,
                status=MitigationStatus.PLANNED,
                success_metrics={
                    "attendance_rate": 0.8,
                    "satisfaction_score": 0.7,
                    "engagement_improvement": 0.15
                },
                actual_results={},
                participant_feedback=[],
                effectiveness_score=None,
                lessons_learned=[],
                follow_up_actions=[]
            )
            interventions.append(intervention)
        
        return interventions
    
    def _identify_target_stakeholders(
        self, detection: ResistanceDetection, organization: Organization
    ) -> List[str]:
        """Identify target stakeholders for mitigation efforts"""
        return detection.affected_areas + ["change_champions", "team_leads"]
    
    def _define_success_criteria(
        self, detection: ResistanceDetection, transformation: Transformation
    ) -> Dict[str, Any]:
        """Define success criteria for mitigation plan"""
        return {
            "resistance_reduction": 0.7,
            "engagement_improvement": 0.2,
            "sentiment_improvement": 0.3,
            "behavioral_compliance": 0.8,
            "stakeholder_satisfaction": 0.75
        }
    
    def _create_mitigation_timeline(
        self, interventions: List[MitigationIntervention], constraints: Optional[Dict[str, Any]]
    ) -> Dict[str, datetime]:
        """Create timeline for mitigation activities"""
        start_date = datetime.now() + timedelta(days=3)
        end_date = start_date + timedelta(days=30)
        
        return {
            "start_date": start_date,
            "end_date": end_date,
            "review_date": start_date + timedelta(days=15),
            "validation_date": end_date + timedelta(days=7)
        }
    
    def _estimate_resource_requirements(
        self, interventions: List[MitigationIntervention], organization: Organization
    ) -> Dict[str, Any]:
        """Estimate resource requirements for mitigation plan"""
        total_hours = sum(i.duration_hours for i in interventions)
        
        return {
            "facilitator_hours": total_hours,
            "participant_hours": total_hours * 10,  # Assuming 10 participants average
            "budget_estimate": total_hours * 150,  # $150 per hour
            "materials_needed": ["presentation_materials", "workshop_supplies"],
            "technology_requirements": ["video_conferencing", "collaboration_tools"]
        }
    
    def _identify_risk_factors(
        self,
        detection: ResistanceDetection,
        interventions: List[MitigationIntervention],
        organization: Organization
    ) -> List[str]:
        """Identify risk factors for mitigation plan"""
        return [
            "low_participation_risk",
            "leadership_support_uncertainty",
            "competing_priorities",
            "resource_constraints",
            "timeline_pressure"
        ]
    
    def _create_contingency_plans(
        self, risk_factors: List[str], strategies: List[MitigationStrategy]
    ) -> List[str]:
        """Create contingency plans for identified risks"""
        return [
            "escalate_to_senior_leadership",
            "adjust_timeline_if_needed",
            "provide_additional_resources",
            "modify_intervention_approach",
            "engage_external_facilitators"
        ]
    
    def _execute_intervention(
        self,
        intervention: MitigationIntervention,
        plan: MitigationPlan,
        organization: Organization,
        execution: MitigationExecution
    ) -> Dict[str, Any]:
        """Execute a specific intervention"""
        # Mock execution results
        return {
            "attendance_rate": 0.85,
            "satisfaction_score": 0.78,
            "engagement_level": 0.72,
            "feedback_sentiment": 0.15,
            "completion_status": "successful"
        }
    
    def _update_execution_progress(
        self,
        execution: MitigationExecution,
        intervention: MitigationIntervention,
        result: Dict[str, Any]
    ) -> MitigationExecution:
        """Update execution progress after intervention completion"""
        execution.completed_interventions.append(intervention.id)
        if intervention.id in execution.active_interventions:
            execution.active_interventions.remove(intervention.id)
        if intervention.id in execution.pending_interventions:
            execution.pending_interventions.remove(intervention.id)
        
        # Update progress percentage
        total_interventions = len(execution.completed_interventions) + len(execution.active_interventions) + len(execution.pending_interventions)
        if total_interventions > 0:
            execution.progress_percentage = (len(execution.completed_interventions) / total_interventions) * 100
        
        return execution
    
    def _initialize_addressing_strategies(self) -> List[ResistanceAddressingStrategy]:
        """Initialize resistance addressing strategies"""
        return [
            ResistanceAddressingStrategy(
                id="strategy_001",
                resistance_type=ResistanceType.ACTIVE_OPPOSITION,
                strategy_name="Direct Engagement and Leadership Intervention",
                description="Address active opposition through direct engagement and leadership support",
                approach_steps=[
                    "Identify key opposition leaders",
                    "Schedule one-on-one meetings",
                    "Address specific concerns",
                    "Provide leadership backing",
                    "Monitor behavioral changes"
                ],
                required_resources={"facilitator_time": 20, "leadership_time": 10},
                typical_duration=21,
                success_rate=0.75,
                best_practices=[
                    "Listen actively to concerns",
                    "Provide clear rationale",
                    "Offer compromise where possible"
                ],
                common_pitfalls=[
                    "Dismissing concerns",
                    "Using authoritarian approach",
                    "Lack of follow-up"
                ],
                effectiveness_factors={"leadership_credibility": 0.8, "communication_quality": 0.7},
                stakeholder_considerations={"individual": ["respect", "autonomy"], "team": ["influence", "dynamics"]}
            )
        ]
    
    def _initialize_mitigation_templates(self) -> List[MitigationTemplate]:
        """Initialize mitigation templates for common scenarios"""
        return [
            MitigationTemplate(
                id="template_001",
                template_name="Passive Resistance Standard Response",
                resistance_types=[ResistanceType.PASSIVE_RESISTANCE],
                severity_levels=[ResistanceSeverity.LOW, ResistanceSeverity.MODERATE],
                template_strategies=[
                    MitigationStrategy.INCENTIVE_ALIGNMENT,
                    MitigationStrategy.PEER_INFLUENCE,
                    MitigationStrategy.TRAINING_SUPPORT
                ],
                template_interventions=[
                    {"type": "team_workshop", "duration": 4, "participants": 10},
                    {"type": "training_session", "duration": 6, "participants": 15}
                ],
                customization_points=["target_audience", "timeline", "resources"],
                success_factors=["peer_support", "clear_benefits", "skill_development"],
                implementation_guide=[
                    "Assess current skill levels",
                    "Design targeted training",
                    "Implement peer support system",
                    "Monitor progress regularly"
                ],
                resource_estimates={"hours": 40, "budget": 6000},
                timeline_template={"preparation": 5, "execution": 15, "follow_up": 10},
                validation_criteria={"engagement_increase": 0.2, "compliance_rate": 0.8}
            )
        ]
    
    # Additional helper methods for resolution tracking and validation
    def _assess_resolution_effectiveness(self, detection, plan, execution): return 0.8
    def _measure_stakeholder_satisfaction(self, plan, execution): return {"employees": 0.75, "managers": 0.80}
    def _identify_behavioral_changes(self, detection, execution): return ["increased_participation", "improved_compliance"]
    def _assess_cultural_impact(self, detection, plan, execution): return {"engagement": 0.15, "trust": 0.10}
    def _extract_lessons_learned(self, plan, execution): return ["early_engagement_critical", "leadership_support_essential"]
    def _generate_recommendations(self, detection, plan, execution): return ["continue_monitoring", "expand_training"]
    def _determine_follow_up(self, effectiveness, changes): return True, datetime.now() + timedelta(days=30)
    def _determine_resolution_method(self, plan): return "multi_intervention_approach"
    def _check_success_criteria(self, plan, data): return {"resistance_reduction": True, "engagement_improvement": True}
    def _calculate_quantitative_results(self, plan, data): return {"engagement_increase": 0.22, "sentiment_improvement": 0.18}
    def _collect_qualitative_feedback(self, plan, execution, data): return ["positive_response", "continued_support_needed"]
    def _assess_stakeholder_perspectives(self, plan, data): return {"employees": {"satisfaction": 0.8}, "managers": {"confidence": 0.85}}
    def _measure_behavioral_indicators(self, plan, data): return {"participation": 0.85, "compliance": 0.90}
    def _assess_cultural_metrics(self, plan, data): return {"trust": 0.78, "collaboration": 0.82}
    def _evaluate_sustainability(self, plan, execution, data): return 0.75
    def _generate_improvement_recommendations(self, plan, execution, data): return ["enhance_communication", "increase_training"]
    def _calculate_validation_confidence(self, criteria, results, feedback): return 0.85