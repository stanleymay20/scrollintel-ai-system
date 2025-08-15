"""
Experiment planning framework for autonomous innovation lab.
"""
import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import logging

from ..models.experimental_design_models import (
    ExperimentPlan, ExperimentType, MethodologyType, ExperimentStatus,
    ExperimentalVariable, ExperimentalCondition, Hypothesis,
    ExperimentalProtocol, ResourceRequirement, ExperimentMilestone,
    ValidationStudy, MethodologyRecommendation, ExperimentOptimization
)


class ExperimentPlanner:
    """
    Automated experiment design and validation study planning system.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.methodology_database = self._initialize_methodology_database()
        self.design_templates = self._initialize_design_templates()
        
    def plan_experiment(
        self,
        research_question: str,
        domain: str,
        constraints: Optional[Dict[str, Any]] = None
    ) -> ExperimentPlan:
        """
        Create comprehensive experiment plan from research question.
        
        Args:
            research_question: The research question to investigate
            domain: Research domain (e.g., 'materials', 'software', 'biology')
            constraints: Optional constraints (budget, time, resources)
            
        Returns:
            Complete experiment plan
        """
        try:
            # Validate inputs
            if not research_question or not research_question.strip():
                raise ValueError("Research question cannot be empty")
            
            # Generate unique plan ID
            plan_id = str(uuid.uuid4())
            
            # Analyze research question to extract key components
            analysis = self._analyze_research_question(research_question, domain)
            
            # Generate hypotheses
            hypotheses = self._generate_hypotheses(analysis)
            
            # Determine optimal experiment type and methodology
            experiment_type = self._select_experiment_type(analysis, constraints)
            methodology = self._select_methodology(analysis, experiment_type, constraints)
            
            # Design experimental variables and conditions
            variables = self._design_variables(analysis, methodology)
            conditions = self._design_conditions(variables, experiment_type)
            
            # Create experimental protocol
            protocol = self._create_protocol(
                analysis, methodology, variables, conditions
            )
            
            # Plan resource requirements
            resources = self._plan_resources(protocol, conditions, constraints)
            
            # Create timeline and milestones
            timeline = self._create_timeline(protocol, resources, constraints)
            
            # Define success criteria and risk assessment
            success_criteria = self._define_success_criteria(hypotheses, analysis)
            risks, mitigations = self._assess_risks(protocol, resources, timeline)
            
            # Estimate completion date
            estimated_completion = self._estimate_completion(timeline)
            
            experiment_plan = ExperimentPlan(
                plan_id=plan_id,
                title=f"Experiment: {analysis['title']}",
                research_question=research_question,
                hypotheses=hypotheses,
                experiment_type=experiment_type,
                methodology=methodology,
                variables=variables,
                conditions=conditions,
                protocol=protocol,
                resource_requirements=resources,
                timeline=timeline,
                success_criteria=success_criteria,
                risk_factors=risks,
                mitigation_strategies=mitigations,
                estimated_completion=estimated_completion
            )
            
            self.logger.info(f"Created experiment plan: {plan_id}")
            return experiment_plan
            
        except Exception as e:
            self.logger.error(f"Error planning experiment: {str(e)}")
            raise
    
    def create_validation_study(
        self,
        experiment_plan: ExperimentPlan,
        validation_requirements: Optional[Dict[str, Any]] = None
    ) -> ValidationStudy:
        """
        Create validation study for experiment plan.
        
        Args:
            experiment_plan: The experiment plan to validate
            validation_requirements: Specific validation requirements
            
        Returns:
            Validation study plan
        """
        try:
            study_id = str(uuid.uuid4())
            
            # Determine validation types needed
            validation_types = self._determine_validation_types(
                experiment_plan, validation_requirements
            )
            
            # Select validation methods
            validation_methods = self._select_validation_methods(
                experiment_plan, validation_types
            )
            
            # Define validation criteria
            validation_criteria = self._define_validation_criteria(
                experiment_plan, validation_methods
            )
            
            # Plan validation timeline
            validation_timeline = self._plan_validation_timeline(
                experiment_plan, validation_methods
            )
            
            # Calculate validation resources
            validation_resources = self._calculate_validation_resources(
                validation_methods, validation_timeline
            )
            
            validation_study = ValidationStudy(
                study_id=study_id,
                experiment_plan_id=experiment_plan.plan_id,
                validation_type=", ".join(validation_types),
                validation_methods=validation_methods,
                validation_criteria=validation_criteria,
                expected_outcomes=self._define_validation_outcomes(validation_criteria),
                validation_timeline=validation_timeline,
                resource_requirements=validation_resources
            )
            
            self.logger.info(f"Created validation study: {study_id}")
            return validation_study
            
        except Exception as e:
            self.logger.error(f"Error creating validation study: {str(e)}")
            raise
    
    def recommend_methodology(
        self,
        research_question: str,
        domain: str,
        constraints: Optional[Dict[str, Any]] = None
    ) -> List[MethodologyRecommendation]:
        """
        Recommend optimal experimental methodologies.
        
        Args:
            research_question: The research question
            domain: Research domain
            constraints: Optional constraints
            
        Returns:
            List of methodology recommendations ranked by suitability
        """
        try:
            analysis = self._analyze_research_question(research_question, domain)
            
            recommendations = []
            
            for methodology in MethodologyType:
                for exp_type in ExperimentType:
                    # Calculate suitability score
                    suitability = self._calculate_methodology_suitability(
                        methodology, exp_type, analysis, constraints
                    )
                    
                    if suitability > 0.3:  # Only include viable options
                        recommendation = MethodologyRecommendation(
                            methodology=methodology,
                            experiment_type=exp_type,
                            suitability_score=suitability,
                            advantages=self._get_methodology_advantages(
                                methodology, exp_type, analysis
                            ),
                            disadvantages=self._get_methodology_disadvantages(
                                methodology, exp_type, analysis
                            ),
                            resource_requirements=self._estimate_methodology_resources(
                                methodology, exp_type, analysis
                            ),
                            estimated_duration=self._estimate_methodology_duration(
                                methodology, exp_type, analysis
                            ),
                            confidence_level=self._calculate_confidence_level(
                                methodology, exp_type, analysis
                            )
                        )
                        recommendations.append(recommendation)
            
            # Sort by suitability score
            recommendations.sort(key=lambda x: x.suitability_score, reverse=True)
            
            self.logger.info(f"Generated {len(recommendations)} methodology recommendations")
            return recommendations[:5]  # Return top 5
            
        except Exception as e:
            self.logger.error(f"Error recommending methodology: {str(e)}")
            raise
    
    def optimize_experiment_design(
        self,
        experiment_plan: ExperimentPlan,
        optimization_goals: List[str]
    ) -> ExperimentOptimization:
        """
        Optimize experiment design for specific goals.
        
        Args:
            experiment_plan: Current experiment plan
            optimization_goals: Goals to optimize for (cost, time, accuracy, etc.)
            
        Returns:
            Optimization recommendations
        """
        try:
            optimization_id = str(uuid.uuid4())
            
            # Analyze current design metrics
            current_metrics = self._analyze_current_metrics(experiment_plan)
            
            # Generate optimization strategies
            strategies = []
            optimized_metrics = current_metrics.copy()
            
            for goal in optimization_goals:
                goal_strategies = self._generate_optimization_strategies(
                    experiment_plan, goal, current_metrics
                )
                strategies.extend(goal_strategies)
                
                # Update optimized metrics
                optimized_metrics.update(
                    self._calculate_optimized_metrics(
                        experiment_plan, goal, goal_strategies
                    )
                )
            
            # Identify trade-offs
            trade_offs = self._identify_trade_offs(
                current_metrics, optimized_metrics, optimization_goals
            )
            
            # Create implementation steps
            implementation_steps = self._create_implementation_steps(strategies)
            
            # Calculate confidence score
            confidence_score = self._calculate_optimization_confidence(
                strategies, current_metrics, optimized_metrics
            )
            
            optimization = ExperimentOptimization(
                optimization_id=optimization_id,
                experiment_plan_id=experiment_plan.plan_id,
                optimization_type=", ".join(optimization_goals),
                current_metrics=current_metrics,
                optimized_metrics=optimized_metrics,
                optimization_strategies=strategies,
                trade_offs=trade_offs,
                implementation_steps=implementation_steps,
                confidence_score=confidence_score
            )
            
            self.logger.info(f"Created optimization plan: {optimization_id}")
            return optimization
            
        except Exception as e:
            self.logger.error(f"Error optimizing experiment design: {str(e)}")
            raise
    
    def _analyze_research_question(
        self, 
        research_question: str, 
        domain: str
    ) -> Dict[str, Any]:
        """Analyze research question to extract key components."""
        # Simplified analysis - in practice would use NLP
        # Handle any domain gracefully
        valid_domains = ['engineering', 'chemistry', 'physics', 'biology', 'computer_science', 
                        'social_sciences', 'medicine', 'psychology', 'economics']
        
        # If domain not recognized, default to general
        if domain not in valid_domains:
            domain = 'general'
        
        analysis = {
            'title': research_question[:50] + "..." if len(research_question) > 50 else research_question,
            'domain': domain,
            'complexity': 'medium',  # Would be determined by analysis
            'variables_suggested': ['independent_var', 'dependent_var'],
            'sample_size_estimate': 100,
            'duration_estimate': timedelta(weeks=8)
        }
        return analysis
    
    def _generate_hypotheses(self, analysis: Dict[str, Any]) -> List[Hypothesis]:
        """Generate testable hypotheses from analysis."""
        hypotheses = []
        
        # Generate primary hypothesis
        hypothesis = Hypothesis(
            hypothesis_id=str(uuid.uuid4()),
            statement=f"Primary hypothesis for {analysis['title']}",
            null_hypothesis="No significant effect exists",
            alternative_hypothesis="Significant effect exists",
            variables_involved=analysis['variables_suggested']
        )
        hypotheses.append(hypothesis)
        
        return hypotheses
    
    def _select_experiment_type(
        self, 
        analysis: Dict[str, Any], 
        constraints: Optional[Dict[str, Any]]
    ) -> ExperimentType:
        """Select optimal experiment type."""
        # Simplified selection logic
        if analysis['complexity'] == 'high':
            return ExperimentType.FACTORIAL
        elif analysis['domain'] in ['software', 'computational']:
            return ExperimentType.CONTROLLED
        else:
            return ExperimentType.RANDOMIZED
    
    def _select_methodology(
        self,
        analysis: Dict[str, Any],
        experiment_type: ExperimentType,
        constraints: Optional[Dict[str, Any]]
    ) -> MethodologyType:
        """Select optimal methodology."""
        if analysis['domain'] in ['software', 'computational']:
            return MethodologyType.COMPUTATIONAL
        elif experiment_type == ExperimentType.FACTORIAL:
            return MethodologyType.QUANTITATIVE
        else:
            return MethodologyType.MIXED_METHODS
    
    def _design_variables(
        self,
        analysis: Dict[str, Any],
        methodology: MethodologyType
    ) -> List[ExperimentalVariable]:
        """Design experimental variables."""
        variables = []
        
        # Create independent variable
        independent_var = ExperimentalVariable(
            name="independent_variable",
            variable_type="independent",
            data_type="continuous",
            measurement_unit="units",
            description="Primary independent variable"
        )
        variables.append(independent_var)
        
        # Create dependent variable
        dependent_var = ExperimentalVariable(
            name="dependent_variable",
            variable_type="dependent",
            data_type="continuous",
            measurement_unit="outcome_units",
            description="Primary outcome variable"
        )
        variables.append(dependent_var)
        
        return variables
    
    def _design_conditions(
        self,
        variables: List[ExperimentalVariable],
        experiment_type: ExperimentType
    ) -> List[ExperimentalCondition]:
        """Design experimental conditions."""
        conditions = []
        
        # Create control condition
        control_condition = ExperimentalCondition(
            condition_id=str(uuid.uuid4()),
            name="Control",
            variables={"independent_variable": 0},
            sample_size=50,
            description="Control condition"
        )
        conditions.append(control_condition)
        
        # Create treatment condition
        treatment_condition = ExperimentalCondition(
            condition_id=str(uuid.uuid4()),
            name="Treatment",
            variables={"independent_variable": 1},
            sample_size=50,
            description="Treatment condition"
        )
        conditions.append(treatment_condition)
        
        return conditions
    
    def _create_protocol(
        self,
        analysis: Dict[str, Any],
        methodology: MethodologyType,
        variables: List[ExperimentalVariable],
        conditions: List[ExperimentalCondition]
    ) -> ExperimentalProtocol:
        """Create detailed experimental protocol."""
        protocol = ExperimentalProtocol(
            protocol_id=str(uuid.uuid4()),
            title=f"Protocol for {analysis['title']}",
            objective=f"Investigate {analysis['title']}",
            methodology=methodology,
            procedures=[
                "1. Prepare experimental setup",
                "2. Randomize participants to conditions",
                "3. Collect baseline measurements",
                "4. Apply experimental manipulation",
                "5. Collect outcome measurements",
                "6. Debrief participants"
            ],
            materials_required=[
                "Measurement instruments",
                "Data collection software",
                "Experimental materials"
            ],
            safety_considerations=[
                "Ensure participant safety",
                "Follow ethical guidelines",
                "Maintain data confidentiality"
            ],
            quality_controls=[
                "Calibrate instruments",
                "Train data collectors",
                "Implement double data entry"
            ],
            data_collection_methods=[
                "Automated data collection",
                "Manual observations",
                "Survey responses"
            ],
            analysis_plan="Statistical analysis using appropriate tests",
            estimated_duration=analysis['duration_estimate']
        )
        
        return protocol
    
    def _plan_resources(
        self,
        protocol: ExperimentalProtocol,
        conditions: List[ExperimentalCondition],
        constraints: Optional[Dict[str, Any]]
    ) -> List[ResourceRequirement]:
        """Plan resource requirements."""
        resources = []
        
        # Adjust costs based on constraints
        budget_factor = 1.0
        if constraints and 'budget' in constraints:
            budget_factor = min(1.0, constraints['budget'] / 10000.0)
        
        # Personnel
        personnel_cost = 5000.0 * budget_factor
        personnel = ResourceRequirement(
            resource_type="personnel",
            resource_name="Research Assistant",
            quantity_needed=max(1, int(2 * budget_factor)),
            duration_needed=protocol.estimated_duration,
            cost_estimate=personnel_cost
        )
        resources.append(personnel)
        
        # Equipment
        equipment_cost = 2000.0 * budget_factor
        equipment = ResourceRequirement(
            resource_type="equipment",
            resource_name="Measurement Equipment",
            quantity_needed=1,
            duration_needed=protocol.estimated_duration,
            cost_estimate=equipment_cost
        )
        resources.append(equipment)
        
        # Computational
        computational_cost = 500.0 * budget_factor
        computational = ResourceRequirement(
            resource_type="computational",
            resource_name="Computing Resources",
            quantity_needed=1,
            duration_needed=protocol.estimated_duration,
            cost_estimate=computational_cost
        )
        resources.append(computational)
        
        return resources
    
    def _create_timeline(
        self,
        protocol: ExperimentalProtocol,
        resources: List[ResourceRequirement],
        constraints: Optional[Dict[str, Any]]
    ) -> List[ExperimentMilestone]:
        """Create experiment timeline with milestones."""
        milestones = []
        start_date = datetime.now()
        
        # Planning phase
        planning = ExperimentMilestone(
            milestone_id=str(uuid.uuid4()),
            name="Planning Complete",
            description="Experiment planning and setup complete",
            target_date=start_date + timedelta(weeks=1),
            deliverables=["Finalized protocol", "Resource allocation"],
            completion_criteria=["All resources secured", "Protocol approved"]
        )
        milestones.append(planning)
        
        # Data collection phase
        data_collection = ExperimentMilestone(
            milestone_id=str(uuid.uuid4()),
            name="Data Collection Complete",
            description="All experimental data collected",
            target_date=start_date + timedelta(weeks=6),
            dependencies=[planning.milestone_id],
            deliverables=["Complete dataset", "Quality control report"],
            completion_criteria=["All participants completed", "Data quality verified"]
        )
        milestones.append(data_collection)
        
        # Analysis phase
        analysis = ExperimentMilestone(
            milestone_id=str(uuid.uuid4()),
            name="Analysis Complete",
            description="Statistical analysis completed",
            target_date=start_date + timedelta(weeks=8),
            dependencies=[data_collection.milestone_id],
            deliverables=["Analysis results", "Statistical report"],
            completion_criteria=["All analyses completed", "Results validated"]
        )
        milestones.append(analysis)
        
        return milestones
    
    def _define_success_criteria(
        self,
        hypotheses: List[Hypothesis],
        analysis: Dict[str, Any]
    ) -> List[str]:
        """Define success criteria for experiment."""
        criteria = [
            "Statistical significance achieved (p < 0.05)",
            "Effect size meets practical significance threshold",
            "Data quality standards met",
            "All planned analyses completed",
            "Results are reproducible"
        ]
        return criteria
    
    def _assess_risks(
        self,
        protocol: ExperimentalProtocol,
        resources: List[ResourceRequirement],
        timeline: List[ExperimentMilestone]
    ) -> Tuple[List[str], List[str]]:
        """Assess risks and mitigation strategies."""
        risks = [
            "Participant dropout",
            "Equipment failure",
            "Data quality issues",
            "Timeline delays",
            "Budget overruns"
        ]
        
        mitigations = [
            "Recruit additional participants",
            "Have backup equipment available",
            "Implement quality control procedures",
            "Build buffer time into schedule",
            "Monitor budget closely"
        ]
        
        return risks, mitigations
    
    def _estimate_completion(self, timeline: List[ExperimentMilestone]) -> datetime:
        """Estimate experiment completion date."""
        if timeline:
            return max(milestone.target_date for milestone in timeline)
        return datetime.now() + timedelta(weeks=12)
    
    def _initialize_methodology_database(self) -> Dict[str, Any]:
        """Initialize methodology knowledge database."""
        return {
            'quantitative': {
                'strengths': ['Statistical power', 'Generalizability'],
                'weaknesses': ['Limited depth', 'Context loss'],
                'suitable_domains': ['engineering', 'physics', 'chemistry']
            },
            'qualitative': {
                'strengths': ['Rich insights', 'Context understanding'],
                'weaknesses': ['Limited generalizability', 'Subjective'],
                'suitable_domains': ['social sciences', 'user research']
            }
        }
    
    def _initialize_design_templates(self) -> Dict[str, Any]:
        """Initialize experiment design templates."""
        return {
            'controlled': {
                'min_conditions': 2,
                'recommended_sample_size': 30,
                'typical_duration': timedelta(weeks=4)
            },
            'factorial': {
                'min_conditions': 4,
                'recommended_sample_size': 100,
                'typical_duration': timedelta(weeks=8)
            }
        }
    
    # Additional helper methods for validation, optimization, etc.
    def _determine_validation_types(self, experiment_plan, requirements):
        """Determine types of validation needed."""
        return ['internal', 'construct']
    
    def _select_validation_methods(self, experiment_plan, validation_types):
        """Select appropriate validation methods."""
        return ['peer review', 'statistical validation', 'replication']
    
    def _define_validation_criteria(self, experiment_plan, validation_methods):
        """Define criteria for validation."""
        return ['Expert approval', 'Statistical significance', 'Reproducibility']
    
    def _plan_validation_timeline(self, experiment_plan, validation_methods):
        """Plan timeline for validation activities."""
        return []  # Simplified for now
    
    def _calculate_validation_resources(self, validation_methods, timeline):
        """Calculate resources needed for validation."""
        return []  # Simplified for now
    
    def _define_validation_outcomes(self, validation_criteria):
        """Define expected validation outcomes."""
        return ['Validated design', 'Approved protocol']
    
    def _calculate_methodology_suitability(self, methodology, exp_type, analysis, constraints):
        """Calculate suitability score for methodology."""
        return 0.8  # Simplified scoring
    
    def _get_methodology_advantages(self, methodology, exp_type, analysis):
        """Get advantages of methodology."""
        return ['High reliability', 'Cost effective']
    
    def _get_methodology_disadvantages(self, methodology, exp_type, analysis):
        """Get disadvantages of methodology."""
        return ['Time intensive', 'Resource demanding']
    
    def _estimate_methodology_resources(self, methodology, exp_type, analysis):
        """Estimate resources for methodology."""
        return []  # Simplified for now
    
    def _estimate_methodology_duration(self, methodology, exp_type, analysis):
        """Estimate duration for methodology."""
        return timedelta(weeks=6)
    
    def _calculate_confidence_level(self, methodology, exp_type, analysis):
        """Calculate confidence level for methodology."""
        return 0.85
    
    def _analyze_current_metrics(self, experiment_plan):
        """Analyze current design metrics."""
        return {
            'cost': 10000.0,
            'duration_weeks': 8,
            'accuracy': 0.85,
            'reliability': 0.90
        }
    
    def _generate_optimization_strategies(self, experiment_plan, goal, current_metrics):
        """Generate optimization strategies for goal."""
        return [f'Optimize for {goal}']
    
    def _calculate_optimized_metrics(self, experiment_plan, goal, strategies):
        """Calculate optimized metrics."""
        return {'cost': 8000.0}  # Example optimization
    
    def _identify_trade_offs(self, current_metrics, optimized_metrics, goals):
        """Identify trade-offs in optimization."""
        return ['Cost reduction may impact accuracy']
    
    def _create_implementation_steps(self, strategies):
        """Create implementation steps for optimization."""
        return ['Step 1: Implement strategy', 'Step 2: Monitor results']
    
    def _calculate_optimization_confidence(self, strategies, current_metrics, optimized_metrics):
        """Calculate confidence in optimization."""
        return 0.75