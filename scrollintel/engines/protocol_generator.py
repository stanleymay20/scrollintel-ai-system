"""
Protocol generation system for experimental design.
"""
import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import logging
from dataclasses import dataclass

from ..models.experimental_design_models import (
    ExperimentalProtocol, MethodologyType, ExperimentType,
    ExperimentalVariable, ExperimentalCondition, Hypothesis
)


@dataclass
class ProtocolTemplate:
    """Template for generating experimental protocols."""
    template_id: str
    name: str
    methodology: MethodologyType
    experiment_types: List[ExperimentType]
    procedure_steps: List[str]
    required_materials: List[str]
    safety_requirements: List[str]
    quality_controls: List[str]
    data_collection_methods: List[str]
    analysis_approaches: List[str]
    typical_duration: timedelta
    complexity_level: str  # low, medium, high


@dataclass
class ProtocolOptimization:
    """Protocol optimization result."""
    original_protocol: ExperimentalProtocol
    optimized_protocol: ExperimentalProtocol
    optimization_strategies: List[str]
    improvements: Dict[str, Any]
    trade_offs: List[str]
    confidence_score: float


@dataclass
class ProtocolValidation:
    """Protocol validation result."""
    protocol_id: str
    validation_checks: List[str]
    passed_checks: List[str]
    failed_checks: List[str]
    warnings: List[str]
    recommendations: List[str]
    overall_score: float
    is_valid: bool


class ProtocolGenerator:
    """
    System for generating detailed experimental protocols and procedures.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.protocol_templates = self._initialize_protocol_templates()
        self.standardization_rules = self._initialize_standardization_rules()
        self.quality_criteria = self._initialize_quality_criteria()
    
    def generate_protocol(
        self,
        experiment_context: Dict[str, Any],
        methodology: MethodologyType,
        experiment_type: ExperimentType,
        variables: List[ExperimentalVariable],
        conditions: List[ExperimentalCondition],
        hypotheses: List[Hypothesis]
    ) -> ExperimentalProtocol:
        """
        Generate detailed experimental protocol.
        
        Args:
            experiment_context: Context including domain, objectives, constraints
            methodology: Experimental methodology
            experiment_type: Type of experiment
            variables: Experimental variables
            conditions: Experimental conditions
            hypotheses: Research hypotheses
            
        Returns:
            Detailed experimental protocol
        """
        try:
            # Select appropriate template
            template = self._select_protocol_template(
                methodology, experiment_type, experiment_context
            )
            
            # Generate protocol components
            procedures = self._generate_procedures(
                template, variables, conditions, experiment_context
            )
            
            materials = self._generate_materials_list(
                template, variables, conditions, experiment_context
            )
            
            safety_considerations = self._generate_safety_considerations(
                template, experiment_context
            )
            
            quality_controls = self._generate_quality_controls(
                template, variables, conditions
            )
            
            data_collection_methods = self._generate_data_collection_methods(
                template, variables, methodology
            )
            
            analysis_plan = self._generate_analysis_plan(
                hypotheses, variables, methodology
            )
            
            # Estimate duration
            estimated_duration = self._estimate_protocol_duration(
                template, procedures, conditions, experiment_context
            )
            
            # Create protocol
            protocol = ExperimentalProtocol(
                protocol_id=str(uuid.uuid4()),
                title=f"Protocol for {experiment_context.get('title', 'Experiment')}",
                objective=experiment_context.get('objective', 'Research objective'),
                methodology=methodology,
                procedures=procedures,
                materials_required=materials,
                safety_considerations=safety_considerations,
                quality_controls=quality_controls,
                data_collection_methods=data_collection_methods,
                analysis_plan=analysis_plan,
                estimated_duration=estimated_duration
            )
            
            self.logger.info(f"Generated protocol: {protocol.protocol_id}")
            return protocol
            
        except Exception as e:
            self.logger.error(f"Error generating protocol: {str(e)}")
            raise
    
    def optimize_protocol(
        self,
        protocol: ExperimentalProtocol,
        optimization_goals: List[str],
        constraints: Optional[Dict[str, Any]] = None
    ) -> ProtocolOptimization:
        """
        Optimize experimental protocol for specific goals.
        
        Args:
            protocol: Original protocol
            optimization_goals: Goals (efficiency, accuracy, cost, etc.)
            constraints: Optional constraints
            
        Returns:
            Protocol optimization result
        """
        try:
            # Analyze current protocol
            current_metrics = self._analyze_protocol_metrics(protocol)
            
            # Apply optimization strategies
            optimized_protocol = self._create_optimized_protocol(protocol)
            optimization_strategies = []
            improvements = {}
            
            for goal in optimization_goals:
                if goal == "efficiency":
                    strategies, metrics = self._optimize_for_efficiency(
                        optimized_protocol, constraints
                    )
                    optimization_strategies.extend(strategies)
                    improvements.update(metrics)
                
                elif goal == "accuracy":
                    strategies, metrics = self._optimize_for_accuracy(
                        optimized_protocol, constraints
                    )
                    optimization_strategies.extend(strategies)
                    improvements.update(metrics)
                
                elif goal == "cost":
                    strategies, metrics = self._optimize_for_cost(
                        optimized_protocol, constraints
                    )
                    optimization_strategies.extend(strategies)
                    improvements.update(metrics)
                
                elif goal == "reliability":
                    strategies, metrics = self._optimize_for_reliability(
                        optimized_protocol, constraints
                    )
                    optimization_strategies.extend(strategies)
                    improvements.update(metrics)
            
            # Identify trade-offs
            trade_offs = self._identify_optimization_trade_offs(
                protocol, optimized_protocol, optimization_goals
            )
            
            # Calculate confidence score
            confidence_score = self._calculate_optimization_confidence(
                optimization_strategies, improvements, trade_offs
            )
            
            optimization_result = ProtocolOptimization(
                original_protocol=protocol,
                optimized_protocol=optimized_protocol,
                optimization_strategies=optimization_strategies,
                improvements=improvements,
                trade_offs=trade_offs,
                confidence_score=confidence_score
            )
            
            self.logger.info(f"Optimized protocol with confidence {confidence_score:.2f}")
            return optimization_result
            
        except Exception as e:
            self.logger.error(f"Error optimizing protocol: {str(e)}")
            raise
    
    def standardize_protocol(
        self,
        protocol: ExperimentalProtocol,
        standards: Optional[List[str]] = None
    ) -> ExperimentalProtocol:
        """
        Standardize protocol according to best practices.
        
        Args:
            protocol: Protocol to standardize
            standards: Specific standards to apply
            
        Returns:
            Standardized protocol
        """
        try:
            standardized_protocol = self._create_protocol_copy(protocol)
            
            # Apply standardization rules
            if not standards:
                standards = list(self.standardization_rules.keys())
            
            for standard in standards:
                if standard in self.standardization_rules:
                    rule = self.standardization_rules[standard]
                    standardized_protocol = self._apply_standardization_rule(
                        standardized_protocol, rule
                    )
            
            # Ensure protocol completeness
            standardized_protocol = self._ensure_protocol_completeness(
                standardized_protocol
            )
            
            # Update protocol ID to indicate standardization
            standardized_protocol.protocol_id = f"{protocol.protocol_id}_standardized"
            
            self.logger.info(f"Standardized protocol: {standardized_protocol.protocol_id}")
            return standardized_protocol
            
        except Exception as e:
            self.logger.error(f"Error standardizing protocol: {str(e)}")
            raise
    
    def validate_protocol(
        self,
        protocol: ExperimentalProtocol,
        validation_criteria: Optional[List[str]] = None
    ) -> ProtocolValidation:
        """
        Validate protocol quality and completeness.
        
        Args:
            protocol: Protocol to validate
            validation_criteria: Specific criteria to check
            
        Returns:
            Protocol validation result
        """
        try:
            if not validation_criteria:
                validation_criteria = list(self.quality_criteria.keys())
            
            validation_checks = []
            passed_checks = []
            failed_checks = []
            warnings = []
            recommendations = []
            
            for criterion in validation_criteria:
                if criterion in self.quality_criteria:
                    check_result = self._perform_validation_check(
                        protocol, criterion, self.quality_criteria[criterion]
                    )
                    
                    validation_checks.append(criterion)
                    
                    if check_result['passed']:
                        passed_checks.append(criterion)
                    else:
                        failed_checks.append(criterion)
                        if check_result.get('warning'):
                            warnings.append(check_result['warning'])
                        if check_result.get('recommendation'):
                            recommendations.append(check_result['recommendation'])
            
            # Calculate overall score
            overall_score = len(passed_checks) / len(validation_checks) if validation_checks else 0.0
            is_valid = overall_score >= 0.8 and len(failed_checks) == 0
            
            validation_result = ProtocolValidation(
                protocol_id=protocol.protocol_id,
                validation_checks=validation_checks,
                passed_checks=passed_checks,
                failed_checks=failed_checks,
                warnings=warnings,
                recommendations=recommendations,
                overall_score=overall_score,
                is_valid=is_valid
            )
            
            self.logger.info(
                f"Validated protocol {protocol.protocol_id}: "
                f"Score {overall_score:.2f}, Valid: {is_valid}"
            )
            
            return validation_result
            
        except Exception as e:
            self.logger.error(f"Error validating protocol: {str(e)}")
            raise
    
    def _initialize_protocol_templates(self) -> Dict[str, ProtocolTemplate]:
        """Initialize protocol templates database."""
        templates = {}
        
        # Quantitative controlled experiment template
        templates['quantitative_controlled'] = ProtocolTemplate(
            template_id='quantitative_controlled',
            name='Quantitative Controlled Experiment',
            methodology=MethodologyType.QUANTITATIVE,
            experiment_types=[ExperimentType.CONTROLLED, ExperimentType.RANDOMIZED],
            procedure_steps=[
                'Define experimental setup',
                'Randomize participants/subjects to conditions',
                'Collect baseline measurements',
                'Apply experimental manipulation',
                'Collect outcome measurements',
                'Conduct post-experiment assessments',
                'Debrief participants (if applicable)'
            ],
            required_materials=[
                'Measurement instruments',
                'Data collection software',
                'Randomization tools',
                'Experimental materials'
            ],
            safety_requirements=[
                'Ensure participant safety',
                'Follow ethical guidelines',
                'Maintain data confidentiality',
                'Emergency procedures in place'
            ],
            quality_controls=[
                'Calibrate instruments before use',
                'Train data collectors',
                'Implement double data entry',
                'Monitor data quality continuously'
            ],
            data_collection_methods=[
                'Automated data collection',
                'Standardized measurements',
                'Real-time monitoring',
                'Quality checks'
            ],
            analysis_approaches=[
                'Descriptive statistics',
                'Inferential statistics',
                'Effect size calculations',
                'Confidence intervals'
            ],
            typical_duration=timedelta(weeks=8),
            complexity_level='medium'
        )
        
        # Qualitative observational template
        templates['qualitative_observational'] = ProtocolTemplate(
            template_id='qualitative_observational',
            name='Qualitative Observational Study',
            methodology=MethodologyType.QUALITATIVE,
            experiment_types=[ExperimentType.OBSERVATIONAL, ExperimentType.LONGITUDINAL],
            procedure_steps=[
                'Define observation framework',
                'Select observation sites/contexts',
                'Establish rapport with participants',
                'Conduct systematic observations',
                'Record detailed field notes',
                'Conduct follow-up interviews',
                'Member checking and validation'
            ],
            required_materials=[
                'Recording equipment',
                'Field notebooks',
                'Interview guides',
                'Transcription software'
            ],
            safety_requirements=[
                'Respect participant privacy',
                'Maintain confidentiality',
                'Follow ethical guidelines',
                'Cultural sensitivity'
            ],
            quality_controls=[
                'Multiple observers',
                'Inter-rater reliability',
                'Member checking',
                'Triangulation of data sources'
            ],
            data_collection_methods=[
                'Participant observation',
                'In-depth interviews',
                'Document analysis',
                'Photo/video documentation'
            ],
            analysis_approaches=[
                'Thematic analysis',
                'Grounded theory',
                'Narrative analysis',
                'Content analysis'
            ],
            typical_duration=timedelta(weeks=12),
            complexity_level='high'
        )
        
        # Computational simulation template
        templates['computational_simulation'] = ProtocolTemplate(
            template_id='computational_simulation',
            name='Computational Simulation Study',
            methodology=MethodologyType.COMPUTATIONAL,
            experiment_types=[ExperimentType.CONTROLLED, ExperimentType.FACTORIAL],
            procedure_steps=[
                'Define simulation parameters',
                'Implement computational model',
                'Validate model against known results',
                'Design simulation experiments',
                'Execute simulation runs',
                'Collect and analyze results',
                'Sensitivity analysis'
            ],
            required_materials=[
                'Computing resources',
                'Simulation software',
                'Model validation data',
                'Analysis tools'
            ],
            safety_requirements=[
                'Data backup procedures',
                'Version control',
                'Computational resource limits',
                'Result verification'
            ],
            quality_controls=[
                'Model validation',
                'Reproducibility checks',
                'Parameter sensitivity analysis',
                'Cross-validation'
            ],
            data_collection_methods=[
                'Automated data logging',
                'Parameter sweeps',
                'Monte Carlo sampling',
                'Statistical sampling'
            ],
            analysis_approaches=[
                'Statistical analysis',
                'Sensitivity analysis',
                'Uncertainty quantification',
                'Model comparison'
            ],
            typical_duration=timedelta(weeks=4),
            complexity_level='medium'
        )
        
        return templates
    
    def _initialize_standardization_rules(self) -> Dict[str, Dict[str, Any]]:
        """Initialize protocol standardization rules."""
        return {
            'procedure_clarity': {
                'description': 'Ensure procedures are clear and detailed',
                'requirements': [
                    'Each step must be specific and actionable',
                    'Include timing information',
                    'Specify responsible personnel',
                    'Include decision points and alternatives'
                ]
            },
            'safety_compliance': {
                'description': 'Ensure safety requirements are comprehensive',
                'requirements': [
                    'Risk assessment completed',
                    'Emergency procedures defined',
                    'Personal protective equipment specified',
                    'Hazard mitigation strategies included'
                ]
            },
            'quality_assurance': {
                'description': 'Implement quality control measures',
                'requirements': [
                    'Quality checkpoints defined',
                    'Validation procedures included',
                    'Error detection mechanisms',
                    'Corrective action procedures'
                ]
            },
            'data_integrity': {
                'description': 'Ensure data collection integrity',
                'requirements': [
                    'Data collection procedures standardized',
                    'Data validation rules defined',
                    'Backup and recovery procedures',
                    'Chain of custody maintained'
                ]
            }
        }
    
    def _initialize_quality_criteria(self) -> Dict[str, Dict[str, Any]]:
        """Initialize protocol quality criteria."""
        return {
            'completeness': {
                'description': 'Protocol includes all necessary components',
                'checks': [
                    'Objective clearly stated',
                    'Procedures detailed',
                    'Materials list complete',
                    'Safety considerations included',
                    'Quality controls defined',
                    'Analysis plan specified'
                ]
            },
            'clarity': {
                'description': 'Protocol is clear and unambiguous',
                'checks': [
                    'Steps are specific',
                    'Technical terms defined',
                    'Timing specified',
                    'Responsibilities assigned'
                ]
            },
            'feasibility': {
                'description': 'Protocol is practically feasible',
                'checks': [
                    'Resources available',
                    'Timeline realistic',
                    'Skills requirements met',
                    'Equipment accessible'
                ]
            },
            'reproducibility': {
                'description': 'Protocol enables reproducible results',
                'checks': [
                    'Procedures standardized',
                    'Parameters specified',
                    'Controls included',
                    'Validation methods defined'
                ]
            }
        }
    
    def _select_protocol_template(
        self,
        methodology: MethodologyType,
        experiment_type: ExperimentType,
        context: Dict[str, Any]
    ) -> ProtocolTemplate:
        """Select appropriate protocol template."""
        # Find templates matching methodology and experiment type
        matching_templates = []
        
        for template in self.protocol_templates.values():
            if (template.methodology == methodology and 
                experiment_type in template.experiment_types):
                matching_templates.append(template)
        
        if not matching_templates:
            # Fallback to first template matching methodology
            for template in self.protocol_templates.values():
                if template.methodology == methodology:
                    matching_templates.append(template)
                    break
        
        if not matching_templates:
            # Ultimate fallback
            return list(self.protocol_templates.values())[0]
        
        # Select best matching template (simplified selection)
        return matching_templates[0]
    
    def _generate_procedures(
        self,
        template: ProtocolTemplate,
        variables: List[ExperimentalVariable],
        conditions: List[ExperimentalCondition],
        context: Dict[str, Any]
    ) -> List[str]:
        """Generate detailed procedures from template."""
        procedures = template.procedure_steps.copy()
        
        # Customize procedures based on variables and conditions
        if variables:
            variable_setup = f"Configure variables: {', '.join(v.name for v in variables)}"
            procedures.insert(1, variable_setup)
        
        if conditions:
            condition_setup = f"Prepare {len(conditions)} experimental conditions"
            procedures.insert(2, condition_setup)
        
        # Add domain-specific procedures
        domain = context.get('domain', 'general')
        if domain == 'engineering':
            procedures.append('Perform system validation tests')
        elif domain == 'biology':
            procedures.append('Maintain biological sample integrity')
        elif domain == 'psychology':
            procedures.append('Conduct participant debriefing')
        
        return procedures
    
    def _generate_materials_list(
        self,
        template: ProtocolTemplate,
        variables: List[ExperimentalVariable],
        conditions: List[ExperimentalCondition],
        context: Dict[str, Any]
    ) -> List[str]:
        """Generate materials list from template."""
        materials = template.required_materials.copy()
        
        # Add variable-specific materials
        for variable in variables:
            if variable.measurement_unit:
                materials.append(f"Measurement device for {variable.name}")
        
        # Add condition-specific materials
        for condition in conditions:
            materials.append(f"Materials for condition: {condition.name}")
        
        # Add domain-specific materials
        domain = context.get('domain', 'general')
        if domain == 'chemistry':
            materials.extend(['Chemical reagents', 'Safety equipment', 'Fume hood'])
        elif domain == 'physics':
            materials.extend(['Precision instruments', 'Calibration standards'])
        
        return list(set(materials))  # Remove duplicates
    
    def _generate_safety_considerations(
        self,
        template: ProtocolTemplate,
        context: Dict[str, Any]
    ) -> List[str]:
        """Generate safety considerations from template."""
        safety = template.safety_requirements.copy()
        
        # Add domain-specific safety requirements
        domain = context.get('domain', 'general')
        if domain == 'chemistry':
            safety.extend([
                'Chemical safety protocols',
                'Waste disposal procedures',
                'Ventilation requirements'
            ])
        elif domain == 'biology':
            safety.extend([
                'Biosafety protocols',
                'Contamination prevention',
                'Specimen handling procedures'
            ])
        elif domain == 'engineering':
            safety.extend([
                'Equipment safety checks',
                'Electrical safety protocols',
                'Mechanical hazard prevention'
            ])
        
        return safety
    
    def _generate_quality_controls(
        self,
        template: ProtocolTemplate,
        variables: List[ExperimentalVariable],
        conditions: List[ExperimentalCondition]
    ) -> List[str]:
        """Generate quality control measures."""
        quality_controls = template.quality_controls.copy()
        
        # Add variable-specific quality controls
        for variable in variables:
            if variable.variable_type == 'dependent':
                quality_controls.append(f"Validate {variable.name} measurements")
        
        # Add condition-specific quality controls
        if len(conditions) > 1:
            quality_controls.append('Verify condition randomization')
            quality_controls.append('Monitor condition fidelity')
        
        return quality_controls
    
    def _generate_data_collection_methods(
        self,
        template: ProtocolTemplate,
        variables: List[ExperimentalVariable],
        methodology: MethodologyType
    ) -> List[str]:
        """Generate data collection methods."""
        methods = template.data_collection_methods.copy()
        
        # Add variable-specific methods
        for variable in variables:
            if variable.data_type == 'continuous':
                methods.append(f"Continuous monitoring of {variable.name}")
            elif variable.data_type == 'categorical':
                methods.append(f"Categorical assessment of {variable.name}")
        
        # Add methodology-specific methods
        if methodology == MethodologyType.QUANTITATIVE:
            methods.append('Statistical sampling procedures')
        elif methodology == MethodologyType.QUALITATIVE:
            methods.append('Narrative data collection')
        
        return methods
    
    def _generate_analysis_plan(
        self,
        hypotheses: List[Hypothesis],
        variables: List[ExperimentalVariable],
        methodology: MethodologyType
    ) -> str:
        """Generate analysis plan."""
        plan_components = []
        
        # Add hypothesis-specific analysis
        for hypothesis in hypotheses:
            plan_components.append(f"Test hypothesis: {hypothesis.statement}")
        
        # Add variable-specific analysis
        dependent_vars = [v for v in variables if v.variable_type == 'dependent']
        independent_vars = [v for v in variables if v.variable_type == 'independent']
        
        if dependent_vars and independent_vars:
            plan_components.append(
                f"Analyze relationship between {independent_vars[0].name} and {dependent_vars[0].name}"
            )
        
        # Add methodology-specific analysis
        if methodology == MethodologyType.QUANTITATIVE:
            plan_components.extend([
                'Descriptive statistics for all variables',
                'Inferential statistical tests',
                'Effect size calculations',
                'Confidence interval estimation'
            ])
        elif methodology == MethodologyType.QUALITATIVE:
            plan_components.extend([
                'Thematic analysis of qualitative data',
                'Pattern identification',
                'Narrative synthesis',
                'Member validation'
            ])
        elif methodology == MethodologyType.COMPUTATIONAL:
            plan_components.extend([
                'Statistical analysis of simulation results',
                'Sensitivity analysis',
                'Parameter optimization',
                'Model validation'
            ])
        
        return '; '.join(plan_components)
    
    def _estimate_protocol_duration(
        self,
        template: ProtocolTemplate,
        procedures: List[str],
        conditions: List[ExperimentalCondition],
        context: Dict[str, Any]
    ) -> timedelta:
        """Estimate protocol execution duration."""
        base_duration = template.typical_duration
        
        # Adjust for number of procedures
        procedure_factor = len(procedures) / len(template.procedure_steps)
        
        # Adjust for number of conditions
        condition_factor = max(1.0, len(conditions) / 2.0)
        
        # Adjust for complexity
        complexity = context.get('complexity', 'medium')
        complexity_factors = {'low': 0.8, 'medium': 1.0, 'high': 1.3}
        complexity_factor = complexity_factors.get(complexity, 1.0)
        
        # Calculate adjusted duration
        adjusted_duration = base_duration * procedure_factor * condition_factor * complexity_factor
        
        return adjusted_duration
    
    def _analyze_protocol_metrics(self, protocol: ExperimentalProtocol) -> Dict[str, Any]:
        """Analyze current protocol metrics."""
        return {
            'procedure_count': len(protocol.procedures),
            'material_count': len(protocol.materials_required),
            'safety_count': len(protocol.safety_considerations),
            'quality_control_count': len(protocol.quality_controls),
            'duration_days': protocol.estimated_duration.days,
            'complexity_score': self._calculate_complexity_score(protocol)
        }
    
    def _calculate_complexity_score(self, protocol: ExperimentalProtocol) -> float:
        """Calculate protocol complexity score."""
        # Simplified complexity calculation
        procedure_complexity = len(protocol.procedures) * 0.1
        material_complexity = len(protocol.materials_required) * 0.05
        safety_complexity = len(protocol.safety_considerations) * 0.1
        
        return min(1.0, procedure_complexity + material_complexity + safety_complexity)
    
    def _create_optimized_protocol(self, protocol: ExperimentalProtocol) -> ExperimentalProtocol:
        """Create a copy of protocol for optimization."""
        return ExperimentalProtocol(
            protocol_id=f"{protocol.protocol_id}_optimized",
            title=protocol.title,
            objective=protocol.objective,
            methodology=protocol.methodology,
            procedures=protocol.procedures.copy(),
            materials_required=protocol.materials_required.copy(),
            safety_considerations=protocol.safety_considerations.copy(),
            quality_controls=protocol.quality_controls.copy(),
            data_collection_methods=protocol.data_collection_methods.copy(),
            analysis_plan=protocol.analysis_plan,
            estimated_duration=protocol.estimated_duration
        )
    
    def _optimize_for_efficiency(
        self,
        protocol: ExperimentalProtocol,
        constraints: Optional[Dict[str, Any]]
    ) -> Tuple[List[str], Dict[str, Any]]:
        """Optimize protocol for efficiency."""
        strategies = [
            'Streamline procedures',
            'Parallelize independent steps',
            'Automate data collection',
            'Reduce redundant quality checks'
        ]
        
        # Reduce duration by 15%
        protocol.estimated_duration = timedelta(
            days=int(protocol.estimated_duration.days * 0.85)
        )
        
        # Streamline procedures
        if len(protocol.procedures) > 5:
            protocol.procedures = protocol.procedures[:5] + ['Combined remaining steps']
        
        metrics = {
            'duration_reduction': 0.15,
            'procedure_streamlining': 0.20
        }
        
        return strategies, metrics
    
    def _optimize_for_accuracy(
        self,
        protocol: ExperimentalProtocol,
        constraints: Optional[Dict[str, Any]]
    ) -> Tuple[List[str], Dict[str, Any]]:
        """Optimize protocol for accuracy."""
        strategies = [
            'Add validation steps',
            'Increase quality controls',
            'Implement redundant measurements',
            'Add calibration procedures'
        ]
        
        # Add quality controls
        protocol.quality_controls.extend([
            'Additional measurement validation',
            'Cross-validation procedures',
            'Calibration checks'
        ])
        
        metrics = {
            'accuracy_improvement': 0.25,
            'quality_control_increase': 0.30
        }
        
        return strategies, metrics
    
    def _optimize_for_cost(
        self,
        protocol: ExperimentalProtocol,
        constraints: Optional[Dict[str, Any]]
    ) -> Tuple[List[str], Dict[str, Any]]:
        """Optimize protocol for cost reduction."""
        strategies = [
            'Reduce material requirements',
            'Use alternative methods',
            'Minimize resource usage',
            'Optimize sample sizes'
        ]
        
        # Reduce materials
        if len(protocol.materials_required) > 3:
            protocol.materials_required = protocol.materials_required[:3]
        
        metrics = {
            'cost_reduction': 0.20,
            'material_optimization': 0.25
        }
        
        return strategies, metrics
    
    def _optimize_for_reliability(
        self,
        protocol: ExperimentalProtocol,
        constraints: Optional[Dict[str, Any]]
    ) -> Tuple[List[str], Dict[str, Any]]:
        """Optimize protocol for reliability."""
        strategies = [
            'Add replication procedures',
            'Implement error checking',
            'Standardize all steps',
            'Add verification points'
        ]
        
        # Add reliability measures
        protocol.procedures.append('Replicate key measurements')
        protocol.quality_controls.append('Inter-rater reliability checks')
        
        metrics = {
            'reliability_improvement': 0.30,
            'standardization_increase': 0.25
        }
        
        return strategies, metrics
    
    def _identify_optimization_trade_offs(
        self,
        original: ExperimentalProtocol,
        optimized: ExperimentalProtocol,
        goals: List[str]
    ) -> List[str]:
        """Identify trade-offs from optimization."""
        trade_offs = []
        
        if 'efficiency' in goals:
            trade_offs.append('Reduced duration may impact thoroughness')
        
        if 'cost' in goals:
            trade_offs.append('Cost reduction may affect quality')
        
        if 'accuracy' in goals and 'efficiency' in goals:
            trade_offs.append('Accuracy improvements may increase duration')
        
        return trade_offs
    
    def _calculate_optimization_confidence(
        self,
        strategies: List[str],
        improvements: Dict[str, Any],
        trade_offs: List[str]
    ) -> float:
        """Calculate confidence in optimization."""
        base_confidence = 0.8
        
        # Reduce confidence based on number of trade-offs
        confidence_reduction = len(trade_offs) * 0.1
        
        # Adjust based on improvement magnitude
        avg_improvement = sum(improvements.values()) / len(improvements) if improvements else 0
        if avg_improvement > 0.3:
            confidence_reduction += 0.1  # Large improvements may be risky
        
        return max(0.1, base_confidence - confidence_reduction)
    
    def _create_protocol_copy(self, protocol: ExperimentalProtocol) -> ExperimentalProtocol:
        """Create a copy of the protocol."""
        return ExperimentalProtocol(
            protocol_id=protocol.protocol_id,
            title=protocol.title,
            objective=protocol.objective,
            methodology=protocol.methodology,
            procedures=protocol.procedures.copy(),
            materials_required=protocol.materials_required.copy(),
            safety_considerations=protocol.safety_considerations.copy(),
            quality_controls=protocol.quality_controls.copy(),
            data_collection_methods=protocol.data_collection_methods.copy(),
            analysis_plan=protocol.analysis_plan,
            estimated_duration=protocol.estimated_duration
        )
    
    def _apply_standardization_rule(
        self,
        protocol: ExperimentalProtocol,
        rule: Dict[str, Any]
    ) -> ExperimentalProtocol:
        """Apply a standardization rule to protocol."""
        # Apply rule based on description
        description = rule.get('description', '')
        
        if 'procedure_clarity' in description.lower():
            # Add timing information to procedures
            enhanced_procedures = []
            for i, procedure in enumerate(protocol.procedures):
                enhanced_procedures.append(f"Step {i+1}: {procedure} (Est. time: 30 min)")
            protocol.procedures = enhanced_procedures
        
        elif 'safety_compliance' in description.lower():
            # Add comprehensive safety measures
            if 'Risk assessment completed' not in protocol.safety_considerations:
                protocol.safety_considerations.append('Risk assessment completed')
            if 'Emergency procedures defined' not in protocol.safety_considerations:
                protocol.safety_considerations.append('Emergency procedures defined')
        
        return protocol
    
    def _ensure_protocol_completeness(
        self,
        protocol: ExperimentalProtocol
    ) -> ExperimentalProtocol:
        """Ensure protocol has all required components."""
        # Ensure minimum requirements
        if len(protocol.procedures) < 3:
            protocol.procedures.extend(['Additional procedure step', 'Final validation step'])
        
        if len(protocol.safety_considerations) < 2:
            protocol.safety_considerations.extend(['General safety protocols', 'Emergency procedures'])
        
        if len(protocol.quality_controls) < 2:
            protocol.quality_controls.extend(['Quality validation', 'Result verification'])
        
        return protocol
    
    def _perform_validation_check(
        self,
        protocol: ExperimentalProtocol,
        criterion: str,
        criteria_def: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform a specific validation check."""
        result = {'passed': True, 'warning': None, 'recommendation': None}
        
        if criterion == 'completeness':
            if not protocol.objective:
                result['passed'] = False
                result['recommendation'] = 'Add clear objective statement'
            elif len(protocol.procedures) < 3:
                result['passed'] = False
                result['recommendation'] = 'Add more detailed procedures'
        
        elif criterion == 'clarity':
            unclear_procedures = [p for p in protocol.procedures if len(p.split()) < 3]
            if unclear_procedures:
                result['passed'] = False
                result['recommendation'] = 'Make procedure steps more specific'
        
        elif criterion == 'feasibility':
            if protocol.estimated_duration.days > 365:
                result['warning'] = 'Protocol duration may be too long'
                result['recommendation'] = 'Consider breaking into phases'
        
        elif criterion == 'reproducibility':
            if len(protocol.quality_controls) < 2:
                result['passed'] = False
                result['recommendation'] = 'Add more quality control measures'
        
        return result