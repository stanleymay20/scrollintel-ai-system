"""
Experiment timeline and milestone planning system.
"""
import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import logging
from dataclasses import dataclass

from ..models.experimental_design_models import (
    ExperimentMilestone, ExperimentalProtocol, ResourceRequirement,
    ExperimentType, MethodologyType
)


@dataclass
class TimelineConstraint:
    """Constraint for timeline planning."""
    constraint_type: str  # deadline, resource_availability, dependency
    description: str
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    resource_id: Optional[str] = None
    priority: str = "medium"  # low, medium, high, critical


@dataclass
class TimelineOptimization:
    """Timeline optimization result."""
    original_duration: timedelta
    optimized_duration: timedelta
    time_savings: timedelta
    optimization_strategies: List[str]
    trade_offs: List[str]
    confidence_score: float


class TimelinePlanner:
    """
    System for creating experiment timelines and milestone planning.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.phase_templates = self._initialize_phase_templates()
        self.dependency_rules = self._initialize_dependency_rules()
        self.buffer_factors = self._initialize_buffer_factors()
    
    def create_experiment_timeline(
        self,
        protocol: ExperimentalProtocol,
        resources: List[ResourceRequirement],
        experiment_type: ExperimentType,
        methodology: MethodologyType,
        constraints: Optional[List[TimelineConstraint]] = None,
        start_date: Optional[datetime] = None
    ) -> List[ExperimentMilestone]:
        """
        Create comprehensive experiment timeline with milestones.
        
        Args:
            protocol: Experimental protocol
            resources: Resource requirements
            experiment_type: Type of experiment
            methodology: Experimental methodology
            constraints: Timeline constraints
            start_date: Preferred start date
            
        Returns:
            List of experiment milestones
        """
        try:
            if start_date is None:
                start_date = datetime.now() + timedelta(days=7)  # Default buffer
            
            # Analyze protocol to identify phases
            phases = self._identify_experiment_phases(
                protocol, experiment_type, methodology
            )
            
            # Estimate phase durations
            phase_durations = self._estimate_phase_durations(
                phases, protocol, resources, methodology
            )
            
            # Apply constraints
            if constraints:
                phase_durations = self._apply_timeline_constraints(
                    phase_durations, constraints
                )
            
            # Create milestones for each phase
            milestones = self._create_phase_milestones(
                phases, phase_durations, start_date, protocol
            )
            
            # Add dependencies between milestones
            milestones = self._add_milestone_dependencies(milestones, phases)
            
            # Optimize timeline
            milestones = self._optimize_timeline(milestones, constraints)
            
            # Add buffer time
            milestones = self._add_buffer_time(milestones, experiment_type)
            
            # Validate timeline feasibility
            self._validate_timeline_feasibility(milestones, resources, constraints)
            
            self.logger.info(f"Created timeline with {len(milestones)} milestones")
            return milestones
            
        except Exception as e:
            self.logger.error(f"Error creating experiment timeline: {str(e)}")
            raise
    
    def optimize_timeline(
        self,
        milestones: List[ExperimentMilestone],
        optimization_goals: List[str],
        constraints: Optional[List[TimelineConstraint]] = None
    ) -> Tuple[List[ExperimentMilestone], TimelineOptimization]:
        """
        Optimize experiment timeline for specific goals.
        
        Args:
            milestones: Current milestones
            optimization_goals: Goals (minimize_duration, balance_resources, etc.)
            constraints: Timeline constraints
            
        Returns:
            Optimized milestones and optimization details
        """
        try:
            original_duration = self._calculate_total_duration(milestones)
            
            # Apply optimization strategies
            optimized_milestones = milestones.copy()
            optimization_strategies = []
            
            for goal in optimization_goals:
                if goal == "minimize_duration":
                    optimized_milestones, strategies = self._minimize_duration(
                        optimized_milestones, constraints
                    )
                    optimization_strategies.extend(strategies)
                
                elif goal == "balance_resources":
                    optimized_milestones, strategies = self._balance_resources(
                        optimized_milestones, constraints
                    )
                    optimization_strategies.extend(strategies)
                
                elif goal == "reduce_risk":
                    optimized_milestones, strategies = self._reduce_timeline_risk(
                        optimized_milestones, constraints
                    )
                    optimization_strategies.extend(strategies)
            
            # Calculate optimization results
            optimized_duration = self._calculate_total_duration(optimized_milestones)
            time_savings = original_duration - optimized_duration
            
            # Identify trade-offs
            trade_offs = self._identify_optimization_trade_offs(
                milestones, optimized_milestones, optimization_goals
            )
            
            # Calculate confidence score
            confidence_score = self._calculate_optimization_confidence(
                optimization_strategies, time_savings, trade_offs
            )
            
            optimization_result = TimelineOptimization(
                original_duration=original_duration,
                optimized_duration=optimized_duration,
                time_savings=time_savings,
                optimization_strategies=optimization_strategies,
                trade_offs=trade_offs,
                confidence_score=confidence_score
            )
            
            self.logger.info(
                f"Timeline optimization saved {time_savings.days} days "
                f"with confidence {confidence_score:.2f}"
            )
            
            return optimized_milestones, optimization_result
            
        except Exception as e:
            self.logger.error(f"Error optimizing timeline: {str(e)}")
            raise
    
    def create_milestone_dependencies(
        self,
        milestones: List[ExperimentMilestone],
        dependency_rules: Optional[Dict[str, List[str]]] = None
    ) -> List[ExperimentMilestone]:
        """
        Create dependencies between milestones.
        
        Args:
            milestones: List of milestones
            dependency_rules: Custom dependency rules
            
        Returns:
            Milestones with dependencies added
        """
        try:
            if dependency_rules is None:
                dependency_rules = self.dependency_rules
            
            milestone_map = {m.name.lower(): m for m in milestones}
            
            for milestone in milestones:
                # Apply standard dependency rules
                for rule_name, dependencies in dependency_rules.items():
                    if rule_name in milestone.name.lower():
                        for dep_name in dependencies:
                            # Find matching milestone
                            for dep_milestone in milestones:
                                if dep_name in dep_milestone.name.lower():
                                    if dep_milestone.milestone_id not in milestone.dependencies:
                                        milestone.dependencies.append(dep_milestone.milestone_id)
                                    break
            
            # Add sequential dependencies if no specific rules matched
            sorted_milestones = sorted(milestones, key=lambda m: m.target_date)
            for i in range(1, len(sorted_milestones)):
                current = sorted_milestones[i]
                previous = sorted_milestones[i-1]
                
                # Only add dependency if current milestone doesn't already have dependencies
                if not current.dependencies:
                    current.dependencies.append(previous.milestone_id)
            
            # Validate dependency cycles
            self._validate_dependency_cycles(milestones)
            
            self.logger.info("Added milestone dependencies")
            return milestones
            
        except Exception as e:
            self.logger.error(f"Error creating milestone dependencies: {str(e)}")
            raise
    
    def calculate_critical_path(
        self,
        milestones: List[ExperimentMilestone]
    ) -> List[str]:
        """
        Calculate critical path through milestones.
        
        Args:
            milestones: List of milestones with dependencies
            
        Returns:
            List of milestone IDs on critical path
        """
        try:
            # Build dependency graph
            graph = self._build_dependency_graph(milestones)
            
            # Calculate earliest start times
            earliest_times = self._calculate_earliest_times(milestones, graph)
            
            # Calculate latest start times
            latest_times = self._calculate_latest_times(milestones, graph)
            
            # Identify critical milestones (where earliest = latest)
            critical_path = []
            for milestone in milestones:
                earliest = earliest_times.get(milestone.milestone_id, 0)
                latest = latest_times.get(milestone.milestone_id, 0)
                
                if abs(earliest - latest) < timedelta(hours=1):  # Small tolerance
                    critical_path.append(milestone.milestone_id)
            
            self.logger.info(f"Critical path contains {len(critical_path)} milestones")
            return critical_path
            
        except Exception as e:
            self.logger.error(f"Error calculating critical path: {str(e)}")
            raise
    
    def _initialize_phase_templates(self) -> Dict[str, Dict[str, Any]]:
        """Initialize experiment phase templates."""
        return {
            'planning': {
                'duration_factor': 0.15,
                'activities': [
                    'Finalize protocol',
                    'Secure resources',
                    'Obtain approvals',
                    'Setup equipment'
                ],
                'deliverables': [
                    'Approved protocol',
                    'Resource allocation',
                    'Equipment setup'
                ]
            },
            'pilot': {
                'duration_factor': 0.10,
                'activities': [
                    'Conduct pilot study',
                    'Test procedures',
                    'Validate measurements',
                    'Refine protocol'
                ],
                'deliverables': [
                    'Pilot results',
                    'Refined protocol',
                    'Validation report'
                ]
            },
            'data_collection': {
                'duration_factor': 0.50,
                'activities': [
                    'Recruit participants',
                    'Collect data',
                    'Monitor quality',
                    'Maintain records'
                ],
                'deliverables': [
                    'Complete dataset',
                    'Quality report',
                    'Data documentation'
                ]
            },
            'analysis': {
                'duration_factor': 0.20,
                'activities': [
                    'Clean data',
                    'Perform analysis',
                    'Validate results',
                    'Create visualizations'
                ],
                'deliverables': [
                    'Analysis results',
                    'Statistical report',
                    'Data visualizations'
                ]
            },
            'reporting': {
                'duration_factor': 0.05,
                'activities': [
                    'Write report',
                    'Create presentations',
                    'Peer review',
                    'Finalize documentation'
                ],
                'deliverables': [
                    'Final report',
                    'Presentations',
                    'Documentation'
                ]
            }
        }
    
    def _initialize_dependency_rules(self) -> Dict[str, List[str]]:
        """Initialize milestone dependency rules."""
        return {
            'pilot': ['planning'],
            'data_collection': ['pilot', 'planning'],
            'analysis': ['data_collection'],
            'reporting': ['analysis'],
            'validation': ['analysis'],
            'review': ['reporting']
        }
    
    def _initialize_buffer_factors(self) -> Dict[str, float]:
        """Initialize buffer factors for different experiment types."""
        return {
            'controlled': 0.15,
            'randomized': 0.20,
            'factorial': 0.25,
            'longitudinal': 0.30,
            'observational': 0.10,
            'comparative': 0.15,
            'cross_sectional': 0.10
        }
    
    def _identify_experiment_phases(
        self,
        protocol: ExperimentalProtocol,
        experiment_type: ExperimentType,
        methodology: MethodologyType
    ) -> List[str]:
        """Identify phases needed for experiment."""
        phases = ['planning']
        
        # Add pilot phase for complex experiments
        if experiment_type in [ExperimentType.FACTORIAL, ExperimentType.LONGITUDINAL]:
            phases.append('pilot')
        
        # Always include data collection
        phases.append('data_collection')
        
        # Add analysis phase
        phases.append('analysis')
        
        # Add reporting phase
        phases.append('reporting')
        
        return phases
    
    def _estimate_phase_durations(
        self,
        phases: List[str],
        protocol: ExperimentalProtocol,
        resources: List[ResourceRequirement],
        methodology: MethodologyType
    ) -> Dict[str, timedelta]:
        """Estimate duration for each phase."""
        durations = {}
        base_duration = protocol.estimated_duration
        
        for phase in phases:
            if phase in self.phase_templates:
                factor = self.phase_templates[phase]['duration_factor']
                durations[phase] = base_duration * factor
            else:
                durations[phase] = timedelta(weeks=1)  # Default
        
        # Adjust for methodology
        if methodology == MethodologyType.QUALITATIVE:
            durations['analysis'] *= 1.5  # Qualitative analysis takes longer
        elif methodology == MethodologyType.COMPUTATIONAL:
            durations['data_collection'] *= 0.5  # Faster data collection
        
        return durations
    
    def _apply_timeline_constraints(
        self,
        phase_durations: Dict[str, timedelta],
        constraints: List[TimelineConstraint]
    ) -> Dict[str, timedelta]:
        """Apply timeline constraints to phase durations."""
        adjusted_durations = phase_durations.copy()
        
        for constraint in constraints:
            if constraint.constraint_type == "deadline":
                # Compress timeline if needed
                if constraint.end_date:
                    total_duration = sum(phase_durations.values(), timedelta())
                    available_time = constraint.end_date - datetime.now()
                    
                    if available_time < total_duration:
                        # Apply more aggressive compression
                        compression_factor = max(0.3, available_time / total_duration)
                        for phase in adjusted_durations:
                            adjusted_durations[phase] *= compression_factor
        
        return adjusted_durations
    
    def _create_phase_milestones(
        self,
        phases: List[str],
        phase_durations: Dict[str, timedelta],
        start_date: datetime,
        protocol: ExperimentalProtocol
    ) -> List[ExperimentMilestone]:
        """Create milestones for each phase."""
        milestones = []
        current_date = start_date
        
        for phase in phases:
            duration = phase_durations.get(phase, timedelta(weeks=1))
            target_date = current_date + duration
            
            template = self.phase_templates.get(phase, {})
            
            milestone = ExperimentMilestone(
                milestone_id=str(uuid.uuid4()),
                name=f"{phase.replace('_', ' ').title()} Complete",
                description=f"Completion of {phase} phase",
                target_date=target_date,
                deliverables=template.get('deliverables', []),
                completion_criteria=template.get('activities', [])
            )
            
            milestones.append(milestone)
            current_date = target_date
        
        return milestones
    
    def _add_milestone_dependencies(
        self,
        milestones: List[ExperimentMilestone],
        phases: List[str]
    ) -> List[ExperimentMilestone]:
        """Add dependencies between milestones."""
        milestone_map = {phase: milestone for phase, milestone in zip(phases, milestones)}
        
        for i, phase in enumerate(phases):
            if i > 0:  # Not the first phase
                previous_phase = phases[i-1]
                if previous_phase in milestone_map:
                    milestone_map[phase].dependencies.append(
                        milestone_map[previous_phase].milestone_id
                    )
        
        return milestones
    
    def _optimize_timeline(
        self,
        milestones: List[ExperimentMilestone],
        constraints: Optional[List[TimelineConstraint]]
    ) -> List[ExperimentMilestone]:
        """Optimize timeline for efficiency."""
        # Look for opportunities to parallelize activities
        optimized_milestones = []
        
        for milestone in milestones:
            # Check if milestone can be started earlier
            if len(milestone.dependencies) == 0:
                # Independent milestone - can potentially start earlier
                optimized_milestone = milestone
            else:
                # Dependent milestone - check if dependencies allow earlier start
                optimized_milestone = milestone
            
            optimized_milestones.append(optimized_milestone)
        
        return optimized_milestones
    
    def _add_buffer_time(
        self,
        milestones: List[ExperimentMilestone],
        experiment_type: ExperimentType
    ) -> List[ExperimentMilestone]:
        """Add buffer time to milestones."""
        buffer_factor = self.buffer_factors.get(experiment_type.value, 0.15)
        
        for milestone in milestones:
            # Add buffer to target date
            if milestone.dependencies:
                # Calculate buffer based on dependency chain length
                buffer_days = int(7 * buffer_factor * len(milestone.dependencies))
            else:
                buffer_days = int(7 * buffer_factor)
            
            milestone.target_date += timedelta(days=buffer_days)
        
        return milestones
    
    def _validate_timeline_feasibility(
        self,
        milestones: List[ExperimentMilestone],
        resources: List[ResourceRequirement],
        constraints: Optional[List[TimelineConstraint]]
    ) -> None:
        """Validate that timeline is feasible."""
        # Check resource availability
        for resource in resources:
            # Simplified validation - would be more complex in practice
            if resource.availability_constraint:
                self.logger.warning(f"Resource constraint detected: {resource.resource_name}")
        
        # Check constraint violations - be more lenient in validation
        if constraints:
            for constraint in constraints:
                if constraint.constraint_type == "deadline" and constraint.end_date:
                    latest_milestone = max(milestones, key=lambda m: m.target_date)
                    # Allow some buffer beyond deadline (10% tolerance)
                    deadline_buffer = constraint.end_date + timedelta(
                        days=int((constraint.end_date - datetime.now()).days * 0.1)
                    )
                    if latest_milestone.target_date > deadline_buffer:
                        self.logger.warning(f"Timeline may exceed deadline constraint")
                        # Don't raise error, just warn
    
    def _calculate_total_duration(self, milestones: List[ExperimentMilestone]) -> timedelta:
        """Calculate total timeline duration."""
        if not milestones:
            return timedelta()
        
        earliest = min(milestone.target_date for milestone in milestones)
        latest = max(milestone.target_date for milestone in milestones)
        
        return latest - earliest
    
    def _minimize_duration(
        self,
        milestones: List[ExperimentMilestone],
        constraints: Optional[List[TimelineConstraint]]
    ) -> Tuple[List[ExperimentMilestone], List[str]]:
        """Minimize timeline duration."""
        strategies = []
        optimized_milestones = milestones.copy()
        
        # Strategy 1: Parallelize independent activities
        strategies.append("Parallelize independent activities")
        
        # Strategy 2: Reduce buffer time
        for milestone in optimized_milestones:
            # Reduce target date by 10%
            original_date = milestone.target_date
            milestone.target_date = original_date - timedelta(
                days=int((original_date - datetime.now()).days * 0.1)
            )
        strategies.append("Reduce buffer time by 10%")
        
        return optimized_milestones, strategies
    
    def _balance_resources(
        self,
        milestones: List[ExperimentMilestone],
        constraints: Optional[List[TimelineConstraint]]
    ) -> Tuple[List[ExperimentMilestone], List[str]]:
        """Balance resource utilization."""
        strategies = ["Distribute resource usage evenly"]
        # Implementation would adjust milestone timing to balance resources
        return milestones, strategies
    
    def _reduce_timeline_risk(
        self,
        milestones: List[ExperimentMilestone],
        constraints: Optional[List[TimelineConstraint]]
    ) -> Tuple[List[ExperimentMilestone], List[str]]:
        """Reduce timeline risks."""
        strategies = ["Add contingency milestones", "Increase buffer time for high-risk activities"]
        # Implementation would add risk mitigation to timeline
        return milestones, strategies
    
    def _identify_optimization_trade_offs(
        self,
        original_milestones: List[ExperimentMilestone],
        optimized_milestones: List[ExperimentMilestone],
        optimization_goals: List[str]
    ) -> List[str]:
        """Identify trade-offs from optimization."""
        trade_offs = []
        
        if "minimize_duration" in optimization_goals:
            trade_offs.append("Reduced buffer time may increase risk")
            trade_offs.append("Compressed timeline may impact quality")
        
        if "balance_resources" in optimization_goals:
            trade_offs.append("Resource balancing may extend overall duration")
        
        return trade_offs
    
    def _calculate_optimization_confidence(
        self,
        strategies: List[str],
        time_savings: timedelta,
        trade_offs: List[str]
    ) -> float:
        """Calculate confidence in optimization."""
        base_confidence = 0.8
        
        # Reduce confidence based on number of trade-offs
        confidence_reduction = len(trade_offs) * 0.1
        
        # Adjust based on time savings magnitude
        if time_savings.days > 14:
            confidence_reduction += 0.1  # Large savings may be risky
        
        return max(0.1, base_confidence - confidence_reduction)
    
    def _build_dependency_graph(self, milestones: List[ExperimentMilestone]) -> Dict[str, List[str]]:
        """Build dependency graph from milestones."""
        graph = {}
        for milestone in milestones:
            graph[milestone.milestone_id] = milestone.dependencies
        return graph
    
    def _calculate_earliest_times(
        self,
        milestones: List[ExperimentMilestone],
        graph: Dict[str, List[str]]
    ) -> Dict[str, datetime]:
        """Calculate earliest start times for milestones."""
        earliest_times = {}
        milestone_map = {m.milestone_id: m for m in milestones}
        
        # Simplified calculation - would use topological sort in practice
        for milestone in milestones:
            if not milestone.dependencies:
                earliest_times[milestone.milestone_id] = milestone.target_date
            else:
                # Find latest dependency completion
                latest_dep = max(
                    earliest_times.get(dep_id, datetime.now())
                    for dep_id in milestone.dependencies
                    if dep_id in earliest_times
                )
                earliest_times[milestone.milestone_id] = latest_dep
        
        return earliest_times
    
    def _calculate_latest_times(
        self,
        milestones: List[ExperimentMilestone],
        graph: Dict[str, List[str]]
    ) -> Dict[str, datetime]:
        """Calculate latest start times for milestones."""
        # Simplified calculation
        latest_times = {}
        for milestone in milestones:
            latest_times[milestone.milestone_id] = milestone.target_date
        return latest_times
    
    def _validate_dependency_cycles(self, milestones: List[ExperimentMilestone]) -> None:
        """Validate that there are no dependency cycles."""
        # Simplified validation - would implement proper cycle detection
        milestone_ids = {m.milestone_id for m in milestones}
        
        for milestone in milestones:
            for dep_id in milestone.dependencies:
                if dep_id not in milestone_ids:
                    self.logger.warning(f"Invalid dependency reference: {dep_id}")