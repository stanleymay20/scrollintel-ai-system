"""
Crisis Preparedness Enhancement Engine

System for crisis preparedness assessment, simulation, and capability development.
"""

import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
import logging
import random

from ..models.crisis_preparedness_models import (
    PreparednessAssessment, PreparednessLevel, CrisisSimulation, SimulationType,
    TrainingProgram, TrainingType, CapabilityDevelopment, CapabilityArea,
    SimulationResult, PreparednessReport
)


class CrisisPreparednessEngine:
    """Engine for crisis preparedness assessment and enhancement"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.assessment_frameworks = self._initialize_assessment_frameworks()
        self.simulation_templates = self._initialize_simulation_templates()
        self.training_catalog = self._initialize_training_catalog()
    
    def assess_crisis_preparedness(
        self, 
        organization_data: Dict[str, Any],
        assessor_id: str
    ) -> PreparednessAssessment:
        """Conduct comprehensive crisis preparedness assessment"""
        try:
            assessment_id = str(uuid.uuid4())
            
            # Assess each capability area
            capability_scores = self._assess_capabilities(organization_data)
            capability_levels = self._determine_capability_levels(capability_scores)
            
            # Calculate overall preparedness
            overall_score = sum(capability_scores.values()) / len(capability_scores)
            overall_level = self._determine_overall_preparedness_level(overall_score)
            
            # Identify strengths and weaknesses
            strengths = self._identify_preparedness_strengths(capability_scores, organization_data)
            weaknesses = self._identify_preparedness_weaknesses(capability_scores, organization_data)
            gaps = self._identify_preparedness_gaps(capability_scores, organization_data)
            
            # Risk assessment
            high_risk_scenarios = self._identify_high_risk_scenarios(organization_data)
            vulnerability_areas = self._identify_vulnerability_areas(capability_scores)
            
            # Generate recommendations
            improvement_priorities = self._prioritize_improvements(capability_scores, gaps)
            recommended_actions = self._generate_improvement_actions(weaknesses, gaps)
            
            assessment = PreparednessAssessment(
                id=assessment_id,
                assessment_date=datetime.now(),
                assessor_id=assessor_id,
                overall_preparedness_level=overall_level,
                overall_score=overall_score,
                capability_scores=capability_scores,
                capability_levels=capability_levels,
                strengths=strengths,
                weaknesses=weaknesses,
                gaps_identified=gaps,
                high_risk_scenarios=high_risk_scenarios,
                vulnerability_areas=vulnerability_areas,
                improvement_priorities=improvement_priorities,
                recommended_actions=recommended_actions,
                assessment_methodology="Comprehensive Multi-Capability Framework",
                data_sources=["organizational_data", "historical_incidents", "capability_audit"],
                confidence_level=self._calculate_assessment_confidence(organization_data)
            )
            
            self.logger.info(f"Completed preparedness assessment: {assessment_id}")
            return assessment
            
        except Exception as e:
            self.logger.error(f"Error conducting preparedness assessment: {str(e)}")
            raise
    
    def create_crisis_simulation(
        self, 
        simulation_type: SimulationType,
        participants: List[str],
        facilitator_id: str,
        custom_parameters: Optional[Dict[str, Any]] = None
    ) -> CrisisSimulation:
        """Create crisis simulation exercise"""
        try:
            simulation_id = str(uuid.uuid4())
            
            # Get simulation template
            template = self.simulation_templates.get(simulation_type.value, {})
            
            # Apply custom parameters
            if custom_parameters:
                template.update(custom_parameters)
            
            simulation = CrisisSimulation(
                id=simulation_id,
                simulation_type=simulation_type,
                title=template.get("title", f"{simulation_type.value.replace('_', ' ').title()} Simulation"),
                description=template.get("description", "Crisis response simulation exercise"),
                scenario_details=template.get("scenario_details", "Detailed scenario to be provided"),
                complexity_level=template.get("complexity_level", "Medium"),
                duration_minutes=template.get("duration_minutes", 120),
                participants=participants,
                learning_objectives=template.get("learning_objectives", []),
                success_criteria=template.get("success_criteria", []),
                facilitator_id=facilitator_id,
                scheduled_date=datetime.now() + timedelta(days=7),  # Default to next week
                actual_start_time=None,
                actual_end_time=None,
                participant_performance={},
                objectives_achieved=[],
                lessons_learned=[],
                improvement_areas=[],
                simulation_status="scheduled",
                feedback_collected=False,
                report_generated=False
            )
            
            self.logger.info(f"Created crisis simulation: {simulation_id}")
            return simulation
            
        except Exception as e:
            self.logger.error(f"Error creating crisis simulation: {str(e)}")
            raise
    
    def execute_simulation(
        self, 
        simulation: CrisisSimulation
    ) -> Dict[str, Any]:
        """Execute crisis simulation and collect results"""
        try:
            # Mark simulation as started
            simulation.actual_start_time = datetime.now()
            simulation.simulation_status = "in_progress"
            
            # Simulate execution (in real implementation, this would involve actual exercise)
            execution_results = self._simulate_execution(simulation)
            
            # Update simulation with results
            simulation.actual_end_time = datetime.now()
            simulation.participant_performance = execution_results["participant_performance"]
            simulation.objectives_achieved = execution_results["objectives_achieved"]
            simulation.lessons_learned = execution_results["lessons_learned"]
            simulation.improvement_areas = execution_results["improvement_areas"]
            simulation.simulation_status = "completed"
            
            self.logger.info(f"Executed simulation: {simulation.id}")
            return execution_results
            
        except Exception as e:
            self.logger.error(f"Error executing simulation: {str(e)}")
            raise
    
    def develop_training_program(
        self, 
        capability_area: CapabilityArea,
        target_audience: List[str],
        training_type: TrainingType
    ) -> TrainingProgram:
        """Develop crisis response training program"""
        try:
            program_id = str(uuid.uuid4())
            
            # Get training template
            template = self.training_catalog.get(f"{capability_area.value}_{training_type.value}", {})
            
            program = TrainingProgram(
                id=program_id,
                program_name=template.get("name", f"{capability_area.value.replace('_', ' ').title()} Training"),
                training_type=training_type,
                description=template.get("description", "Crisis response training program"),
                modules=template.get("modules", []),
                duration_hours=template.get("duration_hours", 8.0),
                target_audience=target_audience,
                prerequisites=template.get("prerequisites", []),
                learning_objectives=template.get("learning_objectives", []),
                competencies_developed=template.get("competencies_developed", []),
                assessment_methods=template.get("assessment_methods", []),
                delivery_method=template.get("delivery_method", "In-person"),
                instructor_requirements=template.get("instructor_requirements", []),
                materials_needed=template.get("materials_needed", []),
                created_date=datetime.now(),
                last_updated=datetime.now(),
                version="1.0",
                approval_status="draft"
            )
            
            self.logger.info(f"Developed training program: {program_id}")
            return program
            
        except Exception as e:
            self.logger.error(f"Error developing training program: {str(e)}")
            raise
    
    def create_capability_development_plan(
        self, 
        capability_area: CapabilityArea,
        current_assessment: PreparednessAssessment,
        target_level: PreparednessLevel
    ) -> CapabilityDevelopment:
        """Create capability development plan"""
        try:
            plan_id = str(uuid.uuid4())
            current_level = current_assessment.capability_levels[capability_area]
            
            # Generate development objectives
            objectives = self._generate_development_objectives(capability_area, current_level, target_level)
            
            # Generate improvement actions
            actions = self._generate_improvement_actions_for_capability(capability_area, current_level, target_level)
            
            # Determine training requirements
            training_requirements = self._determine_training_requirements(capability_area, current_level, target_level)
            
            # Estimate resource needs
            resource_needs = self._estimate_resource_needs(capability_area, actions)
            
            # Create milestones
            milestones = self._create_development_milestones(actions)
            
            plan = CapabilityDevelopment(
                id=plan_id,
                capability_area=capability_area,
                current_level=current_level,
                target_level=target_level,
                development_objectives=objectives,
                improvement_actions=actions,
                training_requirements=training_requirements,
                resource_needs=resource_needs,
                start_date=datetime.now(),
                target_completion_date=datetime.now() + timedelta(days=180),  # 6 months
                milestones=milestones,
                current_progress=0.0,
                completed_actions=[],
                pending_actions=actions,
                success_indicators=self._define_success_indicators(capability_area, target_level),
                measurement_methods=self._define_measurement_methods(capability_area),
                responsible_team=f"{capability_area.value.replace('_', ' ').title()} Team",
                budget_allocated=self._estimate_budget(actions, resource_needs),
                status="planned"
            )
            
            self.logger.info(f"Created capability development plan: {plan_id}")
            return plan
            
        except Exception as e:
            self.logger.error(f"Error creating capability development plan: {str(e)}")
            raise
    
    def generate_preparedness_report(
        self, 
        assessment: PreparednessAssessment,
        simulations: List[CrisisSimulation],
        training_programs: List[TrainingProgram]
    ) -> PreparednessReport:
        """Generate comprehensive preparedness report"""
        try:
            report_id = str(uuid.uuid4())
            
            # Generate report content
            executive_summary = self._generate_executive_summary(assessment, simulations, training_programs)
            current_state = self._generate_current_state_assessment(assessment)
            gap_analysis = self._generate_gap_analysis(assessment)
            recommendations = self._generate_improvement_recommendations_report(assessment)
            roadmap = self._generate_implementation_roadmap(assessment)
            
            # Compile supporting data
            assessment_data = self._compile_assessment_data(assessment)
            simulation_results = self._compile_simulation_results(simulations)
            training_outcomes = self._compile_training_outcomes(training_programs)
            
            report = PreparednessReport(
                id=report_id,
                report_title=f"Crisis Preparedness Assessment Report - {assessment.assessment_date.strftime('%Y-%m-%d')}",
                report_type="comprehensive",
                executive_summary=executive_summary,
                current_state_assessment=current_state,
                gap_analysis=gap_analysis,
                improvement_recommendations=recommendations,
                implementation_roadmap=roadmap,
                assessment_data=assessment_data,
                simulation_results=simulation_results,
                training_outcomes=training_outcomes,
                generated_date=datetime.now(),
                report_period="Current Assessment",
                author_id="preparedness_engine",
                review_status="draft"
            )
            
            self.logger.info(f"Generated preparedness report: {report_id}")
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating preparedness report: {str(e)}")
            raise
    
    def _assess_capabilities(self, organization_data: Dict[str, Any]) -> Dict[CapabilityArea, float]:
        """Assess each capability area"""
        scores = {}
        
        for capability in CapabilityArea:
            # Mock scoring based on organization data
            base_score = organization_data.get(f"{capability.value}_score", 70.0)
            # Add some variation based on organization maturity
            maturity_factor = organization_data.get("organizational_maturity", 0.8)
            scores[capability] = min(100.0, base_score * maturity_factor + random.uniform(-5, 5))
        
        return scores
    
    def _determine_capability_levels(self, scores: Dict[CapabilityArea, float]) -> Dict[CapabilityArea, PreparednessLevel]:
        """Determine preparedness level for each capability"""
        levels = {}
        
        for capability, score in scores.items():
            if score >= 90:
                levels[capability] = PreparednessLevel.EXCELLENT
            elif score >= 80:
                levels[capability] = PreparednessLevel.GOOD
            elif score >= 70:
                levels[capability] = PreparednessLevel.ADEQUATE
            elif score >= 60:
                levels[capability] = PreparednessLevel.POOR
            else:
                levels[capability] = PreparednessLevel.CRITICAL
        
        return levels
    
    def _determine_overall_preparedness_level(self, overall_score: float) -> PreparednessLevel:
        """Determine overall preparedness level"""
        if overall_score >= 90:
            return PreparednessLevel.EXCELLENT
        elif overall_score >= 80:
            return PreparednessLevel.GOOD
        elif overall_score >= 70:
            return PreparednessLevel.ADEQUATE
        elif overall_score >= 60:
            return PreparednessLevel.POOR
        else:
            return PreparednessLevel.CRITICAL
    
    def _identify_preparedness_strengths(self, scores: Dict[CapabilityArea, float], data: Dict[str, Any]) -> List[str]:
        """Identify preparedness strengths"""
        strengths = []
        
        for capability, score in scores.items():
            if score >= 85:
                strengths.append(f"Strong {capability.value.replace('_', ' ')} capabilities")
        
        return strengths
    
    def _identify_preparedness_weaknesses(self, scores: Dict[CapabilityArea, float], data: Dict[str, Any]) -> List[str]:
        """Identify preparedness weaknesses"""
        weaknesses = []
        
        for capability, score in scores.items():
            if score < 70:
                weaknesses.append(f"Weak {capability.value.replace('_', ' ')} capabilities")
        
        return weaknesses
    
    def _identify_preparedness_gaps(self, scores: Dict[CapabilityArea, float], data: Dict[str, Any]) -> List[str]:
        """Identify preparedness gaps"""
        gaps = []
        
        # Identify specific gaps based on low scores
        for capability, score in scores.items():
            if score < 75:
                gaps.append(f"Gap in {capability.value.replace('_', ' ')} preparedness")
        
        return gaps
    
    def _identify_high_risk_scenarios(self, data: Dict[str, Any]) -> List[str]:
        """Identify high-risk crisis scenarios"""
        # Mock high-risk scenarios based on organization profile
        industry = data.get("industry", "technology")
        
        risk_scenarios = {
            "technology": ["System outage", "Security breach", "Data loss"],
            "finance": ["Regulatory investigation", "Financial crisis", "Reputation crisis"],
            "healthcare": ["Data breach", "Regulatory compliance", "Supply chain disruption"],
            "manufacturing": ["Supply chain disruption", "Natural disaster", "Safety incident"]
        }
        
        return risk_scenarios.get(industry, ["System outage", "Security breach", "Reputation crisis"])
    
    def _identify_vulnerability_areas(self, scores: Dict[CapabilityArea, float]) -> List[str]:
        """Identify vulnerability areas"""
        vulnerabilities = []
        
        # Find lowest scoring capabilities
        sorted_capabilities = sorted(scores.items(), key=lambda x: x[1])
        
        for capability, score in sorted_capabilities[:3]:  # Top 3 vulnerabilities
            if score < 80:
                vulnerabilities.append(capability.value.replace('_', ' ').title())
        
        return vulnerabilities
    
    def _prioritize_improvements(self, scores: Dict[CapabilityArea, float], gaps: List[str]) -> List[str]:
        """Prioritize improvement areas"""
        priorities = []
        
        # Prioritize based on lowest scores and critical gaps
        sorted_capabilities = sorted(scores.items(), key=lambda x: x[1])
        
        for capability, score in sorted_capabilities:
            if score < 75:
                priorities.append(f"Improve {capability.value.replace('_', ' ')} capabilities")
        
        return priorities[:5]  # Top 5 priorities
    
    def _generate_improvement_actions(self, weaknesses: List[str], gaps: List[str]) -> List[str]:
        """Generate improvement actions"""
        actions = []
        
        for weakness in weaknesses:
            actions.append(f"Address {weakness.lower()}")
        
        for gap in gaps:
            actions.append(f"Close {gap.lower()}")
        
        return actions
    
    def _calculate_assessment_confidence(self, data: Dict[str, Any]) -> float:
        """Calculate assessment confidence level"""
        data_completeness = len(data) / 20  # Assume 20 key data points
        return min(100.0, data_completeness * 100)
    
    def _simulate_execution(self, simulation: CrisisSimulation) -> Dict[str, Any]:
        """Simulate crisis exercise execution"""
        # Mock simulation results
        participant_performance = {}
        for participant in simulation.participants:
            participant_performance[participant] = random.uniform(60, 95)
        
        objectives_achieved = simulation.learning_objectives[:random.randint(1, len(simulation.learning_objectives))]
        
        lessons_learned = [
            "Improved communication protocols needed",
            "Decision-making process can be streamlined",
            "Resource allocation requires better coordination"
        ]
        
        improvement_areas = [
            "Faster initial response time",
            "Better stakeholder communication",
            "Enhanced team coordination"
        ]
        
        return {
            "participant_performance": participant_performance,
            "objectives_achieved": objectives_achieved,
            "lessons_learned": lessons_learned,
            "improvement_areas": improvement_areas
        }
    
    def _initialize_assessment_frameworks(self) -> Dict[str, Any]:
        """Initialize assessment frameworks"""
        return {
            "comprehensive": "Multi-capability assessment framework",
            "rapid": "Quick assessment framework",
            "focused": "Targeted capability assessment"
        }
    
    def _initialize_simulation_templates(self) -> Dict[str, Any]:
        """Initialize simulation templates"""
        return {
            "system_outage": {
                "title": "System Outage Crisis Simulation",
                "description": "Major system failure requiring immediate response",
                "scenario_details": "Critical production systems have failed, affecting customer operations",
                "complexity_level": "High",
                "duration_minutes": 180,
                "learning_objectives": ["Rapid response", "Communication", "Recovery planning"],
                "success_criteria": ["Response time < 30 minutes", "All stakeholders notified", "Recovery plan executed"]
            },
            "security_breach": {
                "title": "Security Breach Response Simulation",
                "description": "Data security incident requiring coordinated response",
                "scenario_details": "Potential data breach detected, requiring investigation and response",
                "complexity_level": "High",
                "duration_minutes": 240,
                "learning_objectives": ["Incident response", "Legal compliance", "Customer communication"],
                "success_criteria": ["Breach contained", "Authorities notified", "Customers informed"]
            }
        }
    
    def _initialize_training_catalog(self) -> Dict[str, Any]:
        """Initialize training program catalog"""
        return {
            "crisis_detection_tabletop_exercise": {
                "name": "Crisis Detection Workshop",
                "description": "Training on early crisis detection and assessment",
                "modules": ["Detection methods", "Assessment techniques", "Escalation procedures"],
                "duration_hours": 4.0,
                "learning_objectives": ["Identify crisis indicators", "Assess crisis severity", "Trigger appropriate response"]
            },
            "communication_simulation_drill": {
                "name": "Crisis Communication Simulation",
                "description": "Hands-on crisis communication training",
                "modules": ["Message development", "Stakeholder management", "Media relations"],
                "duration_hours": 6.0,
                "learning_objectives": ["Develop clear messages", "Manage stakeholder communications", "Handle media inquiries"]
            }
        }
    
    # Additional helper methods for capability development
    def _generate_development_objectives(self, capability: CapabilityArea, current: PreparednessLevel, target: PreparednessLevel) -> List[str]:
        """Generate development objectives"""
        return [f"Improve {capability.value.replace('_', ' ')} from {current.value} to {target.value}"]
    
    def _generate_improvement_actions_for_capability(self, capability: CapabilityArea, current: PreparednessLevel, target: PreparednessLevel) -> List[str]:
        """Generate improvement actions for specific capability"""
        return [f"Enhance {capability.value.replace('_', ' ')} processes and procedures"]
    
    def _determine_training_requirements(self, capability: CapabilityArea, current: PreparednessLevel, target: PreparednessLevel) -> List[str]:
        """Determine training requirements"""
        return [f"{capability.value.replace('_', ' ').title()} training program"]
    
    def _estimate_resource_needs(self, capability: CapabilityArea, actions: List[str]) -> List[str]:
        """Estimate resource needs"""
        return ["Training materials", "Subject matter experts", "Simulation tools"]
    
    def _create_development_milestones(self, actions: List[str]) -> List[Dict[str, Any]]:
        """Create development milestones"""
        milestones = []
        for i, action in enumerate(actions):
            milestones.append({
                "milestone": f"Complete {action}",
                "target_date": (datetime.now() + timedelta(days=30 * (i + 1))).isoformat(),
                "status": "pending"
            })
        return milestones
    
    def _define_success_indicators(self, capability: CapabilityArea, target: PreparednessLevel) -> List[str]:
        """Define success indicators"""
        return [f"Achieve {target.value} level in {capability.value.replace('_', ' ')}"]
    
    def _define_measurement_methods(self, capability: CapabilityArea) -> List[str]:
        """Define measurement methods"""
        return ["Assessment scores", "Simulation performance", "Peer evaluation"]
    
    def _estimate_budget(self, actions: List[str], resources: List[str]) -> float:
        """Estimate budget requirements"""
        return len(actions) * 5000 + len(resources) * 2000  # Mock calculation
    
    # Report generation helper methods
    def _generate_executive_summary(self, assessment: PreparednessAssessment, simulations: List[CrisisSimulation], training: List[TrainingProgram]) -> str:
        """Generate executive summary"""
        return f"Overall preparedness level: {assessment.overall_preparedness_level.value.title()} ({assessment.overall_score:.1f}%)"
    
    def _generate_current_state_assessment(self, assessment: PreparednessAssessment) -> str:
        """Generate current state assessment"""
        return f"Current preparedness assessment shows {len(assessment.strengths)} strengths and {len(assessment.weaknesses)} areas for improvement"
    
    def _generate_gap_analysis(self, assessment: PreparednessAssessment) -> str:
        """Generate gap analysis"""
        return f"Identified {len(assessment.gaps_identified)} critical gaps requiring immediate attention"
    
    def _generate_improvement_recommendations_report(self, assessment: PreparednessAssessment) -> str:
        """Generate improvement recommendations for report"""
        return f"Recommended {len(assessment.recommended_actions)} improvement actions across {len(assessment.improvement_priorities)} priority areas"
    
    def _generate_implementation_roadmap(self, assessment: PreparednessAssessment) -> str:
        """Generate implementation roadmap"""
        return "Phased implementation approach over 6-12 months with quarterly milestones"
    
    def _compile_assessment_data(self, assessment: PreparednessAssessment) -> Dict[str, Any]:
        """Compile assessment data"""
        return {
            "overall_score": assessment.overall_score,
            "capability_scores": {k.value: v for k, v in assessment.capability_scores.items()},
            "strengths_count": len(assessment.strengths),
            "weaknesses_count": len(assessment.weaknesses)
        }
    
    def _compile_simulation_results(self, simulations: List[CrisisSimulation]) -> List[Dict[str, Any]]:
        """Compile simulation results"""
        return [
            {
                "simulation_id": sim.id,
                "type": sim.simulation_type.value,
                "participants": len(sim.participants),
                "status": sim.simulation_status
            }
            for sim in simulations
        ]
    
    def _compile_training_outcomes(self, training: List[TrainingProgram]) -> List[Dict[str, Any]]:
        """Compile training outcomes"""
        return [
            {
                "program_id": prog.id,
                "name": prog.program_name,
                "type": prog.training_type.value,
                "duration": prog.duration_hours
            }
            for prog in training
        ]