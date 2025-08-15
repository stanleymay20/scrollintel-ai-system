"""
Change Champion Development Engine

Comprehensive system for identifying, developing, and managing change champions
within organizational transformation initiatives.
"""

import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import logging
from dataclasses import asdict
import random

from ..models.change_champion_models import (
    ChangeChampionProfile, ChampionLevel, ChampionRole, ChangeCapability,
    ChampionIdentificationCriteria, ChampionDevelopmentProgram, LearningModule,
    PracticalAssignment, ChampionTrainingSession, ChampionNetwork, NetworkStatus,
    NetworkActivity, ChampionMentorship, ChampionPerformanceMetrics,
    ChampionRecognition, NetworkCoordinationPlan
)

logger = logging.getLogger(__name__)


class ChangeChampionDevelopmentEngine:
    """Engine for change champion identification, development, and network management"""
    
    def __init__(self):
        self.identification_criteria = self._initialize_identification_criteria()
        self.development_programs = self._initialize_development_programs()
        self.capability_weights = self._initialize_capability_weights()
        self.training_resources = self._initialize_training_resources()
        self.network_templates = self._initialize_network_templates()
        
    def _initialize_identification_criteria(self) -> Dict[str, ChampionIdentificationCriteria]:
        """Initialize criteria for identifying change champions"""
        return {
            "standard": ChampionIdentificationCriteria(
                id="standard_criteria",
                name="Standard Change Champion Criteria",
                description="General criteria for identifying change champions across all levels",
                required_capabilities=[
                    ChangeCapability.CHANGE_ADVOCACY,
                    ChangeCapability.INFLUENCE_BUILDING,
                    ChangeCapability.COMMUNICATION
                ],
                minimum_scores={
                    ChangeCapability.CHANGE_ADVOCACY: 60,
                    ChangeCapability.INFLUENCE_BUILDING: 55,
                    ChangeCapability.COMMUNICATION: 65,
                    ChangeCapability.CULTURAL_SENSITIVITY: 50
                },
                influence_requirements={
                    "network_size": 15,
                    "cross_department_connections": 3,
                    "credibility_score": 70
                },
                experience_requirements=[
                    "Participated in at least one change initiative",
                    "Demonstrated leadership in team projects",
                    "Positive peer feedback on collaboration"
                ],
                role_preferences=[
                    "Team Lead", "Project Manager", "Senior Specialist",
                    "Department Coordinator", "Training Specialist"
                ],
                department_coverage=[
                    "Operations", "HR", "IT", "Sales", "Marketing", "Finance"
                ],
                cultural_factors=[
                    "Embraces organizational values",
                    "Demonstrates cultural alignment",
                    "Shows respect for diversity"
                ],
                exclusion_criteria=[
                    "Recent performance issues",
                    "Resistance to previous changes",
                    "Limited availability for additional responsibilities"
                ],
                weight_factors={
                    "capability_scores": 0.4,
                    "influence_network": 0.25,
                    "experience": 0.2,
                    "cultural_fit": 0.15
                }
            ),
            "senior": ChampionIdentificationCriteria(
                id="senior_criteria",
                name="Senior Change Champion Criteria",
                description="Criteria for identifying senior-level change champions",
                required_capabilities=[
                    ChangeCapability.CHANGE_ADVOCACY,
                    ChangeCapability.INFLUENCE_BUILDING,
                    ChangeCapability.COMMUNICATION,
                    ChangeCapability.COACHING_MENTORING,
                    ChangeCapability.PROJECT_COORDINATION
                ],
                minimum_scores={
                    ChangeCapability.CHANGE_ADVOCACY: 75,
                    ChangeCapability.INFLUENCE_BUILDING: 70,
                    ChangeCapability.COMMUNICATION: 80,
                    ChangeCapability.COACHING_MENTORING: 65,
                    ChangeCapability.PROJECT_COORDINATION: 70
                },
                influence_requirements={
                    "network_size": 30,
                    "cross_department_connections": 5,
                    "credibility_score": 80
                },
                experience_requirements=[
                    "Led multiple change initiatives",
                    "Mentored other employees",
                    "Demonstrated strategic thinking"
                ],
                role_preferences=[
                    "Manager", "Director", "Senior Manager",
                    "Principal", "Lead Architect"
                ],
                department_coverage=["All departments"],
                cultural_factors=[
                    "Champions organizational values",
                    "Drives cultural transformation",
                    "Models inclusive behavior"
                ],
                exclusion_criteria=[
                    "Overcommitted with current responsibilities",
                    "Lack of senior leadership support"
                ],
                weight_factors={
                    "capability_scores": 0.35,
                    "influence_network": 0.3,
                    "experience": 0.25,
                    "cultural_fit": 0.1
                }
            )
        }
    
    def _initialize_development_programs(self) -> Dict[str, ChampionDevelopmentProgram]:
        """Initialize change champion development programs"""
        return {
            "foundation": ChampionDevelopmentProgram(
                id="foundation_program",
                name="Change Champion Foundation Program",
                description="Basic program for new change champions",
                target_level=ChampionLevel.DEVELOPING,
                target_roles=[ChampionRole.ADVOCATE, ChampionRole.FACILITATOR],
                duration_weeks=8,
                learning_modules=[
                    LearningModule(
                        id="mod_change_basics",
                        title="Change Management Fundamentals",
                        description="Introduction to change management principles and practices",
                        target_capabilities=[ChangeCapability.CHANGE_ADVOCACY],
                        learning_objectives=[
                            "Understand change management theory",
                            "Identify change resistance patterns",
                            "Apply basic change facilitation techniques"
                        ],
                        content_type="workshop",
                        duration_hours=16,
                        delivery_method="in_person",
                        materials=["Change Management Handbook", "Case Studies"],
                        assessments=["Knowledge Quiz", "Role Play Exercise"],
                        completion_criteria=["80% attendance", "Pass assessment"]
                    ),
                    LearningModule(
                        id="mod_communication",
                        title="Effective Change Communication",
                        description="Communication strategies for change initiatives",
                        target_capabilities=[ChangeCapability.COMMUNICATION],
                        learning_objectives=[
                            "Craft compelling change messages",
                            "Adapt communication to different audiences",
                            "Handle difficult conversations"
                        ],
                        content_type="workshop",
                        duration_hours=12,
                        delivery_method="virtual",
                        materials=["Communication Toolkit", "Message Templates"],
                        assessments=["Presentation Exercise", "Peer Feedback"],
                        completion_criteria=["Complete all exercises", "Positive peer feedback"]
                    )
                ],
                practical_assignments=[
                    PracticalAssignment(
                        id="assign_change_project",
                        title="Lead Small Change Initiative",
                        description="Lead a small-scale change project in your department",
                        target_capabilities=[
                            ChangeCapability.CHANGE_ADVOCACY,
                            ChangeCapability.PROJECT_COORDINATION
                        ],
                        assignment_type="project",
                        duration_weeks=4,
                        deliverables=[
                            "Change project plan",
                            "Stakeholder communication plan",
                            "Progress reports",
                            "Final presentation"
                        ],
                        success_metrics=[
                            "Project completion on time",
                            "Stakeholder satisfaction > 80%",
                            "Measurable change outcomes"
                        ],
                        support_provided=["Mentor guidance", "Resource access"],
                        evaluation_criteria=[
                            "Project planning quality",
                            "Communication effectiveness",
                            "Results achieved"
                        ],
                        mentor_involvement=True
                    )
                ],
                mentorship_component=True,
                peer_learning_groups=True,
                certification_available=True,
                success_criteria=[
                    "Complete all learning modules",
                    "Successfully complete practical assignment",
                    "Demonstrate improved capability scores",
                    "Receive positive mentor feedback"
                ],
                prerequisites=["Manager nomination", "Commitment agreement"],
                resources_required=["Training budget", "Mentor assignment", "Time allocation"]
            ),
            "advanced": ChampionDevelopmentProgram(
                id="advanced_program",
                name="Advanced Change Leadership Program",
                description="Advanced program for senior change champions",
                target_level=ChampionLevel.SENIOR,
                target_roles=[ChampionRole.TRAINER, ChampionRole.MENTOR, ChampionRole.STRATEGIST],
                duration_weeks=12,
                learning_modules=[
                    LearningModule(
                        id="mod_strategic_change",
                        title="Strategic Change Leadership",
                        description="Leading large-scale organizational transformation",
                        target_capabilities=[
                            ChangeCapability.CHANGE_ADVOCACY,
                            ChangeCapability.PROJECT_COORDINATION
                        ],
                        learning_objectives=[
                            "Design transformation strategies",
                            "Manage complex stakeholder networks",
                            "Navigate organizational politics"
                        ],
                        content_type="simulation",
                        duration_hours=24,
                        delivery_method="blended",
                        materials=["Strategy Frameworks", "Case Studies", "Simulation Tools"],
                        assessments=["Strategy Presentation", "Simulation Performance"],
                        completion_criteria=["Pass simulation", "Peer evaluation > 85%"]
                    ),
                    LearningModule(
                        id="mod_coaching_mentoring",
                        title="Coaching and Mentoring Skills",
                        description="Developing others as change champions",
                        target_capabilities=[ChangeCapability.COACHING_MENTORING],
                        learning_objectives=[
                            "Master coaching techniques",
                            "Design mentoring programs",
                            "Develop change capability in others"
                        ],
                        content_type="workshop",
                        duration_hours=20,
                        delivery_method="in_person",
                        materials=["Coaching Toolkit", "Mentoring Guide"],
                        assessments=["Coaching Practice", "Mentoring Plan"],
                        completion_criteria=["Demonstrate coaching skills", "Create mentoring plan"]
                    )
                ],
                practical_assignments=[
                    PracticalAssignment(
                        id="assign_mentor_champions",
                        title="Mentor New Change Champions",
                        description="Mentor 2-3 new change champions through foundation program",
                        target_capabilities=[
                            ChangeCapability.COACHING_MENTORING,
                            ChangeCapability.TRAINING_DELIVERY
                        ],
                        assignment_type="mentoring",
                        duration_weeks=8,
                        deliverables=[
                            "Mentoring plans for each mentee",
                            "Regular progress reports",
                            "Mentee development assessments",
                            "Mentoring reflection report"
                        ],
                        success_metrics=[
                            "Mentee program completion rate > 90%",
                            "Mentee satisfaction > 85%",
                            "Demonstrated mentee capability improvement"
                        ],
                        support_provided=["Mentoring resources", "Program coordinator support"],
                        evaluation_criteria=[
                            "Mentoring plan quality",
                            "Mentee development outcomes",
                            "Mentoring relationship effectiveness"
                        ],
                        mentor_involvement=False
                    )
                ],
                mentorship_component=False,  # They become mentors
                peer_learning_groups=True,
                certification_available=True,
                success_criteria=[
                    "Complete all learning modules",
                    "Successfully mentor new champions",
                    "Lead strategic change initiative",
                    "Achieve senior champion certification"
                ],
                prerequisites=[
                    "Complete foundation program",
                    "2+ years change champion experience",
                    "Senior leadership endorsement"
                ],
                resources_required=[
                    "Advanced training budget",
                    "Executive sponsor",
                    "Strategic project assignment"
                ]
            )
        }
    
    def _initialize_capability_weights(self) -> Dict[ChangeCapability, float]:
        """Initialize weights for change champion capabilities"""
        return {
            ChangeCapability.CHANGE_ADVOCACY: 0.15,
            ChangeCapability.INFLUENCE_BUILDING: 0.14,
            ChangeCapability.COMMUNICATION: 0.13,
            ChangeCapability.TRAINING_DELIVERY: 0.10,
            ChangeCapability.RESISTANCE_MANAGEMENT: 0.12,
            ChangeCapability.NETWORK_BUILDING: 0.11,
            ChangeCapability.FEEDBACK_COLLECTION: 0.08,
            ChangeCapability.COACHING_MENTORING: 0.09,
            ChangeCapability.PROJECT_COORDINATION: 0.06,
            ChangeCapability.CULTURAL_SENSITIVITY: 0.02
        }
    
    def _initialize_training_resources(self) -> Dict[str, List[Dict[str, Any]]]:
        """Initialize training resources for each capability"""
        return {
            "change_advocacy": [
                {
                    "type": "workshop",
                    "title": "Change Advocacy Masterclass",
                    "duration": 16,
                    "description": "Learn to become an effective change advocate"
                },
                {
                    "type": "online_course",
                    "title": "Building Change Momentum",
                    "duration": 8,
                    "description": "Online course on creating change momentum"
                }
            ],
            "influence_building": [
                {
                    "type": "workshop",
                    "title": "Influence Without Authority",
                    "duration": 12,
                    "description": "Build influence across organizational levels"
                },
                {
                    "type": "coaching",
                    "title": "Personal Influence Coaching",
                    "duration": 20,
                    "description": "One-on-one coaching to build influence skills"
                }
            ]
        }
    
    def _initialize_network_templates(self) -> Dict[str, Dict[str, Any]]:
        """Initialize change champion network templates"""
        return {
            "departmental": {
                "name": "Departmental Change Network",
                "description": "Change champions within a single department",
                "optimal_size": "8-12 champions",
                "structure": "Hub and spoke with department lead",
                "meeting_frequency": "Bi-weekly",
                "focus_areas": ["Department-specific changes", "Local resistance management"]
            },
            "cross_functional": {
                "name": "Cross-Functional Change Network",
                "description": "Change champions across multiple departments",
                "optimal_size": "15-20 champions",
                "structure": "Matrix with department representatives",
                "meeting_frequency": "Monthly",
                "focus_areas": ["Organization-wide changes", "Cross-department coordination"]
            }
        }
    
    def identify_potential_champions(
        self,
        organization_id: str,
        employee_data: List[Dict[str, Any]],
        criteria_type: str = "standard",
        target_count: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Identify potential change champions from employee data"""
        try:
            logger.info(f"Identifying potential change champions for organization {organization_id}")
            
            criteria = self.identification_criteria.get(criteria_type)
            if not criteria:
                raise ValueError(f"Unknown criteria type: {criteria_type}")
            
            candidates = []
            
            for employee in employee_data:
                score = self._evaluate_champion_potential(employee, criteria)
                
                if score >= 70:  # Minimum threshold for consideration
                    candidate = {
                        "employee_id": employee["id"],
                        "name": employee["name"],
                        "role": employee["role"],
                        "department": employee["department"],
                        "champion_score": score,
                        "strengths": self._identify_champion_strengths(employee, criteria),
                        "development_areas": self._identify_development_areas(employee, criteria),
                        "recommended_level": self._recommend_champion_level(score),
                        "recommended_roles": self._recommend_champion_roles(employee, score)
                    }
                    candidates.append(candidate)
            
            # Sort by score and limit if requested
            candidates.sort(key=lambda x: x["champion_score"], reverse=True)
            
            if target_count:
                candidates = candidates[:target_count]
            
            logger.info(f"Identified {len(candidates)} potential change champions")
            return candidates
            
        except Exception as e:
            logger.error(f"Error identifying potential champions: {str(e)}")
            raise
    
    def _evaluate_champion_potential(
        self,
        employee: Dict[str, Any],
        criteria: ChampionIdentificationCriteria
    ) -> float:
        """Evaluate an employee's potential as a change champion"""
        scores = {}
        
        # Capability scores
        capability_score = 0
        for capability, min_score in criteria.minimum_scores.items():
            emp_score = employee.get("capabilities", {}).get(capability.value, 0)
            capability_score += max(0, emp_score - min_score) / (100 - min_score) * 100
        
        capability_score = capability_score / len(criteria.minimum_scores)
        scores["capability_scores"] = capability_score
        
        # Influence network score
        network_size = employee.get("network_size", 0)
        cross_dept = employee.get("cross_department_connections", 0)
        credibility = employee.get("credibility_score", 0)
        
        influence_score = (
            min(network_size / criteria.influence_requirements["network_size"], 1) * 40 +
            min(cross_dept / criteria.influence_requirements["cross_department_connections"], 1) * 30 +
            credibility / 100 * 30
        )
        scores["influence_network"] = influence_score
        
        # Experience score
        experience_items = employee.get("experience", [])
        experience_score = min(len(experience_items) / len(criteria.experience_requirements), 1) * 100
        scores["experience"] = experience_score
        
        # Cultural fit score
        cultural_alignment = employee.get("cultural_alignment_score", 0)
        scores["cultural_fit"] = cultural_alignment
        
        # Calculate weighted total
        total_score = sum(
            scores[factor] * weight 
            for factor, weight in criteria.weight_factors.items()
        )
        
        return min(total_score, 100)
    
    def _identify_champion_strengths(
        self,
        employee: Dict[str, Any],
        criteria: ChampionIdentificationCriteria
    ) -> List[str]:
        """Identify employee's strengths as potential champion"""
        strengths = []
        
        capabilities = employee.get("capabilities", {})
        for capability, score in capabilities.items():
            if score >= 80:
                strengths.append(f"Strong {capability.replace('_', ' ').title()} skills")
        
        if employee.get("network_size", 0) > 20:
            strengths.append("Extensive professional network")
        
        if employee.get("credibility_score", 0) > 85:
            strengths.append("High credibility and trust")
        
        return strengths
    
    def _identify_development_areas(
        self,
        employee: Dict[str, Any],
        criteria: ChampionIdentificationCriteria
    ) -> List[str]:
        """Identify development areas for potential champion"""
        development_areas = []
        
        capabilities = employee.get("capabilities", {})
        for capability, min_score in criteria.minimum_scores.items():
            emp_score = capabilities.get(capability.value, 0)
            if emp_score < min_score + 20:  # Within 20 points of minimum
                development_areas.append(f"Strengthen {capability.value.replace('_', ' ').title()}")
        
        if employee.get("cross_department_connections", 0) < 3:
            development_areas.append("Build cross-departmental relationships")
        
        return development_areas
    
    def _recommend_champion_level(self, score: float) -> ChampionLevel:
        """Recommend appropriate champion level based on score"""
        if score >= 90:
            return ChampionLevel.SENIOR
        elif score >= 80:
            return ChampionLevel.ACTIVE
        elif score >= 70:
            return ChampionLevel.DEVELOPING
        else:
            return ChampionLevel.EMERGING
    
    def _recommend_champion_roles(
        self,
        employee: Dict[str, Any],
        score: float
    ) -> List[ChampionRole]:
        """Recommend appropriate champion roles"""
        roles = []
        capabilities = employee.get("capabilities", {})
        
        if capabilities.get("change_advocacy", 0) >= 75:
            roles.append(ChampionRole.ADVOCATE)
        
        if capabilities.get("communication", 0) >= 80:
            roles.append(ChampionRole.FACILITATOR)
        
        if capabilities.get("training_delivery", 0) >= 70:
            roles.append(ChampionRole.TRAINER)
        
        if capabilities.get("coaching_mentoring", 0) >= 75:
            roles.append(ChampionRole.MENTOR)
        
        if capabilities.get("project_coordination", 0) >= 70:
            roles.append(ChampionRole.COORDINATOR)
        
        if score >= 85 and len(roles) >= 3:
            roles.append(ChampionRole.STRATEGIST)
        
        return roles if roles else [ChampionRole.ADVOCATE]
    
    def create_champion_profile(
        self,
        employee_data: Dict[str, Any],
        champion_assessment: Dict[str, Any]
    ) -> ChangeChampionProfile:
        """Create comprehensive change champion profile"""
        try:
            logger.info(f"Creating champion profile for {employee_data['name']}")
            
            # Calculate capability scores
            capabilities = {}
            for capability in ChangeCapability:
                score = champion_assessment.get("capabilities", {}).get(capability.value, 0)
                capabilities[capability] = score
            
            # Determine champion level and roles
            overall_score = sum(capabilities.values()) / len(capabilities)
            champion_level = self._recommend_champion_level(overall_score)
            champion_roles = self._recommend_champion_roles(employee_data, overall_score)
            
            profile = ChangeChampionProfile(
                id=str(uuid.uuid4()),
                employee_id=employee_data["id"],
                name=employee_data["name"],
                role=employee_data["role"],
                department=employee_data["department"],
                organization_id=employee_data["organization_id"],
                champion_level=champion_level,
                champion_roles=champion_roles,
                capabilities=capabilities,
                influence_network=champion_assessment.get("influence_network", []),
                credibility_score=champion_assessment.get("credibility_score", 0),
                engagement_score=champion_assessment.get("engagement_score", 0),
                availability_score=champion_assessment.get("availability_score", 0),
                motivation_score=champion_assessment.get("motivation_score", 0),
                cultural_fit_score=champion_assessment.get("cultural_fit_score", 0),
                change_experience=champion_assessment.get("change_experience", []),
                training_completed=[],
                certifications=[],
                mentorship_relationships=[],
                success_metrics={}
            )
            
            logger.info(f"Champion profile created for {employee_data['name']}")
            return profile
            
        except Exception as e:
            logger.error(f"Error creating champion profile: {str(e)}")
            raise
    
    def design_development_program(
        self,
        champions: List[ChangeChampionProfile],
        program_objectives: List[str],
        constraints: Dict[str, Any]
    ) -> ChampionDevelopmentProgram:
        """Design customized development program for champions"""
        try:
            logger.info(f"Designing development program for {len(champions)} champions")
            
            # Analyze champion levels and needs
            level_distribution = {}
            capability_gaps = {}
            
            for champion in champions:
                level = champion.champion_level
                level_distribution[level] = level_distribution.get(level, 0) + 1
                
                for capability, score in champion.capabilities.items():
                    if score < 70:  # Below proficient level
                        if capability not in capability_gaps:
                            capability_gaps[capability] = []
                        capability_gaps[capability].append(champion.id)
            
            # Determine program structure
            if len(champions) <= 10 and all(c.champion_level == ChampionLevel.EMERGING for c in champions):
                base_program = self.development_programs["foundation"]
            else:
                base_program = self.development_programs["advanced"]
            
            # Customize program based on needs
            customized_modules = []
            for module in base_program.learning_modules:
                # Check if module addresses common capability gaps
                module_needed = any(
                    capability in capability_gaps 
                    for capability in module.target_capabilities
                )
                
                if module_needed or len(module.target_capabilities) == 0:
                    customized_modules.append(module)
            
            # Add specialized modules for specific gaps
            for capability, affected_champions in capability_gaps.items():
                if len(affected_champions) >= 3:  # If 3+ champions need this capability
                    specialized_module = self._create_specialized_module(capability)
                    customized_modules.append(specialized_module)
            
            # Create customized program
            program = ChampionDevelopmentProgram(
                id=str(uuid.uuid4()),
                name=f"Customized Champion Development - {datetime.now().strftime('%Y-%m')}",
                description=f"Tailored program for {len(champions)} change champions",
                target_level=base_program.target_level,
                target_roles=base_program.target_roles,
                duration_weeks=max(base_program.duration_weeks, len(customized_modules) * 2),
                learning_modules=customized_modules,
                practical_assignments=base_program.practical_assignments,
                mentorship_component=len(champions) > 5,
                peer_learning_groups=len(champions) >= 8,
                certification_available=True,
                success_criteria=program_objectives,
                prerequisites=base_program.prerequisites,
                resources_required=self._calculate_resource_requirements(champions, customized_modules)
            )
            
            logger.info(f"Development program designed with {len(customized_modules)} modules")
            return program
            
        except Exception as e:
            logger.error(f"Error designing development program: {str(e)}")
            raise
    
    def _create_specialized_module(self, capability: ChangeCapability) -> LearningModule:
        """Create specialized learning module for specific capability"""
        capability_modules = {
            ChangeCapability.RESISTANCE_MANAGEMENT: LearningModule(
                id=f"mod_specialized_{capability.value}",
                title="Advanced Resistance Management",
                description="Specialized training in managing change resistance",
                target_capabilities=[capability],
                learning_objectives=[
                    "Identify resistance patterns early",
                    "Apply advanced resistance management techniques",
                    "Convert resistance into support"
                ],
                content_type="workshop",
                duration_hours=12,
                delivery_method="in_person",
                materials=["Resistance Management Toolkit", "Case Studies"],
                assessments=["Scenario Analysis", "Action Planning"],
                completion_criteria=["Pass scenario assessment", "Create action plan"]
            ),
            ChangeCapability.NETWORK_BUILDING: LearningModule(
                id=f"mod_specialized_{capability.value}",
                title="Strategic Network Building",
                description="Building and leveraging professional networks for change",
                target_capabilities=[capability],
                learning_objectives=[
                    "Map stakeholder networks",
                    "Build strategic relationships",
                    "Leverage networks for change support"
                ],
                content_type="workshop",
                duration_hours=8,
                delivery_method="virtual",
                materials=["Network Mapping Tools", "Relationship Building Guide"],
                assessments=["Network Map Creation", "Relationship Plan"],
                completion_criteria=["Complete network map", "Develop relationship plan"]
            )
        }
        
        return capability_modules.get(capability, LearningModule(
            id=f"mod_specialized_{capability.value}",
            title=f"Specialized {capability.value.replace('_', ' ').title()} Training",
            description=f"Focused training on {capability.value.replace('_', ' ')}",
            target_capabilities=[capability],
            learning_objectives=[f"Improve {capability.value.replace('_', ' ')} skills"],
            content_type="workshop",
            duration_hours=8,
            delivery_method="blended",
            materials=["Training Materials"],
            assessments=["Skills Assessment"],
            completion_criteria=["Pass assessment"]
        ))
    
    def _calculate_resource_requirements(
        self,
        champions: List[ChangeChampionProfile],
        modules: List[LearningModule]
    ) -> List[str]:
        """Calculate resource requirements for development program"""
        requirements = []
        
        total_training_hours = sum(module.duration_hours for module in modules)
        total_participant_hours = total_training_hours * len(champions)
        
        requirements.append(f"Training budget for {total_participant_hours} participant hours")
        requirements.append(f"Trainer resources for {len(modules)} modules")
        
        if len(champions) > 10:
            requirements.append("Multiple training cohorts")
        
        if any("in_person" in module.delivery_method for module in modules):
            requirements.append("Training facilities")
        
        if any("virtual" in module.delivery_method for module in modules):
            requirements.append("Virtual training platform")
        
        requirements.append("Program coordination resources")
        requirements.append("Assessment and certification materials")
        
        return requirements
    
    def create_champion_network(
        self,
        champions: List[ChangeChampionProfile],
        network_type: str,
        objectives: List[str]
    ) -> ChampionNetwork:
        """Create change champion network"""
        try:
            logger.info(f"Creating {network_type} champion network with {len(champions)} champions")
            
            # Select network lead (highest scoring senior champion)
            senior_champions = [c for c in champions if c.champion_level in [ChampionLevel.SENIOR, ChampionLevel.MASTER]]
            if senior_champions:
                network_lead = max(senior_champions, key=lambda c: sum(c.capabilities.values()))
            else:
                network_lead = max(champions, key=lambda c: sum(c.capabilities.values()))
            
            # Select coordinators (2-3 champions with coordination skills)
            coordinators = [
                c for c in champions 
                if ChampionRole.COORDINATOR in c.champion_roles and c.id != network_lead.id
            ][:3]
            
            # Determine coverage areas
            if network_type == "departmental":
                coverage_areas = list(set(c.department for c in champions))
            else:
                coverage_areas = ["Organization-wide", "Cross-functional initiatives"]
            
            network = ChampionNetwork(
                id=str(uuid.uuid4()),
                name=f"{network_type.title()} Change Champion Network",
                organization_id=champions[0].organization_id,
                network_type=network_type,
                champions=[c.id for c in champions],
                network_lead=network_lead.id,
                coordinators=[c.id for c in coordinators],
                coverage_areas=coverage_areas,
                network_status=NetworkStatus.FORMING,
                formation_date=datetime.now(),
                objectives=objectives,
                success_metrics=[
                    "Network engagement score > 80%",
                    "Change initiative success rate > 85%",
                    "Champion satisfaction > 90%"
                ],
                communication_channels=["Monthly meetings", "Slack channel", "Email updates"],
                meeting_schedule="Monthly",
                governance_structure={
                    "lead": network_lead.id,
                    "coordinators": [c.id for c in coordinators],
                    "decision_making": "Consensus with lead approval",
                    "reporting": "Monthly to leadership"
                },
                performance_metrics={}
            )
            
            logger.info(f"Champion network created with {len(champions)} members")
            return network
            
        except Exception as e:
            logger.error(f"Error creating champion network: {str(e)}")
            raise
    
    def plan_network_coordination(
        self,
        network: ChampionNetwork,
        coordination_period: str,
        key_initiatives: List[str]
    ) -> NetworkCoordinationPlan:
        """Create coordination plan for champion network"""
        try:
            logger.info(f"Planning coordination for network {network.name}")
            
            plan = NetworkCoordinationPlan(
                id=str(uuid.uuid4()),
                network_id=network.id,
                coordination_period=coordination_period,
                objectives=network.objectives,
                key_initiatives=key_initiatives,
                resource_allocation={
                    "meeting_time": "4 hours/month per champion",
                    "coordination_time": "8 hours/month for coordinators",
                    "leadership_time": "2 hours/month for network lead"
                },
                communication_strategy={
                    "regular_meetings": "Monthly 2-hour sessions",
                    "quick_updates": "Bi-weekly 30-min check-ins",
                    "async_communication": "Slack/Teams channels",
                    "reporting": "Monthly dashboard updates"
                },
                training_schedule=[
                    {
                        "month": 1,
                        "focus": "Network formation and goal setting",
                        "duration": "4 hours"
                    },
                    {
                        "month": 3,
                        "focus": "Advanced change techniques",
                        "duration": "6 hours"
                    },
                    {
                        "month": 6,
                        "focus": "Network effectiveness review",
                        "duration": "4 hours"
                    }
                ],
                performance_targets={
                    "network_engagement": 85,
                    "initiative_success_rate": 80,
                    "champion_satisfaction": 90,
                    "knowledge_sharing_frequency": 2  # per month
                },
                risk_mitigation=[
                    "Regular engagement monitoring",
                    "Backup coordinator identification",
                    "Flexible meeting scheduling",
                    "Recognition and reward programs"
                ],
                success_metrics=[
                    "Achieve all performance targets",
                    "Complete all key initiatives on time",
                    "Maintain high champion retention (>90%)",
                    "Demonstrate measurable change impact"
                ],
                review_schedule="Monthly performance reviews, Quarterly strategic reviews",
                stakeholder_engagement=[
                    "Monthly updates to leadership",
                    "Quarterly presentations to executives",
                    "Regular communication with HR",
                    "Feedback sessions with employees"
                ]
            )
            
            logger.info(f"Network coordination plan created for {coordination_period}")
            return plan
            
        except Exception as e:
            logger.error(f"Error planning network coordination: {str(e)}")
            raise
    
    def measure_champion_performance(
        self,
        champion_id: str,
        measurement_period: str,
        performance_data: Dict[str, Any]
    ) -> ChampionPerformanceMetrics:
        """Measure individual change champion performance"""
        try:
            logger.info(f"Measuring performance for champion {champion_id}")
            
            metrics = ChampionPerformanceMetrics(
                champion_id=champion_id,
                measurement_period=measurement_period,
                change_initiatives_supported=performance_data.get("change_initiatives_supported", 0),
                training_sessions_delivered=performance_data.get("training_sessions_delivered", 0),
                employees_influenced=performance_data.get("employees_influenced", 0),
                resistance_cases_resolved=performance_data.get("resistance_cases_resolved", 0),
                feedback_sessions_conducted=performance_data.get("feedback_sessions_conducted", 0),
                network_engagement_score=performance_data.get("network_engagement_score", 0),
                peer_rating=performance_data.get("peer_rating", 0),
                manager_rating=performance_data.get("manager_rating", 0),
                change_success_contribution=performance_data.get("change_success_contribution", 0),
                knowledge_sharing_score=performance_data.get("knowledge_sharing_score", 0),
                mentorship_effectiveness=performance_data.get("mentorship_effectiveness", 0),
                innovation_contributions=performance_data.get("innovation_contributions", 0),
                cultural_alignment_score=performance_data.get("cultural_alignment_score", 0),
                overall_performance_score=self._calculate_overall_performance_score(performance_data),
                recognition_received=performance_data.get("recognition_received", []),
                development_areas=performance_data.get("development_areas", [])
            )
            
            logger.info(f"Performance metrics calculated for champion {champion_id}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error measuring champion performance: {str(e)}")
            raise
    
    def _calculate_overall_performance_score(self, performance_data: Dict[str, Any]) -> float:
        """Calculate overall performance score for champion"""
        weights = {
            "change_initiatives_supported": 0.15,
            "training_sessions_delivered": 0.10,
            "employees_influenced": 0.12,
            "resistance_cases_resolved": 0.13,
            "network_engagement_score": 0.15,
            "peer_rating": 0.10,
            "manager_rating": 0.10,
            "change_success_contribution": 0.15
        }
        
        total_score = 0
        for metric, weight in weights.items():
            value = performance_data.get(metric, 0)
            # Normalize different metrics to 0-100 scale
            if metric in ["peer_rating", "manager_rating", "network_engagement_score", "change_success_contribution"]:
                normalized_value = value  # Already 0-100
            else:
                # For count-based metrics, use a reasonable scale
                max_values = {
                    "change_initiatives_supported": 10,
                    "training_sessions_delivered": 20,
                    "employees_influenced": 100,
                    "resistance_cases_resolved": 15
                }
                max_val = max_values.get(metric, 10)
                normalized_value = min(value / max_val * 100, 100)
            
            total_score += normalized_value * weight
        
        return min(total_score, 100)