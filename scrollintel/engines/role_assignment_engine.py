"""
Role Assignment Engine for Crisis Leadership Excellence

This engine provides clear role and responsibility assignment during crisis situations,
optimizing assignments based on individual strengths and ensuring role clarity.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class RoleType(Enum):
    """Types of crisis roles"""
    CRISIS_COMMANDER = "crisis_commander"
    TECHNICAL_LEAD = "technical_lead"
    COMMUNICATION_LEAD = "communication_lead"
    RESOURCE_COORDINATOR = "resource_coordinator"
    STAKEHOLDER_LIAISON = "stakeholder_liaison"
    OPERATIONS_MANAGER = "operations_manager"
    SECURITY_SPECIALIST = "security_specialist"
    LEGAL_ADVISOR = "legal_advisor"
    MEDIA_HANDLER = "media_handler"
    TEAM_COORDINATOR = "team_coordinator"


class SkillLevel(Enum):
    """Skill proficiency levels"""
    NOVICE = 1
    INTERMEDIATE = 2
    ADVANCED = 3
    EXPERT = 4
    MASTER = 5


@dataclass
class PersonSkill:
    """Individual skill assessment"""
    skill_name: str
    level: SkillLevel
    years_experience: int
    recent_performance: float  # 0.0 to 1.0
    crisis_experience: bool = False


@dataclass
class Person:
    """Person available for crisis role assignment"""
    id: str
    name: str
    current_availability: float  # 0.0 to 1.0
    skills: List[PersonSkill]
    preferred_roles: List[RoleType]
    stress_tolerance: float  # 0.0 to 1.0
    leadership_experience: int  # years
    crisis_history: List[str] = field(default_factory=list)
    current_workload: float = 0.0  # 0.0 to 1.0


@dataclass
class RoleRequirement:
    """Requirements for a specific crisis role"""
    role_type: RoleType
    required_skills: List[str]
    minimum_skill_level: SkillLevel
    leadership_required: bool
    stress_tolerance_required: float
    priority: int  # 1 = highest priority
    estimated_workload: float  # 0.0 to 1.0


@dataclass
class RoleAssignment:
    """Assignment of a person to a crisis role"""
    person_id: str
    role_type: RoleType
    assignment_confidence: float  # 0.0 to 1.0
    responsibilities: List[str]
    reporting_structure: Dict[str, str]
    assignment_time: datetime
    expected_duration: Optional[int] = None  # hours


@dataclass
class AssignmentResult:
    """Result of role assignment process"""
    assignments: List[RoleAssignment]
    unassigned_roles: List[RoleType]
    assignment_quality_score: float
    recommendations: List[str]
    backup_assignments: Dict[RoleType, List[str]]


class RoleAssignmentEngine:
    """Engine for assigning crisis roles based on individual strengths"""
    
    def __init__(self):
        self.role_definitions = self._initialize_role_definitions()
        self.assignment_history: List[RoleAssignment] = []
        
    def _initialize_role_definitions(self) -> Dict[RoleType, RoleRequirement]:
        """Initialize standard role requirements"""
        return {
            RoleType.CRISIS_COMMANDER: RoleRequirement(
                role_type=RoleType.CRISIS_COMMANDER,
                required_skills=["leadership", "decision_making", "crisis_management"],
                minimum_skill_level=SkillLevel.ADVANCED,
                leadership_required=True,
                stress_tolerance_required=0.9,
                priority=1,
                estimated_workload=0.9
            ),
            RoleType.TECHNICAL_LEAD: RoleRequirement(
                role_type=RoleType.TECHNICAL_LEAD,
                required_skills=["technical_expertise", "problem_solving", "system_architecture"],
                minimum_skill_level=SkillLevel.ADVANCED,
                leadership_required=True,
                stress_tolerance_required=0.8,
                priority=2,
                estimated_workload=0.8
            ),
            RoleType.COMMUNICATION_LEAD: RoleRequirement(
                role_type=RoleType.COMMUNICATION_LEAD,
                required_skills=["communication", "public_relations", "stakeholder_management"],
                minimum_skill_level=SkillLevel.ADVANCED,
                leadership_required=True,
                stress_tolerance_required=0.8,
                priority=2,
                estimated_workload=0.7
            ),
            RoleType.RESOURCE_COORDINATOR: RoleRequirement(
                role_type=RoleType.RESOURCE_COORDINATOR,
                required_skills=["resource_management", "logistics", "coordination"],
                minimum_skill_level=SkillLevel.INTERMEDIATE,
                leadership_required=False,
                stress_tolerance_required=0.7,
                priority=3,
                estimated_workload=0.6
            ),
            RoleType.STAKEHOLDER_LIAISON: RoleRequirement(
                role_type=RoleType.STAKEHOLDER_LIAISON,
                required_skills=["stakeholder_management", "communication", "negotiation"],
                minimum_skill_level=SkillLevel.INTERMEDIATE,
                leadership_required=False,
                stress_tolerance_required=0.7,
                priority=3,
                estimated_workload=0.6
            )
        }
    
    def assign_roles(
        self,
        crisis_id: str,
        available_people: List[Person],
        required_roles: List[RoleType],
        crisis_severity: float = 0.5
    ) -> AssignmentResult:
        """
        Assign roles to people based on their strengths and role requirements
        
        Args:
            crisis_id: Unique identifier for the crisis
            available_people: List of people available for assignment
            required_roles: List of roles that need to be filled
            crisis_severity: Severity of crisis (0.0 to 1.0)
            
        Returns:
            AssignmentResult with role assignments and recommendations
        """
        try:
            logger.info(f"Starting role assignment for crisis {crisis_id}")
            
            # Calculate person-role compatibility scores
            compatibility_matrix = self._calculate_compatibility_matrix(
                available_people, required_roles, crisis_severity
            )
            
            # Perform optimal assignment
            assignments = self._optimize_assignments(
                compatibility_matrix, available_people, required_roles
            )
            
            # Generate backup assignments
            backup_assignments = self._generate_backup_assignments(
                compatibility_matrix, assignments, available_people
            )
            
            # Calculate quality score
            quality_score = self._calculate_assignment_quality(assignments, compatibility_matrix)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                assignments, available_people, required_roles, quality_score
            )
            
            # Identify unassigned roles
            assigned_roles = {assignment.role_type for assignment in assignments}
            unassigned_roles = [role for role in required_roles if role not in assigned_roles]
            
            result = AssignmentResult(
                assignments=assignments,
                unassigned_roles=unassigned_roles,
                assignment_quality_score=quality_score,
                recommendations=recommendations,
                backup_assignments=backup_assignments
            )
            
            # Store assignment history
            self.assignment_history.extend(assignments)
            
            logger.info(f"Role assignment completed with quality score: {quality_score:.2f}")
            return result
            
        except Exception as e:
            logger.error(f"Error in role assignment: {str(e)}")
            raise
    
    def _calculate_compatibility_matrix(
        self,
        people: List[Person],
        roles: List[RoleType],
        crisis_severity: float
    ) -> Dict[Tuple[str, RoleType], float]:
        """Calculate compatibility scores between people and roles"""
        matrix = {}
        
        for person in people:
            for role in roles:
                if role in self.role_definitions:
                    score = self._calculate_person_role_score(
                        person, self.role_definitions[role], crisis_severity
                    )
                    matrix[(person.id, role)] = score
                else:
                    matrix[(person.id, role)] = 0.0
        
        return matrix
    
    def _calculate_person_role_score(
        self,
        person: Person,
        role_req: RoleRequirement,
        crisis_severity: float
    ) -> float:
        """Calculate how well a person fits a specific role"""
        score = 0.0
        
        # Check skill requirements
        skill_score = self._calculate_skill_match(person, role_req)
        score += skill_score * 0.4
        
        # Check availability
        availability_score = person.current_availability * (1 - person.current_workload)
        score += availability_score * 0.2
        
        # Check stress tolerance
        stress_score = min(person.stress_tolerance / role_req.stress_tolerance_required, 1.0)
        score += stress_score * 0.2
        
        # Check leadership requirement
        if role_req.leadership_required:
            leadership_score = min(person.leadership_experience / 5.0, 1.0)  # 5 years = max score
            score += leadership_score * 0.1
        else:
            score += 0.1  # No penalty for non-leadership roles
        
        # Check role preference
        if role_req.role_type in person.preferred_roles:
            score += 0.1
        
        # Adjust for crisis severity
        crisis_adjustment = 1.0 + (crisis_severity * 0.2)  # Up to 20% bonus for high severity
        score *= crisis_adjustment
        
        return min(score, 1.0)
    
    def _calculate_skill_match(self, person: Person, role_req: RoleRequirement) -> float:
        """Calculate how well person's skills match role requirements"""
        person_skills = {skill.skill_name: skill for skill in person.skills}
        
        total_score = 0.0
        skill_count = len(role_req.required_skills)
        
        for required_skill in role_req.required_skills:
            if required_skill in person_skills:
                skill = person_skills[required_skill]
                
                # Base score from skill level
                level_score = skill.level.value / SkillLevel.MASTER.value
                
                # Bonus for crisis experience
                crisis_bonus = 0.2 if skill.crisis_experience else 0.0
                
                # Performance factor
                performance_factor = skill.recent_performance
                
                skill_score = (level_score + crisis_bonus) * performance_factor
                total_score += min(skill_score, 1.0)
            else:
                # Penalty for missing required skill
                total_score += 0.0
        
        return total_score / skill_count if skill_count > 0 else 0.0
    
    def _optimize_assignments(
        self,
        compatibility_matrix: Dict[Tuple[str, RoleType], float],
        people: List[Person],
        roles: List[RoleType]
    ) -> List[RoleAssignment]:
        """Optimize role assignments using greedy algorithm with constraints"""
        assignments = []
        assigned_people = set()
        assigned_roles = set()
        
        # Sort roles by priority
        sorted_roles = sorted(roles, key=lambda r: self.role_definitions.get(r, RoleRequirement(
            role_type=r, required_skills=[], minimum_skill_level=SkillLevel.NOVICE,
            leadership_required=False, stress_tolerance_required=0.5, priority=10,
            estimated_workload=0.5
        )).priority)
        
        for role in sorted_roles:
            if role in assigned_roles:
                continue
                
            best_person = None
            best_score = 0.0
            
            for person in people:
                if person.id in assigned_people:
                    continue
                    
                score = compatibility_matrix.get((person.id, role), 0.0)
                if score > best_score:
                    best_score = score
                    best_person = person
            
            if best_person and best_score > 0.3:  # Minimum threshold
                assignment = self._create_role_assignment(best_person, role, best_score)
                assignments.append(assignment)
                assigned_people.add(best_person.id)
                assigned_roles.add(role)
        
        return assignments
    
    def _create_role_assignment(
        self,
        person: Person,
        role: RoleType,
        confidence: float
    ) -> RoleAssignment:
        """Create a role assignment with responsibilities and reporting structure"""
        responsibilities = self._get_role_responsibilities(role)
        reporting_structure = self._determine_reporting_structure(role)
        
        return RoleAssignment(
            person_id=person.id,
            role_type=role,
            assignment_confidence=confidence,
            responsibilities=responsibilities,
            reporting_structure=reporting_structure,
            assignment_time=datetime.now()
        )
    
    def _get_role_responsibilities(self, role: RoleType) -> List[str]:
        """Get standard responsibilities for a role type"""
        responsibilities_map = {
            RoleType.CRISIS_COMMANDER: [
                "Overall crisis response leadership",
                "Strategic decision making",
                "Resource allocation approval",
                "Stakeholder communication oversight",
                "Crisis resolution coordination"
            ],
            RoleType.TECHNICAL_LEAD: [
                "Technical problem diagnosis",
                "Solution architecture design",
                "Technical team coordination",
                "System recovery planning",
                "Technical risk assessment"
            ],
            RoleType.COMMUNICATION_LEAD: [
                "Internal communication coordination",
                "External stakeholder messaging",
                "Media relations management",
                "Communication strategy development",
                "Message consistency oversight"
            ],
            RoleType.RESOURCE_COORDINATOR: [
                "Resource inventory management",
                "Resource allocation coordination",
                "Vendor and supplier liaison",
                "Emergency procurement",
                "Resource utilization tracking"
            ],
            RoleType.STAKEHOLDER_LIAISON: [
                "Stakeholder identification and mapping",
                "Stakeholder communication management",
                "Relationship maintenance",
                "Feedback collection and analysis",
                "Stakeholder satisfaction monitoring"
            ]
        }
        
        return responsibilities_map.get(role, ["Role-specific responsibilities to be defined"])
    
    def _determine_reporting_structure(self, role: RoleType) -> Dict[str, str]:
        """Determine reporting relationships for the role"""
        if role == RoleType.CRISIS_COMMANDER:
            return {"reports_to": "CEO/Board", "manages": "All crisis team members"}
        elif role in [RoleType.TECHNICAL_LEAD, RoleType.COMMUNICATION_LEAD]:
            return {"reports_to": "Crisis Commander", "manages": "Respective team members"}
        else:
            return {"reports_to": "Crisis Commander", "manages": "None"}
    
    def _generate_backup_assignments(
        self,
        compatibility_matrix: Dict[Tuple[str, RoleType], float],
        primary_assignments: List[RoleAssignment],
        people: List[Person]
    ) -> Dict[RoleType, List[str]]:
        """Generate backup assignments for each role"""
        backup_assignments = {}
        assigned_people = {assignment.person_id for assignment in primary_assignments}
        
        for assignment in primary_assignments:
            role = assignment.role_type
            backups = []
            
            # Find top 2 backup candidates
            candidates = []
            for person in people:
                if person.id not in assigned_people:
                    score = compatibility_matrix.get((person.id, role), 0.0)
                    if score > 0.2:  # Minimum backup threshold
                        candidates.append((person.id, score))
            
            # Sort by score and take top 2
            candidates.sort(key=lambda x: x[1], reverse=True)
            backups = [person_id for person_id, _ in candidates[:2]]
            
            backup_assignments[role] = backups
        
        return backup_assignments
    
    def _calculate_assignment_quality(
        self,
        assignments: List[RoleAssignment],
        compatibility_matrix: Dict[Tuple[str, RoleType], float]
    ) -> float:
        """Calculate overall quality of assignments"""
        if not assignments:
            return 0.0
        
        total_confidence = sum(assignment.assignment_confidence for assignment in assignments)
        return total_confidence / len(assignments)
    
    def _generate_recommendations(
        self,
        assignments: List[RoleAssignment],
        people: List[Person],
        required_roles: List[RoleType],
        quality_score: float
    ) -> List[str]:
        """Generate recommendations for improving assignments"""
        recommendations = []
        
        if quality_score < 0.7:
            recommendations.append("Consider additional training for assigned personnel")
        
        assigned_roles = {assignment.role_type for assignment in assignments}
        unassigned_roles = set(required_roles) - assigned_roles
        
        if unassigned_roles:
            recommendations.append(f"Critical roles unassigned: {', '.join([role.value for role in unassigned_roles])}")
        
        # Check for overloaded assignments
        person_workloads = {}
        for assignment in assignments:
            person_id = assignment.person_id
            role_workload = self.role_definitions.get(assignment.role_type, RoleRequirement(
                role_type=assignment.role_type, required_skills=[], minimum_skill_level=SkillLevel.NOVICE,
                leadership_required=False, stress_tolerance_required=0.5, priority=10,
                estimated_workload=0.5
            )).estimated_workload
            
            person_workloads[person_id] = person_workloads.get(person_id, 0.0) + role_workload
        
        for person_id, workload in person_workloads.items():
            if workload > 0.8:
                recommendations.append(f"Person {person_id} may be overloaded (workload: {workload:.1f})")
        
        return recommendations
    
    def get_role_clarity_communication(self, assignment: RoleAssignment) -> Dict[str, any]:
        """Generate clear role communication for an assignment"""
        person_name = f"Person {assignment.person_id}"  # In real implementation, lookup actual name
        
        return {
            "assignment_summary": {
                "person": person_name,
                "role": assignment.role_type.value.replace('_', ' ').title(),
                "confidence": f"{assignment.assignment_confidence:.1%}",
                "assignment_time": assignment.assignment_time.isoformat()
            },
            "responsibilities": assignment.responsibilities,
            "reporting_structure": assignment.reporting_structure,
            "success_criteria": self._get_role_success_criteria(assignment.role_type),
            "communication_protocols": self._get_communication_protocols(assignment.role_type),
            "escalation_procedures": self._get_escalation_procedures(assignment.role_type)
        }
    
    def _get_role_success_criteria(self, role: RoleType) -> List[str]:
        """Get success criteria for role performance"""
        criteria_map = {
            RoleType.CRISIS_COMMANDER: [
                "Crisis resolved within target timeframe",
                "Stakeholder satisfaction maintained",
                "Team coordination effectiveness",
                "Decision quality and speed"
            ],
            RoleType.TECHNICAL_LEAD: [
                "Technical issues identified and resolved",
                "System stability restored",
                "Technical team productivity",
                "Solution effectiveness"
            ],
            RoleType.COMMUNICATION_LEAD: [
                "Message consistency across channels",
                "Stakeholder communication timeliness",
                "Media relations effectiveness",
                "Internal team information flow"
            ]
        }
        
        return criteria_map.get(role, ["Role performance meets expectations"])
    
    def _get_communication_protocols(self, role: RoleType) -> Dict[str, str]:
        """Get communication protocols for the role"""
        return {
            "update_frequency": "Every 30 minutes during active crisis",
            "reporting_method": "Status dashboard and direct communication",
            "escalation_trigger": "Any issue requiring immediate attention",
            "documentation_required": "All decisions and actions must be logged"
        }
    
    def _get_escalation_procedures(self, role: RoleType) -> List[str]:
        """Get escalation procedures for the role"""
        return [
            "Immediate escalation for issues beyond role authority",
            "Escalate if unable to reach key stakeholders within 15 minutes",
            "Escalate any legal or regulatory concerns immediately",
            "Escalate if crisis severity increases beyond current response capability"
        ]
    
    def confirm_role_assignment(self, assignment_id: str, person_confirmation: bool) -> Dict[str, any]:
        """Confirm role assignment with the assigned person"""
        # In real implementation, would lookup assignment by ID
        return {
            "assignment_confirmed": person_confirmation,
            "confirmation_time": datetime.now().isoformat(),
            "next_steps": [
                "Review role responsibilities and success criteria",
                "Confirm communication protocols",
                "Begin role-specific crisis response activities",
                "Establish regular check-in schedule"
            ] if person_confirmation else [
                "Identify alternative assignment options",
                "Reassess person availability and constraints",
                "Consider role requirement adjustments",
                "Activate backup assignment if available"
            ]
        }