"""
Crisis Team Formation Engine

This engine provides rapid assembly of appropriate crisis response teams with
optimized team composition based on crisis type, skill matching, and availability.
"""

from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import asdict
import asyncio

from ..models.team_coordination_models import (
    Person, CrisisTeam, TeamFormationRequest, RoleAssignment, 
    TeamComposition, SkillMatch, TeamRole, SkillLevel, 
    AvailabilityStatus, CrisisTeamTemplate, Skill
)
from ..models.crisis_models_simple import Crisis, CrisisType

logger = logging.getLogger(__name__)


class CrisisTeamFormationEngine:
    """Engine for rapid crisis team assembly and optimization"""
    
    def __init__(self):
        self.personnel_registry = {}  # person_id -> Person
        self.team_templates = {}  # crisis_type -> CrisisTeamTemplate
        self.active_teams = {}  # team_id -> CrisisTeam
        self.skill_requirements_db = self._initialize_skill_requirements()
        self.team_composition_rules = self._initialize_team_composition_rules()
        
    async def form_crisis_team(self, request: TeamFormationRequest) -> CrisisTeam:
        """Rapidly assemble appropriate crisis response team"""
        logger.info(f"Forming crisis team for crisis {request.crisis_id} of type {request.crisis_type}")
        
        # Get team composition requirements
        composition = self._get_team_composition(request.crisis_type, request.severity_level)
        
        # Find available personnel
        available_personnel = await self._get_available_personnel()
        
        # Match personnel to roles
        role_matches = await self._match_personnel_to_roles(
            available_personnel, composition, request
        )
        
        # Optimize team composition
        optimal_team = await self._optimize_team_composition(
            role_matches, composition, request
        )
        
        # Create role assignments
        role_assignments = await self._create_role_assignments(optimal_team, composition)
        
        # Form the crisis team
        crisis_team = CrisisTeam(
            crisis_id=request.crisis_id,
            team_name=f"Crisis Response Team - {request.crisis_type}",
            crisis_type=request.crisis_type,
            team_lead_id=optimal_team.get('team_lead', ''),
            members=list(optimal_team.values()),
            role_assignments=role_assignments,
            team_status="forming"
        )
        
        # Setup communication channels
        await self._setup_team_communication(crisis_team)
        
        # Register team
        self.active_teams[crisis_team.id] = crisis_team
        
        logger.info(f"Crisis team {crisis_team.id} formed with {len(crisis_team.members)} members")
        return crisis_team
    
    async def optimize_team_composition(self, team_id: str, crisis_type: str) -> CrisisTeam:
        """Optimize existing team composition based on crisis type"""
        if team_id not in self.active_teams:
            raise ValueError(f"Team {team_id} not found")
        
        team = self.active_teams[team_id]
        composition = self._get_team_composition(crisis_type, 3)  # Default medium severity
        
        # Analyze current team effectiveness
        effectiveness_analysis = await self._analyze_team_effectiveness(team, composition)
        
        # Identify optimization opportunities
        optimization_suggestions = await self._identify_optimization_opportunities(
            team, composition, effectiveness_analysis
        )
        
        # Apply optimizations
        optimized_team = await self._apply_team_optimizations(team, optimization_suggestions)
        
        self.active_teams[team_id] = optimized_team
        return optimized_team
    
    async def match_skills_to_availability(self, required_skills: List[str]) -> List[SkillMatch]:
        """Match required skills to available personnel"""
        available_personnel = await self._get_available_personnel()
        skill_matches = []
        
        for person in available_personnel:
            for role in TeamRole:
                match = await self._calculate_skill_match(person, role, required_skills)
                if match.overall_match_score > 0.3:  # Minimum threshold
                    skill_matches.append(match)
        
        # Sort by match score
        skill_matches.sort(key=lambda x: x.overall_match_score, reverse=True)
        return skill_matches
    
    def _get_team_composition(self, crisis_type: str, severity_level: int) -> TeamComposition:
        """Get optimal team composition for crisis type and severity"""
        base_compositions = {
            "system_outage": TeamComposition(
                crisis_type=crisis_type,
                required_roles=[
                    TeamRole.CRISIS_LEADER,
                    TeamRole.TECHNICAL_LEAD,
                    TeamRole.COMMUNICATIONS_LEAD,
                    TeamRole.CUSTOMER_LIAISON
                ],
                team_size_range=(4, 8),
                skill_requirements={
                    "system_administration": SkillLevel.ADVANCED,
                    "incident_response": SkillLevel.INTERMEDIATE,
                    "communication": SkillLevel.INTERMEDIATE,
                    "customer_service": SkillLevel.INTERMEDIATE
                },
                experience_requirements={
                    "system_outage": 2,
                    "crisis_management": 1
                },
                availability_requirements={
                    "immediate": 1.0,
                    "sustained": 0.8
                }
            ),
            "security_breach": TeamComposition(
                crisis_type=crisis_type,
                required_roles=[
                    TeamRole.CRISIS_LEADER,
                    TeamRole.SECURITY_LEAD,
                    TeamRole.TECHNICAL_LEAD,
                    TeamRole.LEGAL_ADVISOR,
                    TeamRole.COMMUNICATIONS_LEAD
                ],
                team_size_range=(5, 10),
                skill_requirements={
                    "cybersecurity": SkillLevel.EXPERT,
                    "forensics": SkillLevel.ADVANCED,
                    "legal_compliance": SkillLevel.INTERMEDIATE,
                    "crisis_communication": SkillLevel.ADVANCED
                },
                experience_requirements={
                    "security_breach": 3,
                    "incident_response": 2
                },
                availability_requirements={
                    "immediate": 1.0,
                    "sustained": 0.9
                }
            ),
            "financial_crisis": TeamComposition(
                crisis_type=crisis_type,
                required_roles=[
                    TeamRole.CRISIS_LEADER,
                    TeamRole.EXECUTIVE_LIAISON,
                    TeamRole.COMMUNICATIONS_LEAD,
                    TeamRole.LEGAL_ADVISOR,
                    TeamRole.RESOURCE_COORDINATOR
                ],
                team_size_range=(5, 12),
                skill_requirements={
                    "financial_analysis": SkillLevel.EXPERT,
                    "strategic_planning": SkillLevel.ADVANCED,
                    "stakeholder_management": SkillLevel.ADVANCED,
                    "legal_compliance": SkillLevel.INTERMEDIATE
                },
                experience_requirements={
                    "financial_crisis": 2,
                    "executive_communication": 3
                },
                availability_requirements={
                    "immediate": 0.8,
                    "sustained": 1.0
                }
            )
        }
        
        composition = base_compositions.get(crisis_type, base_compositions["system_outage"])
        
        # Adjust for severity level
        if severity_level >= 4:  # High/Critical severity
            # Increase team size for high severity
            min_size, max_size = composition.team_size_range
            composition.team_size_range = (min_size + 2, max_size + 4)
            
            # Add additional roles for high severity
            if TeamRole.EXECUTIVE_LIAISON not in composition.required_roles:
                composition.required_roles.append(TeamRole.EXECUTIVE_LIAISON)
            if TeamRole.DOCUMENTATION_LEAD not in composition.required_roles:
                composition.required_roles.append(TeamRole.DOCUMENTATION_LEAD)
        
        return composition
    
    async def _get_available_personnel(self) -> List[Person]:
        """Get list of available personnel for crisis response"""
        available = []
        
        for person in self.personnel_registry.values():
            # In crisis situations, we may need to use overloaded personnel
            # Primary criteria: availability status
            if person.availability_status == AvailabilityStatus.AVAILABLE:
                available.append(person)
            # Secondary: people who are busy but not in another crisis
            elif (person.availability_status == AvailabilityStatus.BUSY and 
                  person.current_workload < 0.95):  # Still has some capacity
                available.append(person)
        
        return available
    
    async def _match_personnel_to_roles(
        self, 
        personnel: List[Person], 
        composition: TeamComposition,
        request: TeamFormationRequest
    ) -> Dict[TeamRole, List[SkillMatch]]:
        """Match available personnel to required roles"""
        role_matches = {}
        
        for role in composition.required_roles:
            matches = []
            for person in personnel:
                match = await self._calculate_skill_match(person, role, request.required_skills)
                # Lower threshold for crisis situations - we need people even if not perfect
                if match.overall_match_score > 0.1:  # Very low minimum threshold
                    matches.append(match)
            
            # If no matches found, create basic matches for all available personnel
            if not matches and personnel:
                for person in personnel:
                    basic_match = SkillMatch(
                        person_id=person.id,
                        role=role,
                        skill_match_score=0.3,  # Basic capability assumption
                        experience_match_score=0.2,
                        availability_score=1.0 - person.current_workload,
                        overall_match_score=0.3,
                        missing_skills=[],
                        strengths=[],
                        match_rationale=f"Emergency assignment - {person.name} available for {role.value}"
                    )
                    matches.append(basic_match)
            
            # Sort by match score
            matches.sort(key=lambda x: x.overall_match_score, reverse=True)
            role_matches[role] = matches[:10]  # Top 10 candidates per role
        
        return role_matches
    
    async def _calculate_skill_match(
        self, 
        person: Person, 
        role: TeamRole, 
        required_skills: List[str]
    ) -> SkillMatch:
        """Calculate how well a person matches a role"""
        
        # Get role skill requirements
        role_skills = self._get_role_skill_requirements(role)
        
        # Calculate skill match score
        skill_scores = []
        missing_skills = []
        strengths = []
        
        for skill_name, required_level in role_skills.items():
            person_skill = self._find_person_skill(person, skill_name)
            if person_skill:
                skill_score = self._calculate_skill_level_score(person_skill.level, required_level)
                skill_scores.append(skill_score)
                if skill_score > 0.8:
                    strengths.append(skill_name)
            else:
                skill_scores.append(0.0)
                missing_skills.append(skill_name)
        
        skill_match_score = sum(skill_scores) / len(skill_scores) if skill_scores else 0.0
        
        # Calculate experience match score
        crisis_experience = person.crisis_experience.get(role.value, 0)
        experience_match_score = min(1.0, crisis_experience / 5.0)  # Normalize to 0-1
        
        # Calculate availability score
        availability_score = 1.0 - person.current_workload
        if person.availability_status != AvailabilityStatus.AVAILABLE:
            availability_score *= 0.5
        
        # Calculate overall match score
        weights = {"skill": 0.5, "experience": 0.3, "availability": 0.2}
        overall_score = (
            skill_match_score * weights["skill"] +
            experience_match_score * weights["experience"] +
            availability_score * weights["availability"]
        )
        
        # Generate match rationale
        rationale = self._generate_match_rationale(
            person, role, skill_match_score, experience_match_score, 
            availability_score, strengths, missing_skills
        )
        
        return SkillMatch(
            person_id=person.id,
            role=role,
            skill_match_score=skill_match_score,
            experience_match_score=experience_match_score,
            availability_score=availability_score,
            overall_match_score=overall_score,
            missing_skills=missing_skills,
            strengths=strengths,
            match_rationale=rationale
        )
    
    async def _optimize_team_composition(
        self,
        role_matches: Dict[TeamRole, List[SkillMatch]],
        composition: TeamComposition,
        request: TeamFormationRequest
    ) -> Dict[str, str]:
        """Optimize team composition using constraint satisfaction"""
        
        # Initialize team assignment
        team_assignment = {}
        assigned_people = set()
        
        # Sort roles by importance/difficulty to fill
        role_priority = self._get_role_priority(composition.required_roles)
        
        # Assign people to roles in priority order
        for role in role_priority:
            if role not in role_matches:
                continue
                
            # Find best available match
            for match in role_matches[role]:
                if match.person_id not in assigned_people:
                    team_assignment[role.value] = match.person_id
                    assigned_people.add(match.person_id)
                    break
        
        # If no assignments were made, try to assign anyone available with minimum threshold
        if not team_assignment:
            # Get all available personnel and assign them to any role they can fill
            available_personnel = await self._get_available_personnel()
            if available_personnel:
                # Assign the first available person to crisis leader role
                best_person = max(available_personnel, key=lambda p: self._get_leadership_score(p.id))
                team_assignment['crisis_leader'] = best_person.id
                assigned_people.add(best_person.id)
                
                # Try to fill other roles with remaining personnel
                for role in role_priority[1:]:  # Skip crisis_leader as it's already assigned
                    for person in available_personnel:
                        if person.id not in assigned_people:
                            team_assignment[role.value] = person.id
                            assigned_people.add(person.id)
                            break
        
        # Ensure we have a team lead
        if 'crisis_leader' not in team_assignment and team_assignment:
            # Promote the best available person to team lead
            best_person = max(
                team_assignment.values(),
                key=lambda pid: self._get_leadership_score(pid)
            )
            team_assignment['team_lead'] = best_person
        elif 'crisis_leader' in team_assignment:
            team_assignment['team_lead'] = team_assignment['crisis_leader']
        
        return team_assignment
    
    async def _create_role_assignments(
        self,
        team_assignment: Dict[str, str],
        composition: TeamComposition
    ) -> List[RoleAssignment]:
        """Create detailed role assignments for team members"""
        assignments = []
        
        for role_name, person_id in team_assignment.items():
            if role_name == 'team_lead':
                continue  # Handle separately
                
            try:
                role = TeamRole(role_name)
            except ValueError:
                continue  # Skip invalid roles
            
            # Get role responsibilities
            responsibilities = self._get_role_responsibilities(role)
            required_skills = list(self._get_role_skill_requirements(role).keys())
            
            # Calculate assignment confidence
            person = self.personnel_registry.get(person_id)
            confidence = 0.8  # Default confidence
            if person:
                skill_match = await self._calculate_skill_match(person, role, [])
                confidence = skill_match.overall_match_score
            
            assignment = RoleAssignment(
                person_id=person_id,
                role=role,
                assignment_confidence=confidence,
                responsibilities=responsibilities,
                required_skills=required_skills,
                assignment_rationale=f"Best available match for {role.value} with {confidence:.2f} confidence"
            )
            assignments.append(assignment)
        
        return assignments
    
    async def _setup_team_communication(self, team: CrisisTeam) -> None:
        """Setup communication channels for crisis team"""
        team.communication_channels = {
            "primary_chat": f"crisis-team-{team.id}",
            "voice_channel": f"crisis-voice-{team.id}",
            "video_conference": f"https://meet.company.com/crisis-{team.id}",
            "status_board": f"https://status.company.com/crisis-{team.id}",
            "documentation": f"https://docs.company.com/crisis-{team.id}"
        }
        
        # Setup escalation contacts
        team.escalation_contacts = [
            team.team_lead_id,
            "crisis-management@company.com",
            "executive-team@company.com"
        ]
    
    async def _analyze_team_effectiveness(
        self, 
        team: CrisisTeam, 
        composition: TeamComposition
    ) -> Dict[str, Any]:
        """Analyze current team effectiveness"""
        analysis = {
            "skill_coverage": 0.0,
            "experience_level": 0.0,
            "availability_score": 0.0,
            "team_balance": 0.0,
            "communication_efficiency": 0.0,
            "gaps": [],
            "strengths": []
        }
        
        # Analyze skill coverage
        required_skills = composition.skill_requirements
        covered_skills = 0
        total_skills = len(required_skills)
        
        for skill_name, required_level in required_skills.items():
            team_has_skill = False
            for member_id in team.members:
                person = self.personnel_registry.get(member_id)
                if person and self._person_has_skill(person, skill_name, required_level):
                    team_has_skill = True
                    break
            if team_has_skill:
                covered_skills += 1
            else:
                analysis["gaps"].append(f"Missing {skill_name} at {required_level.value} level")
        
        analysis["skill_coverage"] = covered_skills / total_skills if total_skills > 0 else 0.0
        
        return analysis
    
    async def _identify_optimization_opportunities(
        self,
        team: CrisisTeam,
        composition: TeamComposition,
        effectiveness_analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Identify opportunities to optimize team composition"""
        opportunities = []
        
        # Check for skill gaps
        if effectiveness_analysis["skill_coverage"] < 0.8:
            opportunities.append({
                "type": "skill_gap",
                "description": "Team lacks required skills",
                "priority": "high",
                "action": "add_specialist"
            })
        
        # Check for overloaded members
        for member_id in team.members:
            person = self.personnel_registry.get(member_id)
            if person and person.current_workload > 0.9:
                opportunities.append({
                    "type": "workload_balance",
                    "description": f"Member {person.name} is overloaded",
                    "priority": "medium",
                    "action": "redistribute_tasks"
                })
        
        return opportunities
    
    async def _apply_team_optimizations(
        self,
        team: CrisisTeam,
        optimizations: List[Dict[str, Any]]
    ) -> CrisisTeam:
        """Apply optimization suggestions to team"""
        for optimization in optimizations:
            if optimization["action"] == "add_specialist":
                # Find and add specialist
                available_personnel = await self._get_available_personnel()
                # Implementation would add best available specialist
                pass
            elif optimization["action"] == "redistribute_tasks":
                # Redistribute workload among team members
                # Implementation would rebalance assignments
                pass
        
        return team
    
    def _get_role_skill_requirements(self, role: TeamRole) -> Dict[str, SkillLevel]:
        """Get skill requirements for a specific role"""
        requirements = {
            TeamRole.CRISIS_LEADER: {
                "crisis_management": SkillLevel.EXPERT,
                "leadership": SkillLevel.ADVANCED,
                "decision_making": SkillLevel.ADVANCED,
                "communication": SkillLevel.ADVANCED
            },
            TeamRole.TECHNICAL_LEAD: {
                "system_administration": SkillLevel.EXPERT,
                "troubleshooting": SkillLevel.ADVANCED,
                "technical_communication": SkillLevel.INTERMEDIATE
            },
            TeamRole.COMMUNICATIONS_LEAD: {
                "crisis_communication": SkillLevel.EXPERT,
                "public_relations": SkillLevel.ADVANCED,
                "stakeholder_management": SkillLevel.ADVANCED
            },
            TeamRole.SECURITY_LEAD: {
                "cybersecurity": SkillLevel.EXPERT,
                "incident_response": SkillLevel.ADVANCED,
                "forensics": SkillLevel.INTERMEDIATE
            },
            TeamRole.LEGAL_ADVISOR: {
                "legal_compliance": SkillLevel.EXPERT,
                "risk_assessment": SkillLevel.ADVANCED,
                "regulatory_knowledge": SkillLevel.ADVANCED
            }
        }
        
        return requirements.get(role, {})
    
    def _get_role_responsibilities(self, role: TeamRole) -> List[str]:
        """Get list of responsibilities for a role"""
        responsibilities = {
            TeamRole.CRISIS_LEADER: [
                "Overall crisis response coordination",
                "Strategic decision making",
                "Stakeholder communication",
                "Resource allocation decisions",
                "Team coordination and motivation"
            ],
            TeamRole.TECHNICAL_LEAD: [
                "Technical problem diagnosis",
                "Solution implementation oversight",
                "System recovery coordination",
                "Technical team management",
                "Technical communication to leadership"
            ],
            TeamRole.COMMUNICATIONS_LEAD: [
                "External communication management",
                "Media relations",
                "Customer communication",
                "Message consistency coordination",
                "Stakeholder updates"
            ],
            TeamRole.SECURITY_LEAD: [
                "Security incident response",
                "Threat assessment and mitigation",
                "Forensic investigation coordination",
                "Security protocol implementation",
                "Compliance verification"
            ]
        }
        
        return responsibilities.get(role, [])
    
    def _find_person_skill(self, person: Person, skill_name: str) -> Optional[Skill]:
        """Find a specific skill in person's skill set"""
        for skill in person.skills:
            if skill.name.lower() == skill_name.lower():
                return skill
        return None
    
    def _calculate_skill_level_score(self, person_level: SkillLevel, required_level: SkillLevel) -> float:
        """Calculate score based on skill level match"""
        level_values = {
            SkillLevel.BEGINNER: 1,
            SkillLevel.INTERMEDIATE: 2,
            SkillLevel.ADVANCED: 3,
            SkillLevel.EXPERT: 4,
            SkillLevel.MASTER: 5
        }
        
        person_value = level_values[person_level]
        required_value = level_values[required_level]
        
        if person_value >= required_value:
            return 1.0  # Meets or exceeds requirement
        else:
            return person_value / required_value  # Partial match
    
    def _person_has_skill(self, person: Person, skill_name: str, required_level: SkillLevel) -> bool:
        """Check if person has skill at required level"""
        skill = self._find_person_skill(person, skill_name)
        if not skill:
            return False
        
        level_values = {
            SkillLevel.BEGINNER: 1,
            SkillLevel.INTERMEDIATE: 2,
            SkillLevel.ADVANCED: 3,
            SkillLevel.EXPERT: 4,
            SkillLevel.MASTER: 5
        }
        
        return level_values[skill.level] >= level_values[required_level]
    
    def _get_role_priority(self, roles: List[TeamRole]) -> List[TeamRole]:
        """Get roles sorted by priority for assignment"""
        priority_order = [
            TeamRole.CRISIS_LEADER,
            TeamRole.TECHNICAL_LEAD,
            TeamRole.SECURITY_LEAD,
            TeamRole.COMMUNICATIONS_LEAD,
            TeamRole.LEGAL_ADVISOR,
            TeamRole.OPERATIONS_LEAD,
            TeamRole.CUSTOMER_LIAISON,
            TeamRole.EXECUTIVE_LIAISON,
            TeamRole.RESOURCE_COORDINATOR,
            TeamRole.DOCUMENTATION_LEAD
        ]
        
        # Return roles in priority order, only including those that are required
        return [role for role in priority_order if role in roles]
    
    def _get_leadership_score(self, person_id: str) -> float:
        """Calculate leadership score for person"""
        person = self.personnel_registry.get(person_id)
        if not person:
            return 0.0
        
        leadership_skill = self._find_person_skill(person, "leadership")
        crisis_experience = person.crisis_experience.get("crisis_leader", 0)
        
        skill_score = 0.0
        if leadership_skill:
            level_values = {
                SkillLevel.BEGINNER: 0.2,
                SkillLevel.INTERMEDIATE: 0.4,
                SkillLevel.ADVANCED: 0.6,
                SkillLevel.EXPERT: 0.8,
                SkillLevel.MASTER: 1.0
            }
            skill_score = level_values[leadership_skill.level]
        
        experience_score = min(1.0, crisis_experience / 10.0)
        
        return (skill_score * 0.7) + (experience_score * 0.3)
    
    def _generate_match_rationale(
        self,
        person: Person,
        role: TeamRole,
        skill_score: float,
        experience_score: float,
        availability_score: float,
        strengths: List[str],
        missing_skills: List[str]
    ) -> str:
        """Generate human-readable rationale for role assignment"""
        rationale_parts = []
        
        if skill_score > 0.8:
            rationale_parts.append(f"Strong skill match ({skill_score:.2f})")
        elif skill_score > 0.6:
            rationale_parts.append(f"Good skill match ({skill_score:.2f})")
        else:
            rationale_parts.append(f"Adequate skill match ({skill_score:.2f})")
        
        if experience_score > 0.5:
            rationale_parts.append(f"relevant experience")
        
        if availability_score > 0.8:
            rationale_parts.append(f"high availability")
        
        if strengths:
            rationale_parts.append(f"strengths in {', '.join(strengths[:2])}")
        
        if missing_skills:
            rationale_parts.append(f"needs development in {', '.join(missing_skills[:2])}")
        
        return f"{person.name} selected for {role.value}: " + ", ".join(rationale_parts)
    
    def _initialize_skill_requirements(self) -> Dict[str, Any]:
        """Initialize skill requirements database"""
        return {
            "crisis_types": {},
            "role_skills": {},
            "skill_levels": {}
        }
    
    def _initialize_team_composition_rules(self) -> Dict[str, Any]:
        """Initialize team composition rules"""
        return {
            "size_constraints": {},
            "role_dependencies": {},
            "optimization_weights": {}
        }
    
    # Personnel management methods
    def add_person(self, person: Person) -> None:
        """Add person to personnel registry"""
        self.personnel_registry[person.id] = person
        logger.info(f"Added person {person.name} to personnel registry")
    
    def update_person_availability(self, person_id: str, status: AvailabilityStatus) -> None:
        """Update person's availability status"""
        if person_id in self.personnel_registry:
            self.personnel_registry[person_id].availability_status = status
            self.personnel_registry[person_id].updated_at = datetime.utcnow()
    
    def get_team_by_id(self, team_id: str) -> Optional[CrisisTeam]:
        """Get crisis team by ID"""
        return self.active_teams.get(team_id)
    
    def deactivate_team(self, team_id: str) -> None:
        """Deactivate crisis team"""
        if team_id in self.active_teams:
            self.active_teams[team_id].team_status = "disbanded"
            self.active_teams[team_id].deactivation_time = datetime.utcnow()