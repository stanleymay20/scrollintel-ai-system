"""
Crisis Leadership Guidance Engine
Provides best practices, decision support, and effectiveness assessment for crisis leadership
"""
import uuid
from datetime import datetime
from typing import List, Dict, Optional, Any
from ..models.leadership_guidance_models import (
    CrisisType, LeadershipStyle, DecisionUrgency,
    LeadershipBestPractice, DecisionContext, LeadershipRecommendation,
    LeadershipAssessment, CoachingGuidance
)

class LeadershipGuidanceEngine:
    def __init__(self):
        self.best_practices = self._initialize_best_practices()
        self.leadership_patterns = self._initialize_leadership_patterns()
        self.assessment_criteria = self._initialize_assessment_criteria()
    
    def get_leadership_guidance(self, context: DecisionContext) -> LeadershipRecommendation:
        """Provide leadership guidance based on crisis context"""
        try:
            # Analyze crisis context
            recommended_style = self._determine_leadership_style(context)
            
            # Get relevant best practices
            relevant_practices = self._get_relevant_practices(context.crisis_type)
            
            # Generate key actions
            key_actions = self._generate_key_actions(context, relevant_practices)
            
            # Develop communication strategy
            communication_strategy = self._develop_communication_strategy(context)
            
            # Prioritize stakeholders
            stakeholder_priorities = self._prioritize_stakeholders(context)
            
            # Identify risk mitigation steps
            risk_mitigation = self._identify_risk_mitigation(context)
            
            # Define success metrics
            success_metrics = self._define_success_metrics(context)
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(context)
            
            # Generate rationale
            rationale = self._generate_rationale(context, recommended_style)
            
            return LeadershipRecommendation(
                id=str(uuid.uuid4()),
                context=context,
                recommended_style=recommended_style,
                key_actions=key_actions,
                communication_strategy=communication_strategy,
                stakeholder_priorities=stakeholder_priorities,
                risk_mitigation_steps=risk_mitigation,
                success_metrics=success_metrics,
                confidence_score=confidence_score,
                rationale=rationale
            )
            
        except Exception as e:
            raise Exception(f"Failed to generate leadership guidance: {str(e)}")
    
    def assess_leadership_effectiveness(self, leader_id: str, crisis_id: str, 
                                     performance_data: Dict[str, Any]) -> LeadershipAssessment:
        """Assess leadership effectiveness during crisis"""
        try:
            # Calculate performance scores
            decision_quality = self._assess_decision_quality(performance_data)
            communication_effectiveness = self._assess_communication_effectiveness(performance_data)
            stakeholder_confidence = self._assess_stakeholder_confidence(performance_data)
            team_morale_impact = self._assess_team_morale_impact(performance_data)
            crisis_resolution_speed = self._assess_resolution_speed(performance_data)
            
            # Calculate overall effectiveness
            overall_effectiveness = (
                decision_quality * 0.25 +
                communication_effectiveness * 0.20 +
                stakeholder_confidence * 0.20 +
                team_morale_impact * 0.15 +
                crisis_resolution_speed * 0.20
            )
            
            # Identify strengths and improvement areas
            strengths = self._identify_strengths(performance_data)
            improvement_areas = self._identify_improvement_areas(performance_data)
            
            # Generate coaching recommendations
            coaching_recommendations = self._generate_coaching_recommendations(
                improvement_areas, performance_data
            )
            
            return LeadershipAssessment(
                leader_id=leader_id,
                crisis_id=crisis_id,
                assessment_time=datetime.now(),
                decision_quality_score=decision_quality,
                communication_effectiveness=communication_effectiveness,
                stakeholder_confidence=stakeholder_confidence,
                team_morale_impact=team_morale_impact,
                crisis_resolution_speed=crisis_resolution_speed,
                overall_effectiveness=overall_effectiveness,
                strengths=strengths,
                improvement_areas=improvement_areas,
                coaching_recommendations=coaching_recommendations
            )
            
        except Exception as e:
            raise Exception(f"Failed to assess leadership effectiveness: {str(e)}")
    
    def provide_coaching_guidance(self, assessment: LeadershipAssessment) -> List[CoachingGuidance]:
        """Provide detailed coaching guidance based on assessment"""
        try:
            coaching_guidance = []
            
            for area in assessment.improvement_areas:
                guidance = self._create_coaching_guidance(assessment, area)
                coaching_guidance.append(guidance)
            
            return coaching_guidance
            
        except Exception as e:
            raise Exception(f"Failed to provide coaching guidance: {str(e)}")
    
    def _determine_leadership_style(self, context: DecisionContext) -> LeadershipStyle:
        """Determine optimal leadership style for crisis context"""
        # High urgency and clear technical issues -> Directive
        if context.time_pressure == DecisionUrgency.IMMEDIATE:
            if context.crisis_type in [CrisisType.TECHNICAL_OUTAGE, CrisisType.SECURITY_BREACH]:
                return LeadershipStyle.DIRECTIVE
        
        # Complex stakeholder situations -> Collaborative
        if len(context.stakeholders_affected) > 5:
            return LeadershipStyle.COLLABORATIVE
        
        # Team morale issues -> Supportive
        if context.crisis_type == CrisisType.LEADERSHIP_CRISIS:
            return LeadershipStyle.SUPPORTIVE
        
        # Major organizational change needed -> Transformational
        if context.crisis_type in [CrisisType.FINANCIAL_CRISIS, CrisisType.MARKET_VOLATILITY]:
            return LeadershipStyle.TRANSFORMATIONAL
        
        # Default to adaptive for complex situations
        return LeadershipStyle.ADAPTIVE
    
    def _get_relevant_practices(self, crisis_type: CrisisType) -> List[LeadershipBestPractice]:
        """Get best practices relevant to crisis type"""
        return [practice for practice in self.best_practices 
                if practice.crisis_type == crisis_type]
    
    def _generate_key_actions(self, context: DecisionContext, 
                            practices: List[LeadershipBestPractice]) -> List[str]:
        """Generate key leadership actions based on context and best practices"""
        actions = []
        
        # Immediate response actions
        if context.time_pressure == DecisionUrgency.IMMEDIATE:
            actions.extend([
                "Activate crisis response team immediately",
                "Establish clear command structure",
                "Initiate stakeholder communication protocol"
            ])
        
        # Crisis-specific actions from best practices
        for practice in practices[:3]:  # Top 3 most relevant
            actions.extend(practice.implementation_steps[:2])  # Top 2 steps each
        
        return actions[:8]  # Limit to 8 key actions
    
    def _develop_communication_strategy(self, context: DecisionContext) -> str:
        """Develop communication strategy based on crisis context"""
        if context.crisis_type == CrisisType.SECURITY_BREACH:
            return "Transparent, security-focused communication with regular updates and clear remediation steps"
        elif context.crisis_type == CrisisType.TECHNICAL_OUTAGE:
            return "Frequent technical updates with clear timelines and impact mitigation measures"
        elif context.crisis_type == CrisisType.FINANCIAL_CRISIS:
            return "Confident, strategic communication focusing on stability and recovery plans"
        else:
            return "Clear, empathetic communication with focus on resolution and stakeholder support"
    
    def _prioritize_stakeholders(self, context: DecisionContext) -> List[str]:
        """Prioritize stakeholders based on crisis impact"""
        priority_map = {
            CrisisType.TECHNICAL_OUTAGE: ["customers", "technical_team", "executives", "partners"],
            CrisisType.SECURITY_BREACH: ["customers", "regulators", "executives", "security_team"],
            CrisisType.FINANCIAL_CRISIS: ["investors", "board", "employees", "customers"],
            CrisisType.REGULATORY_ISSUE: ["regulators", "legal_team", "executives", "customers"]
        }
        
        return priority_map.get(context.crisis_type, 
                              ["executives", "employees", "customers", "partners"])
    
    def _identify_risk_mitigation(self, context: DecisionContext) -> List[str]:
        """Identify risk mitigation steps"""
        mitigation_steps = [
            "Establish clear escalation procedures",
            "Implement regular status monitoring",
            "Prepare contingency plans for worst-case scenarios"
        ]
        
        if context.crisis_type == CrisisType.SECURITY_BREACH:
            mitigation_steps.extend([
                "Isolate affected systems immediately",
                "Engage cybersecurity experts",
                "Prepare legal compliance documentation"
            ])
        elif context.crisis_type == CrisisType.FINANCIAL_CRISIS:
            mitigation_steps.extend([
                "Secure emergency funding sources",
                "Implement cost reduction measures",
                "Engage financial advisors"
            ])
        
        return mitigation_steps
    
    def _define_success_metrics(self, context: DecisionContext) -> List[str]:
        """Define success metrics for crisis leadership"""
        base_metrics = [
            "Crisis resolution time",
            "Stakeholder satisfaction scores",
            "Team performance maintenance",
            "Communication effectiveness rating"
        ]
        
        crisis_specific = {
            CrisisType.TECHNICAL_OUTAGE: ["System uptime restoration", "Customer impact minimization"],
            CrisisType.SECURITY_BREACH: ["Security incident containment", "Data protection compliance"],
            CrisisType.FINANCIAL_CRISIS: ["Financial stability restoration", "Investor confidence maintenance"]
        }
        
        base_metrics.extend(crisis_specific.get(context.crisis_type, []))
        return base_metrics
    
    def _calculate_confidence_score(self, context: DecisionContext) -> float:
        """Calculate confidence score for recommendations"""
        base_confidence = 0.7
        
        # Adjust based on information availability
        info_completeness = len(context.available_information) / 10.0
        base_confidence += min(info_completeness * 0.2, 0.2)
        
        # Adjust based on crisis complexity
        if context.severity_level <= 3:
            base_confidence += 0.1
        elif context.severity_level >= 8:
            base_confidence -= 0.1
        
        return min(max(base_confidence, 0.0), 1.0)
    
    def _generate_rationale(self, context: DecisionContext, style: LeadershipStyle) -> str:
        """Generate rationale for leadership recommendations"""
        return f"Recommended {style.value} leadership approach based on {context.crisis_type.value} " \
               f"with {context.time_pressure.value} urgency and {len(context.stakeholders_affected)} " \
               f"affected stakeholders. This approach optimizes for rapid decision-making while " \
               f"maintaining stakeholder confidence and team effectiveness."
    
    def _assess_decision_quality(self, performance_data: Dict[str, Any]) -> float:
        """Assess quality of decisions made during crisis"""
        # Placeholder implementation - would integrate with actual decision tracking
        return performance_data.get('decision_quality', 0.75)
    
    def _assess_communication_effectiveness(self, performance_data: Dict[str, Any]) -> float:
        """Assess effectiveness of crisis communication"""
        return performance_data.get('communication_effectiveness', 0.80)
    
    def _assess_stakeholder_confidence(self, performance_data: Dict[str, Any]) -> float:
        """Assess stakeholder confidence levels"""
        return performance_data.get('stakeholder_confidence', 0.70)
    
    def _assess_team_morale_impact(self, performance_data: Dict[str, Any]) -> float:
        """Assess impact on team morale"""
        return performance_data.get('team_morale', 0.75)
    
    def _assess_resolution_speed(self, performance_data: Dict[str, Any]) -> float:
        """Assess speed of crisis resolution"""
        return performance_data.get('resolution_speed', 0.70)
    
    def _identify_strengths(self, performance_data: Dict[str, Any]) -> List[str]:
        """Identify leadership strengths"""
        strengths = []
        
        if performance_data.get('decision_quality', 0) > 0.8:
            strengths.append("Excellent decision-making under pressure")
        if performance_data.get('communication_effectiveness', 0) > 0.8:
            strengths.append("Clear and effective crisis communication")
        if performance_data.get('stakeholder_confidence', 0) > 0.8:
            strengths.append("Strong stakeholder relationship management")
        
        return strengths if strengths else ["Maintained team stability during crisis"]
    
    def _identify_improvement_areas(self, performance_data: Dict[str, Any]) -> List[str]:
        """Identify areas for improvement"""
        improvements = []
        
        if performance_data.get('decision_quality', 1) < 0.7:
            improvements.append("Decision-making speed and quality")
        if performance_data.get('communication_effectiveness', 1) < 0.7:
            improvements.append("Crisis communication clarity")
        if performance_data.get('stakeholder_confidence', 1) < 0.7:
            improvements.append("Stakeholder confidence management")
        
        return improvements if improvements else ["Proactive crisis preparation"]
    
    def _generate_coaching_recommendations(self, improvement_areas: List[str], 
                                         performance_data: Dict[str, Any]) -> List[str]:
        """Generate coaching recommendations"""
        recommendations = []
        
        for area in improvement_areas:
            if "decision-making" in area.lower():
                recommendations.append("Practice rapid decision-making frameworks")
            elif "communication" in area.lower():
                recommendations.append("Develop crisis communication templates and practice")
            elif "stakeholder" in area.lower():
                recommendations.append("Enhance stakeholder mapping and engagement strategies")
        
        return recommendations
    
    def _create_coaching_guidance(self, assessment: LeadershipAssessment, 
                                focus_area: str) -> CoachingGuidance:
        """Create detailed coaching guidance for specific area"""
        current_score = 0.6  # Default current performance
        target_score = 0.85  # Target performance
        
        # Map focus area to specific guidance
        guidance_map = {
            "Decision-making speed and quality": {
                "strategies": [
                    "Practice OODA loop decision framework",
                    "Develop decision trees for common scenarios",
                    "Use structured decision-making templates"
                ],
                "exercises": [
                    "Crisis simulation exercises",
                    "Rapid decision-making drills",
                    "Case study analysis"
                ]
            },
            "Crisis communication clarity": {
                "strategies": [
                    "Develop clear communication templates",
                    "Practice stakeholder-specific messaging",
                    "Use structured communication frameworks"
                ],
                "exercises": [
                    "Media interview practice",
                    "Stakeholder presentation drills",
                    "Crisis communication simulations"
                ]
            }
        }
        
        guidance_info = guidance_map.get(focus_area, {
            "strategies": ["Focus on systematic improvement"],
            "exercises": ["Regular practice and feedback"]
        })
        
        return CoachingGuidance(
            assessment_id=assessment.leader_id,
            focus_area=focus_area,
            current_performance=current_score,
            target_performance=target_score,
            improvement_strategies=guidance_info["strategies"],
            practice_exercises=guidance_info["exercises"],
            success_indicators=[
                "Improved performance metrics",
                "Positive stakeholder feedback",
                "Faster crisis resolution"
            ],
            timeline="4-6 weeks",
            resources=[
                "Crisis leadership training materials",
                "Simulation exercises",
                "Peer mentoring sessions"
            ]
        )
    
    def _initialize_best_practices(self) -> List[LeadershipBestPractice]:
        """Initialize leadership best practices database"""
        return [
            LeadershipBestPractice(
                id="bp_tech_outage_1",
                crisis_type=CrisisType.TECHNICAL_OUTAGE,
                practice_name="Rapid Technical Response",
                description="Immediate technical team mobilization and clear communication",
                implementation_steps=[
                    "Activate technical response team within 15 minutes",
                    "Establish clear command structure",
                    "Begin stakeholder communication immediately",
                    "Implement status page updates every 30 minutes"
                ],
                success_indicators=[
                    "Response team activated within target time",
                    "Clear communication established",
                    "Stakeholder satisfaction maintained"
                ],
                common_pitfalls=[
                    "Delayed team activation",
                    "Unclear communication",
                    "Inadequate stakeholder updates"
                ],
                effectiveness_score=0.85,
                applicable_scenarios=["System outages", "Service disruptions", "Performance issues"]
            ),
            LeadershipBestPractice(
                id="bp_security_breach_1",
                crisis_type=CrisisType.SECURITY_BREACH,
                practice_name="Security Incident Response",
                description="Immediate containment and transparent communication",
                implementation_steps=[
                    "Isolate affected systems immediately",
                    "Engage security team and external experts",
                    "Begin legal and regulatory compliance procedures",
                    "Prepare transparent customer communication"
                ],
                success_indicators=[
                    "Breach contained within 1 hour",
                    "Regulatory compliance maintained",
                    "Customer trust preserved"
                ],
                common_pitfalls=[
                    "Delayed containment",
                    "Inadequate legal preparation",
                    "Poor customer communication"
                ],
                effectiveness_score=0.90,
                applicable_scenarios=["Data breaches", "Cyber attacks", "Security vulnerabilities"]
            )
        ]
    
    def _initialize_leadership_patterns(self) -> Dict[str, Any]:
        """Initialize leadership pattern recognition"""
        return {
            "high_pressure_decisions": {
                "indicators": ["time_pressure", "stakeholder_impact", "resource_constraints"],
                "recommended_approach": "directive_with_consultation"
            },
            "complex_stakeholder_management": {
                "indicators": ["multiple_stakeholders", "conflicting_interests", "regulatory_involvement"],
                "recommended_approach": "collaborative_consensus_building"
            }
        }
    
    def _initialize_assessment_criteria(self) -> Dict[str, Dict[str, float]]:
        """Initialize assessment criteria and weights"""
        return {
            "decision_quality": {
                "speed": 0.3,
                "accuracy": 0.4,
                "stakeholder_impact": 0.3
            },
            "communication_effectiveness": {
                "clarity": 0.4,
                "timeliness": 0.3,
                "stakeholder_satisfaction": 0.3
            },
            "crisis_resolution": {
                "speed": 0.4,
                "completeness": 0.3,
                "stakeholder_satisfaction": 0.3
            }
        }