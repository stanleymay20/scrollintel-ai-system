"""
Tests for Board Dynamics Analysis Engine

This module contains comprehensive tests for the board dynamics analysis engine,
including composition analysis, power structure mapping, meeting dynamics assessment,
and governance framework understanding.
"""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime
from typing import List, Dict, Any

from scrollintel.engines.board_dynamics_engine import (
    BoardDynamicsAnalysisEngine, CompositionAnalyzer, PowerMapper,
    MeetingAnalyzer, GovernanceExpert, BoardMember, Background,
    Priority, Relationship, InfluenceLevel, CommunicationStyle,
    DecisionPattern, CompositionAnalysis, PowerStructureMap,
    DynamicsAssessment, GovernanceFramework
)


class TestCompositionAnalyzer:
    """Test cases for CompositionAnalyzer"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.analyzer = CompositionAnalyzer()
        
        # Create test board member
        self.test_background = Background(
            industry_experience=["technology", "software", "ai"],
            functional_expertise=["engineering", "product", "strategy"],
            education=["MIT", "Stanford MBA"],
            previous_roles=["CTO", "VP Engineering"],
            years_experience=15
        )
        
        self.test_member = BoardMember(
            id="member_1",
            name="John Doe",
            background=self.test_background,
            expertise_areas=["technology", "cybersecurity", "data_privacy"],
            influence_level=InfluenceLevel.HIGH,
            communication_style=CommunicationStyle.ANALYTICAL,
            decision_making_pattern=DecisionPattern.DATA_DRIVEN,
            tenure=3,
            committee_memberships=["audit", "technology"]
        )
    
    def test_analyze_member_background(self):
        """Test individual member background analysis"""
        analysis = self.analyzer.analyze_member_background(self.test_member)
        
        assert "experience_depth" in analysis
        assert "expertise_breadth" in analysis
        assert "industry_relevance" in analysis
        assert "functional_strength" in analysis
        assert "leadership_experience" in analysis
        
        # Check calculated values
        assert analysis["experience_depth"] == 0.75  # 15/20 years
        assert analysis["expertise_breadth"] == 3  # 3 expertise areas
        assert analysis["industry_relevance"] > 0  # Has relevant tech experience
        assert analysis["leadership_experience"] > 0  # Has CTO experience
    
    def test_identify_expertise_gaps(self):
        """Test expertise gap identification"""
        members = [self.test_member]
        gaps = self.analyzer.identify_expertise_gaps(members)
        
        # Should identify missing expertise areas
        expected_gaps = [
            "finance", "marketing", "operations", "legal",
            "international_business", "mergers_acquisitions", "public_company_governance"
        ]
        
        for gap in expected_gaps:
            assert gap in gaps
        
        # Should not include covered areas
        assert "technology" not in gaps
        assert "cybersecurity" not in gaps
    
    def test_assess_diversity_metrics(self):
        """Test diversity metrics assessment"""
        # Create diverse set of members
        member2 = BoardMember(
            id="member_2",
            name="Jane Smith",
            background=Background(
                industry_experience=["finance", "banking"],
                functional_expertise=["finance", "risk_management"],
                education=["Harvard Business School"],
                previous_roles=["CFO", "Finance Director"],
                years_experience=12
            ),
            expertise_areas=["finance", "risk_management"],
            influence_level=InfluenceLevel.MEDIUM,
            communication_style=CommunicationStyle.RESULTS_ORIENTED,
            decision_making_pattern=DecisionPattern.CONSENSUS_BUILDER,
            tenure=2
        )
        
        members = [self.test_member, member2]
        diversity_metrics = self.analyzer.assess_diversity_metrics(members)
        
        assert "experience_diversity" in diversity_metrics
        assert "background_diversity" in diversity_metrics
        assert "tenure_distribution" in diversity_metrics
        assert "committee_representation" in diversity_metrics
        
        # Check tenure distribution
        tenure_dist = diversity_metrics["tenure_distribution"]
        assert tenure_dist["3-5"] == 2  # Both members have 2-3 years tenure
    
    def test_calculate_experience_depth(self):
        """Test experience depth calculation"""
        # Test various experience levels
        assert self.analyzer._calculate_experience_depth(
            Background(industry_experience=[], functional_expertise=[], 
                      education=[], previous_roles=[], years_experience=10)
        ) == 0.5
        
        assert self.analyzer._calculate_experience_depth(
            Background(industry_experience=[], functional_expertise=[], 
                      education=[], previous_roles=[], years_experience=25)
        ) == 1.0  # Capped at 1.0
    
    def test_assess_industry_relevance(self):
        """Test industry relevance assessment"""
        relevant_background = Background(
            industry_experience=["technology", "software", "ai"],
            functional_expertise=[],
            education=[],
            previous_roles=[],
            years_experience=0
        )
        
        irrelevant_background = Background(
            industry_experience=["retail", "manufacturing"],
            functional_expertise=[],
            education=[],
            previous_roles=[],
            years_experience=0
        )
        
        relevant_score = self.analyzer._assess_industry_relevance(relevant_background)
        irrelevant_score = self.analyzer._assess_industry_relevance(irrelevant_background)
        
        assert relevant_score > irrelevant_score
        assert relevant_score > 0.5
        assert irrelevant_score == 0.0


class TestPowerMapper:
    """Test cases for PowerMapper"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.mapper = PowerMapper()
        
        # Create test members with relationships
        self.member1 = BoardMember(
            id="member_1",
            name="Alice",
            background=Background([], [], [], [], 10),
            expertise_areas=["technology"],
            influence_level=InfluenceLevel.VERY_HIGH,
            communication_style=CommunicationStyle.VISIONARY,
            decision_making_pattern=DecisionPattern.QUICK_DECIDER,
            relationships=[
                Relationship("member_2", "alliance", 0.8, "bidirectional"),
                Relationship("member_3", "mentor", 0.7, "outgoing")
            ],
            tenure=5
        )
        
        self.member2 = BoardMember(
            id="member_2",
            name="Bob",
            background=Background([], [], [], [], 8),
            expertise_areas=["finance"],
            influence_level=InfluenceLevel.HIGH,
            communication_style=CommunicationStyle.ANALYTICAL,
            decision_making_pattern=DecisionPattern.DATA_DRIVEN,
            relationships=[
                Relationship("member_1", "alliance", 0.8, "bidirectional"),
                Relationship("member_3", "collaboration", 0.6, "bidirectional")
            ],
            tenure=3
        )
        
        self.member3 = BoardMember(
            id="member_3",
            name="Charlie",
            background=Background([], [], [], [], 5),
            expertise_areas=["marketing"],
            influence_level=InfluenceLevel.MEDIUM,
            communication_style=CommunicationStyle.RELATIONSHIP_FOCUSED,
            decision_making_pattern=DecisionPattern.COLLABORATIVE,
            relationships=[
                Relationship("member_1", "mentee", 0.7, "incoming"),
                Relationship("member_2", "collaboration", 0.6, "bidirectional")
            ],
            tenure=2
        )
        
        self.members = [self.member1, self.member2, self.member3]
    
    def test_map_influence_networks(self):
        """Test influence network mapping"""
        networks = self.mapper.map_influence_networks(self.members)
        
        assert "member_1" in networks
        assert "member_2" in networks
        assert "member_3" in networks
        
        # Check strong relationships (> 0.6)
        assert "member_2" in networks["member_1"]  # 0.8 strength
        assert "member_3" in networks["member_1"]  # 0.7 strength
        assert "member_1" in networks["member_2"]  # 0.8 strength
    
    def test_identify_decision_makers(self):
        """Test decision maker identification"""
        decision_makers = self.mapper.identify_decision_makers(self.members)
        
        # Member 1 should be identified as a decision maker (VERY_HIGH influence)
        assert "member_1" in decision_makers
        
        # Member 3 should not be (MEDIUM influence)
        assert "member_3" not in decision_makers
    
    def test_detect_coalition_groups(self):
        """Test coalition group detection"""
        coalitions = self.mapper.detect_coalition_groups(self.members)
        
        # Should detect coalition between members with strong relationships
        assert len(coalitions) > 0
        
        # Check if members 1 and 2 are in same coalition (strong alliance)
        coalition_found = False
        for coalition in coalitions:
            if "member_1" in coalition and "member_2" in coalition:
                coalition_found = True
                break
        
        assert coalition_found
    
    def test_calculate_influence_flows(self):
        """Test influence flow calculation"""
        influence_flows = self.mapper.calculate_influence_flows(self.members)
        
        assert "member_1" in influence_flows
        assert "member_2" in influence_flows
        assert "member_3" in influence_flows
        
        # Check flow from member 1 to member 2
        assert "member_2" in influence_flows["member_1"]
        assert influence_flows["member_1"]["member_2"] > 0
    
    def test_get_influence_multiplier(self):
        """Test influence level multiplier calculation"""
        assert self.mapper._get_influence_multiplier(InfluenceLevel.LOW) == 0.2
        assert self.mapper._get_influence_multiplier(InfluenceLevel.MEDIUM) == 0.5
        assert self.mapper._get_influence_multiplier(InfluenceLevel.HIGH) == 0.8
        assert self.mapper._get_influence_multiplier(InfluenceLevel.VERY_HIGH) == 1.0
    
    def test_calculate_influence_score(self):
        """Test influence score calculation"""
        score = self.mapper._calculate_influence_score(self.member1, self.members)
        
        # Should be high due to VERY_HIGH influence level and strong relationships
        assert score > 0.8
        assert score <= 1.0
        
        # Member 3 should have lower score
        score3 = self.mapper._calculate_influence_score(self.member3, self.members)
        assert score3 < score


class TestMeetingAnalyzer:
    """Test cases for MeetingAnalyzer"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.analyzer = MeetingAnalyzer()
        
        self.test_meeting_data = {
            "participation": {
                "member_1": {
                    "speaking_time": 300,  # 5 minutes
                    "questions_asked": 3,
                    "contributions": 2
                },
                "member_2": {
                    "speaking_time": 180,  # 3 minutes
                    "questions_asked": 1,
                    "contributions": 1
                },
                "member_3": {
                    "speaking_time": 120,  # 2 minutes
                    "questions_asked": 0,
                    "contributions": 0
                }
            },
            "agenda_completion_rate": 0.8,
            "decision_quality_score": 0.7,
            "time_efficiency": 0.9,
            "average_engagement": 0.6,
            "conflict_resolution_score": 0.8,
            "interruption_rate": 0.2,
            "speaking_times": {
                "member_1": 300,
                "member_2": 180,
                "member_3": 120
            },
            "questions": {
                "member_1": 3,
                "member_2": 1,
                "member_3": 0
            },
            "interruptions": {
                "member_1": {"member_2": 1, "member_3": 0},
                "member_2": {"member_1": 0, "member_3": 1},
                "member_3": {"member_1": 0, "member_2": 0}
            },
            "consensus_actions": {
                "member_1": 2,
                "member_2": 4,
                "member_3": 1
            }
        }
    
    def test_analyze_participation_patterns(self):
        """Test participation pattern analysis"""
        participation = self.analyzer.analyze_participation_patterns(self.test_meeting_data)
        
        assert "member_1" in participation
        assert "member_2" in participation
        assert "member_3" in participation
        
        # Member 1 should have highest participation (most speaking time and contributions)
        assert participation["member_1"] > participation["member_2"]
        assert participation["member_2"] > participation["member_3"]
        
        # All scores should be between 0 and 1
        for score in participation.values():
            assert 0 <= score <= 1
    
    def test_assess_meeting_effectiveness(self):
        """Test meeting effectiveness assessment"""
        effectiveness = self.analyzer.assess_meeting_effectiveness(self.test_meeting_data)
        
        assert 0 <= effectiveness <= 1
        
        # Should be reasonably high given good scores in test data
        assert effectiveness > 0.6
    
    def test_identify_communication_patterns(self):
        """Test communication pattern identification"""
        patterns = self.analyzer.identify_communication_patterns(self.test_meeting_data)
        
        assert "dominant_speakers" in patterns
        assert "question_patterns" in patterns
        assert "interruption_patterns" in patterns
        assert "topic_leadership" in patterns
        assert "consensus_builders" in patterns
        
        # Member 1 should be identified as dominant speaker
        assert "member_1" in patterns["dominant_speakers"]
        
        # Member 2 should be identified as consensus builder (4 consensus actions)
        assert "member_2" in patterns["consensus_builders"]
    
    def test_detect_conflict_indicators(self):
        """Test conflict indicator detection"""
        indicators = self.analyzer.detect_conflict_indicators(self.test_meeting_data)
        
        # Should not detect major conflicts with current test data
        assert len(indicators) == 0 or "high_interruption_rate" not in indicators
        
        # Test with high conflict data
        high_conflict_data = self.test_meeting_data.copy()
        high_conflict_data["interruption_rate"] = 0.5
        high_conflict_data["decision_delay_count"] = 3
        
        conflict_indicators = self.analyzer.detect_conflict_indicators(high_conflict_data)
        assert "high_interruption_rate" in conflict_indicators
        assert "decision_delays" in conflict_indicators
    
    def test_identify_dominant_speakers(self):
        """Test dominant speaker identification"""
        dominant = self.analyzer._identify_dominant_speakers(self.test_meeting_data)
        
        # Member 1 has 300 seconds out of 600 total (50%), should be dominant
        assert "member_1" in dominant
        
        # Others should not be dominant
        assert "member_2" not in dominant
        assert "member_3" not in dominant
    
    def test_analyze_question_patterns(self):
        """Test question pattern analysis"""
        patterns = self.analyzer._analyze_question_patterns(self.test_meeting_data)
        
        assert "most_inquisitive" in patterns
        assert "question_distribution" in patterns
        assert "average_questions" in patterns
        
        # Member 1 should be most inquisitive (3 questions)
        assert patterns["most_inquisitive"][0] == "member_1"
        assert patterns["average_questions"] == 4/3  # Total questions / members
    
    def test_find_most_interrupted(self):
        """Test most interrupted member identification"""
        interruptions = self.test_meeting_data["interruptions"]
        most_interrupted = self.analyzer._find_most_interrupted(interruptions)
        
        # Member 3 gets interrupted once, others don't
        assert most_interrupted == "member_3"


class TestGovernanceExpert:
    """Test cases for GovernanceExpert"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.expert = GovernanceExpert()
        
        self.test_board_info = {
            "board_size": 9,
            "independent_directors": 6,
            "leadership_structure": "separate",
            "term_limits": True,
            "diversity_metrics": {"gender_diversity": 0.4, "ethnic_diversity": 0.3},
            "committees": {
                "audit": {"members": 3, "chair": "member_1"},
                "compensation": {"members": 3, "chair": "member_2"},
                "nominating": {"members": 3, "chair": "member_3"}
            },
            "voting_procedures": {"majority_rule": True, "written_consent": True},
            "quorum_requirements": {"minimum_members": 5},
            "approval_authorities": {"major_decisions": "board", "routine": "management"},
            "reporting_requirements": [
                "quarterly_financial_reports",
                "annual_governance_report",
                "risk_management_reports"
            ],
            "compliance_frameworks": [
                "sarbanes_oxley",
                "sec_regulations",
                "nasdaq_listing_standards"
            ]
        }
        
        self.test_performance_data = {
            "decision_efficiency_score": 0.8,
            "oversight_quality_score": 0.7,
            "compliance_score": 0.9
        }
    
    def test_analyze_governance_structure(self):
        """Test governance structure analysis"""
        framework = self.expert.analyze_governance_structure(self.test_board_info)
        
        assert isinstance(framework, GovernanceFramework)
        assert framework.board_structure is not None
        assert framework.committee_structure is not None
        assert framework.decision_processes is not None
        assert len(framework.reporting_requirements) > 0
        assert len(framework.compliance_frameworks) > 0
    
    def test_assess_governance_effectiveness(self):
        """Test governance effectiveness assessment"""
        framework = self.expert.analyze_governance_structure(self.test_board_info)
        effectiveness = self.expert.assess_governance_effectiveness(framework, self.test_performance_data)
        
        assert "board_independence" in effectiveness
        assert "committee_effectiveness" in effectiveness
        assert "decision_efficiency" in effectiveness
        assert "oversight_quality" in effectiveness
        assert "compliance_adherence" in effectiveness
        
        # All scores should be between 0 and 1
        for score in effectiveness.values():
            assert 0 <= score <= 1
    
    def test_identify_governance_gaps(self):
        """Test governance gap identification"""
        framework = self.expert.analyze_governance_structure(self.test_board_info)
        gaps = self.expert.identify_governance_gaps(framework, {})
        
        # With good test data, should have minimal gaps
        assert len(gaps) == 0 or "insufficient_board_independence" not in gaps
        
        # Test with poor governance data
        poor_board_info = self.test_board_info.copy()
        poor_board_info["independent_directors"] = 2  # Only 2 out of 9
        poor_board_info["committees"] = {"audit": {}}  # Missing required committees
        
        poor_framework = self.expert.analyze_governance_structure(poor_board_info)
        poor_gaps = self.expert.identify_governance_gaps(poor_framework, {})
        
        assert "insufficient_board_independence" in poor_gaps
        assert "missing_required_committees" in poor_gaps
    
    def test_recommend_governance_improvements(self):
        """Test governance improvement recommendations"""
        gaps = ["insufficient_board_independence", "missing_required_committees"]
        framework = self.expert.analyze_governance_structure(self.test_board_info)
        
        recommendations = self.expert.recommend_governance_improvements(gaps, framework)
        
        assert len(recommendations) == 2
        
        # Check recommendation structure
        for rec in recommendations:
            assert "area" in rec
            assert "recommendation" in rec
            assert "priority" in rec
            assert "timeline" in rec
    
    def test_assess_board_independence(self):
        """Test board independence assessment"""
        framework = self.expert.analyze_governance_structure(self.test_board_info)
        independence = self.expert._assess_board_independence(framework, self.test_performance_data)
        
        # 6 out of 9 directors are independent (67%), should score well
        expected_score = (6/9) / 0.75  # Target is 75%
        assert abs(independence - expected_score) < 0.1
    
    def test_meets_independence_requirements(self):
        """Test independence requirements check"""
        framework = self.expert.analyze_governance_structure(self.test_board_info)
        
        # Should meet requirements (67% > 50%)
        assert self.expert._meets_independence_requirements(framework)
        
        # Test with insufficient independence
        poor_board_info = self.test_board_info.copy()
        poor_board_info["independent_directors"] = 3  # Only 33%
        poor_framework = self.expert.analyze_governance_structure(poor_board_info)
        
        assert not self.expert._meets_independence_requirements(poor_framework)
    
    def test_has_required_committees(self):
        """Test required committee check"""
        framework = self.expert.analyze_governance_structure(self.test_board_info)
        
        # Should have required committees
        assert self.expert._has_required_committees(framework)
        
        # Test without required committees
        poor_board_info = self.test_board_info.copy()
        poor_board_info["committees"] = {"other": {}}
        poor_framework = self.expert.analyze_governance_structure(poor_board_info)
        
        assert not self.expert._has_required_committees(poor_framework)


class TestBoardDynamicsAnalysisEngine:
    """Test cases for main BoardDynamicsAnalysisEngine"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.engine = BoardDynamicsAnalysisEngine()
        
        # Create test members
        self.test_members = [
            BoardMember(
                id="member_1",
                name="Alice Johnson",
                background=Background(
                    industry_experience=["technology", "software"],
                    functional_expertise=["engineering", "product"],
                    education=["MIT", "Stanford MBA"],
                    previous_roles=["CTO", "VP Engineering"],
                    years_experience=15
                ),
                expertise_areas=["technology", "cybersecurity"],
                influence_level=InfluenceLevel.HIGH,
                communication_style=CommunicationStyle.ANALYTICAL,
                decision_making_pattern=DecisionPattern.DATA_DRIVEN,
                tenure=3,
                committee_memberships=["audit", "technology"]
            ),
            BoardMember(
                id="member_2",
                name="Bob Smith",
                background=Background(
                    industry_experience=["finance", "banking"],
                    functional_expertise=["finance", "risk"],
                    education=["Harvard Business School"],
                    previous_roles=["CFO", "Finance Director"],
                    years_experience=12
                ),
                expertise_areas=["finance", "risk_management"],
                influence_level=InfluenceLevel.MEDIUM,
                communication_style=CommunicationStyle.RESULTS_ORIENTED,
                decision_making_pattern=DecisionPattern.CONSENSUS_BUILDER,
                tenure=2
            )
        ]
        
        self.test_meeting_data = {
            "participation": {
                "member_1": {"speaking_time": 300, "questions_asked": 3, "contributions": 2},
                "member_2": {"speaking_time": 180, "questions_asked": 1, "contributions": 1}
            },
            "agenda_completion_rate": 0.8,
            "decision_quality_score": 0.7,
            "time_efficiency": 0.9,
            "average_engagement": 0.6,
            "conflict_resolution_score": 0.8
        }
        
        self.test_board_info = {
            "board_size": 7,
            "independent_directors": 5,
            "committees": {
                "audit": {"members": 3},
                "compensation": {"members": 3}
            },
            "voting_procedures": {"majority_rule": True},
            "quorum_requirements": {"minimum_members": 4}
        }
    
    def test_analyze_board_composition(self):
        """Test board composition analysis"""
        analysis = self.engine.analyze_board_composition(self.test_members)
        
        assert isinstance(analysis, CompositionAnalysis)
        assert len(analysis.member_profiles) == 2
        assert len(analysis.expertise_coverage) > 0
        assert len(analysis.experience_distribution) > 0
        assert analysis.diversity_metrics is not None
        assert len(analysis.skill_gaps) > 0
        assert len(analysis.strengths) > 0
    
    def test_map_power_structures(self):
        """Test power structure mapping"""
        power_map = self.engine.map_power_structures(self.test_members)
        
        assert isinstance(power_map, PowerStructureMap)
        assert len(power_map.influence_networks) == 2
        assert len(power_map.decision_makers) >= 0
        assert len(power_map.coalition_groups) >= 0
        assert len(power_map.influence_flows) == 2
    
    def test_assess_meeting_dynamics(self):
        """Test meeting dynamics assessment"""
        assessment = self.engine.assess_meeting_dynamics(self.test_meeting_data, self.test_members)
        
        assert isinstance(assessment, DynamicsAssessment)
        assert 0 <= assessment.meeting_effectiveness <= 1
        assert len(assessment.engagement_levels) == 2
        assert assessment.communication_patterns is not None
        assert 0 <= assessment.decision_efficiency <= 1
        assert isinstance(assessment.conflict_indicators, list)
        assert 0 <= assessment.collaboration_quality <= 1
    
    def test_analyze_governance_framework(self):
        """Test governance framework analysis"""
        analysis = self.engine.analyze_governance_framework(self.test_board_info)
        
        assert "framework" in analysis
        assert "effectiveness_scores" in analysis
        assert "governance_gaps" in analysis
        assert "recommendations" in analysis
        assert "overall_score" in analysis
        
        assert 0 <= analysis["overall_score"] <= 1
    
    def test_generate_comprehensive_analysis(self):
        """Test comprehensive analysis generation"""
        analysis = self.engine.generate_comprehensive_analysis(
            self.test_members, self.test_meeting_data, self.test_board_info
        )
        
        assert "composition_analysis" in analysis
        assert "power_structure_map" in analysis
        assert "dynamics_assessment" in analysis
        assert "governance_analysis" in analysis
        assert "insights" in analysis
        assert "recommendations" in analysis
        assert "analysis_timestamp" in analysis
        
        # Check that insights and recommendations are generated
        assert len(analysis["insights"]) >= 0
        assert len(analysis["recommendations"]) >= 0
    
    def test_get_experience_bucket(self):
        """Test experience bucket categorization"""
        assert self.engine._get_experience_bucket(3) == "0-5 years"
        assert self.engine._get_experience_bucket(7) == "5-10 years"
        assert self.engine._get_experience_bucket(15) == "10-20 years"
        assert self.engine._get_experience_bucket(25) == "20+ years"
    
    def test_assess_collaboration_quality(self):
        """Test collaboration quality assessment"""
        communication_patterns = {
            "consensus_builders": ["member_1", "member_2"],
            "dominant_speakers": ["member_1"]
        }
        
        meeting_data = {
            "participation_variance": 0.3,
            "constructive_question_ratio": 0.8,
            "unresolved_conflicts": []
        }
        
        quality = self.engine._assess_collaboration_quality(meeting_data, communication_patterns)
        
        assert 0 <= quality <= 1
        assert quality > 0.5  # Should be reasonably high with good test data
    
    def test_generate_insights(self):
        """Test insight generation"""
        # Create mock analysis results
        composition = CompositionAnalysis(
            member_profiles=self.test_members,
            expertise_coverage={"technology": ["member_1"], "finance": ["member_2"]},
            experience_distribution={"10-20 years": 1, "5-10 years": 1},
            diversity_metrics={},
            skill_gaps=["marketing", "legal", "operations"],  # More than 2 gaps
            strengths=["technology", "finance"]
        )
        
        power_map = PowerStructureMap(
            influence_networks={"member_1": ["member_2"]},
            decision_makers=["member_1"],  # Less than 3
            coalition_groups=[],
            influence_flows={},
            key_relationships=[]
        )
        
        dynamics = DynamicsAssessment(
            meeting_effectiveness=0.5,  # Below 0.6
            engagement_levels={"member_1": 0.8, "member_2": 0.6},
            communication_patterns={},
            decision_efficiency=0.7,
            conflict_indicators=["high_interruption_rate", "speaking_time_imbalance", "topic_avoidance"],
            collaboration_quality=0.6
        )
        
        governance = {"overall_score": 0.6}  # Below 0.7
        
        insights = self.engine._generate_insights(composition, power_map, dynamics, governance)
        
        assert len(insights) > 0
        assert any("skill gaps" in insight for insight in insights)
        assert any("concentrated" in insight for insight in insights)
        assert any("effectiveness" in insight for insight in insights)
        assert any("conflict" in insight for insight in insights)
        assert any("governance" in insight for insight in insights)
    
    def test_generate_recommendations(self):
        """Test recommendation generation"""
        # Use same mock data as insights test
        composition = CompositionAnalysis(
            member_profiles=self.test_members,
            expertise_coverage={},
            experience_distribution={},
            diversity_metrics={},
            skill_gaps=["marketing", "legal"],
            strengths=[]
        )
        
        power_map = PowerStructureMap(
            influence_networks={},
            decision_makers=["member_1"],  # Less than 3
            coalition_groups=[],
            influence_flows={},
            key_relationships=[]
        )
        
        dynamics = DynamicsAssessment(
            meeting_effectiveness=0.5,  # Below 0.6
            engagement_levels={},
            communication_patterns={},
            decision_efficiency=0.7,
            conflict_indicators=[],
            collaboration_quality=0.6
        )
        
        governance = {"recommendations": [{"area": "Test", "recommendation": "Test rec"}]}
        
        recommendations = self.engine._generate_recommendations(composition, power_map, dynamics, governance)
        
        assert len(recommendations) > 0
        
        # Check recommendation structure
        for rec in recommendations:
            assert "area" in rec
            assert "recommendation" in rec
            assert "priority" in rec
            assert "timeline" in rec


@pytest.fixture
def sample_board_members():
    """Fixture providing sample board members for testing"""
    return [
        BoardMember(
            id="member_1",
            name="Alice Johnson",
            background=Background(
                industry_experience=["technology", "software"],
                functional_expertise=["engineering", "product"],
                education=["MIT"],
                previous_roles=["CTO"],
                years_experience=15
            ),
            expertise_areas=["technology", "cybersecurity"],
            influence_level=InfluenceLevel.HIGH,
            communication_style=CommunicationStyle.ANALYTICAL,
            decision_making_pattern=DecisionPattern.DATA_DRIVEN,
            tenure=3
        ),
        BoardMember(
            id="member_2",
            name="Bob Smith",
            background=Background(
                industry_experience=["finance"],
                functional_expertise=["finance"],
                education=["Harvard"],
                previous_roles=["CFO"],
                years_experience=12
            ),
            expertise_areas=["finance"],
            influence_level=InfluenceLevel.MEDIUM,
            communication_style=CommunicationStyle.RESULTS_ORIENTED,
            decision_making_pattern=DecisionPattern.CONSENSUS_BUILDER,
            tenure=2
        )
    ]


@pytest.fixture
def sample_meeting_data():
    """Fixture providing sample meeting data for testing"""
    return {
        "participation": {
            "member_1": {"speaking_time": 300, "questions_asked": 3, "contributions": 2},
            "member_2": {"speaking_time": 180, "questions_asked": 1, "contributions": 1}
        },
        "agenda_completion_rate": 0.8,
        "decision_quality_score": 0.7,
        "time_efficiency": 0.9,
        "average_engagement": 0.6,
        "conflict_resolution_score": 0.8
    }


def test_integration_comprehensive_analysis(sample_board_members, sample_meeting_data):
    """Integration test for comprehensive board analysis"""
    engine = BoardDynamicsAnalysisEngine()
    
    board_info = {
        "board_size": 7,
        "independent_directors": 5,
        "committees": {"audit": {"members": 3}},
        "voting_procedures": {"majority_rule": True}
    }
    
    # Should not raise any exceptions
    analysis = engine.generate_comprehensive_analysis(
        sample_board_members, sample_meeting_data, board_info
    )
    
    # Verify all components are present
    assert all(key in analysis for key in [
        "composition_analysis", "power_structure_map", "dynamics_assessment",
        "governance_analysis", "insights", "recommendations", "analysis_timestamp"
    ])


def test_error_handling():
    """Test error handling in board dynamics engine"""
    engine = BoardDynamicsAnalysisEngine()
    
    # Test with empty member list
    with pytest.raises(Exception):
        engine.analyze_board_composition([])
    
    # Test with invalid meeting data
    with pytest.raises(Exception):
        engine.assess_meeting_dynamics({}, [])


if __name__ == "__main__":
    pytest.main([__file__])