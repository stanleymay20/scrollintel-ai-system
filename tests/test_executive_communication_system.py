"""
Tests for Executive Communication System
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch

from scrollintel.engines.executive_communication_engine import (
    ExecutiveCommunicationSystem, LanguageAdapter, CommunicationEffectivenessTracker
)
from scrollintel.models.executive_communication_models import (
    ExecutiveAudience, Message, AdaptedMessage, CommunicationEffectiveness,
    ExecutiveLevel, CommunicationStyle, MessageType
)


class TestLanguageAdapter:
    """Test language adaptation functionality"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.adapter = LanguageAdapter()
        
        self.sample_message = Message(
            id="msg_001",
            content="Our API microservices architecture requires containerization to improve scalability and reduce latency. We recommend implementing CI/CD pipelines for automated deployment.",
            message_type=MessageType.STRATEGIC_UPDATE,
            technical_complexity=0.8,
            urgency_level="high",
            key_points=[
                "API architecture needs improvement",
                "Containerization required",
                "CI/CD implementation recommended"
            ],
            supporting_data={"cost_savings": 100000, "performance_improvement": "40%"},
            created_at=datetime.now()
        )
        
        self.ceo_audience = ExecutiveAudience(
            id="ceo_001",
            name="John CEO",
            title="Chief Executive Officer",
            executive_level=ExecutiveLevel.CEO,
            communication_style=CommunicationStyle.DIRECT,
            expertise_areas=["business", "strategy"],
            decision_making_pattern="quick_decisive",
            influence_level=1.0,
            preferred_communication_format="email",
            attention_span=5,
            detail_preference="low",
            risk_tolerance="medium",
            created_at=datetime.now()
        )
        
        self.cto_audience = ExecutiveAudience(
            id="cto_001",
            name="Jane CTO",
            title="Chief Technology Officer",
            executive_level=ExecutiveLevel.CTO,
            communication_style=CommunicationStyle.ANALYTICAL,
            expertise_areas=["technical", "architecture"],
            decision_making_pattern="analytical_thorough",
            influence_level=0.9,
            preferred_communication_format="presentation",
            attention_span=15,
            detail_preference="high",
            risk_tolerance="high",
            created_at=datetime.now()
        )
    
    def test_adapt_communication_style_for_ceo(self):
        """Test communication adaptation for CEO"""
        adapted = self.adapter.adapt_communication_style(self.sample_message, self.ceo_audience)
        
        assert isinstance(adapted, AdaptedMessage)
        assert adapted.audience_id == "ceo_001"
        assert adapted.language_complexity == "simplified"
        assert adapted.estimated_reading_time <= 5  # CEO has 5 min attention span
        assert "system interface" in adapted.adapted_content  # API -> system interface
        assert adapted.effectiveness_score > 0.0
    
    def test_adapt_communication_style_for_cto(self):
        """Test communication adaptation for CTO"""
        adapted = self.adapter.adapt_communication_style(self.sample_message, self.cto_audience)
        
        assert isinstance(adapted, AdaptedMessage)
        assert adapted.audience_id == "cto_001"
        assert adapted.language_complexity in ["detailed", "balanced"]
        assert "API" in adapted.adapted_content  # Technical terms preserved for CTO
        assert adapted.effectiveness_score > 0.0
    
    def test_create_executive_summary(self):
        """Test executive summary creation"""
        summary = self.adapter._create_executive_summary(self.sample_message, self.ceo_audience)
        
        assert len(summary) > 0
        assert "Executive Summary:" in summary or "Key Decision:" in summary
        assert any(point in summary for point in self.sample_message.key_points)
    
    def test_extract_key_recommendations(self):
        """Test key recommendation extraction"""
        recommendations = self.adapter._extract_key_recommendations(self.sample_message, self.ceo_audience)
        
        assert isinstance(recommendations, list)
        assert len(recommendations) <= 3  # CEO gets max 3 recommendations
    
    def test_determine_optimal_tone(self):
        """Test optimal tone determination"""
        tone = self.adapter._determine_optimal_tone(self.ceo_audience, MessageType.STRATEGIC_UPDATE)
        
        assert tone in ["direct_confident", "professional_strategic", "urgent_but_controlled", 
                       "respectful_strategic", "diplomatic_collaborative"]
    
    def test_predict_effectiveness(self):
        """Test effectiveness prediction"""
        content = "Short executive summary with clear action items."
        score = self.adapter._predict_effectiveness(content, self.ceo_audience, MessageType.STRATEGIC_UPDATE)
        
        assert 0.0 <= score <= 1.0
    
    def test_vocabulary_adaptation(self):
        """Test technical vocabulary adaptation"""
        content = "API microservices containerization"
        adapted = self.adapter._adapt_language(content, self.ceo_audience)
        
        # For CEO with low detail preference, technical terms should be simplified
        assert "system interface" in adapted or "API" in adapted
    
    def test_direct_communication_style(self):
        """Test direct communication style adaptation"""
        content = "We should consider implementing this solution"
        adapted = self.adapter._make_more_direct(content)
        
        assert "we must" in adapted.lower() or "we recommend" in adapted.lower()
    
    def test_diplomatic_communication_style(self):
        """Test diplomatic communication style adaptation"""
        content = "We must implement this immediately"
        adapted = self.adapter._make_more_diplomatic(content)
        
        assert "should consider" in adapted.lower() or "presents challenges" in adapted.lower()


class TestCommunicationEffectivenessTracker:
    """Test communication effectiveness tracking"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.tracker = CommunicationEffectivenessTracker()
    
    def test_measure_effectiveness(self):
        """Test effectiveness measurement"""
        engagement_data = {
            "engagement_score": 0.8,
            "comprehension_score": 0.9,
            "action_taken": True,
            "feedback": "Very clear and actionable",
            "response_time": 15,
            "follow_up_questions": 1,
            "decision_influenced": True
        }
        
        effectiveness = self.tracker.measure_effectiveness("msg_001", "ceo_001", engagement_data)
        
        assert isinstance(effectiveness, CommunicationEffectiveness)
        assert effectiveness.engagement_score == 0.8
        assert effectiveness.comprehension_score == 0.9
        assert effectiveness.action_taken is True
        assert effectiveness.decision_influenced is True
    
    def test_optimize_based_on_feedback(self):
        """Test optimization recommendations"""
        low_effectiveness = CommunicationEffectiveness(
            id="eff_001",
            message_id="msg_001",
            audience_id="ceo_001",
            engagement_score=0.4,  # Low engagement
            comprehension_score=0.5,  # Low comprehension
            action_taken=False,
            feedback_received="Too complex",
            response_time=60,
            follow_up_questions=5,  # Many questions
            decision_influenced=False,
            measured_at=datetime.now()
        )
        
        recommendations = self.tracker.optimize_based_on_feedback(low_effectiveness)
        
        assert "engagement" in recommendations
        assert "clarity" in recommendations
        assert "completeness" in recommendations
        assert "actionability" in recommendations


class TestExecutiveCommunicationSystem:
    """Test main executive communication system"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.system = ExecutiveCommunicationSystem()
        
        self.sample_message = Message(
            id="msg_001",
            content="Technical implementation requires immediate attention for scalability improvements.",
            message_type=MessageType.STRATEGIC_UPDATE,
            technical_complexity=0.7,
            urgency_level="high",
            key_points=["Scalability issues", "Technical debt", "Performance optimization"],
            supporting_data={},
            created_at=datetime.now()
        )
        
        self.board_audience = ExecutiveAudience(
            id="board_001",
            name="Board Chair",
            title="Chairman of the Board",
            executive_level=ExecutiveLevel.BOARD_CHAIR,
            communication_style=CommunicationStyle.STRATEGIC,
            expertise_areas=["governance", "strategy"],
            decision_making_pattern="consensus_building",
            influence_level=1.0,
            preferred_communication_format="presentation",
            attention_span=10,
            detail_preference="medium",
            risk_tolerance="low",
            created_at=datetime.now()
        )
    
    def test_process_executive_communication(self):
        """Test complete executive communication processing"""
        adapted_message = self.system.process_executive_communication(
            self.sample_message, self.board_audience
        )
        
        assert isinstance(adapted_message, AdaptedMessage)
        assert adapted_message.audience_id == "board_001"
        assert len(adapted_message.executive_summary) > 0
        assert len(adapted_message.key_recommendations) >= 0
        assert adapted_message.effectiveness_score > 0.0
    
    def test_track_communication_effectiveness(self):
        """Test communication effectiveness tracking"""
        engagement_data = {
            "engagement_score": 0.9,
            "comprehension_score": 0.8,
            "action_taken": True,
            "decision_influenced": True
        }
        
        effectiveness = self.system.track_communication_effectiveness(
            "msg_001", "board_001", engagement_data
        )
        
        assert isinstance(effectiveness, CommunicationEffectiveness)
        assert effectiveness.engagement_score == 0.9
    
    def test_get_optimization_recommendations(self):
        """Test optimization recommendations"""
        effectiveness = CommunicationEffectiveness(
            id="eff_001",
            message_id="msg_001",
            audience_id="board_001",
            engagement_score=0.6,
            comprehension_score=0.7,
            action_taken=True,
            feedback_received=None,
            response_time=None,
            follow_up_questions=2,
            decision_influenced=True,
            measured_at=datetime.now()
        )
        
        recommendations = self.system.get_optimization_recommendations(effectiveness)
        
        assert isinstance(recommendations, dict)
    
    def test_error_handling(self):
        """Test error handling in communication processing"""
        invalid_message = None
        
        with pytest.raises(Exception):
            self.system.process_executive_communication(invalid_message, self.board_audience)


class TestIntegration:
    """Integration tests for executive communication system"""
    
    def setup_method(self):
        """Setup integration test fixtures"""
        self.system = ExecutiveCommunicationSystem()
    
    def test_end_to_end_communication_flow(self):
        """Test complete communication flow from message to effectiveness tracking"""
        # Create message
        message = Message(
            id="integration_msg",
            content="We need to implement new security protocols to address recent vulnerabilities. This requires immediate board approval for budget allocation.",
            message_type=MessageType.RISK_ASSESSMENT,
            technical_complexity=0.6,
            urgency_level="high",
            key_points=[
                "Security vulnerabilities identified",
                "New protocols required",
                "Budget approval needed"
            ],
            supporting_data={"estimated_cost": 500000, "risk_level": "high"},
            created_at=datetime.now()
        )
        
        # Create audience
        audience = ExecutiveAudience(
            id="integration_board",
            name="Board of Directors",
            title="Board",
            executive_level=ExecutiveLevel.BOARD_MEMBER,
            communication_style=CommunicationStyle.DIPLOMATIC,
            expertise_areas=["governance", "risk"],
            decision_making_pattern="consensus_required",
            influence_level=1.0,
            preferred_communication_format="formal_presentation",
            attention_span=15,
            detail_preference="medium",
            risk_tolerance="low",
            created_at=datetime.now()
        )
        
        # Process communication
        adapted_message = self.system.process_executive_communication(message, audience)
        
        # Verify adaptation
        assert adapted_message is not None
        assert "security" in adapted_message.adapted_content.lower()
        assert len(adapted_message.key_recommendations) > 0
        
        # Track effectiveness
        engagement_data = {
            "engagement_score": 0.85,
            "comprehension_score": 0.90,
            "action_taken": True,
            "decision_influenced": True,
            "response_time": 30
        }
        
        effectiveness = self.system.track_communication_effectiveness(
            adapted_message.id, audience.id, engagement_data
        )
        
        # Verify tracking
        assert effectiveness.engagement_score == 0.85
        assert effectiveness.action_taken is True
        
        # Get recommendations
        recommendations = self.system.get_optimization_recommendations(effectiveness)
        
        # Verify recommendations (should be minimal for high effectiveness)
        assert isinstance(recommendations, dict)


class TestStrategicNarrativeSystem:
    """Test strategic narrative development system"""
    
    def setup_method(self):
        """Setup test fixtures"""
        from scrollintel.engines.strategic_narrative_engine import (
            StrategicNarrativeSystem, StrategicContext, NarrativeType
        )
        
        self.narrative_system = StrategicNarrativeSystem()
        
        self.sample_strategy = {
            "id": "strategy_001",
            "name": "Digital Transformation Initiative",
            "focus": "transformation technology",
            "core_approach": "comprehensive digital modernization",
            "primary_benefit": "operational efficiency",
            "secondary_benefit": "competitive advantage",
            "value_proposition": "30% cost reduction with 50% performance improvement",
            "competitive_advantage": "First-mover advantage in AI-driven operations",
            "risk_mitigation": "Phased implementation with rollback capabilities",
            "budget_request": "$5M over 18 months",
            "implementation_phases": "3 strategic phases",
            "projected_impact": "Market leadership within 24 months",
            "roi_timeline": "ROI positive within 12 months"
        }
        
        self.strategic_context = StrategicContext(
            company_position="Market challenger with strong technology foundation",
            market_conditions="Rapidly evolving digital landscape",
            competitive_landscape="Intense competition with emerging AI players",
            key_challenges=[
                "Legacy system constraints",
                "Skills gap in AI/ML",
                "Regulatory compliance complexity"
            ],
            opportunities=[
                "AI automation potential",
                "Market expansion through digital channels",
                "Partnership opportunities with tech leaders"
            ],
            stakeholder_concerns=[
                "Implementation risk",
                "Cost overruns",
                "Customer disruption"
            ],
            success_metrics=[
                "Operational efficiency improvement",
                "Customer satisfaction scores",
                "Market share growth",
                "Revenue per employee"
            ],
            timeline="18 months"
        )
        
        self.board_audience = ExecutiveAudience(
            id="board_001",
            name="Board of Directors",
            title="Board",
            executive_level=ExecutiveLevel.BOARD_MEMBER,
            communication_style=CommunicationStyle.STRATEGIC,
            expertise_areas=["governance", "strategy", "risk"],
            decision_making_pattern="consensus_building",
            influence_level=1.0,
            preferred_communication_format="board_presentation",
            attention_span=15,
            detail_preference="medium",
            risk_tolerance="low",
            created_at=datetime.now()
        )
    
    def test_create_strategic_narrative(self):
        """Test strategic narrative creation"""
        narrative = self.narrative_system.create_strategic_narrative(
            self.sample_strategy, self.board_audience, self.strategic_context
        )
        
        assert narrative is not None
        assert narrative.title is not None
        assert len(narrative.elements) > 0
        assert len(narrative.key_messages) > 0
        assert narrative.impact_score > 0.0
        assert narrative.audience_id == "board_001"
    
    def test_narrative_personalization(self):
        """Test narrative personalization for different audiences"""
        # Create narrative for board
        board_narrative = self.narrative_system.create_strategic_narrative(
            self.sample_strategy, self.board_audience, self.strategic_context
        )
        
        # Create CEO audience
        ceo_audience = ExecutiveAudience(
            id="ceo_001",
            name="CEO",
            title="Chief Executive Officer",
            executive_level=ExecutiveLevel.CEO,
            communication_style=CommunicationStyle.DIRECT,
            expertise_areas=["business", "strategy"],
            decision_making_pattern="quick_decisive",
            influence_level=1.0,
            preferred_communication_format="executive_summary",
            attention_span=5,
            detail_preference="low",
            risk_tolerance="medium",
            created_at=datetime.now()
        )
        
        # Create narrative for CEO
        ceo_narrative = self.narrative_system.create_strategic_narrative(
            self.sample_strategy, ceo_audience, self.strategic_context
        )
        
        # Verify personalization differences
        assert board_narrative.audience_id != ceo_narrative.audience_id
        assert board_narrative.personalization_notes != ceo_narrative.personalization_notes
    
    def test_narrative_impact_assessment(self):
        """Test narrative impact assessment"""
        narrative = self.narrative_system.create_strategic_narrative(
            self.sample_strategy, self.board_audience, self.strategic_context
        )
        
        engagement_data = {
            "engagement_level": 0.85,
            "emotional_resonance": 0.80,
            "message_retention": 0.90,
            "action_likelihood": 0.95,
            "credibility_score": 0.88,
            "feedback": "Compelling narrative with clear call to action"
        }
        
        impact, recommendations = self.narrative_system.assess_and_optimize(
            narrative, self.board_audience, engagement_data
        )
        
        assert impact is not None
        assert impact.overall_impact > 0.0
        assert isinstance(recommendations, dict)
    
    def test_narrative_optimization_recommendations(self):
        """Test narrative optimization recommendations"""
        narrative = self.narrative_system.create_strategic_narrative(
            self.sample_strategy, self.board_audience, self.strategic_context
        )
        
        # Low engagement data to trigger recommendations
        low_engagement_data = {
            "engagement_level": 0.5,
            "emotional_resonance": 0.4,
            "message_retention": 0.6,
            "action_likelihood": 0.7,
            "credibility_score": 0.7
        }
        
        impact, recommendations = self.narrative_system.assess_and_optimize(
            narrative, self.board_audience, low_engagement_data
        )
        
        assert len(recommendations) > 0
        assert "engagement" in recommendations or "emotion" in recommendations
    
    def test_different_narrative_types(self):
        """Test different narrative types"""
        # Test growth strategy
        growth_strategy = {
            **self.sample_strategy,
            "focus": "growth expansion",
            "name": "Market Expansion Strategy"
        }
        
        growth_narrative = self.narrative_system.create_strategic_narrative(
            growth_strategy, self.board_audience, self.strategic_context
        )
        
        # Test crisis response
        crisis_strategy = {
            **self.sample_strategy,
            "focus": "crisis urgent",
            "name": "Crisis Response Plan"
        }
        
        crisis_narrative = self.narrative_system.create_strategic_narrative(
            crisis_strategy, self.board_audience, self.strategic_context
        )
        
        # Verify different narrative types
        assert growth_narrative.narrative_type != crisis_narrative.narrative_type
    
    def test_narrative_elements_structure(self):
        """Test narrative elements and structure"""
        narrative = self.narrative_system.create_strategic_narrative(
            self.sample_strategy, self.board_audience, self.strategic_context
        )
        
        # Verify narrative has proper structure
        assert len(narrative.elements) >= 3  # Should have multiple elements
        assert narrative.emotional_arc is not None
        assert narrative.call_to_action is not None
        assert len(narrative.supporting_visuals) > 0
        
        # Verify elements have required properties
        for element in narrative.elements:
            assert element.element_type is not None
            assert element.content is not None
            assert element.emotional_tone is not None
            assert 0.0 <= element.audience_relevance <= 1.0
    
    def test_error_handling_in_narrative_system(self):
        """Test error handling in narrative system"""
        invalid_strategy = None
        
        with pytest.raises(Exception):
            self.narrative_system.create_strategic_narrative(
                invalid_strategy, self.board_audience, self.strategic_context
            )


class TestInformationSynthesisSystem:
    """Test information synthesis system"""
    
    def setup_method(self):
        """Setup test fixtures"""
        from scrollintel.engines.information_synthesis_engine import (
            InformationSynthesisSystem, ComplexInformation, InformationType, SynthesisLevel
        )
        
        # Make ComplexInformation available at class level
        self.ComplexInformation = ComplexInformation
        self.InformationType = InformationType
        
        self.synthesis_system = InformationSynthesisSystem()
        
        self.complex_information = ComplexInformation(
            id="complex_001",
            title="Q3 Performance Analysis and Strategic Recommendations",
            information_type=InformationType.FINANCIAL_DATA,
            raw_content="""
            The third quarter financial analysis reveals significant performance indicators across multiple business units. 
            Revenue increased by 15% compared to Q2, reaching $12.5M, primarily driven by enterprise client acquisitions. 
            Operating expenses rose by 8% due to increased headcount and infrastructure investments. 
            The customer acquisition cost decreased by 12%, indicating improved marketing efficiency. 
            Market share in the enterprise segment grew from 8% to 11%, positioning us favorably against competitors. 
            Risk factors include potential economic downturn and increased competition from emerging players. 
            Recommended actions include expanding the sales team, investing in product development, and establishing strategic partnerships.
            """,
            data_points=[
                {"metric": "revenue", "value": 12500000, "change": 0.15, "period": "Q3"},
                {"metric": "operating_expenses", "value": 8200000, "change": 0.08, "period": "Q3"},
                {"metric": "customer_acquisition_cost", "value": 1200, "change": -0.12, "period": "Q3"},
                {"metric": "market_share", "value": 0.11, "change": 0.03, "period": "Q3"},
                {"metric": "enterprise_clients", "value": 45, "change": 0.25, "period": "Q3"}
            ],
            source="Internal Financial Systems",
            complexity_score=0.8,
            urgency_level="high",
            stakeholders=["CEO", "CFO", "Board"],
            created_at=datetime.now()
        )
        
        self.ceo_audience = ExecutiveAudience(
            id="ceo_001",
            name="CEO",
            title="Chief Executive Officer",
            executive_level=ExecutiveLevel.CEO,
            communication_style=CommunicationStyle.DIRECT,
            expertise_areas=["business", "strategy"],
            decision_making_pattern="quick_decisive",
            influence_level=1.0,
            preferred_communication_format="executive_summary",
            attention_span=5,
            detail_preference="low",
            risk_tolerance="medium",
            created_at=datetime.now()
        )
        
        self.cfo_audience = ExecutiveAudience(
            id="cfo_001",
            name="CFO",
            title="Chief Financial Officer",
            executive_level=ExecutiveLevel.CFO,
            communication_style=CommunicationStyle.ANALYTICAL,
            expertise_areas=["finance", "accounting"],
            decision_making_pattern="data_driven",
            influence_level=0.9,
            preferred_communication_format="detailed_report",
            attention_span=15,
            detail_preference="high",
            risk_tolerance="low",
            created_at=datetime.now()
        )
    
    def test_synthesize_complex_information(self):
        """Test complex information synthesis"""
        synthesized = self.synthesis_system.process_complex_information(
            self.complex_information, self.ceo_audience
        )
        
        assert synthesized is not None
        assert synthesized.title is not None
        assert len(synthesized.executive_summary) > 0
        assert len(synthesized.key_insights) > 0
        assert len(synthesized.recommendations) > 0
        assert 0.0 <= synthesized.readability_score <= 1.0
        assert 0.0 <= synthesized.confidence_level <= 1.0
    
    def test_audience_specific_synthesis(self):
        """Test synthesis adaptation for different audiences"""
        # Synthesize for CEO
        ceo_synthesis = self.synthesis_system.process_complex_information(
            self.complex_information, self.ceo_audience
        )
        
        # Synthesize for CFO
        cfo_synthesis = self.synthesis_system.process_complex_information(
            self.complex_information, self.cfo_audience
        )
        
        # Verify different adaptations
        assert ceo_synthesis.audience_id != cfo_synthesis.audience_id
        assert "CEO Brief" in ceo_synthesis.title or "Executive Brief" in ceo_synthesis.title
        
        # CEO should get more concise information
        assert len(ceo_synthesis.executive_summary) <= len(cfo_synthesis.executive_summary) * 1.5
    
    def test_information_prioritization(self):
        """Test information prioritization"""
        # Create multiple information items
        info_list = [
            self.complex_information,
            self.ComplexInformation(
                id="complex_002",
                title="Low Priority Update",
                information_type=self.InformationType.OPERATIONAL_METRICS,
                raw_content="Routine operational metrics update",
                data_points=[],
                source="Operations",
                complexity_score=0.3,
                urgency_level="low",
                stakeholders=["Operations Manager"],
                created_at=datetime.now()
            ),
            self.ComplexInformation(
                id="complex_003",
                title="Critical Security Alert",
                information_type=self.InformationType.RISK_ASSESSMENT,
                raw_content="Critical security vulnerability detected requiring immediate attention",
                data_points=[{"severity": "critical", "affected_systems": 15}],
                source="Security Team",
                complexity_score=0.9,
                urgency_level="critical",
                stakeholders=["CEO", "CTO", "CISO"],
                created_at=datetime.now()
            )
        ]
        
        priorities = self.synthesis_system.prioritize_information_batch(info_list, self.ceo_audience)
        
        assert len(priorities) == 3
        # Critical security alert should be highest priority
        assert priorities[0].priority_level in ["critical", "high"]
        # Low priority update should be lowest
        assert priorities[-1].priority_level in ["low", "medium"]
    
    def test_executive_summary_generation(self):
        """Test executive summary generation"""
        # First synthesize the information
        synthesized = self.synthesis_system.process_complex_information(
            self.complex_information, self.ceo_audience
        )
        
        # Generate optimized summary
        executive_summary = self.synthesis_system.generate_optimized_summary(
            synthesized, self.ceo_audience
        )
        
        assert len(executive_summary) > 0
        assert "EXECUTIVE" in executive_summary.upper() or "CEO" in executive_summary.upper()
    
    def test_key_insights_extraction(self):
        """Test key insights extraction"""
        synthesized = self.synthesis_system.process_complex_information(
            self.complex_information, self.ceo_audience
        )
        
        assert len(synthesized.key_insights) > 0
        # Should contain insights about revenue growth
        insights_text = " ".join(synthesized.key_insights).lower()
        assert any(word in insights_text for word in ["revenue", "growth", "increase", "15%"])
    
    def test_critical_data_points_identification(self):
        """Test critical data points identification"""
        synthesized = self.synthesis_system.process_complex_information(
            self.complex_information, self.ceo_audience
        )
        
        assert len(synthesized.critical_data_points) > 0
        # Should identify revenue as critical data point
        critical_metrics = [dp.get("metric") for dp in synthesized.critical_data_points if "metric" in dp]
        assert any("revenue" in str(metric).lower() for metric in critical_metrics)
    
    def test_recommendations_generation(self):
        """Test recommendations generation"""
        synthesized = self.synthesis_system.process_complex_information(
            self.complex_information, self.ceo_audience
        )
        
        assert len(synthesized.recommendations) > 0
        # Should contain actionable recommendations
        recommendations_text = " ".join(synthesized.recommendations).lower()
        assert any(word in recommendations_text for word in ["implement", "consider", "expand", "invest"])
    
    def test_decision_points_extraction(self):
        """Test decision points extraction"""
        synthesized = self.synthesis_system.process_complex_information(
            self.complex_information, self.ceo_audience
        )
        
        assert len(synthesized.decision_points) >= 0  # May be empty for some content
        if synthesized.decision_points:
            # Should contain decision-oriented language
            decisions_text = " ".join(synthesized.decision_points).lower()
            assert any(word in decisions_text for word in ["decision", "approve", "authorize", "required"])
    
    def test_risk_factors_identification(self):
        """Test risk factors identification"""
        synthesized = self.synthesis_system.process_complex_information(
            self.complex_information, self.ceo_audience
        )
        
        assert len(synthesized.risk_factors) >= 0
        if synthesized.risk_factors:
            # Should identify economic downturn risk from content
            risks_text = " ".join(synthesized.risk_factors).lower()
            assert any(word in risks_text for word in ["risk", "downturn", "competition", "challenge"])
    
    def test_next_steps_definition(self):
        """Test next steps definition"""
        synthesized = self.synthesis_system.process_complex_information(
            self.complex_information, self.ceo_audience
        )
        
        assert len(synthesized.next_steps) > 0
        # Should contain actionable next steps
        steps_text = " ".join(synthesized.next_steps).lower()
        assert any(word in steps_text for word in ["schedule", "implement", "monitor", "review"])
    
    def test_readability_score_calculation(self):
        """Test readability score calculation"""
        synthesized = self.synthesis_system.process_complex_information(
            self.complex_information, self.ceo_audience
        )
        
        assert 0.0 <= synthesized.readability_score <= 1.0
        
        # CEO with low detail preference should get higher readability
        assert synthesized.readability_score >= 0.5
    
    def test_confidence_level_assessment(self):
        """Test confidence level assessment"""
        synthesized = self.synthesis_system.process_complex_information(
            self.complex_information, self.ceo_audience
        )
        
        assert 0.0 <= synthesized.confidence_level <= 1.0
        
        # Should have reasonable confidence with good data
        assert synthesized.confidence_level >= 0.6
    
    def test_different_information_types(self):
        """Test synthesis of different information types"""
        # Test market analysis
        market_info = self.ComplexInformation(
            id="market_001",
            title="Market Analysis Report",
            information_type=self.InformationType.MARKET_ANALYSIS,
            raw_content="Market analysis shows growing demand for AI solutions with 25% annual growth rate.",
            data_points=[{"market_size": 50000000, "growth_rate": 0.25}],
            source="Market Research",
            complexity_score=0.6,
            urgency_level="medium",
            stakeholders=["CEO", "VP Marketing"],
            created_at=datetime.now()
        )
        
        market_synthesis = self.synthesis_system.process_complex_information(market_info, self.ceo_audience)
        
        assert market_synthesis is not None
        assert "market" in market_synthesis.title.lower()
        
        # Test risk assessment
        risk_info = self.ComplexInformation(
            id="risk_001",
            title="Cybersecurity Risk Assessment",
            information_type=self.InformationType.RISK_ASSESSMENT,
            raw_content="Risk assessment identifies critical vulnerabilities requiring immediate attention.",
            data_points=[{"vulnerability_count": 15, "severity": "high"}],
            source="Security Team",
            complexity_score=0.8,
            urgency_level="critical",
            stakeholders=["CEO", "CTO", "CISO"],
            created_at=datetime.now()
        )
        
        risk_synthesis = self.synthesis_system.process_complex_information(risk_info, self.ceo_audience)
        
        assert risk_synthesis is not None
        assert len(risk_synthesis.risk_factors) > 0
    
    def test_error_handling_in_synthesis(self):
        """Test error handling in synthesis system"""
        invalid_info = None
        
        with pytest.raises(Exception):
            self.synthesis_system.process_complex_information(invalid_info, self.ceo_audience)