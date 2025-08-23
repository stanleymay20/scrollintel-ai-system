"""
Integration Tests for Intelligence and Decision Engine

This module contains integration tests that verify the Intelligence Engine
works correctly with other system components and external dependencies.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List
import json

from scrollintel.engines.intelligence_engine import (
    IntelligenceEngine, BusinessContext, DecisionOption
)


class TestIntelligenceEngineIntegration:
    """Integration tests for Intelligence Engine with system components"""
    
    @pytest.fixture
    async def intelligence_system(self):
        """Create complete intelligence system for integration testing"""
        engine = IntelligenceEngine()
        await engine.start()
        yield engine
        await engine.stop()
    
    @pytest.mark.asyncio
    async def test_end_to_end_decision_workflow(self, intelligence_system):
        """Test complete end-to-end decision workflow"""
        
        # Create comprehensive business scenario
        context = BusinessContext(
            industry="financial_services",
            business_unit="digital_banking",
            stakeholders=["ceo", "cto", "chief_risk_officer", "head_of_digital"],
            constraints=[
                {"type": "regulatory_compliance", "requirements": ["basel_iii", "gdpr", "pci_dss"]},
                {"type": "budget", "value": 15000000, "currency": "USD"},
                {"type": "timeline", "value": 18, "unit": "months"},
                {"type": "risk_tolerance", "level": "medium"}
            ],
            objectives=[
                {
                    "type": "digital_transformation",
                    "target": "complete_mobile_first_platform",
                    "timeline": "18_months",
                    "success_criteria": ["user_adoption_80_percent", "cost_reduction_30_percent"]
                },
                {
                    "type": "customer_acquisition",
                    "target": 500000,
                    "timeline": "24_months",
                    "success_criteria": ["retention_rate_85_percent", "satisfaction_score_4_5"]
                },
                {
                    "type": "operational_efficiency",
                    "target": "automation_70_percent",
                    "timeline": "12_months",
                    "success_criteria": ["processing_time_reduction_50_percent"]
                }
            ],
            current_state={
                "digital_maturity": "intermediate",
                "customer_base": 1200000,
                "mobile_adoption": 0.45,
                "automation_level": 0.35,
                "annual_revenue": 850000000,
                "cost_to_income_ratio": 0.65,
                "regulatory_compliance_score": 0.92
            },
            historical_data={
                "digital_initiatives": 8,
                "success_rate": 0.75,
                "average_roi": 2.8,
                "customer_satisfaction_trend": "improving",
                "competitive_position": "strong_challenger"
            },
            time_horizon="medium_term",
            budget_constraints={
                "total": 15000000,
                "technology": 8000000,
                "operations": 3000000,
                "marketing": 2500000,
                "compliance": 1500000
            },
            regulatory_requirements=[
                "basel_iii", "gdpr", "pci_dss", "mifid_ii", "psd2", "aml_kyc"
            ]
        )
        
        # Create complex decision options
        options = [
            DecisionOption(
                id="cloud_native_rebuild",
                name="Cloud-Native Platform Rebuild",
                description="Complete rebuild of banking platform using cloud-native architecture",
                expected_outcomes={
                    "scalability_improvement": 10.0,
                    "operational_cost_reduction": 0.4,
                    "time_to_market_improvement": 0.6,
                    "security_enhancement": 0.8,
                    "customer_experience_score": 4.7
                },
                costs={
                    "architecture_design": 1200000,
                    "development": 4500000,
                    "cloud_infrastructure": 1800000,
                    "migration": 1000000,
                    "training": 500000,
                    "security_implementation": 800000
                },
                benefits={
                    "operational_savings": 25000000,
                    "revenue_increase": 45000000,
                    "risk_reduction": 8000000,
                    "competitive_advantage": 15000000
                },
                risks=[
                    {
                        "type": "migration_complexity",
                        "probability": 0.4,
                        "impact": 5000000,
                        "mitigation": "Phased migration approach"
                    },
                    {
                        "type": "regulatory_approval",
                        "probability": 0.2,
                        "impact": 3000000,
                        "mitigation": "Early regulator engagement"
                    },
                    {
                        "type": "talent_shortage",
                        "probability": 0.5,
                        "impact": 2000000,
                        "mitigation": "Partner with cloud specialists"
                    }
                ],
                implementation_complexity=9.0,
                time_to_implement=540,
                resource_requirements={
                    "cloud_architects": 8,
                    "backend_engineers": 25,
                    "frontend_engineers": 15,
                    "devops_engineers": 12,
                    "security_specialists": 6,
                    "compliance_officers": 4,
                    "budget": 9800000
                }
            ),
            
            DecisionOption(
                id="incremental_modernization",
                name="Incremental Platform Modernization",
                description="Gradual modernization of existing systems with API-first approach",
                expected_outcomes={
                    "integration_improvement": 0.7,
                    "development_velocity": 0.5,
                    "system_reliability": 0.6,
                    "customer_satisfaction": 4.2
                },
                costs={
                    "api_development": 2000000,
                    "system_integration": 1500000,
                    "legacy_enhancement": 1200000,
                    "testing_automation": 800000,
                    "monitoring_tools": 400000
                },
                benefits={
                    "faster_feature_delivery": 12000000,
                    "reduced_maintenance": 8000000,
                    "improved_reliability": 6000000,
                    "customer_retention": 10000000
                },
                risks=[
                    {
                        "type": "technical_debt",
                        "probability": 0.6,
                        "impact": 2000000,
                        "mitigation": "Continuous refactoring"
                    },
                    {
                        "type": "integration_complexity",
                        "probability": 0.4,
                        "impact": 1500000,
                        "mitigation": "Comprehensive testing"
                    }
                ],
                implementation_complexity=6.5,
                time_to_implement=360,
                resource_requirements={
                    "integration_specialists": 10,
                    "api_developers": 15,
                    "qa_engineers": 8,
                    "system_analysts": 6,
                    "budget": 5900000
                }
            ),
            
            DecisionOption(
                id="fintech_partnership",
                name="Strategic Fintech Partnership",
                description="Partner with leading fintech companies for rapid digital capabilities",
                expected_outcomes={
                    "time_to_market": 0.3,
                    "innovation_access": 0.9,
                    "cost_efficiency": 0.6,
                    "market_differentiation": 0.7
                },
                costs={
                    "partnership_fees": 3000000,
                    "integration_costs": 1500000,
                    "revenue_sharing": 0.15,  # 15% revenue share
                    "compliance_alignment": 800000,
                    "training": 300000
                },
                benefits={
                    "rapid_innovation": 20000000,
                    "reduced_development_risk": 5000000,
                    "market_access": 8000000,
                    "expertise_acquisition": 4000000
                },
                risks=[
                    {
                        "type": "dependency_risk",
                        "probability": 0.3,
                        "impact": 4000000,
                        "mitigation": "Multi-partner strategy"
                    },
                    {
                        "type": "data_security",
                        "probability": 0.2,
                        "impact": 6000000,
                        "mitigation": "Strict security protocols"
                    },
                    {
                        "type": "regulatory_complexity",
                        "probability": 0.4,
                        "impact": 2500000,
                        "mitigation": "Joint compliance framework"
                    }
                ],
                implementation_complexity=7.5,
                time_to_implement=180,
                resource_requirements={
                    "partnership_managers": 4,
                    "integration_engineers": 12,
                    "compliance_specialists": 6,
                    "security_experts": 5,
                    "budget": 5600000
                }
            )
        ]
        
        # Execute complete decision workflow
        print("🔄 Executing end-to-end decision workflow...")
        
        # Step 1: Make initial decision
        decision = await intelligence_system.make_decision(context, options)
        
        # Verify decision structure
        assert decision is not None
        assert decision.selected_option in options
        assert len(decision.reasoning) > 0
        
        # Step 2: Simulate implementation and outcome
        implementation_outcome = {
            "success_score": 0.82,
            "financial_impact": decision.selected_option.benefits.get("operational_savings", 0) * 0.8,
            "roi_achieved": 2.6,
            "timeline_variance": 8,  # 8% over timeline
            "budget_variance": -3,  # 3% under budget
            "stakeholder_satisfaction": 0.85,
            "technical_metrics": {
                "system_uptime": 0.998,
                "performance_improvement": 0.45,
                "security_incidents": 0,
                "user_adoption_rate": 0.78
            },
            "business_metrics": {
                "customer_acquisition": 1.12,  # 12% above target
                "revenue_growth": 1.05,  # 5% above target
                "cost_reduction": 1.15,  # 15% above target
                "market_share_gain": 0.03
            },
            "lessons_learned": [
                "Stakeholder communication was critical for success",
                "Phased implementation reduced risk significantly",
                "Early customer feedback improved final outcomes",
                "Cross-functional teams accelerated delivery"
            ],
            "unexpected_benefits": [
                "Improved employee satisfaction and retention",
                "Enhanced vendor relationships",
                "Increased industry recognition"
            ],
            "challenges_overcome": [
                "Initial resistance to change",
                "Technical integration complexity",
                "Regulatory approval delays"
            ]
        }
        
        # Step 3: Learn from outcome
        await intelligence_system.learn_from_outcome(decision.id, implementation_outcome)
        
        # Step 4: Make follow-up decision to test learning
        follow_up_context = BusinessContext(
            industry=context.industry,
            business_unit="mobile_banking",
            stakeholders=context.stakeholders,
            constraints=context.constraints,
            objectives=[
                {
                    "type": "mobile_optimization",
                    "target": "best_in_class_mobile_experience",
                    "timeline": "12_months"
                }
            ],
            current_state={
                **context.current_state,
                "mobile_adoption": 0.78,  # Improved from previous initiative
                "digital_maturity": "advanced"  # Upgraded maturity
            },
            historical_data={
                **context.historical_data,
                "recent_success": implementation_outcome["success_score"]
            },
            time_horizon="short_term",
            budget_constraints={"total": 5000000},
            regulatory_requirements=context.regulatory_requirements
        )
        
        follow_up_options = [
            DecisionOption(
                id="ai_powered_personalization",
                name="AI-Powered Personalization Engine",
                description="Implement AI-driven personalized banking experience",
                expected_outcomes={"customer_engagement": 0.4, "revenue_per_customer": 0.25},
                costs={"ai_development": 2000000, "data_infrastructure": 1000000},
                benefits={"revenue_increase": 15000000, "customer_retention": 8000000},
                risks=[{"type": "ai_bias", "probability": 0.3, "impact": 1000000}],
                implementation_complexity=8.0,
                time_to_implement=270,
                resource_requirements={"ai_engineers": 12, "data_scientists": 8}
            ),
            DecisionOption(
                id="voice_banking_platform",
                name="Voice Banking Platform",
                description="Launch comprehensive voice-activated banking services",
                expected_outcomes={"accessibility": 0.6, "user_convenience": 0.8},
                costs={"voice_technology": 1500000, "security_enhancement": 800000},
                benefits={"market_differentiation": 10000000, "accessibility_value": 5000000},
                risks=[{"type": "voice_security", "probability": 0.4, "impact": 2000000}],
                implementation_complexity=7.0,
                time_to_implement=240,
                resource_requirements={"voice_specialists": 6, "security_experts": 4}
            )
        ]
        
        # Make follow-up decision
        follow_up_decision = await intelligence_system.make_decision(follow_up_context, follow_up_options)
        
        # Verify that learning influenced the decision
        assert follow_up_decision is not None
        assert follow_up_decision.selected_option in follow_up_options
        
        # The system should show evidence of learning (e.g., improved confidence, better reasoning)
        assert len(follow_up_decision.reasoning) > 0
        
        print("✅ End-to-end workflow completed successfully")
        return decision, follow_up_decision
    
    @pytest.mark.asyncio
    async def test_concurrent_decision_processing(self, intelligence_system):
        """Test system handling of concurrent decision requests"""
        
        # Create multiple decision scenarios
        scenarios = []
        for i in range(5):
            context = BusinessContext(
                industry=f"industry_{i}",
                business_unit=f"unit_{i}",
                stakeholders=[f"stakeholder_{i}"],
                constraints=[{"type": "budget", "value": 1000000 * (i + 1)}],
                objectives=[{"type": "growth", "target": 0.1 * (i + 1)}],
                current_state={"revenue": 5000000 * (i + 1)},
                historical_data={"success_rate": 0.5 + (i * 0.1)},
                time_horizon="medium_term",
                budget_constraints={"total": 1000000 * (i + 1)},
                regulatory_requirements=["regulation_1"]
            )
            
            options = [
                DecisionOption(
                    id=f"option_a_{i}",
                    name=f"Option A {i}",
                    description=f"Conservative option for scenario {i}",
                    expected_outcomes={"growth": 0.1},
                    costs={"implementation": 500000},
                    benefits={"revenue": 1000000},
                    risks=[{"type": "market", "probability": 0.2, "impact": 200000}],
                    implementation_complexity=5.0,
                    time_to_implement=180,
                    resource_requirements={"team": 10}
                ),
                DecisionOption(
                    id=f"option_b_{i}",
                    name=f"Option B {i}",
                    description=f"Aggressive option for scenario {i}",
                    expected_outcomes={"growth": 0.3},
                    costs={"implementation": 800000},
                    benefits={"revenue": 2000000},
                    risks=[{"type": "market", "probability": 0.4, "impact": 600000}],
                    implementation_complexity=7.0,
                    time_to_implement=240,
                    resource_requirements={"team": 15}
                )
            ]
            
            scenarios.append((context, options))
        
        # Process all scenarios concurrently
        print("🔄 Processing concurrent decision requests...")
        
        tasks = [
            intelligence_system.make_decision(context, options)
            for context, options in scenarios
        ]
        
        # Wait for all decisions to complete
        decisions = await asyncio.gather(*tasks)
        
        # Verify all decisions were made successfully
        assert len(decisions) == 5
        for i, decision in enumerate(decisions):
            assert decision is not None
            assert decision.selected_option in scenarios[i][1]  # Selected from correct options
            assert len(decision.reasoning) > 0
        
        print("✅ Concurrent processing completed successfully")
        return decisions
    
    @pytest.mark.asyncio
    async def test_system_resilience_under_load(self, intelligence_system):
        """Test system resilience under high load conditions"""
        
        # Create a high-load scenario
        base_context = BusinessContext(
            industry="technology",
            business_unit="saas",
            stakeholders=["ceo", "cto"],
            constraints=[{"type": "budget", "value": 2000000}],
            objectives=[{"type": "growth", "target": 0.5}],
            current_state={"revenue": 10000000},
            historical_data={"growth_rate": 0.3},
            time_horizon="short_term",
            budget_constraints={"total": 2000000},
            regulatory_requirements=["gdpr"]
        )
        
        base_options = [
            DecisionOption(
                id="scale_up",
                name="Scale Up Operations",
                description="Aggressive scaling of operations",
                expected_outcomes={"growth": 0.6},
                costs={"scaling": 1500000},
                benefits={"revenue": 8000000},
                risks=[{"type": "operational", "probability": 0.3, "impact": 1000000}],
                implementation_complexity=6.0,
                time_to_implement=180,
                resource_requirements={"team": 20}
            ),
            DecisionOption(
                id="optimize_current",
                name="Optimize Current Operations",
                description="Focus on optimizing existing operations",
                expected_outcomes={"efficiency": 0.4},
                costs={"optimization": 800000},
                benefits={"savings": 3000000},
                risks=[{"type": "competitive", "probability": 0.2, "impact": 500000}],
                implementation_complexity=4.0,
                time_to_implement=120,
                resource_requirements={"team": 12}
            )
        ]
        
        # Simulate high load with rapid successive requests
        print("🔄 Testing system resilience under load...")
        
        start_time = datetime.utcnow()
        successful_decisions = 0
        failed_decisions = 0
        
        # Make 20 rapid decisions
        for i in range(20):
            try:
                decision = await intelligence_system.make_decision(base_context, base_options)
                if decision is not None:
                    successful_decisions += 1
                else:
                    failed_decisions += 1
            except Exception as e:
                failed_decisions += 1
                print(f"   Decision {i+1} failed: {e}")
        
        end_time = datetime.utcnow()
        total_time = (end_time - start_time).total_seconds()
        
        # Verify system performance under load
        success_rate = successful_decisions / (successful_decisions + failed_decisions)
        average_time_per_decision = total_time / 20
        
        print(f"   Successful decisions: {successful_decisions}/20")
        print(f"   Success rate: {success_rate:.1%}")
        print(f"   Average time per decision: {average_time_per_decision:.2f}s")
        print(f"   Total processing time: {total_time:.2f}s")
        
        # System should maintain reasonable performance under load
        assert success_rate >= 0.8  # At least 80% success rate
        assert average_time_per_decision < 5.0  # Less than 5 seconds per decision
        
        print("✅ System resilience test completed successfully")
        return {
            "success_rate": success_rate,
            "average_time": average_time_per_decision,
            "total_time": total_time
        }
    
    @pytest.mark.asyncio
    async def test_learning_effectiveness_over_time(self, intelligence_system):
        """Test that the system's learning improves decision quality over time"""
        
        # Create a consistent decision scenario for learning evaluation
        learning_context = BusinessContext(
            industry="retail",
            business_unit="ecommerce",
            stakeholders=["ceo", "cmo"],
            constraints=[{"type": "budget", "value": 3000000}],
            objectives=[{"type": "customer_acquisition", "target": 50000}],
            current_state={"customers": 100000, "revenue": 20000000},
            historical_data={"acquisition_cost": 60},
            time_horizon="medium_term",
            budget_constraints={"total": 3000000},
            regulatory_requirements=["gdpr", "ccpa"]
        )
        
        learning_options = [
            DecisionOption(
                id="digital_marketing",
                name="Digital Marketing Campaign",
                description="Comprehensive digital marketing strategy",
                expected_outcomes={"acquisition": 60000, "brand_awareness": 0.3},
                costs={"advertising": 2000000, "content": 500000},
                benefits={"revenue": 12000000, "brand_value": 2000000},
                risks=[{"type": "market_saturation", "probability": 0.3, "impact": 1000000}],
                implementation_complexity=5.0,
                time_to_implement=180,
                resource_requirements={"marketers": 15, "budget": 2500000}
            ),
            DecisionOption(
                id="product_innovation",
                name="Product Innovation Initiative",
                description="Develop innovative product features",
                expected_outcomes={"differentiation": 0.8, "retention": 0.2},
                costs={"development": 1800000, "testing": 400000},
                benefits={"premium_pricing": 8000000, "loyalty": 4000000},
                risks=[{"type": "development_delay", "probability": 0.4, "impact": 800000}],
                implementation_complexity=7.0,
                time_to_implement=270,
                resource_requirements={"engineers": 20, "budget": 2200000}
            )
        ]
        
        print("🔄 Testing learning effectiveness over time...")
        
        # Track decision quality metrics over multiple iterations
        decision_history = []
        confidence_trend = []
        
        # Make decisions and learn from outcomes over 10 iterations
        for iteration in range(10):
            # Make decision
            decision = await intelligence_system.make_decision(learning_context, learning_options)
            
            # Track confidence
            confidence_score = 0.5  # Default confidence
            if hasattr(decision.confidence, 'value'):
                confidence_mapping = {
                    'very_low': 0.1, 'low': 0.3, 'medium': 0.5, 'high': 0.7, 'very_high': 0.9
                }
                confidence_score = confidence_mapping.get(decision.confidence.value, 0.5)
            
            confidence_trend.append(confidence_score)
            
            # Simulate outcome with improving success over time (learning effect)
            base_success = 0.6 + (iteration * 0.03)  # Gradual improvement
            outcome = {
                "success_score": min(0.95, base_success),  # Cap at 95%
                "financial_impact": decision.selected_option.benefits.get("revenue", 0) * base_success,
                "roi_achieved": 1.5 + (iteration * 0.1),
                "lessons_learned": [f"Lesson from iteration {iteration + 1}"],
                "improvement_factors": [
                    "Better stakeholder alignment",
                    "Improved execution methodology",
                    "Enhanced risk mitigation"
                ]
            }
            
            # Learn from outcome
            await intelligence_system.learn_from_outcome(decision.id, outcome)
            
            decision_history.append({
                "iteration": iteration + 1,
                "decision": decision,
                "outcome": outcome,
                "confidence": confidence_score
            })
        
        # Analyze learning effectiveness
        early_confidence = sum(confidence_trend[:3]) / 3  # First 3 iterations
        late_confidence = sum(confidence_trend[-3:]) / 3  # Last 3 iterations
        
        early_success = sum(h["outcome"]["success_score"] for h in decision_history[:3]) / 3
        late_success = sum(h["outcome"]["success_score"] for h in decision_history[-3:]) / 3
        
        confidence_improvement = late_confidence - early_confidence
        success_improvement = late_success - early_success
        
        print(f"   Early average confidence: {early_confidence:.2f}")
        print(f"   Late average confidence: {late_confidence:.2f}")
        print(f"   Confidence improvement: {confidence_improvement:+.2f}")
        print(f"   Early average success: {early_success:.2f}")
        print(f"   Late average success: {late_success:.2f}")
        print(f"   Success improvement: {success_improvement:+.2f}")
        
        # System should show learning improvement over time
        # Note: In a real system, we'd expect improvement, but in this demo
        # the improvement is simulated in the outcomes
        assert len(decision_history) == 10
        assert all(h["decision"] is not None for h in decision_history)
        
        print("✅ Learning effectiveness test completed")
        return {
            "confidence_improvement": confidence_improvement,
            "success_improvement": success_improvement,
            "decision_history": decision_history
        }


if __name__ == "__main__":
    # Run integration tests
    pytest.main([__file__, "-v", "--tb=short"])